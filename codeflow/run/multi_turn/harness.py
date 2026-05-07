import json
import subprocess
import os
import sys
import argparse
import re
import ast
import shutil
import time
import textwrap

# Add parent directory to path to import src.utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import get_uuid, has_print

def parse_args():
    parser = argparse.ArgumentParser(description="Run harness evaluation on multi-turn problems (Legacy Mode)")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output results')
    parser.add_argument('--temp_code', type=str, default='temp/temp_code.py', help='Path to temp code file')
    parser.add_argument('--assert_code', type=str, default='temp/assert_code.py', help='Path to assert code file')
    parser.add_argument('--main_code', type=str, default='temp/main_code.py', help='Path to main code file')
    parser.add_argument(
        '--isolate_subtasks',
        action='store_true',
        help='Enable golden isolation across subtasks: do not pass generated code context between subtasks',
    )
    parser.add_argument(
        '--require_successful_prereq',
        action='store_true',
        help='In chained mode, only pass context when previous subtask is fully successful; otherwise block dependent subtasks',
    )
    parser.add_argument(
        '--golden_upstream_for_eval',
        action='store_true',
        help='Evaluate each subtask with golden upstream dependency context (node-only success rate)',
    )
    parser.add_argument(
        '--disable_runtime_golden_label',
        action='store_true',
        help='Disable automatic golden-label annotation in rollout output JSON',
    )
    return parser.parse_args()


def _extract_solution_code(problem):
    """Get a best-effort code solution block from problem-level solutions."""
    solutions = problem.get("solutions", [])
    for item in solutions:
        if isinstance(item, dict) and item.get("type") == "code":
            content = item.get("content", "")
            code = extract_code(content)
            if code.strip():
                return code
    return ""


def _extract_function_block_from_code(code_text, function_name):
    """Extract a Python function block by name from raw code text."""
    if not code_text or not function_name:
        return ""
    lines = code_text.splitlines()
    pat = re.compile(r"^\s*def\s+" + re.escape(function_name) + r"\s*\(")

    start = -1
    for i, line in enumerate(lines):
        if pat.match(line):
            start = i
            break
    if start < 0:
        return ""

    base_indent = len(lines[start]) - len(lines[start].lstrip(" "))
    end = len(lines)
    for j in range(start + 1, len(lines)):
        cur = lines[j]
        if not cur.strip():
            continue
        cur_indent = len(cur) - len(cur.lstrip(" "))
        if cur_indent <= base_indent and re.match(r"^\s*(def|class)\s+", cur):
            end = j
            break

    block = "\n".join(lines[start:end]).rstrip()
    return textwrap.dedent(block)


def _pick_subproblem_golden_code(subproblem, fallback_problem_code):
    """Pick golden code for a subproblem from explicit fields or fallback extraction."""
    for key in ["solution", "golden_solution", "reference_solution", "canonical_solution"]:
        val = subproblem.get(key)
        if isinstance(val, str) and val.strip():
            code = extract_code(val)
            if code.strip():
                return code

    name = subproblem.get("name", "")
    return _extract_function_block_from_code(fallback_problem_code, name)


def _collect_dep_closure(sub_name, deps_by_name):
    """Return transitive dependency order using DFS post-order."""
    visited = set()
    ordered = []

    def dfs(name):
        if name in visited:
            return
        visited.add(name)
        for p in deps_by_name.get(name, []):
            dfs(p)
        ordered.append(name)

    for d in deps_by_name.get(sub_name, []):
        dfs(d)
    return ordered


def _build_golden_context_for_subtask(subproblem_name, deps_by_name, golden_by_name):
    """Build context from golden code of all transitive dependencies only."""
    ordered = _collect_dep_closure(subproblem_name, deps_by_name)
    chunks = []
    for name in ordered:
        code = golden_by_name.get(name, "").strip()
        if code:
            chunks.append(code)
    return "\n\n".join(chunks).rstrip()


def _annotate_runtime_golden_labels(problem, subproblems, fallback_problem_code):
    """Annotate golden-label retrievability directly in rollout output JSON."""
    total = len(subproblems)
    retrievable = 0
    missing = []

    for subproblem in subproblems:
        code = _pick_subproblem_golden_code(subproblem, fallback_problem_code)
        ok = bool(code.strip())
        subproblem["golden_label_retrievable_runtime"] = ok
        if ok:
            retrievable += 1
        else:
            missing.append(subproblem.get("name", ""))

    ratio = (retrievable / float(total)) if total > 0 else 0.0
    problem["golden_label_total_subproblems"] = total
    problem["golden_label_retrievable_subproblems"] = retrievable
    problem["golden_label_retrievable_ratio"] = ratio
    problem["golden_label_missing_subproblems"] = missing
    problem["golden_label_filter_recommended"] = (retrievable == 0)
    problem["golden_label_partial_available"] = (0 < retrievable < total)

def extract_code(text):
    """
    Extracts Python code from Markdown code blocks.
    If no blocks are found, returns the original text stripped of whitespace.
    """
    if not text:
        return ""
    # Pattern to match ```python ... ``` or ``` ... ```
    pattern = r'```(?:python)?\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        valid_blocks = [m.strip() for m in matches if m.strip()]
        return "\n\n".join(valid_blocks)
    return text.strip()

def check_syntax(code_str):
    """
    Checks if the code is valid Python syntax using AST.
    """
    if not code_str:
        return False
    try:
        ast.parse(code_str)
        return True
    except (SyntaxError, ValueError):
        return False


def _evaluate_single_rollout(subproblem, code, turn_num, overall_turns, args, temp_dir, base_context):
    result_list = []
    eval_start = time.perf_counter()

    if not check_syntax(code):
        test_len = len(subproblem.get("test_code", []))
        fallback_len = test_len if test_len > 0 else 1
        return [0] * fallback_len, time.perf_counter() - eval_start

    if not subproblem.get("test_code"):
        return [], time.perf_counter() - eval_start

    # Last turn: execute as script-style main.
    if turn_num == overall_turns:
        function_name = subproblem["name"]

        input_raw = subproblem["test_code"][0]["input"]
        if isinstance(input_raw, str):
            input_ = input_raw.strip("[]'").replace('\\n', '\n')
        else:
            input_ = str(input_raw)
        input_ = ' '.join(input_.split())

        output_raw = subproblem["test_code"][0]["output"]
        output = output_raw.replace('\\n', '\n') if isinstance(output_raw, str) else str(output_raw)

        try:
            with open(args.main_code, 'w', encoding='utf-8') as main:
                main.write(base_context.rstrip())
                if base_context.strip():
                    main.write("\n")
                main.write("\nimport sys\n")
                main.write(code + "\n")
                if has_print(code):
                    main.write(f"{function_name}()")
                else:
                    main.write(f"print({function_name}())")
        except Exception:
            return [0], time.perf_counter() - eval_start

        try:
            result = subprocess.run(
                [sys.executable, args.main_code],
                capture_output=True,
                text=True,
                input=input_,
                timeout=5
            )
            stdout_res = result.stdout.strip()
            if stdout_res == output.strip() or stdout_res.strip("'") == output.strip("'"):
                result_list.append(1)
            else:
                result_list.append(0)
        except subprocess.TimeoutExpired:
            result_list.append(0)
        except Exception:
            result_list.append(0)
        finally:
            if os.path.exists(args.main_code):
                os.remove(args.main_code)

        return result_list, time.perf_counter() - eval_start

    # Intermediate turn: append to context module and run per-test assertion.
    function_name = subproblem["name"]
    input_list = []
    output_list = []

    for i in subproblem["test_code"]:
        inp = i["input"]
        if isinstance(inp, str):
            inp = inp.replace(",)", ")")
        input_list.append(inp)
        output_list.append(i["output"])

    try:
        with open(args.temp_code, 'w', encoding='utf-8') as file:
            file.write(base_context.rstrip())
            if base_context.strip():
                file.write("\n")
            file.write("\n" + code)
    except Exception:
        test_len = len(input_list)
        fallback_len = test_len if test_len > 0 else 1
        return [0] * fallback_len, time.perf_counter() - eval_start

    for inp, outp in zip(input_list, output_list):
        with open(args.assert_code, 'w', encoding='utf-8') as file:
            file.write("import sys\n")
            file.write(f"sys.path.append('{os.path.abspath(temp_dir)}')\n")
            module_name = os.path.splitext(os.path.basename(args.temp_code))[0]
            file.write(f"from {module_name} import *\n")
            file.write(f"print({function_name}{inp})")

        try:
            result = subprocess.run(
                [sys.executable, args.assert_code],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                result_list.append(0)
            else:
                actual = result.stdout.strip()
                expected = str(outp).strip()
                if actual == expected or actual.strip("'") == expected.strip("'"):
                    result_list.append(1)
                else:
                    result_list.append(0)
        except subprocess.TimeoutExpired:
            result_list.append(0)
        except Exception:
            result_list.append(0)
        finally:
            if os.path.exists(args.assert_code):
                os.remove(args.assert_code)

    return result_list, time.perf_counter() - eval_start

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.dirname(args.temp_code)
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)

    try:
        json_data = json.load(open(args.input_path, encoding='utf-8'))
    except Exception as e:
        print(f"Error loading JSON input: {e}")
        return

    print(f"Start harness evaluation: {len(json_data)} problems")
    uuid_set = get_uuid(args.output_dir)

    for problem in json_data:
        uuid = problem.get("problem-id") or problem.get("task_id")
        
        # Skip if already processed
        if uuid in uuid_set:
            continue

        # Clean up temp files before processing a new problem
        for file_path in [args.temp_code, args.assert_code, args.main_code]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

        subproblems = problem.get("subproblems", [])
        turn_num = 0
        overall_turns = problem.get("overall-turns", 0)
        prereq_context_ready = True
        fallback_problem_code = _extract_solution_code(problem)
        deps_by_name = {sp.get("name", ""): sp.get("dependencies", []) for sp in subproblems}
        golden_by_name = {
            sp.get("name", ""): _pick_subproblem_golden_code(sp, fallback_problem_code)
            for sp in subproblems
        }
        if not args.disable_runtime_golden_label:
            _annotate_runtime_golden_labels(problem, subproblems, fallback_problem_code)

        for subproblem in subproblems:
            turn_num += 1

            # Golden isolation mode: clear temp context before each subtask.
            if args.isolate_subtasks and os.path.exists(args.temp_code):
                try:
                    os.remove(args.temp_code)
                except OSError:
                    pass

            rollout_candidates = subproblem.get("rollout_candidates") or []
            if not rollout_candidates and subproblem.get("generated"):
                rollout_candidates = [{
                    "rollout_id": 0,
                    "generated": subproblem.get("generated", ""),
                    "original_output": subproblem.get("original_output", ""),
                    "infer_latency_sec": 0.0,
                }]

            if not rollout_candidates:
                subproblem['harness_result'] = [0]
                subproblem['rollout_records'] = []
                subproblem['selected_rollout_id'] = -1
                if args.require_successful_prereq and (not args.isolate_subtasks):
                    prereq_context_ready = False
                continue

            # Strict prerequisite mode: if previous required subtask failed,
            # do not evaluate current dependent subtask rollouts.
            if (
                args.require_successful_prereq
                and (not args.golden_upstream_for_eval)
                and (not args.isolate_subtasks)
                and turn_num > 1
                and (not prereq_context_ready)
            ):
                test_len = len(subproblem.get("test_code", []))
                fallback_len = test_len if test_len > 0 else 1
                blocked_result = [0] * fallback_len
                subproblem['harness_result'] = blocked_result
                subproblem['rollout_records'] = [{
                    "rollout_id": c.get("rollout_id", idx),
                    "infer_latency_sec": float(c.get("infer_latency_sec", 0.0) or 0.0),
                    "infer_batch_latency_sec": c.get("infer_batch_latency_sec"),
                    "infer_per_rollout_latency_sec": c.get("infer_per_rollout_latency_sec"),
                    "latency_observation": c.get("latency_observation"),
                    "eval_latency_sec": 0.0,
                    "total_latency_sec": float(c.get("infer_latency_sec", 0.0) or 0.0),
                    "harness_result": blocked_result,
                    "pass_rate": 0.0,
                    "success": 0,
                    "blocked_by_prereq_failure": 1,
                } for idx, c in enumerate(rollout_candidates)]
                subproblem['selected_rollout_id'] = -1
                continue

            base_context = ""
            if args.golden_upstream_for_eval:
                sub_name = subproblem.get("name", "")
                base_context = _build_golden_context_for_subtask(sub_name, deps_by_name, golden_by_name)
            elif (not args.isolate_subtasks) and os.path.exists(args.temp_code):
                try:
                    with open(args.temp_code, 'r', encoding='utf-8') as temp:
                        base_context = temp.read().rstrip()
                except Exception:
                    base_context = ""

            rollout_records = []
            best_record = None
            best_code = ""
            best_original_output = ""

            for idx, candidate in enumerate(rollout_candidates):
                candidate_code = extract_code(candidate.get("generated") or candidate.get("original_output") or "")
                infer_latency = float(candidate.get("infer_latency_sec", 0.0) or 0.0)

                result_list, eval_latency = _evaluate_single_rollout(
                    subproblem=subproblem,
                    code=candidate_code,
                    turn_num=turn_num,
                    overall_turns=overall_turns,
                    args=args,
                    temp_dir=temp_dir,
                    base_context=base_context,
                )

                pass_rate = (sum(result_list) / float(len(result_list))) if result_list else 0.0
                success = int(len(result_list) > 0 and all(x == 1 for x in result_list))
                total_latency = infer_latency + eval_latency

                record = {
                    "rollout_id": candidate.get("rollout_id", idx),
                    "infer_latency_sec": infer_latency,
                    "infer_batch_latency_sec": candidate.get("infer_batch_latency_sec"),
                    "infer_per_rollout_latency_sec": candidate.get("infer_per_rollout_latency_sec"),
                    "latency_observation": candidate.get("latency_observation"),
                    "eval_latency_sec": eval_latency,
                    "total_latency_sec": total_latency,
                    "harness_result": result_list,
                    "pass_rate": pass_rate,
                    "success": success,
                }
                rollout_records.append(record)

                if best_record is None:
                    best_record = record
                    best_code = candidate_code
                    best_original_output = candidate.get("original_output", "")
                else:
                    prev_score = (best_record["pass_rate"], -best_record["total_latency_sec"])
                    now_score = (record["pass_rate"], -record["total_latency_sec"])
                    if now_score > prev_score:
                        best_record = record
                        best_code = candidate_code
                        best_original_output = candidate.get("original_output", "")

            subproblem['rollout_records'] = rollout_records
            subproblem['harness_result'] = best_record["harness_result"] if best_record else [0]
            subproblem['selected_rollout_id'] = best_record["rollout_id"] if best_record else -1
            subproblem['generated'] = best_code
            if best_original_output:
                subproblem['original_output'] = best_original_output

            current_success = bool(best_record and best_record.get("success", 0) == 1)
            if args.require_successful_prereq and (not args.isolate_subtasks):
                prereq_context_ready = current_success

            # Persist best code only when chained context is enabled.
            if (
                (not args.isolate_subtasks)
                and (not args.golden_upstream_for_eval)
                and turn_num < overall_turns
                and best_code
                and (not args.require_successful_prereq or current_success)
            ):
                with open(args.temp_code, 'w', encoding='utf-8') as file:
                    file.write(base_context)
                    if base_context and not base_context.endswith('\n'):
                        file.write("\n")
                    file.write(best_code)
            elif (
                (not args.isolate_subtasks)
                and (not args.golden_upstream_for_eval)
                and args.require_successful_prereq
                and (not current_success)
            ):
                if os.path.exists(args.temp_code):
                    try:
                        os.remove(args.temp_code)
                    except OSError:
                        pass

        # Save result for the problem
        file_name = os.path.join(args.output_dir, f"{uuid}.json")
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(problem, f, ensure_ascii=False)
            print(f"Processed {uuid}")
        except Exception as e:
            print(f"Failed to save {uuid}: {e}")

        # Cleanup temp history file for the next problem
        if os.path.exists(args.temp_code):
            os.remove(args.temp_code)

if __name__ == '__main__':
    main()
