import argparse
import ast
import hashlib
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from contextlib import nullcontext
from pathlib import Path

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
RUN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if RUN_DIR not in sys.path:
    sys.path.append(RUN_DIR)

from src.local import ChatModel
from src import utils as comp_utils
from src import utils_repo as repo_utils

clean_code_block = comp_utils.clean_code_block
ensure_python_code_block = comp_utils.ensure_python_code_block
ensure_python_code_block_main = comp_utils.ensure_python_code_block_main
extract_code = comp_utils.extract_code
get_input = comp_utils.get_input
has_print = comp_utils.has_print
CURRENT_DATASET = "comp"


def _configure_dataset_utils(dataset):
    globals()["CURRENT_DATASET"] = dataset
    selected = repo_utils if dataset == "repo" else comp_utils
    globals()["clean_code_block"] = selected.clean_code_block
    globals()["ensure_python_code_block"] = selected.ensure_python_code_block
    globals()["ensure_python_code_block_main"] = selected.ensure_python_code_block_main
    globals()["extract_code"] = selected.extract_code
    globals()["get_input"] = selected.get_input


def _extract_function_block_from_generated(code_text, function_name):
    if not code_text or not function_name:
        return code_text
    text = re.sub(r"<think>.*?</think>", "", code_text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.replace("```python", "").replace("```", "").strip()
    lines = text.splitlines()
    pat = re.compile(r"^\s*def\s+" + re.escape(function_name) + r"\s*\(")
    starts = [idx for idx, line in enumerate(lines) if pat.match(line)]
    if not starts:
        return text
    start = starts[-1]
    base_indent = len(lines[start]) - len(lines[start].lstrip(" "))
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        cur = lines[idx]
        if not cur.strip():
            continue
        cur_indent = len(cur) - len(cur.lstrip(" "))
        if cur_indent <= base_indent and re.match(r"^\s*(def|class)\s+", cur):
            end = idx
            break
    imports = [
        line
        for line in lines[:start]
        if line.lstrip().startswith(("import ", "from "))
        and "<think>" not in line
        and "</think>" not in line
    ]
    block = textwrap.dedent("\n".join(lines[start:end])).rstrip()
    if imports:
        return "\n".join(imports + ["", block]).strip()
    return block


def _clean_candidate_output(output, subproblem):
    code_output = extract_code(output)
    code_output = clean_code_block(code_output)
    if CURRENT_DATASET == "repo":
        code_output = _extract_function_block_from_generated(
            code_output,
            subproblem.get("name", ""),
        )
    return code_output


def _atomic_write_json(path, payload):
    parent_dir = os.path.dirname(path) or "."
    os.makedirs(parent_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=parent_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _problem_identity(problem):
    return problem.get("problem-id") or problem.get("task_id") or "unknown_problem"


def _default_worker_id():
    return f"{socket.gethostname()}:{os.getpid()}"


def _now_str():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "item"


def _stable_int_hash(value):
    data = str(value).encode("utf-8", errors="replace")
    return int(hashlib.sha1(data).hexdigest()[:16], 16)


def _default_scan_seed(worker_id):
    raw = {
        "worker_id": worker_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time_ns": time.time_ns(),
    }
    return _stable_int_hash(json.dumps(raw, ensure_ascii=False, sort_keys=True))


def _worker_problem_indices(total_problems, worker_id, worker_shard_count, worker_shard_index):
    if total_problems <= 0:
        return []

    if worker_shard_count > 0:
        if not 0 <= worker_shard_index < worker_shard_count:
            raise ValueError(
                f"worker_shard_index must be in [0, {worker_shard_count}), got {worker_shard_index}"
            )
        indices = [
            idx for idx in range(total_problems) if (idx % worker_shard_count) == worker_shard_index
        ]
    else:
        if worker_shard_index != 0:
            raise ValueError(
                f"worker_shard_index requires worker_shard_count > 0, got index={worker_shard_index}"
            )
        indices = list(range(total_problems))

    if not indices:
        return []

    start_offset = _stable_int_hash(worker_id) % len(indices)
    return indices[start_offset:] + indices[:start_offset]


def _log_event(event, **fields):
    payload = {
        "ts": _now_str(),
        "event": event,
        **fields,
    }
    print("[codeflow] " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _safe_div(numerator, denominator):
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _count_tokens_from_text(tokenizer, text):
    if not text:
        return 0
    try:
        token_ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
        return len(token_ids)
    except Exception:
        return 0


def _prompt_token_count(chat_model, messages, generated):
    if generated:
        prompt_token_ids = getattr(generated[0], "prompt_token_ids", None)
        if prompt_token_ids is not None:
            try:
                return len(prompt_token_ids)
            except TypeError:
                pass
    try:
        prompt_text = chat_model.format_chat(messages)
    except Exception:
        prompt_text = ""
    return _count_tokens_from_text(chat_model.tokenizer, prompt_text)


def _output_token_count(chat_model, sampled):
    if isinstance(sampled, dict):
        token_ids = sampled.get("token_ids")
        if token_ids is not None:
            try:
                return len(token_ids)
            except TypeError:
                pass
        return _count_tokens_from_text(chat_model.tokenizer, sampled.get("text", ""))
    token_ids = getattr(sampled, "token_ids", None)
    if token_ids is not None:
        try:
            return len(token_ids)
        except TypeError:
            pass
    return _count_tokens_from_text(chat_model.tokenizer, getattr(sampled, "text", ""))


def _candidate_work_cost(candidate):
    total_tokens = candidate.get("total_tokens")
    try:
        if total_tokens is not None:
            return max(float(total_tokens), 1e-9)
    except (TypeError, ValueError):
        pass

    prompt_tokens = candidate.get("prompt_tokens")
    output_tokens = candidate.get("output_tokens")
    try:
        if prompt_tokens is not None or output_tokens is not None:
            return max(float(prompt_tokens or 0.0) + float(output_tokens or 0.0), 1e-9)
    except (TypeError, ValueError):
        pass

    infer_latency = candidate.get("infer_latency_sec")
    try:
        if infer_latency is not None:
            return max(float(infer_latency), 1e-9)
    except (TypeError, ValueError):
        pass

    return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_record(record):
    pass_rate = _safe_float(record.get("pass_rate"))
    total_latency = _safe_float(record.get("total_latency_sec"))
    return (
        pass_rate if pass_rate is not None else -1.0,
        -(total_latency if total_latency is not None else float("inf")),
    )


def _check_syntax(code_str):
    if not code_str:
        return False
    max_chars = int(os.environ.get("CODEFLOW_MAX_SYNTAX_CHECK_CHARS", "2000000"))
    if max_chars > 0 and len(code_str) > max_chars:
        return False
    try:
        import ast

        ast.parse(code_str)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError):
        return False


def _normalize_stdin_input(input_raw):
    if isinstance(input_raw, str):
        value = input_raw.strip("[]'").replace("\\n", "\n")
    else:
        value = str(input_raw)
    return " ".join(value.split())


def _normalize_expected_output(output_raw):
    return output_raw.replace("\\n", "\n") if isinstance(output_raw, str) else str(output_raw)


def _repo_eval_context():
    import collections as _collections
    import datetime as _datetime
    import itertools as _itertools
    import pathlib as _pathlib

    return {
        "collections": _collections,
        "datetime": _datetime,
        "itertools": _itertools,
        "math": __import__("math"),
        "pathlib": _pathlib,
        "Path": _pathlib.Path,
        "PosixPath": _pathlib.PosixPath,
        "WindowsPath": _pathlib.WindowsPath,
    }


def _is_repo_call_input(input_raw):
    if not isinstance(input_raw, str):
        return False
    try:
        parsed = ast.literal_eval(input_raw)
    except Exception:
        try:
            parsed = eval(input_raw, {"__builtins__": {}}, _repo_eval_context())
        except Exception:
            return False
    return (
        isinstance(parsed, tuple)
        and len(parsed) == 2
        and isinstance(parsed[0], (tuple, list))
        and isinstance(parsed[1], dict)
    )


def _should_use_repo_call_eval(test_cases, eval_style):
    if eval_style == "repo":
        return True
    if eval_style == "comp":
        return False
    return bool(test_cases) and all(_is_repo_call_input(case.get("input")) for case in test_cases)


def _evaluate_repo_call_rollout(
    subproblem,
    code,
    temp_code_path,
    assert_code_path,
    temp_dir,
    base_context,
    timeout_sec,
):
    result_list = []
    eval_start = time.perf_counter()
    test_cases = subproblem.get("test_code", [])
    if not test_cases:
        return [], time.perf_counter() - eval_start
    if not _check_syntax(code):
        return [0] * len(test_cases), time.perf_counter() - eval_start

    try:
        with open(temp_code_path, "w", encoding="utf-8") as file:
            if base_context.strip():
                file.write(base_context.rstrip())
                file.write("\n")
            file.write(code.rstrip())
            file.write("\n")
    except Exception:
        return [0] * len(test_cases), time.perf_counter() - eval_start

    function_name = subproblem["name"]
    module_name = Path(temp_code_path).stem

    for test_case in test_cases:
        driver_code = f"""
import sys
import os
import math
import collections
import itertools
import datetime as _dt_module
import pathlib as _pl_module

sys.path.append({repr(os.path.abspath(temp_dir))})
from {module_name} import *

eval_ctx = globals().copy()
eval_ctx["datetime"] = _dt_module
eval_ctx["pathlib"] = _pl_module
eval_ctx["Path"] = _pl_module.Path
eval_ctx["PosixPath"] = _pl_module.PosixPath
eval_ctx["WindowsPath"] = _pl_module.WindowsPath

def _same(actual, expected):
    if actual == expected:
        return True
    if isinstance(actual, float) and isinstance(expected, float):
        return abs(actual - expected) < 1e-6
    return False

try:
    args, kwargs = eval({repr(test_case.get("input"))}, eval_ctx)
    expected_output = eval({repr(test_case.get("output"))}, eval_ctx)
    actual_output = {function_name}(*args, **kwargs)
    if _same(actual_output, expected_output):
        print("PASSED")
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
"""
        with open(assert_code_path, "w", encoding="utf-8") as file:
            file.write(driver_code)
        try:
            result = subprocess.run(
                [sys.executable, assert_code_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            result_list.append(1 if result.returncode == 0 and "PASSED" in result.stdout else 0)
        except subprocess.TimeoutExpired:
            result_list.append(0)
        except Exception:
            result_list.append(0)
        finally:
            if os.path.exists(assert_code_path):
                os.remove(assert_code_path)

    if os.path.exists(temp_code_path):
        os.remove(temp_code_path)

    return result_list, time.perf_counter() - eval_start


def _evaluate_final_turn_rollout(
    subproblem,
    code,
    main_code_path,
    assert_code_path,
    temp_dir,
    base_context,
    timeout_sec,
    eval_style,
):
    result_list = []
    eval_start = time.perf_counter()
    test_cases = subproblem.get("test_code", [])
    if not test_cases:
        return [], time.perf_counter() - eval_start
    if _should_use_repo_call_eval(test_cases, eval_style):
        return _evaluate_repo_call_rollout(
            subproblem=subproblem,
            code=code,
            temp_code_path=main_code_path,
            assert_code_path=assert_code_path,
            temp_dir=temp_dir,
            base_context=base_context,
            timeout_sec=timeout_sec,
        )
    if not _check_syntax(code):
        return [0] * len(test_cases), time.perf_counter() - eval_start

    function_name = subproblem["name"]
    try:
        with open(main_code_path, "w", encoding="utf-8") as main:
            if base_context.strip():
                main.write(base_context.rstrip())
                main.write("\n")
            main.write("import sys\n")
            main.write(code.rstrip())
            main.write("\n")
            if has_print(code):
                main.write(f"{function_name}()\n")
            else:
                main.write(f"print({function_name}())\n")
    except Exception:
        return [0] * len(test_cases), time.perf_counter() - eval_start

    try:
        for test_case in test_cases:
            input_ = _normalize_stdin_input(test_case["input"])
            output = _normalize_expected_output(test_case["output"])
            try:
                result = subprocess.run(
                    [sys.executable, main_code_path],
                    capture_output=True,
                    text=True,
                    input=input_,
                    timeout=timeout_sec,
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
        if os.path.exists(main_code_path):
            os.remove(main_code_path)

    return result_list, time.perf_counter() - eval_start


def _evaluate_intermediate_turn_rollout(
    subproblem,
    code,
    temp_code_path,
    assert_code_path,
    temp_dir,
    base_context,
    timeout_sec,
    eval_style,
):
    result_list = []
    eval_start = time.perf_counter()
    test_cases = subproblem.get("test_code", [])
    if not test_cases:
        return [], time.perf_counter() - eval_start
    if _should_use_repo_call_eval(test_cases, eval_style):
        return _evaluate_repo_call_rollout(
            subproblem=subproblem,
            code=code,
            temp_code_path=temp_code_path,
            assert_code_path=assert_code_path,
            temp_dir=temp_dir,
            base_context=base_context,
            timeout_sec=timeout_sec,
        )
    if not _check_syntax(code):
        return [0] * len(test_cases), time.perf_counter() - eval_start

    try:
        with open(temp_code_path, "w", encoding="utf-8") as file:
            if base_context.strip():
                file.write(base_context.rstrip())
                file.write("\n")
            file.write(code.rstrip())
            file.write("\n")
    except Exception:
        return [0] * len(test_cases), time.perf_counter() - eval_start

    function_name = subproblem["name"]
    module_name = Path(temp_code_path).stem

    for test_case in test_cases:
        inp = test_case["input"]
        if isinstance(inp, str):
            inp = inp.replace(",)", ")")
        outp = test_case["output"]

        with open(assert_code_path, "w", encoding="utf-8") as file:
            file.write("import sys\n")
            file.write(f"sys.path.append({repr(os.path.abspath(temp_dir))})\n")
            file.write(f"from {module_name} import *\n")
            file.write(f"print({function_name}{inp})\n")

        try:
            result = subprocess.run(
                [sys.executable, assert_code_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
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
            if os.path.exists(assert_code_path):
                os.remove(assert_code_path)

    if os.path.exists(temp_code_path):
        os.remove(temp_code_path)

    return result_list, time.perf_counter() - eval_start


def _evaluate_single_rollout(
    subproblem,
    code,
    turn_num,
    overall_turns,
    temp_code_path,
    assert_code_path,
    main_code_path,
    temp_dir,
    base_context,
    timeout_sec,
    eval_style="auto",
):
    if turn_num == overall_turns:
        return _evaluate_final_turn_rollout(
            subproblem=subproblem,
            code=code,
            main_code_path=main_code_path,
            assert_code_path=assert_code_path,
            temp_dir=temp_dir,
            base_context=base_context,
            timeout_sec=timeout_sec,
            eval_style=eval_style,
        )

    return _evaluate_intermediate_turn_rollout(
        subproblem=subproblem,
        code=code,
        temp_code_path=temp_code_path,
        assert_code_path=assert_code_path,
        temp_dir=temp_dir,
        base_context=base_context,
        timeout_sec=timeout_sec,
        eval_style=eval_style,
    )


def _extract_solution_code(problem):
    solutions = problem.get("solutions", [])
    for item in solutions:
        if isinstance(item, dict) and item.get("type") == "code":
            content = item.get("content", "")
            code = extract_code(content)
            if code.strip():
                return code
    return ""


def _extract_function_block_from_code(code_text, function_name):
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
    for key in ["solution", "golden_solution", "reference_solution", "canonical_solution"]:
        val = subproblem.get(key)
        if isinstance(val, str) and val.strip():
            code = extract_code(val)
            if code.strip():
                return code

    name = subproblem.get("name", "")
    return _extract_function_block_from_code(fallback_problem_code, name)


def _collect_dep_closure(sub_name, deps_by_name):
    visited = set()
    ordered = []

    def dfs(name):
        if name in visited:
            return
        visited.add(name)
        for parent in deps_by_name.get(name, []):
            dfs(parent)
        ordered.append(name)

    for dep in deps_by_name.get(sub_name, []):
        dfs(dep)
    return ordered


def _build_golden_history_for_subtask(subproblem_name, deps_by_name, golden_by_name):
    ordered = _collect_dep_closure(subproblem_name, deps_by_name)
    history = []
    missing = []
    for name in ordered:
        code = golden_by_name.get(name, "").strip()
        if code:
            history.append(code)
        else:
            missing.append(name)
    return history, missing


def _subproblem_upstream_code(subproblem):
    upstream_generated = subproblem.get("upstream_generated")
    if isinstance(upstream_generated, str) and upstream_generated.strip():
        return upstream_generated
    generated = subproblem.get("generated")
    if isinstance(generated, str):
        return generated
    return ""


def _is_subproblem_completed(subproblem):
    rollout_candidates = subproblem.get("rollout_candidates") or []
    return bool(rollout_candidates) or ("generated" in subproblem)


def _restore_history_and_start_turn(subproblems):
    history = []
    turn_number = 1
    for subproblem in subproblems:
        if not _is_subproblem_completed(subproblem):
            break
        history.append(_subproblem_upstream_code(subproblem))
        turn_number += 1
    return history, turn_number


def _lookup_subproblem(problem, subproblem_name):
    subproblems = problem.get("subproblems", [])
    for idx, subproblem in enumerate(subproblems):
        if subproblem.get("name") == subproblem_name:
            return idx, subproblem
    return None, None


def _load_json_if_exists(path):
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_prior_problem(problem_id, directory, cache):
    if not directory:
        return None
    key = (directory, problem_id)
    if key not in cache:
        cache[key] = _load_json_if_exists(os.path.join(directory, f"{problem_id}.json"))
    return cache[key]


def _candidate_code(candidate):
    if not isinstance(candidate, dict):
        return ""
    generated = candidate.get("generated")
    if isinstance(generated, str) and generated.strip():
        return generated
    return extract_code(candidate.get("original_output", "") or "")


def _select_success_record(records):
    best_record = None
    for record in records or []:
        if int(bool(record.get("success"))) != 1:
            continue
        if best_record is None or _score_record(record) > _score_record(best_record):
            best_record = record
    return best_record


def _find_prior_success_upstream(problem_id, subproblem_name, args, verify_cache, rollout_cache):
    verify_problem = _load_prior_problem(problem_id, args.upstream_verify_dir, verify_cache)
    if not isinstance(verify_problem, dict):
        return None
    _, verify_subproblem = _lookup_subproblem(verify_problem, subproblem_name)
    if not isinstance(verify_subproblem, dict):
        return None

    best_record = _select_success_record(verify_subproblem.get("rollout_records") or [])
    if best_record is None:
        return None

    rollout_problem = _load_prior_problem(problem_id, args.upstream_rollout_dir, rollout_cache)
    if not isinstance(rollout_problem, dict):
        rollout_problem = verify_problem
    _, rollout_subproblem = _lookup_subproblem(rollout_problem, subproblem_name)
    if not isinstance(rollout_subproblem, dict):
        return None

    selected_rollout_id = best_record.get("rollout_id")
    for idx, candidate in enumerate(rollout_subproblem.get("rollout_candidates") or []):
        rollout_id = candidate.get("rollout_id", idx)
        if rollout_id != selected_rollout_id:
            continue
        code = _candidate_code(candidate)
        if not code.strip():
            return None
        return {
            "code": code,
            "rollout_id": selected_rollout_id,
            "source": "prior_verified_success",
            "pass_rate": best_record.get("pass_rate"),
            "success": best_record.get("success"),
        }
    return None


def _base_context_from_history(history):
    return "\n\n".join(item.strip() for item in history if isinstance(item, str) and item.strip()).rstrip()


def _evaluate_current_rollouts_for_upstream(
    problem_id,
    subproblem,
    rollout_candidates,
    current_turn,
    overall_turns,
    base_context,
    infer_problem_temp_dir,
    verify_timeout_sec,
    eval_style,
):
    rollout_records = []
    best_success_record = None
    for idx, candidate in enumerate(rollout_candidates):
        candidate_code = _candidate_code(candidate)
        task_dir = infer_problem_temp_dir / f"upstream_verify_{current_turn}_{idx}"
        task_dir.mkdir(parents=True, exist_ok=True)
        temp_code_path = str(task_dir / "temp_code.py")
        assert_code_path = str(task_dir / "assert_code.py")
        main_code_path = str(task_dir / "main_code.py")
        try:
            result_list, eval_latency = _evaluate_single_rollout(
                subproblem=subproblem,
                code=candidate_code,
                turn_num=current_turn,
                overall_turns=overall_turns,
                temp_code_path=temp_code_path,
                assert_code_path=assert_code_path,
                main_code_path=main_code_path,
                temp_dir=str(task_dir),
                base_context=base_context,
                timeout_sec=verify_timeout_sec,
                eval_style=eval_style,
            )
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

        infer_latency = float(candidate.get("infer_latency_sec", 0.0) or 0.0)
        pass_rate = (sum(result_list) / float(len(result_list))) if result_list else 0.0
        success = int(len(result_list) > 0 and all(item == 1 for item in result_list))
        record = {
            "rollout_id": candidate.get("rollout_id", idx),
            "infer_latency_sec": infer_latency,
            "infer_batch_latency_sec": candidate.get("infer_batch_latency_sec"),
            "infer_per_rollout_latency_sec": candidate.get("infer_per_rollout_latency_sec"),
            "latency_observation": candidate.get("latency_observation"),
            "eval_latency_sec": eval_latency,
            "total_latency_sec": infer_latency + eval_latency,
            "prompt_tokens": candidate.get("prompt_tokens"),
            "output_tokens": candidate.get("output_tokens"),
            "total_tokens": candidate.get("total_tokens"),
            "cost": _candidate_work_cost(candidate),
            "harness_result": result_list,
            "pass_rate": pass_rate,
            "success": success,
        }
        rollout_records.append(record)
        if success == 1 and (best_success_record is None or _score_record(record) > _score_record(best_success_record)):
            best_success_record = record

    if best_success_record is not None:
        selected_rollout_id = best_success_record["rollout_id"]
        selected_source = "current_verified_success"
        selected_record = best_success_record
    else:
        selected_rollout_id = rollout_candidates[0].get("rollout_id", 0) if rollout_candidates else 0
        selected_source = "fallback_rollout_0"
        selected_record = None
        for idx, record in enumerate(rollout_records):
            if record.get("rollout_id", idx) == selected_rollout_id:
                selected_record = record
                break

    selected_candidate = rollout_candidates[0] if rollout_candidates else {
        "rollout_id": 0,
        "original_output": "",
        "generated": "",
    }
    for idx, candidate in enumerate(rollout_candidates):
        if candidate.get("rollout_id", idx) == selected_rollout_id:
            selected_candidate = candidate
            break

    return {
        "rollout_records": rollout_records,
        "selected_rollout_id": selected_rollout_id,
        "selected_candidate": selected_candidate,
        "selected_record": selected_record,
        "upstream_code": _candidate_code(selected_candidate),
        "upstream_source": selected_source,
    }


def _build_rollout_candidates(chat_model, outputs, batch_infer_latency_sec, prompt_tokens, subproblem, is_final_turn):
    per_rollout_latency = batch_infer_latency_sec / float(max(len(outputs), 1))
    rollout_candidates = []
    for rollout_id, sampled in enumerate(outputs):
        output = sampled.get("text", "") if isinstance(sampled, dict) else sampled.text
        observed_latency = (
            _safe_float(sampled.get("infer_latency_sec")) if isinstance(sampled, dict) else None
        )
        observed_batch_latency = (
            _safe_float(sampled.get("infer_batch_latency_sec")) if isinstance(sampled, dict) else None
        )
        latency_observation = (
            sampled.get("latency_observation") if isinstance(sampled, dict) else "batch_request_average"
        )
        infer_latency = observed_latency if observed_latency is not None else batch_infer_latency_sec
        infer_batch_latency = observed_batch_latency if observed_batch_latency is not None else batch_infer_latency_sec
        if is_final_turn:
            output = ensure_python_code_block_main(output, subproblem)
        else:
            output = ensure_python_code_block(output)

        code_output = _clean_candidate_output(output, subproblem)
        output_tokens = _output_token_count(chat_model, sampled)
        total_tokens = prompt_tokens + output_tokens
        rollout_candidates.append(
            {
                "rollout_id": rollout_id,
                "infer_latency_sec": infer_latency,
                "infer_batch_latency_sec": infer_batch_latency,
                "infer_per_rollout_latency_sec": per_rollout_latency,
                "latency_observation": latency_observation,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": total_tokens,
                "cost_source": "tokens",
                "original_output": output,
                "generated": code_output,
            }
        )
    if not rollout_candidates:
        rollout_candidates.append(
            {
                "rollout_id": 0,
                "infer_latency_sec": batch_infer_latency_sec,
                "infer_batch_latency_sec": batch_infer_latency_sec,
                "infer_per_rollout_latency_sec": batch_infer_latency_sec,
                "latency_observation": "empty_batch_fallback",
                "prompt_tokens": prompt_tokens,
                "output_tokens": 0,
                "total_tokens": prompt_tokens,
                "cost": prompt_tokens,
                "cost_source": "tokens",
                "original_output": "",
                "generated": "",
            }
        )
    return rollout_candidates, per_rollout_latency


def _build_batch_plan(
    problemid,
    subproblems,
    start_turn,
    overall_turns,
    problem_description_now,
    history,
    args,
    prior_verify_cache,
    prior_rollout_cache,
):
    if not args.upstream_verify_dir:
        return []

    virtual_history = list(history)
    batch_plan = []
    for turn in range(start_turn, overall_turns + 1):
        subproblem = subproblems[turn - 1]
        subproblem_name = subproblem.get("name", f"turn_{turn}")
        upstream_choice = _find_prior_success_upstream(
            problemid,
            subproblem_name,
            args,
            prior_verify_cache,
            prior_rollout_cache,
        )
        if upstream_choice is None:
            break
        input_text = get_input(
            subproblem,
            turn,
            overall_turns,
            problem_description_now,
            virtual_history,
        )
        messages = [
            {
                "role": "system",
                "content": "You are a Programming Expert. You always provide correct and reliable code solutions.",
            },
            {"role": "user", "content": input_text},
        ]
        batch_plan.append(
            {
                "turn": turn,
                "subproblem": subproblem,
                "subproblem_name": subproblem_name,
                "input_text": input_text,
                "messages": messages,
                "upstream_choice": upstream_choice,
            }
        )
        virtual_history.append(upstream_choice["code"])
    return batch_plan


def _load_checkpoint_problem(path, expected_problem_id):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        pass

    raw_text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    candidates = []

    while idx < len(raw_text):
        while idx < len(raw_text) and raw_text[idx].isspace():
            idx += 1
        if idx >= len(raw_text):
            break
        try:
            obj, next_idx = decoder.raw_decode(raw_text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
        idx = next_idx

    for obj in reversed(candidates):
        if _problem_identity(obj) == expected_problem_id:
            _atomic_write_json(path, obj)
            return obj
    if candidates:
        _atomic_write_json(path, candidates[-1])
        return candidates[-1]
    raise ValueError(f"Could not recover a valid JSON checkpoint from {path}")


class _ClaimHeartbeat:
    def __init__(self, heartbeat_paths, interval_sec):
        self.heartbeat_paths = [Path(path) for path in heartbeat_paths]
        self.interval_sec = max(1.0, float(interval_sec))
        self._stop = threading.Event()
        self._thread = None

    def _touch_all(self):
        for heartbeat_path in self.heartbeat_paths:
            try:
                heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                heartbeat_path.touch()
            except OSError:
                continue

    def _run(self):
        while not self._stop.wait(self.interval_sec):
            self._touch_all()

    def __enter__(self):
        self._touch_all()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)
        self._touch_all()


def _resolve_claim_root(output_dir, claim_dir):
    if claim_dir:
        return Path(claim_dir)
    return Path(output_dir) / ".claims"


def _status_root(output_dir):
    return Path(output_dir) / ".status"


def _worker_status_path(output_dir, worker_id):
    return _status_root(output_dir) / "workers" / f"{_safe_name(worker_id)}.json"


def _summary_status_path(output_dir):
    return _status_root(output_dir) / "summary.json"


def _done_index_dir(claim_root):
    return Path(claim_root) / "_done_index"


def _active_index_dir(claim_root):
    return Path(claim_root) / "_active_index"


def _claim_name_for_problem(output_path, problem):
    problem_id = _problem_identity(problem)
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(problem_id)).strip("._") or "problem"
    key = json.dumps(
        {
            "output_path": str(Path(output_path).resolve()),
            "problem_id": problem_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{base}_{digest}"


def _done_marker_path(claim_root, claim_name):
    return _done_index_dir(claim_root) / f"{claim_name}.json"


def _active_marker_path(claim_root, claim_name):
    return _active_index_dir(claim_root) / f"{claim_name}.json"


def _load_done_claim_names(claim_root):
    done_dir = _done_index_dir(claim_root)
    try:
        entries = os.listdir(done_dir)
    except (FileNotFoundError, OSError):
        return set()
    done_names = set()
    for name in entries:
        if name.endswith(".json"):
            done_names.add(name[:-5])
    return done_names


def _write_done_marker(claim_root, claim_name, payload):
    done_dir = _done_index_dir(claim_root)
    done_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_done_marker_path(claim_root, claim_name), payload)


def _write_active_marker(claim_root, claim_name, payload):
    active_dir = _active_index_dir(claim_root)
    active_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_active_marker_path(claim_root, claim_name), payload)


def _remove_done_marker(claim_root, claim_name):
    marker_path = _done_marker_path(claim_root, claim_name)
    try:
        marker_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _remove_active_marker(claim_root, claim_name):
    marker_path = _active_marker_path(claim_root, claim_name)
    try:
        marker_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _path_exists_safe(path):
    try:
        return Path(path).exists()
    except OSError:
        return False


def _active_marker_age_sec(claim_root, claim_name):
    marker_path = _active_marker_path(claim_root, claim_name)
    try:
        return max(0.0, time.time() - marker_path.stat().st_mtime)
    except OSError:
        return None


def _is_claim_name_active(claim_root, claim_name, claim_timeout_sec):
    age = _active_marker_age_sec(claim_root, claim_name)
    if age is None:
        return False, None
    return age <= float(claim_timeout_sec), age


def _count_active_claims(claim_root, claim_timeout_sec):
    active_dir = _active_index_dir(claim_root)
    now = time.time()
    active = 0
    try:
        entries = list(active_dir.iterdir())
    except (FileNotFoundError, OSError):
        return 0
    for entry in entries:
        try:
            is_file = entry.is_file()
        except OSError:
            continue
        if not is_file or not entry.name.endswith(".json"):
            continue
        claim_name = entry.name[:-5]
        if _path_exists_safe(_done_marker_path(claim_root, claim_name)):
            continue
        try:
            age = max(0.0, now - entry.stat().st_mtime)
        except OSError:
            continue
        if age <= float(claim_timeout_sec):
            active += 1
    return active


def _write_worker_status(output_dir, worker_id, payload):
    total_problems = int(payload.get("total_problems", 0) or 0)
    completed_count = int(payload.get("completed_count", 0) or 0)
    processed_turns = int(payload.get("processed_turns", 0) or 0)
    processed_rollouts = int(payload.get("processed_rollouts", 0) or 0)
    infer_batch_seconds = float(payload.get("infer_batch_seconds", 0.0) or 0.0)
    latest_outputs = int(payload.get("latest_outputs", 0) or 0)
    latest_batch_infer_latency_sec = float(payload.get("latest_batch_infer_latency_sec", 0.0) or 0.0)
    status_path = _worker_status_path(output_dir, worker_id)
    payload = {
        "worker_id": worker_id,
        "updated_at": _now_str(),
        "completed_fraction": _safe_div(completed_count, total_problems),
        "avg_turns_per_infer_sec": _safe_div(processed_turns, infer_batch_seconds),
        "avg_rollouts_per_infer_sec": _safe_div(processed_rollouts, infer_batch_seconds),
        "avg_infer_sec_per_turn": _safe_div(infer_batch_seconds, processed_turns),
        "avg_infer_sec_per_rollout": _safe_div(infer_batch_seconds, processed_rollouts),
        "latest_rollouts_per_infer_sec": _safe_div(latest_outputs, latest_batch_infer_latency_sec),
        **payload,
    }
    try:
        _atomic_write_json(status_path, payload)
    except OSError as exc:
        _log_event(
            "worker_status_write_error",
            worker_id=worker_id,
            status_path=str(status_path),
            error=f"{type(exc).__name__}: {exc}",
        )


def _write_summary_status(output_dir, claim_root, total_problems, claim_timeout_sec, payload):
    done_markers = len(_load_done_claim_names(claim_root))
    active_claims = _count_active_claims(claim_root, claim_timeout_sec)
    processed_turns = int(payload.get("processed_turns", 0) or 0)
    processed_rollouts = int(payload.get("processed_rollouts", 0) or 0)
    infer_batch_seconds = float(payload.get("infer_batch_seconds", 0.0) or 0.0)
    summary = {
        "updated_at": _now_str(),
        "total_problems": total_problems,
        "done_markers": done_markers,
        "active_claims": active_claims,
        "done_fraction": _safe_div(done_markers, total_problems),
        "remaining_problems_estimate": max(0, int(total_problems) - int(done_markers) - int(active_claims)),
        "avg_turns_per_infer_sec": _safe_div(processed_turns, infer_batch_seconds),
        "avg_rollouts_per_infer_sec": _safe_div(processed_rollouts, infer_batch_seconds),
        "avg_infer_sec_per_turn": _safe_div(infer_batch_seconds, processed_turns),
        "avg_infer_sec_per_rollout": _safe_div(infer_batch_seconds, processed_rollouts),
        **payload,
    }
    summary_path = _summary_status_path(output_dir)
    try:
        _atomic_write_json(summary_path, summary)
    except OSError as exc:
        _log_event(
            "summary_status_write_error",
            status_path=str(summary_path),
            error=f"{type(exc).__name__}: {exc}",
        )


def _claim_last_update_sec(claim_path):
    newest = None
    for candidate in [
        Path(claim_path) / "heartbeat",
        Path(claim_path) / "worker_info.json",
        Path(claim_path),
    ]:
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        newest = mtime if newest is None else max(newest, mtime)
    return newest


def _checkpoint_is_complete(problem):
    subproblems = problem.get("subproblems", [])
    overall_turns = int(problem.get("overall-turns", 0) or 0)
    _, turn_number = _restore_history_and_start_turn(subproblems)
    return overall_turns > 0 and turn_number > overall_turns


def _done_claim_is_valid(output_path, expected_problem_id):
    if not os.path.exists(output_path):
        return False
    try:
        problem = _load_checkpoint_problem(output_path, expected_problem_id)
    except Exception:
        return False
    return _problem_identity(problem) == expected_problem_id and _checkpoint_is_complete(problem)


def _pending_worker_problem_indices(data, worker_problem_indices, output_dir, claim_root, done_claim_names):
    pending_indices = []
    for data_index in worker_problem_indices:
        problem = data[data_index]
        problemid = _problem_identity(problem)
        output_path = os.path.join(output_dir, f"{problemid}.json")
        claim_name = _claim_name_for_problem(output_path, problem)
        if claim_name in done_claim_names and os.path.exists(output_path):
            continue
        if claim_name in done_claim_names:
            _remove_done_marker(claim_root, claim_name)
            done_claim_names.discard(claim_name)
        pending_indices.append(data_index)
    return pending_indices


def _shuffle_problem_indices(problem_indices, seed):
    shuffled = list(problem_indices)
    rng = random.Random(int(seed))
    rng.shuffle(shuffled)
    return shuffled


def _try_claim_problem(output_path, problem, args):
    claim_root = _resolve_claim_root(args.output_dir, args.claim_dir)
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_name = _claim_name_for_problem(output_path, problem)
    claim_path = claim_root / claim_name
    done_marker_path = _done_marker_path(claim_root, claim_name)
    active_marker_path = _active_marker_path(claim_root, claim_name)
    worker_info_path = claim_path / "worker_info.json"
    done_path = claim_path / "done.json"
    heartbeat_path = claim_path / "heartbeat"
    now = time.time()
    released_stale = False

    for _ in range(2):
        try:
            claim_path.mkdir()
        except FileExistsError:
            if _path_exists_safe(done_marker_path):
                if os.path.exists(output_path):
                    _remove_active_marker(claim_root, claim_name)
                    return {
                        "enabled": True,
                        "acquired": False,
                        "reason": "done",
                        "claim_path": str(claim_path),
                        "claim_name": claim_name,
                        "released_stale_claim": released_stale,
                    }
                _remove_done_marker(claim_root, claim_name)

            active_marker_is_fresh, active_marker_age = _is_claim_name_active(
                claim_root,
                claim_name,
                args.claim_timeout_sec,
            )
            if active_marker_is_fresh:
                return {
                    "enabled": True,
                    "acquired": False,
                    "reason": "active",
                    "claim_path": str(claim_path),
                    "claim_name": claim_name,
                    "claim_age_sec": active_marker_age,
                    "released_stale_claim": released_stale,
                }

            if _path_exists_safe(done_path):
                if not _done_claim_is_valid(output_path, _problem_identity(problem)):
                    try:
                        shutil.rmtree(claim_path)
                        _remove_done_marker(claim_root, claim_name)
                        _remove_active_marker(claim_root, claim_name)
                        now = time.time()
                        continue
                    except FileNotFoundError:
                        now = time.time()
                        continue
                    except OSError:
                        pass
                _write_done_marker(
                    claim_root,
                    claim_name,
                    {
                        "status": "done",
                        "upgraded_from_done_claim": True,
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "problem_id": _problem_identity(problem),
                        "output_path": str(Path(output_path).resolve()),
                    },
                )
                _remove_active_marker(claim_root, claim_name)
                return {
                    "enabled": True,
                    "acquired": False,
                    "reason": "done",
                    "claim_path": str(claim_path),
                    "claim_name": claim_name,
                    "released_stale_claim": released_stale,
                }

            last_update = _claim_last_update_sec(claim_path)
            age = None if last_update is None else max(0.0, now - last_update)
            if age is not None and age > float(args.claim_timeout_sec):
                try:
                    shutil.rmtree(claim_path)
                    _remove_active_marker(claim_root, claim_name)
                    released_stale = True
                    now = time.time()
                    continue
                except FileNotFoundError:
                    now = time.time()
                    continue
                except OSError:
                    pass

            return {
                "enabled": True,
                "acquired": False,
                "reason": "active",
                "claim_path": str(claim_path),
                "claim_name": claim_name,
                "claim_age_sec": age,
                "released_stale_claim": released_stale,
            }

        worker_info = {
            "worker_id": args.worker_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "problem_id": _problem_identity(problem),
            "output_path": str(Path(output_path).resolve()),
            "input_file": str(Path(args.input_file).resolve()),
        }
        _atomic_write_json(worker_info_path, worker_info)
        _write_active_marker(claim_root, claim_name, worker_info)
        heartbeat_path.touch()
        return {
            "enabled": True,
            "acquired": True,
            "claim_path": str(claim_path),
            "claim_name": claim_name,
            "claim_root": str(claim_root),
            "heartbeat_path": str(heartbeat_path),
            "active_marker_path": str(active_marker_path),
            "done_path": str(done_path),
            "released_stale_claim": released_stale,
        }

    return {
        "enabled": True,
        "acquired": False,
        "reason": "active",
        "claim_path": str(claim_path),
        "claim_name": claim_name,
        "released_stale_claim": released_stale,
    }


def _finalize_claim(claim_info, status, payload):
    if not claim_info.get("enabled") or not claim_info.get("acquired"):
        return
    done_path = claim_info.get("done_path")
    if not done_path:
        return
    claim_name = claim_info.get("claim_name")
    claim_root = claim_info.get("claim_root")
    if claim_name and claim_root:
        _remove_active_marker(claim_root, claim_name)
    done_payload = {
        "status": status,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **payload,
    }
    _atomic_write_json(done_path, done_payload)


def main(args):
    _configure_dataset_utils(args.dataset)
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    claim_root = _resolve_claim_root(args.output_dir, args.claim_dir)
    claim_root.mkdir(parents=True, exist_ok=True)
    done_claim_names = _load_done_claim_names(claim_root)
    print(f"Using claim dir: {claim_root}")
    print(f"Loaded {len(done_claim_names)} done markers")

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chat_model = ChatModel(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        n_rollouts_hint=args.n_rollouts,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    prior_verify_cache = {}
    prior_rollout_cache = {}

    total_problems = len(data)
    worker_problem_indices = _worker_problem_indices(
        total_problems=total_problems,
        worker_id=args.worker_id,
        worker_shard_count=args.worker_shard_count,
        worker_shard_index=args.worker_shard_index,
    )
    pending_worker_problem_indices = _pending_worker_problem_indices(
        data=data,
        worker_problem_indices=worker_problem_indices,
        output_dir=args.output_dir,
        claim_root=claim_root,
        done_claim_names=done_claim_names,
    )
    scan_seed = args.scan_seed if args.scan_seed is not None else _default_scan_seed(args.worker_id)
    pending_worker_problem_indices = _shuffle_problem_indices(
        pending_worker_problem_indices,
        scan_seed,
    )
    claimed_count = 0
    skipped_done_count = 0
    skipped_active_count = 0
    stale_reclaimed_count = 0
    completed_count = 0
    processed_turns = 0
    processed_rollouts = 0
    infer_batch_seconds = 0.0
    assigned_problem_count = len(worker_problem_indices)
    pending_problem_count = len(pending_worker_problem_indices)

    _log_event(
        "worker_scan_plan",
        worker_id=args.worker_id,
        total_problems=total_problems,
        assigned_problem_count=assigned_problem_count,
        pending_problem_count=pending_problem_count,
        worker_shard_count=args.worker_shard_count,
        worker_shard_index=args.worker_shard_index,
        scan_seed=scan_seed,
        first_problem_indices=[idx + 1 for idx in pending_worker_problem_indices[:10]],
    )

    _write_worker_status(
        args.output_dir,
        args.worker_id,
        {
            "status": "starting",
            "total_problems": total_problems,
            "assigned_problem_count": assigned_problem_count,
            "pending_problem_count": pending_problem_count,
            "worker_shard_count": args.worker_shard_count,
            "worker_shard_index": args.worker_shard_index,
            "scan_seed": scan_seed,
            "claimed_count": claimed_count,
            "completed_count": completed_count,
            "processed_turns": processed_turns,
            "processed_rollouts": processed_rollouts,
            "infer_batch_seconds": infer_batch_seconds,
        },
    )
    _write_summary_status(
        args.output_dir,
        claim_root,
        total_problems,
        args.claim_timeout_sec,
        {
            "worker_id": args.worker_id,
            "status": "starting",
            "assigned_problem_count": assigned_problem_count,
            "pending_problem_count": pending_problem_count,
            "worker_shard_count": args.worker_shard_count,
            "worker_shard_index": args.worker_shard_index,
            "scan_seed": scan_seed,
            "claimed_count": claimed_count,
            "completed_count": completed_count,
        },
    )

    for assigned_position, data_index in enumerate(pending_worker_problem_indices, start=1):
        problem_index = data_index + 1
        problem = data[data_index]
        problemid = _problem_identity(problem)
        output_path = os.path.join(args.output_dir, f"{problemid}.json")
        claim_name = _claim_name_for_problem(output_path, problem)
        if claim_name in done_claim_names:
            if os.path.exists(output_path):
                skipped_done_count += 1
                _log_event(
                    "problem_skip_done_marker",
                    problem_id=problemid,
                    problem_index=problem_index,
                    total_problems=total_problems,
                )
                _write_worker_status(
                    args.output_dir,
                    args.worker_id,
                    {
                        "status": "skipping_done",
                        "problem_id": problemid,
                        "problem_index": problem_index,
                        "assigned_position": assigned_position,
                        "assigned_problem_count": assigned_problem_count,
                        "total_problems": total_problems,
                        "claimed_count": claimed_count,
                        "completed_count": completed_count,
                        "skipped_done_count": skipped_done_count,
                        "processed_turns": processed_turns,
                        "processed_rollouts": processed_rollouts,
                        "infer_batch_seconds": infer_batch_seconds,
                    },
                )
                _write_summary_status(
                    args.output_dir,
                    claim_root,
                    total_problems,
                    args.claim_timeout_sec,
                    {
                        "worker_id": args.worker_id,
                        "status": "skipping_done",
                        "current_problem_id": problemid,
                        "current_problem_index": problem_index,
                        "assigned_position": assigned_position,
                        "assigned_problem_count": assigned_problem_count,
                        "claimed_count": claimed_count,
                        "completed_count": completed_count,
                        "skipped_done_count": skipped_done_count,
                        "skipped_active_count": skipped_active_count,
                        "processed_turns": processed_turns,
                        "processed_rollouts": processed_rollouts,
                        "infer_batch_seconds": infer_batch_seconds,
                    },
                )
                continue
            _remove_done_marker(claim_root, claim_name)
            done_claim_names.discard(claim_name)

        claim_info = _try_claim_problem(output_path, problem, args)

        if claim_info.get("released_stale_claim"):
            stale_reclaimed_count += 1

        if not claim_info["acquired"]:
            if claim_info.get("reason") == "done":
                skipped_done_count += 1
                _log_event(
                    "problem_skip_done_claim",
                    problem_id=problemid,
                    problem_index=problem_index,
                    total_problems=total_problems,
                )
            else:
                skipped_active_count += 1
                claim_age = claim_info.get("claim_age_sec")
                _log_event(
                    "problem_skip_active_claim",
                    problem_id=problemid,
                    problem_index=problem_index,
                    total_problems=total_problems,
                    claim_age_sec=claim_age,
                )
            _write_worker_status(
                args.output_dir,
                args.worker_id,
                {
                    "status": "skipping",
                    "problem_id": problemid,
                    "problem_index": problem_index,
                    "assigned_position": assigned_position,
                    "assigned_problem_count": assigned_problem_count,
                    "total_problems": total_problems,
                    "claimed_count": claimed_count,
                    "completed_count": completed_count,
                    "skipped_done_count": skipped_done_count,
                    "skipped_active_count": skipped_active_count,
                    "processed_turns": processed_turns,
                    "processed_rollouts": processed_rollouts,
                    "infer_batch_seconds": infer_batch_seconds,
                },
            )
            _write_summary_status(
                args.output_dir,
                claim_root,
                total_problems,
                args.claim_timeout_sec,
                {
                    "worker_id": args.worker_id,
                    "status": "skipping",
                    "current_problem_id": problemid,
                    "current_problem_index": problem_index,
                    "assigned_position": assigned_position,
                    "assigned_problem_count": assigned_problem_count,
                    "claimed_count": claimed_count,
                    "completed_count": completed_count,
                    "skipped_done_count": skipped_done_count,
                    "skipped_active_count": skipped_active_count,
                    "stale_reclaimed_count": stale_reclaimed_count,
                    "processed_turns": processed_turns,
                    "processed_rollouts": processed_rollouts,
                    "infer_batch_seconds": infer_batch_seconds,
                },
            )
            continue

        claimed_count += 1
        _log_event(
            "problem_claimed",
            problem_id=problemid,
            problem_index=problem_index,
            total_problems=total_problems,
            claim_name=claim_info["claim_name"],
        )
        _write_summary_status(
            args.output_dir,
            claim_root,
            total_problems,
            args.claim_timeout_sec,
            {
                "worker_id": args.worker_id,
                "status": "claimed",
                "current_problem_id": problemid,
                "current_problem_index": problem_index,
                "assigned_position": assigned_position,
                "assigned_problem_count": assigned_problem_count,
                "claimed_count": claimed_count,
                "completed_count": completed_count,
                "skipped_done_count": skipped_done_count,
                "skipped_active_count": skipped_active_count,
                "stale_reclaimed_count": stale_reclaimed_count,
                "processed_turns": processed_turns,
                "processed_rollouts": processed_rollouts,
                "infer_batch_seconds": infer_batch_seconds,
            },
        )
        claim_ctx = (
            _ClaimHeartbeat(
                [
                    claim_info["heartbeat_path"],
                    claim_info["active_marker_path"],
                ],
                args.claim_heartbeat_sec,
            )
            if claim_info.get("enabled")
            else nullcontext()
        )

        try:
            with claim_ctx:
                if os.path.exists(output_path):
                    saved_problem = _load_checkpoint_problem(output_path, problemid)
                    if _problem_identity(saved_problem) == problemid:
                        problem = saved_problem

                if _checkpoint_is_complete(problem):
                    _log_event(
                        "problem_skip_completed_checkpoint",
                        problem_id=problemid,
                        problem_index=problem_index,
                        total_problems=total_problems,
                    )
                    _finalize_claim(
                        claim_info,
                        "done",
                        {
                            "worker_id": args.worker_id,
                            "problem_id": problemid,
                            "output_path": str(Path(output_path).resolve()),
                            "resume_from_existing_checkpoint": True,
                        },
                    )
                    _write_done_marker(
                        claim_root,
                        claim_info["claim_name"],
                        {
                            "status": "done",
                            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "problem_id": problemid,
                            "output_path": str(Path(output_path).resolve()),
                            "resume_from_existing_checkpoint": True,
                            "worker_id": args.worker_id,
                        },
                    )
                    done_claim_names.add(claim_info["claim_name"])
                    _write_worker_status(
                        args.output_dir,
                        args.worker_id,
                        {
                            "status": "skipping_done",
                            "problem_id": problemid,
                            "problem_index": problem_index,
                            "assigned_position": assigned_position,
                            "assigned_problem_count": assigned_problem_count,
                            "total_problems": total_problems,
                            "claimed_count": claimed_count,
                            "completed_count": completed_count,
                            "skipped_done_count": skipped_done_count,
                            "processed_turns": processed_turns,
                            "processed_rollouts": processed_rollouts,
                            "infer_batch_seconds": infer_batch_seconds,
                        },
                    )
                    _write_summary_status(
                        args.output_dir,
                        claim_root,
                        total_problems,
                        args.claim_timeout_sec,
                        {
                            "worker_id": args.worker_id,
                            "status": "skipping_done",
                            "current_problem_id": problemid,
                            "current_problem_index": problem_index,
                            "assigned_position": assigned_position,
                            "assigned_problem_count": assigned_problem_count,
                            "claimed_count": claimed_count,
                            "completed_count": completed_count,
                            "skipped_done_count": skipped_done_count,
                            "skipped_active_count": skipped_active_count,
                            "processed_turns": processed_turns,
                            "processed_rollouts": processed_rollouts,
                            "infer_batch_seconds": infer_batch_seconds,
                        },
                    )
                    continue

                problem_description_now = problem["problem-description"]
                subproblems = problem["subproblems"]
                overall_turns = problem["overall-turns"]
                history, turn_number = _restore_history_and_start_turn(subproblems)
                resumed_from_checkpoint = turn_number > 1
                verify_problem_temp_dir = Path(args.verify_temp_root) / str(problemid)
                verify_problem_temp_dir.mkdir(parents=True, exist_ok=True)
                fallback_problem_code = _extract_solution_code(problem)
                deps_by_name = {
                    sp.get("name", ""): sp.get("dependencies", [])
                    for sp in subproblems
                }
                golden_by_name = {
                    sp.get("name", ""): _pick_subproblem_golden_code(sp, fallback_problem_code)
                    for sp in subproblems
                }

                _log_event(
                    "problem_start",
                    problem_id=problemid,
                    problem_index=problem_index,
                    total_problems=total_problems,
                    overall_turns=overall_turns,
                    start_turn=turn_number,
                    resumed_from_checkpoint=resumed_from_checkpoint,
                    completed_turns_before_start=turn_number - 1,
                )
                _write_worker_status(
                    args.output_dir,
                    args.worker_id,
                    {
                        "status": "running_problem",
                        "problem_id": problemid,
                        "problem_index": problem_index,
                        "assigned_position": assigned_position,
                        "assigned_problem_count": assigned_problem_count,
                        "total_problems": total_problems,
                        "overall_turns": overall_turns,
                        "current_turn": turn_number,
                        "claimed_count": claimed_count,
                        "completed_count": completed_count,
                        "processed_turns": processed_turns,
                        "processed_rollouts": processed_rollouts,
                        "infer_batch_seconds": infer_batch_seconds,
                        "resumed_from_checkpoint": resumed_from_checkpoint,
                    },
                )

                while turn_number <= overall_turns:
                    batch_plan = _build_batch_plan(
                        problemid,
                        subproblems,
                        turn_number,
                        overall_turns,
                        problem_description_now,
                        history,
                        args,
                        prior_verify_cache,
                        prior_rollout_cache,
                    )
                    if len(batch_plan) >= 2 and args.allow_turn_batch_generation:
                        batch_messages = [item["messages"] for item in batch_plan]
                        _log_event(
                            "generate_batch_begin",
                            problem_id=problemid,
                            problem_index=problem_index,
                            total_problems=total_problems,
                            batch_turns=[item["turn"] for item in batch_plan],
                            batch_size=len(batch_plan),
                            n_rollouts=args.n_rollouts,
                        )
                        t0 = time.perf_counter()
                        generated_batch = chat_model.generate_batch(
                            batch_messages,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            n=args.n_rollouts,
                        )
                        batch_elapsed_sec = time.perf_counter() - t0
                        per_turn_batch_sec = batch_elapsed_sec / float(len(batch_plan))
                        infer_batch_seconds += batch_elapsed_sec
                        _log_event(
                            "generate_batch_end",
                            problem_id=problemid,
                            problem_index=problem_index,
                            total_problems=total_problems,
                            batch_turns=[item["turn"] for item in batch_plan],
                            batch_size=len(batch_plan),
                            batch_infer_latency_sec=round(batch_elapsed_sec, 6),
                        )
                        for item, request_output in zip(batch_plan, generated_batch):
                            current_turn = item["turn"]
                            subproblem = item["subproblem"]
                            subproblem_name = item["subproblem_name"]
                            input_text = item["input_text"]
                            upstream_choice = item["upstream_choice"]
                            outputs = request_output.outputs if request_output else []
                            prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
                            prompt_tokens = len(prompt_token_ids) if prompt_token_ids is not None else _count_tokens_from_text(
                                chat_model.tokenizer,
                                chat_model.format_chat(item["messages"]),
                            )
                            rollout_candidates, per_rollout_latency = _build_rollout_candidates(
                                chat_model,
                                outputs,
                                per_turn_batch_sec,
                                prompt_tokens,
                                subproblem,
                                is_final_turn=(current_turn == overall_turns),
                            )
                            processed_turns += 1
                            processed_rollouts += max(len(outputs), 1)
                            chosen = rollout_candidates[0]
                            subproblem.update({"prompt": input_text})
                            subproblem.update({"rollout_candidates": rollout_candidates})
                            subproblem["original_output"] = chosen["original_output"]
                            subproblem["generated"] = chosen["generated"]
                            subproblem["upstream_generated"] = upstream_choice["code"]
                            subproblem["upstream_source"] = upstream_choice["source"]
                            subproblem["upstream_rollout_id"] = upstream_choice["rollout_id"]
                            subproblem["upstream_selected_success"] = int(bool(upstream_choice.get("success")))
                            history.append(upstream_choice["code"])
                            _atomic_write_json(output_path, problem)
                            _log_event(
                                "checkpoint_saved",
                                problem_id=problemid,
                                problem_index=problem_index,
                                total_problems=total_problems,
                                turn=current_turn,
                                overall_turns=overall_turns,
                                subproblem_name=subproblem_name,
                                output_path=str(Path(output_path).resolve()),
                                processed_turns=processed_turns,
                                processed_rollouts=processed_rollouts,
                                infer_batch_seconds=round(infer_batch_seconds, 6),
                                upstream_source=upstream_choice["source"],
                                batch_mode=True,
                            )
                            _write_worker_status(
                                args.output_dir,
                                args.worker_id,
                                {
                                    "status": "checkpoint_saved",
                                    "problem_id": problemid,
                                    "problem_index": problem_index,
                                    "total_problems": total_problems,
                                    "overall_turns": overall_turns,
                                    "current_turn": current_turn,
                                    "subproblem_name": subproblem_name,
                                    "latest_outputs": len(outputs),
                                    "latest_batch_infer_latency_sec": per_turn_batch_sec,
                                    "latest_per_rollout_latency_sec": per_rollout_latency,
                                    "claimed_count": claimed_count,
                                    "completed_count": completed_count,
                                    "processed_turns": processed_turns,
                                    "processed_rollouts": processed_rollouts,
                                    "infer_batch_seconds": infer_batch_seconds,
                                },
                            )
                            _write_summary_status(
                                args.output_dir,
                                claim_root,
                                total_problems,
                                args.claim_timeout_sec,
                                {
                                    "worker_id": args.worker_id,
                                    "status": "running",
                                    "current_problem_id": problemid,
                                    "current_problem_index": problem_index,
                                    "current_turn": current_turn,
                                    "overall_turns": overall_turns,
                                    "subproblem_name": subproblem_name,
                                    "latest_outputs": len(outputs),
                                    "latest_batch_infer_latency_sec": per_turn_batch_sec,
                                    "latest_per_rollout_latency_sec": per_rollout_latency,
                                    "latest_rollouts_per_infer_sec": _safe_div(
                                        len(outputs), per_turn_batch_sec
                                    ),
                                    "claimed_count": claimed_count,
                                    "completed_count": completed_count,
                                    "processed_turns": processed_turns,
                                    "processed_rollouts": processed_rollouts,
                                    "infer_batch_seconds": infer_batch_seconds,
                                },
                            )
                        turn_number += len(batch_plan)
                        continue

                    subproblem = subproblems[turn_number - 1]
                    current_turn = turn_number
                    subproblem_name = subproblem.get("name", f"turn_{current_turn}")
                    current_history = history
                    using_golden_upstream = False
                    missing_golden_dependencies = []
                    if args.golden_upstream_for_gen and current_turn > 1:
                        golden_history, missing_golden_dependencies = _build_golden_history_for_subtask(
                            subproblem_name,
                            deps_by_name,
                            golden_by_name,
                        )
                        if not missing_golden_dependencies:
                            current_history = golden_history
                            using_golden_upstream = True
                    _log_event(
                        "turn_start",
                        problem_id=problemid,
                        problem_index=problem_index,
                        total_problems=total_problems,
                        turn=current_turn,
                        overall_turns=overall_turns,
                        subproblem_name=subproblem_name,
                        n_rollouts=args.n_rollouts,
                        using_golden_upstream=using_golden_upstream,
                        missing_golden_dependencies=missing_golden_dependencies,
                    )
                    _write_worker_status(
                        args.output_dir,
                        args.worker_id,
                        {
                            "status": "generating",
                            "problem_id": problemid,
                            "problem_index": problem_index,
                            "total_problems": total_problems,
                            "overall_turns": overall_turns,
                            "current_turn": current_turn,
                            "subproblem_name": subproblem_name,
                            "generate_started_at": _now_str(),
                            "using_golden_upstream": using_golden_upstream,
                            "missing_golden_dependencies": missing_golden_dependencies,
                            "claimed_count": claimed_count,
                            "completed_count": completed_count,
                            "processed_turns": processed_turns,
                            "processed_rollouts": processed_rollouts,
                            "infer_batch_seconds": infer_batch_seconds,
                        },
                    )
                    _write_summary_status(
                        args.output_dir,
                        claim_root,
                        total_problems,
                        args.claim_timeout_sec,
                        {
                            "worker_id": args.worker_id,
                            "status": "generating",
                            "current_problem_id": problemid,
                            "current_problem_index": problem_index,
                            "overall_turns": overall_turns,
                            "current_turn": current_turn,
                            "subproblem_name": subproblem_name,
                            "generate_started_at": _now_str(),
                            "using_golden_upstream": using_golden_upstream,
                            "missing_golden_dependencies": missing_golden_dependencies,
                            "claimed_count": claimed_count,
                            "completed_count": completed_count,
                            "skipped_done_count": skipped_done_count,
                            "skipped_active_count": skipped_active_count,
                            "processed_turns": processed_turns,
                            "processed_rollouts": processed_rollouts,
                            "infer_batch_seconds": infer_batch_seconds,
                        },
                    )
                    input_text = get_input(
                        subproblem,
                        turn_number,
                        overall_turns,
                        problem_description_now,
                        current_history,
                    )
                    input_all = [
                        {
                            "role": "system",
                            "content": "You are a Programming Expert. You always provide correct and reliable code solutions.",
                        },
                        {"role": "user", "content": input_text},
                    ]
                    _log_event(
                        "generate_begin",
                        problem_id=problemid,
                        problem_index=problem_index,
                        total_problems=total_problems,
                        turn=current_turn,
                        overall_turns=overall_turns,
                        subproblem_name=subproblem_name,
                        n_rollouts=args.n_rollouts,
                        using_golden_upstream=using_golden_upstream,
                        missing_golden_dependencies=missing_golden_dependencies,
                    )
                    t0 = time.perf_counter()
                    outputs = chat_model.generate_rollouts_with_timings(
                        input_all,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        n=args.n_rollouts,
                    )
                    batch_infer_latency_sec = time.perf_counter() - t0
                    prompt_token_ids = outputs[0].get("prompt_token_ids") if outputs and isinstance(outputs[0], dict) else None
                    prompt_tokens = (
                        len(prompt_token_ids)
                        if prompt_token_ids is not None
                        else _count_tokens_from_text(chat_model.tokenizer, chat_model.format_chat(input_all))
                    )
                    rollout_candidates, per_rollout_latency = _build_rollout_candidates(
                        chat_model,
                        outputs,
                        batch_infer_latency_sec,
                        prompt_tokens,
                        subproblem,
                        is_final_turn=(current_turn == overall_turns),
                    )
                    infer_batch_seconds += batch_infer_latency_sec
                    processed_turns += 1
                    processed_rollouts += max(len(outputs), 1)
                    _log_event(
                        "generate_end",
                        problem_id=problemid,
                        problem_index=problem_index,
                        total_problems=total_problems,
                        turn=current_turn,
                        overall_turns=overall_turns,
                        subproblem_name=subproblem_name,
                        outputs=len(outputs),
                        batch_infer_latency_sec=round(batch_infer_latency_sec, 6),
                        per_rollout_latency_sec=round(per_rollout_latency, 6),
                        rollouts_per_infer_sec=round(
                            _safe_div(len(outputs), batch_infer_latency_sec) or 0.0, 6
                        ),
                    )
                    chosen = rollout_candidates[0]
                    subproblem.update({"prompt": input_text})
                    subproblem.update({"rollout_candidates": rollout_candidates})

                    upstream_choice = None
                    if args.upstream_verify_dir:
                        upstream_choice = _find_prior_success_upstream(
                            problemid,
                            subproblem_name,
                            args,
                            prior_verify_cache,
                            prior_rollout_cache,
                        )

                    if upstream_choice is not None:
                        subproblem["original_output"] = chosen["original_output"]
                        subproblem["generated"] = chosen["generated"]
                        subproblem["upstream_generated"] = upstream_choice["code"]
                        subproblem["upstream_source"] = upstream_choice["source"]
                        subproblem["upstream_rollout_id"] = upstream_choice["rollout_id"]
                        subproblem["upstream_selected_success"] = int(bool(upstream_choice.get("success")))
                        _log_event(
                            "upstream_selected_from_prior_verify",
                            problem_id=problemid,
                            problem_index=problem_index,
                            total_problems=total_problems,
                            turn=current_turn,
                            overall_turns=overall_turns,
                            subproblem_name=subproblem_name,
                            upstream_rollout_id=upstream_choice["rollout_id"],
                            upstream_source=upstream_choice["source"],
                        )
                    elif args.verify_upstream_on_miss:
                        verify_result = _evaluate_current_rollouts_for_upstream(
                            problem_id=problemid,
                            subproblem=subproblem,
                            rollout_candidates=rollout_candidates,
                            current_turn=current_turn,
                            overall_turns=overall_turns,
                            base_context=_base_context_from_history(current_history),
                            infer_problem_temp_dir=verify_problem_temp_dir,
                            verify_timeout_sec=args.verify_timeout_sec,
                            eval_style=args.dataset,
                        )
                        subproblem["rollout_records"] = verify_result["rollout_records"]
                        subproblem["selected_rollout_id"] = verify_result["selected_rollout_id"]
                        subproblem["harness_result"] = (
                            verify_result["selected_record"].get("harness_result", [0])
                            if verify_result["selected_record"]
                            else [0]
                        )
                        subproblem["original_output"] = verify_result["selected_candidate"].get("original_output", "")
                        subproblem["generated"] = verify_result["selected_candidate"].get("generated", "")
                        subproblem["upstream_generated"] = verify_result["upstream_code"]
                        subproblem["upstream_source"] = verify_result["upstream_source"]
                        subproblem["upstream_rollout_id"] = verify_result["selected_rollout_id"]
                        subproblem["upstream_selected_success"] = int(
                            verify_result["upstream_source"] == "current_verified_success"
                        )
                        _log_event(
                            "upstream_selected_after_verify",
                            problem_id=problemid,
                            problem_index=problem_index,
                            total_problems=total_problems,
                            turn=current_turn,
                            overall_turns=overall_turns,
                            subproblem_name=subproblem_name,
                            upstream_rollout_id=verify_result["selected_rollout_id"],
                            upstream_source=verify_result["upstream_source"],
                        )
                    else:
                        subproblem["original_output"] = chosen["original_output"]
                        subproblem["generated"] = chosen["generated"]
                        subproblem["upstream_generated"] = chosen["generated"]
                        subproblem["upstream_source"] = "fallback_rollout_0"
                        subproblem["upstream_rollout_id"] = chosen.get("rollout_id", 0)
                        subproblem["upstream_selected_success"] = 0

                    history.append(subproblem.get("upstream_generated", chosen["generated"]))
                    _atomic_write_json(output_path, problem)
                    _log_event(
                        "checkpoint_saved",
                        problem_id=problemid,
                        problem_index=problem_index,
                        total_problems=total_problems,
                        turn=current_turn,
                        overall_turns=overall_turns,
                        subproblem_name=subproblem_name,
                        output_path=str(Path(output_path).resolve()),
                        processed_turns=processed_turns,
                        processed_rollouts=processed_rollouts,
                        infer_batch_seconds=round(infer_batch_seconds, 6),
                    )
                    _write_worker_status(
                        args.output_dir,
                        args.worker_id,
                        {
                            "status": "checkpoint_saved",
                            "problem_id": problemid,
                            "problem_index": problem_index,
                            "total_problems": total_problems,
                            "overall_turns": overall_turns,
                            "current_turn": current_turn,
                            "subproblem_name": subproblem_name,
                            "latest_outputs": len(outputs),
                            "latest_batch_infer_latency_sec": batch_infer_latency_sec,
                            "latest_per_rollout_latency_sec": per_rollout_latency,
                            "claimed_count": claimed_count,
                            "completed_count": completed_count,
                            "processed_turns": processed_turns,
                            "processed_rollouts": processed_rollouts,
                            "infer_batch_seconds": infer_batch_seconds,
                        },
                    )
                    _write_summary_status(
                        args.output_dir,
                        claim_root,
                        total_problems,
                        args.claim_timeout_sec,
                        {
                            "worker_id": args.worker_id,
                            "status": "running",
                            "current_problem_id": problemid,
                            "current_problem_index": problem_index,
                            "current_turn": current_turn,
                            "overall_turns": overall_turns,
                            "subproblem_name": subproblem_name,
                            "latest_outputs": len(outputs),
                            "latest_batch_infer_latency_sec": batch_infer_latency_sec,
                            "latest_per_rollout_latency_sec": per_rollout_latency,
                            "latest_rollouts_per_infer_sec": _safe_div(
                                len(outputs), batch_infer_latency_sec
                            ),
                            "claimed_count": claimed_count,
                            "completed_count": completed_count,
                            "processed_turns": processed_turns,
                            "processed_rollouts": processed_rollouts,
                            "infer_batch_seconds": infer_batch_seconds,
                        },
                    )
                    turn_number += 1

                shutil.rmtree(verify_problem_temp_dir, ignore_errors=True)
                _finalize_claim(
                    claim_info,
                    "done",
                    {
                        "worker_id": args.worker_id,
                        "problem_id": problemid,
                        "output_path": str(Path(output_path).resolve()),
                        "resume_from_existing_checkpoint": resumed_from_checkpoint,
                    },
                )
                _write_done_marker(
                    claim_root,
                    claim_info["claim_name"],
                    {
                        "status": "done",
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "problem_id": problemid,
                        "output_path": str(Path(output_path).resolve()),
                        "resume_from_existing_checkpoint": resumed_from_checkpoint,
                        "worker_id": args.worker_id,
                    },
                )
                done_claim_names.add(claim_info["claim_name"])
                completed_count += 1
                _log_event(
                    "problem_done",
                    problem_id=problemid,
                    problem_index=problem_index,
                    total_problems=total_problems,
                    overall_turns=overall_turns,
                    claimed_count=claimed_count,
                    completed_count=completed_count,
                    processed_turns=processed_turns,
                    processed_rollouts=processed_rollouts,
                    infer_batch_seconds=round(infer_batch_seconds, 6),
                    output_path=str(Path(output_path).resolve()),
                )
                _write_worker_status(
                    args.output_dir,
                    args.worker_id,
                    {
                        "status": "problem_done",
                        "problem_id": problemid,
                        "problem_index": problem_index,
                        "total_problems": total_problems,
                        "overall_turns": overall_turns,
                        "claimed_count": claimed_count,
                        "completed_count": completed_count,
                        "processed_turns": processed_turns,
                        "processed_rollouts": processed_rollouts,
                        "infer_batch_seconds": infer_batch_seconds,
                    },
                )
                _write_summary_status(
                    args.output_dir,
                    claim_root,
                    total_problems,
                    args.claim_timeout_sec,
                    {
                        "worker_id": args.worker_id,
                        "status": "problem_done",
                        "current_problem_id": problemid,
                        "current_problem_index": problem_index,
                        "overall_turns": overall_turns,
                        "claimed_count": claimed_count,
                        "completed_count": completed_count,
                        "skipped_done_count": skipped_done_count,
                        "skipped_active_count": skipped_active_count,
                        "stale_reclaimed_count": stale_reclaimed_count,
                        "processed_turns": processed_turns,
                        "processed_rollouts": processed_rollouts,
                        "infer_batch_seconds": infer_batch_seconds,
                    },
                )
        except Exception as exc:
            try:
                shutil.rmtree(verify_problem_temp_dir, ignore_errors=True)
            except Exception:
                pass
            _log_event(
                "problem_error",
                problem_id=problemid,
                problem_index=problem_index,
                total_problems=total_problems,
                error=f"{type(exc).__name__}: {exc}",
            )
            _finalize_claim(
                claim_info,
                "error",
                {
                    "worker_id": args.worker_id,
                    "problem_id": problemid,
                    "output_path": str(Path(output_path).resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            _write_worker_status(
                args.output_dir,
                args.worker_id,
                {
                    "status": "error",
                    "problem_id": problemid,
                    "problem_index": problem_index,
                    "total_problems": total_problems,
                    "claimed_count": claimed_count,
                    "completed_count": completed_count,
                    "skipped_done_count": skipped_done_count,
                    "skipped_active_count": skipped_active_count,
                    "stale_reclaimed_count": stale_reclaimed_count,
                    "processed_turns": processed_turns,
                    "processed_rollouts": processed_rollouts,
                    "infer_batch_seconds": infer_batch_seconds,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            _write_summary_status(
                args.output_dir,
                claim_root,
                total_problems,
                args.claim_timeout_sec,
                {
                    "worker_id": args.worker_id,
                    "status": "error",
                    "current_problem_id": problemid,
                    "current_problem_index": problem_index,
                    "claimed_count": claimed_count,
                    "completed_count": completed_count,
                    "skipped_done_count": skipped_done_count,
                    "skipped_active_count": skipped_active_count,
                    "stale_reclaimed_count": stale_reclaimed_count,
                    "processed_turns": processed_turns,
                    "processed_rollouts": processed_rollouts,
                    "infer_batch_seconds": infer_batch_seconds,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

    _write_worker_status(
        args.output_dir,
        args.worker_id,
        {
            "status": "completed",
            "total_problems": total_problems,
            "assigned_problem_count": assigned_problem_count,
            "worker_shard_count": args.worker_shard_count,
            "worker_shard_index": args.worker_shard_index,
            "claimed_count": claimed_count,
            "completed_count": completed_count,
            "skipped_done_count": skipped_done_count,
            "skipped_active_count": skipped_active_count,
            "stale_reclaimed_count": stale_reclaimed_count,
            "processed_turns": processed_turns,
            "processed_rollouts": processed_rollouts,
            "infer_batch_seconds": infer_batch_seconds,
        },
    )
    _write_summary_status(
        args.output_dir,
        claim_root,
        total_problems,
        args.claim_timeout_sec,
        {
            "worker_id": args.worker_id,
            "status": "completed",
            "assigned_problem_count": assigned_problem_count,
            "worker_shard_count": args.worker_shard_count,
            "worker_shard_index": args.worker_shard_index,
            "claimed_count": claimed_count,
            "completed_count": completed_count,
            "skipped_done_count": skipped_done_count,
            "skipped_active_count": skipped_active_count,
            "stale_reclaimed_count": stale_reclaimed_count,
            "processed_turns": processed_turns,
            "processed_rollouts": processed_rollouts,
            "infer_batch_seconds": infer_batch_seconds,
        },
    )
    _log_event(
        "run_summary",
        total_problems=total_problems,
        claimed_count=claimed_count,
        completed_count=completed_count,
        skipped_done_count=skipped_done_count,
        skipped_active_count=skipped_active_count,
        stale_reclaimed_count=stale_reclaimed_count,
        processed_turns=processed_turns,
        processed_rollouts=processed_rollouts,
        infer_batch_seconds=round(infer_batch_seconds, 6),
        claim_dir=str(claim_root),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-turn code generation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument(
        "--dataset",
        choices=["comp", "repo"],
        default="comp",
        help="Dataset family. Controls prompt templates, extraction, and immediate verification style.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--n_rollouts", type=int, default=1, help="Number of rollouts per subproblem")
    parser.add_argument("--max_num_seqs", type=int, default=0, help="vLLM max_num_seqs (0 means auto)")
    parser.add_argument("--max_tokens", type=int, default=38912, help="Maximum generation tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling cutoff")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization",
    )
    parser.add_argument(
        "--claim-dir",
        type=str,
        default="",
        help="Shared claim directory. Defaults to <output_dir>/.claims.",
    )
    parser.add_argument(
        "--claim-timeout-sec",
        type=float,
        default=1800.0,
        help="Reclaim a problem if its heartbeat has not been updated for this many seconds.",
    )
    parser.add_argument(
        "--claim-heartbeat-sec",
        type=float,
        default=30.0,
        help="Heartbeat update interval while processing a claimed problem.",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default="",
        help="Optional worker identifier. Defaults to hostname:pid.",
    )
    parser.add_argument(
        "--worker-shard-count",
        type=int,
        default=0,
        help="Optional number of worker shards. If > 0, this worker only scans its assigned shard.",
    )
    parser.add_argument(
        "--worker-shard-index",
        type=int,
        default=0,
        help="Optional shard index for this worker in [0, worker_shard_count). Ignored when worker_shard_count=0.",
    )
    parser.add_argument(
        "--scan-seed",
        type=int,
        default=None,
        help="Optional seed used to shuffle this worker's pending problem scan order. Defaults to a startup-specific seed.",
    )
    parser.add_argument(
        "--upstream-rollout-dir",
        type=str,
        default="",
        help="Optional directory containing prior per-problem rollout JSONs with full candidate outputs.",
    )
    parser.add_argument(
        "--upstream-verify-dir",
        type=str,
        default="",
        help="Optional directory containing prior per-problem verified JSONs with rollout_records.",
    )
    parser.add_argument(
        "--verify-upstream-on-miss",
        action="store_true",
        help="If no prior successful upstream rollout exists, verify current node rollouts immediately and persist rollout_records.",
    )
    parser.add_argument(
        "--verify-timeout-sec",
        type=float,
        default=5.0,
        help="Per-test timeout used for immediate upstream verification fallback.",
    )
    parser.add_argument(
        "--verify-temp-root",
        type=str,
        default="temp/inference_upstream_verify",
        help="Temporary directory root used for immediate upstream verification fallback.",
    )
    parser.add_argument(
        "--golden-upstream-for-gen",
        action="store_true",
        help="Use golden upstream dependency code during generation when runtime labels are available.",
    )
    parser.add_argument(
        "--allow-turn-batch-generation",
        action="store_true",
        help=(
            "Allow batching multiple DAG turns into one vLLM generate_batch call. "
            "Disabled by default because it cannot provide per-rollout finish latency."
        ),
    )

    args = parser.parse_args()
    if not args.worker_id:
        args.worker_id = _default_worker_id()
    if args.worker_shard_count < 0:
        raise ValueError(f"worker_shard_count must be >= 0, got {args.worker_shard_count}")
    main(args)
