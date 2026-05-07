import json
import os
import argparse
import time
import sys
import fcntl
import tempfile
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.api import ChatModelAPI
from src.utils_api import get_filenames_without_extension, extract_code, get_input, ensure_python_code_block, ensure_python_code_block_main
# Use this file if you are evaluating on Codeflowbench-repo dataset.
# from src.utils_repo import (
#     get_filenames_without_extension,
#     extract_code,
#     get_input,
#     ensure_python_code_block,
#     ensure_python_code_block_main,
#     clean_code_block,
# )


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


def _is_subproblem_completed(subproblem):
    rollout_candidates = subproblem.get("rollout_candidates") or []
    return bool(rollout_candidates) or ("generated" in subproblem)


def _restore_history_and_start_turn(subproblems):
    history = []
    turn_number = 1
    for subproblem in subproblems:
        if not _is_subproblem_completed(subproblem):
            break
        history.append(subproblem.get("generated", ""))
        turn_number += 1
    return history, turn_number


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
        if obj.get("problem-id") == expected_problem_id:
            _atomic_write_json(path, obj)
            return obj
    if candidates:
        _atomic_write_json(path, candidates[-1])
        return candidates[-1]
    raise ValueError(f"Could not recover a valid JSON checkpoint from {path}")


class _ProblemFileLock:
    def __init__(self, lock_path):
        self.lock_path = lock_path
        self._fd = None

    def __enter__(self):
        parent_dir = os.path.dirname(self.lock_path) or "."
        os.makedirs(parent_dir, exist_ok=True)
        self._fd = open(self.lock_path, "a+", encoding="utf-8")
        fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        self._fd.seek(0)
        self._fd.truncate()
        self._fd.write(f"{os.getpid()}\n")
        self._fd.flush()
        os.fsync(self._fd.fileno())
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is None:
            return
        try:
            self._fd.seek(0)
            self._fd.truncate()
            self._fd.flush()
            os.fsync(self._fd.fileno())
        finally:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None


def main(args):
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chat_model = ChatModelAPI(api_url=args.api_url, api_key=args.api_key, model_name=args.model_name)
    for problem in data:
        problem_id = problem["problem-id"]
        output_path = os.path.join(output_dir, f"{problem_id}.json")
        lock_path = output_path + ".lock"

        try:
            with _ProblemFileLock(lock_path):
                if os.path.exists(output_path):
                    saved_problem = _load_checkpoint_problem(output_path, problem_id)
                    if saved_problem.get("problem-id") == problem_id:
                        problem = saved_problem

                problem_description = problem.get("problem-description", "")
                subproblems = problem["subproblems"]
                overall_turns = problem["overall-turns"]
                history, turn_number = _restore_history_and_start_turn(subproblems)

                if turn_number > overall_turns:
                    print(f"Skipping {problem_id}: completed checkpoint exists")
                    continue

                if turn_number > 1:
                    print(f"Resuming Problem {problem_id} from turn {turn_number}...")

                for subproblem in subproblems[turn_number - 1 :]:
                    input_text = get_input(subproblem, turn_number, overall_turns, problem_description, history)
                    turn_number += 1

                    rollout_candidates = []
                    for rollout_id in range(args.n_rollouts):
                        t0 = time.perf_counter()
                        generated = chat_model.generate(
                            input_text,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                        infer_latency_sec = time.perf_counter() - t0
                        output = generated.choices[0].message.content

                        if turn_number == overall_turns + 1:
                            output = ensure_python_code_block_main(output, subproblem)
                        else:
                            output = ensure_python_code_block(output)

                        code_output = extract_code(output)
                        rollout_candidates.append({
                            "rollout_id": rollout_id,
                            "infer_latency_sec": infer_latency_sec,
                            "original_output": output,
                            "generated": code_output,
                        })

                    # Legacy compatibility: keep a canonical generated code for downstream scripts.
                    chosen = rollout_candidates[0]
                    subproblem.update({
                        "original_output": chosen["original_output"],
                        "prompt": input_text,
                        "generated": chosen["generated"],
                        "rollout_candidates": rollout_candidates,
                    })
                    history.append(chosen["generated"])
                    _atomic_write_json(output_path, problem)

        except Exception as e:
            print(f"[Skipped] Error occurred while processing problem_id={problem_id}: {e}")
            continue

        print(f"Completed. Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name, used in output folder naming")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, required=True,help="Base output directory path")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the model")
    parser.add_argument("--api_url", type=str, required=True, help="URL of the inference API")
    parser.add_argument("--n_rollouts", type=int, default=1, help="Number of rollouts per subproblem")
    parser.add_argument("--max_tokens", type=int, default=38912, help="Maximum generation tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling cutoff")
    args = parser.parse_args()

    main(args)
