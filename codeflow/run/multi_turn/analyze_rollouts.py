import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import has_print


class ProgressBar:
    def __init__(self, total_problems, total_nodes, total_rollouts, enabled=True):
        self.total_problems = max(0, int(total_problems))
        self.total_nodes = max(0, int(total_nodes))
        self.total_rollouts = max(0, int(total_rollouts))
        self.enabled = enabled and sys.stderr.isatty()
        self.done_problems = 0
        self.done_nodes = 0
        self.done_rollouts = 0
        self._last_width = 0
        self._lock = threading.Lock()

    def _bar(self, done, total, width=24):
        if total <= 0:
            return "-" * width
        ratio = min(1.0, max(0.0, done / float(total)))
        filled = int(ratio * width)
        return "#" * filled + "-" * (width - filled)

    def _render(self):
        if not self.enabled:
            return
        problem_bar = self._bar(self.done_problems, self.total_problems)
        node_text = f"{self.done_nodes}/{self.total_nodes}" if self.total_nodes else "0/0"
        rollout_text = f"{self.done_rollouts}/{self.total_rollouts}" if self.total_rollouts else "0/0"
        line = (
            f"\r[{problem_bar}] problems {self.done_problems}/{self.total_problems} "
            f"| nodes {node_text} | rollouts {rollout_text}"
        )
        pad = max(0, self._last_width - len(line))
        sys.stderr.write(line + (" " * pad))
        sys.stderr.flush()
        self._last_width = len(line)

    def advance_rollouts(self, count=1):
        with self._lock:
            self.done_rollouts = min(
                self.total_rollouts,
                self.done_rollouts + max(0, int(count)),
            )
            self._render()

    def advance_nodes(self, count=1):
        with self._lock:
            self.done_nodes = min(
                self.total_nodes,
                self.done_nodes + max(0, int(count)),
            )
            self._render()

    def advance_problem(self, count=1):
        with self._lock:
            self.done_problems = min(
                self.total_problems,
                self.done_problems + max(0, int(count)),
            )
            self._render()

    def finish(self):
        if not self.enabled:
            return
        with self._lock:
            self.done_problems = self.total_problems
            self.done_nodes = self.total_nodes
            self.done_rollouts = self.total_rollouts
            self._render()
            sys.stderr.write("\n")
            sys.stderr.flush()


class _PathFileLock:
    def __init__(self, lock_path):
        self.lock_path = str(lock_path)
        self._fd = None

    def __enter__(self):
        parent_dir = os.path.dirname(self.lock_path) or "."
        os.makedirs(parent_dir, exist_ok=True)
        self._fd = open(self.lock_path, "a+", encoding="utf-8")
        fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is None:
            return
        try:
            self._fd.flush()
            os.fsync(self._fd.fileno())
        finally:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None


class _ClaimHeartbeat:
    def __init__(self, heartbeat_path, interval_sec):
        self.heartbeat_path = Path(heartbeat_path)
        self.interval_sec = max(1.0, float(interval_sec))
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.wait(self.interval_sec):
            try:
                self.heartbeat_path.touch()
            except OSError:
                return

    def __enter__(self):
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.touch()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)
        try:
            self.heartbeat_path.touch()
        except OSError:
            pass


def _default_worker_id():
    return f"{socket.gethostname()}:{os.getpid()}"


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def _atomic_write_json(path, data):
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _problem_identity(problem):
    return problem.get("problem-id") or problem.get("task_id") or "unknown_problem"


def _claim_name_for_entry(entry):
    problem = entry["problem"]
    problem_id = _problem_identity(problem)
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(problem_id)).strip("._") or "problem"
    key = json.dumps(
        {
            "source_path": str(Path(entry["source_path"]).resolve()),
            "problem_index_in_file": entry["problem_index_in_file"],
            "problem_id": problem_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{base}_{entry['problem_index_in_file']}_{digest}"


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


def _try_claim_problem_entry(entry, args):
    if not args.claim_dir:
        return {
            "enabled": False,
            "acquired": True,
            "claim_path": None,
            "released_stale_claim": False,
        }

    claim_root = Path(args.claim_dir)
    claim_root.mkdir(parents=True, exist_ok=True)
    claim_name = _claim_name_for_entry(entry)
    claim_path = claim_root / claim_name
    worker_info_path = claim_path / "worker_info.json"
    done_path = claim_path / "done.json"
    heartbeat_path = claim_path / "heartbeat"
    now = time.time()
    released_stale = False

    for _ in range(2):
        try:
            claim_path.mkdir()
        except FileExistsError:
            if done_path.exists():
                return {
                    "enabled": True,
                    "acquired": False,
                    "reason": "done",
                    "claim_path": str(claim_path),
                    "released_stale_claim": released_stale,
                }

            last_update = _claim_last_update_sec(claim_path)
            age = None if last_update is None else max(0.0, now - last_update)
            if age is not None and age > float(args.claim_timeout_sec):
                try:
                    shutil.rmtree(claim_path)
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
                "claim_age_sec": age,
                "released_stale_claim": released_stale,
            }

        worker_info = {
            "worker_id": args.worker_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_path": entry["source_path"],
            "problem_index_in_file": entry["problem_index_in_file"],
            "problem_id": _problem_identity(entry["problem"]),
        }
        _atomic_write_json(worker_info_path, worker_info)
        heartbeat_path.touch()
        return {
            "enabled": True,
            "acquired": True,
            "claim_path": str(claim_path),
            "heartbeat_path": str(heartbeat_path),
            "done_path": str(done_path),
            "released_stale_claim": released_stale,
        }

    return {
        "enabled": True,
        "acquired": False,
        "reason": "active",
        "claim_path": str(claim_path),
        "released_stale_claim": released_stale,
    }


def _finalize_claim(claim_info, status, payload):
    if not claim_info.get("enabled") or not claim_info.get("acquired"):
        return
    done_path = claim_info.get("done_path")
    if not done_path:
        return
    done_payload = {
        "status": status,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **payload,
    }
    _atomic_write_json(done_path, done_payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze multi-turn inference rollout outputs directly. "
            "If rollout_records are missing, this script evaluates rollout_candidates "
            "against test_code first, then summarizes rollout/node/DAG statistics."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSON file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--dataset",
        choices=["auto", "comp", "repo"],
        default="auto",
        help="Dataset family used for testcase evaluation. auto detects repo-style (args, kwargs) inputs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write the summary JSON.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search JSON files when --input is a directory.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Do not run testcase evaluation; only summarize already-present rollout_records.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-evaluate rollout_candidates even when rollout_records are already present.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate evaluation time from input size; do not run testcase evaluation.",
    )
    parser.add_argument(
        "--save-evaluated-dir",
        type=str,
        default="",
        help="Optional directory to save per-problem JSON after rollout evaluation is attached.",
    )
    parser.add_argument(
        "--write-back",
        action="store_true",
        help="Write evaluated rollout_records and selected rollout info back to the source JSON file.",
    )
    parser.add_argument(
        "--temp-root",
        type=str,
        default="temp/analyze_rollouts",
        help="Directory for temporary evaluation files.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=5.0,
        help="Per-test execution timeout in seconds.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Fallback pass_rate threshold used when rollout_records have no explicit success field.",
    )
    parser.add_argument(
        "--isolate-subtasks",
        action="store_true",
        help="Do not pass generated code context between subtasks during evaluation.",
    )
    parser.add_argument(
        "--require-successful-prereq",
        action="store_true",
        help="Only pass generated context to the next subtask when the chosen previous rollout is fully successful.",
    )
    parser.add_argument(
        "--golden-upstream-for-eval",
        action="store_true",
        help="Evaluate each node with golden upstream dependency context when available.",
    )
    parser.add_argument(
        "--disable-runtime-golden-label",
        action="store_true",
        help="Disable golden-label availability annotation in saved/evaluated payloads.",
    )
    parser.add_argument(
        "--sample-problems",
        type=int,
        default=0,
        help="Randomly sample this many problems/DAGs before evaluation and analysis.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.0,
        help="Randomly sample this fraction of problems/DAGs before evaluation and analysis.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for problem sampling.",
    )
    parser.add_argument(
        "--estimate-eval-latency-sec",
        type=float,
        default=0.05,
        help="Fallback assumed eval latency per rollout when no observed eval latency is available.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Number of concurrent workers used to evaluate rollout candidates within a subtask.",
    )
    parser.add_argument(
        "--problem-workers",
        type=int,
        default=1,
        help="Number of concurrent workers used to evaluate different problems/DAGs in parallel.",
    )
    parser.add_argument(
        "--claim-dir",
        type=str,
        default="",
        help="Optional shared directory for problem-level task claiming across multiple machines/workers.",
    )
    parser.add_argument(
        "--claim-timeout-sec",
        type=float,
        default=1800.0,
        help="Reclaim a problem lock if its heartbeat has not been updated for this many seconds.",
    )
    parser.add_argument(
        "--claim-heartbeat-sec",
        type=float,
        default=30.0,
        help="Heartbeat update interval in seconds while processing a claimed problem.",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default="",
        help="Optional identifier written into claim metadata. Defaults to hostname:pid.",
    )
    parser.add_argument(
        "--keep-llm-outputs",
        action="store_true",
        help="Keep prompt/code/text outputs in persisted JSON. By default they are stripped to reduce write size.",
    )
    return parser.parse_args()


def collect_json_files(input_path, recursive):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(p for p in path.glob(pattern) if p.is_file())


def load_problems_from_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], {"path": str(path), "error": f"json_load_failed: {exc}"}

    if isinstance(data, dict):
        if "subproblems" in data:
            return [data], None
        return [], {"path": str(path), "error": "json_is_not_problem_payload"}

    if isinstance(data, list):
        problems = [item for item in data if isinstance(item, dict) and "subproblems" in item]
        if problems:
            return problems, None
        return [], {"path": str(path), "error": "json_list_has_no_problem_payload"}

    return [], {"path": str(path), "error": "unsupported_json_top_level"}


def format_duration(seconds):
    if seconds is None:
        return None
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def get_rollout_candidates_for_subproblem(subproblem):
    rollout_candidates = subproblem.get("rollout_candidates") or []
    if rollout_candidates:
        return rollout_candidates
    if subproblem.get("generated"):
        return [
            {
                "rollout_id": 0,
                "generated": subproblem.get("generated", ""),
                "original_output": subproblem.get("original_output", ""),
                "infer_latency_sec": 0.0,
            }
        ]
    return []


def load_problem_entries(json_files):
    entries = []
    files_with_problem_payload = 0
    files_skipped = []

    for json_file in json_files:
        problems, err = load_problems_from_json(json_file)
        if err is not None:
            files_skipped.append(err)
            continue
        files_with_problem_payload += 1
        for idx, problem in enumerate(problems):
            entries.append(
                {
                    "source_path": str(json_file),
                    "problem_index_in_file": idx,
                    "problem": problem,
                }
            )

    return entries, files_with_problem_payload, files_skipped


def inventory_problem_entries(entries):
    inventory = {
        "problems": len(entries),
        "nodes": 0,
        "nodes_with_rollout_candidates": 0,
        "nodes_with_rollout_records": 0,
        "nodes_needing_eval": 0,
        "candidate_rollouts": 0,
        "existing_rollout_records": 0,
        "rollouts_to_evaluate": 0,
    }

    for entry in entries:
        problem = entry["problem"]
        for subproblem in problem.get("subproblems", []):
            inventory["nodes"] += 1
            rollout_candidates = get_rollout_candidates_for_subproblem(subproblem)
            rollout_records = subproblem.get("rollout_records") or []

            if rollout_candidates:
                inventory["nodes_with_rollout_candidates"] += 1
                inventory["candidate_rollouts"] += len(rollout_candidates)
            if rollout_records:
                inventory["nodes_with_rollout_records"] += 1
                inventory["existing_rollout_records"] += len(rollout_records)
            else:
                if rollout_candidates:
                    inventory["nodes_needing_eval"] += 1
                    inventory["rollouts_to_evaluate"] += len(rollout_candidates)

    return inventory


def sample_problem_entries(entries, sample_problems, sample_fraction, sample_seed):
    total = len(entries)
    if total == 0:
        return entries, {"enabled": False, "selected_problems": 0, "total_problems": 0}

    if sample_problems > 0 and sample_fraction > 0:
        raise ValueError("Use only one of --sample-problems or --sample-fraction.")
    if sample_fraction < 0 or sample_fraction > 1:
        raise ValueError("--sample-fraction must be in [0, 1].")
    if sample_problems < 0:
        raise ValueError("--sample-problems must be >= 0.")

    if sample_problems > 0:
        target = min(total, sample_problems)
    elif sample_fraction > 0:
        target = min(total, max(1, int(round(total * sample_fraction))))
    else:
        target = total

    if target >= total:
        return entries, {
            "enabled": False,
            "selected_problems": total,
            "total_problems": total,
            "sample_seed": sample_seed,
        }

    rng = random.Random(sample_seed)
    selected_indices = sorted(rng.sample(range(total), target))
    sampled_entries = [entries[i] for i in selected_indices]
    return sampled_entries, {
        "enabled": True,
        "selected_problems": target,
        "total_problems": total,
        "sample_seed": sample_seed,
        "selected_indices_preview": selected_indices[:20],
    }


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bool_success(record, threshold):
    if "success" in record:
        return bool(record["success"])

    harness_result = record.get("harness_result")
    if isinstance(harness_result, list) and harness_result:
        return all(item == 1 for item in harness_result)

    pass_rate = safe_float(record.get("pass_rate"))
    if pass_rate is not None:
        return pass_rate >= threshold

    return None


def init_latency_bucket():
    return {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": None, "max": None}


def update_latency_bucket(bucket, value):
    if value is None:
        return
    bucket["count"] += 1
    bucket["sum"] += value
    bucket["sum_sq"] += value * value
    bucket["min"] = value if bucket["min"] is None else min(bucket["min"], value)
    bucket["max"] = value if bucket["max"] is None else max(bucket["max"], value)


def finalize_latency_bucket(bucket):
    count = bucket["count"]
    if count == 0:
        return {"count": 0}

    mean = bucket["sum"] / count
    variance = max(0.0, (bucket["sum_sq"] / count) - (mean * mean))
    return {
        "count": count,
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(variance),
        "min": bucket["min"],
        "max": bucket["max"],
    }


def init_group_bucket():
    return {
        "nodes": 0,
        "nodes_with_records": 0,
        "nodes_with_nonzero_success_rate": 0,
        "nodes_with_zero_success_rate": 0,
        "nodes_with_any_success": 0,
        "rollouts": 0,
        "successful_rollouts": 0,
        "infer_latency_sec": init_latency_bucket(),
        "total_latency_sec": init_latency_bucket(),
    }


def finalize_group_bucket(bucket):
    nodes_with_records = bucket["nodes_with_records"]
    rollouts = bucket["rollouts"]
    return {
        "nodes": bucket["nodes"],
        "nodes_with_records": nodes_with_records,
        "nodes_with_nonzero_success_rate": bucket["nodes_with_nonzero_success_rate"],
        "nodes_with_zero_success_rate": bucket["nodes_with_zero_success_rate"],
        "node_nonzero_success_rate": (
            bucket["nodes_with_nonzero_success_rate"] / nodes_with_records if nodes_with_records else None
        ),
        "nodes_with_any_success": bucket["nodes_with_any_success"],
        "node_any_success_rate": (
            bucket["nodes_with_any_success"] / nodes_with_records if nodes_with_records else None
        ),
        "rollouts": rollouts,
        "successful_rollouts": bucket["successful_rollouts"],
        "rollout_success_rate": (
            bucket["successful_rollouts"] / rollouts if rollouts else None
        ),
        "infer_latency_sec": finalize_latency_bucket(bucket["infer_latency_sec"]),
        "total_latency_sec": finalize_latency_bucket(bucket["total_latency_sec"]),
    }


def update_group_bucket(
    bucket,
    node_has_records,
    node_has_any_success,
    node_has_nonzero_success_rate,
    rollout_records,
):
    bucket["nodes"] += 1
    if node_has_records:
        bucket["nodes_with_records"] += 1
    if node_has_records and node_has_nonzero_success_rate is True:
        bucket["nodes_with_nonzero_success_rate"] += 1
    if node_has_records and node_has_nonzero_success_rate is False:
        bucket["nodes_with_zero_success_rate"] += 1
    if node_has_any_success:
        bucket["nodes_with_any_success"] += 1

    for record in rollout_records:
        bucket["rollouts"] += 1
        if record["success"] is True:
            bucket["successful_rollouts"] += 1
        update_latency_bucket(bucket["total_latency_sec"], record["total_latency_sec"])


def extract_code(text):
    if not text:
        return ""
    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        valid_blocks = [m.strip() for m in matches if m.strip()]
        return "\n\n".join(valid_blocks)
    return text.strip()


def check_syntax(code_str):
    if not code_str:
        return False
    try:
        ast.parse(code_str)
        return True
    except (SyntaxError, ValueError):
        return False


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


def _extract_last_function_block_from_generated(code_text, function_name):
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


def _candidate_code_for_eval(candidate, subproblem, dataset):
    test_cases = subproblem.get("test_code", [])
    if dataset == "repo" or (dataset == "auto" and _should_use_repo_call_eval(test_cases, "auto")):
        raw = candidate.get("original_output") or candidate.get("generated") or ""
        code = extract_code(raw)
        code = _extract_last_function_block_from_generated(code, subproblem.get("name", ""))
        return code
    raw = candidate.get("generated") or candidate.get("original_output") or ""
    return extract_code(raw)


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


def _build_golden_context_for_subtask(subproblem_name, deps_by_name, golden_by_name):
    ordered = _collect_dep_closure(subproblem_name, deps_by_name)
    chunks = []
    for name in ordered:
        code = golden_by_name.get(name, "").strip()
        if code:
            chunks.append(code)
    return "\n\n".join(chunks).rstrip()


def _annotate_runtime_golden_labels(problem, subproblems, fallback_problem_code):
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
        "math": math,
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


def _should_use_repo_call_eval(test_cases, dataset):
    if dataset == "repo":
        return True
    if dataset == "comp":
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
    if not check_syntax(code):
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


def _score_record(record):
    pass_rate = safe_float(record.get("pass_rate"))
    total_latency = safe_float(record.get("total_latency_sec"))
    return (
        pass_rate if pass_rate is not None else -1.0,
        -(total_latency if total_latency is not None else float("inf")),
    )


def _evaluate_final_turn_rollout(
    subproblem,
    code,
    main_code_path,
    assert_code_path,
    temp_dir,
    base_context,
    timeout_sec,
    dataset,
):
    result_list = []
    eval_start = time.perf_counter()

    test_cases = subproblem.get("test_code", [])
    if not test_cases:
        return [], time.perf_counter() - eval_start
    if _should_use_repo_call_eval(test_cases, dataset):
        return _evaluate_repo_call_rollout(
            subproblem=subproblem,
            code=code,
            temp_code_path=main_code_path,
            assert_code_path=assert_code_path,
            temp_dir=temp_dir,
            base_context=base_context,
            timeout_sec=timeout_sec,
        )

    if not check_syntax(code):
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
    dataset,
):
    result_list = []
    eval_start = time.perf_counter()

    test_cases = subproblem.get("test_code", [])
    if not test_cases:
        return [], time.perf_counter() - eval_start
    if _should_use_repo_call_eval(test_cases, dataset):
        return _evaluate_repo_call_rollout(
            subproblem=subproblem,
            code=code,
            temp_code_path=temp_code_path,
            assert_code_path=assert_code_path,
            temp_dir=temp_dir,
            base_context=base_context,
            timeout_sec=timeout_sec,
        )

    if not check_syntax(code):
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
    dataset="auto",
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
            dataset=dataset,
        )

    return _evaluate_intermediate_turn_rollout(
        subproblem=subproblem,
        code=code,
        temp_code_path=temp_code_path,
        assert_code_path=assert_code_path,
        temp_dir=temp_dir,
        base_context=base_context,
        timeout_sec=timeout_sec,
        dataset=dataset,
    )


def _normalize_existing_rollout_records(subproblem, threshold):
    records = subproblem.get("rollout_records") or []
    normalized = []
    for idx, record in enumerate(records):
        infer_latency = safe_float(record.get("infer_latency_sec"))
        infer_batch_latency = safe_float(record.get("infer_batch_latency_sec"))
        infer_per_rollout_latency = safe_float(record.get("infer_per_rollout_latency_sec"))
        eval_latency = safe_float(record.get("eval_latency_sec"))
        total_latency = safe_float(record.get("total_latency_sec"))
        if total_latency is None and infer_latency is not None and eval_latency is not None:
            total_latency = infer_latency + eval_latency
        harness_result = record.get("harness_result")
        pass_rate = safe_float(record.get("pass_rate"))
        if pass_rate is None and isinstance(harness_result, list) and harness_result:
            pass_rate = sum(harness_result) / float(len(harness_result))
        success = bool_success(record, threshold)
        normalized.append(
            {
                "rollout_id": record.get("rollout_id", idx),
                "infer_latency_sec": infer_latency,
                "infer_batch_latency_sec": infer_batch_latency,
                "infer_per_rollout_latency_sec": infer_per_rollout_latency,
                "latency_observation": record.get("latency_observation"),
                "eval_latency_sec": eval_latency,
                "total_latency_sec": total_latency,
                "harness_result": harness_result if isinstance(harness_result, list) else [],
                "pass_rate": pass_rate if pass_rate is not None else 0.0,
                "success": int(bool(success)) if success is not None else 0,
            }
        )
    return normalized


def _evaluate_candidate_rollout_task(task):
    task_dir = Path(task["task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    temp_code_path = str(task_dir / "temp_code.py")
    assert_code_path = str(task_dir / "assert_code.py")
    main_code_path = str(task_dir / "main_code.py")

    try:
        result_list, eval_latency = _evaluate_single_rollout(
            subproblem=task["subproblem"],
            code=task["candidate_code"],
            turn_num=task["turn_num"],
            overall_turns=task["overall_turns"],
            temp_code_path=temp_code_path,
            assert_code_path=assert_code_path,
            main_code_path=main_code_path,
            temp_dir=str(task_dir),
            base_context=task["base_context"],
            timeout_sec=task["timeout_sec"],
            dataset=task["dataset"],
        )
        pass_rate = (sum(result_list) / float(len(result_list))) if result_list else 0.0
        success = int(len(result_list) > 0 and all(x == 1 for x in result_list))
        total_latency = task["infer_latency"] + eval_latency
        return {
            "idx": task["idx"],
            "record": {
                "rollout_id": task["rollout_id"],
                "infer_latency_sec": task["infer_latency"],
                "infer_batch_latency_sec": task.get("infer_batch_latency"),
                "infer_per_rollout_latency_sec": task.get("infer_per_rollout_latency"),
                "latency_observation": task.get("latency_observation"),
                "eval_latency_sec": eval_latency,
                "total_latency_sec": total_latency,
                "harness_result": result_list,
                "pass_rate": pass_rate,
                "success": success,
            },
        }
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


def evaluate_problem_rollouts(problem, args, progress=None):
    evaluation_meta = {
        "problem_evaluated": False,
        "nodes_evaluated": 0,
        "rollouts_evaluated": 0,
    }
    if args.skip_eval:
        return evaluation_meta

    uuid = problem.get("problem-id") or problem.get("task_id") or "unknown_problem"
    problem_temp_dir = Path(args.temp_root) / str(uuid)
    problem_temp_dir.mkdir(parents=True, exist_ok=True)
    temp_code_path = str(problem_temp_dir / "temp_code.py")
    assert_code_path = str(problem_temp_dir / "assert_code.py")
    main_code_path = str(problem_temp_dir / "main_code.py")

    subproblems = problem.get("subproblems", [])
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

    for turn_num, subproblem in enumerate(subproblems, start=1):
        for file_path in [temp_code_path, assert_code_path, main_code_path]:
            if os.path.exists(file_path) and turn_num == 1:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

        if args.isolate_subtasks and os.path.exists(temp_code_path):
            try:
                os.remove(temp_code_path)
            except OSError:
                pass

        rollout_candidates = get_rollout_candidates_for_subproblem(subproblem)

        if not rollout_candidates and not (subproblem.get("rollout_records") or []):
            subproblem["harness_result"] = [0]
            subproblem["rollout_records"] = []
            subproblem["selected_rollout_id"] = -1
            if args.require_successful_prereq and (not args.isolate_subtasks):
                prereq_context_ready = False
            continue

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
            rollout_records = [
                {
                    "rollout_id": candidate.get("rollout_id", idx),
                    "infer_latency_sec": float(candidate.get("infer_latency_sec", 0.0) or 0.0),
                    "infer_batch_latency_sec": candidate.get("infer_batch_latency_sec"),
                    "infer_per_rollout_latency_sec": candidate.get("infer_per_rollout_latency_sec"),
                    "latency_observation": candidate.get("latency_observation"),
                    "eval_latency_sec": 0.0,
                    "total_latency_sec": float(candidate.get("infer_latency_sec", 0.0) or 0.0),
                    "harness_result": blocked_result,
                    "pass_rate": 0.0,
                    "success": 0,
                    "blocked_by_prereq_failure": 1,
                }
                for idx, candidate in enumerate(rollout_candidates)
            ]
            subproblem["rollout_records"] = rollout_records
            subproblem["harness_result"] = blocked_result
            subproblem["selected_rollout_id"] = -1
            subproblem["generated"] = ""
            evaluation_meta["problem_evaluated"] = True
            evaluation_meta["nodes_evaluated"] += 1
            evaluation_meta["rollouts_evaluated"] += len(rollout_records)
            if progress is not None:
                progress.advance_rollouts(len(rollout_records))
                progress.advance_nodes(1)
            continue

        if args.golden_upstream_for_eval:
            base_context = _build_golden_context_for_subtask(
                subproblem.get("name", ""),
                deps_by_name,
                golden_by_name,
            )
        elif (not args.isolate_subtasks) and os.path.exists(temp_code_path):
            try:
                with open(temp_code_path, "r", encoding="utf-8") as temp:
                    base_context = temp.read().rstrip()
            except Exception:
                base_context = ""
        else:
            base_context = ""

        rollout_records = subproblem.get("rollout_records") or []
        best_record = None
        best_code = subproblem.get("generated", "") or ""
        best_original_output = subproblem.get("original_output", "") or ""

        if rollout_records and not args.force_eval:
            rollout_records = _normalize_existing_rollout_records(subproblem, args.success_threshold)
        else:
            tasks = []
            for idx, candidate in enumerate(rollout_candidates):
                tasks.append(
                    {
                        "idx": idx,
                        "rollout_id": candidate.get("rollout_id", idx),
                        "candidate_code": _candidate_code_for_eval(
                            candidate,
                            subproblem,
                            args.dataset,
                        ),
                        "infer_latency": float(candidate.get("infer_latency_sec", 0.0) or 0.0),
                        "infer_batch_latency": safe_float(candidate.get("infer_batch_latency_sec")),
                        "infer_per_rollout_latency": safe_float(candidate.get("infer_per_rollout_latency_sec")),
                        "latency_observation": candidate.get("latency_observation"),
                        "subproblem": subproblem,
                        "turn_num": turn_num,
                        "overall_turns": overall_turns,
                        "base_context": base_context,
                        "timeout_sec": args.timeout_sec,
                        "dataset": args.dataset,
                        "task_dir": str(problem_temp_dir / f"candidate_{turn_num}_{idx}"),
                    }
                )

            rollout_records = [None] * len(tasks)
            max_workers = min(max(1, int(args.eval_workers)), len(tasks)) if tasks else 1
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(_evaluate_candidate_rollout_task, task): task["idx"]
                        for task in tasks
                    }
                    for future in as_completed(future_to_idx):
                        result = future.result()
                        rollout_records[result["idx"]] = result["record"]
                        if progress is not None:
                            progress.advance_rollouts(1)
            else:
                for task in tasks:
                    result = _evaluate_candidate_rollout_task(task)
                    rollout_records[result["idx"]] = result["record"]
                    if progress is not None:
                        progress.advance_rollouts(1)

            rollout_records = [record for record in rollout_records if record is not None]

            subproblem["rollout_records"] = rollout_records
            evaluation_meta["problem_evaluated"] = True
            evaluation_meta["nodes_evaluated"] += 1
            evaluation_meta["rollouts_evaluated"] += len(rollout_records)
            if progress is not None:
                progress.advance_nodes(1)

        selected_rollout_id = subproblem.get("selected_rollout_id")
        for record in rollout_records:
            if selected_rollout_id is not None and record.get("rollout_id") == selected_rollout_id:
                best_record = record
                break
        if best_record is None:
            for record in rollout_records:
                if best_record is None or _score_record(record) > _score_record(best_record):
                    best_record = record

        if best_record is not None:
            selected_rollout_id = best_record["rollout_id"]
            for idx, candidate in enumerate(rollout_candidates):
                if candidate.get("rollout_id", idx) == selected_rollout_id:
                    best_code = _candidate_code_for_eval(candidate, subproblem, args.dataset)
                    best_original_output = candidate.get("original_output", best_original_output)
                    break

        subproblem["selected_rollout_id"] = selected_rollout_id if selected_rollout_id is not None else -1
        subproblem["harness_result"] = best_record.get("harness_result", [0]) if best_record else [0]
        subproblem["generated"] = best_code
        if best_original_output:
            subproblem["original_output"] = best_original_output

        current_success = bool(best_record and safe_int(best_record.get("success")) == 1)
        if args.require_successful_prereq and (not args.isolate_subtasks):
            prereq_context_ready = current_success

        if (
            (not args.isolate_subtasks)
            and (not args.golden_upstream_for_eval)
            and turn_num < overall_turns
            and best_code
            and (not args.require_successful_prereq or current_success)
        ):
            with open(temp_code_path, "w", encoding="utf-8") as file:
                if base_context:
                    file.write(base_context)
                    if not base_context.endswith("\n"):
                        file.write("\n")
                file.write(best_code)
        elif (
            (not args.isolate_subtasks)
            and (not args.golden_upstream_for_eval)
            and args.require_successful_prereq
            and (not current_success)
            and os.path.exists(temp_code_path)
        ):
            try:
                os.remove(temp_code_path)
            except OSError:
                pass

    shutil.rmtree(problem_temp_dir, ignore_errors=True)
    return evaluation_meta


def analyze_problem(problem, summary, success_threshold, source_path=None):
    summary["problems"] += 1
    summary["dags"]["total"] += 1

    problem_id = problem.get("problem-id") or problem.get("task_id") or f"problem_{summary['problems']}"
    subproblems = problem.get("subproblems", [])
    problem_all_nodes_evaluated = True
    problem_has_zero_success_node = False
    per_node_details = []

    for turn_index, subproblem in enumerate(subproblems, start=1):
        summary["nodes"]["total"] += 1

        candidates = subproblem.get("rollout_candidates")
        records = subproblem.get("rollout_records")
        selected_rollout_id = subproblem.get("selected_rollout_id")
        depth = subproblem.get("depth")

        candidate_infer_latencies = []
        turn_key = str(turn_index)
        depth_key = str(depth) if depth is not None else "unknown"
        turn_bucket = summary["breakdown"]["turn_index"].setdefault(turn_key, init_group_bucket())
        depth_bucket = summary["breakdown"]["depth"].setdefault(depth_key, init_group_bucket())

        if isinstance(candidates, list) and candidates:
            summary["nodes"]["with_rollout_candidates"] += 1
            summary["rollouts"]["candidate_total"] += len(candidates)
            for candidate in candidates:
                infer_latency = safe_float(candidate.get("infer_latency_sec"))
                candidate_infer_latencies.append(infer_latency)
                update_latency_bucket(summary["latency_sec"]["infer_all"], infer_latency)
                update_latency_bucket(turn_bucket["infer_latency_sec"], infer_latency)
                update_latency_bucket(depth_bucket["infer_latency_sec"], infer_latency)
        else:
            summary["nodes"]["missing_rollout_candidates"] += 1
            candidates = []

        normalized_records = []
        if isinstance(records, list) and records:
            summary["nodes"]["with_rollout_records"] += 1

            selected_found = False
            node_has_any_success = False
            node_has_success_labels = False

            for idx, record in enumerate(records):
                infer_latency = safe_float(record.get("infer_latency_sec"))
                eval_latency = safe_float(record.get("eval_latency_sec"))
                total_latency = safe_float(record.get("total_latency_sec"))
                if total_latency is None and infer_latency is not None and eval_latency is not None:
                    total_latency = infer_latency + eval_latency
                if not candidate_infer_latencies:
                    update_latency_bucket(summary["latency_sec"]["infer_all"], infer_latency)
                    update_latency_bucket(turn_bucket["infer_latency_sec"], infer_latency)
                    update_latency_bucket(depth_bucket["infer_latency_sec"], infer_latency)

                success = bool_success(record, success_threshold)
                if success is not None:
                    node_has_success_labels = True
                if success is True:
                    node_has_any_success = True

                record_rollout_id = record.get("rollout_id", idx)
                if selected_rollout_id is not None and record_rollout_id == selected_rollout_id:
                    selected_found = True
                    update_latency_bucket(summary["latency_sec"]["selected_total"], total_latency)

                harness_result = record.get("harness_result")
                pass_rate = safe_float(record.get("pass_rate"))
                if pass_rate is None and isinstance(harness_result, list) and harness_result:
                    pass_rate = sum(harness_result) / float(len(harness_result))

                normalized = {
                    "rollout_id": record_rollout_id,
                    "success": success,
                    "infer_latency_sec": infer_latency,
                    "eval_latency_sec": eval_latency,
                    "total_latency_sec": total_latency,
                    "harness_result": harness_result if isinstance(harness_result, list) else [],
                    "pass_rate": pass_rate if pass_rate is not None else 0.0,
                }
                normalized_records.append(normalized)

                summary["rollouts"]["evaluated_total"] += 1
                if success is True:
                    summary["rollouts"]["successful_total"] += 1

                update_latency_bucket(summary["latency_sec"]["eval_all"], eval_latency)
                update_latency_bucket(summary["latency_sec"]["total_all"], total_latency)
                if success is True:
                    update_latency_bucket(summary["latency_sec"]["infer_success_only"], infer_latency)
                    update_latency_bucket(summary["latency_sec"]["eval_success_only"], eval_latency)
                    update_latency_bucket(summary["latency_sec"]["total_success_only"], total_latency)

            if node_has_success_labels:
                summary["nodes"]["with_success_labels"] += 1
            if node_has_any_success:
                summary["nodes"]["with_any_successful_rollout"] += 1
            if selected_rollout_id is not None:
                summary["nodes"]["with_selected_rollout_id"] += 1
                if not selected_found:
                    summary["nodes"]["selected_rollout_id_missing_record"] += 1

            successful_rollouts = sum(1 for record in normalized_records if record["success"] is True)
            node_rollouts = len(normalized_records)
            node_rollout_success_rate = (
                successful_rollouts / node_rollouts if node_rollouts else None
            )
            node_mean_infer_latency_sec = (
                sum(
                    record["infer_latency_sec"]
                    for record in normalized_records
                    if record["infer_latency_sec"] is not None
                )
                / sum(1 for record in normalized_records if record["infer_latency_sec"] is not None)
                if any(record["infer_latency_sec"] is not None for record in normalized_records)
                else None
            )
            node_mean_eval_latency_sec = (
                sum(
                    record["eval_latency_sec"]
                    for record in normalized_records
                    if record["eval_latency_sec"] is not None
                )
                / sum(1 for record in normalized_records if record["eval_latency_sec"] is not None)
                if any(record["eval_latency_sec"] is not None for record in normalized_records)
                else None
            )
            node_mean_total_latency_sec = (
                sum(
                    record["total_latency_sec"]
                    for record in normalized_records
                    if record["total_latency_sec"] is not None
                )
                / sum(1 for record in normalized_records if record["total_latency_sec"] is not None)
                if any(record["total_latency_sec"] is not None for record in normalized_records)
                else None
            )
            node_has_nonzero_success_rate = (
                node_rollout_success_rate > 0.0 if node_rollout_success_rate is not None else None
            )

            if node_rollout_success_rate is not None:
                summary["_node_rollout_success_rate_sum"] += node_rollout_success_rate
                summary["_node_rollout_success_rate_count"] += 1
            if node_has_nonzero_success_rate is False:
                summary["nodes"]["with_zero_success_rate"] += 1
                problem_has_zero_success_node = True

            update_group_bucket(
                turn_bucket,
                True,
                node_has_any_success,
                node_has_nonzero_success_rate,
                normalized_records,
            )
            update_group_bucket(
                depth_bucket,
                True,
                node_has_any_success,
                node_has_nonzero_success_rate,
                normalized_records,
            )
            per_node_details.append(
                {
                    "turn_index": turn_index,
                    "name": subproblem.get("name"),
                    "depth": depth,
                    "rollouts": node_rollouts,
                    "successful_rollouts": successful_rollouts,
                    "rollout_success_rate": node_rollout_success_rate,
                    "expected_infer_latency_sec": node_mean_infer_latency_sec,
                    "expected_eval_latency_sec": node_mean_eval_latency_sec,
                    "expected_total_latency_sec": node_mean_total_latency_sec,
                    "selected_rollout_id": selected_rollout_id,
                    "selected_success": next(
                        (
                            record["success"]
                            for record in normalized_records
                            if record["rollout_id"] == selected_rollout_id
                        ),
                        None,
                    ),
                }
            )
        else:
            summary["nodes"]["missing_rollout_records"] += 1
            problem_all_nodes_evaluated = False
            if selected_rollout_id is not None:
                summary["nodes"]["with_selected_rollout_id"] += 1
                summary["nodes"]["selected_rollout_id_missing_record"] += 1

            update_group_bucket(turn_bucket, False, False, None, [])
            update_group_bucket(depth_bucket, False, False, None, [])
            per_node_details.append(
                {
                    "turn_index": turn_index,
                    "name": subproblem.get("name"),
                    "depth": depth,
                    "rollouts": 0,
                    "successful_rollouts": 0,
                    "rollout_success_rate": None,
                    "expected_infer_latency_sec": None,
                    "expected_eval_latency_sec": None,
                    "expected_total_latency_sec": None,
                    "selected_rollout_id": selected_rollout_id,
                    "selected_success": None,
                }
            )

    if problem_all_nodes_evaluated:
        summary["dags"]["fully_evaluated"] += 1
        if not problem_has_zero_success_node:
            summary["dags"]["without_zero_success_node"] += 1

    summary["per_problem_node_success_rates"].append(
        {
            "problem_id": problem_id,
            "source_path": source_path,
            "node_count": len(subproblems),
            "fully_evaluated": problem_all_nodes_evaluated,
            "has_zero_success_node": problem_has_zero_success_node,
            "all_nodes_nonzero_success_rate": (
                problem_all_nodes_evaluated and (not problem_has_zero_success_node)
            ),
            "nodes": per_node_details,
        }
    )


def _build_persistable_problem(problem, keep_llm_outputs=False):
    if keep_llm_outputs:
        return problem

    persisted_problem = copy.deepcopy(problem)
    for subproblem in persisted_problem.get("subproblems", []):
        subproblem.pop("prompt", None)
        subproblem.pop("generated", None)
        subproblem.pop("original_output", None)

        rollout_candidates = subproblem.get("rollout_candidates")
        if isinstance(rollout_candidates, list):
            compact_candidates = []
            for idx, candidate in enumerate(rollout_candidates):
                if not isinstance(candidate, dict):
                    continue
                compact_candidates.append(
                    {
                        "rollout_id": candidate.get("rollout_id", idx),
                        "infer_latency_sec": safe_float(candidate.get("infer_latency_sec")),
                        "infer_batch_latency_sec": safe_float(candidate.get("infer_batch_latency_sec")),
                        "infer_per_rollout_latency_sec": safe_float(candidate.get("infer_per_rollout_latency_sec")),
                        "latency_observation": candidate.get("latency_observation"),
                    }
                )
            subproblem["rollout_candidates"] = compact_candidates

    return persisted_problem


def write_evaluated_problem(problem, save_dir, keep_llm_outputs=False):
    uuid = _problem_identity(problem) or f"problem_{id(problem)}"
    output_path = Path(save_dir) / f"{uuid}.json"
    _atomic_write_json(output_path, _build_persistable_problem(problem, keep_llm_outputs=keep_llm_outputs))


def write_back_problem_to_source(entry, keep_llm_outputs=False):
    source_path = Path(entry["source_path"])
    problem_index = entry["problem_index_in_file"]
    updated_problem = _build_persistable_problem(entry["problem"], keep_llm_outputs=keep_llm_outputs)
    lock_path = str(source_path) + ".lock"

    with _PathFileLock(lock_path):
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if problem_index != 0:
                raise IndexError(f"Problem index {problem_index} out of range for dict payload: {source_path}")
            data = updated_problem
        elif isinstance(data, list):
            if problem_index < 0 or problem_index >= len(data):
                raise IndexError(f"Problem index {problem_index} out of range for list payload: {source_path}")
            data[problem_index] = updated_problem
        else:
            raise ValueError(f"Unsupported JSON top level for write-back: {source_path}")

        _atomic_write_json(source_path, data)


def process_problem_entry(entry, args, progress=None):
    problem = entry["problem"]
    source_path = entry["source_path"]
    claim_info = _try_claim_problem_entry(entry, args)
    if not claim_info["acquired"]:
        return {
            "status": "skipped_claim",
            "entry": entry,
            "source_path": source_path,
            "problem_id": _problem_identity(problem),
            "claim_info": claim_info,
        }

    try:
        claim_ctx = (
            _ClaimHeartbeat(claim_info["heartbeat_path"], args.claim_heartbeat_sec)
            if claim_info.get("enabled")
            else nullcontext()
        )
        with claim_ctx:
            evaluation_meta = evaluate_problem_rollouts(problem, args, progress=progress)
            if args.save_evaluated_dir:
                write_evaluated_problem(problem, args.save_evaluated_dir, keep_llm_outputs=args.keep_llm_outputs)
            if args.write_back:
                write_back_problem_to_source(entry, keep_llm_outputs=args.keep_llm_outputs)
    except Exception as exc:
        _finalize_claim(
            claim_info,
            "error",
            {
                "worker_id": args.worker_id,
                "source_path": source_path,
                "problem_id": _problem_identity(problem),
                "error": f"evaluation_failed: {exc}",
            },
        )
        return {
            "status": "error",
            "entry": entry,
            "source_path": source_path,
            "error": f"evaluation_failed: {exc}",
            "problem_id": _problem_identity(problem),
            "claim_info": claim_info,
        }

    _finalize_claim(
        claim_info,
        "done",
        {
            "worker_id": args.worker_id,
            "source_path": source_path,
            "problem_id": _problem_identity(problem),
            "problem_evaluated": evaluation_meta["problem_evaluated"],
            "nodes_evaluated": evaluation_meta["nodes_evaluated"],
            "rollouts_evaluated": evaluation_meta["rollouts_evaluated"],
        },
    )
    return {
        "status": "processed",
        "entry": entry,
        "source_path": source_path,
        "problem_id": _problem_identity(problem),
        "evaluation_meta": evaluation_meta,
        "claim_info": claim_info,
    }


def main():
    args = parse_args()
    if not args.worker_id:
        args.worker_id = _default_worker_id()
    json_files = collect_json_files(args.input, args.recursive)
    problem_entries, files_with_problem_payload, files_skipped = load_problem_entries(json_files)
    full_inventory = inventory_problem_entries(problem_entries)
    selected_entries, sampling_meta = sample_problem_entries(
        problem_entries,
        args.sample_problems,
        args.sample_fraction,
        args.sample_seed,
    )
    selected_inventory = inventory_problem_entries(selected_entries)

    summary = {
        "input_path": str(Path(args.input).resolve()),
        "files_scanned": len(json_files),
        "files_with_problem_payload": files_with_problem_payload,
        "files_skipped": files_skipped,
        "problems": 0,
        "sampling": sampling_meta,
        "inventory": {
            "full_input": full_inventory,
            "selected_for_run": selected_inventory,
        },
        "estimate": {
            "assumed_eval_latency_sec_per_rollout": args.estimate_eval_latency_sec,
            "selected_rollouts_to_evaluate": selected_inventory["rollouts_to_evaluate"],
            "full_rollouts_to_evaluate": full_inventory["rollouts_to_evaluate"],
            "selected_eval_seconds_assumed": (
                selected_inventory["rollouts_to_evaluate"] * args.estimate_eval_latency_sec
            ),
            "selected_eval_duration_assumed": format_duration(
                selected_inventory["rollouts_to_evaluate"] * args.estimate_eval_latency_sec
            ),
            "full_eval_seconds_assumed": (
                full_inventory["rollouts_to_evaluate"] * args.estimate_eval_latency_sec
            ),
            "full_eval_duration_assumed": format_duration(
                full_inventory["rollouts_to_evaluate"] * args.estimate_eval_latency_sec
            ),
            "observed_eval_latency_sec_per_rollout": None,
            "selected_eval_seconds_observed": None,
            "selected_eval_duration_observed": None,
            "full_eval_seconds_extrapolated_from_sample": None,
            "full_eval_duration_extrapolated_from_sample": None,
        },
        "nodes": {
            "total": 0,
            "with_rollout_candidates": 0,
            "with_rollout_records": 0,
            "with_selected_rollout_id": 0,
            "with_zero_success_rate": 0,
            "with_any_successful_rollout": 0,
            "with_success_labels": 0,
            "missing_rollout_candidates": 0,
            "missing_rollout_records": 0,
            "selected_rollout_id_missing_record": 0,
        },
        "rollouts": {
            "candidate_total": 0,
            "evaluated_total": 0,
            "successful_total": 0,
            "success_rate": None,
        },
        "latency_sec": {
            "infer_all": init_latency_bucket(),
            "infer_success_only": init_latency_bucket(),
            "eval_all": init_latency_bucket(),
            "eval_success_only": init_latency_bucket(),
            "total_all": init_latency_bucket(),
            "total_success_only": init_latency_bucket(),
            "selected_total": init_latency_bucket(),
        },
        "node_level": {
            "mean_node_rollout_success_rate": None,
            "node_nonzero_success_rate": None,
            "any_success_rate": None,
            "mean_rollouts_per_node": None,
            "mean_candidates_per_node": None,
        },
        "dags": {
            "total": 0,
            "fully_evaluated": 0,
            "without_zero_success_node": 0,
            "without_zero_success_node_rate": None,
        },
        "evaluation": {
            "problems_evaluated_in_run": 0,
            "nodes_evaluated_in_run": 0,
            "rollouts_evaluated_in_run": 0,
        },
        "claiming": {
            "enabled": bool(args.claim_dir),
            "claim_dir": str(Path(args.claim_dir).resolve()) if args.claim_dir else "",
            "worker_id": args.worker_id,
            "claim_timeout_sec": args.claim_timeout_sec if args.claim_dir else None,
            "claim_heartbeat_sec": args.claim_heartbeat_sec if args.claim_dir else None,
            "problems_claimed_in_run": 0,
            "problems_skipped_done": 0,
            "problems_skipped_active": 0,
            "stale_claims_released": 0,
        },
        "coverage": {},
        "breakdown": {
            "turn_index": {},
            "depth": {},
        },
        "per_problem_node_success_rates": [],
        "_node_rollout_success_rate_sum": 0.0,
        "_node_rollout_success_rate_count": 0,
    }

    if args.estimate_only:
        del summary["_node_rollout_success_rate_sum"]
        del summary["_node_rollout_success_rate_count"]
        output_text = json.dumps(summary, ensure_ascii=False, indent=2)
        print(output_text)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text + "\n", encoding="utf-8")
            print(f"\nSaved summary to {output_path}")
        return

    progress = ProgressBar(
        total_problems=selected_inventory["problems"],
        total_nodes=selected_inventory["nodes_needing_eval"],
        total_rollouts=selected_inventory["rollouts_to_evaluate"],
        enabled=(not args.skip_eval) and (not args.claim_dir),
    )

    problem_workers = min(max(1, int(args.problem_workers)), max(1, len(selected_entries)))

    if problem_workers > 1:
        with ThreadPoolExecutor(max_workers=problem_workers) as executor:
            futures = [
                executor.submit(process_problem_entry, entry, args, progress)
                for entry in selected_entries
            ]
            for future in as_completed(futures):
                result = future.result()
                source_path = result["source_path"]
                summary["evaluation"]["current_source_path"] = source_path
                claim_info = result.get("claim_info") or {}
                if args.claim_dir and claim_info.get("released_stale_claim"):
                    summary["claiming"]["stale_claims_released"] += 1

                if result["status"] == "skipped_claim":
                    skip_reason = claim_info.get("reason")
                    if skip_reason == "done":
                        summary["claiming"]["problems_skipped_done"] += 1
                    else:
                        summary["claiming"]["problems_skipped_active"] += 1
                    continue

                if result["status"] == "error":
                    summary["files_skipped"].append(
                        {
                            "path": source_path,
                            "error": result["error"],
                            "problem_id": result["problem_id"],
                        }
                    )
                    progress.advance_problem(1)
                    continue

                entry = result["entry"]
                problem = entry["problem"]
                evaluation_meta = result["evaluation_meta"]
                if args.claim_dir:
                    summary["claiming"]["problems_claimed_in_run"] += 1

                if evaluation_meta["problem_evaluated"]:
                    summary["evaluation"]["problems_evaluated_in_run"] += 1
                summary["evaluation"]["nodes_evaluated_in_run"] += evaluation_meta["nodes_evaluated"]
                summary["evaluation"]["rollouts_evaluated_in_run"] += evaluation_meta["rollouts_evaluated"]

                analyze_problem(
                    problem,
                    summary,
                    args.success_threshold,
                    source_path=source_path,
                )
                progress.advance_problem(1)
    else:
        for entry in selected_entries:
            source_path = entry["source_path"]
            summary["evaluation"]["current_source_path"] = source_path

            result = process_problem_entry(entry, args, progress)
            claim_info = result.get("claim_info") or {}
            if args.claim_dir and claim_info.get("released_stale_claim"):
                summary["claiming"]["stale_claims_released"] += 1

            if result["status"] == "skipped_claim":
                skip_reason = claim_info.get("reason")
                if skip_reason == "done":
                    summary["claiming"]["problems_skipped_done"] += 1
                else:
                    summary["claiming"]["problems_skipped_active"] += 1
                continue

            if result["status"] == "error":
                summary["files_skipped"].append(
                    {
                        "path": source_path,
                        "error": result["error"],
                        "problem_id": result["problem_id"],
                    }
                )
                progress.advance_problem(1)
                continue

            problem = entry["problem"]
            evaluation_meta = result["evaluation_meta"]
            if args.claim_dir:
                summary["claiming"]["problems_claimed_in_run"] += 1

            if evaluation_meta["problem_evaluated"]:
                summary["evaluation"]["problems_evaluated_in_run"] += 1
            summary["evaluation"]["nodes_evaluated_in_run"] += evaluation_meta["nodes_evaluated"]
            summary["evaluation"]["rollouts_evaluated_in_run"] += evaluation_meta["rollouts_evaluated"]

            analyze_problem(
                problem,
                summary,
                args.success_threshold,
                source_path=source_path,
            )
            progress.advance_problem(1)

    evaluated_total = summary["rollouts"]["evaluated_total"]
    nodes_with_records = summary["nodes"]["with_rollout_records"]
    nodes_with_candidates = summary["nodes"]["with_rollout_candidates"]

    if evaluated_total:
        summary["rollouts"]["success_rate"] = (
            summary["rollouts"]["successful_total"] / evaluated_total
        )
    if nodes_with_records:
        summary["node_level"]["mean_node_rollout_success_rate"] = (
            summary["_node_rollout_success_rate_sum"] / summary["_node_rollout_success_rate_count"]
            if summary["_node_rollout_success_rate_count"]
            else None
        )
        summary["node_level"]["any_success_rate"] = (
            summary["nodes"]["with_any_successful_rollout"] / nodes_with_records
        )
        summary["node_level"]["mean_rollouts_per_node"] = (
            summary["rollouts"]["evaluated_total"] / nodes_with_records
        )
        summary["node_level"]["node_nonzero_success_rate"] = (
            (nodes_with_records - summary["nodes"]["with_zero_success_rate"]) / nodes_with_records
        )
    if nodes_with_candidates:
        summary["node_level"]["mean_candidates_per_node"] = (
            summary["rollouts"]["candidate_total"] / nodes_with_candidates
        )

    observed_eval_mean = summary["latency_sec"]["eval_all"]["sum"] / summary["latency_sec"]["eval_all"]["count"] if summary["latency_sec"]["eval_all"]["count"] else None
    if observed_eval_mean is not None:
        summary["estimate"]["observed_eval_latency_sec_per_rollout"] = observed_eval_mean
        summary["estimate"]["selected_eval_seconds_observed"] = (
            selected_inventory["rollouts_to_evaluate"] * observed_eval_mean
        )
        summary["estimate"]["selected_eval_duration_observed"] = format_duration(
            summary["estimate"]["selected_eval_seconds_observed"]
        )
        summary["estimate"]["full_eval_seconds_extrapolated_from_sample"] = (
            full_inventory["rollouts_to_evaluate"] * observed_eval_mean
        )
        summary["estimate"]["full_eval_duration_extrapolated_from_sample"] = format_duration(
            summary["estimate"]["full_eval_seconds_extrapolated_from_sample"]
        )

    node_total = summary["nodes"]["total"]
    summary["coverage"] = {
        "file_payload_rate": (
            summary["files_with_problem_payload"] / summary["files_scanned"]
            if summary["files_scanned"]
            else None
        ),
        "node_rollout_candidates_rate": (
            summary["nodes"]["with_rollout_candidates"] / node_total if node_total else None
        ),
        "node_rollout_records_rate": (
            summary["nodes"]["with_rollout_records"] / node_total if node_total else None
        ),
        "node_success_label_rate": (
            summary["nodes"]["with_success_labels"] / node_total if node_total else None
        ),
        "node_selected_rollout_rate": (
            summary["nodes"]["with_selected_rollout_id"] / node_total if node_total else None
        ),
    }
    if summary["dags"]["fully_evaluated"]:
        summary["dags"]["without_zero_success_node_rate"] = (
            summary["dags"]["without_zero_success_node"] / summary["dags"]["fully_evaluated"]
        )

    summary["latency_sec"] = {
        key: finalize_latency_bucket(bucket)
        for key, bucket in summary["latency_sec"].items()
    }
    summary["breakdown"]["turn_index"] = {
        key: finalize_group_bucket(bucket)
        for key, bucket in sorted(
            summary["breakdown"]["turn_index"].items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
        )
    }
    summary["breakdown"]["depth"] = {
        key: finalize_group_bucket(bucket)
        for key, bucket in sorted(
            summary["breakdown"]["depth"].items(),
            key=lambda item: (
                int(item[0]) if str(item[0]).isdigit() else 10**9,
                item[0],
            ),
        )
    }

    del summary["_node_rollout_success_rate_sum"]
    del summary["_node_rollout_success_rate_count"]
    summary["evaluation"].pop("current_source_path", None)
    progress.finish()

    output_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output_text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
        print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
