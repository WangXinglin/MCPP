"""
English note DAG English note(English note claim + English note).

English note:
  - English note DAG English note shard
  - English note DAG English note JSONL English note
  - English note worker English note,English note DAG English note
  - English note claim English note,English note worker English note DAG

English note:
  - English note DAG English note,English note
  - English note,English note DAG
  - English note worker English note,claim English note worker English note

English note:
  results/shards/raw_budget_1.0_deadline_ratio_1.0_all_dag42.jsonl
"""

import os
import sys
import json
import argparse
import logging
import time
import random
import socket
import hashlib
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.dag import DAG
from simulator.run_experiment import (
    _run_one_cell_v2,
    _run_one_cell_v2_chunk,
    get_method_names,
    write_record_jsonl_with_traces,
    _normalize_baseline_method_set,
)
from simulator.types import normalize_budget_cost_mode

logger = logging.getLogger(__name__)


def _build_mode_tag(only_loader: bool, baseline_only: bool, baseline_method_set: str = "all") -> str:
    if only_loader:
        return "loader_only"
    if baseline_only:
        baseline_method_set = _normalize_baseline_method_set(baseline_method_set)
        if baseline_method_set != "all":
            return f"baseline_{baseline_method_set}"
        return "baseline_only"
    return "all"


def _budget_cost_tag(budget_cost_mode: str) -> str:
    return f"budget_cost_{normalize_budget_cost_mode(budget_cost_mode)}"


def _budget_input_type(budget_ratio: float | None, budget_value: float | None) -> str:
    if budget_value is not None:
        return "value"
    if budget_ratio is not None:
        return "ratio"
    raise ValueError("Either --budget_ratio or --budget_value must be provided")


def _budget_axis_value(budget_ratio: float | None, budget_value: float | None) -> float:
    return float(budget_value) if budget_value is not None else float(budget_ratio)


def _budget_tag(budget_ratio: float | None, budget_value: float | None) -> str:
    mode = _budget_input_type(budget_ratio, budget_value)
    value = _budget_axis_value(budget_ratio, budget_value)
    return f"budget_{mode}_{value}"


def _raw_budget_stem(budget_ratio: float | None, budget_value: float | None) -> str:
    if budget_value is not None:
        return f"raw_budget_value_{float(budget_value)}"
    return f"raw_budget_{float(budget_ratio)}"


def load_dag_indices(path: str) -> list[int]:
    """Load a selected-DAG index file across historical JSON shapes."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_indices = data
    elif isinstance(data, dict):
        raw_indices = None
        for key in ("dag_indices", "selected_dag_indices", "selected_indices", "indices"):
            if key in data:
                raw_indices = data[key]
                break
        if raw_indices is None:
            raise KeyError(
                "DAG indices file must contain one of keys: "
                "dag_indices, selected_dag_indices, selected_indices, indices. "
                f"Found keys: {sorted(data.keys())}"
            )
    else:
        raise TypeError(
            f"DAG indices file must be a JSON object or list, got {type(data).__name__}"
        )

    if not isinstance(raw_indices, list):
        raise TypeError(f"DAG indices must be a list, got {type(raw_indices).__name__}")
    return [int(idx) for idx in raw_indices]


def _dag_output_name(
    budget_ratio: float | None,
    deadline_tag: str,
    mode_tag: str,
    dag_idx: int,
    budget_cost_mode: str,
    budget_value: float | None = None,
) -> str:
    return (
        f"{_raw_budget_stem(budget_ratio, budget_value)}_{deadline_tag}_{_budget_cost_tag(budget_cost_mode)}_"
        f"{mode_tag}_dag{dag_idx}.jsonl"
    )


def _claim_name(
    budget_ratio: float | None,
    deadline_tag: str,
    mode_tag: str,
    dag_idx: int,
    budget_cost_mode: str,
    budget_value: float | None = None,
) -> str:
    return (
        f"claim_{_budget_tag(budget_ratio, budget_value)}_{deadline_tag}_{_budget_cost_tag(budget_cost_mode)}_"
        f"{mode_tag}_dag{dag_idx}.json"
    )


def _json_dump_atomic(path: str, payload: dict) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _read_json_file(path: str) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-pid{os.getpid()}"


def _derive_shuffle_seed(
    seed: int,
    budget_ratio: float | None,
    deadline_tag: str,
    mode_tag: str,
    budget_cost_mode: str,
    worker_id: str,
    budget_value: float | None = None,
) -> int:
    material = f"{seed}|{_budget_tag(budget_ratio, budget_value)}|{deadline_tag}|{mode_tag}|{budget_cost_mode}|{worker_id}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _claim_path(
    claim_dir: str,
    budget_ratio: float | None,
    deadline_tag: str,
    mode_tag: str,
    dag_idx: int,
    budget_cost_mode: str,
    budget_value: float | None = None,
) -> str:
    return os.path.join(
        claim_dir,
        _claim_name(budget_ratio, deadline_tag, mode_tag, dag_idx, budget_cost_mode, budget_value),
    )


def _try_acquire_claim(
    claim_path: str,
    worker_id: str,
    claim_timeout_sec: float,
    claim_metadata: dict,
) -> bool:
    now = time.time()
    payload = {
        "worker_id": worker_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "claimed_at": now,
        "heartbeat_at": now,
        "metadata": claim_metadata,
    }

    for _ in range(2):
        try:
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            return True
        except FileExistsError:
            existing = _read_json_file(claim_path) or {}
            heartbeat_at = float(existing.get("heartbeat_at", existing.get("claimed_at", 0.0)) or 0.0)
            if heartbeat_at > 0 and (now - heartbeat_at) <= claim_timeout_sec:
                return False
            try:
                os.remove(claim_path)
                logger.warning("Removed stale claim %s (age=%.1fs)", claim_path, now - heartbeat_at)
            except FileNotFoundError:
                pass
            except OSError:
                return False

    return False


def _release_claim(claim_path: str, worker_id: str) -> None:
    payload = _read_json_file(claim_path)
    if payload is None:
        return
    if payload.get("worker_id") != worker_id:
        return
    try:
        os.remove(claim_path)
    except FileNotFoundError:
        pass


class _ClaimHeartbeat:
    def __init__(self, claim_path: str, worker_id: str, interval_sec: float):
        self.claim_path = claim_path
        self.worker_id = worker_id
        self.interval_sec = max(1.0, float(interval_sec))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_sec + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            payload = _read_json_file(self.claim_path)
            if payload is None:
                return
            if payload.get("worker_id") != self.worker_id:
                return
            payload["heartbeat_at"] = time.time()
            try:
                _json_dump_atomic(self.claim_path, payload)
            except OSError:
                logger.warning("Failed to refresh claim heartbeat: %s", self.claim_path)
                return


def _detect_available_cpus() -> int:
    """
    English note CPU English note.

    English note(English note):
      1. LOADER_AUTO_CPU_CORES override
      2. cgroup CPU quota
      3. cpuset / CPU affinity
      4. os.cpu_count()

    LOADER_AUTO_CPU_POLICY controls how quota and affinity are combined:
      effective (default): min(quota, affinity) when both are available
      quota: prefer quota
      affinity/logical: prefer affinity / cpuset
      max: use the largest detected signal
    """
    override = os.environ.get("LOADER_AUTO_CPU_CORES", "").strip()
    if override:
        try:
            cpus = max(1, int(override))
            logger.info("Using LOADER_AUTO_CPU_CORES override: %d cores", cpus)
            return cpus
        except ValueError:
            logger.warning("Ignoring invalid LOADER_AUTO_CPU_CORES=%r", override)

    quota_cpus = None
    cpuset_cpus = None
    affinity_cpus = None
    online_cpus = None

    # ── cgroup quota: v2 then v1 ──
    try:
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            parts = f.read().strip().split()
        if parts[0] != "max":
            quota = int(parts[0])
            period = int(parts[1])
            quota_cpus = max(1, (quota + period - 1) // period)
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass

    if quota_cpus is None:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
                quota = int(f.read().strip())
            if quota > 0:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                    period = int(f.read().strip())
                quota_cpus = max(1, (quota + period - 1) // period)
        except (FileNotFoundError, PermissionError, ValueError):
            pass

    def count_cpu_list(text: str) -> int | None:
        total = 0
        for part in text.strip().split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                total += max(0, int(hi) - int(lo) + 1)
            else:
                int(part)
                total += 1
        return total if total > 0 else None

    for path in (
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset.cpus",
        "/sys/fs/cgroup/cpuset/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
    ):
        try:
            with open(path, "r") as f:
                cpuset_cpus = count_cpu_list(f.read())
            if cpuset_cpus:
                break
        except (FileNotFoundError, PermissionError, ValueError):
            pass

    try:
        affinity_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass

    online_cpus = os.cpu_count() or 1
    affinity_cap = cpuset_cpus or affinity_cpus or online_cpus
    policy = os.environ.get("LOADER_AUTO_CPU_POLICY", "effective").strip().lower()

    if policy == "quota":
        cpus = quota_cpus or affinity_cap or 1
    elif policy in {"affinity", "logical"}:
        cpus = affinity_cap or quota_cpus or 1
    elif policy == "max":
        cpus = max(value for value in (quota_cpus, affinity_cap, online_cpus, 1) if value)
    else:
        if quota_cpus and affinity_cap:
            cpus = min(quota_cpus, affinity_cap)
        else:
            cpus = quota_cpus or affinity_cap or 1

    logger.info(
        "Detected CPU capacity: quota=%s, cpuset=%s, affinity=%s, online=%s, policy=%s -> %d workers",
        quota_cpus if quota_cpus is not None else "none",
        cpuset_cpus if cpuset_cpus is not None else "none",
        affinity_cpus if affinity_cpus is not None else "none",
        online_cpus,
        policy,
        cpus,
    )
    return max(1, int(cpus))


def scan_completed_dags(output_path: str, n_exec_trials: int, n_methods: int) -> set:
    """
    English note,English note dag_idx.

    English note dag_idx "English note" = English note n_exec_trials x n_methods English note.
    English note(English note kill),English note dag_idx English note,
    English note(English note),English note.
    """
    if not os.path.exists(output_path):
        return set()

    expected_per_dag = n_exec_trials * n_methods
    dag_counts = defaultdict(int)

    # English note dag_idx English note
    n_lines = 0
    n_bad = 0
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
                dag_idx = rec["dag_idx"]
                dag_counts[dag_idx] += 1
            except (json.JSONDecodeError, KeyError):
                n_bad += 1

    # English note / English note dag_idx
    completed = set()
    incomplete = set()
    for dag_idx, count in dag_counts.items():
        if count == expected_per_dag:
            completed.add(dag_idx)
        else:
            incomplete.add(dag_idx)

    if incomplete:
        # English note dag_idx → English note,English note
        logger.info(
            "Found %d completed DAGs, %d incomplete DAGs in %s. "
            "Cleaning incomplete records ...",
            len(completed), len(incomplete), output_path,
        )
        _clean_incomplete(output_path, completed)

    if n_bad > 0:
        logger.warning("Skipped %d malformed lines in %s", n_bad, output_path)

    return completed


def _clean_incomplete(output_path: str, completed_dags: set):
    """
    English note,English note dag_idx English note.

    English note kill English note dag_idx English note.
    """
    tmp_path = output_path + ".tmp"
    n_kept = 0
    n_dropped = 0

    with open(output_path, "r") as f_in, open(tmp_path, "w") as f_out:
        for line in f_in:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                rec = json.loads(line_stripped)
                if rec["dag_idx"] in completed_dags:
                    f_out.write(line_stripped + "\n")
                    n_kept += 1
                else:
                    n_dropped += 1
            except (json.JSONDecodeError, KeyError):
                n_dropped += 1

    # English note
    os.replace(tmp_path, output_path)
    logger.info("Cleaned: kept %d records, dropped %d incomplete records", n_kept, n_dropped)


def run_shard(
    dag_pool_path: str,
    budget_ratio: float | None,
    dag_start: int,
    dag_end: int,
    n_exec_trials: int,
    seed: int,
    latency_dist: str,
    n_workers: int,
    output_dir: str,
    deadline_tag: str,
    force_restart: bool = False,
    dag_indices: list = None,
    budget_value: float | None = None,
    deadline_value: float | None = None,
    deadline_ratio: float = 1.0,
    budget_cost_mode: str = "token_cost_mean",
    only_loader: bool = False,
    baseline_only: bool = False,
    baseline_method_set: str = "all",
    claim_dir: str | None = None,
    worker_id: str | None = None,
    claim_timeout_sec: float = 1800.0,
    claim_heartbeat_sec: float = 30.0,
):
    """
    English note:English note budget ratio/value English note DAG English note DAG English note.

    dag_indices English note:
      English note dag_indices English note None English note,English note DAG(English note dag_start/dag_end).
      DAG English note dag_pool_path English note,English note dag_idx English note seed English note.

    English note:English note DAG English note,English note.
    """
    # English note DAG English note
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)
    baseline_method_set = _normalize_baseline_method_set(baseline_method_set)
    budget_mode = _budget_input_type(budget_ratio, budget_value)
    budget_axis = _budget_axis_value(budget_ratio, budget_value)
    budget_label = _budget_tag(budget_ratio, budget_value)
    dag_pool = DAG.load_pool_jsonl(dag_pool_path)
    n_total = len(dag_pool)
    mode_tag = _build_mode_tag(
        only_loader=only_loader,
        baseline_only=baseline_only,
        baseline_method_set=baseline_method_set,
    )
    worker_id = worker_id or _default_worker_id()
    claim_dir = claim_dir or os.path.join(output_dir, "_claims")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(claim_dir, exist_ok=True)

    # English note (dag_idx, full_dict) English note
    if dag_indices is not None:
        # English note:English note DAG
        dag_items = []
        for idx in dag_indices:
            if 0 <= idx < n_total:
                dag_items.append((idx, dag_pool[idx]))
            else:
                logger.warning("dag_idx=%d out of range [0, %d), skipping.", idx, n_total)
        n_dags = len(dag_items)
        desc_range = f"{n_dags} selected DAGs"
    else:
        # English note:[dag_start, dag_end)
        dag_end = min(dag_end, n_total)
        if dag_start >= dag_end:
            logger.warning("dag_start=%d >= dag_end=%d, nothing to do.", dag_start, dag_end)
            return 0
        dag_items = [(dag_start + i, dag_pool[dag_start + i])
                     for i in range(dag_end - dag_start)]
        n_dags = len(dag_items)
        desc_range = f"DAGs [{dag_start}, {dag_end})"

    n_methods = len(get_method_names(
        only_loader=only_loader,
        baseline_only=baseline_only,
        baseline_method_set=baseline_method_set,
    ))

    if n_workers is None or n_workers <= 0:
        n_workers = _detect_available_cpus()
    if n_exec_trials > 0 and n_workers > n_exec_trials:
        logger.info(
            "Capping shard worker count from %d to n_exec_trials=%d; "
            "a single DAG cannot create more execution-trial chunks than trials.",
            n_workers,
            n_exec_trials,
        )
        n_workers = n_exec_trials

    shuffle_seed = _derive_shuffle_seed(
        seed=seed,
        budget_ratio=budget_ratio,
        budget_value=budget_value,
        deadline_tag=deadline_tag,
        mode_tag=mode_tag,
        budget_cost_mode=budget_cost_mode,
        worker_id=worker_id,
    )
    rng = random.Random(shuffle_seed)
    rng.shuffle(dag_items)

    remaining_sims = n_dags * n_exec_trials * n_methods
    logger.info(
        "Atomic DAG mode: %s, %s (%d DAGs), "
        "exec_trials=%d, max_sims=%d, n_workers=%d, worker_id=%s, shuffle_seed=%d, "
        "only_loader=%s, baseline_only=%s, baseline_method_set=%s, budget_cost_mode=%s",
        budget_label, desc_range, n_dags,
        n_exec_trials, remaining_sims, n_workers, worker_id, shuffle_seed,
        only_loader, baseline_only, baseline_method_set, budget_cost_mode,
    )

    pending_dags = []
    completed_before = 0
    for dag_idx, full_dict in dag_items:
        dag_output_path = os.path.join(
            output_dir,
            _dag_output_name(budget_ratio, deadline_tag, mode_tag, dag_idx, budget_cost_mode, budget_value),
        )
        if force_restart and os.path.exists(dag_output_path):
            logger.info("--restart flag: removing existing %s", dag_output_path)
            os.remove(dag_output_path)

        completed = scan_completed_dags(dag_output_path, n_exec_trials, n_methods=n_methods)
        if dag_idx in completed:
            completed_before += 1
            continue

        dag_kind = full_dict.get("dag_kind", "unknown")
        n_nodes = full_dict.get("n_nodes", len(full_dict.get("nodes", [])))
        exec_seeds = [
            seed + dag_idx * 10007 + exec_trial * 7 + 1
            for exec_trial in range(n_exec_trials)
        ]
        pending_dags.append((dag_idx, dag_kind, n_nodes, full_dict, exec_seeds, dag_output_path))

    if completed_before > 0:
        logger.info(
            "Resuming: %d/%d DAGs already completed, %d candidates remain",
            completed_before, n_dags, len(pending_dags),
        )
    if not pending_dags:
        logger.info("All DAGs in this selection already completed. Nothing to do.")
        return 0

    n_cells = len(pending_dags)
    n_written = 0
    n_claimed = 0
    n_busy = 0

    pbar = tqdm(
        total=n_cells,
        desc=f"{budget_label} atomic-dag",
        unit="dag",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} dags "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
        file=sys.stderr,
        dynamic_ncols=True,
    )

    chunk_size = max(1, n_exec_trials // n_workers) if n_workers > 1 else n_exec_trials

    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for dag_i, (dag_idx, dag_kind, n_nodes, full_dict, exec_seeds, dag_output_path) in enumerate(pending_dags):
            claim_path = _claim_path(
                claim_dir,
                budget_ratio,
                deadline_tag,
                mode_tag,
                dag_idx,
                budget_cost_mode,
                budget_value,
            )
            claim_metadata = {
                "dag_idx": dag_idx,
                "budget_input_type": budget_mode,
                "budget_axis_value": budget_axis,
                "budget_ratio": budget_ratio,
                "budget_value": budget_value,
                "deadline_tag": deadline_tag,
                "mode_tag": mode_tag,
                "output_path": dag_output_path,
            }

            if not _try_acquire_claim(
                claim_path=claim_path,
                worker_id=worker_id,
                claim_timeout_sec=claim_timeout_sec,
                claim_metadata=claim_metadata,
            ):
                n_busy += 1
                pbar.set_postfix_str(
                    f"busy={n_busy} claimed={n_claimed} written={n_written}",
                    refresh=False,
                )
                pbar.update(1)
                continue

            n_claimed += 1
            heartbeat = _ClaimHeartbeat(claim_path, worker_id, claim_heartbeat_sec)
            heartbeat.start()
            try:
                completed = scan_completed_dags(dag_output_path, n_exec_trials, n_methods=n_methods)
                if dag_idx in completed:
                    pbar.set_postfix_str(
                        f"dag_idx={dag_idx} already_done claimed={n_claimed}",
                        refresh=False,
                    )
                    pbar.update(1)
                    continue

                if pool is not None:
                    dag_records = []
                    chunk_futures = []
                    for c in range(0, n_exec_trials, chunk_size):
                        c_end = min(c + chunk_size, n_exec_trials)
                        chunk_seeds = exec_seeds[c:c_end]
                        chunk_args = (
                            dag_idx, dag_kind, n_nodes, full_dict,
                            budget_ratio, chunk_seeds, latency_dist,
                            c,
                            deadline_value, deadline_ratio,
                            only_loader,
                            baseline_only,
                            budget_cost_mode,
                            budget_value,
                            baseline_method_set,
                        )
                        chunk_futures.append(pool.submit(_run_one_cell_v2_chunk, chunk_args))

                    for future in as_completed(chunk_futures):
                        dag_records.extend(future.result())
                else:
                    args = (
                        dag_idx, dag_kind, n_nodes, full_dict,
                        budget_ratio, exec_seeds, latency_dist,
                        deadline_value, deadline_ratio, only_loader, baseline_only,
                        budget_cost_mode, budget_value, baseline_method_set,
                    )
                    dag_records = _run_one_cell_v2(args)

                with open(dag_output_path, "w") as f_out:
                    for rec in dag_records:
                        write_record_jsonl_with_traces(f_out, rec, output_dir)
                    f_out.flush()

                n_written += len(dag_records)
                logger.info(
                    "DAG %d/%d (dag_idx=%d) done: %d records → %s",
                    dag_i + 1, n_cells, dag_idx, len(dag_records), dag_output_path,
                )
                pbar.set_postfix_str(
                    f"dag_idx={dag_idx} claimed={n_claimed} busy={n_busy} written={n_written}",
                    refresh=False,
                )
                pbar.update(1)
            finally:
                heartbeat.stop()
                _release_claim(claim_path, worker_id)
    finally:
        if pool is not None:
            pool.shutdown()

    pbar.close()

    logger.info(
        "Atomic DAG run done. %d new records written, %d DAGs claimed, %d DAGs busy, "
        "%d DAGs were already complete → %s",
        n_written, n_claimed, n_busy, completed_before, output_dir,
    )
    return n_written


def _format_deadline_tag(deadline_value: float | None, deadline_ratio: float) -> str:
    if deadline_value is not None:
        return f"deadline_value_{deadline_value}"
    return f"deadline_ratio_{deadline_ratio}"


def main():
    parser = argparse.ArgumentParser(
        description="Run atomic DAG simulation tasks with shuffled ordering and claim-based exclusion"
    )
    parser.add_argument("--dag_pool_path", type=str, required=True,
                        help="Path to dag_pool.jsonl")
    parser.add_argument("--budget_ratio", type=float, default=None,
                        help="Relative budget ratio for this shard")
    parser.add_argument("--budget_value", type=float, default=None,
                        help="Absolute work budget for this shard; when set, no nominal-budget scaling is used")
    parser.add_argument("--dag_start", type=int, default=0,
                        help="Start DAG index (inclusive, ignored if --dag_indices_file is set)")
    parser.add_argument("--dag_end", type=int, default=0,
                        help="End DAG index (exclusive, ignored if --dag_indices_file is set)")
    parser.add_argument("--dag_indices_file", type=str, default=None,
                        help="JSON file with selected DAG indices "
                             "(overrides --dag_start/--dag_end). "
                             "File should have a 'dag_indices' array.")
    parser.add_argument("--dag_shard_start", type=int, default=None,
                        help="When using --dag_indices_file, process indices "
                             "[dag_shard_start:dag_shard_end) within the list (for sharding)")
    parser.add_argument("--dag_shard_end", type=int, default=None,
                        help="When using --dag_indices_file, process indices up to this position")
    parser.add_argument("--n_exec_trials", type=int, default=1000,
                        help="Number of execution trials per DAG (default: 1000)")
    parser.add_argument("--deadline_value", type=float, default=None,
                        help="Absolute deadline for this shard (default: None)")
    parser.add_argument("--deadline_ratio", type=float, default=1.0,
                        help="Relative deadline ratio against nominal deadline (default: 1.0)")
    parser.add_argument(
        "--budget_cost_mode",
        type=str,
        default="token_cost_mean",
        choices=["token_cost_mean", "output_money_cost_mean", "cost_mean", "lat_mean"],
        help="Work-budget kappa_v type: output tokens, output dollars, cost, or latency mean.",
    )
    parser.add_argument("--only_loader", action="store_true",
                        help="Run only the primary deadline-aware method for focused debugging")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Run only baseline methods and exclude the deadline-aware method")
    parser.add_argument(
        "--baseline_method_set",
        type=str,
        default="all",
        choices=["all", "seq", "static"],
        help="Subset of baseline methods to run when baseline methods are enabled.",
    )
    parser.add_argument("--output_dir", type=str, default="results/shards",
                        help="Output directory for shard files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--latency_dist", type=str, default="lognormal",
                        help="Latency distribution (default: lognormal)")
    parser.add_argument("--n_workers", type=int, default=0,
                        help="Number of parallel workers (0=auto, default: 0)")
    parser.add_argument("--restart", action="store_true",
                        help="Force restart: discard existing progress and start from scratch")
    parser.add_argument("--claim_dir", type=str, default=None,
                        help="Directory for DAG claim files (default: <output_dir>/_claims)")
    parser.add_argument("--worker_id", type=str, default=None,
                        help="Stable worker identifier used for shuffle order and claim ownership")
    parser.add_argument("--claim_timeout_sec", type=float, default=1800.0,
                        help="Claim timeout in seconds before a stale DAG can be stolen")
    parser.add_argument("--claim_heartbeat_sec", type=float, default=30.0,
                        help="Heartbeat interval in seconds for refreshing active claims")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    dag_indices = None

    budget_cost_mode = normalize_budget_cost_mode(args.budget_cost_mode)
    baseline_method_set = _normalize_baseline_method_set(args.baseline_method_set)
    budget_input_type = _budget_input_type(args.budget_ratio, args.budget_value)
    budget_axis = _budget_axis_value(args.budget_ratio, args.budget_value)
    budget_tag = _budget_tag(args.budget_ratio, args.budget_value)
    deadline_tag = _format_deadline_tag(args.deadline_value, args.deadline_ratio)
    budget_cost_tag = _budget_cost_tag(budget_cost_mode)
    mode_tag = _build_mode_tag(
        only_loader=args.only_loader,
        baseline_only=args.baseline_only,
        baseline_method_set=baseline_method_set,
    )
    method_names = get_method_names(
        only_loader=args.only_loader,
        baseline_only=args.baseline_only,
        baseline_method_set=baseline_method_set,
    )

    if args.dag_indices_file:
        # ── English note ──
        all_indices = load_dag_indices(args.dag_indices_file)

        # English note(English note)
        shard_start = args.dag_shard_start if args.dag_shard_start is not None else 0
        shard_end = args.dag_shard_end if args.dag_shard_end is not None else len(all_indices)
        dag_indices = all_indices[shard_start:shard_end]

        meta_name = (
            f"atomic_meta_{budget_tag}_{deadline_tag}_{budget_cost_tag}_"
            f"{mode_tag}_sel{shard_start}-{shard_end - 1}.json"
        )

        logger.info(
            "Index-list mode: %d DAGs from %s [%d:%d]",
            len(dag_indices), args.dag_indices_file, shard_start, shard_end,
        )
    else:
        # ── English note:English note DAG English note,English note DAG English note ──
        meta_name = (
            f"atomic_meta_{budget_tag}_{deadline_tag}_{budget_cost_tag}_"
            f"{mode_tag}_dag{args.dag_start}-{args.dag_end - 1}.json"
        )

    # English note
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, meta_name)
    meta_payload = dict(vars(args))
    meta_payload["deadline_tag"] = deadline_tag
    meta_payload["budget_input_type"] = budget_input_type
    meta_payload["budget_axis_value"] = budget_axis
    meta_payload["budget_tag"] = budget_tag
    meta_payload["mode_tag"] = mode_tag
    meta_payload["budget_cost_mode"] = budget_cost_mode
    meta_payload["baseline_method_set"] = baseline_method_set
    meta_payload["method_names"] = method_names
    meta_payload["n_methods"] = len(method_names)
    meta_payload["atomic_dag_outputs"] = True
    meta_payload["claim_dir_effective"] = args.claim_dir or os.path.join(args.output_dir, "_claims")
    with open(meta_path, "w") as f:
        json.dump(meta_payload, f, indent=2, default=str)

    t0 = time.time()
    n_written = run_shard(
        dag_pool_path=args.dag_pool_path,
        budget_ratio=args.budget_ratio,
        budget_value=args.budget_value,
        dag_start=args.dag_start,
        dag_end=args.dag_end,
        n_exec_trials=args.n_exec_trials,
        seed=args.seed,
        latency_dist=args.latency_dist,
        n_workers=args.n_workers if args.n_workers > 0 else None,
        output_dir=args.output_dir,
        deadline_tag=deadline_tag,
        force_restart=args.restart,
        dag_indices=dag_indices,
        deadline_value=args.deadline_value,
        deadline_ratio=args.deadline_ratio,
        budget_cost_mode=budget_cost_mode,
        only_loader=args.only_loader,
        baseline_only=args.baseline_only,
        baseline_method_set=baseline_method_set,
        claim_dir=args.claim_dir,
        worker_id=args.worker_id,
        claim_timeout_sec=args.claim_timeout_sec,
        claim_heartbeat_sec=args.claim_heartbeat_sec,
    )
    elapsed = time.time() - t0

    print(f"\nAtomic DAG run complete: {n_written} new records in {elapsed:.1f}s → {args.output_dir}")


if __name__ == "__main__":
    main()
