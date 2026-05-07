import os
import sys
import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm

import numpy as np
import pandas as pd
import math
import time

from .task_library import TaskLibrary
from .dag import DAG, DAGGenerator
from .simulator import Simulator, SimulationResult
from .types import DEFAULT_MODEL_ID, expected_node_work_cost, normalize_budget_cost_mode, pricing_model_id_for_node
from .LOADER_policies import (
    Policy,
    SequentialPolicy,
    UniformPolicy,
    RandomPolicy,
    LatencyWeightedPolicy,
    SuccessRateWeightedPolicy,
    CriticalPathStaticPolicy,
)
from .mc_portfolio_policy import MCPortfolioRolloutPolicy

logger = logging.getLogger(__name__)


BASELINE_METHOD_ORDER = (
    "sequential",
    "sequential_2",
    "sequential_4",
    "sequential_8",
    "sequential_16",
    "sequential_32",
    "sequential_64",
    "uniform",
    "random",
    "latency_weighted",
    "success_rate_weighted",
    "critical_path_static",
)
PRIMARY_DEADLINE_METHOD = "mc_portfolio_rollout"
METHOD_ORDER = BASELINE_METHOD_ORDER + (PRIMARY_DEADLINE_METHOD,)
SEQ_BASELINE_METHODS = {
    "sequential",
    "sequential_2",
    "sequential_4",
    "sequential_8",
    "sequential_16",
    "sequential_32",
    "sequential_64",
}
STATIC_BASELINE_METHODS = set(BASELINE_METHOD_ORDER) - SEQ_BASELINE_METHODS
BASELINE_METHOD_SET_CHOICES = ("all", "seq", "static")
DEADLINE_TRACE_RECORD_KEY = "_deadline_trace_events"


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _deadline_trace_level() -> str:
    level = os.environ.get("DEADLINE_TRACE_LEVEL", "theta").strip().lower()
    if level not in {"summary", "theta", "theta_full", "iter"}:
        return "theta"
    return level


def _deadline_theta_grid_size() -> int:
    raw = os.environ.get("DEADLINE_THETA_GRID_SIZE", "10")
    try:
        return max(2, int(raw))
    except ValueError as exc:
        raise ValueError(f"Invalid DEADLINE_THETA_GRID_SIZE={raw!r}; expected an integer >= 2") from exc


def _deadline_theta_grid_spacing() -> str:
    return os.environ.get("DEADLINE_THETA_GRID_SPACING", "log_upper").strip().lower()


def _deadline_budget_reservation_mode() -> str:
    return os.environ.get("DEADLINE_BUDGET_RESERVATION_MODE", "hard").strip().lower()


def _deadline_budget_reservation_c() -> float:
    raw = os.environ.get("DEADLINE_BUDGET_RESERVATION_C", "2.0")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid DEADLINE_BUDGET_RESERVATION_C={raw!r}; expected a float >= 0") from exc
    if value < 0.0:
        raise ValueError(f"Invalid DEADLINE_BUDGET_RESERVATION_C={raw!r}; expected a float >= 0")
    return value


def _deadline_budget_reservation_alpha() -> float:
    raw = os.environ.get("DEADLINE_BUDGET_RESERVATION_ALPHA", "0.0")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid DEADLINE_BUDGET_RESERVATION_ALPHA={raw!r}; expected a float in [0, 1]"
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Invalid DEADLINE_BUDGET_RESERVATION_ALPHA={raw!r}; expected a float in [0, 1]"
        )
    return value


def _safe_trace_component(value) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _deadline_trace_id(
    dag_idx: int,
    exec_idx: int,
    budget_ratio: float,
    deadline_axis_value: float,
    budget_cost_mode: str,
) -> str:
    return (
        f"dag{_safe_trace_component(dag_idx)}"
        f"_trial{_safe_trace_component(exec_idx)}"
        f"_br{_safe_trace_component(budget_ratio)}"
        f"_deadline{_safe_trace_component(deadline_axis_value)}"
        f"_budgetcost{_safe_trace_component(budget_cost_mode)}"
    )


def resolve_budget_value(
    budget_nominal: float,
    budget_ratio: float | None = None,
    budget_value: float | None = None,
) -> tuple[float, str, float, float | None, float | None]:
    """
    Resolve the work budget used by simulation.

    Ratio mode keeps the historical behavior:
        budget = budget_ratio * budget_nominal

    Value mode is absolute:
        budget = budget_value

    Returns:
        budget, input_type, axis_value, ratio_requested, value_requested
    """
    if budget_value is not None:
        value = float(budget_value)
        return value, "value", value, None, value

    if budget_ratio is None:
        raise ValueError("Either budget_ratio or budget_value must be provided")
    ratio = float(budget_ratio)
    return ratio * float(budget_nominal), "ratio", ratio, ratio, None


def resolve_deadline_value(
    deadline_nominal: float,
    deadline_value: float | None = None,
    deadline_ratio: float = 1.0,
) -> tuple[float, str, float, float | None, float | None]:
    """
    Resolve the wall-clock deadline used by simulation.

    Value mode is already supported and is absolute. Ratio mode keeps the
    historical nominal-deadline scaling.
    """
    if deadline_value is not None:
        value = float(deadline_value)
        return value, "value", value, None, value

    ratio = float(deadline_ratio)
    return ratio * float(deadline_nominal), "ratio", ratio, ratio, None


def write_record_jsonl_with_traces(f_out, record: dict, output_dir: str) -> None:
    rec = dict(record)
    trace_events = rec.pop(DEADLINE_TRACE_RECORD_KEY, None)
    if trace_events:
        trace_dir = os.path.join(output_dir, "traces")
        os.makedirs(trace_dir, exist_ok=True)
        trace_id = rec.get("deadline_trace_id") or trace_events[0].get("trace_id") or "deadline_trace"
        trace_name = f"{_safe_trace_component(trace_id)}.jsonl"
        trace_path = os.path.join(trace_dir, trace_name)
        with open(trace_path, "a", encoding="utf-8") as f_trace:
            for event in trace_events:
                f_trace.write(json.dumps(event, ensure_ascii=False) + "\n")
        rec["deadline_trace_path"] = os.path.join("traces", trace_name)
    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _normalize_baseline_method_set(baseline_method_set: str = "all") -> str:
    baseline_method_set = (baseline_method_set or "all").strip().lower()
    aliases = {
        "all": "all",
        "baseline": "all",
        "baselines": "all",
        "seq": "seq",
        "seq_only": "seq",
        "sequential": "seq",
        "sequential_only": "seq",
        "static": "static",
        "static_only": "static",
        "nonseq": "static",
        "non_seq": "static",
    }
    normalized = aliases.get(baseline_method_set)
    if normalized is None:
        raise ValueError(
            f"Unsupported baseline_method_set={baseline_method_set!r}; "
            f"expected one of {BASELINE_METHOD_SET_CHOICES}"
        )
    return normalized


def _filter_baseline_methods(methods, baseline_method_set: str = "all"):
    baseline_method_set = _normalize_baseline_method_set(baseline_method_set)
    if baseline_method_set == "all":
        return list(methods)
    allowed = SEQ_BASELINE_METHODS if baseline_method_set == "seq" else STATIC_BASELINE_METHODS
    return [method for method in methods if method in allowed]


def get_method_names(
    only_loader: bool = False,
    baseline_only: bool = False,
    baseline_method_set: str = "all",
) -> List[str]:
    if only_loader and baseline_only:
        raise ValueError("only_loader and baseline_only cannot both be True")
    if only_loader:
        return [PRIMARY_DEADLINE_METHOD]
    baseline_methods = _filter_baseline_methods(BASELINE_METHOD_ORDER, baseline_method_set)
    if baseline_only:
        return baseline_methods
    return baseline_methods + [PRIMARY_DEADLINE_METHOD]


def method_uses_hard_deadline(method: str) -> bool:
    """
    Whether a method should be truncated by the requested wall-clock deadline
    during simulation.

    Baselines are evaluated offline against deadline thresholds from their full
    completion makespan, so they run without a hard deadline cap. The primary
    deadline-aware method is the only one that is actively constrained online.
    """
    return method == PRIMARY_DEADLINE_METHOD


def execution_deadline_for_method(method: str, requested_deadline: float) -> float:
    if method_uses_hard_deadline(method):
        return float(requested_deadline)
    return float("inf")


def node_budget_allocations_for_policy(policy: Policy) -> Dict[int, float] | None:
    alloc = getattr(policy, "node_budget_allocations", None)
    if alloc is None:
        return None
    return {int(nid): float(amount) for nid, amount in alloc.items()}


def configure_deadline_trace_for_policy(
    policy: Policy,
    *,
    trace_id: str,
    context: Dict,
) -> None:
    configure = getattr(policy, "configure_trace", None)
    if configure is None:
        return
    configure(
        trace_id=trace_id,
        enabled=_env_flag("DEADLINE_TRACE", "0"),
        trace_level=_deadline_trace_level(),
        context=context,
    )


def add_deadline_trace_to_record(record: Dict, policy: Policy) -> None:
    summary_fn = getattr(policy, "deadline_trace_summary", None)
    if summary_fn is None:
        return
    record.update(summary_fn())
    pop_fn = getattr(policy, "pop_deadline_trace_events", None)
    if pop_fn is not None:
        events = pop_fn()
        if events:
            record[DEADLINE_TRACE_RECORD_KEY] = events


def no_progress_termination_reason(sim: Simulator) -> str:
    last_failure = getattr(sim, "last_start_rollout_failure_reason", None)
    if last_failure == "deadline_exhausted" or sim.deadline_reached():
        return "deadline_exhausted"
    if last_failure == "budget_exhausted":
        return "budget_exhausted"
    if sim.can_launch_any_rollout():
        return "stalled"
    return "budget_exhausted"


def parse_allocation_key(key: Any) -> Tuple[int, str]:
    if isinstance(key, tuple) and len(key) == 2:
        return int(key[0]), str(key[1])
    return int(key), DEFAULT_MODEL_ID


def apply_policy_allocation(sim: Simulator, policy: Policy) -> None:
    """
    Ask the policy for target concurrency on the current ready set, then
    launch additional rollouts to close the gap from current running counts.

    This matches the event-driven replanning semantics in the theory note:
    policy outputs are interpreted as desired in-flight widths after the
    current event, not as a fresh batch that requires a global barrier.
    """
    t0 = time.perf_counter()
    target = policy.allocate(sim)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if target or (not getattr(policy, "dispatch_once_static", False)):
        sim.planner_calls += 1
        sim.planner_wall_ms += elapsed_ms

    # Launch only the missing concurrency gap. Existing in-flight rollouts are
    # preserved; replanning only appends new rollouts when target > current.
    items = []
    for key, desired in target.items():
        nid, model_id = parse_allocation_key(key)
        current = sim.running_count(nid)
        launches = max(0, int(desired) - current)
        if launches > 0:
            items.append((launches, nid, model_id))

    # Larger gap first.
    items.sort(reverse=True)

    if getattr(policy, "uses_atomic_event_action", False):
        action = {(nid, model_id): launches for launches, nid, model_id in items}
        if action:
            sim.start_rollout_action_atomic(action)
        return

    for launches, nid, model_id in items:
        for _ in range(launches):
            if model_id == DEFAULT_MODEL_ID:
                ok = sim.start_rollout(nid)
            else:
                ok = sim.start_rollout(nid, model_id=model_id)
            if not ok:
                if getattr(sim, "last_start_rollout_failure_reason", None) == "deadline_exhausted":
                    return
                break


def run_theory_batch_simulation(sim: Simulator, policy: Policy) -> SimulationResult:
    """
    Synchronous batch loop for the MC portfolio rollout policy.

    This is the execution semantics used in the theory note: each policy call
    returns a fresh Bellman action over the current ready set, the simulator
    executes that whole batch, observes successes, and replans from the new
    state. Baseline policies continue to use the event-driven simulator below.
    """
    while True:
        if sim.done():
            sim.termination_reason = "completed"
            break

        if sim.deadline_reached():
            sim.termination_reason = "deadline_exhausted"
            break

        if sim.theory_min_remaining_time() > sim.remaining_deadline() + 1e-12:
            sim.time = sim.deadline
            sim.termination_reason = "deadline_exhausted"
            break

        t0 = time.perf_counter()
        action = policy.allocate(sim)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        sim.planner_calls += 1
        sim.planner_wall_ms += elapsed_ms

        if not action or all(int(v) <= 0 for v in action.values()):
            if sim.done():
                sim.termination_reason = "completed"
            elif sim.deadline_reached():
                sim.termination_reason = "deadline_exhausted"
            elif sim.theory_min_remaining_time() > sim.remaining_deadline() + 1e-12:
                sim.time = sim.deadline
                sim.termination_reason = "deadline_exhausted"
            else:
                sim.termination_reason = no_progress_termination_reason(sim)
            break

        ok = sim.execute_theory_batch_action(action)
        if not ok:
            if sim.termination_reason == "running":
                sim.termination_reason = no_progress_termination_reason(sim)
            break

    return sim.collect_result()


def run_single_simulation(sim: Simulator, policy: Policy) -> SimulationResult:
    """
    Event-driven simulation loop.

    Replanning trigger (theory-aligned event-driven mode):
      After initialization, the policy is called again whenever a significant
      success/failure event occurs. Policies return target concurrency on the
      current ready set, and the simulator launches only the additional rollouts
      needed to reach those targets. Existing in-flight rollouts are untouched.

    Termination reasons:
      - completed: all DAG nodes finished
      - budget_exhausted: no more rollout can be launched under remaining budget
      - deadline_exhausted: wall-clock deadline reached before completion
      - stalled: simulation cannot make progress even though budget is not the blocker
    """
    if getattr(policy, "uses_theory_batch_execution", False):
        return run_theory_batch_simulation(sim, policy)

    while True:
        if sim.deadline_reached():
            sim.termination_reason = "deadline_exhausted"
            break

        if sim.done():
            sim.termination_reason = "completed"
            break

        # If nothing is in flight, try to launch work. This covers the initial
        # state as well as states where all current rollouts have settled.
        if len(sim.event_queue) == 0:
            apply_policy_allocation(sim, policy)
            if sim.termination_reason != "running":
                break
            if len(sim.event_queue) == 0:
                if sim.done():
                    sim.termination_reason = "completed"
                else:
                    sim.termination_reason = no_progress_termination_reason(sim)
                break

        next_event_time = sim.next_event_time()
        if next_event_time is not None and next_event_time > sim.deadline:
            sim.time = sim.deadline
            sim.termination_reason = "deadline_exhausted"
            break

        event_info = sim.advance_to_next_event()
        if event_info is None:
            if sim.done():
                sim.termination_reason = "completed"
            elif sim.deadline_reached():
                sim.termination_reason = "deadline_exhausted"
            else:
                sim.termination_reason = no_progress_termination_reason(sim)
            break

        _node_id, significant = event_info
        if significant and (not sim.done()) and (not sim.deadline_reached()):
            apply_policy_allocation(sim, policy)
            if sim.termination_reason != "running":
                break

    return sim.collect_result()


def estimate_nominal_budget(
    dag,
    confidence: float = 0.95,
    budget_cost_mode: str = "token_cost_mean",
) -> float:
    """
    Work-budget nominal scale based on required confidence level and the
    selected kappa_v type.
    """
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)
    total = 0.0
    for _, node in dag.nodes.items():
        tpl = node.assigned_template
        p_v = min(max(tpl.success_prob, 1e-9), 1.0 - 1e-9)
        kappa_v = expected_node_work_cost(
            tpl,
            node.empirical_samples,
            budget_cost_mode,
            model_id=pricing_model_id_for_node(node),
        )
        r_required = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - p_v))
        r_required = max(1, int(r_required))
        total += r_required * kappa_v
    return total


def estimate_nominal_deadline(dag) -> float:
    """
    DAG-aware nominal deadline scale based on a deterministic critical-path proxy.

    Each node is assigned the proxy duration lat_mean / p_v, which matches
    the expected completion time of a single-width retry-until-success node.
    The nominal deadline is then the longest source-to-sink path under these
    node costs.
    """
    if not dag.nodes:
        return 0.0

    node_cost: Dict[int, float] = {}
    for nid, node in dag.nodes.items():
        tpl = node.assigned_template
        p_v = min(max(tpl.success_prob, 1e-9), 1.0 - 1e-9)
        node_cost[nid] = max(tpl.lat_mean, 1e-9) / p_v

    order = dag.topological_order()[::-1]
    longest_from: Dict[int, float] = {}
    for nid in order:
        children = dag.nodes[nid].children
        if not children:
            longest_from[nid] = node_cost[nid]
        else:
            longest_from[nid] = node_cost[nid] + max(longest_from[ch] for ch in children)

    roots = dag.roots()
    if not roots:
        return 0.0
    return float(max(longest_from[nid] for nid in roots))


def _success_by_deadline(success: np.ndarray, makespan: np.ndarray, deadlines: Sequence[float]) -> np.ndarray:
    out = []
    success_bool = success.astype(bool)
    for t in deadlines:
        out.append(float(np.mean(success_bool & (makespan <= float(t)))))
    return np.asarray(out, dtype=float)


def _m_alpha(success: np.ndarray, makespan: np.ndarray, alpha: float = 0.9) -> float:
    n = len(success)
    if n == 0:
        return float("nan")

    sr = float(np.mean(success))
    if sr < alpha:
        return float("inf")

    successful = np.sort(makespan[success.astype(bool)])
    k = int(math.ceil(alpha * n) - 1)
    k = min(max(k, 0), len(successful) - 1)
    return float(successful[k])


def _timeout_aware_makespan(success: np.ndarray, makespan: np.ndarray, horizon_ref: np.ndarray) -> float:
    penalized = np.where(success.astype(bool), makespan, horizon_ref)
    return float(np.mean(penalized))


def _penalized_quantile(success: np.ndarray, makespan: np.ndarray, horizon_ref: np.ndarray, q: float = 0.95) -> float:
    penalized = np.where(success.astype(bool), makespan, horizon_ref)
    return float(np.quantile(penalized, q))


def _cvar_makespan(success: np.ndarray, makespan: np.ndarray, horizon_ref: np.ndarray, alpha: float = 0.95) -> float:
    penalized = np.where(success.astype(bool), makespan, horizon_ref)
    q = np.quantile(penalized, alpha)
    tail = penalized[penalized >= q]
    return float(np.mean(tail)) if len(tail) > 0 else float("nan")


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(stat_fn(values[idx]))

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def summarize_results(
    results: List['SimulationResult'],
    m_alpha_list: Sequence[float] = (0.7, 0.8, 0.9),
    deadline_quantiles: Sequence[float] = (0.8, 0.9, 0.95),
    ci_bootstrap_n: int = 500,
    ci_alpha: float = 0.05,
    ci_seed: int = 0,
) -> Dict[str, float]:
    """
    Summarize simulation outputs.

    Added metrics:
      - timeout_aware_makespan / p95_penalized_makespan / cvar95_makespan
      - M_alpha_* (service-level makespan)
      - success_by_deadline_at_q* (deadline-based success)
      - success_rate_ci_lo/hi (bootstrap CI)
      - horizon_ref (used for timeout-aware penalization)
    """
    if len(results) == 0:
        summary = {
            "success_rate": np.nan,
            "budget_exhausted_rate": np.nan,
            "deadline_exhausted_rate": np.nan,
            "stalled_rate": np.nan,
            "avg_makespan_completed_only": np.nan,
            "std_makespan_completed_only": np.nan,
            "p95_makespan_completed_only": np.nan,
            "avg_makespan_all": np.nan,
            "std_makespan_all": np.nan,
            "avg_total_cost": np.nan,
            "avg_total_rollouts": np.nan,
            "avg_completion_fraction": np.nan,
            "std_completion_fraction": np.nan,
            "horizon_ref": np.nan,
            "timeout_aware_makespan": np.nan,
            "p95_penalized_makespan": np.nan,
            "cvar95_makespan": np.nan,
            "success_rate_ci_lo": np.nan,
            "success_rate_ci_hi": np.nan,
        }
        for alpha in m_alpha_list:
            summary[f"M_alpha_{alpha:.2f}"] = np.nan
        for q in deadline_quantiles:
            key = str(q).replace(".", "p")
            summary[f"success_by_deadline_at_q{key}"] = np.nan
        return summary

    # English note
    all_makespans = np.asarray([r.makespan for r in results], dtype=float)
    successes = np.asarray([float(r.success) for r in results], dtype=float)
    successes_bool = successes.astype(bool)
    total_costs = np.asarray([r.total_cost for r in results], dtype=float)
    completion_fractions = np.asarray([r.completion_fraction for r in results], dtype=float)
    total_rollouts = np.asarray([r.total_rollouts for r in results], dtype=float)

    completed_results = [r for r in results if r.termination_reason == "completed"]
    budget_exhausted_results = [r for r in results if r.termination_reason == "budget_exhausted"]
    deadline_exhausted_results = [r for r in results if r.termination_reason == "deadline_exhausted"]
    stalled_results = [r for r in results if r.termination_reason == "stalled"]

    # English note:English note,English note
    def calc_std(arr: np.ndarray) -> float:
        return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    # English note completed_only English note
    if len(completed_results) > 0:
        completed_makespans = np.asarray([r.makespan for r in completed_results], dtype=float)
        avg_ms_comp = float(np.mean(completed_makespans))
        std_ms_comp = calc_std(completed_makespans)
        p95_ms_comp = float(np.percentile(completed_makespans, 95))
    else:
        avg_ms_comp = np.nan
        std_ms_comp = np.nan
        p95_ms_comp = np.nan

    # English note:English note makespan,English note
    horizon_ref = float(np.max(all_makespans))
    horizon_arr = np.full_like(all_makespans, horizon_ref, dtype=float)

    timeout_aw_ms = _timeout_aware_makespan(successes_bool, all_makespans, horizon_arr)
    p95_pen_ms = _penalized_quantile(successes_bool, all_makespans, horizon_arr, q=0.95)
    cvar95_ms = _cvar_makespan(successes_bool, all_makespans, horizon_arr, alpha=0.95)

    # success_rate bootstrap CI
    ci_lo, ci_hi = _bootstrap_ci(
        successes,
        stat_fn=lambda x: float(np.mean(x)),
        n_boot=ci_bootstrap_n,
        alpha=ci_alpha,
        seed=ci_seed,
    )

    # deadline English note makespan English note;English note,English note horizon_ref
    if np.any(successes_bool):
        successful_makespans = all_makespans[successes_bool]
        deadlines = [float(np.quantile(successful_makespans, q)) for q in deadline_quantiles]
    else:
        deadlines = [horizon_ref for _ in deadline_quantiles]
    success_deadline_vals = _success_by_deadline(successes_bool, all_makespans, deadlines)

    # English note:English note
    summary = {
        # 1. English note:English note
        "success_rate": float(np.mean(successes)),
        "success_rate_ci_lo": ci_lo,
        "success_rate_ci_hi": ci_hi,
        "budget_exhausted_rate": float(len(budget_exhausted_results) / len(results)),
        "deadline_exhausted_rate": float(len(deadline_exhausted_results) / len(results)),
        "stalled_rate": float(len(stalled_results) / len(results)),
        "avg_makespan_completed_only": avg_ms_comp,
        "std_makespan_completed_only": std_ms_comp,
        "p95_makespan_completed_only": p95_ms_comp,

        # 2. English note (English note,English note run)
        "avg_makespan_all": float(np.mean(all_makespans)),
        "std_makespan_all": calc_std(all_makespans),
        "horizon_ref": horizon_ref,
        "timeout_aware_makespan": timeout_aw_ms,
        "p95_penalized_makespan": p95_pen_ms,
        "cvar95_makespan": cvar95_ms,

        # 3. English note
        "avg_total_cost": float(np.mean(total_costs)),
        "avg_total_rollouts": float(np.mean(total_rollouts)),

        # 4. English note
        "avg_completion_fraction": float(np.mean(completion_fractions)),
        "std_completion_fraction": calc_std(completion_fractions),
    }

    # M_alpha English note
    for alpha in m_alpha_list:
        summary[f"M_alpha_{alpha:.2f}"] = _m_alpha(successes_bool, all_makespans, alpha=alpha)

    # success_by_deadline English note
    for q, val in zip(deadline_quantiles, success_deadline_vals):
        key = str(q).replace(".", "p")
        summary[f"success_by_deadline_at_q{key}"] = float(val)

    return summary


def budget_to_target_success(summary_df: pd.DataFrame, target_success: float = 0.9) -> pd.DataFrame:
    """
    Compute the minimal budget ratio needed to reach target success for each
    (dag_kind, n_nodes, method) group.

    summary_df must contain columns:
      dag_kind, n_nodes, method, budget_ratio, success_rate
    """
    required_cols = {"dag_kind", "n_nodes", "method", "budget_ratio", "success_rate"}
    missing_cols = required_cols.difference(summary_df.columns)
    if missing_cols:
        raise ValueError(f"summary_df missing required columns: {sorted(missing_cols)}")

    rows = []
    group_cols = ["dag_kind", "n_nodes", "method"]

    for keys, sub in summary_df.groupby(group_cols, sort=False):
        sub = sub.sort_values("budget_ratio")
        sat = sub[sub["success_rate"] >= float(target_success)]
        budget_ratio = float(sat["budget_ratio"].iloc[0]) if len(sat) > 0 else float("inf")

        row = {c: v for c, v in zip(group_cols, keys)}
        row["target_success"] = float(target_success)
        row["budget_ratio_to_target_success"] = budget_ratio
        rows.append(row)

    return pd.DataFrame(rows)


def build_dag(lib: TaskLibrary, dag_kind: str, n_nodes: int, seed: int):
    gen = DAGGenerator(lib, seed=seed)

    if dag_kind == "layered":
        return gen.generate_layered_random_dag(
            n_nodes=n_nodes,
            n_layers=5,
            edge_prob=0.5,
        )

    if dag_kind == "chain":
        return gen.generate_chain(n_nodes)

    if dag_kind == "fork_join":
        return gen.generate_fork_join(
            width=max(2, n_nodes // 4),
            middle_depth=2,
        )

    raise ValueError(f"Unknown dag_kind={dag_kind}")


def _build_policies(
    dag,
    budget,
    deadline,
    seed,
    trial,
    colgen_overrides=None,
    only_loader=False,
    baseline_only=False,
    budget_cost_mode="token_cost_mean",
    baseline_method_set="all",
):
    """English note(English note,English note worker English note).

    Parameters
    ----------
    colgen_overrides : dict | None
        English note.English note LoaderDualColumnGenPolicy English note.
        English note key: max_dual_iters, stable_rounds, eta_beta, eta_lambda,
                     max_cg_iters, inner_dual_iters, max_active_paths, add_path_tol
        English note:English note LOADER_COLGEN_PARAMS_PATH English note JSON English note
        English note key;English note colgen English note,English note.
    only_loader : bool
        English note True,English note LOADER English note
    baseline_only : bool
        English note True,English note baseline English note,English note deadline English note
    """
    if only_loader and baseline_only:
        raise ValueError("only_loader and baseline_only cannot both be True")
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)
    baseline_method_set = _normalize_baseline_method_set(baseline_method_set)

    # Main deadline-aware algorithm kept for the PDF-aligned pipeline.
    loader_policies = {
        PRIMARY_DEADLINE_METHOD: MCPortfolioRolloutPolicy(
            full_dag=dag,
            seed=seed + 1009 * trial,
        ),
    }

    # English note(English note baseline)
    baseline_policies = {
        "sequential": SequentialPolicy(),
        "sequential_2": SequentialPolicy(rollout_width=2),
        "sequential_4": SequentialPolicy(rollout_width=4),
        "sequential_8": SequentialPolicy(rollout_width=8),
        "sequential_16": SequentialPolicy(rollout_width=16),
        "sequential_32": SequentialPolicy(rollout_width=32),
        "sequential_64": SequentialPolicy(rollout_width=64),
        "uniform": UniformPolicy(dag=dag, budget=budget, budget_cost_mode=budget_cost_mode),
        "random": RandomPolicy(dag=dag, budget=budget, seed=seed + trial, budget_cost_mode=budget_cost_mode),
        "latency_weighted": LatencyWeightedPolicy(dag=dag, budget=budget, budget_cost_mode=budget_cost_mode),
        "success_rate_weighted": SuccessRateWeightedPolicy(dag=dag, budget=budget, budget_cost_mode=budget_cost_mode),
        "critical_path_static": CriticalPathStaticPolicy(dag=dag, budget=budget, budget_cost_mode=budget_cost_mode),
    }

    if only_loader:
        return loader_policies
    baseline_policies = {
        method: policy
        for method, policy in baseline_policies.items()
        if method in _filter_baseline_methods(BASELINE_METHOD_ORDER, baseline_method_set)
    }
    if baseline_only:
        return baseline_policies

    baseline_policies.update(loader_policies)
    return baseline_policies


def _run_one_trial(args):
    """
    English note worker English note:English note trial English note.

    English note,English note ProcessPoolExecutor pickle English note.
    English note TaskLibrary English note DAG(English note),
    English note worker English note.
    """
    if len(args) == 9:
        (csv_path, dag_kind, n_nodes, budget_ratio, seed, trial,
         latency_dist, deadline_value, deadline_ratio) = args
        only_loader = False
        baseline_only = False
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 10:
        (csv_path, dag_kind, n_nodes, budget_ratio, seed, trial,
         latency_dist, deadline_value, deadline_ratio, only_loader) = args
        baseline_only = False
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 11:
        (csv_path, dag_kind, n_nodes, budget_ratio, seed, trial,
         latency_dist, deadline_value, deadline_ratio, only_loader, baseline_only) = args
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 12:
        (csv_path, dag_kind, n_nodes, budget_ratio, seed, trial,
         latency_dist, deadline_value, deadline_ratio, only_loader, baseline_only, budget_cost_mode) = args
        budget_value = None
    elif len(args) == 13:
        (csv_path, dag_kind, n_nodes, budget_ratio, seed, trial,
         latency_dist, deadline_value, deadline_ratio, only_loader, baseline_only,
         budget_cost_mode, budget_value) = args
        baseline_method_set = "all"
    elif len(args) == 14:
        (csv_path, dag_kind, n_nodes, budget_ratio, seed, trial,
         latency_dist, deadline_value, deadline_ratio, only_loader, baseline_only,
         budget_cost_mode, budget_value, baseline_method_set) = args
    else:
        raise ValueError(f"_run_one_trial expected 9, 10, 11, 12, 13, or 14 args, got {len(args)}")
    if len(args) < 12:
        budget_value = None
    if len(args) < 14:
        baseline_method_set = "all"
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)
    baseline_method_set = _normalize_baseline_method_set(baseline_method_set)

    lib = TaskLibrary(csv_path)
    dag = build_dag(lib, dag_kind, n_nodes, seed + 17 * trial)
    nominal_budget = estimate_nominal_budget(dag, budget_cost_mode=budget_cost_mode)
    nominal_deadline = estimate_nominal_deadline(dag)
    budget, _budget_input_type, _budget_axis_value, _budget_ratio_requested, _budget_value_requested = (
        resolve_budget_value(nominal_budget, budget_ratio=budget_ratio, budget_value=budget_value)
    )
    deadline, _deadline_input_type, _deadline_axis_value, _deadline_ratio_requested, _deadline_value_requested = (
        resolve_deadline_value(nominal_deadline, deadline_value=deadline_value, deadline_ratio=deadline_ratio)
    )

    policies = _build_policies(
        dag,
        budget,
        deadline,
        seed,
        trial,
        only_loader=only_loader,
        baseline_only=baseline_only,
        budget_cost_mode=budget_cost_mode,
        baseline_method_set=baseline_method_set,
    )

    trial_results = []
    for method, policy in policies.items():
        execution_deadline = execution_deadline_for_method(method, deadline)
        sim = Simulator(
            dag=dag,
            budget=budget,
            deadline=execution_deadline,
            seed=seed + 131 * trial + abs(hash(method)) % 997,
            latency_dist=latency_dist,
            node_budget_allocations=node_budget_allocations_for_policy(policy),
            budget_cost_mode=budget_cost_mode,
        )
        result = run_single_simulation(sim, policy)
        logger.debug(
            "trial=%d method=%-25s reason=%-18s makespan=%.3f cost=%.3f",
            trial, method, result.termination_reason, result.makespan, result.total_cost,
        )
        trial_results.append((method, result))
    return trial_results


def run_experiment(
    csv_path: str,
    dag_kind: str = "layered",
    n_nodes: int = 30,
    budget_ratio: float = 0.5,
    budget_value: float | None = None,
    n_trials: int = 10,
    seed: int = 0,
    latency_dist: str = "lognormal",
    n_workers: int = None,
    deadline_value: float | None = None,
    deadline_ratio: float = 1.0,
    budget_cost_mode: str = "token_cost_mean",
) -> Dict[str, Dict[str, float]]:
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, n_trials)

    args_list = [
        (
            csv_path, dag_kind, n_nodes, budget_ratio, seed, trial,
            latency_dist, deadline_value, deadline_ratio,
            False, False, budget_cost_mode, budget_value,
        )
        for trial in range(n_trials)
    ]

    outputs = defaultdict(list)

    pbar = tqdm(
        total=n_trials,
        desc=f"{dag_kind} n={n_nodes} br={budget_ratio}",
        unit="trial",
        file=sys.stderr,
        dynamic_ncols=True,
    )

    if n_workers <= 1:
        # English note(English note n_trials=1 English note)
        for args in args_list:
            for method, result in _run_one_trial(args):
                outputs[method].append(result)
            pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run_one_trial, args) for args in args_list]
            for future in as_completed(futures):
                for method, result in future.result():
                    outputs[method].append(result)
                pbar.update(1)

    pbar.close()
    return {method: summarize_results(results) for method, results in outputs.items()}


def sweep(
    csv_path: str,
    dag_kinds: List[str],
    n_nodes_list: List[int],
    budget_ratios: List[float],
    budget_values: List[float] | None = None,
    n_trials: int = 10,
    seed: int = 0,
    latency_dist: str = "lognormal",
    n_workers: int = None,
    deadline_value: float | None = None,
    deadline_ratio: float = 1.0,
    budget_cost_mode: str = "token_cost_mean",
) -> pd.DataFrame:
    """
    English note.

    English note (dag_kind, n_nodes, budget_ratio, trial) English note,
    English note,English note CPU English note.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)

    # English note (config, trial) English note
    all_args = []
    all_keys = []
    for dag_kind in dag_kinds:
        for n_nodes in n_nodes_list:
            budget_axes = [(budget_ratio, None) for budget_ratio in budget_ratios]
            if budget_values is not None:
                budget_axes = [(None, budget_value) for budget_value in budget_values]
            for budget_ratio, budget_value in budget_axes:
                for trial in range(n_trials):
                    all_args.append((
                        csv_path, dag_kind, n_nodes, budget_ratio, seed, trial,
                        latency_dist, deadline_value, deadline_ratio,
                        False, False, budget_cost_mode, budget_value,
                    ))
                    all_keys.append((dag_kind, n_nodes, budget_value if budget_value is not None else budget_ratio))

    total = len(all_args)
    n_configs = len(dag_kinds) * len(n_nodes_list) * len(budget_ratios)
    logger.info(
        "[sweep] Total tasks: %d (%d configs x %d trials) | Workers: %d",
        total, n_configs, n_trials, n_workers,
    )

    # English note:config_key -> method -> [SimulationResult, ...]
    config_outputs = defaultdict(lambda: defaultdict(list))

    # English note:English note,English note,English note,English note
    pbar = tqdm(
        total=total,
        desc="Sweep",
        unit="trial",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} trials "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
        file=sys.stderr,
        dynamic_ncols=True,
    )

    def _update_progress(key):
        """English note:+1 English note postfix English note."""
        dag_k, nn, br = key
        pbar.set_postfix_str(f"{dag_k} n={nn} br={br}", refresh=False)
        pbar.update(1)

    if n_workers <= 1:
        for key, args in zip(all_keys, all_args):
            trial_results = _run_one_trial(args)
            for method, result in trial_results:
                config_outputs[key][method].append(result)
            _update_progress(key)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_one_trial, args): key
                for args, key in zip(all_args, all_keys)
            }
            for future in as_completed(futures):
                key = futures[future]
                trial_results = future.result()
                for method, result in trial_results:
                    config_outputs[key][method].append(result)
                _update_progress(key)

    pbar.close()

    # English note DataFrame
    rows = []
    for (dag_kind, n_nodes, budget_ratio), method_results in config_outputs.items():
        for method, results in method_results.items():
            row = {
                "dag_kind": dag_kind,
                "n_nodes": n_nodes,
                "budget_ratio": budget_ratio,
                "method": method,
            }
            row.update(summarize_results(results))
            rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# sweep_v2: English note(DAG English note x English note)
# =====================================================================

def _run_one_cell_v2(args):
    """
    English note worker English note.

    English note (DAG, English note) English note,
    English note flat dict,English note dict English note.

    English note:
      - English note cell English note DAG(English note + English note)
        (lat_mean, lat_std, success_prob English note → "English note")
      - English note (English note, exec_trial) English note
        (English note → "English note")
    """
    if len(args) == 9:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, deadline_value, deadline_ratio) = args
        only_loader = False
        baseline_only = False
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 10:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, deadline_value, deadline_ratio, only_loader) = args
        baseline_only = False
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 11:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, deadline_value, deadline_ratio, only_loader, baseline_only) = args
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 12:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, deadline_value, deadline_ratio,
         only_loader, baseline_only, budget_cost_mode) = args
        budget_value = None
    elif len(args) == 13:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, deadline_value, deadline_ratio,
         only_loader, baseline_only, budget_cost_mode, budget_value) = args
        baseline_method_set = "all"
    elif len(args) == 14:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, deadline_value, deadline_ratio,
         only_loader, baseline_only, budget_cost_mode, budget_value, baseline_method_set) = args
    else:
        raise ValueError(f"_run_one_cell_v2 expected 9, 10, 11, 12, 13, or 14 args, got {len(args)}")
    if len(args) < 12:
        budget_value = None
    if len(args) < 14:
        baseline_method_set = "all"
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)
    baseline_method_set = _normalize_baseline_method_set(baseline_method_set)

    dag = DAG.from_full_dict(full_dict)

    budget_nominal = estimate_nominal_budget(dag, budget_cost_mode=budget_cost_mode)
    deadline_nominal = estimate_nominal_deadline(dag)
    budget, budget_input_type, budget_axis_value, budget_ratio_requested, budget_value_requested = (
        resolve_budget_value(budget_nominal, budget_ratio=budget_ratio, budget_value=budget_value)
    )
    deadline, deadline_input_type, deadline_axis_value, deadline_ratio_requested, deadline_value_requested = (
        resolve_deadline_value(deadline_nominal, deadline_value=deadline_value, deadline_ratio=deadline_ratio)
    )

    method_names = list(_build_policies(
        dag,
        budget,
        deadline,
        seed=dag_idx,
        trial=0,
        only_loader=only_loader,
        baseline_only=baseline_only,
        budget_cost_mode=budget_cost_mode,
        baseline_method_set=baseline_method_set,
    ).keys())

    # English note method English note,English note exec_seed English note
    method_offsets = {}
    for i, method in enumerate(method_names):
        method_offsets[method] = (i + 1) * 104_729

    records = []
    for exec_idx, exec_seed in enumerate(exec_seeds):
        policies = _build_policies(
            dag,
            budget,
            deadline,
            seed=dag_idx,
            trial=exec_idx,
            only_loader=only_loader,
            baseline_only=baseline_only,
            budget_cost_mode=budget_cost_mode,
            baseline_method_set=baseline_method_set,
        )
        for method, policy in policies.items():
            sim_seed = exec_seed + method_offsets[method]
            deadline_enforced = method_uses_hard_deadline(method)
            execution_deadline = execution_deadline_for_method(method, deadline)
            if method == PRIMARY_DEADLINE_METHOD:
                trace_id = _deadline_trace_id(
                    dag_idx=dag_idx,
                    exec_idx=exec_idx,
                    budget_ratio=budget_axis_value,
                    deadline_axis_value=deadline_axis_value,
                    budget_cost_mode=budget_cost_mode,
                )
                configure_deadline_trace_for_policy(
                    policy,
                    trace_id=trace_id,
                    context={
                        "dag_idx": int(dag_idx),
                        "dag_kind": dag_kind,
                        "n_nodes": int(n_nodes),
                        "exec_trial": int(exec_idx),
                        "budget_input_type": budget_input_type,
                        "budget_axis_value": float(budget_axis_value),
                        "budget_ratio": float(budget_axis_value),
                        "budget_ratio_requested": None if budget_ratio_requested is None else float(budget_ratio_requested),
                        "budget_value_requested": None if budget_value_requested is None else float(budget_value_requested),
                        "budget_cost_mode": budget_cost_mode,
                        "deadline_input_type": deadline_input_type,
                        "deadline_axis_value": float(deadline_axis_value),
                        "deadline": float(deadline),
                        "sim_seed": int(sim_seed),
                    },
                )
            sim = Simulator(
                dag=dag,
                budget=budget,
                deadline=execution_deadline,
                seed=sim_seed,
                latency_dist=latency_dist,
                node_budget_allocations=node_budget_allocations_for_policy(policy),
                budget_cost_mode=budget_cost_mode,
            )
            result = run_single_simulation(sim, policy)

            record = {
                # ── English note ──
                "dag_idx": dag_idx,
                "dag_kind": dag_kind,
                "n_nodes": n_nodes,
                "exec_trial": exec_idx,
                "budget_input_type": budget_input_type,
                "budget_axis_value": budget_axis_value,
                "budget_ratio": budget_axis_value,
                "budget_ratio_requested": budget_ratio_requested,
                "budget_value_requested": budget_value_requested,
                "budget_cost_mode": budget_cost_mode,
                "budget_nominal": budget_nominal,
                "budget": budget,
                "method": method,
                # ── English note ──
                "success": result.success,
                "termination_reason": result.termination_reason,
                "deadline_input_type": deadline_input_type,
                "deadline_axis_value": deadline_axis_value,
                "deadline_ratio_requested": deadline_ratio_requested,
                "deadline_value_requested": deadline_value_requested,
                "deadline_nominal": deadline_nominal,
                "deadline": deadline,
                "execution_deadline": None if not deadline_enforced else execution_deadline,
                "deadline_enforced": deadline_enforced,
                "makespan": result.makespan,
                "total_cost": result.total_cost,
                "total_rollouts": result.total_rollouts,
                "completion_fraction": result.completion_fraction,
            }
            add_deadline_trace_to_record(record, policy)
            records.append(record)

    return records


def _run_one_cell_v2_chunk(args):
    """
    _run_one_cell_v2 English note:English note exec_seeds English note.

    English note trial English note--English note DAG English note,English note DAG English note trial
    English note chunk,English note.

    args English note exec_seeds English note exec_idx_offset English note.
    """
    if len(args) == 10:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, exec_idx_offset, deadline_value, deadline_ratio) = args
        only_loader = False
        baseline_only = False
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 11:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, exec_idx_offset, deadline_value, deadline_ratio, only_loader) = args
        baseline_only = False
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 12:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, exec_idx_offset, deadline_value, deadline_ratio, only_loader, baseline_only) = args
        budget_cost_mode = "token_cost_mean"
    elif len(args) == 13:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, exec_idx_offset, deadline_value,
         deadline_ratio, only_loader, baseline_only, budget_cost_mode) = args
        budget_value = None
    elif len(args) == 14:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, exec_idx_offset, deadline_value,
         deadline_ratio, only_loader, baseline_only, budget_cost_mode, budget_value) = args
        baseline_method_set = "all"
    elif len(args) == 15:
        (dag_idx, dag_kind, n_nodes, full_dict,
         budget_ratio, exec_seeds, latency_dist, exec_idx_offset, deadline_value,
         deadline_ratio, only_loader, baseline_only, budget_cost_mode, budget_value,
         baseline_method_set) = args
    else:
        raise ValueError(f"_run_one_cell_v2_chunk expected 10, 11, 12, 13, 14, or 15 args, got {len(args)}")
    if len(args) < 13:
        budget_value = None
    if len(args) < 15:
        baseline_method_set = "all"
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)
    baseline_method_set = _normalize_baseline_method_set(baseline_method_set)

    dag = DAG.from_full_dict(full_dict)

    budget_nominal = estimate_nominal_budget(dag, budget_cost_mode=budget_cost_mode)
    deadline_nominal = estimate_nominal_deadline(dag)
    budget, budget_input_type, budget_axis_value, budget_ratio_requested, budget_value_requested = (
        resolve_budget_value(budget_nominal, budget_ratio=budget_ratio, budget_value=budget_value)
    )
    deadline, deadline_input_type, deadline_axis_value, deadline_ratio_requested, deadline_value_requested = (
        resolve_deadline_value(deadline_nominal, deadline_value=deadline_value, deadline_ratio=deadline_ratio)
    )

    method_names = list(_build_policies(
        dag,
        budget,
        deadline,
        seed=dag_idx,
        trial=0,
        only_loader=only_loader,
        baseline_only=baseline_only,
        budget_cost_mode=budget_cost_mode,
        baseline_method_set=baseline_method_set,
    ).keys())

    method_offsets = {}
    for i, method in enumerate(method_names):
        method_offsets[method] = (i + 1) * 104_729

    records = []
    for local_idx, exec_seed in enumerate(exec_seeds):
        exec_idx = exec_idx_offset + local_idx
        policies = _build_policies(
            dag,
            budget,
            deadline,
            seed=dag_idx,
            trial=exec_idx,
            only_loader=only_loader,
            baseline_only=baseline_only,
            budget_cost_mode=budget_cost_mode,
            baseline_method_set=baseline_method_set,
        )
        for method, policy in policies.items():
            sim_seed = exec_seed + method_offsets[method]
            deadline_enforced = method_uses_hard_deadline(method)
            execution_deadline = execution_deadline_for_method(method, deadline)
            if method == PRIMARY_DEADLINE_METHOD:
                trace_id = _deadline_trace_id(
                    dag_idx=dag_idx,
                    exec_idx=exec_idx,
                    budget_ratio=budget_axis_value,
                    deadline_axis_value=deadline_axis_value,
                    budget_cost_mode=budget_cost_mode,
                )
                configure_deadline_trace_for_policy(
                    policy,
                    trace_id=trace_id,
                    context={
                        "dag_idx": int(dag_idx),
                        "dag_kind": dag_kind,
                        "n_nodes": int(n_nodes),
                        "exec_trial": int(exec_idx),
                        "budget_input_type": budget_input_type,
                        "budget_axis_value": float(budget_axis_value),
                        "budget_ratio": float(budget_axis_value),
                        "budget_ratio_requested": None if budget_ratio_requested is None else float(budget_ratio_requested),
                        "budget_value_requested": None if budget_value_requested is None else float(budget_value_requested),
                        "budget_cost_mode": budget_cost_mode,
                        "deadline_input_type": deadline_input_type,
                        "deadline_axis_value": float(deadline_axis_value),
                        "deadline": float(deadline),
                        "sim_seed": int(sim_seed),
                    },
                )
            sim = Simulator(
                dag=dag,
                budget=budget,
                deadline=execution_deadline,
                seed=sim_seed,
                latency_dist=latency_dist,
                node_budget_allocations=node_budget_allocations_for_policy(policy),
                budget_cost_mode=budget_cost_mode,
            )
            result = run_single_simulation(sim, policy)

            record = {
                "dag_idx": dag_idx,
                "dag_kind": dag_kind,
                "n_nodes": n_nodes,
                "exec_trial": exec_idx,
                "budget_input_type": budget_input_type,
                "budget_axis_value": budget_axis_value,
                "budget_ratio": budget_axis_value,
                "budget_ratio_requested": budget_ratio_requested,
                "budget_value_requested": budget_value_requested,
                "budget_cost_mode": budget_cost_mode,
                "budget_nominal": budget_nominal,
                "budget": budget,
                "method": method,
                "success": result.success,
                "termination_reason": result.termination_reason,
                "deadline_input_type": deadline_input_type,
                "deadline_axis_value": deadline_axis_value,
                "deadline_ratio_requested": deadline_ratio_requested,
                "deadline_value_requested": deadline_value_requested,
                "deadline_nominal": deadline_nominal,
                "deadline": deadline,
                "execution_deadline": None if not deadline_enforced else execution_deadline,
                "deadline_enforced": deadline_enforced,
                "makespan": result.makespan,
                "total_cost": result.total_cost,
                "total_rollouts": result.total_rollouts,
                "completion_fraction": result.completion_fraction,
            }
            add_deadline_trace_to_record(record, policy)
            records.append(record)

    return records


def sweep_v2(
    dag_pool_path: str,
    budget_ratios: List[float],
    budget_values: List[float] | None = None,
    n_exec_trials: int = 10,
    seed: int = 42,
    latency_dist: str = "lognormal",
    n_workers: int = None,
    output_dir: str = None,
    deadline_value: float | None = None,
    deadline_ratio: float = 1.0,
    budget_cost_mode: str = "token_cost_mean",
):
    """
    English note.

    English note:
      English note 0 English note - DAG(English note + English note):English note dag_pool.jsonl English note,
               English note DAG English note,English note.
               English note DAG English note.
      English note 1 English note - English note(English note + English note/English note):
               English note DAG English note.

    English note:
      English note budget_ratio English note JSONL English note:
        output_dir/raw_budget_{budget_ratio}.jsonl
      English note JSON English note,English note.

    English note:
      dag_pool_path:      English note DAG English note(dag_pool.jsonl,English note+English note)
      budget_ratios:      English note
      n_exec_trials:      English note DAG English note(English note 1 English note)
      seed:               English note
      latency_dist:       English note ("lognormal" / "gaussian" / "gamma")
      n_workers:          English note (None=English note)
      output_dir:         English note(English note),English note budget_ratio English note JSONL English note
    """
    if output_dir is None:
        raise ValueError("output_dir is required")
    budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)

    os.makedirs(output_dir, exist_ok=True)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # English note DAG English note(English note + English note,JSONL English note)
    dag_pool = DAG.load_pool_jsonl(dag_pool_path)
    n_dags = len(dag_pool)
    logger.info("[sweep_v2] Loaded %d DAGs (with templates) from %s", n_dags, dag_pool_path)

    # English note DAG English note(English note budget_ratio English note)
    dag_args = []
    for dag_idx, full_dict in enumerate(dag_pool):
        dag_kind = full_dict.get("dag_kind", "unknown")
        n_nodes = full_dict.get("n_nodes", len(full_dict.get("nodes", [])))
        exec_seeds = [
            seed + dag_idx * 10007 + exec_trial * 7 + 1
            for exec_trial in range(n_exec_trials)
        ]
        dag_args.append((dag_idx, dag_kind, n_nodes, full_dict, exec_seeds))

    total_mc = n_dags * n_exec_trials
    if dag_pool:
        sample_dag = DAG.from_full_dict(dag_pool[0])
        sample_budget_nominal = estimate_nominal_budget(sample_dag, budget_cost_mode=budget_cost_mode)
        first_budget_value = budget_values[0] if budget_values else None
        sample_budget, _budget_input_type, _budget_axis_value, _budget_ratio_requested, _budget_value_requested = (
            resolve_budget_value(
                sample_budget_nominal,
                budget_ratio=budget_ratios[0] if budget_ratios else None,
                budget_value=first_budget_value,
            )
        )
        sample_deadline = (
            float(deadline_value)
            if deadline_value is not None
            else float(deadline_ratio * estimate_nominal_deadline(sample_dag))
        )
        n_methods = len(_build_policies(
            sample_dag,
            sample_budget,
            sample_deadline,
            seed=0,
            trial=0,
            budget_cost_mode=budget_cost_mode,
        ))
    else:
        n_methods = 0
    budget_specs = [(float(br), None) for br in budget_ratios]
    if budget_values is not None:
        budget_specs = [(None, float(value)) for value in budget_values]
    n_sims_total = total_mc * len(budget_specs) * n_methods
    logger.info(
        "[sweep_v2] %d DAGs x %d exec = %d MC samples per (budget, method) | "
        "%d budget settings | ~%d total sims | Workers: %d",
        n_dags, n_exec_trials, total_mc,
        len(budget_specs), n_sims_total, n_workers,
    )

    total_written = 0

    # English note budget setting English note,English note JSONL English note
    for br_idx, (budget_ratio, budget_value) in enumerate(budget_specs):
        budget_input_type = "value" if budget_value is not None else "ratio"
        budget_axis_value = budget_value if budget_value is not None else budget_ratio
        jsonl_name = (
            f"raw_budget_value_{budget_axis_value}.jsonl"
            if budget_input_type == "value"
            else f"raw_budget_{budget_axis_value}.jsonl"
        )
        jsonl_path = os.path.join(output_dir, jsonl_name)

        # English note budget setting English note cell English note
        br_args = []
        for (dag_idx, dag_kind, n_nodes, full_dict, exec_seeds) in dag_args:
            br_args.append((
                dag_idx, dag_kind, n_nodes, full_dict,
                budget_ratio, exec_seeds, latency_dist, deadline_value, deadline_ratio,
                False, False, budget_cost_mode, budget_value,
            ))

        n_cells = len(br_args)
        logger.info(
            "[sweep_v2] budget_%s=%s (%d/%d) - %d cells → %s",
            budget_input_type, budget_axis_value, br_idx + 1, len(budget_specs), n_cells, jsonl_path,
        )

        pbar = tqdm(
            total=n_cells,
            desc=f"budget={budget_axis_value}",
            unit="cell",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} cells "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
            file=sys.stderr,
            dynamic_ncols=True,
        )

        n_written = 0

        with open(jsonl_path, "w") as f_out:
            if n_workers <= 1:
                for args in br_args:
                    records = _run_one_cell_v2(args)
                    for rec in records:
                        write_record_jsonl_with_traces(f_out, rec, output_dir)
                    n_written += len(records)
                    pbar.set_postfix_str(f"written={n_written}", refresh=False)
                    pbar.update(1)
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(_run_one_cell_v2, args): args
                        for args in br_args
                    }
                    for future in as_completed(futures):
                        records = future.result()
                        for rec in records:
                            write_record_jsonl_with_traces(f_out, rec, output_dir)
                        n_written += len(records)
                        pbar.set_postfix_str(f"written={n_written}", refresh=False)
                        pbar.update(1)

        pbar.close()
        total_written += n_written
        logger.info(
            "[sweep_v2] budget_%s=%s done - %d records → %s",
            budget_input_type, budget_axis_value, n_written, jsonl_path,
        )

    logger.info("[sweep_v2] All done. Total %d records across %d files in %s",
                total_written, len(budget_specs), output_dir)
    return total_written


def _setup_logging(debug: bool = False):
    """English note."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LOADER DAG simulation experiments")
    parser.add_argument(
        "--debug", action="store_true",
        help="English note:English note + English note + DEBUG English note",
    )
    args = parser.parse_args()

    _setup_logging(debug=args.debug)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root_dir, "simulator", "example_rollout_stats.csv")
    output_path = os.path.join(root_dir, "results", "offline_dag_sim_results.csv")

    if args.debug:
        # ---- English note:2 English note,English note ----
        logger.info("=== DEBUG MODE: English note + English note ===")
        df = sweep(
            csv_path=csv_path,
            dag_kinds=["chain"],
            n_nodes_list=[10],
            budget_ratios=[1.0],
            n_trials=2,
            seed=42,
            latency_dist="lognormal",
            n_workers=1,               # English note,English note
        )
    else:
        df = sweep(
            csv_path=csv_path,
            dag_kinds=["layered", "chain", "fork_join"],
            n_nodes_list=[20, 50],
            budget_ratios=[0.3, 0.5, 0.8],
            n_trials=1000,
            seed=42,
            latency_dist="lognormal",
        )

    logger.info("\n%s", df.head())
    df.to_csv(output_path, index=False)
    logger.info("Saved to %s", output_path)
