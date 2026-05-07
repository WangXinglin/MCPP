import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dag import DAG
from .LOADER_policies import Policy
from .simulator import Simulator
from .types import (
    DEFAULT_MODEL_ID,
    output_money_cost_from_tokens,
    output_price_per_million_tokens,
    realized_rollout_work_cost,
)

if str(os.environ.get("MC_PORTFOLIO_NUMBA_KERNEL", "off")).strip().lower() == "on":
    try:
        from numba import njit
    except Exception:  # pragma: no cover - optional acceleration dependency
        njit = None  # type: ignore[assignment]
else:
    njit = None  # type: ignore[assignment]


ModelId = str
Action = Tuple[Tuple[int, int], ...]
Continuation = Tuple[ModelId, int]
ModelAction = Tuple[Tuple[int, ModelId, int], ...]
PendingEvent = Tuple[float, int, ModelId, Optional[bool]]
_THREAD_MODEL: Optional["_TheoryModel"] = None
_THREAD_STATE: Optional["_MCState"] = None
_THREAD_MAX_ROLLOUT_STEPS = 10000
_THREAD_USE_NUMBA_KERNEL = False
_THREAD_NUMBA_DATA: Optional["_NumbaModelData"] = None


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return max(min_value, int(raw))
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; expected an integer >= {min_value}") from exc


def _env_float(name: str, default: float, min_value: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float >= {min_value}") from exc
    if value < min_value:
        raise ValueError(f"Invalid {name}={raw!r}; expected a float >= {min_value}")
    return value


def _env_choice(name: str, default: str, choices: Sequence[str]) -> str:
    raw = os.environ.get(name, default)
    value = str(raw).strip().lower()
    allowed = {str(choice).lower() for choice in choices}
    if value not in allowed:
        raise ValueError(f"Invalid {name}={raw!r}; expected one of {sorted(allowed)}")
    return value


def _default_pair_workers() -> int:
    n_cpus = os.cpu_count()
    if n_cpus is None or n_cpus <= 0:
        return 1
    return max(1, int(n_cpus) // 2)


def _base_ks(max_k: int = 64) -> Tuple[int, ...]:
    out = []
    k = 1
    while k <= max_k:
        out.append(k)
        k *= 4
    return tuple(out)


def _parse_base_ks(raw: Optional[str | Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw.strip():
            return None
        parts = raw.replace(";", ",").replace(" ", ",").split(",")
    else:
        parts = [str(item) for item in raw]
    out = []
    seen = set()
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid MC_PORTFOLIO_BASE_KS entry {text!r}; expected positive integers") from exc
        if value <= 0:
            raise ValueError(f"Invalid MC_PORTFOLIO_BASE_KS entry {text!r}; expected positive integers")
        if value not in seen:
            seen.add(value)
            out.append(value)
    if not out:
        raise ValueError("MC_PORTFOLIO_BASE_KS must contain at least one positive integer")
    return tuple(out)


def _parse_model_ids(raw: Optional[str | Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = [str(item) for item in raw]
    raw_out = [str(part).strip() for part in parts if str(part).strip()]
    seen = set()
    out = []
    for model_id in raw_out:
        if model_id in seen:
            continue
        seen.add(model_id)
        out.append(model_id)
    if not out:
        raise ValueError("MC_PORTFOLIO_MODEL_IDS/model_ids must contain at least one model id.")
    return tuple(out)


def _normalize_action(items: Iterable[Tuple[int, int]]) -> Action:
    clean = [(int(nid), int(count)) for nid, count in items if int(count) > 0]
    clean.sort(key=lambda x: x[0])
    seen = set()
    for nid, _count in clean:
        if nid in seen:
            raise ValueError(f"Duplicate node id {nid} in MC action.")
        seen.add(nid)
    return tuple(clean)


def _normalize_model_action(items: Iterable[Tuple[int, str, int]]) -> ModelAction:
    clean = [
        (int(nid), str(model_id), int(count))
        for nid, model_id, count in items
        if int(count) > 0
    ]
    clean.sort(key=lambda x: x[0])
    seen = set()
    for nid, _model_id, _count in clean:
        if nid in seen:
            raise ValueError(f"Duplicate node id {nid} in MC model action.")
        seen.add(nid)
    return tuple(clean)


def _count_action_to_model_action(model_id: str, action: Action) -> ModelAction:
    model_id = str(model_id)
    return _normalize_model_action((nid, model_id, count) for nid, count in action)


def _model_action_counts(action: ModelAction) -> Dict[str, int]:
    return {str(nid): int(count) for nid, _model_id, count in action}


def _model_action_models(action: ModelAction) -> Dict[str, str]:
    return {str(nid): str(model_id) for nid, model_id, _count in action}


def _model_action_legacy_model(action: Optional[ModelAction]) -> Optional[str]:
    if not action:
        return None
    models = {str(model_id) for _nid, model_id, _count in action}
    if len(models) == 1:
        return next(iter(models))
    return "mixed"


def _default_model_for_compat(model: "_TheoryModel") -> str:
    if DEFAULT_MODEL_ID in model.model_ids:
        return DEFAULT_MODEL_ID
    if not model.model_ids:
        return DEFAULT_MODEL_ID
    return str(model.model_ids[0])


def _pending_event_parts(event: Tuple) -> PendingEvent:
    if len(event) == 4:
        remaining, nid, model_id, success = event
        if (
            type(remaining) is float
            and remaining >= 0.0
            and type(nid) is int
            and type(model_id) is str
            and (success is None or type(success) is bool)
        ):
            return event  # type: ignore[return-value]
        return (
            float(max(remaining, 0.0)),
            int(nid),
            str(model_id),
            None if success is None else bool(success),
        )
    if len(event) == 3:
        remaining, nid, success = event
        return (
            float(max(remaining, 0.0)),
            int(nid),
            DEFAULT_MODEL_ID,
            None if success is None else bool(success),
        )
    raise ValueError(f"Invalid pending event shape: {event!r}")


def _pending_event_sort_key(event: PendingEvent) -> Tuple[float, int, str, int]:
    return (
        event[0],
        event[1],
        event[2],
        1 if event[3] is None else int(bool(event[3])),
    )


def _normalize_pending(events: Iterable[PendingEvent]) -> Tuple[PendingEvent, ...]:
    normalized = [_pending_event_parts(event) for event in events]
    if len(normalized) < 2:
        return tuple(normalized)

    prev_key = _pending_event_sort_key(normalized[0])
    for event in normalized[1:]:
        key = _pending_event_sort_key(event)
        if key < prev_key:
            return tuple(sorted(normalized, key=_pending_event_sort_key))
        prev_key = key
    return tuple(normalized)


def _pending_node_ids_from_pending(events: Iterable[PendingEvent]) -> FrozenSet[int]:
    return frozenset(_pending_event_parts(event)[1] for event in events)


def _pending_node_ids(state: "_MCState") -> FrozenSet[int]:
    cached = getattr(state, "pending_node_ids", None)
    if cached is not None:
        return cached
    pending_node_ids = _pending_node_ids_from_pending(state.pending)
    object.__setattr__(state, "pending_node_ids", pending_node_ids)
    return pending_node_ids


def _advance_to_next_replan_state(
    model: "_TheoryModel",
    state: "_MCState",
    rng: np.random.Generator,
) -> Optional["_MCState"]:
    cur = state
    while cur.pending:
        pending = cur.pending
        remaining, nid, model_id, sampled_success = _pending_event_parts(pending[0])
        if remaining > cur.h + 1e-12:
            return None

        # cur.pending is already sorted by remaining time. Subtracting the same
        # elapsed time from every pending event preserves that order.
        advanced_pending_items = []
        still_running = False
        for event in pending[1:]:
            old_remaining, old_nid, old_model_id, old_success = _pending_event_parts(event)
            if old_nid == nid:
                still_running = True
            advanced_pending_items.append((
                max(0.0, old_remaining - remaining),
                old_nid,
                old_model_id,
                old_success,
            ))
        advanced_pending = tuple(advanced_pending_items)
        completed = set(cur.completed)
        partial_success = set(cur.partial_success)
        success = (
            model.sample_success(nid, model_id, rng)
            if sampled_success is None
            else bool(sampled_success)
        )
        if nid not in completed and success:
            partial_success.add(nid)

        if nid not in completed and (not still_running) and nid in partial_success:
            completed.add(nid)
            partial_success.discard(nid)

        next_state = _MCState(
            completed=frozenset(completed),
            h=max(0.0, cur.h - remaining),
            b=cur.b,
            pending=advanced_pending,
            partial_success=frozenset(partial_success),
        )

        if model.done(next_state.completed) or (not still_running):
            return next_state
        cur = next_state
    return cur


def _sample_transition_rng(
    model: "_TheoryModel",
    state: "_MCState",
    current_action_or_model_id: str | Action | ModelAction,
    action: Optional[Action | np.random.Generator] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional["_MCState"]:
    current_action: ModelAction
    if rng is None:
        # Backward-compatible call shape:
        # _sample_transition_rng(model, state, action_or_model_action, rng)
        rng = action  # type: ignore[assignment]
        action = current_action_or_model_id  # type: ignore[assignment]
        if action and len(action[0]) == 3:  # type: ignore[index]
            current_action = _normalize_model_action(action)  # type: ignore[arg-type]
        else:
            current_action = _count_action_to_model_action(_default_model_for_compat(model), action)  # type: ignore[arg-type]
    elif action is not None:
        # Global-model count action, used by Seq-(m,k) continuation.
        current_action = _count_action_to_model_action(str(current_action_or_model_id), action)  # type: ignore[arg-type]
    else:
        current_action = _normalize_model_action(current_action_or_model_id)  # type: ignore[arg-type]
    if rng is None:
        raise ValueError("_sample_transition_rng requires an rng")
    if not current_action:
        return None

    cost = 0.0
    max_latency = 0.0
    pending = list(state.pending)
    pending_nodes = _pending_node_ids(state)
    completed_nodes = state.completed
    node_model_ids = model.node_model_ids
    sample_empirical_rollout_batch = model.sample_empirical_rollout_batch
    sample_rollout_realization = model.sample_rollout_realization
    seen_nodes = set()
    for nid, current_model_id, count in current_action:
        nid = int(nid)
        current_model_id = str(current_model_id)
        if nid in seen_nodes:
            return None
        seen_nodes.add(nid)
        if nid in completed_nodes or nid in pending_nodes:
            return None
        if current_model_id not in node_model_ids.get(nid, ()):
            return None
        count_int = int(count)
        batch = (
            sample_empirical_rollout_batch(nid, current_model_id, count_int, rng)
            if count_int > 1
            else None
        )
        if batch is not None:
            batch_cost, batch_max_latency, batch_pending = batch
            cost += batch_cost
            max_latency = max(max_latency, batch_max_latency)
            pending.extend(batch_pending)
            continue

        for _ in range(count_int):
            sampled_latency, success, realized_cost = sample_rollout_realization(nid, current_model_id, rng)
            cost += realized_cost
            max_latency = max(max_latency, sampled_latency)
            pending.append((sampled_latency, nid, current_model_id, success))

    if cost > state.b + 1e-12:
        return None
    if max_latency > state.h + 1e-12:
        return None

    launched_state = _MCState(
        completed=state.completed,
        h=state.h,
        b=max(0.0, state.b - cost),
        pending=_normalize_pending(pending),
        partial_success=state.partial_success,
    )
    return _advance_to_next_replan_state(model, launched_state, rng)


def _rollout_seq_model_k_rng(
    model: "_TheoryModel",
    state: "_MCState",
    continuation_model_id: str,
    k: int,
    max_rollout_steps: int,
    rng: np.random.Generator,
) -> bool:
    continuation_model_id = str(continuation_model_id)
    cur = state
    for _step in range(max_rollout_steps):
        if model.done(cur.completed):
            return True
        if cur.h <= 0.0:
            return False

        ready = model.dispatchable_ready_nodes(cur)
        if not ready:
            if cur.pending:
                cur = _advance_to_next_replan_state(model, cur, rng)
                if cur is None:
                    return False
                continue
            return False
        k_int = int(k)
        action = tuple((int(nid), k_int) for nid in ready)
        cur = _sample_transition_rng(model, cur, continuation_model_id, action, rng)
        if cur is None:
            return False
    return False


def _rollout_seq_k_rng(
    model: "_TheoryModel",
    state: "_MCState",
    k: int,
    max_rollout_steps: int,
    rng: np.random.Generator,
) -> bool:
    return _rollout_seq_model_k_rng(
        model,
        state,
        _default_model_for_compat(model),
        k,
        max_rollout_steps,
        rng,
    )


def _thread_eval_initializer(
    model: "_TheoryModel",
    state: "_MCState",
    max_rollout_steps: int,
    use_numba_kernel: bool = False,
) -> None:
    global _THREAD_MODEL, _THREAD_STATE, _THREAD_MAX_ROLLOUT_STEPS, _THREAD_USE_NUMBA_KERNEL, _THREAD_NUMBA_DATA
    _THREAD_MODEL = model
    _THREAD_STATE = state
    _THREAD_MAX_ROLLOUT_STEPS = int(max_rollout_steps)
    _THREAD_USE_NUMBA_KERNEL = bool(use_numba_kernel)
    _THREAD_NUMBA_DATA = _build_numba_model_data(model) if _THREAD_USE_NUMBA_KERNEL else None


def _thread_eval_chunk(task: Tuple[int, ModelAction, str, int, int, int]) -> Tuple[int, int, int, int, int]:
    pair_idx, action, continuation_model_id, k, n_samples, seed = task
    if _THREAD_MODEL is None or _THREAD_STATE is None:
        raise RuntimeError("MC portfolio thread worker was not initialized")

    if _THREAD_USE_NUMBA_KERNEL:
        result = _numba_eval_chunk(task)
        if result is not None:
            idx, successes, used = result
            return idx, successes, used, 1, 0

    rng = np.random.default_rng(int(seed))
    successes = 0
    for _ in range(int(n_samples)):
        next_state = _sample_transition_rng(_THREAD_MODEL, _THREAD_STATE, action, rng)
        if next_state is not None and _rollout_seq_model_k_rng(
            _THREAD_MODEL,
            next_state,
            continuation_model_id,
            int(k),
            _THREAD_MAX_ROLLOUT_STEPS,
            rng,
        ):
            successes += 1
    fallback_chunks = 1 if _THREAD_USE_NUMBA_KERNEL else 0
    return int(pair_idx), int(successes), int(n_samples), 0, int(fallback_chunks)


@dataclass(frozen=True)
class _MCState:
    completed: FrozenSet[int]
    h: float
    b: float
    pending: Tuple[PendingEvent, ...] = ()
    partial_success: FrozenSet[int] = field(default_factory=frozenset)
    pending_node_ids: Optional[FrozenSet[int]] = field(default=None, init=False, compare=False)


@dataclass(frozen=True)
class _NumbaModelData:
    node_ids: Tuple[int, ...]
    model_ids: Tuple[str, ...]
    node_to_idx: Dict[int, int]
    model_to_idx: Dict[str, int]
    topo_indices: np.ndarray
    parent_masks: np.ndarray
    sample_counts: np.ndarray
    sample_latencies: np.ndarray
    sample_successes: np.ndarray
    sample_costs: np.ndarray
    success_prob: np.ndarray


if njit is not None:
    @njit(cache=True)
    def _numba_success_sort_key(success_code):
        return 1 if success_code == -1 else int(success_code)


    @njit(cache=True)
    def _numba_pending_less(rem_a, node_a, model_a, succ_a, rem_b, node_b, model_b, succ_b):
        if rem_a < rem_b:
            return True
        if rem_a > rem_b:
            return False
        if node_a < node_b:
            return True
        if node_a > node_b:
            return False
        if model_a < model_b:
            return True
        if model_a > model_b:
            return False
        return _numba_success_sort_key(succ_a) < _numba_success_sort_key(succ_b)


    @njit(cache=True)
    def _numba_sort_pending(p_count, p_rem, p_node, p_model, p_success):
        for i in range(1, p_count):
            rem = p_rem[i]
            node = p_node[i]
            model = p_model[i]
            success = p_success[i]
            j = i - 1
            while j >= 0 and _numba_pending_less(
                rem,
                node,
                model,
                success,
                p_rem[j],
                p_node[j],
                p_model[j],
                p_success[j],
            ):
                p_rem[j + 1] = p_rem[j]
                p_node[j + 1] = p_node[j]
                p_model[j + 1] = p_model[j]
                p_success[j + 1] = p_success[j]
                j -= 1
            p_rem[j + 1] = rem
            p_node[j + 1] = node
            p_model[j + 1] = model
            p_success[j + 1] = success


    @njit(cache=True)
    def _numba_done(completed_mask, all_done_mask):
        return (completed_mask & all_done_mask) == all_done_mask


    @njit(cache=True)
    def _numba_advance(
        completed_mask,
        partial_mask,
        h,
        p_count,
        p_rem,
        p_node,
        p_model,
        p_success,
        success_prob,
        all_done_mask,
    ):
        while p_count > 0:
            remaining = p_rem[0]
            nid = p_node[0]
            model_idx = p_model[0]
            success_code = p_success[0]
            if remaining > h + 1e-12:
                return -1, completed_mask, partial_mask, h, p_count

            still_running = False
            new_count = p_count - 1
            for i in range(new_count):
                old_nid = p_node[i + 1]
                if old_nid == nid:
                    still_running = True
                p_rem[i] = max(0.0, p_rem[i + 1] - remaining)
                p_node[i] = old_nid
                p_model[i] = p_model[i + 1]
                p_success[i] = p_success[i + 1]
            p_count = new_count

            node_bit = np.int64(1) << nid
            node_completed = (completed_mask & node_bit) != 0
            if success_code == -1:
                success = np.random.random() < success_prob[nid, model_idx]
            else:
                success = success_code == 1
            if (not node_completed) and success:
                partial_mask |= node_bit

            if (not node_completed) and (not still_running) and ((partial_mask & node_bit) != 0):
                completed_mask |= node_bit
                partial_mask &= ~node_bit

            h = max(0.0, h - remaining)
            if _numba_done(completed_mask, all_done_mask) or (not still_running):
                return 1, completed_mask, partial_mask, h, p_count

        return 1, completed_mask, partial_mask, h, p_count


    @njit(cache=True)
    def _numba_sample_transition(
        completed_mask,
        partial_mask,
        h,
        b,
        p_count,
        p_rem,
        p_node,
        p_model,
        p_success,
        action_nodes,
        action_models,
        action_counts,
        sample_counts,
        sample_latencies,
        sample_successes,
        sample_costs,
        success_prob,
        parent_masks,
        topo_indices,
        all_done_mask,
        pending_capacity,
    ):
        cost = 0.0
        max_latency = 0.0
        occupied_mask = completed_mask
        for i in range(p_count):
            occupied_mask |= np.int64(1) << p_node[i]
        seen_mask = np.int64(0)

        for action_idx in range(len(action_nodes)):
            nid = action_nodes[action_idx]
            model_idx = action_models[action_idx]
            count = action_counts[action_idx]
            node_bit = np.int64(1) << nid
            if (seen_mask & node_bit) != 0:
                return -1, completed_mask, partial_mask, h, b, p_count
            seen_mask |= node_bit
            if (occupied_mask & node_bit) != 0:
                return -1, completed_mask, partial_mask, h, b, p_count
            n_emp = sample_counts[nid, model_idx]
            if n_emp <= 0:
                return -1, completed_mask, partial_mask, h, b, p_count
            if p_count + count > pending_capacity:
                return -1, completed_mask, partial_mask, h, b, p_count
            for _ in range(count):
                sample_idx = np.random.randint(0, n_emp)
                latency = sample_latencies[nid, model_idx, sample_idx]
                success = sample_successes[nid, model_idx, sample_idx]
                realized_cost = sample_costs[nid, model_idx, sample_idx]
                cost += realized_cost
                if latency > max_latency:
                    max_latency = latency
                p_rem[p_count] = latency
                p_node[p_count] = nid
                p_model[p_count] = model_idx
                p_success[p_count] = success
                p_count += 1
            occupied_mask |= node_bit

        if cost > b + 1e-12:
            return -1, completed_mask, partial_mask, h, b, p_count
        if max_latency > h + 1e-12:
            return -1, completed_mask, partial_mask, h, b, p_count

        b = max(0.0, b - cost)
        _numba_sort_pending(p_count, p_rem, p_node, p_model, p_success)
        status, completed_mask, partial_mask, h, p_count = _numba_advance(
            completed_mask,
            partial_mask,
            h,
            p_count,
            p_rem,
            p_node,
            p_model,
            p_success,
            success_prob,
            all_done_mask,
        )
        return status, completed_mask, partial_mask, h, b, p_count


    @njit(cache=True)
    def _numba_rollout_one(
        init_completed_mask,
        init_partial_mask,
        init_h,
        init_b,
        init_p_count,
        init_p_rem,
        init_p_node,
        init_p_model,
        init_p_success,
        action_nodes,
        action_models,
        action_counts,
        continuation_model_idx,
        k,
        max_rollout_steps,
        sample_counts,
        sample_latencies,
        sample_successes,
        sample_costs,
        success_prob,
        parent_masks,
        topo_indices,
        all_done_mask,
        pending_capacity,
    ):
        completed_mask = init_completed_mask
        partial_mask = init_partial_mask
        h = init_h
        b = init_b
        p_count = init_p_count
        p_rem = np.empty(pending_capacity, dtype=np.float64)
        p_node = np.empty(pending_capacity, dtype=np.int64)
        p_model = np.empty(pending_capacity, dtype=np.int64)
        p_success = np.empty(pending_capacity, dtype=np.int8)
        for i in range(init_p_count):
            p_rem[i] = init_p_rem[i]
            p_node[i] = init_p_node[i]
            p_model[i] = init_p_model[i]
            p_success[i] = init_p_success[i]

        status, completed_mask, partial_mask, h, b, p_count = _numba_sample_transition(
            completed_mask,
            partial_mask,
            h,
            b,
            p_count,
            p_rem,
            p_node,
            p_model,
            p_success,
            action_nodes,
            action_models,
            action_counts,
            sample_counts,
            sample_latencies,
            sample_successes,
            sample_costs,
            success_prob,
            parent_masks,
            topo_indices,
            all_done_mask,
            pending_capacity,
        )
        if status == -1:
            return 0

        for _step in range(max_rollout_steps):
            if _numba_done(completed_mask, all_done_mask):
                return 1
            if h <= 0.0:
                return 0

            ready_count = 0
            ready_nodes = np.empty(len(topo_indices), dtype=np.int64)
            pending_mask = np.int64(0)
            for i in range(p_count):
                pending_mask |= np.int64(1) << p_node[i]
            for topo_pos in range(len(topo_indices)):
                nid = topo_indices[topo_pos]
                node_bit = np.int64(1) << nid
                if (completed_mask & node_bit) != 0:
                    continue
                if (pending_mask & node_bit) != 0:
                    continue
                if (parent_masks[nid] & completed_mask) != parent_masks[nid]:
                    continue
                ready_nodes[ready_count] = nid
                ready_count += 1

            if ready_count == 0:
                if p_count > 0:
                    status, completed_mask, partial_mask, h, p_count = _numba_advance(
                        completed_mask,
                        partial_mask,
                        h,
                        p_count,
                        p_rem,
                        p_node,
                        p_model,
                        p_success,
                        success_prob,
                        all_done_mask,
                    )
                    if status == -1:
                        return 0
                    continue
                return 0

            cont_nodes = np.empty(ready_count, dtype=np.int64)
            cont_models = np.empty(ready_count, dtype=np.int64)
            cont_counts = np.empty(ready_count, dtype=np.int64)
            for i in range(ready_count):
                nid = ready_nodes[i]
                if sample_counts[nid, continuation_model_idx] <= 0:
                    return 0
                cont_nodes[i] = nid
                cont_models[i] = continuation_model_idx
                cont_counts[i] = k

            status, completed_mask, partial_mask, h, b, p_count = _numba_sample_transition(
                completed_mask,
                partial_mask,
                h,
                b,
                p_count,
                p_rem,
                p_node,
                p_model,
                p_success,
                cont_nodes,
                cont_models,
                cont_counts,
                sample_counts,
                sample_latencies,
                sample_successes,
                sample_costs,
                success_prob,
                parent_masks,
                topo_indices,
                all_done_mask,
                pending_capacity,
            )
            if status == -1:
                return 0
        return 0


    @njit(cache=True)
    def _numba_eval_chunk_kernel(
        seed,
        n_samples,
        init_completed_mask,
        init_partial_mask,
        init_h,
        init_b,
        init_p_count,
        init_p_rem,
        init_p_node,
        init_p_model,
        init_p_success,
        action_nodes,
        action_models,
        action_counts,
        continuation_model_idx,
        k,
        max_rollout_steps,
        sample_counts,
        sample_latencies,
        sample_successes,
        sample_costs,
        success_prob,
        parent_masks,
        topo_indices,
        all_done_mask,
        pending_capacity,
    ):
        successes = 0
        for sample_idx in range(n_samples):
            np.random.seed((int(seed) + sample_idx * 1000003) % 2147483647)
            successes += _numba_rollout_one(
                init_completed_mask,
                init_partial_mask,
                init_h,
                init_b,
                init_p_count,
                init_p_rem,
                init_p_node,
                init_p_model,
                init_p_success,
                action_nodes,
                action_models,
                action_counts,
                continuation_model_idx,
                k,
                max_rollout_steps,
                sample_counts,
                sample_latencies,
                sample_successes,
                sample_costs,
                success_prob,
                parent_masks,
                topo_indices,
                all_done_mask,
                pending_capacity,
            )
        return successes


class _TheoryModel:
    def __init__(self, sim: Simulator, model_ids: Optional[Sequence[str]] = None):
        self.node_ids = tuple(sim.dag.node_ids)
        self.topo_order = tuple(sim.dag.topological_order())
        self.parents: Dict[int, Tuple[int, ...]] = {}
        self.children: Dict[int, Tuple[int, ...]] = {}
        self.tau: Dict[Tuple[int, str], float] = {}
        self.kappa: Dict[Tuple[int, str], float] = {}
        self.p: Dict[Tuple[int, str], float] = {}
        self.latency_dist = sim.latency_dist
        self.budget_cost_mode = sim.budget_cost_mode
        self.statistics_from_template = bool(getattr(sim, "_planner_statistics_from_template", False))
        self.empirical_cache: Dict[Tuple[int, str], tuple] = dict(
            getattr(sim, "_planning_empirical_cache", getattr(sim, "_empirical_cache", {}))
        )
        self.templates: Dict[Tuple[int, str], Any] = {}

        available_by_node: Dict[int, Tuple[str, ...]] = {
            int(nid): tuple(str(model_id) for model_id in sim.model_ids_for_node(nid))
            for nid in self.node_ids
        }
        if model_ids is not None:
            requested = _parse_model_ids(model_ids)
            if not requested:
                raise ValueError("MCPortfolioRolloutPolicy model_ids must contain at least one model id.")
            requested_set = set(requested)
            present_anywhere = set().union(*(set(ids) for ids in available_by_node.values())) if available_by_node else set()
            self.node_model_ids = {
                nid: tuple(model_id for model_id in available if model_id in requested_set)
                for nid, available in available_by_node.items()
            }
            self.model_ids = tuple(model_id for model_id in requested if model_id in present_anywhere)
        else:
            order: List[str] = []
            for nid in self.node_ids:
                for model_id in available_by_node[nid]:
                    if model_id not in order:
                        order.append(model_id)
            self.node_model_ids = dict(available_by_node)
            self.model_ids = tuple(order)
            if not self.model_ids and self.node_ids:
                raise ValueError("No model_id is available on any DAG node for MC portfolio search.")

        for nid in self.node_ids:
            st = sim.runtime[nid]
            self.parents[nid] = tuple(st.parents)
            self.children[nid] = tuple(st.children)
            for model_id in self.node_model_ids[nid]:
                key = (int(nid), str(model_id))
                self.templates[key] = sim.node_template(nid, model_id)
                self.tau[key] = float(max(sim.node_batch_time(nid, model_id), 1e-9))
                self.kappa[key] = float(max(sim.node_work_cost(nid, model_id), 1e-12))
                cached = self.empirical_cache.get(key)
                if cached is not None and not self.statistics_from_template:
                    _lats, succs, _costs, _token_costs = cached
                    self.p[key] = float(min(max(float(np.mean(succs)), 0.0), 1.0))
                else:
                    self.p[key] = float(min(max(sim.node_success_prob(nid, model_id), 0.0), 1.0))

        self._ready_cache: Dict[FrozenSet[int], List[int]] = {}
        self._min_work_cache: Dict[FrozenSet[int], float] = {}
        self._min_time_cache: Dict[FrozenSet[int], float] = {}

    def done(self, completed: FrozenSet[int]) -> bool:
        return len(completed) == len(self.node_ids)

    def ready_nodes(self, completed: FrozenSet[int]) -> List[int]:
        cached = self._ready_cache.get(completed)
        if cached is not None:
            return list(cached)
        ready = [
            nid
            for nid in self.topo_order
            if nid not in completed and all(parent in completed for parent in self.parents[nid])
        ]
        self._ready_cache[completed] = ready
        return list(ready)

    def dispatchable_ready_nodes(self, state: _MCState) -> List[int]:
        pending_nodes = _pending_node_ids(state)
        return [
            nid
            for nid in self.ready_nodes(state.completed)
            if nid not in pending_nodes
        ]

    def model_action_cost(self, action: ModelAction) -> float:
        return float(
            sum(self.kappa[(int(nid), str(model_id))] * count for nid, model_id, count in action)
        )

    def model_action_time(self, action: ModelAction) -> float:
        if not action:
            return 0.0
        return float(max(self.tau[(int(nid), str(model_id))] for nid, model_id, _count in action))

    def action_cost(self, model_id: str | Action, action: Optional[Action] = None) -> float:
        if action is None:
            action = model_id  # type: ignore[assignment]
            model_id = _default_model_for_compat(self)
        model_id = str(model_id)
        total = 0.0
        for nid, count in action:
            nid = int(nid)
            if model_id not in self.node_model_ids.get(nid, ()):
                return float("inf")
            total += self.kappa[(nid, model_id)] * count
        return float(total)

    def action_time(self, model_id: str | Action, action: Optional[Action] = None) -> float:
        if action is None:
            action = model_id  # type: ignore[assignment]
            model_id = _default_model_for_compat(self)
        model_id = str(model_id)
        if not action:
            return 0.0
        values = []
        for nid, _count in action:
            nid = int(nid)
            if model_id not in self.node_model_ids.get(nid, ()):
                return float("inf")
            values.append(self.tau[(nid, model_id)])
        return float(max(values))

    def sample_latency(
        self,
        nid: int,
        model_id: str | np.random.Generator = DEFAULT_MODEL_ID,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        if rng is None:
            rng = model_id  # type: ignore[assignment]
            model_id = _default_model_for_compat(self)
        if rng is None:
            raise ValueError("sample_latency requires an rng")
        model_id = str(model_id)
        tpl = self.templates[(int(nid), model_id)]
        mu = max(float(tpl.lat_mean), 1e-9)
        sigma = max(float(tpl.lat_std), 1e-9)

        if self.latency_dist == "gaussian":
            return float(max(rng.normal(mu, sigma), 1e-3))

        if self.latency_dist == "gamma":
            shape = (mu / sigma) ** 2 if sigma > 0 else 1e6
            scale = (sigma ** 2) / mu if mu > 0 else 1e-3
            return float(max(rng.gamma(shape, scale), 1e-3))

        variance = sigma ** 2
        phi = math.sqrt(variance + mu ** 2)
        sigma_ln = math.sqrt(max(math.log((phi ** 2) / (mu ** 2)), 1e-9))
        mu_ln = math.log((mu ** 2) / phi)
        return float(max(rng.lognormal(mu_ln, sigma_ln), 1e-3))

    def sample_rollout_realization(
        self,
        nid: int,
        model_id: str | np.random.Generator = DEFAULT_MODEL_ID,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, bool, float]:
        if rng is None:
            rng = model_id  # type: ignore[assignment]
            model_id = _default_model_for_compat(self)
        if rng is None:
            raise ValueError("sample_rollout_realization requires an rng")
        model_id = str(model_id)
        nid = int(nid)
        tpl = self.templates[(nid, model_id)]
        cached = self.empirical_cache.get((nid, model_id))
        if cached is not None:
            lats, succs, costs, token_costs = cached
            idx = int(rng.integers(0, len(lats)))
            sampled_latency = float(lats[idx])
            success = bool(succs[idx])
            if self.budget_cost_mode == "lat_mean":
                realized_cost = sampled_latency
            elif self.budget_cost_mode == "token_cost_mean":
                realized_cost = float(token_costs[idx])
            elif self.budget_cost_mode == "output_money_cost_mean":
                realized_cost = output_money_cost_from_tokens(model_id, float(token_costs[idx]))
            else:
                realized_cost = float(costs[idx])
            return sampled_latency, success, realized_cost

        sampled_latency = self.sample_latency(nid, model_id, rng)
        success = bool(rng.random() < self.p[(nid, model_id)])
        realized_cost = realized_rollout_work_cost(
            tpl,
            sampled_latency=sampled_latency,
            budget_cost_mode=self.budget_cost_mode,
            model_id=model_id,
        )
        return sampled_latency, success, realized_cost

    def sample_empirical_rollout_batch(
        self,
        nid: int,
        model_id: str | int,
        count: Optional[int | np.random.Generator] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Tuple[float, float, List[PendingEvent]]]:
        if rng is None:
            # Backward-compatible call shape:
            # sample_empirical_rollout_batch(nid, count, rng)
            rng = count  # type: ignore[assignment]
            count = model_id  # type: ignore[assignment]
            model_id = _default_model_for_compat(self)
        if rng is None or count is None:
            raise ValueError("sample_empirical_rollout_batch requires count and rng")
        model_id = str(model_id)
        nid = int(nid)
        cached = self.empirical_cache.get((nid, model_id))
        if cached is None:
            return None

        count = int(count)
        if count <= 0:
            return 0.0, 0.0, []

        lats, succs, costs, token_costs = cached
        if len(lats) == 0:
            return None
        idxs = rng.integers(0, len(lats), size=count)
        latencies = lats[idxs]
        successes = succs[idxs]

        if self.budget_cost_mode == "lat_mean":
            cost_values = latencies
        elif self.budget_cost_mode == "token_cost_mean":
            cost_values = token_costs[idxs]
        elif self.budget_cost_mode == "output_money_cost_mean":
            cost_values = token_costs[idxs] * (output_price_per_million_tokens(model_id) / 1_000_000.0)
        else:
            cost_values = costs[idxs]

        nid_int = int(nid)
        pending_events = [
            (float(latency), nid_int, model_id, bool(success))
            for latency, success in zip(latencies, successes)
        ]
        return (
            float(np.sum(cost_values, dtype=np.float64)),
            float(np.max(latencies)),
            pending_events,
        )

    def sample_success(
        self,
        nid: int,
        model_id: str | np.random.Generator = DEFAULT_MODEL_ID,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        if rng is None:
            rng = model_id  # type: ignore[assignment]
            model_id = _default_model_for_compat(self)
        if rng is None:
            raise ValueError("sample_success requires an rng")
        model_id = str(model_id)
        return bool(rng.random() < self.p[(int(nid), model_id)])

    def min_remaining_work(self, completed: FrozenSet[int]) -> float:
        cached = self._min_work_cache.get(completed)
        if cached is not None:
            return cached
        value = float(
            sum(
                min(self.kappa[(nid, model_id)] for model_id in self.node_model_ids[nid])
                for nid in self.node_ids
                if nid not in completed
            )
        )
        self._min_work_cache[completed] = value
        return value

    def min_remaining_time(self, completed: FrozenSet[int]) -> float:
        cached = self._min_time_cache.get(completed)
        if cached is not None:
            return cached
        if self.done(completed):
            self._min_time_cache[completed] = 0.0
            return 0.0

        dist: Dict[int, float] = {}
        for nid in reversed(self.topo_order):
            if nid in completed:
                continue
            child_times = [
                dist[ch]
                for ch in self.children[nid]
                if ch in dist and ch not in completed
            ]
            dist[nid] = min(self.tau[(nid, model_id)] for model_id in self.node_model_ids[nid]) + (
                max(child_times) if child_times else 0.0
            )

        roots = self.ready_nodes(completed)
        if not roots:
            self._min_time_cache[completed] = float("inf")
            return float("inf")
        value = float(max(dist.get(nid, 0.0) for nid in roots))
        self._min_time_cache[completed] = value
        return value


def _build_numba_model_data(model: _TheoryModel) -> Optional[_NumbaModelData]:
    if njit is None:
        return None
    if hasattr(model, "_numba_model_data"):
        return getattr(model, "_numba_model_data")

    n_nodes = len(model.node_ids)
    n_models = len(model.model_ids)
    if n_nodes <= 0 or n_models <= 0 or n_nodes > 60:
        setattr(model, "_numba_model_data", None)
        return None

    node_to_idx = {int(nid): idx for idx, nid in enumerate(model.node_ids)}
    model_to_idx = {str(model_id): idx for idx, model_id in enumerate(model.model_ids)}
    topo_indices = np.asarray([node_to_idx[int(nid)] for nid in model.topo_order], dtype=np.int64)
    parent_masks = np.zeros(n_nodes, dtype=np.int64)
    success_prob = np.zeros((n_nodes, n_models), dtype=np.float64)

    max_samples = 0
    for nid in model.node_ids:
        for model_id in model.node_model_ids[int(nid)]:
            cached = model.empirical_cache.get((int(nid), str(model_id)))
            if cached is None or len(cached[0]) == 0:
                setattr(model, "_numba_model_data", None)
                return None
            max_samples = max(max_samples, int(len(cached[0])))
    if max_samples <= 0:
        setattr(model, "_numba_model_data", None)
        return None

    sample_counts = np.zeros((n_nodes, n_models), dtype=np.int64)
    sample_latencies = np.zeros((n_nodes, n_models, max_samples), dtype=np.float64)
    sample_successes = np.zeros((n_nodes, n_models, max_samples), dtype=np.int8)
    sample_costs = np.zeros((n_nodes, n_models, max_samples), dtype=np.float64)

    for nid in model.node_ids:
        nid_int = int(nid)
        node_idx = node_to_idx[nid_int]
        mask = 0
        for parent in model.parents[nid_int]:
            mask |= 1 << node_to_idx[int(parent)]
        parent_masks[node_idx] = mask

        for model_id in model.node_model_ids[nid_int]:
            model_id = str(model_id)
            if model_id not in model_to_idx:
                continue
            model_idx = model_to_idx[model_id]
            cached = model.empirical_cache.get((nid_int, model_id))
            if cached is None:
                continue
            lats, succs, costs, token_costs = cached
            n = int(len(lats))
            if n <= 0:
                continue
            sample_counts[node_idx, model_idx] = n
            sample_latencies[node_idx, model_idx, :n] = np.asarray(lats, dtype=np.float64)
            sample_successes[node_idx, model_idx, :n] = np.asarray(succs, dtype=np.int8)
            if model.budget_cost_mode == "lat_mean":
                cost_values = np.asarray(lats, dtype=np.float64)
            elif model.budget_cost_mode == "token_cost_mean":
                cost_values = np.asarray(token_costs, dtype=np.float64)
            elif model.budget_cost_mode == "output_money_cost_mean":
                cost_values = np.asarray(token_costs, dtype=np.float64) * (
                    output_price_per_million_tokens(model_id) / 1_000_000.0
                )
            else:
                cost_values = np.asarray(costs, dtype=np.float64)
            sample_costs[node_idx, model_idx, :n] = cost_values
            success_prob[node_idx, model_idx] = float(model.p[(nid_int, model_id)])

    data = _NumbaModelData(
        node_ids=tuple(int(nid) for nid in model.node_ids),
        model_ids=tuple(str(model_id) for model_id in model.model_ids),
        node_to_idx=node_to_idx,
        model_to_idx=model_to_idx,
        topo_indices=topo_indices,
        parent_masks=parent_masks,
        sample_counts=sample_counts,
        sample_latencies=sample_latencies,
        sample_successes=sample_successes,
        sample_costs=sample_costs,
        success_prob=success_prob,
    )
    setattr(model, "_numba_model_data", data)
    return data


def _encode_numba_state(
    data: _NumbaModelData,
    state: _MCState,
) -> Optional[Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    completed_mask = 0
    for nid in state.completed:
        idx = data.node_to_idx.get(int(nid))
        if idx is None:
            return None
        completed_mask |= 1 << idx

    partial_mask = 0
    for nid in state.partial_success:
        idx = data.node_to_idx.get(int(nid))
        if idx is None:
            return None
        partial_mask |= 1 << idx

    pending = tuple(_pending_event_parts(event) for event in state.pending)
    p_count = len(pending)
    p_rem = np.empty(p_count, dtype=np.float64)
    p_node = np.empty(p_count, dtype=np.int64)
    p_model = np.empty(p_count, dtype=np.int64)
    p_success = np.empty(p_count, dtype=np.int8)
    for idx, (remaining, nid, model_id, success) in enumerate(pending):
        node_idx = data.node_to_idx.get(int(nid))
        model_idx = data.model_to_idx.get(str(model_id))
        if node_idx is None or model_idx is None:
            return None
        p_rem[idx] = float(remaining)
        p_node[idx] = int(node_idx)
        p_model[idx] = int(model_idx)
        p_success[idx] = -1 if success is None else (1 if bool(success) else 0)

    return (
        int(completed_mask),
        int(partial_mask),
        int(p_count),
        p_rem,
        p_node,
        p_model,
        p_success,
    )


def _encode_numba_action(
    data: _NumbaModelData,
    action: ModelAction,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    action_nodes = np.empty(len(action), dtype=np.int64)
    action_models = np.empty(len(action), dtype=np.int64)
    action_counts = np.empty(len(action), dtype=np.int64)
    for idx, (nid, model_id, count) in enumerate(action):
        node_idx = data.node_to_idx.get(int(nid))
        model_idx = data.model_to_idx.get(str(model_id))
        if node_idx is None or model_idx is None:
            return None
        if data.sample_counts[node_idx, model_idx] <= 0:
            return None
        action_nodes[idx] = int(node_idx)
        action_models[idx] = int(model_idx)
        action_counts[idx] = int(count)
    return action_nodes, action_models, action_counts


def _numba_eval_chunk(task: Tuple[int, ModelAction, str, int, int, int]) -> Optional[Tuple[int, int, int]]:
    if njit is None or _THREAD_MODEL is None or _THREAD_STATE is None or _THREAD_NUMBA_DATA is None:
        return None
    pair_idx, action, continuation_model_id, k, n_samples, seed = task
    data = _THREAD_NUMBA_DATA
    state_encoded = _encode_numba_state(data, _THREAD_STATE)
    action_encoded = _encode_numba_action(data, action)
    continuation_model_idx = data.model_to_idx.get(str(continuation_model_id))
    if state_encoded is None or action_encoded is None or continuation_model_idx is None:
        return None

    (
        completed_mask,
        partial_mask,
        p_count,
        p_rem,
        p_node,
        p_model,
        p_success,
    ) = state_encoded
    action_nodes, action_models, action_counts = action_encoded
    max_count = int(max([int(k), *(int(count) for _nid, _model_id, count in action)] or [int(k)]))
    pending_capacity = max(256, int(p_count) + len(data.node_ids) * max_count + int(np.sum(action_counts)) + 16)
    all_done_mask = (1 << len(data.node_ids)) - 1
    successes = _numba_eval_chunk_kernel(
        int(seed),
        int(n_samples),
        int(completed_mask),
        int(partial_mask),
        float(_THREAD_STATE.h),
        float(_THREAD_STATE.b),
        int(p_count),
        p_rem,
        p_node,
        p_model,
        p_success,
        action_nodes,
        action_models,
        action_counts,
        int(continuation_model_idx),
        int(k),
        int(_THREAD_MAX_ROLLOUT_STEPS),
        data.sample_counts,
        data.sample_latencies,
        data.sample_successes,
        data.sample_costs,
        data.success_prob,
        data.parent_masks,
        data.topo_indices,
        int(all_done_mask),
        int(pending_capacity),
    )
    return int(pair_idx), int(successes), int(n_samples)


class MCPortfolioRolloutPolicy(Policy):
    """
    Monte Carlo portfolio rollout search from mc_portfolio_rollout_search.pdf.

    State is (completed nodes, remaining deadline, remaining budget). Actions
    assign every ready node one model_id and a positive log-scale rollout count
    in {1,2,4,...,128}. For each current model-action and each global
    Seq-(model,k) continuation policy, the policy estimates the true
    deadline-before-success probability by Monte Carlo rollout and executes
    only the selected first action. The outer simulator evaluates the online
    closed-loop policy by replanning after each significant execution event.
    """

    uses_theory_batch_execution = False
    uses_atomic_event_action = True

    def __init__(
        self,
        full_dag: DAG,
        *,
        base_ks: Optional[Sequence[int]] = None,
        rollout_samples: Optional[int] = None,
        confidence_delta: Optional[float] = None,
        max_exact_actions: Optional[int] = None,
        max_candidate_actions: Optional[int] = None,
        max_rollout_steps: Optional[int] = None,
        pair_workers: Optional[int] = None,
        sample_chunks_per_pair: Optional[int] = None,
        random_free_n: Optional[int] = None,
        model_ids: Optional[Sequence[str]] = None,
        seed: int = 0,
    ):
        self.full_dag = full_dag
        self.configured_model_ids = (
            _parse_model_ids(model_ids)
            if model_ids is not None
            else _parse_model_ids(os.environ.get("MC_PORTFOLIO_MODEL_IDS"))
        )
        configured_base_ks = (
            _parse_base_ks(base_ks)
            if base_ks is not None
            else _parse_base_ks(os.environ.get("MC_PORTFOLIO_BASE_KS"))
        )
        self.base_ks = tuple(configured_base_ks or _base_ks())
        if not self.base_ks:
            raise ValueError("MCPortfolioRolloutPolicy requires at least one positive Seq-k value")

        self.rollout_samples = int(rollout_samples or _env_int("MC_PORTFOLIO_M", 64, min_value=1))
        self.confidence_delta = float(
            confidence_delta
            if confidence_delta is not None
            else _env_float("MC_PORTFOLIO_DELTA", 0.05, min_value=1e-12)
        )
        if self.confidence_delta > 1.0:
            raise ValueError("confidence_delta must be in (0, 1]")

        self.random_free_n = int(
            random_free_n
            if random_free_n is not None
            else _env_int("MC_PORTFOLIO_RANDOM_FREE_N", 3, min_value=1)
        )
        n_levels = len(self.base_ks)
        default_max_exact_actions = n_levels ** self.random_free_n
        default_max_candidate_actions = len(self.base_ks) + n_levels ** (self.random_free_n + 1)

        self.max_exact_actions = int(
            max_exact_actions
            if max_exact_actions is not None
            else _env_int("MC_PORTFOLIO_MAX_EXACT_ACTIONS", default_max_exact_actions, min_value=1)
        )
        self.max_candidate_actions = int(
            max_candidate_actions
            if max_candidate_actions is not None
            else _env_int("MC_PORTFOLIO_MAX_CANDIDATE_ACTIONS", default_max_candidate_actions, min_value=1)
        )
        self.max_candidate_actions = max(self.max_candidate_actions, len(self.base_ks))
        self.max_rollout_steps = int(
            max_rollout_steps
            if max_rollout_steps is not None
            else _env_int("MC_PORTFOLIO_MAX_ROLLOUT_STEPS", 10000, min_value=1)
        )
        self.pair_workers = int(
            pair_workers
            if pair_workers is not None
            else _env_int("MC_PORTFOLIO_PAIR_WORKERS", _default_pair_workers(), min_value=1)
        )
        self.sample_chunks_per_pair = int(
            sample_chunks_per_pair
            if sample_chunks_per_pair is not None
            else _env_int("MC_PORTFOLIO_SAMPLE_CHUNKS", 1, min_value=1)
        )
        self.parallel_backend = _env_choice(
            "MC_PORTFOLIO_PARALLEL_BACKEND",
            "thread",
            ("thread", "process"),
        )
        self.use_numba_kernel = (
            _env_choice(
                "MC_PORTFOLIO_NUMBA_KERNEL",
                "off",
                ("off", "on"),
            )
            == "on"
        )
        self.rng = np.random.default_rng(seed)

        self.trace_id: Optional[str] = None
        self.trace_enabled = False
        self.trace_level = "summary"
        self.trace_context: Dict[str, Any] = {}
        self.trace_events: List[Dict[str, Any]] = []

        self.planner_call_count = 0
        self.planner_pair_eval_count = 0
        self.planner_rollout_count = 0
        self.planner_wall_ms = 0.0
        self.backup_wall_ms = 0.0
        self.numba_chunk_hits = 0
        self.numba_chunk_fallbacks = 0
        self.last_numba_data_ready = False
        self.last_best_score: Optional[float] = None
        self.last_best_k: Optional[int] = None
        self.last_model_ids: Optional[Tuple[str, ...]] = None
        self.last_best_current_model: Optional[str] = None
        self.last_best_current_models: Optional[Dict[str, str]] = None
        self.last_best_continuation_model: Optional[str] = None
        self.last_n_model_pairs: Optional[int] = None
        self.last_n_continuation_policies: Optional[int] = None
        self.last_best_action: Optional[Dict[str, int]] = None
        self.last_epsilon: Optional[float] = None
        self.last_candidate_action_count: Optional[int] = None
        self.last_candidate_mode: Optional[str] = None
        self.last_planner_wall_ms: Optional[float] = None
        self.last_backup_wall_ms: Optional[float] = None
        self.last_numba_chunk_hits = 0
        self.last_numba_chunk_fallbacks = 0
        self.last_numba_data_ready = False

    def configure_trace(
        self,
        trace_id: str,
        enabled: bool = False,
        trace_level: str = "theta",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trace_id = str(trace_id)
        self.trace_enabled = bool(enabled)
        self.trace_level = trace_level if trace_level in ("summary", "theta", "theta_full", "iter") else "theta"
        self.trace_context = dict(context or {})
        self.trace_events = []
        self.planner_call_count = 0
        self.planner_pair_eval_count = 0
        self.planner_rollout_count = 0
        self.planner_wall_ms = 0.0
        self.backup_wall_ms = 0.0
        self.numba_chunk_hits = 0
        self.numba_chunk_fallbacks = 0
        self.last_numba_data_ready = False
        self.last_best_score = None
        self.last_best_k = None
        self.last_model_ids = None
        self.last_best_current_model = None
        self.last_best_current_models = None
        self.last_best_continuation_model = None
        self.last_n_model_pairs = None
        self.last_n_continuation_policies = None
        self.last_best_action = None
        self.last_epsilon = None
        self.last_candidate_action_count = None
        self.last_candidate_mode = None
        self.last_planner_wall_ms = None
        self.last_backup_wall_ms = None
        self.last_numba_chunk_hits = 0
        self.last_numba_chunk_fallbacks = 0
        self.last_numba_data_ready = False

    def deadline_trace_summary(self) -> Dict[str, Any]:
        # The deadline_planner_* keys are kept for the existing JSONL analysis
        # pipeline. The mc_portfolio_* keys carry the actual new semantics.
        lower = None
        if self.last_best_score is not None and self.last_epsilon is not None:
            lower = max(0.0, float(self.last_best_score) - 2.0 * float(self.last_epsilon))
        return {
            "deadline_trace_id": self.trace_id,
            "deadline_planner_n_calls": int(self.planner_call_count),
            "deadline_planner_n_theta_evals": int(self.planner_pair_eval_count),
            "deadline_planner_last_best_theta": self.last_best_k,
            "deadline_planner_last_best_score": self.last_best_score,
            "deadline_planner_last_best_lb": lower,
            "deadline_planner_last_best_phi": None,
            "deadline_planner_last_plan_size": self.last_candidate_action_count,
            "mc_portfolio_base_ks": list(self.base_ks),
            "mc_portfolio_model_ids": list(self.last_model_ids or self.configured_model_ids or ()),
            "mc_portfolio_rollout_samples": int(self.rollout_samples),
            "mc_portfolio_pair_evals": int(self.planner_pair_eval_count),
            "mc_portfolio_rollouts": int(self.planner_rollout_count),
            "mc_portfolio_planner_wall_ms": float(self.planner_wall_ms),
            "mc_portfolio_last_planner_wall_ms": self.last_planner_wall_ms,
            "mc_portfolio_backup_wall_ms": float(self.backup_wall_ms),
            "mc_portfolio_last_backup_wall_ms": self.last_backup_wall_ms,
            "mc_portfolio_last_best_k": self.last_best_k,
            "mc_portfolio_last_best_current_model": self.last_best_current_model,
            "mc_portfolio_last_best_current_models": self.last_best_current_models,
            "mc_portfolio_last_best_continuation_model": self.last_best_continuation_model,
            "mc_portfolio_n_model_pairs": self.last_n_model_pairs,
            "mc_portfolio_n_continuation_policies": self.last_n_continuation_policies,
            "mc_portfolio_last_best_score": self.last_best_score,
            "mc_portfolio_last_best_action": self.last_best_action,
            "mc_portfolio_last_epsilon": self.last_epsilon,
            "mc_portfolio_last_candidate_actions": self.last_candidate_action_count,
            "mc_portfolio_last_candidate_mode": self.last_candidate_mode,
            "mc_portfolio_max_exact_actions": int(self.max_exact_actions),
            "mc_portfolio_max_candidate_actions": int(self.max_candidate_actions),
            "mc_portfolio_pair_workers": int(self.pair_workers),
            "mc_portfolio_sample_chunks_per_pair": int(self.sample_chunks_per_pair),
            "mc_portfolio_parallel_backend": self.parallel_backend,
            "mc_portfolio_numba_kernel": bool(self.use_numba_kernel),
            "mc_portfolio_numba_chunk_hits": int(self.numba_chunk_hits),
            "mc_portfolio_numba_chunk_fallbacks": int(self.numba_chunk_fallbacks),
            "mc_portfolio_last_numba_chunk_hits": int(self.last_numba_chunk_hits),
            "mc_portfolio_last_numba_chunk_fallbacks": int(self.last_numba_chunk_fallbacks),
            "mc_portfolio_last_numba_data_ready": bool(self.last_numba_data_ready),
            "mc_portfolio_random_free_n": int(self.random_free_n),
        }

    def pop_deadline_trace_events(self) -> List[Dict[str, Any]]:
        events = self.trace_events
        self.trace_events = []
        return events

    def _finish_planner_timing(self, t0: float) -> float:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.planner_wall_ms += elapsed_ms
        self.last_planner_wall_ms = elapsed_ms
        return elapsed_ms

    def allocate(self, sim: Simulator) -> Dict[Any, int]:
        planner_t0 = time.perf_counter()
        model = _TheoryModel(sim, self.configured_model_ids)
        self.last_model_ids = tuple(model.model_ids)
        self.last_n_model_pairs = len(model.model_ids)
        self.last_n_continuation_policies = len(model.model_ids) * len(self.base_ks)
        pending = _normalize_pending(
            (
                max(0.0, float(ev.finish_time) - float(sim.time)),
                int(ev.node_id),
                str(getattr(ev, "model_id", DEFAULT_MODEL_ID)),
                None,
            )
            for ev in getattr(sim, "event_queue", [])
            if ev.node_id not in sim.completed_nodes
        )
        partial_success = frozenset(
            int(nid)
            for nid, st in sim.runtime.items()
            if (not st.completed) and len(st.running_rollouts) > 0 and st.num_success > 0
        )
        state = _MCState(
            completed=frozenset(sim.completed_nodes),
            h=float(sim.remaining_deadline()),
            b=float(sim.remaining_budget),
            pending=pending,
            partial_success=partial_success,
        )
        call_idx = self.planner_call_count
        self.planner_call_count += 1

        ready = model.dispatchable_ready_nodes(state)
        if (
            not ready
            or state.h <= 0.0
        ):
            self.last_best_score = None
            self.last_best_k = None
            self.last_best_current_model = None
            self.last_best_current_models = None
            self.last_best_continuation_model = None
            self.last_best_action = None
            self.last_epsilon = None
            self.last_candidate_action_count = 0
            self.last_candidate_mode = "pruned_before_action"
            planner_wall_ms = self._finish_planner_timing(planner_t0)
            self._record_trace(
                sim=sim,
                model=model,
                state=state,
                ready=ready,
                call_idx=call_idx,
                actions=[],
                generation_meta={"mode": "pruned_before_action"},
                best_action=None,
                best_continuation_model=None,
                best_k=None,
                best_score=0.0,
                epsilon=None,
                pair_evals=0,
                rollouts_used=0,
                planner_wall_ms=planner_wall_ms,
                backup_wall_ms=None,
            )
            return {}

        candidate_actions, candidate_meta = self._generate_candidate_actions(model, state, ready)
        if not candidate_actions:
            self.last_best_score = None
            self.last_best_k = None
            self.last_best_current_model = None
            self.last_best_current_models = None
            self.last_best_continuation_model = None
            self.last_best_action = None
            self.last_epsilon = None
            self.last_candidate_action_count = 0
            self.last_candidate_mode = candidate_meta.get("mode")
            planner_wall_ms = self._finish_planner_timing(planner_t0)
            self._record_trace(
                sim=sim,
                model=model,
                state=state,
                ready=ready,
                call_idx=call_idx,
                actions=[],
                generation_meta=candidate_meta,
                best_action=None,
                best_continuation_model=None,
                best_k=None,
                best_score=0.0,
                epsilon=None,
                pair_evals=0,
                rollouts_used=0,
                planner_wall_ms=planner_wall_ms,
                backup_wall_ms=None,
            )
            return {}

        # Approximate the Bellman backup over (current action, Seq-k continuation):
        #   (a_model, m_cont, k) = argmax Q_hat_{m_cont,k}(s, a_model).
        n_policies = len(model.model_ids) * len(self.base_ks)
        epsilon = self._hoeffding_epsilon(len(candidate_actions), n_policies)
        backup_t0 = time.perf_counter()
        self.last_numba_data_ready = bool(
            self.use_numba_kernel and _build_numba_model_data(model) is not None
        )
        (
            a_hat,
            m_cont_hat,
            k_hat,
            q_hat,
            pair_evals,
            rollouts_used,
        ) = self._monte_carlo_portfolio_backup(model, state, candidate_actions)
        backup_wall_ms = (time.perf_counter() - backup_t0) * 1000.0
        self.backup_wall_ms += backup_wall_ms
        self.last_backup_wall_ms = backup_wall_ms

        self.planner_pair_eval_count += pair_evals
        self.planner_rollout_count += rollouts_used
        self.last_best_score = None if a_hat is None else float(max(q_hat, 0.0))
        self.last_best_k = k_hat
        self.last_best_current_model = _model_action_legacy_model(a_hat)
        self.last_best_current_models = None if a_hat is None else _model_action_models(a_hat)
        self.last_best_continuation_model = m_cont_hat
        self.last_best_action = None if a_hat is None else _model_action_counts(a_hat)
        self.last_epsilon = epsilon
        self.last_candidate_action_count = len(candidate_actions)
        self.last_candidate_mode = candidate_meta.get("mode")

        planner_wall_ms = self._finish_planner_timing(planner_t0)
        self._record_trace(
            sim=sim,
            model=model,
            state=state,
            ready=ready,
            call_idx=call_idx,
            actions=candidate_actions,
            generation_meta=candidate_meta,
            best_action=a_hat,
            best_continuation_model=m_cont_hat,
            best_k=k_hat,
            best_score=max(q_hat, 0.0) if a_hat is not None else 0.0,
            epsilon=epsilon,
            pair_evals=pair_evals,
            rollouts_used=rollouts_used,
            planner_wall_ms=planner_wall_ms,
            backup_wall_ms=backup_wall_ms,
        )

        if a_hat is None:
            return {}
        return {
            (int(nid), str(model_id)): int(count)
            for nid, model_id, count in a_hat
        }

    def _monte_carlo_portfolio_backup(
        self,
        model: _TheoryModel,
        state: _MCState,
        actions: Sequence[ModelAction],
    ) -> Tuple[Optional[ModelAction], Optional[str], Optional[int], float, int, int]:
        """
        Monte Carlo estimate of the restricted Bellman backup.

        For every tuple (a_model, m_cont, k), estimate
        Q_{m_cont,k}(s, a_model) by first applying the per-node model current
        action and then rolling out the global Seq-(m_cont,k) continuation
        policy. Budget/deadline failure is handled in the sampled transition,
        so over-budget actions can still be traced and scored as value 0
        through the Bellman path.
        """
        pair_meta: Dict[int, Tuple[ModelAction, str, int, float, float, int]] = {}
        pair_evals = 0
        pair_idx = 0

        for action in actions:
            action_cost = model.model_action_cost(action)
            action_time = model.model_action_time(action)
            executable = bool(action)
            if not executable:
                pair_evals += len(model.model_ids) * len(self.base_ks)
                continue
            total_rollouts = sum(count for _nid, _model_id, count in action)
            for continuation_model_id in model.model_ids:
                for k in self.base_ks:
                    pair_meta[pair_idx] = (
                        action,
                        str(continuation_model_id),
                        int(k),
                        action_cost,
                        action_time,
                        total_rollouts,
                    )
                    pair_idx += 1
                    pair_evals += 1

        if not pair_meta:
            return None, None, None, -1.0, pair_evals, 0
        # English notesample_chunks_per_pairEnglish note,forEnglish note,English noteexportEnglish note
        n_chunks = min(max(1, self.sample_chunks_per_pair), self.rollout_samples)
        seed_high = np.iinfo(np.int64).max
        chunk_tasks: List[Tuple[int, ModelAction, str, int, int, int]] = []
        for idx, (
            action,
            continuation_model_id,
            k,
            _action_cost,
            _action_time,
            _total_rollouts,
        ) in pair_meta.items():
            if n_chunks == 1:
                seed = int(self.rng.integers(0, seed_high))
                chunk_tasks.append((
                    idx,
                    action,
                    continuation_model_id,
                    k,
                    self.rollout_samples,
                    seed,
                ))
                continue

            #English note
            base = self.rollout_samples // n_chunks
            remainder = self.rollout_samples % n_chunks
            for chunk_idx in range(n_chunks):
                n_samples = base + (1 if chunk_idx < remainder else 0)
                if n_samples <= 0:
                    continue
                seed = int(self.rng.integers(0, seed_high))
                chunk_tasks.append((
                    idx,
                    action,
                    continuation_model_id,
                    k,
                    n_samples,
                    seed,
                ))

        successes_by_pair = {idx: 0 for idx in pair_meta}
        used_by_pair = {idx: 0 for idx in pair_meta}
        numba_chunk_hits = 0
        numba_chunk_fallbacks = 0

        def collect_result(result):
            nonlocal numba_chunk_hits, numba_chunk_fallbacks
            idx, successes, used, chunk_hits, chunk_fallbacks = result
            successes_by_pair[idx] += successes
            used_by_pair[idx] += used
            numba_chunk_hits += int(chunk_hits)
            numba_chunk_fallbacks += int(chunk_fallbacks)

        def run_with_executor(executor_cls):
            with executor_cls(
                max_workers=self.pair_workers,
                initializer=_thread_eval_initializer,
                initargs=(model, state, self.max_rollout_steps, self.use_numba_kernel),
            ) as executor:
                futures = [executor.submit(_thread_eval_chunk, task) for task in chunk_tasks]
                for future in as_completed(futures):
                    collect_result(future.result())

        if self.pair_workers <= 1:
            _thread_eval_initializer(model, state, self.max_rollout_steps, self.use_numba_kernel)
            for task in chunk_tasks:
                collect_result(_thread_eval_chunk(task))
        elif self.parallel_backend == "process":
            try:
                run_with_executor(ProcessPoolExecutor)
            except (NotImplementedError, OSError, PermissionError):
                run_with_executor(ThreadPoolExecutor)
        else:
            run_with_executor(ThreadPoolExecutor)

        self.last_numba_chunk_hits = int(numba_chunk_hits)
        self.last_numba_chunk_fallbacks = int(numba_chunk_fallbacks)
        self.numba_chunk_hits += int(numba_chunk_hits)
        self.numba_chunk_fallbacks += int(numba_chunk_fallbacks)

        best_action: Optional[ModelAction] = None
        best_continuation_model: Optional[str] = None
        best_k: Optional[int] = None
        best_score = -1.0
        best_cost = float("inf")
        best_time = float("inf")
        best_rollouts = float("inf")
        rollouts_used = 0

        for idx in sorted(pair_meta):
            (
                action,
                continuation_model_id,
                k,
                action_cost,
                action_time,
                total_rollouts,
            ) = pair_meta[idx]
            used = used_by_pair[idx]
            rollouts_used += used
            score = successes_by_pair[idx] / max(used, 1)
            if self._is_better(
                float(score),
                action_cost,
                action_time,
                total_rollouts,
                k,
                best_score,
                best_cost,
                best_time,
                best_rollouts,
                best_k,
            ):
                best_score = float(score)
                best_action = action
                best_continuation_model = str(continuation_model_id)
                best_k = int(k)
                best_cost = action_cost
                best_time = action_time
                best_rollouts = total_rollouts

        return best_action, best_continuation_model, best_k, best_score, pair_evals, rollouts_used

    def _generate_candidate_actions(
        self,
        model: _TheoryModel,
        state: _MCState,
        ready: Sequence[int],
    ) -> Tuple[List[ModelAction], Dict[str, Any]]:
        ordered: Dict[ModelAction, None] = {}

        def add(action: ModelAction) -> None:
            if action:
                ordered.setdefault(action, None)

        def first_model(nid: int) -> str:
            return str(model.node_model_ids[int(nid)][0])

        unavailable_ready = [
            int(nid)
            for nid in ready
            if not model.node_model_ids.get(int(nid), ())
        ]
        if unavailable_ready:
            return [], {
                "mode": "no_available_model_for_ready_node",
                "candidate_gap_known": True,
                "ready_width": int(len(ready)),
                "unavailable_ready_nodes": unavailable_ready,
                "configured_model_ids": list(self.configured_model_ids or ()),
                "available_model_ids_on_dag": list(model.model_ids),
            }

        # Required for portfolio safe improvement: every Seq-k current action
        # is present as a candidate. Budget failure is evaluated in the
        # transition, not during candidate generation.
        for k in self.base_ks:
            add(_normalize_model_action((nid, first_model(nid), k) for nid in ready))
            for model_id in model.model_ids:
                if all(model_id in model.node_model_ids[int(nid)] for nid in ready):
                    add(_normalize_model_action((nid, model_id, k) for nid in ready))

        # readyEnglish note;rEnglish notebudgetEnglish notebudgetEnglish note
        levels_by_node: List[List[Tuple[str, int]]] = []
        n_full = 1
        for nid in ready:
            levels = [
                (str(model_id), int(k))
                for model_id in model.node_model_ids[int(nid)]
                for k in self.base_ks
            ]
            levels_by_node.append(levels)
            n_full *= len(levels)
            if n_full > self.max_exact_actions:
                break

        if (
            len(ready) <= self.random_free_n
            and n_full <= self.max_exact_actions
            and len(levels_by_node) == len(ready)
        ):
            for choices in product(*levels_by_node):
                add(_normalize_model_action(
                    (nid, model_id, count)
                    for nid, (model_id, count) in zip(ready, choices)
                ))
            return list(ordered.keys()), {
                "mode": "full_log_scale",
                "full_action_count": int(n_full),
                "candidate_gap_known": True,
                "max_exact_actions": int(self.max_exact_actions),
                "random_free_n": int(self.random_free_n),
            }

        # Candidate-subset path for wide ready frontiers. To avoid injecting a
        # structural priority prior, sample a small free set whose log-scale
        # choices are enumerated independently. All other ready nodes share one
        # positive common background level when feasible as a group.
        ready_list = list(ready)
        free_width = min(len(ready_list), max(1, int(self.random_free_n)))
        free_nodes = tuple(
            int(nid)
            for nid in self.rng.choice(ready_list, size=free_width, replace=False)
        )
        free_set = set(free_nodes)
        fixed_nodes = tuple(int(nid) for nid in ready_list if nid not in free_set)

        free_levels_by_node: List[List[Tuple[str, int]]] = []
        free_space_count = 1
        for nid in free_nodes:
            levels = [
                (str(model_id), int(k))
                for model_id in model.node_model_ids[int(nid)]
                for k in self.base_ks
            ]
            free_levels_by_node.append(levels)
            free_space_count *= len(levels)

        if fixed_nodes:
            shared_levels = list(self.base_ks)
        else:
            shared_levels = [self.base_ks[0]]

        random_subspace_count = 0
        for shared_k in shared_levels:
            fixed_part = [(nid, first_model(nid), shared_k) for nid in fixed_nodes]
            for free_choices in product(*free_levels_by_node):
                action = _normalize_model_action(
                    list(fixed_part)
                    + [
                        (nid, model_id, count)
                        for nid, (model_id, count) in zip(free_nodes, free_choices)
                    ]
                )
                before = len(ordered)
                add(action)
                if len(ordered) > before:
                    random_subspace_count += 1

        actions = list(ordered.keys())
        truncated = len(actions) > self.max_candidate_actions
        if len(actions) > self.max_candidate_actions:
            # Base actions were inserted first; preserve them and trim only the
            # heuristic tail to keep the safe-improvement candidates.
            actions = actions[: self.max_candidate_actions]

        return actions, {
            "mode": "random_free_shared_level",
            "candidate_gap_known": False,
            "max_exact_actions": int(self.max_exact_actions),
            "max_candidate_actions": int(self.max_candidate_actions),
            "ready_width": int(len(ready)),
            "random_free_n_requested": int(self.random_free_n),
            "random_free_n_used": int(free_width),
            "random_free_node_ids": [int(nid) for nid in free_nodes],
            "fixed_node_count": int(len(fixed_nodes)),
            "shared_levels": [int(k) for k in shared_levels],
            "free_space_count": int(free_space_count),
            "random_subspace_actions": int(random_subspace_count),
            "truncated_by_max_candidate_actions": bool(truncated),
            "candidate_actions": int(len(actions)),
        }

    def _hoeffding_epsilon(self, n_actions: int, n_policies: int) -> float:
        n_pairs = max(1, int(n_actions) * int(n_policies))
        delta = min(max(self.confidence_delta, 1e-12), 1.0)
        return float(math.sqrt(math.log(2.0 * n_pairs / delta) / (2.0 * self.rollout_samples)))

    @staticmethod
    def _is_better(
        score: float,
        cost: float,
        delta: float,
        total_rollouts: int,
        k: int,
        best_score: float,
        best_cost: float,
        best_delta: float,
        best_rollouts: float,
        best_k: Optional[int],
    ) -> bool:
        tol = 1e-12
        return score > best_score + tol

    def _record_trace(
        self,
        *,
        sim: Simulator,
        model: _TheoryModel,
        state: _MCState,
        ready: Sequence[int],
        call_idx: int,
        actions: Sequence[ModelAction],
        generation_meta: Dict[str, Any],
        best_action: Optional[ModelAction],
        best_continuation_model: Optional[str],
        best_k: Optional[int],
        best_score: float,
        epsilon: Optional[float],
        pair_evals: int,
        rollouts_used: int,
        planner_wall_ms: float,
        backup_wall_ms: Optional[float],
    ) -> None:
        if not self.trace_enabled:
            return

        legacy_theta_evals = []
        for k in self.base_ks:
            legacy_theta_evals.append({
                "theta": float(k),
                "seq_k": int(k),
                "feasible": True,
                "selected": best_k == k,
                "n_iters": 0,
                "subproblem_bisect_iters_total": 0,
                "subproblem_bisect_iters_max": 0,
                "subproblem_bracket_iters_total": 0,
                "subproblem_bracket_iters_max": 0,
            })

        remaining_nodes = len(model.node_ids) - len(state.completed)
        event = {
            **self.trace_context,
            "trace_id": self.trace_id,
            "planner_call_idx": int(call_idx),
            "sim_time": float(sim.time),
            "remaining_deadline": float(state.h),
            "remaining_budget": float(state.b),
            "completed_nodes": int(len(state.completed)),
            "remaining_nodes": int(remaining_nodes),
            "ready_nodes": [int(nid) for nid in ready],
            "mc_base_ks": [int(k) for k in self.base_ks],
            "mc_model_ids": [str(model_id) for model_id in model.model_ids],
            "mc_n_model_pairs": int(len(model.model_ids)),
            "mc_n_continuation_policies": int(len(model.model_ids) * len(self.base_ks)),
            "mc_rollout_samples": int(self.rollout_samples),
            "mc_candidate_action_count": int(len(actions)),
            "mc_candidate_generation": dict(generation_meta),
            "mc_pair_evals": int(pair_evals),
            "mc_rollouts_used": int(rollouts_used),
            "mc_planner_wall_ms": float(planner_wall_ms),
            "mc_backup_wall_ms": None if backup_wall_ms is None else float(backup_wall_ms),
            "mc_pair_workers": int(self.pair_workers),
            "mc_sample_chunks_per_pair": int(self.sample_chunks_per_pair),
            "mc_parallel_backend": self.parallel_backend,
            "mc_numba_kernel": bool(self.use_numba_kernel),
            "mc_numba_chunk_hits": int(self.numba_chunk_hits),
            "mc_numba_chunk_fallbacks": int(self.numba_chunk_fallbacks),
            "mc_last_numba_chunk_hits": int(self.last_numba_chunk_hits),
            "mc_last_numba_chunk_fallbacks": int(self.last_numba_chunk_fallbacks),
            "mc_last_numba_data_ready": bool(self.last_numba_data_ready),
            "mc_best_score": float(best_score),
            "mc_best_current_model": _model_action_legacy_model(best_action),
            "mc_best_current_models": None if best_action is None else _model_action_models(best_action),
            "mc_best_continuation_model": best_continuation_model,
            "mc_best_k": None if best_k is None else int(best_k),
            "mc_best_action": None if best_action is None else _model_action_counts(best_action),
            "mc_hoeffding_epsilon": epsilon,
            "mc_safe_improvement_lb_estimate": (
                None if epsilon is None else max(0.0, float(best_score) - 2.0 * float(epsilon))
            ),
            "theta_evals": legacy_theta_evals,
            "n_theta_evals": int(len(legacy_theta_evals)),
            "n_feasible_theta_evals": int(len(legacy_theta_evals)),
            "selected_theta": None if best_k is None else float(best_k),
            "selected_current_model": _model_action_legacy_model(best_action),
            "selected_current_models": None if best_action is None else _model_action_models(best_action),
            "selected_continuation_model": best_continuation_model,
            "selected_score": float(best_score),
            "selected_lb": None if epsilon is None else max(0.0, float(best_score) - 2.0 * float(epsilon)),
            "selected_phi": None,
            "selected_plan_size": int(len(actions)),
        }
        if self.trace_level in ("theta_full", "iter"):
            event["mc_candidate_actions"] = [
                {
                    str(nid): {
                        "model_id": str(model_id),
                        "count": int(count),
                    }
                    for nid, model_id, count in action
                }
                for action in actions[: min(len(actions), 128)]
            ]
        self.trace_events.append(event)
