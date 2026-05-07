import math
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from .dag import DAG
from .types import (
    DEFAULT_MODEL_ID,
    RunningRollout,
    NodeRuntimeState,
    SimulationResult,
    expected_node_work_cost,
    get_model_ids_for_node,
    get_model_templates_for_node,
    normalize_budget_cost_mode,
    output_money_cost_from_tokens,
    pricing_model_id_for_node,
    realized_rollout_work_cost,
)


def _build_empirical_cache(dag: DAG) -> Dict[Tuple[int, str], tuple]:
    cache: Dict[Tuple[int, str], tuple] = {}
    for nid, node in dag.nodes.items():
        sample_groups: Dict[str, List[dict]] = {}
        if node.empirical_samples:
            sample_groups[DEFAULT_MODEL_ID] = node.empirical_samples
        if node.empirical_samples_by_model:
            for model_id, samples in node.empirical_samples_by_model.items():
                sample_groups[str(model_id)] = samples
        for model_id, samples in sample_groups.items():
            if not samples:
                continue
            lats = np.array([s["latency"] for s in samples], dtype=np.float64)
            succs = np.array([s["success"] for s in samples], dtype=np.int8)
            costs = np.array(
                [s.get("cost", s.get("token_cost", s["latency"])) for s in samples],
                dtype=np.float64,
            )
            token_costs = np.array(
                [
                    s.get("token_cost", s.get("total_tokens", s.get("cost", s["latency"])))
                    for s in samples
                ],
                dtype=np.float64,
            )
            lats = np.maximum(lats, 1e-3)
            costs = np.maximum(costs, 1e-9)
            token_costs = np.maximum(token_costs, 1e-9)
            cache[(int(nid), str(model_id))] = (lats, succs, costs, token_costs)
    return cache


class Simulator:
    """
    Deadline-aware work-budget simulator.

    Time unit:
        wall-clock latency sampled per rollout

    Work-budget unit:
        realized rollout cost charged at launch time

    Each rollout:
      - samples a latency L
      - consumes work budget equal to the sampled cost immediately when launched
      - finishes after L in wall-clock time
      - succeeds with Bernoulli(success_prob)

    This matches the deadline-aware surrogate more closely:
      - time and work are distinct dimensions
      - deadline is enforced on wall-clock time
      - hard budget is enforced online on realized launched work

    English note:
      - English note DAG English note(lat_mean, lat_std, success_prob)
      - English note rollout English note/English note
      - English note,English note

    Empirical English note:
      - English note DAG English note empirical_samples English note,English note
        (latency, success, cost) English note,English note
      - template English note policy English note(English note,English note)
      - English note → English note
    """

    def __init__(
        self,
        dag: DAG,
        budget: float,
        deadline: float,
        seed: int = 0,
        latency_dist: str = "lognormal",
        execution_dag: DAG = None,
        node_budget_allocations: Optional[Dict[int, float]] = None,
        budget_cost_mode: str = "token_cost_mean",
    ):
        self.dag = dag
        self.budget = float(budget)
        self.remaining_budget = float(budget)
        self.deadline = float(deadline)
        self.rng = np.random.default_rng(seed)
        self.latency_dist = latency_dist
        self.budget_cost_mode = normalize_budget_cost_mode(budget_cost_mode)

        # English note:English note execution_dag English note,
        # English note sim.runtime[nid].template English note dag(English note)English note,
        # English note(start_rollout)English note execution_dag English note.
        # English note execution_dag English note None English note,English note.
        self._exec_templates: Dict[Tuple[int, str], Any] = {}
        if execution_dag is not None:
            for nid, node in execution_dag.nodes.items():
                for model_id, template in get_model_templates_for_node(node).items():
                    self._exec_templates[(int(nid), str(model_id))] = template
                if node.assigned_template is not None and (int(nid), DEFAULT_MODEL_ID) not in self._exec_templates:
                    self._exec_templates[(int(nid), DEFAULT_MODEL_ID)] = node.assigned_template

        self._node_budget_remaining: Optional[Dict[int, float]] = None
        if node_budget_allocations is not None:
            self._node_budget_remaining = {
                int(nid): float(max(amount, 0.0))
                for nid, amount in node_budget_allocations.items()
            }

        self.time = 0.0
        self.rollout_counter = 0
        self.planner_calls = 0
        self.planner_wall_ms = 0.0
        self.termination_reason = "running"
        self.last_start_rollout_failure_reason: Optional[str] = None

        # English note empirical English note(numpy English note,English note).
        # execution cache English note;planning cache English note MC planner English note
        # empirical rollout English note.run_experiment English note execution_dag,English note
        # English note cache,English note.robustness English note DAG English note,planner
        # English note planning pool English note,execution English note true DAG English note.
        empirical_source = execution_dag if execution_dag is not None else dag
        self._empirical_cache: Dict[Tuple[int, str], tuple] = _build_empirical_cache(empirical_source)
        self._planning_empirical_cache: Dict[Tuple[int, str], tuple] = (
            self._empirical_cache if execution_dag is None else _build_empirical_cache(dag)
        )
        self._planner_statistics_from_template = False

        self.runtime: Dict[int, NodeRuntimeState] = {}
        for nid, node in dag.nodes.items():
            self.runtime[nid] = NodeRuntimeState(
                node_id=nid,
                template=node.assigned_template,
                parents=list(node.parents),
                children=list(node.children),
                unlocked=(len(node.parents) == 0),
            )

        self.completed_nodes: Set[int] = set()
        self.event_queue: List[RunningRollout] = []

    def remaining_deadline(self) -> float:
        return max(0.0, self.deadline - self.time)

    def deadline_reached(self) -> bool:
        return self.time >= self.deadline

    def next_event_time(self) -> Optional[float]:
        if not self.event_queue:
            return None
        return float(self.event_queue[0].finish_time)

    def ready_nodes(self) -> List[int]:
        out = []
        for nid, st in self.runtime.items():
            if st.completed:
                continue
            if not st.unlocked:
                continue
            # Event-driven replanning should only allocate to dispatchable
            # nodes. Nodes with in-flight rollouts stay in the execution state
            # but are not part of the current action set.
            if len(st.running_rollouts) > 0:
                continue
            out.append(nid)
        return out

    def running_count(self, node_id: int) -> int:
        return len(self.runtime[node_id].running_rollouts)

    def node_delta_to_round_end(self, node_id: int) -> float:
        st = self.runtime[node_id]
        if len(st.running_rollouts) == 0:
            return 0.0
        finish_times = [
            ev.finish_time
            for ev in self.event_queue
            if ev.node_id == node_id and ev.rollout_id in st.running_rollouts
        ]
        if not finish_times:
            return 0.0
        return max(0.0, max(finish_times) - self.time)

    def sample_latency(self, mu: float, sigma: float) -> float:
        sigma = max(sigma, 1e-9)

        if self.latency_dist == "gaussian":
            x = self.rng.normal(mu, sigma)
            return float(max(x, 1e-3))

        if self.latency_dist == "gamma":
            shape = (mu / sigma) ** 2 if sigma > 0 else 1e6
            scale = (sigma ** 2) / mu if mu > 0 else 1e-3
            return float(max(self.rng.gamma(shape, scale), 1e-3))

        # lognormal
        variance = sigma ** 2
        phi = math.sqrt(variance + mu ** 2)
        sigma_ln = math.sqrt(max(math.log((phi ** 2) / (mu ** 2)), 1e-9))
        mu_ln = math.log((mu ** 2) / phi)
        x = self.rng.lognormal(mu_ln, sigma_ln)
        return float(max(x, 1e-3))

    def model_ids_for_node(self, node_id: int) -> Tuple[str, ...]:
        return get_model_ids_for_node(self.dag.nodes[int(node_id)])

    def all_model_ids(self) -> Tuple[str, ...]:
        common: Optional[Set[str]] = None
        order: List[str] = []
        for nid in self.dag.node_ids:
            ids = tuple(str(model_id) for model_id in self.model_ids_for_node(nid))
            if common is None:
                order = list(ids)
                common = set(ids)
            else:
                common &= set(ids)
        if common is None:
            return tuple()
        out = tuple(model_id for model_id in order if model_id in common)
        if not out and self.dag.node_ids:
            raise ValueError("No common model_id is available across all DAG nodes.")
        return out

    def node_template(self, node_id: int, model_id: str = DEFAULT_MODEL_ID):
        """Template visible to policies/planners."""
        node_id = int(node_id)
        model_id = str(model_id)
        node = self.dag.nodes[node_id]
        model_templates = get_model_templates_for_node(node)
        if model_id in model_templates:
            return model_templates[model_id]
        if model_id == DEFAULT_MODEL_ID and node.assigned_template is not None:
            return node.assigned_template
        available = ", ".join(model_templates.keys()) or "<none>"
        raise ValueError(
            f"Node {node_id} is missing statistics for model_id={model_id!r}. "
            f"Available model_ids: {available}"
        )

    def _execution_template(self, node_id: int, model_id: str = DEFAULT_MODEL_ID):
        """Template used by the physical execution sampler."""
        node_id = int(node_id)
        model_id = str(model_id)
        key = (node_id, model_id)
        if key in self._exec_templates:
            return self._exec_templates[key]
        return self.node_template(node_id, model_id)

    def can_launch_any_rollout(self) -> bool:
        """
        Pre-launch budget feasibility proxy.

        The actual charged cost is the sampled cost drawn in start_rollout().
        Before launching we do not know that realization, so this check uses the
        execution template's mean cost as a proxy for "likely affordable".
        """
        ready = self.ready_nodes()
        for nid in ready:
            st = self.runtime[nid]
            if st.completed:
                continue
            model_ids = self.model_ids_for_node(nid) or (pricing_model_id_for_node(self.dag.nodes[nid]),)
            for model_id in model_ids:
                work_cost = self.node_work_cost(nid, model_id=model_id)
                if self.remaining_budget < work_cost:
                    continue
                if self._node_budget_remaining is not None:
                    node_remaining = self._node_budget_remaining.get(nid, 0.0)
                    if node_remaining < work_cost:
                        continue
                    return True
                return True
        return False

    def node_work_cost(self, node_id: int, model_id: str = DEFAULT_MODEL_ID) -> float:
        node_id = int(node_id)
        model_id = str(model_id)
        pricing_model_id = (
            pricing_model_id_for_node(self.dag.nodes[node_id])
            if self.budget_cost_mode == "output_money_cost_mean" and model_id == DEFAULT_MODEL_ID
            else model_id
        )
        tpl = self.node_template(node_id, model_id)
        cache_key = (node_id, model_id)
        planning_cache = self._planning_empirical_cache
        if (not self._planner_statistics_from_template) and cache_key in planning_cache:
            lats, _succs, costs, token_costs = planning_cache[cache_key]
            if self.budget_cost_mode == "lat_mean":
                return float(max(np.mean(lats), 1e-9))
            if self.budget_cost_mode == "token_cost_mean":
                token_cost_mean = getattr(tpl, "token_cost_mean", None)
                if token_cost_mean is not None:
                    return float(max(float(token_cost_mean), 1e-9))
                return float(max(np.mean(token_costs), 1e-9))
            if self.budget_cost_mode == "output_money_cost_mean":
                token_cost_mean = getattr(tpl, "token_cost_mean", None)
                if token_cost_mean is not None:
                    return output_money_cost_from_tokens(pricing_model_id, token_cost_mean)
                return output_money_cost_from_tokens(pricing_model_id, float(max(np.mean(token_costs), 1e-9)))
            return float(max(np.mean(costs), 1e-9))
        return expected_node_work_cost(
            tpl,
            budget_cost_mode=self.budget_cost_mode,
            model_id=pricing_model_id,
        )

    def node_batch_time(self, node_id: int, model_id: str = DEFAULT_MODEL_ID) -> float:
        """Deterministic per-attempt duration used by theory batch policies."""
        tpl = self.node_template(node_id, model_id)
        return float(max(tpl.lat_mean, 1e-9))

    def node_success_prob(self, node_id: int, model_id: str = DEFAULT_MODEL_ID) -> float:
        """Single-attempt success probability used by theory batch policies."""
        tpl = self.node_template(node_id, model_id)
        return float(min(max(tpl.success_prob, 0.0), 1.0))

    def theory_min_remaining_work(self) -> float:
        """
        Mean-cost lower estimate of work required to finish under the PDF batch
        model.

        Every unfinished node must receive at least one rollout before the DAG
        can complete. Under stochastic token-cost execution this is not a hard
        impossibility certificate, because realized rollout costs may fall below
        their means; MC planning therefore does not use it as a pre-planning
        hard prune.
        """
        total = 0.0
        for nid, st in self.runtime.items():
            if not st.completed:
                total += self.node_work_cost(nid)
        return float(total)

    def theory_min_remaining_time(self) -> float:
        """
        Critical-path lower bound under deterministic batch durations.

        This assumes every remaining node succeeds on its first future batch and
        all currently ready nodes can be run in parallel, so exceeding the
        remaining deadline proves failure before any Monte Carlo work is spent.
        """
        if self.done():
            return 0.0

        order = self.dag.topological_order()[::-1]
        dist: Dict[int, float] = {}
        for nid in order:
            st = self.runtime[nid]
            if st.completed:
                continue
            child_times = [
                dist[ch]
                for ch in st.children
                if ch in dist and not self.runtime[ch].completed
            ]
            dist[nid] = self.node_batch_time(nid) + (max(child_times) if child_times else 0.0)

        roots = [
            nid
            for nid, st in self.runtime.items()
            if not st.completed and all(self.runtime[p].completed for p in st.parents)
        ]
        if not roots:
            return float("inf")
        return float(max(dist.get(nid, 0.0) for nid in roots))

    def _sample_rollout_realization(
        self,
        node_id: int,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> Tuple[float, bool, float]:
        """
        Sample one concrete rollout outcome from the execution distribution.

        This is the shared execution sampler for event-driven rollouts and
        synchronous MC batches. In empirical mode, latency, success and cost are
        resampled from the same observed row.
        """
        node_id = int(node_id)
        model_id = str(model_id)
        exec_tpl = self._execution_template(node_id, model_id)

        cache_key = (node_id, model_id)
        if cache_key in self._empirical_cache:
            lats, succs, costs, token_costs = self._empirical_cache[cache_key]
            idx = self.rng.integers(0, len(lats))
            sampled_latency = float(lats[idx])
            success = bool(succs[idx])
            empirical_sample = {
                "latency": sampled_latency,
                "success": int(success),
                "cost": float(costs[idx]),
                "token_cost": float(token_costs[idx]),
            }
            pricing_model_id = (
                pricing_model_id_for_node(self.dag.nodes[node_id])
                if self.budget_cost_mode == "output_money_cost_mean" and model_id == DEFAULT_MODEL_ID
                else model_id
            )
            realized_compute_cost = realized_rollout_work_cost(
                exec_tpl,
                sampled_latency=sampled_latency,
                empirical_sample=empirical_sample,
                budget_cost_mode=self.budget_cost_mode,
                model_id=pricing_model_id,
            )
            return sampled_latency, success, realized_compute_cost

        sampled_latency = self.sample_latency(exec_tpl.lat_mean, exec_tpl.lat_std)
        success = bool(self.rng.random() < exec_tpl.success_prob)
        pricing_model_id = (
            pricing_model_id_for_node(self.dag.nodes[node_id])
            if self.budget_cost_mode == "output_money_cost_mean" and model_id == DEFAULT_MODEL_ID
            else model_id
        )
        realized_compute_cost = realized_rollout_work_cost(
            exec_tpl,
            sampled_latency=sampled_latency,
            budget_cost_mode=self.budget_cost_mode,
            model_id=pricing_model_id,
        )
        return sampled_latency, success, realized_compute_cost

    def execute_theory_batch_action(self, action: Dict[Any, int]) -> bool:
        """
        Execute one synchronous Bellman batch action from the theory note.

        The action maps every ready node id -> positive rollout count. Missing
        ready nodes and non-positive counts are rejected because they encode a
        per-node zero action. Each requested rollout is sampled from the same
        execution distribution used by start_rollout(); the batch then charges
        realized costs and advances by the maximum sampled latency.
        """
        self.last_start_rollout_failure_reason = None
        if self.done():
            self.termination_reason = "completed"
            return True
        if self.deadline_reached():
            self.last_start_rollout_failure_reason = "deadline_exhausted"
            self.termination_reason = "deadline_exhausted"
            return False

        ready = set(self.ready_nodes())
        by_node: Dict[int, Tuple[str, int]] = {}
        for key, raw_count in action.items():
            nid, model_id = self._parse_action_key(key)
            if nid not in ready:
                continue
            count = int(raw_count)
            if count <= 0:
                continue
            if nid in by_node:
                self.last_start_rollout_failure_reason = "duplicate_node_model_action"
                self.termination_reason = "stalled"
                return False
            by_node[nid] = (model_id, count)

        if any(nid not in by_node for nid in ready):
            self.last_start_rollout_failure_reason = "zero_node_action"
            self.termination_reason = "stalled"
            return False

        if not by_node:
            self.last_start_rollout_failure_reason = "all_zero_action"
            self.termination_reason = "stalled"
            return False

        batch_cost = 0.0
        batch_time = 0.0
        costs_by_node: Dict[int, float] = {}
        successes_by_node: Dict[int, int] = {}
        for nid, (model_id, count) in by_node.items():
            node_cost = 0.0
            node_successes = 0
            for _ in range(count):
                sampled_latency, success, realized_cost = self._sample_rollout_realization(nid, model_id)
                batch_time = max(batch_time, sampled_latency)
                node_cost += realized_cost
                if success:
                    node_successes += 1
            costs_by_node[nid] = node_cost
            successes_by_node[nid] = node_successes
            batch_cost += node_cost

        if batch_time > self.remaining_deadline() + 1e-12:
            self.last_start_rollout_failure_reason = "deadline_exhausted"
            self.time = self.deadline
            self.termination_reason = "deadline_exhausted"
            return False

        if batch_cost > self.remaining_budget + 1e-12:
            self.last_start_rollout_failure_reason = "budget_exhausted"
            self.termination_reason = "budget_exhausted"
            return False

        if self._node_budget_remaining is not None:
            for nid, node_cost in costs_by_node.items():
                if node_cost > self._node_budget_remaining.get(nid, 0.0) + 1e-12:
                    self.last_start_rollout_failure_reason = "budget_exhausted"
                    self.termination_reason = "budget_exhausted"
                    return False

        self.remaining_budget -= batch_cost
        if self._node_budget_remaining is not None:
            for nid, node_cost in costs_by_node.items():
                self._node_budget_remaining[nid] -= node_cost
        successes: List[int] = []
        for nid, (_model_id, count) in by_node.items():
            st = self.runtime[nid]
            st.cumulative_cost += costs_by_node[nid]
            st.num_started += count
            st.num_finished += count
            self.rollout_counter += count

            n_success = successes_by_node[nid]
            st.num_success += n_success
            if n_success > 0:
                successes.append(nid)

        self.time += batch_time
        for nid in successes:
            self.mark_node_complete(nid)

        if self.done():
            self.termination_reason = "completed"
        elif self.deadline_reached():
            self.termination_reason = "deadline_exhausted"
        else:
            self.termination_reason = "running"
        return True

    def _parse_action_key(self, key: Any) -> Tuple[int, str]:
        if isinstance(key, tuple) and len(key) == 2:
            return int(key[0]), str(key[1])
        return int(key), DEFAULT_MODEL_ID

    def start_rollout_action_atomic(self, action: Dict[Any, int]) -> bool:
        """
        Atomically launch an event-driven action.

        The action is sampled as a whole first. If the sampled total cost or
        sampled wall-clock extent is infeasible, no rollout from the action is
        launched. This preserves the MC/PDF "no clipping" semantics while still
        using the normal event queue after a feasible launch.
        """
        self.last_start_rollout_failure_reason = None
        if self.done():
            self.termination_reason = "completed"
            return True
        if self.deadline_reached():
            self.last_start_rollout_failure_reason = "deadline_exhausted"
            self.termination_reason = "deadline_exhausted"
            return False

        ready = set(self.ready_nodes())
        by_node: Dict[int, Tuple[str, int]] = {}
        for key, raw_count in action.items():
            nid, model_id = self._parse_action_key(key)
            if nid not in ready:
                continue
            count = int(raw_count)
            if count <= 0:
                continue
            if nid in by_node:
                self.last_start_rollout_failure_reason = "duplicate_node_model_action"
                self.termination_reason = "stalled"
                return False
            by_node[nid] = (model_id, count)

        if any(nid not in by_node for nid in ready):
            self.last_start_rollout_failure_reason = "zero_node_action"
            self.termination_reason = "stalled"
            return False

        if not by_node:
            self.last_start_rollout_failure_reason = "all_zero_action"
            self.termination_reason = "stalled"
            return False

        samples_by_node: Dict[int, Tuple[str, List[Tuple[float, bool, float]]]] = {}
        costs_by_node: Dict[int, float] = {}
        total_cost = 0.0
        max_latency = 0.0
        for nid, (model_id, count) in by_node.items():
            node_samples: List[Tuple[float, bool, float]] = []
            node_cost = 0.0
            for _ in range(count):
                sampled_latency, success, realized_cost = self._sample_rollout_realization(nid, model_id)
                node_samples.append((sampled_latency, success, realized_cost))
                node_cost += realized_cost
                max_latency = max(max_latency, sampled_latency)
            samples_by_node[nid] = (model_id, node_samples)
            costs_by_node[nid] = node_cost
            total_cost += node_cost

        if max_latency > self.remaining_deadline() + 1e-12:
            self.last_start_rollout_failure_reason = "deadline_exhausted"
            self.time = self.deadline
            self.termination_reason = "deadline_exhausted"
            return False

        if total_cost > self.remaining_budget + 1e-12:
            self.last_start_rollout_failure_reason = "budget_exhausted"
            self.termination_reason = "budget_exhausted"
            return False

        if self._node_budget_remaining is not None:
            for nid, node_cost in costs_by_node.items():
                if node_cost > self._node_budget_remaining.get(nid, 0.0) + 1e-12:
                    self.last_start_rollout_failure_reason = "budget_exhausted"
                    self.termination_reason = "budget_exhausted"
                    return False

        self.remaining_budget -= total_cost
        if self._node_budget_remaining is not None:
            for nid, node_cost in costs_by_node.items():
                self._node_budget_remaining[nid] -= node_cost

        for nid, (model_id, node_samples) in samples_by_node.items():
            st = self.runtime[nid]
            for sampled_latency, success, realized_cost in node_samples:
                st.cumulative_cost += realized_cost
                st.num_started += 1
                rid = self.rollout_counter
                self.rollout_counter += 1
                st.running_rollouts.add(rid)
                heapq.heappush(
                    self.event_queue,
                    RunningRollout(
                        finish_time=self.time + sampled_latency,
                        rollout_id=rid,
                        node_id=nid,
                        success=success,
                        cost=realized_cost,
                        model_id=model_id,
                    )
                )

        self.termination_reason = "running"
        return True

    def start_rollout(self, node_id: int, model_id: str = DEFAULT_MODEL_ID) -> bool:
        self.last_start_rollout_failure_reason = None
        model_id = str(model_id)
        st = self.runtime[node_id]
        if st.completed:
            self.last_start_rollout_failure_reason = "completed"
            return False
        if self.deadline_reached():
            self.last_start_rollout_failure_reason = "deadline_exhausted"
            return False

        sampled_latency, success, realized_compute_cost = self._sample_rollout_realization(node_id, model_id)

        if self.remaining_budget < realized_compute_cost:
            self.last_start_rollout_failure_reason = "budget_exhausted"
            return False
        if self._node_budget_remaining is not None:
            node_remaining = self._node_budget_remaining.get(node_id, 0.0)
            if node_remaining < realized_compute_cost:
                self.last_start_rollout_failure_reason = "budget_exhausted"
                return False

        self.remaining_budget -= realized_compute_cost
        if self._node_budget_remaining is not None:
            self._node_budget_remaining[node_id] -= realized_compute_cost
        st.cumulative_cost += realized_compute_cost
        st.num_started += 1

        rid = self.rollout_counter
        self.rollout_counter += 1
        st.running_rollouts.add(rid)

        finish_time = self.time + sampled_latency

        heapq.heappush(
            self.event_queue,
            RunningRollout(
                finish_time=finish_time,
                rollout_id=rid,
                node_id=node_id,
                success=success,
                cost=realized_compute_cost,
                model_id=model_id,
            )
        )
        return True

    def mark_node_complete(self, node_id: int):
        st = self.runtime[node_id]
        if st.completed:
            return
        st.completed = True
        self.completed_nodes.add(node_id)

        for child in st.children:
            child_st = self.runtime[child]
            if all(self.runtime[p].completed for p in child_st.parents):
                child_st.unlocked = True

    def advance_to_next_event(self) -> Optional[Tuple[int, bool]]:
        """
        Pop the earliest completion event and update node state.

        Returns:
            None              - if event queue is empty (no progress).
            (node_id, True)   - the node's current rollout batch has ended and
                                 the node is now done (at least one success) OR
                                 all rollouts in the batch failed.
            (node_id, False)  - a rollout completed but the node still has
                                 other rollouts in flight (no state‐change
                                 that warrants replanning).
        """
        if not self.event_queue:
            return None

        ev = heapq.heappop(self.event_queue)
        self.time = ev.finish_time

        st = self.runtime[ev.node_id]
        if ev.rollout_id in st.running_rollouts:
            st.running_rollouts.remove(ev.rollout_id)
        st.num_finished += 1

        if (not st.completed) and ev.success:
            st.num_success += 1

        if (not st.completed) and len(st.running_rollouts) == 0 and st.num_success > 0:
            self.mark_node_complete(ev.node_id)

        # Determine whether this event constitutes a "significant" state change:
        #   - node just completed at the end of its batch, or
        #   - node has no more running rollouts (all launched rollouts failed)
        significant = st.completed or len(st.running_rollouts) == 0
        return (ev.node_id, significant)

    def done(self) -> bool:
        return len(self.completed_nodes) == len(self.runtime)

    def completion_fraction(self) -> float:
        return len(self.completed_nodes) / len(self.runtime)

    def collect_result(self) -> SimulationResult:
        stats = {}
        for nid, st in self.runtime.items():
            stats[nid] = {
                "completed": float(st.completed),
                "num_started": float(st.num_started),
                "num_finished": float(st.num_finished),
                "num_success": float(st.num_success),
                "cost": float(st.cumulative_cost),
            }

        reason = "completed" if self.done() else self.termination_reason

        return SimulationResult(
            success=self.done(),
            termination_reason=reason,
            makespan=float(self.time),
            deadline=float(self.deadline),
            remaining_deadline=float(self.remaining_deadline()),
            total_cost=float(self.budget - self.remaining_budget),
            planner_calls=self.planner_calls,
            planner_wall_ms=float(self.planner_wall_ms),
            total_rollouts=self.rollout_counter,
            completion_fraction=float(self.completion_fraction()),
            node_stats=stats,
        )
