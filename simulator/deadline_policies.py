import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dag import DAG
from .deadline_surrogate import (
    compute_phi_and_weights,
    deadline_risk_score,
    deadline_success_lower_bound,
    expected_work_surrogate,
    min_feasible_r,
    node_risk_cost,
    solve_node_subproblem,
)
from .simulator import Simulator
from .LOADER_policies import Policy


class _DeadlinePlanner:
    def __init__(
        self,
        theta_multipliers: Sequence[float],
        max_iters: int = 25,
        dual_lr: float = 0.2,
        tol: float = 1e-4,
        theta_grid_size: Optional[int] = None,
        theta_grid_spacing: str = "log_upper",
        theta_upper_gap_ratio: float = 1e-6,
        budget_reservation_mode: str = "hard",
        budget_reservation_c: float = 2.0,
        budget_reservation_alpha: float = 0.0,
    ):
        self.theta_multipliers = tuple(float(x) for x in theta_multipliers)
        self.theta_grid_size = max(2, int(theta_grid_size or 10))
        self.theta_grid_spacing = self._normalize_theta_grid_spacing(theta_grid_spacing)
        self.theta_upper_gap_ratio = min(max(float(theta_upper_gap_ratio), 1e-12), 0.25)
        self.budget_reservation_mode = self._normalize_budget_reservation_mode(budget_reservation_mode)
        self.budget_reservation_c = max(0.0, float(budget_reservation_c))
        self.budget_reservation_alpha = min(max(float(budget_reservation_alpha), 0.0), 1.0)
        self.max_iters = int(max_iters)
        self.dual_lr = float(dual_lr)
        self.tol = float(tol)
        self.beta_floor = 1e-9
        self.beta_cap = 1e6
        self.last_solve_trace: Dict[str, Any] = {}

    @staticmethod
    def _normalize_theta_grid_spacing(value: str) -> str:
        text = str(value or "log_upper").strip().lower().replace("-", "_")
        aliases = {
            "log": "log_upper",
            "log_upper": "log_upper",
            "analytic_upper_log_gap": "log_upper",
            "linear": "linear",
            "lin": "linear",
        }
        if text not in aliases:
            raise ValueError(
                f"Unsupported theta_grid_spacing={value!r}. "
                "Expected one of: log_upper, linear."
            )
        return aliases[text]

    @staticmethod
    def _normalize_budget_reservation_mode(value: str) -> str:
        text = str(value or "hard").strip().lower().replace("-", "_")
        aliases = {
            "hard": "hard",
            "batch": "hard",
            "none": "hard",
            "0": "hard",
            "adaptive": "adaptive",
            "deadline_adaptive": "adaptive",
            "deadline_adaptive_budget": "adaptive",
            "fixed": "fixed_alpha",
            "fixed_alpha": "fixed_alpha",
            "constant": "fixed_alpha",
            "constant_alpha": "fixed_alpha",
            "continuation": "continuation",
            "expected_continuation": "continuation",
            "infinite": "continuation",
            "1": "continuation",
        }
        if text not in aliases:
            raise ValueError(
                f"Unsupported budget_reservation_mode={value!r}. "
                "Expected one of: hard, adaptive, fixed_alpha, continuation."
            )
        return aliases[text]

    def _downstream_time(self, sim: Simulator, remaining_dag: DAG) -> Dict[int, float]:
        downstream: Dict[int, float] = {}
        for nid in reversed(remaining_dag.topological_order()):
            children = remaining_dag.nodes[nid].children
            if not children:
                downstream[nid] = 0.0
                continue
            best = 0.0
            for child in children:
                child_tau = max(float(sim.runtime[child].template.lat_mean), 1e-9)
                best = max(best, child_tau + downstream.get(child, 0.0))
            downstream[nid] = best
        return downstream

    def _reservation_alpha(
        self,
        remaining_deadline: float,
        downstream_time: float,
        tau: float,
    ) -> float:
        mode = self.budget_reservation_mode
        if mode == "hard":
            return 0.0
        if mode == "fixed_alpha":
            return self.budget_reservation_alpha
        if mode == "continuation":
            return 1.0

        local_deadline = max(0.0, float(remaining_deadline) - float(downstream_time))
        attempt_windows = max(1, int(math.floor(local_deadline / max(float(tau), 1e-9))))
        return min(max(1.0 - self.budget_reservation_c / attempt_windows, 0.0), 1.0)

    def _theta_upper_bound(self, sim: Simulator, remaining_dag: DAG) -> float:
        """
        Analytic theta upper bound for the current remaining DAG state.

        This follows the analytic bound:
            theta < b / sum_v tau_v / -log(1-p_v)
        where b is the current remaining budget variable used by the planner.
        """
        budget = float(max(sim.remaining_budget, 0.0))
        if budget <= 0.0 or not remaining_dag.node_ids:
            return 0.0

        denom = 0.0
        for nid in remaining_dag.node_ids:
            tpl = sim.runtime[nid].template
            tau = max(float(tpl.lat_mean), 1e-12)
            p = min(max(float(tpl.success_prob), 1e-12), 1.0 - 1e-12)
            success_log = max(-math.log1p(-p), 1e-12)
            denom += tau / success_log

        if denom <= 0.0 or not math.isfinite(denom):
            return 0.0
        upper = budget / denom
        return float(upper) if math.isfinite(upper) and upper > 0.0 else 0.0

    def _theta_grid(self, sim: Simulator, remaining_dag: DAG) -> List[float]:
        upper = self._theta_upper_bound(sim, remaining_dag)
        if upper <= 0.0:
            return [1e-12]

        n_grid = self.theta_grid_size
        lower = max(1e-12, upper * 1e-6)
        upper_gap = max(1e-12, upper * self.theta_upper_gap_ratio)
        max_theta = max(lower, upper - upper_gap)
        if max_theta <= lower or n_grid <= 1:
            return [max_theta]

        if self.theta_grid_spacing == "linear":
            step = (max_theta - lower) / max(n_grid - 1, 1)
            grid = [lower + idx * step for idx in range(n_grid)]
            return sorted(set(round(theta, 15) for theta in grid))

        # Log-space the distance to the upper bound. This makes adjacent grid
        # points increasingly dense as theta approaches the analytic upper bound.
        gap_start = max(upper - lower, 1e-12)
        gap_end = max(upper - max_theta, 1e-12)
        log_start = math.log(gap_start)
        log_end = math.log(gap_end)
        grid = []
        for idx in range(n_grid):
            alpha = idx / max(n_grid - 1, 1)
            gap = math.exp(log_start + alpha * (log_end - log_start))
            theta = upper - gap
            if theta > 0.0 and theta < upper:
                grid.append(theta)

        return sorted(set(round(theta, 15) for theta in grid)) or [max_theta]

    def _node_params(
        self,
        sim: Simulator,
        remaining_dag: DAG,
        theta: float,
    ) -> Optional[Dict[int, Dict[str, float]]]:
        params: Dict[int, Dict[str, float]] = {}
        min_cost = 0.0
        downstream_time = self._downstream_time(sim, remaining_dag)
        remaining_deadline = sim.remaining_deadline()
        for nid in remaining_dag.node_ids:
            tpl = sim.runtime[nid].template
            tau = max(tpl.lat_mean, 1e-9)
            p = tpl.success_prob
            kappa = sim.node_work_cost(nid)
            r_min = min_feasible_r(theta, tau, p)
            reservation_alpha = self._reservation_alpha(
                remaining_deadline,
                downstream_time.get(nid, 0.0),
                tau,
            )
            params[nid] = {
                "tau": tau,
                "p": p,
                "kappa": kappa,
                "r_min": r_min,
                "downstream_time": downstream_time.get(nid, 0.0),
                "reservation_alpha": reservation_alpha,
            }
            min_cost += expected_work_surrogate(kappa, p, r_min, reservation_alpha)
        if min_cost > sim.remaining_budget + 1e-9:
            return None
        return params

    def _solve_for_theta(
        self,
        sim: Simulator,
        remaining_dag: DAG,
        theta: float,
        trace_level: str = "summary",
    ) -> Tuple[Optional[Tuple[Dict[int, float], Dict[int, float], float, float]], Dict[str, Any]]:
        theta_trace: Dict[str, Any] = {
            "theta": float(theta),
            "feasible": False,
            "selected": False,
        }
        node_params = self._node_params(sim, remaining_dag, theta)
        if node_params is None:
            theta_trace["reason"] = "min_cost_exceeds_budget"
            return None, theta_trace
        if not node_params:
            theta_trace.update({
                "feasible": True,
                "phi": 0.0,
                "best_phi": 0.0,
                "final_beta": 0.0,
                "n_iters": 0,
                "final_total_cost": 0.0,
                "plan_size": 0,
                "subproblem_bisect_iters_total": 0,
                "subproblem_bisect_iters_max": 0,
                "subproblem_bracket_iters_total": 0,
                "subproblem_bracket_iters_max": 0,
            })
            return ({}, {}, 0.0, theta), theta_trace

        plan = {nid: params["r_min"] for nid, params in node_params.items()}
        psi0 = {
            nid: node_risk_cost(theta, params["tau"], params["p"], plan[nid])
            for nid, params in node_params.items()
        }
        phi0, weights0 = compute_phi_and_weights(remaining_dag, psi0)
        beta = 0.0
        best_plan = dict(plan)
        best_weights = dict(weights0)
        best_phi = float(phi0)
        budget = max(sim.remaining_budget, 1e-9)
        prev_weights: Optional[Dict[int, float]] = None
        final_total_cost = sum(
            expected_work_surrogate(
                params["kappa"],
                params["p"],
                plan[nid],
                params.get("reservation_alpha", 0.0),
            )
            for nid, params in node_params.items()
        )
        n_iters = 0
        iter_traces: List[Dict[str, Any]] = []
        bisect_iters_total = 0
        bisect_iters_max = 0
        bracket_iters_total = 0
        bracket_iters_max = 0

        for iter_idx in range(self.max_iters):
            psi = {
                nid: node_risk_cost(theta, params["tau"], params["p"], plan[nid])
                for nid, params in node_params.items()
            }
            _phi_prev, weights = compute_phi_and_weights(remaining_dag, psi)

            new_plan: Dict[int, float] = {}
            total_cost = 0.0
            iter_bisect_total = 0
            iter_bisect_max = 0
            iter_bracket_total = 0
            iter_bracket_max = 0
            iter_subproblem_stats: Dict[str, Dict[str, int]] = {}
            for nid, params in node_params.items():
                subproblem_stats: Dict[str, int] = {}
                rv = solve_node_subproblem(
                    theta=theta,
                    tau_v=params["tau"],
                    p_v=params["p"],
                    kappa_v=params["kappa"],
                    weight_v=max(weights.get(nid, 0.0), 0.0),
                    beta=beta,
                    r_min=params["r_min"],
                    budget_hint=budget,
                    reservation_alpha=params.get("reservation_alpha", 0.0),
                    tol=self.tol,
                    stats=subproblem_stats,
                )
                new_plan[nid] = rv
                total_cost += expected_work_surrogate(
                    params["kappa"],
                    params["p"],
                    rv,
                    params.get("reservation_alpha", 0.0),
                )
                bisection_count = int(subproblem_stats.get("bisect_iters", 0))
                bracket_count = int(subproblem_stats.get("bracket_iters", 0))
                iter_bisect_total += bisection_count
                iter_bisect_max = max(iter_bisect_max, bisection_count)
                iter_bracket_total += bracket_count
                iter_bracket_max = max(iter_bracket_max, bracket_count)
                if trace_level == "iter":
                    iter_subproblem_stats[str(nid)] = {
                        "bisect_iters": bisection_count,
                        "bracket_iters": bracket_count,
                    }

            # English note,English note,English note
            violation = (total_cost - budget) / budget
            if violation > 0.0:
                if beta <= 0.0:
                    avg_weight = sum(max(weights.get(nid, 0.0), 0.0) for nid in node_params) / max(len(node_params), 1)
                    beta = max(self.beta_floor, avg_weight)
                beta = min(self.beta_cap, beta * math.exp(self.dual_lr * violation))
            elif beta > 0.0:
                beta = max(0.0, beta * math.exp(self.dual_lr * violation))
                if beta < self.beta_floor:
                    beta = 0.0

            psi_new = {
                nid: node_risk_cost(theta, params["tau"], params["p"], new_plan[nid])
                for nid, params in node_params.items()
            }
            phi_new, new_weights = compute_phi_and_weights(remaining_dag, psi_new)

            feasible_cost = sum(
                expected_work_surrogate(
                    params["kappa"],
                    params["p"],
                    new_plan[nid],
                    params.get("reservation_alpha", 0.0),
                )
                for nid, params in node_params.items()
            )
            final_total_cost = feasible_cost
            if feasible_cost <= budget * (1.0 + self.tol) and phi_new < best_phi:
                best_phi = phi_new
                best_plan = dict(new_plan)
                best_weights = dict(new_weights)

            plan_delta = max(abs(new_plan[nid] - plan[nid]) for nid in new_plan) if new_plan else 0.0
            weight_delta = 0.0
            if prev_weights is not None:
                weight_delta = max(
                    abs(new_weights.get(nid, 0.0) - prev_weights.get(nid, 0.0))
                    for nid in new_weights
                ) if new_weights else 0.0

            plan = new_plan
            prev_weights = dict(new_weights)
            n_iters = iter_idx + 1
            bisect_iters_total += iter_bisect_total
            bisect_iters_max = max(bisect_iters_max, iter_bisect_max)
            bracket_iters_total += iter_bracket_total
            bracket_iters_max = max(bracket_iters_max, iter_bracket_max)
            if trace_level == "iter":
                iter_traces.append({
                    "iter": iter_idx,
                    "beta": float(beta),
                    "total_cost": float(total_cost),
                    "feasible_cost": float(feasible_cost),
                    "violation": float(violation),
                    "phi": float(phi_new),
                    "best_phi": float(best_phi),
                    "plan_delta": float(plan_delta),
                    "weight_delta": float(weight_delta),
                    "subproblem_bisect_iters_total": int(iter_bisect_total),
                    "subproblem_bisect_iters_max": int(iter_bisect_max),
                    "subproblem_bracket_iters_total": int(iter_bracket_total),
                    "subproblem_bracket_iters_max": int(iter_bracket_max),
                    "subproblem_iters_by_node": iter_subproblem_stats,
                })
            if plan_delta <= self.tol and weight_delta <= self.tol:
                break

        if not math.isfinite(best_phi):
            theta_trace["reason"] = "nonfinite_phi"
            return None, theta_trace

        theta_trace.update({
            "feasible": True,
            "phi": float(best_phi),
            "best_phi": float(best_phi),
            "final_beta": float(beta),
            "n_iters": int(n_iters),
            "final_total_cost": float(final_total_cost),
            "budget_reservation_mode": self.budget_reservation_mode,
            "budget_reservation_c": float(self.budget_reservation_c),
            "budget_reservation_alpha": float(self.budget_reservation_alpha),
            "reservation_alpha_min": float(min(params["reservation_alpha"] for params in node_params.values())),
            "reservation_alpha_max": float(max(params["reservation_alpha"] for params in node_params.values())),
            "reservation_alpha_mean": float(
                sum(params["reservation_alpha"] for params in node_params.values()) / max(len(node_params), 1)
            ),
            "plan_size": int(len(best_plan)),
            "subproblem_bisect_iters_total": int(bisect_iters_total),
            "subproblem_bisect_iters_max": int(bisect_iters_max),
            "subproblem_bracket_iters_total": int(bracket_iters_total),
            "subproblem_bracket_iters_max": int(bracket_iters_max),
        })
        if trace_level in ("theta_full", "iter"):
            theta_trace["plan"] = {str(nid): float(value) for nid, value in best_plan.items()}
            theta_trace["weights"] = {str(nid): float(value) for nid, value in best_weights.items()}
        if trace_level == "iter":
            theta_trace["iters"] = iter_traces
        return (best_plan, best_weights, best_phi, theta), theta_trace

    def solve(
        self,
        sim: Simulator,
        remaining_dag: DAG,
        trace_level: str = "summary",
    ) -> Optional[Tuple[Dict[int, float], Dict[int, float], float, float]]:
        best = None
        best_score = float("inf")
        best_lb = None
        h = sim.remaining_deadline()
        theta_upper_bound = self._theta_upper_bound(sim, remaining_dag)
        theta_grid = self._theta_grid(sim, remaining_dag)
        theta_traces: List[Dict[str, Any]] = []
        n_feasible_theta = 0
        n_positive_lb = 0
        for theta_idx, theta in enumerate(theta_grid):
            solved, theta_trace = self._solve_for_theta(
                sim,
                remaining_dag,
                theta,
                trace_level=trace_level,
            )
            theta_trace["theta_idx"] = int(theta_idx)
            if solved is None:
                theta_traces.append(theta_trace)
                continue
            plan, weights, phi, theta_val = solved
            score = deadline_risk_score(phi, theta_val, h)
            lb = deadline_success_lower_bound(phi, theta_val, h)
            theta_trace["score"] = float(score)
            theta_trace["lb"] = float(lb)
            theta_traces.append(theta_trace)
            n_feasible_theta += 1
            if lb > 0.0:
                n_positive_lb += 1
            if score < best_score:
                best_score = score
                best_lb = lb
                best = solved

        selected_theta = None
        selected_phi = None
        selected_plan_size = 0
        if best is not None:
            plan, _weights, selected_phi, selected_theta = best
            selected_plan_size = len(plan)
            for theta_trace in theta_traces:
                if theta_trace.get("feasible") and abs(float(theta_trace["theta"]) - float(selected_theta)) <= 1e-15:
                    theta_trace["selected"] = True
                    break

        self.last_solve_trace = {
            "remaining_deadline": float(h),
            "theta_upper_bound": float(theta_upper_bound),
            "theta_grid_method": (
                "linear"
                if self.theta_grid_spacing == "linear"
                else "analytic_upper_log_gap"
            ),
            "theta_grid_spacing": self.theta_grid_spacing,
            "budget_reservation_mode": self.budget_reservation_mode,
            "budget_reservation_c": float(self.budget_reservation_c),
            "budget_reservation_alpha": float(self.budget_reservation_alpha),
            "theta_grid": [float(theta) for theta in theta_grid],
            "theta_evals": theta_traces,
            "n_theta_evals": int(len(theta_traces)),
            "n_feasible_theta_evals": int(n_feasible_theta),
            "n_positive_lb_theta_evals": int(n_positive_lb),
            "selected_theta": None if selected_theta is None else float(selected_theta),
            "selected_score": None if best is None else float(best_score),
            "selected_lb": None if best is None else float(best_lb),
            "selected_phi": None if selected_phi is None else float(selected_phi),
            "selected_plan_size": int(selected_plan_size),
        }
        return best

    def discrete_ready_targets(
        self,
        sim: Simulator,
        ready_nodes: List[int],
        continuous_plan: Dict[int, float],
        weights: Dict[int, float],
        theta: float,
        remaining_dag: Optional[DAG] = None,
    ) -> Dict[int, int]:
        if not ready_nodes:
            return {}

        current = {nid: sim.running_count(nid) for nid in ready_nodes}
        min_ready_cost = sum(
            max(0, 1 - current[nid]) * sim.node_work_cost(nid)
            for nid in ready_nodes
        )
        require_one_per_ready = min_ready_cost <= sim.remaining_budget + 1e-12

        def target_launch_cost(targets: Dict[int, int]) -> float:
            return sum(
                max(0, targets[nid] - current[nid]) * sim.node_work_cost(nid)
                for nid in ready_nodes
            )

        def enforce_ready_minimum(targets: Dict[int, int]) -> Tuple[Dict[int, int], float]:
            if not require_one_per_ready:
                cost = target_launch_cost(targets)
                return targets, max(0.0, sim.remaining_budget - cost)

            min_targets = {
                nid: max(current[nid], 1)
                for nid in ready_nodes
            }
            enforced = dict(targets)
            for nid, min_target in min_targets.items():
                enforced[nid] = max(enforced.get(nid, current[nid]), min_target)

            enforced_cost = target_launch_cost(enforced)
            if enforced_cost <= sim.remaining_budget + 1e-12:
                return enforced, max(0.0, sim.remaining_budget - enforced_cost)

            return min_targets, max(0.0, sim.remaining_budget - min_ready_cost)

        def greedy_targets() -> Tuple[Dict[int, int], float]:
            targets = dict(current)
            available = sim.remaining_budget
            if require_one_per_ready:
                # Theory-aligned execution constraint: if the remaining budget
                # can start every currently ready node once, do that before
                # spending extra launches on higher-risk nodes. DAG success
                # requires every node to eventually complete.
                for nid in ready_nodes:
                    extra_needed = max(0, 1 - targets[nid])
                    if extra_needed <= 0:
                        continue
                    kappa = sim.node_work_cost(nid)
                    targets[nid] += extra_needed
                    available -= extra_needed * kappa

            desired_upper = {}
            for nid in ready_nodes:
                desired = max(0.0, continuous_plan.get(nid, 0.0))
                min_target = max(current[nid], 1 if require_one_per_ready else current[nid])
                floor_target = max(min_target, int(math.floor(desired)))
                desired_upper[nid] = max(floor_target, int(math.ceil(desired)), current[nid] + 1)
                extra_needed = max(0, floor_target - current[nid])
                if targets[nid] >= floor_target:
                    continue
                extra_needed = max(0, floor_target - targets[nid])
                kappa = sim.node_work_cost(nid)
                launchable = min(extra_needed, int((available + 1e-12) // kappa)) if kappa > 0 else extra_needed
                targets[nid] += launchable
                available -= launchable * kappa

            while True:
                best_nid = None
                best_gain = -1.0
                for nid in ready_nodes:
                    if targets[nid] >= desired_upper[nid]:
                        continue
                    kappa = sim.node_work_cost(nid)
                    if available + 1e-12 < kappa:
                        continue
                    tpl = sim.runtime[nid].template
                    cur_r = float(max(targets[nid], 0.25))
                    next_r = cur_r + 1.0
                    psi_cur = node_risk_cost(theta, tpl.lat_mean, tpl.success_prob, cur_r)
                    psi_next = node_risk_cost(theta, tpl.lat_mean, tpl.success_prob, next_r)
                    gain = max(0.0, weights.get(nid, 1e-9) * (psi_cur - psi_next)) / max(kappa, 1e-9)
                    if gain > best_gain:
                        best_gain = gain
                        best_nid = nid
                if best_nid is None:
                    break
                available -= sim.node_work_cost(best_nid)
                targets[best_nid] += 1

            return targets, available

        def exact_floor_ceil_targets() -> Tuple[Dict[int, int], float]:
            if remaining_dag is None:
                return greedy_targets()

            choices: List[Tuple[int, List[int]]] = []
            n_combinations = 1
            for nid in ready_nodes:
                desired = max(0.0, continuous_plan.get(nid, 0.0))
                min_target = max(current[nid], 1 if require_one_per_ready else current[nid])
                floor_target = max(min_target, int(math.floor(desired)))
                ceil_target = max(floor_target, int(math.ceil(desired)))
                node_choices = sorted({min_target, floor_target, ceil_target})
                choices.append((nid, node_choices))
                n_combinations *= len(node_choices)

            # Exact floor/ceil enumeration is the theory-aligned path. Fall
            # back to the previous marginal rounding when the ready frontier is
            # too wide for exhaustive online enumeration.
            if n_combinations > 4096:
                return greedy_targets()

            def candidate_phi(candidate_targets: Dict[int, int]) -> float:
                psi = {}
                for nid in remaining_dag.node_ids:
                    tpl = sim.runtime[nid].template
                    r_val = (
                        float(candidate_targets[nid])
                        if nid in candidate_targets
                        else float(continuous_plan.get(nid, 1e-9))
                    )
                    psi[nid] = node_risk_cost(theta, tpl.lat_mean, tpl.success_prob, max(r_val, 1e-9))
                phi, _weights = compute_phi_and_weights(remaining_dag, psi)
                return phi

            best_targets = dict(current)
            best_cost = 0.0
            best_phi = float("inf")

            def visit(idx: int, candidate: Dict[int, int], cost: float) -> None:
                nonlocal best_targets, best_cost, best_phi
                if cost > sim.remaining_budget + 1e-12:
                    return
                if idx == len(choices):
                    phi = candidate_phi(candidate)
                    if not math.isfinite(phi):
                        return
                    if phi < best_phi - 1e-12 or (
                        abs(phi - best_phi) <= 1e-12 and cost < best_cost
                    ):
                        best_phi = phi
                        best_cost = cost
                        best_targets = dict(candidate)
                    return

                nid, node_choices = choices[idx]
                kappa = sim.node_work_cost(nid)
                for value in node_choices:
                    candidate[nid] = value
                    visit(idx + 1, candidate, cost + max(0, value - current[nid]) * kappa)
                candidate.pop(nid, None)

            visit(0, {}, 0.0)
            return best_targets, max(0.0, sim.remaining_budget - best_cost)

        targets, available = exact_floor_ceil_targets()
        targets, available = enforce_ready_minimum(targets)

        # Safety fallback for discretization/numerical edge cases:
        # if the optimizer produced no new launch while the simulator's own
        # budget proxy says at least one ready rollout is affordable, dispatch
        # the cheapest affordable ready node instead of stalling with work left.
        if all(targets[nid] <= current[nid] for nid in ready_nodes) and sim.can_launch_any_rollout():
            affordable = [
                nid for nid in ready_nodes
                if available + 1e-12 >= sim.node_work_cost(nid)
            ]
            if affordable:
                best_nid = min(affordable, key=lambda nid: sim.node_work_cost(nid))
                targets[best_nid] = current[best_nid] + 1

        return targets


class DeadlineStaticPolicy(Policy):
    def __init__(
        self,
        dag: DAG,
        budget: float,
        deadline: float,
        theta_multipliers: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        theta_grid_size: Optional[int] = None,
        theta_grid_spacing: str = "log_upper",
        max_iters: int = 25,
        budget_reservation_mode: str = "hard",
        budget_reservation_c: float = 2.0,
        budget_reservation_alpha: float = 0.0,
    ):
        self.dag = dag
        self.budget = float(budget)
        self.deadline = float(deadline)
        self._planner = _DeadlinePlanner(
            theta_multipliers=theta_multipliers,
            theta_grid_size=theta_grid_size,
            theta_grid_spacing=theta_grid_spacing,
            max_iters=max_iters,
            budget_reservation_mode=budget_reservation_mode,
            budget_reservation_c=budget_reservation_c,
            budget_reservation_alpha=budget_reservation_alpha,
        )
        self._desired: Dict[int, int] = {nid: 1 for nid in dag.node_ids}
        self._weights: Dict[int, float] = {nid: 1.0 for nid in dag.node_ids}
        self._theta: float = 1.0
        self._initialized = False

    def _initialize(self, sim: Simulator) -> None:
        if self._initialized:
            return
        solved = self._planner.solve(sim, self.dag)
        if solved is not None:
            plan, weights, _phi, theta = solved
            self._weights = dict(weights)
            self._theta = theta
            self._desired = {
                nid: max(1, int(round(plan.get(nid, 1.0))))
                for nid in self.dag.node_ids
            }
        self._initialized = True

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        self._initialize(sim)
        ready = sim.ready_nodes()
        if not ready:
            return {}
        return {
            nid: max(sim.running_count(nid), self._desired.get(nid, 1))
            for nid in ready
        }


class DeadlineRecedingPolicy(Policy):
    def __init__(
        self,
        full_dag: DAG,
        theta_multipliers: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
        theta_grid_size: Optional[int] = None,
        theta_grid_spacing: str = "log_upper",
        max_iters: int = 25,
        dual_lr: float = 0.2,
        budget_reservation_mode: str = "hard",
        budget_reservation_c: float = 2.0,
        budget_reservation_alpha: float = 0.0,
    ):
        self.full_dag = full_dag
        self._planner = _DeadlinePlanner(
            theta_multipliers=theta_multipliers,
            theta_grid_size=theta_grid_size,
            theta_grid_spacing=theta_grid_spacing,
            max_iters=max_iters,
            dual_lr=dual_lr,
            budget_reservation_mode=budget_reservation_mode,
            budget_reservation_c=budget_reservation_c,
            budget_reservation_alpha=budget_reservation_alpha,
        )
        self._cached_completed: Optional[frozenset] = None
        self._cached_remaining_dag: Optional[DAG] = None
        self.trace_id: Optional[str] = None
        self.trace_enabled = False
        self.trace_level = "summary"
        self.trace_context: Dict[str, Any] = {}
        self.trace_events: List[Dict[str, Any]] = []
        self.planner_call_count = 0
        self.planner_theta_eval_count = 0
        self.last_best_theta: Optional[float] = None
        self.last_best_score: Optional[float] = None
        self.last_best_lb: Optional[float] = None
        self.last_best_phi: Optional[float] = None
        self.last_plan_size: Optional[int] = None

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
        self.planner_theta_eval_count = 0
        self.last_best_theta = None
        self.last_best_score = None
        self.last_best_lb = None
        self.last_best_phi = None
        self.last_plan_size = None

    def deadline_trace_summary(self) -> Dict[str, Any]:
        return {
            "deadline_trace_id": self.trace_id,
            "deadline_planner_n_calls": int(self.planner_call_count),
            "deadline_planner_n_theta_evals": int(self.planner_theta_eval_count),
            "deadline_planner_last_best_theta": self.last_best_theta,
            "deadline_planner_last_best_score": self.last_best_score,
            "deadline_planner_last_best_lb": self.last_best_lb,
            "deadline_planner_last_best_phi": self.last_best_phi,
            "deadline_planner_last_plan_size": self.last_plan_size,
            "deadline_budget_reservation_mode": self._planner.budget_reservation_mode,
            "deadline_budget_reservation_c": self._planner.budget_reservation_c,
            "deadline_budget_reservation_alpha": self._planner.budget_reservation_alpha,
        }

    def pop_deadline_trace_events(self) -> List[Dict[str, Any]]:
        events = self.trace_events
        self.trace_events = []
        return events

    def _remaining_dag(self, sim: Simulator) -> DAG:
        current_completed = frozenset(sim.completed_nodes)
        if current_completed != self._cached_completed:
            self._cached_completed = current_completed
            self._cached_remaining_dag = self.full_dag.induced_remaining_subgraph(sim.completed_nodes)
        return self._cached_remaining_dag

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready = sim.ready_nodes()
        if not ready:
            return {}
        if sim.remaining_deadline() <= 0.0:
            return {}

        remaining_dag = self._remaining_dag(sim)
        call_idx = self.planner_call_count
        self.planner_call_count += 1
        solved = self._planner.solve(
            sim,
            remaining_dag,
            trace_level=self.trace_level if self.trace_enabled else "summary",
        )
        solve_trace = dict(self._planner.last_solve_trace or {})
        theta_evals = list(solve_trace.get("theta_evals", []))
        self.planner_theta_eval_count += len(theta_evals)
        if solve_trace.get("selected_theta") is not None:
            self.last_best_theta = solve_trace.get("selected_theta")
            self.last_best_score = solve_trace.get("selected_score")
            self.last_best_lb = solve_trace.get("selected_lb")
            self.last_best_phi = solve_trace.get("selected_phi")
            self.last_plan_size = solve_trace.get("selected_plan_size")

        if self.trace_enabled:
            self.trace_events.append({
                **self.trace_context,
                "trace_id": self.trace_id,
                "planner_call_idx": int(call_idx),
                "sim_time": float(sim.time),
                "remaining_deadline": float(sim.remaining_deadline()),
                "remaining_budget": float(sim.remaining_budget),
                "completed_nodes": int(len(sim.completed_nodes)),
                "remaining_nodes": int(len(remaining_dag.node_ids)),
                "ready_nodes": [int(nid) for nid in ready],
                **solve_trace,
            })
        if solved is None:
            targets = {nid: sim.running_count(nid) for nid in ready}
            available = sim.remaining_budget
            for nid in ready:
                if targets[nid] > 0:
                    continue
                kappa = sim.node_work_cost(nid)
                if available + 1e-12 < kappa:
                    continue
                targets[nid] = 1
                available -= kappa
            return targets

        plan, weights, _phi, theta = solved
        return self._planner.discrete_ready_targets(
            sim,
            ready,
            plan,
            weights,
            theta,
            remaining_dag=remaining_dag,
        )
