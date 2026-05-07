import math
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

try:
    from scipy.special import lambertw
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from .dag import DAG
from . import Simulator


# -----------------------------
# shared helpers
# -----------------------------

def topo_order(dag: DAG) -> List[int]:
    return dag.topological_order()


def normalize_simplex(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 1e-12)
    s = x.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return x / s


def longest_path_dist_from_nodes(dag: DAG, node_cost: Dict[int, float]) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    order = topo_order(dag)[::-1]
    dist: Dict[int, float] = {}
    nxt: Dict[int, Optional[int]] = {}
    for u in order:
        if len(dag.nodes[u].children) == 0:
            dist[u] = node_cost[u]
            nxt[u] = None
        else:
            best_child = max(dag.nodes[u].children, key=lambda v: dist[v])
            dist[u] = node_cost[u] + dist[best_child]
            nxt[u] = best_child
    return dist, nxt


def dag_longest_path_value(dag: DAG, node_cost: Dict[int, float]) -> float:
    """
    English note node_cost English note DAG English note surrogate makespan proxy:
    English note source-to-sink English note node_cost.
    """
    if len(dag.nodes) == 0:
        return 0.0
    dist, _ = longest_path_dist_from_nodes(dag, node_cost)
    roots = dag.roots()
    if not roots:
        return 0.0
    return max(dist[u] for u in roots)


def recover_one_critical_path(dag: DAG, node_cost: Dict[int, float]) -> List[int]:
    if len(dag.nodes) == 0:
        return []
    dist, nxt = longest_path_dist_from_nodes(dag, node_cost)
    roots = dag.roots()
    if not roots:
        return []
    start = max(roots, key=lambda u: dist[u])
    path = [start]
    cur = start
    while nxt[cur] is not None:
        cur = nxt[cur]
        path.append(cur)
    return path


def recover_top_k_paths(dag: DAG, node_cost: Dict[int, float], k: int = 4) -> List[List[int]]:
    """
    PDF English note/MWU,English note;
    English note active-path extraction,English note.
    """
    if len(dag.nodes) == 0:
        return []
    tmp_cost = dict(node_cost)
    paths: List[List[int]] = []
    for _ in range(k):
        path = recover_one_critical_path(dag, tmp_cost)
        if not path or path in paths:
            break
        paths.append(path)
        for u in path:
            tmp_cost[u] *= 0.5
    return paths

def enumerate_all_source_sink_paths(
    dag: DAG,
    max_paths: Optional[int] = None,
) -> List[List[int]]:
    """
    English note remaining DAG English note source-to-sink English note.

    English note:
    - English note"English note"English note;
    - English note layered / fork-join DAG English note;
    - English note max_paths English note,English note,English note.
    """
    if len(dag.nodes) == 0:
        return []

    roots = dag.roots()
    sinks = set(dag.sinks())
    if not roots or not sinks:
        return []

    paths: List[List[int]] = []

    def dfs(u: int, cur: List[int]) -> None:
        if max_paths is not None and len(paths) >= max_paths:
            raise RuntimeError(
                f"enumerate_all_source_sink_paths exceeded max_paths={max_paths}. "
                f"Remaining DAG is too large for full-path formulation."
            )

        cur.append(u)
        if u in sinks:
            paths.append(list(cur))
        else:
            for v in dag.nodes[u].children:
                dfs(v, cur)
        cur.pop()

    for r in roots:
        dfs(r, [])

    return paths


def dedup_paths(paths: List[List[int]]) -> List[List[int]]:
    seen: Set[Tuple[int, ...]] = set()
    out: List[List[int]] = []
    for p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def extend_simplex_with_new_coordinate(lambdas: np.ndarray, eps_new: float = 0.05) -> np.ndarray:
    """
    English note active set English note,English note lambda English note.
    """
    eps_new = min(max(float(eps_new), 1e-6), 0.25)

    if lambdas.size == 0:
        return np.array([1.0], dtype=float)

    old = np.asarray(lambdas, dtype=float)
    old = normalize_simplex(old)
    old = old * (1.0 - eps_new)
    new = np.concatenate([old, np.array([eps_new], dtype=float)])
    return normalize_simplex(new)


def tuple_path(path: List[int]) -> Tuple[int, ...]:
    return tuple(path)


def path_node_costs_under_plan(
    sim: Simulator,
    remaining_nodes: List[int],
    plan: Dict[int, int],
) -> Dict[int, float]:
    """
    English note plan,English note surrogate English note
    d_tilde_v(r_v) English note node_cost.
    """
    out: Dict[int, float] = {}
    for nid in remaining_nodes:
        st = sim.runtime[nid]
        delta = sim.node_delta_to_round_end(nid)
        r_cur = float(sim.running_count(nid))
        r_v = float(plan.get(nid, max(1, int(r_cur))))
        out[nid] = expected_latency_surrogate(
            delta=delta,
            t_v=st.template.lat_mean,
            p_v=st.template.success_prob,
            r_cur=r_cur,
            r_v=r_v,
        )
    return out


def longest_path_not_in_set(
    dag: DAG,
    node_cost: Dict[int, float],
    excluded: Set[Tuple[int, ...]],
) -> List[int]:
    """
    English note node_cost English note,English note excluded English note source-to-sink English note.
    English note column generation English note"English note"English note.

    English note:English note DAG English note DP(O(V+E))English note,
    English note excluded English note,English note.
    """
    # English note O(V+E) DP English note
    best_path = recover_one_critical_path(dag, node_cost)
    if not best_path:
        return []

    if tuple_path(best_path) not in excluded:
        return best_path

    # English note active set English note,English note
    all_paths = enumerate_all_source_sink_paths(dag)
    if not all_paths:
        return []

    best_path = []
    best_val = -1.0

    for p in all_paths:
        key = tuple_path(p)
        if key in excluded:
            continue
        val = sum(node_cost[u] for u in p)
        if val > best_val:
            best_val = val
            best_path = p

    return best_path
# -----------------------------
# exact formulas from PDF
# -----------------------------

def expected_latency_surrogate(
    delta: float,
    t_v: float,
    p_v: float,
    r_cur: float,
    r_v: float,
) -> float:
    """
    PDF formula:
        d_tilde_v(r_v) = delta_v + (1 - p_v)^(r_v^cur) * t_v / (1 - (1 - p_v)^r_v)

    English note:
    - r_cur English note r_v^cur
    - r_v   English note
    """
    a_v = 1.0 - p_v
    y_cur = a_v ** max(r_cur, 1e-9)
    y_fut = a_v ** max(r_v, 1e-9)
    y_fut = min(max(y_fut, 1e-12), 1.0 - 1e-12)
    return delta + y_cur * t_v / (1.0 - y_fut)


def expected_compute_cost_surrogate(
    delta: float,
    t_v: float,
    p_v: float,
    r_cur: float,
    r_v: float,
) -> float:
    """
    PDF formula:
        c_v(r_v) = r_v^cur * delta_v + r_v * d_tilde_v(r_v)
    """
    d_tilde = expected_latency_surrogate(delta, t_v, p_v, r_cur, r_v)
    return r_cur * delta + r_v * d_tilde


def node_objective(
    delta: float,
    t_v: float,
    p_v: float,
    r_cur: float,
    r_v: float,
    w_v: float,
    beta: float,
) -> float:
    d_tilde = expected_latency_surrogate(delta, t_v, p_v, r_cur, r_v)
    c_v = expected_compute_cost_surrogate(delta, t_v, p_v, r_cur, r_v)
    return w_v * d_tilde + beta * c_v


def lambert_closed_form_r(
    p_v: float,
    w_v: float,
    beta: float,
) -> float:
    """
    Lambert W English note r_star.

    r_star English note (p_v, w_v, beta) English note:
      - beta English note(English note)→ r_star English note
      - beta English note(English note)→ r_star English note

    English note c_v > 500(exp(-c_v) English note)English note Lambert W₋₁ English note:
      W_{-1}(-e^{-c}) ≈ -(c + ln c)
    English note r_star,English note.
    """
    _FALLBACK_R = 64.0  # English note

    # probability guards
    a_v = 1.0 - p_v
    a_v = min(max(a_v, 1e-12), 1.0 - 1e-12)
    b_v = -math.log(a_v)

    # β English note 1e-30,English note r_star English note
    # (English note 1e-8 English note r_star ≈ 57,English note)
    beta = max(beta, 1e-30)

    # if node weight is essentially zero, optimal action is minimum
    w_v = max(w_v, 0.0)
    alpha_v = b_v * w_v / beta

    # Near alpha = 0, the solution approaches r = 1
    if alpha_v <= 1e-10:
        return 1.0

    c_v = 1.0 + alpha_v

    # c_v English note exp(-c_v) English note,English note scipy
    if c_v > 500.0:
        # W_{-1}(-e^{-c}) ≈ -(c + ln c)  English note c → ∞
        # → y_star = 1/(c + ln c)
        # → r_star = (1/b_v) * ln(c + ln c)
        try:
            ln_c = math.log(c_v)
            w_approx = -(c_v + ln_c)
            y_star = -1.0 / w_approx
            y_star = min(max(y_star, 1e-15), 1.0 - 1e-12)
            r_star = -(1.0 / b_v) * math.log(y_star)
            if np.isfinite(r_star) and r_star >= 1.0:
                return float(r_star)
        except Exception:
            pass
        return _FALLBACK_R

    z = -math.exp(-c_v)

    # Clamp away from the exact branch point -1/e to avoid nan from scipy.
    branch_point = -1.0 / math.e
    if z <= branch_point:
        z = branch_point + 1e-12

    try:
        w = lambertw(z, k=-1)
        w_real = float(w.real)

        if not np.isfinite(w_real):
            return _FALLBACK_R

        y_star = -1.0 / w_real

        if not np.isfinite(y_star) or y_star <= 0.0:
            return _FALLBACK_R

        # y must lie in (0,1)
        y_star = min(max(y_star, 1e-12), 1.0 - 1e-12)

        r_star = -(1.0 / b_v) * math.log(y_star)

        if not np.isfinite(r_star):
            return _FALLBACK_R

        return float(max(r_star, 1.0))

    except Exception:
        return _FALLBACK_R


def discretize_neighbor_pair(r_cont: float, min_r: int = 1, max_r: int | None = None):
    """English note r English note (lo, hi),English note [min_r, max_r] English note."""
    if not np.isfinite(r_cont):
        return min_r, min_r

    lo = max(min_r, int(math.floor(r_cont)))
    hi = max(min_r, int(math.ceil(r_cont)))

    if max_r is not None:
        lo = min(lo, max_r)
        hi = min(hi, max_r)

    return lo, hi


# -----------------------------
# base policy interface
# -----------------------------

class Policy:
    def allocate(self, sim: Simulator) -> Dict[int, int]:
        raise NotImplementedError


class SequentialPolicy(Policy):
    """
    English note Baseline:
    Node-level Parallel (English note)
    Subtask-level Sequential (English note 1 English note,English note)
    """
    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready = sim.ready_nodes()
        if not ready:
            return {}
        
        # English note,English note 1 English note rollout
        return {nid: 1 for nid in ready}


class UniformPolicy(Policy):
    """
    English note(English note).

    English note:
      1. English note:budget_per_node = budget / n_nodes
      2. English note lat_mean English note rollout English note:
         r_v = floor(budget_per_node / lat_mean_v)
      3. English note

    English note:English note,English note rollout,
    English note rollout.
    """

    def __init__(self, dag: DAG, budget: float):
        n_nodes = len(dag.node_ids)
        budget_per_node = budget / max(n_nodes, 1)

        self.desired: Dict[int, int] = {}
        for nid in dag.node_ids:
            tpl = dag.nodes[nid].assigned_template
            t_v = max(tpl.lat_mean, 1e-9)
            self.desired[nid] = max(1, int(budget_per_node / t_v))

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready = sim.ready_nodes()
        if not ready:
            return {}
        return {nid: self.desired[nid] for nid in ready if nid in self.desired}


class RandomPolicy(Policy):
    """
    English note(English note).

    English note:
      1. English note(Dirichlet English note)
      2. English note lat_mean English note rollout English note:
         r_v = floor(budget_v / lat_mean_v)
      3. English note

    English note:English note,English note.
    """

    def __init__(self, dag: DAG, budget: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_nodes = len(dag.node_ids)

        # Dirichlet(1,...,1) English note
        #English note
        fractions = rng.dirichlet(np.ones(n_nodes))

        self.desired: Dict[int, int] = {}
        for i, nid in enumerate(dag.node_ids):
            tpl = dag.nodes[nid].assigned_template
            t_v = max(tpl.lat_mean, 1e-9)
            budget_v = budget * fractions[i]
            self.desired[nid] = max(1, int(budget_v / t_v))

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready = sim.ready_nodes()
        if not ready:
            return {}
        return {nid: self.desired[nid] for nid in ready if nid in self.desired}


class LatencyWeightedPolicy(Policy):
    """
    English note.

    English note:
      1. English note = lat_mean_v(English note,English note)
      2. English note:
         budget_v = budget · lat_mean_v / Σ(lat_mean)
      3. English note lat_mean English note rollout English note:
         r_v = floor(budget_v / lat_mean_v)
            = floor(budget / Σ(lat_mean))
      4. English note

    English note:English note,English note rollout English note
    English note,English note rollout English note.
    English note:English note budget_v / lat_mean_v = budget / Σ(lat_mean) English note
    English note,English note rollout.
    """

    def __init__(self, dag: DAG, budget: float):
        total_lat = sum(
            max(dag.nodes[nid].assigned_template.lat_mean, 1e-9)
            for nid in dag.node_ids
        )
        total_lat = max(total_lat, 1e-12)

        self.desired: Dict[int, int] = {}
        for nid in dag.node_ids:
            tpl = dag.nodes[nid].assigned_template
            t_v = max(tpl.lat_mean, 1e-9)
            budget_v = budget * t_v / total_lat
            self.desired[nid] = max(1, int(budget_v / t_v))

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready = sim.ready_nodes()
        if not ready:
            return {}
        return {nid: self.desired[nid] for nid in ready if nid in self.desired}


class SuccessRateWeightedPolicy(Policy):
    """
    English note.

    English note:
      1. English note = (1 - p_v),English note(English note,English note)
      2. English note:
         budget_v = budget · (1 - p_v) / Σ(1 - p_i)
      3. English note lat_mean English note rollout English note:
         r_v = floor(budget_v / lat_mean_v)
      4. English note

    English note:English note,English note rollout
    English note.
    """

    def __init__(self, dag: DAG, budget: float):
        # English note (1 - p_v)
        fail_weights: Dict[int, float] = {}
        for nid in dag.node_ids:
            tpl = dag.nodes[nid].assigned_template
            p_v = tpl.success_prob
            fail_weights[nid] = max(1.0 - p_v, 1e-9)  # English note 0

        total_fail = sum(fail_weights.values())
        total_fail = max(total_fail, 1e-12)

        self.desired: Dict[int, int] = {}
        for nid in dag.node_ids:
            tpl = dag.nodes[nid].assigned_template
            t_v = max(tpl.lat_mean, 1e-9)
            budget_v = budget * fail_weights[nid] / total_fail
            self.desired[nid] = max(1, int(budget_v / t_v))

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready = sim.ready_nodes()
        if not ready:
            return {}
        return {nid: self.desired[nid] for nid in ready if nid in self.desired}


class CriticalPathStaticPolicy(Policy):
    """
    English note.

    English note,English note--English note baseline.

    English note:
      1. English note lat_mean English note dist[v]
         dist[v] English note → English note(English note)
      2. English note dist[v] / Σ(dist) English note
      3. English note lat_mean English note rollout English note:
         budget_v = budget · dist[v] / Σ(dist)
         r_v = floor(budget_v / lat_mean_v)
      4. English note
    """

    def __init__(self, dag: DAG, budget: float):
        # English note
        node_cost = {nid: dag.nodes[nid].assigned_template.lat_mean for nid in dag.node_ids}
        dist, _ = longest_path_dist_from_nodes(dag, node_cost)
        total_dist = sum(dist[nid] for nid in dag.node_ids)
        total_dist = max(total_dist, 1e-12)

        self.desired: Dict[int, int] = {}
        for nid in dag.node_ids:
            tpl = dag.nodes[nid].assigned_template
            t_v = max(tpl.lat_mean, 1e-9)
            budget_v = budget * dist[nid] / total_dist
            self.desired[nid] = max(1, int(budget_v / t_v))

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        return {nid: self.desired[nid] for nid in sim.ready_nodes() if nid in self.desired}


# -----------------------------
# LOADER Static (one-shot): English note
# -----------------------------

class LoaderStaticPolicy(Policy):
    """
    LOADER English note(English note baseline).

    English note LoaderDualPolicy English note(Lambert W English note +
    MWU English note + English note),English note DAG English note,English note
    English note r_v English note.

    English note EM English note(multi-start):
    - English note 0 English note:English note r=1 English note(English note baseline)
    - English note 1..K English note:English note r_v(English note),β,λ(Dirichlet English note)
    - English note
    - English note makespan English note

    English note,English note baseline:
    - English note CriticalPathStaticPolicy:English note,English note Lambert W English note
    - English note LoaderDualPolicy:English note,English note
    """

    def __init__(
        self,
        dag: DAG,
        budget: float,
        max_dual_iters: int = 30,
        stable_rounds: int = 3,
        eta_beta: float = 0.05,
        eta_lambda: float = 0.5,
        n_active_paths: int = 4,
        n_restarts: int = 3,
        seed: Optional[int] = None,
    ):
        all_nodes = dag.node_ids
        rng = np.random.RandomState(seed)

        # English note:English note restart English note
        global_best_plan: Optional[Dict[int, int]] = None
        global_best_obj = float('inf')

        for restart_idx in range(n_restarts):
            plan, obj = self._run_one_start(
                dag, all_nodes, budget, max_dual_iters, stable_rounds,
                eta_beta, eta_lambda, n_active_paths, rng, restart_idx,
            )
            if plan is not None and obj < global_best_obj:
                global_best_obj = obj
                global_best_plan = plan

        if global_best_plan is not None:
            self.desired: Dict[int, int] = global_best_plan
        else:
            self.desired = {nid: 1 for nid in all_nodes}

    @staticmethod
    def _run_one_start(
        dag: DAG,
        all_nodes: List[int],
        budget: float,
        max_dual_iters: int,
        stable_rounds: int,
        eta_beta: float,
        eta_lambda: float,
        n_active_paths: int,
        rng: np.random.RandomState,
        restart_idx: int,
    ) -> Tuple[Optional[Dict[int, int]], float]:
        """
        English note,English note,English note (English note, makespan English note).

        restart_idx == 0 English note r=1 English note(English note);
        restart_idx >= 1 English note r_v,β,λ.
        """
        # --- English note:English note proxy cost English note ---
        init_proxy_cost: Dict[int, float] = {}
        for nid in all_nodes:
            tpl = dag.nodes[nid].assigned_template
            if restart_idx == 0:
                # English note:English note r=1 English note
                r_init = 1.0
            else:
                # English note:English note r_v ∈ [1, r_max]
                t_v = max(tpl.lat_mean, 1e-9)
                r_max = max(2.0, budget / max(len(all_nodes), 1) / t_v)
                r_init = float(rng.uniform(1.0, r_max))
            init_proxy_cost[nid] = expected_latency_surrogate(
                delta=0.0, t_v=tpl.lat_mean, p_v=tpl.success_prob,
                r_cur=r_init, r_v=r_init,
            )

        active_paths = recover_top_k_paths(dag, init_proxy_cost, k=n_active_paths)
        if len(active_paths) == 0:
            return {nid: 1 for nid in all_nodes}, float('inf')

        # --- English note ---
        if restart_idx == 0:
            lambdas = np.ones(len(active_paths), dtype=float) / len(active_paths)
            beta = 1.0
        else:
            # Dirichlet English note → English note
            lambdas = rng.dirichlet(np.ones(len(active_paths)))
            # β English note,English note [0.01, 10]
            beta = float(np.exp(rng.uniform(np.log(0.01), np.log(10.0))))

        # --- English note ---
        prev_plan: Optional[Dict[int, int]] = None
        stable_count = 0
        best_feasible_plan: Optional[Dict[int, int]] = None
        best_feasible_score = float("inf")

        for step in range(max_dual_iters):
            # Step 1: English note,English note Lambert W English note r_v
            w_v: Dict[int, float] = {nid: 0.0 for nid in all_nodes}
            for lam, path in zip(lambdas, active_paths):
                for nid in path:
                    if nid in w_v:
                        w_v[nid] += float(lam)
            for nid in all_nodes:
                w_v[nid] = max(w_v[nid], 1e-3)

            plan: Dict[int, int] = {}
            for nid in all_nodes:
                tpl = dag.nodes[nid].assigned_template
                r_cont = lambert_closed_form_r(
                    p_v=tpl.success_prob,
                    w_v=w_v[nid],
                    beta=beta,
                )
                lo, hi = discretize_neighbor_pair(r_cont)
                obj_lo = node_objective(
                    delta=0.0, t_v=tpl.lat_mean, p_v=tpl.success_prob,
                    r_cur=1.0, r_v=float(lo), w_v=w_v[nid], beta=beta,
                )
                obj_hi = node_objective(
                    delta=0.0, t_v=tpl.lat_mean, p_v=tpl.success_prob,
                    r_cur=1.0, r_v=float(hi), w_v=w_v[nid], beta=beta,
                )
                plan[nid] = lo if obj_lo <= obj_hi else hi

            # Step 2: English note
            total_cost = 0.0
            for nid in all_nodes:
                tpl = dag.nodes[nid].assigned_template
                total_cost += expected_compute_cost_surrogate(
                    delta=0.0, t_v=tpl.lat_mean, p_v=tpl.success_prob,
                    r_cur=1.0, r_v=float(plan[nid]),
                )

            is_feasible = total_cost <= budget

            if is_feasible:
                node_cost = {
                    nid: expected_latency_surrogate(
                        delta=0.0,
                        t_v=dag.nodes[nid].assigned_template.lat_mean,
                        p_v=dag.nodes[nid].assigned_template.success_prob,
                        r_cur=1.0,
                        r_v=float(plan[nid]),
                    )
                    for nid in all_nodes
                }
                plan_score = dag_longest_path_value(dag, node_cost)
                if plan_score < best_feasible_score:
                    best_feasible_score = plan_score
                    best_feasible_plan = dict(plan)

            # Step 3: English note
            if prev_plan is not None and all(
                plan.get(nid) == prev_plan.get(nid) for nid in all_nodes
            ):
                stable_count += 1
            else:
                stable_count = 0
            prev_plan = dict(plan)

            if stable_count >= stable_rounds:
                if is_feasible:
                    node_cost = {
                        nid: expected_latency_surrogate(
                            delta=0.0,
                            t_v=dag.nodes[nid].assigned_template.lat_mean,
                            p_v=dag.nodes[nid].assigned_template.success_prob,
                            r_cur=1.0,
                            r_v=float(plan[nid]),
                        )
                        for nid in all_nodes
                    }
                    plan_score = dag_longest_path_value(dag, node_cost)
                    if plan_score < best_feasible_score:
                        best_feasible_score = plan_score
                        best_feasible_plan = dict(plan)
                break

            # Step 4: English note + MWU
            beta = max(1e-8, beta + eta_beta * (total_cost - budget))

            path_vals = np.array([
                sum(
                    expected_latency_surrogate(
                        delta=0.0, t_v=dag.nodes[nid].assigned_template.lat_mean,
                        p_v=dag.nodes[nid].assigned_template.success_prob,
                        r_cur=1.0, r_v=float(plan.get(nid, 1)),
                    )
                    for nid in path
                )
                for path in active_paths
            ], dtype=float)
            norm = max(float(path_vals.max()), 1e-12)
            gains = path_vals / norm
            lambdas = lambdas * np.exp(eta_lambda * gains)
            lambdas = normalize_simplex(lambdas)

        # English note restart English note surrogate makespan English note
        if best_feasible_plan is not None:
            return best_feasible_plan, best_feasible_score

        return None, float('inf')

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready = sim.ready_nodes()
        if not ready:
            return {}
        return {nid: self.desired.get(nid, 1) for nid in ready}

class _LoaderDualExactBase(Policy):
    """
    English note LOADER English note:
    - English note budget_slack English note
    - English note top-k diversity heuristic
    - English note:Lambert-W best response,English note,English note MWU,plan English note
    """

    def __init__(
        self,
        full_dag: DAG,
        max_dual_iters: int = 30,
        stable_rounds: int = 3,
        eta_beta: float = 0.05,
        eta_lambda: float = 0.5,
    ):
        self.full_dag = full_dag
        self.max_dual_iters = max_dual_iters
        self.stable_rounds = stable_rounds
        self.eta_beta = eta_beta
        self.eta_lambda = eta_lambda

        # English note:English note DAG English note(remaining_dag English note completed_nodes,English note)
        self._cached_completed: Optional[frozenset] = None
        self._cached_remaining_dag: Optional[DAG] = None

    def _remaining_dag(self, sim: Simulator) -> DAG:
        current_completed = frozenset(sim.completed_nodes)
        if current_completed != self._cached_completed:
            self._cached_completed = current_completed
            self._cached_remaining_dag = self.full_dag.induced_remaining_subgraph(sim.completed_nodes)
        return self._cached_remaining_dag

    def _path_latency(self, sim: Simulator, path: List[int], plan: Dict[int, int]) -> float:
        total = 0.0
        for nid in path:
            st = sim.runtime[nid]
            delta = sim.node_delta_to_round_end(nid)
            r_cur = float(sim.running_count(nid))
            r_v = float(plan.get(nid, max(1, int(r_cur))))
            total += expected_latency_surrogate(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_cur,
                r_v=r_v,
            )
        return total

    def _plan_given_duals(
        self,
        sim: Simulator,
        all_remaining_nodes: List[int],
        active_paths: List[List[int]],
        lambdas: np.ndarray,
        beta: float,
    ) -> Dict[int, int]:
        # w_v = sum_{P contains v} lambda_P
        w_v = {nid: 0.0 for nid in all_remaining_nodes}
        for lam, path in zip(lambdas, active_paths):
            for nid in path:
                if nid in w_v:
                    w_v[nid] += float(lam)

        # English note active set English note,English note,English note
        for nid in all_remaining_nodes:
            w_v[nid] = max(w_v[nid], 1e-3)

        plan: Dict[int, int] = {}
        for nid in all_remaining_nodes:
            st = sim.runtime[nid]
            delta = sim.node_delta_to_round_end(nid)
            r_cur = float(sim.running_count(nid))

            r_cont = lambert_closed_form_r(
                p_v=st.template.success_prob,
                w_v=w_v[nid],
                beta=beta,
            )
            lo, hi = discretize_neighbor_pair(
                r_cont=r_cont,
                min_r=1,
            )

            obj_lo = node_objective(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_cur,
                r_v=float(lo),
                w_v=w_v[nid],
                beta=beta,
            )
            obj_hi = node_objective(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_cur,
                r_v=float(hi),
                w_v=w_v[nid],
                beta=beta,
            )

            plan[nid] = lo if obj_lo <= obj_hi else hi

        return plan

    def _build_node_snap(self, sim: Simulator, all_remaining_nodes: List[int]) -> Dict[int, Tuple[float, float, float, float]]:
        """English note (delta, r_cur, t_v, p_v),English note allocate English note."""
        snap: Dict[int, Tuple[float, float, float, float]] = {}
        for nid in all_remaining_nodes:
            st = sim.runtime[nid]
            snap[nid] = (
                sim.node_delta_to_round_end(nid),
                float(sim.running_count(nid)),
                st.template.lat_mean,
                st.template.success_prob,
            )
        return snap

    def _plan_given_duals_fast(
        self,
        all_remaining_nodes: List[int],
        active_paths: List[List[int]],
        lambdas: np.ndarray,
        beta: float,
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> Dict[int, int]:
        """_plan_given_duals English note:English note node_snap English note sim."""
        w_v = {nid: 0.0 for nid in all_remaining_nodes}
        for lam, path in zip(lambdas, active_paths):
            for nid in path:
                if nid in w_v:
                    w_v[nid] += float(lam)

        for nid in all_remaining_nodes:
            w_v[nid] = max(w_v[nid], 1e-3)

        plan: Dict[int, int] = {}
        for nid in all_remaining_nodes:
            delta, r_cur, t_v, p_v = node_snap[nid]

            r_cont = lambert_closed_form_r(p_v=p_v, w_v=w_v[nid], beta=beta)
            lo, hi = discretize_neighbor_pair(r_cont=r_cont, min_r=1)

            obj_lo = node_objective(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=float(lo),
                w_v=w_v[nid], beta=beta,
            )
            obj_hi = node_objective(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=float(hi),
                w_v=w_v[nid], beta=beta,
            )

            plan[nid] = lo if obj_lo <= obj_hi else hi

        return plan

    def _path_latency_fast(
        self,
        path: List[int],
        plan: Dict[int, int],
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        """_path_latency English note:English note node_snap."""
        total = 0.0
        for nid in path:
            if nid not in node_snap:
                continue  # English note,English note
            delta, r_cur, t_v, p_v = node_snap[nid]
            r_v = float(plan.get(nid, max(1, int(r_cur))))
            total += expected_latency_surrogate(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=r_v,
            )
        return total

    def _total_surrogate_cost_fast(
        self,
        all_remaining_nodes: List[int],
        plan: Dict[int, int],
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        total_cost = 0.0
        for nid in all_remaining_nodes:
            delta, r_cur, t_v, p_v = node_snap[nid]
            r_v = float(plan[nid])
            total_cost += expected_compute_cost_surrogate(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=r_v,
            )
        return total_cost

    def _plan_node_costs_fast(
        self,
        all_remaining_nodes: List[int],
        plan: Dict[int, int],
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> Dict[int, float]:
        node_cost: Dict[int, float] = {}
        for nid in all_remaining_nodes:
            delta, r_cur, t_v, p_v = node_snap[nid]
            r_v = float(plan.get(nid, max(1, int(r_cur))))
            node_cost[nid] = expected_latency_surrogate(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=r_v,
            )
        return node_cost

    def _plan_surrogate_makespan_fast(
        self,
        remaining_dag: DAG,
        all_remaining_nodes: List[int],
        plan: Dict[int, int],
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        node_cost = self._plan_node_costs_fast(all_remaining_nodes, plan, node_snap)
        return dag_longest_path_value(remaining_dag, node_cost)

    def _maybe_update_best_feasible_fast(
        self,
        remaining_dag: DAG,
        all_remaining_nodes: List[int],
        plan: Dict[int, int],
        node_snap: Dict[int, Tuple[float, float, float, float]],
        best_plan: Optional[Dict[int, int]],
        best_score: float,
    ) -> Tuple[Optional[Dict[int, int]], float]:
        plan_score = self._plan_surrogate_makespan_fast(
            remaining_dag=remaining_dag,
            all_remaining_nodes=all_remaining_nodes,
            plan=plan,
            node_snap=node_snap,
        )
        if plan_score < best_score:
            return dict(plan), plan_score
        return best_plan, best_score

    def _total_surrogate_cost(
        self,
        sim: Simulator,
        all_remaining_nodes: List[int],
        plan: Dict[int, int],
    ) -> float:
        total_cost = 0.0
        for nid in all_remaining_nodes:
            st = sim.runtime[nid]
            delta = sim.node_delta_to_round_end(nid)
            r_cur = float(sim.running_count(nid))
            r_v = float(plan[nid])
            total_cost += expected_compute_cost_surrogate(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_cur,
                r_v=r_v,
            )
        return total_cost
# -----------------------------
# LOADER: PDF-faithful version
# -----------------------------
class LoaderDualAllPathsPolicy(_LoaderDualExactBase):
    """
    English note A:English note
    English note remaining DAG English note source-to-sink English note lambda_P.

    English note"lambda English note"English note,
    English note path English note,English note DAG / chain / English note fork-join English note.
    """

    def __init__(
        self,
        full_dag: DAG,
        max_dual_iters: int = 30,
        stable_rounds: int = 3,
        eta_beta: float = 0.05,
        eta_lambda: float = 0.5,
        max_paths: Optional[int] = 5000,
    ):
        super().__init__(
            full_dag=full_dag,
            max_dual_iters=max_dual_iters,
            stable_rounds=stable_rounds,
            eta_beta=eta_beta,
            eta_lambda=eta_lambda,
        )
        self.max_paths = max_paths
        # English note(English note DAG English note completed_nodes,English note)
        self._paths_completed: Optional[frozenset] = None
        self._cached_active_paths: Optional[List[List[int]]] = None

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready_nodes = sim.ready_nodes()
        if not ready_nodes:
            return {}

        remaining_dag = self._remaining_dag(sim)
        all_remaining_nodes = remaining_dag.node_ids
        if not all_remaining_nodes:
            return {}

        # English note:English note completed_nodes English note(English note,English note)
        current_completed = frozenset(sim.completed_nodes)
        if current_completed != self._paths_completed:
            self._paths_completed = current_completed
            try:
                paths = enumerate_all_source_sink_paths(
                    remaining_dag,
                    max_paths=self.max_paths,
                )
            except RuntimeError:
                self._cached_active_paths = None  # English note,English note
                return {nid: 1 for nid in ready_nodes}
            self._cached_active_paths = dedup_paths(paths)

        active_paths = self._cached_active_paths
        if not active_paths:
            return {nid: 1 for nid in ready_nodes}

        # English note:English note
        lambdas = np.ones(len(active_paths), dtype=float) / len(active_paths)
        beta = 1.0

        prev_plan: Optional[Dict[int, int]] = None
        stable_count = 0

        best_feasible_plan: Optional[Dict[int, int]] = None
        best_feasible_score = float("inf")

        # English note(allocate English note sim English note,English note)
        node_snap = self._build_node_snap(sim, all_remaining_nodes)

        for _ in range(self.max_dual_iters):
            plan = self._plan_given_duals_fast(
                all_remaining_nodes=all_remaining_nodes,
                active_paths=active_paths,
                lambdas=lambdas,
                beta=beta,
                node_snap=node_snap,
            )

            total_cost = self._total_surrogate_cost_fast(all_remaining_nodes, plan, node_snap)
            is_feasible = total_cost <= sim.remaining_budget

            if is_feasible:
                best_feasible_plan, best_feasible_score = self._maybe_update_best_feasible_fast(
                    remaining_dag=remaining_dag,
                    all_remaining_nodes=all_remaining_nodes,
                    plan=plan,
                    node_snap=node_snap,
                    best_plan=best_feasible_plan,
                    best_score=best_feasible_score,
                )

            if prev_plan is not None and all(
                plan.get(nid) == prev_plan.get(nid) for nid in all_remaining_nodes
            ):
                stable_count += 1
            else:
                stable_count = 0
            prev_plan = dict(plan)

            if stable_count >= self.stable_rounds:
                if is_feasible:
                    best_feasible_plan, best_feasible_score = self._maybe_update_best_feasible_fast(
                        remaining_dag=remaining_dag,
                        all_remaining_nodes=all_remaining_nodes,
                        plan=plan,
                        node_snap=node_snap,
                        best_plan=best_feasible_plan,
                        best_score=best_feasible_score,
                    )
                break

            beta = max(1e-12, beta + self.eta_beta * (total_cost - sim.remaining_budget))

            path_vals = np.array(
                [self._path_latency_fast(path, plan, node_snap) for path in active_paths],
                dtype=float,
            )
            norm = max(float(path_vals.max()), 1e-12)
            gains = path_vals / norm
            lambdas = lambdas * np.exp(self.eta_lambda * gains)
            lambdas = normalize_simplex(lambdas)

        if best_feasible_plan is not None:
            return {nid: best_feasible_plan[nid] for nid in ready_nodes}

        return {nid: 1 for nid in ready_nodes}
    
class LoaderDualColumnGenPolicy(_LoaderDualExactBase):
    """
    English note B:English note / column generation English note

    English note:
    1. active set English note surrogate English note;
    2. English note restricted active set English note MWU + beta English note;
    3. English note plan English note d_tilde English note node cost;
    4. English note remaining DAG English note"English note"English note(English note);
    5. English note active set English note,English note;
    6. English note.

    English note top-k + 0.5 English note.
    """

    def __init__(
        self,
        full_dag: DAG,
        max_dual_iters: int = 30,
        stable_rounds: int = 3,
        eta_beta: float = 0.05,
        eta_lambda: float = 0.5,
        max_cg_iters: int = 8,
        inner_dual_iters: int = 8,
        max_active_paths: int = 64,
        add_path_tol: float = 1e-6,
    ):
        super().__init__(
            full_dag=full_dag,
            max_dual_iters=max_dual_iters,
            stable_rounds=stable_rounds,
            eta_beta=eta_beta,
            eta_lambda=eta_lambda,
        )
        self.max_cg_iters = max_cg_iters
        self.inner_dual_iters = inner_dual_iters
        self.max_active_paths = max_active_paths
        self.add_path_tol = add_path_tol

    def _initial_path(self, sim: Simulator, remaining_dag: DAG) -> List[int]:
        init_cost: Dict[int, float] = {}
        for nid in remaining_dag.node_ids:
            st = sim.runtime[nid]
            delta = sim.node_delta_to_round_end(nid)
            r_ref = max(1.0, float(sim.running_count(nid)))
            init_cost[nid] = expected_latency_surrogate(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_ref,
                r_v=r_ref,
            )
        return recover_one_critical_path(remaining_dag, init_cost)

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready_nodes = sim.ready_nodes()
        if not ready_nodes:
            return {}

        remaining_dag = self._remaining_dag(sim)
        all_remaining_nodes = remaining_dag.node_ids
        if not all_remaining_nodes:
            return {}

        first_path = self._initial_path(sim, remaining_dag)
        if not first_path:
            return {nid: 1 for nid in ready_nodes}

        active_paths: List[List[int]] = [first_path]
        active_set: Set[Tuple[int, ...]] = {tuple_path(first_path)}

        lambdas = np.array([1.0], dtype=float)
        beta = 1.0

        global_best_plan: Optional[Dict[int, int]] = None
        global_best_score = float("inf")

        # English note
        node_snap = self._build_node_snap(sim, all_remaining_nodes)

        for _cg in range(self.max_cg_iters):
            prev_plan: Optional[Dict[int, int]] = None
            stable_count = 0

            # ---- English note restricted active set English note ----
            for _inner in range(min(self.inner_dual_iters, self.max_dual_iters)):
                plan = self._plan_given_duals_fast(
                    all_remaining_nodes=all_remaining_nodes,
                    active_paths=active_paths,
                    lambdas=lambdas,
                    beta=beta,
                    node_snap=node_snap,
                )

                total_cost = self._total_surrogate_cost_fast(all_remaining_nodes, plan, node_snap)
                is_feasible = total_cost <= sim.remaining_budget

                if is_feasible:
                    global_best_plan, global_best_score = self._maybe_update_best_feasible_fast(
                        remaining_dag=remaining_dag,
                        all_remaining_nodes=all_remaining_nodes,
                        plan=plan,
                        node_snap=node_snap,
                        best_plan=global_best_plan,
                        best_score=global_best_score,
                    )

                if prev_plan is not None and all(
                    plan.get(nid) == prev_plan.get(nid) for nid in all_remaining_nodes
                ):
                    stable_count += 1
                else:
                    stable_count = 0
                prev_plan = dict(plan)

                if stable_count >= self.stable_rounds:
                    if is_feasible:
                        global_best_plan, global_best_score = self._maybe_update_best_feasible_fast(
                            remaining_dag=remaining_dag,
                            all_remaining_nodes=all_remaining_nodes,
                            plan=plan,
                            node_snap=node_snap,
                            best_plan=global_best_plan,
                            best_score=global_best_score,
                        )
                    break

                beta = max(1e-12, beta + self.eta_beta * (total_cost - sim.remaining_budget))

                path_vals = np.array(
                    [self._path_latency_fast(path, plan, node_snap) for path in active_paths],
                    dtype=float,
                )
                norm = max(float(path_vals.max()), 1e-12)
                gains = path_vals / norm
                lambdas = lambdas * np.exp(self.eta_lambda * gains)
                lambdas = normalize_simplex(lambdas)

            if prev_plan is None:
                break

            # ---- English note:English note"English note" ----
            # English note node_snap English note sim
            node_cost: Dict[int, float] = {}
            for nid in all_remaining_nodes:
                delta, r_cur, t_v, p_v = node_snap[nid]
                r_v = float(prev_plan.get(nid, max(1, int(r_cur))))
                node_cost[nid] = expected_latency_surrogate(
                    delta=delta, t_v=t_v, p_v=p_v,
                    r_cur=r_cur, r_v=r_v,
                )

            new_path = longest_path_not_in_set(
                dag=remaining_dag,
                node_cost=node_cost,
                excluded=active_set,
            )

            if not new_path:
                break

            new_path_value = sum(node_cost[u] for u in new_path)
            cur_active_best = max(
                sum(node_cost[u] for u in p) for p in active_paths
            )

            # English note active set English note"English note",English note
            if new_path_value <= cur_active_best + self.add_path_tol:
                break

            if len(active_paths) >= self.max_active_paths:
                break

            active_paths.append(new_path)
            active_set.add(tuple_path(new_path))
            lambdas = extend_simplex_with_new_coordinate(lambdas, eps_new=0.05)

        if global_best_plan is not None:
            return {nid: global_best_plan[nid] for nid in ready_nodes}

        return {nid: 1 for nid in ready_nodes}
    
class LoaderDualPolicy(Policy):
    def __init__(
        self,
        full_dag: DAG,
        max_dual_iters: int = 30,
        stable_rounds: int = 3,
        eta_beta: float = 0.05,
        eta_lambda: float = 0.5,
        n_active_paths: int = 4,
    ):
        self.full_dag = full_dag
        self.max_dual_iters = max_dual_iters
        self.stable_rounds = stable_rounds
        self.eta_beta = eta_beta
        self.eta_lambda = eta_lambda
        self.n_active_paths = n_active_paths
        # English note:English note completed_nodes English note
        self._cached_completed: Optional[frozenset] = None
        self._cached_remaining_dag: Optional[DAG] = None

    def _remaining_dag(self, sim: Simulator) -> DAG:
        current_completed = frozenset(sim.completed_nodes)
        if current_completed != self._cached_completed:
            self._cached_completed = current_completed
            self._cached_remaining_dag = self.full_dag.induced_remaining_subgraph(sim.completed_nodes)
        return self._cached_remaining_dag

    def _node_proxy_cost_for_path_generation(self, sim: Simulator, remaining_dag: DAG) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for nid in remaining_dag.node_ids:
            st = sim.runtime[nid]
            delta = sim.node_delta_to_round_end(nid)
            # English note,English note;English note,English note 1 English note
            # English note PDF English note r_v >= 1 English note 
            r_ref = max(1.0, float(sim.running_count(nid))) 
            
            out[nid] = expected_latency_surrogate(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_ref, # English note r_cur English note [cite: 27]
                r_v=r_ref,   # English note r_v English note [cite: 27]
            )
        return out

    def _active_paths(self, sim: Simulator, remaining_dag: DAG) -> List[List[int]]:
        node_cost = self._node_proxy_cost_for_path_generation(sim, remaining_dag)
        return recover_top_k_paths(remaining_dag, node_cost, k=self.n_active_paths)

    def _path_latency(self, sim: Simulator, path: List[int], plan: Dict[int, int]) -> float:
        total = 0.0
        for nid in path:
            st = sim.runtime[nid]
            delta = sim.node_delta_to_round_end(nid)
            r_cur = float(sim.running_count(nid))
            r_v = float(plan.get(nid, max(1, int(r_cur))))
            total += expected_latency_surrogate(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_cur,
                r_v=r_v,
            )
        return total

    def _plan_given_duals(
        self,
        sim: Simulator,
        ready_nodes: List[int],
        active_paths: List[List[int]],
        lambdas: np.ndarray,
        beta: float,
    ) -> Dict[int, int]:
        # PDF: w_v = sum_{P contains v} lambda_P
        w_v = {nid: 0.0 for nid in ready_nodes}
        for lam, path in zip(lambdas, active_paths):
            for nid in path:
                if nid in w_v:
                    w_v[nid] += float(lam)

        # English note active path English note ready English note,English note
        for nid in ready_nodes:
            w_v[nid] = max(w_v[nid], 1e-3)

        plan: Dict[int, int] = {}
        for nid in ready_nodes:
            st = sim.runtime[nid]
            delta = sim.node_delta_to_round_end(nid)
            r_cur = float(sim.running_count(nid))

            r_cont = lambert_closed_form_r(
                p_v=st.template.success_prob,
                w_v=w_v[nid],
                beta=beta,
            )

            lo, hi = discretize_neighbor_pair(
                r_cont=r_cont,
                min_r=1,
            )

            obj_lo = node_objective(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_cur,
                r_v=float(lo),
                w_v=w_v[nid],
                beta=beta,
            )
            obj_hi = node_objective(
                delta=delta,
                t_v=st.template.lat_mean,
                p_v=st.template.success_prob,
                r_cur=r_cur,
                r_v=float(hi),
                w_v=w_v[nid],
                beta=beta,
            )

            plan[nid] = lo if obj_lo <= obj_hi else hi

        return plan

    def _plan_given_duals_fast(
        self,
        ready_nodes: List[int],
        active_paths: List[List[int]],
        lambdas: np.ndarray,
        beta: float,
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> Dict[int, int]:
        """_plan_given_duals English note:English note node_snap English note sim."""
        w_v = {nid: 0.0 for nid in ready_nodes}
        for lam, path in zip(lambdas, active_paths):
            for nid in path:
                if nid in w_v:
                    w_v[nid] += float(lam)

        for nid in ready_nodes:
            w_v[nid] = max(w_v[nid], 1e-3)

        plan: Dict[int, int] = {}
        for nid in ready_nodes:
            delta, r_cur, t_v, p_v = node_snap[nid]

            r_cont = lambert_closed_form_r(p_v=p_v, w_v=w_v[nid], beta=beta)
            lo, hi = discretize_neighbor_pair(r_cont=r_cont, min_r=1)

            obj_lo = node_objective(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=float(lo),
                w_v=w_v[nid], beta=beta,
            )
            obj_hi = node_objective(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=float(hi),
                w_v=w_v[nid], beta=beta,
            )

            plan[nid] = lo if obj_lo <= obj_hi else hi

        return plan

    def _path_latency_fast(
        self,
        path: List[int],
        plan: Dict[int, int],
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        """_path_latency English note:English note node_snap."""
        total = 0.0
        for nid in path:
            if nid not in node_snap:
                continue  # English note,English note
            delta, r_cur, t_v, p_v = node_snap[nid]
            r_v = float(plan.get(nid, max(1, int(r_cur))))
            total += expected_latency_surrogate(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=r_v,
            )
        return total

    def _plan_node_costs_fast(
        self,
        all_remaining_nodes: List[int],
        plan: Dict[int, int],
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> Dict[int, float]:
        node_cost: Dict[int, float] = {}
        for nid in all_remaining_nodes:
            delta, r_cur, t_v, p_v = node_snap[nid]
            r_v = float(plan.get(nid, max(1, int(r_cur))))
            node_cost[nid] = expected_latency_surrogate(
                delta=delta, t_v=t_v, p_v=p_v,
                r_cur=r_cur, r_v=r_v,
            )
        return node_cost

    def _plan_surrogate_makespan_fast(
        self,
        remaining_dag: DAG,
        all_remaining_nodes: List[int],
        plan: Dict[int, int],
        node_snap: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        node_cost = self._plan_node_costs_fast(all_remaining_nodes, plan, node_snap)
        return dag_longest_path_value(remaining_dag, node_cost)

    def allocate(self, sim: Simulator) -> Dict[int, int]:
        ready_nodes = sim.ready_nodes()
        if not ready_nodes:
            return {}

        remaining_dag = self._remaining_dag(sim)
        active_paths = self._active_paths(sim, remaining_dag)
        if len(active_paths) == 0:
            return {nid: 1 for nid in ready_nodes}

        # English note:English note
        lambdas = np.ones(len(active_paths), dtype=float) / len(active_paths)
        beta = 1.0

        prev_plan: Optional[Dict[int, int]] = None
        stable_count = 0
        
        best_feasible_plan: Optional[Dict[int, int]] = None
        best_feasible_score = float("inf")

        all_remaining_nodes = remaining_dag.node_ids

        # English note:allocate English note sim.time/event_queue English note,English note
        _node_snap: Dict[int, Tuple[float, float, float, float]] = {}
        for nid in all_remaining_nodes:
            st = sim.runtime[nid]
            delta = sim.node_delta_to_round_end(nid)
            r_cur = float(sim.running_count(nid))
            _node_snap[nid] = (delta, r_cur, st.template.lat_mean, st.template.success_prob)

        for step in range(self.max_dual_iters):
            plan = self._plan_given_duals_fast(all_remaining_nodes, active_paths, lambdas, beta, _node_snap)

            # English note
            total_cost = 0.0
            for nid in all_remaining_nodes:
                delta, r_cur, t_v, p_v = _node_snap[nid]
                r_v = float(plan[nid])
                total_cost += expected_compute_cost_surrogate(
                    delta=delta, t_v=t_v, p_v=p_v,
                    r_cur=r_cur, r_v=r_v,
                )

            is_feasible = total_cost <= sim.remaining_budget

            # English note surrogate makespan English note
            if is_feasible:
                plan_score = self._plan_surrogate_makespan_fast(
                    remaining_dag=remaining_dag,
                    all_remaining_nodes=all_remaining_nodes,
                    plan=plan,
                    node_snap=_node_snap,
                )
                if plan_score < best_feasible_score:
                    best_feasible_score = plan_score
                    best_feasible_plan = dict(plan)

            # PDF English note
            if prev_plan is not None and all(plan.get(nid) == prev_plan.get(nid) for nid in all_remaining_nodes):
                stable_count += 1
            else:
                stable_count = 0
            prev_plan = dict(plan)

            if stable_count >= self.stable_rounds:
                if is_feasible:
                    plan_score = self._plan_surrogate_makespan_fast(
                        remaining_dag=remaining_dag,
                        all_remaining_nodes=all_remaining_nodes,
                        plan=plan,
                        node_snap=_node_snap,
                    )
                    if plan_score < best_feasible_score:
                        best_feasible_score = plan_score
                        best_feasible_plan = dict(plan)
                break

            # English note MWU English note
            beta = max(1e-8, beta + self.eta_beta * (total_cost - sim.remaining_budget))

            path_vals = np.array(
                [self._path_latency_fast(path, plan, _node_snap) for path in active_paths],
                dtype=float
            )
            norm = max(float(path_vals.max()), 1e-12)
            gains = path_vals / norm

            lambdas = lambdas * np.exp(self.eta_lambda * gains)
            lambdas = normalize_simplex(lambdas)

        # English note
        if best_feasible_plan is not None:
            return {nid: best_feasible_plan[nid] for nid in ready_nodes}
        
        return {nid: 1 for nid in ready_nodes}