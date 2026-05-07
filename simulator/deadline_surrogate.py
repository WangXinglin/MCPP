import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .dag import DAG


def logsumexp(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return float("-inf")
    vmax = max(vals)
    if not math.isfinite(vmax):
        return vmax
    return vmax + math.log(sum(math.exp(v - vmax) for v in vals))


def clamp_failure_prob(p_v: float) -> float:
    return min(max(1.0 - p_v, 1e-12), 1.0 - 1e-12)


def min_feasible_r(theta: float, tau_v: float, p_v: float, eps: float = 1e-6) -> float:
    q_v = clamp_failure_prob(p_v)
    b_v = -math.log(q_v)
    return max(eps, (theta * tau_v) / max(b_v, 1e-12) + eps)


# deadline-adaptive budget reservation cost
def expected_work_surrogate(
    kappa_v: float,
    p_v: float,
    r_v: float,
    reservation_alpha: float = 0.0,
) -> float:
    r_eff = max(r_v, 1e-9)
    alpha = min(max(float(reservation_alpha), 0.0), 1.0)
    batch_cost = kappa_v * r_eff
    if alpha <= 0.0:
        return batch_cost

    q_v = clamp_failure_prob(p_v)
    success_prob = max(1.0 - q_v ** r_eff, 1e-12)
    continuation_cost = batch_cost / success_prob
    return (1.0 - alpha) * batch_cost + alpha * continuation_cost


# expected_work_surrogateEnglish noter_vEnglish note
def expected_work_prime(
    kappa_v: float,
    p_v: float,
    r_v: float,
    reservation_alpha: float = 0.0,
) -> float:
    alpha = min(max(float(reservation_alpha), 0.0), 1.0)
    if alpha <= 0.0:
        return kappa_v

    q_v = clamp_failure_prob(p_v)
    r_eff = max(r_v, 1e-9)
    y_v = q_v ** r_eff
    b_v = -math.log(q_v)
    denom = max((1.0 - y_v) ** 2, 1e-24)
    continuation_prime = kappa_v * (1.0 - y_v - r_eff * b_v * y_v) / denom
    return (1.0 - alpha) * kappa_v + alpha * continuation_prime

# English noterisk cost (section 7)
# def node_risk_cost(theta: float, tau_v: float, p_v: float, r_v: float) -> float:
#     q_v = clamp_failure_prob(p_v)
#     r_eff = max(r_v, 1e-9)
#     y_v = q_v ** r_eff
#     ay = math.exp(theta * tau_v) * y_v
#     ay = min(max(ay, 1e-15), 1.0 - 1e-12) ### ？？？
#     return theta * tau_v + math.log(max(1.0 - y_v, 1e-15)) - math.log(max(1.0 - ay, 1e-15))
def node_risk_cost(theta: float, tau_v: float, p_v: float, r_v: float) -> float:
    q_v = clamp_failure_prob(p_v)   # must return q_v in (0, 1)
    r_eff = max(r_v, 1e-9)          # purely numerical safeguard

    # work in log-space first for stability
    log_y = r_eff * math.log(q_v)           # log(q_v^{r_v})
    log_ay = theta * tau_v + log_y          # log(q_v^{r_v} * exp(theta*tau_v))

    # PDF feasibility: q_v^{r_v} * exp(theta*tau_v) < 1
    # do NOT silently clamp an infeasible point back into feasibility
    if log_ay >= math.log(1.0 - 1e-12):
        return float("inf")   # or raise InfeasibleThetaError

    y_v = math.exp(log_y)
    ay = math.exp(log_ay)

    # now only do tiny clips for floating-point protection
    y_v = min(max(y_v, 1e-300), 1.0 - 1e-15)
    ay  = min(max(ay,  1e-300), 1.0 - 1e-15)

    return theta * tau_v + math.log1p(-y_v) - math.log1p(-ay)

# English note,English noterisk costEnglish noter_vEnglish note
def node_risk_prime(theta: float, tau_v: float, p_v: float, r_v: float) -> float:
    q_v = clamp_failure_prob(p_v)
    b_v = -math.log(q_v)
    r_eff = max(r_v, 1e-9)
    log_y = r_eff * math.log(q_v)
    log_ay = theta * tau_v + log_y
    if log_ay >= math.log(1.0 - 1e-12):
        return float("-inf")

    y_v = min(max(math.exp(log_y), 1e-300), 1.0 - 1e-15)
    ay = min(max(math.exp(log_ay), 1e-300), 1.0 - 1e-15)
    denom = (1.0 - y_v) * (1.0 - ay)
    if denom <= 1e-300:
        return float("-inf")
    return -b_v * (ay - y_v) / denom


### English note"English note + English note",English notew_v (section 5)
def compute_phi_and_weights(
    dag: DAG,
    psi_by_node: Dict[int, float],
) -> Tuple[float, Dict[int, float]]:
    if not dag.node_ids:
        return 0.0, {}

    order = dag.topological_order()
    log_f: Dict[int, float] = {}
    for nid in order:
        psi_v = psi_by_node[nid]
        parents = dag.nodes[nid].parents
        if not parents:
            log_f[nid] = psi_v
        else:
            log_f[nid] = psi_v + logsumexp(log_f[p] for p in parents)

    sinks = dag.sinks()
    log_z = logsumexp(log_f[nid] for nid in sinks)

    log_g: Dict[int, float] = {}
    for nid in reversed(order):
        children = dag.nodes[nid].children
        if not children:
            log_g[nid] = 0.0
        else:
            log_g[nid] = logsumexp(psi_by_node[ch] + log_g[ch] for ch in children)

    weights: Dict[int, float] = {}
    for nid in order:
        log_w = log_f[nid] + log_g[nid] - log_z
        # ！clipEnglish note,English note,English notebug
        weights[nid] = float(np.clip(math.exp(log_w), 0.0, 1.0))

    return float(log_z), weights


def deadline_risk_score(phi: float, theta: float, deadline: float) -> float:
    """
    Unclipped Chernoff risk score J = Phi_theta(r) - theta * deadline.

    Smaller is better. This is the quantity used for selecting theta; the
    clipped success lower bound below is only for reporting.
    """
    return float(phi - theta * deadline)


def deadline_success_lower_bound_from_score(score: float) -> float:
    """Clipped success lower bound for reporting."""
    score = float(score)
    if score >= 0.0:
        return 0.0
    if score <= -50.0:
        return 1.0
    return float(-math.expm1(score))


def deadline_success_lower_bound(phi: float, theta: float, deadline: float) -> float:
    return deadline_success_lower_bound_from_score(
        deadline_risk_score(phi, theta, deadline)
    )


def solve_node_subproblem(
    theta: float,
    tau_v: float,
    p_v: float,
    kappa_v: float,
    weight_v: float,
    beta: float,
    r_min: float,
    budget_hint: float,
    reservation_alpha: float = 0.0,
    tol: float = 1e-6,
    stats: Optional[Dict[str, int]] = None,
) -> float:
    if stats is not None:
        stats["bracket_iters"] = 0
        stats["bisect_iters"] = 0

    if weight_v <= 0.0 and beta <= 0.0:
        return float(r_min)
    if beta <= 0.0:
        # Without a positive budget multiplier, the bounded objective proxy
        # should still stay within the current hard budget scale.
        return float(max(r_min, budget_hint / max(kappa_v, 1e-9)))

    def deriv(rv: float) -> float:
        return (
            weight_v * node_risk_prime(theta, tau_v, p_v, rv)
            + beta * expected_work_prime(kappa_v, p_v, rv, reservation_alpha)
        )

    eps = max(1e-8, 1e-6 * max(1.0, r_min))
    lo = float(r_min) + eps
    dlo = deriv(lo)
    if dlo >= 0.0:
        return lo

    # upper_cap = max(lo + 1.0, budget_hint / max(kappa_v, 1e-9) + 8.0, 2.0 * lo + 8.0, 512.0)
    # hi = max(lo + 1.0, 2.0 * lo + 1.0)
    # dhi = deriv(hi)
    # while dhi < 0.0 and hi < upper_cap:
    #     hi = min(upper_cap, hi * 2.0)
    #     dhi = deriv(hi)

    # if dhi < 0.0:
    #     return hi

    hi = max(lo + 1.0, 2.0 * lo + 1.0)
    dhi = deriv(hi)
    for bracket_iter in range(60):
        if dhi >= 0.0:
            break
        hi *= 2.0
        dhi = deriv(hi)
        if stats is not None:
            stats["bracket_iters"] = bracket_iter + 1

    if dhi < 0.0:
        raise RuntimeError("Failed to bracket root although beta > 0.")

    #English note,English notederiv(rv) = 0English notervEnglish note English note10^-6English note80English note
    for bisect_iter in range(80):
        mid = 0.5 * (lo + hi)
        dmid = deriv(mid)
        if dmid <= 0.0:
            lo = mid
        else:
            hi = mid
        if stats is not None:
            stats["bisect_iters"] = bisect_iter + 1
        if hi - lo <= tol * max(1.0, lo):
            break
    return 0.5 * (lo + hi)


def solve_budgeted_subproblem(
    theta: float,
    node_params: Dict[int, Dict[str, float]],
    weights: Dict[int, float],
    budget: float,
    tol: float = 1e-5,
) -> Optional[Tuple[Dict[int, float], float, float]]:
    if not node_params:
        return {}, 0.0, 0.0

    mins = {nid: params["r_min"] for nid, params in node_params.items()}
    min_cost = sum(
        expected_work_surrogate(
            params["kappa"],
            params["p"],
            mins[nid],
            params.get("reservation_alpha", 0.0),
        )
        for nid, params in node_params.items()
    )
    if min_cost > budget + tol:
        return None

    def solve_for_beta(beta: float) -> Tuple[Dict[int, float], float]:
        plan: Dict[int, float] = {}
        total_cost = 0.0
        for nid, params in node_params.items():
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
                tol=tol,
            )
            plan[nid] = rv
            # English note
            total_cost += expected_work_surrogate(
                params["kappa"],
                params["p"],
                rv,
                params.get("reservation_alpha", 0.0),
            )
        return plan, total_cost

    plan_lo, cost_lo = solve_for_beta(0.0)
    if cost_lo <= budget + tol:
        return plan_lo, cost_lo, 0.0

    beta_lo = 0.0
    beta_hi = 1.0
    plan_hi, cost_hi = solve_for_beta(beta_hi)
    while cost_hi > budget + tol and beta_hi < 1e12:
        beta_lo = beta_hi
        beta_hi *= 2.0
        plan_hi, cost_hi = solve_for_beta(beta_hi)

    if cost_hi > budget + tol:
        return plan_hi, cost_hi, beta_hi

    best_plan = dict(plan_hi)
    best_cost = float(cost_hi)
    best_beta = float(beta_hi)

    for _ in range(80):
        beta_mid = 0.5 * (beta_lo + beta_hi)
        plan_mid, cost_mid = solve_for_beta(beta_mid)
        if cost_mid <= budget + tol:
            beta_hi = beta_mid
            best_plan = dict(plan_mid)
            best_cost = float(cost_mid)
            best_beta = float(beta_mid)
        else:
            beta_lo = beta_mid
        if abs(cost_mid - budget) <= tol * max(1.0, budget):
            break
        if beta_hi - beta_lo <= 1e-8 * max(1.0, beta_hi):
            break

    return best_plan, best_cost, best_beta
