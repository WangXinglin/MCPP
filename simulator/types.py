import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


BUDGET_COST_MODES = ("token_cost_mean", "output_money_cost_mean", "cost_mean", "lat_mean")
DEFAULT_MODEL_ID = "default"
OUTPUT_PRICE_PER_1M_TOKENS = {
    "30B_ins": 0.8,
    "30B_think": 2.4,
    "4B_ins": 0.42,
    "4B_think": 1.26,
    "goedel32b": 2.8,
    "goedel8b": 0.7,
    "kimina7b": 0.5,
}


def normalize_budget_cost_mode(mode: str = "token_cost_mean") -> str:
    mode = (mode or "token_cost_mean").strip()
    aliases = {
        "token": "token_cost_mean",
        "token_cost": "token_cost_mean",
        "token_cost_mean": "token_cost_mean",
        "money": "output_money_cost_mean",
        "money_cost": "output_money_cost_mean",
        "output_money": "output_money_cost_mean",
        "output_money_cost": "output_money_cost_mean",
        "output_money_cost_mean": "output_money_cost_mean",
        "cost": "cost_mean",
        "cost_mean": "cost_mean",
        "latency": "lat_mean",
        "latency_mean": "lat_mean",
        "lat_mean": "lat_mean",
    }
    normalized = aliases.get(mode)
    if normalized is None:
        raise ValueError(
            f"Unsupported budget_cost_mode={mode!r}; "
            f"expected one of {BUDGET_COST_MODES}"
        )
    return normalized


def output_price_per_million_tokens(model_id: str) -> float:
    model_id = str(model_id)
    if model_id == DEFAULT_MODEL_ID:
        model_id = os.environ.get("OUTPUT_MONEY_DEFAULT_MODEL_ID", model_id)
    try:
        return float(OUTPUT_PRICE_PER_1M_TOKENS[model_id])
    except KeyError as exc:
        raise ValueError(
            "output_money_cost_mean requires a known model_id. "
            f"Got model_id={model_id!r}; known ids={sorted(OUTPUT_PRICE_PER_1M_TOKENS)}. "
            "For legacy single-model DAG pools, set OUTPUT_MONEY_DEFAULT_MODEL_ID "
            "to the pool model id, e.g. 30B_ins."
        ) from exc


def output_money_cost_from_tokens(model_id: str, output_tokens: float) -> float:
    return float(max(float(output_tokens), 1e-9) * output_price_per_million_tokens(model_id) / 1_000_000.0)


@dataclass
class TaskTemplate:
    success_prob: float
    lat_mean: float
    lat_std: float
    cost_mean: float
    token_cost_mean: Optional[float] = None


@dataclass
class NodeSpec:
    node_id: int
    parents: List[int]
    children: List[int]
    layer: int
    assigned_template: TaskTemplate
    empirical_samples: Optional[List[dict]] = None  # [{"latency": ..., "success": ..., "cost": ...}, ...]
    model_templates: Optional[Dict[str, TaskTemplate]] = None
    empirical_samples_by_model: Optional[Dict[str, List[dict]]] = None
    legacy_template_model: Optional[str] = None


@dataclass(order=True)
class RunningRollout:
    finish_time: float
    rollout_id: int = field(compare=False)
    node_id: int = field(compare=False)
    success: bool = field(compare=False)
    cost: float = field(compare=False)
    model_id: str = field(default=DEFAULT_MODEL_ID, compare=False)


@dataclass
class NodeRuntimeState:
    node_id: int
    template: TaskTemplate
    parents: List[int]
    children: List[int]

    completed: bool = False
    unlocked: bool = False

    num_started: int = 0
    num_finished: int = 0
    num_success: int = 0
    cumulative_cost: float = 0.0

    running_rollouts: Set[int] = field(default_factory=set)


@dataclass
class SimulationResult:
    success: bool
    termination_reason: str
    makespan: float
    deadline: float
    remaining_deadline: float
    total_cost: float
    planner_calls: int
    planner_wall_ms: float
    total_rollouts: int
    completion_fraction: float
    node_stats: Dict[int, Dict[str, float]]


def _mean_positive(values: List[float], fallback: float) -> float:
    clean = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            clean.append(value)
    if not clean:
        return float(max(fallback, 1e-9))
    return float(max(sum(clean) / len(clean), 1e-9))


def get_model_templates_for_node(node: NodeSpec) -> Dict[str, TaskTemplate]:
    """
    Return the real model templates available for a node.

    Old single-model nodes expose assigned_template as the default model. For
    model-aware nodes, only model_templates keys are real candidate models; an
    assigned_template compatibility alias does not add DEFAULT_MODEL_ID.
    """
    if node.model_templates:
        return {str(model_id): tpl for model_id, tpl in node.model_templates.items()}
    if node.assigned_template is not None:
        return {DEFAULT_MODEL_ID: node.assigned_template}
    return {}


def get_model_ids_for_node(node: NodeSpec) -> Tuple[str, ...]:
    return tuple(get_model_templates_for_node(node).keys())


def get_empirical_samples_for_node_model(
    node: NodeSpec,
    model_id: str = DEFAULT_MODEL_ID,
) -> Optional[List[dict]]:
    model_id = str(model_id)
    if node.empirical_samples_by_model and model_id in node.empirical_samples_by_model:
        return node.empirical_samples_by_model[model_id]
    if model_id == DEFAULT_MODEL_ID:
        return node.empirical_samples
    return None


def pricing_model_id_for_node(node: NodeSpec) -> str:
    legacy_model = getattr(node, "legacy_template_model", None)
    if legacy_model:
        return str(legacy_model)
    if node.model_templates:
        if DEFAULT_MODEL_ID in node.model_templates:
            return DEFAULT_MODEL_ID
        return str(next(iter(node.model_templates)))
    return os.environ.get("OUTPUT_MONEY_DEFAULT_MODEL_ID", DEFAULT_MODEL_ID)


def expected_node_work_cost(
    template: TaskTemplate,
    empirical_samples: Optional[List[dict]] = None,
    budget_cost_mode: str = "token_cost_mean",
    model_id: str = DEFAULT_MODEL_ID,
) -> float:
    """
    Expected per-rollout work cost used by nominal budget and planners.

    token_cost_mean uses template.token_cost_mean when available, otherwise
    empirical sample["token_cost"]/["total_tokens"] and finally sample["cost"].
    output_money_cost_mean uses the same output-token estimate as
    token_cost_mean and converts it to dollars with the model-specific output
    price.
    cost_mean uses empirical sample["cost"] when available and falls back to
    token cost fields only for backwards compatibility. lat_mean uses sampled
    latency statistics.
    """
    mode = normalize_budget_cost_mode(budget_cost_mode)
    if mode == "lat_mean":
        if empirical_samples:
            return _mean_positive(
                [
                    sample.get("latency", sample.get("total_latency_sec", template.lat_mean))
                    for sample in empirical_samples
                    if isinstance(sample, dict)
                ],
                template.lat_mean,
            )
        return float(max(template.lat_mean, 1e-9))

    if mode in ("token_cost_mean", "output_money_cost_mean"):
        token_cost_mean = getattr(template, "token_cost_mean", None)
        if token_cost_mean is not None:
            try:
                token_value = float(max(float(token_cost_mean), 1e-9))
            except (TypeError, ValueError):
                pass
            else:
                if mode == "output_money_cost_mean":
                    return output_money_cost_from_tokens(model_id, token_value)
                return token_value
        if empirical_samples:
            token_value = _mean_positive(
                [
                    sample.get(
                        "token_cost",
                        sample.get("total_tokens", sample.get("cost", template.cost_mean)),
                    )
                    for sample in empirical_samples
                    if isinstance(sample, dict)
                ],
                template.cost_mean,
            )
            if mode == "output_money_cost_mean":
                return output_money_cost_from_tokens(model_id, token_value)
            return token_value
        if mode == "output_money_cost_mean":
            return output_money_cost_from_tokens(model_id, template.cost_mean)
        return float(max(template.cost_mean, 1e-9))

    if empirical_samples:
        return _mean_positive(
            [
                sample.get(
                    "cost",
                    sample.get("token_cost", sample.get("total_tokens", template.cost_mean)),
                )
                for sample in empirical_samples
                if isinstance(sample, dict)
            ],
            template.cost_mean,
        )
    return float(max(template.cost_mean, 1e-9))


def realized_rollout_work_cost(
    template: TaskTemplate,
    sampled_latency: float,
    empirical_sample: Optional[dict] = None,
    budget_cost_mode: str = "token_cost_mean",
    model_id: str = DEFAULT_MODEL_ID,
) -> float:
    """
    Per-rollout work cost charged by Simulator.start_rollout().

    This must stay aligned with expected_node_work_cost(): lat_mean mode charges
    sampled latency, token_cost_mean mode charges sampled token cost,
    output_money_cost_mean charges sampled output-token cost converted to
    dollars by model_id, and cost_mean mode charges sampled cost. Fallbacks
    keep older DAG pools usable.
    """
    mode = normalize_budget_cost_mode(budget_cost_mode)
    if mode == "lat_mean":
        return float(max(sampled_latency, 1e-9))
    if empirical_sample is not None:
        if mode in ("token_cost_mean", "output_money_cost_mean"):
            fallback = getattr(template, "token_cost_mean", None)
            if fallback is None:
                fallback = template.cost_mean
            value = empirical_sample.get(
                "token_cost",
                empirical_sample.get("total_tokens", empirical_sample.get("cost", fallback)),
            )
        else:
            value = empirical_sample.get(
                "cost",
                empirical_sample.get("token_cost", empirical_sample.get("total_tokens", template.cost_mean)),
            )
        try:
            value = float(max(float(value), 1e-9))
        except (TypeError, ValueError):
            fallback_value = (
                getattr(template, "token_cost_mean", None)
                if mode in ("token_cost_mean", "output_money_cost_mean")
                else None
            )
            if fallback_value is None:
                fallback_value = template.cost_mean
            if mode == "output_money_cost_mean":
                return output_money_cost_from_tokens(model_id, fallback_value)
            return float(max(float(fallback_value), 1e-9))
        if mode == "output_money_cost_mean":
            return output_money_cost_from_tokens(model_id, value)
        return value
    if mode in ("token_cost_mean", "output_money_cost_mean") and getattr(template, "token_cost_mean", None) is not None:
        token_value = float(max(float(template.token_cost_mean), 1e-9))
        if mode == "output_money_cost_mean":
            return output_money_cost_from_tokens(model_id, token_value)
        return token_value
    if mode == "output_money_cost_mean":
        return output_money_cost_from_tokens(model_id, template.cost_mean)
    return float(max(template.cost_mean, 1e-9))
