import argparse
import json
import math
import os
from typing import Dict, List, Tuple


def _compute_status(subproblem: dict) -> str:
    rollout_records = subproblem.get("rollout_records") or []
    if rollout_records:
        valid_success = [int(r.get("success", 0)) for r in rollout_records if isinstance(r, dict)]
        if not valid_success:
            return "unknown"
        if any(x == 1 for x in valid_success):
            return "success"
        return "failed"

    results = subproblem.get("harness_result") or []
    valid = [int(x) for x in results if isinstance(x, (int, float)) and int(x) in (0, 1)]
    if not valid:
        return "unknown"
    if all(x == 1 for x in valid):
        return "success"
    return "failed"


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _work_cost_from_payload(payload: dict) -> float | None:
    token_cost = _token_cost_from_payload(payload)
    if token_cost is not None:
        return token_cost

    explicit_cost = _safe_float(payload.get("cost"))
    if explicit_cost is not None:
        return max(explicit_cost, 1e-9)

    total_tokens = _safe_float(payload.get("total_tokens"))
    if total_tokens is not None:
        return max(total_tokens, 1e-9)

    prompt_tokens = _safe_float(payload.get("prompt_tokens"))
    output_tokens = _safe_float(payload.get("output_tokens"))
    if prompt_tokens is not None or output_tokens is not None:
        return max((prompt_tokens or 0.0) + (output_tokens or 0.0), 1e-9)

    return None


def _token_cost_from_payload(payload: dict) -> float | None:
    output_tokens = _safe_float(payload.get("output_tokens"))
    if output_tokens is not None:
        return max(output_tokens, 1e-9)
    return None


def _work_cost_source(payload: dict) -> str | None:
    if _token_cost_from_payload(payload) is not None:
        return "output_tokens"
    if _safe_float(payload.get("cost")) is not None:
        return "explicit_cost"
    if _safe_float(payload.get("total_tokens")) is not None:
        return "tokens"
    if _safe_float(payload.get("prompt_tokens")) is not None or _safe_float(payload.get("output_tokens")) is not None:
        return "tokens"
    return None


def _candidate_cost_map(subproblem: dict) -> Dict[int, float]:
    mapping: Dict[int, float] = {}
    for idx, candidate in enumerate(subproblem.get("rollout_candidates") or []):
        if not isinstance(candidate, dict):
            continue
        cost = _token_cost_from_payload(candidate)
        if cost is None:
            continue
        rollout_id = candidate.get("rollout_id", idx)
        mapping[int(rollout_id)] = cost
    return mapping


def _candidate_batch_latency_map(subproblem: dict) -> Dict[int, float]:
    mapping: Dict[int, float] = {}
    for idx, candidate in enumerate(subproblem.get("rollout_candidates") or []):
        if not isinstance(candidate, dict):
            continue
        latency = _safe_float(candidate.get("infer_batch_latency_sec"))
        if latency is None:
            continue
        rollout_id = candidate.get("rollout_id", idx)
        mapping[int(rollout_id)] = max(latency, 1e-9)
    return mapping


def _rollout_execution_latency(
    record: dict,
    candidate_batch_latencies: Dict[int, float] | None = None,
    rollout_id: int | None = None,
) -> float | None:
    batch_latency = _safe_float(record.get("infer_batch_latency_sec"))
    if batch_latency is None and candidate_batch_latencies is not None:
        batch_latency = candidate_batch_latencies.get(rollout_id)
    if batch_latency is not None:
        return max(batch_latency, 1e-9)

    infer_latency = _safe_float(record.get("infer_latency_sec"))
    if infer_latency is not None:
        return max(infer_latency, 1e-9)

    total_latency = _safe_float(record.get("total_latency_sec"))
    if total_latency is not None:
        return max(total_latency, 1e-9)

    return None


def _build_node_template(subproblem: dict) -> Tuple[dict, List[dict]]:
    rollout_records = subproblem.get("rollout_records") or []
    if rollout_records:
        empirical_samples: List[dict] = []
        successes: List[int] = []
        latencies: List[float] = []
        costs: List[float] = []
        candidate_costs = _candidate_cost_map(subproblem)
        candidate_batch_latencies = _candidate_batch_latency_map(subproblem)
        explicit_cost_present = any(
            _work_cost_source(r) == "explicit_cost" for r in rollout_records if isinstance(r, dict)
        ) or any(
            _work_cost_source(c) == "explicit_cost" for c in (subproblem.get("rollout_candidates") or []) if isinstance(c, dict)
        )
        token_cost_present = any(
            _work_cost_source(r) == "output_tokens" for r in rollout_records if isinstance(r, dict)
        ) or any(
            _work_cost_source(c) == "output_tokens" for c in (subproblem.get("rollout_candidates") or []) if isinstance(c, dict)
        )

        for r in rollout_records:
            if not isinstance(r, dict):
                continue
            success = int(r.get("success", 0))
            total_latency = _safe_float(r.get("total_latency_sec"))
            rollout_id = int(r.get("rollout_id", len(empirical_samples)))
            latency = _rollout_execution_latency(r, candidate_batch_latencies, rollout_id)
            if latency is None:
                latency = float(total_latency or 0.0)
            infer_batch_latency = _safe_float(r.get("infer_batch_latency_sec"))
            if infer_batch_latency is None:
                infer_batch_latency = candidate_batch_latencies.get(rollout_id)
            token_cost = _token_cost_from_payload(r)
            if token_cost is None:
                token_cost = candidate_costs.get(rollout_id)
            cost = token_cost
            if cost is None:
                cost = _work_cost_from_payload(r)
            if cost is None:
                cost = float(r.get("total_latency_sec", 0.0) or 0.0)
            successes.append(success)
            latencies.append(latency)
            costs.append(cost)
            empirical_samples.append({
                "rollout_id": rollout_id,
                "infer_latency_sec": _safe_float(r.get("infer_latency_sec")),
                "infer_batch_latency_sec": infer_batch_latency,
                "infer_per_rollout_latency_sec": _safe_float(r.get("infer_per_rollout_latency_sec")),
                "latency_observation": r.get("latency_observation"),
                "eval_latency_sec": _safe_float(r.get("eval_latency_sec")),
                "total_latency_sec": total_latency,
                "latency": latency,
                "success": success,
                "token_cost": token_cost,
                "cost": cost,
                "pass_rate": _safe_float(r.get("pass_rate")),
            })

        if not empirical_samples:
            empirical_samples = [{"latency": 1.0, "success": 0, "cost": 1.0}]
            successes = [0]
            latencies = [1.0]

        success_prob = sum(successes) / float(len(successes))
        lat_mean = sum(latencies) / float(len(latencies))
        variance = sum((x - lat_mean) ** 2 for x in latencies) / float(len(latencies))
        lat_std = max(math.sqrt(variance), 1e-3)
        cost_mean = sum(costs) / float(len(costs))
        token_costs = [
            sample["token_cost"]
            for sample in empirical_samples
            if sample.get("token_cost") is not None
        ]

        template = {
            "success_prob": float(success_prob),
            "lat_mean": float(lat_mean),
            "lat_std": float(lat_std),
            "cost_mean": float(cost_mean),
            "token_cost_mean": (
                float(sum(token_costs) / len(token_costs))
                if token_costs
                else None
            ),
            "cost_source": (
                "output_tokens"
                if token_cost_present
                else ("explicit_cost" if explicit_cost_present else "latency_fallback")
            ),
        }
        return template, empirical_samples

    results = subproblem.get("harness_result") or []
    test_cases = subproblem.get("test_code") or []

    valid = [int(x) for x in results if isinstance(x, (int, float)) and int(x) in (0, 1)]
    if valid:
        success_prob = sum(valid) / float(len(valid))
    elif test_cases:
        success_prob = 0.0
    else:
        success_prob = 0.5

    # Use test-case count as a stable latency/cost proxy when raw timing is unavailable.
    complexity_proxy = max(len(test_cases), len(valid), 1)
    lat_mean = float(complexity_proxy)
    lat_std = max(0.2 * lat_mean, 1e-3)
    cost_mean = float(complexity_proxy)

    template = {
        "success_prob": float(success_prob),
        "lat_mean": lat_mean,
        "lat_std": lat_std,
        "cost_mean": cost_mean,
    }

    empirical_samples: List[dict] = []
    if valid:
        for v in valid:
            empirical_samples.append({
                "latency": lat_mean,
                "success": int(v),
                "cost": cost_mean,
            })
    else:
        empirical_samples.append({
            "latency": lat_mean,
            "success": int(success_prob >= 0.5),
            "cost": cost_mean,
        })

    return template, empirical_samples


def convert_problem(problem: dict) -> dict:
    subproblems = problem.get("subproblems") or []
    name_to_idx: Dict[str, int] = {}
    for i, sp in enumerate(subproblems):
        name_to_idx[sp.get("name", f"node_{i}")] = i

    max_depth = 0
    for sp in subproblems:
        depth = sp.get("depth")
        if isinstance(depth, int):
            max_depth = max(max_depth, depth)

    parents_map: Dict[int, List[int]] = {i: [] for i in range(len(subproblems))}
    children_map: Dict[int, List[int]] = {i: [] for i in range(len(subproblems))}

    for i, sp in enumerate(subproblems):
        deps = sp.get("dependencies") or []
        for dep in deps:
            if dep in name_to_idx:
                p = name_to_idx[dep]
                parents_map[i].append(p)
                children_map[p].append(i)

    nodes = []
    for i, sp in enumerate(subproblems):
        template, empirical_samples = _build_node_template(sp)
        depth = sp.get("depth")
        if isinstance(depth, int):
            layer = max_depth - depth
        else:
            layer = i

        nodes.append({
            "node_id": i,
            "parents": sorted(set(parents_map[i])),
            "children": sorted(set(children_map[i])),
            "layer": int(layer),
            "template": template,
            "empirical_samples": empirical_samples,
            "meta": {
                "name": sp.get("name", f"node_{i}"),
                "status": _compute_status(sp),
                "harness_result": sp.get("harness_result", []),
                "n_rollouts": len(sp.get("rollout_records") or []),
                "num_tests": len(sp.get("test_code") or []),
            },
        })

    return {
        "dag_kind": "codeflow_comp",
        "source": "codeflowbench_comp_test",
        "problem_id": problem.get("problem-id"),
        "codeflow_source_index": problem.get("_source_index"),
        "n_nodes": len(nodes),
        "overall_turns": problem.get("overall-turns"),
        "overall_depth": problem.get("overall-depth"),
        "nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CodeFlowBench-Comp harness output into simulator DAG pool JSONL format"
    )
    parser.add_argument(
        "--input_harness",
        type=str,
        required=True,
        help="Path to CodeFlow harness output JSON (e.g. codeflow/output/harness/<model>_multi_turn.json)",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="pregenerated_dags/dag_pool.jsonl",
        help="Output DAG pool JSONL path used by simulator",
    )
    parser.add_argument(
        "--output_indices",
        type=str,
        default="",
        help="Optional selected_dag_indices.json path. Defaults next to output_jsonl.",
    )
    args = parser.parse_args()

    with open(args.input_harness, "r", encoding="utf-8") as f:
        problems = json.load(f)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    converted = [convert_problem(p) for p in problems]

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for dag in converted:
            f.write(json.dumps(dag, ensure_ascii=False) + "\n")

    output_indices = args.output_indices or os.path.join(
        os.path.dirname(args.output_jsonl),
        "selected_dag_indices.json",
    )
    with open(output_indices, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dag_indices": list(range(len(converted))),
                "source": args.output_jsonl,
                "index_semantics": "row index in the converted CodeFlow DAG pool",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    n_nodes_total = sum(d["n_nodes"] for d in converted)
    print(f"Converted DAGs: {len(converted)}")
    print(f"Total nodes: {n_nodes_total}")
    print(f"Saved JSONL: {args.output_jsonl}")
    print(f"Saved indices: {output_indices}")


if __name__ == "__main__":
    main()
