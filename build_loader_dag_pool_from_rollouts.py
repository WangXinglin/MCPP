import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a simulator DAG pool from evaluated CodeFlow multi-turn rollout JSONs. "
            "By default, only keep DAGs whose every node has rollout success rate > 0, "
            "and convert each node's rollout_records into empirical_samples for resampling."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an evaluated problem JSON file or a directory containing evaluated JSON files.",
    )
    parser.add_argument(
        "--output-dag-pool",
        type=str,
        required=True,
        help="Path to write output dag_pool.jsonl.",
    )
    parser.add_argument(
        "--output-indices",
        type=str,
        default="",
        help="Optional path to write selected_dag_indices.json. Defaults next to dag_pool.jsonl.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="",
        help="Optional path to write conversion summary JSON. Defaults next to dag_pool.jsonl.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search JSON files when --input is a directory.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Fallback pass_rate threshold when rollout_records have no explicit success field.",
    )
    parser.add_argument(
        "--allow-zero-success-node",
        action="store_true",
        help="Keep DAGs even if some nodes have rollout success rate == 0.",
    )
    parser.add_argument(
        "--allow-missing-node-records",
        action="store_true",
        help="Keep DAGs even if some nodes have no rollout_records.",
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


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def rollout_cost(record):
    token_cost = rollout_token_cost(record)
    if token_cost is not None:
        return token_cost

    explicit_cost = safe_float(record.get("cost"))
    if explicit_cost is not None:
        return max(explicit_cost, 1e-9)

    total_tokens = safe_float(record.get("total_tokens"))
    if total_tokens is not None:
        return max(total_tokens, 1e-9)

    prompt_tokens = safe_float(record.get("prompt_tokens"))
    output_tokens = safe_float(record.get("output_tokens"))
    if prompt_tokens is not None or output_tokens is not None:
        return max((prompt_tokens or 0.0) + (output_tokens or 0.0), 1e-9)

    return None


def rollout_token_cost(record):
    output_tokens = safe_float(record.get("output_tokens"))
    if output_tokens is not None:
        return max(output_tokens, 1e-9)
    return None


def rollout_cost_source(record):
    if rollout_token_cost(record) is not None:
        return "output_tokens"
    if safe_float(record.get("cost")) is not None:
        return "explicit_cost"
    if safe_float(record.get("total_tokens")) is not None:
        return "tokens"
    if safe_float(record.get("prompt_tokens")) is not None or safe_float(record.get("output_tokens")) is not None:
        return "tokens"
    return None


def candidate_token_cost_map(subproblem):
    mapping = {}
    for idx, candidate in enumerate(subproblem.get("rollout_candidates") or []):
        if not isinstance(candidate, dict):
            continue
        rollout_id = candidate.get("rollout_id", idx)
        cost = rollout_token_cost(candidate)
        if cost is None:
            continue
        mapping[rollout_id] = cost
    return mapping


def candidate_batch_latency_map(subproblem):
    mapping = {}
    for idx, candidate in enumerate(subproblem.get("rollout_candidates") or []):
        if not isinstance(candidate, dict):
            continue
        rollout_id = candidate.get("rollout_id", idx)
        latency = safe_float(candidate.get("infer_batch_latency_sec"))
        if latency is None:
            continue
        mapping[rollout_id] = max(latency, 1e-9)
    return mapping


def rollout_execution_latency(record, candidate_batch_latencies=None, rollout_id=None):
    batch_latency = safe_float(record.get("infer_batch_latency_sec"))
    if batch_latency is None and candidate_batch_latencies is not None:
        batch_latency = candidate_batch_latencies.get(rollout_id)
    if batch_latency is not None:
        return max(batch_latency, 1e-9)

    infer_latency = safe_float(record.get("infer_latency_sec"))
    if infer_latency is not None:
        return max(infer_latency, 1e-9)

    total_latency = safe_float(record.get("total_latency_sec"))
    if total_latency is not None:
        return max(total_latency, 1e-9)

    return None


def token_scaled_batch_latency(batch_latency, token_cost, max_token_cost):
    if batch_latency is None or token_cost is None or max_token_cost is None or max_token_cost <= 0.0:
        return None
    ratio = min(max(float(token_cost) / float(max_token_cost), 0.0), 1.0)
    return max(float(batch_latency) * ratio, 1e-9)


def normalize_rollout_records(subproblem, success_threshold):
    records = subproblem.get("rollout_records") or []
    token_costs = candidate_token_cost_map(subproblem)
    batch_latencies = candidate_batch_latency_map(subproblem)
    effective_token_costs = []
    for idx, record in enumerate(records):
        rollout_id = record.get("rollout_id", idx)
        token_cost = rollout_token_cost(record)
        if token_cost is None:
            token_cost = token_costs.get(rollout_id)
        if token_cost is not None:
            effective_token_costs.append(token_cost)
    max_token_cost = max(effective_token_costs) if effective_token_costs else None

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

        success = bool_success(record, success_threshold)
        rollout_id = record.get("rollout_id", idx)
        if infer_batch_latency is None:
            infer_batch_latency = batch_latencies.get(rollout_id)
        token_cost = rollout_token_cost(record)
        if token_cost is None:
            token_cost = token_costs.get(rollout_id)
        execution_latency = token_scaled_batch_latency(
            infer_batch_latency,
            token_cost,
            max_token_cost,
        )
        if execution_latency is None:
            execution_latency = rollout_execution_latency(record, batch_latencies, rollout_id)
        cost = token_cost
        if cost is None:
            cost = rollout_cost(record)
        if cost is None and total_latency is not None:
            # Backward-compatible fallback for older artifacts with no token fields.
            cost = max(total_latency, 1e-9)
        normalized.append(
            {
                "rollout_id": rollout_id,
                "infer_latency_sec": infer_latency,
                "infer_batch_latency_sec": infer_batch_latency,
                "infer_per_rollout_latency_sec": infer_per_rollout_latency,
                "latency_observation": record.get("latency_observation"),
                "eval_latency_sec": eval_latency,
                "total_latency_sec": total_latency,
                "latency": execution_latency,
                "token_cost": token_cost,
                "cost": cost,
                "pass_rate": pass_rate if pass_rate is not None else 0.0,
                "success": int(bool(success)) if success is not None else 0,
            }
        )
    return normalized


def compute_children(parents_by_idx):
    children_by_idx = {idx: [] for idx in parents_by_idx}
    for idx, parents in parents_by_idx.items():
        for parent in parents:
            children_by_idx[parent].append(idx)
    for idx in children_by_idx:
        children_by_idx[idx] = sorted(set(children_by_idx[idx]))
    return children_by_idx


def compute_depths(parents_by_idx):
    memo = {}
    visiting = set()

    def dfs(idx):
        if idx in memo:
            return memo[idx]
        if idx in visiting:
            raise ValueError("Cycle detected while computing depths.")
        visiting.add(idx)
        parents = parents_by_idx[idx]
        if not parents:
            depth = 0
        else:
            depth = max(dfs(parent) for parent in parents) + 1
        visiting.remove(idx)
        memo[idx] = depth
        return depth

    for idx in parents_by_idx:
        dfs(idx)
    return memo


def build_dag_entry(problem, source_path, success_threshold, allow_zero_success_node, require_all_nodes_evaluated):
    problem_id = problem.get("problem-id") or problem.get("task_id") or "unknown_problem"
    subproblems = problem.get("subproblems", [])
    if not subproblems:
        return None, {"problem_id": problem_id, "reason": "no_subproblems", "source_path": source_path}

    names = [subproblem.get("name") or f"node_{idx}" for idx, subproblem in enumerate(subproblems)]
    if len(set(names)) != len(names):
        return None, {"problem_id": problem_id, "reason": "duplicate_subproblem_names", "source_path": source_path}
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    parents_by_idx = {}
    normalized_by_idx = {}
    node_success_rates = []

    for idx, subproblem in enumerate(subproblems):
        dependency_names = subproblem.get("dependencies") or []
        parents = []
        for dep_name in dependency_names:
            if dep_name not in name_to_idx:
                return None, {
                    "problem_id": problem_id,
                    "reason": f"missing_dependency:{dep_name}",
                    "source_path": source_path,
                }
            parents.append(name_to_idx[dep_name])
        parents_by_idx[idx] = sorted(set(parents))

        normalized_records = normalize_rollout_records(subproblem, success_threshold)
        normalized_by_idx[idx] = normalized_records

        if require_all_nodes_evaluated and not normalized_records:
            return None, {
                "problem_id": problem_id,
                "reason": f"missing_rollout_records:{subproblem.get('name', idx)}",
                "source_path": source_path,
            }

        if normalized_records:
            successful_rollouts = sum(record["success"] for record in normalized_records)
            node_success_rate = successful_rollouts / float(len(normalized_records))
            node_success_rates.append(node_success_rate)
            if (not allow_zero_success_node) and node_success_rate <= 0.0:
                return None, {
                    "problem_id": problem_id,
                    "reason": f"zero_success_node:{subproblem.get('name', idx)}",
                    "source_path": source_path,
                }

    depths = compute_depths(parents_by_idx)
    children_by_idx = compute_children(parents_by_idx)

    nodes = []
    for idx, subproblem in enumerate(subproblems):
        normalized_records = normalized_by_idx[idx]
        explicit_cost_present = any(
            rollout_cost_source(record) == "explicit_cost" for record in (subproblem.get("rollout_records") or [])
        ) or any(
            rollout_cost_source(candidate) == "explicit_cost" for candidate in (subproblem.get("rollout_candidates") or [])
        )
        token_cost_present = any(
            rollout_cost_source(record) == "output_tokens" for record in (subproblem.get("rollout_records") or [])
        ) or any(
            rollout_cost_source(candidate) == "output_tokens" for candidate in (subproblem.get("rollout_candidates") or [])
        )
        observed_latencies = [
            record["latency"]
            for record in normalized_records
            if record["latency"] is not None
        ]
        if not observed_latencies:
            return None, {
                "problem_id": problem_id,
                "reason": f"missing_infer_latency:{subproblem.get('name', idx)}",
                "source_path": source_path,
            }

        lat_mean = sum(observed_latencies) / float(len(observed_latencies))
        variance = sum((value - lat_mean) ** 2 for value in observed_latencies) / float(len(observed_latencies))
        lat_std = max(math.sqrt(max(variance, 0.0)), 1e-6)
        success_prob = sum(record["success"] for record in normalized_records) / float(len(normalized_records))
        empirical_samples = [
            {
                "rollout_id": record["rollout_id"],
                "infer_latency_sec": record["infer_latency_sec"],
                "infer_batch_latency_sec": record["infer_batch_latency_sec"],
                "infer_per_rollout_latency_sec": record["infer_per_rollout_latency_sec"],
                "latency_observation": record.get("latency_observation"),
                "eval_latency_sec": record["eval_latency_sec"],
                "total_latency_sec": record["total_latency_sec"],
                "latency": record["latency"],
                "success": int(record["success"]),
                "token_cost": record.get("token_cost"),
                "cost": record["cost"] if record.get("cost") is not None else record["latency"],
                "pass_rate": record["pass_rate"],
            }
            for record in normalized_records
            if record["latency"] is not None
        ]
        observed_costs = [
            record["cost"]
            for record in normalized_records
            if record.get("cost") is not None
        ]
        if not observed_costs:
            observed_costs = observed_latencies
        observed_token_costs = [
            record["token_cost"]
            for record in normalized_records
            if record.get("token_cost") is not None
        ]

        nodes.append(
            {
                "node_id": idx,
                "parents": parents_by_idx[idx],
                "children": children_by_idx[idx],
                "layer": int(subproblem.get("depth")) if isinstance(subproblem.get("depth"), int) else depths[idx],
                "template": {
                    "success_prob": success_prob,
                    "lat_mean": lat_mean,
                    "lat_std": lat_std,
                    "cost_mean": sum(observed_costs) / float(len(observed_costs)),
                    "token_cost_mean": (
                        sum(observed_token_costs) / float(len(observed_token_costs))
                        if observed_token_costs
                        else None
                    ),
                },
                "empirical_samples": empirical_samples,
                "name": subproblem.get("name"),
                "turn_index": idx + 1,
                "selected_rollout_id": subproblem.get("selected_rollout_id"),
                "rollout_success_rate": success_prob,
                "cost_source": (
                    "output_tokens"
                    if token_cost_present
                    else ("explicit_cost" if explicit_cost_present else "latency_fallback")
                ),
            }
        )

    dag_entry = {
        "dag_kind": "codeflow_rollout",
        "n_nodes": len(nodes),
        "nodes": nodes,
        "problem_id": problem_id,
        "source_path": source_path,
        "overall_turns": problem.get("overall-turns"),
        "overall_depth": problem.get("overall-depth"),
    }
    return dag_entry, {
        "problem_id": problem_id,
        "source_path": source_path,
        "node_count": len(nodes),
        "mean_node_success_rate": sum(node_success_rates) / float(len(node_success_rates)) if node_success_rates else None,
    }


def main():
    args = parse_args()
    json_files = collect_json_files(args.input, args.recursive)

    problems_total = 0
    dag_pool = []
    files_skipped = []
    included = []
    excluded = []

    for json_file in json_files:
        problems, err = load_problems_from_json(json_file)
        if err is not None:
            files_skipped.append(err)
            continue

        for problem in problems:
            problems_total += 1
            try:
                dag_entry, meta = build_dag_entry(
                    problem=problem,
                    source_path=str(json_file),
                    success_threshold=args.success_threshold,
                    allow_zero_success_node=args.allow_zero_success_node,
                    require_all_nodes_evaluated=(not args.allow_missing_node_records),
                )
            except Exception as exc:
                dag_entry = None
                meta = {
                    "problem_id": problem.get("problem-id") or problem.get("task_id") or "unknown_problem",
                    "reason": f"build_failed:{exc}",
                    "source_path": str(json_file),
                }
            if dag_entry is None:
                excluded.append(meta)
                continue
            dag_pool.append(dag_entry)
            included.append(meta)

    output_dag_pool = Path(args.output_dag_pool)
    output_dag_pool.parent.mkdir(parents=True, exist_ok=True)
    with output_dag_pool.open("w", encoding="utf-8") as f:
        for entry in dag_pool:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    output_indices = Path(args.output_indices) if args.output_indices else output_dag_pool.parent / "selected_dag_indices.json"
    output_indices.parent.mkdir(parents=True, exist_ok=True)
    output_indices.write_text(
        json.dumps({"dag_indices": list(range(len(dag_pool)))}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    output_summary = Path(args.output_summary) if args.output_summary else output_dag_pool.parent / "build_loader_dag_pool_summary.json"
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_path": str(Path(args.input).resolve()),
        "files_scanned": len(json_files),
        "files_skipped": files_skipped,
        "problems_total": problems_total,
        "dags_written": len(dag_pool),
        "dags_excluded": len(excluded),
        "filter": {
            "allow_zero_success_node": args.allow_zero_success_node,
            "require_all_nodes_evaluated": (not args.allow_missing_node_records),
            "success_threshold": args.success_threshold,
        },
        "outputs": {
            "dag_pool_jsonl": str(output_dag_pool.resolve()),
            "selected_dag_indices_json": str(output_indices.resolve()),
        },
        "included_preview": included[:20],
        "excluded_preview": excluded[:50],
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
