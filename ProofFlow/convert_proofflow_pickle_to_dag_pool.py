#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "matplotlib"))

from _bootstrap_proofflow import bootstrap_canonical_proofflow

bootstrap_canonical_proofflow()

from proofflow.io import RenamedUnpickler


DEFAULT_SEMANTIC_SCORE_THRESHOLD = 0.0


def _to_dict(item):
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    return {}


def _safe_int(v, d=0):
    try:
        return int(v)
    except Exception:
        return d


def _safe_float(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def _semantic_score(sp: dict):
    score = sp.get("score") or {}
    if not isinstance(score, dict):
        return None
    raw = score.get("semantic_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _semantic_pass(sp: dict, threshold: float) -> bool:
    if float(threshold) <= 0.0:
        return True
    score = _semantic_score(sp)
    return score is not None and score > float(threshold)


def _rollout_latency(record: dict) -> float:
    total = _safe_float(record.get("total_latency_sec"), None)
    if total is not None:
        return max(total, 1e-9)
    latency = _safe_float(record.get("latency_sec"), None)
    if latency is not None:
        return max(latency, 1e-9)
    infer = _safe_float(record.get("infer_latency_sec"), 0.0)
    verify = _safe_float(record.get("verify_latency_sec"), 0.0)
    return max(infer + verify, 1e-9)


def _usage_per_call(payload: dict) -> dict:
    summary = payload.get("usage_summary") or {}
    n_calls = max(1, _safe_int(summary.get("n_calls"), 0))
    input_tokens = _safe_float(summary.get("input_tokens"), 0.0) / float(n_calls)
    output_tokens = _safe_float(summary.get("output_tokens"), 0.0) / float(n_calls)
    total_tokens = _safe_float(summary.get("total_tokens"), 0.0) / float(n_calls)
    api_cost_usd = _safe_float(summary.get("cost_usd"), 0.0) / float(n_calls)
    if total_tokens <= 0.0:
        return {}
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "token_cost": total_tokens,
        "api_cost_usd": api_cost_usd,
    }


def _rollout_samples(payload: dict, success_key: str, semantic_ok: bool) -> List[dict]:
    samples = []
    usage = _usage_per_call(payload)
    for idx, record in enumerate(payload.get("rollout_records") or []):
        if not isinstance(record, dict):
            continue
        latency = _rollout_latency(record)
        success = 1 if semantic_ok and bool(record.get(success_key, False)) else 0
        samples.append({
            "rollout_id": record.get("rollout_id", idx),
            "infer_latency_sec": _safe_float(record.get("infer_latency_sec"), 0.0),
            "verify_latency_sec": _safe_float(record.get("verify_latency_sec"), 0.0),
            "total_latency_sec": latency,
            "latency": latency,
            "success": success,
            "cost": latency,
            "lean_pass": bool(record.get("lean_pass", False)),
            success_key: bool(record.get(success_key, False)),
            "semantic_pass": bool(semantic_ok),
        })
        samples[-1].update(usage)
    return samples


def _node_stats(
    sp: dict,
    semantic_score_threshold: float = DEFAULT_SEMANTIC_SCORE_THRESHOLD,
) -> Tuple[dict, List[dict], str]:
    solved = sp.get("solved_lemma") or {}
    form = sp.get("formalization") or {}
    node_id = str(sp.get("id", ""))
    is_terminal = node_id.startswith(("tc_", "def_"))
    semantic_ok = _semantic_pass(sp, semantic_score_threshold)

    empirical_samples = _rollout_samples(solved, "lean_verify", semantic_ok)
    if not empirical_samples:
        empirical_samples = _rollout_samples(form, "lean_pass", semantic_ok)

    if is_terminal:
        success = 1 if semantic_ok and bool(form.get("lean_pass", False)) else 0
    else:
        success = 1 if semantic_ok and bool(solved.get("lean_verify", False)) else 0
    if empirical_samples:
        latencies = [sample["latency"] for sample in empirical_samples]
        lat = sum(latencies) / float(len(latencies))
        success_prob = sum(sample["success"] for sample in empirical_samples) / float(len(empirical_samples))
        token_costs = [
            _safe_float(sample.get("token_cost", sample.get("total_tokens")), 0.0)
            for sample in empirical_samples
        ]
        token_costs = [value for value in token_costs if value > 0.0]
        token_cost_mean = (
            sum(token_costs) / float(len(token_costs))
            if token_costs
            else None
        )
    else:
        form_tries = max(1, _safe_int(form.get("tries", 1), 1))
        solve_tries = max(1, _safe_int(solved.get("tries", 1), 1))
        lat = float(form_tries + solve_tries)
        success_prob = float(success)
        empirical_samples = [{"latency": lat, "success": success, "cost": lat}]
        token_cost_mean = None

    template = {
        "success_prob": success_prob,
        "lat_mean": lat,
        "lat_std": max(0.2 * lat, 1e-3),
        "cost_mean": lat,
    }
    if token_cost_mean is not None:
        template["token_cost_mean"] = token_cost_mean
    status = "success" if success == 1 else "failed"
    return template, empirical_samples, status


def convert_pickle_file(
    fp: Path,
    semantic_score_threshold: float = DEFAULT_SEMANTIC_SCORE_THRESHOLD,
) -> dict:
    with fp.open("rb") as f:
        try:
            data = RenamedUnpickler(f).load()
        except Exception:
            f.seek(0)
            data = pickle.load(f)

    proof_items = data.get("proof_items") or []
    items = [_to_dict(x) for x in proof_items]

    id_to_idx: Dict[str, int] = {}
    for i, it in enumerate(items):
        id_to_idx[str(it.get("id", f"n{i}"))] = i

    parents_map: Dict[int, List[int]] = {i: [] for i in range(len(items))}
    children_map: Dict[int, List[int]] = {i: [] for i in range(len(items))}

    for i, it in enumerate(items):
        for dep in (it.get("dependencies") or []):
            dep = str(dep)
            if dep in id_to_idx:
                p = id_to_idx[dep]
                parents_map[i].append(p)
                children_map[p].append(i)

    nodes = []
    for i, it in enumerate(items):
        template, empirical_samples, status = _node_stats(
            it,
            semantic_score_threshold=semantic_score_threshold,
        )
        nodes.append({
            "node_id": i,
            "parents": sorted(set(parents_map[i])),
            "children": sorted(set(children_map[i])),
            "layer": i,
            "template": template,
            "empirical_samples": empirical_samples,
            "meta": {
                "id": it.get("id", f"n{i}"),
                "status": status,
                "dependencies": it.get("dependencies", []),
                "semantic_score": _semantic_score(it),
                "semantic_score_threshold": float(semantic_score_threshold),
                "semantic_pass": _semantic_pass(it, semantic_score_threshold),
                "formalization_tries": _safe_int((it.get("formalization") or {}).get("tries", 1), 1),
                "solver_tries": _safe_int((it.get("solved_lemma") or {}).get("tries", 1), 1),
            },
        })

    return {
        "dag_kind": "proofflow_benchmark",
        "source": "benchmark_0409",
        "problem_id": fp.stem,
        "n_nodes": len(nodes),
        "overall_turns": len(nodes),
        "overall_depth": len(nodes),
        "nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ProofFlow pickle outputs into DAG pool JSONL")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .pickle files")
    parser.add_argument("--output_jsonl", type=str, default="../pregenerated_dags/proofflow_dag_pool.jsonl")
    parser.add_argument(
        "--semantic_score_threshold",
        type=float,
        default=float(os.getenv("SEMANTIC_SCORE_THRESHOLD", str(DEFAULT_SEMANTIC_SCORE_THRESHOLD))),
        help=(
            "Minimum item.score.semantic_score for a node rollout to count as successful. "
            "<=0 disables semantic gating and matches the official ProofFlow execution semantics."
        ),
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted([p for p in in_dir.iterdir() if p.suffix in (".pickle", ".pkl")])
    converted = []
    for fp in files:
        try:
            converted.append(
                convert_pickle_file(
                    fp,
                    semantic_score_threshold=args.semantic_score_threshold,
                )
            )
        except Exception as e:
            print(f"WARN skip {fp.name}: {e}")

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for dag in converted:
            f.write(json.dumps(dag, ensure_ascii=False) + "\n")

    total_nodes = sum(d["n_nodes"] for d in converted)
    print(f"Converted DAGs: {len(converted)}")
    print(f"Total nodes: {total_nodes}")
    print(f"Saved JSONL: {out}")


if __name__ == "__main__":
    main()
