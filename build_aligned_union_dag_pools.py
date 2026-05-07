#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from build_loader_dag_pool_from_rollouts import (
    build_dag_entry,
    compute_children,
    compute_depths,
    normalize_rollout_records,
    rollout_cost_source,
)
from split_codeflow_inference_json import iter_top_level_array_items


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build model-specific, row-aligned CodeFlow DAG pools over the union "
            "of usable DAGs from multiple evaluated model runs. The union can be "
            "taken from already-built full dag_pool.jsonl files with strict "
            "per-DAG eligibility checked inside this script, then the aligned "
            "model pools are rebuilt from harness/evaluated rollout sources or "
            "reused from per-model dag_pool files while retaining partially "
            "available nodes. "
            "Each output dag_pool has the same problem_id order and therefore the "
            "same dag_idx semantics. Zero-success nodes are retained in aligned "
            "model pools. Missing model/problem entries can either error, be "
            "dropped, or be imputed as always-failing infinite-cost rollouts. "
            "A combined model-aware dag_pool is also written for the simulator."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Model name and source path. Without --dag-pool for this NAME, PATH "
            "must be an evaluated CodeFlow harness JSON array, a single problem "
            "JSON, or a directory of JSON files. With --dag-pool for this NAME, "
            "PATH is metadata and may be the same full dag_pool.jsonl path. Repeat "
            "once per model."
        ),
    )
    parser.add_argument(
        "--dag-pool",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Existing per-model dag_pool.jsonl, normally the full pool. When "
            "provided for a model, union membership is strict-filtered from this "
            "pool, while aligned entries are read from the same full pool so "
            "partially available model/node statistics are preserved. Repeat once "
            "per reused model."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where aligned per-model dag_pool.jsonl files are written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan JSON files when a model PATH is a directory.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Fallback pass_rate threshold when rollout_records have no explicit success field.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["error", "drop", "impute"],
        default="impute",
        help=(
            "What to do when a union problem cannot be built for some model because "
            "records/latency/dependencies are missing. error writes a report and exits; "
            "drop removes those problem_ids from the final aligned set; impute keeps "
            "the problem_id and fills the missing model entry with success_rate=0 and "
            "infinite latency/cost/token_cost samples."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N scanned problems per pass. 0 disables progress.",
    )
    parser.add_argument(
        "--combined-dag-pool",
        default="",
        help=(
            "Path to write the combined model-aware dag_pool.jsonl consumed by "
            "run_shard.py. Defaults to <output-root>/multi_model_dag_pool.jsonl."
        ),
    )
    parser.add_argument(
        "--combined-indices",
        default="",
        help=(
            "Path to write indices for the combined model-aware dag_pool. Defaults "
            "to <output-root>/multi_model_selected_dag_indices.json."
        ),
    )
    parser.add_argument(
        "--combined-summary",
        default="",
        help=(
            "Path to write summary metadata for the combined model-aware dag_pool. "
            "Defaults to <output-root>/multi_model_build_summary.json."
        ),
    )
    parser.add_argument(
        "--baseline-model",
        default="",
        help=(
            "Model whose template/empirical_samples are exposed through the legacy "
            "single-model fields used by baseline policies. Defaults to model named "
            "'default' if present, otherwise the first --model."
        ),
    )
    return parser.parse_args()


INF_VALUE = float("inf")


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("._") or "model"


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --model {spec!r}; expected NAME=PATH")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name:
        raise ValueError(f"Invalid --model {spec!r}; model name is empty")
    if not raw_path:
        raise ValueError(f"Invalid --model {spec!r}; path is empty")
    return name, Path(raw_path)


def parse_named_path_specs(specs: list[str], label: str) -> dict[str, Path]:
    out = {}
    for spec in specs:
        name, path = parse_model_spec(spec)
        if name in out:
            raise ValueError(f"Duplicate {label} name: {name}")
        out[name] = path
    return out


def first_nonspace_char(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                return ""
            for ch in chunk:
                if not ch.isspace():
                    return ch


def iter_problem_json_files(path: Path, recursive: bool):
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    pattern = "**/*.json" if recursive else "*.json"
    for item in sorted(path.glob(pattern)):
        if item.is_file():
            yield item


def problem_id(problem: dict) -> str:
    return str(problem.get("problem-id") or problem.get("task_id") or "unknown_problem")


def iter_problems_from_file(path: Path):
    marker = first_nonspace_char(path)
    if marker == "[":
        for source_idx, item_text in iter_top_level_array_items(str(path)):
            try:
                item = json.loads(item_text)
            except Exception as exc:
                yield None, {
                    "path": str(path),
                    "source_index": source_idx,
                    "error": f"json_item_load_failed:{exc}",
                }
                continue
            if isinstance(item, dict) and "subproblems" in item:
                yield item, {"path": str(path), "source_index": source_idx}
            else:
                yield None, {
                    "path": str(path),
                    "source_index": source_idx,
                    "error": "json_item_is_not_problem_payload",
                }
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        yield None, {"path": str(path), "error": f"json_load_failed:{exc}"}
        return

    if isinstance(data, dict) and "subproblems" in data:
        yield data, {"path": str(path), "source_index": 0}
        return

    if isinstance(data, list):
        for source_idx, item in enumerate(data):
            if isinstance(item, dict) and "subproblems" in item:
                yield item, {"path": str(path), "source_index": source_idx}
            else:
                yield None, {
                    "path": str(path),
                    "source_index": source_idx,
                    "error": "json_item_is_not_problem_payload",
                }
        return

    yield None, {"path": str(path), "error": "unsupported_json_top_level"}


def iter_model_problems(source_path: Path, recursive: bool):
    for json_file in iter_problem_json_files(source_path, recursive):
        yield from iter_problems_from_file(json_file)


def build_for_problem(
    problem: dict,
    source_meta: dict,
    *,
    success_threshold: float,
    allow_zero_success_node: bool,
):
    source_path = source_meta["path"]
    return build_dag_entry(
        problem=problem,
        source_path=source_path,
        success_threshold=success_threshold,
        allow_zero_success_node=allow_zero_success_node,
        require_all_nodes_evaluated=True,
    )


def scan_strict_eligible(model: dict, args) -> dict:
    eligible_ids = []
    eligible_set = set()
    reference_entries_by_id = {}
    excluded = []
    files_skipped = []
    duplicate_problem_ids = []
    seen = set()
    scanned = 0
    started_at = time.time()

    for problem, meta in iter_model_problems(model["path"], args.recursive):
        if problem is None:
            files_skipped.append(meta)
            continue
        scanned += 1
        pid = problem_id(problem)
        if pid in seen:
            duplicate_problem_ids.append(pid)
        seen.add(pid)

        dag_entry, build_meta = build_for_problem(
            problem,
            meta,
            success_threshold=args.success_threshold,
            allow_zero_success_node=False,
        )
        if dag_entry is None:
            excluded.append(build_meta)
        elif pid not in eligible_set:
            eligible_set.add(pid)
            eligible_ids.append(pid)
            reference_entries_by_id[pid] = dag_entry

        if args.progress_every and scanned % args.progress_every == 0:
            elapsed = max(time.time() - started_at, 1e-9)
            print(
                f"[strict] {model['name']}: scanned={scanned} "
                f"eligible={len(eligible_ids)} rate={scanned / elapsed:.2f}/s",
                file=sys.stderr,
                flush=True,
            )

    return {
        "model": model["name"],
        "source": str(model["path"]),
        "problems_scanned": scanned,
        "strict_eligible_problem_ids": eligible_ids,
        "strict_eligible_count": len(eligible_ids),
        "strict_excluded_count": len(excluded),
        "strict_excluded_preview": excluded[:50],
        "files_skipped": files_skipped[:50],
        "files_skipped_count": len(files_skipped),
        "duplicate_problem_ids_preview": sorted(set(duplicate_problem_ids))[:50],
        "duplicate_problem_ids_count": len(set(duplicate_problem_ids)),
        "_reference_entries_by_id": reference_entries_by_id,
    }


def _entry_problem_id(entry: dict):
    return entry.get("problem_id") or entry.get("aligned_union_problem_id")


def _pool_report_meta(
    *,
    problem_id_value: str | None,
    path: Path | str,
    line_no: int | None,
    reason: str,
    model_name: str | None = None,
    node_id: int | None = None,
) -> dict:
    meta = {
        "problem_id": problem_id_value,
        "path": str(path),
        "reason": reason,
    }
    if line_no is not None:
        meta["line_no"] = line_no
    if model_name is not None:
        meta["model"] = model_name
    if node_id is not None:
        meta["node_id"] = node_id
    return meta


def _template_for_pool_node(
    node: dict,
    *,
    model_name: str,
    problem_id_value: str,
    node_id: int,
):
    template = node.get("template")
    if isinstance(template, dict):
        return template, None

    model_templates = node.get("model_templates") or {}
    if model_name in model_templates and isinstance(model_templates[model_name], dict):
        return model_templates[model_name], None
    if len(model_templates) == 1:
        only_template = next(iter(model_templates.values()))
        if isinstance(only_template, dict):
            return only_template, None
    return None, (
        f"missing_template:problem_id={problem_id_value}:"
        f"model={model_name}:node_id={node_id}"
    )


def _template_availability_reason(template: dict | None, *, require_positive_success: bool) -> str | None:
    if not isinstance(template, dict):
        return "missing_template"

    values = {}
    for key in ("success_prob", "lat_mean", "cost_mean"):
        if key not in template:
            return f"missing_template_key:{key}"
        try:
            value = float(template.get(key))
        except (TypeError, ValueError):
            return f"nonnumeric_template_key:{key}"
        if not math.isfinite(value):
            return f"nonfinite_template_key:{key}"
        values[key] = value

    if require_positive_success and values["success_prob"] <= 0.0:
        return "zero_success_node"
    return None


def _node_pool_available(
    node: dict,
    *,
    model_name: str,
    problem_id_value: str,
    require_positive_success: bool,
) -> tuple[bool, str | None]:
    if bool(node.get("aligned_imputed_missing")):
        return False, "imputed_missing_node"
    try:
        node_id = int(node.get("node_id"))
    except (TypeError, ValueError):
        return False, "invalid_node_id"
    template, reason = _template_for_pool_node(
        node,
        model_name=model_name,
        problem_id_value=problem_id_value,
        node_id=node_id,
    )
    if reason is not None:
        return False, reason
    reason = _template_availability_reason(
        template,
        require_positive_success=require_positive_success,
    )
    if reason is not None:
        return False, reason
    return True, None


def _int_list_or_reason(value, *, field: str, node_id: int):
    if value is None:
        return [], None
    if not isinstance(value, list):
        return None, f"invalid_{field}_list:node_id={node_id}"
    out = []
    for raw in value:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            return None, f"invalid_{field}_id:{raw!r}:node_id={node_id}"
    return sorted(set(out)), None


def _dag_pool_entry_structure_check(
    entry: dict,
    *,
    model_name: str,
    problem_id_value: str,
    dag_pool_path: Path,
    line_no: int | None,
) -> tuple[bool, dict]:
    nodes = entry.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False, _pool_report_meta(
            problem_id_value=problem_id_value,
            path=dag_pool_path,
            line_no=line_no,
            model_name=model_name,
            reason="missing_or_empty_nodes",
        )

    if entry.get("n_nodes") is not None:
        try:
            expected_n_nodes = int(entry.get("n_nodes"))
        except (TypeError, ValueError):
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                reason="invalid_n_nodes",
            )
        if expected_n_nodes != len(nodes):
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                reason=f"n_nodes_mismatch:expected={expected_n_nodes}:actual={len(nodes)}",
            )

    parents_by_id = {}
    children_by_id = {}
    for raw_node in nodes:
        if not isinstance(raw_node, dict):
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                reason="node_is_not_object",
            )
        try:
            node_id = int(raw_node.get("node_id"))
        except (TypeError, ValueError):
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                reason="invalid_node_id",
            )
        if node_id in parents_by_id:
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                node_id=node_id,
                reason="duplicate_node_id",
            )

        parents, reason = _int_list_or_reason(raw_node.get("parents", []), field="parents", node_id=node_id)
        if reason is not None:
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                node_id=node_id,
                reason=reason,
            )
        children, reason = _int_list_or_reason(raw_node.get("children", []), field="children", node_id=node_id)
        if reason is not None:
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                node_id=node_id,
                reason=reason,
            )

        parents_by_id[node_id] = parents
        children_by_id[node_id] = children

    node_ids = set(parents_by_id)
    for node_id, parents in parents_by_id.items():
        missing_parents = sorted(parent for parent in parents if parent not in node_ids)
        if missing_parents:
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                node_id=node_id,
                reason=f"missing_parent_nodes:{missing_parents}",
            )
    for node_id, children in children_by_id.items():
        missing_children = sorted(child for child in children if child not in node_ids)
        if missing_children:
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                node_id=node_id,
                reason=f"missing_child_nodes:{missing_children}",
            )

    expected_children = compute_children(parents_by_id)
    for node_id in sorted(node_ids):
        if children_by_id[node_id] != expected_children[node_id]:
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                node_id=node_id,
                reason=(
                    "children_do_not_match_parents:"
                    f"expected={expected_children[node_id]}:actual={children_by_id[node_id]}"
                ),
            )

    try:
        compute_depths(parents_by_id)
    except Exception as exc:
        return False, _pool_report_meta(
            problem_id_value=problem_id_value,
            path=dag_pool_path,
            line_no=line_no,
            model_name=model_name,
            reason=f"topology_not_dag:{exc}",
        )

    return True, {
        "problem_id": problem_id_value,
        "path": str(dag_pool_path),
        "line_no": line_no,
        "model": model_name,
        "node_count": len(nodes),
    }


def _dag_pool_entry_strict_eligibility(
    entry: dict,
    *,
    model_name: str,
    problem_id_value: str,
    dag_pool_path: Path,
    line_no: int | None,
) -> tuple[bool, dict]:
    ok, meta = _dag_pool_entry_structure_check(
        entry,
        model_name=model_name,
        problem_id_value=problem_id_value,
        dag_pool_path=dag_pool_path,
        line_no=line_no,
    )
    if not ok:
        return False, meta

    for node in entry.get("nodes", []):
        try:
            node_id = int(node.get("node_id"))
        except (TypeError, ValueError):
            node_id = None
        available, reason = _node_pool_available(
            node,
            model_name=model_name,
            problem_id_value=problem_id_value,
            require_positive_success=True,
        )
        if not available:
            return False, _pool_report_meta(
                problem_id_value=problem_id_value,
                path=dag_pool_path,
                line_no=line_no,
                model_name=model_name,
                node_id=node_id,
                reason=reason or "node_unavailable",
            )

    return True, meta


def scan_existing_dag_pool_eligible(model: dict, dag_pool_path: Path) -> dict:
    eligible_ids = []
    eligible_set = set()
    reference_entries_by_id = {}
    excluded = []
    files_skipped = []
    duplicate_problem_ids = []
    malformed_lines = 0
    missing_problem_id_lines = 0
    scanned = 0
    seen_problem_ids = set()

    try:
        with dag_pool_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                scanned += 1
                try:
                    entry = json.loads(text)
                except Exception as exc:
                    malformed_lines += 1
                    files_skipped.append(
                        {
                            "path": str(dag_pool_path),
                            "line_no": line_no,
                            "error": f"json_line_load_failed:{exc}",
                        }
                    )
                    continue
                pid = _entry_problem_id(entry)
                if not pid:
                    missing_problem_id_lines += 1
                    continue
                pid = str(pid)
                if pid in seen_problem_ids:
                    duplicate_problem_ids.append(pid)
                    continue
                seen_problem_ids.add(pid)

                eligible, eligibility_meta = _dag_pool_entry_strict_eligibility(
                    entry,
                    model_name=model["name"],
                    problem_id_value=pid,
                    dag_pool_path=dag_pool_path,
                    line_no=line_no,
                )
                if not eligible:
                    excluded.append(eligibility_meta)
                    continue
                if pid not in eligible_set:
                    eligible_set.add(pid)
                    eligible_ids.append(pid)
                    reference_entries_by_id[pid] = entry
    except FileNotFoundError:
        raise FileNotFoundError(f"dag_pool for model {model['name']} does not exist: {dag_pool_path}")

    return {
        "model": model["name"],
        "source": str(dag_pool_path),
        "union_source": "dag_pool",
        "problems_scanned": scanned,
        "strict_eligible_problem_ids": eligible_ids,
        "strict_eligible_count": len(eligible_ids),
        "strict_excluded_count": len(excluded),
        "strict_excluded_preview": excluded[:50],
        "files_skipped": files_skipped[:50],
        "files_skipped_count": len(files_skipped),
        "malformed_lines": malformed_lines,
        "missing_problem_id_lines": missing_problem_id_lines,
        "duplicate_problem_ids_preview": sorted(set(duplicate_problem_ids))[:50],
        "duplicate_problem_ids_count": len(set(duplicate_problem_ids)),
        "_reference_entries_by_id": reference_entries_by_id,
    }


def load_entries_from_dag_pool(model: dict, dag_pool_path: Path, union_order: list[str]) -> dict:
    union_set = set(union_order)
    entries_by_id = {}
    build_failures = {}
    duplicate_problem_ids = []
    malformed_lines = []
    structural_invalid = []
    missing_problem_id_lines = 0
    scanned = 0

    try:
        with dag_pool_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                scanned += 1
                try:
                    entry = json.loads(text)
                except Exception as exc:
                    malformed_lines.append(
                        {
                            "path": str(dag_pool_path),
                            "line_no": line_no,
                            "error": f"json_line_load_failed:{exc}",
                        }
                    )
                    continue
                pid = _entry_problem_id(entry)
                if not pid:
                    missing_problem_id_lines += 1
                    continue
                pid = str(pid)
                if pid not in union_set:
                    continue
                if pid in entries_by_id:
                    duplicate_problem_ids.append(pid)
                    continue
                structurally_valid, structure_meta = _dag_pool_entry_structure_check(
                    entry,
                    model_name=model["name"],
                    problem_id_value=pid,
                    dag_pool_path=dag_pool_path,
                    line_no=line_no,
                )
                if not structurally_valid:
                    structural_invalid.append(structure_meta)
                    build_failures[pid] = structure_meta
                    continue
                entries_by_id[pid] = entry
    except FileNotFoundError:
        raise FileNotFoundError(f"dag_pool for model {model['name']} does not exist: {dag_pool_path}")

    missing_problem_ids = [pid for pid in union_order if pid not in entries_by_id]
    return {
        "model": model["name"],
        "source": str(dag_pool_path),
        "entry_source": "dag_pool",
        "problems_scanned": scanned,
        "entries_by_id": entries_by_id,
        "build_failures": build_failures,
        "partial_salvage_meta": {},
        "missing_problem_ids": missing_problem_ids,
        "skipped_parse_preview": malformed_lines[:50],
        "skipped_parse_count": len(malformed_lines),
        "structural_invalid_preview": structural_invalid[:50],
        "structural_invalid_count": len(structural_invalid),
        "missing_problem_id_lines": missing_problem_id_lines,
        "duplicate_problem_ids_preview": sorted(set(duplicate_problem_ids))[:50],
        "duplicate_problem_ids_count": len(set(duplicate_problem_ids)),
    }


def _imputed_node_from_topology(
    subproblem: dict,
    *,
    idx: int,
    parents: list[int],
    children: list[int],
    layer: int,
    reason: str,
) -> dict:
    reference_samples = subproblem.get("rollout_records") or []
    return {
        "node_id": idx,
        "parents": parents,
        "children": children,
        "layer": layer,
        "template": {
            "success_prob": 0.0,
            "lat_mean": INF_VALUE,
            "lat_std": INF_VALUE,
            "cost_mean": INF_VALUE,
            "token_cost_mean": INF_VALUE,
        },
        "empirical_samples": _imputed_samples_from_reference(reference_samples),
        "name": subproblem.get("name"),
        "turn_index": idx + 1,
        "selected_rollout_id": subproblem.get("selected_rollout_id"),
        "rollout_success_rate": 0.0,
        "cost_source": "imputed_missing_inf",
        "aligned_imputed_missing": True,
        "aligned_node_imputation_reason": reason,
    }


def _node_from_normalized_records(
    subproblem: dict,
    *,
    idx: int,
    parents: list[int],
    children: list[int],
    layer: int,
    normalized_records: list[dict],
) -> dict:
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
    return {
        "node_id": idx,
        "parents": parents,
        "children": children,
        "layer": layer,
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
        "aligned_imputed_missing": False,
        "aligned_partial_salvaged_node": True,
    }


def build_partial_dag_entry(
    problem: dict,
    source_path: str,
    *,
    success_threshold: float,
    failure_meta: dict | None,
) -> tuple[dict | None, dict]:
    problem_id_value = problem_id(problem)
    subproblems = problem.get("subproblems", [])
    if not subproblems:
        return None, {"problem_id": problem_id_value, "reason": "partial_no_subproblems", "source_path": source_path}

    names = [subproblem.get("name") or f"node_{idx}" for idx, subproblem in enumerate(subproblems)]
    if len(set(names)) != len(names):
        return None, {"problem_id": problem_id_value, "reason": "partial_duplicate_subproblem_names", "source_path": source_path}
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    parents_by_idx = {}
    normalized_by_idx = {}
    unavailable_nodes = []
    available_nodes = []

    for idx, subproblem in enumerate(subproblems):
        dependency_names = subproblem.get("dependencies") or []
        parents = []
        for dep_name in dependency_names:
            if dep_name not in name_to_idx:
                return None, {
                    "problem_id": problem_id_value,
                    "reason": f"partial_missing_dependency:{dep_name}",
                    "source_path": source_path,
                }
            parents.append(name_to_idx[dep_name])
        parents_by_idx[idx] = sorted(set(parents))

        normalized_records = normalize_rollout_records(subproblem, success_threshold)
        observed_latencies = [
            record["latency"]
            for record in normalized_records
            if record["latency"] is not None
        ]
        if normalized_records and observed_latencies:
            normalized_by_idx[idx] = normalized_records
            available_nodes.append(idx)
        else:
            reason = (
                f"missing_infer_latency:{subproblem.get('name', idx)}"
                if normalized_records
                else f"missing_rollout_records:{subproblem.get('name', idx)}"
            )
            normalized_by_idx[idx] = []
            unavailable_nodes.append({"node_id": idx, "name": subproblem.get("name"), "reason": reason})

    try:
        depths = compute_depths(parents_by_idx)
    except Exception as exc:
        return None, {
            "problem_id": problem_id_value,
            "reason": f"partial_topology_failed:{exc}",
            "source_path": source_path,
        }
    children_by_idx = compute_children(parents_by_idx)

    nodes = []
    for idx, subproblem in enumerate(subproblems):
        parents = parents_by_idx[idx]
        children = children_by_idx[idx]
        layer = int(subproblem.get("depth")) if isinstance(subproblem.get("depth"), int) else depths[idx]
        normalized_records = normalized_by_idx[idx]
        if normalized_records:
            nodes.append(
                _node_from_normalized_records(
                    subproblem,
                    idx=idx,
                    parents=parents,
                    children=children,
                    layer=layer,
                    normalized_records=normalized_records,
                )
            )
        else:
            reason = next(item["reason"] for item in unavailable_nodes if item["node_id"] == idx)
            nodes.append(
                _imputed_node_from_topology(
                    subproblem,
                    idx=idx,
                    parents=parents,
                    children=children,
                    layer=layer,
                    reason=reason,
                )
            )

    if not available_nodes:
        return None, {
            "problem_id": problem_id_value,
            "reason": "partial_no_available_nodes",
            "source_path": source_path,
            "original_failure": failure_meta,
        }

    entry = {
        "dag_kind": "codeflow_rollout",
        "n_nodes": len(nodes),
        "nodes": nodes,
        "problem_id": problem_id_value,
        "source_path": source_path,
        "overall_turns": problem.get("overall-turns"),
        "overall_depth": problem.get("overall-depth"),
        "aligned_partial_imputed_missing": bool(unavailable_nodes),
        "aligned_partial_unavailable_nodes": unavailable_nodes,
        "aligned_partial_available_node_ids": available_nodes,
        "aligned_partial_original_failure": failure_meta,
    }
    return entry, {
        "problem_id": problem_id_value,
        "source_path": source_path,
        "node_count": len(nodes),
        "partial_available_node_count": len(available_nodes),
        "partial_unavailable_node_count": len(unavailable_nodes),
        "partial_unavailable_nodes": unavailable_nodes,
        "original_failure": failure_meta,
    }


def build_model_entries(model: dict, union_order: list[str], args) -> dict:
    if model.get("dag_pool_path") is not None:
        return load_entries_from_dag_pool(model, model["dag_pool_path"], union_order)

    union_set = set(union_order)
    entries_by_id = {}
    build_failures = {}
    partial_salvage_meta = {}
    skipped_parse = []
    scanned = 0
    started_at = time.time()

    for problem, meta in iter_model_problems(model["path"], args.recursive):
        if problem is None:
            skipped_parse.append(meta)
            continue
        scanned += 1
        pid = problem_id(problem)
        if pid not in union_set or pid in entries_by_id:
            continue

        dag_entry, build_meta = build_for_problem(
            problem,
            meta,
            success_threshold=args.success_threshold,
            allow_zero_success_node=True,
        )
        if dag_entry is None:
            partial_entry, partial_meta = build_partial_dag_entry(
                problem,
                meta["path"],
                success_threshold=args.success_threshold,
                failure_meta=build_meta,
            )
            if partial_entry is None:
                build_failures[pid] = partial_meta
            else:
                entries_by_id[pid] = partial_entry
                partial_salvage_meta[pid] = partial_meta
        else:
            entries_by_id[pid] = dag_entry

        if args.progress_every and scanned % args.progress_every == 0:
            elapsed = max(time.time() - started_at, 1e-9)
            print(
                f"[aligned] {model['name']}: scanned={scanned} "
                f"built={len(entries_by_id)}/{len(union_order)} rate={scanned / elapsed:.2f}/s",
                file=sys.stderr,
                flush=True,
            )

    missing_problem_ids = [
        pid
        for pid in union_order
        if pid not in entries_by_id
    ]
    return {
        "model": model["name"],
        "source": str(model["path"]),
        "problems_scanned": scanned,
        "entries_by_id": entries_by_id,
        "build_failures": build_failures,
        "partial_salvage_meta": partial_salvage_meta,
        "missing_problem_ids": missing_problem_ids,
        "skipped_parse_preview": skipped_parse[:50],
        "skipped_parse_count": len(skipped_parse),
    }


def _public_report(report: dict) -> dict:
    return {key: value for key, value in report.items() if not str(key).startswith("_")}


def _collect_reference_entries(strict_reports: list[dict], aligned_reports: list[dict]) -> dict:
    refs = {}
    for report in strict_reports:
        for pid, entry in report.get("_reference_entries_by_id", {}).items():
            refs.setdefault(pid, entry)
    for report in aligned_reports:
        for pid, entry in report.get("entries_by_id", {}).items():
            refs.setdefault(pid, entry)
    return refs


def _imputed_samples_from_reference(reference_samples: list[dict] | None) -> list[dict]:
    reference_samples = reference_samples or []
    count = max(1, len(reference_samples))
    out = []
    for idx in range(count):
        ref = reference_samples[idx] if idx < len(reference_samples) and isinstance(reference_samples[idx], dict) else {}
        out.append(
            {
                "rollout_id": ref.get("rollout_id", idx),
                "infer_latency_sec": INF_VALUE,
                "infer_batch_latency_sec": INF_VALUE,
                "infer_per_rollout_latency_sec": INF_VALUE,
                "latency_observation": "imputed_missing_model",
                "eval_latency_sec": INF_VALUE,
                "total_latency_sec": INF_VALUE,
                "latency": INF_VALUE,
                "success": 0,
                "token_cost": INF_VALUE,
                "cost": INF_VALUE,
                "pass_rate": 0.0,
                "aligned_imputed_missing": True,
            }
        )
    return out


def impute_missing_dag_entry(reference_entry: dict, model: dict, failure_meta: dict | None) -> dict:
    entry = copy.deepcopy(reference_entry)
    entry["source_path"] = str(model["path"])
    entry["aligned_imputed_missing"] = True
    entry["aligned_imputed_model"] = model["name"]
    entry["aligned_imputation_reason"] = failure_meta or {"reason": "missing_problem_or_build_failed"}

    for node in entry.get("nodes", []):
        reference_samples = node.get("empirical_samples") or []
        imputed_samples = _imputed_samples_from_reference(reference_samples)
        node["template"] = {
            "success_prob": 0.0,
            "lat_mean": INF_VALUE,
            "lat_std": INF_VALUE,
            "cost_mean": INF_VALUE,
            "token_cost_mean": INF_VALUE,
        }
        node["empirical_samples"] = imputed_samples
        node.pop("model_templates", None)
        node.pop("empirical_samples_by_model", None)
        node["rollout_success_rate"] = 0.0
        node["cost_source"] = "imputed_missing_inf"
        node["aligned_imputed_missing"] = True

    return entry


def missing_by_model_from_reports(aligned_reports: list[dict]) -> dict:
    missing_by_model = {}
    for report in aligned_reports:
        if report["missing_problem_ids"]:
            missing_by_model[report["model"]] = {
                "missing_problem_ids": report["missing_problem_ids"],
                "missing_count": len(report["missing_problem_ids"]),
                "build_failures_preview": {
                    pid: report["build_failures"].get(pid)
                    for pid in report["missing_problem_ids"][:50]
                    if pid in report["build_failures"]
                },
                "missing_problem_ids_preview": report["missing_problem_ids"][:50],
            }
    return missing_by_model


def apply_imputation_policy(
    models: list[dict],
    aligned_reports: list[dict],
    reference_entries_by_id: dict,
) -> tuple[dict, dict]:
    imputed_by_model = {}
    unresolved_by_model = {}

    for model, report in zip(models, aligned_reports):
        imputed_problem_ids = []
        unresolved_problem_ids = []
        for pid in list(report["missing_problem_ids"]):
            reference_entry = reference_entries_by_id.get(pid)
            if reference_entry is None:
                unresolved_problem_ids.append(pid)
                continue
            report["entries_by_id"][pid] = impute_missing_dag_entry(
                reference_entry,
                model,
                report["build_failures"].get(pid),
            )
            imputed_problem_ids.append(pid)

        report["imputed_problem_ids"] = imputed_problem_ids
        report["missing_problem_ids"] = unresolved_problem_ids

        if imputed_problem_ids:
            imputed_by_model[report["model"]] = {
                "imputed_count": len(imputed_problem_ids),
                "imputed_problem_ids_preview": imputed_problem_ids[:50],
            }
        if unresolved_problem_ids:
            unresolved_by_model[report["model"]] = {
                "missing_count": len(unresolved_problem_ids),
                "missing_problem_ids_preview": unresolved_problem_ids[:50],
            }

    return imputed_by_model, unresolved_by_model


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_model_pool(
    output_root: Path,
    model: dict,
    final_problem_ids: list[str],
    entries_by_id: dict,
    source_path: Path,
    summary_extra: dict,
) -> dict:
    model_dir = output_root / sanitize_name(model["name"])
    model_dir.mkdir(parents=True, exist_ok=True)
    dag_pool_path = model_dir / "dag_pool.jsonl"
    indices_path = model_dir / "selected_dag_indices.json"
    summary_path = model_dir / "build_summary.json"

    with dag_pool_path.open("w", encoding="utf-8") as f:
        for idx, pid in enumerate(final_problem_ids):
            entry = dict(entries_by_id[pid])
            entry["aligned_union_idx"] = idx
            entry["aligned_union_problem_id"] = pid
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    indices_payload = {
        "dag_indices": list(range(len(final_problem_ids))),
        "problem_ids": final_problem_ids,
        "index_semantics": "row index in this aligned model dag_pool; shared across all generated model pools",
    }
    write_json(indices_path, indices_payload)

    summary = {
        "model": model["name"],
        "source": str(source_path.resolve()),
        "dags_written": len(final_problem_ids),
        "outputs": {
            "dag_pool_jsonl": str(dag_pool_path.resolve()),
            "selected_dag_indices_json": str(indices_path.resolve()),
        },
        **summary_extra,
    }
    write_json(summary_path, summary)
    return {
        "model": model["name"],
        "output_dir": str(model_dir.resolve()),
        "dag_pool_jsonl": str(dag_pool_path.resolve()),
        "selected_dag_indices_json": str(indices_path.resolve()),
        "build_summary_json": str(summary_path.resolve()),
        "dags_written": len(final_problem_ids),
    }


def choose_baseline_model(models: list[dict], requested: str) -> str:
    model_names = [model["name"] for model in models]
    if requested:
        if requested not in model_names:
            raise ValueError(
                f"--baseline-model {requested!r} is not one of configured models: {model_names}"
            )
        return requested
    if "default" in model_names:
        return "default"
    if not model_names:
        raise ValueError("At least one --model is required.")
    return model_names[0]


def _nodes_by_id(entry: dict, *, model_name: str, problem_id_value: str) -> dict:
    nodes = {}
    for node in entry.get("nodes", []):
        if "node_id" not in node:
            raise ValueError(f"Model {model_name} problem {problem_id_value} has a node without node_id.")
        nid = int(node["node_id"])
        if nid in nodes:
            raise ValueError(f"Model {model_name} problem {problem_id_value} has duplicate node_id={nid}.")
        nodes[nid] = node
    return nodes


def _node_topology(node: dict) -> tuple:
    return (
        int(node["node_id"]),
        tuple(int(parent) for parent in node.get("parents", [])),
        tuple(int(child) for child in node.get("children", [])),
    )


def _template_for_model_node(node: dict, model_name: str, problem_id_value: str, nid: int) -> dict:
    if node.get("template") is not None:
        return copy.deepcopy(node["template"])
    model_templates = node.get("model_templates") or {}
    if model_name in model_templates:
        return copy.deepcopy(model_templates[model_name])
    if len(model_templates) == 1:
        return copy.deepcopy(next(iter(model_templates.values())))
    raise ValueError(
        f"Missing template for problem_id={problem_id_value} node_id={nid} model={model_name!r}."
    )


def _empirical_samples_for_model_node(node: dict, model_name: str):
    if node.get("empirical_samples") is not None:
        return copy.deepcopy(node["empirical_samples"])
    samples_by_model = node.get("empirical_samples_by_model") or {}
    if model_name in samples_by_model:
        return copy.deepcopy(samples_by_model[model_name])
    if len(samples_by_model) == 1:
        return copy.deepcopy(next(iter(samples_by_model.values())))
    return None


def _template_is_finite(template: dict | None) -> bool:
    if not isinstance(template, dict):
        return False
    for key in ("success_prob", "lat_mean", "cost_mean"):
        try:
            value = float(template.get(key))
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def _template_is_available(template: dict | None, *, require_positive_success: bool = True) -> bool:
    return (
        _template_is_finite(template)
        and _template_availability_reason(
            template,
            require_positive_success=require_positive_success,
        )
        is None
    )


def _node_model_available(node: dict, model_name: str, problem_id_value: str, nid: int) -> bool:
    if bool(node.get("aligned_imputed_missing")):
        return False
    try:
        template = _template_for_model_node(node, model_name, problem_id_value, nid)
    except ValueError:
        return False
    return _template_is_available(template, require_positive_success=True)


def _validate_compatible_topology(
    *,
    problem_id_value: str,
    baseline_model_name: str,
    baseline_nodes: dict,
    model_name: str,
    model_nodes: dict,
) -> None:
    baseline_ids = set(baseline_nodes)
    model_ids = set(model_nodes)
    if baseline_ids != model_ids:
        missing = sorted(baseline_ids.difference(model_ids))
        extra = sorted(model_ids.difference(baseline_ids))
        raise ValueError(
            f"Topology mismatch for problem_id={problem_id_value}: model={model_name!r} "
            f"does not match baseline_model={baseline_model_name!r}; missing_nodes={missing}, extra_nodes={extra}"
        )
    for nid in sorted(baseline_ids):
        base_topology = _node_topology(baseline_nodes[nid])
        model_topology = _node_topology(model_nodes[nid])
        if base_topology != model_topology:
            raise ValueError(
                f"Topology mismatch for problem_id={problem_id_value} node_id={nid}: "
                f"model={model_name!r} has parents/children different from "
                f"baseline_model={baseline_model_name!r}."
            )


def build_combined_dag_entry(
    *,
    problem_id_value: str,
    aligned_idx: int,
    models: list[dict],
    entries_by_model: dict,
    baseline_model_name: str,
) -> dict:
    baseline_entry = entries_by_model[baseline_model_name]
    combined = copy.deepcopy(baseline_entry)
    combined["problem_id"] = problem_id_value
    combined["aligned_union_idx"] = aligned_idx
    combined["aligned_union_problem_id"] = problem_id_value
    combined["model_ids"] = [model["name"] for model in models]
    combined["baseline_model"] = baseline_model_name
    combined["source_path_by_model"] = {
        model["name"]: entries_by_model[model["name"]].get("source_path", str(model["path"]))
        for model in models
    }
    combined["aligned_imputed_by_model"] = {
        model["name"]: bool(entries_by_model[model["name"]].get("aligned_imputed_missing"))
        for model in models
    }
    combined["aligned_partial_imputed_by_model"] = {
        model["name"]: bool(entries_by_model[model["name"]].get("aligned_partial_imputed_missing"))
        for model in models
    }
    combined["model_availability_semantics"] = (
        "Models omitted from a node's model_templates are unavailable for that node; "
        "imputed missing nodes are unavailable, while other real nodes from the same "
        "partial model/problem entry remain available."
    )

    baseline_nodes = _nodes_by_id(
        baseline_entry,
        model_name=baseline_model_name,
        problem_id_value=problem_id_value,
    )
    nodes_by_model = {}
    for model in models:
        model_name = model["name"]
        model_nodes = _nodes_by_id(
            entries_by_model[model_name],
            model_name=model_name,
            problem_id_value=problem_id_value,
        )
        _validate_compatible_topology(
            problem_id_value=problem_id_value,
            baseline_model_name=baseline_model_name,
            baseline_nodes=baseline_nodes,
            model_name=model_name,
            model_nodes=model_nodes,
        )
        nodes_by_model[model_name] = model_nodes

    combined_nodes = []
    for baseline_node in baseline_entry.get("nodes", []):
        nid = int(baseline_node["node_id"])
        node_out = copy.deepcopy(baseline_node)
        node_out.pop("model_templates", None)
        node_out.pop("empirical_samples_by_model", None)

        model_templates = {}
        empirical_samples_by_model = {}
        model_node_metadata = {}
        available_model_ids = []
        for model in models:
            model_name = model["name"]
            model_node = nodes_by_model[model_name][nid]
            available = _node_model_available(model_node, model_name, problem_id_value, nid)
            if available:
                model_templates[model_name] = _template_for_model_node(
                    model_node,
                    model_name,
                    problem_id_value,
                    nid,
                )
                model_samples = _empirical_samples_for_model_node(model_node, model_name)
                if model_samples is not None:
                    empirical_samples_by_model[model_name] = model_samples
                available_model_ids.append(model_name)
            model_node_metadata[model_name] = {
                "rollout_success_rate": model_node.get("rollout_success_rate"),
                "cost_source": model_node.get("cost_source"),
                "aligned_imputed_missing": bool(model_node.get("aligned_imputed_missing")),
                "available": bool(available),
            }

        if not available_model_ids:
            raise ValueError(
                f"No available finite model statistics for problem_id={problem_id_value} node_id={nid}."
            )

        legacy_model_name = (
            baseline_model_name
            if baseline_model_name in available_model_ids
            else available_model_ids[0]
        )
        legacy_node = nodes_by_model[legacy_model_name][nid]
        legacy_template = _template_for_model_node(
            legacy_node,
            legacy_model_name,
            problem_id_value,
            nid,
        )
        legacy_samples = _empirical_samples_for_model_node(legacy_node, legacy_model_name)
        node_out["template"] = legacy_template
        if legacy_samples is not None:
            node_out["empirical_samples"] = legacy_samples
        else:
            node_out.pop("empirical_samples", None)
        node_out["legacy_template_model"] = legacy_model_name
        node_out["available_model_ids"] = available_model_ids
        node_out["aligned_imputed_missing"] = False
        node_out["rollout_success_rate"] = legacy_node.get(
            "rollout_success_rate",
            legacy_template.get("success_prob") if isinstance(legacy_template, dict) else None,
        )
        if legacy_node.get("cost_source") is not None:
            node_out["cost_source"] = legacy_node.get("cost_source")

        node_out["model_templates"] = model_templates
        if empirical_samples_by_model:
            node_out["empirical_samples_by_model"] = empirical_samples_by_model
        node_out["model_node_metadata"] = model_node_metadata
        combined_nodes.append(node_out)

    combined["nodes"] = combined_nodes
    combined["n_nodes"] = len(combined_nodes)
    return combined


def write_combined_pool(
    output_root: Path,
    args,
    models: list[dict],
    final_problem_ids: list[str],
    aligned_reports: list[dict],
    baseline_model_name: str,
    summary_extra: dict,
) -> dict:
    dag_pool_path = Path(args.combined_dag_pool) if args.combined_dag_pool else output_root / "multi_model_dag_pool.jsonl"
    indices_path = Path(args.combined_indices) if args.combined_indices else output_root / "multi_model_selected_dag_indices.json"
    summary_path = Path(args.combined_summary) if args.combined_summary else output_root / "multi_model_build_summary.json"

    reports_by_model = {report["model"]: report for report in aligned_reports}
    dag_pool_path.parent.mkdir(parents=True, exist_ok=True)
    with dag_pool_path.open("w", encoding="utf-8") as f:
        for idx, pid in enumerate(final_problem_ids):
            entries_by_model = {
                model["name"]: reports_by_model[model["name"]]["entries_by_id"][pid]
                for model in models
            }
            entry = build_combined_dag_entry(
                problem_id_value=pid,
                aligned_idx=idx,
                models=models,
                entries_by_model=entries_by_model,
                baseline_model_name=baseline_model_name,
            )
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    indices_payload = {
        "dag_indices": list(range(len(final_problem_ids))),
        "problem_ids": final_problem_ids,
        "index_semantics": "row index in the combined model-aware dag_pool",
        "model_ids": [model["name"] for model in models],
        "baseline_model": baseline_model_name,
    }
    write_json(indices_path, indices_payload)

    summary = {
        "dag_pool_jsonl": str(dag_pool_path.resolve()),
        "selected_dag_indices_json": str(indices_path.resolve()),
        "model_ids": [model["name"] for model in models],
        "baseline_model": baseline_model_name,
        "dags_written": len(final_problem_ids),
        **summary_extra,
    }
    write_json(summary_path, summary)
    return {
        "dag_pool_jsonl": str(dag_pool_path.resolve()),
        "selected_dag_indices_json": str(indices_path.resolve()),
        "build_summary_json": str(summary_path.resolve()),
        "model_ids": [model["name"] for model in models],
        "baseline_model": baseline_model_name,
        "dags_written": len(final_problem_ids),
    }


def main() -> int:
    args = parse_args()
    dag_pool_specs = parse_named_path_specs(args.dag_pool, "--dag-pool")
    models = []
    seen_names = set()
    for spec in args.model:
        name, path = parse_model_spec(spec)
        if name in seen_names:
            raise ValueError(f"Duplicate model name: {name}")
        seen_names.add(name)
        models.append({"name": name, "path": path, "dag_pool_path": dag_pool_specs.get(name)})

    if dag_pool_specs:
        model_names = {model["name"] for model in models}
        unknown_specs = sorted(set(dag_pool_specs).difference(model_names))
        if unknown_specs:
            raise ValueError(
                "--dag-pool was provided for unknown model names: "
                + ", ".join(unknown_specs)
            )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_model_name = choose_baseline_model(models, args.baseline_model)

    if dag_pool_specs and len(dag_pool_specs) == len(models):
        union_source = "dag_pool"
    elif dag_pool_specs:
        union_source = "mixed_dag_pool_and_harness_strict_scan"
    else:
        union_source = "harness_strict_scan"
    print(
        f"[1/3] Reading union candidates across {len(models)} models "
        f"(union_source={union_source})",
        flush=True,
    )
    strict_reports = []
    union_order = []
    union_seen = set()
    for model in models:
        if model.get("dag_pool_path") is not None:
            report = scan_existing_dag_pool_eligible(model, model["dag_pool_path"])
        else:
            report = scan_strict_eligible(model, args)
        strict_reports.append(report)
        for pid in report["strict_eligible_problem_ids"]:
            if pid not in union_seen:
                union_seen.add(pid)
                union_order.append(pid)
        print(
            f"  {model['name']}: strict_eligible={report['strict_eligible_count']} "
            f"excluded={report.get('strict_excluded_count', 0)}",
            flush=True,
        )

    print(f"[2/3] Building per-model DAG entries over union size={len(union_order)}", flush=True)
    aligned_reports = []
    for model in models:
        report = build_model_entries(model, union_order, args)
        aligned_reports.append(report)
        print(
            f"  {model['name']}: buildable={len(report['entries_by_id'])}/{len(union_order)} "
            f"missing_or_failed={len(report['missing_problem_ids'])}",
            flush=True,
        )

    missing_by_model_before_policy = missing_by_model_from_reports(aligned_reports)
    reference_entries_by_id = _collect_reference_entries(strict_reports, aligned_reports)
    imputed_by_model = {}
    unresolved_imputation_by_model = {}

    if missing_by_model_before_policy and args.missing_policy == "impute":
        imputed_by_model, unresolved_imputation_by_model = apply_imputation_policy(
            models,
            aligned_reports,
            reference_entries_by_id,
        )
        for report in aligned_reports:
            print(
                f"  {report['model']}: imputed={len(report.get('imputed_problem_ids', []))} "
                f"unresolved={len(report['missing_problem_ids'])}",
                flush=True,
            )

    missing_by_model = missing_by_model_from_reports(aligned_reports)
    strict_reports_public = [_public_report(report) for report in strict_reports]

    dropped_problem_ids = []
    if missing_by_model and args.missing_policy == "error":
        manifest = {
            "status": "error",
            "reason": "Some union problem_ids cannot be built for one or more models.",
            "hint": "Fix missing verify/rollout records, or rerun with --missing-policy impute/drop.",
            "output_root": str(output_root.resolve()),
            "strict_union_problem_count": len(union_order),
            "strict_union_problem_ids": union_order,
            "missing_by_model": missing_by_model,
            "missing_by_model_before_policy": missing_by_model_before_policy,
            "strict_reports": strict_reports_public,
        }
        write_json(output_root / "union_manifest.json", manifest)
        print(
            f"ERROR: {sum(v['missing_count'] for v in missing_by_model.values())} "
            f"model/problem build gaps. See {output_root / 'union_manifest.json'}",
            file=sys.stderr,
        )
        return 2

    if missing_by_model and args.missing_policy == "impute":
        manifest = {
            "status": "error",
            "reason": "Some union problem_ids could not be imputed because no reference DAG topology was available.",
            "output_root": str(output_root.resolve()),
            "strict_union_problem_count": len(union_order),
            "strict_union_problem_ids": union_order,
            "missing_by_model": missing_by_model,
            "missing_by_model_before_policy": missing_by_model_before_policy,
            "imputed_by_model": imputed_by_model,
            "unresolved_imputation_by_model": unresolved_imputation_by_model,
            "strict_reports": strict_reports_public,
        }
        write_json(output_root / "union_manifest.json", manifest)
        print(
            f"ERROR: imputation still has "
            f"{sum(v['missing_count'] for v in missing_by_model.values())} unresolved model/problem gaps. "
            f"See {output_root / 'union_manifest.json'}",
            file=sys.stderr,
        )
        return 2

    if missing_by_model and args.missing_policy == "drop":
        bad = set()
        for report in aligned_reports:
            bad.update(report["missing_problem_ids"])
        dropped_problem_ids = [pid for pid in union_order if pid in bad]
        final_problem_ids = [pid for pid in union_order if pid not in bad]
    else:
        final_problem_ids = list(union_order)

    print(f"[3/4] Writing aligned pools final_size={len(final_problem_ids)}", flush=True)
    outputs = []
    for model, report in zip(models, aligned_reports):
        extra = {
            "strict_union_problem_count": len(union_order),
            "final_aligned_problem_count": len(final_problem_ids),
            "dropped_problem_count": len(dropped_problem_ids),
            "dropped_problem_ids_preview": dropped_problem_ids[:50],
            "missing_problem_count_before_policy": (
                missing_by_model_before_policy.get(report["model"], {}).get("missing_count", 0)
            ),
            "missing_problem_ids_preview_before_policy": (
                missing_by_model_before_policy.get(report["model"], {}).get("missing_problem_ids_preview", [])
            ),
            "build_failures_preview": {
                pid: report["build_failures"].get(pid)
                for pid in missing_by_model_before_policy.get(report["model"], {}).get("missing_problem_ids_preview", [])
                if pid in report["build_failures"]
            },
            "partial_salvaged_problem_count": len(report.get("partial_salvage_meta", {})),
            "partial_salvaged_problem_ids_preview": list(report.get("partial_salvage_meta", {}))[:50],
            "imputed_problem_count": len(report.get("imputed_problem_ids", [])),
            "imputed_problem_ids_preview": report.get("imputed_problem_ids", [])[:50],
            "skipped_parse_count": report["skipped_parse_count"],
            "skipped_parse_preview": report["skipped_parse_preview"],
            "structural_invalid_count": report.get("structural_invalid_count", 0),
            "structural_invalid_preview": report.get("structural_invalid_preview", []),
            "filter": {
                "union_membership": "strictly buildable in at least one model with zero-success nodes disallowed",
                "aligned_pool_zero_success_nodes": "retained",
                "aligned_pool_missing_entries": (
                    "imputed_success_zero_latency_cost_token_cost_infinity"
                    if args.missing_policy == "impute"
                    else "not_imputed"
                ),
                "require_all_nodes_evaluated": True,
                "success_threshold": args.success_threshold,
                "missing_policy": args.missing_policy,
            },
        }
        outputs.append(
            write_model_pool(
                output_root,
                model,
                final_problem_ids,
                report["entries_by_id"],
                model["path"],
                extra,
            )
        )

    print(
        f"[4/4] Writing combined model-aware pool baseline_model={baseline_model_name}",
        flush=True,
    )
    combined_output = write_combined_pool(
        output_root,
        args,
        models,
        final_problem_ids,
        aligned_reports,
        baseline_model_name,
        {
            "strict_union_problem_count": len(union_order),
            "final_aligned_problem_count": len(final_problem_ids),
            "dropped_problem_count": len(dropped_problem_ids),
            "missing_policy": args.missing_policy,
            "missing_by_model_before_policy": missing_by_model_before_policy,
            "imputed_by_model": imputed_by_model,
            "partial_salvaged_by_model": {
                report["model"]: {
                    "partial_salvaged_problem_count": len(report.get("partial_salvage_meta", {})),
                    "partial_salvaged_problem_ids_preview": list(report.get("partial_salvage_meta", {}))[:50],
                }
                for report in aligned_reports
                if report.get("partial_salvage_meta")
            },
        },
    )

    manifest = {
        "status": "ok",
        "output_root": str(output_root.resolve()),
        "models": [
            {
                "name": model["name"],
                "source": str(model["path"].resolve()),
                "dag_pool_source": (
                    str(model["dag_pool_path"].resolve())
                    if model.get("dag_pool_path") is not None
                    else None
                ),
            }
            for model in models
        ],
        "union_source": union_source,
        "strict_union_problem_count": len(union_order),
        "final_aligned_problem_count": len(final_problem_ids),
        "strict_union_problem_ids": union_order,
        "final_aligned_problem_ids": final_problem_ids,
        "dropped_problem_count": len(dropped_problem_ids),
        "dropped_problem_ids": dropped_problem_ids,
        "missing_policy": args.missing_policy,
        "missing_by_model": missing_by_model,
        "missing_by_model_before_policy": missing_by_model_before_policy,
        "imputed_by_model": imputed_by_model,
        "partial_salvaged_by_model": {
            report["model"]: {
                "partial_salvaged_problem_count": len(report.get("partial_salvage_meta", {})),
                "partial_salvaged_problem_ids_preview": list(report.get("partial_salvage_meta", {}))[:50],
            }
            for report in aligned_reports
            if report.get("partial_salvage_meta")
        },
        "strict_reports": strict_reports_public,
        "outputs": outputs,
        "combined_output": combined_output,
    }
    write_json(output_root / "union_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
