"""
English note.

English note results/shards/ English note raw_budget_*_dag*.jsonl English note,
English note budget_ratio English note raw_budget_{ratio}.jsonl,
English note.

English note:
    python merge_shards.py \
        --shard_dir results/shards \
        --output_dir results/merged \
        [--n_dags 10000] [--n_exec_trials 1000]

English note:
    results/merged/raw_budget_0.3.jsonl
    results/merged/raw_budget_0.5.jsonl
    ...
    results/merged/params.json
    results/merged/merge_report.json
"""

import os
import sys
import json
import glob
import re
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def _format_deadline_dir(deadline_mode: str, deadline_axis_value: float) -> str:
    return f"deadline_{deadline_mode}_{deadline_axis_value}"


def _format_budget_dir(budget_input_type: str, budget_axis_value: float) -> str:
    return f"budget_{budget_input_type}_{budget_axis_value}"


def _format_raw_budget_name(budget_input_type: str, budget_axis_value: float) -> str:
    if budget_input_type == "value":
        return f"raw_budget_value_{budget_axis_value}.jsonl"
    return f"raw_budget_{budget_axis_value}.jsonl"


def _format_budget_cost_dir(budget_cost_mode: str) -> str:
    return f"budget_cost_{budget_cost_mode}"


def _parse_shard_filename(basename: str):
    patterns = [
        re.compile(
            r"raw_budget_value_(?P<bval>.+?)_(?P<deadline_tag>deadline_(?:ratio|value)_.+?)_"
            r"(?:(?P<budget_cost_tag>budget_cost_(?:token_cost_mean|output_money_cost_mean|cost_mean|lat_mean))_)?"
            r"(?:(?P<mode_tag>all|baseline_only|loader_only)_)?"
            r"(?P<stype>dag|sel)(?P<start>\d+)-(?P<end>\d+)\.jsonl"
        ),
        re.compile(
            r"raw_budget_(?P<br>.+?)_(?P<deadline_tag>deadline_(?:ratio|value)_.+?)_"
            r"(?:(?P<budget_cost_tag>budget_cost_(?:token_cost_mean|output_money_cost_mean|cost_mean|lat_mean))_)?"
            r"(?:(?P<mode_tag>all|baseline_only|loader_only)_)?"
            r"(?P<stype>dag|sel)(?P<start>\d+)-(?P<end>\d+)\.jsonl"
        ),
        re.compile(
            r"raw_budget_value_(?P<bval>.+?)_"
            r"(?:(?P<budget_cost_tag>budget_cost_(?:token_cost_mean|output_money_cost_mean|cost_mean|lat_mean))_)?"
            r"(?:(?P<mode_tag>all|baseline_only|loader_only)_)?"
            r"(?P<stype>dag|sel)(?P<start>\d+)-(?P<end>\d+)\.jsonl"
        ),
        re.compile(
            r"raw_budget_(?P<br>.+?)_"
            r"(?:(?P<budget_cost_tag>budget_cost_(?:token_cost_mean|output_money_cost_mean|cost_mean|lat_mean))_)?"
            r"(?:(?P<mode_tag>all|baseline_only|loader_only)_)?"
            r"(?P<stype>dag|sel)(?P<start>\d+)-(?P<end>\d+)\.jsonl"
        ),
        re.compile(
            r"raw_budget_value_(?P<bval>.+?)_(?P<deadline_tag>deadline_(?:ratio|value)_.+?)_"
            r"(?:(?P<budget_cost_tag>budget_cost_(?:token_cost_mean|output_money_cost_mean|cost_mean|lat_mean))_)?"
            r"(?:(?P<mode_tag>all|baseline_only|loader_only)_)?"
            r"(?P<stype>dag|sel)(?P<idx>\d+)\.jsonl"
        ),
        re.compile(
            r"raw_budget_(?P<br>.+?)_(?P<deadline_tag>deadline_(?:ratio|value)_.+?)_"
            r"(?:(?P<budget_cost_tag>budget_cost_(?:token_cost_mean|output_money_cost_mean|cost_mean|lat_mean))_)?"
            r"(?:(?P<mode_tag>all|baseline_only|loader_only)_)?"
            r"(?P<stype>dag|sel)(?P<idx>\d+)\.jsonl"
        ),
        re.compile(
            r"raw_budget_value_(?P<bval>.+?)_"
            r"(?:(?P<budget_cost_tag>budget_cost_(?:token_cost_mean|output_money_cost_mean|cost_mean|lat_mean))_)?"
            r"(?:(?P<mode_tag>all|baseline_only|loader_only)_)?"
            r"(?P<stype>dag|sel)(?P<idx>\d+)\.jsonl"
        ),
        re.compile(
            r"raw_budget_(?P<br>.+?)_"
            r"(?:(?P<budget_cost_tag>budget_cost_(?:token_cost_mean|output_money_cost_mean|cost_mean|lat_mean))_)?"
            r"(?:(?P<mode_tag>all|baseline_only|loader_only)_)?"
            r"(?P<stype>dag|sel)(?P<idx>\d+)\.jsonl"
        ),
    ]
    for pat in patterns:
        m = pat.match(basename)
        if not m:
            continue
        deadline_tag = m.groupdict().get("deadline_tag") or "deadline_ratio_1.0"
        dmatch = re.match(r"deadline_(ratio|value)_(.+)", deadline_tag)
        if not dmatch:
            raise ValueError(f"Cannot parse deadline tag from {basename}")
        budget_cost_tag = m.groupdict().get("budget_cost_tag") or "budget_cost_token_cost_mean"
        budget_cost_mode = budget_cost_tag[len("budget_cost_"):]
        idx = m.groupdict().get("idx")
        dag_start = int(idx) if idx is not None else int(m.group("start"))
        dag_end = int(idx) if idx is not None else int(m.group("end"))
        if m.groupdict().get("bval") is not None:
            budget_input_type = "value"
            budget_axis_value = float(m.group("bval"))
            budget_ratio = None
        else:
            budget_input_type = "ratio"
            budget_axis_value = float(m.group("br"))
            budget_ratio = budget_axis_value
        return {
            "budget_input_type": budget_input_type,
            "budget_axis_value": budget_axis_value,
            "budget_ratio": budget_ratio,
            "deadline_tag": deadline_tag,
            "deadline_mode": dmatch.group(1),
            "deadline_axis_value": float(dmatch.group(2)),
            "budget_cost_tag": budget_cost_tag,
            "budget_cost_mode": budget_cost_mode,
            "mode_tag": m.groupdict().get("mode_tag") or "all",
            "shard_type": m.group("stype"),
            "dag_start": dag_start,
            "dag_end": dag_end,
        }
    raise ValueError(f"Unrecognized shard filename: {basename}")


def discover_shards(shard_dir: str) -> dict:
    """
    English note,English note budget_ratio English note.

    English note: {budget_ratio: [shard_path, ...]}
    """
    pattern = os.path.join(shard_dir, "raw_budget_*_dag*.jsonl")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No shard files found matching {pattern}\n"
            f"Expected format: raw_budget_<ratio>_dag<start>-<end>.jsonl"
        )

    groups = defaultdict(list)
    for fp in files:
        basename = os.path.basename(fp)
        # English note: raw_budget_1.0_dag0-999.jsonl
        m = re.match(r"raw_budget_(.+?)_dag(\d+)-(\d+)\.jsonl", basename)
        if not m:
            print(f"  WARNING: Skipping unrecognized file: {basename}")
            continue
        budget_ratio = float(m.group(1))
        dag_start = int(m.group(2))
        dag_end = int(m.group(3))
        groups[budget_ratio].append({
            "path": fp,
            "dag_start": dag_start,
            "dag_end": dag_end,
            "basename": basename,
        })

    # English note budget_ratio English note dag_start English note
    for br in groups:
        groups[br].sort(key=lambda x: x["dag_start"])

    return dict(groups)


def discover_shards_generic(shard_dir: str, shard_type: str = "auto") -> tuple[dict, str]:
    """
    English note,English note budget_ratio English note,English note dag/sel English note.

    shard_type:
      - "dag": English note raw_budget_*_dag*.jsonl
      - "sel": English note raw_budget_*_sel*.jsonl
      - "auto": English note;English note dag English note sel,English note

    English note:
      (groups, resolved_type)
    """
    if shard_type not in {"auto", "dag", "sel"}:
        raise ValueError(f"Invalid shard_type: {shard_type}")

    pattern_dag = os.path.join(shard_dir, "raw_budget_*_dag*.jsonl")
    pattern_sel = os.path.join(shard_dir, "raw_budget_*_sel*.jsonl")

    files_dag = sorted(glob.glob(pattern_dag))
    files_sel = sorted(glob.glob(pattern_sel))

    if shard_type == "dag":
        files = files_dag
        resolved_type = "dag"
    elif shard_type == "sel":
        if files_sel:
            files = files_sel
            resolved_type = "sel"
        else:
            files = files_dag
            resolved_type = "dag"
    else:
        has_dag = len(files_dag) > 0
        has_sel = len(files_sel) > 0
        if has_dag and has_sel:
            raise RuntimeError(
                "Both dag and sel shard files are present. "
                "Please set --shard-type dag or --shard-type sel explicitly."
            )
        if has_dag:
            files = files_dag
            resolved_type = "dag"
        elif has_sel:
            files = files_sel
            resolved_type = "sel"
        else:
            files = []
            resolved_type = "dag"

    if not files:
        expected = "raw_budget_<ratio>_dag<start>-<end>.jsonl" if resolved_type == "dag" \
            else "raw_budget_<ratio>_sel<start>-<end>.jsonl"
        raise FileNotFoundError(
            f"No shard files found for type='{resolved_type}' in {shard_dir}\n"
            f"Expected format: {expected}"
        )

    groups = defaultdict(list)
    if resolved_type == "dag":
        suffix_type = "dag"
    else:
        suffix_type = "sel"

    for fp in files:
        basename = os.path.basename(fp)
        try:
            parsed = _parse_shard_filename(basename)
        except ValueError:
            print(f"  WARNING: Skipping unrecognized file: {basename}")
            continue
        if parsed["shard_type"] != suffix_type:
            continue
        group_key = (
            parsed["budget_cost_mode"],
            parsed["deadline_tag"],
            parsed["budget_input_type"],
            parsed["budget_axis_value"],
        )
        groups[group_key].append({
            "path": fp,
            "dag_start": parsed["dag_start"],
            "dag_end": parsed["dag_end"],
            "basename": basename,
            "deadline_tag": parsed["deadline_tag"],
            "deadline_mode": parsed["deadline_mode"],
            "deadline_axis_value": parsed["deadline_axis_value"],
            "budget_input_type": parsed["budget_input_type"],
            "budget_axis_value": parsed["budget_axis_value"],
            "budget_cost_mode": parsed["budget_cost_mode"],
            "budget_ratio": parsed["budget_ratio"],
        })

    for key in groups:
        groups[key].sort(key=lambda x: x["dag_start"])

    return dict(groups), resolved_type


def verify_coverage(shards: list, n_dags: int = None, expected_indices: list[int] | None = None) -> dict:
    """
    English note DAG English note,English note,English note.

    English note.
    """
    report = {"gaps": [], "overlaps": [], "covered_range": None}

    if not shards:
        report["error"] = "No shards"
        return report

    if expected_indices is not None:
        observed_ids = []
        for s in shards:
            if s["dag_start"] != s["dag_end"]:
                report["overlaps"].append(
                    f"Shard {s['basename']} is not a single-DAG atomic output: "
                    f"[{s['dag_start']}, {s['dag_end']}]"
                )
            observed_ids.extend(range(s["dag_start"], s["dag_end"] + 1))

        expected_sorted = sorted(int(x) for x in expected_indices)
        observed_sorted = sorted(observed_ids)
        report["covered_range"] = (
            f"{len(observed_sorted)} DAG ids "
            f"(min={observed_sorted[0] if observed_sorted else 'NA'}, "
            f"max={observed_sorted[-1] if observed_sorted else 'NA'})"
        )

        expected_set = set(expected_sorted)
        observed_set = set(observed_sorted)
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        duplicates = sorted({x for x in observed_sorted if observed_sorted.count(x) > 1})

        if missing:
            preview = missing[:10]
            suffix = " ..." if len(missing) > 10 else ""
            report["gaps"].append(f"Missing selected DAG ids: {preview}{suffix}")
        if extra:
            preview = extra[:10]
            suffix = " ..." if len(extra) > 10 else ""
            report["overlaps"].append(f"Unexpected DAG ids present: {preview}{suffix}")
        if duplicates:
            preview = duplicates[:10]
            suffix = " ..." if len(duplicates) > 10 else ""
            report["overlaps"].append(f"Duplicate DAG ids present: {preview}{suffix}")

        report["ok"] = len(report["gaps"]) == 0 and len(report["overlaps"]) == 0
        return report

    # English note
    prev_end = shards[0]["dag_start"]
    if prev_end != 0:
        report["gaps"].append(f"Missing DAGs [0, {prev_end})")

    for s in shards:
        if s["dag_start"] > prev_end:
            report["gaps"].append(f"Missing DAGs [{prev_end}, {s['dag_start']})")
        elif s["dag_start"] < prev_end:
            report["overlaps"].append(
                f"Overlap: {s['basename']} starts at {s['dag_start']} "
                f"but previous shard ended at {prev_end}"
            )
        prev_end = s["dag_end"] + 1  # dag_end English note inclusive

    total_end = shards[-1]["dag_end"] + 1
    report["covered_range"] = f"[0, {total_end})"

    if n_dags is not None and total_end < n_dags:
        report["gaps"].append(f"Missing DAGs [{total_end}, {n_dags})")
    if n_dags is not None and total_end > n_dags:
        report["overlaps"].append(
            f"Shards cover {total_end} DAGs but expected {n_dags}"
        )

    report["ok"] = len(report["gaps"]) == 0 and len(report["overlaps"]) == 0
    return report


def merge_budget_ratio_line(shards: list, output_path: str) -> int:
    """English note budget_ratio English note,English note."""
    total_lines = 0
    with open(output_path, "w") as f_out:
        for s in shards:
            with open(s["path"], "r") as f_in:
                for line in f_in:
                    f_out.write(line)
                    total_lines += 1
    return total_lines


def merge_budget_ratio_fast(shards: list, output_path: str, buffer_size: int) -> int:
    """English note,English note Python English note."""
    total_lines = 0
    with open(output_path, "wb") as f_out:
        for s in shards:
            file_lines = 0
            saw_bytes = False
            last_byte = b"\n"
            with open(s["path"], "rb") as f_in:
                while True:
                    chunk = f_in.read(buffer_size)
                    if not chunk:
                        break
                    saw_bytes = True
                    last_byte = chunk[-1:]
                    file_lines += chunk.count(b"\n")
                    f_out.write(chunk)
            if saw_bytes and last_byte != b"\n":
                file_lines += 1
            total_lines += file_lines
    return total_lines


def merge_budget_ratio(
    shards: list,
    output_path: str,
    copy_mode: str = "line",
    buffer_size: int = 16 * 1024 * 1024,
) -> int:
    if copy_mode == "fast":
        return merge_budget_ratio_fast(shards, output_path, buffer_size)
    return merge_budget_ratio_line(shards, output_path)


def build_merge_tasks(grouped_by_budget: dict, output_dir: str, sample_meta: dict | None) -> list[dict]:
    tasks = []
    for budget_key in sorted(grouped_by_budget.keys(), key=lambda x: (x[0], x[1])):
        budget_input_type, budget_axis_value = budget_key
        budget_dir = _format_budget_dir(budget_input_type, budget_axis_value)
        for budget_cost_mode, deadline_tag, shards in sorted(
            grouped_by_budget[budget_key],
            key=lambda item: (item[0], item[2][0]["deadline_mode"], item[2][0]["deadline_axis_value"]),
        ):
            first_shard = shards[0]
            budget_cost_dir = _format_budget_cost_dir(budget_cost_mode)
            deadline_dir = _format_deadline_dir(
                first_shard["deadline_mode"], first_shard["deadline_axis_value"]
            )
            merged_subdir = os.path.join(output_dir, budget_dir, budget_cost_dir, deadline_dir)
            tasks.append(
                {
                    "budget_input_type": budget_input_type,
                    "budget_axis_value": budget_axis_value,
                    "br": budget_axis_value,
                    "budget_cost_mode": budget_cost_mode,
                    "deadline_tag": deadline_tag,
                    "shards": shards,
                    "first_shard": first_shard,
                    "merged_subdir": merged_subdir,
                    "sample_meta": sample_meta,
                }
            )
    return tasks


def process_merge_task(
    task: dict,
    n_dags: int | None,
    n_exec_trials: int | None,
    expected_methods: int | None,
    expected_indices: list[int] | None,
    copy_mode: str,
    buffer_size: int,
    shard_dir: str,
) -> dict:
    budget_input_type = task["budget_input_type"]
    budget_axis_value = task["budget_axis_value"]
    br = task["br"]
    budget_cost_mode = task["budget_cost_mode"]
    deadline_tag = task["deadline_tag"]
    shards = task["shards"]
    first_shard = task["first_shard"]
    merged_subdir = task["merged_subdir"]
    sample_meta = task["sample_meta"]
    logs = []
    group_ok = True

    os.makedirs(merged_subdir, exist_ok=True)

    if sample_meta:
        params = {
            "dag_pool_path": sample_meta.get("dag_pool_path"),
            "budget_input_type": budget_input_type,
            "budget_axis_value": budget_axis_value,
            "budget_ratios": [br] if budget_input_type == "ratio" else [],
            "budget_values": [budget_axis_value] if budget_input_type == "value" else [],
            "budget_ratio": br,
            "budget_value": budget_axis_value if budget_input_type == "value" else None,
            "budget_cost_mode": budget_cost_mode,
            "n_exec_trials": sample_meta.get("n_exec_trials"),
            "seed": sample_meta.get("seed"),
            "latency_dist": sample_meta.get("latency_dist"),
            "n_dags": n_dags,
            "merge_source": shard_dir,
            "deadline_mode": first_shard["deadline_mode"],
            "deadline_axis_value": first_shard["deadline_axis_value"],
            "deadline_tag": deadline_tag,
            "n_methods": expected_methods,
            "method_names": sample_meta.get("method_names"),
            "mode_tag": sample_meta.get("mode_tag"),
            "atomic_dag_outputs": sample_meta.get("atomic_dag_outputs", False),
        }
        params_path = os.path.join(merged_subdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2, default=str)
        logs.append(f"Params snapshot: {params_path}")

    logs.append(f"Deadline {deadline_tag} | Budget {budget_input_type} {budget_axis_value} | Budget cost {budget_cost_mode}:")
    logs.append(f"  Shards: {len(shards)}")

    coverage = verify_coverage(shards, n_dags, expected_indices=expected_indices)
    logs.append(f"  Coverage: {coverage['covered_range']}")
    if not coverage["ok"]:
        group_ok = False
        for g in coverage["gaps"]:
            logs.append(f"  ⚠ GAP: {g}")
        for o in coverage["overlaps"]:
            logs.append(f"  ⚠ OVERLAP: {o}")

    output_path = os.path.join(merged_subdir, _format_raw_budget_name(budget_input_type, budget_axis_value))
    n_lines = merge_budget_ratio(shards, output_path, copy_mode=copy_mode, buffer_size=buffer_size)
    fsize_mb = os.path.getsize(output_path) / (1024 * 1024)
    logs.append(f"  Merged: {n_lines:,} lines → {output_path} ({fsize_mb:.1f} MB)")

    if n_dags is not None and n_exec_trials is not None and expected_methods is not None:
        expected = n_dags * n_exec_trials * expected_methods
        if n_lines != expected:
            logs.append(f"  ⚠ Expected {expected:,} lines but got {n_lines:,}")
            group_ok = False
        else:
            logs.append(f"  ✓ Line count matches expected ({expected:,})")

    report_key = f"{budget_cost_mode}|{deadline_tag}|budget_{budget_input_type}_{budget_axis_value}"
    report_value = {
        "budget_cost_mode": budget_cost_mode,
        "budget_input_type": budget_input_type,
        "budget_axis_value": budget_axis_value,
        "deadline_tag": deadline_tag,
        "deadline_mode": first_shard["deadline_mode"],
        "deadline_axis_value": first_shard["deadline_axis_value"],
        "budget_ratio": br,
        "budget_value": budget_axis_value if budget_input_type == "value" else None,
        "n_shards": len(shards),
        "n_lines": n_lines,
        "coverage": coverage,
        "output_file": output_path,
        "copy_mode": copy_mode,
    }
    return {
        "ok": group_ok,
        "logs": logs,
        "report_key": report_key,
        "report_value": report_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge simulation shard files by budget_ratio"
    )
    parser.add_argument("--shard_dir", type=str, default="results/shards",
                        help="Directory containing shard JSONL files")
    parser.add_argument("--output_dir", type=str, default="results/merged",
                        help="Output directory for merged files")
    parser.add_argument("--n_dags", type=int, default=None,
                        help="Expected total DAG count (for validation)")
    parser.add_argument("--n_exec_trials", type=int, default=None,
                        help="Expected exec trials (for line count validation)")
    parser.add_argument(
        "--shard-type",
        type=str,
        default="auto",
        choices=["auto", "dag", "sel"],
        help="Shard filename type: auto | dag | sel (default: auto)",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="line",
        choices=["line", "fast"],
        help="Copy implementation: line preserves old behavior; fast uses large binary chunks",
    )
    parser.add_argument(
        "--copy-buffer-mb",
        type=int,
        default=16,
        help="Buffer size in MiB for --copy-mode fast",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent (deadline, budget) groups to merge concurrently",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.copy_buffer_mb < 1:
        raise ValueError("--copy-buffer-mb must be >= 1")
    copy_buffer_size = args.copy_buffer_mb * 1024 * 1024

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Shard directory: {args.shard_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Copy mode: {args.copy_mode} (buffer={args.copy_buffer_mb} MiB)")
    print(f"Merge jobs: {args.jobs}")
    print()

    # English note
    groups, resolved_type = discover_shards_generic(args.shard_dir, args.shard_type)
    print(f"Found {sum(len(v) for v in groups.values())} shard files "
            f"across {len(groups)} (deadline, budget) groups (type={resolved_type})")
    print()

    merge_report = {
        "shard_dir": args.shard_dir,
        "output_dir": args.output_dir,
        "shard_type": resolved_type,
        "groups": {},
    }
    all_ok = True

    meta_files = sorted(glob.glob(os.path.join(args.shard_dir, "shard_meta_*.json")))
    if not meta_files:
        meta_files = sorted(glob.glob(os.path.join(args.shard_dir, "atomic_meta_*.json")))
    sample_meta = None
    if meta_files:
        with open(meta_files[0]) as f:
            sample_meta = json.load(f)
    expected_methods = sample_meta.get("n_methods") if sample_meta else None
    expected_indices = None
    if sample_meta and sample_meta.get("dag_indices_file"):
        dag_indices_path = sample_meta["dag_indices_file"]
        if os.path.exists(dag_indices_path):
            with open(dag_indices_path) as f:
                idx_data = json.load(f)
            expected_indices = list(idx_data.get("dag_indices", []))

    grouped_by_budget = defaultdict(list)
    for (budget_cost_mode, deadline_tag, budget_input_type, budget_axis_value), shards in groups.items():
        grouped_by_budget[(budget_input_type, budget_axis_value)].append((budget_cost_mode, deadline_tag, shards))

    tasks = build_merge_tasks(grouped_by_budget, args.output_dir, sample_meta)

    def run_task(task):
        return process_merge_task(
            task,
            n_dags=args.n_dags,
            n_exec_trials=args.n_exec_trials,
            expected_methods=expected_methods,
            expected_indices=expected_indices,
            copy_mode=args.copy_mode,
            buffer_size=copy_buffer_size,
            shard_dir=args.shard_dir,
        )

    if args.jobs == 1:
        results = [run_task(task) for task in tasks]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_task = {executor.submit(run_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                results.append(future.result())

    for result in results:
        for line in result["logs"]:
            print(line)
        print()
        if not result["ok"]:
            all_ok = False
        merge_report["groups"][result["report_key"]] = result["report_value"]

    # English note
    report_path = os.path.join(args.output_dir, "merge_report.json")
    merge_report["all_ok"] = all_ok
    with open(report_path, "w") as f:
        json.dump(merge_report, f, indent=2, default=str)
    print(f"Merge report: {report_path}")

    if all_ok:
        print("\n✓ All merges completed successfully.")
        print(f"\nNext step: python analyze.py {args.output_dir}/budget_*/budget_cost_*/deadline_*")
    else:
        print("\n⚠ Some issues detected. Check the report above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
