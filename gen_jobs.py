"""
English note.

English note (budget_ratio x DAG English note) English note shard English note,
English note QS English note.

English note budget_ratio English note shard English note,English note budget_ratio
English note(English note br English note DAG English note br English note 20 English note).

English note:
  1. English note GPU English note(English note GPU English note kill)
  2. English note CPU English note
  3. English note kill GPU English note

English note:
    python gen_jobs.py \
        --n_dags 10000 \
        --dags_per_shard 1000 \
        --budget_ratios 0.3 0.5 1.0 2.0 5.0 10.0 \
        --n_exec_trials 1000 \
        [--shard_size_map '2.0:120,5.0:50,10.0:50'] \
        [--output_dir results/shards] \
        [--dag_pool_path pregenerated_dags/dag_pool.jsonl]
        [--gpu_script /path/to/optional_gpu_keepalive.py]
        [--no_gpu]  # English note GPU English note

English note:
    jobs.sh        - English note shard English note shell English note(English note)
    jobs_list.txt  - English note(English note QS English note)

English note(English note:br<=1.0 English note 1000 DAGs/shard, br=2.0 English note 120, br>=5.0 English note 50):
    python gen_jobs.py --shard_size_map '2.0:120,5.0:50,10.0:50'
    # br=0.3: 10 shards, br=2.0: 84 shards, br=10.0: 200 shards, ...
"""

import os
import sys
import argparse
import math
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.run_experiment import get_method_names, _normalize_baseline_method_set

# GPU English note
DEFAULT_GPU_SCRIPT = ""


def load_dag_indices(path: str) -> list[int]:
    """Load a selected-DAG index file across historical JSON shapes."""
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_indices = data
    elif isinstance(data, dict):
        raw_indices = None
        for key in ("dag_indices", "selected_dag_indices", "selected_indices", "indices"):
            if key in data:
                raw_indices = data[key]
                break
        if raw_indices is None:
            raise KeyError(
                "DAG indices file must contain one of keys: "
                "dag_indices, selected_dag_indices, selected_indices, indices. "
                f"Found keys: {sorted(data.keys())}"
            )
    else:
        raise TypeError(
            f"DAG indices file must be a JSON object or list, got {type(data).__name__}"
        )

    if not isinstance(raw_indices, list):
        raise TypeError(f"DAG indices must be a list, got {type(raw_indices).__name__}")
    return [int(idx) for idx in raw_indices]


def _resolve_shard_size(br: float, default_size: int, shard_size_map: dict) -> int:
    """English note budget_ratio English note shard English note."""
    # English note
    if br in shard_size_map:
        return shard_size_map[br]
    # English note(English note)
    for k, v in shard_size_map.items():
        if abs(float(k) - float(br)) < 1e-9:
            return v
    return default_size


def _wrap_gpu(shard_cmd: str, python_cmd: str, gpu_script: str, no_gpu: bool) -> str:
    """English note GPU English note."""
    if no_gpu or not gpu_script:
        return shard_cmd
    return (
        f"bash -c '"
        f"{python_cmd} {gpu_script} --ratio 0.3 --iters 1000000000 &"
        f" GPU_PID=$!; "
        f"echo \"[GPU keepalive] started pid=$GPU_PID\"; "
        f"{shard_cmd}; "
        f"SHARD_EXIT=$?; "
        f"kill $GPU_PID 2>/dev/null; wait $GPU_PID 2>/dev/null; "
        f"echo \"[GPU keepalive] stopped\"; "
        f"exit $SHARD_EXIT"
        f"'"
    )


def _deadline_specs(
    deadline_values: list[float] | None = None,
    deadline_ratios: list[float] | None = None,
) -> list[tuple[str, float]]:
    if deadline_values:
        return [("value", float(v)) for v in deadline_values]
    ratios = deadline_ratios if deadline_ratios else [1.0]
    return [("ratio", float(v)) for v in ratios]


def _budget_specs(
    budget_values: list[float] | None = None,
    budget_ratios: list[float] | None = None,
) -> list[tuple[str, float]]:
    if budget_values:
        return [("value", float(v)) for v in budget_values]
    ratios = budget_ratios if budget_ratios else [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    return [("ratio", float(v)) for v in ratios]


def _parse_budget_deadline_values_map(text: str) -> list[tuple[float, list[float]]]:
    """
    Parse absolute budget-to-deadline values.

    Format:
        "70000:200,500,1000;200000:500,1000,3000"
    """
    if not text:
        return []

    specs = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                "Invalid --budget_deadline_values_map entry "
                f"{chunk!r}; expected '<budget>:<deadline1>,<deadline2>,...'"
            )
        budget_text, deadlines_text = chunk.split(":", 1)
        budget = float(budget_text.strip())
        deadlines = [
            float(item.strip())
            for item in deadlines_text.replace("|", ",").split(",")
            if item.strip()
        ]
        if not deadlines:
            raise ValueError(f"No deadlines provided for budget {budget}")
        specs.append((budget, deadlines))
    return specs


def _budget_deadline_specs(
    budget_deadline_values_map: list[tuple[float, list[float]]] | None = None,
    budget_values: list[float] | None = None,
    budget_ratios: list[float] | None = None,
    deadline_values: list[float] | None = None,
    deadline_ratios: list[float] | None = None,
) -> list[tuple[str, float, str, float]]:
    if budget_deadline_values_map:
        out = []
        for budget_value, deadlines in budget_deadline_values_map:
            for deadline_value in deadlines:
                out.append(("value", float(deadline_value), "value", float(budget_value)))
        return out

    out = []
    for deadline_mode, deadline_axis_value in _deadline_specs(deadline_values, deadline_ratios):
        for budget_mode, budget_axis_value in _budget_specs(budget_values, budget_ratios):
            out.append((deadline_mode, deadline_axis_value, budget_mode, budget_axis_value))
    return out


def gen_jobs(
    n_dags: int,
    dags_per_shard: int,
    budget_ratios: list | None,
    n_exec_trials: int,
    dag_pool_path: str,
    output_dir: str,
    seed: int,
    latency_dist: str,
    n_workers: int,
    python_cmd: str,
    gpu_script: str,
    no_gpu: bool,
    budget_cost_mode: str = "lat_mean",
    budget_values: list[float] | None = None,
    budget_deadline_values_map: list[tuple[float, list[float]]] | None = None,
    shard_size_map: dict = None,
    dag_indices_file: str = None,
    deadline_values: list[float] | None = None,
    deadline_ratios: list[float] | None = None,
    only_loader: bool = False,
    baseline_only: bool = False,
    baseline_method_set: str = "all",
):
    """English note shard English note.

    English note:
      1. English note(English note):English note dag_start/dag_end English note
      2. English note(dag_indices_file):English note
    """
    if shard_size_map is None:
        shard_size_map = {}
    if only_loader and baseline_only:
        raise ValueError("only_loader and baseline_only cannot both be True")
    baseline_method_set = _normalize_baseline_method_set(baseline_method_set)

    method_args = ""
    if only_loader:
        method_args = " --only_loader"
    elif baseline_only:
        method_args = " --baseline_only"
    if baseline_method_set != "all":
        method_args += f" --baseline_method_set {baseline_method_set}"

    commands = []
    job_info = []  # (deadline_mode, deadline_value, budget_mode, budget_value, shard_size, n_shards)

    # English note,English note
    if dag_indices_file:
        n_indices = len(load_dag_indices(dag_indices_file))
        if n_dags is not None and n_dags > 0:
            n_indices = min(n_indices, n_dags)
    else:
        n_indices = None

    for deadline_mode, deadline_axis_value, budget_mode, budget_axis_value in _budget_deadline_specs(
        budget_deadline_values_map=budget_deadline_values_map,
        budget_values=budget_values,
        budget_ratios=budget_ratios,
        deadline_values=deadline_values,
        deadline_ratios=deadline_ratios,
    ):
        deadline_args = (
            f"--deadline_value {deadline_axis_value}"
            if deadline_mode == "value"
            else f"--deadline_ratio {deadline_axis_value}"
        )

        budget_args = (
            f"--budget_value {budget_axis_value}"
            if budget_mode == "value"
            else f"--budget_ratio {budget_axis_value}"
        )
        shard_size = _resolve_shard_size(budget_axis_value, dags_per_shard, shard_size_map)

        if dag_indices_file:
            n_shards = math.ceil(n_indices / shard_size)
            job_info.append((deadline_mode, deadline_axis_value, budget_mode, budget_axis_value, shard_size, n_shards))

            for shard_idx in range(n_shards):
                s_start = shard_idx * shard_size
                s_end = min(s_start + shard_size, n_indices)

                shard_cmd = (
                    f"{python_cmd} run_shard.py "
                    f"--dag_pool_path {dag_pool_path} "
                    f"{budget_args} "
                    f"--budget_cost_mode {budget_cost_mode} "
                    f"{deadline_args} "
                    f"--dag_indices_file {dag_indices_file} "
                    f"--dag_shard_start {s_start} --dag_shard_end {s_end} "
                    f"--n_exec_trials {n_exec_trials} "
                    f"--output_dir {output_dir} "
                    f"--seed {seed} "
                    f"--latency_dist {latency_dist} "
                    f"--n_workers {n_workers}"
                    f"{method_args}"
                )
                commands.append(_wrap_gpu(shard_cmd, python_cmd, gpu_script, no_gpu))
        else:
            n_shards = math.ceil(n_dags / shard_size)
            job_info.append((deadline_mode, deadline_axis_value, budget_mode, budget_axis_value, shard_size, n_shards))

            for shard_idx in range(n_shards):
                dag_start = shard_idx * shard_size
                dag_end = min(dag_start + shard_size, n_dags)

                shard_cmd = (
                    f"{python_cmd} run_shard.py "
                    f"--dag_pool_path {dag_pool_path} "
                    f"{budget_args} "
                    f"--budget_cost_mode {budget_cost_mode} "
                    f"{deadline_args} "
                    f"--dag_start {dag_start} --dag_end {dag_end} "
                    f"--n_exec_trials {n_exec_trials} "
                    f"--output_dir {output_dir} "
                    f"--seed {seed} "
                    f"--latency_dist {latency_dist} "
                    f"--n_workers {n_workers}"
                    f"{method_args}"
                )
                commands.append(_wrap_gpu(shard_cmd, python_cmd, gpu_script, no_gpu))

    total_jobs = len(commands)
    return commands, job_info, total_jobs


def _parse_shard_size_map(s: str) -> dict:
    """English note 'br1:size1,br2:size2,...' English note shard size English note.
    
    English note: '2.0:120,5.0:50,10.0:50'
    """
    if not s:
        return {}
    result = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split(":")
        result[float(k.strip())] = int(v.strip())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate parallel shard job commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
English note:
  # English note 1000 DAGs/shard (English note)
  python gen_jobs.py --n_dags 10000 --dags_per_shard 1000

  # English note budget_ratio English note shard
  python gen_jobs.py --n_dags 10000 --dags_per_shard 1000 \\
      --shard_size_map '2.0:120,5.0:50,10.0:50'

  # English note budget_ratio English note shard
  python gen_jobs.py --n_dags 10000 --dags_per_shard 50 \\
      --budget_ratios 2.0 5.0 10.0 \\
      --shard_size_map '2.0:120,5.0:50,10.0:50'
"""
    )
    parser.add_argument("--n_dags", type=int, default=10000,
                        help="Total number of DAGs in pool (default: 10000)")
    parser.add_argument("--dags_per_shard", type=int, default=1000,
                        help="Default DAGs per shard (default: 1000)")
    parser.add_argument("--shard_size_map", type=str, default="",
                        help="Per-budget-ratio shard sizes, format: "
                             "'br1:size1,br2:size2,...' "
                             "e.g. '2.0:120,5.0:50,10.0:50'. "
                             "Unspecified ratios use --dags_per_shard.")
    parser.add_argument("--dag_indices_file", type=str, default=None,
                        help="JSON file with selected DAG indices. "
                             "When set, shards use --dag_indices_file + "
                             "--dag_shard_start/end instead of --dag_start/end.")
    parser.add_argument("--budget_ratios", type=float, nargs="+",
                        default=None,
                        help="Budget ratios (default: 0.3 0.5 1.0 2.0 5.0 10.0)")
    parser.add_argument("--budget_value", type=float, default=None,
                        help="Absolute work budget for every generated shard command")
    parser.add_argument("--budget_values", type=float, nargs="+", default=None,
                        help="Absolute work budget sweep values")
    parser.add_argument(
        "--budget_deadline_values_map",
        type=str,
        default="",
        help=(
            "Absolute budget-specific deadline grid. Format: "
            "'budget:deadline1,deadline2;budget2:deadline1,deadline2'. "
            "When set, overrides --budget_values/--budget_ratios and "
            "--deadline_values/--deadline_ratios."
        ),
    )
    parser.add_argument(
        "--budget_cost_mode",
        type=str,
        default="token_cost_mean",
        choices=["token_cost_mean", "output_money_cost_mean", "cost_mean", "lat_mean"],
        help="Work-budget kappa_v type passed to run_shard.py.",
    )
    parser.add_argument("--n_exec_trials", type=int, default=1000,
                        help="Execution trials per DAG (default: 1000)")
    parser.add_argument("--deadline_value", type=float, default=None,
                        help="Absolute deadline for every generated shard command")
    parser.add_argument("--deadline_values", type=float, nargs="+", default=None,
                        help="Absolute deadline sweep values")
    parser.add_argument("--deadline_ratio", type=float, default=1.0,
                        help="Relative deadline ratio for every generated shard command (default: 1.0)")
    parser.add_argument("--deadline_ratios", type=float, nargs="+", default=None,
                        help="Relative deadline ratio sweep values")
    parser.add_argument("--dag_pool_path", type=str,
                        default="pregenerated_dags/dag_pool.jsonl",
                        help="Path to DAG pool JSONL")
    parser.add_argument("--output_dir", type=str, default="results/shards",
                        help="Output directory for shard files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency_dist", type=str, default="lognormal")
    parser.add_argument("--n_workers", type=int, default=0,
                        help="Workers per shard (0=auto)")
    parser.add_argument("--python_cmd", type=str, default="python3",
                        help="Python command (default: python3)")
    parser.add_argument("--gpu_script", type=str, default=DEFAULT_GPU_SCRIPT,
                        help=f"Path to GPU keepalive script (default: {DEFAULT_GPU_SCRIPT})")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU keepalive (for CPU-only queues)")
    parser.add_argument("--only_loader", action="store_true",
                        help="Generate shard commands that run only the primary MC portfolio method")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Generate shard commands that run only baseline methods")
    parser.add_argument(
        "--baseline_method_set",
        type=str,
        default="all",
        choices=["all", "seq", "static"],
        help="Subset of baseline methods to include: all, seq, or static.",
    )
    parser.add_argument("--jobs_list_path", type=str, default="jobs_list.txt",
                        help="Path to write job list (one command per line)")
    parser.add_argument("--jobs_script_path", type=str, default="jobs.sh",
                        help="Path to write executable shell script with all jobs")
    args = parser.parse_args()

    shard_size_map = _parse_shard_size_map(args.shard_size_map)
    budget_deadline_values_map = _parse_budget_deadline_values_map(args.budget_deadline_values_map)

    budget_values = args.budget_values
    budget_ratios = args.budget_ratios
    if budget_values is None and args.budget_value is not None:
        budget_values = [args.budget_value]
    if budget_values is None and budget_ratios is None:
        budget_ratios = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    if budget_deadline_values_map:
        budget_values = None
        budget_ratios = None

    deadline_values = args.deadline_values
    deadline_ratios = args.deadline_ratios
    if deadline_values is None and args.deadline_value is not None:
        deadline_values = [args.deadline_value]
    if deadline_values is None and deadline_ratios is None:
        deadline_ratios = [args.deadline_ratio]
    if budget_deadline_values_map:
        deadline_values = None
        deadline_ratios = None

    commands, job_info, total_jobs = gen_jobs(
        n_dags=args.n_dags,
        dags_per_shard=args.dags_per_shard,
        budget_ratios=budget_ratios,
        budget_values=budget_values,
        budget_deadline_values_map=budget_deadline_values_map,
        n_exec_trials=args.n_exec_trials,
        dag_pool_path=args.dag_pool_path,
        output_dir=args.output_dir,
        seed=args.seed,
        latency_dist=args.latency_dist,
        n_workers=args.n_workers,
        python_cmd=args.python_cmd,
        gpu_script=args.gpu_script,
        no_gpu=args.no_gpu,
        budget_cost_mode=args.budget_cost_mode,
        shard_size_map=shard_size_map,
        dag_indices_file=args.dag_indices_file,
        deadline_values=deadline_values,
        deadline_ratios=deadline_ratios,
        only_loader=args.only_loader,
        baseline_only=args.baseline_only,
        baseline_method_set=args.baseline_method_set,
    )

    jobs_list_path = args.jobs_list_path
    jobs_script_path = args.jobs_script_path
    os.makedirs(os.path.dirname(jobs_list_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(jobs_script_path) or ".", exist_ok=True)

    n_methods = len(get_method_names(
        only_loader=args.only_loader,
        baseline_only=args.baseline_only,
        baseline_method_set=args.baseline_method_set,
    ))

    # English note jobs_list(English note,English note QS English note)
    with open(jobs_list_path, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    # English note jobs.sh(English note bash English note,English note)
    with open(jobs_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated: {total_jobs} shard jobs\n")
        f.write(f"# Shard breakdown per budget_ratio:\n")
        for deadline_mode, deadline_axis_value, budget_mode, budget_axis_value, sz, ns in job_info:
            sims = sz * args.n_exec_trials * n_methods
            f.write(
                f"#   deadline_{deadline_mode}={deadline_axis_value} budget_{budget_mode}={budget_axis_value}: "
                f"{ns} shards x {sz} DAGs = {ns * sz} DAGs, {sims:,} sims/shard\n"
            )
        if not args.no_gpu:
            f.write(f"# GPU keepalive: {args.gpu_script}\n")
        f.write("\n")
        f.write(f"set -e\n")
        f.write(f"mkdir -p {args.output_dir}\n\n")
        for i, cmd in enumerate(commands):
            f.write(f"echo '[Job {i+1}/{total_jobs}] Starting ...'\n")
            f.write(cmd + "\n\n")
    os.chmod(jobs_script_path, 0o755)

    # English note
    total_sims = sum(sz * args.n_exec_trials * n_methods * ns for _, _, _, _, sz, ns in job_info)
    print("=" * 65)
    print("Job generation complete")
    if budget_deadline_values_map:
        print(f"  Budget/deadline map: {budget_deadline_values_map}")
    elif budget_values is not None:
        print(f"  Budget values:     {budget_values}")
    else:
        print(f"  Budget ratios:     {budget_ratios}")
    print(f"  DAGs:              {args.n_dags}")
    print(f"  Default shard:     {args.dags_per_shard} DAGs/shard")
    if shard_size_map:
        print(f"  Custom shards:     {shard_size_map}")
    print()
    print(f"  {'Deadline':>18s}  {'Budget':>18s}  {'Shard Size':>10s}  {'#Shards':>7s}  {'Sims/Shard':>12s}")
    print(f"  {'─'*18}  {'─'*18}  {'─'*10}  {'─'*7}  {'─'*12}")
    for deadline_mode, deadline_axis_value, budget_mode, budget_axis_value, sz, ns in job_info:
        sims = sz * args.n_exec_trials * n_methods
        deadline_label = f"{deadline_mode}={deadline_axis_value}"
        budget_label = f"{budget_mode}={budget_axis_value}"
        print(f"  {deadline_label:>18s}  {budget_label:>18s}  {sz:10d}  {ns:7d}  {sims:12,}")
    print()
    print(f"  Exec trials:       {args.n_exec_trials}")
    print(f"  Budget cost mode:  {args.budget_cost_mode}")
    if budget_deadline_values_map:
        print("  Deadline values:   budget-specific")
    elif deadline_values is not None:
        print(f"  Deadline values:   {deadline_values}")
    else:
        print(f"  Deadline ratios:   {deadline_ratios}")
    print(f"  Total jobs:        {total_jobs}")
    print(f"  Total sims:        {total_sims:,}")
    print(f"  GPU keepalive:     {'OFF' if args.no_gpu else args.gpu_script}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Job list:          {jobs_list_path} ({total_jobs} lines)")
    print(f"  Job script:        {jobs_script_path}")
    print("=" * 65)
    print(f"\nSubmit each line of {jobs_list_path} to QS platform, or run:")
    print(f"  bash {jobs_script_path}")


if __name__ == "__main__":
    main()
