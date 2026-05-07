"""
English note br=0.3/0.5/1.0 English note 2000 English note DAG,
English note.English note dag_indices English note br=2.0/5.0/10.0 English note.

English note:
  1. English note br=0.3/0.5/1.0 English note shard English note,English note br English note dag_idx
  2. English note
  3. English note 2000 English note dag_idx(English note seed English note)
  4. English note dag_idx English note,English note
  5. English note dag_indices English note

English note:
    python subset_dags.py [--n_sample 2000] [--seed 42]
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from collections import defaultdict


def find_complete_dags(shard_pattern: str, n_exec_trials: int = 1000, n_methods: int = 10) -> set:
    """English note shard English note,English note dag_idx."""
    expected_per_dag = n_exec_trials * n_methods
    dag_counts = defaultdict(int)

    files = sorted(glob.glob(shard_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {shard_pattern}")

    for fp in files:
        print(f"  Scanning {os.path.basename(fp)} ...")
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    dag_counts[rec["dag_idx"]] += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    completed = {idx for idx, count in dag_counts.items() if count == expected_per_dag}
    return completed


def extract_subset(shard_pattern: str, selected_dags: set, output_path: str):
    """English note shard English note dag_idx English note,English note."""
    files = sorted(glob.glob(shard_pattern))
    n_written = 0

    with open(output_path, "w") as f_out:
        for fp in files:
            print(f"  Filtering {os.path.basename(fp)} ...")
            with open(fp, "r") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec["dag_idx"] in selected_dags:
                            f_out.write(line + "\n")
                            n_written += 1
                    except (json.JSONDecodeError, KeyError):
                        pass

    return n_written


def main():
    parser = argparse.ArgumentParser(description="Subset DAGs from completed br=0.3/0.5/1.0 data")
    parser.add_argument("--n_sample", type=int, default=2000,
                        help="Number of DAGs to sample (default: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for DAG selection (default: 42)")
    parser.add_argument("--shard_dir", type=str, default="results/shards",
                        help="Directory containing shard JSONL files")
    parser.add_argument("--output_dir", type=str, default="results/subset_2k",
                        help="Output directory for subset data")
    parser.add_argument("--n_exec_trials", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    budget_ratios = [0.3, 0.5, 1.0]

    # ── Step 1: English note br English note DAG ──
    print("=" * 60)
    print("Step 1: Scanning completed DAGs per budget_ratio")
    print("=" * 60)

    completed_per_br = {}
    for br in budget_ratios:
        pattern = os.path.join(args.shard_dir, f"raw_budget_{br}_dag*.jsonl")
        print(f"\nbr={br}:")
        completed = find_complete_dags(pattern, n_exec_trials=args.n_exec_trials)
        completed_per_br[br] = completed
        print(f"  Complete DAGs: {len(completed)}")

    # ── Step 2: English note ──
    print("\n" + "=" * 60)
    print("Step 2: Finding common DAGs across all budget_ratios")
    print("=" * 60)

    common_dags = completed_per_br[budget_ratios[0]]
    for br in budget_ratios[1:]:
        common_dags = common_dags & completed_per_br[br]

    common_dags_sorted = sorted(common_dags)
    print(f"\nCommon complete DAGs: {len(common_dags_sorted)}")

    if len(common_dags_sorted) < args.n_sample:
        print(f"\nWARNING: Only {len(common_dags_sorted)} common DAGs available, "
              f"requested {args.n_sample}. Using all available.")
        selected = common_dags_sorted
    else:
        # ── Step 3: English note n_sample English note ──
        rng = np.random.default_rng(seed=args.seed)
        selected_indices = rng.choice(len(common_dags_sorted), size=args.n_sample, replace=False)
        selected_indices.sort()
        selected = [common_dags_sorted[i] for i in selected_indices]

    selected_set = set(selected)

    print(f"Selected {len(selected)} DAGs (seed={args.seed})")
    print(f"  Range: [{min(selected)}, {max(selected)}]")
    print(f"  First 10: {selected[:10]}")

    # ── English note ──
    indices_path = os.path.join(args.output_dir, "selected_dag_indices.json")
    with open(indices_path, "w") as f:
        json.dump({
            "n_sample": len(selected),
            "seed": args.seed,
            "n_common_available": len(common_dags_sorted),
            "dag_indices": selected,
        }, f, indent=2)
    print(f"  Saved: {indices_path}")

    # ── Step 4: English note ──
    print("\n" + "=" * 60)
    print("Step 3: Extracting subset data for br=0.3/0.5/1.0")
    print("=" * 60)

    for br in budget_ratios:
        print(f"\nbr={br}:")
        pattern = os.path.join(args.shard_dir, f"raw_budget_{br}_dag*.jsonl")
        output_path = os.path.join(args.output_dir, f"raw_budget_{br}.jsonl")
        n_written = extract_subset(pattern, selected_set, output_path)

        expected = len(selected) * args.n_exec_trials * 10  # 10 methods
        fsize_mb = os.path.getsize(output_path) / (1024 * 1024)
        status = "✓" if n_written == expected else "✗"
        print(f"  {status} Written: {n_written:,} records (expected: {expected:,}), "
              f"size: {fsize_mb:.1f} MB")

    # ── English note ──
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  DAGs selected:    {len(selected)}")
    print(f"  Budget ratios:    {budget_ratios}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Indices file:     {indices_path}")
    print(f"  Data files:       raw_budget_{{0.3,0.5,1.0}}.jsonl")
    print()
    print(f"  Next: run br=2.0/5.0/10.0 with --dag_indices_file {indices_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
