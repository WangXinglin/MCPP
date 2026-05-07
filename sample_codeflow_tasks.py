#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def _task_id(task: dict, fallback_idx: int) -> str:
    if "problem-id" in task:
        return str(task["problem-id"])
    if "task_id" in task:
        return str(task["task_id"])
    return f"idx_{fallback_idx}"


def _load_data(input_path: Path) -> list:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input dataset must be a JSON list")
    if not data:
        raise ValueError("Input dataset is empty")
    return data


def _select_by_indices(data: list, indices: list) -> list:
    total = len(data)
    selected = []
    for idx in indices:
        if not isinstance(idx, int):
            raise ValueError(f"Index must be int, got: {idx}")
        if idx < 0 or idx >= total:
            raise ValueError(f"Index out of range: {idx}, total={total}")
        task = dict(data[idx])
        task["_source_index"] = idx
        selected.append(task)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample a subset of tasks from CodeFlowBench JSON dataset"
    )
    parser.add_argument("--input", required=True, help="Input dataset JSON path")
    parser.add_argument("--output", required=True, help="Output sampled JSON path")
    parser.add_argument("--n", type=int, default=0, help="Number of tasks to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_indices",
        type=str,
        default="",
        help="Optional output JSON path for selected index metadata",
    )
    parser.add_argument(
        "--indices_file",
        type=str,
        default="",
        help="Optional JSON file with preselected indices to restore exact subset",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data = _load_data(input_path)
    total = len(data)

    if args.indices_file:
        with Path(args.indices_file).open("r", encoding="utf-8") as f:
            indices_payload = json.load(f)

        if isinstance(indices_payload, dict):
            if "selected_indices" in indices_payload:
                chosen = list(indices_payload["selected_indices"])
            elif "indices" in indices_payload:
                chosen = list(indices_payload["indices"])
            elif "dag_indices" in indices_payload:
                chosen = list(indices_payload["dag_indices"])
            else:
                raise ValueError("indices_file must contain key 'selected_indices', 'indices', or 'dag_indices'")
        elif isinstance(indices_payload, list):
            chosen = list(indices_payload)
        else:
            raise ValueError("indices_file must be a JSON object or array")

        sampled = _select_by_indices(data, chosen)
        mode = "indices"
    else:
        if args.n <= 0:
            raise ValueError("--n must be > 0 when --indices_file is not provided")

        n = min(args.n, total)
        rng = random.Random(args.seed)
        indices = list(range(total))
        rng.shuffle(indices)
        chosen = sorted(indices[:n])
        sampled = []
        for i in chosen:
            task = dict(data[i])
            task["_source_index"] = i
            sampled.append(task)
        mode = "sample"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    if args.output_indices:
        indices_path = Path(args.output_indices)
        indices_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input": str(input_path),
            "total_tasks": total,
            "selected_count": len(chosen),
            "seed": args.seed,
            "mode": mode,
            "selected_indices": chosen,
            "selected_task_ids": [_task_id(data[i], i) for i in chosen],
        }
        with indices_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    ids_preview = [_task_id(sampled[i], i) for i in range(min(10, len(sampled)))]
    print(f"Input: {input_path}")
    print(f"Total tasks: {total}")
    print(f"Sampled tasks: {len(sampled)}")
    print(f"Seed: {args.seed}")
    print(f"Mode: {mode}")
    print(f"Output: {output_path}")
    if args.output_indices:
        print(f"Indices output: {args.output_indices}")
    if args.indices_file:
        print(f"Indices source: {args.indices_file}")
    print(f"Preview task ids: {ids_preview}")


if __name__ == "__main__":
    main()
