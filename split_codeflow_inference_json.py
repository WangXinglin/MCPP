#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stream-split a large CodeFlow combined inference JSON list into "
            "smaller valid JSON list shards without loading the whole file."
        )
    )
    parser.add_argument("--input", required=True, help="Large combined inference JSON file.")
    parser.add_argument("--output-dir", required=True, help="Directory for shard JSON files.")
    parser.add_argument("--items-per-shard", type=int, required=True, help="Problems per shard.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Only emit problems with original list index >= this value.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=0,
        help="Only emit problems with original list index < this value. 0 means no upper bound.",
    )
    parser.add_argument(
        "--add-source-index",
        action="store_true",
        help="Parse each problem and add _source_index before writing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N top-level problems scanned. 0 disables progress.",
    )
    return parser.parse_args()


def iter_top_level_array_items(path, chunk_size=1024 * 1024):
    in_string = False
    escape = False
    depth = 0
    started_array = False
    item_started = False
    item_chunks = []
    item_index = 0

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            for ch in chunk:
                if not started_array:
                    if ch.isspace():
                        continue
                    if ch != "[":
                        raise ValueError("Input JSON must be a top-level array.")
                    started_array = True
                    continue

                if not item_started:
                    if ch.isspace() or ch == ",":
                        continue
                    if ch == "]":
                        return
                    item_started = True
                    item_chunks = [ch]
                    in_string = ch == '"'
                    escape = False
                    depth = 1 if ch in "[{" else 0
                    if depth == 0 and not in_string:
                        raise ValueError("Expected top-level array items to be JSON objects/arrays.")
                    continue

                item_chunks.append(ch)

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue
                if ch in "[{":
                    depth += 1
                    continue
                if ch in "]}":
                    depth -= 1
                    if depth == 0:
                        yield item_index, "".join(item_chunks)
                        item_index += 1
                        item_started = False
                        item_chunks = []

        if item_started or in_string or depth != 0:
            raise ValueError("Unexpected EOF while reading a top-level array item.")
        if not started_array:
            raise ValueError("Input JSON is empty or not a JSON array.")


class ShardWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.temp_path = None
        self.shard_index = 0
        self.start_idx = None
        self.end_idx = None
        self.count = 0
        self.manifest = []

    def _open(self, source_idx):
        self.start_idx = source_idx
        self.end_idx = source_idx
        self.count = 0
        self.temp_path = self.output_dir / f".shard_{self.shard_index:06d}.json.tmp"
        self.file = open(self.temp_path, "w", encoding="utf-8")
        self.file.write("[\n")

    def write_item(self, source_idx, item_text):
        if self.file is None:
            self._open(source_idx)
        if self.count:
            self.file.write(",\n")
        self.file.write(item_text)
        self.count += 1
        self.end_idx = source_idx + 1

    def close_current(self):
        if self.file is None:
            return None
        self.file.write("\n]\n")
        self.file.close()
        final_path = self.output_dir / (
            f"shard_{self.shard_index:06d}_dag{self.start_idx}-{self.end_idx - 1}.json"
        )
        os.replace(self.temp_path, final_path)
        entry = {
            "shard_index": self.shard_index,
            "start_index": self.start_idx,
            "end_index": self.end_idx,
            "count": self.count,
            "path": str(final_path),
        }
        self.manifest.append(entry)
        self.file = None
        self.temp_path = None
        self.shard_index += 1
        self.start_idx = None
        self.end_idx = None
        self.count = 0
        return entry


def maybe_add_source_index(item_text, source_idx):
    obj = json.loads(item_text)
    if isinstance(obj, dict):
        obj["_source_index"] = source_idx
    return json.dumps(obj, ensure_ascii=False)


def main():
    args = parse_args()
    if args.items_per_shard <= 0:
        raise ValueError("--items-per-shard must be > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.end_index and args.end_index <= args.start_index:
        raise ValueError("--end-index must be > --start-index")

    writer = ShardWriter(args.output_dir)
    scanned = 0
    emitted = 0
    started_at = time.time()

    for source_idx, item_text in iter_top_level_array_items(args.input):
        scanned = source_idx + 1
        if source_idx < args.start_index:
            continue
        if args.end_index and source_idx >= args.end_index:
            break

        if args.add_source_index:
            item_text = maybe_add_source_index(item_text, source_idx)
        writer.write_item(source_idx, item_text)
        emitted += 1

        if writer.count >= args.items_per_shard:
            entry = writer.close_current()
            print(
                f"[split] wrote shard {entry['shard_index']} "
                f"dag{entry['start_index']}-{entry['end_index'] - 1} count={entry['count']}",
                flush=True,
            )

        if args.progress_every and scanned % args.progress_every == 0:
            elapsed = max(time.time() - started_at, 1e-9)
            print(
                f"[split] scanned={scanned} emitted={emitted} rate={scanned / elapsed:.2f}/s",
                file=sys.stderr,
                flush=True,
            )

    last = writer.close_current()
    if last:
        print(
            f"[split] wrote shard {last['shard_index']} "
            f"dag{last['start_index']}-{last['end_index'] - 1} count={last['count']}",
            flush=True,
        )

    manifest_path = Path(args.output_dir) / "manifest.json"
    payload = {
        "input": str(Path(args.input).resolve()),
        "items_per_shard": args.items_per_shard,
        "start_index": args.start_index,
        "end_index": args.end_index or None,
        "scanned": scanned,
        "emitted": emitted,
        "n_shards": len(writer.manifest),
        "shards": writer.manifest,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[split] saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
