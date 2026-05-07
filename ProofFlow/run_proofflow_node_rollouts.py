#!/usr/bin/env python3
"""
Reroll ProofFlow nodes while reusing existing benchmark artifacts.

This entrypoint is intentionally narrower than run_proofflow_benchmark.py:
- no proof-graph generation
- no full task rerun
- reuse existing proof_items/formalizations from prior pickle files
- reroll only node-local model calls needed for latency/success/token estimates
"""

import argparse
import csv
import json
import os
import pickle
import random
import re
import shutil
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "matplotlib"))

from _bootstrap_proofflow import bootstrap_canonical_proofflow

bootstrap_canonical_proofflow()

from proofflow import LeanServer, LLMManager
from proofflow.io import RenamedUnpickler
from proofflow.lean_check import process_lean_string
from proofflow.proof_formalize import run_formalizer_prompt
from proofflow.proof_prover import run_solver_prompt
from proofflow.utils import remove_imports


ROOT_DIR = Path(__file__).resolve().parent
INPUT_PRICE_PER_MTOK = 0.7
OUTPUT_PRICE_PER_MTOK = 2.5
CONTEXT_MODE_CHOICES = (
    "official_no_context",
    "no_dependency_context",
    "strict_cached_only",
    "cached_then_roll",
    "allow_formalization_fallback",
)
SORRY_RE = re.compile(r"\b(?:sorry|admit)\b")
DEFAULT_SEMANTIC_SCORE_THRESHOLD = 0.0


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value in (None, "") else value


def _build_model_info(model: str, base_url: str = "", api_key: str = "") -> dict:
    info = {"model": model}
    if base_url:
        info["base_url"] = base_url
    if api_key:
        info["api_key"] = api_key
    return info


def _rough_tokens(text: str) -> int:
    return max(1, len(str(text)) // 4)


def _input_tokens_from_log(log: dict) -> int:
    prompt_tokens = log.get("prompt_tokens")
    if isinstance(prompt_tokens, int):
        return prompt_tokens

    messages = list(log.get("messages") or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return sum(_rough_tokens(msg.get("content", "")) for msg in messages)


def _output_tokens_from_log(log: dict) -> int:
    generated = log.get("generated_tokens")
    if isinstance(generated, int):
        return generated
    return _rough_tokens(log.get("response", ""))


def _usage_records(logs: List[dict]) -> List[dict]:
    records = []
    for idx, log in enumerate(logs):
        input_tokens = _input_tokens_from_log(log)
        output_tokens = _output_tokens_from_log(log)
        records.append(
            {
                "call_idx": idx,
                "model": log.get("model", ""),
                "duration_sec": float(log.get("duration", 0.0) or 0.0),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens),
                "cost_usd": float(
                    input_tokens * INPUT_PRICE_PER_MTOK / 1_000_000.0
                    + output_tokens * OUTPUT_PRICE_PER_MTOK / 1_000_000.0
                ),
            }
        )
    return records


def _usage_summary(records: List[dict]) -> dict:
    input_tokens = sum(int(record["input_tokens"]) for record in records)
    output_tokens = sum(int(record["output_tokens"]) for record in records)
    total_tokens = input_tokens + output_tokens
    return {
        "n_calls": int(len(records)),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
        "cost_usd": float(
            input_tokens * INPUT_PRICE_PER_MTOK / 1_000_000.0
            + output_tokens * OUTPUT_PRICE_PER_MTOK / 1_000_000.0
        ),
    }


class LocalVLLMBatchModel:
    def __init__(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 40960,
        max_num_seqs: int = 0,
        dtype: str = "bfloat16",
        system_prompt_path: Optional[str] = None,
    ) -> None:
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vllm and transformers are required for --model_backend vllm_local"
            ) from exc

        self.model_path = model_path
        self.max_model_len = int(max_model_len)
        self.SamplingParams = SamplingParams
        self.system_prompt = (
            Path(system_prompt_path).read_text(encoding="utf-8")
            if system_prompt_path
            else None
        )
        if max_num_seqs <= 0:
            max_num_seqs = self._auto_max_num_seqs()
        print(
            "[vLLM] loading",
            json.dumps(
                {
                    "model_path": model_path,
                    "tensor_parallel_size": int(tensor_parallel_size),
                    "gpu_memory_utilization": float(gpu_memory_utilization),
                    "max_model_len": int(max_model_len),
                    "max_num_seqs": int(max_num_seqs),
                    "dtype": dtype,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            dtype=str(dtype),
            disable_custom_all_reduce=True,
        )

    @staticmethod
    def _auto_max_num_seqs() -> int:
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_gb >= 110:
                    return 64
                if total_gb >= 70:
                    return 48
                if total_gb >= 40:
                    return 32
        except Exception:
            pass
        return 16

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        messages = list(messages)
        if self.system_prompt and (not messages or messages[0].get("role") != "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def _effective_max_tokens(self, prompts: List[str], requested_max_tokens: int) -> int:
        effective = int(requested_max_tokens)
        for prompt in prompts:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
            effective = min(effective, max(1, self.max_model_len - len(prompt_ids)))
        return max(1, effective)

    def generate_messages_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[dict]:
        prompts = [self._messages_to_prompt(messages) for messages in batch_messages]
        prompt_token_counts = [
            len(self.tokenizer(prompt, add_special_tokens=False).get("input_ids", []))
            for prompt in prompts
        ]
        sampling_params = self.SamplingParams(
            max_tokens=self._effective_max_tokens(prompts, max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            n=1,
        )
        t0 = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params)
        batch_latency_sec = time.perf_counter() - t0
        records = []
        for idx, output in enumerate(outputs):
            sampled_outputs = getattr(output, "outputs", []) or []
            sampled = sampled_outputs[0] if sampled_outputs else None
            text = getattr(sampled, "text", "") if sampled is not None else ""
            token_ids = getattr(sampled, "token_ids", None) if sampled is not None else None
            output_tokens = len(token_ids) if token_ids is not None else _rough_tokens(text)
            input_tokens = prompt_token_counts[idx]
            records.append(
                {
                    "text": text,
                    "infer_latency_sec": batch_latency_sec,
                    "infer_batch_latency_sec": batch_latency_sec,
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "total_tokens": int(input_tokens + output_tokens),
                    "latency_observation": "vllm_batch_generate",
                }
            )
        return records


class OpenAIChatBatchModel:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "local",
        system_prompt_path: Optional[str] = None,
        request_parallelism: int = 16,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required for --model_backend api workflow") from exc
        self.model_path = model
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_parallelism = max(1, int(request_parallelism))
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.system_prompt = (
            Path(system_prompt_path).read_text(encoding="utf-8")
            if system_prompt_path
            else None
        )

    def _messages_with_system(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = list(messages)
        if self.system_prompt and (not messages or messages[0].get("role") != "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def _generate_one(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> dict:
        request_messages = self._messages_with_system(messages)
        t0 = time.perf_counter()
        kwargs = {
            "model": self.model,
            "messages": request_messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if int(top_k) > 0:
            kwargs["extra_body"] = {"top_k": int(top_k)}
        completion = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        choice = completion.choices[0] if completion.choices else None
        message = getattr(choice, "message", None)
        text = getattr(message, "content", "") if message is not None else ""
        usage = getattr(completion, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if input_tokens <= 0:
            input_tokens = sum(_rough_tokens(msg.get("content", "")) for msg in request_messages)
        if output_tokens <= 0:
            output_tokens = _rough_tokens(text)
        return {
            "text": text,
            "infer_latency_sec": latency,
            "infer_batch_latency_sec": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_observation": "openai_compatible_request",
        }

    def generate_messages_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[dict]:
        outputs: List[Optional[dict]] = [None] * len(batch_messages)
        with ThreadPoolExecutor(max_workers=min(self.request_parallelism, max(1, len(batch_messages)))) as executor:
            futures = {
                executor.submit(
                    self._generate_one,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ): idx
                for idx, messages in enumerate(batch_messages)
            }
            for future in as_completed(futures):
                outputs[futures[future]] = future.result()
        return [output or {} for output in outputs]


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        try:
            return RenamedUnpickler(f).load()
        except Exception:
            f.seek(0)
            return pickle.load(f)


def _save_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _stage_names_for_mode(mode: str) -> List[str]:
    if mode == "workflow":
        return ["workflow"]
    if mode == "both":
        return ["formalizer", "solver"]
    if mode in {"formalizer", "solver"}:
        return [mode]
    return [str(mode)]


def _metadata_completed_stages(metadata: dict) -> set[str]:
    stages = set(str(s) for s in (metadata.get("node_reroll_completed_stages") or []))
    if metadata.get("node_reroll_status") == "completed":
        mode = metadata.get("node_reroll_mode")
        if mode:
            stages.update(_stage_names_for_mode(str(mode)))
    return stages


def _checkpoint_is_completed(path: Path, mode: Optional[str] = None) -> bool:
    if not path.exists():
        return False
    try:
        payload = _load_pickle(path)
    except Exception:
        return False
    metadata = payload.get("run_metadata") or {}
    if mode is None:
        return metadata.get("node_reroll_status") == "completed"
    completed = _metadata_completed_stages(metadata)
    return set(_stage_names_for_mode(mode)).issubset(completed)


def _done_index_dir(claim_dir: Path) -> Path:
    return claim_dir / "_done_index"


def _done_marker_path(claim_dir: Path, stem: str, stage_key: str = "default") -> Path:
    return _done_index_dir(claim_dir) / stage_key / f"{stem}.json"


def _write_done_marker(claim_dir: Path, stem: str, payload: dict, stage_key: str = "default") -> None:
    (_done_index_dir(claim_dir) / stage_key).mkdir(parents=True, exist_ok=True)
    _write_json_atomic(_done_marker_path(claim_dir, stem, stage_key), payload)


def _remove_done_marker(claim_dir: Path, stem: str, stage_key: str = "default") -> None:
    try:
        _done_marker_path(claim_dir, stem, stage_key).unlink()
    except (FileNotFoundError, OSError):
        pass


def _try_claim_task(
    *,
    claim_dir: Path,
    output_dir: Path,
    stem: str,
    mode: str,
    claim_timeout_sec: int,
) -> Optional[Path]:
    stage_key = "_".join(_stage_names_for_mode(mode))
    claim_path = claim_dir / stage_key / stem
    heartbeat_path = claim_path / "heartbeat.json"
    worker_info_path = claim_path / "worker_info.json"
    done_marker_path = _done_marker_path(claim_dir, stem, stage_key)

    if done_marker_path.exists():
        if _checkpoint_is_completed(output_dir / f"{stem}.pickle", mode):
            return None
        _remove_done_marker(claim_dir, stem, stage_key)

    if claim_path.exists():
        is_stale = True
        if heartbeat_path.exists():
            try:
                payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
                updated_at = float(payload.get("updated_at", 0.0))
                is_stale = (time.time() - updated_at) > float(claim_timeout_sec)
            except Exception:
                is_stale = True
        elif worker_info_path.exists():
            try:
                payload = json.loads(worker_info_path.read_text(encoding="utf-8"))
                updated_at = float(payload.get("time", 0.0))
                is_stale = (time.time() - updated_at) > float(claim_timeout_sec)
            except Exception:
                is_stale = True
        if is_stale:
            shutil.rmtree(claim_path, ignore_errors=True)

    try:
        claim_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return None

    now = time.time()
    worker_info = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time": now,
    }
    _write_json_atomic(worker_info_path, worker_info)
    _write_json_atomic(
        heartbeat_path,
        {
            "hostname": worker_info["hostname"],
            "pid": worker_info["pid"],
            "started_at": now,
            "updated_at": now,
        },
    )
    return claim_path


class ClaimHeartbeat:
    def __init__(
        self,
        *,
        claim_path: Optional[Path],
        progress_path: Path,
        worker_id: str,
        interval_sec: int,
    ) -> None:
        self.claim_path = claim_path
        self.progress_path = progress_path
        self.worker_id = worker_id
        self.interval_sec = max(1, int(interval_sec))
        self.payload: Dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def update(self, payload: Dict[str, Any]) -> None:
        self.payload = dict(payload)
        self.payload["updated_at"] = time.time()
        _write_json_atomic(self.progress_path, self.payload)

    def start(self) -> None:
        if self._thread is not None:
            return

        def run() -> None:
            while not self._stop_event.wait(self.interval_sec):
                now = time.time()
                if self.claim_path is not None:
                    _write_json_atomic(
                        self.claim_path / "heartbeat.json",
                        {
                            "worker_id": self.worker_id,
                            "hostname": socket.gethostname(),
                            "pid": os.getpid(),
                            "updated_at": now,
                        },
                    )
                if self.payload:
                    payload = dict(self.payload)
                    payload["updated_at"] = now
                    _write_json_atomic(self.progress_path, payload)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1)


def _is_terminal(item) -> bool:
    return str(getattr(item, "id", "")).startswith(("tc_", "def_"))


def _has_formalization(item) -> bool:
    formalization = getattr(item, "formalization", None) or {}
    return bool(formalization.get("lean_code"))


def _selected_node_ids(raw: str) -> Optional[set[str]]:
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _iter_source_pickles(source_dir: Path) -> Iterable[Path]:
    yield from sorted([p for p in source_dir.iterdir() if p.suffix in (".pickle", ".pkl")])


def _worker_file_indices(
    *,
    total_files: int,
    worker_shard_count: int,
    worker_shard_index: int,
) -> List[int]:
    if worker_shard_count > 0:
        if not 0 <= worker_shard_index < worker_shard_count:
            raise ValueError(
                f"worker_shard_index must be in [0, {worker_shard_count}), got {worker_shard_index}"
            )
        return [idx for idx in range(total_files) if idx % worker_shard_count == worker_shard_index]
    if worker_shard_index != 0:
        raise ValueError("worker_shard_index requires worker_shard_count > 0")
    return list(range(total_files))


def _shuffled_indices(indices: List[int], scan_seed: str) -> List[int]:
    shuffled = list(indices)
    rng = random.Random()
    rng.seed(str(scan_seed))
    rng.shuffle(shuffled)
    return shuffled


def _append_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "node_id",
        "node_kind",
        "stage",
        "blocked",
        "context_mode",
        "context_source",
        "missing_dependencies",
        "n_rollouts",
        "success_rate",
        "selected_success",
        "latency_mean_sec",
        "latency_min_sec",
        "latency_max_sec",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_task_summary(
    summary_dir: Path,
    task: str,
    rows: List[dict],
    summary_key: Optional[str] = None,
) -> None:
    payload = {
        "task": task,
        "updated_at": time.time(),
        "n_rows": len(rows),
        "rows": rows,
    }
    name = summary_key or task
    _write_json_atomic(summary_dir / f"{name}.json", payload)


def _finalize_summaries(output_dir: Path) -> List[dict]:
    summary_dir = output_dir / "summaries"
    rows: List[dict] = []
    for path in sorted(summary_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.extend(payload.get("rows") or [])

    summary_jsonl = output_dir / "node_rollout_summary.jsonl"
    summary_csv = output_dir / "node_rollout_summary.csv"
    with summary_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv(summary_csv, rows)
    return rows


def _stage_summary(task: str, item, stage: str, result: dict, usage: dict) -> dict:
    records = result.get("rollout_records") or []
    success_key = "lean_verify" if stage == "solver" else "lean_pass"
    successes = [bool(record.get(success_key, False)) for record in records]
    latencies = [
        float(record.get("total_latency_sec", record.get("latency_sec", 0.0)) or 0.0)
        for record in records
    ]
    n_rollouts = len(records) if records else int(result.get("tries", 0) or 0)
    success_rate = (
        sum(1 for ok in successes if ok) / float(len(successes))
        if successes
        else float(bool(result.get(success_key, False)))
    )
    node_id = str(getattr(item, "id", ""))
    return {
        "task": task,
        "node_id": node_id,
        "node_kind": "terminal" if _is_terminal(item) else "lemma",
        "stage": stage,
        "n_rollouts": int(n_rollouts),
        "success_rate": float(success_rate),
        "selected_success": bool(result.get(success_key, False)),
        "latency_mean_sec": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_min_sec": float(min(latencies)) if latencies else 0.0,
        "latency_max_sec": float(max(latencies)) if latencies else 0.0,
        **usage,
    }


def _formalizer_user_prompt(
    item,
    dependency_context_code: str = "",
    original_proof: str = "",
    dependency_context_note: str = "",
) -> str:
    node_id = str(getattr(item, "id", ""))
    dependencies = [str(dep) for dep in (getattr(item, "dependencies", []) or [])]

    if not _is_terminal(item):
        lemma_header = f"lemma {node_id}"
        prompt = f"""Please autoformalize the following natural language problem proof step in Lean 4.
Use the following lemma name: {lemma_header}
The natural language statement is: {item.statement}
The dependencies are: {dependencies}

This is the lean code skeleton you need to use:

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

{lemma_header}
[place correct hypothesis here] :
[place goal here] := by
sorry
```

Important: **Please write only one lemma or theorem**!!
"""
    else:
        prompt = rf"""Please autoformalize the following natural language theorem condition in Lean 4.
Use the following name: {node_id}

The natural language statement is: {item.statement}

These the lean code skeleton you need to use (please make needed changes and fill ????):

```lean4
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat Filter

variable [place correct hypothesis here]
```

Do not produce a theorem or a proof. Only provide the Lean 4 code.
Warning: this is not a lemma/theorem, it is a theorem condition. For this problem make use of "variable" and follow the following examples.

Name: tc\_1; Statement: Let \$(a\_n)\$ be a sequence of positive real numbers.
Lean 4 formalization:

```lean4
variable (a : ℕ → ℝ)
(tc_1 : ∀ n, 0 < a n)
```

Name: tc\_2; Statement: Let \$A\$ be a \$2 x 2\$ real matrix with eigenvalues \$\lambda\_1 = 3\$ and \$\lambda\_2 = -2\$.
Lean 4 formalization:

```lean4
variable (A : Matrix (Fin 2) (Fin 2) ℝ)
(tc_2 : ∃ v1 v2 : Fin 2 → ℝ, v1 ≠ 0 ∧ v2 ≠ 0 ∧ A.vecMul v1 = 3 • v1 ∧ A.vecMul v2 = -2 • v2)
```"""

    if dependency_context_code:
        prompt += f"""

This proof step depends on previous proof steps. The following Lean 4 code
contains verified declarations and proofs you may use:

```lean4
{dependency_context_code}
```

Focus on the original formalization task and use the verified Lean code only
as context for variables, declarations, and prior facts.
"""

    if dependency_context_note:
        prompt += dependency_context_note

    if original_proof:
        prompt += "\n\nThis formalization task is part of a larger proof:\n"
        prompt += original_proof
        prompt += "\nUse the full proof only for extra context such as variable types and domains."

    return prompt


def _solver_user_prompt(item, dependency_context_code: str = "") -> str:
    prompt = f"""
This is the lemma/theorem I want you to prove:
{item.statement}

Complete the following Lean 4 code (**do not remove imports**):

```lean4
{item.formalization["lean_code"]}
```

You can adapt previous lean4 lemma statement to fit the goal, specially if you encounter errors.
"""
    if dependency_context_code:
        prompt += f"""

The following Lean 4 code contains already verified previous proof steps and declarations
that this node may depend on. You may use these names directly:

```lean4
{dependency_context_code}
```
"""
    if not item.formalization.get("lean_pass"):
        prompt += "\nThe previous Lean4 code I sent you contains errors. Please take that into account."
    return prompt


def _official_formalizer_dependency_context_note(item, all_items: List[Any]) -> str:
    dependencies = [str(dep) for dep in (getattr(item, "dependencies", []) or [])]
    if not dependencies:
        return ""
    id_to_item = _item_by_id(all_items)
    parts = [
        "\n\n This proof step depend on previous proof steps, namely steps "
        + str(dependencies)
        + ".",
        "Please make use use of their formal lean4 code, which contains relevant "
        "lean4 hypothesis and type declarations you may use:",
    ]
    for dep_id in dependencies:
        dep_item = id_to_item.get(dep_id)
        if dep_item is None:
            parts.append(
                f"Lean code not found or incorrect. Here is the natural language "
                f"statement of step {dep_id}:"
            )
            continue
        formalization = getattr(dep_item, "formalization", None) or {}
        if (
            formalization.get("lean_code")
            and formalization.get("lean_pass")
            and not _contains_sorry(formalization.get("lean_code"))
        ):
            parts.append(remove_imports(formalization["lean_code"]))
        else:
            parts.append(
                f"Lean code not found or incorrect. Here is the natural language "
                f"statement of step {dep_id}: {getattr(dep_item, 'statement', '')}"
            )
    parts.append(
        "Focus on the original formalization task I gave you and use the previous "
        "Lean codes, extra context, type declarations, variables domains, etc. "
        "You can assume the information is correct. Make use of it!"
    )
    return "\n".join(parts)


def _extract_lean_code(text_input: str) -> str:
    text = str(text_input)
    matches = re.findall(r"```(?:lean4|lean)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return process_lean_string(matches[-1].strip())

    stripped = text.strip()
    if re.search(r"\b(lemma|theorem|variable|def)\b", stripped):
        return process_lean_string(stripped)
    raise ValueError("No Lean 4 code block found.")


def _item_by_id(proof_items: List[Any]) -> Dict[str, Any]:
    return {str(getattr(item, "id", "")): item for item in proof_items}


def _dependency_closure_ids(item, id_to_item: Dict[str, Any]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    def visit(node_id: str) -> None:
        if node_id in seen:
            return
        dep_item = id_to_item.get(node_id)
        if dep_item is None:
            return
        for parent_id in getattr(dep_item, "dependencies", []) or []:
            visit(str(parent_id))
        seen.add(node_id)
        ordered.append(node_id)

    for dep_id in getattr(item, "dependencies", []) or []:
        visit(str(dep_id))
    return ordered


def _contains_sorry(code: Any) -> bool:
    return bool(SORRY_RE.search(str(code or "")))


def _context_safe_code(code: Any) -> str:
    text = str(code or "")
    return "" if _contains_sorry(text) else text


def _semantic_score_for_item(item) -> Optional[float]:
    score = getattr(item, "score", None) or {}
    if not isinstance(score, dict):
        return None
    raw = score.get("semantic_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _semantic_context_source_for_item(item, semantic_score_threshold: float) -> str:
    if float(semantic_score_threshold) <= 0.0:
        return ""
    score = _semantic_score_for_item(item)
    if score is None:
        return "missing_semantic_score"
    if score <= float(semantic_score_threshold):
        return "low_semantic_score"
    return ""


def _context_code_for_item(
    dep_item,
    context_mode: str,
    *,
    semantic_score_threshold: float = DEFAULT_SEMANTIC_SCORE_THRESHOLD,
) -> tuple[str, str]:
    solved = getattr(dep_item, "solved_lemma", None) or {}
    formalization = getattr(dep_item, "formalization", None) or {}
    semantic_blocker = _semantic_context_source_for_item(dep_item, semantic_score_threshold)

    if _is_terminal(dep_item):
        code = _context_safe_code(formalization.get("lean_code"))
        if formalization.get("lean_code") and not code:
            return "", "terminal_formalization_sorry"
        if code and formalization.get("lean_pass") and semantic_blocker:
            return "", semantic_blocker
        if code and formalization.get("lean_pass"):
            return code, "terminal_formalization"
        return "", "missing"

    if solved.get("lean_verify"):
        code = _context_safe_code(solved.get("candidate_lean_code") or solved.get("lean_code"))
        if code and semantic_blocker:
            return "", semantic_blocker
        if code:
            return code, "solved_rollout"
        return "", "solved_rollout_sorry"

    if (
        context_mode == "allow_formalization_fallback"
        and formalization.get("lean_code")
        and formalization.get("lean_pass")
    ):
        code = _context_safe_code(formalization.get("lean_code"))
        if code and semantic_blocker:
            return "", semantic_blocker
        if code:
            return code, "formalization_fallback"
        return "", "formalization_fallback_sorry"

    return "", "missing"


def _dependency_context(
    item,
    id_to_item: Dict[str, Any],
    *,
    context_mode: str,
    semantic_score_threshold: float = DEFAULT_SEMANTIC_SCORE_THRESHOLD,
) -> tuple[str, List[str], List[str], Dict[str, str]]:
    if context_mode in {"official_no_context", "no_dependency_context"}:
        return "", [], [], {}

    codes = []
    used_ids = []
    missing_ids = []
    sources: Dict[str, str] = {}
    for dep_id in _dependency_closure_ids(item, id_to_item):
        dep_item = id_to_item.get(dep_id)
        if dep_item is None:
            missing_ids.append(dep_id)
            sources[dep_id] = "missing_item"
            continue
        code, source = _context_code_for_item(
            dep_item,
            context_mode,
            semantic_score_threshold=semantic_score_threshold,
        )
        sources[dep_id] = source
        if code:
            codes.append(remove_imports(code))
            used_ids.append(dep_id)
        else:
            missing_ids.append(dep_id)
    return "\n\n".join(codes), used_ids, missing_ids, sources


def _validate_solver_response(
    *,
    response_text: str,
    dependency_context_code: str,
    lean_server: LeanServer,
) -> dict:
    candidate_code = _extract_lean_code(response_text)
    verify_code = candidate_code
    if dependency_context_code:
        verify_code = process_lean_string(
            dependency_context_code + "\n\n" + remove_imports(candidate_code)
        )
    lean_pass, lean_verify, error_msg = lean_server.check_lean_string(verify_code)
    return {
        "lean_code": verify_code,
        "candidate_lean_code": candidate_code,
        "lean_pass": bool(lean_pass),
        "lean_verify": bool(lean_verify),
        "error_msg": error_msg,
    }


def _validate_formalizer_response(
    *,
    response_text: str,
    lean_server: LeanServer,
) -> dict:
    lean_code = _extract_lean_code(response_text)
    lean_pass, _lean_verify, error_msg = lean_server.check_lean_string(lean_code)
    return {
        "lean_code": lean_code,
        "lean_pass": bool(lean_pass),
        "error_msg": [] if lean_pass else error_msg,
    }


def _run_formalizer_vllm_batch_for_task(
    *,
    task: str,
    proof_items: List[Any],
    lean_server: LeanServer,
    vllm_model: LocalVLLMBatchModel,
    selected_ids: Optional[set[str]],
    include_terminals: bool,
    n_rollouts: int,
    batch_size: int,
    verify_parallelism: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    heartbeat: Optional[ClaimHeartbeat] = None,
    started_at: Optional[float] = None,
) -> List[dict]:
    node_jobs = []
    flat_messages = []
    flat_meta = []

    for item in proof_items:
        node_id = str(getattr(item, "id", ""))
        if selected_ids is not None and node_id not in selected_ids:
            continue
        if _is_terminal(item) and not include_terminals:
            continue
        node_jobs.append(
            {
                "item": item,
                "node_id": node_id,
                "rollout_records": [],
            }
        )
        messages = [{"role": "user", "content": _formalizer_user_prompt(item)}]
        for rollout_id in range(int(n_rollouts)):
            flat_messages.append(messages)
            flat_meta.append((len(node_jobs) - 1, rollout_id))

    if not flat_messages:
        return []

    batch_size = max(1, int(batch_size or len(flat_messages)))
    verify_parallelism = max(1, int(verify_parallelism))
    generated_records: List[Optional[dict]] = [None] * len(flat_messages)
    for start in range(0, len(flat_messages), batch_size):
        end = min(start + batch_size, len(flat_messages))
        print(f"  [vllm] formalizer generate prompts {start + 1}-{end}/{len(flat_messages)}")
        batch_outputs = vllm_model.generate_messages_batch(
            flat_messages[start:end],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        for local_idx, output in enumerate(batch_outputs):
            generated_records[start + local_idx] = output
        if heartbeat is not None:
            heartbeat.update(
                {
                    "task": task,
                    "status": "in_progress",
                    "worker_id": heartbeat.worker_id,
                    "started_at": started_at or time.time(),
                    "current_stage": "formalizer_vllm_generate",
                    "generated_rollouts": end,
                    "total_rollouts": len(flat_messages),
                    "completed_node_stages": 0,
                }
            )

    def verify_one(idx: int) -> tuple[int, dict]:
        output = generated_records[idx] or {}
        node_job_idx, rollout_id = flat_meta[idx]
        total_t0 = time.perf_counter()
        verify_t0 = time.perf_counter()
        candidate = {
            "lean_code": "",
            "lean_pass": False,
            "error_msg": "failure to validate generated response",
        }
        try:
            candidate = _validate_formalizer_response(
                response_text=str(output.get("text", "")),
                lean_server=lean_server,
            )
        except Exception as exc:
            candidate["error_msg"] = str(exc)
        verify_latency_sec = time.perf_counter() - verify_t0
        infer_latency_sec = float(output.get("infer_latency_sec") or 0.0)
        total_latency_sec = infer_latency_sec + verify_latency_sec
        input_tokens = int(output.get("input_tokens", 0) or 0)
        output_tokens = int(output.get("output_tokens", 0) or 0)
        record = {
            "rollout_id": int(rollout_id),
            "infer_latency_sec": infer_latency_sec,
            "infer_batch_latency_sec": float(output.get("infer_batch_latency_sec") or 0.0),
            "verify_latency_sec": verify_latency_sec,
            "total_latency_sec": total_latency_sec,
            "latency_sec": total_latency_sec,
            "lean_pass": bool(candidate.get("lean_pass", False)),
            "lean_code": candidate.get("lean_code", ""),
            "error_msg": candidate.get("error_msg", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(output.get("total_tokens", input_tokens + output_tokens) or 0),
            "latency_observation": output.get("latency_observation", "vllm_batch_generate"),
        }
        record["cost_usd"] = float(
            input_tokens * INPUT_PRICE_PER_MTOK / 1_000_000.0
            + output_tokens * OUTPUT_PRICE_PER_MTOK / 1_000_000.0
        )
        record["wall_latency_sec"] = time.perf_counter() - total_t0
        return node_job_idx, record

    completed = 0
    print(f"  [lean] formalizer verify {len(flat_messages)} rollouts with workers={verify_parallelism}")
    with ThreadPoolExecutor(max_workers=verify_parallelism) as executor:
        futures = [executor.submit(verify_one, idx) for idx in range(len(flat_messages))]
        for future in as_completed(futures):
            node_job_idx, record = future.result()
            node_jobs[node_job_idx]["rollout_records"].append(record)
            completed += 1
            if heartbeat is not None and (completed == len(flat_messages) or completed % max(1, n_rollouts) == 0):
                heartbeat.update(
                    {
                        "task": task,
                        "status": "in_progress",
                        "worker_id": heartbeat.worker_id,
                        "started_at": started_at or time.time(),
                        "current_stage": "formalizer_lean_verify",
                        "verified_rollouts": completed,
                        "total_rollouts": len(flat_messages),
                        "completed_node_stages": 0,
                    }
                )

    rows = []
    for node_job in node_jobs:
        item = node_job["item"]
        rollout_records = sorted(node_job["rollout_records"], key=lambda x: int(x.get("rollout_id", 0)))
        successful = [record for record in rollout_records if record.get("lean_pass")]
        selected = (
            min(successful, key=lambda record: float(record.get("latency_sec", 0.0)))
            if successful
            else min(rollout_records, key=lambda record: float(record.get("latency_sec", 0.0)))
        )
        usage_records = [
            {
                "call_idx": int(record.get("rollout_id", idx)),
                "model": vllm_model.model_path,
                "duration_sec": float(record.get("infer_latency_sec", 0.0)),
                "input_tokens": int(record.get("input_tokens", 0)),
                "output_tokens": int(record.get("output_tokens", 0)),
                "total_tokens": int(record.get("total_tokens", 0)),
                "cost_usd": float(record.get("cost_usd", 0.0)),
            }
            for idx, record in enumerate(rollout_records)
        ]
        formalization = {
            "lean_code": selected.get("lean_code", ""),
            "lean_pass": bool(selected.get("lean_pass", False)),
            "error_msg": [] if selected.get("lean_pass", False) else selected.get("error_msg", ""),
            "infer_latency_sec": float(selected.get("infer_latency_sec", 0.0)),
            "verify_latency_sec": float(selected.get("verify_latency_sec", 0.0)),
            "total_latency_sec": float(selected.get("total_latency_sec", 0.0)),
            "latency_sec": float(selected.get("latency_sec", 0.0)),
            "tries": int(n_rollouts),
            "selected_rollout_id": int(selected.get("rollout_id", 0)),
            "rollout_records": rollout_records,
            "pass_rate": sum(1 for record in rollout_records if record.get("lean_pass")) / float(len(rollout_records)),
            "attempt_history": [],
            "usage_records": usage_records,
            "usage_summary": _usage_summary(usage_records),
            "reroll_started_at": started_at or time.time(),
        }
        item.formalization = formalization
        row = _stage_summary(task, item, "formalizer", formalization, formalization["usage_summary"])
        row.update(
            {
                "blocked": False,
                "context_mode": "",
                "context_source": "",
                "missing_dependencies": "",
            }
        )
        rows.append(row)
        print(
            f"  {node_job['node_id']} formalizer pass_rate={row['success_rate']:.3f} "
            f"cost=${row['cost_usd']:.4f}"
        )

    return rows


def _run_solver_vllm_ready_jobs(
    *,
    task: str,
    lean_server: LeanServer,
    vllm_model: LocalVLLMBatchModel,
    node_jobs: List[dict],
    n_rollouts: int,
    batch_size: int,
    verify_parallelism: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    context_mode: str,
    round_idx: int,
    heartbeat: Optional[ClaimHeartbeat] = None,
    started_at: Optional[float] = None,
) -> List[dict]:
    flat_messages = []
    flat_meta = []

    for job_idx, node_job in enumerate(node_jobs):
        item = node_job["item"]
        dependency_context_code = node_job["dependency_context_code"]
        prompt = _solver_user_prompt(item, dependency_context_code)
        messages = [{"role": "user", "content": prompt}]
        for rollout_id in range(int(n_rollouts)):
            flat_messages.append(messages)
            flat_meta.append((job_idx, rollout_id))

    if not flat_messages:
        return []

    batch_size = max(1, int(batch_size or len(flat_messages)))
    verify_parallelism = max(1, int(verify_parallelism))
    generated_records: List[Optional[dict]] = [None] * len(flat_messages)
    for start in range(0, len(flat_messages), batch_size):
        end = min(start + batch_size, len(flat_messages))
        print(f"  [vllm] generate prompts {start + 1}-{end}/{len(flat_messages)}")
        batch_outputs = vllm_model.generate_messages_batch(
            flat_messages[start:end],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        for local_idx, output in enumerate(batch_outputs):
            generated_records[start + local_idx] = output
        if heartbeat is not None:
            heartbeat.update(
                {
                    "task": task,
                    "status": "in_progress",
                    "worker_id": heartbeat.worker_id,
                    "started_at": started_at or time.time(),
                    "current_stage": f"solver_vllm_generate_round_{round_idx}",
                    "generated_rollouts": end,
                    "total_rollouts": len(flat_messages),
                    "completed_node_stages": 0,
                }
            )

    def verify_one(idx: int) -> tuple[int, dict]:
        output = generated_records[idx] or {}
        node_job_idx, rollout_id = flat_meta[idx]
        node_job = node_jobs[node_job_idx]
        total_t0 = time.perf_counter()
        verify_t0 = time.perf_counter()
        candidate = {
            "lean_code": "",
            "candidate_lean_code": "",
            "lean_pass": False,
            "lean_verify": False,
            "error_msg": "failure to validate generated response",
        }
        try:
            candidate = _validate_solver_response(
                response_text=str(output.get("text", "")),
                dependency_context_code=node_job["dependency_context_code"],
                lean_server=lean_server,
            )
        except Exception as exc:
            candidate["error_msg"] = str(exc)
        verify_latency_sec = time.perf_counter() - verify_t0
        infer_latency_sec = float(output.get("infer_latency_sec") or 0.0)
        total_latency_sec = infer_latency_sec + verify_latency_sec
        input_tokens = int(output.get("input_tokens", 0) or 0)
        output_tokens = int(output.get("output_tokens", 0) or 0)
        record = {
            "rollout_id": int(rollout_id),
            "infer_latency_sec": infer_latency_sec,
            "infer_batch_latency_sec": float(output.get("infer_batch_latency_sec") or 0.0),
            "verify_latency_sec": verify_latency_sec,
            "total_latency_sec": total_latency_sec,
            "latency_sec": total_latency_sec,
            "lean_pass": bool(candidate.get("lean_pass", False)),
            "lean_verify": bool(candidate.get("lean_verify", False)),
            "lean_code": candidate.get("lean_code", ""),
            "candidate_lean_code": candidate.get("candidate_lean_code", ""),
            "error_msg": candidate.get("error_msg", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(output.get("total_tokens", input_tokens + output_tokens) or 0),
            "latency_observation": output.get("latency_observation", "vllm_batch_generate"),
            "dependency_context_node_ids": node_job["dependency_context_ids"],
        }
        record["cost_usd"] = float(
            input_tokens * INPUT_PRICE_PER_MTOK / 1_000_000.0
            + output_tokens * OUTPUT_PRICE_PER_MTOK / 1_000_000.0
        )
        record["wall_latency_sec"] = time.perf_counter() - total_t0
        return node_job_idx, record

    completed = 0
    print(f"  [lean] verify {len(flat_messages)} rollouts with workers={verify_parallelism}")
    with ThreadPoolExecutor(max_workers=verify_parallelism) as executor:
        futures = [executor.submit(verify_one, idx) for idx in range(len(flat_messages))]
        for future in as_completed(futures):
            node_job_idx, record = future.result()
            node_jobs[node_job_idx]["rollout_records"].append(record)
            completed += 1
            if heartbeat is not None and (completed == len(flat_messages) or completed % max(1, n_rollouts) == 0):
                heartbeat.update(
                    {
                        "task": task,
                        "status": "in_progress",
                        "worker_id": heartbeat.worker_id,
                        "started_at": started_at or time.time(),
                        "current_stage": f"solver_lean_verify_round_{round_idx}",
                        "verified_rollouts": completed,
                        "total_rollouts": len(flat_messages),
                        "completed_node_stages": 0,
                    }
                )

    rows = []
    for node_job in node_jobs:
        item = node_job["item"]
        rollout_records = sorted(node_job["rollout_records"], key=lambda x: int(x.get("rollout_id", 0)))
        successful = [record for record in rollout_records if record.get("lean_verify")]
        selected = (
            min(successful, key=lambda record: float(record.get("latency_sec", 0.0)))
            if successful
            else min(rollout_records, key=lambda record: float(record.get("latency_sec", 0.0)))
        )
        usage_records = [
            {
                "call_idx": int(record.get("rollout_id", idx)),
                "model": vllm_model.model_path,
                "duration_sec": float(record.get("infer_latency_sec", 0.0)),
                "input_tokens": int(record.get("input_tokens", 0)),
                "output_tokens": int(record.get("output_tokens", 0)),
                "total_tokens": int(record.get("total_tokens", 0)),
                "cost_usd": float(record.get("cost_usd", 0.0)),
            }
            for idx, record in enumerate(rollout_records)
        ]
        solved = {
            "lean_code": selected.get("lean_code", ""),
            "candidate_lean_code": selected.get("candidate_lean_code", ""),
            "lean_pass": bool(selected.get("lean_pass", False)),
            "lean_verify": bool(selected.get("lean_verify", False)),
            "error_msg": selected.get("error_msg", ""),
            "infer_latency_sec": float(selected.get("infer_latency_sec", 0.0)),
            "verify_latency_sec": float(selected.get("verify_latency_sec", 0.0)),
            "total_latency_sec": float(selected.get("total_latency_sec", 0.0)),
            "latency_sec": float(selected.get("latency_sec", 0.0)),
            "tries": int(n_rollouts),
            "selected_rollout_id": int(selected.get("rollout_id", 0)),
            "rollout_records": rollout_records,
            "verify_rate": sum(1 for record in rollout_records if record.get("lean_verify")) / float(len(rollout_records)),
            "attempt_history": [],
            "dependency_context_node_ids": node_job["dependency_context_ids"],
            "dependency_context_sources": node_job.get("dependency_context_sources", {}),
            "context_mode": context_mode,
            "scheduler_round": int(round_idx),
            "usage_records": usage_records,
            "usage_summary": _usage_summary(usage_records),
            "reroll_started_at": started_at or time.time(),
        }
        item.solved_lemma = solved
        row = _stage_summary(task, item, "solver", solved, solved["usage_summary"])
        row.update(
            {
                "blocked": False,
                "context_mode": context_mode,
                "context_source": ",".join(sorted(set(node_job.get("dependency_context_sources", {}).values()))),
                "missing_dependencies": "",
            }
        )
        rows.append(row)
        print(
            f"  {node_job['node_id']} solver verify_rate={row['success_rate']:.3f} "
            f"deps={len(node_job['dependency_context_ids'])} round={round_idx} cost=${row['cost_usd']:.4f}"
        )
    return rows


def _vllm_generate_and_verify_rollouts(
    *,
    task: str,
    node_id: str,
    stage: str,
    messages: List[Dict[str, str]],
    lean_server: LeanServer,
    vllm_model: LocalVLLMBatchModel,
    n_rollouts: int,
    batch_size: int,
    verify_parallelism: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    dependency_context_code: str = "",
) -> List[dict]:
    flat_messages = [messages for _ in range(int(n_rollouts))]
    batch_size = max(1, int(batch_size or len(flat_messages)))
    verify_parallelism = max(1, int(verify_parallelism))
    generated_records: List[Optional[dict]] = [None] * len(flat_messages)

    for start in range(0, len(flat_messages), batch_size):
        end = min(start + batch_size, len(flat_messages))
        print(f"  [vllm] {node_id} {stage} generate prompts {start + 1}-{end}/{len(flat_messages)}")
        batch_outputs = vllm_model.generate_messages_batch(
            flat_messages[start:end],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        for local_idx, output in enumerate(batch_outputs):
            generated_records[start + local_idx] = output

    def verify_one(idx: int) -> dict:
        output = generated_records[idx] or {}
        total_t0 = time.perf_counter()
        verify_t0 = time.perf_counter()
        if stage == "formalizer":
            candidate = {
                "lean_code": "",
                "lean_pass": False,
                "error_msg": "failure to validate generated response",
            }
            try:
                candidate = _validate_formalizer_response(
                    response_text=str(output.get("text", "")),
                    lean_server=lean_server,
                )
            except Exception as exc:
                candidate["error_msg"] = str(exc)
            success_fields = {"lean_pass": bool(candidate.get("lean_pass", False))}
        else:
            candidate = {
                "lean_code": "",
                "candidate_lean_code": "",
                "lean_pass": False,
                "lean_verify": False,
                "error_msg": "failure to validate generated response",
            }
            try:
                candidate = _validate_solver_response(
                    response_text=str(output.get("text", "")),
                    dependency_context_code=dependency_context_code,
                    lean_server=lean_server,
                )
            except Exception as exc:
                candidate["error_msg"] = str(exc)
            success_fields = {
                "lean_pass": bool(candidate.get("lean_pass", False)),
                "lean_verify": bool(candidate.get("lean_verify", False)),
                "candidate_lean_code": candidate.get("candidate_lean_code", ""),
            }

        verify_latency_sec = time.perf_counter() - verify_t0
        infer_latency_sec = float(output.get("infer_latency_sec") or 0.0)
        input_tokens = int(output.get("input_tokens", 0) or 0)
        output_tokens = int(output.get("output_tokens", 0) or 0)
        total_latency_sec = infer_latency_sec + verify_latency_sec
        record = {
            "rollout_id": int(idx),
            "infer_latency_sec": infer_latency_sec,
            "infer_batch_latency_sec": float(output.get("infer_batch_latency_sec") or 0.0),
            "verify_latency_sec": verify_latency_sec,
            "total_latency_sec": total_latency_sec,
            "latency_sec": total_latency_sec,
            "lean_code": candidate.get("lean_code", ""),
            "error_msg": candidate.get("error_msg", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": int(output.get("total_tokens", input_tokens + output_tokens) or 0),
            "latency_observation": output.get("latency_observation", "vllm_batch_generate"),
            "wall_latency_sec": time.perf_counter() - total_t0,
            **success_fields,
        }
        record["cost_usd"] = float(
            input_tokens * INPUT_PRICE_PER_MTOK / 1_000_000.0
            + output_tokens * OUTPUT_PRICE_PER_MTOK / 1_000_000.0
        )
        return record

    records: List[dict] = []
    print(f"  [lean] {node_id} {stage} verify {len(flat_messages)} rollouts with workers={verify_parallelism}")
    with ThreadPoolExecutor(max_workers=verify_parallelism) as executor:
        futures = [executor.submit(verify_one, idx) for idx in range(len(flat_messages))]
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda record: int(record.get("rollout_id", 0)))


def _usage_records_from_rollout_records(records: List[dict], model: str) -> List[dict]:
    return [
        {
            "call_idx": int(record.get("rollout_id", idx)),
            "model": model,
            "duration_sec": float(record.get("infer_latency_sec", 0.0)),
            "input_tokens": int(record.get("input_tokens", 0)),
            "output_tokens": int(record.get("output_tokens", 0)),
            "total_tokens": int(record.get("total_tokens", 0)),
            "cost_usd": float(record.get("cost_usd", 0.0)),
        }
        for idx, record in enumerate(records)
    ]


def _run_workflow_vllm_for_task(
    *,
    task: str,
    proof_items: List[Any],
    lean_server: LeanServer,
    formalize_model: LocalVLLMBatchModel,
    solver_model: LocalVLLMBatchModel,
    selected_ids: Optional[set[str]],
    include_terminals: bool,
    n_rollouts: int,
    batch_size: int,
    verify_parallelism: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    original_proof: str,
    heartbeat: Optional[ClaimHeartbeat] = None,
    started_at: Optional[float] = None,
) -> List[dict]:
    rows: List[dict] = []
    for item_idx, item in enumerate(proof_items, start=1):
        node_id = str(getattr(item, "id", ""))
        if selected_ids is not None and node_id not in selected_ids:
            continue
        terminal = _is_terminal(item)

        if include_terminals or not terminal:
            if formalize_model is solver_model:
                formalize_model.system_prompt = (ROOT_DIR / "prompts" / "lemma_formalizer.md").read_text(encoding="utf-8")
            note = _official_formalizer_dependency_context_note(item, proof_items)
            prompt = _formalizer_user_prompt(
                item,
                original_proof=original_proof,
                dependency_context_note=note,
            )
            rollout_records = _vllm_generate_and_verify_rollouts(
                task=task,
                node_id=node_id,
                stage="formalizer",
                messages=[{"role": "user", "content": prompt}],
                lean_server=lean_server,
                vllm_model=formalize_model,
                n_rollouts=n_rollouts,
                batch_size=batch_size,
                verify_parallelism=verify_parallelism,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            successful = [record for record in rollout_records if record.get("lean_pass")]
            selected = (
                min(successful, key=lambda record: float(record.get("latency_sec", 0.0)))
                if successful
                else min(rollout_records, key=lambda record: float(record.get("latency_sec", 0.0)))
            )
            usage_records = _usage_records_from_rollout_records(
                rollout_records,
                getattr(formalize_model, "model_path", ""),
            )
            formalization = {
                "lean_code": selected.get("lean_code", ""),
                "lean_pass": bool(selected.get("lean_pass", False)),
                "error_msg": [] if selected.get("lean_pass", False) else selected.get("error_msg", ""),
                "infer_latency_sec": float(selected.get("infer_latency_sec", 0.0)),
                "verify_latency_sec": float(selected.get("verify_latency_sec", 0.0)),
                "total_latency_sec": float(selected.get("total_latency_sec", 0.0)),
                "latency_sec": float(selected.get("latency_sec", 0.0)),
                "tries": int(n_rollouts),
                "selected_rollout_id": int(selected.get("rollout_id", 0)),
                "rollout_records": rollout_records,
                "pass_rate": sum(1 for record in rollout_records if record.get("lean_pass")) / float(len(rollout_records)),
                "attempt_history": [],
                "usage_records": usage_records,
                "usage_summary": _usage_summary(usage_records),
                "reroll_started_at": started_at or time.time(),
                "workflow_node_index": item_idx,
            }
            item.formalization = formalization
            row = _stage_summary(task, item, "formalizer", formalization, formalization["usage_summary"])
            row.update(
                {
                    "blocked": False,
                    "context_mode": "workflow_official_order",
                    "context_source": "official_previous_formalizations",
                    "missing_dependencies": "",
                }
            )
            rows.append(row)
            print(f"  {node_id} formalizer pass_rate={row['success_rate']:.3f} cost=${row['cost_usd']:.4f}")
            if heartbeat is not None:
                heartbeat.update(
                    {
                        "task": task,
                        "status": "in_progress",
                        "worker_id": heartbeat.worker_id,
                        "started_at": started_at or time.time(),
                        "n_nodes": len(proof_items),
                        "current_node_id": node_id,
                        "current_stage": "formalizer",
                        "completed_node_stages": len(rows),
                    }
                )

        if terminal:
            continue
        if not _has_formalization(item):
            print(f"  {node_id} solver skipped: missing reusable formalization")
            continue
        if solver_model is formalize_model:
            solver_model.system_prompt = (ROOT_DIR / "prompts" / "lemma_prover.md").read_text(encoding="utf-8")
        prompt = _solver_user_prompt(item, "")
        rollout_records = _vllm_generate_and_verify_rollouts(
            task=task,
            node_id=node_id,
            stage="solver",
            messages=[{"role": "user", "content": prompt}],
            lean_server=lean_server,
            vllm_model=solver_model,
            n_rollouts=n_rollouts,
            batch_size=batch_size,
            verify_parallelism=verify_parallelism,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            dependency_context_code="",
        )
        successful = [record for record in rollout_records if record.get("lean_verify")]
        selected = (
            min(successful, key=lambda record: float(record.get("latency_sec", 0.0)))
            if successful
            else min(rollout_records, key=lambda record: float(record.get("latency_sec", 0.0)))
        )
        usage_records = _usage_records_from_rollout_records(
            rollout_records,
            getattr(solver_model, "model_path", ""),
        )
        solved = {
            "lean_code": selected.get("lean_code", ""),
            "candidate_lean_code": selected.get("candidate_lean_code", ""),
            "lean_pass": bool(selected.get("lean_pass", False)),
            "lean_verify": bool(selected.get("lean_verify", False)),
            "error_msg": selected.get("error_msg", ""),
            "infer_latency_sec": float(selected.get("infer_latency_sec", 0.0)),
            "verify_latency_sec": float(selected.get("verify_latency_sec", 0.0)),
            "total_latency_sec": float(selected.get("total_latency_sec", 0.0)),
            "latency_sec": float(selected.get("latency_sec", 0.0)),
            "tries": int(n_rollouts),
            "selected_rollout_id": int(selected.get("rollout_id", 0)),
            "rollout_records": rollout_records,
            "verify_rate": sum(1 for record in rollout_records if record.get("lean_verify")) / float(len(rollout_records)),
            "attempt_history": [],
            "usage_records": usage_records,
            "usage_summary": _usage_summary(usage_records),
            "reroll_started_at": started_at or time.time(),
            "workflow_node_index": item_idx,
        }
        item.solved_lemma = solved
        row = _stage_summary(task, item, "solver", solved, solved["usage_summary"])
        row.update(
            {
                "blocked": False,
                "context_mode": "workflow_official_order",
                "context_source": "current_formalization",
                "missing_dependencies": "",
            }
        )
        rows.append(row)
        print(f"  {node_id} solver verify_rate={row['success_rate']:.3f} cost=${row['cost_usd']:.4f}")
        if heartbeat is not None:
            heartbeat.update(
                {
                    "task": task,
                    "status": "in_progress",
                    "worker_id": heartbeat.worker_id,
                    "started_at": started_at or time.time(),
                    "n_nodes": len(proof_items),
                    "current_node_id": node_id,
                    "current_stage": "solver",
                    "completed_node_stages": len(rows),
                }
            )
    return rows


def _blocked_solver_result(
    *,
    missing_ids: List[str],
    context_mode: str,
    started_at: Optional[float],
) -> dict:
    usage_records: List[dict] = []
    return {
        "lean_code": "",
        "candidate_lean_code": "",
        "lean_pass": False,
        "lean_verify": False,
        "error_msg": "blocked_missing_dependency_context",
        "infer_latency_sec": 0.0,
        "verify_latency_sec": 0.0,
        "total_latency_sec": 0.0,
        "latency_sec": 0.0,
        "tries": 0,
        "selected_rollout_id": -1,
        "rollout_records": [],
        "verify_rate": 0.0,
        "attempt_history": [],
        "dependency_context_node_ids": [],
        "blocked_missing_dependencies": list(missing_ids),
        "context_mode": context_mode,
        "usage_records": usage_records,
        "usage_summary": _usage_summary(usage_records),
        "reroll_started_at": started_at or time.time(),
    }


def _mark_solver_blocked(
    *,
    task: str,
    item,
    missing_ids: List[str],
    context_mode: str,
    started_at: Optional[float],
) -> dict:
    solved = _blocked_solver_result(
        missing_ids=missing_ids,
        context_mode=context_mode,
        started_at=started_at,
    )
    item.solved_lemma = solved
    row = _stage_summary(task, item, "solver", solved, solved["usage_summary"])
    row.update(
        {
            "blocked": True,
            "context_mode": context_mode,
            "context_source": "missing_dependency_context",
            "missing_dependencies": ",".join(missing_ids),
        }
    )
    return row


def _missing_dependencies_can_wait(
    *,
    missing_ids: List[str],
    unresolved: Dict[str, Any],
    selected_ids: Optional[set[str]],
) -> bool:
    if not missing_ids:
        return False
    for dep_id in missing_ids:
        dep_item = unresolved.get(dep_id)
        if dep_item is None:
            return False
        if selected_ids is not None and dep_id not in selected_ids:
            return False
        if _is_terminal(dep_item) or not _has_formalization(dep_item):
            return False
    return True


def _run_solver_vllm_batch_for_task(
    *,
    task: str,
    proof_items: List[Any],
    lean_server: LeanServer,
    vllm_model: LocalVLLMBatchModel,
    selected_ids: Optional[set[str]],
    n_rollouts: int,
    batch_size: int,
    verify_parallelism: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    context_mode: str,
    semantic_score_threshold: float,
    heartbeat: Optional[ClaimHeartbeat] = None,
    started_at: Optional[float] = None,
) -> List[dict]:
    id_to_item = _item_by_id(proof_items)
    rows: List[dict] = []
    unresolved: Dict[str, Any] = {}

    for item in proof_items:
        node_id = str(getattr(item, "id", ""))
        if selected_ids is not None and node_id not in selected_ids:
            continue
        if _is_terminal(item):
            continue
        if not _has_formalization(item):
            print(f"  {node_id} solver skipped: missing reusable formalization")
            continue
        unresolved[node_id] = item

    if not unresolved:
        return rows

    round_idx = 0
    while unresolved:
        ready_jobs: List[dict] = []
        blocked_jobs: List[tuple[str, Any, List[str]]] = []
        deferred = 0

        for node_id, item in list(unresolved.items()):
            (
                dependency_context_code,
                dependency_context_ids,
                missing_ids,
                context_sources,
            ) = _dependency_context(
                item,
                id_to_item,
                context_mode=context_mode,
                semantic_score_threshold=semantic_score_threshold,
            )

            if not missing_ids:
                ready_jobs.append(
                    {
                        "item": item,
                        "node_id": node_id,
                        "dependency_context_code": dependency_context_code,
                        "dependency_context_ids": dependency_context_ids,
                        "dependency_context_sources": context_sources,
                        "rollout_records": [],
                    }
                )
                continue

            if context_mode == "cached_then_roll" and _missing_dependencies_can_wait(
                missing_ids=missing_ids,
                unresolved=unresolved,
                selected_ids=selected_ids,
            ):
                deferred += 1
                continue

            blocked_jobs.append((node_id, item, missing_ids))

        for node_id, item, missing_ids in blocked_jobs:
            rows.append(
                _mark_solver_blocked(
                    task=task,
                    item=item,
                    missing_ids=missing_ids,
                    context_mode=context_mode,
                    started_at=started_at,
                )
            )
            unresolved.pop(node_id, None)
            print(
                f"  {node_id} solver blocked: missing dependency context "
                f"{','.join(missing_ids)}"
            )

        if ready_jobs:
            round_idx += 1
            print(
                f"  [scheduler] solver round={round_idx} "
                f"ready_nodes={len(ready_jobs)} deferred_nodes={deferred} "
                f"blocked_nodes={len(blocked_jobs)}"
            )
            rows.extend(
                _run_solver_vllm_ready_jobs(
                    task=task,
                    lean_server=lean_server,
                    vllm_model=vllm_model,
                    node_jobs=ready_jobs,
                    n_rollouts=n_rollouts,
                    batch_size=batch_size,
                    verify_parallelism=verify_parallelism,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    context_mode=context_mode,
                    round_idx=round_idx,
                    heartbeat=heartbeat,
                    started_at=started_at,
                )
            )
            for node_job in ready_jobs:
                unresolved.pop(node_job["node_id"], None)
            continue

        if unresolved:
            # No ready node can make progress. This is usually an upstream node
            # whose own dependencies failed to produce verified context.
            for node_id, item in list(unresolved.items()):
                _context_code, _used_ids, missing_ids, _sources = _dependency_context(
                    item,
                    id_to_item,
                    context_mode=context_mode,
                    semantic_score_threshold=semantic_score_threshold,
                )
                rows.append(
                    _mark_solver_blocked(
                        task=task,
                        item=item,
                        missing_ids=missing_ids,
                        context_mode=context_mode,
                        started_at=started_at,
                    )
                )
                unresolved.pop(node_id, None)
                print(
                    f"  {node_id} solver blocked: no ready dependency path "
                    f"{','.join(missing_ids)}"
                )

    return rows


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Reuse existing ProofFlow pickles and reroll selected nodes only."
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=["infer", "finalize", "all", "paths", "status"],
        default="infer",
        help="infer runs rollout workers; finalize combines per-task summaries.",
    )
    parser.add_argument("--source_pickle_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lean_server_url", default=_env("LEAN_SERVER_URL", ""))
    parser.add_argument("--lean_project_path", default=_env("LEAN_PROJECT_PATH", ""))
    parser.add_argument("--lean_home", default=_env("LEAN_HOME", ""))
    parser.add_argument("--lean_tmp_base", default=_env("LEAN_TMP_BASE", "/tmp"))
    parser.add_argument("--lean_timeout", type=int, default=int(_env("LEAN_TIMEOUT", "120")))
    parser.add_argument("--solver_model", default=_env("GOEDEL_SOLVER_MODEL_LOC", ""))
    parser.add_argument("--solver_base_url", default=_env("GOEDEL_SOLVER_URL", ""))
    parser.add_argument("--solver_api_key", default=_env("GOEDEL_SOLVER_API_KEY", "local"))
    parser.add_argument("--formalize_model", default=_env("GOEDEL_FORMALIZER_MODEL_LOC", ""))
    parser.add_argument("--formalize_base_url", default=_env("GOEDEL_FORMALIZER_URL", ""))
    parser.add_argument("--formalize_api_key", default=_env("GOEDEL_FORMALIZER_API_KEY", "local"))
    parser.add_argument("--n_rollouts", type=int, default=8)
    parser.add_argument("--rollout_parallelism", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=38912)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--semantic_score_threshold",
        type=float,
        default=float(_env("SEMANTIC_SCORE_THRESHOLD", str(DEFAULT_SEMANTIC_SCORE_THRESHOLD))),
        help=(
            "Minimum item.score.semantic_score required before a formalization "
            "can be reused as dependency context. <=0 disables semantic gating "
            "and matches the official ProofFlow execution semantics."
        ),
    )
    parser.add_argument(
        "--model_backend",
        choices=["api", "vllm_local"],
        default=_env("PROOFFLOW_NODE_MODEL_BACKEND", "api"),
        help="api uses LLMManager/OpenAI-compatible calls; vllm_local loads vLLM LLM in this worker.",
    )
    parser.add_argument(
        "--context_mode",
        choices=CONTEXT_MODE_CHOICES,
        default=_env("CONTEXT_MODE", "official_no_context"),
        help=(
            "Dependency context policy for vllm_local solver rollouts. "
            "official_no_context matches the official ProofFlow solver path "
            "by not blocking on dependency context; "
            "cached_then_roll uses existing verified context first, then waits "
            "for upstream solver rollouts; strict_cached_only never unlocks "
            "from newly generated upstream rollouts; allow_formalization_fallback "
            "also accepts no-sorry passing formalizations above the semantic score threshold."
        ),
    )
    parser.add_argument("--vllm_model_path", default=_env("MODEL_PATH", ""))
    parser.add_argument("--vllm_formalizer_model_path", default=_env("FORMALIZER_MODEL_PATH", ""))
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=int(_env("TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=float(_env("GPU_MEMORY_UTILIZATION", "0.95")))
    parser.add_argument("--vllm_max_model_len", type=int, default=int(_env("VLLM_MAX_MODEL_LEN", "40960")))
    parser.add_argument("--vllm_max_num_seqs", type=int, default=int(_env("MAX_NUM_SEQS", "0")))
    parser.add_argument("--vllm_dtype", default=_env("VLLM_DTYPE", "bfloat16"))
    parser.add_argument(
        "--vllm_batch_size",
        type=int,
        default=int(_env("VLLM_BATCH_SIZE", "0")),
        help="Number of flattened node-rollout prompts per vLLM generate call. 0 means all prompts for one pickle.",
    )
    parser.add_argument("--verify_parallelism", type=int, default=int(_env("VERIFY_PARALLELISM", "8")))
    parser.add_argument(
        "--mode",
        choices=["solver", "formalizer", "both", "workflow"],
        default="solver",
        help=(
            "solver reuses current-node formalization and rerolls only lemma/theorem proofs; "
            "workflow replays each DAG in ProofFlow order as formalizer->solver per node."
        ),
    )
    parser.add_argument(
        "--include_terminals",
        action="store_true",
        help="Reroll tc_/def_ formalization nodes too. Ignored in solver-only mode.",
    )
    parser.add_argument("--node_ids", default="", help="Comma-separated node ids to reroll.")
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--claim_dir",
        default="",
        help="Shared claim directory for multi-worker runs. Defaults to <output_dir>/.claims.",
    )
    parser.add_argument("--claim_timeout_sec", type=int, default=7200)
    parser.add_argument("--claim_heartbeat_sec", type=int, default=30)
    parser.add_argument("--worker_id", default="")
    parser.add_argument("--scan_seed", default="")
    parser.add_argument("--worker_shard_count", type=int, default=0)
    parser.add_argument("--worker_shard_index", type=int, default=0)
    args = parser.parse_args()

    source_dir = Path(args.source_pickle_dir)
    output_dir = Path(args.output_dir)
    summary_jsonl = output_dir / "node_rollout_summary.jsonl"
    summary_csv = output_dir / "node_rollout_summary.csv"
    summary_dir = output_dir / "summaries"
    progress_dir = output_dir / "progress"
    selected_ids = _selected_node_ids(args.node_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_shard_count < 0:
        raise ValueError(f"worker_shard_count must be >= 0, got {args.worker_shard_count}")
    if not args.worker_id:
        args.worker_id = f"{socket.gethostname()}:{os.getpid()}"
    if not args.scan_seed:
        args.scan_seed = f"{args.worker_id}:{time.time_ns()}"

    if args.action in {"finalize", "paths", "status"}:
        if args.action == "status":
            source_files = list(_iter_source_pickles(source_dir))
            if args.max_tasks > 0:
                source_files = source_files[: args.max_tasks]
            completed = 0
            for source_path in source_files:
                out_path = output_dir / source_path.name
                if _checkpoint_is_completed(out_path, args.mode):
                    completed += 1
            total = len(source_files)
            pending = max(0, total - completed)
            print(f"STATUS_MODE={args.mode}")
            print(f"STATUS_TOTAL={total}")
            print(f"STATUS_COMPLETED={completed}")
            print(f"STATUS_PENDING={pending}")
            return
        rows = _finalize_summaries(output_dir) if args.action == "finalize" else []
        print("ProofFlow node rollout settings")
        print(f"  ACTION:              {args.action}")
        print(f"  SOURCE_PICKLE_DIR:   {source_dir}")
        print(f"  OUTPUT_DIR:          {output_dir}")
        print(f"  SUMMARY_JSONL:       {summary_jsonl}")
        print(f"  SUMMARY_CSV:         {summary_csv}")
        print(f"  CLAIM_DIR:           {args.claim_dir or (output_dir / '.claims')}")
        print(f"  LEAN_HOME:           {args.lean_home or '<env/default>'}")
        print(f"  LEAN_TMP_BASE:       {args.lean_tmp_base}")
        print(f"  LEAN_TIMEOUT:        {args.lean_timeout}")
        print(f"  WORKER_ID:           {args.worker_id}")
        print(f"  WORKER_SHARD:        {args.worker_shard_index}/{args.worker_shard_count}")
        print(f"  N_ROLLOUTS:          {args.n_rollouts}")
        print(f"  ROLLOUT_PARALLELISM: {args.rollout_parallelism}")
        print(f"  MODEL_BACKEND:       {args.model_backend}")
        print(f"  CONTEXT_MODE:        {args.context_mode}")
        print(f"  SEMANTIC_THRESHOLD:  {args.semantic_score_threshold}")
        print(f"  VLLM_MODEL_PATH:     {args.vllm_model_path or '<none>'}")
        print(f"  VLLM_FORMALIZER:     {args.vllm_formalizer_model_path or '<none>'}")
        if args.action == "finalize":
            total_cost = sum(float(row.get("cost_usd", 0.0)) for row in rows)
            print(f"Finalized rows: {len(rows)}")
            print(f"Estimated API cost from logged tokens: ${total_cost:.4f}")
        return

    if not args.lean_project_path and not args.lean_server_url:
        raise ValueError("Set --lean_project_path or --lean_server_url.")

    if args.lean_project_path:
        lean_server = LeanServer(
            project_path=args.lean_project_path,
            lean_home=args.lean_home or None,
            tmp_base=args.lean_tmp_base,
            timeout=args.lean_timeout,
        )
    else:
        lean_server = LeanServer(api_url=args.lean_server_url, timeout=args.lean_timeout)

    if args.model_backend == "vllm_local":
        if not args.vllm_model_path:
            args.vllm_model_path = args.solver_model
        if not args.vllm_formalizer_model_path:
            args.vllm_formalizer_model_path = args.formalize_model or args.vllm_model_path
        if args.mode in {"solver", "both", "workflow"} and not args.vllm_model_path:
            raise ValueError("Set --vllm_model_path or --solver_model for --model_backend vllm_local")
        if args.mode in {"formalizer", "both", "workflow"} and not args.vllm_formalizer_model_path:
            raise ValueError("Set --vllm_formalizer_model_path, --formalize_model, or --vllm_model_path")

        solver_model = None
        formalize_model = None
        if args.mode in {"solver", "both", "workflow"}:
            solver_model = LocalVLLMBatchModel(
                model_path=args.vllm_model_path,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
                max_num_seqs=args.vllm_max_num_seqs,
                dtype=args.vllm_dtype,
                system_prompt_path=str(ROOT_DIR / "prompts" / "lemma_prover.md"),
            )
        if args.mode in {"formalizer", "both", "workflow"}:
            if solver_model is not None and args.vllm_formalizer_model_path == args.vllm_model_path:
                formalize_model = solver_model
                formalize_model.system_prompt = (ROOT_DIR / "prompts" / "lemma_formalizer.md").read_text(encoding="utf-8")
            else:
                formalize_model = LocalVLLMBatchModel(
                    model_path=args.vllm_formalizer_model_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    max_model_len=args.vllm_max_model_len,
                    max_num_seqs=args.vllm_max_num_seqs,
                    dtype=args.vllm_dtype,
                    system_prompt_path=str(ROOT_DIR / "prompts" / "lemma_formalizer.md"),
                )
    elif args.mode == "workflow":
        if not args.solver_base_url:
            raise ValueError("workflow api mode requires --solver_base_url")
        if not args.formalize_base_url:
            raise ValueError("workflow api mode requires --formalize_base_url")
        solver_model = OpenAIChatBatchModel(
            model=args.solver_model,
            base_url=args.solver_base_url,
            api_key=args.solver_api_key,
            system_prompt_path=str(ROOT_DIR / "prompts" / "lemma_prover.md"),
            request_parallelism=args.rollout_parallelism,
        )
        formalize_model = OpenAIChatBatchModel(
            model=args.formalize_model,
            base_url=args.formalize_base_url,
            api_key=args.formalize_api_key,
            system_prompt_path=str(ROOT_DIR / "prompts" / "lemma_formalizer.md"),
            request_parallelism=args.rollout_parallelism,
        )
    else:
        solver_model = LLMManager(
            model_info=_build_model_info(args.solver_model, args.solver_base_url, args.solver_api_key),
            system_prompt_path=str(ROOT_DIR / "prompts" / "lemma_prover.md"),
            default_max_new_tokens=args.max_tokens,
            default_temperature=args.temperature,
            default_top_p=args.top_p,
            default_top_k=args.top_k,
        )
        formalize_model = None
    if args.model_backend != "vllm_local" and args.mode in {"formalizer", "both"}:
        formalize_model = LLMManager(
            model_info=_build_model_info(
                args.formalize_model,
                args.formalize_base_url,
                args.formalize_api_key,
            ),
            system_prompt_path=str(ROOT_DIR / "prompts" / "lemma_formalizer.md"),
            default_max_new_tokens=args.max_tokens,
            default_temperature=args.temperature,
            default_top_p=args.top_p,
            default_top_k=args.top_k,
        )

    all_rows: List[dict] = []
    source_files = list(_iter_source_pickles(source_dir))
    if args.max_tasks > 0:
        source_files = source_files[: args.max_tasks]
    worker_indices = _worker_file_indices(
        total_files=len(source_files),
        worker_shard_count=args.worker_shard_count,
        worker_shard_index=args.worker_shard_index,
    )
    worker_indices = _shuffled_indices(worker_indices, args.scan_seed)
    claim_dir = Path(args.claim_dir) if args.claim_dir else (output_dir / ".claims")
    claim_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    if args.model_backend == "vllm_local":
        if args.mode in {"solver", "both", "workflow"}:
            print("[backend] solver:", args.vllm_model_path, "@ <in-process vllm>")
        if args.mode in {"formalizer", "both", "workflow"}:
            print("[backend] formalizer:", args.vllm_formalizer_model_path, "@ <in-process vllm>")
    else:
        print("[backend] solver:", args.solver_model, "@", args.solver_base_url or "<local_hf>")
    print("[context] mode:", args.context_mode)
    if formalize_model is not None:
        print("[backend] formalizer:", args.formalize_model, "@", args.formalize_base_url or "<local_hf>")
    print("[scheduler] claim_dir=", claim_dir)
    print(
        "[scheduler] worker_scan_plan=",
        json.dumps(
            {
                "worker_id": args.worker_id,
                "worker_shard": (
                    f"{args.worker_shard_index}/{args.worker_shard_count}"
                    if args.worker_shard_count > 0
                    else "all"
                ),
                "scan_seed": args.scan_seed,
                "total_source_files": len(source_files),
                "assigned_files": len(worker_indices),
                "first_files": [source_files[idx].name for idx in worker_indices[:10]],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    stage_key = "_".join(_stage_names_for_mode(args.mode))

    for task_position, file_idx in enumerate(worker_indices, start=1):
        source_path = source_files[file_idx]
        out_path = output_dir / source_path.name
        task = source_path.stem
        if args.skip_existing and _checkpoint_is_completed(out_path, args.mode):
            print(f"[{task_position}/{len(worker_indices)}] skip completed {args.mode} {out_path.name}")
            continue

        claim_path = _try_claim_task(
            claim_dir=claim_dir,
            output_dir=output_dir,
            stem=task,
            mode=args.mode,
            claim_timeout_sec=args.claim_timeout_sec,
        )
        if claim_path is None:
            print(f"[{task_position}/{len(worker_indices)}] skip claimed/done {task}")
            continue

        task_rows: List[dict] = []
        heartbeat = ClaimHeartbeat(
            claim_path=claim_path,
            progress_path=progress_dir / f"{task}__{stage_key}.progress.json",
            worker_id=args.worker_id,
            interval_sec=args.claim_heartbeat_sec,
        )
        final_status = "error"
        started_at = time.time()
        try:
            input_path = out_path if out_path.exists() else source_path
            payload = _load_pickle(input_path)
            proof_items = payload.get("proof_items") or []
            print(f"[{task_position}/{len(worker_indices)}] {task}: {len(proof_items)} nodes")
            heartbeat.update(
                {
                    "task": task,
                    "status": "in_progress",
                    "worker_id": args.worker_id,
                    "started_at": started_at,
                    "n_nodes": len(proof_items),
                    "completed_node_stages": 0,
                }
            )
            heartbeat.start()

            if args.mode == "workflow":
                if formalize_model is solver_model and formalize_model is not None:
                    formalize_model.system_prompt = (ROOT_DIR / "prompts" / "lemma_formalizer.md").read_text(encoding="utf-8")
                task_rows.extend(
                    _run_workflow_vllm_for_task(
                        task=task,
                        proof_items=proof_items,
                        lean_server=lean_server,
                        formalize_model=formalize_model,
                        solver_model=solver_model,
                        selected_ids=selected_ids,
                        include_terminals=args.include_terminals,
                        n_rollouts=args.n_rollouts,
                        batch_size=args.vllm_batch_size,
                        verify_parallelism=args.verify_parallelism,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        original_proof=str(payload.get("nl_proof") or ""),
                        heartbeat=heartbeat,
                        started_at=started_at,
                    )
                )
            elif args.model_backend == "vllm_local":
                if args.mode in {"formalizer", "both"}:
                    if formalize_model is solver_model and formalize_model is not None:
                        formalize_model.system_prompt = (ROOT_DIR / "prompts" / "lemma_formalizer.md").read_text(encoding="utf-8")
                    task_rows.extend(
                        _run_formalizer_vllm_batch_for_task(
                            task=task,
                            proof_items=proof_items,
                            lean_server=lean_server,
                            vllm_model=formalize_model,
                            selected_ids=selected_ids,
                            include_terminals=args.include_terminals,
                            n_rollouts=args.n_rollouts,
                            batch_size=args.vllm_batch_size,
                            verify_parallelism=args.verify_parallelism,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            heartbeat=heartbeat,
                            started_at=started_at,
                        )
                    )
                if args.mode in {"solver", "both"}:
                    if solver_model is formalize_model and solver_model is not None:
                        solver_model.system_prompt = (ROOT_DIR / "prompts" / "lemma_prover.md").read_text(encoding="utf-8")
                    task_rows.extend(
                        _run_solver_vllm_batch_for_task(
                            task=task,
                            proof_items=proof_items,
                            lean_server=lean_server,
                            vllm_model=solver_model,
                            selected_ids=selected_ids,
                            n_rollouts=args.n_rollouts,
                            batch_size=args.vllm_batch_size,
                            verify_parallelism=args.verify_parallelism,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            context_mode=args.context_mode,
                            semantic_score_threshold=args.semantic_score_threshold,
                            heartbeat=heartbeat,
                            started_at=started_at,
                        )
                    )

            for item in ([] if args.model_backend == "vllm_local" else proof_items):
                node_id = str(getattr(item, "id", ""))
                if selected_ids is not None and node_id not in selected_ids:
                    continue

                terminal = _is_terminal(item)
                if args.mode in {"formalizer", "both"} and (args.include_terminals or not terminal):
                    logs: List[dict] = []
                    started = time.time()
                    formalization = run_formalizer_prompt(
                        item,
                        lean_server=lean_server,
                        all_items=proof_items,
                        model_manager=formalize_model,
                        logs=logs,
                        max_retries=1,
                        rollout_n=args.n_rollouts,
                        rollout_parallelism=args.rollout_parallelism,
                        previous_context=False,
                        original_proof="",
                    )
                    formalization["reroll_started_at"] = started
                    formalization["usage_records"] = _usage_records(logs)
                    formalization["usage_summary"] = _usage_summary(formalization["usage_records"])
                    item.formalization = formalization
                    row = _stage_summary(
                        task,
                        item,
                        "formalizer",
                        formalization,
                        formalization["usage_summary"],
                    )
                    task_rows.append(row)
                    heartbeat.update(
                        {
                            "task": task,
                            "status": "in_progress",
                            "worker_id": args.worker_id,
                            "started_at": started_at,
                            "n_nodes": len(proof_items),
                            "current_node_id": node_id,
                            "current_stage": "formalizer",
                            "completed_node_stages": len(task_rows),
                        }
                    )
                    print(
                        f"  {node_id} formalizer pass_rate={row['success_rate']:.3f} "
                        f"cost=${row['cost_usd']:.4f}"
                    )

                if args.mode in {"solver", "both"} and not terminal:
                    if not _has_formalization(item):
                        print(f"  {node_id} solver skipped: missing reusable formalization")
                        continue
                    logs = []
                    started = time.time()
                    solved = run_solver_prompt(
                        item,
                        lean_server=lean_server,
                        model_manager=solver_model,
                        logs=logs,
                        max_retries=1,
                        rollout_n=args.n_rollouts,
                        rollout_parallelism=args.rollout_parallelism,
                    )
                    solved["reroll_started_at"] = started
                    solved["usage_records"] = _usage_records(logs)
                    solved["usage_summary"] = _usage_summary(solved["usage_records"])
                    item.solved_lemma = solved
                    row = _stage_summary(task, item, "solver", solved, solved["usage_summary"])
                    task_rows.append(row)
                    heartbeat.update(
                        {
                            "task": task,
                            "status": "in_progress",
                            "worker_id": args.worker_id,
                            "started_at": started_at,
                            "n_nodes": len(proof_items),
                            "current_node_id": node_id,
                            "current_stage": "solver",
                            "completed_node_stages": len(task_rows),
                        }
                    )
                    print(
                        f"  {node_id} solver verify_rate={row['success_rate']:.3f} "
                        f"cost=${row['cost_usd']:.4f}"
                    )

            existing_metadata = payload.get("run_metadata") or {}
            completed_stages = _metadata_completed_stages(existing_metadata)
            completed_stages.update(_stage_names_for_mode(args.mode))
            payload["run_metadata"] = {
                **existing_metadata,
                "node_reroll_status": "completed",
                "node_reroll_mode": args.mode,
                "node_reroll_completed_stages": sorted(completed_stages),
                "node_reroll_model_backend": args.model_backend,
                "node_reroll_n_rollouts": int(args.n_rollouts),
                "node_reroll_rollout_parallelism": int(args.rollout_parallelism),
                "node_reroll_verify_parallelism": int(args.verify_parallelism),
                "node_reroll_uses_dependency_context": bool(args.model_backend == "vllm_local"),
                "node_reroll_context_mode": args.context_mode,
                "node_reroll_source_pickle": str(source_path),
                "node_reroll_worker_id": args.worker_id,
                "node_reroll_completed_at": time.time(),
            }
            payload["llm_call_logs"] = []
            _save_pickle(out_path, payload)
            _write_task_summary(summary_dir, task, task_rows, summary_key=f"{task}__{stage_key}")
            heartbeat.update(
                {
                    "task": task,
                    "status": "completed",
                    "worker_id": args.worker_id,
                    "started_at": started_at,
                    "completed_at": time.time(),
                    "n_nodes": len(proof_items),
                    "completed_node_stages": len(task_rows),
                }
            )
            all_rows.extend(task_rows)
            final_status = "completed"
        finally:
            heartbeat.stop()
            if final_status == "completed":
                (claim_path / "done").write_text(
                    f"status=completed time={time.time()} worker_id={args.worker_id}\n",
                    encoding="utf-8",
                )
                _write_done_marker(
                    claim_dir,
                    task,
                    {
                        "status": "completed",
                        "stage": stage_key,
                        "completed_at": time.time(),
                        "worker_id": args.worker_id,
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                    },
                    stage_key=stage_key,
                )
            else:
                _remove_done_marker(claim_dir, task, stage_key=stage_key)
                shutil.rmtree(claim_path, ignore_errors=True)

    rows = _finalize_summaries(output_dir)
    total_cost = sum(float(row.get("cost_usd", 0.0)) for row in all_rows)
    print(f"Saved rerolled pickles: {output_dir}")
    print(f"Saved summary JSONL: {summary_jsonl}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Worker node-stage rows: {len(all_rows)}")
    print(f"Finalized node-stage rows: {len(rows)}")
    print(f"Worker estimated API cost from logged tokens: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
