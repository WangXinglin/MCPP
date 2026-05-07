import time
import os

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class ChatModel:
    def __init__(
        self,
        model_path,
        tensor_parallel_size=1,
        n_rollouts_hint=1,
        max_num_seqs=0,
        gpu_memory_utilization=0.9,
    ):
        """
        Initialize the model and tokenizer
        :param model_path: model path
        :param tensor_parallel_size: GPU parallel number, default is 1
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = 48 * 1024
        self.n_rollouts_hint = max(1, int(n_rollouts_hint))
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_num_seqs = int(max_num_seqs)
        self.use_internal_timing_engine = os.environ.get("CODEFLOW_USE_INTERNAL_VLLM_TIMING", "0") == "1"
        self.enforce_eager = os.environ.get("CODEFLOW_VLLM_ENFORCE_EAGER", "0") == "1"
        self.disable_custom_all_reduce = os.environ.get("CODEFLOW_VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1") != "0"
        self.rollout_chunk_size = int(os.environ.get("CODEFLOW_ROLLOUT_CHUNK_SIZE", "64") or "0")
        self._request_counter = 0
        if self.max_num_seqs <= 0:
            self.max_num_seqs = self._auto_max_num_seqs()
        print(
            f"[vLLM] tensor_parallel_size={self.tensor_parallel_size}, "
            f"max_num_seqs={self.max_num_seqs}, "
            f"gpu_memory_utilization={self.gpu_memory_utilization:.2f}, "
            f"rollout_chunk_size={self.rollout_chunk_size}, "
            f"enforce_eager={self.enforce_eager}, "
            f"disable_custom_all_reduce={self.disable_custom_all_reduce}, "
            f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}, "
            f"NCCL_CUMEM_ENABLE={os.environ.get('NCCL_CUMEM_ENABLE')}, "
            f"NCCL_CUMEM_HOST_ENABLE={os.environ.get('NCCL_CUMEM_HOST_ENABLE')}"
        )
        self.llm, self.tokenizer = self._load_model_and_tokenizer()

    def _auto_max_num_seqs(self):
        # Heuristic for throughput-friendly defaults by rollout count and GPU memory tier.
        base = max(4, min(64, self.n_rollouts_hint))
        cap = 32
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if total_gb >= 110:
                    cap = 64
                elif total_gb >= 70:
                    cap = 48
                elif total_gb >= 40:
                    cap = 32
                else:
                    cap = 16
        except Exception:
            cap = 32
        return int(max(8, min(base, cap)))

    def _load_model_and_tokenizer(self):
        """
        Loading the model and tokenizer
        """
        # Initialize the vLLM's LLM
        llm = LLM(
            model=self.model_path,
            # max_model_len is total context budget (prompt + generation),
            # so it should be larger than target max_tokens.
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=self.enforce_eager,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
        )

        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        return llm, tokenizer

    def format_chat(self, messages):
        """
        Format the message list into the input format required by the model
        :param messages: message list, format is [{'role': 'user', 'content': '...'}, ...]
        :return: formatted prompt
        """
        # English note tokenizer English note apply_chat_template English note
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt

    def _effective_max_tokens(self, prompt, requested_max_tokens):
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
        prompt_tokens = len(prompt_ids)
        available_for_generation = max(1, self.max_model_len - prompt_tokens)
        return min(int(requested_max_tokens), available_for_generation)

    def generate(self, messages, max_tokens=38912, temperature=0.6, top_p=0.95, top_k=20, n=1):
        """
        Generate model responses
        :param messages: message list, format is [{'role': 'user', 'content': '...'}, ...]
        :param max_tokens: maximum number of tokens generated
        :param temperature: randomness of generation
        :param top_p: sample probability cutoff
        :param top_k: top-k cutoff
        :return: model generation results
        """
        # Format the message list into the input format required by the model
        prompt = self.format_chat(messages)
        effective_max_tokens = self._effective_max_tokens(prompt, max_tokens)

        # Set sampling parameters
        sampling_params = SamplingParams(
            max_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
        )

        # Making inferences
        output = self.llm.generate(prompt, sampling_params)

        # Returns the generated result
        return output

    def generate_rollouts_with_timings(self, messages, max_tokens=38912, temperature=0.6, top_p=0.95, top_k=20, n=1):
        """
        Generate N rollouts as N independent vLLM requests and return per-request
        client-observed inference latency. This keeps vLLM's internal batching, but
        avoids hiding all rollout finish times behind one SamplingParams(n=N) request.
        """
        prompt = self.format_chat(messages)
        effective_max_tokens = self._effective_max_tokens(prompt, max_tokens)
        sampling_params = SamplingParams(
            max_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=1,
        )

        engine = getattr(self.llm, "llm_engine", None)
        if (
            not self.use_internal_timing_engine
            or engine is None
            or not hasattr(engine, "add_request")
            or not hasattr(engine, "step")
        ):
            remaining = int(n)
            rollout_offset = 0
            all_outputs = []
            max_chunk = self.rollout_chunk_size if self.rollout_chunk_size > 0 else remaining
            while remaining > 0:
                chunk_n = min(remaining, max_chunk)
                chunk_t0 = time.perf_counter()
                generated = self.llm.generate(prompt, SamplingParams(
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=chunk_n,
                ))
                chunk_latency = time.perf_counter() - chunk_t0
                outputs = generated[0].outputs if generated else []
                for idx, sampled in enumerate(outputs):
                    all_outputs.append(
                        {
                            "rollout_id": rollout_offset + idx,
                            "text": getattr(sampled, "text", ""),
                            "token_ids": getattr(sampled, "token_ids", None),
                            "infer_latency_sec": None,
                            "infer_batch_latency_sec": chunk_latency,
                            "latency_observation": "public_generate_batch_chunk",
                        }
                    )
                rollout_offset += chunk_n
                remaining -= chunk_n
            return all_outputs

        prompt_token_ids = self.tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
        batch_t0 = time.perf_counter()
        request_to_rollout = {}
        results = {}
        prefix = f"rollout-{id(self)}-{self._request_counter}"
        self._request_counter += int(n)

        for rollout_id in range(int(n)):
            request_id = f"{prefix}-{rollout_id}"
            request_to_rollout[request_id] = rollout_id
            engine.add_request(request_id, prompt, sampling_params)

        while engine.has_unfinished_requests():
            request_outputs = engine.step()
            now = time.perf_counter()
            for request_output in request_outputs:
                request_id = getattr(request_output, "request_id", None)
                if request_id not in request_to_rollout or not getattr(request_output, "finished", False):
                    continue
                rollout_id = request_to_rollout[request_id]
                if rollout_id in results:
                    continue
                outputs = getattr(request_output, "outputs", []) or []
                sampled = outputs[0] if outputs else None
                results[rollout_id] = {
                    "rollout_id": rollout_id,
                    "text": getattr(sampled, "text", "") if sampled is not None else "",
                    "token_ids": getattr(sampled, "token_ids", None) if sampled is not None else None,
                    "infer_latency_sec": now - batch_t0,
                    "infer_batch_latency_sec": None,
                    "prompt_token_ids": getattr(request_output, "prompt_token_ids", None) or prompt_token_ids,
                    "latency_observation": "per_request_finished_time",
                }

        batch_latency = time.perf_counter() - batch_t0
        ordered = []
        for rollout_id in range(int(n)):
            record = results.get(rollout_id)
            if record is None:
                record = {
                    "rollout_id": rollout_id,
                    "text": "",
                    "token_ids": None,
                    "infer_latency_sec": None,
                    "latency_observation": "missing_request_output",
                }
            record["infer_batch_latency_sec"] = batch_latency
            ordered.append(record)
        return ordered

    def generate_batch(self, batch_messages, max_tokens=38912, temperature=0.6, top_p=0.95, top_k=20, n=1):
        prompts = [self.format_chat(messages) for messages in batch_messages]
        effective_max_tokens = max(1, min(self._effective_max_tokens(prompt, max_tokens) for prompt in prompts))

        sampling_params = SamplingParams(
            max_tokens=effective_max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
        )
        return self.llm.generate(prompts, sampling_params)
