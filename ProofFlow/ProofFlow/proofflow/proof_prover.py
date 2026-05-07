import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import httpx

from .lean_check import LeanServer, process_lean_string
from .utils import LLMManager


def _is_condition_or_definition(item: Any) -> bool:
    item_id = str(getattr(item, "id", ""))
    if item_id.startswith(("tc_", "def_")):
        return True
    return type(item).__name__ in {"TheoremCondition", "Definition"}


def extract_code_validate(text_input, lean_server):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r"```lean4\n(.*?)\n```", text_input, re.DOTALL)
        if not matches:
            raise ValueError("No Lean 4 code block found.")
    except RuntimeError as e:
        return f"Error during code extraction: {str(e)}. Is ```lean4 ``` written?"

    response = matches[-1].strip()
    response = process_lean_string(response)  # add missing imports

    try:
        lean_pass, lean_verify, error_msg = lean_server.check_lean_string(response)
    except Exception as e:
        raise RuntimeError("Error during Lean code verification " + str(e))

    return {
        "lean_code": response,
        "lean_pass": lean_pass,
        "lean_verify": lean_verify,
        "error_msg": error_msg,
    }  # None if lean_verify else error_msg


def run_solver_prompt(
    item: dict,
    lean_server: LeanServer,
    model_manager: LLMManager,
    logs=None,
    max_retries: int = 3,
    rollout_n: int = 1,
    rollout_parallelism: int = 1,
    prove_negation: bool = False,
) -> tuple:
    """
    Builds the prompt string for the item and calls the LLM API.
    Uses the .md file as a system prompt.
    If the item has dependencies, appends their filtered dicts to the prompt string.
    Returns the parsed and validated JSON, with retry logic if validation fails.
    """

    # Deal with non-correct input cases
    if _is_condition_or_definition(item):
        return {}
    if not item.formalization["lean_code"]:
        return {}

    if prove_negation:
        user_prompt_content = f"""
Your task is **not to prove the given theorem/lemma, but to disprove it by proving its logical negation**.

This is the original lemma/theorem statement I want you to refute:
{item.statement}

Below is the Lean 4 code for the statement. 
You must instead negate the goal and then attempt to prove that negation in Lean 4. 
If the statement cannot be directly negated syntactically, carefully construct the logically equivalent negation.

Please output only valid Lean 4 code with the negated theorem and a proof attempt.

```lean4
{item.formalization["lean_code"]}
```"""

    else:
        user_prompt_content = f"""
This is the lemma/theorem I want you to prove:
{item.statement}

Complete the following Lean 4 code (**do not remove imports**):

```lean4
{item.formalization["lean_code"]}
```

You can adapt previous lean4 lemma statement to fit the goal, specially if you encounter errors.
"""
    if not item.formalization["lean_pass"]:
        user_prompt_content += "/n The previous Lean4 code I sent you contains errors. Please take that into account."

    messages = [{"role": "user", "content": user_prompt_content}]

    results = {
        "lean_code": "",
        "lean_pass": False,
        "lean_verify": False,
        "error_msg": "failure to get any response from LLM",
    }

    if rollout_n > 1:
        rollout_records = []
        base_messages = list(messages)

        def run_single_rollout(rollout_id: int) -> Dict[str, Any]:
            total_t0 = time.perf_counter()
            infer_latency_sec = 0.0
            verify_latency_sec = 0.0
            infer_t0 = total_t0
            verify_t0 = None
            candidate = {
                "lean_code": "",
                "lean_pass": False,
                "lean_verify": False,
                "error_msg": "failure to get any response from LLM",
            }
            try:
                infer_t0 = time.perf_counter()
                response, _ = model_manager.call_llm(list(base_messages), logs=logs)
                infer_latency_sec = time.perf_counter() - infer_t0
                verify_t0 = time.perf_counter()
                candidate = extract_code_validate(response, lean_server)
                verify_latency_sec = time.perf_counter() - verify_t0
            except Exception as e:
                if verify_t0 is not None:
                    verify_latency_sec = time.perf_counter() - verify_t0
                elif infer_latency_sec == 0.0:
                    infer_latency_sec = time.perf_counter() - infer_t0
                candidate["error_msg"] = str(e)

            total_latency_sec = time.perf_counter() - total_t0
            return {
                "rollout_id": rollout_id,
                "infer_latency_sec": infer_latency_sec,
                "verify_latency_sec": verify_latency_sec,
                "total_latency_sec": total_latency_sec,
                "latency_sec": total_latency_sec,
                "lean_pass": bool(candidate.get("lean_pass", False)),
                "lean_verify": bool(candidate.get("lean_verify", False)),
                "lean_code": candidate.get("lean_code", ""),
                "error_msg": candidate.get("error_msg", ""),
            }

        effective_parallelism = max(1, min(int(rollout_parallelism), rollout_n))
        if effective_parallelism == 1:
            for rollout_id in range(rollout_n):
                rollout_records.append(run_single_rollout(rollout_id))
        else:
            with ThreadPoolExecutor(max_workers=effective_parallelism) as executor:
                rollout_records.extend(executor.map(run_single_rollout, range(rollout_n)))

        successful = [r for r in rollout_records if r["lean_verify"]]
        if successful:
            selected = min(successful, key=lambda r: r["latency_sec"])
        else:
            selected = min(rollout_records, key=lambda r: r["latency_sec"])

        results = {
            "lean_code": selected["lean_code"],
            "lean_pass": selected["lean_pass"],
            "lean_verify": selected["lean_verify"],
            "error_msg": selected["error_msg"],
            "infer_latency_sec": selected["infer_latency_sec"],
            "verify_latency_sec": selected["verify_latency_sec"],
            "total_latency_sec": selected["total_latency_sec"],
            "latency_sec": selected["latency_sec"],
            "tries": rollout_n,
            "selected_rollout_id": selected["rollout_id"],
            "rollout_records": rollout_records,
            "verify_rate": (sum(1 for r in rollout_records if r["lean_verify"]) / float(rollout_n)),
            "attempt_history": [],
        }
        return results

    attempt_history = []
    for attempt in range(max_retries):
        try:
            response, messages = model_manager.call_llm(messages, logs=logs)
        except httpx.TimeoutException as e:
            print("OpenAI request failed:", e)

        # Try to validate the response
        try:
            results = extract_code_validate(response, lean_server)
            results["tries"] = attempt + 1
            results["attempt_history"] = attempt_history

            # check if lean is correct -> if yes end loop here
            if results["lean_verify"]:
                return results
            else:  # if not ajust prompt_str
                messages.append(
                    {
                        "role": "user",
                        "content": f"Lean error/warnings: "
                        + str(results["error_msg"])
                        + "\n\n Based on these errors, please correct the previous response. ",
                    }
                )
        except ValueError as e:
            messages.append(
                {
                    "role": "user",
                    "content": f"\n\nError: "
                    + str(e)
                    + "\n\n Based on these errors, please correct the previous response. ",
                }
            )

        attempt_history.append(results)

    return results
