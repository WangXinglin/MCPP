import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import httpx

from .lean_check import LeanServer, process_lean_string
from .utils import LLMManager, remove_imports


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
    except RuntimeError as e:
        raise ValueError("Error during Lean code verification " + str(e))

    return {
        "lean_code": response,
        "lean_pass": lean_pass,
        "error_msg": [] if lean_pass else error_msg,
    }


def run_formalizer_prompt(
    item: dict,
    lean_server: LeanServer,
    model_manager: LLMManager,
    all_items: list = None,
    logs=None,
    max_retries: int = 3,
    rollout_n: int = 1,
    rollout_parallelism: int = 1,
    previous_context: bool = True,  # when formalizing step i provide formalized code of dependencies
    original_proof: str = "",  # when formalizing step i provide original proof
    dependency_context_map: Optional[Dict[str, Dict[str, Any]]] = None,
    dependency_context_mode: str = "verified_only",
) -> tuple:
    """
    Builds the prompt string for the item and calls the LLM API.
    Uses the .md file as a system prompt.
    If the item has dependencies, appends their filtered dicts to the prompt string.
    Returns the parsed and validated JSON, with retry logic if validation fails.
    """

    is_condition_or_def = item.id.startswith("tc_") or item.id.startswith(
        "def_"
    )  # is it theorem condition or not?
    lemma_header = f"lemma {item.id}"
    dependencies = item.dependencies

    def can_use_formalization_context(dep_id: str) -> bool:
        return dep_id.startswith("tc_") or dep_id.startswith("def_") or (
            dependency_context_mode == "allow_formalization_fallback"
        )

    if not is_condition_or_def:
        # Combine all parts
        user_prompt_content = f"""Please autoformalize the following natural language problem proof step in Lean 4.
Use the following lemma name: {lemma_header}
The natural language statement is: {item.statement}
The dependencies are: {dependencies}

This is the  lean code skeleton you need to use:

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
        user_prompt_content = rf"""Please autoformalize the following natural language theorem condition in Lean 4.
Use the following name: {item.id}

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
    # Context can be: previous lean4 code or just goal statements and/or NL statement and/or original proof
    if previous_context:
        previous_context_str = [
            f"\n\n This proof step depend on previous proof steps, namely steps {dependencies}."
        ]
        previous_context_str.append(
            "Please make use use of their formal lean4 code, which contains relevant lean4 hypothesis and type declarations you may use:"
        )
        for d in all_items:
            if d.id in dependencies:
                previous_context_str.append("/n")  # Step {d.id}:")
                context_record = None
                if dependency_context_map:
                    context_record = dependency_context_map.get(d.id)
                if context_record and context_record.get("lean_code"):
                    previous_context_str.append(
                        remove_imports(context_record["lean_code"])
                    )
                elif (
                    hasattr(d, "solved_lemma")
                    and d.solved_lemma
                    and d.solved_lemma.get("lean_code")
                    and d.solved_lemma.get("lean_verify")
                ):
                    previous_context_str.append(
                        remove_imports(d.solved_lemma["lean_code"])
                    )
                elif (
                    hasattr(d, "formalization")
                    and d.formalization
                    and d.formalization["lean_code"]
                    and d.formalization["lean_pass"]
                    and can_use_formalization_context(d.id)
                ):  # check if lean code exists and it runs!
                    previous_context_str.append(
                        remove_imports(d.formalization["lean_code"])
                    )
                else:
                    raise ValueError(
                        f"Missing verified dependency context for step {d.id} while formalizing {item.id}."
                    )
        previous_context_str.append(
            "/n Focus on the original formalization task I gave you and use the prebious Lean codes, extra context, type declarations, variables domains, etc. You can assume the information is correct. Make use of it!"
        )
        previous_context_str = "/n".join(previous_context_str)
        user_prompt_content += previous_context_str

    if original_proof:
        user_prompt_content += "\n\n This formalization task is a proof step which is part of a larger full proof given next:\n"
        user_prompt_content += original_proof
        user_prompt_content += "\nThe full proof may contain extra missing information that you need, specially variable types and domains (e.g 'r' is real and positive). Make use of it, specially if you encounter errors."
        user_prompt_content += "\nHowever, please focus on the original formalization task I gave you and use the previous full proof for extra context only."

    messages = [{"role": "user", "content": user_prompt_content}]

    formalization = {
        "lean_code": "",
        "lean_pass": False,
        "error_msg": "Failure to get a valid formalization from the LLM",
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
                "error_msg": "Failure to get a valid formalization from the LLM",
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

        successful = [r for r in rollout_records if r["lean_pass"]]
        if successful:
            selected = min(successful, key=lambda r: r["latency_sec"])
        else:
            selected = min(rollout_records, key=lambda r: r["latency_sec"])

        formalization = {
            "lean_code": selected["lean_code"],
            "lean_pass": selected["lean_pass"],
            "error_msg": [] if selected["lean_pass"] else selected["error_msg"],
            "infer_latency_sec": selected["infer_latency_sec"],
            "verify_latency_sec": selected["verify_latency_sec"],
            "total_latency_sec": selected["total_latency_sec"],
            "latency_sec": selected["latency_sec"],
            "tries": rollout_n,
            "selected_rollout_id": selected["rollout_id"],
            "rollout_records": rollout_records,
            "pass_rate": (sum(1 for r in rollout_records if r["lean_pass"]) / float(rollout_n)),
            "attempt_history": [],
        }
        return formalization

    attempt_history = []

    for attempt in range(max_retries):
        try:
            response, messages = model_manager.call_llm(messages, logs=logs)
        except httpx.TimeoutException as e:
            print("OpenAI request failed:", e)

        # Try to validate the response
        try:
            formalization = extract_code_validate(response, lean_server)
            formalization["tries"] = attempt + 1
            formalization["attempt_history"] = attempt_history

            # check if lean is correct -> if yes end loop here
            if formalization["lean_pass"]:
                return formalization
            else:  # if not ajust prompt_str
                messages.append(
                    {
                        "role": "user",
                        "content": f"Lean error: "
                        + str(formalization["error_msg"])
                        + "\n\nBased on the error, please correct the previous response. ",
                    }
                )

        except ValueError as e:
            messages.append(
                {
                    "role": "user",
                    "content": f"\n\nError: "
                    + str(e)
                    + "\n\nBased on the error, please correct the previous response. ",
                }
            )

        attempt_history.append(formalization)

    return formalization
