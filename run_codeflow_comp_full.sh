#!/usr/bin/env bash
set -euo pipefail

# Full CodeFlowBench pipeline (two-phase ready):
# phase=prepare: only prepare subset/index files
# phase=test: run inference+harness+stat on prepared/full input
# phase=all: prepare (if needed) then test

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODEFLOW_DIR="$ROOT_DIR/codeflow"

MODE="${MODE:-local}"            # local | api
DATASET="${DATASET:-comp}"       # comp | repo
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-$CODEFLOW_DIR/models/$MODEL_NAME}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
CLAIM_DIR="${CLAIM_DIR:-}"
CLAIM_TIMEOUT_SEC="${CLAIM_TIMEOUT_SEC:-1800}"
CLAIM_HEARTBEAT_SEC="${CLAIM_HEARTBEAT_SEC:-30}"
WORKER_ID="${WORKER_ID:-}"
UPSTREAM_ROLLOUT_DIR="${UPSTREAM_ROLLOUT_DIR:-}"
UPSTREAM_VERIFY_DIR="${UPSTREAM_VERIFY_DIR:-}"
VERIFY_UPSTREAM_ON_MISS="${VERIFY_UPSTREAM_ON_MISS:-0}"
VERIFY_TIMEOUT_SEC="${VERIFY_TIMEOUT_SEC:-5.0}"
VERIFY_TEMP_ROOT="${VERIFY_TEMP_ROOT:-$ROOT_DIR/temp/inference_upstream_verify}"
VERIFY_MODE="${VERIFY_MODE:-harness}" # harness | analyze_rollouts
VERIFY_PROBLEM_WORKERS="${VERIFY_PROBLEM_WORKERS:-1}"
VERIFY_EVAL_WORKERS="${VERIFY_EVAL_WORKERS:-1}"
VERIFY_CLAIM_DIR="${VERIFY_CLAIM_DIR:-}"
VERIFY_KEEP_LLM_OUTPUTS="${VERIFY_KEEP_LLM_OUTPUTS:-1}"
GOLDEN_UPSTREAM_FOR_GEN="${GOLDEN_UPSTREAM_FOR_GEN:-0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PHASE="${PHASE:-all}"            # prepare | test | all
INDEX_FILE="${INDEX_FILE:-}"
SUBSET_FILE="${SUBSET_FILE:-}"
EXP_TAG="${EXP_TAG:-}"

if [[ "$DATASET" == "comp" ]]; then
  DEFAULT_INPUT_FILE="$CODEFLOW_DIR/data/codeflowbench_comp_test.json"
  HARNESS_SCRIPT="$CODEFLOW_DIR/run/multi_turn/harness.py"
elif [[ "$DATASET" == "repo" ]]; then
  DEFAULT_INPUT_FILE="$CODEFLOW_DIR/data/codeflowbench_repo.json"
  HARNESS_SCRIPT="$CODEFLOW_DIR/run/multi_turn/harness_repo.py"
else
  echo "Unsupported DATASET=$DATASET, expected comp or repo"
  exit 1
fi

INPUT_FILE="${INPUT_FILE:-$DEFAULT_INPUT_FILE}"

if [[ -n "$EXP_TAG" ]]; then
  TAG_SUFFIX="$EXP_TAG"
elif [[ -n "$INDEX_FILE" ]]; then
  idx_name="$(basename "$INDEX_FILE")"
  TAG_SUFFIX="${idx_name%.*}"
elif [[ "$SAMPLE_SIZE" -gt 0 ]]; then
  TAG_SUFFIX="n${SAMPLE_SIZE}_seed${SAMPLE_SEED}"
else
  TAG_SUFFIX="full"
fi

RUN_TAG="${MODEL_NAME}_${DATASET}_${TAG_SUFFIX}"

TMP_DIR="${TMP_DIR:-$CODEFLOW_DIR/output/${RUN_TAG}_multi_turn_temp}"

INFER_JSON="$CODEFLOW_DIR/output/inference/${RUN_TAG}_multi_turn.json"
HARNESS_JSON="$CODEFLOW_DIR/output/harness/${RUN_TAG}_multi_turn.json"
STAT_JSON="$CODEFLOW_DIR/result/${RUN_TAG}_multi_turn.json"
DAG_JSONL="${DAG_JSONL:-$ROOT_DIR/pregenerated_dags/dag_pool.jsonl}"

if [[ -z "$SUBSET_FILE" ]]; then
  if [[ -n "$INDEX_FILE" ]]; then
    SUBSET_FILE="$CODEFLOW_DIR/output/subsets/${DATASET}_${TAG_SUFFIX}.json"
  elif [[ "$SAMPLE_SIZE" -gt 0 ]]; then
    SUBSET_FILE="$CODEFLOW_DIR/output/subsets/${DATASET}_n${SAMPLE_SIZE}_seed${SAMPLE_SEED}.json"
  fi
fi

if [[ "$PHASE" != "prepare" && "$PHASE" != "test" && "$PHASE" != "all" ]]; then
  echo "Unsupported PHASE=$PHASE, expected prepare/test/all"
  exit 1
fi
if [[ "$VERIFY_MODE" != "harness" && "$VERIFY_MODE" != "analyze_rollouts" ]]; then
  echo "Unsupported VERIFY_MODE=$VERIFY_MODE, expected harness or analyze_rollouts"
  exit 1
fi

EFFECTIVE_INPUT_FILE="$INPUT_FILE"

prepare_subset() {
  if [[ -n "$INDEX_FILE" ]]; then
    echo "[prepare] Build subset from indices file"
    python "$ROOT_DIR/sample_codeflow_tasks.py" \
      --input "$INPUT_FILE" \
      --output "$SUBSET_FILE" \
      --indices_file "$INDEX_FILE"
    EFFECTIVE_INPUT_FILE="$SUBSET_FILE"
    return
  fi

  if [[ "$SAMPLE_SIZE" -gt 0 ]]; then
    local out_indices="$CODEFLOW_DIR/output/subsets/${DATASET}_n${SAMPLE_SIZE}_seed${SAMPLE_SEED}.indices.json"
    echo "[prepare] Sampling subset from dataset=$DATASET n=$SAMPLE_SIZE seed=$SAMPLE_SEED"
    python "$ROOT_DIR/sample_codeflow_tasks.py" \
      --input "$INPUT_FILE" \
      --output "$SUBSET_FILE" \
      --n "$SAMPLE_SIZE" \
      --seed "$SAMPLE_SEED" \
      --output_indices "$out_indices"
    INDEX_FILE="$out_indices"
    EFFECTIVE_INPUT_FILE="$SUBSET_FILE"
    return
  fi

  EFFECTIVE_INPUT_FILE="$INPUT_FILE"
}

if [[ "$PHASE" == "prepare" || "$PHASE" == "all" ]]; then
  prepare_subset
fi

if [[ "$PHASE" == "prepare" ]]; then
  echo "Prepare completed."
  echo "Input file:   $INPUT_FILE"
  if [[ -n "$SUBSET_FILE" ]]; then
    echo "Subset file:  $SUBSET_FILE"
  fi
  if [[ -n "$INDEX_FILE" ]]; then
    echo "Index file:   $INDEX_FILE"
  fi
  echo "Run tag:      $RUN_TAG"
  exit 0
fi

if [[ "$PHASE" == "test" ]]; then
  if [[ -n "$INDEX_FILE" ]]; then
    prepare_subset
  elif [[ -n "$SUBSET_FILE" ]]; then
    if [[ ! -f "$SUBSET_FILE" ]]; then
      echo "SUBSET_FILE not found: $SUBSET_FILE"
      exit 1
    fi
    EFFECTIVE_INPUT_FILE="$SUBSET_FILE"
  elif [[ "$SAMPLE_SIZE" -gt 0 ]]; then
    echo "PHASE=test with SAMPLE_SIZE requires prepared INDEX_FILE or existing SUBSET_FILE."
    echo "Run PHASE=prepare first to freeze subset for parallel workers."
    exit 1
  else
    EFFECTIVE_INPUT_FILE="$INPUT_FILE"
  fi
fi

echo "[1/4] Inference mode=$MODE model=$MODEL_NAME dataset=$DATASET"
echo "      input file: $EFFECTIVE_INPUT_FILE"
echo "      rollouts per subtask: $N_ROLLOUTS"
if [[ "$MODE" == "local" ]]; then
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Model path not found: $MODEL_PATH"
    echo "Set MODEL_PATH or switch MODE=api."
    exit 1
  fi

  INFER_ARGS=(
    --model_path "$MODEL_PATH"
    --input_file "$EFFECTIVE_INPUT_FILE"
    --output_dir "$TMP_DIR"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}"
    --n_rollouts "$N_ROLLOUTS"
    --max_num_seqs "$MAX_NUM_SEQS"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --claim-timeout-sec "$CLAIM_TIMEOUT_SEC"
    --claim-heartbeat-sec "$CLAIM_HEARTBEAT_SEC"
    --verify-timeout-sec "$VERIFY_TIMEOUT_SEC"
    --verify-temp-root "$VERIFY_TEMP_ROOT"
  )
  if [[ -n "$CLAIM_DIR" ]]; then
    INFER_ARGS+=(--claim-dir "$CLAIM_DIR")
  fi
  if [[ -n "$WORKER_ID" ]]; then
    INFER_ARGS+=(--worker-id "$WORKER_ID")
  fi
  if [[ -n "$UPSTREAM_ROLLOUT_DIR" ]]; then
    INFER_ARGS+=(--upstream-rollout-dir "$UPSTREAM_ROLLOUT_DIR")
  fi
  if [[ -n "$UPSTREAM_VERIFY_DIR" ]]; then
    INFER_ARGS+=(--upstream-verify-dir "$UPSTREAM_VERIFY_DIR")
  fi
  if [[ "$VERIFY_UPSTREAM_ON_MISS" == "1" ]]; then
    INFER_ARGS+=(--verify-upstream-on-miss)
  fi
  if [[ "$GOLDEN_UPSTREAM_FOR_GEN" == "1" ]]; then
    INFER_ARGS+=(--golden-upstream-for-gen)
  fi

  python "$CODEFLOW_DIR/run/multi_turn/inference_local.py" "${INFER_ARGS[@]}"
else
  : "${API_KEY:?API_KEY is required when MODE=api}"
  : "${API_URL:?API_URL is required when MODE=api}"

  python "$CODEFLOW_DIR/run/multi_turn/inference_api.py" \
    --model_name "$MODEL_NAME" \
    --input_file "$EFFECTIVE_INPUT_FILE" \
    --output_dir "$TMP_DIR" \
    --api_key "$API_KEY" \
    --api_url "$API_URL" \
    --n_rollouts "$N_ROLLOUTS"
fi

python "$CODEFLOW_DIR/run/multi_turn/combined.py" \
  --model_name "$RUN_TAG" \
  --combined_dir "$TMP_DIR"

echo "[2/4] Harness (CodeFlowBench-$DATASET)"
if [[ "$VERIFY_MODE" == "analyze_rollouts" ]]; then
  ANALYSIS_JSON="$CODEFLOW_DIR/result/${RUN_TAG}_rollout_analysis.json"
  ANALYZE_ARGS=(
    --input "$INFER_JSON"
    --output "$ANALYSIS_JSON"
    --save-evaluated-dir "$TMP_DIR"
    --problem-workers "$VERIFY_PROBLEM_WORKERS"
    --eval-workers "$VERIFY_EVAL_WORKERS"
    --timeout-sec "$VERIFY_TIMEOUT_SEC"
  )
  if [[ -n "$VERIFY_CLAIM_DIR" ]]; then
    ANALYZE_ARGS+=(--claim-dir "$VERIFY_CLAIM_DIR")
  fi
  if [[ -n "$WORKER_ID" ]]; then
    ANALYZE_ARGS+=(--worker-id "$WORKER_ID")
  fi
  if [[ "$VERIFY_KEEP_LLM_OUTPUTS" == "1" ]]; then
    ANALYZE_ARGS+=(--keep-llm-outputs)
  fi
  python "$CODEFLOW_DIR/run/multi_turn/analyze_rollouts.py" "${ANALYZE_ARGS[@]}"
else
  python "$HARNESS_SCRIPT" \
    --input_path "$INFER_JSON" \
    --output_dir "$TMP_DIR" \
    --model_name "$MODEL_NAME"
fi

python "$CODEFLOW_DIR/run/multi_turn/combined.py" \
  --model_name "$RUN_TAG" \
  --combined_dir "$TMP_DIR" \
  --harness

echo "[3/4] Stats"
python "$CODEFLOW_DIR/run/multi_turn/stat.py" \
  --input "$HARNESS_JSON" \
  --output "$STAT_JSON"

if [[ "$DATASET" == "comp" ]]; then
  echo "[4/4] Convert to simulator DAG JSONL"
  python "$ROOT_DIR/convert_codeflow_comp_to_dag_pool.py" \
    --input_harness "$HARNESS_JSON" \
    --output_jsonl "$DAG_JSONL"
else
  echo "[4/4] Skip DAG conversion for repo dataset"
fi

echo "Done."
echo "Run tag:       $RUN_TAG"
if [[ -n "$INDEX_FILE" ]]; then
  echo "Index file:    $INDEX_FILE"
fi
echo "Harness output: $HARNESS_JSON"
echo "Stat output:    $STAT_JSON"
echo "DAG pool:       $DAG_JSONL"
