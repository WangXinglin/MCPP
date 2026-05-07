#!/bin/bash
#
# LOADER English note
#
# English note:
#   1) English note DAG English note + English note(English note):
#        bash run_experiment.sh prepare
#
#   2) QS English note(English note,English note shard):
#        bash run_experiment.sh auto
#
#   3) English note shard:
#        bash run_experiment.sh shard <shard_id>
#
#   4) English note shard English note,English note + English note:
#        bash run_experiment.sh merge
#        bash run_experiment.sh analyze
#
#   5) English note(English note):
#        bash run_experiment.sh all
#

set -e

# ─── Python English note ──────────────────────────────────────
# English note(QS English note)
ensure_deps() {
    local missing=()
    python3 -c "import numpy"      2>/dev/null || missing+=(numpy)
    python3 -c "import scipy"      2>/dev/null || missing+=(scipy)
    python3 -c "import pandas"     2>/dev/null || missing+=(pandas)
    python3 -c "import tqdm"       2>/dev/null || missing+=(tqdm)
    python3 -c "import matplotlib" 2>/dev/null || missing+=(matplotlib)

    if [ ${#missing[@]} -gt 0 ]; then
        echo "[deps] Installing missing packages: ${missing[*]}"
        pip install --quiet "${missing[@]}"
    else
        echo "[deps] All Python dependencies satisfied."
    fi
}

_is_positive_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) [ "$1" -gt 0 ] ;;
    esac
}

_min_positive_int() {
    local a="$1"
    local b="$2"
    if ! _is_positive_int "${a}"; then
        echo "${b}"
        return
    fi
    if ! _is_positive_int "${b}"; then
        echo "${a}"
        return
    fi
    if [ "${a}" -lt "${b}" ]; then
        echo "${a}"
    else
        echo "${b}"
    fi
}

_ceil_div() {
    local a="$1"
    local b="$2"
    if ! _is_positive_int "${a}" || ! _is_positive_int "${b}"; then
        echo 1
        return
    fi
    echo $(( (a + b - 1) / b ))
}

_largest_power_of_two_leq() {
    local n="$1"
    local value=1
    if ! _is_positive_int "${n}" || [ "${n}" -lt 1 ]; then
        echo 1
        return
    fi
    while [ $(( value * 2 )) -le "${n}" ]; do
        value=$(( value * 2 ))
    done
    echo "${value}"
}

_count_cpu_list() {
    local text="$1"
    python3 -c 'import sys
text = sys.argv[1].strip()
total = 0
for part in text.split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        lo, hi = part.split("-", 1)
        total += max(0, int(hi) - int(lo) + 1)
    else:
        int(part)
        total += 1
print(total if total > 0 else "")' "${text}" 2>/dev/null || true
}

_detect_cgroup_quota_cpus() {
    local quota period
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max || true
        if _is_positive_int "${quota:-}" && _is_positive_int "${period:-}"; then
            _ceil_div "${quota}" "${period}"
            return
        fi
    fi
    if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
        quota="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || true)"
        period="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || true)"
        if _is_positive_int "${quota:-}" && [ "${quota}" -gt 0 ] && _is_positive_int "${period:-}"; then
            _ceil_div "${quota}" "${period}"
            return
        fi
    fi
    echo ""
}

_detect_cpuset_cpus() {
    local path text count
    for path in \
        /sys/fs/cgroup/cpuset.cpus.effective \
        /sys/fs/cgroup/cpuset.cpus \
        /sys/fs/cgroup/cpuset/cpuset.cpus.effective \
        /sys/fs/cgroup/cpuset/cpuset.cpus
    do
        if [ -r "${path}" ]; then
            text="$(cat "${path}" 2>/dev/null || true)"
            count="$(_count_cpu_list "${text}")"
            if _is_positive_int "${count:-}"; then
                echo "${count}"
                return
            fi
        fi
    done
    echo ""
}

_detect_affinity_cpus() {
    local cpus
    if command -v nproc >/dev/null 2>&1; then
        cpus="$(nproc 2>/dev/null || true)"
        if _is_positive_int "${cpus:-}"; then
            echo "${cpus}"
            return
        fi
    fi
    echo ""
}

_detect_online_cpus() {
    local cpus
    cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    if _is_positive_int "${cpus:-}"; then
        echo "${cpus}"
        return
    fi
    cpus="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
    if _is_positive_int "${cpus:-}"; then
        echo "${cpus}"
        return
    fi
    echo ""
}

detect_available_logical_cpus() {
    if _is_positive_int "${LOADER_AUTO_CPU_CORES:-}"; then
        echo "${LOADER_AUTO_CPU_CORES}"
        return
    fi

    local quota_cpus cpuset_cpus affinity_cpus online_cpus affinity_cap policy
    quota_cpus="$(_detect_cgroup_quota_cpus)"
    cpuset_cpus="$(_detect_cpuset_cpus)"
    affinity_cpus="$(_detect_affinity_cpus)"
    online_cpus="$(_detect_online_cpus)"
    affinity_cap="${cpuset_cpus:-${affinity_cpus:-${online_cpus:-}}}"
    policy="${LOADER_AUTO_CPU_POLICY:-effective}"

    case "${policy}" in
        quota)
            echo "${quota_cpus:-${affinity_cap:-1}}"
            ;;
        affinity|logical)
            echo "${affinity_cap:-${quota_cpus:-1}}"
            ;;
        max)
            local best=1
            for value in "${quota_cpus}" "${affinity_cap}" "${online_cpus}"; do
                if _is_positive_int "${value:-}" && [ "${value}" -gt "${best}" ]; then
                    best="${value}"
                fi
            done
            echo "${best}"
            ;;
        effective|*)
            if _is_positive_int "${quota_cpus:-}" && _is_positive_int "${affinity_cap:-}"; then
                _min_positive_int "${quota_cpus}" "${affinity_cap}"
            else
                echo "${quota_cpus:-${affinity_cap:-1}}"
            fi
            ;;
    esac
}

auto_tune_cpu_params() {
    local detected_cpus reserve usable_cpus mode shard_auto pair_auto chunks_auto pair_workers sample_chunks
    local quota_cpus cpuset_cpus affinity_cpus online_cpus trial_limit target_shard_workers
    local planner_pair_cap planner_outer_workers
    detected_cpus="$(detect_available_logical_cpus)"
    quota_cpus="$(_detect_cgroup_quota_cpus)"
    cpuset_cpus="$(_detect_cpuset_cpus)"
    affinity_cpus="$(_detect_affinity_cpus)"
    online_cpus="$(_detect_online_cpus)"
    reserve="${LOADER_AUTO_CPU_RESERVE:-0}"
    if ! _is_positive_int "${reserve}" && [ "${reserve}" != "0" ]; then
        reserve=0
    fi
    usable_cpus=$(( detected_cpus - reserve ))
    if [ "${usable_cpus}" -lt 1 ]; then
        usable_cpus=1
    fi

    shard_auto=0
    pair_auto=0
    chunks_auto=0
    if [ -z "${SHARD_N_WORKERS:-}" ] || [ "${SHARD_N_WORKERS}" = "auto" ]; then
        shard_auto=1
    fi
    if [ -z "${MC_PORTFOLIO_PAIR_WORKERS:-}" ] || [ "${MC_PORTFOLIO_PAIR_WORKERS}" = "auto" ]; then
        pair_auto=1
    fi
    if [ -z "${MC_PORTFOLIO_SAMPLE_CHUNKS:-}" ] || [ "${MC_PORTFOLIO_SAMPLE_CHUNKS}" = "auto" ]; then
        chunks_auto=1
    fi

    trial_limit="${N_EXEC_TRIALS:-0}"
    if ! _is_positive_int "${trial_limit}"; then
        trial_limit=0
    fi

    # auto: choose from the current workload shape.
    # - Baseline-only work has no MC backup, so use outer trial parallelism.
    # - One-DAG MC shards with fewer trials than CPUs need balanced/inner MC
    #   parallelism; otherwise many CPUs have no trial chunk to execute.
    # - inner_full is an explicit pure planner benchmark: keep one outer trial
    #   worker and give the MC planner all usable CPUs.
    # - planner_latency caps the MC planner at a useful inner parallelism and
    #   uses the remaining CPUs for outer trial workers, avoiding inner
    #   over-allocation once policy.allocate() latency has saturated.
    # - Manual LOADER_AUTO_PARALLEL_MODE still overrides this heuristic.
    mode="${LOADER_AUTO_PARALLEL_MODE:-auto}"
    if [ "${shard_auto}" = "1" ] && [ "${pair_auto}" = "1" ]; then
        if [ "${BASELINE_ONLY}" = "1" ]; then
            SHARD_N_WORKERS="${usable_cpus}"
            MC_PORTFOLIO_PAIR_WORKERS=1
        else
            case "${mode}" in
            planner_latency|planner-latency|planner)
                planner_pair_cap="${LOADER_PLANNER_LATENCY_PAIR_WORKERS_CAP:-${LOADER_AUTO_PLANNER_PAIR_WORKERS_CAP:-${LOADER_AUTO_MAX_SAMPLE_CHUNKS:-64}}}"
                if ! _is_positive_int "${planner_pair_cap}"; then
                    planner_pair_cap=64
                fi
                if _is_positive_int "${MC_PORTFOLIO_M:-}" && [ "${planner_pair_cap}" -gt "${MC_PORTFOLIO_M}" ]; then
                    planner_pair_cap="${MC_PORTFOLIO_M}"
                fi
                if [ "${planner_pair_cap}" -gt "${usable_cpus}" ]; then
                    planner_pair_cap="${usable_cpus}"
                fi
                planner_outer_workers="$(_ceil_div "${usable_cpus}" "${planner_pair_cap}")"
                if [ "${trial_limit}" -gt 0 ] && [ "${planner_outer_workers}" -gt "${trial_limit}" ]; then
                    planner_outer_workers="${trial_limit}"
                fi
                if [ "${planner_outer_workers}" -lt 1 ]; then
                    planner_outer_workers=1
                fi
                SHARD_N_WORKERS="${planner_outer_workers}"
                MC_PORTFOLIO_PAIR_WORKERS=$(( usable_cpus / SHARD_N_WORKERS ))
                if [ "${MC_PORTFOLIO_PAIR_WORKERS}" -lt 1 ]; then
                    MC_PORTFOLIO_PAIR_WORKERS=1
                fi
                if [ "${MC_PORTFOLIO_PAIR_WORKERS}" -gt "${planner_pair_cap}" ]; then
                    MC_PORTFOLIO_PAIR_WORKERS="${planner_pair_cap}"
                fi
                ;;
            inner|mc|mc_inner|inner_full)
                SHARD_N_WORKERS=1
                MC_PORTFOLIO_PAIR_WORKERS="${usable_cpus}"
                ;;
            outer_half|process_half|half)
                SHARD_N_WORKERS=$(( (usable_cpus + 1) / 2 ))
                if [ "${SHARD_N_WORKERS}" -lt 1 ]; then
                    SHARD_N_WORKERS=1
                fi
                MC_PORTFOLIO_PAIR_WORKERS=1
                ;;
            outer|process|process_outer)
                SHARD_N_WORKERS="${usable_cpus}"
                MC_PORTFOLIO_PAIR_WORKERS=1
                ;;
            legacy_balanced)
                if [ "${usable_cpus}" -ge 24 ]; then
                    SHARD_N_WORKERS=4
                elif [ "${usable_cpus}" -ge 8 ]; then
                    SHARD_N_WORKERS=2
                else
                    SHARD_N_WORKERS=1
                fi
                MC_PORTFOLIO_PAIR_WORKERS=$(( (usable_cpus + SHARD_N_WORKERS - 1) / SHARD_N_WORKERS ))
                ;;
            balanced|auto|*)
                if [ "${ONLY_LOADER}" = "1" ] && [ "${QS_ONE_DAG_PER_JOB:-0}" = "1" ] && [ "${trial_limit}" -gt 0 ] && [ "${trial_limit}" -lt "${usable_cpus}" ]; then
                    if [ "${usable_cpus}" -ge 96 ]; then
                        target_shard_workers=4
                    elif [ "${usable_cpus}" -ge 24 ]; then
                        target_shard_workers=3
                    elif [ "${usable_cpus}" -ge 8 ]; then
                        target_shard_workers=2
                    else
                        target_shard_workers=1
                    fi
                    SHARD_N_WORKERS="$(_min_positive_int "${target_shard_workers}" "${trial_limit}")"
                    MC_PORTFOLIO_PAIR_WORKERS="$(_ceil_div "${usable_cpus}" "${SHARD_N_WORKERS}")"
                elif [ "${trial_limit}" -gt 0 ] && [ "${trial_limit}" -lt "${usable_cpus}" ] && [ "${BASELINE_ONLY}" != "1" ]; then
                    if [ "${usable_cpus}" -ge 24 ]; then
                        SHARD_N_WORKERS=4
                    elif [ "${usable_cpus}" -ge 8 ]; then
                        SHARD_N_WORKERS=2
                    else
                        SHARD_N_WORKERS=1
                    fi
                    SHARD_N_WORKERS="$(_min_positive_int "${SHARD_N_WORKERS}" "${trial_limit}")"
                    MC_PORTFOLIO_PAIR_WORKERS="$(_ceil_div "${usable_cpus}" "${SHARD_N_WORKERS}")"
                else
                    SHARD_N_WORKERS="${usable_cpus}"
                    MC_PORTFOLIO_PAIR_WORKERS=1
                fi
                ;;
            esac
        fi
    elif [ "${shard_auto}" = "1" ]; then
        pair_workers="${MC_PORTFOLIO_PAIR_WORKERS}"
        if ! _is_positive_int "${pair_workers}"; then
            pair_workers=1
        fi
        SHARD_N_WORKERS=$(( usable_cpus / pair_workers ))
        if [ "${SHARD_N_WORKERS}" -lt 1 ]; then
            SHARD_N_WORKERS=1
        fi
    elif [ "${pair_auto}" = "1" ]; then
        if ! _is_positive_int "${SHARD_N_WORKERS}"; then
            SHARD_N_WORKERS=1
        fi
        MC_PORTFOLIO_PAIR_WORKERS=$(( (usable_cpus + SHARD_N_WORKERS - 1) / SHARD_N_WORKERS ))
        if [ "${MC_PORTFOLIO_PAIR_WORKERS}" -lt 1 ]; then
            MC_PORTFOLIO_PAIR_WORKERS=1
        fi
    fi

    if [ "${trial_limit}" -gt 0 ] && [ "${SHARD_N_WORKERS}" -gt "${trial_limit}" ]; then
        SHARD_N_WORKERS="${trial_limit}"
    fi

    if [ "${chunks_auto}" = "1" ]; then
        pair_workers="${MC_PORTFOLIO_PAIR_WORKERS}"
        if ! _is_positive_int "${pair_workers}"; then
            pair_workers="${usable_cpus}"
        fi
        sample_chunks="${pair_workers}"
        if [ "${mode}" = "planner_latency" ] || [ "${mode}" = "planner-latency" ] || [ "${mode}" = "planner" ]; then
            if _is_positive_int "${planner_pair_cap:-}"; then
                sample_chunks="${planner_pair_cap}"
            fi
        fi
        if _is_positive_int "${MC_PORTFOLIO_M:-}" && [ "${sample_chunks}" -gt "${MC_PORTFOLIO_M}" ]; then
            sample_chunks="${MC_PORTFOLIO_M}"
        fi
        local max_sample_chunks="${LOADER_AUTO_MAX_SAMPLE_CHUNKS:-64}"
        if ! _is_positive_int "${max_sample_chunks}"; then
            max_sample_chunks=64
        fi
        if [ "${sample_chunks}" -gt "${max_sample_chunks}" ]; then
            sample_chunks="${max_sample_chunks}"
        fi
        sample_chunks="$(_largest_power_of_two_leq "${sample_chunks}")"
        MC_PORTFOLIO_SAMPLE_CHUNKS="${sample_chunks}"
    fi

    if [ -z "${MC_PORTFOLIO_PARALLEL_BACKEND:-}" ] || [ "${MC_PORTFOLIO_PARALLEL_BACKEND}" = "auto" ]; then
        if _is_positive_int "${MC_PORTFOLIO_PAIR_WORKERS}" && [ "${MC_PORTFOLIO_PAIR_WORKERS}" -gt 1 ]; then
            MC_PORTFOLIO_PARALLEL_BACKEND=process
        else
            MC_PORTFOLIO_PARALLEL_BACKEND=thread
        fi
    fi

    echo "[auto-cpu] quota=${quota_cpus:-none}, cpuset=${cpuset_cpus:-none}, affinity=${affinity_cpus:-none}, online=${online_cpus:-none}, policy=${LOADER_AUTO_CPU_POLICY:-effective}"
    echo "[auto-cpu] detected_logical_cpus=${detected_cpus}, reserve=${reserve}, usable=${usable_cpus}, mode=${mode}, trial_limit=${trial_limit}"
    echo "[auto-cpu] SHARD_N_WORKERS=${SHARD_N_WORKERS}, MC_PORTFOLIO_PAIR_WORKERS=${MC_PORTFOLIO_PAIR_WORKERS}, MC_PORTFOLIO_SAMPLE_CHUNKS=${MC_PORTFOLIO_SAMPLE_CHUNKS}, MC_PORTFOLIO_PARALLEL_BACKEND=${MC_PORTFOLIO_PARALLEL_BACKEND}, nominal_parallel=$(( SHARD_N_WORKERS * MC_PORTFOLIO_PAIR_WORKERS ))"
}

_runtime_cmd_with_auto_workers() {
    local cmd="$1"
    if [ -z "${SHARD_N_WORKERS:-}" ] || [ "${SHARD_N_WORKERS}" = "auto" ]; then
        echo "${cmd}"
        return
    fi
    echo "${cmd}" | sed -E "s/(--n_workers )[0-9]+/\\1${SHARD_N_WORKERS}/g"
}

_build_claim_scan_order() {
    local n_jobs="$1"
    local seed="$2"
    "${PYTHON}" - "${n_jobs}" "${seed}" <<'PY'
import hashlib
import random
import sys

n_jobs = int(sys.argv[1])
seed = sys.argv[2]
seed_int = int.from_bytes(hashlib.blake2b(seed.encode("utf-8"), digest_size=16).digest(), "big")
order = list(range(max(0, n_jobs)))
random.Random(seed_int).shuffle(order)
for shard_id in order:
    print(shard_id)
PY
}

# ─── English note(English note)──────────────────────────────
N_DAGS="${N_DAGS:-1000}"                # DAG English note(English note subset_1k)
DAGS_PER_SHARD="${DAGS_PER_SHARD:-500}" # English note shard English note DAG English note(English note)
N_EXEC_TRIALS="${N_EXEC_TRIALS:-1000}"  # English note DAG English note
QS_ONE_DAG_PER_JOB="${QS_ONE_DAG_PER_JOB:-0}"  # 1 = QS English note job English note 1 English note DAG
BUDGET_RATIOS="${BUDGET_RATIOS:-${BUDGET_RATIO:-0.3 0.5 1.0 2.0 5.0 10.0}}"
BUDGET_VALUES="${BUDGET_VALUES:-${BUDGET_VALUE:-}}"
BUDGET_DEADLINE_VALUES_MAP="${BUDGET_DEADLINE_VALUES_MAP:-}"
BUDGET_COST_MODE="${BUDGET_COST_MODE:-token_cost_mean}"  # token_cost_mean/output_money_cost_mean/cost_mean/lat_mean
REQUESTED_SHARD_N_WORKERS="${SHARD_N_WORKERS:-auto}"
REQUESTED_MC_PORTFOLIO_PAIR_WORKERS="${MC_PORTFOLIO_PAIR_WORKERS:-auto}"
REQUESTED_MC_PORTFOLIO_SAMPLE_CHUNKS="${MC_PORTFOLIO_SAMPLE_CHUNKS:-auto}"
REQUESTED_MC_PORTFOLIO_PARALLEL_BACKEND="${MC_PORTFOLIO_PARALLEL_BACKEND:-auto}"
SHARD_N_WORKERS="${REQUESTED_SHARD_N_WORKERS}" # run_shard English note trial English note;auto English note workload English note
MC_PORTFOLIO_PAIR_WORKERS="${REQUESTED_MC_PORTFOLIO_PAIR_WORKERS}"
MC_PORTFOLIO_SAMPLE_CHUNKS="${REQUESTED_MC_PORTFOLIO_SAMPLE_CHUNKS}"
MC_PORTFOLIO_PARALLEL_BACKEND="${REQUESTED_MC_PORTFOLIO_PARALLEL_BACKEND}"
MC_PORTFOLIO_M="${MC_PORTFOLIO_M:-64}"
MC_PORTFOLIO_BASE_KS="${MC_PORTFOLIO_BASE_KS:-}"
MC_PORTFOLIO_MAX_CANDIDATE_ACTIONS="${MC_PORTFOLIO_MAX_CANDIDATE_ACTIONS:-2048}"
DEADLINE_TRACE="${DEADLINE_TRACE:-0}"
ONLY_LOADER="${ONLY_LOADER:-0}"                # 1 = only run mc_portfolio_rollout
BASELINE_ONLY="${BASELINE_ONLY:-0}"            # 1 = only run baselines
BASELINE_METHOD_SET="${BASELINE_METHOD_SET:-all}" # all/seq/static baseline subsets
auto_tune_cpu_params
export MC_PORTFOLIO_PAIR_WORKERS MC_PORTFOLIO_SAMPLE_CHUNKS MC_PORTFOLIO_PARALLEL_BACKEND MC_PORTFOLIO_M MC_PORTFOLIO_BASE_KS MC_PORTFOLIO_MAX_CANDIDATE_ACTIONS DEADLINE_TRACE
SEED=42
LATENCY_DIST="${LATENCY_DIST:-lognormal}"
DEADLINE_RATIOS="${DEADLINE_RATIOS:-${DEADLINE_RATIO:-1.0}}"
DEADLINE_VALUES="${DEADLINE_VALUES:-${DEADLINE_VALUE:-}}"
# English note:English note,English note
# English note 1k DAG English note shard English note:
#   br=0.3/0.5: 2 English note;br=1.0: 5 English note;br=2.0: 17 English note;br=5.0: 50 English note;br=10.0: 100 English note
SHARD_SIZE_MAP="${SHARD_SIZE_MAP:-1.0:200,2.0:60,5.0:20,10.0:10}"
# English note 1000 English note DAG English note
USE_DAG_INDICES_FILE="${USE_DAG_INDICES_FILE:-results/subset_1k/selected_dag_indices.json}"

if [ "${QS_ONE_DAG_PER_JOB}" = "1" ]; then
    DAGS_PER_SHARD=1
    SHARD_SIZE_MAP=""
fi

# ─── English note ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAG_POOL_PATH="${DAG_POOL_PATH:-${SCRIPT_DIR}/pregenerated_dags/dag_pool.jsonl}"
# English note,English note.
# English note auto worker English note,English note worker English note RUN_NAME.
RUN_NAME="${RUN_NAME:-orig_pipeline_20260415}"
RUN_ROOT="${SCRIPT_DIR}/results/runs/${RUN_NAME}"
SHARD_DIR="${RUN_ROOT}/shards"
MERGED_DIR="${RUN_ROOT}/merged"
JOBS_LIST="${RUN_ROOT}/jobs_list.txt"
JOBS_SCRIPT="${RUN_ROOT}/jobs.sh"
CLAIM_DIR="${RUN_ROOT}/claims"
PYTHON="python3"
GPU_SCRIPT="${GPU_SCRIPT:-}"
CLAIM_HEARTBEAT_INTERVAL="${CLAIM_HEARTBEAT_INTERVAL:-60}"
CLAIM_STALE_THRESHOLD="${CLAIM_STALE_THRESHOLD:-1800}"
CLAIM_NO_OUTPUT_STALE_THRESHOLD="${CLAIM_NO_OUTPUT_STALE_THRESHOLD:-900}"
CLAIM_NO_PROGRESS_STALE_THRESHOLD="${CLAIM_NO_PROGRESS_STALE_THRESHOLD:-7200}"
AUTO_SWEEP_STALE="${AUTO_SWEEP_STALE:-1}"
AUTO_MAX_CLAIMS="${AUTO_MAX_CLAIMS:-}"
MERGE_COPY_MODE="${MERGE_COPY_MODE:-fast}"  # line/fast
MERGE_COPY_BUFFER_MB="${MERGE_COPY_BUFFER_MB:-16}"
MERGE_JOBS="${MERGE_JOBS:-1}"
STATUS_JOBS="${STATUS_JOBS:-0}"  # 0 = auto, otherwise number of parallel status workers
ANALYZE_JOBS="${ANALYZE_JOBS:-1}"
ANALYZE_SKIP_DEADLINE_AUC="${ANALYZE_SKIP_DEADLINE_AUC:-1}"
ANALYZE_SKIP_SOLVE_SUMMARY="${ANALYZE_SKIP_SOLVE_SUMMARY:-1}"
ANALYZE_SKIP_COMPLETE="${ANALYZE_SKIP_COMPLETE:-1}"
ANALYZE_PROGRESS="${ANALYZE_PROGRESS:-1}"
REPORT_DIR="${REPORT_DIR:-${MERGED_DIR}/budget_failure_report}"
REPORT_WRITE_RECORDS="${REPORT_WRITE_RECORDS:-0}"
REPORT_JOBS="${REPORT_JOBS:-0}"  # 0 = auto, otherwise number of parallel report workers
REPORT_PROGRESS="${REPORT_PROGRESS:-1}"
SUITE_RUN_NAME_PREFIX="${SUITE_RUN_NAME_PREFIX:-${RUN_NAME}}"
SUITE_PORTFOLIO_RUN_NAME="${SUITE_PORTFOLIO_RUN_NAME:-${SUITE_RUN_NAME_PREFIX}}"
SUITE_INCLUDE_PORTFOLIO="${SUITE_INCLUDE_PORTFOLIO:-1}"
PORTFOLIO_DAG_POOL_PATH="${PORTFOLIO_DAG_POOL_PATH:-${DAG_POOL_PATH}}"
BASELINE_DAG_POOL_MAP="${BASELINE_DAG_POOL_MAP:-}"
SUITE_BASELINE_METHOD_SET="${SUITE_BASELINE_METHOD_SET:-${BASELINE_METHOD_SET}}"

# ─── English note ─────────────────────────────────────────

_get_cmd_field() {
    local cmd="$1"
    local pattern="$2"
    echo "${cmd}" | grep -oP "${pattern}" || true
}

_deadline_tag_from_cmd() {
    local cmd="$1"
    local dv dr
    dv=$(_get_cmd_field "${cmd}" '(?<=--deadline_value )\S+')
    dr=$(_get_cmd_field "${cmd}" '(?<=--deadline_ratio )\S+')
    if [ -n "${dv}" ]; then
        echo "deadline_value_${dv}"
    else
        echo "deadline_ratio_${dr:-1.0}"
    fi
}

_budget_tag_from_cmd() {
    local cmd="$1"
    local bv br
    bv=$(_get_cmd_field "${cmd}" '(?<=--budget_value )\S+')
    br=$(_get_cmd_field "${cmd}" '(?<=--budget_ratio )\S+')
    if [ -n "${bv}" ]; then
        echo "budget_value_${bv}"
    else
        echo "budget_ratio_${br:-unknown}"
    fi
}

_raw_budget_stem_from_cmd() {
    local cmd="$1"
    local bv br
    bv=$(_get_cmd_field "${cmd}" '(?<=--budget_value )\S+')
    br=$(_get_cmd_field "${cmd}" '(?<=--budget_ratio )\S+')
    if [ -n "${bv}" ]; then
        echo "raw_budget_value_${bv}"
    else
        echo "raw_budget_${br:-unknown}"
    fi
}

_budget_deadline_key_from_cmd() {
    local cmd="$1"
    local btag dv dr
    btag=$(_budget_tag_from_cmd "${cmd}")
    dv=$(_get_cmd_field "${cmd}" '(?<=--deadline_value )\S+')
    dr=$(_get_cmd_field "${cmd}" '(?<=--deadline_ratio )\S+')
    if [ -n "${dv}" ]; then
        echo "${btag}@dv=${dv}"
    else
        echo "${btag}@dr=${dr:-1.0}"
    fi
}

_budget_cost_tag_from_cmd() {
    local cmd="$1"
    local mode
    mode=$(_get_cmd_field "${cmd}" '(?<=--budget_cost_mode )\S+')
    echo "budget_cost_${mode:-token_cost_mean}"
}

_output_file_from_cmd() {
    local cmd="$1"
    local out_dir raw_stem dtag bctag ds de ss se
    out_dir=$(_get_cmd_field "${cmd}" '(?<=--output_dir )\S+')
    if [ -z "${out_dir}" ]; then
        out_dir="${SHARD_DIR}"
    fi

    raw_stem=$(_raw_budget_stem_from_cmd "${cmd}")
    dtag=$(_deadline_tag_from_cmd "${cmd}")
    bctag=$(_budget_cost_tag_from_cmd "${cmd}")
    ds=$(_get_cmd_field "${cmd}" '(?<=--dag_start )\S+')
    de=$(_get_cmd_field "${cmd}" '(?<=--dag_end )\S+')
    ss=$(_get_cmd_field "${cmd}" '(?<=--dag_shard_start )\S+')
    se=$(_get_cmd_field "${cmd}" '(?<=--dag_shard_end )\S+')

    if [ -n "${ss}" ] && [ -n "${se}" ]; then
        echo "${out_dir}/${raw_stem}_${dtag}_${bctag}_sel${ss}-$((se - 1)).jsonl"
    elif [ -n "${ds}" ] && [ -n "${de}" ]; then
        echo "${out_dir}/${raw_stem}_${dtag}_${bctag}_dag${ds}-$((de - 1)).jsonl"
    else
        echo ""
    fi
}

_shard_desc_from_cmd() {
    local cmd="$1"
    local dtag ds de ss se
    dtag=$(_deadline_tag_from_cmd "${cmd}")
    ds=$(_get_cmd_field "${cmd}" '(?<=--dag_start )\S+')
    de=$(_get_cmd_field "${cmd}" '(?<=--dag_end )\S+')
    ss=$(_get_cmd_field "${cmd}" '(?<=--dag_shard_start )\S+')
    se=$(_get_cmd_field "${cmd}" '(?<=--dag_shard_end )\S+')

    if [ -n "${ss}" ] && [ -n "${se}" ]; then
        echo "${dtag} sel${ss}-$((se - 1))"
    elif [ -n "${ds}" ] && [ -n "${de}" ]; then
        echo "${dtag} dag${ds}-$((de - 1))"
    else
        echo "unknown"
    fi
}

_claim_age_info() {
    local claim_path="$1"
    local cmd="$2"
    local now="$3"
    local output_file worker_file heartbeat_file
    local output_age heartbeat_age worker_age freshest

    output_file=$(_output_file_from_cmd "${cmd}")
    worker_file="${claim_path}/worker_info"
    heartbeat_file="${claim_path}/heartbeat"

    output_age=""
    if [ -n "${output_file}" ] && [ -f "${output_file}" ]; then
        output_age=$((now - $(stat -c %Y "${output_file}" 2>/dev/null || echo 0)))
        if [ "${output_age}" -ge "${CLAIM_NO_PROGRESS_STALE_THRESHOLD}" ]; then
            echo "${output_age}|output_stalled"
            return
        fi
    fi

    heartbeat_age=""
    if [ -f "${heartbeat_file}" ]; then
        heartbeat_age=$((now - $(stat -c %Y "${heartbeat_file}" 2>/dev/null || echo 0)))
    fi

    worker_age=""
    if [ -f "${worker_file}" ]; then
        worker_age=$((now - $(stat -c %Y "${worker_file}" 2>/dev/null || echo 0)))
    fi

    if [ -n "${output_age}" ]; then
        freshest="${output_age}"
        if [ -n "${heartbeat_age}" ] && [ "${heartbeat_age}" -lt "${freshest}" ]; then
            freshest="${heartbeat_age}"
            echo "${freshest}|heartbeat"
        else
            echo "${freshest}|output"
        fi
        return
    fi

    if [ -n "${worker_age}" ] && [ "${worker_age}" -ge "${CLAIM_NO_OUTPUT_STALE_THRESHOLD}" ]; then
        echo "${worker_age}|no_output"
        return
    fi

    if [ -n "${worker_age}" ] && [ "${worker_age}" -lt "${CLAIM_NO_OUTPUT_STALE_THRESHOLD}" ]; then
        if [ -n "${heartbeat_age}" ] && [ "${heartbeat_age}" -lt "${worker_age}" ]; then
            echo "${heartbeat_age}|heartbeat_no_output"
        else
            echo "${worker_age}|worker_info_no_output"
        fi
        return
    fi

    if [ -n "${heartbeat_age}" ] && [ "${heartbeat_age}" -lt "${CLAIM_NO_OUTPUT_STALE_THRESHOLD}" ]; then
        echo "${heartbeat_age}|heartbeat_no_output"
        return
    fi

    if [ -n "${heartbeat_age}" ]; then
        echo "${heartbeat_age}|no_output"
    elif [ -n "${worker_age}" ]; then
        echo "${worker_age}|no_output"
    else
        echo "$((CLAIM_STALE_THRESHOLD + 1))|missing"
    fi
}

_start_claim_heartbeat() {
    local claim_path="$1"
    (
        local hb_tmp
        hb_tmp="${claim_path}/heartbeat.tmp"
        while true; do
            date -Iseconds > "${hb_tmp}" 2>/dev/null || {
                sleep "${CLAIM_HEARTBEAT_INTERVAL}"
                continue
            }
            mv -f "${hb_tmp}" "${claim_path}/heartbeat" 2>/dev/null || true
            sleep "${CLAIM_HEARTBEAT_INTERVAL}"
        done
    ) >/dev/null 2>&1 &
    echo $!
}

_stop_claim_heartbeat() {
    local hb_pid="$1"
    if [ -n "${hb_pid}" ]; then
        kill "${hb_pid}" 2>/dev/null || true
        wait "${hb_pid}" 2>/dev/null || true
    fi
}

_sweep_stale_claims() {
    local n_jobs="$1"
    local quiet="${2:-0}"
    local now released i claim_path line_num cmd age_info age age_src
    released=0
    now=$(date +%s)

    for i in $(seq 0 $((n_jobs - 1))); do
        claim_path="${CLAIM_DIR}/shard_${i}"
        if [ ! -d "${claim_path}" ] || [ -f "${claim_path}/done" ]; then
            continue
        fi

        line_num=$((i + 1))
        cmd=$(sed -n "${line_num}p" "${JOBS_LIST}")
        age_info=$(_claim_age_info "${claim_path}" "${cmd}" "${now}")
        age="${age_info%%|*}"
        age_src="${age_info#*|}"

        if [ "${age}" -ge "${CLAIM_STALE_THRESHOLD}" ]; then
            rm -rf "${claim_path}"
            released=$((released + 1))
            if [ "${quiet}" -eq 0 ]; then
                echo "  Released shard_${i} (stale, ${age_src} ${age}s ago)"
            fi
        fi
    done

    echo "${released}"
}

do_prepare() {
    ensure_deps
    if [ "${ONLY_LOADER}" = "1" ] && [ "${BASELINE_ONLY}" = "1" ]; then
        echo "Error: ONLY_LOADER=1 and BASELINE_ONLY=1 cannot both be set."
        exit 1
    fi
    if [ -n "${BUDGET_DEADLINE_VALUES_MAP}" ]; then
        DEADLINE_ARGS=""
    elif [ -n "${DEADLINE_VALUES}" ]; then
        DEADLINE_ARGS="--deadline_values ${DEADLINE_VALUES}"
    else
        DEADLINE_ARGS="--deadline_ratios ${DEADLINE_RATIOS}"
    fi
    if [ -n "${BUDGET_DEADLINE_VALUES_MAP}" ]; then
        BUDGET_ARGS=(--budget_deadline_values_map "${BUDGET_DEADLINE_VALUES_MAP}")
    elif [ -n "${BUDGET_VALUES}" ]; then
        BUDGET_ARGS="--budget_values ${BUDGET_VALUES}"
    else
        BUDGET_ARGS="--budget_ratios ${BUDGET_RATIOS}"
    fi
    METHOD_ARGS=()
    if [ "${ONLY_LOADER}" = "1" ]; then
        METHOD_ARGS+=("--only_loader")
    elif [ "${BASELINE_ONLY}" = "1" ]; then
        METHOD_ARGS+=("--baseline_only")
    fi
    if [ "${BASELINE_METHOD_SET}" != "all" ]; then
        METHOD_ARGS+=("--baseline_method_set" "${BASELINE_METHOD_SET}")
    fi
    echo ""
    echo "============================================================"
    echo "Step 1: Using pre-selected DAG subset (${N_DAGS} DAGs) ..."
    echo "============================================================"
    echo "DAG indices file: ${USE_DAG_INDICES_FILE}"
    if [ -n "${BUDGET_DEADLINE_VALUES_MAP}" ]; then
        echo "Budget/deadline value map: ${BUDGET_DEADLINE_VALUES_MAP}"
    elif [ -n "${BUDGET_VALUES}" ]; then
        echo "Budget values: ${BUDGET_VALUES}"
    else
        echo "Budget ratios: ${BUDGET_RATIOS}"
    fi
    echo "Budget cost mode: ${BUDGET_COST_MODE}"
    echo "Shard workers: ${SHARD_N_WORKERS}"
    echo "MC settings: pair_workers=${MC_PORTFOLIO_PAIR_WORKERS}, sample_chunks=${MC_PORTFOLIO_SAMPLE_CHUNKS}, backend=${MC_PORTFOLIO_PARALLEL_BACKEND}, M=${MC_PORTFOLIO_M}, base_ks=${MC_PORTFOLIO_BASE_KS:-default}, max_candidates=${MC_PORTFOLIO_MAX_CANDIDATE_ACTIONS}, trace=${DEADLINE_TRACE}"
    if [ "${ONLY_LOADER}" = "1" ]; then
        echo "Methods: mc_portfolio_rollout only"
    elif [ "${BASELINE_ONLY}" = "1" ]; then
        echo "Methods: baselines only (${BASELINE_METHOD_SET})"
    else
        echo "Methods: baselines (${BASELINE_METHOD_SET}) + mc_portfolio_rollout"
    fi
    if [ "${QS_ONE_DAG_PER_JOB}" = "1" ]; then
        echo "QS job granularity: 1 DAG per submitted job"
    else
        echo "QS job granularity: DAGS_PER_SHARD=${DAGS_PER_SHARD}, SHARD_SIZE_MAP=${SHARD_SIZE_MAP:-<none>}"
    fi
    echo ""

    echo "============================================================"
    echo "Step 2: Generating job list ..."
    echo "============================================================"
    cd "${SCRIPT_DIR}"
    ${PYTHON} gen_jobs.py \
        --n_dags ${N_DAGS} \
        --dags_per_shard ${DAGS_PER_SHARD} \
        ${BUDGET_ARGS[@]} \
        --budget_cost_mode "${BUDGET_COST_MODE}" \
        --n_exec_trials ${N_EXEC_TRIALS} \
        --dag_pool_path "${DAG_POOL_PATH}" \
        --output_dir "${SHARD_DIR}" \
        --seed ${SEED} \
        --latency_dist ${LATENCY_DIST} \
        --n_workers "${SHARD_N_WORKERS}" \
        ${DEADLINE_ARGS} \
        --gpu_script "${GPU_SCRIPT}" \
        --shard_size_map "${SHARD_SIZE_MAP}" \
        --dag_indices_file "${USE_DAG_INDICES_FILE}" \
        "${METHOD_ARGS[@]}" \
        --jobs_list_path "${JOBS_LIST}" \
        --jobs_script_path "${JOBS_SCRIPT}"
    echo ""

    # English note claim English note(English note)
    rm -rf "${CLAIM_DIR}"
    mkdir -p "${CLAIM_DIR}"

    N_JOBS=$(wc -l < "${JOBS_LIST}")
    echo "============================================================"
    echo "Preparation complete!"
    echo "  Run name:   ${RUN_NAME}"
    echo "  Run root:   ${RUN_ROOT}"
    echo "  DAG pool:   ${DAG_POOL_PATH}"
    if [ -n "${BUDGET_DEADLINE_VALUES_MAP}" ]; then
        echo "  Deadline:   budget-specific"
    elif [ -n "${DEADLINE_VALUES}" ]; then
        echo "  Deadline:   values=${DEADLINE_VALUES}"
    else
        echo "  Deadline:   ratios=${DEADLINE_RATIOS}"
    fi
    if [ -n "${BUDGET_DEADLINE_VALUES_MAP}" ]; then
        echo "  Budget:     deadline-map=${BUDGET_DEADLINE_VALUES_MAP}"
    elif [ -n "${BUDGET_VALUES}" ]; then
        echo "  Budget:     values=${BUDGET_VALUES}"
    else
        echo "  Budget:     ratios=${BUDGET_RATIOS}"
    fi
    echo "  Job list:   ${JOBS_LIST} (${N_JOBS} jobs)"
    if [ "${QS_ONE_DAG_PER_JOB}" = "1" ]; then
        echo "  QS atomic:  enabled (each jobs_list line has one selected DAG)"
    fi
    echo ""
    echo "QS batch submit (all jobs use the SAME command):"
    echo "  cd ${SCRIPT_DIR} && bash run_experiment.sh auto"
    echo ""
    echo "Submit ${N_JOBS} copies of this command to QS platform."
    echo "============================================================"
}

do_shard() {
    ensure_deps
    SHARD_ID=$1
    if [ -z "${SHARD_ID}" ]; then
        echo "Usage: bash run_experiment.sh shard <shard_id>"
        echo "  shard_id: 0-based index into jobs_list.txt"
        exit 1
    fi

    if [ ! -f "${JOBS_LIST}" ]; then
        echo "Error: ${JOBS_LIST} not found. Run 'prepare' first."
        exit 1
    fi

    N_JOBS=$(wc -l < "${JOBS_LIST}")
    if [ "${SHARD_ID}" -ge "${N_JOBS}" ]; then
        echo "Error: shard_id=${SHARD_ID} out of range (total ${N_JOBS} jobs)"
        exit 1
    fi

    # English note SHARD_ID English note(0-based)
    LINE_NUM=$((SHARD_ID + 1))
    CMD=$(sed -n "${LINE_NUM}p" "${JOBS_LIST}")

    echo "============================================================"
    echo "Running shard ${SHARD_ID}/${N_JOBS}:"
    echo "  ${CMD}"
    echo "============================================================"
    cd "${SCRIPT_DIR}"
    eval ${CMD}
}

# ─── English note(QS English note)─────────────────────
# English note:English note mkdir English note,NFS English note
#   - English note shard English note claims/shard_<id> English note
#   - mkdir English note = English note,English note = English note
#   - English note worker English note,English note
#   - English note shard English note,English note,English note
do_auto() {
    ensure_deps

    if [ ! -f "${JOBS_LIST}" ]; then
        echo "Error: ${JOBS_LIST} not found. Run 'prepare' first."
        exit 1
    fi

    mkdir -p "${CLAIM_DIR}"
    N_JOBS=$(wc -l < "${JOBS_LIST}")
    COMPLETED=0
    CLAIMS_ATTEMPTED=0
    LAST_EXIT=0
    CURRENT_HEARTBEAT_PID=""
    CLAIM_SCAN_ORDER=()
    CLAIM_SCAN_CURSOR=0
    CLAIM_SCAN_SEED="${RUN_NAME}|$(hostname)|$$|$(date +%s%N 2>/dev/null || date +%s)|${RANDOM}"
    CLAIM_SCAN_SEED_HASH=$(printf "%s" "${CLAIM_SCAN_SEED}" | cksum | awk '{print $1}')
    while IFS= read -r shard_id; do
        CLAIM_SCAN_ORDER+=("${shard_id}")
    done < <(_build_claim_scan_order "${N_JOBS}" "${CLAIM_SCAN_SEED}")
    trap '_stop_claim_heartbeat "${CURRENT_HEARTBEAT_PID}"' EXIT

    echo "============================================================"
    echo "[auto] Worker started on $(hostname), pid=$$"
    echo "[auto] Total ${N_JOBS} shards available"
    echo "[auto] Claim scan order: shuffled per worker (seed_hash=${CLAIM_SCAN_SEED_HASH})"
    echo "[auto] Heartbeat interval: ${CLAIM_HEARTBEAT_INTERVAL}s | stale threshold: ${CLAIM_STALE_THRESHOLD}s"
    echo "[auto] Runtime shard workers: ${SHARD_N_WORKERS} (auto-detected on this worker)"
    echo "[auto] MC settings: pair_workers=${MC_PORTFOLIO_PAIR_WORKERS}, sample_chunks=${MC_PORTFOLIO_SAMPLE_CHUNKS}, backend=${MC_PORTFOLIO_PARALLEL_BACKEND}, M=${MC_PORTFOLIO_M}, base_ks=${MC_PORTFOLIO_BASE_KS:-default}, max_candidates=${MC_PORTFOLIO_MAX_CANDIDATE_ACTIONS}, trace=${DEADLINE_TRACE}"
    if [ -n "${AUTO_MAX_CLAIMS}" ]; then
        echo "[auto] Claim limit for this worker: ${AUTO_MAX_CLAIMS}"
    fi
    echo "============================================================"

    while true; do
        if [ -n "${AUTO_MAX_CLAIMS}" ] && [ "${CLAIMS_ATTEMPTED}" -ge "${AUTO_MAX_CLAIMS}" ]; then
            echo "[auto] Claim limit reached. This worker completed ${COMPLETED} shard(s)."
            break
        fi

        if [ "${AUTO_SWEEP_STALE}" != "0" ]; then
            RELEASED=$(_sweep_stale_claims "${N_JOBS}" 1)
            if [ "${RELEASED}" -gt 0 ]; then
                echo "[auto] Released ${RELEASED} stale claim(s) before scanning."
            fi
        fi

        # English note shard,English note
        CLAIMED_ID=-1
        for ((scan_i = 0; scan_i < N_JOBS; scan_i++)); do
            order_idx=$(( (CLAIM_SCAN_CURSOR + scan_i) % N_JOBS ))
            i="${CLAIM_SCAN_ORDER[order_idx]}"
            if mkdir "${CLAIM_DIR}/shard_${i}" 2>/dev/null; then
                CLAIMED_ID=${i}
                CLAIM_SCAN_CURSOR=$(( (order_idx + 1) % N_JOBS ))
                echo "hostname=$(hostname), pid=$$, time=$(date -Iseconds)" \
                    > "${CLAIM_DIR}/shard_${i}/worker_info"
                CURRENT_HEARTBEAT_PID=$(_start_claim_heartbeat "${CLAIM_DIR}/shard_${i}")
                break
            fi
        done

        # English note,English note
        if [ ${CLAIMED_ID} -eq -1 ]; then
            echo ""
            echo "[auto] All shards claimed. This worker completed ${COMPLETED} shard(s)."
            break
        fi

        echo ""
        echo "[auto] Claimed shard ${CLAIMED_ID}/${N_JOBS} (this worker's #$((COMPLETED + 1)))"
        CLAIMS_ATTEMPTED=$((CLAIMS_ATTEMPTED + 1))

        # English note shard
        LINE_NUM=$((CLAIMED_ID + 1))
        CMD=$(sed -n "${LINE_NUM}p" "${JOBS_LIST}")
        CMD=$(_runtime_cmd_with_auto_workers "${CMD}")

        echo "============================================================"
        echo "Running shard ${CLAIMED_ID}/${N_JOBS}:"
        echo "  ${CMD}"
        echo "============================================================"
        cd "${SCRIPT_DIR}"
        set +e
        eval "${CMD}"
        SHARD_EXIT=$?
        set -e

        _stop_claim_heartbeat "${CURRENT_HEARTBEAT_PID}"
        CURRENT_HEARTBEAT_PID=""

        if [ ${SHARD_EXIT} -ne 0 ]; then
            echo "[auto] Shard ${CLAIMED_ID} failed with exit=${SHARD_EXIT}. Releasing claim for retry."
            rm -rf "${CLAIM_DIR}/shard_${CLAIMED_ID}"
            LAST_EXIT=${SHARD_EXIT}
            continue
        fi

        # English note
        echo "done=$(date -Iseconds)" > "${CLAIM_DIR}/shard_${CLAIMED_ID}/done"
        COMPLETED=$((COMPLETED + 1))
        echo "[auto] Shard ${CLAIMED_ID} done. Total completed by this worker: ${COMPLETED}"
    done

    return ${LAST_EXIT}
}

# ─── English note:English note shard English note,English note worker English note ──
# English note:
#   - English note done English note → English note,English note
#   - English note worker_info English note done → English note
#     - English note 10 English note → English note,English note
#     - English note 10 English note → English note,English note claim English note
#   - English note
do_resume() {
    if [ ! -f "${JOBS_LIST}" ]; then
        echo "Error: ${JOBS_LIST} not found. Run 'prepare' first."
        exit 1
    fi

    mkdir -p "${CLAIM_DIR}"
    N_JOBS=$(wc -l < "${JOBS_LIST}")
    NOW=$(date +%s)

    n_done=0
    n_active=0
    n_stale=0
    n_unclaimed=0
    stale_ids=""

    echo "============================================================"
    echo "[resume] Scanning ${N_JOBS} shards ..."
    echo "============================================================"

    for i in $(seq 0 $((N_JOBS - 1))); do
        claim_path="${CLAIM_DIR}/shard_${i}"

        if [ ! -d "${claim_path}" ]; then
            n_unclaimed=$((n_unclaimed + 1))
            continue
        fi

        # English note
        if [ -f "${claim_path}/done" ]; then
            n_done=$((n_done + 1))
            continue
        fi

        # English note → English note heartbeat,English note output / worker_info
        LINE_NUM=$((i + 1))
        CMD=$(sed -n "${LINE_NUM}p" "${JOBS_LIST}")
        br=$(_budget_tag_from_cmd "${CMD}")
        shard_desc=$(_shard_desc_from_cmd "${CMD}")
        age_info=$(_claim_age_info "${claim_path}" "${CMD}" "${NOW}")
        age="${age_info%%|*}"
        age_src="${age_info#*|}"

        if [ ${age} -lt ${CLAIM_STALE_THRESHOLD} ]; then
            n_active=$((n_active + 1))
            echo "  shard_${i}: ACTIVE (${br} ${shard_desc}, ${age_src} ${age}s ago)"
        else
            n_stale=$((n_stale + 1))
            stale_ids="${stale_ids} ${i}"
            echo "  shard_${i}: STALE  (${br} ${shard_desc}, ${age_src} ${age}s ago) → will release"
        fi
    done

    echo ""
    echo "============================================================"
    echo "Summary:"
    echo "  Done:      ${n_done}"
    echo "  Active:    ${n_active}"
    echo "  Stale:     ${n_stale}  (will be released)"
    echo "  Unclaimed: ${n_unclaimed}"
    echo "  Available after resume: $((n_stale + n_unclaimed))"
    echo "============================================================"

    if [ ${n_stale} -eq 0 ]; then
        echo "[resume] No stale shards found. Nothing to clean."
        return
    fi

    echo ""
    echo "[resume] Releasing ${n_stale} stale claim(s) ..."
    for sid in ${stale_ids}; do
        rm -rf "${CLAIM_DIR}/shard_${sid}"
        echo "  Released shard_${sid}"
    done

    echo ""
    echo "[resume] Done. Now submit new workers with:"
    echo "  cd ${SCRIPT_DIR} && bash run_experiment.sh auto"
    echo ""
    echo "  Recommended: submit $((n_stale + n_unclaimed)) single-GPU tasks"
}

# ─── English note:English note,English note ─────────────────────
# English note budget_ratio English note:done/active/stale/unclaimed.
do_status() {
    if [ ! -f "${JOBS_LIST}" ]; then
        echo "Error: ${JOBS_LIST} not found. Run 'prepare' first."
        exit 1
    fi

    mkdir -p "${CLAIM_DIR}"
    STATUS_JOBS="${STATUS_JOBS}" \
    CLAIM_STALE_THRESHOLD="${CLAIM_STALE_THRESHOLD}" \
    CLAIM_NO_OUTPUT_STALE_THRESHOLD="${CLAIM_NO_OUTPUT_STALE_THRESHOLD}" \
    CLAIM_NO_PROGRESS_STALE_THRESHOLD="${CLAIM_NO_PROGRESS_STALE_THRESHOLD}" \
    "${PYTHON}" - "${JOBS_LIST}" "${CLAIM_DIR}" "${SHARD_DIR}" "${RUN_NAME}" "${RUN_ROOT}" "${SCRIPT_DIR}" <<'PY'
import os
import re
import shlex
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

jobs_list, claim_dir, shard_dir, run_name, run_root, script_dir = sys.argv[1:7]
stale_threshold = int(os.environ.get("CLAIM_STALE_THRESHOLD", "1800"))
no_output_stale_threshold = int(os.environ.get("CLAIM_NO_OUTPUT_STALE_THRESHOLD", "900"))
no_progress_stale_threshold = int(os.environ.get("CLAIM_NO_PROGRESS_STALE_THRESHOLD", "7200"))
status_jobs = int(os.environ.get("STATUS_JOBS", "0") or "0")

with open(jobs_list, "r", encoding="utf-8") as f:
    jobs = [line.rstrip("\n") for line in f]

n_jobs = len(jobs)
if status_jobs <= 0:
    status_jobs = min(32, max(1, os.cpu_count() or 1))

now = int(time.time())


def split_cmd(cmd):
    try:
        return shlex.split(cmd)
    except ValueError:
        return cmd.split()


def field(parts, name, default=""):
    try:
        idx = parts.index(name)
    except ValueError:
        return default
    if idx + 1 >= len(parts):
        return default
    return parts[idx + 1]


def deadline_tag(parts):
    dv = field(parts, "--deadline_value")
    dr = field(parts, "--deadline_ratio")
    return f"deadline_value_{dv}" if dv else f"deadline_ratio_{dr or '1.0'}"


def budget_tag(parts):
    bv = field(parts, "--budget_value")
    br = field(parts, "--budget_ratio", "unknown") or "unknown"
    return f"budget_value_{bv}" if bv else f"budget_ratio_{br}"


def raw_budget_stem(parts):
    bv = field(parts, "--budget_value")
    br = field(parts, "--budget_ratio", "unknown") or "unknown"
    return f"raw_budget_value_{bv}" if bv else f"raw_budget_{br}"


def budget_cost_tag(parts):
    mode = field(parts, "--budget_cost_mode", "token_cost_mean")
    return f"budget_cost_{mode or 'token_cost_mean'}"


def budget_deadline_key(parts):
    dv = field(parts, "--deadline_value")
    dr = field(parts, "--deadline_ratio")
    btag = budget_tag(parts)
    return f"{btag}@dv={dv}" if dv else f"{btag}@dr={dr or '1.0'}"


def output_file(parts):
    out_dir = field(parts, "--output_dir", shard_dir) or shard_dir
    raw_stem = raw_budget_stem(parts)
    dtag = deadline_tag(parts)
    bctag = budget_cost_tag(parts)
    ds = field(parts, "--dag_start")
    de = field(parts, "--dag_end")
    ss = field(parts, "--dag_shard_start")
    se = field(parts, "--dag_shard_end")
    if ss and se:
        return os.path.join(out_dir, f"{raw_stem}_{dtag}_{bctag}_sel{ss}-{int(se) - 1}.jsonl")
    if ds and de:
        return os.path.join(out_dir, f"{raw_stem}_{dtag}_{bctag}_dag{ds}-{int(de) - 1}.jsonl")
    return ""


def shard_desc(parts):
    dtag = deadline_tag(parts)
    ds = field(parts, "--dag_start")
    de = field(parts, "--dag_end")
    ss = field(parts, "--dag_shard_start")
    se = field(parts, "--dag_shard_end")
    if ss and se:
        return f"{dtag} sel{ss}-{int(se) - 1}"
    if ds and de:
        return f"{dtag} dag{ds}-{int(de) - 1}"
    return "unknown"


def file_age(path):
    try:
        return now - int(os.stat(path).st_mtime)
    except OSError:
        return None


def claim_age_info(claim_path, parts):
    out_path = output_file(parts)
    output_age = file_age(out_path) if out_path else None
    if output_age is not None and output_age >= no_progress_stale_threshold:
        return output_age, "output_stalled"

    heartbeat_age = file_age(os.path.join(claim_path, "heartbeat"))
    worker_age = file_age(os.path.join(claim_path, "worker_info"))

    if output_age is not None:
        freshest = output_age
        if heartbeat_age is not None and heartbeat_age < freshest:
            return heartbeat_age, "heartbeat"
        return freshest, "output"

    if worker_age is not None and worker_age >= no_output_stale_threshold:
        return worker_age, "no_output"

    if worker_age is not None and worker_age < no_output_stale_threshold:
        if heartbeat_age is not None and heartbeat_age < worker_age:
            return heartbeat_age, "heartbeat_no_output"
        return worker_age, "worker_info_no_output"

    if heartbeat_age is not None and heartbeat_age < no_output_stale_threshold:
        return heartbeat_age, "heartbeat_no_output"

    if heartbeat_age is not None:
        return heartbeat_age, "no_output"
    if worker_age is not None:
        return worker_age, "no_output"
    return stale_threshold + 1, "missing"


def scan_one(item):
    i, cmd = item
    parts = split_cmd(cmd)
    br = budget_tag(parts)
    br_key = budget_deadline_key(parts)
    claim_path = os.path.join(claim_dir, f"shard_{i}")

    if not os.path.isdir(claim_path):
        return {"idx": i, "br": br, "key": br_key, "state": "unclaimed"}

    if os.path.isfile(os.path.join(claim_path, "done")):
        return {"idx": i, "br": br, "key": br_key, "state": "done"}

    age, age_src = claim_age_info(claim_path, parts)
    state = "active" if age < stale_threshold else "stale"
    return {
        "idx": i,
        "br": br,
        "key": br_key,
        "state": state,
        "age": age,
        "age_src": age_src,
        "desc": shard_desc(parts),
    }


print("============================================================")
print(f"[status] Scanning {n_jobs} shards ...")
print(f"  Run name: {run_name}")
print(f"  Run root: {run_root}")
print(f"  Status jobs: {status_jobs}")
print("============================================================")

with ThreadPoolExecutor(max_workers=status_jobs) as executor:
    results = list(executor.map(scan_one, enumerate(jobs)))

counts = defaultdict(int)
by_key = defaultdict(lambda: defaultdict(int))

for result in sorted(results, key=lambda x: x["idx"]):
    state = result["state"]
    key = result["key"]
    counts[state] += 1
    by_key[key]["total"] += 1
    by_key[key][state] += 1
    if state == "active":
        print(
            f"  shard_{result['idx']}: ACTIVE "
            f"(br={result['br']} {result['desc']}, {result['age_src']} {result['age']}s ago)"
        )
    elif state == "stale":
        print(
            f"  shard_{result['idx']}: STALE  "
            f"(br={result['br']} {result['desc']}, {result['age_src']} {result['age']}s ago)"
        )

n_done = counts["done"]
n_active = counts["active"]
n_stale = counts["stale"]
n_unclaimed = counts["unclaimed"]

print("")
print("============================================================")
print("Summary:")
print(f"  Done:      {n_done}")
print(f"  Active:    {n_active}")
print(f"  Stale:     {n_stale}")
print(f"  Unclaimed: {n_unclaimed}")
print(f"  Remaining to run/recover: {n_stale + n_unclaimed}")
print("============================================================")

print("")
print("Per-(budget,deadline) shard status:")
print(f"{'budget@deadline':<22} {'total':>8} {'done':>8} {'active':>8} {'stale':>10} {'unclaimed':>11}")
print(f"{'----------------------':<22} {'--------':>8} {'--------':>8} {'--------':>8} {'----------':>10} {'-----------':>11}")


def key_sort(value):
    return [float(x) if re.fullmatch(r"-?\d+(?:\.\d+)?", x) else x for x in re.split(r"([0-9.]+)", value)]


for key in sorted(by_key, key=key_sort):
    stats = by_key[key]
    print(
        f"{key:<22} "
        f"{stats['total']:>8d} "
        f"{stats['done']:>8d} "
        f"{stats['active']:>8d} "
        f"{stats['stale']:>10d} "
        f"{stats['unclaimed']:>11d}"
    )

print("")
print("Tips:")
print("  - Auto workers now sweep stale claims automatically using heartbeat files.")
print("  - If stale > 0: run 'bash run_experiment.sh resume' to release them immediately.")
print(f"  - If unclaimed > 0: submit more workers: 'cd {script_dir} && bash run_experiment.sh auto'")
print("  - To validate merged coverage by budget: bash run_experiment.sh merge")
PY
}

do_merge() {
    ensure_deps
    echo "============================================================"
    echo "Merging shards ..."
    echo "============================================================"
    ${PYTHON} "${SCRIPT_DIR}/merge_shards.py" \
        --shard_dir "${SHARD_DIR}" \
        --output_dir "${MERGED_DIR}" \
        --n_dags ${N_DAGS} \
        --n_exec_trials ${N_EXEC_TRIALS} \
        --copy-mode "${MERGE_COPY_MODE}" \
        --copy-buffer-mb "${MERGE_COPY_BUFFER_MB}" \
        --jobs "${MERGE_JOBS}"
}

do_analyze() {
    ensure_deps
    echo "============================================================"
    echo "Running analysis ..."
    echo "============================================================"
    ANALYZE_ARGS=("--jobs" "${ANALYZE_JOBS}")
    if [ "${ANALYZE_SKIP_DEADLINE_AUC}" = "1" ]; then
        ANALYZE_ARGS+=("--skip-deadline-auc")
    fi
    if [ "${ANALYZE_SKIP_SOLVE_SUMMARY}" = "1" ]; then
        ANALYZE_ARGS+=("--skip-solve-summary")
    fi
    if [ "${ANALYZE_SKIP_COMPLETE}" != "1" ]; then
        ANALYZE_ARGS+=("--no-skip-complete")
    fi
    if [ "${ANALYZE_PROGRESS}" != "1" ]; then
        ANALYZE_ARGS+=("--no-progress")
    fi
    ${PYTHON} "${SCRIPT_DIR}/analyze.py" "${MERGED_DIR}" "${ANALYZE_ARGS[@]}"
    echo ""
    echo "Results in: ${MERGED_DIR}/"
}

do_report() {
    ensure_deps
    if [ "${REPORT_JOBS}" = "0" ]; then
        REPORT_JOBS_EFFECTIVE=$(${PYTHON} - <<'PY'
import os
print(min(32, max(1, os.cpu_count() or 1)))
PY
)
    else
        REPORT_JOBS_EFFECTIVE="${REPORT_JOBS}"
    fi
    echo "============================================================"
    echo "Generating budget/failure report ..."
    echo "============================================================"
    echo "Report jobs: ${REPORT_JOBS_EFFECTIVE}"
    REPORT_ARGS=(
        "--input" "${MERGED_DIR}"
        "--output-dir" "${REPORT_DIR}"
        "--jobs" "${REPORT_JOBS_EFFECTIVE}"
    )
    if [ "${REPORT_WRITE_RECORDS}" = "1" ]; then
        REPORT_ARGS+=("--write-records")
    fi
    if [ "${REPORT_PROGRESS}" != "1" ]; then
        REPORT_ARGS+=("--no-progress")
    fi
    ${PYTHON} "${SCRIPT_DIR}/analyze_budget_failure_report.py" "${REPORT_ARGS[@]}"
    echo ""
    echo "Report in: ${REPORT_DIR}/"
}

do_all() {
    echo "WARNING: Running all shards sequentially on this machine."
    echo "This is for debugging/small-scale only."
    echo ""

    do_prepare

    N_JOBS=$(wc -l < "${JOBS_LIST}")
    echo "Running ${N_JOBS} shards sequentially ..."
    for i in $(seq 0 $((N_JOBS - 1))); do
        do_shard ${i}
    done

    do_merge
    do_analyze
    do_report
}

_suite_entries() {
    if [ -z "${BASELINE_DAG_POOL_MAP}" ]; then
        return 0
    fi
    local raw entry model path
    raw="${BASELINE_DAG_POOL_MAP//,/;}"
    IFS=';' read -r -a entries <<< "${raw}"
    for entry in "${entries[@]}"; do
        entry="$(echo "${entry}" | xargs)"
        if [ -z "${entry}" ]; then
            continue
        fi
        if [[ "${entry}" == *"="* ]]; then
            model="${entry%%=*}"
            path="${entry#*=}"
        else
            model="${entry%%:*}"
            path="${entry#*:}"
        fi
        model="$(echo "${model}" | xargs)"
        path="$(echo "${path}" | xargs)"
        if [ -z "${model}" ] || [ -z "${path}" ] || [ "${model}" = "${path}" ]; then
            echo "Invalid BASELINE_DAG_POOL_MAP entry: ${entry}" >&2
            return 1
        fi
        echo "${model}|${path}"
    done
}

_suite_run_one() {
    local action="$1"
    local run_name="$2"
    local dag_pool_path="$3"
    local only_loader="$4"
    local baseline_only="$5"
    local default_model_id="${6:-}"
    local baseline_method_set="${7:-all}"

    echo ""
    echo "============================================================"
    echo "[suite] ${action}: RUN_NAME=${run_name}"
    echo "[suite] DAG_POOL_PATH=${dag_pool_path}"
    echo "============================================================"

    if [ -n "${default_model_id}" ]; then
        RUN_NAME="${run_name}" \
        DAG_POOL_PATH="${dag_pool_path}" \
        ONLY_LOADER="${only_loader}" \
        BASELINE_ONLY="${baseline_only}" \
        BASELINE_METHOD_SET="${baseline_method_set}" \
        SHARD_N_WORKERS="${REQUESTED_SHARD_N_WORKERS}" \
        MC_PORTFOLIO_PAIR_WORKERS="${REQUESTED_MC_PORTFOLIO_PAIR_WORKERS}" \
        MC_PORTFOLIO_SAMPLE_CHUNKS="${REQUESTED_MC_PORTFOLIO_SAMPLE_CHUNKS}" \
        MC_PORTFOLIO_PARALLEL_BACKEND="${REQUESTED_MC_PORTFOLIO_PARALLEL_BACKEND}" \
        OUTPUT_MONEY_DEFAULT_MODEL_ID="${default_model_id}" \
        bash "${SCRIPT_DIR}/run_experiment.sh" "${action}"
    else
        RUN_NAME="${run_name}" \
        DAG_POOL_PATH="${dag_pool_path}" \
        ONLY_LOADER="${only_loader}" \
        BASELINE_ONLY="${baseline_only}" \
        BASELINE_METHOD_SET="${baseline_method_set}" \
        SHARD_N_WORKERS="${REQUESTED_SHARD_N_WORKERS}" \
        MC_PORTFOLIO_PAIR_WORKERS="${REQUESTED_MC_PORTFOLIO_PAIR_WORKERS}" \
        MC_PORTFOLIO_SAMPLE_CHUNKS="${REQUESTED_MC_PORTFOLIO_SAMPLE_CHUNKS}" \
        MC_PORTFOLIO_PARALLEL_BACKEND="${REQUESTED_MC_PORTFOLIO_PARALLEL_BACKEND}" \
        bash "${SCRIPT_DIR}/run_experiment.sh" "${action}"
    fi
}

do_suite_action() {
    local action="$1"
    local entry model path
    local failed=0
    if [ "${SUITE_INCLUDE_PORTFOLIO}" = "1" ]; then
        if ! _suite_run_one "${action}" "${SUITE_PORTFOLIO_RUN_NAME}" "${PORTFOLIO_DAG_POOL_PATH}" 1 0 "" "all"; then
            echo "[suite] ${action} failed for RUN_NAME=${SUITE_PORTFOLIO_RUN_NAME}; continuing with remaining runs."
            failed=1
        fi
    fi
    if [ -z "${BASELINE_DAG_POOL_MAP}" ]; then
        echo "[suite] BASELINE_DAG_POOL_MAP is empty; no baseline runs configured."
        return "${failed}"
    fi
    while IFS= read -r entry; do
        [ -z "${entry}" ] && continue
        model="${entry%%|*}"
        path="${entry#*|}"
        if ! _suite_run_one "${action}" "${SUITE_RUN_NAME_PREFIX}_base${model}" "${path}" 0 1 "${model}" "${SUITE_BASELINE_METHOD_SET}"; then
            echo "[suite] ${action} failed for RUN_NAME=${SUITE_RUN_NAME_PREFIX}_base${model}; continuing with remaining runs."
            failed=1
        fi
    done < <(_suite_entries)
    return "${failed}"
}

# ─── English note ───────────────────────────────────────────

ACTION=${1:-help}

case "${ACTION}" in
    suite_prepare)
        do_suite_action prepare
        ;;
    suite_auto)
        do_suite_action auto
        ;;
    suite_status)
        do_suite_action status
        ;;
    suite_merge)
        do_suite_action merge
        ;;
    prepare)
        do_prepare
        ;;
    auto)
        do_auto
        ;;
    status)
        do_status
        ;;
    resume)
        do_resume
        ;;
    shard)
        do_shard $2
        ;;
    merge)
        do_merge
        ;;
    analyze)
        do_analyze
        ;;
    report)
        do_report
        ;;
    all)
        do_all
        ;;
    *)
        echo "LOADER Simulation Experiment Runner"
        echo ""
        echo "Usage: bash run_experiment.sh <action> [args]"
        echo ""
        echo "Actions:"
        echo "  prepare          Generate DAG pool + job list"
        echo "  suite_prepare    Prepare portfolio + configured single-model baselines"
        echo "  suite_auto       Auto-run portfolio + configured single-model baselines"
        echo "  suite_status     Show status for portfolio + configured baselines"
        echo "  suite_merge      Merge portfolio + configured baselines"
        echo "  auto             Auto-claim & run next unclaimed shard (for QS batch)"
        echo "  status           Show shard progress by budget (non-destructive)"
        echo "  resume           Release crashed shard locks & show status"
        echo "  shard <id>       Run a specific shard (id = 0-based)"
        echo "  merge            Merge all shard results"
        echo "  analyze          Run analysis on merged results"
        echo "  report           Generate budget/failure report from merged results"
        echo "  all              Run everything sequentially (debug only)"
        echo ""
        echo "QS batch workflow:"
        echo "  1. Run once:  bash run_experiment.sh prepare"
        echo "  2. Submit N copies of the SAME command to QS:"
        echo "     cd ${SCRIPT_DIR} && bash run_experiment.sh auto"
        echo "  3. If some tasks crashed, fix code then:"
        echo "     bash run_experiment.sh resume   # release stale locks"
        echo "     # then re-submit auto tasks"
        echo "  4. After all done:  bash run_experiment.sh merge"
        echo "  5. bash run_experiment.sh analyze"
        echo "  6. bash run_experiment.sh report"
        echo ""
        echo "Path notes:"
        echo "  This run uses RUN_NAME=${RUN_NAME}"
        echo "  All artifacts are under ${RUN_ROOT}"
        echo ""
        echo "Atomic QS job mode:"
        echo "  QS_ONE_DAG_PER_JOB=1 RUN_NAME=<new_run> bash run_experiment.sh prepare"
        echo "  QS_ONE_DAG_PER_JOB=1 RUN_NAME=<new_run> bash run_experiment.sh auto"
        echo ""
        ;;
esac
