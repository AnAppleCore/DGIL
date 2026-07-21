#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hongwei/miniconda3/envs/DGIL/bin/python}"
DATA_ROOT="${DGIL_DATA_ROOT:-/data2/datasets}"
GPUS_CSV="${DGIL_GPUS:-0,1,2,3,4,5,6,7}"
SESSION_PREFIX="${SESSION_PREFIX:-dgil_medical_dil}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/screen_logs/${SESSION_PREFIX}_${RUN_STAMP}}"
STATE_ROOT="${STATE_ROOT:-${TMPDIR:-/tmp}/DGIL/${SESSION_PREFIX}_${RUN_STAMP}}"

cd "${PROJECT_ROOT}"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if [[ ${#GPUS[@]} -ne 8 ]]; then
    echo "Error: DGIL_GPUS must contain exactly 8 GPU ids, got: ${GPUS_CSV}" >&2
    exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: Python interpreter is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! command -v screen >/dev/null 2>&1; then
    echo "Error: screen is not installed or unavailable." >&2
    exit 1
fi

config_log_candidates() {
    local config="$1"
    CONFIG_PATH="${config}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

config = json.loads(Path(os.environ["CONFIG_PATH"]).read_text())
raw_init_cls = config["init_cls"]
increment = config["increment"]
init_cls = 0 if raw_init_cls == increment else raw_init_cls
log_dir = Path("logs") / config["model_name"] / config["dataset"] / str(init_cls) / str(increment)
for seed in config["seed"]:
    filename = f'{config["prefix"]}_{seed}_{config["backbone_type"]}.log'
    print(log_dir / filename)
PY
}

is_config_complete() {
    local config="$1"
    local candidates
    mapfile -t candidates < <(config_log_candidates "${config}")
    if [[ ${#candidates[@]} -ne 3 ]]; then
        return 1
    fi

    local log_file
    for log_file in "${candidates[@]}"; do
        if [[ ! -s "${log_file}" ]]; then
            return 1
        fi
        # Use fixed-string match; basic-regex \( \) would miss literal parentheses.
        if ! grep -Fq "Accuracy Matrix (CNN, Domains)" "${log_file}"; then
            return 1
        fi
        if ! grep -Fq "Last Accuracy In (CNN):" "${log_file}"; then
            return 1
        fi
        if ! grep -Fq "Last Accuracy Out (CNN):" "${log_file}"; then
            return 1
        fi
        if ! grep -Fq "Forgetting (CNN):" "${log_file}"; then
            return 1
        fi
    done
    return 0
}

validate_config() {
    local config="$1"
    CONFIG_PATH="${config}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CONFIG_PATH"])
config = json.loads(path.read_text())
assert config["dataset"] in {"midog25", "camelyon17"}, path
assert config["seed"] == [1994, 1995, 1996], path
assert config["device"] == ["0"], path
assert config["domain_incremental"] is True, path
assert config["print_forget"] is True, path
assert config["init_cls"] == 2 and config["increment"] == 0, path
PY
}

run_config() {
    local gpu_id="$1"
    local config="$2"
    local queue_log="$3"

    echo "[$(date '+%F %T')] CHECK ${config}" | tee -a "${queue_log}"
    if is_config_complete "${config}"; then
        echo "[$(date '+%F %T')] SKIP completed ${config}" | tee -a "${queue_log}"
        return 0
    fi

    echo "[$(date '+%F %T')] START ${config} on physical GPU ${gpu_id}" | tee -a "${queue_log}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu_id}" DGIL_DATA_ROOT="${DATA_ROOT}" \
        "${PYTHON_BIN}" main.py --config "${config}" 2>&1 | tee -a "${queue_log}"
    local status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -ne 0 ]]; then
        echo "[$(date '+%F %T')] FAILED ${config} on GPU ${gpu_id}, exit ${status}" | tee -a "${queue_log}"
        return "${status}"
    fi
    if ! is_config_complete "${config}"; then
        echo "[$(date '+%F %T')] FAILED completion check ${config}" | tee -a "${queue_log}"
        return 1
    fi
    echo "[$(date '+%F %T')] DONE ${config} on physical GPU ${gpu_id}" | tee -a "${queue_log}"
}

wait_for_midog_barrier() {
    local queue_log="$1"
    echo "[$(date '+%F %T')] Waiting for all MIDOG25 queues." | tee -a "${queue_log}"
    while true; do
        if compgen -G "${STATE_ROOT}/midog_failed_gpu*" >/dev/null; then
            echo "[$(date '+%F %T')] MIDOG25 phase failed on another GPU; Camelyon17 will not start." | tee -a "${queue_log}"
            return 1
        fi

        local completed=0
        local gpu_id
        for gpu_id in "${GPUS[@]}"; do
            if [[ -f "${STATE_ROOT}/midog_done_gpu${gpu_id}" ]]; then
                completed=$((completed + 1))
            fi
        done
        if [[ ${completed} -eq 8 ]]; then
            echo "[$(date '+%F %T')] MIDOG25 barrier complete; starting Camelyon17." | tee -a "${queue_log}"
            return 0
        fi
        sleep 30
    done
}

run_worker() {
    local gpu_id="$1"
    shift
    local midog_count="$1"
    shift
    local queue_log="${LOG_ROOT}/gpu${gpu_id}.log"
    local -a midog_configs=()
    if [[ "${midog_count}" -gt 0 ]]; then
        midog_configs=("${@:1:${midog_count}}")
        shift "${midog_count}"
    fi
    local -a camelyon_configs=("$@")

    echo "[$(date '+%F %T')] Worker started on physical GPU ${gpu_id}." | tee -a "${queue_log}"
    echo "Python: ${PYTHON_BIN}" | tee -a "${queue_log}"
    echo "DGIL_DATA_ROOT: ${DATA_ROOT}" | tee -a "${queue_log}"

    local config
    for config in "${midog_configs[@]+"${midog_configs[@]}"}"; do
        if ! run_config "${gpu_id}" "${config}" "${queue_log}"; then
            touch "${STATE_ROOT}/midog_failed_gpu${gpu_id}"
            exit 1
        fi
    done
    touch "${STATE_ROOT}/midog_done_gpu${gpu_id}"

    if ! wait_for_midog_barrier "${queue_log}"; then
        exit 1
    fi

    if [[ "${gpu_id}" == "3" && ${#camelyon_configs[@]} -eq 3 ]]; then
        echo "[$(date '+%F %T')] Starting parallel Camelyon17 pair on GPU 3: ${camelyon_configs[0]} and ${camelyon_configs[1]}" | tee -a "${queue_log}"
        set +e
        run_config "${gpu_id}" "${camelyon_configs[0]}" "${queue_log}" &
        local first_pid=$!
        run_config "${gpu_id}" "${camelyon_configs[1]}" "${queue_log}" &
        local second_pid=$!
        wait "${first_pid}"
        local first_status=$?
        wait "${second_pid}"
        local second_status=$?
        set -e
        if [[ ${first_status} -ne 0 || ${second_status} -ne 0 ]]; then
            echo "[$(date '+%F %T')] Parallel Camelyon17 pair failed on GPU 3: statuses ${first_status}/${second_status}" | tee -a "${queue_log}"
            touch "${STATE_ROOT}/camelyon_failed_gpu${gpu_id}"
            exit 1
        fi
        if ! run_config "${gpu_id}" "${camelyon_configs[2]}" "${queue_log}"; then
            touch "${STATE_ROOT}/camelyon_failed_gpu${gpu_id}"
            exit 1
        fi
    else
        for config in "${camelyon_configs[@]+"${camelyon_configs[@]}"}"; do
            if ! run_config "${gpu_id}" "${config}" "${queue_log}"; then
                touch "${STATE_ROOT}/camelyon_failed_gpu${gpu_id}"
                exit 1
            fi
        done
    fi
    touch "${STATE_ROOT}/camelyon_done_gpu${gpu_id}"
    echo "[$(date '+%F %T')] Worker finished on physical GPU ${gpu_id}." | tee -a "${queue_log}"
}

print_plan() {
    echo "MIDOG25 phase"
    echo "  GPU ${GPUS[0]}: ${MIDOG_0[*]:-(none)}"
    echo "  GPU ${GPUS[1]}: ${MIDOG_1[*]:-(none)}"
    echo "  GPU ${GPUS[2]}: ${MIDOG_2[*]:-(none)}"
    echo "  GPU ${GPUS[3]}: ${MIDOG_3[*]:-(none)}"
    echo "  GPU ${GPUS[4]}: ${MIDOG_4[*]:-(none)}"
    echo "  GPU ${GPUS[5]}: ${MIDOG_5[*]:-(none)}"
    echo "  GPU ${GPUS[6]}: ${MIDOG_6[*]:-(none)}"
    echo "  GPU ${GPUS[7]}: ${MIDOG_7[*]:-(none)}"
    echo
    echo "Camelyon17 phase starts only after all MIDOG25 queues finish"
    echo "  GPU ${GPUS[0]}: ${CAM_0[*]:-(none)}"
    echo "  GPU ${GPUS[1]}: ${CAM_1[*]:-(none)}"
    echo "  GPU ${GPUS[2]}: ${CAM_2[*]:-(none)}"
    echo "  GPU ${GPUS[3]}: ${CAM_3[*]:-(none)}"
    echo "  GPU ${GPUS[4]}: ${CAM_4[*]:-(none)}"
    echo "  GPU ${GPUS[5]}: ${CAM_5[*]:-(none)}"
    echo "  GPU ${GPUS[6]}: ${CAM_6[*]:-(none)}"
    echo "  GPU ${GPUS[7]}: ${CAM_7[*]:-(none)}"
}

# Full schedule (default). Already-complete configs are skipped at runtime.
MIDOG_0=(configs/DGIL/midog25/dot_l2p_dgil.json)
MIDOG_1=(configs/DGIL/midog25/codag_dgil.json)
MIDOG_2=(configs/DGIL/midog25/dot_slca_dgil.json)
MIDOG_3=(configs/DGIL/midog25/coda_prompt_dgil.json)
MIDOG_4=(configs/DGIL/midog25/l2p_dgil.json configs/DGIL/midog25/dualprompt_dgil.json)
MIDOG_5=(configs/DGIL/midog25/simon_msl_dgil.json)
MIDOG_6=(configs/DGIL/midog25/slca_dgil.json configs/DGIL/midog25/ranpac_dgil.json)
MIDOG_7=(configs/DGIL/midog25/cmoa_dgil.json)

CAM_0=(configs/DGIL/camelyon17/dot_l2p_dgil.json)
CAM_1=(configs/DGIL/camelyon17/codag_dgil.json)
CAM_2=(configs/DGIL/camelyon17/dot_slca_dgil.json)
CAM_3=(configs/DGIL/camelyon17/coda_prompt_dgil.json)
CAM_4=(configs/DGIL/camelyon17/l2p_dgil.json)
CAM_5=(configs/DGIL/camelyon17/dualprompt_dgil.json)
CAM_6=(configs/DGIL/camelyon17/simon_msl_dgil.json configs/DGIL/camelyon17/slca_dgil.json)
CAM_7=(configs/DGIL/camelyon17/cmoa_dgil.json configs/DGIL/camelyon17/ranpac_dgil.json)

# Corrected medical schedule: rerun the three MIDOG25 methods whose CA stage was disabled,
# then run all Camelyon17 methods on the fixed 10k-per-domain training subset.
# GPU 3 runs DualPrompt and CODA-Prompt concurrently, then starts CMoA after both finish.
apply_remaining_schedule() {
    SESSION_PREFIX="${SESSION_PREFIX:-dgil_medical_dil_quick}"
    if [[ "${SESSION_PREFIX}" == "dgil_medical_dil" ]]; then
        SESSION_PREFIX="dgil_medical_dil_quick"
    fi
    LOG_ROOT="${PROJECT_ROOT}/logs/screen_logs/${SESSION_PREFIX}_${RUN_STAMP}"
    STATE_ROOT="${TMPDIR:-/tmp}/DGIL/${SESSION_PREFIX}_${RUN_STAMP}"

    MIDOG_0=(configs/DGIL/midog25/dot_l2p_dgil.json)
    MIDOG_1=()
    MIDOG_2=(configs/DGIL/midog25/dot_slca_dgil.json)
    MIDOG_3=()
    MIDOG_4=()
    MIDOG_5=()
    MIDOG_6=(configs/DGIL/midog25/slca_dgil.json)
    MIDOG_7=()

    CAM_0=(configs/DGIL/camelyon17/dot_l2p_dgil.json)
    CAM_1=(configs/DGIL/camelyon17/codag_dgil.json)
    CAM_2=(configs/DGIL/camelyon17/dot_slca_dgil.json)
    CAM_3=(configs/DGIL/camelyon17/dualprompt_dgil.json configs/DGIL/camelyon17/coda_prompt_dgil.json configs/DGIL/camelyon17/cmoa_dgil.json)
    CAM_4=(configs/DGIL/camelyon17/l2p_dgil.json)
    CAM_5=(configs/DGIL/camelyon17/simon_msl_dgil.json)
    CAM_6=(configs/DGIL/camelyon17/slca_dgil.json)
    CAM_7=(configs/DGIL/camelyon17/ranpac_dgil.json)
}

MODE="full"
if [[ "${1:-}" == "--remaining" ]]; then
    MODE="remaining"
    apply_remaining_schedule
    shift
fi

ALL_CONFIGS=(
    "${MIDOG_0[@]+"${MIDOG_0[@]}"}"
    "${MIDOG_1[@]+"${MIDOG_1[@]}"}"
    "${MIDOG_2[@]+"${MIDOG_2[@]}"}"
    "${MIDOG_3[@]+"${MIDOG_3[@]}"}"
    "${MIDOG_4[@]+"${MIDOG_4[@]}"}"
    "${MIDOG_5[@]+"${MIDOG_5[@]}"}"
    "${MIDOG_6[@]+"${MIDOG_6[@]}"}"
    "${MIDOG_7[@]+"${MIDOG_7[@]}"}"
    "${CAM_0[@]+"${CAM_0[@]}"}"
    "${CAM_1[@]+"${CAM_1[@]}"}"
    "${CAM_2[@]+"${CAM_2[@]}"}"
    "${CAM_3[@]+"${CAM_3[@]}"}"
    "${CAM_4[@]+"${CAM_4[@]}"}"
    "${CAM_5[@]+"${CAM_5[@]}"}"
    "${CAM_6[@]+"${CAM_6[@]}"}"
    "${CAM_7[@]+"${CAM_7[@]}"}"
)

for config in "${ALL_CONFIGS[@]}"; do
    validate_config "${config}"
done

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Mode: ${MODE}"
    print_plan
    echo
    echo "Validated ${#ALL_CONFIGS[@]} configs. No screen sessions started."
    exit 0
fi

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "$@"
    exit 0
fi

mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"

for gpu_id in "${GPUS[@]}"; do
    session_name="${SESSION_PREFIX}_gpu${gpu_id}"
    if screen -list | grep -q "[.]${session_name}[[:space:]]"; then
        echo "Error: screen session ${session_name} already exists." >&2
        exit 1
    fi
done

started=0
for index in "${!GPUS[@]}"; do
    gpu_id="${GPUS[$index]}"
    declare -n midog_ref="MIDOG_${index}"
    declare -n cam_ref="CAM_${index}"
    midog_queue=("${midog_ref[@]+"${midog_ref[@]}"}")
    camelyon_queue=("${cam_ref[@]+"${cam_ref[@]}"}")
    if [[ ${#midog_queue[@]} -eq 0 && ${#camelyon_queue[@]} -eq 0 ]]; then
        # Still mark both phases done so other GPUs can pass the barrier.
        touch "${STATE_ROOT}/midog_done_gpu${gpu_id}"
        touch "${STATE_ROOT}/camelyon_done_gpu${gpu_id}"
        echo "Skip empty queue on GPU ${gpu_id}."
        continue
    fi
    session_name="${SESSION_PREFIX}_gpu${gpu_id}"

    LOG_ROOT="${LOG_ROOT}" STATE_ROOT="${STATE_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    DGIL_DATA_ROOT="${DATA_ROOT}" DGIL_GPUS="${GPUS_CSV}" SESSION_PREFIX="${SESSION_PREFIX}" \
        screen -dmS "${session_name}" bash "${BASH_SOURCE[0]}" --worker \
        "${gpu_id}" "${#midog_queue[@]}" ${midog_queue[@]+"${midog_queue[@]}"} ${camelyon_queue[@]+"${camelyon_queue[@]}"}
    echo "Started ${session_name}."
    started=$((started + 1))
done

echo
echo "Mode: ${MODE}"
print_plan
echo
echo "Started ${started} screen workers."
echo "Each config runs seeds [1994, 1995, 1996] sequentially."
echo "Already-complete configs are skipped automatically."
echo "Screen logs: ${LOG_ROOT}"
echo "Barrier state: ${STATE_ROOT}"
echo "List sessions: screen -ls"
echo "Attach example: screen -r ${SESSION_PREFIX}_gpu${GPUS[3]}"
