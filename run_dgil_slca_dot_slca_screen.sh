#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-DGIL}"
DATA_ROOT="${DGIL_DATA_ROOT:-/data2/datasets}"
LOG_ROOT="${PROJECT_ROOT}/logs/screen_logs/dgil_slca_dot_slca_$(date +%Y%m%d_%H%M%S)"
SESSION_PREFIX="${SESSION_PREFIX:-dgil_slca}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

if ! command -v screen >/dev/null 2>&1; then
    echo "Error: screen is not installed or not available in PATH." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not available in PATH." >&2
    exit 1
fi

run_queue() {
    local gpu_id="$1"
    shift
    local log_file="${LOG_ROOT}/gpu${gpu_id}.log"

    echo "[$(date '+%F %T')] GPU ${gpu_id} queue started." | tee -a "${log_file}"
    echo "Project root: ${PROJECT_ROOT}" | tee -a "${log_file}"
    echo "Conda env: ${CONDA_ENV}" | tee -a "${log_file}"
    echo "DGIL_DATA_ROOT: ${DATA_ROOT}" | tee -a "${log_file}"
    echo "Configs:" | tee -a "${log_file}"
    printf '  %s\n' "$@" | tee -a "${log_file}"

    for config in "$@"; do
        echo | tee -a "${log_file}"
        echo "[$(date '+%F %T')] START ${config} on physical GPU ${gpu_id}" | tee -a "${log_file}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" DGIL_DATA_ROOT="${DATA_ROOT}" \
            conda run --no-capture-output -n "${CONDA_ENV}" python main.py --config "${config}" \
            2>&1 | tee -a "${log_file}"
        status=${PIPESTATUS[0]}
        if [[ ${status} -ne 0 ]]; then
            echo "[$(date '+%F %T')] FAILED ${config} on physical GPU ${gpu_id} with exit code ${status}" | tee -a "${log_file}"
            exit "${status}"
        fi
        echo "[$(date '+%F %T')] DONE ${config} on physical GPU ${gpu_id}" | tee -a "${log_file}"
    done

    echo "[$(date '+%F %T')] GPU ${gpu_id} queue finished." | tee -a "${log_file}"
}

if [[ "${1:-}" == "--run-queue" ]]; then
    shift
    run_queue "$@"
    exit 0
fi

start_screen_queue() {
    local gpu_id="$1"
    shift
    local session_name="${SESSION_PREFIX}_gpu${gpu_id}"

    if screen -list | grep -q "[.]${session_name}[[:space:]]"; then
        echo "Error: screen session ${session_name} already exists. Attach or stop it first." >&2
        exit 1
    fi

    LOG_ROOT="${LOG_ROOT}" CONDA_ENV="${CONDA_ENV}" DGIL_DATA_ROOT="${DATA_ROOT}" SESSION_PREFIX="${SESSION_PREFIX}" \
        screen -dmS "${session_name}" bash "${BASH_SOURCE[0]}" --run-queue "${gpu_id}" "$@"
    echo "Started screen session ${session_name} for physical GPU ${gpu_id}."
}

# Queue design:
# - One detached screen per physical GPU.
# - Experiments inside each screen are sequential.
# - Config files keep device [\"0\"]; CUDA_VISIBLE_DEVICES maps that logical cuda:0 to the selected physical GPU.

start_screen_queue 0 \
    configs/DGIL_baseline_3_run/officehome/slca_dgil.json \
    configs/DGIL/officehome/dot_slca_dgil_randaug.json \
    configs/DGIL/core50/dot_slca_dgil_randaug.json

start_screen_queue 1 \
    configs/DGIL_baseline_3_run/digitsdg/slca_dgil.json \
    configs/DGIL/digitsdg/dot_slca_dgil_randaug.json \
    configs/DGIL/domainnet/dot_slca_dgil_randaug.json

start_screen_queue 2 \
    configs/DGIL_baseline_3_run/core50/slca_dgil.json \
    configs/DGIL_baseline_3_run/domainnet/slca_dgil.json

echo
echo "All screen sessions started. Logs are under: ${LOG_ROOT}"
echo "List sessions: screen -ls"
echo "Attach examples:"
echo "  screen -r ${SESSION_PREFIX}_gpu0"
echo "  screen -r ${SESSION_PREFIX}_gpu1"
echo "  screen -r ${SESSION_PREFIX}_gpu2"
echo "Detach from a session with: Ctrl-a then d"
