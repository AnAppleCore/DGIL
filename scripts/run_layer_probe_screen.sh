#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-DGIL}"
DATA_ROOT="${DGIL_DATA_ROOT:-/data2/datasets}"
GPUS_CSV="${LAYER_PROBE_GPUS:-5,6,7}"
SESSION_PREFIX="${SESSION_PREFIX:-layer_probe}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/layer_probe}"
NORMALIZATION="${NORMALIZATION:-repo}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LINEAR_METHOD="${LINEAR_METHOD:-ridge}"
PREPROCESS="${PREPROCESS:-standard}"
CPU_THREADS="${LAYER_PROBE_CPU_THREADS:-4}"
DATASETS_CSV="${LAYER_PROBE_DATASETS:-digitsdg,officehome,core50,domainnet}"
BACKBONES_CSV="${LAYER_PROBE_BACKBONES:-default,ibot,mae,dinov2,clip}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/${OUTPUT_DIR}/logs/${SESSION_PREFIX}_${RUN_STAMP}}"

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

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV}"
IFS=',' read -r -a BACKBONES <<< "${BACKBONES_CSV}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "Error: no GPUs specified in LAYER_PROBE_GPUS." >&2
    exit 1
fi

if screen -ls | grep -q "[.]${SESSION_PREFIX}_gpu"; then
    echo "Error: existing ${SESSION_PREFIX}_gpu screen sessions found. Refusing to duplicate." >&2
    screen -ls | grep "[.]${SESSION_PREFIX}_gpu" >&2 || true
    exit 1
fi

TASK_FILE="${LOG_ROOT}/tasks.tsv"
: > "${TASK_FILE}"
for dataset in "${DATASETS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        printf '%s\t%s\n' "${dataset}" "${backbone}" >> "${TASK_FILE}"
    done
done

TOTAL_TASKS=$(wc -l < "${TASK_FILE}")
echo "Prepared ${TOTAL_TASKS} layer-probe tasks in ${TASK_FILE}"
echo "Using GPUs: ${GPUS_CSV}"
echo "Logs: ${LOG_ROOT}"

for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    session_name="${SESSION_PREFIX}_gpu${gpu}"
    worker_log="${LOG_ROOT}/${session_name}.log"
    worker_script="${LOG_ROOT}/${session_name}.sh"
    cat > "${worker_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT}"
export DGIL_DATA_ROOT="${DATA_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
TASK_FILE="${TASK_FILE}"
TOTAL_WORKERS=${#GPUS[@]}
WORKER_ID=${idx}
GPU=${gpu}
LINE_NO=0
while IFS=\$'\t' read -r dataset backbone; do
    if [[ -z "\${dataset}" || -z "\${backbone}" ]]; then
        LINE_NO=\$((LINE_NO + 1))
        continue
    fi
    if (( LINE_NO % TOTAL_WORKERS != WORKER_ID )); then
        LINE_NO=\$((LINE_NO + 1))
        continue
    fi
    echo "[task-start] worker=\${WORKER_ID} gpu=\${GPU} line=\${LINE_NO} dataset=\${dataset} backbone=\${backbone} time=\$(date)"
    conda run -n "${CONDA_ENV}" python "scripts/extract_layerwise_features.py" \
        --dataset "\${dataset}" \
        --backbone "\${backbone}" \
        --normalization "${NORMALIZATION}" \
        --device "\${GPU}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}"
    conda run -n "${CONDA_ENV}" python "scripts/eval_layerwise_feature_probes.py" \
        --dataset "\${dataset}" \
        --backbone "\${backbone}" \
        --normalization "${NORMALIZATION}" \
        --feature-pools cls mean \
        --targets class domain \
        --classifiers linear ncm wncm \
        --linear-method "${LINEAR_METHOD}" \
        --preprocess "${PREPROCESS}" \
        --overwrite
    echo "[task-done] worker=\${WORKER_ID} gpu=\${GPU} line=\${LINE_NO} dataset=\${dataset} backbone=\${backbone} time=\$(date)"
    LINE_NO=\$((LINE_NO + 1))
done < "\${TASK_FILE}"
echo "[worker-done] worker=\${WORKER_ID} gpu=\${GPU} time=\$(date)"
EOF
    chmod +x "${worker_script}"
    screen -dmS "${session_name}" bash -lc "'${worker_script}' > '${worker_log}' 2>&1"
    echo "Started screen ${session_name}, script ${worker_script}, log ${worker_log}"
done

echo "All layer-probe screen workers started. Monitor with: screen -ls and logs in ${LOG_ROOT}"
