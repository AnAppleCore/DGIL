#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yanhongwei/DGIL/DGIL"
LOG_DIR="${ROOT}/logs/screen"
mkdir -p "${LOG_DIR}"

screen_exists() {
  local name="$1"
  screen -ls | grep -q "[.]${name}[[:space:]]"
}

launch_job() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local log_file="$4"

  if screen_exists "${name}"; then
    echo "[$(date '+%F %T')] ${name} already exists; skip"
    return 0
  fi

  echo "[$(date '+%F %T')] launch ${name} on GPU ${gpu}: ${config}"
  screen -S "${name}" -dm bash -lc "cd ${ROOT} && CUDA_VISIBLE_DEVICES=${gpu} python main.py --config ${config} 2>&1 | tee ${log_file}"
}

wait_jobs() {
  local names=("$@")
  local remaining

  while true; do
    remaining=0
    for name in "${names[@]}"; do
      if screen_exists "${name}"; then
        remaining=$((remaining + 1))
      fi
    done

    if [[ "${remaining}" -eq 0 ]]; then
      break
    fi

    echo "[$(date '+%F %T')] waiting for ${remaining} jobs: ${names[*]}"
    sleep 120
  done
}

run_batch() {
  local batch_name="$1"
  shift
  local jobs=("$@")
  local names=()

  echo "[$(date '+%F %T')] start batch ${batch_name}"
  for spec in "${jobs[@]}"; do
    IFS='|' read -r name gpu config log_file <<< "${spec}"
    launch_job "${name}" "${gpu}" "${config}" "${log_file}"
    names+=("${name}")
  done

  wait_jobs "${names[@]}"
  echo "[$(date '+%F %T')] finished batch ${batch_name}"
}

batch_a=(
  "digitsdg_simon_msl_base|4|./configs/DGIL/digitsdg/simon_msl_dgil.json|${LOG_DIR}/digitsdg_simon_msl_base.log"
  "digitsdg_simon_msl_mem5|5|./configs/DGIL/digitsdg/simon_msl_dgil_mem5.json|${LOG_DIR}/digitsdg_simon_msl_mem5.log"
  "officehome_simon_msl_base|6|./configs/DGIL/officehome/simon_msl_dgil.json|${LOG_DIR}/officehome_simon_msl_base.log"
  "officehome_simon_msl_mem5|7|./configs/DGIL/officehome/simon_msl_dgil_mem5.json|${LOG_DIR}/officehome_simon_msl_mem5.log"
)

batch_b=(
  "digitsdg_cmoa|4|./configs/DGIL/digitsdg/cmoa_dgil.json|${LOG_DIR}/digitsdg_cmoa.log"
  "digitsdg_codag|5|./configs/DGIL/digitsdg/codag_dgil.json|${LOG_DIR}/digitsdg_codag.log"
  "officehome_cmoa|6|./configs/DGIL/officehome/cmoa_dgil.json|${LOG_DIR}/officehome_cmoa.log"
  "officehome_codag|7|./configs/DGIL/officehome/codag_dgil.json|${LOG_DIR}/officehome_codag.log"
)

batch_c=(
  "core50_simon_msl_base|4|./configs/DGIL/core50/simon_msl_dgil.json|${LOG_DIR}/core50_simon_msl_base.log"
  "core50_simon_msl_mem5|5|./configs/DGIL/core50/simon_msl_dgil_mem5.json|${LOG_DIR}/core50_simon_msl_mem5.log"
  "core50_cmoa|6|./configs/DGIL/core50/cmoa_dgil.json|${LOG_DIR}/core50_cmoa.log"
  "core50_codag|7|./configs/DGIL/core50/codag_dgil.json|${LOG_DIR}/core50_codag.log"
)

run_batch "a_digitsdg_officehome_simon" "${batch_a[@]}"
run_batch "b_digitsdg_officehome_new_methods" "${batch_b[@]}"
run_batch "c_core50_all_methods" "${batch_c[@]}"

echo "[$(date '+%F %T')] all tail batches finished"
