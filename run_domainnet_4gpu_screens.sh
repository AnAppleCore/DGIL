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

launch_job "domainnet_simon_msl_base" 0 "./configs/DGIL/domainnet/simon_msl_dgil.json" "${LOG_DIR}/domainnet_simon_msl_base.log"
launch_job "domainnet_simon_msl_mem5" 1 "./configs/DGIL/domainnet/simon_msl_dgil_mem5.json" "${LOG_DIR}/domainnet_simon_msl_mem5.log"
launch_job "domainnet_cmoa" 2 "./configs/DGIL/domainnet/cmoa_dgil.json" "${LOG_DIR}/domainnet_cmoa.log"
launch_job "domainnet_codag" 3 "./configs/DGIL/domainnet/codag_dgil.json" "${LOG_DIR}/domainnet_codag.log"

screen -ls
