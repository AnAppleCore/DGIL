#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

eval "$(conda shell.bash hook)"
conda activate DGIL

python scripts/extract_results.py
