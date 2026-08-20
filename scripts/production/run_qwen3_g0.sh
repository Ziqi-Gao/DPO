#!/usr/bin/env bash
set -euo pipefail

for name in MODEL_CONFIG TEACHER_CONFIG PRODUCTION_CONFIG G0_CONFIG PILOT_CONFIG \
  PROJECT_ROOT PYTHON_BIN ACCELERATE_BIN OUTPUT_ROOT HF_HOME GPU_PREFLIGHT_JSON; do
  [[ -n "${!name:-}" ]] || { echo "Set ${name} explicitly" >&2; exit 2; }
done
[[ "${HF_HUB_OFFLINE:?Set HF_HUB_OFFLINE=1}" == 1 ]]
cd "${PROJECT_ROOT}"
run_id=${G0_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
export G0_RUN_ID="${run_id}"
export G0_RUN_DIR=${G0_RUN_DIR:-${OUTPUT_ROOT}/g0/${run_id}}
mkdir -p "${G0_RUN_DIR}/logs"
export SLURM_G0_JOB_NAME=opd-qwen3-g0
"${PYTHON_BIN}" -m posttrain_circuits.cli.record_qwen3_launch \
  --phase g0 --output "${G0_RUN_DIR}/launch_manifest.json"
exec scripts/production/run_g0.sh
