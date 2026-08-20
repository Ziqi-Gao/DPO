#!/usr/bin/env bash
set -euo pipefail

for name in MODEL_CONFIG TEACHER_CONFIG PRODUCTION_CONFIG G0_CONFIG PILOT_CONFIG \
  PROJECT_ROOT PYTHON_BIN ACCELERATE_BIN OUTPUT_ROOT HF_HOME G0_JSON; do
  [[ -n "${!name:-}" ]] || { echo "Set ${name} explicitly" >&2; exit 2; }
done
[[ "${HF_HUB_OFFLINE:?Set HF_HUB_OFFLINE=1}" == 1 ]]
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
run_id=${PILOT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
export PILOT_RUN_ID="${run_id}"
export PILOT_RUN_DIR=${PILOT_RUN_DIR:-${OUTPUT_ROOT}/pilot/${run_id}}
mkdir -p "${PILOT_RUN_DIR}/logs"
"${PYTHON_BIN}" -m posttrain_circuits.cli.record_qwen3_launch \
  --phase seed42_pilot --output "${PILOT_RUN_DIR}/launch_manifest.json"
exec scripts/production/submit_pilot.sh
