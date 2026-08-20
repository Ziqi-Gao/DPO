#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CLUSTER_ENV_FILE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CLUSTER_ENV_FILE}"
fi

: "${PROJECT_ROOT:?Set PROJECT_ROOT or CLUSTER_ENV_FILE}"
: "${PYTHON_BIN:?Set PYTHON_BIN or CLUSTER_ENV_FILE}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT or CLUSTER_ENV_FILE}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

require_protocol_launch_environment() {
  local name
  for name in MODEL_CONFIG TEACHER_CONFIG PRODUCTION_CONFIG G0_CONFIG PILOT_CONFIG \
    PROJECT_ROOT PYTHON_BIN ACCELERATE_BIN OUTPUT_ROOT; do
    [[ -n "${!name:-}" ]] || { echo "Set ${name} explicitly" >&2; return 2; }
  done
}
