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
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

