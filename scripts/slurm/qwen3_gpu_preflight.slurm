#!/usr/bin/env bash
#SBATCH --job-name=opd-qwen3-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=00:30:00
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_protocol_launch_environment

: "${MODEL_CONFIG:?Set MODEL_CONFIG}"
: "${TEACHER_CONFIG:?Set TEACHER_CONFIG}"
: "${PRODUCTION_CONFIG:?Set PRODUCTION_CONFIG}"
: "${G0_CONFIG:?Set G0_CONFIG}"
: "${PILOT_CONFIG:?Set PILOT_CONFIG}"
: "${PROJECT_ROOT:?Set PROJECT_ROOT}"
: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${ACCELERATE_BIN:?Set ACCELERATE_BIN}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${HF_HOME:?Set project-local HF_HOME}"
[[ "${HF_HUB_OFFLINE:?Set HF_HUB_OFFLINE=1}" == 1 ]]

output=${GPU_PREFLIGHT_OUTPUT:?Set GPU_PREFLIGHT_OUTPUT}
mkdir -p "$(dirname "${output}")"
torchrun_bin=$(dirname "${PYTHON_BIN}")/torchrun
"${torchrun_bin}" --standalone --nproc_per_node=4 \
  -m posttrain_circuits.cli.gpu_preflight \
  production="${PRODUCTION_CONFIG}" model="${MODEL_CONFIG}" teacher="${TEACHER_CONFIG}" \
  g0="${G0_CONFIG}" pilot="${PILOT_CONFIG}" task=proofgraph_main \
  task.validation_split_path="${OUTPUT_ROOT}/datasets/proofgraph-validation" \
  output_root="${OUTPUT_ROOT}" --output "${output}"
