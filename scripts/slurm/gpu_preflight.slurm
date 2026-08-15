#!/usr/bin/env bash
#SBATCH --job-name=opd-gpu-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
set -euo pipefail
source "$(dirname "$0")/common.sh"

output=${GPU_PREFLIGHT_OUTPUT:?Set GPU_PREFLIGHT_OUTPUT}
mkdir -p "$(dirname "${output}")"
torchrun_bin=$(dirname "${PYTHON_BIN}")/torchrun
"${torchrun_bin}" --standalone --nproc_per_node=4 \
  -m posttrain_circuits.cli.gpu_preflight \
  model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --output "${output}"
