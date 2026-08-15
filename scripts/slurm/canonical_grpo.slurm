#!/usr/bin/env bash
#SBATCH --job-name=ptc-grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
set -euo pipefail
source "$(dirname "$0")/common.sh"
"${PYTHON_BIN}" -m posttrain_circuits.cli.run_grpo \
  experiment="${EXPERIMENT:-canonical_grpo}" model=qwen25_1p5b seed="${SEED:-42}" \
  production=qwen_primary teacher=qwen25_teacher_7b task=proofgraph_main trainer.backend=accelerate \
  --output "${OUTPUT_ROOT}/runs/${EXPERIMENT:-canonical_grpo}/seed-${SEED:-42}" \
  --confirm-production
