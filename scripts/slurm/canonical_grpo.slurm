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
"${ACCELERATE_BIN:-accelerate}" launch --config_file configs/accelerate/fsdp.yaml \
  -m posttrain_circuits.cli.run_grpo \
  experiment="${EXPERIMENT:-canonical_grpo}" model=qwen25_1p5b seed="${SEED:-42}" \
  production=qwen_primary teacher=qwen25_teacher_7b task=proofgraph_main trainer.backend=accelerate \
  production_safety.initial_checkpoint_path="${INITIAL_CHECKPOINT:?Set INITIAL_CHECKPOINT}" \
  production_safety.initial_checkpoint_hash="${INITIAL_CHECKPOINT_SHA256:?Set INITIAL_CHECKPOINT_SHA256}" \
  --output "${OUTPUT_ROOT}/runs/${EXPERIMENT:-canonical_grpo}/seed-${SEED:-42}" \
  --confirm-production
