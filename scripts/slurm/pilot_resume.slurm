#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
set -euo pipefail
source "$(dirname "$0")/common.sh"

checkpoint="${PILOT_RUN_DIR:?}/runs/canonical_sft/seed-42/checkpoints/step-00000120.pt"
for replica in a b; do
  "${ACCELERATE_BIN}" launch --config_file configs/accelerate/fsdp.yaml \
    -m posttrain_circuits.cli.train pilot=qwen_core experiment=canonical_sft \
    model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
    task.validation_split_path="${PILOT_VALIDATION_SPLIT:?}" \
    production_safety.initial_checkpoint_path="${PILOT_INITIAL_CHECKPOINT:?}" \
    production_safety.initial_checkpoint_hash="${PILOT_INITIAL_CHECKPOINT_SHA256:?}" \
    state_source.store_path="${PILOT_TEACHER_DEMOS:?}" trainer.max_steps=121 trainer.checkpoint_every=1 \
    --resume "${checkpoint}" --output "${PILOT_RUN_DIR}/resume-${replica}" --confirm-production
done
"${PYTHON_BIN}" -m posttrain_circuits.cli.compare_distributed_resume \
  --checkpoint-a "${PILOT_RUN_DIR}/resume-a/checkpoints/step-00000121.pt" \
  --checkpoint-b "${PILOT_RUN_DIR}/resume-b/checkpoints/step-00000121.pt" \
  --metrics-a "${PILOT_RUN_DIR}/resume-a/metrics.jsonl" \
  --metrics-b "${PILOT_RUN_DIR}/resume-b/metrics.jsonl" \
  --world-size 4 --output "${PILOT_RUN_DIR}/distributed_resume.json"
