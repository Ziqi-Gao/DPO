#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=02:00:00
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_protocol_launch_environment
model_config=${MODEL_CONFIG:?Set MODEL_CONFIG}
teacher_config=${TEACHER_CONFIG:?Set TEACHER_CONFIG}
pilot_config=${PILOT_CONFIG:?Set PILOT_CONFIG}

checkpoint="${PILOT_RUN_DIR:?}/runs/canonical_sft/seed-42/checkpoints/step-00000020.pt"
[[ -f "${checkpoint}" ]] || { echo "pilot resume source step 20 is absent" >&2; exit 1; }
for replica in a b; do
  "${ACCELERATE_BIN}" launch --config_file configs/accelerate/fsdp.yaml \
    -m posttrain_circuits.cli.train pilot="${pilot_config}" experiment=canonical_sft \
    model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
    output_root="${OUTPUT_ROOT}" \
    task.validation_split_path="${PILOT_VALIDATION_SPLIT:?}" \
    production_safety.initial_checkpoint_path="${PILOT_INITIAL_CHECKPOINT:?}" \
    production_safety.initial_checkpoint_hash="${PILOT_INITIAL_CHECKPOINT_SHA256:?}" \
    state_source.store_path="${PILOT_TEACHER_DEMOS:?}" trainer.max_steps=120 trainer.checkpoint_every=20 \
    --resume "${checkpoint}" --output "${PILOT_RUN_DIR}/resume-${replica}" --confirm-production
done
resume_a=$("${PYTHON_BIN}" -m posttrain_circuits.cli.resolve_final_checkpoint \
  --manifest "${PILOT_RUN_DIR}/resume-a/manifest.json")
resume_b=$("${PYTHON_BIN}" -m posttrain_circuits.cli.resolve_final_checkpoint \
  --manifest "${PILOT_RUN_DIR}/resume-b/manifest.json")
"${PYTHON_BIN}" -m posttrain_circuits.cli.compare_distributed_resume \
  pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
  --checkpoint-a "${resume_a}" --checkpoint-b "${resume_b}" \
  --metrics-a "${PILOT_RUN_DIR}/resume-a/metrics.jsonl" \
  --metrics-b "${PILOT_RUN_DIR}/resume-b/metrics.jsonl" \
  --world-size 4 --output "${PILOT_RUN_DIR}/distributed_resume.json"
