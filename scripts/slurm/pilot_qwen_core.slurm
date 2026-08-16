#!/usr/bin/env bash
#SBATCH --job-name=opd-qwen-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-7%1
set -euo pipefail
source "$(dirname "$0")/common.sh"

cells=(offline_hard online_hard offline_soft online_soft_opd offline_verified_replay online_verified_replay canonical_sft canonical_grpo)
cell=${cells[$SLURM_ARRAY_TASK_ID]}
seed=42
output="${PILOT_RUN_DIR:?Set PILOT_RUN_DIR}/runs/${cell}/seed-${seed}"
common=(pilot=qwen_core model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main seed=42
  task.validation_split_path="${PILOT_VALIDATION_SPLIT:?Set PILOT_VALIDATION_SPLIT}"
  anti_shortcut.report_path="${PILOT_ANTI_SHORTCUT:?Set PILOT_ANTI_SHORTCUT}"
  production_safety.probe_cohort_manifest="${PILOT_PROBE_MANIFEST:?Set PILOT_PROBE_MANIFEST}"
  production_safety.readiness_report="${PILOT_READINESS_REPORT:?Set PILOT_READINESS_REPORT}"
  production_safety.initial_checkpoint_path="${PILOT_INITIAL_CHECKPOINT:?Set PILOT_INITIAL_CHECKPOINT}"
  production_safety.initial_checkpoint_hash="${PILOT_INITIAL_CHECKPOINT_SHA256:?}")

if [[ "${cell}" == canonical_grpo ]]; then
  "${ACCELERATE_BIN:-accelerate}" launch --config_file configs/accelerate/fsdp.yaml \
    -m posttrain_circuits.cli.run_grpo experiment=canonical_grpo "${common[@]}" \
    trainer.batch_size=8 \
    --output "${output}" --confirm-production
elif [[ "${cell}" == canonical_sft ]]; then
  "${ACCELERATE_BIN:-accelerate}" launch --config_file configs/accelerate/fsdp.yaml \
    -m posttrain_circuits.cli.train experiment=canonical_sft "${common[@]}" \
    state_source.store_path="${PILOT_TEACHER_DEMOS:?Set PILOT_TEACHER_DEMOS}" \
    --output "${output}" --confirm-production
else
  extra=()
  if [[ "${cell}" == offline_* ]]; then
    extra+=(state_source.store_path="${PILOT_COMMON_BANK:?Set PILOT_COMMON_BANK}")
  fi
  "${ACCELERATE_BIN:-accelerate}" launch --config_file configs/accelerate/fsdp.yaml \
    -m posttrain_circuits.cli.train experiment="${cell}" "${common[@]}" "${extra[@]}" \
    --output "${output}" --confirm-production
fi
