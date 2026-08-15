#!/usr/bin/env bash
#SBATCH --job-name=ptc-gemma-mini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-5
set -euo pipefail
source "$(dirname "$0")/common.sh"

cells=(canonical_sft offline_soft online_soft_opd online_hard online_verified_replay canonical_grpo)
cell=${cells[$SLURM_ARRAY_TASK_ID]}
seed=${SEED:-42}
output="${OUTPUT_ROOT}/gemma-mini/${cell}/seed-${seed}"
common=(model=gemma2_2b teacher=gemma2_teacher_9b task=proofgraph_main seed="${seed}")

case "${cell}" in
  canonical_sft)
    "${ACCELERATE_BIN:-accelerate}" launch --config_file configs/accelerate/fsdp.yaml \
      -m posttrain_circuits.cli.train experiment=canonical_sft "${common[@]}" \
      state_source.store_path="${GEMMA_TEACHER_DEMO_STORE:?Set GEMMA_TEACHER_DEMO_STORE}" \
      trainer.backend=accelerate --output "${output}" --confirm-production
    ;;
  canonical_grpo)
    "${PYTHON_BIN}" -m posttrain_circuits.cli.run_grpo experiment=canonical_grpo \
      "${common[@]}" --output "${output}" --confirm-production
    ;;
  *)
    extra=()
    if [[ "${cell}" == offline_* ]]; then
      extra+=(state_source.store_path="${GEMMA_FIXED_BANK_PATH:?Set GEMMA_FIXED_BANK_PATH}")
    fi
    "${ACCELERATE_BIN:-accelerate}" launch --config_file configs/accelerate/fsdp.yaml \
      -m posttrain_circuits.cli.train experiment="${cell}" "${common[@]}" "${extra[@]}" \
      anti_shortcut.report_path="${GEMMA_ANTI_SHORTCUT_REPORT:-${OUTPUT_ROOT}/readiness/gemma-anti-shortcut.json}" \
      production_safety.probe_cohort_manifest="${GEMMA_PROBE_COHORT_MANIFEST:-${OUTPUT_ROOT}/probes/gemma/manifest.json}" \
      trainer.backend=accelerate --output "${output}" --confirm-production
    ;;
esac
