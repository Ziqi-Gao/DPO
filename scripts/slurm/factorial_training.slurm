#!/usr/bin/env bash
#SBATCH --job-name=ptc-factorial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-17
set -euo pipefail
source "$(dirname "$0")/common.sh"
cells=(offline_hard online_hard offline_soft online_soft_opd offline_verified_replay online_verified_replay)
IFS=',' read -r -a seeds <<< "${EXPERIMENT_SEEDS:-42,43,44}"
cell_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))
cell=${cells[$cell_index]}
seed=${seeds[$seed_index]}
"${ACCELERATE_BIN:-accelerate}" launch \
  --config_file configs/accelerate/fsdp.yaml \
  -m posttrain_circuits.cli.train \
  experiment="${cell}" \
  production=qwen_primary \
  model="${MODEL_CONFIG:-qwen25_1p5b}" \
  teacher="${TEACHER_CONFIG:-qwen25_teacher_7b}" \
  task=proofgraph_main \
  seed="${seed}" \
  anti_shortcut.report_path="${ANTI_SHORTCUT_REPORT:-${OUTPUT_ROOT}/readiness/anti_shortcut.json}" \
  production_safety.probe_cohort_manifest="${PROBE_COHORT_MANIFEST:-${OUTPUT_ROOT}/probes/proofgraph/manifest.json}" \
  production_safety.readiness_report="${READINESS_REPORT:-${OUTPUT_ROOT}/readiness/readiness.json}" \
  production_safety.initial_checkpoint_path="${INITIAL_CHECKPOINT:?Set INITIAL_CHECKPOINT}" \
  production_safety.initial_checkpoint_hash="${INITIAL_CHECKPOINT_SHA256:?Set INITIAL_CHECKPOINT_SHA256}" \
  trainer.backend=accelerate \
  --output "${OUTPUT_ROOT}/runs/${cell}/seed-${seed}" \
  --confirm-production
