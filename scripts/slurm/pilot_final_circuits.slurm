#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-circuit-f
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-11%4
set -euo pipefail
source "$(dirname "$0")/common.sh"

cells=(offline_soft online_soft_opd online_hard online_verified_replay canonical_grpo canonical_sft)
cohorts=(base_capable challenge)
cell_index=$((SLURM_ARRAY_TASK_ID / 2))
cohort_index=$((SLURM_ARRAY_TASK_ID % 2))
cell=${cells[$cell_index]}
cohort=${cohorts[$cohort_index]}
checkpoint="${PILOT_RUN_DIR:?}/runs/${cell}/seed-42/checkpoints/step-00000120.pt"
root="${PILOT_RUN_DIR}/circuits/final/${cell}/${cohort}"
source_circuit="${PILOT_RUN_DIR}/circuits/initial/${cohort}/circuit.json"
"${PYTHON_BIN}" -m posttrain_circuits.cli.discover_circuit \
  pilot=qwen_core model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --checkpoint "${checkpoint}" --initial-checkpoint "${PILOT_INITIAL_CHECKPOINT:?}" \
  --probe-cohort-manifest "${PILOT_PROBE_MANIFEST:?}" --cohort "${cohort}" \
  --output "${root}/circuit.json" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.evaluate_circuit \
  pilot=qwen_core model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --circuit-artifact "${root}/circuit.json" --checkpoint "${checkpoint}" \
  --initial-checkpoint "${PILOT_INITIAL_CHECKPOINT}" \
  --probe-cohort-manifest "${PILOT_PROBE_MANIFEST}" --cohort "${cohort}" \
  --transfer-source-circuit "${source_circuit}" \
  --output "${root}/exact_patching.json" --confirm-production
