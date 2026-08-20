#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-circuit-i
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_protocol_launch_environment
model_config=${MODEL_CONFIG:?Set MODEL_CONFIG}
teacher_config=${TEACHER_CONFIG:?Set TEACHER_CONFIG}
pilot_config=${PILOT_CONFIG:?Set PILOT_CONFIG}

IFS=$'\t' read -r stage_label stage cohort < <(
  "${PYTHON_BIN}" -m posttrain_circuits.cli.resolve_pilot_circuit_scope \
    pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" \
    --matrix initial --index "${SLURM_ARRAY_TASK_ID}"
)
root="${PILOT_RUN_DIR:?}/circuits/initial/${stage_label}/${cohort}"
"${PYTHON_BIN}" -m posttrain_circuits.cli.discover_circuit \
  pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
  output_root="${OUTPUT_ROOT}" \
  --stage "${stage}" --checkpoint "${PILOT_INITIAL_CHECKPOINT:?}" \
  --initial-checkpoint "${PILOT_INITIAL_CHECKPOINT}" \
  --probe-cohort-manifest "${PILOT_PROBE_MANIFEST:?}" --cohort "${cohort}" \
  --output "${root}/circuit.json" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.evaluate_circuit \
  pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
  output_root="${OUTPUT_ROOT}" \
  --circuit-artifact "${root}/circuit.json" --checkpoint "${PILOT_INITIAL_CHECKPOINT}" \
  --initial-checkpoint "${PILOT_INITIAL_CHECKPOINT}" \
  --probe-cohort-manifest "${PILOT_PROBE_MANIFEST}" --cohort "${cohort}" \
  --output "${root}/exact_patching.json" --confirm-production
