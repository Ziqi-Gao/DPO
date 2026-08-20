#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-circuit-f
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-31%4
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_protocol_launch_environment
model_config=${MODEL_CONFIG:?Set MODEL_CONFIG}
teacher_config=${TEACHER_CONFIG:?Set TEACHER_CONFIG}
pilot_config=${PILOT_CONFIG:?Set PILOT_CONFIG}

IFS=$'\t' read -r cell stage_label stage cohort < <(
  "${PYTHON_BIN}" -m posttrain_circuits.cli.resolve_pilot_circuit_scope \
    pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" \
    --matrix final --index "${SLURM_ARRAY_TASK_ID}"
)
manifest="${PILOT_RUN_DIR:?}/runs/${cell}/seed-42/manifest.json"
checkpoint=$("${PYTHON_BIN}" -m posttrain_circuits.cli.resolve_final_checkpoint \
  --manifest "${manifest}")
root="${PILOT_RUN_DIR}/circuits/final/${cell}/${stage_label}/${cohort}"
source_circuit="${PILOT_RUN_DIR}/circuits/initial/${stage_label}/${cohort}/circuit.json"
"${PYTHON_BIN}" -m posttrain_circuits.cli.discover_circuit \
  pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
  output_root="${OUTPUT_ROOT}" \
  --stage "${stage}" --checkpoint "${checkpoint}" \
  --initial-checkpoint "${PILOT_INITIAL_CHECKPOINT:?}" \
  --probe-cohort-manifest "${PILOT_PROBE_MANIFEST:?}" --cohort "${cohort}" \
  --output "${root}/circuit.json" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.evaluate_circuit \
  pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
  output_root="${OUTPUT_ROOT}" \
  --circuit-artifact "${root}/circuit.json" --checkpoint "${checkpoint}" \
  --initial-checkpoint "${PILOT_INITIAL_CHECKPOINT}" \
  --probe-cohort-manifest "${PILOT_PROBE_MANIFEST}" --cohort "${cohort}" \
  --transfer-source-circuit "${source_circuit}" \
  --output "${root}/exact_patching.json" --confirm-production
