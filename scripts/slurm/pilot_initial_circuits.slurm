#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-circuit-i
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-1%4
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_protocol_launch_environment
model_config=${MODEL_CONFIG:?Set MODEL_CONFIG}
teacher_config=${TEACHER_CONFIG:?Set TEACHER_CONFIG}
pilot_config=${PILOT_CONFIG:?Set PILOT_CONFIG}

cohorts=(base_capable challenge)
cohort=${cohorts[$SLURM_ARRAY_TASK_ID]}
root="${PILOT_RUN_DIR:?}/circuits/initial/${cohort}"
"${PYTHON_BIN}" -m posttrain_circuits.cli.discover_circuit \
  pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
  output_root="${OUTPUT_ROOT}" \
  --checkpoint "${PILOT_INITIAL_CHECKPOINT:?}" --initial-checkpoint "${PILOT_INITIAL_CHECKPOINT}" \
  --probe-cohort-manifest "${PILOT_PROBE_MANIFEST:?}" --cohort "${cohort}" \
  --output "${root}/circuit.json" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.evaluate_circuit \
  pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
  output_root="${OUTPUT_ROOT}" \
  --circuit-artifact "${root}/circuit.json" --checkpoint "${PILOT_INITIAL_CHECKPOINT}" \
  --initial-checkpoint "${PILOT_INITIAL_CHECKPOINT}" \
  --probe-cohort-manifest "${PILOT_PROBE_MANIFEST}" --cohort "${cohort}" \
  --output "${root}/exact_patching.json" --confirm-production
