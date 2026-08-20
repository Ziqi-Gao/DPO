#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_protocol_launch_environment

for ((index=0; index<32; index++)); do
  IFS=$'\t' read -r cell stage_label stage cohort < <(
    "${PYTHON_BIN}" -m posttrain_circuits.cli.resolve_pilot_circuit_scope \
      pilot="${PILOT_CONFIG:?}" model="${MODEL_CONFIG:?}" teacher="${TEACHER_CONFIG:?}" \
      --matrix final --index "${index}"
  )
  : "${stage:?resolved probe stage is empty}"
  initial="${PILOT_RUN_DIR:?}/circuits/initial/${stage_label}/${cohort}"
  final="${PILOT_RUN_DIR}/circuits/final/${cell}/${stage_label}/${cohort}"
  output="${PILOT_RUN_DIR}/dynamics/${cell}/${stage_label}/${cohort}.json"
  "${PYTHON_BIN}" -m posttrain_circuits.cli.analyze_circuit_dynamics \
    pilot="${PILOT_CONFIG:?}" model="${MODEL_CONFIG:?}" teacher="${TEACHER_CONFIG:?}" \
    --circuits "${initial}/circuit.json" "${final}/circuit.json" \
    --evaluations "${initial}/exact_patching.json" "${final}/exact_patching.json" \
    --transfers "${final}/exact_patching.json" --activation-threshold 0.0 --output "${output}"
done
