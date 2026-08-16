#!/usr/bin/env bash
#SBATCH --job-name=ptc-circuit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
set -euo pipefail
source "$(dirname "$0")/common.sh"
"${PYTHON_BIN}" -m posttrain_circuits.cli.discover_circuit \
  model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main circuit=eap_ig \
  --checkpoint "${CHECKPOINT:?Set CHECKPOINT}" \
  --initial-checkpoint "${INITIAL_CHECKPOINT:?Set INITIAL_CHECKPOINT}" \
  --probe-cohort-manifest "${PROBE_COHORT_MANIFEST:?Set PROBE_COHORT_MANIFEST}" \
  --cohort "${COHORT:?Set COHORT}" \
  --output "${OUTPUT_ROOT}/circuits/${RUN_ID:?Set RUN_ID}/circuit.json" --confirm-production
