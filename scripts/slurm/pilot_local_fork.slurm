#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-fork
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_protocol_launch_environment
model_config=${MODEL_CONFIG:?Set MODEL_CONFIG}
teacher_config=${TEACHER_CONFIG:?Set TEACHER_CONFIG}
pilot_config=${PILOT_CONFIG:?Set PILOT_CONFIG}

run_dir=${PILOT_RUN_DIR:?Set PILOT_RUN_DIR}
checkpoint=$("${PYTHON_BIN}" -m posttrain_circuits.cli.resolve_final_checkpoint \
  --manifest "${run_dir}/runs/canonical_sft/seed-42/manifest.json")
inputs="${run_dir}/local_fork/inputs"
"${PYTHON_BIN}" -m posttrain_circuits.cli.build_local_fork_inputs \
  pilot="${pilot_config}" model="${model_config}" teacher="${teacher_config}" task=proofgraph_main \
  output_root="${OUTPUT_ROOT}" \
  --trajectory-store "${PILOT_COMMON_BANK:?Set PILOT_COMMON_BANK}" --output "${inputs}"
"${PYTHON_BIN}" -m posttrain_circuits.cli.create_fork_bundle \
  pilot="${pilot_config}" experiment=local_fork model="${model_config}" teacher="${teacher_config}" \
  output_root="${OUTPUT_ROOT}" \
  task=proofgraph_main task.validation_split_path="${PILOT_VALIDATION_SPLIT:?}" \
  --checkpoint "${checkpoint}" --prompts "${inputs}/prompts.json" \
  --trajectory-store "${PILOT_COMMON_BANK}" --probe-set "${inputs}/probe_set.json" \
  --output "${run_dir}/local_fork/bundle.pt" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.run_local_fork \
  --bundle "${run_dir}/local_fork/bundle.pt" \
  --output "${run_dir}/local_fork/results.json" --horizons 1 5 20
