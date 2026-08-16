#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-fork
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_dir=${PILOT_RUN_DIR:?Set PILOT_RUN_DIR}
checkpoint="${run_dir}/runs/canonical_sft/seed-42/checkpoints/step-00000120.pt"
inputs="${run_dir}/local_fork/inputs"
"${PYTHON_BIN}" -m posttrain_circuits.cli.build_local_fork_inputs \
  pilot=qwen_core model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --trajectory-store "${PILOT_COMMON_BANK:?Set PILOT_COMMON_BANK}" --output "${inputs}"
"${PYTHON_BIN}" -m posttrain_circuits.cli.create_fork_bundle \
  pilot=qwen_core experiment=local_fork model=qwen25_1p5b teacher=qwen25_teacher_7b \
  task=proofgraph_main task.validation_split_path="${PILOT_VALIDATION_SPLIT:?}" \
  --checkpoint "${checkpoint}" --prompts "${inputs}/prompts.json" \
  --trajectory-store "${PILOT_COMMON_BANK}" --probe-set "${inputs}/probe_set.json" \
  --output "${run_dir}/local_fork/bundle.pt" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.run_local_fork \
  --bundle "${run_dir}/local_fork/bundle.pt" \
  --output "${run_dir}/local_fork/results.json" --horizons 1 5 20
