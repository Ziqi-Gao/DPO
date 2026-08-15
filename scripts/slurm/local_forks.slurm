#!/usr/bin/env bash
#SBATCH --job-name=ptc-fork
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
set -euo pipefail
source "$(dirname "$0")/common.sh"
bundle="${OUTPUT_ROOT}/local_fork/${RUN_ID:?Set RUN_ID}/bundle.pt"
"${PYTHON_BIN}" -m posttrain_circuits.cli.create_fork_bundle \
  production=qwen_primary experiment=local_fork model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --checkpoint "${CHECKPOINT:?Set CHECKPOINT}" \
  --prompts "${FROZEN_PROMPTS:?Set FROZEN_PROMPTS}" \
  --trajectory-store "${FROZEN_TRAJECTORY_STORE:?Set FROZEN_TRAJECTORY_STORE}" \
  --probe-set "${FROZEN_PROBE_SET:?Set FROZEN_PROBE_SET}" \
  --output "${bundle}" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.run_local_fork \
  --bundle "${bundle}" --output "${OUTPUT_ROOT}/local_fork/${RUN_ID}/results.json"
