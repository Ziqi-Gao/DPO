#!/usr/bin/env bash
#SBATCH --job-name=ptc-bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
set -euo pipefail
source "$(dirname "$0")/common.sh"
"${PYTHON_BIN}" -m posttrain_circuits.cli.build_rollout_bank \
  model=qwen25_1p5b task=proofgraph_main state_source=fixed_bank \
  --output "${OUTPUT_ROOT}/rollout_banks/common_mu" --confirm-production

