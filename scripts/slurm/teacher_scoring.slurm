#!/usr/bin/env bash
#SBATCH --job-name=ptc-teacher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
set -euo pipefail
source "$(dirname "$0")/common.sh"
"${PYTHON_BIN}" -m posttrain_circuits.cli.score_teacher \
  model=qwen25_teacher_7b --bank "${OUTPUT_ROOT}/rollout_banks/common_mu" \
  --output "${OUTPUT_ROOT}/rollout_banks/common_mu_scored" --confirm-production

