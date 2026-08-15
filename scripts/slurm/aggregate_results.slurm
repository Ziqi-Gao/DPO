#!/usr/bin/env bash
#SBATCH --job-name=ptc-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail
source "$(dirname "$0")/common.sh"
mapfile -t run_dirs < <(find "${OUTPUT_ROOT}/runs" -mindepth 2 -maxdepth 2 -type d | sort)
"${PYTHON_BIN}" -m posttrain_circuits.cli.aggregate_results "${run_dirs[@]}" \
  --output "${OUTPUT_ROOT}/aggregate/results.json"

