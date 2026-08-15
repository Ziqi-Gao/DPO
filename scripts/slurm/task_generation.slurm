#!/usr/bin/env bash
#SBATCH --job-name=ptc-task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
set -euo pipefail
source "$(dirname "$0")/common.sh"
"${PYTHON_BIN}" -m posttrain_circuits.cli.generate_task \
  task=proofgraph_main split="${SPLIT:-train}" task.seed="${SEED:-42}" \
  --output "${OUTPUT_ROOT}/datasets/${SPLIT:-train}"

