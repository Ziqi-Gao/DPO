#!/usr/bin/env bash
#SBATCH --job-name=opd-pilot-dynamics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_protocol_launch_environment

cells=(offline_soft online_soft_opd online_hard online_verified_replay canonical_grpo canonical_sft)
cohorts=(base_capable challenge)
for cell in "${cells[@]}"; do
  for cohort in "${cohorts[@]}"; do
    initial="${PILOT_RUN_DIR:?}/circuits/initial/${cohort}"
    final="${PILOT_RUN_DIR}/circuits/final/${cell}/${cohort}"
    output="${PILOT_RUN_DIR}/dynamics/${cell}/${cohort}.json"
    "${PYTHON_BIN}" -m posttrain_circuits.cli.analyze_circuit_dynamics \
      --circuits "${initial}/circuit.json" "${final}/circuit.json" \
      --evaluations "${initial}/exact_patching.json" "${final}/exact_patching.json" \
      --transfers "${final}/exact_patching.json" --activation-threshold 0.0 --output "${output}"
  done
done
