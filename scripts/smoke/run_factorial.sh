#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${PYTHON:-"${project_root}/.venv/bin/python"}
output_root=${SMOKE_OUTPUT_ROOT:-"${project_root}/outputs/smoke-factorial"}

cells=(
  offline_hard
  online_hard
  offline_soft
  online_soft_opd
  offline_verified_replay
  online_verified_replay
)

for cell in "${cells[@]}"; do
  "${python_bin}" -m posttrain_circuits.cli.train \
    experiment="${cell}" \
    model=tiny_qwen \
    trainer.max_steps=2 \
    --output "${output_root}/${cell}"
done

echo "factorial smoke complete: ${output_root}"

