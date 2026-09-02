#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${PYTHON:-"/scr/del6500/OPD/envs/opd/bin/python"}
output_root=${SMOKE_OUTPUT_ROOT:-"/data/del6500/OPD/outputs/smoke/factorial"}
dataset_root="${output_root}/dataset"

cd "${project_root}"

split_overrides=(
  task=proofgraph_small
  task.split_sizes.train=20
  task.split_sizes.validation=20
  task.split_sizes.iid_test=20
  task.split_sizes.ood_depth_test=20
  task.split_sizes.ood_structure_test=20
  task.split_sizes.circuit_discovery=20
  task.split_sizes.circuit_validation=20
)
"${python_bin}" -m posttrain_circuits.cli.build_splits \
  "${split_overrides[@]}" --output "${dataset_root}"

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
    task=proofgraph_small \
    task.dataset_family_path="${dataset_root}" \
    task.num_examples=20 \
    trainer.max_steps=2 \
    --output "${output_root}/${cell}"
done

echo "factorial smoke complete: ${output_root}"
