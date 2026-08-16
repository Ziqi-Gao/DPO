#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${PYTHON:-"${project_root}/.venv/bin/python"}
mkdir -p "${project_root}/outputs"
smoke_root=$(mktemp -d "${project_root}/outputs/smoke-repaired-g0.XXXXXX")

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
  "${split_overrides[@]}" --output "${smoke_root}/dataset"
"${python_bin}" -m posttrain_circuits.cli.audit_label_leakage \
  --split-root "${smoke_root}/dataset/validation" \
  --split validation --output "${smoke_root}/label_leakage.json"

"${python_bin}" -m posttrain_circuits.cli.build_rollout_bank \
  model=tiny_qwen state_source.num_generations_per_prompt=4 \
  --output "${smoke_root}/common_bank"

cells=(
  offline_hard
  online_hard
  offline_soft
  online_soft_opd
  offline_verified_replay
  online_verified_replay
)
for cell in "${cells[@]}"; do
  cell_args=(experiment="${cell}" model=tiny_qwen trainer.max_steps=2)
  if [[ "${cell}" == offline_* ]]; then
    cell_args+=(state_source.store_path="${smoke_root}/common_bank")
  fi
  "${python_bin}" -m posttrain_circuits.cli.train \
    "${cell_args[@]}" --output "${smoke_root}/factorial/${cell}"
done

"${python_bin}" -m posttrain_circuits.cli.build_teacher_demos \
  experiment=canonical_sft model=tiny_qwen task.num_examples=4 \
  state_source.num_candidates=2 --output "${smoke_root}/sft/demos"
"${python_bin}" -m posttrain_circuits.cli.train \
  experiment=canonical_sft model=tiny_qwen trainer.max_steps=2 \
  state_source.store_path="${smoke_root}/sft/demos" \
  --output "${smoke_root}/sft/run"

"${python_bin}" -m posttrain_circuits.cli.run_grpo \
  experiment=grpo_random_reward model=tiny_qwen task.num_examples=4 \
  trainer.max_steps=1 trainer.batch_size=4 supervision.num_generations=2 \
  supervision.gradient_accumulation_steps=1 supervision.max_completion_length=8 \
  --output "${smoke_root}/grpo"

"${python_bin}" -m posttrain_circuits.cli.create_fork_bundle \
  experiment=local_fork model=tiny_qwen --seed 42 \
  --output "${smoke_root}/local_fork/bundle.pt"
"${python_bin}" -m posttrain_circuits.cli.run_local_fork \
  --bundle "${smoke_root}/local_fork/bundle.pt" \
  --output "${smoke_root}/local_fork/results.json" --horizons 1

stages=(first_rule_selection intermediate_conclusion final_answer)
for stage in "${stages[@]}"; do
  stage_root="${smoke_root}/circuits/${stage}"
  "${python_bin}" -m posttrain_circuits.cli.discover_circuit \
    task=proofgraph_small circuit=eap_ig model=tiny_qwen circuit.smoke_steps=2 \
    --stage "${stage}" --output "${stage_root}/circuit.json"
  "${python_bin}" -m posttrain_circuits.cli.evaluate_circuit \
    task=proofgraph_small circuit.random_mask_repeats=2 \
    circuit.prompt_bootstrap_samples=20 \
    --circuit-artifact "${stage_root}/circuit.json" \
    --output "${stage_root}/exact_patching.json"
done

"${python_bin}" -m posttrain_circuits.cli.verify_repaired_g0_smoke \
  --root "${smoke_root}" --output "${smoke_root}/verification.json"

echo "CPU-only repaired G0 smoke complete (not a real G0): ${smoke_root}"
