#!/usr/bin/env bash
set -euo pipefail

stage=${1:-}
if [[ -z "${stage}" ]]; then
  echo "usage: $0 <task_generation|rollout_bank|teacher_scoring|factorial_training|gemma_mini_replication|canonical_grpo|circuit_discovery|exact_patching|local_forks|aggregate_results> [--dry-run]"
  exit 2
fi
shift

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
script="${project_root}/scripts/slurm/${stage}.slurm"
if [[ ! -f "${script}" ]]; then
  echo "unknown stage: ${stage}" >&2
  exit 2
fi
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"
partition=${SLURM_PARTITION:?Set SLURM_PARTITION}
if [[ "${stage}" =~ ^(rollout_bank|teacher_scoring|factorial_training|gemma_mini_replication|canonical_grpo|circuit_discovery|exact_patching|local_forks)$ ]]; then
  partition=${SLURM_GPU_PARTITION:-${partition}}
fi
command=(sbatch --account="${SLURM_ACCOUNT}" --partition="${partition}" "${script}")
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  if [[ "${stage}" == "factorial_training" ]]; then
    "${PYTHON_BIN:-python}" -m posttrain_circuits.cli.assert_production_config \
      production=qwen_primary experiment=offline_hard model=qwen25_1p5b \
      teacher=qwen25_teacher_7b task=proofgraph_main trainer.backend=accelerate
    "${PYTHON_BIN:-python}" -m posttrain_circuits.cli.check_factorial_prerequisites \
      production=qwen_primary model="${MODEL_CONFIG:-qwen25_1p5b}" \
      teacher="${TEACHER_CONFIG:-qwen25_teacher_7b}" task=proofgraph_main \
      anti_shortcut.report_path="${ANTI_SHORTCUT_REPORT:-${OUTPUT_ROOT:-outputs}/readiness/anti_shortcut.json}" \
      production_safety.probe_cohort_manifest="${PROBE_COHORT_MANIFEST:-${OUTPUT_ROOT:-outputs}/probes/proofgraph/manifest.json}" \
      production_safety.readiness_report="${READINESS_REPORT:-${OUTPUT_ROOT:-outputs}/readiness/readiness.json}"
  elif [[ "${stage}" == "gemma_mini_replication" ]]; then
    "${PYTHON_BIN:-python}" -m posttrain_circuits.cli.check_factorial_prerequisites \
      model=gemma2_2b teacher=gemma2_teacher_9b task=proofgraph_main \
      anti_shortcut.report_path="${GEMMA_ANTI_SHORTCUT_REPORT:-${OUTPUT_ROOT:-outputs}/readiness/gemma-anti-shortcut.json}" \
      production_safety.probe_cohort_manifest="${GEMMA_PROBE_COHORT_MANIFEST:-${OUTPUT_ROOT:-outputs}/probes/gemma/manifest.json}"
  fi
  "${command[@]}"
fi
