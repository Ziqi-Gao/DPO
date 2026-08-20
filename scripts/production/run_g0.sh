#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${project_root}"
if [[ -d .opd-git && ! -f .git/HEAD ]]; then
  export GIT_DIR="${project_root}/.opd-git"
  export GIT_WORK_TREE="${project_root}"
fi
export PROJECT_ROOT=${PROJECT_ROOT:?Set PROJECT_ROOT explicitly}
export PYTHON_BIN=${PYTHON_BIN:?Set PYTHON_BIN explicitly}
export ACCELERATE_BIN=${ACCELERATE_BIN:?Set ACCELERATE_BIN explicitly}
export OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT explicitly}
export MODEL_CONFIG=${MODEL_CONFIG:?Set MODEL_CONFIG explicitly}
export TEACHER_CONFIG=${TEACHER_CONFIG:?Set TEACHER_CONFIG explicitly}
export PRODUCTION_CONFIG=${PRODUCTION_CONFIG:?Set PRODUCTION_CONFIG explicitly}
export G0_CONFIG=${G0_CONFIG:?Set G0_CONFIG explicitly}
export PILOT_CONFIG=${PILOT_CONFIG:?Set PILOT_CONFIG explicitly}
export HF_HOME=${HF_HOME:-$(dirname "${project_root}")/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
run_id=${G0_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir=${G0_RUN_DIR:-${OUTPUT_ROOT}/g0/${run_id}}
mkdir -p "${run_dir}/logs"
source scripts/production/slurm_supervision.sh
gpu_preflight_json=${GPU_PREFLIGHT_JSON:?Set GPU_PREFLIGHT_JSON to a passed, hash-valid GPU preflight}
export GPU_PREFLIGHT_JSON="${gpu_preflight_json}"

if ! "${PYTHON_BIN:-.venv/bin/python}" -m posttrain_circuits.cli.preflight_g0 \
  g0="${G0_CONFIG}" model="${MODEL_CONFIG}" teacher="${TEACHER_CONFIG}" task=proofgraph_main \
  output_root="${OUTPUT_ROOT}" \
  --gpu-preflight "${gpu_preflight_json}" --output "${run_dir}/preflight.json"; then
  cp "${run_dir}/preflight.json" "${run_dir}/g0.json"
  cp "${run_dir}/preflight.md" "${run_dir}/g0.md"
  exit 1
fi

partition=${SLURM_GPU_PARTITION:-${SLURM_PARTITION:?Set SLURM_PARTITION}}
walltime=${SLURM_G0_TIME:-12:00:00}
require_no_competing_opd_gpu_job "G0"
submission=$(sbatch --parsable --account="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}" \
  --partition="${partition}" --time="${walltime}" \
  --job-name="${SLURM_G0_JOB_NAME:-opd-g0-qwen}" --export=ALL,G0_RUN_DIR="${run_dir}" \
  --output="${run_dir}/logs/g0-%j.out" --error="${run_dir}/logs/g0-%j.err" \
  scripts/slurm/g0_qwen.slurm)
job_id=${submission%%;*}
printf '%s\n' "${job_id}" > "${run_dir}/job_id.txt"

wait_for_slurm_terminal "${job_id}" "${run_dir}/terminal-g0.txt"
"${PYTHON_BIN:-.venv/bin/python}" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("passed") is True else 1)' \
  "${run_dir}/g0.json"
printf '%s\n' "G0 PASS: ${run_dir}/g0.json"
