#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${project_root}"
if [[ -d .opd-git && ! -f .git/HEAD ]]; then
  export GIT_DIR="${project_root}/.opd-git"
  export GIT_WORK_TREE="${project_root}"
fi
export PROJECT_ROOT=${PROJECT_ROOT:-${project_root}}
export PYTHON_BIN=${PYTHON_BIN:-${project_root}/.venv/bin/python}
export ACCELERATE_BIN=${ACCELERATE_BIN:-${project_root}/.venv/bin/accelerate}
export OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
run_id=${G0_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir=${G0_RUN_DIR:-${OUTPUT_ROOT:-outputs}/g0/${run_id}}
mkdir -p "${run_dir}/logs"

if ! "${PYTHON_BIN:-.venv/bin/python}" -m posttrain_circuits.cli.preflight_g0 \
  g0=qwen_eap_separation model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --output "${run_dir}/preflight.json"; then
  cp "${run_dir}/preflight.json" "${run_dir}/g0.json"
  cp "${run_dir}/preflight.md" "${run_dir}/g0.md"
  exit 1
fi

partition=${SLURM_GPU_PARTITION:-${SLURM_PARTITION:?Set SLURM_PARTITION}}
walltime=${SLURM_G0_TIME:-12:00:00}
submission=$(sbatch --parsable --account="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}" \
  --partition="${partition}" --time="${walltime}" --export=ALL,G0_RUN_DIR="${run_dir}" \
  --output="${run_dir}/logs/g0-%j.out" --error="${run_dir}/logs/g0-%j.err" \
  scripts/slurm/g0_qwen.slurm)
job_id=${submission%%;*}
printf '%s\n' "${job_id}" > "${run_dir}/job_id.txt"

while squeue -h -j "${job_id}" | grep -q .; do
  sleep 30
done
state=$(sacct -nP -j "${job_id}" --format=State | sed -n '1p' | cut -d'|' -f1)
if [[ "${state}" != COMPLETED* ]]; then
  echo "G0 job ${job_id} ended in ${state}; inspect ${run_dir}/logs" >&2
  exit 1
fi
"${PYTHON_BIN:-.venv/bin/python}" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("passed") is True else 1)' \
  "${run_dir}/g0.json"
printf '%s\n' "G0 PASS: ${run_dir}/g0.json"
