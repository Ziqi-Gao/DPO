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
export HF_HOME=${HF_HOME:-$(dirname "${project_root}")/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
run_id=${GPU_PREFLIGHT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir=${GPU_PREFLIGHT_RUN_DIR:-${OUTPUT_ROOT}/gpu-preflight/${run_id}}
mkdir -p "${run_dir}/logs"
export GPU_PREFLIGHT_OUTPUT="${run_dir}/gpu_preflight.json"
partition=${SLURM_GPU_PARTITION:-${SLURM_PARTITION:?Set SLURM_PARTITION}}
submission=$(sbatch --parsable --account="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}" \
  --partition="${partition}" --export=ALL --output="${run_dir}/logs/preflight-%j.out" \
  --error="${run_dir}/logs/preflight-%j.err" scripts/slurm/gpu_preflight.slurm)
job_id=${submission%%;*}
printf '%s\n' "${job_id}" > "${run_dir}/job_id.txt"
while squeue -h -j "${job_id}" | grep -q .; do
  sleep 30
done
state=$(sacct -nP -j "${job_id}" --format=State | sed -n '1p' | cut -d'|' -f1)
if [[ "${state}" != COMPLETED* ]]; then
  echo "GPU preflight ${job_id} ended in ${state}; inspect ${run_dir}/logs" >&2
  exit 1
fi
"${PYTHON_BIN}" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("passed") is True else 1)' \
  "${GPU_PREFLIGHT_OUTPUT}"
printf '%s\n' "GPU preflight PASS: ${GPU_PREFLIGHT_OUTPUT}"
