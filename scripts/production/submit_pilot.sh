#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${project_root}"
if [[ -d .opd-git && ! -f .git/HEAD ]]; then
  export GIT_DIR="${project_root}/.opd-git"
  export GIT_WORK_TREE="${project_root}"
fi
g0_json=${G0_JSON:?Set G0_JSON to a passed g0.json}
run_id=${PILOT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir=${PILOT_RUN_DIR:-${OUTPUT_ROOT:-outputs}/pilot/${run_id}}
mkdir -p "${run_dir}/logs"
"${PYTHON_BIN:-.venv/bin/python}" -m posttrain_circuits.cli.prepare_pilot \
  pilot=qwen_core experiment=offline_hard --g0 "${g0_json}" \
  --output "${run_dir}/pilot_manifest.json"

g0_root=$(dirname "${g0_json}")
export PILOT_RUN_DIR="${run_dir}"
export PILOT_COMMON_BANK="${g0_root}/bank/common_mu_scored"
export PILOT_TEACHER_DEMOS="${g0_root}/teacher_demos"
export PILOT_ANTI_SHORTCUT="${g0_root}/anti_shortcut.json"
export PILOT_PROBE_MANIFEST="${g0_root}/probes/cohorts/manifest.json"
export PILOT_READINESS_REPORT="${g0_root}/readiness/readiness.json"
partition=${SLURM_GPU_PARTITION:-${SLURM_PARTITION:?Set SLURM_PARTITION}}
submission=$(sbatch --parsable --account="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}" \
  --partition="${partition}" --export=ALL --output="${run_dir}/logs/pilot-%A_%a.out" \
  --error="${run_dir}/logs/pilot-%A_%a.err" scripts/slurm/pilot_qwen_core.slurm)
job_id=${submission%%;*}
printf '%s\n' "${job_id}" > "${run_dir}/job_ids.txt"
while squeue -h -j "${job_id}" | grep -q .; do
  sleep 30
done
sacct -nP -j "${job_id}" --format=JobID,State,ExitCode > "${run_dir}/terminal_states.txt"
if grep -Ev '^(JobID|[0-9_]+\|COMPLETED)' "${run_dir}/terminal_states.txt" | grep -q .; then
  echo "pilot training array has failed tasks; inspect ${run_dir}/logs" >&2
  exit 1
fi
echo "pilot training stage terminal: ${run_dir}"
