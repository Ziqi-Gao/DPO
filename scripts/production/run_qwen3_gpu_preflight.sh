#!/usr/bin/env bash
set -euo pipefail

for name in MODEL_CONFIG TEACHER_CONFIG PRODUCTION_CONFIG G0_CONFIG PILOT_CONFIG \
  PROJECT_ROOT PYTHON_BIN ACCELERATE_BIN OUTPUT_ROOT HF_HOME; do
  [[ -n "${!name:-}" ]] || { echo "Set ${name} explicitly" >&2; exit 2; }
done
[[ "${HF_HUB_OFFLINE:?Set HF_HUB_OFFLINE=1}" == 1 ]]
cd "${PROJECT_ROOT}"
run_id=${GPU_PREFLIGHT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir=${GPU_PREFLIGHT_RUN_DIR:-${OUTPUT_ROOT}/gpu-preflight/${run_id}}
mkdir -p "${run_dir}/logs"
export GPU_PREFLIGHT_OUTPUT="${run_dir}/gpu_preflight.json"
"${PYTHON_BIN}" -m posttrain_circuits.cli.record_qwen3_launch \
  --phase gpu_preflight --output "${run_dir}/launch_manifest.json"
partition=${SLURM_GPU_PARTITION:-${SLURM_PARTITION:?Set SLURM_PARTITION}}
dependency=()
if [[ -n "${QWEN3_SERIAL_DEPENDENCY:-}" ]]; then
  dependency+=(--dependency="afterany:${QWEN3_SERIAL_DEPENDENCY}")
fi
submission=$(sbatch --parsable --account="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}" \
  --partition="${partition}" "${dependency[@]}" --export=ALL \
  --output="${run_dir}/logs/preflight-%j.out" \
  --error="${run_dir}/logs/preflight-%j.err" scripts/slurm/qwen3_gpu_preflight.slurm)
job_id=${submission%%;*}
printf '%s\n' "${job_id}" > "${run_dir}/job_id.txt"
printf '%s\n' "Qwen3 preflight submitted: ${job_id} (${run_dir})"
