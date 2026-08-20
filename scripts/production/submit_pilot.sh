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
g0_json=${G0_JSON:?Set G0_JSON to a passed g0.json}
run_id=${PILOT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
run_dir=${PILOT_RUN_DIR:-${OUTPUT_ROOT}/pilot/${run_id}}
mkdir -p "${run_dir}/logs"
source scripts/production/slurm_supervision.sh
"${PYTHON_BIN}" -m posttrain_circuits.cli.prepare_pilot \
  pilot="${PILOT_CONFIG}" model="${MODEL_CONFIG}" teacher="${TEACHER_CONFIG}" \
  experiment=offline_hard output_root="${OUTPUT_ROOT}" --g0 "${g0_json}" \
  --output "${run_dir}/pilot_manifest.json"

g0_root=$(dirname "${g0_json}")
export PILOT_RUN_DIR="${run_dir}"
export PILOT_COMMON_BANK="${g0_root}/bank/common_mu_scored"
export PILOT_TEACHER_DEMOS="${g0_root}/teacher_demos"
export PILOT_ANTI_SHORTCUT="${g0_root}/anti_shortcut.json"
export PILOT_PROBE_MANIFEST="${g0_root}/probes/cohorts/manifest.json"
export PILOT_READINESS_REPORT="${g0_root}/readiness/readiness.json"
export PILOT_VALIDATION_SPLIT="${g0_root}/datasets/validation"
export PILOT_INITIAL_CHECKPOINT="${g0_root}/checkpoints/initial.pt"
export PILOT_INITIAL_CHECKPOINT_SHA256=$("${PYTHON_BIN}" -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "${PILOT_INITIAL_CHECKPOINT}")

for cell in offline_hard online_hard offline_soft online_soft_opd \
  offline_verified_replay online_verified_replay canonical_sft canonical_grpo; do
  extra=()
  if [[ "${cell}" == offline_* ]]; then
    extra+=(state_source.store_path="${PILOT_COMMON_BANK}")
  elif [[ "${cell}" == canonical_sft ]]; then
    extra+=(state_source.store_path="${PILOT_TEACHER_DEMOS}")
  fi
  "${PYTHON_BIN}" -m posttrain_circuits.cli.assert_production_config \
    pilot="${PILOT_CONFIG}" experiment="${cell}" model="${MODEL_CONFIG}" teacher="${TEACHER_CONFIG}" \
    task=proofgraph_main task.validation_split_path="${PILOT_VALIDATION_SPLIT}" \
    production_safety.initial_checkpoint_path="${PILOT_INITIAL_CHECKPOINT}" \
    production_safety.initial_checkpoint_hash="${PILOT_INITIAL_CHECKPOINT_SHA256}" \
    "${extra[@]}"
done
partition=${SLURM_GPU_PARTITION:-${SLURM_PARTITION:?Set SLURM_PARTITION}}
job_ids="${run_dir}/job_ids.txt"
: > "${job_ids}"

wait_for_job() {
  local stage=$1
  local job_id=$2
  local terminal="${run_dir}/terminal-${stage}.txt"
  wait_for_slurm_terminal "${job_id}" "${terminal}"
}

submit_stage() {
  local stage=$1
  local walltime=$2
  local script=$3
  local gpu_count=${4:-1}
  if [[ "${gpu_count}" == 4 ]]; then
    require_no_competing_opd_gpu_job "pilot ${stage}"
  fi
  local submission
  submission=$(sbatch --parsable --account="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}" \
    --partition="${partition}" --time="${walltime}" --export=ALL \
    --output="${run_dir}/logs/${stage}-%A_%a.out" \
    --error="${run_dir}/logs/${stage}-%A_%a.err" "${script}")
  local job_id=${submission%%;*}
  printf '%s=%s\n' "${stage}" "${job_id}" >> "${job_ids}"
  wait_for_job "${stage}" "${job_id}"
}

submit_stage training "${PILOT_TRAIN_TIME:-12:00:00}" scripts/slurm/pilot_qwen_core.slurm 4
training_job_id=$(awk -F= '$1=="training" {print $2}' "${job_ids}")
"${PYTHON_BIN}" -m posttrain_circuits.cli.finalize_pilot_training \
  pilot="${PILOT_CONFIG}" model="${MODEL_CONFIG}" teacher="${TEACHER_CONFIG}" \
  production_safety.initial_checkpoint_path="${PILOT_INITIAL_CHECKPOINT}" \
  production_safety.initial_checkpoint_hash="${PILOT_INITIAL_CHECKPOINT_SHA256}" \
  --run-dir "${run_dir}" --terminal "${run_dir}/terminal-training.txt" \
  --job-id "${training_job_id}" --output "${run_dir}/training_artifact_chain.json"
submit_stage initial_circuits "${PILOT_INITIAL_CIRCUIT_TIME:-06:00:00}" \
  scripts/slurm/pilot_initial_circuits.slurm
submit_stage final_circuits "${PILOT_FINAL_CIRCUIT_TIME:-08:00:00}" \
  scripts/slurm/pilot_final_circuits.slurm
submit_stage local_fork "${PILOT_LOCAL_FORK_TIME:-12:00:00}" scripts/slurm/pilot_local_fork.slurm
submit_stage resume "${PILOT_RESUME_TIME:-02:00:00}" scripts/slurm/pilot_resume.slurm 4
submit_stage dynamics "${PILOT_DYNAMICS_TIME:-02:00:00}" scripts/slurm/pilot_dynamics.slurm

"${PYTHON_BIN}" -m posttrain_circuits.cli.finalize_pilot \
  pilot="${PILOT_CONFIG}" model="${MODEL_CONFIG}" teacher="${TEACHER_CONFIG}" \
  --run-dir "${run_dir}" --g0 "${g0_json}" \
  --bank-manifest "${PILOT_COMMON_BANK}/manifest.json" \
  --probe-manifest "${PILOT_PROBE_MANIFEST}" \
  --validation-manifest "${PILOT_VALIDATION_SPLIT}/manifest.json" \
  --job-ids "${job_ids}" --output "${run_dir}/pilot_report.json"
printf '%s\n' "pilot terminal report: ${run_dir}/pilot_report.json"
