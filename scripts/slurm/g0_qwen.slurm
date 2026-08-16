#!/usr/bin/env bash
#SBATCH --job-name=opd-g0-qwen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_dir=${G0_RUN_DIR:?Set G0_RUN_DIR}
data_root="${run_dir}/datasets"
initial_checkpoint="${run_dir}/checkpoints/initial.pt"
demo_store="${run_dir}/teacher_demos"
calibration_run="${run_dir}/calibration_sft"
calibration_checkpoint="${calibration_run}/checkpoints/step-00000120.pt"
probe_scores="${run_dir}/probes/scores.json"
probe_root="${run_dir}/probes/cohorts"
bank_root="${run_dir}/bank/common_mu"
teacher_bank="${run_dir}/bank/common_mu_scored"
circuit_root="${run_dir}/circuits/base_capable"

mkdir -p "${run_dir}/logs" "${run_dir}/checkpoints" "${run_dir}/probes" "${run_dir}/bank"

for split in validation circuit_discovery circuit_validation; do
  "${PYTHON_BIN}" -m posttrain_circuits.cli.generate_task \
    g0=qwen_eap_separation model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
    split="${split}" --output "${data_root}/${split}" --confirm-production
done

"${PYTHON_BIN}" -m posttrain_circuits.cli.export_initial_checkpoint \
  g0=qwen_eap_separation model=qwen25_1p5b --output "${initial_checkpoint}" --confirm-production
initial_checkpoint_hash=$("${PYTHON_BIN}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${initial_checkpoint}")

"${PYTHON_BIN}" -m posttrain_circuits.cli.build_teacher_demos \
  pilot=qwen_core experiment=canonical_sft model=qwen25_1p5b teacher=qwen25_teacher_7b \
  task=proofgraph_main task.num_examples=256 state_source.store_path="${demo_store}" \
  --output "${demo_store}" --confirm-production

"${ACCELERATE_BIN:-accelerate}" launch --config_file configs/accelerate/fsdp.yaml \
  -m posttrain_circuits.cli.train pilot=qwen_core experiment=canonical_sft \
  model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  task.validation_split_path="${data_root}/validation" \
  production_safety.initial_checkpoint_path="${initial_checkpoint}" \
  production_safety.initial_checkpoint_hash="${initial_checkpoint_hash}" \
  state_source.store_path="${demo_store}" trainer.max_steps=120 trainer.checkpoint_every=120 \
  --output "${calibration_run}" --confirm-production

"${PYTHON_BIN}" -m posttrain_circuits.cli.score_probe_candidates \
  g0=qwen_eap_separation model=qwen25_1p5b task=proofgraph_main \
  --initial-checkpoint "${initial_checkpoint}" --calibration-checkpoint "${calibration_checkpoint}" \
  --discovery-split "${data_root}/circuit_discovery" \
  --validation-split "${data_root}/circuit_validation" \
  --task-validation-split "${data_root}/validation" \
  --probe-limit-per-split 128 --task-validation-limit 128 --output "${probe_scores}"

learnability_hash=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "${probe_scores}")
"${PYTHON_BIN}" -m posttrain_circuits.cli.build_probe_cohorts \
  --splits-root "${data_root}" --scores "${probe_scores}" \
  --initial-checkpoint-hash "${initial_checkpoint_hash}" \
  --learnability-evidence-hash "${learnability_hash}" --limit-per-split 128 --output "${probe_root}"

"${PYTHON_BIN}" -m posttrain_circuits.cli.build_rollout_bank \
  g0=qwen_eap_separation model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  task.num_examples=256 --output "${bank_root}" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.score_teacher \
  experiment=offline_soft model=qwen25_teacher_7b task=proofgraph_main \
  --bank "${bank_root}" --output "${teacher_bank}" --confirm-production

"${PYTHON_BIN}" -m posttrain_circuits.cli.evaluate_anti_shortcut \
  g0=qwen_eap_separation model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  production_safety.initial_checkpoint_path="${initial_checkpoint}" \
  production_safety.initial_checkpoint_hash="${initial_checkpoint_hash}" \
  --output "${run_dir}/anti_shortcut.json" --confirm-production

"${PYTHON_BIN}" -m posttrain_circuits.cli.discover_circuit \
  g0=qwen_eap_separation model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --checkpoint "${initial_checkpoint}" --probe-cohort-manifest "${probe_root}/manifest.json" \
  --initial-checkpoint "${initial_checkpoint}" \
  --cohort base_capable --output "${circuit_root}/circuit.json" --confirm-production
"${PYTHON_BIN}" -m posttrain_circuits.cli.evaluate_circuit \
  g0=qwen_eap_separation model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --circuit-artifact "${circuit_root}/circuit.json" --checkpoint "${initial_checkpoint}" \
  --initial-checkpoint "${initial_checkpoint}" \
  --probe-cohort-manifest "${probe_root}/manifest.json" --cohort base_capable \
  --output "${circuit_root}/exact_patching.json" --confirm-production

for replica in a b; do
  "${ACCELERATE_BIN:-accelerate}" launch --config_file configs/accelerate/fsdp.yaml \
    -m posttrain_circuits.cli.train pilot=qwen_core experiment=canonical_sft \
    model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
    task.validation_split_path="${data_root}/validation" \
    production_safety.initial_checkpoint_path="${initial_checkpoint}" \
    production_safety.initial_checkpoint_hash="${initial_checkpoint_hash}" \
    state_source.store_path="${demo_store}" trainer.max_steps=121 trainer.checkpoint_every=1 \
    --resume "${calibration_checkpoint}" --output "${run_dir}/resume-${replica}" --confirm-production
done
"${PYTHON_BIN}" -m posttrain_circuits.cli.compare_distributed_resume \
  --checkpoint-a "${run_dir}/resume-a/checkpoints/step-00000121.pt" \
  --checkpoint-b "${run_dir}/resume-b/checkpoints/step-00000121.pt" \
  --metrics-a "${run_dir}/resume-a/metrics.jsonl" --metrics-b "${run_dir}/resume-b/metrics.jsonl" \
  --world-size 4 --output "${run_dir}/distributed_resume.json"

"${PYTHON_BIN}" -m posttrain_circuits.cli.finalize_g0 \
  g0=qwen_eap_separation model=qwen25_1p5b teacher=qwen25_teacher_7b task=proofgraph_main \
  --base-scores "${probe_scores}" --teacher-store-manifest "${teacher_bank}/manifest.json" \
  --anti-shortcut "${run_dir}/anti_shortcut.json" --probe-manifest "${probe_root}/manifest.json" \
  --circuit "${circuit_root}/circuit.json" --exact-patching "${circuit_root}/exact_patching.json" \
  --compatibility "${circuit_root}/mib_raw/compatibility.json" \
  --distributed-resume "${run_dir}/distributed_resume.json" \
  --initial-checkpoint "${initial_checkpoint}" --gpu-preflight "${GPU_PREFLIGHT_JSON:?}" \
  --job-id "${SLURM_JOB_ID}" \
  --output "${run_dir}/g0.json"
