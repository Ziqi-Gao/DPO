# Core execution plan

The historical designs are retained at `prereg/core_v0.yaml` and `prereg/core_v1.yaml`; the active
frozen design is `prereg/core_v2.yaml`. A production run is valid only when its manifest records
the Git commit that last changed that file, its current SHA-256, and a clean preregistration status.
Editing the preregistration after that commit blocks production until the change is explicitly
committed as a new version.

`prereg/qwen3_v1.yaml` is a separate model/protocol track. It inherits the scientific endpoints
from `core_v2` without editing that frozen file and binds the replacement pair
`Qwen/Qwen3-1.7B` / `Qwen/Qwen3-8B` to exact Hub revisions, tokenizer/chat-template fingerprints,
and the common `qwen3_non_thinking_v1` formatter. Qwen2.5 configurations and artifacts remain
historical evidence and are never renamed or reused by the Qwen3 namespace.

## Execution order and gates

1. Build all isolated ProofGraph splits. Evaluate the initial student and freeze four exact circuit
   manifests: `base_capable/{discovery,validation}` and `challenge/{discovery,validation}`. Challenge
   membership requires an initially incorrect result plus hash-pinned learnability evidence from a
   prior pilot/calibration artifact.
2. Run the semantics-preserving anti-shortcut suite on the initial student. It must pass the gap,
   IID, transformed aggregate, and per-transformation capability floors in `core_v2`; zero/zero is
   a failure. Production submission and trainer entry both require the hash-valid full readiness
   report, not only the anti-shortcut and probe sub-gates.
3. Build one common behavior-policy rollout bank. Every offline factorial cell reads this exact bank
   hash. Teacher demonstrations remain a separate state source used only for canonical SFT.
4. Run the six Qwen2.5-1.5B-Instruct factorial cells and the canonical SFT/GRPO anchors. EAP-IG ranks
   candidates on discovery subsets; all causal claims come from held-out exact activation/path
   patching on validation subsets.
5. At each checkpoint, retain the complete discovery score vector and all same-checkpoint bootstrap
   score vectors. Report full-score Spearman, weighted overlap, cross-checkpoint mask transfer,
   held-out exact-patching effects, and `excess_churn = observed churn - estimator noise churn`.
   Thresholded Jaccard is diagnostic only.
6. Local forks keep nominal 1/5/20-update horizons, but the primary comparison is matched
   `KL(output_new || output_fork)` on the frozen probe set. Update count and parameter norm are
   secondary axes.
7. Before any training claim, run teacher answer/proof/process correctness separately from retained
   top-k mass. Build tokenizer-specific, stage-specific frozen circuit manifests. Final-answer
   circuits are evidence about the answer stage, not a proxy for the complete reasoning process.
8. Run `grpo_random_reward` before the final Qwen mechanism claim. Its Bernoulli marginal is frozen
   before individual completions are visible, and its reward is content-independent thereafter.

## G0 and pilot

`bash scripts/production/run_g0.sh` is the only entry point for G0. It performs a clean-Git,
frozen-prereg, pinned-MIB and Slurm preflight. Before G0, a short four-GPU preflight must establish
CUDA/NCCL operation and load the pinned offline Qwen snapshot with a finite real-model forward.
The hash-valid preflight is bound to the exact Git commit and model revision consumed by G0. G0
then submits and waits for the real Qwen production stack. A hash-bound `g0.json` must say
`passed: true`; submission alone is not success.

Only then may `bash scripts/production/submit_pilot.sh` prepare the explicit
`configs/pilot/qwen_core.yaml` single-seed (42) pilot. The pilot array contains the six factorial
cells plus distinct canonical SFT and GRPO anchors. It is not the three-seed confirmatory grid.
Every cell loads the same byte-hashed initial checkpoint. Formal validation is evaluated at frozen
checkpoint intervals and the final step; canonical GRPO records both step-0 and final validation
and `KL(output_new || output_initial)`. Pilot execution then runs initial circuits, final circuits,
1/5/20 local forks, distributed resume, and noise-corrected dynamics as separately monitored
stages. The pilot report requires all preregistered matched-accuracy summaries to stay within
observed formal-validation ranges; extrapolation is invalid.
Opposite-direction effects remain valid observations and never block escalation by direction.

### Qwen3-v1 execution track

The Qwen3 track uses the new `configs/model/qwen3_1p7b.yaml`,
`configs/teacher/qwen3_teacher_8b.yaml`, `configs/production/qwen3_primary.yaml`,
`configs/g0/qwen3_eap_separation.yaml`, and `configs/pilot/qwen3_core.yaml` profiles. Every
model-facing ProofGraph prompt—rollout, teacher generation/readiness, SFT/GRPO, formal validation,
anti-shortcut evaluation, probe scoring/tokenization, circuit analysis, and local forks—passes
through one formatter with one user message, `add_generation_prompt=true`, and
`enable_thinking=false`. Sampling is a new explicitly named Qwen3 protocol
(`temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`) and must not be described as a
single-variable comparison with the Qwen2.5 runs.

Qwen3 production artifacts live only under `outputs/qwen3-v1`. Initial checkpoints, rollout banks,
teacher artifacts, readiness results, cohorts, tokenized probes, circuits, resume/local-fork state,
G0, and pilot artifacts all fail closed on a missing or cross-model protocol binding. The Qwen3
entry points require explicit `MODEL_CONFIG`, `TEACHER_CONFIG`, `PRODUCTION_CONFIG`, `G0_CONFIG`,
`PILOT_CONFIG`, `PROJECT_ROOT`, `PYTHON_BIN`, `ACCELERATE_BIN`, and `OUTPUT_ROOT`; each run manifest
records them.

Before Qwen3 G0, `scripts/production/run_qwen3_gpu_preflight.sh` runs four ranks and requires CUDA,
NCCL all-reduce, both pinned snapshots in offline mode, simultaneous 1.7B student/8B teacher load,
a real soft-teacher forward/backward update, disjoint per-rank prompt shards, and FSDP save/resume.
`scripts/production/run_qwen3_g0.sh` consumes only its hash-valid passing artifact. A seed-42 pilot
may then be launched by `scripts/production/submit_qwen3_pilot.sh` only after a hash-valid Qwen3
`g0.json` says `passed: true`. The three-seed factorial and Gemma replication remain unauthorized
for this execution.

With only three training seeds, ordinary cluster-robust covariance estimates are not reported as
confirmatory inference. The registered output is seed-level cell means and seed-specific contrasts,
described as descriptive; cluster-robust inference requires at least five training seeds.

### Qwen3-v2 repaired feasibility track

The independently reviewable successor is `prereg/qwen3_v2.yaml`, with profiles
`qwen3_v2_1p7b`, `qwen3_v2_teacher_8b`, `qwen3_v2_primary`,
`qwen3_v2_eap_separation`, and `qwen3_v2_core`. Its scientific artifacts live only under
`outputs/qwen3-v2`; v1 outputs may supply model-cache bytes but never scientific evidence.

All four-GPU stages request 192 GiB. The preflight must demonstrate the enforced cgroup-v1 or
cgroup-v2 memory limit,
32-GiB/20% post-peak headroom, four-rank NCCL, distinct prompt shards, low-CPU-memory student
loading, a rank-zero 8B teacher, finite soft-teacher forward/backward, and FSDP save/resume. Before
each four-GPU `sbatch`, the supervisor checks for any other pending or allocated OPD GPU job.

Training stops at the first of the registered global token budget or `max_steps`. Exact distributed
model-input tokens are reserved before factorial/SFT optimizer windows; GRPO admits only updates
whose conservative distributed maximum fits and reduces the actual rank deltas after each update.
Consumption and stop reason are part of checkpoints, cell manifests, metrics, and the final hash
chain.

The seed-42 circuit plan is the complete feasibility matrix: eight cells × two cohorts × process
and final-answer stages, plus both stages/cohorts at the initial checkpoint. The report verifies 4
initial and 32 final rows, all exact-patching and noise-floor artifacts, local-fork output-KL,
distributed resume, training terminal evidence, and every metrics/checkpoint hash. This is a
pipeline-feasibility gate, not a confirmatory mechanism result. The full three-seed factorial and
Gemma replication remain forbidden without separate authorization.

## Core Gemma mini-replication

Gemma-2-2B-it with Gemma-2-9B-it as the same-family teacher is a core mini-replication, not a full
factorial. `configs/replication/gemma2_2b_core.yaml` and
`scripts/slurm/gemma_mini_replication.slurm` run only the canonical SFT anchor and these contrasts:

- `offline_soft` versus `online_soft_opd`
- `online_hard` versus `online_soft_opd`
- `online_verified_replay` versus `canonical_grpo`

OLMo, Edge Pruning, Track-B long-CoT, delta transplant, and five-seed expansion are secondary or
exploratory and never block the core pipeline. No production GPU job is launched by repository
validation or CPU acceptance commands.
