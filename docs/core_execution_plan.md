# Core execution plan

The historical design is retained at `prereg/core_v0.yaml`; the active frozen design is
`prereg/core_v1.yaml`. A production run is valid only when its manifest records
the Git commit that last changed that file, its current SHA-256, and a clean preregistration status.
Editing the preregistration after that commit blocks production until the change is explicitly
committed as a new version.

## Execution order and gates

1. Build all isolated ProofGraph splits. Evaluate the initial student and freeze four exact circuit
   manifests: `base_capable/{discovery,validation}` and `challenge/{discovery,validation}`. Challenge
   membership requires an initially incorrect result plus hash-pinned learnability evidence from a
   prior pilot/calibration artifact.
2. Run the semantics-preserving anti-shortcut suite on the initial student. It must pass the gap,
   IID, transformed aggregate, and per-transformation capability floors in `core_v1`; zero/zero is
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
