# Local-fork protocol

A fork bundle freezes one model checkpoint, optimizer/scheduler state, all RNG states, prompt batch,
sampled trajectories, behavior log probabilities, exact rewards, teacher top-1 and top-k targets,
pre-update probe outputs, and upstream manifest hashes. The bundle is content-bound by checkpoint,
prompt, and trajectory hashes.

The formal source is fixed to the registered `canonical_sft`, seed-42 run. Its run manifest,
`ExperimentBinding`, final checkpoint, factorial design, dataset manifests, initial checkpoint,
resolved model/tokenizer, prompt protocol, preregistration, and implementation identity are all
validated against an independently selected pilot source. LocalFork configuration does not infer
this identity from a legacy Slurm path convention.

The primary behavioral displacement is `KL(output_new || output_fork)` on the hash-pinned fixed
probe inputs and exact non-padding attention mask saved in the bundle. The frozen rollout-bank,
prompt, and probe manifests are reopened at finalization; selected trajectory records and the probe
token/mask bytes must reproduce the bundle. The source model must reproduce baseline logits exactly,
and every post checkpoint must reproduce its recorded post-update logits. For each nominal
1/5/20-update horizon, hard teacher sets the KL
target and the other branches calibrate learning rate until the configured relative tolerance is
met or the calibration budget is exhausted. Parameter-update norm is always recorded, but both it
and update count are secondary axes rather than the main matched comparison.

Signal branches load that same bundle and run hard teacher, soft teacher, verified replay, and
`centered_policy_gradient`. The policy-gradient branch requires at least four trajectories per
prompt group and within-group reward variance. It freezes
`(reward - group_mean)/(group_std + epsilon)`, uses stored old-policy response log probabilities,
and optimizes a clipped likelihood-ratio surrogate with both positive and negative advantages. No
branch resamples prompts or responses.
Horizons 1, 5, and 20 start independently from the common state; twenty is primary and one is a
gradient diagnostic.

Every trainable parameter appears exactly once in the canonical AdamW group/name mapping and must
have a complete finite AdamW moment row. Apart from the explicit calibrated learning rate, every
parameter-group key and hyperparameter is identical to the bundle and pre checkpoint; moment
shape/dtype and step transitions are exact. Report validation restores the pre checkpoint, frozen
trajectory inputs, pad token, optimizer/scheduler, and RNG, then deterministically replays every
update to independently reproduce each loss/metric row and the complete post state. Calibration
uses the frozen clamped `previous_lr * sqrt(target/observed)` equation; report, bundle, checkpoint,
or calibration-row relabeling is rejected even after rehashing.

Trajectory IDs are globally unique in a bank and are checked again at bundle creation and formal
validation. Bundle, checkpoint-tree, and report publication is no-clobber; the exact tree includes
directory nodes, so an extra empty directory is also a conflict. Identical reruns are idempotent;
conflicting existing artifacts are never overwritten.

The old uncentered binary reward-weighted estimator is retained only as a diagnostic because on
0/1 rewards it is gradient-collinear with positive-only verified replay. Centered policy gradient
must show distinct gradient geometry. This local branch is a controlled policy-gradient comparison,
not full GRPO; official TRL GRPO remains the canonical external anchor.

The state-source fork holds the soft-teacher objective fixed while selecting trajectories from the
common behavior policy, initial student, current fork checkpoint, or teacher. Matching strata are
prompt identity where possible, response length, verifier reward, and teacher entropy.

Unmatched results preserve the configured objectives and learning rates. Matched results calibrate
learning-rate scale against output KL and never alter the objective to force equality. Both results
and calibration residuals are retained. Parameter-norm matching remains an explicitly secondary
sensitivity analysis.
