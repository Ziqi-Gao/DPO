# Local-fork protocol

A fork bundle freezes one model checkpoint, optimizer/scheduler state, all RNG states, prompt batch,
sampled trajectories, behavior log probabilities, exact rewards, teacher top-1 and top-k targets,
pre-update probe outputs, and upstream manifest hashes. The bundle is content-bound by checkpoint,
prompt, and trajectory hashes.

The primary behavioral displacement is `KL(output_new || output_fork)` on the hash-pinned fixed
probe inputs saved in the bundle. For each nominal 1/5/20-update horizon, hard teacher sets the KL
target and the other branches calibrate learning rate until the configured relative tolerance is
met or the calibration budget is exhausted. Parameter-update norm is always recorded, but both it
and update count are secondary axes rather than the main matched comparison.

Signal branches load that same bundle and run hard teacher, soft teacher, verified replay, and an
explicit shared-trajectory 1–0 REINFORCE diagnostic. No branch resamples prompts or responses.
Horizons 1, 5, and 20 start independently from the common state; twenty is primary and one is a
gradient diagnostic.

The state-source fork holds the soft-teacher objective fixed while selecting trajectories from the
common behavior policy, initial student, current fork checkpoint, or teacher. Matching strata are
prompt identity where possible, response length, verifier reward, and teacher entropy.

Unmatched results preserve the configured objectives and learning rates. Matched results calibrate
learning-rate scale against output KL and never alter the objective to force equality. Both results
and calibration residuals are retained. Parameter-norm matching remains an explicitly secondary
sensitivity analysis.
