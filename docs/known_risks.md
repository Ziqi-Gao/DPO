# Known risks

- **Model-family dependence.** Qwen and Gemma may learn different circuits; one backbone cannot
  establish universality.
- **Spurious reward effects.** Exact correctness can correlate with format or length. Format-only and
  random-matched controls diagnose only part of this risk.
- **Circuit estimator noise.** EAP-IG rankings vary with prompts and integration paths. Bootstrap and
  fixed discovery pairs are mandatory.
- **Stage over-interpretation.** A final-answer circuit can omit rule selection or intermediate
  deduction. Claims must name the probed stage and cannot generalize from answer to full reasoning.
- **GQA conversion errors.** Treating query heads as independent K/V heads invalidates Qwen patching;
  compatibility and explicit mappings gate analysis.
- **Attribution/patching approximation.** High attribution need not imply a large exact effect.
  Agreement and calibration must be reported.
- **Self-repair.** Downstream compensation can hide direct necessity. Necessity, sufficiency, direct,
  and final effects are reported together.
- **Synthetic external validity.** ProofGraph controls semantics but does not establish mechanisms of
  open-domain or free-form mathematical reasoning.
- **Insufficient replay positives.** Low success yields unstable or undefined replay batches. The
  trainer retries then fails diagnostically.
- **Teacher/student incompatibility.** Low retained top-k mass or incompatible tokenizers distorts
  distillation. Retained mass and token spans are audited.
- **Online rollout divergence.** Natural online states rapidly diverge between methods. Shared-state
  local forks and canonical-prefix analyses are primary for causal comparisons.
- **Anchor selection bias.** Anchor tasks require a preregistered base-accuracy threshold and frozen
  discovery set before post-training.
- **Cluster nondeterminism.** Distributed kernels may not be bitwise deterministic. Resume is tested
  within a numerical tolerance on each target execution stack.
- **Superseded artifacts.** Pre-v2 datasets encode a label-leaking non-provability task, and pre-v2
  circuit artifacts use prompt-end targets. They are scientific history only and formal loaders
  reject them rather than attempting migration in place.
