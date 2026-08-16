# Circuit protocol

Circuit analysis starts only after a machine-readable compatibility report passes. The report checks
HF/circuit-model logit parity, layer/head counts, GQA query-to-KV mapping, residual and MLP hook
locations, and identity hooks. Qwen GQA K/V tensors have `n_key_value_heads`; they are not indexed as
independent query-head K/V tensors. Parity above tolerance aborts extraction.

Production discovery calls the externally pinned MIB EAP-IG-input adapter. Full node or edge score
vectors and uncertainty are retained. Smoke uses five integration steps, standard discovery ten,
and validation fifty; fifty is never the default. Fixed discovery pairs are reused across methods,
seeds, and checkpoints. The repaired CPU vertical slice uses the in-repository activation-space
EAP-IG implementation with a reduced integration-step count; it validates routing and calibration,
but random tiny-model separation is not production scientific evidence.

Each semantic support-swap pair is expanded into three frozen probe stages:

- `first_rule_selection`: the explicit first rule-ID sequence;
- `intermediate_conclusion`: an explicit derived-literal sequence;
- `final_answer`: the explicit answer-token sequence after the answer prefix.

The tokenized manifest records clean/corrupt model inputs, full target-token sequences, metric
positions, intervention positions, tokenizer hash, semantic-pair hash, and tokenized-pair hash.
Clean/corrupt alignment is fail-closed; padding or a prompt-end shortcut is not accepted. Every
metric is the teacher-forced log probability of the complete target sequence. A final-answer
circuit is evidence only for that stage and is never treated as the complete reasoning circuit.

Masks are evaluated at 0.1, 0.2, 0.5, 1, 2, 5, 10, and 20 percent with counterfactual replacement,
mean replacement, zero ablation, and size/layer-matched random masks. Reports contain the whole
faithfulness curve, CPR area, CMD behavioral-distance curve, necessity, sufficiency where supported,
and uncertainty. Counterfactual replacement is primary; zero is sensitivity analysis.

EAP-IG is candidate-circuit discovery, not causal validation. Its candidates receive held-out exact attention-output, MLP-output, residual, and selected GQA-aware
Q/K/V intervention checks. Necessity replaces clean activations with corrupt ones. Sufficiency keeps
clean circuit activations in a corrupt computation. Attribution ranks are compared with exact
effects on the same frozen stage-specific validation manifest; near-noise agreement prohibits
edge-level interpretation.

Compensation records direct contribution, final behavior change, and downstream change. Repair is a
diagnostic, not a definition of relevance. Dynamics use weighted overlap and full-vector Spearman as
primary metrics, continuous/thresholded churn, bootstrapped final-circuit locking on matched axes,
cross-checkpoint and cross-method transfer, and operational lifecycle labels. Shared canonical proof
prefixes are the primary cross-method comparison; natural rollouts are stored separately.

Every checkpoint reruns discovery on bootstrap resamples of the same frozen discovery cohort and
retains each complete score vector. The primary churn report subtracts the mean same-checkpoint
continuous-churn floor from observed cross-checkpoint churn. It reports that `excess_churn` beside
full-score Spearman, weighted overlap, cross-checkpoint mask transfer, and held-out exact-patching
effects. Thresholded Jaccard is explicitly diagnostic and can never support a churn claim alone.
