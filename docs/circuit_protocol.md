# Circuit protocol

Circuit analysis starts only after a machine-readable compatibility report passes. The report checks
HF/circuit-model logit parity, layer/head counts, GQA query-to-KV mapping, residual and MLP hook
locations, and identity hooks. Qwen GQA K/V tensors have `n_key_value_heads`; they are not indexed as
independent query-head K/V tensors. Parity above tolerance aborts extraction.

Production discovery calls the externally pinned MIB EAP-IG-input adapter. Full node or edge score
vectors and uncertainty are retained. Smoke uses five integration steps, standard discovery ten,
and validation fifty; fifty is never the default. Fixed discovery pairs are reused across methods,
seeds, and checkpoints. The CPU vertical slice is explicitly named exact node screening and must not
be reported as EAP-IG.

Behavior metrics declare a semantic position/span: first proof token, selected rule token,
intermediate conclusion, or final answer. The binary primary is clean-answer versus corrupt-answer
logit difference. Multi-token targets use labeled sequence log probability and are never silently
pooled with single-token metrics.

Masks are evaluated at 0.1, 0.2, 0.5, 1, 2, 5, 10, and 20 percent with counterfactual replacement,
mean replacement, zero ablation, and size/layer-matched random masks. Reports contain the whole
faithfulness curve, CPR area, CMD behavioral-distance curve, necessity, sufficiency where supported,
and uncertainty. Counterfactual replacement is primary; zero is sensitivity analysis.

EAP-IG candidates receive exact attention-output, MLP-output, residual, and selected GQA-aware
Q/K/V intervention checks. Necessity replaces clean activations with corrupt ones. Sufficiency keeps
clean circuit activations in a corrupt computation. Attribution ranks are compared with exact
effects; near-noise agreement prohibits edge-level interpretation.

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
