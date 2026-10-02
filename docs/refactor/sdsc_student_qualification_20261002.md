# Prepared student qualification

The completed preparation job 54562507 fixes step 4 as the common starting-point
candidate. Its SHA-256 is
`23854ce0d6db4cb01cae898beccf1ac7be7613e5ee01ff5b1626daafb8349d31`.
The independently replayed development selection cannot be replaced by a later
checkpoint after seeing formal evaluation. The user authorized continuing the
gated training workflow; this document describes its first qualification stage.

`prereg/amendments/qwen3_student_qualification_v1.json` freezes the checkpoint,
parent preparation evidence, original populations and original thresholds.
Implementation and independent review acceptance precede any GPU execution.
Acceptance of the implementation is separate from acceptance of observed model
results. The old preparation, teacher and student protocols remain unchanged.

The stage performs exactly 896 greedy, native-BF16, 256-token generations:

- The original validation split's first 128 examples. At least 13 original
  `VerificationResult.answer_correct` outcomes are required.
- The original 128-example IID anti-shortcut population and all five original
  semantics-preserving transformations of every example. These 768 responses use
  complete-proof verifier reward. Require IID accuracy at least 0.10, transformed
  mean at least 0.08, every transformation at least 0.05 and the original
  one-sided IID-minus-transformed gap at most 0.05. Wilson intervals are reported;
  they are not an added acceptance gate.

All populations, rendering, tokenizer, parser, verifier, ordering and numerical
thresholds are preserved. Previous diagnostic exposure to validation128 is
disclosed; it cannot choose another preparation candidate. All 896 responses are
retained even when a gate fails. Failure stops progression without threshold
relaxation, checkpoint reselection or an automatic training extension.

The original anti-shortcut evaluator does not truncate long auxiliary prompts.
Independent CPU tokenization confirms maximum prefixes of 1,194 for validation,
1,172 for IID, 1,203 for symbol renaming, 1,172 for each ordering transformation,
1,364 for paraphrasing and 1,988 for OOD distractors. Preserve a 2,244-token
auxiliary inference envelope, including the complete 256-token allowance. This
records the original evaluator's actual shape; training's 1,536-token bound and
the preparation-only 1,600-token development bound remain unchanged.

Checkpoint loading must first restore the saved FP32 master values exactly,
including keys, shapes, dtypes and all tensor values, then create the native-BF16
inference copy. Loading FP32 values directly into an existing BF16 model would
round them before verification. Inference keeps the original greedy arguments,
no extra autocast and no KV cache. This stage neither trains nor resumes an
optimizer.

Execution uses one H100, 24 CPUs and 192 GiB for at most two hours, account
`nwu181`, partition `nairr-gpu-shared`, QoS `nairr-gpu-shared-normal`.
The native128 diagnostic took 12m38s including startup; preparation measured
roughly seven minutes per 256 responses on each GPU. The larger auxiliary
contexts and staging justify the two-hour bound. Maintain at most four
concurrently allocatable GPUs and require the unchanged memory headroom check.
Use the existing SSH master, a reviewed source snapshot and genuine provenance,
one permanent scientific claim and one reconciled submission receipt. Preserve
small reports and complete raw tokens on project storage with read-back hashes;
fetch no weights. An independent CPU auditor reconstructs all prompts, decodes
all response tokens and recomputes original verifier outcomes and gates.

A passing result establishes only these base/anti-shortcut subgates. Formal
initial/G0/pilot acceptance flags remain false. Subsequent work must migrate
artifact bindings to the actual prepared checkpoint, regenerate initial-dependent
rollouts and circuits, load the accepted dense teacher where needed and satisfy
the original calibration-improvement, paired-cohort and circuit gates.

Subsequent training configurations still inherit `5e-4` and must receive an
explicit prospectively reviewed learning-rate repair. Preparation's `5e-5`
does not automatically affect them. Preserve each method's actual semantics:
OPD's trainer completion cap is 128, while the GRPO supervision configuration
uses 256, eight generations and its own batching/accumulation settings. Do not
apply the preparation/G0 batch or length contract indiscriminately to GRPO.
OPD uses teacher scores on student trajectories and RL uses verifier rewards;
teacher-generated reference proofs are not a prerequisite for either method.
Full three-seed factorial and Gemma replication remain outside this scope.
