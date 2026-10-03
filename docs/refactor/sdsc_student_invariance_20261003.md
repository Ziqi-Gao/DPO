# Order-robust common student preparation v2

The user requested improvement after the fixed preparation-v1 candidate failed
qualification job 54606205. All 896 responses completed and were independently
replayed. Original validation passed at 55/128; anti-shortcut proof accuracy
fell from 49/128 IID to 144/640 across five equivalent transformations, a signed
gap of 0.1578125 against the unchanged 0.05 limit. Fact-order and rule-order
successes were 17/128 and 16/128, mostly with valid output formatting. This is a
scientific failure, not an out-of-memory event or an incomplete run.

The old preparation rendered every example's facts and rules in ID order. Its
canonical first support citation was F01, which was always displayed first.
The original step4 development result also hid zero successes among 178 branch
examples behind aggregate 175/512 accuracy. A post-hoc ID-as-position rewrite
recovers only three fact-order cases and no rule-order cases; rewriting model
outputs is neither a sufficient repair nor part of this successor. These
observations support testing presentation augmentation and a more complete
independent-development selection rule. They do not prove a unique causal
mechanism or establish that the new training will succeed.

## Prospective treatment

The authoritative candidate is
`prereg/amendments/qwen3_student_invariance_v2.json`, implemented by
`experiments/protocols/student_invariance.py`. Preserve every accepted v1
preparation/qualification file, failed result, original task/verifier and formal
numerical threshold. No historical checkpoint is reselected after exposure.
All compared methods must eventually begin from identical newly qualified bytes.
This is common task preparation, not one of the compared OPD/RL methods.

Start fresh from the same pinned native Qwen3-1.7B weights and tokenizer. Use
2,048 new base fit examples, retaining signed sibling pairs, starting at pair
seed 110000042. Each appears in four fixed views: original, facts permuted,
rules permuted, and symbols renamed with both orders permuted. The unchanged
anti-shortcut helpers preserve logic and canonical proof citations; canonical
targets are independently verified for every view. The 8,192 training sequences
are ordered by base example then view. No teacher-generated proof is required.

Create 256 new development base examples at seed 120000042, retaining complete
pairs. Every checkpoint evaluates all six views of all examples: original and
all five original anti-shortcut transformations, including 32 OOD distractors.
The fit and development transformation seeds are fixed separately. Assign complete
six-view base-example blocks alternately to the two ranks, so each rank evaluates
128 examples of every view; preserve seed42+global-ordinal. This prevents one
rank from receiving every long OOD case while its peer waits at the reduction. Check
semantic, ID, pair-group and raw-seed isolation against all 144,000 original
family rows, 8,192/512 teacher rows and 2,048/512 previous student preparation
rows. Every view stays with its source partition. Formal validation and
anti-shortcut outcomes have informed this disclosed design; neither their
examples nor their outputs choose a new checkpoint.

Keep full-parameter AdamW at constant 5e-5, betas (0.9,0.95), epsilon 1e-8,
weight decay zero, no added clipping/warmup, W2 FULL_SHARD, global64,
physical microbatch4 per rank and accumulation8. Prompt and padding tokens are
masked; original response/EOS sequence-mean loss and exact FP32 saved-master
reload remain required. All 128 complete optimizer windows are prospectively
bounded by a new **preparation-only 8,000,000 input-token budget**. The original
downstream 2M-token/120-step budget remains unchanged. The complete encoded
population must pass its budget before training; no filtering, truncation,
replacement or automatic extension is permitted.

Pinned-tokenizer measurement found 7,777,131 fit tokens and maximum training
input1535, within the unchanged1536 shape. New development OOD prompts require
up to2198 prefix tokens plus the complete256-token generation allowance. Declare
**2454 for this preparation's development inference only**; do not change
formal qualification2244 or any downstream training/rollout limit. Each rank's
real preflight also performs a finite native-BF16 no-cache forward at2454 using
fixed synthetic valid tokens, without consuming development examples.

## Frozen checkpoint choice

Save and evaluate at steps4,8,16,32,64,128. Complete all128 training steps and all
six full development evaluations, retaining every raw response. Select the
first scheduled checkpoint meeting every following additional development rule:

- Original-view answer correctness is26 through153 out of256, inclusive, keeping
  the prior10%-60% preparation ability band and room for subsequent improvement.
- Original-view complete-proof accuracy>=.10, mean transformed accuracy>=.08,
  every transformed accuracy>=.05, and signed IID-minus-mean gap<=.05.
- Each individual transformation's signed IID-minus-transformation gap<=.05;
  a strong view cannot hide one collapsed view in the mean.
- Original-view complete-proof accuracy>=.10 separately for chain, branch and
  converging-DAG examples. The immutable development population contains
  100/58/98 examples respectively.

The additional per-view and per-structure development rules do not relax or
replace any formal gate. No eligible checkpoint means failure with all evidence
preserved; no budget extension, threshold relaxation or post-formal reselection.
A selected development candidate still requires independent raw auditing and
subsequent qualification. No producer may claim formal-initial/G0/pilot success.

## Reviewed execution

Use separately named controller, node, worker and independent auditor tools
under `tools/sdsc_student_invariance*.py`, plus a new claim/intent namespace.
The old controls remain byte-for-byte frozen. Genuine implementation and distinct
non-author review-only acceptance commits precede execution. Source releases and
Git provenance are separately hash-verified; inspect existing receipts and
require matching sync and submit previews. Never blindly retry an unknown intent.

The matching preflight uses2 H100,24 CPU,384GiB for at most1h, testing four real
updates, full-state restoration, exact reload and eight training-only responses.
An independently reconstructed raw preflight audit is required before fit.
Fit restarts native weights with fresh optimizer/RNG/cursors; it never resumes
preflight weights. Its reviewed upper bound is8h with the same resources. The
old32-step/2048-response preparation took42m20s; qualification took48m26s for896
responses on one GPU. New9216 development responses over two GPUs plus128
updates and checkpoint IO justify the larger preparation walltime; the measured
preflight must still support admission. Keep at most four allocatable GPUs,
account nwu181, shared partition/QoS, original memory-headroom and own-job OOM
checks, exact scheduler-provided CUDA visibility and verified node-local mounts.

Require192GiB free node-local staging space and bound persistent large artifacts
by128GiB. Development response records are split by checkpoint and rank to
preserve16MiB per-file bounds; the fixed prompt file may use32MiB. Total fetched
small artifacts are bounded by112MiB. Preserve and read-back hash all required
checkpoints/results on persistent project storage before allocation exit; fetch
no model weights. Raw audit verifies source/protocol/publication identities,
all prompt encodings, token decoding, original parser/verifier traces, all
view/structure reductions and the earliest eligible selection, without claiming
to independently recompute GPU arithmetic.

Formal acceptance and further calibration/circuit/OPD/RL training remain gated
on actual new results. Full three-seed factorial and Gemma replication remain
outside the authorization. Implementation or CPU test success is not model
success.
