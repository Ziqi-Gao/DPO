# Shared Qwen3-1.7B student preparation

This is the prospective preparation design. The machine-readable review block
records implementation acceptance; the current handoff records actual job and
model status. Implementation acceptance alone establishes no GPU or G0 result.

The user explicitly selected **keep 1.7B and prepare a common initial model**
after requesting repair and experiment submission. This changes the research
starting point from a native instruction-tuned model to a task-prepared model.
Every later method must start from the same independently qualified new bytes.
The original initial, failed training, and completed diagnostic evidence remain
unchanged; this successor cannot retroactively pass the original experiment.

## Evidence and repair

The independently audited paired diagnostic `54560292` changes only constant
learning rate from `5e-4` to `5e-5`. The fixed-batch loss after one update changes
from `1.0555 -> 4.9576` to `1.0555 -> 0.5187`; after four updates the lower-rate
arm has five valid proofs among 32 training examples instead of zero. This is
evidence of excessive update size, not a demonstrated AdamW implementation bug
or proof of held-out improvement. The new preparation fixes `5e-5` prospectively.

The independently audited native initial check `54560398` has zero original
`answer_correct` results among 128 validation examples, below the unchanged
required 13. Repairing subsequent optimization cannot repair this fixed initial
criterion. The user-selected preparation is therefore a separately declared
scientific successor, not an LR-only retry or relabeling of old trained weights.

## Prospective preparation

The machine-readable amendment is
`prereg/amendments/qwen3_student_preparation_v1.json`. Its reviewed values and
implementation binding are authoritative; this document explains their scope.

- Start fresh from pinned Qwen3-1.7B revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and its verified original initial
  checkpoint. Preserve the original tokenizer, non-thinking template, v7 task
  rendering, parser, verifier, and ProofGraph difficulty distribution.
- Construct 2,048 fit and 512 development examples in separate seed namespaces
  starting at `90000042` and `100000042`, retaining complete sibling pairs.
  Verify semantic, example-ID, pair-ID and raw-seed isolation from all 144,000
  original family examples and all 8,192/512 teacher fit/development examples.
- Supervise the generator's canonical symbolic proofs, including EOS, with
  prompt/padding masking and the original sequence-mean response loss. No
  teacher-generated proof store or teacher-model inference is needed for this
  preparation. All encoded sequences must fit the unchanged 1,536-token bound;
  reject truncation and invalid canonical targets.
- Exact CPU tokenization found all fit sequences within 1,417 tokens and a
  complete fit budget of 1,856,564 tokens. Six development prefixes have 1,286
  or 1,292 tokens; their complete canonical training sequences still fit 1,455,
  but the longest prefix plus the full generation allowance needs 1,548.
  Preserve all examples and prospectively declare a **preparation-only**
  development inference limit of 1,600 tokens (prefix at most 1,344 plus 256).
  The FSDP training shape remains 1,536 and downstream formal limits are not
  changed. No truncation, filtering, seed replacement or model-output search is
  used to handle this measured input-length difference.
- Use full-parameter AdamW, constant `5e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`,
  weight decay zero, no added warmup or clipping, seed 42, global batch 64 and
  two equal rank-local batches of 32. Maximum physical microbatch is four;
  eight accumulation calls form one update on each rank. Use actual FSDP
  preparation and production optimizer/loss primitives with a separately named
  canonical-data state source. This is a fixed-two-H100 path, not a claim of
  elastic execution-class certification.
- The fit is one fixed pass of 32 updates through those 2,048 examples, bounded
  by 2,000,000 global nonpadding model-input tokens. Audit the complete encoded
  pass against that limit before training; a failed envelope stops the run.
  Reserve each global window before backward, preserving checkpointed counters.
- Evaluate all 512 independent development examples at updates 4, 8, 16 and 32
  from saved, strictly reloaded dense weights using native BF16 greedy inference
  and the original verifier. The 256-token development cap matches the original
  G0 base scorer. It does not change downstream student rollout cap 128.
- Complete the fixed fit and retain every development result. Select the
  earliest scheduled checkpoint with 52 through 307 original `answer_correct`
  results among 512 development examples, inclusive. This prospective
  preparation band corresponds to at least 10% and at most 60%; it aims to avoid
  an incapable or already saturated starting point. It is an additional
  preparation-selection rule, not a replacement or relaxation of any G0 gate.
  If none qualifies, report failure with no checkpoint reselection, automatic
  budget extension, or validation-guided search.

Original validation examples have already informed diagnosis; they are not new
holdout data and cannot select a preparation checkpoint. The preparation band
does not establish generalization, anti-shortcut behavior, suitable paired
circuit populations, or room for later improvement.

## Execution, review, and result boundaries

Create a genuine implementation commit and obtain non-author reviews of the
scientific protocol, worker and transport. A distinct review-only acceptance
commit binds the implemented protocol; it does not accept an unobserved model.
Preserve all existing 49 student and 47 teacher scientific files and old
amendments. The new worker and transport remain separate from old frozen flows.

Use a fresh two-H100 / 24-CPU / 384-GiB preflight, with at most one hour, before
the matching full preparation fit. The preflight exercises the same canonical
data, model, optimizer, accumulation, checkpoint and reload path for four
updates. It cannot produce a selected common baseline. The full fit restarts
from the original initial with fresh optimizer, RNG and data cursors. Its
initial wall-time envelope is four hours, subject to actual preflight evidence.
The original native check required 12m38s for 128 examples including startup;
four 512-example serial development evaluations therefore need substantially
more time than the historical 33-update training-only calibration. The preflight
also measures a fixed small training-only generation sample for runtime planning;
those responses do not select checkpoints or consume development examples.
The shared partition/QoS are `nairr-gpu-shared` / `nairr-gpu-shared-normal` under
account `nwu181`, and concurrent allocatable GPUs remain at most four.

Check the existing host-specific SSH master without fallback authentication.
Inspect all existing receipts/claims before any submission. Use a matching
snapshot preview/upload and genuine provenance, then remotely verify runtime,
storage and admission before submitting exactly once. Missing acknowledgement
requires reconciliation; no blind retry or re-arming of historical flows.
Each node verifies actual input/output mounts, local free space, unchanged
scheduler GPU visibility, its own memory cgroup, zero own-job OOM events and
the unchanged `max(32 GiB, 20%)` aggregate-memory headroom requirement.

Persist checkpoints and required raw results to project storage and verify
them by read-back before allocation exit. Fetch only bounded reports, full
development response/token evidence, and log excerpts; weights remain on SDSC.
Independent replay must reconstruct the declared development selection from
the original parser/verifier and immutable source/data/checkpoint identities.
Execution success and a selected candidate do not independently accept it as
the common initial model.

Before formal OPD/RL comparison, migrate the successor's artifact bindings and
qualify the selected checkpoint against every unchanged original base,
anti-shortcut, paired-cohort, circuit, calibration-improvement and G0 gate.
Regenerate all initial-dependent artifacts. Every comparison method must use
identical qualified initial bytes, with the preparation treatment disclosed.
No full three-seed factorial or Gemma replication is included.
