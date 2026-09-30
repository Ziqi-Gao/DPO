# Student learning-rate diagnosis after repeated-output collapse

This document fixes the bounded training-only diagnostic design under the
user's existing repair authorization. Read `current_handoff.md` and immutable
submission receipts for actual deployment/job status; source existence does not
establish activation. This diagnostic is
not a successor acceptance of formal calibration settings and cannot start G0,
pilot, a new teacher/prompt protocol, or a factorial. Independent data-path,
implementation and actual-deployment reviews are required before submission.

## Evidence and question

Independent full replay of job54558773 verifies160 prompts,960 responses and12
arms. Initial,step20,step33 all have zero strict successes. Every trained-model
256-token response repeats proof delimiters or R01 fragments. Canonical targets
fit256; no memory failure occurred. The separate initial-only instruction screen
54559253 also has0/32 strict successes in both arms and failed its predeclared
futility criterion. Its311-tensor comparison found the original checkpoint
identical to the pinned offline HF/BF16 load before any overwrite. Stop promotion
of that instruction candidate; this diagnosis keeps the original instruction.

The accepted student-v6 optimizer is full-parameter AdamW with constant5e-4,
betas(.9,.95),eps1e-8,weight_decay0, no warmup or clipping. First-update norm18.75
is compatible with lr*sqrt(1,720,574,976)=20.74. Large early losses and KL make an
excessive update plausible, but different steps use different batches, and the
logged KL covers one training prompt's prefix. Existing checks have not proved
loss shift, repeated stepping or distributed normalization bugs. Test these
possibilities directly instead of changing accepted settings by assumption.

## Frozen scope

- Original parent54548846, accepted science6c04f804b302184b8ff95d00fab404e0531ed8d6.
  Initial checkpoint SHA85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4;
  original resolved configuration SHA05872b4521802813640004bf614dad65a6e1d2c3c3662a7502ca11bec5e15dbe.
- Keep original accepted teacher-v7 prompt and teacher store, student-v6 model,
  pinned runtime/tokenizer, seed42 and original graph/verifier semantics. No
  candidate instruction, few-shot example, answer injection or output repair.
- Training consumes exactly four successive original global64 windows:256
  distinct training prompts in manifest order, original accepted-view demo
  cursor. It does not repeat the first64 as all four training windows.
- Two independently initialized arms, in declared order: original5e-4 then
  candidate5e-5. Each restores the same original initial checkpoint, fresh empty
  AdamW and scheduler, RNG and cursor state. Learning rate is the sole treatment.
  No grid search, warmup, clipping, additional steps or checkpoint selection.
- Preserve W2/FULL_SHARD,29 FSDP wrappers,32 sequences per rank, microbatch4 with
  eight accumulation calls, sequence-mean response loss including EOS, FP32
  optimizer masters and production BF16 forwards. Maximum input length1536.
  Diagnostic step ceiling4 is explicitly separate from formal calibration.
- Resources:2 H100,24 CPU,384 GiB,60 minutes; nwu181,
  nairr-gpu-shared/nairr-gpu-shared-normal. Check real ReqTRES/AllocTRES before
  submission, remaining within the standing four-GPU limit.
- Only original manifest and training data are eligible inputs. No validation,
  test, anti-shortcut or circuit examples/evaluators are loaded or generated.
  Diagnosis never calls the old formal completion/acceptance path.

## Zero-update data-path prerequisite

Before charged execution, independently audit the actual first64 accepted
teacher records consumed by the original cursor, anchored to the original
published ledger/view/manifest. Preserve a bounded raw subset with its selection
and source hashes. Reconstruct the real TeacherDemoStateSource conversion and
collation, including raw/chat prompt, response IDs/text, attention/response masks,
labels, global slots, ranks, attempt IDs and cursor state. Check actual tokenizer
encoding/decoding, valid original proofs, first response token and EOS supervision,
and absence of prompt/padding supervision. Cross-check the historical first
window's59,336 model-input tokens and5,896 response tokens.

The CPU tiny-model CE check uses independently indexed shifted log-softmax for
each sequence, then averages64 sequence losses; its analytic gradients must
match the real production supervisor. On GPU, a separate HF labels loss for each
sequence provides the real-model reference. HF's whole-batch token mean is a
different objective and is not a valid equality reference. The CPU prerequisite
does not claim real1.7B logits were replayed. The GPU worker repeats the actual
input and label checks before any optimizer update.

The actual prerequisite was completed on Quest and independently replayed by a
non-author. Report SHA is
`86e5cca158476439cc9f73881bd7e73869b4a6837d1089284369156c79f9c933`;
capture SHA is
`aae6b2fd3fae601f265998416a9d2de8a7340c8672e1bcf2ef026508ebdff8b5`.
Both replays produced identical report/capture/evidence bytes. All64 original
proofs verify;51 imported scientific files match the genuine parent. The actual
source/collator supervises exactly5,896 response prediction positions, including
EOS, and no prompt or padding position. Shifted CE error is8.89e-16 and analytic
gradient error5.43e-20; all five deterministic corruption cases reject. No data,
mask or shift defect was found. This is a synthetic-logit CPU check, not a real
1.7B optimization result. Detailed evidence and independent review are under
`.sdsc/diagnostics/student-quality-v2/teacher-batch-audit-*`.

## Measurements and failure separation

All diagnostic forwards preserve and restore model training modes plus Python,
NumPy,Torch CPU and CUDA RNG. They must not advance demo cursors, consume training
token reservations or change optimizer state. Both arms use identical measurement
positions and generation paths.

At each optimizer boundary, measure the same fixed first64 real teacher targets
before and after the update, using sequence-mean CE. Compare independent
per-sequence HF labels CE at the same parameter point with the production loss.
Canonical-target NLL on first32 training prompts is separately named: those
canonical strings need not be the selected teacher demonstrations.

The first actual AdamW update records the already reduced/unscaled gradient,
finite/norm statistics, g dot delta, predicted versus observed FP32 master
changes and numerical residuals. Use bounded chunks, not a new full-gradient
host copy. Record actual optimizer state.step, raw scheduler cadence and all
eight accumulation boundaries for each of the four updates. Numerical comparison
tolerances are fixed before execution: per-sequence labels CE uses
2e-5*max(1,abs(HF loss)); first-step FP32 master residuals use
32*eps32*max(abs(old),abs(predicted),learning_rate) per coordinate.
An unexplained mismatch stops causal interpretation and is never called an LR win.

At step0 and step4 only, export and strictly reload the full FP32 state, convert
forward parameters to BF16, then greedily generate256 tokens on the same first32
training prompts. Record all128 paired responses, full token IDs/text and original
verifier traces. Both arms and both time points use this same standalone path;
never call the ambiguous FSDP-delegated generate method. At each arm's step4,
also compare real FSDP-root and reloaded-model logits for the same first4 training
prompts, with attention masks and model modes matched. Report finite absolute,
RMS and top-token differences; do not claim bitwise-identical distributed math.

Interpretation is conditional: only aligned data/CE/optimizer/export evidence
can support the hypothesis that high LR raises fixed-batch loss while lower LR
avoids the same damage. Failure shared by both arms calls for further data or
implementation diagnosis. A reload-only discrepancy identifies a separate
export/inference concern. Improved teacher-forced likelihood with repeated free
generation is still a sequence-quality failure. No metric here is model acceptance.

## Execution and publication

Use a fresh immutable hashed release and new intent/job. Bind all control bytes,
this protocol and externally reviewed data audit. An independent permanent
scientific claim prevents duplicate runs across controller revisions; missing
receipts require reconciliation, not retries. Every Slurm command runs on SDSC
through the existing shared SSH master. No remote service or environment install.

The node rechecks the original source/teacher/checkpoint identities, actual input
and persistent output mounts, node-local disk capacity and the established
32-GiB/20% host-memory headroom. Observed OOM/kill and allocation-failure
counters belonging to this job must remain zero at every measured phase;
shared parent-cgroup history is not assigned to this job. Preserve raw memory
evidence on rejection. Stage and compute on node-local storage. Persist
required results and both final model-only checkpoints with hash read-back before
allocation exit; never rely solely on TMPDIR. Each model-only checkpoint is
bounded at8 GiB. Checkpoints stay on project storage
and are not fetched to Quest.

Small evidence consists of lr-probe.json, lr-batches.jsonl, lr-prompts.jsonl,
lr-records.jsonl, node-result.json and memory.json plus bounded logs/receipts.
Fetch is confined to .sdsc/fetched/<job-id>/ and never overwrites source. Failed
runs preserve partial diagnostic evidence without acceptance or resubmission.

Every teacher,student,formal-prompt,G0,pilot,factorial and scientific-acceptance
flag remains false, even if execution succeeds or low LR looks better. Complete
independent raw-data replay and accounting checks are required before a later
scientific decision. Most importantly, improving trained weights cannot repair
the original initial-checkpoint base-accuracy and anti-shortcut gates. Do not
substitute trained weights for initial or relax those thresholds.

## Quest command sequence

From the actual Quest repository, use `/usr/bin/python3.12` for the control CLI.
Read the handoff and any `.sdsc/student-lr/` plan/receipt first; an existing claim
requires status or reconciliation rather than another submission.

```sh
tools/sdsc check
tools/sdsc sync --dry-run
tools/sdsc sync
/usr/bin/python3.12 tools/sdsc_student_lr.py prepare \
  --run-id <new-synchronized-run-id> \
  --fetch-dir .sdsc/fetched/54548846/fetch-rhidqpza \
  --status-file .sdsc/diagnostics/adapted-student-memory-v6/terminal-status-54548846.json \
  --audit-report .sdsc/diagnostics/student-quality-v2/teacher-batch-audit-report.json \
  --audit-capture .sdsc/diagnostics/student-quality-v2/teacher-batch-audit-capture.json
/usr/bin/python3.12 tools/sdsc_student_lr.py submit --plan <returned-plan> --dry-run
# After independent review of these exact deployed bytes and dry-run evidence:
/usr/bin/python3.12 tools/sdsc_student_lr.py submit --plan <returned-plan> --authorize
/usr/bin/python3.12 tools/sdsc_student_lr.py status --plan <returned-plan>
/usr/bin/python3.12 tools/sdsc_student_lr.py fetch --plan <returned-plan>
```

Every controller operation uses the existing SSH master and remote Slurm. Keep
queries at five-minute intervals during an observation period. Authentication
loss stops observation for manual authentication. A missing receipt calls for
`reconcile --plan <returned-plan>`, never another `submit`. A failed scientific
or execution check preserves evidence and needs a reviewed new diagnosis;
no automatic retry or cancellation is implemented. A successful diagnostic
still has all scientific acceptance flags false.
