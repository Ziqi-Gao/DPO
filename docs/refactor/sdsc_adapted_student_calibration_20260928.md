# Accepted adapted teacher → student calibration

The accepted teacher from fit `54494742`, qualification `54496291` and independent
CPU audit `54497294` now has a separate student consumer. The original stopped
v3 flows remain historical; do not restart them or substitute this teacher into
their original prerequisites.

The current accepted successor,
`prereg/amendments/qwen3_adapted_student_calibration_v4.json`, binds genuine
implementation commit `89a8ffd598d9377ebcbef556bee0057699d9eb35` and distinct
review-only acceptance `cfac02db4cf67c7d1a8c1b09697fcc25de76474b`.
It preserves all 47 accepted producer files and binds the named student
implementation. Acceptance of this protocol is not acceptance of GPU results.

The learned teacher identity is dense checkpoint SHA
`6928f2537dcca5f2d65c1498659e1ebf011845eb72ef364b9544036c2238e9c7`.
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` remains its base
ancestry. The original nine teacher input files are physically pinned and
preserved. The in-memory SFT adapter only materializes the explicit learned
teacher identity columns required by the existing deterministic demo source;
all 2048 candidates, tokens, log probabilities, masks, seeds and order remain
unchanged. Its 256-token teacher acceptance does not assert historical
128-token teacher readiness. Student evaluation stays at 128 tokens.

## V5 checkpoint precision correction

Remote inspection on 2026-09-30 confirms calibration 54509809 performed twenty
AdamW updates and failed when publishing its first step-20 checkpoint. All 311
model tensors have matching names/shapes: the physical initial export is BF16,
whereas Accelerate/FSDP training and full-state export use FP32 masters. The
strict comparator rejected this expected promotion at `lm_head.weight`.
Metrics consume 1,188,770/2,000,000 input tokens; step-20 loss is 2.681661978 and
answer/proof/format validation are each zero. This execution repair changes no
scientific hyperparameters or thresholds and establishes no student quality.

The separate proposed `qwen3_adapted_student_calibration_v5.json` retains exact
v1–v4 authority and all 47 accepted teacher files. Its comparison policy is gated
by agreeing top-level and nested v5 protocol selectors. Fresh runs require
uniform BF16→FP32; resumed runs require FP32→FP32 and valid unique ancestry.
Names/shapes and nonfloating values/dtypes remain exact; nonfinite, complex,
mixed or unreviewed precision is rejected. Original baseline files are unchanged.
The displacement uses FP64; the final tensor hash binds actual FP32 bytes.
Promotion alone produces zero and cannot pass the existing positive-update gate.
Trainer checkpoint publication and independent finalizer share this policy.

Real CPU Accelerator/FSDP tests reproduce the old failure, then exercise actual
save, independent finalizer, resume and the next optimizer update against the
uninterrupted model/optimizer hashes. The new GPU canary freezes an initial BF16
copy before preparation, rechecks its physical file, compares the full FP32
first-update state and requires exact zero displacement/hash equality after
same-world full-state restoration. CPU next-update equivalence is not claimed
as GPU equivalence. The final affected suite passed 346 tests; independent
review separately passed 217 focused tests. Ruff and Python parsing pass. Durable publication must bind the initial and full-state
files to their report. Fresh matching GPU evidence remains required.

The v5 controller retains one-submission protection, old failure rejection,
300-second polling and a fourteen-day bound. No new flow is currently active;
independent implementation acceptance precedes new deployment/submission.

## Execution

1. Check existing `.sdsc/submissions/` and supervision states, then the shared
   SSH master through `tools/sdsc check` on the current Quest host.
2. Review/commit the implementation and protocol acceptance, then perform a
   matching `sync --dry-run` and immutable `sync`. Export genuine provenance
   using `tools/sdsc_provenance.py create` with the actual implementation and
   acceptance commits and upload it after the upload dry-run. The bounded local
   history limit is 64 commits; public refs are never moved to bypass the bound.
3. Submit `qwen3-v2-adapted-preflight` with `--teacher-job-id 54496291` and
   explicit resources below. It restores reviewed science, verifies node-local
   and persistent mounts, stages the real selected teacher, and exercises a
   synthetic global-64 FSDP update through the actual CPU collation and canonical
   supervision boundary, full-state restore and adapted teacher
   forward. It produces `adapted-preflight.json` and durable checkpoint evidence.
4. Prepare a fresh release for `qwen3-v2-adapted-calibration`; submit it only
   after accounting shows every related step COMPLETED/0:0 and published results
   validate. Pass that actual `--preflight-job-id` and
   `--teacher-job-id 54496291`. The upstream must match the same protocol,
   scientific source, teacher, runtime and cache. No automatic retry is allowed.
5. Calibration executes the original full-parameter canonical-SFT entrypoint:
   seed 42, 256 unique prompts × 8 accepted demonstrations, global batch 64,
   maximum microbatch 4, 1536-token model input, 120-step/2M-token ceilings,
   validation/checkpoint every 20 steps. Its original strict artifact validator
   must accept before the wrapper can publish success.

Both tasks use account `nwu181`, partition `nairr-gpu-shared`, QoS
`nairr-gpu-shared-normal`, **2 H100, 24 CPUs, 192 GiB**. Preflight is bounded at
`01:00:00`; initial calibration uses `02:00:00`. Use the existing fixed runtime:
`/expanse/lustre/projects/nwu181/zgao12/OPD/envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12`
and HF cache `/expanse/lustre/projects/nwu181/zgao12/OPD/cache/huggingface`.
All Slurm calls execute remotely over the already authenticated SSH master.

The wrapper checks 256 GiB free node-local disk before copying inputs. It stages
21,350,818 bytes of original teacher evidence and, for calibration, the original
1,097,575,722-byte data family whose bytes were independently regenerated during
teacher isolation. Reusing this dataset does not accept failed teacher job
54345715. Both pinned Hub snapshots remain staged for the unchanged runtime
check; preflight additionally stages the actual learned dense checkpoint.

Outputs are read-back hashed on persistent project storage before the final
receipt. Calibration retains the nine teacher input files under
`artifacts/teacher_demos/`, the initial checkpoint, training checkpoints,
Accelerate state, metrics and reports. No large checkpoint is automatically
fetched to Quest. Status verifies small published bytes and all published file
sizes against the receipt; the allocation performs full large-file read-back.
A future checkpoint consumer must verify large contents again when staging.

Calibration success is **not complete G0, multistep resume equivalence, pilot,
factorial, or a Blackwell execution-class certificate**. The old G0/pilot adapters
still bind the historical teacher; further adaptation and independent review are
required before those stages can consume this learned teacher. Keep every
stage/job/run identity distinct, and reconcile a missing receipt before another
submission.

## CPU review evidence

Before GPU execution, the combined affected CLI/remote/qualification,
new student protocol/adapter/workers and original factorial-completion suite
passed 260 tests. Separate genuine-history export/restore tests passed, including
the 32-commit bound and rejection at 33. Thirteen wrapper regression tests cover
real read-back, tamper rejection, preservation of already published attempts,
nonpublication after semantic failure, and bounded draining of a TERM handler
that emits more than a pipe buffer. Ruff, shell syntax and Git whitespace checks
pass. These tests do not claim any new GPU execution success.

## Slurm startup correction

First preflight `54504816` failed after two seconds (FAILED / 2:0), before
Python or model execution. Slurm executes a spool copy of a submitted shell
script, so its location cannot identify the immutable source release. The new
transport entry `tools/sdsc_student_launch.sh` resolves the unchanged accepted
Python worker through the explicit release argument. Literal argv and exact
resource/input checks remain in the accepted Python worker. All 46 named
student files and 47 teacher files, including the historical launcher, stay
byte-identical. The independent transport review and actual new deployment
identity bind this locator correction; they do not create scientific acceptance
or permit reuse of failed-job evidence. Real regression fixtures execute the
launcher from a separate fake Slurm spool with literal metacharacter arguments.

## Accelerate FSDP correction

Preflight `54504895` passed both ranks and persistent publication in 14m04s.
Its matching calibration `54505782` then failed (FAILED / 1:0 in 5m13s), before
the first optimizer update. The original initial checkpoint export succeeded;
both training ranks rejected their prepared FSDP tree because
`_use_orig_params` was true instead of the reviewed false value.

The pinned Accelerate 1.10.1 launcher defaults `--fsdp_use_orig_params` to true
and writes that value into the child environment. The old shared YAML omitted
the setting. An external environment value cannot repair this because the
launcher overwrites it. The synthetic preflight directly constructed FSDP with
`use_orig_params=False`, so that PASS did not exercise this launcher seam.

The separate v2 protocol and
`configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml` explicitly restore the
already required false setting. Original shared configuration, accepted v1
protocol and teacher configuration remain preserved. The new
`qwen3_accepted_student_v2` group binds the same accepted teacher to the new
student protocol. No model, teacher evidence, scientific threshold, optimizer,
batch/token budget or validator is relaxed. The real parser → launch environment
→ FSDP plugin regression covers the legacy true setting and new false setting.

Execution must wait for a new implementation commit and independent review-only
acceptance of v2, then a fresh matching GPU preflight and calibration. Preserve
54504895 as valid evidence for its original version; the existing strict HEAD,
protocol and named-file guards prevent borrowing it for changed execution.
The failed calibration's full 7,209-byte training log has publication-verified
SHA `8e89ae1970096a1a1583a2628347e2f2280831c65032de9a5eca4d87a041dcb2`;
its bounded logs, original report and final accounting are retained under
`.sdsc/diagnostics/adapted-student-v1/calibration-54505782-*`.

## Student target-device correction

V2 preflight `54506703` completed and passed independent result review in 13m48s.
The finite Quest supervisor then automatically submitted calibration `54506821`.
Its initial checkpoint export succeeded, but both ranks failed at the first
replay-loss cross-entropy: predictions were on cuda:0/cuda:1 while targets stayed
on CPU. Slurm recorded FAILED / 1:0 after 8m31s. The separate read-only observer
retrieved bounded logs and the failure report under
`.sdsc/fetched/54506821/fetch-we8da5a6/`; both control flows are now stopped.
Fetched log tails normalize line endings and are not raw-file hash evidence.

The real CPU collator constructs a `SupervisionBatch`; FSDP transfers the model's
forward arguments to its GPU without moving the original batch retained by the
supervisor. The accepted v3 correction aligns only loss token IDs, response mask
and rewards with the actual output-logits device. It preserves tensor dtypes,
values, sample order, masking, sequence/token normalization and gradients, and
does not mutate the CPU source batch. `CanonicalSFTSupervisor` inherits this
implementation. The shared `losses.py` and all 47 accepted teacher files remain
unchanged.

The earlier preflight built all inputs directly on GPU and called the loss
function, bypassing the failing collation/supervision boundary. V3 instead keeps
the same canary tokens, global window, FSDP/checkpoint checks and numerical loss,
but constructs CPU trajectories with the production collator and executes the
real canonical supervisor. Per-rank evidence must establish this route; missing
or changed evidence fails v3 validation. Historical v1/v2 report validation stays
bound to each original protocol.

The new v3 protocol/config preserves both historical accepted student protocols,
the explicit-false v2 Accelerate YAML, teacher evidence and every scientific
threshold. It requires independent implementation/review-only acceptance,
followed by a new matching real GPU preflight and a separate fresh calibration.
No failed job is retried with its previous intent and no v2 GPU report accepts
the changed v3 implementation. The current handoff records actual acceptance,
deployment and job state; source existence alone is not operational readiness.

## V4 optimizer binding correction

V3 preflight `54507345` passed; its automatically submitted calibration `54507464`
failed after 5m10s at the unchanged AdamW cadence validator. The actual training
metrics file is empty. Neither a called `optimizer.step()` nor a global-step
counter establishes a successful parameter update.

The pinned Accelerate 1.10.1 FSDP1 preparation replaces model parameters when
`use_orig_params=false`, but does not remap an existing optimizer. A deterministic
CPU regression uses real Accelerate, real PyTorch FSDP flattening and AdamW,
with a single-rank FakeProcessGroup for communication only. It reproduces the
same error through `FactorialTrainer`: new model parameters have gradients,
the optimizer owns the old objects, and its state remains empty after stepping.
This is a CPU reproduction, not multi-rank/GPU evidence.

The shared `prepare_accelerate_model_optimizer_scheduler` prepares the model
first, binds the same fresh single-group AdamW to its prepared trainable
parameters, and then prepares the optimizer. The original raw LambdaLR and
optimizer objects, options and step hooks remain intact. Existing state,
gradients, multiple groups, incorrect scheduler binding or incomplete ownership
are rejected before FSDP preparation. Non-FSDP paths retain their existing order.
The original cadence, batch, loss, RNG, teacher and scientific thresholds remain.

V4's two-H100 canary uses this same helper on the actual Qwen3-1.7B model through
the pinned Accelerator, with settings resolved by the real training launcher
from the unchanged v2 YAML. Framework environment is applied after model loading.
FSDP uses the production NCCL default group; CPU/error/publication collectives
and monitored barriers use a separate Gloo group. The known global64 synthetic
window then requires exact prepared-parameter ownership, nonempty FP32 AdamW
state, raw scheduler binding, unchanged cadence, nonzero update and full-state
save/restore. This covers shared production preparation and one synthetic
window; it does not claim the complete production trainer loop was exercised.
Only a new matching GPU result may open the next fresh calibration.

The separate `tools/sdsc_student_supervise_v4.py` binds a fresh accepted-v4
profile; v1/v2/v3 protocols and stopped flow records remain historical.
Source/provenance uploads, review-only acceptance, single-submit claims and
terminal result checks remain mandatory. The bounded genuine-history export
capacity is 64 unpublished commits; every historical tree is still inspected,
with the original per-file, tree/bundle byte limits and exact genuine lineage.
No Git ref is moved and no history is dropped to fit the export.
