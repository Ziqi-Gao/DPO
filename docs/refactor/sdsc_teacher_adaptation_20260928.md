# Qwen3-8B task adaptation and qualification

**Teacher acceptance requires the complete qualification and independent review.**
Protocol status is determined by the real review block and Git lineage of
`prereg/amendments/qwen3_teacher_adaptation_v1.yaml`; implementation or development
success alone does not accept the teacher. The user explicitly
authorized autonomous repair, test submission and continuation until the teacher
training task satisfies its standards. This covers the bounded adaptation work
below. It does not turn an execution PASS into scientific acceptance, waive
independent review, or authorize false checkpoint identities or relaxed gates.

The independently reviewed scientific decision below uses a **teacher-only
256-token readiness generation budget for the adapted-teacher candidate**.
It is a declared new protocol, implemented at
`d1ab8dacd834101b88d806bb6d75a44ae1949cb3` and independently accepted by the
distinct review-only commit `929fb14834852a7c91e6656c76fd1834e1b5007d`.
The eight-step preflight and all historical 128-token evidence retain their
original definitions. Protocol acceptance is separate from result acceptance.

## Accepted result: 2026-09-28

**Teacher training and independent acceptance are complete.** The actual accepted
artifact sets `formal_teacher_accepted=true`; this is a separate result decision
after protocol acceptance, raw-evidence replay and an independent reviewer.

| Actual SDSC job | Outcome | Elapsed |
| --- | --- | --- |
| 54494742 — full teacher fit, 4 H100 | COMPLETED / 0:0; all 512 updates and four epochs | 1h19m36s |
| 54496291 — qualification, 4 H100 | COMPLETED / 0:0; all four ordered stages pass | 1h15m14s |
| 54497294 — independent CPU replay | COMPLETED / 0:0; raw evidence and claims pass | 41m18s |

The unchanged first-development-PASS rule selects **step 128**. All four complete
512-row development evaluations pass, but no later checkpoint replaces this one.
Selected dense checkpoint identity:
`6928f2537dcca5f2d65c1498659e1ebf011845eb72ef364b9544036c2238e9c7`.
The full fit consumed 32,768 sequences / 29,909,768 nonpadding input tokens;
its 13 execution checks pass. Actual peak host memory was 108.24 GiB against
the 192-GiB workload budget, and maximum reserved GPU memory was 43.75 GiB.

Qualification measured, in the fixed order:

- Training probe: **256/256** candidates accepted, **32/32** prompts covered.
- Supplemental validation rows [128:256]: all eight gates pass; answer and
  exact-proof accuracy **100%**, minimum causal shift **+10.250116** and
  minimum retained mass **0.999999821**.
- Formal validation rows [0:128]: all eight gates pass; answer and exact-proof
  accuracy **100%**, minimum causal shift **+5.877720** and minimum retained
  mass **0.999999762**.
- Fresh teacher store: **2,048/2,048** candidates accepted, **256/256** prompts
  covered. Store identity:
  `f2e9e8171e3289bfd8bb356c79df67e56612f2ad067c63a4583b017c79025a88`.

Both validation cohorts also score 1.0 on first-rule, intermediate-conclusion,
whole-target coverage and corrupted-prefix recovery. No gate was lowered;
there was no checkpoint reselection, output rewriting or extra sampling.
This acceptance uses the independently accepted **teacher-only 256-token**
protocol. `original_128_token_readiness_pass_claim=false`; the original first
128 rows' prior exposure remains disclosed. Student rollout/evaluation remains
128 tokens, and **student training has not started**. Its continuation needs
coherent adapted-teacher bindings and a matching new preflight; do not re-arm
old v3 plans or treat the 256-prompt store as the pilot's required 4,096 prompts.

The fit published 71,277,357,278 bytes and qualification published 179,368,485
bytes to verified persistent SDSC storage. The selected 23-file dense checkpoint
(16,746,687,223 bytes) matched the fit publication during qualification staging;
every rank verified its manifest before inference. Independent review binds
these actual origin/load proofs; it does not claim to download or hash those
large weights a second time. Future consumers must verify the weights they load.
Quest fetched only bounded reports, logs and audit records.

The independent CPU replay read and hashed all raw scientific evidence and
permanent claim/outcome files. Its recomputed evidence exactly equals the
producer object. Independent reviewer `/root/teacher_result_review` then checked
this replay, all 47 scientific blobs, genuine Git lineage, all three jobs'
accounting, publication and weight provenance, and actual remote permanent
claims. The finalizer uses that real reviewer identity; the former reviewer's
protocol acceptance remains historical and was not reused as a result signature.
The identity-only control change and publisher pin were independently reviewed,
with five rejection cases and six transport tests passing.

| Evidence | Canonical SHA-256 |
| --- | --- |
| Actual independent audit | `161e86554396f43193d50e103a1a42b7bfcb82242edb79230e0cb6102b2a90f8` |
| Recomputed acceptance evidence | `c7b65df10069ab139179012c5590906783553f487de5bc581e80f40bbcf4801a` |
| Independent result attestation | `4641d1f2c54e627e41d0b2f0e1363c48cada91643fb5c18e946eb19a62a16964` |
| Accepted teacher | `5d6952823441bde567cdf7f5fad8b4625c58ee7e82425aad76c10433d0ec5337` |

Actual audit evidence is under
`.sdsc/fetched/54496291/batch-audit-b04f8e93a2cc7f3e2961de65ecdac7ca/`.
The local acceptance directory is
`.sdsc/diagnostics/teacher-adaptation-v1/accepted-teacher-54496291-4641d1f2c54e627e/`.
Its four files total **8,267 bytes**, with inventory SHA
`8d53783b9fb1d386de5a0a291c2b28e225347168cc7cfed4c0c0aceaf01d9bed`.
They were exclusively published to:

```text
/expanse/lustre/projects/nwu181/zgao12/OPD/teacher-acceptance/54496291/4641d1f2c54e627e41d0b2f0e1363c48cada91643fb5c18e946eb19a62a16964/
```

The publication receipt's physical SHA is
`f006ed152753565611052d2d7255c39b4bba89f67a4aa1f45b8466b0750d7f7f`.
Publication and a subsequent read-only check both returned `complete=true`,
identical file hashes and no missing files; the independent reviewer rechecked
both. The producer and audit reports remain immutable and do not self-accept.
Do not repeat these completed submissions, finalization or publication. See
[workflow recheck commands](../sdsc_workflow.md#当前-teacher-适配与验收).

The remaining sections preserve the experiment design, its historical rationale
and execution contract. They are not instructions to repeat completed stages.

The completed v7 diagnostic, job 54489646, accepted 42/256 candidates and covered
7/32 training prompts. Its necessary 32/32 gate failed; formal v7 readiness was
not run. The source/ledger audits and limits are recorded in
`sdsc_v7_quality_diagnosis_20260928.md`. The original parser correctly rejects
the demonstrated citation and premise errors. Missing TRUE is not the cause of
those rejections. Preserve the failed diagnostics and all stopped flows.

## Intervention and alternatives

The first adaptation candidate trains a low-rank update to the original
**Qwen3-8B** base, then exports a separate merged dense teacher checkpoint.
Training on independently generated canonical proofs addresses both selecting
the correct rule/premises and expressing a serial proof with valid references.
Paired positive/negative graphs require the output to follow the active facts,
which also targets the reasoning measured by the prefix gates. This is a new
teacher-training experiment, not a transparent fix to the original frozen
weights. Its measured efficacy is recorded in the accepted result above.

The alternatives rank below this candidate for the current acceptance target:

1. **Low-rank task adaptation, followed by dense export:** the smallest first
   training attempt in memory and dependency scope. Rank 32 is a capacity
   hypothesis, not evidence of sufficiency. Measure all eight readiness metrics
   on development data, including prefix behavior, before formal progression.
2. **Higher-capacity or full-parameter task adaptation:** a separately frozen
   successor if training/development evidence identifies insufficient capacity.
   Do not silently increase rank, steps or dataset size inside the first run.
3. **Thinking or demonstration prompts:** scientific protocol alternatives,
   rather than equivalent repairs. The current formal evaluator requires
   non-thinking mode and scores immediate rule/literal continuations without a
   generated reasoning trace. Demonstrations also add tokens to prefixes whose
   original 256-example envelope already reaches 1,246 tokens. Neither change
   alone establishes the existing prefix gates or preserves that input envelope.

Do not fix the observed failures by changing the verifier, reordering generated
responses, substituting symbolic answers at inference, or selecting extra
candidates. Those operations do not produce the required original teacher
evidence. The two offline counterfactual audits remain attribution only.

## Frozen first candidate

Base model: `Qwen/Qwen3-8B`, revision
`b968826d9c46dd6066d109eabc6255188de91218`. Keep the pinned tokenizer and its
fingerprint, the existing non-thinking chat template, v7 prompt rendering,
canonical targets and task difficulty distribution. The learned weight update
and the separately declared teacher-readiness generation-budget change below
are the interventions. Other generation settings remain fixed.
The student's Qwen3-1.7B initialization and training method are unchanged.

The adaptation configuration is:

| Setting | First candidate |
| --- | --- |
| LoRA rank / alpha / dropout | 32 / 64 / 0 |
| Target module suffixes | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, `down_proj` |
| Optimization target | Cross-entropy on canonical response tokens and EOS; prompt and padding labels masked |
| Optimizer | AdamW, learning rate 0.0001, weight decay 0.01 |
| Initialization and shuffle seed | 271828 |
| Later full-fit schedule | 16 warmup steps (`ceil(0.03 × 512)`), then cosine decay |
| Gradient clipping | Global norm 1.0 |
| Initial fit data | 8,192 examples, complete sibling pairs |
| Initial fit bound | Four epochs, global batch 64: exactly 512 optimizer updates |
| Development data | 512 distinct examples, complete sibling pairs |
| Later teacher development/readiness generation budget | 256 new tokens; greedy, non-thinking, unchanged seeds |
| Merged development evaluation checkpoints | Optimizer steps 128, 256, 384 and 512 |

Record initialization/shuffle seeds, the exact optimizer defaults, token counts,
microbatch/accumulation policy and package versions before submission. Require an
exact module inventory and trainable-parameter count; reject missing, unexpected
or duplicated target-module matches. No quantized base loading is proposed.
Teacher adaptation may use activation checkpointing for memory, with the fixed
implementation and actual setting recorded.

Existing production model loading prohibits PEFT wrappers. The adaptation
worker is a distinct reviewed path with an isolated derived environment and a
pinned PEFT dependency. It must not weaken that existing production guard or
install into the active fixed SDSC environment. The deployed teacher remains
an ordinary frozen dense model after merging; no adapter wrapper is silently
passed to student training or readiness scoring.

## Data isolation and deterministic construction

Use the existing ProofGraph generator and original range-based task configuration:
depth 2–4, chain/branch/converging-DAG structures, 4–16 distractors, balanced
signed sibling pairs, and the original proof-multiplicity policy. Keep the
canonical proof renderer and full-graph verifier unchanged. Regeneration uses
the original ranges, not only the resolved per-example scalar metadata, because
the two consume different RNG draws.

Reserve **raw pair seeds starting at 70,000,042 for teacher fit** and
**80,000,042 for teacher development**. These are eight-digit seeds, preserving
the generator's eight-digit distractor-symbol width. The two are distinct
teacher-data roles with separate manifests; do not accidentally add a second
10-million split offset by treating the development base as a normal validation
base. Persist the exact ordered seed and example lists, configuration, generator
identity and content hashes. Complete both siblings before any subset boundary.

Before a charged job, verify disjointness of fit/development and **all 144,000
examples in the original seven-split family** using raw seeds, example IDs,
pair-group IDs and canonical semantic identities. Bind the original family
manifest and its actual split-file hashes. This includes original teacher-store
prompts, the student training population, original validation, held-out tests
and circuit cohorts. Abort on any collision or unverifiable family identity.
This comparison may inspect dataset identities; it must not inspect new model
outputs on a formal evaluation cohort.

The pre-job CPU audit for the initial **256 fit / 32 development** population
passed: it regenerated all seven original splits (144,000 examples), matched
every split's actual published byte hash and size, and found zero overlaps in
all four dimensions. Evidence is
`.sdsc/diagnostics/teacher-adaptation-v1/isolation-cpu.json`, SHA-256
`2263ffc5dc439187d384950c1a96603855a10a100862f58411eaaae13ec71093`.
This establishes this preflight population's isolation. The GPU worker repeats
the audit against actual persistent files before training. A later expanded fit
must validate its complete larger population separately.

All fit and development canonical targets must verify, every graph must have
exactly one derivable query polarity, and labels/IDs must agree. Record token
lengths with the pinned tokenizer and reject silent truncation. Do not omit hard
examples, a polarity, long proofs or a topology based on teacher performance.
Training may consume only the fit manifest; development targets must never
enter an optimizer batch. Record and check this boundary in the worker.

The tokenizer audit for the initial 256/32 preflight finds maximum fit lengths
of 1,190 prefix / 163 response including EOS / 1,353 total tokens, and maximum
development lengths of 1,170 / 163 / 1,292. The future 8,192-example fit has
maximum prefix 1,256 and total 1,419 tokens; 60 prefixes exceed the original
1,246-token teacher-store bound, while every full input remains below 1,536.
That later teacher-fit implementation must explicitly use a separate prefix
bound of 1,280 and keep all 8,192 examples. Its 512 development examples have
maximum prefix 1,196 and total 1,359 tokens. Do not filter those 60 fit examples
or change the original formal teacher-store prefix bound. These measured
envelopes are recorded in
`.sdsc/diagnostics/teacher-adaptation-v1/full-fit-token-envelope.json`.

The 256 preflight training examples are the first 128 complete pairs of the
future 8,192-example fit population. Its 32 development examples are the first
16 complete pairs of the separate 512-example development population. A future
32,768-example fit expansion may use a deterministic extension of the fit
namespace, but is **not** a silent expansion of this first candidate. It requires
an explicit successor configuration, bounded resource plan and review.

## First GPU task: eight-step operational preflight

The first exposed SDSC task is `qwen3-v2-teacher-adapt`, initially accepting only
the preflight scope. The Quest CLI uses
`tools/sdsc_teacher_adapt_job.sh` and `tools/sdsc_teacher_adapt.py`, with the
same release/submission/result/runtime identity boundary as the existing probe.
The report is `teacher-adapt.json`.

Resources: **one H100, 24 CPUs, 192 GiB, at most 60 minutes**, account `nwu181`,
partition `nairr-gpu-shared`, QoS `nairr-gpu-shared-normal`. The source and model
are staged on actual node-local scratch, and required outputs are atomically
published to verified persistent project storage. HOME is source/control only.
Recheck the existing shared SSH master, exact runtime, mounts and source hashes;
use fresh run/intent/job IDs and the usual matching dry-run before submission.

Run exactly **eight optimizer updates at global batch 32**, consuming the
256-example preflight fit set once, with initialization/shuffle seed **271828**,
constant learning rate **0.0001**, and no warmup or scheduler. This short schedule
is distinct from the later 512-step cosine fit. The training entrypoint records loss,
finite-gradient checks, clipping, actual trainable modules, optimizer-step count,
nonzero adapter updates, throughput and peak memory. This is a standalone
preflight; the later 8,192-example fit starts from the pinned base and a fresh
specified adapter initialization, not an undocumented warm start.

Merge, save and reload the candidate dense model. Verify complete checkpoint
files by content hashes and validate the reloaded model, tokenizer and output
identity. Measure actual merged-model behavior on the 32 development examples;
this already-submitted preflight retains its original **128-token** greedy
development generation budget.
these results are diagnostic and are not expected to satisfy the final gates
after only eight updates. Adapter/merged comparison may detect export defects,
but an unmerged model's score cannot stand in for the merged model's score.

Preflight PASS requires finite execution, a nonzero update, exact step/data
accounting, a valid reloadable merge and verified persistence. Its report must
say that execution success **does not mean teacher readiness**. Keep
`accepted_science`, `full_teacher_ready`, `readiness_artifact_produced` and
`g0_passed` false. Distinguish teacher adaptation from student experiment
training; never imply the student pipeline has started. Fetch only small
reports/logs; dense weights remain on persistent SDSC storage.

Eight steps do not validate long-run convergence or distributed training. Use
measured throughput and memory to choose the later bounded fit's walltime and
allocation. A four-H100 adaptation path requires its own reviewed batch/loss,
checkpoint and execution tests before use; the student's FSDP/Blackwell
certificate does not certify an 8B LoRA or DDP trainer.

## Development selection and immutable acceptance gates

The later fit is a separate task version/release/intent with the frozen
8,192-example/four-epoch/global-64 envelope above. Evaluate only merged dense
checkpoints at **steps 128, 256, 384 and 512**, using all **512 development
examples** and the accepted 256-token teacher-only generation budget. Select
the **first scheduled checkpoint passing all eight metrics** on that full
development population. Preserve every scheduled report and its exact checkpoint
identity; do not pick a later checkpoint from formal or supplemental outcomes.
If none passes, record the failure and design a reviewed successor using
development evidence. Neither a loss decrease nor 32/32 training-probe coverage
is sufficient. This proposal retains the 512-update training bound; changes to
that bound or a new stopping rule require an explicit successor plan.

The 128-token legacy teacher-readiness budget comes from
`configs/config.yaml`'s `trainer.max_completion_length`, consumed by
`cli/evaluate_teacher_readiness.py`. It is included in the accepted resolved
science configuration. The original Qwen3-v2 preregistration lists the eight
thresholds without a separate readiness generation cap; that does not make the
accepted 128-token runtime setting freely mutable. The teacher store separately
uses 256 new tokens.

The accepted adapted-teacher budget is **256 new tokens for full development,
supplemental confirmation and formal teacher readiness**, while student
rollout/evaluation settings remain unchanged at 128. Introduce a separately
bound teacher-readiness setting; do not globally alter the trainer setting.
This gives canonical responses room to complete: 64/512 development canonical
renderings exceed 128 tokens, with a maximum of 162 before EOS. In the exposed
original first 128 validation rows, only 6 exceed 128, so a verbatim canonical
oracle truncated at 128 achieves 122/128 (95.31%) there. These are serialization
diagnostics, not a proof that every model must fail: exact proof compares parsed
steps, and answer accuracy requires a parse-valid correct answer rather than
byte equality with the canonical rendering. The cap change is a declared
evaluation-budget intervention, **not a demonstrated implementation bug**.

Freeze this decision before full-fit model selection and any supplemental
inference. Bind the cap in the new protocol, resolved configuration and every
development/confirmation/readiness artifact. Preserve all old 128-token reports.
Optional paired 128-token diagnostics must be labeled with their actual budget;
a 256-token PASS must never be called an original 128-token-budget PASS. An
unadapted-base development evaluation at the same 256-token budget can separate
budget effects from weight-adaptation effects, without replacing any gate.

Preserve the original eight numeric readiness thresholds:

| Metric | Required value |
| --- | ---: |
| Parser-gated answer accuracy | at least 0.90 |
| Exact canonical-proof accuracy | at least 0.85 |
| First-rule complete-target top-1 accuracy | at least 0.80 |
| Intermediate-conclusion complete-target top-1 accuracy | at least 0.80 |
| Minimum retained top-128 mass | at least 0.90 |
| Whole-target top-128 coverage | at least 0.90 |
| Corrupted-prefix recovery accuracy | at least 0.70 |
| Minimum causal log-probability shift | at least 0.0, with every structural/strict-positive validity check satisfied |

The exact-proof gate compares the parsed proof steps with the original canonical
proof, not just any valid proof. Answer accuracy still requires a parse-valid
response. Top-1 and coverage require all target tokens; retained mass uses the
minimum, not an average. Prefix scoring covers both sides of the frozen
active-support swaps and retains original probe selection and top-k width.
The scorer's `causal_shift_valid` requires a strictly positive shift for every
scored side; a zero shift must not pass via the numeric lower bound alone.

The separately identified multi-token counterfactual scoring defect must receive
independent implementation/regression review before these measurements. A target
sequence's conditional probability under the opposite context must use that
same target's autoregressive history. Correcting this implementation preserves
the intended metric and thresholds; it does not license old affected causal
shift numbers as new evidence. It does not change the retained v7 generation
failure or its original 7/32 coverage.

After selecting and freezing a merged candidate using teacher-development data,
require the unchanged original 32-prompt training probe: eight candidates per
prompt and 32/32 covered. Preserve request seed 31415, identity-derived
candidate seeds, temperature 0.7, top-p 0.8, top-k 20, min-p 0 and the 256-token
completion limit. This necessary gate cannot replace the full store or readiness.

Only then run one supplemental confirmation on the already frozen original
validation rows **[128:256]**. Its ordered-example hash remains
`532b11ac85fad0b35be1e253c2d30a8ca838ea8849adf77503a43b6433236073`; the
validation-file hash remains
`8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3`.
Under the accepted adapted-teacher amendment, use greedy generation with the
explicit **256-token** completion budget, seed 42 plus row index, original
prefix construction/scoring and all eight thresholds. Do not
expose this cohort during fitting or checkpoint selection. If it fails, preserve
the failure; its outcomes are then exposed and cannot be reused as an independent
confirmation of an adaptively changed teacher.

The original **first 128 validation examples** remain the mandatory formal
readiness population, with their exposure history disclosed. The accepted
adapted-teacher formal evaluator uses those same rows, greedy generation and
all original thresholds, with its explicitly amended **256-token** budget.
Do not replace these rows with the supplemental cohort or relabel this as a
pass of the historical 128-token-budget protocol. Formal adapted-teacher
readiness and a fresh complete **256×8 teacher store** with coverage of every
prompt have now passed independent acceptance. The downstream matching
preflight remains required. The later pilot requires its own full 4,096-prompt store.

## Checkpoint identity, independent review and continuation

Preserve the pinned base snapshot. A merged checkpoint must have a separate
manifest binding every dense weight shard, tokenizer/config identity, adaptation
configuration, training-data hashes, base revision and actual training provenance.
The base commit remains base-model provenance; it is not the identity of the
adapted weights. Every readiness/store/scorer/cache and downstream scientific
binding must identify the adapted checkpoint content. Directory names or a
copied `_commit_hash` are insufficient. Never overwrite the original HF cache
or pretend that a merged model is still the unmodified base teacher.

Before formal use, create a new scientific amendment/protocol defining the
adapted teacher and its weight bindings, without rewriting historical accepted
protocols. Commit the complete implementation as proposed, obtain independent
review of the data boundary, training/export/inference behavior and evidence,
then make the distinct review-only acceptance commit required by the project.
Actual checkpoint evidence and a reviewer judgment are required; no tool may
promote its own boolean into independent acceptance.

Migrate the old v3-pinned SDSC adapters coherently and require a matching new
student/teacher preflight before calibration. Recompute affected safety identities
and make no claim to the old Blackwell execution-class certificate. Preserve all
stopped plans and claims; successor work uses new immutable plans and identities.
Once this teacher's required evidence is accepted, the already authorized gated
student path may proceed. Full factorial/Gemma remain outside this proposal.

Each finite job and Quest monitor must report its actual accounting state,
artifact checks and scientific outcome. Lost receipts require reconciliation;
SSH loss or changed controls stop new operations. Continue authorized work from
recorded evidence, never by re-arming failed flows or resubmitting an unknown
intent. No manual per-stage approval is added by this proposal.

## Qualification execution and bounded independent audit

The separate `qwen3-v2-teacher-qualify` task consumes a verified successful full
fit job through `--teacher-job-id`. It has no caller-selected checkpoint or path:
the control plane derives the first scheduled full-development PASS. It requires
four H100s, 24 requested CPUs, a 192-GiB workload budget and at most two hours,
with account `nwu181`, partition `nairr-gpu` and QoS `nairr-gpu-normal`.
Use the existing discovered PEFT runtime. A genuine accepted protocol bundle and
matching snapshot must pass verification before any permanent claim or `sbatch`.
The GPU worker repeats verification after staging and before inference.

The experiment reservation is independent of checkpoint/protocol hashes, so
changing a checkpoint or protocol name cannot repeat the held-out confirmation.
Each of the four inference stages gets an exclusive durable claim before its
first forward and a sealed outcome afterwards. A failed, missing or unknown
outcome prevents progression. Retain claims after failure; do not delete them
to permit another submission. The worker reports scientific evidence separately
from the independent teacher-acceptance decision.

Default `tools/sdsc fetch JOB_ID` retains its 1-MiB per-file and 8-MiB total
limits. Dense/adapter weights stay on verified project storage. The independent
CPU auditor reads every published scientific file on SDSC, bounded to 256 MiB
physically and 128 MiB of unique content, with logs separately bounded to64 MiB.
The two limits accommodate byte-identical probe manifests independently saved
by all four fit checkpoints; all copies are actually hashed. No evidence is
truncated or omitted to fit the bound. These are transport/audit bounds, not
scientific threshold changes.

After accounting reports COMPLETED/0:0 and publication is verified, submit a
separate bounded audit allocation through the checked existing SSH master.
On its compute node, restore the same bound genuine Git bundle into a fresh,
verified node-local checkout and invoke the verified runtime with CUDA hidden
and OMP/MKL/OpenBLAS/NumExpr threads fixed to one:

```text
VERIFIED_PYTHON -I -B RESTORED_SCIENCE/tools/sdsc_teacher_qualification_audit.py
  --science-root RESTORED_SCIENCE
  --expected-head RECEIPT_SCIENCE_HEAD
  --result-root ACTUAL_QUALIFICATION_RESULT_DIRECTORY
  --publication-sha256 SHA256_OF_FETCHED_PUBLICATION_RECEIPT
  --job-id QUALIFICATION_JOB_ID
  --claims-root /home/zgao12/quest-runs/OPD/teacher-qualification-claims
  --output NEW_SMALL_AUDIT_JSON_OUTSIDE_RESULT_DIRECTORY
```

These are placeholders to fill from actual receipts, not a shell command to run
unchanged. The auditor performs no model inference, weight download or Slurm
operation. Fetch its bounded JSON after actual hash verification. A distinct
reviewer must inspect this replay, original accounting/publication and exposure
history before issuing an attestation bound to the actual implementation and
acceptance commits and evidence hash. Only the independently attested artifact
may set `formal_teacher_accepted=true`; the producer and audit report cannot.

The audit is computationally substantial despite requiring no GPU inference:
the frozen implementation repeatedly reconstructs the full tokenizer
fingerprint, approximately 2,590 times across its two complete evidence replays.
Two measurements with the actual tokenizer on Quest took 0.766 and 0.888
seconds per fingerprint. This suggests roughly 33–39 minutes of that CPU work
alone on Quest; it is not a measured SDSC runtime. Do not run this audit on a
login node. The [Expanse user guide](https://www.sdsc.edu/systems/expanse/user_guide.html)
requires computationally intensive work to use batch nodes.

At 2026-09-28T06:42Z, account `nwu181` exposed only the three NAIRR GPU QoS
associations; `nairr-gpu-shared-normal` had `MinTRES=cpu=1,gres/gpu=1`.
A zero-GPU, one-CPU audit test was rejected with `QOSMinGRES`. A subsequent
`sbatch --test-only` accepted **1 H100 / 1 CPU / 16 GiB / 90 minutes** on
`nairr-gpu-shared` / `nairr-gpu-shared-normal`. Its displayed test-only number
is not a submitted job. This is the smallest discovered allocation for this
account; the GPU remains hidden during the CPU-only audit. Wait for the
four-GPU qualification job to finish before submitting it, and recheck actual
node-local workspace, input/output mounts, runtime and memory use. Keep its
fresh submission intent, real job ID, terminal accounting and published audit
hash separate from the scientific qualification job it audits. Never retry an
unknown submission or overwrite an earlier audit attempt.

The actual independent audit 54497294 completed in **41m18s**, with about
**33m49s child CPU** and **1.2 GiB** peak memory. All job/batch/extern records
are COMPLETED/0:0. Its 558,712-byte result was hash-verified before local use;
physical SHA is
`bd558d9aaac3669f4185ebaca6ebc67b69a14b75f24f795c7d52baed9fdd9dde`.
The long quiet intervals were progressing CPU work, not a deadlock.

The audit exposed two operational defects, repaired without changing the
47 accepted scientific files or retrying a submission. Local submission preview
had resolved SDSC `/home` paths against Quest's different filesystem; it now
checks their remote spelling while retaining remote filesystem validation.
SDSC also returned an empty archived `sacct Comment`. A separate read-only
observer validates this case against the actual pinned live `scontrol` proof
(SHA `eda6de17c291b85b8d7e30becc31b6b624528fae9b6d4bd2d7bf00f27f4907d0`),
original plan and real submission receipt. It preserves the raw empty value,
rejects conflicting nonempty comments, and still requires exact job/resources,
terminal accounting and publication hashes. Both 12-case regression suites and
independent reviews pass. For this completed audit use `batch_audit_observe.py`
for read-only status; do not use the original strict-Comment helper or overwrite
already fetched evidence. These one-shot helpers and their independent review
records remain under `.sdsc/diagnostics/teacher-adaptation-v1/`.
