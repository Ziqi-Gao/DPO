# OPD current handoff

Last updated: 2026-10-02 UTC. **The user selected keeping1.7B and preparing one
common initial model for all methods, and authorized repair and submission.**
The successor is `prereg/amendments/qwen3_student_preparation_v1.json`;
see `docs/refactor/sdsc_student_preparation_20261001.md`. It fixes constantLR5e-5
and prepares fresh native weights on2048 independent symbolic-canonical examples,
with512 independent development examples and a prospective checkpoint rule.
No teacher-generated targets are required. Preserve the original failed initial
and every original scientific threshold. Implementation **625c65d** and separate
non-author protocol/worker/transport/auditor/provenance reviews are complete.
The focused CPU suites pass **149 tests**, including2048 actual-tokenizer response
replays and9 corrupted-evidence rejections. Distinct acceptance **ba8d357** records
implementation readiness, not a model/G0 result. Preflight **54562489 PASSED**:
all accounting rows **COMPLETED/0:0**, elapsed14m10s, queue empty and all execution
validators accepted at05:46:46Z. It completed4 W2/global64 updates/218,500 tokens,
actual full-state restoration and exact saved-master/native-BF16 reload parity.
Peak own-job memory90.92GiB left293.08GiB headroom, with noOOM. All8 training-only
responses independently replayed; no development/initial/G0 acceptance was made.
Required30.02GB of dense/full-state artifacts were persistently read-back hashed.
Small results: `.sdsc/fetched/54562489/fetch-oz9q5io5/`; receipt SHA
`630e210be904f41c4bcfe26f58a3569348836925dd8b8e0d5506fccb25526eae`.

Matching complete preparation **54562507 PASSED**: fresh accounting at
**2026-10-02T22:55:39Z** confirms job/batch/extern **COMPLETED/0:0**, elapsed
**42m20s**, with an empty queue. It completed all32 updates/1,856,564 input tokens
and four complete512-example development evaluations, using
**2H100/24CPU/384GiB/4h** and fresh original native weights. Development answer
counts at steps4/8/16/32 are **175/366/462/503 of512**; full-proof counts are
**175/365/462/503** and format counts **507/508/509/512**. The frozen earliest
52-through307 selection rule selects **step4**, answer/proof175/512 (34.18%).
Freeze `checkpoints/step-00000004.pt`, SHA
`23854ce0d6db4cb01cae898beccf1ac7be7613e5ee01ff5b1626daafb8349d31`,
8,127,108,889 bytes, under the fit intent's persistent result directory. Do not
replace it with later stronger checkpoints after formal exposure. This is
preparation selection only; all formal-initial/student/G0/pilot/factorial and
execution-class acceptance flags remain false.

All checkpoints passed exact saved-master reload and native-BF16 export parity;
step4 full-state restoration passed on both ranks. These observations are not
complete G0 resume or cross-run bitwise-reproducibility evidence: logs retain
CuBLAS/attention determinism warnings. Peak own-job memory after publication was
**111.624GiB**, leaving272.376GiB headroom, with noOOM/failcnt event. Required
54,400,439,960 bytes of dense/full-state artifacts were persistently read-back
hashed. Small results: `.sdsc/fetched/54562507/fetch-hmcmblh_/`; publication SHA
`98aff58e27bc88653453047d7610d5c1b2b648482b1ad72616f3628c8613b1bc`.
The independent raw audit replayed all512 prompts/2048 responses, verifier
traces, four summaries and the exact selection; GPU arithmetic was not
independently recomputed. Audit:
`.sdsc/diagnostics/student-preparation-v1/fit-actual-independent-audit-20261002.json`,
SHA `411fd1f2bb4c0de4fcf30f37c3bc701b25c2edad38da651182bc0012a014cdaf`.

The fit intent is
`524f6a6e8f1541502f315b820880ea4c`; preflight intent is
`39af4ea3f267d92dca306664e6d2f432`. Their immutable plans and submission receipts
are under `.sdsc/student-prepare/`; do not duplicate or re-arm either. No observer
or automatic continuation is active. Deployed release
`20261001T052044Z-dffd4aa0a3f4-ca7b712b` uses genuine science HEADba8d357; later
handoff-only commits do not change that deployment. Controls remain pinned.
Bounded reviews, runtime evidence and independent raw audit are in
`.sdsc/diagnostics/student-preparation-v1/`.

The user authorized continuing the gated workflow. The additive qualification
candidate is `prereg/amendments/qwen3_student_qualification_v1.json`, described
in `docs/refactor/sdsc_student_qualification_20261002.md`. Non-author reviews of
worker, transport, auditor and protocol found no blocker; all159 focused CPU
checks and Ruff pass, including the actual node-to-fresh-worker input seam.
Implementation **48951c1** has distinct non-author exact-commit acceptance;
this review-only commit accepts the qualification implementation, not a GPU or
model result. The job is not yet submitted. Preserve all110 named parent
scientific files. The worker restores
the frozen step4 FP32 masters exactly before native-BF16 inference and retains
all896 original responses: validation128 plus anti-shortcut IID128 and640
transformations. Base requires13 original answer-correct results; anti uses
complete-proof reward with unchanged .10/.08/.05 floors and signed gap<=.05.
Actual pinned tokenization requires2244 auxiliary inference tokens, matching the
original untruncated anti-shortcut evaluator; training1536 is unchanged.
Resources are1H100/24CPU/192GiB/2h; no teacher data or new training is involved.

Next: deploy the accepted implementation and submit this one
qualification; only after independently verified success continue remaining
calibration/cohort/circuit gates. Initial qualification and downstream
calibration/G0 remain unexecuted; no formal OPD/RL comparison is running.
Successor adapters must explicitly load the prepared student and accepted dense
teacher: existing `build_rollout_bank.py`, `evaluate_teacher_readiness.py` and
`score_teacher.py` load native HF models; existing `train.py` loads BF16 before
the FP32 prepared checkpoint and would round masters. Reuse reviewed exact-FP32
load semantics in additive adapters, regenerate initial-dependent artifacts,
and retain original validation128/anti-shortcut/cohort/circuit/calibration gates.
Later OPD/GRPO configurations still inherit5e-4; they need an explicit prospective
LR repair rather than assuming preparation's5e-5 propagates. Preserve method
semantics: OPD trainer completion128 differs from GRPO supervision completion256,
eight generations and its own batch/accumulation settings. Do not apply the
preparation/G0 global64 or length contract indiscriminately to every method.

CPU tests cover genuine two-process global64 loss/update equivalence and actual
full-state save/restore with identical next update. Exact pinned tokenization
finds1,856,564 fit tokens, max1417 per training sequence. Preserve six longer
development prefixes by declaring preparation-only1600-token inference before
GPU execution; FSDP training and downstream limits remain1536. Provenance export
now bounds128 unpublished commits instead of64, retaining genuine history,
per-tree audits and unchanged byte limits; no public ref or history is rewritten.

Both diagnostics completed and were independently audited: original student
fails its base criterion at0/128 versus13 required. LR job54560292 supports
lower5e-5 updates but cannot repair the fixed original initial. Native check
54560398 completed0:0 in12m38s; all128 responses were replayed. Both finite
observers exited; do not re-arm them or repeat their completed jobs.

Adapted teacher fit 54494742, qualification
54496291 and independent CPU audit 54497294 remain accepted under the reviewed
teacher-only 256-token protocol. V5 preflight 54533796 passed; calibration
54533934 completed 33 updates and saved step-20/33 checkpoints but failed the
post-training 192-GiB memory-headroom gate. Its exact peak was not recorded;
there is no demonstrated OOM. Preserve its outputs and stopped flow.

The user authorized repair and resubmission with larger resources. V6 uses
**2 H100 / 24 CPU / 384 GiB**, preserving scientific settings and the strict
32-GiB/20% headroom rule while retaining actual memory measurements on failure.
Implementation **9a19667** and distinct independent acceptance **6c04f80** are
complete; the final CPU suite passed **447 tests**. Matching preflight
**54547547** and automatically submitted calibration **54548846** both passed:
fresh SSH accounting confirms **COMPLETED / 0:0**, respectively **16m38s** and
**29m31s**, with hash-verified results. Calibration completed **33 updates /
1,961,368 nonpadding tokens**, preserving checkpoints on project storage. Its
actual cgroup peak was **190.21 GiB**, leaving **193.79 GiB** headroom under the
384-GiB limit; final validation did not raise the post-training peak. Step-20
answer/proof/format scores remain **0**; execution/calibration PASS is not
student-quality acceptance or complete G0/pilot evidence.

The finite Quest supervisor reached **`calibration_complete`** at
**2026-09-30T18:50:07Z** (13:50 CDT), fetched 13 small files and exited. It
submitted exactly one calibration and no G0/pilot; preserve its plan/claims and
do not re-arm it. State is under
`.sdsc/supervision/student54547547-to-calibration-v6/`. Full G0 still requires
reviewed adapter migration and actual scientific evidence. Central
ServerScheduler observations below remain dated 2026-09-10 and were not
reverified from Quest.

The user explicitly authorizes continued debugging and necessary fresh runs
until model acceptance, without lowering scientific thresholds. Both current
inference diagnostics are now **COMPLETED / 0:0 and fully independently audited**;
**the student remains unaccepted**. Their finite read-only Quest observers have
fetched small results and exited. Do not restart them or resubmit their intents.
All 49 accepted student science files and the nested 47-file teacher contract
remain unchanged. Preserve all original checkpoints and unsuccessful evidence.

Quality diagnosis **54558773** completed in **55m47s**, confirmed at
**2026-09-30T23:14:15Z**; job, batch and extern all exited zero. It used one H100,
24 CPU and 192 GiB. Peak cgroup memory was **74.5391 GiB**, headroom **117.4609
GiB**, with no OOM/failcnt event. Its run is
`20260930T220721Z-0a50670bd73e-43958896`, intent
`728de9ca98016805cf6474adb1ad57b8`, plan SHA
`ca54bab66edc0c8a5cc36adfd65f1b48abdcb4021b93666283b8a263c76efa12`.
Actual results: `.sdsc/fetched/54558773/fetch-nlvs9wo6/`; receipt SHA
`7d0a32ae0be1cb4bd1252d31dcc06c39204ea82bd83330d0cafdf3835990e4d8`.
The independent full-token audit is
`.sdsc/diagnostics/student-quality-v2/full-token-audit.json`, SHA
`27f5f9820a028cbe8959006f754fdd0f9f63ebeb246a3e82bb9fd5d1f8e79935`.
It verified 160 prompt encodings, all 960 response decodings and original
verifier traces, all 12 arm summaries, prefix pairing and publication hashes.
The auditor binds the actual deployed scientific source before imports; its
source-binding addition and original logic received separate independent review.
NLL values are producer-report-bound; only token denominators were independently
recomputed, not model logits or GPU arithmetic.

At both 128/256 caps, initial, step20 and step33 have **zero strict answer/proof
successes** in validation128 and training32. At 256, initial format validity is
13/128 and 2/32. Every step20 response repeats `<proof>`/digits and terminates at
the cap; every step33 response repeats `R01:`. Neither trained checkpoint emits
any valid closing proof/answer structure. Step33 has 55/160 EOS terminations,
but those are malformed too. All 160 canonical targets including EOS fit256.
Validation canonical NLL/token accuracy is **1.10855/89.02% → 2.20222/31.58% →
1.05338/67.58%**; training is **1.00384/89.13% → 2.19470/31.88% →
1.04485/67.47%**. Training has caused repeated-output collapse; longer generation
or recovered teacher-forced NLL does not establish sequence quality.

The preceding diagnosis **54557365** failed after26m05s because its FP32 config
violated the actual Qwen3 BF16 loader contract, before trained-checkpoint
inference. Preserve its intent `d47f1122872cad16aeb8e772f2e37ea2`, run
`20260930T211723Z-49907235ba25-c8e1e004`, and partial320-response results under
that intent's `fetch-186dm58f/`. Its receipt SHA is
`c18b0f1f2ce7b5685e644b743f7ec6b87cad2b457826f7626c544c54fcf51c37`;
missing old NLL remains unknown. Implementation **4c6bdb6** repaired the loader,
partial evidence retention and bounded fetch; **114 CPU tests** plus independent
worker/node/controller/deployment reviews preceded54558773. The old failure was
not OOM and is not the reason for the newly observed model degeneration.
See `docs/refactor/sdsc_student_quality_diagnosis_20260930.md`.

The sole training-only instruction screen **54559253** completed in **5m28s**,
confirmed at **2026-09-30T23:10:48Z**, with all accounting exits zero. It used one
H100,8 CPU,64 GiB; run `20260930T225419Z-8c5ed83b5110-73116a7c`, intent
`e8573086e7b073ec71e46541ff453c2a`, plan SHA
`8936a66bfc2e27de820ab8e4053f12aaac0dc6ab58ac8acc2de6a5cec9155415`.
Results are `.sdsc/fetched/54559253/fetch-empc2123/`, receipt SHA
`3e5595d9c4cdba9aa82d0c0a302441283ff89d7af781f7a9a3f9901ec02d5f01`.
The pre-load check found every one of311 original HF/BF16 state tensors exactly
equal to the initial checkpoint; embedding tying and before/after parameter
hashes remain unchanged. This excludes the hypothesized initial-export mutation
in the pinned runtime; it is not an independent new Hub download comparison.
Both original and candidate instructions score **0/32 strict answer and proof**.
Format validity improves **1/32→14/32**, but the candidate has18 syntax failures,
9 wrong-antecedent proofs,3 invalid-citation proofs and2 incorrect conclusions.
Its predeclared4/32 futility screen is **false**. **Stop promotion of this
candidate**: no new teacher/prompt successor or validation-wording search.

The independent screen audit,
`.sdsc/diagnostics/student-quality-v2/instruction54559253-independent-audit.json`
(SHA `7b43945f01f39351db710269bb20c328dc950b3c39276404993980679a663a9d`),
replayed all32 training prompts/64 responses, complete parser/verifier traces and
both arm metrics. It binds the old first32 training population, genuine parent
49-file science map and31 imported scientific files. The auditor was reviewed
before reading real results;33 fixtures and a separate real-worker compatibility
fixture passed. All scientific acceptance flags remain false. The screen used
native BF16 without explicit autocast, whereas54558773 uses BF16 autocast; the
paired baseline prevents interpreting that context difference as prompt effect.
Frozen design: `docs/refactor/sdsc_student_instruction_screen_20260930.md`.

Preserve the superseded **unsubmitted** instruction draft
`6b36e12f9a946f983581e4302282744b` and its faulty dry-run: `squeue %b=N/A`
incorrectly counted a known1-GPU job aszero. Repair **addb2a3** now cross-checks
real `scontrol` ReqTRES/AllocTRES and per-unit GPU fields, failing closed on
unknown facts.118 CPU tests and actual deployment review passed before54559253.
Execution identities bind controls; an independent permanent scientific claim
plus atomic legacy reservation prevents duplicates across control revisions.
Never delete these claims or blindly retry. Both observer exit records and all
reviews/submission evidence remain under `.sdsc/diagnostics/student-quality-v2/`
and `.sdsc/diagnostics/student-instruction-v1/`.

Independent code/metrics review has not established a loss shift, repeated
optimizer step or W2 gradient-scaling bug. The accepted configuration really is
full-parameter AdamW **5e-4**, betas(.9,.95),eps1e-8,weight_decay0,constant
LambdaLR1.0, without warmup/clipping. With1,720,574,976 unique parameters,
lr*sqrt(N)=20.7399; the actual first-update L2=18.7528 is plausible, not by itself
a double-counting error. Logged KL7.82295 is current||initial over one fixed
training prompt's prefix, not a validation or generated-response KL. Different
step losses consume different64-sample windows. The paired diagnosis
below subsequently tests optimization overshoot; it does not establish an
optimizer implementation defect.

The real first64 teacher-data prerequisite is now complete and independently
replayed on Quest. Exactly64 candidate-0000 records,53,440 prompt tokens and5,896
response tokens match the historical59,336-token first window. All64 proofs,
actual state-source/collator spans, shifted sequence CE and analytic synthetic
CPU gradients agree; five corruption cases reject. Report SHA
`86e5cca158476439cc9f73881bd7e73869b4a6837d1089284369156c79f9c933`, capture SHA
`aae6b2fd3fae601f265998416a9d2de8a7340c8672e1bcf2ef026508ebdff8b5` and independent
review SHA `2fb7197c06f80294745372ee411f87a70b90a7a29cb5f6485a7ef03633e4d84c`
are under `.sdsc/diagnostics/student-quality-v2/teacher-batch-audit-*`.
The source ledger was read/hash-checked once remotely; only1,012,177 bytes were
transferred. No label/mask/shift defect was found. Synthetic CPU logits do not
establish real1.7B/FSDP optimization or any scientific acceptance.

The separate train-only LR diagnosis is implemented in
`tools/sdsc_student_lr{,_job,_probe}.py`, with frozen scope and command sequence
in `docs/refactor/sdsc_student_lr_diagnosis_20260930.md`. It compares5e-4/5e-5,
four successive original global64 windows per arm, original initial/v7/teacher
store, fresh optimizer/RNG/cursors, W2/FULL_SHARD and no validation inputs.
Resources are **2 H100 /24 CPU /384 GiB /1 hour**. It measures fixed-batch
teacher CE, actual first-step AdamW math/cadence, standalone exported-model
responses and root/export logits; neither LR improvement nor diagnostic PASS
is model acceptance. Both final model-only checkpoints must be read-back
hashed on project storage and never fetched. New control code explicitly
rejects observed own-job OOM/allocation-failure counters; the frozen memory
helper only collected them. Preserve old job evidence and all old controls.

The three focused CPU suites pass **153 tests** (40 worker,85 transport,28 node),
including real tiny-Qwen loss, original trainer updates, actual AdamW hooks,
node-to-worker admission, failure publication and OOM negative fixtures.
Implementation **5fb83bb** received independent worker review
`6b253737f232641076d6bc180bd5d9cc1214379bc913e50db4fd3da7c426c1c4`, control/node
review `adeb7e2338e32a12f927c8285e8a29d3f0a4657548e2a327933acac6119e0b6d`, and actual
deployment review `2602e77191e97b3b646712808acaff39d92425b84aa7e90466c7b51ae857c267`.
All accepted only this diagnosis, not scientific or model success. Runtime
Python3.12.13 and all19 fixed package versions were freshly verified.

The first LR job, **54560005**, was acknowledged at **2026-10-01T00:04:57Z**
and is **FAILED /1:0**, confirmed at **00:10:36Z** after130 seconds. Job and
batch failed; extern completed0:0. It produced no training or generation.
Run `20260930T235843Z-d00427625f69-bde6da4a` contains632 files/7,507,811 source
bytes, source SHA `d00427625f69514edfd1c3530dc9e9f68a634973776d99be0f269cd6f1a81ee0`.
Plan `.sdsc/student-lr/e37a9af579142b7e06f02436e6f4dfe9/plan.json` has SHA
`7bbb9b4a6e308b6727968e3807c659be13f1fbe1cc55dfaedc0c53e161680e9e`.
The remote dry-run verified parent/source/CPU audit/runtime and0+2 requested
GPUs within the four-GPU ceiling. Immutable submit records and both permanent
claims are authoritative; never delete them or blindly submit this diagnosis
again. Results remain under project Lustre `student-lr/<intent>/`.

The hash-verified failure was fetched to `.sdsc/fetched/54560005/fetch-tlgm98oa/`;
publication SHA is
`788c79f0771f6f028851df4995238dd351b03987fa6fac794a26803913a619bd`.
Both ranks rejected `read_accepted_teacher_sft` with
`AdaptedStudentProtocolError: student protocol needs real .git or .opd-git metadata`.
The node restored and verified genuine science, but started the worker from the
hashed deployment snapshot, which intentionally has no Git metadata. The
protocol resolves its default repository from cwd. This is a launch defect,
not an optimizer or model-quality result. The report has no arms, raw artifacts
or checkpoints. The finite read-only observer fetched failure evidence and
exited normally; its started/status/finished records remain under
`.sdsc/diagnostics/student-lr-v1/`. Do not restart it.

All12 original controls are byte-verified under that directory's
`frozen-controls/`; the original remote release and permanent claims remain
unchanged. The current candidate starts the worker in verified `science` and
adds a worker-side cwd guard. A new v2 execution identity must explicitly bind
this exact old failed plan, publication and empty pre-training failure, recheck
remote terminal accounting before claiming/submitting, and retain a permanent
v2 claim. Unknown, active, successful or partially trained old jobs forbid
recovery. The original frozen v1 controller, not a new validator, handles old
identity checks. The completed candidate passes178 focused CPU tests
(107 transport,29 node,42 worker), Ruff and whitespace checks. A real child
observes science cwd, its own session and unchanged CUDA visibility. Genuine
6c04 history/49 science files and the original7,923-byte resolved configuration
reproduce old source-cwd rejection and science-cwd acceptance; a missing teacher
inventory remains rejected. No GPU/model success is inferred. Reproduction SHA
is `b17741ab73400fcfd33c463c9a5b1a481f41c48de03b0733f9cfd0d1a32dbec8` under
`.sdsc/diagnostics/student-lr-v1/cwd_failure_repro_result.json`.
Startup implementation **395124b** received independent review SHA
`479a8f6274f8c4d0557a2b7bc6c9ca61d7225a43a5ff135b4d9f0042f97e8897`.
Its first v2 draft remains **unsubmitted**: intent
`0e4ffde51abef1713fca367b7fc9da1a`, run
`20261001T002828Z-8efeff9d77e3-5cdc400d`, plan SHA
`fd39821b3842d04a7402911c670dc7f695ab6616e5f7e6d0f56f3926f507601e`.
Remote dry-run rejected before claims because old54560005 aged out of the
`squeue --jobs` cache (rc1 Invalid job id), while fresh `sacct` still showed
exact FAILED1:0 evidence. Preserve this draft and
`.sdsc/diagnostics/student-lr-v2/dry-run-query-diagnosis.json`.
Queue repair **502d393** now uses a successful complete-owner queue query,
checks all rows/owners and retains its raw receipt before filtering the one
job; it never suppresses a failed query. Frozen v1 accounting/publication
validators still enforce the original failure identity. V2 status uses the
same query so terminal cache expiry cannot masquerade as a transport outage.
The query uses explicit local/all-states/array expansion and strips inherited
SQUEUE_/SACCT_/SBATCH_ selectors. Its130 transport tests pass, including
real-Slurm rejection guards on Quest; unchanged node/worker retain their71
passing tests. Independent queue review SHA is
`9038413dc5e90c58d6ea10b3d814c4183ede91b0f41c2b02413de3b2e4bc6572`;
actual deployment review SHA is
`2c6a122427f6c2ee5852167e3e764606f856d258cd6cb9bfa93e84f51e3ccec1`.

Exactly one new recovery, **54560292**, was acknowledged at **00:52:21Z** and
completed in **18m18s**, with job/batch/extern **COMPLETED0:0**. Final accounting
and semantic publication validation passed at **01:18:20Z**. It used the same
2 H100/24 CPU/384 GiB/1h envelope.
Plan `.sdsc/student-lr/c6de75d4811b00d8765eede76f4a1ece/plan.json` has SHA
`de917729029b93e1c1097757d3ca01a633e076a53471555d27b4afa3743e2c73`.
Run `20261001T004338Z-669fc878fe08-8727a1d7` contains637 files/7,630,392 source
bytes, code SHA `669fc878fe0838a6bc71023cbe1c2d05c7b958265e607e70212e643f08622d4c`.
Its remote dry-run at00:45:19Z freshly confirmed the exact old failed accounting,
three hashed empty-result files, original source/parent/data audit/runtime and
0+2 GPUs within the ceiling. The new permanent claims and actual receipt are
authoritative; no duplicate or unknown-intent retry is permitted. Results belong
to project Lustre `student-lr/c6de75d4811b00d8765eede76f4a1ece/`.

The finite Quest observer under `.sdsc/diagnostics/student-lr-v2/` fetched
terminal evidence and exited at01:13:07Z; tool session70400 is finished. Its
initial accounting result was conservatively false while `squeue --states=all`
still cached the COMPLETED row. The unchanged frozen controller confirmed
success after that cache cleared; no job was retried. Final fetch is
`.sdsc/fetched/54560292/fetch-5e1mw600/`, publication SHA
`8625d8264f3d9b7fee3081cd63cff013d6cb10ab515d21756699cfe583afeaf7`.
The pre-frozen, independently reviewed raw auditor passed on the actual files:
`actual-independent-audit-54560292.json`, SHA
`c9c5c0c1c3a2ff4e1fec85c41dc603ee639941c7f1270142b6628a4840ca7a45`.
It replayed32 prompts,128 complete responses and16 batch records against original
source/tokenizer/verifier and the first64 CPU data audit. GPU logits/weight and
AdamW measurements remain hash-bound producer evidence, not CPU GPU-math replay.

Both arms start with identical model/RNG and fixed64 teacher CE1.055534. At5e-4,
first-step CE rises to4.957614 and step4 is6.939187; train32 proof/answer remain0
and all32 final generations hit the256 cap malformed. At5e-5, fixed CE falls
0.518697→0.304089→0.109616→0.072471; step4 has **5/32 full proof and answer,
31/32 valid format and32/32 EOS**. Each arm consumes237,754 tokens/256 samples in
exactly4 optimizer calls. First-step AdamW coordinate checks havezero violations,
with8 microsteps per call and a negative gradient/update inner product. Four
root/export logit comparisons per arm are bitwise equal. The matched LR-only
comparison supports optimization overshoot, not a demonstrated optimizer/math
implementation bug. It is training-only evidence; no validation/test selection
or scientific threshold changed. A reviewed LR successor remains required.

Peak job memory was65,098,821,632 bytes (about60.63 GiB), with no own-job OOM or
failcnt event. Both8,127,108,761-byte model-only checkpoints were read-back hashed
on persistent project storage; they were not fetched. Preserve all12 controls,
all original/v2 plans/claims and the two unused drafts. No G0/pilot was started.

The original-native-initial check is implemented under
`tools/sdsc_student_initial{,_job,_probe}.py` and
`docs/refactor/sdsc_student_initial_native_check_20261001.md`, implementation
**ecbbfdf**, and is now submitted exactly once as **54560398**.
It reuses the original scorer on unscreened validation128 without extra autocast,
original initial/config/instruction and256-token cap. Fixed resources are
1 H100/8 CPU/64 GiB/30m. Its single base-gate observation cannot accept model/G0.
The final CPU surface has125 passing cases (86 transport,12 node,27 worker), plus
55 independent raw-core cases and actual historical128 tokenizer re-encoding.
Independent worker/node/controller reviews are under
`.sdsc/diagnostics/student-initial-native-v1/`; controller review SHA is
`33006f01cb55fa9289d3cf8f4f35915a70ff94b899a7d8903c99f4f0b09d861f`.
The controller preserves original49 science files plus9 inference-entry pins,
full family staging, permanent scientific claim, complete-owner queue evidence,
current-job64-GiB/five-phase memory and bounded5-file publication. It handles
terminal queue cache only with matching scontrol/sacct state/exit/owner and
keeps historical requested GPUs separate from currently allocatable GPUs.
Fresh connection and all19 installed runtime versions passed at01:11Z/01:15Z.
Independent actual deployment review accepted only this one check (SHA
`ccd7a2bd5ca0c7815ef48d91d748e5246d9956c5f6919493bed4c60c806b2a6d`).
The frozen raw core and execution-admission wrapper received separate non-author
reviews before new GPU output. Wrapper review SHA is
`ec92010d3c6bfbd7abc78cf43f18d600db356ac9b6f174e179947474e5f1e3ca`;
its49 CPU cases and local actual-plan admission passed before GPU results.
The actual diagnostic also passed complete execution/provenance/raw replay;
that success does not accept its failed model-quality result.

Job54560398 was acknowledged at **01:29:26Z**, RUNNING at **01:29:51Z**, under
intent `2b622073793933ada34054021664378e`. Its plan is
`.sdsc/student-initial/2b622073793933ada34054021664378e/plan.json`, SHA
`2830dda5b6dc3846eb0d73bb92944f4e12ef4d97708530f606b8f1de21f929f5`.
Run `20261001T012217Z-260344e37777-f9b1dcfc` has639 files/7,740,269 source bytes,
code SHA `260344e3777791ea7fae6322506c987e6f4a1dada9448428fb6c148124a38e54`.
Remote dry-run verified original parent/source/runtime and0+1 GPUs. Results are
on project Lustre `student-initial/2b622073793933ada34054021664378e/`; HOME holds
only immutable source and small submit metadata. The real live-binding was
fetched/hash-verified as `live-binding-54560398.json`, SHA
`2d051ae3b4f351774303785dabad4c6cdc8b7f48084b2b030d3708ecb0f2c03e`.

The finite Quest observer
`.sdsc/diagnostics/student-initial-native-v1/observe.py` (SHA
`9b1472bf34e83d8be5b865a510db85bde3036fc9c93995a094feefb7c398b758`)
ran on quser43 as tool session78485, confirmed **COMPLETED0:0 at01:44:56Z**,
fetched bounded results and exited at01:44:57Z. Job/batch/extern all completed
with zero exit status; elapsed time was758 seconds. Its real started/status/
finished records are retained; no observer or compute job remains active in
this completed flow. Never repeat its permanent scientific claim.

Actual fetch: `.sdsc/fetched/54560398/fetch-ah0njq8o/`.
Publication SHA `9f2f532d1cbd15e0877d142bfeb99c4f23c11171ede80ee84840e15df1574081`;
producer report SHA `0aec6f9ffb67cda3dfdba59a9f4b786d1b76e0b9b3464c28f5aa0a52be5ee95e`.
Frozen independent execution/raw audit:
`.sdsc/diagnostics/student-initial-native-v1/actual-independent-audit-54560398.json`,
SHA `55b59e7946e39f67e83f9cbfba7157832d5e9f5ed5e602a1e0a2828e70eeba34`.
It verified the real plan/source/publication/accounting/memory and all128 prompt
encodings, response decodings and original parser/verifier traces. Results are
**0/128 answer-correct,0/128 full proofs,13/128 valid format**;51 responses ended
at EOS and77 at the256 cap. Maximum prompt length1194 plus256 fits1536. The
original base criterion13/128 is **false**. Removing the extra autocast has not
rescued it; no checkpoint, instruction, population or threshold changed.
Peak job memory was21,530,996,736 bytes (about20.05 GiB), leaving47,188,480,000
bytes; all five memory boundaries pass and own-job failcnt iszero. This is a
quality failure, not a demonstrated allocation or memory failure.

A separate session on **quser33** recorded a missing shared SSH master at01:33Z
and stopped remote operations without authentication fallback. Its dated
`connection-stopped-quser33.json` and exact recovery of the dry-run display copy
are preserved. That session's earlier RUNNING observation is superseded by the
quser43 observer's shared terminal/fetch/audit records above. Local inspection of
these completed results needs no new authentication. Any future remote operation
must first check the master on its own actual Quest host; authentication on one
host does not establish a master on another. Preserve every completed plan,
claim, release and frozen auditor/control byte.

Following manual authentication, **quser33** `tools/sdsc check` succeeded at
**01:46:08Z**, and again at **01:56:22Z**, using its runtime-derived shared
ControlPath. Remote identity was zgao12/login01; required commands, project
source path and account/partition/QoS checks passed. SSH is no longer the blocker
at this observation. The original128-response audit was also independently
replayed on quser33 with byte-identical output; the unique
`actual-independent-audit-54560398-quser33-replay.json` preserves that replay.

The original1.7B track is blocked by its fixed-initial scientific condition,
even if lower-LR calibration later improves trained quality. Do not submit G0 or
pilot, replace initial with trained weights, or call this complete-model acceptance.
A reviewable direction proposal is
`docs/refactor/sdsc_student_acceptance_recovery_options_20261001.md`: one
prospectively fixed larger native student, or a separately prepared common1.7B
baseline. Both retain original numerical gates and require distinct scientific
protocol/implementation/independent acceptance and actual results. The user
explicitly chose **keep1.7B and prepare a common initial model**. Existing
compute-resource authorization covers the necessary preparation, preflight and
gated continuation without per-stage confirmation. The new prospective
preparation protocol above implements this direction; original-native failure
is not retroactively accepted. Its preflight54562489 passed; full preparation
54562507 is running as described at the top of this handoff. No prepared model,
complete G0 or formal OPD/RL comparison result is accepted yet.
The reviewed paired-loss table, train-only response metrics and numerical-audit
limits are in `docs/refactor/sdsc_student_optimization_findings_20261001.md`.

Complete G0 still needs migrated adapters, two real step20→33 resumes,
adapted-teacher scoring and all26 original gates. The initial checkpoint must
score at least13 `answer_correct` results under the original full verifier on
the unscreened first128 validation examples (answer_accuracy at least0.10).
This requires the original parser and listed-step validity checks; neither an
answer tag alone nor reward==1 is the formal base metric. Complete-proof
accuracy is separately reported; do not silently replace either definition;
`base_capable`/`challenge` use separate circuit populations and cannot change that
denominator. Initial anti-shortcut and circuit gates remain mandatory. Improving
trained weights cannot repair the fixed initial-capability gate; never substitute
a trained checkpoint for initial or relax thresholds to force PASS. The old
G0/pilot adapters are not automatically valid for the new producer identities.
The original-native validation128 check now confirms that this necessary base
gate fails, independently of the earlier autocast diagnostic and training32
screen. It is not an executed complete G0 or an invented anti-shortcut/circuit
outcome. A change of model/initial scientific identity requires a separately
reviewed successor and the user's direction choice, not an automatic LR branch.

This is the canonical current-state summary for the OPD refactor and
ServerScheduler integration. AGENTS.md is authoritative for operating and
approval rules. Source, Git, scientific artifacts, and central scheduler state
must still be verified when mutable.

## Quest development / SDSC Expanse workflow

The user has now explicitly authorized starting formal training on SDSC and
selecting GPU count/resources from the workload. This includes preparation,
model/runtime setup, storage and distributed preflights, and the gated Qwen3-v2
path. Proceed with G0 and then a seed-42 pilot; do not infer authorization for the
full three-seed factorial or Gemma. Do not ask again for the already authorized
work. Scientific gates and duplicate-submission protection still apply.

The latest user instruction explicitly authorizes agents to fix bugs, submit
tests and continue until the teacher meets its standards and passes acceptance.
This includes evidence-led, independently reviewed teacher task adaptation;
do not stop at diagnosis or repeatedly request per-stage authorization. Keep all
original scientific thresholds, disjoint training/evaluation data, genuine
checkpoint identities, independent acceptance and submission reconciliation.
It does not authorize claiming success early or expanding to factorial/Gemma.

A second infrastructure job, `54345521`, completed with ExitCode 0:0 in 18
seconds on H100 node `exp-19-07`. It verified actual read/write/publication on
Lustre at `/expanse/lustre/projects/nwu181/zgao12/OPD/control-results`, without
adding a Lustre constraint. Small results were fetched under
`.sdsc/fetched/54345521/fetch-y8o28zud/`. Each future job must still verify its
own mounts. Project group quota was queried as 50 TiB, with about 381 GiB used
across the group; this is not a personal allocation guarantee.

Preparation uses persistent project root
`/expanse/lustre/projects/nwu181/zgao12/OPD`: a new Conda runtime is
installed at `envs/qwen3-v2-g0-py31213-cu128-v1`, and the exact pinned Qwen3
1.7B/8B snapshots (20,476,854,927 bytes) were downloaded directly into
`cache/huggingface`. Bootstrap status/logs are under the remote `bootstrap/`;
local preparation responses/plans are under `.sdsc/`. Check those before any
continuation or retry; directory existence alone does not mean preparation
completed. Runtime installation completed at 2026-09-18T02:12:30Z, all 19
G0 dependency pins matched and pip check passed. Python 3.12.13 executable SHA
is `2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19`.
A two-H100 preflight passed real model loading, NCCL, one global-64 optimizer
window, same-world state restore, memory and persistent output verification.
Its real job ID is `54345604`, intent `ce3205630acc4742a2bef829e210ec45`,
run `20260918T020133Z-dd9ab354eaeb-d631b1ce`, code SHA
`dd9ab354eaeb944a3e682629e7e851f9d845bd4f4071492f152878243e1c6324`.
Resources were 2 H100 / 24 CPU / 192 GiB / 30 minutes. It completed on
exp-19-04 in 2m48s, with job/batch/extern COMPLETED and ExitCode 0:0.
Actual cgroup ancestor limit was 192 GiB, peak 70.47 GiB and headroom 121.53 GiB;
peak reserved GPU memory was 44.89 / 29.79 GiB. Full model, optimizer, scheduler
and both CPU/CUDA RNG states were saved, perturbed and restored with hash checks.
The 10.95 GB checkpoint remains on project storage; five small reports/logs were
fetched under `.sdsc/fetched/54345604/fetch-q7k9dm9n/`. Final verified status is
`.sdsc/preflight-54345604-final-status.json`. The three prior test-only walltime
probes were estimates, not submitted jobs.
The old Blackwell execution certificate must not be presented as H100 evidence.
New Git-provenance tooling exports bounded genuine committed history separately
from the dirty wrapper snapshot; it must not create a fake clean scientific HEAD.

The formal pipeline stopped after failed teacher-data job **54345715**, intent
`f98dca38ea9d4d9bbe4f3917e151a796`, run
`20260918T021516Z-82d6fc51200a-2cf5f753`. It was submitted at
2026-09-18T02:19:59Z. The supervisor's retained SDSC accounting reports
`FAILED`, ExitCode `1:0`, elapsed **02:21:48**, with the batch step also failed.
Fresh SSH/accounting after reauthentication on quser34 independently confirmed
this same terminal result; the persisted failure artifacts were hash-verified.
Resources: 1 H100 / 24 CPU / 192 GiB, at most 4 hours. It uses unchanged accepted
prompt-v3 science at real HEAD 0215c356; the original teacher generator is serial.
It creates the full 144000-example family and then 2048 candidates on 256
ordered prompts, preserving teacher generation seed 31415 and experiment seed
42. Metadata-only validation passed on the actual restored checkout before
submission, then again on the GPU node. Logs confirm full dataset creation and
teacher weight loading, followed by the original store's coverage rejection:
**252/256 prompts had zero accepted candidates** (126 positive, 126 negative).
There are eight accepted candidates. Exact rejection counts are antecedent_mismatch
1099, response_syntax 829, step_syntax 111 and conclusion_mismatch 1. The literal
TRUE X occurs in 1875 outputs; 1468 copy the fake schema line verbatim. All 829
response-syntax failures are length-terminated. This supports a prompt-induced
schema-copying failure, with additional reasoning failures still possible.
The worker log matches its publication SHA. The wrapper's receipt records
successful persistent read-back of the 46,630,071-byte attempt ledger, manifest,
accepted view and dataset. A secondary legacy diagnostic-copy attempt failed
on `/scr` permissions, but the SDSC wrapper still preserved these artifacts.
Local failure evidence is `.sdsc/fetched/54345715/fetch-qi5c9hbc/`; the ledger
remains on SDSC with SHA
`28233d89872739b2d9d34d2dba4704ba5e857ca737005166c5776249fd8229ab`.
Never blindly resubmit or relax coverage to advance this run.
Admission evidence is `.sdsc/teacher-54345715-admission.json`; status/logs must
query this existing job. Results use the project control-results root, not HOME.

The next implemented task is `qwen3-v2-g0-calibration`: 2 H100 / 24 CPU /
192 GiB, using the original Accelerate FSDP canonical-SFT entrypoint, global
batch 64, at most 120 steps / 2M nonpadding tokens, validation/checkpoint every
20 steps. Its upstream job IDs must be explicit. Before claiming or submitting,
the remote control plane verifies both real Slurm completions, report and source
identities, and every consumed teacher dataset/store file. The worker independently
verifies the same proofs and the actual accepted scientific binding. Independent
CPU review exercised the real clean-checkout composer and train dry-run; it also
validated the actual fetched two-H100 report. This is calibration only: full G0,
multistep resume and the four-H100 preflight remain required before pilot.
The original Quest quser44 supervisor (launch PID 1192904) recorded `stopped`
at **2026-09-18T04:46:57Z** after teacher failure, with `no_retry=true`, no
cancellation, and a successful bounded failure fetch. No calibration submission
was attempted. Its exact plan,
launch evidence, state and log are in
`.sdsc/supervision/teacher-54345715-to-calibration/`. It checks every five
minutes and stops initiating operations after eight hours. Do not re-arm or
edit its pinned control files while active; inspect state and submission receipts
first. No service runs on SDSC. Reconcile the recorded stopped phase
and any submission intent before continuing; SSH authentication loss is not a
reason to repeat a GPU submission.

Its calibration release is `20260918T023430Z-91bb7848d574-73d8d61d`, 463 files /
4,372,559 source bytes; real code SHA is
`91bb7848d5742352708c89711af0f4b2bed9d5d66206d55872df8b7a9f38e910`.
Companion provenance SHA is
`3edd6179d4828c4bfd1053d9f1329bd5ec359d2dca52c64ccf6fd65ee5b3e359`;
both were uploaded and verified. The selected calibration limit is one hour,
with TERM 300 seconds early for stable output publication. Identical-resource
30/60/90-minute test-only estimates all reported immediate starts at 02:35Z;
the one-hour selection retains first-training evaluation/checkpoint/publication
margin beyond the real 2m48s single-update preflight. Evidence is in
`.sdsc/calibration-scheduler-estimates.json`. Those test-only IDs are not jobs.
Calibration has no real job ID; its teacher prerequisite failed and the flow
will not automatically restart. Student training has not begun.

The complete successor remains deployed but recorded **`stopped`** at
2026-09-18T04:49:14Z because its predecessor failed. It was launched on Quest
quser44 as PID 1693084 at 03:54:14Z. Its `stages` map is empty: no G0,
four-H100 preflight or pilot was submitted, and neither G0 nor pilot passed.
The finite flow directory is
`.sdsc/supervision/teacher54345715-g0-pilot-v1/` (plan, deployment, launch,
review, state and log). Its independently reviewed source snapshot is
`20260918T035115Z-7d6f46dcb09a-1e6ddd7b`, 489 files / 4,880,767 source bytes;
code SHA `7d6f46dcb09af2a8f35fe773f807bb5578033e2183ab411487438a3902727824`.
Uploaded genuine Git provenance SHA is
`a74f9d86dd6a1c34a7ae32d362f80fd55467f17ca1fd6103c84b1c552c79521e`;
local/remote plan SHA is
`5b67b19da2e68a0b6b2f306c9f29fb55a77a5c912a86524f5c4dd0231df7f2ea`.
Preserve both stopped flows' plans, claims and receipts; never blindly re-arm
them. The implemented successor requires verified calibration,
then executes G0 → four-H100 preflight → seed-42 pilot, at most four concurrent
GPUs, five-minute polling and a fourteen-day deadline. It stops on failed
science, SSH loss, changed control hashes or unresolved submissions; no blind
retry, auto-cancellation, full factorial or Gemma.

An earlier successful connection check ran from Quest **quser32** at
**2026-09-28T20:13:38Z**, after the user restored the shared master and requested
another attempt. Its runtime-computed path is `$HOME/.ssh/cm/sdsc-$(hostname -s)`.
SSH confirmed zgao12@login02; Slurm commands, source paths and project result
paths passed. Account nwu181 and the shared partition jointly permit
`nairr-gpu-shared-normal`. The check's optional 15-second runtime probe timed
out; a separate bounded check at **20:15:46Z** verified Python 3.12.13 and its
original executable SHA, Torch 2.8.0+cu128, NumPy 1.26.4, Transformers 4.56.2,
Accelerate 1.10.1 and tokenizers 0.22.0 without importing GPU libraries or
changing the environment. Evidence is in the v2 diagnostic directory's
`connection-retry-check.json` and `runtime-recovered-check.json`. Every GPU job
still verifies its own mounts. The earlier command-channel timeout and absent
socket evidence remain preserved; neither was treated as authorization to
retry authentication or repeat an uncertain submission. Both old formal flows
remain stopped.

Earlier teacher prompt diagnostics are complete; none establishes formal readiness.
All used one H100 / 24 CPU / 192 GiB / at most 30 minutes, retained exact raw
outputs, and completed with accounting COMPLETED / 0:0 and verified publication:

| Prompt | Job | Valid candidates | Covered training prompts | Elapsed |
| --- | --- | --- | --- | --- |
| v4 | 54351516 | 0/256 | 0/32 | 18m28s |
| v5 | 54368250 | 24/256 | 4/32 | 20m36s |
| v6 | 54368464 | 0/256 | 0/32 | 27m34s |

Their read-only observers under `.sdsc/supervision/probe<job>-readonly-v1/`
are finished and their small results were fetched. Preserve every receipt,
release and raw ledger; never resubmit those intents. Both formal flows remain
stopped and student training has not started. Diagnostic `passed` means only
execution/publication completed. Prompt-v4/v5/v6 protocols remain proposed;
no independent formal acceptance has been recorded.

Exact replay reproduces all retained verification traces. Cross-job baseline
response IDs, seeds and logprobs match exactly. v5 resolves some grammar errors
but still has fact restatements, invalid/forward citations and antecedent errors;
its answer-tag correctness is only 136/256. All 256 v6 responses instead begin
proof steps with fact IDs rather than required rule calls: 161 complete outputs
fail syntax and 95 hit the token limit. Of the complete responses, 143/161 have
correct answer tags; this is not proof success or general model incapability.
The independent inference audit finds low-entropy filtered outputs, with no
identified RNG/cache/model-loading implementation fault.

After v6, the renderer was restored to exact v5 instruction bytes, SHA
`55edb00c4d57197224c49dddfcb36ad1152640c8f8a19e2b223c1f3e968c504f`,
selected only from training-probe results before GPU validation. The failed v6
implementation `6c12ab9a278686a30b141be02e0313be38481f21`, proposed protocol and
raw ledger remain preserved. All graphs, canonical targets, parser/verifier,
model revisions, sampling seeds, thresholds and 52 execution-safety files are
unchanged. Do not continue prompt micro-tuning or accept v5 on coverage evidence.
Detailed diagnosis and identities are in
`docs/refactor/sdsc_teacher_grammar_repair_20260920.md`; earlier repair evidence
is in `docs/refactor/sdsc_teacher_prompt_repair_20260918.md`.

The separate `qwen3-v2-teacher-capability-probe` is implemented and independently
reviewed for one diagnostic submission. It measures the existing frozen
validation generation and prefix gates using actual v5, without formal Git
bindings or a readiness artifact. The first 128 rows match the preserved split;
prefixes fit 404–1194 tokens, at most 1322 including 128 completion tokens.
Six canonical targets exceed the original output limit; 122 fit, so the 0.85
exact-proof threshold is not structurally unreachable. Answer accuracy still
requires 0.90; all other frozen prefix thresholds remain unchanged. CPU checks
do not establish GPU capability. Actual controls were updated only after the
v6 watcher finished at 2026-09-20T23:55:10Z. The combined control/worker suite
passed 196 tests; after exact v5 restoration, fixture isolation and the
metadata-only prompt-SHA addition, 72 affected tests and independent review pass.
Implementation commit is `78661c2`; this is not scientific acceptance.
Job **54368737**, intent `841f146c4b394884bbf504cdda644893`, was submitted once
at 2026-09-21T00:05:39Z and completed on exp-19-01, COMPLETED / 0:0 in 5m51s.
It used one H100 / 24 CPU / 192 GiB / at most 30 minutes, nwu181,
nairr-gpu-shared / nairr-gpu-shared-normal.
Run `20260921T000425Z-24c06afe7afa-8d07dad1`, source SHA
`24c06afe7afa3df0cd72d503a7e9dcab6136708f1d196175b10047a4b73b6d3a`,
contains 506 files / 5,130,786 bytes with matching dry-run/deployment receipt.
The finite read-only observer is under
`.sdsc/supervision/capability54368737-readonly-v1/`, Quest quser43 PID 1701453;
plan SHA `b3c101d8686b14c07643f2ce52d065646c14ac1b0318693410a85f2c1946e23f`.
It finished at 2026-09-21T00:16:17Z and fetched verified small terminal results;
it submitted no successor. Report SHA is
`70ac8ccfe11034b61711697047d8e344902dd28f6de8cbbdefbeb2262281a51c`.
The separately fetched 8,138,235 bytes of raw evidence are under
`.sdsc/fetched/54368737/raw-verified/`; exact CPU replay reproduced all 128
generation traces and the original scientific reduction of 256 prefix scores.

Only retained top-k mass passed. Answer accuracy is 53/128 (41.41%, gate 90%),
exact proof 23/128 (17.97%, gate 85%), first-rule top1 32.81%, intermediate
top1 0%, target coverage 50%, recovery 25%, minimum causal shift -6.7365.
Independent tokenization replay found no off-by-one error in 384 probe sides.
Every intermediate target begins ` TRUE`, while v5 requests the rule's bare
consequent and none of the 128 generated responses uses that TRUE convention.
All 24 verifier-valid responses use bare positive literals, which the unchanged
parser accepts. This representation mismatch must be addressed before claiming
general reasoning incapability; other rule/citation failures remain real.
The first 128 validation examples have now informed diagnosis. Any subsequent
prompt adaptation must disclose that exposure and must not present reuse of
these examples as independent held-out confirmation.

The user continued the recommended original-teacher/output-contract route on
September 27. Proposed v7 explicitly states positive `TRUE <atom>` and negative
`NOT <atom>` consequents. The unchanged 87-token instruction envelope preserves
all 256 training-prefix counts (402–1246), graph/target bytes and the original
parser/verifier, seeds, model and thresholds. Instruction SHA is
`8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6`.
The probe preserves exact v5 as baseline and records TRUE spelling only as an
observation, never a new acceptance criterion. No teacher fitting is included.
The scientific/configuration suite passed 64 tests and the worker/observer
suite 42. Two independent-review diagnostic extraction edge cases were fixed
and regression-tested without changing the parser or rejected-response bytes.

One training-only diagnostic runs on the original first 32 prompts with
eight candidates; stop larger progression unless all 32 have an accepted
candidate. Before GPU results, supplemental confirmation is fixed to validation
rows [128:256], 64 disjoint complete pairs, ordered-example SHA
`532b11ac85fad0b35be1e253c2d30a8ca838ea8849adf77503a43b6433236073`.
It is conditional, diagnostic-only, and never replaces the original exposed
first-128 readiness gate. Full plan and disclosure:
`docs/refactor/sdsc_output_contract_repair_20260927.md`.
Before job 54472139, quser42 shared-master, remote identity, account/QoS and empty queue
checks passed. A routine metadata check timed out at 15 seconds; a separate
lightweight check verified the existing Python binary hash and package pins.
GPU-node mounts are checked per job. Implementation `b3e128c` received independent
bounded-diagnostic review; the v7 scientific protocol remains proposed.
Job **54472139**, intent `c53cdcdbbb3749ff8b34b29a7cc72699`, was submitted once
at 2026-09-27T05:56:58Z on exp-19-07. Resources: one H100 /
24 CPUs / 192 GiB / 30 minutes, nwu181, nairr-gpu-shared/shared-normal.
Run `20260927T055545Z-2efe95b7b76c-dc55b228`, source SHA
`2efe95b7b76c8820d170a3b8b8dbc6b8c22d930af3103f6494d91b20e32ad824`,
contains 508 files / 5,159,679 bytes, with matching preview/deployment receipt.
Fresh SSH/accounting at 2026-09-27T21:09:42Z confirms **FAILED / 1:0**, elapsed
**28m37s**, with the batch step also failed and the user's queue empty. The
wrapper received SIGTERM and preserved a failure report; model staging,
GPU-node persistent write/read checks and metadata validation had passed.
Its worker log contains only the validation success, with no completed
candidate/coverage measurements. Coverage is unknown, not zero, and this is
not a measured v7 quality failure. The submitted 30-minute request includes
`--signal=B:TERM@60`; the timing is consistent with an early end-of-allocation
signal, but accounting gives `Reason=None`, not an explicit timeout cause.
The precise signal origin and old worker's blocking stage remain unproven.
The reviewed interruption repair and separately authorized fresh run are below;
never reuse this failed job's identity or erase its evidence.
The 4,507-byte bounded fetch is `.sdsc/fetched/54472139/fetch-qx561dgs/`.
Both receipt-listed file hashes were reverified; report SHA is
`b1d5e86ed8b75f5f50a850e6213ed8d822be3446189b564195d20e516f7b77cf`.
Fresh accounting/queue evidence is
`.sdsc/diagnostics/prompt-v7/status-20260927T2109Z.json`.

Its finite read-only observer finished at 2026-09-27T06:27:46Z under
`.sdsc/supervision/probe54472139-readonly-v1/`, launch PID 2629236;
plan SHA `67e2b0373818da8cff26976745148628b68b0099dcaa8eeb69b35a4fe793392c`.
It fetched the terminal reports and started no successor. No new formal
store/calibration/G0/pilot has run.

The user explicitly requested automatic diagnostic checking and automatic
formal continuation after success. `tools/sdsc_auto_continue` was launched
on Quest quser42, PID **2931946**, at 2026-09-27T06:22:37Z. Its plan/review/
launch/state are under `.sdsc/supervision/probe54472139-auto-v1/`, plan SHA
`a6cad6d3e93f8bd24cce49af61569d5d24304bf2391325971a76448c5c168d42`.
It stopped at **2026-09-27T06:32:38Z**, reason `diagnostic_gate_failed`, with
`child_started=false`; no continuation claim was created, no Codex successor
ran, and no further job was submitted or cancelled. Preserve this stopped
flow; absence of a launch claim does not authorize re-arming it.

The user then explicitly authorized **bug repair and fresh submission**.
Real subprocess tests reproduced worker SIGTERM bypassing final publication,
an interrupted progress replacement masking the original signal, and wrapper
shutdown losing unread pipe output. The repair adds durable stage/timing and
candidate progress, asynchronous native-call stacks, safe signal cleanup and
bounded final log draining. The native stack observer caused the new failure
below and has been removed; the remaining evidence-retention fixes are retained.
It preserves the complete scientific loop, all sampling/model/
prompt bytes, 24 threads and 52 execution-safety files. A hidden-GPU SDSC check
completed Torch import in 162.15s with only 2.62 CPU seconds; one/24-thread
comparisons both showed slow imports. This supports startup wait, not a proven
deadlock or a complete explanation of the old failure. Details and limits:
`docs/refactor/sdsc_probe_interruption_repair_20260927.md`.
Control/boundary tests passed 125; independent final worker/wrapper/boundary
review passed 51; retargeted observer/continuation tests passed 50.

Diagnostic **54485969**, intent `c2d9dfddae174c0eb1d075d4738ce4f0`,
was submitted once at **2026-09-27T21:35:17Z** and is now **FAILED / 1:0**,
elapsed **11m33s**, on **exp-19-13**. Resources were one H100 / 24 CPU /
192 GiB / 60 minutes. Run `20260927T213316Z-bb14f0613cee-3bb69dd0`,
code SHA `bb14f0613ceee7dd476d69eaf61e9e337769fc48356c079314b5bca9d312cee3`,
contains 515 files / 5,264,398 source bytes. Its receipt, release and all failure
artifacts remain preserved; no v7 student training started.

The worker exited **-11 (SIGSEGV)** after 600.307789 seconds. It had completed
**28 baseline attempts, no v7 candidates**. The second 300-second asynchronous
stack dump ended mid-frame in Torch `Linear.forward` / `Module._call_impl`.
A bounded real CPU Torch A/B reproduced SIGSEGV with that observer and completed
61,131 finite forwards without it. Local Python 3.12.14 differs from remote
3.12.13; this reproduces the mechanism but does not establish a repaired GPU
PASS. Verified remote report SHA is
`97cc8834738b5c5b24949f3216b5bcd75f1c7599b94aa2763939d87c1a72731a`.
Raw proof and the minimal removal of asynchronous traversal are documented in
`docs/refactor/sdsc_probe_stack_repair_20260928.md`. Ordinary Python signal
cleanup, synchronous exception traceback, durable progress/ledger, and wrapper
publication remain. A native hang/SIGKILL may retain no traceback or final
worker report; never fabricate that evidence. Scientific parameters are unchanged.

Both 54485969 Quest processes are terminal and their host PIDs are gone:
`.sdsc/supervision/probe54485969-readonly-v1/` finished at **21:49:08Z**;
`.sdsc/supervision/probe54485969-auto-v1/` stopped at **21:54:06Z**,
reason `diagnostic_gate_failed`, `child_started=false`. No continuation claim or
successor job exists. Preserve these and the earlier 54472139 flows; never
re-arm them. The user's latest instruction explicitly requests continued
monitoring and automatic formal progression after success.

Following the minimal repair and independent review, fresh diagnostic
**54489646** was submitted once at **2026-09-28T00:59:35Z**, intent
`523ada1d862c428986a4b3dfb671de73`. Fresh remote verification at **02:18Z**
confirms job, batch and extern **COMPLETED / 0:0**, elapsed **29m03s**, with
verified persistent results. It ran on **exp-19-08**. Resources were **one H100 /
24 CPU / 192 GiB / 60 minutes**, nwu181, nairr-gpu-shared/shared-normal.
Run `20260928T005810Z-15cffdded272-cf5d74b8`, code SHA
`15cffdded27232359c518ed4672d3062535f3606126054e60bc97f22b1058260`,
contains 516 files / 5,275,758 source bytes with matching preview/upload.
Its actual receipt is `.sdsc/submissions/523ada1d862c428986a4b3dfb671de73.json`.
The full **32 baseline + 256 candidate** loop completed without SIGSEGV or
interruption. Baseline accepted 3/32; paired v7 candidate-zero accepted 6/32.
All eight v7 candidates yielded **42/256 accepted**, covering only **7/32**
prompts; **25 prompts have no valid candidate**. This fails the preregistered
32/32 advancement gate. The report's `passed=true` means diagnostic execution
and publication only, not scientific readiness. Errors were unknown_citation
96, antecedent_mismatch 61, step_syntax 36 and response_syntax 21. The subsequent
independent CPU diagnosis below reproduces every recorded rejection.
No parsed positive conclusion used TRUE (0/655); no raw TRUE step line was
observed. This is rendering evidence, not permission to alter outputs or gates.
Verified report SHA is
`db2e2c0ff4da20694196a1308f65b75dee8c3f47b777413e922e6bca1c29b3ea`;
the 143,055-byte bounded fetch is `.sdsc/fetched/54489646/fetch-0ol5j72o/`.
The runtime repair now has real complete GPU diagnostic evidence, but v7 failed
the necessary training-probe gate. Its formal readiness evaluator has not run.
No student training, supplemental validation, full teacher
store, calibration, G0 or pilot was started by this recovery.

Two new finite processes launched on **Quest quser33 at 01:05:21Z**:

- `.sdsc/supervision/probe54489646-readonly-v1/`, PID **1849359**,
  canonical plan SHA `5b282102d421e95b95aa30e213e20f76718d3354737136e6807d450c91aec7b1`;
- `.sdsc/supervision/probe54489646-auto-v1/`, PID **1849367**,
  canonical plan SHA `fb22e016b9e2b4b8f7908e4ee8721b47fd628a257471a3e5d155c4453b899388`.

Launch-time host inspection confirmed both alive, PPID 1 and independent
sessions. The observer is now **finished at 01:30:29Z**, and the automatic gate
**stopped at 01:35:27Z**, reason `diagnostic_gate_failed`,
`child_started=false`. Its notice explicitly reports 7/32 versus required 32/32.
No later submission receipt or continuation child exists. Preserve this third
terminal diagnostic flow too; never re-arm it or report it as actively monitoring.
Independent admission review verified the actual receipt, deployed worker,
plan/control/task hashes and absence of earlier start/claim. Preserve all
plan/review/launch/state records and do not edit their pinned controls or task
document while active. Every five minutes the gate consumes verified observation;
only complete accounting/publication and **32/32** candidate coverage can start
one Quest continuation. Failure, stale evidence, changed controls or SSH loss
stop progression without retry/cancellation or automatic prompt adaptation.
Verification passed **31 worker + 21 wrapper/boundary + 50 observation/continuation**
tests, plus ruff/compilation/diff checks and independent science-invariance review.
All 52 execution-safety hashes remain unchanged. Commit `a05a67b` contains
the worker/test repair, retargeted continuation and launch handoff; existing dirty
`.gitignore`, `AGENTS.md` and earlier untracked SDSC files remain preserved.

The untriggered continuation task is designed to handle the fixed supplemental cohort,
original readiness, coherent adapter migration, actual independent acceptance,
fresh complete teacher store/matching preflight and the authorized student
chain. It does not equate diagnostic success with formal acceptance. Codex runs
on Quest with `exec --approve-for-me` (workspace-write and automatic approval
review), once for at most eight hours; the observer has a fourteen-day deadline.
Actual no-operation and read-only SSH handshakes passed, including remote
identity `zgao12`; 73 affected CPU tests and an independent 50-test review pass.
This is a project-owned finite process, not an app scheduled task or SDSC service.
Read [automatic continuation instructions](../sdsc_auto_continuation.md) and its
actual state before any competing work. The old quser42 master was foreground
on `pts/105`; the new flow uses the existing quser33 master. Do not assume
terminal disconnection preserves an SSH master. SSH loss,
stale state, changed controls or unknown submissions stop progression. At that
stage, the blocker was measured teacher-proof quality, not SSH or the previous
runtime crash; the accepted adaptation below subsequently resolves this gate.
The user then requested continued diagnosis. The 2,462,548-byte raw ledger was
fetched once and verified against publication SHA
`ca97a4cef49e2884dea31a2332b6e6aaa310b3174f898966d65772188ca09fc6`.
All **288** original traces replay exactly. Independent audits verify all 256
graphs, labels and canonical proofs, actual saved prompt/response token
round-trips, unique candidate seeds, and exact historical v5 baseline replay.
Canonical targets need only **54–163 tokens including EOS**, within the bound.
No examined prompt-transport or verifier defect explains the failure.

The 96 citation failures include 83 self/forward references, eight rule-ID
references and five undefined earlier steps. All 61 premise mismatches have
the right citation count but wrong literals. Of 235 complete outputs, **73
answer tags are wrong**; another **120 answer-correct outputs have invalid
proofs**. Missing TRUE is not a rejection cause because the parser accepts bare
positive atoms, though the canonical prefix-target mismatch remains. Distinct
seeds produce just 83 distinct within-prompt responses; the 224 generations
after candidate zero add only one newly covered prompt.

Offline attribution only: retaining all original dependencies and reordering
steps yields 56/256 valid traces across 9/32 prompts; ignoring citation names
while retaining the original grounded rule sequence yields 74/256 across 11/32.
Neither alters the official **42/256, 7/32** result or qualifies teacher data.
The measured blocker is reliable serial proof generation under this exact
frozen Qwen3-8B non-thinking configuration; it is not general model incapacity.
The response is the separately reviewed adaptation work below, not a blind
prompt retry, larger sampling budget or relaxed validator.
No supplemental validation examples were exposed by this diagnosis.

Full hypotheses, examples, limitations and CPU reproduction commands:
`docs/refactor/sdsc_v7_quality_diagnosis_20260928.md`; scripts/JSON are retained
under `.sdsc/diagnostics/v7-quality/`, with raw evidence under
`.sdsc/fetched/54489646/raw-verified/`. Independent tokenization, verifier,
diversity and counterfactual reviews pass. This diagnosis changes only that
report and the handoff in Git; no production science, weights, thresholds,
submission or cancellation changed. Original uncommitted files remain untouched.

The subsequent authorized repair found a separate **readiness scoring bug**:
multi-token counterfactual likelihoods used the other target's autoregressive
history. The scorer now evaluates the same complete target under each context,
also correcting alternative likelihoods. A deterministic CPU regression flips
the old incorrect negative shift to the mathematically correct positive value.
Five new tests (three initially failing) and the 41-test affected suite pass;
independent review reran the five tests. Own-context top-1/coverage/mass,
probe construction and all eight thresholds are unchanged. Historical v5 has
87/128 affected target pairs and needs new model forwards for corrected shifts;
its old values are not corrected evidence. The v7 generation failure is unchanged.
See `docs/refactor/qwen3_teacher_prefix_scoring_repair_20260928.md`.

The separate teacher-adaptation path keeps the Qwen3-8B base,
non-thinking template, v7 rendering and strict proof gates. It trains LoRA
rank 32 on all seven attention/MLP projection families and exports/reloads a
new dense checkpoint. The original production adapter prohibition remains.
`tools/sdsc_teacher_adapt.py` and its wrapper expose a fixed **1 H100 / 24 CPU /
192 GiB / 60-minute**, eight-step, global-32 execution preflight through
`tools/sdsc`; preflight PASS is never teacher acceptance. The actual tiny-Qwen
training/merge/reload regression and related tests passed (46), transport and
boundary tests passed (148), and independent worker/transport reviews passed.
Fresh job **54493015** was submitted once at **2026-09-28T03:26:38Z**, intent
`f17b8a53cf8a4d3b85e9d7c8d27c89c9`, and completed on **exp-19-08** with
job/batch **COMPLETED / 0:0**, elapsed **16m48s**, plus verified persistent
results. Run `20260928T032541Z-1ce8c1707b61-4096a87c` contains 525 files /
5,444,453 source bytes; code SHA is
`1ce8c1707b611a4a23e9b0c88c6608a7e370e08c1c91ee312886c431ce3a7f17`.
Its real receipt is `.sdsc/submissions/f17b8a53cf8a4d3b85e9d7c8d27c89c9.json`.
The real eight-step execution preflight passed; **that preflight checkpoint did
not pass teacher quality**. Loss declined from 0.452631 to 0.016239; 227,814 input tokens were
consumed, with nonzero adapter update, unchanged frozen base before merge and
zero dense-save/reload logit error. Steady 32-sample updates took about eight
seconds. Peak reserved GPU memory was 24.62 GiB; cgroup peak was 67.60 GiB,
with 124.40 GiB headroom. The adapted checkpoint identity is
`7f641e70813c33ae25bda42f659d3e124865ace7a7b858aa17d0925a84269da7`.
The 32-example independent development set at the unchanged preflight 128-token
cap passes six of eight metrics: first-rule 0.875, intermediate 1.0, top-k
coverage 1.0, corrupted recovery 1.0, minimum mass 0.999977 and minimum causal
shift +1.334107. Answer 0.875 and exact proof 0.78125 remain below 0.90/0.85.
These are development measurements, not original validation or teacher-store
acceptance. Small results are `.sdsc/fetched/54493015/fetch-3sn8l2nu/`; verified
report SHA is `f0fa4dbe3027373e97b5da865463ef593c7f9d6b5a54b0e103e753e8123d40b4`.
The first fetch exposed an optional dev JSON larger than 1 MiB; the reviewed
fix skips it explicitly while retaining strict required-file limits. The
295,733-byte fetch records that 1,192,578-byte skip; no weights were downloaded.
All old observers/continuations remain terminal and must not be re-armed.
Fresh four-H100 preflight **54493777** was submitted once at
**2026-09-28T04:10:42Z**, intent `2ebd02659bbb44478049b1dd45c16e4f`.
Resources: 4 H100 / 24 CPU / 192 GiB / one hour, `nwu181`,
`nairr-gpu` / `nairr-gpu-normal`. Its release is
`20260928T040828Z-ca6f0ae6dbf0-2a503af2`, 538 files / 5,725,597 bytes,
code SHA `ca6f0ae6dbf0018d94df08196ec2f75a07cdf2a928b2b61044b6bfac1fef650c`.
The matching full-fit release `20260928T040932Z-ca6f0ae6dbf0-8b0a9a17`
is separately uploaded with identical execution bytes, but is now obsolete for
continuation after the memory-contract repair; it was never submitted.
Actual execution-plan SHA is
`90cbcbcc1fe9b11a885bcc9fe81667d02787d972e6117e360d7e2fb38a381475`.
The 04:06Z connection check passed; the four-GPU partition was UP and the
user queue empty before submission. Final transport suite passed 173 tests,
independent boundary review passed 24, worker/loader/identity passed 105,
and real CPU Gloo tests covered four-rank global loss and same-world resume.
These are execution-admission evidence, not real four-GPU or teacher PASS.
Fresh accounting confirms **FAILED / 1:0**, elapsed **3m36s**, before numerical
imports/model loading/training. Source/model staging and actual input/output
Lustre plus node-local ext4 checks passed. The shared-node guard rejected
`cgroup must expose finite 192 GiB and real peak`. Accounting records requested
24 CPU/192 GiB, but actual allocation 72 CPUs and no memory TRES (batch mem=0)
on the exclusive `nairr-gpu` partition with `SelectTypeParameters=CR_CORE`.
The original guard did not retain raw levels, so its exact observed limit cannot
be reconstructed. Preserved failure fetch:
`.sdsc/fetched/54493777/fetch-0ua7wqtr/` (11,205 bytes).
The separately reviewed fit-only repair records the real finite own-job/step
cgroup boundary independently from the requested 192-GiB workload budget.
It retains the original 20%/32-GiB headroom relative to that budget, rejects
unattributable/UID-only counters, and records raw levels before failure.
Actual allocation CPU count is reported separately; computation remains four
ranks at six threads each. Legacy shared/central guards remain unchanged.
Fresh repair preflight **54494477**, intent `9c10a6523c25490cbd6b0453b1fc1876`,
was submitted once at **2026-09-28T04:28:34Z**, with the same explicit
4-H100/24-CPU/192-GiB-request/one-hour envelope. Its independent source release
is `20260928T042618Z-080501637f3e-6f05ef72`, 549 files / 5,893,665 bytes;
code SHA `080501637f3e88f310ee4a2d98c95692f97a8c2e3620264c43950f6d5c5e0fcc`.
Execution-plan v2 SHA is
`a7f3ee0bda10d13158e47d3e28a1e8c4fea4f21e05c60f43a4255af9f3d11185`.
The repair passed 43 memory/worker tests and the 222-case affected boundary
suite, plus independent raw-counter/output-directory review and regression.
This repaired preflight completed **COMPLETED / 0:0 in 11m24s** on exp-19-08.
All 13 execution checks pass, including four-rank global-64 updates, actual
same-world restore, changed-world rejection and dense export/reload. The actual
own-job peak was 69.75 GiB against the unchanged 192-GiB budget; the measured
finite kernel cap was 898.44 GiB, explicitly not a 192-GiB enforced cap.
Stable updates took about 4.42 seconds. Persisted report SHA is
`a04ea2b34f2c380cb40d7a68c068cf722a142cdd51c239cdc8d25fc124e81ffb`;
the 279,059-byte bounded fetch is `.sdsc/fetched/54494477/fetch-5mpzfgb2/`.
Its small development assessment still fails quality and is not acceptance.

Full teacher fit **54494742**, intent `829efc1b139f4c37a89ccc1d0e01d857`,
was submitted once at **2026-09-28T04:42:28Z** after verified preflight success.
It uses independent release `20260928T042742Z-080501637f3e-d1b2dc92` with the
same execution v2 bytes: 4 H100 / 24 requested CPU / 192-GiB workload budget /
four-hour limit, account `nwu181`, partition/QoS `nairr-gpu`/`nairr-gpu-normal`.
Its exact full-fit plan SHA is
`6d94adb71e0f131e7b74db10a86aeb01a3903d3be59129c71d368f931c89db7c`.
Fresh accounting at 06:04 UTC confirms job/batch/extern **COMPLETED / 0:0**;
the job elapsed **1h19m36s**. All 512 updates, four epochs, 32,768 sequences and
29,909,768 nonpadding input tokens completed, with all 13 execution checks passing.
Actual own-job peak was 108.24 GiB; maximum reserved GPU memory was 43.75 GiB.
The wrapper hash-verified 71,277,357,278 bytes on persistent storage. Report SHA
is `a7009dac991b811879af0e99ee0fa052bd94ea3801f7533f402a2ccd78fc8629`;
the 919,727-byte fetch is `.sdsc/fetched/54494742/fetch-n51zizjt/`.

All four complete 512-row development evaluations pass all eight gates. The
unchanged earliest-PASS rule selects step **128**, dense identity
`6928f2537dcca5f2d65c1498659e1ebf011845eb72ef364b9544036c2238e9c7`.
At this checkpoint answer accuracy is 1.0, exact proof 511/512, first-rule and
intermediate top1, target coverage and recovery are all 1.0; minimum causal
shift is +8.399499 and minimum top-k mass is approximately 0.9999998. Later
checkpoints have exact-proof accuracy 1.0 but must not replace the selected one.
Exact development summaries were separately fetched with publication/hash
checks into `.sdsc/fetched/54494742/development-verified/`; summary SHA is
`200911f2afc6f4f805dfa2abe07225f134624f8da548bd559ebeebd3d83c7747`.
Development selection alone does not establish holdout or teacher-store
acceptance. The required qualification, raw replay and independent acceptance
have subsequently completed as recorded below.
Preserve all receipts/releases and reconcile unknown intents; never reuse
failed 54493777 or start a competing GPU flow.
No new detached observer or continuation process is active.

The `qwen3-v2-teacher-qualify` transport, four-rank worker and independent
CPU auditor are implemented and cross-reviewed. They bind the actual fit origin
to 37 explicitly named scientific files (all match the completed full-fit release),
replay all four complete development evaluations and enforce the fixed first-PASS
checkpoint, then permanently claim each ordered qualification stage before
inference. No producer or CPU auditor can self-accept a teacher. Default fetch
remains small; the no-inference remote CPU audit hashes all bounded raw evidence,
checks permanent claim history and returns the actual independently recomputed
acceptance object. Its complete fixture report is 561,526 bytes, below1MiB.
The protocol is `prereg/amendments/qwen3_teacher_adaptation_v1.yaml`,
review-neutral core SHA
`dcd5fca7c87bda603f930e1e053d34073ca12aa079fa884c0e4e87e32ec099f4`.
The joint CPU suite passes **172/172**; independent worker, audit and boundary
reviews also pass, including the final one-line recomputed-object addition.
Review evidence: `.sdsc/diagnostics/teacher-adaptation-v1/qualification-preparation-review.json`,
SHA `a3aec047178968b6ab1bc251d8f4711ac86a4bbd19c850602b9cae5cc0f35a3a`.
Implementation commit is **d1ab8dacd834101b88d806bb6d75a44ae1949cb3**.
Independent reviewer `/root/v7_science_review` accepted this exact implementation
for qualification at **2026-09-28T06:09:44Z**, after checking all 47 named science
blobs, the 37 fit-origin files and actual full-fit/development evidence.
Independent review record canonical SHA is
`c4c5948fdb565a25d67ad5a73bacf7025af69b902007acf68b94f813e8bf429e`.
The separate review-only acceptance commit is
**929fb14834852a7c91e6656c76fd1834e1b5007d** and changes only the YAML review and
this handoff. That commit accepts the protocol only; actual teacher-result
acceptance is separately recorded below. Preserve the frozen selection and all
measurement and independent teacher-result acceptance gates. The Quest checkout retains
unrelated `.gitignore`/`AGENTS.md` modifications; the genuine restored scientific
checkout must be clean and match every accepted named blob.
The genuine provenance bundle was restored locally and passed accepted-lineage,
all-named-blob and clean-checkout verification. Fresh independent release
`20260928T061138Z-d30d5264771a-e6664e24` contains 555 files / 6,131,408 bytes,
code SHA `d30d5264771a6c316aa5f21405973f8bc503c880f85803478950cdc4cfef952d`;
matching preview, upload and verified provenance SHA
`5cd7ceb54364af622ec19315b316cd0884f75bd25d5ea4fa87cd364bb00105c3`
are retained under `.sdsc/diagnostics/teacher-adaptation-v1/qualification-*`.
SSH identity, the exact Python binary/seven runtime pins and persistent login
paths were freshly rechecked before submission; every GPU-node mount remains
independently checked by the job.

Qualification **54496291**, intent `9d7f9e3394c84b6c901aa98713e0dbda`, was
submitted once at **2026-09-28T06:16:23Z** after actual remote prerequisite,
protocol, source-origin, selection and permanent-claim checks. Resources are
**4 H100 / 24 CPU / 192 GiB / two hours**, `nwu181`,
`nairr-gpu` / `nairr-gpu-normal`, no requeue, TERM 300 seconds early.
The 07:39 UTC accounting query confirms job/batch/extern **COMPLETED / 0:0**,
elapsed **1h15m14s**, on exp-19-01, with verified persistent results. All four
ordered stages pass: train_probe covers 32/32 prompts; supplemental and formal
readiness pass all eight gates; formal_store completes 2,048 candidates and
covers 256/256 prompts. Both readiness cohorts have answer/exact-proof,
first/intermediate-rule, target-coverage and recovery accuracy 1.0. Formal
minimum causal shift is +5.877720 and minimum top-k mass 0.999999762.
The publication contains 154 files / 179,368,485 bytes. Its receipt SHA is
`0c8bfc0b31bd965244a9d61a158d62d1b8edd26c1f09373083f756d66ee1162e`;
report SHA is `2b067e9e33e7a50ce0697f8defde697e714f49ddc2caed9769bbcc3c0d82b7ba`.
The 759,825-byte bounded fetch is `.sdsc/fetched/54496291/fetch-7j1keu8s/`.
Immutable final accounting is
`.sdsc/diagnostics/teacher-adaptation-v1/qualification-final-status.json`.
Claims SHA is
`2b82e4d91d4be939a436eb7d2d49b58897fd2cecbe462efbfb889cd2def263f1`,
prerequisite SHA `7cdc344e26f5b01d9f9409504193702a3f8efc3a44ee72ca0fa39c93c87022cf`.
Never retry this completed job or its permanent experiment reservation.
Formal exposure progress must be read from its actual stage claims/outcomes.
The teacher has now passed independent result acceptance. Preserve unrelated
dirty files, producer reports and all old stopped flows. No new observer or
student training was launched; never repeat these completed submissions.

Independent CPU audit **54497294**, intent
`b04f8e93a2cc7f3e2961de65ecdac7ca`, was submitted once at **07:45:08 UTC** and
completed **COMPLETED / 0:0 in 41m18s** on exp-19-05. The frozen tokenizer
replay uses about 33m49s child CPU and 1.2 GiB peak memory, with no model
inference. It runs on verified node-local scratch and reads/publishes through
actual Lustre mounts. Account/QoS inspection rejects zero GPU (`QOSMinGRES`),
so the smallest discovered allocation was **1 H100 / 1 CPU / 16 GiB / 90 min**,
`nairr-gpu-shared` / `nairr-gpu-shared-normal`, with CUDA hidden. Do not run
this expensive replay on a login node or infer a hang from its quiet phase.
Its plan, real receipt, accounting, fetched audit and fetch evidence are under
`.sdsc/fetched/54496291/batch-audit-b04f8e93a2cc7f3e2961de65ecdac7ca/`.
The **558,712-byte** actual audit independently replays all raw scientific
outcomes and permanent claims; its canonical SHA is
`161e86554396f43193d50e103a1a42b7bfcb82242edb79230e0cb6102b2a90f8`.
The recomputed acceptance evidence exactly matches the producer evidence.

Two control-only repairs preserve the deployed scientific bytes and intent:
local preview checks remote path spelling without resolving Quest's different
`/home`; the separate `batch_audit_observe.py` handles SDSC's empty archived
`sacct Comment` only with the pinned actual live `scontrol` submission binding.
It preserves raw accounting and rejects conflicting nonempty comments. Both
12-case regression suites and independent reviews pass. Use this observer's
read-only `status` for this audit; never resubmit, overwrite fetched evidence,
or use the original strict-Comment status helper. All 47 accepted scientific
files remain unchanged.

Independent reviewer **/root/teacher_result_review** verified the actual raw
replay, all named science, original job/publication records, permanent claim
chain, exposure history and selected dense-checkpoint provenance. The distinct
attestation permits the accepted domain finalizer to produce
**formal_teacher_accepted=true**; neither producer nor CPU audit self-accepts.
The accepted teacher canonical SHA is
`5d6952823441bde567cdf7f5fad8b4625c58ee7e82425aad76c10433d0ec5337`.
Local acceptance and review files are under
`.sdsc/diagnostics/teacher-adaptation-v1/accepted-teacher-54496291-4641d1f2c54e627e/`.
All four files (**8,267 bytes**) were published once to:
`/expanse/lustre/projects/nwu181/zgao12/OPD/teacher-acceptance/54496291/4641d1f2c54e627e41d0b2f0e1363c48cada91643fb5c18e946eb19a62a16964/`.
Publication and subsequent read-only check both report complete with identical
hashes and no missing files; the reviewer independently checked both. Keep the
actual publication receipt and `teacher-acceptance-publication-check.json`.
The acceptance is specific to the reviewed teacher-only **256-token** protocol;
all eight thresholds are unchanged, and **original_128_token_readiness_pass_claim
is false**. Detailed evidence and safe recheck commands are in
`docs/refactor/sdsc_teacher_adaptation_20260928.md` and `docs/sdsc_workflow.md`.

The user's subsequent instruction is to start the student path. The new
`qwen3-v2-adapted-preflight` and `qwen3-v2-adapted-calibration` tasks preserve the
original 47 teacher science files and consume the independently accepted dense
identity and exact nine producer/acceptance files. The in-memory SFT view only
adds the explicit learned teacher identity columns; measurements and candidate
order remain unchanged. The independently accepted successor protocol is
`prereg/amendments/qwen3_adapted_student_calibration_v1.json`; genuine distinct
implementation and review-only acceptance commits are required before execution.
Implementation `9129f32ad2f53b8fa9073b417fd615560278a2db` passed independent
review by `/root/student_transport_audit` at 2026-09-28T17:31:10Z, with root and
`/root/student_migration_audit` separately reviewing that agent's worker files.
All 46 named student and 47 frozen producer blobs match real committed bytes.
The combined affected suite passed 260 tests and wrapper regressions passed 13.
Only the protocol review block and this handoff change in the acceptance commit.
Its named student implementation preserves full-parameter canonical SFT,
seed 42, 256×8 candidates, global batch 64, 1536-token model inputs,
120 updates / 2M tokens, evaluation/checkpoint every 20 and student generation128.
Both new tasks use two H100 / 24 CPU / 192 GiB on shared normal QoS, with a
one-hour preflight and two-hour calibration limit. Node-local staging verifies
256 GiB headroom and actual input/persistent-output mounts. All outputs and the
original teacher evidence are read-back verified before persistent publication.
The transport admits only matching actual completed producer/preflight jobs;
unknown submission intents still require reconciliation. On Quest quser32,
the shared master, exact Python binary and package metadata and all four
published acceptance hashes were rechecked; the SDSC user queue was empty.
Scientific acceptance is commit `77603c14802c21e0dce09fe7617705d33425ce94`.
The actual first preflight **54504816**, intent
`a3f7379c0975413a8093d06d757df8a0`, run
`20260928T173238Z-1385fe5349ff-8b5bf384`, was submitted at 17:37:54Z.
Accounting reports FAILED / 2:0 after two seconds, before Python/model execution.
Slurm copied the shell launcher into its spool directory; its `dirname($0)`
lookup therefore searched for the Python worker in Slurm spool instead of the
immutable release. Preserve this job/receipt and bounded fetched startup logs.
Transport implementation `cba65bfecbba26fafffc1c63e2b5a7f0237beba0` adds
`tools/sdsc_student_launch.sh`, which resolves the unchanged accepted Python
worker through the bound absolute release argument. Independent reviewer
`/root/student_migration_audit` accepted this seven-file transport delta at
2026-09-28T17:44:34Z after 47 passing spool/transport/wrapper tests and a real
old-failure/new-success Bash reproduction. All 46 named student and 47 teacher
scientific files remain byte-identical to their original accepted commits.
This review-only handoff commit accepts the locator correction, not new science
or GPU results; the original scientific implementation/acceptance remains
9129f32/77603c1. Genuine provenance may export the descendant transport pair.
The separate transport acceptance commit is
`97093f5f3bd9c05511739c38324e94c4e1c6adf8`.
Fresh preflight **54504895**, intent `db2ff743d58a4762ae371073239db939`,
uses release `20260928T174604Z-0e406f026b38-67bb45b3`, source SHA
`0e406f026b388f81f5f81bb94aacb3a27d9454ad6ba5a1313c47f16013db8d6f`
and genuine provenance SHA
`e03c3f54684b8ab646b72e0949675fce50b1cca46690b2506207715fcccb05be`.
The 18:06 UTC query confirms job/batch/extern COMPLETED / 0:0 in **14m04s**
on exp-19-07. Both ranks pass the global-64 full-parameter optimizer window,
NCCL, full model/optimizer/scheduler/RNG restore, and actual learned teacher
forward. The complete published inventory contains **10,946,233,935 bytes**,
including the top-level report, verified on persistent storage;
the final status independently validates the report and publication inventory.
Report SHA is `87db798b5d80fd003614a5ceb4962885e21095b480d0b487f4c787c13a555777`.
Five small reports/logs (62,050 bytes) were fetched to
`.sdsc/fetched/54504895/fetch-w7t20yxp/`; final accounting is
`.sdsc/diagnostics/adapted-student-v1/preflight-54504895-final-status.json`.
Resources were two H100 / 24 CPU / 192 GiB / one hour. Preserve this accepted
preflight. Independent review by `/root/student_migration_audit` at 18:10:47Z
accepted the actual report, teacher identity, checkpoint inventory and calibration
upstream gate. Host peak/headroom were 104.538/87.462 GiB; reserved GPU peaks
were 44.887/29.787 GiB. Fetched log tails are diagnostic excerpts, not the
byte-identical full persistent logs.

Student calibration **54505782**, intent `f2f1d5c7614a46d7b1aedbef33c066d4`,
was submitted once at **18:11:44 UTC**, binding accepted teacher `54496291`
and verified preflight `54504895`. It uses **2 H100 / 24 CPU / 192 GiB / two
hours**, `nwu181`, `nairr-gpu-shared` / `nairr-gpu-shared-normal`, no requeue.
Fresh release `20260928T180733Z-4392a35b7151-7657d18c` contains 573 files /
6,443,492 source bytes, SHA
`4392a35b71518df85c5669bd1164878f2c2f3cb7d3e7dc0910c97abafa0480a4`.
Genuine provenance SHA is
`4301ea0a6d46fd88a60b983642b99e0faba7a726ecfb4bec65aeb9398ef88af0`;
science HEAD remains `97093f5f3bd9c05511739c38324e94c4e1c6adf8`.
The actual receipt binds prerequisite SHA
`1b6eddf8cfc0dd77a996feea796b901f8ffa7c838f0f9a8017667d61dc277e57`.
Final accounting reports **FAILED / 1:0 in 5m13s** on exp-19-07. Source/science,
all inputs and initial checkpoint export passed. Both training ranks then failed
before their first optimizer update: Accelerate 1.10.1 defaults
`fsdp_use_orig_params=true`, while the unchanged FSDP validator requires false.
The old shared YAML omitted the field, and launcher environment construction
overrides an external variable. This is an execution configuration failure, not
teacher-quality or training-result acceptance. Preserve this run and intent.
Its persistent output is under
`control-results/20260928T180733Z-4392a35b7151-7657d18c/f2f1d5c7614a46d7b1aedbef33c066d4/`.
Dry-run, deployment, provenance, exact command and actual submission/status
evidence are under `.sdsc/diagnostics/adapted-student-v1/calibration-*`.
The published 7,209-byte full `artifacts/train.log` has independently verified
SHA `8e89ae1970096a1a1583a2628347e2f2280831c65032de9a5eca4d87a041dcb2`.
No new detached Quest supervisor was launched. A separate accepted student-v2
execution correction preserves the original v1 protocol, teacher configuration
and all 47 producer science files; it uses a new two-H100 Accelerate YAML with
explicit `fsdp_use_orig_params=false` and a matching student protocol/config
group. All scientific thresholds, teacher data, training budgets and validators
remain unchanged. Independent implementation review and a new matching real GPU
preflight are required before a fresh calibration; no old preflight guard is
waived and no failed job is retried under its original identity.
The real pinned Accelerate parser/environment/plugin regression reproduces
the original guard failure and confirms false through the corrected actual
worker argv; all other generated environment values are equal. The combined
protocol/consumer/calibration/CLI/remote suite passes **242 tests**, and the
separate unchanged-validator/wrapper suite passes **68**. Independent review
additionally passes 122 cases. Python parsing, Ruff and whitespace checks pass.
Completed calibration `logs`/`fetch` now include bounded training/export log
tails; neither downloads checkpoints. These are CPU/transport checks, not a
new v2 GPU PASS. Implementation commit
`3c1f6f1e9bdc798ecb65399165dd1be313014b42` received independent ACCEPT from
`/root/student_migration_audit` at **2026-09-28T18:31:41Z**, with 123 independent
checks including the bounded-log regression. The v2 review-neutral core SHA is
`9277960e4ff3599c325ac0115888280ad32647891fd3841d045822bf7db2a320`.
Distinct review-only acceptance commit
`28c1026cece772a9e3d167d9cc64a64aaa0fd1b3` changes only the v2 review block and handoff.
It grants no new GPU-result acceptance; execute a fresh matching v2 preflight
and only then a fresh calibration at the same reviewed source HEAD.

V2 release `20260928T183316Z-b542b2e7e7fa-1cd4c8d9` was dry-run reviewed and
uploaded successfully: 577 files / 6,478,953 source bytes, code SHA
`b542b2e7e7fa49f4593aab7f361bb5df1b68c04890eb4cdedd6a51553f58a5cf`,
genuine source HEAD `28c1026cece772a9e3d167d9cc64a64aaa0fd1b3`.
Its local genuine provenance artifact is
`.sdsc/provenance/provenance-ec86aaf33c9040c68676d453af6e8b3e`, manifest SHA
`e2a32137229f1711aa32610d42690afedcd83afad018f0fda74586ed7821bc81`.
The first upload timed out; its empty `preflight-provenance-upload.json` remains
failure evidence. Following manual SSH recovery, read-only reconciliation
verified all 577 deployed source files and found no provenance destination,
active matching uploader, temporary upload directory, run claim or queued job.
The same bounded artifact was then uploaded and verified remotely, with genuine
history, all three files, matching wrapper and read-only permissions. Its actual
receipt is `preflight-provenance-upload-recovered.json`; all evidence is under
`.sdsc/diagnostics/adapted-student-v2/`. No existing release was overwritten.

New v2 preflight **54506703**, intent `2b6229dd119b489d8029724d90c6d3b8`, was
submitted exactly once at **2026-09-28T20:18:24Z**, after matching dry-run and
remote prerequisite verification. Resources are **2 H100 / 24 CPUs / 192 GiB /
one hour**, account `nwu181`, partition `nairr-gpu-shared`, QoS
`nairr-gpu-shared-normal`, no requeue. The receipt binds accepted teacher
`54496291`, science HEAD `28c1026`, the v2 protocol and prerequisite SHA
`8f3b98aa9cad3a02e4524289f157b810b745879bb0309558d96e3cb85b638bb6`.
Job, batch and extern completed **COMPLETED / 0:0** on **exp-19-01**, elapsed
**13m48s**; the final queue is empty and publication verification succeeded.
Independent review at **2026-09-28T20:36:44Z** accepted the actual report using
both unchanged strict validators and checked all 46 student science files
against genuine accepted HEAD `28c1026`. Both ranks completed the global-64,
1536-token update and model/optimizer/scheduler/RNG save-restore checks.
Actual host-memory peak was 103.176 GiB with 88.824 GiB headroom; peak reserved
GPU memory was 44.887 / 29.787 GiB. The ten-file 10,946,233,937-byte publication
includes allocation-side checkpoint read-back evidence; large checkpoints
were not downloaded or rehashed on the login node. Five small files totaling
62,049 bytes were fetched into `.sdsc/fetched/54506703/fetch-xhtkty_a/`.
Report SHA is `307156e33ba88c9a6d98939a4e7a44a661e15278ce4b69ba9cb69e155a132bd7`.
Persistent outputs
use `control-results/20260928T183316Z-b542b2e7e7fa-1cd4c8d9/2b6229dd119b489d8029724d90c6d3b8/`.
The actual local receipt is
`.sdsc/submissions/2b6229dd119b489d8029724d90c6d3b8.json`; command, dry-run,
status, independent review and bounded logs are in the v2 diagnostic directory.
The exact final files are `preflight-54506703-final-status.json` and
`preflight-54506703-independent-review.json`. Never submit this run/intent again.
All 12 historical supervisor states remain stopped/finished.

Later control/documentation commits do not change that release's genuine science
HEAD. The separate `tools/sdsc_release_replay.py` preserves its exact deployed
bytes and genuine Git bundle, creates a fresh run/provenance wrapper, and records
the current preparation HEAD separately. Its 22 fixture tests and independent
review passed. The reviewed dry-run for
`20260928T204012Z-b542b2e7e7fa-41669358` contains the same 577 files / 6,478,953
source bytes; its immutable preparation is under
`.sdsc/replays/replay-1e7986506c3f44dc92c08b0f908e856f/`. Source and provenance
uploads were verified; new provenance SHA is
`eef4035f32d96f0ada77d48257358712c06d215d7a4a244cfa26ad24f0b5fbad`, with the
unchanged original genuine history bundle.
The new finite `tools/sdsc_student_supervise` is restricted to verified preflight
54506703 -> one matching canonical-SFT calibration -> terminal verification and
bounded fetch, with 2 H100 / 24 CPU / 192 GiB / 2 hours, 300-second polling and a
14-day deadline. It preserves all scientific gates and uses a permanent shared
stage claim, immutable plan/control hashes and no automatic retry. The user has
explicitly authorized this automatic transition; no per-stage confirmation is
needed. Actual activation must be recorded from its launch/state/submission
evidence, not inferred from source existence. Its plan is
`.sdsc/supervision/student54506703-to-calibration-v2/plan.json`, SHA
`a9a6cd30dd334803aaa5ffac6bf0521fb90c5567dc1a811883558415eabc48a0`.
Independent review passed all 44 new control tests plus direct checks against
real active/terminal Slurm observations, the fetched report and deployed plan.
The combined control/provenance/student-transport suite passed 114 tests;
Ruff, shell parsing and whitespace checks passed.
The matching submit dry-run passed; it did not submit a job. Instructions are
in `docs/sdsc_student_automation.md`.

Control implementation `93f8f8968776fa5ef3d27201e3fbd12b21e29c3b` received
independent ACCEPT at **2026-09-28T20:45:59Z**; its review is in the flow's
`independent-review.json`. The Quest **quser32** process launched at
**20:46:48Z**, PID **2994231**, independently rechecked/fetched the successful
preflight and automatically submitted exactly one calibration **54506821** at
**20:47:19Z**, intent `bffa12a0cfa5429baa44050fcdc2833f`. Its verified receipt
and resource/science bindings are in
`.sdsc/submissions/bffa12a0cfa5429baa44050fcdc2833f.json`; result path is
`control-results/20260928T204012Z-b542b2e7e7fa-41669358/bffa12a0cfa5429baa44050fcdc2833f/`.
Initial accounting and queue agreed **RUNNING** on **exp-19-15**. Logs show
source/history, teacher inputs, dataset and node-local model staging completed,
followed by the calibration phase; the terminal failure is recorded below.

The original supervisor stopped at **20:47:20Z** when its first calibration
query triggered the unrecognized-state guard; it has exited. The submission receipt is
complete, `submission_outcome_unknown=false`, and no submission was repeated.
The first raw status was not retained because classification raised before
recording; the later real status is
`.sdsc/diagnostics/adapted-student-v2/calibration-54506821-initial-status.json`.
Preserve the original stopped state, plan, launch record and permanent claim;
never re-arm that submitting flow. CPU reproduction through the original remote
status composer and classifier matches this error when the queue is active but
the main accounting row is absent. A roughly one-second post-submit observation
and the later RUNNING result support initial accounting latency as the likely
cause; without the original response it is not direct proof of empty accounting.
Independent evidence is `calibration-54506821-independent-control-diagnosis.json`
in the v2 diagnostic directory.

The separate finite `tools/sdsc_student_observe.py` now binds only this reconciled
job, retains raw status before classification, and permits only status/fetch.
It preserves the original frozen control files, stopped state and permanent
submission claim, shares the existing process lock, adds a distinct permanent
observation claim, and cannot extend the old deadline or submit/cancel anything.
Its eleven new tests and independent review passed, including the actual CLI
subprocess boundary; the combined observer/supervisor suite passed 33 tests and
Ruff/whitespace checks passed. The original unknown-state stop and scientific result gates
remain unchanged. The prepared plan is
`.sdsc/supervision/student54506821-readonly-v1/plan.json`, SHA
`06e8955b475558cd2c110ef478b9e5190590293e8eef4714fa22a75899892535`.
Observer implementation `0577527b5885a5541da270ebed39b6d345d23ca6` received
independent ACCEPT at **2026-09-28T20:55:58Z**. It launched on Quest quser32
at **20:56:40Z**, PID **3056553**, then correctly detected terminal failure,
fetched eight small files / **52,966 bytes**, and stopped at **20:56:44Z**.
Both Quest processes have exited; their original claims and records remain.
The observer's real terminal state is in its flow directory and its fetched
evidence is `.sdsc/fetched/54506821/fetch-we8da5a6/`.

Calibration **54506821** ended **FAILED / 1:0**, elapsed **8m31s**, with the
batch step failed and extern completed. Both GPU ranks reached
`FactorialTrainer._training_micro_step` -> `VerifiedReplaySupervisor.compute_loss`
-> `verified_replay_loss`, then cross-entropy rejected CPU target labels paired
with CUDA logits (cuda:0 / cuda:1). This is a real training device-placement
failure, distinct from the earlier FSDP configuration failure and the first
observer's accounting issue. Initial checkpoint export succeeded; no successful
student optimizer step, calibrated artifact or G0 result is established.
Preserve the failed v2 job, accepted v2 protocol/config and teacher artifacts.
A minimal device-placement correction, CPU regression at the actual supervision
call boundary, independent science review and a fresh matching GPU preflight are
required before the next calibration; do not reuse the passed v2 preflight for
changed science or relax its acceptance gates.

The independently accepted v3 successor changes only the three external loss operands in
`VerifiedReplaySupervisor.compute_loss` to the actual logits device. It retains
their dtypes and the CPU source batch; loss mathematics, gradients, RNG, sample
ordering, teacher evidence and all budgets/thresholds are unchanged. Its GPU
preflight now takes the unchanged 64-by-1536 synthetic canary through actual CPU
`TrajectoryRecord`/collation -> canonical supervision -> GPU forward/loss,
preserving FP32 loss, eight four-sample microsteps per rank and the original
optimizer/checkpoint/restore gates. V3 requires exact per-rank evidence for this
route; v1/v2 report paths remain historical. The new protocol/config preserves
v1/v2 artifacts and the explicit-false v2 Accelerate YAML; its 48 named student
files add only the two newly exercised collation/trajectory dependencies.
All 47 teacher files, including the shared loss implementation, remain unchanged.
V3 core SHA is `c727957ed2fa059481772ebbdef68982a0280436d41cb6be4acd5eff1fd70ec9`.

The parent combined suite passed **189 tests** and independent scientific review
passed **146**; additional trainer/scientific-repair regressions passed **79**.
Ruff has no new findings: `artifacts/runs.py` retains the same pre-existing
I001/F401 import findings reproduced from the previous HEAD; other affected
files and whitespace checks pass. These CPU checks are not a v3 GPU PASS.
The new isolated `tools/sdsc_student_supervise_v3.py` reuses the original finite
supervisor under a hashed, accepted-v3 profile and a new actual preflight receipt.
It preserves the old supervisor source and every original claim, waits one
five-minute interval after persisting a known calibration receipt before the
first calibration query, and saves raw status before classification. Unknown
state still stops; no submission is retried and no G0/pilot is activated.
Implementation `6ae57f0ef0a4cd594f1692ed1550c826dc9b37ee` received overall
independent ACCEPT from `/root/student_migration_audit` at
**2026-09-28T21:13:16.995421Z**, incorporating the separately reviewed control
adapter and its 29 passing tests. The exact review is preserved under
`.sdsc/diagnostics/adapted-student-v3/implementation-independent-review.json`.
Distinct acceptance commit `df05bd2a96363829a4bc587c04a2dd116e2a6242` changes only
the v3 protocol review block and this handoff; the actual accepted resolver passed.
SSH/account/partition/QoS/path checks passed at **21:11:03Z**; a fresh
metadata-only runtime check also confirmed the original Python binary SHA and
five package pins without importing GPU libraries.

Fresh v3 preflight **54507345**, intent `b11f4fc0c8d9458c836b3d618d102696`, was
submitted once at **2026-09-28T21:23:04Z** with **2 H100 / 24 CPU / 192 GiB /
one hour**, account `nwu181`, shared partition and shared-normal QoS. Release
`20260928T211432Z-9f932bc1929e-1f1a5883` contains 590 files / 6,676,419 bytes,
code SHA `9f932bc1929e7a549cea64271a089786894b94c7c381c288904ac9fe46090d5b`.
Genuine provenance SHA is
`9c16ed8cd034ff2c1c0313f8043fe409bcd6aa3812b3858ef2fca696467cd6e9`, binding
accepted HEAD `df05bd2`; source and provenance uploads were verified. Its real
receipt binds teacher `54496291` and prerequisite SHA
`ebfa796845102f785485ce56f53b2a638160212eef7abf7036fecd3c0d7e0086`.
It completed on **exp-19-08** in **14m09s**; job/batch/extern accounting is
**COMPLETED / 0:0**, the queue is empty and persistent report validation passes.
Fresh SSH/accounting on **2026-09-29 at 00:47 UTC** confirms this result. Report
SHA is `aaad02a748df41c76c6c41b6e06a83ecb0bc1cefa4d69c3f25d1e1eed5fe53d2`;
bounded artifacts are in `.sdsc/fetched/54507345/fetch-pcjfwi24/`, final status in
`.sdsc/diagnostics/adapted-student-v3/preflight-54507345-final-status.json`.
Both ranks passed the actual CPU collator/canonical-supervision boundary with
CUDA loss operands, finite gradients, a nonzero update and full-state restore.
This validates the device correction's synthetic preflight, not complete training.

The distinct calibration release `20260928T212233Z-9f932bc1929e-028c319b`
replays exactly the same 590 source files and original genuine history bundle.
Its source and new provenance upload are verified; provenance SHA is
`0497c4e0dcc7ca8ae55c1f8cb51cad3cdf5119fa505ed9d3b46eeb44aa3bc287`.
Preparation/deployment evidence is under
`.sdsc/replays/replay-3095a0c5b6b94b66bc0288b261561948/`.
The actual finite flow is
`.sdsc/supervision/student54507345-to-calibration-v3/`, plan SHA
`4d5cfc581ff3a3051b829f49feeb0e1eab081ba5283a04e9663a98fbf5f1bd03`.
It received independent operational ACCEPT from `/root/student_provenance_update`
at **21:26:37Z**, including full genuine-provenance plan reconstruction, all
source hashes and the exact **2 H100 / 24 CPU / 192 GiB / two-hour** calibration
dry-run. It launched on **quser32**, PID **3159562**, at **21:27:14Z** and
automatically submitted exactly one calibration **54507464** at **21:43:06Z**,
intent `0c9f816146a2426f836e0320ae1f2986`, prerequisite SHA
`a35bee52559e858117883a4bb921e5a0cc93e0e5da988e34722a3ec0d9f486fe`.
Fresh accounting confirms **FAILED / 1:0**, elapsed **5m10s** on exp-19-08;
batch failed and extern completed. Both ranks reached backward and optimizer
boundary finalization, then raised `checkpoint AdamW state is empty after an
optimizer update`. This is distinct from the repaired CPU/CUDA-label failure.
The published training metrics file is empty and there is no accepted training
checkpoint or successful parameter-update evidence. Real pinned Accelerate/FSDP
CPU reproduction now establishes stale optimizer parameter references after
FSDP1 flattening; do not weaken the validator.

The finite supervisor stopped at **2026-09-28T21:53:09Z**, fetched eight files /
**55,050 bytes** to `.sdsc/fetched/54507464/fetch-t2d4w129/`, and exited. Its
actual state is `stopped`, `submission_attempted=true`,
`submission_outcome_unknown=false`, `no_retry=true`. No job was cancelled or
repeated. Preserve the plan/profile, receipts, permanent preflight claim and
failed output; never re-arm this submitting flow. It performed the authorized
automatic transition and failure capture, not automatic code repair or chat
notification. A reviewed repair and fresh matching prerequisites are required
before any new calibration; neither G0 nor pilot has passed.

The independently reviewed v4 correction preserves the same fresh single-group AdamW and raw
LambdaLR while preparing FSDP before binding optimizer parameters. It rejects
stateful/ambiguous inputs and leaves non-FSDP preparation unchanged. The actual
trainer CPU regression reproduced the old empty-state failure and now reaches a
nonzero global64 update with real AdamW state. Its communication fixture uses a
single-rank FakeProcessGroup; this is not GPU acceptance. V4's GPU preflight now
calls the same shared helper through the real Accelerator on Qwen3-1.7B,
retaining the synthetic global64 window, device-boundary test, unchanged cadence
validator and full-state restore. A separate Gloo control group preserves CPU
reductions and monitored barriers alongside production NCCL FSDP.
The v4 protocol/config has core SHA
`822b5640da8e47232ce09795e364c73fae16512f1123d2a0b4dfa56c1769f93c`.
Historical v1/v2/v3 and all 47 teacher scientific files remain unchanged.
The separate v4 finite supervisor retains the reviewed single-submit/stop guards.
Genuine provenance's fixed unpublished-history capacity becomes 64, retaining
every original byte/path/lineage check; real Git boundary and hidden historical
path tests pass. Fresh v4 deployment and execution are recorded below; the
earlier v3 result cannot satisfy the changed implementation's prerequisite.
The combined affected suite passed **289 tests**; the actual CPU FSDP suite
includes 13 cases and the original trainer suite 70, covering raw optimizer /
scheduler identity, parameter ownership, nonzero update, RNG and full-state
save/restore followed by an identical next window. Independent control review
passed **63 tests**, including real 64-commit export/restore, 65-commit rejection
and rejection of unsafe historical paths after the old 32-commit boundary.
No new Ruff or whitespace findings were introduced. Existing unrelated source
lint findings and the user's dirty files are preserved. Shared SSH and remote
account/QoS/path checks passed at **2026-09-29T00:58:16Z**, and the fixed Python
SHA and five package versions were rechecked without importing GPU libraries.
Implementation `89a8ffd598d9377ebcbef556bee0057699d9eb35` received independent
ACCEPT from `/root/student_migration_audit` at **2026-09-29T01:05:11.976586Z**,
with **235 independent scientific tests**, all 48 student blobs / 47 frozen
teacher blobs verified, and the separate 63-test control review. The evidence is
`.sdsc/diagnostics/adapted-student-v4/implementation-independent-review.json`.
Distinct acceptance commit `cfac02db4cf67c7d1a8c1b09697fcc25de76474b` changes
only the v4 review block and this handoff; the actual accepted resolver passed.

V4 preflight **54509682**, intent `7f33e6c018224108a41100a496879017`, was
submitted once at **2026-09-29T01:13:07Z** with **2 H100 / 24 CPU / 192 GiB /
one hour**, account `nwu181`, `nairr-gpu-shared` / `nairr-gpu-shared-normal`.
Release `20260929T010618Z-f8208cbe6d74-644c0f92` contains **595 files /
6,739,611 bytes**, code SHA
`f8208cbe6d74e6c54d728417a7ba930e4fbb94b0d6796d9b0f35c4d3d1a3a6d2`.
Its verified genuine provenance is
`332c96ea0624040d57f6ad0f77166b37fe8c1a5ea1cfb1d71ce2b0c5ff997746`;
the actual receipt binds accepted HEAD `cfac02d`, teacher `54496291` and
prerequisite SHA `22f816676fcce32b1c5b15e68823182d53f6f3a3c2c5725e7275d38be064e210`.
It completed on **exp-19-15** in **16m44s**. Final accounting shows main,
batch and extern **COMPLETED / 0:0**, an empty queue and verified report SHA
`cbc7fe0b3a59d97130e6ddc1ec1aca3368a25d2c876073d72b140468fe9600d1`.
Both ranks passed the actual Accelerator preparation, prepared-parameter
ownership, nonempty FP32 AdamW state, unchanged step/scheduler cadence,
nonzero global64 update and complete model/optimizer/scheduler/RNG restore.
Publication read-back verified **21,892,025,091 bytes** across ten files on persistent storage;
only five small reports/logs (**65,179 bytes**) were fetched to
`.sdsc/fetched/54509682/fetch-rcogualm/` (the supervisor independently fetched
the same small set to `fetch-4zf_mnf5/`). Final accounting is in
`.sdsc/diagnostics/adapted-student-v4/preflight-54509682-final-status.json`.
Independent actual-result ACCEPT from `/root/student_migration_audit` at
**01:36:11Z** re-ran the strict v4 report/upstream/accounting/fetch validators,
verified the 48 deployed science files and cross-linked all four checkpoint
file sizes/hashes to the persistent publication inventory. Evidence is
`.sdsc/diagnostics/adapted-student-v4/preflight-54509682-independent-review.json`.
Host memory peaked at **131.248 GiB**, leaving **60.752 GiB** under the
192-GiB cgroup limit; GPU reserved peaks were **41.902 / 30.229 GiB**.
This proves the shared preparation and synthetic window, not full calibration.

The separate calibration release `20260929T011138Z-f8208cbe6d74-172ab29a`
replays those exact bytes and unchanged genuine Git bundle. Its new verified
provenance SHA is
`21df6ab7d3d0a9e0c71ccf06c23214600b67ce50f55409d4bdd94f541365394a`;
evidence is under `.sdsc/replays/replay-de754faa64734356b8a9cf60d8af7ce1/`.
The new flow is `.sdsc/supervision/student54509682-to-calibration-v4/`, plan SHA
`c5eb887c717ee8b61fdb5cb04313cf427db127afb12ee5944994f7229f0de494`.
Independent operational ACCEPT from `/root/student_provenance_update` at
**01:15:48Z** verified actual receipts, all 595 source files, genuine provenance,
the complete rebuilt plan and an empty-blocker **2 H100 / 24 CPU / 192 GiB /
two-hour** calibration dry-run. The finite supervisor launched on **quser32**
at **01:16:36Z**, PID **3739259**. It verified preflight success and automatically
submitted exactly one calibration **54509809** at **01:32:18Z**, intent
`ceafe10f2b974e8d986688d56b114079`, prerequisite SHA
`7227a7bf5871b0e4346e037a1cb7e3dbaea92673cf1ecdcd6435a8fe649ea2dd`.
The supervisor's preserved final accounting records **FAILED / 1:0** after
**26m38s** on **exp-19-15**, with batch failed, extern completed and an empty
queue. Both ranks reached the first **step-20 checkpoint** publication and
raised `baseline/final model tensor metadata differs: lm_head.weight` from
`model_update_evidence`. This has passed the earlier first-update AdamW failure
boundary, but is not a successful calibration. Fresh bounded metadata evidence
below confirms uniform BF16→FP32 promotion rather than any key/shape change.
The successor corrects the explicit precision contract without bypassing the
checkpoint validator.
The receipt records 28 persistently read-back-verified files / **25,354,722,334
bytes**, including nonempty **37,784-byte** training metrics and the temporary
`.step-00000020.accelerate.stage-*` model/optimizer/RNG state. The public
step-20 checkpoint was not committed and the wrapper reports `resumable=false`;
do not treat temporary staging files as an accepted resume point. Metrics were
subsequently fetched and content-hash verified, as summarized below. Initially,
eight small files / **59,474 bytes** were fetched to
`.sdsc/fetched/54509809/fetch-umf99ie0/`.
Results belong to `/expanse/lustre/projects/nwu181/zgao12/OPD/control-results/20260929T011138Z-f8208cbe6d74-172ab29a/ceafe10f2b974e8d986688d56b114079`.
The supervisor recorded `stopped` at **2026-09-29T02:02:24Z**, with
`submission_attempted=true`, `submission_outcome_unknown=false`, `no_retry=true`
and no cancellation. The current local supervision inventory has no active
flow. Preserve the original plan, stage claim, receipts and failed output;
never re-arm this flow or repeat its submission. It does not repair code or
automatically notify the chat. A reviewed checkpoint-contract correction and
fresh matching prerequisites are required before another calibration.

After the user restored authentication on **quser44**, the existing shared
master passed `ssh -O check`; `tools/sdsc check` at **2026-09-30T04:40:27Z**
verified zgao12@login02, Slurm availability, paths and account/partition QoS
`nairr-gpu-shared-normal`. Fresh accounting independently confirms 54509809
FAILED / 1:0 after 26m38s with an empty queue entry. No login retry occurred.

Bounded, inert checkpoint-metadata inspection found all **311 tensors** have
identical names/shapes and uniformly change **BF16 initial → FP32 final**.
The original publication receipt matches the retained receipt. This inspection
read pickle metadata only, did not execute checkpoint globals/load tensor
storage/import Torch, and did not rehash or download the large weight files.
Hash-verified small metrics contain **20 updates / 1,188,770 of 2,000,000 input
tokens**. Step-20 loss is 2.681661978; answer/proof/format validation are all
**0.0**. They are disclosed failure-stage measurements, not student quality
acceptance or authority to change hyperparameters/thresholds.

The independently accepted v5 repair preserves physical initial checkpoint bytes,
all teacher evidence and scientific settings. Its explicit policy compares
fresh BF16 values with FP32 masters, and resumed FP32 with FP32, using FP64
deltas and hashes of actual final bytes. Promotion alone remains zero; exact
inventories/shapes, finite values and valid ancestry remain mandatory. Both
trainer publication and independent finalization use the same narrowly gated
policy. Real CPU Accelerator/FSDP save, finalizer, resume and next-update
regressions pass; these are not two-H100 evidence. V5 GPU preflight adds a
frozen physical BF16 baseline, positive first-update evidence and exact FP32
full-state restoration. It makes no real-GPU next-update equivalence claim.
Independent review accepted implementation `529d46eceab6fca7bf9bb73fc79e82101fbd0fd1`
at 2026-09-30T05:00:02Z; distinct review-only acceptance is
`9169a61e452c0b94064b06e6a449abfff63857f1`. Protocol core SHA is
`14c79d22ae6016140591ee15dd9bb84cafcab53bd86619350c874bb0b1b964e2`, physical
accepted artifact SHA is
`c3e2f33408db18be07d6ec855752ea1c12461958528ac5380ee7b1303dd620e3`.
The matching v5 preflight below passed before the single new calibration.
Preserve all old stopped flows and failed staged checkpoints.
Evidence is under `.sdsc/diagnostics/adapted-student-v5/` and the three verified
small files in `.sdsc/fetched/54509809/verified-small-v5/`.

The new v5 preflight **54533796**, intent
`4e76f478490a4a0fa0e0547b1fb1414b`, was submitted exactly once at
**2026-09-30T05:04:33Z**. Its actual receipt binds accepted science HEAD
`9169a61e452c0b94064b06e6a449abfff63857f1` and teacher 54496291. Resources are
**2 H100 / 24 CPU / 192 GiB / one hour**, nwu181,
nairr-gpu-shared / nairr-gpu-shared-normal. Fresh accounting confirms
**COMPLETED / 0:0 after 20m09s**, with an empty queue entry. Independent strict
validation accepted the actual report and all five persisted precision
checkpoint identities. Report SHA is
`69fba89bd302b598b89188aaa32fdd838f2ee763d9f4af400c12de082e911585`;
evidence is `.sdsc/fetched/54533796/fetch-n5rakjzd/` and
`.sdsc/diagnostics/adapted-student-v5/preflight-54533796-status-20260930.json`.
The preflight's aggregate cgroup peak was 126,727,565,312 bytes (118.02 GiB),
with 73.98 GiB headroom; this is not the subsequent calibration's memory peak.
Run `20260930T050103Z-a99b8a463429-c1902767` contains 600 files / 6,825,905
source bytes after matching dry-run; deployed source SHA is
`a99b8a463429fcbdd0d43978b41e6072cc7319e79b1dfbb0e0bfb03cddb57e9f`.
Verified genuine provenance SHA is
`5c74d06e32fe1cb48a4f74a809dc008036ab02222fce7d98df21076b56f99e75`.
The existing runtime's Python executable SHA and five direct package metadata
pins were freshly rechecked without importing GPU libraries or changing it.

The separate deployed calibration release
`20260930T050336Z-a99b8a463429-8b09fc13` replays the same source bytes and
original bundle. Its verified provenance SHA is
`f7c3f4864a0565115d9c7badfbfdb9f9c87608a4c5c46f953318fe286d88ddc4`.
Replay evidence is `.sdsc/replays/replay-3a223cb3990240d4a88c75d476a59094/`.
The new finite plan is
`.sdsc/supervision/student54533796-to-calibration-v5/plan.json`, SHA
`71658ec65453e6d6d90b505e31f4aa37dd095fdfc5a5899136c00e20c09d6fd0`;
profile SHA is
`9689c0c0f326dee427211b27696eab81e394557ba9cd272b2fa8fd9195b7f8a0`.
It allows one **2 H100 / 24 CPU / 192 GiB / two-hour** calibration only after
strict v5 preflight acceptance, at 300-second intervals for at most fourteen
days. Its calibration dry-run has no blockers; actual remote prerequisites
are reverified before any claim or sbatch. Independent operational review
accepted the actual rebuilt plan, all 600 deployed source files, nine evidence
pins and launch script; review SHA is
`6005ccd83548346481fee0613f7417d210f711c5cd94e86abd20e1a1b2cedbac`.
The finite supervisor launched once on **quser44** at
**2026-09-30T05:10:36Z**, PID **1076537**, process start ticks **20293972**.
After strict preflight acceptance it submitted calibration **54533934** once at
**2026-09-30T05:27:12Z**, intent `d27fa16c0fe2425d9bb1041f6d326295`, using the
reviewed two-hour profile and deployed calibration release above. Fresh SSH
accounting from quser43 confirms **FAILED / 1:0 after 34m42s**; main/batch
failed, extern completed, and the queue entry is empty. The supervisor fetched
bounded terminal evidence and **stopped at 2026-09-30T06:02:21Z** (01:02 CDT).
Its state records `submission_attempted=true`,
`submission_outcome_unknown=false`, `no_retry=true` and `jobs_cancelled=false`.
It is not an active monitor or authorization to re-arm the stopped flow.

The scientific training subprocess completed **33 optimizer updates**, consuming
**1,961,368 of 2,000,000 non-padding input tokens**. The remaining 38,632 tokens
could not admit another exact global-64 optimizer window, so the normal token
budget stop applied. Final update loss was **0.2498967983**; step-20 validation
answer/proof/format rates were all **0.0**. There was no step-33 quality
evaluation. These measurements do not establish student quality acceptance.
Real step-20 and step-33 checkpoints were published, and
`validate_factorial_run_artifacts` returned successfully before the wrapper's
final `memory_envelope()` check raised
`192 GiB cgroup lacks 32 GiB and 20% headroom`. Thus the v5 precision repair
crossed the real training/save boundary, but calibration acceptance failed.
The exact final aggregate peak was not recorded because the memory function
raised before assigning `final_cgroup_memory`; peak exceeded the approximately
153.6 GiB acceptance ceiling, but its exact value and anonymous/file-cache
contributions are unknown. Slurm batch MaxRSS is not the aggregate cgroup peak.
There is no demonstrated host/GPU OOM or evidence yet attributing this to cache.

The publication receipt records **36 files / 91,031,115,461 bytes** verified on
persistent storage under the calibration release's intent directory. Both
checkpoints, corresponding Accelerate model/optimizer/scheduler/RNG state and
final update evidence remain there; wrapper `resumable=false` remains in force.
Bounded terminal fetch is `.sdsc/fetched/54533934/fetch-6b5bekn4/`. An additional
receipt-size/SHA-verified read fetched only the inner calibration report,
`metrics.jsonl` and `factorial_update_evidence.json` into
`.sdsc/fetched/54533934/verified-small-status/`; no weights were downloaded.
Preserve these outputs; the accepted v6 repair below adds the missing stage
measurements and a larger independently reviewed envelope for a fresh run. Do not waive the headroom gate, silently change
the reviewed memory envelope, retry this intent or treat checkpoints as an
accepted resume contract. Inspect stopped state and receipts before any
continuation. Existing stopped flows and unknown-intent protections remain.

The v6 repair is a separately accepted successor, not a reinterpretation of
54533934. A real cgroup fixture reproduced the information loss: 160-GiB and
191-GiB peaks under the old 192-GiB limit both raised the same error without
recording the peak. The actual failed job's node-local mount was ext4 on
`/scratch`, ruling out the TMPDIR-as-tmpfs hypothesis. The exact historical peak
and whether training, checkpoint I/O/cache or final validation dominated remain
unknown. The new student-only `tools/sdsc_student_memory.py` preserves raw
ancestor limits/current/peak, memory.stat and available OOM/event counters on
both success and failure. It never resets counters, drops caches or subtracts
file cache. Calibration records initial, post-export, post-training,
post-validation and failure observations before raising; bounded v6 fetch also
includes the inner report and phase measurements. The frozen teacher memory
helper and all 47 teacher scientific blobs remain unchanged.

The newly authorized envelope is **2 H100 / 24 CPU / 384 GiB**, with one-hour
preflight and two-hour calibration limits; two-rank/global-64 semantics,
teacher inputs, checkpoints, seed and all scientific thresholds are unchanged.
The memory gate remains `max(32 GiB, ceil(actual_limit * 0.20))`, so the new
required headroom is 76.8 GiB. CLI admission permits only the two named student
profiles; verified protocol bytes select 192 GiB for v1-v5 or 384 GiB for v6
before any remote claim/sbatch, and the worker checks the real Slurm allocation.
Fresh quser43 SSH inspection confirmed an empty account queue, partition
capacity and QoS maxima of 762 GiB / three GPUs / 54 CPUs per shared job. The
existing pinned runtime was checked without importing Torch or installing
packages. Discovery/reproduction evidence is under
`.sdsc/diagnostics/adapted-student-memory-v6/`. Independent acceptance binds
implementation `9a196677a269c2e5f1c925da4e43b6145c1cdd7a`; core protocol SHA is
`17a8c3729f60df2f56219d5e8a35427e068c335e22609161830a2630d85654ba`.
The independent review artifact SHA is
`cd1b7f1afee808c16a607bdf8010baa40d762ca4db7607713f2ec47bf5a16f5c`.
The accepted v6 protocol still requires new releases and matching preflight; `tools/sdsc_student_supervise_v6.py` is a finite
one-calibration adapter with 300-second polling and the existing no-retry,
unknown-intent and SSH-loss stops. Its actual launch is recorded below.
The integrated CPU suite passed **447 tests** (24 expected PyTorch FSDP warnings),
including genuine tiny-model save/finalization/resume for both v5 and v6,
cgroup v1/v2 and failed-stage preservation, exact resource/protocol rejection,
and real remote-fetch selection through CLI disk publication. AST and
`git diff --check` passed. CPU tests do not establish the new GPU envelope.

V6 review-only acceptance is `6c04f804b302184b8ff95d00fab404e0531ed8d6`;
physical protocol SHA is
`c701dde9691210dcdd06f4a1076299941fdb6ad8d609280a13e80d5d7a4333f7`.
New preflight **54547547**, intent `dfb7a4ac9a654e37a1e8a9e8dacdc283`, was
submitted exactly once at **2026-09-30T17:58:25Z** with the reviewed one-hour
profile. It completed on **exp-19-07** in **16m38s**; fresh remote accounting
confirms job/batch/extern **COMPLETED / 0:0**, and strict result verification
passed with report SHA
`e978c8d8bfa729f8d8b626e6f478220fdcadc2a60e6e5e46f075826a43d61cb4`.

Preflight release `20260930T175511Z-ea06908ff9b4-91689b38` contains **607 files /
6,914,260 source bytes**, all from the matching dry-run. Source SHA is
`ea06908ff9b4a74e97b8d668584b02c2e76ee68da4861d9ed7dbe5462eb33855`;
genuine bundle SHA is
`4a26fe107997e01d93be9a22d37d1d94a9f7efd4c254b794eb8185328ea8ac10`;
preflight provenance SHA is
`eac360dcca0b7096f426fde82226709aee6af1d862d57c6ef51a6f67df017c23`.
The independent calibration replay `20260930T175809Z-ea06908ff9b4-c0d85d6c`
is deployed with identical source and bundle and new provenance SHA
`ce0f4d1c1f4c292e4c3e88492435a3c2a2da68c2109541e2c6ac3b7d6941ccc4`.
Evidence is `.sdsc/replays/replay-752eafdfccd64070ab1ccc590354b0c4/`.

Actual finite flow: `.sdsc/supervision/student54547547-to-calibration-v6/`.
Plan SHA is `92376fcd00a3cf3f8c94e1cc04c52b0fb77339320b3239f67860ac65fcdd3b63`;
profile SHA is `adc1e59926658512c8cc8fc221b220d01e6290d33ca95596ac5c91ab977516b0`.
Independent operational review SHA is
`f865f700e9341d3be055f24045ddb55aee89ebfd79c0fa8aca882c4309ada733`;
it verified all 607 deployed files, 14 controls, nine evidence pins, the rebuilt
plan, real receipts and exact launcher. The process launched once on **quser43**
at **2026-09-30T18:04:30Z**, PID **1272285**, start ticks **24937236**. After
strict preflight acceptance it submitted exactly one matching
2-H100/24-CPU/384-GiB/two-hour calibration, **54548846**, at
**2026-09-30T18:19:57Z**, intent `594d09aee22c4bdabbe29b0501705344`. It reached
`calibration_complete` at **18:50:07Z** and exited; the original PID is absent.
No retry, cancellation or G0/pilot submission occurred. Preserve its completed
plan, claims, receipts and pinned evidence; never blindly re-arm it.

Calibration **54548846** has fresh job/batch/extern **COMPLETED / 0:0** accounting,
elapsed **29m31s**, and verified report SHA
`91794e528ec83ff0c7f35bd3734d36022d60db2794c3e41c7686beb97fbff4f1`.
It performed **33 full-parameter updates**, consuming **1,961,368 / 2,000,000**
nonpadding model-input tokens and stopping normally before the next global-64
window would exceed the exact budget. The last update loss is **1.2227404267**.
Step-20 answer/proof/format scores are all **0**; there is no step-33 quality
evaluation. Do not infer student-quality acceptance from execution success.

All four memory stages passed. The actual aggregate cgroup peak was
**204,239,155,200 bytes (190.2125 GiB)** after training and remained identical
after final artifact validation. Under the **412,316,860,416-byte (384-GiB)**
limit, headroom was **193.7875 GiB**, above the unchanged **76.8-GiB** minimum.
These observations establish the successful v6 envelope, not the unrecorded v5
peak or the exact operation responsible for either peak. Sampled memory.stat
cache/RSS counters are not the historical peak's composition.

The verified persistent receipt lists **40 files / 91,031,405,690 bytes** under
`/expanse/lustre/projects/nwu181/zgao12/OPD/control-results/20260930T175809Z-ea06908ff9b4-c0d85d6c/594d09aee22c4bdabbe29b0501705344`.
The final step-33 checkpoint SHA is
`d06052e51bbc80ada401110f2fe2e335dbcae5732cf1e0a9cfdee5b637e7f538`.
The supervisor fetched **13 small files / 442,306 bytes**, without weights,
into `.sdsc/fetched/54548846/fetch-rhidqpza/`; fresh terminal status records are
in `.sdsc/diagnostics/adapted-student-memory-v6/terminal-status-54547547.json`
and `terminal-status-54548846.json`. Only documentation changed in this status
inspection; no new compute was submitted and unrelated existing `.gitignore`,
`AGENTS.md` and untracked files remain unstaged.

Keep both old v3 flows stopped. Calibration does not complete G0 or certify
multistep resume/pilot/Blackwell execution; their historical teacher adapters
still need separately reviewed migration. Details and next commands are in
`docs/refactor/sdsc_adapted_student_calibration_20260928.md`.

The new PEFT 0.17.1 runtime is
`/expanse/lustre/projects/nwu181/zgao12/OPD/envs/qwen3-v2-teacher-adapt-peft0171-v1`,
a small system-site-packages venv over the unchanged fixed G0 base. Its wheel
SHA is `3d129d64def3d74779c32a080d2567e5f7b674e77d546e3585138216d903f99e`;
Python binary SHA remains `2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19`.
Installation/pip check passed; the 03:10Z Quest quser33 shared-master check
verified zgao12@login02, exact runtime metadata and project paths. These are
not GPU training or GPU-node mount results. Evidence is under
`.sdsc/diagnostics/teacher-adaptation-v1/` and remote
`bootstrap/teacher-adapt-peft0171-v1/receipt.json`.

Before GPU submission, CPU regeneration matched all seven original split files'
exact SHA/size (144,000 examples) and found no semantic, example-ID, pair-ID or
seed overlap with the independently generated 256-fit/32-dev preflight set.
Its raw seed namespaces begin at 70,000,042 and 80,000,042; all targets verify
and fit the existing token envelope. The future 8,192-fit/512-dev proposal is
separate: 60 fit prefixes need 1,247–1,256 tokens, while all full inputs remain
at most 1,419, below 1,536. Preserve those examples; the future teacher-fit
prefix bound is explicitly 1,280 and does not alter original experiment bounds.
The complete expanded 8,192/512 audit now also passes all 144,000 original rows,
all four identity dimensions and exact original file hashes, preserving all
8,704 canonical targets and 60 long fit prefixes. Evidence:
`.sdsc/diagnostics/teacher-adaptation-v1/full-fit-isolation-cpu.json`, SHA
`19300d9e725264ad62aa32903b0cc8454024b583d7f79555296440d704c8938e`.
Its canonical dataset-manifest SHA is
`742b62a1ee328c8d8f660106265e4a342458fe5145243ecb08368eaa745f502d`.

The accepted teacher-adaptation protocol gives only the adapted teacher a 256-token
development/confirmation/readiness budget. All eight numeric thresholds and
all rows remain; student rollout/evaluation stays at 128. This is a declared
scientific budget intervention, not a correction to the historical 128-token
protocol and never evidence of an original-128-budget PASS. The 512-update,
four-epoch fit evaluates merged checkpoints at 128/256/384/512 and selects the
first passing all eight gates on all 512 development examples, without holdout
reselection. Before this fit, a separately reviewed real four-H100 DDP
preflight must verify global-64 updates, same-world checkpoint reload, export,
development measurement and persistence. Repaired real preflight 54494477
passed these execution gates; full fit 54494742 subsequently completed and
selected its first development-PASS checkpoint as recorded above.
Execution evidence alone does not constitute teacher scientific acceptance.
The fixed supplemental/formal cohorts, complete teacher store, independent
raw-evidence audit and separate teacher-result acceptance now all pass.
Protocol and acceptance conditions: `docs/refactor/sdsc_teacher_adaptation_20260928.md`.

The protocol implementation and separate review-only acceptance are complete.
Teacher-result acceptance and persistent publication are complete. The student
calibration adapter is independently accepted; its matching new GPU preflight
is recorded above. Full G0/pilot adapters still need coherent migration.
Any teacher model/generation-policy change needs its
own reviewed scientific proposal; never rewrite generated outputs or relax
thresholds. The independently reviewed student implementation raises the bounded
provenance history limit from 16 to 32 audited linear unpublished commits,
retaining the genuine final implementation/review pair and all other export
checks. Boundary tests accept 32 and reject 33; no public ref or history moved.
The deployed 16-commit qualification bundle remains valid.
Old preflight 54345604 does not match new
scientific inventory: calibration needs a matching new preflight or separately
reviewed compatibility change. Old v3-pinned adapters/plans must migrate
coherently before any full successor is armed.

The new G0 adapter separates the actual runtime initial-checkpoint hash from the
unchanged accepted base scientific projection and retains all 26 original
scientific checks. Two explicitly named SDSC invocation/review checks replace
the old backend/certificate checks; `execution_class_certified=false`, no fake
ServerScheduler context. Cross-job full-state resume uses real Singularity
private binds to preserve original absolute checkpoint paths and bytes. Login
probes proved host Python/Torch loading, identical inodes through the old deep
scratch path, and exact comma-separated CUDA visibility transfer. These probes
used no GPU and do not substitute for the actual two-resume G0 checks.

Pilot retains the original four-rank batches and all eight seed-42 methods.
The original 256-prompt G0 teacher/bank cannot satisfy its 4096-prompt contract:
16 real one-GPU teacher array tasks (`0-15%4`) use the original per-candidate
seed function, then merge complete ledgers in original order and build the
4096-prompt bank. Training uses real `0-7%1` four-GPU arrays and real accounting.
The new pilot input admission explicitly binds that larger bank to the passed
G0 report without modifying it; all other original finalizer checks remain.
Legacy pilot recovery compares the states the original legacy trainer actually
writes, plus all four native Accelerate rank RNG states; it does not invent
G0-only allocation-neutral checkpoint fields.

Four-card resources were discovered live: `nairr-gpu` with
`nairr-gpu-normal`, account nwu181; shared tasks retain
`nairr-gpu-shared` / `nairr-gpu-shared-normal`. Pilot runtime preparation uses
`envs/qwen3-v2-pilot-trl0222-overlay-v1`, a small independent SDSC venv reading
the verified G0 base plus pinned TRL 0.22.2. Never mutate the running G0 base.
The slow whole-Conda-copy preparation was stopped after a real lightweight
venv probe succeeded; the unused partial target is not a valid runtime.
Fixed MIB revision b759df34433c9e31043ba9e02908ce0bf20e894f and submodule
source inventories are separately recorded. Preparation is now complete:
remote `bootstrap/pipeline-v2/environment.json` and local
`.sdsc/pipeline-environment.json` report all 20 pins, pip check, offline pinned
model configs/tokenizers, MIB/submodule identities and clean source passed.
The new venv plus its base also passed actual readonly container mount/import
checks (`.sdsc/pipeline-container-probe.json`), without using a GPU. The base's
19 dependency versions were rechecked unchanged after installation.
Default-thread import verification had stopped making progress; a bounded
single-thread full-import comparison passed. Only that no-GPU checker was
stopped, followed by complete `--verify-prepared` validation without reinstall.
All four OMP/MKL/OpenBLAS/NumExpr thread limits are now explicit before imports:
one for login preparation and 24 divided by GPU count for stage ranks.

The expanded SDSC CPU suite, actual original CLI parser/configuration fixtures,
deterministic teacher partition-equivalence tests, accounting/receipt/state
machine tests, and independent code reviews cover the new path: **380 CPU tests
passed**, with 14 dependency deprecation warnings; Ruff, shell syntax and diff
checks passed. Final verification and deployed identities are in the flow's
review/launch metadata. No CPU fixture is a real G0 or pilot result; this teacher
run failed coverage and the full multi-stage GPU path remains unexecuted.
The deployed historical flows retain genuine science HEAD 0215c356. The new
prompt repair is a separate proposed successor; inspect Git and the current
review block for its implementation/acceptance state. Existing workflow tools
and docs remain separate from the committed scientific source.

The initial tooling/check/dry-run stage is complete. The user subsequently
explicitly authorized uploading a snapshot and executing one infrastructure
smoke with one H100, four CPUs, 16 GiB and at most five minutes, including
status/log/result checks and fetch. Do not ask again for this same authorization;
that earlier smoke-only scope has been superseded by the formal-training
authorization above. Quest source remains at
`/gpfs/projects/p32737/del6500_home/OPD`; no Codex/editor server or workflow
daemon is installed on SDSC. Historical ServerScheduler retry authorization
below does not authorize an Expanse job. Existing server/Slurm scripts,
environment configuration, scientific code and preregistrations are preserved.

`tools/sdsc` supplies check, snapshot preview/upload, explicitly authorized
submission, receipt reconciliation, status, bounded logs/fetch and cancellation.
Its original executable workload is an infrastructure smoke bounded to one H100,
four CPUs, 16 GiB and five minutes, account nwu181, partition nairr-gpu-shared,
and QoS nairr-gpu-shared-normal (corrected from the initially supplied value
using the live account/partition intersection). This is not G0 or a scientific
execution-class certification.
Snapshots bind working-tree bytes, including eligible uncommitted/new files;
separate releases and atomic submission claims prevent overwrite or duplicate
submission. Jobs verify/stage source on node-local storage and validate small
persistent results before success. HOME results are limited to the dedicated
smoke-results metadata directory, never training data or checkpoints.

After manual authentication, the real check on Quest quser44 successfully reused
`/home/del6500/.ssh/cm/sdsc-quser44` and verified zgao12@login01. Latest evidence
is `.sdsc/check.json` (mutable; the formal runtime check supersedes the original
2026-09-18T01:38:54Z container-only check). Slurm tools,
account association and quota resource expanse_nairr_gpu were verified. Host
Python is /usr/bin/python3.11 (3.11.5, no Torch); module anaconda3/2021.05 exposes
only shared base Python 3.8.8 without Torch. Reuse the observed SingularityPRO
4.1.2 executable /cm/local/apps/singularitypro/4.1/bin/singularity and official
H100 example image
/expanse/projects/qstore/installs/containers/singularity/Expanse-Air/pytorch/pytorch-nvcr-25.03.sif.
Its /usr/bin/python is 3.12.3 with Torch 2.7.0a0+7c8ec84dab.nv25.3, suitable as
a smoke candidate only, not OPD's pinned formal runtime. Container support keeps
staging/publication on the host; only tiny GPU arithmetic runs inside the image.
Image path/size/mtime are checked and recorded, not claimed as a content hash.

The authorized snapshot was uploaded and verified: run ID
`20260918T013816Z-1e6fb5739378-e1ff0d7a`, 445 files, 3,960,118 source bytes;
code SHA-256 `1e6fb5739378f917b15edd54538519a1480bedd0945db738d101426d16f554a8`.
The release is under `/home/zgao12/quest-runs/OPD/releases/`. The newly created
smoke-results root is readable/writable on HOME NFS; account/QoS and container
checks passed again. The one authorized submission received real Slurm job ID
`54345483` at 2026-09-18T01:39:22Z, intent
`ed6b17d1d1fd45a58c4fc206766d6324`; its local receipt is
`.sdsc/submissions/ed6b17d1d1fd45a58c4fc206766d6324.json`. It is now COMPLETED,
ExitCode 0:0, elapsed 16 seconds; batch and extern steps also completed 0:0.
The final status record is `.sdsc/job-54345483-final-status.json` with
`success: true` and verified result hash. No resubmission is authorized.

Real node exp-19-04 exposed one NVIDIA H100 80GB HBM3, CUDA 12.8, visibility
`0`; the finite 128x128 CUDA matrix product matched the CPU reference. Source
was verified/staged under `/scratch/zgao12/job_54345483/` on node-local ext4.
The shared image worked on this node; small JSON/log results were published
to the dedicated HOME NFS result path and read back successfully before exit.
Fetch returned six small artifacts (9,216 bytes) to
`.sdsc/fetched/54345483/fetch-53tjr6t4/`; receipt hashes were checked locally.
Result SHA-256 is `3462d93389b2e8ff3dec45afae1fd0209f12147a9df8b281dab6624b668bf817`.
This first job is infrastructure evidence, not a formal OPD/G0 result. It used
no dataset/model and did not establish Lustre availability. Subsequent storage
and formal-preparation progress is recorded above. HOME remains restricted to
source and small metadata. Later edits do not change an existing release.

Earlier teacher/calibration verification: all 244 isolated CPU tests, Ruff, shell
syntax and diff checks passed after the calibration/supervision review fixes.
Tests cover actual archive/validator and worker/result-validator
interfaces with temporary fixtures; all scheduler/GPU operations are mocked.
These CPU tests are separate from the real GPU evidence above. Independent read-only container
integration review found no blocker. Implementation remains uncommitted;
no reset, checkout, cleanup or new Git commit was used. GitHub fast-forward
previously brought this Quest checkout to 0215c356355b29b5e2b407978a207db2156719e1.
Operating instructions and next commands are in `docs/sdsc_workflow.md`.

## Repository and authority state

The user authorizes this teacher-failure repair and a fresh G0 retry through
armed automatic intake. Agents own git add/commit for authorized work; the
former sandbox Git blocker is resolved. No central service, registration or
GPU-state mutation is authorized from this OPD session.

The diagnostic implementation 7d40de186c5299cd76d4ce05cf4da324cca85175 and
independent science acceptance 76543d24af032b4d8d1cc23331418ec5a080df70
are committed. Its request ran from launch HEAD
cbe7e3de9a5c96bb2426bd748d14a8db0aa0ccd6 and is terminally failed.
The retained real ledger now supports a cause-specific prompt repair described
below. Implementation commit 73fa50a37541d1e09553fc69288f1f895da2ebe3 was
independently accepted without blockers at 2026-09-09T22:24:15Z by
Codex independent reviewer /root/scheduler_update_review. The distinct
review-only acceptance commit cf132e342653a95301ac4274ded984ba88c9e9dc
changes only the review block of
prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_prompt_v3.yaml.
Its SHA-256 is 752fa685d795335527c639fb2b7f6cc3e94aa60ffb9a16d4329bf13099b0888e.
The actual clean-checkout builder accepted the science lineage and unchanged
execution-class certification and generated the single request below.

The reusable execution class remains accepted by joint commit
46352c4b88013761cd43a83282fd3c6251bf2d9e, reviewing implementation
811fd772711d1792d59a2159174886f68721c0c4. Its entrypoint rejects untracked
or ignored source shadows. The accepted central registration remains enabled.
The new renderer files are outside the named safety surface; the existing
fingerprint and certificate validate. CPU prompt-envelope evidence passes and
is recorded below for scientific review. Git contains the earlier lineage history.

Candidate E amendment
prereg/amendments/qwen3_v2_g0_elastic_v1.yaml is unchanged, remains accepted,
and still has SHA-256
ff34cc53a85abe409f65ebe1ad3ca4d46b08a0ecd2117b76a27e22eea633a725.
Its per-G0 four-real-pilot gate remains immutable historical v1 authority. The
accepted v2 successor does not rewrite those bytes; it provides the separately
accepted execution-class path for a new Candidate E request.

The successor artifacts are:

- prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml;
- prereg/execution_safety/qwen3_v2_elastic_training_v1.descriptor.json;
- prereg/execution_safety/qwen3_v2_elastic_training_v1.certification.yaml;
- prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_v1.yaml.

All three review-bearing successor documents have identical accepted metadata
bound to implementation commit 811fd772... and were jointly committed at
46352c4b.... Acceptance alone does not authorize central deployment,
enablement, request generation, or submission.

## Correct scheduler-managed GPU model

Each scientific GPU task has one profile with gpu_count_policy = scheduler.
The values 1, 2, 3, and 4 are ServerScheduler claim-time allocation candidates,
not four project tasks, profiles, requests, filenames, or scientific variants.
ServerScheduler alone compares predicted wait plus runtime, selects the actual
count and ordered GPU UUIDs, and applies normal admission. A running attempt is
never resized.

The qwen3_v2_gpu_preflight and qwen3_v2_g0 builders continue to emit
allocation-neutral protocol-v2 requests. The request contains the normal
identity fields and only workflow_id, plan_sha256, and unit_id as parameters.
It omits execution_profile, resources, GPU count, GPU identity, memory,
utilization, exclusivity, and availability hints.

The sole entrypoint remains
/home/del6500/projects/OPD/scripts/server_scheduler/opd-entrypoint. It opens the
running manifest once with no-follow semantics and holds that descriptor
through the GPU child lifetime. Both GPU handlers reread the exact held bytes,
verify the manifest SHA and allocation digest, cross-check
SERVER_SCHEDULER_GPU_COUNT, require the ordered assigned UUIDs to equal
CUDA_VISIBLE_DEVICES, preserve CUDA_VISIBLE_DEVICES unchanged, and launch only
logical devices 0 through N-1. The CPU handler receives no GPU manifest
descriptor.

## Reusable execution-class certification

The new execution class is qwen3-v2-elastic-training-v1. Its descriptor binds
the exact safety-relevant implementation and fixed runtime, plus canonical
safety projections rather than the identity of one experiment.

The fingerprint covers at least:

- outer entrypoint, GPU handlers, adapter runtime, held-manifest and allocation
  validation;
- fixed Python executables, dependency locks, package manifests, model and
  tokenizer revisions;
- model/tensor shape envelope and supported sequence/prompt-population bounds;
- global batch partition for world sizes 1, 2, 3, and 4, loss scaling, and
  rank-local microbatch schedules;
- exact global non-padding token reservation and optimizer-step ceiling;
- requested/effective FSDP wrapper semantics;
- checkpoint boundary, same-world resume, changed-world rejection, and
  attempt isolation;
- cgroup/per-device memory envelope and supported world sizes.

Accepted successor identities are:

- execution descriptor SHA-256:
  d57c6620d090da503d5dbc5d1415a6690b7eb0110ad7cdaaf4ac1e42e6e0a739;
- execution fingerprint:
  ca27527ea4aa345114bb58859ae39078887a084bc078204c32f8782103381108;
- certification SHA-256:
  7af2ad0f643860d7a9632fb643efc17166e1d868ca8643a72255cb591e9413f2;
- certification core SHA-256:
  2d6a9cc556b085e191ad0b0818f1dde9e6e6e52c8ff6367e8ce16935cf884d44;
- accepted v2 amendment SHA-256:
  9971f63015435dbe595183e5376cb5acb8e62c8c40a62854ccbbac60b2fc307d;
- Candidate E science protocol SHA-256:
  1f408237c9e6099356b3602fdb6fe20f7e8d150312d052f81c9e3875a44aa4ba.

The G0 plan no longer embeds eight W=1/2/3/4 report/completion artifacts.
Under the successor design it binds the exact descriptor, reusable
certification, accepted execution-class amendment, and a separate
per-experiment science protocol through immutable content identities. Handler,
finalizer, report, bundle, completion marker, and semantic validator carry and
cross-check the same execution-class and science identities.

The shared execution-safety kernel is used by preflight fixtures and the actual
training path. It checks batch partition before and during training and emits a
final runtime attestation. One global optimizer window remains 64 logical
samples with rank totals 64; 32/32; 22/21/21; or 16/16/16/16. Loss and token
accounting retain the same global semantics. Same-world resume remains exact;
changed-world resume fails before state load. FSDP remains FULL_SHARD for
world sizes 2 through 4 and the reviewed effective NO_SHARD behavior for world
size 1. G0 itself remains scheduler-managed over all four counts.

## Evidence reuse and invalidation

A new experiment may reuse an accepted execution-class certification only when
its recomputed execution fingerprint matches exactly and its shape/batch/token/
FSDP/checkpoint/memory requirements remain inside the certified envelope.

The following do not by themselves invalidate execution-class certification:

- a fresh job_id, workflow_id, or plan_sha256;
- seed or replication identity;
- output or scratch directory;
- scientific parameters that do not affect distributed execution.

Each new experiment must still have its own accepted scientific protocol,
resolved/config artifacts, immutable inputs, output validation, artifact
inventory, and scientific completion decision. Execution topology
certification is not scientific preregistration and does not substitute for
those checks.

Certification is invalidated for immediate reuse by a change to distributed
execution or its verification, including handler/runtime bytes, fixed
dependencies, model or tensor shapes, maximum sequence length, batch
partition, loss scaling, token accounting, FSDP, checkpoint/resume semantics,
memory envelope, or supported world sizes.

Version 1 deliberately uses an exact named set of 52 safety-critical files for
whole-file identity, plus canonical projections for mixed scientific
configuration. A byte mismatch in that named surface pauses automatic reuse
pending review. It does not automatically require four new real pilots. First
classify or isolate the delta; obtain new GPU evidence only when and to the
extent the actual safety envelope changed. This prevents harmless job/science
identity changes from becoming topology recertification while remaining
fail-closed for uncertain safety changes.

## Candidate E migration and evidence

ServerScheduler's general protocol requires reviewed semantic correctness for
all supported world sizes. It does not require every new scientific experiment
to rerun four real GPU pilots. The old four-real-pilot requirement is an OPD
Candidate E v1 project gate.

The accepted v2 amendment replaces that per-plan matrix with joint acceptance
of one reusable execution-class certificate and one per-experiment science
protocol. Its evidence is deliberately stated without overclaiming:

| World size | Successor certification evidence |
| --- | --- |
| 1 | historical accepted real preflight plus current static fail-closed evidence |
| 2 | historical accepted real preflight plus current static fail-closed evidence |
| 3 | static fail-closed evidence only |
| 4 | static fail-closed evidence only |

The W=1 and W=2 reports predate the successor fingerprint. They directly
observed the predecessor preflight handler and shared production-shaped
NCCL/FSDP/training path, not the successor G0 handler or the full G0 pipeline.
Their reuse is therefore an explicit legacy-evidence condensation accepted by
the independent reviewer after inspection of the historical Git blobs and the
successor delta.

The residual risk is explicit: the W=3 uneven 22/21/21 tail and W=4 topology
have no successful real-GPU observation under this execution class. Static
tests prove fail-closed behavior and actual-world binding, while runtime
attestation and semantic completion validation remain mandatory for whichever
world size the scheduler selects. This is the minimum migration that keeps
dynamic scheduling, does not insert resource hints, preserves Candidate E v1,
and avoids making four pilots a permanent per-experiment tax.

The 2026-09-09 central durable records show the former W=3 request
opd-3d20deb555e18b55a04873cc32769051 and W=4 request
opd-44c3d75c1e8b9180d4e08ef55f173379 were operator-cancelled before launch
as obsolete after formal G0 submission. This status check changed neither job.
The joint acceptance at 46352c4b... permits the Candidate E successor path to
use the reusable class certificate; central deployment and request gates still
apply.

## Deployment proposals and hashes

Both project-owned registration proposals remain disabled. Each task still has
one scheduler-managed elastic profile with no gpu_count field. The current
project proposal identities are:

GPU preflight:

- handler SHA-256:
  7e908dbbe0aef562b2358242336569db53e8daa14104ce873997b7e7b68692b5;
- package-manifest SHA-256:
  e2284b7c20636f3c85f719cdc0179a0ae5973fdf7cb502d247110db0e5cc037c;
- deployment identity:
  2b58f3da9fc4fe116cf4be5b326082dc0ed3da65d38bc1b082aae9d8716fd155;
- disabled proposal SHA-256:
  abda82622cbcec648b0476edac92502a0616110cc9e3a70ebcbb9bd6637bf7fa.

G0:

- handler SHA-256:
  049b4b35a3feba8898f74246d7f223e571b7df43060151c210efa5bc2510f8da;
- package-manifest SHA-256:
  fb2be67fe2f03d6beb1c6f91f2500e1cdf97853704c8e1f27740bad1757a3057;
- deployment identity:
  3ea87466891f18b18896bda78a0bcafe17da053ef440fd0005cf4fb8aa970a91;
- disabled proposal SHA-256:
  82f609530c0931ce7cb1f230ee4cbf6601cc3fa1613a89be53f31a7dce4961c8.

The actual ServerScheduler parser accepted the CPU profile and both
scheduler-managed GPU profiles in the candidate review. Read-only verification
after the separately approved enablement found
/home/del6500/projects/ServerScheduler/config/projects/opd.toml differs from the
G0 proposal above only by enabled = true. Its enabled central SHA-256 is
084661f557594126285efe56cdbebd20cc240e631f97618f45a34c9ba8ca1c39.
No service or task action was performed by this OPD session.

## Current automatic intake workflow

The updated central project-session guide and automatic-intake manual replace
per-request operator handoff with atomic outbox publication and central
receipts within an approved continuous-submission scope. The user explicitly
confirmed that this task submission needs no further authorization. Do not ask
again for permission to submit this retry.

The revised classifier is operational: the latest failed OPD attempt was
classified application_unknown, terminal without automatic retry. The central
operator acknowledged the three older application-only quarantines on
2026-09-09 at 22:40:28Z--22:40:29Z; do not report them as current blockers.
The central 2026-09-10 16:56 CDT handoff reports intake enabled/unblocked.

A direct read of central GPU safety state at 2026-09-10T21:59:17Z found all
four devices in probation with reason "GPU still has compute processes",
no quarantined devices and dispatch_paused=false. No device was ready for
exclusive OPD admission at that snapshot. These observations do not authorize
OPD to alter processes or GPU state. OPD performed no service or GPU operation.

## Current prompt-repair G0 request

The generic builder prepared exactly one fresh, allocation-neutral request from
clean project HEAD cf132e342653a95301ac4274ded984ba88c9e9dc.

- job_id: opd-b4e756837282b3d77f61ca3e309e722e;
- production outbox path: /scr/del6500/OPD/scheduler/outbox/opd-b4e756837282b3d77f61ca3e309e722e.json;
- exact request SHA-256: cb78fe0baeb0a553fffc48994f36e3eaba196b479c1af46b40ea72e14bc0ba0a;
- workflow_id: qwen3-v2-g0-elastic-6b3710a3edaa8a8c8bcae74956decd34;
- plan SHA-256: 5109dcd1e163e4ecdbf4d4083cb2c03c205233eca04e00496f3c84dca59bed69;
- canonical plan: /data/del6500/OPD/workflows/plans/qwen3-v2-g0-elastic-6b3710a3edaa8a8c8bcae74956decd34/5109dcd1e163e4ecdbf4d4083cb2c03c205233eca04e00496f3c84dca59bed69.json;
- staging receipt: /scr/del6500/OPD/tmp/g0-publication-l832n431/publication-receipt.json.

The request contains only protocol version, fresh job ID, project, task,
priority zero and workflow_id/plan_sha256/unit_id. It omits resources and
execution_profile. Strict request, plan/CAS, accepted-science lineage and
unchanged class validation passed. The 22:23:52Z central intake scan remained
enabled and OPD armed with no blocker.

Central durable state now records submission at 2026-09-09T22:25:43Z and
terminal failure at 23:14:41Z, with exit 2 classified as application_unknown.
Attempt 1 ran from launch HEAD 8d4a78d95ed5dd672e8439780dcbbc740b2bca28
with one GPU, 24 CPUs and 192 GiB host RAM. The central runtime_scope_ready
event establishes startup at 23:09:20Z: 18:09:20--18:14:41 CDT on September 9,
lasting 5m21s. Its logs record completed build_splits and
export_initial_checkpoint, then CUDA out of memory in build_teacher_demos;
student training never started. The failed allocation requested another
96 MiB while logical GPU 0 had 34.56 MiB free out of 94.97 GiB total.
The log distinguishes PID 1121808 occupying 91.87 GiB from the reporting
process occupying 3.06 GiB (2.50 GiB allocated by PyTorch; 17.95 MiB reserved
but unallocated). Another process's GPU occupancy is therefore the directly
observed memory-pressure cause, not evidence of a 92-GiB teacher allocation.

The central payload PID was 1121809, while the scientific command references
supervisor PID 1121813. PID 1121808's owner, program and relationship to the
job cannot be established from retained records; do not call it VLLM or assign
it to a user from today's process list. The prelaunch_verified audit event
persists UUID/PCI/lease, not an actual GPU-memory/process snapshot. It cannot
distinguish occupancy already present at admission from a later competing
launch. The suppressed scientific traceback also prevents identifying the
exact failing Python statement. A code fix or smaller token limit is not
established as the remedy by this evidence.

Read-only verification on September 10 found this is still the latest OPD job,
attempts_started=1 and next_attempt_at=null. Its workflow output/completion
directories contain no files, its temporary workspace is deleted, and no new
teacher diagnostic ledger was published. The earlier 2048-row rejection ledger
must not be attributed to this OOM attempt. Authoritative audit events are in
/data/del6500/ServerScheduler/audit/events.jsonl (prelaunch line 2190,
runtime_scope_ready line 2193); OOM is stderr line 5. Stderr SHA-256:
34e3666ae377da9107dbba028baf9c6a97e7a5904de6981e114eaa2d3973463a.

Authoritative state:
/data/del6500/ServerScheduler/state/jobs/opd-b4e756837282b3d77f61ca3e309e722e.json.
Attempt logs:
/scr/del6500/ServerScheduler/logs/opd-b4e756837282b3d77f61ca3e309e722e.attempt-001.stdout.log
and the corresponding .stderr.log. The workflow has no scientific completion
marker. This failure does not establish the repaired prompt's teacher-proof
quality and is distinct from the earlier complete rejection ledger below.
Do not republish or reuse this accepted job ID. This observation made no
central changes and prepared or submitted no new request.

## Latest diagnostic G0 outcome and prompt repair

Task/profile: qwen3_v2_g0 / qwen3-v2-g0-elastic. Job
opd-15db6153a4b750c67fc4706b0c0aceb6 was accepted by automatic intake at
2026-09-09T18:32:12Z and ran once from 18:35:32Z until 21:41:05Z. It received
three GPUs, 24 CPUs and 196608 MiB RAM, then failed build_teacher_demos with
exit 2. Student training never started, and no automatic retry is scheduled.
The old daemon's gpu_unknown label is not evidence of a hardware fault.

Its immutable request remains at
/scr/del6500/OPD/scheduler/outbox/opd-15db6153a4b750c67fc4706b0c0aceb6.json
with SHA-256 c20ad219c4a6d39ba015142a250135f6aa35ceb3d09a811e3530cdba158ba29f.
Do not reuse this accepted ID. Publication evidence is under
/scr/del6500/OPD/tmp/g0-publication-ugi534xk/; central job state and attempt
logs remain authoritative.

The preserved ledger/view/manifest are under
/scr/del6500/OPD/diagnostics/teacher_demos/failure-g9wqngmn/.
All 2048 candidates were generated; only 45 passed, covering 13/256 prompts.
Failures comprise response_syntax 881, step_syntax 872,
antecedent_mismatch 188 and unknown_citation 62. Parsing errors are 87.5% of
rejections. In particular, 828 step-syntax failures restate facts as Sxx: F...
instead of applying a single rule. All 881 length-terminated outputs hit 256
tokens. Correct accepted responses contain only rule steps and use 53–119
tokens. The ledger SHA-256 is
ae1429dcb75ecd5d4b27c317a9d2efa45f3dac49d1d84df4bec433dd8d7f0b61.

The cause-specific repair clarifies the exact rule/citation output contract in
proofgraph/rendering.py and shares it with anti_shortcut.py. Input punctuation
is compacted to preserve the existing prefix bound. Base graph facts, rules,
identifiers, polarity, order, labels and canonical targets are preserved. No verifier,
RNG, sampling limit, candidate count or complete-coverage gate is relaxed.
There is no answer injection or postprocessing of teacher responses. Real
reasoning errors remain possible; improvement requires a new GPU observation.
Details: docs/refactor/qwen3_v2_teacher_prompt_repair_20260909.md.

## Previous failed formal G0 request

The accepted Candidate E builder generated exactly one fresh resource-neutral
request from clean project HEAD
ed1beab1ca1013b4e12cdd3def3dff6a51a3b953:

- outbox path:
  /scr/del6500/OPD/scheduler/outbox/opd-2947686c51c3e93ad1b18e5a3b7d6b22.json;
- outbox SHA-256:
  b34953ad8af479dd29d8f1bb90af18f65fa3b45f9e355bb265f0c7bb303e0443;
- job_id: opd-2947686c51c3e93ad1b18e5a3b7d6b22;
- workflow_id: qwen3-v2-g0-elastic-079b8a7cbc197974d8c7c7e163fb5d53;
- canonical plan SHA-256:
  035605ae98e188242b95debe33096ab7f361fd55dbccd8b76bbabe044cb36626;
- plan path:
  /data/del6500/OPD/workflows/plans/qwen3-v2-g0-elastic-079b8a7cbc197974d8c7c7e163fb5d53/035605ae98e188242b95debe33096ab7f361fd55dbccd8b76bbabe044cb36626.json.

The builder and a direct strict-shape check accepted the request. It contains
only schema_version, job_id, project, task, priority, and the three parameters
workflow_id, plan_sha256, and unit_id. It omits execution_profile, resources,
GPU count, and GPU identity. The canonical WorkflowPlan loader recomputed the
same plan SHA-256. The outbox remains present; central audit records now
confirm its job was submitted on 2026-09-07 at 22:40:24 UTC. Do not resubmit
this accepted job ID.

### Observed execution and failure

Read-only inspection of central durable jobs, retries, audit events, and
attempt logs on 2026-09-09 around 17:49 UTC found:

- task/profile: qwen3_v2_g0 / qwen3-v2-g0-elastic;
- attempt 1 started and its runtime scope became ready at 07:04:25 UTC
  (02:04:25 CDT), with 1 GPU, 24 CPU cores, and 196608 MiB host memory;
- it became failed at 09:40:33 UTC (04:40:33 CDT), after 2 h 36 m 8 s,
  with exit code 2; the lease was released;
- stdout records completed build_splits and export_initial_checkpoint stages,
  then the start of build_teacher_demos; no training stage was reached;
- stderr reports `teacher-demo generation has zero-success prompts` and the
  build_teacher_demos subprocess exiting 2. Exactly 243/256 prompts had no
  accepted candidate (119 positive, 124 negative); raw candidate rejection
  reasons were deleted with the failed workspace;
- the scheduler classified the exit as gpu_unknown and retryable=false;
  attempts_started=1 and next_attempt_at=null. That scheduler classification
  alone is not evidence of a hardware fault;
- the audit records quarantine of the assigned GPU at failure. Its current
  safety state was not queried; any GPU-safety operation belongs to the central
  operator;
- all 14 durable OPD jobs were terminal (4 completed, 10 failed); none were
  pending or running at this observation.

Authoritative state is
/data/del6500/ServerScheduler/state/jobs/opd-2947686c51c3e93ad1b18e5a3b7d6b22.json;
timestamps and retry classification are in
/data/del6500/ServerScheduler/audit/events.jsonl. Attempt logs are
/scr/del6500/ServerScheduler/logs/opd-2947686c51c3e93ad1b18e5a3b7d6b22.attempt-001.stdout.log
and the corresponding .stderr.log. The stderr-reported teacher-demo directory
/scr/del6500/OPD/tmp/qwen3-v2-g0-c0sywxc7/qwen3-v2/teacher_demos was absent
when checked; this check did not locate a retained diagnostic ledger.
This is a real failed G0 execution, not scientific completion or successful
training/FSDP evidence for the successor.

### Historical diagnostic repair

The original failed job deleted its temporary teacher ledger. The repair at
7d40de1... retained newly failed ledger/view/manifest files under the fixed OPD
scratch diagnostic root and preserved the original exception. Its 25 focused
tests have passing evidence. The accepted diagnostics-v2 science successor
and original artifacts remain unchanged. The latest real run demonstrates
that retention works and supplies the response evidence summarized above.
Historical CPU reconstruction and diagnostic test commands are documented in
docs/refactor/qwen3_v2_g0_failure_diagnosis_20260909.md.

## Verification

The 2026-09-10 diagnosis cross-checked durable job/retry records, full attempt
logs, central launch audit, OPD artifact directories and current central GPU
safety state. Independent review confirmed the same process-memory distinction
and missing historical snapshots. Only this handoff changed; git diff --check
passed. No scientific code, test fixture, model or GPU execution was changed
or run, and no request was prepared or submitted by this inspection.

Prompt repair checks on 2026-09-09: 45 tests passed (14 new output-contract
cases, 28 existing ProofGraph/stage-4/teacher-ledger/store cases, and 3 existing
anti-shortcut cases; 14 unrelated cases deselected). The unchanged verifier
still rejects fact restatement, prose, future citations and wrong premises.
All 256 production teacher prompts fit 402–1246 tokens with the pinned
offline chat tokenizer, and all 256 unchanged canonical targets verify.
Evidence: /scr/del6500/OPD/tmp/g0-prompt-repair-20260909/teacher_prompt_envelope.json
(SHA-256 66b4dfb2be5a427af07e726c87cf68b7e2510e6b8de2d8ad75cb18297e097283).
Actual 128-example validation/IID/circuit populations also remain within the
training envelope. Auxiliary serial anti-shortcut inference has an existing
larger envelope: its maximum prefix decreases 2020→1988, and prefix plus its
256-token completion decreases 2276→2244; do not claim it is below 1536.
Evidence: /scr/del6500/OPD/tmp/prompt_auxiliary_envelope_20260909.json.
All 52 safety-file hashes, descriptor/certificate and recomposed science-config
identity validate. AST/import and git diff --check pass. No new GPU run has
been performed by these CPU checks. The prompt-v3 review is now accepted.

Static and CPU/no-GPU verification completed on 2026-09-07:

- final scheduler/science focused suite: 234/234 passed;
- independent read-only candidate review before the final lazy-import repair:
  152/152 certification/science/handler tests and 69/69
  registration/adapter tests passed;
- complete test run: 608 passed and one entrypoint isolation test exposed a
  top-level PyYAML import regression;
- after the minimal repair, that exact test passed 1/1 and its affected
  certification/adapter suite passed 67/67;
- the subsequent independent acceptance review reused those results and ran no
  suite; it rejected e943635... solely for the top-level src shadow gap;
- the narrow repair's real entrypoint shadow test and existing bytecode
  isolation test pass 2/2, including ordinary, .gitignore, and
  .git/info/exclude cases;
- this independent review found no blocker in implementation commit
  811fd772...; the exact review-only transition validator passed and
  test_real_git_joint_acceptance_resolves passed 1/1;
- after the user-created acceptance commit, the actual checkout resolver bound
  acceptance commit 46352c4b..., implementation commit 811fd772..., accepted
  review status, and fingerprint ca27527e... successfully;
- both fixed runtimes pass pip check;
- descriptor recomputation, fingerprint/CAS binding, certificate validation,
  handler/package/deployment hashes, compilation, and git diff --check pass;
- the Candidate E v1 amendment is byte-for-byte unchanged; all three successor
  review blocks are identical, accepted, bind 811fd772..., and are committed at
  46352c4b....

CUDA/NCCL lines emitted by unit fixtures are mocks. Tests explicitly hid CUDA.
No GPU was queried or used.

This migration did not call central submit or dispatch, did not query or alter
a job, queue, lease, allocation, or constraint, did not modify central
ServerScheduler, and did not restart or signal a service. It created the
project-owned immutable plan/CAS inputs and the single outbox request recorded
above; those local files are not evidence of submission. It performed no
external action.
A generated pytest cache was moved recoverably to
/scr/del6500/OPD/tmp/pytest-cache-execution-class-20260907; seven generated
Python bytecode files were removed.

## Required next gates

Implementation, independent review-only acceptance, publication and execution
of the prompt-repair request are complete; that execution failed before student
training. The immediate OOM cause is established as competing process memory;
historical process attribution and admission timing remain unresolved. Before
retry, the central operator needs to establish adequate GPU availability and
investigate concurrent occupancy during the claimed exclusive allocation.
Central GPU/process recovery belongs to that operator. The existing request
must not be published again. Any subsequent retry must use fresh identity and
the applicable scientific, acceptance and intake gates; the prompt-quality
improvement remains unmeasured.

Scientific success still requires validated g0.json, g0_artifacts.tar and the
semantic completion marker. Neither a request receipt nor a CPU test is GPU
success. Never reuse old IDs, force GPU counts, or create availability probes.

## Quarantined legacy scheduler surface

The unsupported inventory remains: 20 files under scripts/slurm, eight
launch/supervision files under scripts/production, and three Python
pilot/finalizer modules listed in
docs/refactor/legacy_scheduler_inventory.md. They are unreachable from the
ServerScheduler handler registry and remain deletion candidates for a separate
approved cleanup. No local GPU selector, nvidia-smi placement logic, CPU/GPU
lock, lease, local queue, Screen/tmux fan-out, or background scheduler was
introduced.

## Documentation map

- Operating rules: AGENTS.md
- Reusable certification design:
  docs/refactor/qwen3_v2_execution_class_certification.md
- Accepted Candidate E v1 amendment:
  prereg/amendments/qwen3_v2_g0_elastic_v1.yaml
- Accepted execution-class successor:
  prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml
- GPU pilot contract: docs/refactor/qwen3_v2_gpu_preflight_pilot.md
- Legacy inventory: docs/refactor/legacy_scheduler_inventory.md
