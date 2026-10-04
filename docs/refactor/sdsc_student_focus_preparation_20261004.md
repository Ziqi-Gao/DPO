# Branch-focused shared initial preparation V5

The completed batch-order diagnostic54659007/54659010 did not provide a broadly
better policy. Fixed primary6/7/8 proof differences were rename-3.646pp,
fact-order+4.948pp and rule-order+1.562pp, with IID+1.042pp. Complete raw and
execution audits passed. See `sdsc_student_batch_probe_20261004.md` for all
identities, accounting, observations and limitations. No diagnostic checkpoint
is promoted, continued or used as the common initial.

## Rationale and prospective scope

Early IID chain proofs were138/144 and133/144 while depth3/4 branches were0/78
and2/78. Even after an exact first-branch prefix, the immediate next step was
invalid31/34 and28/32 times. By step32, IID branch proofs reached39/42 and34/42,
but aggregate ability already exceeded the unchanged initial-model band.
Renamed branch proofs remained24/42 and21/42. Early rename gaps also involve
chains, so branch emphasis is a targeted hypothesis, not an established general
invariance repair. We do not infer an optimizer bug or optimal learning rate
from low, finite training loss.

The new policy changes only fit structure sampling relative to the anchored
control:50% branch/25% chain/25% DAG becomes75%/12.5%/12.5%. Keep four independent
signed pairs, eight views per base and global64. For zero-based windoww and
slots0..3, the first three pairs are branches; slot3 is chain for evenw and DAG
for oddw. Keep depth2+(w+slot)%3. Every window contains48 branch and16 nonbranch
sequences, split24/8 per rank; consecutive windows balance the two nonbranch
types. There are192/32/32 fit bases. Changing sampling avoids introducing an
additional weighted-loss normalization or changing the verified optimizer.
This is a prospective preparation policy; there is no paired causal efficacy
claim for this new run.

Retain the anchored control's complete per-base eight-view order, negative
sibling one-slot rotation, two original-order anchors(identity and rename-only),
50% renamed and12.5% paraphrased views. Six views receive the same independent
joint fact/rule permutation operation. Composing that with an earlier
permutation can naturally restore original order:1536 joint operations yield
1534 net-changed fact orders and1536 net-changed rule orders in this frozen
population. The two naturally restored rows are preserved, without rerolling,
filtering or adding a second policy change. Protocol fields distinguish applied
permutation fraction from net changed-order count.

## Frozen data, training and selection

Protocol: `prereg/amendments/qwen3_student_focus_preparation_v5.json`.
Core module: `src/posttrain_circuits/experiments/protocols/student_focus_preparation.py`.
Fresh namespaces390000042/400000042 are fit/dev;410000042/420000042 are fit/dev
transforms. They were checked before one generation and frozen without seed
search or row replacement. Fit256 bases/2048 views and development256 bases/
1536 views are independent. The new dev contains86 chain/68 branch/102 DAG
examples with original difficulty and six original transformation views.

Manifest18d7e8456e03f1ef274000164460f79882e6da1182a8c81f6d5bd7c016b6ce26
binds all ordered IDs, signed pairs and views. Complete training input tokens
are1,665,064; the four-step preflight consumes205,428. Maximum fit prefix1000,
response206 and total1165 fit the unchanged1344/256/1536 bounds. Development
maxprefix2192+256=2448 fits2454. All targets pass the original parser/verifier.
Original formal anti-shortcut input bound2244 is unchanged.

CPU reconstruction excludes all144000 original examples,8192/512 teacher
examples, all previous preparation populations and complete transformed views,
and all three diagnostic pairs' fit/dev views. It checks semantic, example ID,
pair group and seed separation. Original formal data are used only as exclusion
keys; no formal generated response or score is read for selection. Isolation
SHAf05e23c396278b93dc06769b1ea5cbc454703c7da88270c99c95c2710cf7401b
and token feasibility49b90a406d41d169790214662c6704ed6a73adc51316d6b33b6f656848048a18
are under `.sdsc/diagnostics/student-focus-v5/`. GPU-node staging must separately
verify the actual persisted original-family inputs.

Both preflight and fit start native1.7B54548846 with fresh optimizer/RNG/cursors.
Retain full-parameter FP32 master weights, W2 FULL_SHARD, global64/micro4/accum8,
sequence-mean response+EOS loss, constant AdamW5e-5, betas0.9/0.95, epsilon1e-8,
zero decay/warmup/clipping and exact2M input-token budget. No teacher targets.
The preflight runs four updates, saves/restores step4 and generates only eight
training responses. It never reads development for selection. Full fit requires
its matching real preflight and independent raw audit, then restarts native
weights; it cannot continue a preflight checkpoint.

Full fit completes all32 updates and all12 checkpoints1/2/3/4/5/6/7/8/12/16/24/32,
with1536 responses each:18432 total. Only after the complete run may the original
earliest-eligible selector return a candidate. The original numerical gates
remain: IID answer26..153/256, IID proof>=.10, mean transformed proof>=.08,
each transformed proof>=.05, each IID-minus-view gap<=.05, IID-minus-mean<=.05,
and each IID structure proof>=.10. No eligible checkpoint means failure with
complete evidence; no extension, threshold change or post-hoc reselection.

Each checkpoint retains both ranks' exact311-key FP32 reload and native-BF16
parity. The stronger already-observed batch diagnostic check is also retained:
a synthetic2454-token finite forward on every checkpoint/rank, without extra
development content. This additional execution check changes no training or
scoring sample. A complete worker fit has execution_complete and
preparation_complete true even when selected_checkpoint is null; passed remains
false and the job/publication fail. Raw-audit PASS then means faithful replay of
a rejected run, not model acceptance. Formal896-response qualification remains
separate; all student/formal/G0/pilot/factorial/class acceptance flags stay false.

## Execution and review boundaries

New flat controls are `tools/sdsc_student_focus{,_job,_startup,_worker,_audit}.py`.
Task`qwen3-v2-student-focus-preparation-v5`, namespace`student-focus-v5`, schema
family`quest-sdsc-student-focus-*-v1`, dense format`student_focus_preparation_dense_v5`,
source kind`symbolic_canonical_focus_preparation`. They bind actual single
publication evidence and do not fabricate the old recovery wrapper.

Keep2H100/24CPU/384GiB, accountnwu181, nairr-gpu-shared/
nairr-gpu-shared-normal. Preflight1h including300s publication reserve; fit8h
including600s reserve. Maximum four concurrently allocatable GPUs,300s early
startup,12 threads,192GiB node-local free space and original memory/headroom
checks. Full fit39 small files/19 large files remain within224MiB/128GiB;
preflight16 small/8 large. Fetch only bounded small evidence, never weights.
Unknown submission claims are reconciled, never resubmitted or erased.

A16KiB atomic live-progress file reports phase and the latest complete update's
step/loss/cumulative tokens from a bounded64KiB log tail. It binds job/plan/source,
never claims completion and is excluded from completion artifacts. Partial rows
are skipped; unavailable telemetry does not change training or hide a memory
failure. Dedicated finite foreground observers have no submit/retry/cancel/fetch
operation, poll every60s and stop on failure, unknown state, SSH loss or changed
pins. Preflight observation is bounded2h; fit10h. A separate bounded one-shot
reader observes only this receipted job's published progress, memory and log tail.

All148 prior science files, eight accepted protocols and11 helper byte identities
remain unchanged relative to07d5813b9319ac401f09ce9847d564f841f80a50. Six new science
surfaces make154. Independent implementation acceptance and actual matching
preflight are required before fit submission. CPU checks alone are not GPU success.

## Completed local verification

All409 unique focused CPU cases pass:69 core,169 transport,49 worker and122
auditor. Coverage includes a real two-process CPU/Gloo32-update comparison to
an independent64-sequence mean, step4 restore/next-step behavior, actual
18432-response producer/auditor replay and actual execute/controller seams for
preflight, eligible fit and complete rejected fit. Ruff, formatting, pinned
Python AST and diff checks pass. No local CPU test establishes real GPU success.

Independent surface review passes core/protocol, transport/startup/node, worker
and auditor. Review found and repaired an actual isolation manifest-key mismatch
and missing nested token/isolation report bindings. Actual population-producer
seams now cover those boundaries. The core reviewer independently reconstructed
all256 fit bases/2048 views, compared the anchored policy and confirmed original
selection code unchanged. Full144k isolation was hash-bound rather than rerun.

Evidence under `.sdsc/diagnostics/student-focus-v5/` includes:

- `independent-core-review.json`:4228eaba52dd43b8f1d39f7faaad87f7359dbd4c6492fddfe0acdd60d3b21cb8.
- `root-transport-independent-review.json`:65 targeted cases plus two actual
  producer-isolation seams, all passing.
- `independent-worker-review.json`:7cf3d5ea2c006373be4110abffe9b8253dc6f933d94d62b327d85501797ccef3,
  plus final isolated-Python test-only addendum b1069fea7c7ca985a5a6ae6a96f7e0371837c5528f5303908b4784d14f2f0931.
- `independent-auditor-review.json`:767fe0f4d59dbf7dbd099ee5a90dd6d9784d07cbbb64cceec4c9414eddaaeeb2.
- `independent-observer-reader-review.json`:982dba51bcf6fb5c8514373ecaa3f45280ee6a8dcd2626afccf3bb7b62fa78e6.

The ignored, finite foreground observers/one-shot reader pass124 unique tests;
non-author review ran46 critical cases plus two final CUDA-phase regressions.
No observer is launched yet. Source implementation must next be committed,
reviewed at that exact identity and accepted in a distinct review-only commit.
Actual release/provenance, fresh parent/runtime checks and preflight submission
remain subsequent operations. No V5 model, job or GPU success is claimed here.

## Initial deployment launch correction

Implementation46db8e64e9592d410bca4f9dbf735ff7bcaa09cd and distinct acceptance
262d48e3a06ac83e4d3744d645196993422af3b6 passed two exact-commit reviews and the
actual accepted resolver. Initial release20261004T211027Z-b00ee9b8283e-2c39ad8d
(804 files/12170496B) and genuine provenance
8e210749c19061d67c058d08f9777b732d1d6e91b2ed3fcfea66fdb7dd63c354 remain immutable.
Preflight intentb9489918ded50254d0aba96966394839, plan
a4d208705e845c84c07b00b15e05101dca58b4ad94a99a063b62a4e8719b83c6, failed its
first actual remote dry-run before runpy or any Slurm operation: ssh_operation
checked the controller path against TOOLS[0], which was the startup adapter.
Both deployed files match their genuine recorded hashes; the lookup was wrong.
The prior reviews/tests missed this actual SSH launch boundary.

Read-only exact remote reconciliation in
`failed-dryrun-no-claim-reconciliation.json` verifies the execution claim,
scientific claim, submission directory and result directory are all absent.
There was no GPU job or unknown submission. Preserve the failed plan/release;
never submit it. Correct only the named controller hash lookup and exercise the
actual isolated launch argv with correct/incorrect hashes. The protocol's
scientific core/data/thresholds remain identical. A fresh corrective
implementation and separate review-only acceptance must precede a fresh release,
provenance, plan and actual dry-run. This does not silently patch deployed code.

Corrective regression first reproduced the old defect: the correct-hash actual
controller launch and successful checked-payload forwarding failed, while the
wrong-hash rejection passed(2FAIL/1PASS). After the explicit named controller
lookup, all three new cases and five related entrypoint/claim cases pass(8PASS).
The actual isolated -I/-B/-c launch reaches the real controller's Quest Slurm
guard; an incorrect hash fails before runpy. Total distinct focused cases are
now412, with prior results retained and only the affected boundary rerun.
Ruff/format/diff pass. Corrected controller
36020e7fbaa923dbb3a559894d0e0fb11390e8d88efded30395a968e9620f0d7
and testsb40602091d46c7feb532c4d08de5f6e530c5a648995aea287230180926dc4f37
are bound by the corrective evidence. No other positional TOOLS lookup exists.
