# OPD current handoff

Last updated: 2026-09-10.

This is the canonical current-state summary for the OPD refactor and
ServerScheduler integration. AGENTS.md is authoritative for operating and
approval rules. Source, Git, scientific artifacts, and central scheduler state
must still be verified when mutable.

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
