# OPD current handoff

Last updated: 2026-09-06.

This is the canonical current-state summary for the OPD refactor and
ServerScheduler integration. `AGENTS.md` is authoritative for operating and
approval rules. Source, Git, scientific artifacts, and central scheduler state
must still be verified when mutable.

## Current repository state

Committed HEAD is handoff-only commit
`23dd9b4e775e68783281f2ce0a4adc9a2605c6f7` on `master`. It descends through
handoff-only commits `15606fd6d0dc2a3e6105247e9056b68780428e5b`,
`c3d396bea456691f2aa50609e64954ff3e00fb61`,
`68f845b2390ab4ebb73c367e3aa1f57274f50845`, and
`d9e8ab315d045c05e39885af14a2fdafcf3a06b4` from
Candidate E acceptance commit
`5c0bb34cce288aef8908e498a6f5d3259b998f5b`, whose sole parent is reviewed
Candidate E implementation commit
`58df5d22f7c09ac69b807eff5294296f6927bd1c`. The implementation descends from
historical Candidate D acceptance commit
`30bafb42344edfe2a6ceff473efe41ac0ee79750`.

Candidate E is a complete, committed scheduler-managed 1/2/3/4-GPU
implementation. Commit `58df5d2...` changes exactly 80 paths: 73 previously
tracked paths and seven new paths. Acceptance commit `5c0bb34...` changes only
the amendment review block and this handoff; `d9e8ab3...`, `68f845b...`,
`c3d396b...`, `15606fd...`, and `23dd9b4...` each change only this handoff.
Before this synchronization the tracked worktree and index were clean.
The ignored `.pytest_cache/` created by the 2026-09-06 status audit was moved,
without deletion, to
`/scr/del6500/OPD/tmp/pytest-cache-status-audit-20260906`. The strict
accepted-lineage resolver passed at committed HEAD `23dd9b4...` with no unsafe
untracked paths before this synchronization. The implementation changes cover:

- permanent protocol rules and operational documentation in `AGENTS.md` and
  `docs/refactor/`;
- the G0, teacher-demo, and FSDP configurations under `configs/`;
- both Qwen3-v2 package manifests and disabled registration proposals under
  `deployments/`;
- both hash-bound GPU handlers under `scripts/server_scheduler/`;
- checkpoint, amendment, run, circuit, dataset, model, training, CLI,
  workflow, and scheduler-adapter implementation under
  `src/posttrain_circuits/`;
- integration and unit fixtures under `tests/`.

The seven new files are:

- `configs/accelerate/fsdp_server_scheduler.yaml`;
- `prereg/amendments/qwen3_v2_g0_elastic_v1.yaml`;
- `src/posttrain_circuits/cli/factorial_run_validation.py`;
- `src/posttrain_circuits/learning/training/fsdp_contract.py`;
- `tests/unit/test_fsdp_contract.py`;
- `tests/unit/test_g0_factorial_run_validation.py`;
- `tests/unit/test_g0_stage_contracts.py`.

The Candidate E amendment is now `accepted`, has SHA-256
`ff34cc53a85abe409f65ebe1ad3ca4d46b08a0ecd2117b76a27e22eea633a725`,
and binds `reviewed_implementation_commit` exactly to
`58df5d22f7c09ac69b807eff5294296f6927bd1c`. The separate metadata-only
acceptance commit is `5c0bb34cce288aef8908e498a6f5d3259b998f5b`; the actual
accepted-lineage resolver and both hash-bound handlers accept it.
Candidate D remains a recoverable historical baseline, not acceptance evidence
for Candidate E.

Historical rejected candidates remain documented in Git: Candidate A
`ab2f72f...` polluted the immutable MIB tree with bytecode, Candidate B
`e8138997...` was superseded, and Candidate C `b4c63a3...` crossed the mutable
source trust boundary before authentication. Candidate E retains Candidate D's
hash-bound bootstrap and clean-lineage protections.

## Candidate E scheduler and scientific contract

The only central entrypoint remains:

`/home/del6500/projects/OPD/scripts/server_scheduler/opd-entrypoint`

There are not four request-visible task variants. Each scientific task has one
allocation-neutral request and one scheduler-managed profile; ServerScheduler
chooses one concrete count at claim time:

| Task | Profile | Reviewed allocation | Conservative estimate |
| --- | --- | --- | --- |
| `qwen3_v2_gpu_preflight` | `qwen3-v2-gpu-preflight-elastic` | 24 CPU cores, 196608 MiB, scheduler-chosen 1/2/3/4 exclusive RTX PRO 6000 Blackwell GPUs | 7200 s |
| `qwen3_v2_g0` | `qwen3-v2-g0-elastic` | 24 CPU cores, 196608 MiB, scheduler-chosen 1/2/3/4 exclusive RTX PRO 6000 Blackwell GPUs | 86400 s |

Both proposals use `kind = "gpu"` and
`gpu_count_policy = "scheduler"` and completely omit `gpu_count`. The
internal project profile uses the same zero sentinel as the central parser;
that sentinel is never serialized into a request or proposal and is not a
resource hint. At runtime the entrypoint cross-checks the running manifest and
`SERVER_SCHEDULER_GPU_COUNT`, preserves `CUDA_VISIBLE_DEVICES` exactly, and
launches only logical ranks/devices `0..N-1`. An attempt never resizes.

Normal requests contain only `schema_version`, fresh count-neutral `job_id`,
`project`, `task`, `priority`, and the scientific parameters `workflow_id`,
`plan_sha256`, and `unit_id`. They omit `execution_profile` and `resources`.
GPU count, device, memory, utilization, exclusivity, environment, command, and
path selectors are rejected at both the request and handler boundaries.

One optimizer window is always the same ordered 64-sample logical batch:

| World size | Rank-local sample totals | Physical microbatches | CPU threads/rank |
| --- | --- | --- | --- |
| 1 | `64` | sixteen times `4` | 24 |
| 2 | `32 / 32` | eight times `4` per rank | 12 |
| 3 | `22 / 21 / 21` | `4,4,4,4,4,2` / `4,4,4,4,4,1` / `4,4,4,4,4,1` | 8 |
| 4 | `16 / 16 / 16 / 16` | four times `4` per rank | 6 |

Framework accumulation and rank averaging are scaled back to the same global
sequence mean. The student remains full-parameter training. The prompt
population is exactly 256 unique manifest-ordered IDs; the accepted
teacher-demo view uses a deterministic per-prompt cursor and consumes no RNG.
The measured maximum prompt is 1246 tokens, maximum generation is 256, and the
derived 1502-token input fits the reviewed 1536-token limit. Any overlength
sample is rejected without truncation before a training forward.

Before any backward in a window, ranks sum exact non-padding model-input tokens
and reserve that global total. A whole window that would exceed 2,000,000
tokens is rejected and its cursor/counters are rolled back; the 120 optimizer
step limit is independent. Checkpoints occur only at optimizer boundaries,
store rank-local trainer state and one identical global token state, support
deterministic same-world resume, and reject changed-world or partial-window
resume before model, optimizer, scheduler, RNG, or source state is loaded.
Scheduler retries always use isolated attempt workspaces and start from frozen
scientific inputs.

The requested FSDP strategy is `FULL_SHARD`. Its reviewed effective strategy is
`NO_SHARD` at world size 1, as imposed by PyTorch 2.8, and `FULL_SHARD` at world
sizes 2, 3, and 4. Validation requires one root FSDP wrapper plus every
`Qwen3DecoderLayer` directly wrapped, no missing or extra wrappers, and
`use_orig_params=false`. Single-rank full-state export uses
`offload_to_cpu=false, rank0_only=false`; multi-rank export uses rank-zero CPU
offload. Requested and effective strategies are recorded and validated.

G0 finalization replays the artifact bundle into a fresh private scratch tree
using no-follow, directory-descriptor, inventory, and content checks. Final and
process circuit artifacts each bind their own compatibility document and
semantic evidence. Cross-run validation compares scientific model, optimizer,
scheduler, scaler, RNG, rank, token, and evidence state, but deliberately does
not demand byte-identical framework serialization or hash absolute scratch
paths. Hashing is limited to deployment/dependency identity, immutable
scientific inputs, checkpoints, outputs, and completion evidence; there is no
blanket repository snapshot hash.

The scientific claim remains seed-42 pipeline feasibility only. Candidate E
does not authorize a confirmatory endpoint, the three-seed factorial, or Gemma
replication.

## Independent Candidate E acceptance

At `2026-09-06T02:22:34Z`, an independent static scientific review accepted
exact implementation commit
`58df5d22f7c09ac69b807eff5294296f6927bd1c` with no blocking findings. The
review traced the global 64-sample schedule and sequence-mean loss scaling for
all four world sizes; exact global non-padding token reservation and rollback;
the requested/effective FSDP wrapper contract; optimizer-boundary checkpoint,
same-world two-resume comparison, and pre-load changed-world rejection; count-
neutral requests; the W=3 `22/21/21` path; one-GPU sequence and memory gates;
no-clobber output publication, marker-last completion, archive inventory, and
semantic replay; and both hash-bound deployment identities.

The accepted transition was committed as
`5c0bb34cce288aef8908e498a6f5d3259b998f5b`. It changes only the amendment
review block and this handoff, and has implementation commit `58df5d2...` as
its sole parent.

This acceptance is static evidence, not real-GPU evidence. In particular, the
one-GPU path fails closed at the 81920 MiB per-device reservation, verifies the
actual peak before preflight success, and retains the 196608 MiB cgroup plus
headroom checks. Its real central W=1 pilot has now passed as recorded below.
Distinct real scheduler-managed W=3 and W=4 pilots remain mandatory before G0
request generation.

## Deployment and runtime identities

GPU preflight:

- handler: `e62d370db975302e3f8321ea93d5e281894c3eb09ef814a2b615b7cf7605948c`;
- dependency lock: `c72a990c03b9d841d399d9e0497912fcfd6c78ce3e840406ecd98c921c2b893f`;
- package manifest: `f17dc62af30eb7f96b94322b61b8ea7496a6fd0a1c3c31bb99e872dbd1466663`;
- deployment identity: `f0850abcf441a4dfa3134b875d2c552020115b669c5b81a93f4c62122736e16d`.

G0:

- handler: `a00cb4b0a59e2537bc6a6ce8daf07ed86f6dfe9d61c95f9e32223088cadb8ad4`;
- dependency lock: `18d77da454cb86a097526fe561462f4818cb76c3ca9633481261f1e563f86d90`;
- package manifest: `432107f92111b621474d7b1aa7179284a8a5d73c774ab80d003040e381f1ca73`;
- deployment identity: `99b595c018104a1c3f176a98d4d3c6072cdda92ba1d8a4d78800e4b119de65c1`.

The FSDP config SHA-256 is
`7fe87d579ed92c4cca91ab719a7cf267e5491ba3b520b4a05f28c1e442ffb46a`.
Both fixed runtimes pass `pip check`. The fixed interpreter digest is
`848c64ae0635d363f8bbfc768f94a3be497c0d51acd28cd5087e6e8a13c44801`.
MIB is clean and pinned at
`b759df34433c9e31043ba9e02908ce0bf20e894f`; EAP-IG is pinned at
`7af394a5662de8b23ad6154716a0cd3993d447a3`; the tree contains zero `.pyc`
files.

## Verification

Candidate E final static and CPU-only boundary verification:

- complete test suite before commit: 546/546 passed in 25.91 seconds;
- complete test suite rerun from clean implementation commit
  `58df5d2...`: 546/546 passed in 25.74 seconds; the 19 warnings were dependency
  deprecations, expected NVML-unavailable warnings in a no-GPU test
  environment, and one optimizer-wrapper test warning;
- the recorded independent-review 546/546 result was obtained during the
  proposed-to-accepted review transition, but is not reproducible from the
  committed accepted checkout;
- a 2026-09-06 rerun from accepted HEAD `5c0bb34...` produced 530 passed and
  16 failed in 25.05 seconds. All failures are acceptance-state fixture defects
  in `test_protocol_amendments.py`, `test_qwen3_v2_gpu_preflight_handler.py`,
  and `test_qwen3_v2_g0_handler.py`: they read the checked-in accepted amendment
  as though it were still the proposed template. Production lineage,
  deployment, and handler checks still pass. A focused reproduction produced
  57 passed and the same 16 fixture-only failures. The minimal repair changes
  only those three test files, but committing it requires a new successor
  implementation/acceptance cycle and would invalidate the existing W=2
  evidence. It is therefore deferred until after the active Candidate E pilot
  lineage rather than treated as a production-path blocker;
- affected scheduler/scientific suite: 290/290 passed;
- deployment consistency suite: 26/26 passed;
- both disabled proposals passed the actual ServerScheduler
  `load_registration()` parser; their raw GPU profiles have policy `scheduler`
  and no `gpu_count` field;
- positive request fixtures contain only the allocation-neutral fields, while
  top-level resource fields and all parameter-hidden resource hints fail;
- all 68 changed or new Python files compiled in memory;
- both runtime `pip check` commands, all direct necessary hashes and deployment
  identities, the clean MIB/submodule check, forbidden-capability scan, and
  `git diff --check` passed.

The accepted-checkout complete-suite command was:

```bash
TMPDIR=/scr/del6500/OPD/tmp PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES= \
NVIDIA_VISIBLE_DEVICES=void \
/scr/del6500/OPD/envs/qwen3-v2-g0-v1/bin/python -c \
"import sys; p='/opt/anaconda3/lib/python3.13/site-packages'; sys.path.append(p); import pytest; sys.path.remove(p); raise SystemExit(pytest.main(sys.argv[1:]))" \
tests -q --tb=no
```

The two runtime checks were:

```bash
/scr/del6500/OPD/envs/qwen3-v2-g0-v1/bin/python -m pip check
/scr/del6500/OPD/envs/qwen3-v2-gpu-preflight-v1/bin/python -m pip check
```

The 1/2/3/4 fixtures remain static fail-closed evidence. Candidate E now also
has real, semantically accepted W=1 and W=2 GPU preflights; W=3 and W=4 have
not run successfully under Candidate E and remain required before G0.

During Candidate E construction no GPU was queried or used, no job was run or
submitted, no outbox file was generated or changed, no network was accessed,
and no central registration or service was modified. Read-only central config
and runtime checks were performed. The independent acceptance review likewise
did not query or use a GPU, submit a job, write an outbox file, modify central
configuration, or operate a service.

During the 2026-09-06 status audit no job, central state, or service was
changed. One read-only `doctor` attempt indirectly invoked `nvidia-smi`; the
sandboxed probe exited 9 and observed no GPUs, so it is not evidence of a host
fault. The accepted-checkout pytest audit wrote an ignored `.pytest_cache/`;
it was subsequently moved to the recoverable scratch path recorded above. The
strict accepted-lineage resolver then passed with `unsafe=()` at HEAD
`d9e8ab3...`.

After the central constraint deployment, this OPD session verified the current
accepted lineage at HEAD `68f845b...`, generated the fresh W=1 request recorded
below, and ran only its structural checks plus central `validate`. It did not
arm a constraint, submit a job, query or use a GPU, or change central
configuration or services. An initial request-readiness probe with system
Python failed immediately because that interpreter lacks PyYAML; the fixed OPD
runtime succeeded and no dependency was installed.

The subsequent W=1 completion audit read central durable state, audit events,
logs, and OPD artifacts and ran the current production evidence validator. It
did not submit, retry, cancel, or modify a job, constraint, queue, lease,
service, configuration, or GPU.

After the W=1 handoff commit, this OPD session verified the clean committed
HEAD `15606fd...` and accepted lineage, generated the fresh count-neutral W=3
request recorded below, and ran only request-structure checks plus central
`validate`. It did not arm a constraint, submit or modify a job, query or use a
GPU, or change central configuration or services.

After the W=3 request handoff commit, this OPD session verified that independent
W=3 and W=4 constrained preflights may be pending concurrently, generated the
fresh count-neutral W=4 request recorded below at clean HEAD `23dd9b4...`, and
ran only request-structure checks plus central `validate`. It did not arm a
constraint, submit or modify a job, query or use a GPU, or change central
configuration or services.

## Proposals, requests, and central state

Candidate E project-owned proposals are disabled:

- full G0 proposal:
  `/home/del6500/projects/OPD/deployments/qwen3_v2_g0/registration-proposal-v2.toml`,
  SHA-256
  `957fef1c0569c1dc3c1c9644e9be58c7ba5b3c2177c4f91249e383e10157aab1`;
- preflight-only proposal:
  `/home/del6500/projects/OPD/deployments/qwen3_v2_gpu_preflight/registration-proposal-v2.toml`,
  SHA-256
  `bdf8a0101d8eae1e3ccfaebfc10f55ed82b35c193289a3469dbc3d634a6d0d5b`.

A read-only check on 2026-09-06 shows that central
`config/projects/opd.toml` is Candidate E at SHA-256
`f43c0133e37b6eef394f70c18c9d7f30ce0080fa39482c3c456d37b439a7e8a0`,
with `enabled = true` and integration status
`protocol-v2-enabled-qwen3-v2-elastic-validation`. Relative to the disabled
project proposal, only enablement/status/notes differ. The central parser
reports both GPU profiles as scheduler-managed. Current live service PID could
not be queried during the earlier audit; the later constraint deployment and
service evidence are recorded below.

Candidate E real preflight coverage is currently:

| World size | Status |
| --- | --- |
| 1 | passed and semantically validated |
| 2 | passed and semantically validated |
| 3 | exact-count pilot submitted and pending |
| 4 | exact-count request prepared; not submitted |

The W=2 job is `opd-1f442a3c5c916a557a613b50f1459349`, attempt 1. It was
submitted at `2026-09-06T02:58:18Z`, claimed with 24 CPU cores, 196608 MiB and
two GPUs, and completed with exit code 0 at `2026-09-06T18:31:37Z`; scheduler
runtime was 42.5215 seconds. Its count-neutral request is
`/scr/del6500/OPD/scheduler/outbox/opd-1f442a3c5c916a557a613b50f1459349.json`
at SHA-256
`2cc8e4256d018ac19008a1b94733d69ebb3fecdf9955071f0d06d9af7177386b`.
The report is under workflow
`qwen3-v2-gpu-preflight-elastic-700b6054e4dcfff45a56841d6a90dbce`, has raw
SHA-256 `ca1edc3fd3053fc673e2d2093bd6dd61fe84f876d685e188dac47012205879f1`,
reports `passed = true`, `world_size = 2`, global batch 64, and binds accepted
HEAD `5c0bb34...`. Its completion SHA-256 is
`3760400728a55184dbd18a7b2e250ed919c76546d35fa8458b9f365a4d11671b`;
the current semantic validator accepts the report/completion pair.

The latest read-only central observation contains the pending W=3 preflight,
no running OPD job, and no OPD lease. No `qwen3_v2_g0` job exists.

ServerScheduler deployed the audited one-shot exact-job preflight constraint
at 2026-09-06 14:38 CDT and restarted the service as PID 3288957. Its central
suite passed 194/194. Only `OPD/qwen3_v2_gpu_preflight` is marked eligible;
`qwen3_v2_g0` is ineligible. The constraint stays outside the request, narrows
only the count, preserves normal admission and scheduler UUID placement, waits
without count fallback, is consumed after claim, and suppresses retry after
consumption. The operator must arm the exact job ID before submitting its
request.

The W=1 validation request was prepared at
`/scr/del6500/OPD/scheduler/outbox/opd-9834cc5d0f88c91a94911fd63f06ad60.json`.
Its SHA-256 is
`d10602952166c84e352287270df97ef24bdb7dcbba364587dd731a91d3fc9e6e`,
workflow is
`qwen3-v2-gpu-preflight-elastic-86b52b1018e178883638fa8667a3c407`, and plan
SHA-256 is
`b1f8bcaabe593566e0fc651f99114ce574f091ff9e8248816158072affbd7a72`.
The request has only the normal six top-level fields and the three scientific
parameters, omits `execution_profile` and `resources`, and passes the central
`validate` command. It was generated at clean project HEAD `68f845b...`.
The operator armed it for W=1 at `2026-09-06T19:51:11Z`, submitted it one
second later, and the scheduler consumed the constraint at claim with one
scheduler-selected UUID. Attempt 1 completed with exit code zero at
`19:51:59Z`; scheduler runtime was 44.237 seconds and the scientific interval
was 40.744 seconds. The constraint is consumed and the request must not be
submitted again.

The W=1 report is
`/data/del6500/OPD/workflows/outputs/qwen3-v2-gpu-preflight-elastic-86b52b1018e178883638fa8667a3c407/b1f8bcaabe593566e0fc651f99114ce574f091ff9e8248816158072affbd7a72/gpu-preflight/gpu_preflight.json`
at raw SHA-256
`592ba2a107bf185dd5821d3e14481c5a1c8e124cbebf8924655ee0de450543e2`.
Its completion is
`/data/del6500/OPD/workflows/completions/qwen3-v2-gpu-preflight-elastic-86b52b1018e178883638fa8667a3c407/b1f8bcaabe593566e0fc651f99114ce574f091ff9e8248816158072affbd7a72/gpu-preflight.json`
at raw SHA-256
`e4357b9fab59e27bae997ef2cc3c0edb70112200d1a5dae71862be190c168bb1`.
The current production evidence validator accepts the pair at HEAD
`c3d396b...`. The report proves one real GPU, global batch 64 as sixteen
microbatches of four, 98,304 model-input tokens, finite forward/backward,
nonzero AdamW update, teacher forward, the expected `FULL_SHARD` to `NO_SHARD`
single-rank behavior, FSDP save/resume, and memory/headroom gates.

The roughly 41-second completion is expected for this one-window preflight and
is consistent with the earlier W=2 runtime. It was not an empty or cached
success. A nonfatal NCCL RAS port warning and post-scope `memory.events`
telemetry warning were observed; the NCCL probe and project memory gates
passed, and there was no OOM or retry.

A fresh count-neutral W=3 validation request is prepared at
`/scr/del6500/OPD/scheduler/outbox/opd-3d20deb555e18b55a04873cc32769051.json`.
Its SHA-256 is
`1c5364ed8f540a486074f06a5a4f8322d041cd6bd85f5ffe8dad057e703babc2`,
job ID is `opd-3d20deb555e18b55a04873cc32769051`, and workflow is
`qwen3-v2-gpu-preflight-elastic-942cf1e93d1dbe2bea8327c3c364b531`.
Its plan is
`/data/del6500/OPD/workflows/plans/qwen3-v2-gpu-preflight-elastic-942cf1e93d1dbe2bea8327c3c364b531/104a13ce30dab4e3b1da8ef1f633ab514cdaa1a5a79eed16bbebc69370873e98.json`
at plan SHA-256
`104a13ce30dab4e3b1da8ef1f633ab514cdaa1a5a79eed16bbebc69370873e98`.
It was generated at clean project HEAD `15606fd...`, has exactly the normal six
top-level fields and three scientific parameters, omits `execution_profile`,
`resources`, and any GPU-count hint, and passes central `validate`. A central
operator subsequently armed this exact job ID for W=3 and submitted it at
`2026-09-06T20:33:14Z`. At the latest observation it remains `pending`, its
constraint remains `armed`, and it has no attempt, allocation, lease, or log.

A fresh count-neutral W=4 validation request is prepared at
`/scr/del6500/OPD/scheduler/outbox/opd-44c3d75c1e8b9180d4e08ef55f173379.json`.
Its SHA-256 is
`ba34e7e8f097408343127abd21a2bcdd8d9fb3c16191e3addefaa9961f94e4db`,
job ID is `opd-44c3d75c1e8b9180d4e08ef55f173379`, and workflow is
`qwen3-v2-gpu-preflight-elastic-8e82f3aa463921390718f4239b941ab3`.
Its plan is
`/data/del6500/OPD/workflows/plans/qwen3-v2-gpu-preflight-elastic-8e82f3aa463921390718f4239b941ab3/c1c12bc34ba38d99f77f49083922f72ddfa4ba9913cba4335fb282de59b37999.json`
at plan SHA-256
`c1c12bc34ba38d99f77f49083922f72ddfa4ba9913cba4335fb282de59b37999`.
It was generated at clean project HEAD `23dd9b4...`, has exactly the normal six
top-level fields and three scientific parameters, omits `execution_profile`,
`resources`, and any GPU-count hint, and passes central `validate`. No W=4
constraint has been armed and this request has not been submitted. A central
operator must arm this exact job ID for W=4 before submitting this exact path.
W=3 and W=4 may remain pending concurrently; the central one-GPU-job
concurrency ceiling means they will not run simultaneously.

The stale historical preflight outbox file
`/scr/del6500/OPD/scheduler/outbox/opd-0399c0625f43425dd03cfcc638d234f1.json`
still must not be submitted, edited, or reused. No Candidate E preflight or G0
request from that historical workflow may be reused. No Candidate E G0 request
or G0 output exists. The completed W=2 request must not be resubmitted. G0
request generation still requires the full distinct accepted-lineage
W=1/2/3/4 matrix. The local cache blocker has been cleared. File existence does
not constitute submission.

## Required next gates

1. After committing this handoff-only update, a central operator pre-arms job
   `opd-44c3d75c1e8b9180d4e08ef55f173379` for W=4, verifies the constraint, and
   only then submits its exact prepared outbox path. The existing W=3 job stays
   pending independently.
2. OPD monitors both jobs and semantically validates each terminal report and
   completion under its exact expected world size.
3. Only after the complete matrix is present may OPD generate one fresh G0 request. Its
   absolute path, SHA-256, job ID, project HEAD, clean worktree state, and test
   results must be reported; a central operator then validates and submits that
   exact file under a separate G0 approval.

The 16 accepted-state test-fixture failures remain recorded maintenance debt,
not evidence of a production handler failure. Do not start their successor
acceptance cycle while completing the current matrix, because doing so would
make the already accepted W=2 evidence ineligible and force all four pilots to
restart.

Until gates 1-2 complete, no G0 request can be generated or submitted. W=3 and
W=4 pilots are the active blockers; W=1 and W=2 are accepted.

## Quarantined legacy scheduler surface

The unsupported inventory remains: 20 files under `scripts/slurm`, eight
launch/supervision files under `scripts/production`, and three Python
pilot/finalizer modules listed in
`docs/refactor/legacy_scheduler_inventory.md`. They are not reachable from the
ServerScheduler handler registry and are not supported execution paths. No
local scheduler, queue, lock, lease, Screen/tmux fan-out, background daemon,
service, timer, or socket was introduced. Deletion remains a separately
approved cleanup after caller and external-operator inventory.

## Documentation map

- Operating rules: `AGENTS.md`
- Candidate E amendment: `prereg/amendments/qwen3_v2_g0_elastic_v1.yaml`
- GPU pilot contract: `docs/refactor/qwen3_v2_gpu_preflight_pilot.md`
- Legacy inventory: `docs/refactor/legacy_scheduler_inventory.md`
- Historical Candidate D amendment:
  `prereg/amendments/qwen3_v2_g0_2gpu_v1.yaml`
