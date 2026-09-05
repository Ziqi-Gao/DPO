# OPD current handoff

Last updated: 2026-09-05.

This is the canonical current-state summary for the OPD refactor and
ServerScheduler integration. `AGENTS.md` is authoritative for operating and
approval rules. Source, Git, scientific artifacts, and central scheduler state
must still be verified when mutable.

## Current repository state

Committed HEAD is `778e6a4a9a70a7d5d190fe8710e997a3e05bb142` on `master`,
33 commits ahead of `origin/master`. It records the rejection of candidate C.
The current working tree is an uncommitted replacement implementation,
candidate D. Its amendment remains proposed with SHA-256
`2129555c7ee71e68bedd87aafd34879f32c624e21bc7e143850c5fa1d30d6686`.
Candidate D is ready for a user-created implementation commit, but is not an
accepted implementation and cannot yet generate or submit a request.

Historical candidate status:

- Candidate A `ab2f72f20036b6f488db5d5a335744a6a80b8b82` was rejected
  because runtime verification wrote EAP bytecode into the immutable MIB tree.
- Candidate B `e8138997ad9465e986e1e92ffcf7fd61e8c113e8` was independently
  accepted by review-only commit
  `71a86997cc3eadf55d18b9a5d0405eaaa9ada5ac`. That acceptance remains
  historical evidence only and does not authorize later implementation changes.
- Candidate C `b4c63a384fabce6ece950ea9988183da209e7652` was rejected because
  both GPU handlers imported the mutable checkout's lineage validator before
  authenticating that checkout, and `python -I` ignored its environment-only
  bytecode prohibition.

Candidate D changes these paths:

- `deployments/qwen3_v2_g0/package-manifest.json`
- `deployments/qwen3_v2_gpu_preflight/package-manifest.json`
- `scripts/server_scheduler/opd-entrypoint`
- `scripts/server_scheduler/qwen3-v2-g0-handler.py`
- `scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py`
- `src/posttrain_circuits/artifacts/git_provenance.py`
- `src/posttrain_circuits/artifacts/protocol_amendments.py`
- `src/posttrain_circuits/scheduler_adapter/g0_request.py`
- `src/posttrain_circuits/scheduler_adapter/gpu_preflight_request.py`
- `src/posttrain_circuits/scheduler_adapter/registry.py`
- `tests/unit/test_g0_request.py`
- `tests/unit/test_gpu_preflight_request.py`
- `tests/unit/test_protocol_amendments.py`
- `tests/unit/test_qwen3_v2_g0_handler.py`
- `tests/unit/test_qwen3_v2_gpu_preflight_handler.py`
- `tests/unit/test_scheduler_adapter.py`
- this handoff.

## Candidate D trust and execution boundary

The GPU-preflight handler now carries its complete acceptance-lineage bootstrap
inside the hash-bound handler and imports no project source. The G0 supervisor
and every scientific child independently run the same kind of hash-bound
bootstrap before adding project `src` to `sys.path`. The shared validator then
runs only as defense in depth.

Lineage validation now:

- fixes the proposed amendment bytes by SHA-256 and permits exactly one
  proposed-to-accepted review transition;
- enumerates every commit and separately reads its real parent identity, so
  history simplification cannot hide a merge parent;
- requires a single-parent chain, limits each range to 256 commits, rejects
  empty commits, second amendment changes, source changes followed by reverts,
  and any path other than the amendment/handoff at acceptance or handoff alone
  after acceptance;
- disables global/system Git configuration, replace objects, optional locks,
  repository fsmonitor helpers, and external diffs; changed paths use NUL
  delimiters and strict UTF-8;
- requires a completely clean checkout including submodules and separately
  enumerates untracked files without applying `.gitignore` or
  `.git/info/exclude`; only the non-importable local `.codex/config.toml` and
  isolated source-tree `__pycache__/*.pyc` files are allowed;
- makes the absolute entrypoint perform both clean checks before adding project
  `src` to `sys.path`, then rechecks HEAD/source identity in the hash-bound
  handlers before import, after each G0 scientific child, and before publication;
- opens the source amendment with `O_NOFOLLOW`, holds its descriptor through
  validation, and rechecks inode, bytes, path identity, and the HEAD blob.

`opd-entrypoint` and both GPU handlers establish `sys.dont_write_bytecode` and
an atomically created private scratch `sys.pycache_prefix` before project-source
imports. This prevents both reading stale source-tree bytecode and writing new
bytecode even under isolated Python, where `PYTHONDONTWRITEBYTECODE` is ignored.
The environment-only setting was removed from both deployment contracts.

The only central entrypoint remains:

`/home/del6500/projects/OPD/scripts/server_scheduler/opd-entrypoint`

It validates the running manifest and concrete allocation, preserves
`CUDA_VISIBLE_DEVICES`, consumes the scheduler CPU/GPU count contract, and
dispatches only the code-owned handler registry. The project does not select
physical GPUs, query `nvidia-smi`, maintain locks/leases/queues, fan out with
Screen/tmux, detach workers, or run a local scheduler service.

## Fixed two-GPU scientific scope

Both GPU tasks are intentionally fixed, not scheduler-sized:

| Task | Profile | Allocation | Deployment |
| --- | --- | --- | --- |
| `qwen3_v2_gpu_preflight` | `qwen3-v2-gpu-preflight-2gpu` | 16 CPU cores, 196608 MiB, two exclusive RTX PRO 6000 Blackwell GPUs | `qwen3-v2-gpu-preflight-2gpu-python312-cuda-v5` |
| `qwen3_v2_g0` | `qwen3-v2-g0-2gpu` | 16 CPU cores, 196608 MiB, two exclusive RTX PRO 6000 Blackwell GPUs | `qwen3-v2-g0-2gpu-python312-cuda-v3` |

Neither task may use `gpu_count_policy = "scheduler"`. OPD has not established
equivalent 1/2/3/4-GPU scientific semantics, one-GPU memory safety, three-rank
batch/RNG/thread equivalence, or cross-world-size checkpoint resharding. Both
handlers reject every allocation other than exactly two GPUs and use only
logical CUDA devices 0 and 1 inside the scheduler-provided visibility envelope.

The G0 amendment retains world size 2, per-device batch 4, gradient
accumulation 8, and effective global batch `2 * 4 * 8 = 64`. Before every
optimizer boundary it reserves the exact cross-rank sum of non-padding model
input tokens and skips the update that would exceed 2,000,000 tokens. The 120
optimizer-step limit is an independent ceiling. Checkpoints bind consumed
tokens and world size; cross-world-size resume is rejected. The scientific
claim is seed-42 full-pipeline feasibility only, not a confirmatory endpoint,
three-seed factorial, or Gemma replication.

## Deployment and runtime identities

GPU preflight:

- handler SHA-256:
  `3dd7f607cb760b0bf6dd12e2000c9f826b86a30e24327131d4966fd2f0656347`
- dependency lock SHA-256:
  `c72a990c03b9d841d399d9e0497912fcfd6c78ce3e840406ecd98c921c2b893f`
- package manifest SHA-256:
  `ca3a8ca2db8d8618492c5b1004cfe2cbf9a5867d8469eee00ba21b066f4c2dac`
- deployment identity:
  `52357303f24328d3cd5ebe4b3174840d03fb4eacc657f87e2d6d863726594a82`

G0:

- handler SHA-256:
  `09b53ba8f5cbb2742d131f482d6046dbb2b09c1ae07417dc80c1740ddb7a3ac0`
- dependency lock SHA-256:
  `18d77da454cb86a097526fe561462f4818cb76c3ca9633481261f1e563f86d90`
- package manifest SHA-256:
  `7f3b74484d37bd4b4ee4fbc953b251b5c8ad04b177d5623ba3517d3f784dc2d7`
- deployment identity:
  `d33e4949dd91f2dbcf280097aca77a0939d6bb16cc898c42ad215282a9c65a6a`

Both deployment identity checks and all direct file hashes match. Both fixed
runtimes pass `pip check`. No-GPU production-boundary calls to both handlers'
`_validate_environment()` passed with two virtual CUDA UUIDs. The immutable G0
runtime and MIB roots remain mode 550. MIB is pinned at
`b759df34433c9e31043ba9e02908ce0bf20e894f`, EAP-IG at
`7af394a5662de8b23ad6154716a0cd3993d447a3`, and the complete MIB tree
currently contains zero `.pyc` files.

## Verification

Candidate D static/runtime-boundary results:

- GPU-preflight handler suite: 21/21 passed, including real Git
  source-change/revert and range-hidden merge-parent fixtures.
- G0 handler suite: 19/19 passed, including supervisor and scientific-child
  bootstrap order, amendment tampering, merge/revert, held-file, bytecode, and
  historical accepted-lineage probes.
- Shared amendment suite: 21/21 passed.
- Request/amendment focused suite: 31/31 passed.
- Scheduler adapter suite: 50/50 passed.
- Combined handler/request/validator/adapter/proposal/workflow suite: 148/148
  passed in 7.745 seconds; an independent audit rerun also passed 148/148 in
  7.601 seconds.
- Both runtime `pip check` commands passed; deployment hashes and identities,
  in-memory compilation, and `git diff --check` passed.
- The current ServerScheduler proposal parser accepted both checked-in
  proposals, reported `enabled = false`, and found exactly the declared tasks.

An additional whole-unit `unittest discover` attempt ran 190 test cases but
ended with 26 collection errors because those pytest-authored modules import
`pytest`, which is intentionally absent from the immutable G0 runtime. A
separate scratch-only installation of the project-pinned `pytest==8.4.1` was
attempted both in and outside the sandbox; both failed because the host could
not resolve PyPI. The scratch target was not created. This does not invalidate
the 148-test affected-surface suite and did not modify either runtime.

No GPU computation, scheduler job, outbox write, central registration change,
service operation, or process management occurred during candidate D work.
The only external action was the unsuccessful outbound PyPI download attempt.

## Proposal, requests, and central state

The project-owned proposals remain unchanged and disabled:

- full G0 proposal:
  `/home/del6500/projects/OPD/deployments/qwen3_v2_g0/registration-proposal-v2.toml`,
  SHA-256
  `4f5ad58d58c1c10e9e352d66f6cf8655977e6fdbe0913cac53fd53cb293cdae1`;
- preflight-only proposal:
  `/home/del6500/projects/OPD/deployments/qwen3_v2_gpu_preflight/registration-proposal-v2.toml`,
  SHA-256
  `275035b96e9927de7b7df601e675665789562af0e45fc44a7c8f682a8392217c`.

They declare `enabled = false`. Registration installation, enablement, and
service state are external and were not reverified or changed in candidate D.
Normal generated requests contain only the project/task/priority/fresh job ID
and scientific identities `workflow_id`, `plan_sha256`, and `unit_id`; they
omit `execution_profile`, `resources`, GPU count, device, memory, utilization,
and exclusivity fields.

The old preflight outbox file still exists at
`/scr/del6500/OPD/scheduler/outbox/opd-0399c0625f43425dd03cfcc638d234f1.json`
with SHA-256
`b4d3322ad96145a0de8e07eb686172373bbbf3d269a0c3bcb0fcfb061ae70536`.
It was never submitted, is permanently stale because it predates candidates C
and D, and must not be submitted, edited, or reused. No candidate D request or
G0 request exists. File existence is not submission.

The prior successful two-GPU preflight remains valid evidence for the older
implementation substrate only. Its job was
`opd-23f0d147b3eae04a4a6b402f54e18948`; report SHA-256 was
`c0a8d1dcf66cb278eb522ccba21382728cbe72651496cc0bf2599a7bd386b3b3`
and completion SHA-256 was
`85951e88006a7a2881eb4c4fb0bdbf29a130014c4a22bf75a53303e3ab03bdc2`.
Because candidate D changes handlers and lineage logic, a fresh accepted-lineage
preflight must run and pass before a G0 request can be created.

## Required next gates

1. The user creates one clean candidate D implementation commit containing the
   proposed amendment and all paths listed above.
2. A separate independent scientific review validates that exact commit. Only
   after acceptance may a second commit change the amendment review block and,
   if needed, this handoff; it must bind the exact candidate D commit.
3. A central operator reviews/installs the unchanged disabled proposal, and the
   user separately approves registration enablement.
4. From a clean accepted lineage, prepare exactly one fresh two-GPU preflight
   request; report its absolute path, SHA-256, fresh job ID, project HEAD,
   worktree state, and tests. A central operator validates and submits that
   exact file.
5. Follow the preflight to terminal state and accept it only through OPD's
   semantic completion validator.
6. From the same clean accepted lineage, prepare one fresh G0 request bound to
   that accepted preflight. A central operator validates/submits it only after
   separate G0 approval.

Until gates 1-5 complete, no Qwen3-v2 G0 job may be submitted.

## Quarantined legacy scheduler surface

The unsupported legacy inventory remains unchanged: 20 files under
`scripts/slurm`, eight launch/supervision files under `scripts/production`, and
three Python pilot/finalizer modules listed in
`docs/refactor/legacy_scheduler_inventory.md`. They are not reachable from the
ServerScheduler handler registry and are not supported execution paths. No
local service/timer/socket exists in the repository. Deletion remains a
separate cleanup requiring caller cutover, work drain, rollback evidence,
external operator inventory, and explicit user approval.

## Documentation map

- Operating rules: `AGENTS.md`
- G0 amendment: `prereg/amendments/qwen3_v2_g0_2gpu_v1.yaml`
- G0 review notes: `docs/refactor/qwen3_v2_g0_2gpu_amendment_review.md`
- GPU pilot contract: `docs/refactor/qwen3_v2_gpu_preflight_pilot.md`
- Legacy inventory: `docs/refactor/legacy_scheduler_inventory.md`
