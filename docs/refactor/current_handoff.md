# OPD current handoff

Last updated: 2026-09-03.

This is the single current-state handoff for the OPD refactor and scheduler
integration. `AGENTS.md` is authoritative for repository working rules and
ServerScheduler approval boundaries. Historical scientific-repair evidence is
kept under `docs/archive/`; it is not an execution guide.

## Current repository baseline

The current committed repository baseline is `667cc34` (`docs: require
cross-session handoff synchronization`). The current committed GPU
implementation baseline is `ba6820b` (`fix: use flat FSDP parameters in GPU
preflight`). Important preceding milestones are:

- `8b020de`: split scientific domains from the scheduler boundary.
- `0b96a8e`: add the CPU repository-preflight pilot.
- `3883086`: correct ambient-environment isolation for that pilot.
- `e52f7f5`: add the bounded Qwen3-v2 four-GPU preflight.
- `ecc36fb`: preserve virtual-environment interpreter identity at launch.
- `9c0aaf6`: harden the Blackwell NCCL preflight and diagnostics.
- `a48b918`: update the OPD agent and scheduler rules.
- `a3db7fa`: consolidate this handoff and archive stale reports.

The `ba6820b` GPU correction spans:

- `deployments/qwen3_v2_gpu_preflight/package-manifest.json`
- `docs/refactor/qwen3_v2_gpu_preflight_pilot.md`
- `scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py`
- `src/posttrain_circuits/scheduler_adapter/registry.py`
- `tests/unit/test_qwen3_v2_gpu_preflight_handler.py`

Three independent static reviews found no remaining code, scientific-contract,
or distributed-lifecycle blocker in that change. Review runs reported 61
passing related unit tests and 13 passing focused scheduler-contract tests. The
repository owner committed the reviewed change as `ba6820b`.

## Current architecture

Scientific code is separated into these domains:

- `datasets`: dataset construction, parsing, splitting, and rendering.
- `methods`: SFT, OPD, and RL method implementations.
- `learning`: shared training and optimization mechanics.
- `causal_circuits`: circuit discovery, interventions, and validation.
- `experiments/protocols`: explicit scientific protocols and experiment
  composition.
- `analysis`: downstream analysis and reporting.
- `artifacts`: typed scientific artifact contracts and validation.
- `scheduler_adapter`: the narrow project-owned ServerScheduler boundary.

Only two tasks have completed the repository-side migration into registered
handlers: `repository_preflight` and `qwen3_v2_gpu_preflight`. Legacy Slurm and
local scheduler scripts are historical inventory, not supported execution
paths.

## Adapter contract retained from the refactor

- The stable entrypoint accepts one absolute protocol-v2 job-manifest path. It
  validates the manifest, allowlisted parameters, allocation, execution
  profile, and scheduler-provided environment before dispatch.
- OPD consumes the assigned GPU visibility and never selects host GPU indices.
  There is no project-local queue, lease, retry loop, detached worker, or
  resource scheduler.
- Workflow plans carry scientific identities rather than shell commands or
  resource-selection policy. Inputs are resolved from project-owned immutable
  storage; handler specifications own execution and completion semantics.
- Each migrated task binds a fixed deployment contract and execution profile.
  The child remains in the foreground, receives an explicit environment, and
  writes only to the validated attempt/output locations supplied by the
  adapter.
- Scheduler success alone is insufficient. The handler must produce the exact
  declared outputs and a fresh `ScientificCompletion` accepted by its
  code-owned semantic validator.
- The project outbox helper only prepares a schema-valid request for a migrated
  task. It does not submit, poll, retry, edit central registration, or manage a
  scheduler service.

The former adapter-status document described the CPU-only registry before the
GPU preflight migration and included obsolete launcher details. The current
source of truth is the adapter code, its tests, the deployment manifests, and
the task-specific pilot document linked below.

## Scheduler integration state

A read-only inspection on 2026-09-03 found the central OPD registration enabled
for exactly these two tasks and profiles:

- `repository_preflight` with `repository-preflight-cpu`: 1 CPU, 128 MiB RAM,
  no GPU, 30-second estimate, shareable.
- `qwen3_v2_gpu_preflight` with `qwen3-v2-gpu-preflight-4gpu`: 16 CPUs,
  196608 MiB RAM, 4 GPUs, 81920 MiB per GPU, 95% utilization target,
  exclusive allocation, RTX PRO 6000 Blackwell, 1800-second estimate.

Central registration is external mutable state. Reverify it in the central
ServerScheduler session before any future submission. This repository does not
authorize registration edits, service operations, or submission.

## Execution evidence

- CPU repository preflight `opd-a5248a93209c909b6a2f827120ca5771`
  completed with exit code 0 on its first attempt.
- GPU preflight `opd-94ac576dba1d42437c0f2ee5e5dc8c5f` failed after
  launching the virtual-environment Python through `/proc/self/fd`; `ecc36fb`
  corrected that launcher behavior.
- GPU preflight `opd-6c24ac368ce726229dd463d634b26c89` failed after the
  first NCCL scalar all-reduce stalled and workers received `SIGABRT`;
  `9c0aaf6` added the bounded Blackwell/NCCL diagnostics and mitigation.
- A later request used job ID `opd-929192edc17cd806cb184a8a6013e0b3`
  and plan SHA-256
  `f0382310adbd3fa0c4da873a3bd6227797d305bcce5c9d432b0bdc6a69eb05c0`.
  Its three attempts failed after teacher forward when FSDP exposed empty or
  one-dimensional original parameter shards to the student forward.
- `ba6820b` then changed the fully trainable student to one root FSDP unit with
  `use_orig_params=false`, restricted the optimizer to its one nonempty flat
  parameter shard per rank, and added phase-boundary diagnostics.
- A fresh protocol-v2 request for `ba6820b` was prepared at
  `/scr/del6500/OPD/scheduler/outbox/opd-d23aed0275b16bb906b9c94cb42067a1.json`
  and was centrally submitted. All three attempts on 2026-09-03 passed CUDA,
  NCCL, model loading, teacher forward, and student forward, then failed during
  `loss.backward()` with a `setStorage` access through a 2048-by-2048 parameter
  view whose backing storage had size zero. The resulting illegal CUDA memory
  access ended each attempt with `SIGABRT`; this request is terminal and must
  not be reused.
- The current uncommitted candidate explicitly selects non-reentrant student
  activation checkpointing. This retains checkpointing while avoiding the
  reentrant backward recomputation path through released FSDP flat-parameter
  views. Seventy-three related tests pass, and a tiny Qwen3 model completes a
  real forward/backward with this mode in the fixed Transformers runtime. No
  four-GPU pilot has run for this candidate.

After `9c0aaf6`, the focused repository test suite reported 89 passing tests and
the fixed runtime reported NCCL 2.27.3. The current implementation keeps a Gloo
control group, creates a separate NCCL data group for CUDA tensors and FSDP,
bounds the first NCCL probe, synchronizes distributed failure decisions, and
avoids unsafe NCCL teardown after an unsynchronized failure. Its deployment
contract pins CUDA 12.8 and NCCL 2.27.3. These are static and test-backed
results; successful behavior on the real four-GPU topology still requires a
fresh pilot.

## Current scientific gate

The formal G0 experiment is not ready to submit. The gate opens only after all
of the following are true:

1. Review and commit the non-reentrant checkpointing candidate in a clean
   project checkout.
2. Prepare one new protocol-v2 outbox request with a fresh job ID; do not reuse
   `opd-d23aed0275b16bb906b9c94cb42067a1`.
3. Have the central operator reverify registration, GPU dispatch, and device
   safety state before separately authorizing validation and submission.
4. Capture the authoritative terminal state and complete logs for that run.
5. If it fails, diagnose the exact phase and make only the required
   project-owned correction before forming another clean commit.
6. If it succeeds, independently validate the published scientific completion
   and `gpu_preflight.json` artifact.
7. Only then prepare the separately reviewed G0 handler/profile/request; the
   GPU preflight task itself is not the formal experiment.

Do not submit G0, the seed-42 pilot, a full seed matrix, or a Gemma replication
until these gates and their separately required approvals are satisfied.

## Documentation map

- Current operating rules: [`AGENTS.md`](../../AGENTS.md)
- Current GPU pilot contract:
  [`qwen3_v2_gpu_preflight_pilot.md`](qwen3_v2_gpu_preflight_pilot.md)
- Core-v2 compatibility notes: [`core_v2_migration.md`](../core_v2_migration.md)
- Refactor baseline: [`phase0_baseline.md`](phase0_baseline.md)
- Legacy scheduler inventory:
  [`legacy_scheduler_inventory.md`](legacy_scheduler_inventory.md)
- Historical scientific repair gap analysis:
  [`scientific_repair_v2_gap_analysis.md`](../archive/scientific_repair_v2/scientific_repair_v2_gap_analysis.md)
- Historical scientific repair report:
  [`scientific_repair_v2_report.md`](../archive/scientific_repair_v2/scientific_repair_v2_report.md)
