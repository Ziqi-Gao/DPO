# OPD current handoff

Last updated: 2026-09-03.

This is the single current-state handoff for the OPD refactor and scheduler
integration. `AGENTS.md` is authoritative for repository working rules and
ServerScheduler approval boundaries. Historical scientific-repair evidence is
kept under `docs/archive/`; it is not an execution guide.

## Current repository baseline

The current committed repository and GPU implementation baseline is `252b02a`
(`feat: convert GPU preflight to two GPUs`). Important preceding milestones
are:

- `8b020de`: split scientific domains from the scheduler boundary.
- `0b96a8e`: add the CPU repository-preflight pilot.
- `3883086`: correct ambient-environment isolation for that pilot.
- `e52f7f5`: add the bounded Qwen3-v2 four-GPU preflight.
- `ecc36fb`: preserve virtual-environment interpreter identity at launch.
- `9c0aaf6`: harden the Blackwell NCCL preflight and diagnostics.
- `a48b918`: update the OPD agent and scheduler rules.
- `a3db7fa`: consolidate this handoff and archive stale reports.
- `ba6820b`: use flat FSDP parameters in the GPU preflight.
- `667cc34`: require cross-session handoff synchronization.

The current committed GPU correction spans:

- `deployments/qwen3_v2_gpu_preflight/package-manifest.json`
- `docs/refactor/qwen3_v2_gpu_preflight_pilot.md`
- `scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py`
- `src/posttrain_circuits/scheduler_adapter/registry.py`
- `tests/unit/test_qwen3_v2_gpu_preflight_handler.py`

It keeps the root-FSDP flat-parameter and non-reentrant activation-checkpointing
corrections, and converts only the bounded GPU-preflight task from four
ranks/GPUs to two. The complete change also includes `AGENTS.md`, the disabled
registration proposal, request builder, semantic validator, and their focused
tests. It retains 16 CPU cores and 196608 MiB host memory, uses eight CPU
threads per rank, retains 81920 MiB and 95% utilization per exclusive GPU, and
uses a conservative 3600-second initial runtime. Its plan identity is
`20fbb3315e873ed618d536412f9a53c1928248708280ef217725a17caed4f0be`.

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
- `qwen3_v2_gpu_preflight` with `qwen3-v2-gpu-preflight-2gpu`: 16 CPUs,
  196608 MiB RAM, 2 GPUs, 81920 MiB per GPU, 95% utilization target,
  exclusive allocation, RTX PRO 6000 Blackwell, 3600-second estimate.

Central registration is external mutable state. Reverify it in the central
ServerScheduler session before any future submission. This repository does not
authorize registration edits, service operations, or submission.

The central OPD registration was independently updated and enabled for the
two-GPU profile after user approval. The deployed scheduler now launches each
job in an allocation-specific systemd scope: the authoritative running-manifest
`allocation.memory_mib` becomes the exact `MemoryMax`, swap is disabled, and a
launch guard verifies the cgroup before project code runs. Request-side
`resources` is an optional sparse hint object, but current OPD requests must
continue to omit it. OPD consumes only the documented public
`SERVER_SCHEDULER_*` environment and must not depend on private guard metadata
such as `SERVER_SCHEDULER_MEMORY_MAX_BYTES` or
`SERVER_SCHEDULER_RUNTIME_UNIT`.

The checked-in project-owned proposal remains disabled as a handoff artifact;
this session did not edit the central registration or global dispatch policy.
Central registration, dispatch, and GPU safety state remain external mutable
state and must be reverified by a ServerScheduler operator before submission.

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
- `a6f26fe` committed the non-reentrant student activation-checkpointing
  correction. Request `opd-21968766ca6a6a4e306c7995f2815593`, with outbox
  SHA-256
  `f60f9b98505d722ceef9fd7916ee16ef9fa090bd784020c3286ec333ad2d70cb`,
  was centrally accepted at `2026-09-03T20:35:02Z`. The user withdrew it
  because only two GPUs were available, and its project-owned request copy was
  deleted after digest verification. A central operator then canceled it
  before launch; its terminal state is `failed`, with no allocation, logs, or
  process, and failure reason `operator canceled before launch`. Its published
  plan identity remains
  `f0382310adbd3fa0c4da873a3bd6227797d305bcce5c9d432b0bdc6a69eb05c0`.
  No four-GPU pilot has run for this committed correction.

The committed two-GPU implementation passed 20 focused tests and 74 related
handler, validator, request, adapter, and proposal tests. Syntax, the exact
two-GPU profile, disabled proposal, package manifest, and all deployment hashes
also validated. A tiny Qwen3 model completed a real non-reentrant
forward/backward in the fixed runtime without using a GPU. These were the
static results before the first real two-GPU allocation.

Two-GPU request `opd-d4b63d5a246ed8d06b944b9063a62e1e`, originally written
to `/scr/del6500/OPD/scheduler/outbox/opd-d4b63d5a246ed8d06b944b9063a62e1e.json`
with SHA-256
`8500c5f459950a008fede57e8f7a63bedc668c0267b9794a2cc218b39f53c4ab`,
was centrally submitted at `2026-09-03T21:19:17Z`. Its only attempt completed
the CUDA and NCCL probes, both model loads, teacher and student forwards,
student backward, optimizer step, nonzero update checks, and FSDP save/resume
on the assigned two-GPU topology. Both ranks reached
`rank_training_completed`. Final publication then failed because the old
central launch path exposed the service cgroup's unlimited `memory.max`
instead of an allocation-specific 196608 MiB limit. The durable scheduler
state became terminal `failed` at `2026-09-03T21:27:04Z`; the job ID is
permanently occupied and must never be resubmitted or reused. This is evidence
of a then-live central isolation failure, not evidence requiring an OPD
scientific-code change.

The historical `gpu_fatal` label on that attempt was also a central classifier
false positive: it matched the successful
`phase=nccl_probe_passed ... timeout_seconds=120` log line. The line-local
classifier correction and allocation-specific no-swap systemd scope/guard were
subsequently deployed. They do not change the terminal state or make the old
job ID reusable. The independently approved replacement pilot is recorded
below.

From clean committed HEAD `0730e566144f0482bd80a06b147d4bcf6d7f4379`, the
project builder prepared exactly one replacement request:

- Job ID: `opd-23f0d147b3eae04a4a6b402f54e18948`.
- Outbox path:
  `/scr/del6500/OPD/scheduler/outbox/opd-23f0d147b3eae04a4a6b402f54e18948.json`.
- Request SHA-256:
  `f5a4b30e1870587355bf90e2d10205e88749d5f64405325e31f69459dbe2ea70`.
- Task/profile: `qwen3_v2_gpu_preflight` with
  `qwen3-v2-gpu-preflight-2gpu`.
- Scientific identity: workflow `qwen3-v2-gpu-preflight-v1`, unit
  `gpu-preflight`, and plan
  `20fbb3315e873ed618d536412f9a53c1928248708280ef217725a17caed4f0be`.
- Priority is zero. The request omits `resources` and contains no command,
  path, or environment override.

The exact preparation and verification commands were:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3.12 -m posttrain_circuits.scheduler_adapter.gpu_preflight_request
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/del6500/projects/ServerScheduler/src /usr/bin/python3.12 -c 'from pathlib import Path; from server_scheduler.models import JobRequest; JobRequest.from_file(Path("/scr/del6500/OPD/scheduler/outbox/opd-23f0d147b3eae04a4a6b402f54e18948.json")); print("protocol-v2 schema/parser: ok")'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3.12 -c 'from posttrain_circuits.scheduler_adapter.gpu_preflight_request import build_qwen3_v2_gpu_preflight_plan; from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout; from posttrain_circuits.scheduler_adapter.plan_store import require_published_workflow_plan; layout=WorkflowLayout.production(); require_published_workflow_plan(build_qwen3_v2_gpu_preflight_plan(layout=layout), layout=layout); print("published plan: ok")'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3.12 -c 'from pathlib import Path; from posttrain_circuits.scheduler_adapter.outbox import validate_outbox_request; from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json; payload,_=read_strict_json(Path("/scr/del6500/OPD/scheduler/outbox/opd-23f0d147b3eae04a4a6b402f54e18948.json"), context="prepared request", max_bytes=131072); validate_outbox_request(payload); print("OPD request: ok")'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3.12 -m unittest tests.unit.test_gpu_preflight_request tests.unit.test_qwen3_v2_gpu_preflight tests.unit.test_qwen3_v2_gpu_preflight_handler tests.unit.test_scheduler_adapter tests.unit.test_registration_proposal -v
sha256sum /scr/del6500/OPD/scheduler/outbox/opd-23f0d147b3eae04a4a6b402f54e18948.json
```

The schema/parser, published-plan, and project request validations passed; all
74 related tests passed in 0.874 seconds; and the final digest matched the value
above. The new job ID was absent from central durable jobs/history when first
checked.

The replacement request was later centrally submitted at
`2026-09-04T03:35:34Z` and completed on its first attempt with exit code zero;
central terminal history was updated at `2026-09-04T03:36:02Z`. It used the
registered 16-CPU, 196608-MiB, two-exclusive-GPU allocation on the allowed GPU
0/1 UUIDs. Both ranks completed the CUDA and 0.192-second NCCL probes, offline
model loads, finite teacher/student forwards and soft-teacher losses, finite
backward gradients, nonzero optimizer updates, unique prompt shards, and FSDP
save/resume.

The published report is
`/data/del6500/OPD/workflows/outputs/qwen3-v2-gpu-preflight-v1/20fbb3315e873ed618d536412f9a53c1928248708280ef217725a17caed4f0be/gpu-preflight/gpu_preflight.json`
with file SHA-256
`c0a8d1dcf66cb278eb522ccba21382728cbe72651496cc0bf2599a7bd386b3b3`.
It records `passed: true`, exact 192-GiB cgroup limit
`206158430208`, peak `66826952704`, and remaining headroom
`139331477504` bytes. The `ScientificCompletion` is
`/data/del6500/OPD/workflows/completions/qwen3-v2-gpu-preflight-v1/20fbb3315e873ed618d536412f9a53c1928248708280ef217725a17caed4f0be/gpu-preflight.json`
with file SHA-256
`85951e88006a7a2881eb4c4fb0bdbf29a130014c4a22bf75a53303e3ab03bdc2`.
It binds attempt 1 and code commit
`634d3e86cbfa16689837cc595f48302be1f862d8` and declares all eight reviewed
scientific gates true.

The project acceptance command used the production
`validate_unit_completion` path against the immutable published plan and
returned `OPD semantic completion validation: accepted`. That path revalidated
the completion schema, plan/input identities, output hash, execution
provenance, fixed Qwen3-v2 report contract, and every handler-owned semantic
gate. The NCCL RAS listener reported a non-blocking localhost port warning;
the NCCL data operation and clean communicator teardown still completed. This
OPD acceptance session performed no submission, GPU operation, or central
scheduler mutation.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /usr/bin/python3.12 -c 'from pathlib import Path; from posttrain_circuits.scheduler_adapter.completion import validate_unit_completion; from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout; from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json; from posttrain_circuits.workflows.contracts import WORKFLOW_TASK_REGISTRY, WorkflowPlan; layout=WorkflowLayout.production(); plan_path=Path("/data/del6500/OPD/workflows/plans/qwen3-v2-gpu-preflight-v1/20fbb3315e873ed618d536412f9a53c1928248708280ef217725a17caed4f0be.json"); raw,_=read_strict_json(plan_path, context="published OPD workflow plan", max_bytes=16*1024*1024); plan=WorkflowPlan.from_payload(raw, task_registry=WORKFLOW_TASK_REGISTRY); unit=plan.unit("gpu-preflight", task_registry=WORKFLOW_TASK_REGISTRY); completion=validate_unit_completion(plan, unit, plan_sha256="20fbb3315e873ed618d536412f9a53c1928248708280ef217725a17caed4f0be", layout=layout, expected_execution=None); assert completion["execution"]["job_id"]=="opd-23f0d147b3eae04a4a6b402f54e18948"; assert completion["execution"]["attempt"]==1; print("OPD semantic completion validation: accepted")'
```

After `9c0aaf6`, the focused repository test suite reported 89 passing tests and
the fixed runtime reported NCCL 2.27.3. The current implementation keeps a Gloo
control group, creates a separate NCCL data group for CUDA tensors and FSDP,
bounds the first NCCL probe, synchronizes distributed failure decisions, and
avoids unsafe NCCL teardown after an unsynchronized failure. Its deployment
contract pins CUDA 12.8 and NCCL 2.27.3. These are static and test-backed
results. The successful replacement pilot now validates that bounded preflight
path on the real two-GPU topology; it does not validate or authorize G0.

## Current scientific gate

The bounded two-GPU preflight gate has passed. The formal G0 experiment is
still fail-closed and is not ready to submit. The next allowed project work is:

1. Commit this acceptance handoff so the project checkout is clean.
2. Design and review a distinct two-GPU G0 handler, profile, fixed runtime,
   semantic validator, tests, and disabled registration proposal.
3. Obtain the separate central installation, enablement, request preparation,
   and submission approvals required for that new task.

The successful preflight is evidence for the two-GPU execution substrate only;
it is not approval for G0, seed-42 training, the full seed matrix, replication,
or cleanup.

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
