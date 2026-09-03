# OPD Agent Guide

This file describes the current OPD repository after the scientific-domain and
ServerScheduler refactor. It is authoritative for work performed from an OPD
project session. It does not grant authority over the central scheduler.

## Mandatory ServerScheduler contract

Before changing or running OPD code, read these files in full:

- `/home/del6500/projects/ServerScheduler/docs/project-integration.md`
- `/home/del6500/projects/ServerScheduler/docs/job-contract.md`
- `/home/del6500/projects/ServerScheduler/docs/project-migration-instructions.md`
- `/home/del6500/projects/ServerScheduler/schemas/job-request-v2.schema.json`

The project owns scientific behavior, task validation, execution profiles,
checkpointing, outputs, and scientific completion. ServerScheduler alone owns
queue order, physical CPU/GPU placement, leases, retries, status, logs, runtime
calibration, and cross-project concurrency.

## Filesystem and authority boundaries

OPD may write only these roots:

- code: `/home/del6500/projects/OPD`
- durable data: `/data/del6500/OPD`
- scratch/cache/outbox: `/scr/del6500/OPD`

Use `/scr/del6500/OPD/tmp` for temporary files. Reading central scheduler
configuration, status, and logs for diagnosis is allowed when the sandbox
permits it. Never edit ServerScheduler source, configuration, registrations,
schemas, queues, leases, audit records, performance data, history, or service
files from an OPD session.

Do not start, stop, enable, disable, restart, or signal a system or user
service. Report any required central or host operation to the user for a
separate operator session. BIOS, IOMMU, ACS, driver, and host topology changes
are also external operations.

Do not create a Git worktree. Do not run `git add`, `git commit`, `git amend`,
or rewrite history; give the user exact commands instead. Preserve unrelated
user changes in a dirty checkout.

## Submission and approval rules

An OPD project session may implement handlers, validate contracts, create a
disabled registration proposal, and prepare a project-owned protocol-v2
outbox request. It must not run ServerScheduler `submit`, `dispatch`, or
`serve`, and must not submit Slurm or any other compute job.

The handoff sequence is:

1. Implement and test the OPD handler, scientific validator, deployment
   contract, and execution profile.
2. Finish with a clean, user-created Git commit. GPU preflight execution
   rejects a dirty tracked checkout.
3. Create or update an OPD-owned registration proposal with `enabled = false`.
4. Have a central ServerScheduler operator review/install the proposal.
5. Obtain separate user approval for central registration enablement.
6. Prepare one fresh request under `/scr/del6500/OPD/scheduler/outbox`.
7. Have the central operator validate and submit that exact outbox path.
8. Query central status and logs through terminal state; a queued or
   zero-exit process is not scientific success.
9. Accept success only after OPD's semantic validator accepts the published
   output and scientific completion marker.

Writing an outbox file is not submission. Enabling a registration is not
submission. A successful pilot is not authorization for G0 or a larger
experiment. Registration edits, enablement, request preparation, submission,
service changes, G0, seed-42 training, the three-seed factorial, replication,
and cleanup are separate approval gates.

Never resubmit an already accepted `job_id`. Scheduler-owned automatic retries
keep the original ID and increment `SERVER_SCHEDULER_ATTEMPT`; a new manual
request must receive a fresh opaque OPD job ID. Do not create duplicate jobs to
probe availability.

## Protocol-v2 entrypoint rules

The sole central entrypoint is:

```text
/home/del6500/projects/OPD/scripts/server_scheduler/opd-entrypoint
```

It must remain a foreground process and may dispatch only a task in the
code-owned `HANDLER_REGISTRY`. Every handler must:

- accept only its explicit parameter allowlist;
- reject commands, arbitrary paths, environment overrides, secrets, and
  unregistered resource selectors in `parameters`;
- validate the running protocol-v2 manifest and its concrete allocation;
- consume `SERVER_SCHEDULER_CPU_CORES` and the fixed per-rank thread policy;
- preserve scheduler-provided `CUDA_VISIBLE_DEVICES` exactly;
- treat visible CUDA devices as logical indices `0..N-1`, never host indices;
- stay in the foreground with no Screen/tmux, detached process, local queue,
  lease, lock, retry, or status subsystem;
- confine all results to approved OPD data/scratch roots;
- support at-least-once delivery through isolated attempts or safe resume;
- return zero only after atomic scientific completion has been validated.

Do not select physical GPUs, rewrite `CUDA_VISIBLE_DEVICES`, poll
`nvidia-smi` for placement, infer resources from idle hardware, or implement
project-local CPU/GPU admission control.

Use the project request builders. Do not hand-edit generated outbox JSON. The
normal request contains only protocol version, fresh job ID, project, task,
priority, the registered execution profile when intentionally fixed, and the
handler's allowlisted scientific identity parameters. Hardware allocation is
not chosen through parameters.

## Current migrated scheduler surface

`src/posttrain_circuits/scheduler_adapter/registry.py` currently exposes two
production handlers. All other G0, training, evaluation, and circuit tasks
remain fail-closed until they receive their own reviewed handler, profile,
runtime, validator, tests, disabled proposal, and pilot.

| Task | Profile | Fixed allocation | Result |
| --- | --- | --- | --- |
| `repository_preflight` | `repository-preflight-cpu` | 1 CPU core, 128 MiB, no GPU, 30 s estimate | `preflight_report.json` |
| `qwen3_v2_gpu_preflight` | `qwen3-v2-gpu-preflight-4gpu` | 16 CPU cores, 196608 MiB, 4 exclusive RTX PRO 6000 Blackwell GPUs, 81920 MiB and 95% utilization per GPU, 1800 s estimate | `gpu_preflight.json` |

Both request builders allow exactly `workflow_id`, `plan_sha256`, and
`unit_id`. Their checked-in proposals are documentation/handoff artifacts and
remain disabled even if an independently managed central registration has a
different live state.

Relevant paths:

- CPU handler:
  `scripts/server_scheduler/repository-preflight-handler.py`
- GPU handler:
  `scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py`
- GPU runtime preparation:
  `scripts/server_scheduler/prepare-qwen3-v2-runtime.py`
- deployment contracts: `deployments/repository_preflight/` and
  `deployments/qwen3_v2_gpu_preflight/`
- request builders and runtime boundary:
  `src/posttrain_circuits/scheduler_adapter/`
- operational design notes: `docs/refactor/`

Do not add a repository snapshot hash to scientific inputs. Maintain only the
hashes required by actual artifact, configuration, completion, and deployment
contracts. When handler/runtime bytes change, update the relevant dependency
lock, package manifest, implementation digest, manifest digest, and deployment
identity; do not introduce blanket hashing of unrelated repository content.

## Four-GPU Qwen3-v2 preflight

The fixed runtime is:

```text
/scr/del6500/OPD/envs/qwen3-v2-gpu-preflight-v1/bin/python
```

Runtime preparation occurs on the login node before a charged allocation. It
pins Python 3.12.13, PyTorch 2.8.0+cu128, CUDA 12.8, NCCL 2.27.3,
Transformers 4.56.2, and the remaining direct dependencies. Qwen3-1.7B and
Qwen3-8B must already exist at their pinned revisions below the OPD Hugging
Face cache; production loading is offline only.

This server profile fixes `NCCL_P2P_DISABLE=1` for the reproduced dual-NUMA
RTX PRO 6000 Blackwell first-all-reduce hang. It uses Gloo as the control plane
and an explicit NCCL group for CUDA tensors and FSDP. The first NCCL operation
is a scalar all-reduce with a 120-second group/work timeout. Keep NCCL timeout
diagnostics enabled, preserve the four scheduler-assigned UUIDs, and record
rank, logical device, PCI bus ID, NCCL version, and probe timing in the report.

Student load status, rank-zero teacher load, teacher forward, and final
publication status are synchronized over Gloo so a rank-specific failure does
not masquerade as an NCCL success. The preflight passes only when all four
ranks also complete real offline model forward/backward, finite soft-teacher
loss and gradients, a nonzero update, FSDP save/resume, unique prompt shards,
and the 192-GiB cgroup/headroom checks.

Do not proceed to G0 unless the current committed handler produces a published
`gpu_preflight.json` with `passed: true` and a valid scientific completion.

## Scientific repository structure

The scheduler boundary must not collapse or intermingle scientific domains.
The current package structure is:

- `datasets/`: dataset construction and contracts, separated into
  `proofgraph`, `teacher_demos`, `trajectories`, `anchors`, and
  `circuit_probes`;
- `methods/`: named SFT, OPD, RL, and control method specifications;
- `learning/supervision/`: SFT and teacher-supervision implementations,
  including hard teacher, soft teacher, verified replay, and losses;
- `learning/rl/`: RL contracts and reward definitions;
- `learning/training/`: shared training engines, optimizers, schedules,
  factories, GRPO, canonical SFT, and local scientific fork execution;
- `learning/teacher/`: teacher scoring, caching, demo generation, readiness,
  and seeding;
- `causal_circuits/`: circuit discovery backends, interventions, model
  adapters, metrics, dynamics, faithfulness, and cross-mask validation;
- `experiments/protocols/`: experiment-level scientific protocol definitions;
- `analysis/`: post-run tables, curves, factorial analysis, shared-state
  analysis, and stage-7 summaries;
- `artifacts/`: hashes, config bindings, checkpoints, provenance-compatible
  runs, datasets, completion, and artifact compatibility;
- `scheduler_adapter/`: the only ServerScheduler boundary; it must not absorb
  datasets, methods, training science, or circuit algorithms;
- `cli/`: thin scientific commands that call the domain modules rather than
  reimplementing them.

Keep SFT, OPD, RL, dataset generation, and causal-circuit discovery as distinct
scientific concepts and independently testable task boundaries. Shared
training primitives may be reused, but method identity, teacher/state source,
reward semantics, artifacts, and completion criteria must remain explicit.

## Legacy scheduler policy

Quest/Slurm, Screen/tmux fan-out, local GPU selection, and local resource locks
are not supported execution paths for the refactored project. Do not run,
extend, restore, or preserve compatibility with them. Any remaining files or
references under legacy production scripts are deletion candidates, not
operational guidance. Remove them only in an explicitly approved cleanup that
first inventories callers and preserves scientific code, outputs, checkpoints,
and validated completion markers.

## Required verification and handoff

For scheduler-bound changes, run tests proportionate to the affected surface:

- ordinary unit and scientific-validator tests;
- handler/adapter fixture tests;
- strict manifest and parameter rejection tests;
- environment-injection and CUDA-visibility preservation tests;
- retry, attempt isolation, resume, atomic output, and completion-marker tests;
- deployment lock/manifest/identity consistency checks;
- `git diff --check` and compilation/import checks.

Do not claim a GPU path works from mocks alone. Clearly separate static test
results from evidence produced by an explicitly approved central pilot. The
handoff must state files changed, exact tests and results, current task/profile,
outbox path if one was prepared, whether any job actually ran, remaining
blockers, and the Git commands the user should run.
