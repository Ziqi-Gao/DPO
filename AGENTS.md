# OPD Agent Guide

This file describes the current OPD repository after the scientific-domain and
ServerScheduler refactor. It is authoritative for work performed from an OPD
project session. It does not grant authority over the central scheduler.

## Mandatory ServerScheduler contract

Before changing or running OPD code, read these files in full:

- `/home/del6500/projects/ServerScheduler/docs/project-session-scheduler-v2-guide.md`
- `/home/del6500/projects/ServerScheduler/docs/project-integration.md`
- `/home/del6500/projects/ServerScheduler/docs/job-contract.md`
- `/home/del6500/projects/ServerScheduler/docs/scheduler-managed-gpu-v1.md`
- `/home/del6500/projects/ServerScheduler/docs/project-migration-instructions.md`
- `/home/del6500/projects/ServerScheduler/schemas/job-request-v2.schema.json`

The project owns scientific behavior, task validation, execution profiles,
checkpointing, outputs, and scientific completion. ServerScheduler alone owns
queue order, physical CPU/GPU placement, leases, retries, status, logs, runtime
calibration, and cross-project concurrency.

## Mandatory current-handoff synchronization

The canonical cross-session state is
`docs/refactor/current_handoff.md`. Every top-level OPD session must read it in
full before substantive analysis, planning, edits, or execution. Treat it as a
current-state summary, not as authority over source code, Git, scientific
artifacts, or central scheduler state; verify mutable claims when they matter
to the task.

This is standing authorization to edit only that handoff when a session learns
or creates a material OPD state change, unless the user explicitly requests a
read-only/no-write task. Update it in the same change before the final response
when any of these change:

- repository architecture or supported execution path;
- migrated handler, task, profile, deployment, or fixed runtime;
- scientific protocol, readiness gate, active blocker, or next required step;
- authoritative pilot/experiment outcome or failure diagnosis;
- verification evidence that materially changes confidence in the active path.

Do not update the handoff for unchanged status, exploratory reading, routine
formatting, or facts already recorded. Keep it concise and replace superseded
claims instead of appending a session diary. Git remains the complete history;
do not duplicate a changelog or add blanket hashes. Never claim an uncommitted
candidate is committed, or external scheduler state is current, without
verifying it.

Immediately before editing the handoff, reread it and inspect current Git
status so concurrent work is preserved. Merge non-conflicting facts; do not
overwrite another session's updates. Delegated subagents report handoff-worthy
facts to their parent, and the parent performs the single consolidated handoff
edit unless it explicitly assigns one writer. Before finishing, reread the
result and include the handoff in the task's Git commit. Report the resulting
commit identity and any remaining uncommitted changes to the user.

Codex loads `AGENTS.md` once when a run/session starts. New sessions receive
this policy automatically; an already-running session must be restarted or
explicitly told to reread `AGENTS.md` and the handoff before continuing.

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

The user has given standing authorization for OPD agents to run `git add`
and `git commit` for authorized work. Agents own staging and creating the
required implementation, review/acceptance, and handoff commits; do not ask
the user to perform or separately approve these routine Git operations.
Stage only the task's files and preserve unrelated user changes in a dirty
checkout. Keep independent review and distinct acceptance commits where the
scientific contract requires them. Do not create a Git worktree, amend commits,
or rewrite history without separate explicit authorization.

## Submission and approval rules

An OPD project session may implement handlers, validate contracts, create a
disabled registration proposal, and prepare a project-owned protocol-v2
outbox request. It must not run ServerScheduler `submit`, `dispatch`, or
`serve`, and must not submit Slurm or any other compute job.

The handoff sequence is:

1. Implement and test the OPD handler, scientific validator, deployment
   contract, and execution profile.
2. Finish with a clean Git commit created by the agent. GPU preflight execution
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
priority, and the handler's allowlisted scientific identity parameters. It
omits both `execution_profile` and `resources`, including for a task whose only
reviewed profile is fixed. An explicit profile, CPU hint, or host-memory hint
requires a separately documented central exception; no current OPD request
builder has one. Never hide resource steering in parameters, job IDs, request
filenames, or request-selected configuration.

ServerScheduler alone chooses concrete physical devices. For a
scheduler-managed GPU profile it also compares the centrally permitted 1, 2,
3, and 4 GPU allocations at claim time using predicted wait plus runtime; a
running attempt is never resized. The OPD entrypoint must derive and
cross-check the actual GPU count from both the running manifest and
`SERVER_SCHEDULER_GPU_COUNT`, preserve `CUDA_VISIBLE_DEVICES`, and use only
logical devices `0..N-1`.

Do not propose `gpu_count_policy = "scheduler"` unless the same task and
entrypoint have reviewed scientific semantics and fail-closed fixture evidence
for every centrally permitted count 1, 2, 3, and 4, including batch/token/RNG
semantics, CPU-thread allocation, one-GPU memory safety, the three-GPU case,
same-world checkpoint/resume, and fail-closed changed-world rejection.
Otherwise declare `gpu_count_policy = "fixed"` with one exact registered
count. Candidate E's
accepted v1 implementation makes both Qwen3-v2 GPU entrypoints
scheduler-managed over exactly 1, 2, 3, and 4 GPUs, but its accepted
project-specific first-G0 gate still requires four real pilots. The proposed
execution-class successor is not operational until its amendment and
certificate and Candidate E execution-science protocol are jointly accepted,
and its disabled central registration is separately reviewed, installed, and
enabled. Historical Candidate D
remains only a recoverable fixed-two-GPU baseline.

Treat that 1/2/3/4 proof as certification of a reusable elastic execution
class, not as four task variants and not as a four-pilot prerequisite that is
automatically repeated for every new scientific job. A class certification
must bind a canonical execution-safety fingerprint covering the handler and
fixed runtime, model and sequence shapes, batch partition and loss scaling,
global token accounting, FSDP behavior, checkpoint/resume semantics, CPU and
memory envelope, and supported world sizes. Real allocation evidence belongs
to the certified class. A later job may reuse it only when its recomputed
fingerprint is identical.

Job IDs, scheduler attempts and allocations, seeds, repetitions, output
locations, and scientific parameters that do not affect distributed execution
are evidence provenance or experiment identity, not fingerprint inputs. A
change to the distributed implementation, runtime or dependency identity,
model/sequence shape, batch or loss normalization, token accounting, FSDP,
checkpoint/resume, memory envelope, or supported world sizes invalidates the
certification. Every experiment still requires its own scientific protocol,
configuration, artifact, and completion validation. Do not use execution-class
certification to waive those scientific gates.

For the proposed Qwen3-v2 class, the fingerprint hashes only the explicitly
named safety-critical surfaces enumerated by the descriptor plus an explicit
safety subject for runtime identities, model/sequence shape, batch/loss/token
semantics, FSDP, checkpoint/resume, optimizer state, CPU/thread and memory
envelopes, and supported world sizes. It is not a whole-repository snapshot.
Do not add request builders, scientific protocols, or unrelated source merely
to make the hash surface broader. The v1 descriptor currently hashes each
named implementation surface as a whole file, so a byte change in a shared
named file blocks reuse until reviewed even when the change may prove
semantically irrelevant. Treat this as a conservative implementation boundary,
not as a reason to repeat four GPU pilots automatically: first isolate or
review the delta, and require new real-GPU evidence only when proportionate to
an actual execution-safety change. Reuse checks compare the current kernel and
safety subject; they do not replay every intervening commit or require
direct-parent Git topology.

After preparing an outbox request, report its absolute path, SHA-256, job ID,
project HEAD, tracked-worktree state, and validation results. The outbox is not
automatically consumed, and the file's existence is not evidence of central
submission or status.

## Current migrated scheduler surface

`src/posttrain_circuits/scheduler_adapter/registry.py` exposes two validated
preflight handlers and one acceptance-gated G0 handler. All other training,
evaluation, and circuit tasks remain fail-closed until they receive their own
reviewed handler, profile, runtime, validator, tests, disabled proposal, and
pilot.

| Task | Profile | Reviewed allocation | Result |
| --- | --- | --- | --- |
| `repository_preflight` | `repository-preflight-cpu` | 1 CPU core, 128 MiB, no GPU, 30 s estimate | `preflight_report.json` |
| `qwen3_v2_gpu_preflight` | `qwen3-v2-gpu-preflight-elastic` (`gpu_count_policy = "scheduler"`) | 24 CPU cores, 196608 MiB, scheduler-chosen 1/2/3/4 exclusive RTX PRO 6000 Blackwell GPUs, 81920 MiB and 95% utilization per GPU, 7200 s conservative one-GPU estimate | `gpu_preflight.json` |
| `qwen3_v2_g0` | `qwen3-v2-g0-elastic` (`gpu_count_policy = "scheduler"`) | 24 CPU cores, 196608 MiB, scheduler-chosen 1/2/3/4 exclusive RTX PRO 6000 Blackwell GPUs, 81920 MiB and 95% utilization per GPU, 86400 s conservative one-GPU estimate | `g0.json`, `g0_artifacts.tar` |

All three request builders allow exactly `workflow_id`, `plan_sha256`, and
`unit_id` and omit `execution_profile` and `resources`. Their checked-in
proposals are documentation/handoff artifacts and remain disabled even if an
independently managed central registration has a different live state.

Relevant paths:

- CPU handler:
  `scripts/server_scheduler/repository-preflight-handler.py`
- GPU handler:
  `scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py`
- GPU runtime preparation:
  `scripts/server_scheduler/prepare-qwen3-v2-runtime.py`
- G0 handler and runtime preparation:
  `scripts/server_scheduler/qwen3-v2-g0-handler.py` and
  `scripts/server_scheduler/prepare-qwen3-v2-g0-runtime.py`
- deployment contracts: `deployments/repository_preflight/` and
  `deployments/qwen3_v2_gpu_preflight/`; the G0 candidate is under
  `deployments/qwen3_v2_g0/`
- request builders and runtime boundary:
  `src/posttrain_circuits/scheduler_adapter/`
- operational design notes: `docs/refactor/`

Do not add a repository snapshot hash to scientific inputs. Maintain only the
hashes required by actual artifact, configuration, completion, and deployment
contracts. When handler/runtime bytes change, update the relevant dependency
lock, package manifest, implementation digest, manifest digest, and deployment
identity; do not introduce blanket hashing of unrelated repository content.

## Scheduler-managed Qwen3-v2 preflight

The fixed runtime is:

```text
/scr/del6500/OPD/envs/qwen3-v2-gpu-preflight-v1/bin/python
```

Runtime preparation occurs on the login node before a charged allocation. It
pins Python 3.12.13, PyTorch 2.8.0+cu128, CUDA 12.8, NCCL 2.27.3,
Transformers 4.56.2, and the remaining direct dependencies. Qwen3-1.7B and
Qwen3-8B must already exist at their pinned revisions below the OPD Hugging
Face cache; production loading is offline only.

The server profiles fix `NCCL_P2P_DISABLE=1` for the reproduced dual-NUMA
RTX PRO 6000 Blackwell first-all-reduce hang. It uses Gloo as the control plane
and an explicit NCCL group for CUDA tensors and FSDP. The first NCCL operation
is a scalar all-reduce with a 120-second group/work timeout. Keep NCCL timeout
diagnostics enabled, preserve all scheduler-assigned UUIDs in their supplied
order, and record the actual world size, rank, logical device, PCI bus ID, NCCL
version, per-rank thread count, and probe timing in the report.

Student load status, rank-zero teacher load, teacher forward, and final
publication status are synchronized over Gloo so a rank-specific failure does
not masquerade as an NCCL success. The preflight passes only when every actual
rank also completes a production-shaped exact 64-sequence optimizer window at
the reviewed 1,536-token model-input limit, finite response-masked canonical-SFT
loss and gradients, a nonzero full-parameter AdamW update, FSDP full-state
save/resume, unique prompt shards, and the 192-GiB cgroup/headroom checks. Rank
zero additionally completes the pinned offline teacher forward. Each request
receives a fresh opaque, count-neutral workflow ID; neither that ID nor its
filename may encode a GPU count.

One pilot validates only the world size actually assigned to that attempt. The
accepted Candidate E v1 amendment added an OPD-specific first-G0 gate requiring
four distinct accepted-lineage reports and completions covering world sizes 1,
2, 3, and 4. That gate is not a generic ServerScheduler requirement and must
not be copied into every later experiment. The project-facing request cannot
force a count, so Candidate E's initial matrix is a central validation-plan
responsibility. Once a successor execution class is independently certified,
later jobs with the identical safety fingerprint reference that certification
instead of repeating four pilots.

The proposed v2 certificate uses accepted historical real W=1 and W=2 reports
as explicitly reviewed bridge evidence and has static fail-closed coverage for
W=1/2/3/4. W=3 uneven-tail collectives and W=4 host topology still lack a
successful real-GPU observation. That residual risk must remain visible; v2 is
`proposed` and non-operational until independent acceptance.

## Scheduler-managed Qwen3-v2 G0 amendment

The frozen base preregistration remains unchanged. Candidate E's accepted
amendment is `prereg/amendments/qwen3_v2_g0_elastic_v1.yaml`; its accepted bytes
and four-pilot gate are immutable historical authority and must not be silently
rewritten. Any transition to reusable execution-class certification requires a
new proposed successor amendment and its own implementation/acceptance commits.
One global optimizer window always contains the
same 64 logical samples. With maximum physical microbatch size 4, the reviewed
rank-local sample totals are `64`, `32/32`, `22/21/21`, and `16/16/16/16` for
world sizes 1, 2, 3, and 4. The three-rank tail is exactly `2/1/1`; framework
accumulation and rank averaging are scaled back to the same global sequence
mean. The 2,000,000-token budget remains an exact cross-rank sum of non-padding
model-input tokens reserved before any backward in an optimizer window, and
the 120 optimizer-step ceiling is unchanged.

The active G0 state source is the deterministic accepted teacher-demo cursor
protocol and consumes no RNG. The production prompt population is exactly 256
unique IDs in manifest order, an integer multiple of the 64-slot optimizer
window, so each prompt's rank-local demo cursor advances equivalently at every
reviewed world size. Checkpoints are written only at optimizer boundaries and
store rank-local cumulative trainer state separately while requiring one
identical global token-budget state. Explicit same-world resume is required and
compared twice; changed-world resume, partial-window metadata, or inconsistent
rank cursors must fail before model, optimizer, scheduler, RNG, or source state
is loaded. Fresh scheduler attempts use isolated workspaces and never silently
resume. The student update is full-parameter training; configuration,
ExperimentBinding, checkpoint evidence, and final validation must agree on that
fact.

The reviewed FSDP request remains `FULL_SHARD` for every count. PyTorch 2.8
reduces that request to the scientifically equivalent effective `NO_SHARD`
strategy when the assigned world size is one; effective world sizes 2, 3, and
4 remain `FULL_SHARD`. Preflight, checkpoint, resume, and final artifact
evidence must report both the requested and the actual effective strategy and
must reject a missing, mixed, or count-inconsistent FSDP wrapper tree. The
single-rank full-state export must use `offload_to_cpu=false` and
`rank0_only=false`, matching the fixed Accelerate 1.10.1 workaround; multi-rank
exports use rank-zero CPU offload.

Candidate E v1 retains its historical two-commit, amendment-only acceptance.
The proposed execution-class successor also uses an implementation commit and
a later review-only acceptance commit. The latter jointly reviews the
successor amendment, certificate, and Candidate E execution-science protocol
so no artifact contains its own commit identity:

1. The agent creates a clean implementation commit containing the complete
   implementation, proposed successor amendment, immutable descriptor,
   proposed certificate, and proposed execution-science protocol.
2. After independent review, change only the three `review` blocks to the same
   accepted metadata, bind their `reviewed_implementation_commit` fields to
   step 1, optionally update this handoff, and create one joint acceptance
   commit.
3. Runtime validation requires the implementation and joint acceptance commits
   to be ancestors but does not require a direct-parent relationship. The
   descriptor, certificate, and every safety-critical blob named by the
   descriptor must match at use time; later non-safety commits and merges do
   not invalidate the execution class. Candidate E separately binds its
   accepted execution-science protocol, storage-neutral scientific config, and
   reviewed scientific implementation; later `src/`, `scripts/`, or `configs/`
   changes require a new science review but do not by themselves invalidate the
   execution class. A different experiment must supply its own scientific
   review before reusing the class certificate.

The accepted Candidate E v1 G0 request builder rejects a dirty tracked checkout,
anything other than its distinct accepted-lineage 1/2/3/4 GPU-preflight matrix,
or an unreviewed Git delta. A successor builder must instead reject a proposed
successor amendment, a non-accepted or fingerprint-mismatched execution-class
certification, and any unreviewed safety-critical delta. It must not expand a
class certificate back into experiment-owned count-specific requests or raw
preflight inputs.
The disabled registration proposal is not authorization to install, enable, or
submit. Amendment acceptance, central proposal installation, registration
enablement, preflight submission, G0 request generation, and central G0
validation/submission remain separate approval gates.

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
blockers, completed Git commit identities, and any remaining uncommitted changes.
