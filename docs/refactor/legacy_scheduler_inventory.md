# Legacy scheduler cutover inventory

This inventory is a read-only baseline for the eventual Slurm cutover. It is
not deletion approval, a ServerScheduler registration, a job request, or a
claim that the new handlers are ready. The inventory was generated from the
repository on 2026-09-01. No scheduler command, service command, submission,
poll, cancellation, or cleanup operation was run.

## Recoverable legacy executable surface

The legacy surface contains 20 files under `scripts/slurm`:

- orchestration helpers: `common.sh`;
- dataset and supporting artifacts: `task_generation.slurm`,
  `rollout_bank.slurm`, and `teacher_scoring.slurm`;
- training and anchors: `factorial_training.slurm`, `canonical_grpo.slurm`,
  `gemma_mini_replication.slurm`, and `pilot_qwen_core.slurm`;
- local-fork analysis: `local_forks.slurm` and `pilot_local_fork.slurm`;
- circuit analysis: `circuit_discovery.slurm`, `exact_patching.slurm`,
  `pilot_initial_circuits.slurm`, `pilot_final_circuits.slurm`, and
  `pilot_dynamics.slurm`;
- preflight, resume, G0, and aggregation: `gpu_preflight.slurm`,
  `qwen3_gpu_preflight.slurm`, `pilot_resume.slurm`, `g0_qwen.slurm`, and
  `aggregate_results.slurm`.

There are eight legacy launch/supervision files under `scripts/production`:

- `submit.sh`, `submit_pilot.sh`, and `submit_qwen3_pilot.sh`;
- `run_g0.sh`, `run_qwen3_g0.sh`, `run_gpu_preflight.sh`, and
  `run_qwen3_gpu_preflight.sh`;
- `slurm_supervision.sh`.

Three Python modules remain part of the same quarantined launch/pilot surface:
`src/posttrain_circuits/cli/record_qwen3_launch.py`,
`src/posttrain_circuits/cli/finalize_pilot_training.py` and
`src/posttrain_circuits/cli/finalize_pilot.py`. They are not console entrypoints
or candidate ServerScheduler handlers; their terminal/job-ID protocol must be
retired atomically with the legacy pilot caller after the cutover gates below.

The direct production-to-Slurm caller edges are:

| Production caller | Slurm target or role |
| --- | --- |
| `run_g0.sh` | `g0_qwen.slurm` plus shared terminal supervision |
| `run_qwen3_g0.sh` | delegates to the G0 launcher with Qwen3 settings |
| `run_gpu_preflight.sh` | `gpu_preflight.slurm` |
| `run_qwen3_gpu_preflight.sh` | `qwen3_gpu_preflight.slurm` plus shared terminal supervision |
| `submit.sh` | dynamically selects `scripts/slurm/<stage>.slurm` |
| `submit_pilot.sh` | pilot training, initial/final circuits, local fork, resume, and dynamics scripts |
| `submit_qwen3_pilot.sh` | delegates to the pilot launcher with Qwen3 settings |
| `slurm_supervision.sh` | local `squeue`/`sacct` polling and terminal-state interpretation |

No project-owned service, timer, socket, target, or path unit was found in the
repository. This does not establish that no external service or process calls
these files; external registrations, services, running processes, jobs, locks,
leases, and operator scripts must be inventoried by the appropriate operator
before cutover.

## Scientific callers that must move before deletion

The Slurm files currently invoke scientific CLIs spanning all major domains:

- ProofGraph and data construction: `generate_task`, `build_rollout_bank`,
  `build_teacher_demos`, `score_teacher`, `build_probe_cohorts`, and
  anti-shortcut/readiness checks;
- SFT/OPD/RL: `train`, `run_grpo`, checkpoint export/resolve, distributed
  resume comparison, and pilot/G0 finalization;
- causal circuits: probe scoring, discovery, exact validation, pilot scope,
  and circuit dynamics;
- LocalFork: input construction, bundle construction, and fork execution;
- reporting: aggregation and G0/pilot final reports.

Each CLI must first have a reviewed scheduler-neutral scientific handler with
the same declared input identities, output contract, resume semantics, and
completion validation. A filename replacement alone is not a cutover.

Historical or quarantine references outside the legacy executable directories
remain in the two pilot finalizers above, `docs/core_execution_plan.md`,
`docs/scientific_repair_audit.md`,
`docs/scientific_repair_v2_gap_analysis.md`,
`tests/unit/test_qwen3_protocol.py`, and
`tests/unit/test_qwen3_v2_repairs.py`. These references and tests must be
rewritten or retired atomically with the corresponding caller, not left as
dead instructions.

## Protected scientific evidence

The following are not scheduler cleanup targets:

- preregistration files and frozen scientific protocol records;
- generated datasets, rollout banks, teacher demonstrations, checkpoints,
  circuit artifacts, reports, and validated completion markers;
- historical Slurm terminal evidence already bound into a RunManifest or
  frozen pilot/G0 report;
- tests that characterize historical bytes needed to read already-produced
  artifacts.

The new execution path must use scheduler-neutral ScientificCompletion for new
runs. Historical Slurm evidence must remain identifiable as historical; it
must not be silently reinterpreted as ServerScheduler evidence.

## Independent cutover gates

Physical removal of any file listed above requires all of the following:

1. every scientific handler and task-specific parameter allowlist is migrated
   and independently reviewed;
2. protocol-v2 fixture, path-confinement, idempotency/resume, and completion
   tests pass without loading the live queue;
3. a disabled ServerScheduler registration is reviewed outside this project;
4. a separately approved bounded pilot succeeds and validates its scientific
   completion marker;
5. all legacy callers are switched, legacy work is drained without killing or
   preempting it, and a rollback reference is recorded;
6. an operator checks external services, registrations, live processes, jobs,
   locks, leases, and audit dependencies;
7. the user is shown the exact source files, scratch paths, and external
   service targets and gives explicit cleanup/service approval.

Cleanup must use exact patch-based file deletion. It must not use recursive
deletion, broad globs, broad process killing, service mutation, or deletion of
scientific outputs/completion markers.
