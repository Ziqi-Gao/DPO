# OPD ServerScheduler adapter status

This slice defines a project-owned protocol-v2 boundary.  It does not register
OPD, submit a request, poll a queue, run a pilot, change a service, cut over from
Slurm, or authorize deletion of any legacy execution file.

## Implemented boundary

- `workflows` owns immutable scientific DAG plans.  Six factorial method cells,
  canonical SFT, canonical GRPO, the two GRPO controls, dataset preparation,
  local-fork analysis, and circuit tasks are distinct registry entries with
  exact content-identity names.  Plans contain no command, environment, path,
  resource, profile, job, attempt, lease, status, or retry selector.
- `workflows.catalog` exactly covers those task contracts with immutable
  candidate templates containing only a fixed project module, allowed named
  inputs, declared outputs, and required scientific gates.  The catalog is
  design-time evidence only: it is not a handler registry and cannot supply an
  executable, argv, environment, resource request, working directory, or path
  override.  Catalog membership does not declare a task migrated.
  The named modules are scientific implementation references, not adapter-ready
  handlers: current CLIs do not consume the held-fd ABI and many write directory
  trees or dynamic checkpoints.  In particular, probe-cohort score-input closure,
  formal local-fork gates, and circuit acceptance gates remain migration blockers;
  the catalog's fixed files and gates describe what a future wrapper must enforce.
- The entrypoint accepts only `--job-manifest ABSOLUTE_RUNNING_JSON`.  Boundary
  reads walk from `/` one directory descriptor at a time with no-follow
  `openat` semantics, then require a regular final inode before reading bounded,
  strict UTF-8 JSON.  Symlinks in any ancestor or final component, directories,
  FIFOs, devices, duplicate keys, non-finite numbers, and oversized files fail
  closed.
- Every exported allocation/environment field is cross-checked.  CPU jobs require
  empty GPU visibility.  GPU jobs accept only the scheduler-provided UUID order;
  OPD never writes `CUDA_VISIBLE_DEVICES` and never selects a physical index.
- Workflow, plan, unit, and task identities are resolved from the immutable
  content-addressed plan path.  Output and `ScientificCompletion` paths are
  derived only from those identities and confined to approved OPD roots without
  symlink traversal.
- Every plan input names a `file` or top-level `tree` digest, never a path.  The
  project CAS resolves those identities only below fixed `workflows/inputs`
  locations through held no-follow descriptors.  Files have one SHA-256; trees
  commit to one canonical top-level inventory plus one SHA-256 per listed file,
  with missing, extra, nested, symlinked, or tampered entries rejected.
- A future reviewed handler is a code-owned task-to-profile contract.  It binds
  CPU range, memory range, process count, exact GPU count, per-device memory and
  utilization ranges, exclusivity, and an allowed GPU-model set.  GPU activation
  additionally requires a trusted model-identity observation; none exists in the
  production entrypoint today, so GPU handlers remain fail-closed even before a
  future registry entry could run.
- A handler deployment binds the exact executable, self-contained implementation
  bundle, dependency lock, package manifest, runtime version, isolated Python
  flags (`-I -S`), and a digest of the complete deployment identity.  All four
  artifacts are hash-checked through held descriptors.  The executable and
  implementation are passed to one foreground child through `/proc/self/fd`, the
  working directory is also held, and the checked inodes remain open through
  child completion.  This closes path replacement between verification and
  `exec`.
- The child receives an explicit environment only: validated scheduler GPU
  visibility, validated per-rank thread variables, immutable handler variables,
  and `PYTHONNOUSERSITE=1`.  Ambient `PYTHONPATH`, `PYTHONHOME`, `LD_PRELOAD`,
  `LD_LIBRARY_PATH`, and `LD_AUDIT` cause rejection.  Scheduler lease/state
  variables never reach the child.
- The stable script starts the absolute system Python in isolated mode and
  rejects ambient loader/Python injection variables before importing the OPD
  package.  This is an admission check, not an assertion that the eventual
  project runtime below `/scr` is already valid.
- Job ID, attempt, execution profile, manifest digest, and allocation digest are
  adapter-owned child arguments.  Lease identity is validated at entry and then
  discarded.  A newly written `ScientificCompletion` must name that exact
  execution; reuse accepts a different prior execution only when its execution
  provenance is non-empty and every scientific binding remains valid.
- Completion acceptance is handler-owned: output names and bytes, all three
  configuration-hash relationships, exact validation-gate names and truth, and
  a code-owned semantic validator must agree.  A generic self-declared marker is
  insufficient.
- Each scheduler attempt receives fresh adjacent staging and read-only CAS
  handles plus one fixed writable output-directory handle.  On success the
  adapter validates the path-free attempt draft, publishes the entire output
  directory with `RENAME_NOREPLACE`, and writes `ScientificCompletion` last.
  Failed attempt staging is never reused or inspected by a later attempt.
- Training units bind `config_binding_sha256` separately from ExperimentBinding
  and from three non-aliasing canonical JSON file identities for resolved,
  scientific, and execution configuration.  `ConfigBindingResolver` strictly
  round-trips `ConfigBinding`, reproduces it from the resolved config, and checks
  every CAS identity against the manifest-bound unit.  Production still has no
  handler registration.
- Signals are blocked across spawn/handler installation, restored in the child
  before `exec`, forwarded only to the exact direct child, and represented by the
  child's real negative signal return code.  The entrypoint then terminates with
  that same signal.  `ProcessLookupError` during an exit race is harmless.
- The outbox helper can atomically publish one schema-validated request under
  `/scr/del6500/OPD/scheduler/outbox` only for a task and optional profile already
  present in the immutable code registry.  Callers cannot supply a reviewed set
  or registry.  Parameters are only `workflow_id`, `plan_sha256`, and `unit_id`;
  it never submits or polls the file.  Each preparation uses a fresh opaque job
  ID for that scheduler submission, while the immutable parameters retain the
  scientific identity and completion-marker idempotency.  Failed staging is
  isolated by both job ID and scheduler attempt; final outputs remain keyed only
  by the scientific identity.
- The adapter has no detach, fan-out, resource selection, lock, queue, retry,
  status, submission, or polling mechanism.

## Production path blocker

The descriptor policy intentionally rejects a symlink at **any** ancestor,
including `/scr` itself.  Therefore `/scr/del6500/OPD`, any Python environment
below it, and every parent component must be real directories on the target
host; a `/scr` symlink is a production activation blocker, not something the
adapter resolves through.  Likewise a registered Python executable must be a
regular executable inode, not a symlink.  This condition has not been validated
or remediated by this slice.

## Deliberately narrow pilot

The production handler registry contains only the measured CPU-only
`repository_preflight` pilot. Existing training, dataset, teacher, local-fork,
and circuit CLIs have **not** been declared migrated by this slice and continue
to fail closed. No additional task may be added to a future ServerScheduler
registration until its handler slice supplies:

1. an immutable `HandlerSpec`, `DeploymentContract`, and one or more immutable
   `ExecutionProfileContract` values satisfying the implemented checks;
2. fixed executable, self-contained implementation bundle, dependency lock, and
   package manifest bytes with reviewed SHA-256/deployment identity;
3. exact process/allocation topology and, for GPU, a trusted model-identity
   verifier rather than host-device selection by OPD;
4. deterministic output writers, exact configuration bindings and gate names,
   a code-owned semantic validator, and current-attempt execution provenance;
5. CPU and applicable GPU characterization, descriptor-race and environment
   injection tests, signal tests, resume tests, and at-least-once reuse tests.

Registration edits, outbox submission, a pilot, service changes, cutover, and
legacy Slurm cleanup remain separate user approval gates.
