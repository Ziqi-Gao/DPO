# Instrumented execution recovery for V4 student preparation

The scientific V4 order-coverage experiment has not yet reached training. Its
preflight54623814 stopped before any model load or update at a compound CUDA
guard, after staging8,618,283,254 bytes. The guard discarded which of availability,
visible count or device names failed. Its allocated node was exp-19-05.

The separate diagnostic54626913 ran on exp-19-01 with the same pinned
Python3.12.13/Torch2.8.0+cu128/CUDA12.8 runtime. Job, batch and extern completed
with exit0 in59 seconds. Raw observations show CUDA available, two visible
NVIDIA H10080GB HBM3 devices, successful initialization and a correct tiny FP32
sum on logical devices0 and1. Slurm visibility remained0,1; no physical device
or node was selected. Six memory observations passed, peak0.8891/16GiB.

These observations establish the diagnostic allocation's CUDA readiness. They
do not identify the missing predicate on exp-19-05, prove a repaired host driver,
validate Accelerate/FSDP training, or accept any student model. The diagnosed
code defect is loss of startup evidence and discovery of unusable CUDA after
large staging. Recovery addresses those execution boundaries explicitly.

## Scope of the recovery

Preserve all130 accepted V4 scientific files, all five historical protocol
artifacts, original data/seeds and every scientific threshold. Reuse the original
worker, data factory, checkpoint selector, independent raw auditor and numerical
training implementation. Do not create new scientific V5 data or reselect any
historical failed model.

A small separately reviewed execution binding permits one fresh preflight for
the named zero-update failure, followed only by its matching validated fit. It
has an implementation commit and a later independent review-only acceptance.
The original failed intent and permanent claim remain immutable. A missing
submission acknowledgement requires reconciliation; no automatic retry or node
exclusion is introduced. Fit also acquires the original V4 scientific fit claim
so another execution revision cannot duplicate it.

The new orchestration runs a bounded CUDA child before staging the large inputs.
Its120-second limit is charged to the existing allocation budget, and the child
must exit before training. Each of the two launched ranks records the original
GPU predicates and initialization error before calling the unchanged worker.
All command-line inputs, assigned CUDA visibility, rank identities and scientific
RNG behavior are preserved. The old worker still applies its own independent
CUDA guard and its complete training, export, restoration and result checks.

Startup evidence is bounded and retained on failure separately from the original
scientific artifacts. Completion requires both execution evidence and the
original scientific validator, persistent artifact verification and successful
accounting. A standalone diagnostic PASS cannot substitute for any of these.

## Unchanged training and acceptance

Both stages retain2H100/24CPU/384GiB, account nwu181, partition nairr-gpu-shared,
QoS nairr-gpu-shared-normal, at most four concurrently allocatable GPUs.
Preflight remains at most1h and fit at most8h, including startup checks and
original publication reserves. Node-local space, persistent mounts and original
training cgroup/headroom/OOM gates remain mandatory in the actual allocation.

Preflight runs four original global64 optimizer windows and completes the
original full-state restoration, FP32 export/reload, native BF16 parity and
context-envelope checks. Only its eight training responses are independently
audited; it makes no model-quality claim. Matching fit restarts fresh native1.7B
and completes32 windows/1,614,932 input tokens,12 checkpoints and18,432 raw
development responses under the unchanged FP32/W2 FULL_SHARD/5e-5 settings.
Selection keeps the original capability band, per-structure floors and all
transformation-gap limits. Any scientific failure stops progression.

## Evidence

- Failed preflight plan SHA:
  579e9e029568fca662a830c3e50496dac33b40ee8d71d0c91f7027ea1ff16630.
- Failed publication SHA:
  cf75c0bc2c533b148113103353a9c3e7982972c1235c10160bf800ce7272c396.
- Diagnostic plan SHA:
  d8c40f45e61d50ff25182b7870d4224f07ed342eaec9c53eb430074512bfefda.
- Diagnostic publication SHA:
  f90a64d4085b77326cd346f374491cd6bb51820f21a459544563574f497eae84.
- Diagnostic raw report SHA:
  1bc3c15fe285e9fd1ed28158309124c504774463f26fc28be6e7ec8e81867a60.
- Diagnostic fetch: `.sdsc/fetched/54626913/fetch-cmlk843u/`,61,626 bytes.

## Implementation and verification

The new execution contract, controller, node and startup wrapper compose the
unchanged V4 implementation. The existing local science-plan builder creates
an inner plan; the execution controller wraps it with the separate review and
recovery claim. Both plan hashes remain distinct. Original scientific receipts,
13-key worker inputs, validators and independent raw-audit inventory are reused.
Missing paired execution/scientific acknowledgements must be reconciled before
fit admission. Failed and diagnostic permanent claims are checked through their
original guarded entrypoints.

All103 focused CPU tests pass:34 contract,32 worker and37 transport/node. They
include genuine Git implementation/review-only histories, typed raw startup
rejection, actual node-to-fresh-CPU-worker/Accelerate interfaces, exact original
worker argument forwarding, early failure/timeout/shutdown log retention without
large staging, unique claims and partial acknowledgement rejection. Ruff,
formatting and Python3.12 AST checks pass. Original130 science files and all five
historical protocols retain their accepted bytes. Successful CUDA fixtures are
mocks; real CPU subprocesses establish rejection, not real W2 GPU arithmetic.

The separate read-only observer has22 independently passing fixtures, including
strict combined completion and no query after a deadline or changed control.
It only polls status every60s in a finite foreground process; it does not submit,
retry, fetch, cancel or reconnect SSH. It is not active without an actual new
plan and receipt. A conversation or SSH interruption does not preserve agent
monitoring or automatic error repair.

The contract review block and current handoff are authoritative for implementation
acceptance and execution state. Source acceptance, fresh prerequisite reconciliation
and deployment must precede a recovery submission. At implementation completion,
SSH is absent, the user has been asked to authenticate on quser34, and no recovery
job has been submitted.
