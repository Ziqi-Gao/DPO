# V4 preflight 54623814: CUDA startup failure

The V4 order-coverage preparation has not tested student learning yet. The
new preflight stopped on both ranks before Accelerator construction, data
construction, seeding, model loading or any optimizer step. Preserve its failed
scientific state and do not treat it as evidence against the new augmentation.

At 2026-10-03T17:03:41Z, accounting recorded job and batch FAILED1:0,
extern COMPLETED0:0 and435 seconds. The observer detected failure at17:03:24Z.
The published report contains `ValueError: two assigned H100s required` on both
ranks. All required small evidence was fetched:143,229 bytes under
`.sdsc/fetched/54623814/fetch-y_1imwlu/`. No checkpoints were produced.

## What the evidence establishes

The compound guard in `tools/sdsc_student_order_worker.py` checks
`torch.cuda.is_available()`, `torch.cuda.device_count() == 2` and H100 in both
reported device names. Its AST is identical to the previously successful V3
worker. Short-circuit evaluation and the shared error text lose the distinction
between unavailable CUDA, incorrect device count and incorrect device type.
The current logs do not contain the actual values or a CUDA initialization error.
Six independent CPU witnesses reproduce this ambiguity; they are not a replay
of the failed node's CUDA runtime.

The node retained Slurm CUDA_VISIBLE_DEVICES=`0,1`, W2,24CPUs and384GiB.
The accepted Accelerate YAML has no gpu_ids override. Installed official
Accelerate1.10.1 resolves this to `all` and preserves supplied ordinal and UUID
visibility strings in CPU launch-environment fixtures. No observed source delta
supports an accidental GPU-ID override. This does not establish actual device
visibility inside either failed process.

The fixed Python3.12.13/runtime package metadata, source/provenance restore,
persistent Lustre mounts, node-local ext4 workspace and staged input hashes
passed. Before the GPU guard, the node staged8,618,283,254 bytes of checkpoint,
configuration, dataset and model/tokenizer files. All16 memory samples pass,
peak16.064/384GiB, with no recorded own-job OOM/failure counters.
These checks cannot substitute for actual CUDA availability.

Read-only Slurm evidence places this attempt on exp-19-05 with requested and
allocated gpu:h100:2; that node advertises four H100s. V3 preflight54615006 ran
on exp-19-03 and fit54615110 on exp-19-13. Allocation labels do not identify the
actual failing CUDA predicate. There is no evidence yet proving driver failure,
wrong hardware, missing devices or a code-level CUDA initialization defect.

## Required next diagnostic

Use a new, independently reviewed infrastructure-only probe,2H100/4CPU/16GiB,
at most5min, the same pinned runtime and existing SSH master. Preserve Slurm
visibility, collect availability/count/device names/properties and initialization
errors individually, then require the original two-H100 gate before tiny logical
CUDA tensor operations. No models, training rows, proof generations, FSDP or
large staging are required. Persist bounded raw evidence even on failure.

This diagnostic has a separate identity and no student/preparation/G0 acceptance.
It must not delete a previous claim, retry the failed preparation silently,
select physical GPUs/nodes, modify host services or relax device/scientific gates.
An explicit engineering recovery should follow the observed cause and preserve
all V4 scientific data, seeds and thresholds. The partial new qualification
scaffold remains paused and has no candidate protocol JSON.

## Diagnostic implementation review

The new controller, node and worker are independently reviewed at their exact
source bytes.81 unique focused CPU tests pass (59 transport/node and22 worker);
Ruff, formatting and Python3.12 AST checks pass. A fresh CPU subprocess exercises
the actual node-to-worker interface. Successful GPU paths in CPU fixtures are
mocked; they establish no real CUDA readiness. All130 accepted science files,
19 original controls and five historical protocols remain unchanged.

The new task binds the actual failed plan/publication/report and has one permanent
claim across releases. Unknown submission acknowledgements require reconciliation.
Its16GiB memory gate recomputes raw cgroup limit/peak and requires1GiB headroom;
it does not weaken the training memory contract. Each CUDA call records flushed
progress and a separate raw result/error. Availability, count, initialization,
properties and tiny tensor operations independently determine readiness.
Timeouts and malformed reports cannot promote success and preserve bounded logs.

Non-author controller/worker review:
`5fc36e40b7580d1dbadfa0c80f1189f0fc8764e19ed5e0ea269aa048904e4b2c`.
Non-author node review:
`322384cb34e3f1a22f3f608c37d02dce6651c92d6dbc9656be04a9952e6cbaf7`.
A diagnostic PASS on another node cannot retrospectively identify the missing
predicate on exp-19-05 or automatically admit an unchanged training retry.

## Actual bounded diagnostic

Job54626913 completed on exp-19-01 in59s with job/batch/extern exit0.
All13 raw checks passed: unchanged CVD0,1, two H10080GBHBM3 devices, CUDA
available, initialization successful and correct tiny FP32 sums on logical0/1.
Six memory samples passed; peak0.8891/16GiB. Fetch61,626 bytes is retained in
`.sdsc/fetched/54626913/fetch-cmlk843u/`. Publication SHA is
`f90a64d4085b77326cd346f374491cd6bb51820f21a459544563574f497eae84`;
raw report SHA is
`1bc3c15fe285e9fd1ed28158309124c504774463f26fc28be6e7ec8e81867a60`.

At17:54:45Z squeue still retained the COMPLETED row, so the strict terminal
validator had not yet accepted empty-queue completion. The SSH master then
vanished on quser34. No login retry or new submission followed; manual same-host
authentication was requested. Fresh terminal reconciliation remains required.

This result supports one proposed instrumented execution recovery with an early
CUDA check in its actual allocation. It does not establish why exp-19-05 failed.
See `sdsc_student_order_execution_recovery_20261003.md`; all V4 science and
historical claims remain unchanged.

## Bound evidence

- V4 implementation:8036f8af6cbb67edef6df1c8d8ff0cee65a1d84b;
  review-only acceptance:d4db85327f0165f2aec1ea53caefd1693663f9b2.
- Plan SHA:579e9e029568fca662a830c3e50496dac33b40ee8d71d0c91f7027ea1ff16630.
- Publication SHA:cf75c0bc2c533b148113103353a9c3e7982972c1235c10160bf800ce7272c396.
- Failure report SHA:de0f3152885fb97e15f46c4fb840bfa03dd805da4f90b617edf257ae01476ae6.
- Node report SHA:298f3001c28647eb730595ca331ea5af5f9e0fdef9d8609ef3ba9a6853124fef.
- Independent diagnosis SHA:1412b60b6d62dbf9400d139ca31c50a1371531bd370a138eaec8b284d63c0878.
- Accelerate visibility CPU fixture SHA:210a283b35e7e2249bb8f597539866420b63c31a324b46b76dfe9e7cfa5ac91d.

Local detailed evidence is under `.sdsc/diagnostics/student-order-v4/`.
