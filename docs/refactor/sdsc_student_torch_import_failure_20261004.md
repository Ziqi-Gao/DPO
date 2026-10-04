# Early Torch import timeout and bounded diagnostic

Recovery preflight **54643463** failed before model staging or training. Its
early child read driver version580.178.04, recorded `import_torch` beginning,
and exceeded the accepted120-second startup bound. Accounting records125s,
job/batch FAILED1:0, externCOMPLETED0:0 onexp-19-02. The existing master was
used to reconcile fresh terminal accounting and an empty queue at06:05:21Z.

The fetched paired scientific/execution publications are verified under
`.sdsc/fetched/54643463/fetch-tcvymoay/`. Independent failure review is
`.sdsc/diagnostics/student-order-v4/preflight54643463-independent-import-failure-review.json`,
SHA `2a295348373cf1b6bc35b39f85de0e1c2a702354a568c1c7693b1b7203c48a7c`.
All three own-job memory observations pass, peak599,973,888 bytes; noOOM/failcnt.
No explicit CUDA check, model, optimizer update or model-quality result exists.
This is distinct from54623814's later compound CUDA guard failure.

The successful CUDA diagnostic54626913 explicitly set OMP/MKL/OPENBLAS threads4.
The failed recovery inherited these variables and did not record their actual
values. This policy difference does not establish a thread-count root cause.
Heavy major-fault activity also permits shared-library/cache/I/O explanations.

The additive `tools/sdsc_torch_import_probe*.py` path observes one fresh job:
2H100,24CPU,16GiB,10minutes, accountnwu181, shared partition/QoS. Keeping24CPU
preserves the default-thread/affinity context; the smaller memory envelope is
supported by both observed import peaks below1GiB. The GPUs preserve allocation
and visibility context; the worker performs no explicit CUDA API call or tensor
allocation. It loads no model or data and does no training.

Three fresh isolated Python children run sequentially on the same assigned node:
A1 retains inherited thread variables, B sets only OMP/MKL/OPENBLAS to12,
A2 restores the same inherited values. Each gets at most180seconds within a
540-second total worker budget, leaving publication time within the600-second
allocation. This longer diagnostic observation does not change the accepted
production timeout. A1/A2 expose order/cache confounding; B alone being faster
cannot establish causation.

Each child records its exact selected environment, Python/runtime versions,
monotonic phase times and module origin. `-X importtime`, repeating30-second
faulthandler stacks, and bounded five-second own-child `/proc` samples retain
module, I/O, wait-state, thread and affinity evidence. Atomic partial reports
survive a stuck import. Only each newly created process group can be terminated.
No physical node/GPU is selected and no host/runtime package is modified.

Permanent claims bind this one diagnostic to the exact failed inner/outer plans
and both publications. A matching dry-run and fresh terminal reconciliation are
required; missing acknowledgements require reconciliation and never retry.
Source/control hashes, actual allocation, node-local/persistent mounts, pinned
runtime and own-job memory are checked. All small artifacts are read-back hashed,
with1MiB/file and8MiB total fetch bounds. All scientific acceptance flags stayfalse.
Diagnostic completion requires all three bounded observations, not successful
imports; import readiness is separate and grants no training/science gate.

All accepted V4 scientific, original execution-recovery and prior diagnostic
files, plans, claims and reports remain unchanged. There is no automatic
resubmission or advancement from this diagnostic to preparation.

Verification:142 focused CPU tests pass in13.18s (78 controller,36 worker,
28 node), including actual fresh processes, timeout partial reports, descendant
reaping,1MiB log truncation, original recovery-producer/failure-consumer interfaces,
aggregate validation and all12 node publication artifacts. Ruff/format and
Python3.12 AST checks pass;141 named historical science/dependency files remain
byte-identical. These fixtures use fake Torch and establish no GPU/runtime result.

Independent controller/worker review:
`torch-import-controller-worker-independent-review-20261004.json`, SHA
`3be9022b74e8d58380b86958c257c9fda26abcf99fdf6af17964dc75091927da`.
Independent node review and final TMPDIR-only addendum are under
`import-probe-review/`, final addendum SHA
`c35394648437f1f5ec311a7f180b47a7e40a6ac6c0a7bb64260c72ec417b2b5d`.
All review files are in `.sdsc/diagnostics/student-order-v4/`. Their exact source
hashes bind reviewed bytes; implementation readiness grants no GPU/science result.
