# V4 execution recovery after the early import timeout

Preflight54643463 exceeded its120-second early Torch-import bound before explicit
CUDA checks, model staging or training. Diagnostic54643629 subsequently completed
on the same recorded node: inherited/12-thread/inherited imports took
53.419/1.559/1.571seconds. The first30-second stack was reading a Python module
from the shared Lustre runtime. This supports variable loading/cache overhead;
it does not establish the exact old timeout stack or thread count as its cause.
See `sdsc_student_torch_import_failure_20261004.md` for verified raw evidence.

The new execution-only contract is
`prereg/amendments/qwen3_student_order_execution_recovery_v2.json`. It is proposed
until a complete implementation commit receives independent review followed by
a separate review-only acceptance commit. Accepted v1 files and both failed jobs
remain immutable. A diagnostic completion does not satisfy a training preflight.

The v2 early child gets at most300seconds within the original stage worker
deadline, including its bounded process-group cleanup. Early OMP/MKL/OPENBLAS
variables are explicitly12 for consistency with the existing rank thread policy.
The new small probe adapter enables import timing, timestamped progress and
repeating30-second stacks, then invokes the pinned v1 probe with its original
arguments. The original raw CUDA checks and startup report remain authoritative.
A1/A2 show that this thread setting is not necessary for fast warm imports;
the setting is an execution-policy consistency choice.

Only startup supervision changes. Training still uses the frozen v1 rank wrapper
and the original V4 worker,13 input keys, exact numerical settings, datasets,
RNG, checkpoint schedule, full32-update fit and18,432-response selection audit.
Resources stay2H100/24CPU/384GiB, preflight1h and fit8h; original300/600-second
publication reserves and all model-quality gates stay unchanged.

Timestamped `trace-meta.json` binds the owned process, launch arguments,
effective environment, actual elapsed time, exit/timeout/reaping status and
retained `startup.log` bytes. Logs retain at most1MiB with explicit truncation;
all execution evidence fits2MiB including its receipt, still inside the original
224MiB small-result fetch. A successful startup requires both valid trace evidence
and the unchanged raw CUDA report. Negative traces and logs remain collectible.

The new task/schema and permanent recovery claim bind accepted v2 execution
semantics to the original scientific identity. Admission freshly reconciles the
failed v1 job and completed import diagnostic with exact paired plans, accounting,
artifact hashes and empty queues. It never re-arms prior claims. A fit must also
claim the original unique V4 scientific fit identity and require its own matching
v2 preflight, paired publication and original independent eight-response audit.
Unknown submission acknowledgements are reconciled without retry.

The original scientific source binding and genuine Git provenance remain
separate from this execution contract. Immutable v1 and diagnostic helpers are
hash-pinned dependencies. Current v1-only qualification bindings must explicitly
migrate to accepted v2 recovery evidence before any later candidate qualification;
no candidate can be bound before successful full fit and independent raw replay.

The complete focused CPU suite passes175 tests:87 contract,35 node/probe and53
transport. Ruff, formatting and Python3.12 AST checks pass. Independent node review
also exercised two real process groups with surviving descendants and confirmed
bounded cleanup. Actual CPU producer-to-publication-to-parser fixtures cover both
negative CUDA startup and trace/log contradictions. These tests do not establish
GPU startup, training success or model quality. Exact implementation review and
separate acceptance are required before deployment.
