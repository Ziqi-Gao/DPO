# Qwen3-v2 elastic execution-class certification

## Scheduler boundary

Each scientific task registers exactly one `gpu_count_policy = "scheduler"`
GPU profile. A normal project request never names an execution profile,
resource quantity, GPU count, GPU index, UUID, or current availability. At
claim time ServerScheduler compares its allowed 1/2/3/4-GPU allocation
candidates, performs normal admission, and supplies one fixed count and ordered
UUID visibility for that attempt. OPD preserves that visibility and uses only
logical devices `0..N-1`.

The four world sizes are candidates for one task invocation. They are not four
project tasks, four scientific variants, or four normal requests. The generic
ServerScheduler protocol requires semantic correctness at every supported
world size; it does not require every new experiment to repeat four real GPU
pilots.

## Audit of the Candidate E v1 binding

Candidate E v1 intentionally bound one exact W=1/2/3/4 real-pilot matrix to
its first G0 plan. The binding appeared in:

- `prereg/amendments/qwen3_v2_g0_elastic_v1.yaml`, which remains immutable and
  accepted;
- `artifacts/protocol_amendments.py`, whose v1 parser preserves that historical
  gate;
- the former eight report/completion inputs in `scheduler_adapter/g0_request.py`
  and `workflows/contracts.py`;
- the former matrix replay in `scheduler_adapter/qwen3_v2_g0.py`, the G0
  handler, and `cli/finalize_g0.py`;
- the corresponding request, workflow, validator, handler, finalizer, and
  stage tests; and
- `AGENTS.md`, the handoff, registration notes, and the GPU-pilot guide.

That was an OPD project-level gate, not a ServerScheduler rule. The accepted v1
amendment is not reinterpreted or edited by this migration.

## Reusable class artifacts

The proposed successor uses:

- `prereg/execution_safety/qwen3_v2_elastic_training_v1.descriptor.json` for
  the canonical safety subject and fingerprint;
- `prereg/execution_safety/qwen3_v2_elastic_training_v1.certification.yaml` for
  reviewed static and real evidence; and
- a separately reviewed execution-science protocol for each experiment, such
  as `prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_v1.yaml`.

The fingerprint hashes the safety subject containing the exact named set in
`IMPLEMENTATION_FILE_PATHS`; the certification separately binds the descriptor
bytes by SHA-256. The named set is limited
to the two GPU handlers and deployment identities, the held-manifest and
foreground dispatch boundary, the fixed runtime locks, and code that directly
implements batch/loss/token, accepted-view cursor, FSDP, checkpoint/resume,
atomic checkpoint publication, model loading, and sequence-shape semantics.
Request builders, job identities, output locations, and per-experiment science
protocols are deliberately outside that set.

The descriptor also explicitly fixes the two runtime executable identities and
dependency locks; model revisions and tensor/sequence shapes; the exact global
batch partition including the W=3 `2/1/1` tail; sequence-mean loss scaling;
global token accounting; FSDP wrapper and effective strategies;
checkpoint/resume rules; optimizer precision/state; CPU-thread partition;
GPU/host-memory envelope; and supported world sizes 1, 2, 3, and 4. This is not
a repository snapshot hash. Only the two GPU deployment manifests are included;
request builders, general governance parsers, scientific protocols, output
locations, and unrelated source are not added merely because they exist.

The v1 implementation uses whole-file digests for each named surface. A byte
change anywhere in one of those shared files therefore pauses reuse pending
review, even when the edit may ultimately be classified as unrelated to the
certified behavior. This conservative boundary must not be turned into an
automatic four-pilot rerun: isolate or review the delta first, then collect new
real-GPU evidence only when the changed execution risk warrants it. Future
narrower module extraction may reduce these false invalidations without
weakening the safety closure.

Reuse checks recompute the descriptor against the current named safety surface and
explicit safety subject. They do not replay every intervening Git commit or
require the request and execution commits to be direct descendants containing
four redundant copies of the same blobs. The reviewed implementation must be
an ancestor of the acceptance record, but the acceptance record need not be
its direct child; the acceptance change itself remains review-only.

## Reuse and invalidation

Job and workflow IDs, scheduler attempts and allocations, seed, experiment
repetition, storage/output paths, and scientific parameters that do not affect
distributed execution are deliberately excluded from the fingerprint. A new
experiment may reuse an accepted certificate when the current execution-safety
fingerprint matches exactly.

The certificate is invalidated by a change to the distributed handler or
execution kernel, fixed runtime/dependency identity, model/tensor shape,
sequence bound, batch partition, loss scaling, token accounting, FSDP,
checkpoint/resume, optimizer memory shape, CPU/thread or memory envelope, or
supported world sizes. The changed class then needs a new descriptor,
certification review, and proportionate evidence.

Every experiment still needs its own scientific protocol, config/artifact
bindings, and completion validation. Execution-class certification supplies
topology safety evidence only; it never waives scientific review.

## Candidate E transition

`prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml` is a new proposed
successor. It changes the execution-evidence gate only after a separate
implementation commit and independent acceptance of the successor amendment,
class certificate, and Candidate E science protocol. Until then it is
non-operational, and accepted Candidate E v1 with its four-real-pilot gate
remains authoritative.

The proposed initial certificate records accepted historical real W=1 and W=2
pilots plus fail-closed static coverage for W=1/2/3/4. Those reports predate the
new fingerprint, so they are evidence reviewed through an explicit
certification bridge, not direct observations of the successor handler. W=3's
uneven-tail collective path and W=4's topology remain without a successful
real-GPU observation; that residual risk is explicit and the runtime must fail
closed at the scheduler-selected world size.

The Candidate E execution-science protocol separately binds seed 42 and its
storage-neutral scientific configuration. Its resolver also rejects later
changes under `src/`, `scripts/`, or `configs/`, so execution-class reuse does
not silently waive scientific implementation review. New job IDs, repetitions,
and output directories do not change the execution-class identity. A different
experiment may reuse the execution certificate under its own scientific
protocol and completion checks without inheriting Candidate E science.

All project registration files in this candidate remain disabled proposals.
Nothing here asserts central installation, enablement, submission, or runtime
deployment.
