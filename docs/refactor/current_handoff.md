# OPD current handoff

Last updated: 2026-09-07.

This is the canonical current-state summary for the OPD refactor and
ServerScheduler integration. AGENTS.md is authoritative for operating and
approval rules. Source, Git, scientific artifacts, and central scheduler state
must still be verified when mutable.

## Repository and authority state

The committed checkout is master at joint successor acceptance commit
46352c4b88013761cd43a83282fd3c6251bf2d9e. That commit changes only the three
review-bearing successor artifacts and this handoff, and binds reviewed
implementation commit 811fd772711d1792d59a2159174886f68721c0c4. The
implementation commit's direct parent is the rejected candidate
e94363527fff315e16b7ce99d7dfe3d7b7ee5063. This lineage descends from
Candidate E v1 acceptance commit 5c0bb34cce288aef8908e498a6f5d3259b998f5b,
whose reviewed implementation commit is
58df5d22f7c09ac69b807eff5294296f6927bd1c.

The independent acceptance review of 811fd772... found no blocking findings
and accepts that exact implementation. The narrow repair rejects every
untracked or ignored file below src and scripts/server_scheduler before src is
made importable, except isolated __pycache__/*.pyc files that the earlier
private pycache-prefix boundary cannot import. Its real Git subprocess test
covers ordinary, .gitignore, and .git/info/exclude src/json.py shadows.

The acceptance commit was verified from a clean worktree. The successor is now
accepted at the OPD Git/lineage layer. This handoff-only state synchronization
is the sole subsequent worktree change; it is non-safety-critical and does not
invalidate the accepted execution fingerprint. Central proposal installation,
project enablement, request generation, and submission remain separate gates.

Candidate E amendment
prereg/amendments/qwen3_v2_g0_elastic_v1.yaml is unchanged, remains accepted,
and still has SHA-256
ff34cc53a85abe409f65ebe1ad3ca4d46b08a0ecd2117b76a27e22eea633a725.
Its per-G0 four-real-pilot gate remains immutable historical v1 authority. The
accepted v2 successor does not rewrite those bytes; it provides the separately
accepted execution-class path for a new Candidate E request.

The successor artifacts are:

- prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml;
- prereg/execution_safety/qwen3_v2_elastic_training_v1.descriptor.json;
- prereg/execution_safety/qwen3_v2_elastic_training_v1.certification.yaml;
- prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_v1.yaml.

All three review-bearing successor documents have identical accepted metadata
bound to implementation commit 811fd772... and were jointly committed at
46352c4b.... Acceptance alone does not authorize central deployment,
enablement, request generation, or submission.

## Correct scheduler-managed GPU model

Each scientific GPU task has one profile with gpu_count_policy = scheduler.
The values 1, 2, 3, and 4 are ServerScheduler claim-time allocation candidates,
not four project tasks, profiles, requests, filenames, or scientific variants.
ServerScheduler alone compares predicted wait plus runtime, selects the actual
count and ordered GPU UUIDs, and applies normal admission. A running attempt is
never resized.

The qwen3_v2_gpu_preflight and qwen3_v2_g0 builders continue to emit
allocation-neutral protocol-v2 requests. The request contains the normal
identity fields and only workflow_id, plan_sha256, and unit_id as parameters.
It omits execution_profile, resources, GPU count, GPU identity, memory,
utilization, exclusivity, and availability hints.

The sole entrypoint remains
/home/del6500/projects/OPD/scripts/server_scheduler/opd-entrypoint. It opens the
running manifest once with no-follow semantics and holds that descriptor
through the GPU child lifetime. Both GPU handlers reread the exact held bytes,
verify the manifest SHA and allocation digest, cross-check
SERVER_SCHEDULER_GPU_COUNT, require the ordered assigned UUIDs to equal
CUDA_VISIBLE_DEVICES, preserve CUDA_VISIBLE_DEVICES unchanged, and launch only
logical devices 0 through N-1. The CPU handler receives no GPU manifest
descriptor.

## Reusable execution-class certification

The new execution class is qwen3-v2-elastic-training-v1. Its descriptor binds
the exact safety-relevant implementation and fixed runtime, plus canonical
safety projections rather than the identity of one experiment.

The fingerprint covers at least:

- outer entrypoint, GPU handlers, adapter runtime, held-manifest and allocation
  validation;
- fixed Python executables, dependency locks, package manifests, model and
  tokenizer revisions;
- model/tensor shape envelope and supported sequence/prompt-population bounds;
- global batch partition for world sizes 1, 2, 3, and 4, loss scaling, and
  rank-local microbatch schedules;
- exact global non-padding token reservation and optimizer-step ceiling;
- requested/effective FSDP wrapper semantics;
- checkpoint boundary, same-world resume, changed-world rejection, and
  attempt isolation;
- cgroup/per-device memory envelope and supported world sizes.

Accepted successor identities are:

- execution descriptor SHA-256:
  d57c6620d090da503d5dbc5d1415a6690b7eb0110ad7cdaaf4ac1e42e6e0a739;
- execution fingerprint:
  ca27527ea4aa345114bb58859ae39078887a084bc078204c32f8782103381108;
- certification SHA-256:
  7af2ad0f643860d7a9632fb643efc17166e1d868ca8643a72255cb591e9413f2;
- certification core SHA-256:
  2d6a9cc556b085e191ad0b0818f1dde9e6e6e52c8ff6367e8ce16935cf884d44;
- accepted v2 amendment SHA-256:
  9971f63015435dbe595183e5376cb5acb8e62c8c40a62854ccbbac60b2fc307d;
- Candidate E science protocol SHA-256:
  1f408237c9e6099356b3602fdb6fe20f7e8d150312d052f81c9e3875a44aa4ba.

The G0 plan no longer embeds eight W=1/2/3/4 report/completion artifacts.
Under the successor design it binds the exact descriptor, reusable
certification, accepted execution-class amendment, and a separate
per-experiment science protocol through immutable content identities. Handler,
finalizer, report, bundle, completion marker, and semantic validator carry and
cross-check the same execution-class and science identities.

The shared execution-safety kernel is used by preflight fixtures and the actual
training path. It checks batch partition before and during training and emits a
final runtime attestation. One global optimizer window remains 64 logical
samples with rank totals 64; 32/32; 22/21/21; or 16/16/16/16. Loss and token
accounting retain the same global semantics. Same-world resume remains exact;
changed-world resume fails before state load. FSDP remains FULL_SHARD for
world sizes 2 through 4 and the reviewed effective NO_SHARD behavior for world
size 1. G0 itself remains scheduler-managed over all four counts.

## Evidence reuse and invalidation

A new experiment may reuse an accepted execution-class certification only when
its recomputed execution fingerprint matches exactly and its shape/batch/token/
FSDP/checkpoint/memory requirements remain inside the certified envelope.

The following do not by themselves invalidate execution-class certification:

- a fresh job_id, workflow_id, or plan_sha256;
- seed or replication identity;
- output or scratch directory;
- scientific parameters that do not affect distributed execution.

Each new experiment must still have its own accepted scientific protocol,
resolved/config artifacts, immutable inputs, output validation, artifact
inventory, and scientific completion decision. Execution topology
certification is not scientific preregistration and does not substitute for
those checks.

Certification is invalidated for immediate reuse by a change to distributed
execution or its verification, including handler/runtime bytes, fixed
dependencies, model or tensor shapes, maximum sequence length, batch
partition, loss scaling, token accounting, FSDP, checkpoint/resume semantics,
memory envelope, or supported world sizes.

Version 1 deliberately uses an exact named set of 52 safety-critical files for
whole-file identity, plus canonical projections for mixed scientific
configuration. A byte mismatch in that named surface pauses automatic reuse
pending review. It does not automatically require four new real pilots. First
classify or isolate the delta; obtain new GPU evidence only when and to the
extent the actual safety envelope changed. This prevents harmless job/science
identity changes from becoming topology recertification while remaining
fail-closed for uncertain safety changes.

## Candidate E migration and evidence

ServerScheduler's general protocol requires reviewed semantic correctness for
all supported world sizes. It does not require every new scientific experiment
to rerun four real GPU pilots. The old four-real-pilot requirement is an OPD
Candidate E v1 project gate.

The accepted v2 amendment replaces that per-plan matrix with joint acceptance
of one reusable execution-class certificate and one per-experiment science
protocol. Its evidence is deliberately stated without overclaiming:

| World size | Successor certification evidence |
| --- | --- |
| 1 | historical accepted real preflight plus current static fail-closed evidence |
| 2 | historical accepted real preflight plus current static fail-closed evidence |
| 3 | static fail-closed evidence only |
| 4 | static fail-closed evidence only |

The W=1 and W=2 reports predate the successor fingerprint. They directly
observed the predecessor preflight handler and shared production-shaped
NCCL/FSDP/training path, not the successor G0 handler or the full G0 pipeline.
Their reuse is therefore an explicit legacy-evidence condensation accepted by
the independent reviewer after inspection of the historical Git blobs and the
successor delta.

The residual risk is explicit: the W=3 uneven 22/21/21 tail and W=4 topology
have no successful real-GPU observation under this execution class. Static
tests prove fail-closed behavior and actual-world binding, while runtime
attestation and semantic completion validation remain mandatory for whichever
world size the scheduler selects. This is the minimum migration that keeps
dynamic scheduling, does not insert resource hints, preserves Candidate E v1,
and avoids making four pilots a permanent per-experiment tax.

Previously recorded central W=3/W=4 preflight request or queue state was not
queried in this migration and is not asserted current here. No existing job was
cancelled or changed. Those pilots may remain useful operational evidence, but
they are no longer a generic prerequisite for every future experiment. The
joint acceptance at 46352c4b... permits the Candidate E successor path to use
the reusable class certificate; central deployment and request gates still
apply.

## Deployment proposals and hashes

Both project-owned registration proposals remain disabled. Each task still has
one scheduler-managed elastic profile with no gpu_count field. The current
project proposal identities are:

GPU preflight:

- handler SHA-256:
  7e908dbbe0aef562b2358242336569db53e8daa14104ce873997b7e7b68692b5;
- package-manifest SHA-256:
  e2284b7c20636f3c85f719cdc0179a0ae5973fdf7cb502d247110db0e5cc037c;
- deployment identity:
  2b58f3da9fc4fe116cf4be5b326082dc0ed3da65d38bc1b082aae9d8716fd155;
- disabled proposal SHA-256:
  abda82622cbcec648b0476edac92502a0616110cc9e3a70ebcbb9bd6637bf7fa.

G0:

- handler SHA-256:
  049b4b35a3feba8898f74246d7f223e571b7df43060151c210efa5bc2510f8da;
- package-manifest SHA-256:
  fb2be67fe2f03d6beb1c6f91f2500e1cdf97853704c8e1f27740bad1757a3057;
- deployment identity:
  3ea87466891f18b18896bda78a0bcafe17da053ef440fd0005cf4fb8aa970a91;
- disabled proposal SHA-256:
  82f609530c0931ce7cb1f230ee4cbf6601cc3fa1613a89be53f31a7dce4961c8.

The actual ServerScheduler parser accepted the CPU profile and both proposed
scheduler-managed GPU profiles in the candidate review. No central
registration was changed during this migration.

## Verification

Static and CPU/no-GPU verification completed on 2026-09-07:

- final scheduler/science focused suite: 234/234 passed;
- independent read-only candidate review before the final lazy-import repair:
  152/152 certification/science/handler tests and 69/69
  registration/adapter tests passed;
- complete test run: 608 passed and one entrypoint isolation test exposed a
  top-level PyYAML import regression;
- after the minimal repair, that exact test passed 1/1 and its affected
  certification/adapter suite passed 67/67;
- the subsequent independent acceptance review reused those results and ran no
  suite; it rejected e943635... solely for the top-level src shadow gap;
- the narrow repair's real entrypoint shadow test and existing bytecode
  isolation test pass 2/2, including ordinary, .gitignore, and
  .git/info/exclude cases;
- this independent review found no blocker in implementation commit
  811fd772...; the exact review-only transition validator passed and
  test_real_git_joint_acceptance_resolves passed 1/1;
- after the user-created acceptance commit, the actual checkout resolver bound
  acceptance commit 46352c4b..., implementation commit 811fd772..., accepted
  review status, and fingerprint ca27527e... successfully;
- both fixed runtimes pass pip check;
- descriptor recomputation, fingerprint/CAS binding, certificate validation,
  handler/package/deployment hashes, compilation, and git diff --check pass;
- the Candidate E v1 amendment is byte-for-byte unchanged; all three successor
  review blocks are identical, accepted, bind 811fd772..., and are committed at
  46352c4b....

CUDA/NCCL lines emitted by unit fixtures are mocks. Tests explicitly hid CUDA.
No GPU was queried or used.

This migration did not call central submit or dispatch, did not create or edit
an outbox request, did not query or alter a job, queue, lease, allocation, or
constraint, did not modify central ServerScheduler, did not enable a project,
and did not restart or signal a service. It performed no external action.
A generated pytest cache was moved recoverably to
/scr/del6500/OPD/tmp/pytest-cache-execution-class-20260907; seven generated
Python bytecode files were removed.

## Required next gates

1. The user commits this handoff-only state synchronization. It does not change
   the accepted execution class or scientific protocol.
2. A central operator separately reviews and installs the exact disabled G0
   proposal at deployments/qwen3_v2_g0/registration-proposal-v2.toml (SHA-256
   82f609530c0931ce7cb1f230ee4cbf6601cc3fa1613a89be53f31a7dce4961c8).
   Project enablement and any service action remain separate approvals.
3. After separate user approval, the central operator enables the reviewed
   registration without changing its scheduler-managed GPU policy.
4. Only after central deployment and enablement may OPD generate one fresh,
   allocation-neutral G0 request. The project must report its absolute path,
   SHA-256, job_id, project HEAD, worktree state, and validation result. A
   central operator validates and submits that exact path separately.

No formal G0 request has been generated. Do not use a fixed GPU count or an
exact-job preflight constraint for G0.

## Quarantined legacy scheduler surface

The unsupported inventory remains: 20 files under scripts/slurm, eight
launch/supervision files under scripts/production, and three Python
pilot/finalizer modules listed in
docs/refactor/legacy_scheduler_inventory.md. They are unreachable from the
ServerScheduler handler registry and remain deletion candidates for a separate
approved cleanup. No local GPU selector, nvidia-smi placement logic, CPU/GPU
lock, lease, local queue, Screen/tmux fan-out, or background scheduler was
introduced.

## Documentation map

- Operating rules: AGENTS.md
- Reusable certification design:
  docs/refactor/qwen3_v2_execution_class_certification.md
- Accepted Candidate E v1 amendment:
  prereg/amendments/qwen3_v2_g0_elastic_v1.yaml
- Accepted execution-class successor:
  prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml
- GPU pilot contract: docs/refactor/qwen3_v2_gpu_preflight_pilot.md
- Legacy inventory: docs/refactor/legacy_scheduler_inventory.md
