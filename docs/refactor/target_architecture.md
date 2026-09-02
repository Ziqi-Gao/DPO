# OPD target architecture

This document is the dependency contract for the refactor.  It does not amend a
frozen preregistration, authorize a compute job, register OPD with
ServerScheduler, or authorize removal of a recoverable execution path.

## Scientific question

OPD asks which part of post-training changes a model's causal computation:

1. the states visited by the behavior/current policy;
2. the information carried by the supervision target; or
3. the policy-gradient update rule and reward.

The primary controlled design is therefore a StateSource x Supervisor 2 x 3
factorial.  Fixed-bank and current-policy state sources are crossed with hard
teacher targets, soft teacher top-k forward KL, and exact-verifier-gated replay.
`online_soft_opd` is the unique OPD cell: current-policy states plus the same
soft-teacher objective used by `offline_soft`, with no reward or policy-gradient
term.  Verified teacher-demo SFT and official-TRL GRPO are separate anchors;
format-only and matched-random rewards are GRPO controls.  They are not aliases
for factorial cells.

Circuit analysis is a second, downstream protocol.  Frozen semantic cohorts and
model/tokenizer-specific stage probes feed EAP-IG discovery.  Candidate circuits
are only proposals until held-out exact activation/path patching validates them.
Dynamics, faithfulness, compensation, mask transfer, and matched-accuracy analyses
must retain the complete discovery-to-validation ancestry and may not turn a
thresholded mask or a discovery score into causal evidence by itself.

## Target package ownership

```text
posttrain_circuits/
  artifacts/                 durable bytes, identities, manifests, completion
  datasets/
    proofgraph/              immutable task family, generation, seven-way splits
    trajectories/            typed rollout, scored, verified, and demo records
    teacher_demos/           attempt ledger and accepted-demo view
    circuit_probes/          semantic cohorts and tokenizer-specific stage probes
    anchors/                 non-ProofGraph frozen evaluation data
  methods/                   immutable OPD/SFT/RL/control scientific definitions
  experiments/
    protocols/               factorial, local-fork, and comparison contracts
  learning/
    contracts.py             tensor-free learning protocols
    primitives.py            tensor-bearing batches and model protocols
    state_sources/           fixed-bank and current-policy collection semantics
    supervision/             hard, soft, and verified-replay targets/losses
    teacher/                 teacher scoring and top-k protocol
    training/                trainers, token budget, optimizer, resume
    rl/                      pinned official-TRL adapter and evidence
  causal_circuits/
    discovery/               EAP-IG candidate generation only
    validation/              held-out exact activation/path patching
    metrics/                 faithfulness, CPR/CMD, stability, noise floors
    dynamics/                cross-checkpoint and cross-mask analyses
  workflows/                 immutable DAG plans and single-unit orchestration
  scheduler_adapter/         protocol-v2 validation and foreground dispatch only
  cli/                       thin argument parsing into datasets/workflows/tasks
```

The exact leaf filenames may evolve during reviewed slices; the ownership and
dependency directions below may not.

## Dependency rules

- `artifacts` imports no dataset, method, workflow, scheduler, CLI, or training
  module.  Its low-level contract surface is dependency-light and does not import
  torch.
- Dataset contracts and manifests do not import torch, transformers, a trainer,
  CLI, workflow, or scheduler adapter.  Tensorization belongs to `learning` or
  `causal_circuits`.
- `methods` contains immutable scientific meaning, not runtime trainer objects,
  paths, resources, queues, or process state.
- `experiments.protocols` binds methods and scientific inputs into comparable
  experiments.  Every formal trainer constructs and persists an
  `ExperimentBinding` before the first optimizer update.
- `learning` consumes dataset handles, methods, and artifact contracts.  It does
  not decide which workflow unit runs or which host resource is allocated.
- `causal_circuits` consumes immutable probe/checkpoint handles and writes
  hash-bound discovery or validation artifacts.  Discovery cannot certify itself.
- `workflows` define deterministic scientific DAGs and unit IDs.  They contain no
  queue status, retry loop, GPU selection, lease, or service operation.
- `scheduler_adapter` may validate a running ServerScheduler manifest, resolve an
  allowlisted workflow unit, and synchronously invoke its fixed implementation.
  The registered entrypoint remains in the foreground and waits for any fixed,
  allowlisted Accelerate/FSDP child launcher; it never detaches or fans children
  out as a scheduler.  It validates outputs and publishes or validates only the
  scheduler-neutral `ScientificCompletion`.  ServerScheduler exclusively owns
  attempt state.  The adapter may not submit, poll, schedule, retry, lock,
  daemonize, select a physical GPU, or modify GPU visibility.
- Project request builders may atomically prepare and validate protocol-v2 outbox
  JSON for later external submission.  They never submit or poll it.  Request
  parameters use an exact allowlist of workflow, plan, unit, and content
  identities; they never carry commands, environment overrides, resource
  selection, or scheduler state.
- `cli` modules are thin adapters.  Scientific validation may not exist only in a
  CLI or shell wrapper.

Imports from a canonical package back into a deleted legacy package are forbidden.
The refactor removes old modules and updates every caller atomically; new formal
writers and validators do not add re-export modules, `__getattr__` aliases, legacy
schema defaults, fallback method dispatch, or scheduler-specific terminal fields.
Current formal readers accept only the current schema; historical payload recovery
is an offline migration concern and is not part of the runtime API.

## Identity layers

Every formal unit keeps these identities separate:

1. **Scientific identity** binds protocol track and preregistration, method and
   experiment specs, model/teacher/tokenizer resolved revisions, prompt and
   response-mask protocols, immutable dataset/probe/checkpoint contents, token
   budget unit, optimizer/update semantics, and any method-specific bank, demo,
   or reward-control input.
2. **Resolved identity** hashes the exact composed configuration presented to the
   unit.
3. **Execution-configuration identity** binds concrete storage locations and
   explicit execution overrides.
4. **Attempt provenance** records job, attempt, profile, running-manifest, and
   allocation identities without making them scientific inputs.  A lease is
   validated at runtime but never persisted.
5. **Artifact identity** binds canonical payload content and every declared
   output-byte digest.  A path without its content digest is never sufficient
   ancestry.

Changing a storage root may change resolved and execution identities without
changing scientific identity.  Changing a dataset, ordered prompt schedule,
method objective, model-facing prompt, probe, checkpoint, backend version, batch
contract, or preregistration changes scientific identity.

## Dataset rules

- Formal ProofGraph consumers load a caller-expected, seven-split dataset-family
  manifest.  They never regenerate a train split during training.
- Pair-group ownership and semantic-key isolation are validated across the entire
  family, not independently inside one split directory.
- Trajectory IDs derive from content/protocol identities and actual sampling seed,
  never batch row position.  Stored request seed and actual sampling seed are
  distinct.
- Rollout-bank manifests bind ordered record identities and contents, fixed cursor
  semantics, policy/tokenizer/prompt/sampling protocol, and whether teacher or
  verifier fields are required or forbidden.
- Teacher-demo generation keeps an all-attempt ledger containing successes and
  failures plus a separately hashed accepted-demo view.  Missing success for a
  prompt is an explicit failed or preregistered outcome, never silent dataset
  shrinkage.
- Probe cohort construction keeps counterfactual pair groups together and binds
  the complete ordered candidate population and evidence ancestry.

## Workflow and completion rules

An immutable workflow plan contains scientific units and dependency edges only.
`workflow_id`, `plan_sha256`, and `unit_id` determine output and completion paths;
wall-clock timestamps never define a formal run identity.  A unit writes its
payload files, validates all domain gates, then publishes one no-clobber
`ScientificCompletion`.  Exit code zero without a valid completion marker is not
scientific success.

At-least-once execution may reuse an existing completion only when workflow,
plan, unit, task/run, scientific configuration, inputs, outputs, and validation
checks are identical.  The first valid completion keeps authoritative execution
provenance.

## Migration and deletion gates

Implementation proceeds in reviewed slices: artifacts; methods/protocols;
datasets/learning; circuits; workflows/CLI; storage/config/execution amendment;
ServerScheduler adapter and fixture requests; then full static and CPU acceptance.
Method meaning is defined before datasets, but every `ExperimentBinding` call site
is revalidated after canonical dataset handles replace the old manifest shapes.

Old Slurm and production submission files remain only as a recoverable cutover
surface while required gates are outstanding.  Physical removal requires, in
order, a reviewed disabled registration proposal, protocol-v2 fixture validation,
a separately authorized and successful pilot, approved cutover with drain and a
rollback point, and exact cleanup approval.  No gate authorizes the next one.
Scientific outputs, frozen preregistrations, and validated completion markers are
never cleanup targets.
