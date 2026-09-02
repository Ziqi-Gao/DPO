# LocalFork evidence contract

LocalFork bundle and result writers use format version 4. Earlier path-only
bundle/result payloads are not formal inputs and the new readers reject them;
there is no fallback that upgrades or infers missing evidence.
Every v4 artifact carries the registered
`shared_state_matched_output_kl_v1` protocol identity and LocalForkSpec hash.
Formal bindings have one closed 15-field schema (the 14 standard formal artifact
fields plus `local_fork_source`); unknown formal keys cannot become report fields.

## Source checkpoint ancestry

A production bundle requires an explicit source run manifest. Bundle creation
hash-validates that manifest, parses its `ExperimentBinding`, and semantically
validates the manifest against the method registry. The supplied checkpoint must
be the manifest-declared final checkpoint, must be a non-symlinked regular file
under that run's `checkpoints` directory, and must match the manifest's byte hash.
Its payload is then checked against the same `ExperimentBinding`.
Optimizer states exported by parameter ID or parameter name are canonicalized by
model parameter name before source-to-bundle comparison, so remapped or reordered
moment state cannot pass as an equivalent restoration.

The registered source is not chosen from a path embedded by the bundle. Formal
pilot finalization independently selects
`runs/canonical_sft/seed-42/manifest.json` and passes its path, file hash, and
`ExperimentBinding` hash to the validator. That binding fixes the factorial
design, train/validation datasets, initial checkpoint, model/tokenizer revisions,
prompt protocol, preregistration, and implementation commit.

The bundle's `local_fork_source` binding carries the complete source
`ExperimentBinding` payload/hash, factorial-design hash, method, seed, train and
validation dataset hashes, initial-checkpoint hash, source manifest path plus
payload/file hashes, final-checkpoint path plus byte hash, raw checkpoint state
hashes, effective restored state hashes, and trajectory-bank ancestry. Model,
tokenizer, prompt, preregistration, implementation, and protocol identities must
also match the formal LocalFork configuration. A report validator re-reads the
current source manifest and checkpoint and reruns both semantic validators.

## Branch checkpoint evidence

Every unmatched and matched branch/horizon cell writes separate pre/post
checkpoint evidence. Each checkpoint has an exact field schema and binds:

- bundle ID/hash, LocalFork spec hash, branch, horizon, phase, and any explicit
  calibration learning-rate override;
- model state and model parameter names;
- optimizer state, all moment state, and optimizer-to-model parameter mapping;
- scheduler and RNG state;
- frozen prompt, trajectory, probe input, probe attention-mask, and fork-output
  state;
- canonical state hashes and current file-byte SHA-256.

The optimizer contract is exact: every trainable model parameter occurs once in
canonical model order, parameter-ID/name mappings are bijective, and the AdamW
state-key set equals that parameter set. Each state row contains only `step`,
`exp_avg`, `exp_avg_sq`, plus `max_exp_avg_sq` exactly when AMSGrad is enabled;
moment shapes, dtypes, finiteness, and integer-valued float32 step tensors are
validated. Pre/post step counters must differ by the registered horizon for every
trainable parameter. Parameter-group key sets and all hyperparameters remain
byte-exact from bundle to pre to post, except for the one explicitly authorized
calibration LR; moment shape and dtype must also remain identical to pre.

## Frozen input ancestry

A formal bundle carries exact file and payload identities for the rollout-bank
manifest, LocalFork prompt manifest, and probe manifest, plus the ordered selected
trajectory IDs, every selected record hash, the aggregate selected-record hash,
probe input/mask hashes, and baseline-logit hash. Final validation reopens those
externally selected pilot files with duplicate-key rejection, reruns trajectory
store integrity, and compares prompt text/identity, generation groups, probe token
IDs, the complete binary attention mask, generator version, label semantics,
rollout-generation version, tokenizer revision, and prompt protocol against the
bundle. A self-consistent replacement file cannot redefine the selected bank or
probe.

Trajectory IDs must be globally unique across the complete bank. This is enforced
when `TrajectoryStore` writes or checks the bank, before bundle creation builds an
ID mapping, and again when formal validation reopens the external bank.

Pre-state validation requires exact bundle model, optimizer moments, RNG, prompt,
trajectory, and probe state. The only permitted matched-cell difference before an
update is the explicitly recorded learning rate; normalizing that one field must
recover the exact bundled optimizer and scheduler state. Post-state hashes are
recomputed from the current file and must equal the report's model, optimizer,
moment, scheduler, and RNG claims.

## Final validation

`validate_local_fork_report` requires the expected bundle path and checkpoint
root. It re-reads the current bundle bytes, source manifest/checkpoint, and every
pre/post checkpoint, rejects symlinks and paths outside the declared roots, and
recomputes the complete branch-by-horizon matrix, step counts, calibration
equations and every intermediate calibration-round result, finite metrics, and
primary-validity flags. It also requires the checkpoint tree to contain exactly
the files and directory nodes implied by the report; extra empty directories fail.
The pilot finalizer invokes this validation with
full-protocol and source-binding requirements enabled.

Scientific scalars are recomputed rather than accepted from the report. A model
restored from the bundle must reproduce the baseline logits exactly; each unique
post checkpoint is restored and forwarded on the frozen token IDs and attention
mask to reproduce its recorded logits exactly. The validator then recomputes
`KL(output_new || output_fork)` and full parameter displacement. Independently,
it restores each pre checkpoint's model, optimizer, scheduler, RNG, frozen
trajectories, and pad token and deterministically replays every registered update;
all per-step losses/metrics and complete post model/optimizer/scheduler/RNG/probe
state must match exactly. `loss_evidence.schema_version` is an exact integer, never
a boolean. Calibration learning
rates must equal the registered clamped
`previous_lr * sqrt(target_kl / previous_observed_kl)` rule within the frozen
numeric tolerance.

Bundle files, branch checkpoints, the complete checkpoint directory, and the JSON
report are published with no-clobber semantics. A rerun may reuse an existing
destination only when its canonical state is identical; a conflicting file or
tree is an error. Checkpoints are first built in a unique sibling staging tree and
the complete tree is published atomically.

Missing, extra, stale, relocated, replaced, or partially self-consistent evidence
fails closed. In particular, rehashing a report cannot legitimize a cross-method,
cross-seed, cross-dataset, same-path replacement, or modified checkpoint payload.
