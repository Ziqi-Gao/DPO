# Canonical artifact contract

The posttrain_circuits.artifacts package is the only implementation home for
artifact hashing, durable writes, manifests, provenance, checkpoints, compatibility
validation, configuration identity, and scientific completion.

## Module ownership

- hashing.py owns canonical_json, sha256_value, and sha256_file.
- io.py owns atomic_write_json, publish_json_once, atomic_torch_save, UTC timestamps,
  file fsync, atomic replacement/no-clobber publication, and parent-directory fsync.
- datasets.py owns DatasetManifest.
- runs.py owns preregistration binding, formal artifact provenance, RunManifest,
  run-directory initialization/finalization, and metric appends.
- compatibility.py owns active scientific schema constants and fail-closed loaders.
- checkpoints.py owns complete training checkpoint save/load behavior.
- config_bindings.py owns resolved, scientific, and execution configuration identities.
- completion.py owns the scheduler-neutral scientific completion marker.

Science, data, CLI, and future workflow modules may import these modules. Artifact
modules are dependency leaves: their module-import surface must not import methods,
experiments, data/workflows, a CLI, a scheduler adapter, a local launcher, a resource
manager, or torch. Tensor-dependent operations import torch only inside the called
I/O/checkpoint function. Artifact validators perform canonical hashes and opaque
linkage checks; method-registry and experiment semantics live above this package.

## Hashing scope

Hashes are boundary identities, not a blanket audit mechanism. They cover frozen
external inputs, scientific/config bindings, model checkpoints, top-level dataset
or store manifests, and final completion outputs. Row-level metrics, individual
examples/trajectories, and ordinary intermediate files do not receive independent
hash layers; their containing artifact or manifest is the verification boundary.

## Configuration identities

ConfigBinding records three different hashes:

1. resolved_config_sha256 hashes the exact composed configuration.
2. scientific_config_sha256 replaces storage locations by content identities.
3. execution_config_sha256 hashes concrete storage locations and explicitly supplied
   execution overrides.

The binding has schema_version 3. Its input hashes and storage locators are sorted
immutable tuples, and its execution overrides are stored as canonical JSON rather
than a mutable mapping. recompute_config_binding reconstructs all three identities
from the resolved config and recorded projections; validate_config_binding rejects
any non-canonical, malformed, relocated, or digest-tampered binding.

The storage projection is explicit, not a suffix or substring heuristic. It covers
output_root, prereg_path, state_source.store_path, task.dataset_family_path,
anti_shortcut.report_path, production_safety readiness/probe/initial-checkpoint
locations, and the matched-random calibration path. Every configured input locator
other than output_root requires a same-name SHA-256 content binding. Relocating an
approved root therefore changes resolved and execution identity but not scientific
identity. Changing model, tokenizer, seed, method/objective, token budget, or an
input content hash changes scientific identity.

## Scientific completion

ScientificCompletion contains:

- project, schema version, completion kind, workflow ID, plan SHA-256, unit ID,
  task, and run ID;
- timezone-aware UTC start and completion timestamps, ordered start before finish;
- resolved, scientific, and execution configuration hashes;
- input content hashes;
- absolute output-file paths with declared SHA-256 hashes;
- named scientific validation checks, all of which must be true;
- optional execution identity: job, attempt, profile, running-manifest hash, and
  allocation hash. Lease ownership is deliberately absent.

It intentionally contains no pending/running/failed status, queue position, retry
policy, scheduler state, or process exit-code success claim. Presence of a valid
marker means that project-owned scientific validation passed; a zero process exit
alone never creates it.

Validation always receives explicit approved roots. Production callers use only
/data/del6500/OPD and /scr/del6500/OPD; tests inject their own temporary root. The
marker parent, any existing marker target, and every output are checked without
following symlinks. Outputs and existing marker targets must be regular files, all
parents must be real directories, resolved paths must remain under an approved root,
and every output must match its declared digest.

Publication uses publish_json_once: a fully flushed same-directory temporary file is
hard-linked into place without replacement and the parent directory is fsynced. An
identical racing JSON publication is idempotent and a different one conflicts. For
at-least-once execution, completion then permits a narrower verified fast path: the
same workflow, plan, unit, scientific configuration, input hashes, output paths and
hashes, and scientific checks may reuse the existing marker even when a later attempt
has different job/profile/manifest/allocation metadata. The first valid completion's
timestamps and execution provenance remain authoritative.

## Current-schema boundary

`RunManifest`, `ConfigBinding`, `ExperimentBinding`, and `ScientificCompletion`
are scheduler-neutral. Current readers reject missing experiment/config linkage,
legacy token-budget spellings, and scheduler-specific terminal fields. There is no
runtime compatibility reader or schema-default upgrade path; any future historical
recovery must be an explicit offline migration that cannot feed formal evidence.

Large tensor shards that are written before a final manifest remain protected by the
manifest publication gate: consumers accept them only when the final manifest exists
and all declared hashes validate. Generic torch payload publication uses the single
atomic_torch_save implementation.
