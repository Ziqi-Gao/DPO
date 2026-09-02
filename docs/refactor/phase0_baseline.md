# Phase 0 refactor baseline

This baseline freezes the observable project contracts before the scientific and
scheduler-facing layers are reorganized. It is an engineering baseline, not a new
scientific preregistration and not evidence that any compute job was submitted.

## Protected working-tree state

The refactor began with pre-existing edits in AGENTS.md, README.md, Makefile,
environment/configuration files, two smoke scripts, and
tests/unit/test_qwen3_v2_repairs.py, plus an untracked .codex directory. Those
changes belong to the user. Mechanical import changes in the dirty test are limited
to the canonical module path; its existing assertions and output-root edits remain
untouched.

No worktree was created. No ServerScheduler source, registration, queue, lease,
audit, service, or schema was modified. No ServerScheduler, Slurm, or other compute
job was submitted, and no service was managed.

## Frozen artifact behavior

- JSON files use sorted keys, two-space indentation, UTF-8, and one trailing newline.
- Stable value hashes use compact canonical JSON with sorted mapping keys.
- DatasetManifest hashes exclude sha256 and created_at, and include the serialized examples.
- Trajectory-store hashes exclude sha256 and created_at while binding every declared file hash.
- RunManifest hashes cover every payload field other than the final sha256 field.
- Existing run directories retain manifest.json, resolved_config.yaml, environment.json,
  metrics.jsonl, git_diff.patch, checkpoints, and evaluations.
- Checkpoints retain model, optimizer, scheduler, RNG, prompt scheduler, state source,
  trainer state, token-budget/resume ancestry, dependency, and resolved-config state.
- Formal Qwen3 artifacts continue to fail closed for dirty source, dirty or missing
  preregistration bindings, incomplete prompt/tokenizer bindings, and unavailable Git provenance.

### Post-baseline token-unit correction

This is a post-baseline implementation repair, not a rewrite of the baseline or
the frozen preregistration. The exact active unit is
`global_nonpadding_model_input_tokens_processed`, matching the already frozen
Qwen3-v2 preregistration. Earlier code sometimes emitted the shorter
`global_nonpadding_model_input_tokens`; that spelling is a historical writer
defect, not a second scientific unit. New `RunManifest` and `ExperimentBinding`
writers reject it. A wholly legacy, self-hashed payload with no experiment-binding
fields remains readable by the hash-only artifact reader for audit/migration, but
formal resolvers and pilot finalizers require and semantically validate the v2
experiment binding, so the legacy payload cannot enter new scientific evidence.

The old Python implementation modules are replaced directly by the canonical
posttrain_circuits.artifacts package. All source and test callers switch in the same
change; no compatibility re-export modules remain. Historical artifact payloads
remain hash-readable without being promoted to an active scientific schema. Their
byte/hash validation is intentionally separate from the stricter semantic
validation required by every formal consumer.

## Characterization coverage

The tests under tests/characterization lock:

- historical JSON bytes and dataset/run-manifest hash boundaries;
- run-directory file layout and initialize/finalize lifecycle;
- scientific versus execution configuration identity, immutable binding
  recomputation, and override normalization;
- scheduler-neutral completion-marker output validation, tamper detection,
  workflow/plan/unit identity, approved-root and symlink refusal, UTC ordering,
  publish-once idempotency, cross-attempt reuse, and conflict refusal.

Existing CLI main(argv), dry-run, production-guard, checkpoint, trajectory-store,
provenance, Qwen3 protocol, and scientific repair tests remain the downstream
compatibility suite.

## Deferred migration surfaces

The executable Slurm and production submission paths remain recoverable until a
disabled central registration, fixture validation, separately approved pilot,
caller cutover, drain, rollback point, and exact cleanup approval all exist.
RunManifest therefore temporarily retains slurm_terminal_evidence_sha256 for the
legacy pilot finalizer. New scheduler-neutral completion markers never consume it.

The 561-line training CLI and 715-line GRPO CLI are not moved in this baseline.
Their workflow extraction is a separate reviewed wave so distributed and resume
semantics are not changed together with artifact storage.
