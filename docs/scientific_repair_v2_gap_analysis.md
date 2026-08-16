# Scientific repair v2 gap analysis

## Baseline and inspection record

- Review baseline: `4b3479a53484d417d8c41c58ec95b08e3dedbb60` on the sanitized
  public mirror.
- Working HEAD before repair: `c139b25321694cb3f1dd39ecec3ac40e842adaf8` on `master`.
- The two commits are privacy-rewritten counterparts, not ancestor/descendant
  commits. Their trees differ only at `docs/environment_audit.md`: the public
  mirror generalizes the Quest login-node name.
- Working tree before repair: clean.
- The previously queued old-semantics GPU preflight `9430374` was cancelled
  before allocation (`Elapsed=00:00:00`, no node assigned), and its automatic
  G0/pilot monitor was stopped. This repair launches no GPU or Slurm work.
- Required design documents, `prereg/core_v1.yaml`, all modules under the
  ProofGraph, training, circuits, and teacher packages, `data/splits.py`,
  `cli/finalize_g0.py`, and `cli/run_grpo.py` were inspected before code edits.

## Preserved scientific invariants

| Requirement | Verdict | Evidence |
| --- | --- | --- |
| Qwen2.5-1.5B student and Qwen2.5-7B teacher | already satisfied | `configs/model/qwen25_1p5b.yaml`, `configs/teacher/qwen25_teacher_7b.yaml` |
| Gemma-2-2B-it/9B-it replication | already satisfied | `configs/model/gemma2_2b.yaml`, `configs/teacher/gemma2_teacher_9b.yaml` |
| Six `StateSource × Supervisor` cells | already satisfied | `configs/experiment/{offline,online}_{hard,soft,verified_replay}.yaml`, `training/factories.py:build_state_source`, `training/factories.py:build_supervisor` |
| One common frozen bank for all offline cells | already satisfied | `scripts/production/submit_pilot.sh`, `scripts/slurm/pilot_qwen_core.slurm`; every offline arm receives `PILOT_COMMON_BANK` |
| Canonical SFT and official-TRL GRPO remain distinct anchors | already satisfied | `training/canonical_sft.py`, `training/grpo_backend.py:TrlGrpoBackend`, separate experiment configs and CLIs |
| EAP-IG discovery followed by held-out exact patching | already satisfied | `circuits/mib_eap_ig.py:MibEapIgAdapter`, `cli/discover_circuit.py`, `cli/evaluate_circuit.py`; discovery/validation cohort bytes are separate |
| Byte-identical local-fork restoration and output-KL matching | already satisfied | `training/local_fork.py:create_fork_bundle`, `restore_bundle_fresh`, `run_branch`, `calibrate_learning_rate_for_output_kl`; model, optimizer, scheduler, RNG, prompt, trajectory, target, reward, log-probability, and probe hashes are checked |

## Requirement-by-requirement findings

### 1. Paired signed-entailment ProofGraph task — conflicting implementation

- `tasks/proofgraph/generator.py:ProofGraphTask.generate` sets the positive
  query to `Q`, the negative query to `UNPROVABLE`, and emits an empty negative
  proof. There is no paired graph constructor or `pair_group_id`.
- `tasks/proofgraph/verifier.py:verify_response` defines label 0 by absence of a
  query proof and explicitly accepts an empty proof. It does not require exactly
  one of `query` and `query.flipped()` to be derivable.
- `tasks/proofgraph/renderer.py:render_example` therefore exposes a query-label
  shortcut. `configs/task/proofgraph_main.yaml` has no signed-semantics fields
  and enables a 0.5 multiple-proof fraction.
- Needed: deterministic symmetric signed pairs, randomized semantic-role
  symbols, nonempty proofs for both labels, generator version v3, core multiple
  proof fraction 0, and legacy mode only outside core-v2.

### 2. Pair-aware split isolation and manifests — partially satisfied

- `data/splits.py:canonical_semantic_key` and `assert_split_isolation` prevent
  exact semantic duplicates across splits.
- `build_split` generates independent examples, not atomic semantic pairs;
  neither `TaskExample` nor dataset manifests store a pair-group distribution
  or hash.
- Needed: pair-atomic generation/allocation, `pair_group_id` checks in
  `assert_split_isolation`, and manifest bindings.

### 3. Dedicated label-leakage audit — missing

- Anti-shortcut transformations exist in
  `tasks/proofgraph/anti_shortcut.py`, but no query-only, shallow-feature, or
  bag-of-words audit exists. There is no leakage CLI or production gate.
- Needed: deterministic baseline splits, preregistered thresholds, adversarial
  fixtures, hash-bound artifact, and independent G0/readiness failure.

### 4. Explicit stage-specific circuit probes — conflicting implementation

- No `CircuitProbeSpec` or semantic/tokenized probe-manifest schema exists.
- `cli/discover_circuit.py:metric` and `cli/evaluate_circuit.py:metric` compare
  single-token `1/0` logits at `logits[:, -1]` on the raw task prompt.
- `cli/discover_circuit.py` and production validation construct `query_flip`
  pairs. `_padded_pair` shape-matches with arbitrary right padding rather than
  semantic alignment.
- Needed: frozen semantic probes and per-tokenizer manifests for first-rule,
  intermediate-conclusion, and final-answer stages; explicit target and
  intervention positions; a query-preserving critical-support swap.

### 5. Multi-token sequence metrics across discovery and validation — conflicting implementation

- `circuits/mib_runner.py:_FixedPairDataset` rejects any target that is not
  exactly one token, and `_logit_difference` infers the metric position from
  input length.
- `circuits/exact_patching.py` and `tiny_eap_ig.py` accept an arbitrary metric,
  which is useful infrastructure, but all current ProofGraph callers supply the
  prompt-end one-token metric.
- Needed: teacher-forced target-sequence log probability with explicit
  positions, shared by MIB, tiny EAP-IG, exact/path patching, and faithfulness.

### 6. Centered grouped local policy gradient — conflicting implementation

- `core/types.py:TrajectoryRecord` has no generation-group metadata.
- `training/local_fork.py:SharedTrajectoryReinforceSupervisor` implements
  uncentered binary reward-weighted sequence log likelihood.
- `run_branch` and `cli/run_local_fork.py` call the branch
  `grpo_or_reinforce`. Bundle creation does not require group size or within-group
  reward variance.
- Needed: frozen group membership, group-standardized positive and negative
  advantages, old-policy ratios and clipping for repeated updates, exact branch
  naming, and analytical non-collinearity tests.

### 7. Teacher correctness readiness gate — missing

- `teacher/hf_scorer.py:HuggingFaceTeacherScorer` measures retained top-k mass,
  and `teacher/demo_generation.py` verifier-filters demonstrations.
- There is no frozen validation evaluator for answer, proof, format, first-rule,
  intermediate, target coverage, or recovery prefixes. `finalize_g0.py` gates
  only retained mass.
- Needed: teacher evaluator/CLI, provenance-bound artifact, independent
  thresholds, recovery evaluation, and failure diagnosis.

### 8. Functional-stage G0 evidence — conflicting implementation

- `cli/evaluate_circuit.py` maps component names to `attention`, `mlp`,
  `residual`, or `qkv` and emits `functional_group_count`.
- `cli/finalize_g0.py` requires `functional_group_count >= 2`. This is component
  taxonomy, not semantic functional evidence.
- Needed: separate selected-vs-random exact effects for final-answer and at
  least one process stage, with explicit stage/target manifest bindings.

### 9. Quantitative attribution calibration — partially satisfied

- `circuits/faithfulness.py:faithfulness_sparsity_curve` computes attribution to
  exact-patching Spearman and selected-vs-random CPR data.
- `finalize_g0.py` merely checks that Spearman is non-null. It has no prompt
  bootstrap confidence interval, rank precision rule, or quantitative lower
  bounds.
- Needed: configurable core-v2 thresholds and a fail-closed quantitative gate.

### 10. Distributed-safe canonical GRPO — conflicting implementation

- `training/grpo_backend.py:TrlGrpoBackend` correctly uses the pinned official
  `trl.GRPOTrainer`.
- `scripts/slurm/canonical_grpo.slurm`, the GRPO branch in
  `pilot_qwen_core.slurm`, and the Gemma GRPO branch request four GPUs but invoke
  one ordinary Python process.
- `cli/run_grpo.py` checks only per-device batch divisibility, writes artifacts
  from every rank, uses one training prompt for production output KL, clones
  parameters only for tiny smoke, and leaves production update norm `None`.
- Needed: Accelerate routing, installed-TRL global batch arithmetic, main-only
  writes with barriers, multi-prompt behavioral probes, streaming update norm,
  and distributed preflight routing/tests.

### 11. Random-reward control — partially satisfied

- `configs/experiment/grpo_random_reward.yaml` and
  `rewards/random_matched.py:MatchedRandomReward` exist.
- `training/grpo_data.py` may compute the positive rate from exact rewards in
  the same reward call, so the production control is not bound to a frozen
  calibration artifact. It is absent from the pilot/core-v2 execution matrix.
- Needed: frozen calibration hash/rate, no per-example verifier inspection after
  calibration, and inclusion before the final Qwen mechanism claim.

### 12. Core-v2 preregistration and legacy invalidation — missing

- `prereg/core_v0.yaml` and `core_v1.yaml` are preserved correctly.
- `core/provenance.py:PREREG_PATH`, preflight/evaluation CLIs, and documentation
  bind formal runs to `core_v1.yaml`.
- Dataset, trajectory, probe, and circuit artifacts do not uniformly enforce
  generator, label-semantics, circuit-probe-schema, and prereg versions.
- Needed: new immutable `core_v2.yaml`, active provenance, migration guide, and
  fail-closed core-v2 artifact compatibility validation. Historical artifacts
  must remain on disk.

### 13. Repaired G0 — conflicting implementation

- `cli/finalize_g0.py` has strong hash, checkpoint, anti-shortcut, cohort,
  bootstrap, random-control, compatibility, resume, and sanity checks.
- It lacks signed-dataset validity, label-leakage, teacher correctness,
  tokenizer-aligned stage probes, process-stage causal evidence, and
  quantitative calibration; it retains the invalid component-count gate.
- Needed: stage-specific, version-bound inputs and fail-closed checks listed in
  core-v2.

### 14. Tests and CPU vertical slice — partially satisfied

- Existing CPU suites cover the six factorial arms, SFT, official-TRL GRPO,
  local-fork restoration/KL matching, EAP-IG, exact patching, resume, configs,
  and prereg v1.
- They encode the old unprovable task, prompt-end circuit metric, and
  uncentered fork branch. There is no repaired end-to-end slice or
  `smoke-repaired-g0` target.
- Needed: every invariant in sections 14–15 of the repair instruction and all
  existing acceptance commands retained.

### 15. Documentation — conflicting implementation

- The design and causal-discovery/validation distinctions are documented well.
- README and execution docs still name v0/v1 as active; ProofGraph signed
  semantics, stage probes, centered PG, teacher gate, and artifact migration are
  absent.
- Needed: all requested documents, `docs/core_v2_migration.md`, and final
  `docs/scientific_repair_v2_report.md`.

## Repair decision

Verdict before editing: **NOT SAFE TO RUN G0**. The task label leak, invalid
primary circuit target, replay-equivalent local PG branch, missing teacher gate,
invalid functional-group criterion, insufficient calibration gate, and
single-process GRPO route are scientific blockers. Existing architecture and
historical artifacts will be preserved while these specific implementations
are replaced or version-gated.
