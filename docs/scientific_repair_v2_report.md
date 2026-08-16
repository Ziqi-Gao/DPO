# Scientific repair v2 report

## A. Baseline

- Starting Git commit: `c139b25321694cb3f1dd39ecec3ac40e842adaf8`
  (`Complete hash-bound G0 and seed-42 pilot execution`).
- Final state at report generation: the scientific-repair changes are present in the working tree;
  no production job was launched. The publication commit is intentionally reported by the outer
  handoff because a commit cannot contain its own final hash.
- Initial gap map: `docs/scientific_repair_v2_gap_analysis.md`.
- Files inspected included all task generator/parser/renderer/verifier and split code; circuit probe,
  EAP-IG, MIB, exact patching, faithfulness and dynamics code; local-fork supervisors and runners;
  official-TRL routing and reward controls; provenance/readiness/G0/pilot CLIs; all configuration,
  Slurm, preregistration, test, and requested protocol-document files.

The repair preserved the six StateSource × Supervisor cells, the one frozen behavior-policy bank
for offline cells, distinct canonical SFT and GRPO anchors, Qwen2.5-1.5B as primary, and the
EAP-IG-discovery → held-out-exact-patching causal boundary.

## B. Scientific blockers repaired

### Label-leaking ProofGraph

- Previous behavior: label 1 proved a fixed positive query; label 0 used an `UNPROVABLE` query and
  an empty proof. Query identity and proof presence exposed the label.
- Invalidity: a model could solve the task surface without learning signed deduction.
- New behavior: `ProofGraphTask.generate_pair` creates same-query/same-structure siblings that
  derive exactly one of `Q` and `NOT Q`; both labels require a nonempty proof. Role symbols and rule
  IDs are randomized, pair groups are atomic across splits, and query/surface/BOW audits gate
  readiness.
- Main files/functions: `tasks/proofgraph/generator.py::{generate_pair,_canonical_derivation}`,
  `verifier.py::{closure,verify_response}`, `renderer.py`, `parser.py`, `data/splits.py`, and
  `tasks/proofgraph/label_leakage.py`.
- Protection: `test_paired_signed_entailment_hundreds_have_symmetric_nonempty_proofs`,
  `test_pair_groups_never_cross_splits_and_fixed_support_leak_is_detected`, and existing ProofGraph
  unit/stage tests.

### Prompt-end circuit target

- Previous behavior: the primary metric read one `1/0` token at the end of a raw prompt, and
  `query_flip` could stand in for reasoning-path corruption.
- Invalidity: this can measure query routing or answer formatting while missing rule selection and
  deduction; it also mishandles multi-token targets.
- New behavior: `active_support_path_swap` preserves the query and changes critical evidence. One
  frozen semantic manifest expands into tokenizer-specific `first_rule_selection`,
  `intermediate_conclusion`, and `final_answer` probes with explicit full target sequences, target
  IDs, metric/intervention positions, and hashes. EAP-IG and exact patching consume the same bytes.
- Main files/functions: `circuits/probes.py`, `cli/discover_circuit.py`,
  `cli/evaluate_circuit.py`, `circuits/exact_patching.py`, `tiny_eap_ig.py`, `mib_eap_ig.py`, and
  `mib_runner.py`.
- Protection: `test_all_circuit_stages_use_explicit_aligned_sequence_targets`, circuit backend
  tests, MIB row-schema tests, the three-stage repaired smoke, and old/hash-tampered artifact tests.

### Replay-equivalent local policy gradient

- Previous behavior: uncentered binary reward-weighted sequence loss was named as the policy-gradient
  branch. On 0/1 rewards its gradient is collinear with positive-only replay.
- Invalidity: it did not identify a policy-gradient update-rule contrast.
- New behavior: groups have at least four frozen same-prompt trajectories and reward variance.
  `SharedTrajectoryCenteredPolicyGradientSupervisor` freezes standardized group-relative advantages,
  consumes stored old-policy log probabilities, and applies a clipped likelihood-ratio surrogate.
  The old estimator remains only as an explicit diagnostic.
- Main files/functions: `training/local_fork.py::{create_fork_bundle,
  SharedTrajectoryCenteredPolicyGradientSupervisor,run_branch}`, `cli/create_fork_bundle.py`, and
  `cli/run_local_fork.py`.
- Protection: `test_centered_pg_has_negative_advantages_and_non_replay_gradient_geometry`, local-fork
  integration tests, bundle membership/variance checks, and matched-output-KL smoke.

### Teacher, G0, attribution, and distributed execution

- Previous behavior: retained top-k mass could substitute for teacher correctness; G0 inferred
  functional diversity from component-name types; attribution calibration was qualitative; GRPO
  Slurm routes did not prove multi-process launch/batch arithmetic/main-rank ownership.
- New behavior: teacher readiness separately gates generated answer, exact proof, format, first-rule,
  intermediate, top-k coverage/mass, corrupt-prefix recovery, and causal shifts. G0 requires final
  and process-stage quantitative EAP/exact/random/sanity evidence and no component-type proxy.
  Official TRL batch arithmetic is world-size aware, Slurm uses Accelerate, and only the main rank
  writes final artifacts/checkpoints.
- Main files/functions: `teacher/evaluation.py`, `cli/evaluate_teacher_readiness.py`,
  `cli/finalize_g0.py`, `circuits/faithfulness.py`, `training/grpo_backend.py`, `cli/run_grpo.py`, and
  the canonical/pilot/Gemma/G0 Slurm scripts.
- Protection: teacher correctness-vs-mass/hash tests, GRPO contract/routing/update tests, G0 artifact
  compatibility checks, and official tiny-TRL optimizer-step smoke.

### Artifact versioning and random reward

- Core-v2 fields are centralized in `core/scientific_versions.py`. Dataset, trajectory, cohort,
  leakage, teacher, circuit, exact-patching, GRPO, readiness, G0, and pilot paths are version/hash
  bound. Formal loaders reject old v1 artifacts and circuit content tampering.
- Random reward is calibrated once to a frozen exact-reward positive marginal; after calibration it
  hashes only prompt/index and cannot inspect completion correctness. It remains a distinct core-v2
  control rather than a replacement for canonical GRPO.

## C. ProofGraph evidence

CPU artifact root: `outputs/smoke-repaired-g0.cSWBAX`.

- Validated 140 examples across seven disjoint splits: 70 label-1 proofs and 70 label-0 proofs.
- Example pair `pgpair-0fd1c8543309b182b821` has the same query `SYM_044`, topology hash
  `5e2debb...1173ac`, rule-set hash `6e08db4...e4133`, and five proof steps in both siblings.
  The positive final step derives `SYM_044`; the negative final step derives `NOT SYM_044`.
- Pair-group split isolation: passed globally.
- Leakage audit on the 20-example validation fixture:
  - query-only accuracy: `0.50`;
  - surface-feature accuracy: `0.50`;
  - bag-of-words accuracy: `0.50` (vocabulary size 102).
- The audit deliberately fails on a fixture with a label-specific support token.

## D. Circuit-target evidence

All three rows below share semantic pair hash `a2b0ebd...ed2241` and changed semantic field
`facts.active_support`. The full contexts and hashes are retained in each stage's
`tokenized_probe_manifest.json`.

| Stage | Relevant context suffix | Target | Token IDs | Metric positions | Tokenized pair hash |
|---|---|---|---|---|---|
| `first_rule_selection` | `<proof>\nS01: ` | `R03` | `[60]` | `[98]` | `2073720...91182` |
| `intermediate_conclusion` | `S01: R03(F01) -> ` | `TRUE SYM_001` | `[11, 137]` | `[103, 104]` | `458e22e...fff9` |
| `final_answer` | four-step proof ending `TRUE SYM_072\n</proof>\n<answer>` | `1` | `[20]` | `[138]` | `543ed2e...f6d6` |

The complete CPU slice generated a separate faithfulness curve for every stage. Attribution/exact
Spearman was `0.9762`, `-0.3810`, and `0.3095`, respectively. The negative intermediate value is
retained as a tiny-random-model sanity result and is not claimed as real EAP separation.

## E. Local-fork gradient evidence

Deterministic tiny fixture (one prompt, four frozen trajectories, rewards 1/1/0/0):

- replay gradient norm: `2.29934`;
- centered-PG gradient norm: `11.37958`;
- replay vs centered-PG cosine: `-0.011301`;
- centered advantages: `[+0.999998, +0.999998, -0.999998, -0.999998]`;
- negative-advantage trajectories updated: 2;
- old uncentered-estimator gradient norm: `1.14967`;
- replay vs old-uncentered cosine: `1.000004` (floating-point approximation to collinearity).

All four branches restored the same bundle hashes and completed the one-update matched-output-KL
smoke without regenerating trajectories. Nominal horizons 1/5/20 remain configured; update count and
parameter norm remain secondary axes.

## F. GRPO routing evidence

- Backend: pinned official `trl.GRPOTrainer` (TRL 0.22.2 contract).
- CPU optimizer smoke: world size 1, per-device batch 4, accumulation 1, `num_generations=2`, global
  generation batch 4, two groups/update, one optimizer step, update norm `0.0603856`.
- Four-GPU arithmetic test: world size 4 × per-device batch 2 × accumulation 2 = global batch 16;
  with eight generations this is two groups/update. Non-divisible contracts fail.
- Production dry run:
  `.venv/bin/python -m posttrain_circuits.cli.run_grpo experiment=canonical_grpo
  production=qwen_primary --dry-run --output outputs/dry-run/canonical_grpo` (exit 0).
- Slurm routes invoke `accelerate launch`; runtime evidence records main-process-only writes,
  an all-reduced live-parameter checksum, checkpoint hashes, optimizer/RNG state, and update norm.
- The official tiny-TRL smoke wrote an Accelerate-native resumable state
  (`model.safetensors`, `optimizer.bin`, and `random_states_0.pkl`) and a separate hash-bound
  exported model reference. The recorded distributed-consistency check passed.
- No four-GPU run is claimed in this report.

## G. Commands executed

All listed commands exited 0 unless explicitly noted as the diagnosed first attempt.

| Command | Result |
|---|---|
| `make lint` | passed |
| `make typecheck` | passed; 139 source files |
| `make test-scientific-design` | passed; 18 tests |
| `make test` | passed; 149 tests, 18 environment/deprecation warnings |
| `make validate-configs` | passed; six cells resolved correctly |
| `make smoke-factorial` | passed; six cells × two updates |
| `make smoke-sft` | passed; demos plus two updates |
| `make smoke-grpo` | passed; official TRL, one nonzero update |
| `make smoke-local-fork` | passed; four branches |
| `make smoke-resume` | passed; 4 tests |
| `make smoke-circuits` | passed; EAP-IG plus held-out exact patching |
| `make smoke-repaired-g0` | passed on second run; full CPU-only vertical slice |

The first repaired-G0 smoke attempt stopped after split creation because a CLI did not recognize the
repository's `.opd-git` metadata directory. Formal Git lookups were centralized through the existing
fail-closed fallback, then the entire smoke was restarted from a new output root and passed.

Twelve production commands were also run with `--dry-run`: split construction, all six factorial
cells, canonical SFT, canonical GRPO, G0 discovery, G0 exact patching, and anti-shortcut evaluation.
All resolved to Qwen2.5-1.5B production scale and exited 0 without loading weights. No `sbatch` or
Slurm submission command was executed.

## H. Remaining environment-dependent checks

The following still require the clean, committed, pinned real-GPU environment and therefore remain
unproven here:

- Qwen teacher answer/proof/process capability and top-k mass;
- cached Qwen tokenizer alignment on the frozen real probe cohorts;
- HF–TransformerLens Qwen GQA parity;
- real MIB EAP-IG bootstrap stability, selected-vs-random separation, and held-out exact effects;
- four-rank Accelerate GRPO update and synchronized rank hashes;
- four-rank deterministic checkpoint resume;
- real Qwen G0 and its `g0.json` decision;
- the later Gemma mini-replication.

## I. Next safe action

**NOT SAFE TO RUN G0** at report generation.

Reasons: the repair and `core_v2.yaml` must first be committed so the working tree and frozen-prereg
gate are clean; then the short four-GPU CUDA/NCCL/offline-cache preflight and all real Qwen
environment-dependent gates above must pass. After those exact blockers clear, the next safe action
is the repaired G0 only—not the seed-42 pilot, full three-seed factorial, or Gemma replication.
