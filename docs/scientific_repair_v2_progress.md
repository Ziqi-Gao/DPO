# Scientific repair v2 — completion handoff

The paused repair resumed and the CPU-compatible implementation and verification work completed on
2026-08-16 (America/Chicago). No production GPU experiment was launched during this repair. The
previously queued G0 preflight job `9430374` was cancelled before allocation (`Elapsed=00:00:00`,
no node). The authoritative evidence and remaining real-GPU blockers are recorded in
`docs/scientific_repair_v2_report.md`.

## Baseline and workflow

- Review baseline: `4b3479a53484d417d8c41c58ec95b08e3dedbb60`.
- Starting local working HEAD: `c139b25321694cb3f1dd39ecec3ac40e842adaf8`.
- Git metadata is stored in `.opd-git`; use `git --git-dir=.opd-git --work-tree=.` when the shell has not exported `GIT_DIR`/`GIT_WORK_TREE`.
- Required baseline inspection and document reads were completed.
- Pre-edit classification is recorded in `docs/scientific_repair_v2_gap_analysis.md`.

## Implemented and CPU-validated

1. ProofGraph v3 paired signed entailment
   - Deterministic positive/negative siblings with a common `pair_group_id`.
   - Both labels have nonempty derivations, concluding `Q` or `NOT Q`.
   - Random semantic-role-to-symbol and rule-role mappings.
   - Pair-aware split construction, pair hashes, and core configuration.
   - Dedicated label-leakage audit and CLI.
   - Multiple-proof behavior retained in a secondary configuration.

2. Stage-specific circuit probes
   - Typed semantic and tokenizer-specific manifests.
   - `first_rule_selection`, `intermediate_conclusion`, and `final_answer` stages.
   - Teacher-forced multi-token sequence metrics with explicit positions.
   - Primary `active_support_path_swap` corruption; `query_flip` is auxiliary.
   - MIB, tiny EAP-IG, exact patching, and faithfulness code now consume the frozen probe data.
   - Attribution calibration now includes prompt-bootstrap Spearman intervals and top-k precision.

3. Local fork policy-gradient repair
   - Group metadata and exact membership hashes.
   - Minimum group size and within-group reward-variance checks.
   - Centered, standardized frozen advantages with old-policy ratios and clipping.
   - Branch renamed to `centered_policy_gradient`.
   - Analytical gradient geometry and matched-output-KL tests pass.

4. Teacher and G0 gates
   - Teacher-readiness evaluator/CLI separates full-generation correctness, proof correctness, step accuracy, causal shift, and top-k mass.
   - G0 draft now requires leakage evidence, teacher correctness, a final-answer circuit, a process-stage circuit, matched-random margins, and quantitative attribution calibration.
   - The fake component-type `functional_group_count` criterion was removed from the draft gate.

5. Artifact versioning and preregistration
   - Added `prereg/core_v2.yaml` without modifying v0/v1.
   - Added central core-v2 scientific version constants and fail-closed loaders.
   - Dataset, trajectory, cohort, anti-shortcut, teacher, circuit, exact-patching, readiness, and GRPO artifacts are bound to v2 compatibility fields.
   - Run provenance now points to `core_v2.yaml`.

6. Canonical GRPO and random-reward control
   - Slurm GRPO routes were changed to `accelerate launch`.
   - Added the pinned TRL 0.22.2 global generation-batch arithmetic.
   - Implemented main-rank-only exported artifacts, barriers, multi-prompt output-KL evidence, an all-reduced live-parameter checksum, Accelerate-native resumable state hashes, and a disk-streamed parameter-update norm.
   - Random reward now requires a frozen positive-marginal calibration artifact and refuses per-example verifier rewards.

## Final CPU evidence

- `make smoke-repaired-g0` passed from a clean output root. It validated 140 signed paired examples,
  all six factorial cells over one shared frozen offline bank, canonical SFT, official TRL GRPO, four
  local-fork branches, and all three circuit stages with held-out exact patching.
- The final leakage audit reported query-only, shallow-surface, and bag-of-words accuracy of `0.50`.
- `make test-scientific-design` passed 18 tests; the full suite passed 149 tests.
- Lint, typecheck, configuration validation, individual smoke targets, and all twelve production
  command dry-runs passed. The complete command table is in the final report.

## Remaining environment-dependent work

The real Qwen teacher/process gates, cached production tokenizer alignment, HF–TransformerLens GQA
parity, real MIB EAP-IG/exact-patching evidence, and four-rank Accelerate/resume checks still require
the clean, committed, pinned GPU environment. They are deliberately not claimed by the CPU smoke.
The next execution must follow the gate sequence in the final report and must not launch a pilot,
full factorial, or Gemma replication before its explicit prerequisite and authorization gates.

Current recommendation: **NOT SAFE TO RUN G0**.
