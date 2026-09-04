# Qwen3-v2 two-GPU G0 amendment review

Status: proposed; not scientifically accepted and not execution authorization.

The frozen `prereg/qwen3_v2.yaml` requires four visible GPUs and four
nonduplicate ranks. The bounded seed-42 G0 gate is being migrated to two GPUs
without changing its feasibility-only claim, model identities, optimizer,
global token budget, optimizer-step ceiling, checkpoint/evaluation cadence, or
semantic completion requirements. The machine-readable proposal is
`prereg/amendments/qwen3_v2_g0_2gpu_v1.yaml`.

## Batch and token equivalence

The original effective global batch per optimizer update is:

```text
4 ranks × 4 examples/rank/microbatch × 4 accumulation microsteps = 64 examples
```

The amendment fixes:

```text
2 ranks × 4 examples/rank/microbatch × 8 accumulation microsteps = 64 examples
```

The per-device microbatch remains four, so this change does not increase the
activation footprint of one forward/backward. The trainer reserves each full
optimizer window by summing non-padding model-input tokens across all ranks
before applying the update. The G0 token budget remains exactly 2,000,000 in
that global unit, and the independent ceiling remains 120 optimizer updates.
Changing the number of ranks may change rank-local ordering, so this amendment
is restricted to the seed-42 pipeline-feasibility G0 and cannot support a
confirmatory endpoint, the three-seed factorial, or replication.

## Non-self-referential review binding

The implementation commit contains the full code, configuration, tests,
disabled registration proposal, and the amendment with a `proposed` review
block. It intentionally contains no prediction of its own Git hash.

After reviewing that commit, the acceptance commit may change only the
amendment review block and `docs/refactor/current_handoff.md`. The amendment
records the already-existing implementation commit. Validation proves:

- the reviewed implementation is an ancestor of the acceptance and execution
  commits;
- the amendment's scientific fields are byte-semantically unchanged from the
  proposed document in the implementation commit;
- only the amendment and handoff changed before acceptance; and
- only the handoff may change between acceptance, GPU preflight, request
  generation, and execution.

This permits mandatory handoff synchronization without weakening the reviewed
implementation identity. Any code, configuration, handler, test, or other
protocol change after implementation review invalidates the lineage.

## Independent review checklist

- Confirm world size 2, per-device batch 4, accumulation 8, and effective
  global batch 64.
- Confirm global token accounting and the 2,000,000-token boundary are
  unchanged.
- Confirm the 120-step limit remains an optimizer-step safety ceiling.
- Confirm the claim remains seed-42 full-pipeline feasibility only.
- Confirm the proposed amendment is present in the implementation commit.
- Confirm runtime, deployment hashes, handler, validator, and related tests.
- Record reviewer identity, UTC review time, rationale, and the reviewed
  implementation commit only after completing the independent review.

Acceptance does not install or enable the central registration, submit a
preflight or G0 request, operate a GPU, or authorize any larger experiment.
