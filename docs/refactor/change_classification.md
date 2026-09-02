# Refactor change classification

This is a review aid for deciding which refactor changes are execution-equivalent
and which require an explicit scientific/execution amendment or a new protocol
track before any formal run.  It does not edit or supersede a frozen
preregistration and does not authorize registration, pilot execution, or compute.

## Authority and preserved commitments

The frozen preregistration files remain the exhaustive authority.  The list below
is a non-exhaustive navigation aid, not a substitute for a field-by-field diff:

- the 2 x 3 StateSource x Supervisor factorial and unique reward-free OPD cell;
- the same initial checkpoint and, except for the registered factors, controlled
  optimizer, schedule, batch, prompt schedule, and token budget;
- the common offline bank and separate canonical SFT and GRPO anchors (the exact
  verifier gate for SFT demos is an implementation/training-protocol choice that
  the amendment must confirm, rather than an unqualified core-v2 claim);
- the pinned Qwen3 model, teacher, tokenizer, chat template, prompt, sampling, and
  no-thinking identities;
- paired signed-entailment generation and pair-group split isolation;
- frozen semantic cohorts, tokenizer-specific three-stage probes, primary
  endpoints, and the registered sparsity grid;
- EAP-IG discovery followed by held-out exact activation/path patching;
- matched-accuracy interpolation without extrapolation;
- local-fork branches, horizons, KL orientation, tolerance, and fail-closed rule;
- the registered G0 gates, seed-42 pilot feasibility-only claim scope, effect
  direction not being a success gate, and descriptive status of three-seed
  inference.

Package moves, removal of duplicate implementations, strict typing, canonical
hashing, deterministic workflow IDs, path/content separation, and a protocol-v2
foreground adapter do not by themselves change the scientific estimands above.
Replacing the frozen Slurm terminal-evidence/supervision contract with a
scheduler-neutral completion contract does change formal evidence acceptance and
final gates; it therefore requires an explicit amendment even when the scientific
estimand is unchanged.

## Two-axis classification

Every change receives two independent labels; the labels are not mutually
exclusive.

**Rule source**

- **A1 — frozen commitment:** directly enforces a preregistered rule.
- **A2 — existing implementation protocol:** comes from training documentation,
  configuration, or a prior implementation contract but is not stated with the
  same precision in the frozen preregistration.
- **A3 — new correction/choice:** defines behavior that was previously absent,
  ambiguous, or incorrectly implemented.

**Scientific impact**

- **B0 — structural/metadata:** changes ownership or records an identity without
  changing accepted examples, updates, measurements, interventions, or gates.
- **B1 — acceptance boundary:** changes which old or newly generated artifacts are
  accepted, even if valid current-track values remain numerically identical.
- **B2 — data/training trajectory:** changes examples, ordering, seeds, sampling,
  optimizer boundaries, or updates.
- **B3 — estimand/statistics/intervention:** changes the measured population,
  formula, uncertainty, component mapping, or actual intervention.
- **B4 — decision gate:** changes a readiness/finalizer decision or which evidence
  can support a claim.

Any B2–B4 item is scientifically material.  B1 can also require invalidation and
regeneration.  An A1 item is not automatically execution-equivalent: enforcing a
frozen promise can still change the computation that the previous code performed.

## Provisional per-change ledger

This ledger must be expanded to a field-level code/prereg diff before registration.
“Review” means the amendment must decide invalidation after inspecting real
artifact versions; it never means automatic reuse.

| Finding/change | Source | Impact | Historical invalidation | Minimum regeneration scope |
|---|---|---|---|---|
| Package ownership, deterministic IDs, foreground adapter transport | A2/A3 | B0 | No scientific payload upgrade | New code/manifests only |
| Replace frozen Slurm terminal evidence/polling gates with scheduler-neutral completion evidence | A1/A3 | B1/B4 | Record explicit policy for old Slurm evidence; never reinterpret it as ServerScheduler evidence | Completion evidence and every gate/final report that consumes it |
| Complete model/teacher/tokenizer/prompt/checkpoint/probe/backend binding | A1/A2 | B0/B1 | Old unbound artifacts cannot be formal evidence or be upgraded merely by a new envelope | Recreate from contemporaneously bound primitive evidence, otherwise regenerate upstream and descendants |
| Identical offline-soft/OPD top-k objective | A1 | B1/B2 if prior runtime differed | Review old soft-training runs | Affected training and descendants |
| Frozen dataset-family consumption instead of train regeneration | A1 | B1/B2 | Invalidate runs whose realized train rows/order differ | Dataset family, banks/demos, training, circuits |
| ProofGraph generation, pair construction, split assignment, or anti-shortcut semantic repair | A1/A3 | B2/B4 | Invalidate affected dataset namespace and gates | Dataset family and every descendant |
| Stronger ProofGraph semantic validator only | A1/A3 | B1 | Reject rows newly found invalid | Rebuild rejected splits and descendants |
| Common bank manifest/content/order/cursor enforcement | A1 | B1/B2 | Review bank and resume identities | Bank, affected training/resume |
| Actual sampling seed, current-policy refresh/retry, and ordering semantics | A2/A3 | B2 | Invalidate trajectories/runs that sampled differently | Trajectories, training, descendants |
| Exact token-unit spelling | A1 | B0/B1 | Old short-unit payload is historical-only and cannot be upgraded by re-enveloping | Reconstruct only from unambiguous contemporaneous evidence; otherwise regenerate the artifact/run |
| Token counting or stop-before-update boundary | A1/A3 | B2 | Invalidate runs with different update count | Training/checkpoints/descendants |
| Local-fork bundle/state/group/protocol ancestry | A1 | B0/B1 | Old unbound bundles cannot be primary evidence | Bundle, result, finalizer |
| Local-fork probe attention mask and masked KL | A3 | B3 | Invalidate old matched-KL results | Bundle inputs and all fork results |
| Teacher-demo attempt ledger and zero-success policy | A2/A3 | B2 | Invalidate accepted-only stores if population changes | Attempts, accepted view, SFT and descendants |
| Pair-group cohort assignment and candidate population freeze | A1/A3 | B2/B3 | Invalidate split-pair cohorts | Scores, cohorts, probes, circuits |
| Probe token-boundary, stage/target metric, or intervention-span repair | A1/A3 | B2/B3 | Invalidate affected semantic/tokenized probes | Probe manifests and circuit descendants |
| Byte/content ancestry across discovery→exact, circuit→transfer, and circuit/evaluation→dynamics | A1/A3 | B0/B1/B4 | Path-only and same-path-replaceable artifacts cannot support formal claims | Recreate bound upstream artifacts and all exact/transfer/dynamics/final-report descendants |
| Qwen3 GQA component/KV-head mapping, collisions, coverage, projection location | A1/A3 | B3 | Invalidate incompatible discovery/exact artifacts | Discovery, exact, transfer, dynamics |
| Exact declared-position and multi-edge path intervention repair | A1/A3 | B3 | Invalidate interventions produced by old backend | Exact evaluation and descendants |
| Exact corruption/path construction or capability-eligibility repair | A1/A3 | B2/B3 | Review affected pairs/evaluations | Probes/exact evaluation and descendants |
| Bootstrap pair-group population and discovery resampling | A3 | B3 | Invalidate incompatible uncertainty results | Discovery bootstrap and downstream gates |
| Bootstrap SE and tie/constant-correct Spearman | A3 | B3/B4 | Invalidate old uncertainty/stability gates | Metrics, dynamics, final reports |
| CPR/CMD point/CI estimand, mask threshold, locking/noise aggregation | A1/A3 | B3/B4 | Invalidate mismatched estimates/gates | Metrics, dynamics, final reports |
| Exact layer/type/size-matched control enforcement | A1/A3 | B1/B3/B4 | Old fallback controls cannot support primary gate | Control masks, exact evaluation, reports |
| Primitive-measurement gate recomputation | A1/A3 | B1/B4 | Path/boolean-only reports cannot be formal evidence | Finalizers after upstream artifacts |
| Matched-accuracy/progress circuit trajectory | A1/A3 | B3 | No prior behavioral-only interpolation is upgraded | Bracketing circuits, matching, dynamics |
| Prompt bytes, tokenizer alignment, teacher targets, response masks, update normalization, TRL batching | A1/A3 | B2/B3 | Invalidate whenever realized inputs/updates differ | Training plus all descendants |

## Historical artifact policy

Hash-valid historical readers may parse an old payload for audit and migration.
They do not upgrade it into a current formal artifact.  New formal writers use the
new schema and exact protocol values; formal consumers perform semantic validation
above the generic artifact reader.  A historical payload with a short token-unit
name, absent experiment binding, legacy Slurm terminal field, missing attention
mask, or path-only circuit ancestry is never accepted as evidence for the new
track merely because its self-hash is valid.

Frozen scientific outputs and preregistration files remain immutable.  When a
material correction invalidates an older derived artifact, the amendment records
that status; cleanup never deletes the artifact or its validated marker.

## Required approval path before compute

Before a new formal qwen3-v2-derived run can be registered or executed:

1. finish implementation and adversarial/static/CPU validation;
2. produce a reviewed field-level code/prereg diff that covers this provisional
   ledger plus any later finding, and records code identity, regenerated
   namespace, and the historical-artifact rule for each item;
3. prepare a disabled ServerScheduler registration proposal and protocol-v2
   request fixtures;
4. independently review project and central boundaries;
5. obtain separate approval for registration changes;
6. obtain separate approval for a bounded pilot;
7. only after a successful pilot, seek cutover/drain/rollback approval;
8. seek exact cleanup approval before removing recoverable Slurm files.

No step implies approval of the next one.
