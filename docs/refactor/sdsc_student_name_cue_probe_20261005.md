# Diagnostic of structural hints in atom names

The completed LR diagnostic reduced early runaway generation but left renamed
branch proofs substantially weaker than IID. All ten capped endpoint treatment
rename responses already contain invalid completed inferences, while their valid
canonical targets fit the256-token allowance. Native branch atoms expose two
structural hints: the same-level pair shares a stem, and each path consistently
uses `_L` or `_R`. Ordinary opaque renaming removes both. This motivates a
bounded inference diagnostic, not an assertion that name hints caused every
failure. See `sdsc_student_focus_lr_results_20261005.md` for the observed evidence.

## Fixed scope

Use only the already specified full-exposure step32 weights from both completed
LR jobs54673886 and54673887. These are diagnostic inputs; neither is selected as
a prepared model or promoted. There is no training, optimizer update, checkpoint
search or formal qualification in this task.

The new population has64 fresh bases in32 complete signed pairs:48 depth4 branch
and16 depth4 chain examples, with the existing4–8 distractor range. Two fixed
assignments of an opaque name pool and four conditions produce512 prompts per
model,1024 responses total. No formal896 examples are inspected. Namespaces,
maps, traversal policy, population and readouts must be fixed before execution;
there is no seed search, example filtering, reroll or adaptive extension.

## Four matched conditions

Branch conditions preserve both name hints, break same-level stem pairing only,
break path-wide suffix consistency only, or break both. The pairing intervention
cyclically reassigns right-path stems across the three intermediate levels within
each polarity's path. The path intervention swaps names between the two sides
at a fixed proper subset of levels. A whole-path L/R exchange alone is a graph
symmetry and is not a test of whether the model depends on branch membership.
Chain examples receive fixed generic intermediate-name permutations as a
negative control; their labels do not define a branch-factor estimate.
Each chain has six distinct prompts across its eight rows: block0 `break_pairs`
matches block1 `preserve`, and block0 `break_paths` matches block1 `break_both`.
These prespecified repeated controls are retained and never counted as extra
independent examples. Each branch has eight distinct prompts.

Within each base and map the four variants must be bijective graph renamings,
with identical name multisets and occurrence frequencies, rule/fact order,
labels, citation structure and proof traversal. The actual pinned tokenizer must
verify exact equality of complete model-facing prompt lengths and canonical
response lengths, includingEOS, across variants. Character length alone is not
sufficient. Every target must pass the original verifier and fit256 tokens;
every model prefix plus256 must fit2454. Failed construction stops the task;
it cannot silently remove difficult examples or select different seeds.

## Execution and interpretation

The proposed envelope is one2H100/24CPU/384GiB allocation for90 minutes, with a
4800-second worker bound and600 seconds reserved for persistence. Both models
load serially; rank-local BF16 inference retains greedy decoding, disabled KV
cache, the original prompt protocol and per-candidate RNG discipline. Exact
FP32 master loading, BF16 reload consistency and finite context probes are
required. No new GPU completion or numerical equivalence is established by
this document or CPU fixtures.

The controller must independently bind both historical plans, accepted protocol,
publications, exact step32 artifact hashes and completed accounting. Historical
parent verification remains on its frozen v1 provenance. Only the new consumer
uses the separately reviewed provenance-v2 tools and namespace. It must have its
own implementation commit, independent review and later review-only acceptance
before deployment and submission. Existing scientific controls remain unchanged.

Results retain each model, map, condition and structure, reporting original
proof/answer/format correctness, caps, errors and paired gains/losses. Matched
pairing and suffix effects are descriptive; repeated conditions are not new
independent examples. Any structural prefix-error analysis must be separate
from the unchanged original verifier score. A branch-specific effect replicated
across mappings would support a name-hint hypothesis. Similar outcomes would
reduce its priority only when the comparison is informative. Floor or ceiling
performance cannot distinguish the hypotheses; novel compound-name formatting
can still limit transfer despite matched token lengths. Always report the
absolute performance of the cue-preserving condition. Neither result establishes
an accepted common initial model.

If evidence supports the hint mechanism, a later single-factor training test
can remove that correlation. Otherwise, varying valid proof traversal order is
a separate candidate hypothesis. Merely increasing renamed exposure repeats a
previously attempted direction: V3 already used75% renamed rows and two maps.
Original student-readiness, G0 and subsequent method gates remain intact.

This document defines the implementation design. Current review, source identity,
verification results and any actual job belong in `current_handoff.md` and
immutable submission/evidence records.

CPU verification covers49 core,85 transport and67 worker/auditor cases, including
the actual1024-response producer-to-original-verifier fixture, exact small-model
FP32 loading and BF16 consistency, strict report rejection, actual startup
environment capture, bounded failure publication and genuine v2 history restore
beyond128 commits. Independent graph inspection checks all384 branch factor cells
and16 rank/condition/map balance cells. Full144k/teacher/historical isolation and
actual pinned-tokenizer feasibility also pass. Integration failures in the
isolation schema, error-category naming and loader constant references were
reproduced and fixed before submission. These are CPU execution-contract checks,
not evidence of H100 performance or a scientific name-cue effect.
