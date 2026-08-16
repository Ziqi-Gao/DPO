# ProofGraph specification

Core-v2 ProofGraph is an exact paired signed-entailment task. Literals are randomized symbols or
their explicit negation `NOT X`. Facts bind IDs such as `F01` to literals. Rules bind IDs such as
`R01` to an ordered antecedent list and one consequent:

```text
R01: A AND B -> D
```

A response contains one canonical proof block and one binary answer block. Every example, including
label 0, applies rules in topological order. Each step has a monotone step ID, cites facts or earlier
steps, and states the exact rule consequent:

```text
<proof>
S01: R01(F01,F02) -> D
</proof>
<answer>1</answer>
```

For label 1, the final proof conclusion is `Q`; for label 0, it is `NOT Q`. A zero answer with an
empty proof is invalid. The verifier computes the full forward-chaining closure independently and
requires exactly one query polarity. It rejects malformed blocks, missing rules or citations,
antecedent multisets that do not equal the cited rule, wrong conclusions, non-monotone steps, a
final conclusion different from the label-selected polarity, or an answer bit inconsistent with
that polarity.
Verification returns parse, proof, answer, per-step, reward, and stable error-code fields.

Generation is deterministic in `(generator_version, seed, difficulty)`. `generate_pair` produces
two siblings with the same query, rule set, topology, distractor count, fact/rule order, and proof
length; only the critical signed support differs. Symbol-role and rule-ID mappings are randomized
and persisted. The generator accepts depths through seven and chain, branch, and converging-DAG settings;
training configurations use depths 2–4 and OOD-depth uses 5–7. Label balance is imposed by split
construction, not hoped for after sampling. Canonical semantic hashes ignore example IDs and drive
duplicate/leakage checks.

Circuit discovery defaults to unique canonical proof paths. Training may activate additional paths.
The primary circuit corruption is `active_support_path_swap`: it preserves the query and switches
the critical evidence to the opposite signed support path. `query_flip` remains an auxiliary
query-routing probe and is not used as primary circuit evidence. The broader corruption catalogue is fact truth flip, necessary fact replacement, critical-rule
consequent replacement, critical-rule relocation, query flip, distractor replacement, and alternate
path activation. Every pair binds clean/corrupt prompts and targets, the corruption type, and the
exact semantic field changed. Target-changing corruptions are validated to change the target.

Discovery and validation are separate immutable splits. Train, validation, IID test, OOD-depth,
OOD-structure, circuit-discovery, and circuit-validation seeds live in non-overlapping namespaces,
and a cross-split canonical-structure check is mandatory.
