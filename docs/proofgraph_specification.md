# ProofGraph specification

ProofGraph is a closed-world, exact symbolic deduction task. Literals are uppercase identifiers or
their explicit negation `NOT X`. Facts bind IDs such as `F01` to literals. Rules bind IDs such as
`R01` to an ordered antecedent list and one consequent:

```text
R01: A AND B -> D
```

A response contains one canonical proof block and one binary answer block. A positive proof applies
rules in topological order. Each step has a monotone step ID, cites facts or earlier steps, and states
the exact rule consequent:

```text
<proof>
S01: R01(F01,F02) -> D
</proof>
<answer>1</answer>
```

For a negative query, the canonical proof is empty and the answer is zero. The verifier computes
the full forward-chaining closure independently. It rejects malformed blocks, missing rules or
citations, antecedent multisets that do not equal the cited rule, wrong conclusions, non-monotone
steps, a final conclusion different from the query, or an answer bit inconsistent with closure.
Verification returns parse, proof, answer, per-step, reward, and stable error-code fields.

Generation is deterministic in `(generator_version, seed, difficulty)`. Phase 0 covers one and two
hops. The same generator accepts depths through seven and chain, branch, and converging-DAG settings;
training configurations use depths 2–4 and OOD-depth uses 5–7. Label balance is imposed by split
construction, not hoped for after sampling. Canonical semantic hashes ignore example IDs and drive
duplicate/leakage checks.

Circuit discovery defaults to unique canonical proof paths. Training may activate additional paths.
The implemented corruption catalogue is fact truth flip, necessary fact replacement, critical-rule
consequent replacement, critical-rule relocation, query flip, distractor replacement, and alternate
path activation. Every pair binds clean/corrupt prompts and targets, the corruption type, and the
exact semantic field changed. Target-changing corruptions are validated to change the target.

Discovery and validation are separate immutable splits. Train, validation, IID test, OOD-depth,
OOD-structure, circuit-discovery, and circuit-validation seeds live in non-overlapping namespaces,
and a cross-split canonical-structure check is mandatory.

