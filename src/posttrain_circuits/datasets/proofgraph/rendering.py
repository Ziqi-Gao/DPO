"""Canonical ProofGraph text renderer."""

from __future__ import annotations

from posttrain_circuits.datasets.proofgraph.contracts import ProofStep, TaskExample


RESPONSE_FORMAT_INSTRUCTIONS = (
    "Schema: one rule application per line; no prose or copied facts:\n"
    "<proof>\nS01: R01(F01,F02) -> TRUE X\n</proof>\n<answer>0 or 1</answer>\n"
    "Number consecutively. Cite Fxx/earlier Sxx matching antecedents "
    "exactly. Conclusions TRUE X/NOT X. End proving query:1, negation:0."
)


def render_proof_literal(step: ProofStep) -> str:
    """Use a polarity-explicit, token-symmetric proof conclusion syntax."""

    return str(step.conclusion) if step.conclusion.negated else f"TRUE {step.conclusion.atom}"


def render_example(example: TaskExample) -> str:
    # Compact only delimiters: preserve every graph literal, ID, and ordering.
    # This makes room for explicit response rules inside the reviewed token bound.
    facts = "\n".join(f"{key} {value}" for key, value in example.facts.items())
    rules = "\n".join(
        f"{key} {' AND '.join(str(item) for item in rule.antecedents)} -> {rule.consequent}"
        for key, rule in example.rules.items()
    )
    return (
        f"FACTS\n{facts}\n\nRULES\n{rules}\n\nQUERY\n{example.query}\n\n"
        f"{RESPONSE_FORMAT_INSTRUCTIONS}"
    )


def render_step(step: ProofStep) -> str:
    citations = ",".join(step.citations)
    return f"{step.step_id}: {step.rule_id}({citations}) -> {render_proof_literal(step)}"


def render_target(example: TaskExample) -> str:
    body = "\n".join(render_step(step) for step in example.canonical_proof)
    return f"<proof>\n{body}\n</proof>\n<answer>{example.label}</answer>"
