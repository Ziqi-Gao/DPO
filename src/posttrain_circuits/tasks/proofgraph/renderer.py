"""Canonical ProofGraph text renderer."""

from __future__ import annotations

from posttrain_circuits.tasks.proofgraph.schemas import ProofStep, TaskExample


def render_proof_literal(step: ProofStep) -> str:
    """Use a polarity-explicit, token-symmetric proof conclusion syntax."""

    return str(step.conclusion) if step.conclusion.negated else f"TRUE {step.conclusion.atom}"


def render_example(example: TaskExample) -> str:
    facts = "\n".join(f"{key}: {value}" for key, value in example.facts.items())
    rules = "\n".join(
        f"{key}: {' AND '.join(str(item) for item in rule.antecedents)} -> {rule.consequent}"
        for key, rule in example.rules.items()
    )
    return (
        f"FACTS\n{facts}\n\nRULES\n{rules}\n\nQUERY\nIs {example.query} true?\n\n"
        "OUTPUT FORMAT\n<proof>\nS01: R01(F01,F02) -> CONCLUSION\n"
        "</proof>\n<answer>0 or 1</answer>"
    )


def render_step(step: ProofStep) -> str:
    citations = ",".join(step.citations)
    return f"{step.step_id}: {step.rule_id}({citations}) -> {render_proof_literal(step)}"


def render_target(example: TaskExample) -> str:
    body = "\n".join(render_step(step) for step in example.canonical_proof)
    return f"<proof>\n{body}\n</proof>\n<answer>{example.label}</answer>"
