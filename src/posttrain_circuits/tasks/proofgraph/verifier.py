"""Exact ProofGraph proof verifier."""

from __future__ import annotations

from collections import Counter

from posttrain_circuits.tasks.proofgraph.schemas import (
    Literal,
    ParsedResponse,
    StepVerification,
    TaskExample,
    VerificationResult,
)


def closure(example: TaskExample) -> set[Literal]:
    established = set(example.facts.values())
    changed = True
    while changed:
        changed = False
        for rule in example.rules.values():
            if all(item in established for item in rule.antecedents) and rule.consequent not in established:
                established.add(rule.consequent)
                changed = True
    return established


def verify_response(example: TaskExample, response: ParsedResponse) -> VerificationResult:
    if not response.parse_valid or response.answer is None:
        return VerificationResult(False, False, False, [], 0.0, response.error_code or "parse_invalid")

    established: dict[str, Literal] = dict(example.facts)
    results: list[StepVerification] = []
    seen_steps: set[str] = set()
    for index, step in enumerate(response.steps, start=1):
        expected_id = f"S{index:02d}"
        error: str | None = None
        rule = example.rules.get(step.rule_id)
        if step.step_id != expected_id or step.step_id in seen_steps:
            error = "step_order"
        elif rule is None:
            error = "unknown_rule"
        elif not step.citations or any(citation not in established for citation in step.citations):
            error = "unknown_citation"
        elif Counter(established[citation] for citation in step.citations) != Counter(rule.antecedents):
            error = "antecedent_mismatch"
        elif step.conclusion != rule.consequent:
            error = "conclusion_mismatch"
        valid = error is None
        results.append(StepVerification(step.step_id, valid, error, step.conclusion if valid else None))
        if not valid:
            return VerificationResult(True, False, False, results, 0.0, error)
        established[step.step_id] = step.conclusion
        seen_steps.add(step.step_id)

    semantic_label = int(example.query in closure(example))
    final_matches = bool(response.steps and response.steps[-1].conclusion == example.query)
    proof_valid = final_matches if response.answer == 1 else not response.steps and semantic_label == 0
    answer_correct = response.answer == example.label == semantic_label
    error = None
    if not proof_valid:
        error = "final_conclusion_mismatch"
    elif not answer_correct:
        error = "answer_mismatch"
    reward = float(response.parse_valid and proof_valid and answer_correct)
    return VerificationResult(True, proof_valid, answer_correct, results, reward, error)
