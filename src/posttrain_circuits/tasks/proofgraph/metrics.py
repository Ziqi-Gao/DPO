"""ProofGraph aggregate metrics."""

from __future__ import annotations

from collections.abc import Iterable

from posttrain_circuits.tasks.proofgraph.schemas import VerificationResult


def aggregate_verification(results: Iterable[VerificationResult]) -> dict[str, float]:
    values = list(results)
    if not values:
        return {"format_validity": 0.0, "exact_proof_accuracy": 0.0, "answer_accuracy": 0.0}
    denominator = len(values)
    return {
        "format_validity": sum(result.parse_valid for result in values) / denominator,
        "exact_proof_accuracy": sum(result.reward for result in values) / denominator,
        "answer_accuracy": sum(result.answer_correct for result in values) / denominator,
    }
