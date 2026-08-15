"""Exact normalized-answer verifier."""

from posttrain_circuits.tasks.math_bridge.answer_normalization import normalize_answer


def verify_exact_answer(prediction: str, target: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(target)
