"""Self-repair diagnostic."""

from __future__ import annotations


def repair_score(direct_change: float, final_change: float, epsilon: float = 1e-12) -> float:
    return 1.0 - abs(final_change) / (abs(direct_change) + epsilon)
