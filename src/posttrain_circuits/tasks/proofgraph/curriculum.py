"""Depth/structure curriculum schedule."""

from __future__ import annotations


def difficulty_at_progress(progress: float) -> dict[str, object]:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    if progress < 0.33:
        return {"depth": 2, "structure": "chain", "distractors": 2}
    if progress < 0.66:
        return {"depth": 3, "structure": "branch", "distractors": 4}
    return {"depth": 4, "structure": "converging_dag", "distractors": 8}
