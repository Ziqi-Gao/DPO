"""Circuit locking estimates on matched axes."""

from __future__ import annotations


def locking_time(
    progress: list[float], similarity_to_final: list[float], threshold: float, stable_points: int = 2
) -> float:
    if len(progress) != len(similarity_to_final):
        raise ValueError("locking inputs must be aligned")
    for index in range(len(progress) - stable_points + 1):
        if all(value >= threshold for value in similarity_to_final[index : index + stable_points]):
            return progress[index]
    return float("nan")
