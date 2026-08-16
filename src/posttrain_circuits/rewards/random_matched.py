"""Seeded random reward matching a supplied empirical positive rate."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.scientific_versions import scientific_compatibility_fields


def build_random_reward_calibration(
    *,
    positive_count: int,
    total_count: int,
    source_artifact_hash: str,
    seed: int,
) -> dict[str, Any]:
    if total_count < 1 or not 0 <= positive_count <= total_count:
        raise ValueError("random-reward calibration counts are invalid")
    payload: dict[str, Any] = {
        "format_version": 2,
        **scientific_compatibility_fields(),
        "artifact_kind": "random_reward_positive_marginal_calibration",
        "positive_count": positive_count,
        "total_count": total_count,
        "positive_rate": positive_count / total_count,
        "source_artifact_hash": source_artifact_hash,
        "seed": seed,
        "individual_verifier_results_available_to_reward": False,
    }
    payload["sha256"] = sha256_value(payload)
    return payload


def write_random_reward_calibration(path: Path, artifact: dict[str, Any]) -> None:
    atomic_write_json(path, artifact)


def validate_random_reward_calibration(artifact: dict[str, Any]) -> dict[str, Any]:
    expected = artifact.get("sha256")
    content = {key: value for key, value in artifact.items() if key != "sha256"}
    if expected != sha256_value(content):
        raise ValueError("random-reward calibration hash mismatch")
    if artifact.get("individual_verifier_results_available_to_reward") is not False:
        raise ValueError("random reward must not receive per-example verifier results")
    rate = float(artifact.get("positive_rate", -1.0))
    if not 0.0 <= rate <= 1.0:
        raise ValueError("random-reward calibration positive rate is invalid")
    return artifact


class MatchedRandomReward:
    def __init__(self, seed: int, positive_rate: float | None = None) -> None:
        if positive_rate is not None and not 0.0 <= positive_rate <= 1.0:
            raise ValueError("positive_rate must be in [0, 1]")
        self.seed = seed
        self.positive_rate = positive_rate

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        **kwargs: object,
    ) -> list[float]:
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have equal length")
        if "exact_rewards" in kwargs or "verifier_rewards" in kwargs:
            raise ValueError("matched random reward cannot inspect per-example correctness")
        rate = self.positive_rate
        if rate is None:
            raise ValueError("provide a frozen calibrated positive_rate")
        grouped: dict[str, list[int]] = {}
        for index, prompt in enumerate(prompts):
            grouped.setdefault(prompt, []).append(index)

        selected: set[int] = set()
        if grouped and all(len(indices) >= 2 for indices in grouped.values()):
            for prompt, indices in grouped.items():
                count = round(rate * len(indices))
                if 0.0 < rate < 1.0:
                    count = max(1, min(len(indices) - 1, count))
                keyed = [
                    (
                        sha256_value(
                            [
                                self.seed,
                                prompt,
                                index,
                            ]
                        ),
                        index,
                    )
                    for index in indices
                ]
                random.Random(self.seed).shuffle(keyed)
                selected.update(index for _, index in sorted(keyed)[:count])
        else:
            count = round(rate * len(completions))
            keyed = [
                (
                    sha256_value(
                        [
                            self.seed,
                            prompt,
                            index,
                        ]
                    ),
                    index,
                )
                for index, prompt in enumerate(prompts)
            ]
            random.Random(self.seed).shuffle(keyed)
            selected.update(index for _, index in sorted(keyed)[:count])
        return [float(index in selected) for index in range(len(completions))]
