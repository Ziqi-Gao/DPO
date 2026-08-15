"""Seeded random reward matching a supplied empirical positive rate."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import cast

from posttrain_circuits.core.hashing import sha256_value


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
        exact_rewards = kwargs.get("exact_rewards")
        rate = self.positive_rate
        if exact_rewards is not None:
            values = [float(value) for value in cast(Sequence[float], exact_rewards)]
            rate = sum(value > 0 for value in values) / len(values) if values else 0.0
        if rate is None:
            raise ValueError("provide positive_rate or exact_rewards to match")
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
                                completions[index],
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
                            completion,
                            index,
                        ]
                    ),
                    index,
                )
                for index, (prompt, completion) in enumerate(zip(prompts, completions, strict=True))
            ]
            random.Random(self.seed).shuffle(keyed)
            selected.update(index for _, index in sorted(keyed)[:count])
        return [float(index in selected) for index in range(len(completions))]
