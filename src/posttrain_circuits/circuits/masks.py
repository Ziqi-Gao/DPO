"""Score-threshold and layer-matched random masks."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from posttrain_circuits.circuits.graph import CircuitMask


def top_mask(scores: dict[str, float], sparsity: float) -> CircuitMask:
    if not 0 < sparsity <= 1:
        raise ValueError("sparsity must be in (0, 1]")
    count = max(1, round(len(scores) * sparsity))
    selected = sorted(scores, key=lambda key: abs(scores[key]), reverse=True)[:count]
    return CircuitMask(tuple(selected), count / len(scores))


def _layer(component: str) -> str:
    parts = component.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else "global"


def layer_matched_random_mask(universe: tuple[str, ...], reference: CircuitMask, seed: int) -> CircuitMask:
    rng = random.Random(seed)
    by_layer: dict[str, list[str]] = defaultdict(list)
    for component in universe:
        by_layer[_layer(component)].append(component)
    needed = Counter(_layer(component) for component in reference.components)
    selected: list[str] = []
    for layer, count in needed.items():
        choices = list(by_layer[layer])
        rng.shuffle(choices)
        if len(choices) < count:
            raise ValueError(f"not enough random controls in {layer}")
        selected.extend(choices[:count])
    return CircuitMask(tuple(selected), len(selected) / len(universe))


def layer_size_activation_matched_random_masks(
    universe: tuple[str, ...],
    reference: CircuitMask,
    statistics: dict[str, dict[str, float]],
    *,
    repeats: int,
    seed: int,
) -> list[CircuitMask]:
    if repeats < 2:
        raise ValueError("matched random controls require repeats >= 2")
    if set(universe) != set(statistics):
        raise ValueError("activation statistics must cover the component universe")
    masks = []
    for repeat in range(repeats):
        rng = random.Random(seed + repeat)
        available = set(universe) - set(reference.components)
        if len(available) < len(reference.components):
            raise ValueError("not enough non-circuit components for random controls")
        selected = []
        references = list(reference.components)
        rng.shuffle(references)
        for component in references:
            target = statistics[component]
            same_layer = [
                candidate
                for candidate in universe
                if candidate in available and _layer(candidate) == _layer(component)
            ]
            candidates = same_layer or [candidate for candidate in universe if candidate in available]
            if not candidates:
                raise ValueError("matched random mask exhausted universe")

            def distance(
                candidate: str,
                target_values: dict[str, float] = target,
                randomizer: random.Random = rng,
            ) -> tuple[float, float]:
                values = statistics[candidate]
                size_distance = abs(values["activation_size"] - target_values["activation_size"]) / max(
                    target_values["activation_size"], 1.0
                )
                norm_distance = abs(values["activation_norm"] - target_values["activation_norm"]) / max(
                    target_values["activation_norm"], 1e-12
                )
                jitter = randomizer.random() * 1e-9
                return size_distance + norm_distance, jitter

            choice = min(candidates, key=distance)
            selected.append(choice)
            available.remove(choice)
        masks.append(
            CircuitMask(
                tuple(selected),
                len(selected) / len(universe),
            )
        )
    return masks
