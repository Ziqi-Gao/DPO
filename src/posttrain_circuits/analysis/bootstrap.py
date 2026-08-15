"""Reproducible prompt/seed bootstrap and permutation tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("bootstrap requires at least one observation")
    rng = np.random.default_rng(seed)
    draws = [statistic(rng.choice(array, len(array), replace=True)) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def hierarchical_seed_prompt_bootstrap(
    values_by_seed: dict[int, list[float]], *, samples: int = 1000, seed: int = 0
) -> tuple[float, float]:
    if not values_by_seed:
        raise ValueError("hierarchical bootstrap requires training seeds")
    rng = np.random.default_rng(seed)
    seeds = np.array(sorted(values_by_seed))
    draws = []
    for _ in range(samples):
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        seed_means = []
        for sampled_seed in sampled_seeds:
            prompts = np.asarray(values_by_seed[int(sampled_seed)], dtype=float)
            seed_means.append(float(np.mean(rng.choice(prompts, len(prompts), replace=True))))
        draws.append(float(np.mean(seed_means)))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_permutation_test(
    a: Sequence[float], b: Sequence[float], *, samples: int = 10000, seed: int = 0
) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if av.shape != bv.shape or not len(av):
        raise ValueError("paired permutation test requires equal non-empty arrays")
    differences = av - bv
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    null = [abs(float((differences * rng.choice([-1, 1], len(differences))).mean())) for _ in range(samples)]
    return (sum(value >= observed for value in null) + 1) / (samples + 1)


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, float(values[original_index] * count / rank))
        adjusted[original_index] = running
    return adjusted.tolist()


def hierarchical_seed_prompt_circuit_bootstrap(
    values: dict[int, dict[str, list[float]]],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Resample training seeds, prompts within seeds, and circuits within prompts."""
    if len(values) < 2:
        raise ValueError("three-level bootstrap requires at least two training seeds")
    if samples < 20:
        raise ValueError("three-level bootstrap requires at least 20 samples")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0, 1)")
    for training_seed, prompts in values.items():
        if len(prompts) < 2:
            raise ValueError(
                f"training seed {training_seed} needs at least two prompts",
            )
        for prompt_id, circuit_values in prompts.items():
            if len(circuit_values) < 2:
                raise ValueError(
                    f"prompt {prompt_id} needs at least two circuit replicates",
                )
            if not np.all(np.isfinite(np.asarray(circuit_values, dtype=float))):
                raise ValueError("bootstrap values must all be finite")

    rng = np.random.default_rng(seed)
    training_seeds = np.asarray(sorted(values))
    draws = []
    for _ in range(samples):
        sampled_seeds = rng.choice(
            training_seeds,
            len(training_seeds),
            replace=True,
        )
        seed_means = []
        for sampled_seed in sampled_seeds:
            prompts = values[int(sampled_seed)]
            prompt_ids = np.asarray(sorted(prompts))
            sampled_prompts = rng.choice(
                prompt_ids,
                len(prompt_ids),
                replace=True,
            )
            prompt_means = []
            for prompt_id in sampled_prompts:
                circuits = np.asarray(prompts[str(prompt_id)], dtype=float)
                sampled_circuits = rng.choice(
                    circuits,
                    len(circuits),
                    replace=True,
                )
                prompt_means.append(float(np.mean(sampled_circuits)))
            seed_means.append(float(np.mean(prompt_means)))
        draws.append(float(np.mean(seed_means)))

    alpha = (1.0 - confidence) / 2.0
    estimate = float(
        np.mean(
            [
                circuit_value
                for prompts in values.values()
                for circuits in prompts.values()
                for circuit_value in circuits
            ]
        )
    )
    return {
        "estimate": estimate,
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
        "confidence": confidence,
        "bootstrap_samples": float(samples),
        "training_seed_count": float(len(values)),
        "prompt_count": float(
            sum(len(prompts) for prompts in values.values()),
        ),
        "circuit_replicate_count": float(
            sum(len(circuits) for prompts in values.values() for circuits in prompts.values())
        ),
    }


def hierarchical_metric_intervals(
    metrics: dict[str, dict[int, dict[str, list[float]]]],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    required = {"locking", "cpr", "cmd"}
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"missing hierarchical metrics: {sorted(missing)}")
    return {
        metric: hierarchical_seed_prompt_circuit_bootstrap(
            values,
            samples=samples,
            confidence=confidence,
            seed=seed + index * 10_007,
        )
        for index, (metric, values) in enumerate(sorted(metrics.items()))
    }
