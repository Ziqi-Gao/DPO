"""Time-resolved thresholded/continuous circuit dynamics."""

from __future__ import annotations

import itertools
import random
from typing import Any

import numpy as np


def _aligned(
    a: dict[str, float],
    b: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(a) | set(b))
    return (
        np.array([a.get(key, 0.0) for key in keys]),
        np.array([b.get(key, 0.0) for key in keys]),
    )


def weighted_overlap(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    av, bv = _aligned(a, b)
    denominator = np.maximum(np.abs(av), np.abs(bv)).sum()
    return float(np.minimum(np.abs(av), np.abs(bv)).sum() / denominator) if denominator else 1.0


def attribution_rank_stability(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    av, bv = _aligned(a, b)
    if len(av) < 2:
        return 1.0
    ar = np.argsort(np.argsort(av))
    br = np.argsort(np.argsort(bv))
    return float(np.corrcoef(ar, br)[0, 1])


def continuous_churn(
    a: dict[str, float],
    b: dict[str, float],
) -> float:
    av, bv = _aligned(a, b)
    denominator = np.abs(av).sum() + np.abs(bv).sum()
    return float(np.abs(av - bv).sum() / denominator) if denominator else 0.0


def thresholded_churn(
    a: dict[str, float],
    b: dict[str, float],
    *,
    threshold: float,
) -> float:
    if threshold < 0:
        raise ValueError("churn threshold must be nonnegative")
    active_a = {name for name, value in a.items() if abs(value) >= threshold}
    active_b = {name for name, value in b.items() if abs(value) >= threshold}
    union = active_a | active_b
    return len(active_a ^ active_b) / len(union) if union else 0.0


def estimate_estimator_noise_floor(
    bootstrap_score_vectors: list[dict[str, float]],
    *,
    activation_threshold: float,
) -> dict[str, float | int]:
    """Estimate same-checkpoint score/mask variability across bootstrap reruns."""

    if len(bootstrap_score_vectors) < 2:
        raise ValueError("estimator noise floor requires at least two bootstrap score vectors")
    pairs = list(itertools.combinations(bootstrap_score_vectors, 2))
    continuous = [continuous_churn(left, right) for left, right in pairs]
    thresholded = [thresholded_churn(left, right, threshold=activation_threshold) for left, right in pairs]
    spearman = [attribution_rank_stability(left, right) for left, right in pairs]
    overlaps = [weighted_overlap(left, right) for left, right in pairs]
    return {
        "bootstrap_replicates": len(bootstrap_score_vectors),
        "pairwise_comparisons": len(pairs),
        "estimated_noise_churn": float(np.mean(continuous)),
        "estimated_thresholded_mask_churn": float(np.mean(thresholded)),
        "estimated_thresholded_jaccard": float(1.0 - np.mean(thresholded)),
        "within_checkpoint_full_score_spearman": float(np.mean(spearman)),
        "within_checkpoint_weighted_overlap": float(np.mean(overlaps)),
    }


def circuit_stability_report(
    *,
    source_scores: dict[str, float],
    target_scores: dict[str, float],
    source_bootstrap_score_vectors: list[dict[str, float]],
    target_bootstrap_score_vectors: list[dict[str, float]],
    activation_threshold: float,
    cross_checkpoint_mask_transfer: dict[str, Any] | list[dict[str, Any]],
    heldout_exact_patching_effects: dict[str, Any],
) -> dict[str, Any]:
    """Combine noise-corrected dynamics with the two causal validation layers."""

    if not cross_checkpoint_mask_transfer:
        raise ValueError("stability report requires cross-checkpoint mask-transfer results")
    if not heldout_exact_patching_effects:
        raise ValueError("stability report requires held-out exact-patching effects")
    source_noise = estimate_estimator_noise_floor(
        source_bootstrap_score_vectors,
        activation_threshold=activation_threshold,
    )
    target_noise = estimate_estimator_noise_floor(
        target_bootstrap_score_vectors,
        activation_threshold=activation_threshold,
    )
    observed = continuous_churn(source_scores, target_scores)
    estimated_noise = float(
        (float(source_noise["estimated_noise_churn"]) + float(target_noise["estimated_noise_churn"])) / 2.0
    )
    observed_thresholded = thresholded_churn(
        source_scores,
        target_scores,
        threshold=activation_threshold,
    )
    return {
        "observed_cross_checkpoint_churn": observed,
        "estimated_noise_churn": estimated_noise,
        "excess_churn": observed - estimated_noise,
        "full_score_spearman_stability": attribution_rank_stability(
            source_scores,
            target_scores,
        ),
        "weighted_overlap": weighted_overlap(source_scores, target_scores),
        "thresholded_jaccard": 1.0 - observed_thresholded,
        "thresholded_jaccard_role": "diagnostic_only_not_sole_churn_evidence",
        "noise_floor": {
            "source_checkpoint": source_noise,
            "target_checkpoint": target_noise,
        },
        "cross_checkpoint_mask_transfer": cross_checkpoint_mask_transfer,
        "heldout_exact_patching_effects": heldout_exact_patching_effects,
    }


def classify_edge_lifecycle(
    series: list[float],
    threshold: float,
) -> str:
    if not series:
        raise ValueError("edge lifecycle series cannot be empty")
    active = [abs(value) >= threshold for value in series]
    if all(active):
        return "inherited"
    if not any(active):
        return "never_active"
    first_active = active.index(True)
    if not active[0] and all(active[first_active:]):
        return "newly_born"
    if active[0]:
        first_inactive = active.index(False)
        if not any(active[first_inactive:]):
            return "pruned"
    transitions = sum(left != right for left, right in itertools.pairwise(active))
    if transitions >= 2 and active[-1]:
        return "resurrected"
    return "transient"


def edge_lifecycle(
    score_series: list[dict[str, float]],
    *,
    threshold: float,
) -> dict[str, str]:
    if len(score_series) < 2:
        raise ValueError("edge lifecycle needs at least two checkpoints")
    edges = sorted(set().union(*(set(scores) for scores in score_series)))
    return {
        edge: classify_edge_lifecycle(
            [scores.get(edge, 0.0) for scores in score_series],
            threshold,
        )
        for edge in edges
    }


def locking_index(
    score_series: list[dict[str, float]],
    *,
    churn_tolerance: float,
) -> int:
    if len(score_series) < 2:
        raise ValueError("locking time needs at least two checkpoints")
    final = score_series[-1]
    for index, _scores in enumerate(score_series):
        if all(continuous_churn(later, final) <= churn_tolerance for later in score_series[index:]):
            return index
    return len(score_series) - 1


def locking_time_bootstrap(
    replicate_series: list[list[dict[str, float]]],
    checkpoints: list[int],
    *,
    churn_tolerance: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 20:
        raise ValueError("locking bootstrap requires >=20 samples")
    if not replicate_series:
        raise ValueError("locking bootstrap requires replicates")
    if any(len(series) != len(checkpoints) for series in replicate_series):
        raise ValueError("every bootstrap replicate must cover all checkpoints")
    rng = random.Random(seed)
    locking_steps = []
    for _ in range(samples):
        selected = [
            replicate_series[rng.randrange(len(replicate_series))] for _ in range(len(replicate_series))
        ]
        averaged = []
        for checkpoint_index in range(len(checkpoints)):
            keys = sorted(set().union(*(set(series[checkpoint_index]) for series in selected)))
            averaged.append(
                {
                    key: sum(series[checkpoint_index].get(key, 0.0) for series in selected) / len(selected)
                    for key in keys
                }
            )
        index = locking_index(
            averaged,
            churn_tolerance=churn_tolerance,
        )
        locking_steps.append(checkpoints[index])
    return {
        "estimate": float(np.median(locking_steps)),
        "lower": float(np.quantile(locking_steps, 0.025)),
        "upper": float(np.quantile(locking_steps, 0.975)),
        "confidence": 0.95,
        "bootstrap_samples": samples,
        "churn_tolerance": churn_tolerance,
    }


def summarize_dynamics(
    score_series: list[dict[str, float]],
    checkpoints: list[int],
    *,
    activation_threshold: float,
    churn_tolerance: float,
    bootstrap_replicates_by_checkpoint: list[list[dict[str, float]]] | None = None,
) -> dict[str, Any]:
    if len(score_series) != len(checkpoints):
        raise ValueError("scores and checkpoints must align")
    if bootstrap_replicates_by_checkpoint is not None and len(bootstrap_replicates_by_checkpoint) != len(
        checkpoints
    ):
        raise ValueError("bootstrap replicates and checkpoints must align")
    noise_floors = (
        [
            estimate_estimator_noise_floor(values, activation_threshold=activation_threshold)
            for values in bootstrap_replicates_by_checkpoint
        ]
        if bootstrap_replicates_by_checkpoint is not None
        else None
    )
    transitions = []
    for index, (left, right) in enumerate(itertools.pairwise(score_series)):
        transition = {
            "from_checkpoint": checkpoints[index],
            "to_checkpoint": checkpoints[index + 1],
            "weighted_overlap": weighted_overlap(left, right),
            "spearman": attribution_rank_stability(left, right),
            "thresholded_churn": thresholded_churn(
                left,
                right,
                threshold=activation_threshold,
            ),
            "continuous_churn": continuous_churn(left, right),
        }
        if noise_floors is not None:
            estimated_noise = (
                float(noise_floors[index]["estimated_noise_churn"])
                + float(noise_floors[index + 1]["estimated_noise_churn"])
            ) / 2.0
            transition["estimated_noise_churn"] = estimated_noise
            transition["excess_churn"] = float(transition["continuous_churn"]) - estimated_noise
        transitions.append(transition)
    return {
        "checkpoints": checkpoints,
        "transitions": transitions,
        "estimator_noise_floor": noise_floors,
        "locking_checkpoint": checkpoints[
            locking_index(
                score_series,
                churn_tolerance=churn_tolerance,
            )
        ],
        "edge_lifecycle": edge_lifecycle(
            score_series,
            threshold=activation_threshold,
        ),
    }
