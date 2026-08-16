"""Held-out faithfulness curves, matched controls, and uncertainty."""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict
from typing import Any, cast

import numpy as np

from posttrain_circuits.circuits.dynamics import (
    attribution_rank_stability,
)
from posttrain_circuits.circuits.graph import AblationSpec
from posttrain_circuits.circuits.masks import (
    layer_size_activation_matched_random_masks,
    top_mask,
)

REQUIRED_SPARSITY_GRID = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
)


def validate_sparsity_grid(values: list[float]) -> None:
    if tuple(values) != REQUIRED_SPARSITY_GRID:
        raise ValueError(f"circuit sparsity grid must be exactly {list(REQUIRED_SPARSITY_GRID)}")


def integrate_curve(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    ordered = sorted(points)
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in itertools.pairwise(ordered)
    )


def _mean_evaluations(evaluations: list[Any]) -> dict[str, float]:
    fields = tuple(asdict(evaluations[0]))
    return {
        field: sum(float(getattr(evaluation, field)) for evaluation in evaluations) / len(evaluations)
        for field in fields
    }


def _numeric_mean(
    rows: list[dict[str, Any]],
    field: str,
) -> float:
    values = [cast(float, row[field]) for row in rows]
    return sum(values) / len(values)


def _interval(values: list[float]) -> dict[str, float]:
    return {
        "lower": float(np.quantile(values, 0.025)),
        "estimate": float(np.mean(values)),
        "upper": float(np.quantile(values, 0.975)),
        "confidence": 0.95,
    }


def _bootstrap_curve_summaries(
    primary: list[dict[str, Any]],
    *,
    pair_count: int,
    samples: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    rng = random.Random(seed)
    cpr_values = []
    cmd_values = []
    for _ in range(samples):
        indices = [rng.randrange(pair_count) for _ in range(pair_count)]
        faithfulness_points = []
        deviation_points = []
        for point in primary:
            per_pair = point["per_pair"]
            faithfulness = sum(per_pair[index]["faithfulness"] for index in indices) / pair_count
            deviation = (
                sum(
                    abs(per_pair[index]["clean_metric"] - per_pair[index]["patched_metric"])
                    for index in indices
                )
                / pair_count
            )
            faithfulness_points.append((point["sparsity"], faithfulness))
            deviation_points.append((point["sparsity"], deviation))
        cpr_values.append(integrate_curve(faithfulness_points))
        cmd_values.append(integrate_curve(deviation_points))
    return _interval(cpr_values), _interval(cmd_values)


def _calibration_bootstrap(
    attribution: dict[str, float],
    per_pair_patching: list[dict[str, float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if len(per_pair_patching) < 2:
        raise ValueError("attribution calibration bootstrap needs at least two held-out pairs")
    names = sorted(attribution)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(len(per_pair_patching)) for _ in per_pair_patching]
        aggregate = {
            name: sum(per_pair_patching[index][name] for index in indices) / len(indices) for name in names
        }
        values.append(attribution_rank_stability(attribution, aggregate))
    return _interval(values)


def faithfulness_sparsity_curve(
    backend: Any,
    model: Any,
    scores: dict[str, float],
    metric: Any,
    sparsities: list[float],
    validation_pairs: list[Any],
    *,
    patching_scores: dict[str, float] | None = None,
    patching_scores_per_pair: list[dict[str, float]] | None = None,
    random_seed: int = 0,
    random_repeats: int = 5,
    bootstrap_samples: int = 1000,
    baselines: tuple[str, ...] = (
        "counterfactual_replacement",
        "mean",
        "zero",
    ),
) -> dict[str, Any]:
    validate_sparsity_grid(sparsities)
    if len(validation_pairs) < 2:
        raise ValueError("faithfulness needs at least two held-out validation pairs")
    if bootstrap_samples < 20:
        raise ValueError("prompt bootstrap requires >=20 samples")
    curves: dict[str, list[dict[str, Any]]] = {baseline: [] for baseline in baselines}
    random_curve = []
    random_controls = []
    universe = tuple(scores)
    statistics = backend.activation_statistics(model, universe)
    for index, sparsity in enumerate(sparsities):
        mask = top_mask(scores, sparsity)
        for baseline in baselines:
            evaluations = backend.evaluate_mask_per_pair(
                model,
                validation_pairs,
                mask,
                AblationSpec(baseline),
                metric,
            )
            curves[baseline].append(
                {
                    "sparsity": mask.sparsity,
                    **_mean_evaluations(evaluations),
                    "per_pair": [asdict(evaluation) for evaluation in evaluations],
                }
            )
        controls = layer_size_activation_matched_random_masks(
            universe,
            mask,
            statistics,
            repeats=random_repeats,
            seed=random_seed + index * random_repeats,
        )
        control_rows = []
        for control in controls:
            evaluations = backend.evaluate_mask_per_pair(
                model,
                validation_pairs,
                control,
                AblationSpec("counterfactual_replacement"),
                metric,
            )
            control_rows.append(
                {
                    "components": list(control.components),
                    "sparsity": control.sparsity,
                    **_mean_evaluations(evaluations),
                    "per_pair": [asdict(evaluation) for evaluation in evaluations],
                }
            )
        random_controls.append(control_rows)
        numeric_fields = (
            "clean_metric",
            "corrupt_metric",
            "patched_metric",
            "faithfulness",
            "necessity",
            "sufficiency",
        )
        random_curve.append(
            {
                "sparsity": mask.sparsity,
                **{field: _numeric_mean(control_rows, field) for field in numeric_fields},
            }
        )
    primary = curves["counterfactual_replacement"]
    cpr = integrate_curve([(point["sparsity"], point["faithfulness"]) for point in primary])
    random_cpr = integrate_curve([(point["sparsity"], point["faithfulness"]) for point in random_curve])
    cmd = integrate_curve(
        [
            (
                point["sparsity"],
                abs(point["clean_metric"] - point["patched_metric"]),
            )
            for point in primary
        ]
    )
    cpr_ci, cmd_ci = _bootstrap_curve_summaries(
        primary,
        pair_count=len(validation_pairs),
        samples=bootstrap_samples,
        seed=random_seed,
    )
    calibration = None
    spearman = None
    spearman_ci = None
    topk_precision = None
    if patching_scores is not None:
        shared = sorted(set(scores) & set(patching_scores))
        if len(shared) < 2:
            raise ValueError("attribution/patching calibration needs >=2 components")
        attribution = {name: scores[name] for name in shared}
        patching = {name: patching_scores[name] for name in shared}
        spearman = attribution_rank_stability(
            attribution,
            patching,
        )
        calibration = [
            {
                "component": name,
                "attribution_score": scores[name],
                "exact_patching_score": patching_scores[name],
            }
            for name in shared
        ]
        top_k = max(1, round(len(shared) * 0.1))
        attribution_top = {
            name
            for name, _ in sorted(attribution.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        }
        patching_top = {
            name for name, _ in sorted(patching.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
        }
        topk_precision = len(attribution_top & patching_top) / top_k
        if patching_scores_per_pair is not None:
            aligned_per_pair = [
                {name: row[name] for name in shared}
                for row in patching_scores_per_pair
                if all(name in row for name in shared)
            ]
            if len(aligned_per_pair) != len(patching_scores_per_pair):
                raise ValueError("per-pair exact patching calibration omitted scored components")
            spearman_ci = _calibration_bootstrap(
                attribution,
                aligned_per_pair,
                samples=bootstrap_samples,
                seed=random_seed + 7_331,
            )
    return {
        "sparsity_grid": list(REQUIRED_SPARSITY_GRID),
        "validation_pair_count": len(validation_pairs),
        "curves": curves,
        "random_curve": random_curve,
        "random_controls": random_controls,
        "random_matching": [
            "layer",
            "activation_size",
            "activation_norm",
        ],
        "random_repeats": random_repeats,
        "cpr": cpr,
        "random_cpr": random_cpr,
        "selected_vs_matched_random_cpr_margin": cpr - random_cpr,
        "cmd": cmd,
        "cpr_ci": cpr_ci,
        "cmd_ci": cmd_ci,
        "bootstrap_samples": bootstrap_samples,
        "attribution_patching_spearman": spearman,
        "attribution_patching_spearman_ci": spearman_ci,
        "attribution_exact_topk_precision": topk_precision,
        "calibration": calibration,
        "heldout_exact_patching_effects": {
            "necessity": primary[-1]["necessity"],
            "sufficiency": primary[-1]["sufficiency"],
            "faithfulness": primary[-1]["faithfulness"],
        },
    }
