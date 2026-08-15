"""Planned causal contrasts and matched-location interpolation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

PLANNED_CONTRASTS = (
    ("online_hard", "offline_hard"),
    ("online_soft_opd", "offline_soft"),
    ("online_verified_replay", "offline_verified_replay"),
    ("online_soft_opd", "online_hard"),
    ("online_soft_opd", "online_verified_replay"),
    ("online_verified_replay", "canonical_grpo"),
)


def planned_contrasts(cell_means: dict[str, float]) -> dict[str, float]:
    results = {}
    for left, right in PLANNED_CONTRASTS:
        if left in cell_means and right in cell_means:
            results[f"{left} - {right}"] = cell_means[left] - cell_means[right]
    return results


@dataclass(frozen=True)
class InterpolatedLocation:
    target: float
    progress: float
    lower_index: int
    upper_index: int
    lower_checkpoint: float
    upper_checkpoint: float
    interpolation_weight: float
    validation_artifact_hash: str = ""


def interpolate_progress(progress: list[float], metric: list[float], target: float) -> InterpolatedLocation:
    if len(progress) != len(metric) or len(progress) < 2:
        raise ValueError("interpolation requires aligned curves with at least two points")
    for index, (left, right) in enumerate(itertools.pairwise(metric)):
        if min(left, right) <= target <= max(left, right):
            weight = 0.0 if left == right else (target - left) / (right - left)
            location = progress[index] + weight * (progress[index + 1] - progress[index])
            return InterpolatedLocation(
                target,
                location,
                index,
                index + 1,
                progress[index],
                progress[index + 1],
                weight,
            )
    raise ValueError(f"target {target} is outside the observed metric curve")


def match_validation_accuracy(
    checkpoints: list[float],
    validation_accuracy: list[float],
    target: float,
    *,
    validation_artifact_hash: str,
) -> InterpolatedLocation:
    if not validation_artifact_hash.strip():
        raise ValueError("matched accuracy requires a formal validation artifact hash")
    selected = interpolate_progress(checkpoints, validation_accuracy, target)
    return InterpolatedLocation(
        **{
            **selected.__dict__,
            "validation_artifact_hash": validation_artifact_hash,
        }
    )


@dataclass(frozen=True)
class FactorialObservation:
    outcome: float
    state_source: str
    supervision: str
    training_seed: int
    prompt_id: str
    circuit_replicate: int


@dataclass(frozen=True)
class FactorialCoefficient:
    term: str
    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    fdr_p_value: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class FactorialModelResult:
    formula: str
    covariance_type: str
    observations: int
    training_seed_count: int
    state_sources: tuple[str, ...]
    supervision_modes: tuple[str, ...]
    r_squared: float
    adjusted_r_squared: float
    condition_number: float
    coefficients: tuple[FactorialCoefficient, ...]


def fit_factorial_interaction(
    observations: list[FactorialObservation],
) -> FactorialModelResult:
    """Fit state-source x supervision OLS with seed-clustered or HC3 inference."""
    import pandas as pd
    import statsmodels.formula.api as smf

    from posttrain_circuits.analysis.bootstrap import benjamini_hochberg

    if len(observations) < 8:
        raise ValueError("factorial interaction model needs at least eight observations")
    frame = pd.DataFrame(
        [
            {
                "outcome": observation.outcome,
                "state_source": observation.state_source,
                "supervision": observation.supervision,
                "training_seed": observation.training_seed,
                "prompt_id": observation.prompt_id,
                "circuit_replicate": observation.circuit_replicate,
            }
            for observation in observations
        ]
    )
    if not bool(frame["outcome"].map(lambda value: np.isfinite(value)).all()):
        raise ValueError("factorial outcomes must all be finite")
    state_sources = tuple(sorted(frame["state_source"].unique()))
    supervision_modes = tuple(sorted(frame["supervision"].unique()))
    if len(state_sources) < 2 or len(supervision_modes) < 2:
        raise ValueError("factorial model needs at least two levels for each factor")
    cells = {(row.state_source, row.supervision) for row in observations}
    expected_cells = set(itertools.product(state_sources, supervision_modes))
    if cells != expected_cells:
        missing = sorted(expected_cells - cells)
        raise ValueError(f"factorial design has missing cells: {missing}")
    state_reference = state_sources[0]
    supervision_reference = supervision_modes[0]
    formula = (
        "outcome ~ "
        f"C(state_source, Treatment(reference='{state_reference}')) * "
        f"C(supervision, Treatment(reference='{supervision_reference}'))"
    )
    model = smf.ols(formula, data=frame)
    seed_count = int(frame["training_seed"].nunique())
    if seed_count >= 3:
        result = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": frame["training_seed"],
                "use_correction": True,
            },
        )
        covariance_type = "cluster(training_seed)"
    else:
        result = model.fit(cov_type="HC3")
        covariance_type = "HC3"

    terms = list(result.params.index)
    p_values = [float(result.pvalues[term]) for term in terms if term != "Intercept"]
    adjusted = iter(benjamini_hochberg(p_values))
    confidence = result.conf_int(alpha=0.05)
    coefficients = []
    for term in terms:
        fdr_p_value = float(result.pvalues[term]) if term == "Intercept" else float(next(adjusted))
        coefficients.append(
            FactorialCoefficient(
                term=term,
                estimate=float(result.params[term]),
                standard_error=float(result.bse[term]),
                statistic=float(result.tvalues[term]),
                p_value=float(result.pvalues[term]),
                fdr_p_value=fdr_p_value,
                ci_lower=float(confidence.loc[term, 0]),
                ci_upper=float(confidence.loc[term, 1]),
            )
        )
    return FactorialModelResult(
        formula=formula,
        covariance_type=covariance_type,
        observations=len(observations),
        training_seed_count=seed_count,
        state_sources=state_sources,
        supervision_modes=supervision_modes,
        r_squared=float(result.rsquared),
        adjusted_r_squared=float(result.rsquared_adj),
        condition_number=float(result.condition_number),
        coefficients=tuple(coefficients),
    )
