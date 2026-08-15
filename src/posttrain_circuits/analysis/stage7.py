"""Stage 7 shared-state inference with three-level variability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from posttrain_circuits.analysis.bootstrap import (
    hierarchical_metric_intervals,
)
from posttrain_circuits.analysis.factorial import (
    FactorialObservation,
    fit_factorial_interaction,
)
from posttrain_circuits.analysis.shared_state import SourceMode


@dataclass(frozen=True)
class SharedStateObservation:
    record_id: str
    source_mode: SourceMode
    state_source: str
    supervision: str
    training_seed: int
    circuit_replicate: int
    next_token_metric: float
    locking: float
    cpr: float
    cmd: float

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("shared-state observation needs a record ID")
        if self.source_mode not in ("canonical_prefix", "natural_rollout"):
            raise ValueError(f"invalid shared-state source mode: {self.source_mode}")
        values = (
            self.next_token_metric,
            self.locking,
            self.cpr,
            self.cmd,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("shared-state metrics must all be finite")


def _nested_metric(
    observations: list[SharedStateObservation],
    metric: str,
) -> dict[int, dict[str, list[float]]]:
    nested: dict[int, dict[str, list[float]]] = {}
    for observation in observations:
        prompts = nested.setdefault(observation.training_seed, {})
        prompts.setdefault(observation.record_id, []).append(
            float(getattr(observation, metric)),
        )
    return nested


def _source_analysis(
    observations: list[SharedStateObservation],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not observations:
        raise ValueError("shared-state source partition has no observations")
    intervals = hierarchical_metric_intervals(
        {metric: _nested_metric(observations, metric) for metric in ("locking", "cpr", "cmd")},
        samples=bootstrap_samples,
        seed=seed,
    )
    factorial = fit_factorial_interaction(
        [
            FactorialObservation(
                outcome=observation.next_token_metric,
                state_source=observation.state_source,
                supervision=observation.supervision,
                training_seed=observation.training_seed,
                prompt_id=observation.record_id,
                circuit_replicate=observation.circuit_replicate,
            )
            for observation in observations
        ]
    )
    return {
        "observation_count": len(observations),
        "training_seed_count": len(
            {observation.training_seed for observation in observations},
        ),
        "prompt_count": len(
            {observation.record_id for observation in observations},
        ),
        "circuit_replicates": sorted({observation.circuit_replicate for observation in observations}),
        "next_token_metric_mean": float(
            np.mean([observation.next_token_metric for observation in observations])
        ),
        "metric_intervals": intervals,
        "factorial_interaction": asdict(factorial),
    }


def analyze_shared_state(
    observations: list[SharedStateObservation],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Analyze canonical prefixes and natural rollouts without pooling them."""
    source_modes = {observation.source_mode for observation in observations}
    if source_modes != {"canonical_prefix", "natural_rollout"}:
        raise ValueError(
            "analysis requires both canonical-prefix and natural-rollout observations",
        )
    canonical = [observation for observation in observations if observation.source_mode == "canonical_prefix"]
    natural = [observation for observation in observations if observation.source_mode == "natural_rollout"]
    canonical_ids = {observation.record_id for observation in canonical}
    natural_ids = {observation.record_id for observation in natural}
    overlap = canonical_ids & natural_ids
    if overlap:
        raise ValueError(
            f"source modes reuse observation IDs: {sorted(overlap)[:5]}",
        )
    return {
        "schema_version": 1,
        "primary_estimand": "canonical_prefix_next_token_metric",
        "source_modes_pooled": False,
        "three_level_variability": [
            "training_seed",
            "prompt",
            "circuit_bootstrap",
        ],
        "canonical_prefix": {
            "role": "primary_shared_state_analysis",
            **_source_analysis(
                canonical,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            ),
        },
        "natural_rollout": {
            "role": "separate_sensitivity_analysis",
            "sensitivity_analysis_only": True,
            **_source_analysis(
                natural,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1_000_003,
            ),
        },
    }
