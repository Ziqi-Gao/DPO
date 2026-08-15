"""Causal factorization experiments for post-training circuits."""

from posttrain_circuits.core.types import (
    CounterfactualPair,
    LossOutput,
    SupervisionBatch,
    TrajectoryBatch,
    TrajectoryRecord,
)

__all__ = [
    "CounterfactualPair",
    "LossOutput",
    "SupervisionBatch",
    "TrajectoryBatch",
    "TrajectoryRecord",
]
__version__ = "0.1.0"
