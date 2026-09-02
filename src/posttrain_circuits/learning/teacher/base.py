"""Teacher-scoring protocol."""

from __future__ import annotations

from typing import Protocol

from posttrain_circuits.learning.contracts import TrajectoryBatch


class TeacherScorer(Protocol):
    def score(self, trajectories: TrajectoryBatch) -> TrajectoryBatch: ...
