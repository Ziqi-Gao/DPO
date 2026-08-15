"""Teacher scoring protocol."""

from __future__ import annotations

from typing import Protocol

from posttrain_circuits.core.types import TrajectoryBatch


class TeacherScorer(Protocol):
    def score(self, trajectories: TrajectoryBatch) -> TrajectoryBatch: ...
