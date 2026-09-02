"""Supervisor protocol."""

from __future__ import annotations

from typing import Protocol

from posttrain_circuits.learning.contracts import TrajectoryBatch
from posttrain_circuits.learning.primitives import CausalLM, LossOutput, SupervisionBatch
from posttrain_circuits.learning.rl.contracts import RewardFunction
from posttrain_circuits.learning.teacher.base import TeacherScorer


class Supervisor(Protocol):
    def prepare_targets(
        self,
        trajectories: TrajectoryBatch,
        teacher: TeacherScorer | None,
        verifier: RewardFunction | None,
    ) -> SupervisionBatch: ...

    def compute_loss(self, model: CausalLM, batch: SupervisionBatch) -> LossOutput: ...
