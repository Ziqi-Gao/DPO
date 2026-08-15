"""Teacher top-1 action supervision on arbitrary visited states."""

from __future__ import annotations

from typing import Any

from posttrain_circuits.core.types import LossOutput, SupervisionBatch, TrajectoryBatch
from posttrain_circuits.data.collators import collate_trajectories
from posttrain_circuits.supervision.losses import hard_teacher_loss


class HardTeacherSupervisor:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def prepare_targets(self, trajectories: TrajectoryBatch, teacher: Any, verifier: Any) -> SupervisionBatch:
        del verifier
        scored = teacher.score(trajectories) if teacher is not None else trajectories
        batch = collate_trajectories(scored, pad_token_id=self.pad_token_id, include_teacher=True)
        assert batch.teacher_topk_ids is not None
        batch.target_ids = batch.teacher_topk_ids[..., 0]
        masses = [value for record in scored.records for value in record.teacher_topk_mass]
        entropies = [value for record in scored.records for value in record.teacher_entropy]
        batch.metadata.update(
            teacher_topk_mass=sum(masses) / len(masses),
            teacher_entropy=sum(entropies) / len(entropies),
        )
        return batch

    def compute_loss(self, model: Any, batch: SupervisionBatch) -> LossOutput:
        assert batch.target_ids is not None
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        loss = hard_teacher_loss(logits, batch.target_ids, batch.response_mask)
        return LossOutput(
            loss,
            {
                "hard_ce_loss": float(loss.detach()),
                "teacher_entropy": float(batch.metadata["teacher_entropy"]),
                "teacher_topk_mass": float(batch.metadata["teacher_topk_mass"]),
            },
        )
