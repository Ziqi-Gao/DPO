"""Forward-KL online/offline policy distillation."""

from __future__ import annotations

from typing import Any

from posttrain_circuits.core.types import LossOutput, SupervisionBatch, TrajectoryBatch
from posttrain_circuits.data.collators import collate_trajectories
from posttrain_circuits.supervision.losses import soft_teacher_loss


class SoftTeacherSupervisor:
    def __init__(self, pad_token_id: int, *, topk_mode: str = "renormalized") -> None:
        self.pad_token_id = pad_token_id
        self.topk_mode = topk_mode

    def prepare_targets(self, trajectories: TrajectoryBatch, teacher: Any, verifier: Any) -> SupervisionBatch:
        if verifier is not None:
            raise ValueError("soft teacher / OPD must not receive a verifier reward function")
        scored = teacher.score(trajectories) if teacher is not None else trajectories
        batch = collate_trajectories(scored, pad_token_id=self.pad_token_id, include_teacher=True)
        masses = [value for record in scored.records for value in record.teacher_topk_mass]
        entropies = [value for record in scored.records for value in record.teacher_entropy]
        batch.metadata.update(
            teacher_topk_mass=sum(masses) / len(masses),
            teacher_entropy=sum(entropies) / len(entropies),
        )
        return batch

    def compute_loss(self, model: Any, batch: SupervisionBatch) -> LossOutput:
        assert batch.teacher_topk_ids is not None and batch.teacher_topk_logprobs is not None
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        loss, metrics = soft_teacher_loss(
            logits,
            batch.teacher_topk_ids,
            batch.teacher_topk_logprobs,
            batch.response_mask,
            mode=self.topk_mode,
        )
        return LossOutput(
            loss,
            {
                "soft_kl_loss": float(loss.detach()),
                "teacher_entropy": float(batch.metadata["teacher_entropy"]),
                "teacher_topk_mass": float(batch.metadata["teacher_topk_mass"]),
                **metrics,
            },
        )
