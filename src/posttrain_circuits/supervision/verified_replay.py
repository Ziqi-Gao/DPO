"""Verifier-gated behavior cloning."""

from __future__ import annotations

from typing import Any

from posttrain_circuits.core.types import LossOutput, SupervisionBatch, TrajectoryBatch
from posttrain_circuits.data.collators import collate_trajectories
from posttrain_circuits.supervision.losses import verified_replay_loss


class InsufficientPositiveTrajectories(RuntimeError):
    def __init__(self, *, required: int, received: int, generated: int, retry_limit: int) -> None:
        self.required = required
        self.received = received
        self.generated = generated
        self.retry_limit = retry_limit
        super().__init__(
            "verified replay exhausted "
            f"{retry_limit + 1} sampling attempts with {received}/{generated} successes; "
            f"required at least {required}"
        )


class VerifiedReplaySupervisor:
    def __init__(
        self,
        pad_token_id: int,
        *,
        normalization: str = "sequence",
        minimum_positives: int = 1,
        retry_limit: int = 3,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.normalization = normalization
        self.minimum_positives = minimum_positives
        self.retry_limit = retry_limit

    def prepare_targets(self, trajectories: TrajectoryBatch, teacher: Any, verifier: Any) -> SupervisionBatch:
        del teacher, verifier
        positives = sum(record.verifier_reward == 1.0 for record in trajectories.records)
        if positives < self.minimum_positives:
            raise InsufficientPositiveTrajectories(
                required=self.minimum_positives,
                received=positives,
                generated=len(trajectories.records),
                retry_limit=self.retry_limit,
            )
        batch = collate_trajectories(trajectories, pad_token_id=self.pad_token_id)
        batch.metadata.update(
            generated_trajectories=len(trajectories.records),
            successful_trajectories=positives,
            effective_positive_sequences=positives,
            effective_supervised_tokens=sum(
                sum(record.response_token_mask)
                for record in trajectories.records
                if record.verifier_reward == 1.0
            ),
            reward_rate=positives / len(trajectories.records),
            retry_count=0,
        )
        return batch

    def compute_loss(self, model: Any, batch: SupervisionBatch) -> LossOutput:
        assert batch.rewards is not None
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        loss = verified_replay_loss(
            logits,
            batch.input_ids,
            batch.response_mask,
            batch.rewards,
            normalization=self.normalization,
        )
        return LossOutput(loss, {"verified_replay_loss": float(loss.detach()), **batch.metadata})
