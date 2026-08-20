"""Frozen Hugging Face teacher with response-prefix causal alignment."""

from __future__ import annotations

import copy
from typing import Any

import torch

from posttrain_circuits.core.types import TrajectoryBatch
from posttrain_circuits.teacher.topk import topk_from_logits


class HuggingFaceTeacherScorer:
    def __init__(
        self,
        model: Any | None,
        *,
        teacher_id: str,
        teacher_revision: str,
        top_k: int = 128,
        include_eos: bool = True,
        minimum_retained_mass: float = 0.90,
        fail_below_mass: bool = False,
        distributed_rank_zero: bool = False,
    ) -> None:
        self.model = model.eval() if model is not None else None
        if self.model is not None:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        if self.model is None and not distributed_rank_zero:
            raise ValueError("a local teacher model is required unless rank-zero scoring is enabled")
        self.teacher_id = teacher_id
        self.teacher_revision = teacher_revision
        self.top_k = top_k
        self.include_eos = include_eos
        self.minimum_retained_mass = minimum_retained_mass
        self.fail_below_mass = fail_below_mass
        self.distributed_rank_zero = distributed_rank_zero

    @torch.no_grad()
    def _score_local(self, trajectories: TrajectoryBatch) -> TrajectoryBatch:
        if self.model is None:
            raise RuntimeError("rank zero did not load the frozen teacher model")
        scored = copy.deepcopy(trajectories)
        device = next(self.model.parameters()).device
        for record in scored.records:
            response_ids = list(record.response_ids)
            if not self.include_eos and response_ids and response_ids[-1] == self.model.config.eos_token_id:
                response_ids = response_ids[:-1]
            if not record.input_ids:
                raise ValueError("teacher alignment requires at least one prompt token")
            all_ids = torch.tensor([record.input_ids + response_ids], device=device)
            logits = self.model(input_ids=all_ids).logits[0]
            start = len(record.input_ids) - 1
            # logits[start + t] predicts response_ids[t]. This is the key causal shift.
            response_logits = logits[start : start + len(response_ids)]
            ids, logprobs, mass, entropy = topk_from_logits(
                response_logits, min(self.top_k, response_logits.shape[-1])
            )
            if mass.numel() and float(mass.min()) < self.minimum_retained_mass and self.fail_below_mass:
                raise ValueError(
                    f"teacher retained mass {float(mass.min()):.4f} below "
                    f"threshold {self.minimum_retained_mass:.4f}"
                )
            record.teacher_id = self.teacher_id
            record.teacher_revision = self.teacher_revision
            record.teacher_topk_ids = ids.cpu().tolist()
            record.teacher_topk_logprobs = logprobs.cpu().tolist()
            record.teacher_topk_mass = mass.cpu().tolist()
            record.teacher_entropy = entropy.cpu().tolist()
        return scored

    @torch.no_grad()
    def score(self, trajectories: TrajectoryBatch) -> TrajectoryBatch:
        if not self.distributed_rank_zero:
            return self._score_local(trajectories)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("rank-zero teacher scoring requires an initialized process group")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        batches: list[Any] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(batches, trajectories)
        payload: list[Any] = [None]
        if rank == 0:
            payload[0] = [self._score_local(batch) for batch in batches]
        torch.distributed.broadcast_object_list(payload, src=0)
        scored_batches = payload[0]
        if not isinstance(scored_batches, list) or len(scored_batches) != world_size:
            raise RuntimeError("rank-zero teacher returned an invalid distributed score payload")
        scored = scored_batches[rank]
        if not isinstance(scored, TrajectoryBatch):
            raise RuntimeError("rank-zero teacher returned the wrong trajectory type")
        return scored
