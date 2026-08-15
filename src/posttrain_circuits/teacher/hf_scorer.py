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
        model: Any,
        *,
        teacher_id: str,
        teacher_revision: str,
        top_k: int = 128,
        include_eos: bool = True,
        minimum_retained_mass: float = 0.90,
        fail_below_mass: bool = False,
    ) -> None:
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.teacher_id = teacher_id
        self.teacher_revision = teacher_revision
        self.top_k = top_k
        self.include_eos = include_eos
        self.minimum_retained_mass = minimum_retained_mass
        self.fail_below_mass = fail_below_mass

    @torch.no_grad()
    def score(self, trajectories: TrajectoryBatch) -> TrajectoryBatch:
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
