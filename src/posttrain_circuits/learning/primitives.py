"""Tensor-backed primitives for learning implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass
class SupervisionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    target_ids: torch.Tensor | None = None
    teacher_topk_ids: torch.Tensor | None = None
    teacher_topk_logprobs: torch.Tensor | None = None
    teacher_topk_mass: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    sequence_ids: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


class CausalLM(Protocol):
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> Any: ...
