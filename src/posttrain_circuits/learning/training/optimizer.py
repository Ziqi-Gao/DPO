"""Shared full-parameter optimizer construction."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def build_adamw(
    parameters: Iterable[torch.nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("full-parameter optimizer found no trainable parameters")
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay, betas=betas)


def parameter_update_norm(before: list[torch.Tensor], parameters: Iterable[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for old, current in zip(before, parameters, strict=True):
        delta = current.detach().cpu().double() - old.double()
        total += delta.square().sum()
    return float(total.sqrt())
