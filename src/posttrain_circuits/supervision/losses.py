"""Numerically explicit controlled-supervision losses."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from posttrain_circuits.teacher.topk import topk_kl


def shifted_response_view(
    logits: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.shape[:2] != values.shape or values.shape != response_mask.shape:
        raise ValueError("logits, values, and response mask have incompatible shapes")
    return logits[:, :-1], values[:, 1:], response_mask[:, 1:]


def hard_teacher_loss(
    logits: torch.Tensor, target_ids: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    shifted_logits, shifted_targets, shifted_mask = shifted_response_view(logits, target_ids, response_mask)
    losses = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape_as(shifted_targets)
    if not shifted_mask.any():
        raise ValueError("hard-teacher batch has no valid response positions")
    return losses[shifted_mask].mean()


def soft_teacher_loss(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_logits = logits[:, :-1]
    ids = topk_ids[:, 1:]
    teacher_logs = topk_logprobs[:, 1:]
    mask = response_mask[:, 1:]
    if not mask.any():
        raise ValueError("soft-teacher batch has no valid response positions")
    student_logits = student_logits[mask]
    ids = ids[mask]
    teacher_logs = teacher_logs[mask]
    per_position = topk_kl(teacher_logs, student_logits, ids, mode=mode)
    selected_student_probs = student_logits.log_softmax(-1).gather(-1, ids).exp()
    student_mass = selected_student_probs.sum(-1)
    teacher_argmax = ids[..., 0]
    student_argmax = student_logits.argmax(-1)
    overlap = (teacher_argmax == student_argmax).float()
    return per_position.mean(), {
        "student_mass_on_teacher_topk": float(student_mass.mean().detach()),
        "teacher_student_topk_overlap": float(overlap.mean().detach()),
    }


def verified_replay_loss(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    rewards: torch.Tensor,
    *,
    normalization: str = "sequence",
) -> torch.Tensor:
    shifted_logits, shifted_targets, shifted_mask = shifted_response_view(logits, token_ids, response_mask)
    token_nll = functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape_as(shifted_targets)
    positive = rewards > 0
    if not positive.any():
        raise ValueError("verified replay batch contains zero successful trajectories")
    positive_mask = shifted_mask & positive[:, None]
    if normalization == "token":
        return token_nll[positive_mask].mean()
    if normalization != "sequence":
        raise ValueError("normalization must be 'sequence' or 'token'")
    counts = positive_mask.sum(dim=1).clamp_min(1)
    sequence_nll = (token_nll * positive_mask).sum(dim=1) / counts
    return sequence_nll[positive].mean()
