"""Top-k teacher distribution utilities."""

from __future__ import annotations

import torch


def topk_from_logits(
    logits: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError(f"k={k} must be in [1, {logits.shape[-1]}]")
    log_probs = logits.log_softmax(dim=-1)
    topk_logprobs, topk_ids = log_probs.topk(k, dim=-1)
    mass = topk_logprobs.exp().sum(dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return topk_ids, topk_logprobs, mass, entropy


def topk_kl(
    teacher_logprobs: torch.Tensor,
    student_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    mode: str = "renormalized",
) -> torch.Tensor:
    """KL over retained tokens, optionally with one omitted-vocabulary tail bucket."""
    student_logprobs = student_logits.log_softmax(dim=-1)
    selected_student = student_logprobs.gather(-1, topk_ids)
    if mode == "renormalized":
        log_teacher = teacher_logprobs - torch.logsumexp(teacher_logprobs, dim=-1, keepdim=True)
        log_student = selected_student - torch.logsumexp(selected_student, dim=-1, keepdim=True)
        teacher_probs = log_teacher.exp()
        return (teacher_probs * (log_teacher - log_student)).sum(dim=-1)
    if mode == "tail_bucket":
        teacher_top = teacher_logprobs.exp()
        student_top = selected_student.exp()
        teacher_tail = (1.0 - teacher_top.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
        student_tail = (1.0 - student_top.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
        teacher_probs = torch.cat((teacher_top, teacher_tail), dim=-1)
        student_probs = torch.cat((student_top, student_tail), dim=-1)
        return (teacher_probs * (teacher_probs.log() - student_probs.log())).sum(dim=-1)
    raise ValueError("top-k mode must be 'renormalized' or 'tail_bucket'")
