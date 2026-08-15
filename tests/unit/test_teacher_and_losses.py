from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.core.types import TrajectoryBatch, TrajectoryRecord
from posttrain_circuits.supervision.losses import (
    hard_teacher_loss,
    soft_teacher_loss,
    verified_replay_loss,
)
from posttrain_circuits.teacher.hf_scorer import HuggingFaceTeacherScorer
from posttrain_circuits.teacher.topk import topk_from_logits, topk_kl


class PositionalTeacher(torch.nn.Module):
    def __init__(self, vocab: int = 16) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(eos_token_id=3)
        self.vocab = vocab

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocab), -20.0)
        for position in range(length):
            logits[:, position, position + 4] = 20.0 + self.anchor
        return SimpleNamespace(logits=logits)


def record() -> TrajectoryRecord:
    return TrajectoryRecord(
        trajectory_id="t",
        prompt_id="p",
        split="train",
        prompt_text="p",
        input_ids=[5, 6],
        response_ids=[7, 8],
        response_text="r",
        response_token_mask=[True, True],
        behavior_policy_id="b",
        behavior_policy_revision="r",
        policy_version=0,
        generation_seed=0,
        sampling_temperature=1.0,
        top_p=1.0,
        behavior_logprobs=[0.0, 0.0],
    )


@pytest.mark.unit
def test_teacher_scoring_uses_prefix_position_not_response_position() -> None:
    scorer = HuggingFaceTeacherScorer(
        PositionalTeacher(), teacher_id="teacher", teacher_revision="rev", top_k=1
    )
    scored = scorer.score(TrajectoryBatch([record()], 0)).records[0]
    # Prompt length is two, so logits[1] predicts response token 0 and logits[2] token 1.
    assert scored.teacher_topk_ids == [[5], [6]]


@pytest.mark.unit
def test_hard_loss_masks_prompt_and_padding() -> None:
    logits = torch.randn(1, 5, 9)
    targets = torch.tensor([[8, 7, 6, 5, 4]])
    mask = torch.tensor([[False, False, True, False, False]])
    first = hard_teacher_loss(logits, targets, mask)
    changed = targets.clone()
    changed[0, 0] = 1
    changed[0, 1] = 2
    changed[0, 4] = 3
    second = hard_teacher_loss(logits, changed, mask)
    assert torch.allclose(first, second)


@pytest.mark.unit
def test_soft_kl_is_zero_for_identical_and_nonnegative() -> None:
    logits = torch.randn(2, 3, 7)
    ids, logs, _, _ = topk_from_logits(logits, 7)
    exact = topk_kl(logs, logits, ids, mode="renormalized")
    assert torch.all(exact.abs() < 1e-6)
    perturbed = topk_kl(logs, logits + torch.randn_like(logits), ids, mode="tail_bucket")
    assert torch.all(perturbed >= -1e-6)


@pytest.mark.unit
def test_topk_kl_converges_to_full_vocabulary_kl() -> None:
    teacher_logits = torch.tensor([[2.0, 1.0, -0.5, -1.0]])
    student_logits = torch.tensor([[0.0, 1.5, -0.25, -2.0]])
    teacher_log = teacher_logits.log_softmax(-1)
    student_log = student_logits.log_softmax(-1)
    full = (teacher_log.exp() * (teacher_log - student_log)).sum(-1)
    ids, logs, _, _ = topk_from_logits(teacher_logits, 4)
    approximation = topk_kl(logs, student_logits, ids, mode="tail_bucket")
    assert torch.allclose(approximation, full, atol=1e-6)


@pytest.mark.unit
def test_soft_loss_shift_and_metrics() -> None:
    logits = torch.randn(1, 4, 6)
    ids, logs, _, _ = topk_from_logits(logits[:, :-1], 3)
    padded_ids = torch.zeros(1, 4, 3, dtype=torch.long)
    padded_logs = torch.full((1, 4, 3), float("-inf"))
    padded_ids[:, 1:] = ids
    padded_logs[:, 1:] = logs
    loss, metrics = soft_teacher_loss(
        logits, padded_ids, padded_logs, torch.tensor([[False, True, True, True]]), mode="renormalized"
    )
    assert loss.abs() < 1e-6
    assert metrics["student_mass_on_teacher_topk"] > 0


@pytest.mark.unit
def test_soft_loss_padding_has_finite_gradients() -> None:
    logits = torch.randn(1, 5, 6, requires_grad=True)
    ids, logs, _, _ = topk_from_logits(logits.detach()[:, :2], 3)
    padded_ids = torch.zeros(1, 5, 3, dtype=torch.long)
    padded_logs = torch.full((1, 5, 3), float("-inf"))
    padded_ids[:, 1:3] = ids
    padded_logs[:, 1:3] = logs
    loss, _ = soft_teacher_loss(
        logits,
        padded_ids,
        padded_logs,
        torch.tensor([[False, True, True, False, False]]),
        mode="renormalized",
    )
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.unit
def test_verified_replay_uses_only_successes_and_sequence_normalizes() -> None:
    logits = torch.zeros(2, 5, 4)
    tokens = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.tensor([[False, True, False, False, False], [False, True, True, True, False]])
    only_first = verified_replay_loss(logits, tokens, mask, torch.tensor([1.0, 0.0]))
    both = verified_replay_loss(logits, tokens, mask, torch.tensor([1.0, 1.0]))
    assert torch.allclose(only_first, torch.tensor(4.0).log())
    assert torch.allclose(only_first, both)


@pytest.mark.unit
def test_verified_replay_zero_positive_fails_cleanly() -> None:
    with pytest.raises(ValueError, match="zero successful"):
        verified_replay_loss(
            torch.zeros(1, 3, 4),
            torch.zeros(1, 3, dtype=torch.long),
            torch.tensor([[False, True, True]]),
            torch.tensor([0.0]),
        )
