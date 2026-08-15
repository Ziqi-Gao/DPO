"""Typed records shared by state sources, supervision, and training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class PromptBatch:
    prompt_ids: tuple[str, ...]
    prompt_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.prompt_ids) != len(self.prompt_texts):
            raise ValueError("prompt_ids and prompt_texts must have equal length")


@dataclass
class TrajectoryRecord:
    trajectory_id: str
    prompt_id: str
    split: str
    prompt_text: str
    input_ids: list[int]
    response_ids: list[int]
    response_text: str
    response_token_mask: list[bool]
    behavior_policy_id: str
    behavior_policy_revision: str
    policy_version: int
    generation_seed: int
    sampling_temperature: float
    top_p: float
    behavior_logprobs: list[float]
    verifier_reward: float | None = None
    verification_trace: dict[str, Any] | None = None
    teacher_id: str | None = None
    teacher_revision: str | None = None
    teacher_topk_ids: list[list[int]] = field(default_factory=list)
    teacher_topk_logprobs: list[list[float]] = field(default_factory=list)
    teacher_topk_mass: list[float] = field(default_factory=list)
    teacher_entropy: list[float] = field(default_factory=list)
    created_at: str = ""

    def validate(self) -> None:
        response_length = len(self.response_ids)
        fields = {
            "response_token_mask": len(self.response_token_mask),
            "behavior_logprobs": len(self.behavior_logprobs),
        }
        for name, length in fields.items():
            if length != response_length:
                raise ValueError(f"{name} has length {length}, expected {response_length}")
        aligned_fields: dict[str, Sequence[object]] = {
            "teacher_topk_ids": self.teacher_topk_ids,
            "teacher_topk_logprobs": self.teacher_topk_logprobs,
            "teacher_topk_mass": self.teacher_topk_mass,
            "teacher_entropy": self.teacher_entropy,
        }
        for name, values in aligned_fields.items():
            if values and len(values) != response_length:
                raise ValueError(f"{name} is not aligned to response tokens")


@dataclass
class TrajectoryBatch:
    records: list[TrajectoryRecord]
    policy_version: int

    def validate(self, *, max_policy_lag: int | None = None) -> None:
        for record in self.records:
            record.validate()
            if max_policy_lag is not None and self.policy_version - record.policy_version > max_policy_lag:
                raise ValueError(
                    f"stale trajectory {record.trajectory_id}: policy lag "
                    f"{self.policy_version - record.policy_version} > {max_policy_lag}"
                )


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


class TeacherScorer(Protocol):
    def score(self, trajectories: TrajectoryBatch) -> TrajectoryBatch: ...


class RewardFunction(Protocol):
    def __call__(self, prompts: Sequence[str], completions: Sequence[str], **kwargs: Any) -> list[float]: ...


class StateSource(Protocol):
    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch: ...

    def refresh_if_needed(self, model: Any, step: int) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class Supervisor(Protocol):
    def prepare_targets(
        self,
        trajectories: TrajectoryBatch,
        teacher: TeacherScorer | None,
        verifier: RewardFunction | None,
    ) -> SupervisionBatch: ...

    def compute_loss(self, model: CausalLM, batch: SupervisionBatch) -> LossOutput: ...


@dataclass(frozen=True)
class CounterfactualPair:
    pair_id: str
    clean_example: Any
    corrupt_example: Any
    clean_prompt: str
    corrupt_prompt: str
    clean_target: str
    corrupt_target: str
    corruption_type: str
    changed_semantic_field: str
