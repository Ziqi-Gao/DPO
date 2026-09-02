"""Portable trajectory-record contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from posttrain_circuits.datasets.trajectories.identities import trajectory_id


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
    sampling_request_seed: int
    actual_sampling_seed: int
    sampling_cursor_id: str
    sampling_protocol_id: str
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
    generation_group_id: str = ""
    generation_group_index: int = 0
    prompt_group_size: int = 1
    created_at: str = ""
    raw_prompt_text: str = ""
    prompt_protocol: str = "legacy_raw_v1"
    enable_thinking: bool = False
    chat_template_sha256: str = "legacy-unrecorded"
    raw_prompt_sha256: str = "legacy-unrecorded"
    model_facing_prompt_sha256: str = "legacy-unrecorded"
    tokenizer_fingerprint: str = "legacy-unrecorded"
    top_k: int = 0
    min_p: float = 0.0

    @property
    def expected_trajectory_id(self) -> str:
        return trajectory_id(
            prompt_id=self.prompt_id,
            split=self.split,
            prompt_text=self.prompt_text,
            raw_prompt_text=self.raw_prompt_text,
            input_ids=self.input_ids,
            prompt_protocol=self.prompt_protocol,
            enable_thinking=self.enable_thinking,
            chat_template_sha256=self.chat_template_sha256,
            raw_prompt_sha256=self.raw_prompt_sha256,
            model_facing_prompt_sha256=self.model_facing_prompt_sha256,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            behavior_policy_id=self.behavior_policy_id,
            behavior_policy_revision=self.behavior_policy_revision,
            policy_version=self.policy_version,
            actual_sampling_seed_value=self.actual_sampling_seed,
            sampling_protocol_id=self.sampling_protocol_id,
            sampling_temperature=self.sampling_temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            response_ids=self.response_ids,
            response_token_mask=self.response_token_mask,
            generation_group_index=self.generation_group_index,
            prompt_group_size=self.prompt_group_size,
        )

    def validate(self) -> None:
        if not self.sampling_cursor_id or not self.sampling_protocol_id:
            raise ValueError("sampling cursor and protocol IDs must be non-empty")
        if type(self.sampling_request_seed) is not int or type(self.actual_sampling_seed) is not int:
            raise ValueError("sampling request and actual seeds must be integers")
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
        teacher_fields_present = [bool(values) for values in aligned_fields.values()]
        if any(teacher_fields_present) and not all(teacher_fields_present):
            raise ValueError("teacher score fields must be present or absent together")
        if self.teacher_topk_ids:
            if not self.teacher_id or not self.teacher_revision:
                raise ValueError("teacher scores require teacher ID and revision")
            for ids, logprobs in zip(
                self.teacher_topk_ids,
                self.teacher_topk_logprobs,
                strict=True,
            ):
                if not ids or len(ids) != len(logprobs):
                    raise ValueError("teacher top-k IDs and logprobs must have equal non-zero width")
        if self.generation_group_id:
            if self.prompt_group_size < 2:
                raise ValueError("grouped trajectories require prompt_group_size >= 2")
            if not 0 <= self.generation_group_index < self.prompt_group_size:
                raise ValueError("generation_group_index is outside prompt_group_size")
        if self.trajectory_id != self.expected_trajectory_id:
            raise ValueError("trajectory_id does not match scientific trajectory content")
