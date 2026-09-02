"""Portable contracts shared by learning workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.datasets.trajectories.identities import (
    actual_sampling_seed,
    sampling_cursor_id,
)


@dataclass(frozen=True)
class PromptBatch:
    prompt_ids: tuple[str, ...]
    prompt_texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.prompt_ids) != len(self.prompt_texts):
            raise ValueError("prompt_ids and prompt_texts must have equal length")


@dataclass(frozen=True)
class SamplingCursor:
    optimizer_step: int
    retry_index: int
    prompt_id: str
    generation_index: int

    def __post_init__(self) -> None:
        sampling_cursor_id(
            optimizer_step=self.optimizer_step,
            retry_index=self.retry_index,
            prompt_id=self.prompt_id,
            generation_index=self.generation_index,
        )

    @property
    def cursor_id(self) -> str:
        return sampling_cursor_id(
            optimizer_step=self.optimizer_step,
            retry_index=self.retry_index,
            prompt_id=self.prompt_id,
            generation_index=self.generation_index,
        )


@dataclass(frozen=True)
class SamplingRequest:
    sampling_request_seed: int
    sampling_protocol_id: str
    cursors: tuple[SamplingCursor, ...]

    def __post_init__(self) -> None:
        if not self.sampling_protocol_id:
            raise ValueError("sampling_protocol_id must be non-empty")
        cursor_ids = tuple(cursor.cursor_id for cursor in self.cursors)
        if len(cursor_ids) != len(set(cursor_ids)):
            raise ValueError("sampling request contains duplicate cursors")

    def validate_for(self, prompts: PromptBatch) -> None:
        if len(self.cursors) != len(prompts.prompt_ids):
            raise ValueError("sampling request cursor count must equal prompt batch length")
        for prompt_id, cursor in zip(prompts.prompt_ids, self.cursors, strict=True):
            if cursor.prompt_id != prompt_id:
                raise ValueError("sampling cursor prompt_id does not match its prompt row")

    def seed_for(self, index: int, *, policy_version: int) -> int:
        cursor = self.cursors[index]
        return actual_sampling_seed(
            sampling_request_seed=self.sampling_request_seed,
            policy_version=policy_version,
            sampling_cursor_id_value=cursor.cursor_id,
            sampling_protocol_id=self.sampling_protocol_id,
        )


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


class StateSource(Protocol):
    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch: ...

    def refresh_if_needed(self, model: Any, step: int) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


TrajectoryGenerator = Callable[
    [object, PromptBatch, int, SamplingRequest],
    list[TrajectoryRecord],
]
