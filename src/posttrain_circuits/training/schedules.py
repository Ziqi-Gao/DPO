"""Deterministic prompt and learning-rate schedules."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain_circuits.core.types import PromptBatch


@dataclass
class PromptScheduler:
    prompt_ids: list[str]
    prompt_texts: list[str]
    batch_size: int
    position: int = 0

    def __post_init__(self) -> None:
        if not self.prompt_ids or len(self.prompt_ids) != len(self.prompt_texts):
            raise ValueError("prompt scheduler needs aligned, non-empty prompts")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def next_batch(self) -> PromptBatch:
        indices = [(self.position + offset) % len(self.prompt_ids) for offset in range(self.batch_size)]
        self.position = (self.position + self.batch_size) % len(self.prompt_ids)
        return PromptBatch(
            tuple(self.prompt_ids[index] for index in indices),
            tuple(self.prompt_texts[index] for index in indices),
        )

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.position = state["position"]
