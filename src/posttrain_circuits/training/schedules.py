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
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if not self.prompt_ids or len(self.prompt_ids) != len(self.prompt_texts):
            raise ValueError("prompt scheduler needs aligned, non-empty prompts")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed prompt-scheduler rank")

    @classmethod
    def for_distributed_rank(
        cls,
        prompt_ids: list[str],
        prompt_texts: list[str],
        batch_size: int,
        *,
        rank: int,
        world_size: int,
    ) -> PromptScheduler:
        """Build deterministic, disjoint strided prompt shards before training."""

        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed prompt shard")
        if len(prompt_ids) != len(prompt_texts):
            raise ValueError("distributed prompt shard inputs are not aligned")
        shard_ids = prompt_ids[rank::world_size]
        shard_texts = prompt_texts[rank::world_size]
        if not shard_ids:
            raise ValueError("distributed prompt shard is empty")
        return cls(
            shard_ids,
            shard_texts,
            min(batch_size, len(shard_ids)),
            rank=rank,
            world_size=world_size,
        )

    def next_batch(self) -> PromptBatch:
        indices = [(self.position + offset) % len(self.prompt_ids) for offset in range(self.batch_size)]
        self.position = (self.position + self.batch_size) % len(self.prompt_ids)
        return PromptBatch(
            tuple(self.prompt_ids[index] for index in indices),
            tuple(self.prompt_texts[index] for index in indices),
        )

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position, "rank": self.rank, "world_size": self.world_size}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if (
            int(state.get("rank", self.rank)) != self.rank
            or int(state.get("world_size", self.world_size)) != self.world_size
        ):
            raise ValueError("prompt-scheduler checkpoint belongs to a different rank shard")
        self.position = state["position"]
