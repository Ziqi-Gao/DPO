"""Immutable common-bank state source."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from posttrain_circuits.core.types import PromptBatch, TrajectoryBatch, TrajectoryRecord


class FixedBankStateSource:
    def __init__(self, records: list[TrajectoryRecord]) -> None:
        if not records:
            raise ValueError("fixed bank must contain trajectories")
        self._records = tuple(copy.deepcopy(records))
        self._by_prompt: dict[str, list[TrajectoryRecord]] = defaultdict(list)
        for record in self._records:
            self._by_prompt[record.prompt_id].append(record)
        self._cursor: dict[str, int] = defaultdict(int)

    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch:
        del model, step
        selected: list[TrajectoryRecord] = []
        for prompt_id in prompt_batch.prompt_ids:
            choices = self._by_prompt.get(prompt_id)
            if not choices:
                raise KeyError(f"fixed rollout bank has no trajectory for prompt ID {prompt_id!r}")
            index = self._cursor[prompt_id] % len(choices)
            selected.append(copy.deepcopy(choices[index]))
            self._cursor[prompt_id] += 1
        return TrajectoryBatch(selected, policy_version=0)

    def refresh_if_needed(self, model: Any, step: int) -> None:
        del model, step

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "fixed_bank",
            "cursor": dict(self._cursor),
            "immutable_fingerprint": list(self.immutable_fingerprint),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if tuple(state["immutable_fingerprint"]) != self.immutable_fingerprint:
            raise ValueError("fixed-bank checkpoint fingerprint does not match the loaded bank")
        self._cursor = defaultdict(
            int,
            {str(key): int(value) for key, value in state["cursor"].items()},
        )

    @property
    def immutable_fingerprint(self) -> tuple[str, ...]:
        return tuple(record.trajectory_id for record in self._records)
