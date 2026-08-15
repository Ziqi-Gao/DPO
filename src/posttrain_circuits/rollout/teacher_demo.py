"""Verified teacher-demonstration source for canonical SFT."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from posttrain_circuits.core.types import PromptBatch, TrajectoryBatch, TrajectoryRecord


class TeacherDemoStateSource:
    def __init__(self, demonstrations: list[TrajectoryRecord]) -> None:
        successes = [record for record in demonstrations if record.verifier_reward == 1.0]
        if len(successes) != len(demonstrations) or not successes:
            raise ValueError("teacher demonstrations must be non-empty exact-verifier successes")
        self.demonstrations = tuple(copy.deepcopy(successes))
        self._by_prompt: dict[str, list[TrajectoryRecord]] = defaultdict(list)
        for record in self.demonstrations:
            self._by_prompt[record.prompt_id].append(record)
        self._cursor: dict[str, int] = defaultdict(int)

    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch:
        del model, step
        selected: list[TrajectoryRecord] = []
        for prompt_id in prompt_batch.prompt_ids:
            choices = self._by_prompt.get(prompt_id)
            if not choices:
                raise KeyError(f"teacher-demo store has no verified demonstration for prompt {prompt_id}")
            index = self._cursor[prompt_id] % len(choices)
            selected.append(copy.deepcopy(choices[index]))
            self._cursor[prompt_id] += 1
        return TrajectoryBatch(selected, policy_version=0)

    def refresh_if_needed(self, model: Any, step: int) -> None:
        del model, step

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "teacher_demo",
            "cursor": dict(self._cursor),
            "trajectory_ids": [record.trajectory_id for record in self.demonstrations],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = [record.trajectory_id for record in self.demonstrations]
        if list(state["trajectory_ids"]) != expected:
            raise ValueError("teacher-demo checkpoint does not match the loaded store")
        self._cursor = defaultdict(
            int,
            {str(key): int(value) for key, value in state["cursor"].items()},
        )
