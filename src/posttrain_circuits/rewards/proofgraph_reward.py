"""Exact ProofGraph verifier reward."""

from __future__ import annotations

from collections.abc import Sequence

from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample


class ProofGraphExactReward:
    version = "proofgraph-exact-v1"

    def __init__(self, examples_by_prompt: dict[str, TaskExample]) -> None:
        self.examples_by_prompt = examples_by_prompt
        self.task = ProofGraphTask()

    def __call__(self, prompts: Sequence[str], completions: Sequence[str], **kwargs: object) -> list[float]:
        del kwargs
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have equal length")
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions, strict=True):
            if prompt not in self.examples_by_prompt:
                raise KeyError("exact reward received an unknown prompt")
            parsed = self.task.parse_response(completion)
            rewards.append(self.task.verify(self.examples_by_prompt[prompt], parsed).reward)
        return rewards
