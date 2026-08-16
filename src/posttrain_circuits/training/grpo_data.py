"""ProofGraph dataset rows and TRL-compatible reward adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from posttrain_circuits.rewards.format_only import FormatOnlyReward
from posttrain_circuits.rewards.proofgraph_reward import ProofGraphExactReward
from posttrain_circuits.rewards.random_matched import MatchedRandomReward
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    if isinstance(value, dict) and "content" in value:
        return str(value["content"])
    return str(value)


def build_grpo_rows_and_reward(
    examples: list[TaskExample],
    *,
    reward_name: str,
    seed: int,
    matched_positive_rate: float | None = None,
) -> tuple[list[dict[str, str]], Callable[..., list[float]]]:
    task = ProofGraphTask()
    rows = [{"prompt": task.render(example)} for example in examples]
    examples_by_prompt = {task.render(example): example for example in examples}
    exact = ProofGraphExactReward(examples_by_prompt)
    format_reward = FormatOnlyReward()
    random_reward = MatchedRandomReward(seed, matched_positive_rate)

    def reward(
        prompts: list[Any],
        completions: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        del kwargs
        prompt_texts = [_completion_text(prompt) for prompt in prompts]
        completion_texts = [_completion_text(completion) for completion in completions]
        if reward_name == "exact":
            return exact(prompt_texts, completion_texts)
        if reward_name == "format_only":
            return format_reward(prompt_texts, completion_texts)
        if reward_name == "matched_random":
            if matched_positive_rate is None:
                raise ValueError("matched random GRPO requires a frozen calibration artifact")
            return random_reward(prompt_texts, completion_texts)
        raise ValueError(f"unsupported GRPO reward {reward_name!r}")

    return rows, reward
