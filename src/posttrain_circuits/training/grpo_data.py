"""ProofGraph dataset rows and TRL-compatible reward adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from posttrain_circuits.models.prompt_protocol import format_model_prompt
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
    tokenizer: Any | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, str]], Callable[..., list[float]]]:
    task = ProofGraphTask()
    raw_prompts = [task.render(example) for example in examples]
    if tokenizer is None:
        formatted_prompts = raw_prompts
    else:
        formatted_prompts = [
            format_model_prompt(prompt, tokenizer, model_config).model_facing_prompt for prompt in raw_prompts
        ]
    rows = [{"prompt": prompt} for prompt in formatted_prompts]
    examples_by_prompt = dict(zip(formatted_prompts, examples, strict=True))
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
