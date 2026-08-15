from __future__ import annotations

import pytest

from posttrain_circuits.rewards.format_only import FormatOnlyReward
from posttrain_circuits.rewards.proofgraph_reward import ProofGraphExactReward
from posttrain_circuits.rewards.random_matched import MatchedRandomReward
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask


@pytest.mark.unit
def test_exact_and_format_rewards_are_distinct() -> None:
    task = ProofGraphTask()
    example = task.generate(2, {"positive": True})
    prompt = task.render(example)
    correct = task.canonical_target(example)
    wrong = correct.replace("<answer>1", "<answer>0")
    exact = ProofGraphExactReward({prompt: example})
    assert exact([prompt, prompt], [correct, wrong]) == [1.0, 0.0]
    # Format-only sees no examples and therefore cannot leak semantic correctness.
    assert FormatOnlyReward()([prompt, prompt], [correct, wrong]) == [1.0, 1.0]


@pytest.mark.unit
def test_matched_random_reward_is_deterministic_and_rate_matched() -> None:
    prompts = [f"p{x}" for x in range(10)]
    completions = [f"c{x}" for x in range(10)]
    reward = MatchedRandomReward(7, positive_rate=0.3)
    first = reward(prompts, completions)
    second = reward(prompts, completions)
    assert first == second
    assert sum(first) == 3
