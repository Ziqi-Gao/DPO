from __future__ import annotations

import pytest

from posttrain_circuits.core.types import PromptBatch
from posttrain_circuits.rollout.current_policy import CurrentPolicyStateSource
from posttrain_circuits.utils.smoke import build_smoke_examples, scripted_current_policy_generator


@pytest.mark.unit
def test_current_policy_resume_preserves_refresh_interval_greater_than_one(tokenizer) -> None:  # type: ignore[no-untyped-def]
    examples = build_smoke_examples(2)
    prompts = PromptBatch(
        tuple(example.example_id for example in examples),
        tuple("prompt" for _ in examples),
    )
    generator = scripted_current_policy_generator(examples, tokenizer)
    source = CurrentPolicyStateSource(generator, refresh_interval=3, max_policy_lag=0, seed=17)
    assert source.get_batch(None, prompts, 0).policy_version == 1
    assert source.get_batch(None, prompts, 1).policy_version == 1
    checkpoint_state = source.state_dict()

    expected_before_refresh = source.get_batch(None, prompts, 2)
    expected_after_refresh = source.get_batch(None, prompts, 3)
    assert expected_before_refresh.policy_version == 1
    assert expected_after_refresh.policy_version == 2

    restored = CurrentPolicyStateSource(generator, refresh_interval=3, max_policy_lag=0, seed=17)
    restored.load_state_dict(checkpoint_state)
    actual_before_refresh = restored.get_batch(None, prompts, 2)
    actual_after_refresh = restored.get_batch(None, prompts, 3)
    assert actual_before_refresh.policy_version == expected_before_refresh.policy_version
    assert actual_after_refresh.policy_version == expected_after_refresh.policy_version
    assert [record.trajectory_id for record in actual_after_refresh.records] == [
        record.trajectory_id for record in expected_after_refresh.records
    ]
