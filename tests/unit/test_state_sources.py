from __future__ import annotations

import copy

import pytest

from posttrain_circuits.core.types import PromptBatch
from posttrain_circuits.rollout.current_policy import CurrentPolicyStateSource
from posttrain_circuits.rollout.fixed_bank import FixedBankStateSource
from posttrain_circuits.training.schedules import PromptScheduler
from posttrain_circuits.utils.smoke import (
    build_fixed_bank,
    build_smoke_examples,
    scripted_current_policy_generator,
)


@pytest.mark.unit
def test_fixed_bank_fingerprint_and_records_do_not_change(tokenizer) -> None:  # type: ignore[no-untyped-def]
    examples = build_smoke_examples(2)
    records = build_fixed_bank(examples, tokenizer, 1)
    source = FixedBankStateSource(records)
    fingerprint = source.immutable_fingerprint
    batch = PromptBatch(tuple(x.example_id for x in examples), tuple("p" for _ in examples))
    selected = source.get_batch(None, batch, 0)
    selected.records[0].response_ids[0] = 999
    assert source.immutable_fingerprint == fingerprint
    assert source.get_batch(None, batch, 1).records[0].response_ids[0] != 999


@pytest.mark.unit
def test_online_source_increments_versions_and_rejects_stale(tokenizer) -> None:  # type: ignore[no-untyped-def]
    examples = build_smoke_examples(2)
    prompts = PromptBatch(tuple(x.example_id for x in examples), tuple("p" for _ in examples))
    source = CurrentPolicyStateSource(
        scripted_current_policy_generator(examples, tokenizer), refresh_interval=1, max_policy_lag=0
    )
    assert source.get_batch(None, prompts, 0).policy_version == 1
    assert source.get_batch(None, prompts, 1).policy_version == 2
    stale_generator = scripted_current_policy_generator(examples, tokenizer)

    def stale(model, prompt_batch, policy_version, seed):  # type: ignore[no-untyped-def]
        values = stale_generator(model, prompt_batch, policy_version, seed)
        for value in values:
            value.policy_version = policy_version - 1
        return values

    rejecting = CurrentPolicyStateSource(stale, max_policy_lag=0)
    with pytest.raises(ValueError, match="stale or mismatched trajectory"):
        rejecting.get_batch(None, prompts, 0)
    manual = source.get_batch(None, prompts, 2)
    manual.records[0].policy_version -= 1
    with pytest.raises(ValueError, match="stale trajectory"):
        manual.validate(max_policy_lag=0)


@pytest.mark.unit
def test_fixed_bank_rejects_unknown_prompt(tokenizer) -> None:  # type: ignore[no-untyped-def]
    records = build_fixed_bank(build_smoke_examples(1), tokenizer, 1)
    source = FixedBankStateSource(records)
    with pytest.raises(KeyError, match="no trajectory for prompt ID"):
        source.get_batch(None, PromptBatch(("missing",), ("missing",)), 0)


@pytest.mark.unit
def test_prompt_schedule_is_deterministic_and_restorable() -> None:
    first = PromptScheduler(["a", "b", "c"], ["A", "B", "C"], 2)
    second = copy.deepcopy(first)
    assert first.next_batch() == second.next_batch()
    state = first.state_dict()
    expected = first.next_batch()
    restored = PromptScheduler(["a", "b", "c"], ["A", "B", "C"], 2)
    restored.load_state_dict(state)
    assert restored.next_batch() == expected
