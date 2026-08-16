from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.core.types import PromptBatch
from posttrain_circuits.rollout.generation import hf_generate_trajectories
from posttrain_circuits.teacher.demo_generation import HfTeacherCandidateGenerator
from posttrain_circuits.utils.smoke import build_smoke_examples
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


def _generate(model, tokenizer, seed: int):  # type: ignore[no-untyped-def]
    return hf_generate_trajectories(
        model,
        tokenizer,
        PromptBatch(("short", "long"), ("A", "A B C D")),
        policy_version=3,
        seed=seed,
        max_new_tokens=6,
        temperature=1.0,
        top_p=1.0,
        policy_id="qwen-test",
        policy_revision="checkpoint-test",
    )


@pytest.mark.unit
@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_real_qwen_generation_preserves_prompt_bytes_and_is_deterministic(padding_side: str) -> None:
    tokenizer = build_tiny_tokenizer()
    tokenizer.padding_side = padding_side
    model = build_tiny_qwen(19).eval()
    first = _generate(model, tokenizer, 101)
    second = _generate(model, tokenizer, 101)
    assert [record.response_ids for record in first] == [record.response_ids for record in second]
    for record, text in zip(first, ("A", "A B C D"), strict=True):
        assert record.input_ids == tokenizer(text, add_special_tokens=True)["input_ids"]
        assert len(record.response_ids) == len(record.response_token_mask)
        assert len(record.response_ids) == len(record.behavior_logprobs)
        assert record.generation_seed == 101


@pytest.mark.unit
def test_real_qwen_generation_changes_under_a_different_sampling_seed() -> None:
    tokenizer = build_tiny_tokenizer()
    tokenizer.padding_side = "left"
    model = build_tiny_qwen(23).eval()
    first = _generate(model, tokenizer, 5)
    second = _generate(model, tokenizer, 6)
    assert [record.response_ids for record in first] != [record.response_ids for record in second]


@pytest.mark.unit
def test_generation_keeps_eos_and_removes_post_eos_padding(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_qwen(29).eval()
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)

    def fake_generate(**kwargs):  # type: ignore[no-untyped-def]
        assert "generator" not in kwargs
        prefix = kwargs["input_ids"]
        generated = torch.tensor([[4, eos, pad], [5, 6, eos]], dtype=torch.long)
        sequences = torch.cat((prefix, generated), dim=1)
        scores = tuple(torch.zeros((2, model.config.vocab_size)) for _ in range(3))
        return SimpleNamespace(sequences=sequences, scores=scores)

    monkeypatch.setattr(model, "generate", fake_generate)
    records = _generate(model, tokenizer, 17)
    assert records[0].response_ids == [4, eos]
    assert records[1].response_ids == [5, 6, eos]
    assert all(len(row.response_ids) == len(row.behavior_logprobs) for row in records)


@pytest.mark.unit
def test_teacher_generation_uses_supported_rng_context_with_real_qwen(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_qwen(31).eval()
    original_generate = model.generate

    def checked_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        assert "generator" not in kwargs
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(model, "generate", checked_generate)
    generate = HfTeacherCandidateGenerator(model, tokenizer, max_new_tokens=4)
    example = build_smoke_examples(1, seed=33)[0]
    first = generate(
        example=example,
        candidate_index=0,
        generation_seed=123,
        temperature=1.0,
        top_p=1.0,
    )
    second = generate(
        example=example,
        candidate_index=0,
        generation_seed=123,
        temperature=1.0,
        top_p=1.0,
    )
    assert first == second
