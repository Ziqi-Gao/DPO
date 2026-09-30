"""CPU response replay at the real student evaluation boundary; no GPU or Hub."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.learning.training.evaluation import build_proofgraph_evaluator
from posttrain_circuits.models.prompt_protocol import format_model_prompt

ROOT = Path(__file__).resolve().parents[2]
TASK = ProofGraphTask()


@pytest.fixture
def replay_tokenizer():
    """Lossless local byte tokenizer, explicitly not a production tokenizer claim."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {value: index for index, value in enumerate(alphabet)}
    vocab.update({"[PAD]": len(vocab), "[EOS]": len(vocab) + 1})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="[EOS]", pad_token="[PAD]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}"
    )
    return tokenizer


@pytest.fixture
def replay_examples():
    return [
        TASK.generate(42, {"depth": 1, "structure": "chain", "positive": True}),
        TASK.generate(43, {"depth": 1, "structure": "chain", "positive": False}),
    ]


def _model_config(tokenizer):
    return {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        }
    }


class ResponseReplay(torch.nn.Module):
    """Replay raw continuations through actual formatting/decoding/verification."""

    def __init__(self, tokenizer, responses, *, fail=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=32768)
        self.tokenizer = tokenizer
        self.responses = list(responses)
        self.calls = []
        self.fail = fail

    def generate(self, **kwargs):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert kwargs["do_sample"] is False
        assert kwargs["use_cache"] is False
        assert kwargs["pad_token_id"] == self.tokenizer.pad_token_id
        assert kwargs["eos_token_id"] == self.tokenizer.eos_token_id
        assert kwargs["input_ids"].shape[0] == 1
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("replay generation failure")
        response = self.responses[len(self.calls) - 1]
        ids = self.tokenizer.encode(response, add_special_tokens=False)
        ids = [*ids, self.tokenizer.eos_token_id][: kwargs["max_new_tokens"]]
        return torch.cat((kwargs["input_ids"], torch.tensor([ids])), dim=1)


def _old_evaluator(tokenizer, examples, *, cap=128, input_limit=4096):
    return build_proofgraph_evaluator(
        examples,
        tokenizer,
        max_completion_length=cap,
        max_model_input_length=input_limit,
        model_config=_model_config(tokenizer),
    )


@pytest.mark.parametrize("was_training", [True, False])
def test_existing_student_evaluator_replays_canonical_positive_and_negative(
    replay_tokenizer, replay_examples, was_training
):
    model = ResponseReplay(
        replay_tokenizer, [TASK.canonical_target(example) for example in replay_examples]
    )
    model.train(was_training)
    assert _old_evaluator(replay_tokenizer, replay_examples)(model) == {
        "validation_accuracy": 1.0,
        "exact_proof_accuracy": 1.0,
        "format_validity": 1.0,
    }
    assert model.training is was_training
    for example, call in zip(replay_examples, model.calls, strict=True):
        expected = format_model_prompt(
            TASK.render(example), replay_tokenizer, _model_config(replay_tokenizer)
        ).model_facing_prompt
        assert replay_tokenizer.decode(call["input_ids"][0]) == expected
        assert call["max_new_tokens"] == 128


def test_existing_student_evaluator_raw_truncation_reproduces_all_zero(
    replay_tokenizer, replay_examples
):
    model = ResponseReplay(
        replay_tokenizer, [TASK.canonical_target(example) for example in replay_examples]
    )
    assert _old_evaluator(replay_tokenizer, replay_examples, cap=5)(model) == {
        "validation_accuracy": 0.0,
        "exact_proof_accuracy": 0.0,
        "format_validity": 0.0,
    }
    assert model.training is True


def test_existing_student_evaluator_restores_mode_after_generation_failure(
    replay_tokenizer, replay_examples
):
    model = ResponseReplay(replay_tokenizer, [], fail=True)
    with pytest.raises(RuntimeError, match="replay generation failure"):
        _old_evaluator(replay_tokenizer, replay_examples)(model)
    assert model.training is True


def _worker():
    spec = importlib.util.spec_from_file_location(
        "student_quality_probe_test", ROOT / "tools/sdsc_student_quality_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cap", [128, 256])
@pytest.mark.parametrize("was_training", [True, False])
def test_diagnostic_arm_matches_existing_evaluator_and_retains_raw_evidence(
    replay_tokenizer, replay_examples, cap, was_training
):
    responses = [TASK.canonical_target(example) for example in replay_examples]
    old_model = ResponseReplay(replay_tokenizer, responses)
    old = _old_evaluator(replay_tokenizer, replay_examples, cap=cap)(old_model)
    model = ResponseReplay(replay_tokenizer, responses)
    model.train(was_training)
    observed = []
    summary, records = _worker().evaluate_arm(
        model,
        replay_tokenizer,
        replay_examples,
        _model_config(replay_tokenizer),
        cap=cap,
        on_record=observed.append,
    )
    assert {key: summary[key] for key in old} == old
    assert summary["exact_proof_accuracy"] == 1.0
    assert summary["stop_counts"] == {"eos": 2}
    assert model.training is was_training
    assert records == observed
    for example, response, record, old_call, new_call in zip(
        replay_examples, responses, records, old_model.calls, model.calls, strict=True
    ):
        assert record["response_text"] == response
        assert record["response_ids"] == [
            *replay_tokenizer.encode(response, add_special_tokens=False),
            replay_tokenizer.eos_token_id,
        ]
        assert record["verification"] == asdict(
            TASK.verify(example, TASK.parse_response(response))
        )
        assert torch.equal(old_call["input_ids"], new_call["input_ids"])
        for key in ("max_new_tokens", "do_sample", "pad_token_id", "eos_token_id", "use_cache"):
            assert old_call[key] == new_call[key]


@pytest.mark.parametrize(
    "fault", ["prose", "bad_step", "wrong_citation", "wrong_answer", "duplicate_tag", "tag_only"]
)
def test_diagnostic_answer_observation_never_weakens_strict_verification(
    replay_tokenizer, replay_examples, fault
):
    example = replay_examples[0]
    target = TASK.canonical_target(example)
    if fault == "prose":
        target += " extra explanation"
    elif fault == "bad_step":
        target = target.replace("S01:", "INVALID:")
    elif fault == "wrong_citation":
        target = re.sub(r"\(F\d+\)", "(F99)", target, count=1)
    elif fault == "wrong_answer":
        target = target.replace(f"<answer>{example.label}", f"<answer>{1 - example.label}")
    elif fault == "duplicate_tag":
        target += f"<answer>{example.label}</answer>"
    elif fault == "tag_only":
        target = f"<answer>{example.label}</answer>"
    ids = [
        *replay_tokenizer.encode(target, add_special_tokens=False),
        replay_tokenizer.eos_token_id,
    ]
    record = _worker().response_record(
        example, ids, replay_tokenizer, prompt_ids=[1], prompt_text="prompt", cap=256
    )
    expected = TASK.verify(example, TASK.parse_response(target))
    assert record["verification"] == asdict(expected)
    assert record["verification"]["reward"] == 0.0
    assert record["verification"]["answer_correct"] is False
    assert record["diagnostic_answer_tag_correct"] is (fault not in {"wrong_answer", "duplicate_tag"})
    assert record["response_ids"] == ids
    assert record["response_text"] == target


def test_128_truncation_and_256_exploratory_arm_remain_separate(
    replay_tokenizer, replay_examples
):
    example = replay_examples[0]
    target = " " * 128 + TASK.canonical_target(example)
    summaries = {}
    for cap in (128, 256):
        model = ResponseReplay(replay_tokenizer, [target])
        summary, records = _worker().evaluate_arm(
            model, replay_tokenizer, [example], _model_config(replay_tokenizer), cap=cap
        )
        summaries[cap] = summary
        assert records[0]["max_new_tokens"] == cap
        assert records[0]["response_text"] == replay_tokenizer.decode(
            records[0]["response_ids"], skip_special_tokens=True
        )
    assert summaries[128]["exact_proof_accuracy"] == 0.0
    assert summaries[128]["validation_accuracy"] == 0.0
    assert summaries[128]["format_validity"] == 0.0
    assert summaries[128]["stop_counts"] == {"length": 1}
    assert summaries[256]["exact_proof_accuracy"] == 1.0
    assert summaries[256]["stop_counts"] == {"eos": 1}


def test_diagnostic_arm_restores_mode_on_generation_failure(replay_tokenizer, replay_examples):
    model = ResponseReplay(replay_tokenizer, [], fail=True)
    with pytest.raises(RuntimeError, match="replay generation failure"):
        _worker().evaluate_arm(
            model, replay_tokenizer, replay_examples, _model_config(replay_tokenizer), cap=128
        )
    assert model.training is True


@pytest.mark.parametrize("cap", [0, 1, 127, 129, 257])
def test_diagnostic_arm_rejects_unreviewed_cap_before_generation(replay_tokenizer, replay_examples, cap):
    model = ResponseReplay(replay_tokenizer, [], fail=True)
    with pytest.raises(ValueError, match="128.*256"):
        _worker().evaluate_arm(
            model, replay_tokenizer, replay_examples, _model_config(replay_tokenizer), cap=cap
        )
    assert model.calls == []


def test_both_evaluators_clip_to_identical_1536_input_envelope(
    replay_tokenizer, replay_examples, monkeypatch
):
    monkeypatch.setattr(ProofGraphTask, "render", lambda _self, _example: "x" * 1400)
    responses = [TASK.canonical_target(example) for example in replay_examples]
    old_model = ResponseReplay(replay_tokenizer, responses)
    expected = _old_evaluator(replay_tokenizer, replay_examples, input_limit=1536)(old_model)
    new_model = ResponseReplay(replay_tokenizer, responses)
    summary, records = _worker().evaluate_arm(
        new_model, replay_tokenizer, replay_examples, _model_config(replay_tokenizer), cap=128
    )
    assert {key: summary[key] for key in expected} == expected
    for old_call, new_call, record in zip(old_model.calls, new_model.calls, records, strict=True):
        assert 0 < new_call["max_new_tokens"] < 128
        assert old_call["max_new_tokens"] == new_call["max_new_tokens"]
        assert new_call["max_new_tokens"] + new_call["input_ids"].shape[1] == 1536
        assert record["max_new_tokens"] == new_call["max_new_tokens"]


def test_both_evaluators_reject_prompt_beyond_1536_without_generation(
    replay_tokenizer, replay_examples, monkeypatch
):
    monkeypatch.setattr(ProofGraphTask, "render", lambda _self, _example: "x" * 1600)
    old_model = ResponseReplay(replay_tokenizer, [], fail=True)
    new_model = ResponseReplay(replay_tokenizer, [], fail=True)
    with pytest.raises(ValueError, match="model-input envelope"):
        _old_evaluator(replay_tokenizer, replay_examples, input_limit=1536)(old_model)
    with pytest.raises(ValueError, match="model-input envelope"):
        _worker().evaluate_arm(
            new_model, replay_tokenizer, replay_examples, _model_config(replay_tokenizer), cap=128
        )
    assert old_model.calls == new_model.calls == []
    assert old_model.training is new_model.training is True


def test_both_evaluators_clip_to_model_position_limit(replay_tokenizer, replay_examples):
    examples = replay_examples[:1]
    text = format_model_prompt(
        TASK.render(examples[0]), replay_tokenizer, _model_config(replay_tokenizer)
    ).model_facing_prompt
    prompt_length = len(replay_tokenizer.encode(text, add_special_tokens=False))
    response = [TASK.canonical_target(examples[0])]
    old_model = ResponseReplay(replay_tokenizer, response)
    new_model = ResponseReplay(replay_tokenizer, response)
    old_model.config.max_position_embeddings = prompt_length + 10
    new_model.config.max_position_embeddings = prompt_length + 10
    expected = _old_evaluator(replay_tokenizer, examples, input_limit=1536)(old_model)
    summary, records = _worker().evaluate_arm(
        new_model, replay_tokenizer, examples, _model_config(replay_tokenizer), cap=128
    )
    assert {key: summary[key] for key in expected} == expected
    assert old_model.calls[0]["max_new_tokens"] == new_model.calls[0]["max_new_tokens"] == 10
    assert records[0]["stop_reason"] == "length"


def test_eos_at_limit_is_eos_not_length(replay_tokenizer, replay_examples):
    example = replay_examples[0]
    text = TASK.canonical_target(example)
    ids = [
        *replay_tokenizer.encode(text, add_special_tokens=False),
        replay_tokenizer.eos_token_id,
    ]
    record = _worker().response_record(
        example, ids, replay_tokenizer, prompt_ids=[1], prompt_text="p", cap=len(ids)
    )
    assert record["stop_reason"] == "eos"
    assert record["verification"]["reward"] == 1.0
