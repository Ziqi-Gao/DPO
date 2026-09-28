"""Actual candidate protocol parity and explicit supplemental seed provenance."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.teacher_identity import pinned_base_teacher_identity
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.splits import build_split
from posttrain_circuits.datasets.teacher_demos.contracts import TeacherCandidateOutput
from posttrain_circuits.learning.teacher import adapted_candidate_generation as current
from posttrain_circuits.learning.teacher import demo_generation as original
from posttrain_circuits.models.prompt_protocol import FormattedPrompt


class ReversibleFixtureTokenizer:
    def __init__(self):
        self.texts = []

    def encode(self, text, **_kwargs):
        if text not in self.texts:
            self.texts.append(text)
        return [self.texts.index(text)]

    def decode(self, tokens, **_kwargs):
        return self.texts[tokens[0]] if len(tokens) == 1 else "misaligned"


@pytest.fixture
def fixture(monkeypatch):
    identity = replace(
        pinned_base_teacher_identity(), kind="learned_dense_checkpoint", teacher_checkpoint_sha256="a" * 64
    )
    tokenizer = ReversibleFixtureTokenizer()
    loaded = SimpleNamespace(
        model=object(),
        tokenizer=tokenizer,
        teacher_checkpoint_sha256=identity.teacher_checkpoint_sha256,
        tokenizer_hash=identity.tokenizer_fingerprint,
        prompt_protocol=identity.prompt_protocol,
        chat_template_sha256=identity.chat_template_sha256,
    )
    calls = []

    class Generator:
        def __init__(self, _model, _tokenizer, *, max_new_tokens, model_config):
            assert max_new_tokens == 256

        def __call__(self, **kwargs):
            calls.append(kwargs)
            # Include genuine verifier failures; the producer must retain them.
            text = (
                "malformed proof"
                if kwargs["candidate_index"] == 1
                else ProofGraphTask().canonical_target(kwargs["example"])
            )
            return TeacherCandidateOutput(text, tokenizer.encode(text), [-0.5], "available", "eos")

    def formatted(raw, _tokenizer, _config):
        return FormattedPrompt(
            raw,
            raw,
            identity.prompt_protocol,
            False,
            identity.chat_template_sha256,
            sha256_value(raw),
            sha256_value(raw),
        )

    monkeypatch.setattr(current, "HfTeacherCandidateGenerator", Generator)
    monkeypatch.setattr(current, "format_model_prompt", formatted)
    monkeypatch.setattr(original, "format_model_prompt", formatted)
    monkeypatch.setattr(original, "tokenizer_fingerprint", lambda _value: identity.tokenizer_fingerprint)
    examples = build_split(ProofGraphTask(), "train", 8, 42, {"depth": 2, "distractors": 4})
    observer = SimpleNamespace(phase=lambda *_args, **_kwargs: None)
    return identity, loaded, examples, observer, calls, Generator


def test_four_rank_candidates_match_original_algorithm_and_preserve_all_failures(fixture, tmp_path):
    identity, loaded, examples, observer, calls, generator = fixture
    expected = original.generate_teacher_demonstrations(
        examples,
        loaded.tokenizer,
        generator(loaded.model, loaded.tokenizer, max_new_tokens=256, model_config={}),
        original.TeacherDemoGenerationConfig(
            teacher_id="Qwen/Qwen3-8B",
            teacher_revision=identity.base_revision,
            resolved_teacher_commit=identity.base_revision,
            sampling_request_seed=31415,
            temperature=0.7,
            top_p=0.8,
            candidates_per_prompt=8,
            max_prompt_tokens=1246,
            max_new_tokens=256,
            top_k=20,
            min_p=0.0,
        ),
        model_config={},
    )
    original_calls = list(calls)
    calls.clear()
    observed = []
    for rank in range(4):
        observed.extend(
            current.measure_candidate_shard(
                examples,
                loaded,
                {},
                identity,
                tmp_path / str(rank),
                rank=rank,
                world_size=4,
                observe=observer,
            )
        )
    by_id = {row["attempt_id"]: row for row in observed}
    assert len(observed) == 64
    assert sum(not row["accepted"] for row in observed) == 8
    for attempt in expected.attempts:
        row = asdict(attempt)
        row.pop("teacher_id")
        row.pop("teacher_revision")
        row["teacher_identity_sha256"] = identity.sha256
        assert by_id[row["attempt_id"]] == row

    def signature(row):
        return (row["example"].example_id, row["candidate_index"], row["actual_sampling_seed"])

    assert sorted(map(signature, calls)) == sorted(map(signature, original_calls))
    assert not any("teacher_revision" in row or "resolved_teacher_commit" in row for row in observed)


@pytest.mark.parametrize("fault", ["checkpoint", "tokenizer", "duplicate", "rank", "protocol"])
def test_candidate_rejects_identity_and_population_mismatches_before_inference(
    fixture, tmp_path, fault, monkeypatch
):
    identity, loaded, examples, observer, calls, _ = fixture
    rank = 0
    if fault == "checkpoint":
        loaded.teacher_checkpoint_sha256 = "b" * 64
    elif fault == "tokenizer":
        loaded.tokenizer_hash = "b" * 64
    elif fault == "duplicate":
        examples = [examples[0], examples[0]]
    elif fault == "rank":
        rank = 4
    else:
        formatter = current.format_model_prompt
        monkeypatch.setattr(
            current, "format_model_prompt", lambda *a: replace(formatter(*a), enable_thinking=True)
        )
    with pytest.raises(ValueError):
        current.measure_candidate_shard(
            examples,
            loaded,
            {},
            identity,
            tmp_path / "out",
            rank=rank,
            world_size=4,
            observe=observer,
        )
    assert calls == []


@pytest.mark.parametrize("fault", ["missing", "empty", "nonfinite", "positive", "decode", "oversize"])
def test_actual_outputs_cannot_omit_or_fabricate_token_evidence(fault):
    tokenizer = ReversibleFixtureTokenizer()
    output = TeacherCandidateOutput("proof", tokenizer.encode("proof"), [-0.5], "available", "eos")
    if fault == "missing":
        output = replace(output, response_ids=None)
    elif fault == "empty":
        output = replace(output, response_ids=[], token_logprobs=[])
    elif fault == "nonfinite":
        output = replace(output, token_logprobs=[float("nan")])
    elif fault == "positive":
        output = replace(output, token_logprobs=[0.5])
    elif fault == "decode":
        output = replace(output, response_text="another proof")
    else:
        output = replace(output, response_ids=[0] * 257, token_logprobs=[-0.5] * 257)
    with pytest.raises(ValueError):
        current._actual_output(output, tokenizer)


def test_supplemental_seeds_keep_original_source_indices_across_all_four_ranks(
    fixture, tmp_path, monkeypatch
):
    from posttrain_circuits.cli import evaluate_teacher_readiness

    identity, loaded, _examples, observer, calls, _ = fixture
    examples = build_split(ProofGraphTask(), "validation", 256, 42, {"depth": 2, "distractors": 4})[128:256]
    monkeypatch.setattr(
        evaluate_teacher_readiness, "_prefix_scores", lambda *_a, **_kw: pytest.fail("no probes supplied")
    )
    rows = []
    for rank in range(4):
        result = current.measure_readiness_shard(
            examples,
            [],
            loaded,
            {},
            identity,
            tmp_path / str(rank),
            source_index_start=128,
            rank=rank,
            world_size=4,
            observe=observer,
        )
        rows.extend(result["generations"])
    assert len(rows) == len(calls) == 128
    assert sorted(row["actual_sampling_seed"] for row in rows) == list(range(170, 298))
    assert {row["example_id"] for row in rows} == {example.example_id for example in examples}
    assert {row["global_index"] for row in rows} == set(range(128, 256))
    assert all(row["actual_sampling_seed"] == 42 + row["global_index"] for row in rows)
    assert all(call["temperature"] == 0.0 and call["top_p"] == 1.0 and call["top_k"] == 0 for call in calls)


def test_readiness_rejects_wrong_selected_checkpoint_before_any_inference(fixture, tmp_path):
    identity, loaded, examples, observer, calls, _ = fixture
    loaded.teacher_checkpoint_sha256 = "b" * 64
    with pytest.raises(ValueError, match="identity differs"):
        current.measure_readiness_shard(
            examples,
            [],
            loaded,
            {},
            identity,
            tmp_path / "absent",
            source_index_start=0,
            rank=0,
            world_size=4,
            observe=observer,
        )
    assert calls == [] and not (tmp_path / "absent").exists()
