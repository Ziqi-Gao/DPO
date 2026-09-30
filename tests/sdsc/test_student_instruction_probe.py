"""Real CPU fixtures for the single training-side instruction hypothesis."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from posttrain_circuits.datasets.proofgraph import rendering
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.models import loading
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location(
        "instruction_probe_test", ROOT / "tools/sdsc_student_instruction_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example():
    return ProofGraphTask().generate(81, {"depth": 1, "distractors": 0, "structure": "chain"})


@pytest.fixture
def chat():
    tokenizer = build_tiny_tokenizer()
    # Exercise the actual Qwen3 non-thinking formatter with a local fixture template.
    tokenizer.chat_template = (
        "{{ '<|im_start|>user\\n' + messages[0]['content'] + '<|im_end|>\\n"
        "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    )
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        }
    }
    return tokenizer, config


def test_frozen_candidate_and_exception_restore(probe):
    original = rendering.RESPONSE_FORMAT_INSTRUCTIONS
    probe.candidate_identity()
    assert (
        hashlib.sha256(probe.CANDIDATE.encode()).hexdigest()
        == "8c44dc8a1bc86167e3787cdcd84e20537db69585e3d7ee30b920447072f03c1e"
    )
    with pytest.raises(RuntimeError, match="interrupted"), probe.instruction("candidate"):
        assert rendering.RESPONSE_FORMAT_INSTRUCTIONS == probe.CANDIDATE
        raise RuntimeError("interrupted")
    assert original == rendering.RESPONSE_FORMAT_INSTRUCTIONS


def test_changed_candidate_fails_identity(probe, monkeypatch):
    monkeypatch.setattr(probe, "CANDIDATE", probe.CANDIDATE + "\nGuess the answer.")
    with pytest.raises(ValueError, match="identity"):
        probe.candidate_identity()


def test_prompt_has_no_label_id_metadata_or_target_access(probe, example, chat):
    class GraphOnly:
        facts, rules, query = example.facts, example.rules, example.query

        def __getattr__(self, name):
            raise AssertionError("forbidden prompt input: " + name)

    tokenizer, config = chat
    untouched = asdict(example)
    only_graph = probe.prepare_prompts([GraphOnly()], tokenizer, config, max_positions=1536)
    original = probe.prepare_prompts([example], tokenizer, config, max_positions=1536)
    changed = copy.deepcopy(example)
    changed.label = 1 - changed.label
    changed.example_id = "do-not-read"
    changed.pair_group_id = "do-not-read-pair"
    changed.metadata = {"answer": 1, "proof": "do-not-read"}
    changed.canonical_proof = []
    assert probe.prepare_prompts([changed], tokenizer, config, max_positions=1536) == original == only_graph
    assert asdict(example) == untouched
    pair = original[0]
    assert pair["baseline"]["graph_text_sha256"] == pair["candidate"]["graph_text_sha256"]
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in pair["candidate"]["prompt_text"]


def test_complete_256_allowance_is_prechecked_without_truncation(probe, example, chat):
    tokenizer, config = chat
    rows = probe.prepare_prompts([example], tokenizer, config, max_positions=1536)
    longest = max(len(rows[0][arm]["prompt_ids"]) for arm in ("baseline", "candidate"))
    probe.prepare_prompts([example], tokenizer, config, max_positions=longest + 256)
    with pytest.raises(ValueError, match="complete 256-token"):
        probe.prepare_prompts([example], tokenizer, config, max_positions=longest + 255)


@pytest.mark.parametrize("tied", [False, True])
def test_pristine_exact_tiny_qwen_and_tying(probe, tied):
    model = build_tiny_qwen3(21).to(torch.bfloat16).eval()
    if tied:
        model.config.tie_word_embeddings = True
        model.tie_weights()
    model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
    state = copy.deepcopy(model.state_dict())
    before = probe.parameter_digest(model)
    result = probe.compare_pristine_then_load(model, state)
    assert result["pristine_comparison"]["all_tensors_exact"]
    assert result["pristine_comparison"]["key_count"] == len(state)
    assert result["embedding_tying_before"]["same_parameter_object"] is tied
    assert result["embedding_tying_after"] == result["embedding_tying_before"]
    assert probe.parameter_digest(model) == before


@pytest.mark.parametrize("defect", ["value", "key", "shape", "dtype", "counter", "nan"])
def test_pristine_mismatch_fails_before_any_load_or_mutation(probe, monkeypatch, defect):
    model = build_tiny_qwen3(22).to(torch.bfloat16).eval()
    model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
    state = copy.deepcopy(model.state_dict())
    key = "model.embed_tokens.weight"
    if defect == "value":
        state[key].flatten()[0] += 1
    elif defect == "key":
        state.pop(key)
    elif defect == "shape":
        state[key] = state[key][:-1]
    elif defect == "dtype":
        state[key] = state[key].float()
    elif defect == "counter":
        state["counter"] = state["counter"].int()
    else:
        state[key].flatten()[0] = float("nan")
    before = probe.parameter_digest(model)
    calls = []
    monkeypatch.setattr(model, "load_state_dict", lambda *args, **kwargs: calls.append(1))
    with pytest.raises(ValueError, match="pristine/initial"):
        probe.compare_pristine_then_load(model, state)
    assert calls == []
    assert probe.parameter_digest(model) == before


def test_real_guarded_loader_pristine_comparison_before_checkpoint_copy(probe, tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "configs/model/qwen3_v2_1p7b.yaml").read_text())
    original_config = copy.deepcopy(config)
    base = build_tiny_qwen3(24).to(torch.bfloat16)
    checkpoint = tmp_path / "initial.pt"
    torch.save({"model": copy.deepcopy(base.state_dict())}, checkpoint)
    calls = []

    def model_factory(_name, **kwargs):
        calls.append(kwargs)
        return copy.deepcopy(base)

    monkeypatch.setattr(loading.AutoModelForCausalLM, "from_pretrained", model_factory)
    monkeypatch.setattr(
        loading.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: build_tiny_tokenizer()
    )
    monkeypatch.setattr(
        loading, "_validate_tokenizer_protocol", lambda *args: ("fixture", "qwen3_non_thinking_v1")
    )
    loaded, identity = probe.load_pristine_initial(config, checkpoint)
    assert identity["comparison_performed_before_load"]
    assert identity["pristine_comparison"]["all_tensors_exact"]
    assert calls[0]["torch_dtype"] == torch.bfloat16
    assert calls[0]["attn_implementation"] == "sdpa"
    assert config == original_config
    assert not loaded.model.training
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    with pytest.raises(ValueError, match="BF16 and SDPA"):
        probe.load_pristine_initial(dict(config, torch_dtype="float32"), checkpoint)
    assert len(calls) == 1


def test_real_bf16_cpu_generation_and_nll_leave_weights_and_instructions_unchanged(probe, example, chat):
    tokenizer, config = chat
    model = build_tiny_qwen3(23).to(torch.bfloat16).eval()
    model.config.max_position_embeddings = 1536
    prompts = probe.prepare_prompts([example], tokenizer, config, max_positions=1536)
    before, original = probe.parameter_digest(model), rendering.RESPONSE_FORMAT_INSTRUCTIONS
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for arm in ("baseline", "candidate"):
            records = []
            with probe.instruction(arm):
                actual = probe.Q.teacher_forced_metrics(model, tokenizer, [example], config)
            ids = prompts[0][arm]["prompt_ids"]
            target = [
                *tokenizer.encode(rendering.render_target(example), add_special_tokens=False),
                tokenizer.eos_token_id,
            ]
            input_ids = torch.tensor([ids + target])
            labels = torch.full_like(input_ids, -100)
            labels[:, len(ids) :] = input_ids[:, len(ids) :]
            with torch.no_grad():
                expected = model(input_ids=input_ids, labels=labels).loss
            assert actual["canonical_response_nll"] == pytest.approx(float(expected), rel=2e-7)
            assert not torch.is_autocast_enabled("cpu")
            metrics = probe.generate_arm(model, tokenizer, [example], prompts, arm, on_record=records.append)
            assert len(records) == 1 and metrics["count"] == 1
            assert records[0]["max_new_tokens"] == 256
            assert records[0]["prompt_ids"] == ids
            assert records[0]["prompt_text"] == prompts[0][arm]["prompt_text"]
            assert records[0]["parsed_trace"] == asdict(
                ProofGraphTask().parse_response(records[0]["response_text"])
            )
        assert probe.parameter_digest(model) == before
        assert all(parameter.grad is None for parameter in model.parameters())
        assert original == rendering.RESPONSE_FORMAT_INSTRUCTIONS
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)


def test_candidate_does_not_relax_actual_parser_or_verifier(probe, example):
    task = ProofGraphTask()
    valid = rendering.render_target(example)
    assert task.verify(example, task.parse_response(valid)).reward == 1
    malformed = valid.replace("S01:", "S01", 1)
    wrong_citation = valid.replace("(F", "(UNKNOWN", 1)
    with probe.instruction("candidate"):
        for response in (malformed, wrong_citation, "Answer: 1"):
            result = task.verify(example, task.parse_response(response))
            assert result.reward == 0


def test_train_prefix_reader_does_not_open_other_splits_or_decode_row33(
    probe, tmp_path, example, monkeypatch
):
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    rows = []
    for i in range(32):
        row = asdict(example)
        row["example_id"] = f"train-{i}"
        rows.append(json.dumps(row))
    (root / "train/examples.jsonl").write_text("\n".join(rows) + "\nTHIS ROW MUST NOT BE DECODED\n")
    train_sha = probe.Q.file_sha(root / "train/examples.jsonl")
    (root / "manifest.json").write_text(
        json.dumps({"splits": {"train": {"examples_file_sha256": train_sha}}})
    )
    monkeypatch.setattr(probe, "TRAIN_SHA", train_sha)
    monkeypatch.setattr(probe, "FAMILY_SHA", probe.Q.file_sha(root / "manifest.json"))
    assert len(probe.training_examples(root, tmp_path)) == 32


@pytest.mark.parametrize(
    "candidate,baseline,expected",
    [
        ((4, 4), (0, 0), True),
        ((4, 3), (0, 0), False),
        ((4, 4), (4, 0), False),
        ((4, 4), (0, 4), False),
        ((3, 3), (0, 0), False),
    ],
)
def test_frozen_training_viability_is_separate_from_acceptance(probe, candidate, baseline, expected):
    arms = [
        {"arm": arm, "metrics": {"answer_accuracy": scores[0] / 32, "exact_proof_accuracy": scores[1] / 32}}
        for arm, scores in (("baseline", baseline), ("candidate", candidate))
    ]
    assert probe.training_screen_viable(arms) is expected
    assert all(value is False for value in probe.FALSE_CLAIMS.values())


def test_failed_main_preserves_progress_nll_raw_and_original_error(probe, tmp_path, monkeypatch):
    science = tmp_path / "science"
    (science / "src").mkdir(parents=True)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"science_root": str(science)}))
    output = tmp_path / "output"
    partial = {"schema": probe.SCHEMA, "arms": [{"arm": "baseline", "teacher_forced": {"nll": 1.2}}]}

    def failed(_inputs, directory):
        probe.Q.atomic_json(directory / "progress.json", partial)
        (directory / "instruction-records.jsonl").write_text('{"partial":true}\n')
        raise ValueError("candidate generation failed")

    monkeypatch.setattr(probe, "execute", failed)
    assert probe.main(["--inputs-json", str(inputs), "--output-dir", str(output)]) == 1
    report = json.loads((output / "instruction-probe.json").read_text())
    assert report["arms"] == partial["arms"]
    assert report["error"] == "candidate generation failed"
    assert not report["passed"] and not report["training_screen_viable"]
    assert report["raw_artifacts"][0]["sha256"] == probe.Q.file_sha(output / "instruction-records.jsonl")
    assert all(report[key] is False for key in probe.FALSE_CLAIMS)


@pytest.mark.parametrize("partial_raw", [False, True])
def test_early_pristine_failure_is_reported_without_both_raw_files(probe, tmp_path, monkeypatch, partial_raw):
    science = tmp_path / "science"
    (science / "src").mkdir(parents=True)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"science_root": str(science)}))
    output = tmp_path / "output"

    def failed(_inputs, directory):
        if partial_raw:
            (directory / "instruction-prompts.jsonl").write_text('{"partial":true}\n')
        model = build_tiny_qwen3(26).to(torch.bfloat16)
        altered = copy.deepcopy(model.state_dict())
        altered["model.embed_tokens.weight"].flatten()[0] += 1
        probe.compare_pristine_then_load(model, altered)

    monkeypatch.setattr(probe, "execute", failed)
    assert probe.main(["--inputs-json", str(inputs), "--output-dir", str(output)]) == 1
    report = json.loads((output / "instruction-probe.json").read_text())
    assert "pristine/initial checkpoint tensor differs" in report["error"]
    assert len(report["raw_artifacts"]) == int(partial_raw)
    assert not report["passed"] and not report["diagnostic_complete"]
    assert all(report[key] is False for key in probe.FALSE_CLAIMS)


def test_success_raw_inventory_requires_both_files(probe, tmp_path):
    (tmp_path / "instruction-prompts.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="instruction-records"):
        probe.raw_artifacts(tmp_path, required=True)


def test_failed_raw_read_back_does_not_hide_original_error(probe, tmp_path, monkeypatch):
    science = tmp_path / "science"
    (science / "src").mkdir(parents=True)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"science_root": str(science)}))
    output = tmp_path / "output"

    def failed(_inputs, directory):
        (directory / "instruction-records.jsonl").write_text("{}\n")
        raise ValueError("original inference failure")

    def cannot_read(_path):
        raise OSError("raw read-back unavailable")

    monkeypatch.setattr(probe, "execute", failed)
    monkeypatch.setattr(probe.Q, "file_sha", cannot_read)
    assert probe.main(["--inputs-json", str(inputs), "--output-dir", str(output)]) == 1
    report = json.loads((output / "instruction-probe.json").read_text())
    assert report["error"] == "original inference failure"
    assert report["raw_artifact_error"] == "raw read-back unavailable"
