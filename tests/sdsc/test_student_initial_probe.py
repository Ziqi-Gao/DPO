"""Native initial CPU contracts; no real model, GPU, or scientific acceptance."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.models import loading
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location("test_" + name, ROOT / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def probe():
    return module("sdsc_student_initial_probe")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


@pytest.fixture
def examples():
    return [
        ProofGraphTask().generate(seed, {"depth": 1, "distractors": 0, "structure": "chain"})
        for seed in (81, 82)
    ]


class Characters:
    """Lossless local fixture tokens make actual original parsing observable."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) + 2 for char in text]

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert return_tensors == "pt"
        return SimpleNamespace(
            input_ids=torch.tensor([self.encode(text, add_special_tokens=add_special_tokens)])
        )

    def decode(self, values, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "".join(chr(int(value) - 2) for value in values if int(value) > 1)


class FixedResponses(torch.nn.Module):
    """CPU producer fixture only; records actual original scorer kwargs and context."""

    def __init__(self, texts, tokenizer):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)
        self.texts, self.tokenizer, self.calls, self.returned = texts, tokenizer, [], []

    def generate(self, *args, **kwargs):
        assert not args
        assert not torch.is_grad_enabled() and not self.training
        assert not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda")
        text = self.texts[len(self.calls)]
        self.calls.append(kwargs)
        output = torch.cat(
            (
                kwargs["input_ids"],
                torch.tensor(
                    [[*self.tokenizer.encode(text, add_special_tokens=False), self.tokenizer.eos_token_id]]
                ),
            ),
            dim=1,
        )
        self.returned.append(output)
        return output


def test_original_scorer_exact_kwargs_fullproof_and_answer_are_distinct(probe, examples):
    tokenizer = Characters()
    texts = [render_target(examples[0]), f"<proof></proof><answer>{examples[1].label}</answer>"]
    model = FixedResponses(texts, tokenizer)
    prompts = probe.prepare_prompts(examples, tokenizer, {}, max_positions=2048)
    records = []
    metrics, count, result = probe.score_original(model, tokenizer, examples, prompts, {}, records.append)
    assert result == records and count == 2
    assert metrics == dict(format_validity=1.0, exact_proof_accuracy=0.5, answer_accuracy=1.0)
    assert "generate" not in model.__dict__
    for index, call in enumerate(model.calls):
        assert set(call) == {
            "input_ids",
            "max_new_tokens",
            "do_sample",
            "pad_token_id",
            "eos_token_id",
            "use_cache",
        }
        assert {key: value for key, value in call.items() if key != "input_ids"} == dict(
            max_new_tokens=256, do_sample=False, pad_token_id=0, eos_token_id=1, use_cache=False
        )
        assert call["input_ids"][0].tolist() == prompts[index]["prompt_ids"]
        assert records[index]["response_text"] == texts[index]
        assert records[index]["verification"] == asdict(
            ProofGraphTask().verify(examples[index], ProofGraphTask().parse_response(texts[index]))
        )
    assert records[1]["verification"]["reward"] == 0
    assert records[1]["verification"]["answer_correct"] is True
    assert records[1]["parsed_trace"]["parse_valid"] is True


@pytest.mark.parametrize("text", ["<answer>1</answer>", "<proof>bad line</proof><answer>1</answer>"])
def test_tag_or_malformed_proof_does_not_replace_original_answer_metric(probe, examples, text):
    tokenizer = Characters()
    model = FixedResponses([text], tokenizer)
    prompts = probe.prepare_prompts(examples[:1], tokenizer, {}, max_positions=2048)
    metrics, count, records = probe.score_original(
        model, tokenizer, examples[:1], prompts, {}, lambda row: None
    )
    assert count == 0 and metrics["answer_accuracy"] == metrics["exact_proof_accuracy"] == 0
    assert records[0]["verification"]["parse_valid"] is False


@pytest.mark.parametrize("correct, expected", [(0, False), (12, False), (13, True), (128, True)])
def test_exact_original_threshold(probe, correct, expected):
    assert probe.base_gate(correct, 128) is expected


@pytest.mark.parametrize("correct,total", [(True, 128), (13.0, 128), (13, 127), (-1, 128), (129, 128)])
def test_incomplete_or_type_confused_threshold_rejected(probe, correct, total):
    with pytest.raises(ValueError, match="all 128"):
        probe.base_gate(correct, total)


def test_prompt_uses_original_graph_not_label_id_or_target_and_never_truncates(probe, examples):
    tokenizer = Characters()
    original = examples[0]
    changed = replace(
        original, label=1 - original.label, example_id="untrusted label-like ID", canonical_proof=[]
    )
    before = asdict(original)
    pair = probe.prepare_prompts([original, changed], tokenizer, {}, max_positions=2048)
    assert pair[0]["prompt_text"] == pair[1]["prompt_text"]
    assert pair[0]["prompt_ids"] == pair[1]["prompt_ids"]
    assert asdict(original) == before
    with pytest.raises(ValueError, match="full original 256"):
        probe.prepare_prompts([original], tokenizer, {}, max_positions=len(pair[0]["prompt_ids"]) + 255)


def test_generate_returns_identical_tensor_and_restores_local_override_on_error(probe, examples):
    tokenizer = Characters()
    model = FixedResponses([render_target(examples[0])], tokenizer).eval()
    original = model.generate
    model.generate = original
    prompts = probe.prepare_prompts(examples[:1], tokenizer, {}, max_positions=2048)
    with torch.no_grad(), probe.recording_generate(model, tokenizer, examples[:1], prompts, lambda r: None):
        output = model.generate(
            input_ids=torch.tensor([prompts[0]["prompt_ids"]]),
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=1,
            use_cache=False,
        )
        assert output is model.returned[0]
    assert model.generate is original
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        torch.no_grad(),
        probe.recording_generate(
            model,
            tokenizer,
            examples[:1],
            prompts,
            lambda row: (_ for _ in ()).throw(RuntimeError("capture failed")),
        ),
    ):
        model.calls.clear()
        model.generate(
            input_ids=torch.tensor([prompts[0]["prompt_ids"]]),
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=1,
            use_cache=False,
        )
    assert model.generate is original


def test_recorder_rejects_extra_attention_mask_and_autocast(probe, examples):
    tokenizer = Characters()
    model = FixedResponses([render_target(examples[0])], tokenizer).eval()
    prompts = probe.prepare_prompts(examples[:1], tokenizer, {}, max_positions=2048)
    kwargs = dict(
        input_ids=torch.tensor([prompts[0]["prompt_ids"]]),
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False,
    )
    with (
        pytest.raises(ValueError, match="arguments differ"),
        torch.no_grad(),
        probe.recording_generate(model, tokenizer, examples[:1], prompts, lambda row: None),
    ):
        model.generate(**kwargs, attention_mask=torch.ones_like(kwargs["input_ids"]))
    with (
        pytest.raises(ValueError, match="autocast prohibited"),
        torch.no_grad(),
        torch.autocast("cpu"),
        probe.recording_generate(model, tokenizer, examples[:1], prompts, lambda row: None),
    ):
        model.generate(**kwargs)
    assert model.calls == [] and "generate" not in model.__dict__


def test_actual_bf16_tiny_qwen_original_generate_under_recording(probe, examples):
    tokenizer = build_tiny_tokenizer()
    model = build_tiny_qwen3(64).to(torch.bfloat16).eval()
    # A fixture-only logits processor makes this actual tiny HF generation stop at EOS.
    model.generation_config.suppress_tokens = [i for i in range(model.config.vocab_size) if i != 3]
    prompts = probe.prepare_prompts(examples[:1], tokenizer, {}, max_positions=2048)
    state = copy.deepcopy(model.state_dict())
    metrics, correct, records = probe.score_original(
        model, tokenizer, examples[:1], prompts, {}, lambda row: None
    )
    assert records[0]["response_ids"] == [3]
    assert correct == 0 and metrics["answer_accuracy"] == 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert "generate" not in model.__dict__


def test_actual_guard_and_checkpoint_loader_stay_native_bf16_without_config_mutation(
    probe, monkeypatch, tmp_path
):
    config = yaml.safe_load((ROOT / "configs/model/qwen3_v2_1p7b.yaml").read_text())
    before = copy.deepcopy(config)
    model = build_tiny_qwen3(45).to(torch.bfloat16)
    state = copy.deepcopy(model.state_dict())
    checkpoint = tmp_path / "initial.pt"
    torch.save({"model": state}, checkpoint)
    calls = []

    def local_model(_name, **kwargs):
        calls.append(kwargs)
        return build_tiny_qwen3(99).to(kwargs["torch_dtype"])

    monkeypatch.setattr(loading.AutoModelForCausalLM, "from_pretrained", local_model)
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *a, **kw: build_tiny_tokenizer())
    monkeypatch.setattr(
        loading, "_validate_tokenizer_protocol", lambda tok, cfg: ("fixture", "qwen3_non_thinking_v1")
    )
    loaded = probe.load_original_initial(config, checkpoint, expected_sha256=probe.file_sha(checkpoint))
    assert config == before
    assert calls[0]["torch_dtype"] == torch.bfloat16 and calls[0]["attn_implementation"] == "sdpa"
    assert all(torch.equal(value, loaded.model.state_dict()[name]) for name, value in state.items())
    assert all(
        parameter.dtype == torch.bfloat16 and not parameter.requires_grad
        for parameter in loaded.model.parameters()
    )
    with pytest.raises(ValueError, match="BF16 and SDPA"):
        probe.load_original_initial(
            {**config, "torch_dtype": "float32"}, checkpoint, expected_sha256=probe.file_sha(checkpoint)
        )
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        probe.load_original_initial(config, checkpoint, expected_sha256="0" * 64)


@pytest.mark.parametrize("partial", [0, 1, 2])
def test_cli_early_failure_preserves_existing_progress_and_false_flags(probe, tmp_path, partial):
    output = tmp_path / "artifacts"
    output.mkdir()
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    if partial:
        probe.atomic(
            output / "initial-probe.json",
            dict(raw_record_count=7, metrics={"fixture": 7}, **dict.fromkeys(probe.FLAGS, True)),
        )
        for name in probe.RAW_NAMES[:partial]:
            (output / name).write_text('{"fixture":true}\n')
    assert probe.main(["--inputs-json", str(inputs), "--output-dir", str(output)]) == 1
    report = probe.document(output / "initial-probe.json")
    assert report["passed"] is report["diagnostic_complete"] is report["initial_base_gate_satisfied"] is False
    assert all(report[key] is False for key in probe.FLAGS)
    assert report["error"]["message"] == "unexpected input fields"
    assert len(report["raw_artifacts"]) == partial
    if partial:
        assert report["raw_record_count"] == report["metrics"]["fixture"] == 7


def test_exact_progress_contract(probe, tmp_path):
    probe.progress(tmp_path, "scoring", completed=16)
    result = probe.document(tmp_path / "progress.json")
    assert result["schema"] == "quest-sdsc-student-initial-progress-v1"
    assert result["phase"] == "initial_native_scoring"
    assert result["completed_examples"] == 16 and result["total_examples"] == 128
    assert all(result[key] is False for key in probe.FLAGS)


def test_real_node_inputs_scientific_cwd_and_accepted_history(tmp_path):
    # Reuse only the existing CPU fixture's raw Git restore, not any resolver mock.
    spec = importlib.util.spec_from_file_location(
        "lr_cpu_fixture", ROOT / "tests/sdsc/test_student_lr_probe.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    science = tmp_path / "science"
    fixture.restore_genuine_parent_for_cpu_fixture(science)
    probe = module("sdsc_student_initial_probe")
    node = module("sdsc_student_initial_job")
    cache, output = tmp_path / "huggingface", tmp_path / "artifacts"
    cache.mkdir()
    output.mkdir()
    plan = dict(
        parent={"receipt": {"job_id": "54548846"}},
        run_id="fixture-run",
        code_sha256="a" * 64,
        checkpoints=[dict(size=probe.INITIAL_SIZE)],
        dataset_inputs=[dict(path="manifest.json", sha256=probe.FAMILY_SHA)],
    )
    control = SimpleNamespace(
        PARENT_JOB="54548846", INITIAL_SHA=probe.INITIAL_SHA, sha=probe.digest, canonical=probe.canonical
    )
    inputs = node.build_worker_inputs(
        plan, tmp_path, science, {"job_id": "123456"}, dict(size=7923, sha256=probe.CONFIG_SHA), control
    )
    path = tmp_path / "inputs.json"
    probe.atomic(path, inputs)
    script = """
import importlib.util,os,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('probe',sys.argv[1]);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
inputs=p.document(Path(sys.argv[2]));os.environ['SLURM_JOB_ID']=inputs['job_id'];os.environ['HF_HOME']=inputs['hf_home']
work,sources=p.staged_science(inputs,Path(sys.argv[3]))
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.artifacts.adapted_student_protocol import validate_student_protocol,PROTOCOL_PATH
config=compose_config(['g0=qwen3_v2_eap_separation','experiment=canonical_sft','adapted_teacher=qwen3_accepted_student_v6',f'protocol_amendment_path={PROTOCOL_PATH}'],config_root=Path('configs'))
binding=validate_student_protocol(config)
assert binding.head==p.PARENT_HEAD and len(binding.science_file_sha256)==49
print('real_source_and_protocol_passed')
"""
    command = [
        sys.executable,
        "-B",
        "-c",
        script,
        str(ROOT / "tools/sdsc_student_initial_probe.py"),
        str(path),
        str(output),
    ]
    result = subprocess.run(command, cwd=science, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "real_source_and_protocol_passed"
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "cwd must equal verified science" in result.stderr
    damaged = science / "src/posttrain_circuits/models/loading.py"
    damaged.write_bytes(damaged.read_bytes() + b"\n# fixture source tamper\n")
    result = subprocess.run(command, cwd=science, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "restored source differs" in result.stderr
    assert "real_source_and_protocol_passed" not in result.stdout


def execution_fixture(probe, monkeypatch, examples, tmp_path, *, answer_count, stop_after=None):
    """Synthetic CPU producer, exercising the real 128-row original scorer/report path."""
    cohort = [replace(examples[0], example_id=f"synthetic-native-{index:03d}") for index in range(128)]
    texts = [
        f"<proof></proof><answer>{cohort[index].label}</answer>"
        if index < answer_count
        else "invalid fixture response"
        for index in range(128)
    ]
    if stop_after is not None:
        texts = texts[:stop_after]
    model = FixedResponses(texts, Characters())
    model.config = SimpleNamespace(max_position_embeddings=2048)
    loaded = SimpleNamespace(
        model=model,
        tokenizer=model.tokenizer,
        resolved_model_commit=probe.REVISION,
        resolved_tokenizer_commit=probe.REVISION,
        tokenizer_hash=probe.TOKENIZER_SHA,
        chat_template_sha256=probe.TEMPLATE_SHA,
        prompt_protocol="qwen3_non_thinking_v1",
    )
    monkeypatch.setattr(
        probe, "load_inputs", lambda *args: ({"model": {}}, tmp_path / "fixture.pt", cohort, "d" * 64)
    )
    monkeypatch.setattr(probe, "runtime", lambda: {"scope": "synthetic CPU fixture"})
    monkeypatch.setattr(probe, "load_original_initial", lambda *args: loaded)
    inputs = dict(
        parent_job_id="54548846",
        job_id="123456",
        run_id="synthetic-run",
        source_code_sha256="a" * 64,
        plan_sha256="b" * 64,
    )
    output = tmp_path / "artifacts"
    output.mkdir()
    return inputs, output


@pytest.mark.parametrize("correct, gate", [(12, False), (13, True)])
def test_actual_original_128_row_report_keeps_execution_and_base_gate_separate(
    probe, monkeypatch, examples, tmp_path, correct, gate
):
    inputs, output = execution_fixture(probe, monkeypatch, examples, tmp_path, answer_count=correct)
    report = probe.execute(inputs, output, tmp_path, {"scope": "synthetic source fixture"})
    assert report["passed"] is report["diagnostic_complete"] is True
    assert report["initial_base_gate_satisfied"] is gate
    assert all(report[key] is False for key in probe.FLAGS)
    assert report["answer_correct_count"] == correct and report["metrics"]["answer_accuracy"] == correct / 128
    assert report["metrics"]["exact_proof_accuracy"] == 0
    assert report["raw_record_count"] == report["prompt_count"] == report["evaluated_count"] == 128
    records = [json.loads(line) for line in (output / "initial-records.jsonl").read_text().splitlines()]
    prompts = [json.loads(line) for line in (output / "initial-prompts.jsonl").read_text().splitlines()]
    assert [row["ordinal"] for row in records] == list(range(128))
    assert [row["example"]["example_id"] for row in prompts] == report["ordered_example_ids"]
    assert sum(row["verification"]["answer_correct"] for row in records) == correct
    assert all(row["verification"]["reward"] == 0 for row in records)
    assert report["raw_artifacts"] == probe.raw_artifacts(output, required=True)
    assert probe.document(output / "progress.json")["completed_examples"] == 128


def test_late_original_generation_failure_preserves_actual_seven_completed_records(
    probe, monkeypatch, examples, tmp_path
):
    inputs, output = execution_fixture(probe, monkeypatch, examples, tmp_path, answer_count=13, stop_after=7)
    with pytest.raises(IndexError):
        probe.execute(inputs, output, tmp_path, {"scope": "synthetic source fixture"})
    report = probe.document(output / "initial-probe.json")
    assert report["raw_record_count"] == 7 and report["diagnostic_complete"] is False
    assert all(report[key] is False for key in probe.FLAGS)
    assert len((output / "initial-records.jsonl").read_text().splitlines()) == 7


def test_early_original_error_survives_invalid_raw_file(probe, tmp_path):
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "initial-prompts.jsonl").mkdir()
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    assert probe.main(["--inputs-json", str(inputs), "--output-dir", str(output)]) == 1
    report = probe.document(output / "initial-probe.json")
    assert report["error"]["message"] == "unexpected input fields"
    assert "not regular" in report["raw_artifact_error"]
    assert report["passed"] is report["initial_base_gate_satisfied"] is False


def test_argument_parser_rejects_alternate_cap_or_checkpoint(probe, tmp_path):
    with pytest.raises(SystemExit) as raised:
        probe.main(
            [
                "--inputs-json",
                str(tmp_path / "input.json"),
                "--output-dir",
                str(tmp_path / "out"),
                "--max-new-tokens",
                "512",
            ]
        )
    assert raised.value.code == 2
    assert not (tmp_path / "out").exists()
