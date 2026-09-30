"""Real Qwen3 configuration guard and tiny CPU checkpoint-loading regression.

Hub model/tokenizer factories are replaced by local fixtures. The actual
load_model_and_tokenizer and validate_model_revision run unmodified; only the
production tokenizer-fingerprint check is isolated from this tiny tokenizer.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from posttrain_circuits.core.config import validate_model_revision
from posttrain_circuits.models import loading
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

ROOT = Path(__file__).resolve().parents[2]


def worker():
    spec = importlib.util.spec_from_file_location(
        "student_quality_loading_test", ROOT / "tools/sdsc_student_quality_probe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_base():
    model = build_tiny_qwen3(123).to(torch.bfloat16)
    model.register_buffer("fixture_counter", torch.tensor(7, dtype=torch.int64))
    return model


@pytest.fixture
def model_config():
    config = yaml.safe_load((ROOT / "configs/model/qwen3_v2_1p7b.yaml").read_text())
    validate_model_revision(config)
    return config


@pytest.fixture
def local_hf(monkeypatch):
    calls = []

    def model_factory(_path, **kwargs):
        calls.append({"dtype": kwargs["torch_dtype"], "revision": kwargs["revision"]})
        assert kwargs["torch_dtype"] == torch.bfloat16
        return tiny_base()

    monkeypatch.setattr(loading.AutoModelForCausalLM, "from_pretrained", model_factory)
    monkeypatch.setattr(
        loading.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: build_tiny_tokenizer()
    )
    monkeypatch.setattr(
        loading,
        "_validate_tokenizer_protocol",
        lambda tokenizer, config: ("local-fixture", "qwen3_non_thinking_v1"),
    )
    return calls


def save_checkpoint(path, *, label, defect=None):
    state = copy.deepcopy(tiny_base().state_dict())
    if label != "initial":
        state = {name: value.float() if value.is_floating_point() else value for name, value in state.items()}
        state["model.embed_tokens.weight"].flatten()[0] = 0.1234567
        assert (
            state["model.embed_tokens.weight"].flatten()[0]
            != state["model.embed_tokens.weight"].flatten()[0].bfloat16().float()
        )
    key = "model.embed_tokens.weight"
    if defect == "wrong_floating_dtype":
        state[key] = state[key].double()
    elif defect == "mixed_floating_dtype":
        state[key] = state[key].bfloat16()
    elif defect == "missing_key":
        state.pop(key)
    elif defect == "extra_key":
        state["unexpected.weight"] = torch.ones(1, dtype=state[key].dtype)
    elif defect == "wrong_shape":
        state[key] = state[key][:-1]
    elif defect == "nonfloating_dtype":
        state["fixture_counter"] = state["fixture_counter"].to(torch.int32)
    elif defect == "nonfloating_to_float":
        state["fixture_counter"] = state["fixture_counter"].float()
    elif defect == "complex":
        state["fixture_counter"] = torch.tensor(7 + 1j)
    elif defect == "non_tensor":
        state["fixture_counter"] = None
    elif defect in {"nan", "inf"}:
        state[key].flatten()[0] = float(defect)
    torch.save(
        {"format": "initial_hf_full_state_v1" if label == "initial" else "fixture", "model": state}, path
    )
    return state


def test_original_float32_config_override_reproduces_real_guard_failure(model_config, local_hf):
    original = copy.deepcopy(model_config)
    with pytest.raises(ValueError, match="Qwen3 controlled runs require BF16 and SDPA"):
        loading.load_model_and_tokenizer(dict(model_config, torch_dtype="float32"), for_training=False)
    assert local_hf == []
    assert model_config == original


@pytest.mark.parametrize("label", ["initial", "step20", "step33"])
def test_saved_dtype_loads_exactly_after_original_config_validation(tmp_path, model_config, local_hf, label):
    checkpoint = tmp_path / "checkpoint.pt"
    state = save_checkpoint(checkpoint, label=label)
    config_before = copy.deepcopy(model_config)
    checkpoint_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded = worker().load_checkpoint_for_diagnosis(model_config, checkpoint, label=label)
    assert local_hf == [{"dtype": torch.bfloat16, "revision": model_config["model_revision"]}]
    assert model_config == config_before
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_before
    assert set(loaded.model.state_dict()) == set(state)
    for name, actual in loaded.model.state_dict().items():
        assert actual.dtype == state[name].dtype
        assert torch.equal(actual, state[name])
        assert actual.device.type == "cpu"
    assert not loaded.model.training
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_floating_dtype",
        "mixed_floating_dtype",
        "missing_key",
        "extra_key",
        "wrong_shape",
        "nonfloating_dtype",
        "nonfloating_to_float",
        "complex",
        "non_tensor",
        "nan",
        "inf",
    ],
)
def test_malformed_checkpoint_never_returns_a_loaded_model(tmp_path, model_config, local_hf, defect):
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, label="step20", defect=defect)
    config_before = copy.deepcopy(model_config)
    with pytest.raises(ValueError, match="checkpoint"):
        worker().load_checkpoint_for_diagnosis(model_config, checkpoint, label="step20")
    assert model_config == config_before
    assert not torch.cuda.is_initialized()


def test_new_helper_does_not_bypass_original_model_guard(tmp_path, model_config, local_hf):
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, label="step20")
    model_config["attn_implementation"] = "eager"
    original = copy.deepcopy(model_config)
    with pytest.raises(ValueError, match="Qwen3 controlled runs require BF16 and SDPA"):
        worker().load_checkpoint_for_diagnosis(model_config, checkpoint, label="step20")
    assert local_hf == []
    assert model_config == original


def test_failed_main_retains_completed_arms_and_nll_without_success_claims(tmp_path, monkeypatch):
    probe = worker()
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    output = tmp_path / "output"
    partial = {
        "schema": "quest-sdsc-student-quality-probe-v1",
        "job_id": "fixture",
        "arms": [{"checkpoint": "initial", "cap": 128, "teacher_forced": {"canonical_response_nll": 1.25}}],
        "checkpoint_evidence": [{"label": "initial", "loaded_exactly": True}],
        "passed": True,
        "diagnostic_complete": True,
        "student_accepted": True,
        "g0_passed": True,
        "pilot_passed": True,
        "factorial_ready": True,
    }

    def failed_execute(_inputs, directory):
        probe.atomic_json(directory / "progress.json", partial)
        raise ValueError("later checkpoint failure")

    monkeypatch.setattr(probe, "execute", failed_execute)
    assert probe.main(["--inputs-json", str(inputs), "--output-dir", str(output)]) == 1
    observed = json.loads((output / "quality-probe.json").read_text())
    assert observed["arms"] == partial["arms"]
    assert observed["checkpoint_evidence"] == partial["checkpoint_evidence"]
    assert observed["job_id"] == "fixture"
    assert observed["error"] == "later checkpoint failure"
    for key in (
        "passed",
        "diagnostic_complete",
        "student_accepted",
        "g0_passed",
        "pilot_passed",
        "factorial_ready",
    ):
        assert observed[key] is False
    assert json.loads((output / "progress.json").read_text()) == partial


def test_bad_partial_progress_does_not_mask_original_failure(tmp_path, monkeypatch):
    probe = worker()
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    output = tmp_path / "output"

    def failed_execute(_inputs, directory):
        (directory / "progress.json").write_text("malformed-progress")
        raise ValueError("original checkpoint failure")

    monkeypatch.setattr(probe, "execute", failed_execute)
    assert probe.main(["--inputs-json", str(inputs), "--output-dir", str(output)]) == 1
    observed = json.loads((output / "quality-probe.json").read_text())
    assert observed["error"] == "original checkpoint failure"
    assert observed["passed"] is False
    assert observed["factorial_ready"] is False
