"""Real tiny-Qwen CPU precision tests; no qualification or GPU execution claims."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

from posttrain_circuits.artifacts.checkpoints import torch_state_hash
from posttrain_circuits.artifacts.hashing import sha256_file
from posttrain_circuits.models import prepared_student as module

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_only():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)
    assert not torch.cuda.is_initialized()


def tiny_model(*, tied=True):
    # 28 genuine Qwen3 blocks preserve the native311-key topology with tiny shapes.
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=64,
            tie_word_embeddings=tied,
            use_cache=False,
        )
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert len(model.state_dict()) == 311
    return model


def payload(state):
    return dict(
        format="student_branch_dense_v3",
        model=state,
        global_step=2,
        parent_checkpoint_sha256="1" * 64,
        dataset_sha256="2" * 64,
        protocol_sha256="3" * 64,
        protocol_artifact_sha256="4" * 64,
        learning_rate=5e-5,
        scope="common_student_branch_model_only",
        **dict.fromkeys(module._FLAGS, False),
    )


def write_checkpoint(path, saved, *, master=None):
    torch.save(saved, path)
    return module.PreparedStudentCheckpoint(
        checkpoint_sha256=sha256_file(path),
        checkpoint_size=path.stat().st_size,
        master_model_state_sha256=master or torch_state_hash(saved["model"]),
        checkpoint_step=2,
        parent_checkpoint_sha256="1" * 64,
        dataset_sha256="2" * 64,
        protocol_sha256="3" * 64,
        protocol_artifact_sha256="4" * 64,
    )


@pytest.fixture
def checkpoint(tmp_path):
    model = tiny_model()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    # One FP32 ULP above1 is lost by a BF16-first load. Tied names must agree.
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        state[name].flatten()[0] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    path = tmp_path / "prepared.pt"
    saved = payload(state)
    expected = write_checkpoint(path, saved)
    return path, saved, expected


@pytest.fixture
def base_config():
    return yaml.safe_load((ROOT / "configs/model/qwen3_v2_1p7b.yaml").read_text())


def assert_cpu_rng_unchanged(before):
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("tied", [False, True])
def test_real_tiny_qwen_exact_fp32_restore_preserves_aliases_and_updates(tied):
    model = tiny_model(tied=tied)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        state[name].flatten()[0] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    expected_hash = torch_state_hash(state)
    parameters = {name: id(value) for name, value in model.named_parameters(remove_duplicate=False)}
    result = module._restore_fp32_masters(model, state, expected_hash)
    assert result == torch_state_hash(model.state_dict()) == expected_hash
    actual = model.get_input_embeddings().weight.flatten()[0]
    assert actual.item() != actual.bfloat16().float().item()
    assert {name: id(value) for name, value in model.named_parameters(remove_duplicate=False)} == parameters
    assert (model.get_input_embeddings().weight is model.get_output_embeddings().weight) is tied
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    assert optimizer.state == {}
    # Actual CPU FP32 forward/backward/update, outside any loader-created optimizer.
    ids = torch.tensor([[2, 3, 4, 5]])
    loss = model(input_ids=ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert torch_state_hash(model.state_dict()) != expected_hash
    assert all(
        parameter.dtype == torch.float32 and parameter.requires_grad for parameter in model.parameters()
    )


def test_tied_alias_disagreement_rejects_before_any_parameter_mutation(checkpoint):
    _, saved, _ = checkpoint
    state = copy.deepcopy(saved["model"])
    state["lm_head.weight"].flatten()[1] += 0.1
    model = tiny_model()
    before = torch_state_hash(model.state_dict())
    with pytest.raises(ValueError, match="tied"):
        module._restore_fp32_masters(model, state, torch_state_hash(state))
    assert torch_state_hash(model.state_dict()) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "shape",
        "dtype",
        "nan",
        "inf",
        "meta",
        "sparse",
        "parameter",
        "requires_grad",
        "wrong_hash",
        "one_ulp",
        "signed_zero",
        "frozen_model",
        "bf16_model",
    ],
)
def test_bad_master_states_rejected_without_mutating_model(checkpoint, mutation):
    _, saved, expected = checkpoint
    state = copy.deepcopy(saved["model"])
    model = tiny_model()
    key = "model.layers.0.self_attn.q_proj.weight"
    supplied_hash = expected.master_model_state_sha256
    if mutation == "missing":
        state.pop(key)
    elif mutation == "extra":
        state["unexpected"] = torch.zeros(1)
    elif mutation == "shape":
        state[key] = torch.zeros(1)
    elif mutation == "dtype":
        state[key] = state[key].bfloat16()
    elif mutation in {"nan", "inf"}:
        state[key].flatten()[0] = float(mutation)
    elif mutation == "meta":
        state[key] = state[key].to("meta")
    elif mutation == "sparse":
        state[key] = state[key].to_sparse()
    elif mutation == "parameter":
        state[key] = torch.nn.Parameter(state[key])
    elif mutation == "requires_grad":
        state[key].requires_grad_(True)
    elif mutation == "wrong_hash":
        supplied_hash = "f" * 64
    elif mutation == "one_ulp":
        value = state[key].flatten()
        value[0] = torch.nextafter(value[0], torch.tensor(float("inf")))
    elif mutation == "signed_zero":
        state[key].flatten()[0] = 0.0
        supplied_hash = torch_state_hash(state)
        state[key].flatten()[0] = -0.0
    elif mutation == "frozen_model":
        next(model.parameters()).requires_grad_(False)
    elif mutation == "bf16_model":
        model.bfloat16()
    before = torch_state_hash(model.state_dict())
    with pytest.raises(ValueError):
        module._restore_fp32_masters(model, state, supplied_hash)
    assert torch_state_hash(model.state_dict()) == before


def test_public_artifact_load_preserves_config_rng_and_precision(checkpoint, base_config, monkeypatch):
    path, saved, expected = checkpoint
    before_config = copy.deepcopy(base_config)
    constructor_calls = []

    def tiny_constructor(config):
        # Explicit construction-boundary fixture only. No native architecture
        # claim is made by this test; all restoration/file code is real.
        constructor_calls.append(copy.deepcopy(config))
        return SimpleNamespace(model=tiny_model(), tokenizer=object())

    monkeypatch.setattr(module, "_construct_native_cpu", tiny_constructor)
    before_rng = torch.get_rng_state().clone()
    result = module.load_prepared_student_for_training(base_config, checkpoint_path=path, expected=expected)
    assert_cpu_rng_unchanged(before_rng)
    assert base_config == before_config == constructor_calls[0]
    assert result.checkpoint is expected and result.artifact_load_only is True
    assert result.restored_tensor_count == 311
    assert result.restored_master_model_state_sha256 == expected.master_model_state_sha256
    assert torch_state_hash(result.loaded.model.state_dict()) == torch_state_hash(saved["model"])
    assert all(
        p.dtype == torch.float32 and p.device.type == "cpu" and p.requires_grad
        for p in result.loaded.model.parameters()
    )
    assert not hasattr(result, "student_accepted") and not hasattr(result, "optimizer")


@pytest.mark.parametrize("point", ["constructor", "restore"])
def test_failure_preserves_caller_rng_and_config(checkpoint, base_config, monkeypatch, point):
    path, _, expected = checkpoint

    def broken_constructor(config):
        torch.rand(17)
        if point == "constructor":
            config["torch_dtype"] = "mutated-private-copy"
            raise ValueError("constructor failure")
        return SimpleNamespace(model=tiny_model().bfloat16())

    monkeypatch.setattr(module, "_construct_native_cpu", broken_constructor)
    before_rng, before_config = torch.get_rng_state().clone(), copy.deepcopy(base_config)
    with pytest.raises(ValueError):
        module.load_prepared_student_for_training(base_config, checkpoint_path=path, expected=expected)
    assert_cpu_rng_unchanged(before_rng)
    assert base_config == before_config


def test_non_cpu_global_default_device_does_not_move_loader_to_cuda(checkpoint, base_config, monkeypatch):
    path, _, expected = checkpoint
    seen = []

    def constructor(config):
        seen.append(torch.empty(1).device.type)
        return SimpleNamespace(model=tiny_model())

    monkeypatch.setattr(module, "_construct_native_cpu", constructor)
    before_rng = torch.get_rng_state().clone()
    # Meta exercises a non-CPU default without initializing/depending on CUDA.
    with torch.device("meta"):
        result = module.load_prepared_student_for_training(
            base_config, checkpoint_path=path, expected=expected
        )
    assert seen == ["cpu"] and next(result.loaded.model.parameters()).device.type == "cpu"
    assert_cpu_rng_unchanged(before_rng)


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "student_invariance_dense_v2"),
        ("scope", "optimizer_resume"),
        ("global_step", 4),
        ("global_step", True),
        ("global_step", 2.0),
        ("learning_rate", 5e-4),
        ("parent_checkpoint_sha256", "a" * 64),
        ("dataset_sha256", "a" * 64),
        ("protocol_sha256", "a" * 64),
        ("protocol_artifact_sha256", "a" * 64),
        *[(flag, True) for flag in module._FLAGS],
        *[(flag, 0) for flag in module._FLAGS],
    ],
)
def test_export_metadata_and_false_flags_are_strict(checkpoint, field, value):
    path, saved, _ = checkpoint
    saved[field] = value
    expected = write_checkpoint(path, saved)
    with pytest.raises(ValueError):
        module._read_checkpoint(path, expected)


@pytest.mark.parametrize("field", ["optimizer", "scheduler", "rng", "accepted", "missing_scope"])
def test_model_only_schema_rejects_extra_or_missing_state(checkpoint, field):
    path, saved, _ = checkpoint
    if field == "missing_scope":
        saved.pop("scope")
    else:
        saved[field] = {}
    expected = write_checkpoint(path, saved)
    with pytest.raises(ValueError, match="fields"):
        module._read_checkpoint(path, expected)


@pytest.mark.parametrize("change", ["size", "sha", "relative", "symlink", "parent_symlink", "directory"])
def test_bound_regular_file_and_paths(checkpoint, tmp_path, change):
    path, _, expected = checkpoint
    if change == "size":
        expected = replace(expected, checkpoint_size=expected.checkpoint_size + 1)
    elif change == "sha":
        expected = replace(expected, checkpoint_sha256="f" * 64)
    elif change == "relative":
        path = Path("prepared.pt")
    elif change == "symlink":
        linked = tmp_path / "linked.pt"
        linked.symlink_to(path)
        path = linked
    elif change == "parent_symlink":
        linked = tmp_path / "linked"
        linked.symlink_to(tmp_path, target_is_directory=True)
        path = linked / path.name
    elif change == "directory":
        path = tmp_path
    with pytest.raises((ValueError, IsADirectoryError)):
        module._read_checkpoint(path, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_size", True),
        ("checkpoint_size", 0),
        ("checkpoint_size", 8 * 1024**3 + 1),
        ("checkpoint_step", True),
        ("checkpoint_step", 9),
        ("checkpoint_step", 2.0),
        ("checkpoint_sha256", "A" * 64),
        ("master_model_state_sha256", "bad"),
        ("parent_checkpoint_sha256", None),
        ("dataset_sha256", 7),
        ("protocol_sha256", ""),
        ("protocol_artifact_sha256", "bad"),
    ],
)
def test_expected_identity_is_typed_and_bounded(checkpoint, field, value):
    with pytest.raises(ValueError):
        replace(checkpoint[2], **{field: value})


def test_expected_identity_is_immutable(checkpoint):
    with pytest.raises(FrozenInstanceError):
        checkpoint[2].checkpoint_step = 4


def test_expected_identity_size_accepts_exact_eight_gib_boundary(checkpoint):
    assert replace(checkpoint[2], checkpoint_size=8 * 1024**3).checkpoint_size == 8 * 1024**3


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name_or_path", "Qwen/Qwen3-8B"),
        ("model_revision", "a" * 40),
        ("tokenizer_revision", "a" * 40),
        ("tokenizer_fingerprint", "a" * 64),
        ("torch_dtype", "float32"),
        ("gradient_checkpointing", False),
        ("use_cache", True),
        ("trust_remote_code", True),
        ("allow_unpinned_revision", True),
        ("attn_implementation", "eager"),
    ],
)
def test_native_base_policy_is_not_silently_changed(base_config, field, value):
    base_config[field] = value
    with pytest.raises(ValueError):
        module._validate_base_config(base_config)


def test_autocast_is_rejected_before_artifact_or_constructor(checkpoint, base_config):
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        module.load_prepared_student_for_training(
            base_config, checkpoint_path=checkpoint[0], expected=checkpoint[2]
        )


def test_native_constructor_offline_fp32_contract_and_ancestry(base_config, monkeypatch):
    calls = []
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    config = tiny_model().config
    config._commit_hash = module.QWEN3_STUDENT_REVISION

    class ConfigFactory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("config", name, kwargs))
            return config

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("tokenizer", name, kwargs))
            return tokenizer

    class ModelFactory:
        @staticmethod
        def from_config(value, **kwargs):
            calls.append(("model", value, kwargs))
            assert value is config and kwargs["torch_dtype"] is torch.float32
            return Qwen3ForCausalLM(value)

    monkeypatch.setattr(module, "AutoConfig", ConfigFactory)
    monkeypatch.setattr(module, "AutoTokenizer", TokenizerFactory)
    monkeypatch.setattr(module, "AutoModelForCausalLM", ModelFactory)
    # Tokenizer identity primitive is independently exercised elsewhere; this
    # fixture tests exact constructor arguments/ancestry, not a Hub download.
    monkeypatch.setattr(
        module,
        "_validate_tokenizer_protocol",
        lambda tok, cfg: (module._TOKENIZER_SHA256, "qwen3_non_thinking_v1"),
    )
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        loaded = module._construct_native_cpu(base_config)
    for kind, name, kwargs in calls[:2]:
        assert kind in {"config", "tokenizer"} and name == module.QWEN3_STUDENT
        assert kwargs == dict(
            revision=module.QWEN3_STUDENT_REVISION, local_files_only=True, trust_remote_code=False
        )
    assert calls[2][2] == dict(torch_dtype=torch.float32, attn_implementation="sdpa", trust_remote_code=False)
    assert loaded.model_id == module.QWEN3_STUDENT
    assert loaded.resolved_model_commit == loaded.resolved_tokenizer_commit == module.QWEN3_STUDENT_REVISION
    assert loaded.model.training and loaded.model.is_gradient_checkpointing
    assert all(p.dtype == torch.float32 and p.requires_grad for p in loaded.model.parameters())


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "legacy_raw_v1"),
        ("enable_thinking", True),
        ("enable_thinking", 0),
        ("messages", "system_and_user"),
        ("add_generation_prompt", False),
        ("add_generation_prompt", 1),
        ("chat_template_sha256", "f" * 64),
        ("unreviewed", True),
    ],
)
def test_prompt_policy_identity_is_exact(base_config, field, value):
    base_config["prompt_protocol"][field] = value
    with pytest.raises(ValueError, match="prompt"):
        module._validate_base_config(base_config)


def test_file_replacement_during_load_is_rejected(checkpoint, monkeypatch, tmp_path):
    path, _, expected = checkpoint
    original_hash = module._hash_stream
    count = 0

    def replace_after_hash(stream):
        nonlocal count
        result = original_hash(stream)
        count += 1
        if count == 1:
            replacement = tmp_path / "replacement.pt"
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return result

    monkeypatch.setattr(module, "_hash_stream", replace_after_hash)
    with pytest.raises(ValueError, match="identity changed"):
        module._read_checkpoint(path, expected)


def test_real_tokenizer_fingerprint_rejects_wrong_constructor_tokenizer(base_config, monkeypatch):
    from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer

    architecture = tiny_model().config
    architecture._commit_hash = module.QWEN3_STUDENT_REVISION
    tokenizer = build_tiny_tokenizer()
    monkeypatch.setattr(module, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **kw: architecture))
    monkeypatch.setattr(module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer))
    # This exercises the genuine fingerprint routine before model construction.
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        module._construct_native_cpu(base_config)
