from __future__ import annotations

import copy
import json

import pytest
import torch

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.learning.teacher.adaptation import PREFLIGHT, checkpoint_manifest
from posttrain_circuits.models import adapted_teacher, loading
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.models.prompt_protocol import chat_template_sha256
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """Real local dense save after merging a tiny adapter; no production pin bypass."""
    peft = pytest.importorskip("peft")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    base_id, base_revision = "test-only/tiny-qwen3", "a" * 40
    # The production entry point has no test-mode parameter. These two identities
    # are changed only inside this explicit CPU fixture and restored by pytest.
    monkeypatch.setattr(adapted_teacher, "BASE_MODEL_ID", base_id)
    monkeypatch.setattr(adapted_teacher, "BASE_REVISION", base_revision)
    model, tokenizer = build_tiny_qwen3(71), build_tiny_tokenizer()
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    adapted = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.01)
    adapted.save_pretrained(tmp_path / "adapter", safe_serialization=True)
    dense = adapted.merge_and_unload().eval()
    dense.save_pretrained(tmp_path / "merged", safe_serialization=True)
    tokenizer.save_pretrained(tmp_path / "merged")
    manifest = checkpoint_manifest(
        tmp_path,
        base_revision=base_revision,
        plan={**PREFLIGHT, "base_revision": base_revision},
    )
    (tmp_path / "dense-manifest.json").write_text(json.dumps(manifest))
    config = {
        "model_name_or_path": base_id,
        "model_revision": base_revision,
        "tokenizer_name_or_path": base_id,
        "tokenizer_revision": base_revision,
        "torch_dtype": "float32",
        "attn_implementation": "eager",
        "gradient_checkpointing": False,
        "use_cache": True,
        "trust_remote_code": False,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": chat_template_sha256(tokenizer),
        },
    }
    yield tmp_path, manifest, config, dense, tokenizer
    torch.set_num_threads(previous_threads)


def load(checkpoint, *, expected=None, config=None):
    root, manifest, base_config, _, _ = checkpoint
    return adapted_teacher.load_adapted_teacher(
        root,
        manifest["sha256"] if expected is None else expected,
        base_config if config is None else config,
        device=torch.device("cpu"),
    )


def bind_manifest(root, manifest):
    manifest["sha256"] = sha256_value({k: v for k, v in manifest.items() if k != "sha256"})
    (root / "dense-manifest.json").write_text(json.dumps(manifest))


def rehash_checkpoint(checkpoint):
    root, manifest, _, _, _ = checkpoint
    updated = checkpoint_manifest(root, base_revision=manifest["base_revision"], plan=PREFLIGHT)
    manifest.clear()
    manifest.update(updated)
    bind_manifest(root, manifest)


def test_real_merged_checkpoint_loads_offline_with_distinct_identity(checkpoint, monkeypatch):
    _, manifest, config, expected, tokenizer = checkpoint
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: pytest.fail("CPU load selected CUDA"))
    calls = []
    original = adapted_teacher.AutoModelForCausalLM.from_pretrained

    def observe(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapted_teacher.AutoModelForCausalLM, "from_pretrained", observe)
    result = load(checkpoint)
    inputs = {"input_ids": torch.tensor([[2, 7, 8, 3]])}
    with torch.inference_mode():
        torch.testing.assert_close(result.model(**inputs).logits, expected(**inputs).logits)
    assert result.teacher_checkpoint_sha256 == manifest["sha256"]
    assert result.base_revision == config["model_revision"]
    assert not hasattr(result, "resolved_model_commit")
    assert result.tokenizer_id == config["tokenizer_name_or_path"]
    assert result.requested_tokenizer_revision == config["tokenizer_revision"]
    assert result.tokenizer_hash == tokenizer_fingerprint(tokenizer)
    assert result.chat_template_sha256 == chat_template_sha256(tokenizer)
    assert result.prompt_protocol == "qwen3_non_thinking_v1"
    assert result.model.training is False
    assert all(not parameter.requires_grad for parameter in result.model.parameters())
    assert next(result.model.parameters()).device.type == "cpu"
    assert calls[0]["local_files_only"] is True and calls[0]["trust_remote_code"] is False
    assert calls[0]["use_safetensors"] is True and "revision" not in calls[0]


@pytest.mark.parametrize("mutation", ["corrupt", "truncate", "missing", "extra", "adapter_corrupt"])
def test_all_checkpoint_bytes_verified_before_loading(checkpoint, monkeypatch, mutation):
    root = checkpoint[0]
    weight = root / "merged/model.safetensors"
    if mutation == "corrupt":
        content = bytearray(weight.read_bytes())
        content[-1] ^= 1
        weight.write_bytes(content)
    elif mutation == "truncate":
        weight.write_bytes(weight.read_bytes()[:-1])
    elif mutation == "missing":
        weight.unlink()
    elif mutation == "extra":
        (root / "merged/unlisted.txt").write_text("not in content identity")
    else:
        (root / "adapter/adapter_model.safetensors").write_bytes(b"bad adapter provenance")
    monkeypatch.setattr(
        adapted_teacher.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kw: pytest.fail("loaded before inventory validation"),
    )
    with pytest.raises(ValueError, match="bytes differ|missing or extra"):
        load(checkpoint)


@pytest.mark.parametrize("target", ["root", "merged", "weight", "manifest", "parent"])
def test_symlinks_rejected_before_loading(checkpoint, target):
    root, manifest, config, _, _ = checkpoint
    if target in {"root", "parent"}:
        link = root.parent / (root.name + "-link")
        link.symlink_to(root, target_is_directory=True)
        if target == "parent":
            nested = root / "child"
            nested.mkdir()
            link = link / "child"
        with pytest.raises(ValueError, match="symlink"):
            adapted_teacher.load_adapted_teacher(link, manifest["sha256"], config, device="cpu")
        return
    path = (
        root
        / {"merged": "merged", "weight": "merged/model.safetensors", "manifest": "dense-manifest.json"}[
            target
        ]
    )
    destination = root.parent / (root.name + "-external")
    path.rename(destination)
    path.symlink_to(destination, target_is_directory=target == "merged")
    with pytest.raises(ValueError, match="regular|real|symlink"):
        load(checkpoint)


@pytest.mark.parametrize("mutation", ["wrong_base", "traversal", "duplicate", "boolean_size", "accepted"])
def test_manifest_validation_survives_rebound_outer_hash(checkpoint, mutation):
    root, manifest, _, _, _ = checkpoint
    if mutation == "wrong_base":
        manifest["base_revision"] = "b" * 40
    elif mutation == "traversal":
        manifest["files"][0]["path"] = "merged/../../outside.safetensors"
    elif mutation == "duplicate":
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    elif mutation == "boolean_size":
        manifest["files"][0]["size"] = True
    else:
        manifest["formal_teacher_accepted"] = True
    bind_manifest(root, manifest)
    with pytest.raises(ValueError):
        load(checkpoint)


def test_wrong_expected_identity_and_duplicate_json_key_fail(checkpoint):
    with pytest.raises(ValueError, match="identity differs"):
        load(checkpoint, expected="f" * 64)
    root, manifest, _, _, _ = checkpoint
    text = json.dumps(manifest)
    (root / "dense-manifest.json").write_text('{"sha256":"' + manifest["sha256"] + '",' + text[1:])
    with pytest.raises(ValueError, match="duplicate checkpoint JSON key"):
        load(checkpoint)


@pytest.mark.parametrize(
    "mutation", ["wrong_architecture", "false_commit", "custom_code", "quantized", "adapter"]
)
def test_dense_model_configuration_fail_closed(checkpoint, mutation):
    root = checkpoint[0]
    path = root / "merged/config.json"
    value = json.loads(path.read_text())
    if mutation == "wrong_architecture":
        value["model_type"] = "qwen2"
    elif mutation == "false_commit":
        value["_commit_hash"] = checkpoint[2]["model_revision"]
    elif mutation == "custom_code":
        value["auto_map"] = {"AutoModelForCausalLM": "evil.custom"}
    elif mutation == "quantized":
        value["quantization_config"] = {"load_in_4bit": True}
    else:
        (root / "merged/adapter_config.json").write_text("{}")
    path.write_text(json.dumps(value))
    rehash_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="architecture|Hub commit|custom code|quantization|adapter entry"):
        load(checkpoint)


@pytest.mark.parametrize("field", ["tokenizer_fingerprint", "chat_template_sha256"])
def test_saved_tokenizer_checked_against_explicit_base_identity(checkpoint, field):
    config = copy.deepcopy(checkpoint[2])
    if field == "tokenizer_fingerprint":
        config[field] = "d" * 64
    else:
        config["prompt_protocol"][field] = "d" * 64
    with pytest.raises(ValueError, match="fingerprint differs|chat template differs"):
        load(checkpoint, config=config)


def test_checkpoint_modified_during_transformers_loading_is_rejected(checkpoint, monkeypatch):
    root = checkpoint[0]
    original = adapted_teacher.AutoModelForCausalLM.from_pretrained

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "merged/config.json").write_text("{}")
        return result

    monkeypatch.setattr(adapted_teacher.AutoModelForCausalLM, "from_pretrained", mutate)
    with pytest.raises(ValueError, match="checkpoint changed while loading"):
        load(checkpoint)


def test_missing_dense_weight_cannot_be_silently_randomly_initialized(checkpoint):
    from safetensors.torch import load_file, save_file

    path = checkpoint[0] / "merged/model.safetensors"
    state = load_file(path)
    del state["model.layers.0.self_attn.q_proj.weight"]
    save_file(state, path, metadata={"format": "pt"})
    rehash_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="exactly cover the model"):
        load(checkpoint)


def test_checkpoint_mutation_during_final_device_transfer_is_rejected(checkpoint, monkeypatch):
    root = checkpoint[0]
    original = adapted_teacher.AutoModelForCausalLM.from_pretrained

    def wrap_transfer(*args, **kwargs):
        model, info = original(*args, **kwargs)
        transfer = model.to

        def mutate(*args, **kwargs):
            result = transfer(*args, **kwargs)
            (root / "merged/config.json").write_text("{}")
            return result

        monkeypatch.setattr(model, "to", mutate)
        return model, info

    monkeypatch.setattr(adapted_teacher.AutoModelForCausalLM, "from_pretrained", wrap_transfer)
    with pytest.raises(ValueError, match="checkpoint changed while loading"):
        load(checkpoint)


def test_production_base_config_remains_pinned_and_original_loader_still_prohibits_adapters(monkeypatch):
    config = compose_config(["teacher=qwen3_v2_teacher_8b"])["teacher"]
    adapted_teacher._validate_base(config)
    changed = copy.deepcopy(config)
    changed["model_revision"] = "b" * 40
    with pytest.raises(ValueError, match="revisions"):
        adapted_teacher._validate_base(changed)
    changed = copy.deepcopy(config)
    changed["use_lora"] = True
    monkeypatch.setattr(
        loading.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kw: pytest.fail("adapter load reached Transformers"),
    )
    with pytest.raises(ValueError, match="prohibit LoRA"):
        loading.load_model_and_tokenizer(changed, for_training=False)
    with pytest.raises(ValueError, match="prohibit LoRA"):
        adapted_teacher._validate_base(changed)
