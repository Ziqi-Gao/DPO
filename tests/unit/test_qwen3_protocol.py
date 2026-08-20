from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.cli.finalize_pilot import _hash_valid as pilot_hash_valid
from posttrain_circuits.cli.train import _require_qwen3_store_binding
from posttrain_circuits.core.config import compose_config, validate_config
from posttrain_circuits.core.provenance import (
    RunManifest,
    run_manifest_payload,
    validate_run_manifest_payload,
)
from posttrain_circuits.core.types import PromptBatch
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.models.prompt_protocol import (
    chat_template_sha256,
    format_model_prompt,
)
from posttrain_circuits.rollout.generation import hf_generate_trajectories
from posttrain_circuits.training.schedules import PromptScheduler
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


def _protocol_tokenizer():  # type: ignore[no-untyped-def]
    tokenizer = build_tiny_tokenizer()
    tokenizer.chat_template = "frozen-qwen3-template-v1"

    def apply_chat_template(
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):  # type: ignore[no-untyped-def]
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    tokenizer.apply_chat_template = apply_chat_template  # type: ignore[method-assign]
    return tokenizer


def _protocol_config(tokenizer) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "messages": "single_user",
            "add_generation_prompt": True,
            "chat_template_sha256": chat_template_sha256(tokenizer),
        }
    }


@pytest.mark.unit
def test_qwen3_config_is_exact_and_never_falls_back_to_qwen25() -> None:
    config = compose_config(
        [
            "production=qwen3_primary",
            "experiment=offline_soft",
            "model=qwen3_1p7b",
            "teacher=qwen3_teacher_8b",
            "g0=qwen3_eap_separation",
            "pilot=qwen3_core",
        ],
        config_root=Path("configs"),
    )
    assert config["model"]["model_name_or_path"] == "Qwen/Qwen3-1.7B"
    assert config["teacher"]["model_name_or_path"] == "Qwen/Qwen3-8B"
    assert config["model"]["gradient_checkpointing"] is True
    assert config["model"]["use_cache"] is False
    assert config["teacher"]["gradient_checkpointing"] is False
    assert config["teacher"]["use_cache"] is True
    assert config["protocol_track"] == "qwen3_v1"
    assert "qwen3-v1" in Path(config["output_root"]).parts
    assert config["state_source"]["top_k"] == 20
    assert config["state_source"]["min_p"] == 0.0
    assert config["supervision"]["top_k"] == 20
    assert config["supervision"]["min_p"] == 0.0

    stale = copy.deepcopy(config)
    stale["state_source"]["store_path"] = "outputs/rollout_banks/qwen25"
    with pytest.raises(ValueError, match="escape their namespace"):
        validate_config(stale)


@pytest.mark.unit
def test_non_thinking_formatter_hashes_raw_and_exact_model_facing_bytes() -> None:
    tokenizer = _protocol_tokenizer()
    raw = "FACTS F01 A\nQUERY A"
    formatted = format_model_prompt(raw, tokenizer, _protocol_config(tokenizer))
    assert formatted.raw_prompt == raw
    assert formatted.prompt_protocol == "qwen3_non_thinking_v1"
    assert formatted.enable_thinking is False
    assert formatted.model_facing_prompt.count("<|im_start|>user") == 1
    assert "<think>\n\n</think>" in formatted.model_facing_prompt
    assert formatted.raw_prompt_sha256 != formatted.model_facing_prompt_sha256

    changed = copy.deepcopy(_protocol_config(tokenizer))
    changed["prompt_protocol"]["chat_template_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="chat template differs"):
        format_model_prompt(raw, tokenizer, changed)


@pytest.mark.unit
def test_tokenizer_fingerprint_includes_chat_template() -> None:
    tokenizer = _protocol_tokenizer()
    original = tokenizer_fingerprint(tokenizer)
    tokenizer.chat_template += "-tampered"
    assert tokenizer_fingerprint(tokenizer) != original


@pytest.mark.unit
def test_qwen3_sampling_parameters_reach_hf_generate(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    tokenizer = _protocol_tokenizer()
    model = build_tiny_qwen(55).eval()

    def fake_generate(**kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.8
        assert kwargs["top_k"] == 20
        assert kwargs["min_p"] == 0.0
        prefix = kwargs["input_ids"]
        token = torch.tensor([[4]], dtype=torch.long, device=prefix.device)
        scores = (torch.zeros((1, model.config.vocab_size), device=prefix.device),)
        return SimpleNamespace(sequences=torch.cat((prefix, token), dim=1), scores=scores)

    monkeypatch.setattr(model, "generate", fake_generate)
    record = hf_generate_trajectories(
        model,
        tokenizer,
        PromptBatch(("p0",), ("FACTS F01 A",)),
        policy_version=0,
        seed=17,
        max_new_tokens=1,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        policy_id="Qwen/Qwen3-1.7B",
        policy_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        model_config=_protocol_config(tokenizer),
    )[0]
    assert record.top_k == 20
    assert record.min_p == 0.0
    assert record.raw_prompt_text == "FACTS F01 A"
    assert record.prompt_text != record.raw_prompt_text
    assert len(record.response_ids) == len(record.behavior_logprobs) == 1


@pytest.mark.unit
def test_four_rank_prompt_shards_are_disjoint_and_resume_is_rank_bound() -> None:
    ids = [f"p-{index}" for index in range(16)]
    prompts = [f"prompt-{index}" for index in range(16)]
    schedulers = [
        PromptScheduler.for_distributed_rank(ids, prompts, 2, rank=rank, world_size=4) for rank in range(4)
    ]
    shards = [set(scheduler.prompt_ids) for scheduler in schedulers]
    assert set.union(*shards) == set(ids)
    assert all(shards[left].isdisjoint(shards[right]) for left in range(4) for right in range(left + 1, 4))
    with pytest.raises(ValueError, match="different rank shard"):
        schedulers[1].load_state_dict(schedulers[0].state_dict())


@pytest.mark.unit
def test_qwen3_artifacts_and_run_manifests_fail_closed_on_cross_model_or_tamper() -> None:
    config = compose_config(
        ["production=qwen3_primary", "model=qwen3_1p7b", "teacher=qwen3_teacher_8b"],
        config_root=Path("configs"),
    )
    expected_manifest = {
        "protocol_track": "qwen3_v1",
        "artifact_namespace": "qwen3-v1",
        "prompt_protocol": "qwen3_non_thinking_v1",
        "enable_thinking": False,
        "chat_template_sha256": config["model"]["prompt_protocol"]["chat_template_sha256"],
        "tokenizer_hash": config["model"]["tokenizer_fingerprint"],
        "behavior_policy": {"id": "Qwen/Qwen3-1.7B"},
    }
    _require_qwen3_store_binding(
        expected_manifest,
        config=config,
        expected_behavior_policy="Qwen/Qwen3-1.7B",
    )
    stale = copy.deepcopy(expected_manifest)
    stale["behavior_policy"]["id"] = "Qwen/Qwen2.5-1.5B-Instruct"
    with pytest.raises(ValueError, match="stale/cross-model"):
        _require_qwen3_store_binding(
            stale,
            config=config,
            expected_behavior_policy="Qwen/Qwen3-1.7B",
        )

    manifest = RunManifest(
        run_id="qwen3-run",
        experiment_cell="offline_soft",
        seed=42,
        model_id="Qwen/Qwen3-1.7B",
        model_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        tokenizer_id="Qwen/Qwen3-1.7B",
        tokenizer_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        resolved_model_commit="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        resolved_tokenizer_commit="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        dataset_hashes={"train": "hash"},
        rollout_bank_hash="bank",
        prompt_schedule_hash="schedule",
        raw_prompt_schedule_hash="1" * 64,
        model_facing_prompt_schedule_hash="2" * 64,
        prompt_protocol="qwen3_non_thinking_v1",
        enable_thinking=False,
        chat_template_sha256="3" * 64,
        tokenizer_fingerprint="4" * 64,
        protocol_track="qwen3_v1",
        artifact_namespace="qwen3-v1",
        prereg_version="qwen3_v1",
        prereg_path="prereg/qwen3_v1.yaml",
    )
    payload = run_manifest_payload(manifest)
    validate_run_manifest_payload(payload)
    payload["model_revision"] = "tampered"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_run_manifest_payload(payload)
    assert pilot_hash_valid({"phase": "pilot-input", "sha256": "bad"}) is False


@pytest.mark.unit
def test_qwen3_launch_paths_require_explicit_configs_and_contain_no_qwen25_fallback() -> None:
    paths = (
        Path("scripts/production/run_qwen3_gpu_preflight.sh"),
        Path("scripts/production/run_qwen3_g0.sh"),
        Path("scripts/production/submit_qwen3_pilot.sh"),
        Path("scripts/slurm/qwen3_gpu_preflight.slurm"),
    )
    required = {
        "MODEL_CONFIG",
        "TEACHER_CONFIG",
        "PRODUCTION_CONFIG",
        "G0_CONFIG",
        "PILOT_CONFIG",
        "PROJECT_ROOT",
        "PYTHON_BIN",
        "ACCELERATE_BIN",
        "OUTPUT_ROOT",
    }
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "qwen25" not in text.lower()
        assert required <= {name for name in required if name in text}
