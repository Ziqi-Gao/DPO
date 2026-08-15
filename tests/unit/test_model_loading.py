from __future__ import annotations

from pathlib import Path

import pytest

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.models import loading
from posttrain_circuits.models.loading import (
    assert_tokenizer_compatible,
    load_model_and_tokenizer,
)
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


@pytest.mark.unit
def test_production_loader_pins_revisions_and_records_resolved_commits(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    model = build_tiny_qwen(1)
    tokenizer = build_tiny_tokenizer()
    model.config._commit_hash = "resolved-model"
    tokenizer._commit_hash = "resolved-tokenizer"
    calls: dict[str, dict[str, object]] = {}

    def load_tokenizer(identifier: str, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["tokenizer"] = {"identifier": identifier, **kwargs}
        return tokenizer

    def load_model(identifier: str, **kwargs: object):  # type: ignore[no-untyped-def]
        calls["model"] = {"identifier": identifier, **kwargs}
        return model

    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(loading.AutoModelForCausalLM, "from_pretrained", load_model)
    config = compose_config([], config_root=Path("configs"))["model"]
    bundle = load_model_and_tokenizer(config, for_training=True)
    assert calls["model"]["revision"] == config["model_revision"]
    assert calls["tokenizer"]["revision"] == config["tokenizer_revision"]
    assert bundle.resolved_model_commit == "resolved-model"
    assert bundle.resolved_tokenizer_commit == "resolved-tokenizer"
    assert bundle.tokenizer_hash
    assert model.config.use_cache is False


@pytest.mark.unit
def test_production_loader_rejects_lora_before_loading() -> None:
    config = compose_config([], config_root=Path("configs"))["model"]
    config["use_lora"] = True
    with pytest.raises(ValueError, match="prohibit LoRA"):
        load_model_and_tokenizer(config, for_training=True)


@pytest.mark.unit
def test_student_teacher_tokenizer_compatibility_is_exact() -> None:
    student = build_tiny_tokenizer()
    teacher = build_tiny_tokenizer()
    assert assert_tokenizer_compatible(student, teacher)
    teacher.add_tokens(["TEACHER_ONLY"])
    with pytest.raises(ValueError, match="not exactly compatible"):
        assert_tokenizer_compatible(student, teacher)
