"""Production Hugging Face loading with pinned revisions and adapter prohibition."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain_circuits.core.config import validate_model_revision
from posttrain_circuits.core.hashing import sha256_value

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    requested_model_revision: str
    resolved_model_commit: str
    tokenizer_id: str
    requested_tokenizer_revision: str
    resolved_tokenizer_commit: str
    tokenizer_hash: str


def move_model_to_local_cuda(model: Any) -> Any:
    """Place an inference or teacher model on this process's assigned CUDA device."""

    if not torch.cuda.is_available():
        return model
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside {torch.cuda.device_count()} visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    return model.to(torch.device("cuda", local_rank))


def _resolved_commit(value: Any, fallback: str) -> str:
    init_kwargs = getattr(value, "init_kwargs", None)
    candidates = [
        getattr(getattr(value, "config", None), "_commit_hash", None),
        getattr(value, "_commit_hash", None),
        init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None,
    ]
    commit = next((str(item) for item in candidates if item), fallback)
    if not commit:
        raise RuntimeError("Hugging Face loader did not expose a resolved commit")
    return commit


def tokenizer_fingerprint(tokenizer: Any) -> str:
    vocabulary = tokenizer.get_vocab()
    special = {
        name: getattr(tokenizer, name, None)
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        )
    }
    return sha256_value(
        {
            "vocabulary": sorted((str(token), int(index)) for token, index in vocabulary.items()),
            "special_token_ids": special,
        }
    )


def assert_tokenizer_compatible(student: Any, teacher: Any) -> str:
    student_hash = tokenizer_fingerprint(student)
    teacher_hash = tokenizer_fingerprint(teacher)
    probes = [
        "FACTS F01 A RULES R01 A -> Q",
        "<proof> S01: R01(F01) -> Q </proof> <answer>1</answer>",
    ]
    student_ids = [student.encode(text, add_special_tokens=False) for text in probes]
    teacher_ids = [teacher.encode(text, add_special_tokens=False) for text in probes]
    if student_hash != teacher_hash or student_ids != teacher_ids:
        raise ValueError(
            "student and teacher tokenizers are not exactly compatible; "
            f"student={student_hash}, teacher={teacher_hash}"
        )
    return student_hash


def _reject_adapters(config: dict[str, Any]) -> None:
    adapter_keys = {
        key: value
        for key, value in config.items()
        if any(marker in key.lower() for marker in ("lora", "peft", "adapter")) and bool(value)
    }
    if adapter_keys:
        raise ValueError(f"controlled production runs prohibit LoRA/PEFT/adapters: {sorted(adapter_keys)}")


def _ensure_full_parameter_model(model: Any) -> None:
    if "peft" in type(model).__module__.lower() or "peft" in type(model).__name__.lower():
        raise ValueError("controlled production runs prohibit PEFT model wrappers")
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise ValueError(
            "student loader requires full-parameter training, but parameters are frozen: "
            + ", ".join(frozen[:5])
        )


def load_model_and_tokenizer(
    config: dict[str, Any],
    *,
    for_training: bool,
) -> LoadedModel:
    validate_model_revision(config)
    _reject_adapters(config)
    dtype_name = str(config["torch_dtype"])
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported torch_dtype {dtype_name!r}")
    trust_remote_code = bool(config["trust_remote_code"])
    model_revision = str(config["model_revision"])
    tokenizer_revision = str(config["tokenizer_revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["tokenizer_name_or_path"]),
        revision=tokenizer_revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires either a pad token or EOS fallback")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(config["model_name_or_path"]),
        revision=model_revision,
        torch_dtype=_DTYPES[dtype_name],
        attn_implementation=str(config["attn_implementation"]),
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = bool(config["use_cache"])
    if for_training:
        _ensure_full_parameter_model(model)
        if bool(config["gradient_checkpointing"]):
            model.gradient_checkpointing_enable()
        model.train()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=str(config["model_name_or_path"]),
        requested_model_revision=model_revision,
        resolved_model_commit=_resolved_commit(model, model_revision),
        tokenizer_id=str(config["tokenizer_name_or_path"]),
        requested_tokenizer_revision=tokenizer_revision,
        resolved_tokenizer_commit=_resolved_commit(
            tokenizer,
            tokenizer_revision,
        ),
        tokenizer_hash=tokenizer_fingerprint(tokenizer),
    )
