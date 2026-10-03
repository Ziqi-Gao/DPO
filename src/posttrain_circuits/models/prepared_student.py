"""Exact CPU FP32 loading of a separately identified V3 prepared student.

This low-level loader checks caller-supplied artifact expectations. It does not
establish candidate selection, qualification, G0, or permission to train. The
caller must independently admit those identities before using this model. No
preparation optimizer, scheduler, RNG, or training cursor is restored here.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from posttrain_circuits.artifacts.checkpoints import torch_state_hash
from posttrain_circuits.core.config import (
    QWEN3_CHAT_TEMPLATE_SHA256,
    QWEN3_STUDENT,
    QWEN3_STUDENT_REVISION,
    validate_model_revision,
)
from posttrain_circuits.models.loading import (
    LoadedModel,
    _ensure_full_parameter_model,
    _reject_adapters,
    _resolved_commit,
    _validate_tokenizer_protocol,
)

_TOKENIZER_SHA256 = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
_FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)
_DENSE_FIELDS = {
    "format",
    "model",
    "global_step",
    "parent_checkpoint_sha256",
    "dataset_sha256",
    "protocol_sha256",
    "protocol_artifact_sha256",
    "learning_rate",
    "scope",
    *_FLAGS,
}
_MAX_BYTES = 8 * 1024**3
_KEY_COUNT = 311


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class PreparedStudentCheckpoint:
    """Externally bound expectations, not an acceptance certificate."""

    checkpoint_sha256: str
    checkpoint_size: int
    master_model_state_sha256: str
    checkpoint_step: int
    parent_checkpoint_sha256: str
    dataset_sha256: str
    protocol_sha256: str
    protocol_artifact_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "master_model_state_sha256",
            "parent_checkpoint_sha256",
            "dataset_sha256",
            "protocol_sha256",
            "protocol_artifact_sha256",
        ):
            value = getattr(self, name)
            _require(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                "invalid prepared student " + name,
            )
        _require(
            type(self.checkpoint_size) is int and 0 < self.checkpoint_size <= _MAX_BYTES,
            "prepared student checkpoint size outside the bounded artifact envelope",
        )
        _require(
            type(self.checkpoint_step) is int and self.checkpoint_step in _STEPS,
            "prepared student step is not a V3 export step",
        )


@dataclass(frozen=True, slots=True)
class LoadedPreparedStudent:
    """Trainable FP32 model; native revision fields describe ancestry only."""

    loaded: LoadedModel
    checkpoint: PreparedStudentCheckpoint
    restored_master_model_state_sha256: str
    restored_tensor_count: int
    artifact_load_only: bool = True


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _hash_stream(stream: Any) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _read_checkpoint(path: Path, expected: PreparedStudentCheckpoint) -> dict[str, Any]:
    _require(path.is_absolute() and ".." not in path.parts, "checkpoint path must be absolute and canonical")
    _require(
        not any(part.is_symlink() for part in (path, *path.parents)), "checkpoint path traverses a symlink"
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode), "checkpoint is not a regular file")
        _require(before.st_size == expected.checkpoint_size, "checkpoint size differs from supplied identity")
        _require(_hash_stream(stream) == expected.checkpoint_sha256, "checkpoint SHA-256 differs")
        stream.seek(0)
        # V3 model-only exports contain tensors and primitive metadata only.
        saved = torch.load(stream, map_location="cpu", weights_only=True)
        _require(_hash_stream(stream) == expected.checkpoint_sha256, "checkpoint changed while loading")
        after = os.fstat(stream.fileno())
    _require(
        _file_identity(before) == _file_identity(after) == _file_identity(path.lstat())
        and not any(part.is_symlink() for part in (path, *path.parents)),
        "checkpoint file identity changed while loading",
    )
    _require(isinstance(saved, dict) and set(saved) == _DENSE_FIELDS, "model-only checkpoint fields differ")
    required = {
        "format": "student_branch_dense_v3",
        "global_step": expected.checkpoint_step,
        "parent_checkpoint_sha256": expected.parent_checkpoint_sha256,
        "dataset_sha256": expected.dataset_sha256,
        "protocol_sha256": expected.protocol_sha256,
        "protocol_artifact_sha256": expected.protocol_artifact_sha256,
        "scope": "common_student_branch_model_only",
    }
    _require(all(saved[key] == value for key, value in required.items()), "checkpoint provenance differs")
    _require(type(saved["global_step"]) is int, "checkpoint step must be an integer")
    _require(type(saved["learning_rate"]) is float and saved["learning_rate"] == 5e-5, "V3 export LR differs")
    _require(all(saved[key] is False for key in _FLAGS), "model-only export makes an acceptance claim")
    return saved


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.detach().contiguous().view(torch.uint8), right.detach().contiguous().view(torch.uint8)
    )


def _restore_fp32_masters(model: torch.nn.Module, state: Any, expected_master_sha256: str) -> str:
    """Private CPU restoration seam; tiny models test it without claiming native ancestry."""
    _require(isinstance(state, dict) and len(state) == _KEY_COUNT, "expected exactly 311 saved tensors")
    current = model.state_dict()
    _require(current.keys() == state.keys(), "saved master keys differ from the constructed architecture")
    _ensure_full_parameter_model(model)
    parameters_before = tuple((name, id(p)) for name, p in model.named_parameters(remove_duplicate=False))
    aliases: dict[int, list[str]] = {}
    for name, target in current.items():
        value = state[name]
        _require(
            type(value) is torch.Tensor
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and value.layout == torch.strided
            and not value.is_quantized
            and not value.requires_grad
            and value.shape == target.shape
            and value.numel() > 0,
            "saved master tensor type/device/precision/shape differs: " + name,
        )
        _require(
            target.device.type == "cpu" and target.dtype == torch.float32 and target.layout == torch.strided,
            "constructed model must already contain CPU FP32 masters: " + name,
        )
        _require(bool(torch.isfinite(value).all()), "saved master contains nonfinite values: " + name)
        aliases.setdefault(target.untyped_storage().data_ptr(), []).append(name)
    # load_state_dict copies tied entries in order. Reject disagreement before
    # mutation rather than silently allowing the last alias to win.
    for names in aliases.values():
        first = names[0]
        for name in names[1:]:
            _require(
                current[first].shape == current[name].shape
                and current[first].stride() == current[name].stride()
                and current[first].storage_offset() == current[name].storage_offset()
                and _bits_equal(state[first], state[name]),
                "saved tensors disagree for tied model storage: " + name,
            )
    master_sha256 = torch_state_hash(state)
    _require(master_sha256 == expected_master_sha256, "saved master state hash differs")
    model.load_state_dict(state, strict=True, assign=False)
    restored = model.state_dict()
    _require(
        all(_bits_equal(restored[name], value) for name, value in state.items()), "FP32 reload lost bits"
    )
    _require(torch_state_hash(restored) == master_sha256, "actual restored master hash differs")
    _require(
        tuple((name, id(p)) for name, p in model.named_parameters(remove_duplicate=False))
        == parameters_before,
        "reload replaced parameter objects or tied aliases",
    )
    for names in aliases.values():
        _require(
            len({restored[name].untyped_storage().data_ptr() for name in names}) == 1,
            "reload broke tied parameter storage",
        )
    _ensure_full_parameter_model(model)
    return master_sha256


def _validate_base_config(config: dict[str, Any]) -> None:
    validate_model_revision(config)
    _reject_adapters(config)
    required = {
        "model_name_or_path": QWEN3_STUDENT,
        "tokenizer_name_or_path": QWEN3_STUDENT,
        "model_revision": QWEN3_STUDENT_REVISION,
        "tokenizer_revision": QWEN3_STUDENT_REVISION,
        "tokenizer_fingerprint": _TOKENIZER_SHA256,
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "gradient_checkpointing": True,
        "use_cache": False,
        "trust_remote_code": False,
        "allow_unpinned_revision": False,
    }
    _require(
        all(type(config.get(k)) is type(v) and config.get(k) == v for k, v in required.items()),
        "prepared student base configuration differs from native Qwen3-v2",
    )
    prompt = config.get("prompt_protocol")
    _require(
        isinstance(prompt, dict)
        and prompt
        == {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "messages": "single_user",
            "add_generation_prompt": True,
            "chat_template_sha256": QWEN3_CHAT_TEMPLATE_SHA256,
        }
        and prompt.get("enable_thinking") is False
        and prompt.get("add_generation_prompt") is True,
        "prepared student prompt protocol differs",
    )


def _construct_native_cpu(config: dict[str, Any]) -> LoadedModel:
    """Offline pinned architecture, no redundant native weights or BF16 model copy."""
    architecture = AutoConfig.from_pretrained(
        QWEN3_STUDENT,
        revision=QWEN3_STUDENT_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    _require(
        architecture.model_type == "qwen3"
        and architecture.num_hidden_layers == 28
        and getattr(architecture, "_commit_hash", None) == QWEN3_STUDENT_REVISION
        and not getattr(architecture, "quantization_config", None)
        and not getattr(architecture, "auto_map", None),
        "offline architecture is not the pinned native Qwen3 student",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN3_STUDENT,
        revision=QWEN3_STUDENT_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer_hash, protocol = _validate_tokenizer_protocol(tokenizer, config)
    _require(
        tokenizer.pad_token_id is not None
        and tokenizer.eos_token_id is not None
        and _resolved_commit(tokenizer, QWEN3_STUDENT_REVISION) == QWEN3_STUDENT_REVISION,
        "pinned tokenizer special tokens or revision differ",
    )
    model = AutoModelForCausalLM.from_config(
        architecture,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    _require(type(model).__name__ == "Qwen3ForCausalLM", "constructed student has an unsupported wrapper")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=QWEN3_STUDENT,
        requested_model_revision=QWEN3_STUDENT_REVISION,
        resolved_model_commit=QWEN3_STUDENT_REVISION,
        tokenizer_id=QWEN3_STUDENT,
        requested_tokenizer_revision=QWEN3_STUDENT_REVISION,
        resolved_tokenizer_commit=QWEN3_STUDENT_REVISION,
        tokenizer_hash=tokenizer_hash,
        chat_template_sha256=QWEN3_CHAT_TEMPLATE_SHA256,
        prompt_protocol=protocol,
    )


def load_prepared_student_for_training(
    base_config: dict[str, Any],
    *,
    checkpoint_path: Path | str,
    expected: PreparedStudentCheckpoint,
) -> LoadedPreparedStudent:
    """Restore exact FP32 CPU parameters before a caller constructs AdamW/FSDP.

    ``base_config`` remains unchanged, including its forward BF16 policy. This
    loader returns trainable FP32 masters; mixed precision belongs to the engine.
    Supplied artifact expectations must already be admitted by the caller. This
    function performs no qualification, G0, optimizer, or GPU operation.
    """
    _require(
        type(expected) is PreparedStudentCheckpoint, "explicit immutable checkpoint expectations required"
    )
    _require(isinstance(base_config, dict), "base configuration must be a mapping")
    _require(
        not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
        "prepared student load must occur outside autocast",
    )
    config = copy.deepcopy(base_config)
    _validate_base_config(config)
    # Construction must neither consume the method's CPU RNG nor initialize CUDA.
    # Explicit CPU placement also defeats a caller's non-CPU default device.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        saved = _read_checkpoint(Path(checkpoint_path), expected)
        loaded = _construct_native_cpu(config)
        master_sha256 = _restore_fp32_masters(
            loaded.model, saved["model"], expected.master_model_state_sha256
        )
    _require(
        loaded.model.training
        and loaded.model.is_gradient_checkpointing
        and loaded.model.config.use_cache is False,
        "prepared model training mode differs",
    )
    return LoadedPreparedStudent(
        loaded=loaded,
        checkpoint=expected,
        restored_master_model_state_sha256=master_sha256,
        restored_tensor_count=_KEY_COUNT,
    )
