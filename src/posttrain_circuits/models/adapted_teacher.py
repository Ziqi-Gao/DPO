"""Offline loading of a separately identified, content-verified dense teacher.

This does not relax the original Hugging Face loader's adapter prohibition or
establish scientific acceptance. The expected identity is the canonical payload
hash in ``dense-manifest.json``, not that JSON file's serialization hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.core.config import (
    QWEN3_TEACHER as BASE_MODEL_ID,
)
from posttrain_circuits.core.config import (
    QWEN3_TEACHER_REVISION as BASE_REVISION,
)
from posttrain_circuits.core.config import (
    validate_model_revision,
)
from posttrain_circuits.models.loading import _reject_adapters, _validate_tokenizer_protocol
from posttrain_circuits.models.prompt_protocol import chat_template_sha256

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_MANIFEST_KEYS = {
    "artifact_kind",
    "base_revision",
    "adaptation_plan_sha256",
    "files",
    "formal_teacher_accepted",
    "training_provenance",
    "sha256",
}
_JSON_LIMIT = 4 * 1024 * 1024


@dataclass(frozen=True)
class LoadedAdaptedTeacher:
    model: Any
    tokenizer: Any
    teacher_checkpoint_sha256: str
    base_revision: str
    tokenizer_id: str
    requested_tokenizer_revision: str
    tokenizer_hash: str
    chat_template_sha256: str
    prompt_protocol: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_file(path: Path, *, contents: bool = False) -> tuple[str, tuple[int, ...], bytes]:
    """Hash one regular file without following a symlink or accepting a live write."""
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f"checkpoint file is not regular: {path}")
    if contents:
        _require(before.st_size <= _JSON_LIMIT, f"checkpoint JSON is too large: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    chunks, digest = [], hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        _require(
            _identity(os.fstat(stream.fileno())) == _identity(before),
            f"checkpoint file changed before reading: {path}",
        )
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            if contents:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
    _require(
        _identity(before) == _identity(after) == _identity(path.lstat()),
        f"checkpoint file changed while reading: {path}",
    )
    return digest.hexdigest(), _identity(after), b"".join(chunks)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_unique_object)
    _require(isinstance(value, dict), "checkpoint JSON must be an object")
    return value


def _inventory(root: Path) -> tuple[set[str], dict[str, tuple[int, ...]]]:
    paths: set[str] = set()
    identities = {}
    for folder in ("merged", "adapter"):
        start = root / folder
        _require(stat.S_ISDIR(start.lstat().st_mode), f"missing real {folder} directory")
        for current, directories, filenames in os.walk(start, followlinks=False):
            for path in [Path(current), *(Path(current) / name for name in directories + filenames)]:
                observed = path.lstat()
                _require(
                    stat.S_ISDIR(observed.st_mode) or stat.S_ISREG(observed.st_mode),
                    f"checkpoint contains a symlink or special file: {path}",
                )
                relative = path.relative_to(root).as_posix()
                identities[relative] = _identity(observed)
                if stat.S_ISREG(observed.st_mode):
                    paths.add(relative)
    return paths, identities


def _validate_base(config: dict[str, Any]) -> None:
    validate_model_revision(config)
    _reject_adapters(config)
    _require(
        config["model_name_or_path"] == config["tokenizer_name_or_path"] == BASE_MODEL_ID,
        "adapted teacher requires the fixed Qwen3-8B base and tokenizer",
    )
    _require(
        config["model_revision"] == config["tokenizer_revision"] == BASE_REVISION,
        "adapted teacher base revision differs from the fixed base",
    )
    _require(
        config["trust_remote_code"] is False and not config.get("allow_unpinned_revision"),
        "adapted teacher requires pinned local code",
    )
    _require(_sha(config.get("tokenizer_fingerprint")), "missing pinned tokenizer fingerprint")
    protocol = config.get("prompt_protocol")
    _require(
        isinstance(protocol, dict) and _sha(protocol.get("chat_template_sha256")),
        "missing pinned chat template",
    )
    _require(config["torch_dtype"] in _DTYPES, "unsupported adapted teacher dtype")


def _validated_checkpoint(
    root: Path,
    expected: str,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]], tuple[int, ...]]:
    _require(_sha(expected), "expected adapted teacher identity must be SHA-256")
    _require(
        root.resolve(strict=True) == root and root.is_dir(),
        "checkpoint root or an ancestor is a symlink or not a directory",
    )
    _, manifest_identity, raw = _read_file(root / "dense-manifest.json", contents=True)
    manifest = _json(raw)
    _require(set(manifest) == _MANIFEST_KEYS, "adapted teacher manifest schema differs")
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    _require(
        manifest["sha256"] == expected == sha256_value(payload), "adapted teacher manifest identity differs"
    )
    _require(
        manifest["artifact_kind"] == "adapted_dense_teacher" and manifest["formal_teacher_accepted"] is False,
        "manifest is not an unaccepted dense-teacher checkpoint",
    )
    _require(manifest["base_revision"] == BASE_REVISION, "manifest base revision differs")
    _require(_sha(manifest["adaptation_plan_sha256"]), "invalid adaptation plan identity")
    _require(
        manifest["training_provenance"] is None or isinstance(manifest["training_provenance"], dict),
        "invalid training provenance",
    )
    files = manifest["files"]
    _require(isinstance(files, list) and 0 < len(files) <= 4096, "invalid checkpoint inventory")
    listed: dict[str, dict[str, Any]] = {}
    for item in files:
        _require(
            isinstance(item, dict) and set(item) == {"path", "size", "sha256"},
            "invalid checkpoint file record",
        )
        name = item["path"]
        _require(isinstance(name, str), "invalid checkpoint file path")
        path = PurePosixPath(name)
        _require(
            len(path.parts) >= 2
            and path.parts[0] in {"merged", "adapter"}
            and ".." not in path.parts
            and str(path) == name
            and "\\" not in name,
            "unsafe checkpoint file path",
        )
        _require(name not in listed, "duplicate checkpoint file path")
        _require(
            type(item["size"]) is int and item["size"] >= 0 and _sha(item["sha256"]),
            "invalid checkpoint size or hash",
        )
        listed[name] = item
    observed, identities = _inventory(root)
    _require(observed == set(listed), "checkpoint has missing or extra files")
    _require(
        any(name.startswith("merged/") and name.endswith(".safetensors") for name in listed),
        "checkpoint has no dense safetensors",
    )
    for name, item in listed.items():
        digest, identity, _ = _read_file(root / name)
        _require(
            identity == identities[name] and identity[3] == item["size"] and digest == item["sha256"],
            f"checkpoint bytes differ: {name}",
        )
    return manifest, identities, manifest_identity


def _validate_dense_metadata(root: Path) -> None:
    for name in ("config.json", "tokenizer_config.json"):
        _, _, raw = _read_file(root / "merged" / name, contents=True)
        value = _json(raw)
        _require(not value.get("auto_map"), "dense teacher cannot load custom code")
        if name == "config.json":
            _require(
                value.get("model_type") == "qwen3" and value.get("architectures") == ["Qwen3ForCausalLM"],
                "dense teacher architecture must be Qwen3ForCausalLM",
            )
            _require(not value.get("_commit_hash"), "adapted weights must not claim a Hub commit")
            _require(
                not value.get("quantization_config") and not value.get("peft_config"),
                "dense teacher cannot load quantization or adapters",
            )
    for path in (root / "merged").rglob("*"):
        _require(
            path.name != "adapter_config.json" and path.suffix not in {".bin", ".pt", ".pth"},
            "dense teacher contains an adapter entry point or unsafe weight format",
        )
    index_path = root / "merged" / "model.safetensors.index.json"
    if index_path.exists():
        _, _, raw = _read_file(index_path, contents=True)
        mapping = _json(raw).get("weight_map")
        _require(isinstance(mapping, dict) and bool(mapping), "invalid dense safetensors index")
        for name in mapping.values():
            _require(
                isinstance(name, str)
                and PurePosixPath(name).name == name
                and name.endswith(".safetensors")
                and (root / "merged" / name).is_file(),
                "unsafe or missing dense safetensors shard",
            )


def load_adapted_teacher(
    checkpoint_root: Path | str,
    expected_manifest_sha256: str,
    teacher_base_config: dict[str, Any],
    *,
    device: Any,
) -> LoadedAdaptedTeacher:
    """Verify every saved file, then load a frozen dense teacher entirely offline.

    ``base_revision`` describes ancestry only. ``teacher_checkpoint_sha256`` is
    the actual adapted identity and must be bound by every downstream consumer.
    The caller supplies its assigned device; this function never selects a GPU.
    """
    _validate_base(teacher_base_config)
    root = Path(os.path.abspath(checkpoint_root))
    manifest, identities, manifest_identity = _validated_checkpoint(root, expected_manifest_sha256)
    _validate_dense_metadata(root)
    tokenizer = AutoTokenizer.from_pretrained(
        root / "merged",
        local_files_only=True,
        trust_remote_code=False,
    )
    # Never silently change the saved tokenizer before checking its identity.
    fingerprint, protocol = _validate_tokenizer_protocol(tokenizer, teacher_base_config)
    _require(
        tokenizer.pad_token_id is not None and tokenizer.eos_token_id is not None,
        "saved teacher tokenizer lacks padding or EOS",
    )
    model, info = AutoModelForCausalLM.from_pretrained(
        root / "merged",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
        torch_dtype=_DTYPES[teacher_base_config["torch_dtype"]],
        attn_implementation=str(teacher_base_config["attn_implementation"]),
        low_cpu_mem_usage=bool(teacher_base_config.get("low_cpu_mem_usage", True)),
    )
    _require(
        not any(
            info.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ),
        "dense teacher weights do not exactly cover the model",
    )
    _require(
        type(model).__name__ == "Qwen3ForCausalLM" and model.config.model_type == "qwen3",
        "loaded dense teacher is not Qwen3ForCausalLM",
    )
    _require(
        not getattr(model, "peft_config", None)
        and not any("peft" in type(module).__module__.lower() for module in model.modules())
        and not any("lora_" in name for name, _ in model.named_parameters()),
        "loaded dense teacher contains adapters",
    )
    model.config.use_cache = bool(teacher_base_config["use_cache"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    _require(
        _inventory(root)[1] == identities
        and _identity((root / "dense-manifest.json").lstat()) == manifest_identity,
        "checkpoint changed while loading",
    )
    return LoadedAdaptedTeacher(
        model=model,
        tokenizer=tokenizer,
        teacher_checkpoint_sha256=manifest["sha256"],
        base_revision=manifest["base_revision"],
        tokenizer_id=teacher_base_config["tokenizer_name_or_path"],
        requested_tokenizer_revision=teacher_base_config["tokenizer_revision"],
        tokenizer_hash=fingerprint,
        chat_template_sha256=chat_template_sha256(tokenizer),
        prompt_protocol=protocol,
    )
