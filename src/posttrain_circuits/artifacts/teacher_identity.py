"""Explicit Qwen3 teacher identity, separate from storage and acceptance.

These are metadata contracts, not evidence that checkpoint bytes were read or
that a teacher passed readiness. Consumers must also use the verified dense
loader and a separately accepted scientific protocol. Legacy consumers and
cache formats are deliberately unchanged.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.core.config import (
    QWEN3_CHAT_TEMPLATE_SHA256,
    QWEN3_TEACHER,
    QWEN3_TEACHER_REVISION,
)

QWEN3_TEACHER_TOKENIZER_FINGERPRINT = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = {"pinned_hf_base", "learned_dense_checkpoint"}
_DENSE_MANIFEST_KEYS = {
    "artifact_kind",
    "base_revision",
    "adaptation_plan_sha256",
    "files",
    "formal_teacher_accepted",
    "training_provenance",
    "sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


@dataclass(frozen=True)
class TeacherIdentity:
    format_version: int
    kind: Literal["pinned_hf_base", "learned_dense_checkpoint"]
    base_model_id: str
    base_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_fingerprint: str
    chat_template_sha256: str
    prompt_protocol: str
    enable_thinking: bool
    teacher_checkpoint_sha256: str | None

    def __post_init__(self) -> None:
        _require(
            type(self.format_version) is int and self.format_version == 1,
            "unsupported teacher identity version",
        )
        _require(isinstance(self.kind, str) and self.kind in _KINDS, "unknown teacher weight kind")
        _require(
            self.base_model_id == QWEN3_TEACHER and self.base_revision == QWEN3_TEACHER_REVISION,
            "teacher base must be the genuine pinned Qwen3-8B revision",
        )
        _require(
            self.tokenizer_id == QWEN3_TEACHER
            and self.tokenizer_revision == QWEN3_TEACHER_REVISION
            and self.tokenizer_fingerprint == QWEN3_TEACHER_TOKENIZER_FINGERPRINT,
            "teacher tokenizer identity differs from the pinned tokenizer",
        )
        _require(
            self.chat_template_sha256 == QWEN3_CHAT_TEMPLATE_SHA256
            and self.prompt_protocol == "qwen3_non_thinking_v1"
            and self.enable_thinking is False,
            "teacher identity requires the pinned non-thinking protocol",
        )
        if self.kind == "learned_dense_checkpoint":
            _require(
                _is_sha(self.teacher_checkpoint_sha256),
                "learned teacher requires its actual dense checkpoint SHA-256",
            )
        else:
            _require(
                self.teacher_checkpoint_sha256 is None,
                "pinned HF identity cannot stand in for learned checkpoint bytes",
            )

    @classmethod
    def from_mapping(cls, value: Any) -> TeacherIdentity:
        _require(
            isinstance(value, dict) and set(value) == {field.name for field in fields(cls)},
            "teacher identity has missing or unknown fields",
        )
        return cls(**value)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        """Scientific content identity: no execution paths or acceptance claims."""
        return sha256_value(self.to_mapping())


def pinned_base_teacher_identity() -> TeacherIdentity:
    """Describe the unchanged original HF teacher without relabeling old artifacts."""
    return TeacherIdentity(
        format_version=1,
        kind="pinned_hf_base",
        base_model_id=QWEN3_TEACHER,
        base_revision=QWEN3_TEACHER_REVISION,
        tokenizer_id=QWEN3_TEACHER,
        tokenizer_revision=QWEN3_TEACHER_REVISION,
        tokenizer_fingerprint=QWEN3_TEACHER_TOKENIZER_FINGERPRINT,
        chat_template_sha256=QWEN3_CHAT_TEMPLATE_SHA256,
        prompt_protocol="qwen3_non_thinking_v1",
        enable_thinking=False,
        teacher_checkpoint_sha256=None,
    )


def adapted_teacher_identity(
    manifest: Any,
    *,
    expected_manifest_sha256: str,
) -> TeacherIdentity:
    """Bind validated manifest metadata, without claiming actual byte verification.

    The expected value is ``adaptation.checkpoint_manifest()['sha256']``: the
    canonical payload hash, not a raw JSON-file hash, HF commit or Git commit.
    The dense loader must separately verify every actual saved file and the
    tokenizer against these pins before using this identity for model outputs.
    """
    _require(_is_sha(expected_manifest_sha256), "expected dense manifest identity must be SHA-256")
    _require(
        isinstance(manifest, dict) and set(manifest) == _DENSE_MANIFEST_KEYS,
        "dense teacher manifest has missing or unknown fields",
    )
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    _require(
        manifest["sha256"] == expected_manifest_sha256 == sha256_value(payload),
        "dense teacher manifest identity differs",
    )
    _require(
        manifest["artifact_kind"] == "adapted_dense_teacher" and manifest["formal_teacher_accepted"] is False,
        "dense teacher manifest cannot issue its own acceptance",
    )
    _require(
        manifest["base_revision"] == QWEN3_TEACHER_REVISION, "dense teacher manifest base revision differs"
    )
    _require(_is_sha(manifest["adaptation_plan_sha256"]), "invalid adaptation plan identity")
    _require(
        manifest["training_provenance"] is None or isinstance(manifest["training_provenance"], dict),
        "invalid training provenance metadata",
    )
    records = manifest["files"]
    _require(isinstance(records, list) and 0 < len(records) <= 4096, "invalid dense file inventory")
    names = set()
    for record in records:
        _require(
            isinstance(record, dict) and set(record) == {"path", "size", "sha256"},
            "invalid dense file record",
        )
        name = record["path"]
        _require(isinstance(name, str), "invalid dense file path")
        path = PurePosixPath(name)
        _require(
            len(path.parts) >= 2
            and path.parts[0] in {"merged", "adapter"}
            and ".." not in path.parts
            and str(path) == name
            and "\\" not in name
            and not any(ord(character) < 32 for character in name),
            "unsafe dense file path",
        )
        _require(name not in names, "duplicate dense file path")
        _require(
            type(record["size"]) is int and record["size"] >= 0 and _is_sha(record["sha256"]),
            "invalid dense file size or hash",
        )
        names.add(name)
    _require(
        any(name.startswith("merged/") and name.endswith(".safetensors") for name in names),
        "manifest has no dense safetensors",
    )
    identity = pinned_base_teacher_identity().to_mapping()
    identity.update(kind="learned_dense_checkpoint", teacher_checkpoint_sha256=expected_manifest_sha256)
    return TeacherIdentity.from_mapping(identity)


def _validated_identity(identity: TeacherIdentity) -> TeacherIdentity:
    _require(isinstance(identity, TeacherIdentity), "expected a validated TeacherIdentity")
    return TeacherIdentity.from_mapping(identity.to_mapping())


def teacher_science_bindings(identity: TeacherIdentity) -> dict[str, Any]:
    """Explicit new bindings; never emit base-only learned-model revision aliases."""
    identity = _validated_identity(identity)
    return {
        "teacher_identity_version": identity.format_version,
        "teacher_identity_sha256": identity.sha256,
        "teacher_weight_kind": identity.kind,
        "teacher_base_model_id": identity.base_model_id,
        "teacher_base_revision": identity.base_revision,
        "teacher_checkpoint_sha256": identity.teacher_checkpoint_sha256,
        "teacher_tokenizer_id": identity.tokenizer_id,
        "teacher_tokenizer_revision": identity.tokenizer_revision,
        "tokenizer_fingerprint": identity.tokenizer_fingerprint,
        "chat_template_sha256": identity.chat_template_sha256,
        "prompt_protocol": identity.prompt_protocol,
        "enable_thinking": identity.enable_thinking,
    }


def teacher_content_cache_identity(identity: TeacherIdentity) -> dict[str, Any]:
    """A new cache namespace; this does not migrate or mutate legacy cache entries."""
    identity = _validated_identity(identity)
    return {
        "cache_namespace": "opd_teacher_scores_with_weight_identity_v1",
        "teacher_identity_sha256": identity.sha256,
        "teacher_identity": identity.to_mapping(),
    }


def teacher_content_cache_key(
    identity: TeacherIdentity,
    *,
    input_ids: list[int] | tuple[int, ...],
    response_ids: list[int] | tuple[int, ...],
    top_k: int,
) -> str:
    """Address scores by exact tokens, top-k and the actual learned-weight identity."""
    for tokens in (input_ids, response_ids):
        _require(
            isinstance(tokens, list | tuple)
            and bool(tokens)
            and all(type(value) is int and value >= 0 for value in tokens),
            "cache requires nonempty nonnegative integer token sequences",
        )
    _require(type(top_k) is int and top_k > 0, "cache top_k must be a positive integer")
    return sha256_value(
        {
            **teacher_content_cache_identity(identity),
            "input_ids": list(input_ids),
            "response_ids": list(response_ids),
            "top_k": top_k,
        }
    )


@dataclass(frozen=True)
class TeacherCheckpointLocator:
    """Execution-only locator. Construction validates syntax, not storage contents."""

    checkpoint_root: str

    def __post_init__(self) -> None:
        value = self.checkpoint_root
        _require(
            isinstance(value, str) and bool(value) and not any(ord(character) < 32 for character in value),
            "invalid checkpoint locator",
        )
        path = PurePosixPath(value)
        _require(
            path.is_absolute()
            and str(path) == value
            and ".." not in path.parts
            and not value.startswith("//")
            and "\\" not in value,
            "checkpoint locator must be an absolute canonical POSIX path",
        )

    @classmethod
    def from_mapping(cls, value: Any) -> TeacherCheckpointLocator:
        _require(
            isinstance(value, dict) and set(value) == {"checkpoint_root"},
            "checkpoint locator has missing or unknown fields",
        )
        return cls(**value)


def validate_checkpoint_locator(locator: TeacherCheckpointLocator, *, allowed_root: Path | str) -> Path:
    """Check local storage boundaries; the caller supplies its reviewed allowed root.

    No file content is read and no path becomes a scientific input. Call this on
    the machine consuming the checkpoint; then use the content-verifying loader.
    """
    _require(isinstance(locator, TeacherCheckpointLocator), "expected a TeacherCheckpointLocator")
    TeacherCheckpointLocator.from_mapping(asdict(locator))
    allowed = Path(allowed_root)
    _require(
        allowed.is_absolute() and allowed.resolve(strict=True) == allowed and allowed.is_dir(),
        "allowed checkpoint root must be a real absolute directory",
    )
    path = Path(locator.checkpoint_root)
    _require(
        path.resolve(strict=True) == path and path.is_dir(),
        "checkpoint locator must name an existing directory without symlinks",
    )
    _require(path.is_relative_to(allowed), "checkpoint locator is outside the allowed execution root")
    manifest = path / "dense-manifest.json"
    _require(
        not manifest.is_symlink() and manifest.is_file(), "checkpoint locator lacks a real dense manifest"
    )
    return path
