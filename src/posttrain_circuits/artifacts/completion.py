"""Scheduler-neutral, hash-valid scientific completion markers."""

from __future__ import annotations

import json
import stat
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import publish_json_once


PRODUCTION_ARTIFACT_ROOTS = (
    Path("/data/del6500/OPD"),
    Path("/scr/del6500/OPD"),
)


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _parse_utc_timestamp(value: object, *, name: str) -> datetime:
    text = _require_text(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return parsed


def _validated_roots(approved_roots: Iterable[Path]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for raw_root in approved_roots:
        root = Path(raw_root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError(f"approved artifact root must be an absolute normalized path: {root}")
        try:
            root_status = root.lstat()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"approved artifact root is missing: {root}") from error
        if stat.S_ISLNK(root_status.st_mode):
            raise ValueError(f"approved artifact root must not be a symlink: {root}")
        if not stat.S_ISDIR(root_status.st_mode):
            raise ValueError(f"approved artifact root must be a directory: {root}")
        resolved = root.resolve(strict=True)
        if resolved != root:
            raise ValueError(
                f"approved artifact root must not traverse symlinked ancestors: {root}"
            )
        if root not in roots:
            roots.append(root)
    if not roots:
        raise ValueError("at least one approved artifact root is required")
    return tuple(sorted(roots, key=lambda candidate: len(candidate.parts), reverse=True))


def _lexical_root(path: Path, roots: tuple[Path, ...]) -> tuple[Path, tuple[str, ...]]:
    for root in roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts and ".." not in relative.parts:
            return root, relative.parts
    raise ValueError(f"artifact path is outside approved roots: {path}")


def _validate_parent_chain(
    path: Path,
    *,
    roots: tuple[Path, ...],
) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"artifact path must be an absolute normalized path: {path}")
    root, relative_parts = _lexical_root(path, roots)
    current = root
    for part in relative_parts[:-1]:
        current /= part
        try:
            current_status = current.lstat()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"artifact parent directory is missing: {current}") from error
        if stat.S_ISLNK(current_status.st_mode):
            raise ValueError(f"artifact path must not traverse a symlink: {current}")
        if not stat.S_ISDIR(current_status.st_mode):
            raise ValueError(f"artifact parent must be a directory: {current}")
    resolved_parent = path.parent.resolve(strict=True)
    if resolved_parent != path.parent:
        raise ValueError(f"artifact path must not traverse a symlinked parent: {path.parent}")
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ValueError(f"artifact path escapes its approved root: {path}")
    return root


def _validate_marker_target(path: Path, *, roots: tuple[Path, ...]) -> None:
    _validate_parent_chain(path, roots=roots)
    try:
        target_status = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(target_status.st_mode):
        raise ValueError(f"completion marker target must not be a symlink: {path}")
    if not stat.S_ISREG(target_status.st_mode):
        raise ValueError(f"completion marker target must be a regular file: {path}")


def _validate_output_file(path: Path, *, roots: tuple[Path, ...]) -> None:
    root = _validate_parent_chain(path, roots=roots)
    try:
        target_status = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"declared scientific output is missing: {path}") from error
    if stat.S_ISLNK(target_status.st_mode):
        raise ValueError(f"declared scientific output must not be a symlink: {path}")
    if not stat.S_ISREG(target_status.st_mode):
        raise ValueError(f"declared scientific output must be a regular file: {path}")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"declared scientific output must not traverse a symlink: {path}")
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"declared scientific output escapes its approved root: {path}")


@dataclass(frozen=True)
class ExecutionIdentity:
    job_id: str
    attempt: int
    execution_profile: str
    manifest_sha256: str
    allocation_sha256: str

    def validate(self) -> None:
        _require_text(self.job_id, name="execution.job_id")
        _require_text(self.execution_profile, name="execution.execution_profile")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ValueError("execution.attempt must be a positive integer")
        _require_sha256(self.manifest_sha256, name="execution.manifest_sha256")
        _require_sha256(self.allocation_sha256, name="execution.allocation_sha256")


@dataclass(frozen=True)
class ScientificCompletion:
    workflow_id: str
    plan_sha256: str
    unit_id: str
    task: str
    run_id: str
    started_at: str
    completed_at: str
    scientific_config_sha256: str
    execution_config_sha256: str
    resolved_config_sha256: str
    input_hashes: dict[str, str]
    output_files: dict[str, str]
    scientific_validation: dict[str, bool]
    execution: ExecutionIdentity | None = None
    schema_version: int = 1
    project: str = "OPD"
    completion_kind: str = "scientific"

    def validate(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
            or self.project != "OPD"
        ):
            raise ValueError("unsupported OPD completion schema or project")
        if self.completion_kind != "scientific":
            raise ValueError("completion marker must describe scientific completion")
        _require_text(self.workflow_id, name="workflow_id")
        _require_sha256(self.plan_sha256, name="plan_sha256")
        _require_text(self.unit_id, name="unit_id")
        _require_text(self.task, name="task")
        _require_text(self.run_id, name="run_id")
        started_at = _parse_utc_timestamp(self.started_at, name="started_at")
        completed_at = _parse_utc_timestamp(self.completed_at, name="completed_at")
        if completed_at < started_at:
            raise ValueError("completed_at must not precede started_at")
        for name, digest in (
            ("scientific_config_sha256", self.scientific_config_sha256),
            ("execution_config_sha256", self.execution_config_sha256),
            ("resolved_config_sha256", self.resolved_config_sha256),
        ):
            _require_sha256(digest, name=name)
        if not isinstance(self.input_hashes, dict) or not self.input_hashes:
            raise ValueError("completion marker requires an input-hash mapping")
        if not isinstance(self.output_files, dict) or not self.output_files:
            raise ValueError("completion marker requires a declared-output mapping")
        for name, digest in self.input_hashes.items():
            _require_text(name, name="input hash name")
            _require_sha256(digest, name=f"input_hashes[{name!r}]")
        for raw_path, digest in self.output_files.items():
            if not isinstance(raw_path, str):
                raise ValueError("declared output path must be a string")
            _require_sha256(digest, name=f"output_files[{raw_path!r}]")
            path = Path(raw_path)
            if not path.is_absolute():
                raise ValueError(f"declared output path must be absolute: {path}")
        if not isinstance(self.scientific_validation, dict) or not self.scientific_validation:
            raise ValueError("completion marker requires scientific validation evidence")
        for name in self.scientific_validation:
            _require_text(name, name="scientific validation name")
        failures = [
            name
            for name, passed in self.scientific_validation.items()
            if not isinstance(passed, bool) or passed is not True
        ]
        if failures:
            raise ValueError(f"scientific completion validation failed: {sorted(failures)}")
        if self.execution is not None:
            if not isinstance(self.execution, ExecutionIdentity):
                raise ValueError("scientific completion execution identity is invalid")
            self.execution.validate()


def completion_payload(marker: ScientificCompletion) -> dict[str, Any]:
    marker.validate()
    payload = asdict(marker)
    payload["sha256"] = sha256_value(payload)
    return payload


def _execution_identity(execution_payload: object) -> ExecutionIdentity | None:
    if execution_payload is None:
        return None
    if not isinstance(execution_payload, dict):
        raise ValueError("scientific completion execution identity must be a mapping or null")
    expected_keys = {
        "job_id",
        "attempt",
        "execution_profile",
        "manifest_sha256",
        "allocation_sha256",
    }
    if set(execution_payload) != expected_keys:
        raise ValueError(
            "scientific completion execution fields differ from schema: "
            f"missing={sorted(expected_keys - set(execution_payload))}, "
            f"extra={sorted(set(execution_payload) - expected_keys)}"
        )
    try:
        return ExecutionIdentity(**execution_payload)
    except TypeError as error:
        raise ValueError("scientific completion execution identity is invalid") from error


def _validated_completion_payload(
    payload: dict[str, Any],
    *,
    marker_path: Path,
    roots: tuple[Path, ...],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("scientific completion payload must be a mapping")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != sha256_value(content):
        raise ValueError("scientific completion SHA-256 mismatch")
    expected_keys = {
        "workflow_id",
        "plan_sha256",
        "unit_id",
        "task",
        "run_id",
        "started_at",
        "completed_at",
        "scientific_config_sha256",
        "execution_config_sha256",
        "resolved_config_sha256",
        "input_hashes",
        "output_files",
        "scientific_validation",
        "execution",
        "schema_version",
        "project",
        "completion_kind",
    }
    if set(content) != expected_keys:
        raise ValueError(
            "scientific completion fields differ from schema: "
            f"missing={sorted(expected_keys - set(content))}, "
            f"extra={sorted(set(content) - expected_keys)}"
        )
    execution = _execution_identity(content["execution"])
    marker_content = {key: value for key, value in content.items() if key != "execution"}
    try:
        marker = ScientificCompletion(execution=execution, **marker_content)
    except TypeError as error:
        raise ValueError("scientific completion fields are invalid") from error
    marker.validate()
    _validate_marker_target(marker_path, roots=roots)
    for raw_path, digest in marker.output_files.items():
        output_path = Path(raw_path)
        _validate_output_file(output_path, roots=roots)
        observed = sha256_file(output_path)
        if observed != digest:
            raise ValueError(
                f"declared scientific output hash mismatch: path={output_path}, "
                f"expected={digest}, observed={observed}"
            )
    return payload


def validate_completion_payload(
    payload: dict[str, Any],
    *,
    marker_path: Path,
    approved_roots: Iterable[Path],
) -> dict[str, Any]:
    """Validate marker schema, filesystem containment, and scientific outputs."""

    roots = _validated_roots(approved_roots)
    return _validated_completion_payload(payload, marker_path=Path(marker_path), roots=roots)


def _scientific_identity(payload: dict[str, Any]) -> dict[str, Any]:
    identity_keys = {
        "schema_version",
        "project",
        "completion_kind",
        "workflow_id",
        "plan_sha256",
        "unit_id",
        "task",
        "run_id",
        "scientific_config_sha256",
        "input_hashes",
        "output_files",
        "scientific_validation",
    }
    return {key: payload[key] for key in sorted(identity_keys)}


def _read_existing_marker(path: Path, *, roots: tuple[Path, ...]) -> dict[str, Any]:
    _validate_marker_target(path, roots=roots)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing scientific completion marker is unreadable: {path}") from error
    return _validated_completion_payload(existing, marker_path=path, roots=roots)


def write_completion_marker(
    path: Path,
    marker: ScientificCompletion,
    *,
    approved_roots: Iterable[Path],
) -> dict[str, Any]:
    """Publish once, preserving the first valid execution provenance on retries."""

    marker_path = Path(path)
    roots = _validated_roots(approved_roots)
    payload = completion_payload(marker)
    _validated_completion_payload(payload, marker_path=marker_path, roots=roots)
    try:
        published = publish_json_once(marker_path, payload)
    except FileExistsError:
        existing = _read_existing_marker(marker_path, roots=roots)
        if _scientific_identity(existing) == _scientific_identity(payload):
            return existing
        raise FileExistsError(
            f"conflicting scientific completion marker already exists: {marker_path}"
        )
    observed = _read_existing_marker(marker_path, roots=roots)
    if published != payload:
        raise RuntimeError("scientific completion publication returned unexpected content")
    if observed != payload:
        raise RuntimeError("scientific completion marker changed during publication")
    return observed
