"""Pure-data trajectory-store manifest construction and validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import (
    ROLLOUT_GENERATION_VERSION,
    TRAJECTORY_STORE_VERSION,
    require_scientific_artifact,
    scientific_compatibility_fields,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import utc_now
from posttrain_circuits.datasets.trajectories.identities import SAMPLING_CURSOR_PROTOCOL_ID

MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata-00000.parquet"
TOKENS_FILENAME = "tokens-00000.safetensors"
TEACHER_FILENAME = "teacher-00000.safetensors"
BASE_PAYLOAD_FILES = frozenset({METADATA_FILENAME, TOKENS_FILENAME})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def expected_payload_files(*, teacher_scored: bool) -> frozenset[str]:
    if teacher_scored:
        return BASE_PAYLOAD_FILES | {TEACHER_FILENAME}
    return BASE_PAYLOAD_FILES


def _require_unique_trajectory_ids(values: Sequence[object], *, context: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{context} contains an empty or non-string trajectory_id")
    result = tuple(str(value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{context} contains duplicate trajectory_id values")
    return result


def manifest_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in manifest.items()
        if key not in {"created_at", "sha256"}
    }


def build_store_manifest(
    *,
    behavior_policy: Mapping[str, Any],
    prompt_manifest_hash: str,
    sampling_configuration: Mapping[str, Any],
    sampling_protocol_id: str,
    verifier_version: str,
    teacher_version: str | None,
    top_k: int,
    ordered_trajectory_ids: Sequence[str],
    teacher_scored: bool,
    file_sha256: Mapping[str, str],
    row_counts: Mapping[str, int],
    reward_distribution: Mapping[str, Any],
    length_distribution: Mapping[str, Any],
    effective_supervised_tokens: int,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ids = _require_unique_trajectory_ids(
        ordered_trajectory_ids,
        context="trajectory store manifest",
    )
    metadata = dict(extra_metadata or {})
    prereg_version = str(metadata.pop("prereg_version", "core_v2"))
    manifest: dict[str, Any] = {
        "format_version": TRAJECTORY_STORE_VERSION,
        **scientific_compatibility_fields(prereg_version),
        "behavior_policy": dict(behavior_policy),
        "prompt_manifest_hash": prompt_manifest_hash,
        "sampling_configuration": dict(sampling_configuration),
        "sampling_protocol_id": sampling_protocol_id,
        "sampling_cursor_protocol_id": SAMPLING_CURSOR_PROTOCOL_ID,
        "verifier_version": verifier_version,
        "teacher_version": teacher_version,
        "teacher_scored": teacher_scored,
        "top_k": top_k,
        "total_trajectories": len(ids),
        "ordered_trajectory_ids": list(ids),
        "reward_distribution": dict(reward_distribution),
        "length_distribution": dict(length_distribution),
        "effective_supervised_tokens": effective_supervised_tokens,
        "files": dict(file_sha256),
        "row_counts": dict(row_counts),
        "created_at": utc_now(),
    }
    overlap = set(manifest) & set(metadata)
    if overlap:
        raise ValueError(
            f"extra trajectory-store metadata overwrites reserved keys: {sorted(overlap)}"
        )
    manifest.update(metadata)
    manifest["sha256"] = sha256_value(manifest_hash_payload(manifest))
    validate_store_manifest(manifest)
    return manifest


def validate_store_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("format_version") != TRAJECTORY_STORE_VERSION:
        raise ValueError(
            f"trajectory store format must be exactly version {TRAJECTORY_STORE_VERSION}"
        )
    require_scientific_artifact(
        manifest,
        expected_prereg_version=str(manifest.get("prereg_version", "")),
    )
    total = manifest.get("total_trajectories")
    if type(total) is not int or total < 1:
        raise ValueError("trajectory store total_trajectories must be a positive integer")
    raw_ids = manifest.get("ordered_trajectory_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("trajectory store requires ordered_trajectory_ids")
    ids = _require_unique_trajectory_ids(raw_ids, context="trajectory store manifest")
    if len(ids) != total:
        raise ValueError("ordered trajectory inventory differs from total_trajectories")
    teacher_scored = manifest.get("teacher_scored")
    if type(teacher_scored) is not bool:
        raise ValueError("trajectory store teacher_scored must be boolean")
    expected_files = expected_payload_files(teacher_scored=teacher_scored)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != expected_files:
        raise ValueError(
            "trajectory store file inventory is not exact: "
            f"expected={sorted(expected_files)}, observed={sorted(files) if isinstance(files, Mapping) else files}"
        )
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError("trajectory store files must map exact names to SHA-256 digests")
    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, Mapping) or set(row_counts) != expected_files:
        raise ValueError("trajectory store row-count inventory is not exact")
    if any(type(value) is not int or value != total for value in row_counts.values()):
        raise ValueError("every trajectory payload file must declare the exact trajectory row count")
    sampling_protocol_id = manifest.get("sampling_protocol_id")
    if not isinstance(sampling_protocol_id, str) or not sampling_protocol_id:
        raise ValueError("trajectory store requires a sampling_protocol_id")
    if manifest.get("sampling_cursor_protocol_id") != SAMPLING_CURSOR_PROTOCOL_ID:
        raise ValueError("trajectory store sampling cursor protocol is incompatible")
    top_k = manifest.get("top_k")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("trajectory store top_k must be a non-negative integer")
    teacher_version = manifest.get("teacher_version")
    if teacher_scored:
        if top_k < 1 or not isinstance(teacher_version, str) or not teacher_version:
            raise ValueError("teacher-scored stores require teacher_version and positive top_k")
    elif top_k != 0 or teacher_version is not None:
        raise ValueError("unscored stores forbid teacher_version and teacher top_k")
    if manifest.get("store_kind") in {"rollout_bank", "teacher_scored_rollout_bank"}:
        observed = manifest.get("rollout_generation_version")
        if observed != ROLLOUT_GENERATION_VERSION:
            raise ValueError(
                "rollout bank uses an incompatible sampling/identity protocol: "
                f"expected={ROLLOUT_GENERATION_VERSION}, observed={observed}"
            )
    expected_hash = sha256_value(manifest_hash_payload(manifest))
    if manifest.get("sha256") != expected_hash:
        raise ValueError("trajectory-store manifest hash does not match its contents")


def validate_file_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    validate_store_manifest(manifest)
    expected = {MANIFEST_FILENAME, *manifest["files"]}
    observed = {entry.name for entry in root.iterdir()}
    if observed != expected:
        raise ValueError(
            "trajectory store directory inventory is not exact: "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )
    for name, expected_digest in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"trajectory store payload is not a regular file: {name}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"hash mismatch for {name}: expected {expected_digest}, got {actual_digest}"
            )
