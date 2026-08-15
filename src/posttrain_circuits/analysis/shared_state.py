"""Shared canonical-prefix and separately marked natural-rollout datasets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json

SourceMode = Literal["canonical_prefix", "natural_rollout"]
DatasetSplit = Literal["discovery", "validation"]


@dataclass(frozen=True)
class SharedStateRecord:
    record_id: str
    task: str
    prompt: str
    prefix: str
    continuation: str
    source_mode: SourceMode
    split: DatasetSplit
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.record_id or not self.task:
            raise ValueError("shared-state records need non-empty IDs and tasks")
        if not self.prompt or not self.prefix or not self.continuation:
            raise ValueError("prompt, prefix, and continuation must be non-empty")
        if self.source_mode not in ("canonical_prefix", "natural_rollout"):
            raise ValueError(f"unknown source mode: {self.source_mode}")
        if self.split not in ("discovery", "validation"):
            raise ValueError(f"unknown dataset split: {self.split}")

    @property
    def model_input(self) -> str:
        return self.prompt + self.prefix


def _partition_payload(
    records: list[SharedStateRecord],
    source_mode: SourceMode,
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"{source_mode} partition must not be empty")
    if any(record.source_mode != source_mode for record in records):
        raise ValueError("canonical-prefix and natural-rollout records must never mix")
    ids = [record.record_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate record IDs in {source_mode} partition")
    rows = [asdict(record) for record in records]
    return {
        "schema_version": 1,
        "source_mode": source_mode,
        "record_count": len(rows),
        "split_counts": {
            split: sum(record.split == split for record in records) for split in ("discovery", "validation")
        },
        "records_sha256": sha256_value(rows),
        "records": rows,
    }


def write_partitioned_shared_state(
    output_dir: Path,
    *,
    canonical_prefix: list[SharedStateRecord],
    natural_rollout: list[SharedStateRecord],
) -> dict[str, Any]:
    """Write source modes to different files and bind them with a root manifest."""
    canonical_ids = {record.record_id for record in canonical_prefix}
    natural_ids = {record.record_id for record in natural_rollout}
    overlap = canonical_ids & natural_ids
    if overlap:
        raise ValueError(f"source partitions reuse record IDs: {sorted(overlap)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "canonical_prefix": _partition_payload(
            canonical_prefix,
            "canonical_prefix",
        ),
        "natural_rollout": _partition_payload(
            natural_rollout,
            "natural_rollout",
        ),
    }
    files = {
        "canonical_prefix": "canonical_prefix.json",
        "natural_rollout": "natural_rollout.json",
    }
    for source_mode, filename in files.items():
        atomic_write_json(output_dir / filename, payloads[source_mode])
    manifest = {
        "schema_version": 1,
        "partitions": {
            source_mode: {
                "file": files[source_mode],
                "source_mode": source_mode,
                "record_count": payloads[source_mode]["record_count"],
                "records_sha256": payloads[source_mode]["records_sha256"],
            }
            for source_mode in files
        },
    }
    manifest["manifest_sha256"] = sha256_value(manifest["partitions"])
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_shared_state_partition(
    path: Path,
    *,
    expected_mode: SourceMode,
) -> list[SharedStateRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_mode") != expected_mode:
        raise ValueError(
            f"expected {expected_mode}, found {payload.get('source_mode')}",
        )
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("shared-state partition has no records")
    if payload.get("record_count") != len(rows):
        raise ValueError("shared-state record count mismatch")
    if payload.get("records_sha256") != sha256_value(rows):
        raise ValueError("shared-state partition hash mismatch")
    records = [SharedStateRecord(**row) for row in rows]
    if any(record.source_mode != expected_mode for record in records):
        raise ValueError("shared-state partition contains a mixed source mode")
    return records


def validate_shared_state_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("shared-state manifest has no partitions")
    if set(partitions) != {"canonical_prefix", "natural_rollout"}:
        raise ValueError("shared-state manifest must bind exactly two source modes")
    if manifest.get("manifest_sha256") != sha256_value(partitions):
        raise ValueError("shared-state manifest hash mismatch")
    for source_mode in ("canonical_prefix", "natural_rollout"):
        entry = partitions[source_mode]
        records = load_shared_state_partition(
            path.parent / entry["file"],
            expected_mode=source_mode,
        )
        if entry["record_count"] != len(records):
            raise ValueError(f"{source_mode} manifest count mismatch")
        if entry["records_sha256"] != sha256_value(
            [asdict(record) for record in records],
        ):
            raise ValueError(f"{source_mode} manifest content mismatch")
    return manifest
