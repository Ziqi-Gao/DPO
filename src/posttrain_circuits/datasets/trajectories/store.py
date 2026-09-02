"""Immutable trajectory store with exact file and row inventories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_file
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.datasets.trajectories.codec import (
    read_records,
    validate_payload,
    write_payload,
)
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.datasets.trajectories.manifests import (
    MANIFEST_FILENAME,
    build_store_manifest,
    expected_payload_files,
    validate_file_inventory,
    validate_store_manifest,
)


def _require_unique_trajectory_ids(values: list[Any], *, context: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{context} contains an empty or non-string trajectory_id")
    if len(values) != len(set(values)):
        raise ValueError(f"{context} contains duplicate trajectory_id values")


def _has_teacher_scores(record: TrajectoryRecord) -> bool:
    return any(
        (
            record.teacher_topk_ids,
            record.teacher_topk_logprobs,
            record.teacher_topk_mass,
            record.teacher_entropy,
        )
    )


class TrajectoryStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(
        self,
        records: list[TrajectoryRecord],
        *,
        behavior_policy: dict[str, Any],
        prompt_manifest_hash: str,
        sampling_configuration: dict[str, Any],
        verifier_version: str,
        teacher_version: str | None,
        top_k: int,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not records:
            raise ValueError("trajectory store cannot be empty")
        trajectory_ids = [record.trajectory_id for record in records]
        _require_unique_trajectory_ids(trajectory_ids, context="trajectory store write")
        for record in records:
            record.validate()
        protocols = {record.sampling_protocol_id for record in records}
        if len(protocols) != 1:
            raise ValueError("trajectory store cannot mix sampling protocols")
        scored_status = {_has_teacher_scores(record) for record in records}
        if len(scored_status) != 1:
            raise ValueError("trajectory store cannot mix teacher-scored and unscored records")
        teacher_scored = scored_status == {True}
        if teacher_scored:
            if top_k < 1 or not teacher_version:
                raise ValueError("teacher-scored records require teacher_version and positive top_k")
            for record in records:
                if any(len(position) != top_k for position in record.teacher_topk_ids):
                    raise ValueError("teacher top-k width differs from the requested store top_k")
        elif top_k != 0 or teacher_version is not None:
            raise ValueError("unscored records forbid teacher_version and teacher top_k")
        if self.root.exists():
            if not self.root.is_dir() or any(self.root.iterdir()):
                raise FileExistsError(f"trajectory store output is not empty: {self.root}")
        else:
            self.root.mkdir(parents=True)

        row_counts = write_payload(
            self.root,
            records,
            teacher_scored=teacher_scored,
            teacher_top_k=top_k,
        )
        expected_files = expected_payload_files(teacher_scored=teacher_scored)
        file_sha256 = {
            name: sha256_file(self.root / name)
            for name in sorted(expected_files)
        }
        rewards = [float(record.verifier_reward or 0.0) for record in records]
        lengths = [len(record.response_ids) for record in records]
        manifest = build_store_manifest(
            behavior_policy=behavior_policy,
            prompt_manifest_hash=prompt_manifest_hash,
            sampling_configuration=sampling_configuration,
            sampling_protocol_id=next(iter(protocols)),
            verifier_version=verifier_version,
            teacher_version=teacher_version,
            top_k=top_k,
            ordered_trajectory_ids=trajectory_ids,
            teacher_scored=teacher_scored,
            file_sha256=file_sha256,
            row_counts=row_counts,
            reward_distribution={
                "mean": sum(rewards) / len(rewards),
                "positive": sum(value > 0 for value in rewards),
            },
            length_distribution={
                "minimum": min(lengths),
                "maximum": max(lengths),
                "mean": sum(lengths) / len(lengths),
            },
            effective_supervised_tokens=sum(
                sum(record.response_token_mask) for record in records
            ),
            extra_metadata=extra_metadata,
        )
        atomic_write_json(self.root / MANIFEST_FILENAME, manifest)
        self.check_integrity()
        return manifest

    def check_integrity(self) -> dict[str, Any]:
        manifest_path = self.root / MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("trajectory store manifest is missing or not a regular file")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("trajectory store manifest is not readable JSON") from error
        if not isinstance(manifest, dict):
            raise ValueError("trajectory store manifest must be a JSON object")
        validate_store_manifest(manifest)
        validate_file_inventory(self.root, manifest)
        observed_ids = validate_payload(self.root, manifest)
        expected_ids = tuple(manifest["ordered_trajectory_ids"])
        if observed_ids != expected_ids:
            raise ValueError("trajectory-store metadata order differs from manifest inventory")
        return manifest

    def read(self) -> list[TrajectoryRecord]:
        manifest = self.check_integrity()
        records = read_records(self.root, manifest)
        ids = [record.trajectory_id for record in records]
        _require_unique_trajectory_ids(ids, context="trajectory store read")
        if ids != list(manifest["ordered_trajectory_ids"]):
            raise ValueError("trajectory-store decoded record order differs from manifest inventory")
        return records
