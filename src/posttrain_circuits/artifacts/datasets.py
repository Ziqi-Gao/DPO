"""Immutable dataset manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json, utc_now


@dataclass
class DatasetManifest:
    dataset_id: str
    generator_version: str
    git_commit: str
    task_config: dict[str, Any]
    split_name: str
    seed_range: tuple[int, int]
    num_examples: int
    difficulty_distribution: dict[str, Any]
    dataset_schema_version: str = "proofgraph-dataset-v2-paired"
    label_semantics: str = "signed_entailment"
    prereg_version: str = "core_v2"
    pair_group_count: int = 0
    pair_group_hash: str = ""
    sha256: str = ""
    created_at: str = field(default_factory=utc_now)

    def content_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("sha256", None)
        payload.pop("created_at", None)
        return payload

    def finalize(self, examples: list[Any]) -> DatasetManifest:
        self.sha256 = sha256_value({"manifest": self.content_payload(), "examples": examples})
        return self

    def write(self, path: Path) -> None:
        if not self.sha256:
            raise ValueError("manifest must be finalized before writing")
        atomic_write_json(path, asdict(self))
