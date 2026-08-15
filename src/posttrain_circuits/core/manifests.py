"""Immutable dataset and run manifests."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from posttrain_circuits.core.hashing import sha256_value


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


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
