"""Load only user-provided, content-bound curated math manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from posttrain_circuits.core.hashing import sha256_file


def load_curated_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset_id", "examples_file", "sha256", "deduplicated"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"math manifest is missing keys: {sorted(missing)}")
    if not manifest["deduplicated"]:
        raise ValueError("math bridge requires a curated and deduplicated manifest")
    examples_path = path.parent / manifest["examples_file"]
    if sha256_file(examples_path) != manifest["sha256"]:
        raise ValueError("math examples hash does not match manifest")
    return manifest
