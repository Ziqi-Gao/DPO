"""Deterministic top-level ProofGraph dataset-family manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import (
    CORE_PREREG_VERSION,
    DATASET_SCHEMA_VERSION,
    GENERATOR_VERSION,
    LABEL_SEMANTICS,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.proofgraph.serialization import write_examples_jsonl
from posttrain_circuits.datasets.proofgraph.splits import SPLITS, assert_split_isolation

FAMILY_FORMAT_VERSION = 1
FAMILY_MANIFEST_FIELDS = {
    "format_version",
    "dataset_schema_version",
    "prereg_version",
    "generator_version",
    "label_semantics",
    "splits",
    "sha256",
}
SPLIT_BOUNDARY_FIELDS = {"path", "examples_file_sha256", "num_examples"}


def family_manifest_content(split_boundaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(split_boundaries) != set(SPLITS):
        raise ValueError("ProofGraph family manifest requires exactly seven registered splits")
    return {
        "format_version": FAMILY_FORMAT_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "prereg_version": CORE_PREREG_VERSION,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "splits": {split: split_boundaries[split] for split in SPLITS},
    }


def write_dataset_family(
    root: Path,
    split_examples: dict[str, list[TaskExample]],
) -> dict[str, Any]:
    if set(split_examples) != set(SPLITS):
        raise ValueError("ProofGraph family writer requires exactly seven registered splits")
    ordered = {split: split_examples[split] for split in SPLITS}
    assert_split_isolation(ordered)
    boundaries: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        examples_path = root / split / "examples.jsonl"
        write_examples_jsonl(examples_path, ordered[split])
        boundaries[split] = {
            "path": f"{split}/examples.jsonl",
            "examples_file_sha256": sha256_file(examples_path),
            "num_examples": len(ordered[split]),
        }
    manifest = family_manifest_content(boundaries)
    manifest["sha256"] = sha256_value(manifest)
    atomic_write_json(root / "manifest.json", manifest)
    return manifest
