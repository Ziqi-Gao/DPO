"""Fail-closed loader for one complete seven-split ProofGraph family."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import (
    CORE_PREREG_VERSION,
    DATASET_SCHEMA_VERSION,
    GENERATOR_VERSION,
    LABEL_SEMANTICS,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.proofgraph.manifests import (
    FAMILY_FORMAT_VERSION,
    FAMILY_MANIFEST_FIELDS,
    SPLIT_BOUNDARY_FIELDS,
)
from posttrain_circuits.datasets.proofgraph.serialization import read_examples_jsonl
from posttrain_circuits.datasets.proofgraph.splits import SPLITS, assert_split_isolation


@dataclass(frozen=True)
class DatasetFamilyHandle:
    root: Path
    manifest: dict[str, Any]
    splits: dict[str, list[TaskExample]]

    def examples(self, split: str) -> list[TaskExample]:
        if split not in SPLITS:
            raise ValueError(f"unknown ProofGraph family split {split!r}")
        return self.splits[split]

    def boundary(self, split: str) -> dict[str, Any]:
        if split not in SPLITS:
            raise ValueError(f"unknown ProofGraph family split {split!r}")
        return dict(self.manifest["splits"][split])


def _strict_manifest(path: Path) -> dict[str, Any]:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"ProofGraph family manifest contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_from_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"ProofGraph family manifest contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ProofGraph family manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("ProofGraph family manifest must be a JSON object")
    return value


def _require_exact_tree(root: Path) -> None:
    expected_root_entries = {"manifest.json", *SPLITS}
    observed_root_entries = {path.name for path in root.iterdir()}
    if observed_root_entries != expected_root_entries:
        raise ValueError(
            "ProofGraph family root entries differ from the seven-split contract: "
            f"missing={sorted(expected_root_entries - observed_root_entries)}, "
            f"extra={sorted(observed_root_entries - expected_root_entries)}"
        )
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not stat.S_ISREG(os.lstat(manifest_path).st_mode):
        raise ValueError("ProofGraph family manifest must be a regular non-symlink file")
    for split in SPLITS:
        split_root = root / split
        if split_root.is_symlink() or not stat.S_ISDIR(os.lstat(split_root).st_mode):
            raise ValueError(f"ProofGraph family split must be a real directory: {split}")
        entries = list(split_root.iterdir())
        if len(entries) != 1 or entries[0].name != "examples.jsonl":
            raise ValueError(f"ProofGraph family split {split} must contain only examples.jsonl")
        examples_path = split_root / "examples.jsonl"
        if examples_path.is_symlink() or not stat.S_ISREG(os.lstat(examples_path).st_mode):
            raise ValueError(f"ProofGraph family split file is not regular: {split}")


def _validate_examples(split_examples: dict[str, list[TaskExample]]) -> None:
    example_owner: dict[str, str] = {}
    pair_rows: dict[str, list[tuple[str, int]]] = {}
    for split in SPLITS:
        for example in split_examples[split]:
            if not example.example_id or example.example_id in example_owner:
                raise ValueError(
                    f"ProofGraph family duplicate/malformed example_id: {example.example_id!r}"
                )
            example_owner[example.example_id] = split
            if type(example.label) is not int or example.label not in {0, 1}:
                raise ValueError(f"ProofGraph family example has invalid label: {example.example_id}")
            if not example.pair_group_id:
                raise ValueError(
                    f"ProofGraph family example lacks pair_group_id: {example.example_id}"
                )
            pair_rows.setdefault(example.pair_group_id, []).append((split, example.label))
    for pair_group_id, rows in pair_rows.items():
        if len(rows) != 2 or {label for _, label in rows} != {0, 1}:
            raise ValueError(f"ProofGraph family pair is incomplete: {pair_group_id}")
        if len({split for split, _ in rows}) != 1:
            raise ValueError(f"ProofGraph family pair crosses splits: {pair_group_id}")
    assert_split_isolation(split_examples)


def load_dataset_family(root: Path) -> DatasetFamilyHandle:
    lexical_root = Path(os.path.abspath(os.fspath(root)))
    if lexical_root.is_symlink() or not lexical_root.is_dir():
        raise ValueError(f"ProofGraph dataset family root is not a real directory: {root}")
    _require_exact_tree(lexical_root)
    manifest = _strict_manifest(lexical_root / "manifest.json")
    if set(manifest) != FAMILY_MANIFEST_FIELDS:
        raise ValueError("ProofGraph family manifest fields differ from schema")
    constants = {
        "format_version": FAMILY_FORMAT_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "prereg_version": CORE_PREREG_VERSION,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
    }
    if any(manifest.get(key) != value for key, value in constants.items()):
        raise ValueError("ProofGraph family scientific identity changed")
    content = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != sha256_value(content):
        raise ValueError("ProofGraph family manifest SHA-256 mismatch")
    boundaries = manifest.get("splits")
    if not isinstance(boundaries, dict) or set(boundaries) != set(SPLITS):
        raise ValueError("ProofGraph family manifest does not bind exactly seven splits")

    split_examples: dict[str, list[TaskExample]] = {}
    for split in SPLITS:
        boundary = boundaries[split]
        if not isinstance(boundary, dict) or set(boundary) != SPLIT_BOUNDARY_FIELDS:
            raise ValueError(f"ProofGraph family split boundary changed: {split}")
        expected_path = f"{split}/examples.jsonl"
        if boundary.get("path") != expected_path:
            raise ValueError(f"ProofGraph family split path changed: {split}")
        examples_path = lexical_root / expected_path
        if boundary.get("examples_file_sha256") != sha256_file(examples_path):
            raise ValueError(f"ProofGraph family split bytes changed: {split}")
        examples, row_count = read_examples_jsonl(examples_path)
        expected_count = boundary.get("num_examples")
        if type(expected_count) is not int or expected_count < 1 or expected_count != row_count:
            raise ValueError(f"ProofGraph family split row count changed: {split}")
        split_examples[split] = examples
    _validate_examples(split_examples)
    return DatasetFamilyHandle(lexical_root, manifest, split_examples)
