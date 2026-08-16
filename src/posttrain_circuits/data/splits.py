"""Deterministic, disjoint ProofGraph split construction."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.scientific_versions import (
    DATASET_SCHEMA_VERSION,
    require_core_v2_artifact,
)
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import Literal, ProofStep, Rule, TaskExample

SPLITS = (
    "train",
    "validation",
    "iid_test",
    "ood_depth_test",
    "ood_structure_test",
    "circuit_discovery",
    "circuit_validation",
)


def canonical_semantic_key(example: TaskExample) -> str:
    semantic = {
        "facts": sorted(str(value) for value in example.facts.values()),
        "rules": sorted(
            (
                tuple(sorted(str(item) for item in rule.antecedents)),
                str(rule.consequent),
            )
            for rule in example.rules.values()
        ),
        "query": str(example.query),
    }
    return sha256_value(semantic)


def _split_config(
    split: str,
    difficulty: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = dict(difficulty or {})
    if split == "ood_depth_test":
        cfg.pop("depth", None)
        cfg["depth_range"] = cfg.pop(
            "ood_depth_range",
            [5, 7],
        )
    elif split == "ood_structure_test":
        cfg.pop("structure", None)
        cfg["structures"] = ["converging_dag"]
    if split in {"circuit_discovery", "circuit_validation"}:
        cfg["unique_proof"] = True
        cfg["multiple_valid_proofs"] = False
    return cfg


def build_split(
    task: ProofGraphTask,
    split: str,
    num_examples: int,
    base_seed: int,
    difficulty: dict[str, Any] | None = None,
) -> list[TaskExample]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if num_examples < 1:
        raise ValueError("num_examples must be positive")
    if num_examples % 2:
        raise ValueError("paired signed-entailment splits require an even num_examples")
    cfg = _split_config(split, difficulty)
    multiple_fraction = float(cfg.pop("multiple_valid_proof_fraction", 0.0))
    if not 0.0 <= multiple_fraction <= 1.0:
        raise ValueError("multiple_valid_proof_fraction must be in [0, 1]")
    offset = SPLITS.index(split) * 10_000_000
    examples: list[TaskExample] = []
    seen: set[str] = set()
    seed = base_seed + offset
    while len(examples) < num_examples:
        candidate_cfg = dict(cfg)
        if split == "train" and multiple_fraction > 0:
            threshold = int(multiple_fraction * 10_000)
            multiple = int(sha256_value(seed)[:8], 16) % 10_000 < threshold
            candidate_cfg["multiple_valid_proofs"] = multiple
            candidate_cfg["unique_proof"] = not multiple
        positive, negative = task.generate_pair(seed, candidate_cfg)
        pair = [positive, negative]
        if int(sha256_value([split, seed])[:8], 16) % 2:
            pair.reverse()
        keys = [canonical_semantic_key(candidate) for candidate in pair]
        if not (set(keys) & seen):
            examples.extend(pair)
            seen.update(keys)
        seed += 1
        if seed - (base_seed + offset) > num_examples * 100:
            raise RuntimeError("could not generate enough semantically unique examples")
    return examples


def assert_split_isolation(
    splits: dict[str, list[TaskExample]],
) -> None:
    owner: dict[str, str] = {}
    pair_owner: dict[str, str] = {}
    for split, examples in splits.items():
        for example in examples:
            key = canonical_semantic_key(example)
            if key in owner:
                raise ValueError(f"semantic duplicate across {owner[key]} and {split}: {example.example_id}")
            owner[key] = split
            pair_group_id = example.pair_group_id or str(example.metadata.get("pair_group_id", ""))
            if not pair_group_id:
                raise ValueError(f"paired example has no pair_group_id: {example.example_id}")
            if pair_group_id in pair_owner and pair_owner[pair_group_id] != split:
                raise ValueError(
                    f"pair group crosses {pair_owner[pair_group_id]} and {split}: {pair_group_id}"
                )
            pair_owner[pair_group_id] = split


def build_all_splits(
    task: ProofGraphTask,
    *,
    split_sizes: dict[str, int],
    base_seed: int,
    difficulty: dict[str, Any],
) -> dict[str, list[TaskExample]]:
    missing = set(SPLITS) - set(split_sizes)
    if missing:
        raise ValueError(f"all-split build is missing sizes for {sorted(missing)}")
    result = {
        split: build_split(
            task,
            split,
            int(split_sizes[split]),
            base_seed,
            difficulty,
        )
        for split in SPLITS
    }
    assert_split_isolation(result)
    return result


def difficulty_distribution(
    examples: list[TaskExample],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot summarize an empty split")

    def counts(field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(example.metadata[field]) for example in examples).items()))

    return {
        "depth": counts("depth"),
        "structure": counts("structure"),
        "distractors": counts("distractors"),
        "label": dict(sorted(Counter(str(example.label) for example in examples).items())),
        "proof_multiplicity": counts("proof_multiplicity"),
        "pair_group_count": len({example.pair_group_id for example in examples}),
        "pair_group_hash": sha256_value(sorted(example.pair_group_id for example in examples)),
        "topology": counts("topology_hash"),
    }


def serialize_examples(
    examples: list[TaskExample],
) -> list[dict[str, Any]]:
    return [asdict(example) for example in examples]


def deserialize_example(row: dict[str, Any]) -> TaskExample:
    facts = {str(key): Literal(**value) for key, value in row["facts"].items()}
    rules = {
        str(key): Rule(
            rule_id=str(value["rule_id"]),
            antecedents=tuple(Literal(**item) for item in value["antecedents"]),
            consequent=Literal(**value["consequent"]),
        )
        for key, value in row["rules"].items()
    }
    proof = [
        ProofStep(
            step_id=str(value["step_id"]),
            rule_id=str(value["rule_id"]),
            citations=tuple(str(item) for item in value["citations"]),
            conclusion=Literal(**value["conclusion"]),
        )
        for value in row["canonical_proof"]
    ]
    return TaskExample(
        example_id=str(row["example_id"]),
        facts=facts,
        rules=rules,
        query=Literal(**row["query"]),
        label=int(row["label"]),
        canonical_proof=proof,
        pair_group_id=str(row.get("pair_group_id", row.get("metadata", {}).get("pair_group_id", ""))),
        metadata=dict(row.get("metadata", {})),
    )


def load_frozen_split(root: Path, *, expected_split: str) -> tuple[list[TaskExample], dict[str, Any]]:
    """Load exact immutable split bytes and verify the generating manifest."""

    examples_path = root / "examples.jsonl"
    manifest_path = root / "manifest.json"
    if not examples_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"frozen split is incomplete: {root}")
    rows = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split_name") != expected_split:
        raise ValueError(
            f"frozen split mismatch: expected {expected_split}, observed {manifest.get('split_name')}"
        )
    require_core_v2_artifact(manifest)
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("frozen split uses a pre-core-v2 dataset schema")
    if int(manifest.get("num_examples", -1)) != len(rows):
        raise ValueError("frozen split row count does not match manifest")
    content = {key: value for key, value in manifest.items() if key not in {"sha256", "created_at"}}
    expected_hash = sha256_value({"manifest": content, "examples": rows})
    if manifest.get("sha256") != expected_hash:
        raise ValueError("frozen split manifest hash does not match exact example bytes")
    examples = [deserialize_example(row) for row in rows]
    pair_groups = sorted({example.pair_group_id for example in examples})
    if manifest.get("pair_group_count") != len(pair_groups):
        raise ValueError("frozen split pair-group count mismatch")
    if manifest.get("pair_group_hash") != sha256_value(pair_groups):
        raise ValueError("frozen split pair-group hash mismatch")
    manifest["examples_file_sha256"] = sha256_file(examples_path)
    return examples, manifest
