"""Deterministic, disjoint ProofGraph split generation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

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

