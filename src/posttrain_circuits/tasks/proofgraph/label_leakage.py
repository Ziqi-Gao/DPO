"""Deterministic surface-label leakage audit for paired ProofGraph."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

import torch

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.tasks.proofgraph.generator import (
    GENERATOR_VERSION,
    LABEL_SEMANTICS,
    ProofGraphTask,
)
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample

LEAKAGE_AUDIT_VERSION = "proofgraph-label-leakage-v1"
_WORD = re.compile(r"[A-Za-z_]+|\d+|[^\w\s]", re.UNICODE)


def _accuracy(predictions: list[int], labels: list[int]) -> float:
    if not labels or len(predictions) != len(labels):
        raise ValueError("leakage accuracy requires aligned nonempty values")
    return sum(int(left == right) for left, right in zip(predictions, labels, strict=True)) / len(labels)


def _group_split(examples: list[TaskExample]) -> tuple[list[TaskExample], list[TaskExample]]:
    groups: dict[str, list[TaskExample]] = defaultdict(list)
    for example in examples:
        if not example.pair_group_id:
            raise ValueError("leakage audit requires pair_group_id on every example")
        groups[example.pair_group_id].append(example)
    if len(groups) < 5:
        raise ValueError("leakage audit requires at least five semantic pairs")
    ordered = sorted(groups, key=lambda value: sha256_value([LEAKAGE_AUDIT_VERSION, value]))
    validation_count = max(1, len(ordered) // 5)
    validation_groups = set(ordered[:validation_count])
    train = [example for example in examples if example.pair_group_id not in validation_groups]
    validation = [example for example in examples if example.pair_group_id in validation_groups]
    if {example.pair_group_id for example in train} & {example.pair_group_id for example in validation}:
        raise RuntimeError("leakage baseline split separated pair siblings")
    return train, validation


def _ridge_predict(
    train_features: torch.Tensor,
    train_labels: list[int],
    validation_features: torch.Tensor,
    *,
    ridge: float = 1.0,
) -> list[int]:
    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("ridge features must be matrices")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    train = torch.cat(
        [train_features.double(), torch.ones((train_features.shape[0], 1), dtype=torch.double)],
        dim=1,
    )
    validation = torch.cat(
        [validation_features.double(), torch.ones((validation_features.shape[0], 1), dtype=torch.double)],
        dim=1,
    )
    targets = torch.tensor(train_labels, dtype=torch.double) * 2.0 - 1.0
    if train.shape[1] <= train.shape[0]:
        identity = torch.eye(train.shape[1], dtype=torch.double)
        weights = torch.linalg.solve(train.T @ train + ridge * identity, train.T @ targets)
    else:
        identity = torch.eye(train.shape[0], dtype=torch.double)
        dual = torch.linalg.solve(train @ train.T + ridge * identity, targets)
        weights = train.T @ dual
    scores = validation @ weights
    return [int(value >= 0.0) for value in scores.tolist()]


def _surface_features(example: TaskExample, prompt: str) -> list[float]:
    identifiers = [*example.facts, *example.rules]
    query_text = str(example.query)
    query_positions = [match.start() for match in re.finditer(re.escape(query_text), prompt)]
    digits = sum(character.isdigit() for character in prompt)
    underscores = prompt.count("_")
    return [
        float(len(query_text)),
        float(len(prompt)),
        float(len(prompt.split())),
        float(len(example.facts)),
        float(len(example.rules)),
        float(len(identifiers)),
        float(sum(map(len, identifiers))),
        float(digits),
        float(underscores),
        float(len(query_positions)),
        float(query_positions[0] if query_positions else -1),
        float(query_positions[-1] if query_positions else -1),
    ]


def _bow_matrices(
    train_prompts: list[str],
    validation_prompts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    vocabulary = sorted({token for prompt in train_prompts for token in _WORD.findall(prompt.lower())})
    index = {token: position for position, token in enumerate(vocabulary)}

    def matrix(prompts: list[str]) -> torch.Tensor:
        result = torch.zeros((len(prompts), len(vocabulary)), dtype=torch.double)
        for row, prompt in enumerate(prompts):
            for token, count in Counter(_WORD.findall(prompt.lower())).items():
                if token in index:
                    result[row, index[token]] = float(count)
        return result

    return matrix(train_prompts), matrix(validation_prompts), vocabulary


def _query_only_accuracy(train: list[TaskExample], validation: list[TaskExample]) -> float:
    by_query: dict[str, list[int]] = defaultdict(list)
    for example in train:
        by_query[str(example.query)].append(example.label)
    majority = int(sum(example.label for example in train) * 2 >= len(train))
    predictions = []
    for example in validation:
        labels = by_query.get(str(example.query), [])
        predictions.append(int(sum(labels) * 2 >= len(labels)) if labels else majority)
    return _accuracy(predictions, [example.label for example in validation])


def audit_label_leakage(
    examples: list[TaskExample],
    *,
    maximum_query_only_accuracy: float = 0.55,
    maximum_surface_feature_accuracy: float = 0.55,
    maximum_bow_accuracy: float = 0.60,
    dataset_hash: str = "unspecified",
    code_commit: str = "unavailable",
    prereg_commit: str = "unavailable",
) -> dict[str, Any]:
    """Audit core prompts without using proof semantics as predictor features."""

    if not examples:
        raise ValueError("label-leakage audit requires examples")
    if any(
        not 0.5 <= value <= 1.0
        for value in (
            maximum_query_only_accuracy,
            maximum_surface_feature_accuracy,
            maximum_bow_accuracy,
        )
    ):
        raise ValueError("leakage thresholds must be in [0.5, 1]")
    task = ProofGraphTask()
    prompts = {example.example_id: task.render(example) for example in examples}
    groups: dict[str, list[TaskExample]] = defaultdict(list)
    for example in examples:
        groups[example.pair_group_id].append(example)

    unprovable_absent = all("UNPROVABLE" not in prompt for prompt in prompts.values())
    pair_checks: list[dict[str, Any]] = []
    for pair_group_id, siblings in sorted(groups.items()):
        labels = sorted(example.label for example in siblings)
        reference = siblings[0]
        query_occurrences = [prompts[example.example_id].count(str(example.query)) for example in siblings]
        passed = (
            len(siblings) == 2
            and labels == [0, 1]
            and all(example.query == reference.query for example in siblings)
            and all(example.rules == reference.rules for example in siblings)
            and all(len(example.facts) == len(reference.facts) for example in siblings)
            and all(len(example.rules) == len(reference.rules) for example in siblings)
            and all(
                example.metadata.get("topology_hash") == reference.metadata.get("topology_hash")
                for example in siblings
            )
            and all(
                example.metadata.get("distractors") == reference.metadata.get("distractors")
                for example in siblings
            )
            and len({len(example.canonical_proof) for example in siblings}) == 1
            and len(set(query_occurrences)) == 1
        )
        pair_checks.append(
            {
                "pair_group_id": pair_group_id,
                "passed": passed,
                "query": str(reference.query),
                "rule_set_hash": reference.metadata.get("rule_set_hash"),
                "topology_hash": reference.metadata.get("topology_hash"),
                "query_occurrences": query_occurrences,
            }
        )

    def distribution(field: str) -> dict[int, Counter[str]]:
        output: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
        for example in examples:
            value: Any
            if field == "query":
                value = str(example.query)
            elif field == "query_occurrences":
                value = prompts[example.example_id].count(str(example.query))
            elif field == "fact_count":
                value = len(example.facts)
            elif field == "rule_count":
                value = len(example.rules)
            elif field == "proof_length":
                value = len(example.canonical_proof)
            else:
                value = example.metadata.get(field)
            output[example.label][str(value)] += 1
        return output

    matched_fields = (
        "query",
        "query_occurrences",
        "fact_count",
        "rule_count",
        "topology_hash",
        "proof_depth",
        "proof_length",
        "distractors",
    )
    distributions = {field: distribution(field) for field in matched_fields}
    distribution_checks = {
        field: distributions[field][0] == distributions[field][1] for field in matched_fields
    }

    train, validation = _group_split(examples)
    train_prompts = [prompts[example.example_id] for example in train]
    validation_prompts = [prompts[example.example_id] for example in validation]
    train_labels = [example.label for example in train]
    validation_labels = [example.label for example in validation]
    surface_train = torch.tensor(
        [_surface_features(example, prompts[example.example_id]) for example in train],
        dtype=torch.double,
    )
    surface_validation = torch.tensor(
        [_surface_features(example, prompts[example.example_id]) for example in validation],
        dtype=torch.double,
    )
    surface_accuracy = _accuracy(
        _ridge_predict(surface_train, train_labels, surface_validation), validation_labels
    )
    bow_train, bow_validation, vocabulary = _bow_matrices(train_prompts, validation_prompts)
    bow_accuracy = _accuracy(_ridge_predict(bow_train, train_labels, bow_validation), validation_labels)
    query_accuracy = _query_only_accuracy(train, validation)
    checks = {
        "unprovable_absent": unprovable_absent,
        "all_pairs_valid": bool(pair_checks) and all(row["passed"] for row in pair_checks),
        "query_distribution_matched": distribution_checks["query"],
        "query_occurrence_matched": distribution_checks["query_occurrences"],
        "surface_distributions_matched": all(distribution_checks.values()),
        "query_only_below_threshold": query_accuracy <= maximum_query_only_accuracy,
        "surface_feature_below_threshold": surface_accuracy <= maximum_surface_feature_accuracy,
        "bow_below_threshold": bow_accuracy <= maximum_bow_accuracy,
    }
    payload: dict[str, Any] = {
        "format_version": 1,
        "audit_version": LEAKAGE_AUDIT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "prereg_version": "core_v2",
        "dataset_hash": dataset_hash,
        "code_commit": code_commit,
        "prereg_commit": prereg_commit,
        "num_examples": len(examples),
        "num_pair_groups": len(groups),
        "baseline_split": {
            "method": "pair_group_hash_80_20",
            "train_pair_hash": sha256_value(sorted({example.pair_group_id for example in train})),
            "validation_pair_hash": sha256_value(sorted({example.pair_group_id for example in validation})),
            "train_examples": len(train),
            "validation_examples": len(validation),
        },
        "thresholds": {
            "maximum_query_only_accuracy": maximum_query_only_accuracy,
            "maximum_surface_feature_accuracy": maximum_surface_feature_accuracy,
            "maximum_bow_accuracy": maximum_bow_accuracy,
        },
        "metrics": {
            "query_only_accuracy": query_accuracy,
            "surface_feature_accuracy": surface_accuracy,
            "bow_accuracy": bow_accuracy,
            "bow_vocabulary_size": len(vocabulary),
        },
        "distribution_checks": distribution_checks,
        "pair_checks_hash": sha256_value(pair_checks),
        "checks": checks,
        "passed": all(checks.values()),
        "examples_hash": sha256_value([asdict(example) for example in examples]),
    }
    payload["sha256"] = sha256_value(payload)
    return payload


def validate_label_leakage_artifact(
    artifact: dict[str, Any],
    *,
    expected_dataset_hash: str | None = None,
) -> dict[str, Any]:
    expected = artifact.get("sha256")
    content = {key: value for key, value in artifact.items() if key != "sha256"}
    if expected != sha256_value(content):
        raise ValueError("label-leakage artifact hash mismatch")
    from posttrain_circuits.core.scientific_versions import require_core_v2_artifact

    require_core_v2_artifact(artifact)
    if artifact.get("audit_version") != LEAKAGE_AUDIT_VERSION:
        raise ValueError("unsupported label-leakage audit version")
    if expected_dataset_hash is not None and artifact.get("dataset_hash") != expected_dataset_hash:
        raise ValueError("label-leakage dataset binding mismatch")
    if artifact.get("passed") is not True:
        raise ValueError("label-leakage audit did not pass")
    return artifact
