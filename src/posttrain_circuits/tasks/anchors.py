"""Deterministic anchor pilots with fixed discovery and validation sets."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.tasks.math_bridge.answer_normalization import (
    normalize_answer,
)

AnchorTask = Literal["greater_than", "small_addition", "entity_tracking"]
AnchorSplit = Literal["discovery", "validation"]


class BaseAccuracyBelowThreshold(RuntimeError):
    pass


@dataclass(frozen=True)
class AnchorExample:
    anchor_id: str
    task: AnchorTask
    split: AnchorSplit
    prompt: str
    canonical_solution: str
    answer: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AnchorAccuracy:
    threshold: float
    overall: float
    by_task: dict[str, float]
    passed: bool


def _greater_than(
    rng: random.Random,
    index: int,
    split: AnchorSplit,
) -> AnchorExample:
    left = rng.randint(10, 98)
    right = rng.randint(10, 98)
    while right == left:
        right = rng.randint(10, 98)
    answer = str(max(left, right))
    solution = f"Compare {left} and {right}. The greater integer is {answer}. Final answer: {answer}"
    return AnchorExample(
        f"greater-than-{split}-{index:05d}",
        "greater_than",
        split,
        f"Which integer is greater, {left} or {right}? Answer with one integer.",
        solution,
        answer,
        {"left": left, "right": right},
    )


def _small_addition(
    rng: random.Random,
    index: int,
    split: AnchorSplit,
) -> AnchorExample:
    left = rng.randint(0, 49)
    right = rng.randint(0, 49)
    answer = str(left + right)
    solution = f"Compute {left} + {right} = {answer}. Final answer: {answer}"
    return AnchorExample(
        f"small-addition-{split}-{index:05d}",
        "small_addition",
        split,
        f"What is {left} + {right}? Answer with one integer.",
        solution,
        answer,
        {"left": left, "right": right, "operator": "+"},
    )


def _entity_tracking(
    rng: random.Random,
    index: int,
    split: AnchorSplit,
) -> AnchorExample:
    names = ["Ava", "Ben", "Cora", "Drew", "Eli", "Faye"]
    places = ["amber", "blue", "green", "red", "silver", "violet"]
    selected_names = rng.sample(names, 3)
    selected_places = rng.sample(places, 3)
    tracked_index = rng.randrange(3)
    destination_index = (tracked_index + rng.choice([1, 2])) % 3
    tracked = selected_names[tracked_index]
    destination = selected_places[destination_index]
    facts = " ".join(
        f"{name} starts in the {place} room."
        for name, place in zip(selected_names, selected_places, strict=True)
    )
    prompt = (
        f"{facts} Then {tracked} moves to the {destination} room. "
        f"Where is {tracked}? Answer with the room color."
    )
    solution = (
        f"{tracked} started in {selected_places[tracked_index]} and then moved "
        f"to {destination}. Final answer: {destination}"
    )
    return AnchorExample(
        f"entity-tracking-{split}-{index:05d}",
        "entity_tracking",
        split,
        prompt,
        solution,
        destination,
        {
            "entities": selected_names,
            "initial_places": selected_places,
            "tracked_entity": tracked,
            "destination": destination,
        },
    )


_BUILDERS = {
    "greater_than": _greater_than,
    "small_addition": _small_addition,
    "entity_tracking": _entity_tracking,
}


def build_fixed_anchor_pilots(
    *,
    seed: int,
    discovery_per_task: int,
    validation_per_task: int,
) -> dict[str, list[AnchorExample]]:
    if discovery_per_task < 1 or validation_per_task < 1:
        raise ValueError("anchor discovery and validation sets must be non-empty")
    result: dict[str, list[AnchorExample]] = {
        "discovery": [],
        "validation": [],
    }
    split_counts: tuple[tuple[AnchorSplit, int], ...] = (
        ("discovery", discovery_per_task),
        ("validation", validation_per_task),
    )
    for task_index, (_task, builder) in enumerate(_BUILDERS.items()):
        for split_index, (split, count) in enumerate(split_counts):
            rng = random.Random(seed + task_index * 1_000_003 + split_index * 100_003)
            for index in range(count):
                result[split].append(builder(rng, index, split))
    discovery_prompts = {example.prompt for example in result["discovery"]}
    validation_prompts = {example.prompt for example in result["validation"]}
    overlap = discovery_prompts & validation_prompts
    if overlap:
        raise RuntimeError("anchor discovery/validation prompts overlap")
    return result


def verify_anchor_prediction(
    example: AnchorExample,
    prediction: str,
) -> bool:
    return normalize_answer(prediction) == normalize_answer(example.answer)


def require_base_accuracy(
    examples: list[AnchorExample],
    predictions: dict[str, str],
    *,
    threshold: float,
) -> AnchorAccuracy:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("base accuracy threshold must lie in [0, 1]")
    missing = sorted(example.anchor_id for example in examples if example.anchor_id not in predictions)
    if missing:
        raise ValueError(f"missing anchor predictions: {missing[:5]}")
    by_task: dict[str, float] = {}
    for task in _BUILDERS:
        task_examples = [example for example in examples if example.task == task]
        by_task[task] = sum(
            verify_anchor_prediction(example, predictions[example.anchor_id]) for example in task_examples
        ) / len(task_examples)
    overall = sum(
        verify_anchor_prediction(example, predictions[example.anchor_id]) for example in examples
    ) / len(examples)
    result = AnchorAccuracy(
        threshold=threshold,
        overall=overall,
        by_task=by_task,
        passed=overall >= threshold and all(accuracy >= threshold for accuracy in by_task.values()),
    )
    if not result.passed:
        raise BaseAccuracyBelowThreshold(
            f"base anchor accuracy failed threshold {threshold}: overall={overall:.4f}, by_task={by_task}",
        )
    return result


def write_anchor_pilots(
    output_dir: Path,
    splits: dict[str, list[AnchorExample]],
    *,
    seed: int,
) -> dict[str, Any]:
    if set(splits) != {"discovery", "validation"}:
        raise ValueError("anchor pilot requires discovery and validation splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits: dict[str, Any] = {}
    for split in ("discovery", "validation"):
        examples = splits[split]
        if any(example.split != split for example in examples):
            raise ValueError(f"anchor split label mismatch in {split}")
        rows = [asdict(example) for example in examples]
        payload = {
            "schema_version": 1,
            "split": split,
            "seed": seed,
            "example_count": len(rows),
            "examples_sha256": sha256_value(rows),
            "examples": rows,
        }
        filename = f"{split}.json"
        atomic_write_json(output_dir / filename, payload)
        manifest_splits[split] = {
            "file": filename,
            "example_count": len(rows),
            "examples_sha256": payload["examples_sha256"],
        }
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "tasks": list(_BUILDERS),
        "splits": manifest_splits,
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest
