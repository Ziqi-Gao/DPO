"""Canonical JSONL serialization for ProofGraph examples."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.datasets.proofgraph.contracts import (
    Literal,
    ProofStep,
    Rule,
    TaskExample,
)


def serialize_examples(examples: list[TaskExample]) -> list[dict[str, Any]]:
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
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("ProofGraph example metadata must be a mapping")
    return TaskExample(
        example_id=str(row["example_id"]),
        facts=facts,
        rules=rules,
        query=Literal(**row["query"]),
        label=int(row["label"]),
        canonical_proof=proof,
        pair_group_id=str(row.get("pair_group_id", metadata.get("pair_group_id", ""))),
        metadata=dict(metadata),
    )


def _strict_json_object(text: str, *, context: str) -> dict[str, Any]:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_from_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite value {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def read_examples_jsonl(path: Path) -> tuple[list[TaskExample], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"ProofGraph examples file is not UTF-8: {path}") from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"ProofGraph examples file is empty or contains blank rows: {path}")
    rows = [
        _strict_json_object(line, context=f"{path} line {index}")
        for index, line in enumerate(lines, start=1)
    ]
    try:
        examples = [deserialize_example(row) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ProofGraph examples file has an invalid row: {path}") from error
    return examples, len(lines)


def write_examples_jsonl(path: Path, examples: list[TaskExample]) -> None:
    if not examples:
        raise ValueError("ProofGraph family splits cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = serialize_examples(examples)
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")
