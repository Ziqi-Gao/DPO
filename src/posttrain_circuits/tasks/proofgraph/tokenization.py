"""Semantic token-span audits using every rendered occurrence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json


def _occurrences(text: str, value: str) -> list[tuple[int, int]]:
    if not value:
        raise ValueError("semantic field text cannot be empty")
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(value, cursor)
        if start < 0:
            return spans
        spans.append((start, start + len(value)))
        cursor = start + max(1, len(value))


def tokenization_audit(
    tokenizer: Any,
    text: str,
    semantic_fields: dict[str, list[str]],
    *,
    model_family: str,
) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
    spans: dict[str, list[dict[str, Any]]] = {}
    for role, values in semantic_fields.items():
        spans[role] = []
        for value_index, value in enumerate(dict.fromkeys(values)):
            occurrences = _occurrences(text, value)
            if not occurrences:
                raise ValueError(f"semantic field {value!r} is absent from rendered text")
            for occurrence_index, (start, end) in enumerate(occurrences):
                token_indices = [
                    index
                    for index, (token_start, token_end) in enumerate(offsets)
                    if token_end > start and token_start < end
                ]
                if not token_indices:
                    raise ValueError(f"semantic field {value!r} maps to no tokenizer span")
                spans[role].append(
                    {
                        "text": value,
                        "semantic_value_index": value_index,
                        "occurrence_index": occurrence_index,
                        "character_span": [start, end],
                        "token_span": [
                            min(token_indices),
                            max(token_indices) + 1,
                        ],
                        "token_indices": token_indices,
                        "token_ids": [encoded["input_ids"][index] for index in token_indices],
                    }
                )
    payload = {
        "model_family": model_family,
        "num_tokens": len(encoded["input_ids"]),
        "text_hash": sha256_value(text),
        "spans": spans,
    }
    payload["sha256"] = sha256_value(payload)
    return payload


def semantic_token_indices(
    audit: dict[str, Any],
    roles: list[str],
) -> tuple[int, ...]:
    indices = {
        int(index)
        for role in roles
        for span in audit["spans"].get(role, [])
        for index in span["token_indices"]
    }
    if not indices:
        raise ValueError(f"tokenization audit has no positions for roles {roles}")
    return tuple(sorted(indices))


def write_tokenization_audit(
    path: Path,
    audit: dict[str, Any],
) -> None:
    atomic_write_json(path, audit)
