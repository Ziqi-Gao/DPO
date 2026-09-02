"""Parquet and safetensors codec for trajectory-store payload files."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.datasets.trajectories.manifests import (
    METADATA_FILENAME,
    TEACHER_FILENAME,
    TOKENS_FILENAME,
)

TOKEN_TENSOR_KEYS = frozenset(
    {
        "input_ids",
        "input_ids_offsets",
        "response_ids",
        "response_ids_offsets",
        "response_token_mask",
        "response_token_mask_offsets",
        "behavior_logprobs",
        "behavior_logprobs_offsets",
    }
)
TEACHER_TENSOR_KEYS = frozenset(
    {"topk_ids", "topk_logprobs", "topk_mass", "entropy", "trajectory_offsets"}
)


def _require_tensor_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise RuntimeError(
            "TrajectoryStore requires the 'train' extra: pip install -e '.[train]'"
        ) from error
    return torch, load_file, save_file


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "TrajectoryStore requires the 'train' extra: pip install -e '.[train]'"
        ) from error
    return pa, pq


def _concat_ragged(values: list[list[Any]], *, dtype: Any, torch: Any) -> tuple[Any, Any]:
    offsets = [0]
    flattened: list[Any] = []
    for value in values:
        flattened.extend(value)
        offsets.append(len(flattened))
    return torch.tensor(flattened, dtype=dtype), torch.tensor(offsets, dtype=torch.long)


def write_payload(
    root: Path,
    records: list[TrajectoryRecord],
    *,
    teacher_scored: bool,
    teacher_top_k: int,
) -> dict[str, int]:
    torch, _, save_file = _require_tensor_dependencies()
    pa, pq = _require_pyarrow()
    token_tensors: dict[str, Any] = {}
    for name in ("input_ids", "response_ids"):
        flattened, offsets = _concat_ragged(
            [getattr(record, name) for record in records],
            dtype=torch.long,
            torch=torch,
        )
        token_tensors[name] = flattened
        token_tensors[f"{name}_offsets"] = offsets
    mask, mask_offsets = _concat_ragged(
        [[bool(value) for value in record.response_token_mask] for record in records],
        dtype=torch.bool,
        torch=torch,
    )
    token_tensors["response_token_mask"] = mask
    token_tensors["response_token_mask_offsets"] = mask_offsets
    logprobs, logprob_offsets = _concat_ragged(
        [record.behavior_logprobs for record in records],
        dtype=torch.float32,
        torch=torch,
    )
    token_tensors["behavior_logprobs"] = logprobs
    token_tensors["behavior_logprobs_offsets"] = logprob_offsets
    save_file(token_tensors, root / TOKENS_FILENAME)

    omitted = {
        "input_ids",
        "response_ids",
        "response_token_mask",
        "behavior_logprobs",
        "teacher_topk_ids",
        "teacher_topk_logprobs",
        "teacher_topk_mass",
        "teacher_entropy",
    }
    metadata_rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: value for key, value in asdict(record).items() if key not in omitted}
        row["verification_trace"] = json.dumps(row["verification_trace"], sort_keys=True)
        metadata_rows.append(row)
    pq.write_table(pa.Table.from_pylist(metadata_rows), root / METADATA_FILENAME)

    row_counts = {
        METADATA_FILENAME: len(records),
        TOKENS_FILENAME: len(records),
    }
    if teacher_scored:
        positions = [position for record in records for position in record.teacher_topk_ids]
        logprobs = [position for record in records for position in record.teacher_topk_logprobs]
        if any(len(position) != teacher_top_k for position in positions):
            raise ValueError("teacher top-k width differs from the store manifest")
        teacher_offsets = [0]
        for record in records:
            teacher_offsets.append(teacher_offsets[-1] + len(record.teacher_topk_ids))
        teacher_tensors = {
            "topk_ids": torch.tensor(positions, dtype=torch.long),
            "topk_logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "topk_mass": torch.tensor(
                [value for record in records for value in record.teacher_topk_mass],
                dtype=torch.float32,
            ),
            "entropy": torch.tensor(
                [value for record in records for value in record.teacher_entropy],
                dtype=torch.float32,
            ),
            "trajectory_offsets": torch.tensor(teacher_offsets, dtype=torch.long),
        }
        save_file(teacher_tensors, root / TEACHER_FILENAME)
        row_counts[TEACHER_FILENAME] = len(records)
    return row_counts


def _offset_values(
    values: Any,
    offsets: Any,
    *,
    rows: int,
    name: str,
) -> list[int]:
    if values.ndim < 1 or offsets.ndim != 1:
        raise ValueError(f"trajectory-store {name} tensors have invalid rank")
    raw_offsets = [int(value) for value in offsets.tolist()]
    if len(raw_offsets) != rows + 1:
        raise ValueError(f"trajectory-store {name} offsets do not match metadata rows")
    if raw_offsets[0] != 0 or any(
        current > following for current, following in zip(raw_offsets, raw_offsets[1:])
    ):
        raise ValueError(f"trajectory-store {name} offsets are not monotonic from zero")
    if raw_offsets[-1] != int(values.shape[0]):
        raise ValueError(f"trajectory-store {name} terminal offset differs from tensor length")
    return raw_offsets


def _load_validated_payload(
    root: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    _, load_file, _ = _require_tensor_dependencies()
    _, pq = _require_pyarrow()
    rows = pq.read_table(root / METADATA_FILENAME).to_pylist()
    total = int(manifest["total_trajectories"])
    if len(rows) != total:
        raise ValueError("trajectory-store metadata row count differs from manifest")
    tokens = load_file(root / TOKENS_FILENAME)
    if set(tokens) != TOKEN_TENSOR_KEYS:
        raise ValueError("trajectory-store token tensor inventory is not exact")
    offsets: dict[str, list[int]] = {}
    for name in ("input_ids", "response_ids", "response_token_mask", "behavior_logprobs"):
        offsets[name] = _offset_values(
            tokens[name],
            tokens[f"{name}_offsets"],
            rows=total,
            name=name,
        )
    for index in range(total):
        response_length = offsets["response_ids"][index + 1] - offsets["response_ids"][index]
        mask_length = (
            offsets["response_token_mask"][index + 1]
            - offsets["response_token_mask"][index]
        )
        logprob_length = (
            offsets["behavior_logprobs"][index + 1]
            - offsets["behavior_logprobs"][index]
        )
        if response_length != mask_length or response_length != logprob_length:
            raise ValueError("trajectory-store response offsets are not row-aligned")

    teacher: dict[str, Any] | None = None
    if bool(manifest["teacher_scored"]):
        teacher = load_file(root / TEACHER_FILENAME)
        if set(teacher) != TEACHER_TENSOR_KEYS:
            raise ValueError("trajectory-store teacher tensor inventory is not exact")
        if teacher["topk_ids"].ndim != 2 or teacher["topk_logprobs"].ndim != 2:
            raise ValueError("trajectory-store teacher top-k tensors must be rank two")
        if tuple(teacher["topk_ids"].shape) != tuple(teacher["topk_logprobs"].shape):
            raise ValueError("trajectory-store teacher IDs/logprobs shapes differ")
        if int(teacher["topk_ids"].shape[1]) != int(manifest["top_k"]):
            raise ValueError("trajectory-store teacher top-k width differs from manifest")
        teacher_offsets = _offset_values(
            teacher["topk_ids"],
            teacher["trajectory_offsets"],
            rows=total,
            name="teacher",
        )
        positions = int(teacher["topk_ids"].shape[0])
        if teacher["topk_mass"].ndim != 1 or teacher["entropy"].ndim != 1:
            raise ValueError("trajectory-store teacher summaries must be rank one")
        if int(teacher["topk_mass"].shape[0]) != positions or int(
            teacher["entropy"].shape[0]
        ) != positions:
            raise ValueError("trajectory-store teacher summary lengths differ from positions")
        for index in range(total):
            teacher_length = teacher_offsets[index + 1] - teacher_offsets[index]
            response_length = offsets["response_ids"][index + 1] - offsets["response_ids"][index]
            if teacher_length != response_length:
                raise ValueError("trajectory-store teacher offsets are not response-aligned")
    return rows, tokens, teacher


def validate_payload(root: Path, manifest: dict[str, Any]) -> tuple[str, ...]:
    rows, _, _ = _load_validated_payload(root, manifest)
    try:
        trajectory_ids = tuple(row["trajectory_id"] for row in rows)
    except KeyError as error:
        raise ValueError("trajectory-store metadata lacks trajectory_id") from error
    if any(not isinstance(value, str) or not value for value in trajectory_ids):
        raise ValueError("trajectory-store metadata contains malformed trajectory_id")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("trajectory-store metadata contains duplicate trajectory_id values")
    return trajectory_ids


def read_records(root: Path, manifest: dict[str, Any]) -> list[TrajectoryRecord]:
    rows, tokens, teacher = _load_validated_payload(root, manifest)
    records: list[TrajectoryRecord] = []
    for index, row in enumerate(rows):

        def ragged(name: str) -> list[Any]:
            raw_offsets = tokens[f"{name}_offsets"]
            start = int(raw_offsets[index])
            end = int(raw_offsets[index + 1])
            return tokens[name][start:end].tolist()

        trace = row.get("verification_trace")
        if isinstance(trace, str):
            row["verification_trace"] = json.loads(trace)
        elif trace is None:
            row["verification_trace"] = None
        row["input_ids"] = [int(value) for value in ragged("input_ids")]
        row["response_ids"] = [int(value) for value in ragged("response_ids")]
        row["response_token_mask"] = [bool(value) for value in ragged("response_token_mask")]
        row["behavior_logprobs"] = [float(value) for value in ragged("behavior_logprobs")]
        if teacher is not None:
            teacher_offsets = teacher["trajectory_offsets"]
            start = int(teacher_offsets[index])
            end = int(teacher_offsets[index + 1])
            row["teacher_topk_ids"] = teacher["topk_ids"][start:end].tolist()
            row["teacher_topk_logprobs"] = teacher["topk_logprobs"][start:end].tolist()
            row["teacher_topk_mass"] = teacher["topk_mass"][start:end].tolist()
            row["teacher_entropy"] = teacher["entropy"][start:end].tolist()
        else:
            row["teacher_topk_ids"] = []
            row["teacher_topk_logprobs"] = []
            row["teacher_topk_mass"] = []
            row["teacher_entropy"] = []
        record = TrajectoryRecord(**row)
        record.validate()
        records.append(record)
    return records
