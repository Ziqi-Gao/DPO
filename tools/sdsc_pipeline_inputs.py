#!/usr/bin/env python3
"""Pure selection of node-local inputs from the complete durable inventory.

This module never reads, copies, deletes, rewrites or relocates a file. Returned
keys and records are the original inventory objects; the caller retains the
complete durable inventory for publication and future stages.

Reviewed dependencies (science 0215c356; tools/sdsc_pilot.py):
* Every pilot stage replays G0, including both resumes and their original
  checkpoints, external Accelerate state and initial/checkpoint-20 dependencies.
  Therefore **all g0/** entries remain selected.
* After preflight4, admission reads its reports and publication proofs, not the
  six large files under pilot/preflight4/checkpoint/.
* A teacher-shard consumes the frozen population and creates only its own shard.
  An existing own shard remains visible so the worker refuses to overwrite it.
* pilot-inputs consumes all sixteen complete teacher ledgers. Later stages
  validate the merged teacher store and scored bank, retaining the shard stage
  reports/proofs but never reading the original shard directories again.
* Final training/native state is selected by the *contents* of scientific
  manifests, checkpoint payloads and update evidence. A path/size/hash inventory
  alone cannot prove which such files may be removed. Keep all training and
  resume checkpoint/native trees instead of guessing from step filenames.

The exact selected byte count is a bound on existing input bytes, not on future
checkpoint output growth. The 256-GiB free-space reserve is an admission floor;
it does not establish a worst-case bound for six checkpoints in each of eight
methods, the independent resumes or native GRPO/Accelerate checkpoint copies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePosixPath

STAGES = (
    "g0",
    "preflight4",
    "prepare",
    "teacher-shard",
    "pilot-inputs",
    "train-cell",
    "initial-circuits",
    "final-circuits",
    "local-fork",
    "resume",
    "dynamics",
    "finalize",
)
ARRAY_COUNTS = {"teacher-shard": 16, "train-cell": 8}
AFTER_MERGE = frozenset(STAGES[STAGES.index("train-cell") :])
AFTER_PREFLIGHT = frozenset(STAGES[STAGES.index("prepare") :])
FREE_SPACE_RESERVE_BYTES = 256 * 1024**3
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def logical_path(value):
    require(
        isinstance(value, str) and value and not any(ord(char) < 32 for char in value) and "\\" not in value,
        "invalid logical inventory path",
    )
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and len(path.parts) >= 2
        and path.parts[0] in {"g0", "pilot", "proofs"}
        and ".." not in path.parts
        and path.as_posix() == value,
        "inventory path escapes its fixed logical role",
    )
    return path.parts


def index_for(stage, array_index):
    require(isinstance(stage, str) and stage in STAGES, "unknown pipeline stage")
    if stage not in ARRAY_COUNTS:
        require(array_index is None or array_index == "single", "unexpected array index for scalar stage")
        return None
    if isinstance(array_index, str):
        require(re.fullmatch(r"0|[1-9][0-9]*", array_index), "invalid array index spelling")
        array_index = int(array_index)
    require(
        type(array_index) is int and 0 <= array_index < ARRAY_COUNTS[stage],
        "array index is missing or outside the registered task range",
    )
    return array_index


def validated_inventory(full_inventory):
    require(isinstance(full_inventory, Mapping), "full inventory must be a mapping")
    for name, record in full_inventory.items():
        logical_path(name)
        require(
            isinstance(record, Mapping)
            and type(record.get("size")) is int
            and record["size"] >= 0
            and isinstance(record.get("sha256"), str)
            and SHA256.fullmatch(record["sha256"]),
            "inventory requires original nonnegative byte sizes and SHA-256",
        )
        storage = record.get("storage")
        require(
            isinstance(storage, str)
            and storage
            and not any(ord(char) < 32 for char in storage)
            and "\\" not in storage,
            "invalid persistent storage locator",
        )
        path = PurePosixPath(storage)
        require(
            path.is_absolute()
            and not storage.startswith("//")
            and ".." not in path.parts
            and path.as_posix() == storage,
            "persistent storage locator must be canonical and absolute",
        )
    return full_inventory


def exclusion_reason(name, stage, array_index):
    parts = logical_path(name)
    if stage in AFTER_PREFLIGHT and len(parts) >= 4 and parts[:3] == ("pilot", "preflight4", "checkpoint"):
        return "preflight_checkpoint_verified_by_published_report_not_reloaded"
    if len(parts) >= 4 and parts[:3] == ("pilot", "inputs", "teacher_shards"):
        if stage in AFTER_MERGE:
            return "complete_merged_teacher_store_and_bank_replace_shard_inputs"
        if stage == "teacher-shard" and len(parts) >= 5:
            # Only omit recognized sibling shards. Keep the current shard and
            # any unexpected path visible to the scientific fail-closed checks.
            shard = parts[3]
            if shard in {f"{number:02d}" for number in range(16)} and shard != f"{array_index:02d}":
                return "independent_teacher_array_sibling_not_read"
    return None


def select_inventory(full_inventory, stage, array_index=None):
    """Return selected original records; never mutate or trim durable history.

    Array indices accept an exact int or canonical Slurm decimal spelling.
    Scalar stages accept None or the wrapper's literal ``single`` marker.
    Existing output/claim/report files are retained, preventing this selector
    from turning already executed work into an apparently fresh stage.
    """
    index = index_for(stage, array_index)
    validated_inventory(full_inventory)
    return {
        name: record
        for name, record in full_inventory.items()
        if exclusion_reason(name, stage, index) is None
    }


def selection_summary(full_inventory, stage, array_index=None):
    """Exact existing-byte accounting for the node's free-space admission check.

    Additional source, environment/model staging or output beyond the reserve
    must be accounted for by the outer wrapper. No filesystem capacity is
    inferred from this purely metadata-based function.
    """
    index = index_for(stage, array_index)
    selected = select_inventory(full_inventory, stage, index)
    excluded_by_reason = {}
    for name, record in full_inventory.items():
        if name not in selected:
            reason = exclusion_reason(name, stage, index)
            item = excluded_by_reason.setdefault(reason, {"files": 0, "bytes": 0})
            item["files"] += 1
            item["bytes"] += record["size"]
    total = sum(record["size"] for record in full_inventory.values())
    staged = sum(record["size"] for record in selected.values())
    checkpoint_bytes = sum(
        record["size"]
        for name, record in selected.items()
        if name.startswith(("pilot/runs/", "pilot/resume-a/", "pilot/resume-b/"))
    )
    return {
        "stage": stage,
        "array_index": index,
        "full_inventory_files": len(full_inventory),
        "full_inventory_bytes": total,
        "selected_files": len(selected),
        "selected_bytes": staged,
        "excluded_bytes": total - staged,
        "excluded_by_reason": excluded_by_reason,
        "minimum_free_bytes_before_input_copy": staged + FREE_SPACE_RESERVE_BYTES,
        "output_reserve_bytes": FREE_SPACE_RESERVE_BYTES,
        "retained_pilot_training_and_resume_bytes": checkpoint_bytes,
        "training_checkpoint_pruning": "none_without_manifest_and_payload_dependency_proof",
        "future_output_upper_bound_established": False,
        "durable_inventory_changed": False,
    }
