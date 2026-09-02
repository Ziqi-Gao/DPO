"""Exact-inventory teacher-demo ledger/view store."""

from __future__ import annotations

import json
import os
import stat
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.datasets.teacher_demos.contracts import (
    LOGPROB_AVAILABLE,
    TeacherDemoAttempt,
)
from posttrain_circuits.datasets.teacher_demos.ledger import (
    build_attempt_ledger,
    validate_attempt_ledger,
)
from posttrain_circuits.datasets.teacher_demos.views import (
    build_accepted_view,
    validate_accepted_view,
)

STORE_FORMAT_VERSION = 1
LEDGER_FILENAME = "ledger.json"
VIEW_FILENAME = "accepted_view.json"
MANIFEST_FILENAME = "manifest.json"
STORE_FILES = {LEDGER_FILENAME, VIEW_FILENAME, MANIFEST_FILENAME}
GENERATION_INPUT_FIELDS = {
    "teacher_id",
    "teacher_revision",
    "resolved_teacher_commit",
    "sampling_request_seed",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "candidates_per_prompt",
    "verifier_version",
}
MANIFEST_FIELDS = {
    "format_version",
    "store_kind",
    "artifact_kind",
    "files",
    "ledger_file_sha256",
    "accepted_view_file_sha256",
    "ordered_prompt_ids",
    "prompt_manifest_hash",
    "attempt_count",
    "accepted_count",
    "zero_success_prompt_ids",
    "ready_for_formal_sft",
    "logprob_status_counts",
    "tokenizer_hash",
    "behavior_policy",
    "teacher_demo_generation",
    "protocol_bindings",
    "sha256",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError(f"teacher-demo {name} must be a regular non-symlink file")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"teacher-demo {name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"teacher-demo {name} contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"teacher-demo {name} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"teacher-demo {name} must be a JSON object")
    return payload


def _validate_generation_input(generation: dict[str, Any]) -> None:
    if not isinstance(generation, dict) or set(generation) != GENERATION_INPUT_FIELDS:
        raise ValueError("teacher-demo generation fields differ from the contract")
    for key in ("teacher_id", "teacher_revision", "resolved_teacher_commit", "verifier_version"):
        if not isinstance(generation[key], str) or not generation[key].strip():
            raise ValueError(f"teacher-demo generation {key} is empty")
    if type(generation["sampling_request_seed"]) is not int:
        raise ValueError("teacher-demo generation request seed must be an integer")
    if type(generation["candidates_per_prompt"]) is not int or generation[
        "candidates_per_prompt"
    ] < 1:
        raise ValueError("teacher-demo candidates_per_prompt must be positive")


def _validate_attempt_population(
    attempts: list[TeacherDemoAttempt],
    *,
    ordered_prompt_ids: list[str],
    generation: dict[str, Any],
) -> None:
    if not ordered_prompt_ids or len(ordered_prompt_ids) != len(set(ordered_prompt_ids)):
        raise ValueError("teacher-demo ordered prompt population is empty or duplicated")
    candidates_per_prompt = int(generation["candidates_per_prompt"])
    expected_attempt_ids = [
        f"{prompt_id}:candidate-{candidate_index:04d}"
        for prompt_id in ordered_prompt_ids
        for candidate_index in range(candidates_per_prompt)
    ]
    if [attempt.attempt_id for attempt in attempts] != expected_attempt_ids:
        raise ValueError("teacher-demo ledger does not contain every ordered prompt/candidate attempt")
    prompt_id_set = set(ordered_prompt_ids)
    prompt_identities: dict[str, set[str]] = defaultdict(set)
    for attempt in attempts:
        attempt.validate()
        if attempt.prompt_id not in prompt_id_set:
            raise ValueError("teacher-demo attempt lies outside the ordered prompt population")
        prompt_identities[attempt.prompt_id].add(attempt.prompt_identity_sha256)
        expected = {
            "teacher_id": attempt.teacher_id,
            "teacher_revision": attempt.teacher_revision,
            "sampling_request_seed": attempt.sampling_request_seed,
            "sampling_temperature": attempt.sampling_temperature,
            "top_p": attempt.top_p,
            "top_k": attempt.top_k,
            "min_p": attempt.min_p,
        }
        observed = {
            "teacher_id": generation["teacher_id"],
            "teacher_revision": generation["teacher_revision"],
            "sampling_request_seed": generation["sampling_request_seed"],
            "sampling_temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "top_k": generation["top_k"],
            "min_p": generation["min_p"],
        }
        if expected != observed:
            raise ValueError("teacher-demo attempt sampling/teacher identity differs from generation")
    if set(prompt_identities) != prompt_id_set or any(
        len(values) != 1 for values in prompt_identities.values()
    ):
        raise ValueError("teacher-demo prompt identity changed between sibling attempts")


def _manifest_content(
    *,
    ledger_file_sha256: str,
    accepted_view_file_sha256: str,
    attempts: list[TeacherDemoAttempt],
    ordered_prompt_ids: list[str],
    prompt_manifest_hash: str,
    tokenizer_hash: str,
    generation: dict[str, Any],
    protocol_bindings: dict[str, Any],
) -> dict[str, Any]:
    accepted = [attempt for attempt in attempts if attempt.accepted]
    successes = Counter(attempt.prompt_id for attempt in accepted)
    zero_success_prompt_ids = [
        prompt_id for prompt_id in ordered_prompt_ids if successes[prompt_id] == 0
    ]
    logprob_counts = dict(sorted(Counter(attempt.logprob_status for attempt in attempts).items()))
    ready_for_formal_sft = not zero_success_prompt_ids and all(
        attempt.logprob_status == LOGPROB_AVAILABLE for attempt in accepted
    )
    generation_payload = {
        **generation,
        "prompt_manifest_hash": prompt_manifest_hash,
        "ledger_file_sha256": ledger_file_sha256,
        "accepted_view_file_sha256": accepted_view_file_sha256,
        "candidates_generated": len(attempts),
        "successful_candidates": len(accepted),
        "prompts_with_success": len(ordered_prompt_ids) - len(zero_success_prompt_ids),
        "total_prompts": len(ordered_prompt_ids),
        "zero_success_prompt_ids": zero_success_prompt_ids,
        "retention_rate": len(accepted) / len(attempts),
        "logprob_status_counts": logprob_counts,
    }
    return {
        "format_version": STORE_FORMAT_VERSION,
        "store_kind": "teacher_demo_ledger",
        "artifact_kind": "teacher_demo_store",
        "files": {
            LEDGER_FILENAME: ledger_file_sha256,
            VIEW_FILENAME: accepted_view_file_sha256,
        },
        "ledger_file_sha256": ledger_file_sha256,
        "accepted_view_file_sha256": accepted_view_file_sha256,
        "ordered_prompt_ids": ordered_prompt_ids,
        "prompt_manifest_hash": prompt_manifest_hash,
        "attempt_count": len(attempts),
        "accepted_count": len(accepted),
        "zero_success_prompt_ids": zero_success_prompt_ids,
        "ready_for_formal_sft": ready_for_formal_sft,
        "logprob_status_counts": logprob_counts,
        "tokenizer_hash": tokenizer_hash,
        "behavior_policy": {
            "id": generation["teacher_id"],
            "revision": generation["teacher_revision"],
            "resolved_commit": generation["resolved_teacher_commit"],
        },
        "teacher_demo_generation": generation_payload,
        "protocol_bindings": protocol_bindings,
    }


def write_teacher_demo_store(
    root: Path,
    attempts: list[TeacherDemoAttempt],
    *,
    ordered_prompt_ids: list[str],
    prompt_manifest_hash: str,
    tokenizer_hash: str,
    generation: dict[str, Any],
    protocol_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write every attempt and an accepted-reference view, then fail on zero-success prompts."""

    _validate_generation_input(generation)
    if not _is_sha256(prompt_manifest_hash) or not _is_sha256(tokenizer_hash):
        raise ValueError("teacher-demo prompt/tokenizer bindings must be SHA-256 values")
    _validate_attempt_population(
        attempts,
        ordered_prompt_ids=ordered_prompt_ids,
        generation=generation,
    )
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError(f"teacher-demo output is not empty: {root}")
    else:
        root.mkdir(parents=True)

    ledger = build_attempt_ledger(attempts)
    ledger_path = root / LEDGER_FILENAME
    atomic_write_json(ledger_path, ledger)
    ledger_file_sha256 = sha256_file(ledger_path)
    view = build_accepted_view(attempts, ledger_file_sha256=ledger_file_sha256)
    view_path = root / VIEW_FILENAME
    atomic_write_json(view_path, view)
    accepted_view_file_sha256 = sha256_file(view_path)
    content = _manifest_content(
        ledger_file_sha256=ledger_file_sha256,
        accepted_view_file_sha256=accepted_view_file_sha256,
        attempts=attempts,
        ordered_prompt_ids=list(ordered_prompt_ids),
        prompt_manifest_hash=prompt_manifest_hash,
        tokenizer_hash=tokenizer_hash,
        generation=dict(generation),
        protocol_bindings=dict(protocol_bindings or {}),
    )
    manifest = {**content, "sha256": sha256_value(content)}
    atomic_write_json(root / MANIFEST_FILENAME, manifest)
    _, validated = read_teacher_demo_store(root, require_formal=False)
    if validated["zero_success_prompt_ids"]:
        raise ValueError(
            "teacher-demo generation has zero-success prompts; diagnostic ledger was written: "
            f"{validated['zero_success_prompt_ids']}"
        )
    return validated


def _validate_manifest(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or set(payload) != MANIFEST_FIELDS:
        raise ValueError("teacher-demo manifest fields differ from the ledger/view contract")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != sha256_value(content):
        raise ValueError("teacher-demo manifest hash mismatch")
    if (
        payload.get("format_version") != STORE_FORMAT_VERSION
        or payload.get("store_kind") != "teacher_demo_ledger"
        or payload.get("artifact_kind") != "teacher_demo_store"
    ):
        raise ValueError("legacy teacher-demo stores are not accepted")
    for name in (
        "ledger_file_sha256",
        "accepted_view_file_sha256",
        "prompt_manifest_hash",
        "tokenizer_hash",
    ):
        if not _is_sha256(payload.get(name)):
            raise ValueError(f"teacher-demo manifest {name} is not a SHA-256")
    if payload.get("files") != {
        LEDGER_FILENAME: payload["ledger_file_sha256"],
        VIEW_FILENAME: payload["accepted_view_file_sha256"],
    }:
        raise ValueError("teacher-demo content-file identities differ from ledger/view hashes")
    if not isinstance(payload.get("protocol_bindings"), dict):
        raise ValueError("teacher-demo protocol bindings must be a mapping")
    generation = payload.get("teacher_demo_generation")
    if not isinstance(generation, dict):
        raise ValueError("teacher-demo generation provenance is missing")
    expected_generation = {
        "prompt_manifest_hash": payload["prompt_manifest_hash"],
        "ledger_file_sha256": payload["ledger_file_sha256"],
        "accepted_view_file_sha256": payload["accepted_view_file_sha256"],
        "candidates_generated": payload["attempt_count"],
        "successful_candidates": payload["accepted_count"],
        "total_prompts": len(payload["ordered_prompt_ids"]),
        "zero_success_prompt_ids": payload["zero_success_prompt_ids"],
        "logprob_status_counts": payload["logprob_status_counts"],
    }
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise ValueError("teacher-demo generation provenance differs from manifest identities")
    _validate_generation_input(
        {key: generation[key] for key in GENERATION_INPUT_FIELDS}
    )


def read_teacher_demo_store(
    root: Path,
    *,
    require_formal: bool,
) -> tuple[list[TeacherDemoAttempt], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("teacher-demo store root must be a real directory")
    entries = {path.name for path in root.iterdir()}
    if entries != STORE_FILES:
        raise ValueError(
            "teacher-demo store inventory changed: "
            f"missing={sorted(STORE_FILES - entries)}, extra={sorted(entries - STORE_FILES)}"
        )
    manifest = _strict_json(root / MANIFEST_FILENAME, name="manifest")
    _validate_manifest(manifest)
    if sha256_file(root / LEDGER_FILENAME) != manifest["ledger_file_sha256"]:
        raise ValueError("teacher-demo ledger file hash mismatch")
    if sha256_file(root / VIEW_FILENAME) != manifest["accepted_view_file_sha256"]:
        raise ValueError("teacher-demo accepted-view file hash mismatch")
    attempts = validate_attempt_ledger(
        _strict_json(root / LEDGER_FILENAME, name="attempt ledger")
    )
    generation = manifest["teacher_demo_generation"]
    _validate_attempt_population(
        attempts,
        ordered_prompt_ids=list(manifest["ordered_prompt_ids"]),
        generation={key: generation[key] for key in GENERATION_INPUT_FIELDS},
    )
    accepted = validate_accepted_view(
        _strict_json(root / VIEW_FILENAME, name="accepted view"),
        attempts,
        expected_ledger_file_sha256=manifest["ledger_file_sha256"],
    )
    recomputed = _manifest_content(
        ledger_file_sha256=manifest["ledger_file_sha256"],
        accepted_view_file_sha256=manifest["accepted_view_file_sha256"],
        attempts=attempts,
        ordered_prompt_ids=list(manifest["ordered_prompt_ids"]),
        prompt_manifest_hash=manifest["prompt_manifest_hash"],
        tokenizer_hash=manifest["tokenizer_hash"],
        generation={key: generation[key] for key in GENERATION_INPUT_FIELDS},
        protocol_bindings=dict(manifest["protocol_bindings"]),
    )
    if {**recomputed, "sha256": sha256_value(recomputed)} != manifest:
        raise ValueError("teacher-demo manifest semantics differ from ledger/view files")
    if require_formal:
        if manifest["zero_success_prompt_ids"]:
            raise ValueError("formal SFT rejects a teacher-demo store with zero-success prompts")
        unavailable = [
            attempt.attempt_id
            for attempt in accepted
            if attempt.logprob_status != LOGPROB_AVAILABLE
        ]
        if unavailable or manifest["ready_for_formal_sft"] is not True:
            raise ValueError(
                "formal SFT rejects fixture-unavailable teacher logprobs: "
                f"{unavailable}"
            )
    return accepted, manifest
