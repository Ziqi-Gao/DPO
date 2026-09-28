"""Lossless SFT view of the independently accepted, adapted teacher evidence.

The producer's files are immutable inputs.  This module never rewrites their
origin, creates teacher acceptance, generates another candidate, or labels the
learned weights as the Hub base.  Only the two legacy identity columns are
materialized for the existing deterministic SFT source; every measurement is
copied unchanged and the derived manifest binds the original producer bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.teacher_identity import TeacherIdentity, teacher_science_bindings
from posttrain_circuits.datasets.teacher_demos.contracts import ATTEMPT_FIELDS, TeacherDemoAttempt
from posttrain_circuits.learning.teacher.seeding import (
    TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
    teacher_candidate_seed,
)

ADAPTED_MODEL_ID = "OPD/Qwen3-8B-adapted"
PRODUCER_HEAD = "929fb14834852a7c91e6656c76fd1834e1b5007d"
TEACHER_PROTOCOL_SHA256 = "dcd5fca7c87bda603f930e1e053d34073ca12aa079fa884c0e4e87e32ec099f4"
STORE_SHA256 = "f2e9e8171e3289bfd8bb356c79df67e56612f2ad067c63a4583b017c79025a88"
EVIDENCE_SHA256 = "c7b65df10069ab139179012c5590906783553f487de5bc581e80f40bbcf4801a"
VIEW_SCHEMA = "accepted-adapted-teacher-sft-view-v1"
CANDIDATE_FIELDS = (ATTEMPT_FIELDS - {"teacher_id", "teacher_revision"}) | {"teacher_identity_sha256"}
# These are actual independently audited publication identities, not a newly
# calculated inventory accepted merely because it is internally consistent.
INPUT_FILES = {
    "teacher-store/manifest.json": (
        10369,
        "8dc9928dae74340366b07fc385d0eeb046ddfd288859ef7ac5150fc0436c7a41",
    ),
    "teacher-store/attempts.jsonl": (
        20647754,
        "42e57c6ff09a38b2570c25502428baef885e5d1ff78cbd18adf2f67852b47c60",
    ),
    "teacher-store/accepted-view.json": (
        103400,
        "1b49719cb8e20a5a148d6b9f40a7560624fac5eb343f1a980e7824d825e185ae",
    ),
    "acceptance/accepted-teacher.json": (
        1093,
        "a38be3e22778e9a44fa4fca3a3d5250c161582fb30204aa9615da85adc37d2e7",
    ),
    "acceptance/independent-attestation.json": (
        717,
        "5f4aad09c79b0bb5c0ddcad953ef3d3094a82796ff997b7cbeb8554dc5e30278",
    ),
    "acceptance/publication-review.json": (
        4336,
        "b1d05abeec734627445f6d60c32b4829708d08885623496cf665a3e75b5d0b3b",
    ),
    "acceptance/audit-bindings.json": (
        2121,
        "e40e81d349ff5f083e560fdf8d0d11f1f0cc63e2abcc2466887d4abcc42cfb2d",
    ),
    "acceptance-evidence.json": (548805, "26ae126f121054c3445d5edbe81e44326320759a8a43941fd4d48e75e8a290c8"),
    "producer-report.json": (32223, "2b067e9e33e7a50ce0697f8defde697e714f49ddc2caed9769bbcc3c0d82b7ba"),
}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def configured_identity(config: dict[str, Any]) -> TeacherIdentity:
    """Validate the exact accepted teacher's content-only configuration block."""
    from posttrain_circuits.artifacts.adapted_student_protocol import adapted_teacher_config

    block = config.get("adapted_teacher")
    _require(block == adapted_teacher_config(), "adapted teacher configuration differs from accepted inputs")
    identity = TeacherIdentity.from_mapping(block["teacher_identity"])
    _require(identity.kind == "learned_dense_checkpoint", "SFT requires learned dense teacher identity")
    teacher = config.get("teacher", {})
    _require(
        teacher.get("model_name_or_path") == identity.base_model_id
        and teacher.get("model_revision") == identity.base_revision
        and teacher.get("tokenizer_name_or_path") == identity.tokenizer_id
        and teacher.get("tokenizer_revision") == identity.tokenizer_revision,
        "configured Hub teacher fields must describe the genuine base ancestry",
    )
    return identity


def _json(raw: bytes) -> dict[str, Any]:
    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate adapted teacher JSON field")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("non-finite adapted teacher JSON value: " + value)

    result = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    _require(isinstance(result, dict), "adapted teacher JSON must be an object")
    return result


def _self_hash(document: dict[str, Any], expected: str | None = None) -> str:
    digest = sha256_value({key: value for key, value in document.items() if key != "sha256"})
    _require(document.get("sha256") == digest, "adapted teacher document hash differs")
    _require(expected is None or digest == expected, "adapted teacher external content identity differs")
    return digest


def _file(path: Path, expected: tuple[int, str]) -> bytes:
    _require(path.is_absolute() and ".." not in path.parts, "teacher input path must be absolute")
    _require(
        not any(part.is_symlink() for part in (path, *path.parents)), "teacher input traverses a symlink"
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(
            stat.S_ISREG(before.st_mode) and before.st_size == expected[0], "teacher input size/type differs"
        )
        raw = stream.read(expected[0] + 1)
        after = os.fstat(stream.fileno())
    current = path.lstat()

    def stable(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    _require(stable(before) == stable(after) == stable(current), "teacher input changed during read")
    _require(
        len(raw) == expected[0] and hashlib.sha256(raw).hexdigest() == expected[1],
        "teacher input published bytes differ",
    )
    return raw


def _convert_rows(rows, source, accepted_view, identity):
    prompts = source.get("ordered_prompt_ids")
    _require(
        isinstance(prompts, list) and len(prompts) == len(set(prompts)) == 256,
        "teacher prompt population differs",
    )
    _require(
        len(rows) == 2048 and source.get("attempt_count") == 2048, "teacher candidate population differs"
    )
    accepted = []
    for index, row in enumerate(rows):
        _require(set(row) == CANDIDATE_FIELDS, "adapted candidate fields differ")
        prompt, candidate = prompts[index // 8], index % 8
        expected = {
            "prompt_id": prompt,
            "candidate_index": candidate,
            "attempt_id": f"{prompt}:candidate-{candidate:04d}",
            "teacher_identity_sha256": identity.sha256,
            "sampling_request_seed": 31415,
            "actual_sampling_seed": teacher_candidate_seed(31415, row["prompt_identity_sha256"], candidate),
            "sampling_protocol_id": TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
            "sampling_temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "prompt_protocol": identity.prompt_protocol,
            "chat_template_sha256": identity.chat_template_sha256,
            "tokenizer_fingerprint": identity.tokenizer_fingerprint,
            "enable_thinking": False,
            "accepted": True,
            "verifier_reward": 1.0,
            "logprob_status": "available",
        }
        _require(
            all(row.get(key) == value for key, value in expected.items()),
            "adapted candidate identity/sampling/gate differs",
        )
        converted = {
            key: copy.deepcopy(value) for key, value in row.items() if key != "teacher_identity_sha256"
        }
        converted.update(teacher_id=ADAPTED_MODEL_ID, teacher_revision=identity.teacher_checkpoint_sha256)
        attempt = TeacherDemoAttempt(**converted)
        attempt.validate()
        _require(
            0 < len(attempt.input_ids) <= 1246
            and 0 < len(attempt.response_ids) <= 256
            and len(attempt.input_ids) + len(attempt.response_ids) <= 1536,
            "accepted teacher demonstration exceeds the unchanged input envelope",
        )
        _require(
            attempt.response_token_mask == [True] * len(attempt.response_ids), "response masking changed"
        )
        accepted.append(attempt)
    _require(
        accepted_view.get("attempt_ids") == [item.attempt_id for item in accepted]
        and source.get("covered_prompts") == 256,
        "accepted view omits, adds or reorders candidates",
    )
    return accepted


def _generation(identity: TeacherIdentity) -> dict[str, Any]:
    return {
        "teacher_id": ADAPTED_MODEL_ID,
        "teacher_revision": identity.teacher_checkpoint_sha256,
        "resolved_teacher_commit": identity.teacher_checkpoint_sha256,
        "sampling_request_seed": 31415,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "candidates_per_prompt": 8,
        "max_prompt_tokens": 1246,
        "max_new_tokens": 256,
        "verifier_version": "proofgraph-exact-v1",
        "candidates_generated": 2048,
        "successful_candidates": 2048,
        "prompts_with_success": 256,
        "total_prompts": 256,
        "zero_success_prompt_ids": [],
        "retention_rate": 1.0,
        "logprob_status_counts": {"available": 2048},
        "teacher_identity": identity.to_mapping(),
    }


def read_accepted_teacher_sft(root: Path, *, config: dict[str, Any]):
    """Read the pinned producer bundle and return a derived in-memory SFT view.

    Consumer review is resolved before touching data.  Raw scientific evidence
    was independently replayed at acceptance; its exact externally pinned bytes
    are rehashed here, without repeating model inference or creating acceptance.
    """
    from posttrain_circuits.artifacts.adapted_student_protocol import validate_student_protocol

    validate_student_protocol(config)
    identity = configured_identity(config)
    root = Path(root)
    observed = set()
    for directory, folders, names in os.walk(root, followlinks=False):
        for name in folders + names:
            path = Path(directory) / name
            _require(not path.is_symlink(), "teacher input inventory contains a symlink")
            _require(path.is_dir() or path.is_file(), "teacher input inventory contains a special file")
        observed.update((Path(directory) / name).relative_to(root).as_posix() for name in names)
    _require(observed == set(INPUT_FILES), "teacher input bundle has missing or extra files")
    raw = {name: _file(root / name, record) for name, record in INPUT_FILES.items()}
    docs = {name: _json(value) for name, value in raw.items() if not name.endswith(".jsonl")}
    accepted_teacher = docs["acceptance/accepted-teacher.json"]
    acceptance_hash = _self_hash(accepted_teacher, config["adapted_teacher"]["acceptance_sha256"])
    evidence = docs["acceptance-evidence.json"]
    _self_hash(evidence, EVIDENCE_SHA256)
    _require(
        accepted_teacher.get("formal_teacher_accepted") is True
        and accepted_teacher.get("original_128_token_readiness_pass_claim") is False
        and accepted_teacher.get("protocol_sha256") == TEACHER_PROTOCOL_SHA256
        and accepted_teacher.get("evidence_sha256") == EVIDENCE_SHA256
        and accepted_teacher.get("teacher_identity") == identity.to_mapping(),
        "independent teacher acceptance differs",
    )
    attestation = docs["acceptance/independent-attestation.json"]
    _self_hash(attestation, accepted_teacher["independent_attestation_sha256"])
    _require(
        attestation.get("decision") == "accepted"
        and attestation.get("reviewed_evidence_sha256") == EVIDENCE_SHA256
        and attestation.get("acceptance_commit") == PRODUCER_HEAD,
        "independent teacher attestation differs",
    )
    report = docs["producer-report.json"]
    _require(
        report.get("passed") is True
        and report.get("exit_code") == 0
        and report.get("job_id") == "54496291"
        and report.get("science_git_head") == PRODUCER_HEAD
        and report.get("teacher_store_sha256") == STORE_SHA256
        and report.get("adapted_teacher_sha256") == identity.teacher_checkpoint_sha256,
        "teacher producer identity differs",
    )
    source, view = docs["teacher-store/manifest.json"], docs["teacher-store/accepted-view.json"]
    _self_hash(source, STORE_SHA256)
    _self_hash(view, source["accepted_view_sha256"])
    stage = evidence["external_evidence_bindings"]["formal_store"]["sha256"]
    for document in (source, view):
        _require(
            document.get("teacher_identity") == identity.to_mapping()
            and document.get("protocol_sha256") == TEACHER_PROTOCOL_SHA256
            and document.get("evidence_sha256") == stage
            and document.get("formal_teacher_accepted") is False,
            "producer source claims differ from the independent accepted evidence",
        )
    expected_records = [
        {
            "path": name,
            "size": INPUT_FILES["teacher-store/" + name][0],
            "sha256": INPUT_FILES["teacher-store/" + name][1],
        }
        for name in ("attempts.jsonl", "accepted-view.json")
    ]
    _require(source.get("files") == expected_records, "producer file inventory differs")
    rows = [_json(line) for line in raw["teacher-store/attempts.jsonl"].splitlines()]
    attempts = _convert_rows(rows, source, view, identity)
    generation = _generation(identity)
    manifest = {
        "format_version": 1,
        "store_kind": VIEW_SCHEMA,
        "artifact_kind": "derived_teacher_demo_view",
        "files": {name: record[1] for name, record in INPUT_FILES.items()},
        "ordered_prompt_ids": list(source["ordered_prompt_ids"]),
        "attempt_count": 2048,
        "accepted_count": 2048,
        "zero_success_prompt_ids": [],
        "ready_for_formal_sft": True,
        "tokenizer_hash": identity.tokenizer_fingerprint,
        "behavior_policy": {
            "id": ADAPTED_MODEL_ID,
            "revision": identity.teacher_checkpoint_sha256,
            "resolved_commit": identity.teacher_checkpoint_sha256,
        },
        "teacher_demo_generation": generation,
        "protocol_bindings": {
            **teacher_science_bindings(identity),
            "producer_science_head": PRODUCER_HEAD,
            "qualification_job_id": "54496291",
            "producer_store_sha256": STORE_SHA256,
            "teacher_acceptance_sha256": acceptance_hash,
            "teacher_acceptance_evidence_sha256": EVIDENCE_SHA256,
            "teacher_protocol_sha256": TEACHER_PROTOCOL_SHA256,
        },
        "source_records_unchanged": True,
    }
    manifest["sha256"] = sha256_value(manifest)
    return attempts, manifest


def validate_adapted_sft_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> TeacherIdentity:
    """Bind a derived view at every formal train/manifest entry point."""
    identity = configured_identity(config)
    _self_hash(manifest)
    expected = {
        **teacher_science_bindings(identity),
        "producer_science_head": PRODUCER_HEAD,
        "qualification_job_id": "54496291",
        "producer_store_sha256": STORE_SHA256,
        "teacher_acceptance_sha256": config["adapted_teacher"]["acceptance_sha256"],
        "teacher_acceptance_evidence_sha256": EVIDENCE_SHA256,
        "teacher_protocol_sha256": TEACHER_PROTOCOL_SHA256,
    }
    _require(
        manifest.get("store_kind") == VIEW_SCHEMA
        and manifest.get("protocol_bindings") == expected
        and manifest.get("files") == {name: record[1] for name, record in INPUT_FILES.items()}
        and manifest.get("source_records_unchanged") is True,
        "SFT derived view lost its original teacher provenance",
    )
    prompts = manifest.get("ordered_prompt_ids")
    generation = manifest.get("teacher_demo_generation", {})
    _require(
        manifest.get("attempt_count") == manifest.get("accepted_count") == 2048
        and manifest.get("zero_success_prompt_ids") == []
        and manifest.get("ready_for_formal_sft") is True
        and manifest.get("tokenizer_hash") == identity.tokenizer_fingerprint
        and isinstance(prompts, list)
        and len(prompts) == len(set(prompts)) == 256
        and generation == _generation(identity),
        "SFT derived view population/envelope/teacher identity differs",
    )
    _require(
        manifest.get("behavior_policy")
        == {
            "id": ADAPTED_MODEL_ID,
            "revision": identity.teacher_checkpoint_sha256,
            "resolved_commit": identity.teacher_checkpoint_sha256,
        },
        "adapted teacher cannot be relabeled as the Hub base",
    )
    return identity
