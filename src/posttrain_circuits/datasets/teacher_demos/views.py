"""Accepted-demo reference views over complete attempt ledgers."""

from __future__ import annotations

from typing import Any

from posttrain_circuits.datasets.teacher_demos.contracts import TeacherDemoAttempt

VIEW_FORMAT_VERSION = 1
VIEW_FIELDS = {
    "format_version",
    "artifact_kind",
    "ledger_file_sha256",
    "accepted_attempts",
}
REFERENCE_FIELDS = {"attempt_id", "prompt_id"}


def build_accepted_view(
    attempts: list[TeacherDemoAttempt],
    *,
    ledger_file_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": VIEW_FORMAT_VERSION,
        "artifact_kind": "teacher_demo_accepted_view",
        "ledger_file_sha256": ledger_file_sha256,
        "accepted_attempts": [
            {"attempt_id": attempt.attempt_id, "prompt_id": attempt.prompt_id}
            for attempt in attempts
            if attempt.accepted
        ],
    }


def validate_accepted_view(
    payload: dict[str, Any],
    attempts: list[TeacherDemoAttempt],
    *,
    expected_ledger_file_sha256: str,
) -> list[TeacherDemoAttempt]:
    if not isinstance(payload, dict) or set(payload) != VIEW_FIELDS:
        raise ValueError("teacher-demo accepted-view fields differ from the contract")
    if (
        payload.get("format_version") != VIEW_FORMAT_VERSION
        or payload.get("artifact_kind") != "teacher_demo_accepted_view"
        or payload.get("ledger_file_sha256") != expected_ledger_file_sha256
    ):
        raise ValueError("teacher-demo accepted view is not bound to this ledger")
    raw_references = payload.get("accepted_attempts")
    if not isinstance(raw_references, list):
        raise ValueError("teacher-demo accepted references must be a list")
    references: list[tuple[str, str]] = []
    for raw in raw_references:
        if not isinstance(raw, dict) or set(raw) != REFERENCE_FIELDS:
            raise ValueError("teacher-demo accepted reference fields changed")
        attempt_id = str(raw["attempt_id"])
        prompt_id = str(raw["prompt_id"])
        if not attempt_id or not prompt_id:
            raise ValueError("teacher-demo accepted reference identity is empty")
        references.append((attempt_id, prompt_id))
    expected = [
        (attempt.attempt_id, attempt.prompt_id) for attempt in attempts if attempt.accepted
    ]
    if references != expected:
        raise ValueError("teacher-demo accepted view differs from successful ledger attempts")
    by_id = {attempt.attempt_id: attempt for attempt in attempts}
    return [by_id[attempt_id] for attempt_id, _ in references]
