"""Complete ordered teacher-demo attempt ledgers."""

from __future__ import annotations

from typing import Any

from posttrain_circuits.datasets.teacher_demos.contracts import (
    TeacherDemoAttempt,
    attempt_from_payload,
    attempt_to_payload,
)

LEDGER_FORMAT_VERSION = 1
LEDGER_FIELDS = {"format_version", "artifact_kind", "attempts"}


def build_attempt_ledger(attempts: list[TeacherDemoAttempt]) -> dict[str, Any]:
    if not attempts:
        raise ValueError("teacher-demo attempt ledger cannot be empty")
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("teacher-demo attempt ledger contains duplicate attempt IDs")
    return {
        "format_version": LEDGER_FORMAT_VERSION,
        "artifact_kind": "teacher_demo_attempt_ledger",
        "attempts": [attempt_to_payload(attempt) for attempt in attempts],
    }


def validate_attempt_ledger(payload: dict[str, Any]) -> list[TeacherDemoAttempt]:
    if not isinstance(payload, dict) or set(payload) != LEDGER_FIELDS:
        raise ValueError("teacher-demo ledger fields differ from the contract")
    if (
        payload.get("format_version") != LEDGER_FORMAT_VERSION
        or payload.get("artifact_kind") != "teacher_demo_attempt_ledger"
    ):
        raise ValueError("legacy teacher-demo ledgers are not accepted")
    raw_attempts = payload.get("attempts")
    if not isinstance(raw_attempts, list) or not raw_attempts:
        raise ValueError("teacher-demo attempt ledger cannot be empty")
    attempts = [attempt_from_payload(raw) for raw in raw_attempts]
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("teacher-demo attempt ledger contains duplicate attempt IDs")
    return attempts
