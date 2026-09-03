"""Atomic protocol-v2 request preparation; this module never submits or polls."""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.manifest import IDENTIFIER, PROJECT, WorkflowParameters
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.plan_store import require_published_workflow_plan
from posttrain_circuits.scheduler_adapter.registry import require_handler
from posttrain_circuits.scheduler_adapter.secure_files import (
    publish_bytes_once,
    published_json_bytes,
)
from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json
from posttrain_circuits.workflows.contracts import WORKFLOW_TASK_REGISTRY, WorkflowPlan


REQUEST_BASE_KEYS = frozenset(
    {"schema_version", "job_id", "project", "task", "priority", "parameters"}
)
OPD_JOB_ID = re.compile(r"opd-[0-9a-f]{32}\Z")


def validate_outbox_request(
    payload: object,
) -> dict[str, Any]:
    """Validate the exact request subset emitted by this project builder."""

    if not isinstance(payload, dict):
        raise AdapterValidationError("ServerScheduler outbox request must be an object")
    expected = set(REQUEST_BASE_KEYS)
    if "execution_profile" in payload:
        expected.add("execution_profile")
    if set(payload) != expected:
        raise AdapterValidationError(
            "OPD outbox request fields differ from the reviewed protocol-v2 subset"
        )
    if payload["schema_version"] != 2 or isinstance(payload["schema_version"], bool):
        raise AdapterValidationError("outbox request schema_version must be 2")
    if payload["project"] != PROJECT:
        raise AdapterValidationError("outbox request project must be OPD")
    if not isinstance(payload["job_id"], str) or not OPD_JOB_ID.fullmatch(
        payload["job_id"]
    ):
        raise AdapterValidationError(
            "outbox request job_id must be an opaque OPD submission identifier"
        )
    if not isinstance(payload["task"], str) or not IDENTIFIER.fullmatch(payload["task"]):
        raise AdapterValidationError("outbox request task is not a valid identifier")
    priority = payload["priority"]
    if isinstance(priority, bool) or not isinstance(priority, int) or not -100 <= priority <= 100:
        raise AdapterValidationError("outbox request priority is invalid")
    parameters = WorkflowParameters.from_payload(payload["parameters"])
    if parameters.to_payload() != payload["parameters"]:
        raise AdapterValidationError("outbox request parameters are not canonical")
    handler = require_handler(payload["task"])
    if "execution_profile" in payload:
        profile = payload["execution_profile"]
        if not isinstance(profile, str) or not IDENTIFIER.fullmatch(profile):
            raise AdapterValidationError("outbox execution_profile is invalid")
        handler.profile(profile)
    return payload


def build_outbox_request(
    plan: WorkflowPlan,
    *,
    unit_id: str,
    execution_profile: str | None = None,
) -> dict[str, Any]:
    """Build one submission request without resources, commands, paths, or env."""

    plan.validate(task_registry=WORKFLOW_TASK_REGISTRY)
    unit = plan.unit(unit_id, task_registry=WORKFLOW_TASK_REGISTRY)
    handler = require_handler(unit.task)
    handler.validate_unit_contract(unit)
    plan_sha256 = plan.sha256(task_registry=WORKFLOW_TASK_REGISTRY)
    core: dict[str, Any] = {
        "parameters": {
            "plan_sha256": plan_sha256,
            "unit_id": unit.unit_id,
            "workflow_id": plan.workflow_id,
        },
        "priority": 0,
        "project": PROJECT,
        "schema_version": 2,
        "task": unit.task,
    }
    if execution_profile is not None:
        handler.profile(execution_profile)
        core["execution_profile"] = execution_profile
    # Scientific identity belongs to the immutable workflow parameters.  A job
    # ID identifies one scheduler submission, so a retry of the same unit must
    # receive a fresh ID instead of colliding with durable scheduler history.
    request = {**core, "job_id": f"opd-{secrets.token_hex(16)}"}
    return validate_outbox_request(request)


def prepare_outbox_request(
    plan: WorkflowPlan,
    *,
    unit_id: str,
    execution_profile: str | None = None,
    layout: WorkflowLayout,
) -> Path:
    """Atomically publish a fresh submission request and return its local path."""

    layout.validate()
    require_published_workflow_plan(plan, layout=layout)
    request = build_outbox_request(
        plan,
        unit_id=unit_id,
        execution_profile=execution_profile,
    )
    target = layout.outbox_directory() / f"{request['job_id']}.json"
    try:
        publish_bytes_once(
            target,
            published_json_bytes(request),
            approved_root=layout.scratch_root,
            context="ServerScheduler outbox request",
        )
    except (OSError, TypeError, ValueError) as error:
        raise AdapterValidationError(f"cannot prepare ServerScheduler outbox request: {error}") from error
    observed, _digest = read_strict_json(
        target, context="prepared ServerScheduler outbox request", max_bytes=128 * 1024
    )
    validate_outbox_request(observed)
    if observed != request:
        raise AdapterValidationError("prepared outbox request changed during publication")
    return target
