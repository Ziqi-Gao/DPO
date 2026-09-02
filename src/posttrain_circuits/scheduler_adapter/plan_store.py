"""Content-addressed resolution of immutable OPD workflow plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.manifest import RunningManifest
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout, validate_path_chain
from posttrain_circuits.scheduler_adapter.secure_files import (
    publish_bytes_once,
    published_json_bytes,
)
from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json
from posttrain_circuits.workflows.contracts import (
    WORKFLOW_TASK_REGISTRY,
    WorkflowPlan,
    WorkflowUnit,
)


@dataclass(frozen=True)
class ResolvedWorkflowUnit:
    plan: WorkflowPlan
    unit: WorkflowUnit
    plan_path: Path
    completion_path: Path
    output_paths: Mapping[str, Path]


def _published_plan_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the adapter's strict descriptor-safe publication format."""

    return published_json_bytes(payload)


def require_published_workflow_plan(
    plan: WorkflowPlan,
    *,
    layout: WorkflowLayout,
    task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
) -> Path:
    """Require the exact content and bytes produced by ``publish_workflow_plan``."""

    layout.validate()
    payload = plan.to_payload(task_registry=task_registry)
    plan_sha256 = plan.sha256(task_registry=task_registry)
    path = layout.plan_path(workflow_id=plan.workflow_id, plan_sha256=plan_sha256)
    validate_path_chain(path, approved_roots=(layout.data_root,), final_kind="file")
    observed, file_sha256 = read_strict_json(
        path, context="published OPD workflow plan", max_bytes=16 * 1024 * 1024
    )
    expected_file_sha256 = hashlib.sha256(_published_plan_bytes(payload)).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise AdapterValidationError(
            "published workflow plan bytes differ from the no-clobber project format"
        )
    try:
        restored = WorkflowPlan.from_payload(observed, task_registry=task_registry)
    except (TypeError, ValueError) as error:
        raise AdapterValidationError(f"published workflow plan is invalid: {error}") from error
    if restored != plan or observed != payload:
        raise AdapterValidationError("published workflow plan content differs from the caller plan")
    return path


def publish_workflow_plan(
    plan: WorkflowPlan,
    *,
    layout: WorkflowLayout,
    task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
) -> Path:
    """Publish one immutable content-addressed plan without replacing any file."""

    layout.validate()
    payload = plan.to_payload(task_registry=task_registry)
    path = layout.plan_path(
        workflow_id=plan.workflow_id,
        plan_sha256=plan.sha256(task_registry=task_registry),
    )
    try:
        publish_bytes_once(
            path,
            _published_plan_bytes(payload),
            approved_root=layout.data_root,
            context="immutable OPD workflow plan",
        )
    except (OSError, TypeError, ValueError) as error:
        raise AdapterValidationError(f"cannot publish immutable workflow plan: {error}") from error
    return require_published_workflow_plan(
        plan, layout=layout, task_registry=task_registry
    )


def resolve_workflow_unit(
    manifest: RunningManifest,
    *,
    layout: WorkflowLayout,
    task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
) -> ResolvedWorkflowUnit:
    """Bind exact job parameters to one hash-valid unit and its derived paths."""

    layout.validate()
    parameters = manifest.parameters
    plan_path = layout.plan_path(
        workflow_id=parameters.workflow_id, plan_sha256=parameters.plan_sha256
    )
    validate_path_chain(plan_path, approved_roots=(layout.data_root,), final_kind="file")
    payload, file_sha256 = read_strict_json(
        plan_path, context="immutable OPD workflow plan", max_bytes=16 * 1024 * 1024
    )
    try:
        plan = WorkflowPlan.from_payload(payload, task_registry=task_registry)
    except (TypeError, ValueError) as error:
        raise AdapterValidationError(f"invalid OPD workflow plan: {error}") from error
    if plan.workflow_id != parameters.workflow_id:
        raise AdapterValidationError("workflow plan identity does not match parameters.workflow_id")
    if plan.sha256(task_registry=task_registry) != parameters.plan_sha256:
        raise AdapterValidationError("workflow plan hash does not match parameters.plan_sha256")
    expected_file_sha256 = hashlib.sha256(
        _published_plan_bytes(plan.to_payload(task_registry=task_registry))
    ).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise AdapterValidationError(
            "workflow plan bytes differ from the no-clobber project publication format"
        )
    try:
        unit = plan.unit(parameters.unit_id, task_registry=task_registry)
    except (KeyError, ValueError) as error:
        raise AdapterValidationError(f"invalid requested workflow unit: {error}") from error
    if unit.task != manifest.task:
        raise AdapterValidationError("workflow unit task does not match the running manifest task")
    output_paths = {
        output_name: layout.output_path(
            workflow_id=plan.workflow_id,
            plan_sha256=parameters.plan_sha256,
            unit_id=unit.unit_id,
            output_name=output_name,
        )
        for output_name in unit.output_names
    }
    return ResolvedWorkflowUnit(
        plan=plan,
        unit=unit,
        plan_path=plan_path,
        completion_path=layout.completion_path(
            workflow_id=plan.workflow_id,
            plan_sha256=parameters.plan_sha256,
            unit_id=unit.unit_id,
        ),
        output_paths=output_paths,
    )
