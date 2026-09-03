"""Handler-aware validation of scheduler-neutral ScientificCompletion evidence."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from posttrain_circuits.artifacts.completion import (
    ExecutionIdentity,
    ScientificCompletion,
    completion_payload,
    validate_completion_payload,
)
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import publish_path_no_clobber
from posttrain_circuits.scheduler_adapter.content_store import (
    ATTEMPT_COMPLETION_NAME,
    OutputAttempt,
)
from posttrain_circuits.scheduler_adapter.environment import RuntimeEnvelope
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout, validate_path_chain
from posttrain_circuits.scheduler_adapter.registry import (
    HANDLER_REGISTRY,
    HandlerSpec,
    require_handler,
)
from posttrain_circuits.scheduler_adapter.secure_files import (
    open_regular_at_nofollow,
    open_regular_file_nofollow,
    publish_bytes_once,
    published_json_bytes,
    read_descriptor_bytes,
    sha256_descriptor,
)
from posttrain_circuits.scheduler_adapter.strict_json import (
    parse_strict_json,
    read_strict_json,
)
from posttrain_circuits.workflows.contracts import (
    WORKFLOW_TASK_REGISTRY,
    WorkflowPlan,
    WorkflowUnit,
    unit_identity_sha256,
)


ATTEMPT_COMPLETION_SCHEMA_VERSION = 1
ATTEMPT_COMPLETION_KIND = "scientific_attempt"


@dataclass(frozen=True)
class AttemptCompletionDraft:
    """Path-free scientific evidence written inside one output-attempt fd."""

    workflow_id: str
    plan_sha256: str
    unit_id: str
    task: str
    run_id: str
    started_at: str
    completed_at: str
    scientific_config_sha256: str
    execution_config_sha256: str
    resolved_config_sha256: str
    input_hashes: dict[str, str]
    scientific_validation: dict[str, bool]
    execution: ExecutionIdentity
    schema_version: int = ATTEMPT_COMPLETION_SCHEMA_VERSION
    project: str = "OPD"
    completion_kind: str = ATTEMPT_COMPLETION_KIND

    def validate(self) -> None:
        if (
            self.schema_version != ATTEMPT_COMPLETION_SCHEMA_VERSION
            or isinstance(self.schema_version, bool)
            or self.project != "OPD"
            or self.completion_kind != ATTEMPT_COMPLETION_KIND
            or not isinstance(self.execution, ExecutionIdentity)
        ):
            raise AdapterValidationError("attempt completion contract is invalid")
        # The project-wide marker validator owns timestamps, config digests,
        # input hashes, gates, and execution identity.  A placeholder output is
        # used only for value validation; no path is accepted from the draft.
        try:
            ScientificCompletion(
                workflow_id=self.workflow_id,
                plan_sha256=self.plan_sha256,
                unit_id=self.unit_id,
                task=self.task,
                run_id=self.run_id,
                started_at=self.started_at,
                completed_at=self.completed_at,
                scientific_config_sha256=self.scientific_config_sha256,
                execution_config_sha256=self.execution_config_sha256,
                resolved_config_sha256=self.resolved_config_sha256,
                input_hashes=self.input_hashes,
                output_files={"/opd-output-placeholder": "0" * 64},
                scientific_validation=self.scientific_validation,
                execution=self.execution,
            ).validate()
        except (TypeError, ValueError) as error:
            raise AdapterValidationError(
                f"attempt completion values are invalid: {error}"
            ) from error

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        content = asdict(self)
        return {**content, "sha256": sha256_value(content)}

    @classmethod
    def from_payload(cls, payload: object) -> "AttemptCompletionDraft":
        if not isinstance(payload, dict):
            raise AdapterValidationError("attempt completion must be an object")
        expected = {
            "completed_at",
            "completion_kind",
            "execution",
            "execution_config_sha256",
            "input_hashes",
            "plan_sha256",
            "project",
            "resolved_config_sha256",
            "run_id",
            "schema_version",
            "scientific_config_sha256",
            "scientific_validation",
            "sha256",
            "started_at",
            "task",
            "unit_id",
            "workflow_id",
        }
        if set(payload) != expected:
            raise AdapterValidationError(
                "attempt completion fields differ from the strict schema"
            )
        content = {key: value for key, value in payload.items() if key != "sha256"}
        if payload.get("sha256") != sha256_value(content):
            raise AdapterValidationError("attempt completion SHA-256 mismatch")
        raw_execution = content.get("execution")
        execution_fields = {
            "allocation_sha256",
            "attempt",
            "execution_profile",
            "job_id",
            "manifest_sha256",
        }
        if not isinstance(raw_execution, dict) or set(raw_execution) != execution_fields:
            raise AdapterValidationError("attempt completion execution fields are invalid")
        try:
            execution = ExecutionIdentity(**raw_execution)
            draft = cls(
                execution=execution,
                **{key: value for key, value in content.items() if key != "execution"},
            )
        except TypeError as error:
            raise AdapterValidationError("attempt completion values are invalid") from error
        draft.validate()
        if draft.to_payload() != payload:
            raise AdapterValidationError("attempt completion is not a canonical round-trip")
        return draft


@dataclass(frozen=True)
class CompletionValidationContext:
    workflow_id: str
    plan_sha256: str
    unit_id: str
    task: str
    expected_input_hashes: Mapping[str, str]
    expected_output_paths: Mapping[str, Path]
    expected_execution: ExecutionIdentity | None
    reused: bool


def execution_identity(envelope: RuntimeEnvelope) -> ExecutionIdentity:
    identity = ExecutionIdentity(
        job_id=envelope.job_id,
        attempt=envelope.attempt,
        execution_profile=envelope.execution_profile,
        manifest_sha256=envelope.manifest_sha256,
        allocation_sha256=envelope.allocation_sha256,
    )
    identity.validate()
    return identity


def _completion_path(
    plan: WorkflowPlan, unit: WorkflowUnit, *, plan_sha256: str, layout: WorkflowLayout
) -> Path:
    return layout.completion_path(
        workflow_id=plan.workflow_id,
        plan_sha256=plan_sha256,
        unit_id=unit.unit_id,
    )


def _expected_output_paths(
    plan: WorkflowPlan, unit: WorkflowUnit, *, plan_sha256: str, layout: WorkflowLayout
) -> dict[str, Path]:
    return {
        output_name: layout.output_path(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id=unit.unit_id,
            output_name=output_name,
        )
        for output_name in unit.output_names
    }


def write_attempt_completion_draft(
    attempt: OutputAttempt,
    draft: AttemptCompletionDraft,
    *,
    layout: WorkflowLayout,
) -> Path:
    """Fixture/handler API: publish one path-free draft inside the held attempt."""

    attempt.assert_path_identity()
    draft.validate()
    if (
        draft.workflow_id != attempt.workflow_id
        or draft.plan_sha256 != attempt.plan_sha256
        or draft.unit_id != attempt.unit_id
        or draft.execution.attempt != attempt.attempt
    ):
        raise AdapterValidationError(
            "attempt completion draft differs from its fixed output context"
        )
    path = attempt.path / ATTEMPT_COMPLETION_NAME
    try:
        publish_bytes_once(
            path,
            published_json_bytes(draft.to_payload()),
            approved_root=layout.data_root,
            context="OPD attempt completion draft",
        )
    except (OSError, TypeError, ValueError) as error:
        raise AdapterValidationError(
            f"cannot publish OPD attempt completion draft: {error}"
        ) from error
    return path


def _expected_input_hashes(
    plan: WorkflowPlan,
    unit: WorkflowUnit,
    *,
    plan_sha256: str,
    layout: WorkflowLayout,
    handler_registry: Mapping[str, HandlerSpec],
    task_registry: Mapping[str, tuple[str, ...]],
    cache: dict[str, dict[str, Any]],
) -> dict[str, str]:
    expected = {identity.name: identity.sha256 for identity in unit.content_inputs}
    for dependency in unit.dependencies:
        upstream = plan.unit(dependency.unit_id, task_registry=task_registry)
        upstream_payload = validate_unit_completion(
            plan,
            upstream,
            plan_sha256=plan_sha256,
            layout=layout,
            handler_registry=handler_registry,
            expected_execution=None,
            task_registry=task_registry,
            _cache=cache,
        )
        upstream_path = layout.output_path(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id=upstream.unit_id,
            output_name=dependency.output_name,
        )
        try:
            upstream_digest = upstream_payload["output_files"][str(upstream_path)]
        except (KeyError, TypeError) as error:
            raise AdapterValidationError(
                "dependency completion does not declare its deterministic output path"
            ) from error
        expected[
            f"dependency:{dependency.unit_id}:{dependency.output_name}"
        ] = upstream_digest
    return expected


def _read_attempt_draft(attempt: OutputAttempt) -> AttemptCompletionDraft:
    descriptor = open_regular_at_nofollow(
        attempt.descriptor,
        ATTEMPT_COMPLETION_NAME,
        context="OPD attempt completion draft",
    )
    try:
        raw = read_descriptor_bytes(
            descriptor,
            context="OPD attempt completion draft",
            max_bytes=4 * 1024 * 1024,
        )
    finally:
        os.close(descriptor)
    return AttemptCompletionDraft.from_payload(
        parse_strict_json(raw, context="OPD attempt completion draft")
    )


def _hash_attempt_outputs(
    attempt: OutputAttempt, unit: WorkflowUnit
) -> dict[str, str]:
    try:
        entries = os.listdir(attempt.descriptor)
    except OSError as error:
        raise AdapterValidationError(f"cannot list output attempt: {error}") from error
    expected_entries = set(unit.output_names) | {ATTEMPT_COMPLETION_NAME}
    if set(entries) != expected_entries or len(entries) != len(expected_entries):
        raise AdapterValidationError(
            "output attempt has missing or extra entries"
        )
    hashes: dict[str, str] = {}
    for output_name in unit.output_names:
        descriptor = open_regular_at_nofollow(
            attempt.descriptor,
            output_name,
            context=f"attempt output {output_name!r}",
        )
        try:
            if os.fstat(descriptor).st_nlink != 1:
                raise AdapterValidationError(
                    f"attempt output {output_name!r} must not be hard-linked"
                )
            hashes[output_name] = sha256_descriptor(
                descriptor, context=f"attempt output {output_name!r}"
            )
        finally:
            os.close(descriptor)
    return hashes


def _completion_from_draft(
    draft: AttemptCompletionDraft,
    *,
    plan: WorkflowPlan,
    unit: WorkflowUnit,
    plan_sha256: str,
    layout: WorkflowLayout,
    handler: HandlerSpec,
    expected_inputs: Mapping[str, str],
    output_hashes: Mapping[str, str],
    expected_execution: ExecutionIdentity,
) -> dict[str, Any]:
    expected_identity = {
        "workflow_id": plan.workflow_id,
        "plan_sha256": plan_sha256,
        "unit_id": unit.unit_id,
        "task": unit.task,
        "run_id": unit_identity_sha256(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id=unit.unit_id,
        ),
    }
    for field, expected in expected_identity.items():
        if getattr(draft, field) != expected:
            raise AdapterValidationError(
                f"attempt completion {field} differs from the workflow unit"
            )
    if draft.input_hashes != dict(expected_inputs):
        raise AdapterValidationError(
            "attempt completion input_hashes do not exactly bind the workflow unit"
        )
    if asdict(draft.execution) != asdict(expected_execution):
        raise AdapterValidationError(
            "attempt completion execution differs from the current validated attempt"
        )
    content_hashes = {identity.name: identity.sha256 for identity in unit.content_inputs}
    for completion_field, source_name in handler.config_hash_bindings.items():
        if getattr(draft, completion_field) != content_hashes[source_name]:
            raise AdapterValidationError(
                f"attempt completion {completion_field} differs from {source_name}"
            )
    if set(draft.scientific_validation) != set(handler.required_gate_names):
        raise AdapterValidationError(
            "attempt completion gate names differ from the handler contract"
        )
    output_paths = _expected_output_paths(
        plan, unit, plan_sha256=plan_sha256, layout=layout
    )
    marker = ScientificCompletion(
        workflow_id=draft.workflow_id,
        plan_sha256=draft.plan_sha256,
        unit_id=draft.unit_id,
        task=draft.task,
        run_id=draft.run_id,
        started_at=draft.started_at,
        completed_at=draft.completed_at,
        scientific_config_sha256=draft.scientific_config_sha256,
        execution_config_sha256=draft.execution_config_sha256,
        resolved_config_sha256=draft.resolved_config_sha256,
        input_hashes=dict(draft.input_hashes),
        output_files={
            str(output_paths[name]): output_hashes[name] for name in unit.output_names
        },
        scientific_validation=dict(draft.scientific_validation),
        execution=draft.execution,
    )
    try:
        payload = completion_payload(marker)
    except (TypeError, ValueError) as error:
        raise AdapterValidationError(
            f"attempt completion cannot form a scientific marker: {error}"
        ) from error
    context = CompletionValidationContext(
        workflow_id=plan.workflow_id,
        plan_sha256=plan_sha256,
        unit_id=unit.unit_id,
        task=unit.task,
        expected_input_hashes=expected_inputs,
        expected_output_paths=output_paths,
        expected_execution=expected_execution,
        reused=False,
    )
    try:
        result = handler.semantic_validator(payload, context)
    except Exception as error:
        raise AdapterValidationError(
            f"handler semantic validator {handler.semantic_validator_id!r} failed: {error}"
        ) from error
    if result is not None:
        raise AdapterValidationError("handler semantic validator must return None or raise")
    return payload


def publish_output_attempt(
    plan: WorkflowPlan,
    unit: WorkflowUnit,
    attempt: OutputAttempt,
    *,
    plan_sha256: str,
    layout: WorkflowLayout,
    expected_execution: ExecutionIdentity,
    handler_registry: Mapping[str, HandlerSpec] = HANDLER_REGISTRY,
    task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
) -> dict[str, Any]:
    """Publish one complete output directory no-clobber, then its marker last."""

    plan.validate(task_registry=task_registry)
    if plan.sha256(task_registry=task_registry) != plan_sha256:
        raise AdapterValidationError("output publication received the wrong workflow plan hash")
    if (
        attempt.workflow_id != plan.workflow_id
        or attempt.plan_sha256 != plan_sha256
        or attempt.unit_id != unit.unit_id
        or attempt.job_id != expected_execution.job_id
        or attempt.attempt != expected_execution.attempt
    ):
        raise AdapterValidationError("output attempt differs from the workflow execution")
    attempt.assert_path_identity()
    handler = require_handler(unit.task, registry=handler_registry)
    handler.validate_unit_contract(unit)
    cache: dict[str, dict[str, Any]] = {}
    expected_inputs = _expected_input_hashes(
        plan,
        unit,
        plan_sha256=plan_sha256,
        layout=layout,
        handler_registry=handler_registry,
        task_registry=task_registry,
        cache=cache,
    )
    output_hashes = _hash_attempt_outputs(attempt, unit)
    draft = _read_attempt_draft(attempt)
    payload = _completion_from_draft(
        draft,
        plan=plan,
        unit=unit,
        plan_sha256=plan_sha256,
        layout=layout,
        handler=handler,
        expected_inputs=expected_inputs,
        output_hashes=output_hashes,
        expected_execution=expected_execution,
    )

    try:
        os.unlink(ATTEMPT_COMPLETION_NAME, dir_fd=attempt.descriptor)
        os.fsync(attempt.descriptor)
    except OSError as error:
        raise AdapterValidationError(
            f"cannot seal output attempt before publication: {error}"
        ) from error
    if set(os.listdir(attempt.descriptor)) != set(unit.output_names):
        raise AdapterValidationError("sealed output attempt entries changed")
    final_directory = layout.output_directory(
        workflow_id=plan.workflow_id,
        plan_sha256=plan_sha256,
        unit_id=unit.unit_id,
    )
    try:
        publish_path_no_clobber(attempt.path, final_directory)
    except (OSError, TypeError, ValueError) as error:
        raise AdapterValidationError(
            f"cannot publish output directory without clobbering: {error}"
        ) from error

    marker_path = _completion_path(
        plan, unit, plan_sha256=plan_sha256, layout=layout
    )
    try:
        publish_bytes_once(
            marker_path,
            published_json_bytes(payload),
            approved_root=layout.data_root,
            context="OPD ScientificCompletion",
        )
    except (OSError, TypeError, ValueError) as error:
        raise AdapterValidationError(
            f"cannot publish OPD ScientificCompletion last: {error}"
        ) from error
    return validate_unit_completion(
        plan,
        unit,
        plan_sha256=plan_sha256,
        layout=layout,
        handler_registry=handler_registry,
        expected_execution=expected_execution,
        task_registry=task_registry,
    )


def _read_completion(path: Path, *, layout: WorkflowLayout) -> dict[str, Any]:
    validate_path_chain(path, approved_roots=(layout.data_root,), final_kind="file")
    payload, _digest = read_strict_json(
        path, context="OPD ScientificCompletion", max_bytes=4 * 1024 * 1024
    )
    if not isinstance(payload, dict):
        raise AdapterValidationError("OPD ScientificCompletion must be an object")
    try:
        # Retain the project-wide schema/domain validator, then independently
        # re-hash every output through descriptor-walk no-follow primitives.
        return validate_completion_payload(
            payload,
            marker_path=path,
            approved_roots=(layout.data_root, layout.scratch_root),
        )
    except (OSError, TypeError, ValueError) as error:
        raise AdapterValidationError(f"invalid OPD ScientificCompletion: {error}") from error


def _validate_output_digests(output_files: Mapping[str, str]) -> None:
    for raw_path, expected in output_files.items():
        descriptor = open_regular_file_nofollow(
            Path(raw_path), context="declared scientific output"
        )
        try:
            observed = sha256_descriptor(
                descriptor, context="declared scientific output"
            )
        finally:
            os.close(descriptor)
        if observed != expected:
            raise AdapterValidationError(
                f"declared scientific output hash mismatch: {raw_path}"
            )


def _validate_execution(
    payload: Mapping[str, Any], *, expected: ExecutionIdentity | None
) -> None:
    observed = payload.get("execution")
    if not isinstance(observed, dict):
        raise AdapterValidationError(
            "central ScientificCompletion evidence requires non-empty execution provenance"
        )
    if expected is not None and observed != asdict(expected):
        raise AdapterValidationError(
            "new ScientificCompletion execution does not match the current validated attempt"
        )


def validate_unit_completion(
    plan: WorkflowPlan,
    unit: WorkflowUnit,
    *,
    plan_sha256: str,
    layout: WorkflowLayout,
    handler_registry: Mapping[str, HandlerSpec] = HANDLER_REGISTRY,
    expected_execution: ExecutionIdentity | None = None,
    task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
    _cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one marker; dependencies are reusable prior valid executions."""

    plan.validate(task_registry=task_registry)
    if plan.sha256(task_registry=task_registry) != plan_sha256:
        raise AdapterValidationError("completion validation received the wrong workflow plan hash")
    cache = {} if _cache is None else _cache
    if unit.unit_id in cache:
        return cache[unit.unit_id]
    handler = require_handler(unit.task, registry=handler_registry)
    handler.validate_unit_contract(unit)

    expected_inputs = _expected_input_hashes(
        plan,
        unit,
        plan_sha256=plan_sha256,
        layout=layout,
        handler_registry=handler_registry,
        task_registry=task_registry,
        cache=cache,
    )

    marker_path = _completion_path(plan, unit, plan_sha256=plan_sha256, layout=layout)
    payload = _read_completion(marker_path, layout=layout)
    expected_identity = {
        "workflow_id": plan.workflow_id,
        "plan_sha256": plan_sha256,
        "unit_id": unit.unit_id,
        "task": unit.task,
        "run_id": unit_identity_sha256(
            workflow_id=plan.workflow_id,
            plan_sha256=plan_sha256,
            unit_id=unit.unit_id,
        ),
    }
    for name, expected in expected_identity.items():
        if payload.get(name) != expected:
            raise AdapterValidationError(
                f"ScientificCompletion {name} does not match the workflow unit"
            )
    if payload.get("input_hashes") != expected_inputs:
        raise AdapterValidationError(
            "ScientificCompletion input_hashes do not exactly bind unit content and dependencies"
        )
    output_paths = _expected_output_paths(
        plan, unit, plan_sha256=plan_sha256, layout=layout
    )
    observed_outputs = payload.get("output_files")
    if not isinstance(observed_outputs, dict) or set(observed_outputs) != {
        str(path) for path in output_paths.values()
    }:
        raise AdapterValidationError(
            "ScientificCompletion output_files differ from deterministic handler outputs"
        )
    _validate_output_digests(observed_outputs)
    _validate_execution(payload, expected=expected_execution)

    content_hashes = {identity.name: identity.sha256 for identity in unit.content_inputs}
    for completion_field, source_name in handler.config_hash_bindings.items():
        if payload.get(completion_field) != content_hashes[source_name]:
            raise AdapterValidationError(
                f"ScientificCompletion {completion_field} differs from {source_name}"
            )
    validation = payload.get("scientific_validation")
    if not isinstance(validation, dict) or set(validation) != set(
        handler.required_gate_names
    ):
        raise AdapterValidationError(
            "ScientificCompletion gate names differ from the handler contract"
        )
    context = CompletionValidationContext(
        workflow_id=plan.workflow_id,
        plan_sha256=plan_sha256,
        unit_id=unit.unit_id,
        task=unit.task,
        expected_input_hashes=expected_inputs,
        expected_output_paths=output_paths,
        expected_execution=expected_execution,
        reused=expected_execution is None,
    )
    try:
        result = handler.semantic_validator(payload, context)
    except Exception as error:
        raise AdapterValidationError(
            f"handler semantic validator {handler.semantic_validator_id!r} failed: {error}"
        ) from error
    if result is not None:
        raise AdapterValidationError("handler semantic validator must return None or raise")
    cache[unit.unit_id] = payload
    return payload


def validate_unit_dependencies(
    plan: WorkflowPlan,
    unit: WorkflowUnit,
    *,
    plan_sha256: str,
    layout: WorkflowLayout,
    handler_registry: Mapping[str, HandlerSpec] = HANDLER_REGISTRY,
    task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
) -> None:
    cache: dict[str, dict[str, Any]] = {}
    for dependency in unit.dependencies:
        validate_unit_completion(
            plan,
            plan.unit(dependency.unit_id, task_registry=task_registry),
            plan_sha256=plan_sha256,
            layout=layout,
            handler_registry=handler_registry,
            expected_execution=None,
            task_registry=task_registry,
            _cache=cache,
        )
