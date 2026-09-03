"""Lazy-loaded workflow resolution, held-fd execution, and completion gate."""

from __future__ import annotations

from typing import Mapping

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.scheduler_adapter.completion import (
    execution_identity,
    publish_output_attempt,
    validate_unit_completion,
    validate_unit_dependencies,
)
from posttrain_circuits.scheduler_adapter.config_resolver import (
    ConfigBindingResolver as CasConfigBindingResolver,
)
from posttrain_circuits.scheduler_adapter.content_store import ContentStore
from posttrain_circuits.scheduler_adapter.dispatch import PopenFactory, run_foreground_child
from posttrain_circuits.scheduler_adapter.environment import (
    THREAD_ENVIRONMENT_KEYS,
    RuntimeEnvelope,
)
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.manifest import RunningManifest
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.plan_store import resolve_workflow_unit
from posttrain_circuits.scheduler_adapter.registry import (
    ConfigBindingResolver,
    HANDLER_REGISTRY,
    HandlerSpec,
    PreparedHandler,
    require_handler,
)
from posttrain_circuits.scheduler_adapter.secure_files import regular_file_exists_nofollow


SCHEDULER_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        *THREAD_ENVIRONMENT_KEYS,
    }
)
REJECTED_AMBIENT_ENVIRONMENT = frozenset(
    {"LD_AUDIT", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"}
)


def build_child_environment(
    source: Mapping[str, str], *, handler: HandlerSpec
) -> dict[str, str]:
    """Build an explicit environment; no ambient path/injection variables survive."""

    injected = sorted(
        key for key in REJECTED_AMBIENT_ENVIRONMENT if source.get(key, "")
    )
    if injected:
        raise AdapterValidationError(
            f"ambient loader/Python injection variables are forbidden: {injected}"
        )
    missing_threads = sorted(key for key in THREAD_ENVIRONMENT_KEYS if key not in source)
    if missing_threads:
        raise AdapterValidationError(
            f"validated per-rank thread environment is incomplete: {missing_threads}"
        )
    child = {
        key: source[key]
        for key in sorted(SCHEDULER_ENVIRONMENT_ALLOWLIST)
        if key in source
    }
    child.update(handler.fixed_environment)
    child["PYTHONNOUSERSITE"] = "1"
    if any(key.startswith("SERVER_SCHEDULER_") for key in child):
        raise AdapterValidationError("scheduler attempt environment leaked to child")
    return child


def execute_validated_unit(
    manifest: RunningManifest,
    envelope: RuntimeEnvelope,
    prepared: PreparedHandler,
    *,
    layout: WorkflowLayout,
    handler_registry: Mapping[str, HandlerSpec] = HANDLER_REGISTRY,
    config_binding_resolver: ConfigBindingResolver | None = None,
    environ: Mapping[str, str],
    popen: PopenFactory,
) -> int:
    """Execute one fixed unit; ServerScheduler remains sole attempt-state owner."""

    if (
        envelope.job_id != manifest.job_id
        or envelope.execution_profile != manifest.execution_profile
        or envelope.manifest_sha256 != manifest.manifest_sha256
        or envelope.allocation_sha256
        != sha256_value(manifest.allocation_payload())
    ):
        raise AdapterValidationError("runtime envelope differs from the validated manifest")
    handler = prepared.spec
    if require_handler(manifest.task, registry=handler_registry) is not handler:
        raise AdapterValidationError("prepared handler is not the code-owned task registry entry")
    resolved = resolve_workflow_unit(manifest, layout=layout)
    handler.validate_unit_contract(resolved.unit)
    plan_sha256 = manifest.parameters.plan_sha256

    content_store = ContentStore(layout)
    resolver = (
        CasConfigBindingResolver(content_store)
        if config_binding_resolver is None
        else config_binding_resolver
    )
    if not callable(getattr(resolver, "resolve", None)):
        raise AdapterValidationError("config binding resolver is not code-owned callable")
    with content_store.open_inputs(resolved.unit.content_inputs) as content_inputs:
        resolver.resolve(
            manifest=manifest,
            plan=resolved.plan,
            unit=resolved.unit,
            content_handles=content_inputs.handles,
        )
        if regular_file_exists_nofollow(
            resolved.completion_path, context="existing OPD ScientificCompletion"
        ):
            validate_unit_completion(
                resolved.plan,
                resolved.unit,
                plan_sha256=plan_sha256,
                layout=layout,
                handler_registry=handler_registry,
                expected_execution=None,
            )
            return 0

        validate_unit_dependencies(
            resolved.plan,
            resolved.unit,
            plan_sha256=plan_sha256,
            layout=layout,
            handler_registry=handler_registry,
        )
        child_environment = build_child_environment(environ, handler=handler)
        with content_store.create_output_attempt(
            workflow_id=manifest.parameters.workflow_id,
            plan_sha256=plan_sha256,
            unit_id=manifest.parameters.unit_id,
            job_id=envelope.job_id,
            attempt=envelope.attempt,
        ) as output_attempt:
            exit_code = run_foreground_child(
                prepared.argv(
                    manifest,
                    envelope,
                    content_handles=content_inputs.handles,
                    output_attempt=output_attempt,
                ),
                cwd=prepared.cwd_proc_path,
                environ=child_environment,
                pass_fds=(
                    *prepared.pass_fds,
                    *content_inputs.pass_fds,
                    output_attempt.descriptor,
                ),
                popen=popen,
            )
            if exit_code != 0:
                return exit_code
            publish_output_attempt(
                resolved.plan,
                resolved.unit,
                output_attempt,
                plan_sha256=plan_sha256,
                layout=layout,
                handler_registry=handler_registry,
                expected_execution=execution_identity(envelope),
            )
    return 0
