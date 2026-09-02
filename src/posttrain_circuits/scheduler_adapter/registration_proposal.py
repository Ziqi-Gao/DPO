"""Exact disabled ServerScheduler registration proposal for repository preflight.

This project-owned module produces review material only.  It cannot write a
ServerScheduler registration, enable OPD, publish a request, or submit a job.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.paths import (
    PRODUCTION_CODE_ROOT,
    PRODUCTION_DATA_ROOT,
    PRODUCTION_SCRATCH_ROOT,
)
from posttrain_circuits.scheduler_adapter.repository_preflight import PROFILE_NAME
from posttrain_circuits.workflows.catalog import candidate_entrypoint
from posttrain_circuits.workflows.contracts import WORKFLOW_TASK_REGISTRY


REGISTRATION_SCHEMA_VERSION = 2
PROJECT_NAME = "OPD"
TASK_NAME = "repository_preflight"
CPU_PROFILE_NAME = PROFILE_NAME
INTEGRATION_STATUS = "protocol-v2-disabled-preflight-proposal"
ENTRYPOINT = PRODUCTION_CODE_ROOT / "scripts" / "server_scheduler" / "opd-entrypoint"
PROPOSED_HANDOFF_ROOT = PRODUCTION_SCRATCH_ROOT / "scheduler"

CPU_CORES = 1
MEMORY_MIB = 128
ESTIMATED_RUNTIME_SECONDS = 30.0

_NOTES = (
    "Disabled project-owned proposal only. repository_preflight accepts the fixed "
    "workflow_id/plan_sha256/unit_id parameter contract; job parameters cannot "
    "override commands, environment, project paths, or resources. CPU profile "
    "measured on 2026-09-02 with isolated Python 3.12.13: "
    "end-to-end execution plus completion reuse finished in 0.18 s wall time "
    "at 98% CPU with 23,552 KiB MaxRSS; 128 MiB and 30 s are conservative "
    "admission bounds. "
    "Implementation contract: "
    f"{candidate_entrypoint(TASK_NAME).module}."
)


def _proposal_payload() -> dict[str, Any]:
    if TASK_NAME not in WORKFLOW_TASK_REGISTRY:
        raise AdapterValidationError("repository preflight workflow contract is absent")
    return {
        "enabled": False,
        "integration": {
            "allowed_tasks": [TASK_NAME],
            "entrypoint": str(ENTRYPOINT),
            "notes": _NOTES,
        },
        "integration_status": INTEGRATION_STATUS,
        "name": PROJECT_NAME,
        "paths": {
            "code_root": str(PRODUCTION_CODE_ROOT),
            "data_root": str(PRODUCTION_DATA_ROOT),
            "proposed_handoff_root": str(PROPOSED_HANDOFF_ROOT),
            "scratch_root": str(PRODUCTION_SCRATCH_ROOT),
        },
        "policy": {
            "allow_cpu_jobs": True,
            "allow_gpu_jobs": False,
            "allow_process_signals": False,
            "allow_project_file_mutation": False,
        },
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "tasks": [
            {
                "execution_profiles": [
                    {
                        "cpu_cores_max": CPU_CORES,
                        "cpu_cores_min": CPU_CORES,
                        "cpu_cores_preferred": CPU_CORES,
                        "cpu_scaling_efficiency": 0.0,
                        "estimated_runtime_seconds": ESTIMATED_RUNTIME_SECONDS,
                        "gpu_count": 0,
                        "gpu_exclusivity": "shareable",
                        "gpu_memory_mib": 0,
                        "gpu_models": [],
                        "gpu_utilization_pct": 0,
                        "kind": "cpu",
                        "memory_mib": MEMORY_MIB,
                        "name": CPU_PROFILE_NAME,
                        "resource_mode": "fixed",
                        "scheduling_goal": "latency",
                    }
                ],
                "name": TASK_NAME,
                "selection_policy": "earliest_finish",
            }
        ],
    }


def build_disabled_registration_proposal() -> dict[str, Any]:
    """Return a fresh, exact protocol-v2 proposal with no caller overrides."""

    payload = _proposal_payload()
    validate_disabled_registration_proposal(payload)
    return copy.deepcopy(payload)


def build_registration_proposal_schema() -> dict[str, Any]:
    """Return the project-owned JSON Schema for the one allowed proposal."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://local.invalid/OPD/registration-proposal-v2.schema.json",
        "const": build_disabled_registration_proposal(),
        "title": "OPD disabled repository-preflight registration proposal v2",
    }


def _strict_match(observed: object, expected: object, *, context: str) -> None:
    if type(observed) is not type(expected):
        raise AdapterValidationError(f"{context} has the wrong JSON type")
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        observed_keys = frozenset(observed)
        expected_keys = frozenset(expected)
        if observed_keys != expected_keys:
            raise AdapterValidationError(
                f"{context} fields differ from the disabled proposal contract: "
                f"missing={sorted(expected_keys - observed_keys)}, "
                f"extra={sorted(observed_keys - expected_keys)}"
            )
        for key in expected:
            _strict_match(
                observed[key], expected[key], context=f"{context}.{key}"
            )
        return
    if isinstance(expected, list):
        assert isinstance(observed, list)
        if len(observed) != len(expected):
            raise AdapterValidationError(f"{context} has the wrong array length")
        for index, (observed_item, expected_item) in enumerate(zip(observed, expected)):
            _strict_match(
                observed_item,
                expected_item,
                context=f"{context}[{index}]",
            )
        return
    if observed != expected:
        raise AdapterValidationError(f"{context} differs from the disabled proposal contract")


def validate_disabled_registration_proposal(payload: object) -> None:
    """Fail closed unless *payload* is the exact disabled project proposal."""

    if not isinstance(payload, Mapping):
        raise AdapterValidationError("registration proposal must be an object")
    # Require a plain JSON object and exact nested JSON types.  This prevents
    # bool/int aliasing and rejects every unreviewed task, profile, or path.
    if type(payload) is not dict:
        raise AdapterValidationError("registration proposal must be a plain JSON object")
    _strict_match(payload, _proposal_payload(), context="registration proposal")


__all__ = [
    "CPU_PROFILE_NAME",
    "CPU_CORES",
    "ENTRYPOINT",
    "ESTIMATED_RUNTIME_SECONDS",
    "INTEGRATION_STATUS",
    "MEMORY_MIB",
    "PROJECT_NAME",
    "PROPOSED_HANDOFF_ROOT",
    "REGISTRATION_SCHEMA_VERSION",
    "TASK_NAME",
    "build_disabled_registration_proposal",
    "build_registration_proposal_schema",
    "validate_disabled_registration_proposal",
]
