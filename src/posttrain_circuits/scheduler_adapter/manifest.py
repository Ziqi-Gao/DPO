"""Strict protocol-v2 running-manifest contract for OPD."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json


PROTOCOL_VERSION = 2
PROJECT = "OPD"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PARAMETER_KEYS = frozenset({"workflow_id", "plan_sha256", "unit_id"})
RUNNING_MANIFEST_KEYS = frozenset(
    {
        "allocation",
        "cpu_ids",
        "estimated_runtime_seconds",
        "execution_profile",
        "exit_code",
        "failure_reason",
        "gpu_indices",
        "gpu_pci_bus_ids",
        "gpu_uuids",
        "job_id",
        "numa_node",
        "parameters",
        "priority",
        "project",
        "requested_profile",
        "resources",
        "schema_version",
        "state",
        "stderr_log",
        "stdout_log",
        "submitted_at",
        "task",
        "updated_at",
    }
)
RESOURCE_KEYS = frozenset(
    {
        "cpu_cores",
        "exclusive_gpu",
        "gpu_count",
        "gpu_memory_mib",
        "gpu_utilization_pct",
        "memory_mib",
    }
)


def _exact_keys(payload: Mapping[str, Any], expected: frozenset[str], *, context: str) -> None:
    observed = frozenset(payload)
    if observed != expected:
        raise AdapterValidationError(
            f"{context} fields differ from protocol-v2: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise AdapterValidationError(f"{name} is not a valid identifier")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise AdapterValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterValidationError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise AdapterValidationError(f"{name} is outside its allowed range")
    return value


def _positive_number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise AdapterValidationError(f"{name} must be a finite positive number")
    return float(value)


def _timestamp(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise AdapterValidationError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AdapterValidationError(f"{name} must be an ISO-8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AdapterValidationError(f"{name} must include a timezone")
    return value


def _absolute_path_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise AdapterValidationError(f"{name} must be an absolute path string")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise AdapterValidationError(f"{name} must be absolute and normalized")
    return value


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise AdapterValidationError(f"{name} must be an array")
    result = tuple(_integer(item, name=name) for item in value)
    if len(set(result)) != len(result):
        raise AdapterValidationError(f"{name} must not contain duplicates")
    return result


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str)
        or not item
        or item.strip() != item
        or "," in item
        for item in value
    ):
        raise AdapterValidationError(f"{name} must be an array of non-empty strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise AdapterValidationError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class Allocation:
    cpu_cores: int
    memory_mib: int
    gpu_count: int
    gpu_memory_mib: int
    gpu_utilization_pct: int
    exclusive_gpu: bool

    @classmethod
    def from_payload(cls, payload: object, *, context: str) -> "Allocation":
        if not isinstance(payload, dict):
            raise AdapterValidationError(f"{context} must be an object")
        _exact_keys(payload, RESOURCE_KEYS, context=context)
        exclusive_gpu = payload["exclusive_gpu"]
        if not isinstance(exclusive_gpu, bool):
            raise AdapterValidationError(f"{context}.exclusive_gpu must be boolean")
        allocation = cls(
            cpu_cores=_integer(payload["cpu_cores"], name=f"{context}.cpu_cores", minimum=1),
            memory_mib=_integer(payload["memory_mib"], name=f"{context}.memory_mib", minimum=1),
            gpu_count=_integer(payload["gpu_count"], name=f"{context}.gpu_count"),
            gpu_memory_mib=_integer(
                payload["gpu_memory_mib"], name=f"{context}.gpu_memory_mib"
            ),
            gpu_utilization_pct=_integer(
                payload["gpu_utilization_pct"],
                name=f"{context}.gpu_utilization_pct",
                maximum=100,
            ),
            exclusive_gpu=exclusive_gpu,
        )
        if allocation.gpu_count == 0 and (
            allocation.gpu_memory_mib != 0 or allocation.gpu_utilization_pct != 0
        ):
            raise AdapterValidationError(f"{context} CPU allocation reserves GPU capacity")
        if allocation.gpu_count > 0 and (
            allocation.gpu_memory_mib == 0 or allocation.gpu_utilization_pct == 0
        ):
            raise AdapterValidationError(f"{context} GPU allocation lacks per-device capacity")
        return allocation

    def to_payload(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "exclusive_gpu": self.exclusive_gpu,
            "gpu_count": self.gpu_count,
            "gpu_memory_mib": self.gpu_memory_mib,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "memory_mib": self.memory_mib,
        }


@dataclass(frozen=True)
class WorkflowParameters:
    workflow_id: str
    plan_sha256: str
    unit_id: str

    @classmethod
    def from_payload(cls, payload: object) -> "WorkflowParameters":
        if not isinstance(payload, dict):
            raise AdapterValidationError("running manifest parameters must be an object")
        _exact_keys(payload, PARAMETER_KEYS, context="running manifest parameters")
        return cls(
            workflow_id=_identifier(payload["workflow_id"], name="parameters.workflow_id"),
            plan_sha256=_sha256(payload["plan_sha256"], name="parameters.plan_sha256"),
            unit_id=_identifier(payload["unit_id"], name="parameters.unit_id"),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "plan_sha256": self.plan_sha256,
            "unit_id": self.unit_id,
            "workflow_id": self.workflow_id,
        }


@dataclass(frozen=True)
class RunningManifest:
    job_id: str
    task: str
    parameters: WorkflowParameters
    allocation: Allocation
    requested_resources: Allocation | None
    priority: int
    submitted_at: str
    updated_at: str
    requested_profile: str | None
    execution_profile: str
    estimated_runtime_seconds: float
    gpu_indices: tuple[int, ...]
    gpu_uuids: tuple[str, ...]
    gpu_pci_bus_ids: tuple[str, ...]
    cpu_ids: tuple[int, ...]
    numa_node: int | None
    stdout_log: str
    stderr_log: str
    manifest_sha256: str
    schema_version: int = PROTOCOL_VERSION
    project: str = PROJECT
    state: str = "running"

    @classmethod
    def from_payload(cls, payload: object, *, manifest_sha256: str) -> "RunningManifest":
        if not isinstance(payload, dict):
            raise AdapterValidationError("running manifest must be an object")
        _exact_keys(payload, RUNNING_MANIFEST_KEYS, context="running manifest")
        if payload["schema_version"] != PROTOCOL_VERSION or isinstance(
            payload["schema_version"], bool
        ):
            raise AdapterValidationError("running manifest schema_version must be 2")
        if payload["project"] != PROJECT:
            raise AdapterValidationError("running manifest project must be OPD")
        if payload["state"] != "running":
            raise AdapterValidationError("job manifest must be in running state")
        if payload["exit_code"] is not None or payload["failure_reason"] is not None:
            raise AdapterValidationError("running manifest cannot contain terminal outcome fields")
        execution_profile = _identifier(
            payload["execution_profile"], name="execution_profile"
        )
        requested_profile = payload["requested_profile"]
        if requested_profile is not None:
            requested_profile = _identifier(requested_profile, name="requested_profile")
        if requested_profile is not None and requested_profile != execution_profile:
            raise AdapterValidationError(
                "running execution_profile does not match the explicitly requested profile"
            )
        raw_requested = payload["resources"]
        allocation = Allocation.from_payload(payload["allocation"], context="allocation")
        manifest = cls(
            job_id=_identifier(payload["job_id"], name="job_id"),
            task=_identifier(payload["task"], name="task"),
            parameters=WorkflowParameters.from_payload(payload["parameters"]),
            allocation=allocation,
            requested_resources=(
                None
                if raw_requested is None
                else Allocation.from_payload(raw_requested, context="resources")
            ),
            priority=_integer(payload["priority"], name="priority", minimum=-100, maximum=100),
            submitted_at=_timestamp(payload["submitted_at"], name="submitted_at"),
            updated_at=_timestamp(payload["updated_at"], name="updated_at"),
            requested_profile=requested_profile,
            execution_profile=execution_profile,
            estimated_runtime_seconds=_positive_number(
                payload["estimated_runtime_seconds"], name="estimated_runtime_seconds"
            ),
            gpu_indices=_integer_tuple(payload["gpu_indices"], name="gpu_indices"),
            gpu_uuids=_string_tuple(payload["gpu_uuids"], name="gpu_uuids"),
            gpu_pci_bus_ids=_string_tuple(
                payload["gpu_pci_bus_ids"], name="gpu_pci_bus_ids"
            ),
            cpu_ids=_integer_tuple(payload["cpu_ids"], name="cpu_ids"),
            numa_node=(
                None
                if payload["numa_node"] is None
                else _integer(payload["numa_node"], name="numa_node")
            ),
            stdout_log=_absolute_path_string(payload["stdout_log"], name="stdout_log"),
            stderr_log=_absolute_path_string(payload["stderr_log"], name="stderr_log"),
            manifest_sha256=_sha256(manifest_sha256, name="manifest_sha256"),
        )
        manifest.validate_allocation()
        return manifest

    def validate_allocation(self) -> None:
        if len(self.cpu_ids) != self.allocation.cpu_cores:
            raise AdapterValidationError(
                "running manifest cpu_ids length must equal allocation.cpu_cores"
            )
        gpu_count = self.allocation.gpu_count
        if gpu_count == 0:
            if self.gpu_indices or self.gpu_uuids or self.gpu_pci_bus_ids:
                raise AdapterValidationError("CPU allocation contains GPU placement")
            return
        if len(self.gpu_indices) != gpu_count:
            raise AdapterValidationError("GPU indices length must equal allocation.gpu_count")
        if len(self.gpu_uuids) != gpu_count:
            raise AdapterValidationError(
                "GPU allocation requires one scheduler-provided UUID per visible device"
            )
        if len(self.gpu_pci_bus_ids) != gpu_count:
            raise AdapterValidationError(
                "GPU allocation requires one scheduler-provided PCI bus ID per device"
            )

    def allocation_payload(self) -> dict[str, Any]:
        return {
            "allocation": self.allocation.to_payload(),
            "cpu_ids": list(self.cpu_ids),
            "gpu_indices": list(self.gpu_indices),
            "gpu_pci_bus_ids": list(self.gpu_pci_bus_ids),
            "gpu_uuids": list(self.gpu_uuids),
            "numa_node": self.numa_node,
        }


def load_running_manifest(path: Path) -> RunningManifest:
    payload, digest = read_strict_json(
        Path(path), context="ServerScheduler running manifest", max_bytes=2 * 1024 * 1024
    )
    return RunningManifest.from_payload(payload, manifest_sha256=digest)
