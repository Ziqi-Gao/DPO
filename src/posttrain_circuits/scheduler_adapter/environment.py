"""Cross-validation of scheduler-owned environment and allocation visibility."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.manifest import IDENTIFIER, RunningManifest


THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
INTEGER_TEXT = re.compile(r"0|[1-9][0-9]*\Z")


def _required(environ: MutableMapping[str, str], key: str) -> str:
    value = environ.get(key)
    if value is None or value == "":
        raise AdapterValidationError(f"required scheduler environment variable is missing: {key}")
    return value


def _integer_text(value: str, *, key: str, minimum: int = 0) -> int:
    if not INTEGER_TEXT.fullmatch(value):
        raise AdapterValidationError(f"{key} must be a canonical non-negative integer")
    parsed = int(value)
    if parsed < minimum:
        raise AdapterValidationError(f"{key} is below its allowed minimum")
    return parsed


def _csv(value: str, *, key: str) -> tuple[str, ...]:
    if value == "":
        return ()
    parts = tuple(value.split(","))
    if any(not part or part.strip() != part for part in parts):
        raise AdapterValidationError(f"{key} is not canonical comma-separated text")
    if len(set(parts)) != len(parts):
        raise AdapterValidationError(f"{key} contains duplicates")
    return parts


def _csv_integers(value: str, *, key: str) -> tuple[int, ...]:
    return tuple(_integer_text(part, key=key) for part in _csv(value, key=key))


@dataclass(frozen=True)
class RuntimeEnvelope:
    """Validated attempt data safe to retain; the lease is deliberately absent."""

    job_id: str
    attempt: int
    execution_profile: str
    manifest_path: Path
    manifest_sha256: str
    allocation_sha256: str


def validate_scheduler_environment(
    manifest: RunningManifest,
    *,
    manifest_path: Path,
    environ: MutableMapping[str, str] | None = None,
) -> RuntimeEnvelope:
    environment = os.environ if environ is None else environ
    expected_manifest_path = Path(manifest_path)
    if not expected_manifest_path.is_absolute() or ".." in expected_manifest_path.parts:
        raise AdapterValidationError("--job-manifest path must be absolute and normalized")

    exact_strings = {
        "SERVER_SCHEDULER_JOB_ID": manifest.job_id,
        "SERVER_SCHEDULER_PROJECT": manifest.project,
        "SERVER_SCHEDULER_JOB_MANIFEST": str(expected_manifest_path),
        "SERVER_SCHEDULER_EXECUTION_PROFILE": manifest.execution_profile,
        "SERVER_SCHEDULER_CPU_CORES": str(manifest.allocation.cpu_cores),
        "SERVER_SCHEDULER_CPUSET": ",".join(str(value) for value in manifest.cpu_ids),
        "SERVER_SCHEDULER_MEMORY_MIB": str(manifest.allocation.memory_mib),
        "SERVER_SCHEDULER_GPU_COUNT": str(manifest.allocation.gpu_count),
        "SERVER_SCHEDULER_GPU_MEMORY_MIB": str(manifest.allocation.gpu_memory_mib),
        "SERVER_SCHEDULER_GPU_UTILIZATION_PCT": str(
            manifest.allocation.gpu_utilization_pct
        ),
        "SERVER_SCHEDULER_GPU_EXCLUSIVITY": (
            "exclusive" if manifest.allocation.exclusive_gpu else "shared"
        ),
        "SERVER_SCHEDULER_GPU_INDICES": ",".join(
            str(value) for value in manifest.gpu_indices
        ),
        "SERVER_SCHEDULER_GPU_UUIDS": ",".join(manifest.gpu_uuids),
        "SERVER_SCHEDULER_GPU_PCI_BUS_IDS": ",".join(manifest.gpu_pci_bus_ids),
        "SERVER_SCHEDULER_STDOUT_LOG": manifest.stdout_log,
        "SERVER_SCHEDULER_STDERR_LOG": manifest.stderr_log,
    }
    for key, expected in exact_strings.items():
        observed = environment.get(key)
        # Empty scheduler CSV fields are exported intentionally and must still be present.
        if observed is None:
            raise AdapterValidationError(f"required scheduler environment variable is missing: {key}")
        if observed != expected:
            raise AdapterValidationError(f"{key} does not match the running manifest")

    runtime_text = _required(environment, "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS")
    if runtime_text != f"{manifest.estimated_runtime_seconds:.6f}":
        raise AdapterValidationError(
            "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS does not match the manifest"
        )

    if manifest.numa_node is None:
        if "SERVER_SCHEDULER_NUMA_NODE" in environment:
            raise AdapterValidationError(
                "SERVER_SCHEDULER_NUMA_NODE is set for an allocation without a NUMA node"
            )
    elif environment.get("SERVER_SCHEDULER_NUMA_NODE") != str(manifest.numa_node):
        raise AdapterValidationError("SERVER_SCHEDULER_NUMA_NODE does not match the manifest")

    # Parse the exported forms as well as comparing text, so non-canonical or
    # ambiguous representations cannot accidentally pass future refactors.
    if _csv_integers(
        environment["SERVER_SCHEDULER_CPUSET"], key="SERVER_SCHEDULER_CPUSET"
    ) != manifest.cpu_ids:
        raise AdapterValidationError("scheduler CPU set does not match the allocation")
    if _csv_integers(
        environment["SERVER_SCHEDULER_GPU_INDICES"],
        key="SERVER_SCHEDULER_GPU_INDICES",
    ) != manifest.gpu_indices:
        raise AdapterValidationError("scheduler GPU indices do not match the allocation")
    if _csv(
        environment["SERVER_SCHEDULER_GPU_UUIDS"], key="SERVER_SCHEDULER_GPU_UUIDS"
    ) != manifest.gpu_uuids:
        raise AdapterValidationError("scheduler GPU UUIDs do not match the allocation")
    if _csv(
        environment["SERVER_SCHEDULER_GPU_PCI_BUS_IDS"],
        key="SERVER_SCHEDULER_GPU_PCI_BUS_IDS",
    ) != manifest.gpu_pci_bus_ids:
        raise AdapterValidationError("scheduler GPU PCI bus IDs do not match the allocation")

    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if manifest.allocation.gpu_count == 0:
        if visible != "":
            raise AdapterValidationError(
                "CPU-only allocation requires CUDA_VISIBLE_DEVICES to be absent or empty"
            )
        if environment.get("CUDA_DEVICE_ORDER", "") != "":
            raise AdapterValidationError(
                "CPU-only allocation requires CUDA_DEVICE_ORDER to be absent or empty"
            )
    else:
        if _csv(visible, key="CUDA_VISIBLE_DEVICES") != manifest.gpu_uuids:
            raise AdapterValidationError(
                "CUDA_VISIBLE_DEVICES must be exactly the scheduler-provided GPU UUIDs"
            )
        if environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise AdapterValidationError(
                "GPU allocation requires scheduler-provided CUDA_DEVICE_ORDER=PCI_BUS_ID"
            )

    lease_id = _required(environment, "SERVER_SCHEDULER_LEASE_ID")
    if not IDENTIFIER.fullmatch(lease_id):
        raise AdapterValidationError("SERVER_SCHEDULER_LEASE_ID is not a valid identifier")
    attempt = _integer_text(
        _required(environment, "SERVER_SCHEDULER_ATTEMPT"),
        key="SERVER_SCHEDULER_ATTEMPT",
        minimum=1,
    )
    return RuntimeEnvelope(
        job_id=manifest.job_id,
        attempt=attempt,
        execution_profile=manifest.execution_profile,
        manifest_path=expected_manifest_path,
        manifest_sha256=manifest.manifest_sha256,
        allocation_sha256=sha256_value(manifest.allocation_payload()),
    )


def configure_thread_environment(
    *,
    cpu_cores: int,
    process_count: int,
    environ: MutableMapping[str, str] | None = None,
) -> int:
    """Set and verify an equal, explicit per-rank CPU thread allocation."""

    environment = os.environ if environ is None else environ
    if isinstance(cpu_cores, bool) or not isinstance(cpu_cores, int) or cpu_cores < 1:
        raise AdapterValidationError("cpu_cores must be a positive integer")
    if (
        isinstance(process_count, bool)
        or not isinstance(process_count, int)
        or process_count < 1
    ):
        raise AdapterValidationError("process_count must be a positive integer")
    if process_count > cpu_cores or cpu_cores % process_count != 0:
        raise AdapterValidationError(
            "allocated CPU cores must divide equally across the fixed process count"
        )
    threads_per_rank = cpu_cores // process_count
    expected = str(threads_per_rank)
    for key in THREAD_ENVIRONMENT_KEYS:
        environment[key] = expected
    for key in THREAD_ENVIRONMENT_KEYS:
        if environment.get(key) != expected:
            raise AdapterValidationError(f"failed to establish validated thread variable {key}")
    return threads_per_rank
