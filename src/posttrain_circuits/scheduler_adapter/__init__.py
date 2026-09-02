"""Dependency-light OPD boundary adapter for ServerScheduler protocol v2.

This package initializer intentionally imports neither workflow implementations
nor torch.  The executable entrypoint validates the scheduler boundary first.
"""

from posttrain_circuits.scheduler_adapter.errors import (
    AdapterError,
    AdapterValidationError,
    DispatchError,
)
from posttrain_circuits.scheduler_adapter.manifest import (
    Allocation,
    RunningManifest,
    WorkflowParameters,
    load_running_manifest,
)

__all__ = [
    "AdapterError",
    "AdapterValidationError",
    "Allocation",
    "DispatchError",
    "RunningManifest",
    "WorkflowParameters",
    "load_running_manifest",
]
