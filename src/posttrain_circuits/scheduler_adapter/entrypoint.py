"""Sole foreground ServerScheduler entrypoint for OPD protocol-v2 jobs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import MutableMapping, Sequence

from posttrain_circuits.scheduler_adapter.dispatch import propagate_child_signal
from posttrain_circuits.scheduler_adapter.environment import (
    configure_thread_environment,
    validate_scheduler_environment,
)
from posttrain_circuits.scheduler_adapter.errors import AdapterError, AdapterValidationError
from posttrain_circuits.scheduler_adapter.manifest import load_running_manifest
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.registry import (
    ADDITIONAL_RUNTIME_ROOTS,
    FIXED_RUNTIME_ROOT,
    HANDLER_REGISTRY,
    require_handler,
)


def _parse_manifest_argument(argv: Sequence[str]) -> Path:
    arguments = tuple(argv)
    if len(arguments) != 2 or arguments[0] != "--job-manifest":
        raise AdapterValidationError(
            "entrypoint requires exactly: --job-manifest ABSOLUTE_RUNNING_JSON"
        )
    path = Path(arguments[1])
    if not path.is_absolute() or ".." in path.parts:
        raise AdapterValidationError("--job-manifest must be an absolute normalized path")
    return path


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> int:
    """Validate the entire scheduler boundary before importing workflow code."""

    arguments = sys.argv[1:] if argv is None else argv
    environment = os.environ if environ is None else environ
    try:
        manifest_path = _parse_manifest_argument(arguments)
        manifest = load_running_manifest(manifest_path)
        envelope = validate_scheduler_environment(
            manifest, manifest_path=manifest_path, environ=environment
        )
        handler = require_handler(manifest.task, registry=HANDLER_REGISTRY)
        layout = WorkflowLayout.production()
        layout.validate()
        with handler.prepare(
            manifest,
            approved_code_root=layout.code_root,
            approved_runtime_root=FIXED_RUNTIME_ROOT,
            additional_runtime_roots=ADDITIONAL_RUNTIME_ROOTS,
            observed_gpu_models=None,
        ) as prepared:
            configure_thread_environment(
                cpu_cores=manifest.allocation.cpu_cores,
                process_count=prepared.profile.process_count,
                environ=environment,
            )

            # Keep this import below manifest/env/allocation/deployment/thread
            # validation. runtime imports workflows and completion code.
            from posttrain_circuits.scheduler_adapter.runtime import execute_validated_unit

            return_code = execute_validated_unit(
                manifest,
                envelope,
                prepared,
                layout=layout,
                handler_registry=HANDLER_REGISTRY,
                environ=environment,
                popen=subprocess.Popen,
            )
        return propagate_child_signal(return_code)
    except (AdapterError, OSError, TypeError, ValueError) as error:
        print(f"OPD ServerScheduler entrypoint rejected execution: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
