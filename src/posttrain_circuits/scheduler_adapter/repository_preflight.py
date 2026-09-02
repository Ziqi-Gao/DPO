"""Scientific result validation for the CPU-only repository preflight."""

from __future__ import annotations

from typing import Any, Mapping

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json


TASK_NAME = "repository_preflight"
OUTPUT_NAME = "preflight_report.json"
PROFILE_NAME = "repository-preflight-cpu"
GATE_NAMES = (
    "config_binding",
    "no_gpu_required",
    "runtime_isolation",
)
REPORT_FIELDS = frozenset(
    {
        "allocation_sha256",
        "completed_at",
        "execution",
        "execution_config_sha256",
        "input_hashes",
        "manifest_sha256",
        "plan_sha256",
        "project",
        "report_kind",
        "resolved_config_sha256",
        "run_id",
        "schema_version",
        "scientific_config_sha256",
        "scientific_validation",
        "started_at",
        "task",
        "unit_id",
        "workflow_id",
    }
)


def validate_repository_preflight_completion(
    completion: Mapping[str, Any], context: Any
) -> None:
    """Require the report to reproduce every scheduler-neutral completion fact."""

    if not isinstance(completion, Mapping):
        raise AdapterValidationError("repository preflight completion must be a mapping")
    output_paths = getattr(context, "expected_output_paths", None)
    if not isinstance(output_paths, Mapping) or set(output_paths) != {OUTPUT_NAME}:
        raise AdapterValidationError("repository preflight output contract is invalid")
    expected_inputs = getattr(context, "expected_input_hashes", None)
    if not isinstance(expected_inputs, Mapping):
        raise AdapterValidationError("repository preflight expected inputs are invalid")
    if completion.get("input_hashes") != dict(expected_inputs):
        raise AdapterValidationError("repository preflight completion input hashes differ")
    gates = completion.get("scientific_validation")
    if not isinstance(gates, dict) or tuple(sorted(gates)) != GATE_NAMES:
        raise AdapterValidationError("repository preflight gate contract is invalid")
    if any(value is not True for value in gates.values()):
        raise AdapterValidationError("repository preflight scientific validation failed")

    # The adapter invokes the semantic validator once while outputs are still
    # held in staging and again after their atomic publication. Completion and
    # gate semantics are checked both times; report bytes are checked when the
    # deterministic final path becomes available.
    try:
        output_paths[OUTPUT_NAME].lstat()
    except FileNotFoundError:
        return
    report, _file_sha256 = read_strict_json(
        output_paths[OUTPUT_NAME],
        context="repository preflight report",
        max_bytes=8 * 1024 * 1024,
    )
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
        raise AdapterValidationError("repository preflight report fields are invalid")
    if (
        report.get("schema_version") != 1
        or type(report.get("schema_version")) is not int
        or report.get("project") != "OPD"
        or report.get("task") != TASK_NAME
        or report.get("report_kind") != TASK_NAME
    ):
        raise AdapterValidationError("repository preflight report identity is invalid")

    if report.get("input_hashes") != dict(expected_inputs):
        raise AdapterValidationError("repository preflight report input hashes differ")
    paired_fields = (
        "completed_at",
        "execution",
        "execution_config_sha256",
        "input_hashes",
        "plan_sha256",
        "resolved_config_sha256",
        "run_id",
        "scientific_config_sha256",
        "scientific_validation",
        "started_at",
        "task",
        "unit_id",
        "workflow_id",
    )
    if any(report.get(field) != completion.get(field) for field in paired_fields):
        raise AdapterValidationError(
            "repository preflight report differs from its completion marker"
        )
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise AdapterValidationError("repository preflight execution identity is invalid")
    if (
        report.get("manifest_sha256") != execution.get("manifest_sha256")
        or report.get("allocation_sha256") != execution.get("allocation_sha256")
    ):
        raise AdapterValidationError(
            "repository preflight report duplicates inconsistent execution identities"
        )
    report_gates = report.get("scientific_validation")
    if not isinstance(report_gates, dict) or tuple(sorted(report_gates)) != GATE_NAMES:
        raise AdapterValidationError("repository preflight gate contract is invalid")
    if any(value is not True for value in report_gates.values()):
        raise AdapterValidationError("repository preflight scientific validation failed")


__all__ = [
    "GATE_NAMES",
    "OUTPUT_NAME",
    "PROFILE_NAME",
    "TASK_NAME",
    "validate_repository_preflight_completion",
]
