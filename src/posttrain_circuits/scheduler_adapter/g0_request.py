"""Prepare a scientific-only fixed Qwen3-v2 two-GPU G0 request."""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.config_bindings import ConfigBinding, bind_config
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.git_provenance import (
    require_git_output,
    unsafe_untracked_paths,
)
from posttrain_circuits.artifacts.protocol_amendments import (
    AMENDMENT_ID,
    AMENDMENT_RELATIVE_PATH,
    load_protocol_amendment_bytes,
    resolve_accepted_protocol_amendment,
    validate_accepted_lineage_commit,
    validate_two_gpu_g0_config,
)
from posttrain_circuits.scheduler_adapter.config_resolver import ConfigBindingResolver
from posttrain_circuits.scheduler_adapter.content_store import ContentStore
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.outbox import prepare_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.plan_store import publish_workflow_plan
from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
    ARTIFACT_NAMESPACE,
    GPU_COUNT,
    GPU_PREFLIGHT_COMPLETION_CONTENT_NAME,
    GPU_PREFLIGHT_REPORT_CONTENT_NAME,
    MODEL_REVISION,
    OUTPUT_NAMES,
    PREREG_CONTENT_NAME,
    PROTOCOL_AMENDMENT_CONTENT_NAME,
    PROFILE_NAME,
    TASK_NAME,
    TEACHER_REVISION,
    TOKENIZER_FINGERPRINT,
    UNIT_ID,
    WORKFLOW_ID,
)
from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
    GATE_NAMES as PREFLIGHT_GATES,
    PROFILE_NAME as PREFLIGHT_PROFILE,
    _validate_report as validate_gpu_preflight_report,
)
from posttrain_circuits.scheduler_adapter.secure_files import read_regular_file_nofollow
from posttrain_circuits.workflows.contracts import ContentIdentity, WorkflowPlan, WorkflowUnit


PREREG_RELATIVE_PATH = Path("prereg/qwen3_v2.yaml")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
BASE_CONFIG_OVERRIDES = (
    "g0=qwen3_v2_eap_separation",
    "experiment=canonical_sft",
    "task.num_examples=256",
    "state_source.num_candidates=8",
)
PREFLIGHT_COMPLETION_FIELDS = frozenset(
    {
        "completed_at",
        "completion_kind",
        "execution",
        "execution_config_sha256",
        "input_hashes",
        "output_files",
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
)


@dataclass(frozen=True)
class GpuPreflightEvidence:
    report_raw: bytes
    report_file_sha256: str
    completion_raw: bytes
    completion_file_sha256: str
    git_commit: str


@dataclass(frozen=True)
class PreparedG0Request:
    workflow_id: str
    plan_sha256: str
    unit_id: str
    plan_path: Path
    outbox_path: Path

    def to_payload(self) -> dict[str, str]:
        return {
            "outbox_path": str(self.outbox_path),
            "plan_path": str(self.plan_path),
            "plan_sha256": self.plan_sha256,
            "unit_id": self.unit_id,
            "workflow_id": self.workflow_id,
        }


def _require_clean_checkout(code_root: Path) -> str:
    status = require_git_output(
        code_root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ),
    )
    if status:
        raise AdapterValidationError("G0 request preparation requires a clean checkout")
    try:
        unsafe = unsafe_untracked_paths(code_root)
    except (OSError, UnicodeError, ValueError) as error:
        raise AdapterValidationError(
            "G0 request preparation could not enumerate untracked files"
        ) from error
    if unsafe:
        raise AdapterValidationError(
            "G0 request preparation rejects ignored untracked files"
        )
    value = require_git_output(code_root, ("rev-parse", "HEAD"))
    if GIT_COMMIT.fullmatch(value) is None:
        raise AdapterValidationError("Git did not return one immutable commit identity")
    return value


def _strict_json(raw: bytes, *, context: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise AdapterValidationError(f"{context} contains non-finite number {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise AdapterValidationError(f"{context} contains duplicate key {key!r}")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterValidationError(f"{context} is not strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise AdapterValidationError(f"{context} must be a JSON object")
    return payload


def validate_gpu_preflight_evidence(
    *,
    report_path: Path,
    completion_path: Path,
    current_git_commit: str,
    code_root: Path,
    amendment: Any,
) -> GpuPreflightEvidence:
    """Require one successful preflight from the accepted implementation lineage."""

    report_raw, report_file_sha256 = read_regular_file_nofollow(
        report_path,
        context="published Qwen3-v2 GPU preflight report",
        max_bytes=32 * 1024 * 1024,
    )
    completion_raw, completion_file_sha256 = read_regular_file_nofollow(
        completion_path,
        context="published Qwen3-v2 GPU preflight completion",
        max_bytes=4 * 1024 * 1024,
    )
    report = _strict_json(report_raw, context="GPU preflight report")
    completion = _strict_json(completion_raw, context="GPU preflight completion")
    resolved_config_sha256 = report.get("resolved_config_sha256")
    preregistration_sha256 = report.get("prereg_sha256")
    if not isinstance(resolved_config_sha256, str) or not isinstance(preregistration_sha256, str):
        raise AdapterValidationError("GPU preflight report lacks config/preregistration bindings")
    validate_gpu_preflight_report(
        report,
        resolved_config_sha256=resolved_config_sha256,
        preregistration_sha256=preregistration_sha256,
    )
    if set(completion) != PREFLIGHT_COMPLETION_FIELDS:
        raise AdapterValidationError(
            "GPU preflight completion fields differ from the published contract"
        )
    completion_digest = completion.get("sha256")
    unsigned_completion = {
        key: value for key, value in completion.items() if key != "sha256"
    }
    if completion_digest != sha256_value(unsigned_completion):
        raise AdapterValidationError("GPU preflight completion self-summary is invalid")
    gates = completion.get("scientific_validation")
    execution = completion.get("execution")
    output_files = completion.get("output_files")
    input_hashes = completion.get("input_hashes")
    if (
        completion.get("schema_version") != 1
        or completion.get("completion_kind") != "scientific"
        or completion.get("project") != "OPD"
        or completion.get("task") != "qwen3_v2_gpu_preflight"
        or completion.get("workflow_id") != "qwen3-v2-gpu-preflight-v1"
        or completion.get("unit_id") != "gpu-preflight"
        or not isinstance(gates, dict)
        or tuple(sorted(gates)) != PREFLIGHT_GATES
        or any(value is not True for value in gates.values())
        or not isinstance(execution, dict)
        or execution.get("execution_profile") != PREFLIGHT_PROFILE
        or not isinstance(output_files, dict)
        or len(output_files) != 1
        or next(iter(output_files.values()), None) != report_file_sha256
        or not isinstance(input_hashes, dict)
        or input_hashes.get("resolved_config_sha256") != resolved_config_sha256
        or input_hashes.get("preregistration_sha256") != preregistration_sha256
        or completion.get("resolved_config_sha256") != resolved_config_sha256
        or completion.get("execution_config_sha256")
        != input_hashes.get("execution_config_sha256")
        or completion.get("scientific_config_sha256")
        != input_hashes.get("scientific_config_sha256")
    ):
        raise AdapterValidationError("GPU preflight completion is not a successful published result")
    preflight_commit = report.get("git_commit")
    if not isinstance(preflight_commit, str):
        raise AdapterValidationError("GPU preflight lacks a Git commit")
    try:
        validate_accepted_lineage_commit(
            code_root=code_root,
            candidate_commit=preflight_commit,
            current_binding=amendment,
            expected_head=current_git_commit,
            role="GPU preflight commit",
        )
    except ValueError as error:
        raise AdapterValidationError(
            f"G0 GPU preflight implementation lineage is invalid: {error}"
        ) from error
    return GpuPreflightEvidence(
        report_raw=report_raw,
        report_file_sha256=report_file_sha256,
        completion_raw=completion_raw,
        completion_file_sha256=completion_file_sha256,
        git_commit=preflight_commit,
    )


def fixed_resolved_config(
    *,
    code_commit: str,
    gpu_preflight_report_sha256: str,
    gpu_preflight_completion_sha256: str,
    gpu_preflight_git_commit: str,
    protocol_amendment_sha256: str,
    reviewed_implementation_commit: str,
) -> dict[str, Any]:
    """Return the complete reviewed G0 configuration plus prerequisite identities."""

    if not GIT_COMMIT.fullmatch(code_commit):
        raise AdapterValidationError("G0 code_commit is not a Git identity")
    if (
        len(protocol_amendment_sha256) != 64
        or any(character not in "0123456789abcdef" for character in protocol_amendment_sha256)
    ):
        raise AdapterValidationError("G0 protocol amendment is not a SHA-256 identity")
    if not GIT_COMMIT.fullmatch(reviewed_implementation_commit):
        raise AdapterValidationError("G0 reviewed implementation is not a Git identity")
    if not GIT_COMMIT.fullmatch(gpu_preflight_git_commit):
        raise AdapterValidationError("G0 GPU preflight is not a Git identity")
    if reviewed_implementation_commit in {code_commit, gpu_preflight_git_commit}:
        raise AdapterValidationError("G0 reviewed implementation binding is self-referential")
    from posttrain_circuits.core.config import compose_config

    config = compose_config(list(BASE_CONFIG_OVERRIDES))
    config["scheduler_g0"] = {
        "artifact_namespace": ARTIFACT_NAMESPACE,
        "request_git_commit": code_commit,
        "execution_profile": PROFILE_NAME,
        "gpu_preflight_git_commit": gpu_preflight_git_commit,
        "gpu_preflight_completion_sha256": gpu_preflight_completion_sha256,
        "gpu_preflight_report_sha256": gpu_preflight_report_sha256,
        "model_revision": MODEL_REVISION,
        "process_count": GPU_COUNT,
        "protocol_amendment_id": AMENDMENT_ID,
        "protocol_amendment_sha256": protocol_amendment_sha256,
        "reviewed_implementation_commit": reviewed_implementation_commit,
        "task": TASK_NAME,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
    }
    return config


def _get_path(payload: dict[str, Any], dotted: str) -> object:
    value: object = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise AdapterValidationError(f"fixed config lacks storage locator {dotted!r}")
        value = value[part]
    return value


def _set_path(payload: dict[str, Any], dotted: str, value: object) -> None:
    parts = dotted.split(".")
    parent: object = payload
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            raise AdapterValidationError(f"fixed config lacks storage locator {dotted!r}")
        parent = parent[part]
    if not isinstance(parent, dict) or parts[-1] not in parent:
        raise AdapterValidationError(f"fixed config lacks storage locator {dotted!r}")
    parent[parts[-1]] = value


def _delete_path(payload: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    parent: object = payload
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            raise AdapterValidationError(f"fixed config lacks storage locator {dotted!r}")
        parent = parent[part]
    if not isinstance(parent, dict) or parts[-1] not in parent:
        raise AdapterValidationError(f"fixed config lacks storage locator {dotted!r}")
    del parent[parts[-1]]


def _projections(
    config: dict[str, Any], binding: ConfigBinding
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding_payload = binding.as_dict()
    artifact_hashes = dict(binding.input_artifact_hashes)
    locators = dict(binding.storage_locators)
    scientific = copy.deepcopy(config)
    for dotted, locator in locators.items():
        if str(_get_path(config, dotted)) != locator:
            raise AdapterValidationError("ConfigBinding storage locator changed during projection")
        if dotted in artifact_hashes:
            _set_path(scientific, dotted, {"content_sha256": artifact_hashes[dotted]})
        else:
            _delete_path(scientific, dotted)
    scientific_projection = {
        "config": scientific,
        "input_artifact_hashes": artifact_hashes,
        "schema_version": 3,
    }
    execution_projection = {
        "execution_context": binding_payload["execution_context"],
        "schema_version": 3,
        "storage_locators": locators,
    }
    return scientific_projection, execution_projection


def build_qwen3_v2_g0_plan(
    *,
    layout: WorkflowLayout,
    gpu_preflight_report: Path,
    gpu_preflight_completion: Path,
) -> WorkflowPlan:
    """Materialize one accepted-lineage preflight-bound G0 plan in OPD file CAS."""

    layout.validate()
    code_commit = _require_clean_checkout(layout.code_root)
    amendment = resolve_accepted_protocol_amendment(
        code_root=layout.code_root,
        configured_path=str(AMENDMENT_RELATIVE_PATH),
        expected_head=code_commit,
    )
    evidence = validate_gpu_preflight_evidence(
        report_path=gpu_preflight_report,
        completion_path=gpu_preflight_completion,
        current_git_commit=code_commit,
        code_root=layout.code_root,
        amendment=amendment,
    )
    prereg_path = layout.code_root / PREREG_RELATIVE_PATH
    prereg_raw, prereg_sha256 = read_regular_file_nofollow(
        prereg_path,
        context="Qwen3-v2 preregistration",
        max_bytes=4 * 1024 * 1024,
    )
    amendment_raw, amendment_sha256 = read_regular_file_nofollow(
        amendment.path,
        context="accepted Qwen3-v2 two-GPU G0 amendment",
        max_bytes=4 * 1024 * 1024,
    )
    if amendment_sha256 != amendment.sha256:
        raise AdapterValidationError("accepted protocol amendment changed after Git review")
    amendment_payload = load_protocol_amendment_bytes(amendment_raw)
    if amendment_payload["review"]["status"] != "accepted":
        raise AdapterValidationError("G0 protocol amendment remains unaccepted")
    config = fixed_resolved_config(
        code_commit=code_commit,
        gpu_preflight_report_sha256=evidence.report_file_sha256,
        gpu_preflight_completion_sha256=evidence.completion_file_sha256,
        gpu_preflight_git_commit=evidence.git_commit,
        protocol_amendment_sha256=amendment.sha256,
        reviewed_implementation_commit=amendment.reviewed_implementation_commit,
    )
    validate_two_gpu_g0_config(config, amendment_payload)
    execution_context = {
        "distributed_process_count": GPU_COUNT,
        "execution_profile": PROFILE_NAME,
        "scheduler_protocol": 2,
    }
    input_artifact_hashes = {
        "gpu_preflight_completion": evidence.completion_file_sha256,
        "gpu_preflight_report": evidence.report_file_sha256,
        "prereg_path": prereg_sha256,
        "protocol_amendment_path": amendment.sha256,
    }
    binding = bind_config(
        config,
        input_artifact_hashes=input_artifact_hashes,
        execution_context=execution_context,
    )
    scientific_config, execution_config = _projections(config, binding)
    store = ContentStore(layout)
    config_identities = ConfigBindingResolver(store).materialize(
        binding,
        resolved_config=config,
        scientific_config=scientific_config,
        execution_config=execution_config,
    )
    file_inputs = (
        (PREREG_CONTENT_NAME, prereg_sha256, prereg_raw),
        (PROTOCOL_AMENDMENT_CONTENT_NAME, amendment.sha256, amendment_raw),
        (
            GPU_PREFLIGHT_REPORT_CONTENT_NAME,
            evidence.report_file_sha256,
            evidence.report_raw,
        ),
        (
            GPU_PREFLIGHT_COMPLETION_CONTENT_NAME,
            evidence.completion_file_sha256,
            evidence.completion_raw,
        ),
    )
    identities = list(config_identities.as_tuple())
    for name, digest, raw in file_inputs:
        store.publish_file(sha256=digest, raw=raw)
        identities.append(ContentIdentity(name=name, sha256=digest, kind="file"))
    return WorkflowPlan(
        workflow_id=WORKFLOW_ID,
        units=(
            WorkflowUnit(
                unit_id=UNIT_ID,
                task=TASK_NAME,
                content_inputs=tuple(sorted(identities)),
                dependencies=(),
                output_names=OUTPUT_NAMES,
            ),
        ),
    )


def prepare_qwen3_v2_g0_request(
    *,
    layout: WorkflowLayout,
    gpu_preflight_report: Path,
    gpu_preflight_completion: Path,
) -> PreparedG0Request:
    """Publish one immutable plan and fresh outbox request; never submit or poll."""

    plan = build_qwen3_v2_g0_plan(
        layout=layout,
        gpu_preflight_report=gpu_preflight_report,
        gpu_preflight_completion=gpu_preflight_completion,
    )
    plan_path = publish_workflow_plan(plan, layout=layout)
    outbox_path = prepare_outbox_request(
        plan,
        unit_id=UNIT_ID,
        layout=layout,
    )
    return PreparedG0Request(
        workflow_id=plan.workflow_id,
        plan_sha256=plan.sha256(),
        unit_id=UNIT_ID,
        plan_path=plan_path,
        outbox_path=outbox_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gpu-preflight-report", type=Path, required=True)
    parser.add_argument("--gpu-preflight-completion", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare_qwen3_v2_g0_request(
        layout=WorkflowLayout.production(),
        gpu_preflight_report=args.gpu_preflight_report,
        gpu_preflight_completion=args.gpu_preflight_completion,
    )
    print(json.dumps(receipt.to_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_CONFIG_OVERRIDES",
    "GpuPreflightEvidence",
    "PreparedG0Request",
    "build_qwen3_v2_g0_plan",
    "fixed_resolved_config",
    "prepare_qwen3_v2_g0_request",
    "validate_gpu_preflight_evidence",
]
