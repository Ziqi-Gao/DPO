"""Prepare a scientific-only, allocation-neutral Qwen3-v2 G0 request."""

from __future__ import annotations

import argparse
import copy
import json
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.config_bindings import ConfigBinding, bind_config
from posttrain_circuits.artifacts.git_provenance import (
    require_git_output,
)
from posttrain_circuits.artifacts.execution_science_protocol import (
    ResolvedExecutionScienceProtocol,
    canonical_science_config_sha256,
    resolve_accepted_execution_science_protocol,
    validate_count_neutral_identity,
)
from posttrain_circuits.artifacts.protocol_amendments import (
    load_execution_class_amendment_bytes,
    resolve_accepted_execution_class_amendment,
    validate_elastic_g0_config,
)
from posttrain_circuits.scheduler_adapter.config_resolver import ConfigBindingResolver
from posttrain_circuits.scheduler_adapter.content_store import ContentStore
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.artifacts.execution_safety_certification import (
    CERTIFICATION_CONTENT_NAME,
    CERTIFICATION_ID,
    CERTIFICATION_RELATIVE_PATH,
    DESCRIPTOR_CONTENT_NAME,
    DESCRIPTOR_RELATIVE_PATH,
    EXECUTION_CLASS_ID,
    SUCCESSOR_AMENDMENT_RELATIVE_PATH,
    ExecutionSafetyCertificationBinding,
    current_execution_safety_fingerprint,
    execution_safety_config_projection,
    load_execution_safety_certification_bytes,
    load_execution_safety_descriptor_bytes,
)
from posttrain_circuits.scheduler_adapter.outbox import prepare_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.plan_store import publish_workflow_plan
from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
    ALLOWED_GPU_COUNTS,
    ARTIFACT_NAMESPACE,
    BATCH_PARTITION_PROTOCOL,
    MODEL_REVISION,
    OUTPUT_NAMES,
    PREREG_CONTENT_NAME,
    PROTOCOL_AMENDMENT_ID,
    PROTOCOL_AMENDMENT_CONTENT_NAME,
    TASK_NAME,
    TEACHER_REVISION,
    TOKENIZER_FINGERPRINT,
    UNIT_ID,
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
CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH = (
    "prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_v1.yaml"
)
WORKFLOW_ID_PREFIX = "qwen3-v2-g0-elastic-"
WORKFLOW_ID = re.compile(r"qwen3-v2-g0-elastic-[0-9a-f]{32}\Z")


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
            "--untracked-files=no",
            "--ignore-submodules=none",
        ),
    )
    if status:
        raise AdapterValidationError("G0 request preparation requires a clean checkout")
    value = require_git_output(code_root, ("rev-parse", "HEAD"))
    if GIT_COMMIT.fullmatch(value) is None:
        raise AdapterValidationError("Git did not return one immutable commit identity")
    return value


def new_qwen3_v2_g0_workflow_id() -> str:
    """Return one fresh, opaque and allocation-neutral workflow identity."""

    value = f"{WORKFLOW_ID_PREFIX}{secrets.token_hex(16)}"
    if WORKFLOW_ID.fullmatch(value) is None:
        raise AdapterValidationError("generated G0 workflow identity is invalid")
    try:
        validate_count_neutral_identity(value, name="workflow_id")
    except ValueError as error:
        raise AdapterValidationError(str(error)) from error
    return value


def resolved_config_for_execution_science(
    *,
    code_root: Path,
    code_commit: str,
    execution_science: ResolvedExecutionScienceProtocol,
    execution_safety: ExecutionSafetyCertificationBinding,
    protocol_amendment_sha256: str,
    reviewed_implementation_commit: str,
) -> dict[str, Any]:
    """Compose reviewed experiment science under one reusable execution class."""

    if not GIT_COMMIT.fullmatch(code_commit):
        raise AdapterValidationError("G0 code_commit is not a Git identity")
    if (
        len(protocol_amendment_sha256) != 64
        or any(character not in "0123456789abcdef" for character in protocol_amendment_sha256)
    ):
        raise AdapterValidationError("G0 protocol amendment is not a SHA-256 identity")
    if not GIT_COMMIT.fullmatch(reviewed_implementation_commit):
        raise AdapterValidationError("G0 reviewed implementation is not a Git identity")
    if (
        execution_safety.execution_class_id != EXECUTION_CLASS_ID
        or tuple(execution_safety.supported_world_sizes) != ALLOWED_GPU_COUNTS
        or execution_safety.review_status != "accepted"
        or execution_safety.reviewed_implementation_commit
        != reviewed_implementation_commit
    ):
        raise AdapterValidationError(
            "G0 execution-safety certification is not the reviewed 1..4 class"
        )
    if reviewed_implementation_commit == code_commit:
        raise AdapterValidationError("G0 reviewed implementation binding is self-referential")
    from posttrain_circuits.core.config import compose_config

    if execution_science.binding.review_status != "accepted":
        raise AdapterValidationError("G0 execution science protocol is not accepted")
    try:
        config = compose_config(
            list(execution_science.binding.hydra_override_vector),
            config_root=code_root / "configs",
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise AdapterValidationError(
            f"G0 execution science configuration cannot be composed: {error}"
        ) from error
    if config.get("seed") != execution_science.binding.seed:
        raise AdapterValidationError(
            "G0 composed seed differs from the accepted science protocol"
        )
    config["protocol_amendment_path"] = str(SUCCESSOR_AMENDMENT_RELATIVE_PATH)
    config["scheduler_g0"] = {
        "allocation_contract": "manifest_driven_scheduler_gpu_v1",
        "artifact_namespace": ARTIFACT_NAMESPACE,
        "batch_partition_protocol": BATCH_PARTITION_PROTOCOL,
        "execution_safety_certification_sha256": execution_safety.certification_sha256,
        "execution_safety_class_id": execution_safety.execution_class_id,
        "execution_safety_descriptor_sha256": execution_safety.descriptor_sha256,
        "execution_safety_fingerprint_sha256": execution_safety.fingerprint_sha256,
        "execution_safety_supported_world_sizes": list(
            execution_safety.supported_world_sizes
        ),
        "execution_science_protocol_git_commit": (
            execution_science.acceptance_commit
        ),
        "execution_science_protocol_id": execution_science.binding.protocol_id,
        "execution_science_protocol_path": execution_science.relative_path,
        "execution_science_protocol_reviewed_implementation_commit": (
            execution_science.reviewed_implementation_commit
        ),
        "execution_science_protocol_sha256": execution_science.sha256,
        "model_revision": MODEL_REVISION,
        "protocol_amendment_id": PROTOCOL_AMENDMENT_ID,
        "protocol_amendment_sha256": protocol_amendment_sha256,
        "request_git_commit": code_commit,
        "reviewed_implementation_commit": reviewed_implementation_commit,
        "task": TASK_NAME,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
    }
    if (
        canonical_science_config_sha256(config)
        != execution_science.binding.storage_neutral_resolved_config_sha256
    ):
        raise AdapterValidationError(
            "G0 composed config differs from the accepted science protocol"
        )
    return config


def fixed_resolved_config(
    *,
    code_root: Path,
    code_commit: str,
    execution_science: ResolvedExecutionScienceProtocol,
    execution_safety: ExecutionSafetyCertificationBinding,
    protocol_amendment_sha256: str,
    reviewed_implementation_commit: str,
) -> dict[str, Any]:
    """Compatibility alias for the reviewed Candidate E configuration path."""

    if execution_science.relative_path != CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH:
        raise AdapterValidationError(
            "Candidate E convenience builder requires its exact science protocol"
        )
    return resolved_config_for_execution_science(
        code_root=code_root,
        code_commit=code_commit,
        execution_science=execution_science,
        execution_safety=execution_safety,
        protocol_amendment_sha256=protocol_amendment_sha256,
        reviewed_implementation_commit=reviewed_implementation_commit,
    )


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


def build_execution_science_g0_plan(
    *,
    layout: WorkflowLayout,
    execution_science_protocol_path: str,
    require_candidate_e_science: bool = False,
) -> WorkflowPlan:
    """Materialize accepted experiment science under the certified GPU class."""

    layout.validate()
    code_commit = _require_clean_checkout(layout.code_root)
    try:
        execution_science = resolve_accepted_execution_science_protocol(
            code_root=layout.code_root,
            configured_path=execution_science_protocol_path,
            expected_head=code_commit,
        )
        amendment = resolve_accepted_execution_class_amendment(
            code_root=layout.code_root,
            configured_path=str(SUCCESSOR_AMENDMENT_RELATIVE_PATH),
            expected_head=code_commit,
        )
    except ValueError as error:
        raise AdapterValidationError(
            f"G0 execution-class or scientific lineage is invalid: {error}"
        ) from error
    if execution_science.binding.unit_id != UNIT_ID:
        raise AdapterValidationError(
            "G0 science protocol names a different workflow unit"
        )
    if require_candidate_e_science and (
        execution_science.acceptance_commit != amendment.git_commit
        or execution_science.reviewed_implementation_commit
        != amendment.reviewed_implementation_commit
    ):
        raise AdapterValidationError(
            "Candidate E science and execution class lack one joint acceptance"
        )
    science_execution_class = execution_science.binding.payload["execution_class"]
    descriptor_path = layout.code_root / science_execution_class["descriptor_path"]
    certification_path = layout.code_root / science_execution_class["certification_path"]
    descriptor_raw, descriptor_sha256 = read_regular_file_nofollow(
        descriptor_path,
        context="Qwen3-v2 elastic execution-safety descriptor",
        max_bytes=4 * 1024 * 1024,
    )
    certification_raw, certification_sha256 = read_regular_file_nofollow(
        certification_path,
        context="Qwen3-v2 elastic execution-safety certification",
        max_bytes=4 * 1024 * 1024,
    )
    try:
        descriptor_payload = load_execution_safety_descriptor_bytes(
            descriptor_raw,
            code_root=layout.code_root,
        )
        execution_safety = load_execution_safety_certification_bytes(
            certification_raw,
            descriptor_raw,
            require_accepted=True,
            code_root=layout.code_root,
        )
    except ValueError as error:
        raise AdapterValidationError(
            f"G0 execution-safety certification is invalid: {error}"
        ) from error
    if (
        execution_safety.descriptor_sha256 != descriptor_sha256
        or execution_safety.certification_sha256 != certification_sha256
        or execution_safety.fingerprint_sha256
        != current_execution_safety_fingerprint(layout.code_root)
    ):
        raise AdapterValidationError(
            "G0 execution-safety artifact bytes differ from their certification"
        )
    expected_science_execution_class = {
        "execution_class_id": execution_safety.execution_class_id,
        "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
        "descriptor_sha256": execution_safety.descriptor_sha256,
        "fingerprint_sha256": execution_safety.fingerprint_sha256,
        "certification_id": execution_safety.certification_id,
        "certification_path": str(CERTIFICATION_RELATIVE_PATH),
        "certification_core_sha256": execution_safety.certification_core_sha256,
        "supported_world_sizes": list(execution_safety.supported_world_sizes),
    }
    if science_execution_class != expected_science_execution_class:
        raise AdapterValidationError(
            "G0 science protocol references a different execution-safety class"
        )
    prereg_path = layout.code_root / PREREG_RELATIVE_PATH
    prereg_raw, prereg_sha256 = read_regular_file_nofollow(
        prereg_path,
        context="Qwen3-v2 preregistration",
        max_bytes=4 * 1024 * 1024,
    )
    amendment_raw, amendment_sha256 = read_regular_file_nofollow(
        amendment.path,
        context="accepted Qwen3-v2 scheduler-managed G0 amendment",
        max_bytes=4 * 1024 * 1024,
    )
    if amendment_sha256 != amendment.sha256:
        raise AdapterValidationError("accepted protocol amendment changed after Git review")
    amendment_payload = load_execution_class_amendment_bytes(amendment_raw)
    expected_certification_terms = {
        "execution_class_id": execution_safety.execution_class_id,
        "certification_id": CERTIFICATION_ID,
        "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
        "descriptor_sha256": execution_safety.descriptor_sha256,
        "certification_path": str(CERTIFICATION_RELATIVE_PATH),
        "certification_core_sha256": execution_safety.certification_core_sha256,
        "fingerprint_sha256": execution_safety.fingerprint_sha256,
        "supported_world_sizes": list(execution_safety.supported_world_sizes),
    }
    if (
        amendment_payload["review"]["status"] != "accepted"
        or amendment_payload.get("execution_safety_certification")
        != expected_certification_terms
        or amendment.certification_binding != execution_safety
        or execution_safety.reviewed_implementation_commit
        != amendment.reviewed_implementation_commit
    ):
        raise AdapterValidationError(
            "G0 protocol amendment does not accept the exact execution certification"
        )
    config_builder = (
        fixed_resolved_config
        if require_candidate_e_science
        else resolved_config_for_execution_science
    )
    config = config_builder(
        code_root=layout.code_root,
        code_commit=code_commit,
        execution_science=execution_science,
        execution_safety=execution_safety,
        protocol_amendment_sha256=amendment.sha256,
        reviewed_implementation_commit=amendment.reviewed_implementation_commit,
    )
    if execution_safety_config_projection(config) != descriptor_payload["subject"][
        "resolved_config_safety_projection"
    ]:
        raise AdapterValidationError(
            "G0 config differs from the certified execution-safety projection"
        )
    if require_candidate_e_science:
        validate_elastic_g0_config(config, amendment_payload)
    execution_context = {
        "allocation_contract": "manifest_driven_scheduler_gpu_v1",
        "scheduler_protocol": 2,
    }
    input_artifact_hashes = {
        "execution_safety_certification": certification_sha256,
        "execution_safety_descriptor": descriptor_sha256,
        "execution_science_protocol_path": execution_science.sha256,
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
        (CERTIFICATION_CONTENT_NAME, certification_sha256, certification_raw),
        (DESCRIPTOR_CONTENT_NAME, descriptor_sha256, descriptor_raw),
        (
            "execution_science_protocol_sha256",
            execution_science.sha256,
            execution_science.raw,
        ),
        (PREREG_CONTENT_NAME, prereg_sha256, prereg_raw),
        (PROTOCOL_AMENDMENT_CONTENT_NAME, amendment.sha256, amendment_raw),
    )
    identities = list(config_identities.as_tuple())
    for name, digest, raw in file_inputs:
        store.publish_file(sha256=digest, raw=raw)
        identities.append(ContentIdentity(name=name, sha256=digest, kind="file"))
    return WorkflowPlan(
        workflow_id=new_qwen3_v2_g0_workflow_id(),
        units=(
            WorkflowUnit(
                unit_id=execution_science.binding.unit_id,
                task=TASK_NAME,
                content_inputs=tuple(sorted(identities)),
                dependencies=(),
                output_names=OUTPUT_NAMES,
            ),
        ),
    )


def build_qwen3_v2_g0_plan(*, layout: WorkflowLayout) -> WorkflowPlan:
    """Candidate E seed-42 convenience wrapper around the generic class plan."""

    return build_execution_science_g0_plan(
        layout=layout,
        execution_science_protocol_path=CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
        require_candidate_e_science=True,
    )


def prepare_execution_science_g0_request(
    *,
    layout: WorkflowLayout,
    execution_science_protocol_path: str,
    require_candidate_e_science: bool = False,
) -> PreparedG0Request:
    """Publish one immutable plan and fresh outbox request; never submit or poll."""

    plan = build_execution_science_g0_plan(
        layout=layout,
        execution_science_protocol_path=execution_science_protocol_path,
        require_candidate_e_science=require_candidate_e_science,
    )
    plan_path = publish_workflow_plan(plan, layout=layout)
    outbox_path = prepare_outbox_request(
        plan,
        unit_id=plan.units[0].unit_id,
        layout=layout,
    )
    return PreparedG0Request(
        workflow_id=plan.workflow_id,
        plan_sha256=plan.sha256(),
        unit_id=plan.units[0].unit_id,
        plan_path=plan_path,
        outbox_path=outbox_path,
    )


def prepare_qwen3_v2_g0_request(*, layout: WorkflowLayout) -> PreparedG0Request:
    """Prepare Candidate E seed-42 through its accepted science protocol."""

    return prepare_execution_science_g0_request(
        layout=layout,
        execution_science_protocol_path=CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
        require_candidate_e_science=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--execution-science-protocol",
        default=CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
    )
    args = parser.parse_args()
    is_candidate_e = (
        args.execution_science_protocol
        == CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH
    )
    receipt = prepare_execution_science_g0_request(
        layout=WorkflowLayout.production(),
        execution_science_protocol_path=args.execution_science_protocol,
        require_candidate_e_science=is_candidate_e,
    )
    print(json.dumps(receipt.to_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_CONFIG_OVERRIDES",
    "CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH",
    "PreparedG0Request",
    "build_execution_science_g0_plan",
    "build_qwen3_v2_g0_plan",
    "fixed_resolved_config",
    "new_qwen3_v2_g0_workflow_id",
    "prepare_execution_science_g0_request",
    "prepare_qwen3_v2_g0_request",
    "resolved_config_for_execution_science",
]
