"""Prepare the fixed Qwen3-v2 GPU-preflight request without submitting it."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.config_bindings import ConfigBinding, bind_config
from posttrain_circuits.artifacts.git_provenance import require_git_output
from posttrain_circuits.scheduler_adapter.config_resolver import ConfigBindingResolver
from posttrain_circuits.scheduler_adapter.content_store import ContentStore
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.outbox import prepare_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.plan_store import publish_workflow_plan
from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
    ARTIFACT_NAMESPACE,
    CHAT_TEMPLATE_SHA256,
    GPU_COUNT,
    MODEL_REVISION,
    OUTPUT_NAME,
    PROFILE_NAME,
    PROTOCOL_TRACK,
    TASK_NAME,
    TEACHER_REVISION,
    TOKENIZER_FINGERPRINT,
    UNIT_ID,
    WORKFLOW_ID,
)
from posttrain_circuits.scheduler_adapter.secure_files import read_regular_file_nofollow
from posttrain_circuits.workflows.contracts import ContentIdentity, WorkflowPlan, WorkflowUnit


PREREG_CONTENT_NAME = "preregistration_sha256"
PREREG_RELATIVE_PATH = Path("prereg/qwen3_v2.yaml")
EXECUTION_CONTEXT = {
    "distributed_process_count": GPU_COUNT,
    "execution_profile": PROFILE_NAME,
    "scheduler_protocol": 2,
}
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _clean_git_commit(code_root: Path) -> str:
    status = require_git_output(
        code_root, ("status", "--porcelain", "--untracked-files=no")
    )
    if status:
        raise AdapterValidationError(
            "GPU-preflight request preparation requires a clean tracked checkout"
        )
    commit = require_git_output(code_root, ("rev-parse", "HEAD"))
    if GIT_COMMIT.fullmatch(commit) is None:
        raise AdapterValidationError("Git did not return one immutable commit identity")
    return commit


def fixed_resolved_config(*, code_commit: str) -> dict[str, Any]:
    """Return the complete fixed scientific input without YAML or Hydra dependencies."""

    if GIT_COMMIT.fullmatch(code_commit) is None:
        raise AdapterValidationError("GPU-preflight code_commit is not a Git identity")

    prompt_protocol = {
        "add_generation_prompt": True,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "enable_thinking": False,
        "messages": "single_user",
        "name": "qwen3_non_thinking_v1",
    }
    sampling_protocol = {
        "do_sample": True,
        "min_p": 0.0,
        "name": "qwen3_non_thinking_sampling_v1",
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    }
    return {
        "artifact_namespace": ARTIFACT_NAMESPACE,
        "code_commit": code_commit,
        "config_kind": TASK_NAME,
        "model": {
            "allow_unpinned_revision": False,
            "attn_implementation": "sdpa",
            "gradient_checkpointing": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "model_name_or_path": "Qwen/Qwen3-1.7B",
            "model_revision": MODEL_REVISION,
            "prompt_protocol": copy.deepcopy(prompt_protocol),
            "sampling_protocol": copy.deepcopy(sampling_protocol),
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "tokenizer_name_or_path": "Qwen/Qwen3-1.7B",
            "tokenizer_revision": MODEL_REVISION,
            "torch_dtype": "bfloat16",
            "trust_remote_code": False,
            "use_cache": False,
        },
        "prereg_path": str(PREREG_RELATIVE_PATH),
        "prereg_version": PROTOCOL_TRACK,
        "protocol_track": PROTOCOL_TRACK,
        "resource_budget": {
            "loading_strategy": "low_cpu_mem_student_rank_zero_teacher",
            "minimum_headroom_fraction": 0.20,
            "minimum_headroom_gib": 32,
            "node_memory_gib": 192,
        },
        "teacher": {
            "allow_unpinned_revision": False,
            "attn_implementation": "sdpa",
            "gradient_checkpointing": False,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "model_name_or_path": "Qwen/Qwen3-8B",
            "model_revision": TEACHER_REVISION,
            "prompt_protocol": copy.deepcopy(prompt_protocol),
            "rank_zero_only_training_load": True,
            "sampling_protocol": copy.deepcopy(sampling_protocol),
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "tokenizer_name_or_path": "Qwen/Qwen3-8B",
            "tokenizer_revision": TEACHER_REVISION,
            "torch_dtype": "bfloat16",
            "trust_remote_code": False,
            "use_cache": True,
        },
    }


@dataclass(frozen=True)
class PreparedGpuPreflightRequest:
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


def _validate_fixed_config(config: dict[str, Any]) -> None:
    try:
        model = config["model"]
        teacher = config["teacher"]
        budget = config["resource_budget"]
        prompt = model["prompt_protocol"]
    except (KeyError, TypeError) as error:
        raise AdapterValidationError("fixed GPU-preflight config is incomplete") from error
    if (
        config.get("protocol_track") != PROTOCOL_TRACK
        or GIT_COMMIT.fullmatch(str(config.get("code_commit", ""))) is None
        or config.get("artifact_namespace") != ARTIFACT_NAMESPACE
        or config.get("prereg_path") != str(PREREG_RELATIVE_PATH)
        or config.get("prereg_version") != PROTOCOL_TRACK
        or model.get("model_revision") != MODEL_REVISION
        or model.get("tokenizer_revision") != MODEL_REVISION
        or model.get("tokenizer_fingerprint") != TOKENIZER_FINGERPRINT
        or model.get("trust_remote_code") is not False
        or model.get("allow_unpinned_revision") is not False
        or model.get("local_files_only") is not True
        or model.get("low_cpu_mem_usage") is not True
        or prompt.get("name") != "qwen3_non_thinking_v1"
        or prompt.get("enable_thinking") is not False
        or prompt.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or teacher.get("model_revision") != TEACHER_REVISION
        or teacher.get("tokenizer_revision") != TEACHER_REVISION
        or teacher.get("tokenizer_fingerprint") != TOKENIZER_FINGERPRINT
        or teacher.get("trust_remote_code") is not False
        or teacher.get("allow_unpinned_revision") is not False
        or teacher.get("local_files_only") is not True
        or teacher.get("low_cpu_mem_usage") is not True
        or teacher.get("rank_zero_only_training_load") is not True
        or budget.get("node_memory_gib") != 192
        or budget.get("minimum_headroom_gib") != 32
        or budget.get("minimum_headroom_fraction") != 0.20
        or budget.get("loading_strategy") != "low_cpu_mem_student_rank_zero_teacher"
    ):
        raise AdapterValidationError("composed config differs from the fixed Qwen3-v2 preflight")


def _projections(
    config: dict[str, Any],
    binding: ConfigBinding,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce ConfigBinding's scientific/execution projections exactly."""

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


def build_qwen3_v2_gpu_preflight_plan(*, layout: WorkflowLayout) -> WorkflowPlan:
    """Materialize the fixed config projections and raw preregistration in file CAS."""

    layout.validate()
    config = fixed_resolved_config(code_commit=_clean_git_commit(layout.code_root))
    _validate_fixed_config(config)
    prereg_path = layout.code_root / PREREG_RELATIVE_PATH
    prereg_raw, prereg_sha256 = read_regular_file_nofollow(
        prereg_path,
        context="Qwen3-v2 preregistration",
        max_bytes=4 * 1024 * 1024,
    )
    binding = bind_config(
        config,
        input_artifact_hashes={"prereg_path": prereg_sha256},
        execution_context=EXECUTION_CONTEXT,
    )
    scientific_config, execution_config = _projections(config, binding)
    store = ContentStore(layout)
    config_identities = ConfigBindingResolver(store).materialize(
        binding,
        resolved_config=config,
        scientific_config=scientific_config,
        execution_config=execution_config,
    )
    store.publish_file(sha256=prereg_sha256, raw=prereg_raw)
    preregistration = ContentIdentity(
        name=PREREG_CONTENT_NAME,
        sha256=prereg_sha256,
        kind="file",
    )
    identities = tuple(sorted((*config_identities.as_tuple(), preregistration)))
    return WorkflowPlan(
        workflow_id=WORKFLOW_ID,
        units=(
            WorkflowUnit(
                unit_id=UNIT_ID,
                task=TASK_NAME,
                content_inputs=identities,
                dependencies=(),
                output_names=(OUTPUT_NAME,),
            ),
        ),
    )


def prepare_qwen3_v2_gpu_preflight_request(
    *, layout: WorkflowLayout
) -> PreparedGpuPreflightRequest:
    """Publish one immutable plan and fresh outbox request; never submit or poll."""

    plan = build_qwen3_v2_gpu_preflight_plan(layout=layout)
    plan_path = publish_workflow_plan(plan, layout=layout)
    outbox_path = prepare_outbox_request(
        plan,
        unit_id=UNIT_ID,
        execution_profile=PROFILE_NAME,
        layout=layout,
    )
    return PreparedGpuPreflightRequest(
        workflow_id=plan.workflow_id,
        plan_sha256=plan.sha256(),
        unit_id=UNIT_ID,
        plan_path=plan_path,
        outbox_path=outbox_path,
    )


def main() -> int:
    receipt = prepare_qwen3_v2_gpu_preflight_request(layout=WorkflowLayout.production())
    print(json.dumps(receipt.to_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PREREG_CONTENT_NAME",
    "PreparedGpuPreflightRequest",
    "build_qwen3_v2_gpu_preflight_plan",
    "fixed_resolved_config",
    "prepare_qwen3_v2_gpu_preflight_request",
]
