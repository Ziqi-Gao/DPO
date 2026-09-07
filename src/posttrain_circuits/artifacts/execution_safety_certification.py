"""Reusable artifact certification for the Qwen3-v2 execution class.

The descriptor fingerprints execution-safety facts, not an experiment identity.
The certificate carries reviewed evidence for that descriptor.  A new job may
reuse the certificate only when the descriptor recomputed from the checkout is
byte-for-byte and fingerprint-equivalent to the certified descriptor.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow
from posttrain_circuits.learning.training.execution_safety_kernel import (
    EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION,
    execution_safety_config_projection as _kernel_execution_safety_config_projection,
)


EXECUTION_CLASS_ID = "qwen3-v2-elastic-training-v1"
DESCRIPTOR_RELATIVE_PATH = Path(
    "prereg/execution_safety/qwen3_v2_elastic_training_v1.descriptor.json"
)
CERTIFICATION_RELATIVE_PATH = Path(
    "prereg/execution_safety/qwen3_v2_elastic_training_v1.certification.yaml"
)
SUCCESSOR_AMENDMENT_RELATIVE_PATH = Path(
    "prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml"
)
DESCRIPTOR_CONTENT_NAME = "execution_safety_descriptor_sha256"
CERTIFICATION_CONTENT_NAME = "execution_safety_certification_sha256"
SUPPORTED_WORLD_SIZES = (1, 2, 3, 4)
DESCRIPTOR_KIND = "qwen3_v2_execution_safety_descriptor"
CERTIFICATION_KIND = "qwen3_v2_execution_safety_certification"
FINGERPRINT_SCHEMA = "opd-execution-safety-fingerprint-v1"
CERTIFICATION_ID = "qwen3-v2-elastic-training-v1-initial-certification"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

# These are the reviewed implementation/runtime surfaces whose byte identity
# can change distributed behavior or the verification of that behavior.  Mixed
# scientific configuration files are deliberately represented by the canonical
# safety projection below instead of whole-file hashes, so seed/path/scientific
# threshold changes do not accidentally become topology identity.
IMPLEMENTATION_FILE_PATHS = (
    "configs/accelerate/fsdp_server_scheduler.yaml",
    "deployments/qwen3_v2_g0/dependency-lock.json",
    "deployments/qwen3_v2_g0/package-manifest.json",
    "deployments/qwen3_v2_gpu_preflight/dependency-lock.json",
    "deployments/qwen3_v2_gpu_preflight/package-manifest.json",
    "scripts/server_scheduler/opd-entrypoint",
    "scripts/server_scheduler/qwen3-v2-g0-handler.py",
    "scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py",
    "src/posttrain_circuits/artifacts/checkpoints.py",
    "src/posttrain_circuits/artifacts/execution_safety_certification.py",
    "src/posttrain_circuits/artifacts/execution_safe_io.py",
    "src/posttrain_circuits/artifacts/execution_science_protocol.py",
    "src/posttrain_circuits/artifacts/hashing.py",
    "src/posttrain_circuits/artifacts/protocol_amendments.py",
    "src/posttrain_circuits/cli/compare_distributed_resume.py",
    "src/posttrain_circuits/cli/factorial_run_validation.py",
    "src/posttrain_circuits/cli/finalize_g0.py",
    "src/posttrain_circuits/cli/finalize_pilot_training.py",
    "src/posttrain_circuits/cli/train.py",
    "src/posttrain_circuits/core/config.py",
    "src/posttrain_circuits/core/seeding.py",
    "src/posttrain_circuits/datasets/teacher_demos/contracts.py",
    "src/posttrain_circuits/datasets/teacher_demos/ledger.py",
    "src/posttrain_circuits/datasets/teacher_demos/store.py",
    "src/posttrain_circuits/datasets/teacher_demos/views.py",
    "src/posttrain_circuits/datasets/trajectories/contracts.py",
    "src/posttrain_circuits/learning/collation.py",
    "src/posttrain_circuits/learning/contracts.py",
    "src/posttrain_circuits/learning/primitives.py",
    "src/posttrain_circuits/learning/supervision/losses.py",
    "src/posttrain_circuits/learning/supervision/verified_replay.py",
    "src/posttrain_circuits/learning/teacher/demo_source.py",
    "src/posttrain_circuits/learning/training/canonical_sft.py",
    "src/posttrain_circuits/learning/training/evaluation.py",
    "src/posttrain_circuits/learning/training/execution_safety_kernel.py",
    "src/posttrain_circuits/learning/training/factorial_trainer.py",
    "src/posttrain_circuits/learning/training/factories.py",
    "src/posttrain_circuits/learning/training/fsdp_contract.py",
    "src/posttrain_circuits/learning/training/optimizer.py",
    "src/posttrain_circuits/learning/training/schedules.py",
    "src/posttrain_circuits/learning/training/token_budget.py",
    "src/posttrain_circuits/models/loading.py",
    "src/posttrain_circuits/models/prompt_protocol.py",
    "src/posttrain_circuits/scheduler_adapter/dispatch.py",
    "src/posttrain_circuits/scheduler_adapter/entrypoint.py",
    "src/posttrain_circuits/scheduler_adapter/environment.py",
    "src/posttrain_circuits/scheduler_adapter/manifest.py",
    "src/posttrain_circuits/scheduler_adapter/qwen3_v2_g0.py",
    "src/posttrain_circuits/scheduler_adapter/registry.py",
    "src/posttrain_circuits/scheduler_adapter/runtime.py",
    "src/posttrain_circuits/scheduler_adapter/secure_files.py",
    "src/posttrain_circuits/scheduler_adapter/strict_json.py",
)
LEGACY_PREFLIGHT_EVIDENCE_SURFACE = {
    "observed_preflight_handler_sha256": (
        "e62d370db975302e3f8321ea93d5e281894c3eb09ef814a2b615b7cf7605948c"
    ),
    "observed_runtime_dependency_lock_sha256": (
        "c72a990c03b9d841d399d9e0497912fcfd6c78ce3e840406ecd98c921c2b893f"
    ),
    "observed_runtime_package_manifest_sha256": (
        "f17dc62af30eb7f96b94322b61b8ea7496a6fd0a1c3c31bb99e872dbd1466663"
    ),
    "observed_runtime_executable_sha256": (
        "848c64ae0635d363f8bbfc768f94a3be497c0d51acd28cd5087e6e8a13c44801"
    ),
    "observed_training_contract_sha256": (
        "6ae06f330aa99ca1aad75151c2f95a9d639b8644a68a402cce76117b78793e86"
    ),
}
SUCCESSOR_PREFLIGHT_DELTA_ATTESTATION = (
    "independent_review_of_lineage_bootstrap_delta_required;legacy_pilots_are_not_"
    "direct_observations_of_the_successor_handler"
)

NON_INVALIDATING_DIMENSIONS = (
    "job_id",
    "workflow_id",
    "plan_sha256",
    "seed",
    "replication_id",
    "output_directory",
    "scientific_parameters_not_affecting_distributed_execution",
)
INVALIDATING_DIMENSIONS = (
    "handler_or_execution_implementation",
    "fixed_runtime_or_dependency_lock",
    "model_or_tensor_shape",
    "maximum_sequence_length",
    "batch_partition_or_loss_scaling",
    "token_accounting",
    "fsdp_or_checkpoint_resume",
    "memory_envelope",
    "supported_world_sizes",
)
PROPOSED_REVIEW = {
    "status": "proposed",
    "reviewed_implementation_commit": None,
    "reviewer": None,
    "reviewed_at_utc": None,
    "rationale": None,
}


class ExecutionSafetyCertificationError(ValueError):
    """An execution-safety descriptor or certification is invalid."""


@dataclass(frozen=True)
class ExecutionSafetyDescriptorBinding:
    payload: dict[str, Any]
    execution_class_id: str
    fingerprint_sha256: str
    descriptor_sha256: str
    supported_world_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ExecutionSafetyCertificationBinding:
    execution_class_id: str
    certification_id: str
    fingerprint_sha256: str
    descriptor_sha256: str
    certification_sha256: str
    certification_core_sha256: str
    supported_world_sizes: tuple[int, ...]
    review_status: str
    reviewed_implementation_commit: str | None
    evidence_basis: Mapping[str, Any]


def _strict_json(raw: bytes, *, context: str) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExecutionSafetyCertificationError(
                    f"{context} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExecutionSafetyCertificationError(
                    f"{context} contains non-finite value {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutionSafetyCertificationError(
            f"{context} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutionSafetyCertificationError(f"{context} must be a JSON mapping")
    return payload


def _strict_yaml(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ExecutionSafetyCertificationError(
            f"{context} requires the fixed runtime YAML parser"
        ) from error

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: Any,
        node: Any,
        deep: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ExecutionSafetyCertificationError(
                    "certification keys must be strings"
                )
            if key in result:
                raise ExecutionSafetyCertificationError(
                    f"certification contains duplicate key {key!r}"
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        payload = yaml.load(
            raw.decode("utf-8", errors="strict"), Loader=UniqueKeyLoader
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ExecutionSafetyCertificationError(
            f"{context} is not strict UTF-8 YAML"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutionSafetyCertificationError(f"{context} must be a YAML mapping")
    return payload


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ExecutionSafetyCertificationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _regular_file_bytes(path: Path, *, role: str) -> bytes:
    try:
        return read_regular_bytes_nofollow(path, context=role)
    except ValueError as error:
        raise ExecutionSafetyCertificationError(str(error)) from error


def _certified_successor_preflight_surface(
    descriptor: ExecutionSafetyDescriptorBinding,
) -> dict[str, Any]:
    subject = descriptor.payload["subject"]
    implementation = subject["implementation_files"]
    runtime = subject["fixed_runtime"]["gpu_preflight"]
    return {
        "certified_preflight_handler_sha256": implementation[
            "scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py"
        ],
        "certified_runtime_dependency_lock_sha256": implementation[
            "deployments/qwen3_v2_gpu_preflight/dependency-lock.json"
        ],
        "certified_runtime_executable_sha256": runtime[
            "runtime_executable_sha256"
        ],
        "certified_training_contract_sha256": runtime[
            "training_contract_sha256"
        ],
        "successor_delta_attestation": SUCCESSOR_PREFLIGHT_DELTA_ATTESTATION,
        "successor_handler_directly_observed_by_legacy_pilot": False,
    }


def execution_safety_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project a config using the fingerprinted runtime safety kernel."""

    try:
        return _kernel_execution_safety_config_projection(config)
    except ValueError as error:
        raise ExecutionSafetyCertificationError(str(error)) from error


def composed_execution_safety_config_projection(code_root: Path) -> dict[str, Any]:
    """Compose the production-shaped config and return only its safety identity."""

    from posttrain_circuits.core.config import compose_config

    config = compose_config(
        [
            "g0=qwen3_v2_eap_separation",
            "experiment=canonical_sft",
            "task.num_examples=256",
            "state_source.num_candidates=8",
        ],
        config_root=code_root.resolve() / "configs",
    )
    return execution_safety_config_projection(config)


def _static_execution_safety_subject() -> dict[str, Any]:
    return {
        "batch_partition": {
            "global_logical_batch_size": 64,
            "max_per_rank_microbatch_size": 4,
            "optimizer_microsteps_by_world_size": {
                "1": 16,
                "2": 8,
                "3": 6,
                "4": 4,
            },
            "microbatch_sizes_by_rank_by_world_size": {
                "1": [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
                "2": [
                    [4, 4, 4, 4, 4, 4, 4, 4],
                    [4, 4, 4, 4, 4, 4, 4, 4],
                ],
                "3": [
                    [4, 4, 4, 4, 4, 2],
                    [4, 4, 4, 4, 4, 1],
                    [4, 4, 4, 4, 4, 1],
                ],
                "4": [
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                ],
            },
            "per_rank_samples_by_world_size": {
                "1": [64],
                "2": [32, 32],
                "3": [22, 21, 21],
                "4": [16, 16, 16, 16],
            },
            "protocol": "allocation_neutral_exact_global_batch_v1",
            "sample_order": "one_global_64_slot_window_independent_of_world_size",
        },
        "checkpoint_resume": {
            "changed_world_resume": "rejected_before_state_load",
            "checkpoint_boundary": "completed_optimizer_updates_only",
            "full_state_export_by_world_size": {
                "1": {"offload_to_cpu": False, "rank0_only": False},
                "2": {"offload_to_cpu": True, "rank0_only": True},
                "3": {"offload_to_cpu": True, "rank0_only": True},
                "4": {"offload_to_cpu": True, "rank0_only": True},
            },
            "global_token_budget_state": "identical_across_all_ranks",
            "rank_local_trainer_state": "checkpointed_and_restored_per_rank",
            "retry_policy": "isolated_attempt_workspace_restart_from_frozen_inputs",
            "same_world_resume": "required_and_deterministically_compared_twice",
            "state_source_cursor_protocol": (
                "teacher-demo-round-robin-v2-accepted-view"
            ),
            "state_source_rng": "none",
        },
        "cpu_threading": {
            "allocated_cpu_cores": 24,
            "partition": "allocated_cpu_cores_divided_equally_by_actual_world_size",
            "threads_per_rank_by_world_size": {
                "1": 24,
                "2": 12,
                "3": 8,
                "4": 6,
            },
        },
        "fixed_runtime": {
            "g0": {
                "dependency_lock_path": (
                    "deployments/qwen3_v2_g0/dependency-lock.json"
                ),
                "runtime_executable_sha256": (
                    "848c64ae0635d363f8bbfc768f94a3be497c0d51acd28cd5087e6e8a13c44801"
                ),
                "runtime_executable": (
                    "/scr/del6500/OPD/envs/qwen3-v2-g0-v1/bin/python"
                ),
            },
            "gpu_preflight": {
                "dependency_lock_path": (
                    "deployments/qwen3_v2_gpu_preflight/dependency-lock.json"
                ),
                "runtime_executable_sha256": (
                    "848c64ae0635d363f8bbfc768f94a3be497c0d51acd28cd5087e6e8a13c44801"
                ),
                "runtime_executable": (
                    "/scr/del6500/OPD/envs/qwen3-v2-gpu-preflight-v1/bin/python"
                ),
                "training_contract_sha256": (
                    "6ae06f330aa99ca1aad75151c2f95a9d639b8644a68a402cce76117b78793e86"
                ),
            },
        },
        "distributed_runtime": {
            "control_plane_backend": "gloo",
            "cuda_tensor_backend": "nccl_explicit_process_group",
            "first_collective": "scalar_all_reduce",
            "first_collective_timeout_seconds": 120,
            "nccl_p2p_disable": "1",
            "nccl_version": "2.27.3",
            "scheduler_uuid_order": "preserved",
        },
        "fsdp": {
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "effective_sharding_strategy_by_world_size": {
                "1": "NO_SHARD",
                "2": "FULL_SHARD",
                "3": "FULL_SHARD",
                "4": "FULL_SHARD",
            },
            "expected_transformer_block_count": 28,
            "expected_wrapper_count": 29,
            "requested_sharding_strategy": "FULL_SHARD",
            "root_wrapper_count": 1,
            "state_dict_type": "FULL_STATE_DICT",
            "transformer_layer": "Qwen3DecoderLayer",
            "use_orig_params": False,
            "wrapper_tree": (
                "one_root_plus_every_transformer_block_directly_wrapped"
            ),
        },
        "implementation_files": {},
        "loss_scaling": {
            "objective": "response_mask_sequence_mean_cross_entropy_v1",
            "protocol": (
                "exact_global_sequence_mean_after_fsdp_rank_averaging_v1"
            ),
            "sequence_normalization": True,
        },
        "memory_envelope": {
            "gpu_compute_capability": [12, 0],
            "gpu_memory_budget_mib_per_device": 81920,
            "gpu_minimum_total_memory_mib": 97000,
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "host_memory_mib": 196608,
            "minimum_host_headroom_fraction": 0.2,
            "minimum_host_headroom_gib": 32,
            "dataset_family_loaded_rows": 144000,
            "dataset_family_split_rows": {
                "circuit_discovery": 2000,
                "circuit_validation": 2000,
                "iid_test": 10000,
                "ood_depth_test": 10000,
                "ood_structure_test": 10000,
                "train": 100000,
                "validation": 10000,
            },
            "teacher_demo_max_accepted_records": 2048,
            "teacher_demo_max_candidate_records": 2048,
            "teacher_demo_max_candidates_per_prompt": 8,
            "teacher_demo_max_prompt_count": 256,
        },
        "optimizer_and_precision": {
            "full_parameter_training": True,
            "optimizer_class": "torch.optim.AdamW",
            "optimizer_state_shape": "two_moment_tensors_per_trainable_parameter",
            "student_parameter_dtype": "bfloat16",
            "gradient_accumulation_precision": "framework_managed",
            "learning_rate_is_scientific_not_topology_identity": True,
        },
        "model_and_sequence_shape": {
            "chat_template_sha256": (
                "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
            ),
            "derived_max_model_input_tokens": 1502,
            "evaluation_all_ranks": True,
            "evaluation_max_completion_tokens": 128,
            "evaluation_max_total_tokens": 1536,
            "max_model_input_length": 1536,
            "student_attention_implementation": "sdpa",
            "student_gradient_checkpointing": True,
            "model_id": "Qwen/Qwen3-1.7B",
            "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "student_torch_dtype": "bfloat16",
            "teacher_id": "Qwen/Qwen3-8B",
            "teacher_attention_implementation": "sdpa",
            "teacher_gradient_checkpointing": False,
            "teacher_max_new_tokens": 256,
            "teacher_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "teacher_torch_dtype": "bfloat16",
            "tokenizer_fingerprint": (
                "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
            ),
            "tokenizer_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        },
        "parameter_partition": {
            "invalidating_dimensions": list(INVALIDATING_DIMENSIONS),
            "non_invalidating_dimensions": list(NON_INVALIDATING_DIMENSIONS),
        },
        "resolved_config_safety_projection": copy.deepcopy(
            EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION
        ),
        "scheduler_boundary": {
            "actual_allocation_source": (
                "running_manifest_and_SERVER_SCHEDULER_GPU_COUNT"
            ),
            "cuda_visible_devices": "preserve_exactly",
            "gpu_count_policy": "scheduler",
            "logical_device_ordinals": "0..N-1",
            "physical_gpu_selection": "server_scheduler_only",
            "request_execution_profile": "omitted",
            "request_gpu_count_and_identity": "omitted",
            "request_resources": "omitted",
            "running_attempt_resize": "forbidden",
        },
        "supported_world_sizes": list(SUPPORTED_WORLD_SIZES),
        "token_accounting": {
            "budget": 2_000_000,
            "budget_unit": "global_nonpadding_model_input_tokens_processed",
            "max_optimizer_steps": 120,
            "reservation": (
                "exact_cross_rank_sum_before_any_backward_in_optimizer_window"
            ),
            "stop_boundary": (
                "before_optimizer_update_that_would_exceed_budget"
            ),
        },
    }


def build_execution_safety_subject(code_root: Path) -> dict[str, Any]:
    """Build the exact checkout-bound safety subject used for fingerprinting."""

    root = code_root.resolve()
    subject = _static_execution_safety_subject()
    subject["resolved_config_safety_projection"] = (
        composed_execution_safety_config_projection(root)
    )
    implementation_files: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILE_PATHS:
        raw = _regular_file_bytes(
            root / relative, role=f"execution-safety implementation {relative}"
        )
        implementation_files[relative] = _sha256_bytes(raw)
    subject["implementation_files"] = implementation_files
    return subject


def execution_safety_fingerprint(code_root: Path) -> str:
    """Return the safety fingerprint for the current checkout."""

    return sha256_value(build_execution_safety_subject(code_root))


def build_execution_safety_descriptor(code_root: Path) -> dict[str, Any]:
    subject = build_execution_safety_subject(code_root)
    return {
        "execution_class_id": EXECUTION_CLASS_ID,
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "fingerprint_sha256": sha256_value(subject),
        "kind": DESCRIPTOR_KIND,
        "schema_version": 1,
        "subject": subject,
    }


def _validate_descriptor_payload(
    payload: dict[str, Any], *, code_root: Path | None
) -> None:
    if set(payload) != {
        "execution_class_id",
        "fingerprint_schema",
        "fingerprint_sha256",
        "kind",
        "schema_version",
        "subject",
    }:
        raise ExecutionSafetyCertificationError(
            "execution-safety descriptor fields differ from schema"
        )
    if (
        payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["kind"] != DESCRIPTOR_KIND
        or payload["execution_class_id"] != EXECUTION_CLASS_ID
        or payload["fingerprint_schema"] != FINGERPRINT_SCHEMA
    ):
        raise ExecutionSafetyCertificationError(
            "execution-safety descriptor identity differs"
        )
    subject = payload["subject"]
    if not isinstance(subject, dict):
        raise ExecutionSafetyCertificationError(
            "execution-safety descriptor subject must be a mapping"
        )
    expected_static = _static_execution_safety_subject()
    implementation_files = subject.get("implementation_files")
    if not isinstance(implementation_files, dict) or set(implementation_files) != set(
        IMPLEMENTATION_FILE_PATHS
    ):
        raise ExecutionSafetyCertificationError(
            "execution-safety implementation file set differs"
        )
    for relative, digest in implementation_files.items():
        _require_sha256(digest, name=f"execution-safety implementation {relative}")
    normalized = copy.deepcopy(subject)
    normalized["implementation_files"] = {}
    if normalized != expected_static:
        raise ExecutionSafetyCertificationError(
            "execution-safety semantic subject differs from the reviewed class"
        )
    fingerprint = _require_sha256(
        payload["fingerprint_sha256"], name="execution-safety fingerprint"
    )
    if fingerprint != sha256_value(subject):
        raise ExecutionSafetyCertificationError(
            "execution-safety descriptor fingerprint is inconsistent"
        )
    if code_root is not None and subject != build_execution_safety_subject(code_root):
        raise ExecutionSafetyCertificationError(
            "execution-safety descriptor does not match the current checkout"
        )


def load_execution_safety_descriptor_bytes(
    raw: bytes, *, code_root: Path | None = None
) -> dict[str, Any]:
    payload = _strict_json(raw, context="execution-safety descriptor")
    _validate_descriptor_payload(payload, code_root=code_root)
    return payload


def validate_execution_safety_descriptor_bytes(
    raw: bytes, *, code_root: Path | None = None
) -> ExecutionSafetyDescriptorBinding:
    payload = load_execution_safety_descriptor_bytes(raw, code_root=code_root)
    return ExecutionSafetyDescriptorBinding(
        payload=payload,
        execution_class_id=EXECUTION_CLASS_ID,
        fingerprint_sha256=payload["fingerprint_sha256"],
        descriptor_sha256=_sha256_bytes(raw),
        supported_world_sizes=SUPPORTED_WORLD_SIZES,
    )


def load_current_execution_safety_descriptor(
    code_root: Path,
) -> ExecutionSafetyDescriptorBinding:
    raw = _regular_file_bytes(
        code_root.resolve() / DESCRIPTOR_RELATIVE_PATH,
        role="checked-in execution-safety descriptor",
    )
    return validate_execution_safety_descriptor_bytes(raw, code_root=code_root)


def current_execution_safety_fingerprint(code_root: Path) -> str:
    """Compatibility spelling for consumers validating a running checkout."""

    return execution_safety_fingerprint(code_root)


def execution_safety_world_size_contract(
    descriptor: Mapping[str, Any], world_size: int
) -> dict[str, Any]:
    if isinstance(world_size, bool) or world_size not in SUPPORTED_WORLD_SIZES:
        raise ExecutionSafetyCertificationError(
            "world size is outside the certified execution class"
        )
    payload = dict(descriptor)
    _validate_descriptor_payload(payload, code_root=None)
    subject = payload["subject"]
    key = str(world_size)
    return {
        "world_size": world_size,
        "rank_local_samples": subject["batch_partition"][
            "per_rank_samples_by_world_size"
        ][key],
        "optimizer_microsteps": subject["batch_partition"][
            "optimizer_microsteps_by_world_size"
        ][key],
        "threads_per_rank": subject["cpu_threading"][
            "threads_per_rank_by_world_size"
        ][key],
        "requested_fsdp_sharding_strategy": subject["fsdp"][
            "requested_sharding_strategy"
        ],
        "effective_fsdp_sharding_strategy": subject["fsdp"][
            "effective_sharding_strategy_by_world_size"
        ][key],
        "fsdp_wrapper_count": subject["fsdp"]["expected_wrapper_count"],
        "wrapped_transformer_blocks": subject["fsdp"][
            "expected_transformer_block_count"
        ],
        "fsdp_full_state_export": subject["checkpoint_resume"][
            "full_state_export_by_world_size"
        ][key],
    }


def condense_legacy_gpu_preflight_evidence(
    *,
    plan_path: Path,
    report_path: Path,
    completion_path: Path,
    expected_world_size: int,
    descriptor_raw: bytes,
    code_root: Path,
) -> dict[str, dict[str, Any]]:
    """Validate and condense one legacy real-GPU report for certification.

    The production semantic validator is run against the raw report and
    completion.  Because these Candidate E reports predate the execution-class
    fingerprint, the returned bridge explicitly remains a review attestation.
    Independent certificate review checks the recorded historical Git-backed
    handler/runtime identities once; later jobs reuse the accepted certificate
    without replaying that history.
    """

    if expected_world_size not in (1, 2):
        raise ExecutionSafetyCertificationError(
            "legacy condensation is defined only for the observed W=1/W=2 pilots"
        )
    descriptor = validate_execution_safety_descriptor_bytes(
        descriptor_raw, code_root=code_root
    )
    from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
        OUTPUT_NAME,
        validate_qwen3_v2_gpu_preflight_completion,
    )
    from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json
    from posttrain_circuits.workflows.contracts import WorkflowPlan

    plan_payload, _plan_file_sha256 = read_strict_json(
        plan_path,
        context=f"legacy W={expected_world_size} GPU preflight plan",
        max_bytes=4 * 1024 * 1024,
    )
    try:
        plan = WorkflowPlan.from_payload(plan_payload)
    except ValueError as error:
        raise ExecutionSafetyCertificationError(
            f"legacy W={expected_world_size} workflow plan is invalid: {error}"
        ) from error

    report, report_file_sha256 = read_strict_json(
        report_path,
        context=f"legacy W={expected_world_size} GPU preflight report",
        max_bytes=32 * 1024 * 1024,
    )
    completion, completion_file_sha256 = read_strict_json(
        completion_path,
        context=f"legacy W={expected_world_size} GPU preflight completion",
        max_bytes=4 * 1024 * 1024,
    )
    if not isinstance(report, dict) or not isinstance(completion, dict):
        raise ExecutionSafetyCertificationError(
            "legacy GPU preflight evidence must contain JSON mappings"
        )
    plan_sha256 = _require_sha256(
        completion.get("plan_sha256"), name="legacy preflight plan"
    )
    unit_id = completion.get("unit_id")
    if (
        plan.sha256() != plan_sha256
        or plan.workflow_id != completion.get("workflow_id")
        or not isinstance(unit_id, str)
    ):
        raise ExecutionSafetyCertificationError(
            "legacy completion identity differs from its immutable workflow plan"
        )
    try:
        unit = plan.unit(unit_id)
    except (KeyError, ValueError) as error:
        raise ExecutionSafetyCertificationError(
            "legacy completion unit is absent from its immutable workflow plan"
        ) from error
    if unit.task != "qwen3_v2_gpu_preflight" or unit.output_names != (OUTPUT_NAME,):
        raise ExecutionSafetyCertificationError(
            "legacy workflow unit is not the exact GPU-preflight task"
        )
    context = SimpleNamespace(
        workflow_id=plan.workflow_id,
        expected_input_hashes={
            identity.name: identity.sha256 for identity in unit.content_inputs
        },
        expected_output_paths={OUTPUT_NAME: report_path},
    )
    try:
        validate_qwen3_v2_gpu_preflight_completion(completion, context)
    except ValueError as error:
        raise ExecutionSafetyCertificationError(
            f"legacy W={expected_world_size} production validation failed: {error}"
        ) from error
    if report.get("world_size") != expected_world_size:
        raise ExecutionSafetyCertificationError(
            "legacy GPU preflight world size differs from the condensation target"
        )
    output_files = completion.get("output_files")
    if (
        not isinstance(output_files, dict)
        or report_file_sha256 not in output_files.values()
    ):
        raise ExecutionSafetyCertificationError(
            "legacy completion does not bind the raw report digest"
        )
    execution = completion.get("execution")
    if not isinstance(execution, dict):
        raise ExecutionSafetyCertificationError(
            "legacy completion lacks execution provenance"
        )
    allocation_sha256 = _require_sha256(
        execution.get("allocation_sha256"), name="legacy preflight allocation"
    )
    git_commit = report.get("git_commit")
    if not isinstance(git_commit, str) or GIT_COMMIT.fullmatch(git_commit) is None:
        raise ExecutionSafetyCertificationError(
            "legacy preflight report Git commit is invalid"
        )
    acceptance_commit = "5c0bb34cce288aef8908e498a6f5d3259b998f5b"
    git_environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }

    def git_bytes(*arguments: str, check: bool = True) -> bytes:
        try:
            result = subprocess.run(
                (
                    "/usr/bin/git",
                    "-c",
                    "core.fsmonitor=false",
                    "-C",
                    str(code_root.resolve()),
                    *arguments,
                ),
                check=check,
                capture_output=True,
                env=git_environment,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ExecutionSafetyCertificationError(
                "legacy preflight Git evidence cannot be authenticated"
            ) from error
        return bytes(result.stdout)

    ancestry = subprocess.run(
        (
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(code_root.resolve()),
            "merge-base",
            "--is-ancestor",
            acceptance_commit,
            git_commit,
        ),
        capture_output=True,
        env=git_environment,
    )
    if ancestry.returncode != 0:
        raise ExecutionSafetyCertificationError(
            "legacy preflight report is not descended from Candidate E acceptance"
        )
    changed_raw = git_bytes(
        "diff",
        "--name-only",
        "-z",
        f"{acceptance_commit}..{git_commit}",
        "--",
    )
    if changed_raw and not changed_raw.endswith(b"\0"):
        raise ExecutionSafetyCertificationError(
            "legacy preflight Git changed-path stream is malformed"
        )
    try:
        changed = {
            value.decode("utf-8", errors="strict")
            for value in changed_raw.split(b"\0")
            if value
        }
    except UnicodeDecodeError as error:
        raise ExecutionSafetyCertificationError(
            "legacy preflight Git changed paths are not UTF-8"
        ) from error
    if changed - {"docs/refactor/current_handoff.md"}:
        raise ExecutionSafetyCertificationError(
            "legacy preflight lineage contains non-handoff implementation changes"
        )
    historical_paths = {
        "observed_preflight_handler_sha256": (
            "scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py"
        ),
        "observed_runtime_dependency_lock_sha256": (
            "deployments/qwen3_v2_gpu_preflight/dependency-lock.json"
        ),
        "observed_runtime_package_manifest_sha256": (
            "deployments/qwen3_v2_gpu_preflight/package-manifest.json"
        ),
    }
    for evidence_name, relative in historical_paths.items():
        if hashlib.sha256(git_bytes("show", f"{git_commit}:{relative}")).hexdigest() != (
            LEGACY_PREFLIGHT_EVIDENCE_SURFACE[evidence_name]
        ):
            raise ExecutionSafetyCertificationError(
                f"legacy preflight historical {evidence_name} differs"
            )
    training_contract_sha256 = sha256_value(report.get("training_contract"))
    subject = descriptor.payload["subject"]
    runtime = subject["fixed_runtime"]["gpu_preflight"]
    if training_contract_sha256 != runtime["training_contract_sha256"]:
        raise ExecutionSafetyCertificationError(
            "legacy preflight training contract differs from the execution class"
        )
    workflow_id = completion.get("workflow_id")
    job_id = execution.get("job_id")
    for name, value in (("workflow_id", workflow_id), ("job_id", job_id)):
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 128
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None
        ):
            raise ExecutionSafetyCertificationError(
                f"legacy preflight {name} is invalid"
            )
    real_row = {
        "status": "accepted_historical_real_gpu_pilot",
        "job_id": job_id,
        "workflow_id": workflow_id,
        "plan_sha256": plan_sha256,
        "allocation_sha256": allocation_sha256,
        "git_commit": git_commit,
        "report_sha256": report_file_sha256,
        "completion_sha256": completion_file_sha256,
    }
    bridge_row = {
        "world_size": expected_world_size,
        "report_sha256": report_file_sha256,
        "completion_sha256": completion_file_sha256,
        "report_git_commit": git_commit,
        "candidate_e_acceptance_commit": (
            "5c0bb34cce288aef8908e498a6f5d3259b998f5b"
        ),
        "accepted_lineage_relation": (
            "candidate_e_acceptance_commit"
            if git_commit == "5c0bb34cce288aef8908e498a6f5d3259b998f5b"
            else "handoff_only_descendant_of_candidate_e_acceptance"
        ),
        **LEGACY_PREFLIGHT_EVIDENCE_SURFACE,
        **_certified_successor_preflight_surface(descriptor),
    }
    return {
        "real_gpu_pilot": real_row,
        "legacy_evidence_condensation": bridge_row,
    }


def _validate_review(review: object, *, require_accepted: bool) -> tuple[str, str | None]:
    if not isinstance(review, dict) or set(review) != set(PROPOSED_REVIEW):
        raise ExecutionSafetyCertificationError(
            "execution-safety certification review fields differ"
        )
    status_value = review["status"]
    if status_value == "proposed":
        if review != PROPOSED_REVIEW:
            raise ExecutionSafetyCertificationError(
                "proposed execution-safety certification has review metadata"
            )
        if require_accepted:
            raise ExecutionSafetyCertificationError(
                "execution-safety certification remains proposed"
            )
        return "proposed", None
    if status_value != "accepted":
        raise ExecutionSafetyCertificationError(
            "execution-safety certification review status is invalid"
        )
    commit = review["reviewed_implementation_commit"]
    if not isinstance(commit, str) or GIT_COMMIT.fullmatch(commit) is None:
        raise ExecutionSafetyCertificationError(
            "accepted execution-safety certification lacks an implementation commit"
        )
    for field in ("reviewer", "rationale"):
        if not isinstance(review[field], str) or not review[field].strip():
            raise ExecutionSafetyCertificationError(
                f"accepted execution-safety certification lacks {field}"
            )
    timestamp = review["reviewed_at_utc"]
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ExecutionSafetyCertificationError(
            "execution-safety certification review time must be explicit UTC"
        )
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise ExecutionSafetyCertificationError(
            "execution-safety certification review time is invalid"
        ) from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ExecutionSafetyCertificationError(
            "execution-safety certification review time must use UTC"
        )
    return "accepted", commit


def certification_core_sha256(payload: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(payload))
    core.pop("review", None)
    return sha256_value(core)


def load_execution_safety_certification_bytes(
    raw: bytes,
    descriptor_raw: bytes,
    *,
    require_accepted: bool = True,
    code_root: Path | None = None,
) -> ExecutionSafetyCertificationBinding:
    descriptor = validate_execution_safety_descriptor_bytes(
        descriptor_raw, code_root=code_root
    )
    payload = _strict_yaml(raw, context="execution-safety certification")
    if set(payload) != {
        "schema_version",
        "kind",
        "certification_id",
        "execution_class",
        "evidence_basis",
        "legacy_evidence_condensation",
        "reuse_policy",
        "residual_risk",
        "review",
    }:
        raise ExecutionSafetyCertificationError(
            "execution-safety certification fields differ from schema"
        )
    if (
        payload["schema_version"] != 1
        or isinstance(payload["schema_version"], bool)
        or payload["kind"] != CERTIFICATION_KIND
        or payload["certification_id"] != CERTIFICATION_ID
    ):
        raise ExecutionSafetyCertificationError(
            "execution-safety certification identity differs"
        )
    expected_class = {
        "execution_class_id": EXECUTION_CLASS_ID,
        "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
        "descriptor_sha256": descriptor.descriptor_sha256,
        "fingerprint_sha256": descriptor.fingerprint_sha256,
        "supported_world_sizes": list(SUPPORTED_WORLD_SIZES),
    }
    if payload["execution_class"] != expected_class:
        raise ExecutionSafetyCertificationError(
            "execution-safety certification descriptor binding differs"
        )
    evidence = payload["evidence_basis"]
    if not isinstance(evidence, dict) or set(evidence) != {
        "static_fail_closed_fixtures",
        "real_gpu_pilots",
    }:
        raise ExecutionSafetyCertificationError(
            "execution-safety certification evidence fields differ"
        )
    static = evidence["static_fail_closed_fixtures"]
    if static != {
        "covered_world_sizes": [1, 2, 3, 4],
        "status": "passed",
        "scope": (
            "batch_loss_token_fsdp_checkpoint_memory_and_manifest_fail_closed"
        ),
    }:
        raise ExecutionSafetyCertificationError(
            "execution-safety static evidence differs"
        )
    real = evidence["real_gpu_pilots"]
    if not isinstance(real, dict) or set(real) != {"1", "2", "3", "4"}:
        raise ExecutionSafetyCertificationError(
            "execution-safety real-pilot evidence must cover the four world-size rows"
        )
    gaps: list[int] = []
    for world_size in SUPPORTED_WORLD_SIZES:
        key = str(world_size)
        row = real[key]
        if not isinstance(row, dict):
            raise ExecutionSafetyCertificationError(
                f"execution-safety W={key} evidence must be a mapping"
            )
        if row.get("status") == "accepted_historical_real_gpu_pilot":
            expected_real_fields = {
                "status",
                "job_id",
                "workflow_id",
                "plan_sha256",
                "allocation_sha256",
                "git_commit",
                "report_sha256",
                "completion_sha256",
            }
            if set(row) != expected_real_fields:
                raise ExecutionSafetyCertificationError(
                    f"execution-safety W={key} real evidence fields differ"
                )
            for field in ("job_id", "workflow_id"):
                value = row[field]
                if (
                    not isinstance(value, str)
                    or not value
                    or len(value) > 128
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None
                ):
                    raise ExecutionSafetyCertificationError(
                        f"execution-safety W={key} {field} is invalid"
                    )
            for field in (
                "plan_sha256",
                "allocation_sha256",
                "report_sha256",
                "completion_sha256",
            ):
                _require_sha256(row[field], name=f"W={key} {field}")
            commit = row["git_commit"]
            if not isinstance(commit, str) or GIT_COMMIT.fullmatch(commit) is None:
                raise ExecutionSafetyCertificationError(
                    f"execution-safety W={key} git_commit is invalid"
                )
        elif row == {
            "status": "not_yet_observed",
            "coverage": "static_fail_closed_fixture_only",
        }:
            gaps.append(world_size)
        else:
            raise ExecutionSafetyCertificationError(
                f"execution-safety W={key} real-pilot evidence differs"
            )
    if 1 in gaps or 2 in gaps:
        raise ExecutionSafetyCertificationError(
            "initial execution certification requires accepted W=1 and W=2 pilots"
        )
    condensation = payload["legacy_evidence_condensation"]
    expected_condensation_fields = {
        "bridge_authority",
        "bridge_kind",
        "source_reports_embed_execution_fingerprint",
        "attested_execution_fingerprint_sha256",
        "statement",
        "evidence",
    }
    if not isinstance(condensation, dict) or set(condensation) != (
        expected_condensation_fields
    ):
        raise ExecutionSafetyCertificationError(
            "legacy evidence condensation fields differ"
        )
    expected_statement = (
        "The W1/W2 reports predate this fingerprint and do not directly attest it; "
        "acceptance of this certificate independently reviews the recorded report "
        "semantics, their Git-recorded historical handler/runtime/training-contract "
        "surface, and the safety-relevant delta to the exact successor descriptor."
    )
    if (
        condensation["bridge_authority"] != "certification_review_block"
        or condensation["bridge_kind"]
        != "recorded_legacy_evidence_condensation_requires_independent_certification_review"
        or condensation["source_reports_embed_execution_fingerprint"] is not False
        or condensation["attested_execution_fingerprint_sha256"]
        != descriptor.fingerprint_sha256
        or condensation["statement"] != expected_statement
    ):
        raise ExecutionSafetyCertificationError(
            "legacy evidence condensation overstates or changes its bridge"
        )
    condensed_rows = condensation["evidence"]
    if not isinstance(condensed_rows, dict) or set(condensed_rows) != {"1", "2"}:
        raise ExecutionSafetyCertificationError(
            "legacy evidence condensation must contain exactly W=1 and W=2"
        )
    expected_surface = {
        **LEGACY_PREFLIGHT_EVIDENCE_SURFACE,
        **_certified_successor_preflight_surface(descriptor),
    }
    for world_size, relation in (
        (1, "handoff_only_descendant_of_candidate_e_acceptance"),
        (2, "candidate_e_acceptance_commit"),
    ):
        key = str(world_size)
        row = condensed_rows[key]
        source = real[key]
        expected_row = {
            "world_size": world_size,
            "report_sha256": source["report_sha256"],
            "completion_sha256": source["completion_sha256"],
            "report_git_commit": source["git_commit"],
            "candidate_e_acceptance_commit": (
                "5c0bb34cce288aef8908e498a6f5d3259b998f5b"
            ),
            "accepted_lineage_relation": relation,
            **expected_surface,
        }
        if row != expected_row:
            raise ExecutionSafetyCertificationError(
                f"legacy W={world_size} evidence condensation differs"
            )
    expected_reuse = {
        "reuse_requires_exact_fingerprint_match": True,
        "non_invalidating_dimensions": list(NON_INVALIDATING_DIMENSIONS),
        "invalidating_dimensions": list(INVALIDATING_DIMENSIONS),
        "per_experiment_scientific_validation_still_required": True,
    }
    if payload["reuse_policy"] != expected_reuse:
        raise ExecutionSafetyCertificationError(
            "execution-safety certification reuse policy differs"
        )
    risk = payload["residual_risk"]
    if not isinstance(risk, dict) or set(risk) != {
        "world_sizes_without_real_gpu_pilot",
        "statement",
        "runtime_must_still_fail_closed",
    } or risk["world_sizes_without_real_gpu_pilot"] != gaps or risk[
        "runtime_must_still_fail_closed"
    ] is not True or not isinstance(risk["statement"], str) or not risk[
        "statement"
    ].strip():
        raise ExecutionSafetyCertificationError(
            "execution-safety certification residual risk differs"
        )
    status_value, commit = _validate_review(
        payload["review"], require_accepted=require_accepted
    )
    return ExecutionSafetyCertificationBinding(
        execution_class_id=EXECUTION_CLASS_ID,
        certification_id=CERTIFICATION_ID,
        fingerprint_sha256=descriptor.fingerprint_sha256,
        descriptor_sha256=descriptor.descriptor_sha256,
        certification_sha256=_sha256_bytes(raw),
        certification_core_sha256=certification_core_sha256(payload),
        supported_world_sizes=SUPPORTED_WORLD_SIZES,
        review_status=status_value,
        reviewed_implementation_commit=commit,
        evidence_basis=copy.deepcopy(evidence),
    )


def validate_certification_bytes(
    raw: bytes,
    descriptor_raw: bytes,
    *,
    require_accepted: bool = True,
    code_root: Path | None = None,
) -> ExecutionSafetyCertificationBinding:
    return load_execution_safety_certification_bytes(
        raw,
        descriptor_raw,
        require_accepted=require_accepted,
        code_root=code_root,
    )


__all__ = [
    "CERTIFICATION_CONTENT_NAME",
    "CERTIFICATION_ID",
    "CERTIFICATION_RELATIVE_PATH",
    "DESCRIPTOR_CONTENT_NAME",
    "DESCRIPTOR_RELATIVE_PATH",
    "EXECUTION_CLASS_ID",
    "ExecutionSafetyCertificationBinding",
    "ExecutionSafetyCertificationError",
    "ExecutionSafetyDescriptorBinding",
    "IMPLEMENTATION_FILE_PATHS",
    "INVALIDATING_DIMENSIONS",
    "NON_INVALIDATING_DIMENSIONS",
    "PROPOSED_REVIEW",
    "SUCCESSOR_AMENDMENT_RELATIVE_PATH",
    "SUPPORTED_WORLD_SIZES",
    "build_execution_safety_descriptor",
    "build_execution_safety_subject",
    "certification_core_sha256",
    "condense_legacy_gpu_preflight_evidence",
    "current_execution_safety_fingerprint",
    "execution_safety_fingerprint",
    "execution_safety_world_size_contract",
    "load_current_execution_safety_descriptor",
    "load_execution_safety_certification_bytes",
    "load_execution_safety_descriptor_bytes",
    "validate_certification_bytes",
    "validate_execution_safety_descriptor_bytes",
]
