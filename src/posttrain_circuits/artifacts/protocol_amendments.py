"""Fail-closed reviewed protocol amendments with non-self-referential Git binding."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow
from posttrain_circuits.artifacts.execution_safety_certification import (
    CERTIFICATION_ID as EXECUTION_SAFETY_CERTIFICATION_ID,
    CERTIFICATION_RELATIVE_PATH,
    DESCRIPTOR_RELATIVE_PATH,
    EXECUTION_CLASS_ID,
    INVALIDATING_DIMENSIONS,
    NON_INVALIDATING_DIMENSIONS,
    PROPOSED_REVIEW as EXECUTION_SAFETY_PROPOSED_REVIEW,
    SUCCESSOR_AMENDMENT_RELATIVE_PATH,
    SUPPORTED_WORLD_SIZES,
    ExecutionSafetyCertificationBinding,
    certification_core_sha256,
    load_execution_safety_certification_bytes,
    load_execution_safety_descriptor_bytes,
)
from posttrain_circuits.artifacts.execution_science_protocol import (
    ExecutionScienceProtocolError,
    PROPOSED_REVIEW as EXECUTION_SCIENCE_PROPOSED_REVIEW,
    load_execution_science_protocol_yaml_bytes,
    validate_execution_science_protocol_review_transition,
)


AMENDMENT_RELATIVE_PATH = Path("prereg/amendments/qwen3_v2_g0_elastic_v1.yaml")
BASE_PREREG_RELATIVE_PATH = Path("prereg/qwen3_v2.yaml")
AMENDMENT_ID = "qwen3_v2_g0_elastic_v1"
EXECUTION_CLASS_AMENDMENT_ID = "qwen3_v2_g0_execution_class_v2"
CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH = Path(
    "prereg/execution_science/qwen3_v2_g0_candidate_e_seed42_v1.yaml"
)
CANDIDATE_E_SCIENTIFIC_CONFIG_SCHEMA = "qwen3-v2-candidate-e-scientific-config-v1"
CANDIDATE_E_SCIENTIFIC_CONFIG_SHA256 = (
    "407675667c513735bf5fe56bac3614d0feacbff102467f79c1cd168e8d733468"
)
CANDIDATE_E_SCIENTIFIC_LOCATOR_PATHS = (
    "anti_shortcut.report_path",
    "experiment.random_reward_calibration_path",
    "output_root",
    "production_safety.initial_checkpoint_path",
    "production_safety.probe_cohort_manifest",
    "production_safety.readiness_report",
    "state_source.store_path",
    "task.dataset_family_path",
)
BASE_PREREG_VERSION = "qwen3_v2"
BASE_PREREG_SHA256 = "8d6bdeab0b9302c8824c4709f556c6c41a896bd2cfce21e7794d131d176ba0a4"
FROZEN_IMPLEMENTATION_COMMIT = "b2d505b297dae1d56311616e9a68fb7df14b7bee"
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
ALLOWED_POST_IMPLEMENTATION_PATHS = (
    str(AMENDMENT_RELATIVE_PATH),
    "docs/refactor/current_handoff.md",
)
ALLOWED_AFTER_ACCEPTANCE_PATHS = ("docs/refactor/current_handoff.md",)
PROPOSED_REVIEW = {
    "status": "proposed",
    "reviewed_implementation_commit": None,
    "reviewer": None,
    "reviewed_at_utc": None,
    "rationale": None,
}
MAX_LINEAGE_COMMITS = 256
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


class ProtocolAmendmentError(ValueError):
    """A protocol amendment or its Git review provenance is invalid."""


@dataclass(frozen=True)
class ProtocolAmendmentBinding:
    path: Path
    amendment_id: str
    sha256: str
    git_commit: str
    reviewed_implementation_commit: str


@dataclass(frozen=True)
class ExecutionClassAmendmentBinding:
    path: Path
    amendment_id: str
    sha256: str
    git_commit: str
    reviewed_implementation_commit: str
    descriptor_path: Path
    certification_path: Path
    certification_binding: ExecutionSafetyCertificationBinding


@dataclass(frozen=True)
class _CommitStep:
    commit: str
    parent: str
    changed_paths: tuple[str, ...]


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ProtocolAmendmentError("protocol amendment keys must be strings")
        if key in mapping:
            raise ProtocolAmendmentError(f"protocol amendment contains duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_yaml(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = yaml.load(raw.decode("utf-8", errors="strict"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ProtocolAmendmentError(f"{context} is not strict UTF-8 YAML: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolAmendmentError(f"{context} must be a YAML mapping")
    return value


def load_protocol_amendment_bytes(raw: bytes) -> dict[str, Any]:
    """Parse and schema-validate the scheduler-managed 1--4 GPU amendment."""

    payload = _load_yaml(raw, context="Qwen3-v2 elastic G0 amendment")
    expected_top = {
        "schema_version",
        "amendment_id",
        "base_preregistration",
        "superseded_execution_amendment",
        "scope",
        "allocation_semantics",
        "batch_token_invariants",
        "checkpoint_retry_invariants",
        "scientific_invariants",
        "implementation_acceptance",
        "review",
    }
    if set(payload) != expected_top:
        raise ProtocolAmendmentError("protocol amendment top-level fields differ from schema")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise ProtocolAmendmentError("protocol amendment schema_version must be 1")
    if payload["amendment_id"] != AMENDMENT_ID:
        raise ProtocolAmendmentError("protocol amendment identity differs from the reviewed design")

    expected_base = {
        "path": str(BASE_PREREG_RELATIVE_PATH),
        "version": BASE_PREREG_VERSION,
        "sha256": BASE_PREREG_SHA256,
        "frozen_implementation_commit": FROZEN_IMPLEMENTATION_COMMIT,
    }
    if payload["base_preregistration"] != expected_base:
        raise ProtocolAmendmentError("protocol amendment base preregistration differs")

    expected_superseded = {
        "path": "prereg/amendments/qwen3_v2_g0_2gpu_v1.yaml",
        "amendment_id": "qwen3_v2_g0_2gpu_v1",
        "accepted_sha256": (
            "014b5dc78619870eee3a6f6175323b0418c5c597fa9625fd41c8e15ee353b6c2"
        ),
        "role": "recoverable_fixed_two_gpu_baseline_only",
    }
    if payload["superseded_execution_amendment"] != expected_superseded:
        raise ProtocolAmendmentError(
            "protocol amendment superseded-baseline binding differs"
        )

    expected_scope = {
        "task": "qwen3_v2_g0",
        "workflow_id": "qwen3-v2-g0-elastic-v1",
        "unit_id": "g0",
        "seed": 42,
        "claim_scope": (
            "full_mechanism_pipeline_feasibility_only_not_confirmatory_primary_endpoint"
        ),
    }
    if payload["scope"] != expected_scope:
        raise ProtocolAmendmentError("protocol amendment scope differs from the reviewed design")

    expected_allocation = {
        "allocation_source": "server_scheduler_running_manifest_and_environment",
        "request_side_execution_profile": "forbidden",
        "request_side_gpu_count": "forbidden",
        "request_side_resources": "forbidden",
        "reviewed_world_sizes": [1, 2, 3, 4],
        "fsdp_requested_sharding_strategy": "FULL_SHARD",
        "fsdp_effective_sharding_strategy_by_world_size": {
            "1": "NO_SHARD",
            "2": "FULL_SHARD",
            "3": "FULL_SHARD",
            "4": "FULL_SHARD",
        },
        "cpu_thread_partition": (
            "allocated_cpu_cores_divided_equally_by_actual_world_size"
        ),
        "running_attempt_resize": "forbidden",
        "preflight_required": (
            "successful_accepted_implementation_lineage_elastic_gpu_preflight_matrix"
        ),
    }
    if payload["allocation_semantics"] != expected_allocation:
        raise ProtocolAmendmentError("protocol amendment allocation semantics differ")

    expected_batch = {
        "batch_partition_protocol": "allocation_neutral_exact_global_batch_v1",
        "global_logical_batch_size": 64,
        "max_per_rank_microbatch_size": 4,
        "max_model_input_length": 1536,
        "frozen_prompt_population_max_tokens": 1246,
        "teacher_demo_max_new_tokens": 256,
        "derived_max_model_input_tokens": 1502,
        "overlength_policy": (
            "reject_without_truncation_before_any_training_forward"
        ),
        "preflight_model_input_tokens": 1536,
        "prompt_population_size": 256,
        "prompt_ids_unique": True,
        "accepted_view_prompt_order": "exactly_manifest_ordered_prompt_ids",
        "prompt_population_alignment": (
            "exact_multiple_of_global_logical_batch_size"
        ),
        "optimizer_microsteps_by_world_size": {"1": 16, "2": 8, "3": 6, "4": 4},
        "per_rank_samples_by_world_size": {
            "1": [64],
            "2": [32, 32],
            "3": [22, 21, 21],
            "4": [16, 16, 16, 16],
        },
        "three_gpu_microbatch_schedule": {
            "rank_0": [4, 4, 4, 4, 4, 2],
            "rank_1": [4, 4, 4, 4, 4, 1],
            "rank_2": [4, 4, 4, 4, 4, 1],
        },
        "sample_order": "one_global_64_slot_window_independent_of_world_size",
        "objective_normalization": (
            "exact_global_sequence_mean_after_framework_accumulation_and_rank_averaging"
        ),
        "token_budget": 2_000_000,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "token_accounting": (
            "exact_cross_rank_sum_reserved_before_any_backward_in_each_optimizer_window"
        ),
        "token_stop_boundary": "stop_before_an_optimizer_update_that_would_exceed_budget",
        "max_optimizer_steps": 120,
        "max_steps_role": "independent_safety_ceiling_first_limit_reached_stops_training",
    }
    if payload["batch_token_invariants"] != expected_batch:
        raise ProtocolAmendmentError("protocol amendment batch/token terms differ")
    if (
        expected_batch["frozen_prompt_population_max_tokens"]
        + expected_batch["teacher_demo_max_new_tokens"]
        != expected_batch["derived_max_model_input_tokens"]
        or expected_batch["derived_max_model_input_tokens"]
        > expected_batch["max_model_input_length"]
        or expected_batch["preflight_model_input_tokens"]
        != expected_batch["max_model_input_length"]
    ):
        raise ProtocolAmendmentError(
            "protocol amendment model-input length bound is inconsistent"
        )
    for world_size_text, per_rank in expected_batch[
        "per_rank_samples_by_world_size"
    ].items():
        world_size = int(world_size_text)
        if (
            sum(per_rank) != expected_batch["global_logical_batch_size"]
            or len(per_rank) != world_size
            or max(per_rank) > (
                expected_batch["optimizer_microsteps_by_world_size"][world_size_text]
                * expected_batch["max_per_rank_microbatch_size"]
            )
        ):
            raise ProtocolAmendmentError(
                "protocol amendment global-batch schedule is inconsistent"
            )

    expected_checkpoint = {
        "checkpoint_boundary": "completed_optimizer_updates_only",
        "state_source_kind": "teacher_demo",
        "state_source_cursor_protocol": "teacher-demo-round-robin-v2-accepted-view",
        "state_source_rng": "none",
        "rank_local_trainer_state": "checkpointed_and_restored_per_rank",
        "global_token_budget_state": "identical_across_all_ranks",
        "same_world_resume": "required_and_deterministically_compared",
        "changed_world_resume": "rejected_before_state_load",
        "retry_policy": (
            "every_scheduler_attempt_uses_an_isolated_workspace_and_restarts_from_frozen_inputs"
        ),
    }
    if payload["checkpoint_retry_invariants"] != expected_checkpoint:
        raise ProtocolAmendmentError(
            "protocol amendment checkpoint/retry terms differ"
        )

    expected_scientific = {
        "full_parameter_training": True,
        "unchanged": [
            "model_teacher_and_tokenizer_revisions",
            "prompt_and_sampling_protocols",
            "seed_42",
            "global_logical_sample_order",
            "teacher_demo_state_source_and_sequence_normalized_canonical_sft",
            "dataset_and_prerequisite_hash_bindings",
            "optimizer_and_learning_rate",
            "evaluation_and_checkpoint_cadence_in_optimizer_steps",
            "G0_gate_and_claim_scope",
            "artifact_and_completion_semantics",
        ],
        "numerical_equivalence": {
            "standard": (
                "identical_scientific_protocol_with_expected_distributed_"
                "floating_point_reduction_variation"
            ),
            "bitwise_identity_across_world_sizes": False,
        },
        "forbidden_extensions": [
            "confirmatory_primary_endpoint",
            "full_three_seed_factorial",
            "Gemma_replication",
        ],
    }
    if payload["scientific_invariants"] != expected_scientific:
        raise ProtocolAmendmentError("protocol amendment scientific invariants differ")

    expected_acceptance = {
        "mechanism": "reviewed_implementation_commit_plus_metadata_only_descendant",
        "allowed_post_implementation_paths": list(ALLOWED_POST_IMPLEMENTATION_PATHS),
        "allowed_after_acceptance_paths": list(ALLOWED_AFTER_ACCEPTANCE_PATHS),
        "proposed_document_must_preexist_in_implementation_commit": True,
        "accepted_document_may_change_review_block_only": True,
        "preflight_request_and_execution_must_share_accepted_implementation_lineage": True,
        "required_static_world_size_fixtures": [1, 2, 3, 4],
        "required_central_gpu_pilots_before_g0": [1, 2, 3, 4],
    }
    if payload["implementation_acceptance"] != expected_acceptance:
        raise ProtocolAmendmentError("protocol amendment acceptance mechanism differs")

    review = payload["review"]
    if not isinstance(review, dict) or set(review) != set(PROPOSED_REVIEW):
        raise ProtocolAmendmentError("protocol amendment review fields differ from schema")
    status_value = review["status"]
    if status_value == "proposed":
        if review != PROPOSED_REVIEW:
            raise ProtocolAmendmentError("proposed amendment cannot contain acceptance metadata")
    elif status_value == "accepted":
        implementation_commit = review["reviewed_implementation_commit"]
        if not isinstance(implementation_commit, str) or GIT_COMMIT.fullmatch(
            implementation_commit
        ) is None:
            raise ProtocolAmendmentError("accepted amendment lacks a reviewed implementation commit")
        for field in ("reviewer", "rationale"):
            if not isinstance(review[field], str) or not review[field].strip():
                raise ProtocolAmendmentError(f"accepted amendment lacks {field}")
        timestamp = review["reviewed_at_utc"]
        if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
            raise ProtocolAmendmentError("accepted amendment review time must be explicit UTC")
        try:
            parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
        except ValueError as error:
            raise ProtocolAmendmentError("accepted amendment review time is invalid") from error
        if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ProtocolAmendmentError("accepted amendment review time must use UTC")
    else:
        raise ProtocolAmendmentError("protocol amendment review status is invalid")
    return payload


def _validate_execution_class_review(review: object) -> None:
    if not isinstance(review, dict) or set(review) != set(PROPOSED_REVIEW):
        raise ProtocolAmendmentError(
            "execution-class amendment review fields differ from schema"
        )
    if review["status"] == "proposed":
        if review != PROPOSED_REVIEW:
            raise ProtocolAmendmentError(
                "proposed execution-class amendment cannot contain review metadata"
            )
        return
    if review["status"] != "accepted":
        raise ProtocolAmendmentError(
            "execution-class amendment review status is invalid"
        )
    implementation_commit = review["reviewed_implementation_commit"]
    if not isinstance(implementation_commit, str) or GIT_COMMIT.fullmatch(
        implementation_commit
    ) is None:
        raise ProtocolAmendmentError(
            "accepted execution-class amendment lacks an implementation commit"
        )
    for field in ("reviewer", "rationale"):
        if not isinstance(review[field], str) or not review[field].strip():
            raise ProtocolAmendmentError(
                f"accepted execution-class amendment lacks {field}"
            )
    timestamp = review["reviewed_at_utc"]
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise ProtocolAmendmentError(
            "execution-class amendment review time must be explicit UTC"
        )
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise ProtocolAmendmentError(
            "execution-class amendment review time is invalid"
        ) from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProtocolAmendmentError(
            "execution-class amendment review time must use UTC"
        )


def load_execution_class_amendment_bytes(raw: bytes) -> dict[str, Any]:
    """Parse the proposed reusable-execution-class successor amendment.

    Candidate E's accepted v1 amendment remains immutable and continues to be
    parsed by :func:`load_protocol_amendment_bytes`.  This versioned loader
    cannot reinterpret v1 or silently bless this proposed successor.
    """

    payload = _load_yaml(raw, context="Qwen3-v2 execution-class G0 amendment")
    expected_top = {
        "schema_version",
        "amendment_id",
        "base_preregistration",
        "superseded_execution_amendment",
        "scope",
        "scheduler_managed_allocation",
        "execution_safety_certification",
        "candidate_e_scientific_protocol",
        "evidence_and_reuse",
        "candidate_e_migration",
        "implementation_acceptance",
        "review",
    }
    if set(payload) != expected_top:
        raise ProtocolAmendmentError(
            "execution-class amendment top-level fields differ from schema"
        )
    if (
        payload["schema_version"] != 2
        or isinstance(payload["schema_version"], bool)
        or payload["amendment_id"] != EXECUTION_CLASS_AMENDMENT_ID
    ):
        raise ProtocolAmendmentError("execution-class amendment identity differs")
    expected_base = {
        "path": str(BASE_PREREG_RELATIVE_PATH),
        "version": BASE_PREREG_VERSION,
        "sha256": BASE_PREREG_SHA256,
        "frozen_implementation_commit": FROZEN_IMPLEMENTATION_COMMIT,
    }
    if payload["base_preregistration"] != expected_base:
        raise ProtocolAmendmentError(
            "execution-class amendment base preregistration differs"
        )
    expected_superseded = {
        "path": str(AMENDMENT_RELATIVE_PATH),
        "amendment_id": AMENDMENT_ID,
        "accepted_sha256": (
            "ff34cc53a85abe409f65ebe1ad3ca4d46b08a0ecd2117b76a27e22eea633a725"
        ),
        "role": "accepted_candidate_e_per_g0_four_pilot_gate_preserved_as_history",
    }
    if payload["superseded_execution_amendment"] != expected_superseded:
        raise ProtocolAmendmentError(
            "execution-class amendment predecessor binding differs"
        )
    expected_scope = {
        "task": "qwen3_v2_g0",
        "workflow_id_policy": "fresh_opaque_count_neutral_per_request",
        "workflow_id_prefix": "qwen3-v2-g0-elastic-",
        "unit_id": "g0",
        "science_identity": "per_experiment_execution_science_protocol",
        "claim_scope": (
            "execution_class_only_scientific_claims_remain_per_experiment"
        ),
    }
    if payload["scope"] != expected_scope:
        raise ProtocolAmendmentError("execution-class amendment scope differs")
    expected_allocation = {
        "profile_cardinality": "one_elastic_profile_for_one_scientific_task",
        "gpu_count_policy": "scheduler",
        "allocation_candidates": list(SUPPORTED_WORLD_SIZES),
        "request_execution_profile": "omitted",
        "request_resources": "omitted",
        "request_gpu_count_and_identity": "omitted",
        "count_and_uuid_selection": "server_scheduler_claim_time_only",
        "running_attempt_resize": "forbidden",
    }
    if payload["scheduler_managed_allocation"] != expected_allocation:
        raise ProtocolAmendmentError(
            "execution-class amendment allocation semantics differ"
        )
    certification = payload["execution_safety_certification"]
    if not isinstance(certification, dict) or set(certification) != {
        "execution_class_id",
        "certification_id",
        "descriptor_path",
        "descriptor_sha256",
        "certification_path",
        "certification_core_sha256",
        "fingerprint_sha256",
        "supported_world_sizes",
    }:
        raise ProtocolAmendmentError(
            "execution-class certification binding fields differ"
        )
    if (
        certification["execution_class_id"] != EXECUTION_CLASS_ID
        or certification["certification_id"]
        != EXECUTION_SAFETY_CERTIFICATION_ID
        or certification["descriptor_path"] != str(DESCRIPTOR_RELATIVE_PATH)
        or certification["certification_path"]
        != str(CERTIFICATION_RELATIVE_PATH)
        or certification["supported_world_sizes"]
        != list(SUPPORTED_WORLD_SIZES)
    ):
        raise ProtocolAmendmentError(
            "execution-class certification identity differs"
        )
    for field in (
        "descriptor_sha256",
        "certification_core_sha256",
        "fingerprint_sha256",
    ):
        value = certification[field]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ProtocolAmendmentError(
                f"execution-class certification {field} is not a SHA-256 digest"
            )
    expected_scientific_protocol = {
        "path": str(CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH),
        "protocol_id": "qwen3-v2-g0-candidate-e-seed-42-v1",
        "unit_id": "g0",
        "seed": 42,
        "storage_neutral_resolved_config_sha256": (
            "6c3942f4d6cf87329e1c70ba32b0b8d4cdbd526a3bbcfdcb1afd3e1674662657"
        ),
        "joint_acceptance_with_execution_class": "required",
        "execution_certification_substitutes_for_scientific_review": False,
        "per_experiment_config_artifact_and_completion_validation": "required",
    }
    if payload["candidate_e_scientific_protocol"] != expected_scientific_protocol:
        raise ProtocolAmendmentError(
            "execution-class amendment Candidate E scientific protocol differs"
        )
    expected_evidence = {
        "generic_server_scheduler_requirement": (
            "reviewed_semantic_correctness_for_every_supported_world_size"
        ),
        "project_specific_predecessor_gate": (
            "four_distinct_real_accepted_lineage_pilots_for_each_g0_plan"
        ),
        "successor_gate": (
            "accepted_reusable_execution_class_certification_with_exact_fingerprint_match"
        ),
        "real_gpu_evidence_world_sizes": [1, 2],
        "static_fail_closed_evidence_world_sizes": [1, 2, 3, 4],
        "world_sizes_without_real_gpu_observation": [3, 4],
        "repeat_four_real_pilots_for_each_new_experiment": False,
        "per_experiment_scientific_protocol_and_completion_validation": "required",
        "non_invalidating_dimensions": list(NON_INVALIDATING_DIMENSIONS),
        "invalidating_dimensions": list(INVALIDATING_DIMENSIONS),
    }
    if payload["evidence_and_reuse"] != expected_evidence:
        raise ProtocolAmendmentError(
            "execution-class amendment evidence/reuse policy differs"
        )
    expected_migration = {
        "predecessor_amendment_mutated": False,
        "current_real_gpu_evidence": "W1_and_W2_accepted",
        "current_static_evidence": "W1_W2_W3_W4_passed",
        "remaining_risk": (
            "W3_uneven_tail_and_W4_topology_have_no_successful_real_GPU_observation"
        ),
        "mitigation": (
            "runtime_fail_closed_semantic_artifact_and_completion_validation_at_actual_"
            "scheduler_selected_world_size"
        ),
        "minimum_change": (
            "replace_per_plan_eight_artifact_matrix_with_one_descriptor_and_one_"
            "reusable_certification"
        ),
        "operational_before_independent_acceptance": False,
    }
    if payload["candidate_e_migration"] != expected_migration:
        raise ProtocolAmendmentError(
            "execution-class amendment Candidate E migration terms differ"
        )
    expected_acceptance = {
        "mechanism": (
            "reviewed_implementation_plus_joint_review_and_exact_safety_fingerprint"
        ),
        "allowed_post_implementation_paths": [
            str(SUCCESSOR_AMENDMENT_RELATIVE_PATH),
            str(CERTIFICATION_RELATIVE_PATH),
            str(CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH),
            "docs/refactor/current_handoff.md",
        ],
        "post_acceptance_execution_class_changes": (
            "non_safety_commits_and_merges_allowed_when_artifacts_and_critical_blobs_match"
        ),
        "post_acceptance_candidate_e_science": (
            "exact_reviewed_source_and_scientific_config_required"
        ),
        "proposed_documents_must_preexist_in_implementation_commit": True,
        "amendment_and_certification_review_blocks_only_at_acceptance": True,
        "amendment_and_certification_acceptance_same_commit": True,
        "required_static_world_size_fixtures": [1, 2, 3, 4],
        "required_real_gpu_pilots_for_initial_certification": [1, 2],
    }
    if payload["implementation_acceptance"] != expected_acceptance:
        raise ProtocolAmendmentError(
            "execution-class amendment acceptance mechanism differs"
        )
    _validate_execution_class_review(payload["review"])
    return payload


def validate_elastic_g0_config(config: dict[str, Any], amendment: dict[str, Any]) -> None:
    """Require an allocation-neutral G0 config to implement the amendment."""

    if amendment.get("amendment_id") == EXECUTION_CLASS_AMENDMENT_ID:
        validate_execution_class_g0_config(config, amendment)
        return

    load_protocol_amendment_bytes(yaml.safe_dump(amendment, sort_keys=False).encode("utf-8"))
    trainer = config.get("trainer")
    scheduler = config.get("scheduler_g0")
    if not isinstance(trainer, dict) or not isinstance(scheduler, dict):
        raise ProtocolAmendmentError("elastic G0 config lacks trainer or scheduler binding")
    if "batch_size" in trainer or "gradient_accumulation_steps" in trainer:
        raise ProtocolAmendmentError(
            "elastic G0 config must omit allocation-specific batch fields"
        )
    experiment = config.get("experiment")
    task = config.get("task")
    state_source = config.get("state_source")
    supervision = config.get("supervision")
    if (
        not isinstance(experiment, dict)
        or experiment.get("name") != "canonical_sft"
        or not isinstance(task, dict)
        or task.get("num_examples")
        != amendment["batch_token_invariants"]["prompt_population_size"]
        or not isinstance(state_source, dict)
        or state_source.get("name") != "teacher_demo"
        or not isinstance(supervision, dict)
        or supervision.get("name") != "canonical_sft"
        or supervision.get("normalization") != "sequence"
        or config.get("g0", {}).get("full_parameter_training")
        is not amendment["scientific_invariants"]["full_parameter_training"]
    ):
        raise ProtocolAmendmentError(
            "elastic G0 requires the reviewed prompt population and deterministic "
            "teacher-demo canonical SFT"
        )
    expected = amendment["batch_token_invariants"]
    integer_values = (
        trainer.get("global_batch_size"),
        trainer.get("max_microbatch_size"),
        trainer.get("max_model_input_length"),
        trainer.get("token_budget"),
        trainer.get("max_steps"),
        state_source.get("max_prompt_tokens"),
        state_source.get("max_new_tokens"),
    )
    if any(type(value) is not int for value in integer_values):
        raise ProtocolAmendmentError("elastic G0 batch/token values must be integers")
    observed = {
        "allocation_contract": scheduler.get("allocation_contract"),
        "batch_partition_protocol": trainer.get("batch_partition_protocol"),
        "global_logical_batch_size": trainer.get("global_batch_size"),
        "max_per_rank_microbatch_size": trainer.get("max_microbatch_size"),
        "max_model_input_length": trainer.get("max_model_input_length"),
        "frozen_prompt_population_max_tokens": state_source.get(
            "max_prompt_tokens"
        ),
        "teacher_demo_max_new_tokens": state_source.get("max_new_tokens"),
        "token_budget": trainer.get("token_budget"),
        "token_budget_unit": trainer.get("token_budget_unit"),
        "max_optimizer_steps": trainer.get("max_steps"),
    }
    required = {
        "allocation_contract": "manifest_driven_scheduler_gpu_v1",
        "batch_partition_protocol": expected["batch_partition_protocol"],
        "global_logical_batch_size": expected["global_logical_batch_size"],
        "max_per_rank_microbatch_size": expected["max_per_rank_microbatch_size"],
        "max_model_input_length": expected["max_model_input_length"],
        "frozen_prompt_population_max_tokens": expected[
            "frozen_prompt_population_max_tokens"
        ],
        "teacher_demo_max_new_tokens": expected["teacher_demo_max_new_tokens"],
        "token_budget": expected["token_budget"],
        "token_budget_unit": expected["token_budget_unit"],
        "max_optimizer_steps": expected["max_optimizer_steps"],
    }
    if observed != required:
        raise ProtocolAmendmentError(
            f"elastic G0 config violates batch/token invariants: {observed!r}"
        )
    if (
        config.get("protocol_amendment_path") != str(AMENDMENT_RELATIVE_PATH)
        or config.get("protocol_track") != BASE_PREREG_VERSION
        or scheduler.get("batch_partition_protocol")
        != expected["batch_partition_protocol"]
    ):
        raise ProtocolAmendmentError("elastic G0 config lacks the amendment identity")


def validate_execution_class_g0_config(
    config: dict[str, Any], amendment: dict[str, Any]
) -> None:
    """Validate G0 science plus its reusable execution-class binding."""

    load_execution_class_amendment_bytes(
        yaml.safe_dump(amendment, sort_keys=False).encode("utf-8")
    )
    trainer = config.get("trainer")
    scheduler = config.get("scheduler_g0")
    if not isinstance(trainer, dict) or not isinstance(scheduler, dict):
        raise ProtocolAmendmentError(
            "execution-class G0 config lacks trainer or scheduler binding"
        )
    forbidden_matrix_fields = {
        "gpu_preflight_matrix",
        "gpu_preflight_report_sha256",
        "gpu_preflight_completion_sha256",
        "gpu_preflight_git_commit",
    }
    if forbidden_matrix_fields & set(scheduler):
        raise ProtocolAmendmentError(
            "execution-class G0 config must not bind per-experiment preflight rows"
        )
    expected_scheduler_fields = {
        "allocation_contract",
        "artifact_namespace",
        "batch_partition_protocol",
        "execution_safety_certification_sha256",
        "execution_safety_class_id",
        "execution_safety_descriptor_sha256",
        "execution_safety_fingerprint_sha256",
        "execution_safety_supported_world_sizes",
        "execution_science_protocol_git_commit",
        "execution_science_protocol_id",
        "execution_science_protocol_path",
        "execution_science_protocol_reviewed_implementation_commit",
        "execution_science_protocol_sha256",
        "model_revision",
        "protocol_amendment_id",
        "protocol_amendment_sha256",
        "request_git_commit",
        "reviewed_implementation_commit",
        "task",
        "teacher_revision",
        "tokenizer_fingerprint",
    }
    if set(scheduler) != expected_scheduler_fields:
        raise ProtocolAmendmentError(
            "execution-class G0 scheduler provenance fields differ"
        )
    if "batch_size" in trainer or "gradient_accumulation_steps" in trainer:
        raise ProtocolAmendmentError(
            "execution-class G0 config must omit allocation-specific batch fields"
        )
    experiment = config.get("experiment")
    task = config.get("task")
    state_source = config.get("state_source")
    supervision = config.get("supervision")
    if (
        not isinstance(experiment, dict)
        or experiment.get("name") != "canonical_sft"
        or not isinstance(task, dict)
        or task.get("num_examples") != 256
        or not isinstance(state_source, dict)
        or state_source.get("name") != "teacher_demo"
        or not isinstance(supervision, dict)
        or supervision.get("name") != "canonical_sft"
        or supervision.get("normalization") != "sequence"
        or config.get("g0", {}).get("full_parameter_training") is not True
    ):
        raise ProtocolAmendmentError(
            "execution-class G0 requires the reviewed deterministic teacher-demo "
            "canonical SFT protocol"
        )
    integer_values = (
        trainer.get("global_batch_size"),
        trainer.get("max_microbatch_size"),
        trainer.get("max_model_input_length"),
        trainer.get("token_budget"),
        trainer.get("max_steps"),
        state_source.get("max_prompt_tokens"),
        state_source.get("max_new_tokens"),
    )
    if any(type(value) is not int for value in integer_values):
        raise ProtocolAmendmentError(
            "execution-class G0 batch/token values must be integers"
        )
    expected_science = {
        "batch_partition_protocol": "allocation_neutral_exact_global_batch_v1",
        "global_batch_size": 64,
        "max_microbatch_size": 4,
        "max_model_input_length": 1536,
        "token_budget": 2_000_000,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "max_steps": 120,
        "max_prompt_tokens": 1246,
        "max_new_tokens": 256,
    }
    observed_science = {
        "batch_partition_protocol": trainer.get("batch_partition_protocol"),
        "global_batch_size": trainer.get("global_batch_size"),
        "max_microbatch_size": trainer.get("max_microbatch_size"),
        "max_model_input_length": trainer.get("max_model_input_length"),
        "token_budget": trainer.get("token_budget"),
        "token_budget_unit": trainer.get("token_budget_unit"),
        "max_steps": trainer.get("max_steps"),
        "max_prompt_tokens": state_source.get("max_prompt_tokens"),
        "max_new_tokens": state_source.get("max_new_tokens"),
    }
    if observed_science != expected_science:
        raise ProtocolAmendmentError(
            "execution-class G0 config violates batch/token invariants"
        )
    certification = amendment["execution_safety_certification"]
    expected_scheduler = {
        "allocation_contract": "manifest_driven_scheduler_gpu_v1",
        "batch_partition_protocol": "allocation_neutral_exact_global_batch_v1",
        "execution_safety_class_id": certification["execution_class_id"],
        "execution_safety_descriptor_sha256": certification["descriptor_sha256"],
        "execution_safety_fingerprint_sha256": certification["fingerprint_sha256"],
        "execution_safety_supported_world_sizes": list(SUPPORTED_WORLD_SIZES),
    }
    for field, expected in expected_scheduler.items():
        if scheduler.get(field) != expected:
            raise ProtocolAmendmentError(
                f"execution-class G0 config has invalid {field}"
            )
    certification_sha256 = scheduler.get(
        "execution_safety_certification_sha256"
    )
    if (
        not isinstance(certification_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", certification_sha256) is None
    ):
        raise ProtocolAmendmentError(
            "execution-class G0 config lacks the certification content identity"
        )
    if (
        config.get("protocol_amendment_path")
        != str(SUCCESSOR_AMENDMENT_RELATIVE_PATH)
        or config.get("protocol_track") != BASE_PREREG_VERSION
    ):
        raise ProtocolAmendmentError(
            "execution-class G0 config lacks the successor amendment identity"
        )
    # Candidate E science is reviewed by its separate execution-science
    # protocol artifact.  This validator owns only the execution-class
    # projection; conflating the two would make seed and experiment identity
    # invalidate a reusable GPU-topology certification.


def _delete_optional_dotted_path(payload: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    parent: object = payload
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            return
        parent = parent[part]
    if isinstance(parent, dict):
        parent.pop(parts[-1], None)


def candidate_e_scientific_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return Candidate E science without scheduler provenance or storage locators."""

    if not isinstance(config, dict):
        raise ProtocolAmendmentError("Candidate E scientific config must be a mapping")
    normalized = copy.deepcopy(config)
    normalized.pop("scheduler_g0", None)
    normalized.pop("protocol_amendment_path", None)
    for dotted in CANDIDATE_E_SCIENTIFIC_LOCATOR_PATHS:
        _delete_optional_dotted_path(normalized, dotted)
    return {
        "schema": CANDIDATE_E_SCIENTIFIC_CONFIG_SCHEMA,
        "config": normalized,
    }


def candidate_e_scientific_config_sha256(config: dict[str, Any]) -> str:
    payload = candidate_e_scientific_config_payload(config)
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolAmendmentError(
            "Candidate E scientific config is not canonical JSON"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


# Compatibility name for callers that have not yet migrated their import.  The
# implementation now validates only the elastic amendment and cannot bless the
# historical two-GPU contract.
validate_two_gpu_g0_config = validate_elastic_g0_config


def _git_bytes(code_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            (
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(code_root),
                *arguments,
            ),
            check=True,
            capture_output=True,
            env=_GIT_ENVIRONMENT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProtocolAmendmentError(
            f"protocol amendment Git validation failed: git {' '.join(arguments)}"
        ) from error
    return result.stdout


def _git(code_root: Path, *arguments: str) -> str:
    try:
        return _git_bytes(code_root, *arguments).decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise ProtocolAmendmentError(
            f"protocol amendment Git output is not strict UTF-8: git {' '.join(arguments)}"
        ) from error


def _git_changed_paths(
    code_root: Path,
    parent_commit: str,
    commit: str,
) -> tuple[str, ...]:
    raw = _git_bytes(
        code_root,
        "diff-tree",
        "--no-ext-diff",
        "--no-commit-id",
        "--name-only",
        "--diff-filter=ACDMRTUXB",
        "-r",
        "-z",
        parent_commit,
        commit,
        "--",
    )
    if not raw:
        return ()
    if not raw.endswith(b"\0"):
        raise ProtocolAmendmentError("protocol amendment Git path output is not NUL terminated")
    encoded_paths = raw[:-1].split(b"\0")
    if any(not path for path in encoded_paths):
        raise ProtocolAmendmentError("protocol amendment Git path output contains an empty path")
    try:
        return tuple(path.decode("utf-8", errors="strict") for path in encoded_paths)
    except UnicodeDecodeError as error:
        raise ProtocolAmendmentError(
            "protocol amendment Git path is not strict UTF-8"
        ) from error


def _unsafe_untracked_paths(code_root: Path) -> tuple[str, ...]:
    """Enumerate untracked paths without applying ignore or exclude rules."""

    raw = _git_bytes(code_root, "ls-files", "--others", "-z", "--")
    if not raw:
        return ()
    if not raw.endswith(b"\0"):
        raise ProtocolAmendmentError("protocol amendment untracked paths are malformed")
    try:
        paths = tuple(
            item.decode("utf-8", errors="strict") for item in raw[:-1].split(b"\0")
        )
    except UnicodeDecodeError as error:
        raise ProtocolAmendmentError(
            "protocol amendment untracked path is not strict UTF-8"
        ) from error
    if any(not path for path in paths):
        raise ProtocolAmendmentError("protocol amendment untracked paths contain an empty path")
    unsafe: list[str] = []
    for path in paths:
        parsed = Path(path)
        if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
            raise ProtocolAmendmentError(
                "protocol amendment untracked path is not canonical"
            )
        if path == ".codex/config.toml":
            continue
        if parsed.suffix == ".pyc" and "__pycache__" in parsed.parts:
            continue
        unsafe.append(path)
    return tuple(unsafe)


def _git_commit_parents(code_root: Path, commit: str) -> tuple[str, ...]:
    raw = _git(code_root, "show", "-s", "--format=%P", commit, "--")
    if not raw:
        return ()
    parents = tuple(raw.split())
    if any(GIT_COMMIT.fullmatch(parent) is None for parent in parents):
        raise ProtocolAmendmentError("protocol amendment Git parent output is malformed")
    return parents


def _is_ancestor(code_root: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            (
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(code_root),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ),
            capture_output=True,
            env=_GIT_ENVIRONMENT,
        )
    except OSError as error:
        raise ProtocolAmendmentError("protocol amendment ancestry validation failed") from error
    if result.returncode not in {0, 1}:
        raise ProtocolAmendmentError("protocol amendment ancestry validation failed")
    return result.returncode == 0


def _linear_commit_steps(
    *,
    code_root: Path,
    start_commit: str,
    end_commit: str,
    role: str,
    allow_empty: bool,
) -> tuple[_CommitStep, ...]:
    """Enumerate every commit on one unbroken, single-parent Git chain."""

    if GIT_COMMIT.fullmatch(start_commit) is None or GIT_COMMIT.fullmatch(end_commit) is None:
        raise ProtocolAmendmentError(f"{role} contains a malformed Git commit")
    if start_commit == end_commit:
        if allow_empty:
            return ()
        raise ProtocolAmendmentError(f"{role} commit chain is empty")
    if not _is_ancestor(code_root, start_commit, end_commit):
        raise ProtocolAmendmentError(f"{role} commit chain is disconnected")

    history = _git(
        code_root,
        "rev-list",
        "--reverse",
        "--topo-order",
        f"--max-count={MAX_LINEAGE_COMMITS + 1}",
        f"{start_commit}..{end_commit}",
        "--",
    )
    if not history:
        raise ProtocolAmendmentError(f"{role} commit chain is empty")

    rows = history.splitlines()
    if len(rows) > MAX_LINEAGE_COMMITS:
        raise ProtocolAmendmentError(
            f"{role} commit chain exceeds {MAX_LINEAGE_COMMITS} commits"
        )
    previous = start_commit
    steps: list[_CommitStep] = []
    for row in rows:
        fields = row.split()
        if len(fields) != 1 or GIT_COMMIT.fullmatch(fields[0]) is None:
            raise ProtocolAmendmentError(f"{role} commit chain is malformed")
        commit = fields[0]
        parents = _git_commit_parents(code_root, commit)
        if len(parents) > 1:
            raise ProtocolAmendmentError(f"{role} commit chain contains a merge")
        if len(parents) != 1:
            raise ProtocolAmendmentError(f"{role} commit chain contains a parentless commit")
        parent = parents[0]
        if parent != previous:
            raise ProtocolAmendmentError(f"{role} commit chain is disconnected")
        changed_paths = _git_changed_paths(code_root, parent, commit)
        if not changed_paths:
            raise ProtocolAmendmentError(f"{role} commit chain contains an empty commit")
        steps.append(
            _CommitStep(
                commit=commit,
                parent=parent,
                changed_paths=changed_paths,
            )
        )
        previous = commit
    if previous != end_commit:
        raise ProtocolAmendmentError(f"{role} commit chain does not reach its endpoint")
    return tuple(steps)


def _validate_acceptance_chain(
    *,
    code_root: Path,
    implementation_commit: str,
    candidate_commit: str,
    proposed: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
) -> str:
    """Require exactly one review-only acceptance on an otherwise handoff-only chain."""

    steps = _linear_commit_steps(
        code_root=code_root,
        start_commit=implementation_commit,
        end_commit=candidate_commit,
        role=f"{role} implementation-to-candidate",
        allow_empty=False,
    )
    amendment_path = str(AMENDMENT_RELATIVE_PATH)
    handoff_paths = set(ALLOWED_AFTER_ACCEPTANCE_PATHS)
    acceptance_commit: str | None = None
    accepted_at_transition: dict[str, Any] | None = None
    for step in steps:
        changed = set(step.changed_paths)
        if amendment_path in changed:
            if acceptance_commit is not None:
                raise ProtocolAmendmentError(
                    f"{role} commit chain changes the amendment more than once"
                )
            accepted_at_transition = load_protocol_amendment_bytes(
                _git(
                    code_root,
                    "show",
                    f"{step.commit}:{AMENDMENT_RELATIVE_PATH}",
                ).encode("utf-8")
            )
            validate_review_transition(
                proposed=proposed,
                accepted=accepted_at_transition,
                implementation_commit=implementation_commit,
                current_commit=step.commit,
                changed_paths=step.changed_paths,
                implementation_is_ancestor=True,
            )
            acceptance_commit = step.commit
        elif not changed <= handoff_paths:
            phase = "post-acceptance" if acceptance_commit is not None else "pre-acceptance"
            raise ProtocolAmendmentError(
                f"{role} lineage contains {phase} implementation changes"
            )
    if acceptance_commit is None or accepted_at_transition is None:
        raise ProtocolAmendmentError(
            f"{role} commit chain lacks one review-only amendment acceptance"
        )
    if accepted_at_transition != candidate:
        raise ProtocolAmendmentError(
            f"{role} accepted amendment changed after its review transition"
        )
    return acceptance_commit


def _validate_handoff_only_chain(
    *,
    code_root: Path,
    candidate_commit: str,
    current_commit: str,
    role: str,
) -> None:
    """Require every commit after a provenance candidate to change only the handoff."""

    steps = _linear_commit_steps(
        code_root=code_root,
        start_commit=candidate_commit,
        end_commit=current_commit,
        role=f"{role} candidate-to-current",
        allow_empty=True,
    )
    allowed = set(ALLOWED_AFTER_ACCEPTANCE_PATHS)
    for step in steps:
        if not set(step.changed_paths) <= allowed:
            raise ProtocolAmendmentError(
                f"{role} lineage contains post-acceptance implementation changes"
            )


def validate_review_transition(
    *,
    proposed: dict[str, Any],
    accepted: dict[str, Any],
    implementation_commit: str,
    current_commit: str,
    changed_paths: tuple[str, ...],
    implementation_is_ancestor: bool,
) -> None:
    """Validate the review-only transition independently of Git I/O."""

    load_protocol_amendment_bytes(yaml.safe_dump(proposed, sort_keys=False).encode("utf-8"))
    load_protocol_amendment_bytes(yaml.safe_dump(accepted, sort_keys=False).encode("utf-8"))
    if proposed["review"] != PROPOSED_REVIEW:
        raise ProtocolAmendmentError("implementation commit did not contain the proposed amendment")
    if accepted["review"]["status"] != "accepted":
        raise ProtocolAmendmentError("current amendment has not been accepted")
    if accepted["review"]["reviewed_implementation_commit"] != implementation_commit:
        raise ProtocolAmendmentError("accepted amendment names a different implementation commit")
    if not implementation_is_ancestor or current_commit == implementation_commit:
        raise ProtocolAmendmentError(
            "accepted amendment must be committed after its reviewed implementation"
        )
    normalized = copy.deepcopy(accepted)
    normalized["review"] = copy.deepcopy(PROPOSED_REVIEW)
    if normalized != proposed:
        raise ProtocolAmendmentError(
            "accepted amendment changed scientific terms after implementation review"
        )
    changed = set(changed_paths)
    allowed = set(ALLOWED_POST_IMPLEMENTATION_PATHS)
    if str(AMENDMENT_RELATIVE_PATH) not in changed or not changed <= allowed:
        raise ProtocolAmendmentError(
            "post-review commit range contains non-metadata implementation changes"
        )


def resolve_accepted_protocol_amendment(
    *,
    code_root: Path,
    configured_path: str,
    expected_head: str | None = None,
) -> ProtocolAmendmentBinding:
    """Resolve one accepted amendment and prove its two-commit review transition."""

    code_root = code_root.resolve()
    if configured_path != str(AMENDMENT_RELATIVE_PATH):
        raise ProtocolAmendmentError("configured protocol amendment path is not reviewed")
    path = code_root / AMENDMENT_RELATIVE_PATH
    raw = _regular_file_bytes_for_amendment(
        path, role="protocol amendment"
    )
    accepted = load_protocol_amendment_bytes(raw)
    if accepted["review"]["status"] != "accepted":
        raise ProtocolAmendmentError("protocol amendment remains proposed")
    if _git(
        code_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ProtocolAmendmentError("accepted amendment requires a clean checkout")
    if unsafe := _unsafe_untracked_paths(code_root):
        raise ProtocolAmendmentError(
            f"accepted amendment checkout contains unsafe ignored files: {unsafe!r}"
        )
    current_commit = _git(code_root, "rev-parse", "HEAD")
    if GIT_COMMIT.fullmatch(current_commit) is None:
        raise ProtocolAmendmentError("Git HEAD is not one immutable commit")
    if expected_head is not None and current_commit != expected_head:
        raise ProtocolAmendmentError("amendment validation HEAD differs from execution provenance")
    implementation_commit = accepted["review"]["reviewed_implementation_commit"]
    assert isinstance(implementation_commit, str)
    try:
        proposed_raw = _git(
            code_root,
            "show",
            f"{implementation_commit}:{AMENDMENT_RELATIVE_PATH}",
        ).encode("utf-8")
    except ProtocolAmendmentError as error:
        raise ProtocolAmendmentError(
            "reviewed implementation commit lacks the proposed amendment"
        ) from error
    proposed = load_protocol_amendment_bytes(proposed_raw)
    acceptance_commit = _validate_acceptance_chain(
        code_root=code_root,
        implementation_commit=implementation_commit,
        candidate_commit=current_commit,
        proposed=proposed,
        candidate=accepted,
        role="accepted amendment",
    )
    base_prereg = code_root / BASE_PREREG_RELATIVE_PATH
    if hashlib.sha256(
        _regular_file_bytes_for_amendment(
            base_prereg, role="base preregistration"
        )
    ).hexdigest() != BASE_PREREG_SHA256:
        raise ProtocolAmendmentError("base preregistration bytes changed")
    if _git(code_root, "rev-parse", "HEAD") != current_commit:
        raise ProtocolAmendmentError("amendment validation HEAD changed before completion")
    if _regular_file_bytes_for_amendment(
        path, role="protocol amendment"
    ) != raw:
        raise ProtocolAmendmentError("protocol amendment bytes changed during validation")
    if _git(
        code_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise ProtocolAmendmentError("accepted amendment checkout changed during validation")
    if unsafe := _unsafe_untracked_paths(code_root):
        raise ProtocolAmendmentError(
            f"accepted amendment checkout gained unsafe ignored files: {unsafe!r}"
        )
    return ProtocolAmendmentBinding(
        path=path,
        amendment_id=AMENDMENT_ID,
        sha256=hashlib.sha256(raw).hexdigest(),
        git_commit=acceptance_commit,
        reviewed_implementation_commit=implementation_commit,
    )


def validate_accepted_lineage_commit(
    *,
    code_root: Path,
    candidate_commit: str,
    current_binding: ProtocolAmendmentBinding,
    expected_head: str,
    role: str,
) -> None:
    """Prove that preflight/request provenance shares the accepted implementation."""

    if GIT_COMMIT.fullmatch(candidate_commit) is None:
        raise ProtocolAmendmentError(f"{role} is not a Git commit")
    current_commit = _git(code_root, "rev-parse", "HEAD")
    if current_commit != expected_head:
        raise ProtocolAmendmentError(f"{role} validation HEAD changed")
    implementation_commit = current_binding.reviewed_implementation_commit
    if not _is_ancestor(code_root, implementation_commit, candidate_commit):
        raise ProtocolAmendmentError(f"{role} predates the reviewed implementation")
    if not _is_ancestor(code_root, candidate_commit, current_commit):
        raise ProtocolAmendmentError(f"{role} is not an ancestor of execution HEAD")
    candidate_raw = _git(
        code_root,
        "show",
        f"{candidate_commit}:{AMENDMENT_RELATIVE_PATH}",
    ).encode("utf-8")
    candidate = load_protocol_amendment_bytes(candidate_raw)
    if (
        candidate["review"]["status"] != "accepted"
        or candidate["review"]["reviewed_implementation_commit"]
        != implementation_commit
    ):
        raise ProtocolAmendmentError(f"{role} did not contain the accepted amendment")
    proposed = load_protocol_amendment_bytes(
        _git(
            code_root,
            "show",
            f"{implementation_commit}:{AMENDMENT_RELATIVE_PATH}",
        ).encode("utf-8")
    )
    _validate_acceptance_chain(
        code_root=code_root,
        implementation_commit=implementation_commit,
        candidate_commit=candidate_commit,
        proposed=proposed,
        candidate=candidate,
        role=role,
    )
    _validate_handoff_only_chain(
        code_root=code_root,
        candidate_commit=candidate_commit,
        current_commit=current_commit,
        role=role,
    )
    if _git(code_root, "rev-parse", "HEAD") != current_commit:
        raise ProtocolAmendmentError(f"{role} validation HEAD changed before completion")
    if unsafe := _unsafe_untracked_paths(code_root):
        raise ProtocolAmendmentError(
            f"{role} validation found unsafe ignored files: {unsafe!r}"
        )


def validate_execution_class_review_transition(
    *,
    proposed_amendment: dict[str, Any],
    accepted_amendment: dict[str, Any],
    proposed_certification: dict[str, Any],
    accepted_certification: dict[str, Any],
    implementation_commit: str,
    current_commit: str,
    changed_paths: tuple[str, ...],
    implementation_is_ancestor: bool,
) -> None:
    """Require the joint review commit to change only its three review artifacts."""

    load_execution_class_amendment_bytes(
        yaml.safe_dump(proposed_amendment, sort_keys=False).encode("utf-8")
    )
    load_execution_class_amendment_bytes(
        yaml.safe_dump(accepted_amendment, sort_keys=False).encode("utf-8")
    )
    if proposed_amendment["review"] != PROPOSED_REVIEW:
        raise ProtocolAmendmentError(
            "implementation commit did not contain the proposed execution-class amendment"
        )
    if proposed_certification.get("review") != EXECUTION_SAFETY_PROPOSED_REVIEW:
        raise ProtocolAmendmentError(
            "implementation commit did not contain the proposed execution certification"
        )
    accepted_review = accepted_amendment["review"]
    certification_review = accepted_certification.get("review")
    if accepted_review.get("status") != "accepted" or not isinstance(
        certification_review, dict
    ) or certification_review.get("status") != "accepted":
        raise ProtocolAmendmentError(
            "execution-class amendment and certification must both be accepted"
        )
    if accepted_review != certification_review:
        raise ProtocolAmendmentError(
            "execution-class amendment and certification review identities differ"
        )
    if accepted_review["reviewed_implementation_commit"] != implementation_commit:
        raise ProtocolAmendmentError(
            "execution-class acceptance names a different implementation commit"
        )
    if not implementation_is_ancestor or current_commit == implementation_commit:
        raise ProtocolAmendmentError(
            "execution-class acceptance must descend from its implementation"
        )
    normalized_amendment = copy.deepcopy(accepted_amendment)
    normalized_amendment["review"] = copy.deepcopy(PROPOSED_REVIEW)
    if normalized_amendment != proposed_amendment:
        raise ProtocolAmendmentError(
            "execution-class amendment scientific terms changed at acceptance"
        )
    normalized_certification = copy.deepcopy(accepted_certification)
    normalized_certification["review"] = copy.deepcopy(
        EXECUTION_SAFETY_PROPOSED_REVIEW
    )
    if normalized_certification != proposed_certification:
        raise ProtocolAmendmentError(
            "execution-safety certification evidence changed at acceptance"
        )
    changed = set(changed_paths)
    required = {
        str(SUCCESSOR_AMENDMENT_RELATIVE_PATH),
        str(CERTIFICATION_RELATIVE_PATH),
        str(CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH),
    }
    allowed = required | {"docs/refactor/current_handoff.md"}
    if not required <= changed or not changed <= allowed:
        raise ProtocolAmendmentError(
            "execution-class acceptance commit contains non-review changes"
        )


def resolve_accepted_execution_class_amendment(
    *,
    code_root: Path,
    configured_path: str,
    expected_head: str | None = None,
) -> ExecutionClassAmendmentBinding:
    """Resolve the accepted reusable class against the current safety surface.

    Git history is used only to prove the one joint review transition.  Later
    request, plan, seed, output, and non-safety merge commits are deliberately
    outside this resolver: reuse depends on the current descriptor/CAS and its
    the descriptor's named safety-critical implementation blobs, not on
    per-experiment ancestry.
    """

    root = code_root.resolve()
    if configured_path != str(SUCCESSOR_AMENDMENT_RELATIVE_PATH):
        raise ProtocolAmendmentError(
            "configured execution-class amendment path is not reviewed"
        )
    amendment_path = root / SUCCESSOR_AMENDMENT_RELATIVE_PATH
    descriptor_path = root / DESCRIPTOR_RELATIVE_PATH
    certification_path = root / CERTIFICATION_RELATIVE_PATH
    science_path = root / CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH
    amendment_raw = _regular_file_bytes_for_amendment(
        amendment_path, role="execution-class amendment"
    )
    descriptor_raw = _regular_file_bytes_for_amendment(
        descriptor_path, role="execution-safety descriptor"
    )
    certification_raw = _regular_file_bytes_for_amendment(
        certification_path, role="execution-safety certification"
    )
    science_raw = _regular_file_bytes_for_amendment(
        science_path, role="Candidate E execution-science protocol"
    )
    amendment = load_execution_class_amendment_bytes(amendment_raw)
    if amendment["review"]["status"] != "accepted":
        raise ProtocolAmendmentError("execution-class amendment remains proposed")
    try:
        certification_binding = load_execution_safety_certification_bytes(
            certification_raw,
            descriptor_raw,
            require_accepted=True,
            code_root=root,
        )
    except ValueError as error:
        raise ProtocolAmendmentError(
            f"execution-class certification is invalid: {error}"
        ) from error
    certification_terms = amendment["execution_safety_certification"]
    expected_terms = {
        "execution_class_id": certification_binding.execution_class_id,
        "certification_id": certification_binding.certification_id,
        "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
        "descriptor_sha256": certification_binding.descriptor_sha256,
        "certification_path": str(CERTIFICATION_RELATIVE_PATH),
        "certification_core_sha256": (
            certification_binding.certification_core_sha256
        ),
        "fingerprint_sha256": certification_binding.fingerprint_sha256,
        "supported_world_sizes": list(
            certification_binding.supported_world_sizes
        ),
    }
    if certification_terms != expected_terms:
        raise ProtocolAmendmentError(
            "execution-class amendment does not bind the exact certification"
        )
    if amendment["review"] != _load_yaml(
        certification_raw, context="execution-safety certification"
    )["review"]:
        raise ProtocolAmendmentError(
            "execution-class amendment and certification reviews differ"
        )
    try:
        accepted_science = load_execution_science_protocol_yaml_bytes(
            science_raw, require_accepted=True
        )
    except ExecutionScienceProtocolError as error:
        raise ProtocolAmendmentError(
            f"Candidate E execution-science protocol is invalid: {error}"
        ) from error
    science_review = accepted_science["review"]
    if science_review != amendment["review"]:
        raise ProtocolAmendmentError(
            "execution-class and Candidate E science reviews differ"
        )
    science_terms = amendment["candidate_e_scientific_protocol"]
    science_class = accepted_science["execution_class"]
    if science_terms != {
        "path": str(CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH),
        "protocol_id": accepted_science["protocol_id"],
        "unit_id": accepted_science["scope"]["unit_id"],
        "seed": accepted_science["scope"]["seed"],
        "storage_neutral_resolved_config_sha256": accepted_science[
            "science_config"
        ]["storage_neutral_resolved_config_sha256"],
        "joint_acceptance_with_execution_class": "required",
        "execution_certification_substitutes_for_scientific_review": False,
        "per_experiment_config_artifact_and_completion_validation": "required",
    }:
        raise ProtocolAmendmentError(
            "execution-class amendment does not bind the Candidate E science protocol"
        )
    if science_class != {
        "execution_class_id": certification_binding.execution_class_id,
        "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
        "descriptor_sha256": certification_binding.descriptor_sha256,
        "fingerprint_sha256": certification_binding.fingerprint_sha256,
        "certification_id": certification_binding.certification_id,
        "certification_path": str(CERTIFICATION_RELATIVE_PATH),
        "certification_core_sha256": (
            certification_binding.certification_core_sha256
        ),
        "supported_world_sizes": list(certification_binding.supported_world_sizes),
    }:
        raise ProtocolAmendmentError(
            "Candidate E science protocol names a different execution class"
        )
    if _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    ):
        raise ProtocolAmendmentError(
            "accepted execution-class amendment requires a clean checkout"
        )
    current_commit = _git(root, "rev-parse", "HEAD")
    if GIT_COMMIT.fullmatch(current_commit) is None:
        raise ProtocolAmendmentError("Git HEAD is not one immutable commit")
    if expected_head is not None and current_commit != expected_head:
        raise ProtocolAmendmentError(
            "execution-class validation HEAD differs from execution provenance"
        )
    implementation_commit = amendment["review"]["reviewed_implementation_commit"]
    assert isinstance(implementation_commit, str)
    if certification_binding.reviewed_implementation_commit != implementation_commit:
        raise ProtocolAmendmentError(
            "execution-class certification names a different implementation"
        )
    try:
        proposed_amendment_raw = _git_bytes(
            root,
            "show",
            f"{implementation_commit}:{SUCCESSOR_AMENDMENT_RELATIVE_PATH}",
        )
        proposed_certification_raw = _git_bytes(
            root,
            "show",
            f"{implementation_commit}:{CERTIFICATION_RELATIVE_PATH}",
        )
        implementation_descriptor_raw = _git_bytes(
            root,
            "show",
            f"{implementation_commit}:{DESCRIPTOR_RELATIVE_PATH}",
        )
        proposed_science_raw = _git_bytes(
            root,
            "show",
            f"{implementation_commit}:{CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH}",
        )
    except ProtocolAmendmentError as error:
        raise ProtocolAmendmentError(
            "reviewed implementation lacks the proposed execution-class documents"
        ) from error
    if implementation_descriptor_raw != descriptor_raw:
        raise ProtocolAmendmentError(
            "execution-safety descriptor changed after implementation review"
        )
    try:
        implementation_descriptor = load_execution_safety_descriptor_bytes(
            implementation_descriptor_raw
        )
    except ValueError as error:
        raise ProtocolAmendmentError(
            f"reviewed implementation descriptor is invalid: {error}"
        ) from error
    for relative, expected_digest in implementation_descriptor["subject"][
        "implementation_files"
    ].items():
        try:
            implementation_blob = _git_bytes(
                root, "show", f"{implementation_commit}:{relative}"
            )
        except ProtocolAmendmentError as error:
            raise ProtocolAmendmentError(
                "reviewed implementation lacks safety-critical blob "
                f"{relative}"
            ) from error
        if hashlib.sha256(implementation_blob).hexdigest() != expected_digest:
            raise ProtocolAmendmentError(
                "reviewed implementation safety-critical blob differs from "
                f"descriptor: {relative}"
            )
    proposed_amendment = load_execution_class_amendment_bytes(
        proposed_amendment_raw
    )
    try:
        proposed_certification_binding = (
            load_execution_safety_certification_bytes(
                proposed_certification_raw,
                implementation_descriptor_raw,
                require_accepted=False,
            )
        )
    except ValueError as error:
        raise ProtocolAmendmentError(
            f"proposed execution certification is invalid: {error}"
        ) from error
    if proposed_certification_binding.review_status != "proposed":
        raise ProtocolAmendmentError(
            "implementation commit did not contain a proposed certification"
        )
    proposed_certification = _load_yaml(
        proposed_certification_raw, context="proposed execution certification"
    )
    accepted_certification = _load_yaml(
        certification_raw, context="accepted execution certification"
    )
    try:
        proposed_science = load_execution_science_protocol_yaml_bytes(
            proposed_science_raw
        )
    except ExecutionScienceProtocolError as error:
        raise ProtocolAmendmentError(
            f"proposed Candidate E science protocol is invalid: {error}"
        ) from error
    if proposed_science["review"] != EXECUTION_SCIENCE_PROPOSED_REVIEW:
        raise ProtocolAmendmentError(
            "implementation commit did not contain a proposed science protocol"
        )
    if not _is_ancestor(root, implementation_commit, current_commit) or (
        implementation_commit == current_commit
    ):
        raise ProtocolAmendmentError(
            "execution-class acceptance must descend from its implementation"
        )
    review_paths = (
        SUCCESSOR_AMENDMENT_RELATIVE_PATH,
        CERTIFICATION_RELATIVE_PATH,
        CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
    )
    touching_commits: list[str] = []
    for relative in review_paths:
        touching = _git(
            root,
            "rev-list",
            "--reverse",
            "--ancestry-path",
            f"{implementation_commit}..{current_commit}",
            "--",
            str(relative),
        ).splitlines()
        if len(touching) != 1 or GIT_COMMIT.fullmatch(touching[0]) is None:
            raise ProtocolAmendmentError(
                "execution-class review artifacts must change together exactly once"
            )
        touching_commits.append(touching[0])
    if len(set(touching_commits)) != 1:
        raise ProtocolAmendmentError(
            "execution-class review artifacts were not accepted jointly"
        )
    acceptance_commit = touching_commits[0]
    parents = _git_commit_parents(root, acceptance_commit)
    if len(parents) != 1 or not _is_ancestor(
        root, implementation_commit, acceptance_commit
    ):
        raise ProtocolAmendmentError(
            "execution-class acceptance must be one descendant review commit"
        )
    changed_paths = _git_changed_paths(root, parents[0], acceptance_commit)
    accepted_amendment_raw = _git_bytes(
        root,
        "show",
        f"{acceptance_commit}:{SUCCESSOR_AMENDMENT_RELATIVE_PATH}",
    )
    accepted_certification_raw = _git_bytes(
        root,
        "show",
        f"{acceptance_commit}:{CERTIFICATION_RELATIVE_PATH}",
    )
    accepted_science_raw = _git_bytes(
        root,
        "show",
        f"{acceptance_commit}:{CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH}",
    )
    if (
        accepted_amendment_raw != amendment_raw
        or accepted_certification_raw != certification_raw
        or accepted_science_raw != science_raw
    ):
        raise ProtocolAmendmentError(
            "execution-class review artifacts changed after acceptance"
        )
    accepted_at_transition = load_execution_class_amendment_bytes(
        accepted_amendment_raw
    )
    certification_at_transition = _load_yaml(
        accepted_certification_raw,
        context="accepted execution certification transition",
    )
    validate_execution_class_review_transition(
        proposed_amendment=proposed_amendment,
        accepted_amendment=accepted_at_transition,
        proposed_certification=proposed_certification,
        accepted_certification=certification_at_transition,
        implementation_commit=implementation_commit,
        current_commit=acceptance_commit,
        changed_paths=changed_paths,
        implementation_is_ancestor=True,
    )
    try:
        validate_execution_science_protocol_review_transition(
            proposed=proposed_science,
            accepted=accepted_science,
            implementation_commit=implementation_commit,
            acceptance_commit=acceptance_commit,
        )
    except ExecutionScienceProtocolError as error:
        raise ProtocolAmendmentError(
            f"Candidate E science review transition is invalid: {error}"
        ) from error
    if _git(root, "rev-parse", "HEAD") != current_commit:
        raise ProtocolAmendmentError(
            "execution-class validation HEAD changed before completion"
        )
    for path, raw, role in (
        (amendment_path, amendment_raw, "execution-class amendment"),
        (descriptor_path, descriptor_raw, "execution-safety descriptor"),
        (certification_path, certification_raw, "execution-safety certification"),
        (science_path, science_raw, "Candidate E execution-science protocol"),
    ):
        if _regular_file_bytes_for_amendment(path, role=role) != raw:
            raise ProtocolAmendmentError(f"{role} changed during validation")
    if _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    ):
        raise ProtocolAmendmentError(
            "accepted execution-class checkout changed during validation"
        )
    return ExecutionClassAmendmentBinding(
        path=amendment_path,
        amendment_id=EXECUTION_CLASS_AMENDMENT_ID,
        sha256=hashlib.sha256(amendment_raw).hexdigest(),
        git_commit=acceptance_commit,
        reviewed_implementation_commit=implementation_commit,
        descriptor_path=descriptor_path,
        certification_path=certification_path,
        certification_binding=certification_binding,
    )


def validate_execution_class_lineage_commit(
    *,
    code_root: Path,
    candidate_commit: str,
    current_binding: ExecutionClassAmendmentBinding,
    expected_head: str,
    role: str,
) -> None:
    """Bind request provenance between acceptance and the executing checkout."""

    if GIT_COMMIT.fullmatch(candidate_commit) is None:
        raise ProtocolAmendmentError(f"{role} is not a Git commit")
    root = code_root.resolve()
    current_commit = _git(root, "rev-parse", "HEAD")
    if current_commit != expected_head:
        raise ProtocolAmendmentError(f"{role} validation HEAD changed")
    if not _is_ancestor(root, current_binding.git_commit, candidate_commit):
        raise ProtocolAmendmentError(
            f"{role} predates or is disconnected from execution-class acceptance"
        )
    if not _is_ancestor(root, candidate_commit, current_commit):
        raise ProtocolAmendmentError(
            f"{role} is not an ancestor of the executing checkout"
        )
    amendment_raw = _regular_file_bytes_for_amendment(
        root / SUCCESSOR_AMENDMENT_RELATIVE_PATH,
        role="current execution-class amendment",
    )
    descriptor_raw = _regular_file_bytes_for_amendment(
        root / DESCRIPTOR_RELATIVE_PATH,
        role="current execution-safety descriptor",
    )
    certification_raw = _regular_file_bytes_for_amendment(
        root / CERTIFICATION_RELATIVE_PATH,
        role="current execution-safety certification",
    )
    try:
        certification_binding = load_execution_safety_certification_bytes(
            certification_raw,
            descriptor_raw,
            require_accepted=True,
            code_root=root,
        )
    except ValueError as error:
        raise ProtocolAmendmentError(
            f"{role} current execution-safety binding is invalid: {error}"
        ) from error
    amendment = load_execution_class_amendment_bytes(amendment_raw)
    if amendment["review"]["status"] != "accepted":
        raise ProtocolAmendmentError(f"{role} current execution class is not accepted")
    if (
        current_binding.path != root / SUCCESSOR_AMENDMENT_RELATIVE_PATH
        or current_binding.descriptor_path != root / DESCRIPTOR_RELATIVE_PATH
        or current_binding.certification_path != root / CERTIFICATION_RELATIVE_PATH
        or current_binding.amendment_id != EXECUTION_CLASS_AMENDMENT_ID
        or current_binding.sha256 != hashlib.sha256(amendment_raw).hexdigest()
        or current_binding.certification_binding != certification_binding
        or current_binding.reviewed_implementation_commit
        != amendment["review"]["reviewed_implementation_commit"]
        or amendment["execution_safety_certification"]
        != {
            "execution_class_id": certification_binding.execution_class_id,
            "certification_id": certification_binding.certification_id,
            "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
            "descriptor_sha256": certification_binding.descriptor_sha256,
            "certification_path": str(CERTIFICATION_RELATIVE_PATH),
            "certification_core_sha256": (
                certification_binding.certification_core_sha256
            ),
            "fingerprint_sha256": certification_binding.fingerprint_sha256,
            "supported_world_sizes": list(
                certification_binding.supported_world_sizes
            ),
        }
    ):
        raise ProtocolAmendmentError(
            f"{role} current execution-class fingerprint/CAS differs from its binding"
        )
    if _git(root, "rev-parse", "HEAD") != current_commit:
        raise ProtocolAmendmentError(
            f"{role} validation HEAD changed before completion"
        )


def _regular_file_bytes_for_amendment(path: Path, *, role: str) -> bytes:
    try:
        return read_regular_bytes_nofollow(path, context=role)
    except ValueError as error:
        raise ProtocolAmendmentError(str(error)) from error


__all__ = [
    "ALLOWED_AFTER_ACCEPTANCE_PATHS",
    "ALLOWED_POST_IMPLEMENTATION_PATHS",
    "AMENDMENT_ID",
    "AMENDMENT_RELATIVE_PATH",
    "CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH",
    "CANDIDATE_E_SCIENTIFIC_CONFIG_SCHEMA",
    "CANDIDATE_E_SCIENTIFIC_CONFIG_SHA256",
    "CANDIDATE_E_SCIENTIFIC_LOCATOR_PATHS",
    "EXECUTION_CLASS_AMENDMENT_ID",
    "ExecutionClassAmendmentBinding",
    "ProtocolAmendmentBinding",
    "ProtocolAmendmentError",
    "SUCCESSOR_AMENDMENT_RELATIVE_PATH",
    "load_protocol_amendment_bytes",
    "candidate_e_scientific_config_payload",
    "candidate_e_scientific_config_sha256",
    "load_execution_class_amendment_bytes",
    "resolve_accepted_protocol_amendment",
    "resolve_accepted_execution_class_amendment",
    "validate_accepted_lineage_commit",
    "validate_review_transition",
    "validate_execution_class_g0_config",
    "validate_execution_class_lineage_commit",
    "validate_execution_class_review_transition",
    "validate_two_gpu_g0_config",
]
