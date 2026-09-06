"""Fail-closed reviewed protocol amendments with non-self-referential Git binding."""

from __future__ import annotations

import copy
import hashlib
import re
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


AMENDMENT_RELATIVE_PATH = Path("prereg/amendments/qwen3_v2_g0_elastic_v1.yaml")
BASE_PREREG_RELATIVE_PATH = Path("prereg/qwen3_v2.yaml")
AMENDMENT_ID = "qwen3_v2_g0_elastic_v1"
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


def validate_elastic_g0_config(config: dict[str, Any], amendment: dict[str, Any]) -> None:
    """Require an allocation-neutral G0 config to implement the amendment."""

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
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ProtocolAmendmentError("protocol amendment must be one non-linked regular file")
    raw = path.read_bytes()
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
    if hashlib.sha256(base_prereg.read_bytes()).hexdigest() != BASE_PREREG_SHA256:
        raise ProtocolAmendmentError("base preregistration bytes changed")
    if _git(code_root, "rev-parse", "HEAD") != current_commit:
        raise ProtocolAmendmentError("amendment validation HEAD changed before completion")
    final_metadata = path.lstat()
    if not stat.S_ISREG(final_metadata.st_mode) or final_metadata.st_nlink != 1:
        raise ProtocolAmendmentError("protocol amendment changed file identity during validation")
    if path.read_bytes() != raw:
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


__all__ = [
    "ALLOWED_AFTER_ACCEPTANCE_PATHS",
    "ALLOWED_POST_IMPLEMENTATION_PATHS",
    "AMENDMENT_ID",
    "AMENDMENT_RELATIVE_PATH",
    "ProtocolAmendmentBinding",
    "ProtocolAmendmentError",
    "load_protocol_amendment_bytes",
    "resolve_accepted_protocol_amendment",
    "validate_accepted_lineage_commit",
    "validate_review_transition",
    "validate_two_gpu_g0_config",
]
