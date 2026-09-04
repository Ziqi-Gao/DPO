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


AMENDMENT_RELATIVE_PATH = Path("prereg/amendments/qwen3_v2_g0_2gpu_v1.yaml")
BASE_PREREG_RELATIVE_PATH = Path("prereg/qwen3_v2.yaml")
AMENDMENT_ID = "qwen3_v2_g0_2gpu_v1"
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


class ProtocolAmendmentError(ValueError):
    """A protocol amendment or its Git review provenance is invalid."""


@dataclass(frozen=True)
class ProtocolAmendmentBinding:
    path: Path
    amendment_id: str
    sha256: str
    git_commit: str
    reviewed_implementation_commit: str


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
    """Parse and schema-validate the fixed two-GPU G0 amendment bytes."""

    payload = _load_yaml(raw, context="Qwen3-v2 two-GPU G0 amendment")
    expected_top = {
        "schema_version",
        "amendment_id",
        "base_preregistration",
        "scope",
        "resource_amendment",
        "batch_token_invariants",
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

    expected_scope = {
        "task": "qwen3_v2_g0",
        "workflow_id": "qwen3-v2-g0-v1",
        "unit_id": "g0",
        "execution_profile": "qwen3-v2-g0-2gpu",
        "seed": 42,
        "claim_scope": (
            "full_mechanism_pipeline_feasibility_only_not_confirmatory_primary_endpoint"
        ),
    }
    if payload["scope"] != expected_scope:
        raise ProtocolAmendmentError("protocol amendment scope differs from the reviewed design")

    expected_resource = {
        "original_world_size": 4,
        "amended_world_size": 2,
        "host_memory_gib": 192,
        "minimum_cgroup_headroom_gib": 32,
        "minimum_cgroup_headroom_fraction": 0.20,
        "exclusive_gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "preflight_required": (
            "successful_accepted_implementation_lineage_two_gpu_preflight"
        ),
    }
    if payload["resource_amendment"] != expected_resource:
        raise ProtocolAmendmentError("protocol amendment resource terms differ")

    expected_batch = {
        "per_device_batch_size": 4,
        "original_gradient_accumulation_steps": 4,
        "amended_gradient_accumulation_steps": 8,
        "effective_global_batch_size": 64,
        "effective_global_batch_formula": (
            "amended_world_size_x_per_device_batch_size_x_"
            "amended_gradient_accumulation_steps"
        ),
        "token_budget": 2_000_000,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "token_accounting": "exact_cross_rank_sum_reserved_before_each_optimizer_boundary",
        "token_stop_boundary": "stop_before_an_optimizer_update_that_would_exceed_budget",
        "max_optimizer_steps": 120,
        "max_steps_role": "independent_safety_ceiling_first_limit_reached_stops_training",
        "resume_rule": "consumed_tokens_and_world_size_are_checkpointed_and_must_match",
    }
    if payload["batch_token_invariants"] != expected_batch:
        raise ProtocolAmendmentError("protocol amendment batch/token terms differ")
    if (
        expected_resource["amended_world_size"]
        * expected_batch["per_device_batch_size"]
        * expected_batch["amended_gradient_accumulation_steps"]
        != expected_batch["effective_global_batch_size"]
    ):
        raise ProtocolAmendmentError("protocol amendment global-batch formula is inconsistent")

    expected_scientific = {
        "unchanged": [
            "model_teacher_and_tokenizer_revisions",
            "prompt_and_sampling_protocols",
            "seed_42",
            "dataset_and_prerequisite_hash_bindings",
            "optimizer_and_learning_rate",
            "evaluation_and_checkpoint_cadence_in_optimizer_steps",
            "G0_gate_and_claim_scope",
            "artifact_and_completion_semantics",
        ],
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


def validate_two_gpu_g0_config(config: dict[str, Any], amendment: dict[str, Any]) -> None:
    """Require the resolved G0 config to implement the amendment exactly."""

    load_protocol_amendment_bytes(yaml.safe_dump(amendment, sort_keys=False).encode("utf-8"))
    trainer = config.get("trainer")
    scheduler = config.get("scheduler_g0")
    if not isinstance(trainer, dict) or not isinstance(scheduler, dict):
        raise ProtocolAmendmentError("two-GPU G0 config lacks trainer or scheduler binding")
    expected = amendment["batch_token_invariants"]
    integer_values = (
        scheduler.get("process_count"),
        trainer.get("batch_size"),
        trainer.get("gradient_accumulation_steps"),
        trainer.get("token_budget"),
        trainer.get("max_steps"),
    )
    if any(type(value) is not int for value in integer_values):
        raise ProtocolAmendmentError("two-GPU G0 batch/token values must be integers")
    observed = {
        "world_size": scheduler.get("process_count"),
        "per_device_batch_size": trainer.get("batch_size"),
        "gradient_accumulation_steps": trainer.get("gradient_accumulation_steps"),
        "effective_global_batch_size": (
            scheduler["process_count"]
            * trainer["batch_size"]
            * trainer["gradient_accumulation_steps"]
        ),
        "token_budget": trainer.get("token_budget"),
        "token_budget_unit": trainer.get("token_budget_unit"),
        "max_optimizer_steps": trainer.get("max_steps"),
    }
    required = {
        "world_size": amendment["resource_amendment"]["amended_world_size"],
        "per_device_batch_size": expected["per_device_batch_size"],
        "gradient_accumulation_steps": expected["amended_gradient_accumulation_steps"],
        "effective_global_batch_size": expected["effective_global_batch_size"],
        "token_budget": expected["token_budget"],
        "token_budget_unit": expected["token_budget_unit"],
        "max_optimizer_steps": expected["max_optimizer_steps"],
    }
    if observed != required:
        raise ProtocolAmendmentError(
            f"two-GPU G0 config violates batch/token invariants: {observed!r}"
        )
    if (
        config.get("protocol_amendment_path") != str(AMENDMENT_RELATIVE_PATH)
        or config.get("protocol_track") != BASE_PREREG_VERSION
        or scheduler.get("execution_profile") != "qwen3-v2-g0-2gpu"
    ):
        raise ProtocolAmendmentError("two-GPU G0 config lacks the fixed amendment identity")


def _git(code_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("/usr/bin/git", "-C", str(code_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProtocolAmendmentError(
            f"protocol amendment Git validation failed: git {' '.join(arguments)}"
        ) from error
    return result.stdout.strip()


def _is_ancestor(code_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        (
            "/usr/bin/git",
            "-C",
            str(code_root),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ),
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise ProtocolAmendmentError("protocol amendment ancestry validation failed")
    return result.returncode == 0


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
    if _git(code_root, "status", "--porcelain", "--untracked-files=no"):
        raise ProtocolAmendmentError("accepted amendment requires a clean tracked checkout")
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
    changed_paths = tuple(
        row
        for row in _git(
            code_root,
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{implementation_commit}..{current_commit}",
            "--",
        ).splitlines()
        if row
    )
    validate_review_transition(
        proposed=proposed,
        accepted=accepted,
        implementation_commit=implementation_commit,
        current_commit=current_commit,
        changed_paths=changed_paths,
        implementation_is_ancestor=_is_ancestor(
            code_root,
            implementation_commit,
            current_commit,
        ),
    )
    base_prereg = code_root / BASE_PREREG_RELATIVE_PATH
    if hashlib.sha256(base_prereg.read_bytes()).hexdigest() != BASE_PREREG_SHA256:
        raise ProtocolAmendmentError("base preregistration bytes changed")
    amendment_commit = _git(
        code_root,
        "log",
        "-n",
        "1",
        "--format=%H",
        "--",
        str(AMENDMENT_RELATIVE_PATH),
    )
    if GIT_COMMIT.fullmatch(amendment_commit) is None:
        raise ProtocolAmendmentError("accepted amendment has no committed Git identity")
    return ProtocolAmendmentBinding(
        path=path,
        amendment_id=AMENDMENT_ID,
        sha256=hashlib.sha256(raw).hexdigest(),
        git_commit=amendment_commit,
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
    implementation_to_candidate = tuple(
        row
        for row in _git(
            code_root,
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{implementation_commit}..{candidate_commit}",
            "--",
        ).splitlines()
        if row
    )
    validate_review_transition(
        proposed=proposed,
        accepted=candidate,
        implementation_commit=implementation_commit,
        current_commit=candidate_commit,
        changed_paths=implementation_to_candidate,
        implementation_is_ancestor=True,
    )
    candidate_to_current = {
        row
        for row in _git(
            code_root,
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{candidate_commit}..{current_commit}",
            "--",
        ).splitlines()
        if row
    }
    if not candidate_to_current <= set(ALLOWED_AFTER_ACCEPTANCE_PATHS):
        raise ProtocolAmendmentError(
            f"{role} lineage contains post-acceptance implementation changes"
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
