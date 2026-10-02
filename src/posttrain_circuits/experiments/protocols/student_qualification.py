"""Qualification of the frozen prepared student; preserves original scientific gates."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.git_provenance import _git_environment
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow
from posttrain_circuits.artifacts.teacher_adaptation_protocol import BASE_PREREG_SHA256, REVIEW_PROPOSED
from posttrain_circuits.experiments.protocols.student_preparation import (
    SCIENCE_PATHS as PREPARATION_SCIENCE_PATHS,
)

PROTOCOL_PATH = "prereg/amendments/qwen3_student_qualification_v1.json"
PROTOCOL_ID = "qwen3-student-qualification-v1"
PARENT_HEAD = "ba8d357c61bcfca5bdc055030eed5d73e7b31f16"
PARENT_PROTOCOL_SHA256 = "c66d53c49d79f7bf5f63f7d933dff33bca124ebabdafb1b821ddd94dc1eb970f"
FIT_JOB_ID = "54562507"
FIT_INTENT = "524f6a6e8f1541502f315b820880ea4c"
FIT_PLAN_SHA256 = "3e858360407c78b3e2563959eb84dfc7961ff084366355a45f81ed3d85881b6d"
FIT_PUBLICATION_SHA256 = "98aff58e27bc88653453047d7610d5c1b2b648482b1ad72616f3628c8613b1bc"
FIT_REPORT_SHA256 = "1653f312b0732512fe3710b9424bdf9390646d589fa9dc2b921276f6e75962b3"
FIT_AUDIT_SHA256 = "411fd1f2bb4c0de4fcf30f37c3bc701b25c2edad38da651182bc0012a014cdaf"
CHECKPOINT_SHA256 = "23854ce0d6db4cb01cae898beccf1ac7be7613e5ee01ff5b1626daafb8349d31"
CHECKPOINT_SIZE = 8127108889
CHECKPOINT_STEP = 4
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)
FROZEN_SCIENCE_PATHS = tuple(
    dict.fromkeys(
        (
            *PREPARATION_SCIENCE_PATHS,
            "src/posttrain_circuits/datasets/proofgraph/anti_shortcut.py",
            "src/posttrain_circuits/cli/evaluate_anti_shortcut.py",
            "src/posttrain_circuits/cli/score_probe_candidates.py",
            "configs/anti_shortcut/proofgraph.yaml",
            "tools/sdsc_student_initial_probe.py",
            "tools/sdsc_student_quality_job.py",
        )
    )
)
SCIENCE_PATHS = (
    *FROZEN_SCIENCE_PATHS,
    "src/posttrain_circuits/experiments/protocols/student_qualification.py",
    "tools/sdsc_student_qualify.py",
    "tools/sdsc_student_qualify_job.py",
    "tools/sdsc_student_qualify_worker.py",
    "tools/sdsc_student_qualify_audit.py",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_MAX_BYTES = 4 * 1024 * 1024


class StudentQualificationError(ValueError):
    """The frozen qualification contract was not satisfied."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentQualificationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def proposed_student_qualification_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "opd_student_qualification_protocol",
        "protocol_id": PROTOCOL_ID,
        "scope": {
            "purpose": "fixed_prepared_initial_base_and_anti_shortcut_qualification",
            "execution": "sdsc_h100_single_gpu_inference_only",
            "changes_to_training_or_formal_thresholds": False,
            "producer_may_accept_complete_initial_or_g0": False,
            "full_factorial_or_replication_in_scope": False,
            "prior_validation128_diagnostic_exposure_disclosed": True,
            "original_failed_initial_and_all_evidence_preserved": True,
        },
        "preserved_base": {
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_sha256": BASE_PREREG_SHA256,
            "parent_science_head": PARENT_HEAD,
            "parent_student_protocol": "prereg/amendments/qwen3_student_preparation_v1.json",
            "parent_student_protocol_sha256": PARENT_PROTOCOL_SHA256,
            "all_original_numerical_gates_unchanged": True,
        },
        "prepared_initial": {
            "fit_job_id": FIT_JOB_ID,
            "fit_intent": FIT_INTENT,
            "fit_plan_sha256": FIT_PLAN_SHA256,
            "fit_publication_sha256": FIT_PUBLICATION_SHA256,
            "fit_report_sha256": FIT_REPORT_SHA256,
            "independent_raw_audit_sha256": FIT_AUDIT_SHA256,
            "checkpoint_step": CHECKPOINT_STEP,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_size": CHECKPOINT_SIZE,
            "selection": "frozen_earliest_development_candidate_before_formal_exposure",
            "checkpoint_reselection_or_retraining": False,
            "load": "exact_FP32_saved_masters_before_one_native_BF16_inference_copy",
            "identity": "dense_checkpoint_bytes_never_native_HF_revision",
        },
        "base": {
            "source": "original_family_validation_first128_in_manifest_order",
            "num_examples": 128,
            "metric": "original_VerificationResult.answer_correct",
            "minimum_answer_correct_count": 13,
            "scorer": "posttrain_circuits.cli.score_probe_candidates._score_examples",
            "validation_file_sha256": "8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3",
            "examples_sha256": "ceecd7470309dbb78351f8ecba850b98b6f35bf749aa89ac101498ba4ce921e8",
            "max_model_input_length": 1536,
        },
        "anti_shortcut": {
            "source": "original_build_split_iid_test_128_seed42_then_original_suite",
            "iid_examples": 128,
            "transformed_examples": 640,
            "seed": 42,
            "iid_examples_sha256": "ebb0fc1c7ef05334632f5e7bdf8b798d3e9f0b253357b37f22f5e4b1456852f3",
            "suite_sha256": "c5ed272d5fe8f6dd21a7743dfca44d3bdfa06db44499842b7cc44df807b8ced0",
            "distractor_ood_count": 32,
            "transformations": [
                "entity_symbol_renaming",
                "fact_order_permutation",
                "rule_order_permutation",
                "surface_template_paraphrase",
                "distractor_count_ood",
            ],
            "metric": "original_exact_verifier_reward",
            "minimum_iid_accuracy": 0.10,
            "minimum_transformed_accuracy": 0.08,
            "minimum_per_transformation_accuracy": 0.05,
            "max_shortcut_gap": 0.05,
            "shortcut_gap": "iid_accuracy_minus_equal_weight_mean_of_five_transformed_accuracies",
            "wilson_lower_bound_is_gate": False,
            "sequence_envelope": (
                "original_untruncated_anti_shortcut_context_not_training1536_or_preparation1600"
            ),
            "context_length_checked_against_original_model_capacity_before_inference": True,
            "max_model_input_length": 2244,
            "actual_max_prefix_length": 1988,
        },
        "generation": {
            "max_new_tokens": 256,
            "do_sample": False,
            "use_cache": False,
            "explicit_attention_mask": False,
            "explicit_autocast": False,
            "precision": "native_bfloat16",
            "truncation": False,
            "seed": 42,
            "all_896_responses_required_even_if_one_gate_fails": True,
            "original_tokenizer_template_render_parser_verifier_unchanged": True,
        },
        "evidence": {
            "raw_prompt_response_tokens_and_complete_parser_verifier_traces": True,
            "independent_full_response_replay_required": True,
            "qualification_passed": "base_gate_and_anti_shortcut_gate_after_complete_896",
            "formal_initial_g0_pilot_flags": False,
            "failure": "preserve_complete_evidence_stop_progression_no_reselection_or_threshold_change",
            "gpu_arithmetic_independent_replay_claim": False,
        },
        "execution": {
            "gpu_count": 1,
            "cpu_count": 24,
            "host_memory_gib": 192,
            "walltime_minutes": 120,
            "minimum_headroom_gib": 32,
            "minimum_headroom_fraction": 0.20,
            "maximum_concurrently_allocatable_gpus": 4,
            "blackwell_execution_class_reused": False,
        },
        "science_files": list(SCIENCE_PATHS),
        "review_contract": {
            "mechanism": "implementation_commit_then_distinct_review_only_acceptance_commit",
            "acceptance_allowed_changed_paths": [PROTOCOL_PATH, "docs/refactor/current_handoff.md"],
            "current_named_science_bytes_must_equal_implementation": True,
            "later_unrelated_commits_invalidate_review": False,
        },
        "review": copy.deepcopy(REVIEW_PROPOSED),
    }


def validate_student_qualification_protocol(
    payload: Any, *, require_accepted: bool = False
) -> dict[str, Any]:
    expected = proposed_student_qualification_protocol()
    _require(isinstance(payload, dict) and set(payload) == set(expected), "protocol fields differ")
    review = payload["review"]
    _require(isinstance(review, dict) and set(review) == set(REVIEW_PROPOSED), "review fields differ")
    if review["status"] == "proposed":
        _require(review == REVIEW_PROPOSED and not require_accepted, "qualification protocol is not accepted")
    else:
        _require(review["status"] == "accepted", "unknown review status")
        commit = review["reviewed_implementation_commit"]
        _require(
            isinstance(commit, str) and _COMMIT.fullmatch(commit), "review needs full implementation commit"
        )
        for key in ("reviewer", "rationale"):
            _require(isinstance(review[key], str) and review[key].strip(), f"review requires {key}")
        stamp = review["reviewed_at_utc"]
        _require(isinstance(stamp, str) and _TIMESTAMP.fullmatch(stamp), "review requires UTC timestamp")
        try:
            datetime.fromisoformat(stamp[:-1] + "+00:00")
        except ValueError as error:
            raise StudentQualificationError("invalid review date") from error
    expected["review"] = copy.deepcopy(review)
    try:
        _require(_canonical(payload) == _canonical(expected), "qualification protocol differs from frozen v1")
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentQualificationError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_qualification_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
    _require(isinstance(raw, bytes) and 0 < len(raw) <= _MAX_BYTES, "invalid protocol byte bound")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate protocol JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise StudentQualificationError("invalid protocol JSON") from error
    return validate_student_qualification_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_qualification_protocol(payload)
    checked.pop("review")
    return hashlib.sha256(_canonical(checked)).hexdigest()


@dataclass(frozen=True)
class ResolvedStudentQualificationProtocol:
    payload: dict[str, Any]
    protocol_sha256: str
    artifact_sha256: str
    implementation_commit: str | None
    acceptance_commit: str | None
    head: str
    science_file_sha256: dict[str, str]
    tracked_worktree_clean: bool

    @property
    def science_implementation_sha256(self) -> str:
        return sha256_value(self.science_file_sha256)

    @property
    def review_status(self) -> str:
        return self.payload["review"]["status"]


def resolve_student_qualification_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentQualificationProtocol:
    root = Path(repo_root).absolute()
    _require(root.is_dir() and root.resolve() == root, "source root must be a real directory")
    metadata = Path(git_dir).absolute() if git_dir is not None else root / ".git"
    _require(
        metadata in (root / ".git", root / ".opd-git") and metadata.is_dir() and not metadata.is_symlink(),
        "qualification needs real .git or .opd-git metadata",
    )

    def git(*args: str) -> bytes:
        result = subprocess.run(
            (
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(metadata),
                "--work-tree",
                str(root),
                "-C",
                str(root),
                *args,
            ),
            capture_output=True,
            env=_git_environment(),
            timeout=30,
            check=False,
        )
        _require(result.returncode == 0, f"real Git provenance query failed: {args[0]}")
        return result.stdout

    def text(*args: str) -> str:
        return git(*args).decode("utf-8", errors="strict").strip()

    def read(path: str) -> bytes:
        return read_regular_bytes_nofollow(
            root / path, context=f"student qualification source {path}", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    _require(_COMMIT.fullmatch(head) and (expected_head is None or head == expected_head), "Git HEAD differs")
    _require(text("rev-parse", "--is-shallow-repository") == "false", "complete Git review history required")
    raw = read(PROTOCOL_PATH)
    payload = load_student_qualification_protocol(raw, require_accepted=require_accepted)
    _require(
        hashlib.sha256(read("prereg/qwen3_v2.yaml")).hexdigest() == BASE_PREREG_SHA256,
        "original frozen preregistration changed",
    )
    hashes = {path: hashlib.sha256(read(path)).hexdigest() for path in SCIENCE_PATHS}
    implementation, acceptance = payload["review"]["reviewed_implementation_commit"], None
    if payload["review"]["status"] == "accepted":
        _require(
            implementation != head and text("merge-base", implementation, head) == implementation,
            "review implementation is not a distinct ancestor",
        )
        proposed = load_student_qualification_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
        _require(proposed["review"] == REVIEW_PROPOSED, "implementation protocol must be proposed")
        touching = text(
            "rev-list", "--reverse", "--ancestry-path", f"{implementation}..{head}", "--", PROTOCOL_PATH
        ).splitlines()
        _require(
            len(touching) == 1 and _COMMIT.fullmatch(touching[0]), "exactly one later review change required"
        )
        acceptance = touching[0]
        _require(
            len(text("rev-list", "--parents", "-n", "1", acceptance).split()) == 2,
            "review-only acceptance must have one parent",
        )
        changed = {
            path.decode()
            for path in git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", acceptance).split(b"\0")
            if path
        }
        _require(
            PROTOCOL_PATH in changed and changed <= {PROTOCOL_PATH, "docs/refactor/current_handoff.md"},
            "acceptance contains non-review changes",
        )
        reviewed_raw = git("show", f"{acceptance}:{PROTOCOL_PATH}")
        reviewed = load_student_qualification_protocol(reviewed_raw, require_accepted=True)
        _require(
            reviewed["review"]["reviewed_implementation_commit"] == implementation,
            "acceptance reviews another implementation",
        )
        reviewed["review"] = copy.deepcopy(REVIEW_PROPOSED)
        _require(_canonical(reviewed) == _canonical(proposed), "acceptance changes scientific protocol")
        _require(
            raw == reviewed_raw == git("show", f"{head}:{PROTOCOL_PATH}"),
            "current protocol differs from accepted bytes",
        )
        for revision in (implementation, head):
            _require(
                hashlib.sha256(git("show", f"{revision}:prereg/qwen3_v2.yaml")).hexdigest()
                == BASE_PREREG_SHA256,
                "committed original preregistration changed",
            )
        _require(
            not git(
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--",
                PROTOCOL_PATH,
                "prereg/qwen3_v2.yaml",
                *SCIENCE_PATHS,
            ),
            "named scientific files have staged changes",
        )
        for path, current_sha in hashes.items():
            entry = text("ls-tree", implementation, "--", path).split(maxsplit=1)
            _require(
                entry and entry[0] in {"100644", "100755"}, f"science path is not a regular blob: {path}"
            )
            blob = git("show", f"{implementation}:{path}")
            _require(
                hashlib.sha256(blob).hexdigest() == current_sha and git("show", f"{head}:{path}") == blob,
                f"named science differs from accepted implementation: {path}",
            )
        parent = payload["preserved_base"]
        _require(
            text("merge-base", PARENT_HEAD, implementation) == PARENT_HEAD, "original student lineage missing"
        )
        for path in FROZEN_SCIENCE_PATHS:
            _require(
                git("show", f"{PARENT_HEAD}:{path}") == read(path),
                f"frozen parent student science changed: {path}",
            )
        _require(
            hashlib.sha256(read(parent["parent_student_protocol"])).hexdigest()
            == parent["parent_student_protocol_sha256"],
            "frozen parent student protocol changed",
        )
    clean = not git("diff", "--name-only", "HEAD", "--")
    _require(
        text("rev-parse", "--verify", "HEAD^{commit}") == head and read(PROTOCOL_PATH) == raw,
        "Git HEAD or protocol changed during validation",
    )
    for path, expected_sha in hashes.items():
        _require(
            hashlib.sha256(read(path)).hexdigest() == expected_sha,
            f"source changed during validation: {path}",
        )
    return ResolvedStudentQualificationProtocol(
        payload,
        protocol_core_sha256(payload),
        hashlib.sha256(raw).hexdigest(),
        implementation,
        acceptance,
        head,
        hashes,
        clean,
    )
