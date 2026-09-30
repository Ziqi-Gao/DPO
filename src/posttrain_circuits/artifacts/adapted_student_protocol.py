"""Reviewed, bounded canonical-SFT successor consuming one accepted dense teacher.

This is separate from teacher qualification, the historical G0 amendment and any
execution-class certificate. Git history and named source bytes are checked at
use time; no model/framework import, commit creation or job submission occurs.
"""

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
from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow
from posttrain_circuits.artifacts.teacher_adaptation_protocol import (
    BASE_PREREG_SHA256,
    CHAT_TEMPLATE_SHA256,
    READINESS_THRESHOLDS,
    REVIEW_PROPOSED,
    TOKENIZER_FINGERPRINT,
    resolve_teacher_adaptation_protocol,
)

PROTOCOL_PATH = "prereg/amendments/qwen3_adapted_student_calibration_v6.json"
PROTOCOL_ID = "qwen3-adapted-student-calibration-v6"
ADAPTED_TEACHER_CONFIG_PATH = "configs/adapted_teacher/qwen3_accepted_student_v6.yaml"
ACCELERATE_CONFIG_PATH = "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml"
PREDECESSOR = {
    "protocol_path": "prereg/amendments/qwen3_adapted_student_calibration_v5.json",
    "protocol_id": "qwen3-adapted-student-calibration-v5",
    "implementation_commit": "529d46eceab6fca7bf9bb73fc79e82101fbd0fd1",
    "acceptance_commit": "9169a61e452c0b94064b06e6a449abfff63857f1",
    "protocol_sha256": "14c79d22ae6016140591ee15dd9bb84cafcab53bd86619350c874bb0b1b964e2",
    "artifact_sha256": "c3e2f33408db18be07d6ec855752ea1c12461958528ac5380ee7b1303dd620e3",
}
PRESERVED_PROTOCOLS = (
    {
        "protocol_path": "prereg/amendments/qwen3_adapted_student_calibration_v1.json",
        "protocol_id": "qwen3-adapted-student-calibration-v1",
        "implementation_commit": "9129f32ad2f53b8fa9073b417fd615560278a2db",
        "acceptance_commit": "77603c14802c21e0dce09fe7617705d33425ce94",
        "protocol_sha256": "1c29f633e542ba4de72d25314f68d6508fc0919a6c7335d65bfef72513c62618",
        "artifact_sha256": "677d49599e434b060eebed5a58b2130afbe014be06363e73dffc1ab31f0b07d6",
    },
    {
        "protocol_path": "prereg/amendments/qwen3_adapted_student_calibration_v2.json",
        "protocol_id": "qwen3-adapted-student-calibration-v2",
        "implementation_commit": "3c1f6f1e9bdc798ecb65399165dd1be313014b42",
        "acceptance_commit": "28c1026cece772a9e3d167d9cc64a64aaa0fd1b3",
        "protocol_sha256": "9277960e4ff3599c325ac0115888280ad32647891fd3841d045822bf7db2a320",
        "artifact_sha256": "2c15821442d3ae51da5d86917a5fbf160aac9a1bb1b6ef5e8944dab8f8c69274",
    },
    {
        "protocol_path": "prereg/amendments/qwen3_adapted_student_calibration_v3.json",
        "protocol_id": "qwen3-adapted-student-calibration-v3",
        "implementation_commit": "6ae57f0ef0a4cd594f1692ed1550c826dc9b37ee",
        "acceptance_commit": "df05bd2a96363829a4bc587c04a2dd116e2a6242",
        "protocol_sha256": "c727957ed2fa059481772ebbdef68982a0280436d41cb6be4acd5eff1fd70ec9",
        "artifact_sha256": "0a64f606455830786639e019b736d108d072f02f4d20cf869061ff6059df8d53",
    },
    {
        "protocol_path": "prereg/amendments/qwen3_adapted_student_calibration_v4.json",
        "protocol_id": "qwen3-adapted-student-calibration-v4",
        "implementation_commit": "89a8ffd598d9377ebcbef556bee0057699d9eb35",
        "acceptance_commit": "cfac02db4cf67c7d1a8c1b09697fcc25de76474b",
        "protocol_sha256": "822b5640da8e47232ce09795e364c73fae16512f1123d2a0b4dfa56c1769f93c",
        "artifact_sha256": "61b52355f3ac4c91f6de819c0e0a2711ff9c4069ad9c8313dd080f07df3cc8cf",
    },
)
PRODUCER_HEAD = "929fb14834852a7c91e6656c76fd1834e1b5007d"
PRODUCER_IMPLEMENTATION = "d1ab8dacd834101b88d806bb6d75a44ae1949cb3"
PRODUCER_PROTOCOL_SHA256 = "dcd5fca7c87bda603f930e1e053d34073ca12aa079fa884c0e4e87e32ec099f4"
ACCEPTANCE_SHA256 = "5d6952823441bde567cdf7f5fad8b4625c58ee7e82425aad76c10433d0ec5337"
ACCEPTANCE_INVENTORY_SHA256 = "8d53783b9fb1d386de5a0a291c2b28e225347168cc7cfed4c0c0aceaf01d9bed"
DENSE_TEACHER_SHA256 = "6928f2537dcca5f2d65c1498659e1ebf011845eb72ef364b9544036c2238e9c7"
BASE_TEACHER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
STUDENT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# Only the added consumer and its used execution/validation dependencies. The
# producer resolver separately preserves all 47 existing teacher science files.
SCIENCE_PATHS = (
    ADAPTED_TEACHER_CONFIG_PATH,
    "configs/model/qwen3_v2_1p7b.yaml",
    "configs/state_source/teacher_demo.yaml",
    "configs/supervision/canonical_sft.yaml",
    ACCELERATE_CONFIG_PATH,
    "src/posttrain_circuits/artifacts/adapted_student_protocol.py",
    "src/posttrain_circuits/artifacts/adapted_teacher_sft.py",
    "src/posttrain_circuits/artifacts/runs.py",
    "src/posttrain_circuits/artifacts/checkpoints.py",
    "src/posttrain_circuits/artifacts/config_bindings.py",
    "src/posttrain_circuits/artifacts/completion.py",
    "src/posttrain_circuits/experiments/protocols/specs.py",
    "src/posttrain_circuits/cli/train.py",
    "src/posttrain_circuits/cli/export_initial_checkpoint.py",
    "src/posttrain_circuits/causal_circuits/model/runner.py",
    "src/posttrain_circuits/cli/_common.py",
    "src/posttrain_circuits/cli/factorial_run_validation.py",
    "src/posttrain_circuits/cli/finalize_pilot_training.py",
    "src/posttrain_circuits/core/seeding.py",
    "src/posttrain_circuits/core/readiness.py",
    "src/posttrain_circuits/datasets/proofgraph/family.py",
    "src/posttrain_circuits/datasets/proofgraph/metrics.py",
    "src/posttrain_circuits/datasets/teacher_demos/store.py",
    "src/posttrain_circuits/datasets/trajectories/contracts.py",
    "src/posttrain_circuits/datasets/trajectories/identities.py",
    "src/posttrain_circuits/learning/contracts.py",
    "src/posttrain_circuits/learning/collation.py",
    "src/posttrain_circuits/learning/primitives.py",
    "src/posttrain_circuits/learning/supervision/base.py",
    "src/posttrain_circuits/learning/supervision/verified_replay.py",
    "src/posttrain_circuits/learning/teacher/demo_source.py",
    "src/posttrain_circuits/learning/training/canonical_sft.py",
    "src/posttrain_circuits/learning/training/factorial_trainer.py",
    "src/posttrain_circuits/learning/training/factories.py",
    "src/posttrain_circuits/learning/training/evaluation.py",
    "src/posttrain_circuits/learning/training/optimizer.py",
    "src/posttrain_circuits/learning/training/schedules.py",
    "src/posttrain_circuits/learning/training/token_budget.py",
    "src/posttrain_circuits/learning/training/fsdp_contract.py",
    "src/posttrain_circuits/learning/training/execution_safety_kernel.py",
    "src/posttrain_circuits/methods/sft.py",
    "src/posttrain_circuits/methods/specs.py",
    "src/posttrain_circuits/methods/registry.py",
    "tools/sdsc_student_contract.py",
    "tools/sdsc_student_job.py",
    "tools/sdsc_student_job.sh",
    "tools/sdsc_adapted_training_preflight.py",
    "tools/sdsc_adapted_calibration.py",
    "tools/sdsc_student_memory.py",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_MAX_BYTES = 2 * 1024 * 1024


class AdaptedStudentProtocolError(ValueError):
    """The proposed consumer or its actual source authority differs from v6."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def adapted_teacher_config() -> dict[str, Any]:
    """Fresh exact config identity; weight and store locations are not identities."""
    return {
        "teacher_identity": {
            "format_version": 1,
            "kind": "learned_dense_checkpoint",
            "base_model_id": "Qwen/Qwen3-8B",
            "base_revision": BASE_TEACHER_REVISION,
            "tokenizer_id": "Qwen/Qwen3-8B",
            "tokenizer_revision": BASE_TEACHER_REVISION,
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "teacher_checkpoint_sha256": DENSE_TEACHER_SHA256,
        },
        "acceptance_sha256": ACCEPTANCE_SHA256,
        "acceptance_inventory_sha256": ACCEPTANCE_INVENTORY_SHA256,
        "producer_science_head": PRODUCER_HEAD,
        "qualification_job_id": "54496291",
        "student_protocol_path": PROTOCOL_PATH,
    }


def proposed_adapted_student_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "opd_adapted_teacher_student_calibration_protocol",
        "protocol_id": PROTOCOL_ID,
        "predecessor": copy.deepcopy(PREDECESSOR),
        "preserved_protocols": copy.deepcopy(list(PRESERVED_PROTOCOLS)),
        "repair": {
            "kind": "larger_host_memory_preserving_aggregate_peak_headroom",
            "scientific_settings_and_teacher_evidence_unchanged": True,
            "reuse_predecessor_gpu_preflight": False,
        },
        "scope": {
            "method": "canonical_sft",
            "execution": "sdsc_h100",
            "calibration_only": True,
            "g0_pass_claim": False,
            "pilot_pass_claim": False,
            "execution_class_certified": False,
            "blackwell_execution_class_reused": False,
            "original_128_token_teacher_readiness_pass_claim": False,
        },
        "accepted_teacher": adapted_teacher_config(),
        "producer": {
            "implementation_commit": PRODUCER_IMPLEMENTATION,
            "acceptance_commit": PRODUCER_HEAD,
            "protocol_sha256": PRODUCER_PROTOCOL_SHA256,
            "preserve_all_47_named_science_files": True,
            "store_sha256": "f2e9e8171e3289bfd8bb356c79df67e56612f2ad067c63a4583b017c79025a88",
            "accepted_candidates": 2048,
            "covered_prompts": 256,
            "generation_max_new_tokens": 256,
            "readiness_max_new_tokens": 256,
            "readiness_thresholds": copy.deepcopy(READINESS_THRESHOLDS),
            "allow_repair_resample_or_truncate_demonstrations": False,
        },
        "student": {
            "model_id": "Qwen/Qwen3-1.7B",
            "revision": STUDENT_REVISION,
            "seed": 42,
            "full_parameter_training": True,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 0.0005,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "schedule": "constant_1.0",
            },
            "state_source": "accepted_teacher_demo_cursor",
            "loss": "mean_of_response_token_means_per_sequence",
            "prompt_and_padding_tokens_masked": True,
            "eos_supervised": True,
            "global_batch_size": 64,
            "max_microbatch_size": 4,
            "max_model_input_length": 1536,
            "max_steps": 120,
            "token_budget": 2000000,
            "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
            "token_budget_reserved_before_backward": True,
            "checkpoint_every": 20,
            "evaluation_every": 20,
            "validation_examples": 128,
            "max_completion_length": 128,
            "same_world_resume_required": True,
            "changed_world_resume_rejected_before_state_load": True,
        },
        "execution": {
            "gpu_type": "h100",
            "gpu_count": 2,
            "cpus": 24,
            "host_memory_gib": 384,
            "host_memory_headroom": {"minimum_gib": 32, "minimum_fraction": 0.2},
            "threads_per_rank": 12,
            "fsdp_requested": "FULL_SHARD",
            "fsdp_effective": "FULL_SHARD",
            "fsdp_use_orig_params": False,
            "accelerate_config": ACCELERATE_CONFIG_PATH,
            "supervision_tensor_device": "actual_model_logits_device",
            "canonical_sft_supervisor_device_preflight_required": True,
            "optimizer_parameter_binding": "prepared_model_parameters",
            "production_optimizer_preparation_preflight_required": True,
            "model_update_comparison": {
                "comparison_policy": "bf16_baseline_fp32_master_v1",
                "fresh_baseline_dtype": "bfloat16",
                "resumed_baseline_dtype": "float32",
                "final_dtype": "float32",
                "value_delta_dtype": "float64",
                "parameter_keys_and_shapes_exact": True,
                "nonfloating_tensors_exact": True,
                "final_tensor_hash_uses_actual_bytes": True,
                "promotion_only_counts_as_update": False,
            },
            "checkpoint_precision_preflight_required": True,
            "fresh_matching_real_gpu_preflight_required": True,
            "actual_node_local_workspace_required": True,
            "verify_node_input_and_persistent_output_mounts": True,
            "persist_and_rehash_results_before_job_exit": True,
        },
        "preserved_base": {
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_sha256": BASE_PREREG_SHA256,
            "teacher_result_acceptance_does_not_accept_student": True,
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


def validate_adapted_student_protocol(payload: Any, *, require_accepted: bool = False) -> dict[str, Any]:
    try:
        expected = proposed_adapted_student_protocol()
        if not isinstance(payload, dict) or set(payload) != set(expected):
            raise AdaptedStudentProtocolError("student protocol fields differ")
        review = payload["review"]
        if not isinstance(review, dict) or set(review) != set(REVIEW_PROPOSED):
            raise AdaptedStudentProtocolError("student review fields differ")
        if review["status"] == "proposed":
            if review != REVIEW_PROPOSED or require_accepted:
                raise AdaptedStudentProtocolError("student protocol is proposed, not accepted")
        elif review["status"] == "accepted":
            if not isinstance(review["reviewed_implementation_commit"], str) or not _COMMIT.fullmatch(
                review["reviewed_implementation_commit"]
            ):
                raise AdaptedStudentProtocolError("student review needs full implementation commit")
            for field in ("reviewer", "rationale"):
                if not isinstance(review[field], str) or not review[field].strip():
                    raise AdaptedStudentProtocolError(f"student review requires {field}")
            timestamp = review["reviewed_at_utc"]
            if not isinstance(timestamp, str) or not _UTC.fullmatch(timestamp):
                raise AdaptedStudentProtocolError("student review requires explicit UTC timestamp")
            datetime.fromisoformat(timestamp[:-1] + "+00:00")
        else:
            raise AdaptedStudentProtocolError("unknown student review status")
        expected["review"] = copy.deepcopy(review)
        if _canonical(payload) != _canonical(expected):
            raise AdaptedStudentProtocolError("student protocol differs from fixed calibration specification")
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        if isinstance(error, AdaptedStudentProtocolError):
            raise
        raise AdaptedStudentProtocolError("invalid student protocol value") from error
    return copy.deepcopy(payload)


def load_adapted_student_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_BYTES:
        raise AdaptedStudentProtocolError("student protocol bytes are empty or oversized")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise AdaptedStudentProtocolError("duplicate JSON protocol key")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise AdaptedStudentProtocolError("invalid or duplicate student protocol JSON") from error
    return validate_adapted_student_protocol(value, require_accepted=require_accepted)


def adapted_student_protocol_sha256(payload: Any) -> str:
    core = validate_adapted_student_protocol(payload)
    core.pop("review")
    return hashlib.sha256(_canonical(core)).hexdigest()


@dataclass(frozen=True)
class AdaptedStudentProtocolBinding:
    amendment_id: str
    path: Path
    git_commit: str | None
    sha256: str
    reviewed_implementation_commit: str | None
    protocol_sha256: str
    payload: dict[str, Any]
    science_file_sha256: dict[str, str]
    head: str


def resolve_adapted_student_protocol(
    repo_root: Path | str,
    *,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
    require_accepted: bool = True,
) -> AdaptedStudentProtocolBinding:
    root = Path(repo_root).absolute()
    if not root.is_dir() or root.resolve() != root:
        raise AdaptedStudentProtocolError("student code root must be a real existing directory")
    metadata = Path(git_dir).absolute() if git_dir is not None else root / ".git"
    if metadata not in (root / ".git", root / ".opd-git") or not metadata.is_dir() or metadata.is_symlink():
        raise AdaptedStudentProtocolError("student protocol needs real .git or .opd-git metadata")

    def git(*arguments):
        result = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "--git-dir",
                str(metadata),
                "--work-tree",
                str(root),
                "-C",
                str(root),
                *arguments,
            ],
            capture_output=True,
            env=_git_environment(),
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise AdaptedStudentProtocolError(f"student Git query failed: {arguments[0]}")
        return result.stdout

    def text(*arguments):
        return git(*arguments).decode("utf-8").strip()

    def read(path):
        return read_regular_bytes_nofollow(
            root / path, context="student protocol source", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    if not _COMMIT.fullmatch(head) or (expected_head is not None and expected_head != head):
        raise AdaptedStudentProtocolError("actual student HEAD differs from expected provenance")
    if text("rev-parse", "--is-shallow-repository") != "false" or (metadata / "info/grafts").exists():
        raise AdaptedStudentProtocolError("student review requires complete ungrafted history")
    raw = read(PROTOCOL_PATH)
    payload = load_adapted_student_protocol(raw, require_accepted=require_accepted)
    # Resolve all historical acceptances through actual immutable Git ancestry.
    # V6's changed memory envelope is separately reviewed; old scientific artifacts
    # remain byte-identical and are never reinterpreted as v6 evidence.
    historical_bytes = {}
    for historical in (*PRESERVED_PROTOCOLS, PREDECESSOR):
        predecessor_path = historical["protocol_path"]
        predecessor_implementation = historical["implementation_commit"]
        predecessor_acceptance = historical["acceptance_commit"]
        if text("merge-base", predecessor_acceptance, head) != predecessor_acceptance or text(
            "rev-list", "--parents", "-n", "1", predecessor_acceptance
        ).split() != [predecessor_acceptance, predecessor_implementation]:
            raise AdaptedStudentProtocolError("student predecessor needs its genuine accepted ancestry")
        predecessor_raw = read(predecessor_path)
        if (
            hashlib.sha256(predecessor_raw).hexdigest() != historical["artifact_sha256"]
            or git("show", f"{predecessor_acceptance}:{predecessor_path}") != predecessor_raw
            or git("show", f"{head}:{predecessor_path}") != predecessor_raw
            or git("diff", "--cached", "--name-only", "-z", "--", predecessor_path)
        ):
            raise AdaptedStudentProtocolError("student predecessor accepted artifact changed")
        predecessor_payload = json.loads(predecessor_raw)
        predecessor_review = predecessor_payload.pop("review", {})
        if (
            predecessor_payload.get("protocol_id") != historical["protocol_id"]
            or hashlib.sha256(_canonical(predecessor_payload)).hexdigest() != historical["protocol_sha256"]
            or predecessor_review.get("status") != "accepted"
            or predecessor_review.get("reviewed_implementation_commit") != predecessor_implementation
        ):
            raise AdaptedStudentProtocolError("student predecessor protocol identity differs")
        historical_bytes[predecessor_path] = predecessor_raw
    producer = resolve_teacher_adaptation_protocol(root, git_dir=metadata, expected_head=head)
    if (
        producer.acceptance_commit != PRODUCER_HEAD
        or producer.implementation_commit != PRODUCER_IMPLEMENTATION
        or producer.protocol_sha256 != PRODUCER_PROTOCOL_SHA256
        or len(producer.science_file_sha256) != 47
    ):
        raise AdaptedStudentProtocolError("teacher producer lineage differs from accepted result origin")
    source_hashes = {path: hashlib.sha256(read(path)).hexdigest() for path in SCIENCE_PATHS}
    implementation = payload["review"]["reviewed_implementation_commit"]
    acceptance = None
    if payload["review"]["status"] == "accepted":
        if implementation == head or text("merge-base", implementation, head) != implementation:
            raise AdaptedStudentProtocolError("student implementation must be a distinct ancestor")
        proposed = load_adapted_student_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
        if proposed["review"] != REVIEW_PROPOSED:
            raise AdaptedStudentProtocolError("student implementation did not contain proposed review")
        touching = text(
            "rev-list", "--reverse", "--ancestry-path", f"{implementation}..{head}", "--", PROTOCOL_PATH
        ).splitlines()
        if len(touching) != 1 or not _COMMIT.fullmatch(touching[0]):
            raise AdaptedStudentProtocolError("student protocol requires exactly one later acceptance change")
        acceptance = touching[0]
        parents = text("rev-list", "--parents", "-n", "1", acceptance).split()
        if len(parents) != 2 or parents[1] != implementation:
            raise AdaptedStudentProtocolError("student acceptance must directly follow its implementation")
        changed = {
            item.decode("utf-8")
            for item in git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", acceptance).split(b"\0")
            if item
        }
        if PROTOCOL_PATH not in changed or not changed <= {PROTOCOL_PATH, "docs/refactor/current_handoff.md"}:
            raise AdaptedStudentProtocolError("student acceptance contains non-review changes")
        accepted_raw = git("show", f"{acceptance}:{PROTOCOL_PATH}")
        reviewed = load_adapted_student_protocol(accepted_raw, require_accepted=True)
        reviewed["review"] = copy.deepcopy(REVIEW_PROPOSED)
        if (
            _canonical(reviewed) != _canonical(proposed)
            or raw != accepted_raw
            or git("show", f"{head}:{PROTOCOL_PATH}") != raw
        ):
            raise AdaptedStudentProtocolError("student accepted artifact or review-only transition differs")
        if git("diff", "--cached", "--name-only", "-z", "--", PROTOCOL_PATH, *SCIENCE_PATHS):
            raise AdaptedStudentProtocolError("student named science has staged changes")
        for path, source_sha in source_hashes.items():
            for commit in (implementation, head):
                entry = text("ls-tree", commit, "--", path).split(maxsplit=1)
                if not entry or entry[0] not in {"100644", "100755"}:
                    raise AdaptedStudentProtocolError("student science must be a regular committed blob")
                if hashlib.sha256(git("show", f"{commit}:{path}")).hexdigest() != source_sha:
                    raise AdaptedStudentProtocolError(f"student named scientific source changed: {path}")
    if (
        text("rev-parse", "--verify", "HEAD^{commit}") != head
        or read(PROTOCOL_PATH) != raw
        or any(read(path) != raw_bytes for path, raw_bytes in historical_bytes.items())
    ):
        raise AdaptedStudentProtocolError("student HEAD or protocol changed during verification")
    for path, expected_sha in source_hashes.items():
        if hashlib.sha256(read(path)).hexdigest() != expected_sha:
            raise AdaptedStudentProtocolError("student source changed during verification")
    return AdaptedStudentProtocolBinding(
        amendment_id=PROTOCOL_ID,
        path=root / PROTOCOL_PATH,
        git_commit=acceptance,
        sha256=hashlib.sha256(raw).hexdigest(),
        reviewed_implementation_commit=implementation,
        protocol_sha256=adapted_student_protocol_sha256(payload),
        payload=payload,
        science_file_sha256=source_hashes,
        head=head,
    )


def validate_student_config(config: Any) -> None:
    """Validate fixed scientific settings without resolving Git or reading data."""
    if not isinstance(config, dict):
        raise AdaptedStudentProtocolError("student configuration must be a mapping")
    fixed = {
        "seed": 42,
        "protocol_track": "qwen3_v2",
        "prereg_path": "prereg/qwen3_v2.yaml",
        "prereg_version": "qwen3_v2",
        "protocol_amendment_path": PROTOCOL_PATH,
        "adapted_teacher": adapted_teacher_config(),
        "experiment": {
            "name": "canonical_sft",
            "state_source": "teacher_demo",
            "supervision": "canonical_sft",
            "use_verifier_reward": False,
        },
        "teacher_readiness": READINESS_THRESHOLDS,
    }
    required_sections = {
        "trainer": {
            "backend": "accelerate",
            "max_steps": 120,
            "steps_per_round": 1,
            "learning_rate": 0.0005,
            "weight_decay": 0.0,
            "max_completion_length": 128,
            "token_budget": 2000000,
            "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
            "batch_partition_protocol": "allocation_neutral_exact_global_batch_v1",
            "global_batch_size": 64,
            "max_microbatch_size": 4,
            "max_model_input_length": 1536,
            "checkpoint_every": 20,
            "evaluation_every": 20,
            "validation_examples": 128,
        },
        "task": {"name": "proofgraph", "num_examples": 256, "seed": 42},
        "state_source": {
            "name": "teacher_demo",
            "num_candidates": 8,
            "max_prompt_tokens": 1246,
            "max_new_tokens": 256,
            "require_exact_verifier_success": True,
            "require_complete_prompt_coverage": True,
            "require_behavior_logprobs_for_formal_sft": True,
            "zero_success_policy": "fail_closed",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
        },
        "supervision": {
            "name": "canonical_sft",
            "training_backend": "factorial_trainer",
            "training_backend_version": "in-repository-v1",
            "training_batch_contract": "optimizer-boundary-gradient-accumulation-v1",
            "normalization": "sequence",
            "require_teacher_demo_store": True,
            "mask_prompt_tokens": True,
        },
        "model": {
            "model_name_or_path": "Qwen/Qwen3-1.7B",
            "model_revision": STUDENT_REVISION,
            "tokenizer_name_or_path": "Qwen/Qwen3-1.7B",
            "tokenizer_revision": STUDENT_REVISION,
            "gradient_checkpointing": True,
            "use_cache": False,
        },
        "teacher": {
            "model_name_or_path": "Qwen/Qwen3-8B",
            "model_revision": BASE_TEACHER_REVISION,
            "tokenizer_name_or_path": "Qwen/Qwen3-8B",
            "tokenizer_revision": BASE_TEACHER_REVISION,
            "gradient_checkpointing": False,
            "use_cache": True,
        },
        "g0": {"full_parameter_training": True},
    }
    try:
        for field, expected in fixed.items():
            if _canonical(config.get(field)) != _canonical(expected):
                raise AdaptedStudentProtocolError(f"student config differs: {field}")
        for section, expected in required_sections.items():
            current = config.get(section)
            if not isinstance(current, dict) or any(
                _canonical(current.get(field)) != _canonical(value) for field, value in expected.items()
            ):
                raise AdaptedStudentProtocolError(f"student config differs: {section}")
            if section in {"trainer", "supervision"} and set(current) != set(expected):
                raise AdaptedStudentProtocolError(f"student config has unknown {section} fields")
        for role in ("model", "teacher"):
            current = config[role]
            for field, value in {
                "torch_dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "trust_remote_code": False,
                "allow_unpinned_revision": False,
                "low_cpu_mem_usage": True,
                "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
                "protocol_track": "qwen3_v2",
                "artifact_namespace": "qwen3-v2",
                "prompt_protocol": {
                    "name": "qwen3_non_thinking_v1",
                    "enable_thinking": False,
                    "messages": "single_user",
                    "add_generation_prompt": True,
                    "chat_template_sha256": CHAT_TEMPLATE_SHA256,
                },
                "sampling_protocol": {
                    "name": "qwen3_non_thinking_sampling_v1",
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                },
            }.items():
                if _canonical(current.get(field)) != _canonical(value):
                    raise AdaptedStudentProtocolError(f"student model contract differs: {role}.{field}")
        if any(
            config.get(field)
            for field in (
                "execution_science_protocol_path",
                "execution_safety_descriptor_path",
                "execution_class_certification_path",
                "pilot_profile",
                "production_profile",
            )
        ):
            raise AdaptedStudentProtocolError("calibration cannot borrow an execution-class certificate")
    except (TypeError, ValueError) as error:
        if isinstance(error, AdaptedStudentProtocolError):
            raise
        raise AdaptedStudentProtocolError("invalid student config scalar") from error


def validate_student_protocol(config: Any) -> AdaptedStudentProtocolBinding:
    validate_student_config(config)
    root = Path.cwd()
    metadata = root / (".git" if (root / ".git" / "HEAD").is_file() else ".opd-git")
    return resolve_adapted_student_protocol(root, git_dir=metadata)
