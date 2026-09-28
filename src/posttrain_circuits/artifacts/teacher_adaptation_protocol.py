"""Additive, genuinely reviewed protocol for qualifying a learned dense teacher.

Protocol acceptance authorizes measurement, not teacher-quality acceptance.
This resolver reads actual Git history and named source bytes; it neither edits
the original preregistration nor manufactures a clean checkout or HF revision.
No model, checkpoint weights, formal responses or GPU is loaded here.
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

import yaml

from posttrain_circuits.artifacts.git_provenance import _git_environment
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow

PROTOCOL_PATH = "prereg/amendments/qwen3_teacher_adaptation_v1.yaml"
PROTOCOL_ID = "qwen3-teacher-adaptation-v1"
SUPPLEMENTAL_EXPERIMENT_NAMESPACE = "opd-qwen3-teacher-adaptation-confirmation-20260928"
BASE_PREREG_SHA256 = "8d6bdeab0b9302c8824c4709f556c6c41a896bd2cfce21e7794d131d176ba0a4"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_FINGERPRINT = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
REVIEW_PROPOSED = {
    "status": "proposed",
    "reviewed_implementation_commit": None,
    "reviewer": None,
    "reviewed_at_utc": None,
    "rationale": None,
}
READINESS_THRESHOLDS = {
    "minimum_teacher_answer_accuracy": 0.90,
    "minimum_teacher_exact_proof_accuracy": 0.85,
    "minimum_teacher_first_rule_top1_accuracy": 0.80,
    "minimum_teacher_intermediate_top1_accuracy": 0.80,
    "minimum_teacher_topk_mass": 0.90,
    "minimum_teacher_topk_target_coverage": 0.90,
    "minimum_corrupted_prefix_recovery_accuracy": 0.70,
    "minimum_causal_shift_logprob": 0.0,
}

# Deliberate scientific/qualification surfaces, not a repository snapshot.
# Their actual implementation-commit blobs are compared again at every use.
SCIENCE_PATHS = (
    "configs/config.yaml",
    "configs/task/proofgraph_main.yaml",
    "configs/teacher/qwen3_v2_teacher_8b.yaml",
    "configs/g0/qwen3_v2_eap_separation.yaml",
    "configs/experiment/canonical_sft.yaml",
    "src/posttrain_circuits/core/config.py",
    "src/posttrain_circuits/artifacts/teacher_adaptation_protocol.py",
    "src/posttrain_circuits/artifacts/teacher_acceptance.py",
    "src/posttrain_circuits/artifacts/teacher_identity.py",
    "src/posttrain_circuits/artifacts/compatibility.py",
    "src/posttrain_circuits/artifacts/hashing.py",
    "src/posttrain_circuits/artifacts/io.py",
    "src/posttrain_circuits/artifacts/execution_safe_io.py",
    "src/posttrain_circuits/artifacts/git_provenance.py",
    "src/posttrain_circuits/models/adapted_teacher.py",
    "src/posttrain_circuits/models/loading.py",
    "src/posttrain_circuits/models/prompt_protocol.py",
    "src/posttrain_circuits/datasets/proofgraph/contracts.py",
    "src/posttrain_circuits/datasets/proofgraph/generation.py",
    "src/posttrain_circuits/datasets/proofgraph/splits.py",
    "src/posttrain_circuits/datasets/proofgraph/serialization.py",
    "src/posttrain_circuits/datasets/proofgraph/rendering.py",
    "src/posttrain_circuits/datasets/proofgraph/parsing.py",
    "src/posttrain_circuits/datasets/proofgraph/verification.py",
    "src/posttrain_circuits/datasets/teacher_demos/contracts.py",
    "src/posttrain_circuits/learning/teacher/adaptation.py",
    "src/posttrain_circuits/learning/teacher/adaptation_fit.py",
    "src/posttrain_circuits/learning/teacher/adaptation_evaluation.py",
    "src/posttrain_circuits/learning/teacher/adapted_candidate_generation.py",
    "src/posttrain_circuits/learning/teacher/demo_generation.py",
    "src/posttrain_circuits/learning/teacher/seeding.py",
    "src/posttrain_circuits/learning/teacher/evaluation.py",
    "src/posttrain_circuits/learning/supervision/losses.py",
    "src/posttrain_circuits/cli/evaluate_teacher_readiness.py",
    "src/posttrain_circuits/causal_circuits/metrics/probes.py",
    "tools/sdsc_teacher_fit.py",
    "tools/sdsc_teacher_fit_contract.py",
    "tools/sdsc_teacher_fit_memory.py",
    "tools/sdsc_teacher_fit_job.sh",
    "tools/sdsc_teacher_prepare.py",
    "tools/sdsc_training_preflight.py",
    "tools/sdsc_teacher_adapt.py",
    "tools/sdsc_teacher_probe.py",
    "tools/sdsc_teacher_qualify.py",
    "tools/sdsc_teacher_qualify_contract.py",
    "tools/sdsc_teacher_qualify_job.sh",
    "tools/sdsc_teacher_qualification_audit.py",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_MAX_BYTES = 2 * 1024 * 1024


class TeacherAdaptationProtocolError(ValueError):
    """The candidate protocol or its real provenance does not satisfy v1."""


def proposed_teacher_adaptation_protocol() -> dict[str, Any]:
    """Return a fresh fixed candidate; no selected checkpoint is invented."""
    return {
        "schema_version": 1,
        "kind": "opd_teacher_adaptation_protocol",
        "protocol_id": PROTOCOL_ID,
        "scope": {
            "purpose": "qualification_of_one_development_selected_dense_teacher",
            "execution": "sdsc_h100_teacher_only",
            "protocol_acceptance_is_teacher_acceptance": False,
            "student_training_authorized_by_this_artifact": False,
            "blackwell_execution_class_reused": False,
        },
        "preserved_base": {
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_sha256": BASE_PREREG_SHA256,
            "student_max_completion_length": 128,
            "historical_teacher_readiness_max_new_tokens": 128,
            "claim_original_128_budget_pass": False,
            "original_first_128_validation_previously_exposed": True,
        },
        "teacher": {
            "identity_schema_version": 1,
            "kind": "learned_dense_checkpoint",
            "base_model_id": "Qwen/Qwen3-8B",
            "base_revision": BASE_REVISION,
            "tokenizer_id": "Qwen/Qwen3-8B",
            "tokenizer_revision": BASE_REVISION,
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "checkpoint_identity": "canonical_dense_manifest_sha256_with_actual_file_rehash",
            "checkpoint_selection": "first_scheduled_full_dev_all_eight_pass",
            "checkpoint_locator_is_scientific_identity": False,
            "adapter_wrappers_allowed_for_qualification": False,
        },
        "science": {
            "response_instructions_version": "prompt_v7_positive_TRUE_negative_NOT",
            "response_instructions_sha256": (
                "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
            ),
            "canonical_targets_and_verifier": "unchanged_proofgraph",
            "prefix_conditional_history": "score_each_complete_target_on_its_own_autoregressive_history",
            "prefix_pair_rule": "original_active_support_path_swap_seed42_with_recorded_alignment_skips",
            "prefix_stages": ["first_rule_selection", "intermediate_conclusion"],
            "prefix_top_k": 128,
            "every_causal_side_structurally_valid_and_strictly_positive": True,
            "readiness_thresholds": dict(READINESS_THRESHOLDS),
            "max_new_tokens": 256,
            "budget_amendment_scope": "adapted_teacher_dev_supplemental_and_formal_readiness_only",
            "greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
            "greedy_seed": "42_plus_original_source_index",
        },
        "fit": {
            "train_examples": 8192,
            "development_examples": 512,
            "train_raw_pair_seed_start": 70_000_042,
            "development_raw_pair_seed_start": 80_000_042,
            "dataset_manifest_sha256": "742b62a1ee328c8d8f660106265e4a342458fe5145243ecb08368eaa745f502d",
            "train_examples_sha256": "a85ac939cba53e675d1737c75d9ba77d2516ed0f2cb0def642c97774cc09522f",
            "development_examples_sha256": "3bd34a2e3a743c41a5215317f220c0ae99dba66841776f81565c61768f16e4c5",
            "original_family_rows_to_check": 144000,
            "isolation_dimensions": ["semantic", "example_id", "pair_group_id", "pair_seed"],
            "retain_all_rows": True,
            "formal_rows_used_for_training_or_selection": False,
            "seed": 271828,
            "epochs": 4,
            "global_batch_size": 64,
            "optimizer_steps": 512,
            "world_size": 4,
            "microbatch_per_rank": 1,
            "loss": "mean_of_response_token_means_per_sequence",
            "response_supervision": "canonical_response_and_EOS_prompt_and_padding_masked",
            "lora": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.0,
                "modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            },
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "max_grad_norm": 1.0,
                "warmup_steps": 16,
                "schedule": "linear_warmup_then_cosine",
            },
            "max_prefix_tokens": 1280,
            "max_response_tokens_with_eos": 256,
            "max_model_input_tokens": 1536,
            "evaluation_steps": [128, 256, 384, 512],
            "evaluation_model": "freshly_reloaded_verified_merged_dense_checkpoint",
            "evaluate_all_512_development_rows": True,
            "preserve_all_scheduled_reports_and_complete_all_512_steps": True,
            "selection_rule": "first_scheduled_checkpoint_with_all_eight_development_checks_true",
            "reselect_after_confirmation_exposure": False,
            "preflight_results_are_formal_acceptance": False,
        },
        "cohorts": {
            "train_probe": {
                "split": "train",
                "source_start": 0,
                "source_stop": 32,
                "count": 32,
                "examples_sha256": "3dfdee4142a1b12cb82cd9e002d631849578d5364a26de17292f249bf87ca9ed",
                "candidates_per_prompt": 8,
                "required_covered_prompts": 32,
            },
            "supplemental": {
                "split": "validation",
                "source_start": 128,
                "source_stop": 256,
                "count": 128,
                "examples_sha256": "532b11ac85fad0b35be1e253c2d30a8ca838ea8849adf77503a43b6433236073",
                "source_file_sha256": "8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3",
                "seed_offset": 42,
                "seed_uses_original_source_index": True,
                "required_complete_pairs": 64,
            },
            "formal_readiness": {
                "split": "validation",
                "source_start": 0,
                "source_stop": 128,
                "count": 128,
                "examples_sha256": "ceecd7470309dbb78351f8ecba850b98b6f35bf749aa89ac101498ba4ce921e8",
                "source_file_sha256": "8d9f710b8242a53f37a5714240e2ffd64769aca5828ef1d8e180c81c277f4ed3",
                "seed_offset": 42,
                "seed_uses_original_source_index": True,
                "prior_model_output_exposure_disclosed": True,
            },
            "formal_store": {
                "split": "train",
                "source_start": 0,
                "source_stop": 256,
                "count": 256,
                "examples_sha256": "b2de1ba6a96f602f4ffd511f3c1e3e83104c426315123c2b3cca5bc9e216ea1c",
                "candidates_per_prompt": 8,
                "required_attempts": 2048,
                "required_covered_prompts": 256,
            },
        },
        "candidate_generation": {
            "request_seed": 31415,
            "seed_derivation": "existing_teacher_candidate_seed",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": 256,
            "max_prefix_tokens": 1246,
            "retain_all_attempts_and_actual_tokens_and_logprobs": True,
            "retry_resample_or_repair_generated_proof": False,
        },
        "qualification": {
            "ordered_stages": [
                "selected_checkpoint",
                "train_probe",
                "supplemental",
                "formal_readiness",
                "formal_store",
                "independent_artifact_acceptance",
            ],
            "stop_on_any_failed_gate": True,
            "same_selected_checkpoint_for_all_stages": True,
            "supplemental_experiment_namespace": SUPPLEMENTAL_EXPERIMENT_NAMESPACE,
            "supplemental_claim_before_any_inference": True,
            "supplemental_claim_independent_of_checkpoint_and_protocol_hash": True,
            "preserve_claim_on_failure_or_unknown_outcome": True,
            "supplemental_is_one_shot_no_reselection": True,
            "verify_accounting_receipts_and_actual_persistent_artifacts": True,
            "independent_acceptance_attestation_required": True,
            "producer_may_self_accept_teacher": False,
            "protocol_review_may_self_accept_teacher": False,
            "downstream_student_migration_in_scope": False,
        },
        "science_files": list(SCIENCE_PATHS),
        "review_contract": {
            "mechanism": "implementation_commit_then_distinct_review_only_acceptance_commit",
            "acceptance_commit_self_reference": "forbidden",
            "acceptance_allowed_changed_paths": [PROTOCOL_PATH, "docs/refactor/current_handoff.md"],
            "current_named_science_bytes_must_equal_implementation": True,
            "later_unrelated_commits_invalidate_review": False,
        },
        "review": copy.deepcopy(REVIEW_PROPOSED),
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def validate_teacher_adaptation_protocol(payload: Any, *, require_accepted: bool = False) -> dict[str, Any]:
    """Fail closed on any changed fixed field, extra field or scalar type."""
    try:
        if not isinstance(payload, dict) or set(payload) != set(proposed_teacher_adaptation_protocol()):
            raise TeacherAdaptationProtocolError("protocol has missing or unknown fields")
        review = payload["review"]
        if not isinstance(review, dict) or set(review) != set(REVIEW_PROPOSED):
            raise TeacherAdaptationProtocolError("protocol review has missing or unknown fields")
        if review["status"] == "proposed":
            if review != REVIEW_PROPOSED or require_accepted:
                raise TeacherAdaptationProtocolError("teacher protocol is proposed, not accepted")
        elif review["status"] == "accepted":
            if not isinstance(review["reviewed_implementation_commit"], str) or not _COMMIT.fullmatch(
                review["reviewed_implementation_commit"]
            ):
                raise TeacherAdaptationProtocolError("review requires a genuine full implementation commit")
            for field in ("reviewer", "rationale"):
                if not isinstance(review[field], str) or not review[field].strip():
                    raise TeacherAdaptationProtocolError(f"review requires nonempty {field}")
            timestamp = review["reviewed_at_utc"]
            if not isinstance(timestamp, str) or not _UTC_TIMESTAMP.fullmatch(timestamp):
                raise TeacherAdaptationProtocolError("review timestamp must be explicit UTC")
            datetime.fromisoformat(timestamp[:-1] + "+00:00")
        else:
            raise TeacherAdaptationProtocolError("unknown protocol review status")
        expected = proposed_teacher_adaptation_protocol()
        expected["review"] = copy.deepcopy(review)
        if _canonical(payload) != _canonical(expected):
            raise TeacherAdaptationProtocolError(
                "teacher adaptation protocol differs from the fixed v1 specification"
            )
    except (TypeError, OverflowError, RecursionError, ValueError) as error:
        if isinstance(error, TeacherAdaptationProtocolError):
            raise
        raise TeacherAdaptationProtocolError(f"invalid teacher protocol value: {error}") from error
    return copy.deepcopy(payload)


def load_teacher_adaptation_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
    """Load strict duplicate-free YAML without aliases or implicit review dates."""
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_BYTES:
        raise TeacherAdaptationProtocolError("protocol bytes are empty or exceed their bound")

    class StrictLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise TeacherAdaptationProtocolError("YAML keys must be unique strings")
            result[key] = loader.construct_object(value_node)
        return result

    StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        if any(
            isinstance(token, yaml.tokens.AliasToken | yaml.tokens.AnchorToken) for token in yaml.scan(raw)
        ):
            raise TeacherAdaptationProtocolError("protocol YAML aliases and anchors are forbidden")
        payload = yaml.load(raw, Loader=StrictLoader)
    except (yaml.YAMLError, UnicodeError, RecursionError) as error:
        raise TeacherAdaptationProtocolError("invalid teacher protocol YAML") from error
    return validate_teacher_adaptation_protocol(payload, require_accepted=require_accepted)


def teacher_adaptation_protocol_sha256(payload: Any) -> str:
    """Canonical scientific core identity, independent of later review metadata."""
    checked = validate_teacher_adaptation_protocol(payload)
    checked.pop("review")
    return hashlib.sha256(_canonical(checked)).hexdigest()


def validate_teacher_adaptation_review_transition(
    proposed: Any, accepted: Any, *, implementation_commit: str, acceptance_commit: str
) -> None:
    """Validate the review-only artifact transition, not Git ancestry or quality."""
    if (
        any(
            not isinstance(commit, str) or not _COMMIT.fullmatch(commit)
            for commit in (implementation_commit, acceptance_commit)
        )
        or implementation_commit == acceptance_commit
    ):
        raise TeacherAdaptationProtocolError(
            "implementation and acceptance require distinct real commit identities"
        )
    original = validate_teacher_adaptation_protocol(proposed)
    reviewed = validate_teacher_adaptation_protocol(accepted, require_accepted=True)
    if (
        original["review"] != REVIEW_PROPOSED
        or reviewed["review"]["reviewed_implementation_commit"] != implementation_commit
    ):
        raise TeacherAdaptationProtocolError("acceptance does not review the proposed implementation")
    reviewed["review"] = copy.deepcopy(REVIEW_PROPOSED)
    if _canonical(reviewed) != _canonical(original):
        raise TeacherAdaptationProtocolError("acceptance changes scientific protocol content")


@dataclass(frozen=True)
class ResolvedTeacherAdaptationProtocol:
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


def resolve_teacher_adaptation_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedTeacherAdaptationProtocol:
    """Resolve real review lineage and compare each named current science blob.

    ``git_dir`` is explicit for Quest's project-owned .opd-git metadata; it is
    never inferred from environment variables or substituted with invented
    commits. A deployed producer restores a real Git checkout/provenance bundle.
    Unrelated tracked dirt is reported honestly and is not a scientific hash
    input. Named-file dirt always fails accepted resolution, including staged
    changes, changes committed after review, symlinks and missing files.
    """
    root = Path(repo_root).absolute()
    if not root.is_dir() or root.resolve() != root:
        raise TeacherAdaptationProtocolError("code root must be an existing real directory")
    metadata = Path(git_dir).absolute() if git_dir is not None else root / ".git"
    if metadata not in (root / ".git", root / ".opd-git") or not metadata.is_dir() or metadata.is_symlink():
        raise TeacherAdaptationProtocolError("explicit real .git or .opd-git directory is required")

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
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise TeacherAdaptationProtocolError(f"real Git provenance query failed: {args[0]}")
        return result.stdout

    def text(*args: str) -> str:
        return git(*args).decode("utf-8", errors="strict").strip()

    def read(relative: str) -> bytes:
        return read_regular_bytes_nofollow(
            root / relative, context=f"teacher protocol source {relative}", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    if not _COMMIT.fullmatch(head) or (expected_head is not None and head != expected_head):
        raise TeacherAdaptationProtocolError("actual Git HEAD differs from expected producer provenance")
    if text("rev-parse", "--is-shallow-repository") != "false":
        raise TeacherAdaptationProtocolError("review lineage requires complete real Git history")
    raw = read(PROTOCOL_PATH)
    payload = load_teacher_adaptation_protocol(raw, require_accepted=require_accepted)
    if hashlib.sha256(read("prereg/qwen3_v2.yaml")).hexdigest() != BASE_PREREG_SHA256:
        raise TeacherAdaptationProtocolError("original frozen preregistration bytes changed")
    source_hashes = {path: hashlib.sha256(read(path)).hexdigest() for path in SCIENCE_PATHS}
    implementation = payload["review"]["reviewed_implementation_commit"]
    acceptance = None
    if payload["review"]["status"] == "accepted":
        if implementation == head or text("merge-base", implementation, head) != implementation:
            raise TeacherAdaptationProtocolError(
                "reviewed implementation is not a distinct ancestor of actual HEAD"
            )
        proposed = load_teacher_adaptation_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
        touching = text(
            "rev-list", "--reverse", "--ancestry-path", f"{implementation}..{head}", "--", PROTOCOL_PATH
        ).splitlines()
        if len(touching) != 1 or not _COMMIT.fullmatch(touching[0]):
            raise TeacherAdaptationProtocolError("protocol must have exactly one later acceptance change")
        acceptance = touching[0]
        parents = text("rev-list", "--parents", "-n", "1", acceptance).split()
        if len(parents) != 2:
            raise TeacherAdaptationProtocolError("review-only acceptance must have exactly one parent")
        changed = git("diff-tree", "--no-commit-id", "--name-only", "-r", "-z", acceptance).split(b"\0")
        changed_paths = {value.decode("utf-8") for value in changed if value}
        if PROTOCOL_PATH not in changed_paths or not changed_paths <= {
            PROTOCOL_PATH,
            "docs/refactor/current_handoff.md",
        }:
            raise TeacherAdaptationProtocolError("acceptance commit contains non-review changes")
        accepted_raw = git("show", f"{acceptance}:{PROTOCOL_PATH}")
        validate_teacher_adaptation_review_transition(
            proposed,
            load_teacher_adaptation_protocol(accepted_raw, require_accepted=True),
            implementation_commit=implementation,
            acceptance_commit=acceptance,
        )
        if raw != accepted_raw or git("show", f"{head}:{PROTOCOL_PATH}") != raw:
            raise TeacherAdaptationProtocolError("current protocol bytes differ from accepted Git artifact")
        if (
            hashlib.sha256(git("show", f"{implementation}:prereg/qwen3_v2.yaml")).hexdigest()
            != BASE_PREREG_SHA256
        ):
            raise TeacherAdaptationProtocolError("implementation changed the original preregistration")
        if hashlib.sha256(git("show", f"{head}:prereg/qwen3_v2.yaml")).hexdigest() != BASE_PREREG_SHA256:
            raise TeacherAdaptationProtocolError("actual HEAD changed the original preregistration")
        if git(
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--",
            PROTOCOL_PATH,
            "prereg/qwen3_v2.yaml",
            *SCIENCE_PATHS,
        ):
            raise TeacherAdaptationProtocolError("named scientific files have unreviewed staged changes")
        for path, current_sha in source_hashes.items():
            entry = text("ls-tree", implementation, "--", path).split(maxsplit=1)
            if not entry or entry[0] not in {"100644", "100755"}:
                raise TeacherAdaptationProtocolError(
                    f"reviewed science path is not a regular Git blob: {path}"
                )
            reviewed = git("show", f"{implementation}:{path}")
            if hashlib.sha256(reviewed).hexdigest() != current_sha:
                raise TeacherAdaptationProtocolError(
                    f"named scientific implementation changed after review: {path}"
                )
            if git("show", f"{head}:{path}") != reviewed:
                raise TeacherAdaptationProtocolError(
                    f"actual HEAD contains an unreviewed scientific blob: {path}"
                )
    clean = not git("diff", "--name-only", "HEAD", "--")
    if text("rev-parse", "--verify", "HEAD^{commit}") != head or read(PROTOCOL_PATH) != raw:
        raise TeacherAdaptationProtocolError("Git HEAD or protocol changed during validation")
    if hashlib.sha256(read("prereg/qwen3_v2.yaml")).hexdigest() != BASE_PREREG_SHA256:
        raise TeacherAdaptationProtocolError("original frozen preregistration changed during validation")
    for path, expected_sha in source_hashes.items():
        if hashlib.sha256(read(path)).hexdigest() != expected_sha:
            raise TeacherAdaptationProtocolError(f"named scientific source changed during validation: {path}")
    return ResolvedTeacherAdaptationProtocol(
        payload=payload,
        protocol_sha256=teacher_adaptation_protocol_sha256(payload),
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        implementation_commit=implementation,
        acceptance_commit=acceptance,
        head=head,
        science_file_sha256=source_hashes,
        tracked_worktree_clean=clean,
    )
