"""Prospective common 1.7B preparation, separate from all compared methods.

Canonical symbolic proofs provide the targets. No teacher inference or original
evaluation outcome chooses a checkpoint. Development selection is not G0 or
student acceptance; every later method must begin from the same qualified bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.adapted_student_protocol import SCIENCE_PATHS as PARENT_SCIENCE_PATHS
from posttrain_circuits.artifacts.git_provenance import _git_environment
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow
from posttrain_circuits.artifacts.teacher_adaptation_protocol import (
    BASE_PREREG_SHA256,
    CHAT_TEMPLATE_SHA256,
    REVIEW_PROPOSED,
    TOKENIZER_FINGERPRINT,
)
from posttrain_circuits.artifacts.teacher_adaptation_protocol import (
    SCIENCE_PATHS as TEACHER_SCIENCE_PATHS,
)
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.datasets.proofgraph.splits import (
    assert_split_isolation,
    build_split,
    canonical_semantic_key,
)

PROTOCOL_PATH = "prereg/amendments/qwen3_student_preparation_v1.json"
PROTOCOL_ID = "qwen3-student-preparation-v1"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PARENT_HEAD = "6c04f804b302184b8ff95d00fab404e0531ed8d6"
PARENT_PROTOCOL_SHA256 = "c701dde9691210dcdd06f4a1076299941fdb6ad8d609280a13e80d5d7a4333f7"
RESPONSE_INSTRUCTIONS_SHA256 = "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
FIT_COUNT = 2048
DEV_COUNT = 512
FIT_SEED_START = 90_000_042
DEV_SEED_START = 100_000_042
CHECKPOINT_STEPS = (4, 8, 16, 32)
MAX_INPUT_TOKENS = 2_000_000
MAX_PREFIX_TOKENS = 1344
DEVELOPMENT_MAX_MODEL_INPUT_TOKENS = 1600
DATASET_MANIFEST_SHA256 = "274e5f303d1a44b69790db687fc9fdde43bfae5c27cc2e5c5f656cdd0bf28cc6"
EXAMPLES_SHA256 = {
    "student_fit": "3422669aca22eb67a18dc604b1a2ed7d7dece51911ae12cf196235a09ec60be5",
    "student_dev": "7722e04c365cd122b7ea9454f34062a7aced5e3a67a633b7048cee3aa9333690",
}
DIFFICULTY = {
    "depth_range": [2, 4],
    "ood_depth_range": [5, 7],
    "distractor_range": [4, 16],
    "structures": ["chain", "branch", "converging_dag"],
    "multiple_valid_proof_fraction": 0.0,
}
FROZEN_SCIENCE_PATHS = tuple(dict.fromkeys((*PARENT_SCIENCE_PATHS, *TEACHER_SCIENCE_PATHS)))
SCIENCE_PATHS = tuple(
    dict.fromkeys(
        (
            *FROZEN_SCIENCE_PATHS,
            "src/posttrain_circuits/experiments/protocols/student_preparation.py",
            "tools/sdsc_student_prepare.py",
            "tools/sdsc_student_prepare_job.py",
            "tools/sdsc_student_prepare_worker.py",
            "tools/sdsc_student_prepare_audit.py",
            "tools/sdsc_student_lr_probe.py",
            "tools/sdsc_student_quality_probe.py",
            "tools/sdsc_provenance.py",
        )
    )
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_MAX_BYTES = 4 * 1024 * 1024


class StudentPreparationError(ValueError):
    """The frozen common-baseline preparation contract was not satisfied."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentPreparationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def proposed_student_preparation_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "opd_student_preparation_protocol",
        "protocol_id": PROTOCOL_ID,
        "scope": {
            "purpose": "independently_prepared_common_1p7b_initial_for_all_methods",
            "interpretation": "post_preparation_capability_not_native_instruction_capability",
            "execution": "sdsc_h100_preparation_only",
            "teacher_generated_targets": False,
            "preparation_is_student_or_g0_acceptance": False,
            "downstream_calibration_g0_pilot_in_scope": False,
            "original_failed_initial_is_preserved": True,
            "previous_validation128_exposure_disclosed": True,
        },
        "preserved_base": {
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_sha256": BASE_PREREG_SHA256,
            "parent_science_head": PARENT_HEAD,
            "parent_student_protocol": "prereg/amendments/qwen3_adapted_student_calibration_v6.json",
            "parent_student_protocol_sha256": PARENT_PROTOCOL_SHA256,
            "all_original_numerical_gates_unchanged": True,
            "downstream_student_rollout_max_completion_length": 128,
        },
        "model": {
            "base_model_id": "Qwen/Qwen3-1.7B",
            "base_revision": BASE_REVISION,
            "tokenizer_id": "Qwen/Qwen3-1.7B",
            "tokenizer_revision": BASE_REVISION,
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "response_instructions_sha256": RESPONSE_INSTRUCTIONS_SHA256,
            "initial": "fresh_original_HF_BF16_weights_never_old_trained_checkpoint",
            "trainable_parameters": "all",
            "adapter_training": False,
            "prepared_identity": "actual_dense_checkpoint_sha256_never_base_revision",
        },
        "data": {
            "fit_examples": FIT_COUNT,
            "development_examples": DEV_COUNT,
            "fit_raw_pair_seed_start": FIT_SEED_START,
            "development_raw_pair_seed_start": DEV_SEED_START,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "examples_sha256": dict(EXAMPLES_SHA256),
            "difficulty": copy.deepcopy(DIFFICULTY),
            "generator": "unchanged_ProofGraph_build_split_complete_signed_pairs",
            "targets": "unchanged_symbolic_canonical_proof_and_EOS",
            "fit_order": "deterministic_manifest_order_one_pass_no_rng",
            "all_original_family_rows": 144000,
            "teacher_fit_rows": 8192,
            "teacher_development_rows": 512,
            "isolation_dimensions": ["semantic", "example_id", "pair_group_id", "pair_seed"],
            "filter_truncate_or_repair_rows": False,
            "original_validation_test_or_circuit_rows_used_for_selection": False,
            "max_prefix_tokens": MAX_PREFIX_TOKENS,
            "max_response_tokens_including_eos": 256,
            "max_model_input_tokens": 1536,
            "all_fit_tokens_must_fit_budget_before_any_training": True,
        },
        "training": {
            "seed": 42,
            "world_size": 2,
            "global_batch_size": 64,
            "microbatch_per_rank": 4,
            "gradient_accumulation_steps": 8,
            "optimizer_steps": 32,
            "epochs": 1,
            "input_token_budget": MAX_INPUT_TOKENS,
            "token_budget_unit": "global_nonpadding_model_input_tokens",
            "loss": "response_and_EOS_sequence_mean_cross_entropy",
            "prompt_and_padding_masked": True,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 5e-5,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "schedule": "constant",
                "warmup_steps": 0,
                "gradient_clipping": None,
            },
            "fsdp": "FULL_SHARD",
            "use_orig_params": False,
            "fresh_optimizer_rng_and_cursors": True,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "complete_all_32_steps_even_if_an_earlier_candidate_qualifies": True,
        },
        "development": {
            "all_examples": DEV_COUNT,
            "max_new_tokens": 256,
            "max_model_input_tokens": DEVELOPMENT_MAX_MODEL_INPUT_TOKENS,
            "larger_context_scope": "preparation_development_only_training_and_downstream_1536_unchanged",
            "generation": "greedy_native_BF16_without_extra_autocast",
            "seed": "42_plus_manifest_index",
            "metric": "original_VerificationResult.answer_correct",
            "also_report": ["complete_correct_proof", "format_valid", "EOS", "raw_response_tokens"],
            "selection": "earliest_scheduled_checkpoint_in_prespecified_ability_band",
            "minimum_answer_correct_count": 52,
            "maximum_answer_correct_count": 307,
            "ability_band": [0.10, 0.60],
            "ability_band_is_a_preparation_selection_rule_not_a_G0_threshold": True,
            "all_four_full_development_reports_required": True,
            "no_eligible_checkpoint": "fail_and_preserve_all_evidence_no_automatic_extension",
            "reselection_after_formal_exposure": False,
        },
        "qualification": {
            "independent_raw_response_and_selection_audit": True,
            "persistent_checkpoint_readback_hash_required": True,
            "common_initial_identical_across_all_later_methods": True,
            "regenerate_baseline_rollouts_cohorts_circuits_and_bindings": True,
            "unchanged_base_anti_shortcut_paired_cohort_circuit_and_calibration_improvement_gates": True,
            "missing_challenge_or_saturated_baseline": "fail_never_redefine_cohorts_or_gates",
            "producer_may_self_accept_prepared_initial": False,
        },
        "execution": {
            "preflight_steps": 4,
            "preflight_dev_selection_or_formal_acceptance": False,
            "preflight_real_update_save_reload_and_same_world_resume_required": True,
            "fit_requires_matching_passed_preflight": True,
            "fit_restarts_original_weights_never_preflight_checkpoint": True,
            "gpu_count": 2,
            "cpu_count": 24,
            "host_memory_gib": 384,
            "minimum_headroom_gib": 32,
            "minimum_headroom_fraction": 0.20,
            "preflight_walltime_minutes": 60,
            "fit_walltime_minutes": 240,
            "maximum_concurrently_allocatable_gpus": 4,
            "blackwell_execution_class_reused": False,
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


def validate_student_preparation_protocol(payload: Any, *, require_accepted: bool = False) -> dict[str, Any]:
    expected = proposed_student_preparation_protocol()
    _require(isinstance(payload, dict) and set(payload) == set(expected), "protocol fields differ")
    review = payload["review"]
    _require(isinstance(review, dict) and set(review) == set(REVIEW_PROPOSED), "review fields differ")
    if review["status"] == "proposed":
        _require(review == REVIEW_PROPOSED and not require_accepted, "preparation protocol is not accepted")
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
            raise StudentPreparationError("invalid review date") from error
    expected["review"] = copy.deepcopy(review)
    try:
        _require(_canonical(payload) == _canonical(expected), "preparation protocol differs from frozen v1")
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentPreparationError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_preparation_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
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
        raise StudentPreparationError("invalid protocol JSON") from error
    return validate_student_preparation_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_preparation_protocol(payload)
    checked.pop("review")
    return hashlib.sha256(_canonical(checked)).hexdigest()


def make_examples(difficulty: dict[str, Any]) -> dict[str, list[TaskExample]]:
    actual = {key: difficulty.get(key) for key in DIFFICULTY}
    _require(_canonical(actual) == _canonical(DIFFICULTY), "preparation task difficulty differs")
    task = ProofGraphTask()
    splits = {
        "student_fit": build_split(task, "train", FIT_COUNT, FIT_SEED_START, DIFFICULTY),
        # build_split adds the fixed validation offset of ten million.
        "student_dev": build_split(task, "validation", DEV_COUNT, FIT_SEED_START, DIFFICULTY),
    }
    validate_examples(splits)
    return splits


def validate_examples(splits: dict[str, list[TaskExample]]) -> None:
    _require(set(splits) == {"student_fit", "student_dev"}, "preparation split roles differ")
    assert_split_isolation(splits)
    task = ProofGraphTask()
    ids: set[str] = set()
    for role, expected_count, first_seed in (
        ("student_fit", FIT_COUNT, FIT_SEED_START),
        ("student_dev", DEV_COUNT, DEV_SEED_START),
    ):
        rows = splits[role]
        _require(len(rows) == expected_count, f"{role} row count differs")
        _require(
            sha256_value([asdict(row) for row in rows]) == EXAMPLES_SHA256[role],
            f"{role} bytes or order differ from frozen preparation population",
        )
        groups: dict[str, list[TaskExample]] = {}
        seeds: dict[int, str] = {}
        for row in rows:
            _require(row.example_id not in ids, "duplicate preparation example ID")
            ids.add(row.example_id)
            seed = row.metadata.get("pair_seed")
            _require(
                type(seed) is int and first_seed <= seed < first_seed + expected_count * 100,
                "preparation pair seed outside fixed namespace",
            )
            _require(
                seed not in seeds or seeds[seed] == row.pair_group_id, "pair seed spans different groups"
            )
            seeds[seed] = row.pair_group_id
            groups.setdefault(row.pair_group_id, []).append(row)
            result = task.verify(row, task.parse_response(render_target(row)))
            _require(result.reward == 1.0, "canonical preparation target does not verify")
        _require(min(seeds) == first_seed, "first preparation seed differs")
        _require(len(groups) == expected_count // 2, "preparation pair count differs")
        for pair in groups.values():
            _require(
                len(pair) == 2
                and {row.label for row in pair} == {0, 1}
                and len({row.metadata["pair_seed"] for row in pair}) == 1,
                "preparation must retain complete positive/negative sibling pairs",
            )


def dataset_manifest(splits: dict[str, list[TaskExample]]) -> dict[str, Any]:
    validate_examples(splits)
    return {
        role: {
            "count": len(rows),
            "examples_sha256": sha256_value([asdict(row) for row in rows]),
            "ordered_ids": [row.example_id for row in rows],
            "pair_seeds": sorted({row.metadata["pair_seed"] for row in rows}),
        }
        for role, rows in splits.items()
    }


def isolation_inventory(splits: dict[str, list[TaskExample]]) -> dict[str, set[Any]]:
    validate_examples(splits)
    rows = [row for values in splits.values() for row in values]
    return {
        "semantic": {canonical_semantic_key(row) for row in rows},
        "example_id": {row.example_id for row in rows},
        "pair_group_id": {row.pair_group_id for row in rows},
        "pair_seed": {row.metadata["pair_seed"] for row in rows},
    }


def reject_overlap(inventory: dict[str, set[Any]], row: TaskExample) -> None:
    actual = {
        "semantic": canonical_semantic_key(row),
        "example_id": row.example_id,
        "pair_group_id": row.pair_group_id,
        "pair_seed": row.metadata["pair_seed"],
    }
    for key, value in actual.items():
        _require(value not in inventory[key], f"preparation overlap in {key}")


def audit_isolation(
    splits: dict[str, list[TaskExample]],
    original_family: Iterable[TaskExample],
    teacher_splits: dict[str, list[TaskExample]],
) -> dict[str, Any]:
    inventory = isolation_inventory(splits)
    _require(set(teacher_splits) == {"teacher_fit", "teacher_dev"}, "teacher isolation split roles differ")
    counts = {}
    populations = {
        "original_family": (original_family, 144000),
        "teacher_fit": (teacher_splits["teacher_fit"], 8192),
        "teacher_dev": (teacher_splits["teacher_dev"], 512),
    }
    digests = {}
    for role, (examples, expected_count) in populations.items():
        count, ids, digest = 0, set(), hashlib.sha256()
        for row in examples:
            reject_overlap(inventory, row)
            _require(row.example_id not in ids, f"duplicate {role} row in isolation population")
            ids.add(row.example_id)
            digest.update(_canonical(asdict(row)) + b"\n")
            count += 1
        _require(count == expected_count, f"incomplete {role} isolation population")
        counts[role] = count
        digests[role] = digest.hexdigest()
    return {
        "passed": True,
        "counts": counts,
        "population_stream_sha256": digests,
        "dimensions": list(inventory),
        "preparation_manifest_sha256": sha256_value(dataset_manifest(splits)),
        "scientific_acceptance": False,
    }


def encode_example(example: TaskExample, tokenizer: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    prompt = format_model_prompt(ProofGraphTask().render(example), tokenizer, model_config)
    prefix = tokenizer.encode(prompt.model_facing_prompt, add_special_tokens=False)
    text = render_target(example)
    target = tokenizer.encode(text, add_special_tokens=False)
    _require(
        tokenizer.eos_token_id is not None and prefix and target,
        "preparation requires prefix, target and EOS",
    )
    target = [*target, tokenizer.eos_token_id]
    _require(
        len(prefix) <= MAX_PREFIX_TOKENS and len(target) <= 256 and len(prefix) + len(target) <= 1536,
        "preparation row exceeds fixed token envelope; truncation is forbidden",
    )
    return {
        "example_id": example.example_id,
        "input_ids": [*prefix, *target],
        "labels": [-100] * len(prefix) + target,
        "prefix_length": len(prefix),
        "response_length": len(target),
        "prompt_sha256": prompt.model_facing_prompt_sha256,
        "target_sha256": sha256_value(text),
        "eos_token_id": tokenizer.eos_token_id,
    }


def validate_encoded_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _require(
        len(rows) == FIT_COUNT and len({row["example_id"] for row in rows}) == FIT_COUNT,
        "fit encoding must contain exactly2048 unique rows",
    )
    for row in rows:
        tokens, labels = row["input_ids"], row["labels"]
        prefix, response = row["prefix_length"], row["response_length"]
        _require(
            type(prefix) is int
            and 0 < prefix <= MAX_PREFIX_TOKENS
            and type(response) is int
            and 0 < response <= 256
            and len(tokens) == len(labels) == prefix + response <= 1536
            and all(type(value) is int and value >= 0 for value in tokens)
            and labels[:prefix] == [-100] * prefix
            and labels[prefix:] == tokens[prefix:]
            and type(row["eos_token_id"]) is int
            and tokens[-1] == row["eos_token_id"],
            "invalid preparation response mask or token envelope",
        )
    windows = [
        sum(len(row["input_ids"]) for row in rows[start : start + 64]) for start in range(0, len(rows), 64)
    ]
    _require(sum(windows) <= MAX_INPUT_TOKENS, "complete fixed preparation exceeds2M token budget")
    return {
        "fit_rows": len(rows),
        "optimizer_windows": len(windows),
        "window_input_tokens": windows,
        "total_input_tokens": sum(windows),
        "token_budget": MAX_INPUT_TOKENS,
    }


def select_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    _require(
        isinstance(records, list) and len(records) == len(CHECKPOINT_STEPS),
        "all four scheduled development records are required",
    )
    selected = None
    seen_checkpoints = set()
    fields = {
        "step",
        "checkpoint_sha256",
        "examples_sha256",
        "num_examples",
        "answer_correct",
        "proof_correct",
        "format_valid",
    }
    for expected_step, row in zip(CHECKPOINT_STEPS, records, strict=True):
        _require(isinstance(row, dict) and set(row) == fields, "development record fields differ")
        _require(
            type(row["step"]) is int
            and row["step"] == expected_step
            and type(row["num_examples"]) is int
            and row["num_examples"] == DEV_COUNT,
            "development schedule or population differs",
        )
        for key in ("checkpoint_sha256", "examples_sha256"):
            _require(isinstance(row[key], str) and _SHA256.fullmatch(row[key]), f"invalid development {key}")
        _require(row["checkpoint_sha256"] not in seen_checkpoints, "development checkpoint identity repeated")
        seen_checkpoints.add(row["checkpoint_sha256"])
        _require(
            row["examples_sha256"] == EXAMPLES_SHA256["student_dev"],
            "development population differs from frozen512 rows",
        )
        for key in ("answer_correct", "proof_correct", "format_valid"):
            _require(type(row[key]) is int and 0 <= row[key] <= DEV_COUNT, f"invalid development {key} count")
        _require(
            row["proof_correct"] <= row["answer_correct"] <= row["format_valid"],
            "development exact-proof/answer/format counts inconsistent",
        )
        if selected is None and 52 <= row["answer_correct"] <= 307:
            selected = copy.deepcopy(row)
    return selected


@dataclass(frozen=True)
class ResolvedStudentPreparationProtocol:
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


def resolve_student_preparation_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentPreparationProtocol:
    root = Path(repo_root).absolute()
    _require(root.is_dir() and root.resolve() == root, "source root must be a real directory")
    metadata = Path(git_dir).absolute() if git_dir is not None else root / ".git"
    _require(
        metadata in (root / ".git", root / ".opd-git") and metadata.is_dir() and not metadata.is_symlink(),
        "preparation needs real .git or .opd-git metadata",
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
            root / path, context=f"student preparation source {path}", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    _require(_COMMIT.fullmatch(head) and (expected_head is None or head == expected_head), "Git HEAD differs")
    _require(text("rev-parse", "--is-shallow-repository") == "false", "complete Git review history required")
    raw = read(PROTOCOL_PATH)
    payload = load_student_preparation_protocol(raw, require_accepted=require_accepted)
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
        proposed = load_student_preparation_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
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
        reviewed = load_student_preparation_protocol(reviewed_raw, require_accepted=True)
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
    return ResolvedStudentPreparationProtocol(
        payload,
        protocol_core_sha256(payload),
        hashlib.sha256(raw).hexdigest(),
        implementation,
        acceptance,
        head,
        hashes,
        clean,
    )
