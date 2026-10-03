"""Prospective fresh common student with order/renaming augmentation.

Old preparation and qualification bytes remain immutable. Development, including
all five transformations, alone selects a checkpoint; formal results never do.
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

from posttrain_circuits.artifacts.git_provenance import _git_environment
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow
from posttrain_circuits.artifacts.teacher_adaptation_protocol import (
    BASE_PREREG_SHA256,
    REVIEW_PROPOSED,
)
from posttrain_circuits.artifacts.teacher_adaptation_protocol import (
    CHAT_TEMPLATE_SHA256 as CHAT_TEMPLATE_SHA256,
)
from posttrain_circuits.artifacts.teacher_adaptation_protocol import (
    TOKENIZER_FINGERPRINT as TOKENIZER_FINGERPRINT,
)
from posttrain_circuits.datasets.proofgraph.anti_shortcut import (
    TRANSFORMATIONS,
    _assert_preserved,
    _permuted_mapping,
    _renamed,
    build_anti_shortcut_suite,
)
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.datasets.proofgraph.splits import (
    assert_split_isolation,
    build_split,
    canonical_semantic_key,
)
from posttrain_circuits.experiments.protocols import student_preparation as previous
from posttrain_circuits.experiments.protocols.student_qualification import SCIENCE_PATHS as PARENT_PATHS

PROTOCOL_PATH = "prereg/amendments/qwen3_student_invariance_v2.json"
PROTOCOL_ID = "qwen3-student-invariance-v2"
BASE_REVISION = previous.BASE_REVISION
PARENT_HEAD = "65c689a031902c2397a9e66be35dc8605efefe51"
PARENT_PROTOCOL_SHA256 = "c66d53c49d79f7bf5f63f7d933dff33bca124ebabdafb1b821ddd94dc1eb970f"
RESPONSE_INSTRUCTIONS_SHA256 = previous.RESPONSE_INSTRUCTIONS_SHA256
FIT_BASE_COUNT, DEV_BASE_COUNT = 2048, 256
FIT_COUNT, DEV_COUNT = 8192, 1536
FIT_SEED_START, DEV_SEED_START = 110_000_042, 120_000_042
FIT_TRANSFORM_SEED, DEV_TRANSFORM_SEED = 130_000_042, 140_000_042
FIT_VIEW_NAMES = ("identity", "fact_order_permutation", "rule_order_permutation", "joint_rename_order")
DEV_VIEW_NAMES = ("identity", *TRANSFORMATIONS)
STRUCTURES = ("chain", "branch", "converging_dag")
CHECKPOINT_STEPS = (4, 8, 16, 32, 64, 128)
MAX_INPUT_TOKENS = 8_000_000
MAX_PREFIX_TOKENS = 1344
DEVELOPMENT_MAX_MODEL_INPUT_TOKENS = 2454
DIFFICULTY = copy.deepcopy(previous.DIFFICULTY)
EXAMPLES_SHA256 = {
    "student_fit": "e8ef56f43dbb161257fcf3aec57f01ce9d3248c129d8d649f9810095b17ecf27",
    "student_dev": "7ed88e37ef4995347a60a744b6a95aa5054241cb4371d68e7a18e615c14c88c6",
}
VIEWS_SHA256 = {
    "student_fit": "40fe25a4f8691e3fdad4144f74e9c89ae77617faaf7d5a0939d57f3b8f63b1c3",
    "student_dev": "6a01addb1b0545ee5f728638908ac24a1fe8011fa82538f24ee100dd33e38eba",
}
DATASET_MANIFEST_SHA256 = "5b4bf0df5c6da365abeecd604af6f79daaf44aa9ee0b012b40a85718d9e9d0f5"
FROZEN_SCIENCE_PATHS = tuple(PARENT_PATHS)
SCIENCE_PATHS = (
    *FROZEN_SCIENCE_PATHS,
    "src/posttrain_circuits/experiments/protocols/student_invariance.py",
    "tools/sdsc_student_invariance.py",
    "tools/sdsc_student_invariance_job.py",
    "tools/sdsc_student_invariance_worker.py",
    "tools/sdsc_student_invariance_audit.py",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_MAX_BYTES = 4 * 1024 * 1024


class StudentInvarianceError(ValueError):
    """Frozen invariance-preparation contract rejected an input."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentInvarianceError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def proposed_student_invariance_protocol() -> dict[str, Any]:
    value = previous.proposed_student_preparation_protocol()
    value.update(kind="opd_student_invariance_protocol", protocol_id=PROTOCOL_ID)
    value["scope"].update(
        purpose="independently_prepared_order_robust_common_1p7b_initial_for_all_methods",
        prior_failed_qualification_job="54606205",
        prior_formal_anti_shortcut_exposure_disclosed=True,
        historical_checkpoints_reselected=False,
        task_training_data_augmentation_is_new_scientific_treatment=True,
    )
    value["preserved_base"].update(
        parent_science_head=PARENT_HEAD,
        parent_student_protocol="prereg/amendments/qwen3_student_preparation_v1.json",
        parent_student_protocol_sha256=PARENT_PROTOCOL_SHA256,
        parent_qualification_protocol="prereg/amendments/qwen3_student_qualification_v1.json",
        parent_qualification_protocol_sha256="4aeb5840b1a26c541f6c740d571d2ccc36506c67a88a5a39599629f4a6cc096c",
    )
    value["data"] = dict(
        fit_base_examples=FIT_BASE_COUNT,
        development_base_examples=DEV_BASE_COUNT,
        fit_examples=FIT_COUNT,
        development_examples=DEV_COUNT,
        fit_raw_pair_seed_start=FIT_SEED_START,
        development_raw_pair_seed_start=DEV_SEED_START,
        fit_transform_seed=FIT_TRANSFORM_SEED,
        development_transform_seed=DEV_TRANSFORM_SEED,
        fit_views=list(FIT_VIEW_NAMES),
        development_views=list(DEV_VIEW_NAMES),
        fit_view_order="each_base_then_identity_fact_rule_joint_complete_signed_pairs",
        joint_transform="rename(seed+1), facts(seed+2), rules(seed+3); case_seed=130000042+index*10000",
        development_transform="original_five_transform_suite_seed140000042_distractors32",
        dataset_manifest_sha256=DATASET_MANIFEST_SHA256,
        examples_sha256=dict(EXAMPLES_SHA256),
        views_sha256=dict(VIEWS_SHA256),
        difficulty=copy.deepcopy(DIFFICULTY),
        targets="unchanged_symbolic_canonical_proof_for_each_transformed_example_plus_EOS",
        generator="unchanged_ProofGraph_build_split_complete_signed_pairs",
        all_views_and_siblings_remain_within_same_partition=True,
        isolation_dimensions=["semantic", "example_id", "pair_group_id", "pair_seed"],
        isolation_population_counts=dict(
            original_family=144000,
            teacher_fit=8192,
            teacher_dev=512,
            previous_student_fit=2048,
            previous_student_dev=512,
        ),
        original_validation_test_or_circuit_rows_used_for_selection=False,
        filter_truncate_or_repair_rows=False,
        max_prefix_tokens=MAX_PREFIX_TOKENS,
        max_response_tokens_including_eos=256,
        max_model_input_tokens=1536,
        all_fit_tokens_must_fit_budget_before_any_training=True,
    )
    value["training"].pop("complete_all_32_steps_even_if_an_earlier_candidate_qualifies")
    value["training"].update(
        optimizer_steps=128,
        input_token_budget=MAX_INPUT_TOKENS,
        checkpoint_steps=list(CHECKPOINT_STEPS),
        complete_all_128_steps_even_if_candidate_qualifies=True,
        larger_budget_scope="new_preparation_only_downstream_2M_120steps_unchanged",
    )
    value["development"] = dict(
        base_examples=DEV_BASE_COUNT,
        all_examples=DEV_COUNT,
        views=list(DEV_VIEW_NAMES),
        max_new_tokens=256,
        max_model_input_tokens=DEVELOPMENT_MAX_MODEL_INPUT_TOKENS,
        larger_context_scope="new_preparation_development_only_training1536_formal_qualification2244_unchanged",
        generation="greedy_native_BF16_without_extra_autocast",
        rank_partition="complete_development_base_blocks_round_robin_rank",
        seed="42_plus_manifest_index",
        metric="original_VerificationResult.answer_correct_and_complete_proof_reward",
        selection="earliest_scheduled_checkpoint_passing_all_prespecified_development_gates",
        minimum_answer_correct_count=26,
        maximum_answer_correct_count=153,
        ability_band=[0.10, 0.60],
        minimum_iid_proof_accuracy=0.10,
        minimum_mean_transformed_proof_accuracy=0.08,
        minimum_each_transformation_proof_accuracy=0.05,
        maximum_iid_minus_mean_gap=0.05,
        maximum_iid_minus_each_transformation_gap=0.05,
        minimum_each_structure_iid_proof_accuracy=0.10,
        structure_groups=list(STRUCTURES),
        development_gates_are_additional_preparation_selection_not_formal_threshold_changes=True,
        all_six_full_development_reports_required=True,
        no_eligible_checkpoint="fail_preserve_all_evidence_no_extension_or_posthoc_reselection",
        reselection_after_formal_exposure=False,
    )
    value["execution"].update(
        fit_walltime_minutes=480,
        preflight_fixed2454_token_native_BF16_finite_forward_each_rank=True,
        preflight_independent_raw_audit_required_before_fit=True,
        minimum_node_local_free_gib=192,
        maximum_persistent_large_gib=128,
        maximum_small_fetch_mib=112,
        maximum_prompt_file_mib=32,
        maximum_other_small_file_mib=16,
    )
    value["science_files"] = list(SCIENCE_PATHS)
    value["review_contract"]["acceptance_allowed_changed_paths"] = [
        PROTOCOL_PATH,
        "docs/refactor/current_handoff.md",
    ]
    return value


def validate_student_invariance_protocol(payload: Any, *, require_accepted: bool = False) -> dict[str, Any]:
    expected = proposed_student_invariance_protocol()
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
            raise StudentInvarianceError("invalid review date") from error
    expected["review"] = copy.deepcopy(review)
    try:
        _require(_canonical(payload) == _canonical(expected), "invariance protocol differs from frozen v2")
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentInvarianceError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_invariance_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
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
        raise StudentInvarianceError("invalid protocol JSON") from error
    return validate_student_invariance_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_invariance_protocol(payload)
    checked.pop("review")
    return hashlib.sha256(_canonical(checked)).hexdigest()


@dataclass(frozen=True)
class PreparedView:
    source_example_id: str
    view: str
    example: TaskExample
    prompt: str


def _build_bases() -> dict[str, list[TaskExample]]:
    task = ProofGraphTask()
    return {
        "student_fit": build_split(task, "train", FIT_BASE_COUNT, FIT_SEED_START, DIFFICULTY),
        "student_dev": build_split(task, "validation", DEV_BASE_COUNT, FIT_SEED_START, DIFFICULTY),
    }


def make_examples(difficulty: dict[str, Any]) -> dict[str, list[TaskExample]]:
    _require(
        _canonical({k: difficulty.get(k) for k in DIFFICULTY}) == _canonical(DIFFICULTY), "difficulty changed"
    )
    splits = _build_bases()
    validate_examples(splits)
    return splits


def validate_examples(splits: dict[str, list[TaskExample]]) -> None:
    _require(set(splits) == {"student_fit", "student_dev"}, "base split roles differ")
    assert_split_isolation(splits)
    task = ProofGraphTask()
    for role, count, start in (
        ("student_fit", FIT_BASE_COUNT, FIT_SEED_START),
        ("student_dev", DEV_BASE_COUNT, DEV_SEED_START),
    ):
        rows = splits[role]
        _require(
            len(rows) == count and sha256_value([asdict(x) for x in rows]) == EXAMPLES_SHA256[role],
            "frozen base population differs",
        )
        groups = {}
        for row in rows:
            seed = row.metadata.get("pair_seed")
            _require(type(seed) is int and start <= seed < start + count * 100, "base seed namespace differs")
            groups.setdefault(row.pair_group_id, []).append(row)
            _require(
                task.verify(row, task.parse_response(render_target(row))).reward == 1.0,
                "invalid canonical base target",
            )
        _require(len(groups) == count // 2, "base pair count differs")
        for pair in groups.values():
            _require(
                len(pair) == 2
                and {x.label for x in pair} == {0, 1}
                and len({x.metadata["pair_seed"] for x in pair}) == 1,
                "incomplete signed pair",
            )


def fit_views(examples: list[TaskExample]) -> list[PreparedView]:
    task, result = ProofGraphTask(), []
    for index, source in enumerate(examples):
        seed = FIT_TRANSFORM_SEED + index * 10_000
        fact, rule = copy.deepcopy(source), copy.deepcopy(source)
        fact.facts = _permuted_mapping(fact.facts, seed + 2)
        rule.rules = _permuted_mapping(rule.rules, seed + 3)
        joint = _renamed(source, seed + 1)
        joint.facts = _permuted_mapping(joint.facts, seed + 2)
        joint.rules = _permuted_mapping(joint.rules, seed + 3)
        for name, example in zip(FIT_VIEW_NAMES, (copy.deepcopy(source), fact, rule, joint), strict=True):
            _assert_preserved(source, example)
            prompt = task.render(example)
            example.example_id = f"{source.example_id}-invariance-{name}-{sha256_value(prompt)[:12]}"
            example.metadata = {
                **example.metadata,
                "invariance_view": name,
                "source_example_id": source.example_id,
            }
            result.append(PreparedView(source.example_id, name, example, prompt))
    return result


def development_views(examples: list[TaskExample]) -> list[PreparedView]:
    cases = build_anti_shortcut_suite(examples, seed=DEV_TRANSFORM_SEED, distractor_ood_count=32)
    result = []
    for index, source in enumerate(examples):
        result.append(
            PreparedView(
                source.example_id, "identity", copy.deepcopy(source), ProofGraphTask().render(source)
            )
        )
        for case in cases[index * 5 : (index + 1) * 5]:
            _require(case.source_example_id == source.example_id, "development case order differs")
            result.append(PreparedView(source.example_id, case.transformation, case.example, case.prompt))
    return result


def _manifest_unchecked(splits: dict[str, list[TaskExample]]) -> dict[str, Any]:
    result = {}
    for role, views in (
        ("student_fit", fit_views(splits["student_fit"])),
        ("student_dev", development_views(splits["student_dev"])),
    ):
        bases = splits[role]
        result[role] = dict(
            base_count=len(bases),
            count=len(views),
            examples_sha256=sha256_value([asdict(x) for x in bases]),
            views_sha256=sha256_value([asdict(x) for x in views]),
            ordered_base_ids=[x.example_id for x in bases],
            ordered_ids=[x.example.example_id for x in views],
            pair_seeds=sorted({x.metadata["pair_seed"] for x in bases}),
        )
    return result


def dataset_manifest(splits: dict[str, list[TaskExample]]) -> dict[str, Any]:
    validate_examples(splits)
    result = _manifest_unchecked(splits)
    _require(all(result[k]["views_sha256"] == VIEWS_SHA256[k] for k in result), "view population changed")
    _require(sha256_value(result) == DATASET_MANIFEST_SHA256, "dataset manifest changed")
    return result


def isolation_inventory(splits: dict[str, list[TaskExample]]) -> dict[str, set[Any]]:
    validate_examples(splits)
    rows = [x for values in splits.values() for x in values]
    rows += [v.example for v in fit_views(splits["student_fit"])]
    rows += [v.example for v in development_views(splits["student_dev"])]
    return {
        "semantic": {canonical_semantic_key(x) for x in rows},
        "example_id": {x.example_id for x in rows},
        "pair_group_id": {x.pair_group_id for x in rows},
        "pair_seed": {x.metadata["pair_seed"] for x in rows},
    }


def reject_overlap(inventory: dict[str, set[Any]], row: TaskExample) -> None:
    actual = {
        "semantic": canonical_semantic_key(row),
        "example_id": row.example_id,
        "pair_group_id": row.pair_group_id,
        "pair_seed": row.metadata["pair_seed"],
    }
    for key, value in actual.items():
        _require(value not in inventory[key], f"invariance population overlaps {key}")


def audit_isolation(
    splits: dict[str, list[TaskExample]],
    original_family: Iterable[TaskExample],
    teacher_splits: dict[str, list[TaskExample]],
) -> dict[str, Any]:
    inventory = isolation_inventory(splits)
    _require(set(teacher_splits) == {"teacher_fit", "teacher_dev"}, "teacher isolation roles differ")
    prior = previous.make_examples(DIFFICULTY)
    populations = {
        "original_family": (original_family, 144000),
        "teacher_fit": (teacher_splits["teacher_fit"], 8192),
        "teacher_dev": (teacher_splits["teacher_dev"], 512),
        "previous_student_fit": (prior["student_fit"], 2048),
        "previous_student_dev": (prior["student_dev"], 512),
    }
    counts, digests = {}, {}
    for role, (examples, size) in populations.items():
        count, ids, digest = 0, set(), hashlib.sha256()
        for row in examples:
            reject_overlap(inventory, row)
            _require(row.example_id not in ids, f"duplicate excluded {role} row")
            ids.add(row.example_id)
            digest.update(_canonical(asdict(row)) + b"\n")
            count += 1
        _require(count == size, f"incomplete excluded population {role}")
        counts[role], digests[role] = count, digest.hexdigest()
    return dict(
        passed=True,
        counts=counts,
        population_stream_sha256=digests,
        dimensions=list(inventory),
        preparation_manifest_sha256=sha256_value(dataset_manifest(splits)),
        scientific_acceptance=False,
    )


def encode_view(
    view: PreparedView, tokenizer: Any, model_config: dict[str, Any], *, training: bool = True
) -> dict[str, Any]:
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    task = ProofGraphTask()
    text = render_target(view.example)
    _require(task.verify(view.example, task.parse_response(text)).reward == 1.0, "view target invalid")
    prompt = format_model_prompt(view.prompt, tokenizer, model_config)
    prefix = tokenizer.encode(prompt.model_facing_prompt, add_special_tokens=False)
    target = tokenizer.encode(text, add_special_tokens=False)
    _require(tokenizer.eos_token_id is not None and prefix and target, "prefix/target/EOS missing")
    target = [*target, tokenizer.eos_token_id]
    _require(len(target) <= 256, "target exceeds generation cap")
    if training:
        _require(
            len(prefix) <= MAX_PREFIX_TOKENS and len(prefix) + len(target) <= 1536,
            "training view exceeds1536; truncation forbidden",
        )
    else:
        _require(
            len(prefix) + 256 <= DEVELOPMENT_MAX_MODEL_INPUT_TOKENS,
            "development prefix exceeds2454; truncation forbidden",
        )
    return dict(
        example_id=view.example.example_id,
        input_ids=[*prefix, *target],
        labels=[-100] * len(prefix) + target,
        prefix_length=len(prefix),
        response_length=len(target),
        prompt_sha256=prompt.model_facing_prompt_sha256,
        target_sha256=sha256_value(text),
        eos_token_id=tokenizer.eos_token_id,
    )


def validate_encoded_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _require(
        len(rows) == FIT_COUNT and len({x["example_id"] for x in rows}) == FIT_COUNT,
        "8192 unique fit views required",
    )
    for row in rows:
        ids, labels, prefix, response = (
            row["input_ids"],
            row["labels"],
            row["prefix_length"],
            row["response_length"],
        )
        _require(
            type(prefix) is int
            and 0 < prefix <= MAX_PREFIX_TOKENS
            and type(response) is int
            and 0 < response <= 256
            and len(ids) == len(labels) == prefix + response <= 1536
            and all(type(v) is int and v >= 0 for v in ids)
            and labels[:prefix] == [-100] * prefix
            and labels[prefix:] == ids[prefix:]
            and type(row["eos_token_id"]) is int
            and ids[-1] == row["eos_token_id"],
            "fit token/mask envelope differs",
        )
    windows = [
        sum(len(x["input_ids"]) for x in rows[start : start + 64]) for start in range(0, FIT_COUNT, 64)
    ]
    _require(sum(windows) <= MAX_INPUT_TOKENS, "full preparation exceeds8M budget")
    return dict(
        fit_rows=len(rows),
        optimizer_windows=len(windows),
        window_input_tokens=windows,
        total_input_tokens=sum(windows),
        token_budget=MAX_INPUT_TOKENS,
    )


COUNT_KEYS = {"num_examples", "answer_correct", "proof_correct", "format_valid"}


def _counts(row: Any, count: int) -> None:
    _require(isinstance(row, dict) and set(row) == COUNT_KEYS, "count fields differ")
    _require(all(type(row[k]) is int for k in COUNT_KEYS), "counts must be integers")
    _require(
        row["num_examples"] == count
        and 0 <= row["proof_correct"] <= row["answer_correct"] <= row["format_valid"] <= count,
        "counts inconsistent",
    )


def checkpoint_eligible(row: dict[str, Any]) -> bool:
    groups = row["per_view_counts"]
    iid = groups["identity"]["proof_correct"] / DEV_BASE_COUNT
    transformed = [groups[k]["proof_correct"] / DEV_BASE_COUNT for k in TRANSFORMATIONS]
    return (
        26 <= row["answer_correct"] <= 153
        and iid >= 0.10
        and sum(transformed) / 5 >= 0.08
        and min(transformed) >= 0.05
        and iid - sum(transformed) / 5 <= 0.05
        and all(iid - x <= 0.05 for x in transformed)
        and all(x["proof_correct"] / x["num_examples"] >= 0.10 for x in row["structure_counts"].values())
    )


def select_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    _require(
        isinstance(records, list) and len(records) == len(CHECKPOINT_STEPS), "all six dev reports required"
    )
    fields = {
        "step",
        "checkpoint_sha256",
        "examples_sha256",
        "per_view_counts",
        "structure_counts",
        *COUNT_KEYS,
    }
    bases = _build_bases()["student_dev"]
    structure_sizes = {name: sum(x.metadata["structure"] == name for x in bases) for name in STRUCTURES}
    seen, selected = set(), None
    for step, row in zip(CHECKPOINT_STEPS, records, strict=True):
        _require(
            isinstance(row, dict) and set(row) == fields and type(row["step"]) is int and row["step"] == step,
            "development schedule/fields differ",
        )
        _require(
            isinstance(row["checkpoint_sha256"], str)
            and _SHA256.fullmatch(row["checkpoint_sha256"])
            and row["checkpoint_sha256"] not in seen,
            "checkpoint identity differs",
        )
        seen.add(row["checkpoint_sha256"])
        _require(row["examples_sha256"] == EXAMPLES_SHA256["student_dev"], "dev base identity differs")
        _counts({k: row[k] for k in COUNT_KEYS}, DEV_BASE_COUNT)
        groups = row["per_view_counts"]
        _require(
            isinstance(groups, dict) and set(groups) == set(DEV_VIEW_NAMES), "development view groups differ"
        )
        for group in groups.values():
            _counts(group, DEV_BASE_COUNT)
        _require(groups["identity"] == {k: row[k] for k in COUNT_KEYS}, "identity aggregate differs")
        strata = row["structure_counts"]
        _require(
            isinstance(strata, dict) and set(strata) == set(STRUCTURES), "development structure groups differ"
        )
        for name, counts in strata.items():
            _counts(counts, structure_sizes[name])
        _require(
            all(sum(c[k] for c in strata.values()) == row[k] for k in COUNT_KEYS),
            "structure aggregate differs",
        )
        if selected is None and checkpoint_eligible(row):
            selected = copy.deepcopy(row)
    return selected


@dataclass(frozen=True)
class ResolvedStudentInvarianceProtocol:
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


def resolve_student_invariance_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentInvarianceProtocol:
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
            root / path, context=f"student invariance source {path}", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    _require(_COMMIT.fullmatch(head) and (expected_head is None or head == expected_head), "Git HEAD differs")
    _require(text("rev-parse", "--is-shallow-repository") == "false", "complete Git review history required")
    raw = read(PROTOCOL_PATH)
    payload = load_student_invariance_protocol(raw, require_accepted=require_accepted)
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
        proposed = load_student_invariance_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
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
        reviewed = load_student_invariance_protocol(reviewed_raw, require_accepted=True)
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
        qualification_path = parent["parent_qualification_protocol"]
        qualification_sha = parent["parent_qualification_protocol_sha256"]
        _require(
            hashlib.sha256(read(qualification_path)).hexdigest() == qualification_sha,
            "frozen qualification protocol changed",
        )
        for revision in (PARENT_HEAD, implementation, head):
            _require(
                hashlib.sha256(git("show", f"{revision}:{qualification_path}")).hexdigest()
                == qualification_sha,
                "committed frozen qualification protocol changed",
            )
        _require(
            not git("diff", "--cached", "--name-only", "--", qualification_path),
            "frozen qualification protocol has staged changes",
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
    return ResolvedStudentInvarianceProtocol(
        payload,
        protocol_core_sha256(payload),
        hashlib.sha256(raw).hexdigest(),
        implementation,
        acceptance,
        head,
        hashes,
        clean,
    )
