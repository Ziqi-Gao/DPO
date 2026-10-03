"""Prospective fresh common student with plain and renamed order coverage.

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
    _paraphrased_prompt,
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
from posttrain_circuits.experiments.protocols import student_branch_preparation as previous
from posttrain_circuits.experiments.protocols import student_invariance as invariance_preparation
from posttrain_circuits.experiments.protocols import student_preparation as original_preparation

PROTOCOL_PATH = "prereg/amendments/qwen3_student_order_preparation_v4.json"
PROTOCOL_ID = "qwen3-student-order-preparation-v4"
BASE_REVISION = previous.BASE_REVISION
PARENT_HEAD = "d54749a89836a6b04e18121d561232dddfc0d284"
PARENT_PROTOCOL_SHA256 = "da014c792a5d71e5b967a20904dcdc11df8ae006a1f945899c9dd089a2ff2a94"
FROZEN_PROTOCOL_SHA256 = {**previous.FROZEN_PROTOCOL_SHA256, previous.PROTOCOL_PATH: PARENT_PROTOCOL_SHA256}
RESPONSE_INSTRUCTIONS_SHA256 = previous.RESPONSE_INSTRUCTIONS_SHA256
FIT_BASE_COUNT, DEV_BASE_COUNT = 256, 256
FIT_COUNT, DEV_COUNT = 2048, 1536
FIT_SEED_START, DEV_SEED_START = 190_000_042, 200_000_042
FIT_TRANSFORM_SEED, DEV_TRANSFORM_SEED = 210_000_042, 220_000_042
FIT_VIEW_NAMES = (
    "identity",
    "surface_template_paraphrase",
    "fact_order_permutation",
    "rule_order_permutation",
    "entity_symbol_renaming",
    "rename_fact_order",
    "rename_rule_order",
    "joint_rename_order",
)
DEV_VIEW_NAMES = ("identity", *TRANSFORMATIONS)
STRUCTURES = ("chain", "branch", "converging_dag")
CHECKPOINT_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
MAX_INPUT_TOKENS = 2_000_000
MAX_PREFIX_TOKENS = 1344
DEVELOPMENT_MAX_MODEL_INPUT_TOKENS = 2454
DIFFICULTY = copy.deepcopy(previous.DIFFICULTY)
FIT_DIFFICULTY = {**copy.deepcopy(DIFFICULTY), "distractor_range": [4, 8]}
FIT_PAIR_STRUCTURES = ("branch", "branch", "chain", "converging_dag")
EXAMPLES_SHA256 = {
    "student_fit": "7410a5c43e07a658f653e0dad6cb258a41ef7e0d9d28700b6baddda31cbfeb2f",
    "student_dev": "13e6348c92a65fd572d6018763939093be0fe2420a5e6d7f5713a4dc43db9371",
}
VIEWS_SHA256 = {
    "student_fit": "1a03a781d2033afdfbd2336207b8193eb9f17187604f80dc178ddf0a52e7f3d1",
    "student_dev": "aa776e02c3654c66f2b0bdace7e62b636c70f76c2bc957e63fe6db04390c7753",
}
DATASET_MANIFEST_SHA256 = "ec65903bc92fa34ac40b8434b1c558aedc85758bd32220d7f642520c5479a1dd"
FROZEN_SCIENCE_PATHS = tuple(previous.SCIENCE_PATHS)
SCIENCE_PATHS = (
    *FROZEN_SCIENCE_PATHS,
    "src/posttrain_circuits/experiments/protocols/student_order_preparation.py",
    "tools/sdsc_student_order.py",
    "tools/sdsc_student_order_job.py",
    "tools/sdsc_student_order_worker.py",
    "tools/sdsc_student_order_audit.py",
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
_MAX_BYTES = 4 * 1024 * 1024


class StudentOrderPreparationError(ValueError):
    """Frozen order-preparation contract rejected an input."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentOrderPreparationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def proposed_student_order_preparation_protocol() -> dict[str, Any]:
    value = previous.proposed_student_branch_preparation_protocol()
    value.update(kind="opd_student_order_preparation_protocol", protocol_id=PROTOCOL_ID)
    value["scope"].update(
        purpose="independently_prepared_order_robust_common_1p7b_initial_for_all_methods",
        prior_failed_preparation_job="54615110",
        prior_preparation_development_exposure_disclosed=True,
        historical_checkpoints_reselected=False,
        plain_order_coverage_and_renaming_fraction_are_new_scientific_treatment=True,
        renaming_fraction_changed_from=0.75,
        renaming_fraction_changed_to=0.5,
    )
    value["preserved_base"].update(
        parent_science_head=PARENT_HEAD,
        parent_branch_protocol=previous.PROTOCOL_PATH,
        parent_branch_protocol_sha256=PARENT_PROTOCOL_SHA256,
        frozen_parent_protocol_sha256=dict(FROZEN_PROTOCOL_SHA256),
    )
    value["data"].update(
        fit_raw_pair_seed_start=FIT_SEED_START,
        development_raw_pair_seed_start=DEV_SEED_START,
        fit_transform_seed=FIT_TRANSFORM_SEED,
        development_transform_seed=DEV_TRANSFORM_SEED,
        fit_views=list(FIT_VIEW_NAMES),
        fit_pair_seed_schedule="190000042+pair_index; exactly128_pairs; no_filter_or_replacement",
        fit_renamed_sequence_fraction=0.5,
        fit_plain_fact_permuted_sequence_fraction=0.125,
        fit_plain_rule_permuted_sequence_fraction=0.125,
        fit_total_fact_permuted_sequence_fraction=0.375,
        fit_total_rule_permuted_sequence_fraction=0.375,
        fit_paraphrased_sequence_fraction=0.125,
        fit_transform=(
            "case_seed=210000042+base_index*10000; rename+1; "
            "plain_and_renamed_fact+3/rule+4; joint_rename_fact+3_rule+4"
        ),
        development_transform="original_five_transform_suite_seed220000042_distractors32",
        dataset_manifest_sha256=DATASET_MANIFEST_SHA256,
        examples_sha256=dict(EXAMPLES_SHA256),
        views_sha256=dict(VIEWS_SHA256),
    )
    value["data"]["isolation_population_counts"].update(branch_student_fit=256, branch_student_dev=256)
    value["data"]["isolation_excluded_view_counts"].update(
        branch_student_fit_views=2048, branch_student_dev_views=1536
    )
    value["science_files"] = list(SCIENCE_PATHS)
    value["review_contract"]["acceptance_allowed_changed_paths"] = [
        PROTOCOL_PATH,
        "docs/refactor/current_handoff.md",
    ]
    return value


def validate_student_order_preparation_protocol(
    payload: Any, *, require_accepted: bool = False
) -> dict[str, Any]:
    expected = proposed_student_order_preparation_protocol()
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
            raise StudentOrderPreparationError("invalid review date") from error
    expected["review"] = copy.deepcopy(review)
    try:
        _require(
            _canonical(payload) == _canonical(expected), "order preparation protocol differs from frozen v4"
        )
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentOrderPreparationError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_order_preparation_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
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
        raise StudentOrderPreparationError("invalid protocol JSON") from error
    return validate_student_order_preparation_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_order_preparation_protocol(payload)
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
    fit = []
    for pair_index in range(FIT_BASE_COUNT // 2):
        window, slot = divmod(pair_index, 4)
        seed = FIT_SEED_START + pair_index
        config = dict(FIT_DIFFICULTY, structure=FIT_PAIR_STRUCTURES[slot], depth=2 + (window + slot) % 3)
        pair = list(task.generate_pair(seed, config))
        if int(sha256_value(["train", seed])[:8], 16) % 2:
            pair.reverse()
        fit.extend(pair)
    return {
        "student_fit": fit,
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
        plain_fact, plain_rule = copy.deepcopy(source), copy.deepcopy(source)
        plain_fact.facts = _permuted_mapping(plain_fact.facts, seed + 3)
        plain_rule.rules = _permuted_mapping(plain_rule.rules, seed + 4)
        renamed = _renamed(source, seed + 1)
        fact, rule = copy.deepcopy(renamed), copy.deepcopy(renamed)
        fact.facts = _permuted_mapping(fact.facts, seed + 3)
        rule.rules = _permuted_mapping(rule.rules, seed + 4)
        joint = copy.deepcopy(fact)
        joint.rules = _permuted_mapping(joint.rules, seed + 4)
        variants = list(
            zip(
                FIT_VIEW_NAMES,
                (
                    copy.deepcopy(source),
                    copy.deepcopy(source),
                    plain_fact,
                    plain_rule,
                    renamed,
                    fact,
                    rule,
                    joint,
                ),
                strict=True,
            )
        )
        if source.label == 0:
            variants = variants[1:] + variants[:1]
        for name, example in variants:
            _assert_preserved(source, example)
            prompt = (
                _paraphrased_prompt(example)
                if name == "surface_template_paraphrase"
                else task.render(example)
            )
            example.example_id = f"{source.example_id}-order-preparation-{name}-{sha256_value(prompt)[:12]}"
            example.metadata = {
                **example.metadata,
                "order_preparation_view": name,
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
        _require(value not in inventory[key], f"order preparation population overlaps {key}")


def _audit_previous_views(
    inventory: dict[str, set[Any]],
    invariance: dict[str, list[TaskExample]],
    branch: dict[str, list[TaskExample]],
) -> dict[str, Any]:
    counts, digests = {}, {}
    for role, views, expected in (
        ("invariance_student_fit_views", invariance_preparation.fit_views(invariance["student_fit"]), 8192),
        (
            "invariance_student_dev_views",
            invariance_preparation.development_views(invariance["student_dev"]),
            1536,
        ),
        ("branch_student_fit_views", previous.fit_views(branch["student_fit"]), 2048),
        ("branch_student_dev_views", previous.development_views(branch["student_dev"]), 1536),
    ):
        digest, ids = hashlib.sha256(), set()
        for view in views:
            row = view.example
            reject_overlap(inventory, row)
            _require(row.example_id not in ids, f"duplicate excluded {role} view")
            ids.add(row.example_id)
            digest.update(_canonical(asdict(view)) + b"\n")
        _require(len(views) == expected, f"incomplete excluded view population {role}")
        counts[role], digests[role] = len(views), digest.hexdigest()
    return dict(excluded_view_counts=counts, excluded_view_stream_sha256=digests)


def audit_isolation(
    splits: dict[str, list[TaskExample]],
    original_family: Iterable[TaskExample],
    teacher_splits: dict[str, list[TaskExample]],
) -> dict[str, Any]:
    inventory = isolation_inventory(splits)
    _require(set(teacher_splits) == {"teacher_fit", "teacher_dev"}, "teacher isolation roles differ")
    prior = original_preparation.make_examples(DIFFICULTY)
    invariance = invariance_preparation.make_examples(DIFFICULTY)
    branch = previous.make_examples(DIFFICULTY)
    populations = {
        "original_family": (original_family, 144000),
        "teacher_fit": (teacher_splits["teacher_fit"], 8192),
        "teacher_dev": (teacher_splits["teacher_dev"], 512),
        "previous_student_fit": (prior["student_fit"], 2048),
        "previous_student_dev": (prior["student_dev"], 512),
        "invariance_student_fit": (invariance["student_fit"], 2048),
        "invariance_student_dev": (invariance["student_dev"], 256),
        "branch_student_fit": (branch["student_fit"], 256),
        "branch_student_dev": (branch["student_dev"], 256),
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
        **_audit_previous_views(inventory, invariance, branch),
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
        "2048 unique fit views required",
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
    _require(sum(windows) <= MAX_INPUT_TOKENS, "full preparation exceeds2M budget")
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
        isinstance(records, list) and len(records) == len(CHECKPOINT_STEPS), "all twelve dev reports required"
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
class ResolvedStudentOrderPreparationProtocol:
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


def resolve_student_order_preparation_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentOrderPreparationProtocol:
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
            root / path, context=f"student order preparation source {path}", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    _require(_COMMIT.fullmatch(head) and (expected_head is None or head == expected_head), "Git HEAD differs")
    _require(text("rev-parse", "--is-shallow-repository") == "false", "complete Git review history required")
    raw = read(PROTOCOL_PATH)
    payload = load_student_order_preparation_protocol(raw, require_accepted=require_accepted)
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
        proposed = load_student_order_preparation_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
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
        reviewed = load_student_order_preparation_protocol(reviewed_raw, require_accepted=True)
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
        _require(
            text("merge-base", PARENT_HEAD, implementation) == PARENT_HEAD, "original student lineage missing"
        )
        for path in FROZEN_SCIENCE_PATHS:
            _require(
                git("show", f"{PARENT_HEAD}:{path}") == read(path),
                f"frozen parent student science changed: {path}",
            )
        for path, expected_sha in FROZEN_PROTOCOL_SHA256.items():
            _require(
                hashlib.sha256(read(path)).hexdigest() == expected_sha,
                f"frozen parent protocol changed: {path}",
            )
            for revision in (PARENT_HEAD, implementation, head):
                _require(
                    hashlib.sha256(git("show", f"{revision}:{path}")).hexdigest() == expected_sha,
                    f"committed frozen parent protocol changed: {path}",
                )
            _require(
                not git("diff", "--cached", "--name-only", "--", path),
                "frozen parent protocol has staged changes",
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
    return ResolvedStudentOrderPreparationProtocol(
        payload,
        protocol_core_sha256(payload),
        hashlib.sha256(raw).hexdigest(),
        implementation,
        acceptance,
        head,
        hashes,
        clean,
    )
