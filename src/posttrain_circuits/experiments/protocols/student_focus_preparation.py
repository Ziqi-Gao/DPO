"""Fresh common-initial preparation emphasizing underlearned branch structure.

Only the fit structure proportions change relative to the anchored control.
All original complete-run development selection and formal gates remain intact.
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
from posttrain_circuits.artifacts.teacher_adaptation_protocol import BASE_PREREG_SHA256, REVIEW_PROPOSED
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
from posttrain_circuits.experiments.protocols import student_anchor_probe as anchor_diagnostic
from posttrain_circuits.experiments.protocols import student_batch_probe as previous
from posttrain_circuits.experiments.protocols import student_order_preparation as preparation
from posttrain_circuits.experiments.protocols import student_order_probe as order_diagnostic

PROTOCOL_PATH = "prereg/amendments/qwen3_student_focus_preparation_v5.json"
PROTOCOL_ID = "qwen3-student-focus-preparation-v5"
PARENT_HEAD = "07d5813b9319ac401f09ce9847d564f841f80a50"
PARENT_PROTOCOL_SHA256 = "5c5f4bd45e0902083151eb966904db0e7588e20bedff3766ac00a9acfbb5594d"
BASE_REVISION = previous.BASE_REVISION
CHAT_TEMPLATE_SHA256 = previous.CHAT_TEMPLATE_SHA256
TOKENIZER_FINGERPRINT = previous.TOKENIZER_FINGERPRINT
RESPONSE_INSTRUCTIONS_SHA256 = previous.RESPONSE_INSTRUCTIONS_SHA256
FROZEN_SCIENCE_PATHS = previous.SCIENCE_PATHS
FROZEN_PROTOCOL_SHA256 = {**previous.FROZEN_PROTOCOL_SHA256, previous.PROTOCOL_PATH: PARENT_PROTOCOL_SHA256}
FROZEN_HELPER_SHA256 = {
    "tools/sdsc_cuda_diagnostic_worker.py": (
        "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
    ),
    "tools/sdsc_student_branch_qualify_job.py": (
        "b09059327ec7aa8ec9e01d39408959e1a2e8d85a87e25f2b9e55ab94442751f3"
    ),
    "tools/sdsc_student_contract.py": "7d134403f94e34c13eab4c779f28b828a96b1cd8e4ae7db93c8e34f17c5c5bf2",
    "tools/sdsc_student_initial.py": "e43651ba4b50c176c31490f70a2ffee90d96387853f3a87e0d4382353c3c8908",
    "tools/sdsc_student_job.py": "24629c79dfc986345242dc98624a99afc172165c5587bfd30f4af7b0b9a2d65d",
    "tools/sdsc_student_lr.py": "edf9d6f65fca8b4e345c28b686b804c71c49cf1034b8f352fb923e50eebb3c9e",
    "tools/sdsc_student_lr_job.py": "2814587738ddf6cdde343eff3277f96e165ed7332f2e097ac86752441c524d17",
    "tools/sdsc_student_memory.py": "9f58dfe5cb5cad423ff0c2a1ae2ef637dd89ce38ed00df42e04de3d887791214",
    "tools/sdsc_student_order_job.py": "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
    "tools/sdsc_student_quality.py": "5fcf65d7d999d886067ee5acb898364c696850b5bd778ac6e95976ae91e51e50",
    "tools/sdsc_torch_import_probe_job.py": (
        "22b545b7b1b1a0f710388838ab9491abdf4e93b5554239f61bb441986775a866"
    ),
}
SCIENCE_PATHS = (
    *FROZEN_SCIENCE_PATHS,
    "src/posttrain_circuits/experiments/protocols/student_focus_preparation.py",
    "tools/sdsc_student_focus.py",
    "tools/sdsc_student_focus_job.py",
    "tools/sdsc_student_focus_worker.py",
    "tools/sdsc_student_focus_audit.py",
    "tools/sdsc_student_focus_startup.py",
)
ANCHOR_VIEW_NAMES = ("identity", "entity_symbol_renaming")
FIT_BASE_COUNT, DEV_BASE_COUNT = 256, 256
FIT_COUNT, DEV_COUNT, TRAINING_COUNT = 2048, 1536, 2048
FIT_SEED_START, DEV_SEED_START = 390_000_042, 400_000_042
FIT_TRANSFORM_SEED, DEV_TRANSFORM_SEED = 410_000_042, 420_000_042
FIT_VIEW_NAMES = previous.FIT_VIEW_NAMES
DEV_VIEW_NAMES = previous.DEV_VIEW_NAMES
STRUCTURES = previous.STRUCTURES
CHECKPOINT_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
MAX_INPUT_TOKENS = previous.MAX_INPUT_TOKENS
MAX_PREFIX_TOKENS = previous.MAX_PREFIX_TOKENS
DEVELOPMENT_MAX_MODEL_INPUT_TOKENS = previous.DEVELOPMENT_MAX_MODEL_INPUT_TOKENS
DIFFICULTY = copy.deepcopy(previous.DIFFICULTY)
FIT_DIFFICULTY = copy.deepcopy(previous.FIT_DIFFICULTY)
# Three branch pairs per window; the fourth alternates chain/DAG by window.
FIT_PAIR_STRUCTURES = ("branch", "branch", "branch", "alternating_chain_dag")
EXAMPLES_SHA256 = {
    "student_dev": "b449c1a55b398ea43208cbbc58c25048f0a174a3e73348124133d43eb9e50c38",
    "student_fit": "ec7da988d37e7ff3eb06955da9b6f4d94ac28604b808b826cf80599c763c2327",
}
VIEWS_SHA256 = {
    "student_dev": "9031e3c94bcdb9af4b9e8050ba71159862607a5fb303cf1c91827e16dff9cc8e",
    "student_fit": "4588e212d87ad34430d681b712c8f1abe03cbed26ca13b8fa3d2835da2a8d44e",
}
DATASET_MANIFEST_SHA256 = "18d7e8456e03f1ef274000164460f79882e6da1182a8c81f6d5bd7c016b6ce26"
COUNT_KEYS = {"num_examples", "answer_correct", "proof_correct", "format_valid"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_MAX_BYTES = 4 * 1024**2


class StudentFocusPreparationError(ValueError):
    """The diagnostic contract rejected inconsistent or unreviewed evidence."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentFocusPreparationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def proposed_student_focus_preparation_protocol() -> dict[str, Any]:
    value = preparation.proposed_student_order_preparation_protocol()
    value.update(kind="opd_student_focus_preparation_protocol", protocol_id=PROTOCOL_ID)
    value["scope"] = dict(
        purpose="fresh_branch_focused_common_1p7b_initial_for_all_methods",
        execution="sdsc_h100_preparation_only",
        interpretation="post_preparation_capability_not_native_instruction_capability",
        prior_failed_qualification_job="54606205",
        prior_failed_preparation_job="54643699",
        prior_batch_diagnostic_jobs={"control": "54659007", "treatment": "54659010"},
        prior_preparation_and_diagnostic_exposure_disclosed=True,
        prior_formal_anti_shortcut_exposure_disclosed=True,
        original_failed_initial_is_preserved=True,
        historical_checkpoints_reselected=False,
        sole_training_policy_change_relative_to_anchored_control="fit_structure_sampling_50_25_25_to75_12p5_12p5",
        branch_focus_is_new_scientific_treatment=True,
        effect_or_renaming_improvement_not_established_before_run=True,
        original_family_read_only_for_exclusion_keys=True,
        formal_generated_responses_or_scores_read=False,
        teacher_generated_targets=False,
        preparation_is_student_or_g0_acceptance=False,
        downstream_calibration_g0_pilot_in_scope=False,
    )
    value["preserved_base"].update(
        parent_science_head=PARENT_HEAD,
        parent_batch_probe_protocol=previous.PROTOCOL_PATH,
        parent_batch_probe_protocol_sha256=PARENT_PROTOCOL_SHA256,
        frozen_parent_protocol_sha256=dict(FROZEN_PROTOCOL_SHA256),
    )
    data = value["data"]
    for key in (
        "fit_plain_fact_permuted_sequence_fraction",
        "fit_plain_rule_permuted_sequence_fraction",
        "fit_total_fact_permuted_sequence_fraction",
        "fit_total_rule_permuted_sequence_fraction",
        "fit_per_rank_window_structure_counts",
        "fit_per_window_structure_counts",
    ):
        data.pop(key)
    data.update(
        fit_raw_pair_seed_start=FIT_SEED_START,
        development_raw_pair_seed_start=DEV_SEED_START,
        fit_transform_seed=FIT_TRANSFORM_SEED,
        development_transform_seed=DEV_TRANSFORM_SEED,
        fit_pair_seed_schedule="390000042+pair_index;exactly128_pairs;no_filter_or_replacement",
        fit_structure_schedule="pair_index=4*window+slot;slot0..2_branch;slot3_chain_if_window_even_else_converging_dag;zero_based_indices",
        fit_window="four_signed_pairs;three_branch_one_alternating_chain_DAG;eight_views_per_base;global64",
        fit_structure_sequence_fractions={"branch": 0.75, "chain": 0.125, "converging_dag": 0.125},
        fit_per_window_structure_counts_by_parity={
            "even": {"branch": 48, "chain": 16, "converging_dag": 0},
            "odd": {"branch": 48, "chain": 0, "converging_dag": 16},
        },
        fit_per_rank_window_structure_counts_by_parity={
            "even": {"branch": 24, "chain": 8, "converging_dag": 0},
            "odd": {"branch": 24, "chain": 0, "converging_dag": 8},
        },
        fit_transform="case_seed410000042+base_index*10000;rename+1;fact+3;rule+4;independent_joint_fact+100+2*declared_view_index_rule+101+2*declared_view_index_except_identity_and_entity_symbol_renaming",
        fit_original_order_anchor_sequence_fraction=0.25,
        fit_joint_permutation_sequence_fraction=0.75,
        fit_fact_permutation_applied_sequence_fraction=0.75,
        fit_rule_permutation_applied_sequence_fraction=0.75,
        composed_permutations_may_return_to_original_order=True,
        measured_fit_net_order_changed_rows={"facts": 1534, "rules": 1536},
        development_transform="original_five_transform_suite_seed420000042_distractors32",
        dataset_manifest_sha256=DATASET_MANIFEST_SHA256,
        examples_sha256=dict(EXAMPLES_SHA256),
        views_sha256=dict(VIEWS_SHA256),
        isolation_population_counts={
            **previous.proposed_student_batch_probe_protocol()["data"]["isolation_population_counts"],
            "batch_probe_student_fit": 256,
            "batch_probe_student_dev": 128,
        },
        isolation_excluded_view_counts={
            **previous.proposed_student_batch_probe_protocol()["data"]["isolation_excluded_view_counts"],
            "batch_probe_control_student_fit_views": 2048,
            "batch_probe_treatment_student_fit_views": 2048,
            "batch_probe_student_dev_views": 768,
        },
    )
    value["development"].update(expected_raw_responses=18432)
    value["execution"].update(
        early_startup_timeout_seconds=300,
        startup_thread_count=12,
        flat_publication_binds_execution_and_science=True,
        each_checkpoint_fixed2454_token_native_BF16_finite_forward_each_rank=True,
    )
    value["science_files"] = list(SCIENCE_PATHS)
    value["frozen_helpers"] = dict(FROZEN_HELPER_SHA256)
    value["review_contract"]["acceptance_allowed_changed_paths"] = [
        PROTOCOL_PATH,
        "docs/refactor/current_handoff.md",
    ]
    return value


def validate_student_focus_preparation_protocol(
    payload: Any, *, require_accepted: bool = False
) -> dict[str, Any]:
    expected = proposed_student_focus_preparation_protocol()
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
            raise StudentFocusPreparationError("invalid review date") from error
    expected["review"] = copy.deepcopy(review)
    try:
        _require(
            _canonical(payload) == _canonical(expected), "focus preparation protocol differs from frozen v5"
        )
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentFocusPreparationError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_focus_preparation_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
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
        raise StudentFocusPreparationError("invalid protocol JSON") from error
    return validate_student_focus_preparation_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_focus_preparation_protocol(payload)
    checked.pop("review")
    return hashlib.sha256(_canonical(checked)).hexdigest()


@dataclass(frozen=True)
class PreparedView:
    source_example_id: str
    view: str
    example: TaskExample
    prompt: str


def _build_bases() -> dict[str, list[TaskExample]]:
    task, fit = ProofGraphTask(), []
    for pair_index in range(FIT_BASE_COUNT // 2):
        window, slot = divmod(pair_index, 4)
        seed = FIT_SEED_START + pair_index
        config = dict(
            FIT_DIFFICULTY,
            structure=("branch" if slot < 3 else ("chain" if window % 2 == 0 else "converging_dag")),
            depth=2 + (window + slot) % 3,
        )
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
    result = _build_bases()
    validate_examples(result)
    return result


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
        pairs = {}
        for row in rows:
            seed = row.metadata.get("pair_seed")
            _require(type(seed) is int and start <= seed < start + count * 100, "base seed namespace differs")
            pairs.setdefault(row.pair_group_id, []).append(row)
            _require(
                task.verify(row, task.parse_response(render_target(row))).reward == 1.0,
                "invalid canonical base target",
            )
        _require(len(pairs) == count // 2, "base pair count differs")
        _require(
            all(
                len(pair) == 2
                and {x.label for x in pair} == {0, 1}
                and len({x.metadata["pair_seed"] for x in pair}) == 1
                for pair in pairs.values()
            ),
            "incomplete signed pair",
        )


def fit_views(examples: list[TaskExample]) -> list[PreparedView]:
    task, result = ProofGraphTask(), []
    for index, source in enumerate(examples):
        seed = FIT_TRANSFORM_SEED + index * 10000
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
            if name not in ANCHOR_VIEW_NAMES:
                offset = 100 + 2 * FIT_VIEW_NAMES.index(name)
                example.facts = _permuted_mapping(example.facts, seed + offset)
                example.rules = _permuted_mapping(example.rules, seed + offset + 1)
            _assert_preserved(source, example)
            prompt = (
                _paraphrased_prompt(example)
                if name == "surface_template_paraphrase"
                else task.render(example)
            )
            example.example_id = f"{source.example_id}-focus-v5-{name}-{sha256_value(prompt)[:12]}"
            example.metadata = {
                **example.metadata,
                "focus_preparation_view": name,
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
    keys = dict(
        semantic=canonical_semantic_key(row),
        example_id=row.example_id,
        pair_group_id=row.pair_group_id,
        pair_seed=row.metadata["pair_seed"],
    )
    for key, value in keys.items():
        _require(value not in inventory[key], f"focus preparation population overlaps {key}")


def audit_isolation(
    splits: dict[str, list[TaskExample]],
    original_family: Iterable[TaskExample],
    teacher_splits: dict[str, list[TaskExample]],
) -> dict[str, Any]:
    inventory = isolation_inventory(splits)
    _require(set(teacher_splits) == {"teacher_fit", "teacher_dev"}, "teacher isolation roles differ")
    prior = preparation.original_preparation.make_examples(DIFFICULTY)
    invariance = preparation.invariance_preparation.make_examples(DIFFICULTY)
    branch = preparation.previous.make_examples(DIFFICULTY)
    order = preparation.make_examples(DIFFICULTY)
    order_probe = order_diagnostic.make_examples(DIFFICULTY)
    anchor_probe = anchor_diagnostic.make_examples(DIFFICULTY)
    batch_probe = previous.make_examples(DIFFICULTY)
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
        "order_student_fit": (order["student_fit"], 256),
        "order_student_dev": (order["student_dev"], 256),
        "order_probe_student_fit": (order_probe["student_fit"], 256),
        "order_probe_student_dev": (order_probe["student_dev"], 128),
        "anchor_probe_student_fit": (anchor_probe["student_fit"], 256),
        "anchor_probe_student_dev": (anchor_probe["student_dev"], 128),
        "batch_probe_student_fit": (batch_probe["student_fit"], 256),
        "batch_probe_student_dev": (batch_probe["student_dev"], 128),
    }
    counts, digests = {}, {}
    for role, (examples, size) in populations.items():
        ids, digest = set(), hashlib.sha256()
        for row in examples:
            reject_overlap(inventory, row)
            _require(row.example_id not in ids, f"duplicate excluded {role} row")
            ids.add(row.example_id)
            digest.update(_canonical(asdict(row)) + b"\n")
        _require(len(ids) == size, f"incomplete excluded population {role}")
        counts[role], digests[role] = len(ids), digest.hexdigest()
    view_counts, view_digests = {}, {}
    view_populations = []
    for prefix, module, data, fit_n, dev_n in (
        ("invariance", preparation.invariance_preparation, invariance, 8192, 1536),
        ("branch", preparation.previous, branch, 2048, 1536),
        ("order", preparation, order, 2048, 1536),
    ):
        view_populations.extend(
            [
                (f"{prefix}_student_fit_views", module.fit_views(data["student_fit"]), fit_n),
                (f"{prefix}_student_dev_views", module.development_views(data["student_dev"]), dev_n),
            ]
        )
    for arm in previous.ARMS:
        view_populations.append(
            (
                f"order_probe_{arm}_student_fit_views",
                order_diagnostic.fit_views(order_probe["student_fit"], arm=arm),
                2048,
            )
        )
    view_populations.append(
        (
            "order_probe_student_dev_views",
            order_diagnostic.development_views(order_probe["student_dev"]),
            768,
        )
    )
    for arm in previous.ARMS:
        view_populations.append(
            (
                f"anchor_probe_{arm}_student_fit_views",
                anchor_diagnostic.fit_views(anchor_probe["student_fit"], arm=arm),
                2048,
            )
        )
    view_populations.append(
        (
            "anchor_probe_student_dev_views",
            anchor_diagnostic.development_views(anchor_probe["student_dev"]),
            768,
        )
    )
    for arm in previous.ARMS:
        view_populations.append(
            (
                f"batch_probe_{arm}_student_fit_views",
                previous.fit_views(batch_probe["student_fit"], arm=arm),
                2048,
            )
        )
    view_populations.append(
        ("batch_probe_student_dev_views", previous.development_views(batch_probe["student_dev"]), 768)
    )
    for name, views, expected in view_populations:
        ids = set()
        digest = hashlib.sha256()
        for view in views:
            reject_overlap(inventory, view.example)
            _require(view.example.example_id not in ids, f"duplicate excluded {name} view")
            ids.add(view.example.example_id)
            digest.update(_canonical(asdict(view)) + b"\n")
        _require(len(ids) == expected, f"incomplete excluded view population {name}")
        view_counts[name], view_digests[name] = len(ids), digest.hexdigest()
    return dict(
        passed=True,
        counts=counts,
        population_stream_sha256=digests,
        excluded_view_counts=view_counts,
        excluded_view_stream_sha256=view_digests,
        dimensions=list(inventory),
        dataset_manifest_sha256=sha256_value(dataset_manifest(splits)),
        scientific_acceptance=False,
        formal_generated_responses_or_scores_read=False,
    )


def encode_view(
    view: PreparedView, tokenizer: Any, model_config: dict[str, Any], *, training: bool = True
) -> dict[str, Any]:
    try:
        return preparation.encode_view(view, tokenizer, model_config, training=training)
    except preparation.StudentOrderPreparationError as error:
        raise StudentFocusPreparationError(str(error)) from error


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
class ResolvedStudentFocusPreparationProtocol:
    payload: dict[str, Any]
    protocol_sha256: str
    artifact_sha256: str
    implementation_commit: str | None
    acceptance_commit: str | None
    head: str
    science_file_sha256: dict[str, str]
    tracked_worktree_clean: bool
    frozen_helper_sha256: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def science_implementation_sha256(self) -> str:
        return sha256_value(self.science_file_sha256)

    @property
    def review_status(self) -> str:
        return self.payload["review"]["status"]


def resolve_student_focus_preparation_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentFocusPreparationProtocol:
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
            root / path, context=f"student focus preparation source {path}", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    _require(_COMMIT.fullmatch(head) and (expected_head is None or head == expected_head), "Git HEAD differs")
    _require(text("rev-parse", "--is-shallow-repository") == "false", "complete Git review history required")
    raw = read(PROTOCOL_PATH)
    payload = load_student_focus_preparation_protocol(raw, require_accepted=require_accepted)
    _require(
        hashlib.sha256(read("prereg/qwen3_v2.yaml")).hexdigest() == BASE_PREREG_SHA256,
        "original frozen preregistration changed",
    )
    hashes = {path: hashlib.sha256(read(path)).hexdigest() for path in SCIENCE_PATHS}
    helpers = {path: hashlib.sha256(read(path)).hexdigest() for path in FROZEN_HELPER_SHA256}
    _require(helpers == FROZEN_HELPER_SHA256, "frozen diagnostic helper changed")
    implementation, acceptance = payload["review"]["reviewed_implementation_commit"], None
    if payload["review"]["status"] == "accepted":
        _require(
            implementation != head and text("merge-base", implementation, head) == implementation,
            "review implementation is not a distinct ancestor",
        )
        proposed = load_student_focus_preparation_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
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
        reviewed = load_student_focus_preparation_protocol(reviewed_raw, require_accepted=True)
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
                *FROZEN_HELPER_SHA256,
            ),
            "named scientific files have staged changes",
        )
        for path, current_sha in {**hashes, **helpers}.items():
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
    for path, expected_sha in {**hashes, **helpers}.items():
        _require(
            hashlib.sha256(read(path)).hexdigest() == expected_sha,
            f"source changed during validation: {path}",
        )
    return ResolvedStudentFocusPreparationProtocol(
        payload,
        protocol_core_sha256(payload),
        hashlib.sha256(raw).hexdigest(),
        implementation,
        acceptance,
        head,
        hashes,
        clean,
        helpers,
    )
