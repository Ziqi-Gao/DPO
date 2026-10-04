"""Prospective paired diagnostic of original-order anchor coverage, never model selection.

Both arms use fresh shared isolated task bases and fresh native1.7B. Treatment
restores two named original-order slots while control jointly permutes every slot.
Formal outputs and historical models are unused.
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
from posttrain_circuits.experiments.protocols import student_order_preparation as preparation
from posttrain_circuits.experiments.protocols import student_order_probe as previous

PROTOCOL_PATH = "prereg/amendments/qwen3_student_anchor_probe_v1.json"
PROTOCOL_ID = "qwen3-student-anchor-probe-v1"
PARENT_HEAD = "ebb1fc5c25ac39ce536b51ef4b699af3247699d9"
PARENT_PROTOCOL_SHA256 = "2bb8f3c9993eea7dc6a760639d0607f61896f5fd22599edd3f76bc268ac2d3e0"
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
    "src/posttrain_circuits/experiments/protocols/student_anchor_probe.py",
    "tools/sdsc_student_anchor_probe.py",
    "tools/sdsc_student_anchor_probe_job.py",
    "tools/sdsc_student_anchor_probe_worker.py",
    "tools/sdsc_student_anchor_probe_audit.py",
    "tools/sdsc_student_anchor_probe_startup.py",
)
ARMS = ("control", "treatment")
ANCHOR_VIEW_NAMES = ("identity", "entity_symbol_renaming")
FIT_BASE_COUNT, DEV_BASE_COUNT = 256, 128
FIT_COUNT, DEV_COUNT, TRAINING_COUNT = 2048, 768, 768
FIT_SEED_START, DEV_SEED_START = 310_000_042, 320_000_042
FIT_TRANSFORM_SEED, DEV_TRANSFORM_SEED = 330_000_042, 340_000_042
FIT_VIEW_NAMES = previous.FIT_VIEW_NAMES
DEV_VIEW_NAMES = previous.DEV_VIEW_NAMES
STRUCTURES = previous.STRUCTURES
CHECKPOINT_STEPS = (4, 6, 7, 8, 12)
MAX_INPUT_TOKENS = previous.MAX_INPUT_TOKENS
MAX_PREFIX_TOKENS = previous.MAX_PREFIX_TOKENS
DEVELOPMENT_MAX_MODEL_INPUT_TOKENS = previous.DEVELOPMENT_MAX_MODEL_INPUT_TOKENS
DIFFICULTY = copy.deepcopy(previous.DIFFICULTY)
FIT_DIFFICULTY = copy.deepcopy(previous.FIT_DIFFICULTY)
FIT_PAIR_STRUCTURES = previous.FIT_PAIR_STRUCTURES
EXAMPLES_SHA256 = {
    "student_fit": "66c015c32f55eb631efa377e50fff8ece628ca5ac9e64906b29e1949888c98f7",
    "student_dev": "93ecdd7cd6a71d34a3d21ab6bc655a57784dab9752abe6610b7e92e68a346b8e",
}
VIEWS_SHA256 = {
    "control": {
        "student_fit": "34264f0fd2f5a5a0839863e4790e022c5e3c02a0a31a74c3ba6ca9bb59f29378",
        "student_dev": "6a665e96bc295767b57db586457b9b357339c107422dd6c511a31c35fb101f5a",
    },
    "treatment": {
        "student_fit": "ec939998515d85ebba06e62b86d21c598d308c3d6f4e45678963adffebc9c356",
        "student_dev": "6a665e96bc295767b57db586457b9b357339c107422dd6c511a31c35fb101f5a",
    },
}
ARM_MANIFEST_SHA256 = {
    "control": "ad80de92dd8d3a9f63f87329644e675edd9e68e838c1737500b360b153375cf6",
    "treatment": "0230241416ff424430f1e937f3d136baba678891f8d426a423cf2ed3f0571d31",
}
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_MAX_BYTES = 4 * 1024**2


class StudentAnchorProbeError(ValueError):
    """The diagnostic contract rejected inconsistent or unreviewed evidence."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentAnchorProbeError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def proposed_student_anchor_probe_protocol() -> dict[str, Any]:
    value = previous.proposed_student_order_probe_protocol()
    value.update(kind="opd_student_anchor_probe_protocol", protocol_id=PROTOCOL_ID)
    value["scope"] = dict(
        purpose="paired_single_factor_original_order_anchor_diagnostic",
        execution="sdsc_h100_probe_only",
        arms=list(ARMS),
        prior_failed_preparation_job="54643699",
        prior_order_diagnostic_jobs={"control": "54648255", "treatment": "54648257"},
        prior_order_diagnostic_exposure_disclosed=True,
        prior_preparation_development_exposure_disclosed=True,
        prior_formal_anti_shortcut_exposure_disclosed=True,
        historical_checkpoints_reselected=False,
        teacher_generated_targets=False,
        diagnostic_is_model_or_preparation_acceptance=False,
        downstream_calibration_g0_pilot_in_scope=False,
        formal_generated_responses_or_scores_read=False,
        original_family_read_only_for_exclusion_keys=True,
        same_fit_dev_bases_labels_targets_and_renaming_both_arms=True,
        sole_treatment="skip_joint_permutation_only_identity_and_entity_symbol_renaming_slots",
        intervention_does_not_identify_coverage_vs_difficulty_vs_order_cue_mechanism=True,
    )
    value["preserved_base"].update(
        parent_science_head=PARENT_HEAD,
        parent_order_probe_protocol=previous.PROTOCOL_PATH,
        parent_order_probe_protocol_sha256=PARENT_PROTOCOL_SHA256,
        frozen_parent_protocol_sha256=dict(FROZEN_PROTOCOL_SHA256),
    )
    data = value["data"]
    for key in [
        "dataset_manifest_sha256",
        "fit_plain_fact_permuted_sequence_fraction",
        "fit_plain_rule_permuted_sequence_fraction",
        "fit_total_fact_permuted_sequence_fraction",
        "fit_total_rule_permuted_sequence_fraction",
    ]:
        data.pop(key, None)
    data.update(
        fit_base_examples=FIT_BASE_COUNT,
        development_base_examples=DEV_BASE_COUNT,
        fit_examples=FIT_COUNT,
        development_examples=DEV_COUNT,
        training_examples=TRAINING_COUNT,
        fit_raw_pair_seed_start=FIT_SEED_START,
        development_raw_pair_seed_start=DEV_SEED_START,
        fit_transform_seed=FIT_TRANSFORM_SEED,
        development_transform_seed=DEV_TRANSFORM_SEED,
        fit_pair_seed_schedule="310000042+pair_index; exactly128_pairs; no_filter_or_replacement",
        fit_transform=(
            "same_V4_variants_case_seed330000042+base_index*10000;rename+1;fact+3;rule+4;"
            "control_then_independent_joint_fact+100+2*declared_view_index_rule+101+2*declared_view_index"
        ),
        treatment_transform=(
            "same_control_algorithm_except_skip_joint_step_for_identity_and_entity_symbol_renaming;"
            "remaining_six_model_inputs_targets_and_masks_equal_control"
        ),
        development_transform="original_five_transform_suite_seed340000042_distractors32",
        arm_manifest_sha256=dict(ARM_MANIFEST_SHA256),
        examples_sha256=dict(EXAMPLES_SHA256),
        views_sha256=copy.deepcopy(VIEWS_SHA256),
        training_prefix="first768_of2048_rows;12_complete_global64_windows;no_adaptive_extension",
        all_2048_rows_and_both_arms_audited_before_first_update=True,
        anchor_view_names=list(ANCHOR_VIEW_NAMES),
        arm_original_order_anchor_sequence_fraction={"control": 0.0, "treatment": 0.25},
        arm_joint_permutation_sequence_fraction={"control": 1.0, "treatment": 0.75},
    )
    data["isolation_population_counts"].update(order_probe_student_fit=256, order_probe_student_dev=128)
    data["isolation_excluded_view_counts"].update(
        order_probe_control_student_fit_views=2048,
        order_probe_treatment_student_fit_views=2048,
        order_probe_student_dev_views=768,
    )
    value["training"].pop("complete_all_32_steps_even_if_candidate_qualifies", None)
    value["training"].update(
        optimizer_steps=12,
        checkpoint_steps=list(CHECKPOINT_STEPS),
        complete_all_12_steps_even_if_observed_rates_change=True,
        budget_scope="entire2048_row_population_must_fit2M;execute_only_fixed768_prefix",
        fresh_native_weights_optimizer_rng_cursors_each_arm=True,
    )
    value["development"] = dict(
        base_examples=DEV_BASE_COUNT,
        all_examples=DEV_COUNT,
        views=list(DEV_VIEW_NAMES),
        structure_groups=list(STRUCTURES),
        checkpoint_steps=list(CHECKPOINT_STEPS),
        expected_raw_responses_per_arm=3840,
        all_five_full_development_reports_required=True,
        generation="greedy_native_BF16_without_extra_autocast",
        max_new_tokens=256,
        max_model_input_tokens=DEVELOPMENT_MAX_MODEL_INPUT_TOKENS,
        metric="original_VerificationResult.answer_correct_and_complete_proof_reward",
        rank_partition="complete_development_base_blocks_round_robin_rank",
        seed="42_plus_manifest_index",
        report="raw_counts_rates_iid_minus_each_and_mean_gaps_per_structure_and_view",
        diagnostic_completion_is_independent_of_measured_rates=True,
        eligibility_selection_or_candidate=False,
        formal_thresholds_applied_to_probe_population=False,
        formal_896_responses_and_scores_are_not_inputs=True,
    )
    value["execution"] = dict(
        gpu_count=2,
        cpu_count=24,
        host_memory_gib=384,
        walltime_minutes=120,
        maximum_concurrently_allocatable_gpus=4,
        full_state_restore_step=4,
        exact_fp32_saved_master_reload_and_native_bf16_parity_each_checkpoint=True,
        fixed2454_token_native_bf16_finite_forward_each_rank=True,
        minimum_headroom_fraction=0.2,
        minimum_headroom_gib=32,
        minimum_node_local_free_gib=192,
        maximum_persistent_large_gib=128,
        maximum_small_fetch_mib=224,
        maximum_prompt_file_mib=32,
        maximum_other_small_file_mib=16,
        publication_reserve_seconds=600,
        early_startup_whole_child_seconds=300,
        early_thread_environment={
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "12",
            "OPENBLAS_NUM_THREADS": "12",
        },
        independent_raw_audit_required=True,
        no_automatic_retry=True,
        unique_per_protocol_arm_claim=True,
        blackwell_execution_class_reused=False,
    )
    value["analysis"] = dict(
        complete_reporting="both_arms_all_five_steps_all_six_views_and_each_structure",
        primary_steps=[6, 7, 8],
        primary_view_names=["entity_symbol_renaming", "fact_order_permutation", "rule_order_permutation"],
        primary_descriptive_contrasts="separate_mean_over_primary_steps_of_treatment_minus_control_absolute_proof_rate_for_each_primary_view",
        primary_vector_must_not_be_summed_or_averaged_across_views=True,
        expected_direction="rename_absolute_gain_with_fact_and_rule_absolute_rates_reported_separately",
        retention_is_not_a_noninferiority_or_acceptance_test=True,
        required_companion_metrics="all_absolute_answer_proof_format_rates_and_paired_per_example_wins_losses",
        signed_gap_decomposition="for_each_view_delta_IID_minus_view_gap_equals_delta_IID_absolute_rate_minus_delta_view_absolute_rate",
        lowering_IID_alone_is_not_improvement=True,
        steps4_and12_and_other_views_are_fully_reported=True,
        interpretation="two_slot_anchor_policy_effect_only;coverage_compound_difficulty_and_reintroduced_order_cues_not_separately_identified",
        significance_or_acceptance_threshold=None,
        checkpoint_selection=False,
        adaptive_extension=False,
        automatic_retry=False,
    )
    value["science_files"] = list(SCIENCE_PATHS)
    value["frozen_helpers"] = dict(FROZEN_HELPER_SHA256)
    value["review_contract"]["acceptance_allowed_changed_paths"] = [
        PROTOCOL_PATH,
        "docs/refactor/current_handoff.md",
    ]
    return value


def validate_student_anchor_probe_protocol(payload: Any, *, require_accepted: bool = False) -> dict[str, Any]:
    expected = proposed_student_anchor_probe_protocol()
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
            raise StudentAnchorProbeError("invalid review date") from error
    expected["review"] = copy.deepcopy(review)
    try:
        _require(_canonical(payload) == _canonical(expected), "anchor probe protocol differs from frozen v1")
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentAnchorProbeError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_anchor_probe_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
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
        raise StudentAnchorProbeError("invalid protocol JSON") from error
    return validate_student_anchor_probe_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_anchor_probe_protocol(payload)
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


def fit_views(examples: list[TaskExample], arm: str = "control") -> list[PreparedView]:
    _require(arm in ARMS, "unknown diagnostic arm")
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
            if arm == "control" or name not in ANCHOR_VIEW_NAMES:
                offset = 100 + 2 * FIT_VIEW_NAMES.index(name)
                example.facts = _permuted_mapping(example.facts, seed + offset)
                example.rules = _permuted_mapping(example.rules, seed + offset + 1)
            _assert_preserved(source, example)
            prompt = (
                _paraphrased_prompt(example)
                if name == "surface_template_paraphrase"
                else task.render(example)
            )
            example.example_id = f"{source.example_id}-anchor-probe-{arm}-{name}-{sha256_value(prompt)[:12]}"
            example.metadata = {
                **example.metadata,
                "anchor_probe_view": name,
                "anchor_probe_arm": arm,
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


def _manifest_unchecked(splits: dict[str, list[TaskExample]], *, arm: str) -> dict[str, Any]:
    _require(arm in ARMS, "unknown diagnostic arm")
    result = {}
    for role, views in (
        ("student_fit", fit_views(splits["student_fit"], arm=arm)),
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


def dataset_manifest(splits: dict[str, list[TaskExample]], *, arm: str = "control") -> dict[str, Any]:
    validate_examples(splits)
    result = _manifest_unchecked(splits, arm=arm)
    _require(
        all(result[k]["views_sha256"] == VIEWS_SHA256[arm][k] for k in result), "view population changed"
    )
    _require(sha256_value(result) == ARM_MANIFEST_SHA256[arm], "arm dataset manifest changed")
    return result


def isolation_inventory(splits: dict[str, list[TaskExample]]) -> dict[str, set[Any]]:
    validate_examples(splits)
    rows = [x for values in splits.values() for x in values]
    for arm in ARMS:
        rows += [v.example for v in fit_views(splits["student_fit"], arm=arm)]
    rows += [v.example for v in development_views(splits["student_dev"])]
    return dict(
        semantic={canonical_semantic_key(x) for x in rows},
        example_id={x.example_id for x in rows},
        pair_group_id={x.pair_group_id for x in rows},
        pair_seed={x.metadata["pair_seed"] for x in rows},
    )


def reject_overlap(inventory: dict[str, set[Any]], row: TaskExample) -> None:
    keys = dict(
        semantic=canonical_semantic_key(row),
        example_id=row.example_id,
        pair_group_id=row.pair_group_id,
        pair_seed=row.metadata["pair_seed"],
    )
    for key, value in keys.items():
        _require(value not in inventory[key], f"anchor probe population overlaps {key}")


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
    order_probe = previous.make_examples(DIFFICULTY)
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
                previous.fit_views(order_probe["student_fit"], arm=arm),
                2048,
            )
        )
    view_populations.append(
        (
            "order_probe_student_dev_views",
            previous.development_views(order_probe["student_dev"]),
            768,
        )
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
        arm_manifest_sha256={arm: sha256_value(dataset_manifest(splits, arm=arm)) for arm in ARMS},
        scientific_acceptance=False,
        formal_generated_responses_or_scores_read=False,
    )


def encode_view(
    view: PreparedView, tokenizer: Any, model_config: dict[str, Any], *, training: bool = True
) -> dict[str, Any]:
    try:
        return preparation.encode_view(view, tokenizer, model_config, training=training)
    except preparation.StudentOrderPreparationError as error:
        raise StudentAnchorProbeError(str(error)) from error


def validate_encoded_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        result = preparation.validate_encoded_fit(rows)
    except preparation.StudentOrderPreparationError as error:
        raise StudentAnchorProbeError(str(error)) from error
    return dict(
        **result,
        executed_optimizer_windows=12,
        executed_training_views=TRAINING_COUNT,
        executed_input_tokens=sum(result["window_input_tokens"][:12]),
    )


@dataclass(frozen=True)
class ResolvedStudentAnchorProbeProtocol:
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


def resolve_student_anchor_probe_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentAnchorProbeProtocol:
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
    payload = load_student_anchor_probe_protocol(raw, require_accepted=require_accepted)
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
        proposed = load_student_anchor_probe_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
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
        reviewed = load_student_anchor_probe_protocol(reviewed_raw, require_accepted=True)
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
    return ResolvedStudentAnchorProbeProtocol(
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
