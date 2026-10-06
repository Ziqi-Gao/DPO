"""Prospective common-initial preparation with role-independent branch names.

One fixed training name-pool permutation precedes all eight fit views.
Selection and later qualification remain separate, with original gates intact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
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
from posttrain_circuits.experiments.protocols import student_focus_preparation as previous
from posttrain_circuits.experiments.protocols import student_name_cue_probe as name_diagnostic

PROTOCOL_PATH = "prereg/amendments/qwen3_student_name_invariant_preparation_v1.json"
PROTOCOL_ID = "qwen3-student-name-invariant-preparation-v1"
PARENT_HEAD = "f1674f27b0b488ef1690fb5b67de639d64d3107f"
PARENT_PROTOCOL_SHA256 = "fc6f77ba5b0944b4b9f06a99c4f35bb390443d6deec069c59a19ea87b4eca8ec"
BASE_REVISION = previous.BASE_REVISION
CHAT_TEMPLATE_SHA256 = previous.CHAT_TEMPLATE_SHA256
TOKENIZER_FINGERPRINT = previous.TOKENIZER_FINGERPRINT
RESPONSE_INSTRUCTIONS_SHA256 = previous.RESPONSE_INSTRUCTIONS_SHA256
ADDITIONAL_FROZEN_DEPENDENCIES = (
    "src/posttrain_circuits/__init__.py",
    "src/posttrain_circuits/artifacts/__init__.py",
    "src/posttrain_circuits/core/__init__.py",
    "src/posttrain_circuits/datasets/__init__.py",
    "src/posttrain_circuits/datasets/proofgraph/__init__.py",
    "src/posttrain_circuits/experiments/__init__.py",
    "src/posttrain_circuits/experiments/protocols/__init__.py",
    "src/posttrain_circuits/experiments/protocols/local_fork.py",
    "src/posttrain_circuits/learning/__init__.py",
    "src/posttrain_circuits/learning/teacher/__init__.py",
    "src/posttrain_circuits/methods/__init__.py",
    "src/posttrain_circuits/methods/controls.py",
    "src/posttrain_circuits/methods/opd.py",
    "src/posttrain_circuits/methods/rl.py",
    "src/posttrain_circuits/models/__init__.py",
    "src/posttrain_circuits/artifacts/execution_safety_certification.py",
    "src/posttrain_circuits/artifacts/execution_science_protocol.py",
    "src/posttrain_circuits/artifacts/protocol_amendments.py",
    "src/posttrain_circuits/datasets/teacher_demos/__init__.py",
    "src/posttrain_circuits/datasets/trajectories/__init__.py",
    "src/posttrain_circuits/learning/rl/__init__.py",
    "src/posttrain_circuits/learning/rl/contracts.py",
    "src/posttrain_circuits/learning/supervision/__init__.py",
    "src/posttrain_circuits/learning/teacher/base.py",
    "src/posttrain_circuits/learning/teacher/topk.py",
    "src/posttrain_circuits/learning/training/__init__.py",
)
FROZEN_SCIENCE_PATHS = (*name_diagnostic.SCIENCE_PATHS, *ADDITIONAL_FROZEN_DEPENDENCIES)
FROZEN_PROTOCOL_SHA256 = {
    **name_diagnostic.FROZEN_PROTOCOL_SHA256,
    name_diagnostic.PROTOCOL_PATH: PARENT_PROTOCOL_SHA256,
}
FROZEN_HELPER_SHA256 = dict(name_diagnostic.FROZEN_HELPER_SHA256)
SCIENCE_PATHS = (
    *FROZEN_SCIENCE_PATHS,
    "src/posttrain_circuits/experiments/protocols/student_name_invariant_preparation.py",
    "tools/sdsc_student_name_invariant.py",
    "tools/sdsc_student_name_invariant_job.py",
    "tools/sdsc_student_name_invariant_worker.py",
    "tools/sdsc_student_name_invariant_startup.py",
    "tools/sdsc_student_name_invariant_audit.py",
    "tools/sdsc_student_name_invariant_native.py",
    "tools/sdsc_torch_generated_origins.py",
    "tools/sdsc_runtime_snapshot.py",
)
NATIVE_RUNTIME_SNAPSHOT = {
    "archive": {
        "path": (
            "/expanse/lustre/projects/nwu181/zgao12/OPD/runtime-snapshots/"
            "snapshot-2a3797d216384301877dfaf6a398431a/runtime.bin"
        ),
        "sha256": "7151cc2839d518474b3ab968064a14773a43728c1671aea462b53234da1db83d",
        "size": 7825971697,
    },
    "files_count": 49474,
    "manifest": {
        "path": (
            "/expanse/lustre/projects/nwu181/zgao12/OPD/runtime-snapshots/"
            "snapshot-2a3797d216384301877dfaf6a398431a/manifest.json"
        ),
        "sha256": "2cc4fa21afe2dc180379456d8dd2965e6a694ffb50d52cf0a11802bac0e910cd",
        "size": 10436951,
    },
    "python_relative_path": "bin/python3.12",
    "python_sha256": "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
    "runtime_root_sha256": "67c84e211ff24570ecd1f6a663133a9fd9923b2ddc56e3f14b2ac7c8d8d80819",
    "source_prefix": "/expanse/lustre/projects/nwu181/zgao12/OPD/envs/qwen3-v2-g0-py31213-cu128-v1",
    "total_bytes": 7825971675,
}
ANCHOR_VIEW_NAMES = ("identity", "entity_symbol_renaming")
FIT_BASE_COUNT, DEV_BASE_COUNT = 256, 256
FIT_COUNT, DEV_COUNT, TRAINING_COUNT = 2048, 1536, 2048
FIT_SEED_START, DEV_SEED_START = 490_000_042, 500_000_042
FIT_TRANSFORM_SEED, DEV_TRANSFORM_SEED = 510_000_042, 520_000_042
NAME_PERMUTATION_SEED = 530_000_042
LEARNING_RATE = 2.5e-5
FIT_VIEW_NAMES = previous.FIT_VIEW_NAMES
DEV_VIEW_NAMES = previous.DEV_VIEW_NAMES
STRUCTURES = previous.STRUCTURES
CHECKPOINT_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
MAX_INPUT_TOKENS = previous.MAX_INPUT_TOKENS
MAX_PREFIX_TOKENS = previous.MAX_PREFIX_TOKENS
DEVELOPMENT_MAX_MODEL_INPUT_TOKENS = 2456
DIFFICULTY = copy.deepcopy(previous.DIFFICULTY)
FIT_DIFFICULTY = copy.deepcopy(previous.FIT_DIFFICULTY)
# Three branch pairs per window; the fourth alternates chain/DAG by window.
FIT_PAIR_STRUCTURES = ("branch", "branch", "branch", "alternating_chain_dag")
EXAMPLES_SHA256 = {
    "student_dev": "fe69c8c26836d20371b072e0049bc56a40868a71c884f2bf406ec685ac6ed11a",
    "student_fit": "ea8440b0460510de01a0d5446cb4c053d233a230a8bf7245ed10db99274dcfd5",
}
VIEWS_SHA256 = {
    "student_dev": "d1ecaf3e3167b73fb3157fe2148a68d701a29f4fcf77100449a8990b3097dd59",
    "student_fit": "8bc6702cbb2e1924d1f050e0eddbbd0ea1aafdc1182ff8873d5400b1cbd44bdd",
}
DATASET_MANIFEST_SHA256 = "8abd38e334e7237f3433fcf92c5ee359b671ace83dbc8590259303f3cb2b2837"
NAME_MAPPING_SHA256 = "72a688fff37675770d24dd051becccf4fc2912629749ac2393f99f8293e332f6"
NAME_MAPPING_AUDIT = {
    "branch_by_depth": {
        "2": {
            "identity_maps": 0,
            "moved_atoms": 105,
            "pooled_atoms": 128,
            "retained_same_stem_pairs": 22,
            "same_level_pairs": 64,
            "signed_pairs": 32,
        },
        "3": {
            "identity_maps": 0,
            "moved_atoms": 219,
            "multilevel_paths": 128,
            "pooled_atoms": 256,
            "retained_constant_suffix_paths": 52,
            "retained_same_stem_pairs": 20,
            "same_level_pairs": 128,
            "signed_pairs": 32,
        },
        "4": {
            "identity_maps": 0,
            "moved_atoms": 354,
            "multilevel_paths": 128,
            "pooled_atoms": 384,
            "retained_constant_suffix_paths": 26,
            "retained_same_stem_pairs": 15,
            "same_level_pairs": 192,
            "signed_pairs": 32,
        },
    },
    "filtering_or_resampling": False,
    "scientific_acceptance": False,
    "signed_pairs": 128,
}
NET_ORDER_CHANGED_ROWS = {"facts": 1532, "rules": 1536}
COUNT_KEYS = {"num_examples", "answer_correct", "proof_correct", "format_valid"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_MAX_BYTES = 4 * 1024**2


class StudentNameInvariantPreparationError(ValueError):
    """The diagnostic contract rejected inconsistent or unreviewed evidence."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentNameInvariantPreparationError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def proposed_student_name_invariant_preparation_protocol() -> dict[str, Any]:
    value = previous.proposed_student_focus_preparation_protocol()
    value.update(kind="opd_student_name_invariant_preparation_protocol", protocol_id=PROTOCOL_ID)
    value["scope"] = dict(
        purpose="fresh_name_invariance_preparation_of_common_1p7b_initial_for_all_methods",
        execution="sdsc_h100_preparation_only",
        interpretation="post_preparation_capability_not_native_instruction_capability",
        prior_preparation_and_diagnostic_exposure_disclosed=True,
        prior_formal_anti_shortcut_exposure_disclosed=True,
        cumulative_adaptive_development_is_exploratory=True,
        prior_lr_diagnostic_jobs={"control": "54673886", "treatment": "54673887"},
        fixed_learning_rate_from_prior_lower_lr_recipe=LEARNING_RATE,
        sole_training_policy_change_relative_to_lower_lr_recipe="one_branch_intermediate_original_tagged_name_pool_bijection_before_all_eight_fit_views",
        development_restores_original256_bases_and_twelve_checkpoint_selection=True,
        failed_name_diagnostic_job="54687025",
        failed_name_diagnostic_execution_accepted=False,
        failed_name_diagnostic_publication_sha256="22fdd458b74c895f709cdc596042c6e2c69bce652d8bde5e98662e3e6c3bc35c",
        failed_name_diagnostic_local_raw_replay_sha256="af79caf132b18446ec67220df3cc6a3c7e9fa702cfbc481ebcbe04e31cbbe19f",
        prior_evidence_is_descriptive_failed_run_not_preparation_acceptance=True,
        prospective_training_effect_not_established=True,
        perfectly_decorrelated_finite_sample_not_claimed=True,
        historical_renaming_semantics_not_retroactively_a_protocol_bug=True,
        historical_checkpoints_reselected=False,
        historical_model_outputs_used_as_targets=False,
        original_family_read_only_for_exclusion_keys=True,
        formal_generated_responses_or_scores_read=False,
        teacher_generated_targets=False,
        preparation_is_student_or_g0_acceptance=False,
        downstream_calibration_g0_pilot_in_scope=False,
    )
    value["preserved_base"].update(
        parent_science_head=PARENT_HEAD,
        parent_name_cue_protocol=name_diagnostic.PROTOCOL_PATH,
        parent_name_cue_protocol_sha256=PARENT_PROTOCOL_SHA256,
        frozen_parent_protocol_sha256=dict(FROZEN_PROTOCOL_SHA256),
    )
    data = value["data"]
    exclusions = name_diagnostic.proposed_student_name_cue_probe_protocol()["data"]
    data.update(
        fit_raw_pair_seed_start=FIT_SEED_START,
        development_raw_pair_seed_start=DEV_SEED_START,
        fit_transform_seed=FIT_TRANSFORM_SEED,
        development_transform_seed=DEV_TRANSFORM_SEED,
        fit_pair_seed_schedule="490000042+pair_index;exactly128_pairs;no_filter_or_replacement",
        fit_transform="role_name_bijection_once_before_all8;case_seed510000042+base_index*10000;rename+1;fact+3;rule+4;independent_joint_fact+100+2*declared_view_index_rule+101+2*declared_view_index_except_identity_and_entity_symbol_renaming",
        development_transform="unchanged_original_five_transform_suite_seed520000042_distractors32;no_training_role_permutation",
        measured_fit_net_order_changed_rows=dict(NET_ORDER_CHANGED_ROWS),
        dataset_manifest_sha256=DATASET_MANIFEST_SHA256,
        examples_sha256=dict(EXAMPLES_SHA256),
        views_sha256=dict(VIEWS_SHA256),
        training_name_assignment=dict(
            seed_start=NAME_PERMUTATION_SEED,
            seed_schedule="530000042+original_pair_seed-490000042;shared_both_signed_siblings",
            pool="all_original_branch_intermediate_stem_L_and_stem_R_atoms_across_both_polarities",
            distribution="one_random.Random(seed).shuffle_of_sorted_pool;uniform_bijection;no_filter_resample_or_outcome_selection",
            all_nonpool_atoms_fixed=True,
            original_atom_pool_preserved=True,
            apply_before_all_eight_views=True,
            graph_rule_fact_and_proof_order_and_citations_unchanged=True,
            original_order_anchor_fraction=0.25,
            unmodified_branch_name_anchor_guarantee=False,
            accidental_fixed_points_and_retained_cues_are_kept=True,
            depth2_has_no_multilevel_path_continuity=True,
            chain_and_converging_dag_role_maps_are_identity=True,
            mapping_sha256=NAME_MAPPING_SHA256,
            descriptive_mapping_audit=copy.deepcopy(NAME_MAPPING_AUDIT),
        ),
        original_base_provenance_included_in_isolation=True,
        semantic_key_is_not_alpha_isomorphism_key=True,
        isolation_population_counts={**exclusions["isolation_population_counts"], "name_cue_student_dev": 64},
        isolation_excluded_view_counts={
            **exclusions["isolation_excluded_view_counts"],
            "name_cue_student_dev_views": 512,
        },
    )
    value["training"]["optimizer"]["learning_rate"] = LEARNING_RATE
    value["development"].update(
        max_model_input_tokens=DEVELOPMENT_MAX_MODEL_INPUT_TOKENS,
        larger_context_scope="new_preparation_development2456_only_training1536_formal_qualification2244_unchanged",
        fixed_population_requires_two_extra_context_tokens=True,
    )
    value["execution"].pop("preflight_fixed2454_token_native_BF16_finite_forward_each_rank")
    value["execution"].pop("each_checkpoint_fixed2454_token_native_BF16_finite_forward_each_rank")
    value["execution"].update(
        development_context_probe_tokens=DEVELOPMENT_MAX_MODEL_INPUT_TOKENS,
        preflight_full_development_context_native_BF16_finite_forward_each_rank=True,
        each_checkpoint_full_development_context_native_BF16_finite_forward_each_rank=True,
        historical2454_preflight_is_not_new_shape_evidence=True,
        native_runtime_snapshot=copy.deepcopy(NATIVE_RUNTIME_SNAPSHOT),
        native_runtime_origin_before_after_each_rank=True,
        strict_torch_generated_origin=True,
        native_runtime_stage_in_worker_budget=True,
        maximum_native_file_mib=16,
        maximum_native_artifacts_mib=64,
    )
    value["preserved_base"]["additional_named_frozen_dependencies"] = list(ADDITIONAL_FROZEN_DEPENDENCIES)
    value["science_files"] = list(SCIENCE_PATHS)
    value["frozen_helpers"] = dict(FROZEN_HELPER_SHA256)
    value["review_contract"]["acceptance_allowed_changed_paths"] = [
        PROTOCOL_PATH,
        "docs/refactor/current_handoff.md",
    ]
    return value


def validate_student_name_invariant_preparation_protocol(
    payload: Any, *, require_accepted: bool = False
) -> dict[str, Any]:
    expected = proposed_student_name_invariant_preparation_protocol()
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
            raise StudentNameInvariantPreparationError("invalid review date") from error
    expected["review"] = copy.deepcopy(review)
    try:
        _require(
            _canonical(payload) == _canonical(expected),
            "name invariant preparation protocol differs from frozen v1",
        )
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentNameInvariantPreparationError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_name_invariant_preparation_protocol(
    raw: bytes, *, require_accepted: bool = False
) -> dict[str, Any]:
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
        raise StudentNameInvariantPreparationError("invalid protocol JSON") from error
    return validate_student_name_invariant_preparation_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_name_invariant_preparation_protocol(payload)
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


def training_name_mapping(source: TaskExample) -> dict[str, str]:
    """One fixed uniform tagged-name-pool permutation shared by signed siblings."""
    atoms = name_diagnostic._atoms(source)
    mapping = {atom: atom for atom in sorted(atoms)}
    seed = source.metadata.get("pair_seed")
    _require(
        type(seed) is int and FIT_SEED_START <= seed < FIT_SEED_START + FIT_BASE_COUNT // 2,
        "training name map requires fresh fit pair provenance",
    )
    structure = source.metadata.get("structure")
    _require(structure in STRUCTURES, "unknown fit structure")
    if structure != "branch":
        return mapping
    depth = source.metadata.get("depth")
    _require(type(depth) is int and 2 <= depth <= 4, "branch fit depth differs")
    roles = source.metadata["role_to_symbol"]
    pool = sorted(
        roles[f"{polarity}_intermediate_{level:02d}"] + "_" + side
        for polarity in ("positive", "negative")
        for level in range(1, depth)
        for side in ("L", "R")
    )
    _require(len(set(pool)) == 4 * (depth - 1) and set(pool) <= atoms, "branch name pool differs")
    targets = list(pool)
    random.Random(NAME_PERMUTATION_SEED + seed - FIT_SEED_START).shuffle(targets)
    mapping.update(zip(pool, targets, strict=True))
    _require(set(mapping) == set(mapping.values()) == atoms, "training map must permute original pool")
    return mapping


def name_permuted_base(source: TaskExample) -> TaskExample:
    """Preserve original IDs, provenance and exact inverse graph/proof semantics."""
    return name_diagnostic._mapped(source, training_name_mapping(source))


def name_mapping_manifest(examples: list[TaskExample]) -> list[dict[str, Any]]:
    return [
        dict(
            source_example_id=x.example_id,
            pair_group_id=x.pair_group_id,
            pair_seed=x.metadata["pair_seed"],
            original_semantic_key=canonical_semantic_key(x),
            mapping=training_name_mapping(x),
        )
        for x in examples
    ]


def name_mapping_audit(examples: list[TaskExample]) -> dict[str, Any]:
    """Descriptive retained cues, once per signed pair; never an efficacy gate."""
    seen, by_depth = {}, {}
    for source in examples:
        mapping = training_name_mapping(source)
        group = source.pair_group_id
        if group in seen:
            _require(seen[group] == mapping, "signed siblings have different name maps")
            continue
        seen[group] = mapping
        if source.metadata["structure"] != "branch":
            continue
        depth, roles = source.metadata["depth"], source.metadata["role_to_symbol"]
        counts = by_depth.setdefault(str(depth), Counter())
        counts["signed_pairs"] += 1
        counts["moved_atoms"] += sum(a != b for a, b in mapping.items())
        counts["pooled_atoms"] += 4 * (depth - 1)
        counts["identity_maps"] += int(all(a == b for a, b in mapping.items()))
        for polarity in ("positive", "negative"):
            for level in range(1, depth):
                stem = roles[f"{polarity}_intermediate_{level:02d}"]
                counts["same_level_pairs"] += 1
                counts["retained_same_stem_pairs"] += int(
                    mapping[stem + "_L"][:-2] == mapping[stem + "_R"][:-2]
                )
            if depth > 2:
                for side in ("L", "R"):
                    suffixes = [
                        mapping[roles[f"{polarity}_intermediate_{level:02d}"] + "_" + side][-1]
                        for level in range(1, depth)
                    ]
                    counts["multilevel_paths"] += 1
                    counts["retained_constant_suffix_paths"] += int(len(set(suffixes)) == 1)
    return dict(
        signed_pairs=len(seen),
        branch_by_depth={d: dict(c) for d, c in by_depth.items()},
        scientific_acceptance=False,
        filtering_or_resampling=False,
    )


def fit_views(examples: list[TaskExample]) -> list[PreparedView]:
    task, result = ProofGraphTask(), []
    for index, source in enumerate(examples):
        seed = FIT_TRANSFORM_SEED + index * 10000
        working = name_permuted_base(source)
        plain_fact, plain_rule = copy.deepcopy(working), copy.deepcopy(working)
        plain_fact.facts = _permuted_mapping(plain_fact.facts, seed + 3)
        plain_rule.rules = _permuted_mapping(plain_rule.rules, seed + 4)
        renamed = _renamed(working, seed + 1)
        fact, rule = copy.deepcopy(renamed), copy.deepcopy(renamed)
        fact.facts = _permuted_mapping(fact.facts, seed + 3)
        rule.rules = _permuted_mapping(rule.rules, seed + 4)
        joint = copy.deepcopy(fact)
        joint.rules = _permuted_mapping(joint.rules, seed + 4)
        variants = list(
            zip(
                FIT_VIEW_NAMES,
                (
                    copy.deepcopy(working),
                    copy.deepcopy(working),
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
            example.example_id = f"{source.example_id}-name-invariant-v1-{name}-{sha256_value(prompt)[:12]}"
            example.metadata = {
                **example.metadata,
                "name_invariant_preparation_view": name,
                "training_name_mapping_sha256": sha256_value(training_name_mapping(source)),
                "original_source_semantic_key": canonical_semantic_key(source),
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
    result["student_fit"]["name_mapping_sha256"] = sha256_value(name_mapping_manifest(splits["student_fit"]))
    result["student_fit"]["name_mapping_audit"] = name_mapping_audit(splits["student_fit"])
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
    rows += [name_permuted_base(x) for x in splits["student_fit"]]
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
        _require(value not in inventory[key], f"name invariant preparation population overlaps {key}")


def historical_populations(original_family, teacher_splits):
    yield from name_diagnostic.historical_populations(original_family, teacher_splits)
    prior = name_diagnostic.make_examples()
    yield "name_cue_student_dev", prior["student_dev"], False
    yield "name_cue_student_dev_views", name_diagnostic.development_views(prior["student_dev"]), True


def audit_isolation(
    splits: dict[str, list[TaskExample]],
    original_family: Iterable[TaskExample],
    teacher_splits: dict[str, list[TaskExample]],
) -> dict[str, Any]:
    inventory = isolation_inventory(splits)
    settings = proposed_student_name_invariant_preparation_protocol()["data"]
    counts, digests, view_counts, view_digests = {}, {}, {}, {}
    for name, rows, is_view in historical_populations(original_family, teacher_splits):
        observed_counts, observed_digests = (view_counts, view_digests) if is_view else (counts, digests)
        expected = settings["isolation_excluded_view_counts" if is_view else "isolation_population_counts"]
        _require(
            name in expected and name not in observed_counts, "unexpected or duplicate excluded population"
        )
        ids, digest = set(), hashlib.sha256()
        for row in rows:
            example = row.example if is_view else row
            reject_overlap(inventory, example)
            _require(example.example_id not in ids, f"duplicate excluded {name} row")
            ids.add(example.example_id)
            digest.update(_canonical(asdict(row)) + b"\n")
        _require(len(ids) == expected[name], f"incomplete excluded population {name}")
        observed_counts[name], observed_digests[name] = len(ids), digest.hexdigest()
    _require(
        counts == settings["isolation_population_counts"]
        and view_counts == settings["isolation_excluded_view_counts"],
        "missing excluded population",
    )
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
            "development prefix exceeds2456; truncation forbidden",
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
class ResolvedStudentNameInvariantPreparationProtocol:
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


def resolve_student_name_invariant_preparation_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentNameInvariantPreparationProtocol:
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
            root / path, context=f"student name invariant preparation source {path}", max_bytes=_MAX_BYTES
        )

    head = text("rev-parse", "--verify", "HEAD^{commit}")
    _require(_COMMIT.fullmatch(head) and (expected_head is None or head == expected_head), "Git HEAD differs")
    _require(text("rev-parse", "--is-shallow-repository") == "false", "complete Git review history required")
    raw = read(PROTOCOL_PATH)
    payload = load_student_name_invariant_preparation_protocol(raw, require_accepted=require_accepted)
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
        proposed = load_student_name_invariant_preparation_protocol(
            git("show", f"{implementation}:{PROTOCOL_PATH}")
        )
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
        reviewed = load_student_name_invariant_preparation_protocol(reviewed_raw, require_accepted=True)
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
    return ResolvedStudentNameInvariantPreparationProtocol(
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
