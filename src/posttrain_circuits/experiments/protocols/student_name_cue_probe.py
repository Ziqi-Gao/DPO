"""Fixed-endpoint inference diagnostic of paired stems and branch-side suffixes.

No optimizer, checkpoint choice, promotion or formal qualification is in scope.
All matched naming variants are constructed once and retained, without filtering.
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
from posttrain_circuits.datasets.proofgraph.anti_shortcut import _assert_preserved
from posttrain_circuits.datasets.proofgraph.contracts import Literal, ProofStep, Rule, TaskExample
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.datasets.proofgraph.splits import assert_split_isolation, canonical_semantic_key
from posttrain_circuits.experiments.protocols import student_focus_lr_probe as previous

PROTOCOL_PATH = "prereg/amendments/qwen3_student_name_cue_probe_v1.json"
PROTOCOL_ID = "qwen3-student-name-cue-probe-v1"
PARENT_HEAD = "51e7cafca43c8b88ddf1b4eef5198e3abc6ec550"
PARENT_PROTOCOL_SHA256 = "28eae66afe928a9718c77362ab39abb0b1500adceda13314191bc04e9f7f246f"
BASE_REVISION = previous.BASE_REVISION
CHAT_TEMPLATE_SHA256 = previous.CHAT_TEMPLATE_SHA256
TOKENIZER_FINGERPRINT = previous.TOKENIZER_FINGERPRINT
RESPONSE_INSTRUCTIONS_SHA256 = previous.RESPONSE_INSTRUCTIONS_SHA256
FROZEN_SCIENCE_PATHS = previous.SCIENCE_PATHS
FROZEN_PROTOCOL_SHA256 = {**previous.FROZEN_PROTOCOL_SHA256, previous.PROTOCOL_PATH: PARENT_PROTOCOL_SHA256}
FROZEN_HELPER_SHA256 = dict(previous.FROZEN_HELPER_SHA256)
SCIENCE_PATHS = (
    *FROZEN_SCIENCE_PATHS,
    "src/posttrain_circuits/experiments/protocols/student_name_cue_probe.py",
    "tools/sdsc_student_name_cue_probe.py",
    "tools/sdsc_student_name_cue_probe_job.py",
    "tools/sdsc_student_name_cue_probe_worker.py",
    "tools/sdsc_student_name_cue_probe_audit.py",
    "tools/sdsc_student_name_cue_probe_startup.py",
    "tools/sdsc_provenance_v2.py",
    "tools/sdsc_provenance_upload_v2.py",
)
ARMS = ("control", "treatment")
CONDITIONS = ("preserve", "break_pairs", "break_paths", "break_both")
BLOCKS = (0, 1)
VIEW_NAMES = tuple(f"block{block}_{condition}" for block in BLOCKS for condition in CONDITIONS)
STRUCTURES = ("branch", "chain")
BASE_COUNT, VIEW_COUNT, TOTAL_RESPONSE_COUNT = 64, 512, 1024
BASE_SEED_START, TRANSFORM_SEED = 470_000_042, 480_000_042
CHECKPOINT_STEP = 32
DEVELOPMENT_MAX_MODEL_INPUT_TOKENS, MAX_NEW_TOKENS = 2454, 256
DIFFICULTY = copy.deepcopy(previous.DIFFICULTY)
EXAMPLES_SHA256 = "b7014f95c0e830d493ec9df52024b61d5686de8f66c09bea2c3f7b8cdfd5947b"
VIEWS_SHA256 = "0bbfd8c27744c48da6b9528ea3991581402c7c9825f48167659fbb9c17d88dea"
DATASET_MANIFEST_SHA256 = "67e33370f859f48d324aafaf164e9e18210d19b39dc716849aa2e95db08350e1"
SOURCE_CHECKPOINTS = {
    "control": {
        "job_id": "54673886",
        "plan_sha256": "61ca76a29a0a96113852dc9bb370f280c9ad326135c6e5797776cd4997fe92d2",
        "dataset_sha256": "84227f0858cad084ed7e4ef72317d02c1e0ab18875628b4065c938bc4a510669",
        "protocol_sha256": "eea3e4561042c97492a86b20887c20fc1788a7619337e2f20b6c63bc609e88aa",
        "protocol_artifact_sha256": "28eae66afe928a9718c77362ab39abb0b1500adceda13314191bc04e9f7f246f",
        "intent_id": "11c0db9b296008aeda5df5d008c2572f",
        "learning_rate": 5e-05,
        "model_state_sha256": "4abdfa9926a3d099e2358409e3d68fef83557e98bf2a150851e7d06b2d1599a1",
        "dense_format": "student_focus_lr_probe_dense_v1",
        "parent_checkpoint_sha256": "85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4",
        "report_sha256": "94c609bfbb3e3afce1d7524ec93fbbf768789ecaafb1c7e9297d1923bf3f0acb",
        "publication_sha256": "431d2cff78c620caf652982bf9363071f4d81b1d5ccf5d97b8349a60f1d9ade9",
        "path": "checkpoints/step-00000032.pt",
        "size": 8127108953,
        "sha256": "bd3321d28e0f3f9780d9156a9b6e5df7fa90415f7827c24c348313944baaf7f6",
        "scope": "student_focus_lr_probe_model_only",
        "step": 32,
    },
    "treatment": {
        "job_id": "54673887",
        "plan_sha256": "11e2849a6dbf50a273916c3c8081e467f5e6285130b2dc86b2577becb0b1ccd3",
        "dataset_sha256": "84227f0858cad084ed7e4ef72317d02c1e0ab18875628b4065c938bc4a510669",
        "protocol_sha256": "eea3e4561042c97492a86b20887c20fc1788a7619337e2f20b6c63bc609e88aa",
        "protocol_artifact_sha256": "28eae66afe928a9718c77362ab39abb0b1500adceda13314191bc04e9f7f246f",
        "intent_id": "63b098982006ba6b1faaddfc790c9b8c",
        "learning_rate": 2.5e-05,
        "model_state_sha256": "06a15c58061a22419857e1b4426aabe6900b96504877fbd21a4245ce9402f0f3",
        "dense_format": "student_focus_lr_probe_dense_v1",
        "parent_checkpoint_sha256": "85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4",
        "report_sha256": "edf876a600cc6d7867ddd268538ded6c7984cf74371c267d8d8656e9eeebc808",
        "publication_sha256": "88345bb575423c0eb1276aeef25317bd086d2da0ec2c8ed828f6ab909cd8de8a",
        "path": "checkpoints/step-00000032.pt",
        "size": 8127108953,
        "sha256": "7d7f14ebe39ba2ccae1ee737aeefa9cbfd9143bcd6f2e8cbd69d36143a82d9f5",
        "scope": "student_focus_lr_probe_model_only",
        "step": 32,
    },
}
PARENTS = SOURCE_CHECKPOINTS
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_MAX_BYTES = 4 * 1024**2


class StudentNameCueProbeError(ValueError):
    """A fixed diagnostic scientific or review boundary was violated."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise StudentNameCueProbeError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def proposed_student_name_cue_probe_protocol() -> dict[str, Any]:
    parent = previous.proposed_student_focus_lr_probe_protocol()
    return dict(
        schema_version=1,
        kind="opd_student_name_cue_probe_protocol",
        protocol_id=PROTOCOL_ID,
        scope=dict(
            purpose="inference_only_paired_stem_and_path_suffix_diagnostic",
            execution="sdsc_h100_probe_only",
            model_order=list(ARMS),
            checkpoint_step=32,
            endpoint_chosen_for_equal_complete_training_exposure_not_scores=True,
            prior_preparation_and_diagnostic_exposure_disclosed=True,
            prior_formal_anti_shortcut_exposure_disclosed=True,
            cumulative_adaptive_development_is_exploratory=True,
            original_family_read_only_for_exclusion_keys=True,
            formal_generated_responses_or_scores_read=False,
            training=False,
            checkpoint_selection=False,
            diagnostic_is_model_or_preparation_acceptance=False,
            downstream_calibration_g0_pilot_in_scope=False,
        ),
        preserved_base=dict(
            parent_science_head=PARENT_HEAD,
            parent_protocol=previous.PROTOCOL_PATH,
            parent_protocol_sha256=PARENT_PROTOCOL_SHA256,
            frozen_parent_protocol_sha256=dict(FROZEN_PROTOCOL_SHA256),
            unchanged_original_thresholds=copy.deepcopy(parent["preserved_base"]),
        ),
        model=dict(
            base_model_id="Qwen/Qwen3-1.7B",
            base_revision=BASE_REVISION,
            tokenizer_id="Qwen/Qwen3-1.7B",
            tokenizer_revision=BASE_REVISION,
            tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
            prompt_protocol="qwen3_non_thinking_v1",
            enable_thinking=False,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
            response_instructions_sha256=RESPONSE_INSTRUCTIONS_SHA256,
            sources=copy.deepcopy(SOURCE_CHECKPOINTS),
            native_inference_dtype="bfloat16",
            exact_FP32_checkpoint_master_hash_before_BF16_copy=True,
            optimizer_scheduler_or_training_state_loaded=False,
        ),
        data=dict(
            base_examples=BASE_COUNT,
            views_per_model=VIEW_COUNT,
            total_responses=TOTAL_RESPONSE_COUNT,
            base_raw_pair_seed_start=BASE_SEED_START,
            transform_seed=TRANSFORM_SEED,
            pair_schedule="470000042+pair_index;32_pairs;first24_branch_last8_chain;depth4;odd_pairs_reverse",
            difficulty=copy.deepcopy(DIFFICULTY),
            branch_examples=48,
            chain_examples=16,
            blocks=list(BLOCKS),
            conditions=list(CONDITIONS),
            views=list(VIEW_NAMES),
            row_order="base_then_block_then_condition",
            name_pool="opaque_SYM_index_sha256prefix6;branch_stems_with_L_R_suffix;all_other_atoms_opaque",
            stem_order="one_seeded_shuffle_per_pair_and_polarity;block1_rotates_block0_order_plus1;two_fixed_assignments_not_full_Latin_balance",
            branch_pair_intervention="right_stems_cyclic_shift_plus1_mod3",
            branch_path_intervention="swap_assigned_left_right_names_at_middle_level_only",
            branch_combination="pair_intervention_then_middle_level_swap",
            chain_permutations=[[0, 1, 2], [1, 2, 0], [2, 1, 0], [1, 0, 2]],
            chain_is_generic_name_permutation_negative_control_not_branch_factor_estimate=True,
            same_exact_lexical_multiset_and_prompt_target_literal_frequencies=True,
            exact_complete_model_prefix_and_canonical_EOS_token_lengths_equal_all8_views=True,
            fact_rule_citation_order_and_canonical_proof_order_unchanged=True,
            original_verifier_and_exact_bijective_isomorphism_required=True,
            filter_truncate_repair_rows_or_seed_search=False,
            failure_to_construct_every_variant_stops_before_GPU=True,
            examples_sha256=EXAMPLES_SHA256,
            views_sha256=VIEWS_SHA256,
            dataset_manifest_sha256=DATASET_MANIFEST_SHA256,
            isolation_dimensions=["semantic", "example_id", "pair_group_id", "pair_seed"],
            isolation_population_counts={
                **parent["data"]["isolation_population_counts"],
                "focus_lr_student_fit": 256,
                "focus_lr_student_dev": 128,
            },
            isolation_excluded_view_counts={
                **parent["data"]["isolation_excluded_view_counts"],
                "focus_lr_control_student_fit_views": 2048,
                "focus_lr_treatment_student_fit_views": 2048,
                "focus_lr_student_dev_views": 768,
            },
        ),
        development=dict(
            generation="greedy_native_BF16_without_extra_autocast",
            max_model_input_tokens=DEVELOPMENT_MAX_MODEL_INPUT_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            seed="42_plus_manifest_ordinal_reset_identically_for_each_model",
            rank_partition="rank=(pair_index+block_index+condition_index)%2;both_signed_members_same_rank_per_cell",
            metric="original_VerificationResult.answer_correct_and_complete_proof_reward",
            all512_responses_each_model_required=True,
            diagnostic_completion_is_independent_of_measured_rates=True,
            canonical_target_never_used_as_generation_prefix=True,
        ),
        execution=dict(
            gpu_count=2,
            cpu_count=24,
            host_memory_gib=384,
            walltime_minutes=90,
            worker_deadline_seconds=4800,
            publication_reserve_seconds=600,
            maximum_concurrently_allocatable_gpus=4,
            early_startup_whole_child_seconds=300,
            early_thread_environment={
                "MKL_NUM_THREADS": "12",
                "OMP_NUM_THREADS": "12",
                "OPENBLAS_NUM_THREADS": "12",
            },
            minimum_headroom_fraction=0.2,
            minimum_headroom_gib=32,
            minimum_node_local_free_gib=192,
            maximum_persistent_large_gib=0,
            maximum_small_fetch_mib=128,
            maximum_prompt_file_mib=32,
            maximum_other_small_file_mib=16,
            fixed2454_token_native_bf16_finite_forward_each_rank=True,
            independent_raw_audit_required=True,
            no_automatic_retry=True,
            new_consumer_provenance="quest-sdsc-git-provenance-v2;full_bundle_cap256;named_v2_verifier",
            historical_parent_verification="unchanged_LR_v1_controller_and_provenance",
            serial_models_one_W2_job=True,
            blackwell_execution_class_reused=False,
        ),
        analysis=dict(
            checkpoint_selection=False,
            adaptive_extension=False,
            automatic_retry=False,
            scientific_acceptance=False,
            selected_checkpoint=None,
            formal_thresholds_applied_to_probe_population=False,
            no_cross_model_or_structure_aggregation_for_primary_claim=True,
            primary="per_model_branch_2x2_condition_counts_and_base_paired_wins_losses_separately_in_both_blocks",
            outcomes=[
                "proof_correct",
                "answer_correct",
                "format_valid",
                "length_cap",
                "first_invalid_step",
                "error_code",
            ],
            branch_contrasts=[
                ["preserve", "break_pairs"],
                ["preserve", "break_paths"],
                ["break_paths", "break_both"],
                ["break_pairs", "break_both"],
            ],
            chain_report="generic_permutation_control_only",
            repeated_signed_pairs_and_maps_are_not_independent_samples=True,
            significance_or_acceptance_threshold=None,
            floor_or_ceiling_panels_are_inconclusive_not_hypothesis_falsification=True,
            aligned_baseline_absolute_rates_and_novel_tagged_format_limitation_required=True,
            limits="matched_tagged_name_format;does_not_establish_original_rename_failure_cause_or_population_reliability",
        ),
        science_files=list(SCIENCE_PATHS),
        frozen_helpers=dict(FROZEN_HELPER_SHA256),
        review_contract={
            **copy.deepcopy(parent["review_contract"]),
            "acceptance_allowed_changed_paths": [PROTOCOL_PATH, "docs/refactor/current_handoff.md"],
        },
        review=copy.deepcopy(REVIEW_PROPOSED),
    )


@dataclass(frozen=True)
class PreparedView:
    source_example_id: str
    view: str
    example: TaskExample
    prompt: str
    block: int
    condition: str


def _build_bases() -> dict[str, list[TaskExample]]:
    task, rows = ProofGraphTask(), []
    for pair_index in range(BASE_COUNT // 2):
        config = {**DIFFICULTY, "depth": 4, "structure": "branch" if pair_index < 24 else "chain"}
        pair = list(task.generate_pair(BASE_SEED_START + pair_index, config))
        if pair_index % 2:
            pair.reverse()
        rows.extend(pair)
    return {"student_dev": rows}


def make_examples(difficulty: dict[str, Any] | None = None) -> dict[str, list[TaskExample]]:
    _require(difficulty is None or _canonical(difficulty) == _canonical(DIFFICULTY), "difficulty changed")
    result = _build_bases()
    validate_examples(result)
    return result


def validate_examples(splits: dict[str, list[TaskExample]]) -> None:
    _require(set(splits) == {"student_dev"}, "only diagnostic development bases permitted")
    rows = splits["student_dev"]
    _require(
        len(rows) == BASE_COUNT and sha256_value([asdict(x) for x in rows]) == EXAMPLES_SHA256,
        "frozen base population differs",
    )
    assert_split_isolation(splits)
    task = ProofGraphTask()
    for index in range(BASE_COUNT // 2):
        pair = rows[2 * index : 2 * index + 2]
        _require(
            {x.label for x in pair} == {0, 1} and len({x.pair_group_id for x in pair}) == 1,
            "signed pair incomplete",
        )
        for row in pair:
            _require(
                row.metadata["pair_seed"] == BASE_SEED_START + index
                and row.metadata["depth"] == 4
                and row.metadata["structure"] == ("branch" if index < 24 else "chain"),
                "fixed base schedule differs",
            )
            _require(
                task.verify(row, task.parse_response(render_target(row))).reward == 1.0,
                "invalid canonical base target",
            )


def _atoms(example: TaskExample) -> set[str]:
    return {
        item.atom
        for item in [
            *example.facts.values(),
            example.query,
            *[x for rule in example.rules.values() for x in rule.antecedents],
            *[rule.consequent for rule in example.rules.values()],
        ]
    }


def _literal_frequencies(example: TaskExample) -> tuple[Counter, Counter]:
    prompt = [
        *example.facts.values(),
        example.query,
        *[x for rule in example.rules.values() for x in rule.antecedents],
        *[rule.consequent for rule in example.rules.values()],
    ]
    return Counter(str(x) for x in prompt), Counter(str(x.conclusion) for x in example.canonical_proof)


def name_mapping(source: TaskExample, block: int, condition: str) -> dict[str, str]:
    _require(type(block) is int and block in BLOCKS and condition in CONDITIONS, "unknown naming condition")
    seed = TRANSFORM_SEED + (source.metadata["pair_seed"] - BASE_SEED_START) * 10000
    atoms = sorted(_atoms(source))
    mapping = {atom: f"SYM_{index:03d}_{sha256_value([seed, atom])[:6]}" for index, atom in enumerate(atoms)}
    roles = source.metadata["role_to_symbol"]
    for polarity_index, polarity in enumerate(("positive", "negative")):
        original = [roles[f"{polarity}_intermediate_{level:02d}"] for level in (1, 2, 3)]
        stems = [
            f"SYM_{100 + polarity_index * 3 + level:03d}_{sha256_value([seed, polarity, level])[:6]}"
            for level in range(3)
        ]
        random.Random(seed + polarity_index).shuffle(stems)
        if block == 1:
            stems = stems[1:] + stems[:1]
        if source.metadata["structure"] == "branch":
            for level, atom in enumerate(original):
                right_level = (level + int(condition in ("break_pairs", "break_both"))) % 3
                names = [stems[level] + "_L", stems[right_level] + "_R"]
                if level == 1 and condition in ("break_paths", "break_both"):
                    names.reverse()
                mapping[atom + "_L"], mapping[atom + "_R"] = names
        else:
            permutations = ((0, 1, 2), (1, 2, 0), (2, 1, 0), (1, 0, 2))
            order = permutations[CONDITIONS.index(condition)]
            mapping.update({atom: stems[order[index]] for index, atom in enumerate(original)})
    _require(
        set(mapping) == set(atoms) and len(set(mapping.values())) == len(atoms), "name map not bijective"
    )
    return mapping


def _mapped(source: TaskExample, mapping: dict[str, str]) -> TaskExample:
    _require(
        set(mapping) == _atoms(source) and len(set(mapping.values())) == len(mapping),
        "name map not bijective",
    )

    def literal(value):
        return Literal(mapping[value.atom], value.negated)

    result = copy.deepcopy(source)
    result.facts = {key: literal(value) for key, value in source.facts.items()}
    result.rules = {
        key: Rule(key, tuple(literal(x) for x in value.antecedents), literal(value.consequent))
        for key, value in source.rules.items()
    }
    result.query = literal(source.query)
    result.canonical_proof = [
        ProofStep(x.step_id, x.rule_id, x.citations, literal(x.conclusion)) for x in source.canonical_proof
    ]
    _assert_preserved(source, result)
    # Exact inverse reconstruction guards more than label/reward preservation.
    reverse = {value: key for key, value in mapping.items()}

    def undo(value):
        return Literal(reverse[value.atom], value.negated)

    _require(
        [(key, undo(value)) for key, value in result.facts.items()] == list(source.facts.items())
        and [
            (key, tuple(undo(x) for x in value.antecedents), undo(value.consequent))
            for key, value in result.rules.items()
        ]
        == [(key, value.antecedents, value.consequent) for key, value in source.rules.items()]
        and undo(result.query) == source.query
        and [ProofStep(x.step_id, x.rule_id, x.citations, undo(x.conclusion)) for x in result.canonical_proof]
        == source.canonical_proof,
        "exact graph/proof isomorphism failed",
    )
    return result


def development_views(examples: list[TaskExample]) -> list[PreparedView]:
    result, task = [], ProofGraphTask()
    for source in examples:
        frequencies = None
        for block in BLOCKS:
            for condition in CONDITIONS:
                mapping = name_mapping(source, block, condition)
                example = _mapped(source, mapping)
                actual = _literal_frequencies(example)
                if frequencies is None:
                    frequencies = actual
                _require(actual == frequencies, "matched name multiset or literal frequency changed")
                prompt = task.render(example)
                view = f"block{block}_{condition}"
                example.example_id = f"{source.example_id}-name-cue-{view}-{sha256_value(prompt)[:12]}"
                example.metadata = {
                    **example.metadata,
                    "source_example_id": source.example_id,
                    "name_cue_block": block,
                    "name_cue_condition": condition,
                    "name_cue_mapping": mapping,
                }
                result.append(PreparedView(source.example_id, view, example, prompt, block, condition))
    return result


def _manifest_unchecked(splits: dict[str, list[TaskExample]]) -> dict[str, Any]:
    bases = splits["student_dev"]
    views = development_views(bases)
    return dict(
        base_count=len(bases),
        count=len(views),
        total_responses=2 * len(views),
        examples_sha256=sha256_value([asdict(x) for x in bases]),
        views_sha256=sha256_value([asdict(x) for x in views]),
        ordered_base_ids=[x.example_id for x in bases],
        ordered_ids=[x.example.example_id for x in views],
        pair_seeds=sorted({x.metadata["pair_seed"] for x in bases}),
    )


def dataset_manifest(splits: dict[str, list[TaskExample]]) -> dict[str, Any]:
    validate_examples(splits)
    result = _manifest_unchecked(splits)
    _require(
        result["views_sha256"] == VIEWS_SHA256 and sha256_value(result) == DATASET_MANIFEST_SHA256,
        "frozen diagnostic views/manifest differ",
    )
    return result


def encode_view(view: PreparedView, tokenizer: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    try:
        return previous.encode_view(view, tokenizer, model_config, training=False)
    except previous.StudentFocusLRProbeError as error:
        raise StudentNameCueProbeError(str(error)) from error


def validate_encoded_views(rows: list[dict[str, Any]], views: list[PreparedView]) -> dict[str, Any]:
    _require(len(rows) == len(views) == VIEW_COUNT, "all512 encoded views required")
    _require(
        sha256_value([asdict(view) for view in views]) == VIEWS_SHA256, "encoded view population differs"
    )
    lengths = {}
    for row, view in zip(rows, views, strict=True):
        prefix, response = row["prefix_length"], row["response_length"]
        _require(
            row["example_id"] == view.example.example_id
            and type(prefix) is int
            and type(response) is int
            and 0 < prefix <= 2198
            and 0 < response <= 256,
            "encoded identity or token envelope differs",
        )
        _require(
            len(row["input_ids"]) == prefix + response
            and row["labels"] == [-100] * prefix + row["input_ids"][prefix:]
            and row["input_ids"][-1] == row["eos_token_id"],
            "prefix mask or EOS differs",
        )
        expected = lengths.setdefault(view.source_example_id, (prefix, response))
        _require((prefix, response) == expected, "matched full-prefix/canonical-target BPE lengths differ")
    _require(
        len(lengths) == BASE_COUNT and len({x["example_id"] for x in rows}) == VIEW_COUNT,
        "encoded base/view population differs",
    )
    return dict(
        passed=True,
        views=VIEW_COUNT,
        bases=BASE_COUNT,
        matched_prefix_and_canonical_EOS_lengths=True,
        maximum_prefix_tokens=max(x["prefix_length"] for x in rows),
        maximum_canonical_response_tokens=max(x["response_length"] for x in rows),
        maximum_prefix_plus_cap=max(x["prefix_length"] + 256 for x in rows),
        input_token_lengths_sha256=sha256_value([[x["prefix_length"], x["response_length"]] for x in rows]),
        scientific_acceptance=False,
    )


def validate_student_name_cue_probe_protocol(
    payload: Any, *, require_accepted: bool = False
) -> dict[str, Any]:
    expected = proposed_student_name_cue_probe_protocol()
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
            raise StudentNameCueProbeError("invalid review date") from error
    expected["review"] = copy.deepcopy(review)
    try:
        _require(
            _canonical(payload) == _canonical(expected), "name cue probe protocol differs from frozen v1"
        )
    except (TypeError, OverflowError, RecursionError) as error:
        raise StudentNameCueProbeError("invalid protocol value") from error
    return copy.deepcopy(payload)


def load_student_name_cue_probe_protocol(raw: bytes, *, require_accepted: bool = False) -> dict[str, Any]:
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
        raise StudentNameCueProbeError("invalid protocol JSON") from error
    return validate_student_name_cue_probe_protocol(payload, require_accepted=require_accepted)


def protocol_core_sha256(payload: Any) -> str:
    checked = validate_student_name_cue_probe_protocol(payload)
    checked.pop("review")
    return hashlib.sha256(_canonical(checked)).hexdigest()


@dataclass(frozen=True)
class ResolvedStudentNameCueProbeProtocol:
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


def resolve_student_name_cue_probe_protocol(
    repo_root: Path | str,
    *,
    require_accepted: bool = True,
    git_dir: Path | str | None = None,
    expected_head: str | None = None,
) -> ResolvedStudentNameCueProbeProtocol:
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
    payload = load_student_name_cue_probe_protocol(raw, require_accepted=require_accepted)
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
        proposed = load_student_name_cue_probe_protocol(git("show", f"{implementation}:{PROTOCOL_PATH}"))
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
        reviewed = load_student_name_cue_probe_protocol(reviewed_raw, require_accepted=True)
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
    return ResolvedStudentNameCueProbeProtocol(
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


def isolation_inventory(splits: dict[str, list[TaskExample]]) -> dict[str, set[Any]]:
    validate_examples(splits)
    rows = [*splits["student_dev"], *[v.example for v in development_views(splits["student_dev"])]]
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
        _require(value not in inventory[key], f"name cue probe population overlaps {key}")


def historical_populations(
    original_family: Iterable[TaskExample], teacher_splits: dict[str, list[TaskExample]]
):
    """Yield the full historical producer populations, never formal responses."""
    _require(set(teacher_splits) == {"teacher_fit", "teacher_dev"}, "teacher isolation roles differ")
    yield "original_family", original_family, False
    for role, rows in teacher_splits.items():
        yield role, rows, False
    preparation = previous.preparation
    modules = (
        ("previous", preparation.original_preparation),
        ("invariance", preparation.invariance_preparation),
        ("branch", preparation.previous),
        ("order", preparation),
        ("order_probe", previous.order_diagnostic),
        ("anchor_probe", previous.anchor_diagnostic),
        ("batch_probe", previous.batch_diagnostic),
        ("focus", previous.previous),
        ("focus_lr", previous),
    )
    for name, module in modules:
        values = module.make_examples(DIFFICULTY)
        for role, rows in values.items():
            yield f"{name}_{role}", rows, False
        if name == "previous":
            continue
        if name in {"order_probe", "anchor_probe", "batch_probe", "focus_lr"}:
            for arm in ARMS:
                yield (
                    f"{name}_{arm}_student_fit_views",
                    module.fit_views(values["student_fit"], arm=arm),
                    True,
                )
        else:
            yield f"{name}_student_fit_views", module.fit_views(values["student_fit"]), True
        yield f"{name}_student_dev_views", module.development_views(values["student_dev"]), True


def audit_isolation(
    splits: dict[str, list[TaskExample]],
    original_family: Iterable[TaskExample],
    teacher_splits: dict[str, list[TaskExample]],
) -> dict[str, Any]:
    inventory = isolation_inventory(splits)
    settings = proposed_student_name_cue_probe_protocol()["data"]
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


def rank_for_ordinal(ordinal: int) -> int:
    _require(type(ordinal) is int and 0 <= ordinal < VIEW_COUNT, "invalid global ordinal")
    return (ordinal // 16 + (ordinal // 4) % 2 + ordinal % 4) % 2


def generation_ordinals(rank: int) -> list[int]:
    _require(type(rank) is int and rank in (0, 1), "invalid logical rank")
    return [ordinal for ordinal in range(VIEW_COUNT) if rank_for_ordinal(ordinal) == rank]
