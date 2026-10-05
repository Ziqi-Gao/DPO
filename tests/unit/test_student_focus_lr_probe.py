"""Real paired-data, exclusion, LR-only contract and genuine review boundaries."""

from __future__ import annotations

import copy
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.rendering import render_target
from posttrain_circuits.experiments.protocols import student_focus_lr_probe as prep

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def examples():
    return prep.make_examples(prep.DIFFICULTY)


@pytest.fixture(scope="module")
def arms(examples):
    return {arm: prep.fit_views(examples["student_fit"], arm=arm) for arm in prep.ARMS}


@pytest.mark.parametrize(
    "section,key,new",
    [
        ("data", "fit_raw_pair_seed_start", 42),
        ("data", "training_examples", 768),
        ("data", "filter_truncate_or_repair_rows", True),
        ("training", "optimizer_steps", 12),
        ("training", "input_token_budget", 3000000),
        ("training", "checkpoint_steps", [4, 8, 12]),
        ("development", "base_examples", 256),
        ("development", "eligibility_selection_or_candidate", True),
        ("development", "max_model_input_tokens", 3000),
        ("execution", "walltime_minutes", 480),
        ("execution", "gpu_count", True),
        ("scope", "formal_generated_responses_or_scores_read", True),
        ("scope", "historical_checkpoints_reselected", True),
        ("scope", "diagnostic_is_model_or_preparation_acceptance", True),
    ],
)
def test_typed_contract_rejects_scope_count_threshold_and_budget_changes(section, key, new):
    value = prep.proposed_student_focus_lr_probe_protocol()
    value[section][key] = new
    with pytest.raises(prep.StudentFocusLRProbeError, match="differs"):
        prep.validate_student_focus_lr_probe_protocol(value)


@pytest.mark.parametrize("raw", [b'{"review":{},"review":{}}', b"[]", b"", b"{", b"NaN"])
def test_bounded_strict_json_rejects_ambiguous_documents(raw):
    with pytest.raises(prep.StudentFocusLRProbeError):
        prep.load_student_focus_lr_probe_protocol(raw)


@pytest.mark.parametrize("mutation", ["drop", "reorder", "duplicate", "target", "seed"])
def test_actual_frozen_population_cannot_be_filtered_or_reselected(examples, mutation):
    values = copy.deepcopy(examples)
    if mutation == "drop":
        values["student_fit"].pop()
    elif mutation == "reorder":
        values["student_dev"].reverse()
    elif mutation == "duplicate":
        values["student_fit"][1] = values["student_fit"][0]
    elif mutation == "target":
        values["student_fit"][0].canonical_proof = []
    else:
        values["student_fit"][0].metadata["pair_seed"] = 42
    with pytest.raises(ValueError):
        prep.validate_examples(values)


def test_both_arms_keep_same_two_anchors_renaming_and_joint_order_policy(examples, arms):
    original = {e.example_id: (i, e) for i, e in enumerate(examples["student_fit"])}
    for view in arms["control"]:
        index, source = original[view.source_example_id]
        seed = prep.FIT_TRANSFORM_SEED + index * 10000
        if view.view == "identity":
            assert view.prompt == ProofGraphTask().render(source)
        elif view.view == "entity_symbol_renaming":
            assert view.prompt == ProofGraphTask().render(prep._renamed(source, seed + 1))
        else:
            expected_facts, expected_rules = source.facts, source.rules
            if view.view in {"fact_order_permutation", "rename_fact_order", "joint_rename_order"}:
                expected_facts = prep._permuted_mapping(expected_facts, seed + 3)
            if view.view in {"rule_order_permutation", "rename_rule_order", "joint_rename_order"}:
                expected_rules = prep._permuted_mapping(expected_rules, seed + 4)
            offset = 100 + 2 * prep.FIT_VIEW_NAMES.index(view.view)
            assert list(view.example.facts) == list(prep._permuted_mapping(expected_facts, seed + offset))
            assert list(view.example.rules) == list(prep._permuted_mapping(expected_rules, seed + offset + 1))
        if view.view in {
            "entity_symbol_renaming",
            "rename_fact_order",
            "rename_rule_order",
            "joint_rename_order",
        }:
            renamed = prep._renamed(source, seed + 1)
            assert dict(view.example.facts) == dict(renamed.facts)
            assert dict(view.example.rules) == dict(renamed.rules)
    with pytest.raises(prep.StudentFocusLRProbeError, match="unknown diagnostic arm"):
        prep.fit_views(examples["student_fit"], arm="later_best")


def test_complete_pinned_token_evidence_and_prefix_are_independent_of_rates():
    rows = [
        dict(
            example_id=str(i),
            input_ids=[101, 201, 999],
            labels=[-100, 201, 999],
            prefix_length=1,
            response_length=2,
            eos_token_id=999,
        )
        for i in range(2048)
    ]
    result = prep.validate_encoded_fit(rows)
    assert result["optimizer_windows"] == 32 and result["executed_optimizer_windows"] == 32
    assert result["executed_training_views"] == 2048 and result["executed_input_tokens"] == 6144
    with pytest.raises(ValueError):
        prep.validate_encoded_fit(rows[:768])
    rows[-1]["labels"][0] = 101
    with pytest.raises(ValueError):
        prep.validate_encoded_fit(rows)


@pytest.mark.parametrize("dimension", ["semantic", "example_id", "pair_group_id", "pair_seed"])
def test_historical_isolation_collision_is_rejected(examples, dimension):
    inventory = prep.isolation_inventory(examples)
    old = prep.previous.make_examples(prep.DIFFICULTY)["student_fit"][0]
    values = dict(
        semantic=prep.canonical_semantic_key(old),
        example_id=old.example_id,
        pair_group_id=old.pair_group_id,
        pair_seed=old.metadata["pair_seed"],
    )
    inventory[dimension].add(values[dimension])
    with pytest.raises(prep.StudentFocusLRProbeError, match="overlaps " + dimension):
        prep.reject_overlap(inventory, old)


@pytest.fixture
def reviewed_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Independent fixture")
    for relative in ("prereg/qwen3_v2.yaml", *prep.FROZEN_PROTOCOL_SHA256, *prep.FROZEN_HELPER_SHA256):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    git("add", ".")
    git("commit", "--quiet", "-m", "Preserved parent")
    monkeypatch.setattr(prep, "PARENT_HEAD", git("rev-parse", "HEAD"))
    monkeypatch.setattr(prep, "SCIENCE_PATHS", ("kernel.py",))
    monkeypatch.setattr(prep, "FROZEN_SCIENCE_PATHS", ())
    (tmp_path / "kernel.py").write_text("unchanged reviewed implementation\n")
    protocol = prep.proposed_student_focus_lr_probe_protocol()
    path = tmp_path / prep.PROTOCOL_PATH
    path.write_text(json.dumps(protocol))
    git("add", ".")
    git("commit", "--quiet", "-m", "Proposed implementation")
    implementation = git("rev-parse", "HEAD")
    protocol["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=implementation,
        reviewer="independent-test-reviewer",
        reviewed_at_utc="2026-10-04T03:00:00Z",
        rationale="Reviewed fixture implementation only.",
    )
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Independent review only")
    return tmp_path, git, implementation


class BoundaryTokenizer:
    chat_template = "fixture"
    eos_token_id = 999
    formatted = "prefix<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, view, prefix, response):
        self.view, self.prefix, self.response = view, prefix, response

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": self.view.prompt}]
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.formatted

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == self.formatted:
            return [101] * self.prefix
        assert text == render_target(self.view.example)
        return [201] * self.response


def test_training_mask_eos_and_development_envelopes_are_distinct(examples):
    import hashlib

    view = prep.fit_views(examples["student_fit"][:1])[0]
    token = BoundaryTokenizer(view, 1334, 200)
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256(token.chat_template.encode()).hexdigest(),
        }
    }
    row = prep.encode_view(view, token, config)
    assert len(row["input_ids"]) == 1535
    assert row["labels"][:1334] == [-100] * 1334 and row["labels"][-1] == 999
    token.response = 202
    with pytest.raises(prep.StudentFocusLRProbeError, match="training view"):
        prep.encode_view(view, token, config)
    token.prefix, token.response = 2198, 200
    assert prep.encode_view(view, token, config, training=False)["prefix_length"] == 2198
    token.prefix = 2199
    with pytest.raises(prep.StudentFocusLRProbeError, match="development prefix"):
        prep.encode_view(view, token, config, training=False)


def test_fixed_complete_windows_reject_budget_overrun_before_training():
    rows = [
        dict(
            example_id=str(i),
            input_ids=[101, 201, 999],
            labels=[-100, 201, 999],
            prefix_length=1,
            response_length=2,
            eos_token_id=999,
        )
        for i in range(2048)
    ]
    result = prep.validate_encoded_fit(rows)
    assert result["optimizer_windows"] == 32 and result["window_input_tokens"] == [192] * 32
    for row in rows:
        row.update(input_ids=[101] * 1024 + [201, 999], labels=[-100] * 1024 + [201, 999], prefix_length=1024)
    with pytest.raises(prep.StudentFocusLRProbeError, match="exceeds2M"):
        prep.validate_encoded_fit(rows)


def test_real_git_distinct_review_resolves_and_unrelated_followup_is_allowed(reviewed_repo):
    root, git, implementation = reviewed_repo
    result = prep.resolve_student_focus_lr_probe_protocol(root)
    assert result.implementation_commit == implementation
    assert result.acceptance_commit == git("rev-parse", "HEAD")
    (root / "unrelated.txt").write_text("does not change scientific source\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated follow-up")
    assert prep.resolve_student_focus_lr_probe_protocol(root).acceptance_commit == result.acceptance_commit


@pytest.mark.parametrize("mutation", ["dirty", "staged", "committed", "symlink"])
def test_real_git_source_mutations_are_rejected(reviewed_repo, mutation):
    root, git, _ = reviewed_repo
    source = root / "kernel.py"
    if mutation == "symlink":
        source.unlink()
        source.symlink_to(root / "prereg/qwen3_v2.yaml")
    else:
        source.write_text("unreviewed changed behavior\n")
        if mutation in {"staged", "committed"}:
            git("add", "kernel.py")
        if mutation == "committed":
            git("commit", "--quiet", "-m", "Unreviewed change")
    with pytest.raises((ValueError, OSError)):
        prep.resolve_student_focus_lr_probe_protocol(root)


def test_real_git_second_protocol_change_and_invented_implementation_reject(reviewed_repo):
    root, git, _ = reviewed_repo
    path = root / prep.PROTOCOL_PATH
    protocol = json.loads(path.read_text())
    protocol["review"]["rationale"] += " second acceptance"
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Unreviewed second review")
    with pytest.raises(prep.StudentFocusLRProbeError, match="exactly one"):
        prep.resolve_student_focus_lr_probe_protocol(root)


@pytest.mark.parametrize("protocol_path", sorted(prep.FROZEN_PROTOCOL_SHA256))
def test_all_nine_frozen_parent_protocols_cannot_change(reviewed_repo, protocol_path):
    root, git, _ = reviewed_repo
    path = root / protocol_path
    path.write_text(path.read_text() + "\n")
    with pytest.raises(prep.StudentFocusLRProbeError, match="frozen parent protocol"):
        prep.resolve_student_focus_lr_probe_protocol(root)


@pytest.mark.parametrize("path", sorted(prep.FROZEN_HELPER_SHA256))
def test_exact_frozen_execution_helper_mutation_rejects(reviewed_repo, path):
    root, _git, _implementation = reviewed_repo
    source = root / path
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(prep.StudentFocusLRProbeError, match="frozen diagnostic helper"):
        prep.resolve_student_focus_lr_probe_protocol(root)


def test_both_prior_probes_complete_arms_and_shared_dev_are_excluded_by_contract():
    data = prep.proposed_student_focus_lr_probe_protocol()["data"]
    for prefix in ("order_probe", "anchor_probe"):
        assert data["isolation_population_counts"][prefix + "_student_fit"] == 256
        assert data["isolation_population_counts"][prefix + "_student_dev"] == 128
        counts = data["isolation_excluded_view_counts"]
        assert counts[prefix + "_control_student_fit_views"] == 2048
        assert counts[prefix + "_treatment_student_fit_views"] == 2048
        assert counts[prefix + "_student_dev_views"] == 768
    changed = prep.proposed_student_focus_lr_probe_protocol()
    changed["data"]["isolation_excluded_view_counts"].pop("anchor_probe_treatment_student_fit_views")
    with pytest.raises(prep.StudentFocusLRProbeError, match="differs"):
        prep.validate_student_focus_lr_probe_protocol(changed)


@pytest.mark.parametrize("arm", prep.ARMS)
def test_every_rank_receives_balanced_anchor_slots_and_rename_exposure(examples, arm):
    views = prep.fit_views(examples["student_fit"], arm=arm)
    for start in range(0, len(views), 64):
        window = views[start : start + 64]
        global_anchors = [v for v in window if v.view in prep.ANCHOR_VIEW_NAMES]
        assert Counter(v.example.label for v in global_anchors) == {0: 8, 1: 8}
        for rank in (0, 1):
            rows = window[rank::2]
            anchors = [v for v in rows if v.view in prep.ANCHOR_VIEW_NAMES]
            assert len(anchors) == 8
            assert Counter(v.view for v in anchors) == dict.fromkeys(prep.ANCHOR_VIEW_NAMES, 4)
            # Both declared anchors have even slot indices. The unchanged negative
            # sibling rotation places their negative views on rank1; global-mean
            # training sees both polarities, while each entire rank stays balanced.
            assert Counter(v.example.label for v in anchors) == {1 - rank: 8}
            assert Counter(v.example.label for v in rows) == {0: 16, 1: 16}
            assert sum(v.view == "surface_template_paraphrase" for v in rows) == 4


def test_real_immutable_contract_has_one_intervention_and_no_candidate():
    value = prep.load_student_focus_lr_probe_protocol((ROOT / prep.PROTOCOL_PATH).read_bytes())
    expected = prep.proposed_student_focus_lr_probe_protocol()
    expected["review"] = value["review"]
    assert value == expected
    proposed = prep.proposed_student_focus_lr_probe_protocol()
    with pytest.raises(prep.StudentFocusLRProbeError, match="not accepted"):
        prep.validate_student_focus_lr_probe_protocol(proposed, require_accepted=True)
    assert proposed["scope"]["diagnostic_is_model_or_preparation_acceptance"] is False
    assert proposed["scope"]["formal_generated_responses_or_scores_read"] is False
    assert proposed["development"]["eligibility_selection_or_candidate"] is False
    assert not hasattr(prep, "checkpoint_eligible") and not hasattr(prep, "select_checkpoint")
    assert proposed["training"]["checkpoint_steps"] == [2, 4, 6, 8, 12, 16, 32]
    assert proposed["training"]["optimizer_steps"] == 32
    assert proposed["training"]["input_token_budget"] == 2_000_000
    assert proposed["development"]["expected_raw_responses_per_arm"] == 5376
    assert proposed["execution"]["walltime_minutes"] == 180
    assert proposed["execution"]["publication_reserve_seconds"] == 600
    assert (len(prep.FROZEN_SCIENCE_PATHS), len(prep.SCIENCE_PATHS), len(prep.FROZEN_PROTOCOL_SHA256)) == (
        154,
        160,
        9,
    )
    assert proposed["training"]["arm_learning_rates"] == {"control": 5e-5, "treatment": 2.5e-5}
    assert prep.learning_rate("control") == 5e-5
    assert prep.learning_rate("treatment") == 2.5e-5


@pytest.mark.parametrize("arm", ["", "higher", "best", None, 1, [], {}])
def test_unknown_lr_arm_rejects(arm):
    with pytest.raises(prep.StudentFocusLRProbeError):
        prep.learning_rate(arm)


@pytest.mark.parametrize(
    "section,key,new",
    [
        ("training", "arm_learning_rates", {"control": 5e-5, "treatment": 1e-5}),
        ("training", "arm_learning_rates", {"control": 2.5e-5, "treatment": 5e-5}),
        ("training", "actual_optimizer_and_scheduler_use_selected_arm_learning_rate", False),
        ("analysis", "nominal_learning_rate_sum_pairs", [[4, 16], [6, 12], [8, 8]]),
        ("analysis", "significance_or_acceptance_threshold", 0.05),
        ("scope", "cumulative_adaptive_development_is_exploratory", False),
        ("data", "fit_structure_sequence_fractions", {"branch": 0.5, "chain": 0.25, "converging_dag": 0.25}),
        ("data", "fit_renamed_sequence_fraction", 0.75),
        ("data", "fit_joint_permutation_sequence_fraction", 1.0),
    ],
)
def test_lr_data_and_analysis_changes_reject(section, key, new):
    value = prep.proposed_student_focus_lr_probe_protocol()
    value[section][key] = new
    with pytest.raises(prep.StudentFocusLRProbeError, match="differs"):
        prep.validate_student_focus_lr_probe_protocol(value)


def test_both_arms_every_ordered_row_target_prompt_and_rank_are_identical(examples, arms):
    assert prep.ARM_MANIFEST_SHA256["control"] == prep.ARM_MANIFEST_SHA256["treatment"]
    assert [asdict(v) for v in arms["control"]] == [asdict(v) for v in arms["treatment"]]
    task = ProofGraphTask()
    for arm, rows in arms.items():
        assert len(rows) == 2048
        assert Counter(v.view for v in rows) == dict.fromkeys(prep.FIT_VIEW_NAMES, 256)
        assert sha256_value(prep.dataset_manifest(examples, arm=arm)) == prep.ARM_MANIFEST_SHA256[arm]
        assert prep.fit_row_multiset_sha256(rows) == prep.FIT_ROW_MULTISET_SHA256
        for rank in (0, 1):
            assert [asdict(v) for v in rows[rank::2]] == [asdict(v) for v in arms["control"][rank::2]]
    for view in arms["control"]:
        assert task.verify(view.example, task.parse_response(render_target(view.example))).reward == 1
        assert not any("arm" in key for key in view.example.metadata)
    dev = prep.development_views(examples["student_dev"])
    assert len(dev) == 768
    assert Counter(v.view for v in dev) == dict.fromkeys(prep.DEV_VIEW_NAMES, 128)
    for view in dev:
        assert task.verify(view.example, task.parse_response(render_target(view.example))).reward == 1


def test_real_focus_schedule_does_not_silently_restore_batch_probe_policy(examples, arms):
    assert Counter(x.metadata["structure"] for x in examples["student_fit"]) == {
        "branch": 192,
        "chain": 32,
        "converging_dag": 32,
    }
    assert Counter(x.metadata["structure"] for x in examples["student_dev"]) == {
        "branch": 36,
        "chain": 54,
        "converging_dag": 38,
    }
    for rows in arms.values():
        for window_index in range(32):
            window = rows[window_index * 64 : (window_index + 1) * 64]
            nonbranch = "chain" if window_index % 2 == 0 else "converging_dag"
            assert Counter(v.example.metadata["structure"] for v in window) == {"branch": 48, nonbranch: 16}
            assert len({v.example.pair_group_id for v in window}) == 4
            assert Counter(v.example.label for v in window) == {0: 32, 1: 32}
            for rank in (0, 1):
                shard = window[rank::2]
                assert Counter(v.example.metadata["structure"] for v in shard) == {"branch": 24, nonbranch: 8}
                assert Counter(v.view for v in shard) == dict.fromkeys(prep.FIT_VIEW_NAMES, 4)
    assert all(4 <= x.metadata["distractors"] <= 8 for x in examples["student_fit"])
    assert all(4 <= x.metadata["distractors"] <= 16 for x in examples["student_dev"])


def test_entire_immediate_predecessor_bases_and_views_are_disjoint(examples):
    inventory = prep.isolation_inventory(examples)
    old = prep.previous.make_examples(prep.DIFFICULTY)
    for rows in old.values():
        for row in rows:
            prep.reject_overlap(inventory, row)
    for view in [
        *prep.previous.fit_views(old["student_fit"]),
        *prep.previous.development_views(old["student_dev"]),
    ]:
        prep.reject_overlap(inventory, view.example)
    data = prep.proposed_student_focus_lr_probe_protocol()["data"]
    assert data["isolation_population_counts"]["focus_student_fit"] == 256
    assert data["isolation_population_counts"]["focus_student_dev"] == 256
    assert data["isolation_excluded_view_counts"]["focus_student_fit_views"] == 2048
    assert data["isolation_excluded_view_counts"]["focus_student_dev_views"] == 1536
    with pytest.raises(prep.StudentFocusLRProbeError, match="incomplete excluded population"):
        prep.audit_isolation(examples, [], {"teacher_fit": [], "teacher_dev": []})


def test_analysis_is_vectorial_complete_and_not_posthoc_selected():
    analysis = prep.proposed_student_focus_lr_probe_protocol()["analysis"]
    assert analysis["primary_steps"] == [2, 4, 6, 8, 12, 16, 32]
    assert analysis["primary_view_names"] == list(prep.DEV_VIEW_NAMES[1:])
    assert analysis["nominal_learning_rate_sum_pairs"] == [[4, 8], [6, 12], [8, 16]]
    assert analysis["nominal_progress_pairs_have_different_training_data_exposure"] is True
    assert analysis["nominal_progress_pairs_are_not_equal_model_distance"] is True
    assert analysis["primary_vector_must_not_be_summed_or_averaged_across_views"] is True
    assert analysis["lower_IID_alone_is_not_improvement"] is True
    assert analysis["checkpoint_selection"] is False
    assert analysis["adaptive_extension"] is False
    assert analysis["significance_or_acceptance_threshold"] is None
