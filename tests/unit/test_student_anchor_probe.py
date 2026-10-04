"""Paired diagnostic data, bounded execution population, and independent review."""

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
from posttrain_circuits.experiments.protocols import student_anchor_probe as prep

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def examples():
    return prep.make_examples(prep.DIFFICULTY)


@pytest.fixture(scope="module")
def arms(examples):
    return {arm: prep.fit_views(examples["student_fit"], arm=arm) for arm in prep.ARMS}


def test_proposed_contract_is_diagnostic_only_with_no_hidden_selection():
    value = prep.load_student_anchor_probe_protocol((ROOT / prep.PROTOCOL_PATH).read_bytes())
    assert value == prep.proposed_student_anchor_probe_protocol()
    with pytest.raises(prep.StudentAnchorProbeError, match="not accepted"):
        prep.validate_student_anchor_probe_protocol(value, require_accepted=True)
    assert value["scope"]["diagnostic_is_model_or_preparation_acceptance"] is False
    assert value["scope"]["formal_generated_responses_or_scores_read"] is False
    assert value["scope"]["original_family_read_only_for_exclusion_keys"] is True
    assert value["preserved_base"]["all_original_numerical_gates_unchanged"] is True
    assert value["development"]["eligibility_selection_or_candidate"] is False
    assert not hasattr(prep, "checkpoint_eligible") and not hasattr(prep, "select_checkpoint")
    assert not any(
        k in value["development"]
        for k in ("selection", "ability_band", "minimum_answer_correct_count", "maximum_answer_correct_count")
    )
    assert value["data"]["training_examples"] == 768
    assert value["training"]["optimizer_steps"] == 12
    assert value["training"]["checkpoint_steps"] == [4, 6, 7, 8, 12]
    assert value["development"]["expected_raw_responses_per_arm"] == 3840
    assert len(prep.FROZEN_SCIENCE_PATHS) == 136 and len(prep.SCIENCE_PATHS) == 142


@pytest.mark.parametrize(
    "section,key,new",
    [
        ("data", "fit_raw_pair_seed_start", 42),
        ("data", "training_examples", 2048),
        ("data", "filter_truncate_or_repair_rows", True),
        ("training", "optimizer_steps", 32),
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
    value = prep.proposed_student_anchor_probe_protocol()
    value[section][key] = new
    with pytest.raises(prep.StudentAnchorProbeError, match="differs"):
        prep.validate_student_anchor_probe_protocol(value)


@pytest.mark.parametrize("raw", [b'{"review":{},"review":{}}', b"[]", b"", b"{", b"NaN"])
def test_bounded_strict_json_rejects_ambiguous_documents(raw):
    with pytest.raises(prep.StudentAnchorProbeError):
        prep.load_student_anchor_probe_protocol(raw)


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


def test_both_arms_exact_shared_base_targets_symbols_and_independent_order(examples, arms):
    task = ProofGraphTask()
    assert prep.ARM_MANIFEST_SHA256["control"] != prep.ARM_MANIFEST_SHA256["treatment"]
    for arm in prep.ARMS:
        manifest = prep.dataset_manifest(examples, arm=arm)
        assert sha256_value(manifest) == prep.ARM_MANIFEST_SHA256[arm]
        assert sha256_value([asdict(v) for v in arms[arm]]) == prep.VIEWS_SHA256[arm]["student_fit"]
        assert Counter(v.view for v in arms[arm]) == dict.fromkeys(prep.FIT_VIEW_NAMES, 256)
    seeds = set()
    for ordinal, (control, treatment) in enumerate(zip(arms["control"], arms["treatment"], strict=True)):
        assert control.source_example_id == treatment.source_example_id and control.view == treatment.view
        assert control.example.label == treatment.example.label
        assert control.example.pair_group_id == treatment.example.pair_group_id
        assert dict(control.example.facts) == dict(treatment.example.facts)
        assert dict(control.example.rules) == dict(treatment.example.rules)
        assert render_target(control.example) == render_target(treatment.example)
        source = examples["student_fit"][ordinal // 8]
        if control.view in prep.ANCHOR_VIEW_NAMES:
            assert list(treatment.example.facts) == list(source.facts)
            assert list(treatment.example.rules) == list(source.rules)
            assert list(control.example.facts) != list(treatment.example.facts)
            assert list(control.example.rules) != list(treatment.example.rules)
        else:
            assert list(control.example.facts) == list(treatment.example.facts)
            assert list(control.example.rules) == list(treatment.example.rules)
            assert control.prompt == treatment.prompt
        assert control.example.example_id != treatment.example.example_id
        for view in (control, treatment):
            assert task.verify(view.example, task.parse_response(render_target(view.example))).reward == 1
        for offset in (100, 101):
            seed = (
                prep.FIT_TRANSFORM_SEED
                + (ordinal // 8) * 10000
                + offset
                + 2 * prep.FIT_VIEW_NAMES.index(control.view)
            )
            assert seed not in seeds
            seeds.add(seed)
    assert len(seeds) == 4096
    assert prep.VIEWS_SHA256["control"]["student_dev"] == prep.VIEWS_SHA256["treatment"]["student_dev"]
    dev = prep.development_views(examples["student_dev"])
    assert len(dev) == 768
    assert Counter(v.view for v in dev) == dict.fromkeys(prep.DEV_VIEW_NAMES, 128)
    for view in dev:
        assert task.verify(view.example, task.parse_response(render_target(view.example))).reward == 1


def test_control_preserves_previous_joint_policy_and_only_two_anchor_slots_change(examples, arms):
    for index, source in enumerate(examples["student_fit"]):
        controls = {v.view: v for v in arms["control"][index * 8 : index * 8 + 8]}
        treatments = {v.view: v for v in arms["treatment"][index * 8 : index * 8 + 8]}
        seed = prep.FIT_TRANSFORM_SEED + index * 10000
        expected_rename = prep._renamed(source, seed + 1)
        for name, control in controls.items():
            expected_facts = source.facts
            expected_rules = source.rules
            if name in {"fact_order_permutation", "rename_fact_order", "joint_rename_order"}:
                expected_facts = prep._permuted_mapping(expected_facts, seed + 3)
            if name in {"rule_order_permutation", "rename_rule_order", "joint_rename_order"}:
                expected_rules = prep._permuted_mapping(expected_rules, seed + 4)
            offset = 100 + 2 * prep.FIT_VIEW_NAMES.index(name)
            assert list(control.example.facts) == list(prep._permuted_mapping(expected_facts, seed + offset))
            assert list(control.example.rules) == list(
                prep._permuted_mapping(expected_rules, seed + offset + 1)
            )
        for group in (controls, treatments):
            for name in (
                "entity_symbol_renaming",
                "rename_fact_order",
                "rename_rule_order",
                "joint_rename_order",
            ):
                assert dict(group[name].example.facts) == dict(expected_rename.facts)
                assert dict(group[name].example.rules) == dict(expected_rename.rules)
        assert treatments["identity"].prompt == ProofGraphTask().render(source)
        assert treatments["entity_symbol_renaming"].prompt == ProofGraphTask().render(expected_rename)
        assert sum(controls[name].prompt != treatments[name].prompt for name in controls) == 2
    with pytest.raises(prep.StudentAnchorProbeError, match="unknown diagnostic arm"):
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
    assert result["optimizer_windows"] == 32 and result["executed_optimizer_windows"] == 12
    assert result["executed_training_views"] == 768 and result["executed_input_tokens"] == 2304
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
    with pytest.raises(prep.StudentAnchorProbeError, match="overlaps " + dimension):
        prep.reject_overlap(inventory, old)


def test_all_prior_order_bases_and_transformed_views_excluded(examples):
    inventory = prep.isolation_inventory(examples)
    old = prep.previous.make_examples(prep.DIFFICULTY)
    for rows in old.values():
        for row in rows:
            prep.reject_overlap(inventory, row)
    for view in [
        *prep.previous.fit_views(old["student_fit"], arm="control"),
        *prep.previous.fit_views(old["student_fit"], arm="treatment"),
        *prep.previous.development_views(old["student_dev"]),
    ]:
        prep.reject_overlap(inventory, view.example)
    with pytest.raises(prep.StudentAnchorProbeError, match="incomplete excluded population"):
        prep.audit_isolation(examples, [], {"teacher_fit": [], "teacher_dev": []})


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
    protocol = prep.proposed_student_anchor_probe_protocol()
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
    with pytest.raises(prep.StudentAnchorProbeError, match="training view"):
        prep.encode_view(view, token, config)
    token.prefix, token.response = 2198, 200
    assert prep.encode_view(view, token, config, training=False)["prefix_length"] == 2198
    token.prefix = 2199
    with pytest.raises(prep.StudentAnchorProbeError, match="development prefix"):
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
    with pytest.raises(prep.StudentAnchorProbeError, match="exceeds2M"):
        prep.validate_encoded_fit(rows)


def test_real_git_distinct_review_resolves_and_unrelated_followup_is_allowed(reviewed_repo):
    root, git, implementation = reviewed_repo
    result = prep.resolve_student_anchor_probe_protocol(root)
    assert result.implementation_commit == implementation
    assert result.acceptance_commit == git("rev-parse", "HEAD")
    (root / "unrelated.txt").write_text("does not change scientific source\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated follow-up")
    assert prep.resolve_student_anchor_probe_protocol(root).acceptance_commit == result.acceptance_commit


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
        prep.resolve_student_anchor_probe_protocol(root)


def test_real_git_second_protocol_change_and_invented_implementation_reject(reviewed_repo):
    root, git, _ = reviewed_repo
    path = root / prep.PROTOCOL_PATH
    protocol = json.loads(path.read_text())
    protocol["review"]["rationale"] += " second acceptance"
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Unreviewed second review")
    with pytest.raises(prep.StudentAnchorProbeError, match="exactly one"):
        prep.resolve_student_anchor_probe_protocol(root)


@pytest.mark.parametrize("protocol_path", sorted(prep.FROZEN_PROTOCOL_SHA256))
def test_all_six_frozen_parent_protocols_cannot_change(reviewed_repo, protocol_path):
    root, git, _ = reviewed_repo
    path = root / protocol_path
    path.write_text(path.read_text() + "\n")
    with pytest.raises(prep.StudentAnchorProbeError, match="frozen parent protocol"):
        prep.resolve_student_anchor_probe_protocol(root)


def test_every_training_window_preserves_signed_pair_structure_and_rank_view_balance(examples):
    bases = examples["student_fit"]
    views = prep.fit_views(bases)
    for start in range(0, len(views), 64):
        window = views[start : start + 64]
        assert Counter(v.example.metadata["structure"] for v in window) == {
            "branch": 32,
            "chain": 16,
            "converging_dag": 16,
        }
        assert Counter(v.example.label for v in window) == {0: 32, 1: 32}
        groups = {}
        for view in window:
            groups.setdefault(view.example.pair_group_id, []).append(view)
        assert len(groups) == 4
        for rows in groups.values():
            assert len(rows) == 16 and Counter(v.example.label for v in rows) == {0: 8, 1: 8}
        for rank in (0, 1):
            shard = window[rank::2]
            assert Counter(v.view for v in shard) == dict.fromkeys(prep.FIT_VIEW_NAMES, 4)
            assert Counter(v.example.metadata["structure"] for v in shard) == {
                "branch": 16,
                "chain": 8,
                "converging_dag": 8,
            }
            assert (
                sum(
                    v.view
                    in (
                        "entity_symbol_renaming",
                        "rename_fact_order",
                        "rename_rule_order",
                        "joint_rename_order",
                    )
                    for v in shard
                )
                == 16
            )
            assert Counter(v.example.label for v in shard) == {0: 16, 1: 16}
    first_three_windows = bases[:24]
    for structure in prep.STRUCTURES:
        assert {x.metadata["depth"] for x in first_three_windows if x.metadata["structure"] == structure} == {
            2,
            3,
            4,
        }
    for row in bases:
        assert 4 <= row.metadata["distractors"] <= 8
    assert {row.metadata["distractors"] for row in examples["student_dev"]} == set(range(4, 17))


def test_both_signed_branch_targets_need_both_derived_antecedents(examples):
    task = ProofGraphTask()
    checked = Counter()
    for view in prep.fit_views(examples["student_fit"]):
        ex = view.example
        if ex.metadata["structure"] != "branch":
            continue
        assert len(ex.canonical_proof) == 2 * ex.metadata["depth"] - 1
        final = ex.canonical_proof[-1]
        assert len(final.citations) == 2 and all(c.startswith("S") for c in final.citations)
        broken = copy.deepcopy(ex)
        broken.canonical_proof = broken.canonical_proof[ex.metadata["depth"] - 1 :]
        assert task.verify(ex, task.parse_response(render_target(broken))).reward == 0
        assert task.verify(ex, task.parse_response(render_target(ex))).reward == 1
        checked[ex.label] += 1
    assert checked == {0: 512, 1: 512}


@pytest.mark.parametrize("path", sorted(prep.FROZEN_HELPER_SHA256))
def test_exact_frozen_execution_helper_mutation_rejects(reviewed_repo, path):
    root, _git, _implementation = reviewed_repo
    source = root / path
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(prep.StudentAnchorProbeError, match="frozen diagnostic helper"):
        prep.resolve_student_anchor_probe_protocol(root)


def test_primary_analysis_is_fixed_before_results_without_model_threshold():
    analysis = prep.proposed_student_anchor_probe_protocol()["analysis"]
    assert analysis["primary_steps"] == [6, 7, 8]
    assert analysis["lowering_IID_alone_is_not_improvement"] is True
    assert analysis["significance_or_acceptance_threshold"] is None
    assert analysis["checkpoint_selection"] is False
    assert analysis["primary_view_names"] == [
        "entity_symbol_renaming",
        "fact_order_permutation",
        "rule_order_permutation",
    ]
    assert analysis["primary_vector_must_not_be_summed_or_averaged_across_views"] is True
    assert analysis["retention_is_not_a_noninferiority_or_acceptance_test"] is True
    value = prep.proposed_student_anchor_probe_protocol()
    value["analysis"]["primary_steps"] = [12]
    with pytest.raises(prep.StudentAnchorProbeError, match="differs"):
        prep.validate_student_anchor_probe_protocol(value)


@pytest.mark.parametrize(
    "key,value",
    [
        ("anchor_view_names", ["entity_symbol_renaming"]),
        ("arm_joint_permutation_sequence_fraction", {"control": 1.0, "treatment": 0.5}),
        ("fit_renamed_sequence_fraction", 0.75),
    ],
)
def test_two_slot_intervention_cannot_change_or_be_combined_with_rename_dose(key, value):
    payload = prep.proposed_student_anchor_probe_protocol()
    payload["data"][key] = value
    with pytest.raises(prep.StudentAnchorProbeError, match="differs"):
        prep.validate_student_anchor_probe_protocol(payload)


def test_prior_probe_both_complete_arms_and_shared_dev_are_excluded_by_contract():
    data = prep.proposed_student_anchor_probe_protocol()["data"]
    assert data["isolation_population_counts"]["order_probe_student_fit"] == 256
    assert data["isolation_population_counts"]["order_probe_student_dev"] == 128
    counts = data["isolation_excluded_view_counts"]
    assert counts["order_probe_control_student_fit_views"] == 2048
    assert counts["order_probe_treatment_student_fit_views"] == 2048
    assert counts["order_probe_student_dev_views"] == 768
    changed = prep.proposed_student_anchor_probe_protocol()
    changed["data"]["isolation_excluded_view_counts"].pop("order_probe_treatment_student_fit_views")
    with pytest.raises(prep.StudentAnchorProbeError, match="differs"):
        prep.validate_student_anchor_probe_protocol(changed)


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


def test_actual_encoded_six_slots_equal_and_anchor_supervised_targets_unchanged(examples, arms):
    import hashlib

    class TextTokenizer:
        chat_template = "fixture"
        eos_token_id = 999

        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"] + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        def encode(self, text, *, add_special_tokens):
            # Every input byte contributes deterministically; unlike a length-only
            # fixture, changed prompts or targets cannot silently compare equal.
            raw = text.encode()
            return [
                int.from_bytes(hashlib.sha256(raw[i : i + 8]).digest()[:4], "big")
                for i in range(0, len(raw), 8)
            ]

    tokenizer = TextTokenizer()
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        }
    }
    checked = Counter()
    for control, treatment in zip(arms["control"], arms["treatment"], strict=True):
        c = prep.encode_view(control, tokenizer, config)
        t = prep.encode_view(treatment, tokenizer, config)
        assert c["labels"][c["prefix_length"] :] == t["labels"][t["prefix_length"] :]
        assert c["response_length"] == t["response_length"]
        if control.view not in prep.ANCHOR_VIEW_NAMES:
            assert {k: v for k, v in c.items() if k != "example_id"} == {
                k: v for k, v in t.items() if k != "example_id"
            }
            checked["unchanged_slots"] += 1
        else:
            assert c["input_ids"] != t["input_ids"]
            checked["anchors"] += 1
    assert checked == {"unchanged_slots": 1536, "anchors": 512}
