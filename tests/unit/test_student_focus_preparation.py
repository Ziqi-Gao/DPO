"""Regression for augmented views, robust checkpoint selection and genuine review."""

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
from posttrain_circuits.experiments.protocols import student_focus_preparation as prep

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def examples():
    return prep.make_examples(prep.DIFFICULTY)


def test_protocol_preserves_formal_thresholds_and_discloses_new_preparation():
    value = prep.load_student_focus_preparation_protocol((ROOT / prep.PROTOCOL_PATH).read_bytes())
    value["review"] = copy.deepcopy(prep.REVIEW_PROPOSED)
    assert value == prep.proposed_student_focus_preparation_protocol()
    with pytest.raises(prep.StudentFocusPreparationError, match="not accepted"):
        prep.validate_student_focus_preparation_protocol(value, require_accepted=True)
    assert value["scope"]["historical_checkpoints_reselected"] is False
    assert value["preserved_base"]["all_original_numerical_gates_unchanged"] is True
    assert value["training"]["input_token_budget"] == 2_000_000
    assert value["data"]["max_model_input_tokens"] == 1536
    assert value["development"]["max_model_input_tokens"] == 2454


@pytest.mark.parametrize(
    "section,key,new",
    [
        ("data", "fit_raw_pair_seed_start", 42),
        ("data", "filter_truncate_or_repair_rows", True),
        ("training", "optimizer_steps", 256),
        ("training", "input_token_budget", 3_000_000),
        ("development", "maximum_iid_minus_each_transformation_gap", 0.10),
        ("development", "minimum_each_structure_iid_proof_accuracy", 0),
        ("development", "maximum_answer_correct_count", 256),
        ("development", "reselection_after_formal_exposure", True),
        ("scope", "preparation_is_student_or_g0_acceptance", True),
        ("execution", "gpu_count", True),
    ],
)
def test_protocol_rejects_unreviewed_design_change(section, key, new):
    value = prep.proposed_student_focus_preparation_protocol()
    value[section][key] = new
    with pytest.raises(prep.StudentFocusPreparationError, match="differs"):
        prep.validate_student_focus_preparation_protocol(value)


def test_frozen_views_cover_order_and_renaming_without_corrupting_targets(examples):
    manifest = prep.dataset_manifest(examples)
    assert sha256_value(manifest) == prep.DATASET_MANIFEST_SHA256
    task = ProofGraphTask()
    for role, views, names in (
        ("student_fit", prep.fit_views(examples["student_fit"]), prep.FIT_VIEW_NAMES),
        ("student_dev", prep.development_views(examples["student_dev"]), prep.DEV_VIEW_NAMES),
    ):
        assert Counter(v.view for v in views) == {name: len(examples[role]) for name in names}
        assert len({v.example.example_id for v in views}) == len(views)
        assert sha256_value([asdict(v) for v in views]) == prep.VIEWS_SHA256[role]
        for start, source in enumerate(examples[role]):
            group = views[start * len(names) : (start + 1) * len(names)]
            assert {v.source_example_id for v in group} == {source.example_id}
            assert {v.example.pair_group_id for v in group} == {source.pair_group_id}
            for view in group:
                assert task.verify(view.example, task.parse_response(render_target(view.example))).reward == 1
            keyed = {v.view: v for v in group}
            fact_view = "rename_fact_order" if role == "student_fit" else "fact_order_permutation"
            rule_view = "rename_rule_order" if role == "student_fit" else "rule_order_permutation"
            assert list(keyed[fact_view].example.facts) != list(source.facts)
            assert list(keyed[rule_view].example.rules) != list(source.rules)
        if role == "student_fit":
            assert set(Counter(v.example.pair_group_id for v in views).values()) == {16}


@pytest.mark.parametrize("kind", ["drop", "order", "duplicate", "target", "seed"])
def test_base_population_cannot_be_reselected_or_rewritten(examples, kind):
    changed = copy.deepcopy(examples)
    if kind == "drop":
        changed["student_fit"].pop()
    elif kind == "order":
        changed["student_dev"].reverse()
    elif kind == "duplicate":
        changed["student_fit"][1] = changed["student_fit"][0]
    elif kind == "target":
        changed["student_fit"][0].canonical_proof = []
    else:
        changed["student_fit"][0].metadata["pair_seed"] = 42
    with pytest.raises(ValueError):
        prep.validate_examples(changed)


def counts(n, correct):
    return dict(num_examples=n, answer_correct=correct, proof_correct=correct, format_valid=n)


def dev_record(step, index):
    strata = {"chain": counts(86, 34), "branch": counts(68, 27), "converging_dag": counts(102, 39)}
    return dict(
        step=step,
        checkpoint_sha256=f"{index + 1:064x}",
        examples_sha256=prep.EXAMPLES_SHA256["student_dev"],
        **counts(256, 100),
        per_view_counts={name: counts(256, 100) for name in prep.DEV_VIEW_NAMES},
        structure_counts=strata,
    )


def test_checkpoint_selection_requires_robustness_and_branch_ability():
    rows = [dev_record(step, i) for i, step in enumerate(prep.CHECKPOINT_STEPS)]
    # Aggregate base accuracy is fine; one formerly hidden collapsed ordering must fail.
    rows[0]["per_view_counts"]["rule_order_permutation"] = counts(256, 70)
    # All transforms fine; the formerly invisible branch=0 failure must fail.
    rows[1]["structure_counts"]["branch"] = counts(68, 0)
    rows[1]["structure_counts"]["chain"] = counts(86, 61)
    selected = prep.select_checkpoint(rows)
    assert selected["step"] == 3
    for row in rows:
        row["per_view_counts"]["fact_order_permutation"] = counts(256, 70)
    assert prep.select_checkpoint(rows) is None


@pytest.mark.parametrize(
    "kind",
    ["missing_step", "duplicate_checkpoint", "bad_view_count", "bad_structure_count", "wrong_dev", "boolean"],
)
def test_selection_rejects_incomplete_or_inconsistent_evidence(kind):
    rows = [dev_record(step, i) for i, step in enumerate(prep.CHECKPOINT_STEPS)]
    if kind == "missing_step":
        rows.pop()
    elif kind == "duplicate_checkpoint":
        rows[1]["checkpoint_sha256"] = rows[0]["checkpoint_sha256"]
    elif kind == "bad_view_count":
        rows[0]["per_view_counts"]["identity"]["num_examples"] = 255
    elif kind == "bad_structure_count":
        rows[0]["structure_counts"]["branch"]["num_examples"] = 71
    elif kind == "wrong_dev":
        rows[0]["examples_sha256"] = "0" * 64
    else:
        rows[0]["answer_correct"] = True
    with pytest.raises(prep.StudentFocusPreparationError):
        prep.select_checkpoint(rows)


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
    with pytest.raises(prep.StudentFocusPreparationError, match="training view"):
        prep.encode_view(view, token, config)
    token.prefix, token.response = 2198, 200
    assert prep.encode_view(view, token, config, training=False)["prefix_length"] == 2198
    token.prefix = 2199
    with pytest.raises(prep.StudentFocusPreparationError, match="development prefix"):
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
    with pytest.raises(prep.StudentFocusPreparationError, match="exceeds2M"):
        prep.validate_encoded_fit(rows)


def test_real_git_distinct_review_resolves_and_unrelated_followup_is_allowed(reviewed_repo):
    root, git, implementation = reviewed_repo
    result = prep.resolve_student_focus_preparation_protocol(root)
    assert result.implementation_commit == implementation
    assert result.acceptance_commit == git("rev-parse", "HEAD")
    (root / "unrelated.txt").write_text("does not change scientific source\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated follow-up")
    assert prep.resolve_student_focus_preparation_protocol(root).acceptance_commit == result.acceptance_commit


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
        prep.resolve_student_focus_preparation_protocol(root)


def test_real_git_second_protocol_change_and_invented_implementation_reject(reviewed_repo):
    root, git, _ = reviewed_repo
    path = root / prep.PROTOCOL_PATH
    protocol = json.loads(path.read_text())
    protocol["review"]["rationale"] += " second acceptance"
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Unreviewed second review")
    with pytest.raises(prep.StudentFocusPreparationError, match="exactly one"):
        prep.resolve_student_focus_preparation_protocol(root)


@pytest.mark.parametrize("protocol_path", sorted(prep.FROZEN_PROTOCOL_SHA256))
def test_all_four_frozen_parent_protocols_cannot_change(reviewed_repo, protocol_path):
    root, git, _ = reviewed_repo
    path = root / protocol_path
    path.write_text(path.read_text() + "\n")
    with pytest.raises(prep.StudentFocusPreparationError, match="frozen parent protocol"):
        prep.resolve_student_focus_preparation_protocol(root)


def test_every_training_window_preserves_signed_pair_structure_and_rank_view_balance(examples):
    bases = examples["student_fit"]
    views = prep.fit_views(bases)
    for start in range(0, len(views), 64):
        window = views[start : start + 64]
        assert Counter(v.example.metadata["structure"] for v in window) == {
            "branch": 48,
            ("chain" if (start // 64) % 2 == 0 else "converging_dag"): 16,
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
                "branch": 24,
                ("chain" if (start // 64) % 2 == 0 else "converging_dag"): 8,
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
    first_three_windows = bases[:48]
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
    assert checked == {0: 768, 1: 768}


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
    protocol = prep.proposed_student_focus_preparation_protocol()
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


@pytest.mark.parametrize("path", sorted(prep.FROZEN_HELPER_SHA256))
def test_exact_frozen_execution_helper_mutation_rejects(reviewed_repo, path):
    root, _git, _implementation = reviewed_repo
    source = root / path
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(prep.StudentFocusPreparationError, match="frozen diagnostic helper"):
        prep.resolve_student_focus_preparation_protocol(root)


def test_numerical_training_and_development_contracts_match_original_v4():
    old = prep.preparation.proposed_student_order_preparation_protocol()
    new = prep.proposed_student_focus_preparation_protocol()
    assert new["training"] == old["training"]
    assert {k: v for k, v in new["development"].items() if k != "expected_raw_responses"} == old[
        "development"
    ]
    assert new["development"]["expected_raw_responses"] == 18432
    for k, v in old["execution"].items():
        assert new["execution"][k] == v
    assert new["execution"]["early_startup_timeout_seconds"] == 300
    assert new["execution"]["each_checkpoint_fixed2454_token_native_BF16_finite_forward_each_rank"] is True
    assert len(prep.FROZEN_SCIENCE_PATHS) == 148 and len(set(prep.SCIENCE_PATHS)) == 154


def test_new_sampling_keeps_exact_depth_sign_view_and_anchor_policy(examples):
    task = ProofGraphTask()
    bases = examples["student_fit"]
    views = prep.fit_views(bases)
    assert Counter(x.metadata["structure"] for x in bases) == dict(branch=192, chain=32, converging_dag=32)
    for i, source in enumerate(bases):
        window, slot = divmod(i // 2, 4)
        assert source.metadata["depth"] == 2 + (window + slot) % 3
        expected = "branch" if slot < 3 else ("chain" if window % 2 == 0 else "converging_dag")
        assert source.metadata["structure"] == expected
        group = {v.view: v for v in views[i * 8 : (i + 1) * 8]}
        for name, v in group.items():
            assert v.example.label == source.label and v.example.pair_group_id == source.pair_group_id
            if name in prep.ANCHOR_VIEW_NAMES:
                assert list(v.example.facts) == list(source.facts)
                assert list(v.example.rules) == list(source.rules)
            else:
                assert set(v.example.facts) == set(source.facts)
                assert set(v.example.rules) == set(source.rules)
            renamed = name in (
                "entity_symbol_renaming",
                "rename_fact_order",
                "rename_rule_order",
                "joint_rename_order",
            )
            assert (v.example.query != source.query) == renamed
            assert task.verify(v.example, task.parse_response(render_target(v.example))).reward == 1
    for start in range(0, 2048, 64):
        for rank in (0, 1):
            for v in views[start : start + 64][rank::2]:
                assert v.example.label == (1 - rank if prep.FIT_VIEW_NAMES.index(v.view) % 2 == 0 else rank)


def test_all_prior_diagnostic_populations_and_views_remain_excluded(examples):
    inventory = prep.isolation_inventory(examples)
    payload = prep.proposed_student_focus_preparation_protocol()["data"]
    for prefix, module in (
        ("order_probe", prep.order_diagnostic),
        ("anchor_probe", prep.anchor_diagnostic),
        ("batch_probe", prep.previous),
    ):
        b = module.make_examples(prep.DIFFICULTY)
        assert payload["isolation_population_counts"][prefix + "_student_fit"] == 256
        assert payload["isolation_population_counts"][prefix + "_student_dev"] == 128
        for rows in b.values():
            for row in rows:
                prep.reject_overlap(inventory, row)
        for arm in module.ARMS:
            assert (
                payload["isolation_excluded_view_counts"][prefix + "_" + arm + "_student_fit_views"] == 2048
            )
            for v in module.fit_views(b["student_fit"], arm=arm):
                prep.reject_overlap(inventory, v.example)
        for v in module.development_views(b["student_dev"]):
            prep.reject_overlap(inventory, v.example)
    with pytest.raises(prep.StudentFocusPreparationError, match="overlaps"):
        prep.reject_overlap(inventory, examples["student_fit"][0])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("data", "fit_structure_sequence_fractions", {"branch": 1.0}),
        ("data", "fit_renamed_sequence_fraction", 0.75),
        ("data", "fit_original_order_anchor_sequence_fraction", 0.125),
        ("data", "fit_joint_permutation_sequence_fraction", 1.0),
        ("execution", "early_startup_timeout_seconds", 120),
        ("execution", "fit_walltime_minutes", 360),
        ("scope", "formal_generated_responses_or_scores_read", True),
    ],
)
def test_structure_change_cannot_hide_other_policy_changes(section, key, value):
    p = prep.proposed_student_focus_preparation_protocol()
    p[section][key] = value
    with pytest.raises(prep.StudentFocusPreparationError):
        prep.validate_student_focus_preparation_protocol(p)


@pytest.mark.parametrize("raw", [b'{"review":{},"review":{}}', b"[]", b"", b"{", b"NaN"])
def test_strict_protocol_json_rejects_ambiguity(raw):
    with pytest.raises(prep.StudentFocusPreparationError):
        prep.load_student_focus_preparation_protocol(raw)


def test_composed_permutations_keep_frozen_rows_including_two_net_identity_cases(examples):
    bases = examples["student_fit"]
    views = prep.fit_views(bases)
    counts = Counter()
    unchanged = []
    for i, source in enumerate(bases):
        for v in views[8 * i : 8 * i + 8]:
            if v.view in prep.ANCHOR_VIEW_NAMES:
                continue
            counts["applied"] += 1
            for name in ("facts", "rules"):
                if list(getattr(v.example, name)) != list(getattr(source, name)):
                    counts[name] += 1
                else:
                    unchanged.append((source.example_id, v.view, name))
    assert counts == dict(applied=1536, facts=1534, rules=1536)
    assert unchanged == [
        ("pgpair-159676acb84d1ec7eb8d-neg", "joint_rename_order", "facts"),
        ("pgpair-6782070b5ae625aa47a8-neg", "fact_order_permutation", "facts"),
    ]
    # Applying two legitimate permutations need not change the final order.
    # Preserve those rows; do not silently reroll, filter or change the policy.
    assert prep.proposed_student_focus_preparation_protocol()["data"][
        "measured_fit_net_order_changed_rows"
    ] == dict(facts=1534, rules=1536)
