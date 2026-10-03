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
from posttrain_circuits.experiments.protocols import student_invariance as prep

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def examples():
    return prep.make_examples(prep.DIFFICULTY)


def test_protocol_preserves_formal_thresholds_and_discloses_new_preparation():
    value = prep.load_student_invariance_protocol((ROOT / prep.PROTOCOL_PATH).read_bytes())
    value["review"] = copy.deepcopy(prep.REVIEW_PROPOSED)
    assert value == prep.proposed_student_invariance_protocol()
    with pytest.raises(prep.StudentInvarianceError, match="not accepted"):
        prep.validate_student_invariance_protocol(value, require_accepted=True)
    assert value["scope"]["historical_checkpoints_reselected"] is False
    assert value["preserved_base"]["all_original_numerical_gates_unchanged"] is True
    assert value["training"]["input_token_budget"] == 8_000_000
    assert value["data"]["max_model_input_tokens"] == 1536
    assert value["development"]["max_model_input_tokens"] == 2454


@pytest.mark.parametrize(
    "section,key,new",
    [
        ("data", "fit_raw_pair_seed_start", 42),
        ("data", "filter_truncate_or_repair_rows", True),
        ("training", "optimizer_steps", 256),
        ("training", "input_token_budget", 9_000_000),
        ("development", "maximum_iid_minus_each_transformation_gap", 0.10),
        ("development", "minimum_each_structure_iid_proof_accuracy", 0),
        ("development", "maximum_answer_correct_count", 256),
        ("development", "reselection_after_formal_exposure", True),
        ("scope", "preparation_is_student_or_g0_acceptance", True),
        ("execution", "gpu_count", True),
    ],
)
def test_protocol_rejects_unreviewed_design_change(section, key, new):
    value = prep.proposed_student_invariance_protocol()
    value[section][key] = new
    with pytest.raises(prep.StudentInvarianceError, match="differs"):
        prep.validate_student_invariance_protocol(value)


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
            assert list(keyed["fact_order_permutation"].example.facts) != list(source.facts)
            assert list(keyed["rule_order_permutation"].example.rules) != list(source.rules)
        if role == "student_fit":
            assert set(Counter(v.example.pair_group_id for v in views).values()) == {8}


def test_new_bases_do_not_overlap_previous_preparation(examples):
    inventory = prep.isolation_inventory(examples)
    previous = prep.previous.make_examples(prep.DIFFICULTY)
    for rows in previous.values():
        for row in rows:
            prep.reject_overlap(inventory, row)
    with pytest.raises(prep.StudentInvarianceError, match="overlaps"):
        prep.reject_overlap(inventory, examples["student_fit"][0])
    with pytest.raises(prep.StudentInvarianceError, match="incomplete excluded population"):
        prep.audit_isolation(examples, [], {"teacher_fit": [], "teacher_dev": []})


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
    strata = {"chain": counts(100, 40), "branch": counts(58, 24), "converging_dag": counts(98, 36)}
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
    rows[1]["structure_counts"]["branch"] = counts(58, 0)
    rows[1]["structure_counts"]["chain"] = counts(100, 64)
    selected = prep.select_checkpoint(rows)
    assert selected["step"] == 16
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
        rows[0]["structure_counts"]["branch"]["num_examples"] = 59
    elif kind == "wrong_dev":
        rows[0]["examples_sha256"] = "0" * 64
    else:
        rows[0]["answer_correct"] = True
    with pytest.raises(prep.StudentInvarianceError):
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
    with pytest.raises(prep.StudentInvarianceError, match="training view"):
        prep.encode_view(view, token, config)
    token.prefix, token.response = 2198, 200
    assert prep.encode_view(view, token, config, training=False)["prefix_length"] == 2198
    token.prefix = 2199
    with pytest.raises(prep.StudentInvarianceError, match="development prefix"):
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
        for i in range(8192)
    ]
    result = prep.validate_encoded_fit(rows)
    assert result["optimizer_windows"] == 128 and result["window_input_tokens"] == [192] * 128
    for row in rows:
        row.update(input_ids=[101] * 1024 + [201, 999], labels=[-100] * 1024 + [201, 999], prefix_length=1024)
    with pytest.raises(prep.StudentInvarianceError, match="exceeds8M"):
        prep.validate_encoded_fit(rows)


@pytest.fixture
def reviewed_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Independent fixture")
    for relative in (
        "prereg/qwen3_v2.yaml",
        "prereg/amendments/qwen3_student_preparation_v1.json",
        "prereg/amendments/qwen3_student_qualification_v1.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    git("add", ".")
    git("commit", "--quiet", "-m", "Preserved parent")
    monkeypatch.setattr(prep, "PARENT_HEAD", git("rev-parse", "HEAD"))
    monkeypatch.setattr(prep, "SCIENCE_PATHS", ("kernel.py",))
    monkeypatch.setattr(prep, "FROZEN_SCIENCE_PATHS", ())
    (tmp_path / "kernel.py").write_text("unchanged reviewed implementation\n")
    protocol = prep.proposed_student_invariance_protocol()
    path = tmp_path / prep.PROTOCOL_PATH
    path.write_text(json.dumps(protocol))
    git("add", ".")
    git("commit", "--quiet", "-m", "Proposed implementation")
    implementation = git("rev-parse", "HEAD")
    protocol["review"] = {
        "status": "accepted",
        "reviewed_implementation_commit": implementation,
        "reviewer": "independent-test-reviewer",
        "reviewed_at_utc": "2026-10-01T03:00:00Z",
        "rationale": "Reviewed fixture implementation only.",
    }
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Independent review only")
    return tmp_path, git, implementation


def test_real_git_distinct_review_resolves_and_unrelated_followup_is_allowed(reviewed_repo):
    root, git, implementation = reviewed_repo
    result = prep.resolve_student_invariance_protocol(root)
    assert result.implementation_commit == implementation
    assert result.acceptance_commit == git("rev-parse", "HEAD")
    (root / "unrelated.txt").write_text("does not change scientific source\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated follow-up")
    assert prep.resolve_student_invariance_protocol(root).acceptance_commit == result.acceptance_commit


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
        prep.resolve_student_invariance_protocol(root)


def test_real_git_second_protocol_change_and_invented_implementation_reject(reviewed_repo):
    root, git, _ = reviewed_repo
    path = root / prep.PROTOCOL_PATH
    protocol = json.loads(path.read_text())
    protocol["review"]["rationale"] += " second acceptance"
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Unreviewed second review")
    with pytest.raises(prep.StudentInvarianceError, match="exactly one"):
        prep.resolve_student_invariance_protocol(root)


def test_frozen_qualification_protocol_cannot_change(reviewed_repo):
    root, git, _ = reviewed_repo
    path = root / "prereg/amendments/qwen3_student_qualification_v1.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(prep.StudentInvarianceError, match="frozen qualification"):
        prep.resolve_student_invariance_protocol(root)
