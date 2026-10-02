"""Freeze the selected bytes, original gates and genuine review lineage."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from posttrain_circuits.experiments.protocols import student_qualification as prep

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_artifact_matches_frozen_science():
    protocol = prep.load_student_qualification_protocol((ROOT / prep.PROTOCOL_PATH).read_bytes())
    protocol["review"] = copy.deepcopy(prep.REVIEW_PROPOSED)
    assert protocol == prep.proposed_student_qualification_protocol()
    with pytest.raises(prep.StudentQualificationError, match="not accepted"):
        prep.validate_student_qualification_protocol(protocol, require_accepted=True)
    assert protocol["prepared_initial"]["checkpoint_sha256"] == prep.CHECKPOINT_SHA256
    assert protocol["anti_shortcut"]["max_model_input_length"] == 2244
    assert protocol["base"]["max_model_input_length"] == 1536


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("prepared_initial", "checkpoint_step", 32),
        ("prepared_initial", "checkpoint_sha256", "0" * 64),
        ("prepared_initial", "checkpoint_reselection_or_retraining", True),
        ("base", "minimum_answer_correct_count", 12),
        ("base", "num_examples", 127),
        ("base", "metric", "answer_tag"),
        ("anti_shortcut", "minimum_iid_accuracy", 0.09),
        ("anti_shortcut", "minimum_transformed_accuracy", 0.07),
        ("anti_shortcut", "minimum_per_transformation_accuracy", 0.04),
        ("anti_shortcut", "max_shortcut_gap", 0.06),
        ("anti_shortcut", "metric", "answer_correct"),
        ("anti_shortcut", "iid_examples_sha256", "0" * 64),
        ("anti_shortcut", "max_model_input_length", 1536),
        ("generation", "truncation", True),
        ("generation", "max_new_tokens", 128),
        ("scope", "producer_may_accept_complete_initial_or_g0", True),
        ("execution", "gpu_count", True),
    ],
)
def test_qualification_rejects_changed_model_scope_population_or_gates(section, key, value):
    protocol = prep.proposed_student_qualification_protocol()
    protocol[section][key] = value
    with pytest.raises(prep.StudentQualificationError, match="differs"):
        prep.validate_student_qualification_protocol(protocol)


def test_duplicate_json_and_invented_review_rejected():
    with pytest.raises(prep.StudentQualificationError, match="duplicate"):
        prep.load_student_qualification_protocol(b'{"review":null,"review":{}}')
    protocol = prep.proposed_student_qualification_protocol()
    protocol["review"] = {"status": "accepted"}
    with pytest.raises(prep.StudentQualificationError, match="review fields"):
        prep.validate_student_qualification_protocol(protocol)


@pytest.fixture
def reviewed_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Independent fixture")
    for relative in ("prereg/qwen3_v2.yaml", "prereg/amendments/qwen3_student_preparation_v1.json"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    git("add", ".")
    git("commit", "--quiet", "-m", "Preserved parent")
    monkeypatch.setattr(prep, "PARENT_HEAD", git("rev-parse", "HEAD"))
    monkeypatch.setattr(prep, "SCIENCE_PATHS", ("kernel.py",))
    monkeypatch.setattr(prep, "FROZEN_SCIENCE_PATHS", ())
    (tmp_path / "kernel.py").write_text("unchanged reviewed implementation\n")
    protocol = prep.proposed_student_qualification_protocol()
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
    result = prep.resolve_student_qualification_protocol(root)
    assert result.implementation_commit == implementation
    assert result.acceptance_commit == git("rev-parse", "HEAD")
    (root / "unrelated.txt").write_text("does not change scientific source\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated follow-up")
    assert prep.resolve_student_qualification_protocol(root).acceptance_commit == result.acceptance_commit


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
        prep.resolve_student_qualification_protocol(root)


def test_real_git_second_protocol_change_and_invented_implementation_reject(reviewed_repo):
    root, git, _ = reviewed_repo
    path = root / prep.PROTOCOL_PATH
    protocol = json.loads(path.read_text())
    protocol["review"]["rationale"] += " second acceptance"
    path.write_text(json.dumps(protocol))
    git("add", prep.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Unreviewed second review")
    with pytest.raises(prep.StudentQualificationError, match="exactly one"):
        prep.resolve_student_qualification_protocol(root)
