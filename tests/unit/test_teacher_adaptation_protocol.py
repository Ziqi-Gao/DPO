"""Real temporary Git histories test review authority, not invented commit IDs."""

from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from posttrain_circuits.artifacts.teacher_adaptation_protocol import (
    PROTOCOL_PATH,
    READINESS_THRESHOLDS,
    SCIENCE_PATHS,
    SUPPLEMENTAL_EXPERIMENT_NAMESPACE,
    TeacherAdaptationProtocolError,
    load_teacher_adaptation_protocol,
    proposed_teacher_adaptation_protocol,
    resolve_teacher_adaptation_protocol,
    teacher_adaptation_protocol_sha256,
    validate_teacher_adaptation_protocol,
    validate_teacher_adaptation_review_transition,
)

ROOT = Path(__file__).resolve().parents[2]


def _raw(value):
    return yaml.safe_dump(value, sort_keys=False).encode()


def _accepted(implementation):
    value = proposed_teacher_adaptation_protocol()
    value["review"] = {
        "status": "accepted",
        "reviewed_implementation_commit": implementation,
        "reviewer": "independent test reviewer",
        "reviewed_at_utc": "2026-09-28T04:00:00Z",
        "rationale": "Independent review permits measurement; no teacher-quality acceptance.",
    }
    return value


def test_checked_in_protocol_fixes_science_without_accepting_teacher():
    actual = load_teacher_adaptation_protocol((ROOT / PROTOCOL_PATH).read_bytes())
    assert teacher_adaptation_protocol_sha256(actual) == teacher_adaptation_protocol_sha256(
        proposed_teacher_adaptation_protocol()
    )
    assert actual["science"]["readiness_thresholds"] == READINESS_THRESHOLDS
    assert actual["science"]["every_causal_side_structurally_valid_and_strictly_positive"] is True
    assert actual["science"]["max_new_tokens"] == 256
    assert actual["preserved_base"]["student_max_completion_length"] == 128
    assert actual["preserved_base"]["claim_original_128_budget_pass"] is False
    assert actual["scope"]["protocol_acceptance_is_teacher_acceptance"] is False
    assert actual["fit"]["evaluation_steps"] == [128, 256, 384, 512]
    assert actual["qualification"]["supplemental_experiment_namespace"] == SUPPLEMENTAL_EXPERIMENT_NAMESPACE
    assert actual["cohorts"]["supplemental"]["source_start"] == 128
    assert actual["cohorts"]["supplemental"]["seed_uses_original_source_index"] is True
    assert actual["qualification"]["ordered_stages"] == [
        "selected_checkpoint",
        "train_probe",
        "supplemental",
        "formal_readiness",
        "formal_store",
        "independent_artifact_acceptance",
    ]
    assert "teacher_checkpoint_sha256" not in actual["teacher"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("teacher", "base_revision"), "1" * 40),
        (("teacher", "enable_thinking"), 0),
        (("teacher", "kind"), "pinned_hf_base"),
        (("teacher", "checkpoint_identity"), "resolved_teacher_commit"),
        (("scope", "protocol_acceptance_is_teacher_acceptance"), True),
        (("science", "max_new_tokens"), 128),
        (("science", "prefix_top_k"), 64),
        (("science", "every_causal_side_structurally_valid_and_strictly_positive"), False),
        (("science", "readiness_thresholds", "minimum_teacher_answer_accuracy"), 0.89),
        (("preserved_base", "student_max_completion_length"), 256),
        (("fit", "train_examples"), 8191),
        (("fit", "selection_rule"), "best_checkpoint"),
        (("fit", "evaluation_steps"), [512]),
        (("fit", "reselect_after_confirmation_exposure"), True),
        (("cohorts", "train_probe", "required_covered_prompts"), 31),
        (("cohorts", "supplemental", "source_start"), 0),
        (("cohorts", "supplemental", "seed_uses_original_source_index"), False),
        (("cohorts", "formal_store", "required_attempts"), 256),
        (("qualification", "supplemental_experiment_namespace"), "checkpoint-specific-new-claim"),
        (("qualification", "preserve_claim_on_failure_or_unknown_outcome"), False),
        (("candidate_generation", "request_seed"), 42),
        (("candidate_generation", "retry_resample_or_repair_generated_proof"), True),
        (("candidate_generation", "temperature"), float("nan")),
        (("science_files",), ["src/fake.py"]),
    ],
)
def test_fixed_specification_rejects_mutations_and_type_coercion(path, replacement):
    value = proposed_teacher_adaptation_protocol()
    target = value
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = replacement
    with pytest.raises(TeacherAdaptationProtocolError):
        validate_teacher_adaptation_protocol(value)


@pytest.mark.parametrize("field", ["review", "teacher", "science", "cohorts"])
def test_unknown_fields_fail_closed(field):
    value = proposed_teacher_adaptation_protocol()
    value[field]["unreviewed"] = "anything"
    with pytest.raises(TeacherAdaptationProtocolError):
        load_teacher_adaptation_protocol(_raw(value))


def test_yaml_duplicate_alias_nonstring_and_unknown_tags_rejected():
    raw = _raw(proposed_teacher_adaptation_protocol())
    for broken in (
        raw + b"schema_version: 1\n",
        b"x: &item []\ny: *item\n",
        b"1: invalid\n",
        b"x: !!python/object/apply:os.system ['false']\n",
        b"",
        b"x" * (2 * 1024 * 1024 + 1),
    ):
        with pytest.raises(TeacherAdaptationProtocolError):
            load_teacher_adaptation_protocol(broken)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "approved"),
        ("reviewed_implementation_commit", "a" * 64),
        ("reviewed_implementation_commit", "short"),
        ("reviewer", "  "),
        ("rationale", None),
        ("reviewed_at_utc", "2026-09-28T04:00:00"),
        ("reviewed_at_utc", "2026-09-28Z"),
        ("reviewed_at_utc", "2026-13-99T04:00:00Z"),
    ],
)
def test_accepted_review_requires_complete_genuine_metadata(field, value):
    payload = _accepted("a" * 40)
    payload["review"][field] = value
    with pytest.raises(TeacherAdaptationProtocolError):
        load_teacher_adaptation_protocol(_raw(payload), require_accepted=True)


def test_review_transition_core_hash_is_stable_but_raw_review_hash_changes():
    proposed = proposed_teacher_adaptation_protocol()
    accepted = _accepted("a" * 40)
    validate_teacher_adaptation_review_transition(
        proposed, accepted, implementation_commit="a" * 40, acceptance_commit="b" * 40
    )
    assert teacher_adaptation_protocol_sha256(proposed) == teacher_adaptation_protocol_sha256(accepted)
    assert hashlib.sha256(_raw(proposed)).digest() != hashlib.sha256(_raw(accepted)).digest()
    with pytest.raises(TeacherAdaptationProtocolError, match="distinct"):
        validate_teacher_adaptation_review_transition(
            proposed, accepted, implementation_commit="a" * 40, acceptance_commit="a" * 40
        )
    with pytest.raises(TeacherAdaptationProtocolError, match="proposed"):
        load_teacher_adaptation_protocol(_raw(proposed), require_accepted=True)


class History:
    def __init__(self, root):
        self.root = root
        self.metadata = root / ".git"
        self.env = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_AUTHOR_NAME": "Protocol test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Protocol test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }
        root.mkdir()
        subprocess.run(["/usr/bin/git", "init", "-q", str(root)], env=self.env, check=True)
        for name in SCIENCE_PATHS:
            # Each fixture is a real committed source inventory. Unavailable new
            # producer paths get harmless fixture text, not fake Git identities.
            self.write(name, f"reviewed fixture: {name}\n".encode())
        self.write("prereg/qwen3_v2.yaml", (ROOT / "prereg/qwen3_v2.yaml").read_bytes())
        self.write(PROTOCOL_PATH, _raw(proposed_teacher_adaptation_protocol()))
        self.write("docs/refactor/current_handoff.md", b"fixture handoff\n")
        self.write("notes.txt", b"unrelated tracked note\n")
        self.implementation = self.commit("implementation")
        self.acceptance = None

    def git(self, *args):
        return (
            subprocess.run(
                ["/usr/bin/git", "--git-dir", str(self.metadata), "--work-tree", str(self.root), *args],
                env=self.env,
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def commit(self, message):
        self.git("add", "--all")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def accept(self):
        self.write(PROTOCOL_PATH, _raw(_accepted(self.implementation)))
        self.acceptance = self.commit("independent review only")
        return self.acceptance

    def resolve(self, **kwargs):
        return resolve_teacher_adaptation_protocol(self.root, git_dir=self.metadata, **kwargs)


@pytest.fixture
def history(tmp_path):
    return History(tmp_path / "review-history")


def test_real_implementation_and_review_only_ancestry_with_unrelated_commits(history):
    history.write("notes.txt", b"non science between implementation and review\n")
    history.commit("unrelated intervening commit")
    history.accept()
    history.write("notes.txt", b"later non science commit\n")
    head = history.commit("later unrelated commit")
    result = history.resolve(expected_head=head)
    assert result.implementation_commit == history.implementation
    assert result.acceptance_commit == history.acceptance
    assert result.head == head
    assert result.review_status == "accepted"
    assert result.tracked_worktree_clean is True
    assert set(result.science_file_sha256) == set(SCIENCE_PATHS)
    assert len(result.science_implementation_sha256) == 64
    assert result.payload["scope"]["protocol_acceptance_is_teacher_acceptance"] is False


def test_unrelated_dirty_note_is_reported_honestly_not_fake_clean_head(history):
    history.accept()
    history.write("notes.txt", b"user uncommitted work\n")
    result = history.resolve()
    assert result.tracked_worktree_clean is False
    assert (history.root / "notes.txt").read_bytes() == b"user uncommitted work\n"


def test_quest_metadata_location_must_be_explicit(history):
    history.accept()
    history.metadata.rename(history.root / ".opd-git")
    history.metadata = history.root / ".opd-git"
    with pytest.raises(TeacherAdaptationProtocolError, match="directory"):
        resolve_teacher_adaptation_protocol(history.root)
    assert history.resolve().acceptance_commit == history.acceptance


def test_proposed_does_not_resolve_as_accepted_but_can_be_inspected(history):
    with pytest.raises(TeacherAdaptationProtocolError, match="proposed"):
        history.resolve()
    inspected = history.resolve(require_accepted=False)
    assert inspected.implementation_commit is None and inspected.acceptance_commit is None
    assert inspected.review_status == "proposed"


def test_uncommitted_acceptance_and_self_review_do_not_resolve(history):
    history.write(PROTOCOL_PATH, _raw(_accepted(history.implementation)))
    with pytest.raises(TeacherAdaptationProtocolError, match="distinct ancestor"):
        history.resolve()


def test_acceptance_cannot_contain_code_or_unrelated_changes(history):
    history.write(SCIENCE_PATHS[0], b"code changed together with acceptance\n")
    history.accept()
    with pytest.raises(TeacherAdaptationProtocolError, match="non-review"):
        history.resolve()


@pytest.mark.parametrize("mode", ["dirty", "staged", "later_commit", "later_commit_worktree_reverted"])
def test_named_science_must_match_reviewed_bytes_and_real_head(history, mode):
    history.accept()
    name = SCIENCE_PATHS[0]
    original = (history.root / name).read_bytes()
    history.write(name, b"changed reviewed science\n")
    if mode == "staged":
        history.git("add", name)
        history.write(name, original)
    elif mode.startswith("later_commit"):
        history.commit("unreviewed source change")
        if mode == "later_commit_worktree_reverted":
            history.write(name, original)
    with pytest.raises(TeacherAdaptationProtocolError, match="scientific|science|staged"):
        history.resolve()


@pytest.mark.parametrize("mode", ["missing", "symlink", "ancestor_symlink"])
def test_science_inventory_rejects_missing_or_linked_sources(history, mode):
    history.accept()
    path = history.root / SCIENCE_PATHS[0]
    if mode == "ancestor_symlink":
        directory = history.root / "configs"
        directory.rename(history.root / "relocated")
        directory.symlink_to(history.root / "relocated", target_is_directory=True)
    else:
        value = path.read_bytes()
        path.unlink()
        if mode == "symlink":
            other = history.root / "same-source-bytes"
            other.write_bytes(value)
            path.symlink_to(other)
    with pytest.raises(ValueError):
        history.resolve()


def test_second_protocol_edit_after_acceptance_is_rejected_even_if_review_only(history):
    history.accept()
    payload = _accepted(history.implementation)
    payload["review"]["rationale"] += " Later changed."
    history.write(PROTOCOL_PATH, _raw(payload))
    history.commit("second protocol edit")
    with pytest.raises(TeacherAdaptationProtocolError, match="exactly one"):
        history.resolve()


def test_fake_or_wrong_head_and_changed_frozen_preregistration_rejected(history):
    history.accept()
    with pytest.raises(TeacherAdaptationProtocolError, match="HEAD"):
        history.resolve(expected_head="f" * 40)
    history.write("prereg/qwen3_v2.yaml", b"changed original preregistration\n")
    with pytest.raises(TeacherAdaptationProtocolError, match="frozen preregistration"):
        history.resolve()


def test_unrelated_implementation_commit_cannot_be_claimed_as_ancestor(history):
    history.git("checkout", "--orphan", "unrelated-root")
    history.write("notes.txt", b"other root\n")
    unrelated = history.commit("unrelated root")
    history.git("checkout", "-B", "returned", history.implementation)
    history.write(PROTOCOL_PATH, _raw(_accepted(unrelated)))
    history.commit("claim unrelated implementation")
    with pytest.raises(TeacherAdaptationProtocolError, match="Git provenance|ancestor"):
        history.resolve()


def test_shallow_history_cannot_manufacture_provenance(history):
    history.accept()
    (history.metadata / "shallow").write_text(history.implementation + "\n")
    with pytest.raises(TeacherAdaptationProtocolError, match="complete real Git history"):
        history.resolve()


def test_head_change_during_source_validation_fails_closed(history, monkeypatch):
    import posttrain_circuits.artifacts.teacher_adaptation_protocol as module

    history.accept()
    original = module.read_regular_bytes_nofollow
    triggered = False

    def changed(path, **kwargs):
        nonlocal triggered
        raw = original(path, **kwargs)
        if path.name == "sdsc_teacher_qualify_job.sh" and not triggered:
            triggered = True
            history.write("notes.txt", b"concurrent commit\n")
            history.commit("concurrent head movement")
        return raw

    monkeypatch.setattr(module, "read_regular_bytes_nofollow", changed)
    with pytest.raises(TeacherAdaptationProtocolError, match="changed during"):
        history.resolve()


def test_git_environment_cannot_inject_alternate_repository(history, monkeypatch):
    history.accept()
    monkeypatch.setenv("GIT_DIR", "/not-the-real-repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/not-the-real-worktree")
    assert history.resolve().implementation_commit == history.implementation


def test_proposed_factory_returns_independent_structures():
    first = proposed_teacher_adaptation_protocol()
    untouched = copy.deepcopy(first)
    first["fit"]["lora"]["rank"] = 999
    assert proposed_teacher_adaptation_protocol() == untouched
