"""A draft cannot authorize a model; one reviewed candidate retains original gates."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from posttrain_circuits.experiments.protocols import student_branch_preparation as parent
from posttrain_circuits.experiments.protocols import student_branch_qualification as qualification
from posttrain_circuits.experiments.protocols import student_qualification as original

ROOT = Path(__file__).resolve().parents[2]


def candidate(step=6):
    """Synthetic identity only; never exported as an accepted experiment artifact."""
    digest = lambda label: hashlib.sha256(label.encode()).hexdigest()  # noqa: E731
    return dict(
        fit_job_id="90000001",
        fit_intent="1" * 32,
        fit_plan_sha256=digest("synthetic-plan"),
        fit_publication_sha256=digest("synthetic-publication"),
        fit_report_sha256=digest("synthetic-report"),
        independent_raw_audit_sha256=digest("synthetic-audit"),
        checkpoint_step=step,
        checkpoint_path=f"checkpoints/step-{step:08d}.pt",
        checkpoint_sha256=digest(f"synthetic-checkpoint-{step}"),
        checkpoint_size=8127108889,
        master_model_state_sha256=digest("synthetic-fp32-masters"),
        **qualification.PREPARED_INITIAL_POLICY,
    )


def reviewed(payload, implementation="1" * 40):
    value = copy.deepcopy(payload)
    value["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=implementation,
        reviewer="independent-fixture-reviewer",
        reviewed_at_utc="2026-10-03T00:00:00Z",
        rationale="Synthetic CPU provenance fixture; no real execution claim.",
    )
    return value


def test_unbound_draft_never_satisfies_production_or_accepted_validation():
    payload = qualification.proposed_student_branch_qualification_protocol()
    assert payload["prepared_initial"] is None
    assert qualification.validate_student_branch_qualification_protocol(payload) == payload
    assert qualification.validate_prepared_initial(None, require_bound=False) is None
    with pytest.raises(qualification.StudentBranchQualificationError, match="not bound"):
        qualification.validate_prepared_initial(None)
    with pytest.raises(qualification.StudentBranchQualificationError, match="not accepted"):
        qualification.validate_student_branch_qualification_protocol(payload, require_accepted=True)
    for require_accepted in (True, False):
        with pytest.raises(qualification.StudentBranchQualificationError, match="not bound"):
            qualification.validate_student_branch_qualification_protocol(
                reviewed(payload), require_accepted=require_accepted
            )


def test_non_step4_candidate_is_dynamic_and_independently_copied():
    binding = candidate()
    checked = qualification.validate_prepared_initial(binding)
    assert checked == binding and checked is not binding
    payload = qualification.proposed_student_branch_qualification_protocol(binding)
    assert payload["prepared_initial"]["checkpoint_step"] == 6
    accepted = reviewed(payload)
    assert (
        qualification.validate_student_branch_qualification_protocol(accepted, require_accepted=True)
        == accepted
    )
    binding["checkpoint_step"] = 4
    assert payload["prepared_initial"]["checkpoint_step"] == 6
    other = qualification.proposed_student_branch_qualification_protocol(candidate(3))
    assert qualification.protocol_core_sha256(payload) != qualification.protocol_core_sha256(other)


@pytest.mark.parametrize("step", parent.CHECKPOINT_STEPS)
def test_only_declared_parent_checkpoint_schedule_is_admissible(step):
    assert qualification.validate_prepared_initial(candidate(step))["checkpoint_step"] == step


@pytest.mark.parametrize(
    "key,value",
    [
        ("fit_job_id", 90000001),
        ("fit_job_id", "0"),
        ("fit_job_id", "90000001_1"),
        ("fit_intent", "0" * 32),
        ("fit_intent", "UPPER"),
        ("checkpoint_step", True),
        ("checkpoint_step", 9),
        ("checkpoint_step", 6.0),
        ("checkpoint_path", "checkpoints/step-00000004.pt"),
        ("checkpoint_path", "../step-00000006.pt"),
        ("checkpoint_path", "/checkpoints/step-00000006.pt"),
        ("checkpoint_path", "checkpoints/prepared-selected.pt"),
        ("checkpoint_size", True),
        ("checkpoint_size", 0),
        ("checkpoint_size", qualification.MAX_CHECKPOINT_BYTES + 1),
        ("checkpoint_reselection_or_retraining", True),
        ("checkpoint_reselection_or_retraining", 0),
        ("identity", "native_HF_revision"),
        ("load", "BF16_before_reload"),
        ("selection", "latest_checkpoint_after_formal_failure"),
    ],
)
def test_malformed_or_reselected_identity_is_rejected(key, value):
    binding = candidate()
    binding[key] = value
    with pytest.raises(qualification.StudentBranchQualificationError):
        qualification.validate_prepared_initial(binding)


@pytest.mark.parametrize(
    "key",
    [
        "fit_plan_sha256",
        "fit_publication_sha256",
        "fit_report_sha256",
        "independent_raw_audit_sha256",
        "checkpoint_sha256",
        "master_model_state_sha256",
    ],
)
@pytest.mark.parametrize("value", [None, "0" * 64, "A" * 64, "b" * 63])
def test_every_candidate_digest_is_required_and_not_a_placeholder(key, value):
    binding = candidate()
    binding[key] = value
    with pytest.raises(qualification.StudentBranchQualificationError, match="digest"):
        qualification.validate_prepared_initial(binding)


@pytest.mark.parametrize("mutation", ["missing_master", "extra", "list"])
def test_candidate_exact_shape_rejects_v1_v2_or_extra_fields(mutation):
    binding = candidate()
    if mutation == "missing_master":
        binding.pop("master_model_state_sha256")
    elif mutation == "extra":
        binding["override_checkpoint"] = "alternate.pt"
    else:
        binding = []
    with pytest.raises(qualification.StudentBranchQualificationError, match="fields"):
        qualification.validate_prepared_initial(binding)


def test_original_formal_science_is_identical_and_no_preparation_gates_leak():
    old = original.proposed_student_qualification_protocol()
    new = qualification.proposed_student_branch_qualification_protocol()
    for key in ("base", "anti_shortcut", "generation"):
        assert new[key] == old[key]
    for key, value in old["execution"].items():
        assert new["execution"][key] == value
    assert new["base"]["minimum_answer_correct_count"] == 13
    assert "maximum_answer_correct_count" not in new["base"]
    assert new["anti_shortcut"]["max_model_input_length"] == 2244
    assert new["base"]["max_model_input_length"] == 1536
    assert not any("structure" in key or "per_view_gap" in key for key in new["anti_shortcut"])
    assert new["scope"]["prior_complete_formal896_exposure_job"] == "54606205"
    assert new["scope"]["formal_population_is_unexposed_holdout"] is False
    assert new["evidence"]["formal_initial_g0_pilot_flags"] is False
    assert new["evidence"]["original_formal_prompts_sha256"] == qualification.FORMAL_PROMPTS_SHA256
    assert new["evidence"]["original_formal_prompts_size"] == 13466230


def test_parent_admission_counts_rank_masters_and_immutable_inventory():
    payload = qualification.proposed_student_branch_qualification_protocol()
    spec = payload["parent_fit_admission"]
    assert spec["development_steps"] == list(parent.CHECKPOINT_STEPS)
    assert (spec["optimizer_steps"], spec["training_input_tokens"]) == (32, 1652820)
    assert (spec["development_prompts"], spec["raw_responses"]) == (1536, 18432)
    assert spec["null_selection_admissible"] is False
    assert spec["reload_ranks"] == [0, 1]
    assert spec["reload_fp32_parameter_count"] == 311
    assert spec["parent_report_flags_must_be_false"] == list(original.FLAGS)
    assert qualification.FROZEN_SCIENCE_PATHS == parent.SCIENCE_PATHS
    assert len(qualification.FROZEN_SCIENCE_PATHS) == 125
    assert len(qualification.SCIENCE_PATHS) == len(set(qualification.SCIENCE_PATHS)) == 130
    assert len(qualification.FROZEN_PROTOCOL_SHA256) == 4
    for path, expected in qualification.FROZEN_PROTOCOL_SHA256.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("base", "minimum_answer_correct_count", 12),
        ("base", "num_examples", 127),
        ("anti_shortcut", "max_model_input_length", 2454),
        ("anti_shortcut", "max_shortcut_gap", 0.06),
        ("anti_shortcut", "minimum_iid_accuracy", 0.09),
        ("anti_shortcut", "minimum_transformed_accuracy", 0.07),
        ("anti_shortcut", "minimum_per_transformation_accuracy", 0.04),
        ("generation", "truncation", True),
        ("generation", "all_896_responses_required_even_if_one_gate_fails", False),
        ("parent_fit_admission", "raw_responses", 9216),
        ("parent_fit_admission", "development_prompts", 512),
        ("parent_fit_admission", "reload_ranks", [0]),
        ("parent_fit_admission", "null_selection_admissible", True),
        ("preserved_base", "parent_auditor_sha256", "0" * 64),
        ("preserved_base", "parent_protocol_core_sha256", "0" * 64),
        ("execution", "gpu_count", 2),
        ("execution", "host_memory_gib", 384),
        ("execution", "publication_reserve_seconds", 0),
    ],
)
def test_fixed_science_or_parent_admission_cannot_be_relaxed(section, key, value):
    payload = qualification.proposed_student_branch_qualification_protocol(candidate())
    payload[section][key] = value
    with pytest.raises(qualification.StudentBranchQualificationError, match="differs"):
        qualification.validate_student_branch_qualification_protocol(payload)


def test_protocol_loader_rejects_duplicate_keys_bad_review_and_oversize():
    with pytest.raises(qualification.StudentBranchQualificationError, match="duplicate"):
        qualification.load_student_branch_qualification_protocol(b'{"review":null,"review":{}}')
    with pytest.raises(qualification.StudentBranchQualificationError, match="byte bound"):
        qualification.load_student_branch_qualification_protocol(b" " * (4 * 1024**2 + 1))
    payload = qualification.proposed_student_branch_qualification_protocol(candidate())
    payload["review"] = {"status": "accepted"}
    with pytest.raises(qualification.StudentBranchQualificationError, match="review fields"):
        qualification.validate_student_branch_qualification_protocol(payload)


@pytest.fixture
def repo_builder(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Independent fixture")
    for relative in ("prereg/qwen3_v2.yaml", *qualification.FROZEN_PROTOCOL_SHA256):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "parent.py").write_text("frozen parent scientific source\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Preserved parent")
    monkeypatch.setattr(qualification, "PARENT_HEAD", git("rev-parse", "HEAD"))
    monkeypatch.setattr(qualification, "SCIENCE_PATHS", ("parent.py", "kernel.py"))
    monkeypatch.setattr(qualification, "FROZEN_SCIENCE_PATHS", ("parent.py",))

    def build(*, bound=True, accept=True, changed_parent=False, acceptance_extra=False):
        (tmp_path / "kernel.py").write_text("new reviewed implementation\n")
        if changed_parent:
            (tmp_path / "parent.py").write_text("forbidden parent change before implementation\n")
        payload = qualification.proposed_student_branch_qualification_protocol(candidate() if bound else None)
        path = tmp_path / qualification.PROTOCOL_PATH
        path.write_text(json.dumps(payload))
        git("add", ".")
        git("commit", "--quiet", "-m", "Proposed implementation")
        implementation = git("rev-parse", "HEAD")
        if accept:
            path.write_text(json.dumps(reviewed(payload, implementation)))
            if acceptance_extra:
                (tmp_path / "other.txt").write_text("not review metadata\n")
            git("add", ".")
            git("commit", "--quiet", "-m", "Distinct review acceptance")
        return tmp_path, git, implementation

    return build


def test_real_git_single_candidate_distinct_review_and_unrelated_followup(repo_builder):
    root, git, implementation = repo_builder()
    resolved = qualification.resolve_student_branch_qualification_protocol(root)
    assert resolved.implementation_commit == implementation
    assert resolved.acceptance_commit == git("rev-parse", "HEAD")
    assert resolved.payload["prepared_initial"]["checkpoint_step"] == 6
    assert resolved.tracked_worktree_clean
    (root / "unrelated.txt").write_text("unrelated documentation\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated later commit")
    later = qualification.resolve_student_branch_qualification_protocol(root)
    assert later.acceptance_commit == resolved.acceptance_commit
    assert later.protocol_sha256 == resolved.protocol_sha256


def test_real_git_unbound_draft_resolves_only_as_nonproduction(repo_builder):
    root, _, _ = repo_builder(bound=False, accept=False)
    draft = qualification.resolve_student_branch_qualification_protocol(root, require_accepted=False)
    assert draft.payload["prepared_initial"] is None
    with pytest.raises(qualification.StudentBranchQualificationError, match="not accepted"):
        qualification.resolve_student_branch_qualification_protocol(root)


def test_real_git_null_candidate_cannot_be_accepted(repo_builder):
    root, _, _ = repo_builder(bound=False)
    with pytest.raises(qualification.StudentBranchQualificationError, match="not bound"):
        qualification.resolve_student_branch_qualification_protocol(root, require_accepted=False)


@pytest.mark.parametrize("mutation", ["dirty", "staged", "committed", "symlink"])
@pytest.mark.parametrize("source", ["kernel.py", "parent.py"])
def test_real_git_named_sources_reject_mutation(repo_builder, mutation, source):
    root, git, _ = repo_builder()
    path = root / source
    if mutation == "symlink":
        path.unlink()
        path.symlink_to(root / "prereg/qwen3_v2.yaml")
    else:
        path.write_text("unreviewed mutation\n")
        if mutation in {"staged", "committed"}:
            git("add", source)
        if mutation == "committed":
            git("commit", "--quiet", "-m", "Unreviewed mutation")
    with pytest.raises((ValueError, OSError)):
        qualification.resolve_student_branch_qualification_protocol(root)


@pytest.mark.parametrize("path", qualification.FROZEN_PROTOCOL_SHA256)
@pytest.mark.parametrize("mutation", ["dirty", "staged_only", "committed"])
def test_real_git_all_four_frozen_parent_protocols_reject_change(repo_builder, path, mutation):
    root, git, _ = repo_builder()
    target = root / path
    original_bytes = target.read_bytes()
    target.write_bytes(original_bytes + b"\n")
    if mutation in {"staged_only", "committed"}:
        git("add", path)
    if mutation == "staged_only":
        target.write_bytes(original_bytes)
    elif mutation == "committed":
        git("commit", "--quiet", "-m", "Changed historical protocol")
    with pytest.raises(qualification.StudentBranchQualificationError, match="frozen parent protocol"):
        qualification.resolve_student_branch_qualification_protocol(root)


def test_real_git_parent_cannot_be_changed_in_new_implementation(repo_builder):
    root, _, _ = repo_builder(changed_parent=True)
    with pytest.raises(qualification.StudentBranchQualificationError, match="frozen parent student science"):
        qualification.resolve_student_branch_qualification_protocol(root)


def test_real_git_acceptance_cannot_change_unrelated_files(repo_builder):
    root, _, _ = repo_builder(acceptance_extra=True)
    with pytest.raises(qualification.StudentBranchQualificationError, match="non-review changes"):
        qualification.resolve_student_branch_qualification_protocol(root)


def test_real_git_candidate_swap_in_acceptance_is_rejected(repo_builder):
    root, git, implementation = repo_builder(accept=False)
    payload = qualification.proposed_student_branch_qualification_protocol(candidate(3))
    path = root / qualification.PROTOCOL_PATH
    path.write_text(json.dumps(reviewed(payload, implementation)))
    git("add", qualification.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Attempted candidate swap during review")
    with pytest.raises(qualification.StudentBranchQualificationError, match="changes scientific protocol"):
        qualification.resolve_student_branch_qualification_protocol(root)


def test_real_git_second_protocol_change_is_rejected(repo_builder):
    root, git, _ = repo_builder()
    path = root / qualification.PROTOCOL_PATH
    payload = json.loads(path.read_bytes())
    payload["prepared_initial"] = candidate(3)
    path.write_text(json.dumps(payload))
    git("add", qualification.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Attempted reselection after acceptance")
    with pytest.raises(qualification.StudentBranchQualificationError, match="exactly one"):
        qualification.resolve_student_branch_qualification_protocol(root)


def test_real_git_self_acceptance_is_rejected(repo_builder):
    root, _, implementation = repo_builder(accept=False)
    payload = reviewed(
        qualification.proposed_student_branch_qualification_protocol(candidate()), implementation
    )
    (root / qualification.PROTOCOL_PATH).write_text(json.dumps(payload))
    with pytest.raises(qualification.StudentBranchQualificationError, match="distinct ancestor"):
        qualification.resolve_student_branch_qualification_protocol(root)
