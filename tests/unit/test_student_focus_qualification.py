"""A draft cannot authorize a model; one reviewed candidate retains original gates."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from posttrain_circuits.experiments.protocols import student_focus_preparation as parent
from posttrain_circuits.experiments.protocols import student_focus_qualification as qualification
from posttrain_circuits.experiments.protocols import student_qualification as original

ROOT = Path(__file__).resolve().parents[2]
FIT_SOURCE_HEAD = "3" * 40


def candidate(step=6):
    """Synthetic identity only; never exported as an accepted experiment artifact."""
    digest = lambda label: hashlib.sha256(label.encode()).hexdigest()  # noqa: E731
    return dict(
        fit_job_id="90000001",
        fit_intent="1" * 32,
        fit_plan_sha256=digest("synthetic-plan"),
        fit_source_head=FIT_SOURCE_HEAD,
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
    payload = qualification.proposed_student_focus_qualification_protocol()
    assert payload["prepared_initial"] is None
    assert qualification.validate_student_focus_qualification_protocol(payload) == payload
    assert qualification.validate_prepared_initial(None, require_bound=False) is None
    with pytest.raises(qualification.StudentFocusQualificationError, match="not bound"):
        qualification.validate_prepared_initial(None)
    with pytest.raises(qualification.StudentFocusQualificationError, match="not accepted"):
        qualification.validate_student_focus_qualification_protocol(payload, require_accepted=True)
    for require_accepted in (True, False):
        with pytest.raises(qualification.StudentFocusQualificationError, match="not bound"):
            qualification.validate_student_focus_qualification_protocol(
                reviewed(payload), require_accepted=require_accepted
            )


def test_non_step4_candidate_is_dynamic_and_independently_copied():
    binding = candidate()
    checked = qualification.validate_prepared_initial(binding)
    assert checked == binding and checked is not binding
    payload = qualification.proposed_student_focus_qualification_protocol(binding)
    assert payload["prepared_initial"]["checkpoint_step"] == 6
    accepted = reviewed(payload)
    assert (
        qualification.validate_student_focus_qualification_protocol(accepted, require_accepted=True)
        == accepted
    )
    binding["checkpoint_step"] = 4
    assert payload["prepared_initial"]["checkpoint_step"] == 6
    other = qualification.proposed_student_focus_qualification_protocol(candidate(3))
    assert qualification.protocol_core_sha256(payload) != qualification.protocol_core_sha256(other)


@pytest.mark.parametrize("step", parent.CHECKPOINT_STEPS)
def test_only_declared_parent_checkpoint_schedule_is_admissible(step):
    assert qualification.validate_prepared_initial(candidate(step))["checkpoint_step"] == step


@pytest.mark.parametrize(
    "key,value",
    [
        ("fit_source_head", None),
        ("fit_source_head", "0" * 40),
        ("fit_source_head", "A" * 40),
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
    with pytest.raises(qualification.StudentFocusQualificationError):
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
    with pytest.raises(qualification.StudentFocusQualificationError, match="digest"):
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
    with pytest.raises(qualification.StudentFocusQualificationError, match="fields"):
        qualification.validate_prepared_initial(binding)


def test_original_formal_science_is_identical_and_no_preparation_gates_leak():
    old = original.proposed_student_qualification_protocol()
    new = qualification.proposed_student_focus_qualification_protocol()
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
    payload = qualification.proposed_student_focus_qualification_protocol()
    spec = payload["parent_fit_admission"]
    assert spec["development_steps"] == list(parent.CHECKPOINT_STEPS)
    assert (spec["optimizer_steps"], spec["training_input_tokens"]) == (32, 1665064)
    assert (spec["development_prompts"], spec["raw_responses"]) == (1536, 18432)
    assert spec["null_selection_admissible"] is False
    assert spec["reload_ranks"] == [0, 1]
    assert spec["reload_fp32_parameter_count"] == 311
    assert spec["parent_report_flags_must_be_false"] == list(original.FLAGS)
    assert qualification.FROZEN_SCIENCE_PATHS == parent.SCIENCE_PATHS
    assert len(qualification.FROZEN_SCIENCE_PATHS) == 154
    assert len(qualification.SCIENCE_PATHS) == len(set(qualification.SCIENCE_PATHS)) == 160
    assert len(qualification.FROZEN_PROTOCOL_SHA256) == 9
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
    payload = qualification.proposed_student_focus_qualification_protocol(candidate())
    payload[section][key] = value
    with pytest.raises(qualification.StudentFocusQualificationError, match="differs"):
        qualification.validate_student_focus_qualification_protocol(payload)


def test_protocol_loader_rejects_duplicate_keys_bad_review_and_oversize():
    with pytest.raises(qualification.StudentFocusQualificationError, match="duplicate"):
        qualification.load_student_focus_qualification_protocol(b'{"review":null,"review":{}}')
    with pytest.raises(qualification.StudentFocusQualificationError, match="byte bound"):
        qualification.load_student_focus_qualification_protocol(b" " * (4 * 1024**2 + 1))
    payload = qualification.proposed_student_focus_qualification_protocol(candidate())
    payload["review"] = {"status": "accepted"}
    with pytest.raises(qualification.StudentFocusQualificationError, match="review fields"):
        qualification.validate_student_focus_qualification_protocol(payload)


@pytest.fixture
def repo_builder(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["/usr/bin/git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Independent fixture")
    for relative in (
        "prereg/qwen3_v2.yaml",
        *qualification.FROZEN_PROTOCOL_SHA256,
        *qualification.FROZEN_EXECUTION_SHA256,
        *qualification.FROZEN_QUALIFICATION_HELPER_SHA256,
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "parent.py").write_text("frozen parent scientific source\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Preserved parent")
    monkeypatch.setattr(qualification, "PARENT_HEAD", git("rev-parse", "HEAD"))
    (tmp_path / "fit.txt").write_text("actual fit producer source fixture\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Actual later fit source fixture")
    monkeypatch.setitem(globals(), "FIT_SOURCE_HEAD", git("rev-parse", "HEAD"))
    monkeypatch.setattr(
        qualification,
        "SCIENCE_PATHS",
        ("parent.py", "kernel.py", *qualification.FROZEN_QUALIFICATION_HELPER_SHA256),
    )
    monkeypatch.setattr(qualification, "FROZEN_SCIENCE_PATHS", ("parent.py",))

    def build(*, bound=True, accept=True, changed_parent=False, acceptance_extra=False, same_head=False):
        if same_head:
            monkeypatch.setitem(globals(), "FIT_SOURCE_HEAD", qualification.PARENT_HEAD)
        (tmp_path / "kernel.py").write_text("new reviewed implementation\n")
        if changed_parent:
            (tmp_path / "parent.py").write_text("forbidden parent change before implementation\n")
        payload = qualification.proposed_student_focus_qualification_protocol(candidate() if bound else None)
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
    resolved = qualification.resolve_student_focus_qualification_protocol(root)
    assert resolved.implementation_commit == implementation
    assert resolved.acceptance_commit == git("rev-parse", "HEAD")
    assert resolved.payload["prepared_initial"]["checkpoint_step"] == 6
    assert resolved.tracked_worktree_clean
    (root / "unrelated.txt").write_text("unrelated documentation\n")
    git("add", "unrelated.txt")
    git("commit", "--quiet", "-m", "Unrelated later commit")
    later = qualification.resolve_student_focus_qualification_protocol(root)
    assert later.acceptance_commit == resolved.acceptance_commit
    assert later.protocol_sha256 == resolved.protocol_sha256


def test_real_git_unbound_draft_resolves_only_as_nonproduction(repo_builder):
    root, _, _ = repo_builder(bound=False, accept=False)
    draft = qualification.resolve_student_focus_qualification_protocol(root, require_accepted=False)
    assert draft.payload["prepared_initial"] is None
    with pytest.raises(qualification.StudentFocusQualificationError, match="not accepted"):
        qualification.resolve_student_focus_qualification_protocol(root)


def test_real_git_null_candidate_cannot_be_accepted(repo_builder):
    root, _, _ = repo_builder(bound=False)
    with pytest.raises(qualification.StudentFocusQualificationError, match="not bound"):
        qualification.resolve_student_focus_qualification_protocol(root, require_accepted=False)


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
        qualification.resolve_student_focus_qualification_protocol(root)


@pytest.mark.parametrize("path", qualification.FROZEN_PROTOCOL_SHA256)
@pytest.mark.parametrize("mutation", ["dirty", "staged_only", "committed"])
def test_real_git_all_five_frozen_parent_protocols_reject_change(repo_builder, path, mutation):
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
    with pytest.raises(qualification.StudentFocusQualificationError, match="frozen parent protocol"):
        qualification.resolve_student_focus_qualification_protocol(root)


def test_real_git_parent_cannot_be_changed_in_new_implementation(repo_builder):
    root, _, _ = repo_builder(changed_parent=True)
    with pytest.raises(qualification.StudentFocusQualificationError, match="frozen parent student science"):
        qualification.resolve_student_focus_qualification_protocol(root)


def test_real_git_acceptance_cannot_change_unrelated_files(repo_builder):
    root, _, _ = repo_builder(acceptance_extra=True)
    with pytest.raises(qualification.StudentFocusQualificationError, match="non-review changes"):
        qualification.resolve_student_focus_qualification_protocol(root)


def test_real_git_candidate_swap_in_acceptance_is_rejected(repo_builder):
    root, git, implementation = repo_builder(accept=False)
    payload = qualification.proposed_student_focus_qualification_protocol(candidate(3))
    path = root / qualification.PROTOCOL_PATH
    path.write_text(json.dumps(reviewed(payload, implementation)))
    git("add", qualification.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Attempted candidate swap during review")
    with pytest.raises(qualification.StudentFocusQualificationError, match="changes scientific protocol"):
        qualification.resolve_student_focus_qualification_protocol(root)


def test_real_git_second_protocol_change_is_rejected(repo_builder):
    root, git, _ = repo_builder()
    path = root / qualification.PROTOCOL_PATH
    payload = json.loads(path.read_bytes())
    payload["prepared_initial"] = candidate(3)
    path.write_text(json.dumps(payload))
    git("add", qualification.PROTOCOL_PATH)
    git("commit", "--quiet", "-m", "Attempted reselection after acceptance")
    with pytest.raises(qualification.StudentFocusQualificationError, match="exactly one"):
        qualification.resolve_student_focus_qualification_protocol(root)


def test_real_git_self_acceptance_is_rejected(repo_builder):
    root, _, implementation = repo_builder(accept=False)
    payload = reviewed(
        qualification.proposed_student_focus_qualification_protocol(candidate()), implementation
    )
    (root / qualification.PROTOCOL_PATH).write_text(json.dumps(payload))
    with pytest.raises(qualification.StudentFocusQualificationError, match="distinct ancestor"):
        qualification.resolve_student_focus_qualification_protocol(root)


def test_parent_binding_is_exact_accepted_order_preparation_and_preserves_original_prompts():
    payload = qualification.proposed_student_focus_qualification_protocol()
    frozen = payload["preserved_base"]
    assert frozen["parent_science_head"] == "71d0508407d429e5f700a3908671af1244b63f1f"
    assert frozen["parent_implementation_commit"] == "711db871f2c3cff54be324ec4fe1ea3178a63b11"
    assert frozen["parent_acceptance_commit"] == frozen["parent_science_head"]
    assert frozen["parent_student_protocol"] == parent.PROTOCOL_PATH
    assert (
        frozen["parent_student_protocol_sha256"]
        == hashlib.sha256((ROOT / parent.PROTOCOL_PATH).read_bytes()).hexdigest()
    )
    parent_payload = parent.load_student_focus_preparation_protocol(
        (ROOT / parent.PROTOCOL_PATH).read_bytes(), require_accepted=True
    )
    assert frozen["parent_protocol_core_sha256"] == parent.protocol_core_sha256(parent_payload)
    assert (
        frozen["parent_auditor_sha256"]
        == hashlib.sha256((ROOT / "tools/sdsc_student_focus_audit.py").read_bytes()).hexdigest()
    )
    assert payload["parent_fit_admission"]["task"] == "qwen3-v2-student-focus-preparation-v5"
    assert payload["parent_fit_admission"]["parent_dense_format"] == "student_focus_preparation_dense_v5"
    assert payload["parent_fit_admission"]["parent_dense_scope"] == "student_focus_preparation_model_only"
    assert payload["prepared_initial"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("training_input_tokens", 1652820),
        ("task", "qwen3-v2-student-branch-preparation-v3"),
        ("parent_dense_format", "student_branch_dense_v3"),
        ("parent_dense_scope", "common_student_branch_model_only"),
    ],
)
def test_historical_branch_parent_evidence_cannot_authorize_order_candidate(key, value):
    payload = qualification.proposed_student_focus_qualification_protocol(candidate())
    payload["parent_fit_admission"][key] = value
    with pytest.raises(qualification.StudentFocusQualificationError, match="differs"):
        qualification.validate_student_focus_qualification_protocol(payload)


@pytest.mark.parametrize("filename", qualification.FROZEN_EXECUTION_SHA256)
@pytest.mark.parametrize("mutation", ["dirty", "staged_only", "committed"])
def test_real_git_every_accepted_recovery_dependency_is_frozen(repo_builder, filename, mutation):
    root, git, _ = repo_builder()
    path = root / filename
    old = path.read_bytes()
    path.write_bytes(old + b"\n")
    if mutation in {"staged_only", "committed"}:
        git("add", filename)
    if mutation == "staged_only":
        path.write_bytes(old)
    elif mutation == "committed":
        git("commit", "--quiet", "-m", "Unreviewed execution mutation")
    with pytest.raises(qualification.StudentFocusQualificationError, match="parent execution"):
        qualification.resolve_student_focus_qualification_protocol(root)


@pytest.mark.parametrize("mutation", ["working", "staged_only", "committed"])
def test_real_git_named_qualification_helper_is_immutable(repo_builder, mutation):
    root, git, _ = repo_builder()
    relative = next(iter(qualification.FROZEN_QUALIFICATION_HELPER_SHA256))
    path = root / relative
    old = path.read_bytes()
    path.write_bytes(old + b"\n")
    if mutation != "working":
        git("add", relative)
    if mutation == "staged_only":
        path.write_bytes(old)
    elif mutation == "committed":
        git("commit", "--quiet", "-m", "Unreviewed pure helper change")
    with pytest.raises(qualification.StudentFocusQualificationError):
        qualification.resolve_student_focus_qualification_protocol(root)


def completion_fixture():
    """Synthetic future fit identities; no real candidate or GPU claim."""
    value = candidate()
    value["fit_source_head"] = qualification.PARENT_HEAD
    plan = dict(
        schema="quest-sdsc-student-focus-plan-v1",
        task="qwen3-v2-student-focus-preparation-v5",
        mode="fit",
        intent_id=value["fit_intent"],
        run_id="synthetic-run",
        code_sha256="a" * 64,
        control_sha256=dict(qualification.FROZEN_EXECUTION_SHA256),
        provenance=dict(head=value["fit_source_head"]),
        protocol=dict(
            protocol_sha256=qualification.PARENT_PROTOCOL_CORE_SHA256,
            artifact_sha256=qualification.PARENT_PROTOCOL_SHA256,
            implementation_commit=qualification.PARENT_IMPLEMENTATION_COMMIT,
            acceptance_commit=qualification.PARENT_HEAD,
            head=value["fit_source_head"],
        ),
        **dict.fromkeys(qualification.FLAGS, False),
    )
    raw = qualification._canonical
    value["fit_plan_sha256"] = hashlib.sha256(raw(plan)).hexdigest()
    publication = dict(
        schema="quest-sdsc-student-focus-publication-v1",
        task=plan["task"],
        mode="fit",
        job_id=value["fit_job_id"],
        intent_id=value["fit_intent"],
        run_id=plan["run_id"],
        code_sha256=plan["code_sha256"],
        plan_sha256=value["fit_plan_sha256"],
        passed=True,
        stage_complete=True,
        preparation_complete=True,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        files=[dict(path="prepare-report.json", size=1, sha256=value["fit_report_sha256"])],
        large_files=[
            dict(
                path=value["checkpoint_path"],
                size=value["checkpoint_size"],
                sha256=value["checkpoint_sha256"],
            )
        ],
        **dict.fromkeys(qualification.FLAGS, False),
    )
    publication_raw = raw(publication)
    value["fit_publication_sha256"] = hashlib.sha256(publication_raw).hexdigest()
    status = dict(
        accounting_complete=True,
        success=True,
        stage_complete=True,
        preparation_complete=True,
        publication_verified=True,
        artifact_hashes_verified=True,
        state="COMPLETED",
        job_id=value["fit_job_id"],
        result_sha256=value["fit_report_sha256"],
        publication=publication,
        publication_sha256=value["fit_publication_sha256"],
        queue=dict(returncode=0, stdout=""),
        accounting=dict(returncode=0),
        **dict.fromkeys(qualification.FLAGS, False),
    )
    return value, dict(fit_plan=plan, status=status, publication_raw=publication_raw)


def test_flat_completion_same_accepted_parent_head_is_admitted():
    value, kwargs = completion_fixture()
    assert qualification.validate_parent_completion_evidence(value, **kwargs) == value
    assert "terminal" not in kwargs["status"]


def test_real_git_same_accepted_parent_head_is_admitted(repo_builder):
    root, _, _ = repo_builder(same_head=True)
    assert (
        qualification.resolve_student_focus_qualification_protocol(root).payload["prepared_initial"][
            "fit_source_head"
        ]
        == qualification.PARENT_HEAD
    )


@pytest.mark.parametrize(
    "field",
    [
        "accounting_complete",
        "success",
        "stage_complete",
        "preparation_complete",
        "publication_verified",
        "artifact_hashes_verified",
    ],
)
@pytest.mark.parametrize("bad", [False, None, 1, "true"])
def test_flat_completion_requires_typed_complete_success(field, bad):
    value, kwargs = completion_fixture()
    kwargs["status"][field] = bad
    with pytest.raises(qualification.StudentFocusQualificationError):
        qualification.validate_parent_completion_evidence(value, **kwargs)


@pytest.mark.parametrize(
    "case",
    [
        "preflight",
        "old_schema",
        "old_task",
        "control",
        "source_head",
        "old_protocol",
        "queue",
        "accounting",
        "report",
        "publication_bytes",
        "selected_size",
        "duplicate_checkpoint",
        "formal_acceptance",
    ],
)
def test_rehashed_flat_completion_rejects_contradictions(case):
    value, kwargs = completion_fixture()
    plan = kwargs["fit_plan"]
    status = kwargs["status"]
    pub = status["publication"]
    if case == "preflight":
        plan["mode"] = "preflight"
    elif case == "old_schema":
        plan["schema"] = "quest-sdsc-student-order-execution-plan-v2"
    elif case == "old_task":
        plan["task"] = "qwen3-v2-student-order-preparation-v4"
    elif case == "control":
        plan["control_sha256"]["tools/sdsc_student_focus.py"] = "f" * 64
    elif case == "source_head":
        plan["provenance"]["head"] = "e" * 40
    elif case == "old_protocol":
        plan["protocol"]["artifact_sha256"] = "a" * 64
    elif case == "queue":
        status["queue"]["stdout"] = "90000001|COMPLETED"
    elif case == "accounting":
        status["accounting"]["returncode"] = 1
    elif case == "report":
        pub["files"][0]["sha256"] = "f" * 64
    elif case == "selected_size":
        pub["large_files"][0]["size"] += 1
    elif case == "duplicate_checkpoint":
        pub["large_files"].append(dict(pub["large_files"][0]))
    elif case == "formal_acceptance":
        pub["student_accepted"] = True
    value["fit_plan_sha256"] = hashlib.sha256(qualification._canonical(plan)).hexdigest()
    pub["plan_sha256"] = value["fit_plan_sha256"]
    kwargs["publication_raw"] = qualification._canonical(pub)
    value["fit_publication_sha256"] = status["publication_sha256"] = hashlib.sha256(
        kwargs["publication_raw"]
    ).hexdigest()
    if case == "publication_bytes":
        kwargs["publication_raw"] += b" "
    with pytest.raises(qualification.StudentFocusQualificationError):
        qualification.validate_parent_completion_evidence(value, **kwargs)


def test_actual_completed_focus_preflight_producer_cannot_supply_fit_completion():
    path = ROOT / ".sdsc/diagnostics/student-focus-v5/watch-preflight-54661792/last-status.json"
    if not path.is_file():
        pytest.skip("actual local producer evidence is optional")
    value, kwargs = completion_fixture()
    status = json.loads(path.read_bytes())
    assert (
        status["success"] is True
        and status["accounting_complete"] is True
        and status["preparation_complete"] is False
    )
    kwargs["status"] = status
    with pytest.raises(qualification.StudentFocusQualificationError):
        qualification.validate_parent_completion_evidence(value, **kwargs)


def test_explicit_parent_and_scoring_helper_maps_are_distinct_and_current():
    assert len(qualification.FROZEN_PARENT_CONTROL_SHA256) == 5
    assert len(qualification.FROZEN_EXECUTION_SHA256) == 16
    assert not set(qualification.FROZEN_QUALIFICATION_HELPER_SHA256) & set(
        qualification.FROZEN_EXECUTION_SHA256
    )
    for path, digest in {
        **qualification.FROZEN_EXECUTION_SHA256,
        **qualification.FROZEN_QUALIFICATION_HELPER_SHA256,
    }.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert sum(qualification.PARENT_WINDOW_INPUT_TOKENS) == 1665064
    assert qualification.PARENT_CHECKPOINT_INPUT_TOKENS["4"] == 205428


@pytest.mark.parametrize("retained_queue", [False, True])
def test_frozen_flat_accounting_producer_shape_consumed_without_invented_terminal(retained_queue):
    """Real frozen accounting parser; publications remain explicit synthetic fixtures."""
    value, kwargs = completion_fixture()
    spec = importlib.util.spec_from_file_location(
        "qualification_flat_parent", ROOT / "tools/sdsc_student_focus.py"
    )
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    plan = kwargs["fit_plan"]
    plan["job_name"] = "opd-synthetic-fixture"
    value["fit_plan_sha256"] = hashlib.sha256(qualification._canonical(plan)).hexdigest()
    pub = kwargs["status"]["publication"]
    pub["plan_sha256"] = value["fit_plan_sha256"]
    kwargs["publication_raw"] = qualification._canonical(pub)
    value["fit_publication_sha256"] = kwargs["status"]["publication_sha256"] = hashlib.sha256(
        kwargs["publication_raw"]
    ).hexdigest()
    job = value["fit_job_id"]
    queue = dict(returncode=0, stdout=f"{job}|COMPLETED" if retained_queue else "", stderr="")
    account = dict(
        returncode=0,
        stderr="",
        stdout=(
            f"{job}|COMPLETED|0:0|opd-synthetic-fixture|nwu181|nairr-gpu-shared|nairr-gpu-shared-normal|24|384G|60|"
            f"cpu=24,gres/gpu=2,node=1|{value['fit_intent']}\n"
            f"{job}.batch|COMPLETED|0:0|batch||||||||\n"
            f"{job}.extern|COMPLETED|0:0|extern||||||||\n"
        ),
    )
    observed = controller.validate_accounting(plan, job, queue, account)
    assert "terminal" not in observed
    assert observed["accounting_complete"] is (not retained_queue)
    kwargs["status"].update(observed)
    if retained_queue:
        with pytest.raises(qualification.StudentFocusQualificationError):
            qualification.validate_parent_completion_evidence(value, **kwargs)
    else:
        assert qualification.validate_parent_completion_evidence(value, **kwargs) == value
