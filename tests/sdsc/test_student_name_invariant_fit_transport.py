"""Independent normal-fit transport: real claim/fence seams and dual genuine Git histories."""

import base64
import copy
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c(monkeypatch):
    real = subprocess.Popen

    def guarded(argv, *args, **kwargs):
        name = Path(argv[0]).name if isinstance(argv, list | tuple) else argv.split()[0]
        if name in {"ssh", "sbatch", "squeue", "sacct", "scontrol", "scancel", "srun"}:
            pytest.fail("local fixture attempted remote/Slurm execution")
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded)
    return load("_fit_transport", ROOT / "tools/sdsc_student_name_invariant_fit.py")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    previous_tests = load(
        "_previous_transport", ROOT / "tests/sdsc/test_student_name_invariant_recovery_transport.py"
    )
    previous = previous_tests.plan.__wrapped__(c.recovery, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "science", c.recovery.science)
    monkeypatch.setattr(c, "CONTROL", c.science.CONTROL)
    monkeypatch.setattr(c, "PROJECT", c.science.PROJECT)
    original_tests = load("_science_transport", ROOT / "tests/sdsc/test_student_name_invariant_transport.py")
    audit = original_tests.preflight_audit(c.science, previous["science_plan"])
    audit["document"]["job_id"] = "123456"
    audit["sha256"] = c.sha(c.canonical(audit["document"]) + b"\n")
    inner = c.build_inner(previous["science_plan"], audit)
    execution = dict(
        core_sha256="6" * 64,
        artifact_sha256="7" * 64,
        implementation_commit="5" * 40,
        acceptance_commit="6" * 40,
        head="6" * 40,
        control_file_sha256={p: c.sha((ROOT / p).read_bytes()) for p in c.NEW_TOOLS},
        science_binding=dict(inner["protocol"], head="6" * 40),
    )
    source = dict(
        run_id="separate-fit-execution",
        code_sha256="2" * 64,
        manifest_sha256="3" * 64,
        release=str(c.CONTROL / "releases/separate-fit-execution"),
        provenance=dict(
            directory=str(c.CONTROL / "provenance-v2" / ("9" * 64)), manifest_sha256="9" * 64, head="6" * 40
        ),
    )
    pins = dict(c.contract.PREFLIGHT)
    pins.update(
        job_id="123456",
        intent_id=previous["intent_id"],
        recovery_plan_sha256=c.plan_sha(previous),
        plan_sha256=c.plan_sha(previous["science_plan"]),
        audit_sha256=audit["sha256"],
        publication_sha256=audit["document"]["publication_sha256"],
        result_sha256=audit["document"]["result_sha256"],
        fence_sha256=previous["fence"]["sha256"],
    )
    monkeypatch.setattr(c.contract, "PREFLIGHT", pins)
    value = dict(
        schema=c.SCHEMA,
        task=c.TASK,
        intent_id=inner["intent_id"],
        science_plan=inner,
        science_plan_sha256=c.plan_sha(inner),
        source=source,
        execution=execution,
        preflight_recovery=previous,
        preflight_recovery_sha256=c.plan_sha(previous),
        **dict.fromkeys(c.FLAGS, False),
    )
    c.validate_plan(value)
    # The real predecessor controller publishes its normal receipt and distinct recovery claim.
    request, _ = previous_tests.submit_seam(c.recovery, previous, monkeypatch)
    c.recovery.remote_action(request)
    directory = Path(previous["science_plan"]["submission_dir"])
    entry = c.recovery.entry_record(previous, "123456", at="before", setup_elapsed_seconds=10)
    exit = c.recovery.exit_record(entry, returncode=0, node_returned=True, completed_at="after")
    for name, doc, key in [
        ("recovery-node-entry.json", entry, "entry_sha256"),
        ("recovery-node-exit.json", exit, "exit_sha256"),
    ]:
        raw = c.canonical(doc)
        (directory / name).write_bytes(raw)
        pins[key] = c.sha(raw)
    # One immutable helper instance permits observation instrumentation only in fixtures.
    original_helper = c.helper
    observed = c.recovery.helper("sdsc_observed_command")
    monkeypatch.setattr(
        c,
        "helper",
        lambda name, root=ROOT: observed if name == "sdsc_observed_command" else original_helper(name, root),
    )
    return value


def observed(c, argv, **kwargs):
    previous = load(
        "_observed_fixture", ROOT / "tests/sdsc/test_student_name_invariant_recovery_transport.py"
    )
    return previous.observe(c, argv, **kwargs)


def preflight_seam(c, plan, monkeypatch):
    calls = []

    def inspect(previous, receipt, *, accounting_snapshot=None):
        calls.append(accounting_snapshot)
        return dict(
            job_id=c.contract.PREFLIGHT["job_id"],
            success=True,
            stage_complete=True,
            execution_entry_verified=True,
            execution_exit_verified=True,
            publication_sha256=c.contract.PREFLIGHT["publication_sha256"],
            result_sha256=c.contract.PREFLIGHT["result_sha256"],
            queue={"returncode": 0, "stdout": "", "stderr": ""},
            accounting={"fixture": "COMPLETED 0:0"},
        )

    monkeypatch.setattr(c.recovery, "inspect_job", inspect)
    return calls


def submit_seam(c, plan, monkeypatch, result=None):
    calls = []
    monkeypatch.setattr(c, "verify_source", lambda *a, **k: None)
    preflight_seam(c, plan, monkeypatch)

    def admitted(value):
        prerequisite = c.verify_recovered_preflight(value)
        inner = value["science_plan"]
        for key in ["scientific_claim", "claim", "submission_dir", "result_dir"]:
            assert not Path(inner[key]).exists()
        return dict(preflight=prerequisite)

    monkeypatch.setattr(c, "admission", admitted)
    monkeypatch.setattr(
        c.science,
        "live_binding",
        lambda inner, job: dict(job_id=job, plan_sha256=c.plan_sha(inner), fixture=True),
    )

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        c.check_claims(plan)
        assert not (Path(plan["science_plan"]["submission_dir"]) / "receipt.json").exists()
        return result if result is not None else observed(c, argv, stdout=b"654321\n")

    monkeypatch.setattr(c.helper("sdsc_observed_command"), "run_observed", run)
    request = dict(
        action="submit",
        plan=plan,
        authorize=True,
        dry_run=dict(
            dry_run=True,
            blockers=[],
            plan_sha256=c.plan_sha(plan),
            science_plan_sha256=plan["science_plan_sha256"],
            argv=c.sbatch(plan),
        ),
    )
    return request, calls


def test_normal_fit_uses_old_scientific_source_new_execution_and_untrained_parent(c, plan):
    inner = plan["science_plan"]
    prior = plan["preflight_recovery"]["science_plan"]
    assert inner["mode"] == "fit" and inner["resources"]["time"] == "08:00:00"
    for key in [
        "run_id",
        "release",
        "provenance",
        "code_sha256",
        "protocol",
        "control_sha256",
        "runtime_snapshot",
        "checkpoints",
    ]:
        assert inner[key] == prior[key]
    assert inner["checkpoints"][0]["sha256"] == c.science.INITIAL_SHA
    assert inner["science_identity"]["optimizer_steps"] == 32
    assert inner["scientific_claim"] != prior["scientific_claim"]
    assert inner["release"] != plan["source"]["release"]
    script = c.job_script(plan).decode()
    assert str(Path(plan["source"]["release"]) / "source" / c.NODE) in script
    assert "--time=08:00:00" in c.sbatch(plan)
    assert c.sbatch(plan)[-1] == c.plan_sha(plan)
    assert all(plan[k] is False for k in c.FLAGS)


@pytest.mark.parametrize(
    "case",
    [
        "mode",
        "runtime",
        "resource",
        "checkpoint",
        "protocol",
        "oldsource",
        "newsource",
        "newhead",
        "audit",
        "prior",
        "claim",
        "extra",
        "acceptance",
        "control",
        "flag",
    ],
)
def test_changed_science_or_execution_rejected(c, plan, case):
    p = copy.deepcopy(plan)
    i = p["science_plan"]
    if case == "mode":
        i["mode"] = "preflight"
    elif case == "runtime":
        i["runtime_snapshot"]["archive"]["sha256"] = "0" * 64
    elif case == "resource":
        i["resources"]["time"] = "09:00:00"
    elif case == "checkpoint":
        i["checkpoints"][0]["sha256"] = "0" * 64
    elif case == "protocol":
        i["protocol"]["science_file_sha256"][next(iter(i["protocol"]["science_file_sha256"]))] = "0" * 64
    elif case == "oldsource":
        i["run_id"] = "different"
    elif case == "newsource":
        p["source"]["release"] = "/tmp/foreign"
    elif case == "newhead":
        p["execution"]["head"] = "8" * 40
    elif case == "audit":
        i["preflight"]["audit"]["document"]["raw_replay_passed"] = False
    elif case == "prior":
        p["preflight_recovery_sha256"] = "0" * 64
    elif case == "claim":
        i["scientific_claim"] = p["preflight_recovery"]["science_plan"]["scientific_claim"]
    elif case == "extra":
        p["recovery_claim"] = "forbidden-new-recovery-chain"
    elif case == "acceptance":
        p["execution"]["acceptance_commit"] = p["execution"]["implementation_commit"]
    elif case == "control":
        p["execution"]["control_file_sha256"].pop(c.NODE)
    else:
        p["student_accepted"] = True
    with pytest.raises((ValueError, KeyError)):
        c.validate_plan(p)


def test_real_claim_compatibility_gate_preserves_original_rejection(c, plan, monkeypatch):
    with pytest.raises(ValueError, match="preflight permanent claims"):
        c.science.verify_preflight(plan["science_plan"])
    calls = preflight_seam(c, plan, monkeypatch)
    result = c.verify_recovered_preflight(plan)
    assert result["success"] is True and calls == [None]
    inner = plan["science_plan"]
    directory = Path(inner["submission_dir"])
    directory.mkdir(parents=True)
    (directory / "admission.json").write_bytes(c.canonical(dict(preflight=result)))
    c.verify_recovered_preflight(plan, accounting=False)
    assert calls[-1] == result


@pytest.mark.parametrize(
    "target", ["oldclaim", "recoveryclaim", "priorplan", "entry", "exit", "observation", "receipt", "fence"]
)
def test_changed_actual_prior_evidence_fails(c, plan, monkeypatch, target):
    preflight_seam(c, plan, monkeypatch)
    p = plan["preflight_recovery"]
    d = Path(p["science_plan"]["submission_dir"])
    paths = {
        "oldclaim": Path(p["old_plan"]["scientific_claim"]),
        "recoveryclaim": Path(p["recovery_claim"]),
        "priorplan": d / "plan.json",
        "entry": d / "recovery-node-entry.json",
        "exit": d / "recovery-node-exit.json",
        "observation": d / "submission-observation.json",
        "receipt": d / "execution-receipt.json",
        "fence": Path(p["fence"]["path"]),
    }
    paths[target].write_bytes(b"{}")
    with pytest.raises((ValueError, KeyError)):
        c.verify_recovered_preflight(plan)


def test_once_submission_claims_and_durable_complete_ack(c, plan, monkeypatch):
    old_claim = Path(plan["preflight_recovery"]["old_plan"]["scientific_claim"])
    original = old_claim.read_bytes()
    request, calls = submit_seam(c, plan, monkeypatch)
    result = c.remote_action(request)
    assert result["job_id"] == "654321" and len(calls) == 1 and calls[0][1]["timeout"] == 180
    c.check_claims(plan)
    assert old_claim.read_bytes() == original
    assert not any(
        "recovery-claims" in str(p) for p in Path(plan["science_plan"]["submission_dir"]).iterdir()
    )
    with pytest.raises(AssertionError):
        c.remote_action(request)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"timed_out": True, "output_complete": False},
        {"output_limited": True},
        {"returncode": 1},
        {"child_reaped": False},
        {"stdout": b"not a job\n"},
    ],
)
def test_partial_ack_preserved_never_retried(c, plan, monkeypatch, changes):
    value = observed(c, c.sbatch(plan), **changes)
    request, calls = submit_seam(c, plan, monkeypatch, value)
    with pytest.raises(ValueError):
        c.remote_action(request)
    d = Path(plan["science_plan"]["submission_dir"])
    assert c.document(d / "submission-observation.json") == value
    assert c.document(d / "unknown.json")["no_retry"] is True
    assert Path(plan["science_plan"]["scientific_claim"]).exists()
    assert not (d / "receipt.json").exists() and len(calls) == 1


def test_ack_lost_after_durable_observation_reconciles_without_new_command(c, plan, monkeypatch):
    request, calls = submit_seam(c, plan, monkeypatch)
    original = c.record_receipt
    monkeypatch.setattr(
        c, "record_receipt", lambda *a, **k: (_ for _ in ()).throw(OSError("receipt write lost"))
    )
    with pytest.raises(OSError):
        c.remote_action(request)
    monkeypatch.setattr(c, "record_receipt", original)
    monkeypatch.setattr(
        c.helper("sdsc_observed_command"), "run_observed", lambda *a, **k: pytest.fail("new command")
    )
    assert c.reconcile(plan)["job_id"] == "654321" and len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("pre_model_guards_passed", 1),
        ("student_accepted", 0),
        ("setup_elapsed_seconds", 60.01),
        ("setup_elapsed_seconds", True),
        ("preflight_job_id", "foreign"),
    ],
)
def test_node_binding_strict_types_and_prerequisite(c, plan, field, value):
    entry = c.entry_record(plan, "654321", at="now", setup_elapsed_seconds=4)
    exit = c.exit_record(entry, returncode=0, node_returned=True, completed_at="after")
    c.validate_node_evidence(plan, "654321", entry, exit, require_success=True)
    entry[field] = value
    with pytest.raises(ValueError):
        c.validate_node_evidence(plan, "654321", entry, exit, require_success=True)


def test_failed_original_publication_fetch_keeps_science_bytes(c, plan, monkeypatch):
    request, _ = submit_seam(c, plan, monkeypatch)
    receipt = c.remote_action(request)
    inner = plan["science_plan"]
    root = Path(inner["result_dir"])
    root.mkdir(parents=True)
    original = c.helper("sdsc_student_name_invariant_job")
    original.persist(
        inner,
        c.science,
        None,
        None,
        dict(job_id=receipt["job_id"], stage_complete=False, exit_code=1),
        {},
        lambda label: None,
    )
    publication = c.document(root / "receipt.json")
    status = dict(
        success=False,
        publication_verified=True,
        publication=publication,
        publication_sha256=c.sha((root / "receipt.json").read_bytes()),
    )
    result = c.fetch(plan, receipt, status)
    files = c.validate_download(result, plan)
    assert files["receipt.json"] == (root / "receipt.json").read_bytes()
    assert files["execution/execution-plan.json"] == c.canonical(plan)
    result["files"]["execution/execution-receipt.json"] = base64.b64encode(b"{}").decode()
    with pytest.raises(ValueError):
        c.validate_download(result, plan)


def test_admission_reserves_old_two_and_new_two_only(c, plan, monkeypatch):
    binary = Path(plan["science_plan"]["python"])
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixture")
    binary.chmod(0o700)
    Path(plan["science_plan"]["hf_home"]).mkdir(parents=True)
    monkeypatch.setattr(c.recovery.contract, "PYTHON_SHA256", c.sha(binary.read_bytes()))
    monkeypatch.setattr(c, "verify_execution", lambda p: None)
    monkeypatch.setattr(c.science, "verify_parent", lambda *a, **k: None)
    monkeypatch.setattr(c.science, "verify_runtime", lambda p: None)
    monkeypatch.setattr(c.science, "verify_runtime_snapshot", lambda p: None)
    preflight_seam(c, plan, monkeypatch)
    monkeypatch.setattr(
        c.science, "check_gpu_ceiling", lambda: dict(existing_allocatable_gpus=0, new_gpus=2, limit=4)
    )
    assert c.admission(plan)["gpu_concurrency"]["total_reserved_gpus"] == 4
    monkeypatch.setattr(
        c.science, "check_gpu_ceiling", lambda: dict(existing_allocatable_gpus=1, new_gpus=2, limit=4)
    )
    with pytest.raises(ValueError, match="reserve unknown"):
        c.admission(plan)


def test_json_duplicates_and_nonfinite_rejected(c):
    for raw in [b'{"x":1,"x":2}', b'{"x":NaN}']:
        with pytest.raises(ValueError):
            c.decode(raw)


def test_exact_source_descriptor_and_both_source_verifiers(c, plan, monkeypatch):
    calls = []
    monkeypatch.setattr(c.science, "verify_source", lambda inner: calls.append(inner))
    source = Path(plan["source"]["release"])
    pins = {
        **c.contract.FROZEN_DEPENDENCIES,
        **plan["science_plan"]["protocol"]["science_file_sha256"],
        **plan["execution"]["control_file_sha256"],
        c.CONTRACT_PATH: c.sha((ROOT / c.CONTRACT_PATH).read_bytes()),
    }
    plan["execution"]["artifact_sha256"] = pins[c.CONTRACT_PATH]
    rows = []
    for name in sorted(pins):
        raw = (ROOT / name).read_bytes()
        dest = source / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        rows.append(dict(path=name, size=len(raw), sha256=c.sha(raw), mode=0o644))
    manifest = dict(
        run_id=plan["source"]["run_id"],
        code_sha256=c.sha(c.canonical(rows)),
        files=rows,
        total_bytes=sum(r["size"] for r in rows),
    )
    raw = c.canonical(manifest) + b"\n"
    (source / "manifest.json").write_bytes(raw)
    plan["source"].update(code_sha256=manifest["code_sha256"], manifest_sha256=c.sha(raw))
    assert c.verify_source(plan) == manifest and calls == [plan["science_plan"]]
    (source / "source" / c.CONTROLLER).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source byte"):
        c.verify_source(plan)


def test_real_dual_reviewed_v2_provenance_restores_original_science_and_new_execution(c, tmp_path):
    """Real >128-commit full exports on two genuine HEADs; no mocked resolver or restore."""
    root = tmp_path / "repository"
    provenance = c.helper("sdsc_provenance_v2")
    env = provenance.git_env()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, env=env
        ).stdout

    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(ROOT / ".opd-git"), str(root)],
        check=True,
        capture_output=True,
        env=env,
    )
    git("checkout", "--quiet", c.contract.PARENT_EXECUTION["acceptance_commit"])
    git("config", "user.name", "Independent fit fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("update-ref", "refs/remotes/public/master", "0215c356355b29b5e2b407978a207db2156719e1")

    def export(run, implementation, acceptance):
        names = git("ls-files", "-z").decode().split("\0")[:-1]
        rows = [
            dict(
                path=name,
                size=(root / name).stat().st_size,
                sha256=c.sha((root / name).read_bytes()),
                mode=0o755 if (root / name).stat().st_mode & 0o111 else 0o644,
            )
            for name in sorted(names)
        ]
        wrapper = dict(
            schema="quest-sdsc-snapshot-v1",
            run_id=run,
            git_head=acceptance,
            files=rows,
            code_sha256=c.sha(c.canonical(rows)),
            total_bytes=sum(r["size"] for r in rows),
        )
        path = root / ".sdsc" / ("wrapper-" + run + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(c.canonical(wrapper) + b"\n")
        return provenance.prepare(
            root,
            path,
            accepted_ancestors=[c.contract.PARENT["acceptance_commit"]],
            reviewed_local_implementation=implementation,
            reviewed_local_acceptance=acceptance,
        )

    old_execution = c.recovery.contract.resolve_execution_contract(root).as_dict()
    old_artifact = export(
        "old-science", c.contract.PARENT_EXECUTION["implementation_commit"], old_execution["head"]
    )
    old_inner = dict(
        provenance=dict(
            directory=old_artifact["artifact"],
            manifest_sha256=old_artifact["manifest_sha256"],
            head=old_execution["head"],
        ),
        code_sha256=old_artifact["wrapper_code_sha256"],
        protocol=old_execution["science_binding"],
    )
    previous = dict(science_plan=old_inner, execution=old_execution)
    paths = [*c.NEW_TOOLS, c.CONTRACT_PATH]
    for name in paths:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, root / name)
    payload = c.contract.proposed_execution_contract()
    (root / c.CONTRACT_PATH).write_bytes(c.canonical(payload) + b"\n")
    git("add", "--", *paths)
    git("commit", "--quiet", "-m", "Fixture proposed normal fit execution")
    implementation = git("rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="not accepted"):
        c.contract.resolve_execution_contract(root)
    payload["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=implementation,
        reviewer="Independent fixture",
        reviewed_at_utc="2026-10-06T12:00:00Z",
        rationale="Fixture-only separate fit execution with unchanged scientific source",
    )
    (root / c.CONTRACT_PATH).write_bytes(c.canonical(payload) + b"\n")
    git("add", "--", c.CONTRACT_PATH)
    git("commit", "--quiet", "-m", "Fixture independent review acceptance")
    acceptance = git("rev-parse", "HEAD").decode().strip()
    execution = c.contract.resolve_execution_contract(root, expected_head=acceptance).as_dict()
    artifact = export("new-execution", implementation, acceptance)
    manifest = c.decode((Path(artifact["artifact"]) / "manifest.json").read_bytes())
    assert 128 < len(manifest["local_successor"]["commits"]) <= 256
    value = dict(
        source=dict(
            provenance=dict(
                directory=artifact["artifact"], manifest_sha256=artifact["manifest_sha256"], head=acceptance
            ),
            code_sha256=artifact["wrapper_code_sha256"],
        ),
        execution=execution,
        preflight_recovery=previous,
    )
    try:
        result = c.verify_execution(value, tmp_path / "restored-fit")
        assert result["verified"] and result["git_head"] == acceptance
        assert old_inner["provenance"]["head"] != acceptance
        assert (
            execution["science_binding"]["science_file_sha256"]
            == old_execution["science_binding"]["science_file_sha256"]
        )
        dirty = root / c.CONTROLLER
        dirty.write_bytes(dirty.read_bytes() + b"\n# unreviewed delta\n")
        with pytest.raises(ValueError, match="reviewed implementation"):
            c.contract.resolve_execution_contract(root)
    finally:
        for folder, _dirs, files in os.walk(tmp_path):
            Path(folder).chmod(0o700)
            for name in files:
                path = Path(folder) / name
                if not path.is_symlink():
                    path.chmod(0o600)


def test_prepare_keeps_original_inner_and_creates_separate_reviewed_outer(c, plan, monkeypatch, tmp_path):
    from types import SimpleNamespace

    project = tmp_path / "local-project"
    project.mkdir()
    pins = {
        **c.contract.FROZEN_DEPENDENCIES,
        **plan["science_plan"]["protocol"]["science_file_sha256"],
        **plan["execution"]["control_file_sha256"],
        c.CONTRACT_PATH: c.sha((ROOT / c.CONTRACT_PATH).read_bytes()),
    }
    rows = []
    for name in sorted(pins):
        raw = (ROOT / name).read_bytes()
        p = project / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        rows.append(dict(path=name, size=len(raw), sha256=c.sha(raw), mode=0o644))
    manifest = dict(
        run_id="new-deployed-fit",
        code_sha256=c.sha(c.canonical(rows)),
        files=rows,
        total_bytes=sum(r["size"] for r in rows),
    )
    record = dict(state="deployed", manifest=manifest, deployment={"code_sha256": manifest["code_sha256"]})
    binding = copy.deepcopy(plan["execution"])
    binding["artifact_sha256"] = pins[c.CONTRACT_PATH]
    monkeypatch.setattr(c, "binding", lambda *a: binding)
    monkeypatch.setattr(c, "ROOT", project)
    prior_path = tmp_path / "prior.json"
    prior_path.write_bytes(c.canonical(plan["preflight_recovery"]))
    audit_path = tmp_path / "audit.json"
    audit_path.write_bytes(c.canonical(plan["science_plan"]["preflight"]["audit"]["document"]) + b"\n")
    args = SimpleNamespace(
        preflight_recovery_plan=prior_path,
        preflight_audit=audit_path,
        run_id=manifest["run_id"],
        execution_git_head=binding["head"],
        provenance_dir=plan["source"]["provenance"]["directory"],
        provenance_manifest_sha256=plan["source"]["provenance"]["manifest_sha256"],
    )
    assert not {
        name: (digest, c.sha((project / name).read_bytes()))
        for name, digest in pins.items()
        if digest != c.sha((project / name).read_bytes())
    }
    result = c.prepare(args, SimpleNamespace(run_record=lambda _: record))
    outer = c.load_plan(result["plan"], result["plan_sha256"])
    inner = outer["science_plan"]
    assert inner["run_id"] == plan["preflight_recovery"]["science_plan"]["run_id"]
    assert outer["source"]["run_id"] == manifest["run_id"]
    assert c.document(result["science_plan"]) == inner
    assert not Path(inner["scientific_claim"]).exists()
    with pytest.raises(ValueError, match="already prepared"):
        c.prepare(args, SimpleNamespace(run_record=lambda _: record))


@pytest.mark.parametrize(
    "case", ["failed", "missing_exit", "foreign_publication", "foreign_report", "saved_foreign"]
)
def test_preflight_completion_and_saved_accounting_fail_closed(c, plan, monkeypatch, case):
    calls = preflight_seam(c, plan, monkeypatch)
    good = c.recovery.inspect_job

    def inspect(*a, **kw):
        result = good(*a, **kw)
        if case == "failed":
            result["success"] = False
        elif case == "missing_exit":
            result["execution_exit_verified"] = False
        elif case == "foreign_publication":
            result["publication_sha256"] = "0" * 64
        elif case == "foreign_report":
            result["result_sha256"] = "0" * 64
        else:
            # Production inspector binds saved accounting to prior plan/job before acceptance.
            assert kw["accounting_snapshot"]["job_id"] == "wrong"
            raise ValueError("saved prerequisite accounting identity differs")
        return result

    monkeypatch.setattr(c.recovery, "inspect_job", inspect)
    if case == "saved_foreign":
        d = Path(plan["science_plan"]["submission_dir"])
        d.mkdir(parents=True)
        (d / "admission.json").write_bytes(c.canonical({"preflight": {"job_id": "wrong"}}))
    with pytest.raises(ValueError):
        c.verify_recovered_preflight(plan, accounting=case != "saved_foreign")
    assert calls


@pytest.mark.parametrize("case", ["trailing", "empty", "ambiguous"])
def test_query_reconciliation_is_utc_unique_and_never_resubmits(c, plan, monkeypatch, case):
    request, _ = submit_seam(
        c, plan, monkeypatch, observed(c, c.sbatch(plan), timed_out=True, output_complete=False)
    )
    with pytest.raises(ValueError):
        c.remote_action(request)
    inner = plan["science_plan"]
    user = c.pwd.getpwuid(os.getuid()).pw_name
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        assert argv[0] in {"squeue", "sacct"} and kw["utc_query"] is True
        comment = "" if case == "empty" else inner["intent_id"]
        raw = f"654321|{inner['job_name']}|{comment}|{user}|\n" if argv[0] == "sacct" else ""
        if case == "ambiguous" and argv[0] == "sacct":
            raw += f"654322|{inner['job_name']}|{comment}|{user}|\n"
        return observed(c, argv, stdout=raw.encode(), timeout=45, utc_query=True)

    monkeypatch.setattr(c.helper("sdsc_observed_command"), "run_observed", run)
    if case == "trailing":
        assert c.reconcile(plan)["job_id"] == "654321"
    else:
        with pytest.raises(ValueError, match="zero/ambiguous"):
            c.reconcile(plan)
    assert len(seen) == 2
