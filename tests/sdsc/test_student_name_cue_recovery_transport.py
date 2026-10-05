"""One infrastructure recovery keeps science claims and publishes honest runtime evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path=None):
    spec = importlib.util.spec_from_file_location(name, path or ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c(monkeypatch):
    original = subprocess.run

    def guarded(argv, *a, **kw):
        command = argv[0] if isinstance(argv, list | tuple) else argv.split()[0]
        if Path(command).name in {"ssh", "sbatch", "sacct", "squeue", "scontrol", "srun", "scancel"}:
            pytest.fail("CPU test attempted remote or Slurm operation")
        return original(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "run", guarded)
    return load("sdsc_student_name_cue_recovery")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    fixture = load(
        "_original_name_cue_transport_tests", ROOT / "tests/sdsc/test_student_name_cue_probe_transport.py"
    )
    old = fixture.plan.__wrapped__(c.science, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "CONTROL", c.science.CONTROL)
    monkeypatch.setattr(c, "PROJECT", c.science.PROJECT)
    monkeypatch.setattr(c, "RUNTIME_ARTIFACT_ROOT", c.PROJECT / "runtime-snapshots")
    failed = dict(c.contract.FAILED, intent_id=old["intent_id"], plan_sha256=c.sha(c.canonical(old)))
    monkeypatch.setattr(c.contract, "FAILED", failed)
    inner = copy.deepcopy(old)
    inner["provenance"].update(
        head="e" * 40, manifest_sha256="e" * 64, directory=str(c.CONTROL / "provenance-v2" / ("e" * 64))
    )
    inner["protocol"]["head"] = "e" * 40
    intent = c.science.execution_intent(inner)
    inner.update(
        intent_id=intent,
        submission_dir=str(c.CONTROL / "student-name-cue-probe-submissions" / intent),
        claim=str(c.CONTROL / "student-name-cue-probe-claims" / (intent + ".json")),
        result_dir=str(c.PROJECT / "student-name-cue-probe" / intent),
        job_name="opd-snc-" + intent,
    )
    snapshot = c.helper("sdsc_runtime_snapshot")
    artifact = c.RUNTIME_ARTIFACT_ROOT / ("snapshot-" + "a" * 32)
    descriptor = dict(
        manifest=dict(path=str(artifact / "manifest.json"), size=1024, sha256="a" * 64),
        archive=dict(path=str(artifact / "runtime.bin"), size=len(snapshot.MAGIC) + 1, sha256="b" * 64),
        runtime_root_sha256="c" * 64,
        source_prefix=c.contract.RUNTIME_PREFIX,
        python_relative_path="bin/python3.12",
        python_sha256=c.contract.PYTHON_SHA256,
        files_count=1,
        total_bytes=1,
    )
    pins = {n: c.sha((ROOT / n).read_bytes()) if (ROOT / n).exists() else "f" * 64 for n in c.NEW_TOOLS}
    execution = dict(
        core_sha256="a" * 64,
        artifact_sha256="b" * 64,
        implementation_commit="c" * 40,
        acceptance_commit="d" * 40,
        head=inner["provenance"]["head"],
        control_file_sha256=pins,
        science_binding=inner["protocol"],
    )
    value = dict(
        schema=c.SCHEMA,
        task=c.TASK,
        science_plan=inner,
        execution=execution,
        failed=dict(
            plan=old,
            receipt=c.science.make_receipt(old, failed["job_id"]),
            **{k: v for k, v in failed.items() if k not in ("job_id", "intent_id", "plan_sha256")},
        ),
        runtime_snapshot=descriptor,
        intent_id=intent,
        **dict.fromkeys(c.FLAGS, False),
    )
    value["recovery_claim"] = c.recovery_claim(value)
    return c.validate_plan(value)


def test_fixed_execution_contract_and_all_frozen_dependencies(c):
    value = c.contract.load_execution_contract((ROOT / c.CONTRACT_PATH).read_bytes())
    assert value["scope"]["scientific_change"] is False
    assert value["execution"]["worker_deadline_seconds"] == 4800
    assert value["execution"]["publication_reserve_seconds"] == 600
    assert value["execution"]["early_probe_whole_child_seconds"] == 300
    assert len(c.contract.FROZEN_DEPENDENCIES) == 33
    assert all(c.sha((ROOT / p).read_bytes()) == h for p, h in c.contract.FROZEN_DEPENDENCIES.items())
    assert set(c.EXECUTION_NAMES) == set(c.helper("sdsc_student_name_cue_recovery_audit").NAMES)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("scope", "scientific_change", True),
        ("scope", "automatic_retry", True),
        ("execution", "walltime_seconds", 7200),
        ("execution", "early_probe_whole_child_seconds", 600),
        ("execution", "optimizer_steps", 1),
        ("execution", "maximum_runtime_manifest_mib", 64),
        ("runtime", "python_sha256", "0" * 64),
        ("claims", "maximum_recovery_submissions", 2),
    ],
)
def test_contract_changes_fail_closed(c, section, key, value):
    proposal = c.contract.proposed_execution_contract()
    proposal[section][key] = value
    with pytest.raises(ValueError):
        c.contract.validate_execution_contract(proposal)


def test_review_required_and_duplicate_json_rejected(c):
    raw = c.contract.canonical(c.contract.proposed_execution_contract())
    with pytest.raises(ValueError):
        c.contract.load_execution_contract(raw, require_accepted=True)
    with pytest.raises(ValueError):
        c.contract.load_execution_contract(b'{"schema":"x",' + raw[1:])


def test_fresh_execution_keeps_exact_science_and_original_claim(c, plan):
    assert plan["science_plan"]["intent_id"] != plan["failed"]["plan"]["intent_id"]
    assert plan["science_plan"]["scientific_claim"] == plan["failed"]["plan"]["scientific_claim"]
    assert plan["science_plan"]["parents"] == plan["failed"]["plan"]["parents"]
    assert all(plan[k] is False for k in c.FLAGS)
    before = plan["recovery_claim"]
    plan["runtime_snapshot"]["archive"]["sha256"] = "f" * 64
    plan["science_plan"]["run_id"] = "another-release"
    plan["execution"]["head"] = "0" * 40
    assert c.recovery_claim(plan) == before


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(student_accepted=True),
        lambda p: p["science_plan"]["resources"].update(gpus=4),
        lambda p: p["failed"]["receipt"].update(job_id="54681803"),
        lambda p: p["failed"].update(trace_sha256="0" * 64),
        lambda p: p["science_plan"]["parents"]["control"]["checkpoint"].update(sha256="0" * 64),
        lambda p: p["execution"]["control_file_sha256"].pop("tools/sdsc_runtime_snapshot.py"),
        lambda p: p["execution"].update(head="0" * 40),
        lambda p: p.update(recovery_claim=p["recovery_claim"] + ".new"),
        lambda p: p["runtime_snapshot"].update(source_prefix="/unreviewed/runtime"),
        lambda p: p["runtime_snapshot"]["manifest"].update(size=16 * 1024**2 + 1),
        lambda p: p["runtime_snapshot"]["manifest"].update(
            path=p["runtime_snapshot"]["manifest"]["path"].replace("/snapshot-", "/unreviewed-")
        ),
        lambda p: p["runtime_snapshot"].update(python_sha256="0" * 64),
    ],
)
def test_plan_mutations_rejected(c, plan, change):
    change(plan)
    with pytest.raises((ValueError, KeyError)):
        c.validate_plan(plan)


def test_frozen_dependency_verified_before_python_exec(c, tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    marker = tmp_path / "executed"
    (tools / "sdsc_student_name_cue_probe.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    with pytest.raises(ValueError, match="frozen recovery dependency changed"):
        c.helper("sdsc_student_name_cue_probe", tmp_path)
    assert not marker.exists()


def terminal(c):
    job = c.contract.FAILED["job_id"]
    return dict(
        job_id=job,
        state="FAILED",
        accounting_complete=False,
        queue=dict(returncode=0, stdout=""),
        accounting=dict(
            returncode=0, stdout=f"{job}|FAILED|1:0|\n{job}.batch|FAILED|1:0|\n{job}.extern|COMPLETED|0:0|\n"
        ),
        publication_verified=True,
        publication_sha256=c.contract.FAILED["publication_sha256"],
        success=False,
    )


def test_failed_terminal_does_not_require_success_flag(c):
    assert c.validate_failed_terminal(terminal(c))["accounting_complete"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["queue"].update(stdout="54681802|RUNNING|owner"),
        lambda s: s["accounting"].update(
            stdout=s["accounting"]["stdout"].replace(".batch|FAILED|1:0", ".batch|COMPLETED|0:0")
        ),
        lambda s: s["accounting"].update(stdout=s["accounting"]["stdout"].splitlines()[0]),
        lambda s: s.update(publication_sha256="0" * 64),
        lambda s: s.update(success=True),
    ],
)
def test_failed_terminal_incomplete_or_wrong_rejected(c, mutation):
    state = terminal(c)
    mutation(state)
    with pytest.raises(ValueError):
        c.validate_failed_terminal(state)


def test_failure_raw_byte_identity_rejected_before_inference_decision(c, plan):
    files = {
        n: b"{}"
        for n in ("receipt.json", "node-result.json", "trace-meta.json", "memory.json", "startup.log")
    }
    with pytest.raises(ValueError, match="failed raw artifact differs"):
        c.validate_failed_documents(plan, files)


def setup_submit(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    original = Path(plan["failed"]["plan"]["scientific_claim"])
    original.parent.mkdir(parents=True)
    raw = c.canonical(c.science.scientific_claim_record(plan["failed"]["plan"]))
    original.write_bytes(raw)
    monkeypatch.setattr(c, "admission", lambda p: {})
    proof = dict(dry_run=True, blockers=[], plan_sha256=c.sha(c.canonical(plan)), argv=c.sbatch(plan))
    return original, raw, proof


def test_unknown_ack_claims_once_never_overwrites_consumed_science(c, plan, monkeypatch):
    original, raw, proof = setup_submit(c, plan, monkeypatch)
    calls = []
    monkeypatch.setattr(
        c, "run", lambda argv: calls.append(argv) or dict(returncode=1, stdout="", stderr="lost")
    )
    request = dict(plan=plan, action="submit", authorize=True, dry_run=proof)
    with pytest.raises(ValueError, match="acknowledgement"):
        c.remote_action(request)
    assert len(calls) == 1 and original.read_bytes() == raw
    assert Path(plan["recovery_claim"]).exists() and Path(plan["science_plan"]["claim"]).exists()
    assert (Path(plan["science_plan"]["submission_dir"]) / "unknown.json").exists()
    with pytest.raises(FileExistsError):
        c.remote_action(request)
    assert len(calls) == 1 and original.read_bytes() == raw


def test_submit_writes_both_receipts_but_keeps_original_scientific_claim(c, plan, monkeypatch):
    original, raw, proof = setup_submit(c, plan, monkeypatch)
    monkeypatch.setattr(c, "run", lambda argv: dict(returncode=0, stdout="123456\n", stderr=""))
    monkeypatch.setattr(c.science, "live_binding", lambda inner, job: dict(job_id=job))
    receipt = c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=proof))
    assert receipt["job_id"] == "123456" and original.read_bytes() == raw
    c.check_claims(plan)
    directory = Path(plan["science_plan"]["submission_dir"])
    assert c.document(directory / "execution-receipt.json")["scientific_receipt"] == receipt
    assert c.document(directory / "plan.json") == plan["science_plan"]
    assert c.document(directory / "execution-plan.json") == plan


def test_no_matching_preview_cannot_claim(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    monkeypatch.setattr(c, "admission", lambda p: pytest.fail("admission before matching preview"))
    with pytest.raises(ValueError, match="matching dry-run"):
        c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=None))
    assert not Path(plan["recovery_claim"]).exists()


def test_job_launch_binds_named_node_and_preserves_visibility(c, plan, tmp_path):
    script = c.job_script(plan)
    path = tmp_path / "job.sh"
    path.write_bytes(script)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    assert (
        plan["execution"]["control_file_sha256"]["tools/sdsc_student_name_cue_recovery_job.py"].encode()
        in script
    )
    assert b"CUDA_VISIBLE_DEVICES=" not in script and b"conda-unpack" not in script
    assert c.sbatch(plan)[-1] == c.sha(c.canonical(plan))
    assert c.worker_budget_seconds(plan) == 4800


def test_execution_restore_requires_exact_accepted_outer_binding(c, plan, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        c.science, "verify_science", lambda inner, dest: calls.append((inner, dest)) or {"verified": True}
    )
    monkeypatch.setattr(c, "binding", lambda root, head: plan["execution"])
    assert c.verify_execution(plan, tmp_path / "science") == {"verified": True}
    assert calls[0][0] == plan["science_plan"]
    monkeypatch.setattr(c, "binding", lambda root, head: dict(plan["execution"], head="0" * 40))
    with pytest.raises(ValueError, match="restored accepted execution"):
        c.verify_execution(plan, tmp_path / "other")


def test_actual_reviewed_recovery_restores_full_v2_history_and_rejects_dirty_control(c, tmp_path):
    """Real accepted scientific anchors survive additive execution review and native restore."""
    import os
    import shutil

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
    git("checkout", "--quiet", c.contract.PARENT["acceptance_commit"])
    git("config", "user.name", "Recovery fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("update-ref", "refs/remotes/public/master", "0215c356355b29b5e2b407978a207db2156719e1")
    paths = [*c.NEW_TOOLS, c.CONTRACT_PATH]
    for name in paths:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, root / name)
    payload = c.contract.proposed_execution_contract()
    (root / c.CONTRACT_PATH).write_bytes(c.canonical(payload) + b"\n")
    git("add", "--", *paths)
    git("commit", "--quiet", "-m", "Fixture proposed infrastructure recovery")
    implementation = git("rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="not accepted"):
        c.contract.resolve_execution_contract(root)
    payload["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=implementation,
        reviewer="independent fixture",
        reviewed_at_utc="2026-10-05T21:00:00Z",
        rationale="Fixture-only execution acceptance; original science unchanged",
    )
    (root / c.CONTRACT_PATH).write_bytes(c.canonical(payload) + b"\n")
    git("add", "--", c.CONTRACT_PATH)
    git("commit", "--quiet", "-m", "Fixture review-only execution acceptance")
    acceptance = git("rev-parse", "HEAD").decode().strip()
    execution = c.contract.resolve_execution_contract(root, expected_head=acceptance).as_dict()
    assert execution["science_binding"]["acceptance_commit"] == c.contract.PARENT["acceptance_commit"]
    assert len(execution["science_binding"]["science_file_sha256"]) == 168
    names = git("ls-files", "-z").decode().split("\0")[:-1]
    records = [
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
        run_id="fixture-recovery",
        git_head=acceptance,
        files=records,
        code_sha256=c.sha(c.canonical(records)),
        total_bytes=sum(r["size"] for r in records),
    )
    wrapper_path = root / ".sdsc/wrapper.json"
    wrapper_path.parent.mkdir(exist_ok=True)
    wrapper_path.write_bytes(c.canonical(wrapper) + b"\n")
    artifact = provenance.prepare(
        root,
        wrapper_path,
        accepted_ancestors=[c.contract.PARENT["acceptance_commit"]],
        reviewed_local_implementation=implementation,
        reviewed_local_acceptance=acceptance,
    )
    manifest = json.loads((Path(artifact["artifact"]) / "manifest.json").read_bytes())
    assert len(manifest["local_successor"]["commits"]) > 128
    inner = dict(
        provenance=dict(
            directory=artifact["artifact"], manifest_sha256=artifact["manifest_sha256"], head=acceptance
        ),
        code_sha256=artifact["wrapper_code_sha256"],
        protocol=execution["science_binding"],
    )
    try:
        restored = c.verify_execution(dict(science_plan=inner, execution=execution), tmp_path / "restored")
        assert restored["verified"] is True and restored["git_head"] == acceptance
        dirty = root / "tools/sdsc_student_name_cue_recovery.py"
        dirty.write_bytes(dirty.read_bytes() + b"\n# unreviewed execution change\n")
        with pytest.raises(ValueError, match="reviewed implementation"):
            c.contract.resolve_execution_contract(root, expected_head=acceptance)
    finally:
        for folder, _dirs, files in os.walk(tmp_path):
            Path(folder).chmod(0o700)
            for name in files:
                path = Path(folder) / name
                if not path.is_symlink():
                    path.chmod(0o600)


def published_failure(c, plan, tmp_path):
    import base64

    inner = plan["science_plan"]
    output = Path(inner["result_dir"])
    output.mkdir(parents=True)
    original = c.helper("sdsc_student_name_cue_probe_job")
    original.persist(
        inner,
        c.science,
        None,
        None,
        dict(job_id="123456", stage_complete=False, exit_code=1),
        {},
        lambda label: None,
    )
    execution = tmp_path / "execution-work"
    execution.mkdir()
    node = c.helper("sdsc_student_name_cue_recovery_job")
    node.persist_execution(
        plan,
        c,
        execution,
        dict(
            schema="quest-sdsc-student-name-cue-recovery-node-v1",
            job_id="123456",
            plan_sha256=c.sha(c.canonical(plan)),
            science_plan_sha256=c.sha(c.canonical(inner)),
            stage_complete=False,
            exit_code=1,
            **dict.fromkeys(c.FLAGS, False),
        ),
        {},
    )
    state = c.execution_publication(plan, "123456")
    assert state["execution_publication_verified"] is True and state["execution_complete"] is False
    publication = c.document(output / "receipt.json")
    state.update(
        publication_verified=True,
        publication=publication,
        publication_sha256=c.sha((output / "receipt.json").read_bytes()),
    )
    names = [
        "receipt.json",
        *[r["path"] for r in publication["files"]],
        "execution/receipt.json",
        *["execution/" + r["path"] for r in state["execution_publication"]["files"]],
    ]
    raw = {n: (output / n).read_bytes() for n in names}
    result = dict(
        status=state,
        receipt=c.science.make_receipt(inner, "123456"),
        files={n: base64.b64encode(v).decode() for n, v in raw.items()},
        bytes=sum(map(len, raw.values())),
    )
    return result, raw


def test_actual_original_and_outer_failed_publications_roundtrip(c, plan, tmp_path):
    result, raw = published_failure(c, plan, tmp_path)
    assert c.validate_download(result, plan) == raw
    assert result["status"]["execution_publication"]["large_files"] == []
    assert result["status"]["publication"]["selected_checkpoint"] is None


@pytest.mark.parametrize("mutation", ["job", "science_link", "raw_file", "extra_file"])
def test_outer_fetch_rejects_cross_publication_and_byte_substitution(c, plan, tmp_path, mutation):
    import base64

    result, raw = published_failure(c, plan, tmp_path)
    if mutation in ("job", "science_link"):
        pub = result["status"]["execution_publication"]
        pub["job_id" if mutation == "job" else "scientific_publication_sha256"] = (
            "123457" if mutation == "job" else "0" * 64
        )
        data = c.canonical(pub)
        result["status"]["execution_publication_sha256"] = c.sha(data)
        raw["execution/receipt.json"] = data
    elif mutation == "raw_file":
        raw["execution/memory.json"] = b'{"changed":true}'
    else:
        raw["execution/unpublished.json"] = b"{}"
    result["files"] = {n: base64.b64encode(v).decode() for n, v in raw.items()}
    result["bytes"] = sum(map(len, raw.values()))
    with pytest.raises(ValueError):
        c.validate_download(result, plan)


def test_actual_git_queries_ignore_replace_objects(c, tmp_path):
    repo = tmp_path / "git-replacement"
    repo.mkdir()
    env = c.contract._git_environment()

    def git(*args, environment=env):
        return subprocess.run(
            ["git", "-C", str(repo), *args], env=environment, check=True, capture_output=True
        ).stdout

    git("init", "--quiet")
    git("config", "user.name", "Replacement fixture")
    git("config", "user.email", "fixture@example.invalid")
    control = repo / "control.py"
    control.write_text("substituted bytes\n")
    git("add", "control.py")
    git("commit", "--quiet", "-m", "Substitute")
    substitute = git("rev-parse", "HEAD").decode().strip()
    control.write_text("genuine accepted bytes\n")
    git("add", "control.py")
    git("commit", "--quiet", "-m", "Genuine")
    genuine = git("rev-parse", "HEAD").decode().strip()
    git("replace", genuine, substitute)
    vulnerable = dict(env)
    vulnerable.pop("GIT_NO_REPLACE_OBJECTS")
    assert git("show", genuine + ":control.py", environment=vulnerable) == b"substituted bytes\n"
    assert git("show", genuine + ":control.py") == b"genuine accepted bytes\n"


@pytest.mark.parametrize("corrupt_origin", [False, True])
def test_complete_real_origin_and_startup_publication_through_controller(
    c, plan, tmp_path, monkeypatch, corrupt_origin
):
    """Use actual snapshot/capture/startup writers, mocking hardware observations only."""
    fixture = load("_recovery_job_fixture", ROOT / "tests/sdsc/test_student_name_cue_recovery_job.py")
    modules = [
        c.helper(name)
        for name in (
            "sdsc_student_name_cue_recovery_job",
            "sdsc_student_name_cue_recovery_audit",
            "sdsc_runtime_snapshot",
        )
    ] + [c]
    bundle = fixture.bundle.__wrapped__(modules, tmp_path, monkeypatch)
    node, audit = modules[:2]
    plan["runtime_snapshot"] = bundle["plan"]["runtime_snapshot"]
    bundle["plan"] = plan
    bundle["context"] = node.context_value(
        plan, bundle["stage"], bundle["work"], dict(job_id="12345", cuda_visible_devices="2,3"), c
    )
    files = fixture.complete_execution_files(bundle, monkeypatch)
    output = Path(plan["science_plan"]["result_dir"])
    output.mkdir(parents=True)
    c.helper("sdsc_student_name_cue_probe_job").persist(
        plan["science_plan"],
        c.science,
        None,
        None,
        dict(job_id="12345", stage_complete=False, exit_code=1),
        {},
        lambda label: None,
    )
    work = bundle["work"] / "to-publish"
    work.mkdir()
    for name, raw in files.items():
        if name not in ("memory.json", "recovery-node-result.json"):
            (work / name).write_bytes(raw)
    publication = node.persist_execution(
        plan, c, work, json.loads(files["recovery-node-result.json"]), json.loads(files["memory.json"])
    )
    assert publication["execution_complete"] is True
    if corrupt_origin:
        path = output / "execution/runtime-rank-1-after.json"
        value = json.loads(path.read_bytes())
        value["executable"] = plan["runtime_snapshot"]["source_prefix"] + "/bin/python3.12"
        raw = c.canonical(value)
        path.write_bytes(raw)
        for row in publication["files"]:
            if row["path"] == path.name:
                row.update(size=len(raw), sha256=c.sha(raw))
        (output / "execution/receipt.json").write_bytes(c.canonical(publication))
        with pytest.raises(ValueError, match="relocated native prefix"):
            c.execution_publication(plan, "12345")
    else:
        result = c.execution_publication(plan, "12345")
        assert result["execution_complete"] is True
        assert all(result["execution_publication"][key] is False for key in c.FLAGS)
        assert len(audit.validate_execution_evidence(plan, files)["native_origins"]) == 5


def test_independent_outer_publication_binds_actual_node_job(c, plan, tmp_path):
    result, raw = published_failure(c, plan, tmp_path)
    publication = copy.deepcopy(result["status"]["execution_publication"])
    publication["job_id"] = "123457"
    evidence = {
        name.removeprefix("execution/"): value
        for name, value in raw.items()
        if name.startswith("execution/") and name != "execution/receipt.json"
    }
    with pytest.raises(ValueError, match="identity"):
        c.helper("sdsc_student_name_cue_recovery_audit").validate_execution_publication(
            plan, publication, evidence
        )
