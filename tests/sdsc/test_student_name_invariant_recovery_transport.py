"""Actual local fence, bounded command and unchanged-science recovery seams; no SDSC calls."""

import base64
import copy
import importlib.util
import os
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
    original = subprocess.Popen

    def guarded(argv, *args, **kwargs):
        name = Path(argv[0]).name if isinstance(argv, tuple | list) else argv.split()[0]
        if name in {"ssh", "sbatch", "squeue", "sacct", "scontrol", "scancel", "srun"}:
            pytest.fail("fixture attempted actual Slurm/remote command")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded)
    return load("_new_recovery", ROOT / "tools/sdsc_student_name_invariant_recovery.py")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    fixture = load(
        "_original_transport_fixture", ROOT / "tests/sdsc/test_student_name_invariant_transport.py"
    )
    old = fixture.plan.__wrapped__(c.science, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "CONTROL", c.science.CONTROL)
    monkeypatch.setattr(c, "PROJECT", c.science.PROJECT)
    for folder in (c.CONTROL, c.PROJECT):
        folder.mkdir(mode=0o700)
    fence = c.helper("sdsc_student_execution_fence")
    observed = c.helper("sdsc_observed_command")
    helper = c.helper
    monkeypatch.setattr(
        c,
        "helper",
        lambda name, root=ROOT: fence
        if name == "sdsc_student_execution_fence"
        else observed
        if name == "sdsc_observed_command"
        else helper(name, root),
    )
    monkeypatch.setattr(fence, "CONTROL", c.CONTROL)
    monkeypatch.setattr(fence, "PROJECT", c.PROJECT)
    monkeypatch.setattr(fence, "EXPECTED_UID", os.getuid())
    filesystems = {}
    for label, folder in (("control", c.CONTROL), ("project", c.PROJECT)):
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            filesystems[label] = fence._observe_filesystem(descriptor, folder)["portable"]
        finally:
            os.close(descriptor)
    monkeypatch.setattr(fence, "FILESYSTEMS", filesystems)
    monkeypatch.setattr(fence, "OLD_INTENT", old["intent_id"])
    monkeypatch.setattr(fence, "OLD_PLAN_SHA", c.sha(c.canonical(old)))
    monkeypatch.setattr(
        c.contract,
        "OLD",
        dict(
            intent_id=old["intent_id"],
            plan_sha256=c.sha(c.canonical(old)),
            node_sha256=old["control_sha256"]["tools/sdsc_student_name_invariant_job.py"],
        ),
    )
    monkeypatch.setattr(c.contract, "FROZEN_DEPENDENCIES", dict(old["control_sha256"]))
    side = {k: ("old fixture " + k).encode() for k in fence.SIDEFILES}
    claims = {
        "claim": c.canonical(dict(intent_id=old["intent_id"], plan_sha256=c.sha(c.canonical(old)), at="old")),
        "scientific_claim": c.canonical(c.science.scientific_claim_record(old)),
    }
    monkeypatch.setattr(fence, "SIDEFILES", {k: (len(v), c.sha(v)) for k, v in side.items()})
    monkeypatch.setattr(fence, "OLD_CLAIMS", {k: (len(v), c.sha(v)) for k, v in claims.items()})
    files = {Path(old["submission_dir"]) / "plan.json": c.canonical(old)}
    files.update({Path(old["submission_dir"]) / k: v for k, v in side.items()})
    files.update({Path(old[k]): v for k, v in claims.items()})
    files.update(
        {Path(old["release"]) / "source" / k: (ROOT / k).read_bytes() for k in old["control_sha256"]}
    )
    previous = os.umask(0o077)
    try:
        for path, raw in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            path.chmod(0o600)
        request = fence.build_request(old)
        proof = fence.install(request, authorize=True, dry_run=fence.dry_run(request))
    finally:
        os.umask(previous)
    inner = copy.deepcopy(old)
    inner["run_id"] = "fresh-recovery-release"
    inner["code_sha256"] = "a" * 64
    inner["manifest_sha256"] = "b" * 64
    inner["created_at"] = "2026-10-06T00:00:00Z"
    inner["protocol"]["head"] = "4" * 40
    inner["provenance"] = dict(
        directory=str(c.CONTROL / "provenance-v2" / ("f" * 64)), manifest_sha256="f" * 64, head="4" * 40
    )
    fixture.bind(c.science, inner)
    execution = dict(
        core_sha256="6" * 64,
        artifact_sha256="7" * 64,
        implementation_commit="3" * 40,
        acceptance_commit="4" * 40,
        head="4" * 40,
        science_binding=inner["protocol"],
        control_file_sha256={name: c.sha((ROOT / name).read_bytes()) for name in c.NEW_TOOLS},
    )
    value = dict(
        schema=c.SCHEMA,
        task=c.TASK,
        intent_id=inner["intent_id"],
        science_plan=inner,
        science_plan_sha256=c.sha(c.canonical(inner)),
        execution=execution,
        old_plan=old,
        fence=dict(
            path=request["proof_path"],
            document=proof,
            size=len(fence.canonical(proof)),
            sha256=c.sha(fence.canonical(proof)),
        ),
        **dict.fromkeys(c.FLAGS, False),
    )
    value["recovery_claim"] = c.recovery_claim(value)
    c.validate_plan(value)
    return value


def observe(c, argv, *, stdout=b"123456\n", stderr=b"", timeout=180, utc_query=False, **changes):
    def stream(raw):
        return dict(
            base64=base64.b64encode(raw).decode(),
            retained_bytes=len(raw),
            retained_sha256=c.sha(raw),
            observed_bytes=len(raw),
            observed_sha256=c.sha(raw),
            eof=True,
            truncated=False,
        )

    value = dict(
        schema="opd-observed-command-v1",
        argv=argv,
        timeout_seconds=timeout,
        max_output_bytes=c.CAP,
        stdin="DEVNULL",
        removed_environment_prefixes=["SBATCH_", "SQUEUE_", "SACCT_"],
        utc_query=utc_query,
        spawned=True,
        pid=42,
        returncode=0,
        timed_out=False,
        output_limited=False,
        termination_requested=False,
        child_reaped=True,
        error=None,
        retry_attempted=False,
        output_complete=True,
        started_at="2026-10-06T01:00:00Z",
        ended_at="2026-10-06T01:00:01Z",
        elapsed_seconds=1,
        stdout=stream(stdout),
        stderr=stream(stderr),
    )
    value.update(changes)
    return value


def submit_seam(c, plan, monkeypatch, result=None):
    called = []
    monkeypatch.setattr(c, "verify_source", lambda *a, **k: None)

    # Keep actual no-retry existence and fence checks; mock external admission observations only.
    def admitted(value):
        c.verify_fence(value)
        inner = value["science_plan"]
        assert not Path(value["recovery_claim"]).exists()
        assert not Path(inner["claim"]).exists()
        return dict(accepted_fixture=True)

    monkeypatch.setattr(c, "admission", admitted)
    monkeypatch.setattr(c.science, "live_binding", lambda inner, job: dict(job_id=job, fixture=True))

    def run(argv, **kwargs):
        called.append((argv, kwargs))
        # Claims and exact raw plans exist before sbatch, but no receipt is needed by node.
        c.check_claims(plan)
        assert not (Path(plan["science_plan"]["submission_dir"]) / "receipt.json").exists()
        return result if result is not None else observe(c, argv)

    monkeypatch.setattr(c.helper("sdsc_observed_command"), "run_observed", run)
    request = dict(
        action="submit",
        plan=plan,
        authorize=True,
        dry_run=dict(
            dry_run=True,
            blockers=[],
            plan_sha256=c.sha(c.canonical(plan)),
            science_plan_sha256=plan["science_plan_sha256"],
            argv=c.sbatch(plan),
        ),
    )
    return request, called


def test_actual_fence_and_unchanged_science_plan(c, plan):
    assert c.verify_fence(plan)["old_submission_outcome"] == "unknown"
    old, inner = plan["old_plan"], plan["science_plan"]
    assert inner["scientific_claim"] == old["scientific_claim"]
    assert inner["runtime_snapshot"] == old["runtime_snapshot"]
    assert inner["resources"] == c.science.resources("preflight")
    args = c.sbatch(plan)
    assert args[-1] == c.sha(c.canonical(plan)) and "--time=01:00:00" in args
    assert "--export=NONE" in args and "--no-requeue" in args
    script = c.job_script(plan).decode()
    assert c.NODE in script and "execution-plan.json" in script
    assert plan["execution"]["control_file_sha256"][c.NODE] in script


@pytest.mark.parametrize(
    "mutation",
    [
        "oldhash",
        "oldintent",
        "science",
        "runtime",
        "resources",
        "controls",
        "acceptance",
        "fencehash",
        "fencepath",
        "claim",
        "flag",
        "extra",
        "innerhash",
        "fit",
        "protocol",
    ],
)
def test_plan_rejects_mutation(c, plan, mutation):
    if mutation == "oldhash":
        plan["old_plan"]["created_at"] = "changed"
    elif mutation == "oldintent":
        plan["old_plan"]["intent_id"] = "0" * 32
    elif mutation == "science":
        plan["science_plan"]["dataset_inputs"][0]["sha256"] = "0" * 64
    elif mutation == "runtime":
        plan["science_plan"]["runtime_snapshot"]["archive"]["sha256"] = "0" * 64
    elif mutation == "resources":
        plan["science_plan"]["resources"]["time"] = "02:00:00"
    elif mutation == "controls":
        plan["science_plan"]["control_sha256"][c.science.PROTOCOL_PATH] = "0" * 64
    elif mutation == "acceptance":
        plan["execution"]["acceptance_commit"] = plan["execution"]["implementation_commit"]
    elif mutation == "fencehash":
        plan["fence"]["sha256"] = "0" * 64
    elif mutation == "fencepath":
        plan["fence"]["path"] += ".other"
    elif mutation == "claim":
        plan["recovery_claim"] += ".other"
    elif mutation == "flag":
        plan[c.FLAGS[0]] = True
    elif mutation == "extra":
        plan["retry"] = True
    elif mutation == "innerhash":
        plan["science_plan_sha256"] = "0" * 64
    elif mutation == "fit":
        plan["science_plan"]["mode"] = "fit"
    elif mutation == "protocol":
        plan["science_plan"]["protocol"]["science_file_sha256"]["extra.py"] = "0" * 64
    with pytest.raises((ValueError, KeyError, TypeError)):
        c.validate_plan(plan)


def test_claim_key_independent_new_release_and_uuid(c, plan):
    other = copy.deepcopy(plan)
    other["intent_id"] = "a" * 32
    other["science_plan"]["intent_id"] = "a" * 32
    other["science_plan"]["code_sha256"] = "b" * 64
    other["science_plan"]["run_id"] = "different"
    assert c.recovery_claim(other) == c.recovery_claim(plan)


def test_real_submit_seam_preserves_old_claims_and_observation(c, plan, monkeypatch):
    old = {key: Path(plan["old_plan"][key]).read_bytes() for key in ("claim", "scientific_claim")}
    request, calls = submit_seam(c, plan, monkeypatch)
    receipt = c.remote_action(request)
    assert receipt["job_id"] == "123456" and len(calls) == 1
    assert calls[0][1] == dict(timeout=180, max_output_bytes=c.CAP)
    directory = Path(plan["science_plan"]["submission_dir"])
    assert (
        c.document(directory / "submission-observation.json")["stdout"]["base64"]
        == base64.b64encode(b"123456\n").decode()
    )
    assert not (directory / "unknown.json").exists()
    for key, raw in old.items():
        assert Path(plan["old_plan"][key]).read_bytes() == raw
    c.check_claims(plan)
    with pytest.raises((ValueError, AssertionError, FileExistsError)):
        c.remote_action(request)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        dict(timed_out=True, output_complete=False),
        dict(output_limited=True, output_complete=False),
        dict(returncode=1),
        dict(child_reaped=False),
        dict(error="transport"),
        dict(output_complete=False),
        dict(stdout=None),
    ],
)
def test_unknown_partial_job_ack_preserved_without_receipt_or_retry(c, plan, monkeypatch, changes):
    value = observe(c, c.sbatch(plan), stdout=b"123456\n", stderr=b"controller delayed\n")
    value.update(changes)
    request, calls = submit_seam(c, plan, monkeypatch, value)
    with pytest.raises((ValueError, KeyError, TypeError)):
        c.remote_action(request)
    directory = Path(plan["science_plan"]["submission_dir"])
    assert c.document(directory / "submission-observation.json") == value
    assert c.document(directory / "unknown.json")["no_retry"] is True
    assert not (directory / "receipt.json").exists()
    assert Path(plan["recovery_claim"]).exists() and len(calls) == 1
    with pytest.raises((ValueError, AssertionError, FileExistsError)):
        c.remote_action(request)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation", ["base64", "sha", "observedsha", "retainedbytes", "eof", "truncated", "argv", "utc", "retry"]
)
def test_observation_rejects_corruption(c, plan, mutation):
    argv = c.sbatch(plan)
    value = observe(c, argv)
    if mutation == "base64":
        value["stdout"]["base64"] = "!"
    elif mutation == "sha":
        value["stdout"]["retained_sha256"] = "0" * 64
    elif mutation == "observedsha":
        value["stdout"]["observed_sha256"] = "0" * 64
    elif mutation == "retainedbytes":
        value["stdout"]["retained_bytes"] += 1
    elif mutation == "eof":
        value["stdout"]["eof"] = False
    elif mutation == "truncated":
        value["stdout"]["truncated"] = True
    elif mutation == "argv":
        value["argv"] = ["sbatch", "foreign"]
    elif mutation == "utc":
        value["utc_query"] = True
    elif mutation == "retry":
        value["retry_attempted"] = True
    with pytest.raises((ValueError, TypeError)):
        c.completed_command(value, argv, timeout=180)


def test_live_binding_failure_retains_numeric_receipt_and_reconcile_never_submits(c, plan, monkeypatch):
    request, calls = submit_seam(c, plan, monkeypatch)
    monkeypatch.setattr(
        c.science, "live_binding", lambda *a: (_ for _ in ()).throw(ValueError("not yet visible"))
    )
    with pytest.raises(ValueError):
        c.remote_action(request)
    directory = Path(plan["science_plan"]["submission_dir"])
    assert c.document(directory / "receipt.json")["job_id"] == "123456"
    assert c.reconcile(plan)["job_id"] == "123456"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "case", ["unique", "empty", "foreigncomment", "duplicate", "querytimeout", "willrun"]
)
def test_reconcile_utc_is_readonly_and_requires_unique_exact_intent(c, plan, monkeypatch, case):
    request, _ = submit_seam(
        c, plan, monkeypatch, observe(c, c.sbatch(plan), timed_out=True, output_complete=False)
    )
    with pytest.raises(ValueError):
        c.remote_action(request)
    calls = []
    inner = plan["science_plan"]

    def query(argv, **kwargs):
        calls.append(argv)
        assert argv[0] in ("squeue", "sacct") and kwargs["utc_query"] is True
        job = "54690409" if case == "willrun" else "123456"
        comment = "foreign" if case == "foreigncomment" else inner["intent_id"]
        raw = (job + "|" + inner["job_name"] + "|" + comment + "\n").encode() if argv[0] == "squeue" else b""
        if case == "empty":
            raw = b""
        if case == "duplicate" and argv[0] == "squeue":
            raw += raw.replace(b"123456", b"123457")
        return observe(
            c,
            argv,
            stdout=raw,
            timeout=45,
            utc_query=True,
            timed_out=case == "querytimeout",
            output_complete=case != "querytimeout",
        )

    monkeypatch.setattr(c.helper("sdsc_observed_command"), "run_observed", query)
    if case == "unique":
        assert c.reconcile(plan)["job_id"] == "123456"
    else:
        with pytest.raises(ValueError):
            c.reconcile(plan)
    assert calls and all(a[0] != "sbatch" for a in calls)


@pytest.mark.parametrize("mutation", ["job", "plan", "elapsed", "bool", "flag", "exit", "exception"])
def test_node_evidence_binding(c, plan, mutation):
    entry = c.entry_record(plan, "123456", at="now", setup_elapsed_seconds=12.5)
    exit = c.exit_record(entry, returncode=0, node_returned=True, completed_at="later")
    c.validate_node_evidence(plan, "123456", entry, exit, require_success=True)
    if mutation == "job":
        entry["job_id"] = "123457"
    elif mutation == "plan":
        entry["outer_plan_sha256"] = "0" * 64
    elif mutation == "elapsed":
        entry["setup_elapsed_seconds"] = 60.001
    elif mutation == "bool":
        entry["setup_elapsed_seconds"] = True
    elif mutation == "flag":
        entry[c.FLAGS[0]] = True
    elif mutation == "exit":
        exit["returncode"] = 1
    else:
        exit["node_returned"] = False
    with pytest.raises(ValueError):
        c.validate_node_evidence(plan, "123456", entry, exit, require_success=True)


def test_actual_admission_reserves_unknown_two_gpus(c, plan, monkeypatch, tmp_path):
    inner = plan["science_plan"]
    binary = Path(inner["python"])
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixturepython")
    binary.chmod(0o700)
    Path(inner["hf_home"]).mkdir(parents=True)
    monkeypatch.setattr(c.contract, "PYTHON_SHA256", c.sha(binary.read_bytes()))
    monkeypatch.setattr(c.science, "verify_parent", lambda *a, **k: {"verified": True})
    monkeypatch.setattr(c, "verify_execution", lambda *a, **k: {"verified": True})
    monkeypatch.setattr(c.science, "verify_runtime", lambda *a: {"verified": True})
    monkeypatch.setattr(c.science, "verify_runtime_snapshot", lambda *a: {"verified": True})
    monkeypatch.setattr(
        c.science, "check_gpu_ceiling", lambda: dict(existing_allocatable_gpus=0, new_gpus=2, limit=4)
    )
    assert c.admission(plan)["gpu_concurrency"]["total_reserved_gpus"] == 4
    monkeypatch.setattr(
        c.science, "check_gpu_ceiling", lambda: dict(existing_allocatable_gpus=1, new_gpus=2, limit=4)
    )
    with pytest.raises(ValueError, match="reserve unknown"):
        c.admission(plan)


def test_changed_fence_or_old_claim_stops_admission(c, plan):
    path = Path(plan["old_plan"]["scientific_claim"])
    path.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError):
        c.verify_fence(plan)


def test_source_and_restored_execution_comparison(c, plan, monkeypatch, tmp_path):
    pins = {**plan["execution"]["control_file_sha256"], c.CONTRACT_PATH: plan["execution"]["artifact_sha256"]}
    manifest = dict(files=[dict(path=p, sha256=h) for p, h in pins.items()])
    monkeypatch.setattr(c.science, "verify_source", lambda *a: manifest)
    assert c.verify_source(plan) == manifest
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        c.verify_source(plan)
    monkeypatch.setattr(c.science, "verify_science", lambda *a: {"verified": True})
    monkeypatch.setattr(c, "binding", lambda *a: plan["execution"])
    assert c.verify_execution(plan, tmp_path / "restored")["verified"]
    monkeypatch.setattr(c, "binding", lambda *a: {})
    with pytest.raises(ValueError):
        c.verify_execution(plan, tmp_path / "different")


def test_actual_failed_original_publication_fetch_with_recovery_evidence(c, plan, monkeypatch):
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
    raw = c.validate_download(result, plan)
    assert raw["receipt.json"] == (root / "receipt.json").read_bytes()
    assert raw["recovery/execution-plan.json"] == c.canonical(plan)
    result["files"]["recovery/execution-receipt.json"] = base64.b64encode(b"{}").decode()
    with pytest.raises(ValueError):
        c.validate_download(result, plan)


def test_duplicate_json_rejected(c):
    with pytest.raises(ValueError):
        c.decode(b'{"job":1,"job":2}')


@pytest.mark.parametrize("field,value", [("pre_model_guards_passed", 1), ("student_accepted", 0)])
def test_node_flags_require_json_boolean(c, plan, field, value):
    entry = c.entry_record(plan, "123456", at="now", setup_elapsed_seconds=1)
    entry[field] = value
    with pytest.raises(ValueError):
        c.validate_node_evidence(plan, "123456", entry)


def test_reconcile_durable_complete_ack_without_reissuing_command(c, plan, monkeypatch):
    request, calls = submit_seam(c, plan, monkeypatch)
    original = c.record_receipt
    monkeypatch.setattr(
        c, "record_receipt", lambda *a, **k: (_ for _ in ()).throw(OSError("lost receipt write"))
    )
    with pytest.raises(OSError):
        c.remote_action(request)
    monkeypatch.setattr(c, "record_receipt", original)
    monkeypatch.setattr(
        c.helper("sdsc_observed_command"),
        "run_observed",
        lambda *a, **k: pytest.fail("unexpected new command"),
    )
    assert c.reconcile(plan)["job_id"] == "123456"
    assert len(calls) == 1


def test_reconcile_trailing_sacct_separator_and_empty_comment_remain_distinct(c, plan, monkeypatch):
    request, _ = submit_seam(
        c, plan, monkeypatch, observe(c, c.sbatch(plan), timed_out=True, output_complete=False)
    )
    with pytest.raises(ValueError):
        c.remote_action(request)
    inner = plan["science_plan"]
    user = c.pwd.getpwuid(os.getuid()).pw_name

    def query(argv, **kwargs):
        raw = (
            (f"123456|{inner['job_name']}|{inner['intent_id']}|{user}|\n").encode()
            if argv[0] == "sacct"
            else b""
        )
        return observe(c, argv, stdout=raw, timeout=45, utc_query=True)

    monkeypatch.setattr(c.helper("sdsc_observed_command"), "run_observed", query)
    assert c.reconcile(plan)["job_id"] == "123456"


def test_reconcile_empty_accounting_comment_stays_unknown(c, plan, monkeypatch):
    request, _ = submit_seam(
        c, plan, monkeypatch, observe(c, c.sbatch(plan), timed_out=True, output_complete=False)
    )
    with pytest.raises(ValueError):
        c.remote_action(request)
    inner = plan["science_plan"]
    user = c.pwd.getpwuid(os.getuid()).pw_name

    def query(argv, **kwargs):
        raw = (f"123456|{inner['job_name']}||{user}|\n").encode() if argv[0] == "sacct" else b""
        return observe(c, argv, stdout=raw, timeout=45, utc_query=True)

    monkeypatch.setattr(c.helper("sdsc_observed_command"), "run_observed", query)
    with pytest.raises(ValueError, match="zero/ambiguous"):
        c.reconcile(plan)


def test_actual_reviewed_recovery_restores_full_v2_history_and_rejects_dirty_control(c, tmp_path):
    """Genuine accepted203-file science and all review anchors survive complete v2 export/restore."""
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
    git("commit", "--quiet", "-m", "Fixture proposed fenced preflight recovery")
    implementation = git("rev-parse", "HEAD").decode().strip()
    with pytest.raises(ValueError, match="not accepted"):
        c.contract.resolve_execution_contract(root)
    payload["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=implementation,
        reviewer="independent fixture",
        reviewed_at_utc="2026-10-06T06:00:00Z",
        rationale="Fixture-only fenced execution acceptance; original science unchanged",
    )
    (root / c.CONTRACT_PATH).write_bytes(c.canonical(payload) + b"\n")
    git("add", "--", c.CONTRACT_PATH)
    git("commit", "--quiet", "-m", "Fixture review-only execution acceptance")
    acceptance = git("rev-parse", "HEAD").decode().strip()
    execution = c.contract.resolve_execution_contract(root, expected_head=acceptance).as_dict()
    assert execution["science_binding"]["acceptance_commit"] == c.contract.PARENT["acceptance_commit"]
    assert len(execution["science_binding"]["science_file_sha256"]) == 203
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
        run_id="fixture-fenced-recovery",
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
    manifest = c.decode((Path(artifact["artifact"]) / "manifest.json").read_bytes())
    assert 128 < len(manifest["local_successor"]["commits"]) <= 256
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
        dirty = root / c.CONTROLLER
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


@pytest.mark.parametrize("mutation", ["pid", "nan", "timestamp", "missing", "extra"])
def test_observation_timing_and_stream_schema(c, plan, mutation):
    argv = c.sbatch(plan)
    value = observe(c, argv)
    if mutation == "pid":
        value["pid"] = True
    elif mutation == "nan":
        value["elapsed_seconds"] = float("nan")
    elif mutation == "timestamp":
        value["started_at"] = "2026-10-06T01:00:00-07:00"
    elif mutation == "missing":
        value.pop("ended_at")
    else:
        value["stdout"]["other"] = "unused"
    with pytest.raises((ValueError, KeyError, TypeError)):
        c.completed_command(value, argv, timeout=180)
