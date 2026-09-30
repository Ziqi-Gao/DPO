"""Read-only fixture transport tests: no SSH, Slurm or GPU work."""

import base64
import copy
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def c():
    return module("sdsc_student_quality")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    pins = {name: "a" * 64 for name in c.TOOLS}
    intent = c.sha(c.canonical([c.TASK, c.PARENT_REPORT_SHA, pins[c.TOOLS[2]]]))[:32]
    runtime = str(c.PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
    cache = str(c.PROJECT / "cache/huggingface")
    return dict(
        schema="quest-sdsc-student-quality-plan-v1",
        task=c.TASK,
        scope="inference_diagnostic_only",
        resources=copy.deepcopy(c.RESOURCES),
        run_id="run-1",
        intent_id=intent,
        code_sha256="b" * 64,
        control_sha256=pins,
        release=str(c.CONTROL / "releases/run-1"),
        created_at="2026-09-30T00:00:00Z",
        submission_dir=str(c.CONTROL / "student-quality-submissions" / intent),
        claim=str(c.CONTROL / "student-quality-claims" / (intent + ".json")),
        result_dir=str(c.PROJECT / "student-quality" / intent),
        job_name="opd-sq-" + intent,
        python=runtime,
        hf_home=cache,
        parent=dict(
            report_sha256=c.PARENT_REPORT_SHA,
            receipt=dict(
                job_id=c.PARENT_JOB,
                intent_id=c.PARENT_INTENT,
                submission_dir=str(c.CONTROL / "submissions" / c.PARENT_INTENT),
                python=runtime,
                hf_home=cache,
            ),
        ),
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
    )


@pytest.mark.parametrize("key,value", [("gpus", 2), ("gpus", True), ("mem_gib", 384), ("cpus", 1)])
def test_resource_scope_is_fixed(c, plan, key, value):
    c.validate_plan(plan)
    plan["resources"][key] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("g0_passed", True),
        ("factorial_ready", True),
        ("result_dir", "/tmp/escape"),
        ("intent_id", "c" * 32),
        ("python", "/tmp/python"),
    ],
)
def test_scope_paths_identity_are_fixed(c, plan, key, value):
    plan[key] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_sbatch_is_safe_fixed_argv(c, plan):
    args = c.sbatch(plan)
    assert args[:2] == ["sbatch", "--parsable"]
    assert {"--no-requeue", "--gpus=h100:1", "--mem=196608M", "--export=NONE", "--time=02:00:00"} <= set(args)
    assert args[-1] == c.sha(c.canonical(plan))
    subprocess.run(["bash", "-n"], input=c.job_script(plan), check=True)


def stubs(c, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    monkeypatch.setattr(c, "verify_parent", lambda p, **kw: None)


def test_unknown_ack_keeps_permanent_claim_and_never_retries(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    calls = []

    def lost(args):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, 45)

    monkeypatch.setattr(c, "run", lost)
    with pytest.raises(subprocess.TimeoutExpired):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert Path(plan["claim"]).exists()
    assert c.document(Path(plan["submission_dir"]) / "unknown.json")["no_retry"] is True
    with pytest.raises(ValueError, match="reconcile"):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert len(calls) == 1


def test_no_authorization_never_dispatches(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    monkeypatch.setattr(c, "run", lambda *args: pytest.fail("Slurm called"))
    with pytest.raises(ValueError, match="authorization"):
        c.remote_action(dict(action="submit", plan=plan, authorize=False))
    assert not Path(plan["claim"]).exists()


def test_receipt_precedes_live_binding_failure(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    monkeypatch.setattr(c, "run", lambda args: dict(returncode=0, stdout="12345;expanse\n", stderr=""))
    monkeypatch.setattr(
        c, "live_binding", lambda *args: (_ for _ in ()).throw(ValueError("lost live response"))
    )
    with pytest.raises(ValueError):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert c.document(Path(plan["submission_dir"]) / "receipt.json")["job_id"] == "12345"
    assert (Path(plan["submission_dir"]) / "unknown.json").exists()


def live(plan, c, job="123"):
    return dict(
        job_id=job,
        plan_sha256=c.sha(c.canonical(plan)),
        fields=dict(
            JobId=job,
            JobName=plan["job_name"],
            Account="nwu181",
            Partition="nairr-gpu-shared",
            QOS="nairr-gpu-shared-normal",
            Comment=plan["intent_id"],
            Command=str(Path(plan["submission_dir"]) / "job.sh"),
            WorkDir=plan["submission_dir"],
        ),
    )


def accounting(plan, comment=None):
    row = "|".join(
        [
            "123",
            "COMPLETED",
            "0:0",
            plan["job_name"],
            "nwu181",
            "nairr-gpu-shared",
            "nairr-gpu-shared-normal",
            "24",
            "192Gn",
            "200",
            "cpu=24,node=1,gres/gpu=1",
            plan["intent_id"] if comment is None else comment,
        ]
    )
    return dict(returncode=0, stdout="", stderr=""), dict(
        returncode=0, stdout=row + "\n123.batch|COMPLETED|0:0\n123.extern|COMPLETED|0:0\n", stderr=""
    )


def test_allocation_and_all_steps_required(c, plan):
    queue, account = accounting(plan)
    assert c.validate_accounting(plan, "123", queue, account)["accounting_complete"]
    for bad in [
        account["stdout"].replace("123.extern|COMPLETED|0:0\n", ""),
        account["stdout"].replace("123.batch|COMPLETED|0:0", "123.batch|FAILED|1:0"),
        account["stdout"].replace("gres/gpu=1", "gres/gpu=2"),
        account["stdout"].replace("192Gn", "384Gn"),
    ]:
        with pytest.raises(ValueError):
            c.validate_accounting(plan, "123", queue, {**account, "stdout": bad})


def test_empty_archived_comment_needs_exact_live_binding(c, plan):
    queue, account = accounting(plan, "")
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account)
    assert c.validate_accounting(plan, "123", queue, account, live(plan, c))["accounting_complete"]
    changed = live(plan, c)
    changed["fields"]["Command"] = "/tmp/other.sh"
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account, changed)
    queue, account = accounting(plan, "conflict")
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account, live(plan, c))


def test_disappearance_is_unknown(c, plan):
    empty = dict(returncode=0, stdout="", stderr="")
    value = c.validate_accounting(plan, "123", empty, empty)
    assert value["state"] == "UNKNOWN" and not value["accounting_complete"]


def test_reconcile_cannot_turn_no_match_into_submission(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    directory = Path(plan["submission_dir"])
    directory.mkdir(parents=True)
    Path(plan["claim"]).parent.mkdir(parents=True)
    c.write_once(Path(plan["claim"]), c.canonical(dict(plan_sha256=c.sha(c.canonical(plan)))))
    c.write_once(directory / "plan.json", c.canonical(plan))
    monkeypatch.setattr(c, "run", lambda args: dict(returncode=0, stdout="", stderr=""))
    with pytest.raises(ValueError, match="unknown"):
        c.remote_action(dict(action="reconcile", plan=plan))
    assert not (directory / "receipt.json").exists()


def test_read_rejects_symlink_and_oversize(c, tmp_path):
    target = tmp_path / "file"
    target.write_bytes(b"123")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        c.read(link)
    with pytest.raises(ValueError):
        c.read(target, 2)


def test_node_checks_actual_alloc_and_preserves_cuda(c, plan, monkeypatch):
    node = module("sdsc_student_quality_job")
    values = dict(
        SLURM_JOB_ID="123",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="196608",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
        CUDA_VISIBLE_DEVICES="GPU-abcd",
    )
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    assert node.allocation(plan, c)["cuda_visible_devices"] == "GPU-abcd"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError):
        node.allocation(plan, c)


def download(c, plan, *, success=True):
    raw = {name: b"{}" for name in c.NAMES}
    report = dict(
        passed=success,
        diagnostic_complete=success,
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        job_id="123",
        parent_job_id=c.PARENT_JOB,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        raw_artifacts=[
            dict(path=n, size=len(raw[n]), sha256=c.sha(raw[n]))
            for n in ("quality-records.jsonl", "quality-prompts.jsonl")
        ],
    )
    raw["quality-probe.json"] = c.canonical(report)
    publication = dict(
        task=c.TASK,
        job_id="123",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=c.sha(c.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=success,
        diagnostic_complete=success,
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        persistent_read_back_verified=True,
        files=[dict(path=n, size=len(data), sha256=c.sha(data)) for n, data in raw.items()],
    )
    raw["receipt.json"] = c.canonical(publication)
    return dict(
        receipt=c.make_receipt(plan, "123"),
        status=dict(
            success=success,
            publication=publication,
            publication_sha256=c.sha(raw["receipt.json"]),
            publication_verified=True,
            artifact_hashes_verified=True,
            evidence_state="verified_publication",
        ),
        files={n: base64.b64encode(data).decode() for n, data in raw.items()},
        bytes=sum(map(len, raw.values())),
    )


def test_download_rehashes_full_raw_after_remote_status(c, plan):
    value = download(c, plan)
    assert set(c.NAMES) <= set(c.validate_download(value, plan))
    value["files"]["quality-records.jsonl"] = base64.b64encode(b"[]").decode()
    with pytest.raises(ValueError, match="downloaded scientific bytes"):
        c.validate_download(value, plan)


def test_download_rejects_missing_prompts_and_escape(c, plan):
    value = download(c, plan)
    del value["files"]["quality-prompts.jsonl"]
    value["bytes"] -= 2
    with pytest.raises(ValueError, match="missing complete"):
        c.validate_download(value, plan)
    value = download(c, plan)
    value["files"]["../source"] = ""
    with pytest.raises(ValueError, match="unexpected fetched"):
        c.validate_download(value, plan)


def test_failed_download_requires_the_same_complete_receipt_hash_chain(c, plan):
    value = download(c, plan, success=False)
    assert set(c.NAMES) <= set(c.validate_download(value, plan))
    assert value["status"]["success"] is False
    value["files"]["quality-records.jsonl"] = base64.b64encode(b"[]").decode()
    with pytest.raises(ValueError, match="downloaded scientific bytes"):
        c.validate_download(value, plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("task", "different-task"),
        ("job_id", "456"),
        ("run_id", "different-run"),
        ("intent_id", "f" * 32),
        ("plan_sha256", "f" * 64),
        ("code_sha256", "f" * 64),
        ("factorial_ready", True),
        ("g0_passed", True),
    ],
)
def test_failed_foreign_receipt_rejected_even_with_resealed_transport(c, plan, key, value):
    result = download(c, plan, success=False)
    publication = result["status"]["publication"]
    publication[key] = value
    old = base64.b64decode(result["files"]["receipt.json"])
    raw = c.canonical(publication)
    result["files"]["receipt.json"] = base64.b64encode(raw).decode()
    result["bytes"] += len(raw) - len(old)
    result["status"]["publication_sha256"] = c.sha(raw)
    with pytest.raises(ValueError, match="unbound persistent"):
        c.validate_download(result, plan)


def test_failed_missing_receipt_allows_only_explicitly_unverified_bounded_logs(c, plan):
    result = download(c, plan, success=False)
    result["status"] = dict(
        success=False,
        publication_verified=False,
        artifact_hashes_verified=False,
        evidence_state="unpublished_logs_only",
    )
    result["files"] = {"worker.log": base64.b64encode(b"failed before publication\n").decode()}
    result["bytes"] = len(b"failed before publication\n")
    assert set(c.validate_download(result, plan)) == {"worker.log"}
    result["files"]["quality-records.jsonl"] = base64.b64encode(b"{}").decode()
    result["bytes"] += 2
    with pytest.raises(ValueError, match="only bounded logs"):
        c.validate_download(result, plan)
    del result["files"]["quality-records.jsonl"]
    raw = b"x" * (64 * 1024 + 1)
    result["files"]["worker.log"] = base64.b64encode(raw).decode()
    result["bytes"] = len(raw)
    with pytest.raises(ValueError, match="tail bound"):
        c.validate_download(result, plan)


def stage_failed_publication(c, plan, monkeypatch, *, receipt=True):
    value = download(c, plan, success=False)
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)
    for name, encoded in value["files"].items():
        if name != "receipt.json" or receipt:
            (destination / name).write_bytes(base64.b64decode(encoded))
    (destination / "worker.log").write_bytes(b"failed before the second checkpoint\n")
    directory = Path(plan["submission_dir"])
    directory.mkdir(parents=True)
    c.write_once(directory / "plan.json", c.canonical(plan))
    c.write_once(directory / "receipt.json", c.canonical(c.make_receipt(plan, "123")))
    Path(plan["claim"]).parent.mkdir(parents=True)
    c.write_once(Path(plan["claim"]), c.canonical(dict(plan_sha256=c.sha(c.canonical(plan)))))
    queue, account = accounting(plan)
    account["stdout"] = account["stdout"].replace("COMPLETED|0:0", "FAILED|1:0")
    monkeypatch.setattr(c, "run", lambda args: queue if args[0] == "squeue" else account)
    monkeypatch.setattr(c, "verify_source", lambda plan: None)
    return destination


def test_failed_remote_status_and_fetch_verify_publication_before_return(c, plan, monkeypatch):
    destination = stage_failed_publication(c, plan, monkeypatch)
    observed = c.inspect_job(plan, c.make_receipt(plan, "123"))
    assert observed["state"] == "FAILED" and observed["success"] is False
    assert observed["publication_verified"] and observed["artifact_hashes_verified"]
    result = c.remote_action(dict(action="fetch", plan=plan))
    assert set(c.NAMES) <= set(c.validate_download(result, plan))
    (destination / "quality-records.jsonl").write_bytes(b"[]")
    with pytest.raises(ValueError, match="persistent diagnostic hash"):
        c.inspect_job(plan, c.make_receipt(plan, "123"))


def test_failed_data_change_after_status_is_rejected_by_remote_fetch(c, plan, monkeypatch):
    destination = stage_failed_publication(c, plan, monkeypatch)
    original = c.inspect_job

    def changed(plan, receipt):
        result = original(plan, receipt)
        (destination / "quality-records.jsonl").write_bytes(b"[]")
        return result

    monkeypatch.setattr(c, "inspect_job", changed)
    with pytest.raises(ValueError, match="data changed during fetch"):
        c.remote_action(dict(action="fetch", plan=plan))


def test_no_receipt_remote_fetch_omits_orphan_scientific_files(c, plan, monkeypatch):
    stage_failed_publication(c, plan, monkeypatch, receipt=False)
    result = c.remote_action(dict(action="fetch", plan=plan))
    assert result["status"]["evidence_state"] == "unpublished_logs_only"
    assert result["status"]["artifact_hashes_verified"] is False
    assert set(c.validate_download(result, plan)) == {"worker.log"}


def test_failed_inventory_missing_extra_and_duplicate_files_fail_closed(c, plan):
    original = download(c, plan, success=False)
    missing = copy.deepcopy(original)
    del missing["files"]["quality-prompts.jsonl"]
    missing["bytes"] -= 2
    with pytest.raises(ValueError, match="scientific inventory"):
        c.validate_download(missing, plan)
    for bad in (
        original["status"]["publication"]["files"] * 2,
        [dict(path="other.json", size=0, sha256=c.sha(b""))],
    ):
        publication = {**original["status"]["publication"], "files": bad}
        with pytest.raises(ValueError, match="publication inventory"):
            c.publication_records(publication, plan, "123")


def test_real_failed_54557365_publication_and_raw_hashes(c):
    directory = ROOT / ".sdsc/student-quality/d47f1122872cad16aeb8e772f2e37ea2"
    fetch = directory / "fetch-186dm58f"
    if not fetch.exists():
        pytest.skip("private actual failed-run evidence absent")
    plan = c.document(directory / "plan.json")
    publication = c.document(fetch / "receipt.json")
    records = c.publication_records(publication, plan, "54557365")
    assert publication["passed"] is publication["diagnostic_complete"] is False
    for name, record in records.items():
        raw = c.read(fetch / name)
        assert len(raw) == record["size"] and c.sha(raw) == record["sha256"]
    assert len(c.read(fetch / "quality-records.jsonl").splitlines()) == 320


def test_real_parent_checkpoint_envelope(c):
    folder = ROOT / ".sdsc/fetched/54548846/fetch-rhidqpza"
    if not folder.exists():
        pytest.skip("private actual run evidence absent")
    checkpoints, config = c.checkpoint_rows(
        c.document(folder / "receipt.json"),
        c.document(folder / "adapted-calibration.json", c.PARENT_REPORT_SHA),
    )
    assert [r["label"] for r in checkpoints] == ["initial", "step20", "step33"]
    assert checkpoints[-1]["sha256"] == "d06052e51bbc80ada401110f2fe2e335dbcae5732cf1e0a9cfdee5b637e7f538"
    assert config["path"] == "artifacts/canonical_sft/resolved_config.yaml"
