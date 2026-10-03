"""No-network CUDA diagnostic admission/publication boundaries; no GPU claims."""

from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location("_test_" + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def c(monkeypatch):
    original = subprocess.run

    def guarded(argv, *args, **kwargs):
        if isinstance(argv, list) and Path(argv[0]).name in {
            "ssh",
            "sbatch",
            "sacct",
            "squeue",
            "scontrol",
            "srun",
            "scancel",
        }:
            pytest.fail("unexpected remote/Slurm process")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    return load(ROOT / "tools/sdsc_cuda_diagnostic.py")


@pytest.fixture
def n():
    return load(ROOT / "tools/sdsc_cuda_diagnostic_job.py")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    failed = dict(
        mode="preflight",
        protocol=dict(
            science_file_sha256={"tools/sdsc_student_order.py": c.FROZEN["tools/sdsc_student_order.py"]}
        ),
    )
    monkeypatch.setattr(c, "FAILED_PLAN_SHA", c.sha(c.canonical(failed)))
    monkeypatch.setattr(c._prior, "validate_plan", lambda value: value)
    pins = {k: c.sha((ROOT / k).read_bytes()) for k in c.TOOLS}
    value = dict(
        schema=c.SCHEMA,
        task=c.TASK,
        run_id="cuda-fixture",
        code_sha256="c" * 64,
        manifest_sha256="d" * 64,
        created_at="2026-10-03T18:00:00Z",
        control_sha256=pins,
        resources=copy.deepcopy(c.RESOURCES),
        failed=dict(
            job_id=c.FAILED_JOB,
            plan=failed,
            plan_sha256=c.FAILED_PLAN_SHA,
            publication_sha256=c.FAILED_PUBLICATION_SHA,
            report_sha256=c.FAILED_REPORT_SHA,
        ),
        python=c.PYTHON,
        scope=copy.deepcopy(c.SCOPE),
        **dict.fromkeys(c.FLAGS, False),
    )
    c.bind_paths(value)
    assert c.validate_plan(value) == value
    return value


@pytest.fixture
def report(monkeypatch):
    helper = load(ROOT / "tests/sdsc/test_cuda_diagnostic_worker.py")
    w = helper.worker()
    for key in w.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    for k, v in dict(
        CUDA_VISIBLE_DEVICES="2,3",
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="4",
        SLURM_MEM_PER_NODE="16384",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
    ).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(w.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(w, "read_driver_version", lambda: dict(text="fixture", truncated=False))
    monkeypatch.setattr(w, "import_torch", lambda: helper.FakeTorch())
    return w.diagnose("12345", "2,3")


def test_exact_resources_no_placement_or_training(c, plan):
    argv = c.sbatch(plan)
    assert (
        "--gpus=h100:2" in argv
        and "--cpus-per-task=4" in argv
        and "--mem=16384M" in argv
        and "--time=00:05:00" in argv
    )
    assert "--no-requeue" in argv and "--signal=B:TERM@30" in argv
    assert not any(x.startswith(("--nodelist", "--exclude", "--constraint")) for x in argv)
    assert plan["scope"] == dict(training=False, model_loading=False, scientific_acceptance=False)
    assert (
        str(Path(plan["release"]) / "source/tools/sdsc_cuda_diagnostic_job.py") in c.job_script(plan).decode()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "old"),
        ("task", "gpu-smoke"),
        ("python", "/bin/python"),
        ("intent_id", "0" * 32),
        ("claim", "/tmp/retry"),
        ("result_dir", "/tmp/out"),
        ("student_accepted", True),
        ("g0_passed", True),
        ("execution_class_certified", 1),
    ],
)
def test_strict_plan(c, plan, field, value):
    plan[field] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize("key,value", [("gpus", 1), ("cpus", 24), ("mem_gib", 192), ("time", "00:10:00")])
def test_resource_changes_rejected(c, plan, key, value):
    plan["resources"][key] = value
    c.bind_paths(plan)
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_claim_stable_across_release_and_worker_changes(c, plan):
    oldclaim = plan["claim"]
    oldintent = plan["intent_id"]
    plan["run_id"] = "another"
    plan["control_sha256"]["tools/sdsc_cuda_diagnostic_worker.py"] = "a" * 64
    c.bind_paths(plan)
    assert plan["claim"] == oldclaim and plan["intent_id"] != oldintent


def test_unreviewed_failure_rejected(c, plan):
    plan["failed"]["job_id"] = "54345483"
    c.bind_paths(plan)
    with pytest.raises(ValueError, match="failure"):
        c.validate_plan(plan)


def test_ready_raw_worker_seam(c, report):
    assert c.validate_worker_report(report, "12345", "2,3")["cuda_ready"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "falsecheck",
        "extracheck",
        "missingcheck",
        "count",
        "countbool",
        "available",
        "availableint",
        "runtime",
        "name",
        "deviceindex",
        "sum",
        "sumtype",
        "tinybool",
        "visibility",
        "environment",
        "scope",
    ],
)
def test_raw_evidence_cannot_overclaim_ready(c, report, mutation):
    if mutation == "falsecheck":
        report["checks"]["cuda_available"] = False
    elif mutation == "extracheck":
        report["checks"]["madeup"] = True
    elif mutation == "missingcheck":
        report["checks"].pop("cuda_available")
    elif mutation == "count":
        report["cuda_device_count"]["value"] = 0
    elif mutation == "countbool":
        report["cuda_device_count"]["value"] = True
    elif mutation == "available":
        report["cuda_is_available"]["value"] = False
    elif mutation == "availableint":
        report["cuda_is_available"]["value"] = 1
    elif mutation == "runtime":
        report["runtime_actual"]["torch"] = "wrong"
    elif mutation == "name":
        report["devices"][1]["name"]["value"] = "A100"
    elif mutation == "deviceindex":
        report["devices"][1]["logical_device"] = 0
    elif mutation == "sum":
        report["tiny_allocations"][1]["result"]["value"]["sum"] = 11.0
    elif mutation == "sumtype":
        report["tiny_allocations"][1]["result"]["value"]["sum"] = 10
    elif mutation == "tinybool":
        report["tiny_allocations"][1]["result"]["value"]["passed"] = 1
    elif mutation == "visibility":
        report["cuda_visible_devices"] = "0,1"
    elif mutation == "environment":
        report["environment_after"]["SLURM_JOB_ID"] = "999"
    else:
        report["student_accepted"] = True
    with pytest.raises(ValueError):
        c.validate_worker_report(report, "12345", "2,3")


def test_capture_complete_can_be_cuda_failed(c, report):
    report["checks"]["cuda_available"] = False
    report["cuda_is_available"]["value"] = False
    report["cuda_ready"] = False
    assert c.validate_worker_report(report, "12345")["diagnostic_complete"] is True


def accounting(plan, state="COMPLETED", exitcode="0:0"):
    rows = []
    for suffix in ("", ".batch", ".extern"):
        row = [
            "12345" + suffix,
            "COMPLETED" if suffix == ".extern" else state,
            "0:0" if suffix == ".extern" else exitcode,
            plan["job_name"],
            "nwu181",
            "nairr-gpu-shared",
            "nairr-gpu-shared-normal",
            "4",
            "16Gn",
            "50",
            "cpu=4,gres/gpu=2,node=1",
            plan["intent_id"],
        ]
        rows.append("|".join(row))
    return dict(returncode=0, stdout="\n".join(rows), stderr="")


@pytest.mark.parametrize(
    "state,code,success", [("COMPLETED", "0:0", True), ("FAILED", "1:0", False), ("TIMEOUT", "0:15", False)]
)
def test_accounting_distinguishes_failed_terminal(c, plan, state, code, success):
    result = c.validate_accounting(
        plan, "12345", dict(returncode=0, stdout=""), accounting(plan, state, code)
    )
    assert result["terminal"] and result["accounting_complete"] is success


def test_accounting_rejects_wrong_allocation(c, plan):
    a = accounting(plan)
    a["stdout"] = a["stdout"].replace("cpu=4,gres/gpu=2", "cpu=4,gres/gpu=1")
    with pytest.raises(ValueError, match="allocation"):
        c.validate_accounting(plan, "12345", dict(returncode=0, stdout=""), a)


def test_submit_requires_matching_preview_before_claim(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    with pytest.raises(ValueError, match="matching dry-run"):
        c.remote_action(dict(action="submit", authorize=True, plan=plan, dry_run={}))
    assert not Path(plan["claim"]).exists()


def test_unknown_submit_never_retries(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    monkeypatch.setattr(c, "admission", lambda p: {})
    calls = []

    def run(argv):
        calls.append(argv)
        return dict(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(c, "run", run)
    preview = dict(dry_run=True, blockers=[], plan_sha256=c.sha(c.canonical(plan)), argv=c.sbatch(plan))
    with pytest.raises(ValueError, match="acknowledgement"):
        c.remote_action(dict(action="submit", authorize=True, plan=plan, dry_run=preview))
    assert (
        Path(plan["claim"]).exists()
        and (Path(plan["submission_dir"]) / "unknown.json").exists()
        and len(calls) == 1
    )
    with pytest.raises(FileExistsError):
        c.remote_action(dict(action="submit", authorize=True, plan=plan, dry_run=preview))
    assert len(calls) == 1


def test_zero_reconcile_matches_does_not_rearm(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    directory = Path(plan["submission_dir"])
    directory.mkdir(parents=True)
    Path(plan["claim"]).parent.mkdir(parents=True)
    c.write_once(directory / "plan.json", c.canonical(plan))
    c.write_once(plan["claim"], c.canonical(dict(plan_sha256=c.sha(c.canonical(plan)))))
    calls = []

    def run(argv):
        calls.append(argv)
        return dict(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(c, "run", run)
    with pytest.raises(ValueError, match="zero/ambiguous"):
        c.remote_action(dict(action="reconcile", plan=plan))
    assert [x[0] for x in calls] == ["squeue", "sacct"] and Path(plan["claim"]).exists()


def test_real_prepare_to_source_verifier_seam(c, plan, tmp_path, monkeypatch):
    rows = []
    release = Path(plan["release"])
    source = release / "source"
    for name in c.TOOLS:
        raw = (ROOT / name).read_bytes()
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        rows.append(dict(path=name, size=len(raw), sha256=c.sha(raw), mode=0o644))
    rows.sort(key=lambda x: x["path"])
    manifest = dict(
        files=rows,
        code_sha256=c.sha(c.canonical(rows)),
        run_id=plan["run_id"],
        total_bytes=sum(x["size"] for x in rows),
    )
    raw = c.canonical(manifest) + b"\n"
    (release / "manifest.json").write_bytes(raw)
    failed_path = tmp_path / "failed-plan.json"
    failed_path.write_bytes(c.canonical(plan["failed"]["plan"]))
    fetch = tmp_path / "failed-fetch"
    fetch.mkdir()
    failed_report = dict(
        error="ValueError: two assigned H100s required",
        execution_complete=False,
        raw_artifacts=[],
        **dict.fromkeys(c.FLAGS, False),
    )
    failed_publication = dict(
        job_id=c.FAILED_JOB, plan_sha256=c.FAILED_PLAN_SHA, large_files=[], passed=False
    )
    report_raw = c.canonical(failed_report) + b"\n"
    publication_raw = c.canonical(failed_publication)
    (fetch / "prepare-report.json").write_bytes(report_raw)
    (fetch / "receipt.json").write_bytes(publication_raw)
    monkeypatch.setattr(c, "FAILED_REPORT_SHA", c.sha(report_raw))
    monkeypatch.setattr(c, "FAILED_PUBLICATION_SHA", c.sha(publication_raw))
    monkeypatch.setattr(c, "ROOT", source)
    cli = SimpleNamespace(
        run_record=lambda run_id: dict(
            state="deployed", manifest=manifest, deployment=dict(code_sha256=manifest["code_sha256"])
        )
    )
    prepared = c.prepare(
        SimpleNamespace(run_id=plan["run_id"], failed_plan=failed_path, failed_fetch_dir=fetch), cli
    )
    plan = c.document(Path(prepared["plan"]))
    assert c.verify_source(plan) == manifest
    source.joinpath("tools/sdsc_student_order.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source bytes"):
        c.verify_source(plan)


def memory_fixture(tmp_path, job="12345", limit=16 * 1024**3, peak=2 * 1024**3):
    root = tmp_path / "cgroup"
    directory = root / "memory/slurm" / ("job_" + job) / "step_batch"
    directory.mkdir(parents=True)
    proc = tmp_path / "proc"
    proc.write_text("7:memory:/slurm/job_" + job + "/step_batch\n")
    for d in (directory, directory.parent, root / "memory/slurm", root / "memory"):
        for n, v in [
            ("memory.limit_in_bytes", limit if d == directory.parent else 2**63 - 1),
            ("memory.usage_in_bytes", 1024**3),
            ("memory.max_usage_in_bytes", peak),
            ("memory.failcnt", 0),
        ]:
            (d / n).write_text(str(v))
        (d / "memory.oom_control").write_text("oom_kill 0\nunder_oom 0\n")
    return proc, root, directory.parent


def test_new_16gib_memory_does_not_weaken_parent(c, n, tmp_path):
    proc, root, _ = memory_fixture(tmp_path)
    value = n.observe_memory(c, "12345", proc=proc, root=root)
    assert value["passed"] and value["minimum_headroom_bytes"] == 1024**3
    with pytest.raises(Exception, match="unreviewed student memory"):
        c.helper("sdsc_student_memory").memory_envelope(expected_bytes=16 * 1024**3, proc=proc, root=root)


@pytest.mark.parametrize("mutation", ["limit", "peak", "oom", "wrongjob"])
def test_memory_failure_retains_raw(c, n, tmp_path, mutation):
    proc, root, directory = memory_fixture(tmp_path)
    if mutation == "limit":
        (directory / "memory.limit_in_bytes").write_text(str(192 * 1024**3))
    elif mutation == "peak":
        (directory / "memory.max_usage_in_bytes").write_text(str(16 * 1024**3))
    elif mutation == "oom":
        (directory / "memory.failcnt").write_text("1")
    with pytest.raises(n.MemoryEnvelopeError) as caught:
        n.observe_memory(c, "999" if mutation == "wrongjob" else "12345", proc=proc, root=root)
    assert caught.value.evidence["passed"] is False and caught.value.evidence["ancestors"]


def test_failed_publication_fetch_keeps_observations(c, n, plan, report, tmp_path, monkeypatch):
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    report["cuda_ready"] = False
    report["checks"]["cuda_available"] = False
    report["cuda_is_available"]["value"] = False
    (work / "cuda-diagnostic.json").write_bytes(c.canonical(report) + b"\n")
    (work / "worker.log").write_text("failure details")
    result = dict(
        job_id="12345",
        cuda_visible_devices="2,3",
        plan_sha256=c.sha(c.canonical(plan)),
        exit_code=1,
        diagnostic_complete=False,
        cuda_ready=False,
        **dict.fromkeys(c.FLAGS, False),
    )
    n.persist(plan, c, work, result, {}, lambda phase: None)
    monkeypatch.setattr(c, "job_queue", lambda job: dict(returncode=0, stdout=""))
    monkeypatch.setattr(c, "job_accounting", lambda job: accounting(plan, "FAILED", "1:0"))
    receipt = c.make_receipt(plan, "12345")
    status = c.inspect_job(plan, receipt)
    assert (
        status["publication_verified"]
        and status["diagnostic_complete"]
        and not status["cuda_ready"]
        and not status["success"]
    )
    assert (destination / "worker.log").read_text() == "failure details"


def test_timeout_without_report_preserves_log_and_failure(c, n, plan, tmp_path):
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    (work / "worker.log").write_text("cuda.init begin")
    result = dict(
        job_id="12345",
        cuda_visible_devices="2,3",
        exit_code=1,
        diagnostic_complete=False,
        cuda_ready=False,
        **dict.fromkeys(c.FLAGS, False),
    )
    n.persist(plan, c, work, result, {}, lambda phase: None)
    pub = c.document(destination / "receipt.json")
    assert not pub["cuda_ready"] and not pub["diagnostic_complete"]
    assert (destination / "worker.log").read_text() == "cuda.init begin"


def test_node_allocation_keeps_assigned_ids(c, n, plan, monkeypatch):
    for k, v in dict(
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="4",
        SLURM_MEM_PER_NODE="16384",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
        CUDA_VISIBLE_DEVICES="GPU-x,GPU-y",
    ).items():
        monkeypatch.setenv(k, v)
    before = dict(os.environ)
    identity = n.allocation(plan, c)
    assert identity["cuda_visible_devices"] == "GPU-x,GPU-y" and dict(os.environ) == before
    argv = n.worker_argv(plan, Path("/tmp/fixture"), identity)
    assert argv[-1] == "GPU-x,GPU-y" and "-I" in argv and "accelerate" not in argv


@pytest.mark.parametrize("mutation", ["rawlimit", "summarypeak", "headroom", "time", "backwards"])
def test_memory_reducer_rejects_self_claimed_summary(c, n, tmp_path, mutation):
    proc, root, _ = memory_fixture(tmp_path)
    value = n.observe_memory(c, "12345", proc=proc, root=root)
    memory = {stage: copy.deepcopy(value) for stage in ("initial", "final", "after_publication")}
    assert c.validate_memory(memory, "12345") is None
    row = memory["final"]
    if mutation == "rawlimit":
        next(x for x in row["ancestors"] if x["limit_bytes"] is not None)["limit_bytes"] = 192 * 1024**3
    elif mutation == "summarypeak":
        row["peak_bytes"] += 1
    elif mutation == "headroom":
        row["headroom_bytes"] += 1
    elif mutation == "time":
        row["observed_at_unix"] = float("nan")
    else:
        row["observed_at_unix"] -= 1
    with pytest.raises(ValueError):
        c.validate_memory(memory, "12345")


def test_failed_shutdown_keeps_last_log_without_accepting_report(c, n, plan, report, tmp_path):
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    (work / "cuda-diagnostic.json").write_bytes(c.canonical(report) + b"\n")
    (work / "worker.log").write_text("cuda.init begin")
    result = dict(
        job_id="12345",
        cuda_visible_devices="2,3",
        exit_code=1,
        diagnostic_complete=False,
        cuda_ready=False,
        worker_shutdown_error="still live",
        **dict.fromkeys(c.FLAGS, False),
    )
    n.persist(plan, c, work, result, {}, lambda phase: None)
    pub = c.document(destination / "receipt.json")
    assert not pub["diagnostic_complete"] and not pub["cuda_ready"]
    assert not (destination / "cuda-diagnostic.json").exists()
    assert (destination / "worker.log").read_text() == "cuda.init begin"


@pytest.mark.parametrize("payload", [b"{malformed", b'{"job_id":"wrong"}'])
def test_malformed_worker_report_cannot_hide_last_api_log(c, n, plan, tmp_path, payload):
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    (work / "cuda-diagnostic.json").write_bytes(payload)
    (work / "worker.log").write_text("cuda.init begin")
    result = dict(
        job_id="12345",
        cuda_visible_devices="2,3",
        exit_code=1,
        diagnostic_complete=False,
        cuda_ready=False,
        **dict.fromkeys(c.FLAGS, False),
    )
    n.persist(plan, c, work, result, {}, lambda phase: None)
    pub = c.document(destination / "receipt.json")
    node = c.document(destination / "node-result.json")
    assert node["report_publication_error"] and node["exit_code"] == 1
    assert not pub["diagnostic_complete"] and not pub["cuda_ready"]
    assert not (destination / "cuda-diagnostic.json").exists()
    assert (destination / "worker.log").read_text() == "cuda.init begin"


def test_setup_time_exhaustion_never_launches_child_and_publishes(c, n, plan, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(n.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(n.signal, "signal", lambda *args: None)
    monkeypatch.setattr(n, "load_control", lambda *args: (c, plan))
    monkeypatch.setattr(n, "allocation", lambda *args: dict(job_id="12345", cuda_visible_devices="2,3"))
    monkeypatch.setattr(n, "observe_memory", lambda *args: dict(passed=True, fixture_only=True))
    monkeypatch.setattr(c, "verify_source", lambda value: clock.__setitem__(0, 211.0))
    monkeypatch.setattr(
        n.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("expired setup launched a child")
    )
    assert n.main(["fixture-plan", "fixture-sha"]) == 1
    result = c.document(Path(plan["result_dir"]) / "node-result.json")
    published = c.document(Path(plan["result_dir"]) / "receipt.json")
    assert "source verification exhausted" in result["error"]
    assert result["elapsed_seconds"] == 211.0
    assert not published["cuda_ready"] and not published["diagnostic_complete"]
