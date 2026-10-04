"""Real local process/pipe/timeout seams with synthetic import reports; no CUDA evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location("node_fixture_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def node():
    return load("sdsc_torch_import_probe_job")


@pytest.fixture
def allocation(monkeypatch):
    values = dict(
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="16384",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME="import-fixture",
        SLURM_JOB_ID="12345",
        CUDA_VISIBLE_DEVICES="2,3",
    )
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


@pytest.fixture
def control(node):
    worker = load("sdsc_torch_import_probe_worker")
    old = load("sdsc_student_lr_job")

    def write_once(path, raw):
        with Path(path).open("xb") as stream:
            stream.write(raw)

    def helper(name):
        return {"sdsc_torch_import_probe_worker": worker, "sdsc_student_lr_job": old}[name]

    return SimpleNamespace(
        helper=helper,
        write_once=write_once,
        canonical=node.canonical,
        sha=lambda raw: hashlib.sha256(raw).hexdigest(),
        read=lambda path: Path(path).read_bytes(),
        safe=Path,
        document=lambda path: json.loads(Path(path).read_bytes()),
        CAP=1024**2,
        MAX_FILE=1024**2,
        TASK="sdsc-torch-import-probe-v1",
        NAMES=(
            "import-observations.json",
            "node-result.json",
            "memory.json",
            *(
                name
                for label in node.LABELS
                for name in (label + "-report.json", label + ".log", label + "-proc.jsonl")
            ),
        ),
    )


@pytest.fixture
def fixture_plan(node, tmp_path, allocation, control, monkeypatch):
    source = tmp_path / "release/source/tools"
    source.mkdir(parents=True)
    worker = control.helper("sdsc_torch_import_probe_worker")
    monkeypatch.setattr(worker.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(
        worker,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: SimpleNamespace(
                __version__="2.8.0+cu128",
                __file__="/fixture/torch/__init__.py",
                version=SimpleNamespace(cuda="12.8"),
            )
        ),
    )
    monkeypatch.setattr(worker.faulthandler, "enable", lambda **kwargs: None)
    monkeypatch.setattr(worker.faulthandler, "dump_traceback_later", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker.faulthandler, "cancel_dump_traceback_later", lambda: None)
    report = worker.diagnose(
        label="A1", job_id="12345", expected_cuda_visible_devices="2,3", expected_thread_policy="inherited"
    )
    template = tmp_path / "report-template.json"
    template.write_text(json.dumps(report))
    monkeypatch.setenv("PROBE_FIXTURE_TEMPLATE", str(template))
    script = source / "sdsc_torch_import_probe_worker.py"
    script.write_text("""import json, os, sys, time
from pathlib import Path
args=sys.argv
label=args[args.index("--label")+1]
path=Path(args[args.index("--output-json")+1])
report=json.loads(Path(os.environ["PROBE_FIXTURE_TEMPLATE"]).read_bytes())
report.update(pid=os.getpid(), python_executable=sys.executable, label=label,
              expected_thread_policy="12" if label=="B" else "inherited")
report["environment"]={key:os.environ.get(key) for key in report["environment"]}
report["environment_after"]=report["environment"].copy()
report.pop("checks",None)
mode=os.environ.get("PROBE_FIXTURE_MODE","complete")
if mode=="timeout":
    report.update(events=report["events"][:3], diagnostic_complete=False, import_succeeded=False,
                  import_ready=False, import_elapsed_seconds=None, environment_after=None,
                  runtime_actual={"python":"3.12.13"}, torch_file=None)
if mode=="import_error":
    report.update(import_succeeded=False, import_ready=False,
                  import_error={"type":"ImportError","message":"fixture","traceback":"fixture"},
                  runtime_actual={"python":"3.12.13"},torch_file=None)
if mode=="wrong_runtime": report["runtime_actual"]["python"]="3.11.0"
if mode=="wrong_environment": report["environment_after"]["OMP_NUM_THREADS"]="999"
if mode!="missing": path.write_text(json.dumps(report))
print("child-fixture-ready",flush=True)
if mode=="flood": print("x"*(2*1024**2),flush=True)
if mode=="timeout": time.sleep(30)
if mode=="exit2": sys.exit(2)
""")
    work = tmp_path / "work"
    work.mkdir()
    plan = dict(
        python=sys.executable,
        release=str(tmp_path / "release"),
        job_name="import-fixture",
        result_dir=str(tmp_path / "results"),
        run_id="fixture",
        intent_id="1" * 32,
        code_sha256="a" * 64,
    )
    monkeypatch.setattr(node, "SAMPLE_SECONDS", 0.05)
    monkeypatch.setattr(node, "CHILD_SECONDS", 1.0)
    monkeypatch.setattr(node, "SHUTDOWN_SECONDS", 0.5)
    return plan, work


def observe(node, fixture_plan, control, label="A1"):
    plan, work = fixture_plan
    return node.run_phase(
        plan,
        work,
        node.allocation(plan),
        label,
        dict(os.environ),
        control,
        deadline=time.monotonic() + 4,
        measure=lambda phase: None,
    )


def test_exact_inherited_threads_and_single_changed_condition(node):
    inherited = dict(
        CUDA_VISIBLE_DEVICES="GPU-one,GPU-two",
        OMP_NUM_THREADS="7",
        MKL_NUM_THREADS="19",
        TOKEN="fixture-sensitive",
        PYTHONPATH="untrusted",
    )
    a1, b, a2 = [node.phase_environment(inherited, label) for label in node.LABELS]
    assert a1 == a2 and "OPENBLAS_NUM_THREADS" not in a1 and a1["OMP_NUM_THREADS"] == "7"
    assert {k for k in set(a1) | set(b) if a1.get(k) != b.get(k)} == set(node.THREAD_KEYS)
    assert all(b[k] == "12" for k in node.THREAD_KEYS)
    assert b["CUDA_VISIBLE_DEVICES"] == inherited["CUDA_VISIBLE_DEVICES"]
    assert "PYTHONPATH" not in a1 and inherited["PYTHONPATH"] == "untrusted"


def test_argv_is_fresh_isolated_importtime_worker(node, fixture_plan):
    plan, work = fixture_plan
    argv = node.worker_argv(plan, work, node.allocation(plan), "B")
    assert argv[:6] == [sys.executable, "-I", "-B", "-u", "-X", "importtime"]
    assert argv[-2:] == ["--expected-thread-policy", "12"]
    assert "--expected-cuda-visible-devices" in argv


@pytest.mark.parametrize(
    "key,value",
    [
        ("SLURM_CPUS_PER_TASK", "4"),
        ("SLURM_MEM_PER_NODE", "393216"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("CUDA_VISIBLE_DEVICES", "0,0"),
        ("SLURM_JOB_ACCOUNT", "other"),
        ("SLURM_NTASKS", "2"),
    ],
)
def test_wrong_allocation_rejected_before_child(node, allocation, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        node.allocation({"job_name": "import-fixture"})


@pytest.mark.parametrize("label", ["A1", "B", "A2"])
def test_real_child_pipe_proc_and_report_seam(node, fixture_plan, control, label):
    result = observe(node, fixture_plan, control, label)
    assert result["observation_complete"] is True and result["reaped"] is True
    assert result["report_valid"] and result["exit_code"] == 0 and result["import_ready"] is True
    assert result["proc"]["sample_count"] >= 1
    assert "child-fixture-ready" in (fixture_plan[1] / (label + ".log")).read_text()
    samples = [
        json.loads(line) for line in (fixture_plan[1] / (label + "-proc.jsonl")).read_text().splitlines()
    ]
    assert all(
        row["pid"] == result["pid"] and set(row["files"]) == {"stat", "status", "io", "wchan"}
        for row in samples
    )
    assert result["thread_environment"] == (
        {k: "12" for k in node.THREAD_KEYS}
        if label == "B"
        else {k: os.environ.get(k) for k in node.THREAD_KEYS}
    )


def test_real_timed_out_import_preserves_partial_report_and_reaps_child(
    node, fixture_plan, control, monkeypatch
):
    monkeypatch.setenv("PROBE_FIXTURE_MODE", "timeout")
    monkeypatch.setattr(node, "CHILD_SECONDS", 0.25)
    result = observe(node, fixture_plan, control)
    assert result["timed_out"] and result["reaped"] and result["report_valid"]
    assert result["observation_complete"] is True and result["worker_diagnostic_complete"] is False
    assert result["import_succeeded"] is None and result["import_ready"] is False
    assert result["exit_code"] == -signal.SIGTERM and result["elapsed_seconds"] < 2
    assert not Path("/proc", str(result["pid"])).exists()


def test_import_exception_is_completed_diagnostic_observation(node, fixture_plan, control, monkeypatch):
    monkeypatch.setenv("PROBE_FIXTURE_MODE", "import_error")
    result = observe(node, fixture_plan, control)
    assert result["observation_complete"] and not result["import_succeeded"] and not result["import_ready"]
    assert result["exit_code"] == 0


@pytest.mark.parametrize("mode", ["missing", "exit2", "wrong_runtime", "wrong_environment"])
def test_invalid_instrumentation_never_completes(node, fixture_plan, control, monkeypatch, mode):
    monkeypatch.setenv("PROBE_FIXTURE_MODE", mode)
    result = observe(node, fixture_plan, control)
    assert result["reaped"] and not result["observation_complete"]
    assert result["log"]["size"] > 0 and result["proc"]["sample_count"] >= 1


def test_flood_is_drained_with_bounded_log(node, fixture_plan, control, monkeypatch):
    monkeypatch.setenv("PROBE_FIXTURE_MODE", "flood")
    result = observe(node, fixture_plan, control)
    assert result["observation_complete"]
    assert result["log"]["truncated"] and result["log"]["size"] == node.MAX_LOG
    assert result["log"]["bytes_seen"] > node.MAX_LOG


def test_deadline_rejected_before_any_launch(node, fixture_plan, control, monkeypatch):
    monkeypatch.setattr(node.subprocess, "Popen", lambda *a, **k: pytest.fail("launched after deadline"))
    with pytest.raises(ValueError, match="budget exhausted"):
        plan, work = fixture_plan
        node.run_phase(
            plan,
            work,
            node.allocation(plan),
            "A1",
            dict(os.environ),
            control,
            deadline=time.monotonic() - 1,
            measure=lambda phase: None,
        )


def test_proc_capture_is_bounded_and_only_thread_environment(node, tmp_path):
    proc = tmp_path / "123"
    (proc / "task").mkdir(parents=True)
    for name in ("stat", "status", "io", "wchan"):
        (proc / name).write_text("x" * 20000)
    (proc / "environ").write_bytes(b"SECRET=do-not-copy\0OMP_NUM_THREADS=9\0CUDA_VISIBLE_DEVICES=2,3\0")
    row = node.proc_snapshot(123, "A1", 0, proc=tmp_path)
    assert "do-not-copy" not in json.dumps(row) and "SECRET" not in row["environment"]
    assert row["environment"]["OMP_NUM_THREADS"] == "9"
    assert row["files"]["wchan"]["truncated"] and len(row["files"]["wchan"]["text"]) == 512


def test_failed_sampling_still_reaps_own_child(node, fixture_plan, control, monkeypatch):
    monkeypatch.setenv("PROBE_FIXTURE_MODE", "timeout")

    def failed(_):
        raise ValueError("memory gate failure")

    plan, work = fixture_plan
    result = node.run_phase(
        plan,
        work,
        node.allocation(plan),
        "A1",
        dict(os.environ),
        control,
        deadline=time.monotonic() + 4,
        measure=failed,
    )
    assert result["reaped"] and not result["observation_complete"] and "memory gate" in result["error"]


def test_persistent_readback_failure_cannot_publish_success(node, fixture_plan, control, tmp_path):
    plan, work = fixture_plan
    Path(plan["result_dir"]).mkdir()
    control.validate_memory = lambda *a: None
    result = dict(job_id="12345", diagnostic_complete=False, imports_ready=False, exit_code=2, **node.FLAGS)
    observations = dict(phases=[], diagnostic_complete=False)
    original = control.read
    control.read = (
        lambda path: b"changed" if Path(path).parent == Path(plan["result_dir"]) else original(path)
    )
    with pytest.raises(ValueError, match="readback"):
        node.persist(plan, control, work, result, observations, {}, lambda stage: None)
    assert not (Path(plan["result_dir"]) / "receipt.json").exists()


@pytest.mark.parametrize("mode", ["complete", "timeout", "import_error"])
def test_actual_controller_accepts_real_phase_and_partial_worker_interface(
    node, fixture_plan, control, monkeypatch, mode
):
    actual = load("sdsc_torch_import_probe")
    monkeypatch.setattr(actual, "PYTHON", sys.executable)
    monkeypatch.setenv("PROBE_FIXTURE_MODE", mode)
    if mode == "timeout":
        monkeypatch.setattr(node, "CHILD_SECONDS", 0.2)
    phases = [observe(node, fixture_plan, control, label) for label in node.LABELS]
    plan, work = fixture_plan
    report = dict(
        schema="quest-sdsc-torch-import-observations-v1",
        job_id="12345",
        plan_sha256="a" * 64,
        cuda_visible_devices="2,3",
        inherited_thread_environment={k: os.environ.get(k) for k in node.THREAD_KEYS},
        cwd=str(work),
        python=sys.executable,
        diagnostic_complete=all(p["observation_complete"] for p in phases),
        imports_ready=all(p["import_ready"] for p in phases),
        phases=phases,
        sample_interval_seconds=5,
        maximum_child_seconds=180,
        worker_budget_seconds=540,
        **node.FLAGS,
    )
    reports = {label: control.document(work / (label + "-report.json")) for label in node.LABELS}
    assert actual.validate_observations(report, "12345", "2,3", reports=reports) == report
    for phase in phases:
        actual.validate_proc_samples(phase, (work / (phase["label"] + "-proc.jsonl")).read_bytes())
    assert report["diagnostic_complete"]
    assert report["imports_ready"] is (mode == "complete")


@pytest.mark.parametrize("mode", ["complete", "import_error"])
def test_actual_node_main_publishes_all_conditions_with_actual_controller_validator(
    node, fixture_plan, control, monkeypatch, mode
):
    plan, work = fixture_plan
    actual = load("sdsc_torch_import_probe")
    monkeypatch.setattr(actual, "PYTHON", sys.executable)
    monkeypatch.setenv("PROBE_FIXTURE_MODE", mode)
    monkeypatch.setenv("TMPDIR", str(work))
    monkeypatch.setattr(node, "SAMPLE_SECONDS", 5.0)
    monkeypatch.setattr(node, "CHILD_SECONDS", 180.0)
    helper = control.helper("sdsc_student_lr_job")
    monkeypatch.setattr(
        helper, "mount", lambda path, c: dict(fstype="ext4" if Path(path) == work else "lustre")
    )

    class EnvelopeError(ValueError):
        pass

    fake_node = SimpleNamespace(
        observe_memory=lambda *a: dict(passed=True), MemoryEnvelopeError=EnvelopeError
    )
    original = control.helper
    control.helper = lambda name: fake_node if name == "sdsc_cuda_diagnostic_job" else original(name)
    control.verify_source = lambda plan: None
    control.verify_runtime = lambda plan: dict(python="3.12.13")
    control.validate_memory = lambda *a: None
    control.validate_observations = actual.validate_observations
    control.validate_proc_samples = actual.validate_proc_samples
    monkeypatch.setattr(node, "load_control", lambda *a: (control, plan))
    assert node.main(["fixture-plan", "fixture-sha"]) == 0
    destination = Path(plan["result_dir"])
    published = json.loads((destination / "receipt.json").read_bytes())
    aggregate = json.loads((destination / "import-observations.json").read_bytes())
    assert published["diagnostic_complete"] and aggregate["diagnostic_complete"]
    assert published["imports_ready"] is (mode == "complete")
    assert {row["path"] for row in published["files"]} == set(control.NAMES)
    assert all(published[k] is False for k in node.FLAGS)
    pids = [phase["pid"] for phase in aggregate["phases"]]
    assert len(set(pids)) == 3 and all(not Path("/proc", str(pid)).exists() for pid in pids)
    for row in published["files"]:
        raw = (destination / row["path"]).read_bytes()
        assert len(raw) == row["size"] and hashlib.sha256(raw).hexdigest() == row["sha256"]


def test_memory_failure_clears_publication_success_and_retains_observations(node, fixture_plan, control):
    plan, work = fixture_plan
    destination = Path(plan["result_dir"])
    destination.mkdir()
    control.validate_memory = lambda *a: None
    result = dict(job_id="12345", diagnostic_complete=True, imports_ready=True, exit_code=0, **node.FLAGS)
    observations = dict(phases=[], diagnostic_complete=True, imports_ready=True)

    def fail(_):
        raise ValueError("own memory evidence failed")

    node.persist(plan, control, work, result, observations, {}, fail)
    publication = json.loads((destination / "receipt.json").read_bytes())
    assert (
        result["exit_code"] == 2
        and not publication["diagnostic_complete"]
        and not publication["imports_ready"]
    )
    assert json.loads((destination / "import-observations.json").read_bytes())["imports_ready"] is True
