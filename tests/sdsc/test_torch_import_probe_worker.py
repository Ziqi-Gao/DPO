"""Import diagnostics exercise fresh Python failure and hang paths without GPUs."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools/sdsc_torch_import_probe_worker.py"


def worker():
    spec = importlib.util.spec_from_file_location("_test_torch_import_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(label="A1"):
    return dict(
        label=label,
        job_id="54699991",
        expected_cuda_visible_devices="2,3",
        expected_thread_policy="12" if label == "B" else "inherited",
    )


@pytest.fixture
def setup(monkeypatch):
    module = worker()
    for key in module.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "SLURM_JOB_ID": "54699991",
        "SLURM_CPUS_PER_TASK": "24",
        "SLURM_MEM_PER_NODE": "16384",
        "SLURM_JOB_ACCOUNT": "nwu181",
        "SLURM_JOB_PARTITION": "nairr-gpu-shared",
        "CUDA_VISIBLE_DEVICES": "2,3",
        "PRIVATE_TOKEN": "not-observed-secret",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(module.faulthandler, "enable", lambda **kwargs: None)
    monkeypatch.setattr(module.faulthandler, "dump_traceback_later", lambda *a, **kw: None)
    monkeypatch.setattr(module.faulthandler, "cancel_dump_traceback_later", lambda: None)
    torch = SimpleNamespace(
        __version__="2.8.0+cu128", __file__="/fixture/torch/__init__.py", version=SimpleNamespace(cuda="12.8")
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda name: torch)
    return module, torch


@pytest.mark.parametrize("label", ["A1", "B", "A2"])
def test_import_only_preserves_environment_without_cuda_access(setup, monkeypatch, label):
    module, torch = setup
    if label == "B":
        for key in module.THREAD_KEYS:
            monkeypatch.setenv(key, "12")
    before = dict(os.environ)
    report = module.diagnose(**arguments(label))
    assert report["import_ready"] is report["import_succeeded"] is report["diagnostic_complete"] is True
    assert all(module.validate_report(report, **arguments(label)).values())
    assert dict(os.environ) == before
    assert report["environment"] == report["environment_after"]
    assert report["environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert all(report[key] is False for key in module.FLAGS)
    assert "not-observed-secret" not in json.dumps(report)
    assert not hasattr(torch, "cuda")  # Any accidental CUDA access would fail this fixture.


@pytest.mark.parametrize(
    "key,value",
    [
        ("CUDA_VISIBLE_DEVICES", "0,1"),
        ("SLURM_JOB_ID", "1"),
        ("SLURM_CPUS_PER_TASK", "4"),
        ("SLURM_MEM_PER_NODE", "393216"),
        ("SLURM_JOB_PARTITION", "nairr-gpu"),
    ],
)
def test_allocation_or_baseline_mismatch_fails_before_import(setup, monkeypatch, key, value):
    module, _ = setup
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    report = module.diagnose(**arguments())
    assert report["import_started"] is report["diagnostic_complete"] is False
    assert report["events"][-1]["phase"] == "launch_rejected"
    assert not all(module.validate_report(report, **arguments()).values())


def test_stack_dump_is_armed_and_partial_report_persisted_before_import(setup, monkeypatch, tmp_path):
    module, torch = setup
    output = tmp_path / "report.json"
    called = []
    monkeypatch.setattr(module.faulthandler, "enable", lambda **kw: called.append("enable"))

    def arm(seconds, **kwargs):
        assert seconds == 30 and kwargs == {"repeat": True, "file": sys.stderr, "exit": False}
        called.append("arm")

    def import_torch(name):
        assert called == ["enable", "arm"] and name == "torch"
        partial = json.loads(output.read_text())
        assert partial["import_started"] is partial["stack_dump_armed"] is True
        assert partial["diagnostic_complete"] is False
        module.validate_report(partial, **arguments())
        called.append("import")
        return torch

    monkeypatch.setattr(module.faulthandler, "dump_traceback_later", arm)
    monkeypatch.setattr(module.importlib, "import_module", import_torch)
    monkeypatch.setattr(module.faulthandler, "cancel_dump_traceback_later", lambda: called.append("cancel"))
    module.diagnose(**arguments(), output=output)
    assert called == ["enable", "arm", "import", "cancel"]
    assert not output.with_name("report.json.part").exists()


def test_import_exception_is_completed_diagnostic_not_import_success(setup, monkeypatch):
    module, _ = setup

    def failing(name):
        raise RuntimeError("dependency failed")

    monkeypatch.setattr(module.importlib, "import_module", failing)
    report = module.diagnose(**arguments())
    assert report["diagnostic_complete"] is True
    assert report["import_succeeded"] is report["import_ready"] is False
    assert report["import_error"]["message"] == "dependency failed"
    assert "RuntimeError" in report["import_error"]["traceback"]
    module.validate_report(report, **arguments())


def test_instrumentation_failure_rejects_import(setup, monkeypatch):
    module, _ = setup

    def failing(*args, **kwargs):
        raise OSError("no usable stderr")

    monkeypatch.setattr(module.faulthandler, "dump_traceback_later", failing)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    report = module.diagnose(**arguments())
    assert report["instrumentation_error"]["message"] == "no usable stderr"
    assert report["import_started"] is report["diagnostic_complete"] is False
    module.validate_report(report, **arguments())


def test_environment_mutation_and_wrong_runtime_cannot_claim_import_ready(setup, monkeypatch):
    module, torch = setup

    def importing(name):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        torch.__version__ = "2.9.0"
        return torch

    monkeypatch.setattr(module.importlib, "import_module", importing)
    report = module.diagnose(**arguments())
    assert report["import_succeeded"] is True and report["import_ready"] is False
    checks = module.validate_report(report, **arguments())
    assert checks["environment_preserved"] is checks["runtime_matches"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(import_ready=False),
        lambda r: r.update(student_accepted=True),
        lambda r: r.update(student_accepted=0),
        lambda r: r.update(import_succeeded=1),
        lambda r: r.update(stack_dump_armed=False),
        lambda r: r.update(expected_thread_policy="12"),
        lambda r: r.update(import_error={"type": "RuntimeError", "message": "x", "traceback": "x"}),
        lambda r: r.update(import_elapsed_seconds=float("nan")),
        lambda r: r.update(import_elapsed_seconds=99999),
        lambda r: r["checks"].update(runtime_matches=False),
        lambda r: r["checks"].update(runtime_matches=1),
        lambda r: r["events"][1].update(utc="2026-10-04T01:00:00"),
        lambda r: r["events"].reverse(),
        lambda r: r["events"][1].update(monotonic_seconds=-1),
        lambda r: r["events"][1].update(phase="completed"),
        lambda r: r.update(extra="unknown"),
        lambda r: r["environment"].update(PRIVATE_TOKEN="secret"),
    ],
)
def test_validator_rejects_fabricated_or_inconsistent_raw_evidence(setup, mutation):
    module, _ = setup
    report = copy.deepcopy(module.diagnose(**arguments()))
    mutation(report)
    with pytest.raises((ValueError, TypeError)):
        module.validate_report(report, **arguments())


def fresh_command(tmp_path, fake_source, *, accelerate_watchdog=False):
    module = worker()
    (tmp_path / "torch.py").write_text(fake_source)
    output = tmp_path / "probe.json"
    # Isolated interpreter plus an explicit fixture-only sys.path entry: never load real Torch.
    bootstrap = "import sys,runpy,faulthandler; sys.path.insert(0,sys.argv.pop(1)); "
    if accelerate_watchdog:
        bootstrap += (
            "original=faulthandler.dump_traceback_later; "
            "faulthandler.dump_traceback_later=lambda seconds,**kw: "
            "original(0.05,**kw) if seconds==30 else (_ for _ in ()).throw(AssertionError()); "
        )
    bootstrap += "runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    argv = [
        sys.executable,
        "-I",
        "-B",
        "-X",
        "importtime",
        "-c",
        bootstrap,
        str(tmp_path),
        str(WORKER),
        "--output-json",
        str(output),
        "--label",
        "A1",
        "--job-id",
        "54699991",
        "--expected-cuda-visible-devices",
        "2,3",
        "--expected-thread-policy",
        "inherited",
    ]
    environment = dict(os.environ)
    for key in module.ENVIRONMENT_KEYS:
        environment.pop(key, None)
    environment.update(
        {
            "SLURM_JOB_ID": "54699991",
            "SLURM_CPUS_PER_TASK": "24",
            "SLURM_MEM_PER_NODE": "16384",
            "SLURM_JOB_ACCOUNT": "nwu181",
            "SLURM_JOB_PARTITION": "nairr-gpu-shared",
            "CUDA_VISIBLE_DEVICES": "2,3",
        }
    )
    return argv, environment, output


@pytest.mark.parametrize(
    "source,success",
    [
        ("import time\ntime.sleep(0.1)\n__version__='2.8.0+cu128'\nclass version: cuda='12.8'\n", True),
        ("import time\ntime.sleep(0.1)\nraise ImportError('fixture dependency exception')\n", False),
    ],
)
def test_fresh_python_delayed_and_throwing_import_has_time_and_importtrace(tmp_path, source, success):
    argv, environment, output = fresh_command(tmp_path, source)
    result = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["import_succeeded"] is success
    assert report["diagnostic_complete"] is True
    assert report["import_elapsed_seconds"] >= 0.1
    assert "import time:" in result.stderr
    assert report["environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert all(report[key] is False for key in worker().FLAGS)
    worker().validate_report(report, **arguments())


def test_fresh_hanging_import_retains_partial_report_and_real_stack_dump(tmp_path):
    argv, environment, output = fresh_command(
        tmp_path,
        "import time\ntime.sleep(30)\n",
        accelerate_watchdog=True,
    )
    process = subprocess.Popen(
        argv, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if output.exists() and json.loads(output.read_text())["import_started"]:
                break
            time.sleep(0.01)
        else:
            pytest.fail("fresh worker did not reach import")
        time.sleep(0.15)
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    assert process.returncode != 0
    report = json.loads(output.read_text())
    assert report["stack_dump_armed"] is report["import_started"] is True
    assert report["diagnostic_complete"] is report["import_ready"] is False
    assert "Timeout (0:00:00.050000)!" in stderr and str(tmp_path / "torch.py") in stderr
    assert "import_torch_begin" in stdout and "import_torch_end" not in stdout
    worker().validate_report(report, **arguments())


def test_baseline_preserves_arbitrary_actual_inherited_thread_values(setup, monkeypatch):
    module, _ = setup
    monkeypatch.setenv("OMP_NUM_THREADS", "24")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    report = module.diagnose(**arguments())
    assert report["import_ready"] is True
    assert {key: report["environment"][key] for key in module.THREAD_KEYS} == {
        "OMP_NUM_THREADS": "24",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": None,
    }
    module.validate_report(report, **arguments())


@pytest.mark.parametrize("key", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
def test_twelve_thread_condition_requires_each_actual_variable(setup, monkeypatch, key):
    module, _ = setup
    for name in module.THREAD_KEYS:
        monkeypatch.setenv(name, "12")
    monkeypatch.setenv(key, "24")
    monkeypatch.setattr(module.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    report = module.diagnose(**arguments("B"))
    assert report["import_started"] is report["diagnostic_complete"] is False
    assert report["checks"]["thread_policy_matches"] is False
    module.validate_report(report, **arguments("B"))
