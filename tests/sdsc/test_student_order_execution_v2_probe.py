"""V2 adapter preserves the byte-pinned v1 CUDA report; GPU calls here are fixtures."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, directory="tools"):
    path = ROOT / directory / (name + ".py")
    spec = importlib.util.spec_from_file_location("probe_fixture_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(monkeypatch):
    module = load("sdsc_student_order_execution_v2_probe")
    frozen = module.load_frozen()
    fakes = load("test_cuda_diagnostic_worker", "tests/sdsc")
    for key in set(module.ENVIRONMENT_KEYS) | set(frozen.ENVIRONMENT_KEYS):
        monkeypatch.delenv(key, raising=False)
    for key, value in dict(
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        CUDA_VISIBLE_DEVICES="2,3",
        OMP_NUM_THREADS="12",
        MKL_NUM_THREADS="12",
        OPENBLAS_NUM_THREADS="12",
        TMPDIR="/fixture/work",
    ).items():
        monkeypatch.setenv(key, value)
    torch = fakes.FakeTorch()
    monkeypatch.setattr(frozen, "import_torch", lambda: torch)
    monkeypatch.setattr(frozen.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(module, "load_frozen", lambda: frozen)
    monkeypatch.setattr(module.faulthandler, "enable", lambda **kw: None)
    monkeypatch.setattr(module.faulthandler, "dump_traceback_later", lambda *a, **kw: None)
    monkeypatch.setattr(module.faulthandler, "cancel_dump_traceback_later", lambda: None)
    return module, frozen, torch


def argv(path):
    return ["--output-json", str(path), "--job-id", "12345", "--expected-cuda-visible-devices", "2,3"]


def test_calls_actual_frozen_main_with_original_report_and_no_model(prepared, tmp_path, capsys, monkeypatch):
    module, frozen, torch = prepared
    before = dict(os.environ)
    path = tmp_path / "early-node-startup.json"
    called = []
    original = frozen.main

    def invoke(args):
        called.append(args)
        return original(args)

    monkeypatch.setattr(frozen, "main", invoke)
    assert module.main(argv(path)) == 0
    assert called == [["probe", *argv(path)]]
    report = json.loads(path.read_bytes())
    frozen.validate_report(report, job_id="12345", expected_cuda_visible_devices="2,3", scope="early_node")
    assert report["wrapper_sha256"] == module.FROZEN_SHA and report["original_argv"] is None
    assert report["cuda_ready"] is True and all(report[key] is False for key in module.FLAGS)
    assert dict(os.environ) == before and torch.allocations == []
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    names = [row["event"] for row in events]
    assert names[:3] == ["adapter_started", "stack_dump_armed", "frozen_worker_begin"]
    assert names[-2:] == ["frozen_worker_end", "adapter_finished"]
    raw_progress = [json.loads(row["raw"]) for row in events if row["event"] == "frozen_progress"]
    assert any(row.get("api") == "import_torch" and row["event"] == "begin" for row in raw_progress)
    assert [row["monotonic_seconds"] for row in events] == sorted(row["monotonic_seconds"] for row in events)


def test_stack_watchdog_is_armed_before_frozen_import_and_cancelled(prepared, tmp_path, monkeypatch):
    module, frozen, torch = prepared
    calls = []
    monkeypatch.setattr(module.faulthandler, "enable", lambda **kw: calls.append("enable"))

    def arm(seconds, **kw):
        assert seconds == 30 and kw["repeat"] is True and kw["exit"] is False
        calls.append("arm")

    def importing():
        assert calls == ["enable", "arm"]
        calls.append("import")
        return torch

    monkeypatch.setattr(module.faulthandler, "dump_traceback_later", arm)
    monkeypatch.setattr(module.faulthandler, "cancel_dump_traceback_later", lambda: calls.append("cancel"))
    monkeypatch.setattr(frozen, "import_torch", importing)
    assert module.main(argv(tmp_path / "report.json")) == 0
    assert calls == ["enable", "arm", "import", "cancel"]


@pytest.mark.parametrize("failure", ["unavailable", "count", "init"])
def test_original_cuda_failure_retains_raw_negative_report(prepared, tmp_path, failure):
    module, frozen, torch = prepared
    if failure == "unavailable":
        torch.cuda.available = False
    if failure == "count":
        torch.cuda.count = 1
    if failure == "init":
        torch.cuda.faults[("init", None)] = "fixture CUDA802"
    path = tmp_path / "report.json"
    assert module.main(argv(path)) == 1
    report = json.loads(path.read_bytes())
    frozen.validate_report(report, job_id="12345", expected_cuda_visible_devices="2,3", scope="early_node")
    assert report["diagnostic_complete"] is True and report["cuda_ready"] is False


@pytest.mark.parametrize(
    "key", ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES"]
)
def test_bad_launch_policy_does_not_invoke_frozen_worker(prepared, tmp_path, monkeypatch, key):
    module, frozen, _ = prepared
    monkeypatch.setenv(key, "other")
    monkeypatch.setattr(frozen, "main", lambda _: pytest.fail("invalid policy invoked worker"))
    path = tmp_path / "report.json"
    assert module.main(argv(path)) == 1 and not path.exists()


def test_changed_frozen_wrapper_rejected_before_loading(monkeypatch):
    module = load("sdsc_student_order_execution_v2_probe")
    monkeypatch.setattr(module, "file_sha", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="changed"):
        module.load_frozen()


def test_timed_output_bounds_fragmented_lines_and_keeps_raw_text():
    module = load("sdsc_student_order_execution_v2_probe")
    stream = io.StringIO()
    output = module.TimedOutput(stream, 0)
    output.write('{"api":')
    output.write('"import_torch"}\n')
    first = json.loads(stream.getvalue())
    assert first["raw"] == '{"api":"import_torch"}'
    stream.seek(0)
    stream.truncate(0)
    output.write("x" * (module.MAX_EVENT_BYTES + 1))
    assert output.pending == ""
    assert json.loads(stream.getvalue())["event"] == "output_truncated"
