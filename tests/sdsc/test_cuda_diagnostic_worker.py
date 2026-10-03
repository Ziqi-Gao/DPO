"""Bounded CUDA diagnosis retains independent failures and never weakens admission."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def worker():
    spec = importlib.util.spec_from_file_location(
        "_test_cuda_diagnostic_worker", ROOT / "tools/sdsc_cuda_diagnostic_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCuda:
    def __init__(self, *, available=True, count=2, names=None, faults=None):
        self.available = available
        self.count = count
        self.names = names or ["NVIDIA H100 80GB HBM3"] * max(0, count)
        self.faults = faults or {}
        self.calls = []

    def call(self, api, index=None):
        self.calls.append((api, index))
        fault = self.faults.get((api, index))
        if fault:
            raise RuntimeError(fault)

    def is_available(self):
        self.call("is_available")
        return self.available

    def device_count(self):
        self.call("device_count")
        return self.count

    def init(self):
        self.call("init")

    def get_device_name(self, index):
        self.call("get_device_name", index)
        return self.names[index]

    def get_device_capability(self, index):
        self.call("get_device_capability", index)
        return (9, 0)

    def get_device_properties(self, index):
        self.call("get_device_properties", index)
        return SimpleNamespace(total_memory=80 * 1024**3, uuid=f"GPU-fixture-{index}")

    def synchronize(self, index):
        self.call("synchronize", index)


class FakeTorch:
    __version__ = "2.8.0+cu128"
    version = SimpleNamespace(cuda="12.8")
    float32 = "fixture-float32"

    def __init__(self, cuda=None, sums=None):
        self.cuda = cuda or FakeCuda()
        self.sums = sums or {}
        self.allocations = []

    def tensor(self, values, *, dtype, device):
        assert values == [1.0, 2.0, 3.0, 4.0]
        assert dtype == self.float32
        assert device in ("cuda:0", "cuda:1")
        index = int(device.split(":")[1])
        self.allocations.append(index)
        self.cuda.call("allocate", index)
        return SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: self.sums.get(index, 10.0)))


@pytest.fixture
def setup(monkeypatch):
    module = worker()
    for key in module.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "2,3",
        "SLURM_JOB_ID": "54699999",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_MEM_PER_NODE": "16384",
        "SLURM_JOB_ACCOUNT": "nwu181",
        "SLURM_JOB_PARTITION": "nairr-gpu-shared",
        "PRIVATE_API_TOKEN": "must-never-appear",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(module, "read_driver_version", lambda: {"text": "fixture driver", "truncated": False})

    def run(torch):
        monkeypatch.setattr(module, "import_torch", lambda: torch)
        return module.diagnose("54699999", "2,3")

    return module, run


def assert_failed_observation(report):
    assert report["diagnostic_complete"] is True
    assert report["cuda_ready"] is False
    assert all(report[key] is False for key in worker().FLAGS)
    assert "must-never-appear" not in json.dumps(report)
    assert "PRIVATE_API_TOKEN" not in json.dumps(report)


def test_valid_assigned_pair_uses_only_logical_indices_and_keeps_cvd(setup):
    module, run = setup
    torch = FakeTorch()
    before = dict(os.environ)
    report = run(torch)
    assert report["diagnostic_complete"] is report["cuda_ready"] is True
    assert report["cuda_visible_devices"] == "2,3"
    assert report["environment_after"] == report["environment"]
    assert dict(os.environ) == before
    assert torch.allocations == [0, 1]
    assert all(report[key] is False for key in module.FLAGS)
    assert [row["properties"]["value"]["uuid"] for row in report["devices"]] == [
        "GPU-fixture-0",
        "GPU-fixture-1",
    ]
    assert "must-never-appear" not in json.dumps(report)


@pytest.mark.parametrize(
    "cuda,failed_check",
    [
        (FakeCuda(available=False), "cuda_available"),
        (FakeCuda(count=1), "exactly_two_visible_devices"),
        (FakeCuda(count=4), "exactly_two_visible_devices"),
        (FakeCuda(names=["NVIDIA A100", "NVIDIA H100"]), "both_device_names_are_h100"),
    ],
)
def test_original_gate_failure_retains_inventory_without_allocation(setup, cuda, failed_check):
    _, run = setup
    torch = FakeTorch(cuda)
    report = run(torch)
    assert_failed_observation(report)
    assert report["checks"][failed_check] is False
    assert len(report["devices"]) == (2 if cuda.count == 2 else 0)
    assert ("init", None) in cuda.calls
    if cuda.count != 2:
        assert not any(index is not None for _, index in cuda.calls)
    assert not report["allocation_attempted"] and not torch.allocations


def test_each_cuda_api_error_retains_other_api_results_and_precise_trace(setup):
    _, run = setup
    torch = FakeTorch(
        FakeCuda(
            faults={
                ("is_available", None): "availability fault",
                ("init", None): "CUDA driver initialization error 802",
                ("get_device_name", 0): "name fault",
                ("get_device_properties", 1): "properties fault",
            }
        )
    )
    report = run(torch)
    assert_failed_observation(report)
    assert report["cuda_device_count"] == {"ok": True, "value": 2}
    assert report["cuda_init"]["error"]["type"] == "RuntimeError"
    assert "CUDA driver initialization error 802" in report["cuda_init"]["error"]["traceback"]
    assert report["devices"][0]["capability"]["value"] == [9, 0]
    assert report["devices"][0]["properties"]["ok"] is True
    assert report["devices"][1]["name"]["value"] == "NVIDIA H100 80GB HBM3"
    assert report["devices"][1]["properties"]["ok"] is False
    assert not torch.allocations


def test_count_exception_still_records_available_and_init(setup):
    _, run = setup
    torch = FakeTorch(FakeCuda(faults={("device_count", None): "count failed"}))
    report = run(torch)
    assert_failed_observation(report)
    assert report["cuda_is_available"]["value"] is True
    assert report["cuda_init"]["ok"] is True
    assert report["cuda_device_count"]["error"]["message"] == "count failed"
    assert report["devices"] == [] and torch.allocations == []


@pytest.mark.parametrize("fault_api", ["allocate", "synchronize"])
def test_per_device_cuda_failure_preserves_the_other_device_result(setup, fault_api):
    _, run = setup
    torch = FakeTorch(FakeCuda(faults={(fault_api, 0): "actual device operation failed"}))
    report = run(torch)
    assert_failed_observation(report)
    assert torch.allocations == [0, 1]
    assert report["tiny_allocations"][0]["result"]["ok"] is False
    assert report["tiny_allocations"][1]["result"]["value"]["passed"] is True


def test_nonfinite_allocation_result_fails_and_remains_strict_json(setup):
    _, run = setup
    report = run(FakeTorch(sums={0: float("nan")}))
    assert_failed_observation(report)
    assert report["tiny_allocations"][0]["result"]["value"]["finite"] is False
    assert report["tiny_allocations"][0]["result"]["value"]["sum"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "key,value,failed_check",
    [
        ("CUDA_VISIBLE_DEVICES", "0,1", "visibility_matches"),
        ("SLURM_JOB_ID", "12345", "job_matches"),
        ("SLURM_CPUS_PER_TASK", "24", "allocation_environment_matches"),
        ("SLURM_MEM_PER_NODE", "393216", "allocation_environment_matches"),
    ],
)
def test_changed_assignment_or_envelope_is_never_admitted(setup, monkeypatch, key, value, failed_check):
    _, run = setup
    monkeypatch.setenv(key, value)
    torch = FakeTorch()
    report = run(torch)
    assert_failed_observation(report)
    assert report["checks"][failed_check] is False and torch.allocations == []
    assert torch.cuda.calls == []


def test_visibility_mutation_during_observation_prevents_tiny_allocation(setup, monkeypatch):
    _, run = setup
    torch = FakeTorch()
    monkeypatch.setattr(torch.cuda, "init", lambda: os.environ.__setitem__("CUDA_VISIBLE_DEVICES", "0,1"))
    report = run(torch)
    assert_failed_observation(report)
    assert report["checks"]["visibility_unchanged"] is False
    assert torch.allocations == []


def test_runtime_mismatch_retains_gpu_inventory_but_rejects_ready(setup):
    _, run = setup
    torch = FakeTorch()
    torch.__version__ = "2.8.0+cpu"
    report = run(torch)
    assert_failed_observation(report)
    assert report["runtime_actual"]["torch"] == "2.8.0+cpu"
    assert len(report["devices"]) == 2 and torch.allocations == []


def test_missing_optional_driver_and_uuid_do_not_hide_cuda_result(setup, monkeypatch):
    module, run = setup

    def unavailable():
        raise FileNotFoundError("driver version unavailable")

    monkeypatch.setattr(module, "read_driver_version", unavailable)
    torch = FakeTorch()
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=80 * 1024**3)
    )
    report = run(torch)
    assert report["cuda_ready"] is True
    assert report["driver_version"]["error"]["type"] == "FileNotFoundError"
    assert all(row["properties"]["value"]["uuid_exposed"] is False for row in report["devices"])


def test_torch_import_failure_is_a_completed_failed_diagnosis(setup, monkeypatch):
    module, _ = setup

    def unavailable():
        raise ImportError("runtime missing torch")

    monkeypatch.setattr(module, "import_torch", unavailable)
    report = module.diagnose("54699999", "2,3")
    assert_failed_observation(report)
    assert report["torch_import"]["error"]["type"] == "ImportError"


def test_cli_publishes_failed_observations_and_refuses_overwrite(setup, tmp_path, monkeypatch, capsys):
    module, _ = setup
    monkeypatch.setattr(module, "import_torch", lambda: FakeTorch(FakeCuda(available=False)))
    output = tmp_path / "diagnostic.json"
    argv = ["--output-json", str(output), "--job-id", "54699999", "--expected-cuda-visible-devices", "2,3"]
    assert module.main(argv) == 1
    report = json.loads(output.read_text())
    assert_failed_observation(report)
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == report
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.main(argv)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp-*"))


def test_atomic_publication_does_not_delete_a_preexisting_temporary_file(tmp_path):
    module = worker()
    output = tmp_path / "diagnostic.json"
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(b"preserved prior evidence")
    with pytest.raises(FileExistsError):
        module.publish(output, {"diagnostic_complete": False})
    assert temporary.read_bytes() == b"preserved prior evidence"
    assert not output.exists()


def test_actual_node_argv_fresh_cpu_child_records_cuda_failure(tmp_path):
    import torch

    if not torch.__version__.endswith("+cpu"):
        pytest.skip("this subprocess fixture is limited to an actual CPU Torch build")
    spec = importlib.util.spec_from_file_location(
        "_test_cuda_diagnostic_node", ROOT / "tools/sdsc_cuda_diagnostic_job.py"
    )
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    release = tmp_path / "release"
    source = release / "source/tools"
    source.mkdir(parents=True)
    shutil.copyfile(ROOT / "tools/sdsc_cuda_diagnostic_worker.py", source / "sdsc_cuda_diagnostic_worker.py")
    work = tmp_path / "work"
    work.mkdir()
    output = work / "cuda-diagnostic.json"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1",
        SLURM_JOB_ID="54699999",
        SLURM_CPUS_PER_TASK="4",
        SLURM_MEM_PER_NODE="16384",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        node.worker_argv(
            {"python": sys.executable, "release": str(release)},
            work,
            {"job_id": "54699999", "cuda_visible_devices": "0,1"},
        ),
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(output.read_text())
    assert_failed_observation(report)
    assert report["torch_import"]["ok"] is True
    assert report["cuda_is_available"]["value"] is False
    assert report["cuda_device_count"]["value"] == 0
    assert report["tiny_allocations"] == []
    assert json.loads(result.stdout.splitlines()[-1]) == report


def test_cuda_init_begin_phase_is_flushed_before_the_failing_callback(setup, monkeypatch, capsys):
    _, run = setup
    torch = FakeTorch()

    def initialize():
        captured = capsys.readouterr().out.splitlines()
        assert json.loads(captured[-1])["api"] == "cuda.init"
        assert json.loads(captured[-1])["event"] == "begin"
        raise RuntimeError("initialization failed after the phase record")

    monkeypatch.setattr(torch.cuda, "init", initialize)
    report = run(torch)
    assert_failed_observation(report)
    assert report["cuda_init"]["error"]["message"] == "initialization failed after the phase record"
    assert report["devices"][1]["name"]["ok"] is True
