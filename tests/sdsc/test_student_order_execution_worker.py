"""Startup instrumentation leaves the accepted scientific worker untouched."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location("_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(monkeypatch):
    module = load(ROOT / "tools/sdsc_student_order_execution_worker.py")
    fakes = load(ROOT / "tests/sdsc/test_cuda_diagnostic_worker.py")
    for key in module.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in dict(
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        CUDA_VISIBLE_DEVICES="2,3",
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="2",
    ).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(module.platform, "python_version", lambda: "3.12.13")
    torch = fakes.FakeTorch()
    monkeypatch.setattr(module, "import_torch", lambda: torch)
    return module, torch, fakes


def test_ready_startup_preserves_environment_and_rng_without_allocation(setup):
    module, torch, _ = setup
    env, state = dict(os.environ), random.getstate()
    report = module.startup("run", "12345", "2,3")
    assert report["cuda_ready"] is True
    assert (
        module.validate_report(
            report, job_id="12345", expected_cuda_visible_devices="2,3", scope="rank_entry", rank=0
        )
        is report
    )
    assert dict(os.environ) == env and random.getstate() == state and not torch.allocations
    assert all(report[key] is False for key in module.FLAGS)
    assert module.file_sha(ROOT / "tools/sdsc_student_order_worker.py") == module.ORIGINAL_SHA


@pytest.mark.parametrize("failure", ["unavailable", "count1", "count4", "name", "init", "api"])
def test_independent_cuda_failures_are_retained_without_original_worker(setup, monkeypatch, failure):
    module, _, fakes = setup
    cuda = fakes.FakeCuda()
    if failure == "unavailable":
        cuda.available = False
    elif failure.startswith("count"):
        cuda.count = int(failure[-1])
    elif failure == "name":
        cuda.names[1] = "NVIDIA A100"
    elif failure == "init":
        cuda.faults[("init", None)] = "driver initialization error 802"
    else:
        cuda.faults[("is_available", None)] = "availability API failed"
    monkeypatch.setattr(module, "import_torch", lambda: fakes.FakeTorch(cuda))
    report = module.startup("run", "12345", "2,3")
    assert report["diagnostic_complete"] is True and report["cuda_ready"] is False
    module.validate_report(
        report, job_id="12345", expected_cuda_visible_devices="2,3", scope="rank_entry", rank=0
    )
    assert ("init", None) in cuda.calls
    if failure.startswith("count"):
        assert report["devices"] == [] and all(index is None for _, index in cuda.calls)
    if failure == "init":
        assert "802" in report["cuda_init"]["error"]["traceback"]
        assert len(report["devices"]) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("CUDA_VISIBLE_DEVICES", "0,1"),
        ("SLURM_MEM_PER_NODE", "16384"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "1"),
        ("SLURM_JOB_ID", "999"),
    ],
)
def test_invalid_assignment_does_not_import_torch_or_call_original(setup, monkeypatch, key, value):
    module, _, _ = setup
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(module, "import_torch", lambda: pytest.fail("invalid assignment imported torch"))
    report = module.startup("run", "12345", "2,3")
    assert not report["cuda_ready"] and "error" in report


@pytest.mark.parametrize("source", ["original", "helper"])
def test_changed_dependency_fails_before_cuda(setup, monkeypatch, source):
    module, _, _ = setup
    actual = module.file_sha
    target = module.ORIGINAL if source == "original" else module.HELPER
    monkeypatch.setattr(module, "file_sha", lambda path: "0" * 64 if path.name == target else actual(path))
    monkeypatch.setattr(module, "import_torch", lambda: pytest.fail("changed dependency imported torch"))
    report = module.startup("run", "12345", "2,3")
    assert not report["cuda_ready"] and "error" in report


@pytest.mark.parametrize(
    "mutation", ["available_int", "count_bool", "name", "init", "env", "rank", "rank_bool", "sha", "claim"]
)
def test_validator_recomputes_raw_cuda_and_assignment(setup, mutation):
    module, _, _ = setup
    report = module.startup("run", "12345", "2,3")
    if mutation == "available_int":
        report["cuda_is_available"]["value"] = 1
    elif mutation == "count_bool":
        report["cuda_device_count"]["value"] = True
    elif mutation == "name":
        report["devices"][0]["name"]["value"] = "A100"
    elif mutation == "init":
        report["cuda_init"]["ok"] = False
    elif mutation == "env":
        report["environment_after"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    elif mutation == "rank":
        report["environment"]["WORLD_SIZE"] = report["environment_after"]["WORLD_SIZE"] = "1"
    elif mutation == "rank_bool":
        report["rank"] = False
    elif mutation == "sha":
        report["original_worker_observed_sha256"] = "0" * 64
    else:
        report["student_accepted"] = True
    with pytest.raises(ValueError):
        module.validate_report(
            report, job_id="12345", expected_cuda_visible_devices="2,3", scope="rank_entry", rank=0
        )


def run_args(tmp_path):
    return [
        "run",
        "--execution-output-dir",
        str(tmp_path),
        "--job-id",
        "12345",
        "--expected-cuda-visible-devices",
        "2,3",
        "--inputs-json",
        str(tmp_path / "input.json"),
        "--output-dir",
        str(tmp_path / "scientific"),
    ]


@pytest.mark.parametrize("status", [0, 1, 2])
def test_wrapper_passes_exact_original_args_and_status_without_environment_changes(
    setup, monkeypatch, tmp_path, status
):
    module, _, _ = setup
    before = copy.deepcopy(dict(os.environ))
    received = []

    def original(argv):
        assert dict(os.environ) == before
        assert (tmp_path / "rank-0-startup.json").is_file()
        received.append(argv)
        return status

    monkeypatch.setattr(module, "invoke_original", original)
    assert module.main(run_args(tmp_path)) == status
    expected = ["--inputs-json", str(tmp_path / "input.json"), "--output-dir", str(tmp_path / "scientific")]
    assert received == [expected] and dict(os.environ) == before
    startup = json.loads((tmp_path / "rank-0-startup.json").read_text())
    module.validate_report(
        startup,
        job_id="12345",
        expected_cuda_visible_devices="2,3",
        scope="rank_entry",
        rank=0,
        original_argv=expected,
    )
    assert json.loads((tmp_path / "rank-0-exit.json").read_text())["exit_code"] == status


def test_actual_byte_pinned_original_main_receives_input_and_retains_its_own_failure_schema(setup, tmp_path):
    module, _, _ = setup
    (tmp_path / "input.json").write_text("{}")
    assert module.main(run_args(tmp_path)) == 1
    report = json.loads((tmp_path / "scientific/prepare-report.json").read_text())
    assert report["schema"] == "quest-sdsc-student-order-report-v4"
    assert "preparation input identity fields differ" in report["error"]
    assert report["passed"] is False
    assert not any(path.name.startswith("prepare-") for path in tmp_path.iterdir())


def test_one_failed_rank_has_evidence_and_never_enters_original_worker(setup, monkeypatch, tmp_path):
    module, torch, _ = setup
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    torch.cuda.available = False
    monkeypatch.setattr(module, "invoke_original", lambda argv: pytest.fail("bad CUDA invoked science"))
    assert module.main(run_args(tmp_path)) == 1
    report = json.loads((tmp_path / "rank-1-startup.json").read_text())
    assert report["rank"] == 1 and not report["cuda_ready"]
    assert not (tmp_path / "scientific").exists()


def test_original_exception_is_captured_separately_from_scientific_artifacts(setup, monkeypatch, tmp_path):
    module, _, _ = setup

    def broken(argv):
        raise RuntimeError("original invocation failed")

    monkeypatch.setattr(module, "invoke_original", broken)
    assert module.main(run_args(tmp_path)) == 1
    report = json.loads((tmp_path / "rank-0-exit.json").read_text())
    assert "original invocation failed" in report["error"]["traceback"]
    assert all(report[key] is False for key in module.FLAGS)


def test_probe_completes_without_original_worker_or_training_data(setup, monkeypatch, tmp_path):
    module, torch, _ = setup
    monkeypatch.setattr(module, "invoke_original", lambda argv: pytest.fail("probe invoked training"))
    path = tmp_path / "early-node-startup.json"
    assert (
        module.main(
            [
                "probe",
                "--output-json",
                str(path),
                "--job-id",
                "12345",
                "--expected-cuda-visible-devices",
                "2,3",
            ]
        )
        == 0
    )
    report = json.loads(path.read_text())
    module.validate_report(report, job_id="12345", expected_cuda_visible_devices="2,3", scope="early_node")
    assert torch.allocations == [] and report["original_argv"] is None


@pytest.mark.parametrize("phase", ["probe", "run"])
def test_actual_node_argv_fresh_cpu_children_fail_before_scientific_work(setup, tmp_path, phase):
    import torch

    if not torch.__version__.endswith("+cpu"):
        pytest.skip("fresh process fixture uses only an actual CPU Torch build")
    module, _, _ = setup
    node = load(ROOT / "tools/sdsc_student_order_execution_job.py")
    release = tmp_path / "release"
    source = release / "source"
    (source / "tools").mkdir(parents=True)
    for name in ("sdsc_student_order_execution_worker.py", module.ORIGINAL, module.HELPER):
        shutil.copyfile(ROOT / "tools" / name, source / "tools" / name)
    science = tmp_path / "science"
    (science / "configs/accelerate").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml",
        science / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml",
    )
    execution = tmp_path / "execution"
    execution.mkdir()
    inputs, artifacts = tmp_path / "inputs.json", tmp_path / "artifacts"
    inputs.write_text("{}")
    outer = {"science_plan": {"python": sys.executable, "release": str(release)}}
    identity = {"job_id": "12345", "cuda_visible_devices": "2,3"}
    argv = (
        node.probe_argv(outer, execution, identity)
        if phase == "probe"
        else node.worker_argv(outer, source, science, inputs, artifacts, execution, identity)
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="12")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        argv, cwd=science, env=env, text=True, capture_output=True, timeout=45, check=False
    )
    assert result.returncode != 0, result.stdout
    names = (
        ["early-node-startup.json"] if phase == "probe" else ["rank-0-startup.json", "rank-1-startup.json"]
    )
    reports = [json.loads((execution / name).read_text()) for name in names if (execution / name).exists()]
    assert reports, result.stderr
    for report in reports:
        module.validate_report(
            report,
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            scope="early_node" if phase == "probe" else "rank_entry",
            rank=report["rank"],
            original_argv=None
            if phase == "probe"
            else ["--inputs-json", str(inputs), "--output-dir", str(artifacts)],
        )
        assert report["cuda_is_available"]["value"] is False
        assert report["cuda_device_count"]["value"] == 0 and not report["cuda_ready"]
        assert report["environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert not artifacts.exists()
    assert inputs.read_text() == "{}"
