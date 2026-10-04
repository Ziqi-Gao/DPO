"""Real subprocess tracing and frozen staging seams; all CUDA results are fixtures."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location("v2_fixture_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise ValueError(message)


def write_once(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    node = load("sdsc_student_order_execution_v2_job")
    probe = load("sdsc_student_order_execution_v2_probe")
    helpers = {"sdsc_student_order_execution_v2_probe": probe}

    def helper(name):
        if name not in helpers:
            helpers[name] = load(name)
        return helpers[name]

    c = SimpleNamespace(
        helper=helper,
        require=require,
        write_once=write_once,
        canonical=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")).encode(),
        sha=lambda raw: hashlib.sha256(raw).hexdigest(),
    )
    for key in probe.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    environment = dict(os.environ)
    environment.update(
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        CUDA_VISIBLE_DEVICES="2,3",
        OMP_NUM_THREADS="7",
        MKL_NUM_THREADS="8",
        TMPDIR=str(tmp_path),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    source = tmp_path / "release/source/tools"
    source.mkdir(parents=True)
    script = source / "sdsc_student_order_execution_v2_probe.py"
    script.write_text(
        """import importlib.util,sys,platform,os,time
from types import SimpleNamespace
path=sys.argv.pop(1) if False else """
        + repr(str(ROOT / "tools/sdsc_student_order_execution_v2_probe.py"))
        + """
spec=importlib.util.spec_from_file_location("real_probe",path)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
platform.python_version=lambda:"3.12.13"
mode=os.environ.get("FIXTURE_MODE","good")
class FakeCuda:
    def is_available(self): return mode!="cuda_false"
    def device_count(self): return 2
    def init(self): return None
    def get_device_name(self,index): return "NVIDIA H100 fixture"
    def get_device_capability(self,index): return (9,0)
    def get_device_properties(self,index): return SimpleNamespace(total_memory=80*1024**3,uuid="fixture")
torch=SimpleNamespace(__version__="2.8.0+cu128",version=SimpleNamespace(cuda="12.8"),cuda=FakeCuda())
original_load=probe.load_frozen
def load():
    frozen=original_load()
    def import_torch():
        if mode=="hang": time.sleep(30)
        if mode=="import_error": raise ImportError("fixture import failure")
        if mode=="mutate_env": os.environ["OMP_NUM_THREADS"]="99"
        return torch
    frozen.import_torch=import_torch
    return frozen
probe.load_frozen=load
if mode=="flood": print("x"*(2*1024**2),flush=True)
raise SystemExit(probe.main())
"""
    )
    execution = tmp_path / "work/execution"
    execution.mkdir(parents=True)
    environment["TMPDIR"] = str(execution.parent)
    outer = {"science_plan": {"python": sys.executable, "release": str(tmp_path / "release")}}
    identity = {"job_id": "12345", "cuda_visible_devices": "2,3"}
    return node, probe, c, outer, identity, execution, environment


def run(setup, mode="good", budget=20):
    node, probe, c, outer, identity, execution, environment = setup
    environment["FIXTURE_MODE"] = mode
    meta = node.run_probe(
        outer, c, execution, identity, environment, deadline=time.monotonic() + budget, measure=lambda _: None
    )
    probe.validate_trace_meta(
        meta,
        job_id="12345",
        expected_cuda_visible_devices="2,3",
        expected_python=sys.executable,
        expected_argv=node.probe_argv(outer, execution, identity),
        log_bytes=(execution / "startup.log").read_bytes(),
    )
    return meta


def test_real_adapter_calls_frozen_worker_and_records_policy_without_changing_parent(setup):
    node, probe, c, outer, identity, execution, environment = setup
    before = dict(environment)
    meta = run(setup)
    assert meta["probe_passed"] is True and meta["reaped"] is True and meta["exit_code"] == 0
    assert meta["child_environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert all(meta["child_environment"][key] == "12" for key in probe.THREAD_KEYS)
    assert {key: environment.get(key) for key in probe.THREAD_KEYS} == {
        key: before.get(key) for key in probe.THREAD_KEYS
    }
    assert meta["child_environment"]["TMPDIR"] == str(execution.parent)
    raw = json.loads((execution / "early-node-startup.json").read_bytes())
    frozen = probe.load_frozen()
    frozen.validate_report(raw, job_id="12345", expected_cuda_visible_devices="2,3", scope="early_node")
    assert raw["wrapper_sha256"] == probe.FROZEN_SHA
    assert all(raw[key] is False for key in probe.FLAGS)
    assert b"import time:" in (execution / "startup.log").read_bytes()


@pytest.mark.parametrize("mode", ["cuda_false", "import_error", "mutate_env"])
def test_real_negative_probe_retains_trace_without_claiming_ready(setup, mode):
    meta = run(setup, mode)
    assert meta["probe_passed"] is False and meta["reaped"] is True and meta["exit_code"] == 1
    assert json.loads((setup[5] / "trace-meta.json").read_bytes()) == meta
    assert (setup[5] / "early-node-startup.json").exists()


def test_timeout_reaps_child_and_keeps_original_global_budget(setup):
    meta = run(setup, "hang", budget=5.8)
    assert meta["timed_out"] is True and meta["reaped"] is True and meta["probe_passed"] is False
    assert 5 < meta["time_limit_seconds"] <= 5.8
    assert meta["elapsed_seconds"] < 2
    assert meta["maximum_probe_seconds"] == 300
    assert any(row["event"] == "frozen_worker_begin" for row in meta["adapter_events"])
    assert not (setup[5] / "early-node-startup.json").exists()
    assert not setup[2].helper("sdsc_student_lr_job").group_has_live_members(meta["pid"])


def test_exhausted_budget_rejects_before_popen_and_preserves_negative_trace(setup, monkeypatch):
    monkeypatch.setattr(
        setup[0].subprocess, "Popen", lambda *a, **kw: pytest.fail("no remaining child budget")
    )
    meta = run(setup, budget=4)
    assert meta["pid"] is None and meta["probe_passed"] is False
    assert "exhausted" in meta["error"]


def test_large_log_retains_tail_and_exact_omission_count(setup):
    meta = run(setup, "flood")
    assert meta["probe_passed"] is True
    assert meta["log"]["bytes_seen"] > 2 * 1024**2 and meta["log"]["retained_bytes"] == 1024**2
    assert meta["log"]["truncated"] is True
    assert len((setup[5] / "startup.log").read_bytes()) == 1024**2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(elapsed_seconds=m["time_limit_seconds"] + 2),
        lambda m: m.update(reaped=False),
        lambda m: m.update(timed_out=True),
        lambda m: m.update(events_truncated=True),
        lambda m: m.update(exit_code=1),
        lambda m: m.update(student_accepted=True),
        lambda m: m["child_environment"].update(SLURM_MEM_PER_NODE="16384"),
        lambda m: m["adapter_events"][-1].update(returncode=1),
        lambda m: m["adapter_events"].reverse(),
        lambda m: m["adapter_events"][0].update(utc="2000-01-01T00:00:00+00:00"),
        lambda m: m["adapter_events"][-1].update(elapsed_seconds=9999),
        lambda m: m["adapter_events"][2].update(original_argv=["run"]),
        lambda m: m["log"].update(sha256="0" * 64),
    ],
)
def test_trace_rejects_forged_success_and_changed_bytes(setup, mutation):
    meta = copy.deepcopy(run(setup))
    mutation(meta)
    with pytest.raises((ValueError, TypeError)):
        setup[1].validate_trace_meta(
            meta,
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            expected_python=sys.executable,
            log_bytes=(setup[5] / "startup.log").read_bytes(),
        )


def test_original_rank_launcher_argv_and_thirteen_inputs_are_unchanged(setup, tmp_path):
    node, _, c, outer, identity, execution, _ = setup
    prior = load("sdsc_student_order_execution_job")
    values = (
        outer,
        tmp_path / "source",
        tmp_path / "science",
        tmp_path / "inputs.json",
        tmp_path / "artifacts",
        execution,
        identity,
    )
    assert node.worker_argv(*values) == prior.worker_argv(*values)
    assert any(
        Path(item).name == "sdsc_student_order_execution_worker.py" for item in node.worker_argv(*values)
    )


def test_source_staging_has_explicit_manifest_inputs_and_readonly_modes(tmp_path):
    node = load("sdsc_student_order_execution_v2_job")
    contract = load("sdsc_student_contract")
    release = tmp_path / "release"
    (release / "source/tools").mkdir(parents=True)
    rows = []
    for name, mode in [("main.txt", 0o644), ("tools/run.py", 0o755)]:
        data = (name + "\n").encode()
        p = release / "source" / name
        p.write_bytes(data)
        p.chmod(mode)
        rows.append(dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest(), mode=mode))
    manifest = dict(code_sha256="a" * 64, run_id="fixture", files=rows)
    payload = json.dumps(manifest).encode()
    (release / "manifest.json").write_bytes(payload)
    control = SimpleNamespace(
        safe=Path,
        require=require,
        manifest_records=lambda m: {r["path"]: r for r in m["files"]},
        document=lambda path, digest: json.loads(path.read_bytes())
        if hashlib.sha256(path.read_bytes()).hexdigest() == digest
        else None,
        helper=lambda name: contract,
    )
    result = node.stage_source(
        dict(
            release=str(release),
            code_sha256="a" * 64,
            run_id="fixture",
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
        ),
        tmp_path / "staged",
        control,
    )
    assert result["read_back_verified"] is True and result["files"] == 2
    assert (tmp_path / "staged/main.txt").stat().st_mode & 0o777 == 0o444
    assert (tmp_path / "staged/tools/run.py").stat().st_mode & 0o777 == 0o555
    assert (tmp_path / "staged").stat().st_mode & 0o777 == 0o555


def test_actual_thirteen_field_scientific_input_builder_is_reused(tmp_path, monkeypatch):
    path = ROOT / "tests/sdsc/test_student_order_execution_transport.py"
    spec = importlib.util.spec_from_file_location("old_execution_test", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    control = old.c.__wrapped__(monkeypatch)
    plan = old.plan.__wrapped__(control, tmp_path, monkeypatch)
    node = load("sdsc_student_order_execution_v2_job")
    old.test_node_reuses_original_thirteen_input_fields(control, plan, node, tmp_path)


def test_actual_main_negative_probe_publishes_trace_and_never_stages_inputs(setup, tmp_path, monkeypatch):
    node, probe, _, outer, identity, _, environment = setup
    control = load("sdsc_student_order_execution_v2")
    inner = outer["science_plan"]
    inner.update(
        result_dir=str(tmp_path / "result"),
        parent={"receipt": {"result_dir": str(tmp_path / "parent")}},
        hf_home=str(tmp_path / "cache"),
        mode="preflight",
        run_id="fixture",
        code_sha256="a" * 64,
    )
    outer["execution"] = {
        "core_sha256": "b" * 64,
        "control_file_sha256": {
            "tools/sdsc_student_order_execution_v2_job.py": hashlib.sha256(
                Path(node.__file__).read_bytes()
            ).hexdigest()
        },
    }
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("FIXTURE_MODE", "cuda_false")
    monkeypatch.setattr(node, "load_control", lambda *a: (control, outer))
    monkeypatch.setattr(node, "allocation", lambda *a: identity)
    monkeypatch.setattr(control, "verify_source", lambda *a: None)
    monkeypatch.setattr(control, "verify_preflight", lambda *a, **kw: None)
    monkeypatch.setattr(control.science, "verify_parent", lambda *a: None)
    monkeypatch.setattr(control.science, "verify_runtime", lambda *a: {"fixture": True})
    monkeypatch.setattr(control.science, "worker_budget_seconds", lambda *a: 3300)
    monkeypatch.setattr(control.science, "PROJECT", tmp_path)
    monkeypatch.setattr(node, "mount", lambda path, c: {"fstype": "ext4" if path == scratch else "lustre"})
    monkeypatch.setattr(node.shutil, "disk_usage", lambda path: SimpleNamespace(free=192 * 1024**3))
    monkeypatch.setattr(node, "memory_envelope", lambda *a: {"fixture": True})
    monkeypatch.setattr(
        node, "stage_source", lambda *a: pytest.fail("failed early probe reached source staging")
    )

    def persist(plan, c, artifacts, log, result, memory, measure):
        assert result["stage_complete"] is False and result["exit_code"] == 1
        assert artifacts is None and log is None
        c.write_once(Path(plan["result_dir"]) / "receipt.json", b'{"fixture_failed_science":true}')

    monkeypatch.setattr(node, "persist", persist)
    assert node.main(["fixture", "0" * 64]) == 1
    published = control.execution_publication(outer, "12345")
    assert published["execution_publication_verified"] is True and published["startup_passed"] is False
    root = Path(inner["result_dir"]) / "execution"
    trace = json.loads((root / "trace-meta.json").read_bytes())
    assert trace["reaped"] is True and trace["probe_passed"] is False
    raw = json.loads((root / "early-node-startup.json").read_bytes())
    assert raw["cuda_ready"] is False
    assert {row["path"] for row in published["execution_publication"]["files"]} == {
        "execution-node.json",
        "early-node-startup.json",
        "startup.log",
        "trace-meta.json",
    }
