"""Exercise the actual versioned main through original input and native boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True).encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_once(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)


@pytest.fixture
def flow(tmp_path, monkeypatch):
    node = load("sdsc_student_name_invariant_fit_job")
    base = load("sdsc_student_name_invariant_job")
    lr = load("sdsc_student_lr_job")
    lr_control = load("sdsc_student_lr")
    calls, subprocesses = [], []
    submit = tmp_path / "submission"
    submit.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    old_science = {f"science/{index}.py": sha(str(index).encode()) for index in range(49)}
    proof_path = tmp_path / "prerequisites.json"
    proof_path.write_bytes(canonical({"protocol": {"science_file_sha256": old_science}}))
    plan = dict(
        mode="fit",
        intent_id="a" * 32,
        job_name="unique-fit-job",
        submission_dir=str(submit),
        result_dir=str(tmp_path / "persistent" / "result"),
        hf_home=str(tmp_path / "cache"),
        run_id="original-science-release",
        code_sha256="c" * 64,
        python="/pinned/source/python",
        parent={
            "receipt": {
                "result_dir": str(tmp_path / "parent"),
                "prerequisites_path": str(proof_path),
                "prerequisites_sha256": sha(proof_path.read_bytes()),
            }
        },
        runtime_snapshot={
            "archive": {"path": str(tmp_path / "archive"), "size": 1},
            "total_bytes": 2,
            "manifest": {"path": str(tmp_path / "manifest"), "size": 2, "sha256": sha(b"{}")},
        },
        prerequisites={"path": str(proof_path), "sha256": sha(proof_path.read_bytes())},
        checkpoints=[{"path": "fresh-initial.pt", "sha256": "8" * 64, "size": 3441276375}],
        resolved_config={"path": "resolved_config.yaml", "sha256": sha(b"config"), "size": 6},
        dataset_inputs=[],
        protocol={"protocol_sha256": "d" * 64},
        control_sha256={node.ORIGINAL_NAME: sha(Path(base.__file__).read_bytes())},
    )
    (tmp_path / "manifest").write_bytes(b"{}")
    outer = dict(science_plan=plan)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(node.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(node.signal, "signal", lambda *a: None)
    monkeypatch.setattr(node.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**4))
    environment = dict(
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
        CUDA_VISIBLE_DEVICES="GPU-first,GPU-second",
        TMPDIR=str(scratch),
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    def document(path, expected=None, limit=None):
        raw = Path(path).read_bytes()
        require(expected is None or sha(raw) == expected, "fixture document digest")
        return json.loads(raw)

    def stage_inputs(rows, destination, maximum):
        calls.append(("stage_inputs", rows, destination, maximum))
        destination.mkdir(parents=True)
        for row in rows:
            (destination / row["path"]).write_bytes(b"staged")
        return {"staged": len(rows)}

    class Process:
        def __init__(self, kind):
            self.kind, self.returncode, self.waited = kind, None, False
            self.fail = False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            calls.append(("wait", self.kind, timeout))
            if self.fail:
                raise RuntimeError("worker interrupted")
            self.returncode = 0
            return 0

    def popen(argv, **kwargs):
        calls.append(("runtime_popen", argv, kwargs))
        work = Path(kwargs["cwd"])
        (work / "artifacts/runtime-stage.json").write_bytes(
            canonical({"python": str(work / "runtime/bin/python")})
        )
        process = Process("runtime")
        subprocesses.append(process)
        return process

    monkeypatch.setattr(node.subprocess, "Popen", popen)
    native = SimpleNamespace(
        validate_stage=lambda *a: calls.append("validate_runtime_stage"),
        context_value=lambda p, stage, work, identity, c: dict(
            source_dir=str(work / "source"), science_dir=str(work / "science"), job_id=identity["job_id"]
        ),
        validate_context=lambda *a: calls.append("validate_runtime_context"),
        validate_evidence=lambda *a: calls.append("validate_native_evidence"),
    )

    def verify_science(p, destination):
        assert p is plan
        calls.append("restore_original_science")
        destination.mkdir()
        for index, name in enumerate(old_science):
            path = destination / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(str(index).encode())
        (destination / "protocol.json").write_bytes(b"{}")
        return {"verified": True}

    def stop_worker(process):
        calls.append(("stop_worker", process.kind))

    helpers = {
        "sdsc_student_lr": lr_control,
        "sdsc_student_lr_job": SimpleNamespace(allocation=lr.allocation, stop_worker_group=stop_worker),
        "sdsc_torch_import_probe_job": SimpleNamespace(
            stop_group=lambda p, c, **kw: calls.append(("stop_stage", p.kind, kw))
        ),
        "sdsc_student_name_invariant_native": native,
        "sdsc_student_job": SimpleNamespace(file_hash=lambda path: sha(path.read_bytes())),
        "sdsc_student_contract": SimpleNamespace(stage_inputs=stage_inputs),
        "sdsc_student_branch_qualify_job": SimpleNamespace(
            stage_models=lambda *a: calls.append("stage_models") or {}
        ),
    }
    control = SimpleNamespace(
        require=require,
        sha=sha,
        canonical=canonical,
        safe=Path,
        read=lambda p, *a: Path(p).read_bytes(),
        write_once=write_once,
        document=document,
        helper=lambda name, *a: helpers[name],
        PROJECT=tmp_path,
        CAP=1024**2,
        TASK="original-scientific-task",
        FLAGS=("qualification_passed",),
        PROTOCOL_PATH="protocol.json",
        INITIAL_SHA="8" * 64,
        NATIVE_NAMES=(),
        EXECUTION_NAMES=(),
        NAMES=("prepare-report.json", "prepare-updates.jsonl"),
        _initial=SimpleNamespace(NAMED_SCIENCE_MAP_SHA=sha(canonical(old_science))),
        worker_budget_seconds=lambda p: 28200,
        verify_parent=lambda p: calls.append("parent"),
        verify_science=verify_science,
        verify_preflight=lambda *a, **kw: pytest.fail("old incompatible gate was called"),
        validate_worker_report=lambda *a: calls.append("validate_report"),
        validate_update_records=lambda *a: calls.append("validate_updates"),
        validate_submission_receipt=lambda *a: None,
        live_binding=lambda *a: pytest.fail("node must not query mutable Slurm state"),
    )
    execution = SimpleNamespace(
        science=control,
        helper=lambda name: base,
        science_plan=lambda p: p["science_plan"],
        require=require,
        sha=sha,
        canonical=canonical,
        read=control.read,
        safe=Path,
        document=document,
        write_once=write_once,
        now=lambda: "2026-10-06T07:00:00Z",
        verify_source=lambda p: calls.append("source"),
        check_claims=lambda p: calls.append("claims"),
        verify_execution=lambda p: calls.append("execution"),
        verify_recovered_preflight=lambda p, **kw: calls.append(("recovered_preflight", kw)),
        entry_record=lambda p, job, **kw: dict(job_id=job, **kw),
        exit_record=lambda entry, **kw: dict(entry, **kw),
    )
    monkeypatch.setattr(node, "load_control", lambda *a: (execution, outer))
    monkeypatch.setattr(base, "mount", lambda path, c: {"fstype": "ext4" if path == scratch else "lustre"})
    monkeypatch.setattr(base, "memory_envelope", lambda c, job: {"passed": True})
    monkeypatch.setattr(base, "publish_live_progress", lambda *a: None)

    def stage_source(p, destination, c):
        assert p is plan
        calls.append("stage_original_source")
        destination.mkdir()
        return {"staged": True}

    monkeypatch.setattr(base, "stage_source", stage_source)

    def probe(p, c, artifacts, identity, environment, **kwargs):
        calls.append(("probe", kwargs["deadline"]))
        (artifacts / "early-node-startup.json").write_bytes(b"{}")
        return {"reaped": True, "shutdown_error": None, "probe_passed": True}

    monkeypatch.setattr(base, "run_probe", probe)
    monkeypatch.setattr(base, "validate_startup", lambda *a, **kw: calls.append(("startup", a[5])))
    monkeypatch.setattr(base, "validate_rank_exit", lambda *a: calls.append(("rank_exit", a[4])))

    def worker(p, source, science, inputs, artifacts, execution_dir, identity, environment, stream):
        assert p is plan and execution_dir == artifacts
        calls.append(("worker", json.loads(inputs.read_bytes()), environment))
        for rank in range(2):
            (artifacts / f"rank-{rank}-startup.json").write_bytes(b"{}")
            (artifacts / f"rank-{rank}-exit.json").write_bytes(b"{}")
        (artifacts / "prepare-report.json").write_bytes(b"{}")
        (artifacts / "prepare-updates.jsonl").write_bytes(b"{}\n")
        process = Process("train")
        subprocesses.append(process)
        return process

    monkeypatch.setattr(base, "start_worker", worker)
    published = []
    monkeypatch.setattr(base, "persist", lambda *args: published.append(args))
    return SimpleNamespace(
        node=node,
        base=base,
        control=control,
        execution=execution,
        plan=plan,
        outer=outer,
        root=tmp_path,
        calls=calls,
        subprocesses=subprocesses,
        published=published,
        clock=clock,
    )


def test_actual_main_preserves_original_inputs_order_environment_and_deadlines(flow):
    f = flow
    before = dict(os.environ)
    assert f.node.main(["outer", "digest"]) == 0
    assert f.calls[:5] == [
        "source",
        "claims",
        "execution",
        "parent",
        ("recovered_preflight", {"accounting": False}),
    ]
    assert f.calls.index("stage_original_source") < f.calls.index("restore_original_science")
    runtime = next(c for c in f.calls if isinstance(c, tuple) and c[0] == "runtime_popen")
    assert "sdsc_student_name_invariant_native.py" in runtime[1][4]
    assert runtime[1][-1] == "28195.0"
    assert ("probe", 28200.0) in f.calls
    worker = next(c for c in f.calls if isinstance(c, tuple) and c[0] == "worker")
    inputs, env = worker[1:]
    assert inputs["schema"] == "quest-sdsc-student-name-invariant-inputs-v1"
    assert len(inputs) == 13
    assert inputs["source_code_sha256"] == f.plan["code_sha256"]
    assert inputs["plan_sha256"] == sha(canonical(f.plan))
    assert inputs["initial_checkpoint"]["sha256"] == f.control.INITIAL_SHA
    assert inputs["initial_checkpoint"]["path"].endswith("checkpoints/initial.pt")
    assert env["CUDA_VISIBLE_DEVICES"] == before["CUDA_VISIBLE_DEVICES"]
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "12"
    assert dict(os.environ) == before
    assert f.published[0][0] is f.plan and f.published[0][1] is f.control
    assert f.published[0][4]["task"] == f.control.TASK
    assert f.published[0][4]["stage_complete"] is True
    assert ("stop_worker", "train") in f.calls
    output = json.loads((f.root / "submission/fit-node-exit.json").read_bytes())
    assert output["node_returned"] is True and output["returncode"] == 0


@pytest.mark.parametrize(
    "guard", ["verify_source", "check_claims", "verify_execution", "verify_recovered_preflight"]
)
def test_guard_failure_precedes_runtime_and_model(flow, guard):
    def fail(*a, **kw):
        raise ValueError(guard)

    setattr(flow.execution, guard, fail)
    with pytest.raises(ValueError, match=guard):
        flow.node.main(["outer", "digest"])
    assert not flow.subprocesses
    assert not (flow.root / "submission/fit-node-entry.json").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("SLURM_MEM_PER_NODE", "196608"),
        ("SLURM_CPUS_PER_TASK", "4"),
        ("SLURM_JOB_NAME", "another-plan"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("CUDA_VISIBLE_DEVICES", "0,0"),
    ],
)
def test_actual_allocation_guard_before_admission(flow, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="allocation|assigned GPUs"):
        flow.node.main(["outer", "digest"])
    assert not flow.calls


@pytest.mark.parametrize("mode", ["preflight", "unknown"])
def test_fit_only(flow, mode):
    flow.plan["mode"] = mode
    with pytest.raises(ValueError, match="fit-only"):
        flow.node.main(["outer", "digest"])
    assert not flow.calls


def test_existing_other_job_receipt_stops_before_model(flow):
    (flow.root / "submission/receipt.json").write_bytes(canonical({"job_id": "54321"}))
    with pytest.raises(ValueError, match="acknowledged job"):
        flow.node.main(["outer", "digest"])
    assert not flow.subprocesses


def test_existing_other_job_live_binding_stops_before_model(flow):
    (flow.root / "submission/live-binding.json").write_bytes(canonical({"job_id": "54321"}))
    with pytest.raises(ValueError, match="live binding"):
        flow.node.main(["outer", "digest"])
    assert not flow.subprocesses


def test_setup_budget_stops_before_scientific_staging(flow):
    flow.execution.verify_recovered_preflight = lambda *a, **kw: setattr(flow.clock, "now", 61)
    with pytest.raises(ValueError, match="setup exceeded"):
        flow.node.main(["outer", "digest"])
    assert not flow.subprocesses


def test_setup_consumes_publication_reserve_not_worker_budget(flow):
    flow.execution.verify_recovered_preflight = lambda *a, **kw: setattr(flow.clock, "now", 59)
    assert flow.node.main(["outer", "digest"]) == 0
    assert ("probe", 28259) in flow.calls


def test_probe_failure_persists_original_failure_and_outer_exit(flow, monkeypatch):
    monkeypatch.setattr(
        flow.base,
        "run_probe",
        lambda *a, **kw: {"reaped": True, "shutdown_error": None, "probe_passed": False},
    )
    assert flow.node.main(["outer", "digest"]) == 1
    assert not any(isinstance(c, tuple) and c[0] == "worker" for c in flow.calls)
    assert flow.published[0][4]["stage_complete"] is False
    record = json.loads((flow.root / "submission/fit-node-exit.json").read_bytes())
    assert record["node_returned"] is True and record["returncode"] == 1


def test_worker_failure_stops_group_and_publishes_failure(flow, monkeypatch):
    original = flow.base.start_worker

    def fail(*a, **kw):
        process = original(*a, **kw)
        process.fail = True
        return process

    monkeypatch.setattr(flow.base, "start_worker", fail)
    assert flow.node.main(["outer", "digest"]) == 1
    assert ("stop_worker", "train") in flow.calls
    assert flow.published[0][4]["stage_complete"] is False


def test_publication_exception_has_honest_outer_exit(flow, monkeypatch):
    def fail(*a):
        raise OSError("publication failed")

    monkeypatch.setattr(flow.base, "persist", fail)
    with pytest.raises(OSError, match="publication failed"):
        flow.node.main(["outer", "digest"])
    record = json.loads((flow.root / "submission/fit-node-exit.json").read_bytes())
    assert record["node_returned"] is False and record["error_type"] == "OSError"


def test_preexisting_entry_never_rearms(flow):
    path = flow.root / "submission/fit-node-entry.json"
    path.write_bytes(b"old entry")
    with pytest.raises(FileExistsError):
        flow.node.main(["outer", "digest"])
    assert path.read_bytes() == b"old entry" and not flow.subprocesses


@pytest.mark.parametrize("case", ["digest", "symlink", "empty", "large", "controller_pin", "node_pin"])
def test_real_loader_rejects_before_controller_import(tmp_path, case):
    node = load("sdsc_student_name_invariant_fit_job")
    pins = {name: sha((ROOT / name).read_bytes()) for name in (node.CONTROL_NAME, node.NODE_NAME)}
    if case.endswith("_pin"):
        pins[node.CONTROL_NAME if case == "controller_pin" else node.NODE_NAME] = "0" * 64
    raw = canonical({"execution": {"control_file_sha256": pins}})
    if case == "empty":
        raw = b""
    if case == "large":
        raw = b" " * (node.MAX_PLAN + 1)
    path = tmp_path / "plan.json"
    path.write_bytes(raw)
    if case == "symlink":
        link = tmp_path / "linked.json"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError, match="invalid fit plan|plan changed|source changed"):
        node.load_control(path, "0" * 64 if case == "digest" else sha(raw))


def test_changed_original_node_rejected_before_allocation(flow):
    flow.plan["control_sha256"][flow.node.ORIGINAL_NAME] = "0" * 64
    with pytest.raises(ValueError, match="original preparation node changed"):
        flow.node.main(["outer", "digest"])
    assert not flow.calls


def test_entry_persistence_cannot_spend_past_setup_budget(flow):
    original = flow.execution.write_once

    def slow_write(path, raw):
        original(path, raw)
        if path.name == "fit-node-entry.json":
            flow.clock.now = 61

    flow.execution.write_once = slow_write
    with pytest.raises(ValueError, match="persistence exceeded"):
        flow.node.main(["outer", "digest"])
    assert not flow.subprocesses
    record = json.loads((flow.root / "submission/fit-node-exit.json").read_bytes())
    assert record["node_returned"] is False and record["returncode"] == 1


def test_staging_deadline_does_not_launch_training(flow, monkeypatch):
    original = flow.base.run_probe

    def delayed_probe(*a, **kw):
        result = original(*a, **kw)
        flow.clock.now = 28201
        return result

    monkeypatch.setattr(flow.base, "run_probe", delayed_probe)
    assert flow.node.main(["outer", "digest"]) == 1
    assert not any(isinstance(c, tuple) and c[0] == "worker" for c in flow.calls)
    assert "staging exhausted original wall budget" in flow.published[0][4]["error"]


def test_cleanup_failure_withholds_artifacts_and_success(flow):
    lr = flow.control.helper("sdsc_student_lr_job")

    def fail(process):
        raise RuntimeError("unreaped child")

    lr.stop_worker_group = fail
    assert flow.node.main(["outer", "digest"]) == 1
    assert flow.published[0][2] is None
    assert flow.published[0][4]["stage_complete"] is False
    assert "unreaped child" in flow.published[0][4]["worker_shutdown_error"]


def test_actual_isolated_entry_rejects_bad_digest_without_outputs(tmp_path):
    import subprocess

    path = tmp_path / "execution-plan.json"
    path.write_bytes(b"{}")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "tools/sdsc_student_name_invariant_fit_job.py"),
            str(path),
            "0" * 64,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )
    assert result.returncode != 0 and "fit node plan changed" in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["execution-plan.json"]


@pytest.mark.parametrize("mutation", [None, "job_id", "plan_sha256", "Command", "Comment"])
def test_real_frozen_live_predicate_accepts_saved_timestamp_without_slurm(flow, monkeypatch, mutation):
    plan, job = flow.plan, "12345"
    lr = flow.control.helper("sdsc_student_lr")
    monkeypatch.setattr(lr, "run", lambda *a, **kw: pytest.fail("unexpected Slurm RPC"))
    observed = dict(
        job_id=job,
        plan_sha256=lr.sha(lr.canonical(plan)),
        observed_at="2026-10-01T00:00:00Z",
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
    if mutation in ("job_id", "plan_sha256"):
        observed[mutation] = "wrong"
    elif mutation:
        observed["fields"][mutation] = "wrong"
    (flow.root / "submission/live-binding.json").write_bytes(canonical(observed))
    if mutation:
        with pytest.raises(ValueError, match="live binding"):
            flow.node.main(["outer", "digest"])
        assert not flow.subprocesses
    else:
        assert flow.node.main(["outer", "digest"]) == 0
