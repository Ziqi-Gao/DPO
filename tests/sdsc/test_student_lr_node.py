"""CPU node-boundary fixtures; no cluster or GPU calls."""

import hashlib
import importlib.util
import json
import os
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def node():
    return load("sdsc_student_lr_job")


@pytest.fixture
def control():
    return load("sdsc_student_lr")


def allocation_env(monkeypatch):
    for key, value in dict(
        SLURM_JOB_ID="123",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME="opd-lr-test",
        CUDA_VISIBLE_DEVICES="GPU-b,GPU-a",
    ).items():
        monkeypatch.setenv(key, value)


def test_allocation_preserves_supplied_gpu_order(node, control, monkeypatch):
    allocation_env(monkeypatch)
    before = os.environ["CUDA_VISIBLE_DEVICES"]
    result = node.allocation({"job_name": "opd-lr-test"}, control)
    assert result["world_size"] == 2 and result["cpus"] == 24 and result["memory_mib"] == 393216
    assert result["cuda_visible_devices"] == before == os.environ["CUDA_VISIBLE_DEVICES"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("SLURM_MEM_PER_NODE", "196608"),
        ("SLURM_CPUS_PER_TASK", "8"),
        ("SLURM_JOB_ACCOUNT", "expanse_nairr_gpu"),
        ("SLURM_NTASKS", "2"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("CUDA_VISIBLE_DEVICES", "0,0"),
        ("CUDA_VISIBLE_DEVICES", "-1,0"),
        ("CUDA_VISIBLE_DEVICES", "0, 1"),
    ],
)
def test_wrong_allocation_fails_before_work(node, control, monkeypatch, key, value):
    allocation_env(monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        node.allocation({"job_name": "opd-lr-test"}, control)


def test_fixed_launcher_has_two_processes_and_no_formal_train(node):
    argv = node.worker_argv(
        {"python": "/runtime/python"},
        Path("/work/source"),
        Path("/work/science"),
        Path("/work/inputs.json"),
        Path("/work/artifacts"),
    )
    assert argv[argv.index("--num_processes") + 1] == "2"
    assert argv[argv.index("--num_cpu_threads_per_process") + 1] == "12"
    assert argv[argv.index("--main_process_port") + 1] == "0"
    assert "/work/science/configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml" in argv
    assert "/work/source/tools/sdsc_student_lr_probe.py" in argv
    assert "posttrain_circuits.cli.train" not in argv and "--confirm-production" not in argv


def test_real_child_starts_in_science_checkout(node, tmp_path, monkeypatch):
    import sys

    source, science = tmp_path / "snapshot", tmp_path / "science"
    source.mkdir()
    science.mkdir()
    # Observe a real child, not a mocked Popen or just the configured argument.
    probe = (
        "import os,json; print(json.dumps({'cwd':os.getcwd(),'session':os.getsid(0),"
        "'pid':os.getpid(),'cvd':os.environ['CUDA_VISIBLE_DEVICES']}))"
    )
    monkeypatch.setattr(node, "worker_argv", lambda *args: [sys.executable, "-c", probe])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-b,GPU-a")
    log = tmp_path / "child.log"
    with log.open("xb") as stream:
        process = node.start_worker({}, source, science, None, None, environment, stream)
        assert process.wait(timeout=15) == 0
    observed = json.loads(log.read_text())
    assert observed["cwd"] == str(science.resolve())
    assert observed["session"] == observed["pid"] == process.pid
    assert observed["cvd"] == environment["CUDA_VISIBLE_DEVICES"]


def checkpoint(tmp_path, name, raw=b"small model-only fixture"):
    source = tmp_path / "artifacts" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(raw)
    return source, dict(
        path=name,
        size=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        arm="original" if "5e-4" in name else "candidate",
        step=4,
        scope="diagnostic_model_only",
    )


def test_checkpoint_copy_hashes_source_and_persistent_readback(node, control, tmp_path):
    source, record = checkpoint(tmp_path, node.LARGE_PATHS[0])
    dest = tmp_path / "persistent" / record["path"]
    saved = node.copy_checkpoint(source, dest, record, control)
    assert saved == {key: record[key] for key in ("path", "size", "sha256")}
    assert dest.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        node.copy_checkpoint(source, dest, record, control)


@pytest.mark.parametrize("mutation", ["hash", "size", "symlink", "oversize"])
def test_invalid_checkpoint_never_accepted(node, control, tmp_path, mutation):
    source, record = checkpoint(tmp_path, node.LARGE_PATHS[0])
    if mutation == "hash":
        record["sha256"] = "0" * 64
    elif mutation == "size":
        record["size"] += 1
    elif mutation == "oversize":
        record["size"] = node.MAX_CHECKPOINT + 1
    else:
        target = source.with_suffix(".real")
        source.rename(target)
        source.symlink_to(target)
    with pytest.raises(ValueError):
        node.copy_checkpoint(source, tmp_path / "persistent" / record["path"], record, control)


def publication_fixture(node, control, tmp_path, *, complete=True, count=2):
    rows = [checkpoint(tmp_path, name)[1] for name in node.LARGE_PATHS[:count]]
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "lr-probe.json").write_bytes(control.canonical({"checkpoints": rows}))
    destination = tmp_path / "published"
    destination.mkdir()
    plan = dict(result_dir=str(destination), run_id="test-run", intent_id="a" * 32, code_sha256="b" * 64)
    result = dict(
        job_id="123",
        diagnostic_complete=complete,
        exit_code=0 if complete else 1,
        **dict.fromkeys(control.FLAGS, False),
    )
    return plan, result, artifacts, destination, rows


def test_publication_keeps_large_files_out_of_fetch_inventory(node, control, tmp_path):
    plan, result, artifacts, destination, rows = publication_fixture(node, control, tmp_path)
    node.persist(plan, control, artifacts, None, result, {})
    receipt = json.loads((destination / "receipt.json").read_bytes())
    assert receipt["passed"] and receipt["large_files_read_back_verified"]
    assert receipt["large_files"] == [{k: r[k] for k in ("path", "size", "sha256")} for r in rows]
    assert all(row["path"] in control.NAMES for row in receipt["files"])
    assert all(receipt[flag] is False for flag in control.FLAGS)
    with pytest.raises(ValueError, match="already published"):
        node.persist(plan, control, artifacts, None, result, {})


def test_partial_second_checkpoint_failure_preserves_first_and_marks_failure(node, control, tmp_path):
    plan, result, artifacts, destination, rows = publication_fixture(node, control, tmp_path)
    (artifacts / node.LARGE_PATHS[1]).write_bytes(b"corrupted")
    node.persist(plan, control, artifacts, None, result, {})
    receipt = json.loads((destination / "receipt.json").read_bytes())
    saved = json.loads((destination / "node-result.json").read_bytes())
    assert receipt["passed"] is False and saved["exit_code"] == 1
    assert receipt["large_files"] == [{k: rows[0][k] for k in ("path", "size", "sha256")}]
    assert "checkpoint_publication_error" in saved


def test_missing_required_checkpoint_fails_success_publication(node, control, tmp_path):
    plan, result, artifacts, destination, _ = publication_fixture(node, control, tmp_path, count=1)
    node.persist(plan, control, artifacts, None, result, {})
    receipt = json.loads((destination / "receipt.json").read_bytes())
    assert receipt["passed"] is False and len(receipt["large_files"]) == 1


def test_post_publication_memory_failure_cannot_be_hidden(node, control, tmp_path):
    plan, result, artifacts, destination, _ = publication_fixture(node, control, tmp_path)
    memory = {}

    def measure(stage):
        assert stage == "after_publication"
        assert all((destination / name).exists() for name in node.LARGE_PATHS)
        memory[stage] = {"passed": False, "peak_bytes": 350 * node.GIB}
        raise ValueError("post-copy headroom failed")

    node.persist(plan, control, artifacts, None, result, memory, final_measure=measure)
    receipt = json.loads((destination / "receipt.json").read_bytes())
    assert receipt["passed"] is False and result["exit_code"] == 1
    assert json.loads((destination / "memory.json").read_bytes())["after_publication"]["passed"] is False


def test_failure_before_staging_still_publishes_metadata(node, control, tmp_path):
    dest = tmp_path / "published"
    dest.mkdir()
    plan = dict(result_dir=str(dest), run_id="test-run", intent_id="a" * 32, code_sha256="b" * 64)
    result = dict(job_id="123", diagnostic_complete=False, exit_code=1, **dict.fromkeys(control.FLAGS, False))
    node.persist(plan, control, None, None, result, {"initial": {"passed": False}})
    receipt = json.loads((dest / "receipt.json").read_bytes())
    assert receipt["passed"] is False and receipt["large_files"] == []
    assert {row["path"] for row in receipt["files"]} == {"node-result.json", "memory.json"}


def test_reused_384gib_guard_uses_job_aggregate(node, control, tmp_path):
    proc = tmp_path / "proc"
    proc.write_text("0::/job_123/step/task\n")
    root = tmp_path / "cgroup"
    for name, limit, current, peak in [
        ("job_123/step/task", "max", 2, 3),
        ("job_123/step", 384 * node.GIB, 4, 5),
        ("job_123", 384 * node.GIB, 80, 200),
    ]:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        for k, v in [
            ("memory.max", limit),
            ("memory.current", current * node.GIB),
            ("memory.peak", peak * node.GIB),
        ]:
            (p / k).write_text(str(v))
        (p / "memory.events").write_text("oom 0\noom_kill 0\n")
        (p / "memory.events.local").write_text("oom 0\noom_kill 0\n")
    result = node.memory_envelope(control, "123", proc=proc, root=root)
    assert result["peak_bytes"] == 200 * node.GIB and result["limit_bytes"] == 384 * node.GIB
    (root / "job_123/memory.peak").write_text(str(310 * node.GIB))
    with pytest.raises(ValueError, match="headroom"):
        node.memory_envelope(control, "123", proc=proc, root=root)


def test_group_scan_ignores_zombies_and_other_sessions(node, tmp_path):
    def put(pid, state, group, session):
        directory = tmp_path / str(pid)
        directory.mkdir(exist_ok=True)
        (directory / "stat").write_text(f"{pid} (command ) name) {state} 1 {group} {session} 0\n")

    put(110, "Z", 110, 110)
    put(111, "S", 110, 200)
    assert not node.group_has_live_members(110, tmp_path)
    put(112, "S", 110, 110)
    assert node.group_has_live_members(110, tmp_path)


def test_shutdown_already_exited_leader_still_stops_children(node, monkeypatch):
    calls, running = [], [True]
    process = types.SimpleNamespace(
        pid=123, poll=lambda: 0, wait=lambda timeout: calls.append(("wait", timeout))
    )

    def kill(pgid, number):
        calls.append((pgid, number))
        running[0] = False

    monkeypatch.setattr(node.os, "killpg", kill)
    node.stop_worker_group(process, alive=lambda _: running[0])
    assert calls == [(123, node.signal.SIGTERM), ("wait", 1)]


def test_shutdown_handles_group_disappearing_before_signal(node, monkeypatch):
    checks = iter([True, False, False])
    process = types.SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda timeout: 0)

    def vanished(*args):
        raise ProcessLookupError()

    monkeypatch.setattr(node.os, "killpg", vanished)
    node.stop_worker_group(process, alive=lambda _: next(checks))


def test_shutdown_forces_only_own_group_then_rejects_still_live(node, monkeypatch):
    signals = []
    clock = iter([0, 30, 30, 60])
    monkeypatch.setattr(node.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(node.os, "killpg", lambda group, number: signals.append((group, number)))
    process = types.SimpleNamespace(pid=123, poll=lambda: None, wait=lambda timeout: None)
    with pytest.raises(RuntimeError, match="still live"):
        node.stop_worker_group(process, alive=lambda _: True)
    assert signals == [(123, node.signal.SIGTERM), (123, node.signal.SIGKILL)]


@pytest.mark.parametrize(
    "filename,text",
    [
        ("memory.events", "oom 1\noom_kill 0\n"),
        ("memory.events.local", "oom 0\noom_kill 1\n"),
        ("memory.failcnt", "1"),
    ],
)
def test_low_peak_does_not_hide_own_job_oom_event(node, control, tmp_path, filename, text):
    proc = tmp_path / "proc"
    proc.write_text("0::/job_123/step\n")
    root = tmp_path / "cgroup"
    directory = root / "job_123/step"
    directory.mkdir(parents=True)
    for name, value in [
        ("memory.max", 384 * node.GIB),
        ("memory.current", 10 * node.GIB),
        ("memory.peak", 20 * node.GIB),
    ]:
        (directory / name).write_text(str(value))
    (directory / filename).write_text(text)
    with pytest.raises(ValueError) as caught:
        node.memory_envelope(control, "123", proc=proc, root=root)
    assert caught.value.evidence["passed"] is False
    assert caught.value.evidence["peak_bytes"] == 20 * node.GIB
    assert caught.value.evidence["ancestors"][0]["path"].endswith("/job_123/step")
