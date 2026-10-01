"""Initial-only node transport fixtures; no cluster or GPU operations."""

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
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def node():
    return load("sdsc_student_initial_job")


@pytest.fixture
def control():
    io = load("sdsc_student_instruction")
    return SimpleNamespace(
        require=io.require,
        safe=io.safe,
        read=io.read,
        write_once=io.write_once,
        canonical=io.canonical,
        sha=io.sha,
        FLAGS=io.FLAGS,
        TASK="qwen3-v2-student-initial-native-v1",
        MAX_FETCH=16 * 1024**2,
        NAMES=(
            "initial-probe.json",
            "initial-prompts.jsonl",
            "initial-records.jsonl",
            "node-result.json",
            "memory.json",
        ),
    )


def test_real_child_uses_science_cwd_preserves_gpu_and_has_no_stdin(node, tmp_path):
    source, science = tmp_path / "source", tmp_path / "science"
    (source / "tools").mkdir(parents=True)
    science.mkdir()
    script = source / "tools/sdsc_student_initial_probe.py"
    script.write_text(
        "import sys,os,json; print(json.dumps(dict(cwd=os.getcwd(),pid=os.getpid(),"
        "session=os.getsid(0),cvd=os.environ['CUDA_VISIBLE_DEVICES'],stdin=sys.stdin.read())))"
    )
    logfile = tmp_path / "out"
    with logfile.open("xb") as stream:
        proc = node.start_worker(
            {"python": sys.executable},
            source,
            science,
            tmp_path / "input",
            tmp_path / "artifacts",
            dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-abc"),
            stream,
        )
        assert proc.wait(timeout=15) == 0
    value = json.loads(logfile.read_bytes())
    assert value == dict(cwd=str(science.resolve()), pid=proc.pid, session=proc.pid, cvd="GPU-abc", stdin="")


def publication(tmp_path, control, complete=True):
    artifacts = tmp_path / "artifacts"
    destination = tmp_path / "persistent"
    artifacts.mkdir()
    destination.mkdir()
    (artifacts / "initial-probe.json").write_text('{"diagnostic_complete":false}\n')
    (artifacts / "initial-records.jsonl").write_text('{"partial":true}\n')
    result = dict(
        job_id="123",
        diagnostic_complete=complete,
        exit_code=0 if complete else 1,
        **dict.fromkeys(control.FLAGS, False),
    )
    plan = dict(result_dir=str(destination), run_id="test", intent_id="a" * 32, code_sha256="b" * 64)
    return artifacts, destination, result, plan


def test_terminal_partial_evidence_persists_exactly_without_acceptance(node, control, tmp_path):
    artifacts, dest, result, plan = publication(tmp_path, control, False)
    memory = {}
    node.persist(plan, control, artifacts, None, result, memory)
    receipt = json.loads((dest / "receipt.json").read_bytes())
    assert receipt["passed"] is False and receipt["diagnostic_complete"] is False
    assert all(receipt[k] is False for k in control.FLAGS)
    assert (dest / "initial-records.jsonl").read_bytes() == (artifacts / "initial-records.jsonl").read_bytes()
    for row in receipt["files"]:
        raw = (dest / row["path"]).read_bytes()
        assert row["size"] == len(raw) and row["sha256"] == control.sha(raw)
    with pytest.raises(ValueError, match="already published"):
        node.persist(plan, control, artifacts, None, result, memory)


def test_after_publication_memory_failure_prevents_success(node, control, tmp_path):
    artifacts, dest, result, plan = publication(tmp_path, control)
    memory = {}

    def measured(phase):
        assert phase == "after_publication" and (dest / "initial-records.jsonl").exists()
        memory[phase] = {"passed": False, "memory_failcnt": 1}
        raise ValueError("own job memory allocation failed")

    node.persist(plan, control, artifacts, None, result, memory, final_measure=measured)
    assert json.loads((dest / "receipt.json").read_bytes())["passed"] is False
    assert json.loads((dest / "node-result.json").read_bytes())["exit_code"] == 1
    assert json.loads((dest / "memory.json").read_bytes()) == memory


def test_failure_before_staging_only_publishes_control_evidence(node, control, tmp_path):
    _, dest, result, plan = publication(tmp_path, control, False)
    node.persist(plan, control, None, None, result, {})
    receipt = json.loads((dest / "receipt.json").read_bytes())
    assert {x["path"] for x in receipt["files"]} == {"node-result.json", "memory.json"}
    assert receipt["passed"] is False


def test_symlink_raw_evidence_rejected(node, control, tmp_path):
    artifacts, _, result, plan = publication(tmp_path, control, False)
    path = artifacts / "initial-records.jsonl"
    path.unlink()
    target = tmp_path / "external"
    target.write_text("external")
    path.symlink_to(target)
    with pytest.raises(ValueError):
        node.persist(plan, control, artifacts, None, result, {})


def allocation_env(monkeypatch):
    for k, v in dict(
        SLURM_JOB_ID="123",
        SLURM_CPUS_PER_TASK="8",
        SLURM_MEM_PER_NODE="65536",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME="native-initial",
        CUDA_VISIBLE_DEVICES="GPU-abc",
    ).items():
        monkeypatch.setenv(k, v)


@pytest.mark.parametrize(
    "key,value",
    [
        ("SLURM_MEM_PER_NODE", "16384"),
        ("SLURM_JOB_ACCOUNT", "expanse_nairr_gpu"),
        ("CUDA_VISIBLE_DEVICES", "0,1"),
        ("SLURM_CPUS_PER_TASK", "4"),
    ],
)
def test_wrong_allocation_rejected_before_model(node, control, monkeypatch, key, value):
    allocation_env(monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        node.allocation({"job_name": "native-initial"}, control)


def test_correct_allocation_preserves_visible_device(node, control, monkeypatch):
    allocation_env(monkeypatch)
    result = node.allocation({"job_name": "native-initial"}, control)
    assert result["cuda_visible_devices"] == os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert result["world_size"] == 1 and result["memory_mib"] == 65536


@pytest.mark.parametrize("foreign", [False, True])
def test_measured_limit_is_owned_by_this_job(node, control, tmp_path, foreign):
    control.helper = load
    control.validate_job_memory_events = load("sdsc_student_initial").validate_job_memory_events
    proc = tmp_path / "cgroup-membership"
    proc.write_text("0::/job_123/step_batch\n")
    root = tmp_path / "sys-cgroup"
    task = root / "job_123" / "step_batch"
    task.mkdir(parents=True)
    limit_at = root if foreign else task.parent
    for name, value in (
        ("memory.max", 64 * node.GIB),
        ("memory.current", 8 * node.GIB),
        ("memory.peak", 10 * node.GIB),
    ):
        (limit_at / name).write_text(str(value))
    if foreign:
        with pytest.raises(ValueError, match="must belong to this job") as caught:
            node.memory_envelope(control, "123", proc=proc, root=root)
        assert caught.value.evidence["passed"] is False
    else:
        assert node.memory_envelope(control, "123", proc=proc, root=root)["passed"] is True
