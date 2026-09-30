"""CPU-only fixtures for the screen's own node memory and durable publication."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def control():
    return load("sdsc_student_instruction")


@pytest.fixture
def node():
    return load("sdsc_student_instruction_job")


def cgroup(tmp_path, *, version=1, gib=64, peak=25):
    proc, root = tmp_path / "cgroup", tmp_path / "sys"
    proc.write_text("7:memory:/slurm/job/step/task\n" if version == 1 else "0::/slurm/job/step/task\n")
    boundary = root / "memory" if version == 1 else root
    names = (
        ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
        if version == 1
        else ("memory.max", "memory.current", "memory.peak")
    )
    for relative, limit, current, observed in (
        ("slurm/job/step/task", 2**63 - 4096 if version == 1 else "max", 2 * GIB, 3 * GIB),
        ("slurm/job/step", gib * GIB, 4 * GIB, 5 * GIB),
        ("slurm/job", gib * GIB, 6 * GIB, peak * GIB),
    ):
        path = boundary / relative
        path.mkdir(parents=True, exist_ok=True)
        for name, value in zip(names, (limit, current, observed), strict=True):
            (path / name).write_text(str(value))
        (path / "memory.stat").write_text("anon 123\nfile 456\n")
        (path / "memory.events").write_text("oom 0\noom_kill 0\n")
    return proc, root, boundary / "slurm/job"


@pytest.mark.parametrize("version", [1, 2])
def test_own_64gib_envelope_uses_job_aggregate_and_32gib_headroom(control, node, tmp_path, version):
    proc, root, job = cgroup(tmp_path, version=version, peak=32)
    evidence = node.memory_envelope(control, proc=proc, root=root)
    assert evidence["passed"] and evidence["limit_bytes"] == 64 * GIB
    assert evidence["minimum_headroom_bytes"] == 32 * GIB
    assert evidence["path"] == str(job) and evidence["peak_bytes"] == 32 * GIB
    assert len(evidence["ancestors"]) == 5


@pytest.mark.parametrize("gib,peak", [(64, 33), (384, 25), (192, 25)])
def test_old_training_limit_or_insufficient_own_headroom_rejected(control, node, tmp_path, gib, peak):
    proc, root, _job = cgroup(tmp_path, gib=gib, peak=peak)
    with pytest.raises(node.MemoryEnvelopeError) as caught:
        node.memory_envelope(control, proc=proc, root=root)
    assert caught.value.evidence["passed"] is False
    assert caught.value.evidence["limit_bytes"] == gib * GIB
    assert caught.value.evidence["peak_bytes"] == peak * GIB


def test_missing_aggregate_peak_fails_closed(control, node, tmp_path):
    proc, root, job = cgroup(tmp_path)
    (job / "memory.max_usage_in_bytes").unlink()
    with pytest.raises(node.MemoryEnvelopeError, match="current and peak"):
        node.memory_envelope(control, proc=proc, root=root)


def test_failed_partial_outputs_are_published_exactly_with_false_flags(control, node, tmp_path):
    artifacts, destination = tmp_path / "artifacts", tmp_path / "persisted"
    artifacts.mkdir()
    destination.mkdir()
    raw = b'{"arm":"baseline","ordinal":0}\n'
    (artifacts / "instruction-records.jsonl").write_bytes(raw)
    plan = dict(result_dir=str(destination), run_id="test-run", intent_id="a" * 32, code_sha256="b" * 64)
    result = dict(job_id="123", diagnostic_complete=False, exit_code=1, **dict.fromkeys(control.FLAGS, False))
    node.persist(plan, control, artifacts, None, result, {"initial": {"passed": True}})
    publication = json.loads((destination / "receipt.json").read_text())
    records = control.publication_records(publication, plan, "123")
    assert publication["passed"] is publication["diagnostic_complete"] is False
    assert all(publication[key] is False for key in control.FLAGS)
    assert (destination / "instruction-records.jsonl").read_bytes() == raw
    for name, record in records.items():
        assert control.sha((destination / name).read_bytes()) == record["sha256"]
    with pytest.raises(ValueError, match="already published"):
        node.persist(plan, control, artifacts, None, result, {})


def test_failure_before_artifacts_still_has_a_sealed_metadata_receipt(control, node, tmp_path):
    destination = tmp_path / "persisted"
    destination.mkdir()
    plan = dict(result_dir=str(destination), run_id="test-run", intent_id="a" * 32, code_sha256="b" * 64)
    result = dict(job_id="123", diagnostic_complete=False, exit_code=1, **dict.fromkeys(control.FLAGS, False))
    node.persist(plan, control, None, None, result, {"initial": {"passed": False}})
    receipt = json.loads((destination / "receipt.json").read_text())
    assert {row["path"] for row in receipt["files"]} == {"node-result.json", "memory.json"}
    assert receipt["passed"] is False


def test_early_failure_readback_must_succeed_before_receipt(control, node, tmp_path, monkeypatch):
    destination = tmp_path / "persisted"
    destination.mkdir()
    plan = dict(result_dir=str(destination), run_id="test-run", intent_id="a" * 32, code_sha256="b" * 64)
    result = dict(job_id="123", diagnostic_complete=False, exit_code=1, **dict.fromkeys(control.FLAGS, False))
    monkeypatch.setattr(control, "read", lambda *args: b"changed")
    with pytest.raises(ValueError, match="readback"):
        node.persist(plan, control, None, None, result, {})
    assert not (destination / "receipt.json").exists()
