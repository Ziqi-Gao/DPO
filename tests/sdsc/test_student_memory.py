"""Real cgroup-file fixtures for bounded student memory evidence and failures."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("student_memory", ROOT / "tools/sdsc_student_memory.py")
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)
GIB = 1024**3


def fixture(tmp_path, *, version=1, gib=384, peak=250):
    proc, root = tmp_path / "cgroup", tmp_path / "sys"
    proc.write_text("7:memory:/slurm/job/step/task\n" if version == 1 else "0::/slurm/job/step/task\n")
    boundary = root / "memory" if version == 1 else root
    names = (("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
             if version == 1 else ("memory.max", "memory.current", "memory.peak"))
    for relative, limit, current, observed in (
        ("slurm/job/step/task", 2**63 - 4096 if version == 1 else "max", 20 * GIB, 50 * GIB),
        ("slurm/job/step", gib * GIB, 21 * GIB, 100 * GIB),
        ("slurm/job", gib * GIB, 22 * GIB, peak * GIB),
    ):
        path = boundary / relative
        path.mkdir(parents=True, exist_ok=True)
        for name, value in zip(names, (limit, current, observed), strict=True):
            (path / name).write_text(str(value))
        (path / "memory.stat").write_text("anon 123\nfile 456\n" if version == 2 else "rss 123\ncache 456\ntotal_rss 234\ntotal_cache 567\n")
        if version == 2:
            (path / "memory.events").write_text("high 0\nmax 0\noom 0\noom_kill 0\n")
        else:
            (path / "memory.oom_control").write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 0\n")
            (path / "memory.failcnt").write_text("0")
    return proc, root, boundary / "slurm/job"


@pytest.mark.parametrize("version", [1, 2])
def test_uses_aggregate_job_peak_and_preserves_counters(tmp_path, version):
    proc, root, job = fixture(tmp_path, version=version)
    actual = memory.memory_envelope(expected_bytes=384 * GIB, proc=proc, root=root)
    assert actual["passed"] and actual["path"] == str(job)
    assert actual["peak_bytes"] == 250 * GIB
    assert actual["minimum_headroom_bytes"] == math.ceil(384 * GIB * .2)
    assert actual["memory_stat"] and len(actual["ancestors"]) == 5
    if version == 1:
        assert actual["memory_failcnt"] == 0 and actual["memory_oom_control"]["oom_kill"] == 0
    else:
        assert actual["memory_events"]["oom_kill"] == 0


@pytest.mark.parametrize("gib,peak", [(192, 160), (384, 308), (384, 383)])
def test_headroom_failure_exposes_actual_peak_before_raising(tmp_path, gib, peak):
    proc, root, _job = fixture(tmp_path, gib=gib, peak=peak)
    with pytest.raises(memory.MemoryEnvelopeError, match="headroom") as caught:
        memory.memory_envelope(expected_bytes=gib * GIB, proc=proc, root=root)
    evidence = caught.value.evidence
    assert evidence["passed"] is False and evidence["peak_bytes"] == peak * GIB
    assert evidence["current_bytes"] == 22 * GIB
    assert evidence["memory_stat"]["total_cache"] == 567
    assert str(peak * GIB) in str(caught.value)


def test_failed_stage_is_published_and_reported_before_exception(tmp_path, monkeypatch):
    proc, root, _job = fixture(tmp_path, peak=310)
    measure = memory.memory_envelope
    monkeypatch.setattr(memory, "memory_envelope", lambda **kw: measure(proc=proc, root=root, **kw))
    report = {}
    with pytest.raises(memory.MemoryEnvelopeError):
        memory.record_stage(report, "after_training", tmp_path, expected_bytes=384 * GIB)
    published = json.loads((tmp_path / "memory-after_training.json").read_text())
    assert published == report["memory_stages"]["after_training"]
    assert published["peak_bytes"] == 310 * GIB and not published["passed"]
    with pytest.raises(ValueError, match="already observed"):
        memory.record_stage(report, "after_training", tmp_path, expected_bytes=384 * GIB)


def test_wrong_reviewed_limit_and_unknown_profile_fail_closed(tmp_path):
    proc, root, _job = fixture(tmp_path, gib=192, peak=120)
    with pytest.raises(memory.MemoryEnvelopeError, match="differs") as caught:
        memory.memory_envelope(expected_bytes=384 * GIB, proc=proc, root=root)
    assert caught.value.evidence["limit_bytes"] == 192 * GIB
    assert memory.expected_bytes(memory.V6_PROTOCOL) == 384 * GIB
    assert memory.expected_bytes(memory.V6_PROTOCOL.replace("_v6", "_v5")) == 192 * GIB
    with pytest.raises(ValueError, match="unrecognized"):
        memory.expected_bytes(memory.V6_PROTOCOL.replace("_v6", "_v7"))


def test_missing_peak_remains_failed_with_partial_hierarchy_evidence(tmp_path):
    proc, root, job = fixture(tmp_path)
    (job / "memory.max_usage_in_bytes").unlink()
    with pytest.raises(memory.MemoryEnvelopeError, match="current and peak") as caught:
        memory.memory_envelope(expected_bytes=384 * GIB, proc=proc, root=root)
    assert caught.value.evidence["ancestors"] and not caught.value.evidence["passed"]
