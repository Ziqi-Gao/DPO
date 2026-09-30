"""One existing worker, bounded wait and observable completed-arm evidence."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_wait_exposes_completed_arm_once_without_restarting_worker(tmp_path, monkeypatch, capsys):
    node, control = module("sdsc_student_quality_job"), module("sdsc_student_quality")
    clock, waits = [0.0], []
    monkeypatch.setattr(node, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    arm = dict(
        checkpoint="initial", cohort="validation_exposed", cap=128,
        metrics={"count": 128, "validation_accuracy": 0.0},
        teacher_forced={"canonical_response_nll": 1.25},
    )

    class Child:
        args = ("existing-worker",)

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 3:
                return 0
            clock[0] += timeout
            (tmp_path / "progress.json").write_text(json.dumps({"arms": [arm]}))
            raise subprocess.TimeoutExpired(self.args, timeout)

    node.wait_worker(Child(), timeout=100, artifacts=tmp_path, control=control)
    assert waits == [30, 30, 30]
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == dict(phase="quality_progress", completed_arms=1, **arm)


def test_worker_wait_uses_one_total_deadline(tmp_path, monkeypatch):
    node, control = module("sdsc_student_quality_job"), module("sdsc_student_quality")
    clock, waits = [0.0], []
    monkeypatch.setattr(node, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Child:
        args = ("existing-worker",)

        def wait(self, timeout):
            waits.append(timeout)
            clock[0] += timeout
            raise subprocess.TimeoutExpired(self.args, timeout)

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        node.wait_worker(Child(), timeout=35, artifacts=tmp_path, control=control)
    assert waits == [30, 5]
    assert caught.value.timeout == 35
    assert caught.value.cmd == ("existing-worker",)


def test_progress_never_accepts_more_than_reviewed_arms(tmp_path, monkeypatch):
    node, control = module("sdsc_student_quality_job"), module("sdsc_student_quality")
    monkeypatch.setattr(node, "time", SimpleNamespace(monotonic=lambda: 0.0))
    (tmp_path / "progress.json").write_text(json.dumps({"arms": [{}] * 13}))

    class Child:
        args = ("existing-worker",)

        def wait(self, timeout):
            raise subprocess.TimeoutExpired(self.args, timeout)

    with pytest.raises(ValueError, match="invalid diagnostic progress"):
        node.wait_worker(Child(), timeout=60, artifacts=tmp_path, control=control)
