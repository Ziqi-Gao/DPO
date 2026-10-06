"""New execution guards precede the unmodified preparation node and keep honest exits."""

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


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    node = load("sdsc_student_name_invariant_recovery_job")
    calls = []
    source = ROOT / node.ORIGINAL_NAME

    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    def canonical(value):
        return json.dumps(value, sort_keys=True).encode()

    inner = dict(
        submission_dir=str(tmp_path),
        control_sha256={node.ORIGINAL_NAME: digest(source.read_bytes())},
    )
    plan = dict(science_plan=inner)
    (tmp_path / "plan.json").write_bytes(canonical(inner))

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def write_once(path, raw):
        with path.open("xb") as stream:
            stream.write(raw)

    def original_main(args):
        calls.append(("original_main", args))
        assert (tmp_path / "recovery-node-entry.json").is_file()
        return 0

    original = SimpleNamespace(
        __file__=str(source),
        allocation=lambda p, c: dict(job_id="12345"),
        main=original_main,
    )
    control = SimpleNamespace(
        science=object(),
        helper=lambda name: original,
        sha=digest,
        canonical=canonical,
        require=require,
        read=lambda p: Path(p).read_bytes(),
        safe=Path,
        now=lambda: "2026-10-06T00:00:00Z",
        verify_source=lambda p: calls.append("source"),
        check_claims=lambda p: calls.append("claims"),
        verify_execution=lambda p: calls.append("execution"),
        verify_fence=lambda p: calls.append("fence"),
        entry_record=lambda p, job, **kw: dict(job_id=job, **kw),
        exit_record=lambda entry, **kw: dict(entry, **kw),
        write_once=write_once,
    )
    monkeypatch.setattr(node, "load_control", lambda *a: (control, plan))
    return SimpleNamespace(
        node=node, control=control, plan=plan, original=original, calls=calls, root=tmp_path
    )


def test_guards_then_unchanged_node_arguments_and_environment(fixture, monkeypatch):
    f = fixture
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture-first,GPU-fixture-second")
    before = dict(os.environ)
    assert f.node.main(["outer", "digest"]) == 0
    assert f.calls[:4] == ["source", "claims", "execution", "fence"]
    assert f.calls[4] == (
        "original_main",
        [str(f.root / "plan.json"), f.control.sha(f.control.canonical(f.plan["science_plan"]))],
    )
    assert dict(os.environ) == before
    assert not (f.root / "result").exists()
    record = json.loads((f.root / "recovery-node-exit.json").read_bytes())
    assert record["returncode"] == 0 and record["node_returned"] is True


@pytest.mark.parametrize("guard", ["verify_source", "check_claims", "verify_execution", "verify_fence"])
def test_failed_guard_never_calls_scientific_node(fixture, guard):
    f = fixture

    def fail(_):
        raise ValueError(guard)

    setattr(f.control, guard, fail)
    with pytest.raises(ValueError, match=guard):
        f.node.main(["outer", "digest"])
    assert not any(isinstance(x, tuple) for x in f.calls)
    assert not (f.root / "recovery-node-entry.json").exists()


def test_changed_original_node_rejected(fixture):
    f = fixture
    f.plan["science_plan"]["control_sha256"][f.node.ORIGINAL_NAME] = "0" * 64
    with pytest.raises(ValueError, match="original preparation node changed"):
        f.node.main(["outer", "digest"])
    assert not f.calls


@pytest.mark.parametrize("moments", [(0, 61), (0, 59, 61)])
def test_setup_budget_before_and_after_durable_entry(fixture, monkeypatch, moments):
    f = fixture
    clock = iter(moments)
    monkeypatch.setattr(f.node.time, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError, match="setup|persistence"):
        f.node.main(["outer", "digest"])
    assert not any(isinstance(x, tuple) for x in f.calls)
    if len(moments) == 3:
        exit_record = json.loads((f.root / "recovery-node-exit.json").read_bytes())
        assert exit_record["node_returned"] is False


def test_original_failure_is_preserved(fixture):
    f = fixture
    f.original.main = lambda _: 2
    assert f.node.main(["outer", "digest"]) == 2
    record = json.loads((f.root / "recovery-node-exit.json").read_bytes())
    assert record["returncode"] == 2 and record["node_returned"] is True


@pytest.mark.parametrize("error", [RuntimeError("worker failed"), KeyboardInterrupt()])
def test_original_exception_is_recorded_and_reraised(fixture, error):
    f = fixture

    def fail(_):
        raise error

    f.original.main = fail
    with pytest.raises(type(error)):
        f.node.main(["outer", "digest"])
    record = json.loads((f.root / "recovery-node-exit.json").read_bytes())
    assert record["returncode"] == 1 and record["node_returned"] is False
    assert record["error_type"] == type(error).__name__


def test_preexisting_entry_is_not_overwritten_or_rearmed(fixture):
    f = fixture
    (f.root / "recovery-node-entry.json").write_bytes(b"original attempt")
    with pytest.raises(FileExistsError):
        f.node.main(["outer", "digest"])
    assert (f.root / "recovery-node-entry.json").read_bytes() == b"original attempt"
    assert not any(isinstance(x, tuple) for x in f.calls)


def test_invalid_return_cannot_publish_success(fixture):
    f = fixture
    f.original.main = lambda _: False
    with pytest.raises(ValueError, match="invalid status"):
        f.node.main(["outer", "digest"])
    record = json.loads((f.root / "recovery-node-exit.json").read_bytes())
    assert record["returncode"] == 1 and record["node_returned"] is False


def test_plan_digest_rejected_before_loading_control(tmp_path):
    node = load("sdsc_student_name_invariant_recovery_job")
    plan = tmp_path / "execution-plan.json"
    plan.write_bytes(b"{}")
    with pytest.raises(ValueError, match="plan changed"):
        node.load_control(plan, "0" * 64)


def test_plan_symlink_rejected_before_loading_control(tmp_path):
    node = load("sdsc_student_name_invariant_recovery_job")
    actual = tmp_path / "actual.json"
    actual.write_bytes(b"{}")
    link = tmp_path / "execution-plan.json"
    link.symlink_to(actual)
    with pytest.raises(ValueError, match="invalid recovery plan file"):
        node.load_control(link, hashlib.sha256(b"{}").hexdigest())
