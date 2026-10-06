"""Real Git acceptance boundaries with a small inventory and stubbed parent science.

Complete 203-file science and v2 restoration need the separate activation check.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def contract():
    name = "_independent_name_invariant_recovery_contract"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools/sdsc_student_name_invariant_recovery_contract.py"
    )
    c = importlib.util.module_from_spec(spec)
    sys.modules[name] = c
    spec.loader.exec_module(c)
    return c


@pytest.fixture
def history(tmp_path, monkeypatch, contract):
    c = contract
    root = tmp_path / "repo"
    root.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_AUTHOR_NAME="Fixture",
        GIT_AUTHOR_EMAIL="fixture@example.invalid",
        GIT_COMMITTER_NAME="Fixture",
        GIT_COMMITTER_EMAIL="fixture@example.invalid",
        LC_ALL="C",
    )

    def git(*args, no_replace=True):
        e = dict(env)
        if no_replace:
            e["GIT_NO_REPLACE_OBJECTS"] = "1"
        return (
            subprocess.run(
                ["/usr/bin/git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
                capture_output=True,
                env=e,
                timeout=10,
                check=True,
            )
            .stdout.decode()
            .strip()
        )

    def write(name, raw):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)

    def commit(message):
        git("add", "--all")
        git("commit", "--quiet", "-m", message)
        return git("rev-parse", "HEAD")

    git("init", "--quiet")
    controls = ("tools/execution.py", "tools/guard.py")
    frozen = {"science/frozen.py": c.sha(b"frozen science\n")}
    for p in controls:
        write(p, (p + "\n").encode())
    write("science/frozen.py", b"frozen science\n")
    parent = commit("accepted parent fixture")
    monkeypatch.setattr(c, "CONTROL_PATHS", controls)
    monkeypatch.setattr(c, "FROZEN_DEPENDENCIES", frozen)
    monkeypatch.setattr(c, "PARENT", dict(c.PARENT, acceptance_commit=parent))
    monkeypatch.setattr(
        c,
        "_parent_binding",
        lambda source, metadata, head: dict(head=head, science_file_sha256=dict(frozen), fixture_only=True),
    )
    write(c.CONTRACT_PATH, c.canonical(c.proposed_execution_contract()))
    implementation = commit("proposed implementation")

    def accept(extra=False):
        v = c.proposed_execution_contract()
        v["review"] = dict(
            status="accepted",
            reviewed_implementation_commit=implementation,
            reviewer="Independent fixture",
            reviewed_at_utc="2026-10-06T05:00:00Z",
            rationale="Boundary test only.",
        )
        write(c.CONTRACT_PATH, c.canonical(v))
        if extra:
            write("docs/unreviewed.md", b"not review\n")
        return commit("distinct acceptance")

    def resolve(**kw):
        return c.resolve_execution_contract(root, expected_head=git("rev-parse", "HEAD"), **kw)

    return SimpleNamespace(
        c=c,
        root=root,
        git=git,
        write=write,
        commit=commit,
        parent=parent,
        implementation=implementation,
        accept=accept,
        resolve=resolve,
    )


def test_checked_in_scope(contract):
    c = contract
    value = c.load_execution_contract((ROOT / c.CONTRACT_PATH).read_bytes())
    value["review"] = c.REVIEW_PROPOSED
    assert value == c.proposed_execution_contract()
    assert len(value["frozen_dependencies"]) == 29 and len(value["controls"]) == 5
    assert value["scope"]["stage"] == "preflight_only"
    assert value["execution"]["optimizer_steps"] == 4
    assert value["execution"]["unresolved_old_gpu_reservation"] == 2
    assert value["execution"]["wrapper_setup_is_outside_original_main_timer"] is True
    with pytest.raises(c.ExecutionContractError, match="not accepted"):
        c.load_execution_contract(c.canonical(value), require_accepted=True)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("scope", "scientific_change", True),
        ("scope", "stage", "fit"),
        ("scope", "automatic_retry", True),
        ("execution", "gpu_count", 4),
        ("execution", "worker_deadline_seconds", 3600),
        ("execution", "optimizer_steps", 32),
        ("execution", "unresolved_old_gpu_reservation", 0),
        ("fence", "only_this_operation_winning_atomic_mkdir_is_success", False),
        ("claims", "maximum_recovery_submissions", 2),
        ("parent_science", "acceptance_commit", "0" * 40),
        ("old_unknown_submission", "plan_sha256", "0" * 64),
    ],
)
def test_scope_mutation_rejected(contract, section, key, value):
    v = contract.proposed_execution_contract()
    v[section][key] = value
    with pytest.raises(contract.ExecutionContractError, match="differs"):
        contract.validate_execution_contract(v)


@pytest.mark.parametrize("action", ["change", "remove", "add"])
def test_frozen_inventory_mutation_rejected(contract, action):
    v = contract.proposed_execution_contract()
    frozen = v["frozen_dependencies"]
    first = next(iter(frozen))
    if action == "change":
        frozen[first] = "0" * 64
    elif action == "remove":
        del frozen[first]
    else:
        frozen["tools/unreviewed.py"] = "0" * 64
    with pytest.raises(contract.ExecutionContractError, match="differs"):
        contract.validate_execution_contract(v)


def test_duplicate_json_rejected(contract):
    with pytest.raises(contract.ExecutionContractError, match="invalid execution JSON"):
        contract.load_execution_contract(b'{"schema":"one","schema":"two"}')


def test_proposed_cannot_execute(history):
    h = history
    with pytest.raises(h.c.ExecutionContractError, match="not accepted"):
        h.resolve()
    r = h.resolve(require_accepted=False)
    assert r.implementation_commit is None and r.acceptance_commit is None


def test_genuine_distinct_acceptance(history):
    h = history
    accepted = h.accept()
    r = h.resolve()
    assert r.implementation_commit == h.implementation and r.acceptance_commit == accepted
    assert r.head == accepted and set(r.control_file_sha256) == set(h.c.CONTROL_PATHS)
    assert r.science_binding["fixture_only"] is True
    copied = r.as_dict()
    copied["control_file_sha256"].clear()
    assert r.control_file_sha256


def test_later_noncontrol_commit_allowed(history):
    h = history
    accepted = h.accept()
    h.write("docs/later.md", b"note\n")
    head = h.commit("documentation")
    r = h.resolve()
    assert r.acceptance_commit == accepted and r.head == head


@pytest.mark.parametrize("place", ["worktree", "staged_only", "head_with_restored_worktree"])
def test_control_delta_rejected(history, place):
    h = history
    h.accept()
    p = h.c.CONTROL_PATHS[0]
    original = (h.root / p).read_bytes()
    h.write(p, b"changed\n")
    if place == "staged_only":
        h.git("add", "--", p)
        h.write(p, original)
    elif place == "head_with_restored_worktree":
        h.commit("unreviewed code")
        h.write(p, original)
    with pytest.raises(h.c.ExecutionContractError, match="staged changes|differs from reviewed"):
        h.resolve()


def test_frozen_delta_rejected(history):
    h = history
    h.accept()
    h.write("science/frozen.py", b"changed\n")
    with pytest.raises(h.c.ExecutionContractError, match="frozen execution dependency changed"):
        h.resolve()


def test_acceptance_nonreview_change_rejected(history):
    h = history
    h.accept(extra=True)
    with pytest.raises(h.c.ExecutionContractError, match="non-review changes"):
        h.resolve()


def test_second_review_touch_rejected(history):
    h = history
    h.accept()
    p = h.root / h.c.CONTRACT_PATH
    v = h.c.load_execution_contract(p.read_bytes(), require_accepted=True)
    v["review"]["rationale"] = "Second review mutation."
    h.write(h.c.CONTRACT_PATH, h.c.canonical(v))
    h.commit("second review")
    with pytest.raises(h.c.ExecutionContractError, match="exactly one later execution review"):
        h.resolve()


def test_self_review_and_wrong_head_rejected(history):
    h = history
    accepted = h.accept()
    with pytest.raises(h.c.ExecutionContractError, match="HEAD differs"):
        h.c.resolve_execution_contract(h.root, expected_head=h.implementation)
    v = h.c.load_execution_contract((h.root / h.c.CONTRACT_PATH).read_bytes(), require_accepted=True)
    v["review"]["reviewed_implementation_commit"] = accepted
    h.write(h.c.CONTRACT_PATH, h.c.canonical(v))
    with pytest.raises(h.c.ExecutionContractError, match="distinct ancestor"):
        h.resolve()


def test_real_replace_cannot_hide_changed_head(history):
    h = history
    accepted = h.accept()
    p = h.c.CONTROL_PATHS[0]
    original = (h.root / p).read_bytes()
    h.write(p, b"genuine unreviewed code\n")
    bad = h.commit("unreviewed execution")
    h.write(p, original)
    h.git("replace", bad, accepted)
    assert h.git("show", f"{bad}:{p}", no_replace=False) == original.decode().strip()
    assert h.git("show", f"{bad}:{p}") == "genuine unreviewed code"
    with pytest.raises(h.c.ExecutionContractError, match="differs from reviewed"):
        h.resolve()


def test_shallow_rejected(history):
    h = history
    h.accept()
    (h.root / ".git/shallow").write_text(h.implementation + "\n")
    with pytest.raises(h.c.ExecutionContractError, match="complete execution review history"):
        h.resolve()


def test_control_symlink_rejected(history):
    h = history
    h.accept()
    p = h.root / h.c.CONTROL_PATHS[0]
    raw = p.read_bytes()
    p.unlink()
    target = h.root / "copy.py"
    target.write_bytes(raw)
    p.symlink_to(target)
    with pytest.raises(h.c.ExecutionContractError, match="symlink"):
        h.resolve()


def test_git_injection_stripped(contract, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/untrusted")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    env = contract._git_environment()
    assert "GIT_DIR" not in env and "GIT_CONFIG_COUNT" not in env
    assert env["GIT_NO_REPLACE_OBJECTS"] == "1" and env["GIT_OPTIONAL_LOCKS"] == "0"
