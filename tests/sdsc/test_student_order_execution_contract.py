"""Execution review and incident admission only; never GPU/scientific acceptance."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def c():
    path = ROOT / "tools/sdsc_student_order_execution_contract.py"
    spec = importlib.util.spec_from_file_location("_test_order_execution_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_proposed_artifact_is_exact_and_cannot_admit_execution(c):
    raw = (ROOT / c.CONTRACT_PATH).read_bytes()
    payload = c.load_execution_contract(raw)
    payload["review"] = copy.deepcopy(c.REVIEW_PROPOSED)
    assert payload == c.proposed_execution_contract()
    with pytest.raises(ValueError, match="not accepted"):
        c.validate_execution_contract(payload, require_accepted=True)
    assert set(payload) == {
        "schema",
        "contract_id",
        "parent_science",
        "failed_preflight",
        "diagnostic",
        "controls",
        "frozen_dependencies",
        "scope",
        "execution",
        "claims",
        "review",
    }
    assert not {"data", "training", "development", "prepared_initial"} & payload.keys()
    assert all(payload["scope"][key] is False for key in c.FLAGS)


@pytest.mark.parametrize(
    "path,value",
    [
        (("parent_science", "artifact_sha256"), "0" * 64),
        (("failed_preflight", "job_id"), "54615110"),
        (("diagnostic", "job_id"), "54345483"),
        (("scope", "student_accepted"), True),
        (("scope", "scientific_change"), 0),
        (("execution", "early_probe_timeout_seconds"), 121),
        (("claims", "maximum_preflight_submissions"), True),
        (("claims", "original_v4_fit_scientific_claim_also_required"), False),
    ],
)
def test_no_numeric_scope_lineage_or_retry_expansion(c, path, value):
    payload = c.proposed_execution_contract()
    payload[path[0]][path[1]] = value
    with pytest.raises(ValueError, match="differs"):
        c.validate_execution_contract(payload)


def test_json_duplicate_keys_and_bounds_fail_closed(c):
    with pytest.raises(ValueError):
        c.load_execution_contract(b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError):
        c.load_execution_contract(b" " * (c.MAX_BYTES + 1))


@pytest.fixture
def statuses(c):
    # Synthetic lifecycle evidence; no GPU observation or remote success claim.
    values = []
    for frozen, failed in ((c.FAILED, True), (c.DIAGNOSTIC, False)):
        job = frozen["job_id"]
        state = "FAILED" if failed else "COMPLETED"
        rows = []
        for suffix in ("", ".batch", ".extern"):
            completed = not failed or suffix == ".extern"
            rows.append(
                "|".join(
                    [job + suffix, "COMPLETED" if completed else "FAILED", "0:0" if completed else "1:0"]
                )
            )
        records = [
            {
                "path": "prepare-report.json" if failed else "cuda-diagnostic.json",
                "sha256": frozen["report_sha256"],
                "size": 100,
            }
        ]
        if not failed:
            records.extend(
                {"path": name + ".json", "sha256": frozen[key + "_sha256"], "size": 100}
                for name, key in (("node-result", "node"), ("memory", "memory"))
            )
        publication = dict(
            job_id=job, plan_sha256=frozen["plan_sha256"], files=records, **dict.fromkeys(c.FLAGS, False)
        )
        if failed:
            publication.update(large_files=[], passed=False)
        status = dict(
            job_id=job,
            state=state,
            terminal=True,
            accounting_complete=not failed,
            success=not failed,
            diagnostic_complete=not failed,
            cuda_ready=not failed,
            queue=dict(returncode=0, stdout=""),
            accounting=dict(returncode=0, stdout="\n".join(rows)),
            publication_verified=True,
            artifact_hashes_verified=True,
            publication_sha256=frozen["publication_sha256"],
            publication=publication,
            **dict.fromkeys(c.FLAGS, False),
        )
        values.append(status)
    return values


def test_complete_diagnostic_is_infrastructure_only(c, statuses):
    failed, probe = statuses
    assert c.validate_prerequisites(failed, probe) is None
    assert all(failed[k] is probe[k] is False for k in c.FLAGS)


@pytest.mark.parametrize(
    "mutation",
    [
        "queued",
        "partialaccounting",
        "success",
        "ready",
        "scope",
        "wrongjob",
        "wrongplan",
        "wrongreport",
        "wrongmemory",
        "duplicatefile",
        "failedscientificfit",
        "oldcheckpoint",
        "stepfailure",
    ],
)
def test_prerequisite_failures_are_not_automatic_retry(c, statuses, mutation):
    failed, probe = statuses
    if mutation == "queued":
        probe["queue"]["stdout"] = "54626913|COMPLETED"
    elif mutation == "partialaccounting":
        probe["accounting_complete"] = False
    elif mutation == "success":
        probe["success"] = 1
    elif mutation == "ready":
        probe["cuda_ready"] = False
    elif mutation == "scope":
        probe["student_accepted"] = True
    elif mutation == "wrongjob":
        probe["job_id"] = "54345483"
    elif mutation == "wrongplan":
        probe["publication"]["plan_sha256"] = "0" * 64
    elif mutation == "wrongreport":
        probe["publication"]["files"][0]["sha256"] = "0" * 64
    elif mutation == "wrongmemory":
        probe["publication"]["files"][2]["sha256"] = "0" * 64
    elif mutation == "duplicatefile":
        probe["publication"]["files"].append(copy.deepcopy(probe["publication"]["files"][0]))
    elif mutation == "failedscientificfit":
        failed["job_id"] = "54615110"
    elif mutation == "oldcheckpoint":
        failed["publication"]["large_files"] = [{"path": "checkpoints/step4.pt"}]
    else:
        probe["accounting"]["stdout"] = probe["accounting"]["stdout"].replace(
            "COMPLETED|0:0", "FAILED|1:0", 1
        )
    with pytest.raises(ValueError):
        c.validate_prerequisites(failed, probe)


@pytest.fixture
def reviewed(c, tmp_path, monkeypatch, request):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(root), *args],
            env=c._git_environment(),
            capture_output=True,
            check=True,
        )
        return result.stdout.decode().strip()

    git("init", "--quiet")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Execution review fixture")
    (root / "old-science.txt").write_text("frozen scientific source\n")
    git("add", ".")
    git("commit", "--quiet", "-m", "Parent scientific acceptance fixture")
    parent = copy.deepcopy(c.PARENT)
    parent["acceptance_commit"] = git("rev-parse", "HEAD")
    monkeypatch.setattr(c, "PARENT", parent)
    controls = ("new-control.py", "node-control.py")
    monkeypatch.setattr(c, "CONTROL_PATHS", controls)
    monkeypatch.setattr(
        c, "FROZEN_DEPENDENCIES", {"old-science.txt": c.sha((root / "old-science.txt").read_bytes())}
    )

    # Exercise real execution Git rules; scientific resolver is independently covered
    # by immutable V4 tests and remains invoked in production in a fresh process.
    def science_binding(source, metadata, head):
        assert source == root and metadata == root / ".git"
        return dict(
            protocol_sha256=parent["protocol_sha256"],
            artifact_sha256=parent["artifact_sha256"],
            implementation_commit=parent["implementation_commit"],
            acceptance_commit=parent["acceptance_commit"],
            head=head,
            science_file_sha256={"old-science.txt": c.sha((root / "old-science.txt").read_bytes())},
        )

    monkeypatch.setattr(c, "_parent_binding", science_binding)
    for p in controls:
        (root / p).write_text("# fixed execution implementation\n")
    path = root / c.CONTRACT_PATH
    path.parent.mkdir(parents=True)
    payload = c.proposed_execution_contract()
    path.write_bytes(c.canonical(payload))
    git("add", ".")
    git("commit", "--quiet", "-m", "Proposed execution implementation")
    impl = git("rev-parse", "HEAD")
    payload["review"] = dict(
        status="accepted",
        reviewed_implementation_commit=impl,
        reviewer="independent fixture reviewer",
        reviewed_at_utc="2026-10-03T18:00:00Z",
        rationale="Execution-only fixture review",
    )
    path.write_bytes(c.canonical(payload))
    git("add", c.CONTRACT_PATH)
    if getattr(request, "param", False):
        (root / "new-control.py").write_text("# unreviewed code inside acceptance\n")
        git("add", "new-control.py")
    git("commit", "--quiet", "-m", "Review only")
    return root, git, impl


def test_distinct_real_review_and_descendant_head(c, reviewed):
    root, git, impl = reviewed
    bound = c.resolve_execution_contract(root)
    assert bound.implementation_commit == impl and bound.acceptance_commit == git("rev-parse", "HEAD")
    assert set(bound.as_dict()) == {
        "core_sha256",
        "artifact_sha256",
        "implementation_commit",
        "acceptance_commit",
        "head",
        "control_file_sha256",
        "science_binding",
    }
    (root / "unrelated.md").write_text("later unrelated documentation\n")
    git("add", "unrelated.md")
    git("commit", "--quiet", "-m", "Unrelated descendant")
    later = c.resolve_execution_contract(root)
    assert later.acceptance_commit == bound.acceptance_commit and later.head != bound.head


@pytest.mark.parametrize(
    "mutation",
    ["dirty", "staged", "committed", "symlink", "dependency", "secondreview", "imaginarycommit", "wronghead"],
)
def test_unreviewed_controls_and_forged_reviews_fail_closed(c, reviewed, mutation):
    root, git, _ = reviewed
    path = root / "new-control.py"
    if mutation == "wronghead":
        with pytest.raises(ValueError, match="HEAD"):
            c.resolve_execution_contract(root, expected_head="0" * 40)
        return
    if mutation in {"secondreview", "imaginarycommit"}:
        path = root / c.CONTRACT_PATH
        value = json.loads(path.read_bytes())
        if mutation == "secondreview":
            value["review"]["rationale"] += " changed"
        else:
            value["review"]["reviewed_implementation_commit"] = "0" * 40
        path.write_bytes(c.canonical(value))
        if mutation == "secondreview":
            git("add", c.CONTRACT_PATH)
            git("commit", "--quiet", "-m", "Second unreviewed acceptance")
    elif mutation == "symlink":
        path.unlink()
        path.symlink_to(root / "node-control.py")
    else:
        if mutation == "dependency":
            path = root / "old-science.txt"
        path.write_text("unreviewed delta\n")
        if mutation in {"staged", "committed"}:
            git("add", path.name)
        if mutation == "committed":
            git("commit", "--quiet", "-m", "Unreviewed implementation")
    with pytest.raises((ValueError, OSError)):
        c.resolve_execution_contract(root)


@pytest.mark.parametrize("reviewed", [True], indirect=True)
def test_acceptance_cannot_include_implementation_changes(c, reviewed):
    root, _, _ = reviewed
    with pytest.raises(ValueError, match="non-review"):
        c.resolve_execution_contract(root)
