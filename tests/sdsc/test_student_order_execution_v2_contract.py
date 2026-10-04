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
    path = ROOT / "tools/sdsc_student_order_execution_v2_contract.py"
    spec = importlib.util.spec_from_file_location("_test_order_execution_v2_contract", path)
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
        "previous_execution",
        "diagnostic_implementation_commit",
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
        (("execution", "early_probe_timeout_seconds"), 120),
        (
            ("execution", "early_probe_thread_environment"),
            {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "12", "OPENBLAS_NUM_THREADS": "12"},
        ),
        (("execution", "early_probe_stack_dump_interval_seconds"), 60),
        (("execution", "early_probe_importtime"), False),
        (("execution", "retained_startup_log_max_bytes"), 1048577),
        (("execution", "original_resources_walltime_publication_reserves_unchanged"), False),
        (("execution", "frozen_v1_rank_training_wrapper_unchanged"), False),
        (("scope", "diagnostic_proves_thread_policy_necessary"), True),
        (("scope", "same_node_aba_separates_cache_from_order"), True),
        (("previous_execution", "artifact_sha256"), "0" * 64),
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


# Captured immutable receipt bytes only; accounting below is a CPU fixture.
PUBLICATIONS = [
    {
        "code_sha256": "5dc1752d917fe2db885017378618df4110f22616cb1bd8d586748805398a9054",
        "execution_class_certified": False,
        "factorial_ready": False,
        "files": [
            {
                "path": "node-result.json",
                "sha256": "63099dc1c2d40c7cb4cffae46d53ad1511de72a7f316663f8312fa4837b79468",
                "size": 2634,
            },
            {
                "path": "memory.json",
                "sha256": "8dd9bc960376af4540f830ebd311cb0497930be7d80c6cb7f53a5e0dbdd98464",
                "size": 22897,
            },
        ],
        "formal_initial_accepted": False,
        "g0_passed": False,
        "intent_id": "44340f6e52b16b20d59538170b588576",
        "job_id": "54643463",
        "large_files": [],
        "large_files_read_back_verified": True,
        "mode": "preflight",
        "passed": False,
        "persistent_read_back_verified": True,
        "pilot_passed": False,
        "plan_sha256": "34b45d2b888bf7bc6b7a96c72ce5a9b664a080356cfae2bc8f10dbd4eb9a730d",
        "preparation_complete": False,
        "run_id": "20261004T053548Z-5dc1752d917f-d19d7523",
        "stage_complete": False,
        "student_accepted": False,
        "task": "qwen3-v2-student-order-preparation-v4",
    },
    {
        "execution_class_certified": False,
        "execution_core_sha256": "4f745b038209cd5cdcd884e0acf3cf76b136e4699a41d6c6af6af62ef4aec06e",
        "factorial_ready": False,
        "files": [
            {
                "path": "execution-node.json",
                "sha256": "0e5e39afdda0ea4238872ede638eabaa17415aaf4f4e452c6331d71b77a983ce",
                "size": 1056,
            },
            {
                "path": "startup.log",
                "sha256": "3de24a860f6712ccb2f965a185b5a45fbfd0c890b9a4abfd3201ad1367d315fd",
                "size": 1083,
            },
        ],
        "formal_initial_accepted": False,
        "g0_passed": False,
        "job_id": "54643463",
        "persistent_read_back_verified": True,
        "pilot_passed": False,
        "plan_sha256": "73478b7e5a45324a4f957d3ba60fe2f64053c75cb8ef63cb3db8ff04daf94c07",
        "schema": "quest-sdsc-student-order-execution-publication-v1",
        "science_plan_sha256": "34b45d2b888bf7bc6b7a96c72ce5a9b664a080356cfae2bc8f10dbd4eb9a730d",
        "startup_passed": False,
        "student_accepted": False,
        "task": "qwen3-v2-student-order-execution-v1",
    },
    {
        "code_sha256": "919ed27483c76e8acdf993b30b2cc8f32c7a3e04c4360e744938b0b231999501",
        "cuda_observed": False,
        "diagnostic_complete": True,
        "execution_class_certified": False,
        "factorial_ready": False,
        "files": [
            {
                "path": "import-observations.json",
                "sha256": "9b1fc5ed109db6725b8d094333809c10de535ab88adbc01422780899ab62d1a9",
                "size": 5206,
            },
            {
                "path": "A1-report.json",
                "sha256": "d73d480e4c6d04986747fa5f2abc4c9d9a1907acc44f674b6ccf6f07fa6837d4",
                "size": 4106,
            },
            {
                "path": "A1.log",
                "sha256": "62352ffbe6534a46ce72db89510d2b2876b6e08e85c7910040f0da079f449d86",
                "size": 89298,
            },
            {
                "path": "A1-proc.jsonl",
                "sha256": "50a469e9a0db721bffe1daf01d511517879460a8e48fa651e8c720832452285e",
                "size": 31473,
            },
            {
                "path": "B-report.json",
                "sha256": "d5979d5f17b225a38213eda15292ab63488269a64a62b4fb6ba4194c2598edad",
                "size": 4099,
            },
            {
                "path": "B.log",
                "sha256": "02209c613d1e8dc2ab81e7ecb5ff7d39670c3947a1e59e10a434ecfbc600ab22",
                "size": 80614,
            },
            {
                "path": "B-proc.jsonl",
                "sha256": "c6486efbfe4e374511e49213150eed7eadd0f261c7c7c71505e1b83a1b3343d3",
                "size": 2368,
            },
            {
                "path": "A2-report.json",
                "sha256": "c3be483dfbb7bb05ed96f691355a3dc05db8d200135d2cf0dfa82fa8eaea9db5",
                "size": 4107,
            },
            {
                "path": "A2.log",
                "sha256": "fadb217d7f77cf87d04774439e3b02c2188cc42d2dc8625bb1d6580031a7b92a",
                "size": 80619,
            },
            {
                "path": "A2-proc.jsonl",
                "sha256": "075c493dae154be0c660c051708e713c027c2a289753641bb831e38c521fcc52",
                "size": 2369,
            },
            {
                "path": "node-result.json",
                "sha256": "27e167f0ccdac46b0c32cfb1b100d5ade379688210ec5a16a24afe3e5debabd4",
                "size": 1681,
            },
            {
                "path": "memory.json",
                "sha256": "fde24b590b50b42a6580c16ecd718ae15c744d59641140c05228ec27dac6e541",
                "size": 136443,
            },
        ],
        "formal_initial_accepted": False,
        "g0_passed": False,
        "imports_ready": True,
        "intent_id": "2ac800ac6e8e71ccf2e20068cf290a1b",
        "job_id": "54643629",
        "model_loaded": False,
        "persistent_read_back_verified": True,
        "pilot_passed": False,
        "plan_sha256": "570e39613cff38031375ebbdea149adbc197dd17ad51be8fc74abeb2bd32f21d",
        "run_id": "20261004T062152Z-919ed27483c7-293d2788",
        "student_accepted": False,
        "task": "sdsc-torch-import-probe-v1",
        "training_started": False,
    },
]


@pytest.fixture
def statuses(c):
    values = []
    for frozen, failed, publication in (
        (c.FAILED, True, PUBLICATIONS[0]),
        (c.DIAGNOSTIC, False, PUBLICATIONS[2]),
    ):
        job = frozen["job_id"]
        rows = [
            "|".join(
                [
                    job + suffix,
                    "COMPLETED" if not failed or suffix == ".extern" else "FAILED",
                    "0:0" if not failed or suffix == ".extern" else "1:0",
                ]
            )
            for suffix in ("", ".batch", ".extern")
        ]
        values.append(
            dict(
                job_id=job,
                state="FAILED" if failed else "COMPLETED",
                accounting_complete=not failed,
                success=not failed,
                diagnostic_complete=not failed,
                imports_ready=not failed,
                queue=dict(returncode=0, stdout=""),
                accounting=dict(returncode=0, stdout="\n".join(rows)),
                publication_verified=True,
                artifact_hashes_verified=True,
                publication_sha256=frozen["publication_sha256"],
                publication=copy.deepcopy(publication),
                **dict.fromkeys(c.FLAGS, False),
            )
        )
    values[1]["terminal"] = True
    values[0].update(
        execution_publication_verified=True,
        execution_publication_sha256=c.FAILED["execution_publication_sha256"],
        execution_publication=copy.deepcopy(PUBLICATIONS[1]),
        startup_passed=False,
        stage_complete=False,
        preparation_complete=False,
        scientific_stage_complete=False,
    )
    return values


def test_actual_receipt_graph_is_infrastructure_only(c, statuses):
    assert c.validate_prerequisites(*statuses) is None
    assert "terminal" not in statuses[0] and statuses[1]["terminal"] is True
    assert all(status[key] is False for status in statuses for key in c.FLAGS)


@pytest.mark.parametrize(
    "index,path,value",
    [
        (0, ("job_id",), "54623814"),
        (0, ("success",), True),
        (0, ("accounting_complete",), True),
        (0, ("queue", "stdout"), "54643463|RUNNING"),
        (1, ("queue", "returncode"), False),
        (1, ("accounting", "returncode"), False),
        (1, ("accounting_complete",), 1),
        (1, ("success",), 1),
        (1, ("terminal",), False),
        (1, ("terminal",), 1),
        (1, ("terminal",), None),
        (1, ("imports_ready",), False),
        (1, ("diagnostic_complete",), False),
        (0, ("scientific_stage_complete",), True),
        (0, ("startup_passed",), True),
        (0, ("publication_verified",), False),
        (0, ("execution_publication_verified",), False),
        (1, ("artifact_hashes_verified",), False),
        (0, ("execution_publication_sha256",), "0" * 64),
        (1, ("publication_sha256",), "0" * 64),
        (1, ("student_accepted",), True),
        (1, ("publication", "imports_ready"), False),
        (1, ("publication", "training_started"), True),
        (0, ("publication", "large_files"), [{"path": "checkpoint.pt"}]),
        (0, ("execution_publication", "science_plan_sha256"), "0" * 64),
        (0, ("execution_publication", "execution_core_sha256"), "0" * 64),
        (0, ("execution_publication", "plan_sha256"), "0" * 64),
    ],
)
def test_status_and_publication_mutations_reject(c, statuses, index, path, value):
    target = statuses[index]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        c.validate_prerequisites(*statuses)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "key",
    [
        "job_id",
        "queue",
        "accounting",
        "publication",
        "publication_verified",
        "publication_sha256",
        "artifact_hashes_verified",
        "success",
        "state",
        "accounting_complete",
    ],
)
def test_missing_producer_fields_fail_closed(c, statuses, index, key):
    statuses[index].pop(key)
    with pytest.raises(ValueError):
        c.validate_prerequisites(*statuses)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("suffix", ["partial", "wrongexit", "wrongjob", "duplicate"])
def test_terminal_accounting_is_not_just_a_status_label(c, statuses, index, suffix):
    rows = statuses[index]["accounting"]["stdout"].splitlines()
    if suffix == "partial":
        rows.pop()
    elif suffix == "wrongexit":
        rows[0] = rows[0].rsplit("|", 1)[0] + "|2:0"
    elif suffix == "wrongjob":
        rows[0] = "123|COMPLETED|0:0"
    else:
        rows[2] = rows[1]
    statuses[index]["accounting"]["stdout"] = "\n".join(rows)
    with pytest.raises(ValueError):
        c.validate_prerequisites(*statuses)


def test_frozen_dependency_bytes_and_incident_receipt_hashes(c):
    assert len(c.FROZEN_DEPENDENCIES) == 11
    assert len(c.CONTROL_PATHS) == 4 and c.CONTROL_PATHS[-1].endswith("_v2_probe.py")
    for path, digest in c.FROZEN_DEPENDENCIES.items():
        assert c.sha((ROOT / path).read_bytes()) == digest
    assert (
        c.sha((ROOT / c.PREVIOUS_EXECUTION["contract_path"]).read_bytes())
        == c.PREVIOUS_EXECUTION["artifact_sha256"]
    )
    assert [c.sha(c.canonical(p)) for p in PUBLICATIONS] == [
        c.FAILED["publication_sha256"],
        c.FAILED["execution_publication_sha256"],
        c.DIAGNOSTIC["publication_sha256"],
    ]


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
    previous_path = root / "prior-contract.json"
    previous_path.write_text("accepted historical execution artifact\n")
    git("add", "prior-contract.json")
    git("commit", "--quiet", "-m", "Historical execution and diagnostic fixture")
    previous = copy.deepcopy(c.PREVIOUS_EXECUTION)
    previous.update(
        contract_path="prior-contract.json",
        artifact_sha256=c.sha(previous_path.read_bytes()),
        acceptance_commit=git("rev-parse", "HEAD"),
    )
    monkeypatch.setattr(c, "PREVIOUS_EXECUTION", previous)
    monkeypatch.setattr(c, "DIAGNOSTIC_IMPLEMENTATION_COMMIT", git("rev-parse", "HEAD"))
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
    [
        "dirty",
        "staged",
        "committed",
        "symlink",
        "dependency",
        "priorartifact",
        "secondreview",
        "imaginarycommit",
        "wronghead",
    ],
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
        elif mutation == "priorartifact":
            path = root / "prior-contract.json"
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
