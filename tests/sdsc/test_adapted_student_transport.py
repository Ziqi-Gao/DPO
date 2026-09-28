"""Adapted-student CLI/remote trust boundaries; no SSH, models or actual jobs."""

import copy
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


client_fixture = module("adapted_client_fixture", Path(__file__).with_name("test_cli.py"))
remote_fixture = module("adapted_remote_fixture", Path(__file__).with_name("test_remote.py"))
cli, remote = client_fixture.cli, remote_fixture.remote
TASK = "qwen3-v2-adapted-preflight"
CALIBRATION = "qwen3-v2-adapted-calibration"


def protocol_fixture():
    return dict(
        head="d" * 40,
        protocol_sha256="e" * 64,
        artifact_sha256="f" * 64,
        implementation_commit="1" * 40,
        acceptance_commit="2" * 40,
        review_status="accepted",
        science_file_sha256={"src/science.py": "3" * 64},
    )


def proof_fixture(request):
    return dict(
        schema="quest-sdsc-adapted-student-prerequisites-v1",
        task=request["task"],
        target={key: request[key] for key in ("run_id", "code_sha256", "intent_id")},
        protocol=protocol_fixture(),
        qualification={"receipt": {"job_id": "54496291"}},
        fit={"receipt": {"job_id": "54494742"}},
        preflight=None,
        selected_checkpoint={
            "adapted_teacher_sha256": remote.STUDENT_FIXED_BINDINGS["adapted_teacher_sha256"]
        },
        acceptance={
            "accepted_teacher_sha256": remote.STUDENT_FIXED_BINDINGS["teacher_acceptance_sha256"],
            "inventory_sha256": remote.STUDENT_FIXED_BINDINGS["teacher_acceptance_inventory_sha256"],
        },
    )


@pytest.fixture
def client(monkeypatch):
    fixture = client_fixture.ClientTests(methodName="runTest")
    fixture.setUp()
    manifest = fixture.record()
    manifest["files"] = [
        {"path": "prereg/amendments/qwen3_adapted_student_calibration_v1.json", "sha256": "f" * 64}
    ]
    cli.write_json(cli.state_root() / "runs/run-test.json", {"state": "deployed", "manifest": manifest})
    args = fixture.submit_args(dry_run=True)
    args.task = TASK
    args.gpus, args.cpus, args.mem_gib, args.time = 2, 24, 192, "01:00:00"
    args.teacher_job_id, args.preflight_job_id = "54496291", None
    args.python = "/verified/runtime/bin/python"
    args.result_root = str(cli.PREFLIGHT_STORAGE / "control-results")
    args.hf_home = str(cli.PREFLIGHT_STORAGE / "cache/huggingface")
    args.storage_confirmed = True
    args.provenance_manifest_sha256 = "c" * 64
    args.provenance_dir = cli.REMOTE_ROOT + "/provenance/" + "c" * 64
    monkeypatch.setattr(cli, "require_master", Mock(side_effect=AssertionError("no SSH in fixture")))
    monkeypatch.setattr(cli, "remote", Mock(side_effect=AssertionError("no remote submission in dry-run")))
    yield fixture, args
    fixture.doCleanups()


@pytest.mark.parametrize("task,limit", [(TASK, "01:00:00"), (CALIBRATION, "02:00:00")])
def test_cli_dryrun_exact_two_gpu_argv_without_remote_operations(client, task, limit):
    _, args = client
    args.task, args.time = task, limit
    args.preflight_job_id = "55500001" if task == CALIBRATION else None
    output = io.StringIO()
    with redirect_stdout(output):
        cli.submit_command(args)
    plan = json.loads(output.getvalue())
    argv = plan["sbatch_argv"]
    assert argv[0:2] == ["sbatch", "--parsable"]
    assert all(
        item in argv
        for item in [
            "--gpus=h100:2",
            "--cpus-per-task=24",
            "--mem=192G",
            "--no-requeue",
            "--export=NONE",
            "--signal=B:TERM@300",
        ]
    )
    assert argv[-10].endswith("/tools/sdsc_student_job.sh")
    assert argv[-2:] == [args.provenance_dir, args.provenance_manifest_sha256]
    assert plan["prerequisites_sha256_is_placeholder"] is True
    assert plan["teacher_job_id"] == "54496291"
    cli.remote.assert_not_called()
    cli.require_master.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("teacher_job_id", "54345715"),
        ("preflight_job_id", "55500001"),
        ("provenance_manifest_sha256", None),
        ("provenance_dir", "/wrong"),
        ("gpus", 4),
        ("cpus", 4),
        ("mem_gib", 16),
        ("time", "01:00:01"),
    ],
)
def test_cli_rejects_wrong_upstream_provenance_or_resources(client, field, value):
    _, args = client
    setattr(args, field, value)
    with pytest.raises(cli.UserError):
        cli.submit_command(args)
    cli.remote.assert_not_called()


def test_cli_calibration_requires_new_matching_preflight_id(client):
    _, args = client
    args.task = CALIBRATION
    for job_id in (None, "54345604", "54496291", "54494742", "123;bad"):
        args.preflight_job_id = job_id
        with pytest.raises(cli.UserError, match="matching"):
            cli.submit_command(args)


def test_cli_receipt_binds_deployed_protocol_and_fixed_accepted_teacher(client):
    _, args = client
    request = dict(run_id=args.run_id, intent_id="a" * 32, teacher_job_id="54496291", preflight_job_id=None)
    receipt = dict(
        request,
        prerequisites_path=cli.REMOTE_ROOT + "/submissions/" + "a" * 32 + "/prerequisites.json",
        prerequisites_sha256="b" * 64,
        student_protocol_sha256="e" * 64,
        student_protocol_artifact_sha256="f" * 64,
        **cli.STUDENT_FIXED_BINDINGS,
    )
    cli.validate_student_receipt(receipt, request)
    for key in (*cli.STUDENT_BINDINGS, "teacher_job_id", "prerequisites_path"):
        changed = dict(receipt)
        # The physical protocol is pinned by the saved deployment manifest;
        # the remote accepted-Git resolver supplies its review-neutral core.
        changed[key] = "invalid" if key == "student_protocol_sha256" else "0" * 64
        with pytest.raises(cli.UserError):
            cli.validate_student_receipt(changed, request)


@pytest.fixture
def submitted(tmp_path, monkeypatch):
    fixture = remote_fixture.RemoteBoundaryTests(methodName="runTest")
    fixture.setUp()
    fixture.enable_preflight()
    monkeypatch.setattr(remote, "TEMPORARY_ROOTS", ())
    request = fixture.request
    request.update(
        task=TASK,
        teacher_job_id="54496291",
        preflight_job_id=None,
        provenance_dir=str(fixture.root / "provenance" / ("c" * 64)),
        provenance_manifest_sha256="c" * 64,
    )
    request["resources"].update(gpus=2, cpus=24, mem_gib=192, time="01:00:00")
    files = {
        name: b"# inert fixture\n"
        for name in (
            "tools/sdsc_student_job.py",
            "tools/sdsc_student_job.sh",
            "tools/sdsc_student_contract.py",
            "tools/sdsc_adapted_training_preflight.py",
            "tools/sdsc_adapted_calibration.py",
            "tools/sdsc_teacher_qualify_contract.py",
            "tools/sdsc_provenance.py",
            "src/posttrain_circuits/artifacts/adapted_student_protocol.py",
            "src/posttrain_circuits/artifacts/adapted_teacher_sft.py",
        )
    }
    remote.upload(request, fixture.archive(files))
    provenance = dict(
        provenance_dir=request["provenance_dir"],
        provenance_manifest_sha256="c" * 64,
        science_git_head="d" * 40,
        bundle_sha256="e" * 64,
    )
    monkeypatch.setattr(remote, "verify_provenance", lambda *args: provenance)
    proof = proof_fixture(request)
    monkeypatch.setattr(remote, "student_prerequisites", lambda *args: proof)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return dict(returncode=0, stdout="55500001;expanse\n", stderr="")

    monkeypatch.setattr(remote, "run", run)
    yield SimpleNamespace(request=request, calls=calls, proof=proof, fixture=fixture)
    fixture.doCleanups()


def test_remote_binds_prerequisites_and_calls_sbatch_only_once(submitted):
    receipt = remote.submit(submitted.request)
    assert receipt["job_id"] == "55500001"
    assert receipt["teacher_job_id"] == "54496291"
    assert {key: receipt[key] for key in remote.STUDENT_BINDINGS} == remote.student_bindings(submitted.proof)
    raw = Path(receipt["prerequisites_path"]).read_bytes()
    assert remote.digest(raw) == receipt["prerequisites_sha256"]
    assert json.loads(raw) == submitted.proof
    assert len(submitted.calls) == 1 and submitted.calls[0][0] == "sbatch"
    with pytest.raises(ValueError, match="Intent already exists"):
        remote.submit(submitted.request)
    assert len(submitted.calls) == 1


def test_failed_student_gate_does_not_claim_or_submit(submitted, monkeypatch):
    def fail(*args):
        raise ValueError("matching preflight is not accepted")

    monkeypatch.setattr(remote, "student_prerequisites", fail)
    with pytest.raises(ValueError, match="matching preflight"):
        remote.submit(submitted.request)
    assert not submitted.calls and not (remote.ROOT / "run-claims").exists()


def test_unknown_submission_retains_claim_and_forbids_retry(submitted, monkeypatch):
    monkeypatch.setattr(
        remote, "run", lambda *args, **kwargs: dict(returncode=1, stdout="", stderr="lost receipt")
    )
    result = remote.submit(submitted.request)
    assert result["state"] == "unknown"
    assert list((remote.ROOT / "run-claims").glob("*.json"))
    with pytest.raises(ValueError, match="Intent already exists"):
        remote.submit(submitted.request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("teacher_job_id", "54345715"),
        ("preflight_job_id", "55500001"),
        ("student_protocol_sha256", "e" * 64),
        ("prerequisites_sha256", "e" * 64),
    ],
)
def test_remote_rejects_caller_supplied_scientific_proofs_before_claim(submitted, field, value):
    submitted.request[field] = value
    with pytest.raises(ValueError):
        remote.submit(submitted.request)
    assert not submitted.calls and not (remote.ROOT / "run-claims").exists()


@pytest.fixture
def prerequisite_boundary(tmp_path, monkeypatch):
    release = tmp_path / "release"
    release.mkdir()
    protocol = protocol_fixture()
    files = [{"path": name, "sha256": value} for name, value in protocol["science_file_sha256"].items()]
    files.append(
        {
            "path": "prereg/amendments/qwen3_adapted_student_calibration_v1.json",
            "sha256": protocol["artifact_sha256"],
        }
    )
    (release / "manifest.json").write_text(json.dumps({"files": files}))
    request = dict(
        task=TASK,
        run_id="new-student",
        intent_id="a" * 32,
        code_sha256="b" * 64,
        teacher_job_id="54496291",
        preflight_job_id=None,
        python="/runtime/bin/python",
        hf_home="/cache",
    )
    provenance = dict(
        provenance_dir="/provenance", provenance_manifest_sha256="c" * 64, science_git_head=protocol["head"]
    )
    contract = SimpleNamespace(
        QUALIFICATION_JOB_ID="54496291",
        FIT_JOB_ID="54494742",
        DENSE_SHA256=remote.STUDENT_FIXED_BINDINGS["adapted_teacher_sha256"],
        ACCEPTANCE_ROOT=tmp_path / "acceptance",
        INVENTORY_SHA256=remote.STUDENT_FIXED_BINDINGS["teacher_acceptance_inventory_sha256"],
        verify_teacher_acceptance=Mock(return_value=proof_fixture(request)["acceptance"]),
        input_plans=Mock(return_value=([{"path": "fixture"}], [{"path": "dataset"}])),
    )
    monkeypatch.setattr(remote, "student_contract", lambda _: contract)
    monkeypatch.setattr(
        remote,
        "teacher_qualify_contract",
        lambda _: SimpleNamespace(
            selected_checkpoint=lambda _: {"adapted_teacher_sha256": contract.DENSE_SHA256}
        ),
    )
    monkeypatch.setattr(
        remote, "run", lambda *args, **kwargs: dict(returncode=0, stdout=json.dumps(protocol), stderr="")
    )
    upstream = {
        "54496291": {
            "receipt": {"job_id": "54496291", "teacher_job_id": "54494742", "run_id": "qualification"}
        },
        "54494742": {"receipt": {"job_id": "54494742", "run_id": "fit"}},
    }
    preflight_receipt = dict(
        job_id="55500001",
        task=TASK,
        run_id="preflight",
        teacher_job_id="54496291",
        science_git_head=protocol["head"],
        student_protocol_sha256=protocol["protocol_sha256"],
        student_protocol_artifact_sha256=protocol["artifact_sha256"],
        python=request["python"],
        hf_home=request["hf_home"],
        **remote.STUDENT_FIXED_BINDINGS,
    )
    upstream["55500001"] = dict(receipt=preflight_receipt, release_manifest={"files": files})
    monkeypatch.setattr(
        remote, "upstream_evidence", lambda root, job, task, role: copy.deepcopy(upstream[job])
    )
    monkeypatch.setattr(remote, "calibration_queue_check", lambda: {"returncode": 0, "stdout": ""})
    return SimpleNamespace(
        release=release,
        request=request,
        provenance=provenance,
        protocol=protocol,
        upstream=upstream,
        contract=contract,
    )


def test_actual_prerequisite_builder_requires_both_producers_and_verified_acceptance(prerequisite_boundary):
    f = prerequisite_boundary
    proof = remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)
    assert proof["schema"] == "quest-sdsc-adapted-student-prerequisites-v1"
    assert proof["preflight"] is None
    assert proof["student_inputs"] == [{"path": "fixture"}]
    assert proof["protocol"] == f.protocol
    f.contract.verify_teacher_acceptance.assert_called_once()
    f.contract.input_plans.assert_called_once()


@pytest.mark.parametrize(
    "change", ["proposed", "wrong_head", "source", "same_review_commit", "wrong_fit", "acceptance_failure"]
)
def test_prerequisite_builder_rejects_wrong_lineage_sources_or_acceptance(prerequisite_boundary, change):
    f = prerequisite_boundary
    if change == "proposed":
        f.protocol["review_status"] = "proposed"
    elif change == "wrong_head":
        f.protocol["head"] = "f" * 40
    elif change == "source":
        f.protocol["science_file_sha256"]["src/science.py"] = "f" * 64
    elif change == "same_review_commit":
        f.protocol["acceptance_commit"] = f.protocol["implementation_commit"]
    elif change == "wrong_fit":
        f.upstream["54496291"]["receipt"]["teacher_job_id"] = "99999"
    elif change == "acceptance_failure":
        f.contract.verify_teacher_acceptance.side_effect = ValueError("actual acceptance differs")
    with pytest.raises(ValueError):
        remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)


def test_calibration_needs_matching_preflight_science_teacher_and_runtime(prerequisite_boundary):
    f = prerequisite_boundary
    f.request.update(task=CALIBRATION, preflight_job_id="55500001")
    proof = remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)
    assert proof["preflight"]["receipt"]["job_id"] == "55500001"
    original = copy.deepcopy(f.upstream["55500001"])
    for field in ("science_git_head", "python", "hf_home", *remote.STUDENT_BINDINGS):
        f.upstream["55500001"] = copy.deepcopy(original)
        f.upstream["55500001"]["receipt"][field] = "wrong"
        with pytest.raises(ValueError, match="differs"):
            remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)
    f.upstream["55500001"] = copy.deepcopy(original)
    f.upstream["55500001"]["release_manifest"]["files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="named science"):
        remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)


def test_result_validates_bound_proof_and_actual_publication(submitted, monkeypatch):
    receipt = remote.submit(submitted.request)
    contract = SimpleNamespace(validate_report=Mock(), validate_publication=Mock())
    monkeypatch.setattr(remote, "student_contract", lambda _: contract)
    result = dict(receipt)
    publication = dict(receipt)
    remote.verify_student_result(receipt, result, publication)
    contract.validate_report.assert_called_once_with(result, submitted.proof)
    contract.validate_publication.assert_called_once_with(Path(receipt["result_dir"]), result, publication)
    contract.validate_publication.side_effect = ValueError("persisted artifact changed")
    with pytest.raises(ValueError, match="artifact changed"):
        remote.verify_student_result(receipt, result, publication)
    contract.validate_publication.side_effect = None
    for field in remote.STUDENT_BINDINGS:
        changed = dict(result, **{field: "f" * 64 if result[field] != "f" * 64 else "e" * 64})
        with pytest.raises(ValueError, match="binding differs"):
            remote.verify_student_result(receipt, changed, publication)
    Path(receipt["prerequisites_path"]).write_text("{}")
    with pytest.raises(ValueError, match="proof changed"):
        remote.verify_student_result(receipt, result, publication)
