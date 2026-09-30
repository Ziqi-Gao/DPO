"""Adapted-student CLI/remote trust boundaries; no SSH, models or actual jobs."""

import base64
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
MEMORY_RESULTS = {
    "calibration-details.json": "adapted-calibration.json",
    "memory-initial.json": "memory-initial.json",
    "memory-after_export.json": "memory-after_export.json",
    "memory-after_training.json": "memory-after_training.json",
    "memory-after_validation.json": "memory-after_validation.json",
    "memory-failure.json": "memory-failure.json",
    "training-metrics.jsonl": "canonical_sft/metrics.jsonl",
}


def calibration_fetch_fixture(tmp_path, monkeypatch, version):
    result = tmp_path / "result"
    artifacts = result / "artifacts"
    artifacts.mkdir(parents=True)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "prerequisites.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "protocol_path": f"prereg/amendments/qwen3_adapted_student_calibration_v{version}.json"
                }
            }
        )
    )
    receipt = dict(
        task=CALIBRATION,
        job_id="54509991",
        run_id="bounded-diagnostics",
        intent_id="a" * 32,
        code_sha256="b" * 64,
        result_dir=str(result),
    )
    for name, relative in MEMORY_RESULTS.items():
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps({"diagnostic": name}).encode() + b"\n")
    (result / "adapted-calibration.json").write_text('{"passed": false}\n')
    (artifacts / "initial-canary.pt").write_bytes(b"never fetch model weights")
    (artifacts / "canonical_sft/checkpoint_step_33.pt").write_bytes(b"never fetch a checkpoint")
    monkeypatch.setattr(remote, "bound_receipt", lambda request: (receipt, submission))
    monkeypatch.setattr(remote, "verify_result", lambda receipt: {"verified": False})
    return SimpleNamespace(result=result, artifacts=artifacts, submission=submission, receipt=receipt)


@pytest.mark.parametrize("version", [5, 6])
def test_real_fetch_selects_v6_memory_diagnostics_without_expanding_v5(tmp_path, monkeypatch, version):
    f = calibration_fetch_fixture(tmp_path, monkeypatch, version)
    selected = {name for name, _, _ in remote.selected_files(f.receipt, f.submission, 200)}
    assert selected.intersection(MEMORY_RESULTS) == (set(MEMORY_RESULTS) if version == 6 else set())
    fetched = remote.fetch({})
    expected = {"adapted-calibration.json"} | (set(MEMORY_RESULTS) if version == 6 else set())
    assert {entry["path"] for entry in fetched["files"]} == expected
    assert fetched["skipped"] == []
    assert fetched["total_bytes"] == sum(entry["size"] for entry in fetched["files"])
    for entry in fetched["files"]:
        path = (
            f.artifacts / MEMORY_RESULTS[entry["path"]]
            if entry["path"] in MEMORY_RESULTS
            else f.result / entry["path"]
        )
        raw = path.read_bytes()
        assert base64.b64decode(entry["data_b64"]) == raw
        assert entry["sha256"] == remote.digest(raw)
        assert entry["size"] == len(raw)
        assert entry["tail_only"] is False


@pytest.mark.parametrize("limit", ["file", "total"])
def test_v6_optional_diagnostics_skip_real_file_and_total_limits(tmp_path, monkeypatch, limit):
    f = calibration_fetch_fixture(tmp_path, monkeypatch, 6)
    mib = 1024**2
    if limit == "file":
        (f.artifacts / "memory-failure.json").write_bytes(b"x" * (mib + 1))
        expected = [("memory-failure.json", mib + 1, "optional_file_exceeds_1MiB_limit")]
    else:
        # Three required files plus five optional files fill the actual 8 MiB
        # ceiling; the remaining two optional diagnostics must be skipped.
        for path in (
            f.result / "adapted-calibration.json",
            f.result / "receipt.json",
            f.submission / "control-result.json",
            *(f.artifacts / relative for relative in MEMORY_RESULTS.values()),
        ):
            path.write_bytes(b"x" * mib)
        expected = [
            (name, mib, "optional_file_exceeds_total_fetch_limit")
            for name in ("memory-failure.json", "training-metrics.jsonl")
        ]
    fetched = remote.fetch({})
    assert [(row["path"], row["size"], row["reason"]) for row in fetched["skipped"]] == expected
    names = {entry["path"] for entry in fetched["files"]}
    assert not names.intersection(name for name, _, _ in expected)
    assert all(not name.endswith(".pt") for name in names)
    assert fetched["total_bytes"] == sum(entry["size"] for entry in fetched["files"])
    assert all(entry["size"] <= mib for entry in fetched["files"])
    if limit == "total":
        assert fetched["total_bytes"] == 8 * mib


def test_failed_calibration_fetch_includes_bounded_scientific_log_tails(tmp_path, monkeypatch):
    result = tmp_path / "result"
    artifacts = result / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "train.log").write_bytes(b"old training output\n" * 100000 + b"FSDP contract failed\n")
    (artifacts / "export.log").write_text("initial checkpoint exported\n")
    (artifacts / "initial_checkpoint.pt").write_bytes(b"not a fetch input")
    receipt = dict(
        task=CALIBRATION, job_id="54505782", run_id="failed", code_sha256="a" * 64, result_dir=str(result)
    )
    monkeypatch.setattr(remote, "bound_receipt", lambda request: (receipt, tmp_path / "submission"))
    monkeypatch.setattr(remote, "verify_result", lambda receipt: {"verified": False})
    logs = remote.logs({"lines": 2})["logs"]
    assert logs["train.log"].endswith("FSDP contract failed\n")
    assert len(logs["train.log"].splitlines()) == 2
    fetched = remote.fetch({})
    assert {entry["path"] for entry in fetched["files"]} == {"train.log", "export.log"}
    assert all(entry["tail_only"] and entry["size"] <= 65537 for entry in fetched["files"])
    training = next(entry for entry in fetched["files"] if entry["path"] == "train.log")
    assert base64.b64decode(training["data_b64"]).endswith(b"FSDP contract failed\n")
    assert fetched["result"]["verified"] is False


def protocol_fixture():
    return dict(
        head="d" * 40,
        protocol_sha256="e" * 64,
        artifact_sha256="f" * 64,
        implementation_commit="1" * 40,
        acceptance_commit="2" * 40,
        review_status="accepted",
        protocol_path="prereg/amendments/qwen3_adapted_student_calibration_v5.json",
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
        {"path": "prereg/amendments/qwen3_adapted_student_calibration_v5.json", "sha256": "f" * 64}
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


def cli_calibration_fetch_fixture(client, tmp_path, monkeypatch, version):
    _, args = client
    f = calibration_fetch_fixture(tmp_path, monkeypatch, version)
    f.receipt.update(run_id=args.run_id, student_protocol_artifact_sha256="f" * 64)
    manifest_path = cli.state_root() / "runs/run-test.json"
    record = cli.read_json(manifest_path)
    record["manifest"]["files"][0]["path"] = (
        f"prereg/amendments/qwen3_adapted_student_calibration_v{version}.json"
    )
    cli.write_json(manifest_path, record)
    intent = {
        "request": {"task": CALIBRATION, "run_id": args.run_id},
        "receipt": f.receipt,
    }
    cli.write_json(cli.state_root() / "submissions" / (f.receipt["intent_id"] + ".json"), intent)
    # Only replace the network boundary: file selection, encoding, local
    # receipt lookup, protocol binding and final disk writes all run normally.
    monkeypatch.setattr(cli, "remote", remote.fetch)
    return f


@pytest.mark.parametrize("version,skip_large", [(5, False), (6, False), (6, True)])
def test_real_remote_fetch_reaches_cli_disk_with_protocol_bound_diagnostics(
    client, tmp_path, monkeypatch, version, skip_large
):
    f = cli_calibration_fetch_fixture(client, tmp_path, monkeypatch, version)
    if skip_large:
        (f.artifacts / "memory-failure.json").write_bytes(b"x" * (1024**2 + 1))
    output = io.StringIO()
    with redirect_stdout(output):
        cli.job_command(SimpleNamespace(command="fetch", job_id=f.receipt["job_id"]))
    fetched = json.loads(output.getvalue())
    expected = {"adapted-calibration.json"} | (set(MEMORY_RESULTS) if version == 6 else set())
    if skip_large:
        expected.remove("memory-failure.json")
        assert fetched["skipped"] == [
            {
                "path": "memory-failure.json",
                "size": 1024**2 + 1,
                "reason": "optional_file_exceeds_1MiB_limit",
            }
        ]
    else:
        assert fetched["skipped"] == []
    assert set(fetched["files"]) == expected
    destination = Path(fetched["destination"])
    assert destination.parent == cli.state_root() / "fetched" / f.receipt["job_id"]
    assert {path.name for path in destination.iterdir()} == expected | {"fetch-manifest.json"}
    assert json.loads((destination / "fetch-manifest.json").read_text()) == fetched
    assert fetched["bytes"] == sum((destination / name).stat().st_size for name in expected)
    for name in expected:
        original = f.artifacts / MEMORY_RESULTS[name] if name in MEMORY_RESULTS else f.result / name
        assert (destination / name).read_bytes() == original.read_bytes()


@pytest.mark.parametrize("change", ["historical_manifest", "artifact_hash", "weights"])
def test_cli_rejects_unbound_v6_diagnostics_or_extra_weights(client, tmp_path, monkeypatch, change):
    f = cli_calibration_fetch_fixture(client, tmp_path, monkeypatch, 6)
    if change == "historical_manifest":
        path = cli.state_root() / "runs/run-test.json"
        record = cli.read_json(path)
        record["manifest"]["files"][0]["path"] = (
            "prereg/amendments/qwen3_adapted_student_calibration_v5.json"
        )
        cli.write_json(path, record)
    elif change == "artifact_hash":
        path = cli.state_root() / "submissions" / (f.receipt["intent_id"] + ".json")
        intent = cli.read_json(path)
        intent["receipt"]["student_protocol_artifact_sha256"] = "0" * 64
        cli.write_json(path, intent)
    else:
        response = remote.fetch({})
        response["files"].append(
            {"path": "initial-canary.pt", "data_b64": "", "size": 0, "sha256": remote.digest(b"")}
        )
        monkeypatch.setattr(cli, "remote", lambda _: response)
    with pytest.raises(cli.UserError, match="Unexpected remote fetch"):
        cli.job_command(SimpleNamespace(command="fetch", job_id=f.receipt["job_id"]))


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
    assert argv[-10].endswith("/tools/sdsc_student_launch.sh")
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


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_cli_receipt_binds_deployed_protocol_and_fixed_accepted_teacher(client, version):
    _, args = client
    path = cli.state_root() / "runs/run-test.json"
    record = cli.read_json(path)
    record["manifest"]["files"][0]["path"] = (
        f"prereg/amendments/qwen3_adapted_student_calibration_v{version}.json"
    )
    cli.write_json(path, record)
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
    record["manifest"]["files"].append(dict(record["manifest"]["files"][0]))
    cli.write_json(path, record)
    with pytest.raises(cli.UserError, match="protocol differs"):
        cli.validate_student_receipt(receipt, request)


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
            "tools/sdsc_student_launch.sh",
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
            "path": "prereg/amendments/qwen3_adapted_student_calibration_v5.json",
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
        resources={"mem_gib": 192},
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
        resources={"mem_gib": 192},
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


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_prerequisite_protocol_selection_uses_verified_artifact_not_newest_filename(
    prerequisite_boundary, version
):
    f = prerequisite_boundary
    manifest_path = f.release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact_path = f"prereg/amendments/qwen3_adapted_student_calibration_v{version}.json"
    manifest["files"][-1]["path"] = artifact_path
    if version < 3:
        # Older immutable release contracts did not emit the optional path.
        f.protocol.pop("protocol_path")
    else:
        f.protocol["protocol_path"] = artifact_path
    manifest_path.write_text(json.dumps(manifest))
    assert (
        remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)["protocol"]
        == f.protocol
    )
    f.protocol["protocol_path"] = "prereg/amendments/unknown.json"
    with pytest.raises(ValueError, match="artifact path or bytes"):
        remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)


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


@pytest.mark.parametrize("version,memory", [(5, 192), (6, 384)])
def test_memory_profile_is_bound_before_upstream_reads(prerequisite_boundary, version, memory):
    f = prerequisite_boundary
    path = f"prereg/amendments/qwen3_adapted_student_calibration_v{version}.json"
    f.protocol["protocol_path"] = path
    manifest_path = f.release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][-1]["path"] = path
    manifest_path.write_text(json.dumps(manifest))
    f.request["resources"]["mem_gib"] = memory
    assert remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)["protocol"] == f.protocol
    f.contract.verify_teacher_acceptance.reset_mock()
    for wrong in (192 if memory == 384 else 384, 512, None, True, str(memory)):
        f.request["resources"]["mem_gib"] = wrong
        with pytest.raises(ValueError, match="verified protocol resource"):
            remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)
        f.contract.verify_teacher_acceptance.assert_not_called()


def test_calibration_cannot_reuse_a_different_memory_preflight(prerequisite_boundary):
    f = prerequisite_boundary
    f.request.update(task=CALIBRATION, preflight_job_id="55500001")
    f.upstream["55500001"]["receipt"]["resources"]["mem_gib"] = 384
    with pytest.raises(ValueError, match="verified protocol resource"):
        remote.student_prerequisites(remote.ROOT, f.release, f.request, f.provenance)


def test_cli_new_memory_has_exact_sbatch_argv(client):
    _, args = client
    args.mem_gib = 384
    output = io.StringIO()
    with redirect_stdout(output):
        cli.submit_command(args)
    plan = json.loads(output.getvalue())
    assert "--mem=384G" in plan["sbatch_argv"]
    assert plan["prerequisites_sha256_is_placeholder"] is True
    args.mem_gib = 512
    with pytest.raises(cli.UserError):
        cli.resources(args)
