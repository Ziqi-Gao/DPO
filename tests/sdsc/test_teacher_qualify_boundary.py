"""Real filesystem and subprocess-free boundaries for adapted-teacher qualification."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


contract = module("qualification_contract", ROOT / "tools/sdsc_teacher_qualify_contract.py")
remote = module("qualification_remote", ROOT / "tools/sdsc_remote.py")
cli = module("qualification_cli", ROOT / "tools/sdsc_cli.py")


def claims_fixture(root, monkeypatch):
    binding = dict(
        intent_id="a" * 32,
        run_id="qualification-fixture",
        code_sha256="b" * 64,
        teacher_job_id="12345",
        adapted_teacher_sha256="c" * 64,
        protocol_sha256="d" * 64,
        prerequisites_sha256="e" * 64,
        supplemental_namespace=contract.CLAIM_NAMESPACE,
        validation_file_sha256=contract.VALIDATION_SHA256,
        supplemental_order_sha256=contract.SUPPLEMENTAL_ORDER_SHA256,
    )
    claims_root = root / "teacher-qualification-claims"
    monkeypatch.setattr(contract, "CLAIMS_ROOT", claims_root)
    return dict(
        schema="quest-sdsc-teacher-qualification-reservations-v1",
        binding=binding,
        reservations=contract.reserve_claims(root, binding),
        stage_root=str(claims_root / "stages" / binding["intent_id"]),
        stage_claim_ids={role: binding["intent_id"] + ":" + role for role in contract.ROLES},
    )


def evidence_file(root, claim):
    evidence = dict(
        format_version=1,
        role=claim["role"],
        claim_id=claim["claim_id"],
        predecessor_sha256=claim["predecessor_sha256"],
        attempts=[],
    )
    evidence["sha256"] = contract.sha256_value(evidence)
    path = root / (claim["role"] + ".json")
    path.write_bytes(contract.canonical(evidence) + b"\n")
    return path, evidence


def test_permanent_reservation_prevents_same_checkpoint_retry_and_new_checkpoint_reselection(
    tmp_path, monkeypatch
):
    claims = claims_fixture(tmp_path, monkeypatch)
    assert contract.verify_claims(claims) == claims["binding"]
    for changed in (
        claims["binding"],
        dict(
            claims["binding"], adapted_teacher_sha256="f" * 64, protocol_sha256="0" * 64, intent_id="9" * 32
        ),
    ):
        with pytest.raises(ValueError, match="already reserved"):
            contract.reserve_claims(tmp_path, changed)
    assert len(list((tmp_path / "teacher-qualification-claims/experiments").glob("*.json"))) == 1


def test_stage_claim_precedes_exposure_and_sealed_pass_is_required_for_next_stage(tmp_path, monkeypatch):
    claims = claims_fixture(tmp_path, monkeypatch)
    previous = "f" * 64
    for role in contract.ROLES:
        if role != "train_probe":
            with pytest.raises((FileNotFoundError, ValueError)):
                contract.claim_stage(claims, role, "0" * 64, job_id="23456")
        claim = contract.claim_stage(claims, role, previous, job_id="23456")
        assert (Path(claims["stage_root"]) / (role + ".claim.json")).is_file()
        with pytest.raises(FileExistsError):
            contract.claim_stage(claims, role, previous, job_id="23456")
        path, evidence = evidence_file(tmp_path, claim)
        outcome = contract.seal_stage(claims, role, path, {"metrics_passed": True}, job_id="23456")
        assert outcome["evidence_sha256"] == evidence["sha256"]
        with pytest.raises(FileExistsError):
            contract.seal_stage(claims, role, path, {"metrics_passed": True}, job_id="23456")
        previous = evidence["sha256"]


def test_failed_stage_permanently_blocks_later_inference(tmp_path, monkeypatch):
    claims = claims_fixture(tmp_path, monkeypatch)
    claim = contract.claim_stage(claims, "train_probe", "f" * 64, job_id="23456")
    path, evidence = evidence_file(tmp_path, claim)
    contract.seal_stage(claims, "train_probe", path, {"metrics_passed": False}, job_id="23456")
    with pytest.raises(ValueError, match="previous stage"):
        contract.claim_stage(claims, "supplemental", evidence["sha256"], job_id="23456")
    assert not (Path(claims["stage_root"]) / "supplemental.claim.json").exists()


def test_copied_reservation_cannot_masquerade_as_permanent_claim(tmp_path, monkeypatch):
    claims = claims_fixture(tmp_path, monkeypatch)
    original = Path(claims["reservations"][0]["path"])
    copy_path = tmp_path / "copy.json"
    copy_path.write_bytes(original.read_bytes())
    claims["reservations"][0]["path"] = str(copy_path)
    with pytest.raises(ValueError, match="permanent qualification"):
        contract.verify_claims(claims)


def test_alternative_claim_directory_and_corrupted_original_are_rejected(tmp_path, monkeypatch):
    claims = claims_fixture(tmp_path, monkeypatch)
    copied = copy.deepcopy(claims)
    copied["stage_root"] = str(
        tmp_path / "other/teacher-qualification-claims/stages" / claims["binding"]["intent_id"]
    )
    with pytest.raises(ValueError, match="root"):
        contract.verify_claims(copied)
    Path(claims["reservations"][0]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        contract.verify_claims(claims)


def test_seal_rejects_wrong_job_claim_or_rehashed_predecessor(tmp_path, monkeypatch):
    claims = claims_fixture(tmp_path, monkeypatch)
    claim = contract.claim_stage(claims, "train_probe", "f" * 64, job_id="23456")
    path, evidence = evidence_file(tmp_path, claim)
    with pytest.raises(ValueError):
        contract.seal_stage(claims, "train_probe", path, {"metrics_passed": True}, job_id="99999")
    evidence["predecessor_sha256"] = "0" * 64
    evidence["sha256"] = contract.sha256_value({k: v for k, v in evidence.items() if k != "sha256"})
    path.write_bytes(contract.canonical(evidence))
    with pytest.raises(ValueError):
        contract.seal_stage(claims, "train_probe", path, {"metrics_passed": True}, job_id="23456")


def selection_fixture(root):
    root.mkdir()
    records = []
    for name, raw in [
        ("merged/model.safetensors", b"tiny weight fixture"),
        ("merged/tokenizer.json", b"{}"),
        ("adapter/adapter_config.json", b"{}"),
        ("dense-manifest.json", b'{"fixture":true}'),
    ]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        records.append(dict(path=name, size=len(raw), sha256=contract.hashlib.sha256(raw).hexdigest()))
    return dict(checkpoint_root=str(root), adapted_teacher_sha256="a" * 64, files=records)


def test_checkpoint_staging_rehashes_every_actual_byte_and_preserves_original(tmp_path):
    selection = selection_fixture(tmp_path / "persistent")
    staged = tmp_path / "node-local"
    result = contract.stage_checkpoint(selection, staged)
    assert result["checkpoint_content_rehashed"] and result["node_local_read_back_verified"]
    for row in selection["files"]:
        assert (staged / row["path"]).read_bytes() == (tmp_path / "persistent" / row["path"]).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        contract.stage_checkpoint(selection, staged)


def test_changed_weight_and_external_symlink_fail_staging(tmp_path):
    selection = selection_fixture(tmp_path / "persistent")
    weight = tmp_path / "persistent/merged/model.safetensors"
    weight.write_bytes(b"different contents!")
    with pytest.raises(ValueError, match="checkpoint"):
        contract.stage_checkpoint(selection, tmp_path / "changed")
    weight.unlink()
    weight.symlink_to("/etc/hosts")
    with pytest.raises(ValueError, match="unsafe"):
        contract.stage_checkpoint(selection, tmp_path / "linked")


def test_bounded_raw_fit_evidence_is_read_back_and_never_accepts_weights(tmp_path):
    root = tmp_path / "published"
    root.mkdir()
    raw = b'{"rows":[1,2]}'
    (root / "teacher-fit.json").write_bytes(raw)
    row = dict(path="teacher-fit.json", size=len(raw), sha256=contract.hashlib.sha256(raw).hexdigest())
    evidence = dict(root=str(root), files=[row], total_bytes=len(raw))
    assert contract.stage_fit_evidence(evidence, tmp_path / "staged")["read_back_verified"]
    evidence["total_bytes"] = contract.MAX_FIT_EVIDENCE_BYTES + 1
    with pytest.raises(ValueError, match="bound"):
        contract.stage_fit_evidence(evidence, tmp_path / "oversized")
    evidence.update(total_bytes=len(raw), files=[dict(row, path="model.safetensors")])
    with pytest.raises(ValueError, match="tensors"):
        contract.stage_fit_evidence(evidence, tmp_path / "weight")


def invocation():
    resources = dict(
        account="nwu181",
        partition="nairr-gpu",
        qos="nairr-gpu-normal",
        gpu_type="h100",
        gpus=4,
        cpus=24,
        mem_gib=192,
        time="02:00:00",
    )
    return dict(
        task=contract.TASK,
        resources=resources,
        intent_id="a" * 32,
        run_id="qualify-test",
        code_sha256="b" * 64,
        submission_dir=str(remote.ROOT / "submissions" / ("a" * 32)),
        result_dir="/expanse/lustre/projects/nwu181/zgao12/OPD/results/qualify-test",
        python="/verified/runtime/bin/python",
        hf_home="/verified/huggingface",
        provenance_manifest_sha256="c" * 64,
        provenance_dir=str(remote.ROOT / "provenance" / ("c" * 64)),
    )


def test_exact_qualification_argv_is_finite_ssh_remote_only_and_provenance_bound():
    intent = invocation()
    argv = remote.build_sbatch_argv(intent, remote.ROOT / "releases" / intent["run_id"])
    assert argv[0:2] == ["sbatch", "--parsable"]
    for required in (
        "--gpus=h100:4",
        "--cpus-per-task=24",
        "--mem=192G",
        "--time=02:00:00",
        "--no-requeue",
        "--signal=B:TERM@300",
        "--export=NONE",
    ):
        assert required in argv
    assert argv[-2:] == [intent["provenance_dir"], intent["provenance_manifest_sha256"]]
    assert argv[-10].endswith("/tools/sdsc_teacher_qualify_job.sh")
    for key, value in [("gpus", 1), ("cpus", 72), ("mem_gib", 1024), ("time", "02:00:01")]:
        changed = copy.deepcopy(intent)
        changed["resources"][key] = value
        with pytest.raises(ValueError):
            remote.build_sbatch_argv(changed, remote.ROOT / "releases" / intent["run_id"])


def test_proposed_missing_genuine_git_protocol_cannot_resolve(tmp_path):
    with pytest.raises(ValueError):
        contract.resolved_protocol(ROOT, "0" * 40)


def test_new_task_fetch_allowlist_is_bounded_metadata_only():
    assert cli.TEACHER_QUALIFY_SMALL_RESULTS == remote.TEACHER_QUALIFY_SMALL_RESULTS == contract.SMALL_RESULTS
    assert all("/" not in name and name.endswith(".json") for name in contract.SMALL_RESULTS)
    assert contract.TASK in remote.PROVENANCE_TASKS and contract.TASK not in remote.DIAGNOSTIC_TASKS
    assert cli.TASK_RESULTS[contract.TASK] == remote.TASK_RESULTS[contract.TASK] == contract.RESULT


@pytest.fixture
def remote_submission(tmp_path, monkeypatch):
    fixture_module = module("qualify_remote_test_fixtures", Path(__file__).with_name("test_remote.py"))
    fixture = fixture_module.RemoteBoundaryTests(methodName="runTest")
    fixture.setUp()
    fixture.enable_preflight()
    monkeypatch.setattr(remote, "ROOT", fixture.root)
    monkeypatch.setattr(remote, "PREFLIGHT_STORAGE", fixture_module.remote.PREFLIGHT_STORAGE)
    monkeypatch.setattr(remote, "TEMPORARY_ROOTS", ())
    request = fixture.request
    request.update(
        task=contract.TASK,
        teacher_job_id="12344",
        preflight_job_id=None,
        provenance_dir=str(fixture.root / "provenance" / ("c" * 64)),
        provenance_manifest_sha256="c" * 64,
    )
    request["resources"].update(
        partition="nairr-gpu", qos="nairr-gpu-normal", gpus=4, cpus=24, mem_gib=192, time="02:00:00"
    )
    files = {
        "tools/sdsc_teacher_qualify_job.sh": b"#!/bin/bash\nexit 0\n",
        "tools/sdsc_teacher_qualify.py": b"# fixture worker, never executed\n",
        "tools/sdsc_teacher_qualify_contract.py": (
            ROOT / "tools/sdsc_teacher_qualify_contract.py"
        ).read_bytes(),
        "tools/sdsc_provenance.py": b"# fixture provenance, mocked explicitly\n",
        "src/posttrain_circuits/artifacts/teacher_adaptation_protocol.py": b"# fixture resolver\n",
        "src/posttrain_circuits/artifacts/teacher_acceptance.py": b"# fixture scientific validator\n",
    }
    remote.upload(request, fixture.archive(files))
    provenance = dict(
        provenance_dir=request["provenance_dir"],
        provenance_manifest_sha256="c" * 64,
        science_git_head="d" * 40,
        bundle_sha256="e" * 64,
    )
    monkeypatch.setattr(remote, "verify_provenance", lambda *args: provenance)
    proof = dict(
        schema="quest-sdsc-teacher-qualification-prerequisites-v1",
        task=contract.TASK,
        target={key: request[key] for key in ("run_id", "code_sha256", "intent_id")},
        teacher_job_id=request["teacher_job_id"],
        protocol=dict(
            protocol_sha256="f" * 64, artifact_sha256="0" * 64, science_implementation_sha256="1" * 64
        ),
        selected_checkpoint=dict(adapted_teacher_sha256="2" * 64),
    )
    monkeypatch.setattr(remote, "teacher_qualification_prerequisites", lambda *args: proof)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return dict(returncode=0, stdout="23456;expanse\n", stderr="")

    monkeypatch.setattr(remote, "run", run)
    yield request, calls, proof
    fixture.doCleanups()


def test_remote_qualification_records_permanent_claims_before_single_sbatch(remote_submission):
    request, calls, proof = remote_submission
    receipt = remote.submit(request)
    assert len(calls) == 1 and calls[0][0] == "sbatch"
    assert receipt["job_id"] == "23456" and receipt["teacher_job_id"] == "12344"
    assert receipt["adapted_teacher_sha256"] == proof["selected_checkpoint"]["adapted_teacher_sha256"]
    claims_path = Path(receipt["claims_path"])
    claims = __import__("json").loads(claims_path.read_bytes())
    assert len(claims["reservations"]) == 2
    assert all(Path(row["path"]).is_file() for row in claims["reservations"])
    assert receipt["claims_sha256"] == contract.hashlib.sha256(claims_path.read_bytes()).hexdigest()
    assert contract.verify_claims(claims, expected_root=remote.ROOT / "teacher-qualification-claims")
    with pytest.raises(ValueError, match="Intent already exists"):
        remote.submit(request)
    assert len(calls) == 1


def test_remote_failed_qualification_prerequisite_stops_before_claim_or_sbatch(
    remote_submission, monkeypatch
):
    request, calls, _ = remote_submission

    def rejected(*args):
        raise ValueError("scientific protocol is still proposed")

    monkeypatch.setattr(remote, "teacher_qualification_prerequisites", rejected)
    with pytest.raises(ValueError, match="still proposed"):
        remote.submit(request)
    assert not calls and not (remote.ROOT / "run-claims").exists()
    assert not (remote.ROOT / "teacher-qualification-claims").exists()


def test_unknown_submission_keeps_claims_and_forbids_blind_retry(remote_submission, monkeypatch):
    request, calls, _ = remote_submission
    monkeypatch.setattr(
        remote, "run", lambda *args, **kwargs: dict(returncode=1, stdout="", stderr="lost receipt")
    )
    result = remote.submit(request)
    assert result["state"] == "unknown"
    assert list((remote.ROOT / "teacher-qualification-claims/experiments").glob("*.json"))
    with pytest.raises(ValueError, match="Intent already exists"):
        remote.submit(request)


def report_fixture():
    fit_fixture = module(
        "qualify_fit_boundary_fixture", Path(__file__).with_name("test_teacher_fit_boundary.py")
    )
    protocol = dict(
        head="a" * 40,
        protocol_sha256="b" * 64,
        artifact_sha256="c" * 64,
        implementation_commit="d" * 40,
        acceptance_commit="e" * 40,
        science_implementation_sha256="f" * 64,
    )
    proof = dict(
        target=dict(intent_id="1" * 32, run_id="qualify-fixture", code_sha256="2" * 64),
        teacher_job_id="12344",
        protocol=protocol,
        selected_checkpoint=dict(adapted_teacher_sha256="3" * 64, selected_step=128),
        upstream=dict(publication_receipt_sha256="4" * 64),
    )
    report = dict(
        proof["target"],
        task=contract.TASK,
        artifact_kind="adapted_teacher_qualification_execution",
        job_id="12345",
        teacher_job_id="12344",
        adapted_teacher_sha256="3" * 64,
        selected_step=128,
        science_git_head=protocol["head"],
        protocol_sha256=protocol["protocol_sha256"],
        protocol_artifact_sha256=protocol["artifact_sha256"],
        protocol_implementation_commit=protocol["implementation_commit"],
        protocol_acceptance_commit=protocol["acceptance_commit"],
        science_implementation_sha256=protocol["science_implementation_sha256"],
        fit_publication_receipt_sha256="4" * 64,
        acceptance_evidence_sha256="5" * 64,
        acceptance_evidence_file_sha256="6" * 64,
        selection_sha256="7" * 64,
        passed=True,
        scientific_evidence_verified=True,
        protocol_accepted=True,
        exit_code=0,
        checks={key: True for key in contract.CHECKS},
        stage_summaries={role: {"metrics_passed": True} for role in contract.ROLES},
        world_size=4,
        rank_summaries=[
            dict(
                rank=rank,
                world_size=4,
                logical_device=rank,
                gpu_name="NVIDIA H100",
                torch_num_threads=6,
                cgroup_memory=fit_fixture.memory_fixture("12345"),
            )
            for rank in range(4)
        ],
    )
    for key in (
        "formal_teacher_accepted",
        "accepted_science",
        "full_teacher_ready",
        "g0_passed",
        "execution_class_certified",
        "student_training_started",
        "teacher_training_started",
        "training_started",
        "readiness_artifact_produced",
        "original_128_token_readiness_pass_claim",
    ):
        report[key] = False
    return report, proof


def test_qualification_report_requires_all_stages_actual_ranks_and_no_self_acceptance():
    report, proof = report_fixture()
    contract.validate_report(report, proof)
    mutations = [
        lambda row: row.update(formal_teacher_accepted=True),
        lambda row: row.pop("formal_teacher_accepted"),
        lambda row: row.update(full_teacher_ready=True),
        lambda row: row.update(teacher_training_started=True),
        lambda row: row.update(science_git_head="0" * 40),
        lambda row: row.update(adapted_teacher_sha256="0" * 64),
        lambda row: row.update(protocol_acceptance_commit=row["protocol_implementation_commit"]),
        lambda row: row.update(checks={"formal_store": True}),
        lambda row: row["stage_summaries"]["supplemental"].update(metrics_passed=False),
        lambda row: row["rank_summaries"].pop(),
        lambda row: row["rank_summaries"][2].update(torch_num_threads=24),
        lambda row: row["rank_summaries"][2]["cgroup_memory"].update(job_id="99999"),
    ]
    for mutation in mutations:
        changed = copy.deepcopy(report)
        mutation(changed)
        with pytest.raises(ValueError):
            contract.validate_report(changed, proof)


def test_scientific_inventory_has_distinct_physical_and_unique_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "MAX_SCIENTIFIC_BYTES", 10)
    monkeypatch.setattr(contract, "MAX_UNIQUE_SCIENTIFIC_BYTES", 6)
    report = dict(
        run_id="qualify-fixture", job_id="12345", code_sha256="a" * 64, adapted_teacher_sha256="b" * 64
    )
    for name in ("stages/a.json", "stages/b.json", "stages/c.json"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"abc")

    def manifest():
        files = [
            dict(
                path=p.relative_to(tmp_path).as_posix(),
                size=p.stat().st_size,
                sha256=contract.hashlib.sha256(p.read_bytes()).hexdigest(),
            )
            for p in sorted((tmp_path / "stages").glob("*.json"))
        ]
        value = dict(
            artifact_kind="adapted_teacher_qualification_inventory",
            files=files,
            run_id=report["run_id"],
            job_id=report["job_id"],
            code_sha256=report["code_sha256"],
            teacher_identity={"teacher_checkpoint_sha256": report["adapted_teacher_sha256"]},
            formal_teacher_accepted=False,
        )
        value["sha256"] = contract.sha256_value(value)
        raw = contract.canonical(value)
        (tmp_path / "qualification-manifest.json").write_bytes(raw)
        report.update(
            qualification_inventory_sha256=value["sha256"],
            qualification_manifest_sha256=contract.hashlib.sha256(raw).hexdigest(),
        )

    manifest()
    assert len(contract.validate_inventory(tmp_path, report)) == 3
    (tmp_path / "stages/c.json").write_bytes(b"abcd")
    manifest()
    with pytest.raises(ValueError, match="physical/unique"):
        contract.validate_inventory(tmp_path, report)
    (tmp_path / "stages/c.json").write_bytes(b"abc")
    (tmp_path / "stages/d.json").write_bytes(b"abc")
    manifest()
    with pytest.raises(ValueError, match="physical/unique"):
        contract.validate_inventory(tmp_path, report)


def test_historical_fit_origin_must_match_reviewed_execution_and_shared_science():
    fit = module("qualify_fit_origin_fixture", ROOT / "tools/sdsc_teacher_fit_contract.py")
    paths = (*fit.KERNEL_PATHS, *contract.FIT_SCIENCE_ORIGIN_PATHS)
    manifest = {"files": [{"path": path, "sha256": "a" * 64} for path in paths]}
    execution = fit.execution_plan(manifest)
    upstream = dict(
        release_manifest=manifest,
        report=dict(execution_plan=execution, execution_plan_sha256=contract.sha256_value(execution)),
    )
    protocol = {"science_file_sha256": {path: "a" * 64 for path in paths}}
    assert contract.verify_fit_origin(upstream, protocol) == execution
    for path in (fit.KERNEL_PATHS[0], *contract.FIT_SCIENCE_ORIGIN_PATHS):
        changed = copy.deepcopy(protocol)
        changed["science_file_sha256"][path] = "b" * 64
        with pytest.raises(ValueError, match="fit origin"):
            contract.verify_fit_origin(upstream, changed)
        changed["science_file_sha256"].pop(path)
        with pytest.raises(ValueError, match="fit origin"):
            contract.verify_fit_origin(upstream, changed)
    upstream["report"]["execution_plan_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="fit report execution"):
        contract.verify_fit_origin(upstream, protocol)


def test_acceptance_transport_requires_actual_persistent_claims_and_exact_stage_bytes(tmp_path, monkeypatch):
    claims = claims_fixture(tmp_path, monkeypatch)
    report = dict(job_id="23456", selection_sha256="f" * 64, protocol_sha256="d" * 64)
    output = tmp_path / "output"
    output.mkdir()
    previous = report["selection_sha256"]
    for role in contract.ROLES:
        claim = contract.claim_stage(claims, role, previous, job_id=report["job_id"])
        path, evidence = evidence_file(tmp_path, claim)
        directory = output / "stages" / role
        directory.mkdir(parents=True)
        (directory / "evidence.json").write_bytes(path.read_bytes())
        summary = {"metrics_passed": True}
        outcome = contract.seal_stage(claims, role, path, summary, job_id=report["job_id"])
        for name, value in (("claim", claim), ("outcome", outcome), ("summary", summary)):
            (directory / (name + ".json")).write_bytes(contract.canonical(value))
        previous = evidence["sha256"]
    acceptance = dict(
        artifact_kind="adapted_teacher_acceptance_evidence",
        scientific_evidence_verified=True,
        formal_teacher_accepted=False,
        protocol_sha256=report["protocol_sha256"],
    )
    acceptance["sha256"] = contract.sha256_value(acceptance)
    raw = contract.canonical(acceptance)
    (output / "acceptance-evidence.json").write_bytes(raw)
    report.update(
        acceptance_evidence_sha256=acceptance["sha256"],
        acceptance_evidence_file_sha256=contract.hashlib.sha256(raw).hexdigest(),
    )
    assert contract.validate_acceptance_file(output, report, claims) == acceptance
    for relative, mutation in (
        ("stages/supplemental/claim.json", lambda row: row.update(claimed_before_inference=False)),
        ("stages/formal_readiness/summary.json", lambda row: row.update(extra="changed")),
        ("stages/formal_store/evidence.json", lambda row: row.update(attempts=["changed"])),
        ("stages/train_probe/outcome.json", lambda row: row.update(evidence_file_size=0)),
    ):
        path = output / relative
        original = path.read_bytes()
        value = json.loads(original)
        mutation(value)
        path.write_bytes(contract.canonical(value))
        with pytest.raises(ValueError, match="persistent stage"):
            contract.validate_acceptance_file(output, report, claims)
        path.write_bytes(original)
