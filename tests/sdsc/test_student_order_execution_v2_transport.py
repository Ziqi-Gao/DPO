"""V2 transport retains original scientific gates; no GPU or remote operations.

Original node early-failure/process cases live in the v2 node suite. This suite
retains the unchanged transport boundaries and adds actual prerequisite seams.
"""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location("test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c(monkeypatch):
    original = subprocess.run

    def guarded(argv, *args, **kwargs):
        if Path(argv[0]).name in {"ssh", "squeue", "sacct", "sbatch", "scontrol", "srun", "scancel"}:
            pytest.fail("local fixture attempted remote/Slurm")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    return load(ROOT / "tools/sdsc_student_order_execution_v2.py")


@pytest.fixture
def node():
    return load(ROOT / "tools/sdsc_student_order_execution_v2_job.py")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    old = load(ROOT / "tests/sdsc/test_student_order_transport.py")
    inner = old.plan.__wrapped__(c.science, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "CONTROL", c.science.CONTROL)
    monkeypatch.setattr(c, "PROJECT", c.science.PROJECT)
    # Isolate remote prerequisite plan parsing in this transport fixture. The
    # contract suite separately exercises the real frozen diagnostic identity.
    diagnostic = {"fixture": "no remote prerequisite acceptance"}
    original_helper = c.helper
    contract = original_helper("sdsc_student_order_execution_v2_contract")
    contract.DIAGNOSTIC = dict(contract.DIAGNOSTIC, plan_sha256=c.sha(c.canonical(diagnostic)))
    monkeypatch.setattr(c.diag, "validate_plan", lambda value: value)
    monkeypatch.setattr(
        c,
        "helper",
        lambda name, *args: contract
        if name == "sdsc_student_order_execution_v2_contract"
        else original_helper(name, *args),
    )
    execution = dict(
        core_sha256="9" * 64,
        artifact_sha256="8" * 64,
        implementation_commit="3" * 40,
        acceptance_commit="4" * 40,
        head=inner["protocol"]["head"],
        science_binding=copy.deepcopy(inner["protocol"]),
        control_file_sha256={p: c.sha((ROOT / p).read_bytes()) for p in c.NEW_TOOLS},
    )
    value = dict(
        schema=c.SCHEMA,
        task=c.TASK,
        science_plan=inner,
        execution=execution,
        diagnostic_plan=diagnostic,
        preflight_execution=None,
        intent_id=inner["intent_id"],
        **dict.fromkeys(c.FLAGS, False),
    )
    value["recovery_claim"] = c.recovery_claim(value)
    return c.validate_plan(value)


def fit_plan(c, plan):
    old = load(ROOT / "tests/sdsc/test_student_order_transport.py")
    previous = copy.deepcopy(plan)
    inner = plan["science_plan"]
    inner.update(
        mode="fit",
        resources=c.science.resources("fit"),
        preflight=dict(
            plan=previous["science_plan"],
            plan_sha256=c.sha(c.canonical(previous["science_plan"])),
            audit=old.preflight_audit(c.science, previous["science_plan"]),
        ),
    )
    old.bind(c.science, inner)
    plan.update(preflight_execution=previous, intent_id=inner["intent_id"])
    plan["recovery_claim"] = c.recovery_claim(plan)
    return c.validate_plan(plan)


def test_inner_original_plan_and_exact_resources_remain_valid(c, plan):
    c.science.validate_plan(plan["science_plan"])
    argv = c.sbatch(plan)
    assert argv[:-1] == c.science.sbatch(plan["science_plan"])[:-1]
    assert argv[-1] == c.sha(c.canonical(plan))
    assert "--gpus=h100:2" in argv and "--cpus-per-task=24" in argv and "--mem=393216M" in argv
    assert "--time=01:00:00" in argv and "--signal=B:TERM@180" in argv
    script = c.job_script(plan).decode()
    assert "sdsc_student_order_execution_v2_job.py" in script and "execution-plan.json" in script
    assert "--nodelist" not in script and "--exclude" not in script
    assert plan["science_plan"]["scientific_claim"] != plan["recovery_claim"]


@pytest.mark.parametrize(
    "mutation", ["scope", "claim", "science", "head", "control", "diagnostic", "self_review"]
)
def test_outer_binding_rejects_mutations(c, plan, mutation):
    if mutation == "scope":
        plan["g0_passed"] = True
    elif mutation == "claim":
        plan["recovery_claim"] += ".retry"
    elif mutation == "science":
        plan["science_plan"]["science_identity"]["learning_rate"] = 5e-4
    elif mutation == "head":
        plan["execution"]["head"] = "5" * 40
    elif mutation == "control":
        plan["execution"]["control_file_sha256"].pop(c.NEW_TOOLS[0])
    elif mutation == "diagnostic":
        plan["diagnostic_plan"]["created_at"] = "2026-10-04"
    else:
        plan["execution"]["acceptance_commit"] = plan["execution"]["implementation_commit"]
    with pytest.raises((ValueError, KeyError)):
        c.validate_plan(plan)


def test_recovery_once_claim_cannot_change_with_release_or_execution_controls(c, plan):
    claim = c.recovery_claim(plan)
    plan["science_plan"]["run_id"] = "different-release"
    plan["execution"]["control_file_sha256"][c.NEW_TOOLS[1]] = "f" * 64
    assert c.recovery_claim(plan) == claim


def test_fit_retains_original_unique_science_claim_and_exact_outer_preflight(c, plan):
    fit_plan(c, plan)
    assert c.science.validate_plan(plan["science_plan"])
    assert "--time=08:00:00" in c.sbatch(plan)
    plan["preflight_execution"]["execution"]["artifact_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="fit/recovery"):
        c.validate_plan(plan)


def test_fit_rejects_raw_audit_absence(c, plan):
    fit_plan(c, plan)
    plan["science_plan"]["preflight"].pop("audit")
    with pytest.raises((ValueError, KeyError)):
        c.validate_plan(plan)


def write_claims(c, plan):
    inner = plan["science_plan"]
    directory = Path(inner["submission_dir"])
    directory.mkdir(parents=True)
    for path, data in (
        (directory / "plan.json", inner),
        (directory / "execution-plan.json", plan),
        (Path(inner["claim"]), {"plan_sha256": c.sha(c.canonical(inner))}),
        (Path(plan["recovery_claim"]), c.claim_record(plan)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        c.write_once(path, c.canonical(data))
    if inner["mode"] == "fit":
        path = Path(inner["scientific_claim"])
        path.parent.mkdir(parents=True, exist_ok=True)
        c.write_once(path, c.canonical(c.science.scientific_claim_record(inner)))


def test_original_consumed_preflight_claim_preserved_and_outer_is_required(c, plan):
    write_claims(c, plan)
    old = Path(plan["science_plan"]["scientific_claim"])
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text('{"failed_job":"54623814"}')
    before = old.read_bytes()
    c.check_claims(plan)
    assert old.read_bytes() == before
    Path(plan["recovery_claim"]).write_text("{}")
    with pytest.raises(ValueError, match="permanent"):
        c.check_claims(plan)


def test_fit_requires_both_permanent_claims(c, plan):
    fit_plan(c, plan)
    write_claims(c, plan)
    c.check_claims(plan)
    Path(plan["science_plan"]["scientific_claim"]).write_text("{}")
    with pytest.raises(ValueError, match="scientific claim"):
        c.check_claims(plan)


def test_prepare_wraps_original_plan_and_verifies_deployment(c, plan, tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    inner_path = tmp_path / "science.json"
    inner_path.write_bytes(c.canonical(plan["science_plan"]))
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_bytes(c.canonical(plan["diagnostic_plan"]))
    pins = dict(
        plan["execution"]["control_file_sha256"], **{c.CONTRACT_PATH: plan["execution"]["artifact_sha256"]}
    )
    original_read = c.read
    monkeypatch.setattr(c, "ROOT", root)
    monkeypatch.setattr(c, "binding", lambda *_: plan["execution"])
    monkeypatch.setattr(c.science, "manifest_records", lambda _: {p: {"sha256": h} for p, h in pins.items()})
    monkeypatch.setattr(
        c,
        "read",
        lambda p, *a: (ROOT / p.relative_to(root)).read_bytes()
        if p.is_relative_to(root) and str(p.relative_to(root)) in c.NEW_TOOLS
        else original_read(p, *a),
    )
    # The production resolver is stubbed only for this unaccepted local fixture.
    artifact = root / c.CONTRACT_PATH
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture execution acceptance")
    plan["execution"]["artifact_sha256"] = c.sha(artifact.read_bytes())
    pins[c.CONTRACT_PATH] = plan["execution"]["artifact_sha256"]
    cli = SimpleNamespace(
        run_record=lambda _: dict(
            state="deployed", manifest={"code_sha256": plan["science_plan"]["code_sha256"]}
        )
    )
    args = SimpleNamespace(
        science_plan=inner_path, diagnostic_plan=diagnostic_path, preflight_execution_plan=None
    )
    result = c.prepare(args, cli)
    loaded = json.loads(Path(result["plan"]).read_bytes())
    assert loaded["science_plan"] == json.loads(inner_path.read_bytes())
    assert result["plan_sha256"] == c.sha(c.canonical(loaded))
    with pytest.raises(FileExistsError):
        c.prepare(args, cli)


def test_verify_source_additional_controls_cannot_be_missing(c, plan, monkeypatch):
    monkeypatch.setattr(c.science, "verify_source", lambda *a: {})
    monkeypatch.setattr(c.science, "manifest_records", lambda _: {})
    with pytest.raises(ValueError, match="execution controls"):
        c.verify_source(plan)


def test_node_argv_preserves_original_scientific_args_and_actual_cvd(c, plan, node, tmp_path):
    identity = {"job_id": "123", "cuda_visible_devices": "GPU-a,GPU-b"}
    argv = node.worker_argv(
        plan,
        tmp_path / "source",
        tmp_path / "science",
        tmp_path / "inputs.json",
        tmp_path / "artifacts",
        tmp_path / "execution",
        identity,
    )
    assert argv[-4:] == [
        "--inputs-json",
        str(tmp_path / "inputs.json"),
        "--output-dir",
        str(tmp_path / "artifacts"),
    ]
    assert argv[argv.index("--expected-cuda-visible-devices") + 1] == "GPU-a,GPU-b"
    assert "--num_processes" in argv and "--num_cpu_threads_per_process" in argv and "--gpu_ids" not in argv
    probe = node.probe_argv(plan, tmp_path / "execution", identity)
    assert any(x.endswith("sdsc_student_order_execution_v2_probe.py") for x in probe)
    assert "accelerate.commands.launch" not in probe


def test_node_reuses_original_thirteen_input_fields(c, plan, node, tmp_path):
    inner = plan["science_plan"]
    science_root = tmp_path / "science"
    path = science_root / c.science.PROTOCOL_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / c.science.PROTOCOL_PATH).read_bytes())
    args = (inner, tmp_path, science_root, {"job_id": "123"}, {"a": "b"}, c.science)
    expected = c.helper("sdsc_student_order_job").build_worker_inputs(*args)
    actual = node.build_worker_inputs(*args)
    assert actual == expected and len(actual) == 13
    assert actual["plan_sha256"] == c.sha(c.canonical(inner)) != c.sha(c.canonical(plan))


def test_execution_missing_never_promotes_scientific_success(c, plan, monkeypatch):
    monkeypatch.setattr(
        c.science,
        "inspect_job",
        lambda *a, **k: dict(
            success=True, stage_complete=True, preparation_complete=False, publication_verified=True
        ),
    )
    result = c.inspect_job(plan, {"job_id": "123"})
    assert result["scientific_stage_complete"] is True
    assert result["success"] is False and result["stage_complete"] is False


def publish_failed_execution(c, plan, node):
    inner = plan["science_plan"]
    Path(inner["result_dir"]).mkdir(parents=True)
    result = dict(
        job_id="123",
        plan_sha256=c.sha(c.canonical(plan)),
        cuda_visible_devices="2,3",
        startup_passed=False,
        science_publication_sha256="7" * 64,
        **dict.fromkeys(c.FLAGS, False),
    )
    node.publish_execution(plan, c, None, result)
    return result


def test_failed_startup_publishes_without_claiming_science(c, plan, node):
    publish_failed_execution(c, plan, node)
    result = c.execution_publication(plan, "123")
    assert result["execution_publication_verified"] is True and result["startup_passed"] is False
    assert all(result["execution_publication"][k] is False for k in c.FLAGS)


def test_tampered_execution_publication_rejected(c, plan, node):
    publish_failed_execution(c, plan, node)
    path = Path(plan["science_plan"]["result_dir"]) / "execution/execution-node.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        c.execution_publication(plan, "123")


def test_unknown_acknowledgement_reconciliation_never_resubmits(c, plan, monkeypatch):
    write_claims(c, plan)
    calls = []

    def run(argv):
        calls.append(argv[0])
        return dict(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(c, "run", run)
    with pytest.raises(ValueError, match="never resubmit"):
        c.reconcile(plan)
    assert calls == ["squeue", "sacct"]


def test_reconcile_exact_single_job_makes_both_receipts(c, plan, monkeypatch):
    write_claims(c, plan)
    inner = plan["science_plan"]
    line = "|".join(("123", inner["job_name"], inner["intent_id"])) + "\n"
    monkeypatch.setattr(
        c, "run", lambda argv: dict(returncode=0, stdout=line if argv[0] == "squeue" else "", stderr="")
    )
    receipt = c.reconcile(plan)
    assert receipt["job_id"] == "123" and receipt["task"] == c.science.TASK
    outer = c.document(Path(inner["submission_dir"]) / "execution-receipt.json")
    assert outer["plan_sha256"] == c.sha(c.canonical(plan))
    assert outer["scientific_receipt"] == receipt


def test_fetch_rejects_arbitrary_path_even_with_valid_base64(c, plan):
    receipt = c.science.make_receipt(plan["science_plan"], "123")
    with pytest.raises(ValueError, match="unexpected fetched"):
        c.validate_download(
            dict(receipt=receipt, files={"../model.pt": base64.b64encode(b"bad").decode()}), plan
        )


@pytest.fixture
def startup_worker(monkeypatch):
    tests = load(ROOT / "tests/sdsc/test_student_order_execution_worker.py")
    return tests.setup.__wrapped__(monkeypatch)[0]


def test_actual_startup_validator_rejects_observed_cuda_failure(c, startup_worker, monkeypatch):
    startup_worker.import_torch().cuda.available = False
    report = startup_worker.startup("probe", "12345", "2,3")
    monkeypatch.setattr(c, "helper", lambda name: startup_worker)
    with pytest.raises(ValueError, match="CUDA startup failed"):
        c.validate_startup(report, "12345", "2,3", "early_node")


def test_remote_submit_consumes_once_claim_before_unknown_ack(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda *a: None)
    monkeypatch.setattr(c, "admission", lambda *a: {})
    calls = []

    def run(argv):
        calls.append(argv)
        return dict(returncode=1, stdout="", stderr="fixture unknown")

    monkeypatch.setattr(c, "run", run)
    request = dict(
        action="submit",
        authorize=True,
        plan=plan,
        dry_run=dict(dry_run=True, blockers=[], plan_sha256=c.sha(c.canonical(plan)), argv=c.sbatch(plan)),
    )
    with pytest.raises(ValueError, match="acknowledgement"):
        c.remote_action(request)
    assert len(calls) == 1 and calls[0][0] == "sbatch"
    assert Path(plan["recovery_claim"]).exists()
    assert (Path(plan["science_plan"]["submission_dir"]) / "unknown.json").exists()
    with pytest.raises(FileExistsError):
        c.remote_action(request)
    assert len(calls) == 1


@pytest.mark.parametrize("which", ["failed", "diagnostic"])
def test_prerequisite_missing_or_changed_claim_stops_before_recovery(c, plan, monkeypatch, which):
    calls = []

    def failed(_):
        calls.append("failed claims")
        if which == "failed":
            raise ValueError("failed permanent claims differ")
        return {}

    def diagnostic(request):
        assert request == dict(action="status", plan=plan["diagnostic_plan"])
        calls.append("diagnostic claims")
        raise ValueError("diagnostic permanent claims differ")

    monkeypatch.setattr(c.diag, "verify_failed", failed)
    monkeypatch.setattr(c.diag, "remote_action", diagnostic)
    with pytest.raises(ValueError, match="permanent claims"):
        c.prerequisite_status(plan)
    assert calls[0] == "failed claims"
    assert len(calls) == (1 if which == "failed" else 2)


@pytest.mark.parametrize("change", ["missing", "nonzero", "wrong_rank", "wrong_argv", "false_int"])
def test_combined_success_rejects_missing_or_wrong_rank_exit(c, change):
    args = ["--inputs-json", "/work/inputs.json", "--output-dir", "/work/artifacts"]
    report = dict(
        schema="quest-sdsc-student-order-execution-exit-v1",
        job_id="123",
        rank=1,
        original_worker_sha256=c.FROZEN_WORKER_SHA,
        original_argv=args,
        worker_invoked=True,
        exit_code=0,
        **dict.fromkeys(c.FLAGS, False),
    )
    if change == "missing":
        report.pop("worker_invoked")
    elif change == "nonzero":
        report["exit_code"] = 2
    elif change == "wrong_rank":
        report["rank"] = 0
    elif change == "wrong_argv":
        report["original_argv"] = []
    else:
        report["exit_code"] = False
    with pytest.raises(ValueError, match="rank worker"):
        c.validate_rank_exit(report, "123", 1, args)


def test_fit_waits_for_complete_execution_acknowledgement(c, plan, monkeypatch):
    fit_plan(c, plan)
    previous = plan["preflight_execution"]
    inner = previous["science_plan"]
    write_claims(c, previous)
    receipt = c.science.make_receipt(inner, "123")
    directory = Path(inner["submission_dir"])
    c.write_once(directory / "receipt.json", c.canonical(receipt))
    calls = []

    def inspect(*args, **kwargs):
        calls.append(True)
        return dict(
            job_id="123",
            success=True,
            stage_complete=True,
            publication_sha256="7" * 64,
            result_sha256="8" * 64,
        )

    monkeypatch.setattr(c, "inspect_job", inspect)
    with pytest.raises(FileNotFoundError):
        c.verify_preflight(plan)
    assert calls == []
    c.write_once(
        directory / "execution-receipt.json",
        c.canonical(dict(c.claim_record(previous), job_id="123", scientific_receipt=receipt)),
    )
    assert c.verify_preflight(plan)["success"] is True and len(calls) == 1


def test_new_execution_claim_preserves_original_fit_uniqueness(c, plan):
    original_claim = c._previous.recovery_claim(plan)
    assert c.recovery_claim(plan) != original_claim
    assert "/student-order-execution-v2-claims/" in c.recovery_claim(plan)
    fit_plan(c, plan)
    science_claim = c.science.scientific_claim_record(plan["science_plan"])
    old_core = plan["execution"]["core_sha256"]
    plan["execution"]["core_sha256"] = "f" * 64
    assert c.science.scientific_claim_record(plan["science_plan"]) == science_claim
    plan["execution"]["core_sha256"] = old_core


def test_v2_adds_startup_capacity_without_changing_science_fetch(c):
    assert c.EXECUTION_MAX == 2 * 1024**2
    assert c.CAP == 1024**2
    assert c.science.MAX_FETCH == 224 * 1024**2
    assert "trace-meta.json" in c.EXECUTION_NAMES
    assert c.validate_startup is c._previous.validate_startup
    assert c.validate_rank_exit is c._previous.validate_rank_exit


@pytest.mark.parametrize("size", [1024**2 + 1, 2 * 1024**2 + 1])
def test_execution_download_cannot_exceed_new_per_file_bound(c, plan, size):
    receipt = c.science.make_receipt(plan["science_plan"], "123")
    raw = b"x" * size
    with pytest.raises(ValueError, match="startup bound"):
        c.validate_download(
            dict(
                receipt=receipt, files={"execution/startup.log": base64.b64encode(raw).decode()}, bytes=size
            ),
            plan,
        )


def test_execution_download_total_includes_receipt(c, plan):
    receipt = c.science.make_receipt(plan["science_plan"], "123")
    payload = {
        "execution/startup.log": b"x" * c.CAP,
        "execution/trace-meta.json": b"y" * c.CAP,
        "execution/receipt.json": b"{}",
    }
    with pytest.raises(ValueError, match="startup bound"):
        c.validate_download(
            dict(
                receipt=receipt,
                files={k: base64.b64encode(v).decode() for k, v in payload.items()},
                bytes=sum(map(len, payload.values())),
            ),
            plan,
        )


@pytest.fixture
def actual_prerequisites(c, tmp_path, monkeypatch):
    diagnostic_path = ROOT / ".sdsc/torch-import-probe/2ac800ac6e8e71ccf2e20068cf290a1b/plan.json"
    import_fetch = ROOT / ".sdsc/fetched/54643629/fetch-twg_e5bt"
    old_fetch = ROOT / ".sdsc/fetched/54643463/fetch-tcvymoay"
    if not all(
        path.is_file()
        for path in (diagnostic_path, import_fetch / "fetch-manifest.json", old_fetch / "fetch-manifest.json")
    ):
        pytest.skip("recorded paired failure/import evidence absent")
    diagnostic = json.loads(diagnostic_path.read_bytes())
    previous = diagnostic["failed"]["execution_plan"]
    old_fixture = load(ROOT / "tests/sdsc/test_torch_import_probe_transport.py")
    saved = SimpleNamespace(
        plan=diagnostic,
        execution=previous,
        failed=previous["science_plan"],
        report=json.loads((old_fetch / "node-result.json").read_bytes()),
        publication=json.loads((old_fetch / "receipt.json").read_bytes()),
        execution_publication=json.loads((old_fetch / "execution/receipt.json").read_bytes()),
        capture=json.loads((old_fetch / "fetch-manifest.json").read_bytes()),
    )
    failed = old_fixture.restored_failure.__wrapped__(c.diag, saved, tmp_path, monkeypatch)
    mapped = failed.mapped
    capture = json.loads((import_fetch / "fetch-manifest.json").read_bytes())
    receipt = capture["receipt"]
    destination = mapped(diagnostic["result_dir"])
    destination.mkdir(parents=True)
    for name in (*c.diag.NAMES, "receipt.json"):
        (destination / name).write_bytes((import_fetch / name).read_bytes())
    directory = mapped(diagnostic["submission_dir"])
    directory.mkdir(parents=True)
    for name, value in (
        ("plan.json", diagnostic),
        ("receipt.json", receipt),
        (
            "live-binding.json",
            dict(
                job_id="54643629",
                plan_sha256=c.sha(c.canonical(diagnostic)),
                fields=dict(
                    JobId="54643629",
                    JobName=diagnostic["job_name"],
                    Account="nwu181",
                    Partition="nairr-gpu-shared",
                    QOS="nairr-gpu-shared-normal",
                    Comment=diagnostic["intent_id"],
                    Command=diagnostic["submission_dir"] + "/job.sh",
                    WorkDir=diagnostic["submission_dir"],
                ),
            ),
        ),
    ):
        (directory / name).write_bytes(c.canonical(value))
    claim = mapped(diagnostic["claim"])
    claim.parent.mkdir(parents=True)
    claim.write_bytes(c.canonical(dict(plan_sha256=c.sha(c.canonical(diagnostic)))))
    # Deployment bytes were reviewed separately. This seam preserves both real
    # status producers, stored claims and artifact validators; no remote I/O.
    monkeypatch.setattr(c.diag, "verify_source", lambda plan: None)
    queue = dict(returncode=0, stdout="", stderr="")
    account = copy.deepcopy(capture["status"]["accounting"])
    monkeypatch.setattr(c.diag, "job_queue", lambda job: queue)
    monkeypatch.setattr(c.diag, "job_accounting", lambda job: account)
    return SimpleNamespace(
        plan={"diagnostic_plan": diagnostic},
        failed=failed,
        queue=queue,
        account=account,
        mapped=mapped,
        diagnostic=diagnostic,
    )


def test_real_failed_recovery_and_import_producers_to_v2_gate(c, actual_prerequisites):
    result = c.prerequisite_status(actual_prerequisites.plan)
    assert result["failed"]["state"] == "FAILED" and result["failed"]["startup_passed"] is False
    assert result["diagnostic"]["success"] is result["diagnostic"]["imports_ready"] is True
    assert result["diagnostic"]["job_id"] == "54643629"
    assert all(result["diagnostic"][flag] is False for flag in c.FLAGS)


@pytest.mark.parametrize(
    "mutation",
    [
        "failed_claim",
        "diagnostic_claim",
        "failed_publication",
        "diagnostic_publication",
        "cached_queue",
        "failed_step",
    ],
)
def test_rehashed_recovery_prerequisites_cannot_borrow_other_result(c, actual_prerequisites, mutation):
    p = actual_prerequisites
    if mutation == "failed_claim":
        p.mapped(p.failed.execution["recovery_claim"]).write_text("{}")
    elif mutation == "diagnostic_claim":
        p.mapped(p.diagnostic["claim"]).write_text("{}")
    elif mutation == "failed_publication":
        p.mapped(Path(p.failed.failed["result_dir"]) / "execution/startup.log").write_text("wrong")
    elif mutation == "diagnostic_publication":
        p.mapped(Path(p.diagnostic["result_dir"]) / "A2-report.json").write_text("{}")
    elif mutation == "cached_queue":
        p.queue["stdout"] = "54643629|COMPLETED"
    else:
        p.account["stdout"] = p.account["stdout"].replace(".batch|COMPLETED|0:0", ".batch|FAILED|1:0")
    with pytest.raises((ValueError, KeyError)):
        c.prerequisite_status(p.plan)


@pytest.fixture
def ready_execution_publication(c, plan, node, tmp_path, monkeypatch):
    """Real subprocess/trace publisher and real startup validators; synthetic CUDA/science."""
    node_tests = load(ROOT / "tests/sdsc/test_student_order_execution_v2_job.py")
    setup = node_tests.setup.__wrapped__(tmp_path, monkeypatch)
    meta = node_tests.run(setup)
    assert meta["probe_passed"] is True
    _, _, _, child_plan, identity, source, _ = setup
    # This CPU seam uses its actual interpreter and fixture wrapper location.
    # No acceptance resolver, GPU execution or scientific result is fabricated.
    plan["science_plan"].update(child_plan["science_plan"])
    work = source.parent
    args = ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(work / "artifacts")]
    worker_tests = load(ROOT / "tests/sdsc/test_student_order_execution_worker.py")
    worker = worker_tests.setup.__wrapped__(monkeypatch)[0]
    for rank in (0, 1):
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        report = worker.startup("run", identity["job_id"], identity["cuda_visible_devices"])
        report["original_argv"] = args
        (source / f"rank-{rank}-startup.json").write_bytes(c.canonical(report))
        (source / f"rank-{rank}-exit.json").write_bytes(
            c.canonical(
                dict(
                    schema="quest-sdsc-student-order-execution-exit-v1",
                    job_id=identity["job_id"],
                    rank=rank,
                    original_worker_sha256=c.FROZEN_WORKER_SHA,
                    original_argv=args,
                    worker_invoked=True,
                    exit_code=0,
                    **dict.fromkeys(c.FLAGS, False),
                )
            )
        )
    Path(plan["science_plan"]["result_dir"]).mkdir(parents=True)
    result = dict(
        job_id=identity["job_id"],
        plan_sha256=c.sha(c.canonical(plan)),
        cuda_visible_devices=identity["cuda_visible_devices"],
        startup_passed=True,
        scientific_stage_complete=True,
        work_dir=str(work),
        science_publication_sha256="7" * 64,
        **dict.fromkeys(c.FLAGS, False),
    )
    return SimpleNamespace(source=source, result=result, meta=meta)


def test_real_trace_publication_keeps_original_three_startup_and_science_gates(
    c, plan, node, ready_execution_publication, monkeypatch
):
    p = ready_execution_publication
    node.publish_execution(plan, c, p.source, p.result)
    assert c.execution_publication(plan, "12345")["startup_passed"] is True
    scientific_status = dict(
        success=True,
        stage_complete=True,
        preparation_complete=False,
        publication_verified=True,
        publication_sha256="7" * 64,
    )
    monkeypatch.setattr(c.science, "inspect_job", lambda *a, **kw: dict(scientific_status))
    assert c.inspect_job(plan, {"job_id": "12345"})["success"] is True
    scientific_status["publication_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="bind scientific publication"):
        c.inspect_job(plan, {"job_id": "12345"})


@pytest.mark.parametrize(
    "mutation", ["missing_trace", "wrong_job", "wrong_argv", "wrong_rank", "raw_log", "event_clock"]
)
def test_ready_publication_rejects_missing_or_cross_bound_evidence(
    c, plan, node, ready_execution_publication, mutation
):
    p = ready_execution_publication
    trace_path = p.source / "trace-meta.json"
    if mutation == "missing_trace":
        trace_path.unlink()
    elif mutation == "raw_log":
        (p.source / "startup.log").write_bytes(b"unrelated log\n")
    elif mutation == "wrong_rank":
        path = p.source / "rank-1-startup.json"
        report = json.loads(path.read_bytes())
        report["rank"] = 0
        path.write_bytes(c.canonical(report))
    else:
        meta = json.loads(trace_path.read_bytes())
        if mutation == "wrong_job":
            meta["job_id"] = "54321"
        elif mutation == "wrong_argv":
            meta["argv"][-1] = "other visibility"
        else:
            for event in meta["adapter_events"]:
                event["monotonic_seconds"] += 10000
        trace_path.write_bytes(c.canonical(meta))
    # Recompute the enclosing publication through the actual publisher so these
    # cases exercise semantic cross-binding rather than merely stale file hashes.
    node.publish_execution(plan, c, p.source, p.result)
    with pytest.raises(ValueError):
        c.execution_publication(plan, "12345")


def test_deployment_requires_prior_accepted_execution_artifact(c, plan, monkeypatch):
    contract = c.helper("sdsc_student_order_execution_v2_contract")
    expected = {
        **plan["execution"]["control_file_sha256"],
        **contract.FROZEN_DEPENDENCIES,
        c.CONTRACT_PATH: plan["execution"]["artifact_sha256"],
        contract.PREVIOUS_EXECUTION["contract_path"]: contract.PREVIOUS_EXECUTION["artifact_sha256"],
    }
    rows = {name: {"sha256": digest} for name, digest in expected.items()}
    monkeypatch.setattr(c.science, "verify_source", lambda *a: {})
    monkeypatch.setattr(c.science, "manifest_records", lambda _: rows)
    assert c.verify_source(plan) == {}
    del rows[contract.PREVIOUS_EXECUTION["contract_path"]]
    with pytest.raises(ValueError, match="execution controls"):
        c.verify_source(plan)
