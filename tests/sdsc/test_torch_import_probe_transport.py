"""Import-only transport boundaries; Slurm is forbidden in every CPU fixture.

Recorded-run seams replay local immutable 54643463 bytes when present. Empty
queue in those seams is an explicit synthetic later accounting observation, not
an assertion that the historical fetched queue was empty.
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
INTENT = "44340f6e52b16b20d59538170b588576"
FETCH = ROOT / ".sdsc/fetched/54643463/fetch-tcvymoay"
OUTER = ROOT / ".sdsc/student-order-execution" / INTENT / "plan.json"


def load(path):
    spec = importlib.util.spec_from_file_location("_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c(monkeypatch):
    original = subprocess.run

    def guarded(argv, *args, **kwargs):
        if isinstance(argv, list | tuple) and Path(argv[0]).name in {
            "ssh",
            "sbatch",
            "sacct",
            "squeue",
            "scontrol",
            "srun",
            "scancel",
        }:
            pytest.fail("fixture attempted remote/Slurm command")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    return load(ROOT / "tools/sdsc_torch_import_probe.py")


def build_plan(c, failed, execution):
    pins = {
        name: c.sha((ROOT / name).read_bytes()) if (ROOT / name).is_file() else "a" * 64 for name in c.TOOLS
    }
    value = dict(
        schema=c.SCHEMA,
        task=c.TASK,
        run_id="import-fixture",
        code_sha256="c" * 64,
        manifest_sha256="d" * 64,
        created_at="2026-10-04T06:00:00Z",
        control_sha256=pins,
        resources=copy.deepcopy(c.RESOURCES),
        failed=dict(
            job_id=c.FAILED_JOB,
            plan=failed,
            plan_sha256=c.FAILED_PLAN_SHA,
            publication_sha256=c.FAILED_PUBLICATION_SHA,
            report_sha256=c.FAILED_REPORT_SHA,
            execution_plan=execution,
            execution_plan_sha256=c.FAILED_EXECUTION_PLAN_SHA,
            execution_publication_sha256=c.FAILED_EXECUTION_PUBLICATION_SHA,
        ),
        python=c.PYTHON,
        scope=copy.deepcopy(c.SCOPE),
        **dict.fromkeys(c.FLAGS, False),
    )
    c.bind_paths(value)
    return c.validate_plan(value)


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    failed = dict(mode="preflight", protocol=dict(science_file_sha256={}))
    execution = dict(science_plan=failed)
    monkeypatch.setattr(c, "FAILED_PLAN_SHA", c.sha(c.canonical(failed)))
    monkeypatch.setattr(c, "FAILED_EXECUTION_PLAN_SHA", c.sha(c.canonical(execution)))
    monkeypatch.setattr(c._prior, "validate_plan", lambda value: value)
    monkeypatch.setattr(c._recovery, "validate_plan", lambda value: value)
    return build_plan(c, failed, execution)


@pytest.fixture
def recorded(c, tmp_path, monkeypatch):
    if not OUTER.is_file() or not (FETCH / "fetch-manifest.json").is_file():
        pytest.skip("immutable local 54643463 evidence is absent")
    execution = json.loads(OUTER.read_bytes())
    failed = execution["science_plan"]
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    # No immutable constants or prior validators are replaced in this seam.
    value = build_plan(c, failed, execution)
    return SimpleNamespace(
        plan=value,
        execution=execution,
        failed=failed,
        report=json.loads((FETCH / "node-result.json").read_bytes()),
        publication=json.loads((FETCH / "receipt.json").read_bytes()),
        execution_publication=json.loads((FETCH / "execution/receipt.json").read_bytes()),
        capture=json.loads((FETCH / "fetch-manifest.json").read_bytes()),
    )


def test_actual_failed_report_and_paired_plan_schema(c, recorded):
    raw = (FETCH / "node-result.json").read_bytes()
    assert raw == c.canonical(recorded.report)  # Actual producer has no trailing newline.
    assert c.sha(raw) == c.FAILED_REPORT_SHA
    assert "plan_sha256" in recorded.execution_publication
    assert "execution_plan_sha256" not in recorded.execution_publication
    c.validate_failed_evidence(recorded.report, recorded.publication, recorded.execution_publication)
    c.validate_plan(recorded.plan)


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("report", "job_id", "54623814"),
        ("report", "error", "ValueError: two assigned H100s required"),
        ("report", "exit_code", 0),
        ("report", "student_accepted", True),
        ("publication", "job_id", "54623814"),
        ("publication", "plan_sha256", "a" * 64),
        ("publication", "passed", True),
        ("execution_publication", "job_id", "54626913"),
        ("execution_publication", "plan_sha256", "a" * 64),
        ("execution_publication", "startup_passed", True),
    ],
)
def test_failed_evidence_rejects_cross_job_or_fabricated_success(c, recorded, target, key, value):
    getattr(recorded, target)[key] = value
    with pytest.raises((ValueError, KeyError)):
        c.validate_failed_evidence(recorded.report, recorded.publication, recorded.execution_publication)


@pytest.fixture
def restored_failure(c, recorded, tmp_path, monkeypatch):
    mirror = tmp_path / "remote"
    prefixes = ("/home/zgao12/quest-runs/OPD/", "/expanse/lustre/projects/nwu181/zgao12/OPD/")

    def mapped(path):
        path = Path(path)
        return mirror / str(path).lstrip("/") if str(path).startswith(prefixes) else path

    def write(path, value):
        path = mapped(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(c.canonical(value))

    inner, outer = recorded.failed, recorded.execution
    directory = Path(inner["submission_dir"])
    result = Path(inner["result_dir"])
    for name in (
        "receipt.json",
        "node-result.json",
        "memory.json",
        "execution/receipt.json",
        "execution/execution-node.json",
        "execution/startup.log",
    ):
        target = mapped(result / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((FETCH / name).read_bytes())
    receipt = recorded.capture["receipt"]
    write(directory / "receipt.json", receipt)
    write(directory / "plan.json", inner)
    write(directory / "execution-plan.json", outer)
    write(
        directory / "execution-receipt.json",
        dict(c._recovery.claim_record(outer), job_id=c.FAILED_JOB, scientific_receipt=receipt),
    )
    # The historical sacct row has a blank Comment. Supply a synthetic saved
    # live-binding fixture through the original validator, without querying Slurm.
    write(
        directory / "live-binding.json",
        dict(
            job_id=c.FAILED_JOB,
            plan_sha256=c.sha(c.canonical(inner)),
            fields=dict(
                JobId=c.FAILED_JOB,
                JobName=inner["job_name"],
                Account="nwu181",
                Partition="nairr-gpu-shared",
                QOS="nairr-gpu-shared-normal",
                Comment=inner["intent_id"],
                Command=str(directory / "job.sh"),
                WorkDir=str(directory),
            ),
        ),
    )
    write(inner["claim"], {"plan_sha256": c.sha(c.canonical(inner))})
    write(outer["recovery_claim"], c._recovery.claim_record(outer))
    # Redirect filesystem reads only. The real frozen recovery/science status
    # producers parse the original report, receipt and accounting bytes below.
    for module in (c, c._prior, c._recovery, c._recovery.science):
        old_read = module.read
        monkeypatch.setattr(module, "safe", mapped)
        monkeypatch.setattr(
            module, "read", lambda path, *a, _old=old_read, **kw: _old(mapped(path), *a, **kw)
        )
    account = copy.deepcopy(recorded.capture["status"]["accounting"])
    queue = dict(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(c._recovery.science, "job_queue", lambda job: queue)
    monkeypatch.setattr(c._recovery.science, "job_accounting", lambda job: account)
    return SimpleNamespace(**vars(recorded), mapped=mapped, queue=queue, account=account, receipt=receipt)


def test_actual_recovery_status_producer_to_failed_prerequisite(c, restored_failure):
    status = c.verify_failed(restored_failure.plan)
    assert status["state"] == "FAILED" and status["success"] is False
    assert status["publication_verified"] is status["execution_publication_verified"] is True
    assert status["publication_sha256"] == c.FAILED_PUBLICATION_SHA
    assert status["execution_publication_sha256"] == c.FAILED_EXECUTION_PUBLICATION_SHA
    assert status["startup_passed"] is False


def test_cached_failed_queue_is_not_reconciled_admission(c, restored_failure):
    restored_failure.queue["stdout"] = "54643463|FAILED"
    with pytest.raises(ValueError, match="reconciled"):
        c.verify_failed(restored_failure.plan)


@pytest.mark.parametrize(
    "target", ["claim", "recovery_claim", "execution_receipt", "published_byte", "execution_byte"]
)
def test_actual_failure_rejects_claim_and_publication_tampering(c, restored_failure, target):
    f = restored_failure
    path = {
        "claim": f.failed["claim"],
        "recovery_claim": f.execution["recovery_claim"],
        "execution_receipt": str(Path(f.failed["submission_dir"]) / "execution-receipt.json"),
        "published_byte": str(Path(f.failed["result_dir"]) / "node-result.json"),
        "execution_byte": str(Path(f.failed["result_dir"]) / "execution/startup.log"),
    }[target]
    f.mapped(path).write_bytes(b"{}")
    with pytest.raises((ValueError, KeyError)):
        c.verify_failed(f.plan)


@pytest.mark.parametrize(
    "mutation", ["wrong_job", "completed", "failed_extern", "missing_batch", "query_error"]
)
def test_actual_failure_requires_exact_terminal_step_accounting(c, restored_failure, mutation):
    account = restored_failure.account
    if mutation == "wrong_job":
        account["stdout"] = account["stdout"].replace("54643463", "54623814")
    elif mutation == "completed":
        account["stdout"] = account["stdout"].replace("FAILED|1:0", "COMPLETED|0:0")
    elif mutation == "failed_extern":
        account["stdout"] = account["stdout"].replace(".extern|COMPLETED|0:0", ".extern|FAILED|1:0")
    elif mutation == "missing_batch":
        account["stdout"] = "\n".join(row for row in account["stdout"].splitlines() if ".batch|" not in row)
    else:
        account["returncode"] = 1
    with pytest.raises((ValueError, KeyError)):
        c.verify_failed(restored_failure.plan)


def test_fixed_resources_scope_and_no_placement(c, plan):
    argv = c.sbatch(plan)
    for item in ("--gpus=h100:2", "--cpus-per-task=24", "--mem=16384M", "--time=00:10:00", "--no-requeue"):
        assert item in argv
    assert not any(x.startswith(("--nodelist", "--exclude", "--constraint")) for x in argv)
    assert dict(training=False, model_loading=False, scientific_acceptance=False) == c.SCOPE
    assert c.MAX_FETCH == 8 * 1024**2


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "old"),
        ("task", "training"),
        ("python", "/bin/python"),
        ("claim", "/tmp/retry"),
        ("result_dir", "/tmp/out"),
        ("student_accepted", True),
        ("g0_passed", 0),
        ("execution_class_certified", 1),
    ],
)
def test_plan_rejects_scope_runtime_and_path_changes(c, plan, field, value):
    plan[field] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("gpus", 1),
        ("gpus", True),
        ("cpus", 4),
        ("mem_gib", 384),
        ("time", "00:05:00"),
        ("account", "other"),
        ("qos", "nairr-gpu-normal"),
    ],
)
def test_rebound_resource_changes_are_rejected(c, plan, key, value):
    plan["resources"][key] = value
    c.bind_paths(plan)
    with pytest.raises(ValueError, match="resources"):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "key",
    [
        "job_id",
        "plan_sha256",
        "report_sha256",
        "publication_sha256",
        "execution_plan_sha256",
        "execution_publication_sha256",
    ],
)
def test_rebound_prior_failure_changes_are_rejected(c, plan, key):
    plan["failed"][key] = "54623814" if key == "job_id" else "e" * 64
    c.bind_paths(plan)
    with pytest.raises(ValueError, match="failure"):
        c.validate_plan(plan)


def test_claim_does_not_reset_with_release_or_worker_change(c, plan):
    claim, intent = plan["claim"], plan["intent_id"]
    plan["run_id"] = "new-snapshot"
    plan["control_sha256"]["tools/sdsc_torch_import_probe_worker.py"] = "e" * 64
    c.bind_paths(plan)
    assert plan["claim"] == claim and plan["intent_id"] != intent


def accounting(plan, state="COMPLETED", code="0:0"):
    return dict(
        returncode=0,
        stderr="",
        stdout="\n".join(
            "|".join(
                [
                    "12345" + suffix,
                    "COMPLETED" if suffix == ".extern" else state,
                    "0:0" if suffix == ".extern" else code,
                    plan["job_name"],
                    "nwu181",
                    "nairr-gpu-shared",
                    "nairr-gpu-shared-normal",
                    "24",
                    "16G",
                    "500",
                    "cpu=24,gres/gpu=2,mem=16G,node=1",
                    plan["intent_id"],
                ]
            )
            for suffix in ("", ".batch", ".extern")
        ),
    )


@pytest.mark.parametrize(
    "state,code,complete", [("COMPLETED", "0:0", True), ("FAILED", "1:0", False), ("TIMEOUT", "0:15", False)]
)
def test_accounting_terminal_failure_is_not_completion(c, plan, state, code, complete):
    result = c.validate_accounting(
        plan, "12345", dict(returncode=0, stdout=""), accounting(plan, state, code)
    )
    assert result["terminal"] is True and result["accounting_complete"] is complete


@pytest.mark.parametrize(
    "before,after",
    [
        ("|24|16G|", "|4|16G|"),
        ("|24|16G|", "|24|384G|"),
        ("gres/gpu=2", "gres/gpu=1"),
        ("cpu=24", "cpu=4"),
        ("node=1", "node=2"),
    ],
)
def test_actual_diagnostic_allocation_must_match(c, plan, before, after):
    account = accounting(plan)
    account["stdout"] = account["stdout"].replace(before, after)
    with pytest.raises(ValueError, match="allocation"):
        c.validate_accounting(plan, "12345", dict(returncode=0, stdout=""), account)


def test_submit_requires_exact_dryrun_before_creating_claim(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    with pytest.raises(ValueError, match="matching dry-run"):
        c.remote_action(dict(action="submit", authorize=True, plan=plan, dry_run={}))
    assert not Path(plan["claim"]).exists()


def test_unknown_acknowledgement_consumes_claim_and_cannot_retry(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    monkeypatch.setattr(c, "admission", lambda p: {})
    calls = []
    monkeypatch.setattr(c, "run", lambda argv: calls.append(argv) or dict(returncode=0, stdout="", stderr=""))
    preview = dict(dry_run=True, blockers=[], plan_sha256=c.sha(c.canonical(plan)), argv=c.sbatch(plan))
    with pytest.raises(ValueError, match="acknowledgement"):
        c.remote_action(dict(action="submit", authorize=True, plan=plan, dry_run=preview))
    assert (Path(plan["submission_dir"]) / "unknown.json").is_file()
    assert Path(plan["claim"]).is_file() and len(calls) == 1
    with pytest.raises(FileExistsError):
        c.remote_action(dict(action="submit", authorize=True, plan=plan, dry_run=preview))
    assert len(calls) == 1


@pytest.mark.parametrize("matches", [[], ["12345", "12346"]])
def test_unknown_reconcile_never_rearms(c, plan, monkeypatch, matches):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    directory = Path(plan["submission_dir"])
    directory.mkdir(parents=True)
    Path(plan["claim"]).parent.mkdir(parents=True)
    c.write_once(directory / "plan.json", c.canonical(plan))
    c.write_once(plan["claim"], c.canonical(dict(plan_sha256=c.sha(c.canonical(plan)))))
    calls = []

    def query(argv):
        calls.append(argv)
        return dict(
            returncode=0,
            stderr="",
            stdout="\n".join(f"{job}|{plan['job_name']}|{plan['intent_id']}" for job in matches),
        )

    monkeypatch.setattr(c, "run", query)
    with pytest.raises(ValueError, match="zero/ambiguous"):
        c.remote_action(dict(action="reconcile", plan=plan))
    assert [x[0] for x in calls] == ["squeue", "sacct"]
    assert Path(plan["claim"]).exists() and not (directory / "receipt.json").exists()


def published(c, plan, payloads):
    return dict(
        task=c.TASK,
        job_id="12345",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=c.sha(c.canonical(plan)),
        code_sha256=plan["code_sha256"],
        persistent_read_back_verified=True,
        diagnostic_complete=False,
        imports_ready=False,
        cuda_observed=False,
        model_loaded=False,
        training_started=False,
        files=[dict(path=name, size=len(data), sha256=c.sha(data)) for name, data in payloads.items()],
        **dict.fromkeys(c.FLAGS, False),
    )


def download(c, plan, payloads):
    publication = published(c, plan, payloads)
    payloads = {**payloads, "receipt.json": c.canonical(publication)}
    return dict(
        receipt=c.make_receipt(plan, "12345"),
        status=dict(
            publication_verified=True,
            publication=publication,
            publication_sha256=c.sha(payloads["receipt.json"]),
        ),
        files={name: base64.b64encode(data).decode() for name, data in payloads.items()},
        bytes=sum(map(len, payloads.values())),
    )


def test_bounded_failed_diagnostic_download_retains_partial_arm_evidence(c, plan):
    payloads = {"A1.log": b"Timeout stack\n", "A1-report.json": b"{}", "A1-proc.jsonl": b"{}\n"}
    value = download(c, plan, payloads)
    assert c.validate_download(value, plan) == {
        **payloads,
        "receipt.json": c.canonical(value["status"]["publication"]),
    }


@pytest.mark.parametrize(
    "mutation", ["bytes", "hash", "missing", "path", "oversize", "bool_size", "duplicate"]
)
def test_download_rejects_changed_or_unsafe_publication(c, plan, mutation):
    value = download(c, plan, {"A1.log": b"trace"})
    if mutation == "bytes":
        value["bytes"] += 1
    elif mutation == "hash":
        value["files"]["A1.log"] = base64.b64encode(b"TRACE").decode()
    elif mutation == "missing":
        del value["files"]["A1.log"]
        value["bytes"] -= 5
    elif mutation == "path":
        value["files"]["../../bad"] = value["files"].pop("A1.log")
    else:
        pub = value["status"]["publication"]
        if mutation == "oversize":
            pub["files"][0]["size"] = c.MAX_FILE + 1
        elif mutation == "bool_size":
            pub["files"][0]["size"] = True
        else:
            pub["files"].append(dict(pub["files"][0]))
        raw = c.canonical(pub)
        value["status"]["publication_sha256"] = c.sha(raw)
        value["files"]["receipt.json"] = base64.b64encode(raw).decode()
        value["bytes"] = sum(len(base64.b64decode(x)) for x in value["files"].values())
    with pytest.raises((ValueError, KeyError)):
        c.validate_download(value, plan)


def process_evidence(c):
    sample = dict(
        pid=23456,
        label="A1",
        elapsed_seconds=0.05,
        observed_at_unix=1000.05,
        files={
            name: dict(text="Pid:\t23456\n" if name == "status" else "", truncated=False)
            for name in ("status", "stat", "io", "wchan")
        },
        environment=dict(
            OMP_NUM_THREADS=None, MKL_NUM_THREADS=None, OPENBLAS_NUM_THREADS=None, CUDA_VISIBLE_DEVICES="2,3"
        ),
        environment_truncated=False,
        task_count=1,
        task_count_truncated=False,
    )
    raw = c.canonical(sample) + b"\n"
    phase = dict(
        pid=23456,
        label="A1",
        elapsed_seconds=0.2,
        started_at_unix=1000,
        thread_environment=dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
        cuda_visible_devices="2,3",
        observation_complete=True,
        proc=dict(path="A1-proc.jsonl", size=len(raw), sha256=c.sha(raw), sample_count=1, truncated=False),
    )
    return phase, sample


def test_owned_process_raw_sample_validates(c):
    phase, sample = process_evidence(c)
    assert c.validate_proc_samples(phase, c.canonical(sample) + b"\n") == [sample]


@pytest.mark.parametrize(
    "mutation",
    [
        "pid",
        "label",
        "elapsed",
        "nan",
        "raw_pid",
        "environment",
        "count",
        "empty",
        "missing_file",
        "oversize",
    ],
)
def test_rehashed_process_samples_cannot_forge_observation(c, mutation):
    phase, sample = process_evidence(c)
    if mutation == "pid":
        sample["pid"] = 23457
    elif mutation == "label":
        sample["label"] = "B"
    elif mutation == "elapsed":
        sample["elapsed_seconds"] = 100
    elif mutation == "nan":
        sample["elapsed_seconds"] = float("nan")
    elif mutation == "raw_pid":
        sample["files"]["status"]["text"] = "Pid:\t23457\n"
    elif mutation == "environment":
        sample["environment"]["OMP_NUM_THREADS"] = "12"
    elif mutation == "missing_file":
        del sample["files"]["wchan"]
    elif mutation == "oversize":
        sample["files"]["wchan"]["text"] = "x" * 1537
    raw = (json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if mutation == "empty":
        raw = b""
        phase["proc"]["sample_count"] = 0
    elif mutation == "count":
        phase["proc"]["sample_count"] = 2
    phase["proc"].update(size=len(raw), sha256=c.sha(raw))
    with pytest.raises(ValueError):
        c.validate_proc_samples(phase, raw)


def test_recorded_prepare_verifies_actual_release_source(c, recorded, tmp_path, monkeypatch):
    value = recorded.plan
    release = Path(value["release"])
    source = release / "source"
    names = set(c.TOOLS) | set(recorded.failed["protocol"]["science_file_sha256"])
    rows = []
    for name in sorted(names):
        raw = (ROOT / name).read_bytes()
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        rows.append(dict(path=name, size=len(raw), sha256=c.sha(raw), mode=0o644))
    manifest = dict(
        files=rows,
        code_sha256=c.sha(c.canonical(rows)),
        run_id=value["run_id"],
        total_bytes=sum(row["size"] for row in rows),
    )
    (release / "manifest.json").write_bytes(c.canonical(manifest) + b"\n")
    monkeypatch.setattr(c, "ROOT", source)
    cli = SimpleNamespace(
        run_record=lambda run_id: dict(
            state="deployed", manifest=manifest, deployment=dict(code_sha256=manifest["code_sha256"])
        )
    )
    prepared = c.prepare(
        SimpleNamespace(run_id=value["run_id"], failed_plan=OUTER, failed_fetch_dir=FETCH), cli
    )
    actual = c.document(Path(prepared["plan"]))
    assert c.verify_source(actual) == manifest
    target = source / "tools/sdsc_torch_import_probe_worker.py"
    target.write_bytes(target.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="source bytes"):
        c.verify_source(actual)
