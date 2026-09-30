"""Read-only fixture transport tests: no SSH, Slurm or GPU work."""

import base64
import copy
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


@pytest.fixture
def c():
    return module("sdsc_student_instruction")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    pins = {name: "a" * 64 for name in c.TOOLS}
    pins["tools/sdsc_student_quality.py"] = c.SHARED_SHA
    contract = c.helper("sdsc_student_contract")
    intent = c.execution_intent(pins)
    runtime = str(c.PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
    cache = str(c.PROJECT / "cache/huggingface")
    return dict(
        schema="quest-sdsc-student-instruction-plan-v1",
        task=c.TASK,
        scope="train-only-scientific-prompt-screen",
        resources=copy.deepcopy(c.RESOURCES),
        candidate_sha256=c.CANDIDATE_SHA,
        baseline_sha256=c.BASELINE_SHA,
        screen_document_sha256=pins[c.SCREEN_DOC],
        checkpoints=[
            dict(
                label="initial", path="artifacts/initial_checkpoint.pt", size=3441276375, sha256=c.INITIAL_SHA
            )
        ],
        resolved_config=dict(
            path="artifacts/canonical_sft/resolved_config.yaml", size=7923, sha256=c.CONFIG_SHA
        ),
        dataset_inputs=[
            dict(row, source=str(contract.DATASET_ROOT / row["path"]))
            for row in contract.DATASET_FILES
            if row["path"] in c.DATASET_NAMES
        ],
        run_id="run-1",
        intent_id=intent,
        code_sha256="b" * 64,
        control_sha256=pins,
        release=str(c.CONTROL / "releases/run-1"),
        created_at="2026-09-30T00:00:00Z",
        submission_dir=str(c.CONTROL / "student-instruction-submissions" / intent),
        claim=str(c.CONTROL / "student-instruction-claims" / (intent + ".json")),
        scientific_claim=str(c.scientific_claim_path()),
        result_dir=str(c.PROJECT / "student-instruction" / intent),
        job_name="opd-si-" + intent,
        python=runtime,
        hf_home=cache,
        parent=dict(
            report_sha256=c.PARENT_REPORT_SHA,
            receipt=dict(
                job_id=c.PARENT_JOB,
                intent_id=c.PARENT_INTENT,
                submission_dir=str(c.CONTROL / "submissions" / c.PARENT_INTENT),
                python=runtime,
                hf_home=cache,
            ),
        ),
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
    )


@pytest.mark.parametrize("key,value", [("gpus", 2), ("gpus", True), ("mem_gib", 384), ("cpus", 1)])
def test_resource_scope_is_fixed(c, plan, key, value):
    c.validate_plan(plan)
    plan["resources"][key] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("g0_passed", True),
        ("factorial_ready", True),
        ("result_dir", "/tmp/escape"),
        ("intent_id", "c" * 32),
        ("python", "/tmp/python"),
    ],
)
def test_scope_paths_identity_are_fixed(c, plan, key, value):
    plan[key] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_sbatch_is_safe_fixed_argv(c, plan):
    args = c.sbatch(plan)
    assert args[:2] == ["sbatch", "--parsable"]
    assert {"--no-requeue", "--gpus=h100:1", "--mem=65536M", "--export=NONE", "--time=00:30:00"} <= set(args)
    assert args[-1] == c.sha(c.canonical(plan))
    subprocess.run(["bash", "-n"], input=c.job_script(plan), check=True)


def stubs(c, monkeypatch):
    monkeypatch.setattr(c, "check_gpu_ceiling", lambda: {})
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    monkeypatch.setattr(c, "verify_parent", lambda p, **kw: None)


def test_unknown_ack_keeps_permanent_claim_and_never_retries(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    calls = []

    def lost(args):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, 45)

    monkeypatch.setattr(c, "run", lost)
    with pytest.raises(subprocess.TimeoutExpired):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert Path(plan["claim"]).exists()
    assert c.document(Path(plan["submission_dir"]) / "unknown.json")["no_retry"] is True
    with pytest.raises(ValueError, match="reconcile"):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert len(calls) == 1


def test_no_authorization_never_dispatches(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    monkeypatch.setattr(c, "run", lambda *args: pytest.fail("Slurm called"))
    with pytest.raises(ValueError, match="authorization"):
        c.remote_action(dict(action="submit", plan=plan, authorize=False))
    assert not Path(plan["claim"]).exists()


def test_receipt_precedes_live_binding_failure(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    monkeypatch.setattr(c, "run", lambda args: dict(returncode=0, stdout="12345;expanse\n", stderr=""))
    monkeypatch.setattr(
        c, "live_binding", lambda *args: (_ for _ in ()).throw(ValueError("lost live response"))
    )
    with pytest.raises(ValueError):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert c.document(Path(plan["submission_dir"]) / "receipt.json")["job_id"] == "12345"
    assert (Path(plan["submission_dir"]) / "unknown.json").exists()


def live(plan, c, job="123"):
    return dict(
        job_id=job,
        plan_sha256=c.sha(c.canonical(plan)),
        fields=dict(
            JobId=job,
            JobName=plan["job_name"],
            Account="nwu181",
            Partition="nairr-gpu-shared",
            QOS="nairr-gpu-shared-normal",
            Comment=plan["intent_id"],
            Command=str(Path(plan["submission_dir"]) / "job.sh"),
            WorkDir=plan["submission_dir"],
        ),
    )


def accounting(plan, comment=None):
    row = "|".join(
        [
            "123",
            "COMPLETED",
            "0:0",
            plan["job_name"],
            "nwu181",
            "nairr-gpu-shared",
            "nairr-gpu-shared-normal",
            "8",
            "64Gn",
            "200",
            "cpu=8,node=1,gres/gpu=1",
            plan["intent_id"] if comment is None else comment,
        ]
    )
    return dict(returncode=0, stdout="", stderr=""), dict(
        returncode=0, stdout=row + "\n123.batch|COMPLETED|0:0\n123.extern|COMPLETED|0:0\n", stderr=""
    )


def test_allocation_and_all_steps_required(c, plan):
    queue, account = accounting(plan)
    assert c.validate_accounting(plan, "123", queue, account)["accounting_complete"]
    for bad in [
        account["stdout"].replace("123.extern|COMPLETED|0:0\n", ""),
        account["stdout"].replace("123.batch|COMPLETED|0:0", "123.batch|FAILED|1:0"),
        account["stdout"].replace("gres/gpu=1", "gres/gpu=2"),
        account["stdout"].replace("64Gn", "384Gn"),
    ]:
        with pytest.raises(ValueError):
            c.validate_accounting(plan, "123", queue, {**account, "stdout": bad})


def test_empty_archived_comment_needs_exact_live_binding(c, plan):
    queue, account = accounting(plan, "")
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account)
    assert c.validate_accounting(plan, "123", queue, account, live(plan, c))["accounting_complete"]
    changed = live(plan, c)
    changed["fields"]["Command"] = "/tmp/other.sh"
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account, changed)
    queue, account = accounting(plan, "conflict")
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account, live(plan, c))


def test_disappearance_is_unknown(c, plan):
    empty = dict(returncode=0, stdout="", stderr="")
    value = c.validate_accounting(plan, "123", empty, empty)
    assert value["state"] == "UNKNOWN" and not value["accounting_complete"]


def test_reconcile_cannot_turn_no_match_into_submission(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    directory = Path(plan["submission_dir"])
    directory.mkdir(parents=True)
    Path(plan["claim"]).parent.mkdir(parents=True)
    c.write_once(Path(plan["claim"]), c.canonical(dict(plan_sha256=c.sha(c.canonical(plan)))))
    Path(plan["scientific_claim"]).parent.mkdir(parents=True, exist_ok=True)
    c.write_once(Path(plan["scientific_claim"]), c.canonical(c.scientific_claim_record(plan)))
    c.write_once(c.legacy_claim_path(), c.canonical(c.legacy_reservation(plan)))
    c.write_once(directory / "plan.json", c.canonical(plan))
    monkeypatch.setattr(c, "run", lambda args: dict(returncode=0, stdout="", stderr=""))
    with pytest.raises(ValueError, match="unknown"):
        c.remote_action(dict(action="reconcile", plan=plan))
    assert not (directory / "receipt.json").exists()


def test_read_rejects_symlink_and_oversize(c, tmp_path):
    target = tmp_path / "file"
    target.write_bytes(b"123")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        c.read(link)
    with pytest.raises(ValueError):
        c.read(target, 2)


def test_node_checks_actual_alloc_and_preserves_cuda(c, plan, monkeypatch):
    node = module("sdsc_student_instruction_job")
    values = dict(
        SLURM_JOB_ID="123",
        SLURM_CPUS_PER_TASK="8",
        SLURM_MEM_PER_NODE="65536",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
        CUDA_VISIBLE_DEVICES="GPU-abcd",
    )
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    assert node.allocation(plan, c)["cuda_visible_devices"] == "GPU-abcd"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError):
        node.allocation(plan, c)


def download(c, plan, *, success=True):
    raw = {name: b"{}" for name in c.NAMES}
    report = dict(
        passed=success,
        diagnostic_complete=success,
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
        job_id="123",
        parent_job_id=c.PARENT_JOB,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        raw_artifacts=[
            dict(path=n, size=len(raw[n]), sha256=c.sha(raw[n]))
            for n in ("instruction-records.jsonl", "instruction-prompts.jsonl")
        ],
    )
    report.update(
        schema="quest-sdsc-student-instruction-probe-v1",
        candidate_sha256=c.CANDIDATE_SHA,
        baseline_sha256=c.BASELINE_SHA,
        training_screen_viable=False,
        initial_identity=dict(
            checkpoint_sha256=c.INITIAL_SHA,
            size=3441276375,
            comparison_performed_before_load=True,
            pristine_comparison=dict(all_tensors_exact=True, key_count=310),
            embedding_tying_before={},
            embedding_tying_after={},
        ),
        only_instruction_changed=True,
        generation=dict(
            max_new_tokens=256,
            max_model_input_length=1536,
            do_sample=False,
            use_cache=False,
            truncation=False,
        ),
        raw_record_count=64,
        ordered_train_ids=[str(i) for i in range(32)],
        model_parameters_sha256_before="d" * 64,
        model_parameters_sha256_after="d" * 64,
        dataset_manifest_sha256=next(
            r["sha256"] for r in plan["dataset_inputs"] if r["path"] == "manifest.json"
        ),
        train_file_sha256=next(
            r["sha256"] for r in plan["dataset_inputs"] if r["path"] == "train/examples.jsonl"
        ),
        arms=[
            dict(
                arm=name,
                record_start=i * 32,
                record_count=32,
                metrics=dict(count=32, answer_accuracy=0.0, exact_proof_accuracy=0.0, format_validity=0.0),
                teacher_forced=dict(
                    examples=32,
                    response_tokens_including_eos=3000,
                    canonical_response_nll=1.0,
                    canonical_response_token_accuracy=0.5,
                ),
            )
            for i, name in enumerate(("baseline", "candidate"))
        ],
    )
    raw["instruction-probe.json"] = c.canonical(report)
    publication = dict(
        task=c.TASK,
        job_id="123",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=c.sha(c.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=success,
        diagnostic_complete=success,
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
        persistent_read_back_verified=True,
        files=[dict(path=n, size=len(data), sha256=c.sha(data)) for n, data in raw.items()],
    )
    raw["receipt.json"] = c.canonical(publication)
    return dict(
        receipt=c.make_receipt(plan, "123"),
        status=dict(
            success=success,
            publication=publication,
            publication_sha256=c.sha(raw["receipt.json"]),
            publication_verified=True,
            artifact_hashes_verified=True,
            evidence_state="verified_publication",
        ),
        files={n: base64.b64encode(data).decode() for n, data in raw.items()},
        bytes=sum(map(len, raw.values())),
    )


def test_download_rehashes_full_raw_after_remote_status(c, plan):
    value = download(c, plan)
    assert set(c.NAMES) <= set(c.validate_download(value, plan))
    value["files"]["instruction-records.jsonl"] = base64.b64encode(b"[]").decode()
    with pytest.raises(ValueError, match="downloaded scientific bytes"):
        c.validate_download(value, plan)


def test_download_rejects_missing_prompts_and_escape(c, plan):
    value = download(c, plan)
    del value["files"]["instruction-prompts.jsonl"]
    value["bytes"] -= 2
    with pytest.raises(ValueError, match="missing complete"):
        c.validate_download(value, plan)
    value = download(c, plan)
    value["files"]["../source"] = ""
    with pytest.raises(ValueError, match="unexpected fetched"):
        c.validate_download(value, plan)


def test_failed_download_requires_the_same_complete_receipt_hash_chain(c, plan):
    value = download(c, plan, success=False)
    assert set(c.NAMES) <= set(c.validate_download(value, plan))
    assert value["status"]["success"] is False
    value["files"]["instruction-records.jsonl"] = base64.b64encode(b"[]").decode()
    with pytest.raises(ValueError, match="downloaded scientific bytes"):
        c.validate_download(value, plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("task", "different-task"),
        ("job_id", "456"),
        ("run_id", "different-run"),
        ("intent_id", "f" * 32),
        ("plan_sha256", "f" * 64),
        ("code_sha256", "f" * 64),
        ("factorial_ready", True),
        ("g0_passed", True),
    ],
)
def test_failed_foreign_receipt_rejected_even_with_resealed_transport(c, plan, key, value):
    result = download(c, plan, success=False)
    publication = result["status"]["publication"]
    publication[key] = value
    old = base64.b64decode(result["files"]["receipt.json"])
    raw = c.canonical(publication)
    result["files"]["receipt.json"] = base64.b64encode(raw).decode()
    result["bytes"] += len(raw) - len(old)
    result["status"]["publication_sha256"] = c.sha(raw)
    with pytest.raises(ValueError, match="unbound persistent"):
        c.validate_download(result, plan)


def test_failed_missing_receipt_allows_only_explicitly_unverified_bounded_logs(c, plan):
    result = download(c, plan, success=False)
    result["status"] = dict(
        success=False,
        publication_verified=False,
        artifact_hashes_verified=False,
        evidence_state="unpublished_logs_only",
    )
    result["files"] = {"worker.log": base64.b64encode(b"failed before publication\n").decode()}
    result["bytes"] = len(b"failed before publication\n")
    assert set(c.validate_download(result, plan)) == {"worker.log"}
    result["files"]["instruction-records.jsonl"] = base64.b64encode(b"{}").decode()
    result["bytes"] += 2
    with pytest.raises(ValueError, match="only bounded logs"):
        c.validate_download(result, plan)
    del result["files"]["instruction-records.jsonl"]
    raw = b"x" * (64 * 1024 + 1)
    result["files"]["worker.log"] = base64.b64encode(raw).decode()
    result["bytes"] = len(raw)
    with pytest.raises(ValueError, match="tail bound"):
        c.validate_download(result, plan)


def stage_failed_publication(c, plan, monkeypatch, *, receipt=True):
    value = download(c, plan, success=False)
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)
    for name, encoded in value["files"].items():
        if name != "receipt.json" or receipt:
            (destination / name).write_bytes(base64.b64decode(encoded))
    (destination / "worker.log").write_bytes(b"failed before the second checkpoint\n")
    directory = Path(plan["submission_dir"])
    directory.mkdir(parents=True)
    c.write_once(directory / "plan.json", c.canonical(plan))
    c.write_once(directory / "receipt.json", c.canonical(c.make_receipt(plan, "123")))
    Path(plan["claim"]).parent.mkdir(parents=True)
    c.write_once(Path(plan["claim"]), c.canonical(dict(plan_sha256=c.sha(c.canonical(plan)))))
    Path(plan["scientific_claim"]).parent.mkdir(parents=True, exist_ok=True)
    c.write_once(Path(plan["scientific_claim"]), c.canonical(c.scientific_claim_record(plan)))
    c.write_once(c.legacy_claim_path(), c.canonical(c.legacy_reservation(plan)))
    queue, account = accounting(plan)
    account["stdout"] = account["stdout"].replace("COMPLETED|0:0", "FAILED|1:0")
    monkeypatch.setattr(c, "run", lambda args: queue if args[0] == "squeue" else account)
    monkeypatch.setattr(c, "verify_source", lambda plan: None)
    return destination


def test_failed_remote_status_and_fetch_verify_publication_before_return(c, plan, monkeypatch):
    destination = stage_failed_publication(c, plan, monkeypatch)
    observed = c.inspect_job(plan, c.make_receipt(plan, "123"))
    assert observed["state"] == "FAILED" and observed["success"] is False
    assert observed["publication_verified"] and observed["artifact_hashes_verified"]
    result = c.remote_action(dict(action="fetch", plan=plan))
    assert set(c.NAMES) <= set(c.validate_download(result, plan))
    (destination / "instruction-records.jsonl").write_bytes(b"[]")
    with pytest.raises(ValueError, match="persistent diagnostic hash"):
        c.inspect_job(plan, c.make_receipt(plan, "123"))


def test_failed_data_change_after_status_is_rejected_by_remote_fetch(c, plan, monkeypatch):
    destination = stage_failed_publication(c, plan, monkeypatch)
    original = c.inspect_job

    def changed(plan, receipt):
        result = original(plan, receipt)
        (destination / "instruction-records.jsonl").write_bytes(b"[]")
        return result

    monkeypatch.setattr(c, "inspect_job", changed)
    with pytest.raises(ValueError, match="data changed during fetch"):
        c.remote_action(dict(action="fetch", plan=plan))


def test_no_receipt_remote_fetch_omits_orphan_scientific_files(c, plan, monkeypatch):
    stage_failed_publication(c, plan, monkeypatch, receipt=False)
    result = c.remote_action(dict(action="fetch", plan=plan))
    assert result["status"]["evidence_state"] == "unpublished_logs_only"
    assert result["status"]["artifact_hashes_verified"] is False
    assert set(c.validate_download(result, plan)) == {"worker.log"}


def test_failed_inventory_missing_extra_and_duplicate_files_fail_closed(c, plan):
    original = download(c, plan, success=False)
    missing = copy.deepcopy(original)
    del missing["files"]["instruction-prompts.jsonl"]
    missing["bytes"] -= 2
    with pytest.raises(ValueError, match="scientific inventory"):
        c.validate_download(missing, plan)
    for bad in (
        original["status"]["publication"]["files"] * 2,
        [dict(path="other.json", size=0, sha256=c.sha(b""))],
    ):
        publication = {**original["status"]["publication"], "files": bad}
        with pytest.raises(ValueError, match="publication inventory"):
            c.publication_records(publication, plan, "123")


def test_real_parent_checkpoint_envelope(c):
    folder = ROOT / ".sdsc/fetched/54548846/fetch-rhidqpza"
    if not folder.exists():
        pytest.skip("private actual run evidence absent")
    checkpoints, config = c.checkpoint_rows(
        c.document(folder / "receipt.json"),
        c.document(folder / "adapted-calibration.json", c.PARENT_REPORT_SHA),
    )
    assert [r["label"] for r in checkpoints] == ["initial"]
    assert checkpoints[0]["sha256"] == c.INITIAL_SHA
    assert config["path"] == "artifacts/canonical_sft/resolved_config.yaml"


def live_inventory(job="54558773", *, state="RUNNING", gpus=1, typed=False):
    gpu = "" if gpus == 0 else f",gres/gpu={gpus}"
    if typed and gpus:
        gpu += f",gres/gpu:h100={gpus}"
    tres = "cpu=24,mem=192G,node=1,billing=4800" + gpu
    return (
        f"JobId={job} UserId=zgao12(543540) JobState={state} NumNodes=1 NumTasks=1 "
        f"ReqTRES={tres} "
        + (f"AllocTRES={tres} " if state == "RUNNING" else "")
        + (f"TresPerJob=gres:gpu:h100:{gpus}" if gpus else "")
    )


def inventory_stubs(c, monkeypatch, queue, live):
    monkeypatch.setattr(c.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="zgao12"))

    def run(argv):
        if argv[0] == "squeue":
            return dict(returncode=0, stdout=queue, stderr="")
        assert argv[:4] == ["scontrol", "show", "job", "--oneliner"]
        return dict(returncode=0, stdout=live[argv[-1]], stderr="")

    monkeypatch.setattr(c, "run", run)


def test_actual_missing_queue_gres_counts_one_h100(c, monkeypatch):
    # Exact real 2026-09-30 fields: --gpus produced N/A for %b, but TRES carries one GPU.
    queue = "54558773|RUNNING|N/A\n"
    live = live_inventory()
    inventory_stubs(c, monkeypatch, queue, {"54558773": live})
    actual = c.check_gpu_ceiling()
    assert actual["existing_requested_gpus"] == 1
    assert actual["jobs"][0]["authoritative_fields"]["ReqTRES"].endswith("gres/gpu=1")


def test_private_full_actual_slurm_snapshot(c, monkeypatch):
    path = ROOT / ".sdsc/diagnostics/student-quality-v2/gpu-concurrency-field-evidence.json"
    if not path.is_file():
        pytest.skip("private live field evidence absent")
    actual = json.loads(path.read_text())
    inventory_stubs(c, monkeypatch, actual["queue"]["stdout"], {"54558773": actual["job"]["stdout"]})
    assert c.check_gpu_ceiling()["existing_requested_gpus"] == 1


@pytest.mark.parametrize(
    "state,gpus,typed", [("RUNNING", 0, False), ("PENDING", 3, False), ("RUNNING", 2, True)]
)
def test_cpu_only_pending_and_typed_total_not_double_counted(c, monkeypatch, state, gpus, typed):
    inventory_stubs(
        c, monkeypatch, f"1|{state}|N/A\n", {"1": live_inventory("1", state=state, gpus=gpus, typed=typed)}
    )
    assert c.check_gpu_ceiling()["existing_requested_gpus"] == gpus


def test_expanded_array_identity_and_total(c, monkeypatch):
    live = live_inventory("17") + " ArrayJobId=10 ArrayTaskId=0"
    inventory_stubs(c, monkeypatch, "10_0|RUNNING|N/A\n", {"10_0": live})
    assert c.check_gpu_ceiling()["existing_requested_gpus"] == 1


@pytest.mark.parametrize(
    "kind",
    [
        "missing_req",
        "missing_alloc",
        "foreign",
        "job",
        "conflict",
        "typed_conflict",
        "bad_gpu",
        "bad_perjob",
        "queue_conflict",
        "excess",
        "ambiguous_array",
        "duplicate_job",
        "incomplete_cpu",
        "unknown_state",
    ],
)
def test_unknown_inconsistent_or_excess_gpu_inventory_fails(c, monkeypatch, kind):
    queue, live = "1|RUNNING|N/A\n", live_inventory("1")
    if kind == "missing_req":
        live = live.replace("ReqTRES=", "UnknownTRES=")
    elif kind == "missing_alloc":
        live = live.replace("AllocTRES=", "UnknownTRES=")
    elif kind == "foreign":
        live = live.replace("UserId=zgao12", "UserId=other")
    elif kind == "job":
        live = live.replace("JobId=1 ", "JobId=2 ")
    elif kind == "conflict":
        live = live.replace(
            "AllocTRES=cpu=24,mem=192G,node=1,billing=4800,gres/gpu=1", "AllocTRES=cpu=24,node=1,gres/gpu=2"
        )
    elif kind == "typed_conflict":
        live = live.replace("gres/gpu=1", "gres/gpu=1,gres/gpu:h100=2")
    elif kind == "bad_gpu":
        live = live.replace("gres/gpu=1", "gres/gpu=1,unknown/gpu=2")
    elif kind == "bad_perjob":
        live = live.replace("gres:gpu:h100:1", "gres:gpu:h100:1,gres:gpu:bad")
    elif kind == "queue_conflict":
        queue = "1|RUNNING|gpu:h100:2\n"
    elif kind == "excess":
        live = live_inventory("1", gpus=4)
    elif kind == "ambiguous_array":
        queue = "1_[0-3]|PENDING|N/A\n"
    elif kind == "duplicate_job":
        queue *= 2
    elif kind == "incomplete_cpu":
        live = "JobId=1 UserId=zgao12(1) JobState=PENDING ReqTRES=mem=32G"
    else:
        live = live.replace("RUNNING", "UNKNOWN")
    inventory_stubs(c, monkeypatch, queue, {"1": live})
    with pytest.raises(ValueError):
        c.check_gpu_ceiling()


@pytest.mark.parametrize("kind", ["candidate", "checkpoint", "dataset", "helper", "document"])
def test_screen_is_fixed_initial_train_only(c, plan, kind):
    if kind == "candidate":
        plan["candidate_sha256"] = "e" * 64
    elif kind == "checkpoint":
        plan["checkpoints"].append(dict(plan["checkpoints"][0], label="step20"))
    elif kind == "dataset":
        plan["dataset_inputs"].append(dict(plan["dataset_inputs"][1], path="validation/examples.jsonl"))
    elif kind == "helper":
        plan["control_sha256"]["tools/sdsc_student_quality.py"] = "e" * 64
    else:
        plan["screen_document_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "flag",
    [
        "student_accepted",
        "g0_passed",
        "pilot_passed",
        "factorial_ready",
        "teacher_accepted",
        "teacher_accepted_under_candidate",
        "accepted_science",
        "formal_prompt_accepted",
    ],
)
def test_every_scientific_flag_remains_false(c, plan, flag):
    plan[flag] = True
    with pytest.raises(ValueError, match="acceptance"):
        c.validate_plan(plan)


def worker_report(c, plan):
    downloaded = download(c, plan)
    report = json.loads(base64.b64decode(downloaded["files"]["instruction-probe.json"]))
    records = {row["path"]: row for row in downloaded["status"]["publication"]["files"]}
    return report, records


def test_viable_screen_does_not_change_acceptance(c, plan):
    report, records = worker_report(c, plan)
    report["arms"][1]["metrics"].update(answer_accuracy=4 / 32, exact_proof_accuracy=4 / 32)
    report["training_screen_viable"] = True
    c.validate_worker_report(report, plan, "123", records)
    assert all(report[key] is False for key in c.FLAGS)
    report["arms"][0]["metrics"]["answer_accuracy"] = 4 / 32
    with pytest.raises(ValueError, match="futility"):
        c.validate_worker_report(report, plan, "123", records)


@pytest.mark.parametrize("kind", ["pristine", "tying", "weights", "count", "cap", "nll", "viability"])
def test_report_rejects_incomplete_or_changed_screen(c, plan, kind):
    report, records = worker_report(c, plan)
    if kind == "pristine":
        report["initial_identity"]["pristine_comparison"]["all_tensors_exact"] = False
    elif kind == "tying":
        report["initial_identity"]["embedding_tying_after"] = {"changed": True}
    elif kind == "weights":
        report["model_parameters_sha256_after"] = "e" * 64
    elif kind == "count":
        report["raw_record_count"] = 63
    elif kind == "cap":
        report["generation"]["max_new_tokens"] = 128
    elif kind == "nll":
        report["arms"][0]["teacher_forced"]["canonical_response_nll"] = float("nan")
    else:
        report["training_screen_viable"] = True
    with pytest.raises(ValueError):
        c.validate_worker_report(report, plan, "123", records)


def test_old_quality_publication_cannot_be_borrowed(c, plan):
    old = ROOT / ".sdsc/student-quality/d47f1122872cad16aeb8e772f2e37ea2/fetch-186dm58f/receipt.json"
    if not old.is_file():
        pytest.skip("private historical evidence absent")
    with pytest.raises(ValueError, match="unbound"):
        c.publication_records(c.document(old), plan, "54557365")


def revised_execution(c, plan):
    changed = copy.deepcopy(plan)
    changed["control_sha256"][c.TOOLS[0]] = "e" * 64
    intent = c.execution_intent(changed["control_sha256"])
    old = changed["intent_id"]
    changed["intent_id"] = intent
    for key in ("submission_dir", "claim", "result_dir", "job_name"):
        changed[key] = changed[key].replace(old, intent)
    return changed


def test_execution_revision_preserves_scientific_subject(c, plan):
    changed = revised_execution(c, plan)
    c.validate_plan(changed)
    assert changed["intent_id"] != plan["intent_id"]
    assert changed["scientific_claim"] == plan["scientific_claim"]
    identity = c.science_identity()
    assert identity["initial_sha256"] == c.INITIAL_SHA
    assert identity["generation"]["max_new_tokens"] == 256
    assert "first32-train" in identity["population"]
    assert "without-explicit-autocast" in identity["precision"]


def test_unknown_submission_blocks_another_execution_version(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    calls = []

    def lost(argv):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 45)

    monkeypatch.setattr(c, "run", lost)
    with pytest.raises(subprocess.TimeoutExpired):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    winner = c.document(Path(plan["scientific_claim"]))
    assert winner == c.scientific_claim_record(plan)
    assert c.document(c.legacy_claim_path()) == c.legacy_reservation(plan)
    changed = revised_execution(c, plan)
    with pytest.raises(ValueError, match="winning intent"):
        c.remote_action(dict(action="submit", plan=changed, authorize=True))
    assert len(calls) == 1 and not Path(changed["claim"]).exists()


@pytest.mark.parametrize("kind", ["claim", "directory"])
def test_legacy_unknown_intent_cannot_be_bypassed(c, plan, monkeypatch, kind):
    stubs(c, monkeypatch)
    monkeypatch.setattr(c, "run", lambda argv: pytest.fail("Slurm submit occurred"))
    if kind == "claim":
        path = c.legacy_claim_path()
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    else:
        (c.CONTROL / "student-instruction-submissions" / c.LEGACY_INTENT).mkdir(parents=True)
    with pytest.raises(ValueError, match="legacy"):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert not Path(plan["scientific_claim"]).exists()


def test_cross_version_exclusive_claim_race_never_submits(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    original = c.write_once

    def write(path, raw):
        if path == c.legacy_claim_path():
            path.write_text("legacy runner won")
        return original(path, raw)

    monkeypatch.setattr(c, "write_once", write)
    monkeypatch.setattr(c, "run", lambda argv: pytest.fail("Slurm submit occurred"))
    with pytest.raises(FileExistsError):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert Path(plan["scientific_claim"]).exists()
    assert not Path(plan["claim"]).exists()
    with pytest.raises(ValueError, match="claimed"):
        c.remote_action(dict(action="submit", plan=revised_execution(c, plan), authorize=True))
