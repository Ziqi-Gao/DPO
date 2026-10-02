"""No-network transport contracts for one frozen prepared-student qualification."""

from __future__ import annotations

import base64
import copy
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c(monkeypatch):
    original = subprocess.run

    def guarded(argv, *args, **kwargs):
        command = argv[0] if isinstance(argv, list | tuple) else argv.split()[0]
        if Path(command).name in {"ssh", "sbatch", "sacct", "squeue", "scontrol", "srun", "scancel"}:
            pytest.fail("CPU test attempted remote/Slurm execution")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    return load("sdsc_student_qualify")


@pytest.fixture
def n():
    return load("sdsc_student_qualify_job")


def bind(c, plan):
    plan["science_identity"] = c.science_identity(plan["protocol"])
    intent = c.execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(c.CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(c.CONTROL / "student-qualify-submissions" / intent),
        claim=str(c.CONTROL / "student-qualify-claims" / (intent + ".json")),
        result_dir=str(c.PROJECT / "student-qualify" / intent),
        job_name="opd-squ-" + intent,
    )
    plan["scientific_claim"] = str(c.scientific_claim_path(plan))


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    pins = dict.fromkeys(c.TOOLS, "a" * 64)
    pins.update(c.FROZEN)
    protocol = dict(
        protocol_sha256="b" * 64,
        artifact_sha256="a" * 64,
        implementation_commit="1" * 40,
        acceptance_commit="2" * 40,
        head="2" * 40,
        science_file_sha256=dict(pins),
    )
    # Upstream validator has its own immutable, tested contract. Isolate transport
    # graph shape here without requiring private 54GB historical execution files.
    fit = dict(
        mode="fit",
        python=str(c.PROJECT / "env/bin/python3.12"),
        hf_home=str(c.PROJECT / "cache"),
        protocol=dict(protocol_sha256="d" * 64),
        resolved_config=dict(size=7923, sha256="c" * 64),
    )
    monkeypatch.setattr(c, "FIT_PLAN_SHA", c.sha(c.canonical(fit)))
    monkeypatch.setattr(c._prep, "validate_plan", lambda p: p)
    selected = dict(step=4, checkpoint_sha256=c.PREPARED_SHA)
    audit = dict(
        raw_replay_passed=True,
        job_id=c.FIT_JOB,
        plan_sha256=c.FIT_PLAN_SHA,
        publication_sha256=c.FIT_PUBLICATION_SHA,
        prompts_replayed=512,
        responses_replayed=2048,
        selected_checkpoint=selected,
        development=[],
        **dict.fromkeys(c.FLAGS, False),
    )
    raw = c.canonical(audit)
    monkeypatch.setattr(c, "FIT_AUDIT_SHA", c.sha(raw))
    value = dict(
        schema="quest-sdsc-student-qualification-plan-v1",
        task=c.TASK,
        run_id="qualification-fixture",
        code_sha256="c" * 64,
        manifest_sha256="d" * 64,
        created_at="2026-10-02T00:00:00Z",
        resources=c.resources(),
        protocol=protocol,
        control_sha256=pins,
        fit=dict(
            plan=fit,
            plan_sha256=c.FIT_PLAN_SHA,
            job_id=c.FIT_JOB,
            publication_sha256=c.FIT_PUBLICATION_SHA,
            report_sha256=c.FIT_REPORT_SHA,
            audit=c.audit_payload(raw),
        ),
        prepared_checkpoint=dict(c.PREPARED_ROW),
        python=fit["python"],
        hf_home=fit["hf_home"],
        provenance=dict(
            directory=str(c.CONTROL / "provenance" / ("e" * 64)), manifest_sha256="e" * 64, head="2" * 40
        ),
        **dict.fromkeys(c.FLAGS, False),
    )
    bind(c, value)
    c.validate_plan(value)
    return value


def good_report(c, plan):
    names = [
        "entity_symbol_renaming",
        "fact_order_permutation",
        "rule_order_permutation",
        "surface_template_paraphrase",
        "distractor_count_ood",
    ]
    anti = dict(
        model_checkpoint_hash=c.PREPARED_SHA,
        iid_example_count=128,
        transformed_case_count=640,
        transformation_accuracy=dict.fromkeys(names, 0.5),
        iid_accuracy=0.5,
        transformed_accuracy=0.5,
        shortcut_gap=0.0,
        max_shortcut_gap=0.05,
        minimum_iid_accuracy=0.10,
        minimum_transformed_accuracy=0.08,
        minimum_per_transformation_accuracy=0.05,
        capability_passed=True,
        passed=True,
        dataset_hash="ebb0fc1c7ef05334632f5e7bdf8b798d3e9f0b253357b37f22f5e4b1456852f3",
        suite_hash="c5ed272d5fe8f6dd21a7743dfca44d3bdfa06db44499842b7cc44df807b8ced0",
    )
    anti["sha256"] = c.sha(c.canonical(anti))
    rows = {name: dict(path=name, size=2, sha256=c.sha(b"{}")) for name in c.NAMES[1:3]}
    report = dict(
        schema="quest-sdsc-student-qualification-report-v1",
        task=c.TASK,
        job_id="123",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        preparation_protocol_sha256=plan["fit"]["plan"]["protocol"]["protocol_sha256"],
        prepared_checkpoint_sha256=c.PREPARED_SHA,
        execution_complete=True,
        parameters_unchanged=True,
        raw_record_count=896,
        expected_record_count=896,
        prompt_count=896,
        passed=True,
        qualification_passed=True,
        base_gate_passed=True,
        anti_shortcut_passed=True,
        validation=dict(
            num_examples=128,
            answer_correct=128,
            proof_correct=128,
            format_valid=128,
            answer_accuracy=1.0,
            exact_proof_accuracy=1.0,
            format_validity=1.0,
            passed=True,
        ),
        anti_shortcut=anti,
        iid_examples_sha256=anti["dataset_hash"],
        suite_sha256=anti["suite_hash"],
        raw_artifacts=list(rows.values()),
        loading=dict(
            prepared_checkpoint_sha256=c.PREPARED_SHA,
            exact_saved_master_reload=dict(
                all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True
            ),
            forward_parameter_dtype="torch.bfloat16",
            all_parameters_frozen=True,
            gradient_checkpointing=False,
            use_cache=False,
            master_model_state_sha256="b0e00866275772f91b8df132cce7b5628fb6879125dffcce775a5d6939cc80fe",
            forward_model_state_sha256="f" * 64,
        ),
        generation=dict(
            max_new_tokens=256,
            do_sample=False,
            use_cache=False,
            explicit_attention_mask=False,
            explicit_autocast=False,
            precision="native_bfloat16",
            truncation=False,
            seed=42,
            validation_max_model_input_length=1536,
            anti_shortcut_max_model_input_length=2244,
        ),
        gpu=dict(
            name="NVIDIA H100",
            logical_index=0,
            cpu_threads=24,
            torch="2.8.0+cu128",
            cuda_visible_devices="GPU-7",
        ),
        fit_evidence=dict(
            fit_job_id=c.FIT_JOB,
            publication_sha256=c.FIT_PUBLICATION_SHA,
            report_sha256=c.FIT_REPORT_SHA,
            independent_audit_sha256=c.FIT_AUDIT_SHA,
            selected_checkpoint=c.decode_audit(plan["fit"]["audit"])[1]["selected_checkpoint"],
        ),
        scientific_binding={
            k: plan["protocol"][k]
            for k in ("head", "implementation_commit", "acceptance_commit", "science_file_sha256")
        },
        **dict.fromkeys(c.FLAGS, False),
    )
    report["scientific_binding"]["prereg_commit"] = "3" * 40
    return report, rows


def test_resources_and_base_has_no_upper_threshold(c, plan):
    assert c.resources()["gpus"] == 1 and c.resources()["time"] == "02:00:00"
    argv = c.sbatch(plan)
    assert "--gpus=h100:1" in argv and "--mem=196608M" in argv and "--cpus-per-task=24" in argv
    assert "--no-requeue" in argv and "--export=NONE" in argv and "--constraint=lustre" not in argv
    report, rows = good_report(c, plan)
    assert c.validate_worker_report(report, plan, "123", rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "canonical_grpo"),
        ("student_accepted", True),
        ("formal_initial_accepted", True),
        ("execution_class_certified", True),
        ("result_dir", "/tmp/model"),
        ("python", "/bin/python"),
        ("job_name", "retry"),
        ("intent_id", "a" * 32),
    ],
)
def test_plan_rejects_scope_or_binding(c, plan, field, value):
    plan[field] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize("change", ["step8", "publication", "audit", "self_acceptance", "helper", "fit_plan"])
def test_frozen_prerequisite_rejections(c, plan, change):
    if change == "step8":
        plan["prepared_checkpoint"]["path"] = "checkpoints/step-00000008.pt"
    elif change == "publication":
        plan["fit"]["publication_sha256"] = "f" * 64
    elif change == "audit":
        plan["fit"]["audit"]["base64"] = base64.b64encode(b"{}").decode()
    elif change == "self_acceptance":
        plan["protocol"]["acceptance_commit"] = plan["protocol"]["implementation_commit"]
    elif change == "fit_plan":
        plan["fit"]["plan"]["mode"] = "preflight"
    else:
        plan["control_sha256"]["tools/sdsc_student_quality_job.py"] = "f" * 64
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "change",
    [
        "raw_count",
        "prompt_count",
        "missing_loading",
        "bf16_before_load",
        "dtype",
        "model_mutated",
        "generation",
        "acceptance",
        "old_floor",
        "wrong_transform",
        "base_flag",
        "anti_flag",
        "master_state",
        "wrong_suite",
        "raw_missing",
        "parent",
    ],
)
def test_report_tampering_rejected(c, plan, change):
    report, rows = good_report(c, plan)
    if change == "raw_count":
        report["raw_record_count"] = 895
    elif change == "prompt_count":
        report["prompt_count"] = 128
    elif change == "missing_loading":
        report.pop("loading")
    elif change == "bf16_before_load":
        report["loading"]["exact_saved_master_reload"]["loaded_before_bf16_copy"] = False
    elif change == "dtype":
        report["loading"]["forward_parameter_dtype"] = "torch.float32"
    elif change == "model_mutated":
        report["parameters_unchanged"] = False
    elif change == "generation":
        report["generation"]["anti_shortcut_max_model_input_length"] = 1600
    elif change == "acceptance":
        report["g0_passed"] = True
    elif change == "old_floor":
        report["anti_shortcut"]["minimum_iid_accuracy"] = 0.01
    elif change == "wrong_transform":
        report["anti_shortcut"]["transformation_accuracy"]["made_up"] = report["anti_shortcut"][
            "transformation_accuracy"
        ].pop("distractor_count_ood")
    elif change == "base_flag":
        report["base_gate_passed"] = False
    elif change == "anti_flag":
        report["anti_shortcut_passed"] = False
    elif change == "master_state":
        report["loading"]["master_model_state_sha256"] = "a" * 64
    elif change == "wrong_suite":
        report["anti_shortcut"]["suite_hash"] = "e" * 64
    elif change == "parent":
        report["fit_evidence"]["independent_audit_sha256"] = "f" * 64
    else:
        report["raw_artifacts"].pop()
    with pytest.raises(ValueError):
        c.validate_worker_report(report, plan, "123", rows)


def test_negative_gap_is_accepted_with_capability(c, plan):
    report, rows = good_report(c, plan)
    anti = report["anti_shortcut"]
    anti["iid_accuracy"] = 0.25
    anti["shortcut_gap"] = -0.25
    anti["sha256"] = c.sha(c.canonical({k: v for k, v in anti.items() if k != "sha256"}))
    assert c.validate_worker_report(report, plan, "123", rows)


def test_failed_qualification_remains_complete_and_fetchable(c, n, plan, tmp_path):
    report, rows = good_report(c, plan)
    report["validation"].update(
        answer_correct=12,
        proof_correct=12,
        answer_accuracy=12 / 128,
        exact_proof_accuracy=12 / 128,
        passed=False,
    )
    report.update(base_gate_passed=False, qualification_passed=False, passed=False)
    assert not c.validate_worker_report(report, plan, "123", rows, require_passed=False)
    with pytest.raises(ValueError, match="failed qualification"):
        c.validate_worker_report(report, plan, "123", rows)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "qualification-report.json").write_bytes(c.canonical(report))
    for name in c.NAMES[1:3]:
        (artifacts / name).write_bytes(b"{}")
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)
    result = dict(
        job_id="123",
        stage_complete=False,
        qualification_passed=False,
        exit_code=1,
        **dict.fromkeys(c.FLAGS, False),
    )
    n.persist(plan, c, artifacts, None, result, {}, lambda phase: None)
    publication = c.document(destination / "receipt.json")
    assert publication["passed"] is False and set(c.publication_records(publication, plan, "123")) == set(
        c.NAMES
    )
    files = {p.name: base64.b64encode(p.read_bytes()).decode() for p in destination.iterdir()}
    fetched = dict(
        receipt=c.make_receipt(plan, "123"),
        files=files,
        bytes=sum(len(base64.b64decode(v)) for v in files.values()),
        status=dict(
            publication_verified=True,
            publication=publication,
            publication_sha256=c.sha(c.read(destination / "receipt.json")),
        ),
    )
    assert c.validate_download(fetched, plan)["qualification-records.jsonl"] == b"{}"


def test_worker_cwd_cuda_single_process(c, n, plan, tmp_path, monkeypatch):
    capture = {}
    monkeypatch.setattr(n.subprocess, "Popen", lambda argv, **kwargs: capture.update(argv=argv, **kwargs))
    env = {"CUDA_VISIBLE_DEVICES": "GPU-uuid7"}
    n.start_worker(
        plan, tmp_path / "source", tmp_path / "science", tmp_path / "inputs", tmp_path / "outputs", env, None
    )
    assert capture["cwd"] == tmp_path / "science" and capture["env"] == env and capture["start_new_session"]
    assert capture["argv"][1:4] == ["-I", "-B", "-u"] and "--num_processes" not in capture["argv"]


@pytest.mark.parametrize("visible", ["GPU-7", "0", "MIG-valid-id"])
def test_allocation_preserves_assigned_visible_value(c, n, plan, monkeypatch, visible):
    values = dict(
        SLURM_JOB_ID="123",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="196608",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
        CUDA_VISIBLE_DEVICES=visible,
    )
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    assert n.allocation(plan, c)["cuda_visible_devices"] == visible
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError):
        n.allocation(plan, c)


def test_scientific_claim_stable_across_control_change(c, plan):
    before = plan["scientific_claim"]
    intent = plan["intent_id"]
    plan["control_sha256"]["tools/sdsc_student_qualify_worker.py"] = "9" * 64
    plan["protocol"]["science_file_sha256"]["tools/sdsc_student_qualify_worker.py"] = "9" * 64
    bind(c, plan)
    assert plan["scientific_claim"] == before and plan["intent_id"] != intent
    c.validate_plan(plan)


def test_gpu_ceiling_one_new_gpu(c, monkeypatch):
    monkeypatch.setattr(c, "run", lambda argv: dict(returncode=0, stdout="100|RUNNING|N/A\n", stderr=""))
    monkeypatch.setattr(c._initial, "live_gpu_request", lambda *a, **k: dict(allocatable_gpus=3))
    assert c.check_gpu_ceiling()["new_gpus"] == 1
    monkeypatch.setattr(c._initial, "live_gpu_request", lambda *a, **k: dict(allocatable_gpus=4))
    with pytest.raises(ValueError, match="four allocatable"):
        c.check_gpu_ceiling()


def memory(c, job="123"):
    limit = 192 * 1024**3
    peak = 40 * 1024**3
    path = "/sys/fs/cgroup/slurm/job_" + job
    evidence = dict(
        passed=True,
        expected_limit_bytes=limit,
        limit_bytes=limit,
        path=path,
        peak_bytes=peak,
        current_bytes=peak,
        headroom_bytes=limit - peak,
        minimum_headroom_bytes=max(32 * 1024**3, __import__("math").ceil(limit * 0.2)),
        observed_at_unix=10.0,
        ancestors=[
            dict(path=path, limit_bytes=limit, peak_bytes=peak, current_bytes=peak, memory_events={"oom": 0})
        ],
    )
    return {
        phase: copy.deepcopy(evidence)
        for phase in ["initial", "after_staging", "after_inference", "final", "after_publication"]
    }


def test_exact192_memory_headroom_ownjob_and_publication(c):
    report = memory(c)
    c.validate_memory(report, "123")
    report["after_publication"]["ancestors"][0]["memory_events"]["oom"] = 1
    with pytest.raises(ValueError, match="nonzero"):
        c.validate_memory(report, "123")
    with pytest.raises(ValueError):
        c.validate_memory(memory(c, "999"), "123")
    report = memory(c)
    report.pop("after_publication")
    with pytest.raises(ValueError):
        c.validate_memory(report, "123")


def test_accounting_requires_every_step_and_exact_resources(c, plan):
    row = [
        "123",
        "COMPLETED",
        "0:0",
        plan["job_name"],
        "nwu181",
        "nairr-gpu-shared",
        "nairr-gpu-shared-normal",
        "24",
        "192Gn",
        "00:30:00",
        "cpu=24,gres/gpu=1,node=1",
        plan["intent_id"],
    ]
    queue = dict(returncode=0, stdout="", stderr="")

    def account(rows):
        return dict(returncode=0, stdout="\n".join("|".join(r) for r in rows), stderr="")

    rows = [row, ["123.batch", "COMPLETED", "0:0"], ["123.extern", "COMPLETED", "0:0"]]
    assert c.validate_accounting(plan, "123", queue, account(rows))["accounting_complete"]
    rows[1][2] = "1:0"
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account(rows))
    rows[1][2] = "0:0"
    row[10] = "cpu=24,gres/gpu=2,node=1"
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account(rows))


def test_remote_entrypoint_refuses_local_slurm(c):
    result = subprocess.run(
        [os.sys.executable, "-I", "-B", str(ROOT / "tools/sdsc_student_qualify.py"), "remote"],
        input=b"{}",
        capture_output=True,
    )
    assert result.returncode != 0 and b"Slurm operations cannot run on Quest" in result.stderr


def test_launcher_uses_absolute_release_not_slurm_spool(c, plan, tmp_path):
    script = c.job_script(plan).decode()
    assert str(Path(plan["release"]) / "source/tools/sdsc_student_qualify_job.py") in script
    assert "dirname" not in script and "CUDA_VISIBLE_DEVICES=" not in script
    path = tmp_path / "job.sh"
    path.write_text(script)
    assert subprocess.run(["/bin/bash", "-n", str(path)]).returncode == 0


def test_submission_requires_matching_dry_run_before_claim(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda _: None)
    monkeypatch.setattr(c, "admission", lambda _: pytest.fail("must reject before admission"))
    with pytest.raises(ValueError, match="matching dry-run"):
        c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=None))
    assert not Path(plan["scientific_claim"]).exists()


def test_unknown_submission_claim_is_permanent(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda _: None)
    monkeypatch.setattr(c, "admission", lambda _: {})
    monkeypatch.setattr(c, "run", lambda argv: dict(returncode=1, stdout="", stderr="transport lost"))
    proof = dict(dry_run=True, blockers=[], plan_sha256=c.sha(c.canonical(plan)), argv=c.sbatch(plan))
    with pytest.raises(ValueError, match="acknowledgement"):
        c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=proof))
    assert Path(plan["scientific_claim"]).exists() and Path(plan["claim"]).exists()
    assert (Path(plan["submission_dir"]) / "unknown.json").exists()
    assert not (Path(plan["submission_dir"]) / "receipt.json").exists()


def test_scientific_claim_stops_racing_submit(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda _: None)
    monkeypatch.setattr(c, "admission", lambda _: {})
    path = Path(plan["scientific_claim"])
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    monkeypatch.setattr(c, "run", lambda *_: pytest.fail("must not submit"))
    proof = dict(dry_run=True, blockers=[], plan_sha256=c.sha(c.canonical(plan)), argv=c.sbatch(plan))
    with pytest.raises(FileExistsError):
        c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=proof))


def test_completed_own_jobs_use_live_tres_helper(c, monkeypatch):
    seen = []
    monkeypatch.setattr(c, "run", lambda _: dict(returncode=0, stdout="100|COMPLETED|N/A\n", stderr=""))

    def helper(job, user, gres, queue_state):
        seen.append((job, gres, queue_state))
        return dict(allocatable_gpus=0)

    monkeypatch.setattr(c._initial, "live_gpu_request", helper)
    assert c.check_gpu_ceiling()["existing_allocatable_gpus"] == 0
    assert seen == [("100", "N/A", "COMPLETED")]


def test_fetch_never_accepts_checkpoint_bytes(c, plan):
    receipt = c.make_receipt(plan, "123")
    value = dict(
        receipt=receipt,
        files={"checkpoints/model.pt": base64.b64encode(b"weights").decode()},
        bytes=7,
        status={"publication_verified": False},
    )
    with pytest.raises(ValueError):
        c.validate_download(value, plan)


def test_memory_own_job_oom_rejected(c):
    raw = {"ancestors": [{"path": "/slurm/job_123/step_batch", "memory_events": {"oom": 1}}]}
    with pytest.raises(ValueError, match="nonzero"):
        c.validate_job_memory_events(raw, "123")


def test_deployed_worker_must_match_reviewed_implementation(c, plan):
    plan["control_sha256"]["tools/sdsc_student_qualify_worker.py"] = "8" * 64
    bind(c, plan)
    with pytest.raises(ValueError, match="accepted scientific byte"):
        c.validate_plan(plan)


def test_new_raw_read_bound_does_not_change_old_helper(c, tmp_path):
    path = tmp_path / "complete-raw.jsonl"
    raw = b"x" * (8 * c.CAP + 1)
    path.write_bytes(raw)
    assert c.read(path) == raw
    with pytest.raises(ValueError):
        c._prep._io.read(path)
    assert c.MAX_FILE == 16 * c.CAP and c.MAX_FETCH == 48 * c.CAP


def test_new_raw_read_still_rejects_above_bound(c, tmp_path):
    path = tmp_path / "oversized.jsonl"
    with path.open("wb") as stream:
        stream.truncate(c.MAX_FILE + 1)
    with pytest.raises(ValueError):
        c.read(path)


def test_new_json_read_bound_and_hash_keep_old_document_immutable(c, tmp_path):
    path = tmp_path / "memory.json"
    raw = b'{"ok":true}' + b" " * (8 * c.CAP)
    path.write_bytes(raw)
    assert c.document(path, c.sha(raw)) == {"ok": True}
    with pytest.raises(ValueError):
        c._prep._io.document(path)
    with pytest.raises(ValueError, match="SHA differs"):
        c.document(path, "a" * 64)


def test_unknown_submission_reconciles_exact_job_without_sbatch(c, plan, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda _: None)
    monkeypatch.setattr(c, "admission", lambda _: {})
    monkeypatch.setattr(c, "run", lambda argv: dict(returncode=1, stdout="", stderr="lost acknowledgement"))
    proof = dict(dry_run=True, blockers=[], plan_sha256=c.sha(c.canonical(plan)), argv=c.sbatch(plan))
    with pytest.raises(ValueError):
        c.remote_action(dict(plan=plan, action="submit", authorize=True, dry_run=proof))
    seen = []

    def query(argv):
        seen.append(argv[0])
        assert argv[0] in {"sacct", "squeue"}
        return dict(
            returncode=0, stdout="123|" + plan["job_name"] + "|" + plan["intent_id"] + "\n", stderr=""
        )

    monkeypatch.setattr(c, "run", query)
    monkeypatch.setattr(c, "live_binding", lambda p, j: dict(job_id=j))
    result = c.remote_action(dict(plan=plan, action="reconcile"))
    assert result["job_id"] == "123" and result["reconciled"] is True
    assert seen == ["squeue", "sacct"]
    assert c.document(Path(plan["scientific_claim"])) == c.scientific_claim_record(plan)


def test_zero_reconcile_matches_never_rearms_claim(c, plan, monkeypatch):
    directory = Path(plan["submission_dir"])
    directory.mkdir(parents=True)
    for path, value in [
        (directory / "plan.json", plan),
        (Path(plan["claim"]), dict(plan_sha256=c.sha(c.canonical(plan)))),
        (Path(plan["scientific_claim"]), c.scientific_claim_record(plan)),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(c.canonical(value))
    monkeypatch.setattr(c, "verify_source", lambda _: None)
    monkeypatch.setattr(c, "run", lambda argv: dict(returncode=0, stdout="", stderr=""))
    with pytest.raises(ValueError, match="zero/ambiguous"):
        c.remote_action(dict(plan=plan, action="reconcile"))
    assert Path(plan["claim"]).exists() and Path(plan["scientific_claim"]).exists()
    assert not (directory / "receipt.json").exists()


def test_local_manifest_and_all_named_science_cross_binding(c, plan, tmp_path):
    release = Path(plan["release"])
    source = release / "source"
    names = (*c.TOOLS, "src/scientific_noncontrol.py")
    rows = []
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        rows.append(dict(path=name, size=len(name), sha256=c.sha(name.encode()), mode=0o644))
    plan["control_sha256"] = {r["path"]: r["sha256"] for r in rows if r["path"] in c.TOOLS}
    plan["protocol"]["science_file_sha256"] = {r["path"]: r["sha256"] for r in rows}
    plan["code_sha256"] = c.sha(c.canonical(rows))
    manifest = dict(
        files=rows,
        code_sha256=plan["code_sha256"],
        run_id=plan["run_id"],
        total_bytes=sum(r["size"] for r in rows),
    )
    raw = c.canonical(manifest)
    (release / "manifest.json").write_bytes(raw)
    plan["manifest_sha256"] = c.sha(raw)
    assert c.verify_source(plan) == manifest
    plan["protocol"]["science_file_sha256"]["src/scientific_noncontrol.py"] = "0" * 64
    with pytest.raises(ValueError, match="accepted implementation"):
        c.verify_source(plan)


def test_model_cache_staging_only_native_pinned_revision(c, n, tmp_path):
    cache = tmp_path / "cache"
    dest = tmp_path / "staged"
    snap = cache / "hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(b"{}")
    (snap / "model.safetensors").write_bytes(b"model")
    (snap / "unexpected-secret").write_bytes(b"never-stage")
    assert n.stage_models(cache, dest, c)["files"] == 2
    assert not (dest / snap.relative_to(cache) / "unexpected-secret").exists()
    outside = tmp_path / "outside"
    outside.write_bytes(b"bad")
    (snap / "tokenizer.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escaped"):
        n.stage_models(cache, tmp_path / "another", c)
