"""No-network transport contracts for one frozen prepared-student qualification."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "src/posttrain_circuits/experiments/protocols/student_focus_qualification.py"


@lru_cache
def selected_from_bytes(raw):
    from posttrain_circuits.experiments.protocols import student_focus_preparation as preparation

    return preparation.select_checkpoint(json.loads(raw))


@lru_cache
def development_sizes():
    from posttrain_circuits.experiments.protocols import student_focus_preparation as preparation

    bases = preparation._build_bases()["student_dev"]
    return {name: sum(x.metadata["structure"] == name for x in bases) for name in preparation.STRUCTURES}


def load(name):
    path = ROOT / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache
def core():
    spec = importlib.util.spec_from_file_location("focus_qualification_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def training_evidence():
    return dict(
        training_order_reconstructed=True,
        training_rows_reconstructed=2048,
        consumed_training_rows=2048,
        optimizer_windows_replayed=32,
        full_population_input_tokens=1665064,
        training_input_tokens=1665064,
        checkpoint_cumulative_input_tokens=dict(core().PARENT_CHECKPOINT_INPUT_TOKENS),
        gpu_training_independently_recomputed=False,
    )


def synthetic_completion(c, fit, candidate, qualification):
    """Explicit CPU fixture; separate producer seams exercise frozen validators."""
    publication = dict(
        schema="quest-sdsc-student-focus-publication-v1",
        task=c._fit.TASK,
        mode="fit",
        job_id=candidate["fit_job_id"],
        intent_id=fit["intent_id"],
        run_id=fit["run_id"],
        code_sha256=fit["code_sha256"],
        plan_sha256=candidate["fit_plan_sha256"],
        passed=True,
        stage_complete=True,
        preparation_complete=True,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        files=[dict(path="prepare-report.json", size=2, sha256=candidate["fit_report_sha256"])],
        large_files=[c.checkpoint_row(candidate)],
        **dict.fromkeys(c.FLAGS, False),
    )
    candidate["fit_publication_sha256"] = c.sha(c.canonical(publication))
    status = dict(
        accounting_complete=True,
        success=True,
        stage_complete=True,
        preparation_complete=True,
        publication_verified=True,
        artifact_hashes_verified=True,
        queue=dict(returncode=0, stdout="", stderr=""),
        accounting=dict(returncode=0, stdout="fixture accounting", stderr=""),
        state="COMPLETED",
        job_id=candidate["fit_job_id"],
        result_sha256=candidate["fit_report_sha256"],
        publication=publication,
        publication_sha256=candidate["fit_publication_sha256"],
        **dict.fromkeys(c.FLAGS, False),
    )
    return c.completion_records(status, c.canonical(publication))


@pytest.fixture
def c(monkeypatch):
    original = subprocess.run

    def guarded(argv, *args, **kwargs):
        command = argv[0] if isinstance(argv, list | tuple) else argv.split()[0]
        if Path(command).name in {"ssh", "sbatch", "sacct", "squeue", "scontrol", "srun", "scancel"}:
            pytest.fail("CPU test attempted remote/Slurm execution")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    module = load("sdsc_student_focus_qualify")
    monkeypatch.setattr(module, "qualification_domain", core)
    return module


@pytest.fixture
def n():
    return load("sdsc_student_focus_qualify_job")


def bind(c, plan):
    plan["science_identity"] = c.science_identity(plan["protocol"])
    intent = c.execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(c.CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(c.CONTROL / "student-focus-qualify-submissions" / intent),
        claim=str(c.CONTROL / "student-focus-qualify-claims" / (intent + ".json")),
        result_dir=str(c.PROJECT / "student-focus-qualify" / intent),
        job_name="opd-sfq-" + intent,
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
    from posttrain_circuits.experiments.protocols import student_focus_preparation as preparation

    qualification = core()

    parent = qualification.proposed_student_focus_qualification_protocol()["preserved_base"]
    fit = dict(
        schema="quest-sdsc-student-focus-plan-v1",
        run_id="actual-parent-fixture",
        code_sha256="d" * 64,
        task=c._fit.TASK,
        mode="fit",
        intent_id="9" * 32,
        python=str(c.PROJECT / "env/bin/python3.12"),
        hf_home=str(c.PROJECT / "cache"),
        protocol=dict(
            protocol_sha256=parent["parent_protocol_core_sha256"],
            artifact_sha256=parent["parent_student_protocol_sha256"],
            implementation_commit=parent["parent_implementation_commit"],
            acceptance_commit=parent["parent_acceptance_commit"],
            head="3" * 40,
            science_file_sha256={c._fit.PROTOCOL_MODULE: c.sha(c.read(ROOT / c._fit.PROTOCOL_MODULE))},
        ),
        control_sha256=dict(qualification.FROZEN_EXECUTION_SHA256),
        resolved_config=dict(size=7923, sha256="c" * 64),
        provenance=dict(head="3" * 40),
        **dict.fromkeys(c.FLAGS, False),
    )
    # Parent transport validation has separate full fixtures; use the real frozen
    # scientific selector here while isolating historical Slurm/filesystem state.
    monkeypatch.setattr(c._fit, "validate_plan", lambda p: p)
    monkeypatch.setattr(
        c._fit, "select_development", lambda p, rows: copy.deepcopy(selected_from_bytes(c.canonical(rows)))
    )
    sizes = development_sizes()
    development = []
    for step in preparation.CHECKPOINT_STEPS:
        structures = {
            name: dict(
                num_examples=size, answer_correct=size // 3, proof_correct=size // 3, format_valid=size
            )
            for name, size in sizes.items()
        }
        counts = {key: sum(row[key] for row in structures.values()) for key in preparation.COUNT_KEYS}
        views = {name: dict(counts) for name in preparation.DEV_VIEW_NAMES}
        if step == 1:
            views["fact_order_permutation"].update(answer_correct=0, proof_correct=0)
        development.append(
            dict(
                step=step,
                checkpoint_sha256=c.sha(str(step).encode()),
                examples_sha256=preparation.EXAMPLES_SHA256["student_dev"],
                **counts,
                per_view_counts=views,
                structure_counts=structures,
            )
        )
    selected = copy.deepcopy(selected_from_bytes(c.canonical(development)))
    candidate = dict(
        fit_job_id="12345",
        fit_intent=fit["intent_id"],
        fit_plan_sha256=c.sha(c.canonical(fit)),
        fit_source_head=fit["protocol"]["head"],
        fit_publication_sha256="e" * 64,
        fit_report_sha256="f" * 64,
        checkpoint_step=selected["step"],
        checkpoint_sha256=selected["checkpoint_sha256"],
        checkpoint_path=f"checkpoints/step-{selected['step']:08d}.pt",
        checkpoint_size=8_100_000_000,
        master_model_state_sha256="6" * 64,
        **qualification.PREPARED_INITIAL_POLICY,
    )
    completed = synthetic_completion(c, fit, candidate, qualification)
    audit = dict(
        schema="quest-sdsc-student-focus-raw-audit-v1",
        preparation_complete=True,
        dataset_sha256=preparation.DATASET_MANIFEST_SHA256,
        training_evidence=training_evidence(),
        isolation_evidence=dict(
            producer_evidence_sha256="4" * 64,
            dataset_manifest_sha256=preparation.DATASET_MANIFEST_SHA256,
            producer_isolation_consistency_verified=True,
            historical_isolation_independently_recomputed=False,
        ),
        raw_replay_passed=True,
        mode="fit",
        job_id=candidate["fit_job_id"],
        plan_sha256=candidate["fit_plan_sha256"],
        publication_sha256=candidate["fit_publication_sha256"],
        result_sha256=candidate["fit_report_sha256"],
        auditor_sha256=parent["parent_auditor_sha256"],
        protocol_sha256=fit["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=fit["protocol"]["artifact_sha256"],
        implementation_commit=fit["protocol"]["implementation_commit"],
        acceptance_commit=fit["protocol"]["acceptance_commit"],
        prompts_replayed=1536,
        responses_replayed=18432,
        execution_acceptance_claim=False,
        gpu_numerics_independently_recomputed=False,
        selected_checkpoint=selected,
        development=development,
        **dict.fromkeys(c.FLAGS, False),
    )
    raw = c.canonical(audit) + b"\n"
    candidate["independent_raw_audit_sha256"] = c.sha(raw)
    qualification.validate_prepared_initial(candidate)
    monkeypatch.setattr(
        c, "qualification_payload", lambda _: dict(prepared_initial=candidate, preserved_base=parent)
    )
    value = dict(
        schema="quest-sdsc-student-focus-qualification-plan-v1",
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
            plan_sha256=candidate["fit_plan_sha256"],
            job_id=candidate["fit_job_id"],
            publication_sha256=candidate["fit_publication_sha256"],
            report_sha256=candidate["fit_report_sha256"],
            audit=c.audit_payload(raw, candidate),
            completion=completed,
        ),
        prepared_checkpoint=c.checkpoint_row(candidate),
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
    candidate = c.prepared_initial(plan["protocol"])
    names = [
        "entity_symbol_renaming",
        "fact_order_permutation",
        "rule_order_permutation",
        "surface_template_paraphrase",
        "distractor_count_ood",
    ]
    anti = dict(
        model_checkpoint_hash=candidate["checkpoint_sha256"],
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
        schema="quest-sdsc-student-focus-qualification-report-v1",
        task=c.TASK,
        job_id="123",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        preparation_protocol_sha256=plan["fit"]["plan"]["protocol"]["protocol_sha256"],
        preparation_protocol_artifact_sha256=plan["fit"]["plan"]["protocol"]["artifact_sha256"],
        prepared_checkpoint_sha256=candidate["checkpoint_sha256"],
        prepared_initial=copy.deepcopy(candidate),
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
            prepared_checkpoint_sha256=candidate["checkpoint_sha256"],
            exact_saved_master_reload=dict(
                all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True
            ),
            forward_parameter_dtype="torch.bfloat16",
            all_parameters_frozen=True,
            gradient_checkpointing=False,
            use_cache=False,
            master_model_state_sha256=candidate["master_model_state_sha256"],
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
            fit_job_id=candidate["fit_job_id"],
            completion=c.completion_summary(plan["fit"]["completion"], plan["fit"]["plan"]),
            publication_sha256=candidate["fit_publication_sha256"],
            report_sha256=candidate["fit_report_sha256"],
            independent_audit_sha256=candidate["independent_raw_audit_sha256"],
            checkpoint_path=candidate["checkpoint_path"],
            checkpoint_size=candidate["checkpoint_size"],
            master_model_state_sha256=candidate["master_model_state_sha256"],
            selected_checkpoint=c.decode_audit(plan["fit"]["audit"], candidate, plan["fit"]["plan"])[1][
                "selected_checkpoint"
            ],
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
        ("task", "qwen3-v2-student-branch-qualification-v1"),
        ("schema", "quest-sdsc-student-branch-qualification-plan-v1"),
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
    plan["control_sha256"]["tools/sdsc_student_focus_qualify_worker.py"] = "9" * 64
    plan["protocol"]["science_file_sha256"]["tools/sdsc_student_focus_qualify_worker.py"] = "9" * 64
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
        [os.sys.executable, "-I", "-B", str(ROOT / "tools/sdsc_student_focus_qualify.py"), "remote"],
        input=b"{}",
        capture_output=True,
    )
    assert result.returncode != 0 and b"Slurm operations cannot run on Quest" in result.stderr


def test_launcher_uses_absolute_release_not_slurm_spool(c, plan, tmp_path):
    script = c.job_script(plan).decode()
    assert str(Path(plan["release"]) / "source/tools/sdsc_student_focus_qualify_job.py") in script
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
    plan["control_sha256"]["tools/sdsc_student_focus_qualify_worker.py"] = "8" * 64
    bind(c, plan)
    with pytest.raises(ValueError, match="accepted scientific byte"):
        c.validate_plan(plan)


def test_new_raw_read_bound_does_not_change_old_helper(c, tmp_path):
    path = tmp_path / "complete-raw.jsonl"
    raw = b"x" * (8 * c.CAP + 1)
    path.write_bytes(raw)
    assert c.read(path) == raw
    with pytest.raises(ValueError):
        c._fit._io.read(path)
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
        c._fit._io.document(path)
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


def parent_artifacts(c, plan, monkeypatch):
    candidate = c.prepared_initial(plan["protocol"])
    _, audit = c.decode_audit(plan["fit"]["audit"], candidate, plan["fit"]["plan"])
    checkpoints = [
        dict(
            step=row["step"],
            path=f"checkpoints/step-{row['step']:08d}.pt",
            size=candidate["checkpoint_size"],
            sha256=row["checkpoint_sha256"],
            reload_by_rank=[
                dict(rank=rank, model_state_sha256=candidate["master_model_state_sha256"]) for rank in (0, 1)
            ],
        )
        for row in audit["development"]
    ]
    report = dict(
        optimizer_steps=32,
        training_input_tokens=1665064,
        dataset_sha256=audit["dataset_sha256"],
        data_audit=dict(
            token_envelope=dict(
                fit_rows=2048,
                optimizer_windows=32,
                window_input_tokens=list(core().PARENT_WINDOW_INPUT_TOKENS),
                total_input_tokens=1665064,
                token_budget=2000000,
            )
        ),
        development=audit["development"],
        selected_checkpoint=audit["selected_checkpoint"],
        checkpoints=checkpoints,
    )
    publication = dict(
        large_files=[{key: row[key] for key in ("path", "size", "sha256")} for row in checkpoints]
    )
    # Separately tested complete parent validator; exercise new selection join here.
    monkeypatch.setattr(
        c._fit, "publication_records", lambda *args: {"prepare-data-isolation.json": dict(sha256="4" * 64)}
    )
    monkeypatch.setattr(c._fit, "validate_worker_report", lambda *args: True)
    return candidate, report, publication


def test_selected_checkpoint_lookup_is_not_first_or_fixed_four(c, plan, monkeypatch):
    candidate, report, publication = parent_artifacts(c, plan, monkeypatch)
    selected = c.validate_fit_evidence(
        report, publication, plan["fit"]["plan"], plan["fit"]["audit"], candidate
    )
    assert selected["step"] == 2 and report["checkpoints"][0]["step"] == 1
    assert selected["path"] == "checkpoints/step-00000002.pt"


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "sha",
        "path",
        "size",
        "rank_master",
        "ranks",
        "persistent",
        "tokens",
        "updates",
        "selected",
    ],
)
def test_parent_selected_join_fails_closed(c, plan, monkeypatch, change):
    candidate, report, publication = parent_artifacts(c, plan, monkeypatch)
    row = report["checkpoints"][1]
    if change == "missing":
        report["checkpoints"].pop(1)
    elif change == "duplicate":
        report["checkpoints"].append(copy.deepcopy(row))
    elif change == "sha":
        row["sha256"] = "0" * 64
    elif change == "path":
        row["path"] = "checkpoints/step-00000004.pt"
    elif change == "size":
        row["size"] += 1
    elif change == "rank_master":
        row["reload_by_rank"][1]["model_state_sha256"] = "7" * 64
    elif change == "ranks":
        row["reload_by_rank"].pop()
    elif change == "persistent":
        publication["large_files"][1]["sha256"] = "0" * 64
    elif change == "tokens":
        report["training_input_tokens"] -= 1
    elif change == "updates":
        report["optimizer_steps"] -= 1
    else:
        report["selected_checkpoint"] = None
    with pytest.raises(ValueError):
        c.validate_fit_evidence(report, publication, plan["fit"]["plan"], plan["fit"]["audit"], candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "preflight"),
        ("job_id", "54321"),
        ("plan_sha256", "0" * 64),
        ("publication_sha256", "0" * 64),
        ("result_sha256", "0" * 64),
        ("auditor_sha256", "0" * 64),
        ("protocol_sha256", "0" * 64),
        ("protocol_artifact_sha256", "0" * 64),
        ("implementation_commit", "0" * 40),
        ("acceptance_commit", "0" * 40),
        ("prompts_replayed", 512),
        ("responses_replayed", 2048),
        ("execution_acceptance_claim", True),
        ("gpu_numerics_independently_recomputed", True),
        ("raw_replay_passed", False),
        ("raw_replay_passed", 1),
        ("execution_acceptance_claim", 0),
        ("gpu_numerics_independently_recomputed", 0),
        ("prompts_replayed", 1536.0),
        ("selected_checkpoint", None),
        ("development", []),
    ],
)
def test_parent_audit_identity_rejections(c, plan, field, value):
    candidate = copy.deepcopy(c.prepared_initial(plan["protocol"]))
    raw, audit = c.decode_audit(plan["fit"]["audit"], candidate, plan["fit"]["plan"])
    audit[field] = value
    raw = c.canonical(audit) + b"\n"
    candidate["independent_raw_audit_sha256"] = c.sha(raw)
    with pytest.raises(ValueError):
        c.decode_audit(c.audit_payload(raw, candidate), candidate, plan["fit"]["plan"])


def test_parent_audit_requires_canonical_plus_lf(c, plan):
    candidate = copy.deepcopy(c.prepared_initial(plan["protocol"]))
    raw, _ = c.decode_audit(plan["fit"]["audit"], candidate, plan["fit"]["plan"])
    raw = raw.rstrip(b"\n")
    candidate["independent_raw_audit_sha256"] = c.sha(raw)
    with pytest.raises(ValueError, match="canonical JSON plus LF"):
        c.decode_audit(c.audit_payload(raw, candidate), candidate, plan["fit"]["plan"])


@pytest.mark.parametrize(
    "key", ["protocol_sha256", "artifact_sha256", "implementation_commit", "acceptance_commit", "head"]
)
def test_parent_lineage_cannot_be_self_consistent_other_release(c, plan, key):
    fit = copy.deepcopy(plan["fit"]["plan"])
    fit["protocol"][key] = "0" * len(fit["protocol"][key])
    with pytest.raises(ValueError, match="parent lineage"):
        c.validate_parent_binding(plan["protocol"], fit)


def test_parent_prompt_large_bound_is_separate_from_child(c, tmp_path, monkeypatch):
    path = tmp_path / "prepare-dev-prompts.jsonl"
    raw = b"x" * (16 * c.CAP + 1)
    path.write_bytes(raw)
    rows = {path.name: dict(path=path.name, size=len(raw), sha256=c.sha(raw))}
    monkeypatch.setattr(c._fit, "publication_records", lambda *args: rows)
    assert c.validate_parent_fetch(tmp_path, {}, {}, dict(fit_job_id="12345")) == rows
    with pytest.raises(ValueError):
        c.read(path)
    with path.open("wb") as stream:
        stream.truncate(32 * c.CAP + 1)
    with pytest.raises(ValueError):
        c.validate_parent_fetch(tmp_path, {}, {}, dict(fit_job_id="12345"))
    assert c._fit.MAX_FETCH == 224 * c.CAP and c.MAX_FETCH == 48 * c.CAP


def test_parent_total_raw_bound_enforced(c, tmp_path, monkeypatch):
    for name in ("prepare-dev-prompts.jsonl", "prepare-report.json"):
        (tmp_path / name).write_bytes(b"x" * 4)
    rows = {
        name: dict(path=name, size=4, sha256=c.sha(b"x" * 4))
        for name in ("prepare-dev-prompts.jsonl", "prepare-report.json")
    }
    monkeypatch.setattr(c._fit, "publication_records", lambda *args: rows)
    monkeypatch.setattr(c._fit, "MAX_FETCH", 7)
    with pytest.raises(ValueError, match="parent raw evidence"):
        c.validate_parent_fetch(tmp_path, {}, {}, dict(fit_job_id="12345"))


def test_worker_inputs_use_selected_neutral_path_and_parent_protocol(c, n, plan, tmp_path):
    work, science = tmp_path / "work", tmp_path / "science"
    for path in (
        work / "config/resolved_config.yaml",
        work / "evidence/fit-report.json",
        work / "evidence/fit-audit.json",
        *(
            work / "evidence" / ("fit-" + name.replace("_", "-") + ".json")
            for name in ("plan", *c.COMPLETION_NAMES)
        ),
        science / c._fit.PROTOCOL_PATH,
        science / c.PROTOCOL_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
    inputs = n.build_worker_inputs(plan, work, science, dict(job_id="123"), {}, c)
    assert len(inputs) == 14 and len(inputs["preparation_evidence"]) == 5
    assert inputs["prepared_checkpoint"]["path"] == str(work / "checkpoints/prepared-selected.pt")
    assert inputs["prepared_checkpoint"]["sha256"] == plan["prepared_checkpoint"]["sha256"]
    assert inputs["preparation_protocol"]["path"] == str(science / c._fit.PROTOCOL_PATH)


def test_bound_artifact_reader_rejects_missing_and_modified_candidate(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    path = tmp_path / c.PROTOCOL_PATH
    path.parent.mkdir(parents=True)
    payload = dict(prepared_initial=None, review={})
    raw = c.canonical(payload)
    path.write_bytes(raw)
    protocol = dict(
        artifact_sha256=c.sha(raw), protocol_sha256=c.sha(c.canonical(dict(prepared_initial=None)))
    )
    with pytest.raises(ValueError, match="candidate not frozen"):
        c.prepared_initial(protocol)
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="artifact differs"):
        c.prepared_initial(protocol)


def test_parent_selector_is_executed_in_fresh_interpreter(c, plan):
    original = load("sdsc_student_focus")
    candidate = c.prepared_initial(plan["protocol"])
    _, audit = c.decode_audit(plan["fit"]["audit"], candidate, plan["fit"]["plan"])
    assert original.select_development(plan["fit"]["plan"], audit["development"])["step"] == 2


@pytest.mark.parametrize("field", ["checkpoint_path", "checkpoint_size", "master_model_state_sha256"])
def test_report_selected_artifact_evidence_fully_bound(c, plan, field):
    report, rows = good_report(c, plan)
    report["fit_evidence"].pop(field)
    with pytest.raises(ValueError, match="parent binding"):
        c.validate_worker_report(report, plan, "123", rows)


def test_report_parent_artifact_bound(c, plan):
    report, rows = good_report(c, plan)
    report["preparation_protocol_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity/scope"):
        c.validate_worker_report(report, plan, "123", rows)


def test_report_execution_complete_claim_requires_boolean(c, plan):
    report, rows = good_report(c, plan)
    report["execution_complete"] = 1
    with pytest.raises(ValueError, match="identity/scope"):
        c.validate_worker_report(report, plan, "123", rows)


def test_prior_qualification_report_schema_is_rejected(c, plan):
    report, rows = good_report(c, plan)
    report["schema"] = "quest-sdsc-student-branch-qualification-report-v1"
    with pytest.raises(ValueError, match="identity/scope"):
        c.validate_worker_report(report, plan, "123", rows)


@pytest.mark.parametrize(
    "field,value",
    [("task", "qwen3-v2-student-branch-preparation-v3"), ("mode", "preflight")],
)
def test_prior_fit_or_current_preflight_cannot_admit_candidate(c, plan, monkeypatch, field, value):
    candidate, report, publication = parent_artifacts(c, plan, monkeypatch)
    parent = copy.deepcopy(plan["fit"]["plan"])
    parent[field] = value
    with pytest.raises(ValueError, match="accepted focus fit"):
        c.validate_fit_evidence(report, publication, parent, plan["fit"]["audit"], candidate)


def test_previous_fit_token_count_cannot_admit_candidate(c, plan, monkeypatch):
    candidate, report, publication = parent_artifacts(c, plan, monkeypatch)
    report["training_input_tokens"] = 1652820
    with pytest.raises(ValueError, match="complete focus fit"):
        c.validate_fit_evidence(report, publication, plan["fit"]["plan"], plan["fit"]["audit"], candidate)


def rehash_record(c, records, name, value):
    raw = value if isinstance(value, bytes) else c.canonical(value)
    records[name] = dict(size=len(raw), sha256=c.sha(raw), base64=base64.b64encode(raw).decode())


@pytest.mark.parametrize(
    "mutation",
    [
        "size_bool",
        "oversize",
        "bad_base64",
        "extra_key",
        "trailing_lf",
        "failed",
        "missing_verified",
        "wrong_pub",
        "queued",
        "publication_false",
        "missing_selected",
        "old_schema",
    ],
)
def test_completion_rejects_rehashed_contradictions(c, plan, mutation):
    records = copy.deepcopy(plan["fit"]["completion"])
    if mutation == "size_bool":
        records["verified_status"]["size"] = True
    elif mutation == "oversize":
        records["verified_status"]["size"] = c.MAX_PLAN + 1
    elif mutation == "bad_base64":
        records["verified_status"]["base64"] = "!"
    elif mutation == "extra_key":
        records["execution_plan"] = records["verified_status"]
    elif mutation == "trailing_lf":
        rehash_record(
            c, records, "verified_status", base64.b64decode(records["verified_status"]["base64"]) + b"\n"
        )
    else:
        name = (
            "publication"
            if mutation in {"publication_false", "missing_selected", "old_schema"}
            else "verified_status"
        )
        value = json.loads(base64.b64decode(records[name]["base64"]))
        if mutation == "failed":
            value["success"] = False
        elif mutation == "missing_verified":
            value.pop("artifact_hashes_verified")
        elif mutation == "wrong_pub":
            value["publication_sha256"] = "0" * 64
        elif mutation == "queued":
            value["queue"]["stdout"] = "12345|COMPLETED"
        elif mutation == "publication_false":
            value["passed"] = False
        elif mutation == "missing_selected":
            value["large_files"] = []
        elif mutation == "old_schema":
            value["schema"] = "quest-sdsc-student-order-publication-v1"
        rehash_record(c, records, name, value)
    with pytest.raises(ValueError):
        c.decode_completion_records(records, c.prepared_initial(plan["protocol"]), plan["fit"]["plan"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_rows_reconstructed", 256),
        ("consumed_training_rows", 256),
        ("optimizer_windows_replayed", 4),
        ("full_population_input_tokens", 1614932),
        ("training_input_tokens", 205428),
        ("training_order_reconstructed", 1),
        ("checkpoint_cumulative_input_tokens", {}),
        ("gpu_training_independently_recomputed", True),
    ],
)
def test_rehashed_audit_cannot_replace_full_training_evidence(c, plan, field, value):
    candidate = copy.deepcopy(c.prepared_initial(plan["protocol"]))
    _, audit = c.decode_audit(plan["fit"]["audit"], candidate, plan["fit"]["plan"])
    audit["training_evidence"][field] = value
    raw = c.canonical(audit) + b"\n"
    candidate["independent_raw_audit_sha256"] = c.sha(raw)
    with pytest.raises(ValueError, match="training evidence"):
        c.decode_audit(c.audit_payload(raw, candidate), candidate, plan["fit"]["plan"])


@pytest.mark.parametrize("mutation", ["missing", "wrong_sha", "wrong_size"])
def test_report_requires_all_three_offline_completion_records(c, plan, mutation):
    report, records = good_report(c, plan)
    bundle = report["fit_evidence"]["completion"]
    if mutation == "missing":
        bundle.pop("publication")
    elif mutation == "wrong_sha":
        bundle["verified_status"]["sha256"] = "0" * 64
    else:
        bundle["fit_plan"]["size"] += 1
    with pytest.raises(ValueError, match="parent binding"):
        c.validate_worker_report(report, plan, "123", records)


@pytest.fixture
def parent_storage(c, plan, tmp_path, monkeypatch):
    fit = plan["fit"]["plan"]
    directory = tmp_path / "parent-submission"
    root = tmp_path / "parent-results"
    directory.mkdir()
    root.mkdir()
    fit.update(
        submission_dir=str(directory),
        result_dir=str(root),
        claim=str(tmp_path / "parent-claim.json"),
        scientific_claim=str(tmp_path / "parent-scientific.json"),
    )
    candidate = c.prepared_initial(plan["protocol"])
    candidate["fit_plan_sha256"] = c.sha(c.canonical(fit))
    bundle = synthetic_completion(c, fit, candidate, core())
    plan["fit"]["completion"] = bundle
    decoded = c.decode_completion_records(bundle, candidate, fit)
    status = json.loads(decoded["verified_status"])
    (directory / "plan.json").write_bytes(c.canonical(fit))
    Path(fit["claim"]).write_bytes(
        c.canonical(dict(intent_id=fit["intent_id"], plan_sha256=candidate["fit_plan_sha256"]))
    )
    scientific = dict(science="fixture frozen scientific claim")
    monkeypatch.setattr(c._fit, "scientific_claim_record", lambda _: scientific)
    Path(fit["scientific_claim"]).write_bytes(c.canonical(scientific))
    receipt = dict(job_id=candidate["fit_job_id"])
    (directory / "receipt.json").write_bytes(c.canonical(receipt))
    (root / "receipt.json").write_bytes(decoded["publication"])
    monkeypatch.setattr(c._fit, "validate_submission_receipt", lambda *a: None)
    calls = []

    def inspect(p, r, *, accounting_snapshot=None):
        calls.append(("flat", p, r, accounting_snapshot))
        return copy.deepcopy(status)

    monkeypatch.setattr(c._fit, "inspect_job", inspect)
    doc = c.document
    monkeypatch.setattr(
        c,
        "document",
        lambda path, expected=None: {} if Path(path).name == "prepare-report.json" else doc(path, expected),
    )
    monkeypatch.setattr(c, "validate_fit_evidence", lambda *a: calls.append(("raw_audit", a)))
    return status, calls, directory


def test_parent_fresh_admission_uses_frozen_flat_inspector(c, plan, parent_storage):
    _, calls, _ = parent_storage
    result = c.verify_parent(plan, accounting=True)
    assert result["success"] is True
    assert [r[0] for r in calls] == ["flat", "raw_audit"]
    assert calls[0][1] == plan["fit"]["plan"] and calls[0][3] is None
    assert not hasattr(c, "_recovery")


def test_node_reuses_accounting_but_revalidates_flat_raw_artifacts(c, plan, parent_storage):
    _, calls, _ = parent_storage
    path = Path(plan["submission_dir"])
    path.mkdir(parents=True)
    saved = dict(accounting="previously verified admission accounting")
    (path / "admission.json").write_bytes(c.canonical(dict(fit=saved)))
    c.verify_parent(plan, accounting=False)
    assert calls[0][3] == saved and calls[1][0] == "raw_audit"


@pytest.mark.parametrize("mutation", ["claim", "receipt", "failed", "publication"])
def test_parent_cannot_admit_contradictory_persistent_completion(c, plan, parent_storage, mutation):
    status, calls, directory = parent_storage
    if mutation == "claim":
        Path(plan["fit"]["plan"]["claim"]).write_bytes(b"{}")
    elif mutation == "receipt":
        (directory / "receipt.json").write_bytes(c.canonical(dict(job_id="999")))
    elif mutation == "failed":
        status["success"] = False
    else:
        (Path(plan["fit"]["plan"]["result_dir"]) / "receipt.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        c.verify_parent(plan, accounting=True)
    assert "raw_audit" not in [r[0] for r in calls]


def test_same_parent_acceptance_can_be_actual_fit_producer(c, plan):
    candidate = copy.deepcopy(c.prepared_initial(plan["protocol"]))
    fit = copy.deepcopy(plan["fit"]["plan"])
    fit["protocol"]["head"] = fit["provenance"]["head"] = candidate["fit_source_head"] = core().PARENT_HEAD
    candidate["fit_plan_sha256"] = c.sha(c.canonical(fit))
    records = synthetic_completion(c, fit, candidate, core())
    c.decode_completion_records(records, candidate, fit)
    assert candidate["fit_source_head"] == core().PARENT_HEAD


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_actual_ssh_launch_named_controller_bytes_before_execution(c, plan, tmp_path, wrong_hash):
    from types import SimpleNamespace

    release = tmp_path / "launch"
    release.mkdir()
    (release / "source").symlink_to(ROOT, target_is_directory=True)
    plan["release"], plan["python"] = str(release), sys.executable
    name = "tools/sdsc_student_focus_qualify.py"
    digest = c.sha((ROOT / name).read_bytes())
    plan["control_sha256"][name] = "0" * 64 if wrong_hash else digest
    # Exercise a deliberately reordered inventory: authorization is by exact path.
    c.TOOLS = tuple(reversed(c.TOOLS))
    calls = []

    def ssh(argv, *, data, timeout):
        assert argv[-2] == str(release / "source" / name)
        assert timeout == 240 and json.loads(data) == dict(
            action="dry-run", plan=plan, authorize=False, dry_run=None
        )
        result = subprocess.run(argv, input=data, capture_output=True, timeout=10)
        calls.append(result)
        return result

    with pytest.raises(
        ValueError, match="AssertionError" if wrong_hash else "Slurm operations cannot run on Quest"
    ):
        c.ssh_operation(SimpleNamespace(ssh_call=ssh), plan, "dry-run")
    assert len(calls) == 1
    assert (b"Slurm operations cannot run on Quest" in calls[0].stderr) is (not wrong_hash)


def test_actual_ssh_launch_preserves_remote_argv_and_request(c, plan, tmp_path):
    from types import SimpleNamespace

    script = tmp_path / "release/source/tools/sdsc_student_focus_qualify.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys\nprint(json.dumps(dict(argv=sys.argv, request=json.load(sys.stdin))))\n"
    )
    plan["release"], plan["python"] = str(script.parents[2]), sys.executable
    plan["control_sha256"]["tools/sdsc_student_focus_qualify.py"] = c.sha(script.read_bytes())

    def ssh(argv, *, data, timeout):
        return subprocess.run(argv, input=data, capture_output=True, timeout=10)

    proof = dict(dry_run=True, fixture="no real remote action")
    result = c.ssh_operation(SimpleNamespace(ssh_call=ssh), plan, "submit", True, proof)
    assert result == dict(
        argv=[str(script), "remote"], request=dict(action="submit", plan=plan, authorize=True, dry_run=proof)
    )


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_actual_job_script_executes_named_node_and_exact_plan_arguments(c, plan, tmp_path, wrong_hash):
    node = tmp_path / "release/source/tools/sdsc_student_focus_qualify_job.py"
    node.parent.mkdir(parents=True)
    node.write_text("import json,sys\nprint(json.dumps(sys.argv))\n")
    plan["release"], plan["python"] = str(node.parents[2]), sys.executable
    plan["control_sha256"]["tools/sdsc_student_focus_qualify_job.py"] = (
        "0" * 64 if wrong_hash else c.sha(node.read_bytes())
    )
    script = tmp_path / "job.sh"
    script.write_bytes(c.job_script(plan))
    digest = c.sha(c.canonical(plan))
    result = subprocess.run(["/bin/bash", str(script), digest], capture_output=True, timeout=10)
    if wrong_hash:
        assert result.returncode != 0 and b"AssertionError" in result.stderr and result.stdout == b""
    else:
        assert result.returncode == 0 and json.loads(result.stdout) == [
            str(node),
            str(Path(plan["submission_dir"]) / "plan.json"),
            digest,
        ]


def test_runtime_without_bound_accepted_candidate_is_nonoperational(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        c.prepared_initial(dict(artifact_sha256="a" * 64, protocol_sha256="b" * 64))
    assert not (tmp_path / c.PROTOCOL_PATH).exists()


def test_one_gpu_resource_requires_integer(c, plan):
    plan["resources"]["gpus"] = True
    with pytest.raises(ValueError, match="resources"):
        c.validate_plan(plan)


def test_actual_saved_preflight_is_valid_parent_execution_but_never_fit_candidate(c, plan):
    path = ROOT / ".sdsc/student-focus-v5/aaf558e27fd94b9d6ba47530c162bd38/plan.json"
    fetch = ROOT / ".sdsc/fetched/54661792/fetch-m238fgjm"
    if not path.exists() or not fetch.exists():
        pytest.skip("historical local artifacts unavailable")
    actual = load("sdsc_student_focus")
    previous = json.loads(path.read_bytes())
    actual.validate_plan(previous)
    publication = json.loads((fetch / "receipt.json").read_bytes())
    records = actual.publication_records(publication, previous, "54661792")
    report = json.loads((fetch / "prepare-report.json").read_bytes())
    actual.validate_worker_report(report, previous, "54661792", records)
    actual.validate_update_records((fetch / "prepare-updates.jsonl").read_bytes(), report)
    actual.validate_startup_evidence(previous, fetch, "54661792")
    candidate = copy.deepcopy(c.prepared_initial(plan["protocol"]))
    candidate.update(
        fit_plan_sha256=c.sha(c.canonical(previous)),
        fit_intent=previous["intent_id"],
        fit_source_head=previous["protocol"]["head"],
    )
    with pytest.raises(ValueError, match="bound complete V5 fit"):
        core().validate_parent_completion_evidence(
            candidate, fit_plan=previous, status={}, publication_raw=(fetch / "receipt.json").read_bytes()
        )


def test_actual_execute_full_fit_report_enters_new_selected_join(c, plan, tmp_path, monkeypatch):
    """Actual producer report shape with explicit CPU hardware/toy-token fixtures.

    The frozen execute fixture builds all 32 updates, step4 restore, 12 checkpoints
    and earliest selection. Its CPU token lengths and hardware observations are
    fixture data, so we substitute the fixed independently measured population
    counters before testing this consumer. This does not replay raw18432 or claim
    any real eligible fit; actual admission still requires those bound artifacts.
    """
    spec = importlib.util.spec_from_file_location(
        "_actual_focus_worker_fixture", ROOT / "tests/sdsc/test_student_focus_worker.py"
    )
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    m = producer.worker()
    captured = {}
    execute = m.execute

    def record(*a, **kw):
        value = execute(*a, **kw)
        captured["report"] = value
        return value

    with monkeypatch.context() as scope:
        scope.setattr(producer, "worker", lambda: m)
        scope.setattr(m, "execute", record)
        producer.test_execute_preserves_mode_schedule_restore_and_full_failure_evidence(
            tmp_path, scope, "fit", True
        )
    value = captured["report"]
    assert (
        value["selected_checkpoint"]["step"] == 1
        and len(value["checkpoints"]) == len(value["development"]) == 12
    )
    assert value["source_kind"] == "symbolic_canonical_focus_preparation" and value["optimizer_steps"] == 32
    fit = copy.deepcopy(plan["fit"]["plan"])
    value.update(
        run_id=fit["run_id"],
        source_code_sha256=fit["code_sha256"],
        plan_sha256=c.sha(c.canonical(fit)),
        protocol_sha256=fit["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=fit["protocol"]["artifact_sha256"],
        training_input_tokens=1665064,
    )
    value["data_audit"]["token_envelope"] = dict(
        fit_rows=2048,
        optimizer_windows=32,
        window_input_tokens=list(core().PARENT_WINDOW_INPUT_TOKENS),
        total_input_tokens=1665064,
        token_budget=2000000,
    )
    candidate = copy.deepcopy(c.prepared_initial(plan["protocol"]))
    cp = value["checkpoints"][0]
    candidate.update(
        fit_job_id="123",
        checkpoint_step=cp["step"],
        checkpoint_path=cp["path"],
        checkpoint_size=cp["size"],
        checkpoint_sha256=cp["sha256"],
        master_model_state_sha256=cp["reload_by_rank"][0]["model_state_sha256"],
        fit_report_sha256=c.sha(c.canonical(value)),
    )
    publication = dict(
        schema="quest-sdsc-student-focus-publication-v1",
        task=c._fit.TASK,
        mode="fit",
        job_id="123",
        intent_id=fit["intent_id"],
        run_id=fit["run_id"],
        code_sha256=fit["code_sha256"],
        plan_sha256=c.sha(c.canonical(fit)),
        passed=True,
        stage_complete=True,
        preparation_complete=True,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        files=value["raw_artifacts"],
        large_files=[{key: row[key] for key in ("path", "size", "sha256")} for row in value["checkpoints"]],
        **dict.fromkeys(c.FLAGS, False),
    )
    candidate["fit_publication_sha256"] = c.sha(c.canonical(publication))
    original = c.prepared_initial(plan["protocol"])
    _, audit = c.decode_audit(plan["fit"]["audit"], original, fit)
    audit.update(
        job_id="123",
        publication_sha256=candidate["fit_publication_sha256"],
        result_sha256=candidate["fit_report_sha256"],
        development=value["development"],
        selected_checkpoint=value["selected_checkpoint"],
    )
    audit["isolation_evidence"]["producer_evidence_sha256"] = next(
        row["sha256"] for row in publication["files"] if row["path"] == "prepare-data-isolation.json"
    )
    raw = c.canonical(audit) + b"\n"
    candidate["independent_raw_audit_sha256"] = c.sha(raw)
    actual = load("sdsc_student_focus")
    monkeypatch.setattr(c, "_fit", actual)
    selected = c.validate_fit_evidence(value, publication, fit, c.audit_payload(raw, candidate), candidate)
    assert selected["step"] == 1
    failed = copy.deepcopy(value)
    failed.update(passed=False, selected_checkpoint=None)
    with pytest.raises(ValueError, match="incomplete/overclaiming"):
        c.validate_fit_evidence(failed, publication, fit, c.audit_payload(raw, candidate), candidate)


@pytest.mark.parametrize("mutation", ["float_size", "oversize", "extra", "nonstring"])
def test_audit_record_requires_bounded_typed_exact_envelope(c, plan, mutation):
    record = copy.deepcopy(plan["fit"]["audit"])
    if mutation == "float_size":
        record["size"] = float(record["size"])
    elif mutation == "oversize":
        record["size"] = c.CAP + 1
    elif mutation == "extra":
        record["unused"] = True
    else:
        record["base64"] = []
    with pytest.raises(ValueError, match="unreviewed fit audit"):
        c.decode_audit(record, c.prepared_initial(plan["protocol"]), plan["fit"]["plan"])
