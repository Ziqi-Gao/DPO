"""Local transport, strict completion, raw startup, and owned-child CPU seams."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
        if Path(command).name in {"sbatch", "sacct", "squeue", "scontrol", "srun", "scancel", "ssh"}:
            pytest.fail("CPU fixture attempted real remote/Slurm command")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    return load("sdsc_student_anchor_probe")


@pytest.fixture
def n():
    return load("sdsc_student_anchor_probe_job")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    pins = {name: c.sha((ROOT / name).read_bytes()) for name in c.TOOLS}
    pins.update(c.FROZEN)
    raw_protocol = (ROOT / c.PROTOCOL_PATH).read_bytes()
    core = json.loads(raw_protocol)
    core.pop("review")
    pins[c.PROTOCOL_PATH] = c.sha(raw_protocol)
    pins[c.PROTOCOL_MODULE] = c.sha((ROOT / c.PROTOCOL_MODULE).read_bytes())
    protocol = dict(
        protocol_sha256=c.sha(c.canonical(core)),
        artifact_sha256=pins[c.PROTOCOL_PATH],
        implementation_commit="1" * 40,
        acceptance_commit="2" * 40,
        head="2" * 40,
        science_file_sha256={
            name: c.sha((ROOT / name).read_bytes()) for name in json.loads(raw_protocol)["science_files"]
        },
    )
    receipt = dict(
        job_id=c.PARENT_JOB,
        intent_id=c.PARENT_INTENT,
        submission_dir=str(c.CONTROL / "submissions" / c.PARENT_INTENT),
        science_git_head=c._initial.SCIENCE_HEAD,
        bundle_sha256=c._initial.PARENT_BUNDLE_SHA,
        python=str(c.PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12"),
        hf_home=str(c.PROJECT / "cache/huggingface"),
    )
    contract = c.helper("sdsc_student_contract")
    value = dict(
        schema="quest-sdsc-student-anchor-probe-plan-v1",
        task=c.TASK,
        mode="probe",
        arm="control",
        run_id="new-preparation-run",
        code_sha256="c" * 64,
        manifest_sha256="d" * 64,
        resources=c.resources(),
        control_sha256=pins,
        protocol=protocol,
        provenance=dict(
            directory=str(c.CONTROL / "provenance" / ("e" * 64)), manifest_sha256="e" * 64, head="2" * 40
        ),
        parent=dict(
            receipt=receipt,
            report_sha256=c.PARENT_REPORT_SHA,
            publication_sha256=c._initial.PARENT_PUBLICATION_SHA,
        ),
        checkpoints=[
            dict(
                label="initial", path="artifacts/initial_checkpoint.pt", size=3441276375, sha256=c.INITIAL_SHA
            )
        ],
        resolved_config=dict(
            path="artifacts/canonical_sft/resolved_config.yaml", size=7923, sha256=c.CONFIG_SHA
        ),
        dataset_inputs=[
            dict(row, source=str(contract.DATASET_ROOT / row["path"])) for row in contract.DATASET_FILES
        ],
        python=receipt["python"],
        hf_home=receipt["hf_home"],
        created_at="2026-10-01T00:00:00Z",
        **dict.fromkeys(c.FLAGS, False),
    )
    bind(c, value)
    c.validate_plan(value)
    return value


def bind(c, plan):
    plan["science_identity"] = c.science_identity(plan["arm"], plan["protocol"])
    intent = c.execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(c.CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(c.CONTROL / "student-anchor-probe-submissions" / intent),
        claim=str(c.CONTROL / "student-anchor-probe-claims" / (intent + ".json")),
        result_dir=str(c.PROJECT / "student-anchor-probe" / intent),
        job_name="opd-sap-" + intent,
    )
    plan["scientific_claim"] = str(c.scientific_claim_path(plan))


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "unknown"),
        ("task", "canonical_grpo"),
        ("student_accepted", True),
        ("formal_initial_accepted", True),
        ("execution_class_certified", True),
        ("result_dir", "/tmp/checkpoints"),
        ("python", "/usr/bin/python3"),
        ("job_name", "repeat"),
        ("intent_id", "a" * 32),
    ],
)
def test_plan_rejects_scope_or_binding_change(c, plan, field, value):
    plan[field] = value
    with pytest.raises((ValueError, KeyError)):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "change", ["omit_family", "replace_initial", "change_lr", "self_acceptance", "changed_helper"]
)
def test_scientific_plan_corruption(c, plan, change):
    if change == "omit_family":
        plan["dataset_inputs"].pop()
    elif change == "replace_initial":
        plan["checkpoints"][0]["sha256"] = "f" * 64
    elif change == "change_lr":
        plan["science_identity"]["learning_rate"] = 5e-4
    elif change == "self_acceptance":
        plan["protocol"]["acceptance_commit"] = plan["protocol"]["implementation_commit"]
    else:
        plan["control_sha256"]["tools/sdsc_student_lr.py"] = "f" * 64
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_scientific_claim_survives_changed_execution_control(c, plan):
    before = plan["scientific_claim"]
    old = plan["intent_id"]
    plan["control_sha256"]["tools/sdsc_student_anchor_probe_worker.py"] = "9" * 64
    plan["protocol"]["science_file_sha256"]["tools/sdsc_student_anchor_probe_worker.py"] = "9" * 64
    bind(c, plan)
    assert plan["scientific_claim"] == before and plan["intent_id"] != old
    c.validate_plan(plan)


def test_remote_entrypoint_refuses_local_slurm(c):
    result = subprocess.run(
        [os.sys.executable, "-I", "-B", str(ROOT / "tools/sdsc_student_anchor_probe.py"), "remote"],
        input=b"{}",
        capture_output=True,
    )
    assert result.returncode != 0 and b"Slurm operations cannot run on Quest" in result.stderr


def test_launcher_uses_absolute_release_not_slurm_spool(c, plan, tmp_path):
    script = c.job_script(plan).decode()
    assert str(Path(plan["release"]) / "source/tools/sdsc_student_anchor_probe_job.py") in script
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


def test_gpu_ceiling_counts_two_new_gpus(c, monkeypatch):
    monkeypatch.setattr(c, "run", lambda _: dict(returncode=0, stdout="100|RUNNING|N/A\n", stderr=""))
    monkeypatch.setattr(c._initial, "live_gpu_request", lambda *a, **k: dict(allocatable_gpus=3))
    with pytest.raises(ValueError, match="four allocatable"):
        c.check_gpu_ceiling()


def test_completed_own_jobs_use_live_tres_helper(c, monkeypatch):
    seen = []
    monkeypatch.setattr(c, "run", lambda _: dict(returncode=0, stdout="100|COMPLETED|N/A\n", stderr=""))

    def helper(job, user, gres, queue_state):
        seen.append((job, gres, queue_state))
        return dict(allocatable_gpus=0)

    monkeypatch.setattr(c._initial, "live_gpu_request", helper)
    assert c.check_gpu_ceiling()["existing_allocatable_gpus"] == 0
    assert seen == [("100", "N/A", "COMPLETED")]


def test_memory_own_job_oom_rejected(c):
    raw = {"ancestors": [{"path": "/slurm/job_123/step_batch", "memory_events": {"oom": 1}}]}
    with pytest.raises(ValueError, match="nonzero"):
        c.validate_job_memory_events(raw, "123")


def test_fixed_runtime_checks_all_nineteen_packages(c, plan, monkeypatch):
    identity = dict(python="3.12.13", executable=plan["python"], packages=dict(c.RUNTIME_PACKAGES))
    monkeypatch.setattr(c, "run", lambda *a, **kw: dict(returncode=0, stdout=json.dumps(identity), stderr=""))
    assert c.verify_runtime(plan) == identity and len(identity["packages"]) == 19
    identity["packages"]["accelerate"] = "new-version"
    with pytest.raises(ValueError, match="package identity"):
        c.verify_runtime(plan)


def test_deployed_worker_must_match_reviewed_implementation(c, plan):
    plan["control_sha256"]["tools/sdsc_student_anchor_probe_worker.py"] = "8" * 64
    bind(c, plan)
    with pytest.raises(ValueError, match="accepted scientific byte"):
        c.validate_plan(plan)


def test_missing_executed_science_inventory_rejected(c, plan):
    plan["protocol"]["science_file_sha256"].pop("tools/sdsc_student_lr_probe.py")
    with pytest.raises(ValueError, match="lacks executed"):
        c.validate_plan(plan)


def test_fixed_diagnostic_envelope_and_once_per_arm(c, plan):
    assert c.worker_budget_seconds(plan) == 6600
    assert c.STEPS == (4, 6, 7, 8, 12) and len(c.required_raw_names("probe")) == 15
    assert c.resources()["time"] == "02:00:00"
    assert {"--gpus=h100:2", "--mem=393216M", "--cpus-per-task=24", "--no-requeue", "--export=NONE"} <= set(
        c.sbatch(plan)
    )
    assert not any("constraint" in arg for arg in c.sbatch(plan))
    original = plan["scientific_claim"]
    plan["arm"] = "treatment"
    bind(c, plan)
    c.validate_plan(plan)
    assert original != plan["scientific_claim"]
    with pytest.raises(ValueError):
        c.resources("fit")


@pytest.mark.parametrize(
    "key,value", [("gpus", True), ("gpus", 4), ("cpus", 12), ("mem_gib", 192), ("time", "08:00:00")]
)
def test_resources_typed_and_exact(c, plan, key, value):
    plan["resources"][key] = value
    bind(c, plan)
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_current_proposed_protocol_cannot_operate(c):
    with pytest.raises(ValueError, match="did not resolve"):
        c.protocol_binding(ROOT)


def test_controller_launch_pins_controller_not_startup(c, plan):
    calls = []
    cli = SimpleNamespace(
        ssh_call=lambda argv, **kwargs: (
            calls.append(argv) or SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")
        )
    )
    c.ssh_operation(cli, plan, "status")
    assert calls[0][-1] == plan["control_sha256"]["tools/sdsc_student_anchor_probe.py"]
    assert calls[0][-2].endswith("/sdsc_student_anchor_probe.py")


def report(c, plan):
    spec = c.scientific_specification(plan)
    data = spec["data"]
    worker = load("sdsc_student_anchor_probe_worker")
    views = spec["development"]["views"]
    structures = spec["development"]["structure_groups"]

    def counts(n):
        return dict(answer_correct=0, proof_correct=0, format_valid=0, num_examples=n)

    ranks = []
    for rank in range(2):
        ranks.append(
            dict(
                rank=rank,
                model_state_sha256="a" * 64,
                exact_saved_master_reload=dict(
                    all_tensors_exact=True, key_count=311, loaded_before_bf16_copy=True
                ),
                root_export_logits=[
                    dict(bitwise_equal=True, max_abs_error=0, rms_error=0, argmax_mismatches=0)
                    for _ in range(2)
                ],
                inference_envelope_probe=dict(
                    passed=True,
                    input_length=2454,
                    token_id=17,
                    all_logits_finite=True,
                    use_cache=False,
                    native_bfloat16=True,
                    scope="synthetic_diagnostic_context_no_development_content",
                ),
                counts=counts(384),
                per_view_counts={name: counts(64) for name in views},
                structure_counts={name: counts(n) for name, n in zip(structures, (22, 21, 21), strict=True)},
            )
        )
    cps = [
        dict(
            step=step,
            arm=plan["arm"],
            path=f"checkpoints/step-{step:08d}.pt",
            size=10,
            sha256=c.sha(str(step).encode()),
            scope="student_anchor_probe_diagnostic_model_only",
            reload_by_rank=copy.deepcopy(ranks),
        )
        for step in c.STEPS
    ]
    return dict(
        schema="quest-sdsc-student-anchor-probe-report-v1",
        mode="probe",
        arm=plan["arm"],
        passed=True,
        diagnostic_complete=True,
        execution_complete=True,
        preparation_complete=False,
        selected_checkpoint=None,
        job_id="123",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        initial_checkpoint_sha256=c.INITIAL_SHA,
        optimizer_steps=12,
        global_batch_size=64,
        world_size=2,
        learning_rate=5e-5,
        fit_base_examples=256,
        fit_training_views=2048,
        consumed_fit_training_views=768,
        development_base_examples=128,
        development_views=768,
        dataset_sha256=data["arm_manifest_sha256"][plan["arm"]],
        teacher_data_required=False,
        source_kind="symbolic_canonical_anchor_probe",
        initial_loading=dict(
            all_parameters_trainable=True,
            all_tensors_exact=True,
            checkpoint_sha256=c.INITIAL_SHA,
            key_count=311,
        ),
        fsdp=dict(
            effective_fsdp_sharding_strategy="FULL_SHARD",
            requested_fsdp_sharding_strategy="FULL_SHARD",
            fsdp_wrapper_count=29,
        ),
        development_generation=dict(
            max_new_tokens=256,
            max_model_input_length=2454,
            max_training_model_input_length=1536,
            native_bfloat16=True,
            explicit_autocast=False,
            do_sample=False,
            use_cache=False,
            seed="42_plus_manifest_index",
            rank_partition="complete_development_base_blocks_round_robin_rank",
            truncation=False,
        ),
        data_audit=dict(
            passed=True,
            isolation=dict(
                passed=True,
                counts=data["isolation_population_counts"],
                excluded_view_counts=data["isolation_excluded_view_counts"],
                population_stream_sha256=dict.fromkeys(data["isolation_population_counts"], "a" * 64),
                excluded_view_stream_sha256=dict.fromkeys(data["isolation_excluded_view_counts"], "b" * 64),
                dimensions=data["isolation_dimensions"],
                arm_manifest_sha256=data["arm_manifest_sha256"],
                scientific_acceptance=False,
                formal_generated_responses_or_scores_read=False,
            ),
            token_envelope=dict(
                fit_rows=2048,
                optimizer_windows=32,
                token_budget=2000000,
                total_input_tokens=32000,
                window_input_tokens=[1000] * 32,
                executed_optimizer_windows=12,
                executed_training_views=768,
                executed_input_tokens=12000,
            ),
        ),
        training_input_tokens=12000,
        checkpoint_restore=dict(
            passed=True,
            ranks=[
                dict(
                    rank=rank,
                    step=4,
                    passed=True,
                    actual_accelerate_save_load=True,
                    model_sha256="d" * 64,
                    optimizer_sha256="e" * 64,
                    runtime_sha256="f" * 64,
                )
                for rank in (0, 1)
            ],
        ),
        checkpoints=cps,
        development=[
            worker.aggregate_development(
                cp["reload_by_rank"],
                step=cp["step"],
                checkpoint_sha=cp["sha256"],
                dev_sha=data["examples_sha256"]["student_dev"],
            )
            for cp in cps
        ],
        raw_artifacts=[
            dict(path=name, size=3, sha256="b" * 64) for name in sorted(c.required_raw_names("probe"))
        ],
        **dict.fromkeys(c.FLAGS, False),
    )


def test_zero_accuracy_is_diagnostic_completion_never_model_acceptance(c, plan):
    value = report(c, plan)
    records = {row["path"]: row for row in value["raw_artifacts"]}
    c.validate_worker_report(value, plan, "123", records)
    assert all(dev["answer_correct"] == 0 for dev in value["development"])
    assert value["selected_checkpoint"] is None and value["preparation_complete"] is False


@pytest.mark.parametrize(
    "change",
    [
        "arm",
        "candidate",
        "acceptance",
        "twelve",
        "consumed",
        "zero_grad_not_claimed",
        "dataset",
        "isolation",
        "token_budget",
        "token_prefix",
        "restore",
        "master",
        "rank_master",
        "parity",
        "context",
        "duplicate_rank",
        "missing_step",
        "wrong_dev",
        "boolean_counts",
        "reduction",
        "raw",
    ],
)
def test_diagnostic_rejects_incomplete_or_cross_bound_evidence(c, plan, change):
    value = report(c, plan)
    records = copy.deepcopy({row["path"]: row for row in value["raw_artifacts"]})
    if change == "arm":
        value["arm"] = "treatment"
    elif change == "candidate":
        value["selected_checkpoint"] = {"step": 6}
    elif change == "acceptance":
        value["preparation_complete"] = True
    elif change == "twelve":
        value["optimizer_steps"] = 11
    elif change == "consumed":
        value["consumed_fit_training_views"] = 2048
    elif change == "zero_grad_not_claimed":
        value["initial_loading"]["all_parameters_trainable"] = False
    elif change == "dataset":
        value["dataset_sha256"] = "f" * 64
    elif change == "isolation":
        value["data_audit"]["isolation"]["formal_generated_responses_or_scores_read"] = True
    elif change == "token_budget":
        value["data_audit"]["token_envelope"]["window_input_tokens"].pop()
    elif change == "token_prefix":
        value["training_input_tokens"] = 32000
    elif change == "restore":
        value["checkpoint_restore"]["ranks"][1]["actual_accelerate_save_load"] = False
    elif change == "master":
        value["checkpoints"][0]["reload_by_rank"][0]["exact_saved_master_reload"]["key_count"] = 310
    elif change == "rank_master":
        value["checkpoints"][0]["reload_by_rank"][1]["model_state_sha256"] = "f" * 64
    elif change == "parity":
        value["checkpoints"][0]["reload_by_rank"][1]["root_export_logits"][0]["max_abs_error"] = 1e-8
    elif change == "context":
        value["checkpoints"][1]["reload_by_rank"][1]["inference_envelope_probe"] = None
    elif change == "duplicate_rank":
        value["checkpoints"][0]["reload_by_rank"][1]["rank"] = 0
    elif change == "missing_step":
        value["checkpoints"].pop()
    elif change == "wrong_dev":
        value["development"][0]["checkpoint_sha256"] = "e" * 64
    elif change == "boolean_counts":
        value["checkpoints"][0]["reload_by_rank"][0]["counts"]["answer_correct"] = False
    elif change == "reduction":
        value["development"][0]["answer_correct"] = 1
    else:
        value["raw_artifacts"].pop()
    with pytest.raises((ValueError, KeyError)):
        c.validate_worker_report(value, plan, "123", records)


def publication(c, plan):
    return dict(
        schema="quest-sdsc-student-anchor-probe-publication-v1",
        task=c.TASK,
        mode="probe",
        arm=plan["arm"],
        job_id="123",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        passed=False,
        stage_complete=False,
        diagnostic_complete=False,
        preparation_complete=False,
        selected_checkpoint=None,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        files=[],
        large_files=[],
        **dict.fromkeys(c.FLAGS, False),
    )


@pytest.mark.parametrize(
    "change",
    ["schema", "arm", "claim", "completion", "startup_size", "startup_total", "large", "path", "bool_size"],
)
def test_publication_boundaries(c, plan, change):
    value = publication(c, plan)
    assert c.publication_records(value, plan, "123") == {}
    if change == "schema":
        value["schema"] = "quest-sdsc-student-order-publication-v4"
    elif change == "arm":
        value["arm"] = "treatment"
    elif change == "claim":
        value["formal_initial_accepted"] = True
    elif change == "completion":
        value["diagnostic_complete"] = True
    elif change == "startup_size":
        value["files"] = [dict(path="startup.log", size=c.CAP + 1, sha256="a" * 64)]
    elif change == "startup_total":
        value["files"] = [dict(path=name, size=c.CAP, sha256="a" * 64) for name in c.EXECUTION_NAMES[:3]]
    elif change == "large":
        value["large_files"] = [
            dict(path=f"checkpoints/{i}.pt", size=32 * 1024**3, sha256="a" * 64) for i in range(5)
        ]
    elif change == "path":
        value["files"] = [dict(path="../escape", size=1, sha256="a" * 64)]
    else:
        value["files"] = [dict(path="startup.log", size=True, sha256="a" * 64)]
    with pytest.raises((ValueError, KeyError)):
        c.publication_records(value, plan, "123")


def test_node_persistent_flat_failure_retains_actual_startup(c, n, plan, tmp_path):
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    raw = b'{"failed":true}'
    (artifacts / "early-node-startup.json").write_bytes(raw)
    (artifacts / "startup.log").write_bytes(b"import begins\n")
    result = dict(job_id="123", stage_complete=False, exit_code=1, error="early CUDA failure")
    n.persist(plan, c, artifacts, None, result, {}, lambda _: None)
    value = c.document(destination / "receipt.json")
    assert c.publication_records(value, plan, "123")["early-node-startup.json"]["sha256"] == c.sha(raw)
    assert value["diagnostic_complete"] is False and value["arm"] == "control"
    assert not (destination / "execution").exists()
    with pytest.raises(ValueError):
        n.persist(plan, c, artifacts, None, result, {}, lambda _: None)


def require(value, message):
    if not value:
        raise ValueError(message)


def write_once(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)


@pytest.fixture
def startup_setup(tmp_path, monkeypatch):
    node = load("sdsc_student_anchor_probe_job")
    probe = load("sdsc_student_anchor_probe_startup")
    helpers = {"sdsc_student_anchor_probe_startup": probe}

    def helper(name):
        if name not in helpers:
            helpers[name] = load(name)
        return helpers[name]

    c = SimpleNamespace(
        helper=helper,
        require=require,
        write_once=write_once,
        canonical=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")).encode(),
        sha=lambda raw: hashlib.sha256(raw).hexdigest(),
    )
    for key in probe.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    environment = dict(os.environ)
    environment.update(
        SLURM_JOB_ID="12345",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        CUDA_VISIBLE_DEVICES="2,3",
        OMP_NUM_THREADS="7",
        MKL_NUM_THREADS="8",
        TMPDIR=str(tmp_path),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    source = tmp_path / "release/source/tools"
    source.mkdir(parents=True)
    script = source / "sdsc_student_anchor_probe_startup.py"
    script.write_text(
        """import importlib.util,sys,platform,os,time
from types import SimpleNamespace
path=sys.argv.pop(1) if False else """
        + repr(str(ROOT / "tools/sdsc_student_anchor_probe_startup.py"))
        + """
spec=importlib.util.spec_from_file_location("real_probe",path)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
platform.python_version=lambda:"3.12.13"
mode=os.environ.get("FIXTURE_MODE","good")
class FakeCuda:
    def is_available(self): return mode!="cuda_false"
    def device_count(self): return 2
    def init(self): return None
    def get_device_name(self,index): return "NVIDIA H100 fixture"
    def get_device_capability(self,index): return (9,0)
    def get_device_properties(self,index): return SimpleNamespace(total_memory=80*1024**3,uuid="fixture")
torch=SimpleNamespace(__version__="2.8.0+cu128",version=SimpleNamespace(cuda="12.8"),cuda=FakeCuda())
def import_torch():
    if mode=="hang": time.sleep(30)
    if mode=="import_error": raise ImportError("fixture import failure")
    if mode=="mutate_env": os.environ["OMP_NUM_THREADS"]="99"
    return torch
probe.import_torch=import_torch
if mode=="flood": print("x"*(2*1024**2),flush=True)
raise SystemExit(probe.main())
"""
    )
    execution = tmp_path / "work/execution"
    execution.mkdir(parents=True)
    environment["TMPDIR"] = str(execution.parent)
    outer = {
        "python": sys.executable,
        "release": str(tmp_path / "release"),
        "control_sha256": {
            "tools/sdsc_student_anchor_probe_worker.py": hashlib.sha256(
                (ROOT / "tools/sdsc_student_anchor_probe_worker.py").read_bytes()
            ).hexdigest()
        },
    }
    identity = {"job_id": "12345", "cuda_visible_devices": "2,3"}
    return node, probe, c, outer, identity, execution, environment


def run(startup_setup, mode="good", budget=20):
    node, probe, c, outer, identity, execution, environment = startup_setup
    environment["FIXTURE_MODE"] = mode
    meta = node.run_probe(
        outer, c, execution, identity, environment, deadline=time.monotonic() + budget, measure=lambda _: None
    )
    probe.validate_trace_meta(
        meta,
        worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_anchor_probe_worker.py"],
        job_id="12345",
        expected_cuda_visible_devices="2,3",
        expected_python=sys.executable,
        expected_argv=node.probe_argv(outer, execution, identity),
        log_bytes=(execution / "startup.log").read_bytes(),
    )
    return meta


def test_real_adapter_calls_frozen_worker_and_records_policy_without_changing_parent(startup_setup):
    node, probe, c, outer, identity, execution, environment = startup_setup
    before = dict(environment)
    meta = run(startup_setup)
    assert meta["probe_passed"] is True and meta["reaped"] is True and meta["exit_code"] == 0
    assert meta["child_environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert all(meta["child_environment"][key] == "12" for key in probe.THREAD_KEYS)
    assert {key: environment.get(key) for key in probe.THREAD_KEYS} == {
        key: before.get(key) for key in probe.THREAD_KEYS
    }
    assert meta["child_environment"]["TMPDIR"] == str(execution.parent)
    raw = json.loads((execution / "early-node-startup.json").read_bytes())
    probe.validate_report(
        raw,
        worker_sha256=outer["control_sha256"]["tools/sdsc_student_anchor_probe_worker.py"],
        job_id="12345",
        expected_cuda_visible_devices="2,3",
        scope="early_node",
    )
    assert raw["wrapper_sha256"] == probe.file_sha(probe.__file__)
    assert all(raw[key] is False for key in probe.FLAGS)
    assert b"import time:" in (execution / "startup.log").read_bytes()


@pytest.mark.parametrize("mode", ["cuda_false", "import_error", "mutate_env"])
def test_real_negative_probe_retains_trace_without_claiming_ready(startup_setup, mode):
    meta = run(startup_setup, mode)
    assert meta["probe_passed"] is False and meta["reaped"] is True and meta["exit_code"] == 1
    assert json.loads((startup_setup[5] / "trace-meta.json").read_bytes()) == meta
    assert (startup_setup[5] / "early-node-startup.json").exists()


def test_timeout_reaps_child_and_keeps_original_global_budget(startup_setup):
    meta = run(startup_setup, "hang", budget=5.8)
    assert meta["timed_out"] is True and meta["reaped"] is True and meta["probe_passed"] is False
    assert 5 < meta["time_limit_seconds"] <= 5.8
    assert meta["elapsed_seconds"] < 2
    assert meta["maximum_probe_seconds"] == 300
    assert any(row["event"] == "frozen_worker_begin" for row in meta["adapter_events"])
    assert not (startup_setup[5] / "early-node-startup.json").exists()
    assert not startup_setup[2].helper("sdsc_student_lr_job").group_has_live_members(meta["pid"])


def test_exhausted_budget_rejects_before_popen_and_preserves_negative_trace(startup_setup, monkeypatch):
    monkeypatch.setattr(
        startup_setup[0].subprocess, "Popen", lambda *a, **kw: pytest.fail("no remaining child budget")
    )
    meta = run(startup_setup, budget=4)
    assert meta["pid"] is None and meta["probe_passed"] is False
    assert "exhausted" in meta["error"]


def test_large_log_retains_tail_and_exact_omission_count(startup_setup):
    meta = run(startup_setup, "flood")
    assert meta["probe_passed"] is True
    assert meta["log"]["bytes_seen"] > 2 * 1024**2 and meta["log"]["retained_bytes"] == 1024**2
    assert meta["log"]["truncated"] is True
    assert len((startup_setup[5] / "startup.log").read_bytes()) == 1024**2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(elapsed_seconds=m["time_limit_seconds"] + 2),
        lambda m: m.update(reaped=False),
        lambda m: m.update(timed_out=True),
        lambda m: m.update(events_truncated=True),
        lambda m: m.update(exit_code=1),
        lambda m: m.update(student_accepted=True),
        lambda m: m["child_environment"].update(SLURM_MEM_PER_NODE="16384"),
        lambda m: m["adapter_events"][-1].update(returncode=1),
        lambda m: m["adapter_events"].reverse(),
        lambda m: m["adapter_events"][0].update(utc="2000-01-01T00:00:00+00:00"),
        lambda m: m["adapter_events"][-1].update(elapsed_seconds=9999),
        lambda m: m["adapter_events"][2].update(original_argv=["run"]),
        lambda m: m["log"].update(sha256="0" * 64),
    ],
)
def test_trace_rejects_forged_success_and_changed_bytes(startup_setup, mutation):
    meta = copy.deepcopy(run(startup_setup))
    mutation(meta)
    with pytest.raises((ValueError, TypeError)):
        startup_setup[1].validate_trace_meta(
            meta,
            worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_anchor_probe_worker.py"],
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            expected_python=sys.executable,
            log_bytes=(startup_setup[5] / "startup.log").read_bytes(),
        )


def test_source_staging_has_explicit_manifest_inputs_and_readonly_modes(tmp_path):
    node = load("sdsc_student_anchor_probe_job")
    contract = load("sdsc_student_contract")
    release = tmp_path / "release"
    (release / "source/tools").mkdir(parents=True)
    rows = []
    for name, mode in [("main.txt", 0o644), ("tools/run.py", 0o755)]:
        data = (name + "\n").encode()
        p = release / "source" / name
        p.write_bytes(data)
        p.chmod(mode)
        rows.append(dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest(), mode=mode))
    manifest = dict(code_sha256="a" * 64, run_id="fixture", files=rows)
    payload = json.dumps(manifest).encode()
    (release / "manifest.json").write_bytes(payload)
    control = SimpleNamespace(
        safe=Path,
        require=require,
        manifest_records=lambda m: {r["path"]: r for r in m["files"]},
        document=lambda path, digest: json.loads(path.read_bytes())
        if hashlib.sha256(path.read_bytes()).hexdigest() == digest
        else None,
        helper=lambda name: contract,
    )
    result = node.stage_source(
        dict(
            release=str(release),
            code_sha256="a" * 64,
            run_id="fixture",
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
        ),
        tmp_path / "staged",
        control,
    )
    assert result["read_back_verified"] is True and result["files"] == 2
    assert (tmp_path / "staged/main.txt").stat().st_mode & 0o777 == 0o444
    assert (tmp_path / "staged/tools/run.py").stat().st_mode & 0o777 == 0o555
    assert (tmp_path / "staged").stat().st_mode & 0o777 == 0o555


def test_full_resume_and_all_dense_checkpoints_required(c, plan):
    value = report(c, plan)
    rows = [{key: cp[key] for key in ("path", "size", "sha256")} for cp in value["checkpoints"]]
    rows += [
        dict(path="resume/step-00000004/" + name, size=1, sha256="a" * 64)
        for name in (
            "custom_checkpoint_0.pkl",
            "optimizer.bin",
            "pytorch_model_fsdp.bin",
            "random_states_0.pkl",
            "random_states_1.pkl",
            "runtime-rank-0.json",
            "runtime-rank-1.json",
        )
    ]
    c.validate_large_outputs(value, rows)
    with pytest.raises(ValueError):
        c.validate_large_outputs(value, rows[:-1])
    rows[0]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        c.validate_large_outputs(value, rows)


def test_actual_node_negative_early_probe_publishes_flat_evidence_before_staging(
    startup_setup, c, plan, tmp_path, monkeypatch
):
    node, probe, _, startup_plan, identity, _, environment = startup_setup
    plan["python"] = startup_plan["python"]
    plan["release"] = startup_plan["release"]
    plan["parent"]["receipt"]["result_dir"] = str(tmp_path / "parent")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("FIXTURE_MODE", "cuda_false")
    monkeypatch.setattr(node, "load_control", lambda *a: (c, plan))
    monkeypatch.setattr(node, "allocation", lambda *a: identity)
    monkeypatch.setattr(c, "verify_source", lambda *a: None)
    monkeypatch.setattr(c, "verify_parent", lambda *a: None)
    monkeypatch.setattr(c, "verify_runtime", lambda *a: {"fixture": True})
    monkeypatch.setattr(node, "mount", lambda path, c: {"fstype": "ext4" if path == scratch else "lustre"})
    monkeypatch.setattr(node.shutil, "disk_usage", lambda path: SimpleNamespace(free=192 * 1024**3))
    monkeypatch.setattr(node, "memory_envelope", lambda *a: {"fixture": True})
    monkeypatch.setattr(node.signal, "signal", lambda *a: None)
    monkeypatch.setattr(node, "stage_source", lambda *a: pytest.fail("failed startup reached source staging"))
    assert node.main(["fixture", "0" * 64]) == 1
    root = Path(plan["result_dir"])
    published = c.document(root / "receipt.json")
    rows = c.publication_records(published, plan, identity["job_id"])
    assert not published["diagnostic_complete"] and set(rows) == {
        "early-node-startup.json",
        "startup.log",
        "trace-meta.json",
        "node-result.json",
        "memory.json",
    }
    for name, row in rows.items():
        assert c.sha((root / name).read_bytes()) == row["sha256"]
    assert c.document(root / "early-node-startup.json")["cuda_ready"] is False
    assert c.document(root / "trace-meta.json")["reaped"] is True
    assert c.document(root / "node-result.json")["stage_complete"] is False


@pytest.mark.parametrize("worker_code", [0, 2])
def test_actual_rank_entry_validates_new_worker_and_records_exact_exit(
    startup_setup, tmp_path, monkeypatch, worker_code
):
    _, startup, _, plan, identity, _, environment = startup_setup
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for key in startup.THREAD_KEYS:
        monkeypatch.setenv(key, "12")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    spec = importlib.util.spec_from_file_location(
        "_cuda_fixture", ROOT / "tests/sdsc/test_cuda_diagnostic_worker.py"
    )
    fakes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fakes)
    monkeypatch.setattr(startup, "import_torch", lambda: fakes.FakeTorch())
    monkeypatch.setattr(startup.platform, "python_version", lambda: "3.12.13")
    calls = []
    monkeypatch.setattr(
        startup, "invoke_original", lambda argv, digest: (calls.append((argv, digest)) or worker_code)
    )
    expected_sha = plan["control_sha256"]["tools/sdsc_student_anchor_probe_worker.py"]
    args = [
        "run",
        "--worker-sha256",
        expected_sha,
        "--job-id",
        identity["job_id"],
        "--expected-cuda-visible-devices",
        "2,3",
        "--execution-output-dir",
        str(tmp_path),
        "--inputs-json",
        str(tmp_path / "inputs.json"),
        "--output-dir",
        str(tmp_path),
    ]
    assert startup.main(args) == worker_code
    assert calls == [
        (["--inputs-json", str(tmp_path / "inputs.json"), "--output-dir", str(tmp_path)], expected_sha)
    ]
    report = json.loads((tmp_path / "rank-1-startup.json").read_bytes())
    startup.validate_report(
        report,
        worker_sha256=expected_sha,
        job_id="12345",
        expected_cuda_visible_devices="2,3",
        scope="rank_entry",
        rank=1,
        original_argv=calls[0][0],
    )
    exit_row = json.loads((tmp_path / "rank-1-exit.json").read_bytes())
    assert exit_row["exit_code"] == worker_code and exit_row["original_worker_sha256"] == expected_sha
    assert all(exit_row[key] is False for key in startup.FLAGS)


def test_wrong_accepted_worker_digest_stops_before_training(startup_setup, tmp_path, monkeypatch):
    _, startup, _, _, identity, _, environment = startup_setup
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for key in startup.THREAD_KEYS:
        monkeypatch.setenv(key, "12")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(startup, "invoke_original", lambda *a: pytest.fail("mismatched worker invoked"))
    args = [
        "run",
        "--worker-sha256",
        "f" * 64,
        "--job-id",
        identity["job_id"],
        "--expected-cuda-visible-devices",
        "2,3",
        "--execution-output-dir",
        str(tmp_path),
        "--inputs-json",
        str(tmp_path / "inputs.json"),
        "--output-dir",
        str(tmp_path),
    ]
    assert startup.main(args) == 1
    row = json.loads((tmp_path / "rank-0-startup.json").read_bytes())
    assert row["cuda_ready"] is False and "error" in row
    assert not (tmp_path / "rank-0-exit.json").exists()


def update_records(report):
    result = []
    cumulative = 0
    windows = report["data_audit"]["token_envelope"]["window_input_tokens"]
    for step in range(1, 13):
        total = windows[step - 1]
        cumulative += total
        ranks = []
        for rank in range(2):
            metric = dict(
                verified_replay_loss=0.25,
                parameter_update_norm=0.1,
                step=float(step),
                optimizer_updates=float(step),
                model_facing_input_tokens_processed=float(cumulative),
                model_facing_input_tokens_this_update=total,
                token_budget=2000000,
                token_budget_consumed=cumulative,
                token_budget_remaining=2000000 - cumulative,
                token_budget_unit="global_nonpadding_model_input_tokens_processed",
                local_model_facing_input_tokens_this_update=total // 2,
                prompts_consumed=float(step * 32),
                trajectories_generated=float(step * 32),
                effective_positive_sequences=32.0,
                generated_trajectories=32.0,
                successful_trajectories=32.0,
                retry_count=0.0,
                reward_rate=1.0,
                effective_supervised_tokens_this_update=64.0,
                effective_supervised_tokens=64.0,
                supervised_response_tokens=float(step * 64),
                response_tokens_generated=float(step * 64),
            )
            ranks.append(
                dict(
                    rank=rank,
                    step=step,
                    global_slots=list(range((step - 1) * 64 + rank, step * 64, 2)),
                    global_input_tokens=total,
                    optimizer_calls=[step - 1] * 7 + [step],
                    metric=metric,
                    token_budget=dict(
                        accepted_optimizer_updates=step,
                        budget=2000000,
                        consumed=cumulative,
                        stop_reason=None,
                        unit="global_nonpadding_model_input_tokens_processed",
                    ),
                )
            )
        result.append(dict(step=step, ranks=ranks))
    return result


def test_complete_raw_optimizer_evidence_matches_diagnostic_report(c, plan):
    value = report(c, plan)
    rows = update_records(value)
    raw = b"".join(c.canonical(row) + b"\n" for row in rows)
    result = c.validate_update_records(raw, value)
    assert result == dict(
        optimizer_steps=12,
        world_size=2,
        global_training_views=768,
        training_input_tokens=12000,
        finite_loss_and_positive_update=True,
        gpu_tensors_independently_recomputed=False,
    )


@pytest.mark.parametrize(
    "change",
    [
        "loss",
        "update",
        "nan",
        "rank",
        "slots",
        "calls",
        "global",
        "local",
        "cumulative",
        "prefix",
        "missing",
        "duplicate",
        "boolean",
        "supervised",
        "budget",
    ],
)
def test_rehashed_raw_optimizer_contradictions_rejected(c, plan, change):
    value = report(c, plan)
    rows = update_records(value)
    row = rows[5]["ranks"][1]
    if change == "loss":
        row["metric"]["verified_replay_loss"] = -0.5
    elif change == "update":
        row["metric"]["parameter_update_norm"] = 0
    elif change == "nan":
        row["metric"]["parameter_update_norm"] = float("nan")
    elif change == "rank":
        row["rank"] = 0
    elif change == "slots":
        row["global_slots"][0] -= 64
    elif change == "calls":
        row["optimizer_calls"][0] += 1
    elif change == "global":
        row["global_input_tokens"] += 1
    elif change == "local":
        row["metric"]["local_model_facing_input_tokens_this_update"] += 1
    elif change == "cumulative":
        row["metric"]["token_budget_consumed"] -= 1
    elif change == "prefix":
        value["training_input_tokens"] += 1
    elif change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[5] = copy.deepcopy(rows[4])
    elif change == "boolean":
        row["metric"]["retry_count"] = False
    elif change == "supervised":
        row["metric"]["supervised_response_tokens"] -= 1
    else:
        row["token_budget"]["accepted_optimizer_updates"] -= 1
    raw = b"".join(json.dumps(item, sort_keys=True, separators=(",", ":")).encode() + b"\n" for item in rows)
    # Even an attacker consistently updating the small-artifact checksum cannot
    # turn contradictory raw optimizer evidence into diagnostic completion.
    digest = c.sha(raw)
    assert len(digest) == 64
    with pytest.raises(ValueError):
        c.validate_update_records(raw, value)


def test_rehashed_raw_log_cannot_forge_elapsed_clock_origin(startup_setup):
    meta = run(startup_setup)
    probe = startup_setup[1]
    root = startup_setup[5]
    raw = []
    for line in (root / "startup.log").read_bytes().splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            raw.append(line)
            continue
        if isinstance(row, dict) and row.get("schema") == probe.PROGRESS_SCHEMA:
            row["elapsed_seconds"] = 0.0
            line = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        raw.append(line)
    raw = b"\n".join(raw) + b"\n"
    for event in meta["adapter_events"]:
        event["elapsed_seconds"] = 0.0
    meta["log"].update(bytes_seen=len(raw), retained_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="origins"):
        probe.validate_trace_meta(
            meta,
            worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_anchor_probe_worker.py"],
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            expected_python=sys.executable,
            log_bytes=raw,
        )


@pytest.mark.parametrize("kind", ["plan", "receipt", "publication", "report"])
def test_prior_order_probe_identity_cannot_satisfy_anchor_contract(c, plan, kind):
    """Matching shapes and arbitrary rehashing cannot reuse the previous experiment."""
    if kind == "plan":
        value = copy.deepcopy(plan)
        value["schema"] = "quest-sdsc-student-order-probe-plan-v1"
        value["task"] = "qwen3-v2-student-order-probe-v1"
        with pytest.raises(ValueError, match="scope/resources"):
            c.validate_plan(value)
    elif kind == "receipt":
        value = c.make_receipt(plan, "123")
        value["task"] = "qwen3-v2-student-order-probe-v1"
        with pytest.raises(ValueError, match="receipt differs"):
            c.validate_submission_receipt(value, plan)
    elif kind == "publication":
        value = publication(c, plan)
        value["schema"] = "quest-sdsc-student-order-probe-publication-v1"
        value["task"] = "qwen3-v2-student-order-probe-v1"
        with pytest.raises(ValueError):
            c.publication_records(value, plan, "123")
    else:
        value = report(c, plan)
        value["schema"] = "quest-sdsc-student-order-probe-report-v1"
        with pytest.raises(ValueError, match="incomplete or overclaiming"):
            c.validate_worker_report(value, plan, "123", {r["path"]: r for r in value["raw_artifacts"]})


def test_anchor_scientific_claim_is_distinct_from_prior_order_probe(c, plan):
    prior = load("sdsc_student_order_probe")
    prior_identity = prior.science_identity(plan["arm"], plan["protocol"])
    assert prior_identity != plan["science_identity"]
    assert c.FROZEN == prior.FROZEN
    assert c.resources() == prior.resources()
    assert Path(plan["scientific_claim"]).parent.name == "student-anchor-probe-scientific-claims"
    assert c.sha(c.canonical(prior_identity)) != Path(plan["scientific_claim"]).stem
    for key, directory in (
        ("claim", "student-anchor-probe-claims"),
        ("result_dir", "student-anchor-probe"),
        ("submission_dir", "student-anchor-probe-submissions"),
    ):
        assert Path(plan[key]).parent.name == directory
