"""No-network transport and real filesystem fixtures for common preparation."""

from __future__ import annotations

import base64
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
    return load("sdsc_student_name_invariant")


@pytest.fixture
def n():
    return load("sdsc_student_name_invariant_job")


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
        science_file_sha256=dict(pins),
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
    snapshot = copy.deepcopy(c.RUNTIME_SNAPSHOT)
    snapshot["source_prefix"] = str(c.PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1")
    monkeypatch.setattr(c, "RUNTIME_SNAPSHOT", snapshot)
    specification = c.scientific_specification

    def fixture_specification(value):
        payload = specification(value)
        payload["execution"]["native_runtime_snapshot"] = snapshot
        return payload

    monkeypatch.setattr(c, "scientific_specification", fixture_specification)
    contract = c.helper("sdsc_student_contract")
    value = dict(
        schema="quest-sdsc-student-name-invariant-plan-v1",
        task=c.TASK,
        mode="preflight",
        run_id="new-preparation-run",
        code_sha256="c" * 64,
        manifest_sha256="d" * 64,
        resources=c.resources("preflight"),
        control_sha256=pins,
        protocol=protocol,
        provenance=dict(
            directory=str(c.CONTROL / "provenance-v2" / ("e" * 64)), manifest_sha256="e" * 64, head="2" * 40
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
        runtime_snapshot=snapshot,
        hf_home=receipt["hf_home"],
        preflight=None,
        created_at="2026-10-01T00:00:00Z",
        **dict.fromkeys(c.FLAGS, False),
    )
    bind(c, value)
    c.validate_plan(value)
    return value


def bind(c, plan):
    plan["science_identity"] = c.science_identity(plan["mode"], plan["protocol"])
    intent = c.execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(c.CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(c.CONTROL / "student-name-invariant-v1-submissions" / intent),
        claim=str(c.CONTROL / "student-name-invariant-v1-claims" / (intent + ".json")),
        result_dir=str(c.PROJECT / "student-name-invariant-v1" / intent),
        job_name="opd-sni-" + intent,
    )
    plan["scientific_claim"] = str(c.scientific_claim_path(plan))


def preflight_audit(c, plan):
    value = dict(
        schema="quest-sdsc-student-name-invariant-raw-audit-v1",
        raw_replay_passed=True,
        mode="preflight",
        prompts_replayed=8,
        responses_replayed=8,
        preparation_complete=False,
        job_id="123",
        publication_sha256="7" * 64,
        result_sha256="8" * 64,
        plan_sha256=c.sha(c.canonical(plan)),
        auditor_sha256=plan["control_sha256"]["tools/sdsc_student_name_invariant_audit.py"],
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        implementation_commit=plan["protocol"]["implementation_commit"],
        acceptance_commit=plan["protocol"]["acceptance_commit"],
        execution_acceptance_claim=False,
        gpu_numerics_independently_recomputed=False,
        **dict.fromkeys(c.FLAGS, False),
    )
    return dict(document=value, sha256=c.sha(c.canonical(value) + b"\n"))


def test_resource_stage_limits(c, plan):
    assert c.resources("fit")["time"] == "08:00:00"
    assert c.resources("preflight")["time"] == "01:00:00"
    args = c.sbatch(plan)
    assert "--gpus=h100:2" in args and "--mem=393216M" in args and "--no-requeue" in args
    assert "--export=NONE" in args
    assert "--constraint=lustre" not in args


def test_pinned_publication_reserve_is_inside_total_walltime(c, plan):
    assert c.worker_budget_seconds(plan) == 3300
    plan["mode"] = "fit"
    assert c.worker_budget_seconds(plan) == 28200
    assert c.MAX_RESPONSE == 320 * c.CAP
    assert c.MAX_RESPONSE >= (c.MAX_FETCH * 4 + 2) // 3 + c.CAP


@pytest.mark.parametrize(
    "key,value",
    [
        ("publication_reserve_seconds", {"preflight": 300, "fit": 300}),
        ("maximum_small_fetch_mib", 192),
        ("maximum_persistent_large_gib", 129),
        ("minimum_node_local_free_gib", 128),
        ("fit_walltime_minutes", 600),
    ],
)
def test_node_rejects_unreviewed_execution_envelope(c, plan, monkeypatch, key, value):
    specification = copy.deepcopy(c.scientific_specification(plan))
    specification["execution"][key] = value
    monkeypatch.setattr(c, "scientific_specification", lambda _: specification)
    with pytest.raises(ValueError, match="execution/storage envelope"):
        c.worker_budget_seconds(plan)


def test_raw_inventory_covers_twelve_checkpoints_and_preflight_only_step4(c, plan):
    assert c.STEPS == (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)
    assert len(c.required_raw_names("fit")) == 29
    assert len(c.required_raw_names("preflight")) == 6
    published = set(c.required_raw_names("fit")) | {"prepare-report.json", "node-result.json", "memory.json"}
    assert published <= set(c.NAMES)
    value = report(c, plan)
    assert [row["step"] for row in value["checkpoints"]] == [4]
    value["checkpoints"].insert(0, dict(value["checkpoints"][0], step=1))
    with pytest.raises(ValueError, match="checkpoint schedule"):
        c.validate_worker_report(value, plan, "123", {})


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
    plan["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"] = "9" * 64
    plan["protocol"]["science_file_sha256"]["tools/sdsc_student_name_invariant_worker.py"] = "9" * 64
    bind(c, plan)
    assert plan["scientific_claim"] == before and plan["intent_id"] != old
    c.validate_plan(plan)


def test_fit_requires_exact_matching_preflight(c, plan):
    previous = copy.deepcopy(plan)
    plan.update(
        mode="fit",
        resources=c.resources("fit"),
        preflight=dict(
            plan=previous, plan_sha256=c.sha(c.canonical(previous)), audit=preflight_audit(c, previous)
        ),
    )
    bind(c, plan)
    c.validate_plan(plan)
    plan["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"] = "8" * 64
    plan["protocol"]["science_file_sha256"]["tools/sdsc_student_name_invariant_worker.py"] = "8" * 64
    bind(c, plan)
    with pytest.raises(ValueError, match="fit/preflight"):
        c.validate_plan(plan)


def test_fit_missing_preflight_rejected(c, plan):
    plan.update(mode="fit", resources=c.resources("fit"))
    bind(c, plan)
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "field",
    [
        "mode",
        "job_id",
        "publication_sha256",
        "result_sha256",
        "plan_sha256",
        "auditor_sha256",
        "protocol_sha256",
        "protocol_artifact_sha256",
        "implementation_commit",
        "acceptance_commit",
        "raw_replay_passed",
        "execution_acceptance_claim",
        "gpu_numerics_independently_recomputed",
    ],
)
def test_fit_rejects_invalid_preflight_audit(c, plan, field):
    audit = preflight_audit(c, plan)
    audit["document"][field] = "invalid"
    audit["sha256"] = c.sha(c.canonical(audit["document"]) + b"\n")
    with pytest.raises(ValueError):
        c.validate_preflight_audit(audit, plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_replay_passed", 1),
        ("gpu_numerics_independently_recomputed", True),
        ("gpu_numerics_independently_recomputed", 0),
        ("gpu_numerics_independently_recomputed", None),
        ("execution_acceptance_claim", 0),
    ],
)
def test_preflight_audit_rejects_overclaimed_or_untyped_scope(c, plan, field, value):
    audit = preflight_audit(c, plan)
    audit["document"][field] = value
    audit["sha256"] = c.sha(c.canonical(audit["document"]) + b"\n")
    with pytest.raises(ValueError):
        c.validate_preflight_audit(audit, plan)


def test_preflight_audit_binds_exact_bytes_and_publication(c, plan):
    audit = preflight_audit(c, plan)
    value = audit["document"]
    status = {key: value[key] for key in ("job_id", "publication_sha256", "result_sha256")}
    assert c.validate_preflight_audit(audit, plan, status) == value
    status["publication_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="persistent publication"):
        c.validate_preflight_audit(audit, plan, status)
    audit["sha256"] = c.sha(c.canonical(value))
    with pytest.raises(ValueError, match="audit bytes"):
        c.validate_preflight_audit(audit, plan)


def test_fit_needs_audit_before_submission(c, plan):
    previous = copy.deepcopy(plan)
    plan.update(
        mode="fit",
        resources=c.resources("fit"),
        preflight=dict(plan=previous, plan_sha256=c.sha(c.canonical(previous))),
    )
    bind(c, plan)
    with pytest.raises(ValueError, match="matching preflight"):
        c.validate_plan(plan)


def test_remote_entrypoint_refuses_local_slurm(c):
    result = subprocess.run(
        [os.sys.executable, "-I", "-B", str(ROOT / "tools/sdsc_student_name_invariant.py"), "remote"],
        input=b"{}",
        capture_output=True,
    )
    assert result.returncode != 0 and b"Slurm operations cannot run on Quest" in result.stderr


def test_launcher_uses_absolute_release_not_slurm_spool(c, plan, tmp_path):
    script = c.job_script(plan).decode()
    assert str(Path(plan["release"]) / "source/tools/sdsc_student_name_invariant_job.py") in script
    assert "dirname" not in script and "CUDA_VISIBLE_DEVICES=" not in script
    path = tmp_path / "job.sh"
    path.write_text(script)
    assert subprocess.run(["/bin/bash", "-n", str(path)]).returncode == 0


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_ssh_launcher_executes_actual_controller_or_rejects_hash_before_runpy(c, plan, tmp_path, wrong_hash):
    # Execute the exact controller-generated launcher against actual source.
    # Its real Quest-only guard is the stop point, before request handling or
    # any scheduler/SSH action. The fake CLI replaces only the SSH transport.
    release = tmp_path / "local-release"
    release.mkdir()
    (release / "source").symlink_to(ROOT, target_is_directory=True)
    plan["release"], plan["python"] = str(release), sys.executable
    controller_path = "tools/sdsc_student_name_invariant.py"
    controller_sha = c.sha((ROOT / controller_path).read_bytes())
    plan["control_sha256"][controller_path] = "0" * 64 if wrong_hash else controller_sha
    startup_sha = plan["control_sha256"]["tools/sdsc_student_name_invariant_startup.py"]
    assert startup_sha != controller_sha
    calls = []

    def local_ssh(argv, *, data, timeout):
        assert argv[:4] == [sys.executable, "-I", "-B", "-c"]
        assert argv[-2] == str(release / "source" / controller_path)
        assert json.loads(data) == dict(action="dry-run", plan=plan, authorize=False, dry_run=None)
        assert timeout == 240
        result = subprocess.run(argv, input=data, capture_output=True, timeout=10)
        calls.append(result)
        return result

    expected = "AssertionError" if wrong_hash else "Slurm operations cannot run on Quest"
    with pytest.raises(ValueError, match=expected):
        c.ssh_operation(SimpleNamespace(ssh_call=local_ssh), plan, "dry-run")
    assert len(calls) == 1 and calls[0].returncode != 0
    if wrong_hash:
        assert b"Slurm operations cannot run on Quest" not in calls[0].stderr
        assert b"sdsc_student_name_invariant.py" not in calls[0].stderr
    else:
        assert b"AssertionError" not in calls[0].stderr
        assert b"sdsc_student_name_invariant.py" in calls[0].stderr


def test_ssh_launcher_forwards_remote_argv_request_and_checked_named_payload(c, plan, tmp_path):
    release = tmp_path / "release"
    script = release / "source/tools/sdsc_student_name_invariant.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        "print(json.dumps(dict(argv=sys.argv,request=request,checked_payload_executed=True)))\n"
    )
    plan["release"], plan["python"] = str(release), sys.executable
    plan["control_sha256"]["tools/sdsc_student_name_invariant.py"] = c.sha(script.read_bytes())
    # Keep the real distinct startup pin: reordering TOOLS must not determine
    # which file's bytes authorize the controller path.
    calls = []

    def local_ssh(argv, *, data, timeout):
        calls.append((list(argv), data, timeout))
        return subprocess.run(argv, input=data, capture_output=True, timeout=10)

    proof = {"dry_run": True, "sentinel": "unchanged request"}
    result = c.ssh_operation(SimpleNamespace(ssh_call=local_ssh), plan, "submit", True, proof)
    assert result == dict(
        argv=[str(script), "remote"],
        request=dict(action="submit", plan=plan, authorize=True, dry_run=proof),
        checked_payload_executed=True,
    )
    assert len(calls) == 1


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


def test_no_teacher_store_in_worker_inputs(c, n, plan, tmp_path):
    science = tmp_path / "science"
    (science / c.PROTOCOL_PATH).parent.mkdir(parents=True)
    (science / c.PROTOCOL_PATH).write_text("{}")
    inputs = n.build_worker_inputs(
        plan, tmp_path, science, {"job_id": "123"}, dict.fromkeys((str(i) for i in range(49)), "a" * 64), c
    )
    assert inputs["mode"] == "preflight" and inputs["initial_checkpoint"]["sha256"] == c.INITIAL_SHA
    assert inputs["original_config"]["sha256"] == c.CONFIG_SHA
    assert "teacher_bundle_root" not in inputs


def report(c, plan):
    specification = c.scientific_specification(plan)
    data = specification["data"]
    training = specification["training"]
    return dict(
        schema="quest-sdsc-student-name-invariant-report-v1",
        mode="preflight",
        passed=True,
        execution_complete=True,
        preparation_complete=False,
        job_id="123",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        protocol_sha256=plan["protocol"]["protocol_sha256"],
        initial_checkpoint_sha256=c.INITIAL_SHA,
        optimizer_steps=4,
        global_batch_size=64,
        world_size=2,
        learning_rate=2.5e-5,
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
        fit_base_examples=data["fit_base_examples"],
        fit_training_views=data["fit_examples"],
        consumed_fit_training_views=256,
        development_base_examples=data["development_base_examples"],
        development_views=data["development_examples"],
        dataset_sha256=data["dataset_manifest_sha256"],
        development_generation=dict(
            max_new_tokens=256,
            max_model_input_length=2456,
            max_training_model_input_length=1536,
            native_bfloat16=True,
            explicit_autocast=False,
            do_sample=False,
            use_cache=False,
            seed="42_plus_manifest_index",
            rank_partition="complete_development_base_blocks_round_robin_rank",
            truncation=False,
        ),
        data_audit={
            "passed": True,
            "isolation": dict(
                passed=True,
                counts=dict(data["isolation_population_counts"]),
                excluded_view_counts=dict(data["isolation_excluded_view_counts"]),
                excluded_view_stream_sha256=dict.fromkeys(data["isolation_excluded_view_counts"], "a" * 64),
                dimensions=["semantic", "example_id", "pair_group_id", "pair_seed"],
                dataset_manifest_sha256=data["dataset_manifest_sha256"],
            ),
            "token_envelope": dict(
                window_input_tokens=[192] * training["optimizer_steps"],
                fit_rows=data["fit_examples"],
                optimizer_windows=training["optimizer_steps"],
                token_budget=training["input_token_budget"],
                total_input_tokens=192 * training["optimizer_steps"],
            ),
        },
        training_input_tokens=768,
        development=[],
        selected_checkpoint=None,
        protocol_artifact_sha256=plan["protocol"]["artifact_sha256"],
        teacher_data_required=False,
        source_kind="symbolic_canonical_name_invariant_preparation",
        checkpoint_restore={
            "passed": True,
            "ranks": [
                dict(
                    rank=i,
                    step=4,
                    passed=True,
                    actual_accelerate_save_load=True,
                    model_sha256="e" * 64,
                    optimizer_sha256="f" * 64,
                    runtime_sha256="a" * 64,
                )
                for i in range(2)
            ],
        },
        checkpoints=[
            dict(
                path="checkpoints/step-00000004.pt",
                step=4,
                size=12,
                sha256="c" * 64,
                scope="student_name_invariant_preparation_model_only",
                reload_by_rank=[
                    dict(
                        rank=i,
                        model_state_sha256="a" * 64,
                        exact_saved_master_reload=dict(
                            all_tensors_exact=True, loaded_before_bf16_copy=True, key_count=311
                        ),
                        inference_envelope_probe=dict(
                            passed=True,
                            input_length=2456,
                            token_id=17,
                            all_logits_finite=True,
                            use_cache=False,
                            native_bfloat16=True,
                            scope="synthetic_name_invariant_context_no_development_content",
                        ),
                        root_export_logits=[
                            dict(bitwise_equal=True, max_abs_error=0, rms_error=0, argmax_mismatches=0)
                        ]
                        * 2,
                    )
                    for i in range(2)
                ],
            )
        ],
        raw_artifacts=[
            dict(path=name, size=3, sha256="b" * 64) for name in sorted(c.required_raw_names("preflight"))
        ],
        **dict.fromkeys(c.FLAGS, False),
    )


def test_preflight_execution_does_not_claim_prepared_initial(c, plan):
    value = report(c, plan)
    c.validate_worker_report(value, plan, "123", {r["path"]: r for r in value["raw_artifacts"]})
    value["formal_initial_accepted"] = True
    with pytest.raises(ValueError):
        c.validate_worker_report(value, plan, "123", {})


def fit_report(c, plan):
    from collections import Counter

    from posttrain_circuits.experiments.protocols import student_name_invariant_preparation as protocol

    value = report(c, plan)
    previous = copy.deepcopy(plan)
    plan.update(
        mode="fit",
        resources=c.resources("fit"),
        preflight=dict(
            plan=previous, plan_sha256=c.sha(c.canonical(previous)), audit=preflight_audit(c, previous)
        ),
    )
    bind(c, plan)
    c.validate_plan(plan)
    value.update(
        mode="fit",
        preparation_complete=True,
        consumed_fit_training_views=2048,
        optimizer_steps=c.scientific_specification(plan)["training"]["optimizer_steps"],
        training_input_tokens=value["data_audit"]["token_envelope"]["total_input_tokens"],
        plan_sha256=c.sha(c.canonical(plan)),
    )
    value["raw_artifacts"] = [
        dict(path=name, size=3, sha256="b" * 64) for name in sorted(c.required_raw_names("fit"))
    ]
    prototype = value["checkpoints"][0]
    value["checkpoints"], value["development"] = [], []
    sizes = Counter(x.metadata["structure"] for x in protocol._build_bases()["student_dev"])
    for index, step in enumerate(c.STEPS):
        checkpoint = copy.deepcopy(prototype)
        checkpoint.update(step=step, path=f"checkpoints/step-{step:08d}.pt", sha256=c.sha(str(step).encode()))
        value["checkpoints"].append(checkpoint)
        structures = {
            name: dict(
                num_examples=size, answer_correct=size // 3, proof_correct=size // 3, format_valid=size
            )
            for name, size in sizes.items()
        }
        counts = {key: sum(row[key] for row in structures.values()) for key in protocol.COUNT_KEYS}
        views = {name: dict(counts) for name in protocol.DEV_VIEW_NAMES}
        if index == 0:
            views["fact_order_permutation"].update(proof_correct=0, answer_correct=0)
        value["development"].append(
            dict(
                step=step,
                checkpoint_sha256=checkpoint["sha256"],
                examples_sha256=protocol.EXAMPLES_SHA256["student_dev"],
                **counts,
                per_view_counts=views,
                structure_counts=structures,
            )
        )
    value["selected_checkpoint"] = copy.deepcopy(value["development"][1])
    return value


def test_fit_real_selector_accepts_earliest_robust_checkpoint(c, plan):
    value = fit_report(c, plan)
    c.validate_worker_report(value, plan, "123", {r["path"]: r for r in value["raw_artifacts"]})
    assert value["selected_checkpoint"]["step"] == 2
    value["selected_checkpoint"] = copy.deepcopy(value["development"][0])
    with pytest.raises(ValueError, match="earliest fully eligible"):
        c.validate_worker_report(value, plan, "123", {r["path"]: r for r in value["raw_artifacts"]})


def test_fit_real_selector_rejects_truncated_schedule(c, plan):
    value = fit_report(c, plan)
    value["development"].pop()
    with pytest.raises(ValueError, match="scheduled development"):
        c.validate_worker_report(value, plan, "123", {})


def test_node_publishes_every_checkpoint_response_shard_and_full_state(c, n, plan, tmp_path):
    value = fit_report(c, plan)
    artifacts = tmp_path / "artifacts"
    (artifacts / "checkpoints").mkdir(parents=True)
    (artifacts / "resume").mkdir()
    for checkpoint in value["checkpoints"]:
        raw = str(checkpoint["step"]).encode()
        (artifacts / checkpoint["path"]).write_bytes(raw)
        checkpoint.update(size=len(raw), sha256=c.sha(raw))
    resume = artifacts / "resume/step-00000004"
    resume.mkdir()
    for name in (
        "custom_checkpoint_0.pkl",
        "optimizer.bin",
        "pytorch_model_fsdp.bin",
        "random_states_0.pkl",
        "random_states_1.pkl",
        "runtime-rank-0.json",
        "runtime-rank-1.json",
    ):
        (resume / name).write_bytes(b"actual-small-state-fixture")
    for name in c.EXECUTION_NAMES:
        (artifacts / name).write_bytes(b"{}\n")
    for name in c.required_raw_names("fit"):
        (artifacts / name).write_bytes(b"{}\n")
    (artifacts / "prepare-report.json").write_bytes(c.canonical(value))
    Path(plan["result_dir"]).mkdir(parents=True)
    n.persist(
        plan,
        c,
        artifacts,
        None,
        dict(job_id="123", stage_complete=True, exit_code=0),
        {},
        lambda _: None,
    )
    publication = c.document(Path(plan["result_dir"]) / "receipt.json")
    records = c.publication_records(publication, plan, "123")
    assert publication["passed"] is True and len(records) == 39
    assert len(publication["large_files"]) == 19
    assert {row["path"] for row in publication["large_files"] if row["path"].startswith("checkpoints/")} == {
        f"checkpoints/step-{step:08d}.pt" for step in c.STEPS
    }
    for row in [*publication["files"], *publication["large_files"]]:
        raw = (Path(plan["result_dir"]) / row["path"]).read_bytes()
        assert len(raw) == row["size"] and c.sha(raw) == row["sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_rank_probe",
        "old_context",
        "extra_autocast",
        "few_master_keys",
        "old_training_count",
        "missing_previous_population",
        "missing_old_views",
        "wrong_old_view_count",
        "missing_old_view_digest",
        "invalid_old_view_digest",
        "missing_branch_population",
        "missing_branch_views",
        "wrong_branch_view_count",
        "missing_branch_view_digest",
    ],
)
def test_v4_report_rejects_partial_execution_contract(c, plan, change):
    value = report(c, plan)
    if change == "missing_rank_probe":
        value["checkpoints"][0]["reload_by_rank"][1]["inference_envelope_probe"] = None
    elif change == "old_context":
        value["development_generation"]["max_model_input_length"] = 1600
    elif change == "extra_autocast":
        value["development_generation"]["explicit_autocast"] = True
    elif change == "few_master_keys":
        value["checkpoints"][0]["reload_by_rank"][0]["exact_saved_master_reload"]["key_count"] = 310
    elif change == "old_training_count":
        value["data_audit"]["token_envelope"]["fit_rows"] = 8192
    elif change == "missing_previous_population":
        value["data_audit"]["isolation"]["counts"].pop("previous_student_dev")
    elif change == "missing_old_views":
        value["data_audit"]["isolation"].pop("excluded_view_counts")
    elif change == "wrong_old_view_count":
        value["data_audit"]["isolation"]["excluded_view_counts"]["invariance_student_fit_views"] = 2048
    elif change == "missing_old_view_digest":
        value["data_audit"]["isolation"]["excluded_view_stream_sha256"].pop("invariance_student_dev_views")
    elif change == "missing_branch_population":
        value["data_audit"]["isolation"]["counts"].pop("branch_student_fit")
    elif change == "missing_branch_views":
        value["data_audit"]["isolation"]["excluded_view_counts"].pop("branch_student_dev_views")
    elif change == "wrong_branch_view_count":
        value["data_audit"]["isolation"]["excluded_view_counts"]["branch_student_fit_views"] = 8192
    elif change == "missing_branch_view_digest":
        value["data_audit"]["isolation"]["excluded_view_stream_sha256"].pop("branch_student_fit_views")
    else:
        value["data_audit"]["isolation"]["excluded_view_stream_sha256"]["invariance_student_dev_views"] = None
    with pytest.raises(ValueError):
        c.validate_worker_report(value, plan, "123", {r["path"]: r for r in value["raw_artifacts"]})


def test_node_cannot_accept_partial_raw_evidence(c, plan):
    value = report(c, plan)
    value["raw_artifacts"].pop()
    with pytest.raises(ValueError, match="complete mode-specific raw inventory"):
        c.validate_worker_report(value, plan, "123", {r["path"]: r for r in value["raw_artifacts"]})


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "quest-sdsc-student-branch-report-v3"),
        ("source_kind", "symbolic_canonical_branch_preparation"),
        ("optimizer_steps", 3),
        ("learning_rate", 5e-4),
        ("protocol_sha256", "a" * 64),
        ("world_size", 1),
        ("preparation_complete", True),
    ],
)
def test_report_rejects_wrong_experiment(c, plan, field, value):
    payload = report(c, plan)
    payload[field] = value
    with pytest.raises(ValueError):
        c.validate_worker_report(payload, plan, "123", {})


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


@pytest.mark.parametrize(
    "change", ["token_total", "missing_population", "missing_rank", "parity", "selection"]
)
def test_report_rejects_missing_real_evidence(c, plan, change):
    value = report(c, plan)
    if change == "token_total":
        value["training_input_tokens"] -= 1
    elif change == "missing_population":
        value["data_audit"]["isolation"]["counts"]["original_family"] = 256
    elif change == "missing_rank":
        value["checkpoint_restore"]["ranks"].pop()
    elif change == "parity":
        value["checkpoints"][0]["reload_by_rank"][0]["root_export_logits"][0]["bitwise_equal"] = False
    else:
        value["selected_checkpoint"] = {"step": 4}
    with pytest.raises(ValueError):
        c.validate_worker_report(value, plan, "123", {r["path"]: r for r in value["raw_artifacts"]})


def test_fixed_runtime_checks_all_nineteen_packages(c, plan, monkeypatch):
    identity = dict(python="3.12.13", executable=plan["python"], packages=dict(c.RUNTIME_PACKAGES))
    monkeypatch.setattr(c, "run", lambda *a, **kw: dict(returncode=0, stdout=json.dumps(identity), stderr=""))
    assert c.verify_runtime(plan) == identity and len(identity["packages"]) == 19
    identity["packages"]["accelerate"] = "new-version"
    with pytest.raises(ValueError, match="package identity"):
        c.verify_runtime(plan)


def test_deployed_worker_must_match_reviewed_implementation(c, plan):
    plan["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"] = "8" * 64
    bind(c, plan)
    with pytest.raises(ValueError, match="accepted scientific byte"):
        c.validate_plan(plan)


def test_missing_executed_science_inventory_rejected(c, plan):
    plan["protocol"]["science_file_sha256"].pop("tools/sdsc_student_lr_probe.py")
    with pytest.raises(ValueError, match="lacks executed"):
        c.validate_plan(plan)


def test_new_raw_read_bound_does_not_change_old_helper(c, tmp_path):
    path = tmp_path / "complete-raw.jsonl"
    raw = b"x" * (8 * c.CAP + 1)
    path.write_bytes(raw)
    assert c.read(path) == raw
    with pytest.raises(ValueError):
        c._io.read(path)
    assert c.MAX_FILE == 16 * c.CAP and c.MAX_FETCH == 224 * c.CAP


def test_prompt_only_read_bound_is_larger(c, tmp_path):
    prompt = tmp_path / "prepare-dev-prompts.jsonl"
    prompt.write_bytes(b"x" * (c.MAX_FILE + 1))
    assert len(c.read(prompt)) == c.MAX_FILE + 1
    other = tmp_path / "prepare-dev-records-step-00000004-rank-0.jsonl"
    other.write_bytes(prompt.read_bytes())
    with pytest.raises(ValueError):
        c.read(other)
    with prompt.open("wb") as stream:
        stream.truncate(c.MAX_PROMPTS + 1)
    with pytest.raises(ValueError):
        c.read(prompt)


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
        c._io.document(path)
    with pytest.raises(ValueError, match="SHA differs"):
        c.document(path, "a" * 64)


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
    assert value["passed"] is False and value["preparation_complete"] is False and "arm" not in value
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
    node = load("sdsc_student_name_invariant_job")
    probe = load("sdsc_student_name_invariant_startup")
    helpers = {"sdsc_student_name_invariant_startup": probe}

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
    script = source / "sdsc_student_name_invariant_startup.py"
    script.write_text(
        """import importlib.util,sys,platform,os,time
from types import SimpleNamespace
path=sys.argv.pop(1) if False else """
        + repr(str(ROOT / "tools/sdsc_student_name_invariant_startup.py"))
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
            "tools/sdsc_student_name_invariant_worker.py": hashlib.sha256(
                (ROOT / "tools/sdsc_student_name_invariant_worker.py").read_bytes()
            ).hexdigest()
        },
    }
    identity = {"job_id": "12345", "cuda_visible_devices": "2,3"}

    def fixture_probe_argv(plan, context, stage):
        return [
            stage["python"],
            "-I",
            "-B",
            "-u",
            "-X",
            "importtime",
            str(script),
            "probe",
            "--worker-sha256",
            plan["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"],
            "--output-json",
            str(execution / "early-node-startup.json"),
            "--job-id",
            identity["job_id"],
            "--expected-cuda-visible-devices",
            identity["cuda_visible_devices"],
        ]

    helpers["sdsc_student_name_invariant_native"] = SimpleNamespace(probe_argv=fixture_probe_argv)
    monkeypatch.setattr(
        node, "probe_argv", lambda plan, path, ident: fixture_probe_argv(plan, {}, {"python": plan["python"]})
    )
    return node, probe, c, outer, identity, execution, environment


def run(startup_setup, mode="good", budget=20):
    node, probe, c, outer, identity, execution, environment = startup_setup
    environment["FIXTURE_MODE"] = mode
    meta = node.run_probe(
        outer,
        c,
        execution,
        identity,
        environment,
        deadline=time.monotonic() + budget,
        measure=lambda _: None,
        context={},
        runtime_stage={"python": outer["python"]},
    )
    probe.validate_trace_meta(
        meta,
        worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"],
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
        worker_sha256=outer["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"],
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
            worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"],
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            expected_python=sys.executable,
            log_bytes=(startup_setup[5] / "startup.log").read_bytes(),
        )


def test_source_staging_has_explicit_manifest_inputs_and_readonly_modes(tmp_path):
    node = load("sdsc_student_name_invariant_job")
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


def test_actual_node_initial_failure_publishes_negative_evidence(c, plan, tmp_path, monkeypatch):
    node = load("sdsc_student_name_invariant_job")
    identity = dict(job_id="12345", cuda_visible_devices="0,1")
    plan["parent"]["receipt"]["result_dir"] = str(tmp_path / "parent")
    monkeypatch.setattr(node, "load_control", lambda *a: (c, plan))
    monkeypatch.setattr(node, "allocation", lambda *a: identity)
    monkeypatch.setattr(c, "verify_source", lambda *a: None)
    monkeypatch.setattr(c, "verify_parent", lambda *a: None)
    monkeypatch.setattr(node, "mount", lambda *a: dict(fstype="lustre"))
    monkeypatch.setattr(node.signal, "signal", lambda *a: None)
    monkeypatch.setattr(
        node, "memory_envelope", lambda *a: (_ for _ in ()).throw(ValueError("fixture initial memory guard"))
    )
    monkeypatch.setattr(
        node.subprocess, "Popen", lambda *a, **kw: pytest.fail("initial failure must not launch child")
    )
    assert node.main(["fixture", "0" * 64]) == 1
    destination = Path(plan["result_dir"])
    publication = c.document(destination / "receipt.json")
    rows = c.publication_records(publication, plan, "12345")
    assert set(rows) == {"node-result.json", "memory.json"}
    assert publication["passed"] is False
    assert "fixture initial memory guard" in c.document(destination / "node-result.json")["error"]


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
    expected_sha = plan["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"]
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
    for step in range(1, report["optimizer_steps"] + 1):
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
            worker_sha256=startup_setup[3]["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"],
            job_id="12345",
            expected_cuda_visible_devices="2,3",
            expected_python=sys.executable,
            log_bytes=raw,
        )


@pytest.mark.parametrize("mode", ["preflight", "fit"])
def test_raw_updates_validate_exact_mode_and_actual_prefix(c, plan, mode):
    value = report(c, plan) if mode == "preflight" else fit_report(c, plan)
    rows = update_records(value)
    evidence = c.validate_update_records(b"".join(c.canonical(row) + b"\n" for row in rows), value)
    assert evidence["optimizer_steps"] == (4 if mode == "preflight" else 32)
    assert evidence["global_training_views"] == evidence["optimizer_steps"] * 64
    assert evidence["training_input_tokens"] == value["training_input_tokens"]


@pytest.mark.parametrize("mode", ["preflight", "fit"])
@pytest.mark.parametrize(
    "change",
    [
        "missing_final",
        "rank_slot",
        "update_norm",
        "local_tokens",
        "global_tokens",
        "cumulative",
        "calls",
        "rank",
        "nonfinite",
        "budget_bool",
    ],
)
def test_rehashed_raw_update_contradiction_cannot_complete_mode(c, plan, mode, change):
    value = report(c, plan) if mode == "preflight" else fit_report(c, plan)
    rows = update_records(value)
    row = rows[-1]["ranks"][1]
    if change == "missing_final":
        rows.pop()
    elif change == "rank_slot":
        row["global_slots"][-1] -= 64
    elif change == "update_norm":
        row["metric"]["parameter_update_norm"] = 0
    elif change == "local_tokens":
        row["metric"]["local_model_facing_input_tokens_this_update"] += 1
    elif change == "global_tokens":
        row["global_input_tokens"] += 1
    elif change == "cumulative":
        row["metric"]["supervised_response_tokens"] -= 1
    elif change == "calls":
        row["optimizer_calls"][0] += 1
    elif change == "rank":
        row["rank"] = 0
    elif change == "nonfinite":
        row["metric"]["verified_replay_loss"] = float("nan")
    else:
        row["token_budget"]["accepted_optimizer_updates"] = True
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    assert len(c.sha(raw)) == 64
    with pytest.raises(ValueError):
        c.validate_update_records(raw, value)


@pytest.mark.parametrize("mode,expected", [("preflight", 8), ("fit", 19)])
def test_positive_flat_publication_has_complete_resume_and_mode_dense_files(
    c, n, plan, tmp_path, mode, expected
):
    value = report(c, plan) if mode == "preflight" else fit_report(c, plan)
    artifacts = tmp_path / "complete-artifacts"
    (artifacts / "checkpoints").mkdir(parents=True)
    resume = artifacts / "resume/step-00000004"
    resume.mkdir(parents=True)
    for name in (
        "custom_checkpoint_0.pkl",
        "optimizer.bin",
        "pytorch_model_fsdp.bin",
        "random_states_0.pkl",
        "random_states_1.pkl",
        "runtime-rank-0.json",
        "runtime-rank-1.json",
    ):
        (resume / name).write_bytes(b"fixture-state")
    for cp in value["checkpoints"]:
        raw = str(cp["step"]).encode()
        (artifacts / cp["path"]).write_bytes(raw)
        cp.update(size=len(raw), sha256=c.sha(raw))
    for name in c.required_raw_names(mode) | set(c.EXECUTION_NAMES):
        (artifacts / name).write_bytes(b"{}\n")
    (artifacts / "prepare-report.json").write_bytes(c.canonical(value))
    Path(plan["result_dir"]).mkdir(parents=True)
    n.persist(
        plan, c, artifacts, None, dict(job_id="123", stage_complete=True, exit_code=0), {}, lambda _: None
    )
    published = c.document(Path(plan["result_dir"]) / "receipt.json")
    c.publication_records(published, plan, "123")
    assert len(published["large_files"]) == expected
    assert len(published["files"]) == (16 if mode == "preflight" else 39)
    assert published["preparation_complete"] is (mode == "fit")
    assert not any(k in published for k in ("arm", "selected_checkpoint", "diagnostic_complete"))
    c.validate_large_outputs(value, published["large_files"])
    with pytest.raises(ValueError):
        c.validate_large_outputs(value, published["large_files"][:-1])


def test_failed_full_fit_remains_negative_publication_with_complete_raw_report(c, n, plan, tmp_path):
    value = fit_report(c, plan)
    value.update(passed=False, selected_checkpoint=None)
    artifacts = tmp_path / "failed-fit"
    artifacts.mkdir()
    (artifacts / "prepare-report.json").write_bytes(c.canonical(value))
    (artifacts / "prepare-updates.jsonl").write_bytes(
        b"".join(c.canonical(row) + b"\n" for row in update_records(value))
    )
    Path(plan["result_dir"]).mkdir(parents=True)
    n.persist(
        plan, c, artifacts, None, dict(job_id="123", stage_complete=False, exit_code=1), {}, lambda _: None
    )
    published = c.document(Path(plan["result_dir"]) / "receipt.json")
    c.publication_records(published, plan, "123")
    assert published["passed"] is published["stage_complete"] is published["preparation_complete"] is False
    retained = c.document(Path(plan["result_dir"]) / "prepare-report.json")
    assert retained["execution_complete"] is retained["preparation_complete"] is True
    assert retained["passed"] is False and retained["selected_checkpoint"] is None
    with pytest.raises(ValueError):
        c.validate_worker_report(retained, plan, "123", {})


def test_worker_launch_uses_new_pinned_adapter_and_science_cwd(c, n, plan, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(n.subprocess, "Popen", lambda argv, **kw: (calls.append((argv, kw)) or object()))
    work = tmp_path / "work"
    science = work / "science"
    source = work / "source"
    artifacts = work / "artifacts"
    artifacts.mkdir(parents=True)
    stage = dict(python=str(work / "runtime/bin/python3.12"))
    context = dict(source_dir=str(source), science_dir=str(science), job_id="123", work_dir=str(work))
    (artifacts / "runtime-stage.json").write_bytes(c.canonical(stage))
    (artifacts / "runtime-context.json").write_bytes(c.canonical(context))
    env = {"CUDA_VISIBLE_DEVICES": "5,3", "OMP_NUM_THREADS": "12"}
    identity = {"job_id": "123", "cuda_visible_devices": "5,3"}
    n.start_worker(
        plan,
        source,
        science,
        work / "inputs.json",
        work / "artifacts",
        work / "artifacts",
        identity,
        env,
        None,
    )
    argv, kw = calls[0]
    assert argv[0] == stage["python"] and plan["python"] not in argv
    assert str(source / "tools/sdsc_student_name_invariant_native.py") in argv
    assert argv[argv.index("--phase") + 1] == "run"
    assert argv[argv.index("--plan-sha256") + 1] == c.sha(c.canonical(plan))
    assert argv[argv.index("--num_processes") + 1] == "2"
    assert kw["cwd"] == science and kw["env"] == env and kw["start_new_session"] is True
    assert argv[-2:] == ["--context", str(artifacts / "runtime-context.json")]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "quest-sdsc-student-order-raw-audit-v4"),
        ("responses_replayed", 128),
        ("prompts_replayed", True),
        ("preparation_complete", True),
    ],
)
def test_matching_preflight_requires_new_exact_eight_raw_audit(c, plan, field, value):
    record = preflight_audit(c, plan)
    record["document"][field] = value
    record["sha256"] = c.sha(c.canonical(record["document"]) + b"\n")
    with pytest.raises(ValueError):
        c.validate_preflight_audit(record, plan)


@pytest.fixture
def progress_setup(c, n, plan, tmp_path):
    artifacts = tmp_path / "live-artifacts"
    artifacts.mkdir()
    Path(plan["result_dir"]).mkdir(parents=True)
    value = report(c, plan)
    row = update_records(value)[0]
    (artifacts / "prepare-updates.jsonl").write_bytes(c.canonical(row) + b"\n")
    return artifacts, row, dict(job_id="123")


def test_progress_atomic_replace_is_bounded_and_not_completion(c, n, plan, progress_setup):
    artifacts, row, identity = progress_setup
    first = n.publish_live_progress(plan, c, identity, artifacts, "running-123", 1.0)
    assert first["latest_update"]["available"] is True and first["latest_update"]["step"] == 1
    assert first["phase"] == "training_or_evaluation" and first["completion_claim"] is False
    assert first["latest_update"]["ranks"][0]["loss"] == row["ranks"][0]["metric"]["verified_replay_loss"]
    assert first["plan_sha256"] == c.sha(c.canonical(plan))
    second = n.publish_live_progress(plan, c, identity, artifacts, "final", 2.0)
    out = Path(plan["result_dir"]) / "live-progress.json"
    assert c.document(out) == second and out.stat().st_size <= 16 * 1024
    assert second["completion_claim"] is False and all(second[k] is False for k in c.FLAGS)
    assert "live-progress.json" not in c.NAMES
    assert not list(out.parent.glob(".live-progress-*.tmp"))
    with pytest.raises(ValueError):
        c.publication_records(second, plan, "123")


def test_live_progress_accepts_actual_early_cuda_memory_phase(c, n, plan, progress_setup):
    artifacts, _, identity = progress_setup
    value = n.publish_live_progress(plan, c, identity, artifacts, "cuda-probe-30", 30.0)
    assert value["phase"] == "early_cuda_probe" and value["completion_claim"] is False
    assert c.document(Path(plan["result_dir"]) / "live-progress.json") == value


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "empty",
        "unfinished",
        "invalid_json",
        "huge_line",
        "bad_rank",
        "bad_tokens",
        "bool_step",
        "nan_loss",
    ],
)
def test_live_progress_unavailable_on_incomplete_or_invalid_observation(c, n, plan, progress_setup, case):
    artifacts, row, identity = progress_setup
    path = artifacts / "prepare-updates.jsonl"
    if case == "missing":
        path.unlink()
    elif case == "empty":
        path.write_bytes(b"")
    elif case == "unfinished":
        path.write_bytes(c.canonical(row))
    elif case == "invalid_json":
        path.write_bytes(b"{not JSON}\n")
    elif case == "huge_line":
        path.write_bytes(b"x" * (16 * 1024 + 1) + b"\n")
    else:
        if case == "bad_rank":
            row["ranks"][1]["rank"] = 0
        elif case == "bad_tokens":
            row["ranks"][1]["token_budget"]["consumed"] += 1
        elif case == "bool_step":
            row["step"] = True
        else:
            row["ranks"][0]["metric"]["verified_replay_loss"] = float("nan")
        path.write_bytes(json.dumps(row).encode() + b"\n")
    observed = n.publish_live_progress(plan, c, identity, artifacts, "running-1", 1)
    assert observed["latest_update"]["available"] is False and observed["completion_claim"] is False


def test_live_progress_drops_partial_tail_and_never_reads_weights(c, n, plan, progress_setup):
    artifacts, row, identity = progress_setup
    path = artifacts / "prepare-updates.jsonl"
    path.write_bytes(c.canonical(row) + b"\n" + b'{"step":2')
    (artifacts / "weights.pt").write_bytes(b"not-a-progress-source")
    observed = n.publish_live_progress(plan, c, identity, artifacts, "running-1", 1)
    assert observed["latest_update"]["step"] == 1 and observed["latest_update"]["partial_last_line"] is True
    assert "not-a-progress-source" not in c.canonical(observed).decode()


def test_live_progress_reads_only_last_bounded_tail(c, n, plan, progress_setup):
    artifacts, row, identity = progress_setup
    raw = b"x" * n.LIVE_UPDATE_TAIL_BYTES + b"\n" + c.canonical(row) + b"\n"
    (artifacts / "prepare-updates.jsonl").write_bytes(raw)
    observed = n.publish_live_progress(plan, c, identity, artifacts, "running-1", 1)
    assert observed["latest_update"]["available"] is True


@pytest.mark.parametrize("case", ["symlink", "fifo", "oversize", "bad_phase"])
def test_live_progress_rejects_unsafe_input_without_overwriting_prior(
    c, n, plan, progress_setup, case, tmp_path
):
    artifacts, row, identity = progress_setup
    path = artifacts / "prepare-updates.jsonl"
    path.unlink()
    if case == "symlink":
        target = tmp_path / "elsewhere"
        target.write_bytes(c.canonical(row) + b"\n")
        path.symlink_to(target)
    elif case == "fifo":
        os.mkfifo(path)
    elif case == "oversize":
        with path.open("wb") as stream:
            stream.truncate(c.MAX_FILE + 1)
    else:
        path.write_bytes(c.canonical(row) + b"\n")
    destination = Path(plan["result_dir"]) / "live-progress.json"
    destination.write_bytes(b"prior-observation")
    with pytest.raises((ValueError, OSError)):
        n.publish_live_progress(
            plan, c, identity, artifacts, "claimed_success" if case == "bad_phase" else "running-1", 1
        )
    assert destination.read_bytes() == b"prior-observation"


@pytest.mark.parametrize("mode", ["preflight", "fit"])
def test_actual_completed_core_isolation_enters_report_validator(c, plan, mode):
    # Integration evidence is optional on a fresh checkout, never synthesized
    # from the consumer fixture. This session has the complete144k producer audit.
    artifact = ROOT / ".sdsc/diagnostics/student-name-invariant-preparation-v1/full-isolation-author.json"
    if not artifact.is_file():
        pytest.skip("requires the completed original144k/history isolation producer evidence")
    actual = json.loads(artifact.read_bytes())
    assert actual["no_gpu_execution"] is True
    isolation = actual["audit"]
    assert isolation["passed"] is True and isolation["counts"]["original_family"] == 144000
    assert "dataset_manifest_sha256" in isolation and "preparation_manifest_sha256" not in isolation
    value = report(c, plan) if mode == "preflight" else fit_report(c, plan)
    value["data_audit"]["isolation"] = copy.deepcopy(isolation)
    inventory = {row["path"]: row for row in value["raw_artifacts"]}
    c.validate_worker_report(value, plan, "123", inventory)
    value["data_audit"]["isolation"]["preparation_manifest_sha256"] = value["data_audit"]["isolation"].pop(
        "dataset_manifest_sha256"
    )
    with pytest.raises(ValueError, match="disjoint preparation"):
        c.validate_worker_report(value, plan, "123", inventory)


def test_science_restore_uses_explicit_v2_consumer_and_exact_binding(c, plan, tmp_path, monkeypatch):
    calls = []
    destination = tmp_path / "restored-science"

    def verify(directory, digest, code, restored):
        calls.append((directory, digest, code, restored))
        return {"git_head": plan["provenance"]["head"]}

    helper = c.helper
    monkeypatch.setattr(
        c,
        "helper",
        lambda name: SimpleNamespace(verify=verify) if name == "sdsc_provenance_v2" else helper(name),
    )
    monkeypatch.setattr(c, "protocol_binding", lambda root, head: copy.deepcopy(plan["protocol"]))
    assert c.verify_science(plan, destination)["git_head"] == plan["provenance"]["head"]
    assert calls == [
        (
            Path(plan["provenance"]["directory"]),
            plan["provenance"]["manifest_sha256"],
            plan["code_sha256"],
            destination,
        )
    ]
    monkeypatch.setattr(
        c, "protocol_binding", lambda root, head: dict(plan["protocol"], acceptance_commit="f" * 40)
    )
    with pytest.raises(ValueError, match="restored accepted successor"):
        c.verify_science(plan, destination)


@pytest.mark.parametrize("field", ["manifest", "archive", "runtime_root_sha256", "source_prefix"])
def test_plan_fixed_runtime_identity_cannot_drift(c, plan, field):
    value = copy.deepcopy(plan)
    if field in ("manifest", "archive"):
        value["runtime_snapshot"][field]["sha256"] = "f" * 64
    else:
        value["runtime_snapshot"][field] += "changed"
    with pytest.raises(ValueError, match="runtime snapshot"):
        c.validate_runtime_snapshot(value)


def test_legacy_provenance_namespace_is_rejected(c, plan):
    plan["provenance"]["directory"] = str(c.CONTROL / "provenance" / plan["provenance"]["manifest_sha256"])
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_failed_large_inventory_keeps_all_bounded_origin_and_raw_evidence(c, n, plan, tmp_path, monkeypatch):
    artifacts = tmp_path / "retained-artifacts"
    artifacts.mkdir()
    names = (*c.NATIVE_NAMES, "prepare-report.json", "prepare-updates.jsonl")
    for name in names:
        (artifacts / name).write_bytes(b'{"retained":true}\n')
    destination = Path(plan["result_dir"])
    destination.mkdir(parents=True)

    def fail_inventory(path, control):
        assert all((destination / name).read_bytes() == (artifacts / name).read_bytes() for name in names)
        raise ValueError("fixture incomplete checkpoint file")

    monkeypatch.setattr(n, "original_node", lambda _: SimpleNamespace(large_inventory=fail_inventory))
    result = dict(job_id="123", stage_complete=False, exit_code=1)
    n.persist(plan, c, artifacts, None, result, {}, lambda _: None)
    published = c.document(destination / "receipt.json")
    records = c.publication_records(published, plan, "123")
    assert set(names) <= records.keys() and published["passed"] is False
    assert (
        "fixture incomplete checkpoint file"
        in c.document(destination / "node-result.json")["checkpoint_publication_error"]
    )


def test_runtime_staging_progress_remains_observation_only(c, n, plan, progress_setup):
    artifacts, _, identity = progress_setup
    observed = n.publish_live_progress(plan, c, identity, artifacts, "runtime-staging-123", 30)
    assert observed["phase"] == "native_runtime_staging"
    assert observed["completion_claim"] is False
    assert all(observed[name] is False for name in c.FLAGS)
