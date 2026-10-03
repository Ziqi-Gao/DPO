"""No-network transport and real filesystem fixtures for common preparation."""

from __future__ import annotations

import base64
import copy
import importlib.util
import json
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
        if Path(command).name in {"sbatch", "sacct", "squeue", "scontrol", "srun", "scancel", "ssh"}:
            pytest.fail("CPU fixture attempted real remote/Slurm command")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    return load("sdsc_student_order")


@pytest.fixture
def n():
    return load("sdsc_student_order_job")


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    pins = dict.fromkeys(c.TOOLS, "a" * 64)
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
    contract = c.helper("sdsc_student_contract")
    value = dict(
        schema="quest-sdsc-student-order-plan-v4",
        task=c.TASK,
        mode="preflight",
        run_id="new-preparation-run",
        code_sha256="c" * 64,
        manifest_sha256="d" * 64,
        resources=c.resources("preflight"),
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
        submission_dir=str(c.CONTROL / "student-order-submissions" / intent),
        claim=str(c.CONTROL / "student-order-claims" / (intent + ".json")),
        result_dir=str(c.PROJECT / "student-order" / intent),
        job_name="opd-sor-" + intent,
    )
    plan["scientific_claim"] = str(c.scientific_claim_path(plan))


def preflight_audit(c, plan):
    value = dict(
        raw_replay_passed=True,
        mode="preflight",
        job_id="123",
        publication_sha256="7" * 64,
        result_sha256="8" * 64,
        plan_sha256=c.sha(c.canonical(plan)),
        auditor_sha256=plan["control_sha256"]["tools/sdsc_student_order_audit.py"],
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
    plan["control_sha256"]["tools/sdsc_student_order_worker.py"] = "9" * 64
    plan["protocol"]["science_file_sha256"]["tools/sdsc_student_order_worker.py"] = "9" * 64
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
    plan["control_sha256"]["tools/sdsc_student_order_worker.py"] = "8" * 64
    plan["protocol"]["science_file_sha256"]["tools/sdsc_student_order_worker.py"] = "8" * 64
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
        [os.sys.executable, "-I", "-B", str(ROOT / "tools/sdsc_student_order.py"), "remote"],
        input=b"{}",
        capture_output=True,
    )
    assert result.returncode != 0 and b"Slurm operations cannot run on Quest" in result.stderr


def test_launcher_uses_absolute_release_not_slurm_spool(c, plan, tmp_path):
    script = c.job_script(plan).decode()
    assert str(Path(plan["release"]) / "source/tools/sdsc_student_order_job.py") in script
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


def test_checkpoint_full_copy_readback_and_no_overwrite(c, n, tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"actual state\0" * 500)
    row = dict(path="resume/rank-0.pt", size=source.stat().st_size, sha256=c.sha(source.read_bytes()))
    target = tmp_path / "output" / "rank-0.pt"
    assert n.copy_checkpoint(source, target, row, c) == row
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        n.copy_checkpoint(source, target, row, c)


@pytest.mark.parametrize("bad", ["hash", "size", "symlink"])
def test_checkpoint_rejects_bad_input(c, n, tmp_path, bad):
    path = tmp_path / "input"
    path.write_bytes(b"state")
    row = dict(path="checkpoints/state.pt", size=5, sha256=c.sha(b"state"))
    if bad == "hash":
        row["sha256"] = "a" * 64
    elif bad == "size":
        row["size"] = 4
    else:
        link = tmp_path / "link"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError):
        n.copy_checkpoint(path, tmp_path / "output", row, c)


def test_large_inventory_rejects_external_symlink(c, n, tmp_path):
    root = tmp_path / "artifacts"
    (root / "resume").mkdir(parents=True)
    (root / "resume" / "unsafe").symlink_to("/etc/passwd")
    with pytest.raises(ValueError):
        n.large_inventory(root, c)


def test_worker_starts_science_cwd_and_preserves_cuda(c, n, plan, tmp_path, monkeypatch):
    source = tmp_path / "source"
    science = tmp_path / "science"
    science.mkdir()
    source.mkdir()
    capture = {}

    def popen(argv, **kwargs):
        capture.update(argv=argv, **kwargs)
        return object()

    monkeypatch.setattr(n.subprocess, "Popen", popen)
    environment = {"CUDA_VISIBLE_DEVICES": "GPU-uuid-2,GPU-uuid-7"}
    n.start_worker(plan, source, science, tmp_path / "inputs", tmp_path / "artifacts", environment, None)
    assert capture["cwd"] == science and capture["start_new_session"] is True
    assert capture["env"] == environment and "--num_processes" in capture["argv"]
    assert capture["argv"][capture["argv"].index("--config_file") + 1] == str(
        science / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml"
    )


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
        schema="quest-sdsc-student-order-report-v4",
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
        learning_rate=5e-5,
        fit_base_examples=data["fit_base_examples"],
        fit_training_views=data["fit_examples"],
        development_base_examples=data["development_base_examples"],
        development_views=data["development_examples"],
        dataset_sha256=data["dataset_manifest_sha256"],
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
        data_audit={
            "passed": True,
            "isolation": dict(
                passed=True,
                counts=dict(data["isolation_population_counts"]),
                excluded_view_counts=dict(data["isolation_excluded_view_counts"]),
                excluded_view_stream_sha256=dict.fromkeys(data["isolation_excluded_view_counts"], "a" * 64),
                dimensions=["semantic", "example_id", "pair_group_id", "pair_seed"],
                preparation_manifest_sha256=data["dataset_manifest_sha256"],
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
        source_kind="symbolic_canonical_order_preparation",
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
                scope="common_student_order_model_only",
                reload_by_rank=[
                    dict(
                        rank=i,
                        exact_saved_master_reload=dict(
                            all_tensors_exact=True, loaded_before_bf16_copy=True, key_count=311
                        ),
                        inference_envelope_probe=dict(
                            passed=True,
                            input_length=2454,
                            token_id=17,
                            all_logits_finite=True,
                            use_cache=False,
                            native_bfloat16=True,
                            scope="synthetic_training_only_preflight_no_development_content",
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

    from posttrain_circuits.experiments.protocols import student_order_preparation as protocol

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
        for row in checkpoint["reload_by_rank"]:
            row["inference_envelope_probe"] = None
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
    for index in range(7):
        (artifacts / "resume" / f"state-{index}.pt").write_bytes(b"actual-small-state-fixture")
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
    assert publication["passed"] is True and len(records) == 32
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
    plan["control_sha256"]["tools/sdsc_student_order_worker.py"] = "8" * 64
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
