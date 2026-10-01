"""No-network transport fixtures for the fixed two-arm LR diagnostic."""

import base64
import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def c(monkeypatch):
    original_run = subprocess.run

    def guarded_run(argv, *args, **kwargs):
        command = argv[0] if isinstance(argv, list | tuple) else argv.split()[0]
        if Path(command).name in {"squeue", "sacct", "sbatch", "scancel", "scontrol", "srun"}:
            pytest.fail("CPU transport fixtures must never execute real Slurm commands")
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
    spec = importlib.util.spec_from_file_location("sdsc_student_lr", ROOT / "tools/sdsc_student_lr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    rows = [
        dict(
            global_slot=i,
            prompt_id=f"prompt-{i}",
            attempt_id=f"attempt-{i}",
            input_ids=[1] * 835,
            response_ids=[2] * (92 if i < 8 else 91) + [151645],
            response_token_mask=[True] * (93 if i < 8 else 92),
        )
        for i in range(64)
    ]
    capture = c.canonical(dict(schema="quest-sdsc-student-lr-first64-capture-v1", rows=rows))
    monkeypatch.setattr(c, "AUDIT_CAPTURE_SHA", c.sha(capture))
    report = c.canonical(
        dict(
            schema="quest-sdsc-student-lr-data-audit-v1",
            passed=True,
            diagnostic_complete=True,
            parent_job_id=c.PARENT_JOB,
            initial_checkpoint_sha256=c.INITIAL_SHA,
            baseline_instruction_sha256=c.BASELINE_SHA,
            source_commit="6c04f804b302184b8ff95d00fab404e0531ed8d6",
            capture_sha256=c.sha(capture),
            real_source_and_prepare_targets_verified=True,
            sequence_shifted_ce_verified=True,
            all_response_tokens_supervised=True,
            all_responses_end_in_single_eos=True,
            real_student_logits_verified=False,
            training_started=False,
            first64_input_tokens=53440,
            first64_response_tokens=5896,
            ordered_prompt_ids=[x["prompt_id"] for x in rows],
            selected_attempt_ids=[x["attempt_id"] for x in rows],
            window_sha256=c.sha(c.canonical(rows)),
            **dict.fromkeys(c.FLAGS, False),
        )
    )
    monkeypatch.setattr(c, "AUDIT_REPORT_SHA", c.sha(report))
    pins = dict.fromkeys(c.TOOLS, "a" * 64)
    pins["tools/sdsc_student_quality.py"] = c.SHARED_SHA
    intent = c.execution_intent(pins)
    contract = c.helper("sdsc_student_contract")
    runtime = str(c.PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
    cache = str(c.PROJECT / "cache/huggingface")
    value = dict(
        schema="quest-sdsc-student-lr-plan-v2",
        task=c.TASK,
        worker_task=c.WORKER_TASK,
        recovery_from=copy.deepcopy(c.RECOVERY),
        scope="train-only-learning-rate-diagnostic",
        resources=copy.deepcopy(c.RESOURCES),
        science_identity=c.science_identity(),
        protocol_document_sha256=pins[c.PROTOCOL_DOC],
        audit=dict(report=c.audit_payload(report), capture=c.audit_payload(capture)),
        teacher_inputs=c.teacher_inputs(),
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
        manifest_sha256="d" * 64,
        control_sha256=pins,
        release=str(c.CONTROL / "releases/run-1"),
        created_at="2026-09-30T00:00:00Z",
        submission_dir=str(c.CONTROL / "student-lr-submissions" / intent),
        claim=str(c.CONTROL / "student-lr-claims" / (intent + ".json")),
        scientific_claim=str(c.scientific_claim_path()),
        result_dir=str(c.PROJECT / "student-lr" / intent),
        job_name="opd-slr-" + intent,
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
        **dict.fromkeys(c.FLAGS, False),
    )
    c.validate_plan(value)
    return value


@pytest.mark.parametrize(
    "key,value", [("gpus", 1), ("gpus", True), ("mem_gib", 192), ("cpus", 8), ("time", "02:00:00")]
)
def test_resource_envelope_cannot_change(c, plan, key, value):
    plan["resources"][key] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "key,value",
    [
        ("g0_passed", True),
        ("accepted_science", True),
        ("result_dir", "/tmp/foreign"),
        ("python", "/tmp/python"),
        ("scientific_claim", "/tmp/claim"),
    ],
)
def test_plan_identity_and_scope_fail_closed(c, plan, key, value):
    plan[key] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_original_initial_teacher_train_only_are_fixed(c, plan):
    for path, change in [
        ("checkpoints", lambda x: x.append(dict(x[0], label="step33"))),
        ("teacher_inputs", lambda x: x[0].update(sha256="f" * 64)),
        ("dataset_inputs", lambda x: x.append(dict(x[0], path="validation/examples.jsonl"))),
    ]:
        mutated = copy.deepcopy(plan)
        change(mutated[path])
        with pytest.raises(ValueError):
            c.validate_plan(mutated)
    plan["science_identity"]["candidate_instruction"] = "new prompt"
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize("part", ["report", "capture"])
def test_tampered_embedded_audit_rejected(c, plan, part):
    plan["audit"][part]["base64"] = base64.b64encode(b"changed").decode()
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_audit_missing_false_flags_or_pass_is_rejected(c, plan, monkeypatch):
    for key, value in [("passed", False), ("g0_passed", True), ("schema", "other")]:
        report = json.loads(c.decode_audit(plan["audit"])["report"])
        report[key] = value
        raw = c.canonical(report)
        with monkeypatch.context() as m:
            m.setattr(c, "AUDIT_REPORT_SHA", c.sha(raw))
            changed = copy.deepcopy(plan["audit"])
            changed["report"] = c.audit_payload(raw)
            with pytest.raises(ValueError):
                c.decode_audit(changed)


def test_oversized_capture_or_total_plan_rejected(c, plan):
    plan["audit"]["capture"]["size"] = c.MAX_AUDIT_CAPTURE + 1
    with pytest.raises(ValueError):
        c.validate_plan(plan)
    plan["audit"]["capture"]["size"] = 1
    plan["extra"] = "x" * c.MAX_PLAN
    with pytest.raises(ValueError, match="plan exceeds"):
        c.validate_plan(plan)


def test_sbatch_fixed_safe_argv_and_shell_syntax(c, plan):
    args = c.sbatch(plan)
    assert args[:2] == ["sbatch", "--parsable"]
    assert {
        "--gpus=h100:2",
        "--mem=393216M",
        "--cpus-per-task=24",
        "--time=01:00:00",
        "--no-requeue",
        "--export=NONE",
    } <= set(args)
    assert args[-1] == c.sha(c.canonical(plan))
    subprocess.run(["bash", "-n"], input=c.job_script(plan), check=True)


def stubs(c, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    monkeypatch.setattr(c, "verify_recovery", lambda p: {"verified": True})
    monkeypatch.setattr(c, "verify_parent", lambda p, **k: None)
    monkeypatch.setattr(c, "check_gpu_ceiling", lambda: {})


def test_unknown_submission_keeps_scientific_claim_across_control_change(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    calls = []

    def fail(args):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, 45)

    monkeypatch.setattr(c, "run", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert c.document(Path(plan["submission_dir"]) / "unknown.json")["no_retry"] is True
    assert c.document(Path(plan["scientific_claim"])) == c.scientific_claim_record(plan)
    changed = copy.deepcopy(plan)
    changed["control_sha256"][c.TOOLS[0]] = "c" * 64
    changed["intent_id"] = c.execution_intent(changed["control_sha256"])
    for key, parent in [("submission_dir", "student-lr-submissions"), ("claim", "student-lr-claims")]:
        changed[key] = str(c.CONTROL / parent / (changed["intent_id"] + (".json" if key == "claim" else "")))
    changed["result_dir"] = str(c.PROJECT / "student-lr" / changed["intent_id"])
    changed["job_name"] = "opd-slr-" + changed["intent_id"]
    assert changed["scientific_claim"] == plan["scientific_claim"]
    with pytest.raises(ValueError, match="reconcile"):
        c.remote_action(dict(action="submit", plan=changed, authorize=True))
    assert len(calls) == 1
    assert not (c.CONTROL / "student-instruction-claims").exists()


def test_submit_needs_authorization(c, plan, monkeypatch):
    stubs(c, monkeypatch)
    with pytest.raises(ValueError, match="authorization"):
        c.remote_action(dict(action="submit", plan=plan))
    assert not Path(plan["scientific_claim"]).exists()


def test_dry_run_checks_real_inputs_but_creates_no_claim_or_sbatch(c, plan, monkeypatch):
    checked = []
    monkeypatch.setattr(c, "verify_source", lambda p: checked.append("source"))
    monkeypatch.setattr(c, "verify_recovery", lambda p: checked.append("recovery") or {"verified": True})
    monkeypatch.setattr(c, "verify_parent", lambda p, **k: checked.append(("parent", k)))
    monkeypatch.setattr(c, "check_gpu_ceiling", lambda: dict(existing_requested_gpus=0, new_gpus=2))
    original_sha = c.sha
    executable = Path(plan["python"])
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture-python")
    executable.chmod(0o700)
    Path(plan["hf_home"]).mkdir(parents=True)
    monkeypatch.setattr(
        c,
        "sha",
        lambda raw: (
            "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19"
            if raw == b"fixture-python"
            else original_sha(raw)
        ),
    )
    monkeypatch.setattr(c, "run", lambda args: pytest.fail("dry-run may not issue sbatch"))
    result = c.remote_action(dict(action="dry-run", plan=plan))
    assert checked == ["source", "recovery", ("parent", dict(accounting=True))]
    assert result["recovery"] == {"verified": True}
    assert result["cpu_data_audit_verified"] and result["blockers"] == []
    assert not Path(plan["scientific_claim"]).exists()
    assert not Path(plan["submission_dir"]).exists()


def test_missing_ssh_master_does_not_consume_local_submission(c, plan, tmp_path, monkeypatch):
    root = tmp_path / "quest"
    directory = root / ".sdsc/student-lr" / plan["intent_id"]
    directory.mkdir(parents=True)
    plan_path = directory / "plan.json"
    plan_path.write_bytes(c.canonical(plan))
    original_helper, original_read = c.helper, c.read

    def missing():
        raise ValueError("manual authentication required")

    monkeypatch.setattr(c, "ROOT", root)
    monkeypatch.setattr(
        c,
        "helper",
        lambda name: (
            SimpleNamespace(require_master=missing) if name == "sdsc_cli" else original_helper(name)
        ),
    )
    monkeypatch.setattr(
        c,
        "read",
        lambda path, *args: (
            b"control" if Path(path) in {root / name for name in c.TOOLS} else original_read(path, *args)
        ),
    )
    monkeypatch.setattr(c, "validate_plan", lambda value: value)
    original_sha = c.sha
    monkeypatch.setattr(c, "sha", lambda raw: "a" * 64 if raw == b"control" else original_sha(raw))
    # Local pin gate is independent of plan semantics, tested above.
    plan["control_sha256"] = dict.fromkeys(c.TOOLS, "a" * 64)
    plan_path.write_bytes(c.canonical(plan))
    with pytest.raises(ValueError, match="manual authentication"):
        c.main(["submit", "--plan", str(plan_path), "--authorize"])
    assert not (directory / "submission-started.json").exists()


def fake_live(c, plan, job="123"):
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


def accounting(plan, comment="", cpus="24", mem="384G", gpu="2"):
    return dict(
        returncode=0,
        stderr="",
        stdout=(
            f"123|COMPLETED|0:0|{plan['job_name']}|nwu181|nairr-gpu-shared|nairr-gpu-shared-normal|"
            f"{cpus}|{mem}|60|cpu={cpus},mem={mem},node=1,gres/gpu={gpu}|{comment}\n"
            "123.batch|COMPLETED|0:0|batch||||||||\n123.extern|COMPLETED|0:0|extern||||||||\n"
        ),
    )


def test_empty_archived_comment_requires_real_saved_binding(c, plan):
    queue = dict(returncode=0, stdout="", stderr="")
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, accounting(plan))
    assert c.validate_accounting(plan, "123", queue, accounting(plan), fake_live(c, plan))[
        "accounting_complete"
    ]
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, accounting(plan, "foreign"), fake_live(c, plan))


@pytest.mark.parametrize("kwargs", [dict(cpus="8"), dict(mem="192G"), dict(gpu="1")])
def test_terminal_wrong_allocation_is_not_success(c, plan, kwargs):
    with pytest.raises(ValueError):
        c.validate_accounting(
            plan, "123", dict(returncode=0, stdout=""), accounting(plan, plan["intent_id"], **kwargs)
        )


def test_actual_expansion_na_gres_uses_authoritative_tres(c, monkeypatch):
    monkeypatch.setattr(c.pwd, "getpwuid", lambda _: SimpleNamespace(pw_name="zgao12"))
    fields = (
        "JobId=54558773 UserId=zgao12(543540) JobState=RUNNING NumNodes=1 NumTasks=1 "
        "ReqTRES=cpu=24,mem=192G,node=1,billing=4800,gres/gpu=1 "
        "AllocTRES=cpu=24,mem=192G,node=1,billing=4800,gres/gpu=1 TresPerJob=gres:gpu:h100:1"
    )
    monkeypatch.setattr(
        c,
        "run",
        lambda args: dict(
            returncode=0, stderr="", stdout=("54558773|RUNNING|N/A\n" if args[0] == "squeue" else fields)
        ),
    )
    observed = c.check_gpu_ceiling()
    assert observed["existing_requested_gpus"] == 1 and observed["new_gpus"] == 2


@pytest.mark.parametrize("count", [3, 4])
def test_gpu_budget_counts_new_two_gpus(c, monkeypatch, count):
    monkeypatch.setattr(c, "run", lambda _: dict(returncode=0, stdout="123|RUNNING|N/A"))
    monkeypatch.setattr(c, "live_gpu_request", lambda *args: dict(requested_gpus=count))
    with pytest.raises(ValueError, match="four"):
        c.check_gpu_ceiling()


@pytest.mark.parametrize("text", ["N/A", "gres/gpu=0", "cpu=4,node=1,gres/gpu=2,gres/gpu:h100=1"])
def test_missing_or_conflicting_gpu_identity_rejected(c, text):
    with pytest.raises(ValueError):
        c.gpu_tres(text, complete=True)


def publication(c, plan, files, *, passed=False, large=None):
    return dict(
        task=c.TASK,
        job_id="123",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=c.sha(c.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=passed,
        diagnostic_complete=passed,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        large_files=[] if large is None else large,
        files=[dict(path=name, size=len(raw), sha256=c.sha(raw)) for name, raw in files.items()],
        **dict.fromkeys(c.FLAGS, False),
    )


def download(c, plan, files, pub):
    raw = c.canonical(pub)
    content = dict(files, **{"receipt.json": raw})
    return dict(
        receipt=c.make_receipt(plan, "123"),
        status=dict(
            success=False,
            publication_verified=True,
            artifact_hashes_verified=True,
            publication=pub,
            publication_sha256=c.sha(raw),
            evidence_state="verified_publication",
        ),
        files={name: base64.b64encode(value).decode() for name, value in content.items()},
        bytes=sum(map(len, content.values())),
    )


def test_failed_publication_full_raw_verified_and_tampering_rejected(c, plan):
    files = {"lr-records.jsonl": b'{"partial":true}\n', "node-result.json": b"{}"}
    pub = publication(c, plan, files)
    result = download(c, plan, files, pub)
    assert c.validate_download(result, plan)["lr-records.jsonl"] == files["lr-records.jsonl"]
    result["files"]["lr-records.jsonl"] = base64.b64encode(b'{"partial":fals}\n').decode()
    result["bytes"] = sum(len(base64.b64decode(value)) for value in result["files"].values())
    with pytest.raises(ValueError, match="bytes differ"):
        c.validate_download(result, plan)


@pytest.mark.parametrize("mutation", ["foreign", "missing", "duplicate", "overclaim"])
def test_failed_receipt_identity_inventory_and_flags(c, plan, mutation):
    files = {"lr-records.jsonl": b"{}\n"}
    pub = publication(c, plan, files)
    if mutation == "foreign":
        pub["job_id"] = "987"
    elif mutation == "duplicate":
        pub["files"].append(pub["files"][0])
    elif mutation == "missing":
        pub["files"] = []
    else:
        pub["g0_passed"] = True
    with pytest.raises(ValueError):
        c.validate_download(download(c, plan, files, pub), plan)


def test_unpublished_early_failure_fetches_only_bounded_logs(c, plan):
    raw = b"early error"
    result = dict(
        receipt=c.make_receipt(plan, "123"),
        status=dict(
            success=False,
            publication_verified=False,
            artifact_hashes_verified=False,
            evidence_state="unpublished_logs_only",
        ),
        files={"slurm.err": base64.b64encode(raw).decode()},
        bytes=len(raw),
    )
    assert c.validate_download(result, plan) == {"slurm.err": raw}
    result["files"]["lr-probe.json"] = base64.b64encode(b"{}").decode()
    result["bytes"] += 2
    with pytest.raises(ValueError, match="only bounded logs"):
        c.validate_download(result, plan)


def test_large_checkpoints_never_fetch_and_publication_bounds(c, plan):
    large = [dict(path=name, size=7 * 1024**3, sha256="a" * 64) for name in c.LARGE_NAMES]
    assert set(c.large_records(large, complete=True)) == set(c.LARGE_NAMES)
    with pytest.raises(ValueError):
        c.large_records(large[:1], complete=True)
    with pytest.raises(ValueError):
        c.large_records([dict(large[0], path="../model.pt")], complete=False)
    result = download(c, plan, {}, publication(c, plan, {}, large=large))
    result["files"][c.LARGE_NAMES[0]] = "AA=="
    result["bytes"] += 1
    with pytest.raises(ValueError, match="unexpected fetched"):
        c.validate_download(result, plan)


def memory_fixture():
    result = {}
    limit = 384 * 1024**3
    for ordinal, phase in enumerate(
        ("initial", "after_staging", "after_training", "final", "after_publication"), 1
    ):
        ancestor = dict(
            path="/sys/fs/cgroup/memory/slurm/uid_1/job_123",
            limit_bytes=limit,
            current_bytes=ordinal * 1024**3,
            peak_bytes=ordinal * 1024**3,
            memory_failcnt=0,
            memory_events=None,
            memory_events_local=None,
            memory_oom_control=dict(under_oom=0, oom_kill=0, oom_kill_disable=0),
        )
        result[phase] = dict(
            ancestor,
            ancestors=[ancestor],
            expected_limit_bytes=limit,
            passed=True,
            headroom_bytes=limit - ancestor["peak_bytes"],
            minimum_headroom_bytes=82463372084,
            observed_at_unix=ordinal,
        )
    return result


def nll_fixture(ids, tokens):
    count, extra = divmod(tokens, len(ids))
    rows = [
        dict(
            prompt_id=identity,
            response_tokens=count + int(i < extra),
            correct_tokens=(count + int(i < extra)) // 2,
            hf_sequence_nll=1.0,
            production_sequence_nll=1.0,
            labels_ce_abs_error=0.0,
        )
        for i, identity in enumerate(ids)
    ]
    return dict(
        examples=len(ids),
        response_tokens_including_eos=tokens,
        per_example=rows,
        sequence_mean_nll=1.0,
        hf_sequence_mean_nll=1.0,
        token_mean_nll=1.0,
        token_accuracy=sum(x["correct_tokens"] for x in rows) / tokens,
        max_labels_ce_abs_error=0.0,
    )


def successful_worker(c, plan):
    audit = json.loads(c.decode_audit(plan["audit"])["report"])
    ids = audit["ordered_prompt_ids"]
    cohort256 = ids + [f"future-{i}" for i in range(64, 256)]
    execution = dict(
        protocol="allocation_neutral_exact_global_batch_v1",
        global_batch_size=64,
        max_microbatch_size=4,
        max_model_input_length=1536,
        world_size=2,
        microsteps_per_optimizer_update=8,
        rank_local_sequence_counts=[32, 32],
        microbatch_sizes_by_rank=[[4] * 8, [4] * 8],
        requested_fsdp_sharding_strategy="FULL_SHARD",
        effective_fsdp_sharding_strategy="FULL_SHARD",
        fsdp_wrapper_count=29,
    )
    arms, checkpoints = [], []
    for index, (arm, lr) in enumerate(c.ARMS):
        updates = []
        for step in range(1, 5):
            ranks = [
                dict(
                    rank=rank,
                    step=step,
                    all_parameter_steps_equal=True,
                    parameter_tensors=29,
                    scheduler_last_epoch=step,
                    microstep_sync=[False] * 7 + [True],
                    actual_optimizer_calls=[step - 1] * 7 + [step],
                )
                for rank in range(2)
            ]
            if step == 1:
                for row in ranks:
                    row["first_step_prediction"] = dict(
                        elements=860287488,
                        gradient_squared=1.0,
                        update_squared=1.0,
                        roundoff_violations=0,
                        max_roundoff_ratio=0.0,
                        gradient_dot_update=-1.0,
                        gradient_dot_roundoff_bound=1e-6,
                    )
            updates.append(
                dict(
                    step=step,
                    teacher_target_before=nll_fixture(ids, 5896),
                    teacher_target_after=nll_fixture(ids, 5896),
                    optimizer_by_rank=ranks,
                    cursors_by_rank=[
                        dict(rank=rank, cursor={key: 1 for key in cohort256[rank : step * 64 : 2]})
                        for rank in range(2)
                    ],
                )
            )
        arms.append(
            dict(
                arm=arm,
                learning_rate=lr,
                updates=updates,
                initial_loading=dict(
                    checkpoint_sha256=c.INITIAL_SHA,
                    all_tensors_exact=True,
                    key_count=311,
                    all_parameters_trainable=True,
                ),
                initial_state_sha256="a" * 64,
                initial_rng_by_rank=["b" * 64, "c" * 64],
                prepared_execution=execution,
                teacher_target_initial=nll_fixture(ids, 5896),
                canonical_initial=nll_fixture(ids[:32], 2936),
                canonical_final=nll_fixture(ids[:32], 2936),
                generation=[
                    dict(
                        step=step,
                        record_count=32,
                        record_start=index * 64 + j * 32,
                        metrics=dict(count=32),
                        strict_fp32_loading=dict(all_tensors_exact=True, loaded_before_bf16_copy=True),
                    )
                    for j, step in enumerate((0, 4))
                ],
                root_vs_export_logits=[
                    dict(
                        ordinal=i,
                        elements=100,
                        argmax_mismatches=0,
                        compared_positions=10,
                        max_abs_error=0.0,
                        rms_error=0.0,
                        formal_parity_gate_applied=False,
                    )
                    for i in range(4)
                ],
            )
        )
        checkpoints.append(
            dict(
                path=c.LARGE_NAMES[index],
                size=7 * 1024**3,
                sha256="d" * 64,
                arm=arm,
                step=4,
                scope="diagnostic_model_only",
            )
        )
    files = {"lr-batches.jsonl": b"{}\n", "lr-records.jsonl": b"{}\n", "lr-prompts.jsonl": b"{}\n"}
    records = {name: dict(path=name, size=len(raw), sha256=c.sha(raw)) for name, raw in files.items()}
    report = dict(
        schema="quest-sdsc-student-lr-probe-v1",
        task=c.WORKER_TASK,
        passed=True,
        diagnostic_complete=True,
        job_id="123",
        parent_job_id=c.PARENT_JOB,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        initial_checkpoint_sha256=c.INITIAL_SHA,
        original_instruction_sha256=c.BASELINE_SHA,
        data_audit_sha256=c.AUDIT_REPORT_SHA,
        data_capture_sha256=c.AUDIT_CAPTURE_SHA,
        audited_window_sha256=audit["window_sha256"],
        raw_record_count=128,
        arms=arms,
        generation=c.science_identity()["generation"],
        raw_artifacts=list(records.values()),
        checkpoints=checkpoints,
        **dict.fromkeys(c.FLAGS, False),
    )
    return report, records, files


def test_success_requires_real_two_arm_evidence_and_memory_after_publication(c, plan):
    report, records, files = successful_worker(c, plan)
    c.validate_worker_report(report, plan, "123", records)
    memory = memory_fixture()
    c.validate_memory(memory, "123")
    large = c.worker_checkpoints(report["checkpoints"])
    node = dict(
        task=c.TASK,
        job_id="123",
        parent_job_id=c.PARENT_JOB,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        world_size=2,
        cpus=24,
        memory_mib=393216,
        account="nwu181",
        partition="nairr-gpu-shared",
        diagnostic_complete=True,
        exit_code=0,
        large_artifacts=large,
        **dict.fromkeys(c.FLAGS, False),
    )
    files.update(
        {
            "lr-probe.json": c.canonical(report),
            "memory.json": c.canonical(memory),
            "node-result.json": c.canonical(node),
        }
    )
    pub = publication(c, plan, files, passed=True, large=c.worker_checkpoints(report["checkpoints"]))
    fetched = download(c, plan, files, pub)
    fetched["status"]["success"] = True
    assert set(c.validate_download(fetched, plan)) == set(c.NAMES) | {"receipt.json"}


@pytest.mark.parametrize(
    "case",
    [
        "lr",
        "steps",
        "rank",
        "cadence",
        "gradient",
        "roundoff",
        "rng",
        "sequence_mean",
        "model_scope",
        "generation",
        "nonfinite",
    ],
)
def test_incomplete_or_inconsistent_worker_cannot_succeed(c, plan, case):
    report, records, _ = successful_worker(c, plan)
    arm = report["arms"][0]
    if case == "lr":
        arm["learning_rate"] = 0.005
    elif case == "steps":
        arm["updates"].pop()
    elif case == "rank":
        arm["updates"][0]["optimizer_by_rank"].pop()
    elif case == "cadence":
        arm["updates"][0]["optimizer_by_rank"][0]["actual_optimizer_calls"][0] = 1
    elif case == "gradient":
        arm["updates"][0]["optimizer_by_rank"][0]["first_step_prediction"]["gradient_squared"] = 0
    elif case == "roundoff":
        arm["updates"][0]["optimizer_by_rank"][0]["first_step_prediction"]["roundoff_violations"] = 1
    elif case == "rng":
        arm["initial_rng_by_rank"][0] = "f" * 64
    elif case == "sequence_mean":
        arm["teacher_target_initial"]["sequence_mean_nll"] = 2.0
    elif case == "model_scope":
        report["checkpoints"][0]["scope"] = "formal_initial"
    elif case == "generation":
        arm["generation"][0]["record_count"] = 31
    else:
        arm["root_vs_export_logits"][0]["max_abs_error"] = float("nan")
    with pytest.raises(ValueError):
        c.validate_worker_report(report, plan, "123", records)


@pytest.mark.parametrize(
    "case", ["missing_publication", "foreign_job", "wrong_limit", "peak", "headroom", "time"]
)
def test_memory_requires_original_hierarchy_and_final_copy_peak(c, case):
    memory = memory_fixture()
    value = memory["after_publication"]
    if case == "missing_publication":
        del memory["after_publication"]
    elif case == "foreign_job":
        value["ancestors"][0]["path"] += "4"
    elif case == "wrong_limit":
        value["ancestors"][0]["limit_bytes"] //= 2
    elif case == "peak":
        value["ancestors"][0]["peak_bytes"] += 1
    elif case == "headroom":
        value["headroom_bytes"] += 1
    else:
        value["observed_at_unix"] = 0
    with pytest.raises(ValueError):
        c.validate_memory(memory, "123")


@pytest.mark.parametrize(
    "field,key",
    [
        ("memory_failcnt", None),
        ("memory_events", "oom"),
        ("memory_events", "oom_kill"),
        ("memory_events_local", "oom"),
        ("memory_events_local", "oom_group_kill"),
        ("memory_oom_control", "under_oom"),
        ("memory_oom_control", "oom_kill"),
    ],
)
@pytest.mark.parametrize("count", [1, -1, True, 0.0])
def test_low_peak_cannot_hide_job_oom_or_invalid_counter(c, field, key, count):
    memory = memory_fixture()
    # An intermediate sample must not disappear from terminal validation.
    memory["running-1"] = copy.deepcopy(memory["after_staging"])
    row = memory["running-1"]["ancestors"][0]
    row[field] = count if key is None else {key: count}
    with pytest.raises(ValueError, match="memory.*counter"):
        c.validate_memory(memory, "123")


def test_memory_events_ignore_shared_history_and_distinct_job_prefix(c):
    memory = memory_fixture()
    for evidence in memory.values():
        evidence["ancestors"][0]["memory_oom_control"]["oom_kill_disable"] = 1
        evidence["ancestors"].extend(
            dict(path=path, limit_bytes=None, memory_failcnt=12, memory_events={"oom": 7})
            for path in (
                "/sys/fs/cgroup/memory/slurm/uid_1",
                "/sys/fs/cgroup/memory/slurm/uid_1/job_1234",
            )
        )
    c.validate_memory(memory, "123")


def test_memory_events_reject_malformed_mapping_and_accept_v2_zero(c):
    evidence = memory_fixture()["initial"]
    row = evidence["ancestors"][0]
    row["memory_events"] = {"oom": 0, "oom_kill": 0, "oom_group_kill": 0, "high": 15}
    row["memory_events_local"] = {"oom": 0, "oom_kill": 0}
    c.validate_job_memory_events(evidence, "123")
    row["memory_events_local"] = []
    with pytest.raises(ValueError, match="counter mapping"):
        c.validate_job_memory_events(evidence, "123")


@pytest.fixture
def recovery_case(c, tmp_path, monkeypatch):
    # Exact 642-byte startup report from the sealed v1 job, independently pinned.
    failure = dict(
        schema="quest-sdsc-student-lr-probe-v1",
        passed=False,
        diagnostic_complete=False,
        failed_rank="0",
        raw_artifacts=[],
        **dict.fromkeys(c.FLAGS, False),
        error="ValueError: rank-local phase failed: "
        + json.dumps(
            [
                dict(
                    rank=rank,
                    type="AdaptedStudentProtocolError",
                    message="student protocol needs real .git or .opd-git metadata",
                )
                for rank in range(2)
            ]
        ),
    )
    failure_raw = c.canonical(failure) + b"\n"
    assert len(failure_raw) == 642 and c.sha(failure_raw) == c.RECOVERY["failure_sha256"]
    node = dict(
        exit_code=1,
        diagnostic_complete=False,
        large_artifacts=[],
        job_id="54560005",
        plan_sha256=c.RECOVERY["plan_sha256"],
        **dict.fromkeys(c.FLAGS, False),
    )
    raw = {"lr-probe.json": failure_raw, "node-result.json": c.canonical(node), "memory.json": b"{}"}
    old_plan = dict(
        result_dir=str(tmp_path / "old-result"),
        submission_dir=str(tmp_path / "old-submission"),
        job_name="opd-slr-fixture",
        intent_id=c.RECOVERY["intent_id"],
    )
    Path(old_plan["submission_dir"]).mkdir()
    (Path(old_plan["submission_dir"]) / "live-binding.json").write_bytes(
        c.canonical(fake_live(c, old_plan, "54560005"))
    )
    Path(old_plan["result_dir"]).mkdir()
    published = dict(
        passed=False,
        diagnostic_complete=False,
        large_files=[],
        files=[dict(path=n, size=len(v), sha256=c.sha(v)) for n, v in raw.items()],
    )
    recovery = dict(c.RECOVERY, publication_sha256=c.sha(c.canonical(published)))
    monkeypatch.setattr(c, "RECOVERY", recovery)
    status = dict(
        state="FAILED",
        success=False,
        diagnostic_complete=False,
        accounting_complete=False,
        queue=dict(returncode=0, stdout=""),
        accounting=dict(
            returncode=0,
            stdout=(
                "54560005|FAILED|1:0|opd-slr-fixture|nwu181|nairr-gpu-shared|nairr-gpu-shared-normal|24|384G|130|cpu=24,mem=384G,gres/gpu=2|\n"
                "54560005.batch|FAILED|1:0|batch|nwu181|||24||130|cpu=24,gres/gpu=2,mem=384G|\n"
                "54560005.extern|COMPLETED|0:0|extern|nwu181|||24||130|cpu=24,gres/gpu=2,mem=384G|\n"
            ),
        ),
        publication_verified=True,
        artifact_hashes_verified=True,
        publication=published,
        publication_sha256=recovery["publication_sha256"],
    )
    # Transport/identity is the already-reviewed v1 controller's responsibility.
    previous = SimpleNamespace(
        publication_records=lambda p, plan, job: {r["path"]: r for r in p["files"]},
        validate_accounting=c.validate_accounting,
    )
    for name, data in raw.items():
        (Path(old_plan["result_dir"]) / name).write_bytes(data)
    (Path(old_plan["result_dir"]) / "receipt.json").write_bytes(c.canonical(published))
    return previous, old_plan, status, raw


def test_exact_startup_failure_is_admissible_and_no_checkpoint_allowed(c, recovery_case, monkeypatch):
    previous, old_plan, status, raw = recovery_case
    c.validate_recovery_evidence(previous, old_plan, status, raw)
    monkeypatch.setattr(c, "recovery_context", lambda: (previous, old_plan, {"job_id": "54560005"}))
    monkeypatch.setattr(
        c,
        "run",
        lambda argv: (
            dict(returncode=0, stdout="", stderr="") if argv[0] == "squeue" else status["accounting"]
        ),
    )
    result = c.verify_recovery(dict(recovery_from=c.RECOVERY))
    assert result["verified"] and result["no_training_evidence"] and result["job_id"] == "54560005"
    (Path(old_plan["result_dir"]) / "checkpoints").mkdir()
    with pytest.raises(ValueError, match="checkpoint"):
        c.verify_recovery(dict(recovery_from=c.RECOVERY))


@pytest.mark.parametrize(
    "defect",
    [
        "queue",
        "unknown",
        "success",
        "missing_batch",
        "different_exit",
        "foreign_job",
        "allocation",
        "unpublished",
        "receipt",
        "tampered_raw",
        "training_raw",
        "checkpoint",
        "arms",
        "updates",
        "different_error",
        "node_success",
    ],
)
def test_recovery_rejects_every_non_exact_failure(c, recovery_case, monkeypatch, defect):
    previous, old_plan, status, raw = recovery_case
    if defect == "queue":
        status["queue"]["stdout"] = "54560005|RUNNING\n"
    elif defect == "unknown":
        status["state"] = "UNKNOWN"
    elif defect == "success":
        status["success"] = True
    elif defect == "missing_batch":
        status["accounting"]["stdout"] = "\n".join(
            x for x in status["accounting"]["stdout"].splitlines() if ".batch|" not in x
        )
    elif defect == "different_exit":
        status["accounting"]["stdout"] = status["accounting"]["stdout"].replace("FAILED|1:0", "FAILED|9:0")
    elif defect == "foreign_job":
        status["accounting"]["stdout"] = status["accounting"]["stdout"].replace("54560005", "54560006")
    elif defect == "allocation":
        status["accounting"]["stdout"] = status["accounting"]["stdout"].replace("gres/gpu=2", "gres/gpu=1")
    elif defect == "unpublished":
        status["publication_verified"] = False
    elif defect == "receipt":
        status["publication_sha256"] = "a" * 64
    elif defect == "tampered_raw":
        raw["lr-probe.json"] += b" "
    elif defect == "training_raw":
        raw["lr-records.jsonl"] = b"{}\n"
        status["publication"]["files"].append(dict(path="lr-records.jsonl", size=3, sha256=c.sha(b"{}\n")))
    elif defect == "checkpoint":
        status["publication"]["large_files"] = [dict(path=c.LARGE_NAMES[0], size=1, sha256="a" * 64)]
    else:
        name = "node-result.json" if defect == "node_success" else "lr-probe.json"
        value = json.loads(raw[name])
        if defect == "node_success":
            value["diagnostic_complete"] = True
        elif defect == "different_error":
            value["error"] = "different failure"
        else:
            value[defect] = []
        raw[name] = c.canonical(value)
        for row in status["publication"]["files"]:
            if row["path"] == name:
                row.update(size=len(raw[name]), sha256=c.sha(raw[name]))
        # Even resealed partial/foreign reports must fail semantic verification.
        if name == "lr-probe.json":
            monkeypatch.setitem(c.RECOVERY, "failure_sha256", c.sha(raw[name]))
    with pytest.raises(ValueError):
        c.validate_recovery_evidence(previous, old_plan, status, raw)


@pytest.mark.parametrize("action", ["dry-run", "submit"])
def test_failed_recovery_cannot_write_claim_or_submit(c, plan, monkeypatch, action):
    stubs(c, monkeypatch)

    def reject(value):
        raise ValueError("old job not a verified startup failure")

    monkeypatch.setattr(c, "verify_recovery", reject)
    monkeypatch.setattr(c, "run", lambda args: pytest.fail("recovery failure must never submit"))
    with pytest.raises(ValueError, match="startup failure"):
        c.remote_action(dict(action=action, plan=plan, authorize=True))
    assert not Path(plan["scientific_claim"]).exists() and not Path(plan["claim"]).exists()
    assert not Path(plan["submission_dir"]).exists()


@pytest.mark.parametrize(
    "field,value", [("job_id", "1"), ("plan_sha256", "a" * 64), ("publication_sha256", "b" * 64)]
)
def test_v2_cannot_select_another_recovery_predecessor(c, plan, field, value):
    plan["recovery_from"][field] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.fixture
def queue_owner(c, monkeypatch):
    monkeypatch.setattr(c.os, "getuid", lambda: 12345)
    monkeypatch.setattr(c.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="zgao12"))
    return "zgao12"


def test_archived_job_uses_genuine_full_owner_query_not_invalid_id(c, queue_owner, monkeypatch):
    calls = []
    raw = dict(returncode=0, stdout="777_0|RUNNING|zgao12\n888|PENDING|zgao12\n", stderr="")

    def runner(argv):
        calls.append(argv)
        if "--jobs=54560005" in argv:
            return dict(returncode=1, stdout="", stderr="slurm_load_jobs error: Invalid job id specified\n")
        return raw

    monkeypatch.setattr(c, "run", runner)
    result = c.job_queue("54560005")
    assert result["returncode"] == 0 and result["stdout"] == ""
    assert result["query_raw"] == raw and result["query_argv"] == calls[0]
    assert result["query_uid"] == 12345 and result["query_user"] == queue_owner
    assert result["query_scope"] == "all_jobs_for_effective_uid"
    assert {"--array", "--local", "--states=all", "--user=12345", "--format=%i|%T|%u"} <= set(calls[0])
    assert len(calls) == 1 and not any(a.startswith("--jobs=") for a in calls[0])


def test_owner_queue_filters_exact_scalar_target(c, queue_owner, monkeypatch):
    raw = dict(returncode=0, stderr="", stdout="545600050|PENDING|zgao12\n54560005|COMPLETING|zgao12\n")
    monkeypatch.setattr(c, "run", lambda argv: raw)
    result = c.job_queue("54560005")
    assert result["stdout"] == "54560005|COMPLETING" and result["query_raw"] == raw


@pytest.mark.parametrize(
    "defect",
    [
        "error",
        "bool_rc",
        "oversized_stdout",
        "oversized_stderr",
        "nontext",
        "blank",
        "truncated",
        "duplicate",
        "foreign",
        "unknown_state",
        "array_range",
        "target_array",
        "extra_column",
    ],
)
def test_owner_queue_never_turns_incomplete_or_foreign_query_into_absence(
    c, queue_owner, monkeypatch, defect
):
    raw = dict(returncode=0, stdout="777|RUNNING|zgao12\n", stderr="")
    changes = {
        "error": dict(returncode=1, stdout="", stderr="slurm_load_jobs error: Invalid job id specified\n"),
        "bool_rc": dict(returncode=False),
        "oversized_stdout": dict(stdout="x" * (c.CAP + 1)),
        "oversized_stderr": dict(stderr="x" * 65537),
        "nontext": dict(stdout=None),
        "blank": dict(stdout="\n"),
        "truncated": dict(stdout="777|RUNNING|"),
        "duplicate": dict(stdout="777|RUNNING|zgao12\n777|RUNNING|zgao12\n"),
        "foreign": dict(stdout="777|RUNNING|someone_else\n"),
        "unknown_state": dict(stdout="777|UNRECOGNIZED|zgao12\n"),
        "array_range": dict(stdout="777_[1-7]|PENDING|zgao12\n"),
        "target_array": dict(stdout="54560005_0|PENDING|zgao12\n"),
        "extra_column": dict(stdout="777|RUNNING|zgao12|extra\n"),
    }
    raw.update(changes[defect])
    monkeypatch.setattr(c, "run", lambda argv: raw)
    with pytest.raises(ValueError):
        c.job_queue("54560005")


def test_slurm_query_and_submission_environment_cannot_inject_selectors(c, monkeypatch):
    for name in ("SQUEUE_PARTITION", "SQUEUE_STATES", "SQUEUE_USERS", "SACCT_FORMAT", "SBATCH_MEM"):
        monkeypatch.setenv(name, "injected")
    monkeypatch.setenv("KEEP_NORMAL_ENV", "preserved")
    captured = {}

    def runner(argv, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(c.subprocess, "run", runner)
    assert c.run(["squeue"])["returncode"] == 0
    assert not any(k.startswith(("SQUEUE_", "SACCT_", "SBATCH_")) for k in captured["env"])
    assert captured["env"]["KEEP_NORMAL_ENV"] == "preserved"


@pytest.mark.parametrize("action", ["dry-run", "submit"])
@pytest.mark.parametrize("defect", ["active", "query_failure", "malformed"])
def test_real_recovery_queue_adapter_stops_before_any_new_claim(
    c, plan, recovery_case, queue_owner, monkeypatch, action, defect
):
    previous, old_plan, status, raw = recovery_case
    real_verify = c.verify_recovery
    stubs(c, monkeypatch)
    monkeypatch.setattr(c, "verify_recovery", real_verify)
    monkeypatch.setattr(c, "recovery_context", lambda: (previous, old_plan, {"job_id": "54560005"}))
    queue = {
        "active": dict(returncode=0, stdout="54560005|RUNNING|zgao12\n", stderr=""),
        "query_failure": dict(returncode=1, stdout="", stderr="unavailable"),
        "malformed": dict(returncode=0, stdout="truncated", stderr=""),
    }[defect]

    def runner(argv):
        if argv[0] == "squeue":
            return queue
        if argv[0] == "sacct":
            return status["accounting"]
        pytest.fail("queue rejection must precede sbatch")

    monkeypatch.setattr(c, "run", runner)
    with pytest.raises(ValueError):
        c.remote_action(dict(action=action, plan=plan, authorize=True))
    assert not Path(plan["scientific_claim"]).exists() and not Path(plan["claim"]).exists()
    assert not Path(plan["submission_dir"]).exists()


def test_v2_failed_terminal_inspection_uses_owner_query(c, plan, queue_owner, monkeypatch):
    account = accounting(plan, comment=plan["intent_id"])
    account["stdout"] = account["stdout"].replace("COMPLETED|0:0", "FAILED|1:0")
    calls = []

    def runner(argv):
        calls.append(argv)
        if argv[0] == "squeue":
            assert not any(a.startswith("--jobs=") for a in argv)
            return dict(returncode=0, stdout="777_1|RUNNING|zgao12\n", stderr="")
        assert argv[0] == "sacct"
        return account

    monkeypatch.setattr(c, "run", runner)
    result = c.inspect_job(plan, {"job_id": "123"})
    assert result["state"] == "FAILED" and not result["success"]
    assert result["evidence_state"] == "unpublished_logs_only"
    assert result["queue"]["query_raw"]["stdout"] == "777_1|RUNNING|zgao12\n"
    assert len(calls) == 2
