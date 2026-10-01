"""Local-only contracts for the original native-BF16 initial validation check."""

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
    original = subprocess.run

    def guard(argv, *args, **kwargs):
        command = argv[0] if isinstance(argv, list | tuple) else argv.split()[0]
        if Path(command).name in {"squeue", "sacct", "sbatch", "scancel", "scontrol", "srun"}:
            pytest.fail("CPU tests must never execute actual Slurm commands")
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guard)
    spec = importlib.util.spec_from_file_location("native_initial", ROOT / "tools/sdsc_student_initial.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def plan(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(c, "PROJECT", tmp_path / "project")
    pins = dict.fromkeys(c.TOOLS, "a" * 64)
    pins["tools/sdsc_student_quality.py"] = c.SHARED_SHA
    pins["tools/sdsc_student_instruction_job.py"] = c.MEMORY_HELPER_SHA
    intent = c.execution_intent(pins)
    contract = c.helper("sdsc_student_contract")
    python = str(c.PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
    cache = str(c.PROJECT / "cache/huggingface")
    value = dict(
        schema="quest-sdsc-student-initial-plan-v1",
        task=c.TASK,
        scope="original-initial-native-validation128",
        resources=copy.deepcopy(c.RESOURCES),
        science_identity=c.science_identity(),
        baseline_sha256=c.BASELINE_SHA,
        protocol_document_sha256=pins[c.PROTOCOL_DOC],
        control_sha256=pins,
        intent_id=intent,
        run_id="run-1",
        code_sha256="b" * 64,
        manifest_sha256="d" * 64,
        created_at="2026-10-01T00:00:00Z",
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
        release=str(c.CONTROL / "releases/run-1"),
        submission_dir=str(c.CONTROL / "student-initial-submissions" / intent),
        claim=str(c.CONTROL / "student-initial-claims" / (intent + ".json")),
        scientific_claim=str(c.scientific_claim_path()),
        result_dir=str(c.PROJECT / "student-initial" / intent),
        job_name="opd-sin-" + intent,
        python=python,
        hf_home=cache,
        parent=dict(
            report_sha256=c.PARENT_REPORT_SHA,
            publication_sha256=c.PARENT_PUBLICATION_SHA,
            receipt=dict(
                job_id=c.PARENT_JOB,
                intent_id=c.PARENT_INTENT,
                submission_dir=str(c.CONTROL / "submissions" / c.PARENT_INTENT),
                python=python,
                hf_home=cache,
                science_git_head=c.SCIENCE_HEAD,
                bundle_sha256=c.PARENT_BUNDLE_SHA,
                student_protocol_sha256=c.SCIENCE_PROTOCOL_SHA,
                student_protocol_artifact_sha256=c.SCIENCE_ARTIFACT_SHA,
            ),
        ),
        **dict.fromkeys(c.FLAGS, False),
    )
    return c.validate_plan(value)


def science_map(c):
    names = json.loads((ROOT / "prereg/amendments/qwen3_adapted_student_calibration_v6.json").read_text())[
        "science_files"
    ]
    result = {name: c.sha((ROOT / name).read_bytes()) for name in names}
    assert c.sha(c.canonical(result)) == c.NAMED_SCIENCE_MAP_SHA
    return result


def memory_fixture(job="123"):
    limit = 64 * 1024**3
    values = {}
    for ordinal, phase in enumerate(
        ("initial", "after_staging", "after_inference", "final", "after_publication")
    ):
        peak = (10 + ordinal) * 1024**3
        path = f"/sys/fs/cgroup/memory/slurm/uid_1/job_{job}"
        row = dict(
            path=path,
            limit_bytes=limit,
            current_bytes=peak,
            peak_bytes=peak,
            memory_failcnt=0,
            memory_events=None,
            memory_events_local=None,
            memory_oom_control=dict(under_oom=0, oom_kill=0, oom_kill_disable=0),
        )
        values[phase] = dict(
            expected_limit_bytes=limit,
            limit_bytes=limit,
            current_bytes=peak,
            peak_bytes=peak,
            path=path,
            headroom_bytes=limit - peak,
            minimum_headroom_bytes=32 * 1024**3,
            passed=True,
            observed_at_unix=100 + ordinal,
            ancestors=[row],
            cgroup_version=1,
        )
    return values


def fixture_reports(c, plan, correct=13, proofs=0):
    ids = [f"validation-{i}" for i in range(128)]
    prompts = []
    responses = []
    for i, name in enumerate(ids):
        prompt = dict(
            ordinal=i,
            cohort="validation_exposed",
            example=dict(example_id=name),
            raw_prompt="raw",
            prompt_text="prompt",
            raw_prompt_sha256=c.sha(b"raw"),
            prompt_text_sha256=c.sha(b"prompt"),
            prompt_ids=[1, 2],
            prompt_token_sha256=c.sha(c.canonical([1, 2])),
        )
        prompts.append(prompt)
        responses.append(
            dict(
                ordinal=i,
                example_id=name,
                cohort="validation_exposed",
                response_ids=[3, 151645],
                response_tokens=2,
                max_new_tokens=256,
                response_text="synthetic",
                stop_reason="eos",
                parsed_trace={},
                prompt_text_sha256=prompt["prompt_text_sha256"],
                prompt_token_sha256=prompt["prompt_token_sha256"],
                verification=dict(
                    parse_valid=i < correct,
                    proof_valid=i < proofs,
                    answer_correct=i < correct,
                    reward=float(i < proofs),
                ),
            )
        )
    raw = {
        name: b"".join(c.canonical(row) + b"\n" for row in rows)
        for name, rows in [("initial-prompts.jsonl", prompts), ("initial-records.jsonl", responses)]
    }
    inventory = {name: dict(path=name, size=len(data), sha256=c.sha(data)) for name, data in raw.items()}
    science = science_map(c)
    report = dict(
        schema="quest-sdsc-student-initial-probe-v1",
        task=c.TASK,
        passed=True,
        diagnostic_complete=True,
        job_id="123",
        parent_job_id=c.PARENT_JOB,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        initial_checkpoint_sha256=c.INITIAL_SHA,
        resolved_config_sha256=c.CONFIG_SHA,
        dataset_manifest_sha256=c.FAMILY_SHA,
        dataset_family_semantic_sha256="e" * 64,
        validation_file_sha256=c.VALIDATION_SHA,
        original_instruction_sha256=c.BASELINE_SHA,
        tokenizer_fingerprint=c.TOKENIZER_SHA,
        chat_template_sha256=c.TEMPLATE_SHA,
        model_revision=c.REVISION,
        forward_parameter_dtype="torch.bfloat16",
        generation=copy.deepcopy(c.GENERATION),
        new_test_holdout=False,
        scientific_binding=dict(
            head=c.SCIENCE_HEAD,
            source_pins=c.SOURCE_PINS,
            deployment_integrity_only=True,
            deployment_python_files_verified=228,
            protocol_artifact_sha256=c.SCIENCE_ARTIFACT_SHA,
            protocol_sha256=c.SCIENCE_PROTOCOL_SHA,
            acceptance_commit=c.SCIENCE_HEAD,
            implementation_commit="9a196677a269c2e5f1c925da4e43b6145c1cdd7a",
            science_file_sha256=science,
        ),
        prompt_count=128,
        evaluated_count=128,
        raw_record_count=128,
        ordered_example_ids=ids,
        maximum_prompt_tokens=2,
        answer_correct_count=correct,
        threshold=0.10,
        required_correct=13,
        initial_base_gate_satisfied=correct >= 13,
        metrics=dict(
            format_validity=correct / 128, exact_proof_accuracy=proofs / 128, answer_accuracy=correct / 128
        ),
        gpu=dict(
            name="NVIDIA H100", logical_index=0, torch="2.8.0+cu128", cpu_threads=8, cuda_visible_devices="0"
        ),
        raw_artifacts=list(inventory.values()),
        **dict.fromkeys(c.FLAGS, False),
    )
    node = dict(
        task=c.TASK,
        job_id="123",
        parent_job_id=c.PARENT_JOB,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=c.sha(c.canonical(plan)),
        cpus=8,
        memory_mib=65536,
        world_size=1,
        account="nwu181",
        partition="nairr-gpu-shared",
        diagnostic_complete=True,
        exit_code=0,
        cuda_visible_devices="0",
        science_restore=dict(
            head=c.SCIENCE_HEAD,
            bundle_sha256=c.PARENT_BUNDLE_SHA,
            protocol_sha256=c.SCIENCE_PROTOCOL_SHA,
            original_implementation_only=True,
            science_file_sha256=science,
        ),
        **dict.fromkeys(c.FLAGS, False),
    )
    raw.update(
        {
            "initial-probe.json": c.canonical(report),
            "node-result.json": c.canonical(node),
            "memory.json": c.canonical(memory_fixture()),
        }
    )
    return report, node, raw


def packaged(c, plan, failed=False):
    report, node, raw = fixture_reports(c, plan)
    if failed:
        raw = {
            "initial-probe.json": c.canonical(
                dict(error="failed", passed=False, **dict.fromkeys(c.FLAGS, False))
            )
        }
    pub = dict(
        task=c.TASK,
        job_id="123",
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=c.sha(c.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=not failed,
        diagnostic_complete=not failed,
        persistent_read_back_verified=True,
        files=[dict(path=n, size=len(v), sha256=c.sha(v)) for n, v in raw.items()],
        **dict.fromkeys(c.FLAGS, False),
    )
    raw["receipt.json"] = c.canonical(pub)
    status = dict(
        publication_verified=True,
        artifact_hashes_verified=True,
        publication=pub,
        publication_sha256=c.sha(raw["receipt.json"]),
        success=not failed,
        evidence_state="verified_publication",
        **dict.fromkeys(c.FLAGS, False),
    )
    return dict(
        receipt=c.make_receipt(plan, "123"),
        status=status,
        files={n: base64.b64encode(v).decode() for n, v in raw.items()},
        bytes=sum(map(len, raw.values())),
    )


def stub_remote(c, monkeypatch):
    monkeypatch.setattr(c, "verify_source", lambda p: None)
    monkeypatch.setattr(c, "verify_parent", lambda p, **k: None)
    monkeypatch.setattr(c, "check_gpu_ceiling", lambda: dict(existing_requested_gpus=2, new_gpus=1, limit=4))


@pytest.mark.parametrize(
    "key,value", [("gpus", 2), ("gpus", True), ("mem_gib", 192), ("time", "01:00:00"), ("cpus", 24)]
)
def test_fixed_resources_cannot_be_changed(c, plan, key, value):
    plan["resources"][key] = value
    with pytest.raises(ValueError):
        c.validate_plan(plan)


@pytest.mark.parametrize(
    "defect",
    [
        "checkpoint",
        "config",
        "missing_split",
        "other_dataset",
        "protocol",
        "bundle",
        "parent_publication",
        "autocast",
        "threshold",
        "helper",
        "overclaim",
    ],
)
def test_original_scientific_identity_is_fixed(c, plan, defect):
    if defect == "checkpoint":
        plan["checkpoints"][0]["sha256"] = "f" * 64
    elif defect == "config":
        plan["resolved_config"]["sha256"] = "f" * 64
    elif defect == "missing_split":
        plan["dataset_inputs"].pop()
    elif defect == "other_dataset":
        plan["dataset_inputs"][0]["source"] = "/tmp/dataset"
    elif defect == "protocol":
        plan["parent"]["receipt"]["student_protocol_sha256"] = "f" * 64
    elif defect == "bundle":
        plan["parent"]["receipt"]["bundle_sha256"] = "f" * 64
    elif defect == "parent_publication":
        plan["parent"]["publication_sha256"] = "f" * 64
    elif defect == "autocast":
        plan["science_identity"]["generation"]["autocast_enabled"] = True
    elif defect == "threshold":
        plan["science_identity"]["criterion"]["required_correct"] = 1
    elif defect == "helper":
        plan["control_sha256"]["tools/sdsc_student_instruction_job.py"] = "f" * 64
    elif defect == "overclaim":
        plan["g0_passed"] = True
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_exact_resource_argv_and_no_legacy_candidate_claim(c, plan):
    args = c.sbatch(plan)
    assert {
        "--gpus=h100:1",
        "--mem=65536M",
        "--cpus-per-task=8",
        "--time=00:30:00",
        "--no-requeue",
        "--export=NONE",
    } <= set(args)
    assert not any("constraint" in x for x in args)
    assert not hasattr(c, "LEGACY_INTENT") and not hasattr(c, "CANDIDATE_SHA")
    subprocess.run(["bash", "-n"], input=c.job_script(plan), check=True)


@pytest.mark.parametrize("correct", [0, 12, 13, 128])
def test_base_gate_uses_answer_correct_not_full_proof_or_completion(c, plan, correct):
    report, node, raw = fixture_reports(c, plan, correct=correct, proofs=0)
    records = {n: dict(path=n, size=len(v), sha256=c.sha(v)) for n, v in raw.items()}
    c.validate_worker_report(report, plan, "123", records)
    c.validate_node_result(node, plan, "123")
    c.validate_raw_summary(report, raw["initial-prompts.jsonl"], raw["initial-records.jsonl"])
    assert report["passed"] and report["initial_base_gate_satisfied"] is (correct >= 13)
    assert report["metrics"]["exact_proof_accuracy"] == 0 and all(report[k] is False for k in c.FLAGS)


@pytest.mark.parametrize(
    "defect",
    [
        "count_bool",
        "missing_row",
        "gate",
        "denominator",
        "proof_as_answer",
        "nan",
        "autocast",
        "FP32",
        "source",
        "named_source",
        "source_count",
        "raw_hash",
        "foreign_job",
    ],
)
def test_report_rejects_wrong_counts_scope_and_hashes(c, plan, defect):
    report, node, raw = fixture_reports(c, plan)
    records = {n: dict(path=n, size=len(v), sha256=c.sha(v)) for n, v in raw.items()}
    if defect == "count_bool":
        report["raw_record_count"] = True
    elif defect == "missing_row":
        report["evaluated_count"] = 127
    elif defect == "gate":
        report["initial_base_gate_satisfied"] = False
    elif defect == "denominator":
        report["metrics"]["answer_accuracy"] = 13 / 127
    elif defect == "proof_as_answer":
        report["answer_correct_count"] = 0
    elif defect == "nan":
        report["metrics"]["format_validity"] = float("nan")
    elif defect == "autocast":
        report["generation"]["autocast_enabled"] = True
    elif defect == "FP32":
        report["forward_parameter_dtype"] = "torch.float32"
    elif defect == "source":
        report["scientific_binding"]["head"] = "f" * 40
    elif defect == "source_count":
        report["scientific_binding"]["deployment_python_files_verified"] = 9
    elif defect == "named_source":
        report["scientific_binding"]["science_file_sha256"]["extra"] = "f" * 64
    elif defect == "raw_hash":
        report["raw_artifacts"][0]["sha256"] = "f" * 64
    elif defect == "foreign_job":
        report["job_id"] = "124"
    with pytest.raises(ValueError):
        c.validate_worker_report(report, plan, "123", records)


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "reorder", "token_bool", "text_hash", "count", "reward", "eos", "stop", "blank", "missing"],
)
def test_raw_summary_is_complete_and_bound(c, plan, defect):
    report, node, raw = fixture_reports(c, plan)
    rows = [json.loads(x) for x in raw["initial-records.jsonl"].splitlines()]
    if defect == "duplicate":
        rows[-1] = rows[0]
    elif defect == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == "token_bool":
        rows[0]["response_ids"][0] = True
    elif defect == "text_hash":
        rows[0]["prompt_text_sha256"] = "f" * 64
    elif defect == "count":
        rows[0]["verification"]["answer_correct"] = False
    elif defect == "reward":
        rows[0]["verification"]["reward"] = 1.0
    elif defect == "eos":
        rows[0]["response_ids"] = [151645, 3]
        rows[0]["stop_reason"] = "other"
    elif defect == "stop":
        rows[0]["stop_reason"] = "length"
    elif defect == "missing":
        rows.pop()
    data = b"".join(c.canonical(r) + b"\n" for r in rows) + (b"\n" if defect == "blank" else b"")
    with pytest.raises(ValueError):
        c.validate_raw_summary(report, raw["initial-prompts.jsonl"], data)


@pytest.mark.parametrize(
    "defect", ["phase", "headroom", "own_oom", "bool_counter", "foreign_job", "limit", "peak", "clock"]
)
def test_success_needs_five_strict_memory_phases(c, defect):
    memory = memory_fixture()
    c.validate_memory(memory, "123")
    row = memory["after_publication"]
    ancestor = row["ancestors"][0]
    if defect == "phase":
        memory.pop("after_publication")
    elif defect == "headroom":
        ancestor["peak_bytes"] = 40 * 1024**3
    elif defect == "own_oom":
        ancestor["memory_oom_control"]["oom_kill"] = 1
    elif defect == "bool_counter":
        ancestor["memory_failcnt"] = False
    elif defect == "foreign_job":
        ancestor["path"] = ancestor["path"].replace("123", "999")
    elif defect == "limit":
        ancestor["limit_bytes"] = 384 * 1024**3
    elif defect == "peak":
        row["peak_bytes"] = 1
    elif defect == "clock":
        row["observed_at_unix"] = 1
    with pytest.raises(ValueError):
        c.validate_memory(memory, "123")


def test_shared_oom_history_does_not_condemn_current_job(c):
    memory = memory_fixture()
    for row in memory.values():
        row["ancestors"].append(
            dict(
                path="/sys/fs/cgroup/memory/slurm",
                limit_bytes=None,
                memory_failcnt=4,
                memory_events=dict(oom=7),
            )
        )
    c.validate_memory(memory, "123")


@pytest.mark.parametrize("failed", [False, True])
def test_both_success_and_failed_publication_fetch_bind_exact_bytes(c, plan, failed):
    result = packaged(c, plan, failed)
    data = c.validate_download(result, plan)
    assert "receipt.json" in data and result["status"]["artifact_hashes_verified"]


@pytest.mark.parametrize(
    "defect",
    ["foreign_receipt", "tampered_raw", "missing_receipt", "missing_science", "unverified", "oversized_log"],
)
def test_failed_fetch_cannot_borrow_or_tamper_evidence(c, plan, defect):
    result = packaged(c, plan, True)
    if defect == "foreign_receipt":
        result["receipt"]["task"] = "other"
    elif defect == "tampered_raw":
        old = base64.b64decode(result["files"]["initial-probe.json"])
        result["files"]["initial-probe.json"] = base64.b64encode(old.replace(b"failed", b"broken")).decode()
    elif defect == "missing_receipt":
        result["bytes"] -= len(base64.b64decode(result["files"].pop("receipt.json")))
    elif defect == "missing_science":
        result["bytes"] -= len(base64.b64decode(result["files"].pop("initial-probe.json")))
    elif defect == "unverified":
        result["status"]["artifact_hashes_verified"] = False
    elif defect == "oversized_log":
        result["files"]["slurm.out"] = base64.b64encode(b"x" * 65537).decode()
        result["bytes"] += 65537
    with pytest.raises(ValueError):
        c.validate_download(result, plan)


def test_early_unpublished_failure_only_returns_bounded_logs(c, plan):
    result = dict(
        receipt=c.make_receipt(plan, "123"),
        status=dict(
            success=False,
            publication_verified=False,
            artifact_hashes_verified=False,
            evidence_state="unpublished_logs_only",
        ),
        files={"slurm.err": base64.b64encode(b"failed early").decode()},
        bytes=12,
    )
    assert c.validate_download(result, plan) == {"slurm.err": b"failed early"}
    result["files"]["initial-probe.json"] = base64.b64encode(b"{}").decode()
    result["bytes"] += 2
    with pytest.raises(ValueError):
        c.validate_download(result, plan)


def test_unknown_submission_is_permanent_across_control_versions(c, plan, monkeypatch):
    stub_remote(c, monkeypatch)
    calls = []

    def run(argv):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 45)

    monkeypatch.setattr(c, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert c.document(Path(plan["submission_dir"]) / "unknown.json")["no_retry"] is True
    changed = copy.deepcopy(plan)
    changed["control_sha256"][c.PROTOCOL_DOC] = "e" * 64
    changed["protocol_document_sha256"] = "e" * 64
    new = c.execution_intent(changed["control_sha256"])
    old = plan["intent_id"]
    for field in ("intent_id", "claim", "submission_dir", "result_dir", "job_name"):
        changed[field] = changed[field].replace(old, new)
    c.validate_plan(changed)
    with pytest.raises(ValueError, match="already claimed"):
        c.remote_action(dict(action="submit", plan=changed, authorize=True))
    assert len(calls) == 1 and not (c.CONTROL / "student-instruction-claims").exists()


def test_missing_authorization_never_writes_claim(c, plan, monkeypatch):
    stub_remote(c, monkeypatch)
    with pytest.raises(ValueError, match="authorization"):
        c.remote_action(dict(action="submit", plan=plan))
    assert not Path(plan["scientific_claim"]).exists()


def test_queue_never_uses_archived_job_by_id_or_ignores_foreign_owner(c, monkeypatch):
    monkeypatch.setattr(c.os, "getuid", lambda: 12345)
    monkeypatch.setattr(c.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="zgao12"))
    calls = []

    def run(argv):
        calls.append(argv)
        return dict(returncode=0, stdout="456_0|RUNNING|zgao12\n", stderr="")

    monkeypatch.setattr(c, "run", run)
    value = c.job_queue("123")
    assert value["stdout"] == "" and value["query_raw"]["stdout"] and "--states=all" in calls[0]
    assert not any(x.startswith("--jobs=") for x in calls[0])
    monkeypatch.setattr(c, "run", lambda argv: dict(returncode=0, stdout="456|RUNNING|foreign\n", stderr=""))
    with pytest.raises(ValueError):
        c.job_queue("123")


def test_gpu_tres_na_is_not_cpu_only(c, monkeypatch):
    user = c.pwd.getpwuid(c.os.getuid()).pw_name
    fields = (
        f"JobId=456 UserId={user}(123) JobState=RUNNING ReqTRES=cpu=24,node=1,gres/gpu=2,gres/gpu:h100=2 "
        "AllocTRES=cpu=24,node=1,gres/gpu=2,gres/gpu:h100=2 TresPerJob=gres:gpu:h100:2 Gres=(null)"
    )
    monkeypatch.setattr(
        c,
        "run",
        lambda argv: dict(
            returncode=0, stdout="456|RUNNING|N/A\n" if argv[0] == "squeue" else fields, stderr=""
        ),
    )
    assert c.check_gpu_ceiling()["existing_allocatable_gpus"] == 2
    fields = fields.replace("gpu=2", "gpu=4").replace("h100=2", "h100=4").replace("h100:2", "h100:4")
    with pytest.raises(ValueError, match="four"):
        c.check_gpu_ceiling()


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


def account(plan, comment="", failed=False):
    state, code = ("FAILED", "1:0") if failed else ("COMPLETED", "0:0")
    return dict(
        returncode=0,
        stderr="",
        stdout=(
            f"123|{state}|{code}|{plan['job_name']}|nwu181|nairr-gpu-shared|nairr-gpu-shared-normal|8|64G|60|cpu=8,mem=64G,node=1,gres/gpu=1|{comment}\n"
            f"123.batch|{state}|{code}|batch||||||||\n123.extern|COMPLETED|0:0|extern||||||||\n"
        ),
    )


def test_terminal_accounting_needs_zero_step_exits_and_bound_empty_comment(c, plan):
    queue = dict(returncode=0, stdout="", stderr="")
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account(plan))
    assert c.validate_accounting(plan, "123", queue, account(plan), fake_live(c, plan))["accounting_complete"]
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, account(plan, comment="other"), fake_live(c, plan))
    bad = account(plan)
    bad["stdout"] = bad["stdout"].replace("123.batch|COMPLETED|0:0", "123.batch|FAILED|1:0")
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", queue, bad, fake_live(c, plan))


def test_dry_run_is_read_only_and_checks_runtime_before_submission(c, plan, monkeypatch):
    calls = []
    stub_remote(c, monkeypatch)
    monkeypatch.setattr(c, "verify_parent", lambda p, **kwargs: calls.append(kwargs))
    binary = Path(plan["python"])
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixture-runtime")
    binary.chmod(0o700)
    Path(plan["hf_home"]).mkdir(parents=True)
    real_sha = c.sha
    monkeypatch.setattr(
        c,
        "sha",
        lambda data: "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19"
        if data == b"fixture-runtime"
        else real_sha(data),
    )
    monkeypatch.setattr(c, "run", lambda argv: pytest.fail("dry-run cannot submit"))
    result = c.remote_action(dict(action="dry-run", plan=plan))
    assert result["blockers"] == [] and result["source_verified"] and result["parent_verified"]
    assert (
        calls == [dict(accounting=True)]
        and not Path(plan["claim"]).exists()
        and not Path(plan["scientific_claim"]).exists()
    )


def test_submit_ack_is_saved_once_with_real_live_binding(c, plan, monkeypatch):
    stub_remote(c, monkeypatch)
    calls = []

    def run(argv):
        calls.append(argv)
        return dict(returncode=0, stdout="123;expanse\n", stderr="")

    monkeypatch.setattr(c, "run", run)
    monkeypatch.setattr(c, "live_binding", lambda p, j: fake_live(c, p, j))
    receipt = c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert receipt["job_id"] == "123" and c.document(Path(plan["submission_dir"]) / "receipt.json") == receipt
    with pytest.raises(ValueError):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))
    assert len(calls) == 1


def test_unknown_reconcile_with_zero_matches_never_submits(c, plan, monkeypatch):
    stub_remote(c, monkeypatch)
    monkeypatch.setattr(c, "run", lambda argv: (_ for _ in ()).throw(subprocess.TimeoutExpired(argv, 45)))
    with pytest.raises(subprocess.TimeoutExpired):
        c.remote_action(dict(action="submit", plan=plan, authorize=True))

    def empty(argv):
        assert argv[0] in {"squeue", "sacct"}
        return dict(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(c, "run", empty)
    with pytest.raises(ValueError, match="zero/ambiguous"):
        c.remote_action(dict(action="reconcile", plan=plan))
    assert c.document(Path(plan["submission_dir"]) / "unknown.json")["no_retry"]


def test_missing_master_cannot_consume_local_submit_intent(c, plan, monkeypatch, tmp_path):
    root = tmp_path / "quest"
    folder = root / ".sdsc/student-initial" / plan["intent_id"]
    folder.mkdir(parents=True)
    path = folder / "plan.json"
    path.write_bytes(c.canonical(plan))
    oldhelper, oldread = c.helper, c.read

    def missing():
        raise ValueError("manual authentication required")

    monkeypatch.setattr(c, "ROOT", root)
    monkeypatch.setattr(
        c,
        "helper",
        lambda name: SimpleNamespace(require_master=missing) if name == "sdsc_cli" else oldhelper(name),
    )
    monkeypatch.setattr(
        c,
        "read",
        lambda name, *args: b"control" if Path(name) in {root / n for n in c.TOOLS} else oldread(name, *args),
    )
    monkeypatch.setattr(c, "validate_plan", lambda value: value)
    oldsha = c.sha
    monkeypatch.setattr(c, "sha", lambda data: "a" * 64 if data == b"control" else oldsha(data))
    plan["control_sha256"] = dict.fromkeys(c.TOOLS, "a" * 64)
    path.write_bytes(c.canonical(plan))
    with pytest.raises(ValueError, match="authentication"):
        c.main(["submit", "--plan", str(path), "--authorize"])
    assert not (folder / "submission-started.json").exists()


def test_original_science_identity_returns_independent_nested_values(c):
    value = c.science_identity()
    value["generation"]["autocast_enabled"] = True
    assert c.GENERATION["autocast_enabled"] is False


@pytest.mark.parametrize("state,exit_code", [("COMPLETED", "0:0"), ("FAILED", "1:0"), ("TIMEOUT", "0:15")])
def test_confirmed_terminal_cache_keeps_historical_gpu_request_but_not_allocation(
    c, monkeypatch, state, exit_code
):
    user = c.pwd.getpwuid(c.os.getuid()).pw_name
    live = (
        f"JobId=456 UserId={user}(123) JobState={state} ExitCode={exit_code} ReqTRES=cpu=24,node=1,gres/gpu=2"
    )

    def run(argv):
        if argv[0] == "squeue":
            text = f"456|{state}|N/A\n"
        elif argv[0] == "scontrol":
            text = live
        else:
            assert argv[0] == "sacct"
            text = f"456|{state}|{exit_code}|{user}\n"
        return dict(returncode=0, stdout=text, stderr="")

    monkeypatch.setattr(c, "run", run)
    result = c.check_gpu_ceiling()
    job = result["jobs"][0]
    assert result["existing_allocatable_gpus"] == 0
    assert result["total_historical_requested_gpus"] == job["requested_gpus"] == 2
    assert (
        job["terminal_cache_confirmed"]
        and job["terminal_accounting"]["stdout"]
        and job["live_query"]["stdout"]
    )


@pytest.mark.parametrize(
    "defect",
    [
        "queue_state",
        "live_state",
        "live_owner",
        "live_exit",
        "account_state",
        "account_exit",
        "account_owner",
        "account_missing",
        "account_error",
        "account_duplicate",
    ],
)
def test_terminal_cache_cannot_hide_active_unknown_or_foreign_gpu(c, monkeypatch, defect):
    user = c.pwd.getpwuid(c.os.getuid()).pw_name
    queue = "456|COMPLETED|N/A\n"
    live = f"JobId=456 UserId={user}(123) JobState=COMPLETED ExitCode=0:0 ReqTRES=cpu=24,node=1,gres/gpu=2"
    account = f"456|COMPLETED|0:0|{user}\n"
    rc = 0
    if defect == "queue_state":
        queue = queue.replace("COMPLETED", "RUNNING")
    elif defect == "live_state":
        live = live.replace("JobState=COMPLETED", "JobState=UNKNOWN")
    elif defect == "live_owner":
        live = live.replace(f"UserId={user}", "UserId=foreign")
    elif defect == "live_exit":
        live = live.replace("ExitCode=0:0", "ExitCode=N/A")
    elif defect == "account_state":
        account = account.replace("COMPLETED", "RUNNING")
    elif defect == "account_exit":
        account = account.replace("0:0", "1:0")
    elif defect == "account_owner":
        account = account.replace(user, "foreign")
    elif defect == "account_missing":
        account = ""
    elif defect == "account_error":
        rc = 1
    elif defect == "account_duplicate":
        account += account

    def run(argv):
        if argv[0] == "squeue":
            return dict(returncode=0, stdout=queue, stderr="")
        if argv[0] == "scontrol":
            return dict(returncode=0, stdout=live, stderr="")
        assert argv[0] == "sacct"
        return dict(returncode=rc, stdout=account, stderr="")

    monkeypatch.setattr(c, "run", run)
    with pytest.raises(ValueError):
        c.check_gpu_ceiling()


def test_exact_completed_queue_cache_and_all_successful_steps_are_terminal(c, plan):
    q = dict(returncode=0, stdout="123|COMPLETED\n", stderr="")
    result = c.validate_accounting(plan, "123", q, account(plan), fake_live(c, plan))
    assert result["accounting_complete"] and result["queue"] == q
    for state in ("RUNNING", "COMPLETING", "UNKNOWN", "FAILED"):
        changed = dict(q, stdout="123|" + state + "\n")
        with pytest.raises(ValueError):
            c.validate_accounting(plan, "123", changed, account(plan), fake_live(c, plan))
    incomplete = account(plan)
    incomplete["stdout"] = incomplete["stdout"].replace("123.batch|COMPLETED|0:0", "123.batch|FAILED|1:0")
    with pytest.raises(ValueError):
        c.validate_accounting(plan, "123", q, incomplete, fake_live(c, plan))
