"""CPU execution tests of the four-rank teacher worker's export and control edges."""

from __future__ import annotations

import copy
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.learning.teacher import adaptation_fit
from posttrain_circuits.learning.teacher.adaptation import LORA_MODULES, _tensor_digest, collate
from posttrain_circuits.models import adapted_teacher, loading
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.models.prompt_protocol import chat_template_sha256
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("_test_teacher_fit_worker", ROOT / "tools/sdsc_teacher_fit.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class Observation:
    def phase(self, *_args, **_kwargs):
        pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


@pytest.fixture
def adapter_checkpoint(tmp_path, monkeypatch):
    peft = pytest.importorskip("peft")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: pytest.fail("CPU export touched CUDA"))
    base_id, base_revision = "test-only/worker-tiny-qwen3", "a" * 40
    monkeypatch.setattr(adapted_teacher, "BASE_MODEL_ID", base_id)
    monkeypatch.setattr(adapted_teacher, "BASE_REVISION", base_revision)
    tokenizer = build_tiny_tokenizer()
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    training = peft.get_peft_model(
        build_tiny_qwen3(71),
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=list(LORA_MODULES),
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    with torch.no_grad():
        for name, value in training.named_parameters():
            if "lora_B" in name:
                value.fill_(0.01)
    checkpoint = tmp_path / "step-000001"
    checkpoint.mkdir()
    training.save_pretrained(checkpoint / "adapter", safe_serialization=True)
    calls = []

    def fresh_base(config, *, for_training):
        assert for_training is False
        calls.append(config)
        return SimpleNamespace(model=build_tiny_qwen3(71), tokenizer=tokenizer)

    monkeypatch.setattr(loading, "load_model_and_tokenizer", fresh_base)
    config = {
        "model_name_or_path": base_id,
        "model_revision": base_revision,
        "tokenizer_name_or_path": base_id,
        "tokenizer_revision": base_revision,
        "torch_dtype": "float32",
        "attn_implementation": "sdpa",
        "gradient_checkpointing": False,
        "use_cache": True,
        "trust_remote_code": False,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": chat_template_sha256(tokenizer),
        },
    }
    row = {"input_ids": [2, 5, 9, 3], "labels": [-100, -100, 9, 3]}
    provenance = {
        "actual_plan": {"cpu_test": True},
        "run_id": "cpu-test",
        "job_id": "test-only",
        "code_sha256": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "train_metrics_sha256": "d" * 64,
        "optimizer_steps": 1,
        "consumed_tokens": 256,
    }
    return checkpoint, training, tokenizer, config, row, provenance, calls


def test_real_saved_adapter_exports_fresh_dense_and_verified_reload_without_training_mutation(
    adapter_checkpoint,
):
    checkpoint, training, tokenizer, config, row, provenance, calls = adapter_checkpoint
    before = _tensor_digest(training.named_parameters())
    metadata, expected = worker.export_dense_checkpoint(
        checkpoint,
        row,
        config,
        provenance,
        torch.device("cpu"),
        Observation(),
    )
    assert len(calls) == 1
    assert _tensor_digest(training.named_parameters()) == before
    assert any("lora_" in name for name, _ in training.named_parameters())
    assert metadata["dense_weights_sha256_before"] != metadata["dense_weights_sha256_after"]
    assert metadata["adapter_merge_max_logit_error"] < 1e-5
    manifest = json.loads((checkpoint / "dense-manifest.json").read_text())
    assert metadata["manifest"] == manifest
    assert manifest["adaptation_plan_sha256"] == sha256_value(provenance["actual_plan"])
    assert manifest["training_provenance"] == {
        key: value for key, value in provenance.items() if key != "actual_plan"
    }
    assert manifest["formal_teacher_accepted"] is False
    loaded = adapted_teacher.load_adapted_teacher(
        checkpoint, manifest["sha256"], config, device=torch.device("cpu")
    )
    batch = collate([row], tokenizer.pad_token_id, "cpu")
    with torch.inference_mode():
        actual = (
            loaded.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            .logits[:, -1]
            .float()
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded.teacher_checkpoint_sha256 == manifest["sha256"]
    assert loaded.tokenizer_hash == tokenizer_fingerprint(tokenizer)
    assert not hasattr(loaded, "resolved_model_commit")
    assert not any("lora_" in name for name, _ in loaded.model.named_parameters())
    assert all(not value.requires_grad for value in loaded.model.parameters())
    with pytest.raises(ValueError, match="already exists"):
        worker.export_dense_checkpoint(
            checkpoint, row, config, provenance, torch.device("cpu"), Observation()
        )


def test_export_rejects_zero_merge_before_dense_publication(adapter_checkpoint):
    checkpoint, training, _, config, row, provenance, _ = adapter_checkpoint
    with torch.no_grad():
        for name, value in training.named_parameters():
            if "lora_B" in name:
                value.zero_()
    training.save_pretrained(checkpoint / "adapter", safe_serialization=True)
    with pytest.raises(ValueError, match="did not change actual dense"):
        worker.export_dense_checkpoint(
            checkpoint, row, config, provenance, torch.device("cpu"), Observation()
        )
    assert not (checkpoint / "dense-manifest.json").exists()
    assert not (checkpoint / "merged").exists()


def runtime_fixture(monkeypatch):
    for name, value in {
        "SLURM_JOB_ID": "12345",
        "SLURM_CPUS_PER_TASK": "24",
        "SLURM_MEM_PER_NODE": "196608",
        "SLURM_JOB_CPUS_PER_NODE": "72",
        "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: {"torch": "pinned"}[name])
    monkeypatch.setattr(worker.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(worker, "file_hash", lambda path: "a" * 64)
    contract = SimpleNamespace(
        DEPENDENCIES={"torch": "pinned"},
        FIXED_EXECUTION={"runtime": {"python": "3.12.13", "python_sha256": "a" * 64}},
    )
    guards = SimpleNamespace(require=require, local_environment=lambda args: {"staged": True})
    monkeypatch.setattr(
        worker,
        "sibling",
        lambda name: SimpleNamespace(local_environment=lambda args, guards: {"staged": True}),
    )
    return SimpleNamespace(job_id="12345"), contract, guards


def test_runtime_allocation_preserves_assigned_visibility_and_sets_six_threads(monkeypatch):
    args, contract, guards = runtime_fixture(monkeypatch)
    allocation, runtime, environment = worker.runtime_and_allocation(args, contract, guards)
    assert allocation["cuda_visible_devices"] == "GPU-a,GPU-b,GPU-c,GPU-d"
    assert worker.os.environ["CUDA_VISIBLE_DEVICES"] == allocation["cuda_visible_devices"]
    assert runtime["packages"] == contract.DEPENDENCIES and environment == {"staged": True}
    assert allocation["requested_cpus_per_task"] == 24
    assert allocation["allocated_cpus_on_node"] == 72
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert worker.os.environ[name] == "6"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SLURM_JOB_ID", "999"),
        ("SLURM_CPUS_PER_TASK", "23"),
        ("SLURM_MEM_PER_NODE", "196607"),
        ("SLURM_JOB_CPUS_PER_NODE", ""),
        ("SLURM_JOB_CPUS_PER_NODE", "12"),
        ("SLURM_JOB_CPUS_PER_NODE", "72(x2)"),
        ("CUDA_VISIBLE_DEVICES", "0,1,2"),
        ("CUDA_VISIBLE_DEVICES", "0,1,2,2"),
        ("CUDA_VISIBLE_DEVICES", "0,1,2, -1"),
    ],
)
def test_runtime_rejects_wrong_allocation_before_training(monkeypatch, name, value):
    args, contract, guards = runtime_fixture(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        worker.runtime_and_allocation(args, contract, guards)


def test_nonzero_rank_observation_creates_its_directory_and_preserves_scope(tmp_path):
    report = {
        "task": "qwen3-v2-teacher-fit-preflight",
        "job_id": "123",
        "run_id": "cpu-test",
        "code_sha256": "a" * 64,
    }
    observation = worker.rank_observation(
        worker.sibling("sdsc_teacher_probe"), worker.sibling("sdsc_teacher_adapt"), tmp_path, 2, report
    )
    try:
        record = json.loads((tmp_path / "rank-2-progress/progress.json").read_text())
        assert record["artifact_kind"] == "teacher_adaptation_progress"
        assert record["training_started"] is False
        observation.phase("teacher_training_start")
        record = json.loads((tmp_path / "rank-2-progress/progress.json").read_text())
        assert record["teacher_training_started"] is True
        assert record["student_training_started"] is record["full_teacher_ready"] is False
    finally:
        observation.close()


@pytest.mark.parametrize("failure", ["validation", "export"])
def test_collective_export_failure_prevents_metadata_broadcast_and_reload(monkeypatch, tmp_path, failure):
    calls = []

    def shared(function, group, phase):
        calls.append(phase)
        return function()

    def validate(*args):
        if failure == "validation":
            raise ValueError("saved state identity failed")

    def export(*args):
        raise ValueError("real export failed")

    monkeypatch.setattr(adaptation_fit, "_shared_call", shared)
    monkeypatch.setattr(worker, "validate_checkpoint_for_export", validate)
    monkeypatch.setattr(worker, "export_dense_checkpoint", export)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *_a, **_k: pytest.fail("broadcast after failure")
    )
    with pytest.raises(ValueError, match="failed"):
        worker.export_checkpoint_collectively(
            tmp_path, {}, {}, {}, {}, torch.device("cpu"), Observation(), rank=0, control=None
        )
    assert calls[0] == "saved adapter checkpoint verification"
    assert len(calls) == (1 if failure == "validation" else 2)


def test_changed_world_probe_does_not_accept_an_unrelated_checkpoint_error(monkeypatch, tmp_path):
    def rejected(*_args):
        raise ValueError("teacher checkpoint identity differs: world_size")

    monkeypatch.setattr(adaptation_fit, "validate_fit_checkpoint", rejected)
    assert worker.require_changed_world_rejection(tmp_path) is True

    def corrupt(*_args):
        raise ValueError("teacher checkpoint file differs")

    monkeypatch.setattr(adaptation_fit, "validate_fit_checkpoint", corrupt)
    with pytest.raises(ValueError, match="file differs"):
        worker.require_changed_world_rejection(tmp_path)


@pytest.mark.parametrize("failed", [False, True])
def test_checkpoint_memory_is_measured_before_gather_and_failure_stops_gather(monkeypatch, failed):
    events = []

    def measure():
        events.append("measure")
        if failed:
            raise ValueError("192 GiB cgroup lacks headroom")
        return {"passed": True, "peak_bytes": 123}

    def shared(function, control, phase):
        assert phase == "checkpoint cgroup headroom"
        events.append("shared")
        return function()

    def gather(output, value, *, group):
        events.append("gather")
        assert value == {"rank": 2, "cgroup_memory": {"passed": True, "peak_bytes": 123}}
        output[:] = [{"rank": rank, "cgroup_memory": value["cgroup_memory"]} for rank in range(4)]

    monkeypatch.setattr(worker, "sibling", lambda _name: SimpleNamespace(memory_envelope=measure))
    monkeypatch.setattr(adaptation_fit, "_shared_call", shared)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    if failed:
        with pytest.raises(ValueError, match="lacks headroom"):
            worker.checkpoint_memory_evidence(None, 2)
        assert events == ["shared", "measure"]
    else:
        result = worker.checkpoint_memory_evidence(None, 2)
        assert [row["rank"] for row in result] == [0, 1, 2, 3]
        assert events == ["shared", "measure", "gather"]


def finish_fixture(tmp_path, mode="preflight"):
    contract = worker.sibling("sdsc_teacher_fit_contract")
    actual = contract.actual_plan(mode)
    args = SimpleNamespace(mode=mode, output_dir=tmp_path, work_dir=tmp_path)
    (tmp_path / "train-metrics.jsonl").write_text("{}\n")
    (tmp_path / "data-isolation.json").write_text("{}\n")
    (tmp_path / "prerequisites.json").write_text(json.dumps({"preflight_job_id": "12344"}))
    report = {
        "actual_plan": actual,
        "run_id": "cpu-test",
        "job_id": "12345",
        "code_sha256": "a" * 64,
        "execution_plan_sha256": "b" * 64,
        "actual_plan_sha256": sha256_value(actual),
        "adaptation_dataset_sha256": "c" * 64,
        "accepted_science": False,
        "formal_teacher_accepted": False,
        "full_teacher_ready": False,
        "g0_passed": False,
        "execution_class_certified": False,
        "student_training_started": False,
        "readiness_artifact_produced": False,
        "original_128_token_readiness_pass_claim": False,
    }
    ranks = [
        {
            "rank": rank,
            "training": {
                "completed_updates": actual["optimizer_steps"],
                "consumed_tokens": 2048,
                "same_world_resume_verified": mode == "preflight",
            },
            "cgroup_memory": {"passed": True},
        }
        for rank in range(4)
    ]
    checkpoints = [
        {
            "step": step,
            "dev_metrics_passed": True,
            "adapted_teacher_sha256": str(index + 1) * 64,
            "dense_reload_max_logit_error": 0.0,
            "adapter_merge_max_logit_error": 0.0,
            "cgroup_memory_by_rank": [{"rank": rank, "cgroup_memory": {"passed": True}} for rank in range(4)],
        }
        for index, step in enumerate(actual["checkpoint_steps"])
    ]
    return args, report, checkpoints, ranks, contract


@pytest.mark.parametrize("mode", ["preflight", "full-fit"])
def test_execution_report_never_claims_formal_acceptance_and_selects_first_scheduled_pass(tmp_path, mode):
    args, report, checkpoints, ranks, contract = finish_fixture(tmp_path, mode)
    worker.finish_report(args, report, checkpoints, ranks, True, contract, time.monotonic())
    assert report["passed"] is True and set(report["checks"]) == contract.EXECUTION_CHECKS
    for key in (
        "accepted_science",
        "formal_teacher_accepted",
        "full_teacher_ready",
        "g0_passed",
        "execution_class_certified",
        "student_training_started",
        "readiness_artifact_produced",
        "original_128_token_readiness_pass_claim",
    ):
        assert report[key] is False
    assert report["same_world_resume_performed_this_attempt"] is (mode == "preflight")
    assert report["selected_checkpoint_sha256"] == (
        None if mode == "preflight" else checkpoints[0]["adapted_teacher_sha256"]
    )
    assert report["selection_does_not_stop_training"] is True
    assert report["optimizer_steps"] == (8 if mode == "preflight" else 512)


@pytest.mark.parametrize(
    "defect",
    ["rank_duplicate", "tokens", "resume", "cgroup", "changed_world", "nonfinite", "checkpoint_memory"],
)
def test_execution_report_rejects_missing_or_false_evidence(tmp_path, defect):
    args, report, checkpoints, ranks, contract = finish_fixture(tmp_path)
    if defect == "rank_duplicate":
        ranks.append(copy.deepcopy(ranks[-1]))
    elif defect == "tokens":
        ranks[2]["training"]["consumed_tokens"] += 1
    elif defect == "resume":
        ranks[3]["training"]["same_world_resume_verified"] = False
    elif defect == "cgroup":
        ranks[1]["cgroup_memory"]["passed"] = False
    elif defect == "nonfinite":
        checkpoints[0]["dense_reload_max_logit_error"] = float("nan")
    elif defect == "checkpoint_memory":
        checkpoints[0]["cgroup_memory_by_rank"][2]["cgroup_memory"]["passed"] = False
    with pytest.raises(ValueError):
        worker.finish_report(
            args, report, checkpoints, ranks, defect != "changed_world", contract, time.monotonic()
        )
    assert report.get("passed") is not True
