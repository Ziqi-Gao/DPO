"""CPU regression for the bounded LR diagnostic, not real W2/GPU acceptance."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.teacher_demos.contracts import TeacherDemoAttempt
from posttrain_circuits.learning.teacher.demo_source import TeacherDemoStateSource
from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor
from posttrain_circuits.learning.training.factorial_trainer import FactorialTrainer, TrainerConfig
from posttrain_circuits.learning.training.schedules import PromptScheduler
from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location("_test_" + name, ROOT / "tools" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def probe():
    return module("sdsc_student_lr_probe")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    assert not torch.cuda.is_initialized()


def attempt(index):
    """Synthetic accepted-view record with varying prompt/response spans."""
    response = [5 + index % 10] * (1 + index % 3) + [3]
    return TeacherDemoAttempt(
        attempt_id=f"fixture-{index}:candidate-0000",
        prompt_id=f"fixture-{index}",
        prompt_identity_sha256="a" * 64,
        candidate_index=0,
        raw_prompt_text="fixture graph",
        model_facing_prompt_text="fixture model prompt",
        input_ids=[2, 4] + [6] * (index % 2),
        response_ids=response,
        response_text="fixture response",
        response_token_mask=[True] * len(response),
        behavior_logprobs=[-0.1] * len(response),
        logprob_status="available",
        finish_reason="eos",
        teacher_id="fixture-teacher",
        teacher_revision="fixture-revision",
        sampling_request_seed=42,
        actual_sampling_seed=index,
        sampling_protocol_id="fixture-sampling",
        sampling_temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        verifier_reward=1.0,
        verification_trace={"accepted": True},
        accepted=True,
        prompt_protocol="fixture-chat",
        enable_thinking=False,
        chat_template_sha256="b" * 64,
        raw_prompt_sha256="c" * 64,
        model_facing_prompt_sha256="d" * 64,
        tokenizer_fingerprint="e" * 64,
    )


def row(index):
    value = attempt(index)
    return dict(prompt_id=value.prompt_id, input_ids=value.input_ids, response_ids=value.response_ids)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_qwen_sequence_ce_is_independent_hf_mean_not_token_mean(probe, dtype):
    model = build_tiny_qwen3(129).to(dtype).train()
    model.model.layers[0].self_attn.eval()  # Preserve a genuinely heterogeneous mode tree.
    flags = [item.training for item in model.modules()]
    weights = copy.deepcopy(model.state_dict())
    rng = RNGState.capture().as_dict()
    rows = [row(0), row(2)]
    actual = probe.aggregate_nll(probe.teacher_forced_rows(model, rows))
    assert RNGState.capture().as_dict() == rng
    assert [item.training for item in model.modules()] == flags
    expected = []
    with torch.no_grad():
        model.eval()
        for item in rows:
            ids = torch.tensor([item["input_ids"] + item["response_ids"]])
            labels = ids.clone()
            labels[:, : len(item["input_ids"])] = -100
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids, dtype=torch.bool),
                labels=labels,
                use_cache=False,
            )
            expected.append(float(output.loss))
    assert actual["sequence_mean_nll"] == pytest.approx(sum(expected) / 2, rel=2e-6)
    assert actual["hf_sequence_mean_nll"] == pytest.approx(sum(expected) / 2, rel=2e-6)
    token_mean = (
        sum(value * len(item["response_ids"]) for value, item in zip(expected, rows, strict=False)) / 6
    )
    assert actual["token_mean_nll"] == pytest.approx(token_mean, rel=2e-6)
    assert abs(token_mean - sum(expected) / 2) > 1e-5
    assert actual["response_tokens_including_eos"] == 6
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonfinite_hf_logits_fail_without_update(probe):
    model = build_tiny_qwen3(130)
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        probe.teacher_forced_rows(model, [row(0)])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_measurement_restores_all_rng_modes_on_exception_and_rejects_cursor_change(probe):
    model = build_tiny_qwen3(131).train()
    model.model.layers[0].eval()
    flags = [item.training for item in model.modules()]
    before = RNGState.capture().as_dict()
    with pytest.raises(RuntimeError, match="fixture failure"), probe.measurement(model):
        random.random()
        np.random.rand()
        torch.rand(9)
        assert not any(item.training for item in model.modules())
        raise RuntimeError("fixture failure")
    assert RNGState.capture().as_dict() == before
    assert [item.training for item in model.modules()] == flags
    state = {"cursor": 0}
    part = SimpleNamespace(state_dict=lambda: state.copy())
    trainer = SimpleNamespace(
        state_source=part, prompt_scheduler=part, token_budget=part, global_step=0, cumulative_counts={}
    )
    with pytest.raises(ValueError, match="advanced training state"), probe.measurement(model, trainer):
        state["cursor"] += 1
    assert RNGState.capture().as_dict() == before


def test_measurement_detects_optimizer_moment_mutation_without_full_copies(probe):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    part = SimpleNamespace(state_dict=lambda: {})
    trainer = SimpleNamespace(
        state_source=part,
        prompt_scheduler=part,
        token_budget=part,
        global_step=1,
        cumulative_counts={},
        optimizer=optimizer,
        scheduler=scheduler,
    )
    state = optimizer.state[next(model.parameters())]["exp_avg"]
    with pytest.raises(ValueError, match="advanced training state"), probe.measurement(model, trainer):
        state.add_(1)


@pytest.mark.parametrize("lr", [5e-4, 5e-5])
def test_actual_adamw_four_updates_and_first_step_fp64_prediction(probe, lr):
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.125, -0.125, 1e-8, 20.0]))
    initial = parameter.detach().clone()
    optimizer = torch.optim.AdamW([parameter], lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    observer = probe.AdamWObserver(optimizer, lr, lambda: [initial])
    for step in range(4):
        parameter.grad = torch.tensor([1.0, -0.1, 1e-10, 0.0, -3.0]) / (step + 1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        assert observer.calls == scheduler.last_epoch == step + 1
    first = observer.rows[0]["first_step_prediction"]
    assert first["elements"] == 5 and first["roundoff_violations"] == 0
    assert first["gradient_dot_update"] < 0 and first["update_squared"] > 0
    assert [value["step"] for value in observer.rows] == [1, 2, 3, 4]
    assert all(value["all_parameter_steps_equal"] for value in observer.rows)
    with pytest.raises(ValueError, match="extra optimizer"):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    observer.close()


def test_actual_trainer_eight_microsteps_observed_once_and_no_evaluator(probe, tmp_path):
    model = build_tiny_qwen3(132)
    demos = [attempt(index) for index in range(32)]
    source = TeacherDemoStateSource(demos)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    prompt_scheduler = PromptScheduler([value.prompt_id for value in demos], ["graph"] * 32, 4)
    trainer = FactorialTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=prompt_scheduler,
        state_source=source,
        supervisor=CanonicalSFTSupervisor(0, retry_limit=0),
        config=TrainerConfig(
            max_steps=4,
            gradient_accumulation_steps=8,
            backend="torch_smoke",
            require_evaluation_metrics=False,
        ),
        run_dir=tmp_path,
        evaluation_fn=None,
    )
    observer = probe.AdamWObserver(optimizer, 5e-4, lambda: trainer._parameters_before_update)
    for step in range(1, 5):
        calls = []
        for microstep in range(8):
            prompt = trainer.prompt_scheduler.next_batch()
            trajectories, batch, attempts = trainer._collect_trajectories(prompt)
            trainer._register_collection(prompt, trajectories, batch, attempts)
            sync = trainer._training_micro_step(batch, expected_sequence_count=4)
            assert sync is (microstep == 7)
            calls.append(observer.calls)
        assert calls == [step - 1] * 7 + [step]
        metric = trainer._finish_optimizer_update(started=0)
        assert metric["optimizer_updates"] == step
        assert metric["validation_accuracy"] is None
        assert scheduler.last_epoch == step
    observer.close()
    assert observer.rows[0]["first_step_prediction"]["roundoff_violations"] == 0


def test_first_step_check_detects_double_missing_wrong_sign_and_chunk_tail(probe, monkeypatch):
    monkeypatch.setattr(probe, "CHUNK", 3)
    old = torch.tensor([0.01, -0.03, 0.0, 1e-8, 2.0, -2.0, 0.25])
    grad = torch.tensor([1.0, -1.0, 1e-12, 0.0, 0.2, -0.2, 2.0])
    correct = (old.double() - 5e-4 * grad.double() / (grad.double().abs() + 1e-8)).float()
    good = probe.first_step_statistics(old, grad, correct, learning_rate=5e-4)
    assert good["elements"] == 7 and good["roundoff_violations"] == 0
    for wrong in (old, old + (old - correct), old + 2 * (correct - old)):
        assert probe.first_step_statistics(old, grad, wrong, learning_rate=5e-4)["roundoff_violations"] > 0


def test_real_teacher_source_four_distinct_windows_and_actual_supervision_capture(probe):
    demos = [attempt(index) for index in range(256)]
    ids = [value.prompt_id for value in demos]
    totals, captured, cursors = [], [], []
    for rank in range(2):
        source = TeacherDemoStateSource(demos)
        scheduler = PromptScheduler.for_allocation_neutral_rank(
            ids,
            ["graph"] * 256,
            rank=rank,
            world_size=2,
            global_batch_size=64,
            max_microbatch_size=4,
        )
        supervisor = CanonicalSFTSupervisor(0, retry_limit=0)

        def collect(prompt, source=source, supervisor=supervisor):
            trajectories = source.get_batch(None, prompt, 0)
            return trajectories, supervisor.prepare_targets(trajectories, None, None), 1

        trainer = SimpleNamespace(
            prompt_scheduler=scheduler,
            _batch_partition_plan=scheduler.exact_global_batch_plan,
            _collect_trajectories=collect,
            _register_collection=lambda *args: None,
        )
        seen = []
        for step in range(1, 5):
            batches, records = probe.collect_window(trainer, ids, rank=rank, step=step, pad_token_id=0)
            assert len(batches) == len(records) == 8
            seen.extend(item["prompt_id"] for record in records for item in record["records"])
            if step == 1:
                captured.append(records)
        totals.extend(seen)
        cursors.append(source.state_dict()["cursor"])
    assert len(totals) == len(set(totals)) == 256 and set(totals) == set(ids)
    assert all(value == 1 for cursor in cursors for value in cursor.values())
    expected = [
        dict(
            global_slot=index,
            prompt_id=value.prompt_id,
            attempt_id=value.attempt_id,
            input_ids=value.input_ids,
            response_ids=value.response_ids,
            response_token_mask=value.response_token_mask,
        )
        for index, value in enumerate(demos[:64])
    ]
    assert probe.validate_first_window(captured, expected) == probe.digest(probe.canonical(expected))
    damaged = copy.deepcopy(captured)
    damaged[0][0]["records"][0]["response_ids"][-1] = 9
    with pytest.raises(ValueError, match="audited capture"):
        probe.validate_first_window(damaged, expected)


@pytest.mark.parametrize("defect", ["first_response", "prompt", "padding", "eos", "tokens", "dtype"])
def test_actual_collation_capture_rejects_mask_shift_and_eos_loss(probe, defect):
    demos = [attempt(index) for index in range(4)]
    source = TeacherDemoStateSource(demos)
    scheduler = PromptScheduler([item.prompt_id for item in demos], ["graph"] * 4, 4)
    trajectories = source.get_batch(None, scheduler.next_batch(), 0)
    batch = CanonicalSFTSupervisor(0).prepare_targets(trajectories, None, None)
    probe.batch_capture(trajectories, batch, rank=0, microstep=0, slots=[0, 2, 4, 6], pad_token_id=0)
    start = len(demos[0].input_ids)
    if defect == "first_response":
        batch.response_mask[0, start] = False
    elif defect == "prompt":
        batch.response_mask[0, start - 1] = True
    elif defect == "padding":
        batch.response_mask[0, -1] = True
    elif defect == "eos":
        batch.response_mask[0, start + len(demos[0].response_ids) - 1] = False
    elif defect == "tokens":
        batch.input_ids[0, start] += 1
    else:
        batch.response_mask = batch.response_mask.long()
    with pytest.raises(ValueError):
        probe.batch_capture(trajectories, batch, rank=0, microstep=0, slots=[0, 2, 4, 6], pad_token_id=0)


def test_real_qwen_fp32_export_exact_loading_before_bf16_inference(probe):
    model = build_tiny_qwen3(133).to(torch.bfloat16)
    model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
    state = {
        key: value.float() if value.is_floating_point() else value.clone()
        for key, value in model.state_dict().items()
    }
    state["model.embed_tokens.weight"].flatten()[0] = 0.123456789
    assert (
        state["model.embed_tokens.weight"].flatten()[0]
        != state["model.embed_tokens.weight"].bfloat16().float().flatten()[0]
    )
    original = copy.deepcopy(state)
    result = probe.strict_load_export(model, state)
    assert result == dict(all_tensors_exact=True, key_count=len(state), loaded_before_bf16_copy=True)
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())
    assert all(torch.equal(value, original[key]) for key, value in state.items())


@pytest.mark.parametrize("defect", ["keys", "shape", "dtype", "nan", "nonfloating"])
def test_export_loading_rejects_incompatible_state_before_copy(probe, monkeypatch, defect):
    model = build_tiny_qwen3(134).to(torch.bfloat16)
    model.register_buffer("counter", torch.tensor(3, dtype=torch.int64))
    state = {
        key: value.float() if value.is_floating_point() else value.clone()
        for key, value in model.state_dict().items()
    }
    key = "model.embed_tokens.weight"
    if defect == "keys":
        del state[key]
    elif defect == "shape":
        state[key] = state[key][:-1]
    elif defect == "dtype":
        state[key] = state[key].bfloat16()
    elif defect == "nan":
        state[key].flatten()[0] = float("nan")
    else:
        state["counter"] = state["counter"].int()
    calls = []
    monkeypatch.setattr(model, "load_state_dict", lambda *args, **kwargs: calls.append(1))
    with pytest.raises(ValueError):
        probe.strict_load_export(model, state)
    assert not calls


def test_logit_parity_reports_actual_error_without_quality_threshold(probe, monkeypatch):
    monkeypatch.setattr(probe, "CHUNK", 3)
    reference = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 2.0, 1.0]]])
    changed = reference.clone()
    changed[0, 0, 0] = 9
    result = probe.difference_statistics(reference, changed)
    assert result["max_abs_error"] == 9 and result["rms_error"] == pytest.approx((81 / 6) ** 0.5)
    assert result["argmax_mismatches"] == 1 and result["compared_positions"] == 2
    assert result["formal_parity_gate_applied"] is False
    changed[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        probe.difference_statistics(reference, changed)


def test_original_prompt_does_not_change_with_label_and_generation_is_bounded(probe, monkeypatch):
    task = ProofGraphTask()
    examples = [
        task.generate(index, {"depth": 1, "distractors": 0, "structure": "chain"}) for index in range(32)
    ]
    tokenizer = build_tiny_tokenizer()
    prompts, _ = probe.prepare_prompts(examples, tokenizer, {})
    changed = copy.deepcopy(examples)
    for item in changed:
        item.label = 1 - item.label
        item.canonical_proof = []
    other, _ = probe.prepare_prompts(changed, tokenizer, {})
    assert [item["prompt_ids"] for item in prompts] == [item["prompt_ids"] for item in other]
    model = build_tiny_qwen3(135).train()
    seen = []

    def fixed_response(**kwargs):
        assert (
            kwargs["max_new_tokens"] == 256 and kwargs["do_sample"] is False and kwargs["use_cache"] is False
        )
        assert kwargs["attention_mask"].dtype == torch.bool
        assert not model.training and not torch.is_grad_enabled()
        seen.append(kwargs["input_ids"].clone())
        return torch.cat((kwargs["input_ids"], torch.tensor([[tokenizer.eos_token_id]])), dim=1)

    monkeypatch.setattr(model, "generate", fixed_response)
    records = []
    rng = RNGState.capture().as_dict()
    metrics = probe.generate_records(
        model, tokenizer, examples, prompts, arm="lr-5e-4", step=0, on_record=records.append
    )
    assert len(records) == len(seen) == 32 and metrics["answer_accuracy"] == 0
    assert model.training and RNGState.capture().as_dict() == rng
    assert [record["ordinal"] for record in records] == list(range(32))
    assert all(record["response_ids"] == [tokenizer.eos_token_id] for record in records)
    assert all("parsed_trace" in record for record in records)
    with pytest.raises(ValueError, match="arm/step"):
        probe.generate_records(
            model, tokenizer, examples, prompts, arm="lr-5e-4", step=2, on_record=records.append
        )


def test_model_only_checkpoint_is_exclusive_and_never_claims_acceptance(probe, tmp_path):
    state = copy.deepcopy(build_tiny_qwen3(136).state_dict())
    inputs = dict(job_id="fixture-job", run_id="fixture-run")
    record = probe.checkpoint_record(tmp_path, state, arm="lr-5e-4", lr=5e-4, inputs=inputs)
    payload = torch.load(tmp_path / record["path"], weights_only=True)
    assert record["scope"] == payload["scope"] == "diagnostic_model_only"
    assert all(payload[key] is False for key in probe.FLAGS)
    assert "optimizer" not in payload and "scheduler" not in payload
    assert probe.file_sha(tmp_path / record["path"]) == record["sha256"]
    assert all(torch.equal(value, payload["model"][key]) for key, value in state.items())
    with pytest.raises(ValueError, match="already exists"):
        probe.checkpoint_record(tmp_path, state, arm="lr-5e-4", lr=5e-4, inputs=inputs)


@pytest.mark.parametrize("progress", [False, True])
def test_node_cli_failure_preserves_partial_and_all_false_flags(probe, tmp_path, monkeypatch, progress):
    node = module("sdsc_student_lr_job")
    source, science = tmp_path / "source", tmp_path / "science"
    output, inputs = tmp_path / "artifacts", tmp_path / "inputs.json"
    output.mkdir()
    inputs.write_text("{}")
    if progress:
        probe.atomic(
            output / "lr-probe.json",
            dict(
                schema=probe.SCHEMA,
                raw_record_count=7,
                arms=[dict(generation=[dict(step=0, record_count=7)])],
                **probe.FALSE,
            ),
        )
        probe.append_row(output / "lr-records.jsonl", {"partial_fixture": True})
    monkeypatch.setenv("RANK", "0")
    argv = node.worker_argv({"python": "python-fixture"}, source, science, inputs, output)
    start = argv.index(str(source / "tools/sdsc_student_lr_probe.py")) + 1
    assert probe.main(argv[start:]) == 1
    report = json.loads((output / "lr-probe.json").read_text())
    assert report["passed"] is report["diagnostic_complete"] is False
    assert "input identity" in report["error"]
    assert all(report[key] is False for key in probe.FLAGS)
    if progress:
        assert report["raw_record_count"] == 7
        assert report["arms"][0]["generation"][0]["record_count"] == 7
        assert len(report["raw_artifacts"]) == 1
    else:
        assert report["raw_artifacts"] == []


def test_local_failure_is_bounded_and_does_not_enter_error_collective(probe):
    def fail():
        raise RuntimeError("local fixture")

    with pytest.raises(ValueError, match="local fixture"):
        probe.local_phase(fail, rank=0, world_size=1)


def audit_fixture(probe, work):
    rows = [
        dict(
            global_slot=index,
            prompt_id=f"synthetic-{index}",
            attempt_id=f"synthetic-{index}:candidate-0000",
            input_ids=[5] * 835,
            response_ids=[7] * (92 if index < 8 else 91) + [151645],
            response_token_mask=[True] * (93 if index < 8 else 92),
        )
        for index in range(64)
    ]
    capture = work / "capture.json"
    probe.atomic(capture, dict(schema="quest-sdsc-student-lr-first64-capture-v1", rows=rows))
    report = work / "audit.json"
    value = dict(
        schema="quest-sdsc-student-lr-data-audit-v1",
        passed=True,
        diagnostic_complete=True,
        parent_job_id="54548846",
        initial_checkpoint_sha256=probe.INITIAL_SHA,
        teacher_store_manifest_sha256=probe.STORE_FILE_SHA,
        teacher_store_sha256=probe.STORE_SHA,
        teacher_attempts_sha256=probe.ATTEMPTS_SHA,
        tokenizer_fingerprint=probe.TOKENIZER_SHA,
        baseline_instruction_sha256=probe.INSTRUCTION_SHA,
        capture_sha256=probe.file_sha(capture),
        window_sha256=probe.digest(probe.canonical(rows)),
        ordered_prompt_ids=[row["prompt_id"] for row in rows],
        selected_attempt_ids=[row["attempt_id"] for row in rows],
        **probe.FALSE,
    )
    probe.atomic(report, value)
    return {
        name: dict(path=str(path), size=path.stat().st_size, sha256=probe.file_sha(path))
        for name, path in (("report", report), ("capture", capture))
    }


@pytest.mark.parametrize("scientific_cwd", [True, False])
def test_node_actual_input_producer_to_fresh_worker_consumer(probe, tmp_path, scientific_cwd):
    """No copied input allowlist: execute the actual node producer and worker guard."""
    node = module("sdsc_student_lr_job")
    science, cache, output = tmp_path / "science", tmp_path / "huggingface", tmp_path / "artifacts"
    for path in (science, cache, output):
        path.mkdir()
    plan = dict(
        checkpoints=[dict(size=probe.INITIAL_SIZE)],
        dataset_inputs=[dict(path="manifest.json", sha256=probe.FAMILY_SHA)],
        run_id="fixture-run",
        code_sha256="f" * 64,
    )
    control = SimpleNamespace(
        INITIAL_SHA=probe.INITIAL_SHA, PARENT_JOB="54548846", sha=probe.digest, canonical=probe.canonical
    )
    data_audit = audit_fixture(probe, tmp_path)
    inputs = node.build_worker_inputs(
        plan,
        tmp_path,
        science,
        data_audit,
        {"job_id": "123456"},
        dict(size=7923, sha256=probe.CONFIG_SHA),
        control,
    )
    target = tmp_path / "inputs.json"
    probe.atomic(target, inputs)
    script = """
import importlib.util,json,os,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('worker',sys.argv[1])
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
inputs=worker.document(Path(sys.argv[2]))
os.environ['SLURM_JOB_ID']=inputs['job_id'];os.environ['HF_HOME']=inputs['hf_home']
work=worker.staged_science(inputs,Path(sys.argv[3]))
report,rows=worker.validate_data_audit(inputs,work)
assert len(rows)==64
assert report['window_sha256']==worker.digest(worker.canonical(rows))
print('producer_consumer_fixture_passed')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            str(ROOT / "tools/sdsc_student_lr_probe.py"),
            str(target),
            str(output),
        ],
        cwd=science if scientific_cwd else tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if scientific_cwd:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "producer_consumer_fixture_passed"
    else:
        assert result.returncode != 0
        assert "worker cwd must equal verified science root" in result.stderr
        assert "producer_consumer_fixture_passed" not in result.stdout


@pytest.mark.parametrize(
    "defect",
    [
        "report_sha",
        "size_bool",
        "accepted",
        "eos",
        "mask_type",
        "token_type",
        "extra_row_field",
        "semantic_hash",
    ],
)
def test_audit_admission_rejects_tampered_and_type_confused_capture(probe, tmp_path, defect):
    inputs = {"data_audit": audit_fixture(probe, tmp_path)}
    report_path, capture_path = (Path(inputs["data_audit"][name]["path"]) for name in ("report", "capture"))
    report, capture = probe.document(report_path), probe.document(capture_path)
    if defect == "report_sha":
        inputs["data_audit"]["report"]["sha256"] = "0" * 64
    elif defect == "size_bool":
        inputs["data_audit"]["capture"]["size"] = True
    else:
        if defect == "accepted":
            report["student_accepted"] = True
        elif defect == "semantic_hash":
            report["window_sha256"] = "0" * 64
        else:
            row = capture["rows"][0]
            if defect == "eos":
                row["response_ids"][-1] = 5
            elif defect == "mask_type":
                row["response_token_mask"][0] = 1
            elif defect == "token_type":
                row["input_ids"][0] = True
            else:
                row["unexpected"] = "fixture"
            report["window_sha256"] = probe.digest(probe.canonical(capture["rows"]))
            probe.atomic(capture_path, capture)
            report["capture_sha256"] = probe.file_sha(capture_path)
        probe.atomic(report_path, report)
        for name, path in (("report", report_path), ("capture", capture_path)):
            inputs["data_audit"][name].update(size=path.stat().st_size, sha256=probe.file_sha(path))
    with pytest.raises(ValueError):
        probe.validate_data_audit(inputs, tmp_path)


def test_accelerator_new_arm_resets_eight_call_accumulation_and_rng(probe):
    """Real Accelerate CPU lifecycle; this deliberately makes no W2/FSDP claim."""
    from accelerate import Accelerator

    from posttrain_circuits.core.seeding import seed_everything

    arm_states, all_boundaries = [], []
    for lr in (5e-4, 5e-5):
        seed_everything(42)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0)
        accelerator = Accelerator(
            cpu=True, gradient_accumulation_steps=8, step_scheduler_with_optimizer=False
        )
        prepared, wrapped = accelerator.prepare(model, optimizer)
        arm_states.append(RNGState.capture().as_dict())
        baseline = [value.detach().clone() for value in model.parameters()]
        observer = probe.AdamWObserver(optimizer, lr, lambda baseline=baseline: baseline)
        boundaries = []
        for _ in range(8):
            with accelerator.accumulate(prepared):
                loss = prepared(torch.ones(1, 2)).square().mean()
                accelerator.backward(loss)
                wrapped.step()
                wrapped.zero_grad()
                boundaries.append((accelerator.sync_gradients, observer.calls))
        all_boundaries.append(boundaries)
        observer.close()
        accelerator.free_memory()
        assert accelerator.step == 0
    assert arm_states[0] == arm_states[1]
    assert all_boundaries == [[(False, 0)] * 7 + [(True, 1)]] * 2


def test_local_error_exchange_includes_both_ranks_without_retry(probe, monkeypatch):
    calls = []

    def exchange(output, local):
        calls.append(local)
        output[:] = [local, {"rank": 1, "type": "ValueError", "message": "peer load failed"}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", exchange)
    with pytest.raises(ValueError, match="peer load failed"):
        probe.local_phase(lambda: "local success", rank=0, world_size=2)
    assert calls == [None]


def test_bounded_progress_is_small_and_does_not_claim_success(probe, tmp_path):
    probe.progress_marker(tmp_path, rank=1, phase="must_not_publish", arm="lr-5e-4", step=1)
    assert not (tmp_path / "progress.json").exists()
    probe.progress_marker(
        tmp_path,
        rank=0,
        phase="update_nll_complete",
        arm="lr-5e-5",
        step=4,
        teacher_before=1.2,
        teacher_after=1.3,
    )
    report = probe.document(tmp_path / "progress.json")
    assert report["phase"] == "update_nll_complete" and report["step"] == 4
    assert all(report[key] is False for key in probe.FLAGS)
    assert report["passed"] is report["diagnostic_complete"] is False
    assert (tmp_path / "progress.json").stat().st_size < 1024


def restore_genuine_parent_for_cpu_fixture(destination):
    """Restore actual accepted history locally; no invented commits or resolver mocks."""
    parent = "6c04f804b302184b8ff95d00fab404e0531ed8d6"
    metadata = ROOT / (".git" if (ROOT / ".git" / "HEAD").is_file() else ".opd-git")
    if not metadata.is_dir():
        pytest.skip("genuine accepted-history integration requires a repository clone")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_NO_LAZY_FETCH="1",
        GIT_OPTIONAL_LOCKS="0",
    )

    def git(*args, data=None):
        return subprocess.run(
            ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", *args],
            input=data,
            capture_output=True,
            env=environment,
            timeout=30,
            check=True,
        ).stdout

    # Every write is within this new pytest checkout; the user's refs/index stay untouched.
    git("init", "--quiet", str(destination))
    git("-C", str(destination), "fetch", "--quiet", "--no-tags", str(metadata), parent)
    git("-C", str(destination), "update-ref", "HEAD", parent)
    git("-C", str(destination), "read-tree", parent)
    listing = git("-C", str(destination), "ls-tree", "-r", "-z", parent)
    entries = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        fields, name = entry.split(b"\t", 1)
        mode, kind, object_id = fields.split()
        assert mode in (b"100644", b"100755") and kind == b"blob"
        relative = Path(name.decode())
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative.parts[0] not in (".git", ".opd-git")
        entries.append((relative, mode, object_id))
    payload = git(
        "-C", str(destination), "cat-file", "--batch", data=b"".join(row[2] + b"\n" for row in entries)
    )
    offset = 0
    for relative, mode, object_id in entries:
        end = payload.index(b"\n", offset)
        observed_id, kind, size_text = payload[offset:end].split()
        size = int(size_text)
        assert observed_id == object_id and kind == b"blob"
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload[end + 1 : end + 1 + size])
        path.chmod(0o755 if mode == b"100755" else 0o644)
        offset = end + size + 2
    assert offset == len(payload)
    return parent


def test_genuine_student_metadata_and_teacher_reader_use_cwd(tmp_path, monkeypatch):
    """Actual accepted history and actual reader, without stubbing their call chain."""
    from posttrain_circuits.artifacts import adapted_student_protocol as protocol
    from posttrain_circuits.artifacts.adapted_teacher_sft import read_accepted_teacher_sft
    from posttrain_circuits.artifacts.teacher_adaptation_protocol import TeacherAdaptationProtocolError
    from posttrain_circuits.core.config import compose_config

    science = tmp_path / "science"
    parent = restore_genuine_parent_for_cpu_fixture(science)
    source = tmp_path / "source"
    source.mkdir()
    empty_teacher = tmp_path / "teacher"
    empty_teacher.mkdir()
    config = compose_config(
        [
            "g0=qwen3_v2_eap_separation",
            "experiment=canonical_sft",
            "adapted_teacher=qwen3_accepted_student_v6",
            f"protocol_amendment_path={protocol.PROTOCOL_PATH}",
        ],
        config_root=science / "configs",
    )
    # Match the real launch defect: module bytes/teacher path do not select the Git root.
    monkeypatch.chdir(source)
    with pytest.raises(protocol.AdaptedStudentProtocolError, match="needs real .git or .opd-git"):
        protocol.validate_student_protocol(config)
    with pytest.raises(protocol.AdaptedStudentProtocolError, match="needs real .git or .opd-git"):
        read_accepted_teacher_sft(empty_teacher, config=config)

    monkeypatch.chdir(science)
    binding = protocol.validate_student_protocol(config)
    assert binding.head == binding.git_commit == parent
    assert len(binding.science_file_sha256) == 49
    assert binding.reviewed_implementation_commit == "9a196677a269c2e5f1c925da4e43b6145c1cdd7a"
    # The genuine protocol passes; an absent teacher store still fails its next real gate.
    with pytest.raises(ValueError, match="teacher input bundle has missing or extra files"):
        read_accepted_teacher_sft(empty_teacher, config=config)

    # Correct cwd never bypasses immutable source validation.
    path = science / "src/posttrain_circuits/datasets/proofgraph/rendering.py"
    path.write_bytes(path.read_bytes() + b"\n# deliberate fixture tamper\n")
    with pytest.raises(
        TeacherAdaptationProtocolError, match="named scientific implementation changed after review"
    ):
        read_accepted_teacher_sft(empty_teacher, config=config)
