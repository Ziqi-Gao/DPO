"""Diagnostic worker: actual CPU math/FP32 reload and explicit hardware fixtures."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def worker():
    spec = importlib.util.spec_from_file_location(
        "_test_order_worker", ROOT / "tools/sdsc_student_order_probe_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records(count=128):
    from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord

    result = []
    for index in range(count):
        response = [5 + index % 7] * (1 + index % 3) + [3]
        row = TrajectoryRecord(
            trajectory_id="",
            prompt_id=f"example-{index}",
            split="cpu_fixture",
            prompt_text="fixture",
            input_ids=[2, 4] + [8] * (index % 2),
            response_ids=response,
            response_text="fixture",
            response_token_mask=[True] * len(response),
            behavior_policy_id="symbolic-canonical-proof",
            behavior_policy_revision="v1",
            policy_version=0,
            sampling_request_seed=0,
            actual_sampling_seed=0,
            sampling_cursor_id=str(index),
            sampling_protocol_id="deterministic-canonical-no-sampling-v1",
            sampling_temperature=0.0,
            top_p=1.0,
            behavior_logprobs=[float("nan")] * len(response),
            verifier_reward=1.0,
            verification_trace={"reward": 1.0},
        )
        row.trajectory_id = row.expected_trajectory_id
        row.validate()
        result.append(row)
    return result


def test_source_is_symbolic_one_pass_and_rejects_changed_resume():
    from posttrain_circuits.learning.contracts import PromptBatch

    m = worker()
    source = m.CanonicalSource(records(3))
    source.get_batch(None, PromptBatch(["example-1"], ["fixture"]), 0)
    assert source.state_dict()["kind"] == "symbolic_canonical"
    assert source.records["example-1"].teacher_id is None
    snapshot = source.state_dict()
    source.load_state_dict(snapshot)
    with pytest.raises(ValueError, match="repeated"):
        source.get_batch(None, PromptBatch(["example-1"], ["fixture"]), 0)
    for bad in (
        {**snapshot, "identity_sha256": "0" * 64},
        {**snapshot, "cursor": {"example-0": True}},
        {**snapshot, "cursor": {"missing": 1}},
        {**snapshot, "kind": "teacher_demo"},
    ):
        with pytest.raises(ValueError):
            source.load_state_dict(bad)


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_real_canonical_encoding_has_no_teacher_and_masks_only_response(arm):
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.experiments.protocols import student_order_probe as p
    from posttrain_circuits.learning.contracts import TrajectoryBatch
    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
    from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer

    m = worker()
    tokenizer = build_tiny_tokenizer()
    views = p.fit_views(list(ProofGraphTask().generate_pair(123, p.DIFFICULTY)), arm=arm)
    encoded = [p.encode_view(view, tokenizer, {}) for view in views]
    converted = m.canonical_records(views, encoded, tokenizer, {})
    assert len(converted) == 16 and len({row.prompt_id for row in converted}) == 16
    assert {view.view for view in views} == set(p.FIT_VIEW_NAMES)
    row = converted[0]
    assert row.teacher_id is None and row.teacher_topk_ids == []
    assert row.verifier_reward == 1.0 and row.response_ids[-1] == tokenizer.eos_token_id
    assert row.input_ids + row.response_ids == encoded[0]["input_ids"]
    batch = VerifiedReplaySupervisor(0, retry_limit=0).prepare_targets(
        TrajectoryBatch(converted, 0), None, None
    )
    for index, record in enumerate(converted):
        prefix, end = len(record.input_ids), len(record.input_ids) + len(record.response_ids)
        assert not batch.response_mask[index, :prefix].any()
        assert batch.response_mask[index, prefix:end].all()
        assert not batch.response_mask[index, end:].any()


def test_saved_dense_is_real_reload_and_never_formal_initial(tmp_path):
    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3

    m = worker()
    torch.set_num_threads(1)
    model = build_tiny_qwen3(7)
    state = model.state_dict()
    path = tmp_path / "step.pt"
    m.save_dense(
        state,
        path,
        step=4,
        inputs={"protocol": {"sha256": "a" * 64}, "arm": "control", "mode": "probe"},
        dataset_sha="b" * 64,
        protocol_sha="c" * 64,
    )
    saved = torch.load(path, weights_only=False)
    reloaded = build_tiny_qwen3(8)
    assert m.util.strict_load_export(reloaded, saved["model"])["all_tensors_exact"]
    assert m.tree_hash(reloaded.state_dict()) == m.tree_hash(state)
    assert not saved["student_accepted"] and not saved["formal_initial_accepted"]
    assert saved["format"] == "student_order_probe_dense_v1"
    assert saved["scope"] == "student_order_probe_diagnostic_model_only"
    with pytest.raises(ValueError, match="already exists"):
        m.save_dense(
            state,
            path,
            step=4,
            inputs={"protocol": {"sha256": "a" * 64}, "arm": "control", "mode": "probe"},
            dataset_sha="b" * 64,
            protocol_sha="c" * 64,
        )


def test_step_observer_detects_extra_or_stale_optimizer_cadence():
    m = worker()
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    observer = m.StepObserver(optimizer)
    parameter.sum().backward()
    optimizer.step()
    assert observer.calls == 1
    optimizer.state[parameter]["step"].zero_()
    parameter.sum().backward()
    with pytest.raises(ValueError, match="cadence"):
        optimizer.step()
    observer.close()


def test_failure_preserves_partial_raw_without_claiming_success(tmp_path, monkeypatch):
    m = worker()
    path, output = tmp_path / "inputs.json", tmp_path / "output"
    path.write_text("{}")
    monkeypatch.setenv("RANK", "0")
    assert m.main(["--inputs-json", str(path), "--output-dir", str(output)]) == 1
    report = json.loads((output / "prepare-report.json").read_text())
    assert not report["passed"] and not report["student_accepted"] and not report["g0_passed"]
    assert "input identity" in report["error"]


def test_genuine_native_bf16_long_context_preflight_has_no_development_data():
    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3

    m = worker()
    torch.set_num_threads(1)
    model = build_tiny_qwen3(12).to(dtype=torch.bfloat16).eval()
    model.config.max_position_embeddings = 2454
    before = m.tree_hash(model.state_dict())
    actual = m.inference_envelope_probe(model, enabled=True)
    assert actual["passed"] and actual["input_length"] == 2454 and actual["all_logits_finite"]
    assert m.tree_hash(model.state_dict()) == before
    assert m.inference_envelope_probe(model, enabled=False) is None
    model.config.max_position_embeddings = 2244
    with pytest.raises(ValueError, match="context shorter"):
        m.inference_envelope_probe(model, enabled=True)
    model.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="native BF16"):
        m.inference_envelope_probe(model, enabled=True)


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_execute_all_incorrect_still_completes_diagnostic_without_selection(tmp_path, monkeypatch, arm):
    """Real orchestration/reporting; CUDA, loading and optimization are explicit spies."""
    from types import SimpleNamespace

    import accelerate

    from posttrain_circuits.experiments.protocols import student_order_probe as p

    m = worker()
    bases = p.make_examples(p.DIFFICULTY)
    config = {"model": {}, "task": p.DIFFICULTY}
    settings = m.specification()
    binding = SimpleNamespace(
        payload=settings,
        protocol_sha256="a" * 64,
        artifact_sha256="b" * 64,
        implementation_commit="c" * 40,
    )
    monkeypatch.setattr(
        m, "load_data", lambda *args: (config, tmp_path / "native.pt", bases, {}, {}, binding)
    )
    monkeypatch.setattr(torch, "__version__", "2.8.0+cu128")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda rank: "H100")
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: None)
    monkeypatch.setattr(
        m.importlib.metadata,
        "version",
        lambda name: {
            "transformers": "4.56.2",
            "accelerate": "1.10.1",
            "tokenizers": "0.22.0",
        }[name],
    )
    monkeypatch.setattr(
        accelerate, "Accelerator", lambda **kwargs: SimpleNamespace(num_processes=2, mixed_precision="bf16")
    )
    for key, value in dict(
        RANK="0",
        WORLD_SIZE="2",
        LOCAL_RANK="0",
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    ).items():
        monkeypatch.setenv(key, value)
    loaded = SimpleNamespace(tokenizer=SimpleNamespace())
    monkeypatch.setattr(m.util, "load_training_initial", lambda *args: (loaded, {"fixture_only": True}))
    monkeypatch.setattr(m.util, "local_phase", lambda fn, **kwargs: fn())
    monkeypatch.setattr(m.util, "gather", lambda value: [value, copy.deepcopy(value)])
    monkeypatch.setattr(p, "encode_view", lambda *args, **kwargs: {"input_ids": [1, 2], "prefix_length": 1})
    monkeypatch.setattr(p, "validate_encoded_fit", lambda rows: {"fixture_only": True, "rows": len(rows)})
    monkeypatch.setattr(
        m,
        "canonical_records",
        lambda views, *args: [SimpleNamespace(prompt_id=x.example.example_id) for x in views],
    )
    trainer = SimpleNamespace(_terminate=False, token_budget=SimpleNamespace(consumed=0))
    seen = dict(updates=[], checkpoints=[], restores=[], max_steps=[])

    def build(*args, max_steps, **kwargs):
        seen["max_steps"].append(max_steps)
        return trainer, object(), {"fixture_only": True}

    def train(_trainer, ids, rows, rank, step, *, observer):
        assert len(ids) == len(rows) == 2048 and rank == 0
        seen["updates"].append(step)
        trainer.token_budget.consumed += 128
        return {"step": step, "fixture_only": True}

    def restore(_trainer, output, rank):
        seen["restores"].append(seen["updates"][-1])
        return {"passed": True, "fixture_only": True}

    def checkpoint(_trainer, model_config, tokenizer, views, encoded, **kwargs):
        step, evaluate = kwargs["step"], kwargs["evaluate"]
        seen["checkpoints"].append((step, evaluate, len(views)))
        digest = f"{step:064x}"
        summary = None
        if evaluate:
            count = dict(num_examples=128, answer_correct=0, proof_correct=0, format_valid=0)
            structure = {
                name: dict(
                    count, num_examples=sum(x.metadata["structure"] == name for x in bases["student_dev"])
                )
                for name in p.STRUCTURES
            }
            summary = dict(
                step=step,
                checkpoint_sha256=digest,
                examples_sha256=p.EXAMPLES_SHA256["student_dev"],
                **count,
                per_view_counts={name: copy.deepcopy(count) for name in p.DEV_VIEW_NAMES},
                structure_counts=structure,
            )
        return {"step": step, "sha256": digest}, summary

    monkeypatch.setattr(m, "build_trainer", build)
    monkeypatch.setattr(m, "train_window", train)
    monkeypatch.setattr(m, "full_state_restore", restore)
    monkeypatch.setattr(m, "checkpoint_and_development", checkpoint)
    monkeypatch.setattr(m, "StepObserver", lambda optimizer: SimpleNamespace(close=lambda: None))
    inputs = dict(
        mode="probe",
        arm=arm,
        job_id="123",
        run_id="fixture",
        source_code_sha256="d" * 64,
        plan_sha256="e" * 64,
    )
    report = m.execute(inputs, tmp_path, tmp_path)
    expected_steps = 12
    schedule = list(m.STEPS)
    assert seen["updates"] == list(range(1, expected_steps + 1))
    assert seen["max_steps"] == [expected_steps] and seen["restores"] == [4]
    assert seen["checkpoints"] == [(step, True, 768) for step in schedule]
    assert report["optimizer_steps"] == expected_steps and report["execution_complete"] is True
    assert report["passed"] is report["diagnostic_complete"] is True
    assert report["preparation_complete"] is False and report["selected_checkpoint"] is None
    assert all(report[key] is False for key in m.FLAGS)
    assert report["arm"] == arm and report["consumed_fit_training_views"] == 768
    assert len(report["development"]) == 5
    assert report["fit_base_examples"] == 256 and report["fit_training_views"] == 2048
    saved = json.loads((tmp_path / "prepare-report.json").read_text())
    assert (
        saved == report
        and len((tmp_path / "prepare-updates.jsonl").read_text().splitlines()) == expected_steps
    )
    assert len((tmp_path / "prepare-dev-prompts.jsonl").read_text().splitlines()) == 768


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_actual_dense_reload_and_raw_generation_paths(tmp_path, monkeypatch, arm):
    evaluate = True
    """Real tiny-Qwen forward/export; explicit mock boundaries are CUDA/FSDP/generation."""
    from contextlib import nullcontext
    from types import MethodType, SimpleNamespace

    from posttrain_circuits.experiments.protocols import student_order_probe as p
    from posttrain_circuits.models import loading
    from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer

    m = worker()
    torch.set_num_threads(1)
    tokenizer = build_tiny_tokenizer()
    bases = p.make_examples(p.DIFFICULTY)
    views = p.development_views(bases["student_dev"])
    encoded = [p.encode_view(view, tokenizer, {}, training=not evaluate) for view in views]
    original = exact_shape_tiny_qwen(71)
    masters = copy.deepcopy(original.state_dict())
    original.config.max_position_embeddings = 2454
    original.to(dtype=torch.bfloat16).eval()
    trainer = SimpleNamespace(model=original, _full_model_state_for_checkpoint=lambda: copy.deepcopy(masters))

    def loader(config, for_training=False):
        del config, for_training
        model = exact_shape_tiny_qwen(72)
        model.config.max_position_embeddings = 2454

        def generate(self, *, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 256 and not kwargs["do_sample"] and not kwargs["use_cache"]
            return torch.cat((input_ids, torch.full((1, 1), tokenizer.eos_token_id, dtype=torch.long)), dim=1)

        model.generate = MethodType(generate, model)
        return SimpleNamespace(model=model, tokenizer=tokenizer)

    monkeypatch.setattr(loading, "load_model_and_tokenizer", loader)
    monkeypatch.setattr(m.util, "measurement", lambda *args: nullcontext())
    monkeypatch.setattr(m.util, "local_phase", lambda fn, **kwargs: fn())

    def gather(actual):
        other = copy.deepcopy(actual)
        other["rank"] = 1
        other["counts"] = m.empty_counts()
        other["per_view_counts"] = {name: m.empty_counts() for name in p.DEV_VIEW_NAMES}
        other["structure_counts"] = {name: m.empty_counts() for name in p.STRUCTURES}
        for ordinal in m.generation_ordinals(len(views), 1, evaluate=evaluate):
            other["counts"]["num_examples"] += 1
            if evaluate:
                view = views[ordinal]
                other["per_view_counts"][view.view]["num_examples"] += 1
                if view.view == "identity":
                    other["structure_counts"][view.example.metadata["structure"]]["num_examples"] += 1
        return [actual, other]

    monkeypatch.setattr(m.util, "gather", gather)
    checkpoint, summary = m.checkpoint_and_development(
        trainer,
        {},
        tokenizer,
        views,
        encoded,
        output=tmp_path,
        rank=0,
        step=4,
        inputs={"protocol": {"sha256": "a" * 64}, "arm": arm, "mode": "probe"},
        dataset_sha="b" * 64,
        protocol_sha="c" * 64,
        dev_sha=p.EXAMPLES_SHA256["student_dev"],
        evaluate=evaluate,
    )
    assert checkpoint["reload_by_rank"][0]["exact_saved_master_reload"]["all_tensors_exact"]
    assert all(row["bitwise_equal"] for row in checkpoint["reload_by_rank"][0]["root_export_logits"])
    path = tmp_path / (
        "prepare-dev-records-step-00000004-rank-0.jsonl"
        if evaluate
        else "prepare-preflight-records-rank-0.jsonl"
    )
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(raw) == 384
    assert all(row["arm"] == arm and row["cohort"] == "student_order_probe_dev" for row in raw)
    assert [row["ordinal"] for row in raw] == m.generation_ordinals(len(views), 0, evaluate=evaluate)
    assert all(row["view"] == views[row["ordinal"]].view for row in raw)
    assert all(row["source_example_id"] == views[row["ordinal"]].source_example_id for row in raw)
    assert all(row["checkpoint_sha256"] == checkpoint["sha256"] for row in raw)
    if evaluate:
        assert summary["num_examples"] == 128 and summary["answer_correct"] == 0
        assert all(row["num_examples"] == 128 for row in summary["per_view_counts"].values())
        assert checkpoint["reload_by_rank"][0]["inference_envelope_probe"]["input_length"] == 2454
    else:
        assert summary is None
        assert checkpoint["reload_by_rank"][0]["inference_envelope_probe"]["input_length"] == 2454


def run_distributed_child(output):
    """Real two-process Gloo: equal32 sequence mean, restore then same next update."""
    from posttrain_circuits.core.seeding import seed_everything
    from posttrain_circuits.learning.collation import collate_trajectories
    from posttrain_circuits.learning.contracts import TrajectoryBatch
    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
    from posttrain_circuits.learning.training.factorial_trainer import FactorialTrainer, TrainerConfig
    from posttrain_circuits.learning.training.optimizer import build_adamw
    from posttrain_circuits.learning.training.schedules import PromptScheduler
    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3

    m = worker()
    torch.set_num_threads(1)
    seed_everything(42)
    rows = records()
    encoded = [dict(input_ids=row.input_ids + row.response_ids) for row in rows]
    rank = int(os.environ["RANK"])
    model = build_tiny_qwen3(91)
    initial = copy.deepcopy(model.state_dict())
    optimizer = build_adamw(model.parameters(), learning_rate=m.LR, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    prompt_scheduler = PromptScheduler.for_distributed_rank(
        [row.prompt_id for row in rows], ["fixture"] * len(rows), 4, rank=rank, world_size=2
    )
    trainer = FactorialTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=prompt_scheduler,
        state_source=m.CanonicalSource(rows),
        supervisor=VerifiedReplaySupervisor(0, retry_limit=0),
        config=TrainerConfig(
            max_steps=2,
            token_budget=2_000_000,
            gradient_accumulation_steps=8,
            backend="accelerate",
            require_evaluation_metrics=False,
        ),
        run_dir=output / "training",
    )
    observer = m.StepObserver(optimizer)
    result = m.train_window(trainer, [row.prompt_id for row in rows], encoded, rank, 1, observer=observer)
    assert result["global_slots"] == list(range(rank, 64, 2))
    assert trainer.token_budget.consumed == sum(len(row["input_ids"]) for row in encoded[:64])
    # An independent all64 sequence mean must produce the same AdamW update.
    reference = build_tiny_qwen3(99)
    reference.load_state_dict(initial)
    reference_optimizer = build_adamw(reference.parameters(), learning_rate=m.LR, weight_decay=0)
    batch = collate_trajectories(TrajectoryBatch(rows[:64], 0), pad_token_id=0)
    loss = VerifiedReplaySupervisor(0, retry_limit=0).compute_loss(reference, batch).loss
    loss.backward()
    reference_optimizer.step()
    actual = trainer._accelerator.unwrap_model(trainer.model).state_dict()
    max_error = max(
        float((tensor - reference.state_dict()[name]).abs().max()) for name, tensor in actual.items()
    )
    assert max_error < 3e-7, max_error
    restored = m.full_state_restore(trainer, output, rank)
    assert restored["passed"]
    state_after = m.runtime_state(trainer)
    # Save->next update, explicit same-world reload->identical next update.
    m.train_window(trainer, [row.prompt_id for row in rows], encoded, rank, 2, observer=observer)
    expected_model = m.tree_hash(list(trainer.model.parameters()))
    expected_optimizer = m.tree_hash(trainer.optimizer.state_dict())
    expected_runtime = m.runtime_state(trainer)
    trainer._accelerator.load_state(str(output / "resume" / "step-00000001"))
    m.restore_runtime(trainer, state_after)
    observer.calls = 1
    m.train_window(trainer, [row.prompt_id for row in rows], encoded, rank, 2, observer=observer)
    assert m.tree_hash(list(trainer.model.parameters())) == expected_model
    assert m.tree_hash(trainer.optimizer.state_dict()) == expected_optimizer
    assert m.runtime_state(trainer) == expected_runtime
    m.atomic(
        output / f"rank-{rank}.json",
        dict(passed=True, max_global64_update_error=max_error, next_update_equal=True, restore=restored),
    )
    torch.distributed.destroy_process_group()


def test_actual_two_rank_global64_loss_and_full_state_next_update(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        CUDA_VISIBLE_DEVICES="",
        ACCELERATE_USE_CPU="true",
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    children = []
    for rank in range(2):
        rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank))
        children.append(
            subprocess.Popen(
                [sys.executable, "-B", __file__, "--gloo-child", str(tmp_path)],
                env=rank_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )
    results = [child.communicate(timeout=120)[0] for child in children]
    assert [child.returncode for child in children] == [0, 0], "\n".join(results)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text())["next_update_equal"]


def exact_shape_tiny_qwen(seed):
    """311-key Qwen3 architecture with tiny widths; no hardware claim."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from posttrain_circuits.utils.tiny_model import tiny_vocabulary

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tiny_vocabulary()),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=2454,
            use_cache=False,
        )
    )
    assert len(model.state_dict()) == 311
    return model


if __name__ == "__main__" and "--gloo-child" in sys.argv:
    run_distributed_child(Path(sys.argv[-1]))


@pytest.mark.parametrize(
    "schema,mode,arm,extra",
    [
        ("quest-sdsc-student-order-inputs-v4", "fit", "control", False),
        ("quest-sdsc-student-order-probe-inputs-v1", "fit", "control", False),
        ("quest-sdsc-student-order-probe-inputs-v1", "probe", "other", False),
        ("quest-sdsc-student-order-probe-inputs-v1", "probe", True, False),
        ("quest-sdsc-student-order-probe-inputs-v1", "probe", "control", True),
    ],
)
def test_mode_and_arm_cannot_relabel_original_fit_or_expand_inputs(tmp_path, schema, mode, arm, extra):
    m = worker()
    inputs = dict.fromkeys(
        (
            "schema",
            "mode",
            "arm",
            "job_id",
            "run_id",
            "source_code_sha256",
            "plan_sha256",
            "science_root",
            "original_science_files",
            "dataset_root",
            "hf_home",
            "protocol",
            "original_config",
            "initial_checkpoint",
        )
    )
    inputs.update(schema=schema, mode=mode, arm=arm)
    if extra:
        inputs["learning_rate_override"] = 1e-4
    with pytest.raises(ValueError, match="input identity"):
        m.staged_science(inputs, tmp_path / "artifacts")


def test_complete_panel_rank_shards_and_fixed_diagnostic_schedule():
    m = worker()
    shards = [m.generation_ordinals(768, rank, evaluate=True) for rank in (0, 1)]
    assert len(shards[0]) == len(shards[1]) == 384
    assert set(shards[0]).isdisjoint(shards[1])
    assert sorted(shards[0] + shards[1]) == list(range(768))
    for rank, shard in enumerate(shards):
        assert all((ordinal // 6) % 2 == rank for ordinal in shard)
        assert [sum(ordinal % 6 == view for ordinal in shard) for view in range(6)] == [64] * 6
    assert m.checkpoint_steps("probe") == (4, 6, 7, 8, 12)
    for mode in ("preflight", "fit", "old_fit"):
        with pytest.raises(ValueError):
            m.checkpoint_steps(mode)
    with pytest.raises(ValueError):
        m.generation_ordinals(8, 0, evaluate=False)
    with pytest.raises(ValueError):
        m.generation_ordinals(1536, 0, evaluate=True)


@pytest.mark.parametrize(
    "mutation", ["format", "scope", "arm", "mode", "step", "parent", "accepted", "extra"]
)
def test_checkpoint_metadata_rejects_other_arm_or_preparation_identity(tmp_path, mutation):
    m = worker()
    path = tmp_path / "checkpoint.pt"
    m.save_dense(
        {"fixture": torch.ones(1)},
        path,
        step=4,
        inputs={"arm": "treatment", "protocol": {"sha256": "a" * 64}},
        dataset_sha="b" * 64,
        protocol_sha="c" * 64,
    )
    saved = torch.load(path, weights_only=False)
    kwargs = dict(
        step=4, arm="treatment", dataset_sha="b" * 64, protocol_sha="c" * 64, protocol_artifact_sha="a" * 64
    )
    m.validate_dense_payload(saved, **kwargs)
    if mutation == "format":
        saved["format"] = "student_order_dense_v4"
    elif mutation == "scope":
        saved["scope"] = "common_student_order_model_only"
    elif mutation == "arm":
        saved["arm"] = "control"
    elif mutation == "mode":
        saved["mode"] = "fit"
    elif mutation == "step":
        saved["global_step"] = True
    elif mutation == "parent":
        saved["parent_checkpoint_sha256"] = "0" * 64
    elif mutation == "accepted":
        saved["student_accepted"] = True
    else:
        saved["selected_checkpoint"] = 4
    with pytest.raises(ValueError):
        m.validate_dense_payload(saved, **kwargs)


@pytest.mark.parametrize(
    "mutation",
    [None, "missing_view", "wrong_population", "boolean", "missing_rank", "bad_structure", "count_order"],
)
def test_raw_count_reduction_is_not_a_scientific_acceptance_gate(mutation):
    from posttrain_circuits.datasets.proofgraph.anti_shortcut import TRANSFORMATIONS

    m = worker()
    ranks = []
    for rank in (0, 1):
        empty = dict(num_examples=64, answer_correct=0, proof_correct=0, format_valid=0)
        ranks.append(
            dict(
                rank=rank,
                counts=dict(empty, num_examples=384),
                per_view_counts={name: copy.deepcopy(empty) for name in ("identity", *TRANSFORMATIONS)},
                structure_counts={
                    name: dict(empty, num_examples=n)
                    for name, n in (("chain", 20), ("branch", 24), ("converging_dag", 20))
                },
            )
        )
    if mutation == "missing_view":
        ranks[0]["per_view_counts"].pop("entity_symbol_renaming")
    elif mutation == "wrong_population":
        ranks[0]["per_view_counts"]["identity"]["num_examples"] = 63
    elif mutation == "boolean":
        ranks[0]["counts"]["proof_correct"] = False
    elif mutation == "missing_rank":
        ranks.pop()
    elif mutation == "bad_structure":
        ranks[0]["structure_counts"]["chain"]["num_examples"] = 19
    elif mutation == "count_order":
        ranks[0]["counts"]["proof_correct"] = 1
    if mutation:
        with pytest.raises(ValueError):
            m.aggregate_development(ranks, step=7, checkpoint_sha="a" * 64, dev_sha="b" * 64)
    else:
        value = m.aggregate_development(ranks, step=7, checkpoint_sha="a" * 64, dev_sha="b" * 64)
        assert value["num_examples"] == 128 and value["proof_correct"] == 0
        assert not {"passed", "eligible", "selected_checkpoint", "qualification_passed"} & value.keys()


def test_partial_generation_failure_keeps_raw_evidence_and_false_completion(tmp_path, monkeypatch):
    m = worker()
    inputs = tmp_path / "inputs.json"
    inputs.write_text("{}")
    output = tmp_path / "artifacts"
    output.mkdir()
    raw = output / "prepare-dev-records-step-00000004-rank-0.jsonl"
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(m, "staged_science", lambda *args: tmp_path)

    def broken(*args):
        raw.write_text('{"partial":true}\n')
        raise RuntimeError("explicit generation failure")

    monkeypatch.setattr(m, "execute", broken)
    assert m.main(["--inputs-json", str(inputs), "--output-dir", str(output)]) == 1
    report = json.loads((output / "prepare-report.json").read_text())
    assert report["diagnostic_complete"] is report["execution_complete"] is report["passed"] is False
    assert report["selected_checkpoint"] is None and all(report[key] is False for key in m.FLAGS)
    assert raw.read_text() == '{"partial":true}\n'
    assert report["raw_artifacts"] == [m.artifact(raw, output)]


def audit_fixture_module():
    spec = importlib.util.spec_from_file_location(
        "_order_probe_independent_audit_fixtures",
        Path(__file__).with_name("test_student_order_probe_audit.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_actual_pinned_tokenizer_all_fit_targets_and_loss_masks(arm):
    from posttrain_circuits.experiments.protocols import student_order_probe as p
    from posttrain_circuits.learning.contracts import TrajectoryBatch
    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor

    if not (ROOT / ".sdsc/diagnostics/qwen3-tokenizer-b968826d.json").exists():
        pytest.skip("bounded pinned tokenizer evidence is not present")
    m = worker()
    tokenizer = audit_fixture_module().audit.load_tokenizer()
    config = {
        "prompt_protocol": dict(
            name="qwen3_non_thinking_v1", enable_thinking=False, chat_template_sha256=p.CHAT_TEMPLATE_SHA256
        )
    }
    views = p.fit_views(p.make_examples(p.DIFFICULTY)["student_fit"], arm=arm)
    rows = [p.encode_view(view, tokenizer, config, training=True) for view in views]
    token_audit = p.validate_encoded_fit(rows)
    converted = m.canonical_records(views, rows, tokenizer, config)
    assert len(converted) == 2048
    assert token_audit["executed_training_views"] == 768
    assert len({row.prompt_id for row in converted}) == 2048
    for offset in range(0, 2048, 64):
        window = converted[offset : offset + 64]
        batch = VerifiedReplaySupervisor(0, retry_limit=0).prepare_targets(
            TrajectoryBatch(window, 0), None, None
        )
        for index, record in enumerate(window):
            encoded = rows[offset + index]
            prefix, end = len(record.input_ids), len(record.input_ids) + len(record.response_ids)
            assert record.teacher_id is None and record.teacher_topk_ids == []
            assert record.verifier_reward == 1.0
            assert record.input_ids + record.response_ids == encoded["input_ids"]
            assert record.response_ids[-1] == tokenizer.eos_token_id
            assert not batch.response_mask[index, :prefix].any()
            assert batch.response_mask[index, prefix:end].all()
            assert not batch.response_mask[index, end:].any()


def test_actual_worker_reduction_to_independent_raw_auditor():
    if not (ROOT / ".sdsc/diagnostics/qwen3-tokenizer-b968826d.json").exists():
        pytest.skip("bounded pinned tokenizer evidence is not present")
    fixtures = audit_fixture_module()
    m = worker()
    tokenizer = fixtures.audit.load_tokenizer()
    population = fixtures.audit.population(tokenizer, "probe")
    report, prompts, records, tokenizer = fixtures.make_fixture((tokenizer, population), arm="treatment")
    expected = copy.deepcopy(report["development"])
    report["development"] = []
    for checkpoint in report["checkpoints"]:
        rank_results = []
        for rank, raw in enumerate(records[checkpoint["step"]]):
            row = dict(
                rank=rank,
                counts=m.empty_counts(),
                per_view_counts={name: m.empty_counts() for name in fixtures.audit.VIEWS},
                structure_counts={name: m.empty_counts() for name in fixtures.audit.STRUCTURES},
            )
            for record in raw:
                m.add_verification(row["counts"], record["verification"])
                m.add_verification(row["per_view_counts"][record["view"]], record["verification"])
                if record["view"] == "identity":
                    structure = population[1][record["ordinal"]].metadata["structure"]
                    m.add_verification(row["structure_counts"][structure], record["verification"])
            rank_results.append(row)
        report["development"].append(
            m.aggregate_development(
                rank_results,
                step=checkpoint["step"],
                checkpoint_sha=checkpoint["sha256"],
                dev_sha=population[2],
            )
        )
    assert report["development"] == expected
    replay = fixtures.audit.replay(report, prompts, records, tokenizer)
    assert replay["responses_replayed"] == 3840 and replay["prompts_replayed"] == 768
    assert replay["arm"] == "treatment" and replay["diagnostic_complete"] is True
    assert replay["selected_checkpoint"] is None and replay["preparation_complete"] is False
    assert all(replay[key] is False for key in m.FLAGS)


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_real_node_input_shape_enters_fresh_worker_with_science_cwd(tmp_path, arm):
    from types import SimpleNamespace

    m = worker()
    spec = importlib.util.spec_from_file_location(
        "_prepare_node", ROOT / "tools/sdsc_student_order_probe_job.py"
    )
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    science, output = tmp_path / "science", tmp_path / "output"
    (science / "src").mkdir(parents=True)
    (tmp_path / "huggingface").mkdir()
    output.mkdir()
    protocol_path = "prereg/amendments/qwen3_student_order_probe_v1.json"
    (science / protocol_path).parent.mkdir(parents=True)
    (science / protocol_path).write_text('{"fixture":true}')
    frozen = {}
    for index in range(49):
        name = f"src/frozen_{index}.py"
        (science / name).write_text(f"VALUE = {index}\n")
        frozen[name] = m.file_sha(science / name)
    control = SimpleNamespace(
        read=lambda path: path.read_bytes(),
        PROTOCOL_PATH=protocol_path,
        INITIAL_SHA=m.util.INITIAL_SHA,
        sha=m.digest,
        canonical=m.canonical,
    )
    plan = dict(
        mode="probe",
        arm=arm,
        run_id="fixture",
        code_sha256="a" * 64,
        resolved_config={"size": 7923, "sha256": m.util.CONFIG_SHA},
        protocol={"protocol_sha256": "b" * 64},
    )
    inputs = node.build_worker_inputs(plan, tmp_path, science, {"job_id": "12345"}, frozen, control)
    path = tmp_path / "inputs.json"
    path.write_bytes(m.canonical(inputs))
    child = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
inputs = m.document(Path(sys.argv[2]))
work = m.staged_science(inputs, Path(sys.argv[3]))
assert work == Path(sys.argv[3]).parent
assert len(inputs) == 14 and inputs['mode'] == 'probe'
assert inputs['arm'] in ('control', 'treatment')
assert set(inputs['protocol']) == {'path', 'size', 'sha256', 'protocol_sha256'}
print('node_to_worker_admitted')
"""
    env = dict(os.environ, HF_HOME=str(tmp_path / "huggingface"), SLURM_JOB_ID="12345")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            child,
            str(ROOT / "tools/sdsc_student_order_probe_worker.py"),
            str(path),
            str(output),
        ],
        cwd=science,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "node_to_worker_admitted"
    wrong = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            child,
            str(ROOT / "tools/sdsc_student_order_probe_worker.py"),
            str(path),
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert wrong.returncode != 0 and "cwd must equal" in wrong.stderr
