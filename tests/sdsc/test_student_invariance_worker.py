"""Invariance-worker regressions using real CPU Qwen/AdamW/Accelerate paths."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def worker():
    spec = importlib.util.spec_from_file_location(
        "_test_invariance_worker", ROOT / "tools/sdsc_student_invariance_worker.py"
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


def test_real_canonical_encoding_has_no_teacher_and_masks_only_response():
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.experiments.protocols import student_invariance as p
    from posttrain_circuits.learning.contracts import TrajectoryBatch
    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
    from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer

    m = worker()
    tokenizer = build_tiny_tokenizer()
    views = p.fit_views(list(ProofGraphTask().generate_pair(123, p.DIFFICULTY)))
    examples = [view.example for view in views]
    encoded = [p.encode_view(view, tokenizer, {}) for view in views]
    converted = m.canonical_records(examples, encoded, tokenizer, {})
    assert len(converted) == 8 and len({row.prompt_id for row in converted}) == 8
    assert {view.view for view in views} == set(p.FIT_VIEW_NAMES)
    row = converted[0]
    assert row.teacher_id is None and row.teacher_topk_ids == []
    assert row.verifier_reward == 1.0 and row.response_ids[-1] == tokenizer.eos_token_id
    assert row.input_ids + row.response_ids == encoded[0]["input_ids"]
    batch = VerifiedReplaySupervisor(0, retry_limit=0).prepare_targets(
        TrajectoryBatch(converted, 0), None, None
    )
    assert not batch.response_mask[0, : len(row.input_ids)].any()
    assert batch.response_mask[0, len(row.input_ids) :].all()


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
        inputs={"protocol": {"sha256": "a" * 64}},
        dataset_sha="b" * 64,
        protocol_sha="c" * 64,
    )
    saved = torch.load(path, weights_only=False)
    reloaded = build_tiny_qwen3(8)
    assert m.util.strict_load_export(reloaded, saved["model"])["all_tensors_exact"]
    assert m.tree_hash(reloaded.state_dict()) == m.tree_hash(state)
    assert not saved["student_accepted"] and not saved["formal_initial_accepted"]
    assert saved["format"] == "student_invariance_dense_v2"
    assert saved["scope"] == "common_student_invariance_model_only"
    with pytest.raises(ValueError, match="already exists"):
        m.save_dense(
            state,
            path,
            step=4,
            inputs={"protocol": {"sha256": "a" * 64}},
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


def test_development_aggregation_rejects_missing_transform_and_structure_rows():
    from posttrain_circuits.datasets.proofgraph.anti_shortcut import TRANSFORMATIONS

    m = worker()
    good = dict(num_examples=128, answer_correct=50, proof_correct=45, format_valid=120)
    structure = dict(
        chain=dict(num_examples=50, answer_correct=20, proof_correct=18, format_valid=45),
        branch=dict(num_examples=40, answer_correct=15, proof_correct=13, format_valid=40),
        converging_dag=dict(num_examples=38, answer_correct=15, proof_correct=14, format_valid=35),
    )
    locals_ = [
        dict(
            per_view_counts={name: copy.deepcopy(good) for name in ("identity", *TRANSFORMATIONS)},
            structure_counts=copy.deepcopy(structure),
        )
        for _ in range(2)
    ]
    result = m.aggregate_development(locals_, step=8, checkpoint_sha="a" * 64, dev_sha="b" * 64)
    assert result["num_examples"] == 256 and result["proof_correct"] == 90
    assert result["structure_counts"]["branch"]["num_examples"] == 80
    locals_[0]["per_view_counts"][TRANSFORMATIONS[0]]["num_examples"] -= 1
    with pytest.raises(ValueError, match="incomplete development"):
        m.aggregate_development(locals_, step=8, checkpoint_sha="a" * 64, dev_sha="b" * 64)
    locals_[0]["per_view_counts"][TRANSFORMATIONS[0]]["num_examples"] += 1
    locals_[0]["structure_counts"]["branch"]["num_examples"] -= 1
    with pytest.raises(ValueError, match="structure totals"):
        m.aggregate_development(locals_, step=8, checkpoint_sha="a" * 64, dev_sha="b" * 64)


def test_complete_development_base_shards_balance_every_view_and_keep_seed_identity():
    from collections import Counter

    m = worker()
    shards = [m.generation_ordinals(1536, rank, evaluate=True) for rank in (0, 1)]
    assert len(shards[0]) == len(shards[1]) == 768
    assert not set(shards[0]) & set(shards[1])
    assert set(shards[0]) | set(shards[1]) == set(range(1536))
    for rank, shard in enumerate(shards):
        assert shard == sorted(shard)
        assert Counter(index % 6 for index in shard) == {view: 128 for view in range(6)}
        assert all((ordinal // 6) % 2 == rank for ordinal in shard)
    assert m.generation_ordinals(8, 0, evaluate=False) == [0, 2, 4, 6]
    assert m.generation_ordinals(8, 1, evaluate=False) == [1, 3, 5, 7]
    with pytest.raises(ValueError, match="population"):
        m.generation_ordinals(1535, 0, evaluate=True)


@pytest.mark.parametrize("evaluate", [False, True])
def test_actual_dense_reload_and_raw_generation_paths(tmp_path, monkeypatch, evaluate):
    """Real tiny-Qwen forward/export; explicit mock boundaries are CUDA/FSDP/generation."""
    from contextlib import nullcontext
    from types import MethodType, SimpleNamespace

    from posttrain_circuits.experiments.protocols import student_invariance as p
    from posttrain_circuits.models import loading
    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

    m = worker()
    torch.set_num_threads(1)
    tokenizer = build_tiny_tokenizer()
    bases = p._build_bases()
    views = p.development_views(bases["student_dev"]) if evaluate else p.fit_views(bases["student_fit"][:2])
    encoded = [p.encode_view(view, tokenizer, {}, training=not evaluate) for view in views]
    original = build_tiny_qwen3(71)
    masters = copy.deepcopy(original.state_dict())
    original.config.max_position_embeddings = 2454
    original.to(dtype=torch.bfloat16).eval()
    trainer = SimpleNamespace(model=original, _full_model_state_for_checkpoint=lambda: copy.deepcopy(masters))

    def loader(config, for_training=False):
        del config, for_training
        model = build_tiny_qwen3(72)
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
        inputs={"protocol": {"sha256": "a" * 64}},
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
    assert len(raw) == (768 if evaluate else 4)
    assert [row["ordinal"] for row in raw] == m.generation_ordinals(len(views), 0, evaluate=evaluate)
    assert all(row["view"] == views[row["ordinal"]].view for row in raw)
    assert all(row["source_example_id"] == views[row["ordinal"]].source_example_id for row in raw)
    assert all(row["checkpoint_sha256"] == checkpoint["sha256"] for row in raw)
    if evaluate:
        assert summary["num_examples"] == 256 and summary["answer_correct"] == 0
        assert all(row["num_examples"] == 256 for row in summary["per_view_counts"].values())
        assert checkpoint["reload_by_rank"][0]["inference_envelope_probe"] is None
    else:
        assert summary is None
        assert checkpoint["reload_by_rank"][0]["inference_envelope_probe"]["input_length"] == 2454


@pytest.mark.parametrize("evaluate", [False, True])
def test_pinned_tokenizer_actual_worker_records_replay_independently(tmp_path, monkeypatch, evaluate):
    """Real full-prefix raw seam; tiny model and collectives are explicit CPU mocks."""
    from contextlib import nullcontext
    from types import SimpleNamespace

    from posttrain_circuits.experiments.protocols import student_invariance as p
    from posttrain_circuits.models import loading
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    spec = importlib.util.spec_from_file_location(
        "_invariance_audit_seam", ROOT / "tools/sdsc_student_invariance_audit.py"
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    tokenizer = audit.load_tokenizer()
    m = worker()
    config = {
        "prompt_protocol": {
            "name": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": p.CHAT_TEMPLATE_SHA256,
        }
    }
    bases = p._build_bases()
    views = p.development_views(bases["student_dev"]) if evaluate else p.fit_views(bases["student_fit"])[:8]
    encoded = [p.encode_view(view, tokenizer, config, training=not evaluate) for view in views]
    correct = [not evaluate or (ordinal // 6) % 3 == 0 for ordinal in range(len(views))]
    targets = {
        tuple(row["input_ids"][: row["prefix_length"]]): row["input_ids"][row["prefix_length"] :]
        if correct[ordinal]
        else [tokenizer.eos_token_id]
        for ordinal, row in enumerate(encoded)
    }

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(max_position_embeddings=2454, vocab_size=len(tokenizer))

        def forward(self, *, input_ids, **kwargs):
            del kwargs
            return SimpleNamespace(logits=self.weight.expand(1, input_ids.shape[1], 2))

        def generate(self, *, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 256
            target = targets[tuple(input_ids[0].tolist())]
            return torch.cat((input_ids, torch.tensor([target], dtype=torch.long)), dim=1)

    original = Toy()
    masters = copy.deepcopy(original.state_dict())
    original.to(dtype=torch.bfloat16).eval()
    trainer = SimpleNamespace(model=original, _full_model_state_for_checkpoint=lambda: copy.deepcopy(masters))
    monkeypatch.setattr(
        loading,
        "load_model_and_tokenizer",
        lambda *args, **kwargs: SimpleNamespace(model=Toy(), tokenizer=tokenizer),
    )
    monkeypatch.setattr(m.util, "measurement", lambda *args: nullcontext())
    monkeypatch.setattr(m.util, "local_phase", lambda fn, **kwargs: fn())

    def gather(actual):
        other = copy.deepcopy(actual)
        other["rank"] = 1 - actual["rank"]
        other["counts"] = m.empty_counts()
        other["per_view_counts"] = {name: m.empty_counts() for name in p.DEV_VIEW_NAMES}
        other["structure_counts"] = {name: m.empty_counts() for name in p.STRUCTURES}
        for ordinal in m.generation_ordinals(len(views), other["rank"], evaluate=evaluate):
            verification = dict(
                answer_correct=correct[ordinal], reward=float(correct[ordinal]), parse_valid=correct[ordinal]
            )
            m.add_verification(other["counts"], verification)
            if evaluate:
                view = views[ordinal]
                m.add_verification(other["per_view_counts"][view.view], verification)
                if view.view == "identity":
                    m.add_verification(
                        other["structure_counts"][view.example.metadata["structure"]], verification
                    )
        return [actual, other] if actual["rank"] == 0 else [other, actual]

    monkeypatch.setattr(m.util, "gather", gather)
    checkpoints, summaries, step_records = [], [], {}
    for step in m.STEPS if evaluate else (4,):
        results = []
        for rank in (0, 1):
            checkpoint, summary = m.checkpoint_and_development(
                trainer,
                config,
                tokenizer,
                views,
                encoded,
                output=tmp_path,
                rank=rank,
                step=step,
                inputs={"protocol": {"sha256": "a" * 64}},
                dataset_sha="b" * 64,
                protocol_sha="c" * 64,
                dev_sha=p.EXAMPLES_SHA256["student_dev"],
                evaluate=evaluate,
            )
            results.append(summary)
        assert results[0] == results[1]
        checkpoints.append(checkpoint)
        if summary is not None:
            summaries.append(summary)
        step_records[step] = []
        for rank in (0, 1):
            name = (
                f"prepare-dev-records-step-{step:08d}-rank-{rank}.jsonl"
                if evaluate
                else f"prepare-preflight-records-rank-{rank}.jsonl"
            )
            step_records[step].append(
                [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
            )
    prompts = (
        [
            dict(
                ordinal=ordinal,
                example=asdict(view.example),
                prompt_ids=row["input_ids"][: row["prefix_length"]],
                source_example_id=view.source_example_id,
                view=view.view,
                prompt_text=format_model_prompt(view.prompt, tokenizer, config).model_facing_prompt,
            )
            for ordinal, (view, row) in enumerate(zip(views, encoded, strict=True))
        ]
        if evaluate
        else []
    )
    selected = p.select_checkpoint(summaries) if evaluate else None
    report = dict(
        schema=m.SCHEMA,
        mode="fit" if evaluate else "preflight",
        checkpoints=checkpoints,
        development=summaries,
        selected_checkpoint=selected,
        passed=selected is not None if evaluate else True,
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
        **m.FLAGS,
    )
    result = audit.replay(report, prompts, step_records, tokenizer)
    assert result["raw_replay_passed"] and result["responses_replayed"] == (9216 if evaluate else 8)
    if evaluate:
        assert result["selected_checkpoint"]["step"] == 4
        assert all(row["num_examples"] == 256 for row in summaries[0]["per_view_counts"].values())
    else:
        assert all(row["verification"]["reward"] == 1.0 for group in step_records[4] for row in group)


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
            token_budget=8_000_000,
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


if __name__ == "__main__" and "--gloo-child" in sys.argv:
    run_distributed_child(Path(sys.argv[-1]))


def test_real_node_input_shape_enters_fresh_worker_with_science_cwd(tmp_path):
    from types import SimpleNamespace

    m = worker()
    spec = importlib.util.spec_from_file_location(
        "_prepare_node", ROOT / "tools/sdsc_student_invariance_job.py"
    )
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    science, output = tmp_path / "science", tmp_path / "output"
    (science / "src").mkdir(parents=True)
    (tmp_path / "huggingface").mkdir()
    output.mkdir()
    protocol_path = "prereg/amendments/qwen3_student_invariance_v2.json"
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
        mode="preflight",
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
assert set(inputs['protocol']) == {'path', 'size', 'sha256', 'protocol_sha256'}
print('node_to_worker_admitted')
"""
    env = dict(os.environ, HF_HOME=str(tmp_path / "huggingface"), SLURM_JOB_ID="12345")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            child,
            str(ROOT / "tools/sdsc_student_invariance_worker.py"),
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
            "-B",
            "-c",
            child,
            str(ROOT / "tools/sdsc_student_invariance_worker.py"),
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
