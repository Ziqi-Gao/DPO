"""Preparation-worker regressions using real CPU Qwen/AdamW/Accelerate paths."""

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
        "_test_prepare_worker", ROOT / "tools/sdsc_student_prepare_worker.py"
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
    from posttrain_circuits.experiments.protocols import student_preparation as p
    from posttrain_circuits.learning.contracts import TrajectoryBatch
    from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
    from posttrain_circuits.utils.tiny_model import build_tiny_tokenizer

    m = worker()
    tokenizer = build_tiny_tokenizer()
    examples = [ProofGraphTask().generate(123, p.DIFFICULTY)]
    encoded = [p.encode_example(row, tokenizer, {}) for row in examples]
    converted = m.canonical_records(examples, encoded, tokenizer, {})
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
            max_steps=2, gradient_accumulation_steps=8, backend="accelerate", require_evaluation_metrics=False
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
    spec = importlib.util.spec_from_file_location("_prepare_node", ROOT / "tools/sdsc_student_prepare_job.py")
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    science, output = tmp_path / "science", tmp_path / "output"
    (science / "src").mkdir(parents=True)
    (tmp_path / "huggingface").mkdir()
    output.mkdir()
    protocol_path = "prereg/amendments/qwen3_student_preparation_v1.json"
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
            str(ROOT / "tools/sdsc_student_prepare_worker.py"),
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
            str(ROOT / "tools/sdsc_student_prepare_worker.py"),
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
