from __future__ import annotations

import copy
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from posttrain_circuits.artifacts.checkpoints import (
    accelerator_state_file_hashes,
    checkpoint_runtime_state_hashes,
    save_checkpoint as artifact_save_checkpoint,
)
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import publish_path_no_clobber
from posttrain_circuits.cli.finalize_pilot_training import (
    _validate_factorial_accelerator_state,
    _validate_factorial_rank_states,
)
from posttrain_circuits.cli.train import _validate_teacher_demo_model_input_lengths
from posttrain_circuits.datasets.teacher_demos.contracts import (
    LOGPROB_AVAILABLE,
    TeacherDemoAttempt,
)
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.learning.contracts import PromptBatch, SamplingRequest, TrajectoryBatch
from posttrain_circuits.learning.state_sources.current_policy import CurrentPolicyStateSource
from posttrain_circuits.learning.supervision.verified_replay import (
    InsufficientPositiveTrajectories,
    VerifiedReplaySupervisor,
)
from posttrain_circuits.learning.teacher.demo_source import TeacherDemoStateSource
from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor
from posttrain_circuits.learning.training.factorial_trainer import (
    FactorialTrainer,
    TrainerConfig,
    _allocation_neutral_global_objective_metrics,
    _allocation_neutral_global_public_metrics,
    _checkpoint_publication_preflight_report,
    _extend_resume_ancestry,
    _validate_checkpoint_publication_preflight_reports,
    _validate_optimizer_scheduler_cadence,
    _validate_rank_shard_hashes,
    _validate_rank_state_mappings,
    _validate_resume_world_size,
)
from posttrain_circuits.learning.training.schedules import (
    ExactGlobalBatchPlan,
    PromptScheduler,
)
from posttrain_circuits.utils.smoke import build_smoke_examples, make_trajectory


class SequenceStateSource:
    def __init__(self, batches: list[list[TrajectoryRecord]]) -> None:
        self.batches = batches
        self.calls = 0

    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch:
        del model, prompt_batch, step
        index = min(self.calls, len(self.batches) - 1)
        self.calls += 1
        return TrajectoryBatch(copy.deepcopy(self.batches[index]), policy_version=0)

    def refresh_if_needed(self, model: Any, step: int) -> None:
        del model, step

    def state_dict(self) -> dict[str, Any]:
        return {"calls": self.calls}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.calls = int(state["calls"])


class PromptAwareSequenceStateSource(TeacherDemoStateSource):
    """Return one deterministic verified trajectory for each requested prompt."""

    def __init__(self, record: TrajectoryRecord) -> None:
        self.record = record
        self.calls = 0

    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch:
        del model, step
        self.calls += 1
        return TrajectoryBatch(
            [copy.deepcopy(self.record) for _ in prompt_batch.prompt_ids],
            policy_version=0,
        )

    def refresh_if_needed(self, model: Any, step: int) -> None:
        del model, step

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "teacher_demo", "calls": self.calls}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.calls = int(state["calls"])


class CountingReplaySupervisor(VerifiedReplaySupervisor):
    def __init__(self, pad_token_id: int, **kwargs: Any) -> None:
        super().__init__(pad_token_id, **kwargs)
        self.loss_calls = 0

    def compute_loss(self, model: Any, batch):  # type: ignore[no-untyped-def]
        self.loss_calls += 1
        return super().compute_loss(model, batch)


class CountingCanonicalSFTSupervisor(CanonicalSFTSupervisor):
    def __init__(self, pad_token_id: int, **kwargs: Any) -> None:
        super().__init__(pad_token_id, **kwargs)
        self.loss_calls = 0

    def compute_loss(self, model: Any, batch):  # type: ignore[no-untyped-def]
        self.loss_calls += 1
        return super().compute_loss(model, batch)


class FakeAccelerator:
    def __init__(self, *, rank: int = 0, world_size: int = 1) -> None:
        self.process_index = rank
        self.num_processes = world_size
        self.load_state_calls = 0

    def load_state(self, path: str) -> None:
        del path
        self.load_state_calls += 1

    def wait_for_everyone(self) -> None:
        pass


class CheckpointingFakeAccelerator(FakeAccelerator):
    def save_state(self, path: str) -> None:
        state_dir = Path(path)
        state_dir.mkdir(exist_ok=True)
        (state_dir / "state.bin").write_bytes(b"fake-accelerator-state")

    def get_state_dict(self, model: torch.nn.Module) -> dict[str, Any]:
        return model.state_dict()


class AccumulationScopeAccelerator(FakeAccelerator):
    def __init__(self) -> None:
        super().__init__()
        self.active = False
        self.sync_gradients = True
        self.events: list[str] = []

    @contextmanager
    def accumulate(self, model):  # type: ignore[no-untyped-def]
        del model
        self.active = True
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")
            self.active = False

    def backward(self, loss: torch.Tensor) -> None:
        assert self.active
        self.events.append("backward")
        loss.backward()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        assert self.active


class InitializingFakeAccelerator(FakeAccelerator):
    instances: list["InitializingFakeAccelerator"] = []

    def __init__(
        self,
        *,
        gradient_accumulation_steps: int,
        step_scheduler_with_optimizer: bool,
    ) -> None:
        super().__init__()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step_scheduler_with_optimizer = step_scheduler_with_optimizer
        self.prepared: tuple[Any, ...] | None = None
        self.registered: list[Any] = []
        self.is_main_process = True
        type(self).instances.append(self)

    def prepare(self, *items: Any) -> tuple[Any, ...]:
        self.prepared = items
        return items

    def register_for_checkpointing(self, item: Any) -> None:
        self.registered.append(item)


class ScopeAssertingReplaySupervisor(VerifiedReplaySupervisor):
    def __init__(self, pad_token_id: int, accelerator: AccumulationScopeAccelerator) -> None:
        super().__init__(pad_token_id)
        self.accelerator = accelerator

    def compute_loss(self, model: Any, batch):  # type: ignore[no-untyped-def]
        assert self.accelerator.active
        self.accelerator.events.append("forward")
        return super().compute_loss(model, batch)


def _valid_accelerate_resume_payload(trainer: FactorialTrainer) -> dict[str, Any]:
    prompt_state = dict(trainer.prompt_scheduler.state_dict())
    source_state = dict(trainer.state_source.state_dict())
    source_state.update(rank=0, world_size=1)
    trainer_state = copy.deepcopy(trainer._trainer_runtime_state())
    payload = {
        "format": "accelerate_fsdp_full_export_v1",
        "manifest_hashes": dict(trainer.manifest_hashes),
        "world_size": 1,
        "global_step": trainer.global_step,
        "online_rollout_round": trainer.online_rollout_round,
        "policy_version": int(getattr(trainer.state_source, "policy_version", 0)),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer.scheduler.state_dict()),
        "rank_shard_hashes": [],
        "prompt_scheduler": prompt_state,
        "prompt_scheduler_by_rank": [prompt_state],
        "state_source": source_state,
        "state_source_by_rank": [source_state],
        "trainer_state": trainer_state,
        "token_budget": copy.deepcopy(trainer_state["token_budget"]),
        "resume_ancestry": [],
        "accelerate_state_dir": "must-not-be-read",
    }
    if trainer._batch_partition_plan is not None:
        trainer_state.update(rank=0, world_size=1)
        payload["trainer_state"] = trainer_state
        payload["trainer_state_by_rank"] = [copy.deepcopy(trainer_state)]
    return payload


def _advance_optimizer_scheduler(
    trainer: FactorialTrainer,
    *,
    steps: int = 1,
) -> None:
    """Create real AdamW state at the exact raw-scheduler cadence."""

    for _ in range(steps):
        for parameter in trainer.model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        trainer.optimizer.step()
        trainer.scheduler.step()
        trainer.optimizer.zero_grad(set_to_none=True)


def _teacher_demo_attempt(record: TrajectoryRecord) -> TeacherDemoAttempt:
    digest_a = "a" * 64
    digest_b = "b" * 64
    return TeacherDemoAttempt(
        attempt_id=f"{record.prompt_id}:candidate-0000",
        prompt_id=record.prompt_id,
        prompt_identity_sha256=digest_a,
        candidate_index=0,
        raw_prompt_text=record.raw_prompt_text or record.prompt_text,
        model_facing_prompt_text=record.prompt_text,
        input_ids=list(record.input_ids),
        response_ids=list(record.response_ids),
        response_text=record.response_text,
        response_token_mask=[True for _ in record.response_ids],
        behavior_logprobs=[-0.1 for _ in record.response_ids],
        logprob_status=LOGPROB_AVAILABLE,
        finish_reason="eos",
        teacher_id="teacher/demo-fixture",
        teacher_revision="fixture-revision",
        sampling_request_seed=record.sampling_request_seed,
        actual_sampling_seed=record.actual_sampling_seed,
        sampling_protocol_id=record.sampling_protocol_id,
        sampling_temperature=record.sampling_temperature,
        top_p=record.top_p,
        top_k=record.top_k,
        min_p=record.min_p,
        verifier_reward=1.0,
        verification_trace={"accepted": True},
        accepted=True,
        prompt_protocol="teacher-demo-fixture-v1",
        enable_thinking=False,
        chat_template_sha256=digest_a,
        raw_prompt_sha256=digest_a,
        model_facing_prompt_sha256=digest_b,
        tokenizer_fingerprint=digest_b,
    )


@pytest.mark.unit
def test_factorial_checkpoint_preflight_and_rank_state_contracts(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    run_root = tmp_path / "run"
    checkpoints = run_root / "checkpoints"
    checkpoints.mkdir(parents=True)
    targets = {
        "metadata": checkpoints / "step-00000001.pt",
        "accelerator_state": checkpoints / "step-00000001.accelerate",
    }
    reports = [
        _checkpoint_publication_preflight_report(
            targets,
            expected_run_root=run_root,
            rank=rank,
        )
        for rank in range(2)
    ]
    targets["metadata"].write_bytes(b"main-created-after-collective-sample")
    assert _validate_checkpoint_publication_preflight_reports(
        reports,
        world_size=2,
    ) == {name: path.absolute() for name, path in targets.items()}
    staging = checkpoints / ".metadata-stage"
    staging.write_bytes(b"new-checkpoint")
    with pytest.raises(FileExistsError):
        publish_path_no_clobber(staging, targets["metadata"])
    assert targets["metadata"].read_bytes() == b"main-created-after-collective-sample"
    assert staging.read_bytes() == b"new-checkpoint"
    existing_reports = [
        _checkpoint_publication_preflight_report(
            targets,
            expected_run_root=run_root,
            rank=rank,
        )
        for rank in range(2)
    ]
    with pytest.raises(RuntimeError, match="preflight failed"):
        _validate_checkpoint_publication_preflight_reports(
            existing_reports,
            world_size=2,
        )

    states = [
        {"position": 1, "rank": 0, "world_size": 2},
        {"position": 2, "rank": 1, "world_size": 2},
    ]
    assert _validate_rank_state_mappings(
        states,
        name="prompt_scheduler_by_rank",
        world_size=2,
    ) == states
    with pytest.raises(ValueError, match="exactly one mapping per rank"):
        _validate_rank_state_mappings(
            states[:1],
            name="prompt_scheduler_by_rank",
            world_size=2,
        )
    with pytest.raises(ValueError, match="non-mapping"):
        _validate_rank_state_mappings(
            [states[0], "bad"],
            name="state_source_by_rank",
            world_size=2,
        )
    with pytest.raises(ValueError, match="rank metadata differs"):
        _validate_rank_state_mappings(
            [states[1], states[0]],
            name="state_source_by_rank",
            world_size=2,
        )
    wrong_world = [dict(state) for state in states]
    wrong_world[1]["world_size"] = 3
    with pytest.raises(ValueError, match="world_size metadata differs"):
        _validate_rank_state_mappings(
            wrong_world,
            name="state_source_by_rank",
            world_size=2,
        )
    assert _validate_resume_world_size(2, running_world_size=2) == 2
    with pytest.raises(ValueError, match="differs from the running world"):
        _validate_resume_world_size(1, running_world_size=2)


@pytest.mark.unit
def test_factorial_finalizer_validates_exact_external_accelerator_tree_and_rank_states(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    run_root = tmp_path / "run"
    checkpoint_path = run_root / "checkpoints" / "step-00000001.pt"
    checkpoint_path.parent.mkdir(parents=True)
    state_root = checkpoint_path.with_suffix(".accelerate")
    state_root.mkdir()
    state_file = state_root / "state.bin"
    state_file.write_bytes(b"state")
    files = accelerator_state_file_hashes(
        state_root,
        expected_run_root=run_root,
    )
    prompt_states = [
        {"position": 1, "rank": 0, "world_size": 2},
        {"position": 2, "rank": 1, "world_size": 2},
    ]
    source_states = [
        {"kind": "fixed", "rank": 0, "world_size": 2},
        {"kind": "fixed", "rank": 1, "world_size": 2},
    ]
    payload = {
        "format": "accelerate_fsdp_full_export_v1",
        "accelerate_state_dir": str(state_root.absolute()),
        "accelerate_state_files": files,
        "accelerate_state_sha256": sha256_value(files),
        "world_size": 2,
        "prompt_scheduler": prompt_states[0],
        "prompt_scheduler_by_rank": prompt_states,
        "state_source": source_states[0],
        "state_source_by_rank": source_states,
    }
    config = {"trainer": {"backend": "accelerate"}}
    assert _validate_factorial_accelerator_state(
        run_root,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=payload,
        resolved_config=config,
    ) == files
    assert _validate_factorial_rank_states(payload) == 2

    state_file.unlink()
    with pytest.raises(ValueError):
        _validate_factorial_accelerator_state(
            run_root,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=payload,
            resolved_config=config,
        )
    state_file.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="content differs"):
        _validate_factorial_accelerator_state(
            run_root,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=payload,
            resolved_config=config,
        )
    state_file.write_bytes(b"state")
    extra = state_root / "extra.bin"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="content differs"):
        _validate_factorial_accelerator_state(
            run_root,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=payload,
            resolved_config=config,
        )
    extra.unlink()
    empty = state_root / "extra-empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="unbound empty directory"):
        _validate_factorial_accelerator_state(
            run_root,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=payload,
            resolved_config=config,
        )
    empty.rmdir()
    symlink = state_root / "linked-state.bin"
    symlink.symlink_to("state.bin")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        _validate_factorial_accelerator_state(
            run_root,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=payload,
            resolved_config=config,
        )

    wrong_rank = copy.deepcopy(payload)
    wrong_rank["state_source_by_rank"][1]["rank"] = 0
    with pytest.raises(ValueError, match="rank differs"):
        _validate_factorial_rank_states(wrong_rank)
    wrong_world = copy.deepcopy(payload)
    wrong_world["prompt_scheduler_by_rank"][1]["world_size"] = 3
    with pytest.raises(ValueError, match="world_size differs"):
        _validate_factorial_rank_states(wrong_world)


@pytest.mark.unit
def test_factorial_checkpoint_same_step_is_published_once(
    tmp_path,
    tokenizer,
    tiny_model,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    trainer = _trainer(
        model=tiny_model,
        source=SequenceStateSource([]),
        supervisor=VerifiedReplaySupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(max_steps=1),
        run_dir=tmp_path / "single-publication",
        prompt_id="prompt-1",
    )
    trainer.global_step = 1
    trainer.token_budget.consumed = 1
    trainer.token_budget.accepted_optimizer_updates = 1
    trainer.cumulative_counts["model_facing_input_tokens_processed"] = 1.0
    with torch.no_grad():
        next(trainer.model.parameters()).add_(0.01)
    writes = 0

    def counting_save(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal writes
        writes += 1
        return artifact_save_checkpoint(*args, **kwargs)

    monkeypatch.setattr(
        "posttrain_circuits.learning.training.factorial_trainer.save_checkpoint",
        counting_save,
    )
    checkpoint = trainer.run_dir / "checkpoints" / "step-00000001.pt"
    first = trainer.save(checkpoint)
    second = trainer.save(checkpoint)
    assert first == second
    assert writes == 1
    checkpoint.write_bytes(b"same-path-replacement")
    with pytest.raises(RuntimeError, match="bytes changed"):
        trainer.save(checkpoint)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("terminal_case", "expected_reason"),
    (
        ("max_steps", "max_steps_safety_limit"),
        ("signal", "signal_at_optimizer_boundary"),
        ("signal_at_max", "max_steps_safety_limit"),
        ("exact_budget", "token_budget_exactly_consumed"),
        ("exact_budget_at_max", "token_budget_exactly_consumed"),
    ),
)
def test_terminal_reason_is_bound_before_first_final_checkpoint_save(
    tmp_path,
    tokenizer,
    tiny_model,
    terminal_case: str,
    expected_reason: str,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=84)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=84,
        behavior_policy_id="terminal-checkpoint-test",
    )
    supervisor = VerifiedReplaySupervisor(tokenizer.pad_token_id)
    prepared = supervisor.prepare_targets(
        TrajectoryBatch([positive], policy_version=0),
        teacher=None,
        verifier=None,
    )
    update_tokens = int(prepared.attention_mask.sum().item())
    trainer = _trainer(
        model=tiny_model,
        source=SequenceStateSource([[positive]]),
        supervisor=supervisor,
        config=TrainerConfig(
            max_steps=(
                1
                if terminal_case in {
                    "max_steps",
                    "signal_at_max",
                    "exact_budget_at_max",
                }
                else 2
            ),
            token_budget=(
                update_tokens
                if terminal_case in {"exact_budget", "exact_budget_at_max"}
                else update_tokens * 3
            ),
            checkpoint_every=1,
        ),
        run_dir=tmp_path / terminal_case,
        prompt_id=example.example_id,
    )
    if terminal_case in {"signal", "signal_at_max", "exact_budget_at_max"}:
        finish_update = trainer._finish_optimizer_update

        def finish_and_signal(*, started: float) -> dict[str, Any]:
            metric = finish_update(started=started)
            trainer._terminate = True
            return metric

        trainer._finish_optimizer_update = finish_and_signal  # type: ignore[method-assign]

    history = trainer.train()

    expected_path = (
        trainer.run_dir / "checkpoints" / "step-00000001.pt"
    ).absolute()
    assert len(history) == 1
    assert trainer.global_step == 1
    assert trainer.final_checkpoint_path == expected_path
    assert not expected_path.with_name("step-00000001-terminal.pt").exists()
    payload = torch.load(expected_path, map_location="cpu", weights_only=False)
    assert payload["global_step"] == trainer.global_step
    assert payload["token_budget"] == trainer.token_budget.state_dict()
    assert payload["trainer_state"]["token_budget"] == trainer.token_budget.state_dict()
    assert payload["token_budget"]["stop_reason"] == expected_reason


@pytest.mark.unit
def test_factorial_resume_ancestry_is_content_addressed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    ancestry = _extend_resume_ancestry([], checkpoint)
    assert ancestry == [
        "sha256:47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    ]
    with pytest.raises(ValueError, match="repeated checkpoint"):
        _extend_resume_ancestry(ancestry, checkpoint)
    with pytest.raises(ValueError, match="cycle"):
        _extend_resume_ancestry(
            [f"sha256:{'a' * 64}", f"sha256:{'a' * 64}"],
            checkpoint,
        )


@pytest.mark.unit
def test_two_rank_prompt_shard_bindings_roundtrip_and_reject_cross_rank_resume() -> None:
    shards = ["a" * 64, "b" * 64]
    assert _validate_rank_shard_hashes(
        shards,
        expected_rank_hash=shards[0],
        rank=0,
        world_size=2,
    ) == shards
    assert _validate_rank_shard_hashes(
        shards,
        expected_rank_hash=shards[1],
        rank=1,
        world_size=2,
    ) == shards
    with pytest.raises(ValueError, match="differs for distributed rank 1"):
        _validate_rank_shard_hashes(
            shards,
            expected_rank_hash=shards[0],
            rank=1,
            world_size=2,
        )


@pytest.mark.unit
def test_accelerate_accumulation_context_wraps_forward_and_backward(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=79)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=79,
        behavior_policy_id="accumulation-scope-test",
    )
    accelerator = AccumulationScopeAccelerator()
    supervisor = ScopeAssertingReplaySupervisor(tokenizer.pad_token_id, accelerator)
    trainer = _trainer(
        model=tiny_model,
        source=SequenceStateSource([[positive]]),
        supervisor=supervisor,
        config=TrainerConfig(max_steps=1),
        run_dir=tmp_path / "accumulation-scope",
        prompt_id=example.example_id,
    )
    trainer._accelerator = accelerator
    supervision = supervisor.prepare_targets(
        TrajectoryBatch([positive], policy_version=0),
        teacher=None,
        verifier=None,
    )
    assert trainer._training_micro_step(supervision) is True
    assert accelerator.events == ["enter", "forward", "backward", "exit"]


@pytest.mark.unit
def test_accelerate_keeps_scheduler_raw_and_registers_its_checkpoint_state(
    tmp_path,
    tokenizer,
    tiny_model,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    module = types.ModuleType("accelerate")
    module.Accelerator = InitializingFakeAccelerator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "accelerate", module)
    InitializingFakeAccelerator.instances.clear()
    example = build_smoke_examples(1, seed=93)[0]
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler([example.example_id], ["prompt"], 1),
        state_source=SequenceStateSource([]),
        supervisor=VerifiedReplaySupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=1,
            backend="accelerate",
            gradient_accumulation_steps=4,
        ),
        run_dir=tmp_path / "raw-scheduler-construction",
    )

    accelerator = InitializingFakeAccelerator.instances[-1]
    assert accelerator.gradient_accumulation_steps == 4
    assert accelerator.step_scheduler_with_optimizer is False
    assert accelerator.prepared == (tiny_model, optimizer)
    assert accelerator.registered == [scheduler]
    assert trainer.scheduler is scheduler


@pytest.mark.unit
def test_flop_parameter_count_is_captured_before_accelerate_wrapping(
    tmp_path,
    tokenizer,
    tiny_model,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    replacement_model = torch.nn.Linear(1, 1)

    class ReplacingAccelerator(InitializingFakeAccelerator):
        def prepare(self, *items: Any) -> tuple[Any, ...]:
            self.prepared = items
            return replacement_model, items[1]

    module = types.ModuleType("accelerate")
    module.Accelerator = ReplacingAccelerator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "accelerate", module)
    example = build_smoke_examples(1, seed=94)[0]
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    expected_parameter_count = sum(
        parameter.numel() for parameter in tiny_model.parameters()
    )

    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler([example.example_id], ["prompt"], 1),
        state_source=SequenceStateSource([]),
        supervisor=VerifiedReplaySupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(max_steps=1, backend="accelerate"),
        run_dir=tmp_path / "pre-wrap-parameter-count",
    )

    assert trainer.model is replacement_model
    assert trainer._full_parameter_count == expected_parameter_count
    assert trainer._full_parameter_count != sum(
        parameter.numel() for parameter in replacement_model.parameters()
    )


@pytest.mark.unit
def test_allocation_neutral_sequence_mean_loss_scaling_recovers_exact_global_mean() -> None:
    """Undoing Accelerate/DDP averaging leaves each 64-slot sequence weight once."""

    for world_size, expected_microsteps in ((1, 16), (2, 8), (3, 6), (4, 4)):
        plans = [ExactGlobalBatchPlan(64, 4, rank, world_size) for rank in range(world_size)]
        assert all(
            plan.microsteps_per_optimizer_update == expected_microsteps for plan in plans
        )
        # A local sequence-mean loss of one has a numerator equal to its local
        # batch size.  Accelerate divides by A and FSDP/DDP averages by W.
        recovered_global_mean = sum(
            plan.accelerate_sequence_mean_loss_multiplier(local_batch_size)
            / expected_microsteps
            / world_size
            for plan in plans
            for local_batch_size in plan.microbatch_sizes
        )
        assert recovered_global_mean == pytest.approx(1.0)


@pytest.mark.unit
def test_allocation_neutral_objective_metrics_are_global_sequence_means() -> None:
    """Every supported allocation reports the same 64-sequence objective."""

    slot_values = {slot: float(slot + 1) for slot in range(64)}
    expected_mean = sum(slot_values.values()) / 64
    for world_size in (1, 2, 3, 4):
        summaries: list[dict[str, Any]] = []
        for rank in range(world_size):
            plan = ExactGlobalBatchPlan(64, 4, rank, world_size)
            local_numerator = 0.0
            for microbatch_index, local_batch_size in enumerate(plan.microbatch_sizes):
                slots = plan.global_slots_for_microbatch(microbatch_index)
                assert len(slots) == local_batch_size
                local_mean = sum(slot_values[slot] for slot in slots) / local_batch_size
                local_numerator += local_mean * local_batch_size
            summaries.append(
                {
                    "rank": rank,
                    "sequence_count": plan.local_sequence_count,
                    "metric_numerators": {"verified_replay_loss": local_numerator},
                }
            )
        assert sum(int(row["sequence_count"]) for row in summaries) == 64
        metrics = _allocation_neutral_global_objective_metrics(
            summaries,
            world_size=world_size,
            global_batch_size=64,
        )
        assert metrics["verified_replay_loss"] == pytest.approx(expected_mean)
        assert metrics["objective_metric_sequence_count"] == 64.0


@pytest.mark.unit
@pytest.mark.parametrize("world_size", (1, 2, 3, 4))
def test_allocation_neutral_public_metrics_are_global_for_every_world_size(
    world_size: int,
) -> None:
    """Public counters never expose a rank-0 shard as the scientific total."""

    global_step = 2
    parameter_count = 100
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        plan = ExactGlobalBatchPlan(64, 4, rank, world_size)
        local_sequences = plan.local_sequence_count
        rows.append(
            {
                "rank": rank,
                "local_sequence_count": local_sequences,
                "collection": {
                    "generated_trajectories": float(local_sequences),
                    "successful_trajectories": float(local_sequences),
                    "effective_positive_sequences": float(local_sequences),
                    "effective_supervised_tokens": float(local_sequences * 5),
                    "retry_count": 0.0,
                },
                "cumulative": {
                    "prompts_consumed": float(global_step * local_sequences),
                    "trajectories_generated": float(global_step * local_sequences),
                    "response_tokens_generated": float(global_step * local_sequences * 7),
                    "supervised_response_tokens": float(global_step * local_sequences * 5),
                    "model_facing_input_tokens_processed": float(global_step * 192),
                    "forward_backward_flop_estimate": float(
                        6 * parameter_count * global_step * local_sequences * 5
                    ),
                },
                "local_supervised_tokens": float(local_sequences * 5),
                "local_input_tokens": local_sequences * 3,
                "global_input_tokens": 192,
                "parameter_update_squared_norm": float(rank + 1),
                "peak_allocated_gpu_memory": float(10 + rank),
                "wall_clock_seconds": float(20 + rank),
            }
        )

    metrics = _allocation_neutral_global_public_metrics(
        rows,
        world_size=world_size,
        global_batch_size=64,
        global_step=global_step,
    )

    assert [row["local_sequence_count"] for row in rows] == {
        1: [64],
        2: [32, 32],
        3: [22, 21, 21],
        4: [16, 16, 16, 16],
    }[world_size]
    assert metrics["collection_metrics"]["generated_trajectories"] == 64.0
    assert metrics["collection_metrics"]["reward_rate"] == 1.0
    assert metrics["effective_supervised_tokens_this_update"] == 320.0
    assert metrics["cumulative_counts"]["prompts_consumed"] == 128.0
    assert metrics["cumulative_counts"]["trajectories_generated"] == 128.0
    assert metrics["cumulative_counts"]["model_facing_input_tokens_processed"] == 384.0
    assert metrics["cumulative_counts"]["forward_backward_flop_estimate"] == float(
        6 * parameter_count * 128 * 5
    )
    assert metrics["parameter_update_norm"] == pytest.approx(
        sum(range(1, world_size + 1)) ** 0.5
    )
    assert metrics["peak_allocated_gpu_memory"] == float(9 + world_size)
    assert metrics["wall_clock_seconds"] == float(19 + world_size)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("world_size", "microsteps"),
    ((1, 16), (2, 8), (3, 6), (4, 4)),
)
def test_raw_scheduler_cadence_is_one_step_per_global_update(
    world_size: int,
    microsteps: int,
) -> None:
    """Accumulation count changes with W; AdamW/LambdaLR cadence does not."""

    assert ExactGlobalBatchPlan(64, 4, 0, world_size).microsteps_per_optimizer_update == (
        microsteps
    )
    _validate_optimizer_scheduler_cadence(
        optimizer_state={
            "state": {
                0: {"step": torch.tensor(2.0)},
                1: {"step": 2},
            },
            "param_groups": [{"params": [0, 1]}],
        },
        scheduler_state={"last_epoch": 2, "_step_count": 3},
        global_step=2,
    )


@pytest.mark.unit
@pytest.mark.parametrize("defect", ("scheduler_epoch", "scheduler_count", "adam_step"))
def test_optimizer_scheduler_cadence_rejects_tampering(defect: str) -> None:
    optimizer_state: dict[str, Any] = {
        "state": {0: {"step": torch.tensor(2.0)}},
        "param_groups": [{"params": [0]}],
    }
    scheduler_state = {"last_epoch": 2, "_step_count": 3}
    if defect == "scheduler_epoch":
        scheduler_state["last_epoch"] = 1
    elif defect == "scheduler_count":
        scheduler_state["_step_count"] = 2
    else:
        optimizer_state["state"][0]["step"] = torch.tensor(1.0)
    with pytest.raises(ValueError):
        _validate_optimizer_scheduler_cadence(
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            global_step=2,
        )


@pytest.mark.unit
@pytest.mark.parametrize("world_size", (1, 2, 3, 4))
def test_allocation_neutral_rejected_window_rollback_covers_every_rank(
    world_size: int,
    tokenizer,
) -> None:  # type: ignore[no-untyped-def]
    """Static rank fixtures prove an unadmitted window restores every cursor."""

    example = build_smoke_examples(1, seed=87)[0]
    record = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=87,
        behavior_policy_id="teacher-demo-fixture",
    )
    attempt = _teacher_demo_attempt(record)
    supervisor = CanonicalSFTSupervisor(tokenizer.pad_token_id)
    for rank in range(world_size):
        plan = ExactGlobalBatchPlan(64, 4, rank, world_size)
        trainer = object.__new__(FactorialTrainer)
        trainer._batch_partition_plan = plan
        trainer._accumulation_micro_step = 0
        trainer._parameters_before_update = None
        trainer.prompt_scheduler = PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=rank,
            world_size=world_size,
            global_batch_size=64,
            max_microbatch_size=4,
        )
        trainer.state_source = TeacherDemoStateSource([attempt])
        trainer.cumulative_counts = {
            "prompts_consumed": 0.0,
            "trajectories_generated": 0.0,
            "response_tokens_generated": 0.0,
            "supervised_response_tokens": 0.0,
            "model_facing_input_tokens_processed": 0.0,
            "forward_backward_flop_estimate": 0.0,
        }
        trainer._pending_collection = {
            "generated_trajectories": 0.0,
            "successful_trajectories": 0.0,
            "effective_positive_sequences": 0.0,
            "effective_supervised_tokens": 0.0,
            "retry_count": 0.0,
        }
        trainer._pending_processed_tokens = 0.0
        trainer._pending_input_tokens = 0
        trainer._current_global_update_tokens = 0
        trainer._pending_metric_sums = {}
        trainer._pending_metric_calls = 0
        trainer._pending_metric_sequence_count = 0
        snapshot = trainer._capture_rejected_window_snapshot()

        for expected_size in plan.microbatch_sizes:
            prompts = trainer.prompt_scheduler.next_batch()
            assert len(prompts.prompt_ids) == expected_size
            trajectories = trainer.state_source.get_batch(None, prompts, 0)
            supervision = supervisor.prepare_targets(trajectories, None, None)
            trainer._register_collection(prompts, trajectories, supervision, attempts=1)
        assert trainer.prompt_scheduler.state_dict()["global_slot_cursor"] == 64
        assert sum(trainer.state_source.state_dict()["cursor"].values()) == plan.local_sequence_count
        assert trainer.cumulative_counts["prompts_consumed"] == plan.local_sequence_count

        # This is the branch taken after collective token reservation rejects.
        trainer._restore_rejected_window_snapshot(snapshot)
        assert trainer.prompt_scheduler.state_dict()["global_slot_cursor"] == 0
        assert trainer.prompt_scheduler.state_dict()["microbatch_index"] == 0
        assert sum(trainer.state_source.state_dict()["cursor"].values()) == 0
        assert trainer.cumulative_counts["prompts_consumed"] == 0.0
        assert all(value == 0.0 for value in trainer._pending_collection.values())


@pytest.mark.unit
def test_allocation_neutral_rejected_window_rolls_back_to_committed_boundary(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    """A rejected second window cannot advance checkpointed collection state."""

    example = build_smoke_examples(1, seed=82)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=82,
        behavior_policy_id="allocation-neutral-test",
    )
    source = PromptAwareSequenceStateSource(positive)
    supervisor = CountingCanonicalSFTSupervisor(tokenizer.pad_token_id)
    probe_batch = supervisor.prepare_targets(
        TrajectoryBatch([copy.deepcopy(positive) for _ in range(4)], policy_version=0),
        teacher=None,
        verifier=None,
    )
    committed_window_tokens = int(probe_batch.attention_mask.sum().item()) * 16
    prompt_scheduler = PromptScheduler.for_allocation_neutral_rank(
        [example.example_id],
        ["prompt"],
        rank=0,
        world_size=1,
        global_batch_size=64,
        max_microbatch_size=4,
    )
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=prompt_scheduler,
        state_source=source,
        supervisor=supervisor,
        config=TrainerConfig(
            max_steps=2,
            token_budget=committed_window_tokens * 2 - 1,
            checkpoint_every=1,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / "rejected-window",
    )
    optimizer_steps = 0
    original_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counting_step  # type: ignore[method-assign]
    history = trainer.train()

    assert len(history) == 1
    assert trainer.global_step == 1
    assert optimizer_steps == 1
    _validate_optimizer_scheduler_cadence(
        optimizer_state=trainer.optimizer.state_dict(),
        scheduler_state=trainer.scheduler.state_dict(),
        global_step=trainer.global_step,
    )
    assert supervisor.loss_calls == 16
    assert trainer.token_budget.consumed == committed_window_tokens
    assert trainer.token_budget.accepted_optimizer_updates == 1
    assert trainer.token_budget.stop_reason == "token_budget_exhausted_before_next_optimizer_update"
    assert source.calls == 16
    assert trainer.prompt_scheduler.state_dict()["global_slot_cursor"] == 64
    assert trainer.prompt_scheduler.state_dict()["microbatch_index"] == 0
    assert trainer.cumulative_counts["prompts_consumed"] == 64.0
    assert trainer._pending_collection == {
        "generated_trajectories": 0.0,
        "successful_trajectories": 0.0,
        "effective_positive_sequences": 0.0,
        "effective_supervised_tokens": 0.0,
        "retry_count": 0.0,
    }
    assert trainer._pending_metric_sums == {}
    assert trainer._pending_metric_sequence_count == 0
    checkpoint = trainer.run_dir / "checkpoints" / "step-00000001.pt"
    terminal_checkpoint = (
        trainer.run_dir / "checkpoints" / "step-00000001-terminal.pt"
    ).absolute()
    assert trainer.final_checkpoint_path == terminal_checkpoint
    assert checkpoint.is_file()
    assert terminal_checkpoint.is_file()
    cadence_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert cadence_payload["token_budget"]["stop_reason"] is None
    payload = torch.load(terminal_checkpoint, map_location="cpu", weights_only=False)
    assert payload["token_budget"] == trainer.token_budget.state_dict()
    assert payload["token_budget"]["stop_reason"] == (
        "token_budget_exhausted_before_next_optimizer_update"
    )
    assert payload["prompt_scheduler"]["global_slot_cursor"] == 64
    assert payload["prompt_scheduler"]["microbatch_index"] == 0
    assert payload["state_source"]["calls"] == 16
    assert payload["trainer_state"]["cumulative_counts"]["prompts_consumed"] == 64.0
    assert payload["trainer_state"]["token_budget"]["consumed"] == committed_window_tokens
    assert payload["trainer_state"]["token_budget"]["accepted_optimizer_updates"] == 1


@pytest.mark.unit
def test_allocation_neutral_checkpoint_state_binds_runtime_world_size(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=80)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=80,
        behavior_policy_id="teacher-demo-fixture",
    )
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    prompt_scheduler = PromptScheduler.for_allocation_neutral_rank(
        [example.example_id],
        ["prompt"],
        rank=0,
        world_size=1,
        global_batch_size=64,
        max_microbatch_size=4,
    )
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=prompt_scheduler,
        state_source=PromptAwareSequenceStateSource(positive),
        supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=1,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / "allocation-neutral-checkpoint",
    )
    state = trainer._checkpoint_batch_partition_state()
    assert state is not None
    assert state["world_size"] == 1
    assert state["microbatch_sizes_by_rank"] == [[4] * 16]
    assert not {
        "requested_fsdp_sharding_strategy",
        "effective_fsdp_sharding_strategy",
        "fsdp_wrapper_count",
    }.intersection(state)
    trainer._validate_checkpoint_batch_partition({"trainer_state": {"batch_partition": state}})
    tampered = copy.deepcopy(state)
    tampered["world_size"] = 2
    with pytest.raises(ValueError, match="batch partition differs"):
        trainer._validate_checkpoint_batch_partition(
            {"trainer_state": {"batch_partition": tampered}}
        )


@pytest.mark.unit
@pytest.mark.parametrize("defect", ["overlength", "misaligned_attention_mask"])
def test_allocation_neutral_supervision_tensors_fail_closed_before_forward(
    tmp_path,
    tokenizer,
    tiny_model,
    defect: str,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=89)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=89,
        behavior_policy_id="teacher-demo-fixture",
    )
    supervisor = CanonicalSFTSupervisor(tokenizer.pad_token_id)
    supervision = supervisor.prepare_targets(
        TrajectoryBatch([copy.deepcopy(positive)], policy_version=0),
        teacher=None,
        verifier=None,
    )
    width = int(supervision.input_ids.shape[1])
    limit = width - 1 if defect == "overlength" else width
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=0,
            world_size=1,
            global_batch_size=64,
            max_microbatch_size=4,
        ),
        state_source=PromptAwareSequenceStateSource(positive),
        supervisor=supervisor,
        config=TrainerConfig(
            max_steps=1,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=limit,
        ),
        run_dir=tmp_path / f"model-input-{defect}",
    )
    if defect == "misaligned_attention_mask":
        supervision.attention_mask = supervision.attention_mask[:, :-1]
        match = "attention_mask is not aligned"
    else:
        match = "exceeds max_model_input_length"
    with pytest.raises(ValueError, match=match):
        trainer._validate_supervision_model_input_length(supervision)


@pytest.mark.unit
@pytest.mark.parametrize(
    "defect", [None, "prompt", "response", "manifest", "memory"]
)
def test_teacher_demo_store_model_input_envelope_is_checked_in_full(
    tokenizer,
    defect: str | None,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=90)[0]
    record = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=90,
        behavior_policy_id="teacher-demo-fixture",
    )
    attempt = _teacher_demo_attempt(record)
    max_prompt_tokens = len(attempt.input_ids) + 1
    max_new_tokens = len(attempt.response_ids) + 1
    max_model_input_length = max_prompt_tokens + max_new_tokens
    manifest = {
        "attempt_count": 1,
        "accepted_count": 1,
        "teacher_demo_generation": {
            "candidates_per_prompt": 1,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
        }
    }
    if defect == "prompt":
        attempt.input_ids.extend([1, 1])
        match = "prompt exceeds max_prompt_tokens"
    elif defect == "response":
        attempt.response_ids.extend([1, 1])
        match = "response exceeds max_new_tokens"
    elif defect == "manifest":
        manifest["teacher_demo_generation"]["max_new_tokens"] += 1
        match = "generation envelope differs"
    elif defect == "memory":
        manifest["attempt_count"] = 2
        match = "memory envelope"
    else:
        evidence = _validate_teacher_demo_model_input_lengths(
            [attempt],
            manifest,
            expected_prompt_count=1,
            candidates_per_prompt=1,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            max_model_input_length=max_model_input_length,
        )
        assert evidence["observed_max_prompt_tokens"] == len(attempt.input_ids)
        assert evidence["observed_max_response_tokens"] == len(attempt.response_ids)
        assert evidence["observed_max_model_input_tokens"] == (
            len(attempt.input_ids) + len(attempt.response_ids)
        )
        return
    with pytest.raises(ValueError, match=match):
        _validate_teacher_demo_model_input_lengths(
            [attempt],
            manifest,
            expected_prompt_count=1,
            candidates_per_prompt=1,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            max_model_input_length=max_model_input_length,
        )


@pytest.mark.unit
def test_allocation_neutral_accelerate_save_binds_rank_trainer_state(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=87)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=87,
        behavior_policy_id="teacher-demo-fixture",
    )
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=0,
            world_size=1,
            global_batch_size=64,
            max_microbatch_size=4,
        ),
        state_source=PromptAwareSequenceStateSource(positive),
        supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=1,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / "rank-trainer-state-save",
    )
    trainer._accelerator = CheckpointingFakeAccelerator()
    trainer._full_model_state_for_checkpoint = (  # type: ignore[method-assign]
        lambda: trainer.model.state_dict()
    )
    _advance_optimizer_scheduler(trainer)
    trainer.global_step = 1
    trainer.token_budget.consumed = 1
    trainer.token_budget.accepted_optimizer_updates = 1
    trainer.token_budget.stop_reason = "max_steps_safety_limit"
    trainer.cumulative_counts.update(
        prompts_consumed=64.0,
        trajectories_generated=64.0,
        response_tokens_generated=64.0,
        supervised_response_tokens=64.0,
        model_facing_input_tokens_processed=1.0,
        forward_backward_flop_estimate=64.0,
    )
    with torch.no_grad():
        next(trainer.model.parameters()).add_(0.01)

    checkpoint = trainer.run_dir / "checkpoints" / "step-00000001.pt"
    trainer.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["trainer_state"] == payload["trainer_state_by_rank"][0]
    assert payload["trainer_state_by_rank"][0]["rank"] == 0
    assert payload["trainer_state_by_rank"][0]["world_size"] == 1
    assert payload["trainer_state_by_rank"][0]["cumulative_counts"] == trainer.cumulative_counts
    assert "trainer_state_by_rank" in checkpoint_runtime_state_hashes(payload)


@pytest.mark.unit
def test_allocation_neutral_w3_resume_restores_rank_local_trainer_counters(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=88)[0]
    record = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=88,
        behavior_policy_id="teacher-demo-fixture",
    )
    source = TeacherDemoStateSource([_teacher_demo_attempt(record)])
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=0,
            world_size=1,
            global_batch_size=64,
            max_microbatch_size=4,
        ),
        state_source=source,
        supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=2,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / "w3-rank-trainer-resume",
    )
    accelerator = FakeAccelerator(rank=1, world_size=3)
    trainer._accelerator = accelerator
    trainer.prompt_scheduler = PromptScheduler.for_allocation_neutral_rank(
        [example.example_id],
        ["prompt"],
        rank=1,
        world_size=3,
        global_batch_size=64,
        max_microbatch_size=4,
    )
    trainer._batch_partition_plan = ExactGlobalBatchPlan(64, 4, 1, 3)
    trainer._gradient_accumulation_steps = 6
    batch_partition = trainer._checkpoint_batch_partition_state()
    assert batch_partition is not None

    prompt_states: list[dict[str, Any]] = []
    source_states: list[dict[str, Any]] = []
    trainer_states: list[dict[str, Any]] = []
    count_states: list[dict[str, float]] = []
    token_budget = trainer.token_budget.state_dict()
    token_budget.update(consumed=33, accepted_optimizer_updates=1)
    for rank, local_sequences in enumerate((22, 21, 21)):
        rank_scheduler = PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=rank,
            world_size=3,
            global_batch_size=64,
            max_microbatch_size=4,
        )
        for _ in range(6):
            rank_scheduler.next_batch()
        prompt_states.append(dict(rank_scheduler.state_dict()))
        source_state = dict(source.state_dict())
        source_state["cursor"] = {example.example_id: local_sequences}
        source_state.update(rank=rank, world_size=3)
        source_states.append(source_state)
        cumulative_counts = {
            "prompts_consumed": float(local_sequences),
            "trajectories_generated": float(local_sequences),
            "response_tokens_generated": float(local_sequences * (rank + 2)),
            "supervised_response_tokens": float(local_sequences * (rank + 1)),
            "model_facing_input_tokens_processed": 33.0,
            "forward_backward_flop_estimate": float(1000 + rank),
        }
        count_states.append(cumulative_counts)
        trainer_states.append(
            {
                "cumulative_counts": cumulative_counts,
                "accumulation_micro_step": 0,
                "token_budget": copy.deepcopy(token_budget),
                "batch_partition": copy.deepcopy(batch_partition),
                "rank": rank,
                "world_size": 3,
            }
        )

    checkpoint = trainer.run_dir / "checkpoints" / "step-00000001.pt"
    state_dir = checkpoint.with_suffix(".accelerate")
    state_dir.mkdir()
    (state_dir / "state.bin").write_bytes(b"w3-state")
    state_files = accelerator_state_file_hashes(
        state_dir,
        expected_run_root=trainer.run_dir,
    )
    _advance_optimizer_scheduler(trainer)
    payload = {
        "format": "accelerate_fsdp_full_export_v1",
        "manifest_hashes": {},
        "world_size": 3,
        "global_step": 1,
        "online_rollout_round": 0,
        "policy_version": 0,
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer.scheduler.state_dict()),
        "rank_shard_hashes": [],
        "prompt_scheduler": prompt_states[0],
        "prompt_scheduler_by_rank": prompt_states,
        "state_source": source_states[0],
        "state_source_by_rank": source_states,
        "trainer_state": trainer_states[0],
        "trainer_state_by_rank": trainer_states,
        "token_budget": token_budget,
        "resume_ancestry": [],
        "accelerate_state_dir": str(state_dir.absolute()),
        "accelerate_state_files": state_files,
        "accelerate_state_sha256": sha256_value(state_files),
    }
    with pytest.raises(ValueError, match="wrapper count"):
        _validate_factorial_rank_states(
            payload,
            expected_max_steps=2,
            expected_max_model_input_length=1536,
        )
    missing_rank_table = copy.deepcopy(payload)
    missing_rank_table.pop("trainer_state_by_rank")
    with pytest.raises(ValueError):
        _validate_factorial_rank_states(missing_rank_table, expected_max_steps=2)
    wrong_trainer_alias = copy.deepcopy(payload)
    wrong_trainer_alias["trainer_state"]["cumulative_counts"]["prompts_consumed"] = 0.0
    with pytest.raises(ValueError):
        _validate_factorial_rank_states(wrong_trainer_alias, expected_max_steps=2)
    torch.save(payload, checkpoint)

    trainer.resume(checkpoint)
    assert accelerator.load_state_calls == 1
    assert trainer.cumulative_counts == count_states[1]
    assert trainer.cumulative_counts != count_states[0]
    assert sum(trainer.state_source.state_dict()["cursor"].values()) == 21

    inconsistent = copy.deepcopy(payload)
    inconsistent["trainer_state_by_rank"][2]["token_budget"]["consumed"] = 34
    inconsistent["trainer_state_by_rank"][2]["cumulative_counts"][
        "model_facing_input_tokens_processed"
    ] = 34.0
    with pytest.raises(ValueError):
        _validate_factorial_rank_states(inconsistent, expected_max_steps=2)
    inconsistent_checkpoint = trainer.run_dir / "checkpoints" / "inconsistent-w3.pt"
    torch.save(inconsistent, inconsistent_checkpoint)
    accelerator.load_state_calls = 0
    with pytest.raises(ValueError, match="token budget differs across"):
        trainer.resume(inconsistent_checkpoint)
    assert accelerator.load_state_calls == 0


@pytest.mark.unit
def test_allocation_neutral_training_rejects_non_teacher_demo_source(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=83)[0]
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with pytest.raises(ValueError, match="deterministic teacher-demo"):
        FactorialTrainer(
            model=tiny_model,
            optimizer=optimizer,
            scheduler=scheduler,
            prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
                [example.example_id],
                ["prompt"],
                rank=0,
                world_size=1,
                global_batch_size=64,
                max_microbatch_size=4,
            ),
            state_source=SequenceStateSource([]),
            supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
            config=TrainerConfig(
                max_steps=1,
                batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
                global_batch_size=64,
                max_microbatch_size=4,
                max_model_input_length=1536,
            ),
            run_dir=tmp_path / "non-teacher-demo-source",
        )


@pytest.mark.unit
def test_cross_world_resume_rejects_before_model_or_optimizer_state_load(
    tmp_path,
    tokenizer,
    tiny_model,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """Explicit same-world resume is allowed; scheduler fresh attempts pass no resume path."""

    example = build_smoke_examples(1, seed=81)[0]
    trainer = _trainer(
        model=tiny_model,
        source=SequenceStateSource([]),
        supervisor=VerifiedReplaySupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(max_steps=1),
        run_dir=tmp_path / "cross-world-resume",
        prompt_id=example.example_id,
    )
    checkpoint = tmp_path / "wrong-world.pt"
    torch.save({"manifest_hashes": {}, "world_size": 2}, checkpoint)
    parameters_before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    load_called = False

    def must_not_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal load_called
        del args, kwargs
        load_called = True
        raise AssertionError("cross-world resume reached mutable checkpoint loading")

    monkeypatch.setattr(
        "posttrain_circuits.learning.training.factorial_trainer.load_checkpoint",
        must_not_load,
    )
    with pytest.raises(ValueError, match="differs from the running world"):
        trainer.resume(checkpoint)
    assert load_called is False
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, trainer.model.parameters(), strict=True)
    )


@pytest.mark.unit
def test_accelerate_cross_world_resume_rejects_before_load_state(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=84)[0]
    trainer = _trainer(
        model=tiny_model,
        source=SequenceStateSource([]),
        supervisor=VerifiedReplaySupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(max_steps=1),
        run_dir=tmp_path / "accelerate-cross-world-resume",
        prompt_id=example.example_id,
    )
    accelerator = FakeAccelerator(world_size=2)
    trainer._accelerator = accelerator
    checkpoint = tmp_path / "accelerate-wrong-world.pt"
    torch.save(_valid_accelerate_resume_payload(trainer), checkpoint)

    with pytest.raises(ValueError, match="differs from the running world"):
        trainer.resume(checkpoint)
    assert accelerator.load_state_calls == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "defect",
    (
        "trainer_state_fields",
        "partial_accumulation",
        "token_budget",
        "prompt_state",
        "source_state",
    ),
)
def test_accelerate_invalid_runtime_metadata_rejects_before_load_state(
    tmp_path,
    tokenizer,
    tiny_model,
    defect: str,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=85)[0]
    trainer = _trainer(
        model=tiny_model,
        source=SequenceStateSource([]),
        supervisor=VerifiedReplaySupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(max_steps=1),
        run_dir=tmp_path / f"accelerate-invalid-{defect}",
        prompt_id=example.example_id,
    )
    accelerator = FakeAccelerator()
    trainer._accelerator = accelerator
    payload = _valid_accelerate_resume_payload(trainer)
    if defect == "trainer_state_fields":
        payload["trainer_state"] = {"accumulation_micro_step": 0}
    elif defect == "partial_accumulation":
        payload["trainer_state"]["accumulation_micro_step"] = 1
    elif defect == "token_budget":
        payload["trainer_state"]["token_budget"]["consumed"] = (
            trainer.config.token_budget + 1
        )
        payload["token_budget"] = copy.deepcopy(payload["trainer_state"]["token_budget"])
    elif defect == "prompt_state":
        payload["prompt_scheduler_by_rank"][0].pop("position")
        payload["prompt_scheduler"] = copy.deepcopy(payload["prompt_scheduler_by_rank"][0])
    elif defect == "source_state":
        payload["state_source_by_rank"][0].pop("calls")
        payload["state_source"] = copy.deepcopy(payload["state_source_by_rank"][0])
    checkpoint = tmp_path / f"accelerate-invalid-{defect}.pt"
    torch.save(payload, checkpoint)

    with pytest.raises((KeyError, ValueError)):
        trainer.resume(checkpoint)
    assert accelerator.load_state_calls == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "defect",
    (
        "partial_prompt_microbatch",
        "advanced_global_cursor",
        "advanced_teacher_cursor",
        "missing_trainer_state_by_rank",
        "rank_local_trainer_counter",
        "policy_version",
        "online_rollout_round",
    ),
)
def test_allocation_neutral_resume_rejects_non_boundary_state_before_load_state(
    tmp_path,
    tokenizer,
    tiny_model,
    defect: str,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=86)[0]
    record = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=86,
        behavior_policy_id="teacher-demo-fixture",
    )
    source = TeacherDemoStateSource([_teacher_demo_attempt(record)])
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=0,
            world_size=1,
            global_batch_size=64,
            max_microbatch_size=4,
        ),
        state_source=source,
        supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=1,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / f"allocation-neutral-resume-{defect}",
    )
    accelerator = FakeAccelerator()
    trainer._accelerator = accelerator
    payload = _valid_accelerate_resume_payload(trainer)
    if defect == "partial_prompt_microbatch":
        payload["prompt_scheduler_by_rank"][0]["microbatch_index"] = 1
    elif defect == "advanced_global_cursor":
        payload["prompt_scheduler_by_rank"][0]["global_slot_cursor"] = 64
    elif defect == "advanced_teacher_cursor":
        payload["state_source_by_rank"][0]["cursor"] = {example.example_id: 1}
    elif defect == "missing_trainer_state_by_rank":
        payload.pop("trainer_state_by_rank")
    elif defect == "rank_local_trainer_counter":
        payload["trainer_state_by_rank"][0]["cumulative_counts"]["prompts_consumed"] = 1.0
        payload["trainer_state"] = copy.deepcopy(payload["trainer_state_by_rank"][0])
    elif defect == "policy_version":
        payload["policy_version"] = 1
    elif defect == "online_rollout_round":
        payload["online_rollout_round"] = 1
    payload["prompt_scheduler"] = copy.deepcopy(payload["prompt_scheduler_by_rank"][0])
    payload["state_source"] = copy.deepcopy(payload["state_source_by_rank"][0])
    checkpoint = tmp_path / f"allocation-neutral-non-boundary-{defect}.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError):
        trainer.resume(checkpoint)
    assert accelerator.load_state_calls == 0


@pytest.mark.unit
@pytest.mark.parametrize("defect", ("scheduler", "optimizer"))
def test_allocation_neutral_resume_rejects_cadence_tampering_before_load_state(
    tmp_path,
    tokenizer,
    tiny_model,
    defect: str,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=89)[0]
    record = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=89,
        behavior_policy_id="teacher-demo-fixture",
    )
    source = TeacherDemoStateSource([_teacher_demo_attempt(record)])
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=0,
            world_size=1,
            global_batch_size=64,
            max_microbatch_size=4,
        ),
        state_source=source,
        supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=2,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / f"cadence-tamper-{defect}",
    )
    accelerator = FakeAccelerator()
    trainer._accelerator = accelerator
    payload = _valid_accelerate_resume_payload(trainer)
    if defect == "scheduler":
        payload["scheduler"]["last_epoch"] = 1
    else:
        payload["optimizer"]["state"] = {0: {"step": torch.tensor(1.0)}}
    checkpoint = (trainer.run_dir / "checkpoints" / "step-00000000.pt").absolute()
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError):
        trainer.resume(checkpoint)
    assert accelerator.load_state_calls == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "stop_reason",
    (
        "token_budget_exactly_consumed",
        "token_budget_exhausted_before_next_optimizer_update",
        "signal_at_optimizer_boundary",
        "max_steps_safety_limit",
    ),
)
def test_allocation_neutral_resume_accepts_only_nonterminal_periodic_checkpoints(
    tmp_path,
    tokenizer,
    tiny_model,
    stop_reason: str,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=92)[0]
    record = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=92,
        behavior_policy_id="teacher-demo-fixture",
    )
    source = TeacherDemoStateSource([_teacher_demo_attempt(record)])
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=0,
            world_size=1,
            global_batch_size=64,
            max_microbatch_size=4,
        ),
        state_source=source,
        supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=1 if stop_reason == "max_steps_safety_limit" else 2,
            token_budget=100,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / f"terminal-resume-{stop_reason}",
    )
    accelerator = FakeAccelerator()
    trainer._accelerator = accelerator
    for _ in range(16):
        prompts = trainer.prompt_scheduler.next_batch()
        trainer.state_source.get_batch(None, prompts, 0)
    _advance_optimizer_scheduler(trainer)
    trainer.global_step = 1
    trainer.token_budget.accepted_optimizer_updates = 1
    trainer.token_budget.consumed = 1
    if stop_reason == "token_budget_exactly_consumed":
        trainer.token_budget.consumed = trainer.token_budget.budget
    trainer.token_budget.stop_reason = stop_reason
    trainer.cumulative_counts.update(
        prompts_consumed=64.0,
        trajectories_generated=64.0,
        model_facing_input_tokens_processed=float(trainer.token_budget.consumed),
    )
    payload = _valid_accelerate_resume_payload(trainer)
    checkpoint = (trainer.run_dir / "checkpoints" / "terminal.pt").absolute()
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="non-terminal periodic checkpoint"):
        trainer.resume(checkpoint)
    assert accelerator.load_state_calls == 0


@pytest.mark.unit
def test_allocation_neutral_resume_rejects_alternate_accelerator_state_sibling(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=90)[0]
    record = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=90,
        behavior_policy_id="teacher-demo-fixture",
    )
    source = TeacherDemoStateSource([_teacher_demo_attempt(record)])
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=0,
            world_size=1,
            global_batch_size=64,
            max_microbatch_size=4,
        ),
        state_source=source,
        supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=2,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / "alternate-state-sibling",
    )
    accelerator = FakeAccelerator()
    trainer._accelerator = accelerator
    payload = _valid_accelerate_resume_payload(trainer)
    checkpoint = (trainer.run_dir / "checkpoints" / "step-00000000.pt").absolute()
    alternate_state_dir = checkpoint.with_name("other.accelerate")
    alternate_state_dir.mkdir()
    (alternate_state_dir / "state.bin").write_bytes(b"alternate")
    payload["accelerate_state_dir"] = str(alternate_state_dir)
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="differs from its checkpoint sibling"):
        trainer.resume(checkpoint)
    assert accelerator.load_state_calls == 0


@pytest.mark.unit
def test_allocation_neutral_resume_rejects_noncanonical_metadata_path(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=91)[0]
    record = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=91,
        behavior_policy_id="teacher-demo-fixture",
    )
    source = TeacherDemoStateSource([_teacher_demo_attempt(record)])
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler.for_allocation_neutral_rank(
            [example.example_id],
            ["prompt"],
            rank=0,
            world_size=1,
            global_batch_size=64,
            max_microbatch_size=4,
        ),
        state_source=source,
        supervisor=CanonicalSFTSupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=2,
            batch_partition_protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
        ),
        run_dir=tmp_path / "noncanonical-metadata",
    )
    checkpoint = (trainer.run_dir / "checkpoints" / "step-00000000.pt").absolute()
    torch.save({}, checkpoint)
    alias = checkpoint.with_name("metadata-alias.pt")
    alias.symlink_to(checkpoint)

    with pytest.raises(ValueError, match="path is not canonical"):
        trainer.resume(alias)
    with pytest.raises(ValueError, match="path is not canonical"):
        trainer.resume(Path("relative-checkpoint.pt"))


def _trainer(
    *,
    model: torch.nn.Module,
    source: Any,
    supervisor: VerifiedReplaySupervisor,
    config: TrainerConfig,
    run_dir: Path,
    prompt_id: str,
) -> FactorialTrainer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return FactorialTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler([prompt_id], ["prompt"], 1),
        state_source=source,
        supervisor=supervisor,
        config=config,
        run_dir=run_dir,
        evaluation_fn=lambda _: {
            "validation_accuracy": 0.0,
            "exact_proof_accuracy": 0.0,
            "format_validity": 0.0,
        },
    )


@pytest.mark.unit
def test_verified_replay_retries_in_trainer_and_counts_only_positive_tokens(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=4)[0]
    negative = make_trajectory(
        example,
        tokenizer,
        successful=False,
        policy_version=0,
        seed=1,
        behavior_policy_id="test",
    )
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=2,
        behavior_policy_id="test",
    )
    source = SequenceStateSource([[negative], [positive]])
    supervisor = VerifiedReplaySupervisor(
        tokenizer.pad_token_id,
        minimum_positives=1,
        retry_limit=2,
    )
    trainer = _trainer(
        model=tiny_model,
        source=source,
        supervisor=supervisor,
        config=TrainerConfig(max_steps=1),
        run_dir=tmp_path / "retry",
        prompt_id=example.example_id,
    )
    metric = trainer.train()[0]
    positive_tokens = sum(positive.response_token_mask)
    assert source.calls == 2
    assert metric["generated_trajectories"] == 2
    assert metric["successful_trajectories"] == 1
    assert metric["effective_positive_sequences"] == 1
    assert metric["effective_supervised_tokens"] == positive_tokens
    assert metric["reward_rate"] == pytest.approx(0.5)
    assert metric["retry_count"] == 1
    assert metric["prompts_consumed"] == 2
    assert metric["trajectories_generated"] == 2
    assert metric["supervised_response_tokens"] == positive_tokens


@pytest.mark.unit
def test_steps_per_round_and_gradient_accumulation_are_effective(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=6)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=3,
        behavior_policy_id="test",
    )
    source = SequenceStateSource([[positive]])
    supervisor = CountingReplaySupervisor(
        tokenizer.pad_token_id,
        minimum_positives=1,
        retry_limit=0,
    )
    trainer = _trainer(
        model=tiny_model,
        source=source,
        supervisor=supervisor,
        config=TrainerConfig(
            max_steps=3,
            steps_per_round=2,
            gradient_accumulation_steps=2,
        ),
        run_dir=tmp_path / "loop",
        prompt_id=example.example_id,
    )
    history = trainer.train()
    tokens_per_forward = sum(positive.response_token_mask)
    assert trainer.global_step == 3
    assert source.calls == 3
    assert supervisor.loss_calls == 6
    assert history[1]["trajectories_generated"] == 2
    assert history[-1]["prompts_consumed"] == 3
    assert history[-1]["trajectories_generated"] == 3
    assert history[-1]["supervised_response_tokens"] == tokens_per_forward * 6
    assert history[-1]["optimizer_updates"] == 3


@pytest.mark.unit
def test_current_policy_retries_and_accumulation_share_one_optimizer_boundary_version(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=66)[0]
    observed_versions: list[int] = []

    def generator(
        model: Any,
        prompts: PromptBatch,
        policy_version: int,
        sampling_request: SamplingRequest,
    ) -> list[TrajectoryRecord]:
        del model, prompts
        observed_versions.append(policy_version)
        return [
            make_trajectory(
                example,
                tokenizer,
                successful=len(observed_versions) != 1,
                policy_version=policy_version,
                seed=sampling_request.sampling_request_seed,
                behavior_policy_id="current-policy",
                sampling_cursor=sampling_request.cursors[0],
                sampling_protocol_id=sampling_request.sampling_protocol_id,
            )
        ]

    source = CurrentPolicyStateSource(generator, refresh_interval=1, max_policy_lag=0)
    trainer = _trainer(
        model=tiny_model,
        source=source,
        supervisor=VerifiedReplaySupervisor(
            tokenizer.pad_token_id,
            minimum_positives=1,
            retry_limit=1,
        ),
        config=TrainerConfig(
            max_steps=2,
            steps_per_round=1,
            gradient_accumulation_steps=2,
        ),
        run_dir=tmp_path / "current-policy-boundaries",
        prompt_id=example.example_id,
    )
    trainer.train()
    assert observed_versions == [1, 1, 1, 2, 2]
    assert source.state_dict()["refresh_boundary"] == "optimizer_update"
    assert source.state_dict()["retry_forces_refresh"] is False


@pytest.mark.unit
def test_verified_replay_exhaustion_is_explicit(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=8)[0]
    negative = make_trajectory(
        example,
        tokenizer,
        successful=False,
        policy_version=0,
        seed=4,
        behavior_policy_id="test",
    )
    source = SequenceStateSource([[negative]])
    trainer = _trainer(
        model=tiny_model,
        source=source,
        supervisor=VerifiedReplaySupervisor(
            tokenizer.pad_token_id,
            minimum_positives=1,
            retry_limit=2,
        ),
        config=TrainerConfig(max_steps=1),
        run_dir=tmp_path / "exhausted",
        prompt_id=example.example_id,
    )
    with pytest.raises(InsufficientPositiveTrajectories) as raised:
        trainer.train()
    assert raised.value.generated == 3
    assert raised.value.received == 0
    assert source.calls == 3


@pytest.mark.unit
def test_formal_training_requires_evaluation_callback_before_updates(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=10)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=5,
        behavior_policy_id="test",
    )
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with pytest.raises(ValueError, match="evaluation callback"):
        FactorialTrainer(
            model=tiny_model,
            optimizer=optimizer,
            scheduler=scheduler,
            prompt_scheduler=PromptScheduler([example.example_id], ["prompt"], 1),
            state_source=SequenceStateSource([[positive]]),
            supervisor=VerifiedReplaySupervisor(tokenizer.pad_token_id),
            config=TrainerConfig(max_steps=1, require_evaluation_metrics=True),
            run_dir=tmp_path / "formal",
        )


@pytest.mark.unit
def test_formal_evaluation_runs_at_frozen_intervals_and_final_step(
    tmp_path,
    tokenizer,
    tiny_model,
) -> None:  # type: ignore[no-untyped-def]
    example = build_smoke_examples(1, seed=12)[0]
    positive = make_trajectory(
        example,
        tokenizer,
        successful=True,
        policy_version=0,
        seed=6,
        behavior_policy_id="test",
    )
    calls: list[int] = []
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = FactorialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler=PromptScheduler([example.example_id], ["prompt"], 1),
        state_source=SequenceStateSource([[positive]]),
        supervisor=VerifiedReplaySupervisor(tokenizer.pad_token_id),
        config=TrainerConfig(
            max_steps=5,
            evaluation_every=2,
            require_evaluation_metrics=True,
        ),
        run_dir=tmp_path / "scheduled-eval",
        evaluation_fn=lambda _: (
            calls.append(trainer.global_step)
            or {
                "validation_accuracy": 0.5,
                "exact_proof_accuracy": 0.25,
                "format_validity": 1.0,
            }
        ),
    )
    history = trainer.train()
    assert calls == [2, 4, 5]
    assert history[0]["validation_accuracy"] is None
    assert history[1]["validation_accuracy"] == 0.5
    assert history[-1]["validation_accuracy"] == 0.5
