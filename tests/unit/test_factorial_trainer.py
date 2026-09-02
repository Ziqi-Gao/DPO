from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from posttrain_circuits.artifacts.checkpoints import (
    accelerator_state_file_hashes,
    save_checkpoint as artifact_save_checkpoint,
)
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import publish_path_no_clobber
from posttrain_circuits.cli.finalize_pilot_training import (
    _validate_factorial_accelerator_state,
    _validate_factorial_rank_states,
)
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.learning.contracts import PromptBatch, SamplingRequest, TrajectoryBatch
from posttrain_circuits.learning.state_sources.current_policy import CurrentPolicyStateSource
from posttrain_circuits.learning.supervision.verified_replay import (
    InsufficientPositiveTrajectories,
    VerifiedReplaySupervisor,
)
from posttrain_circuits.learning.training.factorial_trainer import (
    FactorialTrainer,
    TrainerConfig,
    _checkpoint_publication_preflight_report,
    _extend_resume_ancestry,
    _validate_checkpoint_publication_preflight_reports,
    _validate_rank_shard_hashes,
    _validate_rank_state_mappings,
    _validate_resume_world_size,
)
from posttrain_circuits.learning.training.schedules import PromptScheduler
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


class CountingReplaySupervisor(VerifiedReplaySupervisor):
    def __init__(self, pad_token_id: int, **kwargs: Any) -> None:
        super().__init__(pad_token_id, **kwargs)
        self.loss_calls = 0

    def compute_loss(self, model: Any, batch):  # type: ignore[no-untyped-def]
        self.loss_calls += 1
        return super().compute_loss(model, batch)


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
