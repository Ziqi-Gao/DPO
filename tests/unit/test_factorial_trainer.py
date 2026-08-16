from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from posttrain_circuits.core.types import PromptBatch, TrajectoryBatch, TrajectoryRecord
from posttrain_circuits.supervision.verified_replay import (
    InsufficientPositiveTrajectories,
    VerifiedReplaySupervisor,
)
from posttrain_circuits.training.factorial_trainer import FactorialTrainer, TrainerConfig
from posttrain_circuits.training.schedules import PromptScheduler
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


def _trainer(
    *,
    model: torch.nn.Module,
    source: SequenceStateSource,
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
