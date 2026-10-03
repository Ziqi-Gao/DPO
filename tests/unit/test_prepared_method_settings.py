"""Actual optimizer behavior and official TRL CPU construction, never a GPU claim."""

from __future__ import annotations

import copy

import pytest
import torch

from posttrain_circuits.learning.training.prepared_method_settings import (
    build_prepared_grpo_backend,
    build_prepared_opd_optimization,
)


def gradient_step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    assert torch.isfinite(loss)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    lr_used = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
    return lr_used, scheduler.get_last_lr()[0]


def test_opd_real_adamw_constant_lr_and_state_restore_continuation():
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = build_prepared_opd_optimization(model.parameters())
    assert scheduler.optimizer is optimizer and not optimizer.state
    assert [id(value) for value in optimizer.param_groups[0]["params"]] == [
        id(value) for value in model.parameters()
    ]
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
    assert [gradient_step(model, optimizer, scheduler) for _ in range(2)] == [(5e-5, 5e-5)] * 2
    assert all(int(state["step"]) == 2 for state in optimizer.state.values())

    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(model.state_dict())
    resumed_optimizer, resumed_scheduler = build_prepared_opd_optimization(restored.parameters())
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    resumed_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    assert gradient_step(model, optimizer, scheduler) == gradient_step(
        restored, resumed_optimizer, resumed_scheduler
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.parameters(), restored.parameters(), strict=True)
    )
    assert all(int(state["step"]) == 3 for state in resumed_optimizer.state.values())
    assert resumed_scheduler.last_epoch == scheduler.last_epoch == 3


@pytest.mark.parametrize("kind", ["empty", "partially_frozen", "bf16", "duplicate"])
def test_opd_refuses_silent_partial_or_rounded_parameter_training(kind):
    model = torch.nn.Linear(3, 2)
    parameters = list(model.parameters())
    if kind == "empty":
        parameters = []
    elif kind == "partially_frozen":
        parameters[0].requires_grad_(False)
    elif kind == "bf16":
        model.bfloat16()
    else:
        parameters.append(parameters[0])
    with pytest.raises(ValueError, match="trainable FP32|duplicate parameter"):
        build_prepared_opd_optimization(parameters)


def test_real_official_trl_build_optimizer_and_120_step_linear_schedule(tmp_path):
    # These imports deliberately do not skip or fake missing official packages.
    # Run with the existing project .venv and CUDA_VISIBLE_DEVICES=''.
    import importlib.metadata

    from datasets import Dataset
    from trl import GRPOTrainer

    from posttrain_circuits.utils.tiny_model import build_tiny_qwen3, build_tiny_tokenizer

    assert importlib.metadata.version("trl") == "0.22.2"
    model, tokenizer = build_tiny_qwen3(), build_tiny_tokenizer()
    dataset = Dataset.from_list([{"prompt": "FACTS A RULES A -> Q QUERY Q"}] * 8)

    def reward(completions, **kwargs):
        del kwargs
        return [0.0] * len(completions)

    backend = build_prepared_grpo_backend(seed=42, use_cpu=True)
    trainer = backend.build(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        train_dataset=dataset,
        output_dir=str(tmp_path / "official-trainer"),
    )
    assert isinstance(trainer, GRPOTrainer)
    assert trainer.args.use_cpu is True and trainer.accelerator.num_processes == 1
    # Honest W1 constructor evidence: 8x8=64. The future W4 path is 256, not
    # observed here; the helper retains its prospective W4 reservation 589824.
    assert trainer.args.per_device_train_batch_size == 8
    assert trainer.args.gradient_accumulation_steps == 8
    assert trainer.args.generation_batch_size == 64 and trainer.args.num_generations == 8
    assert trainer.args.max_steps == trainer.registered_max_steps == 120
    assert trainer.args.seed == trainer.args.data_seed == 42
    assert trainer.args.max_prompt_length == 2048
    assert trainer.args.max_completion_length == trainer.generation_config.max_new_tokens == 256
    assert trainer.generation_config.temperature == 0.7
    assert trainer.generation_config.top_p == 0.8
    assert trainer.generation_config.top_k == 20
    assert trainer.generation_config.min_p == 0.0
    assert trainer.args.learning_rate == 5e-5
    assert trainer.args.lr_scheduler_type.value == "linear"
    assert trainer.args.warmup_steps == 0 and trainer.args.warmup_ratio == 0.0
    assert (
        trainer.args.loss_type == "dapo" and trainer.args.scale_rewards == "none" and trainer.args.beta == 0.0
    )
    assert trainer.token_budget_callback.settings is backend.settings
    assert trainer.token_budget_callback.consumed == backend.settings.initial_token_budget_consumed == 0
    assert backend.settings.initial_optimizer_steps == 0
    assert trainer.token_budget_callback.settings.token_budget == 2_000_000
    assert backend.settings.reserved_tokens_per_update == 589824
    assert trainer.optimizer is None and trainer.lr_scheduler is None
    trainer.create_optimizer_and_scheduler(num_training_steps=120)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.lr_scheduler.optimizer is trainer.optimizer
    # Actual Trainer param groups override torch.optim.AdamW's unrelated
    # defaults['weight_decay']; inspect the groups that really update tensors.
    for group in trainer.optimizer.param_groups:
        assert group["lr"] == 5e-5 and group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8 and group["weight_decay"] == 0.0
    observed = [gradient_step(model, trainer.optimizer, trainer.lr_scheduler) for _ in range(2)]
    assert observed[0] == pytest.approx((5e-5, 5e-5 * 119 / 120))
    assert observed[1] == pytest.approx((5e-5 * 119 / 120, 5e-5 * 118 / 120))
    assert all(int(state["step"]) == 2 for state in trainer.optimizer.state.values())
    assert not torch.cuda.is_initialized()
    # Loss above is a local parameter objective, not GRPO rollout/reward training.


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"seed": -1}, {"seed": 1, "use_cpu": 1}])
def test_grpo_runtime_argument_types_rejected_before_build(kwargs):
    with pytest.raises(ValueError):
        build_prepared_grpo_backend(**kwargs)
