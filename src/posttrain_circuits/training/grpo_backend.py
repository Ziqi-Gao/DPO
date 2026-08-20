"""Canonical GRPO adapter using the official TRL implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import TrainerCallback

from posttrain_circuits.training.token_budget import TOKEN_BUDGET_UNIT, distributed_token_sum


@dataclass(frozen=True)
class GrpoSettings:
    beta: float = 0.0
    num_generations: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    max_completion_length: int = 128
    loss_type: str = "dapo"
    scale_rewards: bool | str = False
    gradient_accumulation_steps: int = 1
    max_steps: int = 2
    per_device_train_batch_size: int = 4
    learning_rate: float = 5e-4
    max_prompt_length: int = 128
    use_cpu: bool = False
    gradient_checkpointing: bool = False
    seed: int = 0
    token_budget: int = 1024
    token_budget_unit: str = TOKEN_BUDGET_UNIT
    reserved_tokens_per_update: int = 1
    initial_token_budget_consumed: int = 0
    initial_optimizer_steps: int = 0


class GrpoTokenBudgetCallback(TrainerCallback):
    """Stop TRL at a boundary whose worst-case next update cannot fit."""

    def __init__(self, settings: GrpoSettings, processed_tokens: list[int] | None = None) -> None:
        if settings.token_budget < 1 or settings.token_budget_unit != TOKEN_BUDGET_UNIT:
            raise ValueError("GRPO requires the registered global token budget unit")
        if settings.reserved_tokens_per_update < 1:
            raise ValueError("GRPO reserved_tokens_per_update must be positive")
        if not 0 <= settings.initial_token_budget_consumed <= settings.token_budget:
            raise ValueError("GRPO resume token consumption is outside the registered budget")
        self.settings = settings
        self.consumed = settings.initial_token_budget_consumed
        self.processed_tokens = processed_tokens if processed_tokens is not None else [0]
        self.stop_reason: str | None = None

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        local_delta = self.processed_tokens[0]
        self.processed_tokens[0] = 0
        if local_delta < 1:
            raise RuntimeError("GRPO optimizer update has no measured model-facing tokens")
        global_delta = distributed_token_sum(local_delta)
        if global_delta > self.settings.reserved_tokens_per_update:
            raise RuntimeError("GRPO actual distributed tokens exceeded its conservative reservation")
        self.consumed += global_delta
        if self.consumed > self.settings.token_budget:
            raise RuntimeError("GRPO exceeded its registered global token budget")
        if self.settings.token_budget - self.consumed < self.settings.reserved_tokens_per_update:
            control.should_training_stop = True
            self.stop_reason = "token_budget_reserved_boundary"
        elif int(getattr(state, "global_step", 0)) >= self.settings.max_steps:
            self.stop_reason = "max_steps_safety_limit"
        return control

    def state_dict(self) -> dict[str, Any]:
        return {
            "budget": self.settings.token_budget,
            "unit": self.settings.token_budget_unit,
            "consumed": self.consumed,
            "reserved_tokens_per_update": self.settings.reserved_tokens_per_update,
            "stop_reason": self.stop_reason or "max_steps_safety_limit",
        }


def resolve_grpo_batch_contract(settings: GrpoSettings, *, world_size: int) -> dict[str, int | str]:
    """Resolve the pinned TRL 0.22 generation-batch contract.

    In TRL 0.22.2 the default ``steps_per_generation`` equals gradient
    accumulation, and ``generation_batch_size`` is per-device batch times
    process count times that value. That generation batch, not merely the
    per-device microbatch, must be divisible by ``num_generations``.
    """

    if world_size < 1:
        raise ValueError("GRPO world_size must be positive")
    if settings.num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    global_micro_batch = world_size * settings.per_device_train_batch_size
    effective_global_batch = global_micro_batch * settings.gradient_accumulation_steps
    generation_batch_size = effective_global_batch
    if generation_batch_size % settings.num_generations:
        raise ValueError(
            "TRL generation_batch_size must be divisible by num_generations: "
            f"{generation_batch_size} % {settings.num_generations} != 0"
        )
    return {
        "trl_version_contract": "0.22.2-default-steps_per_generation",
        "world_size": world_size,
        "per_device_train_batch_size": settings.per_device_train_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "global_micro_batch_size": global_micro_batch,
        "effective_global_batch_size": effective_global_batch,
        "generation_batch_size": generation_batch_size,
        "steps_per_generation": settings.gradient_accumulation_steps,
        "num_generations": settings.num_generations,
        "groups_per_update": generation_batch_size // settings.num_generations,
    }


class TrlGrpoBackend:
    def __init__(self, settings: GrpoSettings) -> None:
        self.settings = settings

    def build(
        self,
        *,
        model: Any,
        reward_funcs: Any,
        train_dataset: Any,
        output_dir: str,
        **kwargs: Any,
    ) -> Any:
        try:
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as error:
            raise RuntimeError("canonical GRPO requires the 'rl' extra: pip install -e '.[rl]'") from error
        scale_rewards = self.settings.scale_rewards
        if isinstance(scale_rewards, bool):
            scale_rewards = "group" if scale_rewards else "none"

        remaining = self.settings.token_budget - self.settings.initial_token_budget_consumed
        effective_max_steps = self.settings.max_steps
        if (
            self.settings.initial_optimizer_steps >= self.settings.max_steps
            or remaining < self.settings.reserved_tokens_per_update
        ):
            raise ValueError("GRPO token budget cannot admit one complete optimizer update")
        config = GRPOConfig(
            output_dir=output_dir,
            beta=self.settings.beta,
            num_generations=self.settings.num_generations,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
            min_p=self.settings.min_p,
            max_completion_length=self.settings.max_completion_length,
            loss_type=self.settings.loss_type,
            scale_rewards=scale_rewards,
            gradient_accumulation_steps=self.settings.gradient_accumulation_steps,
            max_steps=effective_max_steps,
            report_to="none",
            per_device_train_batch_size=self.settings.per_device_train_batch_size,
            learning_rate=self.settings.learning_rate,
            max_prompt_length=self.settings.max_prompt_length,
            use_cpu=self.settings.use_cpu,
            bf16=not self.settings.use_cpu,
            fp16=False,
            logging_steps=1,
            gradient_checkpointing=self.settings.gradient_checkpointing,
            seed=self.settings.seed,
            data_seed=self.settings.seed,
            optim="adamw_torch",
            save_strategy="no",
            dataloader_pin_memory=not self.settings.use_cpu,
            include_num_input_tokens_seen=False,
        )
        processed_tokens = [0]

        class TokenBudgetGRPOTrainer(GRPOTrainer):  # type: ignore[misc, valid-type]
            token_budget_callback: GrpoTokenBudgetCallback
            registered_max_steps: int
            token_budget_effective_max_steps: int

            def compute_loss(self, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
                prompt_mask = inputs.get("prompt_mask")
                completion_mask = inputs.get("completion_mask")
                if prompt_mask is None or completion_mask is None:
                    raise RuntimeError("GRPO token accounting requires prompt and completion masks")
                processed_tokens[0] += int(prompt_mask.sum().item() + completion_mask.sum().item())
                return GRPOTrainer.compute_loss(self, model, inputs, *args, **kwargs)  # type: ignore[misc]

        token_budget_callback = GrpoTokenBudgetCallback(self.settings, processed_tokens)
        trainer = TokenBudgetGRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=config,
            train_dataset=train_dataset,
            callbacks=[token_budget_callback],
            **kwargs,
        )
        trainer.token_budget_callback = token_budget_callback
        trainer.registered_max_steps = self.settings.max_steps
        trainer.token_budget_effective_max_steps = effective_max_steps
        return trainer
