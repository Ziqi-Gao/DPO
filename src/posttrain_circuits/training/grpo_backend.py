"""Canonical GRPO adapter using the official TRL implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GrpoSettings:
    beta: float = 0.0
    num_generations: int = 8
    temperature: float = 1.0
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

        config = GRPOConfig(
            output_dir=output_dir,
            beta=self.settings.beta,
            num_generations=self.settings.num_generations,
            temperature=self.settings.temperature,
            max_completion_length=self.settings.max_completion_length,
            loss_type=self.settings.loss_type,
            scale_rewards=scale_rewards,
            gradient_accumulation_steps=self.settings.gradient_accumulation_steps,
            max_steps=self.settings.max_steps,
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
        )
        return GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=config,
            train_dataset=train_dataset,
            **kwargs,
        )
