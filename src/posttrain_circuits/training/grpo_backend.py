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
