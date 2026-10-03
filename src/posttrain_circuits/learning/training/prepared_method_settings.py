"""Prospective lower-LR optimization components for explicit method adapters.

These helpers do not load/admit models, alter existing CLIs, launch jobs, or
certify W4 execution. OPD's W4 microbatch4/accumulation4 wiring remains the caller's
responsibility. GRPO settings preserve the existing W4 batch8/accumulation8 path.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from posttrain_circuits.learning.training.grpo_backend import GrpoSettings, TrlGrpoBackend
from posttrain_circuits.learning.training.optimizer import build_adamw
from posttrain_circuits.learning.training.token_budget import maximum_grpo_tokens_per_update


def build_prepared_opd_optimization(
    parameters: Iterable[torch.nn.Parameter],
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Create fresh FP32 AdamW and its constant 5e-5 scheduler.

    Preserve the existing OPD betas (.9, .95), epsilon (default 1e-8), and weight
    decay 0. The returned scheduler owns this exact optimizer; recording an LR in
    a TrainerConfig alone would not configure the optimizer used for training.
    Checks cover only the supplied iterable; the caller must establish complete
    model coverage. Parameter objects are retained without copying or casting.
    """
    parameters = tuple(parameters)
    if not parameters or any(
        not isinstance(parameter, torch.nn.Parameter)
        or not parameter.requires_grad
        or parameter.dtype != torch.float32
        for parameter in parameters
    ):
        raise ValueError("prepared OPD requires nonempty, trainable FP32 parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("prepared OPD received duplicate parameter objects")
    optimizer = build_adamw(parameters, learning_rate=5e-5, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    return optimizer, scheduler


def build_prepared_grpo_backend(*, seed: int, use_cpu: bool = False) -> TrlGrpoBackend:
    """Return the unchanged backend with an explicit 5e-5, 120-step configuration.

    The official backend owns its linear zero-warmup schedule and AdamW defaults
    (.9, .999, epsilon 1e-8, actual parameter-group weight decay 0). Do not replace
    those defaults with OPD's beta2=.95. ``use_cpu`` permits real CPU constructor
    tests; their actual W1 batch is not W4 execution evidence. The prospective W4
    token reservation remains conservative and is not a runtime allocation check.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("GRPO seed must be a nonnegative integer")
    if type(use_cpu) is not bool:
        raise ValueError("GRPO use_cpu must be a boolean")
    settings = GrpoSettings(
        learning_rate=5e-5,
        max_steps=120,
        token_budget=2_000_000,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        num_generations=8,
        max_prompt_length=2048,
        max_completion_length=256,
        reserved_tokens_per_update=maximum_grpo_tokens_per_update(
            world_size=4,
            per_device_batch_size=8,
            gradient_accumulation_steps=8,
            max_prompt_length=2048,
            max_completion_length=256,
        ),
        beta=0.0,
        loss_type="dapo",
        scale_rewards=False,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        gradient_checkpointing=True,
        seed=seed,
        use_cpu=use_cpu,
    )
    return TrlGrpoBackend(settings)
