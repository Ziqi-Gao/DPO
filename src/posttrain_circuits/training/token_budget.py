"""Global, distributed, optimizer-boundary token-budget accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

TOKEN_BUDGET_UNIT = "global_nonpadding_model_input_tokens"


def distributed_token_sum(local_tokens: int) -> int:
    if local_tokens < 0:
        raise ValueError("local token count cannot be negative")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_tokens
    backend = torch.distributed.get_backend()
    device = torch.device("cpu")
    if backend == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
    value = torch.tensor(local_tokens, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return int(value.cpu())


def distributed_consensus_token_count(global_tokens: int) -> int:
    """Assert that a framework-provided global counter agrees on every rank."""

    if global_tokens < 0:
        raise ValueError("global token count cannot be negative")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return global_tokens
    backend = torch.distributed.get_backend()
    device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
    minimum = torch.tensor(global_tokens, dtype=torch.int64, device=device)
    maximum = minimum.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    if int(minimum.cpu()) != int(maximum.cpu()):
        raise RuntimeError("framework global token counter differs across distributed ranks")
    return int(maximum.cpu())


@dataclass
class TokenBudgetState:
    budget: int
    unit: str = TOKEN_BUDGET_UNIT
    consumed: int = 0
    accepted_optimizer_updates: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("token_budget must be positive")
        if self.unit != TOKEN_BUDGET_UNIT:
            raise ValueError(f"unsupported token budget unit: {self.unit}")
        if not 0 <= self.consumed <= self.budget:
            raise ValueError("token budget consumed value is outside the registered budget")

    @property
    def remaining(self) -> int:
        return self.budget - self.consumed

    def reserve_optimizer_update(self, local_tokens: int) -> tuple[bool, int]:
        """Atomically admit one complete optimizer window across every rank."""

        global_tokens = distributed_token_sum(local_tokens)
        if global_tokens < 1:
            raise ValueError("an optimizer update must process at least one model-facing token")
        if global_tokens > self.remaining:
            self.stop_reason = "token_budget_exhausted_before_next_optimizer_update"
            return False, global_tokens
        self.consumed += global_tokens
        self.accepted_optimizer_updates += 1
        if self.consumed == self.budget:
            self.stop_reason = "token_budget_exactly_consumed"
        return True, global_tokens

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        restored = TokenBudgetState(
            budget=int(payload["budget"]),
            unit=str(payload["unit"]),
            consumed=int(payload["consumed"]),
            accepted_optimizer_updates=int(payload.get("accepted_optimizer_updates", 0)),
            stop_reason=payload.get("stop_reason"),
        )
        if restored.budget != self.budget or restored.unit != self.unit:
            raise ValueError("resume token budget differs from the frozen run configuration")
        self.consumed = restored.consumed
        self.accepted_optimizer_updates = restored.accepted_optimizer_updates
        self.stop_reason = restored.stop_reason


def maximum_grpo_tokens_per_update(
    *,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    max_prompt_length: int,
    max_completion_length: int,
) -> int:
    values = (
        world_size,
        per_device_batch_size,
        gradient_accumulation_steps,
        max_prompt_length,
        max_completion_length,
    )
    if any(value < 1 for value in values):
        raise ValueError("GRPO token reservation inputs must all be positive")
    return (
        world_size
        * per_device_batch_size
        * gradient_accumulation_steps
        * (max_prompt_length + max_completion_length)
    )
