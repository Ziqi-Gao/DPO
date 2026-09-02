"""Reward-function contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class RewardFunction(Protocol):
    def __call__(self, prompts: Sequence[str], completions: Sequence[str], **kwargs: Any) -> list[float]: ...
