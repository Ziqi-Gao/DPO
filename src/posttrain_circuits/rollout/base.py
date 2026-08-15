"""Rollout helpers."""

from __future__ import annotations

from collections.abc import Callable

from posttrain_circuits.core.types import PromptBatch, TrajectoryRecord

TrajectoryGenerator = Callable[[object, PromptBatch, int, int], list[TrajectoryRecord]]
