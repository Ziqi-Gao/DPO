"""Configuration-driven state-source and supervision factories."""

from __future__ import annotations

from typing import Any

from posttrain_circuits.core.types import StateSource, Supervisor, TrajectoryRecord
from posttrain_circuits.rollout.base import TrajectoryGenerator
from posttrain_circuits.rollout.current_policy import CurrentPolicyStateSource
from posttrain_circuits.rollout.fixed_bank import FixedBankStateSource
from posttrain_circuits.rollout.teacher_demo import TeacherDemoStateSource
from posttrain_circuits.supervision.hard_teacher import HardTeacherSupervisor
from posttrain_circuits.supervision.soft_teacher import SoftTeacherSupervisor
from posttrain_circuits.supervision.verified_replay import VerifiedReplaySupervisor
from posttrain_circuits.training.canonical_sft import CanonicalSFTSupervisor


def build_supervisor(config: dict[str, Any], *, pad_token_id: int) -> Supervisor:
    """Build the exact configured objective without consulting experiment-name strings."""
    name = str(config["name"])
    if name == "hard_teacher":
        return HardTeacherSupervisor(pad_token_id)
    if name == "soft_teacher":
        return SoftTeacherSupervisor(
            pad_token_id,
            topk_mode=str(config.get("topk_mode", "renormalized")),
        )
    if name == "verified_replay":
        return VerifiedReplaySupervisor(
            pad_token_id,
            normalization=str(config.get("normalization", "sequence")),
            minimum_positives=int(config.get("minimum_positives", 1)),
            retry_limit=int(config.get("retry_limit", 3)),
        )
    if name == "canonical_sft":
        return CanonicalSFTSupervisor(
            pad_token_id,
            normalization=str(config.get("normalization", "sequence")),
        )
    raise ValueError(f"unsupported supervision configuration {name!r}")


def build_state_source(
    config: dict[str, Any],
    *,
    fixed_bank: list[TrajectoryRecord] | None = None,
    current_generator: TrajectoryGenerator | None = None,
    teacher_demos: list[TrajectoryRecord] | None = None,
    seed: int = 0,
) -> StateSource:
    """Build the configured source and require its corresponding data dependency."""
    name = str(config["name"])
    if name == "fixed_bank":
        if fixed_bank is None:
            raise ValueError("fixed_bank state source requires an immutable rollout bank")
        return FixedBankStateSource(fixed_bank)
    if name == "current_policy":
        if current_generator is None:
            raise ValueError("current_policy state source requires a trajectory generator")
        return CurrentPolicyStateSource(
            current_generator,
            refresh_interval=int(config.get("refresh_interval", 1)),
            max_policy_lag=int(config.get("max_policy_lag", 0)),
            seed=seed,
        )
    if name == "teacher_demo":
        if teacher_demos is None:
            raise ValueError("teacher_demo state source requires a verified demonstration store")
        return TeacherDemoStateSource(teacher_demos)
    raise ValueError(f"unsupported state-source configuration {name!r}")
