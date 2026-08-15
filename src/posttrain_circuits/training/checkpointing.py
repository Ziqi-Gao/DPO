"""Atomic checkpoints including optimizer, scheduler, RNG, and rollout state."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.core.seeding import RNGState


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    prompt_scheduler_state: dict[str, Any],
    global_step: int,
    policy_version: int,
    online_rollout_round: int,
    resolved_config: dict[str, Any],
    manifest_hashes: dict[str, str],
    state_source_state: dict[str, Any] | None = None,
    trainer_state: dict[str, Any] | None = None,
    accelerator_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    git_commit: str = "unavailable",
    dependency_versions: dict[str, str] | None = None,
    resume_ancestry: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler_state,
        "accelerator": accelerator_state,
        "rng": RNGState.capture().as_dict(),
        "prompt_scheduler": prompt_scheduler_state,
        "state_source": state_source_state or {},
        "trainer_state": trainer_state or {},
        "global_step": global_step,
        "policy_version": policy_version,
        "online_rollout_round": online_rollout_round,
        "resolved_config": resolved_config,
        "manifest_hashes": manifest_hashes,
        "git_commit": git_commit,
        "dependency_versions": dependency_versions or {"torch": torch.__version__},
        "resume_ancestry": list(resume_ancestry or []),
        "torch_version": torch.__version__,
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a generic torch payload from one designated process."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    RNGState(**payload["rng"]).restore()
    return payload
