"""Shared-state local-fork bundles, exact restoration, and matching."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import publish_torch_once
from posttrain_circuits.artifacts.compatibility import (
    GENERATOR_VERSION,
    LABEL_SEMANTICS,
    ROLLOUT_GENERATION_VERSION,
)
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.experiments.protocols.local_fork import LOCAL_FORK_SPEC
from posttrain_circuits.learning.collation import collate_trajectories
from posttrain_circuits.learning.contracts import PromptBatch, TrajectoryBatch
from posttrain_circuits.learning.primitives import LossOutput, SupervisionBatch
from posttrain_circuits.learning.supervision.base import Supervisor
from posttrain_circuits.learning.supervision.hard_teacher import HardTeacherSupervisor
from posttrain_circuits.learning.supervision.soft_teacher import SoftTeacherSupervisor
from posttrain_circuits.learning.supervision.verified_replay import (
    VerifiedReplaySupervisor,
)


@dataclass(frozen=True)
class ForkBundleManifest:
    format_version: int
    protocol_id: str
    bundle_id: str
    manifest_sha256: str
    local_fork_spec_sha256: str
    source_method_id: str
    source_seed: int
    model_spec_hash: str
    model_parameter_names_hash: str
    optimizer_parameter_names_hash: str
    trainable_parameter_names_hash: str
    input_evidence_hash: str
    optimizer_class: str
    scheduler_class: str
    checkpoint_hash: str
    optimizer_hash: str
    optimizer_moment_hash: str
    scheduler_hash: str
    rng_hash: str
    prompt_hash: str
    trajectory_hash: str
    probe_input_hash: str
    probe_attention_mask_hash: str
    probe_output_hash: str
    manifest_hashes: dict[str, str]
    formal_binding: dict[str, Any]
    policy_version: int
    group_membership_hash: str
    minimum_group_size: int
    branch_ids: tuple[str, ...]
    horizons: tuple[int, ...]
    primary_horizon: int
    require_within_group_reward_variance: bool
    require_frozen_group_advantages: bool
    require_old_policy_logprobs: bool
    clip_epsilon: float
    divergence: str
    probe_kl_mask_protocol: str
    primary_comparison_axis: str
    secondary_comparison_axes: tuple[str, ...]
    output_kl_relative_tolerance: float
    maximum_calibration_rounds: int
    calibration_minimum_learning_rate: float
    calibration_maximum_scale: float
    calibration_numeric_tolerance: float
    out_of_tolerance_action: str


_BRANCH_CHECKPOINT_FORMAT_VERSION = 2

_BRANCH_CHECKPOINT_FIELDS = {
    "format_version",
    "phase",
    "bundle_id",
    "bundle_manifest_sha256",
    "local_fork_protocol_id",
    "local_fork_spec_sha256",
    "branch",
    "horizon",
    "comparison_mode",
    "calibration_round",
    "learning_rate",
    "learning_rate_override",
    "pad_token_id",
    "model",
    "model_parameter_names",
    "trainable_parameter_names",
    "optimizer",
    "optimizer_parameter_names",
    "scheduler",
    "rng",
    "prompts",
    "trajectories",
    "probe_input_ids",
    "probe_attention_mask",
    "pre_update_outputs",
    "observed_probe_outputs",
    "optimizer_steps",
    "loss_evidence",
    "initial_hashes",
    "state_hashes",
}

_CHECKPOINT_EVIDENCE_FIELDS = {
    "path",
    "sha256",
    "payload_state_sha256",
    "phase",
    "state_hashes",
}

_LOSS_EVIDENCE_FIELDS = {
    "schema_version",
    "phase",
    "objective",
    "step_losses",
    "step_metrics",
    "final_loss",
}

_INPUT_EVIDENCE_FIELDS = {
    "schema_version",
    "local_fork_protocol_id",
    "local_fork_spec_sha256",
    "generator_version",
    "label_semantics",
    "rollout_generation_version",
    "probe_kl_mask_protocol",
    "trajectory_bank_manifest_path",
    "trajectory_bank_manifest_file_sha256",
    "trajectory_bank_content_sha256",
    "prompt_manifest_path",
    "prompt_manifest_file_sha256",
    "prompt_manifest_payload_sha256",
    "probe_manifest_path",
    "probe_manifest_file_sha256",
    "probe_manifest_payload_sha256",
    "selected_trajectory_ids",
    "selected_trajectory_hashes",
    "selected_trajectories_sha256",
    "probe_input_hash",
    "probe_attention_mask_hash",
    "pre_update_output_hash",
}

_PROMPT_MANIFEST_FIELDS = {
    "format_version",
    "local_fork_protocol_id",
    "local_fork_spec_sha256",
    "generator_version",
    "label_semantics",
    "rollout_generation_version",
    "prompt_protocol",
    "prompt_ids",
    "prompt_texts",
    "trajectory_ids",
    "generation_groups",
    "trajectory_store_hash",
    "group_membership_hash",
    "sha256",
}

_PROBE_MANIFEST_FIELDS = {
    "format_version",
    "local_fork_protocol_id",
    "local_fork_spec_sha256",
    "generator_version",
    "label_semantics",
    "rollout_generation_version",
    "prompt_protocol",
    "input_ids",
    "attention_mask",
    "prompt_ids",
    "trajectory_ids",
    "trajectory_store_hash",
    "tokenizer_revision",
    "group_membership_hash",
    "probe_kl_mask_protocol",
    "sha256",
}


@dataclass
class RestoredFork:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    prompts: PromptBatch
    trajectories: TrajectoryBatch
    probe_input_ids: torch.Tensor
    probe_attention_mask: torch.Tensor
    pre_update_outputs: torch.Tensor
    initial_hashes: dict[str, str]


_ADAMW_BASE_STATE_FIELDS = frozenset({"step", "exp_avg", "exp_avg_sq"})


def _finite_tensor(value: torch.Tensor, *, name: str) -> None:
    if value.is_complex() or not torch.is_floating_point(value):
        raise ValueError(f"{name} must use a real floating dtype")
    if not bool(torch.isfinite(value.detach().float()).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def _step_value(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if (
            value.numel() != 1
            or value.is_complex()
            or not torch.is_floating_point(value)
            or value.dtype != torch.float32
        ):
            raise ValueError(f"{name} must be one float32 scalar tensor")
        observed = float(value.detach().cpu().item())
    else:
        raise ValueError(f"{name} must be one real floating scalar tensor")
    if not math.isfinite(observed) or observed < 1 or not observed.is_integer():
        raise ValueError(f"{name} must be a finite positive integer-valued scalar")
    return int(observed)


def _validate_adamw_optimizer_contract(
    optimizer_state: Any,
    optimizer_parameter_names: Any,
    *,
    model_state: Any,
    model_parameter_names: Any,
    trainable_parameter_names: Any,
    context: str,
) -> dict[str, int]:
    """Validate the exact full-parameter AdamW reference/name/moment bijection."""

    if not isinstance(optimizer_state, dict) or set(optimizer_state) != {
        "state",
        "param_groups",
    }:
        raise ValueError(f"{context} optimizer must contain exactly state/param_groups")
    if not isinstance(model_state, dict):
        raise ValueError(f"{context} model state must be a mapping")
    if not isinstance(model_parameter_names, list) or not all(
        isinstance(name, str) and name for name in model_parameter_names
    ):
        raise ValueError(f"{context} model parameter names are malformed")
    if len(model_parameter_names) != len(set(model_parameter_names)) or any(
        name not in model_state or not isinstance(model_state[name], torch.Tensor)
        for name in model_parameter_names
    ):
        raise ValueError(f"{context} model parameter names are not a one-to-one state mapping")
    if not isinstance(trainable_parameter_names, list) or not all(
        isinstance(name, str) and name for name in trainable_parameter_names
    ):
        raise ValueError(f"{context} trainable parameter names are malformed")
    if (
        len(trainable_parameter_names) != len(set(trainable_parameter_names))
        or any(name not in model_parameter_names for name in trainable_parameter_names)
    ):
        raise ValueError(f"{context} trainable parameter contract is not one-to-one")
    trainable_set = set(trainable_parameter_names)
    canonical_trainable_order = [
        name for name in model_parameter_names if name in trainable_set
    ]
    if canonical_trainable_order != trainable_parameter_names:
        raise ValueError(
            f"{context} trainable parameters are not in canonical model order"
        )
    groups = optimizer_state["param_groups"]
    states = optimizer_state["state"]
    if not isinstance(groups, list) or not groups or not isinstance(states, dict):
        raise ValueError(f"{context} optimizer groups/state are malformed")
    if not isinstance(optimizer_parameter_names, list) or len(
        optimizer_parameter_names
    ) != len(groups):
        raise ValueError(f"{context} optimizer parameter-name groups are misaligned")

    reference_to_name: dict[Any, str] = {}
    reference_amsgrad: dict[Any, bool] = {}
    flattened_names: list[str] = []
    for group_index, (group, names) in enumerate(
        zip(groups, optimizer_parameter_names, strict=True)
    ):
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError(f"{context} optimizer group {group_index} is malformed")
        references = group["params"]
        if not isinstance(names, list) or len(names) != len(references) or not all(
            isinstance(name, str) and name for name in names
        ):
            raise ValueError(f"{context} optimizer group {group_index} name mapping changed")
        amsgrad = group.get("amsgrad")
        if type(amsgrad) is not bool:
            raise ValueError(f"{context} optimizer group {group_index} lacks boolean amsgrad")
        for reference, name in zip(references, names, strict=True):
            if isinstance(reference, bool) or not isinstance(reference, int | str):
                raise ValueError(f"{context} optimizer parameter reference has invalid type")
            if isinstance(reference, int) and reference < 0:
                raise ValueError(f"{context} optimizer parameter reference is negative")
            if isinstance(reference, str) and not reference:
                raise ValueError(f"{context} optimizer parameter reference is empty")
            if reference in reference_to_name or name in flattened_names:
                raise ValueError(f"{context} optimizer parameter mapping is not bijective")
            reference_to_name[reference] = name
            reference_amsgrad[reference] = amsgrad
            flattened_names.append(name)
    if flattened_names != trainable_parameter_names:
        raise ValueError(
            f"{context} optimizer parameters must contain every trainable parameter "
            "exactly once in canonical model order"
        )
    if set(states) != set(reference_to_name):
        raise ValueError(
            f"{context} optimizer state keys differ from the required trainable parameter set"
        )

    steps: dict[str, int] = {}
    for reference, name in reference_to_name.items():
        state = states[reference]
        if not isinstance(state, dict):
            raise ValueError(f"{context} AdamW state for {name!r} is malformed")
        required = set(_ADAMW_BASE_STATE_FIELDS)
        if reference_amsgrad[reference]:
            required.add("max_exp_avg_sq")
        if set(state) != required:
            raise ValueError(
                f"{context} AdamW state fields for {name!r} differ from {sorted(required)}"
            )
        parameter = model_state[name]
        first = state["exp_avg"]
        second = state["exp_avg_sq"]
        if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
            raise ValueError(f"{context} AdamW moments for {name!r} must be tensors")
        if first.shape != parameter.shape or second.shape != parameter.shape:
            raise ValueError(f"{context} AdamW moment shape for {name!r} changed")
        if first.dtype != second.dtype:
            raise ValueError(f"{context} AdamW moment dtypes for {name!r} differ")
        _finite_tensor(first, name=f"{context} exp_avg[{name}]")
        _finite_tensor(second, name=f"{context} exp_avg_sq[{name}]")
        if bool((second < 0).any().item()):
            raise ValueError(f"{context} AdamW exp_avg_sq for {name!r} is negative")
        if reference_amsgrad[reference]:
            maximum = state["max_exp_avg_sq"]
            if not isinstance(maximum, torch.Tensor) or maximum.shape != parameter.shape:
                raise ValueError(f"{context} AdamW AMSGrad moment for {name!r} changed")
            if maximum.dtype != first.dtype:
                raise ValueError(f"{context} AdamW AMSGrad dtype for {name!r} changed")
            _finite_tensor(maximum, name=f"{context} max_exp_avg_sq[{name}]")
            if bool((maximum < second).any().item()):
                raise ValueError(
                    f"{context} AdamW max_exp_avg_sq for {name!r} is below exp_avg_sq"
                )
        steps[name] = _step_value(state["step"], name=f"{context} step[{name}]")
    return steps


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return copy.deepcopy(value)


def _move_supervision_batch(batch: SupervisionBatch, device: torch.device) -> SupervisionBatch:
    for item in fields(batch):
        value = getattr(batch, item.name)
        if isinstance(value, torch.Tensor):
            setattr(batch, item.name, value.to(device))
    return batch


def state_hash(value: Any) -> str:
    """Hash nested PyTorch state with unambiguous type/length framing."""

    digest = hashlib.sha256()

    def framed(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"T")
            tensor = item.detach().cpu().contiguous()
            framed(str(tensor.dtype).encode())
            framed(str(tuple(tensor.shape)).encode())
            framed(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"D")
            framed(str(len(item)).encode())
            for key in sorted(item, key=lambda value: repr(value)):
                update(key)
                update(item[key])
        elif isinstance(item, list):
            digest.update(b"L")
            framed(str(len(item)).encode())
            for child in item:
                update(child)
        elif isinstance(item, tuple):
            digest.update(b"Q")
            framed(str(len(item)).encode())
            for child in item:
                update(child)
        else:
            digest.update(b"V")
            framed(type(item).__qualname__.encode())
            framed(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _require_sha256_text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _optimizer_parameter_names(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[list[str]]:
    canonical = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameter_names = {id(parameter): name for name, parameter in canonical}
    result: list[list[str]] = []
    for group in optimizer.param_groups:
        try:
            result.append([parameter_names[id(parameter)] for parameter in group["params"]])
        except KeyError as error:
            raise ValueError(
                "optimizer contains a non-trainable or non-model parameter"
            ) from error
    flattened = [name for group in result for name in group]
    expected = [name for name, _ in canonical]
    if flattened != expected or len(flattened) != len(set(flattened)):
        raise ValueError(
            "optimizer must contain every trainable model parameter exactly once "
            "in canonical model order"
        )
    return result


def _trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or len(names) != len(set(names)):
        raise ValueError("local-fork model trainable parameter mapping is empty or ambiguous")
    return names


def _require_regular_confined_file(
    path: Path,
    *,
    root: Path,
    name: str,
) -> Path:
    """Resolve one regular file without accepting a symlink at any owned component."""

    lexical_root = Path(os.path.abspath(os.fspath(root)))
    lexical_path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"{name} is outside its allowed root: {path}") from error
    try:
        resolved_root = lexical_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{name} root is missing: {root}") from error
    if resolved_root != lexical_root or not stat.S_ISDIR(os.lstat(lexical_root).st_mode):
        raise ValueError(f"{name} root must be a real directory, not a symlink: {root}")
    current = lexical_root
    for component in relative.parts:
        current = current / component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as error:
            raise ValueError(f"{name} is missing: {path}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} must not traverse a symlink: {current}")
    if not stat.S_ISREG(os.lstat(lexical_path).st_mode):
        raise ValueError(f"{name} must be a regular file: {path}")
    resolved_path = lexical_path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"{name} resolves outside its allowed root: {path}")
    return resolved_path


def _branch_checkpoint_state_hashes(payload: dict[str, Any]) -> dict[str, str]:
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("local-fork branch checkpoint optimizer state is malformed")
    model_parameter_names = payload.get("model_parameter_names")
    optimizer_parameter_names = payload.get("optimizer_parameter_names")
    trainable_parameter_names = payload.get("trainable_parameter_names")
    if not isinstance(model_parameter_names, list) or not all(
        isinstance(value, str) and value for value in model_parameter_names
    ):
        raise ValueError("local-fork branch checkpoint model parameter mapping is malformed")
    if not isinstance(optimizer_parameter_names, list) or not all(
        isinstance(group, list)
        and all(isinstance(value, str) and value for value in group)
        for group in optimizer_parameter_names
    ):
        raise ValueError("local-fork branch checkpoint optimizer parameter mapping is malformed")
    flattened_optimizer_names = [name for group in optimizer_parameter_names for name in group]
    if (
        len(set(model_parameter_names)) != len(model_parameter_names)
        or len(set(flattened_optimizer_names)) != len(flattened_optimizer_names)
        or flattened_optimizer_names != trainable_parameter_names
        or any(name not in payload.get("model", {}) for name in model_parameter_names)
    ):
        raise ValueError(
            "local-fork branch checkpoint model/optimizer parameter mapping is inconsistent"
        )
    steps = _validate_adamw_optimizer_contract(
        optimizer,
        optimizer_parameter_names,
        model_state=payload.get("model"),
        model_parameter_names=model_parameter_names,
        trainable_parameter_names=trainable_parameter_names,
        context="local-fork branch checkpoint",
    )
    if payload.get("optimizer_steps") != steps:
        raise ValueError("local-fork branch checkpoint optimizer-step evidence changed")
    observed_probe_outputs = payload.get("observed_probe_outputs")
    if not isinstance(observed_probe_outputs, torch.Tensor):
        raise ValueError("local-fork branch checkpoint observed probe outputs must be a tensor")
    loss_evidence = payload.get("loss_evidence")
    if not isinstance(loss_evidence, dict):
        raise ValueError("local-fork branch checkpoint loss evidence must be a mapping")
    content = {key: value for key, value in payload.items() if key != "state_hashes"}
    return {
        "payload": state_hash(content),
        "model": state_hash(payload["model"]),
        "model_parameter_names": sha256_value(model_parameter_names),
        "optimizer": state_hash(optimizer),
        "optimizer_moments": state_hash(optimizer["state"]),
        "optimizer_parameter_names": sha256_value(optimizer_parameter_names),
        "trainable_parameter_names": sha256_value(trainable_parameter_names),
        "scheduler": state_hash(payload["scheduler"]),
        "rng": state_hash(payload["rng"]),
        "prompts": sha256_value(payload["prompts"]),
        "trajectories": sha256_value(payload["trajectories"]),
        "probe_inputs": state_hash(payload["probe_input_ids"]),
        "probe_attention_mask": state_hash(payload["probe_attention_mask"]),
        "probe_outputs": state_hash(payload["pre_update_outputs"]),
        "observed_probe_outputs": state_hash(observed_probe_outputs),
        "optimizer_steps": sha256_value(steps),
        "loss_evidence": state_hash(loss_evidence),
        "initial_hashes": sha256_value(payload["initial_hashes"]),
    }


def _write_branch_checkpoint(
    path: Path,
    *,
    phase: str,
    bundle_payload: dict[str, Any],
    branch: str,
    horizon: int,
    comparison_mode: str,
    calibration_round: int,
    restored: RestoredFork,
    learning_rate_override: float | None,
    pad_token_id: int,
    observed_probe_outputs: torch.Tensor,
    loss_evidence: dict[str, Any],
) -> dict[str, Any]:
    if phase not in {"pre", "post"}:
        raise ValueError(f"unsupported local-fork checkpoint phase: {phase}")
    if comparison_mode not in {"unmatched", "matched"}:
        raise ValueError(f"unsupported local-fork comparison mode: {comparison_mode}")
    if type(calibration_round) is not int or calibration_round < 0:
        raise ValueError("local-fork calibration round must be a non-negative integer")
    if (comparison_mode == "unmatched") != (calibration_round == 0):
        raise ValueError("local-fork comparison mode/calibration round are inconsistent")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("local-fork pad token ID must be a non-negative integer")
    model_state = _cpu_tree(restored.model.state_dict())
    optimizer_state = _cpu_tree(restored.optimizer.state_dict())
    model_parameter_names = [name for name, _ in restored.model.named_parameters()]
    trainable_parameter_names = _trainable_parameter_names(restored.model)
    optimizer_parameter_names = _optimizer_parameter_names(
        restored.model,
        restored.optimizer,
    )
    optimizer_steps = _validate_adamw_optimizer_contract(
        optimizer_state,
        optimizer_parameter_names,
        model_state=model_state,
        model_parameter_names=model_parameter_names,
        trainable_parameter_names=trainable_parameter_names,
        context=f"local-fork {phase} checkpoint",
    )
    payload = {
        "format_version": _BRANCH_CHECKPOINT_FORMAT_VERSION,
        "phase": phase,
        "bundle_id": bundle_payload["manifest"]["bundle_id"],
        "bundle_manifest_sha256": bundle_payload["manifest"]["manifest_sha256"],
        "local_fork_protocol_id": bundle_payload["manifest"]["protocol_id"],
        "local_fork_spec_sha256": bundle_payload["manifest"]["local_fork_spec_sha256"],
        "branch": branch,
        "horizon": horizon,
        "comparison_mode": comparison_mode,
        "calibration_round": calibration_round,
        "learning_rate": float(restored.optimizer.param_groups[0]["lr"]),
        "learning_rate_override": learning_rate_override,
        "pad_token_id": pad_token_id,
        "model": model_state,
        "model_parameter_names": model_parameter_names,
        "trainable_parameter_names": trainable_parameter_names,
        "optimizer": optimizer_state,
        "optimizer_parameter_names": optimizer_parameter_names,
        "scheduler": _cpu_tree(restored.scheduler.state_dict()),
        "rng": RNGState.capture().as_dict(),
        "prompts": asdict(restored.prompts),
        "trajectories": [asdict(record) for record in restored.trajectories.records],
        "probe_input_ids": restored.probe_input_ids,
        "probe_attention_mask": restored.probe_attention_mask,
        "pre_update_outputs": restored.pre_update_outputs,
        "observed_probe_outputs": observed_probe_outputs.detach().cpu().clone(),
        "optimizer_steps": optimizer_steps,
        "loss_evidence": _cpu_tree(loss_evidence),
        "initial_hashes": restored.initial_hashes,
    }
    payload["state_hashes"] = _branch_checkpoint_state_hashes(payload)
    try:
        publish_torch_once(path, payload)
    except FileExistsError:
        existing_path = _require_regular_confined_file(
            path,
            root=path.parent,
            name=f"existing local-fork {phase} checkpoint",
        )
        existing = torch.load(existing_path, map_location="cpu", weights_only=False)
        if not isinstance(existing, dict) or state_hash(existing) != state_hash(payload):
            raise FileExistsError(f"conflicting local-fork checkpoint already exists: {path}")
    resolved = _require_regular_confined_file(
        path,
        root=path.parent,
        name=f"local-fork {phase} checkpoint",
    )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "payload_state_sha256": payload["state_hashes"]["payload"],
        "phase": phase,
        "state_hashes": payload["state_hashes"],
    }


def validate_branch_checkpoint_evidence(
    evidence: Any,
    *,
    checkpoint_root: Path,
    bundle_payload: dict[str, Any],
    branch: str,
    horizon: int,
    phase: str,
    comparison_mode: str,
    calibration_round: int,
) -> dict[str, Any]:
    """Re-read one current branch checkpoint and verify bytes, state, and ancestry."""

    bundle_manifest = bundle_payload["manifest"]
    if comparison_mode not in {"unmatched", "matched"}:
        raise ValueError("local-fork checkpoint evidence has an invalid comparison mode")
    if type(calibration_round) is not int or calibration_round < 0:
        raise ValueError("local-fork checkpoint evidence has an invalid calibration round")
    if (comparison_mode == "unmatched") != (calibration_round == 0):
        raise ValueError("local-fork checkpoint comparison mode/calibration round changed")
    if not isinstance(evidence, dict) or set(evidence) != _CHECKPOINT_EVIDENCE_FIELDS:
        raise ValueError(
            f"local-fork {phase} checkpoint evidence is partial or has extra fields"
        )
    if evidence.get("phase") != phase:
        raise ValueError(f"local-fork {phase} checkpoint evidence changed phase")
    path = _require_regular_confined_file(
        Path(str(evidence.get("path", ""))),
        root=checkpoint_root,
        name=f"local-fork {phase} checkpoint",
    )
    expected_file_sha256 = evidence.get("sha256")
    if expected_file_sha256 != sha256_file(path):
        raise ValueError(f"local-fork {phase} checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_file_sha256 != sha256_file(path):
        raise ValueError(f"local-fork {phase} checkpoint changed while being validated")
    if not isinstance(payload, dict) or set(payload) != _BRANCH_CHECKPOINT_FIELDS:
        raise ValueError(f"local-fork {phase} checkpoint payload schema changed")
    observed_state_hashes = _branch_checkpoint_state_hashes(payload)
    if payload.get("state_hashes") != observed_state_hashes:
        raise ValueError(f"local-fork {phase} checkpoint state hashes changed")
    if evidence.get("state_hashes") != observed_state_hashes:
        raise ValueError(f"local-fork {phase} report state evidence changed")
    if evidence.get("payload_state_sha256") != observed_state_hashes["payload"]:
        raise ValueError(f"local-fork {phase} checkpoint canonical state digest changed")
    expected_linkage = {
        "format_version": _BRANCH_CHECKPOINT_FORMAT_VERSION,
        "phase": phase,
        "bundle_id": bundle_manifest.get("bundle_id"),
        "bundle_manifest_sha256": bundle_manifest.get("manifest_sha256"),
        "local_fork_protocol_id": bundle_manifest.get("protocol_id"),
        "local_fork_spec_sha256": bundle_manifest.get("local_fork_spec_sha256"),
        "branch": branch,
        "horizon": horizon,
        "comparison_mode": comparison_mode,
        "calibration_round": calibration_round,
    }
    linkage_mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected_linkage.items()
        if payload.get(key) != value
    }
    if linkage_mismatches:
        raise ValueError(
            f"local-fork {phase} checkpoint lost branch/bundle ancestry: {linkage_mismatches}"
        )
    expected_initial_hashes = {
        "model": bundle_manifest.get("checkpoint_hash"),
        "optimizer": bundle_manifest.get("optimizer_hash"),
        "optimizer_moments": bundle_manifest.get("optimizer_moment_hash"),
        "scheduler": bundle_manifest.get("scheduler_hash"),
        "rng": bundle_manifest.get("rng_hash"),
        "prompts": bundle_manifest.get("prompt_hash"),
        "trajectories": bundle_manifest.get("trajectory_hash"),
        "probe_inputs": bundle_manifest.get("probe_input_hash"),
        "probe_attention_mask": bundle_manifest.get("probe_attention_mask_hash"),
        "probe_outputs": bundle_manifest.get("probe_output_hash"),
    }
    if payload.get("initial_hashes") != expected_initial_hashes:
        raise ValueError(f"local-fork {phase} checkpoint initial state changed")
    if observed_state_hashes["optimizer_parameter_names"] != bundle_manifest.get(
        "optimizer_parameter_names_hash"
    ):
        raise ValueError(f"local-fork {phase} checkpoint optimizer mapping changed")
    if observed_state_hashes["trainable_parameter_names"] != bundle_manifest.get(
        "trainable_parameter_names_hash"
    ):
        raise ValueError(f"local-fork {phase} checkpoint trainable mapping changed")
    for key in (
        "prompts",
        "trajectories",
        "probe_inputs",
        "probe_attention_mask",
        "probe_outputs",
    ):
        if observed_state_hashes[key] != expected_initial_hashes[key]:
            raise ValueError(f"local-fork {phase} checkpoint static scientific state changed: {key}")
    observed_probe = payload.get("observed_probe_outputs")
    baseline_probe = payload.get("pre_update_outputs")
    if not isinstance(observed_probe, torch.Tensor) or not isinstance(
        baseline_probe, torch.Tensor
    ):
        raise ValueError(f"local-fork {phase} checkpoint probe logits are malformed")
    if observed_probe.shape != baseline_probe.shape or observed_probe.ndim != 3:
        raise ValueError(f"local-fork {phase} checkpoint probe-logit shape changed")
    _finite_tensor(observed_probe, name=f"local-fork {phase} observed probe logits")
    _finite_tensor(baseline_probe, name=f"local-fork {phase} baseline probe logits")
    loss_evidence = payload.get("loss_evidence")
    if not isinstance(loss_evidence, dict) or set(loss_evidence) != _LOSS_EVIDENCE_FIELDS:
        raise ValueError(f"local-fork {phase} checkpoint loss evidence schema changed")
    branch_objective = next(
        specification.objective
        for specification in LOCAL_FORK_SPEC.branches
        if specification.branch_id == branch
    )
    if (
        type(loss_evidence.get("schema_version")) is not int
        or loss_evidence.get("schema_version") != 1
        or loss_evidence.get("phase") != phase
        or loss_evidence.get("objective") != branch_objective
    ):
        raise ValueError(f"local-fork {phase} checkpoint loss evidence ancestry changed")
    checkpoint_losses = loss_evidence.get("step_losses")
    checkpoint_metrics = loss_evidence.get("step_metrics")
    if not isinstance(checkpoint_losses, list) or not isinstance(checkpoint_metrics, list):
        raise ValueError(f"local-fork {phase} checkpoint loss rows are malformed")
    pad_token_id = payload.get("pad_token_id")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError(f"local-fork {phase} checkpoint pad token ID changed")
    override = payload.get("learning_rate_override")
    if override is not None and (
        isinstance(override, bool)
        or not isinstance(override, int | float)
        or not math.isfinite(float(override))
        or float(override) <= 0.0
    ):
        raise ValueError("local-fork learning-rate override is invalid")
    if comparison_mode == "unmatched" and override is not None:
        raise ValueError("unmatched local-fork checkpoint cannot claim an LR override")
    if comparison_mode == "matched" and override is None:
        raise ValueError("matched local-fork checkpoint requires its calibrated LR override")
    learning_rate = payload.get("learning_rate")
    if type(learning_rate) is not float or not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(f"local-fork {phase} checkpoint learning rate is invalid")
    expected_lr = (
        float(override)
        if override is not None
        else float(bundle_payload["optimizer"]["param_groups"][0]["lr"])
    )
    if learning_rate != expected_lr or any(
        type(group.get("lr")) not in {int, float}
        or isinstance(group.get("lr"), bool)
        or float(group["lr"]) != expected_lr
        for group in payload["optimizer"]["param_groups"]
    ):
        raise ValueError(f"local-fork {phase} checkpoint learning-rate evidence changed")
    _validate_optimizer_param_groups(
        bundle_payload["optimizer"],
        payload["optimizer"],
        allowed_value_changes=frozenset({"lr"}),
        context=f"local-fork {phase} checkpoint versus bundle",
    )
    if phase == "pre":
        if (
            checkpoint_losses
            or checkpoint_metrics
            or loss_evidence.get("final_loss") is not None
            or not torch.equal(observed_probe, baseline_probe)
        ):
            raise ValueError("local-fork pre checkpoint contains post-update evidence")
    else:
        if len(checkpoint_losses) != horizon or len(checkpoint_metrics) != horizon:
            raise ValueError("local-fork post checkpoint loss evidence differs from horizon")
        if not checkpoint_losses or loss_evidence.get("final_loss") != checkpoint_losses[-1]:
            raise ValueError("local-fork post checkpoint final loss differs from its last step")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in checkpoint_losses
        ) or any(not isinstance(row, dict) for row in checkpoint_metrics):
            raise ValueError("local-fork post checkpoint loss evidence is non-finite/malformed")
    if phase == "pre":
        for key in (
            "model",
            "optimizer_moments",
            "rng",
            "prompts",
            "trajectories",
            "probe_inputs",
            "probe_attention_mask",
            "probe_outputs",
        ):
            if observed_state_hashes[key] != expected_initial_hashes[key]:
                raise ValueError(f"local-fork pre checkpoint differs from bundle state: {key}")
        observed_optimizer = copy.deepcopy(payload["optimizer"])
        expected_optimizer = copy.deepcopy(bundle_payload["optimizer"])
        if len(observed_optimizer["param_groups"]) != len(expected_optimizer["param_groups"]):
            raise ValueError("local-fork pre checkpoint optimizer groups changed")
        for observed_group, expected_group in zip(
            observed_optimizer["param_groups"],
            expected_optimizer["param_groups"],
            strict=True,
        ):
            observed_group["lr"] = expected_group["lr"]
        if state_hash(observed_optimizer) != state_hash(expected_optimizer):
            raise ValueError("local-fork pre checkpoint changed optimizer state beyond learning rate")
        observed_scheduler = copy.deepcopy(payload["scheduler"])
        expected_scheduler = copy.deepcopy(bundle_payload["scheduler"])
        observed_scheduler["base_lrs"] = expected_scheduler.get("base_lrs")
        if state_hash(observed_scheduler) != state_hash(expected_scheduler):
            raise ValueError("local-fork pre checkpoint changed scheduler state beyond learning rate")
    return payload


def _strict_json_object(path: Path, *, name: str) -> dict[str, Any]:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_from_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{name} contains a non-finite JSON constant: {value}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _optimizer_state_by_parameter_name(
    optimizer_state: dict[str, Any],
    optimizer_parameter_names: list[list[str]],
) -> dict[str, Any]:
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        raise ValueError("optimizer state lacks parameter groups/state")
    if len(groups) != len(optimizer_parameter_names):
        raise ValueError("optimizer parameter-name groups are misaligned")
    reference_to_name: dict[Any, str] = {}
    canonical_groups = []
    for group, names in zip(groups, optimizer_parameter_names, strict=True):
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError("optimizer parameter group is malformed")
        references = group["params"]
        if len(references) != len(names):
            raise ValueError("optimizer parameter-name mapping length changed")
        for reference, name in zip(references, names, strict=True):
            if reference in reference_to_name or name in reference_to_name.values():
                raise ValueError("optimizer parameter-name mapping is not one-to-one")
            reference_to_name[reference] = name
        canonical_groups.append(
            {
                **{key: copy.deepcopy(value) for key, value in group.items() if key != "params"},
                "params": list(names),
            }
        )
    unknown = set(states) - set(reference_to_name)
    if unknown:
        raise ValueError("optimizer moment state contains an unmapped parameter")
    canonical_states = {
        reference_to_name[reference]: copy.deepcopy(value)
        for reference, value in states.items()
    }
    return {"state": canonical_states, "param_groups": canonical_groups}


def _validate_optimizer_param_groups(
    reference_optimizer: Any,
    observed_optimizer: Any,
    *,
    allowed_value_changes: frozenset[str],
    context: str,
) -> None:
    """Require exact group topology and hyperparameters except named protocol changes."""

    if not isinstance(reference_optimizer, dict) or not isinstance(
        observed_optimizer, dict
    ):
        raise ValueError(f"{context} optimizer state must be a mapping")
    reference_groups = reference_optimizer.get("param_groups")
    observed_groups = observed_optimizer.get("param_groups")
    if not isinstance(reference_groups, list) or not isinstance(observed_groups, list):
        raise ValueError(f"{context} optimizer parameter groups are malformed")
    if len(reference_groups) != len(observed_groups):
        raise ValueError(f"{context} optimizer parameter-group count changed")
    for group_index, (reference, observed) in enumerate(
        zip(reference_groups, observed_groups, strict=True)
    ):
        if not isinstance(reference, dict) or not isinstance(observed, dict):
            raise ValueError(f"{context} optimizer group {group_index} is malformed")
        if set(reference) != set(observed):
            raise ValueError(
                f"{context} optimizer group {group_index} key set changed: "
                f"expected={sorted(reference)}, observed={sorted(observed)}"
            )
        missing_allowed = allowed_value_changes - set(reference)
        if missing_allowed:
            raise ValueError(
                f"{context} optimizer group {group_index} lacks allowed keys "
                f"{sorted(missing_allowed)}"
            )
        for key in reference:
            if key in allowed_value_changes:
                continue
            if state_hash(reference[key]) != state_hash(observed[key]):
                raise ValueError(
                    f"{context} optimizer group {group_index} hyperparameter "
                    f"{key!r} changed"
                )


def validate_branch_optimizer_transition(
    pre_checkpoint: dict[str, Any],
    post_checkpoint: dict[str, Any],
    *,
    bundle_payload: dict[str, Any],
    horizon: int,
) -> None:
    """Validate exact AdamW group/moment topology across one branch transition."""

    if type(horizon) is not int or horizon < 1:
        raise ValueError("local-fork AdamW transition requires a positive integer horizon")
    for name, payload in (("pre", pre_checkpoint), ("post", post_checkpoint)):
        if not isinstance(payload, dict):
            raise ValueError(f"local-fork {name} checkpoint must be a mapping")
        if payload.get("optimizer_parameter_names") != bundle_payload.get(
            "optimizer_parameter_names"
        ):
            raise ValueError(f"local-fork {name} optimizer parameter mapping changed")
        _validate_adamw_optimizer_contract(
            payload.get("optimizer"),
            payload.get("optimizer_parameter_names"),
            model_state=payload.get("model"),
            model_parameter_names=payload.get("model_parameter_names"),
            trainable_parameter_names=payload.get("trainable_parameter_names"),
            context=f"local-fork {name} checkpoint transition",
        )

    _validate_optimizer_param_groups(
        bundle_payload["optimizer"],
        pre_checkpoint["optimizer"],
        allowed_value_changes=frozenset({"lr"}),
        context="local-fork bundle-to-pre transition",
    )
    _validate_optimizer_param_groups(
        bundle_payload["optimizer"],
        post_checkpoint["optimizer"],
        allowed_value_changes=frozenset({"lr"}),
        context="local-fork bundle-to-post transition",
    )
    _validate_optimizer_param_groups(
        pre_checkpoint["optimizer"],
        post_checkpoint["optimizer"],
        allowed_value_changes=frozenset(),
        context="local-fork pre-to-post transition",
    )
    pre_named = _optimizer_state_by_parameter_name(
        pre_checkpoint["optimizer"],
        pre_checkpoint["optimizer_parameter_names"],
    )["state"]
    post_named = _optimizer_state_by_parameter_name(
        post_checkpoint["optimizer"],
        post_checkpoint["optimizer_parameter_names"],
    )["state"]
    if set(pre_named) != set(post_named):
        raise ValueError("local-fork AdamW moment parameter set changed")
    for parameter_name in pre_checkpoint["trainable_parameter_names"]:
        before = pre_named[parameter_name]
        after = post_named[parameter_name]
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(
                f"local-fork AdamW state for {parameter_name!r} is malformed"
            )
        if set(before) != set(after):
            raise ValueError(
                f"local-fork AdamW state fields for {parameter_name!r} changed"
            )
        before_step = _step_value(
            before["step"], name=f"local-fork pre step[{parameter_name}]"
        )
        after_step = _step_value(
            after["step"], name=f"local-fork post step[{parameter_name}]"
        )
        if after_step - before_step != horizon:
            raise ValueError(
                f"local-fork AdamW step for {parameter_name!r} did not advance by horizon"
            )
        for field in sorted(set(before) - {"step"}):
            before_moment = before[field]
            after_moment = after[field]
            if not isinstance(before_moment, torch.Tensor) or not isinstance(
                after_moment, torch.Tensor
            ):
                raise ValueError(
                    f"local-fork AdamW {field} for {parameter_name!r} must be tensors"
                )
            if (
                after_moment.shape != before_moment.shape
                or after_moment.dtype != before_moment.dtype
            ):
                raise ValueError(
                    f"local-fork AdamW {field} shape/dtype for {parameter_name!r} changed"
                )
            _finite_tensor(
                before_moment,
                name=f"local-fork pre {field}[{parameter_name}]",
            )
            _finite_tensor(
                after_moment,
                name=f"local-fork post {field}[{parameter_name}]",
            )


def _source_checkpoint_state_hashes(
    payload: dict[str, Any],
    optimizer_parameter_names: list[list[str]],
    *,
    model_parameter_names: list[str],
    trainable_parameter_names: list[str],
) -> dict[str, str]:
    required = {"model", "optimizer", "scheduler", "rng"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"source checkpoint is incomplete; missing {sorted(missing)}")
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    if not isinstance(model, dict) or not isinstance(optimizer, dict):
        raise ValueError("source checkpoint model/optimizer state must be mappings")
    moments = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if not isinstance(moments, dict) or not moments:
        raise ValueError("source checkpoint lacks optimizer moment state")
    if not isinstance(groups, list) or not groups:
        raise ValueError("source checkpoint lacks optimizer parameter groups")
    key_type = payload.get("optimizer_state_key_type", "parameter_id")
    if key_type == "parameter_name":
        source_parameter_names = []
        for group in groups:
            references = group.get("params") if isinstance(group, dict) else None
            if not isinstance(references, list) or not all(
                isinstance(reference, str) and reference for reference in references
            ):
                raise ValueError(
                    "parameter-name optimizer checkpoint has malformed parameter references"
                )
            source_parameter_names.append(list(references))
        if source_parameter_names != optimizer_parameter_names:
            raise ValueError(
                "parameter-name/FSDP optimizer mapping differs from canonical model order"
            )
    elif key_type == "parameter_id":
        for group in groups:
            references = group.get("params") if isinstance(group, dict) else None
            if not isinstance(references, list) or not all(
                type(reference) is int and reference >= 0 for reference in references
            ):
                raise ValueError(
                    "parameter-ID optimizer checkpoint has malformed parameter references"
                )
        source_parameter_names = optimizer_parameter_names
    else:
        raise ValueError(f"unsupported optimizer state key type: {key_type}")
    _validate_adamw_optimizer_contract(
        optimizer,
        source_parameter_names,
        model_state=model,
        model_parameter_names=model_parameter_names,
        trainable_parameter_names=trainable_parameter_names,
        context="local-fork source checkpoint",
    )
    parameter_references = {
        "optimizer_state_key_type": key_type,
        "param_groups": [group.get("params") if isinstance(group, dict) else None for group in groups],
    }
    return {
        "model": state_hash(model),
        "model_parameter_names": sha256_value(model_parameter_names),
        "trainable_parameter_names": sha256_value(trainable_parameter_names),
        "optimizer": state_hash(optimizer),
        "optimizer_moments": state_hash(moments),
        "optimizer_parameter_references": sha256_value(parameter_references),
        "optimizer_named": state_hash(
            _optimizer_state_by_parameter_name(optimizer, source_parameter_names)
        ),
        "scheduler": state_hash(payload["scheduler"]),
        "rng": state_hash(payload["rng"]),
    }


def _effective_bundle_state_hashes(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    source_rng: Any,
) -> dict[str, str]:
    optimizer_state = optimizer.state_dict()
    moments = optimizer_state.get("state")
    if not isinstance(moments, dict) or not moments:
        raise ValueError("effective local-fork optimizer lacks moment state")
    model_state = model.state_dict()
    model_parameter_names = [name for name, _ in model.named_parameters()]
    trainable_parameter_names = _trainable_parameter_names(model)
    optimizer_parameter_names = _optimizer_parameter_names(model, optimizer)
    _validate_adamw_optimizer_contract(
        optimizer_state,
        optimizer_parameter_names,
        model_state=model_state,
        model_parameter_names=model_parameter_names,
        trainable_parameter_names=trainable_parameter_names,
        context="effective local-fork optimizer",
    )
    return {
        "model": state_hash(model.state_dict()),
        "model_parameter_names": sha256_value(model_parameter_names),
        "trainable_parameter_names": sha256_value(trainable_parameter_names),
        "optimizer": state_hash(optimizer_state),
        "optimizer_moments": state_hash(moments),
        "optimizer_parameter_names": sha256_value(
            optimizer_parameter_names
        ),
        "optimizer_named": state_hash(
            _optimizer_state_by_parameter_name(
                optimizer_state,
                optimizer_parameter_names,
            )
        ),
        "scheduler": state_hash(scheduler.state_dict()),
        "rng": state_hash(source_rng),
    }


def build_local_fork_source_binding(
    *,
    source_manifest_path: Path,
    checkpoint_path: Path,
    checkpoint_payload: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_seed: int,
    expected_protocol_track: str,
    expected_preregistration_sha256: str,
    expected_implementation_commit: str,
    expected_model_id: str,
    expected_model_revision: str,
    expected_model_resolved_revision: str,
    expected_tokenizer_id: str,
    expected_tokenizer_revision: str,
    expected_tokenizer_resolved_revision: str,
    expected_tokenizer_fingerprint: str,
    expected_prompt_protocol: str,
    expected_chat_template_sha256: str,
    trajectory_bank_sha256: str,
) -> dict[str, Any]:
    """Build a formal source binding only from a valid manifest-declared checkpoint."""

    manifest_path = _require_regular_confined_file(
        source_manifest_path,
        root=source_manifest_path.parent,
        name="local-fork source run manifest",
    )
    manifest_file_sha256 = sha256_file(manifest_path)
    manifest = _strict_json_object(manifest_path, name="local-fork source run manifest")
    from posttrain_circuits.artifacts.runs import validate_run_manifest_payload
    from posttrain_circuits.experiments.protocols.specs import (
        validate_checkpoint_experiment_binding_payload,
        validate_run_manifest_experiment_binding,
    )

    validate_run_manifest_payload(manifest)
    binding = validate_run_manifest_experiment_binding(manifest)
    expected_binding = {
        "seed": expected_seed,
        "protocol_track": expected_protocol_track,
        "preregistration_sha256": expected_preregistration_sha256,
        "implementation_commit": expected_implementation_commit,
        "implementation_dirty": False,
        "model_id": expected_model_id,
        "model_requested_revision": expected_model_revision,
        "model_resolved_revision": expected_model_resolved_revision,
        "tokenizer_id": expected_tokenizer_id,
        "tokenizer_requested_revision": expected_tokenizer_revision,
        "tokenizer_resolved_revision": expected_tokenizer_resolved_revision,
        "tokenizer_fingerprint": expected_tokenizer_fingerprint,
        "prompt_protocol": expected_prompt_protocol,
        "chat_template_sha256": expected_chat_template_sha256,
    }
    binding_mismatches = {
        key: {"expected": value, "observed": getattr(binding, key)}
        for key, value in expected_binding.items()
        if getattr(binding, key) != value
    }
    if binding_mismatches:
        raise ValueError(
            "local-fork source run differs from the requested model/seed/protocol: "
            f"{binding_mismatches}"
        )
    declared = manifest.get("final_checkpoint_path")
    if not isinstance(declared, str) or not Path(declared).is_absolute():
        raise ValueError("source run manifest requires an absolute final checkpoint path")
    checkpoint_root = manifest_path.parent / "checkpoints"
    declared_path = _require_regular_confined_file(
        Path(declared),
        root=checkpoint_root,
        name="local-fork source final checkpoint",
    )
    if declared != str(declared_path):
        raise ValueError("source run manifest final checkpoint path is not canonical")
    supplied_path = _require_regular_confined_file(
        checkpoint_path,
        root=checkpoint_root,
        name="local-fork supplied source checkpoint",
    )
    if declared_path != supplied_path:
        raise ValueError("local-fork checkpoint is not the source manifest's final checkpoint")
    checkpoint_sha256 = sha256_file(supplied_path)
    if manifest.get("final_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("local-fork source checkpoint bytes differ from its run manifest")
    current_payload = torch.load(supplied_path, map_location="cpu", weights_only=False)
    if checkpoint_sha256 != sha256_file(supplied_path):
        raise ValueError("local-fork source checkpoint changed while being validated")
    if not isinstance(current_payload, dict):
        raise ValueError("local-fork source checkpoint payload must be a mapping")
    if state_hash(current_payload) != state_hash(checkpoint_payload):
        raise ValueError("local-fork source checkpoint changed while the bundle was being built")
    validate_checkpoint_experiment_binding_payload(current_payload, binding)
    if manifest_file_sha256 != sha256_file(manifest_path):
        raise ValueError("local-fork source run manifest changed while being validated")
    optimizer_parameter_names = _optimizer_parameter_names(model, optimizer)
    model_parameter_names = [name for name, _ in model.named_parameters()]
    trainable_parameter_names = _trainable_parameter_names(model)
    checkpoint_state_hashes = _source_checkpoint_state_hashes(
        current_payload,
        optimizer_parameter_names,
        model_parameter_names=model_parameter_names,
        trainable_parameter_names=trainable_parameter_names,
    )
    effective_bundle_state_hashes = _effective_bundle_state_hashes(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        source_rng=current_payload["rng"],
    )
    for key in (
        "model",
        "model_parameter_names",
        "trainable_parameter_names",
        "optimizer_named",
        "scheduler",
        "rng",
    ):
        if checkpoint_state_hashes[key] != effective_bundle_state_hashes[key]:
            raise ValueError(
                "local-fork effective bundle state differs from source checkpoint: "
                f"{key}"
            )
    source = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_payload_sha256": str(manifest["sha256"]),
        "experiment_binding": binding.to_payload(),
        "experiment_binding_sha256": binding.scientific_sha256,
        "factorial_design_sha256": binding.factorial_design_sha256,
        "method_id": binding.method_id,
        "seed": binding.seed,
        "train_dataset_sha256": binding.train_dataset_sha256,
        "validation_dataset_sha256": binding.validation_dataset_sha256,
        "initial_checkpoint_sha256": binding.initial_checkpoint_sha256,
        "final_checkpoint_path": str(supplied_path),
        "final_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_state_hashes": checkpoint_state_hashes,
        "effective_bundle_state_hashes": effective_bundle_state_hashes,
        "optimizer_parameter_names": optimizer_parameter_names,
        "trajectory_bank_sha256": trajectory_bank_sha256,
    }
    from posttrain_circuits.experiments.protocols.local_fork import (
        validate_local_fork_source_binding,
    )

    return validate_local_fork_source_binding(source)


def validate_local_fork_source_binding_files(
    source_binding: Any,
    *,
    bundle_payload: dict[str, Any],
) -> dict[str, Any]:
    """Re-read source manifest/checkpoint bytes and re-run semantic validators."""

    from posttrain_circuits.experiments.protocols.local_fork import (
        validate_local_fork_source_binding,
    )

    source = validate_local_fork_source_binding(source_binding)
    bundle_manifest = bundle_payload["manifest"]
    manifest_path = _require_regular_confined_file(
        Path(source["manifest_path"]),
        root=Path(source["manifest_path"]).parent,
        name="local-fork source run manifest",
    )
    if source["manifest_file_sha256"] != sha256_file(manifest_path):
        raise ValueError("local-fork source run manifest bytes changed")
    manifest = _strict_json_object(manifest_path, name="local-fork source run manifest")
    from posttrain_circuits.artifacts.runs import validate_run_manifest_payload
    from posttrain_circuits.experiments.protocols.specs import (
        validate_checkpoint_experiment_binding_payload,
        validate_run_manifest_experiment_binding,
    )

    validate_run_manifest_payload(manifest)
    binding = validate_run_manifest_experiment_binding(manifest)
    if source["manifest_file_sha256"] != sha256_file(manifest_path):
        raise ValueError("local-fork source run manifest changed while being validated")
    if manifest.get("sha256") != source["manifest_payload_sha256"]:
        raise ValueError("local-fork source run manifest payload identity changed")
    if binding.to_payload() != source["experiment_binding"] or (
        binding.scientific_sha256 != source["experiment_binding_sha256"]
    ):
        raise ValueError("local-fork source ExperimentBinding changed")
    declared = manifest.get("final_checkpoint_path")
    if declared != source["final_checkpoint_path"]:
        raise ValueError("local-fork source final checkpoint path changed")
    checkpoint_path = _require_regular_confined_file(
        Path(source["final_checkpoint_path"]),
        root=manifest_path.parent / "checkpoints",
        name="local-fork source final checkpoint",
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        checkpoint_sha256 != source["final_checkpoint_sha256"]
        or manifest.get("final_checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("local-fork source final checkpoint bytes changed")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_sha256 != sha256_file(checkpoint_path):
        raise ValueError("local-fork source final checkpoint changed while being validated")
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("local-fork source checkpoint payload must be a mapping")
    validate_checkpoint_experiment_binding_payload(checkpoint_payload, binding)
    if _source_checkpoint_state_hashes(
        checkpoint_payload,
        source["optimizer_parameter_names"],
        model_parameter_names=bundle_payload["model_parameter_names"],
        trainable_parameter_names=bundle_payload["trainable_parameter_names"],
    ) != source["checkpoint_state_hashes"]:
        raise ValueError("local-fork source checkpoint canonical state changed")
    effective = source["effective_bundle_state_hashes"]
    expected_effective = {
        "model": bundle_manifest.get("checkpoint_hash"),
        "model_parameter_names": bundle_manifest.get("model_parameter_names_hash"),
        "trainable_parameter_names": bundle_manifest.get(
            "trainable_parameter_names_hash"
        ),
        "optimizer": bundle_manifest.get("optimizer_hash"),
        "optimizer_moments": bundle_manifest.get("optimizer_moment_hash"),
        "optimizer_parameter_names": bundle_manifest.get("optimizer_parameter_names_hash"),
        "optimizer_named": state_hash(
            _optimizer_state_by_parameter_name(
                bundle_payload["optimizer"],
                bundle_payload["optimizer_parameter_names"],
            )
        ),
        "scheduler": bundle_manifest.get("scheduler_hash"),
        "rng": bundle_manifest.get("rng_hash"),
    }
    if effective != expected_effective:
        raise ValueError("local-fork bundle state differs from its source checkpoint binding")
    if source["optimizer_parameter_names"] != bundle_payload["optimizer_parameter_names"]:
        raise ValueError("local-fork optimizer parameter mapping differs from its source binding")
    raw = source["checkpoint_state_hashes"]
    for key in (
        "model",
        "model_parameter_names",
        "trainable_parameter_names",
        "optimizer_named",
        "scheduler",
        "rng",
    ):
        if raw[key] != effective[key]:
            raise ValueError(
                f"local-fork effective bundle state differs from source checkpoint: {key}"
            )
    if source["trajectory_bank_sha256"] != bundle_manifest.get("manifest_hashes", {}).get("bank"):
        raise ValueError("local-fork trajectory-bank ancestry changed")
    return source


def validate_bundle_file_evidence(
    evidence: Any,
    *,
    expected_path: Path,
    embedded_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
        raise ValueError("local-fork bundle-file evidence is partial or has extra fields")
    expected = _require_regular_confined_file(
        expected_path,
        root=expected_path.parent,
        name="local-fork bundle",
    )
    observed = _require_regular_confined_file(
        Path(str(evidence.get("path", ""))),
        root=expected_path.parent,
        name="local-fork bound bundle",
    )
    if observed != expected:
        raise ValueError("local-fork report points at a different bundle file")
    expected_file_sha256 = evidence.get("sha256")
    if expected_file_sha256 != sha256_file(observed):
        raise ValueError("local-fork bundle bytes changed")
    payload = load_fork_bundle(observed)
    if expected_file_sha256 != sha256_file(observed):
        raise ValueError("local-fork bundle changed while being validated")
    if sha256_value(payload.get("manifest")) != sha256_value(embedded_manifest):
        raise ValueError("local-fork embedded bundle manifest differs from current bundle bytes")
    return payload


def validate_checkpoint_tree_exact(
    checkpoint_root: Path,
    *,
    expected_paths: set[str],
) -> None:
    """Reject unbound, missing, non-regular, or symlinked checkpoint tree nodes."""

    lexical_root = Path(os.path.abspath(os.fspath(checkpoint_root)))
    try:
        resolved_root = lexical_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("local-fork checkpoint root is missing") from error
    if resolved_root != lexical_root or not stat.S_ISDIR(os.lstat(lexical_root).st_mode):
        raise ValueError("local-fork checkpoint root must be a real directory")
    expected_files: set[str] = set()
    expected_directories: set[str] = set()
    for expected_path in expected_paths:
        resolved_path = _require_regular_confined_file(
            Path(expected_path),
            root=lexical_root,
            name="expected local-fork checkpoint",
        )
        expected_files.add(str(resolved_path))
        relative_parent = resolved_path.relative_to(resolved_root).parent
        while relative_parent != Path("."):
            expected_directories.add(str(resolved_root / relative_parent))
            relative_parent = relative_parent.parent
    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda value: value.name):
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"local-fork checkpoint tree contains a symlink: {child}")
            if stat.S_ISDIR(mode):
                observed_directories.add(str(child.resolve(strict=True)))
                visit(child)
            elif stat.S_ISREG(mode):
                observed_files.add(str(child.resolve(strict=True)))
            else:
                raise ValueError(
                    f"local-fork checkpoint tree contains a non-regular entry: {child}"
                )

    visit(lexical_root)
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise ValueError(
            "local-fork checkpoint tree differs from report evidence: "
            f"missing_files={sorted(expected_files - observed_files)}, "
            f"extra_files={sorted(observed_files - expected_files)}, "
            "missing_directories="
            f"{sorted(expected_directories - observed_directories)}, "
            "extra_directories="
            f"{sorted(observed_directories - expected_directories)}"
        )


def _validate_input_evidence(
    evidence: Any,
    *,
    trajectories: list[dict[str, Any]],
    probe_input_ids: torch.Tensor,
    probe_attention_mask: torch.Tensor,
    pre_update_outputs: torch.Tensor,
    formal: bool,
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != _INPUT_EVIDENCE_FIELDS:
        raise ValueError("local-fork input evidence fields differ from schema")
    expected_constants = {
        "schema_version": 1,
        "local_fork_protocol_id": LOCAL_FORK_SPEC.protocol_id,
        "local_fork_spec_sha256": LOCAL_FORK_SPEC.scientific_sha256,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
        "probe_kl_mask_protocol": LOCAL_FORK_SPEC.probe_kl_mask_protocol,
    }
    if type(evidence.get("schema_version")) is not int or any(
        evidence.get(key) != value for key, value in expected_constants.items()
    ):
        raise ValueError("local-fork input evidence scientific compatibility changed")
    trajectory_ids = [record.get("trajectory_id") for record in trajectories]
    if any(not isinstance(value, str) or not value for value in trajectory_ids) or len(
        trajectory_ids
    ) != len(set(trajectory_ids)):
        raise ValueError("local-fork input evidence trajectory IDs are malformed/duplicate")
    trajectory_hashes = {
        str(record.get("trajectory_id")): sha256_value(record) for record in trajectories
    }
    expected_content = {
        "selected_trajectory_ids": trajectory_ids,
        "selected_trajectory_hashes": trajectory_hashes,
        "selected_trajectories_sha256": sha256_value(trajectories),
        "probe_input_hash": state_hash(probe_input_ids),
        "probe_attention_mask_hash": state_hash(probe_attention_mask),
        "pre_update_output_hash": state_hash(pre_update_outputs),
    }
    if any(evidence.get(key) != value for key, value in expected_content.items()):
        raise ValueError("local-fork input evidence differs from selected bank/probe bytes")
    path_fields = (
        "trajectory_bank_manifest_path",
        "prompt_manifest_path",
        "probe_manifest_path",
    )
    digest_fields = (
        "trajectory_bank_manifest_file_sha256",
        "trajectory_bank_content_sha256",
        "prompt_manifest_file_sha256",
        "prompt_manifest_payload_sha256",
        "probe_manifest_file_sha256",
        "probe_manifest_payload_sha256",
    )
    if formal:
        for name in path_fields:
            value = evidence.get(name)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"formal local-fork input evidence requires absolute {name}")
        for name in digest_fields:
            _require_sha256_text(evidence.get(name), name=f"local-fork input {name}")
    elif any(evidence.get(name) is not None for name in (*path_fields, *digest_fields)):
        raise ValueError("smoke local-fork input evidence cannot claim formal files")
    return evidence


def _validate_frozen_input_manifest(
    payload: Any,
    *,
    expected_fields: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(f"{name} fields differ from schema")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != sha256_value(content):
        raise ValueError(f"{name} payload SHA-256 mismatch")
    constants = {
        "format_version": 1,
        "local_fork_protocol_id": LOCAL_FORK_SPEC.protocol_id,
        "local_fork_spec_sha256": LOCAL_FORK_SPEC.scientific_sha256,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
    }
    if type(payload.get("format_version")) is not int or any(
        payload.get(key) != value for key, value in constants.items()
    ):
        raise ValueError(f"{name} scientific compatibility changed")
    for field in ("prompt_ids", "trajectory_ids"):
        values = payload.get(field)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"{name} {field} must be a non-empty string list")
        if field == "trajectory_ids" and len(values) != len(set(values)):
            raise ValueError(f"{name} {field} contains duplicate identities")
    if not isinstance(payload.get("prompt_protocol"), str) or not payload["prompt_protocol"]:
        raise ValueError(f"{name} prompt_protocol is malformed")
    if "prompt_texts" in payload:
        prompt_texts = payload.get("prompt_texts")
        if not isinstance(prompt_texts, list) or len(prompt_texts) != len(
            payload["prompt_ids"]
        ) or any(not isinstance(value, str) or not value for value in prompt_texts):
            raise ValueError(f"{name} prompt_texts are malformed")
    if "generation_groups" in payload:
        generation_groups = payload.get("generation_groups")
        if not isinstance(generation_groups, dict) or any(
            not isinstance(group_id, str)
            or not group_id
            or not isinstance(members, list)
            or not members
            or any(not isinstance(value, str) or not value for value in members)
            for group_id, members in generation_groups.items()
        ):
            raise ValueError(f"{name} generation_groups are malformed")
    for field in ("trajectory_store_hash", "group_membership_hash", "sha256"):
        _require_sha256_text(payload.get(field), name=f"{name} {field}")
    return payload


def validate_bundle_input_evidence_files(
    bundle_payload: dict[str, Any],
    *,
    expected_trajectory_bank_manifest_path: Path,
    expected_prompt_manifest_path: Path,
    expected_probe_manifest_path: Path,
) -> None:
    """Re-read the externally selected pilot bank/prompt/probe manifests."""

    evidence = bundle_payload["input_evidence"]
    expected_paths = {
        "trajectory_bank_manifest_path": Path(expected_trajectory_bank_manifest_path),
        "prompt_manifest_path": Path(expected_prompt_manifest_path),
        "probe_manifest_path": Path(expected_probe_manifest_path),
    }
    for field, expected_path in expected_paths.items():
        selected = _require_regular_confined_file(
            expected_path,
            root=expected_path.parent,
            name=f"pilot-selected local-fork {field}",
        )
        observed = _require_regular_confined_file(
            Path(evidence[field]),
            root=expected_path.parent,
            name=f"bundle-bound local-fork {field}",
        )
        if observed != selected:
            raise ValueError(f"local-fork {field} differs from the pilot-selected input")
        digest_field = field.replace("_path", "_file_sha256")
        if evidence[digest_field] != sha256_file(selected):
            raise ValueError(f"local-fork {field} bytes changed")
    bank = _strict_json_object(
        expected_paths["trajectory_bank_manifest_path"],
        name="local-fork trajectory bank manifest",
    )
    prompts = _strict_json_object(
        expected_paths["prompt_manifest_path"],
        name="local-fork prompt manifest",
    )
    probe = _strict_json_object(
        expected_paths["probe_manifest_path"],
        name="local-fork probe manifest",
    )
    prompts = _validate_frozen_input_manifest(
        prompts,
        expected_fields=_PROMPT_MANIFEST_FIELDS,
        name="local-fork prompt manifest",
    )
    probe = _validate_frozen_input_manifest(
        probe,
        expected_fields=_PROBE_MANIFEST_FIELDS,
        name="local-fork probe manifest",
    )
    if bank.get("sha256") != evidence["trajectory_bank_content_sha256"]:
        raise ValueError("local-fork trajectory bank content identity changed")
    compatibility_expected = {
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
    }
    if any(bank.get(key) != value for key, value in compatibility_expected.items()):
        raise ValueError("local-fork trajectory bank compatibility changed")
    if prompts.get("sha256") != evidence["prompt_manifest_payload_sha256"]:
        raise ValueError("local-fork prompt manifest payload identity changed")
    if probe.get("sha256") != evidence["probe_manifest_payload_sha256"]:
        raise ValueError("local-fork probe manifest payload identity changed")
    if prompts.get("trajectory_ids") != evidence["selected_trajectory_ids"] or probe.get(
        "trajectory_ids"
    ) != evidence["selected_trajectory_ids"]:
        raise ValueError("local-fork pilot manifests select different trajectory rows")
    if probe.get("probe_kl_mask_protocol") != LOCAL_FORK_SPEC.probe_kl_mask_protocol:
        raise ValueError("local-fork pilot probe mask protocol changed")
    model_spec = bundle_payload.get("model_spec")
    if not isinstance(model_spec, dict):
        raise ValueError("local-fork bundle lacks its model specification")
    expected_prompt_protocol = model_spec.get("resolved_prompt_protocol")
    expected_tokenizer_revision = model_spec.get("tokenizer_revision")
    if (
        bank.get("tokenizer_hash")
        != model_spec.get("resolved_tokenizer_fingerprint")
        or prompts.get("prompt_protocol") != expected_prompt_protocol
        or probe.get("prompt_protocol") != expected_prompt_protocol
        or probe.get("tokenizer_revision") != expected_tokenizer_revision
    ):
        raise ValueError("local-fork prompt/probe model protocol binding changed")
    bank_sha256 = evidence["trajectory_bank_content_sha256"]
    if any(
        row.get("trajectory_store_hash") != bank_sha256 for row in (prompts, probe)
    ):
        raise ValueError("local-fork prompt/probe trajectory-bank binding changed")
    trajectory_rows = bundle_payload["trajectories"]
    selected_ids = evidence["selected_trajectory_ids"]
    expected_prompt_ids = [row["prompt_id"] for row in trajectory_rows]
    expected_prompt_texts = [row["prompt_text"] for row in trajectory_rows]
    if (
        prompts.get("trajectory_ids") != selected_ids
        or probe.get("trajectory_ids") != selected_ids
        or prompts.get("prompt_ids") != expected_prompt_ids
        or probe.get("prompt_ids") != expected_prompt_ids
        or prompts.get("prompt_texts") != expected_prompt_texts
        or list(bundle_payload["prompts"]["prompt_ids"]) != expected_prompt_ids
        or list(bundle_payload["prompts"]["prompt_texts"]) != expected_prompt_texts
    ):
        raise ValueError("local-fork prompt/probe rows differ from the frozen trajectories")
    expected_groups: dict[str, list[str]] = defaultdict(list)
    for row in trajectory_rows:
        expected_groups[str(row["generation_group_id"])].append(str(row["trajectory_id"]))
    expected_groups = dict(sorted(expected_groups.items()))
    expected_group_hash = sha256_value(expected_groups)
    if (
        prompts.get("generation_groups") != expected_groups
        or prompts.get("group_membership_hash") != expected_group_hash
        or probe.get("group_membership_hash") != expected_group_hash
    ):
        raise ValueError("local-fork prompt/probe group membership changed")
    probe_rows = probe.get("input_ids")
    probe_masks = probe.get("attention_mask")
    if (
        not isinstance(probe_rows, list)
        or not probe_rows
        or not all(
            isinstance(row, list)
            and row
            and all(type(value) is int and value >= 0 for value in row)
            for row in probe_rows
        )
        or not isinstance(probe_masks, list)
        or len(probe_masks) != len(probe_rows)
        or not all(
            isinstance(mask, list)
            and len(mask) == len(row)
            and all(type(value) is int and value in {0, 1} for value in mask)
            for row, mask in zip(probe_rows, probe_masks, strict=True)
        )
        or bundle_payload["probe_input_ids"].tolist() != probe_rows
        or bundle_payload["probe_attention_mask"].tolist() != probe_masks
    ):
        raise ValueError("local-fork probe inputs/mask differ from the pilot probe manifest")
    from posttrain_circuits.datasets.trajectories.store import TrajectoryStore

    store = TrajectoryStore(expected_paths["trajectory_bank_manifest_path"].parent)
    current_manifest = store.check_integrity()
    if current_manifest.get("sha256") != evidence["trajectory_bank_content_sha256"]:
        raise ValueError("local-fork trajectory bank files changed")
    bank_records = store.read()
    bank_trajectory_ids = [record.trajectory_id for record in bank_records]
    if any(
        not isinstance(value, str) or not value for value in bank_trajectory_ids
    ) or len(bank_trajectory_ids) != len(set(bank_trajectory_ids)):
        raise ValueError(
            "formal local-fork trajectory bank contains duplicate/malformed trajectory_id values"
        )
    records = {record.trajectory_id: record for record in bank_records}
    try:
        selected = [
            asdict(records[trajectory_id])
            for trajectory_id in evidence["selected_trajectory_ids"]
        ]
    except KeyError as error:
        raise ValueError("local-fork selected trajectory disappeared from the bank") from error
    if (
        {row["trajectory_id"]: sha256_value(row) for row in selected}
        != evidence["selected_trajectory_hashes"]
        or sha256_value(selected) != evidence["selected_trajectories_sha256"]
        or selected != bundle_payload["trajectories"]
    ):
        raise ValueError("local-fork selected trajectory bytes differ from the frozen bank")
    for field, expected_path in expected_paths.items():
        digest_field = field.replace("_path", "_file_sha256")
        if evidence[digest_field] != sha256_file(expected_path):
            raise ValueError(f"local-fork {field} changed while being validated")


def create_fork_bundle(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    prompts: PromptBatch,
    trajectories: TrajectoryBatch,
    pre_update_outputs: torch.Tensor,
    probe_input_ids: torch.Tensor | None = None,
    probe_attention_mask: torch.Tensor | None = None,
    manifest_hashes: dict[str, str],
    input_evidence: dict[str, Any] | None = None,
    formal_binding: dict[str, Any] | None = None,
    model_spec: dict[str, Any] | None = None,
    minimum_group_size: int = LOCAL_FORK_SPEC.minimum_group_size,
) -> ForkBundleManifest:
    formal_binding_payload = dict(formal_binding or {})
    if formal_binding_payload:
        from posttrain_circuits.experiments.protocols.local_fork import (
            validate_local_fork_formal_binding,
        )

        validate_local_fork_formal_binding(formal_binding_payload)
    trajectories.validate()
    trajectory_ids = [record.trajectory_id for record in trajectories.records]
    if any(not isinstance(value, str) or not value for value in trajectory_ids) or len(
        trajectory_ids
    ) != len(set(trajectory_ids)):
        raise ValueError("fork bundle trajectories contain duplicate/malformed trajectory_id values")
    if minimum_group_size != LOCAL_FORK_SPEC.minimum_group_size:
        raise ValueError(
            "policy-gradient fork minimum group size differs from the registered local-fork spec"
        )
    groups: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in trajectories.records:
        if not record.behavior_logprobs or record.verifier_reward is None:
            raise ValueError("fork records need behavior log probabilities and exact verifier rewards")
        if not record.teacher_topk_ids:
            raise ValueError("fork records need both hard and soft teacher targets")
        if not record.generation_group_id:
            raise ValueError("fork records require frozen generation_group_id metadata")
        groups[record.generation_group_id].append(record)
    group_membership = {}
    for group_id, records in sorted(groups.items()):
        if len(records) < minimum_group_size:
            raise ValueError(
                f"fork group {group_id} has {len(records)} trajectories; requires {minimum_group_size}"
            )
        if len({record.prompt_id for record in records}) != 1:
            raise ValueError(f"fork group {group_id} mixes prompts")
        if len({float(record.verifier_reward or 0.0) for record in records}) < 2:
            raise ValueError(f"fork group {group_id} has no reward variance")
        if any(record.prompt_group_size != len(records) for record in records):
            raise ValueError(f"fork group {group_id} prompt_group_size metadata is inconsistent")
        indices = sorted(record.generation_group_index for record in records)
        if indices != list(range(len(records))):
            raise ValueError(f"fork group {group_id} indices are not contiguous")
        group_membership[group_id] = [
            record.trajectory_id for record in sorted(records, key=lambda item: item.generation_group_index)
        ]
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("fork bundles currently require torch.optim.AdamW")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise ValueError("fork bundles currently require LambdaLR schedulers")
    model_state = _cpu_tree(model.state_dict())
    optimizer_state = _cpu_tree(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    optimizer_parameter_names = _optimizer_parameter_names(model, optimizer)
    model_parameter_names = [name for name, _ in model.named_parameters()]
    trainable_parameter_names = _trainable_parameter_names(model)
    _validate_adamw_optimizer_contract(
        optimizer_state,
        optimizer_parameter_names,
        model_state=model_state,
        model_parameter_names=model_parameter_names,
        trainable_parameter_names=trainable_parameter_names,
        context="local-fork bundle",
    )
    rng_state = RNGState.capture().as_dict()
    prompt_payload = asdict(prompts)
    trajectory_payload = [asdict(record) for record in trajectories.records]
    if probe_input_ids is None:
        probe_input_ids = torch.tensor([trajectories.records[0].input_ids])
    if probe_attention_mask is None:
        raise ValueError("local-fork bundles require an explicit probe attention mask")
    probe_inputs = probe_input_ids.detach().cpu()
    probe_mask = probe_attention_mask.detach().cpu()
    probe_outputs = pre_update_outputs.detach().cpu()
    if probe_inputs.ndim != 2 or probe_mask.ndim != 2 or probe_outputs.ndim != 3:
        raise ValueError(
            "fork probe inputs/mask/logits must have shapes [batch, seq], [batch, seq], "
            "and [batch, seq, vocab]"
        )
    if probe_mask.shape != probe_inputs.shape:
        raise ValueError("fork probe attention mask does not align with input IDs")
    if torch.is_floating_point(probe_inputs) or torch.is_complex(probe_inputs):
        raise ValueError("fork probe input IDs must use an integer dtype")
    if torch.is_floating_point(probe_mask) or torch.is_complex(probe_mask):
        raise ValueError("fork probe attention mask must use an integer or boolean dtype")
    if not torch.all((probe_mask == 0) | (probe_mask == 1)):
        raise ValueError("fork probe attention mask must be binary")
    if torch.any(probe_mask.sum(dim=1) < 1):
        raise ValueError("every fork probe row must contain at least one valid token")
    if probe_inputs.shape[:2] != probe_outputs.shape[:2]:
        raise ValueError("fork probe inputs and logits do not align")
    _finite_tensor(probe_outputs, name="local-fork pre-update probe logits")
    model_spec_payload = dict(model_spec or {})
    if input_evidence is None:
        if formal_binding_payload:
            raise ValueError("formal local-fork bundles require external input evidence")
        input_evidence_payload = {
            "schema_version": 1,
            "local_fork_protocol_id": LOCAL_FORK_SPEC.protocol_id,
            "local_fork_spec_sha256": LOCAL_FORK_SPEC.scientific_sha256,
            "generator_version": GENERATOR_VERSION,
            "label_semantics": LABEL_SEMANTICS,
            "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
            "probe_kl_mask_protocol": LOCAL_FORK_SPEC.probe_kl_mask_protocol,
            "trajectory_bank_manifest_path": None,
            "trajectory_bank_manifest_file_sha256": None,
            "trajectory_bank_content_sha256": None,
            "prompt_manifest_path": None,
            "prompt_manifest_file_sha256": None,
            "prompt_manifest_payload_sha256": None,
            "probe_manifest_path": None,
            "probe_manifest_file_sha256": None,
            "probe_manifest_payload_sha256": None,
            "selected_trajectory_ids": [row["trajectory_id"] for row in trajectory_payload],
            "selected_trajectory_hashes": {
                row["trajectory_id"]: sha256_value(row) for row in trajectory_payload
            },
            "selected_trajectories_sha256": sha256_value(trajectory_payload),
            "probe_input_hash": state_hash(probe_inputs),
            "probe_attention_mask_hash": state_hash(probe_mask),
            "pre_update_output_hash": state_hash(probe_outputs),
        }
    else:
        input_evidence_payload = copy.deepcopy(input_evidence)
    _validate_input_evidence(
        input_evidence_payload,
        trajectories=trajectory_payload,
        probe_input_ids=probe_inputs,
        probe_attention_mask=probe_mask,
        pre_update_outputs=probe_outputs,
        formal=bool(formal_binding_payload),
    )
    optimizer_class = type(optimizer).__name__
    scheduler_class = type(scheduler).__name__
    manifest = ForkBundleManifest(
        format_version=4,
        protocol_id=LOCAL_FORK_SPEC.protocol_id,
        bundle_id="",
        manifest_sha256="",
        local_fork_spec_sha256=LOCAL_FORK_SPEC.scientific_sha256,
        source_method_id=LOCAL_FORK_SPEC.source_method_id,
        source_seed=LOCAL_FORK_SPEC.source_seed,
        model_spec_hash=sha256_value(model_spec_payload),
        model_parameter_names_hash=sha256_value(model_parameter_names),
        optimizer_parameter_names_hash=sha256_value(optimizer_parameter_names),
        trainable_parameter_names_hash=sha256_value(trainable_parameter_names),
        input_evidence_hash=sha256_value(input_evidence_payload),
        optimizer_class=optimizer_class,
        scheduler_class=scheduler_class,
        checkpoint_hash=state_hash(model_state),
        optimizer_hash=state_hash(optimizer_state),
        optimizer_moment_hash=state_hash(optimizer_state["state"]),
        scheduler_hash=state_hash(scheduler_state),
        rng_hash=state_hash(rng_state),
        prompt_hash=sha256_value(prompt_payload),
        trajectory_hash=sha256_value(trajectory_payload),
        probe_input_hash=state_hash(probe_inputs),
        probe_attention_mask_hash=state_hash(probe_mask),
        probe_output_hash=state_hash(probe_outputs),
        manifest_hashes=manifest_hashes,
        formal_binding=formal_binding_payload,
        policy_version=trajectories.policy_version,
        group_membership_hash=sha256_value(group_membership),
        minimum_group_size=minimum_group_size,
        branch_ids=LOCAL_FORK_SPEC.branch_ids,
        horizons=LOCAL_FORK_SPEC.horizons,
        primary_horizon=LOCAL_FORK_SPEC.primary_horizon,
        require_within_group_reward_variance=(
            LOCAL_FORK_SPEC.require_within_group_reward_variance
        ),
        require_frozen_group_advantages=LOCAL_FORK_SPEC.require_frozen_group_advantages,
        require_old_policy_logprobs=LOCAL_FORK_SPEC.require_old_policy_logprobs,
        clip_epsilon=LOCAL_FORK_SPEC.clip_epsilon,
        divergence=LOCAL_FORK_SPEC.divergence,
        probe_kl_mask_protocol=LOCAL_FORK_SPEC.probe_kl_mask_protocol,
        primary_comparison_axis=LOCAL_FORK_SPEC.primary_comparison_axis,
        secondary_comparison_axes=LOCAL_FORK_SPEC.secondary_comparison_axes,
        output_kl_relative_tolerance=LOCAL_FORK_SPEC.output_kl_relative_tolerance,
        maximum_calibration_rounds=LOCAL_FORK_SPEC.maximum_calibration_rounds,
        calibration_minimum_learning_rate=(
            LOCAL_FORK_SPEC.calibration_minimum_learning_rate
        ),
        calibration_maximum_scale=LOCAL_FORK_SPEC.calibration_maximum_scale,
        calibration_numeric_tolerance=LOCAL_FORK_SPEC.calibration_numeric_tolerance,
        out_of_tolerance_action=LOCAL_FORK_SPEC.out_of_tolerance_action,
    )
    if formal_binding_payload:
        source = formal_binding_payload["local_fork_source"]
        expected_source_state = {
            "model": manifest.checkpoint_hash,
            "model_parameter_names": manifest.model_parameter_names_hash,
            "trainable_parameter_names": manifest.trainable_parameter_names_hash,
            "optimizer": manifest.optimizer_hash,
            "optimizer_moments": manifest.optimizer_moment_hash,
            "optimizer_parameter_names": manifest.optimizer_parameter_names_hash,
            "optimizer_named": state_hash(
                _optimizer_state_by_parameter_name(
                    optimizer_state,
                    optimizer_parameter_names,
                )
            ),
            "scheduler": manifest.scheduler_hash,
            "rng": manifest.rng_hash,
        }
        if source["effective_bundle_state_hashes"] != expected_source_state:
            raise ValueError("formal local-fork bundle differs from its source checkpoint state")
        if source["trajectory_bank_sha256"] != manifest.manifest_hashes.get("bank"):
            raise ValueError("formal local-fork bundle differs from its trajectory-bank ancestry")
    manifest_content = {
        key: value
        for key, value in asdict(manifest).items()
        if key not in {"bundle_id", "manifest_sha256"}
    }
    manifest_sha256 = sha256_value(manifest_content)
    bundle_id = "fork-" + manifest_sha256[:16]
    manifest = ForkBundleManifest(
        bundle_id=bundle_id,
        manifest_sha256=manifest_sha256,
        **{
            key: value
            for key, value in asdict(manifest).items()
            if key not in {"bundle_id", "manifest_sha256"}
        },
    )
    bundle_payload = {
        "manifest": asdict(manifest),
        "model_spec": model_spec_payload,
        "model": model_state,
        "model_parameter_names": model_parameter_names,
        "trainable_parameter_names": trainable_parameter_names,
        "input_evidence": input_evidence_payload,
        "optimizer": optimizer_state,
        "optimizer_parameter_names": optimizer_parameter_names,
        "optimizer_class": optimizer_class,
        "scheduler": scheduler_state,
        "scheduler_class": scheduler_class,
        "rng": rng_state,
        "prompts": prompt_payload,
        "trajectories": trajectory_payload,
        "probe_input_ids": probe_inputs,
        "probe_attention_mask": probe_mask,
        "pre_update_outputs": probe_outputs,
    }
    try:
        publish_torch_once(path, bundle_payload)
    except FileExistsError:
        existing_path = _require_regular_confined_file(
            path,
            root=path.parent,
            name="existing local-fork bundle",
        )
        existing = torch.load(existing_path, map_location="cpu", weights_only=False)
        if not isinstance(existing, dict) or state_hash(existing) != state_hash(bundle_payload):
            raise FileExistsError(f"conflicting local-fork bundle already exists: {path}")
    return manifest


def _validate_fork_bundle_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("local-fork bundle payload must be a mapping")
    expected_payload_fields = {
        "manifest",
        "model_spec",
        "model",
        "model_parameter_names",
        "trainable_parameter_names",
        "input_evidence",
        "optimizer",
        "optimizer_parameter_names",
        "optimizer_class",
        "scheduler",
        "scheduler_class",
        "rng",
        "prompts",
        "trajectories",
        "probe_input_ids",
        "probe_attention_mask",
        "pre_update_outputs",
    }
    if set(payload) != expected_payload_fields:
        raise ValueError(
            "local-fork bundle fields differ from schema: "
            f"missing={sorted(expected_payload_fields - set(payload))}, "
            f"extra={sorted(set(payload) - expected_payload_fields)}"
        )
    raw_manifest = payload["manifest"]
    if not isinstance(raw_manifest, dict):
        raise ValueError("local-fork bundle manifest must be a mapping")
    expected_manifest_fields = {item.name for item in fields(ForkBundleManifest)}
    if set(raw_manifest) != expected_manifest_fields:
        raise ValueError("local-fork bundle manifest fields differ from format v4")
    try:
        manifest = ForkBundleManifest(**raw_manifest)
    except TypeError as error:
        raise ValueError("local-fork bundle manifest has invalid runtime types") from error
    if type(manifest.format_version) is not int or manifest.format_version != 4:
        raise ValueError("local-fork bundle is not format version 4")
    for name in (
        "manifest_sha256",
        "local_fork_spec_sha256",
        "model_spec_hash",
        "model_parameter_names_hash",
        "optimizer_parameter_names_hash",
        "trainable_parameter_names_hash",
        "input_evidence_hash",
        "checkpoint_hash",
        "optimizer_hash",
        "optimizer_moment_hash",
        "scheduler_hash",
        "rng_hash",
        "prompt_hash",
        "trajectory_hash",
        "probe_input_hash",
        "probe_attention_mask_hash",
        "probe_output_hash",
        "group_membership_hash",
    ):
        _require_sha256_text(getattr(manifest, name), name=f"local-fork manifest {name}")
    if not isinstance(manifest.bundle_id, str) or not manifest.bundle_id.startswith("fork-"):
        raise ValueError("local-fork bundle ID is malformed")
    for name in (
        "source_method_id",
        "protocol_id",
        "optimizer_class",
        "scheduler_class",
        "divergence",
        "probe_kl_mask_protocol",
        "primary_comparison_axis",
        "out_of_tolerance_action",
    ):
        if not isinstance(getattr(manifest, name), str):
            raise ValueError(f"local-fork manifest {name} must be a string")
    for name in (
        "source_seed",
        "policy_version",
        "minimum_group_size",
        "primary_horizon",
        "maximum_calibration_rounds",
    ):
        if type(getattr(manifest, name)) is not int:
            raise ValueError(f"local-fork manifest {name} must be an integer")
    for name in (
        "require_within_group_reward_variance",
        "require_frozen_group_advantages",
        "require_old_policy_logprobs",
    ):
        if type(getattr(manifest, name)) is not bool:
            raise ValueError(f"local-fork manifest {name} must be a boolean")
    for name in (
        "clip_epsilon",
        "output_kl_relative_tolerance",
        "calibration_minimum_learning_rate",
        "calibration_maximum_scale",
        "calibration_numeric_tolerance",
    ):
        value = getattr(manifest, name)
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"local-fork manifest {name} must be a finite float")
    for name in ("branch_ids", "horizons", "secondary_comparison_axes"):
        if not isinstance(getattr(manifest, name), tuple):
            raise ValueError(f"local-fork manifest {name} must be a tuple")
    if type(manifest.minimum_group_size) is not int or (
        manifest.minimum_group_size != LOCAL_FORK_SPEC.minimum_group_size
    ):
        raise ValueError("local-fork bundle group size differs from the registered protocol")
    if type(manifest.policy_version) is not int or manifest.policy_version < 0:
        raise ValueError("local-fork bundle policy_version is invalid")
    if not isinstance(manifest.manifest_hashes, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in manifest.manifest_hashes.items()
    ):
        raise ValueError("local-fork bundle manifest_hashes are malformed")
    from posttrain_circuits.experiments.protocols.local_fork import (
        validate_local_fork_formal_binding,
    )

    validate_local_fork_formal_binding(manifest.formal_binding)
    if not isinstance(payload["model_spec"], dict):
        raise ValueError("local-fork model_spec must be a mapping")
    protocol_expected = {
        "protocol_id": LOCAL_FORK_SPEC.protocol_id,
        "local_fork_spec_sha256": LOCAL_FORK_SPEC.scientific_sha256,
        "source_method_id": LOCAL_FORK_SPEC.source_method_id,
        "source_seed": LOCAL_FORK_SPEC.source_seed,
        "branch_ids": LOCAL_FORK_SPEC.branch_ids,
        "horizons": LOCAL_FORK_SPEC.horizons,
        "primary_horizon": LOCAL_FORK_SPEC.primary_horizon,
        "require_within_group_reward_variance": (
            LOCAL_FORK_SPEC.require_within_group_reward_variance
        ),
        "require_frozen_group_advantages": (
            LOCAL_FORK_SPEC.require_frozen_group_advantages
        ),
        "require_old_policy_logprobs": LOCAL_FORK_SPEC.require_old_policy_logprobs,
        "clip_epsilon": LOCAL_FORK_SPEC.clip_epsilon,
        "divergence": LOCAL_FORK_SPEC.divergence,
        "probe_kl_mask_protocol": LOCAL_FORK_SPEC.probe_kl_mask_protocol,
        "primary_comparison_axis": LOCAL_FORK_SPEC.primary_comparison_axis,
        "secondary_comparison_axes": LOCAL_FORK_SPEC.secondary_comparison_axes,
        "output_kl_relative_tolerance": LOCAL_FORK_SPEC.output_kl_relative_tolerance,
        "maximum_calibration_rounds": LOCAL_FORK_SPEC.maximum_calibration_rounds,
        "calibration_minimum_learning_rate": (
            LOCAL_FORK_SPEC.calibration_minimum_learning_rate
        ),
        "calibration_maximum_scale": LOCAL_FORK_SPEC.calibration_maximum_scale,
        "calibration_numeric_tolerance": LOCAL_FORK_SPEC.calibration_numeric_tolerance,
        "out_of_tolerance_action": LOCAL_FORK_SPEC.out_of_tolerance_action,
    }
    protocol_mismatches = {
        key: {"expected": value, "observed": getattr(manifest, key)}
        for key, value in protocol_expected.items()
        if getattr(manifest, key) != value
    }
    if protocol_mismatches:
        raise ValueError(f"local-fork bundle protocol changed: {protocol_mismatches}")
    manifest_content = {
        key: value
        for key, value in asdict(manifest).items()
        if key not in {"bundle_id", "manifest_sha256"}
    }
    observed_manifest_hash = sha256_value(manifest_content)
    if manifest.manifest_sha256 != observed_manifest_hash:
        raise ValueError("local-fork bundle manifest SHA-256 mismatch")
    if manifest.bundle_id != "fork-" + observed_manifest_hash[:16]:
        raise ValueError("local-fork bundle ID differs from its full manifest hash")
    if payload["optimizer_class"] != manifest.optimizer_class or manifest.optimizer_class != "AdamW":
        raise ValueError("local-fork optimizer class binding changed")
    if payload["scheduler_class"] != manifest.scheduler_class or manifest.scheduler_class != "LambdaLR":
        raise ValueError("local-fork scheduler class binding changed")
    observed_hashes = {
        "model_spec_hash": sha256_value(payload["model_spec"]),
        "model_parameter_names_hash": sha256_value(payload["model_parameter_names"]),
        "optimizer_parameter_names_hash": sha256_value(payload["optimizer_parameter_names"]),
        "trainable_parameter_names_hash": sha256_value(
            payload["trainable_parameter_names"]
        ),
        "input_evidence_hash": sha256_value(payload["input_evidence"]),
        "checkpoint_hash": state_hash(payload["model"]),
        "optimizer_hash": state_hash(payload["optimizer"]),
        "optimizer_moment_hash": state_hash(payload["optimizer"]["state"]),
        "scheduler_hash": state_hash(payload["scheduler"]),
        "rng_hash": state_hash(payload["rng"]),
        "prompt_hash": sha256_value(payload["prompts"]),
        "trajectory_hash": sha256_value(payload["trajectories"]),
        "probe_input_hash": state_hash(payload["probe_input_ids"]),
        "probe_attention_mask_hash": state_hash(payload["probe_attention_mask"]),
        "probe_output_hash": state_hash(payload["pre_update_outputs"]),
    }
    payload_mismatches = {
        key: {"expected": getattr(manifest, key), "observed": value}
        for key, value in observed_hashes.items()
        if getattr(manifest, key) != value
    }
    if payload_mismatches:
        raise ValueError(f"local-fork bundle payload changed: {payload_mismatches}")
    _validate_adamw_optimizer_contract(
        payload["optimizer"],
        payload["optimizer_parameter_names"],
        model_state=payload["model"],
        model_parameter_names=payload["model_parameter_names"],
        trainable_parameter_names=payload["trainable_parameter_names"],
        context="local-fork bundle",
    )
    _validate_input_evidence(
        payload["input_evidence"],
        trajectories=payload["trajectories"],
        probe_input_ids=payload["probe_input_ids"],
        probe_attention_mask=payload["probe_attention_mask"],
        pre_update_outputs=payload["pre_update_outputs"],
        formal=bool(manifest.formal_binding),
    )
    probe_inputs = payload["probe_input_ids"]
    probe_mask = payload["probe_attention_mask"]
    probe_outputs = payload["pre_update_outputs"]
    if not all(isinstance(value, torch.Tensor) for value in (probe_inputs, probe_mask, probe_outputs)):
        raise ValueError("local-fork probe inputs, mask, and outputs must be tensors")
    if probe_inputs.ndim != 2 or probe_mask.shape != probe_inputs.shape:
        raise ValueError("local-fork probe attention mask shape changed")
    if torch.is_floating_point(probe_inputs) or torch.is_complex(probe_inputs):
        raise ValueError("local-fork probe input IDs must use an integer dtype")
    if probe_outputs.ndim != 3 or probe_outputs.shape[:2] != probe_inputs.shape:
        raise ValueError("local-fork probe outputs do not align with the exact probe inputs")
    if torch.is_floating_point(probe_mask) or torch.is_complex(probe_mask):
        raise ValueError("local-fork probe attention mask must use an integer or boolean dtype")
    if not torch.all((probe_mask == 0) | (probe_mask == 1)):
        raise ValueError("local-fork probe attention mask must be binary")
    if torch.any(probe_mask.sum(dim=1) < 1):
        raise ValueError("local-fork probe attention mask contains an empty row")
    _finite_tensor(probe_outputs, name="local-fork bundled baseline probe logits")
    prompts = payload.get("prompts")
    if not isinstance(prompts, dict) or set(prompts) != {"prompt_ids", "prompt_texts"}:
        raise ValueError("local-fork prompt batch fields differ from schema")
    if not all(
        isinstance(prompts[name], tuple)
        and prompts[name]
        and all(isinstance(value, str) and value for value in prompts[name])
        for name in ("prompt_ids", "prompt_texts")
    ) or len(prompts["prompt_ids"]) != len(prompts["prompt_texts"]):
        raise ValueError("local-fork prompt batch is malformed")
    trajectories = payload["trajectories"]
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("local-fork bundle requires trajectories")
    if any(not isinstance(record, dict) for record in trajectories):
        raise ValueError("local-fork trajectory payload must contain mappings")
    trajectory_ids = [record.get("trajectory_id") for record in trajectories]
    if any(not isinstance(value, str) or not value for value in trajectory_ids) or len(
        trajectory_ids
    ) != len(set(trajectory_ids)):
        raise ValueError(
            "local-fork bundle contains duplicate/malformed trajectory_id values"
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in trajectories:
        try:
            typed_record = TrajectoryRecord(**record)
        except TypeError as error:
            raise ValueError("local-fork trajectory fields differ from TrajectoryRecord") from error
        typed_record.validate()
        if not typed_record.response_ids or not any(typed_record.response_token_mask):
            raise ValueError("local-fork trajectories require response tokens in the loss mask")
        if not typed_record.behavior_logprobs or any(
            not math.isfinite(float(value)) for value in typed_record.behavior_logprobs
        ):
            raise ValueError("local-fork trajectories require finite old-policy log-probabilities")
        if (
            isinstance(typed_record.verifier_reward, bool)
            or not isinstance(typed_record.verifier_reward, int | float)
            or not math.isfinite(float(typed_record.verifier_reward))
        ):
            raise ValueError("local-fork trajectories require a finite verifier reward")
        if not typed_record.teacher_topk_ids or not typed_record.teacher_topk_logprobs:
            raise ValueError("local-fork trajectories require frozen hard and soft teacher targets")
        if any(
            not ids
            or len(ids) != len(logprobs)
            or any(not math.isfinite(float(value)) for value in logprobs)
            for ids, logprobs in zip(
                typed_record.teacher_topk_ids,
                typed_record.teacher_topk_logprobs,
                strict=True,
            )
        ):
            raise ValueError("local-fork teacher top-k targets are empty, misaligned, or non-finite")
        group_id = record.get("generation_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("local-fork trajectory is missing generation-group identity")
        groups[group_id].append(record)
    group_membership = {}
    for group_id, records in sorted(groups.items()):
        if len(records) < manifest.minimum_group_size:
            raise ValueError("local-fork bundle contains a group below minimum_group_size")
        if len({record.get("prompt_id") for record in records}) != 1:
            raise ValueError("local-fork bundle generation group mixes prompts")
        if len({record.get("verifier_reward") for record in records}) < 2:
            raise ValueError("local-fork bundle generation group has no reward variance")
        ordered = sorted(records, key=lambda record: int(record["generation_group_index"]))
        if [record.get("generation_group_index") for record in ordered] != list(
            range(len(ordered))
        ):
            raise ValueError("local-fork bundle generation-group indices are not contiguous")
        if any(record.get("prompt_group_size") != len(ordered) for record in ordered):
            raise ValueError("local-fork bundle prompt_group_size metadata changed")
        group_membership[group_id] = [record.get("trajectory_id") for record in ordered]
    if sha256_value(group_membership) != manifest.group_membership_hash:
        raise ValueError("local-fork bundle group-membership hash mismatch")
    if manifest.formal_binding:
        validate_local_fork_source_binding_files(
            manifest.formal_binding["local_fork_source"],
            bundle_payload=payload,
        )
    return payload


def load_fork_bundle(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _validate_fork_bundle_payload(payload)


def _validate_probe_output_recomputation(
    *,
    model_state: dict[str, Any],
    model_parameter_names: list[str],
    trainable_parameter_names: list[str],
    probe_input_ids: torch.Tensor,
    probe_attention_mask: torch.Tensor,
    expected_outputs: torch.Tensor,
    model_factory: Callable[[], torch.nn.Module],
    context: str,
) -> None:
    model = model_factory()
    if [name for name, _ in model.named_parameters()] != model_parameter_names:
        raise ValueError(f"{context} model parameter mapping changed")
    if _trainable_parameter_names(model) != trainable_parameter_names:
        raise ValueError(f"{context} trainable parameter mapping changed")
    model.load_state_dict(model_state)
    device = next(model.parameters()).device
    inputs = probe_input_ids.to(device)
    attention_mask = probe_attention_mask.to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        observed = model(
            input_ids=inputs,
            attention_mask=attention_mask,
        ).logits.detach().cpu()
    model.train(was_training)
    if observed.shape != expected_outputs.shape or observed.dtype != expected_outputs.dtype:
        raise ValueError(f"{context} recomputed probe logits changed shape/dtype")
    if not torch.equal(observed, expected_outputs):
        raise ValueError(f"{context} recomputed probe logits differ from checkpoint evidence")


def validate_bundle_probe_recomputation(
    payload: dict[str, Any],
    *,
    model_factory: Callable[[], torch.nn.Module],
) -> None:
    """Recompute the fork baseline from the externally restored source model."""

    _validate_probe_output_recomputation(
        model_state=payload["model"],
        model_parameter_names=payload["model_parameter_names"],
        trainable_parameter_names=payload["trainable_parameter_names"],
        probe_input_ids=payload["probe_input_ids"],
        probe_attention_mask=payload["probe_attention_mask"],
        expected_outputs=payload["pre_update_outputs"],
        model_factory=model_factory,
        context="local-fork source baseline",
    )


def validate_checkpoint_probe_recomputation(
    payload: dict[str, Any],
    *,
    model_factory: Callable[[], torch.nn.Module],
) -> None:
    """Recompute one post-update probe result from the saved model bytes."""

    _validate_probe_output_recomputation(
        model_state=payload["model"],
        model_parameter_names=payload["model_parameter_names"],
        trainable_parameter_names=payload["trainable_parameter_names"],
        probe_input_ids=payload["probe_input_ids"],
        probe_attention_mask=payload["probe_attention_mask"],
        expected_outputs=payload["observed_probe_outputs"],
        model_factory=model_factory,
        context="local-fork post checkpoint",
    )


def _restore_optimizer(
    model: torch.nn.Module,
    state: dict[str, Any],
    parameter_names: list[list[str]],
    trainable_parameter_names: list[str],
    optimizer_class: str,
) -> torch.optim.Optimizer:
    state = copy.deepcopy(state)
    if optimizer_class != "AdamW":
        raise ValueError(f"unsupported bundled optimizer class {optimizer_class!r}")
    named_parameters = dict(model.named_parameters())
    if len(parameter_names) != len(state["param_groups"]):
        raise ValueError("optimizer parameter-group metadata is inconsistent")
    groups = []
    for saved_group, saved_names in zip(
        state["param_groups"],
        parameter_names,
        strict=True,
    ):
        group = {
            key: copy.deepcopy(value)
            for key, value in saved_group.items()
            if key not in {"params", "initial_lr"}
        }
        if len(saved_names) != len(saved_group["params"]):
            raise ValueError("optimizer parameter-name metadata is inconsistent")
        try:
            group["params"] = [named_parameters[name] for name in saved_names]
        except KeyError as error:
            raise ValueError(f"bundled optimizer parameter is absent from model: {error}") from error
        groups.append(group)
    # PyTorch optimizers require construction before load_state_dict; no update
    # occurs with this temporary shell, and the complete moment state is loaded
    # immediately below.
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(state["param_groups"][0]["lr"]),
    )
    optimizer.load_state_dict(state)
    _validate_adamw_optimizer_contract(
        optimizer.state_dict(),
        _optimizer_parameter_names(model, optimizer),
        model_state=model.state_dict(),
        model_parameter_names=[name for name, _ in model.named_parameters()],
        trainable_parameter_names=trainable_parameter_names,
        context="restored local-fork optimizer",
    )
    return optimizer


def restore_bundle_fresh(
    payload: dict[str, Any],
    *,
    model_factory: Callable[[], torch.nn.Module],
) -> RestoredFork:
    model = model_factory()
    if [name for name, _ in model.named_parameters()] != payload["model_parameter_names"]:
        raise ValueError("fresh local-fork model parameter mapping changed")
    if _trainable_parameter_names(model) != payload["trainable_parameter_names"]:
        raise ValueError("fresh local-fork model trainable parameter mapping changed")
    model.load_state_dict(payload["model"])
    optimizer = _restore_optimizer(
        model,
        payload["optimizer"],
        payload["optimizer_parameter_names"],
        payload["trainable_parameter_names"],
        str(payload["optimizer_class"]),
    )
    if payload["scheduler_class"] != "LambdaLR":
        raise ValueError(f"unsupported bundled scheduler class {payload['scheduler_class']!r}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _: 1.0,
    )
    scheduler.load_state_dict(copy.deepcopy(payload["scheduler"]))
    # Model construction consumes RNG. Restore the captured state only after all
    # branch-local objects exist so every branch begins at the exact same draw.
    RNGState(**payload["rng"]).restore()
    prompts = PromptBatch(
        tuple(payload["prompts"]["prompt_ids"]),
        tuple(payload["prompts"]["prompt_texts"]),
    )
    records = [TrajectoryRecord(**record) for record in payload["trajectories"]]
    trajectories = TrajectoryBatch(
        records,
        int(payload["manifest"]["policy_version"]),
    )
    hashes = {
        "model": state_hash(model.state_dict()),
        "optimizer": state_hash(optimizer.state_dict()),
        "optimizer_moments": state_hash(optimizer.state_dict()["state"]),
        "scheduler": state_hash(scheduler.state_dict()),
        "rng": state_hash(RNGState.capture().as_dict()),
        "prompts": sha256_value(asdict(prompts)),
        "trajectories": sha256_value([asdict(record) for record in records]),
        "probe_inputs": state_hash(payload["probe_input_ids"]),
        "probe_attention_mask": state_hash(payload["probe_attention_mask"]),
        "probe_outputs": state_hash(payload["pre_update_outputs"]),
    }
    expected = {
        "model": payload["manifest"]["checkpoint_hash"],
        "optimizer": payload["manifest"]["optimizer_hash"],
        "optimizer_moments": payload["manifest"]["optimizer_moment_hash"],
        "scheduler": payload["manifest"]["scheduler_hash"],
        "rng": payload["manifest"]["rng_hash"],
        "prompts": payload["manifest"]["prompt_hash"],
        "trajectories": payload["manifest"]["trajectory_hash"],
        "probe_inputs": payload["manifest"]["probe_input_hash"],
        "probe_attention_mask": payload["manifest"]["probe_attention_mask_hash"],
        "probe_outputs": payload["manifest"]["probe_output_hash"],
    }
    if hashes != expected:
        raise ValueError(f"fork bundle restoration mismatch: expected={expected}, actual={hashes}")
    return RestoredFork(
        model,
        optimizer,
        scheduler,
        prompts,
        trajectories,
        payload["probe_input_ids"].clone(),
        payload["probe_attention_mask"].clone(),
        payload["pre_update_outputs"].clone(),
        hashes,
    )


class SharedTrajectoryUncenteredReinforceDiagnostic:
    """Historical binary estimator retained only for replay-collinearity tests."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def prepare_targets(
        self,
        trajectories: TrajectoryBatch,
        teacher: Any,
        verifier: Any,
    ) -> SupervisionBatch:
        del teacher, verifier
        return collate_trajectories(
            trajectories,
            pad_token_id=self.pad_token_id,
        )

    def compute_loss(
        self,
        model: Any,
        batch: SupervisionBatch,
    ) -> LossOutput:
        assert batch.rewards is not None
        logits = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        ).logits[:, :-1]
        labels = batch.input_ids[:, 1:]
        mask = batch.response_mask[:, 1:]
        token_logprob = -functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        lengths = mask.sum(dim=1).clamp_min(1)
        sequence_logprob = (token_logprob * mask).sum(dim=1) / lengths
        loss = -(batch.rewards * sequence_logprob).mean()
        return LossOutput(
            loss,
            {"uncentered_reinforce_loss": float(loss.detach())},
        )


class SharedTrajectoryCenteredPolicyGradientSupervisor:
    """Frozen grouped advantages with an old-policy clipped surrogate."""

    def __init__(
        self,
        pad_token_id: int,
        *,
        clip_epsilon: float = LOCAL_FORK_SPEC.clip_epsilon,
        advantage_epsilon: float = 1e-6,
    ) -> None:
        if not 0 < clip_epsilon < 1:
            raise ValueError("centered policy-gradient clip epsilon must be in (0, 1)")
        self.pad_token_id = pad_token_id
        self.clip_epsilon = clip_epsilon
        self.advantage_epsilon = advantage_epsilon

    def prepare_targets(
        self,
        trajectories: TrajectoryBatch,
        teacher: Any,
        verifier: Any,
    ) -> SupervisionBatch:
        del teacher, verifier
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(trajectories.records):
            if not record.generation_group_id:
                raise ValueError("centered policy gradient requires generation groups")
            groups[record.generation_group_id].append(index)
        advantages = torch.empty(len(trajectories.records), dtype=torch.float32)
        for group_id, indices in groups.items():
            rewards = torch.tensor(
                [float(trajectories.records[index].verifier_reward or 0.0) for index in indices],
                dtype=torch.float32,
            )
            if len(indices) < 2 or float(rewards.var(unbiased=False)) <= 0.0:
                raise ValueError(f"centered policy-gradient group {group_id} has no reward variance")
            centered = rewards - rewards.mean()
            scaled = centered / (rewards.std(unbiased=False) + self.advantage_epsilon)
            advantages[torch.tensor(indices)] = scaled
        batch = collate_trajectories(trajectories, pad_token_id=self.pad_token_id)
        old_sequence_logprobs = []
        for record in trajectories.records:
            included = [
                value
                for value, include in zip(record.behavior_logprobs, record.response_token_mask, strict=True)
                if include
            ]
            if not included:
                raise ValueError("centered policy gradient requires behavior log-probabilities")
            old_sequence_logprobs.append(sum(included))
        batch.metadata.update(
            {
                "generation_group_ids": [record.generation_group_id for record in trajectories.records],
                "frozen_advantages": advantages,
                "old_sequence_logprobs": torch.tensor(old_sequence_logprobs, dtype=torch.float32),
                "clip_epsilon": self.clip_epsilon,
            }
        )
        return batch

    def compute_loss(self, model: Any, batch: SupervisionBatch) -> LossOutput:
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits[:, :-1]
        labels = batch.input_ids[:, 1:]
        mask = batch.response_mask[:, 1:]
        token_logprob = -functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape_as(labels)
        sequence_logprob = (token_logprob * mask).sum(dim=1)
        advantages = batch.metadata["frozen_advantages"].to(sequence_logprob.device)
        old = batch.metadata["old_sequence_logprobs"].to(sequence_logprob.device)
        ratio = torch.exp((sequence_logprob - old).clamp(-20.0, 20.0))
        clipped = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        surrogate = torch.minimum(ratio * advantages, clipped * advantages)
        loss = -surrogate.mean()
        return LossOutput(
            loss,
            {
                "centered_policy_gradient_loss": float(loss.detach()),
                "positive_advantage_count": float((advantages > 0).sum()),
                "negative_advantage_count": float((advantages < 0).sum()),
                "mean_probability_ratio": float(ratio.mean().detach()),
            },
        )


def _branch_supervisor(branch: str, pad_token_id: int) -> Supervisor:
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("local-fork pad token ID must be a non-negative integer")
    supervisors: dict[str, Supervisor] = {
        "hard_teacher": HardTeacherSupervisor(pad_token_id),
        "soft_teacher": SoftTeacherSupervisor(pad_token_id),
        "verified_replay": VerifiedReplaySupervisor(pad_token_id),
        "centered_policy_gradient": SharedTrajectoryCenteredPolicyGradientSupervisor(
            pad_token_id
        ),
    }
    if tuple(supervisors) != LOCAL_FORK_SPEC.branch_ids:
        raise RuntimeError("local-fork implementation dispatch differs from LocalForkSpec")
    try:
        return supervisors[branch]
    except KeyError as error:
        raise ValueError(f"unknown fork branch {branch!r}") from error


def _probe_kl(
    initial_logits: torch.Tensor,
    current_logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    if initial_logits.shape != current_logits.shape or initial_logits.ndim != 3:
        raise ValueError("local-fork KL logits must share [batch, sequence, vocabulary] shape")
    if attention_mask.shape != initial_logits.shape[:2]:
        raise ValueError("local-fork KL mask does not align with logits")
    valid = attention_mask.to(dtype=torch.bool, device=current_logits.device)
    valid_count = int(valid.sum().item())
    if valid_count < 1:
        raise ValueError("local-fork KL requires at least one non-padding probe position")
    fork_log = initial_logits.float().log_softmax(dim=-1)
    new_log = current_logits.float().log_softmax(dim=-1)
    # Primary behavioral displacement: KL(output_new || output_fork).
    per_position = (new_log.exp() * (new_log - fork_log)).sum(dim=-1)
    return float(per_position.masked_select(valid).sum() / valid_count)


def recompute_parameter_update_norm(
    pre_model_state: dict[str, Any],
    post_model_state: dict[str, Any],
    parameter_names: list[str],
) -> float:
    """Recompute the full parameter displacement from checkpoint tensor bytes."""

    if set(pre_model_state) != set(post_model_state):
        raise ValueError("local-fork pre/post model state keys differ")
    if len(parameter_names) != len(set(parameter_names)) or any(
        name not in pre_model_state for name in parameter_names
    ):
        raise ValueError("local-fork parameter-norm mapping is incomplete or ambiguous")
    total = torch.zeros((), dtype=torch.float64)
    for name in parameter_names:
        before = pre_model_state[name]
        after = post_model_state[name]
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            raise ValueError(f"local-fork parameter {name!r} is not a tensor")
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"local-fork parameter {name!r} shape/dtype changed")
        delta = after.detach().cpu().double() - before.detach().cpu().double()
        if not bool(torch.isfinite(delta).all().item()):
            raise ValueError(f"local-fork parameter {name!r} update is non-finite")
        total += delta.square().sum()
    return float(total.sqrt().item())


def validate_branch_deterministic_replay(
    pre_checkpoint: dict[str, Any],
    post_checkpoint: dict[str, Any],
    *,
    bundle_payload: dict[str, Any],
    model_factory: Callable[[], torch.nn.Module],
) -> dict[str, Any]:
    """Replay every update from the pre checkpoint and verify independent evidence."""

    if not callable(model_factory):
        raise ValueError("local-fork deterministic replay requires a model factory")
    linkage_fields = (
        "bundle_id",
        "bundle_manifest_sha256",
        "local_fork_protocol_id",
        "local_fork_spec_sha256",
        "branch",
        "horizon",
        "comparison_mode",
        "calibration_round",
        "learning_rate",
        "learning_rate_override",
        "pad_token_id",
        "model_parameter_names",
        "trainable_parameter_names",
        "optimizer_parameter_names",
        "prompts",
        "trajectories",
        "probe_input_ids",
        "probe_attention_mask",
        "pre_update_outputs",
        "initial_hashes",
    )
    for field in linkage_fields:
        if state_hash(pre_checkpoint.get(field)) != state_hash(post_checkpoint.get(field)):
            raise ValueError(
                f"local-fork deterministic replay pre/post field {field!r} changed"
            )
    horizon = pre_checkpoint.get("horizon")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("local-fork deterministic replay horizon is invalid")
    branch = pre_checkpoint.get("branch")
    if not isinstance(branch, str):
        raise ValueError("local-fork deterministic replay branch is invalid")
    pad_token_id = pre_checkpoint.get("pad_token_id")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("local-fork deterministic replay pad token ID is invalid")
    validate_branch_optimizer_transition(
        pre_checkpoint,
        post_checkpoint,
        bundle_payload=bundle_payload,
        horizon=horizon,
    )

    model = model_factory()
    if [name for name, _ in model.named_parameters()] != pre_checkpoint[
        "model_parameter_names"
    ]:
        raise ValueError("local-fork replay model parameter mapping changed")
    if _trainable_parameter_names(model) != pre_checkpoint["trainable_parameter_names"]:
        raise ValueError("local-fork replay trainable parameter mapping changed")
    model.load_state_dict(copy.deepcopy(pre_checkpoint["model"]))
    model.train(True)
    model.zero_grad(set_to_none=True)
    optimizer = _restore_optimizer(
        model,
        pre_checkpoint["optimizer"],
        pre_checkpoint["optimizer_parameter_names"],
        pre_checkpoint["trainable_parameter_names"],
        str(bundle_payload["optimizer_class"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scheduler.load_state_dict(copy.deepcopy(pre_checkpoint["scheduler"]))
    # LambdaLR construction consults ``initial_lr`` and performs an initial
    # scheduler step. Reload the exact pre optimizer groups afterward so an
    # explicitly calibrated LR cannot be reset to the bundled base LR.
    optimizer.load_state_dict(copy.deepcopy(pre_checkpoint["optimizer"]))
    if state_hash(model.state_dict()) != pre_checkpoint["state_hashes"]["model"]:
        raise ValueError("local-fork replay did not restore the pre model exactly")
    if state_hash(optimizer.state_dict()) != pre_checkpoint["state_hashes"]["optimizer"]:
        raise ValueError("local-fork replay did not restore the pre optimizer exactly")
    if state_hash(scheduler.state_dict()) != pre_checkpoint["state_hashes"]["scheduler"]:
        raise ValueError("local-fork replay did not restore the pre scheduler exactly")

    prompt_payload = pre_checkpoint["prompts"]
    if not isinstance(prompt_payload, dict) or set(prompt_payload) != {
        "prompt_ids",
        "prompt_texts",
    }:
        raise ValueError("local-fork replay prompt batch is malformed")
    prompts = PromptBatch(
        tuple(prompt_payload["prompt_ids"]),
        tuple(prompt_payload["prompt_texts"]),
    )
    if sha256_value(asdict(prompts)) != pre_checkpoint["state_hashes"]["prompts"]:
        raise ValueError("local-fork replay prompt bytes changed")
    try:
        records = [
            TrajectoryRecord(**copy.deepcopy(record))
            for record in pre_checkpoint["trajectories"]
        ]
    except (TypeError, KeyError) as error:
        raise ValueError("local-fork replay trajectories are malformed") from error
    trajectory_ids = [record.trajectory_id for record in records]
    if any(not value for value in trajectory_ids) or len(trajectory_ids) != len(
        set(trajectory_ids)
    ):
        raise ValueError("local-fork replay trajectories contain duplicate identities")
    policy_version = bundle_payload["manifest"]["policy_version"]
    trajectories = TrajectoryBatch(records, int(policy_version))
    trajectories.validate(max_policy_lag=0)
    if any(record.policy_version != trajectories.policy_version for record in records):
        raise ValueError("local-fork replay trajectories mix policy versions")
    if sha256_value([asdict(record) for record in records]) != pre_checkpoint[
        "state_hashes"
    ]["trajectories"]:
        raise ValueError("local-fork replay trajectory bytes changed")

    supervisor = _branch_supervisor(branch, pad_token_id)
    prepared = supervisor.prepare_targets(trajectories, None, None)
    prepared = _move_supervision_batch(prepared, next(model.parameters()).device)
    # Target preparation happened before the original pre checkpoint captured RNG.
    # Restore afterward so the first replayed update begins from those exact bytes.
    RNGState(**copy.deepcopy(pre_checkpoint["rng"])).restore()
    if state_hash(RNGState.capture().as_dict()) != pre_checkpoint["state_hashes"]["rng"]:
        raise ValueError("local-fork replay did not restore the pre RNG exactly")

    step_losses: list[float] = []
    step_metrics: list[dict[str, Any]] = []
    for _ in range(horizon):
        output = supervisor.compute_loss(model, prepared)
        loss = float(output.loss.detach())
        metrics = _cpu_tree(output.metrics)
        if not math.isfinite(loss) or not isinstance(metrics, dict):
            raise ValueError("local-fork deterministic replay produced malformed loss evidence")
        output.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step_losses.append(loss)
        step_metrics.append(metrics)

    probe_ids = pre_checkpoint["probe_input_ids"].to(next(model.parameters()).device)
    probe_mask = pre_checkpoint["probe_attention_mask"].to(probe_ids.device)
    model.eval()
    with torch.no_grad():
        observed_probe_outputs = model(
            input_ids=probe_ids,
            attention_mask=probe_mask,
        ).logits.detach().cpu()
    model.train(True)
    replay_model = model.state_dict()
    replay_optimizer = optimizer.state_dict()
    replay_scheduler = scheduler.state_dict()
    replay_rng = RNGState.capture().as_dict()
    replay_hashes = {
        "model": state_hash(replay_model),
        "optimizer": state_hash(replay_optimizer),
        "optimizer_moments": state_hash(replay_optimizer["state"]),
        "scheduler": state_hash(replay_scheduler),
        "rng": state_hash(replay_rng),
        "observed_probe_outputs": state_hash(observed_probe_outputs),
    }
    expected_hashes = {
        key: post_checkpoint["state_hashes"][key]
        for key in replay_hashes
    }
    if replay_hashes != expected_hashes:
        raise ValueError(
            "local-fork deterministic replay state differs from post checkpoint: "
            f"expected={expected_hashes}, observed={replay_hashes}"
        )
    loss_evidence = post_checkpoint["loss_evidence"]
    replay_loss_evidence = {
        "step_losses": step_losses,
        "step_metrics": step_metrics,
        "final_loss": step_losses[-1],
    }
    expected_loss_evidence = {
        key: loss_evidence[key] for key in replay_loss_evidence
    }
    if state_hash(replay_loss_evidence) != state_hash(expected_loss_evidence):
        raise ValueError(
            "local-fork deterministic replay loss/step metrics differ from post checkpoint"
        )
    return {
        **replay_loss_evidence,
        "state_hashes": replay_hashes,
    }


def run_branch(
    *,
    bundle_payload: dict[str, Any],
    model_factory: Callable[[], torch.nn.Module],
    branch: str,
    horizon: int,
    pad_token_id: int,
    checkpoint_root: Path,
    comparison_mode: str,
    calibration_round: int,
    learning_rate_override: float | None = None,
) -> dict[str, Any]:
    if horizon not in LOCAL_FORK_SPEC.horizons:
        raise ValueError("fork horizon is not registered in LocalForkSpec")
    restored = restore_bundle_fresh(
        bundle_payload,
        model_factory=model_factory,
    )
    if learning_rate_override is not None:
        for group in restored.optimizer.param_groups:
            group["lr"] = learning_rate_override
        restored.scheduler.base_lrs = [learning_rate_override for _ in restored.scheduler.base_lrs]
    restored.model.train(True)
    restored.model.zero_grad(set_to_none=True)
    before_model_state = _cpu_tree(restored.model.state_dict())
    model_parameter_names = [name for name, _ in restored.model.named_parameters()]
    supervisor = _branch_supervisor(branch, pad_token_id)
    branch_spec = next(
        specification
        for specification in LOCAL_FORK_SPEC.branches
        if specification.branch_id == branch
    )
    prepared = supervisor.prepare_targets(
        restored.trajectories,
        None,
        None,
    )
    prepared = _move_supervision_batch(
        prepared,
        next(restored.model.parameters()).device,
    )
    pre_checkpoint = checkpoint_root / "pre.pt"
    post_checkpoint = checkpoint_root / "post.pt"
    pre_checkpoint_evidence = _write_branch_checkpoint(
        pre_checkpoint,
        phase="pre",
        bundle_payload=bundle_payload,
        branch=branch,
        horizon=horizon,
        comparison_mode=comparison_mode,
        calibration_round=calibration_round,
        restored=restored,
        learning_rate_override=learning_rate_override,
        pad_token_id=pad_token_id,
        observed_probe_outputs=restored.pre_update_outputs,
        loss_evidence={
            "schema_version": 1,
            "phase": "pre",
            "objective": branch_spec.objective,
            "step_losses": [],
            "step_metrics": [],
            "final_loss": None,
        },
    )
    losses = []
    step_metrics = []
    for _ in range(horizon):
        output = supervisor.compute_loss(restored.model, prepared)
        output.loss.backward()
        restored.optimizer.step()
        restored.scheduler.step()
        restored.optimizer.zero_grad(set_to_none=True)
        losses.append(float(output.loss.detach()))
        step_metrics.append(output.metrics)
    probe_ids = restored.probe_input_ids.to(next(restored.model.parameters()).device)
    probe_mask = restored.probe_attention_mask.to(probe_ids.device)
    was_training = restored.model.training
    restored.model.eval()
    with torch.no_grad():
        current_outputs = restored.model(
            input_ids=probe_ids,
            attention_mask=probe_mask,
        ).logits.cpu()
    restored.model.train(was_training)
    post_checkpoint_evidence = _write_branch_checkpoint(
        post_checkpoint,
        phase="post",
        bundle_payload=bundle_payload,
        branch=branch,
        horizon=horizon,
        comparison_mode=comparison_mode,
        calibration_round=calibration_round,
        restored=restored,
        learning_rate_override=learning_rate_override,
        pad_token_id=pad_token_id,
        observed_probe_outputs=current_outputs,
        loss_evidence={
            "schema_version": 1,
            "phase": "post",
            "objective": branch_spec.objective,
            "step_losses": losses,
            "step_metrics": step_metrics,
            "final_loss": losses[-1],
        },
    )
    return {
        "bundle_id": bundle_payload["manifest"]["bundle_id"],
        "bundle_manifest_sha256": bundle_payload["manifest"]["manifest_sha256"],
        "local_fork_protocol_id": bundle_payload["manifest"]["protocol_id"],
        "local_fork_spec_sha256": bundle_payload["manifest"]["local_fork_spec_sha256"],
        "trajectory_bank_sha256": bundle_payload["manifest"]["manifest_hashes"].get("bank"),
        "branch": branch,
        "horizon": horizon,
        "comparison_mode": comparison_mode,
        "calibration_round": calibration_round,
        "initial_hashes": restored.initial_hashes,
        "initial_parameter_hash": restored.initial_hashes["model"],
        "initial_optimizer_moment_hash": restored.initial_hashes["optimizer_moments"],
        "initial_scheduler_hash": restored.initial_hashes["scheduler"],
        "initial_rng_hash": restored.initial_hashes["rng"],
        "initial_trajectory_hash": restored.initial_hashes["trajectories"],
        "probe_input_hash": restored.initial_hashes["probe_inputs"],
        "probe_attention_mask_hash": restored.initial_hashes["probe_attention_mask"],
        "initial_probe_output_hash": restored.initial_hashes["probe_outputs"],
        "probe_kl_valid_token_count": int(restored.probe_attention_mask.sum().item()),
        "optimizer_updates": horizon,
        "step_losses": losses,
        "step_metrics": step_metrics,
        "group_membership_hash": bundle_payload["manifest"]["group_membership_hash"],
        "loss": losses[-1],
        "parameter_update_norm": recompute_parameter_update_norm(
            before_model_state,
            _cpu_tree(restored.model.state_dict()),
            model_parameter_names,
        ),
        "probe_output_kl_new_to_fork": _probe_kl(
            restored.pre_update_outputs,
            current_outputs,
            restored.probe_attention_mask,
        ),
        "learning_rate": float(restored.optimizer.param_groups[0]["lr"]),
        "checkpoint_evidence": {
            "pre": pre_checkpoint_evidence,
            "post": post_checkpoint_evidence,
        },
        "post_model_hash": state_hash(restored.model.state_dict()),
        "post_optimizer_hash": state_hash(restored.optimizer.state_dict()),
        "post_optimizer_moment_hash": state_hash(restored.optimizer.state_dict()["state"]),
        "post_scheduler_hash": state_hash(restored.scheduler.state_dict()),
        "post_rng_hash": state_hash(RNGState.capture().as_dict()),
    }


def calibrate_learning_rate_for_update_norm(
    reference_norm: float,
    observed_norm: float,
    learning_rate: float,
    *,
    minimum: float = 1e-8,
) -> float:
    if reference_norm <= 0 or observed_norm <= 0:
        raise ValueError("update-norm calibration requires positive observed and reference norms")
    return max(
        minimum,
        learning_rate * reference_norm / observed_norm,
    )


def calibrate_learning_rate_for_output_kl(
    target_kl: float,
    observed_kl: float,
    learning_rate: float,
    *,
    minimum: float = LOCAL_FORK_SPEC.calibration_minimum_learning_rate,
    maximum_scale: float = LOCAL_FORK_SPEC.calibration_maximum_scale,
) -> float:
    """Calibrate LR using the local quadratic KL approximation."""

    if target_kl <= 0 or observed_kl <= 0 or learning_rate <= 0:
        raise ValueError("output-KL calibration requires positive target, observation, and LR")
    if (
        minimum != LOCAL_FORK_SPEC.calibration_minimum_learning_rate
        or maximum_scale != LOCAL_FORK_SPEC.calibration_maximum_scale
    ):
        raise ValueError("local-fork calibration constants differ from LocalForkSpec")
    scale = (target_kl / observed_kl) ** 0.5
    scale = min(maximum_scale, max(1.0 / maximum_scale, scale))
    return max(minimum, learning_rate * scale)


def output_kl_match_status(
    target_kl: float,
    observed_kl: float,
    *,
    relative_tolerance: float,
) -> dict[str, float | bool]:
    if target_kl <= 0 or observed_kl < 0:
        raise ValueError("output-KL match status requires positive target and nonnegative observation")
    if not 0 < relative_tolerance < 1:
        raise ValueError("output-KL relative tolerance must be in (0, 1)")
    error = observed_kl - target_kl
    relative_error = error / target_kl
    return {
        "absolute_error": error,
        "relative_error": relative_error,
        "within_tolerance": abs(relative_error) <= relative_tolerance,
    }


def _teacher_entropy(record: TrajectoryRecord) -> float:
    if not record.teacher_entropy:
        return -1.0
    return sum(record.teacher_entropy) / len(record.teacher_entropy)


def match_state_source_forks(
    sources: dict[str, list[TrajectoryRecord]],
) -> dict[str, Any]:
    required = {
        "common_behavior",
        "initial_student",
        "current_fork_checkpoint",
        "teacher_policy",
    }
    if set(sources) != required:
        raise ValueError(f"state-source fork requires exactly {sorted(required)}")
    strata_by_source = {}
    for source, records in sources.items():
        strata: dict[tuple[Any, ...], list[TrajectoryRecord]] = defaultdict(list)
        for record in records:
            key = (
                record.prompt_id,
                len(record.response_ids) // 8,
                float(record.verifier_reward or 0.0),
                round(_teacher_entropy(record), 1),
            )
            strata[key].append(record)
        strata_by_source[source] = strata
    common_strata = set.intersection(*(set(strata) for strata in strata_by_source.values()))
    selected: dict[str, list[str]] = {source: [] for source in sources}
    strata_manifest = []
    for key in sorted(common_strata, key=repr):
        count = min(len(strata_by_source[source][key]) for source in sources)
        if count < 1:
            continue
        for source in sources:
            selected[source].extend(record.trajectory_id for record in strata_by_source[source][key][:count])
        strata_manifest.append({"stratum": list(key), "count": count})
    matched_count = len(next(iter(selected.values())))
    if matched_count < 1:
        raise ValueError("state-source forks have no common matched strata")
    return {
        "matching_fields": [
            "prompt_id",
            "response_length_bin_8",
            "verifier_reward",
            "teacher_entropy_0.1",
        ],
        "matched_count_per_source": matched_count,
        "selected_trajectory_ids": selected,
        "strata": strata_manifest,
        "sha256": sha256_value(selected),
    }
