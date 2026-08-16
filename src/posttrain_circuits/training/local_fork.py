"""Shared-state local-fork bundles, exact restoration, and matching."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.core.types import (
    LossOutput,
    PromptBatch,
    SupervisionBatch,
    Supervisor,
    TrajectoryBatch,
    TrajectoryRecord,
)
from posttrain_circuits.data.collators import collate_trajectories
from posttrain_circuits.supervision.hard_teacher import HardTeacherSupervisor
from posttrain_circuits.supervision.soft_teacher import SoftTeacherSupervisor
from posttrain_circuits.supervision.verified_replay import (
    VerifiedReplaySupervisor,
)
from posttrain_circuits.training.optimizer import parameter_update_norm


@dataclass(frozen=True)
class ForkBundleManifest:
    bundle_id: str
    checkpoint_hash: str
    optimizer_hash: str
    optimizer_moment_hash: str
    scheduler_hash: str
    rng_hash: str
    prompt_hash: str
    trajectory_hash: str
    probe_input_hash: str
    probe_output_hash: str
    manifest_hashes: dict[str, str]
    policy_version: int
    group_membership_hash: str
    minimum_group_size: int


@dataclass
class RestoredFork:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    prompts: PromptBatch
    trajectories: TrajectoryBatch
    probe_input_ids: torch.Tensor
    pre_update_outputs: torch.Tensor
    initial_hashes: dict[str, str]


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


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
    manifest_hashes: dict[str, str],
    model_spec: dict[str, Any] | None = None,
    minimum_group_size: int = 4,
) -> ForkBundleManifest:
    trajectories.validate()
    if minimum_group_size < 2:
        raise ValueError("policy-gradient fork minimum group size must be at least two")
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
        raise TypeError("fork bundles currently require torch.optim.AdamW")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise TypeError("fork bundles currently require LambdaLR schedulers")
    model_state = _cpu_tree(model.state_dict())
    optimizer_state = _cpu_tree(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_parameter_names = []
    for group in optimizer.param_groups:
        try:
            optimizer_parameter_names.append(
                [parameter_names[id(parameter)] for parameter in group["params"]]
            )
        except KeyError as error:
            raise ValueError("optimizer contains a parameter not owned by the bundled model") from error
    rng_state = RNGState.capture().as_dict()
    prompt_payload = asdict(prompts)
    trajectory_payload = [asdict(record) for record in trajectories.records]
    if probe_input_ids is None:
        probe_input_ids = torch.tensor([trajectories.records[0].input_ids])
    probe_inputs = probe_input_ids.detach().cpu()
    probe_outputs = pre_update_outputs.detach().cpu()
    if probe_inputs.ndim != 2 or probe_outputs.ndim != 3:
        raise ValueError("fork probe inputs/logits must have shapes [batch, seq] and [batch, seq, vocab]")
    if probe_inputs.shape[:2] != probe_outputs.shape[:2]:
        raise ValueError("fork probe inputs and logits do not align")
    manifest = ForkBundleManifest(
        bundle_id="",
        checkpoint_hash=state_hash(model_state),
        optimizer_hash=state_hash(optimizer_state),
        optimizer_moment_hash=state_hash(optimizer_state["state"]),
        scheduler_hash=state_hash(scheduler_state),
        rng_hash=state_hash(rng_state),
        prompt_hash=sha256_value(prompt_payload),
        trajectory_hash=sha256_value(trajectory_payload),
        probe_input_hash=state_hash(probe_inputs),
        probe_output_hash=state_hash(probe_outputs),
        manifest_hashes=manifest_hashes,
        policy_version=trajectories.policy_version,
        group_membership_hash=sha256_value(group_membership),
        minimum_group_size=minimum_group_size,
    )
    bundle_id = "fork-" + sha256_value(asdict(manifest))[:16]
    manifest = ForkBundleManifest(
        bundle_id=bundle_id,
        **{key: value for key, value in asdict(manifest).items() if key != "bundle_id"},
    )
    _atomic_torch_save(
        path,
        {
            "manifest": asdict(manifest),
            "model_spec": model_spec or {},
            "model": model_state,
            "optimizer": optimizer_state,
            "optimizer_parameter_names": optimizer_parameter_names,
            "optimizer_class": type(optimizer).__name__,
            "scheduler": scheduler_state,
            "scheduler_class": type(scheduler).__name__,
            "rng": rng_state,
            "prompts": prompt_payload,
            "trajectories": trajectory_payload,
            "probe_input_ids": probe_inputs,
            "pre_update_outputs": probe_outputs,
        },
    )
    return manifest


def load_fork_bundle(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _restore_optimizer(
    model: torch.nn.Module,
    state: dict[str, Any],
    parameter_names: list[list[str]],
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
    return optimizer


def restore_bundle_fresh(
    payload: dict[str, Any],
    *,
    model_factory: Callable[[], torch.nn.Module],
) -> RestoredFork:
    model = model_factory()
    model.load_state_dict(payload["model"])
    optimizer = _restore_optimizer(
        model,
        payload["optimizer"],
        payload["optimizer_parameter_names"],
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
        clip_epsilon: float = 0.2,
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


def _probe_kl(
    initial_logits: torch.Tensor,
    current_logits: torch.Tensor,
) -> float:
    fork_log = initial_logits.float().log_softmax(dim=-1)
    new_log = current_logits.float().log_softmax(dim=-1)
    # Primary behavioral displacement: KL(output_new || output_fork).
    return float((new_log.exp() * (new_log - fork_log)).sum(dim=-1).mean())


def run_branch(
    *,
    bundle_payload: dict[str, Any],
    model_factory: Callable[[], torch.nn.Module],
    branch: str,
    horizon: int,
    pad_token_id: int,
    checkpoint_root: Path,
    learning_rate_override: float | None = None,
) -> dict[str, Any]:
    if horizon < 1:
        raise ValueError("fork horizon must be positive")
    restored = restore_bundle_fresh(
        bundle_payload,
        model_factory=model_factory,
    )
    if learning_rate_override is not None:
        for group in restored.optimizer.param_groups:
            group["lr"] = learning_rate_override
        restored.scheduler.base_lrs = [learning_rate_override for _ in restored.scheduler.base_lrs]
    before = [parameter.detach().cpu().clone() for parameter in restored.model.parameters()]
    supervisors: dict[str, Supervisor] = {
        "hard_teacher": HardTeacherSupervisor(pad_token_id),
        "soft_teacher": SoftTeacherSupervisor(pad_token_id),
        "verified_replay": VerifiedReplaySupervisor(pad_token_id),
        "centered_policy_gradient": SharedTrajectoryCenteredPolicyGradientSupervisor(pad_token_id),
    }
    if branch not in supervisors:
        raise ValueError(f"unknown fork branch {branch!r}")
    supervisor = supervisors[branch]
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
    _atomic_torch_save(
        pre_checkpoint,
        {
            "bundle_id": bundle_payload["manifest"]["bundle_id"],
            "model": _cpu_tree(restored.model.state_dict()),
            "optimizer": _cpu_tree(restored.optimizer.state_dict()),
            "scheduler": restored.scheduler.state_dict(),
            "rng": RNGState.capture().as_dict(),
            "prompts": asdict(restored.prompts),
            "trajectories": [asdict(record) for record in restored.trajectories.records],
            "probe_input_ids": restored.probe_input_ids,
            "pre_update_outputs": restored.pre_update_outputs,
            "initial_hashes": restored.initial_hashes,
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
    was_training = restored.model.training
    restored.model.eval()
    with torch.no_grad():
        current_outputs = restored.model(input_ids=probe_ids).logits.cpu()
    restored.model.train(was_training)
    _atomic_torch_save(
        post_checkpoint,
        {
            "bundle_id": bundle_payload["manifest"]["bundle_id"],
            "model": _cpu_tree(restored.model.state_dict()),
            "optimizer": _cpu_tree(restored.optimizer.state_dict()),
            "scheduler": restored.scheduler.state_dict(),
            "rng": RNGState.capture().as_dict(),
            "prompts": asdict(restored.prompts),
            "trajectories": [asdict(record) for record in restored.trajectories.records],
            "probe_input_ids": restored.probe_input_ids,
            "pre_update_outputs": restored.pre_update_outputs,
            "initial_hashes": restored.initial_hashes,
        },
    )
    return {
        "branch": branch,
        "horizon": horizon,
        "initial_hashes": restored.initial_hashes,
        "initial_parameter_hash": restored.initial_hashes["model"],
        "initial_optimizer_moment_hash": restored.initial_hashes["optimizer_moments"],
        "initial_scheduler_hash": restored.initial_hashes["scheduler"],
        "initial_rng_hash": restored.initial_hashes["rng"],
        "initial_trajectory_hash": restored.initial_hashes["trajectories"],
        "probe_input_hash": restored.initial_hashes["probe_inputs"],
        "step_losses": losses,
        "step_metrics": step_metrics,
        "group_membership_hash": bundle_payload["manifest"]["group_membership_hash"],
        "loss": losses[-1],
        "parameter_update_norm": parameter_update_norm(
            before,
            restored.model.parameters(),
        ),
        "probe_output_kl_new_to_fork": _probe_kl(
            restored.pre_update_outputs,
            current_outputs,
        ),
        "learning_rate": float(restored.optimizer.param_groups[0]["lr"]),
        "pre_checkpoint": str(pre_checkpoint),
        "post_checkpoint": str(post_checkpoint),
        "post_model_hash": state_hash(restored.model.state_dict()),
        "post_optimizer_hash": state_hash(restored.optimizer.state_dict()),
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
    minimum: float = 1e-8,
    maximum_scale: float = 10.0,
) -> float:
    """Calibrate LR using the local quadratic KL approximation."""

    if target_kl <= 0 or observed_kl <= 0 or learning_rate <= 0:
        raise ValueError("output-KL calibration requires positive target, observation, and LR")
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
