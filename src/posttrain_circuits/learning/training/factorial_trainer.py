"""One trainer for the entire 2 x 3 controlled factorial grid."""

from __future__ import annotations

import os
import signal
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.artifacts.checkpoints import (
    accelerator_state_file_hashes,
    load_checkpoint,
    load_checkpoint_model_state,
    model_update_evidence,
    save_checkpoint,
    torch_state_hash,
    validate_accelerator_state_directory,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import publish_path_no_clobber, publish_torch_once
from posttrain_circuits.artifacts.runs import append_metric
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.learning.contracts import PromptBatch, StateSource, TrajectoryBatch
from posttrain_circuits.learning.primitives import SupervisionBatch
from posttrain_circuits.learning.supervision.base import Supervisor
from posttrain_circuits.learning.supervision.verified_replay import (
    InsufficientPositiveTrajectories,
    VerifiedReplaySupervisor,
)
from posttrain_circuits.learning.training.optimizer import parameter_update_norm
from posttrain_circuits.learning.training.schedules import PromptScheduler
from posttrain_circuits.learning.training.token_budget import TOKEN_BUDGET_UNIT, TokenBudgetState

_REQUIRED_EVALUATION_METRICS = (
    "validation_accuracy",
    "exact_proof_accuracy",
    "format_validity",
)
_COLLECTION_METRICS = (
    "generated_trajectories",
    "successful_trajectories",
    "effective_positive_sequences",
    "effective_supervised_tokens",
    "retry_count",
)


def _validate_rank_shard_hashes(
    observed: object,
    *,
    expected_rank_hash: str | None,
    rank: int,
    world_size: int,
) -> list[str]:
    if expected_rank_hash is None:
        if observed not in (None, []):
            raise ValueError("checkpoint unexpectedly contains rank-shard bindings")
        return []
    if not isinstance(observed, list) or len(observed) != world_size:
        raise ValueError("checkpoint rank-shard bindings differ from the distributed world")
    if not 0 <= rank < world_size:
        raise ValueError("distributed rank is outside the checkpoint world")
    for digest in observed:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("checkpoint rank-shard binding is not a SHA-256 digest")
    if len(set(observed)) != len(observed):
        raise ValueError("checkpoint rank-shard bindings are not unique")
    if observed[rank] != expected_rank_hash:
        raise ValueError(f"checkpoint prompt shard differs for distributed rank {rank}")
    return observed


def _extend_resume_ancestry(observed: object, checkpoint: Path) -> list[str]:
    if not isinstance(observed, list):
        raise ValueError("checkpoint resume ancestry must be a list")
    normalized: list[str] = []
    for value in observed:
        if (
            not isinstance(value, str)
            or not value.startswith("sha256:")
            or len(value) != 71
            or any(character not in "0123456789abcdef" for character in value[7:])
        ):
            raise ValueError("checkpoint resume ancestry is not content-addressed")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("checkpoint resume ancestry contains a cycle")
    current = f"sha256:{sha256_file(checkpoint)}"
    if current in normalized:
        raise ValueError("checkpoint resume ancestry contains a repeated checkpoint")
    return [*normalized, current]


def _checkpoint_publication_preflight_report(
    targets: Mapping[str, Path],
    *,
    expected_run_root: Path,
    rank: int,
) -> dict[str, object]:
    """Sample every no-clobber target without raising before a collective."""

    run_root = Path(expected_run_root).absolute()
    normalized_targets = {
        str(name): str(Path(path).absolute()) for name, path in sorted(targets.items())
    }
    report: dict[str, object] = {
        "rank": rank,
        "run_root": str(run_root),
        "targets": normalized_targets,
        "passed": False,
        "error": None,
    }
    try:
        if not normalized_targets or len(set(normalized_targets.values())) != len(
            normalized_targets
        ):
            raise ValueError("checkpoint publication targets are empty or overlap")
        if run_root.is_symlink() or not run_root.is_dir():
            raise ValueError("checkpoint run root is not a regular directory")
        if run_root.resolve(strict=True) != run_root:
            raise ValueError("checkpoint run root traverses a symlink")
        for name, raw_target in normalized_targets.items():
            target = Path(raw_target)
            try:
                relative_parent = target.parent.relative_to(run_root)
            except ValueError as error:
                raise ValueError(f"checkpoint target {name} is outside the run root") from error
            current = run_root
            for part in relative_parent.parts:
                current /= part
                if current.is_symlink() or not current.is_dir():
                    raise ValueError(
                        f"checkpoint target {name} has an unsafe parent: {current}"
                    )
            try:
                target_status = target.lstat()
            except FileNotFoundError:
                target_status = None
            if target_status is not None:
                raise FileExistsError(f"checkpoint target {name} already exists")
        report["passed"] = True
    except (OSError, ValueError) as error:
        report["error"] = str(error)
    return report


def _validate_checkpoint_publication_preflight_reports(
    rows: list[object],
    *,
    world_size: int,
) -> dict[str, Path]:
    if len(rows) != world_size or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("checkpoint publication preflight omitted a rank")
    normalized = [dict(row) for row in rows if isinstance(row, dict)]
    if [row.get("rank") for row in normalized] != list(range(world_size)):
        raise RuntimeError("checkpoint publication preflight rank order changed")
    target_maps = [row.get("targets") for row in normalized]
    roots = {row.get("run_root") for row in normalized}
    if (
        any(not isinstance(targets, dict) for targets in target_maps)
        or any(targets != target_maps[0] for targets in target_maps[1:])
        or len(roots) != 1
    ):
        raise RuntimeError("distributed ranks disagree on checkpoint publication targets")
    failures = [
        {"rank": row.get("rank"), "error": row.get("error")}
        for row in normalized
        if row.get("passed") is not True
    ]
    if failures:
        raise RuntimeError(f"checkpoint publication preflight failed: {failures}")
    first = target_maps[0]
    assert isinstance(first, dict)
    return {str(name): Path(str(path)) for name, path in first.items()}


def _validate_rank_state_mappings(
    observed: object,
    *,
    name: str,
    world_size: int,
) -> list[dict[str, Any]]:
    if not isinstance(observed, list) or len(observed) != world_size:
        raise ValueError(f"checkpoint {name} must contain exactly one mapping per rank")
    if any(not isinstance(item, Mapping) for item in observed):
        raise ValueError(f"checkpoint {name} contains a non-mapping rank state")
    normalized = [dict(item) for item in observed]
    for index, state in enumerate(normalized):
        if type(state.get("rank")) is not int or state.get("rank") != index:
            raise ValueError(f"checkpoint {name} rank metadata differs at index {index}")
        if type(state.get("world_size")) is not int or state.get("world_size") != world_size:
            raise ValueError(f"checkpoint {name} world_size metadata differs at rank {index}")
    return normalized


def _validate_resume_world_size(observed: object, *, running_world_size: int) -> int:
    if type(running_world_size) is not int or running_world_size < 1:
        raise ValueError("running distributed world_size is invalid")
    if type(observed) is not int or observed != running_world_size:
        raise ValueError("resume checkpoint world_size differs from the running world")
    return observed


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int = 2
    token_budget: int = 1024
    token_budget_unit: str = TOKEN_BUDGET_UNIT
    steps_per_round: int = 1
    learning_rate: float = 5e-4
    checkpoint_every: int = 1
    evaluation_every: int = 1
    backend: str = "torch_smoke"
    gradient_accumulation_steps: int = 1
    max_completion_length: int = 128
    require_evaluation_metrics: bool = False

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.token_budget_unit != TOKEN_BUDGET_UNIT:
            raise ValueError(f"unsupported token_budget_unit {self.token_budget_unit!r}")
        if self.steps_per_round < 1:
            raise ValueError("steps_per_round must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.evaluation_every < 1:
            raise ValueError("evaluation_every must be positive")


class FactorialTrainer:
    """Shared optimizer/update/checkpoint loop; only source and supervisor vary."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        prompt_scheduler: PromptScheduler,
        state_source: StateSource,
        supervisor: Supervisor,
        config: TrainerConfig,
        run_dir: Path,
        teacher: Any = None,
        verifier: Any = None,
        resolved_config: dict[str, Any] | None = None,
        manifest_hashes: dict[str, str] | None = None,
        git_commit: str = "unavailable",
        implementation_dirty: bool = True,
        dependency_versions: dict[str, str] | None = None,
        resume_ancestry: list[str] | None = None,
        rank_shard_hash: str | None = None,
        probe_input_ids: torch.Tensor | None = None,
        evaluation_fn: Callable[[torch.nn.Module], dict[str, float]] | None = None,
    ) -> None:
        if config.backend not in {"torch_smoke", "accelerate"}:
            raise ValueError("backend must be 'torch_smoke' or 'accelerate'")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prompt_scheduler = prompt_scheduler
        self.state_source = state_source
        self.supervisor = supervisor
        self.config = config
        self.run_dir = run_dir
        self.teacher = teacher
        self.verifier = verifier
        self.resolved_config = resolved_config or {}
        self.manifest_hashes = manifest_hashes or {}
        raw_baseline_path = str(
            self.resolved_config.get("production_safety", {}).get(
                "initial_checkpoint_path",
                "",
            )
        ).strip()
        self._update_norm_baseline_model: dict[str, Any] | None = None
        if raw_baseline_path:
            self._update_norm_baseline_path = Path(raw_baseline_path).absolute()
            if not self._update_norm_baseline_path.is_file():
                raise ValueError("Factorial update-norm baseline checkpoint is missing")
            observed_baseline_hash = sha256_file(self._update_norm_baseline_path)
            expected_baseline_hash = self.manifest_hashes.get("initial_checkpoint")
            if (
                expected_baseline_hash is not None
                and observed_baseline_hash != expected_baseline_hash
            ):
                raise ValueError("Factorial update-norm baseline checkpoint bytes changed")
            self._update_norm_baseline_sha256 = observed_baseline_hash
        else:
            self._update_norm_baseline_model = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            self._update_norm_baseline_path = (
                Path(run_dir).absolute() / "checkpoints" / ".in-memory-update-baseline.pt"
            )
            self._update_norm_baseline_sha256 = self.manifest_hashes.get(
                "initial_checkpoint",
                torch_state_hash(self._update_norm_baseline_model),
            )
        self.git_commit = git_commit
        self.implementation_dirty = implementation_dirty
        self.dependency_versions = dependency_versions or {}
        self.resume_ancestry = list(resume_ancestry or [])
        self.rank_shard_hash = rank_shard_hash
        self.probe_input_ids = probe_input_ids
        self.evaluation_fn = evaluation_fn
        if config.require_evaluation_metrics and evaluation_fn is None:
            raise ValueError("formal training requires an evaluation callback before any update")
        self.global_step = 0
        self.token_budget = TokenBudgetState(config.token_budget, config.token_budget_unit)
        self.online_rollout_round = 0
        self.cumulative_counts: dict[str, float] = {
            "prompts_consumed": 0.0,
            "trajectories_generated": 0.0,
            "response_tokens_generated": 0.0,
            "supervised_response_tokens": 0.0,
            "model_facing_input_tokens_processed": 0.0,
            "forward_backward_flop_estimate": 0.0,
        }
        self._pending_collection: dict[str, float] = {key: 0.0 for key in _COLLECTION_METRICS}
        self._pending_processed_tokens = 0.0
        self._pending_input_tokens = 0
        self._current_global_update_tokens = 0
        self._pending_metric_sums: dict[str, float] = {}
        self._pending_metric_calls = 0
        self._accumulation_micro_step = 0
        self._parameters_before_update: list[torch.Tensor] | None = None
        self._terminate = False
        self._accelerator: Any = None
        self._last_checkpoint_step: int | None = None
        self._last_checkpoint_path: Path | None = None
        self._last_checkpoint_sha256: str | None = None
        self._last_accelerator_state_files: dict[str, str] | None = None
        self._last_accelerator_state_sha256: str | None = None
        if config.backend == "accelerate":
            try:
                from accelerate import Accelerator
            except ImportError as error:
                raise RuntimeError("Accelerate backend requested; install the 'train' extra") from error
            self._accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
            self.model, self.optimizer, self.scheduler = self._accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
            )
        self.is_main_process = self._accelerator is None or bool(self._accelerator.is_main_process)
        self.initial_probe_log_probs: torch.Tensor | None = None
        if self.probe_input_ids is not None:
            device = next(self.model.parameters()).device
            self.probe_input_ids = self.probe_input_ids.to(device)
            with torch.no_grad():
                self.initial_probe_log_probs = (
                    self.model(input_ids=self.probe_input_ids).logits.log_softmax(dim=-1).detach()
                )
        if self.is_main_process:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "checkpoints").mkdir(exist_ok=True)
            (self.run_dir / "metrics.jsonl").touch(exist_ok=True)
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
        with suppress(ValueError):
            signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        del signum, frame
        self._terminate = True

    def _distributed_position(self) -> tuple[int, int]:
        if self._accelerator is not None:
            return (
                int(getattr(self._accelerator, "process_index", 0)),
                int(getattr(self._accelerator, "num_processes", 1)),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _gather_rank_shard_hashes(self) -> list[str]:
        rank, world_size = self._distributed_position()
        gathered: list[str | None] = [self.rank_shard_hash]
        if world_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("distributed rank-shard binding requires a process group")
            gathered = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered, self.rank_shard_hash)
        if all(value is None for value in gathered):
            return []
        if any(value is None for value in gathered):
            raise ValueError("only some distributed ranks supplied prompt-shard bindings")
        normalized = [str(value) for value in gathered]
        _validate_rank_shard_hashes(
            normalized,
            expected_rank_hash=self.rank_shard_hash,
            rank=rank,
            world_size=world_size,
        )
        return normalized

    def _collective_checkpoint_preflight(
        self,
        targets: Mapping[str, Path],
    ) -> dict[str, Path]:
        rank, world_size = self._distributed_position()
        local_report = _checkpoint_publication_preflight_report(
            targets,
            expected_run_root=self.run_dir,
            rank=rank,
        )
        rows: list[object] = [local_report]
        if world_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("distributed checkpoint preflight requires a process group")
            rows = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(
                rows,
                local_report,
            )
        validated = _validate_checkpoint_publication_preflight_reports(
            rows,
            world_size=world_size,
        )
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
        elif world_size > 1:
            torch.distributed.barrier()
        return validated

    def _gather_rank_runtime_states(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rank, world_size = self._distributed_position()

        def bind_rank_state(raw: Mapping[str, Any], *, name: str) -> dict[str, Any]:
            state = dict(raw)
            if "rank" in state and (
                type(state["rank"]) is not int or state["rank"] != rank
            ):
                raise ValueError(f"{name} reported the wrong distributed rank")
            if "world_size" in state and (
                type(state["world_size"]) is not int
                or state["world_size"] != world_size
            ):
                raise ValueError(f"{name} reported the wrong distributed world_size")
            state["rank"] = rank
            state["world_size"] = world_size
            return state

        prompt_state = bind_rank_state(
            self.prompt_scheduler.state_dict(),
            name="prompt scheduler",
        )
        source_state = bind_rank_state(
            self.state_source.state_dict(),
            name="state source",
        )
        prompt_states: list[dict[str, Any]] = [prompt_state]
        source_states: list[dict[str, Any]] = [source_state]
        if world_size > 1:
            prompt_states = [{} for _ in range(world_size)]
            source_states = [{} for _ in range(world_size)]
            torch.distributed.all_gather_object(prompt_states, prompt_state)
            torch.distributed.all_gather_object(source_states, source_state)
        return (
            _validate_rank_state_mappings(
                prompt_states,
                name="prompt_scheduler_by_rank",
                world_size=world_size,
            ),
            _validate_rank_state_mappings(
                source_states,
                name="state_source_by_rank",
                world_size=world_size,
            ),
        )

    def _collective_staging_directory(self, target: Path) -> Path:
        rank, world_size = self._distributed_position()
        holder: list[object] = [None]
        if rank == 0:
            holder[0] = tempfile.mkdtemp(
                prefix=f".{target.name}.stage-",
                dir=target.parent,
            )
        if world_size > 1:
            torch.distributed.broadcast_object_list(holder, src=0)
        if not isinstance(holder[0], str):
            raise RuntimeError("checkpoint staging directory was not shared by rank zero")
        staging = Path(holder[0]).absolute()
        if staging.parent != target.parent or staging.is_symlink() or not staging.is_dir():
            raise RuntimeError("checkpoint staging directory is outside its target parent")
        return staging

    def _raise_collective_publication_error(self, error: str | None) -> None:
        rank, world_size = self._distributed_position()
        holder: list[object] = [error if rank == 0 else None]
        if world_size > 1:
            torch.distributed.broadcast_object_list(holder, src=0)
        if holder[0] is not None:
            raise RuntimeError(f"checkpoint no-clobber publication failed: {holder[0]}")

    def _model_update_evidence(self, final_state: Mapping[str, Any]) -> tuple[float, str]:
        if self._update_norm_baseline_model is not None:
            baseline = self._update_norm_baseline_model
        else:
            if sha256_file(self._update_norm_baseline_path) != self._update_norm_baseline_sha256:
                raise ValueError("Factorial update-norm baseline checkpoint bytes changed")
            baseline = load_checkpoint_model_state(self._update_norm_baseline_path)
        norm, final_hash = model_update_evidence(baseline, final_state)
        if norm <= 0.0:
            raise RuntimeError("Factorial checkpoint has no parameter change from its bound baseline")
        return norm, final_hash

    def _reuse_current_checkpoint_if_valid(self, path: Path) -> str | None:
        if self._last_checkpoint_step != self.global_step:
            return None
        expected_path = Path(path).absolute()
        if self._last_checkpoint_path != expected_path:
            raise RuntimeError("one optimizer step cannot publish multiple checkpoint paths")
        if expected_path.is_symlink() or not expected_path.is_file():
            raise RuntimeError("previously published checkpoint is missing or no longer regular")
        observed_sha256 = sha256_file(expected_path)
        if observed_sha256 != self._last_checkpoint_sha256:
            raise RuntimeError("previously published checkpoint bytes changed")
        if self._accelerator is not None:
            if (
                self._last_accelerator_state_files is None
                or self._last_accelerator_state_sha256 is None
            ):
                raise RuntimeError("published Accelerate checkpoint lost its state binding")
            validate_accelerator_state_directory(
                expected_path.with_suffix(".accelerate"),
                expected_run_root=self.run_dir,
                expected_files=self._last_accelerator_state_files,
                expected_sha256=self._last_accelerator_state_sha256,
            )
        return observed_sha256

    def _collect_trajectories(
        self,
        prompts: PromptBatch,
    ) -> tuple[TrajectoryBatch, SupervisionBatch, int]:
        replay = self.supervisor if isinstance(self.supervisor, VerifiedReplaySupervisor) else None
        retry_limit = replay.retry_limit if replay is not None else 0
        minimum_positives = replay.minimum_positives if replay is not None else 0
        records: list[TrajectoryRecord] = []
        policy_version = 0
        attempts = 0
        positives = 0
        for attempt in range(retry_limit + 1):
            batch = self.state_source.get_batch(
                self.model,
                prompts,
                self.global_step,
            )
            attempts = attempt + 1
            records.extend(batch.records)
            policy_version = batch.policy_version
            positives = sum(record.verifier_reward == 1.0 for record in records)
            if replay is None or positives >= minimum_positives:
                break
        else:
            raise InsufficientPositiveTrajectories(
                required=minimum_positives,
                received=positives,
                generated=len(records),
                retry_limit=retry_limit,
            )

        trajectories = TrajectoryBatch(records, policy_version)
        supervision = self.supervisor.prepare_targets(
            trajectories,
            self.teacher,
            self.verifier,
        )
        effective_tokens = sum(
            sum(record.response_token_mask)
            for record in records
            if replay is None or record.verifier_reward == 1.0
        )
        supervision.metadata.update(
            generated_trajectories=len(records),
            successful_trajectories=positives,
            effective_positive_sequences=positives,
            effective_supervised_tokens=effective_tokens,
            reward_rate=positives / len(records) if records else 0.0,
            retry_count=attempts - 1,
        )
        return trajectories, supervision, attempts

    def _register_collection(
        self,
        prompts: PromptBatch,
        trajectories: TrajectoryBatch,
        supervision: SupervisionBatch,
        attempts: int,
    ) -> None:
        generated = float(len(trajectories.records))
        response_tokens = float(sum(len(record.response_ids) for record in trajectories.records))
        self.cumulative_counts["prompts_consumed"] += len(prompts.prompt_ids) * attempts
        self.cumulative_counts["trajectories_generated"] += generated
        self.cumulative_counts["response_tokens_generated"] += response_tokens
        for key in _COLLECTION_METRICS:
            self._pending_collection[key] += float(supervision.metadata.get(key, 0.0))

    @staticmethod
    def _gradients_are_finite(model: torch.nn.Module) -> bool:
        return all(
            parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        )

    def _record_objective_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._pending_metric_sums[key] = self._pending_metric_sums.get(key, 0.0) + float(value)
        self._pending_metric_calls += 1

    def _training_micro_step(
        self,
        supervision: SupervisionBatch,
    ) -> bool:
        if self._parameters_before_update is None:
            self._parameters_before_update = [
                parameter.detach().cpu().clone() for parameter in self.model.parameters()
            ]
        output = self.supervisor.compute_loss(self.model, supervision)
        if not torch.isfinite(output.loss):
            raise FloatingPointError(f"non-finite loss before optimizer step {self.global_step + 1}")
        self._record_objective_metrics(output.metrics)
        self._accumulation_micro_step += 1
        expected_sync = self._accumulation_micro_step == self.config.gradient_accumulation_steps

        if self._accelerator is None:
            if self._accumulation_micro_step == 1:
                self.optimizer.zero_grad(set_to_none=True)
            (output.loss / self.config.gradient_accumulation_steps).backward()
            sync_gradients = expected_sync
            if sync_gradients:
                if not self._gradients_are_finite(self.model):
                    raise FloatingPointError(
                        f"non-finite gradient before optimizer step {self.global_step + 1}"
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            with self._accelerator.accumulate(self.model):
                self._accelerator.backward(output.loss)
                sync_gradients = bool(self._accelerator.sync_gradients)
                if sync_gradients:
                    self._accelerator.unscale_gradients(self.optimizer)
                    if not self._gradients_are_finite(self.model):
                        raise FloatingPointError(
                            f"non-finite gradient before optimizer step {self.global_step + 1}"
                        )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            if sync_gradients != expected_sync:
                raise RuntimeError("Accelerate accumulation boundary differs from trainer state")

        base_supervised_tokens = float(
            supervision.metadata.get(
                "effective_supervised_tokens",
                supervision.response_mask.sum().item(),
            )
        )
        self._pending_processed_tokens += base_supervised_tokens
        self.cumulative_counts["supervised_response_tokens"] += base_supervised_tokens
        self._pending_input_tokens += int(supervision.attention_mask.sum().item())
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        flop_estimate = 6.0 * parameter_count * base_supervised_tokens
        self.cumulative_counts["forward_backward_flop_estimate"] += flop_estimate

        if sync_gradients:
            self._accumulation_micro_step = 0
        return sync_gradients

    def _evaluation_metrics(self) -> dict[str, float | None]:
        metrics: dict[str, float | None] = dict.fromkeys(_REQUIRED_EVALUATION_METRICS)
        due = self.global_step % self.config.evaluation_every == 0 or (
            self.global_step == self.config.max_steps
        )
        if not due:
            return metrics
        if self.evaluation_fn is not None:
            metrics.update(self.evaluation_fn(self.model))
        if self.config.require_evaluation_metrics:
            missing = [name for name in _REQUIRED_EVALUATION_METRICS if metrics.get(name) is None]
            if missing:
                raise RuntimeError(
                    "formal training requires non-null evaluation metrics: " + ", ".join(missing)
                )
        return metrics

    def _output_kl(self) -> float | None:
        if self.initial_probe_log_probs is None or self.probe_input_ids is None:
            return None
        with torch.no_grad():
            current_log_probs = self.model(input_ids=self.probe_input_ids).logits.log_softmax(dim=-1)
            current_probs = current_log_probs.exp()
            return float(
                (current_probs * (current_log_probs - self.initial_probe_log_probs))
                .sum(dim=-1)
                .mean()
                .detach()
            )

    def _finish_optimizer_update(
        self,
        *,
        started: float,
    ) -> dict[str, Any]:
        self.global_step += 1
        self.online_rollout_round = int(getattr(self.state_source, "policy_version", 0))
        objective_metrics = {
            key: value / self._pending_metric_calls for key, value in self._pending_metric_sums.items()
        }
        generated = self._pending_collection["generated_trajectories"]
        successful = self._pending_collection["successful_trajectories"]
        collection_metrics = {
            **self._pending_collection,
            "reward_rate": successful / generated if generated else 0.0,
        }
        if self._parameters_before_update is None:
            raise RuntimeError("optimizer update has no parameter snapshot")
        self.cumulative_counts["model_facing_input_tokens_processed"] = float(self.token_budget.consumed)
        metric: dict[str, Any] = {
            **objective_metrics,
            **collection_metrics,
            "step": float(self.global_step),
            **self.cumulative_counts,
            "effective_supervised_tokens_this_update": (self._pending_processed_tokens),
            "local_model_facing_input_tokens_this_update": self._pending_input_tokens,
            "model_facing_input_tokens_this_update": self._current_global_update_tokens,
            "token_budget": self.token_budget.budget,
            "token_budget_unit": self.token_budget.unit,
            "token_budget_consumed": self.token_budget.consumed,
            "token_budget_remaining": self.token_budget.remaining,
            "optimizer_updates": float(self.global_step),
            "parameter_update_norm": parameter_update_norm(
                self._parameters_before_update,
                self.model.parameters(),
            ),
            "output_kl_from_initial": self._output_kl(),
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_allocated_gpu_memory": float(
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            **self._evaluation_metrics(),
        }
        self._pending_collection = {key: 0.0 for key in _COLLECTION_METRICS}
        self._pending_processed_tokens = 0.0
        self._pending_input_tokens = 0
        self._current_global_update_tokens = 0
        self._pending_metric_sums = {}
        self._pending_metric_calls = 0
        self._parameters_before_update = None
        return metric

    def train(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        started = time.perf_counter()
        queued_microbatches: list[SupervisionBatch] = []
        while self.global_step < self.config.max_steps and not self._terminate:
            if self.token_budget.remaining == 0:
                self.token_budget.stop_reason = "token_budget_exactly_consumed"
                break
            while len(queued_microbatches) < self.config.gradient_accumulation_steps:
                prompts = self.prompt_scheduler.next_batch()
                trajectories, supervision, attempts = self._collect_trajectories(prompts)
                self._register_collection(prompts, trajectories, supervision, attempts)
                queued_microbatches.extend([supervision] * self.config.steps_per_round)
            window = queued_microbatches[: self.config.gradient_accumulation_steps]
            del queued_microbatches[: self.config.gradient_accumulation_steps]
            local_window_tokens = sum(int(batch.attention_mask.sum().item()) for batch in window)
            admitted, global_window_tokens = self.token_budget.reserve_optimizer_update(local_window_tokens)
            if not admitted:
                break
            self._current_global_update_tokens = global_window_tokens
            for supervision in window:
                sync_gradients = self._training_micro_step(supervision)
                if not sync_gradients:
                    continue
                metric = self._finish_optimizer_update(started=started)
                history.append(metric)
                if self.is_main_process:
                    append_metric(self.run_dir / "metrics.jsonl", metric)
                if self.global_step % self.config.checkpoint_every == 0 or self._terminate:
                    self.save(self.run_dir / "checkpoints" / f"step-{self.global_step:08d}.pt")
                break
        if self.token_budget.stop_reason is None:
            self.token_budget.stop_reason = (
                "signal_at_optimizer_boundary" if self._terminate else "max_steps_safety_limit"
            )
        if self.is_main_process and self.config.require_evaluation_metrics:
            append_metric(
                self.run_dir / "metrics.jsonl",
                {
                    "event": "training_stop",
                    "step": float(self.global_step),
                    "token_budget": self.token_budget.budget,
                    "token_budget_unit": self.token_budget.unit,
                    "token_budget_consumed": self.token_budget.consumed,
                    "token_budget_remaining": self.token_budget.remaining,
                    "stop_reason": self.token_budget.stop_reason,
                },
            )
        if self.global_step > 0:
            self.save(self.run_dir / "checkpoints" / f"step-{self.global_step:08d}.pt")
        return history

    def save(self, path: Path) -> str:
        if self._accumulation_micro_step != 0:
            raise RuntimeError("checkpoint requested inside a partial gradient-accumulation window")
        path = Path(path).absolute()
        reused = self._reuse_current_checkpoint_if_valid(path)
        if reused is not None:
            return reused
        if self._accelerator is not None:
            state_dir = path.with_suffix(".accelerate")
            validated_targets = self._collective_checkpoint_preflight(
                {"metadata": path, "accelerator_state": state_dir}
            )
            if validated_targets != {
                "accelerator_state": state_dir,
                "metadata": path,
            }:
                raise RuntimeError("checkpoint publication preflight changed its targets")
            staging_state_dir = self._collective_staging_directory(state_dir)
            self._accelerator.save_state(str(staging_state_dir))
            self._accelerator.wait_for_everyone()
            accelerator_state_files = accelerator_state_file_hashes(
                staging_state_dir,
                expected_run_root=self.run_dir,
            )
            accelerator_state_sha256 = sha256_value(accelerator_state_files)
            rank_shard_hashes = self._gather_rank_shard_hashes()
            full_model_state = self._accelerator.get_state_dict(self.model)
            optimizer_state_key_type = "parameter_id"
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                if isinstance(self.model, FSDP):
                    optimizer_state = FSDP.full_optim_state_dict(
                        self.model,
                        self.optimizer,
                        rank0_only=True,
                    )
                    optimizer_state_key_type = "parameter_name"
                else:
                    optimizer_state = self.optimizer.state_dict()
            except ImportError:
                optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
            prompt_scheduler_states, state_source_states = self._gather_rank_runtime_states()
            _, world_size = self._distributed_position()
            scaler = getattr(self._accelerator, "scaler", None)
            self._accelerator.wait_for_everyone()
            publication_error = None
            if self.is_main_process:
                try:
                    parameter_norm, final_model_hash = self._model_update_evidence(
                        full_model_state
                    )
                    token_budget_state = self.token_budget.state_dict()
                    payload = {
                        "format": "accelerate_fsdp_full_export_v1",
                        "model": full_model_state,
                        "optimizer": optimizer_state,
                        "optimizer_state_key_type": optimizer_state_key_type,
                        "scheduler": scheduler_state,
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "accelerate_state_dir": str(state_dir.absolute()),
                        "accelerate_state_files": accelerator_state_files,
                        "accelerate_state_sha256": accelerator_state_sha256,
                        "rng": RNGState.capture().as_dict(),
                        "prompt_scheduler": prompt_scheduler_states[0],
                        "prompt_scheduler_by_rank": prompt_scheduler_states,
                        "state_source": state_source_states[0],
                        "state_source_by_rank": state_source_states,
                        "trainer_state": {
                            "cumulative_counts": self.cumulative_counts,
                            "accumulation_micro_step": self._accumulation_micro_step,
                            "token_budget": token_budget_state,
                        },
                        "token_budget": token_budget_state,
                        "global_step": self.global_step,
                        "world_size": world_size,
                        "policy_version": int(getattr(self.state_source, "policy_version", 0)),
                        "online_rollout_round": self.online_rollout_round,
                        "resolved_config": self.resolved_config,
                        "manifest_hashes": self.manifest_hashes,
                        "git_commit": self.git_commit,
                        "implementation_dirty": self.implementation_dirty,
                        "dependency_versions": self.dependency_versions,
                        "resume_ancestry": self.resume_ancestry,
                        "rank_shard_hashes": rank_shard_hashes,
                        "parameter_update_norm": parameter_norm,
                        "final_model_state_hash": final_model_hash,
                        "update_norm_baseline_checkpoint_path": str(
                            self._update_norm_baseline_path
                        ),
                        "update_norm_baseline_checkpoint_sha256": (
                            self._update_norm_baseline_sha256
                        ),
                    }
                    publish_path_no_clobber(staging_state_dir, state_dir)
                    publish_torch_once(path, payload)
                except BaseException as error:  # share failure before any rank exits
                    publication_error = f"{type(error).__name__}: {error}"
            self._raise_collective_publication_error(publication_error)
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("checkpoint metadata publication did not complete")
            validate_accelerator_state_directory(
                state_dir,
                expected_run_root=self.run_dir,
                expected_files=accelerator_state_files,
                expected_sha256=accelerator_state_sha256,
            )
            checkpoint_sha256 = sha256_file(path)
            self._last_checkpoint_step = self.global_step
            self._last_checkpoint_path = path
            self._last_checkpoint_sha256 = checkpoint_sha256
            self._last_accelerator_state_files = accelerator_state_files
            self._last_accelerator_state_sha256 = accelerator_state_sha256
            return checkpoint_sha256
        validated_targets = self._collective_checkpoint_preflight({"metadata": path})
        if validated_targets != {"metadata": path}:
            raise RuntimeError("checkpoint publication preflight changed its target")
        rank_shard_hashes = self._gather_rank_shard_hashes()
        prompt_scheduler_states, state_source_states = self._gather_rank_runtime_states()
        rank, world_size = self._distributed_position()
        publication_error = None
        if rank == 0:
            descriptor, staging_name = tempfile.mkstemp(
                prefix=f".{path.name}.stage-",
                dir=path.parent,
            )
            os.close(descriptor)
            staging_path = Path(staging_name)
            try:
                model_state = self.model.state_dict()
                parameter_norm, final_model_hash = self._model_update_evidence(model_state)
                save_checkpoint(
                    staging_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    prompt_scheduler_state=prompt_scheduler_states[0],
                    prompt_scheduler_states=prompt_scheduler_states,
                    state_source_state=state_source_states[0],
                    state_source_states=state_source_states,
                    trainer_state={
                        "cumulative_counts": self.cumulative_counts,
                        "accumulation_micro_step": self._accumulation_micro_step,
                        "token_budget": self.token_budget.state_dict(),
                    },
                    global_step=self.global_step,
                    world_size=world_size,
                    policy_version=int(getattr(self.state_source, "policy_version", 0)),
                    online_rollout_round=self.online_rollout_round,
                    resolved_config=self.resolved_config,
                    manifest_hashes=self.manifest_hashes,
                    accelerator_state=None,
                    scaler_state=None,
                    git_commit=self.git_commit,
                    implementation_dirty=self.implementation_dirty,
                    dependency_versions=self.dependency_versions,
                    resume_ancestry=self.resume_ancestry,
                    rank_shard_hashes=rank_shard_hashes,
                    parameter_update_norm=parameter_norm,
                    final_model_state_hash=final_model_hash,
                    update_norm_baseline_checkpoint_path=str(
                        self._update_norm_baseline_path
                    ),
                    update_norm_baseline_checkpoint_sha256=(
                        self._update_norm_baseline_sha256
                    ),
                )
                publish_path_no_clobber(staging_path, path)
            except BaseException as error:  # share failure before any rank exits
                publication_error = f"{type(error).__name__}: {error}"
                with suppress(FileNotFoundError):
                    staging_path.unlink()
        self._raise_collective_publication_error(publication_error)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("checkpoint metadata publication did not complete")
        checkpoint_sha256 = sha256_file(path)
        self._last_checkpoint_step = self.global_step
        self._last_checkpoint_path = path
        self._last_checkpoint_sha256 = checkpoint_sha256
        self._last_accelerator_state_files = None
        self._last_accelerator_state_sha256 = None
        return checkpoint_sha256

    def resume(self, path: Path) -> None:
        path = Path(path).absolute()
        if self._accelerator is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format") != "accelerate_fsdp_full_export_v1":
                raise ValueError("Accelerate resume requires an Accelerate/FSDP checkpoint export")
            if payload.get("manifest_hashes") != self.manifest_hashes:
                raise ValueError("resume checkpoint ExperimentBinding/design hashes changed")
            rank = int(getattr(self._accelerator, "process_index", 0))
            world_size = int(getattr(self._accelerator, "num_processes", 1))
            _validate_resume_world_size(
                payload.get("world_size"),
                running_world_size=world_size,
            )
            _validate_rank_shard_hashes(
                payload.get("rank_shard_hashes"),
                expected_rank_hash=self.rank_shard_hash,
                rank=rank,
                world_size=world_size,
            )
            state_dir = Path(str(payload.get("accelerate_state_dir", "")))
            state_files = payload.get("accelerate_state_files")
            if not isinstance(state_files, dict):
                raise ValueError("Accelerate resume checkpoint lacks state file hashes")
            validate_accelerator_state_directory(
                state_dir,
                expected_run_root=path.resolve().parent.parent,
                expected_files=state_files,
                expected_sha256=str(payload.get("accelerate_state_sha256", "")),
            )
            prompt_states = _validate_rank_state_mappings(
                payload.get("prompt_scheduler_by_rank"),
                name="prompt_scheduler_by_rank",
                world_size=world_size,
            )
            source_states = _validate_rank_state_mappings(
                payload.get("state_source_by_rank"),
                name="state_source_by_rank",
                world_size=world_size,
            )
            if (
                payload.get("prompt_scheduler") != prompt_states[0]
                or payload.get("state_source") != source_states[0]
            ):
                raise ValueError("Accelerate resume checkpoint rank-0 runtime state is inconsistent")
            self._accelerator.load_state(str(state_dir))
            self._accelerator.wait_for_everyone()
            self.prompt_scheduler.load_state_dict(prompt_states[rank])
            self.state_source.load_state_dict(source_states[rank])
            self.global_step = int(payload["global_step"])
            self.online_rollout_round = int(payload["online_rollout_round"])
            self.cumulative_counts.update(payload.get("trainer_state", {}).get("cumulative_counts", {}))
            self._accumulation_micro_step = int(
                payload.get("trainer_state", {}).get("accumulation_micro_step", 0)
            )
            self.token_budget.load_state_dict(payload["trainer_state"]["token_budget"])
            if self._accumulation_micro_step != 0:
                raise ValueError("checkpoint contains an unsupported partial accumulation window")
            self.resume_ancestry = _extend_resume_ancestry(
                payload.get("resume_ancestry", []),
                path,
            )
            self._update_norm_baseline_path = path
            self._update_norm_baseline_sha256 = sha256_file(path)
            self._update_norm_baseline_model = None
            return
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_manifest_hashes=self.manifest_hashes,
        )
        rank, world_size = self._distributed_position()
        _validate_resume_world_size(
            payload.get("world_size"),
            running_world_size=world_size,
        )
        _validate_rank_shard_hashes(
            payload.get("rank_shard_hashes"),
            expected_rank_hash=self.rank_shard_hash,
            rank=rank,
            world_size=world_size,
        )
        prompt_states = _validate_rank_state_mappings(
            payload.get("prompt_scheduler_by_rank"),
            name="prompt_scheduler_by_rank",
            world_size=world_size,
        )
        source_states = _validate_rank_state_mappings(
            payload.get("state_source_by_rank"),
            name="state_source_by_rank",
            world_size=world_size,
        )
        if (
            payload.get("prompt_scheduler") != prompt_states[0]
            or payload.get("state_source") != source_states[0]
        ):
            raise ValueError("resume checkpoint rank-0 runtime state is inconsistent")
        self.prompt_scheduler.load_state_dict(prompt_states[rank])
        self.state_source.load_state_dict(source_states[rank])
        self.global_step = int(payload["global_step"])
        self.online_rollout_round = int(payload["online_rollout_round"])
        trainer_state = payload.get("trainer_state", {})
        self.cumulative_counts.update(trainer_state.get("cumulative_counts", {}))
        self._accumulation_micro_step = int(trainer_state.get("accumulation_micro_step", 0))
        self.token_budget.load_state_dict(trainer_state["token_budget"])
        if self._accumulation_micro_step != 0:
            raise ValueError("checkpoint contains an unsupported partial accumulation window")
        scaler = getattr(self._accelerator, "scaler", None)
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        self.resume_ancestry = _extend_resume_ancestry(
            payload.get("resume_ancestry", []),
            path,
        )
        self._update_norm_baseline_path = path
        self._update_norm_baseline_sha256 = sha256_file(path)
        self._update_norm_baseline_model = None
