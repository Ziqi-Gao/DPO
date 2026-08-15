"""One trainer for the entire 2 x 3 controlled factorial grid."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.core.provenance import append_metric
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.core.types import (
    PromptBatch,
    StateSource,
    SupervisionBatch,
    Supervisor,
    TrajectoryBatch,
    TrajectoryRecord,
)
from posttrain_circuits.supervision.verified_replay import (
    InsufficientPositiveTrajectories,
    VerifiedReplaySupervisor,
)
from posttrain_circuits.training.checkpointing import atomic_torch_save, load_checkpoint, save_checkpoint
from posttrain_circuits.training.optimizer import parameter_update_norm
from posttrain_circuits.training.schedules import PromptScheduler

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


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int = 2
    steps_per_round: int = 1
    learning_rate: float = 5e-4
    checkpoint_every: int = 1
    backend: str = "torch_smoke"
    gradient_accumulation_steps: int = 1
    max_completion_length: int = 128
    require_evaluation_metrics: bool = False

    def __post_init__(self) -> None:
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.steps_per_round < 1:
            raise ValueError("steps_per_round must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")


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
        dependency_versions: dict[str, str] | None = None,
        resume_ancestry: list[str] | None = None,
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
        self.git_commit = git_commit
        self.dependency_versions = dependency_versions or {}
        self.resume_ancestry = list(resume_ancestry or [])
        self.probe_input_ids = probe_input_ids
        self.evaluation_fn = evaluation_fn
        if config.require_evaluation_metrics and evaluation_fn is None:
            raise ValueError("formal training requires an evaluation callback before any update")
        self.global_step = 0
        self.online_rollout_round = 0
        self.cumulative_counts: dict[str, float] = {
            "prompts_consumed": 0.0,
            "trajectories_generated": 0.0,
            "response_tokens_generated": 0.0,
            "supervised_response_tokens": 0.0,
            "forward_backward_flop_estimate": 0.0,
        }
        self._pending_collection: dict[str, float] = {key: 0.0 for key in _COLLECTION_METRICS}
        self._pending_processed_tokens = 0.0
        self._pending_metric_sums: dict[str, float] = {}
        self._pending_metric_calls = 0
        self._accumulation_micro_step = 0
        self._parameters_before_update: list[torch.Tensor] | None = None
        self._terminate = False
        self._accelerator: Any = None
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
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        flop_estimate = 6.0 * parameter_count * base_supervised_tokens
        self.cumulative_counts["forward_backward_flop_estimate"] += flop_estimate

        if sync_gradients:
            self._accumulation_micro_step = 0
        return sync_gradients

    def _evaluation_metrics(self) -> dict[str, float | None]:
        metrics: dict[str, float | None] = dict.fromkeys(_REQUIRED_EVALUATION_METRICS)
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
        metric: dict[str, Any] = {
            **objective_metrics,
            **collection_metrics,
            "step": float(self.global_step),
            **self.cumulative_counts,
            "effective_supervised_tokens_this_update": (self._pending_processed_tokens),
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
        self._pending_metric_sums = {}
        self._pending_metric_calls = 0
        self._parameters_before_update = None
        return metric

    def train(self) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        started = time.perf_counter()
        while self.global_step < self.config.max_steps:
            prompts = self.prompt_scheduler.next_batch()
            trajectories, supervision, attempts = self._collect_trajectories(prompts)
            self._register_collection(
                prompts,
                trajectories,
                supervision,
                attempts,
            )
            for _ in range(self.config.steps_per_round):
                sync_gradients = self._training_micro_step(supervision)
                if not sync_gradients:
                    continue
                metric = self._finish_optimizer_update(started=started)
                history.append(metric)
                if self.is_main_process:
                    append_metric(self.run_dir / "metrics.jsonl", metric)
                if self.global_step % self.config.checkpoint_every == 0 or self._terminate:
                    self.save(self.run_dir / "checkpoints" / f"step-{self.global_step:08d}.pt")
                if self._terminate or self.global_step >= self.config.max_steps:
                    break
            if self._terminate:
                break
        return history

    def save(self, path: Path) -> None:
        if self._accumulation_micro_step != 0:
            raise RuntimeError("checkpoint requested inside a partial gradient-accumulation window")
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
            state_dir = path.with_suffix(".accelerate")
            self._accelerator.save_state(str(state_dir))
            self._accelerator.wait_for_everyone()
            full_model_state = self._accelerator.get_state_dict(self.model)
            optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
            scaler = getattr(self._accelerator, "scaler", None)
            if self.is_main_process:
                atomic_torch_save(
                    path,
                    {
                        "format": "accelerate_fsdp_full_export_v1",
                        "model": full_model_state,
                        "optimizer": optimizer_state,
                        "scheduler": scheduler_state,
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "accelerate_state_dir": str(state_dir.resolve()),
                        "rng": RNGState.capture().as_dict(),
                        "prompt_scheduler": self.prompt_scheduler.state_dict(),
                        "state_source": self.state_source.state_dict(),
                        "trainer_state": {
                            "cumulative_counts": self.cumulative_counts,
                            "accumulation_micro_step": self._accumulation_micro_step,
                        },
                        "global_step": self.global_step,
                        "policy_version": int(getattr(self.state_source, "policy_version", 0)),
                        "online_rollout_round": self.online_rollout_round,
                        "resolved_config": self.resolved_config,
                        "manifest_hashes": self.manifest_hashes,
                        "git_commit": self.git_commit,
                        "dependency_versions": self.dependency_versions,
                        "resume_ancestry": self.resume_ancestry,
                    },
                )
            self._accelerator.wait_for_everyone()
            return
        scaler_state = None
        accelerator_state = None
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            prompt_scheduler_state=self.prompt_scheduler.state_dict(),
            state_source_state=self.state_source.state_dict(),
            trainer_state={
                "cumulative_counts": self.cumulative_counts,
                "accumulation_micro_step": self._accumulation_micro_step,
            },
            global_step=self.global_step,
            policy_version=int(getattr(self.state_source, "policy_version", 0)),
            online_rollout_round=self.online_rollout_round,
            resolved_config=self.resolved_config,
            manifest_hashes=self.manifest_hashes,
            accelerator_state=accelerator_state,
            scaler_state=scaler_state,
            git_commit=self.git_commit,
            dependency_versions=self.dependency_versions,
            resume_ancestry=self.resume_ancestry,
        )

    def resume(self, path: Path) -> None:
        if self._accelerator is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format") != "accelerate_fsdp_full_export_v1":
                raise ValueError("Accelerate resume requires an Accelerate/FSDP checkpoint export")
            self._accelerator.load_state(str(payload["accelerate_state_dir"]))
            self._accelerator.wait_for_everyone()
            self.prompt_scheduler.load_state_dict(payload["prompt_scheduler"])
            self.state_source.load_state_dict(payload["state_source"])
            self.global_step = int(payload["global_step"])
            self.online_rollout_round = int(payload["online_rollout_round"])
            self.cumulative_counts.update(payload.get("trainer_state", {}).get("cumulative_counts", {}))
            self._accumulation_micro_step = int(
                payload.get("trainer_state", {}).get("accumulation_micro_step", 0)
            )
            self.resume_ancestry = [*payload.get("resume_ancestry", []), str(path)]
            return
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.prompt_scheduler.load_state_dict(payload["prompt_scheduler"])
        self.state_source.load_state_dict(payload["state_source"])
        self.global_step = int(payload["global_step"])
        self.online_rollout_round = int(payload["online_rollout_round"])
        trainer_state = payload.get("trainer_state", {})
        self.cumulative_counts.update(trainer_state.get("cumulative_counts", {}))
        self._accumulation_micro_step = int(trainer_state.get("accumulation_micro_step", 0))
        if self._accumulation_micro_step != 0:
            raise ValueError("checkpoint contains an unsupported partial accumulation window")
        scaler = getattr(self._accelerator, "scaler", None)
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        self.resume_ancestry = [
            *payload.get("resume_ancestry", []),
            str(path),
        ]
