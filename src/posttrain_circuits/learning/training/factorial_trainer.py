"""One trainer for the entire 2 x 3 controlled factorial grid."""

from __future__ import annotations

import copy
import math
import os
import signal
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
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
from posttrain_circuits.learning.teacher.demo_source import TeacherDemoStateSource
from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor
from posttrain_circuits.learning.training.fsdp_contract import (
    full_state_dict_options,
    validate_model_fsdp_sharding,
)
from posttrain_circuits.learning.training.optimizer import parameter_update_norm
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
    LEGACY_BATCH_PARTITION_PROTOCOL,
    ExactGlobalBatchPlan,
    PromptScheduler,
)
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
_NON_OBJECTIVE_SUPERVISION_METRICS = frozenset((*_COLLECTION_METRICS, "reward_rate"))
_CUMULATIVE_COUNT_KEYS = (
    "prompts_consumed",
    "trajectories_generated",
    "response_tokens_generated",
    "supervised_response_tokens",
    "model_facing_input_tokens_processed",
    "forward_backward_flop_estimate",
)


def _environment_distributed_position() -> tuple[int, int]:
    """Read the launcher topology before constructing ``Accelerator``.

    Accelerate needs its accumulation count at construction time.  The
    scheduler-owned launcher is the authority for this environment; a mismatch
    with Accelerate's observed topology is checked again after preparation.
    """

    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    raw_rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        world_size = int(raw_world_size)
        rank = int(raw_rank)
    except ValueError as error:
        raise ValueError("distributed launcher environment contains a non-integer rank/world size") from error
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("distributed launcher environment has an invalid rank/world size")
    return rank, world_size


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


def _allocation_neutral_global_objective_metrics(
    rows: list[object],
    *,
    world_size: int,
    global_batch_size: int,
) -> dict[str, float]:
    """Combine rank-local, sequence-weighted objective summaries exactly.

    Each row is produced after all local microbatches in one optimizer window:
    a metric numerator is the sum of ``local_sequence_mean * local_batch``.
    The only valid denominator for Candidate E is the 64 global logical
    sequences, including the three-rank ``2/1/1`` tail.
    """

    if type(world_size) is not int or world_size < 1:
        raise ValueError("allocation-neutral objective world_size is invalid")
    if type(global_batch_size) is not int or global_batch_size < 1:
        raise ValueError("allocation-neutral objective global batch size is invalid")
    if len(rows) != world_size:
        raise ValueError("allocation-neutral objective summaries omit a rank")
    metric_keys: tuple[str, ...] | None = None
    numerators: dict[str, float] = {}
    denominator = 0
    for expected_rank, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError("allocation-neutral objective summary is not a mapping")
        if raw_row.get("rank") != expected_rank:
            raise ValueError("allocation-neutral objective rank order changed")
        local_sequence_count = raw_row.get("sequence_count")
        if type(local_sequence_count) is not int or local_sequence_count < 1:
            raise ValueError("allocation-neutral objective sequence count is invalid")
        raw_numerators = raw_row.get("metric_numerators")
        if not isinstance(raw_numerators, Mapping) or not raw_numerators:
            raise ValueError("allocation-neutral objective metrics are missing")
        if any(not isinstance(name, str) for name in raw_numerators):
            raise ValueError("allocation-neutral objective metric key is invalid")
        keys = tuple(sorted(raw_numerators))
        if metric_keys is None:
            metric_keys = keys
            numerators = {name: 0.0 for name in keys}
        elif keys != metric_keys:
            raise ValueError("allocation-neutral objective metric keys differ across ranks")
        for name in keys:
            value = raw_numerators[name]
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError("allocation-neutral objective metric numerator is invalid")
            numerators[name] += float(value)
        denominator += local_sequence_count
    if denominator != global_batch_size:
        raise ValueError(
            "allocation-neutral objective denominator differs from the global batch size"
        )
    return {
        **{name: value / denominator for name, value in numerators.items()},
        "objective_metric_sequence_count": float(denominator),
    }


def _optimizer_step_number(value: object) -> int:
    """Normalize the scalar AdamW step representation without rounding it."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("optimizer state step is not scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("optimizer state step is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError("optimizer state step is not a non-negative integer")
    return int(numeric)


def _validate_optimizer_scheduler_cadence(
    *,
    optimizer_state: object,
    scheduler_state: object,
    global_step: int,
) -> None:
    """Bind one raw LambdaLR/AdamW state to a global optimizer boundary."""

    if type(global_step) is not int or global_step < 0:
        raise ValueError("optimizer cadence global_step is invalid")
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("checkpoint scheduler state is missing")
    last_epoch = scheduler_state.get("last_epoch")
    step_count = scheduler_state.get("_step_count")
    if type(last_epoch) is not int or last_epoch != global_step:
        raise ValueError("scheduler last_epoch differs from global optimizer step")
    if type(step_count) is not int or step_count != global_step + 1:
        raise ValueError("scheduler step count differs from global optimizer step")

    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer state is missing")
    states = optimizer_state.get("state")
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(states, Mapping) or not isinstance(param_groups, list) or not param_groups:
        raise ValueError("checkpoint AdamW state is incomplete")
    if global_step == 0:
        if states:
            raise ValueError("checkpoint AdamW state exists before the first optimizer update")
        return
    if not states:
        raise ValueError("checkpoint AdamW state is empty after an optimizer update")
    for state in states.values():
        if not isinstance(state, Mapping) or "step" not in state:
            raise ValueError("checkpoint AdamW parameter state lacks its step")
        if _optimizer_step_number(state["step"]) != global_step:
            raise ValueError("AdamW parameter step differs from global optimizer step")


def _parameter_update_squared_norm(
    before: list[torch.Tensor],
    parameters: Any,
) -> float:
    """Return one rank's double-precision squared parameter displacement."""

    total = torch.zeros((), dtype=torch.float64)
    for old, current in zip(before, parameters, strict=True):
        delta = current.detach().cpu().double() - old.double()
        total += delta.square().sum()
    result = float(total)
    if not math.isfinite(result) or result < 0:
        raise ValueError("local parameter update squared norm is invalid")
    return result


def _allocation_neutral_global_public_metrics(
    rows: list[object],
    *,
    world_size: int,
    global_batch_size: int,
    global_step: int,
) -> dict[str, Any]:
    """Aggregate public audit metrics while retaining rank-local checkpoints."""

    if type(world_size) is not int or world_size not in (1, 2, 3, 4):
        raise ValueError("allocation-neutral public metric world_size is invalid")
    if global_batch_size != 64 or type(global_batch_size) is not int:
        raise ValueError("allocation-neutral public metric global batch is invalid")
    if type(global_step) is not int or global_step < 1:
        raise ValueError("allocation-neutral public metric optimizer step is invalid")
    if len(rows) != world_size:
        raise ValueError("allocation-neutral public metrics omit a rank")

    collection_totals = {name: 0.0 for name in _COLLECTION_METRICS}
    cumulative_totals = {
        name: 0.0
        for name in _CUMULATIVE_COUNT_KEYS
        if name != "model_facing_input_tokens_processed"
    }
    global_input_tokens: int | None = None
    cumulative_input_tokens: float | None = None
    supervised_tokens = 0.0
    local_input_token_sum = 0
    update_squared_norm = 0.0
    maximum_peak_memory = 0.0
    maximum_wall_clock = 0.0
    normalized_rows: list[dict[str, Any]] = []

    for expected_rank, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or raw_row.get("rank") != expected_rank:
            raise ValueError("allocation-neutral public metric rank order changed")
        row = dict(raw_row)
        plan = ExactGlobalBatchPlan(64, 4, expected_rank, world_size)
        if row.get("local_sequence_count") != plan.local_sequence_count:
            raise ValueError("allocation-neutral public metric local sequence count changed")

        collection = row.get("collection")
        cumulative = row.get("cumulative")
        if not isinstance(collection, Mapping) or set(collection) != set(_COLLECTION_METRICS):
            raise ValueError("allocation-neutral public collection metrics are incomplete")
        if not isinstance(cumulative, Mapping) or set(cumulative) != set(
            _CUMULATIVE_COUNT_KEYS
        ):
            raise ValueError("allocation-neutral public cumulative metrics are incomplete")
        for name, value in (*collection.items(), *cumulative.items()):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    f"allocation-neutral public metric {name!r} is invalid"
                )
        expected_local_sequences = global_step * plan.local_sequence_count
        if (
            float(collection["generated_trajectories"])
            != float(plan.local_sequence_count)
            or float(cumulative["prompts_consumed"])
            != float(expected_local_sequences)
            or float(cumulative["trajectories_generated"])
            != float(expected_local_sequences)
        ):
            raise ValueError("allocation-neutral public sample counters changed")

        local_supervised_tokens = row.get("local_supervised_tokens")
        local_input_tokens = row.get("local_input_tokens")
        row_global_input_tokens = row.get("global_input_tokens")
        local_squared_norm = row.get("parameter_update_squared_norm")
        peak_memory = row.get("peak_allocated_gpu_memory")
        wall_clock = row.get("wall_clock_seconds")
        numeric_values = (
            local_supervised_tokens,
            local_squared_norm,
            peak_memory,
            wall_clock,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in numeric_values
        ):
            raise ValueError("allocation-neutral public rank diagnostic is invalid")
        if type(local_input_tokens) is not int or local_input_tokens < 0:
            raise ValueError("allocation-neutral local input-token count is invalid")
        if type(row_global_input_tokens) is not int or row_global_input_tokens < 1:
            raise ValueError("allocation-neutral global input-token count is invalid")
        if not math.isclose(
            float(local_supervised_tokens),
            float(collection["effective_supervised_tokens"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("allocation-neutral supervised-token metrics disagree")

        if global_input_tokens is None:
            global_input_tokens = row_global_input_tokens
            cumulative_input_tokens = float(
                cumulative["model_facing_input_tokens_processed"]
            )
        elif (
            row_global_input_tokens != global_input_tokens
            or float(cumulative["model_facing_input_tokens_processed"])
            != cumulative_input_tokens
        ):
            raise ValueError("allocation-neutral global token counters differ across ranks")

        for name in _COLLECTION_METRICS:
            collection_totals[name] += float(collection[name])
        for name in cumulative_totals:
            cumulative_totals[name] += float(cumulative[name])
        supervised_tokens += float(local_supervised_tokens)
        local_input_token_sum += local_input_tokens
        update_squared_norm += float(local_squared_norm)
        maximum_peak_memory = max(maximum_peak_memory, float(peak_memory))
        maximum_wall_clock = max(maximum_wall_clock, float(wall_clock))
        normalized_rows.append(row)

    assert global_input_tokens is not None
    assert cumulative_input_tokens is not None
    if local_input_token_sum != global_input_tokens:
        raise ValueError("allocation-neutral local token sum differs from the reserved global count")
    if collection_totals["generated_trajectories"] != float(global_batch_size):
        raise ValueError("allocation-neutral optimizer window does not contain 64 trajectories")
    expected_global_sequences = float(global_step * global_batch_size)
    if (
        cumulative_totals["prompts_consumed"] != expected_global_sequences
        or cumulative_totals["trajectories_generated"] != expected_global_sequences
    ):
        raise ValueError("allocation-neutral cumulative sample count is not global")

    generated = collection_totals["generated_trajectories"]
    successful = collection_totals["successful_trajectories"]
    cumulative_counts = {
        **cumulative_totals,
        "model_facing_input_tokens_processed": cumulative_input_tokens,
    }
    return {
        "collection_metrics": {
            **collection_totals,
            "reward_rate": successful / generated if generated else 0.0,
        },
        "cumulative_counts": cumulative_counts,
        "effective_supervised_tokens_this_update": supervised_tokens,
        "parameter_update_norm": math.sqrt(update_squared_norm),
        "peak_allocated_gpu_memory": maximum_peak_memory,
        "wall_clock_seconds": maximum_wall_clock,
        "rank_rows": normalized_rows,
    }


@dataclass(frozen=True)
class _ResumeRuntimePreflight:
    prompt_states: list[dict[str, Any]]
    source_states: list[dict[str, Any]]
    cumulative_counts: dict[str, float]
    token_budget_state: dict[str, Any]
    global_step: int
    online_rollout_round: int
    resume_ancestry: list[str]


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
    batch_partition_protocol: str = LEGACY_BATCH_PARTITION_PROTOCOL
    global_batch_size: int | None = None
    max_microbatch_size: int | None = None
    max_model_input_length: int | None = None
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
        if self.batch_partition_protocol == LEGACY_BATCH_PARTITION_PROTOCOL:
            if (
                self.global_batch_size is not None
                or self.max_microbatch_size is not None
                or self.max_model_input_length is not None
            ):
                raise ValueError("legacy batch partition must not declare exact-global batch fields")
        elif self.batch_partition_protocol == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1:
            if self.steps_per_round != 1:
                raise ValueError(
                    "allocation-neutral exact-global batching requires steps_per_round=1"
                )
            if (
                self.global_batch_size is None
                or self.max_microbatch_size is None
                or self.max_model_input_length is None
            ):
                raise ValueError(
                    "allocation-neutral exact-global batching requires global_batch_size "
                    "max_microbatch_size, and max_model_input_length"
                )
            if (
                type(self.max_model_input_length) is not int
                or self.max_model_input_length < 1
            ):
                raise ValueError(
                    "allocation-neutral max_model_input_length must be a positive integer"
                )
            # Constructing the one-rank view validates the frozen 64/4 protocol
            # values without binding this configuration to a running allocation.
            ExactGlobalBatchPlan(
                global_batch_size=self.global_batch_size,
                max_microbatch_size=self.max_microbatch_size,
                rank=0,
                world_size=1,
            )
        else:
            raise ValueError(
                f"unsupported trainer batch_partition_protocol {self.batch_partition_protocol!r}"
            )
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive")
        if self.evaluation_every < 1:
            raise ValueError("evaluation_every must be positive")

    @property
    def uses_allocation_neutral_batching(self) -> bool:
        return self.batch_partition_protocol == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1


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
        self._full_parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        if self._full_parameter_count < 1:
            raise ValueError("Factorial trainer model has no parameters")
        self.prompt_scheduler = prompt_scheduler
        self.state_source = state_source
        self.supervisor = supervisor
        self.config = config
        self._batch_partition_plan: ExactGlobalBatchPlan | None = None
        self._gradient_accumulation_steps = config.gradient_accumulation_steps
        if config.uses_allocation_neutral_batching:
            if not isinstance(supervisor, CanonicalSFTSupervisor):
                raise ValueError(
                    "allocation-neutral exact-global batching is currently reviewed only for canonical SFT"
                )
            if supervisor.normalization != "sequence":
                raise ValueError(
                    "allocation-neutral canonical SFT requires sequence-normalized loss"
                )
            if (
                not isinstance(state_source, TeacherDemoStateSource)
                or state_source.state_dict().get("kind") != "teacher_demo"
            ):
                raise ValueError(
                    "allocation-neutral exact-global batching is currently reviewed only for "
                    "the deterministic teacher-demo state source"
                )
            scheduler_plan = prompt_scheduler.exact_global_batch_plan
            if scheduler_plan is None:
                raise ValueError(
                    "allocation-neutral trainer requires an allocation-neutral prompt scheduler"
                )
            assert config.global_batch_size is not None
            assert config.max_microbatch_size is not None
            if (
                scheduler_plan.global_batch_size != config.global_batch_size
                or scheduler_plan.max_microbatch_size != config.max_microbatch_size
            ):
                raise ValueError("prompt scheduler differs from the configured exact-global batch contract")
            if config.backend == "accelerate":
                launch_rank, launch_world_size = _environment_distributed_position()
                launch_plan = ExactGlobalBatchPlan(
                    global_batch_size=config.global_batch_size,
                    max_microbatch_size=config.max_microbatch_size,
                    rank=launch_rank,
                    world_size=launch_world_size,
                )
                if scheduler_plan != launch_plan:
                    raise ValueError(
                        "prompt scheduler differs from the scheduler-owned launcher allocation"
                    )
                self._gradient_accumulation_steps = (
                    launch_plan.microsteps_per_optimizer_update
                )
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
            name: 0.0 for name in _CUMULATIVE_COUNT_KEYS
        }
        self._pending_collection: dict[str, float] = {key: 0.0 for key in _COLLECTION_METRICS}
        self._pending_processed_tokens = 0.0
        self._pending_input_tokens = 0
        self._current_global_update_tokens = 0
        self._pending_metric_sums: dict[str, float] = {}
        self._pending_metric_calls = 0
        self._pending_metric_sequence_count = 0
        self._accumulation_micro_step = 0
        self._parameters_before_update: list[torch.Tensor] | None = None
        self._terminate = False
        self._accelerator: Any = None
        self._fsdp_sharding_contract: dict[str, object] | None = None
        self._last_checkpoint_step: int | None = None
        self._last_checkpoint_path: Path | None = None
        self._last_checkpoint_sha256: str | None = None
        self._last_checkpoint_token_budget_state: dict[str, Any] | None = None
        self._last_accelerator_state_files: dict[str, str] | None = None
        self._last_accelerator_state_sha256: str | None = None
        self.final_checkpoint_path: Path | None = None
        if config.backend == "accelerate":
            try:
                from accelerate import Accelerator
            except ImportError as error:
                raise RuntimeError("Accelerate backend requested; install the 'train' extra") from error
            self._accelerator = Accelerator(
                gradient_accumulation_steps=self._gradient_accumulation_steps,
                step_scheduler_with_optimizer=False,
            )
            self.model, self.optimizer = self._accelerator.prepare(
                self.model,
                self.optimizer,
            )
            # The trainer owns a scientific optimizer-step cadence rather than
            # a prepared-dataloader cadence.  Register the raw scheduler for
            # state save/load and step it explicitly once per real update.
            self._accelerator.register_for_checkpointing(self.scheduler)
            if config.uses_allocation_neutral_batching:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                if not isinstance(self.model, FSDP):
                    raise RuntimeError(
                        "allocation-neutral Accelerate training requires the prepared "
                        "model root to be a real FSDP wrapper"
                    )
                _, prepared_world_size = self._distributed_position()
                self._fsdp_sharding_contract = validate_model_fsdp_sharding(
                    self.model,
                    world_size=prepared_world_size,
                    fsdp_type=FSDP,
                )
        self._configure_batch_partition_runtime()
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

    def _validate_live_optimizer_scheduler_cadence(self) -> None:
        """Assert exact-global runtime state is at one optimizer boundary."""

        if self._batch_partition_plan is None:
            return
        _validate_optimizer_scheduler_cadence(
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            global_step=self.global_step,
        )

    def _configure_batch_partition_runtime(self) -> None:
        """Bind the reviewed global protocol to the actual running allocation."""

        if not self.config.uses_allocation_neutral_batching:
            return
        rank, world_size = self._distributed_position()
        assert self.config.global_batch_size is not None
        assert self.config.max_microbatch_size is not None
        plan = ExactGlobalBatchPlan(
            global_batch_size=self.config.global_batch_size,
            max_microbatch_size=self.config.max_microbatch_size,
            rank=rank,
            world_size=world_size,
        )
        scheduler_plan = self.prompt_scheduler.exact_global_batch_plan
        if scheduler_plan != plan:
            raise ValueError(
                "allocation-neutral prompt scheduler does not match the running rank/world size"
            )
        if self._accelerator is None and world_size > 1:
            raise RuntimeError(
                "allocation-neutral multi-rank training requires Accelerate so FSDP/DDP "
                "reduction semantics are explicit"
            )
        if self._accelerator is not None and (
            self._gradient_accumulation_steps != plan.microsteps_per_optimizer_update
        ):
            raise RuntimeError(
                "Accelerate was initialized with the wrong allocation-neutral accumulation count"
            )
        self._batch_partition_plan = plan
        self._gradient_accumulation_steps = plan.microsteps_per_optimizer_update

    def _checkpoint_batch_partition_state(self) -> dict[str, Any] | None:
        """Return allocation-specific checkpoint evidence for the elastic protocol."""

        plan = self._batch_partition_plan
        if plan is None:
            return None
        rank, world_size = self._distributed_position()
        running_plan = ExactGlobalBatchPlan(
            global_batch_size=plan.global_batch_size,
            max_microbatch_size=plan.max_microbatch_size,
            rank=rank,
            world_size=world_size,
        )
        if running_plan != plan or self.prompt_scheduler.exact_global_batch_plan != plan:
            raise RuntimeError("allocation-neutral batch plan changed after trainer initialization")
        rank_plans = [
            ExactGlobalBatchPlan(
                global_batch_size=plan.global_batch_size,
                max_microbatch_size=plan.max_microbatch_size,
                rank=member_rank,
                world_size=plan.world_size,
            )
            for member_rank in range(plan.world_size)
        ]
        state: dict[str, Any] = {
            "protocol": ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
            "global_batch_size": plan.global_batch_size,
            "max_microbatch_size": plan.max_microbatch_size,
            "max_model_input_length": self.config.max_model_input_length,
            "world_size": plan.world_size,
            "microsteps_per_optimizer_update": plan.microsteps_per_optimizer_update,
            "rank_local_sequence_counts": [
                member_plan.local_sequence_count for member_plan in rank_plans
            ],
            "microbatch_sizes_by_rank": [
                list(member_plan.microbatch_sizes) for member_plan in rank_plans
            ],
        }
        if self.config.backend == "accelerate":
            if self._fsdp_sharding_contract is None:
                raise RuntimeError(
                    "production allocation-neutral checkpoint lacks an FSDP strategy contract"
                )
            observed_contract = validate_model_fsdp_sharding(
                self.model,
                world_size=world_size,
            )
            if observed_contract != self._fsdp_sharding_contract:
                raise RuntimeError("prepared FSDP strategy contract changed during training")
            state.update(observed_contract)
        return state

    def _full_model_state_for_checkpoint(self) -> dict[str, Any]:
        """Export a full FSDP state without Accelerate's unsafe W=1 options."""

        if self._accelerator is None:
            raise RuntimeError("full Accelerate checkpoint export requires an Accelerator")
        if self.config.uses_allocation_neutral_batching:
            from torch.distributed.fsdp import (
                FullStateDictConfig,
                FullyShardedDataParallel as FSDP,
                StateDictType,
            )

            if not isinstance(self.model, FSDP):
                raise RuntimeError(
                    "production allocation-neutral checkpoint model is not an FSDP root"
                )
            _, world_size = self._distributed_position()
            options = full_state_dict_options(world_size)
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(**options),
            ):
                return self.model.state_dict()
        return self._accelerator.get_state_dict(self.model)

    def _validate_checkpoint_batch_partition(self, payload: Mapping[str, Any]) -> None:
        """Reject checkpoints that do not bind this exact running allocation."""

        expected = self._checkpoint_batch_partition_state()
        if expected is None:
            # Legacy checkpoints deliberately retain their historical format and
            # behavior.  Only the opt-in protocol gains this extra evidence.
            return
        trainer_state = payload.get("trainer_state")
        if not isinstance(trainer_state, Mapping):
            raise ValueError("checkpoint has no trainer state for allocation-neutral batch validation")
        if trainer_state.get("batch_partition") != expected:
            raise ValueError(
                "resume checkpoint batch partition differs from the running allocation"
            )

    def _validate_trainer_runtime_state(
        self,
        trainer_state: object,
        *,
        global_step: int,
        rank_bound: bool,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Validate one rank's immutable trainer metadata without loading it."""

        if not isinstance(trainer_state, Mapping):
            raise ValueError("resume checkpoint trainer_state is missing")
        expected_trainer_keys = {
            "cumulative_counts",
            "accumulation_micro_step",
            "token_budget",
        }
        expected_batch_partition = self._checkpoint_batch_partition_state()
        if expected_batch_partition is not None:
            expected_trainer_keys.add("batch_partition")
        if rank_bound:
            expected_trainer_keys.update(("rank", "world_size"))
        if set(trainer_state) != expected_trainer_keys:
            raise ValueError("resume checkpoint trainer_state fields are incomplete or unexpected")
        if trainer_state.get("accumulation_micro_step") != 0 or type(
            trainer_state.get("accumulation_micro_step")
        ) is not int:
            raise ValueError("checkpoint contains an unsupported partial accumulation window")
        if (
            expected_batch_partition is not None
            and trainer_state.get("batch_partition") != expected_batch_partition
        ):
            raise ValueError(
                "resume checkpoint batch partition differs from the running allocation"
            )

        raw_counts = trainer_state.get("cumulative_counts")
        if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(_CUMULATIVE_COUNT_KEYS):
            raise ValueError("resume checkpoint cumulative counts are incomplete")
        cumulative_counts: dict[str, float] = {}
        for name in _CUMULATIVE_COUNT_KEYS:
            value = raw_counts[name]
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"resume checkpoint cumulative count {name!r} is invalid")
            cumulative_counts[name] = float(value)

        raw_token_budget = trainer_state.get("token_budget")
        expected_budget_keys = set(self.token_budget.state_dict())
        if (
            not isinstance(raw_token_budget, Mapping)
            or set(raw_token_budget) != expected_budget_keys
        ):
            raise ValueError("resume checkpoint token budget state is incomplete")
        for name in ("budget", "consumed", "accepted_optimizer_updates"):
            if type(raw_token_budget.get(name)) is not int:
                raise ValueError(f"resume checkpoint token budget {name!r} is invalid")
        if not isinstance(raw_token_budget.get("unit"), str):
            raise ValueError("resume checkpoint token budget unit is invalid")
        allowed_stop_reasons = {
            None,
            "token_budget_exactly_consumed",
            "token_budget_exhausted_before_next_optimizer_update",
            "signal_at_optimizer_boundary",
            "max_steps_safety_limit",
        }
        if raw_token_budget.get("stop_reason") not in allowed_stop_reasons:
            raise ValueError("resume checkpoint token budget stop reason is invalid")
        token_budget_state = dict(raw_token_budget)
        checked_budget = TokenBudgetState(self.config.token_budget, self.config.token_budget_unit)
        checked_budget.load_state_dict(token_budget_state)
        if checked_budget.accepted_optimizer_updates != global_step:
            raise ValueError("resume checkpoint token budget update count differs from global_step")
        if (
            cumulative_counts["model_facing_input_tokens_processed"]
            != float(checked_budget.consumed)
        ):
            raise ValueError("resume checkpoint token budget differs from cumulative input tokens")
        if (
            checked_budget.stop_reason == "token_budget_exactly_consumed"
            and checked_budget.remaining != 0
        ):
            raise ValueError("resume checkpoint exact-consumption reason has remaining tokens")
        if (
            checked_budget.remaining == 0
            and checked_budget.stop_reason != "token_budget_exactly_consumed"
        ):
            raise ValueError("resume checkpoint consumed its budget without the exact stop reason")
        if expected_batch_partition is not None:
            reason = checked_budget.stop_reason
            remaining = checked_budget.remaining
            if reason is None:
                valid_endpoint = global_step < self.config.max_steps and remaining > 0
            elif reason == "token_budget_exactly_consumed":
                valid_endpoint = (
                    remaining == 0
                    and 0 < global_step <= self.config.max_steps
                )
            elif reason == "max_steps_safety_limit":
                valid_endpoint = global_step == self.config.max_steps and remaining > 0
            elif reason in {
                "token_budget_exhausted_before_next_optimizer_update",
                "signal_at_optimizer_boundary",
            }:
                valid_endpoint = global_step < self.config.max_steps and remaining > 0
            else:  # guarded by ``allowed_stop_reasons`` above
                valid_endpoint = False
            if not valid_endpoint:
                raise ValueError(
                    "allocation-neutral checkpoint stop reason contradicts its endpoint"
                )
        return cumulative_counts, token_budget_state

    def _validate_exact_trainer_state_rows(
        self,
        observed: object,
        *,
        global_step: int,
        world_size: int,
        top_level_token_budget: object,
    ) -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, Any]]:
        """Validate every allocation-neutral rank counter before save or load."""

        trainer_states = _validate_rank_state_mappings(
            observed,
            name="trainer_state_by_rank",
            world_size=world_size,
        )
        count_states: list[dict[str, float]] = []
        budget_states: list[dict[str, Any]] = []
        for member_rank, trainer_state in enumerate(trainer_states):
            cumulative_counts, token_budget_state = self._validate_trainer_runtime_state(
                trainer_state,
                global_step=global_step,
                rank_bound=True,
            )
            assert self.config.global_batch_size is not None
            assert self.config.max_microbatch_size is not None
            member_plan = ExactGlobalBatchPlan(
                global_batch_size=self.config.global_batch_size,
                max_microbatch_size=self.config.max_microbatch_size,
                rank=member_rank,
                world_size=world_size,
            )
            expected_local_sequences = global_step * member_plan.local_sequence_count
            if (
                cumulative_counts["prompts_consumed"] != float(expected_local_sequences)
                or cumulative_counts["trajectories_generated"]
                != float(expected_local_sequences)
            ):
                raise ValueError(
                    "allocation-neutral trainer counters are not at the rank-local "
                    "optimizer boundary"
                )
            count_states.append(cumulative_counts)
            budget_states.append(token_budget_state)
        if any(state != budget_states[0] for state in budget_states[1:]):
            raise ValueError("allocation-neutral token budget differs across checkpoint ranks")
        if top_level_token_budget != budget_states[0]:
            raise ValueError("resume checkpoint top-level token budget state is inconsistent")
        return trainer_states, count_states, budget_states[0]

    def _expected_allocation_neutral_teacher_cursor(
        self,
        *,
        global_step: int,
        rank: int,
        world_size: int,
    ) -> dict[str, int]:
        """Reconstruct the exact rank-local cursor from global logical slots."""

        assert self.config.global_batch_size is not None
        expected: dict[str, int] = {}
        for window_index in range(global_step):
            window_start = window_index * self.config.global_batch_size
            for slot in range(rank, self.config.global_batch_size, world_size):
                prompt_id = self.prompt_scheduler.prompt_ids[
                    (window_start + slot) % len(self.prompt_scheduler.prompt_ids)
                ]
                expected[prompt_id] = expected.get(prompt_id, 0) + 1
        return expected

    def _preflight_resume_runtime_metadata(
        self,
        payload: Mapping[str, Any],
        *,
        checkpoint_path: Path,
        rank: int,
        world_size: int,
    ) -> _ResumeRuntimePreflight:
        """Validate every mutable runtime input before loading training state."""

        global_step = payload.get("global_step")
        if type(global_step) is not int or not 0 <= global_step <= self.config.max_steps:
            raise ValueError("resume checkpoint global_step is invalid")
        online_rollout_round = payload.get("online_rollout_round")
        if type(online_rollout_round) is not int or online_rollout_round < 0:
            raise ValueError("resume checkpoint online_rollout_round is invalid")
        policy_version = payload.get("policy_version")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("resume checkpoint policy_version is invalid")
        if self._batch_partition_plan is not None and (
            policy_version != 0 or online_rollout_round != 0
        ):
            raise ValueError(
                "allocation-neutral teacher-demo resume requires policy and rollout version zero"
            )
        if self._batch_partition_plan is not None:
            _validate_optimizer_scheduler_cadence(
                optimizer_state=payload.get("optimizer"),
                scheduler_state=payload.get("scheduler"),
                global_step=global_step,
            )

        trainer_state = payload.get("trainer_state")
        if self._batch_partition_plan is None:
            cumulative_counts, raw_token_budget = self._validate_trainer_runtime_state(
                trainer_state,
                global_step=global_step,
                rank_bound=False,
            )
            if payload.get("token_budget") != raw_token_budget:
                raise ValueError("resume checkpoint top-level token budget state is inconsistent")
        else:
            trainer_state_rows = payload.get("trainer_state_by_rank")
            if trainer_state_rows is None:
                # Historical single-rank torch-smoke checkpoints predate the
                # rank table and have no cross-rank ambiguity.  New Accelerate
                # exports, including W=1, must always carry the strict table.
                if world_size != 1 or payload.get("format") == "accelerate_fsdp_full_export_v1":
                    raise ValueError(
                        "allocation-neutral Accelerate checkpoint lacks trainer_state_by_rank"
                    )
                if not isinstance(trainer_state, Mapping):
                    raise ValueError("resume checkpoint trainer_state is missing")
                bound_trainer_state = dict(trainer_state)
                if bound_trainer_state.get("rank", 0) != 0:
                    raise ValueError("checkpoint trainer_state rank metadata differs at index 0")
                if bound_trainer_state.get("world_size", 1) != 1:
                    raise ValueError(
                        "checkpoint trainer_state world_size metadata differs at rank 0"
                    )
                bound_trainer_state.update(rank=0, world_size=1)
                trainer_state_rows = [bound_trainer_state]
            trainer_states, cumulative_count_rows, raw_token_budget = (
                self._validate_exact_trainer_state_rows(
                    trainer_state_rows,
                    global_step=global_step,
                    world_size=world_size,
                    top_level_token_budget=payload.get("token_budget"),
                )
            )
            if (
                payload.get("trainer_state_by_rank") is not None
                and trainer_state != trainer_states[0]
            ):
                raise ValueError("resume checkpoint rank-0 trainer state is inconsistent")
            cumulative_counts = cumulative_count_rows[rank]

        if (
            self._batch_partition_plan is not None
            and raw_token_budget.get("stop_reason") is not None
        ):
            raise ValueError(
                "allocation-neutral resume requires a non-terminal periodic checkpoint"
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
        expected_prompt_keys = set(self.prompt_scheduler.state_dict())
        expected_source_keys = set(self.state_source.state_dict()) | {"rank", "world_size"}
        for member_rank, (prompt_state, source_state) in enumerate(
            zip(prompt_states, source_states, strict=True)
        ):
            if set(prompt_state) != expected_prompt_keys:
                raise ValueError("resume checkpoint prompt scheduler state is incomplete")
            if set(source_state) != expected_source_keys:
                raise ValueError("resume checkpoint state-source state is incomplete")
            if self._batch_partition_plan is not None:
                assert self.config.global_batch_size is not None
                assert self.config.max_microbatch_size is not None
                member_plan = ExactGlobalBatchPlan(
                    global_batch_size=self.config.global_batch_size,
                    max_microbatch_size=self.config.max_microbatch_size,
                    rank=member_rank,
                    world_size=world_size,
                )
                expected_global_cursor = global_step * member_plan.global_batch_size
                if (
                    prompt_state.get("microbatch_index") != 0
                    or prompt_state.get("global_slot_cursor") != expected_global_cursor
                ):
                    raise ValueError(
                        "allocation-neutral prompt scheduler is not at the checkpointed "
                        "optimizer boundary"
                    )
                cursor = source_state.get("cursor")
                attempt_ids = source_state.get("attempt_ids")
                if (
                    source_state.get("kind") != "teacher_demo"
                    or not isinstance(cursor, Mapping)
                    or any(
                        not isinstance(key, str)
                        or type(value) is not int
                        or value < 0
                        for key, value in cursor.items()
                    )
                    or dict(cursor)
                    != self._expected_allocation_neutral_teacher_cursor(
                        global_step=global_step,
                        rank=member_rank,
                        world_size=world_size,
                    )
                    or not isinstance(attempt_ids, list)
                    or not attempt_ids
                    or any(not isinstance(value, str) or not value for value in attempt_ids)
                    or len(set(attempt_ids)) != len(attempt_ids)
                ):
                    raise ValueError(
                        "resume checkpoint teacher-demo source is not at the checkpointed "
                        "optimizer boundary"
                    )
            prompt_probe = copy.copy(self.prompt_scheduler)
            prompt_probe.rank = member_rank
            prompt_probe.world_size = world_size
            prompt_probe.load_state_dict(copy.deepcopy(prompt_state))
            source_probe = copy.copy(self.state_source)
            source_probe.load_state_dict(copy.deepcopy(source_state))

        return _ResumeRuntimePreflight(
            prompt_states=prompt_states,
            source_states=source_states,
            cumulative_counts=cumulative_counts,
            token_budget_state=dict(raw_token_budget),
            global_step=global_step,
            online_rollout_round=online_rollout_round,
            resume_ancestry=_extend_resume_ancestry(
                payload.get("resume_ancestry"),
                checkpoint_path,
            ),
        )

    def _trainer_runtime_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "cumulative_counts": self.cumulative_counts,
            "accumulation_micro_step": self._accumulation_micro_step,
            "token_budget": self.token_budget.state_dict(),
        }
        batch_partition = self._checkpoint_batch_partition_state()
        if batch_partition is not None:
            state["batch_partition"] = batch_partition
        return state

    def _capture_rejected_window_snapshot(self) -> dict[str, Any]:
        """Capture exact-mode state before collection advances a new window.

        Candidate E admits tokens only after its 64 logical prompts have been
        collected.  A rejected reservation must therefore roll the collection
        cursors and their accounting back to the preceding optimizer boundary.
        The token budget itself is intentionally excluded: it keeps the
        terminal rejection reason while its consumed count remains unchanged.
        """

        if self._batch_partition_plan is None:
            raise RuntimeError("only allocation-neutral windows are transactional")
        if self._accumulation_micro_step != 0 or self._parameters_before_update is not None:
            raise RuntimeError("cannot snapshot a partial optimizer window")
        return {
            "prompt_scheduler": copy.deepcopy(self.prompt_scheduler.state_dict()),
            "state_source": copy.deepcopy(self.state_source.state_dict()),
            "cumulative_counts": copy.deepcopy(self.cumulative_counts),
            "pending_collection": copy.deepcopy(self._pending_collection),
            "pending_processed_tokens": self._pending_processed_tokens,
            "pending_input_tokens": self._pending_input_tokens,
            "current_global_update_tokens": self._current_global_update_tokens,
            "pending_metric_sums": copy.deepcopy(self._pending_metric_sums),
            "pending_metric_calls": self._pending_metric_calls,
            "pending_metric_sequence_count": self._pending_metric_sequence_count,
        }

    def _restore_rejected_window_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore state advanced only by a token-budget-rejected collection."""

        self.prompt_scheduler.load_state_dict(dict(snapshot["prompt_scheduler"]))
        self.state_source.load_state_dict(dict(snapshot["state_source"]))
        self.cumulative_counts = dict(snapshot["cumulative_counts"])
        self._pending_collection = dict(snapshot["pending_collection"])
        self._pending_processed_tokens = float(snapshot["pending_processed_tokens"])
        self._pending_input_tokens = int(snapshot["pending_input_tokens"])
        self._current_global_update_tokens = int(snapshot["current_global_update_tokens"])
        self._pending_metric_sums = dict(snapshot["pending_metric_sums"])
        self._pending_metric_calls = int(snapshot["pending_metric_calls"])
        self._pending_metric_sequence_count = int(snapshot["pending_metric_sequence_count"])
        if self._accumulation_micro_step != 0 or self._parameters_before_update is not None:
            raise RuntimeError("rejected optimizer window unexpectedly reached backward")

    @staticmethod
    def _supervision_sequence_count(supervision: SupervisionBatch) -> int:
        if supervision.input_ids.ndim < 1:
            raise ValueError("supervision batch lacks a sequence dimension")
        count = int(supervision.input_ids.shape[0])
        if count < 1:
            raise ValueError("supervision batch is empty")
        return count

    def _loss_for_backward(
        self,
        loss: torch.Tensor,
        supervision: SupervisionBatch,
        *,
        expected_sequence_count: int | None,
    ) -> torch.Tensor:
        """Convert local objective means to the configured backward convention."""

        plan = self._batch_partition_plan
        if plan is None:
            return loss
        local_sequence_count = self._supervision_sequence_count(supervision)
        if expected_sequence_count is not None and local_sequence_count != expected_sequence_count:
            raise ValueError(
                "supervision sequence count differs from the allocation-neutral microbatch plan"
            )
        if local_sequence_count > plan.max_microbatch_size:
            raise ValueError("supervision batch exceeds the allocation-neutral microbatch cap")
        if self._accelerator is not None:
            return loss * plan.accelerate_sequence_mean_loss_multiplier(local_sequence_count)
        # The single-rank smoke backend performs its own accumulation below, so
        # it must receive a numerator contribution, not an already averaged
        # accumulation loss.  Multi-rank non-Accelerate execution is rejected
        # in ``_configure_batch_partition_runtime``.
        return loss * plan.manual_distributed_sequence_mean_loss_multiplier(
            local_sequence_count
        )

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

    def _gather_exact_trainer_states(self) -> list[dict[str, Any]]:
        """Gather and validate rank-local exact-mode accounting at a boundary."""

        if self._batch_partition_plan is None:
            raise RuntimeError("legacy checkpoints do not gather rank-local trainer states")
        rank, world_size = self._distributed_position()
        trainer_state = dict(self._trainer_runtime_state())
        trainer_state.update(rank=rank, world_size=world_size)
        trainer_states: list[object] = [trainer_state]
        if world_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("distributed trainer-state binding requires a process group")
            trainer_states = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(trainer_states, trainer_state)
        normalized, _, _ = self._validate_exact_trainer_state_rows(
            trainer_states,
            global_step=self.global_step,
            world_size=world_size,
            top_level_token_budget=self.token_budget.state_dict(),
        )
        return normalized

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

    def _reuse_current_checkpoint_if_valid(
        self,
        path: Path,
        *,
        allow_terminal_revision: bool = False,
    ) -> str | None:
        if self._last_checkpoint_step != self.global_step:
            return None
        expected_path = Path(path).absolute()
        published_path = self._last_checkpoint_path
        published_budget = self._last_checkpoint_token_budget_state
        if published_path is None or published_budget is None:
            raise RuntimeError("published checkpoint lost its runtime-state binding")
        if published_path.is_symlink() or not published_path.is_file():
            raise RuntimeError("previously published checkpoint is missing or no longer regular")
        observed_sha256 = sha256_file(published_path)
        if observed_sha256 != self._last_checkpoint_sha256:
            raise RuntimeError("previously published checkpoint bytes changed")
        if self._accelerator is not None:
            if (
                self._last_accelerator_state_files is None
                or self._last_accelerator_state_sha256 is None
            ):
                raise RuntimeError("published Accelerate checkpoint lost its state binding")
            validate_accelerator_state_directory(
                published_path.with_suffix(".accelerate"),
                expected_run_root=self.run_dir,
                expected_files=self._last_accelerator_state_files,
                expected_sha256=self._last_accelerator_state_sha256,
            )
        current_budget = self.token_budget.state_dict()
        if published_path != expected_path:
            canonical_path = (
                self.run_dir
                / "checkpoints"
                / f"step-{self.global_step:08d}.pt"
            ).absolute()
            terminal_path = canonical_path.with_name(
                f"step-{self.global_step:08d}-terminal.pt"
            )
            published_nonreason = dict(published_budget)
            current_nonreason = dict(current_budget)
            published_reason = published_nonreason.pop("stop_reason", None)
            current_reason = current_nonreason.pop("stop_reason", None)
            if not (
                allow_terminal_revision
                and published_path == canonical_path
                and expected_path == terminal_path
                and published_nonreason == current_nonreason
                and published_reason is None
                and current_reason is not None
            ):
                raise RuntimeError("one optimizer step cannot publish multiple checkpoint paths")
            return None
        if current_budget != published_budget:
            raise RuntimeError(
                "checkpoint runtime state changed after publication; publish a terminal revision"
            )
        return observed_sha256

    def _bind_known_terminal_stop_reason(self) -> None:
        """Bind a terminal reason before any checkpoint for the final update."""

        if self.token_budget.stop_reason is not None:
            return
        if self.global_step >= self.config.max_steps:
            self.token_budget.stop_reason = "max_steps_safety_limit"
        elif self._terminate:
            self.token_budget.stop_reason = "signal_at_optimizer_boundary"

    def _publish_final_checkpoint(self) -> Path:
        """Publish and expose the checkpoint that binds final runtime state.

        A periodic checkpoint can already own the canonical path for this
        optimizer step when the following token reservation or a late signal
        makes that same boundary terminal.  Preserve the immutable first file
        and publish exactly one explicitly named terminal revision.
        """

        canonical_path = (
            self.run_dir / "checkpoints" / f"step-{self.global_step:08d}.pt"
        ).absolute()
        current_budget = self.token_budget.state_dict()
        allow_terminal_revision = False
        final_path = canonical_path
        if self._last_checkpoint_step == self.global_step:
            if self._last_checkpoint_token_budget_state == current_budget:
                if self._last_checkpoint_path is None:
                    raise RuntimeError("published checkpoint lost its path binding")
                final_path = self._last_checkpoint_path
            else:
                final_path = canonical_path.with_name(
                    f"step-{self.global_step:08d}-terminal.pt"
                )
                allow_terminal_revision = True
        self.save(final_path, _allow_terminal_revision=allow_terminal_revision)
        self.final_checkpoint_path = final_path
        return final_path

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
        self._validate_supervision_model_input_length(supervision)
        return trajectories, supervision, attempts

    def _validate_supervision_model_input_length(
        self,
        supervision: SupervisionBatch,
    ) -> None:
        """Reject an overlength or misaligned batch without truncating identity.

        The accepted teacher-demo IDs already bind every token, so truncation
        would silently change the scientific example.  Exact-global G0 scans
        the complete accepted view in ``train`` before reaching this method;
        this tensor check additionally guards every materialized microbatch.
        """

        limit = self.config.max_model_input_length
        if limit is None:
            return
        input_ids = supervision.input_ids
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("supervision input_ids must be one non-empty rank-2 tensor")
        expected_prefix = tuple(input_ids.shape[:2])
        if expected_prefix[1] > limit:
            raise ValueError(
                "supervision tensor width exceeds max_model_input_length"
            )
        for name in (
            "attention_mask",
            "response_mask",
            "target_ids",
            "teacher_topk_ids",
            "teacher_topk_logprobs",
            "teacher_topk_mass",
            "sequence_ids",
        ):
            value = getattr(supervision, name)
            if value is None:
                continue
            if value.ndim < 2 or tuple(value.shape[:2]) != expected_prefix:
                raise ValueError(
                    f"supervision tensor {name} is not aligned to input_ids width"
                )

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

    def _record_objective_metrics(
        self,
        metrics: dict[str, float],
        *,
        sequence_count: int,
    ) -> None:
        """Accumulate objective values in the convention used by this trainer.

        Legacy training retains its historical equal-per-microstep metrics.
        Candidate E instead retains per-sequence numerators; only after a
        full optimizer window can ranks combine them with a denominator of 64.
        Collection counters remain in ``_pending_collection`` and deliberately
        retain their existing per-rank audit semantics.
        """

        plan = self._batch_partition_plan
        if plan is None:
            for key, value in metrics.items():
                self._pending_metric_sums[key] = self._pending_metric_sums.get(key, 0.0) + float(value)
            self._pending_metric_calls += 1
            return
        if sequence_count < 1 or sequence_count > plan.max_microbatch_size:
            raise ValueError("allocation-neutral objective metric sequence count is invalid")
        for key, value in metrics.items():
            if key in _NON_OBJECTIVE_SUPERVISION_METRICS:
                continue
            self._pending_metric_sums[key] = (
                self._pending_metric_sums.get(key, 0.0) + float(value) * sequence_count
            )
        self._pending_metric_sequence_count += sequence_count

    def _objective_metrics_for_optimizer_update(self) -> dict[str, float]:
        """Finalize optimizer-boundary objective metrics without tail bias."""

        plan = self._batch_partition_plan
        if plan is None:
            return {
                key: value / self._pending_metric_calls
                for key, value in self._pending_metric_sums.items()
            }
        rank, world_size = self._distributed_position()
        if world_size != plan.world_size or rank != plan.rank:
            raise RuntimeError("allocation-neutral objective metric topology changed")
        if self._pending_metric_sequence_count != plan.local_sequence_count:
            raise RuntimeError(
                "allocation-neutral optimizer window did not account for its local sequence count"
            )
        local_summary = {
            "rank": rank,
            "sequence_count": self._pending_metric_sequence_count,
            "metric_numerators": dict(self._pending_metric_sums),
        }
        summaries: list[object] = [local_summary]
        if world_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("allocation-neutral objective aggregation requires a process group")
            summaries = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(summaries, local_summary)
        return _allocation_neutral_global_objective_metrics(
            summaries,
            world_size=world_size,
            global_batch_size=plan.global_batch_size,
        )

    def _training_micro_step(
        self,
        supervision: SupervisionBatch,
        *,
        expected_sequence_count: int | None = None,
    ) -> bool:
        accumulation_context = (
            nullcontext()
            if self._accelerator is None
            else self._accelerator.accumulate(self.model)
        )
        # Accelerate's accumulation context must cover the forward pass as well
        # as backward/step so DDP/FSDP can suppress intermediate gradient sync.
        self._validate_supervision_model_input_length(supervision)
        with accumulation_context:
            if self._parameters_before_update is None:
                self._parameters_before_update = [
                    parameter.detach().cpu().clone() for parameter in self.model.parameters()
                ]
            output = self.supervisor.compute_loss(self.model, supervision)
            if not torch.isfinite(output.loss):
                raise FloatingPointError(
                    f"non-finite loss before optimizer step {self.global_step + 1}"
                )
            backward_loss = self._loss_for_backward(
                output.loss,
                supervision,
                expected_sequence_count=expected_sequence_count,
            )
            self._record_objective_metrics(
                output.metrics,
                sequence_count=self._supervision_sequence_count(supervision),
            )
            self._accumulation_micro_step += 1
            expected_sync = self._accumulation_micro_step == self._gradient_accumulation_steps

            if self._accelerator is None:
                if self._accumulation_micro_step == 1:
                    self.optimizer.zero_grad(set_to_none=True)
                if self._batch_partition_plan is None:
                    (backward_loss / self._gradient_accumulation_steps).backward()
                else:
                    backward_loss.backward()
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
                self._accelerator.backward(backward_loss)
                sync_gradients = bool(self._accelerator.sync_gradients)
                if sync_gradients:
                    self._accelerator.unscale_gradients(self.optimizer)
                    if not self._gradients_are_finite(self.model):
                        raise FloatingPointError(
                            f"non-finite gradient before optimizer step {self.global_step + 1}"
                        )
                self.optimizer.step()
                if sync_gradients:
                    if bool(getattr(self.optimizer, "step_was_skipped", False)):
                        raise FloatingPointError(
                            f"optimizer step {self.global_step + 1} was skipped"
                        )
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
        if self._accelerator is not None:
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
        flop_estimate = 6.0 * self._full_parameter_count * base_supervised_tokens
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

    def _allocation_neutral_public_metric_state(
        self,
        *,
        started: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Collect one global public row and explicit rank-local diagnostics."""

        plan = self._batch_partition_plan
        if plan is None or self._parameters_before_update is None:
            raise RuntimeError("allocation-neutral public metrics lack an optimizer window")
        rank, world_size = self._distributed_position()
        if rank != plan.rank or world_size != plan.world_size:
            raise RuntimeError("allocation-neutral public metric topology changed")
        local_peak_memory = float(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        local_wall_clock = time.perf_counter() - started
        local_summary: dict[str, Any] = {
            "rank": rank,
            "local_sequence_count": plan.local_sequence_count,
            "collection": dict(self._pending_collection),
            "cumulative": dict(self.cumulative_counts),
            "local_supervised_tokens": self._pending_processed_tokens,
            "local_input_tokens": self._pending_input_tokens,
            "global_input_tokens": self._current_global_update_tokens,
            "parameter_update_squared_norm": _parameter_update_squared_norm(
                self._parameters_before_update,
                self.model.parameters(),
            ),
            "peak_allocated_gpu_memory": local_peak_memory,
            "wall_clock_seconds": local_wall_clock,
        }
        rows: list[object] = [local_summary]
        if world_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError(
                    "allocation-neutral public metric aggregation requires a process group"
                )
            rows = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(rows, local_summary)
        aggregated = _allocation_neutral_global_public_metrics(
            rows,
            world_size=world_size,
            global_batch_size=plan.global_batch_size,
            global_step=self.global_step,
        )
        local_diagnostics = {
            "rank": rank,
            "world_size": world_size,
            "sequence_count_this_update": plan.local_sequence_count,
            "collection_metrics": dict(self._pending_collection),
            "cumulative_counts": dict(self.cumulative_counts),
            "effective_supervised_tokens_this_update": self._pending_processed_tokens,
            "model_facing_input_tokens_this_update": self._pending_input_tokens,
            "parameter_update_norm": math.sqrt(
                float(local_summary["parameter_update_squared_norm"])
            ),
            "peak_allocated_gpu_memory": local_peak_memory,
            "wall_clock_seconds": local_wall_clock,
        }
        return aggregated, local_diagnostics

    def _finish_optimizer_update(
        self,
        *,
        started: float,
    ) -> dict[str, Any]:
        self.global_step += 1
        self.online_rollout_round = int(getattr(self.state_source, "policy_version", 0))
        self._validate_live_optimizer_scheduler_cadence()
        objective_metrics = self._objective_metrics_for_optimizer_update()
        if self._parameters_before_update is None:
            raise RuntimeError("optimizer update has no parameter snapshot")
        self.cumulative_counts["model_facing_input_tokens_processed"] = float(self.token_budget.consumed)
        local_diagnostics: dict[str, Any] | None = None
        if self._batch_partition_plan is None:
            generated = self._pending_collection["generated_trajectories"]
            successful = self._pending_collection["successful_trajectories"]
            collection_metrics = {
                **self._pending_collection,
                "reward_rate": successful / generated if generated else 0.0,
            }
            public_cumulative_counts = dict(self.cumulative_counts)
            effective_supervised_tokens = self._pending_processed_tokens
            update_norm = parameter_update_norm(
                self._parameters_before_update,
                self.model.parameters(),
            )
            peak_memory = float(
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            wall_clock = time.perf_counter() - started
        else:
            aggregated, local_diagnostics = self._allocation_neutral_public_metric_state(
                started=started
            )
            collection_metrics = dict(aggregated["collection_metrics"])
            public_cumulative_counts = dict(aggregated["cumulative_counts"])
            effective_supervised_tokens = float(
                aggregated["effective_supervised_tokens_this_update"]
            )
            update_norm = float(aggregated["parameter_update_norm"])
            peak_memory = float(aggregated["peak_allocated_gpu_memory"])
            wall_clock = float(aggregated["wall_clock_seconds"])
        metric: dict[str, Any] = {
            **objective_metrics,
            **collection_metrics,
            "step": float(self.global_step),
            **public_cumulative_counts,
            "effective_supervised_tokens_this_update": effective_supervised_tokens,
            "local_model_facing_input_tokens_this_update": self._pending_input_tokens,
            "model_facing_input_tokens_this_update": self._current_global_update_tokens,
            "token_budget": self.token_budget.budget,
            "token_budget_unit": self.token_budget.unit,
            "token_budget_consumed": self.token_budget.consumed,
            "token_budget_remaining": self.token_budget.remaining,
            "optimizer_updates": float(self.global_step),
            "parameter_update_norm": update_norm,
            "output_kl_from_initial": self._output_kl(),
            "wall_clock_seconds": wall_clock,
            "peak_allocated_gpu_memory": peak_memory,
            **self._evaluation_metrics(),
        }
        if local_diagnostics is not None:
            metric["local_rank_diagnostics"] = local_diagnostics
        self._pending_collection = {key: 0.0 for key in _COLLECTION_METRICS}
        self._pending_processed_tokens = 0.0
        self._pending_input_tokens = 0
        self._current_global_update_tokens = 0
        self._pending_metric_sums = {}
        self._pending_metric_calls = 0
        self._pending_metric_sequence_count = 0
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
            plan = self._batch_partition_plan
            rejected_window_snapshot: dict[str, Any] | None = None
            if plan is not None:
                # ``steps_per_round=1`` is part of the protocol, so an exact
                # optimizer window must never carry uncheckpointed prefetch
                # from its predecessor.  This makes the snapshot below an
                # optimizer-boundary transaction on every rank.
                if queued_microbatches:
                    raise RuntimeError(
                        "allocation-neutral training cannot retain a prefetched microbatch"
                    )
                rejected_window_snapshot = self._capture_rejected_window_snapshot()
            while len(queued_microbatches) < self._gradient_accumulation_steps:
                prompts = self.prompt_scheduler.next_batch()
                trajectories, supervision, attempts = self._collect_trajectories(prompts)
                self._register_collection(prompts, trajectories, supervision, attempts)
                queued_microbatches.extend([supervision] * self.config.steps_per_round)
            window = queued_microbatches[: self._gradient_accumulation_steps]
            del queued_microbatches[: self._gradient_accumulation_steps]
            if plan is not None:
                expected_sizes = plan.microbatch_sizes
                observed_sizes = tuple(
                    self._supervision_sequence_count(batch) for batch in window
                )
                if observed_sizes != expected_sizes:
                    raise ValueError(
                        "allocation-neutral optimizer window differs from the reviewed microbatch plan"
                    )
            local_window_tokens = sum(int(batch.attention_mask.sum().item()) for batch in window)
            admitted, global_window_tokens = self.token_budget.reserve_optimizer_update(local_window_tokens)
            if not admitted:
                # ``reserve_optimizer_update`` reaches the same decision on
                # every rank.  Roll back only collection/runtime state, while
                # retaining its terminal budget reason, so the boundary
                # checkpoint cannot skip rejected logical slots on resume.
                if rejected_window_snapshot is not None:
                    self._restore_rejected_window_snapshot(rejected_window_snapshot)
                    queued_microbatches.clear()
                break
            self._current_global_update_tokens = global_window_tokens
            for index, supervision in enumerate(window):
                expected_sequence_count = (
                    plan.microbatch_sizes[index] if plan is not None else None
                )
                sync_gradients = self._training_micro_step(
                    supervision,
                    expected_sequence_count=expected_sequence_count,
                )
                if not sync_gradients:
                    continue
                metric = self._finish_optimizer_update(started=started)
                history.append(metric)
                if self.is_main_process:
                    append_metric(self.run_dir / "metrics.jsonl", metric)
                self._bind_known_terminal_stop_reason()
                if self.global_step % self.config.checkpoint_every == 0 or self._terminate:
                    self.save(self.run_dir / "checkpoints" / f"step-{self.global_step:08d}.pt")
                break
        if self.token_budget.stop_reason is None:
            self.token_budget.stop_reason = (
                "max_steps_safety_limit"
                if self.global_step >= self.config.max_steps
                else "signal_at_optimizer_boundary"
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
            self._publish_final_checkpoint()
        return history

    def save(self, path: Path, *, _allow_terminal_revision: bool = False) -> str:
        if self._accumulation_micro_step != 0:
            raise RuntimeError("checkpoint requested inside a partial gradient-accumulation window")
        self._validate_live_optimizer_scheduler_cadence()
        path = Path(path).absolute()
        reused = self._reuse_current_checkpoint_if_valid(
            path,
            allow_terminal_revision=_allow_terminal_revision,
        )
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
            full_model_state = self._full_model_state_for_checkpoint()
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
            trainer_states = (
                self._gather_exact_trainer_states()
                if self._batch_partition_plan is not None
                else None
            )
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
                        "trainer_state": (
                            trainer_states[0]
                            if trainer_states is not None
                            else self._trainer_runtime_state()
                        ),
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
                    if trainer_states is not None:
                        payload["trainer_state_by_rank"] = trainer_states
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
            self._last_checkpoint_token_budget_state = copy.deepcopy(
                self.token_budget.state_dict()
            )
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
                    trainer_state=self._trainer_runtime_state(),
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
        self._last_checkpoint_token_budget_state = copy.deepcopy(
            self.token_budget.state_dict()
        )
        self._last_accelerator_state_files = None
        self._last_accelerator_state_sha256 = None
        return checkpoint_sha256

    def resume(self, path: Path) -> None:
        """Resume one explicit, same-world invocation from an optimizer boundary.

        This is distinct from a scheduler fresh attempt: the handler starts a
        fresh attempt from frozen inputs without ``--resume``.  When a caller
        explicitly supplies a checkpoint, every cross-world case must fail
        before model, optimizer, scheduler, RNG, or rank-local source state is
        loaded.
        """

        supplied_path = Path(path)
        if self._batch_partition_plan is not None:
            if (
                not supplied_path.is_absolute()
                or ".." in supplied_path.parts
                or supplied_path.is_symlink()
            ):
                raise ValueError(
                    "allocation-neutral resume checkpoint path is not canonical"
                )
            try:
                resolved_path = supplied_path.resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError(
                    "allocation-neutral resume checkpoint is missing"
                ) from error
            if resolved_path != supplied_path or not resolved_path.is_file():
                raise ValueError(
                    "allocation-neutral resume checkpoint is not a regular canonical file"
                )
            path = resolved_path
        else:
            path = supplied_path.absolute()
        if self._accelerator is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping):
                raise ValueError("Accelerate resume checkpoint payload is not a mapping")
            if payload.get("format") != "accelerate_fsdp_full_export_v1":
                raise ValueError("Accelerate resume requires an Accelerate/FSDP checkpoint export")
            if payload.get("manifest_hashes") != self.manifest_hashes:
                raise ValueError("resume checkpoint ExperimentBinding/design hashes changed")
            rank = int(getattr(self._accelerator, "process_index", 0))
            world_size = int(getattr(self._accelerator, "num_processes", 1))
            # This check intentionally precedes every ``load_state`` call: a
            # scheduler fresh attempt may receive a different allocation, and
            # it must fail before FSDP/model/optimizer state can be mutated.
            _validate_resume_world_size(
                payload.get("world_size"),
                running_world_size=world_size,
            )
            preflight = self._preflight_resume_runtime_metadata(
                payload,
                checkpoint_path=path,
                rank=rank,
                world_size=world_size,
            )
            state_dir_value = payload.get("accelerate_state_dir")
            expected_state_dir = path.with_suffix(".accelerate")
            if (
                not isinstance(state_dir_value, str)
                or state_dir_value != str(expected_state_dir)
                or Path(state_dir_value) != expected_state_dir
            ):
                raise ValueError(
                    "Accelerate resume state directory differs from its checkpoint sibling"
                )
            state_dir = expected_state_dir
            state_files = payload.get("accelerate_state_files")
            if not isinstance(state_files, dict):
                raise ValueError("Accelerate resume checkpoint lacks state file hashes")
            validate_accelerator_state_directory(
                state_dir,
                expected_run_root=path.resolve().parent.parent,
                expected_files=state_files,
                expected_sha256=str(payload.get("accelerate_state_sha256", "")),
            )
            self._accelerator.load_state(str(state_dir))
            self._accelerator.wait_for_everyone()
            self.global_step = preflight.global_step
            self._validate_live_optimizer_scheduler_cadence()
            self.prompt_scheduler.load_state_dict(preflight.prompt_states[rank])
            self.state_source.load_state_dict(preflight.source_states[rank])
            self.online_rollout_round = preflight.online_rollout_round
            self.cumulative_counts = dict(preflight.cumulative_counts)
            self._accumulation_micro_step = 0
            self.token_budget.load_state_dict(preflight.token_budget_state)
            self.resume_ancestry = preflight.resume_ancestry
            self._update_norm_baseline_path = path
            self._update_norm_baseline_sha256 = sha256_file(path)
            self._update_norm_baseline_model = None
            return
        # ``load_checkpoint`` mutates model, optimizer, scheduler, and RNG.
        # Preflight the serialized metadata first so a cross-world resume is
        # rejected before any of those mutable states are touched.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("resume checkpoint payload is not a mapping")
        if payload.get("manifest_hashes") != self.manifest_hashes:
            raise ValueError("checkpoint scientific manifest hashes differ from the resumed run")
        rank, world_size = self._distributed_position()
        _validate_resume_world_size(
            payload.get("world_size"),
            running_world_size=world_size,
        )
        preflight = self._preflight_resume_runtime_metadata(
            payload,
            checkpoint_path=path,
            rank=rank,
            world_size=world_size,
        )
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_manifest_hashes=self.manifest_hashes,
        )
        self.global_step = preflight.global_step
        self._validate_live_optimizer_scheduler_cadence()
        self.prompt_scheduler.load_state_dict(preflight.prompt_states[rank])
        self.state_source.load_state_dict(preflight.source_states[rank])
        self.online_rollout_round = preflight.online_rollout_round
        self.cumulative_counts = dict(preflight.cumulative_counts)
        self._accumulation_micro_step = 0
        self.token_budget.load_state_dict(preflight.token_budget_state)
        scaler = getattr(self._accelerator, "scaler", None)
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        self.resume_ancestry = preflight.resume_ancestry
        self._update_norm_baseline_path = path
        self._update_norm_baseline_sha256 = sha256_file(path)
        self._update_norm_baseline_model = None
