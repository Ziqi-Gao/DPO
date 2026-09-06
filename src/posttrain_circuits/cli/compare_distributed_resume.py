"""Validate two independent resumes against one uninterrupted reference run."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.checkpoints import checkpoint_runtime_state_hashes
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.artifacts.runs import formal_artifact_binding
from posttrain_circuits.cli.factorial_run_validation import (
    _CHECKPOINT_FIELDS,
    FactorialRunArtifacts,
    strict_json_object,
    validate_factorial_run_artifacts,
)
from posttrain_circuits.cli.finalize_pilot_training import (
    _require_confined_regular_file,
    _validate_checkpoint_scientific_binding,
    _validate_factorial_accelerator_state,
    _validate_factorial_rank_states,
)
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.learning.training.fsdp_contract import (
    REQUESTED_FSDP_SHARDING_STRATEGY,
    effective_fsdp_sharding_strategy,
)
from posttrain_circuits.learning.training.factorial_trainer import (
    _validate_optimizer_scheduler_cadence,
)
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
)


_SUPPORTED_WORLD_SIZES = (1, 2, 3, 4)
FIXED_DISTRIBUTED_RESUME_TOLERANCE = 1e-6
_CORE_RUNTIME_STATE_NAMES = {
    "manifest_hashes",
    "model",
    "optimizer",
    "prompt_scheduler_by_rank",
    "rank_shard_hashes",
    "rng",
    "scaler",
    "scheduler",
    "state_source_by_rank",
    "token_budget",
    "trainer_state",
    "trainer_state_by_rank",
}


def _logical_workspace_path(path: Path, *, workspace: Path, name: str) -> str:
    """Return a canonical path-free name for a validated workspace artifact."""

    try:
        relative = Path(path).relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"{name} is outside the distributed resume workspace") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name} has no canonical workspace-relative name")
    return relative.as_posix()


def _source_workspace(config: Mapping[str, Any]) -> Path:
    raw = config.get("output_root")
    if not isinstance(raw, str) or not raw:
        raise ValueError("distributed resume config lacks its source output_root")
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts or str(path) != raw:
        raise ValueError("distributed resume source output_root is not canonical")
    return path


_RESUME_SOURCE_FIELDS = {
    "accelerator_state",
    "batch_partition",
    "checkpoint",
    "config_binding_sha256",
    "experiment_binding_sha256",
    "git_commit",
    "global_step",
    "implementation_dirty",
    "manifest_hashes",
    "rank_shard_hashes",
    "resolved_config_sha256",
    "resume_ancestry",
    "runtime_state_hashes",
    "sha256",
    "token_budget",
    "world_size",
}


def _require_fixed_tolerance(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != FIXED_DISTRIBUTED_RESUME_TOLERANCE
    ):
        raise ValueError(
            "distributed resume tolerance must equal the reviewed fixed value 1e-6"
        )
    return FIXED_DISTRIBUTED_RESUME_TOLERANCE


def _last_metric(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"resume metrics are empty: {path}")
    selected = [row for row in rows if any(key.endswith("_loss") for key in row)]
    if not selected:
        raise ValueError(f"resume metrics have no optimizer-update loss: {path}")
    return selected[-1]


def _objective_loss(row: dict[str, object]) -> float:
    values = [
        float(value)
        for key, value in row.items()
        if key.endswith("_loss")
        and type(value) in (int, float)
        and math.isfinite(float(value))
    ]
    if len(values) != 1:
        raise ValueError(f"expected exactly one objective loss metric, observed {len(values)}")
    return values[0]


def _metric_step(row: Mapping[str, object]) -> int:
    value = row.get("step")
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) < 0
        or not float(value).is_integer()
    ):
        raise ValueError("resume metric has an invalid optimizer step")
    return int(value)


def _validate_exact_checkpoint(
    payload: object,
    *,
    name: str,
    world_size: int,
    max_steps: int,
    max_model_input_length: int = 1536,
) -> dict[str, Any]:
    """Small in-memory exact-state validator retained for fixtures and resume preflight."""

    if not isinstance(payload, dict):
        raise ValueError(f"{name} resume checkpoint is not a mapping")
    if payload.get("world_size") != world_size:
        raise ValueError(f"{name} resume checkpoint belongs to a different world size")
    trainer_state = payload.get("trainer_state")
    partition = (
        trainer_state.get("batch_partition")
        if isinstance(trainer_state, Mapping)
        else None
    )
    if (
        not isinstance(partition, Mapping)
        or partition.get("protocol") != ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1
    ):
        raise ValueError(f"{name} resume checkpoint lacks the exact-global batch protocol")
    _validate_factorial_rank_states(
        payload,
        expected_batch_partition_protocol=ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
        expected_max_steps=max_steps,
        expected_max_model_input_length=max_model_input_length,
    )
    _validate_optimizer_scheduler_cadence(
        optimizer_state=payload.get("optimizer"),
        scheduler_state=payload.get("scheduler"),
        global_step=payload.get("global_step"),
    )
    return payload


def _resume_source(
    path: Path,
    *,
    workspace: Path,
) -> tuple[Path, str]:
    expected = workspace / "calibration" / "checkpoints" / "step-00000020.pt"
    source = _require_confined_regular_file(
        path,
        root=workspace / "calibration" / "checkpoints",
        name="distributed resume source checkpoint",
    )
    if source != expected:
        raise ValueError("distributed resume source is not the reviewed step-20 checkpoint")
    return source, sha256_file(source)


def _validated_runs(
    *,
    workspace: Path,
    config: dict[str, Any],
    world_size: int,
    code_commit: str,
    resume_source_sha256: str,
) -> tuple[FactorialRunArtifacts, FactorialRunArtifacts, FactorialRunArtifacts]:
    reference = validate_factorial_run_artifacts(
        workspace / "calibration",
        expected_world_size=world_size,
        expected_resolved_config=config,
        expected_code_commit=code_commit,
        expected_resume_ancestry=(),
    )
    ancestry = (f"sha256:{resume_source_sha256}",)
    resume_source = workspace / "calibration" / "checkpoints" / "step-00000020.pt"
    left = validate_factorial_run_artifacts(
        workspace / "resume-a",
        expected_world_size=world_size,
        expected_resolved_config=config,
        expected_code_commit=code_commit,
        expected_resume_ancestry=ancestry,
        expected_update_norm_baseline_checkpoint_path=resume_source,
    )
    right = validate_factorial_run_artifacts(
        workspace / "resume-b",
        expected_world_size=world_size,
        expected_resolved_config=config,
        expected_code_commit=code_commit,
        expected_resume_ancestry=ancestry,
        expected_update_norm_baseline_checkpoint_path=resume_source,
    )
    return reference, left, right


def _validate_resume_source_artifacts(
    source: Path,
    *,
    source_sha256: str,
    reference: FactorialRunArtifacts,
    config: dict[str, Any],
    world_size: int,
) -> dict[str, Any]:
    """Validate the exact step-20 checkpoint and its sibling Accelerate tree."""

    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_FIELDS:
        raise ValueError("distributed resume source checkpoint fields differ from the exact format")
    if sha256_file(source) != source_sha256:
        raise ValueError("distributed resume source checkpoint bytes changed")
    if payload.get("format") != "accelerate_fsdp_full_export_v1":
        raise ValueError("distributed resume source is not an Accelerate/FSDP checkpoint")
    if payload.get("global_step") != 20 or type(payload.get("global_step")) is not int:
        raise ValueError("distributed resume source is not the semantic step-20 checkpoint")
    if payload.get("resolved_config") != config:
        raise ValueError("distributed resume source resolved config differs from the active G0 run")
    if (
        payload.get("git_commit") != reference.manifest.get("git_commit")
        or payload.get("implementation_dirty") is not False
    ):
        raise ValueError("distributed resume source implementation binding is invalid")
    if payload.get("resume_ancestry") != []:
        raise ValueError("distributed resume source must come from the uninterrupted reference run")
    _validate_checkpoint_scientific_binding(
        source,
        binding=reference.experiment_binding,
    )
    _validate_exact_checkpoint(
        payload,
        name="step-20 source",
        world_size=world_size,
        max_steps=120,
        max_model_input_length=1536,
    )
    actual_workspace = reference.root.parent
    bound_workspace = _source_workspace(config)
    path_relocation = (
        None
        if bound_workspace == actual_workspace
        else (bound_workspace, actual_workspace)
    )
    accelerator_files = _validate_factorial_accelerator_state(
        reference.root,
        checkpoint_path=source,
        checkpoint_payload=payload,
        resolved_config=config,
        path_relocation=path_relocation,
    )
    if accelerator_files != payload.get("accelerate_state_files"):
        raise ValueError("distributed resume source Accelerator state inventory changed")
    accelerator_path = source.with_suffix(".accelerate")
    bound_accelerator_path = (
        bound_workspace
        / Path(accelerator_path).relative_to(actual_workspace)
    )
    if Path(str(payload.get("accelerate_state_dir", ""))) != bound_accelerator_path:
        raise ValueError("distributed resume source Accelerator state path changed")
    binding: dict[str, Any] = {
        "accelerator_state": {
            "files": payload["accelerate_state_files"],
            "logical_path": _logical_workspace_path(
                accelerator_path,
                workspace=actual_workspace,
                name="distributed resume source Accelerator state",
            ),
            "sha256": payload["accelerate_state_sha256"],
        },
        "batch_partition": payload["trainer_state"]["batch_partition"],
        "checkpoint": {
            "format": payload["format"],
            "logical_path": _logical_workspace_path(
                source,
                workspace=actual_workspace,
                name="distributed resume source checkpoint",
            ),
            "sha256": source_sha256,
        },
        "config_binding_sha256": sha256_file(reference.config_binding_path),
        "experiment_binding_sha256": reference.experiment_binding.scientific_sha256,
        "git_commit": payload["git_commit"],
        "global_step": payload["global_step"],
        "implementation_dirty": payload["implementation_dirty"],
        "manifest_hashes": payload["manifest_hashes"],
        "rank_shard_hashes": payload["rank_shard_hashes"],
        "resolved_config_sha256": reference.manifest["resolved_config_sha256"],
        "resume_ancestry": payload["resume_ancestry"],
        "runtime_state_hashes": checkpoint_runtime_state_hashes(payload),
        "token_budget": payload["token_budget"],
        "world_size": payload["world_size"],
    }
    binding["sha256"] = sha256_value(binding)
    return binding


def _build_comparison_payload(
    *,
    config: dict[str, Any],
    world_size: int,
    tolerance: float,
    workspace: Path,
    resume_source: Path,
    resume_source_sha256: str,
    resume_source_binding: Mapping[str, Any],
    formal_binding: Mapping[str, Any],
    reference: FactorialRunArtifacts,
    left: FactorialRunArtifacts,
    right: FactorialRunArtifacts,
) -> dict[str, Any]:
    tolerance = _require_fixed_tolerance(tolerance)
    source_checkpoint_binding = (
        resume_source_binding.get("checkpoint")
        if isinstance(resume_source_binding, Mapping)
        else None
    )
    if (
        not isinstance(resume_source_binding, Mapping)
        or set(resume_source_binding) != _RESUME_SOURCE_FIELDS
        or not isinstance(source_checkpoint_binding, Mapping)
        or resume_source_binding.get("sha256")
        != sha256_value(
            {
                key: value
                for key, value in resume_source_binding.items()
                if key != "sha256"
            }
        )
        or source_checkpoint_binding.get("logical_path")
        != _logical_workspace_path(
            resume_source,
            workspace=workspace,
            name="distributed resume source checkpoint",
        )
        or source_checkpoint_binding.get("sha256") != resume_source_sha256
    ):
        raise ValueError("distributed resume source binding is incomplete or inconsistent")
    trainer_config = config.get("trainer")
    max_steps = trainer_config.get("max_steps") if isinstance(trainer_config, dict) else None
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("distributed resume comparison requires a positive trainer.max_steps")

    reference_metric = _last_metric(reference.metrics_path)
    left_metric = _last_metric(left.metrics_path)
    right_metric = _last_metric(right.metrics_path)
    reference_loss = _objective_loss(reference_metric)
    left_loss = _objective_loss(left_metric)
    right_loss = _objective_loss(right_metric)
    loss_errors = {
        "left_vs_reference": abs(left_loss - reference_loss),
        "right_vs_reference": abs(right_loss - reference_loss),
        "left_vs_right": abs(left_loss - right_loss),
    }
    artifacts = {
        "reference": reference,
        "left": left,
        "right": right,
    }
    payloads = {name: run.checkpoint_payload for name, run in artifacts.items()}
    bindings = {name: run.content_binding() for name, run in artifacts.items()}
    fsdp_contracts = {
        name: binding["fsdp_sharding_contract"] for name, binding in bindings.items()
    }
    reference_fsdp_contract = fsdp_contracts["reference"]
    expected_fsdp_strategy = {
        "requested_fsdp_sharding_strategy": REQUESTED_FSDP_SHARDING_STRATEGY,
        "effective_fsdp_sharding_strategy": effective_fsdp_sharding_strategy(
            world_size
        ),
    }
    runtime_hashes = {
        name: binding["runtime_state_hashes"] for name, binding in bindings.items()
    }
    if any(not _CORE_RUNTIME_STATE_NAMES.issubset(hashes) for hashes in runtime_hashes.values()):
        raise ValueError("distributed resume runtime-state hash inventory is incomplete")
    reference_core = {
        name: runtime_hashes["reference"][name]
        for name in sorted(_CORE_RUNTIME_STATE_NAMES)
    }
    left_core = {
        name: runtime_hashes["left"][name] for name in sorted(_CORE_RUNTIME_STATE_NAMES)
    }
    right_core = {
        name: runtime_hashes["right"][name] for name in sorted(_CORE_RUNTIME_STATE_NAMES)
    }
    expected_ancestry = [f"sha256:{resume_source_sha256}"]
    global_step = payloads["reference"]["global_step"]
    checks = {
        "world_size_supported": world_size in _SUPPORTED_WORLD_SIZES,
        "checkpoint_world_size_matches": all(
            payload["world_size"] == world_size for payload in payloads.values()
        ),
        "optimizer_step_identical": all(
            payload["global_step"] == global_step for payload in payloads.values()
        ),
        "optimizer_step_within_limit": 0 < global_step <= max_steps,
        "metric_steps_match_checkpoints": (
            _metric_step(reference_metric) == global_step
            and _metric_step(left_metric) == payloads["left"]["global_step"]
            and _metric_step(right_metric) == payloads["right"]["global_step"]
        ),
        "resume_ancestry_exact": (
            payloads["reference"]["resume_ancestry"] == []
            and payloads["left"]["resume_ancestry"] == expected_ancestry
            and payloads["right"]["resume_ancestry"] == expected_ancestry
            and payloads["left"]["update_norm_baseline_checkpoint_sha256"]
            == resume_source_sha256
            and payloads["right"]["update_norm_baseline_checkpoint_sha256"]
            == resume_source_sha256
        ),
        "core_runtime_state_matches_uninterrupted_reference": (
            reference_core == left_core == right_core
        ),
        "checkpoint_format_identical": all(
            payload["format"] == "accelerate_fsdp_full_export_v1"
            for payload in payloads.values()
        ),
        "fsdp_sharding_contract_identical": (
            fsdp_contracts["reference"]
            == fsdp_contracts["left"]
            == fsdp_contracts["right"]
            and {
                key: reference_fsdp_contract.get(key)
                for key in expected_fsdp_strategy
            }
            == expected_fsdp_strategy
            and type(reference_fsdp_contract.get("fsdp_wrapper_count")) is int
            and reference_fsdp_contract["fsdp_wrapper_count"] > 0
        ),
        "optimizer_state_key_type_identical": (
            payloads["reference"]["optimizer_state_key_type"]
            == payloads["left"]["optimizer_state_key_type"]
            == payloads["right"]["optimizer_state_key_type"]
        ),
        "policy_and_rollout_state_identical": all(
            payload["policy_version"] == 0 and payload["online_rollout_round"] == 0
            for payload in payloads.values()
        ),
        "token_budget_state_identical": (
            payloads["reference"]["token_budget"]
            == payloads["left"]["token_budget"]
            == payloads["right"]["token_budget"]
        ),
        "terminal_reason_identical": (
            reference.manifest["training_stop_reason"]
            == left.manifest["training_stop_reason"]
            == right.manifest["training_stop_reason"]
        ),
        "next_loss_within_tolerance": all(
            error <= tolerance for error in loss_errors.values()
        ),
        "input_bindings_self_hashed": all(
            binding["sha256"]
            == sha256_value({key: value for key, value in binding.items() if key != "sha256"})
            for binding in bindings.values()
        ),
        "run_roots_are_distinct": len({run.root for run in artifacts.values()}) == 3,
    }
    content: dict[str, Any] = {
        "format_version": 3,
        "passed": all(checks.values()),
        "world_size": world_size,
        "global_step": global_step,
        "fsdp_sharding_contract": reference_fsdp_contract,
        "max_optimizer_steps": max_steps,
        "checks": checks,
        "next_loss_absolute_errors": loss_errors,
        "tolerance": FIXED_DISTRIBUTED_RESUME_TOLERANCE,
        "resume_source": dict(resume_source_binding),
        "runs": bindings,
        **dict(formal_binding),
    }
    return content


def validate_distributed_resume_report(
    path: Path,
    *,
    config: dict[str, Any],
    expected_world_size: int,
    expected_formal_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a resume comparison from bound run artifacts and compare byte-for-byte."""

    path = Path(path).absolute()
    workspace = path.parent
    report_path = _require_confined_regular_file(
        path,
        root=workspace,
        name="distributed resume report",
    )
    observed = strict_json_object(report_path, name="distributed resume report")
    digest = observed.get("sha256")
    if digest != sha256_value({key: value for key, value in observed.items() if key != "sha256"}):
        raise ValueError("distributed resume report SHA-256 mismatch")
    if observed.get("world_size") != expected_world_size:
        raise ValueError("distributed resume report belongs to a different world size")
    source_field = observed.get("resume_source")
    if not isinstance(source_field, dict) or set(source_field) != _RESUME_SOURCE_FIELDS:
        raise ValueError("distributed resume report lacks an exact source binding")
    source_checkpoint = source_field.get("checkpoint")
    if not isinstance(source_checkpoint, dict) or set(source_checkpoint) != {
        "format",
        "logical_path",
        "sha256",
    }:
        raise ValueError("distributed resume report source checkpoint binding changed")
    logical_source = source_checkpoint.get("logical_path")
    if logical_source != "calibration/checkpoints/step-00000020.pt":
        raise ValueError("distributed resume report source checkpoint name changed")
    source, source_sha256 = _resume_source(
        workspace / str(logical_source),
        workspace=workspace,
    )
    if source_checkpoint.get("sha256") != source_sha256:
        raise ValueError("distributed resume source bytes changed after comparison")
    formal_binding = (
        dict(expected_formal_binding)
        if expected_formal_binding is not None
        else formal_artifact_binding(config)
    )
    reference, left, right = _validated_runs(
        workspace=workspace,
        config=config,
        world_size=expected_world_size,
        code_commit=str(formal_binding["code_commit"]),
        resume_source_sha256=source_sha256,
    )
    source_binding = _validate_resume_source_artifacts(
        source,
        source_sha256=source_sha256,
        reference=reference,
        config=config,
        world_size=expected_world_size,
    )
    tolerance = _require_fixed_tolerance(observed.get("tolerance"))
    expected = _build_comparison_payload(
        config=config,
        world_size=expected_world_size,
        tolerance=float(tolerance),
        workspace=workspace,
        resume_source=source,
        resume_source_sha256=source_sha256,
        resume_source_binding=source_binding,
        formal_binding=formal_binding,
        reference=reference,
        left=left,
        right=right,
    )
    expected["sha256"] = sha256_value(expected)
    if observed != expected:
        raise ValueError("distributed resume report differs from revalidated run artifacts")
    if observed.get("passed") is not True or any(
        value is not True for value in observed["checks"].values()
    ):
        raise ValueError("distributed resume comparison did not pass every exact check")
    return observed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare two independent distributed resumes with an uninterrupted run"
    )
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--resume-source", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=FIXED_DISTRIBUTED_RESUME_TOLERANCE,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError("distributed resume comparison supports only world sizes 1, 2, 3, and 4")
    tolerance = _require_fixed_tolerance(args.tolerance)
    workspace = args.workspace.absolute()
    if args.output.absolute() != workspace / "distributed_resume.json":
        raise ValueError("distributed resume output differs from the fixed G0 workspace path")
    config = compose_config(args.overrides)
    formal_binding = formal_artifact_binding(config)
    source, source_sha256 = _resume_source(
        args.resume_source,
        workspace=workspace,
    )
    reference, left, right = _validated_runs(
        workspace=workspace,
        config=config,
        world_size=args.world_size,
        code_commit=str(formal_binding["code_commit"]),
        resume_source_sha256=source_sha256,
    )
    source_binding = _validate_resume_source_artifacts(
        source,
        source_sha256=source_sha256,
        reference=reference,
        config=config,
        world_size=args.world_size,
    )
    payload = _build_comparison_payload(
        config=config,
        world_size=args.world_size,
        tolerance=tolerance,
        workspace=workspace,
        resume_source=source,
        resume_source_sha256=source_sha256,
        resume_source_binding=source_binding,
        formal_binding=formal_binding,
        reference=reference,
        left=left,
        right=right,
    )
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)
    if payload["passed"] is not True:
        raise SystemExit("distributed checkpoint/resume comparison failed")


if __name__ == "__main__":
    main()


__all__ = [
    "_validate_exact_checkpoint",
    "validate_distributed_resume_report",
]
