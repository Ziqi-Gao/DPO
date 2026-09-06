"""Strict validation for completed allocation-neutral Factorial runs.

The G0 pipeline consumes training output in several different processes.  This
module gives those consumers one fail-closed validator instead of letting each
consumer trust a different subset of the run manifest, checkpoint, update
evidence, and external Accelerate state tree.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from posttrain_circuits.artifacts.checkpoints import checkpoint_runtime_state_hashes
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.runs import RunManifest, validate_run_manifest_payload
from posttrain_circuits.cli.finalize_pilot_training import (
    _load_run_config_binding,
    _require_confined_regular_file,
    _validate_checkpoint_scientific_binding,
    _validate_factorial_accelerator_state,
    _validate_factorial_rank_states,
    _validate_factorial_update_evidence,
    _validate_resolved_config_yaml_sha256,
)
from posttrain_circuits.datasets.teacher_demos.store import read_teacher_demo_store
from posttrain_circuits.experiments.protocols.specs import (
    ExperimentBinding,
    validate_run_manifest_experiment_binding,
)
from posttrain_circuits.learning.training.schedules import (
    ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
)
from posttrain_circuits.learning.training.factorial_trainer import (
    _validate_optimizer_scheduler_cadence,
)
from posttrain_circuits.learning.training.token_budget import TOKEN_BUDGET_UNIT


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_CHECKPOINT_NAME = re.compile(r"step-([0-9]{8})(?:-terminal)?\.pt\Z")
_SUPPORTED_WORLD_SIZES = (1, 2, 3, 4)
_CHECKPOINT_FORMAT = "accelerate_fsdp_full_export_v1"
_MAX_MODEL_INPUT_LENGTH = 1536
_FINAL_STOP_REASONS = {
    "max_steps_safety_limit",
    "token_budget_exactly_consumed",
    "token_budget_exhausted_before_next_optimizer_update",
}
_CHECKPOINT_FIELDS = {
    "accelerate_state_dir",
    "accelerate_state_files",
    "accelerate_state_sha256",
    "dependency_versions",
    "final_model_state_hash",
    "format",
    "git_commit",
    "global_step",
    "implementation_dirty",
    "manifest_hashes",
    "model",
    "online_rollout_round",
    "optimizer",
    "optimizer_state_key_type",
    "parameter_update_norm",
    "policy_version",
    "prompt_scheduler",
    "prompt_scheduler_by_rank",
    "rank_shard_hashes",
    "resolved_config",
    "resume_ancestry",
    "rng",
    "scaler",
    "scheduler",
    "state_source",
    "state_source_by_rank",
    "token_budget",
    "trainer_state",
    "trainer_state_by_rank",
    "update_norm_baseline_checkpoint_path",
    "update_norm_baseline_checkpoint_sha256",
    "world_size",
}
_EVIDENCE_FIELDS = {
    "checkpoint",
    "checkpoint_runtime_state_hashes",
    "experiment_binding_sha256",
    "factorial_design_sha256",
    "final_checkpoint_sha256",
    "final_model_state_hash",
    "format_version",
    "git_commit",
    "global_step",
    "implementation_dirty",
    "manifest_hashes",
    "method_spec_sha256",
    "metric_update_rows",
    "metric_update_rows_sha256",
    "metrics_sha256",
    "parameter_update_norm",
    "resume_ancestry",
    "sha256",
    "token_budget",
    "update_norm_baseline_checkpoint_path",
    "update_norm_baseline_checkpoint_sha256",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Factorial resolved config keys must be strings")
        if key in mapping:
            raise ValueError(f"Factorial resolved config contains duplicate key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _unique_json_object(
    pairs: list[tuple[str, Any]],
    *,
    name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        result[key] = value
    return result


def strict_json_object(path: Path, *, name: str) -> dict[str, Any]:
    """Read one regular UTF-8 JSON object, rejecting duplicate/non-finite input."""

    if path.is_symlink() or not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _unique_json_object(pairs, name=name),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_real_directory(path: Path, *, name: str) -> Path:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise ValueError(f"{name} must be an absolute normalized non-symlink directory")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_dir():
        raise ValueError(f"{name} must be an absolute normalized non-symlink directory")
    return resolved


def _source_relative_path(path: Path, *, source_root: Path, name: str) -> Path:
    """Validate an immutable source path and return its safe relative name."""

    path = Path(path)
    source_root = Path(source_root)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or not source_root.is_absolute()
        or ".." in source_root.parts
    ):
        raise ValueError(f"{name} must be an absolute normalized source path")
    try:
        relative = path.relative_to(source_root)
    except ValueError as error:
        raise ValueError(f"{name} is outside its source run directory") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name} has no canonical source-relative name")
    return relative


def _terminal_state(
    *,
    global_step: int,
    max_steps: int,
    token_budget: Mapping[str, Any],
) -> str:
    reason = token_budget.get("stop_reason")
    consumed = token_budget.get("consumed")
    budget = token_budget.get("budget")
    if reason not in _FINAL_STOP_REASONS:
        raise ValueError("completed Factorial run has no accepted scientific stop reason")
    if type(consumed) is not int or type(budget) is not int or not 0 < consumed <= budget:
        raise ValueError("completed Factorial run has an invalid token-budget endpoint")
    if reason == "token_budget_exactly_consumed":
        valid = consumed == budget and global_step <= max_steps
    elif reason == "token_budget_exhausted_before_next_optimizer_update":
        valid = consumed < budget and global_step < max_steps
    else:
        valid = consumed < budget and global_step == max_steps
    if not valid:
        raise ValueError("Factorial token-budget stop reason contradicts its endpoint")
    return str(reason)


def _validate_metrics_endpoint(
    path: Path,
    *,
    global_step: int,
    token_budget: Mapping[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(
                line,
                object_pairs_hook=lambda pairs: _unique_json_object(
                    pairs,
                    name=f"metrics line {line_number}",
                ),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"metrics contains non-finite value {token}")
                ),
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"metrics line {line_number} is not valid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"metrics line {line_number} is not a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError("completed Factorial run has empty metrics")
    terminal = rows[-1]
    if terminal != {
        "event": "training_stop",
        "step": float(global_step),
        "token_budget": token_budget["budget"],
        "token_budget_consumed": token_budget["consumed"],
        "token_budget_remaining": token_budget["budget"] - token_budget["consumed"],
        "token_budget_unit": token_budget["unit"],
        "stop_reason": token_budget["stop_reason"],
    }:
        raise ValueError("Factorial terminal metric differs from its checkpoint state")


@dataclass(frozen=True)
class FactorialRunArtifacts:
    """Validated active-G0 run artifacts plus a compact content binding."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    resolved_config_path: Path
    config_binding_path: Path
    metrics_path: Path
    checkpoint_path: Path
    checkpoint_payload: dict[str, Any]
    evidence_path: Path
    evidence: dict[str, Any]
    experiment_binding: ExperimentBinding
    accelerator_state_path: Path
    source_workspace: Path | None = None

    def _logical_path(self, path: Path) -> str:
        """Return a stable workspace-relative name for one validated artifact."""

        workspace = self.root.parent
        try:
            relative = Path(path).relative_to(workspace)
        except ValueError as error:
            raise ValueError("Factorial artifact is outside its validated workspace") from error
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("Factorial artifact has no canonical workspace-relative name")
        return relative.as_posix()

    def content_binding(self) -> dict[str, Any]:
        """Build a path-free identity for consumers outside the run directory.

        Each run is already validated against its byte-level manifest and
        Accelerate inventory.  This cross-directory binding therefore carries
        logical names plus scientific/content identities, never an ephemeral
        absolute scratch path or a second hash of a path-bearing manifest.
        """

        checkpoint = self.checkpoint_payload
        batch_partition = checkpoint["trainer_state"]["batch_partition"]
        content: dict[str, Any] = {
            "schema_version": 2,
            "accelerator_state": {
                "files": checkpoint["accelerate_state_files"],
                "logical_path": self._logical_path(self.accelerator_state_path),
                "sha256": checkpoint["accelerate_state_sha256"],
            },
            "batch_partition": batch_partition,
            "checkpoint": {
                "format": checkpoint["format"],
                "logical_path": self._logical_path(self.checkpoint_path),
                "sha256": sha256_file(self.checkpoint_path),
            },
            "config_binding": {
                "logical_path": self._logical_path(self.config_binding_path),
                "sha256": sha256_file(self.config_binding_path),
            },
            "experiment_binding_sha256": self.experiment_binding.scientific_sha256,
            "factorial_update_evidence": {
                "logical_path": self._logical_path(self.evidence_path),
                "sha256": sha256_file(self.evidence_path),
                "semantic_sha256": self.evidence["sha256"],
            },
            "final_model_state_hash": checkpoint["final_model_state_hash"],
            "fsdp_sharding_contract": {
                "requested_fsdp_sharding_strategy": batch_partition[
                    "requested_fsdp_sharding_strategy"
                ],
                "effective_fsdp_sharding_strategy": batch_partition[
                    "effective_fsdp_sharding_strategy"
                ],
                "fsdp_wrapper_count": batch_partition["fsdp_wrapper_count"],
            },
            "global_step": checkpoint["global_step"],
            "manifest": {
                "logical_path": self._logical_path(self.manifest_path),
                "sha256": sha256_file(self.manifest_path),
                "semantic_sha256": self.manifest["sha256"],
            },
            "metrics": {
                "logical_path": self._logical_path(self.metrics_path),
                "sha256": sha256_file(self.metrics_path),
            },
            "rank_shard_hashes": checkpoint["rank_shard_hashes"],
            "resume_ancestry": checkpoint["resume_ancestry"],
            "resolved_config": {
                "logical_path": self._logical_path(self.resolved_config_path),
                "sha256": sha256_file(self.resolved_config_path),
                "semantic_sha256": self.manifest["resolved_config_sha256"],
            },
            "run_directory": self._logical_path(self.root),
            "runtime_state_hashes": checkpoint_runtime_state_hashes(checkpoint),
            "token_budget": checkpoint["token_budget"],
            "training_stop_reason": self.manifest["training_stop_reason"],
            "world_size": checkpoint["world_size"],
        }
        content["sha256"] = sha256_value(content)
        return content


def validate_factorial_run_artifacts(
    root: Path,
    *,
    expected_world_size: int,
    expected_resolved_config: Mapping[str, Any],
    expected_checkpoint_path: Path | None = None,
    expected_update_norm_baseline_checkpoint_path: Path | None = None,
    expected_code_commit: str | None = None,
    expected_resume_ancestry: Sequence[str] | None = None,
) -> FactorialRunArtifacts:
    """Validate one completed Candidate-E canonical-SFT run end to end."""

    if expected_world_size not in _SUPPORTED_WORLD_SIZES:
        raise ValueError("expected Factorial world size is outside 1..4")
    if not isinstance(expected_resolved_config, Mapping):
        raise TypeError("expected resolved Factorial config must be a mapping")
    expected_config = dict(expected_resolved_config)
    root = _require_real_directory(root, name="Factorial run root")
    actual_workspace = root.parent
    raw_source_workspace = expected_config.get("output_root")
    if not isinstance(raw_source_workspace, str) or not raw_source_workspace:
        raise ValueError("active G0 config lacks its source output_root")
    source_workspace = Path(raw_source_workspace)
    if (
        not source_workspace.is_absolute()
        or ".." in source_workspace.parts
        or str(source_workspace) != raw_source_workspace
    ):
        raise ValueError("active G0 source output_root is not canonical")
    source_root = source_workspace / root.name
    path_relocation = (
        None
        if source_workspace == actual_workspace
        else (source_workspace, actual_workspace)
    )
    manifest_path = _require_confined_regular_file(
        root / "manifest.json", root=root, name="Factorial run manifest"
    )
    resolved_config_path = _require_confined_regular_file(
        root / "resolved_config.yaml", root=root, name="Factorial resolved config"
    )
    config_binding_path = _require_confined_regular_file(
        root / "config_binding.json", root=root, name="Factorial ConfigBinding"
    )
    metrics_path = _require_confined_regular_file(
        root / "metrics.jsonl", root=root, name="Factorial metrics"
    )
    evidence_path = _require_confined_regular_file(
        root / "factorial_update_evidence.json",
        root=root,
        name="Factorial update evidence",
    )

    manifest = validate_run_manifest_payload(
        strict_json_object(manifest_path, name="Factorial run manifest")
    )
    expected_manifest_fields = {field.name for field in fields(RunManifest)} | {"sha256"}
    if set(manifest) != expected_manifest_fields:
        raise ValueError("Factorial run manifest fields differ from the current schema")
    binding = validate_run_manifest_experiment_binding(manifest)
    try:
        resolved_config = yaml.load(
            resolved_config_path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except yaml.YAMLError as error:
        raise ValueError("Factorial resolved config is not strict YAML") from error
    if not isinstance(resolved_config, dict):
        raise ValueError("Factorial resolved config must be a mapping")
    if resolved_config != expected_config:
        raise ValueError("Factorial resolved config differs from the active G0 invocation")
    strict_json_object(config_binding_path, name="Factorial ConfigBinding")
    _load_run_config_binding(
        root,
        manifest=manifest,
        resolved_config=resolved_config,
        experiment_binding=binding,
    )
    _validate_resolved_config_yaml_sha256(root, manifest=manifest)

    trainer = resolved_config.get("trainer")
    task = resolved_config.get("task")
    experiment = resolved_config.get("experiment")
    state_source = resolved_config.get("state_source")
    supervision = resolved_config.get("supervision")
    if not all(
        isinstance(value, dict)
        for value in (trainer, task, experiment, state_source, supervision)
    ):
        raise ValueError("active G0 resolved config lacks required scientific mappings")
    assert isinstance(trainer, dict)
    assert isinstance(task, dict)
    assert isinstance(experiment, dict)
    assert isinstance(state_source, dict)
    assert isinstance(supervision, dict)
    expected_science = {
        "backend": trainer.get("backend") == "accelerate",
        "batch_protocol": trainer.get("batch_partition_protocol")
        == ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
        "global_batch": trainer.get("global_batch_size") == 64,
        "max_microbatch": trainer.get("max_microbatch_size") == 4,
        "max_model_input_length": trainer.get("max_model_input_length")
        == _MAX_MODEL_INPUT_LENGTH,
        "max_steps": type(trainer.get("max_steps")) is int and trainer["max_steps"] == 120,
        "token_budget": trainer.get("token_budget") == 2_000_000,
        "token_unit": trainer.get("token_budget_unit") == TOKEN_BUDGET_UNIT,
        "prompt_population": task.get("num_examples") == 256,
        "experiment": experiment.get("name") == "canonical_sft",
        "state_source": state_source.get("name") == "teacher_demo",
        "supervision": supervision.get("name") == "canonical_sft"
        and supervision.get("normalization") == "sequence",
        "g0_full_parameter": resolved_config.get("g0", {}).get(
            "full_parameter_training"
        )
        is True,
        "binding_full_parameter": binding.full_parameter_training is True,
        "seed": binding.seed == 42 and manifest.get("seed") == 42,
        "protocol": binding.protocol_track == "qwen3_v2"
        and manifest.get("protocol_track") == "qwen3_v2",
    }
    failures = sorted(name for name, passed in expected_science.items() if not passed)
    if failures:
        raise ValueError(f"active G0 Factorial scientific contract differs: {failures}")
    if expected_code_commit is not None and (
        binding.implementation_commit != expected_code_commit
        or manifest.get("git_commit") != expected_code_commit
    ):
        raise ValueError("Factorial run belongs to a different implementation commit")
    if manifest.get("dirty_working_tree") is not False or binding.implementation_dirty is not False:
        raise ValueError("active G0 Factorial run was produced by a dirty implementation")

    raw_checkpoint = manifest.get("final_checkpoint_path")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError("Factorial run manifest lacks final_checkpoint_path")
    source_checkpoint_path = Path(raw_checkpoint)
    checkpoint_relative = _source_relative_path(
        source_checkpoint_path,
        source_root=source_root / "checkpoints",
        name="Factorial final checkpoint",
    )
    checkpoint_path = _require_confined_regular_file(
        root / "checkpoints" / checkpoint_relative,
        root=root / "checkpoints",
        name="Factorial final checkpoint",
    )
    if expected_checkpoint_path is not None and checkpoint_path != Path(
        expected_checkpoint_path
    ).absolute():
        raise ValueError("Factorial consumer checkpoint differs from the manifest endpoint")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (
        _SHA256.fullmatch(str(manifest.get("final_checkpoint_sha256", ""))) is None
        or manifest.get("final_checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("Factorial final checkpoint bytes differ from the run manifest")
    if manifest.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("Factorial metrics bytes differ from the run manifest")

    checkpoint_payload = _validate_checkpoint_scientific_binding(
        checkpoint_path,
        binding=binding,
    )
    if set(checkpoint_payload) != _CHECKPOINT_FIELDS:
        raise ValueError("active G0 Factorial checkpoint fields differ from the exact format")
    if checkpoint_payload.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError("active G0 Factorial checkpoint format is not Accelerate/FSDP")
    if checkpoint_payload.get("resolved_config") != resolved_config:
        raise ValueError("Factorial checkpoint resolved config differs from its run")
    if checkpoint_payload.get("world_size") != expected_world_size:
        raise ValueError("Factorial checkpoint differs from the scheduler-assigned world size")
    if checkpoint_payload.get("optimizer_state_key_type") not in {
        "parameter_id",
        "parameter_name",
    }:
        raise ValueError("Factorial optimizer-state key identity is invalid")
    if "scaler" not in checkpoint_payload or (
        checkpoint_payload["scaler"] is not None
        and (
            not isinstance(checkpoint_payload["scaler"], Mapping)
            or not checkpoint_payload["scaler"]
        )
    ):
        raise ValueError("Factorial mixed-precision scaler state is invalid")
    if not isinstance(checkpoint_payload.get("dependency_versions"), Mapping) or not checkpoint_payload[
        "dependency_versions"
    ]:
        raise ValueError("Factorial checkpoint dependency versions are missing")
    _validate_optimizer_scheduler_cadence(
        optimizer_state=checkpoint_payload.get("optimizer"),
        scheduler_state=checkpoint_payload.get("scheduler"),
        global_step=checkpoint_payload.get("global_step"),
    )

    bound_store_path = Path(str(state_source.get("store_path", "")))
    expected_bound_store_path = source_workspace / "teacher_demos"
    if bound_store_path != expected_bound_store_path:
        raise ValueError("active G0 teacher-demo store is outside the attempt workspace")
    store_path = actual_workspace / "teacher_demos"
    accepted_attempts, teacher_manifest = read_teacher_demo_store(
        store_path,
        require_formal=True,
    )
    prompt_ids = list(teacher_manifest["ordered_prompt_ids"])
    attempt_ids = [attempt.attempt_id for attempt in accepted_attempts]
    if len(prompt_ids) != 256 or len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("active G0 teacher-demo store lacks exactly 256 unique prompts")
    if (
        teacher_manifest.get("sha256") != binding.teacher_demo_manifest_sha256
        or manifest.get("rollout_bank_hash") != teacher_manifest.get("sha256")
        or checkpoint_payload.get("manifest_hashes", {}).get("state_source")
        != teacher_manifest.get("sha256")
    ):
        raise ValueError("Factorial teacher-demo content binding differs across artifacts")

    _validate_factorial_rank_states(
        checkpoint_payload,
        expected_batch_partition_protocol=ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
        expected_max_steps=int(trainer["max_steps"]),
        expected_max_model_input_length=int(trainer["max_model_input_length"]),
        expected_prompt_ids=prompt_ids,
        expected_attempt_ids=attempt_ids,
    )
    accelerator_files = _validate_factorial_accelerator_state(
        root,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        resolved_config=resolved_config,
        path_relocation=path_relocation,
    )
    if accelerator_files != checkpoint_payload["accelerate_state_files"]:
        raise ValueError("Factorial Accelerator state inventory changed after validation")
    accelerator_state_path = _require_real_directory(
        checkpoint_path.with_suffix(".accelerate"),
        name="Factorial Accelerator state directory",
    )

    evidence = strict_json_object(evidence_path, name="Factorial update evidence")
    if set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("Factorial update-evidence fields differ from the exact format")
    evidence_sha256 = _validate_factorial_update_evidence(
        root,
        manifest=manifest,
        binding=binding,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        resolved_config=resolved_config,
        path_relocation=path_relocation,
    )
    if evidence_sha256 != evidence.get("sha256"):
        raise ValueError("Factorial update-evidence identity changed during validation")

    global_step = checkpoint_payload["global_step"]
    match = _FINAL_CHECKPOINT_NAME.fullmatch(checkpoint_path.name)
    if match is None or int(match.group(1)) != global_step:
        raise ValueError("Factorial final checkpoint name differs from its optimizer step")
    token_budget = checkpoint_payload["token_budget"]
    reason = _terminal_state(
        global_step=global_step,
        max_steps=int(trainer["max_steps"]),
        token_budget=token_budget,
    )
    if (
        manifest.get("token_budget") != token_budget["budget"]
        or manifest.get("token_budget_unit") != token_budget["unit"]
        or manifest.get("token_budget_consumed") != token_budget["consumed"]
        or manifest.get("training_stop_reason") != reason
        or checkpoint_payload["trainer_state"].get("token_budget") != token_budget
        or not isinstance(manifest.get("end_time"), str)
        or not manifest["end_time"]
    ):
        raise ValueError("Factorial terminal state differs across manifest and checkpoint")
    _validate_metrics_endpoint(
        metrics_path,
        global_step=global_step,
        token_budget=token_budget,
    )
    if expected_resume_ancestry is not None and checkpoint_payload.get(
        "resume_ancestry"
    ) != list(expected_resume_ancestry):
        raise ValueError("Factorial run has the wrong checkpoint resume ancestry")
    if expected_update_norm_baseline_checkpoint_path is not None:
        bound_baseline_path = Path(
            str(checkpoint_payload.get("update_norm_baseline_checkpoint_path", ""))
        )
        baseline_relative = _source_relative_path(
            bound_baseline_path,
            source_root=source_workspace,
            name="Factorial update-norm baseline",
        )
        actual_baseline_path = actual_workspace / baseline_relative
        if actual_baseline_path != Path(
            expected_update_norm_baseline_checkpoint_path
        ).absolute():
            raise ValueError("Factorial run has the wrong update-norm baseline path")

    return FactorialRunArtifacts(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        resolved_config_path=resolved_config_path,
        config_binding_path=config_binding_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        evidence_path=evidence_path,
        evidence=evidence,
        experiment_binding=binding,
        accelerator_state_path=accelerator_state_path,
        source_workspace=source_workspace,
    )


__all__ = [
    "FactorialRunArtifacts",
    "strict_json_object",
    "validate_factorial_run_artifacts",
]
