"""Bind completed pilot cell outputs to Slurm terminal evidence."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from posttrain_circuits.artifacts.checkpoints import (
    accelerator_state_file_hashes,
    checkpoint_runtime_state_hashes,
    load_checkpoint_model_state,
    model_update_evidence,
    torch_state_hash,
    validate_accelerator_state_directory,
    validate_native_trainer_checkpoint_files,
)
from posttrain_circuits.artifacts.config_bindings import ConfigBinding
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.artifacts.runs import (
    formal_artifact_binding,
    validate_run_manifest_payload,
)
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.experiments.protocols.specs import (
    CONTROLLED_FACTORIAL,
    ExperimentBinding,
    validate_checkpoint_experiment_binding_payload,
    validate_experiment_binding_resolved_config,
    validate_run_manifest_experiment_binding,
)
from posttrain_circuits.methods.registry import FACTORIAL_METHOD_IDS, PILOT_METHOD_IDS
from posttrain_circuits.methods.specs import REGISTERED_CURRENT_POLICY, TRL_GRPO_BACKEND
from posttrain_circuits.learning.training.token_budget import TOKEN_BUDGET_UNIT


def _load_run_config_binding(
    root: Path,
    *,
    manifest: dict[str, Any],
    resolved_config: dict[str, Any],
    experiment_binding: ExperimentBinding,
) -> ConfigBinding:
    """Load the sole persisted config identity and bind every manifest hash to it."""

    binding_path = root / "config_binding.json"
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run ConfigBinding is not a mapping: {binding_path}")
    config_binding = ConfigBinding.from_dict(payload)
    required = {
        "resolved_config_sha256": config_binding.resolved_config_sha256,
        "scientific_config_sha256": config_binding.scientific_config_sha256,
        "execution_config_sha256": config_binding.execution_config_sha256,
    }
    mismatches = {
        name: {"expected": expected, "observed": manifest.get(name)}
        for name, expected in required.items()
        if manifest.get(name) != expected
    }
    if manifest.get("config_binding") != config_binding.as_dict():
        mismatches["config_binding"] = {
            "expected": config_binding.as_dict(),
            "observed": manifest.get("config_binding"),
        }
    if mismatches:
        raise ValueError(f"run manifest config binding mismatch: {mismatches}")
    validate_experiment_binding_resolved_config(
        experiment_binding,
        resolved_config,
        config_binding,
    )
    return config_binding


def _validate_resolved_config_yaml_sha256(
    root: Path,
    *,
    manifest: dict[str, Any],
) -> str:
    """Validate only the serialized YAML bytes against their dedicated digest."""

    resolved_path = root / "resolved_config.yaml"
    observed = sha256_file(resolved_path)
    if manifest.get("resolved_config_yaml_sha256") != observed:
        raise ValueError(f"resolved-config YAML byte digest mismatch: {resolved_path}")
    return observed


def _terminal_tasks(path: Path, job_id: str) -> dict[int, dict[str, str]]:
    tasks: dict[int, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("|")
        if len(fields) < 3 or "_" not in fields[0]:
            continue
        raw_id, state, exit_code = fields[:3]
        parent, raw_index = raw_id.split("_", 1)
        if parent != job_id or not raw_index.isdigit():
            continue
        index = int(raw_index)
        if index in tasks:
            raise ValueError(f"duplicate Slurm terminal record for array task {index}")
        tasks[index] = {"job_id_raw": raw_id, "state": state, "exit_code": exit_code}
    if set(tasks) != set(range(len(PILOT_METHOD_IDS))):
        raise ValueError("Slurm terminal evidence does not contain exactly the eight pilot tasks")
    if any(row["state"] != "COMPLETED" or row["exit_code"] != "0:0" for row in tasks.values()):
        raise ValueError("pilot training array contains a non-success terminal task")
    return tasks


def _validate_checkpoint_scientific_binding(
    path: Path,
    *,
    binding: ExperimentBinding,
) -> dict[str, Any]:
    """Verify the immutable binding is inside, not merely beside, a checkpoint."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"pilot checkpoint payload is not a mapping: {path}")
    validate_checkpoint_experiment_binding_payload(payload, binding)
    return payload


def _require_confined_regular_file(path: Path, *, root: Path, name: str) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute normalized path")
    resolved_root = root.resolve(strict=True)
    try:
        relative = path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name} is outside its expected run directory") from error
    current = resolved_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink: {current}")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return resolved


def _require_absolute_regular_file(path: Path, *, name: str) -> Path:
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise ValueError(f"{name} must be an absolute normalized non-symlink path")
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return resolved


def _recompute_grpo_update_norm(
    root: Path,
    *,
    evidence: dict[str, Any],
    checkpoint_payload: dict[str, Any],
) -> tuple[float, str]:
    import torch

    baseline_path_value = evidence.get("update_norm_baseline_checkpoint_path")
    if (
        not isinstance(baseline_path_value, str)
        or checkpoint_payload.get("update_norm_baseline_checkpoint_path")
        != baseline_path_value
    ):
        raise ValueError("GRPO update-norm baseline path binding is inconsistent")
    baseline_sha256 = evidence.get("update_norm_baseline_checkpoint_sha256")
    if (
        not isinstance(baseline_sha256, str)
        or checkpoint_payload.get("update_norm_baseline_checkpoint_sha256")
        != baseline_sha256
    ):
        raise ValueError("GRPO update-norm baseline digest binding is inconsistent")
    baseline_path = _require_absolute_regular_file(
        Path(baseline_path_value),
        name="GRPO update-norm baseline checkpoint",
    )
    if sha256_file(baseline_path) != baseline_sha256:
        raise ValueError("GRPO update-norm baseline checkpoint bytes changed")
    baseline_model = load_checkpoint_model_state(baseline_path)

    snapshot = evidence.get("initial_parameter_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("production GRPO evidence lacks its initial parameter snapshot")
    snapshot_content = {key: value for key, value in snapshot.items() if key != "sha256"}
    if snapshot.get("sha256") != sha256_value(snapshot_content):
        raise ValueError("GRPO initial parameter snapshot hash mismatch")
    manifest_path = _require_confined_regular_file(
        Path(str(evidence.get("initial_parameter_snapshot_manifest_path", ""))),
        root=root,
        name="GRPO initial parameter snapshot manifest",
    )
    expected_manifest_path = (root / "initial_parameter_snapshot" / "manifest.json").resolve()
    if manifest_path != expected_manifest_path:
        raise ValueError("GRPO initial parameter snapshot manifest is in the wrong directory")
    if sha256_file(manifest_path) != evidence.get(
        "initial_parameter_snapshot_manifest_file_sha256"
    ):
        raise ValueError("GRPO initial parameter snapshot manifest bytes changed")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != snapshot:
        raise ValueError("GRPO embedded and on-disk initial snapshots differ")

    tensors = snapshot.get("tensors")
    final_model = checkpoint_payload.get("model")
    if not isinstance(tensors, dict) or not tensors or not isinstance(final_model, dict):
        raise ValueError("GRPO snapshot/final checkpoint model state is incomplete")
    if set(tensors) != set(baseline_model) or set(tensors) != set(final_model):
        raise ValueError(
            "GRPO baseline, initial snapshot, and final model tensor inventories differ"
        )
    squared = 0.0
    snapshot_root = (root / "initial_parameter_snapshot").resolve(strict=True)
    bound_snapshot_files = {
        "manifest.json": str(
            evidence.get("initial_parameter_snapshot_manifest_file_sha256", "")
        )
    }
    for name, metadata in tensors.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"GRPO snapshot metadata is invalid for {name}")
        tensor_path = _require_confined_regular_file(
            Path(str(metadata.get("path", ""))),
            root=snapshot_root,
            name=f"GRPO initial parameter tensor {name}",
        )
        if sha256_file(tensor_path) != metadata.get("sha256"):
            raise ValueError(f"GRPO initial parameter tensor bytes changed: {name}")
        relative_tensor_path = tensor_path.relative_to(snapshot_root).as_posix()
        if relative_tensor_path in bound_snapshot_files:
            raise ValueError("GRPO snapshot maps multiple parameters to one file")
        bound_snapshot_files[relative_tensor_path] = str(metadata.get("sha256", ""))
        before = torch.load(tensor_path, map_location="cpu", weights_only=True)
        baseline_tensor = baseline_model[name]
        after = final_model[name]
        if (
            not isinstance(before, torch.Tensor)
            or not isinstance(baseline_tensor, torch.Tensor)
            or not isinstance(after, torch.Tensor)
        ):
            raise ValueError(f"GRPO model state is not tensor-valued: {name}")
        if list(before.shape) != metadata.get("shape") or str(before.dtype) != metadata.get(
            "dtype"
        ):
            raise ValueError(f"GRPO initial parameter metadata changed: {name}")
        if before.shape != after.shape:
            raise ValueError(f"GRPO initial/final parameter shape changed: {name}")
        if before.dtype != after.dtype:
            raise ValueError(f"GRPO initial/final parameter dtype changed: {name}")
        baseline_cpu = baseline_tensor.detach().cpu().contiguous()
        before_cpu = before.detach().cpu().contiguous()
        if (
            before_cpu.shape != baseline_cpu.shape
            or before_cpu.dtype != baseline_cpu.dtype
            or not torch.equal(
                before_cpu.reshape(-1).view(torch.uint8),
                baseline_cpu.reshape(-1).view(torch.uint8),
            )
        ):
            raise ValueError(
                f"GRPO initial snapshot differs from its bound baseline checkpoint: {name}"
            )
        squared += float(torch.sum((after.detach().cpu().float() - before.float()) ** 2).item())
    observed_snapshot_files = accelerator_state_file_hashes(
        snapshot_root,
        expected_run_root=root,
    )
    if observed_snapshot_files != bound_snapshot_files:
        raise ValueError("GRPO initial parameter snapshot tree differs from its evidence binding")
    norm = math.sqrt(squared)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("GRPO final model did not change from its bound initial snapshot")
    return norm, torch_state_hash(final_model)


def _validate_factorial_accelerator_state(
    root: Path,
    *,
    checkpoint_path: Path,
    checkpoint_payload: dict[str, Any],
    resolved_config: dict[str, Any],
) -> dict[str, str] | None:
    """Validate the external Accelerate tree named by a Factorial checkpoint."""

    trainer_config = resolved_config.get("trainer")
    backend = trainer_config.get("backend") if isinstance(trainer_config, dict) else None
    is_accelerate = checkpoint_payload.get("format") == "accelerate_fsdp_full_export_v1"
    if backend == "accelerate" and not is_accelerate:
        raise ValueError("Factorial Accelerate run lacks its Accelerate/FSDP checkpoint format")
    if not is_accelerate:
        return None

    state_path_value = checkpoint_payload.get("accelerate_state_dir")
    state_files = checkpoint_payload.get("accelerate_state_files")
    state_sha256 = checkpoint_payload.get("accelerate_state_sha256")
    if not isinstance(state_path_value, str) or not state_path_value:
        raise ValueError("Factorial checkpoint lacks accelerate_state_dir")
    if not isinstance(state_files, dict) or not state_files:
        raise ValueError("Factorial checkpoint lacks accelerate_state_files")
    if (
        not isinstance(state_sha256, str)
        or len(state_sha256) != 64
        or any(character not in "0123456789abcdef" for character in state_sha256)
    ):
        raise ValueError("Factorial checkpoint has an invalid accelerate_state_sha256")

    expected_state_path = checkpoint_path.with_suffix(".accelerate").absolute()
    state_path = Path(state_path_value)
    if state_path != expected_state_path or state_path_value != str(expected_state_path):
        raise ValueError("Factorial accelerate_state_dir differs from its checkpoint location")
    return validate_accelerator_state_directory(
        state_path,
        expected_run_root=root,
        expected_files=state_files,
        expected_sha256=state_sha256,
    )


def _validate_factorial_rank_states(checkpoint_payload: dict[str, Any]) -> int:
    world_size = checkpoint_payload.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Factorial checkpoint world_size is invalid")
    rank_states: dict[str, list[dict[str, Any]]] = {}
    for name in ("prompt_scheduler_by_rank", "state_source_by_rank"):
        observed = checkpoint_payload.get(name)
        if not isinstance(observed, list) or len(observed) != world_size:
            raise ValueError(f"Factorial checkpoint {name} does not cover the exact world")
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(observed):
            if not isinstance(raw, dict):
                raise ValueError(f"Factorial checkpoint {name} contains a non-mapping state")
            state = dict(raw)
            if type(state.get("rank")) is not int or state.get("rank") != index:
                raise ValueError(f"Factorial checkpoint {name} rank differs at index {index}")
            if (
                type(state.get("world_size")) is not int
                or state.get("world_size") != world_size
            ):
                raise ValueError(
                    f"Factorial checkpoint {name} world_size differs at rank {index}"
                )
            normalized.append(state)
        rank_states[name] = normalized
    if (
        checkpoint_payload.get("prompt_scheduler")
        != rank_states["prompt_scheduler_by_rank"][0]
        or checkpoint_payload.get("state_source")
        != rank_states["state_source_by_rank"][0]
    ):
        raise ValueError("Factorial checkpoint rank-0 runtime state is inconsistent")
    return world_size


def _validate_grpo_update_evidence(
    root: Path,
    *,
    manifest: dict[str, Any],
    binding: ExperimentBinding,
    checkpoint_path: Path,
    checkpoint_payload: dict[str, Any],
) -> str:
    path = root / "grpo_update_evidence.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("GRPO update evidence must be a mapping")
    content = {key: value for key, value in evidence.items() if key != "sha256"}
    if evidence.get("sha256") != sha256_value(content):
        raise ValueError("GRPO update evidence SHA-256 mismatch")
    required = {
        "backend": binding.backend_id,
        "backend_version": binding.backend_version,
        "backend_batch_contract": binding.backend_batch_contract,
        "method_spec_sha256": binding.method_spec_sha256,
        "experiment_binding_sha256": binding.scientific_sha256,
        "factorial_design_sha256": binding.factorial_design_sha256,
        "resolved_batch_contract_sha256": binding.resolved_batch_contract_sha256,
        "current_policy_contract": asdict(REGISTERED_CURRENT_POLICY),
        "initial_checkpoint_hash": binding.initial_checkpoint_sha256,
        "git_commit": binding.implementation_commit,
        "implementation_dirty": binding.implementation_dirty,
        "trl_train_called": True,
        "parameters_changed": True,
        "resume_ancestry": manifest.get("resume_ancestry", []),
    }
    mismatches = {
        key: {"expected": value, "observed": evidence.get(key)}
        for key, value in required.items()
        if evidence.get(key) != value
    }
    batch_contract = evidence.get("batch_contract")
    current_policy = evidence.get("current_policy_contract")
    if not isinstance(batch_contract, dict) or not isinstance(current_policy, dict):
        mismatches["batch_contract"] = {
            "expected": "batch and current-policy mappings",
            "observed": batch_contract,
        }
    elif sha256_value({**batch_contract, "current_policy": current_policy}) != (
        binding.resolved_batch_contract_sha256
    ):
        mismatches["batch_contract_sha256"] = {
            "expected": binding.resolved_batch_contract_sha256,
            "observed": sha256_value({**batch_contract, "current_policy": current_policy}),
        }
    token_state = evidence.get("token_budget")
    checkpoint_token_state = checkpoint_payload.get("token_budget")
    if not isinstance(token_state, dict) or token_state != checkpoint_token_state:
        mismatches["token_budget"] = {
            "expected": checkpoint_token_state,
            "observed": token_state,
        }
    else:
        token_required = {
            "budget": manifest.get("token_budget"),
            "unit": TOKEN_BUDGET_UNIT,
            "consumed": manifest.get("token_budget_consumed"),
            "stop_reason": manifest.get("training_stop_reason"),
        }
        token_mismatches = {
            key: {"expected": value, "observed": token_state.get(key)}
            for key, value in token_required.items()
            if token_state.get(key) != value
        }
        if token_mismatches:
            mismatches["token_budget_manifest"] = token_mismatches
    if evidence.get("optimizer_steps") != checkpoint_payload.get("global_step"):
        mismatches["optimizer_steps"] = {
            "expected": checkpoint_payload.get("global_step"),
            "observed": evidence.get("optimizer_steps"),
        }
    if manifest.get("dataset_hashes", {}).get("grpo_update_evidence") != evidence.get(
        "sha256"
    ):
        mismatches["dataset_hashes.grpo_update_evidence"] = {
            "expected": evidence.get("sha256"),
            "observed": manifest.get("dataset_hashes", {}).get("grpo_update_evidence"),
        }
    if evidence.get("final_checkpoint_hash") != manifest.get("final_checkpoint_sha256"):
        mismatches["final_checkpoint_hash"] = {
            "expected": manifest.get("final_checkpoint_sha256"),
            "observed": evidence.get("final_checkpoint_hash"),
        }
    if sha256_file(checkpoint_path) != evidence.get("final_checkpoint_hash"):
        mismatches["final_checkpoint_bytes"] = {
            "expected": sha256_file(checkpoint_path),
            "observed": evidence.get("final_checkpoint_hash"),
        }
    evidence_checkpoint = Path(str(evidence.get("checkpoint", "")))
    if (
        not evidence_checkpoint.is_absolute()
        or evidence_checkpoint.resolve() != checkpoint_path.resolve()
    ):
        mismatches["checkpoint"] = {
            "expected": str(checkpoint_path.resolve()),
            "observed": evidence.get("checkpoint"),
        }
    if checkpoint_payload.get("resume_ancestry", []) != manifest.get("resume_ancestry", []):
        mismatches["checkpoint.resume_ancestry"] = {
            "expected": manifest.get("resume_ancestry", []),
            "observed": checkpoint_payload.get("resume_ancestry"),
        }

    optimizer = checkpoint_payload.get("optimizer")
    if not isinstance(optimizer, dict):
        mismatches["optimizer"] = {
            "expected": "external Accelerator state binding",
            "observed": optimizer,
        }
    else:
        state_files = optimizer.get("files")
        state_sha256 = optimizer.get("content_sha256")
        state_path = optimizer.get("accelerate_state_path")
        if (
            state_files != evidence.get("accelerate_state_hashes")
            or state_sha256 != evidence.get("accelerate_state_sha256")
            or state_path != evidence.get("accelerate_state_path")
            or state_path != checkpoint_payload.get("accelerate_state_path")
            or state_sha256 != checkpoint_payload.get("accelerate_state_sha256")
        ):
            mismatches["accelerate_state_binding"] = {
                "expected": {
                    "path": state_path,
                    "files": state_files,
                    "sha256": state_sha256,
                },
                "observed": {
                    "path": evidence.get("accelerate_state_path"),
                    "files": evidence.get("accelerate_state_hashes"),
                    "sha256": evidence.get("accelerate_state_sha256"),
                },
            }
        elif isinstance(state_path, str) and isinstance(state_files, dict):
            state_world_size = checkpoint_payload.get("world_size")
            validate_native_trainer_checkpoint_files(
                state_files,
                world_size=state_world_size if type(state_world_size) is int else 0,
            )
            validate_accelerator_state_directory(
                Path(state_path),
                expected_run_root=root,
                expected_files=state_files,
                expected_sha256=str(state_sha256),
            )
            trainer_state = json.loads(
                (Path(state_path) / "trainer_state.json").read_text(encoding="utf-8")
            )
            if (
                not isinstance(trainer_state, dict)
                or trainer_state.get("global_step") != checkpoint_payload.get("global_step")
            ):
                mismatches["native_trainer_global_step"] = {
                    "expected": checkpoint_payload.get("global_step"),
                    "observed": (
                        trainer_state.get("global_step")
                        if isinstance(trainer_state, dict)
                        else trainer_state
                    ),
                }

    world_size = evidence.get("world_size")
    expected_world_size = 4 if binding.protocol_track.startswith("qwen3_") else world_size
    if (
        type(world_size) is not int
        or world_size != expected_world_size
        or checkpoint_payload.get("world_size") != world_size
        or not isinstance(batch_contract, dict)
        or batch_contract.get("world_size") != world_size
        or evidence.get("distributed_consistency_passed") is not True
    ):
        mismatches["distributed_world"] = {
            "expected": expected_world_size,
            "observed": world_size,
        }
    consensus = evidence.get("distributed_training_consensus")
    summaries = evidence.get("rank_local_parameter_summaries")
    if isinstance(world_size, int) and isinstance(consensus, list):
        ranks = [row.get("rank") for row in consensus if isinstance(row, dict)]
        comparable = [
            {key: value for key, value in row.items() if key != "rank"}
            for row in consensus
            if isinstance(row, dict)
        ]
        consensus_expected = {
            "global_step": checkpoint_payload.get("global_step"),
            "token_budget": checkpoint_token_state,
            "experiment_binding_sha256": binding.scientific_sha256,
            "current_policy_contract": asdict(REGISTERED_CURRENT_POLICY),
            "optimizer_boundary": "completed_optimizer_step",
        }
        if (
            len(consensus) != world_size
            or ranks != list(range(world_size))
            or len(comparable) != world_size
            or any(row != consensus_expected for row in comparable)
        ):
            mismatches["distributed_training_consensus"] = {
                "expected": consensus_expected,
                "observed": consensus,
            }
    else:
        mismatches["distributed_training_consensus"] = {
            "expected": "one exact row per rank",
            "observed": consensus,
        }
    if isinstance(world_size, int) and isinstance(summaries, list):
        summaries_valid = len(summaries) == world_size
        for index, row in enumerate(summaries):
            if not isinstance(row, dict):
                summaries_valid = False
                continue
            row_content = {key: value for key, value in row.items() if key != "sha256"}
            numeric = (row.get("local_sum"), row.get("local_squared_sum"))
            summaries_valid = summaries_valid and (
                set(row)
                == {"rank", "local_numel", "local_sum", "local_squared_sum", "sha256"}
                and row.get("rank") == index
                and type(row.get("local_numel")) is int
                and row.get("local_numel", 0) > 0
                and row.get("sha256") == sha256_value(row_content)
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in numeric
                )
            )
        if not summaries_valid:
            mismatches["rank_local_parameter_summaries"] = {
                "expected": "one finite self-hashed summary per rank",
                "observed": summaries,
            }
    else:
        mismatches["rank_local_parameter_summaries"] = {
            "expected": "one summary per rank",
            "observed": summaries,
        }

    observed_norm, final_model_hash = _recompute_grpo_update_norm(
        root,
        evidence=evidence,
        checkpoint_payload=checkpoint_payload,
    )
    reported_norm = evidence.get("parameter_update_norm")
    if (
        isinstance(reported_norm, bool)
        or not isinstance(reported_norm, (int, float))
        or not math.isclose(
            float(reported_norm),
            observed_norm,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or observed_norm <= 0.0
    ):
        mismatches["parameter_update_norm"] = {
            "expected": observed_norm,
            "observed": reported_norm,
        }
    if evidence.get("final_model_state_hash") != final_model_hash:
        mismatches["final_model_state_hash"] = {
            "expected": final_model_hash,
            "observed": evidence.get("final_model_state_hash"),
        }
    snapshot = evidence.get("initial_parameter_snapshot")
    snapshot_sha256 = snapshot.get("sha256") if isinstance(snapshot, dict) else None
    if checkpoint_payload.get("initial_parameter_snapshot_sha256") != snapshot_sha256:
        mismatches["initial_parameter_snapshot_sha256"] = {
            "expected": snapshot_sha256,
            "observed": checkpoint_payload.get("initial_parameter_snapshot_sha256"),
        }
    baseline = evidence.get("update_norm_baseline_checkpoint_sha256")
    if checkpoint_payload.get("update_norm_baseline_checkpoint_sha256") != baseline:
        mismatches["update_norm_baseline_checkpoint_sha256"] = {
            "expected": baseline,
            "observed": checkpoint_payload.get("update_norm_baseline_checkpoint_sha256"),
        }
    ancestry = manifest.get("resume_ancestry", [])
    ancestry_valid = isinstance(ancestry, list) and all(
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
        for value in ancestry
    )
    if not ancestry_valid or len(set(ancestry)) != len(ancestry):
        mismatches["resume_ancestry_format"] = {
            "expected": "a unique sequence of sha256:<64 lowercase hex> identifiers",
            "observed": ancestry,
        }
        expected_baseline = None
    else:
        expected_baseline = (
            ancestry[-1][7:] if ancestry else binding.initial_checkpoint_sha256
        )
    if baseline != expected_baseline:
        mismatches["update_norm_baseline"] = {
            "expected": expected_baseline,
            "observed": baseline,
        }
    if mismatches:
        raise ValueError(f"GRPO update evidence differs from ExperimentBinding: {mismatches}")
    return str(evidence["sha256"])


def _validate_factorial_update_evidence(
    root: Path,
    *,
    manifest: dict[str, Any],
    binding: ExperimentBinding,
    checkpoint_path: Path,
    checkpoint_payload: dict[str, Any],
    resolved_config: dict[str, Any],
) -> str:
    """Independently prove that a complete Factorial checkpoint changed its baseline."""

    evidence_path = root / "factorial_update_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise ValueError("Factorial update evidence must be a mapping")
    content = {key: value for key, value in evidence.items() if key != "sha256"}
    if evidence.get("sha256") != sha256_value(content):
        raise ValueError("Factorial update evidence SHA-256 mismatch")

    _validate_factorial_accelerator_state(
        root,
        checkpoint_path=checkpoint_path,
        checkpoint_payload=checkpoint_payload,
        resolved_config=resolved_config,
    )
    _validate_factorial_rank_states(checkpoint_payload)

    ancestry = checkpoint_payload.get("resume_ancestry")
    ancestry_valid = isinstance(ancestry, list) and all(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
        for value in ancestry
    )
    if not ancestry_valid or len(set(ancestry)) != len(ancestry):
        raise ValueError("Factorial checkpoint ancestry is not a unique content-hash chain")
    if ancestry != manifest.get("resume_ancestry", []):
        raise ValueError("Factorial checkpoint ancestry differs from its run manifest")
    expected_baseline_hash = (
        ancestry[-1][7:] if ancestry else binding.initial_checkpoint_sha256
    )
    baseline_hash = checkpoint_payload.get("update_norm_baseline_checkpoint_sha256")
    baseline_path = Path(
        str(checkpoint_payload.get("update_norm_baseline_checkpoint_path", ""))
    )
    if baseline_hash != expected_baseline_hash:
        raise ValueError("Factorial update-norm baseline differs from checkpoint ancestry")
    if (
        not baseline_path.is_absolute()
        or ".." in baseline_path.parts
        or baseline_path.is_symlink()
        or not baseline_path.is_file()
        or baseline_path.resolve(strict=True) != baseline_path
        or sha256_file(baseline_path) != baseline_hash
    ):
        raise ValueError("Factorial update-norm baseline file is absent or changed")
    if not ancestry:
        configured_baseline = Path(
            str(
                resolved_config.get("production_safety", {}).get(
                    "initial_checkpoint_path",
                    "",
                )
            )
        ).absolute()
        if baseline_path != configured_baseline:
            raise ValueError("Factorial update-norm baseline is not the registered initial checkpoint")

    baseline_model = load_checkpoint_model_state(baseline_path)
    final_model = checkpoint_payload.get("model")
    if not isinstance(final_model, dict):
        raise ValueError("Factorial final checkpoint has no model state")
    observed_norm, observed_final_hash = model_update_evidence(
        baseline_model,
        final_model,
    )
    if not math.isfinite(observed_norm) or observed_norm <= 0.0:
        raise ValueError("Factorial final model did not change from its bound baseline")

    metrics_path = root / "metrics.jsonl"
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    update_rows = [
        {
            "step": row.get("step"),
            "parameter_update_norm": row.get("parameter_update_norm"),
        }
        for row in metric_rows
        if isinstance(row, dict) and row.get("parameter_update_norm") is not None
    ]
    global_step = checkpoint_payload.get("global_step")
    if not update_rows or int(float(update_rows[-1]["step"])) != global_step:
        raise ValueError("Factorial metrics do not reach the final optimizer step")
    if any(
        isinstance(row["parameter_update_norm"], bool)
        or not isinstance(row["parameter_update_norm"], (int, float))
        or not math.isfinite(float(row["parameter_update_norm"]))
        or float(row["parameter_update_norm"]) <= 0.0
        for row in update_rows
    ):
        raise ValueError("Factorial metrics contain a zero or non-finite optimizer update")

    state_hashes = checkpoint_runtime_state_hashes(checkpoint_payload)
    checkpoint_hash = sha256_file(checkpoint_path)
    metrics_hash = sha256_file(metrics_path)
    required = {
        "format_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "final_checkpoint_sha256": checkpoint_hash,
        "metrics_sha256": metrics_hash,
        "metric_update_rows": update_rows,
        "metric_update_rows_sha256": sha256_value(update_rows),
        "global_step": global_step,
        "parameter_update_norm": observed_norm,
        "final_model_state_hash": observed_final_hash,
        "update_norm_baseline_checkpoint_path": str(baseline_path),
        "update_norm_baseline_checkpoint_sha256": expected_baseline_hash,
        "token_budget": checkpoint_payload.get("token_budget"),
        "resume_ancestry": ancestry,
        "checkpoint_runtime_state_hashes": state_hashes,
        "manifest_hashes": checkpoint_payload.get("manifest_hashes"),
        "experiment_binding_sha256": binding.scientific_sha256,
        "factorial_design_sha256": binding.factorial_design_sha256,
        "method_spec_sha256": binding.method_spec_sha256,
        "git_commit": binding.implementation_commit,
        "implementation_dirty": binding.implementation_dirty,
    }
    mismatches: dict[str, Any] = {}
    for key, expected in required.items():
        observed = evidence.get(key)
        if key == "parameter_update_norm":
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                mismatches[key] = {"expected": expected, "observed": observed}
        elif observed != expected:
            mismatches[key] = {"expected": expected, "observed": observed}
    for field, expected in (
        ("parameter_update_norm", observed_norm),
        ("final_model_state_hash", observed_final_hash),
    ):
        observed = checkpoint_payload.get(field)
        if field == "parameter_update_norm":
            matches = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        else:
            matches = observed == expected
        if not matches:
            mismatches[f"checkpoint.{field}"] = {
                "expected": expected,
                "observed": observed,
            }
    dataset_hashes = manifest.get("dataset_hashes")
    if (
        not isinstance(dataset_hashes, dict)
        or dataset_hashes.get("factorial_update_evidence") != evidence.get("sha256")
    ):
        mismatches["dataset_hashes.factorial_update_evidence"] = {
            "expected": evidence.get("sha256"),
            "observed": (
                dataset_hashes.get("factorial_update_evidence")
                if isinstance(dataset_hashes, dict)
                else dataset_hashes
            ),
        }
    token_state = checkpoint_payload.get("token_budget")
    token_required = {
        "budget": manifest.get("token_budget"),
        "unit": manifest.get("token_budget_unit"),
        "consumed": manifest.get("token_budget_consumed"),
        "accepted_optimizer_updates": global_step,
    }
    if not isinstance(token_state, dict) or any(
        token_state.get(key) != expected for key, expected in token_required.items()
    ):
        mismatches["token_budget"] = {
            "expected": token_required,
            "observed": token_state,
        }
    if mismatches:
        raise ValueError(f"Factorial update evidence differs from checkpoint: {mismatches}")
    return str(evidence["sha256"])


def bind_pilot_training_outputs(
    *, run_dir: Path, terminal: Path, job_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    expected = formal_artifact_binding(config)
    tasks = _terminal_tasks(terminal, job_id)
    terminal_hash = sha256_file(terminal)
    budget = int(config["trainer"]["token_budget"])
    initial_hash = str(config["production_safety"]["initial_checkpoint_hash"])
    bound: dict[str, Any] = {}
    scientific_bindings: dict[str, ExperimentBinding] = {}
    pending_manifests: list[tuple[str, Path, dict[str, Any]]] = []
    for index, cell in enumerate(PILOT_METHOD_IDS):
        root = run_dir / "runs" / cell / "seed-42"
        manifest_path = root / "manifest.json"
        manifest = validate_run_manifest_payload(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        scientific_binding = validate_run_manifest_experiment_binding(manifest)
        scientific_bindings[cell] = scientific_binding
        required = {
            "experiment_cell": cell,
            "seed": 42,
            "protocol_track": expected["protocol_track"],
            "artifact_namespace": expected["artifact_namespace"],
            "model_revision": expected["model_revision"],
            "protocol_teacher_revision": expected["teacher_revision"],
            "tokenizer_revision": expected["tokenizer_revision"],
            "tokenizer_fingerprint": expected["tokenizer_fingerprint"],
            "chat_template_sha256": expected["chat_template_sha256"],
            "prompt_protocol": expected["prompt_protocol"],
            "enable_thinking": expected["enable_thinking"],
            "git_commit": expected["code_commit"],
            "prereg_path": expected["prereg_path"],
            "prereg_version": expected["prereg_version"],
            "prereg_git_commit": expected["prereg_commit"],
            "prereg_sha256": expected["prereg_sha256"],
            "token_budget": budget,
            "token_budget_unit": TOKEN_BUDGET_UNIT,
            "dirty_working_tree": False,
            "prereg_dirty": False,
        }
        mismatches = {
            key: {"expected": value, "observed": manifest.get(key)}
            for key, value in required.items()
            if manifest.get(key) != value
        }
        if manifest.get("dataset_hashes", {}).get("initial_checkpoint") != initial_hash:
            mismatches["dataset_hashes.initial_checkpoint"] = {
                "expected": initial_hash,
                "observed": manifest.get("dataset_hashes", {}).get("initial_checkpoint"),
            }
        if mismatches:
            raise ValueError(f"pilot cell provenance mismatch for {cell}: {mismatches}")
        if not any(key.startswith("prerequisite_probe") for key in manifest["dataset_hashes"]):
            raise ValueError(f"pilot cell {cell} did not bind its frozen probe manifest")
        metrics = root / "metrics.jsonl"
        checkpoint = Path(str(manifest.get("final_checkpoint_path", "")))
        if not checkpoint.resolve().is_relative_to((root / "checkpoints").resolve()):
            raise ValueError(f"pilot final checkpoint is outside its cell directory: {cell}")
        if sha256_file(metrics) != manifest.get("metrics_sha256"):
            raise ValueError(f"pilot metrics changed before terminal binding: {cell}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != manifest.get("final_checkpoint_sha256"):
            raise ValueError(f"pilot final checkpoint changed before terminal binding: {cell}")
        checkpoint_payload = _validate_checkpoint_scientific_binding(
            checkpoint,
            binding=scientific_binding,
        )
        consumed = int(manifest.get("token_budget_consumed", -1))
        if not 0 <= consumed <= budget or not manifest.get("training_stop_reason"):
            raise ValueError(f"pilot token budget evidence is incomplete for {cell}")
        resolved = root / "resolved_config.yaml"
        resolved_payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(resolved_payload, dict):
            raise ValueError(f"pilot resolved config is not a mapping: {cell}")
        config_binding = _load_run_config_binding(
            root,
            manifest=manifest,
            resolved_config=resolved_payload,
            experiment_binding=scientific_binding,
        )
        resolved_yaml_sha256 = _validate_resolved_config_yaml_sha256(
            root,
            manifest=manifest,
        )
        resolved_required = {
            "experiment.name": cell,
            "seed": 42,
            "protocol_track": expected["protocol_track"],
            "prereg_path": expected["prereg_path"],
            "prereg_version": expected["prereg_version"],
            "model.model_revision": expected["model_revision"],
            "teacher.model_revision": expected["teacher_revision"],
            "trainer.token_budget": budget,
            "trainer.token_budget_unit": TOKEN_BUDGET_UNIT,
            "production_safety.initial_checkpoint_hash": initial_hash,
        }

        def nested_value(dotted: str, payload: dict[str, Any] = resolved_payload) -> Any:
            value: Any = payload
            for part in dotted.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            return value

        resolved_mismatches = {
            key: {"expected": value, "observed": nested_value(key)}
            for key, value in resolved_required.items()
            if nested_value(key) != value
        }
        if resolved_mismatches:
            raise ValueError(f"pilot resolved config mismatch for {cell}: {resolved_mismatches}")
        grpo_evidence_hash = None
        factorial_evidence_hash = None
        if scientific_binding.backend_id == TRL_GRPO_BACKEND:
            grpo_evidence_hash = _validate_grpo_update_evidence(
                root,
                manifest=manifest,
                binding=scientific_binding,
                checkpoint_path=checkpoint,
                checkpoint_payload=checkpoint_payload,
            )
        else:
            factorial_evidence_hash = _validate_factorial_update_evidence(
                root,
                manifest=manifest,
                binding=scientific_binding,
                checkpoint_path=checkpoint,
                checkpoint_payload=checkpoint_payload,
                resolved_config=resolved_payload,
            )
        dataset_hashes = manifest.get("dataset_hashes", {})
        probe_hashes = sorted(
            str(value) for key, value in dataset_hashes.items() if key.startswith("prerequisite_probe")
        )
        binding = {
            "cell": cell,
            "seed": 42,
            "resolved_config_sha256": config_binding.resolved_config_sha256,
            "scientific_config_sha256": config_binding.scientific_config_sha256,
            "execution_config_sha256": config_binding.execution_config_sha256,
            "resolved_config_yaml_sha256": resolved_yaml_sha256,
            "metrics_sha256": sha256_file(metrics),
            "final_checkpoint_sha256": sha256_file(checkpoint),
            "terminal_evidence_sha256": terminal_hash,
            "execution_task": tasks[index],
            "token_budget": budget,
            "token_budget_consumed": consumed,
            "token_budget_unit": TOKEN_BUDGET_UNIT,
            "training_stop_reason": manifest["training_stop_reason"],
            "dataset_hashes_sha256": sha256_value(dataset_hashes),
            "probe_manifest_hashes": probe_hashes,
            "initial_checkpoint_sha256": initial_hash,
            "state_source_artifact_sha256": str(manifest.get("rollout_bank_hash", "")),
            "experiment_binding_sha256": scientific_binding.scientific_sha256,
            "factorial_design_sha256": scientific_binding.factorial_design_sha256,
            "method_spec_sha256": scientific_binding.method_spec_sha256,
            "grpo_update_evidence_sha256": grpo_evidence_hash,
            "factorial_update_evidence_sha256": factorial_evidence_hash,
        }
        binding["sha256"] = sha256_value(binding)
        manifest["pilot_terminal_binding"] = binding
        manifest["sha256"] = sha256_value({key: value for key, value in manifest.items() if key != "sha256"})
        pending_manifests.append((cell, manifest_path, manifest))

    CONTROLLED_FACTORIAL.validate_bindings(
        scientific_bindings[cell] for cell in FACTORIAL_METHOD_IDS
    )
    CONTROLLED_FACTORIAL.validate_canonical_grpo_anchor_pair(
        scientific_bindings["online_verified_replay"],
        scientific_bindings["canonical_grpo"],
    )
    for cell, manifest_path, manifest in pending_manifests:
        atomic_write_json(manifest_path, manifest)
        bound[cell] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "experiment_binding_sha256": scientific_bindings[cell].scientific_sha256,
            "factorial_design_sha256": scientific_bindings[cell].factorial_design_sha256,
        }
    result: dict[str, Any] = {"job_id": job_id, "terminal_sha256": terminal_hash, "cells": bound}
    result["sha256"] = sha256_value(result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = bind_pilot_training_outputs(
        run_dir=args.run_dir,
        terminal=args.terminal,
        job_id=args.job_id,
        config=compose_config(args.overrides),
    )
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
