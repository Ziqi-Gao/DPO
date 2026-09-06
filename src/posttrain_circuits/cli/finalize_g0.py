"""Finalize the hash-bound G0 decision from production artifacts only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import (
    require_scientific_artifact,
    scientific_compatibility_fields,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json, utc_now
from posttrain_circuits.artifacts.runs import formal_artifact_binding, require_git_output
from posttrain_circuits.causal_circuits.dynamics import estimate_estimator_noise_floor
from posttrain_circuits.datasets.circuit_probes.cohorts import validate_probe_cohort_manifest
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.cli.compare_distributed_resume import (
    validate_distributed_resume_report,
)
from posttrain_circuits.cli.factorial_run_validation import strict_json_object
from posttrain_circuits.learning.training.fsdp_contract import (
    REQUESTED_FSDP_SHARDING_STRATEGY,
    effective_fsdp_sharding_strategy,
)
from posttrain_circuits.core.readiness import build_readiness_report, validate_anti_shortcut_report
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.datasets.proofgraph.leakage import validate_label_leakage_artifact
from posttrain_circuits.learning.teacher.evaluation import validate_teacher_readiness_artifact


_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_WORLD_SIZES = (1, 2, 3, 4)
_AMENDMENT_ID = "qwen3_v2_g0_elastic_v1"
_BATCH_PARTITION_PROTOCOL = "allocation_neutral_exact_global_batch_v1"
_GLOBAL_BATCH_SIZE = 64
_MAX_MICROBATCH_SIZE = 4
_PROMPT_POPULATION_SIZE = 256
_SAMPLES_BY_WORLD_SIZE = {
    1: [64],
    2: [32, 32],
    3: [22, 21, 21],
    4: [16, 16, 16, 16],
}
_MICROBATCHES_BY_WORLD_SIZE = {
    1: [[4] * 16],
    2: [[4] * 8, [4] * 8],
    3: [[4, 4, 4, 4, 4, 2], [4, 4, 4, 4, 4, 1], [4, 4, 4, 4, 4, 1]],
    4: [[4] * 4, [4] * 4, [4] * 4, [4] * 4],
}
G0_CHECK_NAMES = tuple(
    sorted(
        (
            "anti_shortcut",
            "artifact_reconstruction",
            "attribution_exact_calibration",
            "base_capable_probes",
            "base_task_accuracy",
            "batch_token_invariants",
            "bootstrap_stability",
            "calibration_anchor_improves_accuracy",
            "distinct_stage_manifests",
            "distributed_checkpoint_resume",
            "distributed_resume_fsdp_strategy",
            "final_stage_eap_beats_matched_random",
            "fixed_bank_mixed_rewards",
            "full_corruption_sanity",
            "gpu_preflight",
            "gpu_preflight_binding",
            "hf_transformerlens_gqa_parity",
            "identity_sanity",
            "label_leakage",
            "probe_scoring_binding",
            "process_stage_eap_beats_matched_random",
            "protocol_amendment",
            "qwen3_protocol_bindings",
            "qwen3_qk_norm_hook_semantics",
            "split_probe_isolation",
            "teacher_correctness",
            "teacher_topk_mass",
        )
    )
)
G0_REPORT_FIELDS = frozenset(
    {
        "allocation_sha256",
        "artifact_hashes",
        "artifact_namespace",
        "batch_token_contract",
        "chat_template_sha256",
        "checks",
        "circuit_probe_schema_version",
        "code_commit",
        "created_at",
        "enable_thinking",
        "format_version",
        "generator_version",
        "git_commit",
        "job_ids",
        "label_semantics",
        "launch_environment",
        "metrics",
        "model_revision",
        "passed",
        "phase",
        "prereg_commit",
        "prereg_path",
        "prereg_sha256",
        "prereg_version",
        "prompt_protocol",
        "protocol_amendment_git_commit",
        "protocol_amendment_id",
        "protocol_amendment_sha256",
        "protocol_prereg_version",
        "protocol_track",
        "request_git_commit",
        "resolved_config_sha256",
        "reviewed_implementation_commit",
        "sha256",
        "teacher_revision",
        "tokenizer_fingerprint",
        "tokenizer_revision",
        "world_size",
    }
)


@dataclass(frozen=True)
class ScientificInvocationContext:
    """Provenance supplied only by the validated ServerScheduler handler ABI."""

    allocation_sha256: str
    code_commit: str
    gpu_preflight_git_commit: str
    protocol_amendment_sha256: str
    request_git_commit: str
    reviewed_implementation_commit: str
    world_size: int

    def validate(self) -> None:
        for name in (
            "code_commit",
            "gpu_preflight_git_commit",
            "request_git_commit",
            "reviewed_implementation_commit",
        ):
            if _GIT_COMMIT.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"trusted G0 context {name} is not a Git commit")
        for name in ("allocation_sha256", "protocol_amendment_sha256"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"trusted G0 context {name} is not a SHA-256 digest")
        if self.world_size not in _ALLOWED_WORLD_SIZES:
            raise ValueError("trusted G0 context world_size is outside 1..4")
        if self.reviewed_implementation_commit in {
            self.code_commit,
            self.gpu_preflight_git_commit,
            self.request_git_commit,
        }:
            raise ValueError("trusted G0 context has a self-referential review binding")


def _batch_token_contract(config: dict[str, Any], world_size: int) -> dict[str, Any]:
    schedules = _MICROBATCHES_BY_WORLD_SIZE[world_size]
    samples = _SAMPLES_BY_WORLD_SIZE[world_size]
    if tuple(sum(schedule) for schedule in schedules) != tuple(samples):
        raise ValueError("trusted G0 batch schedule is internally inconsistent")
    return {
        "batch_partition_protocol": str(config["trainer"]["batch_partition_protocol"]),
        "requested_fsdp_sharding_strategy": REQUESTED_FSDP_SHARDING_STRATEGY,
        "effective_fsdp_sharding_strategy": effective_fsdp_sharding_strategy(
            world_size
        ),
        "global_logical_batch_size": int(config["trainer"]["global_batch_size"]),
        "max_per_rank_microbatch_size": int(config["trainer"]["max_microbatch_size"]),
        "max_model_input_length": int(config["trainer"]["max_model_input_length"]),
        "full_parameter_training": bool(config["g0"]["full_parameter_training"]),
        "prompt_population_size": int(config["task"]["num_examples"]),
        "prompt_ids_unique": True,
        "accepted_view_prompt_order": "exactly_manifest_ordered_prompt_ids",
        "prompt_population_alignment": "exact_multiple_of_global_logical_batch_size",
        "microbatch_schedule_by_rank": schedules,
        "optimizer_microsteps": len(schedules[0]),
        "samples_by_rank": samples,
        "world_size": world_size,
        "token_budget": int(config["trainer"]["token_budget"]),
        "token_budget_unit": str(config["trainer"]["token_budget_unit"]),
        "max_optimizer_steps": int(config["trainer"]["max_steps"]),
    }


def _read(path: Path) -> dict[str, Any]:
    return strict_json_object(path, name=f"G0 artifact {path.name}")


def _preflight_fsdp_wrapper_count(
    gpu_preflight: dict[str, Any],
    *,
    world_size: int,
) -> int:
    """Bind the G0 FSDP tree to the selected preflight's reviewed topology."""

    if world_size not in _ALLOWED_WORLD_SIZES:
        raise ValueError("GPU preflight FSDP topology world size is outside 1..4")
    rows = gpu_preflight.get("rank_training_checks")
    if not isinstance(rows, list) or len(rows) != world_size:
        raise ValueError("GPU preflight FSDP topology omits a rank")
    wrapper_counts: set[int] = set()
    transformer_counts: set[int] = set()
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or type(row.get("rank")) is not int
            or row.get("rank") != rank
        ):
            raise ValueError("GPU preflight FSDP topology rank order is invalid")
        wrappers = row.get("fsdp_wrapper_count")
        transformers = row.get("wrapped_transformer_blocks")
        if (
            row.get("fsdp_transformer_layer") != "Qwen3DecoderLayer"
            or type(wrappers) is not int
            or type(transformers) is not int
            or transformers < 1
            or wrappers != transformers + 1
        ):
            raise ValueError("GPU preflight FSDP transformer topology is invalid")
        wrapper_counts.add(wrappers)
        transformer_counts.add(transformers)
    if len(wrapper_counts) != 1 or len(transformer_counts) != 1:
        raise ValueError("GPU preflight ranks disagree on the FSDP transformer topology")
    return next(iter(wrapper_counts))


def _require_finite_numbers(value: Any, *, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_numbers(item, name=f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_numbers(item, name=f"{name}[{index}]")


def validate_probe_score_artifact(path: Path) -> dict[str, Any]:
    """Read the stage-7 score artifact without JSON or numeric ambiguity."""

    payload = _read(path)
    _require_finite_numbers(payload, name="probe score artifact")
    digest = payload.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("probe score artifact lacks a lowercase SHA-256")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    if digest != sha256_value(unsigned):
        raise ValueError("probe score artifact hash mismatch")
    scores = payload.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("probe score artifact lacks a non-empty score mapping")
    for example_id, row in scores.items():
        if (
            not isinstance(example_id, str)
            or not example_id
            or not isinstance(row, dict)
            or set(row) != {"initial_correct", "learnable_after_post_training"}
            or type(row["initial_correct"]) is not bool
            or type(row["learnable_after_post_training"]) is not bool
        ):
            raise ValueError("probe score artifact contains an invalid score row")
    for name in ("initial_validation_metrics", "calibrated_validation_metrics"):
        metrics = payload.get(name)
        if (
            not isinstance(metrics, dict)
            or type(metrics.get("answer_accuracy")) not in (int, float)
            or not math.isfinite(float(metrics["answer_accuracy"]))
        ):
            raise ValueError(f"probe score artifact has invalid {name}")
    return payload


def _require_formal_binding(
    artifact: dict[str, Any],
    expected: dict[str, Any],
    *,
    name: str,
) -> None:
    keys = [
        "protocol_track",
        "artifact_namespace",
        "model_revision",
        "teacher_revision",
        "tokenizer_revision",
        "tokenizer_fingerprint",
        "chat_template_sha256",
        "prompt_protocol",
        "enable_thinking",
        "code_commit",
        "prereg_path",
        "prereg_version",
        "prereg_commit",
        "prereg_sha256",
    ]
    if "protocol_amendment_id" in expected:
        keys.extend(
            (
                "protocol_amendment_id",
                "protocol_amendment_path",
                "protocol_amendment_git_commit",
                "protocol_amendment_sha256",
                "reviewed_implementation_commit",
            )
        )
    mismatches = {
        key: {"expected": expected[key], "observed": artifact.get(key)}
        for key in keys
        if artifact.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(f"{name} formal binding mismatch: {mismatches}")


def _teacher_readiness_formal_binding(artifact: dict[str, Any]) -> dict[str, Any]:
    bindings = artifact.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("teacher-readiness artifact has no binding mapping")
    required = (
        "student_model_revision",
        "teacher_model_revision",
        "student_tokenizer_revision",
    )
    missing = [name for name in required if not str(bindings.get(name, "")).strip()]
    if missing:
        raise ValueError(f"teacher-readiness artifact lacks normalization fields: {missing}")
    normalized = dict(bindings)
    normalized["model_revision"] = normalized.pop("student_model_revision")
    normalized["teacher_revision"] = normalized.pop("teacher_model_revision")
    normalized["tokenizer_revision"] = normalized["student_tokenizer_revision"]
    return normalized


def _stage_compatibility_passes(
    compatibility: dict[str, Any],
    circuit: dict[str, Any],
) -> bool:
    """Validate one stage's parity report and its exact circuit binding."""

    digest = compatibility.get("sha256")
    payload = {
        key: value
        for key, value in compatibility.items()
        if key not in {"sha256", "hf_identity_max_error"}
    }
    return (
        isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
        and compatibility.get("passed") is True
        and compatibility.get("hf_identity_passed") is True
        and compatibility.get("transformerlens_parity_passed") is True
        and digest == sha256_value(payload)
        and circuit.get("model_compatibility_hash") == digest
    )


def _qwen3_qk_norm_hooks_pass(compatibility: dict[str, Any]) -> bool:
    """Require Qwen3 query/key hooks before normalization and RoPE."""

    hook_positions = compatibility.get("hook_positions")
    if not isinstance(hook_positions, dict):
        return False
    query_hooks = hook_positions.get("query_projection")
    key_hooks = hook_positions.get("key_projection")
    return (
        isinstance(query_hooks, list)
        and isinstance(key_hooks, list)
        and bool(query_hooks and key_hooks)
        and all(
            isinstance(value, str) and "pre_q_norm_pre_rope" in value
            for value in query_hooks
        )
        and all(
            isinstance(value, str) and "pre_k_norm_pre_rope" in value
            for value in key_hooks
        )
    )


def replay_g0_decision(
    workspace: Path,
    report: dict[str, Any],
    *,
    config: dict[str, Any],
    scientific_context: ScientificInvocationContext,
    formal_binding: dict[str, Any],
    job_id: str,
) -> dict[str, bool]:
    """Revalidate a bundled G0 decision from its complete extracted workspace."""

    scientific_context.validate()
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError("replayed G0 workspace must be a real directory")
    paths = {
        "initial_checkpoint": workspace / "initial_checkpoint.pt",
        "gpu_preflight": workspace / "gpu_preflight.json",
        "base_scores": workspace / "probe_scores.json",
        "teacher_store_manifest": workspace / "teacher_scores" / "manifest.json",
        "teacher_readiness": workspace / "teacher_readiness.json",
        "label_leakage": workspace / "label_leakage.json",
        "anti_shortcut": workspace / "anti_shortcut.json",
        "probe_manifest": workspace / "probes" / "manifest.json",
        "final_circuit": workspace / "circuits" / "final_answer" / "circuit.json",
        "final_exact_patching": workspace
        / "circuits"
        / "final_answer"
        / "exact_patching.json",
        "process_circuit": workspace
        / "circuits"
        / "first_rule_selection"
        / "circuit.json",
        "process_exact_patching": workspace
        / "circuits"
        / "first_rule_selection"
        / "exact_patching.json",
        "final_compatibility": workspace
        / "circuits"
        / "final_answer"
        / "mib_raw"
        / "compatibility.json",
        "process_compatibility": workspace
        / "circuits"
        / "first_rule_selection"
        / "mib_raw"
        / "compatibility.json",
        "distributed_resume": workspace / "distributed_resume.json",
    }
    prereg_version = str(config["prereg_version"])
    base = validate_probe_score_artifact(paths["base_scores"])
    teacher = TrajectoryStore(paths["teacher_store_manifest"].parent).check_integrity()
    require_scientific_artifact(teacher, expected_prereg_version=prereg_version)
    initial_checkpoint_hash = sha256_file(paths["initial_checkpoint"])
    label_leakage = validate_label_leakage_artifact(_read(paths["label_leakage"]))
    teacher_readiness_raw = _read(paths["teacher_readiness"])
    tokenized_prefix = teacher_readiness_raw.get("tokenized_prefix_manifest")
    if not isinstance(tokenized_prefix, dict) or tokenized_prefix.get("sha256") != sha256_value(
        {key: value for key, value in tokenized_prefix.items() if key != "sha256"}
    ):
        raise ValueError("teacher-readiness tokenized prefix manifest is invalid")
    readiness_bindings = {
        **formal_binding,
        "teacher_model_id": str(config["teacher"]["model_name_or_path"]),
        "teacher_model_revision": str(config["teacher"]["model_revision"]),
        "student_model_id": str(config["model"]["model_name_or_path"]),
        "student_model_revision": str(config["model"]["model_revision"]),
        "student_tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "teacher_tokenizer_revision": str(config["teacher"]["tokenizer_revision"]),
        "tokenizer_revision": str(config["teacher"]["tokenizer_revision"]),
        "dataset_hash": str(label_leakage["dataset_hash"]),
        "prefix_probe_hash": str(tokenized_prefix["sha256"]),
    }
    readiness_bindings.pop("teacher_revision")
    readiness_bindings.pop("model_revision")
    teacher_readiness = validate_teacher_readiness_artifact(
        teacher_readiness_raw,
        expected_bindings=readiness_bindings,
    )
    anti = validate_anti_shortcut_report(
        paths["anti_shortcut"],
        max_shortcut_gap=float(config["anti_shortcut"]["max_shortcut_gap"]),
        expected_model_checkpoint_hash=initial_checkpoint_hash,
    )
    probes = validate_probe_cohort_manifest(
        paths["probe_manifest"],
        expected_initial_checkpoint_hash=initial_checkpoint_hash,
    )
    final_circuit = _read(paths["final_circuit"])
    final_exact = _read(paths["final_exact_patching"])
    process_circuit = _read(paths["process_circuit"])
    process_exact = _read(paths["process_exact_patching"])
    for artifact in (final_circuit, final_exact, process_circuit, process_exact):
        require_scientific_artifact(
            artifact,
            expected_prereg_version=prereg_version,
            require_circuit_schema=True,
            require_hash=True,
        )
    final_compatibility = _read(paths["final_compatibility"])
    process_compatibility = _read(paths["process_compatibility"])
    resume = validate_distributed_resume_report(
        paths["distributed_resume"],
        config=config,
        expected_world_size=scientific_context.world_size,
        expected_formal_binding=formal_binding,
    )
    gpu_preflight = _read(paths["gpu_preflight"])
    gpu_digest = gpu_preflight.pop("sha256", None)
    if gpu_digest != sha256_value(gpu_preflight) or gpu_preflight.get("passed") is not True:
        raise ValueError("G0 requires a passed, hash-valid GPU preflight")
    gpu_preflight["sha256"] = gpu_digest
    preflight_fsdp_wrapper_count = _preflight_fsdp_wrapper_count(
        gpu_preflight,
        world_size=scientific_context.world_size,
    )

    teacher_readiness_formal = _teacher_readiness_formal_binding(teacher_readiness)
    bound_artifacts = {
        "base_scores": base,
        "teacher_store": teacher,
        "teacher_readiness": teacher_readiness_formal,
        "label_leakage": label_leakage,
        "anti_shortcut": anti,
        "probe_manifest": probes,
        "final_circuit": final_circuit,
        "final_exact_patching": final_exact,
        "process_circuit": process_circuit,
        "process_exact_patching": process_exact,
        "final_compatibility": final_compatibility,
        "process_compatibility": process_compatibility,
        "distributed_resume": resume,
        "gpu_preflight": gpu_preflight,
    }
    for artifact_name, artifact in bound_artifacts.items():
        if artifact_name != "gpu_preflight":
            _require_formal_binding(artifact, formal_binding, name=artifact_name)

    final_noise = estimate_estimator_noise_floor(
        final_circuit.get("bootstrap_score_vectors", []), activation_threshold=0.0
    )
    process_noise = estimate_estimator_noise_floor(
        process_circuit.get("bootstrap_score_vectors", []), activation_threshold=0.0
    )
    metrics = base.get("initial_validation_metrics", {})
    calibrated_metrics = base.get("calibrated_validation_metrics", {})
    teacher_mass = teacher.get("teacher_topk_mass", {})
    bank_total = int(teacher.get("total_trajectories", 0))
    bank_positive = int(teacher.get("reward_distribution", {}).get("positive", 0))
    checks: dict[str, bool] = {
        "base_task_accuracy": float(metrics.get("answer_accuracy", 0.0))
        >= float(config["anti_shortcut"]["minimum_iid_accuracy"]),
        "teacher_topk_mass": float(teacher_mass.get("minimum", 0.0)) >= 0.90,
        "teacher_correctness": teacher_readiness.get("passed") is True,
        "label_leakage": label_leakage.get("passed") is True,
        "fixed_bank_mixed_rewards": 0 < bank_positive < bank_total,
        "calibration_anchor_improves_accuracy": float(
            calibrated_metrics.get("answer_accuracy", 0.0)
        )
        > float(metrics.get("answer_accuracy", 0.0)),
        "anti_shortcut": anti.get("passed") is True,
        "base_capable_probes": probes["cohorts"]["base_capable"]["discovery"][
            "num_examples"
        ]
        > 0
        and probes["cohorts"]["base_capable"]["validation"]["num_examples"] > 0,
        "probe_scoring_binding": base.get("initial_checkpoint_sha256")
        == initial_checkpoint_hash
        and probes.get("scoring_manifest_hash") == base.get("sha256")
        and probes.get("eligibility_evidence_ancestry")
        == base.get("eligibility_evidence_ancestry"),
        "hf_transformerlens_gqa_parity": _stage_compatibility_passes(
            final_compatibility,
            final_circuit,
        )
        and _stage_compatibility_passes(
            process_compatibility,
            process_circuit,
        ),
        "bootstrap_stability": min(
            float(final_noise["within_checkpoint_full_score_spearman"]),
            float(process_noise["within_checkpoint_full_score_spearman"]),
        )
        >= float(config["g0"]["minimum_bootstrap_spearman"]),
        "final_stage_eap_beats_matched_random": float(
            final_exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_selected_vs_random_cpr_margin"]),
        "process_stage_eap_beats_matched_random": float(
            process_exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_selected_vs_random_cpr_margin"]),
        "distinct_stage_manifests": final_circuit.get("probe_stage") == "final_answer"
        and process_circuit.get("probe_stage")
        in {"first_rule_selection", "intermediate_conclusion"}
        and final_circuit.get("stage_target_manifest_hash")
        != process_circuit.get("stage_target_manifest_hash"),
        "identity_sanity": all(
            artifact.get("sanity_checks", {}).get("identity_passed") is True
            for artifact in (final_exact, process_exact)
        ),
        "full_corruption_sanity": all(
            artifact.get("sanity_checks", {}).get("full_corruption_passed") is True
            for artifact in (final_exact, process_exact)
        ),
        "attribution_exact_calibration": all(
            float(artifact.get("attribution_patching_spearman", float("-inf")))
            >= float(config["g0"]["minimum_attribution_exact_spearman"])
            and float(
                artifact.get("attribution_patching_spearman_ci", {}).get(
                    "lower", float("-inf")
                )
            )
            >= float(config["g0"]["minimum_spearman_bootstrap_lower_bound"])
            for artifact in (final_exact, process_exact)
        ),
        "distributed_checkpoint_resume": resume.get("format_version") == 2
        and resume.get("passed") is True
        and int(resume.get("world_size", 0)) == scientific_context.world_size
        and resume.get("max_optimizer_steps") == int(config["trainer"]["max_steps"])
        and isinstance(resume.get("checks"), dict)
        and bool(resume["checks"])
        and all(value is True for value in resume["checks"].values()),
        "split_probe_isolation": probes.get("frozen_before_training") is True,
        "artifact_reconstruction": all(
            circuit.get("checkpoint_sha256") == initial_checkpoint_hash
            and exact.get("artifacts", {}).get("checkpoint_sha256")
            == initial_checkpoint_hash
            for circuit, exact in (
                (final_circuit, final_exact),
                (process_circuit, process_exact),
            )
        ),
        "gpu_preflight": gpu_preflight.get("passed") is True
        and int(gpu_preflight.get("world_size", 0)) == scientific_context.world_size,
    }
    checks["gpu_preflight_binding"] = (
        gpu_preflight.get("git_commit") == scientific_context.gpu_preflight_git_commit
        and gpu_preflight.get("model_revision") == config["model"]["model_revision"]
        and gpu_preflight.get("teacher_revision") == config["teacher"]["model_revision"]
    )
    qwen3_bindings = {
        "protocol_track": config["protocol_track"],
        "artifact_namespace": config["model"]["artifact_namespace"],
        "prompt_protocol": "qwen3_non_thinking_v1",
        "enable_thinking": False,
        "chat_template_sha256": config["model"]["prompt_protocol"][
            "chat_template_sha256"
        ],
        "tokenizer_fingerprint": config["model"]["tokenizer_fingerprint"],
    }
    checks["qwen3_protocol_bindings"] = all(
        all(artifact.get(key) == value for key, value in qwen3_bindings.items())
        for artifact in bound_artifacts.values()
    )
    checks["qwen3_qk_norm_hook_semantics"] = (
        _qwen3_qk_norm_hooks_pass(final_compatibility)
        and _qwen3_qk_norm_hooks_pass(process_compatibility)
    )
    batch_token_contract = _batch_token_contract(config, scientific_context.world_size)
    checks["protocol_amendment"] = (
        formal_binding.get("protocol_amendment_id") == _AMENDMENT_ID
        and formal_binding.get("protocol_amendment_sha256")
        == scientific_context.protocol_amendment_sha256
        and formal_binding.get("reviewed_implementation_commit")
        == scientific_context.reviewed_implementation_commit
        and formal_binding.get("reviewed_implementation_commit")
        != scientific_context.code_commit
    )
    checks["batch_token_invariants"] = batch_token_contract == {
        "batch_partition_protocol": _BATCH_PARTITION_PROTOCOL,
        "requested_fsdp_sharding_strategy": REQUESTED_FSDP_SHARDING_STRATEGY,
        "effective_fsdp_sharding_strategy": effective_fsdp_sharding_strategy(
            scientific_context.world_size
        ),
        "global_logical_batch_size": _GLOBAL_BATCH_SIZE,
        "max_per_rank_microbatch_size": _MAX_MICROBATCH_SIZE,
        "max_model_input_length": 1536,
        "full_parameter_training": True,
        "prompt_population_size": _PROMPT_POPULATION_SIZE,
        "prompt_ids_unique": True,
        "accepted_view_prompt_order": "exactly_manifest_ordered_prompt_ids",
        "prompt_population_alignment": "exact_multiple_of_global_logical_batch_size",
        "microbatch_schedule_by_rank": _MICROBATCHES_BY_WORLD_SIZE[
            scientific_context.world_size
        ],
        "optimizer_microsteps": len(
            _MICROBATCHES_BY_WORLD_SIZE[scientific_context.world_size][0]
        ),
        "samples_by_rank": _SAMPLES_BY_WORLD_SIZE[scientific_context.world_size],
        "world_size": scientific_context.world_size,
        "token_budget": 2_000_000,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "max_optimizer_steps": 120,
    }
    resume_fsdp_contract = resume.get("fsdp_sharding_contract")
    checks["distributed_resume_fsdp_strategy"] = (
        isinstance(resume_fsdp_contract, dict)
        and resume_fsdp_contract.get("requested_fsdp_sharding_strategy")
        == batch_token_contract["requested_fsdp_sharding_strategy"]
        and resume_fsdp_contract.get("effective_fsdp_sharding_strategy")
        == batch_token_contract["effective_fsdp_sharding_strategy"]
        and type(resume_fsdp_contract.get("fsdp_wrapper_count")) is int
        and resume_fsdp_contract["fsdp_wrapper_count"]
        == preflight_fsdp_wrapper_count
    )
    if tuple(sorted(checks)) != G0_CHECK_NAMES or any(value is not True for value in checks.values()):
        raise ValueError("replayed G0 decision did not pass every exact scientific check")

    expected_metrics = {
        "base_accuracy": metrics.get("answer_accuracy"),
        "calibrated_accuracy": calibrated_metrics.get("answer_accuracy"),
        "teacher_topk_mass_minimum": teacher_mass.get("minimum"),
        "shortcut_gap": anti.get("shortcut_gap"),
        "iid_accuracy": anti.get("iid_accuracy"),
        "transformed_accuracy": anti.get("transformed_accuracy"),
        "label_leakage": label_leakage.get("metrics"),
        "teacher_readiness": teacher_readiness.get("metrics"),
        "stages": {
            "final_answer": {
                "bootstrap_spearman": final_noise[
                    "within_checkpoint_full_score_spearman"
                ],
                "selected_vs_random_cpr_margin": final_exact.get(
                    "selected_vs_matched_random_cpr_margin"
                ),
                "attribution_exact_spearman": final_exact.get(
                    "attribution_patching_spearman"
                ),
            },
            str(process_circuit.get("probe_stage")): {
                "bootstrap_spearman": process_noise[
                    "within_checkpoint_full_score_spearman"
                ],
                "selected_vs_random_cpr_margin": process_exact.get(
                    "selected_vs_matched_random_cpr_margin"
                ),
                "attribution_exact_spearman": process_exact.get(
                    "attribution_patching_spearman"
                ),
            },
        },
    }
    expected_artifact_hashes = {
        name: sha256_file(path) for name, path in paths.items()
    }
    expected_protocol = scientific_compatibility_fields(prereg_version)
    static_expected = {
        "format_version": 2,
        **expected_protocol,
        "phase": "G0",
        "protocol_track": str(config["protocol_track"]),
        "protocol_prereg_version": str(config["protocol_track"]),
        "artifact_namespace": str(config["model"]["artifact_namespace"]),
        "batch_token_contract": batch_token_contract,
        "prompt_protocol": str(config["model"]["prompt_protocol"]["name"]),
        "enable_thinking": False,
        "chat_template_sha256": str(
            config["model"]["prompt_protocol"]["chat_template_sha256"]
        ),
        "tokenizer_fingerprint": str(config["model"]["tokenizer_fingerprint"]),
        "passed": True,
        "checks": checks,
        "metrics": expected_metrics,
        "job_ids": [job_id],
        "git_commit": scientific_context.code_commit,
        "code_commit": scientific_context.code_commit,
        "prereg_commit": formal_binding["prereg_commit"],
        "prereg_path": formal_binding["prereg_path"],
        "prereg_sha256": formal_binding["prereg_sha256"],
        "protocol_amendment_git_commit": formal_binding[
            "protocol_amendment_git_commit"
        ],
        "protocol_amendment_id": formal_binding["protocol_amendment_id"],
        "protocol_amendment_sha256": formal_binding["protocol_amendment_sha256"],
        "reviewed_implementation_commit": formal_binding[
            "reviewed_implementation_commit"
        ],
        "request_git_commit": scientific_context.request_git_commit,
        "allocation_sha256": scientific_context.allocation_sha256,
        "world_size": scientific_context.world_size,
        "resolved_config_sha256": sha256_value(config),
        "model_revision": str(config["model"]["model_revision"]),
        "teacher_revision": str(config["teacher"]["model_revision"]),
        "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "artifact_hashes": expected_artifact_hashes,
    }
    if set(report) != G0_REPORT_FIELDS:
        raise ValueError("replayed inner G0 report fields differ from the exact schema")
    if any(report.get(key) != value for key, value in static_expected.items()):
        raise ValueError("replayed inner G0 report differs from reconstructed evidence")
    launch_environment = report.get("launch_environment")
    expected_launch_keys = {
        "MODEL_CONFIG",
        "TEACHER_CONFIG",
        "PRODUCTION_CONFIG",
        "G0_CONFIG",
        "PILOT_CONFIG",
        "PROJECT_ROOT",
        "PYTHON_BIN",
        "ACCELERATE_BIN",
        "OUTPUT_ROOT",
    }
    if not isinstance(launch_environment, dict) or set(launch_environment) != expected_launch_keys:
        raise ValueError("replayed inner G0 launch environment schema changed")
    if not isinstance(report.get("created_at"), str) or not report["created_at"]:
        raise ValueError("replayed inner G0 report has no creation timestamp")
    digest = report.get("sha256")
    if digest != sha256_value({key: value for key, value in report.items() if key != "sha256"}):
        raise ValueError("replayed inner G0 report SHA-256 mismatch")
    return checks


def main(
    argv: list[str] | None = None,
    *,
    scientific_context: ScientificInvocationContext | None = None,
) -> None:
    if not isinstance(scientific_context, ScientificInvocationContext):
        raise RuntimeError(
            "finalize_g0 requires the handler-owned scientific invocation context"
        )
    scientific_context.validate()
    parser = argparse.ArgumentParser(description="Finalize real Qwen G0")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--teacher-store-manifest", type=Path, required=True)
    parser.add_argument("--teacher-readiness", type=Path, required=True)
    parser.add_argument("--label-leakage", type=Path, required=True)
    parser.add_argument("--anti-shortcut", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--final-circuit", "--circuit", dest="final_circuit", type=Path, required=True)
    parser.add_argument(
        "--final-exact-patching",
        "--exact-patching",
        dest="final_exact_patching",
        type=Path,
        required=True,
    )
    parser.add_argument("--process-circuit", type=Path, required=True)
    parser.add_argument("--process-exact-patching", type=Path, required=True)
    parser.add_argument("--final-compatibility", type=Path, required=True)
    parser.add_argument("--process-compatibility", type=Path, required=True)
    parser.add_argument("--distributed-resume", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    formal_binding = formal_artifact_binding(config)
    git_commit = require_git_output(["rev-parse", "HEAD"])
    expected_trusted_binding = {
        "protocol_amendment_id": _AMENDMENT_ID,
        "protocol_amendment_sha256": scientific_context.protocol_amendment_sha256,
        "reviewed_implementation_commit": scientific_context.reviewed_implementation_commit,
    }
    if (
        git_commit != scientific_context.code_commit
        or any(
            formal_binding.get(name) != value
            for name, value in expected_trusted_binding.items()
        )
    ):
        raise ValueError("trusted G0 context differs from the accepted scientific binding")
    prereg_version = str(config["prereg_version"])
    base = validate_probe_score_artifact(args.base_scores)
    teacher = TrajectoryStore(args.teacher_store_manifest.parent).check_integrity()
    require_scientific_artifact(
        teacher,
        expected_prereg_version=prereg_version,
    )
    initial_checkpoint_hash = sha256_file(args.initial_checkpoint)
    label_leakage = validate_label_leakage_artifact(_read(args.label_leakage))
    teacher_readiness_raw = _read(args.teacher_readiness)
    tokenized_prefix = teacher_readiness_raw.get("tokenized_prefix_manifest", {})
    if not isinstance(tokenized_prefix, dict) or tokenized_prefix.get("sha256") != sha256_value(
        {key: value for key, value in tokenized_prefix.items() if key != "sha256"}
    ):
        raise ValueError("teacher-readiness tokenized prefix manifest is invalid")
    readiness_bindings = {
        **formal_binding,
        "teacher_model_id": str(config["teacher"]["model_name_or_path"]),
        "teacher_model_revision": str(config["teacher"]["model_revision"]),
        "student_model_id": str(config["model"]["model_name_or_path"]),
        "student_model_revision": str(config["model"]["model_revision"]),
        "student_tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "teacher_tokenizer_revision": str(config["teacher"]["tokenizer_revision"]),
        "tokenizer_revision": str(config["teacher"]["tokenizer_revision"]),
        "dataset_hash": str(label_leakage["dataset_hash"]),
        "prefix_probe_hash": str(teacher_readiness_raw["tokenized_prefix_manifest"]["sha256"]),
    }
    readiness_bindings.pop("teacher_revision")
    readiness_bindings.pop("model_revision")
    teacher_readiness = validate_teacher_readiness_artifact(
        teacher_readiness_raw,
        expected_bindings=readiness_bindings,
    )
    anti = validate_anti_shortcut_report(
        args.anti_shortcut,
        max_shortcut_gap=float(config["anti_shortcut"]["max_shortcut_gap"]),
        expected_model_checkpoint_hash=initial_checkpoint_hash,
    )
    probes = validate_probe_cohort_manifest(
        args.probe_manifest,
        expected_initial_checkpoint_hash=initial_checkpoint_hash,
    )
    final_circuit = _read(args.final_circuit)
    final_exact = _read(args.final_exact_patching)
    process_circuit = _read(args.process_circuit)
    process_exact = _read(args.process_exact_patching)
    for artifact in (final_circuit, final_exact, process_circuit, process_exact):
        require_scientific_artifact(
            artifact,
            expected_prereg_version=prereg_version,
            require_circuit_schema=True,
            require_hash=True,
        )
    final_compatibility = _read(args.final_compatibility)
    process_compatibility = _read(args.process_compatibility)
    resume = validate_distributed_resume_report(
        args.distributed_resume,
        config=config,
        expected_world_size=scientific_context.world_size,
    )
    gpu_preflight = _read(args.gpu_preflight)
    gpu_digest = gpu_preflight.pop("sha256", None)
    if gpu_digest != sha256_value(gpu_preflight) or gpu_preflight.get("passed") is not True:
        raise ValueError("G0 requires a passed, hash-valid GPU preflight")
    gpu_preflight["sha256"] = gpu_digest
    preflight_fsdp_wrapper_count = _preflight_fsdp_wrapper_count(
        gpu_preflight,
        world_size=scientific_context.world_size,
    )
    if str(config.get("protocol_track", "")).startswith("qwen3_"):
        teacher_readiness_formal = _teacher_readiness_formal_binding(teacher_readiness)
        bound_artifacts = {
            "base_scores": base,
            "teacher_store": teacher,
            "teacher_readiness": teacher_readiness_formal,
            "label_leakage": label_leakage,
            "anti_shortcut": anti,
            "probe_manifest": probes,
            "final_circuit": final_circuit,
            "final_exact_patching": final_exact,
            "process_circuit": process_circuit,
            "process_exact_patching": process_exact,
            "final_compatibility": final_compatibility,
            "process_compatibility": process_compatibility,
            "distributed_resume": resume,
            "gpu_preflight": gpu_preflight,
        }
        for artifact_name, artifact in bound_artifacts.items():
            if artifact_name == "gpu_preflight":
                continue
            _require_formal_binding(artifact, formal_binding, name=artifact_name)
    final_noise = estimate_estimator_noise_floor(
        final_circuit.get("bootstrap_score_vectors", []), activation_threshold=0.0
    )
    process_noise = estimate_estimator_noise_floor(
        process_circuit.get("bootstrap_score_vectors", []), activation_threshold=0.0
    )
    metrics = base.get("initial_validation_metrics", {})
    calibrated_metrics = base.get("calibrated_validation_metrics", {})
    teacher_mass = teacher.get("teacher_topk_mass", {})
    bank_total = int(teacher.get("total_trajectories", 0))
    bank_positive = int(teacher.get("reward_distribution", {}).get("positive", 0))
    checks = {
        "base_task_accuracy": float(metrics.get("answer_accuracy", 0.0))
        >= float(config["anti_shortcut"]["minimum_iid_accuracy"]),
        "teacher_topk_mass": float(teacher_mass.get("minimum", 0.0)) >= 0.90,
        "teacher_correctness": teacher_readiness.get("passed") is True,
        "label_leakage": label_leakage.get("passed") is True,
        "fixed_bank_mixed_rewards": 0 < bank_positive < bank_total,
        "calibration_anchor_improves_accuracy": float(calibrated_metrics.get("answer_accuracy", 0.0))
        > float(metrics.get("answer_accuracy", 0.0)),
        "anti_shortcut": anti.get("passed") is True,
        "base_capable_probes": probes["cohorts"]["base_capable"]["discovery"]["num_examples"] > 0
        and probes["cohorts"]["base_capable"]["validation"]["num_examples"] > 0,
        "probe_scoring_binding": base.get("initial_checkpoint_sha256") == initial_checkpoint_hash
        and probes.get("scoring_manifest_hash") == base.get("sha256")
        and probes.get("eligibility_evidence_ancestry")
        == base.get("eligibility_evidence_ancestry"),
        "hf_transformerlens_gqa_parity": _stage_compatibility_passes(
            final_compatibility,
            final_circuit,
        )
        and _stage_compatibility_passes(
            process_compatibility,
            process_circuit,
        ),
        "bootstrap_stability": min(
            float(final_noise["within_checkpoint_full_score_spearman"]),
            float(process_noise["within_checkpoint_full_score_spearman"]),
        )
        >= float(config["g0"]["minimum_bootstrap_spearman"]),
        "final_stage_eap_beats_matched_random": float(
            final_exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_selected_vs_random_cpr_margin"]),
        "process_stage_eap_beats_matched_random": float(
            process_exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_selected_vs_random_cpr_margin"]),
        "distinct_stage_manifests": final_circuit.get("probe_stage") == "final_answer"
        and process_circuit.get("probe_stage") in {"first_rule_selection", "intermediate_conclusion"}
        and final_circuit.get("stage_target_manifest_hash")
        != process_circuit.get("stage_target_manifest_hash"),
        "identity_sanity": all(
            artifact.get("sanity_checks", {}).get("identity_passed") is True
            for artifact in (final_exact, process_exact)
        ),
        "full_corruption_sanity": all(
            artifact.get("sanity_checks", {}).get("full_corruption_passed") is True
            for artifact in (final_exact, process_exact)
        ),
        "attribution_exact_calibration": all(
            float(artifact.get("attribution_patching_spearman", float("-inf")))
            >= float(config["g0"]["minimum_attribution_exact_spearman"])
            and float(artifact.get("attribution_patching_spearman_ci", {}).get("lower", float("-inf")))
            >= float(config["g0"]["minimum_spearman_bootstrap_lower_bound"])
            for artifact in (final_exact, process_exact)
        ),
        "distributed_checkpoint_resume": resume.get("format_version") == 2
        and resume.get("passed") is True
        and int(resume.get("world_size", 0)) == scientific_context.world_size
        and resume.get("max_optimizer_steps") == int(config["trainer"]["max_steps"])
        and isinstance(resume.get("checks"), dict)
        and bool(resume["checks"])
        and all(value is True for value in resume["checks"].values()),
        "split_probe_isolation": probes.get("frozen_before_training") is True,
        "artifact_reconstruction": all(
            circuit.get("checkpoint_sha256") == initial_checkpoint_hash
            and exact.get("artifacts", {}).get("checkpoint_sha256") == initial_checkpoint_hash
            for circuit, exact in (
                (final_circuit, final_exact),
                (process_circuit, process_exact),
            )
        ),
        "gpu_preflight": gpu_preflight.get("passed") is True
        and int(gpu_preflight.get("world_size", 0)) == scientific_context.world_size,
    }
    checks["gpu_preflight_binding"] = (
        gpu_preflight.get("git_commit")
        == scientific_context.gpu_preflight_git_commit
        and gpu_preflight.get("model_revision") == config["model"]["model_revision"]
        and gpu_preflight.get("teacher_revision") == config["teacher"]["model_revision"]
    )
    prereg_path = str(config["prereg_path"])
    prereg_commit = require_git_output(["log", "-n", "1", "--format=%H", "--", prereg_path])
    if str(config.get("protocol_track", "")).startswith("qwen3_"):
        qwen3_bindings = {
            "protocol_track": config["protocol_track"],
            "artifact_namespace": config["model"]["artifact_namespace"],
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": config["model"]["prompt_protocol"]["chat_template_sha256"],
            "tokenizer_fingerprint": config["model"]["tokenizer_fingerprint"],
        }
        checks["qwen3_protocol_bindings"] = all(
            all(artifact.get(key) == value for key, value in qwen3_bindings.items())
            for artifact in bound_artifacts.values()
        )
        checks["qwen3_qk_norm_hook_semantics"] = (
            _qwen3_qk_norm_hooks_pass(final_compatibility)
            and _qwen3_qk_norm_hooks_pass(process_compatibility)
        )
    batch_token_contract = _batch_token_contract(config, scientific_context.world_size)
    checks["protocol_amendment"] = (
        formal_binding.get("protocol_amendment_id") == _AMENDMENT_ID
        and isinstance(formal_binding.get("protocol_amendment_sha256"), str)
        and len(formal_binding["protocol_amendment_sha256"]) == 64
        and formal_binding.get("reviewed_implementation_commit") != git_commit
    )
    checks["batch_token_invariants"] = batch_token_contract == {
        "batch_partition_protocol": _BATCH_PARTITION_PROTOCOL,
        "requested_fsdp_sharding_strategy": REQUESTED_FSDP_SHARDING_STRATEGY,
        "effective_fsdp_sharding_strategy": effective_fsdp_sharding_strategy(
            scientific_context.world_size
        ),
        "global_logical_batch_size": _GLOBAL_BATCH_SIZE,
        "max_per_rank_microbatch_size": _MAX_MICROBATCH_SIZE,
        "max_model_input_length": 1536,
        "full_parameter_training": True,
        "prompt_population_size": _PROMPT_POPULATION_SIZE,
        "prompt_ids_unique": True,
        "accepted_view_prompt_order": "exactly_manifest_ordered_prompt_ids",
        "prompt_population_alignment": "exact_multiple_of_global_logical_batch_size",
        "microbatch_schedule_by_rank": _MICROBATCHES_BY_WORLD_SIZE[
            scientific_context.world_size
        ],
        "optimizer_microsteps": len(
            _MICROBATCHES_BY_WORLD_SIZE[scientific_context.world_size][0]
        ),
        "samples_by_rank": _SAMPLES_BY_WORLD_SIZE[scientific_context.world_size],
        "world_size": scientific_context.world_size,
        "token_budget": 2_000_000,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "max_optimizer_steps": 120,
    }
    resume_fsdp_contract = resume.get("fsdp_sharding_contract")
    checks["distributed_resume_fsdp_strategy"] = (
        isinstance(resume_fsdp_contract, dict)
        and resume_fsdp_contract.get("requested_fsdp_sharding_strategy")
        == batch_token_contract["requested_fsdp_sharding_strategy"]
        and resume_fsdp_contract.get("effective_fsdp_sharding_strategy")
        == batch_token_contract["effective_fsdp_sharding_strategy"]
        and type(resume_fsdp_contract.get("fsdp_wrapper_count")) is int
        and resume_fsdp_contract["fsdp_wrapper_count"]
        == preflight_fsdp_wrapper_count
    )
    if tuple(sorted(checks)) != G0_CHECK_NAMES:
        raise RuntimeError("G0 decision check schema differs from the reviewed contract")
    payload: dict[str, Any] = {
        "format_version": 2,
        **scientific_compatibility_fields(prereg_version),
        "phase": "G0",
        "protocol_track": str(config.get("protocol_track", "core_v2")),
        "protocol_prereg_version": str(config.get("protocol_track", "core_v2")),
        "artifact_namespace": str(config["model"].get("artifact_namespace", "legacy")),
        "batch_token_contract": batch_token_contract,
        "prompt_protocol": str(config["model"].get("prompt_protocol", {}).get("name", "legacy_raw_v1")),
        "enable_thinking": False,
        "chat_template_sha256": str(
            config["model"].get("prompt_protocol", {}).get("chat_template_sha256", "legacy-unrecorded")
        ),
        "tokenizer_fingerprint": str(config["model"].get("tokenizer_fingerprint", "legacy-unrecorded")),
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "base_accuracy": metrics.get("answer_accuracy"),
            "calibrated_accuracy": calibrated_metrics.get("answer_accuracy"),
            "teacher_topk_mass_minimum": teacher_mass.get("minimum"),
            "shortcut_gap": anti.get("shortcut_gap"),
            "iid_accuracy": anti.get("iid_accuracy"),
            "transformed_accuracy": anti.get("transformed_accuracy"),
            "label_leakage": label_leakage.get("metrics"),
            "teacher_readiness": teacher_readiness.get("metrics"),
            "stages": {
                "final_answer": {
                    "bootstrap_spearman": final_noise["within_checkpoint_full_score_spearman"],
                    "selected_vs_random_cpr_margin": final_exact.get("selected_vs_matched_random_cpr_margin"),
                    "attribution_exact_spearman": final_exact.get("attribution_patching_spearman"),
                },
                str(process_circuit.get("probe_stage")): {
                    "bootstrap_spearman": process_noise["within_checkpoint_full_score_spearman"],
                    "selected_vs_random_cpr_margin": process_exact.get(
                        "selected_vs_matched_random_cpr_margin"
                    ),
                    "attribution_exact_spearman": process_exact.get("attribution_patching_spearman"),
                },
            },
        },
        "job_ids": [args.job_id],
        "git_commit": git_commit,
        "code_commit": git_commit,
        "prereg_commit": prereg_commit,
        "prereg_path": prereg_path,
        "prereg_sha256": sha256_file(Path(prereg_path)),
        "protocol_amendment_git_commit": formal_binding[
            "protocol_amendment_git_commit"
        ],
        "protocol_amendment_id": formal_binding["protocol_amendment_id"],
        "protocol_amendment_sha256": formal_binding["protocol_amendment_sha256"],
        "reviewed_implementation_commit": formal_binding[
            "reviewed_implementation_commit"
        ],
        "request_git_commit": scientific_context.request_git_commit,
        "allocation_sha256": scientific_context.allocation_sha256,
        "world_size": scientific_context.world_size,
        "resolved_config_sha256": sha256_value(config),
        "model_revision": str(config["model"]["model_revision"]),
        "teacher_revision": str(config["teacher"]["model_revision"]),
        "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
        "launch_environment": {
            name: os.environ.get(name)
            for name in (
                "MODEL_CONFIG",
                "TEACHER_CONFIG",
                "PRODUCTION_CONFIG",
                "G0_CONFIG",
                "PILOT_CONFIG",
                "PROJECT_ROOT",
                "PYTHON_BIN",
                "ACCELERATE_BIN",
                "OUTPUT_ROOT",
            )
        },
        "artifact_hashes": {
            name: sha256_file(path)
            for name, path in {
                "initial_checkpoint": args.initial_checkpoint,
                "gpu_preflight": args.gpu_preflight,
                "base_scores": args.base_scores,
                "teacher_store_manifest": args.teacher_store_manifest,
                "teacher_readiness": args.teacher_readiness,
                "label_leakage": args.label_leakage,
                "anti_shortcut": args.anti_shortcut,
                "probe_manifest": args.probe_manifest,
                "final_circuit": args.final_circuit,
                "final_exact_patching": args.final_exact_patching,
                "process_circuit": args.process_circuit,
                "process_exact_patching": args.process_exact_patching,
                "final_compatibility": args.final_compatibility,
                "process_compatibility": args.process_compatibility,
                "distributed_resume": args.distributed_resume,
            }.items()
        },
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)
    readiness_evidence = {
        "base_task_accuracy_nontrivial": (
            checks["base_task_accuracy"],
            f"formal validation accuracy={metrics.get('answer_accuracy')}",
        ),
        "pilot_improves_accuracy": (
            checks["calibration_anchor_improves_accuracy"],
            f"calibration={calibrated_metrics.get('answer_accuracy')} base={metrics.get('answer_accuracy')}",
        ),
        "verifier_deterministic": (
            checks["anti_shortcut"],
            "anti-shortcut semantic preservation and exact-verifier artifacts passed",
        ),
        "fixed_bank_mixed_rewards": (
            0 < bank_positive < bank_total,
            f"positive={bank_positive} total={bank_total}",
        ),
        "teacher_topk_mass_acceptable": (
            checks["teacher_topk_mass"],
            f"minimum retained mass={teacher_mass.get('minimum')}",
        ),
        "hf_circuit_logit_parity": (
            checks["hf_transformerlens_gqa_parity"],
            "stage compatibility artifacts="
            f"{sha256_file(args.final_compatibility)},"
            f"{sha256_file(args.process_compatibility)}",
        ),
        "eap_ig_beats_random": (
            checks["final_stage_eap_beats_matched_random"]
            and checks["process_stage_eap_beats_matched_random"],
            "both final-answer and process-stage selected circuits beat matched random controls",
        ),
        "exact_patching_distinguishes_groups": (
            checks["distinct_stage_manifests"],
            "distinct frozen target manifests supply stage-specific functional evidence",
        ),
        "attribution_bootstrap_stable": (
            checks["bootstrap_stability"],
            "both stage-specific within-checkpoint Spearman thresholds passed",
        ),
        "checkpoint_resume_verified": (
            checks["distributed_checkpoint_resume"],
            f"resume artifact={sha256_file(args.distributed_resume)}",
        ),
        "split_leakage_absent": (
            checks["split_probe_isolation"],
            f"probe manifest={probes['sha256']}",
        ),
        "anti_shortcut_gap": (
            checks["anti_shortcut"],
            f"shortcut gap={anti.get('shortcut_gap')}",
        ),
        "probe_cohorts_frozen": (
            checks["base_capable_probes"],
            f"probe manifest={probes['sha256']}",
        ),
        "teacher_correctness": (
            checks["teacher_correctness"],
            f"teacher readiness artifact={sha256_file(args.teacher_readiness)}",
        ),
        "label_leakage": (
            checks["label_leakage"],
            f"label leakage artifact={sha256_file(args.label_leakage)}",
        ),
    }
    readiness = build_readiness_report(
        readiness_evidence,
        bindings={
            **formal_binding,
            "initial_checkpoint_hash": initial_checkpoint_hash,
            "dataset_hash": str(anti["dataset_hash"]),
            "suite_hash": str(anti["suite_hash"]),
            "code_commit": git_commit,
            "prereg_commit": prereg_commit,
            "prereg_version": prereg_version,
        },
    )
    readiness.write(args.output.parent / "readiness")
    report = args.output.with_suffix(".md")
    report.write_text(
        "# G0 report\n\n"
        f"Result: **{'PASS' if payload['passed'] else 'FAIL'}**\n\n"
        + "\n".join(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
        + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise SystemExit("G0 failed; pilot launch is forbidden")


if __name__ == "__main__":
    main()
