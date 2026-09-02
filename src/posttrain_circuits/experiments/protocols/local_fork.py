"""Immutable shared-state local-fork comparison protocol."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.compatibility import scientific_compatibility_fields


_SOURCE_BINDING_FIELDS = {
    "schema_version",
    "manifest_path",
    "manifest_file_sha256",
    "manifest_payload_sha256",
    "experiment_binding",
    "experiment_binding_sha256",
    "factorial_design_sha256",
    "method_id",
    "seed",
    "train_dataset_sha256",
    "validation_dataset_sha256",
    "initial_checkpoint_sha256",
    "final_checkpoint_path",
    "final_checkpoint_sha256",
    "checkpoint_state_hashes",
    "effective_bundle_state_hashes",
    "optimizer_parameter_names",
    "trajectory_bank_sha256",
}

_SOURCE_CHECKPOINT_STATE_FIELDS = {
    "model",
    "model_parameter_names",
    "trainable_parameter_names",
    "optimizer",
    "optimizer_moments",
    "optimizer_parameter_references",
    "optimizer_named",
    "scheduler",
    "rng",
}

_EFFECTIVE_BUNDLE_STATE_FIELDS = {
    "model",
    "model_parameter_names",
    "trainable_parameter_names",
    "optimizer",
    "optimizer_moments",
    "optimizer_parameter_names",
    "optimizer_named",
    "scheduler",
    "rng",
}

_FORMAL_ARTIFACT_BINDING_FIELDS = {
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
}

_FORMAL_BINDING_FIELDS = _FORMAL_ARTIFACT_BINDING_FIELDS | {"local_fork_source"}

_RESULT_CELL_FIELDS = {
    "branch",
    "horizon",
    "primary_comparison_axis",
    "secondary_comparison_axes",
    "unmatched",
    "matched",
    "calibration_trace",
    "calibration",
}

_BRANCH_RESULT_FIELDS = {
    "bundle_id",
    "bundle_manifest_sha256",
    "local_fork_protocol_id",
    "local_fork_spec_sha256",
    "trajectory_bank_sha256",
    "branch",
    "horizon",
    "comparison_mode",
    "calibration_round",
    "initial_hashes",
    "initial_parameter_hash",
    "initial_optimizer_moment_hash",
    "initial_scheduler_hash",
    "initial_rng_hash",
    "initial_trajectory_hash",
    "probe_input_hash",
    "probe_attention_mask_hash",
    "initial_probe_output_hash",
    "probe_kl_valid_token_count",
    "optimizer_updates",
    "step_losses",
    "step_metrics",
    "group_membership_hash",
    "loss",
    "parameter_update_norm",
    "probe_output_kl_new_to_fork",
    "learning_rate",
    "checkpoint_evidence",
    "post_model_hash",
    "post_optimizer_hash",
    "post_optimizer_moment_hash",
    "post_scheduler_hash",
    "post_rng_hash",
}

_CALIBRATION_FIELDS = {
    "target_output_kl_new_to_fork",
    "observed_unmatched_output_kl",
    "matched_output_kl",
    "matched_absolute_error",
    "matched_relative_error",
    "within_tolerance",
    "relative_tolerance",
    "calibration_rounds",
    "learning_rate",
    "matched_parameter_update_norm",
}

_CALIBRATION_TRACE_FIELDS = {
    "round",
    "learning_rate",
    "target_output_kl_new_to_fork",
    "observed_output_kl_new_to_fork",
    "absolute_error",
    "relative_error",
    "within_tolerance",
    "result",
}

_INVALID_CELL_FIELDS = {
    "branch",
    "horizon",
    "target_output_kl",
    "observed_output_kl",
    "relative_error",
}

_BRANCH_METRIC_FIELDS = {
    "hard_teacher": (
        {"hard_ce_loss", "teacher_entropy", "teacher_topk_mass"},
        set(),
    ),
    "soft_teacher": (
        {
            "soft_kl_loss",
            "teacher_entropy",
            "teacher_topk_mass",
            "student_mass_on_teacher_topk",
            "teacher_student_topk_overlap",
        },
        set(),
    ),
    "verified_replay": (
        {"verified_replay_loss", "reward_rate"},
        {
            "generated_trajectories",
            "successful_trajectories",
            "effective_positive_sequences",
            "effective_supervised_tokens",
            "retry_count",
        },
    ),
    "centered_policy_gradient": (
        {
            "centered_policy_gradient_loss",
            "positive_advantage_count",
            "negative_advantage_count",
            "mean_probability_ratio",
        },
        set(),
    ),
}

_REPORT_BASE_FIELDS = {
    "format_version",
    "prereg_version",
    "generator_version",
    "label_semantics",
    "circuit_probe_schema_version",
    "bundle",
    "bundle_file",
    "bundle_id",
    "bundle_manifest_sha256",
    "local_fork_protocol_id",
    "local_fork_spec_sha256",
    "requested_horizons",
    "requested_output_kl_relative_tolerance",
    "requested_maximum_calibration_rounds",
    "full_protocol_conformant",
    "results",
    "valid_for_primary_analysis",
    "invalid_cells",
    "sha256",
}


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def validate_local_fork_source_binding(payload: Any) -> dict[str, Any]:
    """Validate the immutable source run/checkpoint identity carried by a formal fork."""

    if not isinstance(payload, Mapping) or set(payload) != _SOURCE_BINDING_FIELDS:
        observed = set(payload) if isinstance(payload, Mapping) else set()
        raise ValueError(
            "local-fork source binding fields differ from schema: "
            f"missing={sorted(_SOURCE_BINDING_FIELDS - observed)}, "
            f"extra={sorted(observed - _SOURCE_BINDING_FIELDS)}"
        )
    source = dict(payload)
    if type(source.get("schema_version")) is not int or source.get("schema_version") != 1:
        raise ValueError("unsupported local-fork source binding schema version")
    for name in ("manifest_path", "final_checkpoint_path"):
        value = source.get(name)
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(f"local-fork source binding requires an absolute {name}")
    for name in (
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "experiment_binding_sha256",
        "factorial_design_sha256",
        "train_dataset_sha256",
        "validation_dataset_sha256",
        "initial_checkpoint_sha256",
        "final_checkpoint_sha256",
        "trajectory_bank_sha256",
    ):
        _require_sha256(source.get(name), name=f"local_fork_source.{name}")
    raw_experiment = source.get("experiment_binding")
    if not isinstance(raw_experiment, Mapping):
        raise ValueError("local-fork source binding lacks its ExperimentBinding payload")
    from posttrain_circuits.experiments.protocols.specs import (
        validate_experiment_binding_payload,
    )

    binding = validate_experiment_binding_payload(
        raw_experiment,
        expected_sha256=source["experiment_binding_sha256"],
    )
    expected_summary = {
        "factorial_design_sha256": binding.factorial_design_sha256,
        "method_id": binding.method_id,
        "seed": binding.seed,
        "train_dataset_sha256": binding.train_dataset_sha256,
        "validation_dataset_sha256": binding.validation_dataset_sha256,
        "initial_checkpoint_sha256": binding.initial_checkpoint_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": source.get(key)}
        for key, value in expected_summary.items()
        if source.get(key) != value
    }
    if mismatches:
        raise ValueError(f"local-fork source ExperimentBinding was relabeled: {mismatches}")
    if source.get("method_id") != LOCAL_FORK_SPEC.source_method_id or source.get(
        "seed"
    ) != LOCAL_FORK_SPEC.source_seed:
        raise ValueError(
            "local-fork source differs from the registered canonical_sft/seed-42 anchor"
        )
    checkpoint_state = source.get("checkpoint_state_hashes")
    effective_state = source.get("effective_bundle_state_hashes")
    for name, value, expected_fields in (
        ("checkpoint_state_hashes", checkpoint_state, _SOURCE_CHECKPOINT_STATE_FIELDS),
        ("effective_bundle_state_hashes", effective_state, _EFFECTIVE_BUNDLE_STATE_FIELDS),
    ):
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise ValueError(f"local-fork source {name} is partial or has extra fields")
        for key, digest in value.items():
            _require_sha256(digest, name=f"local_fork_source.{name}.{key}")
    optimizer_parameter_names = source.get("optimizer_parameter_names")
    if not isinstance(optimizer_parameter_names, list) or not all(
        isinstance(group, list)
        and all(isinstance(name, str) and name for name in group)
        for group in optimizer_parameter_names
    ):
        raise ValueError("local-fork source optimizer parameter mapping is malformed")
    flattened_names = [name for group in optimizer_parameter_names for name in group]
    if not flattened_names or len(flattened_names) != len(set(flattened_names)):
        raise ValueError("local-fork source optimizer parameter mapping is not one-to-one")
    if sha256_value(optimizer_parameter_names) != effective_state[
        "optimizer_parameter_names"
    ]:
        raise ValueError("local-fork source optimizer parameter mapping hash changed")
    for key in (
        "model",
        "model_parameter_names",
        "trainable_parameter_names",
        "optimizer_named",
        "scheduler",
        "rng",
    ):
        if checkpoint_state[key] != effective_state[key]:
            raise ValueError(
                f"local-fork source checkpoint/effective bundle state differs: {key}"
            )
    return source


def validate_local_fork_formal_binding(payload: Any) -> dict[str, Any]:
    """Reject partial, extended, or mistyped formal LocalFork bindings."""

    if not isinstance(payload, Mapping):
        raise ValueError("local-fork formal binding must be a mapping")
    binding = dict(payload)
    if not binding:
        return binding
    if set(binding) != _FORMAL_BINDING_FIELDS:
        raise ValueError(
            "local-fork formal binding fields differ from schema: "
            f"missing={sorted(_FORMAL_BINDING_FIELDS - set(binding))}, "
            f"extra={sorted(set(binding) - _FORMAL_BINDING_FIELDS)}"
        )
    if type(binding.get("enable_thinking")) is not bool:
        raise ValueError("local-fork formal enable_thinking must be a boolean")
    for name in _FORMAL_ARTIFACT_BINDING_FIELDS - {"enable_thinking"}:
        value = binding.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"local-fork formal {name} must be a non-empty string")
    for name in ("tokenizer_fingerprint", "chat_template_sha256", "prereg_sha256"):
        _require_sha256(binding[name], name=f"local-fork formal {name}")
    validate_local_fork_source_binding(binding["local_fork_source"])
    return binding


@dataclass(frozen=True, slots=True)
class LocalForkBranchSpec:
    branch_id: str
    objective: str
    reward_signal: str
    reward_usage: str
    uses_policy_gradient: bool

    def __post_init__(self) -> None:
        if not self.branch_id or not self.objective:
            raise ValueError("local-fork branch ID and objective must be non-empty")
        if self.reward_usage not in {"none", "selection_gate", "policy_gradient"}:
            raise ValueError(f"unsupported local-fork reward usage: {self.reward_usage}")
        if self.uses_policy_gradient != (self.reward_usage == "policy_gradient"):
            raise ValueError("local-fork policy-gradient flag differs from reward usage")
        if self.reward_usage == "none" and self.reward_signal != "none":
            raise ValueError("reward-free local-fork branches cannot expose a reward signal")
        if self.reward_usage != "none" and self.reward_signal == "none":
            raise ValueError("gated/PG local-fork branches require a reward signal")


@dataclass(frozen=True, slots=True)
class LocalForkSpec:
    protocol_id: str
    source_method_id: str
    source_seed: int
    branches: tuple[LocalForkBranchSpec, ...]
    horizons: tuple[int, ...]
    primary_horizon: int
    minimum_group_size: int
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
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported LocalForkSpec schema version")
        if self.source_method_id != "canonical_sft" or self.source_seed != 42:
            raise ValueError("formal local-fork source must be canonical_sft/seed-42")
        if len({branch.branch_id for branch in self.branches}) != len(self.branches):
            raise ValueError("local-fork branch IDs must be unique")
        if tuple(branch.branch_id for branch in self.branches) != (
            "hard_teacher",
            "soft_teacher",
            "verified_replay",
            "centered_policy_gradient",
        ):
            raise ValueError("local-fork branches differ from the registered four-way contrast")
        if self.horizons != (1, 5, 20) or self.primary_horizon != 20:
            raise ValueError("local-fork horizons differ from the frozen 1/5/20 protocol")
        if self.minimum_group_size < 4 or self.require_within_group_reward_variance is not True:
            raise ValueError("local-fork PG identification requires groups of four with variance")
        if not self.require_frozen_group_advantages or not self.require_old_policy_logprobs:
            raise ValueError("centered PG must freeze advantages and bind old-policy logprobs")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("local-fork clip epsilon must lie in (0, 1)")
        if self.divergence != "KL(output_new || output_fork)":
            raise ValueError("local-fork divergence orientation is part of the scientific contract")
        if self.probe_kl_mask_protocol != "all_nonpadding_model_facing_prompt_positions_v1":
            raise ValueError("local-fork KL must exclude right-padding positions")
        if self.primary_comparison_axis != "matched_output_kl_new_to_fork":
            raise ValueError("local-fork primary axis must be matched output KL")
        if self.secondary_comparison_axes != ("update_count", "parameter_update_norm"):
            raise ValueError("local-fork secondary comparison axes changed")
        if not 0.0 < self.output_kl_relative_tolerance < 1.0:
            raise ValueError("local-fork relative tolerance must lie in (0, 1)")
        if self.maximum_calibration_rounds < 0:
            raise ValueError("local-fork calibration rounds must be non-negative")
        if (
            not 0.0 < self.calibration_minimum_learning_rate < 1.0
            or self.calibration_maximum_scale < 1.0
            or not 0.0 < self.calibration_numeric_tolerance < 1.0
        ):
            raise ValueError("local-fork calibration numeric constants are invalid")
        if self.out_of_tolerance_action != "invalidate_cell_and_exit_nonzero":
            raise ValueError("out-of-tolerance local-fork cells must fail closed")

    @property
    def branch_ids(self) -> tuple[str, ...]:
        return tuple(branch.branch_id for branch in self.branches)

    @property
    def scientific_sha256(self) -> str:
        return sha256_value(asdict(self))

    def validate_experiment_config(self, experiment: Mapping[str, Any]) -> None:
        expected = {
            "name": "local_fork",
            "protocol_id": self.protocol_id,
            "source_method_id": self.source_method_id,
            "source_seed": self.source_seed,
            "branches": list(self.branch_ids),
            "minimum_group_size": self.minimum_group_size,
            "require_within_group_reward_variance": self.require_within_group_reward_variance,
            "require_frozen_group_advantages": self.require_frozen_group_advantages,
            "require_old_policy_logprobs": self.require_old_policy_logprobs,
            "clip_epsilon": self.clip_epsilon,
            "horizons": list(self.horizons),
            "primary_horizon": self.primary_horizon,
            "primary_match_axis": "output_kl_new_to_fork",
            "divergence": self.divergence,
            "probe_kl_mask_protocol": self.probe_kl_mask_protocol,
            "secondary_axes": list(self.secondary_comparison_axes),
            "output_kl_relative_tolerance": self.output_kl_relative_tolerance,
            "maximum_calibration_rounds": self.maximum_calibration_rounds,
            "calibration_minimum_learning_rate": self.calibration_minimum_learning_rate,
            "calibration_maximum_scale": self.calibration_maximum_scale,
            "calibration_numeric_tolerance": self.calibration_numeric_tolerance,
            "out_of_tolerance_action": self.out_of_tolerance_action,
        }
        mismatches = {
            name: {"expected": value, "observed": experiment.get(name)}
            for name, value in expected.items()
            if experiment.get(name) != value
        }
        if mismatches:
            raise ValueError(f"local-fork config differs from LocalForkSpec: {mismatches}")


LOCAL_FORK_SPEC = LocalForkSpec(
    protocol_id="shared_state_matched_output_kl_v1",
    source_method_id="canonical_sft",
    source_seed=42,
    branches=(
        LocalForkBranchSpec(
            "hard_teacher",
            "teacher_top1_response_token_cross_entropy",
            "none",
            "none",
            False,
        ),
        LocalForkBranchSpec(
            "soft_teacher",
            "teacher_topk_forward_kl",
            "none",
            "none",
            False,
        ),
        LocalForkBranchSpec(
            "verified_replay",
            "sequence_normalized_nll_on_exact_verifier_successes",
            "exact",
            "selection_gate",
            False,
        ),
        LocalForkBranchSpec(
            "centered_policy_gradient",
            "frozen_group_advantage_old_policy_clipped_surrogate",
            "exact",
            "policy_gradient",
            True,
        ),
    ),
    horizons=(1, 5, 20),
    primary_horizon=20,
    minimum_group_size=4,
    require_within_group_reward_variance=True,
    require_frozen_group_advantages=True,
    require_old_policy_logprobs=True,
    clip_epsilon=0.2,
    divergence="KL(output_new || output_fork)",
    probe_kl_mask_protocol="all_nonpadding_model_facing_prompt_positions_v1",
    primary_comparison_axis="matched_output_kl_new_to_fork",
    secondary_comparison_axes=("update_count", "parameter_update_norm"),
    output_kl_relative_tolerance=0.25,
    maximum_calibration_rounds=2,
    calibration_minimum_learning_rate=1e-8,
    calibration_maximum_scale=10.0,
    calibration_numeric_tolerance=1e-12,
    out_of_tolerance_action="invalidate_cell_and_exit_nonzero",
)


def validate_local_fork_report(
    report: Mapping[str, Any],
    *,
    checkpoint_root: Path,
    bundle_path: Path,
    require_full_protocol: bool = False,
    require_source_binding: bool = False,
    expected_source_manifest_path: Path | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_source_experiment_binding_sha256: str | None = None,
    source_model_factory: Any | None = None,
    expected_trajectory_bank_manifest_path: Path | None = None,
    expected_prompt_manifest_path: Path | None = None,
    expected_probe_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute result/bundle/spec linkage instead of trusting report booleans."""

    if not isinstance(report, Mapping):
        raise ValueError("local-fork report must be a mapping")
    payload = dict(report)
    expected_report_hash = payload.pop("sha256", None)
    _require_sha256(expected_report_hash, name="local-fork report sha256")
    if expected_report_hash != sha256_value(payload):
        raise ValueError("local-fork report SHA-256 mismatch")
    if type(report.get("format_version")) is not int or report.get("format_version") != 4:
        raise ValueError("local-fork report is not format version 4")
    bundle = report.get("bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError("local-fork report lacks its exact bundle manifest")
    bundle_content = {
        key: value
        for key, value in bundle.items()
        if key not in {"bundle_id", "manifest_sha256"}
    }
    observed_bundle_hash = sha256_value(bundle_content)
    if bundle.get("manifest_sha256") != observed_bundle_hash:
        raise ValueError("local-fork report embedded bundle manifest SHA-256 mismatch")
    if bundle.get("bundle_id") != "fork-" + observed_bundle_hash[:16]:
        raise ValueError("local-fork report embedded bundle ID mismatch")
    required_bundle = {
        "format_version": 4,
        "protocol_id": LOCAL_FORK_SPEC.protocol_id,
        "local_fork_spec_sha256": LOCAL_FORK_SPEC.scientific_sha256,
        "source_method_id": LOCAL_FORK_SPEC.source_method_id,
        "source_seed": LOCAL_FORK_SPEC.source_seed,
        "minimum_group_size": LOCAL_FORK_SPEC.minimum_group_size,
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
    bundle_mismatches = {}
    for key, value in required_bundle.items():
        observed = bundle.get(key)
        normalized = tuple(observed) if isinstance(value, tuple) and isinstance(observed, list) else observed
        if normalized != value:
            bundle_mismatches[key] = {"expected": value, "observed": observed}
    if bundle_mismatches:
        raise ValueError(f"local-fork report bundle protocol changed: {bundle_mismatches}")
    if report.get("local_fork_spec_sha256") != bundle.get("local_fork_spec_sha256"):
        raise ValueError("local-fork report spec hash is not inherited from its bundle")
    if report.get("local_fork_protocol_id") != bundle.get("protocol_id"):
        raise ValueError("local-fork report protocol ID is not inherited from its bundle")
    if report.get("bundle_id") != bundle.get("bundle_id"):
        raise ValueError("local-fork report bundle ID mismatch")
    if report.get("bundle_manifest_sha256") != bundle.get("manifest_sha256"):
        raise ValueError("local-fork report bundle manifest hash mismatch")
    from posttrain_circuits.learning.training.local_fork import (
        _probe_kl,
        recompute_parameter_update_norm,
        validate_bundle_probe_recomputation,
        validate_bundle_input_evidence_files,
        validate_branch_checkpoint_evidence,
        validate_branch_deterministic_replay,
        validate_checkpoint_probe_recomputation,
        validate_bundle_file_evidence,
        validate_checkpoint_tree_exact,
        validate_local_fork_source_binding_files,
    )

    bundle_file = report.get("bundle_file")
    current_bundle_payload = validate_bundle_file_evidence(
        bundle_file,
        expected_path=bundle_path,
        embedded_manifest=dict(bundle),
    )
    if source_model_factory is None or not callable(source_model_factory):
        raise ValueError(
            "local-fork validation requires a model factory for exact probe-logit "
            "recomputation"
        )
    recomputation_model = source_model_factory()

    def recomputation_model_factory() -> Any:
        return recomputation_model

    validate_bundle_probe_recomputation(
        current_bundle_payload,
        model_factory=recomputation_model_factory,
    )
    if require_source_binding:
        if (
            expected_trajectory_bank_manifest_path is None
            or expected_prompt_manifest_path is None
            or expected_probe_manifest_path is None
        ):
            raise ValueError(
                "formal local-fork validation requires externally selected bank/prompt/probe inputs"
            )
        validate_bundle_input_evidence_files(
            current_bundle_payload,
            expected_trajectory_bank_manifest_path=expected_trajectory_bank_manifest_path,
            expected_prompt_manifest_path=expected_prompt_manifest_path,
            expected_probe_manifest_path=expected_probe_manifest_path,
        )
    formal_binding = bundle.get("formal_binding")
    formal_binding = validate_local_fork_formal_binding(formal_binding)
    expected_report_fields = _REPORT_BASE_FIELDS | set(formal_binding)
    if set(report) != expected_report_fields:
        raise ValueError(
            "local-fork report fields differ from format v4: "
            f"missing={sorted(expected_report_fields - set(report))}, "
            f"extra={sorted(set(report) - expected_report_fields)}"
        )
    expected_compatibility = scientific_compatibility_fields(
        str(formal_binding.get("prereg_version", report.get("prereg_version", "")))
    )
    compatibility_mismatches = {
        key: {"expected": value, "observed": report.get(key)}
        for key, value in expected_compatibility.items()
        if report.get(key) != value
    }
    if compatibility_mismatches:
        raise ValueError(
            "local-fork report scientific compatibility changed: "
            f"{compatibility_mismatches}"
        )
    formal_mismatches = {
        key: {"expected": value, "observed": report.get(key)}
        for key, value in formal_binding.items()
        if report.get(key) != value
    }
    if formal_mismatches:
        raise ValueError(f"local-fork report relabeled its formal binding: {formal_mismatches}")
    source_binding = formal_binding.get("local_fork_source")
    if source_binding is None:
        if require_source_binding:
            raise ValueError("formal local-fork report lacks its source run/checkpoint binding")
    else:
        validated_source = validate_local_fork_source_binding_files(
            source_binding,
            bundle_payload=current_bundle_payload,
        )
        source_experiment = validated_source["experiment_binding"]
        if require_source_binding:
            if (
                expected_source_manifest_path is None
                or expected_source_manifest_sha256 is None
                or expected_source_experiment_binding_sha256 is None
            ):
                raise ValueError(
                    "formal local-fork validation requires an externally selected source anchor"
                )
            selected_source = Path(expected_source_manifest_path).resolve(strict=True)
            observed_source = Path(validated_source["manifest_path"]).resolve(strict=True)
            source_anchor_expected = {
                "manifest_path": str(selected_source),
                "manifest_file_sha256": expected_source_manifest_sha256,
                "experiment_binding_sha256": expected_source_experiment_binding_sha256,
                "method_id": LOCAL_FORK_SPEC.source_method_id,
                "seed": LOCAL_FORK_SPEC.source_seed,
            }
            source_anchor_observed = {
                **validated_source,
                "manifest_path": str(observed_source),
            }
            source_anchor_mismatches = {
                key: {"expected": value, "observed": source_anchor_observed.get(key)}
                for key, value in source_anchor_expected.items()
                if source_anchor_observed.get(key) != value
            }
            if source_anchor_mismatches:
                raise ValueError(
                    "local-fork source differs from the externally selected pilot source: "
                    f"{source_anchor_mismatches}"
                )
        source_formal_expected = {
            "protocol_track": formal_binding.get("protocol_track"),
            "preregistration_sha256": formal_binding.get("prereg_sha256"),
            "implementation_commit": formal_binding.get("code_commit"),
            "implementation_dirty": False,
            "model_requested_revision": formal_binding.get("model_revision"),
            "teacher_requested_revision": formal_binding.get("teacher_revision"),
            "tokenizer_requested_revision": formal_binding.get("tokenizer_revision"),
            "tokenizer_fingerprint": formal_binding.get("tokenizer_fingerprint"),
            "chat_template_sha256": formal_binding.get("chat_template_sha256"),
            "prompt_protocol": formal_binding.get("prompt_protocol"),
            "enable_thinking": formal_binding.get("enable_thinking"),
        }
        source_formal_mismatches = {
            key: {"expected": value, "observed": source_experiment.get(key)}
            for key, value in source_formal_expected.items()
            if source_experiment.get(key) != value
        }
        if source_formal_mismatches:
            raise ValueError(
                "local-fork source binding differs from its formal protocol/model identity: "
                f"{source_formal_mismatches}"
            )
        bundle_model_spec = current_bundle_payload.get("model_spec")
        if not isinstance(bundle_model_spec, Mapping):
            raise ValueError("formal local-fork bundle lacks its resolved model specification")
        source_model_expected = {
            "seed": bundle_model_spec.get("seed"),
            "protocol_track": bundle_model_spec.get("protocol_track"),
            "model_id": bundle_model_spec.get("model_name_or_path"),
            "model_requested_revision": bundle_model_spec.get("model_revision"),
            "model_resolved_revision": bundle_model_spec.get("resolved_model_commit"),
            "tokenizer_id": bundle_model_spec.get("tokenizer_name_or_path"),
            "tokenizer_requested_revision": bundle_model_spec.get("tokenizer_revision"),
            "tokenizer_resolved_revision": bundle_model_spec.get(
                "resolved_tokenizer_commit"
            ),
            "tokenizer_fingerprint": bundle_model_spec.get(
                "resolved_tokenizer_fingerprint"
            ),
            "prompt_protocol": bundle_model_spec.get("resolved_prompt_protocol"),
            "chat_template_sha256": bundle_model_spec.get(
                "resolved_chat_template_sha256"
            ),
        }
        source_model_mismatches = {
            key: {"expected": value, "observed": source_experiment.get(key)}
            for key, value in source_model_expected.items()
            if source_experiment.get(key) != value
        }
        if source_model_mismatches:
            raise ValueError(
                "local-fork source ExperimentBinding differs from bundled model identity: "
                f"{source_model_mismatches}"
            )
    requested_horizons = report.get("requested_horizons")
    if not isinstance(requested_horizons, list) or not requested_horizons or any(
        type(value) is not int for value in requested_horizons
    ):
        raise ValueError("local-fork requested horizons must be an integer list")
    if len(set(requested_horizons)) != len(requested_horizons) or any(
        value not in LOCAL_FORK_SPEC.horizons for value in requested_horizons
    ):
        raise ValueError("local-fork report contains unregistered/duplicate horizons")
    if requested_horizons != [
        value for value in LOCAL_FORK_SPEC.horizons if value in set(requested_horizons)
    ]:
        raise ValueError("local-fork requested horizons are not in registered order")
    tolerance = report.get("requested_output_kl_relative_tolerance")
    rounds_limit = report.get("requested_maximum_calibration_rounds")
    if type(tolerance) is not float or tolerance != LOCAL_FORK_SPEC.output_kl_relative_tolerance:
        raise ValueError("local-fork report tolerance differs from LocalForkSpec")
    if type(rounds_limit) is not int or rounds_limit != LOCAL_FORK_SPEC.maximum_calibration_rounds:
        raise ValueError("local-fork report calibration limit differs from LocalForkSpec")
    for field in ("full_protocol_conformant", "valid_for_primary_analysis"):
        if type(report.get(field)) is not bool:
            raise ValueError(f"local-fork report {field} must be a boolean")
    full_requested = (
        tuple(requested_horizons) == LOCAL_FORK_SPEC.horizons
    )

    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("local-fork report results must be a list")
    expected_cells = {
        (branch, horizon)
        for horizon in requested_horizons
        for branch in LOCAL_FORK_SPEC.branch_ids
    }
    observed_cells: set[tuple[str, int]] = set()
    observed_cell_order: list[tuple[str, int]] = []
    checkpoint_paths: set[str] = set()
    recomputed_post_payloads: set[str] = set()
    replayed_transitions: set[tuple[str, str]] = set()
    hard_targets: dict[int, float] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("local-fork result cell must be a mapping")
        if set(result) != _RESULT_CELL_FIELDS:
            raise ValueError("local-fork result cell is partial or has extra fields")
        branch = result.get("branch")
        horizon = result.get("horizon")
        if not isinstance(branch, str) or type(horizon) is not int:
            raise ValueError("local-fork result cell has invalid branch/horizon types")
        cell = (branch, horizon)
        if cell not in expected_cells:
            raise ValueError(f"unregistered local-fork result cell: {cell}")
        if cell in observed_cells:
            raise ValueError(f"duplicate local-fork result cell: {cell}")
        observed_cells.add(cell)
        observed_cell_order.append(cell)
        if result.get("primary_comparison_axis") != LOCAL_FORK_SPEC.primary_comparison_axis:
            raise ValueError("local-fork result changed its primary comparison axis")
        if not isinstance(result.get("secondary_comparison_axes"), list) or result.get(
            "secondary_comparison_axes"
        ) != list(LOCAL_FORK_SPEC.secondary_comparison_axes):
            raise ValueError("local-fork result changed its secondary comparison axes")
        unmatched = result.get("unmatched")
        matched = result.get("matched")
        calibration = result.get("calibration")
        if not all(isinstance(value, Mapping) for value in (unmatched, matched, calibration)):
            raise ValueError("local-fork result cell lacks unmatched/matched/calibration evidence")
        if set(calibration) != _CALIBRATION_FIELDS:
            raise ValueError("local-fork calibration evidence is partial or has extra fields")
        if type(calibration.get("within_tolerance")) is not bool:
            raise ValueError("local-fork calibration within_tolerance must be a boolean")
        calibration_trace = result.get("calibration_trace")
        if not isinstance(calibration_trace, list):
            raise ValueError("local-fork calibration trace must be a list")
        phase_rows: list[tuple[str, Mapping[str, Any]]] = [
            ("unmatched", unmatched),
            ("matched", matched),
        ]
        for index, trace in enumerate(calibration_trace, start=1):
            if not isinstance(trace, Mapping) or set(trace) != _CALIBRATION_TRACE_FIELDS:
                raise ValueError("local-fork calibration trace is partial or has extra fields")
            trace_result = trace.get("result")
            if not isinstance(trace_result, Mapping):
                raise ValueError("local-fork calibration trace lacks its branch result")
            if type(trace.get("round")) is not int or trace.get("round") != index:
                raise ValueError("local-fork calibration trace rounds are not contiguous")
            if type(trace.get("within_tolerance")) is not bool:
                raise ValueError("local-fork calibration trace tolerance flag must be boolean")
            phase_rows.append((f"calibration-round-{index}", trace_result))
        for phase_name, phase in phase_rows:
            if set(phase) != _BRANCH_RESULT_FIELDS:
                raise ValueError(
                    f"local-fork {phase_name} result is partial or has extra fields"
                )
            linkage = {
                "bundle_id": bundle.get("bundle_id"),
                "bundle_manifest_sha256": bundle.get("manifest_sha256"),
                "local_fork_protocol_id": LOCAL_FORK_SPEC.protocol_id,
                "local_fork_spec_sha256": LOCAL_FORK_SPEC.scientific_sha256,
                "branch": branch,
                "horizon": horizon,
                "probe_attention_mask_hash": bundle.get("probe_attention_mask_hash"),
                "group_membership_hash": bundle.get("group_membership_hash"),
                "trajectory_bank_sha256": bundle.get("manifest_hashes", {}).get("bank"),
            }
            phase_mismatches = {
                key: {"expected": value, "observed": phase.get(key)}
                for key, value in linkage.items()
                if phase.get(key) != value
            }
            if phase_mismatches:
                raise ValueError(
                    f"local-fork {phase_name} result lost bundle/spec linkage: {phase_mismatches}"
                )
            expected_initial_hashes = {
                "model": bundle.get("checkpoint_hash"),
                "optimizer": bundle.get("optimizer_hash"),
                "optimizer_moments": bundle.get("optimizer_moment_hash"),
                "scheduler": bundle.get("scheduler_hash"),
                "rng": bundle.get("rng_hash"),
                "prompts": bundle.get("prompt_hash"),
                "trajectories": bundle.get("trajectory_hash"),
                "probe_inputs": bundle.get("probe_input_hash"),
                "probe_attention_mask": bundle.get("probe_attention_mask_hash"),
                "probe_outputs": bundle.get("probe_output_hash"),
            }
            if phase.get("initial_hashes") != expected_initial_hashes:
                raise ValueError(f"local-fork {phase_name} branch did not start from the bundle")
            direct_initial = {
                "initial_parameter_hash": bundle.get("checkpoint_hash"),
                "initial_optimizer_moment_hash": bundle.get("optimizer_moment_hash"),
                "initial_scheduler_hash": bundle.get("scheduler_hash"),
                "initial_rng_hash": bundle.get("rng_hash"),
                "initial_trajectory_hash": bundle.get("trajectory_hash"),
                "probe_input_hash": bundle.get("probe_input_hash"),
                "initial_probe_output_hash": bundle.get("probe_output_hash"),
            }
            direct_mismatches = {
                key: {"expected": value, "observed": phase.get(key)}
                for key, value in direct_initial.items()
                if phase.get(key) != value
            }
            if direct_mismatches:
                raise ValueError(
                    f"local-fork {phase_name} initial-state evidence changed: {direct_mismatches}"
                )
            checkpoint_evidence = phase.get("checkpoint_evidence")
            if not isinstance(checkpoint_evidence, Mapping) or set(checkpoint_evidence) != {
                "pre",
                "post",
            }:
                raise ValueError(
                    f"local-fork {phase_name} checkpoint evidence is partial or has extra phases"
                )
            pre_checkpoint = validate_branch_checkpoint_evidence(
                checkpoint_evidence["pre"],
                checkpoint_root=checkpoint_root,
                bundle_payload=current_bundle_payload,
                branch=branch,
                horizon=horizon,
                phase="pre",
                comparison_mode=str(phase.get("comparison_mode", "")),
                calibration_round=phase.get("calibration_round"),
            )
            post_checkpoint = validate_branch_checkpoint_evidence(
                checkpoint_evidence["post"],
                checkpoint_root=checkpoint_root,
                bundle_payload=current_bundle_payload,
                branch=branch,
                horizon=horizon,
                phase="post",
                comparison_mode=str(phase.get("comparison_mode", "")),
                calibration_round=phase.get("calibration_round"),
            )
            transition_identity = (
                str(checkpoint_evidence["pre"]["payload_state_sha256"]),
                str(checkpoint_evidence["post"]["payload_state_sha256"]),
            )
            if transition_identity not in replayed_transitions:
                validate_branch_deterministic_replay(
                    pre_checkpoint,
                    post_checkpoint,
                    bundle_payload=current_bundle_payload,
                    model_factory=recomputation_model_factory,
                )
                replayed_transitions.add(transition_identity)
            post_payload_identity = str(
                checkpoint_evidence["post"]["payload_state_sha256"]
            )
            if post_payload_identity not in recomputed_post_payloads:
                validate_checkpoint_probe_recomputation(
                    post_checkpoint,
                    model_factory=recomputation_model_factory,
                )
                recomputed_post_payloads.add(post_payload_identity)
            checkpoint_paths.update(
                {
                    str(checkpoint_evidence["pre"]["path"]),
                    str(checkpoint_evidence["post"]["path"]),
                }
            )
            if pre_checkpoint["model_parameter_names"] != post_checkpoint[
                "model_parameter_names"
            ]:
                raise ValueError("local-fork model parameter mapping changed across the branch")
            if phase.get("learning_rate") != pre_checkpoint.get("learning_rate") or (
                phase.get("learning_rate") != post_checkpoint.get("learning_rate")
            ):
                raise ValueError("local-fork checkpoint learning rate differs from its result")
            post_hash_mismatches = {
                key: {"expected": value, "observed": phase.get(key)}
                for key, value in {
                    "post_model_hash": post_checkpoint["state_hashes"]["model"],
                    "post_optimizer_hash": post_checkpoint["state_hashes"]["optimizer"],
                    "post_optimizer_moment_hash": post_checkpoint["state_hashes"][
                        "optimizer_moments"
                    ],
                    "post_scheduler_hash": post_checkpoint["state_hashes"]["scheduler"],
                    "post_rng_hash": post_checkpoint["state_hashes"]["rng"],
                }.items()
                if phase.get(key) != value
            }
            if post_hash_mismatches:
                raise ValueError(
                    "local-fork post-checkpoint state differs from its result: "
                    f"{post_hash_mismatches}"
                )
            recomputed_parameter_norm = recompute_parameter_update_norm(
                pre_checkpoint["model"],
                post_checkpoint["model"],
                pre_checkpoint["model_parameter_names"],
            )
            recomputed_probe_kl = _probe_kl(
                pre_checkpoint["pre_update_outputs"],
                post_checkpoint["observed_probe_outputs"],
                pre_checkpoint["probe_attention_mask"],
            )
            numeric_tolerance = LOCAL_FORK_SPEC.calibration_numeric_tolerance
            if not math.isclose(
                float(phase.get("parameter_update_norm", float("nan"))),
                recomputed_parameter_norm,
                rel_tol=numeric_tolerance,
                abs_tol=numeric_tolerance,
            ):
                raise ValueError(
                    "local-fork parameter update norm differs from pre/post checkpoint bytes"
                )
            if not math.isclose(
                float(phase.get("probe_output_kl_new_to_fork", float("nan"))),
                recomputed_probe_kl,
                rel_tol=numeric_tolerance,
                abs_tol=numeric_tolerance,
            ):
                raise ValueError("local-fork probe KL differs from checkpoint logits/mask")
            pre_steps = pre_checkpoint["optimizer_steps"]
            post_steps = post_checkpoint["optimizer_steps"]
            if set(pre_steps) != set(post_steps) or any(
                post_steps[name] - pre_steps[name] != horizon for name in pre_steps
            ):
                raise ValueError("local-fork AdamW step increment differs from its horizon")
            loss_evidence = post_checkpoint["loss_evidence"]
            if (
                phase.get("step_losses") != loss_evidence["step_losses"]
                or phase.get("step_metrics") != loss_evidence["step_metrics"]
                or phase.get("loss") != loss_evidence["final_loss"]
                or phase.get("loss") != phase.get("step_losses", [None])[-1]
            ):
                raise ValueError("local-fork reported loss differs from checkpoint loss evidence")
            if (
                type(phase.get("optimizer_updates")) is not int
                or phase["optimizer_updates"] != horizon
            ):
                raise ValueError("local-fork branch update count differs from its horizon")
            losses = phase.get("step_losses")
            metrics = phase.get("step_metrics")
            if not isinstance(losses, list) or len(losses) != horizon:
                raise ValueError("local-fork step-loss count differs from its horizon")
            if not isinstance(metrics, list) or len(metrics) != horizon:
                raise ValueError("local-fork step-metric count differs from its horizon")
            if any(not isinstance(row, Mapping) for row in metrics):
                raise ValueError("local-fork step metrics must be mappings")
            float_metric_fields, integer_metric_fields = _BRANCH_METRIC_FIELDS[branch]
            expected_metric_fields = float_metric_fields | integer_metric_fields
            for row in metrics:
                if set(row) != expected_metric_fields:
                    raise ValueError("local-fork step-metric fields differ from branch schema")
                if any(
                    type(row[field]) is not float or not math.isfinite(row[field])
                    for field in float_metric_fields
                ) or any(type(row[field]) is not int for field in integer_metric_fields):
                    raise ValueError("local-fork step-metric types differ from branch schema")
            numeric_values = (
                *losses,
                phase.get("loss"),
                phase.get("parameter_update_norm"),
                phase.get("probe_output_kl_new_to_fork"),
                phase.get("learning_rate"),
            )
            if any(
                type(value) is not float or not math.isfinite(value)
                for value in numeric_values
            ):
                raise ValueError("local-fork branch numeric evidence must use finite floats")
            if float(phase["parameter_update_norm"]) < 0.0:
                raise ValueError("local-fork parameter update norm cannot be negative")
            if float(phase["probe_output_kl_new_to_fork"]) < 0.0:
                raise ValueError("local-fork probe KL cannot be negative")
            if float(phase["learning_rate"]) <= 0.0:
                raise ValueError("local-fork branch learning rate must be positive")
            valid_tokens = phase.get("probe_kl_valid_token_count")
            expected_valid_tokens = int(pre_checkpoint["probe_attention_mask"].sum().item())
            if type(valid_tokens) is not int or valid_tokens != expected_valid_tokens:
                raise ValueError("local-fork KL valid-token count differs from its exact mask")
        rounds = calibration.get("calibration_rounds")
        if type(rounds) is not int or not 0 <= rounds <= rounds_limit:
            raise ValueError("local-fork calibration rounds exceed the requested bound")
        if len(calibration_trace) != rounds:
            raise ValueError("local-fork calibration trace length differs from its round count")
        if unmatched.get("comparison_mode") != "unmatched" or unmatched.get(
            "calibration_round"
        ) != 0:
            raise ValueError("local-fork unmatched result changed its comparison ancestry")
        if rounds == 0:
            if matched != unmatched:
                raise ValueError(
                    "zero-round local-fork matching must reuse the exact unmatched result"
                )
        else:
            if matched.get("comparison_mode") != "matched" or matched.get(
                "calibration_round"
            ) != rounds:
                raise ValueError("local-fork matched result changed its calibration ancestry")
            if calibration_trace[-1]["result"] != matched:
                raise ValueError("local-fork matched result is not the final calibration round")
            previous_lr = float(unmatched["learning_rate"])
            previous_observed_kl = float(
                unmatched["probe_output_kl_new_to_fork"]
            )
            for index, trace in enumerate(calibration_trace, start=1):
                trace_result = trace["result"]
                if trace_result.get("comparison_mode") != "matched" or trace_result.get(
                    "calibration_round"
                ) != index:
                    raise ValueError("local-fork calibration trace changed branch ancestry")
                target_kl = trace.get("target_output_kl_new_to_fork")
                observed_kl = trace.get("observed_output_kl_new_to_fork")
                trace_lr = trace.get("learning_rate")
                if any(
                    type(value) is not float or not math.isfinite(value)
                    for value in (
                        target_kl,
                        observed_kl,
                        trace_lr,
                        trace.get("absolute_error"),
                        trace.get("relative_error"),
                    )
                ):
                    raise ValueError("local-fork calibration trace requires finite floats")
                if float(target_kl) <= 0.0 or float(observed_kl) < 0.0 or float(trace_lr) <= 0.0:
                    raise ValueError("local-fork calibration trace has an invalid numeric domain")
                if previous_observed_kl <= 0.0:
                    raise ValueError(
                        "local-fork calibration cannot continue from zero observed KL"
                    )
                scale = math.sqrt(float(target_kl) / previous_observed_kl)
                scale = min(
                    LOCAL_FORK_SPEC.calibration_maximum_scale,
                    max(1.0 / LOCAL_FORK_SPEC.calibration_maximum_scale, scale),
                )
                expected_next_lr = max(
                    LOCAL_FORK_SPEC.calibration_minimum_learning_rate,
                    previous_lr * scale,
                )
                if not math.isclose(
                    float(trace_lr),
                    expected_next_lr,
                    rel_tol=LOCAL_FORK_SPEC.calibration_numeric_tolerance,
                    abs_tol=LOCAL_FORK_SPEC.calibration_numeric_tolerance,
                ):
                    raise ValueError(
                        "local-fork calibration learning rate differs from the registered "
                        "previous_lr * sqrt(target/observed) rule"
                    )
                relative_error = (float(observed_kl) - float(target_kl)) / float(target_kl)
                trace_expected = {
                    "learning_rate": trace_result["learning_rate"],
                    "observed_output_kl_new_to_fork": trace_result[
                        "probe_output_kl_new_to_fork"
                    ],
                    "absolute_error": float(observed_kl) - float(target_kl),
                    "relative_error": relative_error,
                    "within_tolerance": abs(relative_error) <= tolerance,
                }
                trace_mismatches = {
                    key: {"expected": value, "observed": trace.get(key)}
                    for key, value in trace_expected.items()
                    if trace.get(key) != value
                }
                if trace_mismatches:
                    raise ValueError(
                        "local-fork calibration trace equations changed: "
                        f"{trace_mismatches}"
                    )
                previous_lr = float(trace_lr)
                previous_observed_kl = float(observed_kl)
            if any(bool(trace["within_tolerance"]) for trace in calibration_trace[:-1]):
                raise ValueError("local-fork calibration continued after reaching tolerance")
            if calibration_trace[-1]["within_tolerance"] is not True and rounds != rounds_limit:
                raise ValueError("local-fork calibration stopped early while outside tolerance")
        if branch == "hard_teacher":
            hard_targets[horizon] = float(unmatched["probe_output_kl_new_to_fork"])
    if observed_cells != expected_cells:
        raise ValueError(
            "local-fork branch/horizon matrix is incomplete: "
            f"missing={sorted(expected_cells - observed_cells)}, "
            f"extra={sorted(observed_cells - expected_cells)}"
        )
    expected_cell_order = [
        (branch, horizon)
        for horizon in requested_horizons
        for branch in LOCAL_FORK_SPEC.branch_ids
    ]
    if observed_cell_order != expected_cell_order:
        raise ValueError("local-fork result cells are not in registered branch/horizon order")
    validate_checkpoint_tree_exact(
        checkpoint_root,
        expected_paths=checkpoint_paths,
    )

    invalid_cells = []
    for result in results:
        branch = str(result["branch"])
        horizon = int(result["horizon"])
        unmatched = result["unmatched"]
        matched = result["matched"]
        calibration = result["calibration"]
        target = hard_targets[horizon]
        observed_unmatched = float(unmatched["probe_output_kl_new_to_fork"])
        observed_matched = float(matched["probe_output_kl_new_to_fork"])
        if target <= 0.0:
            raise ValueError("local-fork hard-teacher reference KL must be positive")
        for trace in result["calibration_trace"]:
            if trace.get("target_output_kl_new_to_fork") != target:
                raise ValueError("local-fork calibration trace changed the hard-teacher target")
        relative_error = (observed_matched - target) / target
        expected_calibration = {
            "target_output_kl_new_to_fork": target,
            "observed_unmatched_output_kl": observed_unmatched,
            "matched_output_kl": observed_matched,
            "matched_absolute_error": observed_matched - target,
            "matched_relative_error": relative_error,
            "within_tolerance": abs(relative_error) <= tolerance,
            "relative_tolerance": tolerance,
            "learning_rate": matched["learning_rate"],
            "matched_parameter_update_norm": matched["parameter_update_norm"],
        }
        float_calibration_fields = _CALIBRATION_FIELDS - {
            "within_tolerance",
            "calibration_rounds",
        }
        if any(
            type(calibration.get(key)) is not float
            or not math.isfinite(calibration[key])
            for key in float_calibration_fields
        ):
            raise ValueError("local-fork calibration numeric fields must be finite floats")
        for key, value in expected_calibration.items():
            if calibration.get(key) != value:
                raise ValueError(
                    f"local-fork calibration equation changed for {(branch, horizon)}: {key}"
                )
        for key in ("learning_rate", "matched_parameter_update_norm"):
            value = calibration.get(key)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or float(value) < 0.0
            ):
                raise ValueError(f"local-fork calibration has invalid {key}")
        if float(calibration["learning_rate"]) <= 0.0:
            raise ValueError("local-fork calibration learning rate must be positive")
        rounds = calibration.get("calibration_rounds")
        if calibration.get("within_tolerance") is not True:
            invalid_cells.append((branch, horizon))
    valid = not invalid_cells
    if report.get("valid_for_primary_analysis") is not valid:
        raise ValueError("local-fork primary-analysis flag differs from recomputed calibration")
    full_conformant = full_requested and valid
    if report.get("full_protocol_conformant") is not full_conformant:
        raise ValueError("local-fork full-protocol flag differs from recomputed validity")
    if require_full_protocol and not full_conformant:
        raise ValueError("formal analysis requires a valid complete local-fork protocol")
    reported_invalid = report.get("invalid_cells")
    if not isinstance(reported_invalid, list):
        raise ValueError("local-fork invalid_cells must be a list")
    expected_invalid_rows = []
    by_cell = {(str(row["branch"]), int(row["horizon"])): row for row in results}
    for branch, horizon in invalid_cells:
        row = by_cell[(branch, horizon)]
        target = hard_targets[horizon]
        observed = float(row["matched"]["probe_output_kl_new_to_fork"])
        expected_invalid_rows.append(
            {
                "branch": branch,
                "horizon": horizon,
                "target_output_kl": target,
                "observed_output_kl": observed,
                "relative_error": (observed - target) / target,
            }
        )
    for row in reported_invalid:
        if not isinstance(row, Mapping) or set(row) != _INVALID_CELL_FIELDS:
            raise ValueError("local-fork invalid-cell row is partial or has extra fields")
        if not isinstance(row.get("branch"), str) or type(row.get("horizon")) is not int:
            raise ValueError("local-fork invalid-cell row has invalid identity types")
        if any(
            type(row.get(key)) is not float or not math.isfinite(row[key])
            for key in ("target_output_kl", "observed_output_kl", "relative_error")
        ):
            raise ValueError("local-fork invalid-cell row has non-finite numeric evidence")
    if reported_invalid != expected_invalid_rows:
        raise ValueError("local-fork invalid-cell evidence differs from recomputed calibration")
    return dict(report)


__all__ = [
    "LOCAL_FORK_SPEC",
    "LocalForkBranchSpec",
    "LocalForkSpec",
    "validate_local_fork_formal_binding",
    "validate_local_fork_report",
    "validate_local_fork_source_binding",
]
