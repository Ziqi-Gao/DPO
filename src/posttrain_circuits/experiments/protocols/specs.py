"""Factorial design and hash-bound experiment comparability contracts."""

from __future__ import annotations

import base64
import binascii
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.checkpoints import torch_state_hash
from posttrain_circuits.artifacts.config_bindings import ConfigBinding, validate_config_binding
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.methods.registry import (
    FACTORIAL_METHOD_IDS,
    FACTORIAL_METHOD_SPECS,
    PILOT_METHOD_IDS,
    get_method_spec,
)

TOKEN_BUDGET_UNIT = "global_nonpadding_model_input_tokens_processed"
PLANNED_CONTRASTS = (
    ("online_hard", "offline_hard"),
    ("online_soft_opd", "offline_soft"),
    ("online_verified_replay", "offline_verified_replay"),
    ("online_soft_opd", "online_hard"),
    ("online_soft_opd", "online_verified_replay"),
    ("online_verified_replay", "canonical_grpo"),
)
MATCHED_ACCURACY_PAIRS = (
    ("offline_soft", "online_soft_opd"),
    ("online_hard", "online_soft_opd"),
    ("online_verified_replay", "canonical_grpo"),
)


def _require_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_text(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_int(value: int, *, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _require_bool(value: bool, *, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _optional_sha256(value: str | None, *, name: str) -> None:
    if value is not None:
        _require_sha256(value, name=name)


@dataclass(frozen=True, slots=True)
class ExperimentBinding:
    """Complete immutable scientific identity for one training start."""

    experiment_id: str
    method_id: str
    method_spec_sha256: str
    factorial_design_sha256: str
    scientific_config_sha256: str
    seed: int
    protocol_track: str
    preregistration_sha256: str
    implementation_commit: str
    implementation_dirty: bool
    model_id: str
    model_requested_revision: str
    model_resolved_revision: str
    teacher_id: str
    teacher_requested_revision: str
    teacher_resolved_revision: str
    tokenizer_id: str
    tokenizer_requested_revision: str
    tokenizer_resolved_revision: str
    tokenizer_fingerprint: str
    prompt_protocol: str
    chat_template_sha256: str
    enable_thinking: bool
    model_facing_prompt_schedule_sha256: str
    probe_manifest_sha256: str
    task_protocol_sha256: str
    state_source_protocol_sha256: str
    supervision_protocol_sha256: str
    response_mask_protocol: str
    max_completion_length: int
    full_parameter_training: bool
    initial_checkpoint_sha256: str
    train_dataset_sha256: str
    validation_dataset_sha256: str
    optimizer_spec_sha256: str
    scheduler_spec_sha256: str
    backend_id: str
    backend_version: str
    backend_batch_contract: str
    resolved_batch_contract_sha256: str
    token_budget: int
    token_budget_unit: str = TOKEN_BUDGET_UNIT
    teacher_soft_objective: str | None = None
    teacher_topk_mode: str | None = None
    teacher_top_k: int | None = None
    teacher_topk_include_eos: bool | None = None
    teacher_topk_minimum_retained_mass: float | None = None
    teacher_topk_fail_below_minimum_retained_mass: bool | None = None
    offline_bank_manifest_sha256: str | None = None
    offline_bank_content_sha256: str | None = None
    offline_bank_initial_cursor_sha256: str | None = None
    teacher_demo_manifest_sha256: str | None = None
    teacher_demo_content_sha256: str | None = None
    teacher_demo_exact_verifier_gate: bool | None = None
    teacher_demo_generation_sha256: str | None = None
    reward_calibration_manifest_sha256: str | None = None
    reward_calibration_content_sha256: str | None = None
    schema_version: int = 3

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 3:
            raise ValueError("unsupported ExperimentBinding schema version")
        for value, name in (
            (self.experiment_id, "experiment_id"),
            (self.method_id, "method_id"),
            (self.protocol_track, "protocol_track"),
            (self.implementation_commit, "implementation_commit"),
            (self.model_id, "model_id"),
            (self.model_requested_revision, "model_requested_revision"),
            (self.model_resolved_revision, "model_resolved_revision"),
            (self.teacher_id, "teacher_id"),
            (self.teacher_requested_revision, "teacher_requested_revision"),
            (self.teacher_resolved_revision, "teacher_resolved_revision"),
            (self.tokenizer_id, "tokenizer_id"),
            (self.tokenizer_requested_revision, "tokenizer_requested_revision"),
            (self.tokenizer_resolved_revision, "tokenizer_resolved_revision"),
            (self.prompt_protocol, "prompt_protocol"),
            (self.response_mask_protocol, "response_mask_protocol"),
            (self.backend_id, "backend_id"),
            (self.backend_version, "backend_version"),
            (self.backend_batch_contract, "backend_batch_contract"),
        ):
            _require_text(value, name=name)
        _require_int(self.seed, name="seed")
        _require_int(self.token_budget, name="token_budget", minimum=1)
        _require_int(self.max_completion_length, name="max_completion_length", minimum=1)
        _require_bool(self.enable_thinking, name="enable_thinking")
        _require_bool(self.full_parameter_training, name="full_parameter_training")
        _require_bool(self.implementation_dirty, name="implementation_dirty")
        if self.token_budget_unit != TOKEN_BUDGET_UNIT:
            raise ValueError(f"unsupported token budget unit: {self.token_budget_unit!r}")
        for value, name in (
            (self.method_spec_sha256, "method_spec_sha256"),
            (self.factorial_design_sha256, "factorial_design_sha256"),
            (self.scientific_config_sha256, "scientific_config_sha256"),
            (self.preregistration_sha256, "preregistration_sha256"),
            (self.initial_checkpoint_sha256, "initial_checkpoint_sha256"),
            (self.train_dataset_sha256, "train_dataset_sha256"),
            (self.validation_dataset_sha256, "validation_dataset_sha256"),
            (self.tokenizer_fingerprint, "tokenizer_fingerprint"),
            (self.chat_template_sha256, "chat_template_sha256"),
            (
                self.model_facing_prompt_schedule_sha256,
                "model_facing_prompt_schedule_sha256",
            ),
            (self.probe_manifest_sha256, "probe_manifest_sha256"),
            (self.task_protocol_sha256, "task_protocol_sha256"),
            (self.state_source_protocol_sha256, "state_source_protocol_sha256"),
            (self.supervision_protocol_sha256, "supervision_protocol_sha256"),
            (self.optimizer_spec_sha256, "optimizer_spec_sha256"),
            (self.scheduler_spec_sha256, "scheduler_spec_sha256"),
            (self.resolved_batch_contract_sha256, "resolved_batch_contract_sha256"),
        ):
            _require_sha256(value, name=name)
        method = get_method_spec(self.method_id)
        expected_method_hash = method.scientific_sha256
        if self.method_spec_sha256 != expected_method_hash:
            raise ValueError(
                f"method_spec_sha256 does not bind {self.method_id}: "
                f"expected={expected_method_hash}, observed={self.method_spec_sha256}"
            )
        if (
            self.backend_id != method.training_backend
            or self.backend_version != method.training_backend_version
            or self.backend_batch_contract != method.training_batch_contract
        ):
            raise ValueError("experiment backend identity differs from its MethodSpec")
        for value, name in (
            (self.offline_bank_manifest_sha256, "offline_bank_manifest_sha256"),
            (self.offline_bank_content_sha256, "offline_bank_content_sha256"),
            (self.offline_bank_initial_cursor_sha256, "offline_bank_initial_cursor_sha256"),
            (self.teacher_demo_manifest_sha256, "teacher_demo_manifest_sha256"),
            (self.teacher_demo_content_sha256, "teacher_demo_content_sha256"),
            (self.teacher_demo_generation_sha256, "teacher_demo_generation_sha256"),
            (
                self.reward_calibration_manifest_sha256,
                "reward_calibration_manifest_sha256",
            ),
            (
                self.reward_calibration_content_sha256,
                "reward_calibration_content_sha256",
            ),
        ):
            _optional_sha256(value, name=name)
        offline_values = (
            self.offline_bank_manifest_sha256,
            self.offline_bank_content_sha256,
            self.offline_bank_initial_cursor_sha256,
        )
        if method.requires_common_rollout_bank != all(value is not None for value in offline_values):
            raise ValueError(f"{self.method_id} common-bank binding differs from its MethodSpec")
        if any(value is not None for value in offline_values) and not all(
            value is not None for value in offline_values
        ):
            raise ValueError("offline bank manifest/content/cursor bindings are atomic")
        demo_values = (
            self.teacher_demo_manifest_sha256,
            self.teacher_demo_content_sha256,
            self.teacher_demo_generation_sha256,
        )
        if method.requires_teacher_demo_store != all(value is not None for value in demo_values):
            raise ValueError(f"{self.method_id} teacher-demo binding differs from its MethodSpec")
        if any(value is not None for value in demo_values) and not all(
            value is not None for value in demo_values
        ):
            raise ValueError("teacher-demo manifest/content/generation bindings are atomic")
        if method.requires_teacher_demo_store:
            if self.teacher_demo_exact_verifier_gate is not True:
                raise ValueError("canonical SFT must bind an exact-verifier-gated demo store")
        elif self.teacher_demo_exact_verifier_gate is not None:
            raise ValueError("non-SFT methods cannot claim a teacher-demo verifier gate")
        random_control = method.role == "control" and method.reward_signal == "matched_random"
        calibration_values = (
            self.reward_calibration_manifest_sha256,
            self.reward_calibration_content_sha256,
        )
        if random_control != all(value is not None for value in calibration_values):
            raise ValueError(
                "only grpo_random_reward binds the frozen matched-random calibration artifact"
            )
        if any(value is not None for value in calibration_values) and not all(
            value is not None for value in calibration_values
        ):
            raise ValueError("reward calibration manifest/content bindings are atomic")
        soft = method.soft_teacher_objective
        soft_values = (
            self.teacher_soft_objective,
            self.teacher_topk_mode,
            self.teacher_top_k,
            self.teacher_topk_include_eos,
            self.teacher_topk_minimum_retained_mass,
            self.teacher_topk_fail_below_minimum_retained_mass,
        )
        if (soft is not None) != all(value is not None for value in soft_values):
            raise ValueError("soft-teacher objective/top-k provenance is incomplete")
        if any(value is not None for value in soft_values) and not all(
            value is not None for value in soft_values
        ):
            raise ValueError("soft-teacher objective/top-k fields are atomic")
        if soft is not None:
            _require_int(self.teacher_top_k, name="teacher_top_k", minimum=1)  # type: ignore[arg-type]
            _require_bool(
                self.teacher_topk_include_eos,  # type: ignore[arg-type]
                name="teacher_topk_include_eos",
            )
            if type(self.teacher_topk_minimum_retained_mass) is not float:
                raise ValueError("teacher_topk_minimum_retained_mass must be a float")
            _require_bool(
                self.teacher_topk_fail_below_minimum_retained_mass,  # type: ignore[arg-type]
                name="teacher_topk_fail_below_minimum_retained_mass",
            )
            observed_soft = (
                self.teacher_soft_objective,
                self.teacher_topk_mode,
                self.teacher_top_k,
                self.teacher_topk_include_eos,
                self.teacher_topk_minimum_retained_mass,
                self.teacher_topk_fail_below_minimum_retained_mass,
            )
            expected_soft = (
                soft.objective,
                soft.topk_mode,
                soft.teacher_top_k,
                soft.include_eos,
                soft.minimum_retained_mass,
                soft.fail_below_minimum_retained_mass,
            )
            if observed_soft != expected_soft:
                raise ValueError("soft-teacher binding differs from the registered top-k protocol")

    @property
    def scientific_sha256(self) -> str:
        return sha256_value(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExperimentBinding":
        if not isinstance(payload, Mapping):
            raise ValueError("experiment binding payload must be a mapping")
        expected = {item.name for item in fields(cls)}
        observed = set(payload)
        if observed != expected:
            raise ValueError(
                "experiment binding payload fields differ from schema: "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        try:
            return cls(**dict(payload))
        except TypeError as error:
            raise ValueError("experiment binding payload has invalid runtime types") from error

    def controlled_comparison_key(self) -> tuple[Any, ...]:
        """Fields that must match within one training-seed comparison block."""

        return (
            self.seed,
            self.protocol_track,
            self.preregistration_sha256,
            self.implementation_commit,
            self.implementation_dirty,
            self.model_id,
            self.model_requested_revision,
            self.model_resolved_revision,
            self.teacher_id,
            self.teacher_requested_revision,
            self.teacher_resolved_revision,
            self.tokenizer_id,
            self.tokenizer_requested_revision,
            self.tokenizer_resolved_revision,
            self.tokenizer_fingerprint,
            self.prompt_protocol,
            self.chat_template_sha256,
            self.enable_thinking,
            self.model_facing_prompt_schedule_sha256,
            self.probe_manifest_sha256,
            self.task_protocol_sha256,
            self.response_mask_protocol,
            self.max_completion_length,
            self.full_parameter_training,
            self.initial_checkpoint_sha256,
            self.train_dataset_sha256,
            self.validation_dataset_sha256,
            self.optimizer_spec_sha256,
            self.scheduler_spec_sha256,
            self.backend_id,
            self.backend_version,
            self.backend_batch_contract,
            self.resolved_batch_contract_sha256,
            self.token_budget,
            self.token_budget_unit,
            self.factorial_design_sha256,
        )

    def anchor_comparison_key(self) -> tuple[Any, ...]:
        """Nuisance variables shared by replay and the canonical GRPO anchor."""

        return (
            self.seed,
            self.protocol_track,
            self.preregistration_sha256,
            self.implementation_commit,
            self.implementation_dirty,
            self.model_id,
            self.model_requested_revision,
            self.model_resolved_revision,
            self.teacher_id,
            self.teacher_requested_revision,
            self.teacher_resolved_revision,
            self.tokenizer_id,
            self.tokenizer_requested_revision,
            self.tokenizer_resolved_revision,
            self.tokenizer_fingerprint,
            self.prompt_protocol,
            self.chat_template_sha256,
            self.enable_thinking,
            self.model_facing_prompt_schedule_sha256,
            self.probe_manifest_sha256,
            self.task_protocol_sha256,
            self.state_source_protocol_sha256,
            self.max_completion_length,
            self.full_parameter_training,
            self.initial_checkpoint_sha256,
            self.train_dataset_sha256,
            self.validation_dataset_sha256,
            self.token_budget,
            self.token_budget_unit,
            self.factorial_design_sha256,
        )

    def assert_comparable_to(self, other: "ExperimentBinding") -> None:
        if self.controlled_comparison_key() != other.controlled_comparison_key():
            raise ValueError(
                "experiment bindings differ on a controlled comparison dimension: "
                f"left={self.experiment_id}, right={other.experiment_id}"
            )


@dataclass(frozen=True, slots=True)
class FactorialDesignSpec:
    design_id: str
    state_source_levels: tuple[str, ...]
    supervision_levels: tuple[str, ...]
    cell_ids: tuple[str, ...]
    planned_contrasts: tuple[tuple[str, str], ...]
    matched_accuracy_pairs: tuple[tuple[str, str], ...]
    common_bank_policy: str = "one_frozen_behavior_policy_bank_for_all_offline_cells"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported FactorialDesignSpec schema version")
        _require_text(self.design_id, name="design_id")
        if self.state_source_levels != ("fixed_bank", "current_policy"):
            raise ValueError("the controlled factorial requires fixed-bank and current-policy levels")
        if self.supervision_levels != (
            "hard_teacher",
            "soft_teacher",
            "verified_replay",
        ):
            raise ValueError("the controlled factorial requires the three registered supervisors")
        if self.cell_ids != FACTORIAL_METHOD_IDS:
            raise ValueError("factorial cell IDs differ from the canonical method registry")
        observed = {(spec.state_source, spec.supervision) for spec in FACTORIAL_METHOD_SPECS}
        expected = {
            (state_source, supervision)
            for state_source in self.state_source_levels
            for supervision in self.supervision_levels
        }
        if observed != expected or len(observed) != len(self.cell_ids):
            raise ValueError(f"factorial method grid is incomplete: observed={sorted(observed)}")
        known_methods = set(PILOT_METHOD_IDS)
        for left, right in (*self.planned_contrasts, *self.matched_accuracy_pairs):
            if left not in known_methods or right not in known_methods or left == right:
                raise ValueError(f"invalid preregistered comparison: {(left, right)}")

    @property
    def scientific_sha256(self) -> str:
        return sha256_value(asdict(self))

    def validate_bindings(self, bindings: Iterable[ExperimentBinding]) -> None:
        rows = tuple(bindings)
        if not rows:
            raise ValueError("factorial comparison requires experiment bindings")
        grouped: dict[int, list[ExperimentBinding]] = defaultdict(list)
        seen: set[tuple[int, str]] = set()
        for binding in rows:
            if binding.method_id not in self.cell_ids:
                raise ValueError(f"non-factorial method in factorial bindings: {binding.method_id}")
            if binding.factorial_design_sha256 != self.scientific_sha256:
                raise ValueError("factorial binding is attached to a different design spec")
            key = (binding.seed, binding.method_id)
            if key in seen:
                raise ValueError(f"duplicate factorial binding for seed/method: {key}")
            seen.add(key)
            grouped[binding.seed].append(binding)
        for seed, seed_rows in grouped.items():
            observed_ids = {row.method_id for row in seed_rows}
            if observed_ids != set(self.cell_ids):
                missing = sorted(set(self.cell_ids) - observed_ids)
                extra = sorted(observed_ids - set(self.cell_ids))
                raise ValueError(
                    f"seed {seed} does not contain the full factorial grid: "
                    f"missing={missing}, extra={extra}"
                )
            reference = seed_rows[0]
            for row in seed_rows[1:]:
                reference.assert_comparable_to(row)
        for dimension, binding_field in (
            ("state_source", "state_source_protocol_sha256"),
            ("supervision", "supervision_protocol_sha256"),
        ):
            grouped_protocols: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                level = str(getattr(get_method_spec(row.method_id), dimension))
                grouped_protocols[level].add(str(getattr(row, binding_field)))
            changed = {
                level: sorted(hashes)
                for level, hashes in grouped_protocols.items()
                if len(hashes) != 1
            }
            if changed:
                raise ValueError(
                    f"factorial {dimension} protocol changed within a level: {changed}"
                )
        offline_banks = {
            (
                row.offline_bank_manifest_sha256,
                row.offline_bank_content_sha256,
                row.offline_bank_initial_cursor_sha256,
            )
            for row in rows
            if get_method_spec(row.method_id).requires_common_rollout_bank
        }
        if len(offline_banks) != 1 or any(value is None for value in next(iter(offline_banks))):
            raise ValueError(
                "all offline factorial bindings must share one frozen bank and initial cursor"
            )

    def validate_canonical_grpo_anchor_pair(
        self,
        replay: ExperimentBinding,
        grpo: ExperimentBinding,
    ) -> None:
        if (replay.method_id, grpo.method_id) != (
            "online_verified_replay",
            "canonical_grpo",
        ):
            raise ValueError("canonical GRPO comparison must be paired with online verified replay")
        if replay.factorial_design_sha256 != self.scientific_sha256:
            raise ValueError("replay anchor binding uses a different factorial design")
        if grpo.factorial_design_sha256 != self.scientific_sha256:
            raise ValueError("GRPO anchor binding uses a different factorial design")
        if replay.anchor_comparison_key() != grpo.anchor_comparison_key():
            raise ValueError("canonical GRPO anchor differs on a shared nuisance variable")


CONTROLLED_FACTORIAL = FactorialDesignSpec(
    design_id="state_source_x_supervision_2x3_v1",
    state_source_levels=("fixed_bank", "current_policy"),
    supervision_levels=("hard_teacher", "soft_teacher", "verified_replay"),
    cell_ids=FACTORIAL_METHOD_IDS,
    planned_contrasts=PLANNED_CONTRASTS,
    matched_accuracy_pairs=MATCHED_ACCURACY_PAIRS,
)


def _artifact_content_hash(manifest: Mapping[str, Any], *, name: str) -> str:
    artifact_hash = manifest.get("sha256")
    _require_sha256(artifact_hash, name=f"{name}.sha256")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"{name} requires a non-empty content-file hash mapping")
    for file_name, digest in files.items():
        _require_text(file_name, name=f"{name} file name")
        _require_sha256(digest, name=f"{name} content digest")
    return sha256_value(dict(files))


def _task_protocol_storage_keys(
    value: Any,
    *,
    prefix: str = "task_protocol",
) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            dotted = f"{prefix}.{key}"
            if any(marker in key.lower() for marker in ("path", "directory", "output_root")):
                found.append(dotted)
            found.extend(_task_protocol_storage_keys(child, prefix=dotted))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_task_protocol_storage_keys(child, prefix=f"{prefix}[{index}]"))
    return found


def _without_storage_locators(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_storage_locators(child)
            for key, child in value.items()
            if not any(
                marker in str(key).lower()
                for marker in ("path", "directory", "output_root")
            )
        }
    if isinstance(value, (list, tuple)):
        return [_without_storage_locators(child) for child in value]
    return value


def build_experiment_binding(
    *,
    config: dict[str, Any],
    config_binding: ConfigBinding,
    implementation_commit: str,
    implementation_dirty: bool,
    model_resolved_revision: str,
    tokenizer_resolved_revision: str,
    tokenizer_fingerprint: str,
    teacher_resolved_revision: str,
    initial_checkpoint_sha256: str,
    train_dataset_sha256: str,
    validation_dataset_sha256: str,
    model_facing_prompt_schedule_sha256: str,
    probe_manifest_sha256: str,
    task_protocol: Mapping[str, Any],
    optimizer_spec: Mapping[str, Any],
    scheduler_spec: Mapping[str, Any],
    resolved_batch_contract: Mapping[str, Any],
    full_parameter_training: bool,
    offline_bank_manifest: Mapping[str, Any] | None = None,
    offline_bank_initial_cursor: Mapping[str, Any] | None = None,
    teacher_demo_manifest: Mapping[str, Any] | None = None,
    reward_calibration: Mapping[str, Any] | None = None,
) -> ExperimentBinding:
    """Build the sole runtime ExperimentBinding from validated, resolved inputs."""

    validate_config_binding(config, config_binding)

    experiment = config.get("experiment")
    model = config.get("model")
    teacher = config.get("teacher")
    trainer = config.get("trainer")
    task = config.get("task")
    state_source = config.get("state_source")
    supervision = config.get("supervision")
    if not all(
        isinstance(value, Mapping)
        for value in (
            experiment,
            model,
            teacher,
            trainer,
            task,
            state_source,
            supervision,
        )
    ):
        raise ValueError(
            "experiment binding requires resolved experiment/model/teacher/trainer/task/"
            "state-source/supervision mappings"
        )
    method = get_method_spec(str(experiment.get("name", "")))
    method.validate_resolved_config(config)
    prereg_path = Path(str(config.get("prereg_path", "")))
    if not prereg_path.is_file():
        raise ValueError("experiment binding requires an existing preregistration file")
    prompt = model.get("prompt_protocol", {})
    if not isinstance(prompt, Mapping):
        raise ValueError("experiment binding requires a model prompt protocol mapping")
    path_like_task_fields = _task_protocol_storage_keys(task_protocol)
    if path_like_task_fields:
        raise ValueError(
            "task protocol must replace storage locators with bound content hashes: "
            f"{sorted(path_like_task_fields)}"
        )

    teacher_id = str(
        teacher.get("model_name_or_path", teacher.get("teacher_id", ""))
    )
    teacher_requested = str(
        teacher.get("model_revision", teacher.get("teacher_revision", ""))
    )
    soft = method.soft_teacher_objective
    offline_manifest_hash = None
    offline_content_hash = None
    offline_cursor_hash = None
    if offline_bank_manifest is not None:
        offline_manifest_hash = str(offline_bank_manifest.get("sha256", ""))
        offline_content_hash = _artifact_content_hash(
            offline_bank_manifest,
            name="offline bank",
        )
        if not isinstance(offline_bank_initial_cursor, Mapping):
            raise ValueError("offline bank binding requires its exact initial cursor")
        offline_cursor_hash = sha256_value(dict(offline_bank_initial_cursor))
    elif offline_bank_initial_cursor is not None:
        raise ValueError("offline bank cursor cannot be bound without its manifest")

    demo_manifest_hash = None
    demo_content_hash = None
    demo_generation_hash = None
    demo_gate = None
    if teacher_demo_manifest is not None:
        demo_manifest_hash = str(teacher_demo_manifest.get("sha256", ""))
        demo_content_hash = _artifact_content_hash(
            teacher_demo_manifest,
            name="teacher demo",
        )
        generation = teacher_demo_manifest.get("teacher_demo_generation")
        if not isinstance(generation, Mapping):
            raise ValueError("teacher-demo manifest lacks generation provenance")
        generation_required = {
            "teacher_id": teacher_id,
            "teacher_revision": teacher_requested,
            "resolved_teacher_commit": teacher_resolved_revision,
        }
        generation_mismatches = {
            key: {"expected": value, "observed": generation.get(key)}
            for key, value in generation_required.items()
            if generation.get(key) != value
        }
        if generation_mismatches:
            raise ValueError(
                f"teacher-demo generation provenance changed: {generation_mismatches}"
            )
        if str(generation.get("verifier_version", "")) != "proofgraph-exact-v1":
            raise ValueError("teacher-demo store was not gated by the registered exact verifier")
        demo_generation_hash = sha256_value(dict(generation))
        demo_gate = config.get("state_source", {}).get("require_exact_verifier_success")

    calibration_manifest_hash = None
    calibration_content_hash = None
    if reward_calibration is not None:
        calibration_manifest_hash = str(reward_calibration.get("sha256", ""))
        calibration_content = {
            key: value for key, value in reward_calibration.items() if key != "sha256"
        }
        calibration_content_hash = sha256_value(calibration_content)
        if calibration_manifest_hash != calibration_content_hash:
            raise ValueError("reward calibration manifest/content hash mismatch")
        if reward_calibration.get("individual_verifier_results_available_to_reward") is not False:
            raise ValueError("reward calibration leaked individual verifier results")

    chat_template = str(prompt.get("chat_template_sha256", ""))
    _require_sha256(chat_template, name="model.prompt_protocol.chat_template_sha256")
    return ExperimentBinding(
        experiment_id=(
            f"{config.get('protocol_track', config.get('prereg_version', ''))}:"
            f"{method.method_id}:seed-{config.get('seed')}:"
            f"{CONTROLLED_FACTORIAL.design_id}"
        ),
        method_id=method.method_id,
        method_spec_sha256=method.scientific_sha256,
        factorial_design_sha256=CONTROLLED_FACTORIAL.scientific_sha256,
        scientific_config_sha256=config_binding.scientific_config_sha256,
        seed=config.get("seed"),
        protocol_track=str(config.get("protocol_track", config.get("prereg_version", ""))),
        preregistration_sha256=sha256_file(prereg_path),
        implementation_commit=implementation_commit,
        implementation_dirty=implementation_dirty,
        model_id=str(model.get("model_name_or_path", "")),
        model_requested_revision=str(model.get("model_revision", "")),
        model_resolved_revision=model_resolved_revision,
        teacher_id=teacher_id,
        teacher_requested_revision=teacher_requested,
        teacher_resolved_revision=teacher_resolved_revision,
        tokenizer_id=str(model.get("tokenizer_name_or_path", "")),
        tokenizer_requested_revision=str(model.get("tokenizer_revision", "")),
        tokenizer_resolved_revision=tokenizer_resolved_revision,
        tokenizer_fingerprint=tokenizer_fingerprint,
        prompt_protocol=str(prompt.get("name", "legacy_raw_v1")),
        chat_template_sha256=chat_template,
        enable_thinking=prompt.get("enable_thinking", False),
        model_facing_prompt_schedule_sha256=model_facing_prompt_schedule_sha256,
        probe_manifest_sha256=probe_manifest_sha256,
        task_protocol_sha256=sha256_value(dict(task_protocol)),
        state_source_protocol_sha256=sha256_value(
            _without_storage_locators(state_source)
        ),
        supervision_protocol_sha256=sha256_value(
            _without_storage_locators(supervision)
        ),
        response_mask_protocol=(
            "trl_completion_mask_v1"
            if method.supervision == "grpo"
            else "response_tokens_only_including_eos_v1"
        ),
        max_completion_length=config.get("supervision", {}).get(
            "max_completion_length",
            trainer.get("max_completion_length", 0),
        ),
        full_parameter_training=full_parameter_training,
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        train_dataset_sha256=train_dataset_sha256,
        validation_dataset_sha256=validation_dataset_sha256,
        optimizer_spec_sha256=sha256_value(dict(optimizer_spec)),
        scheduler_spec_sha256=sha256_value(dict(scheduler_spec)),
        backend_id=method.training_backend,
        backend_version=method.training_backend_version,
        backend_batch_contract=method.training_batch_contract,
        resolved_batch_contract_sha256=sha256_value(dict(resolved_batch_contract)),
        token_budget=config.get("trainer", {}).get("token_budget"),
        token_budget_unit=str(
            config.get("trainer", {}).get("token_budget_unit", TOKEN_BUDGET_UNIT)
        ),
        teacher_soft_objective=soft.objective if soft is not None else None,
        teacher_topk_mode=soft.topk_mode if soft is not None else None,
        teacher_top_k=soft.teacher_top_k if soft is not None else None,
        teacher_topk_include_eos=soft.include_eos if soft is not None else None,
        teacher_topk_minimum_retained_mass=(
            soft.minimum_retained_mass if soft is not None else None
        ),
        teacher_topk_fail_below_minimum_retained_mass=(
            soft.fail_below_minimum_retained_mass if soft is not None else None
        ),
        offline_bank_manifest_sha256=offline_manifest_hash,
        offline_bank_content_sha256=offline_content_hash,
        offline_bank_initial_cursor_sha256=offline_cursor_hash,
        teacher_demo_manifest_sha256=demo_manifest_hash,
        teacher_demo_content_sha256=demo_content_hash,
        teacher_demo_exact_verifier_gate=demo_gate,
        teacher_demo_generation_sha256=demo_generation_hash,
        reward_calibration_manifest_sha256=calibration_manifest_hash,
        reward_calibration_content_sha256=calibration_content_hash,
    )


def validate_experiment_binding_payload(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str | None = None,
) -> ExperimentBinding:
    binding = ExperimentBinding.from_payload(payload)
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, name="experiment_binding_sha256")
        if binding.scientific_sha256 != expected_sha256:
            raise ValueError("experiment binding SHA-256 mismatch")
    return binding


def validate_run_manifest_experiment_binding(
    manifest: Mapping[str, Any],
) -> ExperimentBinding:
    """Semantically bind an opaque RunManifest payload to the method registry."""

    if not isinstance(manifest, Mapping):
        raise ValueError("run manifest must be a mapping")
    raw_binding = manifest.get("experiment_binding")
    if not isinstance(raw_binding, Mapping):
        raise ValueError("run manifest requires an ExperimentBinding payload")
    binding = validate_experiment_binding_payload(
        raw_binding,
        expected_sha256=manifest.get("experiment_binding_sha256"),
    )
    dataset_hashes = manifest.get("dataset_hashes")
    if not isinstance(dataset_hashes, Mapping):
        raise ValueError("run manifest dataset_hashes must be a mapping")
    required = {
        "experiment_cell": binding.method_id,
        "seed": binding.seed,
        "model_id": binding.model_id,
        "model_revision": binding.model_requested_revision,
        "resolved_model_commit": binding.model_resolved_revision,
        "tokenizer_id": binding.tokenizer_id,
        "tokenizer_revision": binding.tokenizer_requested_revision,
        "resolved_tokenizer_commit": binding.tokenizer_resolved_revision,
        "tokenizer_fingerprint": binding.tokenizer_fingerprint,
        "prompt_protocol": binding.prompt_protocol,
        "enable_thinking": binding.enable_thinking,
        "chat_template_sha256": binding.chat_template_sha256,
        "model_facing_prompt_schedule_hash": binding.model_facing_prompt_schedule_sha256,
        "token_budget": binding.token_budget,
        "token_budget_unit": binding.token_budget_unit,
        "factorial_design_sha256": binding.factorial_design_sha256,
        "scientific_config_sha256": binding.scientific_config_sha256,
        "git_commit": binding.implementation_commit,
        "dirty_working_tree": binding.implementation_dirty,
        "teacher_id": binding.teacher_id,
        "teacher_revision": binding.teacher_requested_revision,
        "resolved_teacher_commit": binding.teacher_resolved_revision,
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in required.items()
        if manifest.get(key) != value
    }
    dataset_required = {
        "train": binding.train_dataset_sha256,
        "validation": binding.validation_dataset_sha256,
        "initial_checkpoint": binding.initial_checkpoint_sha256,
        "probe_manifest": binding.probe_manifest_sha256,
    }
    for key, value in dataset_required.items():
        if dataset_hashes.get(key) != value:
            mismatches[f"dataset_hashes.{key}"] = {
                "expected": value,
                "observed": dataset_hashes.get(key),
            }
    if binding.factorial_design_sha256 != CONTROLLED_FACTORIAL.scientific_sha256:
        mismatches["factorial_design_sha256"] = {
            "expected": CONTROLLED_FACTORIAL.scientific_sha256,
            "observed": binding.factorial_design_sha256,
        }
    source_binding = (
        binding.offline_bank_manifest_sha256 or binding.teacher_demo_manifest_sha256
    )
    if source_binding is not None and manifest.get("rollout_bank_hash") != source_binding:
        mismatches["rollout_bank_hash"] = {
            "expected": source_binding,
            "observed": manifest.get("rollout_bank_hash"),
        }
    if mismatches:
        raise ValueError(f"run manifest differs from its ExperimentBinding: {mismatches}")
    return binding


def validate_experiment_binding_resolved_config(
    binding: ExperimentBinding,
    config: dict[str, Any],
    config_binding: ConfigBinding,
) -> None:
    """Recompute path-free method protocol hashes from the bound resolved config."""

    validate_config_binding(config, config_binding)
    if binding.scientific_config_sha256 != config_binding.scientific_config_sha256:
        raise ValueError("ExperimentBinding scientific config differs from ConfigBinding")

    method = get_method_spec(binding.method_id)
    method.validate_resolved_config(config)
    state_source = config.get("state_source")
    supervision = config.get("supervision")
    if not isinstance(state_source, Mapping) or not isinstance(supervision, Mapping):
        raise ValueError("resolved binding config lacks state-source/supervision mappings")
    expected = {
        "state_source_protocol_sha256": sha256_value(
            _without_storage_locators(state_source)
        ),
        "supervision_protocol_sha256": sha256_value(
            _without_storage_locators(supervision)
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": getattr(binding, key)}
        for key, value in expected.items()
        if getattr(binding, key) != value
    }
    if mismatches:
        raise ValueError(f"ExperimentBinding differs from resolved method config: {mismatches}")


def validate_checkpoint_experiment_binding_payload(
    payload: Mapping[str, Any],
    binding: ExperimentBinding,
) -> None:
    """Validate the scientific identity stored inside a checkpoint payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    method = get_method_spec(binding.method_id)
    if method.training_backend == "trl.GRPOTrainer":
        required = {
            "experiment_binding_sha256": binding.scientific_sha256,
            "factorial_design_sha256": binding.factorial_design_sha256,
            "method_spec_sha256": binding.method_spec_sha256,
            "trl_version": binding.backend_version,
            "trl_batch_contract": binding.backend_batch_contract,
            "current_policy_contract": (
                asdict(method.current_policy) if method.current_policy is not None else None
            ),
            "initial_checkpoint_sha256": binding.initial_checkpoint_sha256,
            "git_commit": binding.implementation_commit,
            "implementation_dirty": binding.implementation_dirty,
        }
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in required.items()
            if payload.get(key) != value
        }
    else:
        manifest_hashes = payload.get("manifest_hashes")
        if not isinstance(manifest_hashes, Mapping):
            raise ValueError("checkpoint lacks scientific manifest hashes")
        required = {
            "experiment_binding": binding.scientific_sha256,
            "factorial_design": binding.factorial_design_sha256,
            "method_spec": binding.method_spec_sha256,
            "dataset_train": binding.train_dataset_sha256,
            "dataset_validation": binding.validation_dataset_sha256,
            "initial_checkpoint": binding.initial_checkpoint_sha256,
            "model_facing_prompt_schedule": binding.model_facing_prompt_schedule_sha256,
            "probe_manifest": binding.probe_manifest_sha256,
        }
        mismatches = {
            f"manifest_hashes.{key}": {
                "expected": value,
                "observed": manifest_hashes.get(key),
            }
            for key, value in required.items()
            if manifest_hashes.get(key) != value
        }
        for key, value in {
            "git_commit": binding.implementation_commit,
            "implementation_dirty": binding.implementation_dirty,
        }.items():
            if payload.get(key) != value:
                mismatches[key] = {
                    "expected": value,
                    "observed": payload.get(key),
                }
        _validate_factorial_checkpoint_runtime_payload(payload, binding)
    if mismatches:
        raise ValueError(f"checkpoint scientific binding changed: {mismatches}")


def _validate_factorial_checkpoint_runtime_payload(
    payload: Mapping[str, Any],
    binding: ExperimentBinding,
) -> None:
    """Reject a metadata-only or semantically non-resumable Factorial checkpoint."""

    import torch

    model = payload.get("model")
    if not isinstance(model, Mapping) or not model:
        raise ValueError("Factorial checkpoint model state is missing or empty")
    if any(not isinstance(value, torch.Tensor) for value in model.values()):
        raise ValueError("Factorial checkpoint model state must be tensor-valued")
    observed_model_hash = torch_state_hash(dict(model))
    if payload.get("final_model_state_hash") != observed_model_hash:
        raise ValueError("Factorial checkpoint final model-state hash differs from its model")

    optimizer = payload.get("optimizer")
    if (
        not isinstance(optimizer, Mapping)
        or not isinstance(optimizer.get("state"), Mapping)
        or not optimizer.get("state")
        or not isinstance(optimizer.get("param_groups"), list)
        or not optimizer.get("param_groups")
    ):
        raise ValueError("Factorial checkpoint optimizer state is incomplete")
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, Mapping) or not scheduler:
        raise ValueError("Factorial checkpoint scheduler state is incomplete")

    global_step = payload.get("global_step")
    if type(global_step) is not int or global_step < 1:
        raise ValueError("Factorial checkpoint global_step must be a positive integer")
    scheduler_epoch = scheduler.get("last_epoch")
    if type(scheduler_epoch) is not int or scheduler_epoch < global_step:
        raise ValueError("Factorial checkpoint scheduler state trails global_step")

    rng = payload.get("rng")
    if not isinstance(rng, Mapping) or set(rng) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("Factorial checkpoint RNG state is incomplete")
    for name in ("python", "numpy"):
        value = rng.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Factorial checkpoint {name} RNG state is invalid")
        try:
            base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error) as error:
            raise ValueError(f"Factorial checkpoint {name} RNG state is invalid") from error
    cpu_rng = rng.get("torch_cpu")
    cuda_rng = rng.get("torch_cuda")
    if (
        not isinstance(cpu_rng, list)
        or not cpu_rng
        or any(type(value) is not int or not 0 <= value <= 255 for value in cpu_rng)
        or not isinstance(cuda_rng, list)
        or any(
            not isinstance(state, list)
            or not state
            or any(type(value) is not int or not 0 <= value <= 255 for value in state)
            for state in cuda_rng
        )
    ):
        raise ValueError("Factorial checkpoint torch RNG state is invalid")

    trainer_state = payload.get("trainer_state")
    token_budget = payload.get("token_budget")
    if (
        not isinstance(trainer_state, Mapping)
        or not isinstance(trainer_state.get("cumulative_counts"), Mapping)
        or trainer_state.get("accumulation_micro_step") != 0
        or not isinstance(trainer_state.get("token_budget"), Mapping)
        or trainer_state.get("token_budget") != token_budget
    ):
        raise ValueError("Factorial checkpoint trainer state is incomplete or mid-update")
    if not isinstance(token_budget, Mapping):
        raise ValueError("Factorial checkpoint token-budget state is missing")
    token_required = {
        "budget": binding.token_budget,
        "unit": TOKEN_BUDGET_UNIT,
        "accepted_optimizer_updates": global_step,
    }
    token_mismatches = {
        key: {"expected": value, "observed": token_budget.get(key)}
        for key, value in token_required.items()
        if token_budget.get(key) != value
    }
    consumed = token_budget.get("consumed")
    if type(consumed) is not int or not 0 < consumed <= binding.token_budget:
        token_mismatches["consumed"] = {
            "expected": f"an integer in [1, {binding.token_budget}]",
            "observed": consumed,
        }
    counts = trainer_state.get("cumulative_counts")
    observed_count = (
        counts.get("model_facing_input_tokens_processed")
        if isinstance(counts, Mapping)
        else None
    )
    if (
        isinstance(observed_count, bool)
        or not isinstance(observed_count, (int, float))
        or not math.isfinite(float(observed_count))
        or consumed != int(observed_count)
    ):
        token_mismatches["cumulative_model_facing_tokens"] = {
            "expected": consumed,
            "observed": observed_count,
        }
    if token_mismatches:
        raise ValueError(f"Factorial checkpoint token-budget state changed: {token_mismatches}")

    world_size = payload.get("world_size")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("Factorial checkpoint world_size is invalid")
    rank_states = {
        "prompt_scheduler_by_rank": payload.get("prompt_scheduler_by_rank"),
        "state_source_by_rank": payload.get("state_source_by_rank"),
    }
    for name, states in rank_states.items():
        if (
            not isinstance(states, list)
            or len(states) != world_size
            or any(not isinstance(state, Mapping) for state in states)
        ):
            raise ValueError(f"Factorial checkpoint {name} is incomplete")
    if (
        payload.get("prompt_scheduler") != rank_states["prompt_scheduler_by_rank"][0]
        or payload.get("state_source") != rank_states["state_source_by_rank"][0]
    ):
        raise ValueError("Factorial checkpoint rank-0 prompt/source state is inconsistent")
    rank_shards = payload.get("rank_shard_hashes")
    if (
        not isinstance(rank_shards, list)
        or len(rank_shards) != world_size
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in rank_shards
        )
        or len(set(rank_shards)) != len(rank_shards)
    ):
        raise ValueError("Factorial checkpoint rank-shard bindings are incomplete")

    reported_norm = payload.get("parameter_update_norm")
    if (
        isinstance(reported_norm, bool)
        or not isinstance(reported_norm, (int, float))
        or not math.isfinite(float(reported_norm))
        or float(reported_norm) <= 0.0
    ):
        raise ValueError("Factorial checkpoint parameter_update_norm must be finite and positive")
    ancestry = payload.get("resume_ancestry")
    if not isinstance(ancestry, list) or any(
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
        for value in ancestry
    ):
        raise ValueError("Factorial checkpoint resume ancestry is invalid")
    if len(set(ancestry)) != len(ancestry):
        raise ValueError("Factorial checkpoint resume ancestry contains a cycle")
    baseline_hash = payload.get("update_norm_baseline_checkpoint_sha256")
    expected_baseline_hash = ancestry[-1][7:] if ancestry else binding.initial_checkpoint_sha256
    if baseline_hash != expected_baseline_hash:
        raise ValueError("Factorial checkpoint update-norm baseline binding changed")
    baseline_path = payload.get("update_norm_baseline_checkpoint_path")
    if not isinstance(baseline_path, str) or not Path(baseline_path).is_absolute():
        raise ValueError("Factorial checkpoint update-norm baseline path is invalid")

__all__ = [
    "CONTROLLED_FACTORIAL",
    "ExperimentBinding",
    "FactorialDesignSpec",
    "MATCHED_ACCURACY_PAIRS",
    "PLANNED_CONTRASTS",
    "TOKEN_BUDGET_UNIT",
    "build_experiment_binding",
    "validate_checkpoint_experiment_binding_payload",
    "validate_experiment_binding_resolved_config",
    "validate_experiment_binding_payload",
    "validate_run_manifest_experiment_binding",
]
