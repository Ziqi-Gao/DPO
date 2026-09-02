"""Immutable, scheduler-neutral contracts for OPD scientific workflow DAGs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from posttrain_circuits.artifacts.hashing import (
    canonical_json as canonical_json,
    sha256_value,
)


WORKFLOW_SCHEMA_VERSION = 2
PROJECT = "OPD"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CONTENT_KINDS = frozenset({"file", "tree"})

# The registry is deliberately tensor-free and does not import ``methods``.
# Each training method remains a distinct task so the six factorial cells and
# the SFT/RL anchors cannot be collapsed by orchestration.  Values are the exact
# names of content digests a plan must bind.  Registry membership does *not*
# claim that a ServerScheduler registration or execution handler already exists.
_TRAINING_BINDING = (
    "config_binding_sha256",
    "execution_config_sha256",
    "experiment_binding_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
)
WORKFLOW_TASK_REGISTRY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "repository_preflight": (
            "config_binding_sha256",
            "execution_config_sha256",
            "resolved_config_sha256",
            "scientific_config_sha256",
        ),
        "offline_hard": _TRAINING_BINDING,
        "online_hard": _TRAINING_BINDING,
        "offline_soft": _TRAINING_BINDING,
        "online_soft_opd": _TRAINING_BINDING,
        "offline_verified_replay": _TRAINING_BINDING,
        "online_verified_replay": _TRAINING_BINDING,
        "canonical_sft": _TRAINING_BINDING,
        "canonical_grpo": _TRAINING_BINDING,
        "grpo_format_reward": _TRAINING_BINDING,
        "grpo_random_reward": _TRAINING_BINDING,
        "build_splits": (
            "dataset_family_spec_sha256",
            "generator_protocol_sha256",
            "preregistration_sha256",
        ),
        "build_rollout_bank": (
            "behavior_checkpoint_sha256",
            "dataset_family_manifest_sha256",
            "sampling_protocol_sha256",
            "tokenizer_fingerprint_sha256",
        ),
        "build_teacher_demos": (
            "dataset_family_manifest_sha256",
            "teacher_generation_protocol_sha256",
            "teacher_revision_sha256",
            "verifier_protocol_sha256",
        ),
        "score_teacher": (
            "rollout_bank_manifest_sha256",
            "teacher_revision_sha256",
            "teacher_scoring_protocol_sha256",
            "tokenizer_fingerprint_sha256",
        ),
        "build_probe_cohorts": (
            "dataset_family_manifest_sha256",
            "probe_cohort_protocol_sha256",
        ),
        "run_local_fork": (
            "local_fork_protocol_sha256",
            "source_checkpoint_sha256",
            "trajectory_bank_manifest_sha256",
        ),
        "discover_circuit": (
            "checkpoint_sha256",
            "circuit_discovery_protocol_sha256",
            "probe_manifest_sha256",
        ),
        "validate_circuit": (
            "candidate_circuit_sha256",
            "checkpoint_sha256",
            "circuit_validation_protocol_sha256",
            "heldout_probe_manifest_sha256",
        ),
        "analyze_circuit_dynamics": (
            "checkpoint_family_sha256",
            "circuit_dynamics_protocol_sha256",
            "probe_manifest_sha256",
            "validated_circuit_set_sha256",
        ),
    }
)
WORKFLOW_TASK_ALLOWLIST = frozenset(WORKFLOW_TASK_REGISTRY)


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a valid identifier")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    observed = frozenset(payload)
    if observed != expected:
        raise ValueError(
            f"{context} fields differ from schema: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _validate_task_registry(task_registry: Mapping[str, tuple[str, ...]]) -> None:
    if not isinstance(task_registry, Mapping) or not task_registry:
        raise ValueError("workflow task registry must be a non-empty mapping")
    for task, required_names in task_registry.items():
        _require_identifier(task, name="workflow task registry key")
        if not isinstance(required_names, tuple) or not required_names:
            raise ValueError(
                f"workflow task {task!r} must require a non-empty content identity tuple"
            )
        for name in required_names:
            _require_identifier(name, name=f"workflow task {task!r} content identity")
        if tuple(sorted(required_names)) != required_names or len(set(required_names)) != len(
            required_names
        ):
            raise ValueError(f"workflow task {task!r} has a non-canonical registry entry")


@dataclass(frozen=True, order=True)
class ContentIdentity:
    """A named immutable input; storage locations are intentionally absent."""

    name: str
    sha256: str
    kind: str

    def validate(self) -> None:
        _require_identifier(self.name, name="content identity name")
        _require_sha256(self.sha256, name=f"content identity {self.name!r}")
        if self.kind not in CONTENT_KINDS:
            raise ValueError(
                f"content identity {self.name!r} kind must be 'file' or 'tree'"
            )

    def to_payload(self) -> dict[str, str]:
        self.validate()
        return {"kind": self.kind, "name": self.name, "sha256": self.sha256}

    @classmethod
    def from_payload(cls, payload: object) -> "ContentIdentity":
        if not isinstance(payload, dict):
            raise ValueError("content identity must be an object")
        _exact_keys(
            payload,
            frozenset({"kind", "name", "sha256"}),
            context="content identity",
        )
        identity = cls(
            name=payload["name"], sha256=payload["sha256"], kind=payload["kind"]
        )
        identity.validate()
        return identity


@dataclass(frozen=True, order=True)
class Dependency:
    """An edge to one named output of another unit in the same plan."""

    unit_id: str
    output_name: str

    def validate(self) -> None:
        _require_identifier(self.unit_id, name="dependency unit_id")
        _require_identifier(self.output_name, name="dependency output_name")

    def to_payload(self) -> dict[str, str]:
        self.validate()
        return {"output_name": self.output_name, "unit_id": self.unit_id}

    @classmethod
    def from_payload(cls, payload: object) -> "Dependency":
        if not isinstance(payload, dict):
            raise ValueError("workflow dependency must be an object")
        _exact_keys(
            payload,
            frozenset({"unit_id", "output_name"}),
            context="workflow dependency",
        )
        dependency = cls(
            unit_id=payload["unit_id"], output_name=payload["output_name"]
        )
        dependency.validate()
        return dependency


@dataclass(frozen=True)
class WorkflowUnit:
    """One scientific unit, containing no runtime or storage selectors."""

    unit_id: str
    task: str
    content_inputs: tuple[ContentIdentity, ...]
    dependencies: tuple[Dependency, ...]
    output_names: tuple[str, ...]

    def validate(self, *, task_registry: Mapping[str, tuple[str, ...]]) -> None:
        _validate_task_registry(task_registry)
        _require_identifier(self.unit_id, name="workflow unit_id")
        _require_identifier(self.task, name=f"workflow unit {self.unit_id!r} task")
        if self.task not in task_registry:
            raise ValueError(
                f"workflow unit {self.unit_id!r} task is not allowlisted: {self.task!r}"
            )
        if not isinstance(self.content_inputs, tuple):
            raise ValueError("workflow unit content_inputs must be a tuple")
        if not isinstance(self.dependencies, tuple):
            raise ValueError("workflow unit dependencies must be a tuple")
        if not isinstance(self.output_names, tuple) or not self.output_names:
            raise ValueError("workflow unit output_names must be a non-empty tuple")
        for identity in self.content_inputs:
            if not isinstance(identity, ContentIdentity):
                raise ValueError("workflow unit content_inputs are invalid")
            identity.validate()
        for dependency in self.dependencies:
            if not isinstance(dependency, Dependency):
                raise ValueError("workflow unit dependencies are invalid")
            dependency.validate()
        for output_name in self.output_names:
            _require_identifier(output_name, name="workflow output name")

        input_names = [identity.name for identity in self.content_inputs]
        if len(set(input_names)) != len(input_names):
            raise ValueError(f"workflow unit {self.unit_id!r} has duplicate content inputs")
        if tuple(sorted(self.content_inputs)) != self.content_inputs:
            raise ValueError(
                f"workflow unit {self.unit_id!r} content inputs are not canonical"
            )
        required_names = task_registry[self.task]
        if tuple(input_names) != required_names:
            raise ValueError(
                f"workflow unit {self.unit_id!r} content identities differ from its task "
                f"contract: expected={required_names}, observed={tuple(input_names)}"
            )
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"workflow unit {self.unit_id!r} has duplicate dependencies")
        if tuple(sorted(self.dependencies)) != self.dependencies:
            raise ValueError(
                f"workflow unit {self.unit_id!r} dependencies are not canonical"
            )
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError(f"workflow unit {self.unit_id!r} has duplicate outputs")
        if tuple(sorted(self.output_names)) != self.output_names:
            raise ValueError(f"workflow unit {self.unit_id!r} outputs are not canonical")

    def to_payload(
        self, *, task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY
    ) -> dict[str, Any]:
        self.validate(task_registry=task_registry)
        return {
            "content_inputs": [identity.to_payload() for identity in self.content_inputs],
            "dependencies": [edge.to_payload() for edge in self.dependencies],
            "output_names": list(self.output_names),
            "task": self.task,
            "unit_id": self.unit_id,
        }

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
    ) -> "WorkflowUnit":
        if not isinstance(payload, dict):
            raise ValueError("workflow unit must be an object")
        _exact_keys(
            payload,
            frozenset(
                {"unit_id", "task", "content_inputs", "dependencies", "output_names"}
            ),
            context="workflow unit",
        )
        raw_inputs = payload["content_inputs"]
        raw_dependencies = payload["dependencies"]
        raw_outputs = payload["output_names"]
        if not isinstance(raw_inputs, list):
            raise ValueError("workflow unit content_inputs must be an array")
        if not isinstance(raw_dependencies, list):
            raise ValueError("workflow unit dependencies must be an array")
        if not isinstance(raw_outputs, list):
            raise ValueError("workflow unit output_names must be an array")
        unit = cls(
            unit_id=payload["unit_id"],
            task=payload["task"],
            content_inputs=tuple(ContentIdentity.from_payload(item) for item in raw_inputs),
            dependencies=tuple(Dependency.from_payload(item) for item in raw_dependencies),
            output_names=tuple(raw_outputs),
        )
        unit.validate(task_registry=task_registry)
        return unit


@dataclass(frozen=True)
class WorkflowPlan:
    """A canonical scientific DAG whose digest commits to every unit and edge."""

    workflow_id: str
    units: tuple[WorkflowUnit, ...]
    schema_version: int = WORKFLOW_SCHEMA_VERSION
    project: str = PROJECT

    def validate(
        self, *, task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY
    ) -> None:
        if self.schema_version != WORKFLOW_SCHEMA_VERSION or isinstance(
            self.schema_version, bool
        ):
            raise ValueError("unsupported workflow plan schema_version")
        if self.project != PROJECT:
            raise ValueError("workflow plan project must be OPD")
        _require_identifier(self.workflow_id, name="workflow_id")
        if not isinstance(self.units, tuple) or not self.units:
            raise ValueError("workflow plan units must be a non-empty tuple")
        _validate_task_registry(task_registry)
        unit_ids = [unit.unit_id for unit in self.units]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("workflow plan contains duplicate unit IDs")
        if tuple(sorted(unit_ids)) != tuple(unit_ids):
            raise ValueError("workflow plan units are not in canonical unit_id order")
        unit_by_id = {unit.unit_id: unit for unit in self.units}
        for unit in self.units:
            unit.validate(task_registry=task_registry)
            for dependency in unit.dependencies:
                if dependency.unit_id not in unit_by_id:
                    raise ValueError(
                        f"workflow unit {unit.unit_id!r} has unknown dependency "
                        f"{dependency.unit_id!r}"
                    )
                if dependency.output_name not in unit_by_id[dependency.unit_id].output_names:
                    raise ValueError(
                        f"workflow dependency {dependency.unit_id!r}/"
                        f"{dependency.output_name!r} names an unknown output"
                    )
        self._validate_acyclic(unit_by_id)

    def _validate_acyclic(self, unit_by_id: Mapping[str, WorkflowUnit]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(unit_id: str) -> None:
            if unit_id in visiting:
                raise ValueError(f"workflow dependency cycle includes {unit_id!r}")
            if unit_id in visited:
                return
            visiting.add(unit_id)
            for dependency in unit_by_id[unit_id].dependencies:
                visit(dependency.unit_id)
            visiting.remove(unit_id)
            visited.add(unit_id)

        for unit_id in sorted(unit_by_id):
            visit(unit_id)

    def content_payload(
        self, *, task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY
    ) -> dict[str, Any]:
        self.validate(task_registry=task_registry)
        return {
            "project": self.project,
            "schema_version": self.schema_version,
            "units": [
                unit.to_payload(task_registry=task_registry) for unit in self.units
            ],
            "workflow_id": self.workflow_id,
        }

    def sha256(
        self, *, task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY
    ) -> str:
        return sha256_value(self.content_payload(task_registry=task_registry))

    def to_payload(
        self, *, task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY
    ) -> dict[str, Any]:
        content = self.content_payload(task_registry=task_registry)
        return {**content, "sha256": sha256_value(content)}

    def unit(
        self,
        unit_id: str,
        *,
        task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
    ) -> WorkflowUnit:
        self.validate(task_registry=task_registry)
        _require_identifier(unit_id, name="unit_id")
        for unit in self.units:
            if unit.unit_id == unit_id:
                return unit
        raise KeyError(f"workflow plan has no unit {unit_id!r}")

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        task_registry: Mapping[str, tuple[str, ...]] = WORKFLOW_TASK_REGISTRY,
    ) -> "WorkflowPlan":
        if not isinstance(payload, dict):
            raise ValueError("workflow plan must be an object")
        expected = frozenset(
            {"schema_version", "project", "workflow_id", "units", "sha256"}
        )
        _exact_keys(payload, expected, context="workflow plan")
        raw_units = payload["units"]
        if not isinstance(raw_units, list):
            raise ValueError("workflow plan units must be an array")
        plan = cls(
            schema_version=payload["schema_version"],
            project=payload["project"],
            workflow_id=payload["workflow_id"],
            units=tuple(
                WorkflowUnit.from_payload(item, task_registry=task_registry)
                for item in raw_units
            ),
        )
        plan.validate(task_registry=task_registry)
        observed_sha256 = _require_sha256(payload["sha256"], name="workflow plan sha256")
        if observed_sha256 != plan.sha256(task_registry=task_registry):
            raise ValueError("workflow plan SHA-256 mismatch")
        if plan.to_payload(task_registry=task_registry) != payload:
            raise ValueError("workflow plan is not a canonical strict round-trip")
        return plan


def unit_identity_sha256(*, workflow_id: str, plan_sha256: str, unit_id: str) -> str:
    return _workflow_identity(
        kind="unit",
        workflow_id=workflow_id,
        plan_sha256=plan_sha256,
        unit_id=unit_id,
    )


def output_identity_sha256(
    *, workflow_id: str, plan_sha256: str, unit_id: str, output_name: str
) -> str:
    _require_identifier(output_name, name="output_name")
    return _workflow_identity(
        kind="output",
        workflow_id=workflow_id,
        plan_sha256=plan_sha256,
        unit_id=unit_id,
        output_name=output_name,
    )


def completion_identity_sha256(
    *, workflow_id: str, plan_sha256: str, unit_id: str
) -> str:
    return _workflow_identity(
        kind="completion",
        workflow_id=workflow_id,
        plan_sha256=plan_sha256,
        unit_id=unit_id,
    )


def _workflow_identity(
    *,
    kind: str,
    workflow_id: str,
    plan_sha256: str,
    unit_id: str,
    output_name: str | None = None,
) -> str:
    _require_identifier(workflow_id, name="workflow_id")
    _require_sha256(plan_sha256, name="plan_sha256")
    _require_identifier(unit_id, name="unit_id")
    payload: dict[str, str] = {
        "kind": kind,
        "plan_sha256": plan_sha256,
        "unit_id": unit_id,
        "workflow_id": workflow_id,
    }
    if output_name is not None:
        payload["output_name"] = output_name
    return sha256_value(payload)
