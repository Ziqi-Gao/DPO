"""Immutable candidate entry templates; this catalog cannot activate handlers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from posttrain_circuits.workflows.contracts import (
    IDENTIFIER,
    WORKFLOW_TASK_REGISTRY,
)


MODULE_NAME = re.compile(r"posttrain_circuits\.cli\.[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class CandidateEntrypointTemplate:
    """A scientific shape only: no executable, argv, env, resource, or path."""

    task: str
    module: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    gate_names: tuple[str, ...]

    def validate(self) -> None:
        if not isinstance(self.task, str) or not IDENTIFIER.fullmatch(self.task):
            raise ValueError("candidate task must be an identifier")
        if not isinstance(self.module, str) or not MODULE_NAME.fullmatch(self.module):
            raise ValueError("candidate module must be a fixed OPD CLI module name")
        for values, context in (
            (self.input_names, "candidate input names"),
            (self.output_names, "candidate output names"),
            (self.gate_names, "candidate gate names"),
        ):
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{context} must be a non-empty tuple")
            if any(
                not isinstance(value, str) or not IDENTIFIER.fullmatch(value)
                for value in values
            ):
                raise ValueError(f"{context} must contain identifiers")
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise ValueError(f"{context} must be sorted and unique")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "gate_names": list(self.gate_names),
            "input_names": list(self.input_names),
            "module": self.module,
            "output_names": list(self.output_names),
            "task": self.task,
        }


def _template(
    task: str,
    module: str,
    *,
    outputs: tuple[str, ...],
    gates: tuple[str, ...],
) -> CandidateEntrypointTemplate:
    return CandidateEntrypointTemplate(
        task=task,
        module=module,
        input_names=WORKFLOW_TASK_REGISTRY[task],
        output_names=outputs,
        gate_names=gates,
    )


_TRAINING_OUTPUTS = ("checkpoint.pt", "metrics.jsonl", "run_manifest.json")
_TRAINING_GATES = ("checkpoint_integrity", "config_binding", "finite_metrics")

_CANDIDATE_TEMPLATES = (
    _template(
        "offline_hard",
        "posttrain_circuits.cli.train",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "online_hard",
        "posttrain_circuits.cli.train",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "offline_soft",
        "posttrain_circuits.cli.train",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "online_soft_opd",
        "posttrain_circuits.cli.train",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "offline_verified_replay",
        "posttrain_circuits.cli.train",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "online_verified_replay",
        "posttrain_circuits.cli.train",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "canonical_sft",
        "posttrain_circuits.cli.train",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "canonical_grpo",
        "posttrain_circuits.cli.run_grpo",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "grpo_format_reward",
        "posttrain_circuits.cli.run_grpo",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "grpo_random_reward",
        "posttrain_circuits.cli.run_grpo",
        outputs=_TRAINING_OUTPUTS,
        gates=_TRAINING_GATES,
    ),
    _template(
        "build_splits",
        "posttrain_circuits.cli.build_splits",
        outputs=("dataset_family.json",),
        gates=("dataset_integrity", "split_disjointness"),
    ),
    _template(
        "build_rollout_bank",
        "posttrain_circuits.cli.build_rollout_bank",
        outputs=("rollout_bank.json",),
        gates=("prompt_binding", "trajectory_integrity"),
    ),
    _template(
        "build_teacher_demos",
        "posttrain_circuits.cli.build_teacher_demos",
        outputs=("teacher_demo_store.json",),
        gates=("attempt_ledger_integrity", "verifier_binding"),
    ),
    _template(
        "score_teacher",
        "posttrain_circuits.cli.score_teacher",
        outputs=("teacher_score_bank.json",),
        gates=("source_bank_binding", "teacher_revision_binding"),
    ),
    _template(
        "build_probe_cohorts",
        "posttrain_circuits.cli.build_probe_cohorts",
        outputs=("probe_cohort_manifest.json",),
        gates=("cohort_disjointness", "source_manifest_binding"),
    ),
    _template(
        "run_local_fork",
        "posttrain_circuits.cli.run_local_fork",
        outputs=("local_fork_report.json",),
        gates=("branch_binding", "output_kl_match"),
    ),
    _template(
        "discover_circuit",
        "posttrain_circuits.cli.discover_circuit",
        outputs=("circuit_candidate.json",),
        gates=("probe_binding", "stage_separation"),
    ),
    _template(
        "validate_circuit",
        "posttrain_circuits.cli.evaluate_circuit",
        outputs=("circuit_validation.json",),
        gates=("faithfulness", "heldout_probe_binding"),
    ),
    _template(
        "analyze_circuit_dynamics",
        "posttrain_circuits.cli.analyze_circuit_dynamics",
        outputs=("circuit_dynamics.json",),
        gates=("checkpoint_family_binding", "validated_circuit_binding"),
    ),
)

CANDIDATE_ENTRYPOINT_CATALOG: Mapping[
    str, CandidateEntrypointTemplate
] = MappingProxyType({template.task: template for template in _CANDIDATE_TEMPLATES})


def validate_candidate_catalog(
    catalog: Mapping[str, CandidateEntrypointTemplate] = CANDIDATE_ENTRYPOINT_CATALOG,
) -> None:
    if not isinstance(catalog, MappingProxyType):
        raise ValueError("candidate entrypoint catalog must be immutable")
    if set(catalog) != set(WORKFLOW_TASK_REGISTRY):
        raise ValueError("candidate catalog must exactly cover workflow task contracts")
    if len(catalog) != len(_CANDIDATE_TEMPLATES):
        raise ValueError("candidate catalog contains duplicate task templates")
    for task, template in catalog.items():
        if not isinstance(template, CandidateEntrypointTemplate):
            raise ValueError("candidate catalog entry has the wrong type")
        template.validate()
        if template.task != task:
            raise ValueError("candidate catalog key differs from template task")
        if template.input_names != WORKFLOW_TASK_REGISTRY[task]:
            raise ValueError("candidate inputs differ from workflow task contract")


def candidate_entrypoint(task: str) -> CandidateEntrypointTemplate:
    validate_candidate_catalog()
    try:
        return CANDIDATE_ENTRYPOINT_CATALOG[task]
    except KeyError as error:
        raise KeyError(f"no candidate entrypoint template for task {task!r}") from error


validate_candidate_catalog()


__all__ = [
    "CANDIDATE_ENTRYPOINT_CATALOG",
    "CandidateEntrypointTemplate",
    "candidate_entrypoint",
    "validate_candidate_catalog",
]
