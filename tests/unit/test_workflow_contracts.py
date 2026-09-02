from __future__ import annotations

import copy
import importlib.util
import unittest
from dataclasses import FrozenInstanceError
from types import MappingProxyType

from posttrain_circuits.artifacts.hashing import canonical_json as artifact_canonical_json
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.workflows.catalog import (
    CANDIDATE_ENTRYPOINT_CATALOG,
    candidate_entrypoint,
    validate_candidate_catalog,
)
from posttrain_circuits.workflows.contracts import (
    WORKFLOW_TASK_ALLOWLIST,
    WORKFLOW_TASK_REGISTRY,
    ContentIdentity,
    Dependency,
    WorkflowPlan,
    WorkflowUnit,
    canonical_json,
    completion_identity_sha256,
    output_identity_sha256,
    unit_identity_sha256,
)


def _digest(name: str) -> str:
    return sha256_value({"identity": name})


def _unit(
    unit_id: str,
    task: str,
    *,
    dependencies: tuple[Dependency, ...] = (),
    outputs: tuple[str, ...] = ("result.json",),
    registry: dict[str, tuple[str, ...]] | None = None,
) -> WorkflowUnit:
    selected = WORKFLOW_TASK_REGISTRY if registry is None else registry
    return WorkflowUnit(
        unit_id=unit_id,
        task=task,
        content_inputs=tuple(
            ContentIdentity(
                name=name, sha256=_digest(f"{unit_id}:{name}"), kind="file"
            )
            for name in selected[task]
        ),
        dependencies=dependencies,
        output_names=outputs,
    )


class WorkflowContractTests(unittest.TestCase):
    def test_candidate_entrypoint_catalog_exactly_covers_scientific_tasks(self):
        self.assertIsInstance(CANDIDATE_ENTRYPOINT_CATALOG, MappingProxyType)
        self.assertEqual(
            set(CANDIDATE_ENTRYPOINT_CATALOG), set(WORKFLOW_TASK_REGISTRY)
        )
        validate_candidate_catalog()
        for task, template in CANDIDATE_ENTRYPOINT_CATALOG.items():
            with self.subTest(task=task):
                self.assertIs(candidate_entrypoint(task), template)
                self.assertEqual(template.input_names, WORKFLOW_TASK_REGISTRY[task])
                self.assertIsNotNone(importlib.util.find_spec(template.module))
                self.assertEqual(
                    set(template.to_payload()),
                    {"gate_names", "input_names", "module", "output_names", "task"},
                )
                self.assertFalse(
                    {
                        "argv",
                        "command",
                        "cwd",
                        "env",
                        "path",
                        "profile",
                        "resources",
                    }
                    & set(template.to_payload())
                )

    def test_candidate_entrypoint_catalog_and_templates_are_immutable(self):
        template = candidate_entrypoint("offline_hard")
        with self.assertRaises(TypeError):
            CANDIDATE_ENTRYPOINT_CATALOG["new_task"] = template
        with self.assertRaises(FrozenInstanceError):
            template.module = "posttrain_circuits.cli.run_grpo"
        with self.assertRaises(KeyError):
            candidate_entrypoint("unregistered_task")

    def test_uses_the_single_artifact_canonical_contract(self):
        value = {"z": [3, 2, 1], "a": {"unicode": "因果"}}
        self.assertIs(canonical_json, artifact_canonical_json)
        self.assertEqual(canonical_json(value), artifact_canonical_json(value))
        self.assertEqual(sha256_value(value), sha256_value(value))

    def test_scientific_training_tasks_are_not_collapsed(self):
        factorial = {
            "offline_hard",
            "online_hard",
            "offline_soft",
            "online_soft_opd",
            "offline_verified_replay",
            "online_verified_replay",
        }
        anchors = {"canonical_sft", "canonical_grpo"}
        controls = {"grpo_format_reward", "grpo_random_reward"}
        self.assertTrue(factorial | anchors | controls <= WORKFLOW_TASK_ALLOWLIST)
        self.assertEqual(len(factorial), 6)
        for task in factorial | anchors | controls:
            self.assertEqual(
                WORKFLOW_TASK_REGISTRY[task],
                (
                    "config_binding_sha256",
                    "execution_config_sha256",
                    "experiment_binding_sha256",
                    "resolved_config_sha256",
                    "scientific_config_sha256",
                ),
            )
        self.assertNotIn("opd-train", WORKFLOW_TASK_ALLOWLIST)
        self.assertNotIn("rl-train", WORKFLOW_TASK_ALLOWLIST)
        for task, names in WORKFLOW_TASK_REGISTRY.items():
            with self.subTest(task=task):
                self.assertTrue(names)
                self.assertEqual(names, tuple(sorted(names)))
                self.assertEqual(len(names), len(set(names)))

    def test_strict_plan_round_trip_and_hash(self):
        first = _unit("a", "offline_hard", outputs=("checkpoint.bin",))
        second = _unit(
            "b",
            "discover_circuit",
            dependencies=(Dependency("a", "checkpoint.bin"),),
            outputs=("candidate.json",),
        )
        plan = WorkflowPlan(workflow_id="pilot", units=(first, second))
        payload = plan.to_payload()
        restored = WorkflowPlan.from_payload(copy.deepcopy(payload))
        self.assertEqual(restored, plan)
        self.assertEqual(payload["sha256"], sha256_value(plan.content_payload()))

    def test_task_requires_exact_content_identity_names(self):
        missing = WorkflowUnit(
            unit_id="cell",
            task="offline_hard",
            content_inputs=(),
            dependencies=(),
            output_names=("result.json",),
        )
        with self.assertRaisesRegex(ValueError, "content identities differ"):
            missing.validate(task_registry=WORKFLOW_TASK_REGISTRY)
        extra = WorkflowUnit(
            unit_id="cell",
            task="offline_hard",
            content_inputs=(
                ContentIdentity(
                    "experiment_binding_sha256", _digest("binding"), "file"
                ),
                ContentIdentity(
                    "unexpected_sha256", _digest("unexpected"), "file"
                ),
            ),
            dependencies=(),
            output_names=("result.json",),
        )
        with self.assertRaisesRegex(ValueError, "content identities differ"):
            extra.validate(task_registry=WORKFLOW_TASK_REGISTRY)
        unknown = WorkflowUnit(
            unit_id="cell",
            task="unknown_task",
            content_inputs=(),
            dependencies=(),
            output_names=("result.json",),
        )
        with self.assertRaisesRegex(ValueError, "not allowlisted"):
            unknown.validate(task_registry=WORKFLOW_TASK_REGISTRY)

    def test_directly_constructed_invalid_plan_is_validated_by_all_accessors(self):
        invalid = WorkflowPlan(
            workflow_id="bad",
            units=(
                WorkflowUnit(
                    unit_id="cell",
                    task="offline_hard",
                    content_inputs=(),
                    dependencies=(),
                    output_names=("result.json",),
                ),
            ),
        )
        for operation in (
            lambda: invalid.unit("cell"),
            invalid.sha256,
            invalid.to_payload,
            invalid.content_payload,
        ):
            with self.assertRaises(ValueError):
                operation()

    def test_duplicate_unknown_and_cyclic_dependencies_fail_closed(self):
        registry = {"produce": ("source_sha256",), "consume": ("config_sha256",)}
        first = _unit("a", "produce", registry=registry)
        duplicate = WorkflowPlan(workflow_id="duplicate", units=(first, first))
        with self.assertRaisesRegex(ValueError, "duplicate unit"):
            duplicate.validate(task_registry=registry)

        unknown = WorkflowPlan(
            workflow_id="unknown",
            units=(
                _unit(
                    "a",
                    "consume",
                    dependencies=(Dependency("missing", "result.json"),),
                    registry=registry,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown dependency"):
            unknown.validate(task_registry=registry)

        unknown_output = WorkflowPlan(
            workflow_id="unknown-output",
            units=(
                first,
                _unit(
                    "b",
                    "consume",
                    dependencies=(Dependency("a", "missing.json"),),
                    registry=registry,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown output"):
            unknown_output.validate(task_registry=registry)

        a = _unit(
            "a",
            "produce",
            dependencies=(Dependency("b", "result.json"),),
            registry=registry,
        )
        b = _unit(
            "b",
            "consume",
            dependencies=(Dependency("a", "result.json"),),
            registry=registry,
        )
        cyclic = WorkflowPlan(workflow_id="cycle", units=(a, b))
        with self.assertRaisesRegex(ValueError, "cycle"):
            cyclic.validate(task_registry=registry)

    def test_noncanonical_order_and_duplicate_edges_are_rejected(self):
        registry = {"produce": ("a_sha256", "b_sha256")}
        unit = WorkflowUnit(
            unit_id="cell",
            task="produce",
            content_inputs=(
                ContentIdentity("b_sha256", _digest("b"), "file"),
                ContentIdentity("a_sha256", _digest("a"), "file"),
            ),
            dependencies=(),
            output_names=("result.json",),
        )
        with self.assertRaisesRegex(ValueError, "not canonical"):
            unit.validate(task_registry=registry)

        base = _unit("a", "offline_hard")
        dependent = _unit(
            "b",
            "offline_hard",
            dependencies=(
                Dependency("a", "result.json"),
                Dependency("a", "result.json"),
            ),
        )
        plan = WorkflowPlan(workflow_id="dup-edge", units=(base, dependent))
        with self.assertRaisesRegex(ValueError, "duplicate dependencies"):
            plan.validate()

    def test_strict_payload_rejects_extra_fields_and_hash_tampering(self):
        plan = WorkflowPlan(workflow_id="pilot", units=(_unit("cell", "offline_hard"),))
        extra = plan.to_payload()
        extra["resources"] = {"gpu_count": 4}
        with self.assertRaisesRegex(ValueError, "fields differ"):
            WorkflowPlan.from_payload(extra)
        tampered = plan.to_payload()
        tampered["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            WorkflowPlan.from_payload(tampered)

    def test_deterministic_identity_layers_are_distinct(self):
        plan = WorkflowPlan(workflow_id="pilot", units=(_unit("cell", "offline_hard"),))
        digest = plan.sha256()
        unit_identity = unit_identity_sha256(
            workflow_id=plan.workflow_id, plan_sha256=digest, unit_id="cell"
        )
        output_identity = output_identity_sha256(
            workflow_id=plan.workflow_id,
            plan_sha256=digest,
            unit_id="cell",
            output_name="result.json",
        )
        completion_identity = completion_identity_sha256(
            workflow_id=plan.workflow_id, plan_sha256=digest, unit_id="cell"
        )
        self.assertEqual(len({unit_identity, output_identity, completion_identity}), 3)
        self.assertEqual(
            unit_identity,
            unit_identity_sha256(
                workflow_id=plan.workflow_id, plan_sha256=digest, unit_id="cell"
            ),
        )

    def test_plan_payload_has_no_runtime_selector_fields(self):
        payload = WorkflowPlan(
            workflow_id="pilot", units=(_unit("cell", "offline_hard"),)
        ).to_payload()
        forbidden = {
            "command",
            "cwd",
            "env",
            "execution_profile",
            "gpu",
            "job_id",
            "lease_id",
            "path",
            "resources",
            "retry",
            "state",
            "status",
        }

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value), set())
            return set()

        self.assertFalse(keys(payload) & forbidden)


if __name__ == "__main__":
    unittest.main()
