from __future__ import annotations

import copy
import inspect
import tomllib
import unittest
from pathlib import Path

from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.registration_proposal import (
    CPU_CORES,
    CPU_PROFILE_NAME,
    ENTRYPOINT,
    ESTIMATED_RUNTIME_SECONDS,
    INTEGRATION_STATUS,
    MEMORY_MIB,
    TASK_NAME,
    build_disabled_registration_proposal,
    build_registration_proposal_schema,
    validate_disabled_registration_proposal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_PATH = (
    PROJECT_ROOT
    / "deployments"
    / "repository_preflight"
    / "registration-proposal-v2.toml"
)


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


class RegistrationProposalTests(unittest.TestCase):
    def test_checked_in_toml_is_the_exact_disabled_proposal(self):
        with PROPOSAL_PATH.open("rb") as handle:
            proposal = tomllib.load(handle)
        self.assertEqual(proposal, build_disabled_registration_proposal())
        validate_disabled_registration_proposal(proposal)

    def test_builder_is_parameterless_disabled_and_protocol_v2(self):
        self.assertEqual(
            tuple(inspect.signature(build_disabled_registration_proposal).parameters),
            (),
        )
        proposal = build_disabled_registration_proposal()
        validate_disabled_registration_proposal(proposal)
        self.assertEqual(proposal["schema_version"], 2)
        self.assertIs(proposal["enabled"], False)
        self.assertEqual(proposal["integration_status"], INTEGRATION_STATUS)
        self.assertEqual(proposal["name"], "OPD")
        self.assertEqual(proposal["integration"]["entrypoint"], str(ENTRYPOINT))
        self.assertEqual(proposal["integration"]["allowed_tasks"], [TASK_NAME])
        self.assertTrue(str(ENTRYPOINT).startswith(proposal["paths"]["code_root"] + "/"))

    def test_proposal_has_one_measured_cpu_only_profile(self):
        proposal = build_disabled_registration_proposal()
        self.assertEqual(len(proposal["tasks"]), 1)
        task = proposal["tasks"][0]
        self.assertEqual(task["name"], TASK_NAME)
        self.assertEqual(task["selection_policy"], "earliest_finish")
        self.assertEqual(len(task["execution_profiles"]), 1)
        profile = task["execution_profiles"][0]
        self.assertEqual(profile["name"], CPU_PROFILE_NAME)
        self.assertEqual(profile["kind"], "cpu")
        self.assertEqual(profile["resource_mode"], "fixed")
        self.assertEqual(
            profile["cpu_cores_min"],
            profile["cpu_cores_preferred"],
        )
        self.assertEqual(profile["cpu_cores_preferred"], profile["cpu_cores_max"])
        self.assertEqual(profile["cpu_cores_min"], CPU_CORES)
        self.assertEqual(profile["cpu_cores_preferred"], CPU_CORES)
        self.assertEqual(profile["cpu_cores_max"], CPU_CORES)
        self.assertEqual(profile["memory_mib"], MEMORY_MIB)
        self.assertEqual(
            profile["estimated_runtime_seconds"], ESTIMATED_RUNTIME_SECONDS
        )
        self.assertEqual(profile["gpu_count"], 0)
        self.assertEqual(profile["gpu_memory_mib"], 0)
        self.assertEqual(profile["gpu_utilization_pct"], 0)
        self.assertEqual(profile["gpu_exclusivity"], "shareable")
        self.assertEqual(profile["gpu_models"], [])
        self.assertIs(proposal["policy"]["allow_gpu_jobs"], False)
        self.assertIn(
            "23,552 KiB MaxRSS",
            proposal["integration"]["notes"],
        )

    def test_schema_is_an_exact_fresh_copy(self):
        first = build_registration_proposal_schema()
        second = build_registration_proposal_schema()
        self.assertEqual(first["const"], build_disabled_registration_proposal())
        first["const"]["enabled"] = True
        self.assertIs(second["const"]["enabled"], False)

    def test_builder_returns_fresh_payloads(self):
        first = build_disabled_registration_proposal()
        first["paths"]["code_root"] = "/unreviewed"
        second = build_disabled_registration_proposal()
        self.assertNotEqual(first, second)
        self.assertEqual(second["paths"]["code_root"], "/home/del6500/projects/OPD")

    def test_unreviewed_enable_task_profile_path_and_resource_edits_fail_closed(self):
        cases = (
            (("enabled",), True),
            (("integration", "allowed_tasks"), ["repository_preflight", "train"]),
            (("integration", "entrypoint"), "/unreviewed/entrypoint"),
            (("paths", "scratch_root"), "/unreviewed/scratch"),
            (("tasks", 0, "name"), "offline_hard"),
            (("tasks", 0, "execution_profiles", 0, "name"), "unreviewed-profile"),
            (("tasks", 0, "execution_profiles", 0, "memory_mib"), 2),
        )
        for path, replacement in cases:
            with self.subTest(path=path):
                proposal = build_disabled_registration_proposal()
                target = proposal
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = replacement
                with self.assertRaises(AdapterValidationError):
                    validate_disabled_registration_proposal(proposal)

    def test_extra_command_environment_and_manifest_override_fields_are_rejected(self):
        forbidden = {"argv", "command", "cwd", "env", "parameters", "resources"}
        proposal = build_disabled_registration_proposal()
        self.assertFalse(_nested_keys(proposal) & forbidden)
        for key in forbidden:
            with self.subTest(key=key):
                tampered = copy.deepcopy(proposal)
                tampered["integration"][key] = "unreviewed"
                with self.assertRaises(AdapterValidationError):
                    validate_disabled_registration_proposal(tampered)

    def test_bool_does_not_alias_a_numeric_resource(self):
        proposal = build_disabled_registration_proposal()
        proposal["tasks"][0]["execution_profiles"][0]["memory_mib"] = True
        with self.assertRaisesRegex(AdapterValidationError, "wrong JSON type"):
            validate_disabled_registration_proposal(proposal)


if __name__ == "__main__":
    unittest.main()
