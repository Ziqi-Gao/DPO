from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from posttrain_circuits.scheduler_adapter.outbox import validate_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.preflight_request import (
    UNIT_ID,
    WORKFLOW_ID,
    build_repository_preflight_plan,
    prepare_repository_preflight_request,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PreflightRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".preflight-request-test-",
            dir=PROJECT_ROOT,
        )
        root = Path(self._temporary.name)
        self.layout = WorkflowLayout(
            code_root=root / "code",
            data_root=root / "data",
            scratch_root=root / "scratch",
        )
        for path in (
            self.layout.code_root,
            self.layout.data_root,
            self.layout.scratch_root,
        ):
            path.mkdir()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_plan_contains_only_required_config_binding_inputs(self) -> None:
        plan = build_repository_preflight_plan(layout=self.layout)
        unit = plan.unit(UNIT_ID)
        self.assertEqual(plan.workflow_id, WORKFLOW_ID)
        self.assertEqual(
            tuple(identity.name for identity in unit.content_inputs),
            (
                "config_binding_sha256",
                "execution_config_sha256",
                "resolved_config_sha256",
                "scientific_config_sha256",
            ),
        )
        self.assertEqual(unit.output_names, ("preflight_report.json",))

    def test_formal_outbox_preparation_uses_fresh_job_ids_and_has_no_resource_surface(self) -> None:
        first = prepare_repository_preflight_request(layout=self.layout)
        second = prepare_repository_preflight_request(layout=self.layout)
        self.assertEqual(first.workflow_id, second.workflow_id)
        self.assertEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(first.unit_id, second.unit_id)
        self.assertEqual(first.plan_path, second.plan_path)
        self.assertNotEqual(first.outbox_path, second.outbox_path)
        first_request = json.loads(first.outbox_path.read_text(encoding="utf-8"))
        second_request = json.loads(second.outbox_path.read_text(encoding="utf-8"))
        validate_outbox_request(first_request)
        validate_outbox_request(second_request)
        self.assertNotEqual(first_request["job_id"], second_request["job_id"])
        self.assertEqual(first_request["task"], "repository_preflight")
        self.assertEqual(
            first_request["parameters"],
            {
                "plan_sha256": first.plan_sha256,
                "unit_id": UNIT_ID,
                "workflow_id": WORKFLOW_ID,
            },
        )
        self.assertEqual(first_request["parameters"], second_request["parameters"])
        self.assertFalse(
            {
                "command",
                "cwd",
                "env",
                "execution_profile",
                "path",
                "resources",
            }
            & set(first_request)
        )


if __name__ == "__main__":
    unittest.main()
