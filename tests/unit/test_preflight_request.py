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

    def test_formal_outbox_preparation_is_idempotent_and_has_no_resource_surface(self) -> None:
        first = prepare_repository_preflight_request(layout=self.layout)
        second = prepare_repository_preflight_request(layout=self.layout)
        self.assertEqual(first, second)
        request = json.loads(first.outbox_path.read_text(encoding="utf-8"))
        validate_outbox_request(request)
        self.assertEqual(request["task"], "repository_preflight")
        self.assertEqual(
            request["parameters"],
            {
                "plan_sha256": first.plan_sha256,
                "unit_id": UNIT_ID,
                "workflow_id": WORKFLOW_ID,
            },
        )
        self.assertFalse(
            {
                "command",
                "cwd",
                "env",
                "execution_profile",
                "path",
                "resources",
            }
            & set(request)
        )


if __name__ == "__main__":
    unittest.main()
