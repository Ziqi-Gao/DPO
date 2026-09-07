from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from posttrain_circuits.artifacts.protocol_amendments import ProtocolAmendmentError
from posttrain_circuits.scheduler_adapter.gpu_preflight_request import (
    PREREG_CONTENT_NAME,
    _accepted_lineage_git_commit,
    _clean_git_commit,
    _execution_class_successor_present,
    build_qwen3_v2_gpu_preflight_plan,
    fixed_resolved_config,
    prepare_qwen3_v2_gpu_preflight_request,
)
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.outbox import validate_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
    OUTPUT_NAME,
    TASK_NAME,
    UNIT_ID,
    WORKFLOW_ID_PREFIX,
    validate_preflight_workflow_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_COMMIT = "a" * 40
FIXTURE_WORKFLOW_ID = f"{WORKFLOW_ID_PREFIX}{'1' * 32}"


class GpuPreflightRequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".gpu-preflight-request-test-",
            dir=PROJECT_ROOT,
        )
        root = Path(self._temporary.name)
        self.layout = WorkflowLayout(
            code_root=PROJECT_ROOT,
            data_root=root / "data",
            scratch_root=root / "scratch",
        )
        self.layout.data_root.mkdir()
        self.layout.scratch_root.mkdir()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _content_bytes(self, *, sha256: str) -> bytes:
        return self.layout.content_path(kind="file", sha256=sha256).read_bytes()

    def test_fixed_config_is_complete_local_only_and_fresh(self) -> None:
        first = fixed_resolved_config(code_commit=CODE_COMMIT)
        second = fixed_resolved_config(code_commit=CODE_COMMIT)
        first["model"]["model_revision"] = "tampered"
        self.assertNotEqual(first, second)
        self.assertEqual(second["config_kind"], TASK_NAME)
        self.assertEqual(second["code_commit"], CODE_COMMIT)
        self.assertNotIn("resource_budget", second)
        self.assertIs(second["model"]["local_files_only"], True)
        self.assertIs(second["teacher"]["local_files_only"], True)
        self.assertEqual(second["model"]["model_name_or_path"], "Qwen/Qwen3-1.7B")
        self.assertEqual(second["teacher"]["model_name_or_path"], "Qwen/Qwen3-8B")

    def test_clean_commit_checks_tracked_files_and_submodules(self) -> None:
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.gpu_preflight_request.require_git_output",
            side_effect=("", CODE_COMMIT),
        ) as git:
            self.assertEqual(_clean_git_commit(PROJECT_ROOT), CODE_COMMIT)
        self.assertEqual(
            git.call_args_list[0].args[1],
            (
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "--ignore-submodules=none",
            ),
        )
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.require_git_output",
                return_value=" M tracked.py",
            ),
            self.assertRaisesRegex(AdapterValidationError, "clean checkout"),
        ):
            _clean_git_commit(PROJECT_ROOT)

    def test_plan_binds_exact_config_and_raw_preregistration_file(self) -> None:
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.gpu_preflight_request._accepted_lineage_git_commit",
            return_value=CODE_COMMIT,
        ):
            plan = build_qwen3_v2_gpu_preflight_plan(
                layout=self.layout,
                workflow_id=FIXTURE_WORKFLOW_ID,
            )
        unit = plan.unit(UNIT_ID)
        self.assertEqual(plan.workflow_id, FIXTURE_WORKFLOW_ID)
        self.assertEqual(unit.task, TASK_NAME)
        self.assertEqual(unit.output_names, (OUTPUT_NAME,))
        self.assertEqual(
            tuple(identity.name for identity in unit.content_inputs),
            (
                "config_binding_sha256",
                "execution_config_sha256",
                "preregistration_sha256",
                "resolved_config_sha256",
                "scientific_config_sha256",
            ),
        )
        identities = {identity.name: identity for identity in unit.content_inputs}
        preregistration = identities[PREREG_CONTENT_NAME]
        self.assertEqual(
            self._content_bytes(sha256=preregistration.sha256),
            (PROJECT_ROOT / "prereg" / "qwen3_v2.yaml").read_bytes(),
        )
        binding = json.loads(
            self._content_bytes(sha256=identities["config_binding_sha256"].sha256)
        )
        resolved = json.loads(
            self._content_bytes(sha256=identities["resolved_config_sha256"].sha256)
        )
        scientific = json.loads(
            self._content_bytes(sha256=identities["scientific_config_sha256"].sha256)
        )
        execution = json.loads(
            self._content_bytes(sha256=identities["execution_config_sha256"].sha256)
        )
        self.assertEqual(
            binding["input_artifact_hashes"],
            {"prereg_path": preregistration.sha256},
        )
        self.assertEqual(resolved, fixed_resolved_config(code_commit=CODE_COMMIT))
        self.assertEqual(
            scientific["config"]["prereg_path"],
            {"content_sha256": preregistration.sha256},
        )
        self.assertEqual(
            execution["storage_locators"], {"prereg_path": "prereg/qwen3_v2.yaml"}
        )
        self.assertEqual(
            execution["execution_context"],
            {
                "allocation_contract": "manifest_driven_scheduler_gpu_v1",
                "scheduler_protocol": 2,
            },
        )
        serialized_plan = json.dumps(plan.to_payload())
        self.assertNotIn("repository_snapshot", serialized_plan)
        self.assertNotIn("resource_budget", serialized_plan)
        self.assertNotIn("distributed_process_count", serialized_plan)
        self.assertNotIn("execution_profile", serialized_plan)
        self.assertNotIn("gpu_count", serialized_plan)

    def test_outbox_is_fresh_scientific_only_and_has_no_profile_or_resource_surface(self) -> None:
        class FixedHandler:
            @staticmethod
            def validate_unit_contract(unit) -> None:  # type: ignore[no-untyped-def]
                if unit.task != TASK_NAME:
                    raise AssertionError("request builder selected the wrong task")

        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._accepted_lineage_git_commit",
                return_value=CODE_COMMIT,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.outbox.require_handler",
                return_value=FixedHandler(),
            ),
        ):
            first = prepare_qwen3_v2_gpu_preflight_request(layout=self.layout)
            second = prepare_qwen3_v2_gpu_preflight_request(layout=self.layout)
            self.assertNotEqual(first.workflow_id, second.workflow_id)
            self.assertNotEqual(first.plan_sha256, second.plan_sha256)
            self.assertNotEqual(first.plan_path, second.plan_path)
            self.assertNotEqual(first.outbox_path, second.outbox_path)
            first_request = json.loads(first.outbox_path.read_text(encoding="utf-8"))
            second_request = json.loads(second.outbox_path.read_text(encoding="utf-8"))
            validate_outbox_request(first_request)
            validate_outbox_request(second_request)
        self.assertNotEqual(first_request["job_id"], second_request["job_id"])
        self.assertEqual(
            validate_preflight_workflow_id(first.workflow_id), first.workflow_id
        )
        self.assertEqual(
            validate_preflight_workflow_id(second.workflow_id), second.workflow_id
        )
        self.assertEqual(first_request["task"], TASK_NAME)
        self.assertEqual(
            first_request["parameters"],
            {
                "plan_sha256": first.plan_sha256,
                "unit_id": UNIT_ID,
                "workflow_id": first.workflow_id,
            },
        )
        self.assertEqual(
            second_request["parameters"],
            {
                "plan_sha256": second.plan_sha256,
                "unit_id": UNIT_ID,
                "workflow_id": second.workflow_id,
            },
        )
        self.assertFalse(
            {"command", "cwd", "env", "execution_profile", "path", "resources"}
            & set(first_request)
        )
        self.assertNotIn("repository_snapshot", json.dumps(first_request))

    def test_request_commit_requires_current_accepted_lineage(self) -> None:
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._clean_git_commit",
                return_value=CODE_COMMIT,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._execution_class_successor_present",
                return_value=True,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.resolve_accepted_execution_class_amendment",
                return_value=object(),
            ) as resolve,
        ):
            self.assertEqual(_accepted_lineage_git_commit(PROJECT_ROOT), CODE_COMMIT)
        resolve.assert_called_once_with(
            code_root=PROJECT_ROOT,
            configured_path="prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml",
            expected_head=CODE_COMMIT,
        )

    def test_historical_checkout_without_successor_uses_v1_lineage(self) -> None:
        binding = object()
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._clean_git_commit",
                return_value=CODE_COMMIT,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._execution_class_successor_present",
                return_value=False,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.resolve_accepted_protocol_amendment",
                return_value=binding,
            ) as resolve,
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.validate_accepted_lineage_commit"
            ) as validate,
        ):
            self.assertEqual(_accepted_lineage_git_commit(PROJECT_ROOT), CODE_COMMIT)
        resolve.assert_called_once_with(
            code_root=PROJECT_ROOT,
            configured_path="prereg/amendments/qwen3_v2_g0_elastic_v1.yaml",
            expected_head=CODE_COMMIT,
        )
        validate.assert_called_once_with(
            code_root=PROJECT_ROOT,
            candidate_commit=CODE_COMMIT,
            current_binding=binding,
            expected_head=CODE_COMMIT,
            role="GPU preflight request commit",
        )

    def test_successor_presence_check_distinguishes_only_missing_file(self) -> None:
        self.assertIs(_execution_class_successor_present(PROJECT_ROOT), True)

    def test_request_commit_rejects_unaccepted_lineage(self) -> None:
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._clean_git_commit",
                return_value=CODE_COMMIT,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._execution_class_successor_present",
                return_value=True,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.resolve_accepted_execution_class_amendment",
                side_effect=ProtocolAmendmentError("amendment remains proposed"),
            ),
            self.assertRaisesRegex(AdapterValidationError, "implementation lineage"),
        ):
            _accepted_lineage_git_commit(PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
