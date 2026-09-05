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
    build_qwen3_v2_gpu_preflight_plan,
    fixed_resolved_config,
    prepare_qwen3_v2_gpu_preflight_request,
)
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.outbox import validate_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
    GPU_COUNT,
    OUTPUT_NAME,
    PROFILE_NAME,
    TASK_NAME,
    UNIT_ID,
    WORKFLOW_ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_COMMIT = "a" * 40


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
        self.assertEqual(second["resource_budget"]["node_memory_gib"], 192)
        self.assertIs(second["model"]["local_files_only"], True)
        self.assertIs(second["teacher"]["local_files_only"], True)
        self.assertEqual(second["model"]["model_name_or_path"], "Qwen/Qwen3-1.7B")
        self.assertEqual(second["teacher"]["model_name_or_path"], "Qwen/Qwen3-8B")

    def test_clean_commit_checks_all_untracked_files_and_submodules(self) -> None:
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.require_git_output",
                side_effect=("", CODE_COMMIT),
            ) as git,
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.unsafe_untracked_paths",
                return_value=(),
            ),
        ):
            self.assertEqual(_clean_git_commit(PROJECT_ROOT), CODE_COMMIT)
        self.assertEqual(
            git.call_args_list[0].args[1],
            (
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ),
        )
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.require_git_output",
                return_value="?? untracked.py",
            ),
            self.assertRaisesRegex(AdapterValidationError, "clean checkout"),
        ):
            _clean_git_commit(PROJECT_ROOT)

        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.require_git_output",
                return_value="",
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.unsafe_untracked_paths",
                return_value=("src/ignored.py",),
            ),
            self.assertRaisesRegex(AdapterValidationError, "ignored untracked"),
        ):
            _clean_git_commit(PROJECT_ROOT)

    def test_plan_binds_exact_config_and_raw_preregistration_file(self) -> None:
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.gpu_preflight_request._accepted_lineage_git_commit",
            return_value=CODE_COMMIT,
        ):
            plan = build_qwen3_v2_gpu_preflight_plan(layout=self.layout)
        unit = plan.unit(UNIT_ID)
        self.assertEqual(plan.workflow_id, WORKFLOW_ID)
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
                "distributed_process_count": GPU_COUNT,
                "execution_profile": PROFILE_NAME,
                "scheduler_protocol": 2,
            },
        )
        self.assertNotIn("repository_snapshot", json.dumps(plan.to_payload()))

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
            self.assertEqual(first.plan_sha256, second.plan_sha256)
            self.assertEqual(first.plan_path, second.plan_path)
            self.assertNotEqual(first.outbox_path, second.outbox_path)
            first_request = json.loads(first.outbox_path.read_text(encoding="utf-8"))
            second_request = json.loads(second.outbox_path.read_text(encoding="utf-8"))
            validate_outbox_request(first_request)
            validate_outbox_request(second_request)
        self.assertNotEqual(first_request["job_id"], second_request["job_id"])
        self.assertEqual(first_request["task"], TASK_NAME)
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
            {"command", "cwd", "env", "execution_profile", "path", "resources"}
            & set(first_request)
        )
        self.assertNotIn("repository_snapshot", json.dumps(first_request))

    def test_request_commit_requires_current_accepted_lineage(self) -> None:
        binding = object()
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._clean_git_commit",
                return_value=CODE_COMMIT,
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
            configured_path="prereg/amendments/qwen3_v2_g0_2gpu_v1.yaml",
            expected_head=CODE_COMMIT,
        )
        validate.assert_called_once_with(
            code_root=PROJECT_ROOT,
            candidate_commit=CODE_COMMIT,
            current_binding=binding,
            expected_head=CODE_COMMIT,
            role="GPU preflight request commit",
        )

    def test_request_commit_rejects_unaccepted_lineage(self) -> None:
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request._clean_git_commit",
                return_value=CODE_COMMIT,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.gpu_preflight_request.resolve_accepted_protocol_amendment",
                side_effect=ProtocolAmendmentError("amendment remains proposed"),
            ),
            self.assertRaisesRegex(AdapterValidationError, "implementation lineage"),
        ):
            _accepted_lineage_git_commit(PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
