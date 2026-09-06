from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from posttrain_circuits.artifacts.protocol_amendments import (
    AMENDMENT_RELATIVE_PATH,
    load_protocol_amendment_bytes,
)
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.g0_request import (
    GpuPreflightEvidence,
    _require_clean_checkout,
    build_qwen3_v2_g0_plan,
    fixed_resolved_config,
    prepare_qwen3_v2_g0_request,
)
from posttrain_circuits.scheduler_adapter.outbox import validate_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
    OUTPUT_NAMES,
    TASK_NAME,
    UNIT_ID,
    WORKFLOW_ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
IMPLEMENTATION_COMMIT = "b" * 40


class G0RequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix=".g0-request-", dir=PROJECT_ROOT)
        root = Path(self.temporary.name)
        self.layout = WorkflowLayout(
            code_root=PROJECT_ROOT,
            data_root=root / "data",
            scratch_root=root / "scratch",
        )
        self.layout.data_root.mkdir()
        self.layout.scratch_root.mkdir()
        self.evidence = {}
        for world_size in (1, 2, 3, 4):
            report_raw = f'{{"world_size":{world_size}}}\n'.encode()
            completion_raw = f'{{"completion_world_size":{world_size}}}\n'.encode()
            self.evidence[world_size] = GpuPreflightEvidence(
                world_size=world_size,
                allocation_sha256=str(world_size) * 64,
                workflow_id=(
                    "qwen3-v2-gpu-preflight-elastic-"
                    + format(world_size, "032x")
                ),
                report_raw=report_raw,
                report_file_sha256=hashlib.sha256(report_raw).hexdigest(),
                completion_raw=completion_raw,
                completion_file_sha256=hashlib.sha256(completion_raw).hexdigest(),
                git_commit=COMMIT,
            )
        proposed = load_protocol_amendment_bytes(
            (PROJECT_ROOT / AMENDMENT_RELATIVE_PATH).read_bytes()
        )
        accepted = copy.deepcopy(proposed)
        accepted["review"] = {
            "status": "accepted",
            "reviewed_implementation_commit": IMPLEMENTATION_COMMIT,
            "reviewer": "independent-reviewer-id",
            "reviewed_at_utc": "2026-09-04T05:00:00Z",
            "rationale": "Scheduler-managed G0 implementation accepted.",
        }
        self.amendment_path = root / "accepted-amendment.yaml"
        self.amendment_path.write_text(
            yaml.safe_dump(accepted, sort_keys=False),
            encoding="utf-8",
        )
        self.amendment_sha256 = hashlib.sha256(
            self.amendment_path.read_bytes()
        ).hexdigest()
        self.amendment = SimpleNamespace(
            path=self.amendment_path,
            amendment_id="qwen3_v2_g0_elastic_v1",
            sha256=self.amendment_sha256,
            git_commit="c" * 40,
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_clean_commit_checks_all_untracked_files_and_submodules(self) -> None:
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.require_git_output",
                side_effect=("", COMMIT),
            ) as git,
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.unsafe_untracked_paths",
                return_value=(),
            ),
        ):
            self.assertEqual(_require_clean_checkout(PROJECT_ROOT), COMMIT)
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
                "posttrain_circuits.scheduler_adapter.g0_request.require_git_output",
                return_value="",
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.unsafe_untracked_paths",
                return_value=("src/ignored.py",),
            ),
            self.assertRaisesRegex(AdapterValidationError, "ignored untracked"),
        ):
            _require_clean_checkout(PROJECT_ROOT)

    def _build(self):  # type: ignore[no-untyped-def]
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request._require_clean_checkout",
                return_value=COMMIT,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.validate_gpu_preflight_evidence",
                side_effect=lambda **kwargs: self.evidence[
                    kwargs["expected_world_size"]
                ],
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.resolve_accepted_protocol_amendment",
                return_value=self.amendment,
            ),
        ):
            return build_qwen3_v2_g0_plan(
                layout=self.layout,
                gpu_preflight_reports={
                    world_size: Path(f"/unused/w{world_size}-report.json")
                    for world_size in (1, 2, 3, 4)
                },
                gpu_preflight_completions={
                    world_size: Path(f"/unused/w{world_size}-completion.json")
                    for world_size in (1, 2, 3, 4)
                },
            )

    def test_fixed_config_binds_full_preflight_matrix_without_allocation(self) -> None:
        config = fixed_resolved_config(
            code_commit=COMMIT,
            gpu_preflight_evidence=self.evidence,
            protocol_amendment_sha256=self.amendment_sha256,
            reviewed_implementation_commit=IMPLEMENTATION_COMMIT,
        )
        self.assertEqual(config["scheduler_g0"]["request_git_commit"], COMMIT)
        self.assertEqual(set(config["scheduler_g0"]["gpu_preflight_matrix"]), {"1", "2", "3", "4"})
        self.assertEqual(config["trainer"]["global_batch_size"], 64)
        serialized = json.dumps(config, sort_keys=True)
        for forbidden in ("execution_profile", "process_count", "gpu_count", "resource_budget"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(
            config["scheduler_g0"]["reviewed_implementation_commit"],
            IMPLEMENTATION_COMMIT,
        )
        self.assertEqual(config["g0_profile"], "qwen3_v2_eap_separation")

    def test_plan_has_exact_prerequisite_content_and_outputs(self) -> None:
        plan = self._build()
        unit = plan.unit(UNIT_ID)
        self.assertEqual(plan.workflow_id, WORKFLOW_ID)
        self.assertEqual(unit.task, TASK_NAME)
        self.assertEqual(unit.output_names, OUTPUT_NAMES)
        self.assertEqual(
            tuple(identity.name for identity in unit.content_inputs),
            (
                "config_binding_sha256",
                "execution_config_sha256",
                "gpu_preflight_w1_completion_sha256",
                "gpu_preflight_w1_report_sha256",
                "gpu_preflight_w2_completion_sha256",
                "gpu_preflight_w2_report_sha256",
                "gpu_preflight_w3_completion_sha256",
                "gpu_preflight_w3_report_sha256",
                "gpu_preflight_w4_completion_sha256",
                "gpu_preflight_w4_report_sha256",
                "preregistration_sha256",
                "protocol_amendment_sha256",
                "resolved_config_sha256",
                "scientific_config_sha256",
            ),
        )
        identities = {identity.name: identity for identity in unit.content_inputs}
        amendment_identity = identities["protocol_amendment_sha256"]
        self.assertEqual(amendment_identity.sha256, self.amendment_sha256)
        self.assertEqual(
            self.layout.content_path(
                kind="file",
                sha256=amendment_identity.sha256,
            ).read_bytes(),
            self.amendment_path.read_bytes(),
        )
        binding = json.loads(
            self.layout.content_path(
                kind="file",
                sha256=identities["config_binding_sha256"].sha256,
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            binding["input_artifact_hashes"]["protocol_amendment_path"],
            self.amendment_sha256,
        )

    def test_request_has_only_scientific_identity_and_no_profile_or_resource_override(self) -> None:
        class FixedHandler:
            @staticmethod
            def validate_unit_contract(unit) -> None:  # type: ignore[no-untyped-def]
                self.assertEqual(unit.task, TASK_NAME)

        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.build_qwen3_v2_g0_plan",
                return_value=self._build(),
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.outbox.require_handler",
                return_value=FixedHandler(),
            ),
        ):
            receipt = prepare_qwen3_v2_g0_request(
                layout=self.layout,
                gpu_preflight_reports={
                    world_size: Path(f"/unused/w{world_size}-report.json")
                    for world_size in (1, 2, 3, 4)
                },
                gpu_preflight_completions={
                    world_size: Path(f"/unused/w{world_size}-completion.json")
                    for world_size in (1, 2, 3, 4)
                },
            )
            payload = json.loads(receipt.outbox_path.read_text(encoding="utf-8"))
            validate_outbox_request(payload)
        self.assertEqual(payload["task"], TASK_NAME)
        self.assertEqual(
            payload["parameters"],
            {
                "plan_sha256": receipt.plan_sha256,
                "unit_id": UNIT_ID,
                "workflow_id": WORKFLOW_ID,
            },
        )
        self.assertFalse(
            {"command", "cwd", "env", "execution_profile", "path", "resources"}
            & set(payload)
        )


if __name__ == "__main__":
    unittest.main()
