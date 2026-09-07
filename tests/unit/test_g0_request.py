from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from posttrain_circuits.artifacts.execution_safety_certification import (
    CERTIFICATION_ID,
    CERTIFICATION_RELATIVE_PATH,
    DESCRIPTOR_RELATIVE_PATH,
    EXECUTION_CLASS_ID,
    EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION,
)
from posttrain_circuits.artifacts.execution_science_protocol import (
    canonical_science_config_sha256,
)
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.g0_request import (
    BASE_CONFIG_OVERRIDES,
    CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
    build_execution_science_g0_plan,
    build_qwen3_v2_g0_plan,
    fixed_resolved_config,
    new_qwen3_v2_g0_workflow_id,
    prepare_execution_science_g0_request,
    resolved_config_for_execution_science,
    _require_clean_checkout,
)
from posttrain_circuits.scheduler_adapter.outbox import validate_outbox_request
from posttrain_circuits.scheduler_adapter.paths import WorkflowLayout
from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
    OUTPUT_NAMES,
    TASK_NAME,
    UNIT_ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
CLASS_IMPLEMENTATION_COMMIT = "b" * 40
SCIENCE_ACCEPTANCE_COMMIT = "d" * 40


class G0RequestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".g0-request-", dir=PROJECT_ROOT
        )
        root = Path(self.temporary.name)
        self.layout = WorkflowLayout(
            code_root=PROJECT_ROOT,
            data_root=root / "data",
            scratch_root=root / "scratch",
        )
        self.layout.data_root.mkdir()
        self.layout.scratch_root.mkdir()

        self.descriptor_raw = b'{"execution_class_id":"qwen3-v2-elastic-training-v1"}\n'
        self.certification_raw = b"schema_version: 1\nreview:\n  status: accepted\n"
        self.prereg_raw = b"version: qwen3_v2\n"
        self.amendment_raw = (
            b"amendment_id: qwen3_v2_g0_execution_class_v2\n"
            b"review:\n  status: accepted\n"
        )
        self.science_raw = b"kind: opd_execution_science_protocol\n"
        self.fingerprint = "f" * 64
        self.execution_safety = SimpleNamespace(
            execution_class_id=EXECUTION_CLASS_ID,
            certification_id=CERTIFICATION_ID,
            fingerprint_sha256=self.fingerprint,
            descriptor_sha256=hashlib.sha256(self.descriptor_raw).hexdigest(),
            certification_sha256=hashlib.sha256(self.certification_raw).hexdigest(),
            certification_core_sha256="6" * 64,
            supported_world_sizes=(1, 2, 3, 4),
            review_status="accepted",
            reviewed_implementation_commit=CLASS_IMPLEMENTATION_COMMIT,
            evidence_basis={},
        )
        self.descriptor_payload = {
            "subject": {
                "resolved_config_safety_projection": copy.deepcopy(
                    EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION
                )
            }
        }
        science_config = compose_config(
            [*BASE_CONFIG_OVERRIDES, "seed=42"],
            config_root=PROJECT_ROOT / "configs",
        )
        self.science_execution_class = {
            "execution_class_id": EXECUTION_CLASS_ID,
            "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
            "descriptor_sha256": self.execution_safety.descriptor_sha256,
            "fingerprint_sha256": self.fingerprint,
            "certification_id": CERTIFICATION_ID,
            "certification_path": str(CERTIFICATION_RELATIVE_PATH),
            "certification_core_sha256": (
                self.execution_safety.certification_core_sha256
            ),
            "supported_world_sizes": [1, 2, 3, 4],
        }
        self.science = SimpleNamespace(
            path=PROJECT_ROOT / CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
            relative_path=CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
            raw=self.science_raw,
            sha256=hashlib.sha256(self.science_raw).hexdigest(),
            acceptance_commit=SCIENCE_ACCEPTANCE_COMMIT,
            reviewed_implementation_commit=CLASS_IMPLEMENTATION_COMMIT,
            binding=SimpleNamespace(
                payload={"execution_class": self.science_execution_class},
                protocol_id="qwen3-v2-g0-candidate-e-seed-42-v1",
                unit_id=UNIT_ID,
                seed=42,
                execution_class_id=EXECUTION_CLASS_ID,
                execution_safety_fingerprint_sha256=self.fingerprint,
                storage_neutral_resolved_config_sha256=(
                    canonical_science_config_sha256(science_config)
                ),
                hydra_override_vector=(*BASE_CONFIG_OVERRIDES, "seed=42"),
                review_status="accepted",
                reviewed_implementation_commit=CLASS_IMPLEMENTATION_COMMIT,
                artifact_sha256=hashlib.sha256(self.science_raw).hexdigest(),
            ),
        )
        self.certification_terms = {
            "execution_class_id": EXECUTION_CLASS_ID,
            "certification_id": CERTIFICATION_ID,
            "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
            "descriptor_sha256": self.execution_safety.descriptor_sha256,
            "certification_path": str(CERTIFICATION_RELATIVE_PATH),
            "certification_core_sha256": (
                self.execution_safety.certification_core_sha256
            ),
            "fingerprint_sha256": self.fingerprint,
            "supported_world_sizes": [1, 2, 3, 4],
        }
        self.amendment_path = root / "accepted-amendment.yaml"
        self.amendment_path.write_bytes(self.amendment_raw)
        self.amendment_sha256 = hashlib.sha256(self.amendment_raw).hexdigest()
        self.amendment = SimpleNamespace(
            path=self.amendment_path,
            amendment_id="qwen3_v2_g0_execution_class_v2",
            sha256=self.amendment_sha256,
            git_commit=SCIENCE_ACCEPTANCE_COMMIT,
            reviewed_implementation_commit=CLASS_IMPLEMENTATION_COMMIT,
            certification_binding=self.execution_safety,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_clean_commit_checks_all_untracked_files_and_submodules(self) -> None:
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.require_git_output",
                side_effect=("", COMMIT),
            ) as git,
        ):
            self.assertEqual(_require_clean_checkout(PROJECT_ROOT), COMMIT)
        self.assertEqual(
            git.call_args_list[0].args[1],
            (
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "--ignore-submodules=none",
            ),
        )

    def _read_artifact(self, path: Path, **_kwargs):  # type: ignore[no-untyped-def]
        if path == PROJECT_ROOT / DESCRIPTOR_RELATIVE_PATH:
            raw = self.descriptor_raw
        elif path == PROJECT_ROOT / CERTIFICATION_RELATIVE_PATH:
            raw = self.certification_raw
        elif path == PROJECT_ROOT / "prereg/qwen3_v2.yaml":
            raw = self.prereg_raw
        elif path == self.amendment_path:
            raw = self.amendment_raw
        else:
            raise AssertionError(f"unexpected artifact read: {path}")
        return raw, hashlib.sha256(raw).hexdigest()

    def _build(
        self,
        *,
        candidate_e: bool = False,
        science=None,  # type: ignore[no-untyped-def]
    ):
        selected_science = self.science if science is None else science
        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request._require_clean_checkout",
                return_value=COMMIT,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.resolve_accepted_execution_science_protocol",
                return_value=selected_science,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.resolve_accepted_execution_class_amendment",
                return_value=self.amendment,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.read_regular_file_nofollow",
                side_effect=self._read_artifact,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.load_execution_class_amendment_bytes",
                return_value={
                    "execution_safety_certification": self.certification_terms,
                    "review": {"status": "accepted"},
                },
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.validate_elastic_g0_config"
            ) as candidate_validator,
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.load_execution_safety_certification_bytes",
                return_value=self.execution_safety,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.load_execution_safety_descriptor_bytes",
                return_value=self.descriptor_payload,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.current_execution_safety_fingerprint",
                return_value=self.fingerprint,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.secrets.token_hex",
                return_value="1" * 32,
            ),
        ):
            plan = build_execution_science_g0_plan(
                layout=self.layout,
                execution_science_protocol_path=selected_science.relative_path,
                require_candidate_e_science=candidate_e,
            )
        if candidate_e:
            candidate_validator.assert_called_once()
        else:
            candidate_validator.assert_not_called()
        return plan

    def test_science_config_binds_class_without_allocation_or_fixed_seed(self) -> None:
        config = fixed_resolved_config(
            code_root=PROJECT_ROOT,
            code_commit=COMMIT,
            execution_science=self.science,
            execution_safety=self.execution_safety,
            protocol_amendment_sha256=self.amendment_sha256,
            reviewed_implementation_commit=CLASS_IMPLEMENTATION_COMMIT,
        )
        scheduler_g0 = config["scheduler_g0"]
        self.assertEqual(scheduler_g0["request_git_commit"], COMMIT)
        self.assertEqual(
            scheduler_g0["execution_science_protocol_sha256"], self.science.sha256
        )
        self.assertEqual(scheduler_g0["execution_safety_class_id"], EXECUTION_CLASS_ID)
        self.assertEqual(config["seed"], 42)
        serialized = json.dumps(config, sort_keys=True)
        for forbidden in (
            '"execution_profile"',
            '"gpu_count"',
            '"gpu_preflight_matrix"',
            '"process_count"',
            '"resources"',
        ):
            self.assertNotIn(forbidden, serialized)

        changed = copy.deepcopy(self.science)
        changed.binding = copy.deepcopy(self.science.binding)
        changed.binding.seed = 43
        changed.binding.hydra_override_vector = (*BASE_CONFIG_OVERRIDES, "seed=43")
        changed_config = compose_config(
            list(changed.binding.hydra_override_vector),
            config_root=PROJECT_ROOT / "configs",
        )
        changed.binding.storage_neutral_resolved_config_sha256 = (
            canonical_science_config_sha256(changed_config)
        )
        resolved = resolved_config_for_execution_science(
            code_root=PROJECT_ROOT,
            code_commit=COMMIT,
            execution_science=changed,
            execution_safety=self.execution_safety,
            protocol_amendment_sha256=self.amendment_sha256,
            reviewed_implementation_commit=CLASS_IMPLEMENTATION_COMMIT,
        )
        self.assertEqual(resolved["seed"], 43)

    def test_plan_has_reusable_class_and_per_experiment_science_cas(self) -> None:
        plan = self._build()
        unit = plan.unit(UNIT_ID)
        self.assertEqual(plan.workflow_id, "qwen3-v2-g0-elastic-" + "1" * 32)
        self.assertEqual(unit.task, TASK_NAME)
        self.assertEqual(unit.output_names, OUTPUT_NAMES)
        self.assertEqual(
            tuple(identity.name for identity in unit.content_inputs),
            (
                "config_binding_sha256",
                "execution_config_sha256",
                "execution_safety_certification_sha256",
                "execution_safety_descriptor_sha256",
                "execution_science_protocol_sha256",
                "preregistration_sha256",
                "protocol_amendment_sha256",
                "resolved_config_sha256",
                "scientific_config_sha256",
            ),
        )
        identities = {identity.name: identity for identity in unit.content_inputs}
        self.assertEqual(
            self.layout.content_path(
                kind="file",
                sha256=identities["execution_science_protocol_sha256"].sha256,
            ).read_bytes(),
            self.science_raw,
        )
        binding = json.loads(
            self.layout.content_path(
                kind="file",
                sha256=identities["config_binding_sha256"].sha256,
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            binding["input_artifact_hashes"]["execution_science_protocol_path"],
            self.science.sha256,
        )
        self.assertNotIn("gpu_preflight_w1_report", binding["input_artifact_hashes"])

    def test_plan_rejects_science_bound_to_a_different_fingerprint(self) -> None:
        changed = copy.deepcopy(self.science)
        changed.binding = copy.deepcopy(self.science.binding)
        changed.binding.payload = copy.deepcopy(self.science.binding.payload)
        changed.binding.payload["execution_class"]["fingerprint_sha256"] = "0" * 64
        with self.assertRaisesRegex(AdapterValidationError, "different execution"):
            self._build(science=changed)

    def test_candidate_e_wrapper_retains_separate_science_gate(self) -> None:
        self._build(candidate_e=True)
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.g0_request.build_execution_science_g0_plan",
            return_value=SimpleNamespace(),
        ) as generic:
            build_qwen3_v2_g0_plan(layout=self.layout)
        generic.assert_called_once_with(
            layout=self.layout,
            execution_science_protocol_path=CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH,
            require_candidate_e_science=True,
        )

    def test_workflow_identity_is_fresh_and_count_neutral(self) -> None:
        with mock.patch(
            "posttrain_circuits.scheduler_adapter.g0_request.secrets.token_hex",
            side_effect=("1" * 32, "2" * 32),
        ):
            first = new_qwen3_v2_g0_workflow_id()
            second = new_qwen3_v2_g0_workflow_id()
        self.assertNotEqual(first, second)
        self.assertNotIn("w1", first)
        self.assertNotIn("4gpu", first)

    def test_request_has_only_scientific_identity_and_no_resource_override(self) -> None:
        plan = self._build()

        class FixedHandler:
            @staticmethod
            def validate_unit_contract(unit) -> None:  # type: ignore[no-untyped-def]
                self.assertEqual(unit.task, TASK_NAME)

        with (
            mock.patch(
                "posttrain_circuits.scheduler_adapter.g0_request.build_execution_science_g0_plan",
                return_value=plan,
            ),
            mock.patch(
                "posttrain_circuits.scheduler_adapter.outbox.require_handler",
                return_value=FixedHandler(),
            ),
        ):
            receipt = prepare_execution_science_g0_request(
                layout=self.layout,
                execution_science_protocol_path=(
                    CANDIDATE_E_SCIENCE_PROTOCOL_RELATIVE_PATH
                ),
            )
            payload = json.loads(receipt.outbox_path.read_text(encoding="utf-8"))
            validate_outbox_request(payload)
        self.assertEqual(payload["task"], TASK_NAME)
        self.assertEqual(
            payload["parameters"],
            {
                "plan_sha256": receipt.plan_sha256,
                "unit_id": UNIT_ID,
                "workflow_id": plan.workflow_id,
            },
        )
        self.assertFalse(
            {"command", "cwd", "env", "execution_profile", "path", "resources"}
            & set(payload)
        )


if __name__ == "__main__":
    unittest.main()
