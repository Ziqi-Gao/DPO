from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.cli.finalize_g0 import G0_CHECK_NAMES
from posttrain_circuits.scheduler_adapter import qwen3_v2_g0 as g0_contract
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
    BUNDLE_NAME,
    GATE_NAMES,
    MODEL_REVISION,
    NODE_MEMORY_BYTES,
    OUTPUT_NAMES,
    PROFILE_NAME,
    PROVENANCE_VALIDATION,
    REPORT_NAME,
    REQUIRED_BUNDLE_PATHS,
    TEACHER_REVISION,
    batch_token_contract,
    execution_context,
    gpu_preflight_completion_content_name,
    gpu_preflight_report_content_name,
    validate_qwen3_v2_g0_completion,
)


class Qwen3V2G0CompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix=".g0-validator-", dir=Path.cwd())
        self.root = Path(self.temporary.name)
        self.inputs = {
            "config_binding_sha256": "1" * 64,
            "execution_config_sha256": "2" * 64,
            "preregistration_sha256": "b" * 64,
            "protocol_amendment_sha256": "c" * 64,
            "resolved_config_sha256": "d" * 64,
            "scientific_config_sha256": "e" * 64,
        }
        evidence_characters = iter("3456789a")
        for world_size in (1, 2, 3, 4):
            self.inputs[gpu_preflight_completion_content_name(world_size)] = (
                next(evidence_characters) * 64
            )
            self.inputs[gpu_preflight_report_content_name(world_size)] = (
                next(evidence_characters) * 64
            )
        self.commit = "8" * 40

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _bytes(payload: object) -> bytes:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()

    def _write_valid_outputs(
        self, world_size: int = 3
    ) -> tuple[dict[str, object], SimpleNamespace]:
        source_workspace = Path(
            f"/scr/del6500/OPD/tmp/qwen3-v2-g0-test{world_size:04d}/qwen3-v2"
        )
        allocation_sha256 = "f" * 64
        contract = batch_token_contract(world_size)
        inner: dict[str, object] = {
            "checks": {name: True for name in G0_CHECK_NAMES},
            "code_commit": self.commit,
            "git_commit": self.commit,
            "model_revision": MODEL_REVISION,
            "passed": True,
            "phase": "G0",
            "prereg_sha256": self.inputs["preregistration_sha256"],
            "protocol_amendment_id": "qwen3_v2_g0_elastic_v1",
            "protocol_amendment_sha256": self.inputs[
                "protocol_amendment_sha256"
            ],
            "reviewed_implementation_commit": "7" * 40,
            "request_git_commit": self.commit,
            "allocation_sha256": allocation_sha256,
            "batch_token_contract": contract,
            "teacher_revision": TEACHER_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "world_size": world_size,
        }
        inner["sha256"] = sha256_value(inner)
        members = {name: b"{}\n" for name in REQUIRED_BUNDLE_PATHS}
        members["g0.json"] = self._bytes(inner)
        members["distributed_resume.json"] = self._bytes({"format_version": 3})
        members[
            "calibration/checkpoints/step-00000020.accelerate/state.bin"
        ] = b"state\n"
        bundle_path = self.root / BUNDLE_NAME
        with tarfile.open(bundle_path, "w:") as archive:
            for name, raw in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                archive.addfile(info, fileobj=__import__("io").BytesIO(raw))
        inventory = [
            {
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
            for name, raw in sorted(members.items())
        ]
        execution = {
            "allocation_sha256": allocation_sha256,
            "attempt": 1,
            "execution_profile": PROFILE_NAME,
            "job_id": "opd-" + "a" * 32,
            "manifest_sha256": "b" * 64,
        }
        peak = 64 * 1024**3
        report: dict[str, object] = {
            "artifact_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            "artifact_inventory": inventory,
            "artifact_namespace": "qwen3-v2",
            "batch_token_contract": contract,
            "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
            "code_commit": self.commit,
            "completed_at": "2026-09-04T05:00:00Z",
            "cgroup_memory": {
                "current_bytes": 32 * 1024**3,
                "headroom_bytes": NODE_MEMORY_BYTES - peak,
                "limit_bytes": NODE_MEMORY_BYTES,
                "minimum_required_headroom_bytes": max(
                    32 * 1024**3, int(NODE_MEMORY_BYTES * 0.20)
                ),
                "passed": True,
                "peak_bytes": peak,
                "requested_bytes": NODE_MEMORY_BYTES,
            },
            "enable_thinking": False,
            "execution": execution,
            "execution_context": execution_context(world_size),
            "git_commit": self.commit,
            "gpu_preflight_completion_sha256": self.inputs[
                gpu_preflight_completion_content_name(world_size)
            ],
            "gpu_preflight_git_commit": self.commit,
            "gpu_preflight_matrix": {
                str(count): {
                    "allocation_sha256": str(count) * 64,
                    "completion_sha256": self.inputs[
                        gpu_preflight_completion_content_name(count)
                    ],
                    "git_commit": self.commit,
                    "report_sha256": self.inputs[
                        gpu_preflight_report_content_name(count)
                    ],
                    "workflow_id": (
                        "qwen3-v2-gpu-preflight-elastic-"
                        + format(count, "032x")
                    ),
                    "world_size": count,
                }
                for count in (1, 2, 3, 4)
            },
            "gpu_preflight_report_sha256": self.inputs[
                gpu_preflight_report_content_name(world_size)
            ],
            "inner_g0_report_sha256": hashlib.sha256(members["g0.json"]).hexdigest(),
            "model_revision": MODEL_REVISION,
            "passed": True,
            "phase": "qwen3_v2_g0",
            "prereg_commit": "c" * 40,
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_sha256": self.inputs["preregistration_sha256"],
            "prereg_version": "qwen3_v2",
            "protocol_amendment_git_commit": "6" * 40,
            "protocol_amendment_id": "qwen3_v2_g0_elastic_v1",
            "protocol_amendment_sha256": self.inputs[
                "protocol_amendment_sha256"
            ],
            "provenance_validation": PROVENANCE_VALIDATION,
            "prompt_protocol": "qwen3_non_thinking_v1",
            "protocol_track": "qwen3_v2",
            "resolved_config_sha256": self.inputs["resolved_config_sha256"],
            "request_git_commit": self.commit,
            "schema_version": 1,
            "started_at": "2026-09-04T04:00:00Z",
            "reviewed_implementation_commit": "7" * 40,
            "teacher_revision": TEACHER_REVISION,
            "tokenizer_fingerprint": (
                "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
            ),
            "tokenizer_revision": MODEL_REVISION,
            "world_size": world_size,
        }
        report["sha256"] = sha256_value(report)
        report_path = self.root / REPORT_NAME
        report_path.write_bytes(self._bytes(report))
        context = SimpleNamespace(
            expected_input_hashes=self.inputs,
            expected_output_paths={
                BUNDLE_NAME: bundle_path,
                REPORT_NAME: report_path,
            },
            expected_execution=SimpleNamespace(**execution),
            source_workspace=source_workspace,
        )
        completion = {
            "execution_config_sha256": self.inputs["execution_config_sha256"],
            "input_hashes": self.inputs,
            "resolved_config_sha256": self.inputs["resolved_config_sha256"],
            "scientific_config_sha256": self.inputs["scientific_config_sha256"],
            "scientific_validation": {name: True for name in GATE_NAMES},
        }
        return completion, context

    def _rebuild_bundle(
        self,
        context: SimpleNamespace,
        transform: object,
    ) -> None:
        bundle_path = context.expected_output_paths[BUNDLE_NAME]
        members: dict[str, tuple[bytes, tarfile.TarInfo]] = {}
        with tarfile.open(bundle_path, "r:") as archive:
            for member in archive:
                stream = archive.extractfile(member)
                members[member.name] = (
                    b"" if stream is None else stream.read(),
                    member,
                )
        transform(members)
        bundle_path.unlink()
        with tarfile.open(bundle_path, "w:") as archive:
            for name, (raw, source_info) in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.type = source_info.type
                info.linkname = source_info.linkname
                archive.addfile(info, fileobj=None if not info.isfile() else io.BytesIO(raw))
        report_path = context.expected_output_paths[REPORT_NAME]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["artifact_inventory"] = [
            {
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
            for name, (raw, info) in sorted(members.items())
        ]
        report["artifact_bundle_sha256"] = hashlib.sha256(
            bundle_path.read_bytes()
        ).hexdigest()
        if "g0.json" in members:
            report["inner_g0_report_sha256"] = hashlib.sha256(
                members["g0.json"][0]
            ).hexdigest()
        report["sha256"] = sha256_value(
            {key: value for key, value in report.items() if key != "sha256"}
        )
        report_path.write_bytes(self._bytes(report))

    def _validate_with_replay_stub(
        self,
        completion: dict[str, object],
        context: SimpleNamespace,
    ) -> None:
        replayed: list[Path] = []

        def replay(extracted: Path, inner: dict[str, object], report: object) -> None:
            replayed.append(extracted)
            self.assertNotEqual(extracted, context.source_workspace)
            self.assertEqual(extracted.name, "qwen3-v2")
            self.assertEqual(extracted.parent.parent, g0_contract.REPLAY_SCRATCH_ROOT)
            self.assertTrue(extracted.parent.name.startswith(".qwen3-v2-g0-replay-"))
            self.assertTrue((extracted / "g0.json").is_file())
            self.assertTrue(
                (
                    extracted
                    / "calibration/checkpoints/step-00000020.pt"
                ).is_file()
            )
            state = (
                extracted
                / "calibration/checkpoints/step-00000020.accelerate/state.bin"
            )
            if not state.is_file():
                raise AdapterValidationError("missing Accelerator state")
            self.assertEqual(tuple(sorted(inner["checks"])), G0_CHECK_NAMES)

        with mock.patch.object(
            g0_contract,
            "_replay_extracted_workspace",
            side_effect=replay,
        ):
            validate_qwen3_v2_g0_completion(completion, context)
        self.assertEqual(len(replayed), 1)
        self.assertFalse(replayed[0].parent.exists())

    def test_reconstructs_all_world_sizes_before_semantic_replay(self) -> None:
        self.assertIn(
            "circuits/final_answer/mib_raw/compatibility.json",
            REQUIRED_BUNDLE_PATHS,
        )
        self.assertIn(
            "circuits/first_rule_selection/mib_raw/compatibility.json",
            REQUIRED_BUNDLE_PATHS,
        )
        for world_size in (1, 2, 3, 4):
            with self.subTest(world_size=world_size):
                completion, context = self._write_valid_outputs(world_size)
                self.assertEqual(tuple(sorted(context.expected_output_paths)), OUTPUT_NAMES)
                self._validate_with_replay_stub(completion, context)
                contract = batch_token_contract(world_size)
                self.assertEqual(
                    contract["requested_fsdp_sharding_strategy"], "FULL_SHARD"
                )
                self.assertEqual(
                    contract["effective_fsdp_sharding_strategy"],
                    "NO_SHARD" if world_size == 1 else "FULL_SHARD",
                )

    def test_dummy_artifacts_cannot_pass_without_semantic_replay(self) -> None:
        completion, context = self._write_valid_outputs(world_size=1)
        with self.assertRaisesRegex(AdapterValidationError, "artifact replay"):
            validate_qwen3_v2_g0_completion(completion, context)

    def test_replay_rejects_missing_checkpoint_and_accelerator_state(self) -> None:
        completion, context = self._write_valid_outputs(world_size=2)
        self._rebuild_bundle(
            context,
            lambda members: members.pop(
                "calibration/checkpoints/step-00000020.pt"
            ),
        )
        with self.assertRaisesRegex(AdapterValidationError, "lacks required evidence"):
            validate_qwen3_v2_g0_completion(completion, context)

        completion, context = self._write_valid_outputs(world_size=2)
        self._rebuild_bundle(
            context,
            lambda members: members.pop(
                "calibration/checkpoints/step-00000020.accelerate/state.bin"
            ),
        )
        with self.assertRaisesRegex(AdapterValidationError, "missing Accelerator state"):
            self._validate_with_replay_stub(completion, context)

    def test_replay_requires_both_stage_compatibility_artifacts(self) -> None:
        completion, context = self._write_valid_outputs(world_size=2)
        self._rebuild_bundle(
            context,
            lambda members: members.pop(
                "circuits/first_rule_selection/mib_raw/compatibility.json"
            ),
        )
        with self.assertRaisesRegex(AdapterValidationError, "lacks required evidence"):
            validate_qwen3_v2_g0_completion(completion, context)

    def test_replay_rejects_link_and_escape_members(self) -> None:
        completion, context = self._write_valid_outputs(world_size=3)

        def link(members: dict[str, tuple[bytes, tarfile.TarInfo]]) -> None:
            _raw, info = members["anti_shortcut.json"]
            info.type = tarfile.SYMTYPE
            info.linkname = "g0.json"
            members["anti_shortcut.json"] = (b"", info)

        self._rebuild_bundle(context, link)
        with self.assertRaisesRegex(AdapterValidationError, "regular files"):
            validate_qwen3_v2_g0_completion(completion, context)

        completion, context = self._write_valid_outputs(world_size=3)

        def escape(members: dict[str, tuple[bytes, tarfile.TarInfo]]) -> None:
            members["../escape"] = members.pop("anti_shortcut.json")

        self._rebuild_bundle(context, escape)
        with self.assertRaisesRegex(AdapterValidationError, "not confined"):
            validate_qwen3_v2_g0_completion(completion, context)

    def test_inner_report_must_contain_the_exact_check_schema(self) -> None:
        completion, context = self._write_valid_outputs(world_size=4)

        def omit_check(members: dict[str, tuple[bytes, tarfile.TarInfo]]) -> None:
            raw, info = members["g0.json"]
            payload = json.loads(raw)
            payload["checks"].pop("teacher_correctness")
            payload["sha256"] = sha256_value(
                {key: value for key, value in payload.items() if key != "sha256"}
            )
            members["g0.json"] = (self._bytes(payload), info)

        self._rebuild_bundle(context, omit_check)
        with self.assertRaisesRegex(AdapterValidationError, "registered gate"):
            validate_qwen3_v2_g0_completion(completion, context)

    def test_rejects_wrong_effective_fsdp_strategy_for_running_world(self) -> None:
        completion, context = self._write_valid_outputs(world_size=1)
        report_path = context.expected_output_paths[REPORT_NAME]
        report = json.loads(report_path.read_text())
        report["batch_token_contract"][
            "effective_fsdp_sharding_strategy"
        ] = "FULL_SHARD"
        report["sha256"] = sha256_value(
            {key: value for key, value in report.items() if key != "sha256"}
        )
        report_path.write_bytes(self._bytes(report))
        with self.assertRaisesRegex(AdapterValidationError, "protocol"):
            validate_qwen3_v2_g0_completion(completion, context)

    def test_rejects_preflight_that_predates_amendment_acceptance(self) -> None:
        completion, context = self._write_valid_outputs()
        report_path = context.expected_output_paths[REPORT_NAME]
        report = json.loads(report_path.read_text())
        report["gpu_preflight_git_commit"] = report[
            "reviewed_implementation_commit"
        ]
        report["gpu_preflight_matrix"][str(report["world_size"])][
            "git_commit"
        ] = report["reviewed_implementation_commit"]
        report["sha256"] = sha256_value(
            {key: value for key, value in report.items() if key != "sha256"}
        )
        report_path.write_bytes(self._bytes(report))
        with self.assertRaisesRegex(AdapterValidationError, "predates"):
            validate_qwen3_v2_g0_completion(completion, context)

    def test_rejects_report_from_another_running_allocation(self) -> None:
        completion, context = self._write_valid_outputs(world_size=3)
        context.expected_execution.allocation_sha256 = "0" * 64
        with self.assertRaisesRegex(AdapterValidationError, "running attempt"):
            validate_qwen3_v2_g0_completion(completion, context)

    def test_rejects_reused_preflight_matrix_evidence(self) -> None:
        completion, context = self._write_valid_outputs(world_size=2)
        report_path = context.expected_output_paths[REPORT_NAME]
        report = json.loads(report_path.read_text())
        report["gpu_preflight_matrix"]["4"]["workflow_id"] = report[
            "gpu_preflight_matrix"
        ]["3"]["workflow_id"]
        report["sha256"] = sha256_value(
            {key: value for key, value in report.items() if key != "sha256"}
        )
        report_path.write_bytes(self._bytes(report))
        with self.assertRaisesRegex(AdapterValidationError, "reuses workflow"):
            validate_qwen3_v2_g0_completion(completion, context)


if __name__ == "__main__":
    unittest.main()
