from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
    BATCH_TOKEN_CONTRACT,
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
    validate_qwen3_v2_g0_completion,
)


class Qwen3V2G0CompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix=".g0-validator-", dir=Path.cwd())
        self.root = Path(self.temporary.name)
        self.inputs = {
            "config_binding_sha256": "1" * 64,
            "execution_config_sha256": "2" * 64,
            "gpu_preflight_completion_sha256": "3" * 64,
            "gpu_preflight_report_sha256": "4" * 64,
            "preregistration_sha256": "5" * 64,
            "protocol_amendment_sha256": "8" * 64,
            "resolved_config_sha256": "6" * 64,
            "scientific_config_sha256": "7" * 64,
        }
        self.commit = "8" * 40

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _bytes(payload: object) -> bytes:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()

    def _write_valid_outputs(self) -> tuple[dict[str, object], SimpleNamespace]:
        inner: dict[str, object] = {
            "checks": {
                "distributed_checkpoint_resume": True,
                "gpu_preflight": True,
            },
            "code_commit": self.commit,
            "git_commit": self.commit,
            "model_revision": MODEL_REVISION,
            "passed": True,
            "phase": "G0",
            "prereg_sha256": self.inputs["preregistration_sha256"],
            "protocol_amendment_id": "qwen3_v2_g0_2gpu_v1",
            "protocol_amendment_sha256": self.inputs[
                "protocol_amendment_sha256"
            ],
            "reviewed_implementation_commit": "7" * 40,
            "request_git_commit": self.commit,
            "batch_token_contract": BATCH_TOKEN_CONTRACT,
            "teacher_revision": TEACHER_REVISION,
            "tokenizer_revision": MODEL_REVISION,
        }
        inner["sha256"] = sha256_value(inner)
        members = {name: b"{}\n" for name in REQUIRED_BUNDLE_PATHS}
        members["g0.json"] = self._bytes(inner)
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
            "allocation_sha256": "9" * 64,
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
            "batch_token_contract": BATCH_TOKEN_CONTRACT,
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
            "execution_context": {
                "allocation_visibility": "preserved",
                "distributed_launcher": "accelerate_fsdp_foreground",
                "mode": "server_scheduler_foreground",
                "nccl_p2p_policy": "disabled",
                "visible_device_count": 2,
            },
            "git_commit": self.commit,
            "gpu_preflight_completion_sha256": self.inputs[
                "gpu_preflight_completion_sha256"
            ],
            "gpu_preflight_git_commit": self.commit,
            "gpu_preflight_report_sha256": self.inputs["gpu_preflight_report_sha256"],
            "inner_g0_report_sha256": hashlib.sha256(members["g0.json"]).hexdigest(),
            "model_revision": MODEL_REVISION,
            "passed": True,
            "phase": "qwen3_v2_g0",
            "prereg_commit": "c" * 40,
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_sha256": self.inputs["preregistration_sha256"],
            "prereg_version": "qwen3_v2",
            "protocol_amendment_git_commit": "6" * 40,
            "protocol_amendment_id": "qwen3_v2_g0_2gpu_v1",
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
            "world_size": 2,
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
        )
        completion = {
            "execution_config_sha256": self.inputs["execution_config_sha256"],
            "input_hashes": self.inputs,
            "resolved_config_sha256": self.inputs["resolved_config_sha256"],
            "scientific_config_sha256": self.inputs["scientific_config_sha256"],
            "scientific_validation": {name: True for name in GATE_NAMES},
        }
        return completion, context

    def test_accepts_complete_hash_bound_artifact_bundle(self) -> None:
        completion, context = self._write_valid_outputs()
        self.assertEqual(tuple(sorted(context.expected_output_paths)), OUTPUT_NAMES)
        validate_qwen3_v2_g0_completion(completion, context)

    def test_rejects_preflight_that_predates_amendment_acceptance(self) -> None:
        completion, context = self._write_valid_outputs()
        report_path = context.expected_output_paths[REPORT_NAME]
        report = json.loads(report_path.read_text())
        report["gpu_preflight_git_commit"] = report[
            "reviewed_implementation_commit"
        ]
        report["sha256"] = sha256_value(
            {key: value for key, value in report.items() if key != "sha256"}
        )
        report_path.write_bytes(self._bytes(report))
        with self.assertRaisesRegex(AdapterValidationError, "predates"):
            validate_qwen3_v2_g0_completion(completion, context)


if __name__ == "__main__":
    unittest.main()
