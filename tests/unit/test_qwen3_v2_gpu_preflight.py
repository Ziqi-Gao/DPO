from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
    CHAT_TEMPLATE_SHA256,
    GATE_NAMES,
    GPU_MODEL,
    MODEL_REVISION,
    NCCL_PROBE_TIMEOUT_SECONDS,
    NCCL_VERSION,
    OUTPUT_NAME,
    TEACHER_REVISION,
    TOKENIZER_FINGERPRINT,
    validate_qwen3_v2_gpu_preflight_completion,
)
from posttrain_circuits.scheduler_adapter.secure_files import published_json_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _digest(label: str) -> str:
    return sha256_value({"fixture": label})


def _inputs() -> dict[str, str]:
    return {
        "config_binding_sha256": _digest("binding"),
        "execution_config_sha256": _digest("execution"),
        "preregistration_sha256": _digest("preregistration"),
        "resolved_config_sha256": _digest("resolved"),
        "scientific_config_sha256": _digest("scientific"),
    }


def _rank_row(rank: int) -> dict[str, object]:
    return {
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "checkpoint_sha256": _digest(f"checkpoint-{rank}"),
        "enable_thinking": False,
        "fsdp_save_resume": True,
        "gradients_finite": True,
        "loading_strategy": "low_cpu_mem_student_rank_zero_teacher",
        "max_memory_allocated": 1024 + rank,
        "max_memory_reserved": 2048 + rank,
        "model_facing_prompt_sha256": _digest(f"model-prompt-{rank}"),
        "parameter_update_nonzero": True,
        "process_max_rss_bytes": 4096 + rank,
        "prompt_protocol": "qwen3_non_thinking_v1",
        "rank": rank,
        "raw_prompt_sha256": _digest(f"raw-prompt-{rank}"),
        "soft_teacher_loss": 0.25 + rank,
        "soft_teacher_loss_finite": True,
        "student_forward_finite": True,
        "student_revision": MODEL_REVISION,
        "teacher_forward_finite": True,
        "teacher_loaded_on_this_rank": rank == 0,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "tokenizer_revision": MODEL_REVISION,
        "unique_rank_prompt_shard": True,
    }


def _report(inputs: dict[str, str]) -> dict[str, object]:
    rows = [_rank_row(rank) for rank in range(4)]
    limit = 192 * 1024**3
    peak = 100 * 1024**3
    minimum = max(32 * 1024**3, int(limit * 0.20))
    content: dict[str, object] = {
        "artifact_namespace": "qwen3-v2",
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "code_commit": "a" * 40,
        "cgroup_memory": {
            "current_bytes": 90 * 1024**3,
            "headroom_bytes": limit - peak,
            "limit_bytes": limit,
            "minimum_required_headroom_bytes": minimum,
            "observed_peak_bytes": peak,
            "passed": True,
            "peak_bytes": peak,
            "requested_bytes": limit,
        },
        "created_at": "2026-09-03T00:00:00Z",
        "devices": [
            {
                "capability": [12, 0],
                "logical_index": rank,
                "name": GPU_MODEL,
                "pci_bus_id": f"0000:{0x48 + rank:02x}:00.0",
                "rank": rank,
                "total_memory": 97_887 * 1024**2,
                "visible_cuda_identifier": f"GPU-fixture-{rank}",
            }
            for rank in range(4)
        ],
        "enable_thinking": False,
        "execution_context": {
            "allocation_visibility": "preserved",
            "distributed_launcher": "environment_rank_passthrough",
            "mode": "server_scheduler_foreground",
            "nccl_p2p_policy": "disabled",
            "visible_device_count": 4,
        },
        "git_commit": "a" * 40,
        "model_revision": MODEL_REVISION,
        "nccl_all_reduce": True,
        "nccl_diagnostics": [
            {
                "control_backend": "gloo",
                "data_backend": "nccl",
                "elapsed_seconds": 0.25 + rank,
                "observed_sum": 10.0,
                "p2p_disabled": True,
                "rank": rank,
                "timeout_seconds": NCCL_PROBE_TIMEOUT_SECONDS,
            }
            for rank in range(4)
        ],
        "nccl_runtime_version": NCCL_VERSION,
        "passed": True,
        "phase": "gpu_preflight",
        "prereg_commit": "b" * 40,
        "prereg_path": "prereg/qwen3_v2.yaml",
        "prereg_sha256": inputs["preregistration_sha256"],
        "prereg_version": "qwen3_v2",
        "prompt_protocol": "qwen3_non_thinking_v1",
        "protocol_track": "qwen3_v2",
        "qwen_forward_finite": True,
        "rank_prompt_hashes_unique": True,
        "rank_training_checks": rows,
        "rank_zero_teacher_load_count": 1,
        "resolved_config_sha256": inputs["resolved_config_sha256"],
        "resolved_model_commit": MODEL_REVISION,
        "resolved_teacher_commit": TEACHER_REVISION,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "tokenizer_hash": TOKENIZER_FINGERPRINT,
        "tokenizer_revision": MODEL_REVISION,
        "torch_cuda_version": "12.8",
        "torch_version": "2.8.0+cu128",
        "visible_cuda_devices": 4,
        "world_size": 4,
    }
    return {**content, "sha256": sha256_value(content)}


def _completion(inputs: dict[str, str]) -> dict[str, object]:
    return {
        "execution_config_sha256": inputs["execution_config_sha256"],
        "input_hashes": inputs,
        "resolved_config_sha256": inputs["resolved_config_sha256"],
        "scientific_config_sha256": inputs["scientific_config_sha256"],
        "scientific_validation": {name: True for name in GATE_NAMES},
    }


def _context(tmp_path: Path, inputs: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        expected_input_hashes=inputs,
        expected_output_paths={OUTPUT_NAME: tmp_path / OUTPUT_NAME},
    )


class Qwen3V2GpuPreflightValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".gpu-preflight-validator-test-", dir=PROJECT_ROOT
        )
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_accepts_staging_and_strict_published_report(self) -> None:
        inputs = _inputs()
        completion = _completion(inputs)
        context = _context(self.root, inputs)
        validate_qwen3_v2_gpu_preflight_completion(completion, context)
        context.expected_output_paths[OUTPUT_NAME].write_bytes(
            published_json_bytes(_report(inputs))
        )
        validate_qwen3_v2_gpu_preflight_completion(completion, context)

    def test_rejects_rehashed_semantic_tampering(self) -> None:
        mutations = (
            (lambda report: report.update({"model_revision": "0" * 40}), "fixed Qwen3-v2"),
            (
                lambda report: report["devices"][0].update({"name": "unreviewed"}),
                "unreviewed GPU model",
            ),
            (
                lambda report: report["rank_training_checks"][2].update(
                    {"fsdp_save_resume": False}
                ),
                "fsdp_save_resume",
            ),
            (
                lambda report: report["rank_training_checks"][3].update(
                    {
                        "model_facing_prompt_sha256": report["rank_training_checks"][0][
                            "model_facing_prompt_sha256"
                        ]
                    }
                ),
                "prompt shards",
            ),
            (
                lambda report: report["cgroup_memory"].update({"headroom_bytes": 1}),
                "internally inconsistent",
            ),
            (
                lambda report: report["nccl_diagnostics"][1].update(
                    {"p2p_disabled": False}
                ),
                "NCCL diagnostic",
            ),
            (
                lambda report: report.update({"torch_cuda_version": "unreviewed"}),
                "CUDA-enabled Torch",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                inputs = _inputs()
                report = copy.deepcopy(_report(inputs))
                mutation(report)
                unsigned = {key: value for key, value in report.items() if key != "sha256"}
                report["sha256"] = sha256_value(unsigned)
                path = self.root / OUTPUT_NAME
                path.write_bytes(published_json_bytes(report))
                with self.assertRaisesRegex(AdapterValidationError, message):
                    validate_qwen3_v2_gpu_preflight_completion(
                        _completion(inputs), _context(self.root, inputs)
                    )
                path.unlink()

    def test_rejects_bad_self_summary_and_gate_set(self) -> None:
        inputs = _inputs()
        report = _report(inputs)
        report["sha256"] = "0" * 64
        (self.root / OUTPUT_NAME).write_bytes(published_json_bytes(report))
        with self.assertRaisesRegex(AdapterValidationError, "self-summary"):
            validate_qwen3_v2_gpu_preflight_completion(
                _completion(inputs), _context(self.root, inputs)
            )

        (self.root / OUTPUT_NAME).unlink()
        completion = _completion(inputs)
        completion["scientific_validation"] = {"config_binding": True}
        with self.assertRaisesRegex(AdapterValidationError, "gate names"):
            validate_qwen3_v2_gpu_preflight_completion(
                completion, _context(self.root, inputs)
            )

    def test_rejects_service_level_193_gib_cgroup(self) -> None:
        inputs = _inputs()
        report = _report(inputs)
        memory = report["cgroup_memory"]
        self.assertIsInstance(memory, dict)
        limit = 193 * 1024**3
        observed = int(memory["observed_peak_bytes"])
        memory.update(
            {
                "headroom_bytes": limit - observed,
                "limit_bytes": limit,
                "minimum_required_headroom_bytes": max(
                    32 * 1024**3, int(limit * 0.20)
                ),
            }
        )
        unsigned = {key: value for key, value in report.items() if key != "sha256"}
        report["sha256"] = sha256_value(unsigned)
        (self.root / OUTPUT_NAME).write_bytes(published_json_bytes(report))
        with self.assertRaisesRegex(AdapterValidationError, "internally inconsistent"):
            validate_qwen3_v2_gpu_preflight_completion(
                _completion(inputs), _context(self.root, inputs)
            )


if __name__ == "__main__":
    unittest.main()
