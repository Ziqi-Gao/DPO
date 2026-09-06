from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
    BATCH_PARTITION_PROTOCOL,
    CANONICAL_SFT_OBJECTIVE,
    CHAT_TEMPLATE_SHA256,
    CPU_CORE_COUNT,
    GATE_NAMES,
    GPU_MODEL,
    GPU_MEMORY_BUDGET_MIB,
    GLOBAL_LOGICAL_BATCH_SIZE,
    MAX_MODEL_INPUT_LENGTH,
    MAX_PER_RANK_MICROBATCH_SIZE,
    MODEL_REVISION,
    NCCL_PROBE_TIMEOUT_SECONDS,
    NCCL_VERSION,
    OUTPUT_NAME,
    PROFILE_NAME,
    SUPPORTED_GPU_COUNTS,
    TEACHER_REVISION,
    TOKENIZER_FINGERPRINT,
    WORKFLOW_ID_PREFIX,
    microbatch_sizes,
    nccl_expected_sum,
    training_contract,
    validate_qwen3_v2_gpu_preflight_completion,
)
from posttrain_circuits.scheduler_adapter.secure_files import published_json_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ID = f"{WORKFLOW_ID_PREFIX}{'a' * 32}"


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


def _rank_row(rank: int, gpu_count: int) -> dict[str, object]:
    schedule = list(microbatch_sizes(gpu_count, rank))
    slots = list(range(rank, GLOBAL_LOGICAL_BATCH_SIZE, gpu_count))
    return {
        "batch_partition_protocol": BATCH_PARTITION_PROTOCOL,
        "canonical_sft_loss": 0.25,
        "canonical_sft_loss_finite": True,
        "canonical_sft_objective": CANONICAL_SFT_OBJECTIVE,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "checkpoint_sha256": _digest(f"checkpoint-{rank}"),
        "checkpoint_manifest_sha256": _digest("checkpoint-manifest"),
        "checkpoint_runtime_sha256": _digest(f"checkpoint-runtime-{rank}"),
        "effective_fsdp_sharding_strategy": (
            "NO_SHARD" if gpu_count == 1 else "FULL_SHARD"
        ),
        "enable_thinking": False,
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_save_resume": True,
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_transformer_layer": "Qwen3DecoderLayer",
        "fsdp_wrapper_count": 29,
        "gradients_finite": True,
        "global_logical_batch_size": GLOBAL_LOGICAL_BATCH_SIZE,
        "global_model_input_tokens": (
            GLOBAL_LOGICAL_BATCH_SIZE * MAX_MODEL_INPUT_LENGTH
        ),
        "global_slots": slots,
        "loading_strategy": "low_cpu_mem_student_rank_zero_teacher",
        "local_sequence_count": len(slots),
        "loss_scaling_protocol": (
            "exact_global_sequence_mean_after_fsdp_rank_averaging_v1"
        ),
        "max_memory_allocated": 1024 + rank,
        "max_memory_reserved": 2048 + rank,
        "max_model_input_length": MAX_MODEL_INPUT_LENGTH,
        "max_per_rank_microbatch_size": MAX_PER_RANK_MICROBATCH_SIZE,
        "microbatch_sizes": schedule,
        "microsteps_per_optimizer_update": len(schedule),
        "model_facing_prompt_sha256": _digest(f"model-prompt-{rank}"),
        "observed_max_model_input_length": MAX_MODEL_INPUT_LENGTH,
        "optimizer_class": "torch.optim.AdamW",
        "optimizer_state_restored": True,
        "optimizer_updates": 1,
        "parameter_update_nonzero": True,
        "process_max_rss_bytes": 4096 + rank,
        "prompt_protocol": "qwen3_non_thinking_v1",
        "rank": rank,
        "raw_prompt_sha256": _digest(f"raw-prompt-{rank}"),
        "requested_fsdp_sharding_strategy": "FULL_SHARD",
        "response_mask_tokens": len(slots) * 256,
        "response_tokens_per_sequence": 256,
        "rng_state_restored": True,
        "scheduler_state_restored": True,
        "sequence_normalization": True,
        "student_forward_finite": True,
        "student_revision": MODEL_REVISION,
        "teacher_forward_finite": True,
        "teacher_loaded_on_this_rank": rank == 0,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "tokenizer_revision": MODEL_REVISION,
        "unique_rank_prompt_shard": True,
        "wrapped_transformer_blocks": 28,
    }


def _execution() -> dict[str, object]:
    return {
        "allocation_sha256": "1" * 64,
        "attempt": 1,
        "execution_profile": PROFILE_NAME,
        "job_id": "opd-fixture",
        "manifest_sha256": "2" * 64,
    }


def _report(inputs: dict[str, str], *, gpu_count: int = 2) -> dict[str, object]:
    rows = [_rank_row(rank, gpu_count) for rank in range(gpu_count)]
    checkpoint_sha256 = _digest(f"checkpoint-w{gpu_count}")
    for row in rows:
        row["checkpoint_sha256"] = checkpoint_sha256
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
            for rank in range(gpu_count)
        ],
        "enable_thinking": False,
        "execution": _execution(),
        "execution_context": {
            "allocation_contract": "manifest_driven_scheduler_gpu_v1",
            "allocation_visibility": "preserved",
            "cpu_core_count": CPU_CORE_COUNT,
            "distributed_launcher": "environment_rank_passthrough",
            "gpu_count_policy": "scheduler",
            "mode": "server_scheduler_foreground",
            "nccl_p2p_policy": "disabled",
            "threads_per_rank": CPU_CORE_COUNT // gpu_count,
            "visible_device_count": gpu_count,
        },
        "git_commit": "a" * 40,
        "model_revision": MODEL_REVISION,
        "nccl_all_reduce": True,
        "nccl_diagnostics": [
            {
                "control_backend": "gloo",
                "data_backend": "nccl",
                "elapsed_seconds": 0.25 + rank,
                "observed_sum": nccl_expected_sum(gpu_count),
                "p2p_disabled": True,
                "rank": rank,
                "timeout_seconds": NCCL_PROBE_TIMEOUT_SECONDS,
            }
            for rank in range(gpu_count)
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
        "training_contract": training_contract(),
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "tokenizer_hash": TOKENIZER_FINGERPRINT,
        "tokenizer_revision": MODEL_REVISION,
        "torch_cuda_version": "12.8",
        "torch_version": "2.8.0+cu128",
        "visible_cuda_devices": gpu_count,
        "world_size": gpu_count,
    }
    return {**content, "sha256": sha256_value(content)}


def _completion(inputs: dict[str, str]) -> dict[str, object]:
    return {
        "execution_config_sha256": inputs["execution_config_sha256"],
        "execution": _execution(),
        "input_hashes": inputs,
        "resolved_config_sha256": inputs["resolved_config_sha256"],
        "scientific_config_sha256": inputs["scientific_config_sha256"],
        "scientific_validation": {name: True for name in GATE_NAMES},
        "workflow_id": WORKFLOW_ID,
    }


def _context(tmp_path: Path, inputs: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        expected_input_hashes=inputs,
        expected_output_paths={OUTPUT_NAME: tmp_path / OUTPUT_NAME},
        workflow_id=WORKFLOW_ID,
    )


class Qwen3V2GpuPreflightValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".gpu-preflight-validator-test-", dir=PROJECT_ROOT
        )
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_accepts_staging_and_strict_published_report_for_all_counts(self) -> None:
        for gpu_count in SUPPORTED_GPU_COUNTS:
            with self.subTest(gpu_count=gpu_count):
                inputs = _inputs()
                completion = _completion(inputs)
                context = _context(self.root, inputs)
                validate_qwen3_v2_gpu_preflight_completion(completion, context)
                context.expected_output_paths[OUTPUT_NAME].write_bytes(
                    published_json_bytes(_report(inputs, gpu_count=gpu_count))
                )
                validate_qwen3_v2_gpu_preflight_completion(completion, context)
                context.expected_output_paths[OUTPUT_NAME].unlink()

    def test_rejects_wrong_effective_fsdp_strategy_for_every_count(self) -> None:
        for gpu_count in SUPPORTED_GPU_COUNTS:
            with self.subTest(gpu_count=gpu_count):
                inputs = _inputs()
                report = _report(inputs, gpu_count=gpu_count)
                report["rank_training_checks"][0][
                    "effective_fsdp_sharding_strategy"
                ] = "FULL_SHARD" if gpu_count == 1 else "NO_SHARD"
                report["sha256"] = sha256_value(
                    {key: value for key, value in report.items() if key != "sha256"}
                )
                path = self.root / OUTPUT_NAME
                path.write_bytes(published_json_bytes(report))
                with self.assertRaisesRegex(
                    AdapterValidationError, "fixed Qwen3-v2 protocol"
                ):
                    validate_qwen3_v2_gpu_preflight_completion(
                        _completion(inputs), _context(self.root, inputs)
                    )
                path.unlink()

    def test_rejects_rehashed_semantic_tampering(self) -> None:
        mutations = (
            (lambda report: report.update({"model_revision": "0" * 40}), "fixed Qwen3-v2"),
            (
                lambda report: report["devices"][0].update({"name": "unreviewed"}),
                "unreviewed GPU model",
            ),
            (
                lambda report: report["rank_training_checks"][1].update(
                    {"fsdp_save_resume": False}
                ),
                "fsdp_save_resume",
            ),
            (
                lambda report: report["training_contract"].update(
                    {"max_model_input_length": 1024}
                ),
                "top-level execution gates",
            ),
            (
                lambda report: report["training_contract"].update(
                    {"weight_decay": False}
                ),
                "top-level execution gates",
            ),
            (
                lambda report: report["rank_training_checks"][0].update(
                    {"microbatch_sizes": [1]}
                ),
                "fixed Qwen3-v2 protocol",
            ),
            (
                lambda report: report["rank_training_checks"][0].update(
                    {"observed_max_model_input_length": 1024}
                ),
                "fixed Qwen3-v2 protocol",
            ),
            (
                lambda report: report["rank_training_checks"][0].update(
                    {"fsdp_state_dict_type": "SHARDED_STATE_DICT"}
                ),
                "fixed Qwen3-v2 protocol",
            ),
            (
                lambda report: report["rank_training_checks"][0].update(
                    {"requested_fsdp_sharding_strategy": "NO_SHARD"}
                ),
                "fixed Qwen3-v2 protocol",
            ),
            (
                lambda report: report["rank_training_checks"][0].update(
                    {"fsdp_wrapper_count": 28}
                ),
                "wrapper count",
            ),
            (
                lambda report: report["rank_training_checks"][0].update(
                    {"optimizer_updates": True}
                ),
                "optimizer_updates is not an integer",
            ),
            (
                lambda report: report["rank_training_checks"][0].update(
                    {"max_memory_reserved": GPU_MEMORY_BUDGET_MIB * 1024**2 + 1}
                ),
                "registered memory envelope",
            ),
            (
                lambda report: report["rank_training_checks"][1].update(
                    {"checkpoint_sha256": _digest("different-checkpoint")}
                ),
                "full checkpoint disagree",
            ),
            (
                lambda report: report["rank_training_checks"][1].update(
                    {
                        "model_facing_prompt_sha256": report["rank_training_checks"][0][
                            "model_facing_prompt_sha256"
                        ]
                    }
                ),
                "prompts",
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
                lambda report: report["execution"].update(
                    {"allocation_sha256": "9" * 64}
                ),
                "completion allocation",
            ),
            (
                lambda report: report.update({"world_size": 5}),
                "outside the reviewed 1..4 topology",
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

    def test_rejects_nonopaque_or_mismatched_workflow_identity(self) -> None:
        inputs = _inputs()
        for workflow_id in (
            "qwen3-v2-gpu-preflight-elastic-2gpu",
            f"{WORKFLOW_ID_PREFIX}{'A' * 32}",
        ):
            completion = _completion(inputs)
            completion["workflow_id"] = workflow_id
            with self.subTest(workflow_id=workflow_id), self.assertRaisesRegex(
                AdapterValidationError, "opaque instance"
            ):
                validate_qwen3_v2_gpu_preflight_completion(
                    completion, _context(self.root, inputs)
                )

        context = _context(self.root, inputs)
        context.workflow_id = f"{WORKFLOW_ID_PREFIX}{'b' * 32}"
        with self.assertRaisesRegex(AdapterValidationError, "workflow plan"):
            validate_qwen3_v2_gpu_preflight_completion(_completion(inputs), context)

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
