"""Scientific completion contract for the elastic Qwen3-v2 GPU preflight."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.learning.training.fsdp_contract import (
    EFFECTIVE_FSDP_SHARDING_STRATEGY_BY_WORLD_SIZE,
    REQUESTED_FSDP_SHARDING_STRATEGY,
    effective_fsdp_sharding_strategy,
)
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json


TASK_NAME = "qwen3_v2_gpu_preflight"
PROFILE_NAME = "qwen3-v2-gpu-preflight-elastic"
OUTPUT_NAME = "gpu_preflight.json"
WORKFLOW_ID_PREFIX = "qwen3-v2-gpu-preflight-elastic-"
UNIT_ID = "gpu-preflight"

SUPPORTED_GPU_COUNTS = (1, 2, 3, 4)
CPU_CORE_COUNT = 24
GLOBAL_LOGICAL_BATCH_SIZE = 64
MAX_PER_RANK_MICROBATCH_SIZE = 4
MAX_MODEL_INPUT_LENGTH = 1536
GPU_MEMORY_BUDGET_MIB = 81920
FROZEN_PROMPT_POPULATION_MAX_TOKENS = 1246
TEACHER_DEMO_MAX_NEW_TOKENS = 256
CANONICAL_SFT_OBJECTIVE = "response_mask_sequence_mean_cross_entropy_v1"
BATCH_PARTITION_PROTOCOL = "allocation_neutral_exact_global_batch_v1"
LOSS_SCALING_PROTOCOL = "exact_global_sequence_mean_after_fsdp_rank_averaging_v1"
FSDP_STATE_DICT_TYPE = "FULL_STATE_DICT"
FSDP_AUTO_WRAP_POLICY = "TRANSFORMER_BASED_WRAP"
FSDP_TRANSFORMER_LAYER = "Qwen3DecoderLayer"
OPTIMIZER_CLASS = "torch.optim.AdamW"
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
GPU_MODEL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
GPU_MINIMUM_TOTAL_MEMORY_MIB = 97_000
GPU_COMPUTE_CAPABILITY = (12, 0)
NCCL_VERSION = "2.27.3"
NCCL_PROBE_TIMEOUT_SECONDS = 120
NODE_MEMORY_GIB = 192
MINIMUM_HEADROOM_GIB = 32
MINIMUM_HEADROOM_FRACTION = 0.20

MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEACHER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_REVISION = MODEL_REVISION
TOKENIZER_FINGERPRINT = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
PROTOCOL_TRACK = "qwen3_v2"
ARTIFACT_NAMESPACE = "qwen3-v2"
PROMPT_PROTOCOL = "qwen3_non_thinking_v1"
PREREG_PATH = "prereg/qwen3_v2.yaml"
PREREG_VERSION = "qwen3_v2"

GATE_NAMES = (
    "allocation_contract",
    "config_binding",
    "cuda_runtime",
    "fsdp_resume",
    "memory_headroom",
    "nccl_all_reduce",
    "offline_pinned_models",
    "real_forward_backward",
)

REPORT_FIELDS = frozenset(
    {
        "artifact_namespace",
        "chat_template_sha256",
        "code_commit",
        "cgroup_memory",
        "created_at",
        "devices",
        "enable_thinking",
        "execution",
        "execution_context",
        "git_commit",
        "model_revision",
        "nccl_all_reduce",
        "nccl_diagnostics",
        "nccl_runtime_version",
        "passed",
        "phase",
        "prereg_commit",
        "prereg_path",
        "prereg_sha256",
        "prereg_version",
        "prompt_protocol",
        "protocol_track",
        "qwen_forward_finite",
        "rank_prompt_hashes_unique",
        "rank_training_checks",
        "rank_zero_teacher_load_count",
        "resolved_config_sha256",
        "resolved_model_commit",
        "resolved_teacher_commit",
        "sha256",
        "teacher_revision",
        "training_contract",
        "tokenizer_fingerprint",
        "tokenizer_hash",
        "tokenizer_revision",
        "torch_cuda_version",
        "torch_version",
        "visible_cuda_devices",
        "world_size",
    }
)
DEVICE_FIELDS = frozenset(
    {
        "capability",
        "logical_index",
        "name",
        "pci_bus_id",
        "rank",
        "total_memory",
        "visible_cuda_identifier",
    }
)
NCCL_DIAGNOSTIC_FIELDS = frozenset(
    {
        "control_backend",
        "data_backend",
        "elapsed_seconds",
        "observed_sum",
        "p2p_disabled",
        "rank",
        "timeout_seconds",
    }
)
RANK_CHECK_FIELDS = frozenset(
    {
        "chat_template_sha256",
        "batch_partition_protocol",
        "canonical_sft_loss",
        "canonical_sft_loss_finite",
        "canonical_sft_objective",
        "checkpoint_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_runtime_sha256",
        "enable_thinking",
        "effective_fsdp_sharding_strategy",
        "fsdp_auto_wrap_policy",
        "fsdp_save_resume",
        "fsdp_state_dict_type",
        "fsdp_transformer_layer",
        "fsdp_wrapper_count",
        "gradients_finite",
        "global_logical_batch_size",
        "global_model_input_tokens",
        "global_slots",
        "loading_strategy",
        "local_sequence_count",
        "loss_scaling_protocol",
        "max_memory_allocated",
        "max_memory_reserved",
        "max_model_input_length",
        "max_per_rank_microbatch_size",
        "microbatch_sizes",
        "microsteps_per_optimizer_update",
        "model_facing_prompt_sha256",
        "observed_max_model_input_length",
        "optimizer_class",
        "optimizer_state_restored",
        "optimizer_updates",
        "parameter_update_nonzero",
        "process_max_rss_bytes",
        "prompt_protocol",
        "rank",
        "raw_prompt_sha256",
        "requested_fsdp_sharding_strategy",
        "response_mask_tokens",
        "response_tokens_per_sequence",
        "rng_state_restored",
        "scheduler_state_restored",
        "sequence_normalization",
        "student_forward_finite",
        "student_revision",
        "teacher_forward_finite",
        "teacher_loaded_on_this_rank",
        "teacher_revision",
        "tokenizer_fingerprint",
        "tokenizer_revision",
        "unique_rank_prompt_shard",
        "wrapped_transformer_blocks",
    }
)
CGROUP_MEMORY_FIELDS = frozenset(
    {
        "current_bytes",
        "headroom_bytes",
        "limit_bytes",
        "minimum_required_headroom_bytes",
        "observed_peak_bytes",
        "passed",
        "peak_bytes",
        "requested_bytes",
    }
)
EXECUTION_FIELDS = frozenset(
    {
        "allocation_sha256",
        "attempt",
        "execution_profile",
        "job_id",
        "manifest_sha256",
    }
)
WORKFLOW_ID_PATTERN = re.compile(
    rf"{re.escape(WORKFLOW_ID_PREFIX)}[0-9a-f]{{32}}\Z"
)


def _execution_context(gpu_count: int) -> dict[str, object]:
    return {
        "allocation_contract": "manifest_driven_scheduler_gpu_v1",
        "allocation_visibility": "preserved",
        "cpu_core_count": CPU_CORE_COUNT,
        "distributed_launcher": "environment_rank_passthrough",
        "gpu_count_policy": "scheduler",
        "mode": "server_scheduler_foreground",
        "nccl_p2p_policy": "disabled",
        "threads_per_rank": CPU_CORE_COUNT // gpu_count,
        "visible_device_count": gpu_count,
    }


def nccl_expected_sum(gpu_count: int) -> float:
    if gpu_count not in SUPPORTED_GPU_COUNTS:
        _fail("GPU count is outside the reviewed elastic topology")
    return float(gpu_count * (gpu_count + 1) // 2)


def validate_preflight_workflow_id(value: object) -> str:
    """Require one opaque, count-neutral preflight publication namespace."""

    if not isinstance(value, str) or WORKFLOW_ID_PATTERN.fullmatch(value) is None:
        _fail("GPU preflight workflow_id is not a canonical opaque instance")
    return value


def _fail(message: str) -> None:
    raise AdapterValidationError(message)


def _strict_equal(observed: object, expected: object) -> bool:
    """Compare decoded JSON without treating booleans as numeric values."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _strict_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _strict_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def microbatch_sizes(gpu_count: int, rank: int) -> tuple[int, ...]:
    """Return the reviewed rank-local schedule for one global batch of 64."""

    if gpu_count not in SUPPORTED_GPU_COUNTS or not 0 <= rank < gpu_count:
        _fail("rank is outside the reviewed elastic topology")
    local_sequence_count = len(range(rank, GLOBAL_LOGICAL_BATCH_SIZE, gpu_count))
    microsteps = math.ceil(
        GLOBAL_LOGICAL_BATCH_SIZE
        / (gpu_count * MAX_PER_RANK_MICROBATCH_SIZE)
    )
    remaining = local_sequence_count
    result: list[int] = []
    for _ in range(microsteps):
        size = min(MAX_PER_RANK_MICROBATCH_SIZE, remaining)
        if size < 1:
            _fail("exact optimizer window would issue an empty microbatch")
        result.append(size)
        remaining -= size
    if remaining != 0:
        _fail("exact optimizer window did not consume its local sequences")
    return tuple(result)


def training_contract() -> dict[str, object]:
    """Return the allocation-neutral production-shaped preflight contract."""

    return {
        "batch_partition_protocol": BATCH_PARTITION_PROTOCOL,
        "canonical_sft_objective": CANONICAL_SFT_OBJECTIVE,
        "derived_max_model_input_tokens": (
            FROZEN_PROMPT_POPULATION_MAX_TOKENS + TEACHER_DEMO_MAX_NEW_TOKENS
        ),
        "fsdp_auto_wrap_policy": FSDP_AUTO_WRAP_POLICY,
        "fsdp_requested_sharding_strategy": REQUESTED_FSDP_SHARDING_STRATEGY,
        "fsdp_effective_sharding_strategy_by_world_size": {
            str(world_size): strategy
            for world_size, strategy in EFFECTIVE_FSDP_SHARDING_STRATEGY_BY_WORLD_SIZE.items()
        },
        "fsdp_state_dict_type": FSDP_STATE_DICT_TYPE,
        "fsdp_transformer_layer": FSDP_TRANSFORMER_LAYER,
        "frozen_prompt_population_max_tokens": FROZEN_PROMPT_POPULATION_MAX_TOKENS,
        "gpu_memory_budget_mib": GPU_MEMORY_BUDGET_MIB,
        "global_logical_batch_size": GLOBAL_LOGICAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "loss_scaling_protocol": LOSS_SCALING_PROTOCOL,
        "max_model_input_length": MAX_MODEL_INPUT_LENGTH,
        "max_per_rank_microbatch_size": MAX_PER_RANK_MICROBATCH_SIZE,
        "optimizer_class": OPTIMIZER_CLASS,
        "overlength_policy": "reject_without_truncation_before_any_training_forward",
        "sequence_normalization": True,
        "teacher_demo_max_new_tokens": TEACHER_DEMO_MAX_NEW_TOKENS,
        "weight_decay": WEIGHT_DECAY,
    }


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} is not a lowercase SHA-256 digest")
    return value


def _git_commit(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} is not a lowercase Git commit")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} is not an integer at least {minimum}")
    return value


def _utc_timestamp(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail(f"{name} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise AdapterValidationError(f"{name} is not an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(f"{name} must use UTC")
    return value


def _validate_devices(value: object, *, gpu_count: int) -> None:
    if not isinstance(value, list) or len(value) != gpu_count:
        _fail(f"GPU preflight must report exactly {gpu_count} devices")
    ranks: list[int] = []
    pci_bus_ids: list[str] = []
    visible_identifiers: list[str] = []
    minimum_bytes = GPU_MINIMUM_TOTAL_MEMORY_MIB * 1024**2
    for row in value:
        if not isinstance(row, dict) or set(row) != DEVICE_FIELDS:
            _fail("GPU preflight device fields differ from the reviewed contract")
        rank = _integer(row["rank"], name="GPU device rank")
        ranks.append(rank)
        if _integer(row["logical_index"], name="GPU logical index") != rank:
            _fail("GPU logical index differs from its single-node rank")
        pci_bus_id = row["pci_bus_id"]
        if not isinstance(pci_bus_id, str) or re.fullmatch(
            r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci_bus_id
        ) is None:
            _fail("GPU preflight PCI bus identity is not canonical")
        pci_bus_ids.append(pci_bus_id)
        visible_identifier = row["visible_cuda_identifier"]
        if (
            not isinstance(visible_identifier, str)
            or not visible_identifier
            or visible_identifier.strip() != visible_identifier
        ):
            _fail("GPU preflight visible CUDA identity is not canonical")
        visible_identifiers.append(visible_identifier)
        if row["name"] != GPU_MODEL:
            _fail("GPU preflight observed an unreviewed GPU model")
        if _integer(row["total_memory"], name="GPU total memory", minimum=1) < minimum_bytes:
            _fail("GPU preflight device memory is below the reviewed server minimum")
        capability = row["capability"]
        if capability != list(GPU_COMPUTE_CAPABILITY):
            _fail("GPU preflight compute capability differs from the reviewed server")
    if sorted(ranks) != list(range(gpu_count)):
        _fail(f"GPU preflight device ranks are not exactly 0..{gpu_count - 1}")
    if (
        len(set(pci_bus_ids)) != gpu_count
        or len(set(visible_identifiers)) != gpu_count
    ):
        _fail("GPU preflight device identities are not unique")


def _validate_nccl_diagnostics(value: object, *, gpu_count: int) -> None:
    if not isinstance(value, list) or len(value) != gpu_count:
        _fail(f"GPU preflight must report exactly {gpu_count} NCCL diagnostics")
    ranks: list[int] = []
    expected_sum = nccl_expected_sum(gpu_count)
    for row in value:
        if not isinstance(row, dict) or set(row) != NCCL_DIAGNOSTIC_FIELDS:
            _fail("GPU preflight NCCL diagnostic fields differ from the reviewed contract")
        rank = _integer(row["rank"], name="NCCL diagnostic rank")
        ranks.append(rank)
        elapsed = row["elapsed_seconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0.0
            or elapsed > NCCL_PROBE_TIMEOUT_SECONDS + 1.0
        ):
            _fail("GPU preflight NCCL probe elapsed time is invalid")
        if (
            row["control_backend"] != "gloo"
            or row["data_backend"] != "nccl"
            or row["p2p_disabled"] is not True
            or row["timeout_seconds"] != NCCL_PROBE_TIMEOUT_SECONDS
            or isinstance(row["observed_sum"], bool)
            or not isinstance(row["observed_sum"], (int, float))
            or not math.isfinite(row["observed_sum"])
            or row["observed_sum"] != expected_sum
        ):
            _fail("GPU preflight NCCL diagnostic did not pass its fixed contract")
    if sorted(ranks) != list(range(gpu_count)):
        _fail(
            f"GPU preflight NCCL diagnostic ranks are not exactly 0..{gpu_count - 1}"
        )


def _validate_rank_checks(
    value: object, *, gpu_count: int
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != gpu_count:
        _fail(f"GPU preflight must report exactly {gpu_count} rank training checks")
    rows: list[dict[str, Any]] = []
    ranks: list[int] = []
    prompt_hashes: list[str] = []
    checkpoint_hashes: list[str] = []
    checkpoint_manifest_hashes: list[str] = []
    canonical_losses: list[float] = []
    for raw_row in value:
        if not isinstance(raw_row, dict) or set(raw_row) != RANK_CHECK_FIELDS:
            _fail("GPU preflight rank-check fields differ from the reviewed contract")
        row = raw_row
        rank = _integer(row["rank"], name="GPU preflight rank")
        ranks.append(rank)
        for field in (
            "raw_prompt_sha256",
            "model_facing_prompt_sha256",
            "checkpoint_sha256",
            "checkpoint_manifest_sha256",
            "checkpoint_runtime_sha256",
        ):
            _sha256(row[field], name=f"rank {rank} {field}")
        prompt_hashes.append(row["model_facing_prompt_sha256"])
        checkpoint_hashes.append(row["checkpoint_sha256"])
        checkpoint_manifest_hashes.append(row["checkpoint_manifest_sha256"])
        expected_microbatches = list(microbatch_sizes(gpu_count, rank))
        expected_slots = list(range(rank, GLOBAL_LOGICAL_BATCH_SIZE, gpu_count))
        for field, minimum in (
            ("global_logical_batch_size", 1),
            ("global_model_input_tokens", 1),
            ("local_sequence_count", 1),
            ("max_model_input_length", 1),
            ("max_per_rank_microbatch_size", 1),
            ("microsteps_per_optimizer_update", 1),
            ("observed_max_model_input_length", 1),
            ("optimizer_updates", 1),
            ("response_mask_tokens", 1),
            ("response_tokens_per_sequence", 1),
            ("fsdp_wrapper_count", 1),
        ):
            _integer(row[field], name=f"rank {rank} {field}", minimum=minimum)
        if (
            not isinstance(row["microbatch_sizes"], list)
            or any(type(item) is not int for item in row["microbatch_sizes"])
            or not isinstance(row["global_slots"], list)
            or any(type(item) is not int for item in row["global_slots"])
        ):
            _fail("GPU preflight rank schedule contains a non-integer value")
        if (
            row["prompt_protocol"] != PROMPT_PROTOCOL
            or row["enable_thinking"] is not False
            or row["student_revision"] != MODEL_REVISION
            or row["teacher_revision"] != TEACHER_REVISION
            or row["tokenizer_revision"] != TOKENIZER_REVISION
            or row["tokenizer_fingerprint"] != TOKENIZER_FINGERPRINT
            or row["chat_template_sha256"] != CHAT_TEMPLATE_SHA256
            or row["loading_strategy"] != "low_cpu_mem_student_rank_zero_teacher"
            or row["batch_partition_protocol"] != BATCH_PARTITION_PROTOCOL
            or row["canonical_sft_objective"] != CANONICAL_SFT_OBJECTIVE
            or row["sequence_normalization"] is not True
            or row["loss_scaling_protocol"] != LOSS_SCALING_PROTOCOL
            or row["optimizer_class"] != OPTIMIZER_CLASS
            or row["optimizer_updates"] != 1
            or row["global_logical_batch_size"] != GLOBAL_LOGICAL_BATCH_SIZE
            or row["max_per_rank_microbatch_size"]
            != MAX_PER_RANK_MICROBATCH_SIZE
            or row["max_model_input_length"] != MAX_MODEL_INPUT_LENGTH
            or row["observed_max_model_input_length"]
            != MAX_MODEL_INPUT_LENGTH
            or row["global_model_input_tokens"]
            != GLOBAL_LOGICAL_BATCH_SIZE * MAX_MODEL_INPUT_LENGTH
            or row["microbatch_sizes"] != expected_microbatches
            or row["microsteps_per_optimizer_update"]
            != len(expected_microbatches)
            or row["global_slots"] != expected_slots
            or row["local_sequence_count"] != len(expected_slots)
            or row["response_tokens_per_sequence"]
            != TEACHER_DEMO_MAX_NEW_TOKENS
            or row["response_mask_tokens"]
            != len(expected_slots) * TEACHER_DEMO_MAX_NEW_TOKENS
            or row["requested_fsdp_sharding_strategy"]
            != REQUESTED_FSDP_SHARDING_STRATEGY
            or row["effective_fsdp_sharding_strategy"]
            != effective_fsdp_sharding_strategy(gpu_count)
            or row["fsdp_state_dict_type"] != FSDP_STATE_DICT_TYPE
            or row["fsdp_auto_wrap_policy"] != FSDP_AUTO_WRAP_POLICY
            or row["fsdp_transformer_layer"] != FSDP_TRANSFORMER_LAYER
        ):
            _fail("GPU preflight rank is not bound to the fixed Qwen3-v2 protocol")
        for field in (
            "teacher_forward_finite",
            "student_forward_finite",
            "canonical_sft_loss_finite",
            "gradients_finite",
            "parameter_update_nonzero",
            "unique_rank_prompt_shard",
            "fsdp_save_resume",
            "optimizer_state_restored",
            "scheduler_state_restored",
            "rng_state_restored",
        ):
            if row[field] is not True:
                _fail(f"GPU preflight rank {rank} failed {field}")
        if row["teacher_loaded_on_this_rank"] is not (rank == 0):
            _fail("GPU preflight teacher must be loaded on rank zero only")
        loss = row["canonical_sft_loss"]
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(loss)
            or loss <= 0
        ):
            _fail("GPU preflight canonical-SFT loss is not finite and positive")
        canonical_losses.append(float(loss))
        wrapped_transformer_blocks = _integer(
            row["wrapped_transformer_blocks"],
            name=f"rank {rank} wrapped transformer blocks",
            minimum=1,
        )
        if row["fsdp_wrapper_count"] != wrapped_transformer_blocks + 1:
            _fail("GPU preflight FSDP wrapper count differs from its auto-wrap evidence")
        allocated = _integer(
            row["max_memory_allocated"], name=f"rank {rank} allocated GPU peak", minimum=1
        )
        reserved = _integer(
            row["max_memory_reserved"], name=f"rank {rank} reserved GPU peak", minimum=1
        )
        if reserved < allocated or reserved > GPU_MEMORY_BUDGET_MIB * 1024**2:
            _fail("GPU preflight GPU peak is outside the registered memory envelope")
        _integer(row["process_max_rss_bytes"], name=f"rank {rank} MaxRSS", minimum=1)
        rows.append(row)
    if (
        sorted(ranks) != list(range(gpu_count))
        or len(set(prompt_hashes)) != gpu_count
        or len(set(checkpoint_hashes)) != 1
        or len(set(checkpoint_manifest_hashes)) != 1
        or len(set(canonical_losses)) != 1
        or sorted(slot for row in rows for slot in row["global_slots"])
        != list(range(GLOBAL_LOGICAL_BATCH_SIZE))
    ):
        _fail(
            "GPU preflight rank identities, exact slots, prompts, or full checkpoint disagree"
        )
    return rows


def _validate_cgroup_memory(value: object, *, rank_rows: list[dict[str, Any]]) -> None:
    if not isinstance(value, dict) or set(value) != CGROUP_MEMORY_FIELDS:
        _fail("GPU preflight cgroup-memory fields differ from the reviewed contract")
    if value["passed"] is not True:
        _fail("GPU preflight cgroup-memory gate did not pass")
    limit = _integer(value["limit_bytes"], name="cgroup memory limit", minimum=1)
    current = _integer(value["current_bytes"], name="cgroup current memory")
    peak = _integer(value["peak_bytes"], name="cgroup peak memory", minimum=1)
    requested = _integer(value["requested_bytes"], name="requested memory", minimum=1)
    observed = _integer(value["observed_peak_bytes"], name="observed memory peak", minimum=1)
    headroom = _integer(value["headroom_bytes"], name="memory headroom")
    minimum = _integer(
        value["minimum_required_headroom_bytes"], name="minimum memory headroom", minimum=1
    )
    expected_requested = NODE_MEMORY_GIB * 1024**3
    expected_minimum = max(
        MINIMUM_HEADROOM_GIB * 1024**3,
        int(limit * MINIMUM_HEADROOM_FRACTION),
    )
    process_peak = sum(int(row["process_max_rss_bytes"]) for row in rank_rows)
    if (
        requested != expected_requested
        or limit != requested
        or current > limit
        or peak > limit
        or observed != max(peak, process_peak)
        or headroom != limit - observed
        or minimum != expected_minimum
        or headroom < minimum
    ):
        _fail("GPU preflight cgroup-memory evidence is internally inconsistent")


def _validate_report(
    report: object,
    *,
    resolved_config_sha256: str,
    preregistration_sha256: str,
    completion_execution: object,
) -> None:
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
        _fail("GPU preflight report fields differ from the reviewed contract")
    digest = _sha256(report["sha256"], name="GPU preflight report sha256")
    unsigned = {key: value for key, value in report.items() if key != "sha256"}
    if digest != sha256_value(unsigned):
        _fail("GPU preflight report self-summary is invalid")
    gpu_count = _integer(
        report["world_size"], name="GPU preflight world_size", minimum=1
    )
    visible_device_count = _integer(
        report["visible_cuda_devices"],
        name="GPU preflight visible_cuda_devices",
        minimum=1,
    )
    if gpu_count not in SUPPORTED_GPU_COUNTS:
        _fail("GPU preflight world_size is outside the reviewed 1..4 topology")
    if CPU_CORE_COUNT % gpu_count != 0:
        _fail("GPU preflight world_size does not divide the CPU allocation")
    if (
        report["phase"] != "gpu_preflight"
        or report["passed"] is not True
        or visible_device_count != gpu_count
        or report["nccl_all_reduce"] is not True
        or report["nccl_runtime_version"] != NCCL_VERSION
        or report["qwen_forward_finite"] is not True
        or report["protocol_track"] != PROTOCOL_TRACK
        or report["artifact_namespace"] != ARTIFACT_NAMESPACE
        or report["execution_context"] != _execution_context(gpu_count)
        or not _strict_equal(report["training_contract"], training_contract())
    ):
        _fail("GPU preflight top-level execution gates did not pass")
    execution = report["execution"]
    if (
        not isinstance(execution, dict)
        or set(execution) != EXECUTION_FIELDS
        or execution != completion_execution
        or execution.get("execution_profile") != PROFILE_NAME
    ):
        _fail("GPU preflight report execution differs from its completion allocation")
    torch_version = report["torch_version"]
    cuda_version = report["torch_cuda_version"]
    if (
        not isinstance(torch_version, str)
        or not torch_version.startswith("2.8.0")
        or cuda_version != "12.8"
    ):
        _fail("GPU preflight did not use the reviewed CUDA-enabled Torch release")
    if (
        report["model_revision"] != MODEL_REVISION
        or report["resolved_model_commit"] != MODEL_REVISION
        or report["teacher_revision"] != TEACHER_REVISION
        or report["resolved_teacher_commit"] != TEACHER_REVISION
        or report["tokenizer_revision"] != TOKENIZER_REVISION
        or report["tokenizer_hash"] != TOKENIZER_FINGERPRINT
        or report["tokenizer_fingerprint"] != TOKENIZER_FINGERPRINT
        or report["chat_template_sha256"] != CHAT_TEMPLATE_SHA256
        or report["prompt_protocol"] != PROMPT_PROTOCOL
        or report["enable_thinking"] is not False
        or report["prereg_path"] != PREREG_PATH
        or report["prereg_version"] != PREREG_VERSION
        or report["prereg_sha256"] != preregistration_sha256
        or report["resolved_config_sha256"] != resolved_config_sha256
    ):
        _fail("GPU preflight report differs from the fixed Qwen3-v2 inputs")
    git_commit = _git_commit(report["git_commit"], name="GPU preflight git_commit")
    if report["code_commit"] != git_commit:
        _fail("GPU preflight code_commit differs from git_commit")
    _git_commit(report["prereg_commit"], name="GPU preflight prereg_commit")
    _utc_timestamp(report["created_at"], name="GPU preflight created_at")
    _validate_devices(report["devices"], gpu_count=gpu_count)
    _validate_nccl_diagnostics(report["nccl_diagnostics"], gpu_count=gpu_count)
    rank_rows = _validate_rank_checks(
        report["rank_training_checks"], gpu_count=gpu_count
    )
    if report["rank_prompt_hashes_unique"] is not True:
        _fail("GPU preflight did not attest unique rank prompt shards")
    if report["rank_zero_teacher_load_count"] != 1:
        _fail("GPU preflight did not load exactly one rank-zero teacher")
    _validate_cgroup_memory(report["cgroup_memory"], rank_rows=rank_rows)


def validate_qwen3_v2_gpu_preflight_completion(
    completion: Mapping[str, Any], context: Any
) -> None:
    """Validate the scientific output and every fixed Qwen3-v2 preflight gate."""

    if not isinstance(completion, Mapping):
        _fail("GPU preflight completion must be a mapping")
    workflow_id = validate_preflight_workflow_id(completion.get("workflow_id"))
    expected_workflow_id = getattr(context, "workflow_id", None)
    if expected_workflow_id is not None and workflow_id != expected_workflow_id:
        _fail("GPU preflight completion workflow_id differs from its workflow plan")
    output_paths = getattr(context, "expected_output_paths", None)
    if not isinstance(output_paths, Mapping) or set(output_paths) != {OUTPUT_NAME}:
        _fail("GPU preflight output contract is invalid")
    expected_inputs = getattr(context, "expected_input_hashes", None)
    if not isinstance(expected_inputs, Mapping):
        _fail("GPU preflight expected inputs are invalid")
    if completion.get("input_hashes") != dict(expected_inputs):
        _fail("GPU preflight completion input hashes differ")
    required_inputs = {
        "config_binding_sha256",
        "execution_config_sha256",
        "preregistration_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
    }
    if set(expected_inputs) != required_inputs:
        _fail("GPU preflight content inputs differ from the reviewed contract")
    for name, digest in expected_inputs.items():
        _sha256(digest, name=f"GPU preflight input {name}")
    for field in (
        "execution_config_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
    ):
        if completion.get(field) != expected_inputs[field]:
            _fail(f"GPU preflight completion {field} differs from its content input")
    gates = completion.get("scientific_validation")
    if not isinstance(gates, dict) or tuple(sorted(gates)) != GATE_NAMES:
        _fail("GPU preflight completion gate names differ from the reviewed contract")
    if any(value is not True for value in gates.values()):
        _fail("GPU preflight completion contains a failed scientific gate")

    # The adapter validates once while staging is descriptor-held and once after
    # atomic publication. Report bytes are available only for the latter call.
    try:
        output_paths[OUTPUT_NAME].lstat()
    except FileNotFoundError:
        return
    report, _file_sha256 = read_strict_json(
        output_paths[OUTPUT_NAME],
        context="Qwen3-v2 GPU preflight report",
        max_bytes=32 * 1024 * 1024,
    )
    _validate_report(
        report,
        resolved_config_sha256=expected_inputs["resolved_config_sha256"],
        preregistration_sha256=expected_inputs["preregistration_sha256"],
        completion_execution=completion.get("execution"),
    )


__all__ = [
    "ARTIFACT_NAMESPACE",
    "BATCH_PARTITION_PROTOCOL",
    "CANONICAL_SFT_OBJECTIVE",
    "CHAT_TEMPLATE_SHA256",
    "GATE_NAMES",
    "CPU_CORE_COUNT",
    "GPU_MODEL",
    "GLOBAL_LOGICAL_BATCH_SIZE",
    "MAX_MODEL_INPUT_LENGTH",
    "MAX_PER_RANK_MICROBATCH_SIZE",
    "MODEL_REVISION",
    "NCCL_PROBE_TIMEOUT_SECONDS",
    "NCCL_VERSION",
    "OUTPUT_NAME",
    "PROFILE_NAME",
    "PROTOCOL_TRACK",
    "SUPPORTED_GPU_COUNTS",
    "TASK_NAME",
    "TEACHER_REVISION",
    "TOKENIZER_FINGERPRINT",
    "UNIT_ID",
    "WORKFLOW_ID_PREFIX",
    "microbatch_sizes",
    "nccl_expected_sum",
    "training_contract",
    "validate_preflight_workflow_id",
    "validate_qwen3_v2_gpu_preflight_completion",
]
