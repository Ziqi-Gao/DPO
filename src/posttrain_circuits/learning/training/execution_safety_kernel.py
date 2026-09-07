"""Stable scheduler-managed GPU execution-safety semantics.

This module intentionally excludes experiment identity, output locations, and
scientific thresholds.  It is the reusable execution class shared by request
validation, the runtime handler, training finalization, and completion replay.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from posttrain_circuits.learning.training.fsdp_contract import (
    REQUESTED_FSDP_SHARDING_STRATEGY,
    effective_fsdp_sharding_strategy,
)


EXECUTION_CLASS_ID = "qwen3-v2-elastic-training-v1"
KERNEL_SCHEMA_VERSION = 1
SUPPORTED_WORLD_SIZES = (1, 2, 3, 4)
BATCH_PARTITION_PROTOCOL = "allocation_neutral_exact_global_batch_v1"
GLOBAL_BATCH_SIZE = 64
MAX_MICROBATCH_SIZE = 4
MAX_MODEL_INPUT_LENGTH = 1536
MAX_PROMPT_POPULATION_SIZE = 256
TOKEN_BUDGET = 2_000_000
MAX_OPTIMIZER_STEPS = 120

SAMPLES_BY_WORLD_SIZE = {
    1: (64,),
    2: (32, 32),
    3: (22, 21, 21),
    4: (16, 16, 16, 16),
}
MICROBATCHES_BY_WORLD_SIZE = {
    1: ((4,) * 16,),
    2: ((4,) * 8, (4,) * 8),
    3: (
        (4, 4, 4, 4, 4, 2),
        (4, 4, 4, 4, 4, 1),
        (4, 4, 4, 4, 4, 1),
    ),
    4: ((4,) * 4, (4,) * 4, (4,) * 4, (4,) * 4),
}

EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION = {
    "experiment": {
        "state_source": "teacher_demo",
        "supervision": "canonical_sft",
        "use_verifier_reward": False,
    },
    "full_parameter_training": True,
    "model": {
        "allow_unpinned_revision": False,
        "attn_implementation": "sdpa",
        "gradient_checkpointing": True,
        "low_cpu_mem_usage": True,
        "model_name_or_path": "Qwen/Qwen3-1.7B",
        "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "prompt_protocol": {
            "add_generation_prompt": True,
            "chat_template_sha256": (
                "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
            ),
            "enable_thinking": False,
            "messages": "single_user",
            "name": "qwen3_non_thinking_v1",
        },
        "tokenizer_fingerprint": (
            "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
        ),
        "tokenizer_name_or_path": "Qwen/Qwen3-1.7B",
        "tokenizer_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "torch_dtype": "bfloat16",
        "trust_remote_code": False,
        "use_cache": False,
    },
    "state_source": {
        "max_new_tokens": 256,
        "max_prompt_tokens": 1246,
        "name": "teacher_demo",
        "num_candidates_within_memory_envelope": True,
        "require_behavior_logprobs_for_formal_sft": True,
        "require_complete_prompt_coverage": True,
        "require_exact_verifier_success": True,
        "zero_success_policy": "fail_closed",
    },
    "supervision": {
        "mask_prompt_tokens": True,
        "name": "canonical_sft",
        "normalization": "sequence",
        "training_backend": "factorial_trainer",
        "training_backend_version": "in-repository-v1",
        "training_batch_contract": "optimizer-boundary-gradient-accumulation-v1",
    },
    "task": {
        "dataset_family_within_memory_envelope": True,
        "prompt_population_within_memory_envelope": True,
    },
    "teacher": {
        "allow_unpinned_revision": False,
        "attn_implementation": "sdpa",
        "gradient_checkpointing": False,
        "low_cpu_mem_usage": True,
        "model_name_or_path": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "rank_zero_only_training_load": True,
        "tokenizer_fingerprint": (
            "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
        ),
        "tokenizer_name_or_path": "Qwen/Qwen3-8B",
        "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "torch_dtype": "bfloat16",
        "trust_remote_code": False,
        "use_cache": True,
    },
    "trainer": {
        "backend": "accelerate",
        "batch_partition_protocol": BATCH_PARTITION_PROTOCOL,
        "checkpoint_every": 20,
        "evaluation_every": 20,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "max_completion_length": 128,
        "max_microbatch_size": MAX_MICROBATCH_SIZE,
        "max_model_input_length": MAX_MODEL_INPUT_LENGTH,
        "max_steps": MAX_OPTIMIZER_STEPS,
        "steps_per_round": 1,
        "token_budget": TOKEN_BUDGET,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "validation_examples": 128,
    },
}


def _sha256_value(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"execution-safety config lacks mapping {name}")
    return value


def execution_safety_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project only distributed execution and bounded-memory semantics."""

    if not isinstance(config, Mapping):
        raise ValueError("execution-safety config must be a mapping")

    def selected(name: str, fields: tuple[str, ...]) -> dict[str, Any]:
        value = _section(config, name)
        return {field: copy.deepcopy(value.get(field)) for field in fields}

    experiment = _section(config, "experiment")
    g0 = _section(config, "g0")
    state_source = _section(config, "state_source")
    task = _section(config, "task")
    num_candidates = state_source.get("num_candidates")
    if type(num_candidates) is not int or not 1 <= num_candidates <= 8:
        raise ValueError(
            "teacher-demo candidate count exceeds the certified memory envelope"
        )
    prompt_population = task.get("num_examples")
    _validate_prompt_population(prompt_population)
    split_limits = {
        "circuit_discovery": 2000,
        "circuit_validation": 2000,
        "iid_test": 10000,
        "ood_depth_test": 10000,
        "ood_structure_test": 10000,
        "train": 100000,
        "validation": 10000,
    }
    split_sizes = task.get("split_sizes")
    if (
        not isinstance(split_sizes, Mapping)
        or set(split_sizes) != set(split_limits)
        or any(
            type(split_sizes[name]) is not int
            or not 1 <= split_sizes[name] <= limit
            for name, limit in split_limits.items()
        )
    ):
        raise ValueError("dataset split sizes exceed the certified memory envelope")
    return {
        "experiment": {
            field: copy.deepcopy(experiment.get(field))
            for field in ("state_source", "supervision", "use_verifier_reward")
        },
        "full_parameter_training": copy.deepcopy(g0.get("full_parameter_training")),
        "model": selected(
            "model",
            (
                "allow_unpinned_revision",
                "attn_implementation",
                "gradient_checkpointing",
                "low_cpu_mem_usage",
                "model_name_or_path",
                "model_revision",
                "prompt_protocol",
                "tokenizer_fingerprint",
                "tokenizer_name_or_path",
                "tokenizer_revision",
                "torch_dtype",
                "trust_remote_code",
                "use_cache",
            ),
        ),
        "state_source": {
            **selected(
                "state_source",
                (
                    "max_new_tokens",
                    "max_prompt_tokens",
                    "name",
                    "require_behavior_logprobs_for_formal_sft",
                    "require_complete_prompt_coverage",
                    "require_exact_verifier_success",
                    "zero_success_policy",
                ),
            ),
            "num_candidates_within_memory_envelope": True,
        },
        "supervision": selected(
            "supervision",
            (
                "mask_prompt_tokens",
                "name",
                "normalization",
                "training_backend",
                "training_backend_version",
                "training_batch_contract",
            ),
        ),
        "task": {
            "dataset_family_within_memory_envelope": True,
            "prompt_population_within_memory_envelope": True,
        },
        "teacher": selected(
            "teacher",
            (
                "allow_unpinned_revision",
                "attn_implementation",
                "gradient_checkpointing",
                "low_cpu_mem_usage",
                "model_name_or_path",
                "model_revision",
                "rank_zero_only_training_load",
                "tokenizer_fingerprint",
                "tokenizer_name_or_path",
                "tokenizer_revision",
                "torch_dtype",
                "trust_remote_code",
                "use_cache",
            ),
        ),
        "trainer": selected(
            "trainer",
            (
                "backend",
                "batch_partition_protocol",
                "checkpoint_every",
                "evaluation_every",
                "global_batch_size",
                "max_completion_length",
                "max_microbatch_size",
                "max_model_input_length",
                "max_steps",
                "steps_per_round",
                "token_budget",
                "token_budget_unit",
                "validation_examples",
            ),
        ),
    }


def _validate_prompt_population(value: object) -> int:
    if (
        type(value) is not int
        or not GLOBAL_BATCH_SIZE <= value <= MAX_PROMPT_POPULATION_SIZE
        or value % GLOBAL_BATCH_SIZE
    ):
        raise ValueError(
            "prompt population is outside the certified aligned memory envelope"
        )
    return value


def batch_token_contract(
    world_size: int,
    prompt_population_size: int,
) -> dict[str, Any]:
    """Return the exact global-64 realization for one claimed allocation."""

    if type(world_size) is not int or world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError("world size is outside the certified 1..4 execution class")
    prompt_population_size = _validate_prompt_population(prompt_population_size)
    schedules = MICROBATCHES_BY_WORLD_SIZE[world_size]
    samples = SAMPLES_BY_WORLD_SIZE[world_size]
    if (
        len(schedules) != world_size
        or tuple(sum(schedule) for schedule in schedules) != samples
        or sum(samples) != GLOBAL_BATCH_SIZE
        or len({len(schedule) for schedule in schedules}) != 1
    ):
        raise ValueError("execution-safety batch partition is inconsistent")
    return {
        "accepted_view_prompt_order": "exactly_manifest_ordered_prompt_ids",
        "batch_partition_protocol": BATCH_PARTITION_PROTOCOL,
        "effective_fsdp_sharding_strategy": effective_fsdp_sharding_strategy(
            world_size
        ),
        "full_parameter_training": True,
        "global_logical_batch_size": GLOBAL_BATCH_SIZE,
        "max_model_input_length": MAX_MODEL_INPUT_LENGTH,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "max_per_rank_microbatch_size": MAX_MICROBATCH_SIZE,
        "microbatch_schedule_by_rank": [list(row) for row in schedules],
        "optimizer_microsteps": len(schedules[0]),
        "prompt_ids_unique": True,
        "prompt_population_alignment": "exact_multiple_of_global_logical_batch_size",
        "prompt_population_size": prompt_population_size,
        "requested_fsdp_sharding_strategy": REQUESTED_FSDP_SHARDING_STRATEGY,
        "samples_by_rank": list(samples),
        "token_budget": TOKEN_BUDGET,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "world_size": world_size,
    }


def distributed_resume_checks(
    resume: Mapping[str, Any],
    *,
    max_optimizer_steps: int,
    world_size: int,
    certified_fsdp_wrapper_count: int,
) -> dict[str, bool]:
    """Validate actual-world checkpoint/resume and FSDP evidence."""

    if type(world_size) is not int or world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError("resume world size is outside the certified execution class")
    fsdp_contract = resume.get("fsdp_sharding_contract")
    return {
        "distributed_checkpoint_resume": (
            resume.get("format_version") == 3
            and resume.get("passed") is True
            and type(resume.get("world_size")) is int
            and resume.get("world_size") == world_size
            and type(resume.get("max_optimizer_steps")) is int
            and resume.get("max_optimizer_steps") == max_optimizer_steps
            and isinstance(resume.get("checks"), Mapping)
            and bool(resume["checks"])
            and all(value is True for value in resume["checks"].values())
        ),
        "distributed_resume_fsdp_strategy": (
            isinstance(fsdp_contract, Mapping)
            and fsdp_contract.get("requested_fsdp_sharding_strategy")
            == REQUESTED_FSDP_SHARDING_STRATEGY
            and fsdp_contract.get("effective_fsdp_sharding_strategy")
            == effective_fsdp_sharding_strategy(world_size)
            and type(fsdp_contract.get("fsdp_wrapper_count")) is int
            and fsdp_contract["fsdp_wrapper_count"]
            == certified_fsdp_wrapper_count
        ),
    }


def build_runtime_execution_safety_attestation(
    *,
    config: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    fingerprint_sha256: str,
    resume: Mapping[str, Any],
    world_size: int,
) -> dict[str, Any]:
    """Fail closed and summarize the handler's independent final safety gate."""

    subject = descriptor.get("subject")
    if (
        descriptor.get("execution_class_id") != EXECUTION_CLASS_ID
        or descriptor.get("fingerprint_sha256") != fingerprint_sha256
        or not isinstance(subject, Mapping)
    ):
        raise ValueError("runtime descriptor differs from the certified class")
    projection = execution_safety_config_projection(config)
    if (
        projection != EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION
        or projection != subject.get("resolved_config_safety_projection")
    ):
        raise ValueError("runtime config differs from the certified safety projection")
    fsdp = subject.get("fsdp")
    if not isinstance(fsdp, Mapping):
        raise ValueError("runtime descriptor lacks its FSDP contract")
    wrapper_count = fsdp.get("expected_wrapper_count")
    if type(wrapper_count) is not int or wrapper_count < 1:
        raise ValueError("runtime descriptor FSDP wrapper count is invalid")
    checks = distributed_resume_checks(
        resume,
        max_optimizer_steps=MAX_OPTIMIZER_STEPS,
        world_size=world_size,
        certified_fsdp_wrapper_count=wrapper_count,
    )
    if not all(checks.values()):
        raise ValueError("actual-world checkpoint/resume safety evidence failed")
    contract = batch_token_contract(world_size, _section(config, "task")["num_examples"])
    return {
        "batch_token_contract": contract,
        "checks": checks,
        "config_safety_projection_sha256": _sha256_value(projection),
        "execution_class_id": EXECUTION_CLASS_ID,
        "fingerprint_sha256": fingerprint_sha256,
        "schema_version": KERNEL_SCHEMA_VERSION,
        "world_size": world_size,
    }


def validate_runtime_execution_safety_attestation(
    value: object,
    *,
    expected_fingerprint_sha256: str,
    world_size: int,
) -> Mapping[str, Any]:
    """Validate the immutable handler-produced execution-safety summary."""

    if not isinstance(value, Mapping) or set(value) != {
        "batch_token_contract",
        "checks",
        "config_safety_projection_sha256",
        "execution_class_id",
        "fingerprint_sha256",
        "schema_version",
        "world_size",
    }:
        raise ValueError("runtime execution-safety attestation fields differ")
    checks = value["checks"]
    if (
        value["schema_version"] != KERNEL_SCHEMA_VERSION
        or isinstance(value["schema_version"], bool)
        or value["execution_class_id"] != EXECUTION_CLASS_ID
        or value["fingerprint_sha256"] != expected_fingerprint_sha256
        or value["world_size"] != world_size
        or not isinstance(value["config_safety_projection_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}", value["config_safety_projection_sha256"]
        )
        is None
        or not isinstance(checks, Mapping)
        or set(checks)
        != {"distributed_checkpoint_resume", "distributed_resume_fsdp_strategy"}
        or not all(check is True for check in checks.values())
        or not isinstance(value["batch_token_contract"], Mapping)
    ):
        raise ValueError("runtime execution-safety attestation did not pass")
    prompt_population = value["batch_token_contract"].get("prompt_population_size")
    if value["batch_token_contract"] != batch_token_contract(
        world_size, prompt_population
    ):
        raise ValueError("runtime execution-safety batch attestation differs")
    return value


__all__ = [
    "BATCH_PARTITION_PROTOCOL",
    "EXECUTION_CLASS_ID",
    "EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION",
    "GLOBAL_BATCH_SIZE",
    "MAX_MICROBATCH_SIZE",
    "MAX_MODEL_INPUT_LENGTH",
    "SUPPORTED_WORLD_SIZES",
    "batch_token_contract",
    "build_runtime_execution_safety_attestation",
    "distributed_resume_checks",
    "execution_safety_config_projection",
    "validate_runtime_execution_safety_attestation",
]
