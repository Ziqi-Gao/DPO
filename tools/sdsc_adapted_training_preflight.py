#!/usr/bin/env python3
"""Two-H100 execution preflight with the independently accepted dense teacher.

The physical global-64 FSDP canary is retained from sdsc_training_preflight;
only the teacher loading/identity and explicit reviewed successor binding differ.
This worker never submits jobs or grants G0, calibration or pilot acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import json
import math
import os
import re
import resource
import signal
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

TASK = "qwen3-v2-adapted-preflight"
KIND = "sdsc_h100_adapted_training_preflight_v1"
RESULT = "adapted-preflight.json"
WORLD_SIZE = 2
DEVICE_PROTOCOL_PATH = "prereg/amendments/qwen3_adapted_student_calibration_v3.json"
OPTIMIZER_PROTOCOL_PATH = "prereg/amendments/qwen3_adapted_student_calibration_v4.json"


def sibling(name):
    spec = importlib.util.spec_from_file_location(
        "_adapted_preflight_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the previously exercised, unchanged shared-partition execution guards.
# This module and its sibling imports perform no numerical imports at bootstrap.
guards = sibling("sdsc_training_preflight")
require = guards.require
log_phase = guards.log_phase
MODELS = guards.MODELS
collective_phase = guards.collective_phase
canary_rows = guards.canary_rows
tensor_digest = guards.tensor_digest
optimizer_digest = guards.optimizer_digest
checkpoint_names = guards.checkpoint_names
checkpoint_file = guards.checkpoint_file
verify_checkpoint = guards.verify_checkpoint
publish_json = guards.publish_json
memory_envelope = guards.memory_envelope


def staged_directory(path, work, label):
    require(
        path.is_absolute()
        and ".." not in path.parts
        and path.is_dir()
        and not any(item.is_symlink() for item in (path, *path.parents))
        and work in path.parents,
        label + " must be a real staged directory beneath the node workspace",
    )
    return path


def validate_inputs(args, contract=None):
    """Verify publication pins before importing any model/scientific package."""
    for name in (
        "code_sha256",
        "teacher_acceptance_sha256",
        "teacher_checkpoint_sha256",
        "student_protocol_sha256",
    ):
        require(re.fullmatch(r"[a-f0-9]{64}", getattr(args, name)), "invalid " + name)
    require(re.fullmatch(r"[a-f0-9]{40}", args.science_git_head), "invalid real science Git HEAD")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id), "invalid run ID")
    for name in ("science_root", "teacher_checkpoint_root", "teacher_acceptance"):
        staged_directory(getattr(args, name), args.work_dir, name)
    verified = (contract or sibling("sdsc_student_contract")).verify_teacher_acceptance(
        args.teacher_acceptance,
        expected_inventory_sha256=args.teacher_acceptance_sha256,
    )
    accepted = verified["accepted_teacher"]
    identity = verified["teacher_identity"]
    require(
        verified["inventory_sha256"] == args.teacher_acceptance_sha256
        and accepted["sha256"] == verified["accepted_teacher_sha256"]
        and accepted["formal_teacher_accepted"] is True
        and accepted["original_128_token_readiness_pass_claim"] is False
        and accepted["teacher_identity"] == identity
        and identity["kind"] == "learned_dense_checkpoint"
        and identity["teacher_checkpoint_sha256"] == args.teacher_checkpoint_sha256,
        "accepted teacher publication does not bind this dense checkpoint",
    )
    return {
        "teacher_identity": identity,
        "accepted_teacher_sha256": verified["accepted_teacher_sha256"],
        "teacher_acceptance_inventory_sha256": verified["inventory_sha256"],
        "student_protocol_sha256": args.student_protocol_sha256,
        "science_git_head": args.science_git_head,
    }


def validate_loaded_teacher(loaded, identity):
    """A base Hub revision is ancestry, never an adapted model identity."""
    pairs = {
        "teacher_checkpoint_sha256": "teacher_checkpoint_sha256",
        "base_revision": "base_revision",
        "tokenizer_id": "tokenizer_id",
        "requested_tokenizer_revision": "tokenizer_revision",
        "tokenizer_hash": "tokenizer_fingerprint",
        "chat_template_sha256": "chat_template_sha256",
        "prompt_protocol": "prompt_protocol",
    }
    require(identity["kind"] == "learned_dense_checkpoint", "teacher is not a learned dense checkpoint")
    require(
        all(getattr(loaded, key, None) == identity[expected] for key, expected in pairs.items()),
        "loaded dense teacher differs from independent acceptance",
    )


def teacher_probe_evidence(identity):
    return {
        "finite": True,
        "dense_checkpoint_verified": True,
        "teacher_identity": dict(identity),
        "base_revision_is_ancestry_only": True,
    }


def expected_batch_contract():
    """Pure transport projection of the unchanged two-rank global-64 kernel."""
    return {
        "accepted_view_prompt_order": "exactly_manifest_ordered_prompt_ids",
        "batch_partition_protocol": "allocation_neutral_exact_global_batch_v1",
        "effective_fsdp_sharding_strategy": "FULL_SHARD",
        "full_parameter_training": True,
        "global_logical_batch_size": 64,
        "max_model_input_length": 1536,
        "max_optimizer_steps": 120,
        "max_per_rank_microbatch_size": 4,
        "microbatch_schedule_by_rank": [[4] * 8, [4] * 8],
        "optimizer_microsteps": 8,
        "prompt_ids_unique": True,
        "prompt_population_alignment": "exact_multiple_of_global_logical_batch_size",
        "prompt_population_size": 256,
        "requested_fsdp_sharding_strategy": "FULL_SHARD",
        "samples_by_rank": [32, 32],
        "token_budget": 2000000,
        "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
        "world_size": 2,
    }


def activate_scientific_source(root):
    require(
        not any(
            name == "posttrain_circuits" or name.startswith("posttrain_circuits.") for name in sys.modules
        ),
        "scientific modules were loaded before staged checkout validation",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))


def validate_memory(memory):
    limit = 192 * 1024**3
    peak = memory.get("peak_bytes")
    current = memory.get("current_bytes")
    required = max(32 * 1024**3, math.ceil(limit * 0.20))
    require(
        memory.get("passed") is True
        and memory.get("limit_bytes") == limit
        and type(peak) is int
        and type(current) is int
        and 0 < current <= peak <= limit
        and memory.get("headroom_bytes") == limit - peak >= required
        and memory.get("minimum_headroom_bytes") == required,
        "preflight cgroup memory/headroom evidence differs",
    )
    ancestors = memory.get("ancestors", [])
    finite = [entry["limit_bytes"] for entry in ancestors if entry.get("limit_bytes") is not None]
    require(finite and min(finite) == limit, "preflight cgroup ancestor limits are missing/inconsistent")
    selected = [entry for entry in ancestors if entry.get("path") == memory.get("path")]
    require(
        len(selected) == 1
        and all(
            selected[0].get(key) == memory[key] for key in ("limit_bytes", "current_bytes", "peak_bytes")
        ),
        "preflight selected cgroup is not bound to ancestor evidence",
    )


def validate_completed_report(report, expected):
    expected_contract = expected_batch_contract()
    binding_keys = {
        "teacher_identity",
        "accepted_teacher_sha256",
        "teacher_acceptance_inventory_sha256",
        "student_protocol_sha256",
        "science_git_head",
    }
    if "student_protocol_path" in expected:
        binding_keys.add("student_protocol_path")
    require(binding_keys <= set(expected), "preflight expected scientific bindings are incomplete")
    require(
        all(report.get(key) == value for key, value in expected.items()),
        "preflight scientific bindings differ",
    )
    require(
        report.get("kind") == KIND
        and report.get("task") == TASK
        and report.get("passed") is True
        and report.get("exit_code") == 0
        and report.get("world_size") == 2,
        "a passed real two-H100 preflight report is required",
    )
    for key in (
        "g0_passed",
        "pilot_passed",
        "factorial_ready",
        "execution_class_certified",
        "uses_blackwell_certificate",
    ):
        require(report.get(key) is False, "preflight must not claim G0/Blackwell certification")
    require(re.fullmatch(r"[1-9][0-9]*", str(report.get("job_id", ""))), "preflight job ID is missing")
    require(
        re.fullmatch(r"[a-f0-9]{64}", report.get("code_sha256", "")), "preflight code identity is missing"
    )
    ranks = report.get("ranks")
    require(isinstance(ranks, list) and len(ranks) == 2, "both rank reports are required")
    visibility = None
    checkpoint_files = None
    for rank, item in enumerate(ranks):
        require(item.get("passed") is True, "one preflight rank did not pass")
        require(
            all(
                item.get(key) == report[key]
                for key in ("kind", "task", "run_id", "job_id", "code_sha256", *binding_keys)
            ),
            "preflight rank identity differs",
        )
        require(
            all(
                item.get(key) is False
                for key in (
                    "g0_passed",
                    "pilot_passed",
                    "factorial_ready",
                    "execution_class_certified",
                    "uses_blackwell_certificate",
                )
            ),
            "rank report claims unsupported certification",
        )
        allocation = item["allocation"]
        require(
            allocation.get("rank") == allocation.get("local_rank") == rank
            and allocation.get("world_size") == 2
            and allocation.get("threads_per_rank") == 12
            and allocation.get("job_id") == report["job_id"],
            "preflight rank allocation differs",
        )
        visible = allocation.get("cuda_visible_devices", "")
        devices = visible.split(",")
        require(
            len(devices) == len(set(devices)) == 2
            and all(device and device.strip() == device for device in devices)
            and visibility in (None, visible),
            "preflight CUDA visibility differs across ranks",
        )
        visibility = visible
        runtime = item["runtime"]
        require(
            runtime.get("python") == "3.12.13" and runtime.get("packages") == guards.DEPENDENCIES,
            "preflight runtime does not match the pinned training runtime",
        )
        for model, revision in guards.MODELS.items():
            require(item["pinned_cache"][model]["revision"] == revision, "preflight model revision differs")
        gpu = item["gpu"]
        require(
            "H100" in gpu.get("name", "")
            and gpu.get("compute_capability") == [9, 0]
            and gpu.get("total_memory_bytes", 0) >= 75 * 1024**3
            and gpu.get("logical_device") == rank,
            "preflight did not observe the required H100 devices",
        )
        require(
            item.get("nccl_all_reduce") is True
            and item.get("finite_gradients") is True
            and item.get("parameter_update_nonzero") is True
            and item.get("full_state_optimizer_scheduler_rng_restore") is True,
            "preflight collective/update/full-state restore evidence is missing",
        )
        for name in ("local_parameter_sha256_before", "local_parameter_sha256_after"):
            require(
                re.fullmatch(r"[a-f0-9]{64}", item.get(name, "")), "preflight parameter digest is missing"
            )
        require(
            item["local_parameter_sha256_before"] != item["local_parameter_sha256_after"],
            "preflight parameter update was zero",
        )
        losses = item.get("losses", [])
        require(
            len(losses) == 8 and all(math.isfinite(value) for value in losses), "invalid preflight losses"
        )
        require(
            item.get("batch_contract") == expected_contract
            and item.get("local_global_slots") == list(range(rank, 64, 2))
            and item.get("reserved_global_nonpadding_tokens") == 64 * 1536,
            "preflight batch/token/window semantics differ",
        )
        if expected.get("student_protocol_path") in {DEVICE_PROTOCOL_PATH, OPTIMIZER_PROTOCOL_PATH}:
            require(
                item.get("supervision_boundary") == supervision_boundary_evidence(f"cuda:{rank}"),
                "v3 preflight did not exercise CPU collation and canonical supervision on this rank",
            )
        if expected.get("student_protocol_path") == OPTIMIZER_PROTOCOL_PATH:
            require(
                item.get("optimizer_preparation") == optimizer_preparation_evidence(),
                "v4 preflight did not validate production FSDP optimizer preparation",
            )
        fsdp = item["fsdp"]
        require(
            fsdp.get("requested_fsdp_sharding_strategy") == "FULL_SHARD"
            and fsdp.get("effective_fsdp_sharding_strategy") == "FULL_SHARD"
            and fsdp.get("fsdp_wrapper_count") == 29,
            "preflight FSDP wrapper evidence differs",
        )
        require(
            item.get("rank_zero_teacher") == teacher_probe_evidence(expected["teacher_identity"]),
            "accepted dense teacher forward did not pass",
        )
        checkpoint = item["checkpoint"]
        files = checkpoint.get("files", [])
        require(
            checkpoint.get("world_size") == 2
            and checkpoint.get("all_files_verified_before_restore") is True
            and checkpoint.get("local_restored_parameter_sha256") == item["local_parameter_sha256_after"]
            and re.fullmatch(r"[a-f0-9]{64}", checkpoint.get("local_restored_optimizer_sha256", ""))
            and [record["path"] for record in files] == guards.checkpoint_names(2)
            and (checkpoint_files is None or checkpoint_files == files),
            "preflight checkpoint inventory/restoration differs",
        )
        for record in files:
            require(
                type(record.get("size")) is int
                and record["size"] > 0
                and re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", "")),
                "preflight checkpoint file identity is missing",
            )
        checkpoint_files = files
        require(
            item["node_local_mount"]["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"},
            "preflight scratch is not local",
        )
        for key in ("initial_cgroup_memory", "cgroup_memory"):
            validate_memory(item[key])
        allocated, reserved = item["peak_gpu_allocated_bytes"], item["peak_gpu_reserved_bytes"]
        require(0 < allocated <= reserved < gpu["total_memory_bytes"], "preflight GPU memory has no headroom")
    require(ranks[0].get("node") and ranks[0]["node"] == ranks[1].get("node"), "preflight was not one node")
    return {"job_id": report["job_id"], "run_id": report["run_id"], "world_size": 2, "report_validated": True}


def supervision_boundary_evidence(device):
    return {
        "scope": "synthetic_production_shape_only",
        "collator": "collate_trajectories",
        "supervisor": "CanonicalSFTSupervisor",
        "original_batch_device": "cpu",
        "loss_operands_device": device,
        "loss_logits_dtype": "float32",
        "loss_normalization": "sequence",
        "response_tokens_per_sequence": 256,
        "microsteps": 8,
        "original_batch_preserved": True,
    }


def canonical_canary_batch(rows, slots, pad_token_id):
    """Retain the original synthetic tokens while exercising production collation."""
    import torch

    from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
    from posttrain_circuits.learning.contracts import TrajectoryBatch
    from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor

    require(len(rows) == len(slots) == 4 and all(len(row) == 1536 for row in rows), "canary shape differs")
    records = []
    for row, slot in zip(rows, slots, strict=True):
        record = TrajectoryRecord(
            trajectory_id="",
            prompt_id=f"synthetic-canary-slot-{slot}",
            split="train",
            prompt_text="synthetic execution canary, not teacher-quality evidence",
            input_ids=list(row[:-256]),
            response_ids=list(row[-256:]),
            response_text="synthetic",
            response_token_mask=[True] * 256,
            behavior_policy_id="synthetic-canary",
            behavior_policy_revision="unchanged-global64-canary-v1",
            policy_version=0,
            sampling_request_seed=42,
            actual_sampling_seed=42 + slot,
            sampling_cursor_id=f"synthetic-canary-slot-{slot}",
            sampling_protocol_id="synthetic-no-sampling-v1",
            sampling_temperature=1.0,
            top_p=1.0,
            behavior_logprobs=[0.0] * 256,
            verifier_reward=1.0,
        )
        record.trajectory_id = record.expected_trajectory_id
        records.append(record)
    supervisor = CanonicalSFTSupervisor(pad_token_id, normalization="sequence")
    batch = supervisor.prepare_targets(TrajectoryBatch(records, 0), None, None)
    require(
        all(
            getattr(batch, key).device.type == "cpu"
            for key in ("input_ids", "attention_mask", "response_mask", "rewards")
        )
        and batch.input_ids.tolist() == rows
        and bool(batch.attention_mask.all())
        and not bool(batch.response_mask[:, :-256].any())
        and bool(batch.response_mask[:, -256:].all())
        and torch.equal(batch.rewards, torch.ones(4)),
        "CPU collator changed the fixed canary tokens, masks or rewards",
    )
    return supervisor, batch


def canonical_canary_loss(student, supervisor, batch):
    """Keep the original FP32 loss policy around the real supervisor boundary."""
    import torch

    observed = []
    original = {
        key: getattr(batch, key).clone()
        for key in ("input_ids", "attention_mask", "response_mask", "rewards")
    }

    def forward(**kwargs):
        require(
            all(value.device.type == "cpu" for value in kwargs.values()), "canary must enter forward on CPU"
        )
        logits = student(**kwargs).logits
        require(bool(torch.isfinite(logits).all()), "student forward is not finite")
        observed.append(str(logits.device))
        return SimpleNamespace(logits=logits.float())

    output = supervisor.compute_loss(forward, batch)
    require(len(observed) == 1, "canonical supervisor must forward exactly once")
    require(
        all(
            getattr(batch, key).device.type == "cpu"
            and getattr(batch, key).dtype == value.dtype
            and torch.equal(getattr(batch, key), value)
            for key, value in original.items()
        ),
        "canonical supervisor mutated the original CPU batch",
    )
    return output.loss, observed[0]


class GlooControlPlane:
    """Keep CPU/error/publication collectives on Gloo with production NCCL FSDP."""

    def __init__(self, distributed, group):
        self.distributed = distributed
        self.group = group

    def __getattr__(self, name):
        return getattr(self.distributed, name)

    def all_reduce(self, tensor, **kwargs):
        return self.distributed.all_reduce(tensor, **{"group": self.group, **kwargs})

    def all_gather_object(self, output, value, **kwargs):
        if kwargs.get("group") is None:
            kwargs["group"] = self.group
        return self.distributed.all_gather_object(output, value, **kwargs)

    def broadcast_object_list(self, values, **kwargs):
        return self.distributed.broadcast_object_list(values, **{"group": self.group, **kwargs})

    def monitored_barrier(self, **kwargs):
        return self.distributed.monitored_barrier(**{"group": self.group, **kwargs})


def production_accelerator(science_root):
    """Resolve the real training launcher YAML/defaults without launching a process."""
    from accelerate import Accelerator
    from accelerate.commands.launch import _validate_launch_command, launch_command_parser
    from accelerate.utils import FullyShardedDataParallelPlugin
    from accelerate.utils.launch import prepare_multi_gpu_env

    path = science_root / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml"
    arguments = launch_command_parser().parse_args(
        ["--config_file", str(path), "--num_cpu_threads_per_process", "12",
         "--main_process_port", "0", "-m", "posttrain_circuits.cli.train"]
    )
    arguments, _, _ = _validate_launch_command(arguments)
    generated = prepare_multi_gpu_env(arguments)
    require(arguments.use_fsdp and arguments.num_processes == 2, "production launcher topology differs")
    require(
        generated.get("FSDP_VERSION") == "1"
        and generated.get("FSDP_USE_ORIG_PARAMS") == "false"
        and generated.get("FSDP_SHARDING_STRATEGY") == "FULL_SHARD"
        and generated.get("FSDP_AUTO_WRAP_POLICY") == "TRANSFORMER_BASED_WRAP"
        and generated.get("FSDP_SYNC_MODULE_STATES") == "true"
        and generated.get("FSDP_CPU_RAM_EFFICIENT_LOADING") == "true"
        and generated.get("ACCELERATE_MIXED_PRECISION") == "bf16",
        "production FSDP launcher defaults differ",
    )
    # These are the same generated framework options used by the training child.
    # Do not replace CUDA visibility, rendezvous or any scheduler allocation.
    os.environ.update({k: v for k, v in generated.items() if k.startswith(("FSDP_", "ACCELERATE_"))})
    return Accelerator(
        fsdp_plugin=FullyShardedDataParallelPlugin(),
        gradient_accumulation_steps=8,
        step_scheduler_with_optimizer=False,
    )


def optimizer_preparation_evidence():
    return {
        "helper": "prepare_accelerate_model_optimizer_scheduler",
        "framework": "Accelerate_FSDP1",
        "use_orig_params": False,
        "prepared_parameter_ownership_verified": True,
        "raw_scheduler_optimizer_binding_verified": True,
        "optimizer_state_nonempty": True,
        "optimizer_master_dtype": "float32",
        "global_step": 1,
        "scheduler_last_epoch": 1,
        "scheduler_step_count": 2,
        "scope": "shared_production_preparation_then_one_synthetic_global64_window",
    }


def verify_optimizer_preparation(student, optimizer, scheduler):
    import torch

    from posttrain_circuits.learning.training.factorial_trainer import (
        _validate_optimizer_scheduler_cadence,
        validate_prepared_optimizer_binding,
    )

    validate_prepared_optimizer_binding(student, optimizer, scheduler)
    _validate_optimizer_scheduler_cadence(
        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(), global_step=1
    )
    require(
        all(p.dtype == torch.float32 for p in student.parameters())
        and all(p in optimizer.state for p in student.parameters() if p.requires_grad),
        "production prepared parameters lack FP32 AdamW state",
    )
    return optimizer_preparation_evidence()


def run_canary(args, identity, evidence):
    activate_scientific_source(args.science_root)
    import torch
    import torch.distributed as dist
    import yaml
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from posttrain_circuits.artifacts.adapted_student_protocol import resolve_adapted_student_protocol
    from posttrain_circuits.learning.supervision.losses import verified_replay_loss
    from posttrain_circuits.learning.training.execution_safety_kernel import batch_token_contract
    from posttrain_circuits.learning.training.fsdp_contract import (
        full_state_dict_options,
        validate_model_fsdp_sharding,
    )
    from posttrain_circuits.models.adapted_teacher import load_adapted_teacher
    from posttrain_circuits.models.loading import assert_tokenizer_compatible, load_model_and_tokenizer
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    protocol = resolve_adapted_student_protocol(args.science_root, expected_head=args.science_git_head)
    require(protocol.protocol_sha256 == args.student_protocol_sha256, "accepted student protocol differs")
    require(
        protocol.amendment_id
        in {
            "qwen3-adapted-student-calibration-v1",
            "qwen3-adapted-student-calibration-v2",
            "qwen3-adapted-student-calibration-v3",
            "qwen3-adapted-student-calibration-v4",
        },
        "unrecognized student canary protocol",
    )
    optimizer_boundary = protocol.amendment_id == "qwen3-adapted-student-calibration-v4"
    device_boundary = optimizer_boundary or protocol.amendment_id == "qwen3-adapted-student-calibration-v3"
    if device_boundary:
        expected_path = OPTIMIZER_PROTOCOL_PATH if optimizer_boundary else DEVICE_PROTOCOL_PATH
        require(
            protocol.path.relative_to(args.science_root).as_posix() == expected_path,
            "wrong student preflight protocol path",
        )
        evidence["student_protocol_path"] = expected_path
    rank, world = identity["rank"], identity["world_size"]
    log_phase(rank, "gpu_and_distributed_checks")
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == world, "visible CUDA device count differs"
    )
    require(torch.version.cuda == "12.8", "Torch CUDA runtime differs from 12.8")
    require(tuple(torch.cuda.nccl.version()) == (2, 27, 3), "NCCL runtime differs from 2.27.3")
    properties = [torch.cuda.get_device_properties(index) for index in range(world)]
    require(
        all(
            "H100" in item.name and (item.major, item.minor) == (9, 0) and item.total_memory >= 75 * 1024**3
            for item in properties
        ),
        "two H100 80GB GPUs required",
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.set_num_threads(12)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(device)
    if optimizer_boundary:
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        data_group = dist.group.WORLD
        control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=600))
        dist = GlooControlPlane(dist, control_group)
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=600))
        data_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=120))
    value = torch.tensor(float(rank + 1), device=device)
    work = dist.all_reduce(value, group=data_group, async_op=True)
    work.wait(timeout=timedelta(seconds=120))
    require(value.item() == 3.0, "first NCCL all-reduce returned incorrect value")
    evidence.update(
        nccl_all_reduce=True,
        gpu={
            "name": properties[rank].name,
            "total_memory_bytes": properties[rank].total_memory,
            "compute_capability": [9, 0],
            "logical_device": rank,
        },
    )
    model_config = yaml.safe_load((args.science_root / "configs/model/qwen3_v2_1p7b.yaml").read_text())
    teacher_config = yaml.safe_load(
        (args.science_root / "configs/teacher/qwen3_v2_teacher_8b.yaml").read_text()
    )
    for config, expected in ((model_config, "Qwen/Qwen3-1.7B"), (teacher_config, "Qwen/Qwen3-8B")):
        require(
            config["model_name_or_path"] == expected
            and config["model_revision"] == MODELS[expected]
            and config["tokenizer_name_or_path"] == expected
            and config["tokenizer_revision"] == MODELS[expected]
            and config["torch_dtype"] == "bfloat16"
            and config["trust_remote_code"] is False,
            "model configuration differs from the fixed offline pins",
        )
    log_phase(rank, "student_load")
    student_bundle = collective_phase(dist, lambda: load_model_and_tokenizer(model_config, for_training=True))
    model, tokenizer = student_bundle.model, student_bundle.tokenizer
    require(student_bundle.resolved_model_commit == MODELS["Qwen/Qwen3-1.7B"], "student revision mismatch")
    layer_types = {type(module) for module in model.modules() if type(module).__name__ == "Qwen3DecoderLayer"}
    require(len(layer_types) == 1, "student decoder layer type differs")
    if optimizer_boundary:
        from posttrain_circuits.learning.training.factorial_trainer import (
            prepare_accelerate_model_optimizer_scheduler,
        )

        accelerator = production_accelerator(args.science_root)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        student, prepared_optimizer, scheduler = prepare_accelerate_model_optimizer_scheduler(
            accelerator, model, optimizer, scheduler
        )
        optimizer = prepared_optimizer.optimizer
    else:
        student = FSDP(
            model,
            process_group=data_group,
            device_id=device,
            use_orig_params=False,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=layer_types
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
            ),
        )
    evidence["fsdp"] = validate_model_fsdp_sharding(student, world_size=world, fsdp_type=FSDP)
    require(evidence["fsdp"]["fsdp_wrapper_count"] == 29, "student must contain 28 wrapped decoder layers")
    teacher = None

    def teacher_probe():
        nonlocal teacher
        if rank != 0:
            return None
        bundle = load_adapted_teacher(
            args.teacher_checkpoint_root,
            args.teacher_checkpoint_sha256,
            teacher_config,
            device=device,
        )
        validate_loaded_teacher(bundle, evidence["teacher_identity"])
        assert_tokenizer_compatible(tokenizer, bundle.tokenizer)
        teacher = bundle.model
        prompt = format_model_prompt(
            "FACTS F01: A\nRULES R01: A -> B\nQUERY: B", bundle.tokenizer, teacher_config
        ).model_facing_prompt
        inputs = bundle.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            output = teacher(**inputs).logits
        require(bool(torch.isfinite(output).all()), "adapted teacher forward is not finite")
        return teacher_probe_evidence(evidence["teacher_identity"])

    teacher_result = collective_phase(dist, teacher_probe)
    broadcast = [teacher_result]
    dist.broadcast_object_list(broadcast, src=0)
    evidence["rank_zero_teacher"] = broadcast[0]
    contract = batch_token_contract(world, 256)
    evidence["batch_contract"] = contract
    slots = list(range(rank, 64, world))
    require(
        contract["samples_by_rank"][rank] == len(slots)
        and contract["microbatch_schedule_by_rank"][rank] == [4] * 8,
        "global batch kernel changed",
    )
    rows = canary_rows(tokenizer, slots, format_model_prompt, model_config)
    local_tokens = torch.tensor(len(rows) * 1536, dtype=torch.int64)
    dist.all_reduce(local_tokens)
    require(local_tokens.item() == 64 * 1536 <= contract["token_budget"], "global token reservation differs")
    evidence["reserved_global_nonpadding_tokens"] = local_tokens.item()
    if not optimizer_boundary:
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    before = tensor_digest(torch, student.named_parameters())
    losses = []
    log_phase(rank, "global64_optimizer_window")
    for microstep in range(8):
        if device_boundary:
            supervisor, supervision = canonical_canary_batch(
                rows[microstep * 4 : (microstep + 1) * 4],
                slots[microstep * 4 : (microstep + 1) * 4],
                tokenizer.pad_token_id,
            )
        else:
            input_ids = torch.tensor(rows[microstep * 4 : (microstep + 1) * 4], device=device)
            response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            response_mask[:, -256:] = True
        with contextlib.nullcontext() if microstep == 7 else student.no_sync():
            if device_boundary:
                loss, output_device = canonical_canary_loss(student, supervisor, supervision)
                require(output_device == str(device), "canonical loss used another rank's device")
            else:
                logits = student(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
                require(bool(torch.isfinite(logits).all()), "student forward is not finite")
                loss = verified_replay_loss(
                    logits.float(),
                    input_ids,
                    response_mask,
                    torch.ones(4, device=device),
                    normalization="sequence",
                )
            require(bool(torch.isfinite(loss)), "response-masked loss is not finite")
            losses.append(float(loss.detach()))
            (loss * (world * 4 / 64)).backward()
        if device_boundary:
            del loss, supervision, supervisor
        else:
            del logits, loss, input_ids, response_mask
    if device_boundary:
        evidence["supervision_boundary"] = supervision_boundary_evidence(str(device))
    gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
    require(
        gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
        "nonfinite/empty gradients",
    )
    if optimizer_boundary:
        prepared_optimizer.step()
        require(not prepared_optimizer.step_was_skipped, "production optimizer step was skipped")
    else:
        optimizer.step()
    scheduler.step()
    if optimizer_boundary:
        evidence["optimizer_preparation"] = verify_optimizer_preparation(student, optimizer, scheduler)
    after = tensor_digest(torch, student.named_parameters())
    require(before != after, "full-parameter AdamW update was zero")
    evidence.update(
        losses=losses,
        finite_gradients=True,
        parameter_update_nonzero=True,
        local_global_slots=slots,
        local_parameter_sha256_before=before,
        local_parameter_sha256_after=after,
    )
    saved_optim = optimizer_digest(torch, optimizer)
    runtime = {
        "scheduler": scheduler.state_dict(),
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(device).clone(),
    }
    checkpoint = args.output_dir / "checkpoint"
    log_phase(rank, "full_state_save_and_restore")
    collective_phase(dist, lambda: checkpoint.mkdir(exist_ok=True))
    runtime_file = checkpoint / f"rank-{rank}-runtime.pt"
    collective_phase(dist, lambda: torch.save(runtime, runtime_file))
    options = full_state_dict_options(world)
    with FSDP.state_dict_type(
        student,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(**options),
        FullOptimStateDictConfig(**options),
    ):
        model_state = student.state_dict()
        optim_state = FSDP.optim_state_dict(student, optimizer)

    def save_full(model_payload, optimizer_payload):
        if rank == 0:
            torch.save(model_payload, checkpoint / "model-full.pt")
            torch.save(optimizer_payload, checkpoint / "optimizer-full.pt")
            publish_json(
                checkpoint / "manifest.json",
                {
                    "kind": "sdsc_h100_canary_checkpoint_v1",
                    "world_size": identity["world_size"],
                    "code_sha256": args.code_sha256,
                    "files": [checkpoint_file(checkpoint / name) for name in checkpoint_names(world)],
                },
            )

    collective_phase(dist, functools.partial(save_full, model_state, optim_state))
    del model_state, optim_state
    checkpoint_manifest = collective_phase(
        dist, lambda: verify_checkpoint(checkpoint, args.code_sha256, world)
    )
    with torch.no_grad():
        next(parameter for parameter in student.parameters() if parameter.numel()).add_(1)
        next(
            value
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and value.numel()
        ).add_(1)
    scheduler.step()
    torch.rand(1)
    torch.rand(1, device=device)
    state = collective_phase(
        dist, lambda: torch.load(checkpoint / "model-full.pt", map_location="cpu", weights_only=False)
    )
    with FSDP.state_dict_type(
        student, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    ):
        loaded = student.load_state_dict(state)
    del state
    full_optim = collective_phase(
        dist,
        lambda: torch.load(checkpoint / "optimizer-full.pt", map_location="cpu", weights_only=False)
        if rank == 0
        else None,
    )
    sharded = FSDP.scatter_full_optim_state_dict(full_optim, student, optim=optimizer, group=data_group)
    optimizer.load_state_dict(sharded)
    del full_optim, sharded
    restored = torch.load(runtime_file, map_location="cpu", weights_only=False)
    scheduler.load_state_dict(restored["scheduler"])
    torch.set_rng_state(restored["cpu_rng"])
    torch.cuda.set_rng_state(restored["cuda_rng"], device)
    require(
        not loaded.missing_keys
        and not loaded.unexpected_keys
        and tensor_digest(torch, student.named_parameters()) == after
        and optimizer_digest(torch, optimizer) == saved_optim
        and scheduler.state_dict() == runtime["scheduler"]
        and torch.equal(torch.get_rng_state(), runtime["cpu_rng"])
        and torch.equal(torch.cuda.get_rng_state(device), runtime["cuda_rng"]),
        "same-world checkpoint restore differs",
    )
    evidence["full_state_optimizer_scheduler_rng_restore"] = True
    evidence["checkpoint"] = {
        "directory": str(checkpoint),
        "world_size": world,
        "local_restored_parameter_sha256": after,
        "local_restored_optimizer_sha256": saved_optim,
        "large_files_require_caller_persistence": True,
        "files": checkpoint_manifest["files"],
        "all_files_verified_before_restore": True,
    }
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_reserved(device)
    require(0 < peak < properties[rank].total_memory, "GPU peak left no memory headroom")
    evidence.update(
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_gpu_reserved_bytes=peak,
        cgroup_memory=memory_envelope(),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"], "CUDA visibility changed"
    )
    del teacher
    log_phase(rank, "rank_complete")
    return dist


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "work-dir",
        "output-dir",
        "hf-home",
        "science-root",
        "teacher-checkpoint-root",
        "teacher-acceptance",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "run-id",
        "code-sha256",
        "science-git-head",
        "teacher-checkpoint-sha256",
        "teacher-acceptance-sha256",
        "student-protocol-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    identity = guards.allocation_identity(os.environ)
    mount = guards.local_paths(args.work_dir, args.output_dir)
    bindings = validate_inputs(args)
    runtime = guards.runtime_identity()
    snapshots = guards.pinned_cache(args.hf_home)
    memory = memory_envelope()
    os.environ.update(
        HF_HOME=str(args.hf_home),
        HF_HUB_CACHE=str(args.hf_home / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="12",
        MKL_NUM_THREADS="12",
        OPENBLAS_NUM_THREADS="12",
        NUMEXPR_NUM_THREADS="12",
        TOKENIZERS_PARALLELISM="false",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = args.output_dir / f"rank-{identity['rank']}.json"
    require(
        not rank_path.exists() and not (args.output_dir / "checkpoint").exists(),
        "preflight output already used",
    )
    evidence = {
        "kind": KIND,
        "task": TASK,
        "passed": False,
        "g0_passed": False,
        "pilot_passed": False,
        "factorial_ready": False,
        "execution_class_certified": False,
        "uses_blackwell_certificate": False,
        "scope": (
            "synthetic_one_global64_optimizer_window_and_same_world_state_restore_with_accepted_dense_teacher"
        ),
        "not_tested": [
            "new_teacher_quality_measurement",
            "student_calibration",
            "full_G0",
            "multistep_training_resume_equivalence",
            "other_world_sizes",
            "persistent_storage_publication",
            "full_dataset_host_memory",
        ],
        **bindings,
        "run_id": args.run_id,
        "job_id": identity["job_id"],
        "code_sha256": args.code_sha256,
        "allocation": identity,
        "node": socket.gethostname(),
        "runtime": runtime,
        "pinned_cache": snapshots,
        "node_local_mount": mount,
        "initial_cgroup_memory": memory,
    }
    started, distributed = time.monotonic(), None

    def interrupted(signum, _frame):
        raise InterruptedError(f"adapted preflight received signal {signum}")

    old_handlers = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        distributed = run_canary(args, identity, evidence)
        if "student_protocol_path" in evidence:
            bindings["student_protocol_path"] = evidence["student_protocol_path"]
        evidence["passed"] = True
        evidence["elapsed_seconds"] = time.monotonic() - started
        publish_json(rank_path, evidence)
        reports = [None] * WORLD_SIZE
        distributed.all_gather_object(reports, evidence)
        require(all(report["passed"] for report in reports), "one rank did not pass")
        require(
            all(all(report[key] == value for key, value in bindings.items()) for report in reports),
            "rank teacher or scientific bindings differ",
        )
        if identity["rank"] == 0:
            publish_json(
                args.output_dir / RESULT,
                {
                    "kind": KIND,
                    "task": TASK,
                    "passed": True,
                    "exit_code": 0,
                    "world_size": identity["world_size"],
                    "g0_passed": False,
                    "pilot_passed": False,
                    "factorial_ready": False,
                    "execution_class_certified": False,
                    "uses_blackwell_certificate": False,
                    **bindings,
                    "run_id": args.run_id,
                    "job_id": identity["job_id"],
                    "code_sha256": args.code_sha256,
                    "staged_hf_home": str(args.hf_home),
                    "ranks": reports,
                },
            )
        distributed.monitored_barrier(timeout=timedelta(seconds=120), wait_all_ranks=True)
    except BaseException as error:
        evidence.update(
            passed=False, error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started
        )
        failure = args.output_dir / f"rank-{identity['rank']}-failure.json"
        with contextlib.suppress(OSError):
            publish_json(failure, evidence)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        cleanup = distributed or sys.modules.get("torch.distributed")
        if cleanup is not None and cleanup.is_initialized():
            cleanup.destroy_process_group()
    print(json.dumps({"rank": identity["rank"], "passed": True, "g0_passed": False}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
