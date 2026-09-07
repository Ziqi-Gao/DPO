#!/usr/bin/python3
"""Foreground, scheduler-managed Qwen3-v2 G0 handler for protocol v2.

ServerScheduler owns the physical allocation.  This supervisor accepts only
adapter-held content descriptors, preserves CUDA visibility, runs the reviewed
scientific stages serially, and publishes one report plus a complete immutable
artifact tar.  Failed scheduler attempts never reuse a staging directory.
"""

from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK = "qwen3_v2_g0"
PROFILE = "qwen3-v2-g0-elastic"
PROJECT = "OPD"
REPORT_NAME = "g0.json"
BUNDLE_NAME = "g0_artifacts.tar"
COMPLETION_NAME = ".opd-scientific-completion.json"
COMPLETION_KIND = "scientific_attempt"
SOURCE_ROOT = Path("/home/del6500/projects/OPD")
SOURCE_PACKAGE_ROOT = SOURCE_ROOT / "src"
FSDP_CONFIG = SOURCE_ROOT / "configs" / "accelerate" / "fsdp_server_scheduler.yaml"
FSDP_CONFIG_SHA256 = "7fe87d579ed92c4cca91ab719a7cf267e5491ba3b520b4a05f28c1e442ffb46a"
MIB_REPOSITORY = Path("/scr/del6500/OPD/vendor/MIB-circuit-track-v1")
MIB_REVISION = "b759df34433c9e31043ba9e02908ce0bf20e894f"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEACHER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_FINGERPRINT = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
PREREGISTRATION_SHA256 = "8d6bdeab0b9302c8824c4709f556c6c41a896bd2cfce21e7794d131d176ba0a4"
AMENDMENT_RELATIVE_PATH = Path(
    "prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml"
)
CERTIFICATION_RELATIVE_PATH = Path(
    "prereg/execution_safety/qwen3_v2_elastic_training_v1.certification.yaml"
)
DESCRIPTOR_RELATIVE_PATH = Path(
    "prereg/execution_safety/qwen3_v2_elastic_training_v1.descriptor.json"
)
EXECUTION_CLASS_ID = "qwen3-v2-elastic-training-v1"
CERTIFICATION_ID = "qwen3-v2-elastic-training-v1-initial-certification"
FINGERPRINT_SCHEMA = "opd-execution-safety-fingerprint-v1"
HANDOFF_RELATIVE_PATH = "docs/refactor/current_handoff.md"
PROPOSED_REVIEW = {
    "status": "proposed",
    "reviewed_implementation_commit": None,
    "reviewer": None,
    "reviewed_at_utc": None,
    "rationale": None,
}
ALLOWED_REVIEW_PATHS = frozenset(
    {
        str(AMENDMENT_RELATIVE_PATH),
        str(CERTIFICATION_RELATIVE_PATH),
        HANDOFF_RELATIVE_PATH,
    }
)
ALLOWED_GPU_COUNTS = (1, 2, 3, 4)
CPU_CORE_COUNT = 24
THREADS_PER_RANK = {count: CPU_CORE_COUNT // count for count in ALLOWED_GPU_COUNTS}
BATCH_PARTITION_PROTOCOL = "allocation_neutral_exact_global_batch_v1"
NODE_MEMORY_BYTES = 192 * 1024**3
MINIMUM_HEADROOM_BYTES = max(32 * 1024**3, int(NODE_MEMORY_BYTES * 0.20))
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_PREREG_BYTES = 4 * 1024 * 1024
MAX_AMENDMENT_BYTES = 1024 * 1024
MAX_CERTIFICATION_BYTES = 4 * 1024 * 1024
MAX_DESCRIPTOR_BYTES = 4 * 1024 * 1024
MAX_SCIENCE_PROTOCOL_BYTES = 4 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
PROC_SELF_FD = re.compile(r"/proc/self/fd/(0|[1-9][0-9]*)\Z")
POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")
G0_WORKFLOW_ID = re.compile(r"qwen3-v2-g0-elastic-[0-9a-f]{32}\Z")
SCIENCE_PROTOCOL_PATH = re.compile(
    r"prereg/execution_science/[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\.yaml\Z"
)
SCIENCE_OVERRIDE_KEY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]*(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)*\Z"
)
SCIENCE_STORAGE_LOCATOR_PATHS = (
    ("output_root",),
    ("prereg_path",),
    ("protocol_amendment_path",),
    ("execution_science_protocol_path",),
    ("state_source", "store_path"),
    ("task", "dataset_family_path"),
    ("anti_shortcut", "report_path"),
    ("production_safety", "readiness_report"),
    ("production_safety", "probe_cohort_manifest"),
    ("production_safety", "initial_checkpoint_path"),
    ("experiment", "random_reward_calibration_path"),
)
SCIENCE_SCHEDULER_PROVENANCE_PATHS = (
    ("scheduler",),
    ("server_scheduler",),
    ("scheduler_g0",),
    ("scheduler_preflight",),
    ("scheduler_provenance",),
    ("execution_context",),
    ("job_id",),
    ("workflow_id",),
    ("plan_sha256",),
    ("unit_id",),
    ("attempt",),
)
SCIENCE_RESOURCE_KEYS = frozenset(
    {
        "cpu_core_count",
        "cpu_count",
        "cpu_cores",
        "cpu_thread_count",
        "cpu_threads",
        "cpus",
        "cuda",
        "cuda_device",
        "cuda_visible_devices",
        "device",
        "device_map",
        "device_uuid",
        "devices",
        "exclusive",
        "execution_profile",
        "global_rank",
        "gpu",
        "gpu_count",
        "gpu_count_policy",
        "gpu_identity",
        "gpu_id",
        "gpu_ids",
        "gpu_index",
        "gpu_indices",
        "gpu_memory",
        "gpu_memory_mib",
        "gpu_memory_utilization",
        "gpu_uuid",
        "gpus",
        "host_memory",
        "host_memory_mib",
        "local_rank",
        "memory",
        "memory_mib",
        "mkl_num_threads",
        "node_rank",
        "nproc_per_node",
        "num_cpus",
        "num_gpu",
        "num_gpus",
        "omp_num_threads",
        "pci_bus_id",
        "pci_bus_ids",
        "per_rank_threads",
        "physical_gpu",
        "physical_gpu_index",
        "physical_gpu_uuid",
        "rank",
        "resources",
        "supported_world_sizes",
        "thread_count",
        "threads",
        "utilization",
        "world_size",
        "world_sizes",
    }
)

# This pre-import copy is intentional.  It prevents an unreviewed project
# module from weakening the descriptor before that module becomes importable.
EXECUTION_SAFETY_IMPLEMENTATION_PATHS = (
    "configs/accelerate/fsdp_server_scheduler.yaml",
    "deployments/qwen3_v2_g0/dependency-lock.json",
    "deployments/qwen3_v2_g0/package-manifest.json",
    "deployments/qwen3_v2_gpu_preflight/dependency-lock.json",
    "deployments/qwen3_v2_gpu_preflight/package-manifest.json",
    "scripts/server_scheduler/opd-entrypoint",
    "scripts/server_scheduler/qwen3-v2-g0-handler.py",
    "scripts/server_scheduler/qwen3-v2-gpu-preflight-handler.py",
    "src/posttrain_circuits/artifacts/checkpoints.py",
    "src/posttrain_circuits/artifacts/execution_safety_certification.py",
    "src/posttrain_circuits/artifacts/execution_safe_io.py",
    "src/posttrain_circuits/artifacts/execution_science_protocol.py",
    "src/posttrain_circuits/artifacts/hashing.py",
    "src/posttrain_circuits/artifacts/protocol_amendments.py",
    "src/posttrain_circuits/cli/compare_distributed_resume.py",
    "src/posttrain_circuits/cli/factorial_run_validation.py",
    "src/posttrain_circuits/cli/finalize_g0.py",
    "src/posttrain_circuits/cli/finalize_pilot_training.py",
    "src/posttrain_circuits/cli/train.py",
    "src/posttrain_circuits/core/config.py",
    "src/posttrain_circuits/core/seeding.py",
    "src/posttrain_circuits/datasets/teacher_demos/contracts.py",
    "src/posttrain_circuits/datasets/teacher_demos/ledger.py",
    "src/posttrain_circuits/datasets/teacher_demos/store.py",
    "src/posttrain_circuits/datasets/teacher_demos/views.py",
    "src/posttrain_circuits/datasets/trajectories/contracts.py",
    "src/posttrain_circuits/learning/collation.py",
    "src/posttrain_circuits/learning/contracts.py",
    "src/posttrain_circuits/learning/primitives.py",
    "src/posttrain_circuits/learning/supervision/losses.py",
    "src/posttrain_circuits/learning/supervision/verified_replay.py",
    "src/posttrain_circuits/learning/teacher/demo_source.py",
    "src/posttrain_circuits/learning/training/canonical_sft.py",
    "src/posttrain_circuits/learning/training/evaluation.py",
    "src/posttrain_circuits/learning/training/execution_safety_kernel.py",
    "src/posttrain_circuits/learning/training/factorial_trainer.py",
    "src/posttrain_circuits/learning/training/factories.py",
    "src/posttrain_circuits/learning/training/fsdp_contract.py",
    "src/posttrain_circuits/learning/training/optimizer.py",
    "src/posttrain_circuits/learning/training/schedules.py",
    "src/posttrain_circuits/learning/training/token_budget.py",
    "src/posttrain_circuits/models/loading.py",
    "src/posttrain_circuits/models/prompt_protocol.py",
    "src/posttrain_circuits/scheduler_adapter/dispatch.py",
    "src/posttrain_circuits/scheduler_adapter/entrypoint.py",
    "src/posttrain_circuits/scheduler_adapter/environment.py",
    "src/posttrain_circuits/scheduler_adapter/manifest.py",
    "src/posttrain_circuits/scheduler_adapter/qwen3_v2_g0.py",
    "src/posttrain_circuits/scheduler_adapter/registry.py",
    "src/posttrain_circuits/scheduler_adapter/runtime.py",
    "src/posttrain_circuits/scheduler_adapter/secure_files.py",
    "src/posttrain_circuits/scheduler_adapter/strict_json.py",
)
EXPECTED_INPUT_NAMES = (
    "config_binding_sha256",
    "execution_config_sha256",
    "execution_safety_certification_sha256",
    "execution_safety_descriptor_sha256",
    "execution_science_protocol_sha256",
    "preregistration_sha256",
    "protocol_amendment_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
)
YAML_INPUT_NAMES = frozenset(
    {
        "execution_safety_certification_sha256",
        "execution_science_protocol_sha256",
        "preregistration_sha256",
        "protocol_amendment_sha256",
    }
)
JSON_INPUT_NAMES = frozenset(EXPECTED_INPUT_NAMES) - YAML_INPUT_NAMES
GATE_NAMES = (
    "allocation_contract",
    "artifact_bundle",
    "batch_token_invariants",
    "config_binding",
    "distributed_resume",
    "execution_safety_certification",
    "execution_science_protocol",
    "g0_semantic_decision",
    "memory_headroom",
    "offline_pinned_runtime",
    "protocol_amendment",
    "scientific_artifact_chain",
)
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
FIXED_ENVIRONMENT = {
    "HF_HOME": "/scr/del6500/OPD/cache/huggingface",
    "HF_HUB_CACHE": "/scr/del6500/OPD/cache/huggingface/hub",
    "HF_HUB_OFFLINE": "1",
    "MIB_REPOSITORY": str(MIB_REPOSITORY),
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,NET,COLL",
    "NCCL_P2P_DISABLE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "TMPDIR": "/scr/del6500/OPD/tmp",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
}
_PROCESS_PYCACHE_PREFIX: str | None = None
SCIENTIFIC_CLIS = {
    "audit_label_leakage": "posttrain_circuits.cli.audit_label_leakage",
    "build_probe_cohorts": "posttrain_circuits.cli.build_probe_cohorts",
    "build_rollout_bank": "posttrain_circuits.cli.build_rollout_bank",
    "build_splits": "posttrain_circuits.cli.build_splits",
    "build_teacher_demos": "posttrain_circuits.cli.build_teacher_demos",
    "compare_distributed_resume": "posttrain_circuits.cli.compare_distributed_resume",
    "discover_circuit": "posttrain_circuits.cli.discover_circuit",
    "evaluate_anti_shortcut": "posttrain_circuits.cli.evaluate_anti_shortcut",
    "evaluate_circuit": "posttrain_circuits.cli.evaluate_circuit",
    "evaluate_teacher_readiness": "posttrain_circuits.cli.evaluate_teacher_readiness",
    "export_initial_checkpoint": "posttrain_circuits.cli.export_initial_checkpoint",
    "finalize_g0": "posttrain_circuits.cli.finalize_g0",
    "score_probe_candidates": "posttrain_circuits.cli.score_probe_candidates",
    "score_teacher": "posttrain_circuits.cli.score_teacher",
    "train": "posttrain_circuits.cli.train",
}


class G0Error(RuntimeError):
    """Fail-closed invocation, runtime, or scientific validation error."""


@dataclass(frozen=True)
class ContentHandle:
    name: str
    sha256: str
    descriptor: int


@dataclass(frozen=True)
class Invocation:
    workflow_id: str
    plan_sha256: str
    unit_id: str
    run_id: str
    job_id: str
    attempt: int
    execution_profile: str
    gpu_count: int
    manifest_sha256: str
    allocation_sha256: str
    running_manifest_descriptor: int
    content_handles: tuple[ContentHandle, ...]
    output_descriptor: int

    @property
    def input_hashes(self) -> dict[str, str]:
        return {item.name: item.sha256 for item in self.content_handles}

    @property
    def execution(self) -> dict[str, Any]:
        return {
            "allocation_sha256": self.allocation_sha256,
            "attempt": self.attempt,
            "execution_profile": self.execution_profile,
            "job_id": self.job_id,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class BootstrapLineage:
    amendment_sha256: str
    amendment_git_commit: str
    reviewed_implementation_commit: str
    request_git_commit: str


@dataclass(frozen=True)
class BootstrapScienceProtocol:
    science_protocol_path: str
    science_protocol_sha256: str
    science_protocol_id: str
    science_protocol_git_commit: str
    science_protocol_reviewed_implementation_commit: str
    unit_id: str
    hydra_override_vector: tuple[str, ...]


@dataclass(frozen=True)
class Stage:
    name: str
    cli: str
    argv: tuple[str, ...]
    distributed: bool = False


class _SingleValue(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied exactly once")
        setattr(namespace, self.dest, values)


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise G0Error(f"value is not canonical JSON: {error}") from error


def _sha256_value(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _published_json_bytes(payload: object) -> bytes:
    return (_canonical_json(payload) + "\n").encode("utf-8")


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise G0Error(f"{name} is not a valid identifier")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise G0Error(f"{name} is not a lowercase SHA-256 digest")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _log_phase(phase: str, **details: object) -> None:
    print(_canonical_json({"event": phase, **details}), flush=True)


def _outer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for option in (
        "workflow-id",
        "plan-sha256",
        "unit-id",
        "run-id",
        "job-id",
        "attempt",
        "execution-profile",
        "gpu-count",
        "manifest-sha256",
        "allocation-sha256",
        "running-manifest-handle",
        "output-attempt-handle",
        "attempt-completion-name",
    ):
        parser.add_argument(
            f"--{option}",
            dest=option.replace("-", "_"),
            action=_SingleValue,
            required=True,
            default=None,
        )
    parser.add_argument(
        "--content-handle",
        action="append",
        nargs=4,
        metavar=("NAME", "KIND", "SHA256", "PROC_PATH"),
        required=True,
    )
    return parser


def _held_descriptor(value: object, *, name: str) -> int:
    if not isinstance(value, str):
        raise G0Error(f"{name} must be a descriptor path")
    match = PROC_SELF_FD.fullmatch(value)
    if match is None:
        raise G0Error(f"{name} must use canonical /proc/self/fd/N form")
    descriptor = int(match.group(1))
    if descriptor < 3:
        raise G0Error(f"{name} must not alias a standard stream")
    try:
        os.fstat(descriptor)
    except OSError as error:
        raise G0Error(f"{name} descriptor is not open") from error
    return descriptor


def _parse_outer(argv: Sequence[str] | None) -> Invocation:
    args = _outer_parser().parse_args(argv)
    workflow_id = _identifier(args.workflow_id, name="workflow_id")
    if G0_WORKFLOW_ID.fullmatch(workflow_id) is None:
        raise G0Error("workflow_id is not a fresh count-neutral G0 identity")
    if args.attempt_completion_name != COMPLETION_NAME:
        raise G0Error("attempt completion name differs from the fixed ABI")
    if args.execution_profile != PROFILE:
        raise G0Error("execution profile differs from the reviewed G0 profile")
    if (
        not isinstance(args.gpu_count, str)
        or POSITIVE_INTEGER.fullmatch(args.gpu_count) is None
        or int(args.gpu_count) not in ALLOWED_GPU_COUNTS
    ):
        raise G0Error("adapter GPU count must be one of the reviewed values 1..4")
    if not isinstance(args.attempt, str) or POSITIVE_INTEGER.fullmatch(args.attempt) is None:
        raise G0Error("attempt must be a canonical positive integer")
    handles: list[ContentHandle] = []
    for raw_name, kind, digest, proc_path in args.content_handle:
        name = _identifier(raw_name, name="content handle name")
        if kind != "file":
            raise G0Error(f"G0 input {name!r} must be a file")
        handles.append(
            ContentHandle(
                name=name,
                sha256=_sha256(digest, name=f"content handle {name!r}"),
                descriptor=_held_descriptor(proc_path, name=f"content handle {name!r}"),
            )
        )
    if tuple(item.name for item in handles) != EXPECTED_INPUT_NAMES:
        raise G0Error("content handles differ from the fixed G0 ABI")
    descriptors = tuple(item.descriptor for item in handles)
    if len(set(descriptors)) != len(descriptors):
        raise G0Error("content handles alias the same descriptor")
    running_manifest_descriptor = _held_descriptor(
        args.running_manifest_handle, name="running manifest handle"
    )
    output_descriptor = _held_descriptor(args.output_attempt_handle, name="output attempt handle")
    if (
        running_manifest_descriptor in descriptors
        or output_descriptor in (*descriptors, running_manifest_descriptor)
        or not stat.S_ISDIR(os.fstat(output_descriptor).st_mode)
    ):
        raise G0Error("output attempt handle is invalid or aliases an input")
    return Invocation(
        workflow_id=workflow_id,
        plan_sha256=_sha256(args.plan_sha256, name="plan_sha256"),
        unit_id=_identifier(args.unit_id, name="unit_id"),
        run_id=_sha256(args.run_id, name="run_id"),
        job_id=_identifier(args.job_id, name="job_id"),
        attempt=int(args.attempt),
        execution_profile=PROFILE,
        gpu_count=int(args.gpu_count),
        manifest_sha256=_sha256(args.manifest_sha256, name="manifest_sha256"),
        allocation_sha256=_sha256(args.allocation_sha256, name="allocation_sha256"),
        running_manifest_descriptor=running_manifest_descriptor,
        content_handles=tuple(handles),
        output_descriptor=output_descriptor,
    )


def _read_held_file(descriptor: int, *, context: str, max_bytes: int) -> bytes:
    before = os.fstat(descriptor)
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    if not stat.S_ISREG(before.st_mode) or flags & os.O_ACCMODE != os.O_RDONLY:
        raise G0Error(f"{context} must be a held read-only regular file")
    if before.st_size < 1 or before.st_size > max_bytes:
        raise G0Error(f"{context} has an invalid size")
    raw = os.pread(descriptor, before.st_size, 0)
    after = os.fstat(descriptor)
    identity = lambda row: (  # noqa: E731
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise G0Error(f"{context} changed while being read")
    return raw


def _validate_held_running_manifest(invocation: Invocation) -> None:
    """Independently bind the child to the exact ordered scheduler allocation."""

    raw = _read_held_file(
        invocation.running_manifest_descriptor,
        context="running manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if hashlib.sha256(raw).hexdigest() != invocation.manifest_sha256:
        raise G0Error("running manifest bytes differ from manifest_sha256")
    payload = _strict_json(raw, context="running manifest")
    required = {
        "allocation", "cpu_ids", "estimated_runtime_seconds", "execution_profile",
        "exit_code", "failure_reason", "gpu_indices", "gpu_pci_bus_ids",
        "gpu_uuids", "job_id", "numa_node", "parameters", "priority",
        "project", "requested_profile", "resources", "schema_version", "state",
        "stderr_log", "stdout_log", "submitted_at", "task", "updated_at",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise G0Error("running manifest fields differ from protocol-v2")
    parameters = payload.get("parameters")
    allocation = payload.get("allocation")
    if (
        payload.get("schema_version") != 2
        or payload.get("project") != "OPD"
        or payload.get("state") != "running"
        or payload.get("task") != TASK
        or payload.get("job_id") != invocation.job_id
        or payload.get("execution_profile") != invocation.execution_profile
        or payload.get("requested_profile") is not None
        or payload.get("resources") is not None
        or payload.get("exit_code") is not None
        or payload.get("failure_reason") is not None
        or parameters
        != {
            "workflow_id": invocation.workflow_id,
            "plan_sha256": invocation.plan_sha256,
            "unit_id": invocation.unit_id,
        }
        or not isinstance(allocation, dict)
        or allocation
        != {
            "cpu_cores": CPU_CORE_COUNT,
            "exclusive_gpu": True,
            "gpu_count": invocation.gpu_count,
            "gpu_memory_mib": 81920,
            "gpu_utilization_pct": 95,
            "memory_mib": 196608,
        }
    ):
        raise G0Error("running manifest identity or allocation differs from invocation")
    gpu_uuids = payload.get("gpu_uuids")
    gpu_indices = payload.get("gpu_indices")
    gpu_pci_bus_ids = payload.get("gpu_pci_bus_ids")
    cpu_ids = payload.get("cpu_ids")
    if (
        not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != invocation.gpu_count
        or len(set(gpu_uuids)) != len(gpu_uuids)
        or any(not isinstance(value, str) or not value or "," in value for value in gpu_uuids)
        or not isinstance(gpu_indices, list)
        or len(gpu_indices) != invocation.gpu_count
        or not isinstance(gpu_pci_bus_ids, list)
        or len(gpu_pci_bus_ids) != invocation.gpu_count
        or not isinstance(cpu_ids, list)
        or len(cpu_ids) != allocation.get("cpu_cores")
    ):
        raise G0Error("running manifest concrete allocation is incomplete")
    allocation_payload = {
        "allocation": allocation,
        "cpu_ids": cpu_ids,
        "gpu_indices": gpu_indices,
        "gpu_pci_bus_ids": gpu_pci_bus_ids,
        "gpu_uuids": gpu_uuids,
        "numa_node": payload.get("numa_node"),
    }
    if _sha256_value(allocation_payload) != invocation.allocation_sha256:
        raise G0Error("running manifest allocation digest differs from invocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.split(",") != gpu_uuids:
        raise G0Error(
            "CUDA_VISIBLE_DEVICES differs from ordered running-manifest GPU UUIDs"
        )


def _strict_json(raw: bytes, *, context: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise G0Error(f"{context} contains a non-finite number: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise G0Error(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G0Error(f"{context} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise G0Error(f"{context} must be a JSON object")
    return value


def _read_inputs(
    invocation: Invocation,
) -> tuple[dict[str, dict[str, Any]], bytes, bytes, dict[str, bytes]]:
    payloads: dict[str, dict[str, Any]] = {}
    raw_inputs: dict[str, bytes] = {}
    prereg = b""
    amendment = b""
    inodes: set[tuple[int, int]] = set()
    for handle in invocation.content_handles:
        metadata = os.fstat(handle.descriptor)
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in inodes:
            raise G0Error("content handles alias the same inode")
        inodes.add(inode)
        raw = _read_held_file(
            handle.descriptor,
            context=f"content {handle.name!r}",
            max_bytes=(
                MAX_PREREG_BYTES
                if handle.name == "preregistration_sha256"
                else MAX_AMENDMENT_BYTES
                if handle.name == "protocol_amendment_sha256"
                else MAX_SCIENCE_PROTOCOL_BYTES
                if handle.name == "execution_science_protocol_sha256"
                else MAX_CERTIFICATION_BYTES
                if handle.name == "execution_safety_certification_sha256"
                else MAX_DESCRIPTOR_BYTES
                if handle.name == "execution_safety_descriptor_sha256"
                else MAX_INPUT_BYTES
            ),
        )
        if hashlib.sha256(raw).hexdigest() != handle.sha256:
            raise G0Error(f"content {handle.name!r} differs from its digest")
        raw_inputs[handle.name] = raw
        if handle.name == "preregistration_sha256":
            prereg = raw
            continue
        if handle.name == "protocol_amendment_sha256":
            amendment = raw
            continue
        if handle.name in {
            "execution_safety_certification_sha256",
            "execution_science_protocol_sha256",
        }:
            # Reviewed YAML artifacts are parsed only by the pre-import
            # bootstrap and then again by shared project validators.
            continue
        payload = _strict_json(raw, context=f"content {handle.name!r}")
        if handle.name in JSON_INPUT_NAMES and raw != _canonical_json(payload).encode("utf-8"):
            # Config CAS uses canonical bytes without a trailing newline.  The
            # descriptor is a separately schema-validated reviewed artifact,
            # so its exact CAS digest (rather than whitespace) is authoritative.
            if handle.name != "execution_safety_descriptor_sha256":
                raise G0Error(f"content {handle.name!r} is not canonical JSON bytes")
        payloads[handle.name] = payload
    if not prereg:
        raise G0Error("preregistration content is empty")
    if not amendment:
        raise G0Error("protocol amendment content is empty")
    return payloads, prereg, amendment, raw_inputs


def _install_source_path() -> None:
    value = str(SOURCE_PACKAGE_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def _configure_bytecode_isolation() -> None:
    """Keep every project-source import away from source-tree bytecode caches."""

    global _PROCESS_PYCACHE_PREFIX
    if _PROCESS_PYCACHE_PREFIX is None:
        root = Path(FIXED_ENVIRONMENT["TMPDIR"])
        root.mkdir(mode=0o750, parents=True, exist_ok=True)
        _PROCESS_PYCACHE_PREFIX = tempfile.mkdtemp(
            prefix=f".qwen3-v2-g0-pycache-disabled-{os.getpid()}-",
            dir=root,
        )
        os.chmod(_PROCESS_PYCACHE_PREFIX, 0o700)
        atexit.register(_remove_empty_bytecode_cache, _PROCESS_PYCACHE_PREFIX)
    sys.dont_write_bytecode = True
    sys.pycache_prefix = _PROCESS_PYCACHE_PREFIX


def _remove_empty_bytecode_cache(path: str) -> None:
    try:
        os.rmdir(path)
    except FileNotFoundError:
        pass
    except OSError:
        # Never recursively remove unexpected evidence.
        pass


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _git_process(*arguments: str, allowed_returncodes: frozenset[int] = frozenset({0})) -> Any:
    try:
        result = subprocess.run(
            ("/usr/bin/git", "-c", "core.fsmonitor=false", *arguments),
            cwd=SOURCE_ROOT,
            check=False,
            capture_output=True,
            env=_git_environment(),
        )
    except OSError as error:
        raise G0Error("trusted Git lineage validation could not execute") from error
    if result.returncode not in allowed_returncodes:
        raise G0Error("trusted Git lineage validation failed")
    return result


def _git_bytes(*arguments: str) -> bytes:
    return bytes(_git_process(*arguments).stdout)


def _unsafe_untracked_paths() -> tuple[str, ...]:
    """Find untracked paths without honoring any ignore or exclude source."""

    raw = _git_bytes("ls-files", "--others", "-z", "--")
    if not raw:
        return ()
    if not raw.endswith(b"\0"):
        raise G0Error("Git returned a malformed untracked-path list")
    try:
        paths = tuple(
            item.decode("utf-8", errors="strict") for item in raw[:-1].split(b"\0")
        )
    except UnicodeDecodeError as error:
        raise G0Error("Git returned a non-UTF-8 untracked path") from error
    if any(not path for path in paths):
        raise G0Error("Git returned an empty untracked path")
    unsafe: list[str] = []
    for path in paths:
        parsed = Path(path)
        if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
            raise G0Error("Git returned a non-canonical untracked path")
        if path == ".codex/config.toml":
            continue
        if parsed.suffix == ".pyc" and "__pycache__" in parsed.parts:
            continue
        unsafe.append(path)
    return tuple(unsafe)


def _mib_git_bytes(*arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ("/usr/bin/git", "-c", "core.fsmonitor=false", *arguments),
            cwd=MIB_REPOSITORY,
            check=False,
            capture_output=True,
            env=_git_environment(),
        )
    except OSError as error:
        raise G0Error("fixed MIB Git validation could not execute") from error
    if result.returncode != 0:
        raise G0Error("fixed MIB Git validation failed")
    return bytes(result.stdout)


def _git(*arguments: str) -> str:
    try:
        value = _git_bytes(*arguments).decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise G0Error("Git returned a non-ASCII commit identity") from error
    if GIT_COMMIT.fullmatch(value) is None:
        raise G0Error("Git did not return one immutable commit identity")
    return value


def _require_clean_git() -> str:
    if _git_bytes(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--ignore-submodules=none",
    ):
        raise G0Error("G0 requires a clean source checkout")
    return _git("rev-parse", "HEAD")


def _git_file_bytes(commit: str, relative_path: Path | str) -> bytes:
    if GIT_COMMIT.fullmatch(commit) is None:
        raise G0Error("Git file lookup commit is invalid")
    path = str(relative_path)
    parsed = Path(path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise G0Error("Git file lookup path is invalid")
    return _git_bytes("show", f"{commit}:{path}")


def _source_regular_bytes(path: Path, *, context: str, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise G0Error(f"{context} is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise G0Error(f"{context} must be one non-linked regular file")
        return _read_held_file(descriptor, context=context, max_bytes=max_bytes)
    finally:
        os.close(descriptor)


def _bootstrap_json(raw: bytes, *, context: str) -> dict[str, Any]:
    def mapping(pairs: list[tuple[object, object]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if not isinstance(key, str) or key in result:
                raise G0Error(f"{context} contains an invalid or duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=mapping,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise G0Error(f"{context} is not strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise G0Error(f"{context} must be a JSON mapping")
    return payload


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    if GIT_COMMIT.fullmatch(ancestor) is None or GIT_COMMIT.fullmatch(descendant) is None:
        raise G0Error("lineage ancestry endpoint is invalid")
    result = _git_process(
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        allowed_returncodes=frozenset({0, 1}),
    )
    return result.returncode == 0


def _commit_parents(commit: str) -> tuple[str, ...]:
    try:
        value = _git_bytes("show", "-s", "--format=%P", commit).decode(
            "ascii", errors="strict"
        ).strip()
    except UnicodeDecodeError as error:
        raise G0Error("Git returned non-ASCII commit parents") from error
    parents = tuple(value.split()) if value else ()
    if any(GIT_COMMIT.fullmatch(parent) is None for parent in parents):
        raise G0Error("Git returned an invalid parent identity")
    return parents


def _changed_paths(parent: str, commit: str) -> frozenset[str]:
    raw = _git_bytes(
        "diff",
        "--no-ext-diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=ACDMRTUXB",
        "-z",
        parent,
        commit,
        "--",
    )
    if raw and not raw.endswith(b"\0"):
        raise G0Error("Git returned a malformed changed-path list")
    try:
        return frozenset(
            item.decode("utf-8", errors="strict")
            for item in raw.rstrip(b"\0").split(b"\0")
            if item
        )
    except UnicodeDecodeError as error:
        raise G0Error("Git lineage contains a non-UTF-8 path") from error


def _bootstrap_yaml(raw: bytes, *, context: str) -> dict[str, Any]:
    # PyYAML is part of the fixed deployment, unlike any module below project src/.
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: UniqueKeyLoader,
        node: Any,
        deep: bool = False,
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in mapping:
                raise G0Error(f"{context} contains an invalid or duplicate key")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        payload = yaml.load(raw.decode("utf-8", errors="strict"), Loader=UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G0Error(f"{context} is not strict UTF-8 YAML") from error
    if not isinstance(payload, dict):
        raise G0Error(f"{context} must be a YAML mapping")
    return payload


def _accepted_review_metadata(accepted_raw: bytes) -> tuple[dict[str, Any], str]:
    accepted = _bootstrap_yaml(accepted_raw, context="accepted protocol amendment")
    review = accepted.get("review")
    if not isinstance(review, dict) or set(review) != set(PROPOSED_REVIEW):
        raise G0Error("accepted protocol amendment review fields differ")
    implementation = review.get("reviewed_implementation_commit")
    if review.get("status") != "accepted" or not isinstance(implementation, str):
        raise G0Error("protocol amendment is not accepted")
    if GIT_COMMIT.fullmatch(implementation) is None:
        raise G0Error("accepted protocol amendment names an invalid implementation commit")
    for field in ("reviewer", "rationale"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise G0Error(f"accepted protocol amendment lacks {field}")
    timestamp = review.get("reviewed_at_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise G0Error("accepted protocol amendment review timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise G0Error("accepted protocol amendment review timestamp is invalid") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise G0Error("accepted protocol amendment review timestamp is not UTC")
    return accepted, implementation


def _science_protocol_path(value: object) -> str:
    if not isinstance(value, str) or SCIENCE_PROTOCOL_PATH.fullmatch(value) is None:
        raise G0Error(
            "execution science protocol path must be a direct "
            "prereg/execution_science/*.yaml artifact"
        )
    parsed = Path(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise G0Error("execution science protocol path is not canonical")
    return value


def _remove_science_path(payload: dict[str, Any], parts: tuple[str, ...]) -> None:
    parent: object = payload
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            return
        parent = parent[part]
    if isinstance(parent, dict):
        parent.pop(parts[-1], None)


def _reject_science_resource_fields(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise G0Error("resolved science config contains a non-string key")
            lowered = key.lower().replace("-", "_")
            collapsed = re.sub(r"[^a-z0-9]", "", key.lower())
            forbidden_collapsed = {
                re.sub(r"[^a-z0-9]", "", candidate)
                for candidate in SCIENCE_RESOURCE_KEYS
            }
            if (
                lowered in SCIENCE_RESOURCE_KEYS
                or collapsed in forbidden_collapsed
                or lowered.startswith(
                    (
                        "physical_gpu",
                        "gpu_count",
                        "gpu_uuid",
                        "gpu_index",
                        "cuda_device",
                        "device_map",
                        "nproc_per_node",
                        "world_size",
                    )
                )
                or (
                    lowered.startswith("cpu_")
                    and any(token in lowered for token in ("core", "thread"))
                )
                or (
                    lowered.startswith(("host_memory", "gpu_memory"))
                    and any(token in lowered for token in ("mib", "gib", "bytes"))
                )
            ):
                raise G0Error(
                    "resolved science config contains forbidden resource steering "
                    f"field {'.'.join((*path, key))!r}"
                )
            _reject_science_resource_fields(item, path=(*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_science_resource_fields(item, path=(*path, str(index)))


def _bootstrap_science_config_sha256(resolved: dict[str, Any]) -> str:
    try:
        normalized = json.loads(_canonical_json(resolved))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise G0Error("resolved science config is not strict JSON data") from error
    if not isinstance(normalized, dict):
        raise G0Error("resolved science config must be a mapping")
    for parts in SCIENCE_STORAGE_LOCATOR_PATHS:
        _remove_science_path(normalized, parts)
    for parts in SCIENCE_SCHEDULER_PROVENANCE_PATHS:
        _remove_science_path(normalized, parts)
    _reject_science_resource_fields(normalized)
    return _sha256_value(normalized)


def _bootstrap_hydra_override_vector(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 128:
        raise G0Error("execution science Hydra override vector is invalid")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 1024
            or item != item.strip()
            or any(ord(character) < 32 for character in item)
            or "=" not in item
        ):
            raise G0Error(f"execution science Hydra override {index} is invalid")
        key, serialized = item.split("=", 1)
        components = tuple(part.lower().replace("-", "_") for part in key.split("."))
        if (
            SCIENCE_OVERRIDE_KEY.fullmatch(key) is None
            or not serialized
            or key in seen
            or any(
                component in SCIENCE_RESOURCE_KEYS
                or component.startswith(
                    ("scheduler", "server_scheduler", "gpu_", "physical_gpu")
                )
                for component in components
            )
        ):
            raise G0Error(
                f"execution science Hydra override {index} contains invalid or "
                "resource-steering syntax"
            )
        seen.add(key)
        result.append(item)
    return tuple(result)


def _accepted_science_protocol_metadata(
    accepted_raw: bytes,
) -> tuple[dict[str, Any], str]:
    accepted = _bootstrap_yaml(
        accepted_raw,
        context="accepted execution science protocol",
    )
    review = accepted.get("review")
    if not isinstance(review, dict) or set(review) != set(PROPOSED_REVIEW):
        raise G0Error("accepted execution science protocol review fields differ")
    implementation = review.get("reviewed_implementation_commit")
    if review.get("status") != "accepted" or not isinstance(implementation, str):
        raise G0Error("execution science protocol is not accepted")
    if GIT_COMMIT.fullmatch(implementation) is None:
        raise G0Error("execution science protocol names an invalid implementation commit")
    for field in ("reviewer", "rationale"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise G0Error(f"accepted execution science protocol lacks {field}")
    timestamp = review.get("reviewed_at_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise G0Error("execution science protocol review timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise G0Error("execution science protocol review timestamp is invalid") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise G0Error("execution science protocol review timestamp is not UTC")
    return accepted, implementation


def _bootstrap_accepted_lineage(
    *,
    code_commit: str,
    resolved: dict[str, Any],
    amendment: bytes,
    descriptor_raw: bytes,
    certification_raw: bytes,
) -> BootstrapLineage:
    """Authenticate the reviewed execution class before importing project code."""

    if _git("rev-parse", "HEAD") != code_commit:
        raise G0Error("trusted lineage HEAD changed before source import")
    source_amendment = _source_regular_bytes(
        SOURCE_ROOT / AMENDMENT_RELATIVE_PATH,
        context="source protocol amendment",
        max_bytes=MAX_AMENDMENT_BYTES,
    )
    source_descriptor = _source_regular_bytes(
        SOURCE_ROOT / DESCRIPTOR_RELATIVE_PATH,
        context="source execution-safety descriptor",
        max_bytes=MAX_DESCRIPTOR_BYTES,
    )
    source_certification = _source_regular_bytes(
        SOURCE_ROOT / CERTIFICATION_RELATIVE_PATH,
        context="source execution-safety certification",
        max_bytes=MAX_CERTIFICATION_BYTES,
    )
    if (
        source_amendment != amendment
        or source_descriptor != descriptor_raw
        or source_certification != certification_raw
        or _git_file_bytes(code_commit, AMENDMENT_RELATIVE_PATH) != source_amendment
        or _git_file_bytes(code_commit, DESCRIPTOR_RELATIVE_PATH) != source_descriptor
        or _git_file_bytes(code_commit, CERTIFICATION_RELATIVE_PATH)
        != source_certification
    ):
        raise G0Error("held execution-class artifacts differ from the clean checkout")

    accepted_amendment, implementation = _accepted_review_metadata(source_amendment)
    accepted_certification = _bootstrap_yaml(
        source_certification,
        context="accepted execution-safety certification",
    )
    certification_review = accepted_certification.get("review")
    if certification_review != accepted_amendment.get("review"):
        raise G0Error("amendment and certification review identities differ")
    proposed_amendment = _bootstrap_yaml(
        _git_file_bytes(implementation, AMENDMENT_RELATIVE_PATH),
        context="reviewed proposed amendment",
    )
    proposed_certification = _bootstrap_yaml(
        _git_file_bytes(implementation, CERTIFICATION_RELATIVE_PATH),
        context="reviewed proposed certification",
    )
    if (
        proposed_amendment.get("review") != PROPOSED_REVIEW
        or proposed_certification.get("review") != PROPOSED_REVIEW
    ):
        raise G0Error("reviewed implementation did not contain both proposals")
    normalized_amendment = dict(accepted_amendment)
    normalized_amendment["review"] = dict(PROPOSED_REVIEW)
    normalized_certification = dict(accepted_certification)
    normalized_certification["review"] = dict(PROPOSED_REVIEW)
    if (
        normalized_amendment != proposed_amendment
        or normalized_certification != proposed_certification
    ):
        raise G0Error("accepted execution-class review changed non-review content")

    amendment_commit = _git(
        "log",
        "-n1",
        "--format=%H",
        code_commit,
        "--",
        str(AMENDMENT_RELATIVE_PATH),
    )
    certification_commit = _git(
        "log",
        "-n1",
        "--format=%H",
        code_commit,
        "--",
        str(CERTIFICATION_RELATIVE_PATH),
    )
    acceptance_parents = _commit_parents(amendment_commit)
    acceptance_parent = acceptance_parents[0] if len(acceptance_parents) == 1 else ""
    changed_at_acceptance = (
        _changed_paths(acceptance_parent, amendment_commit)
        if acceptance_parent
        else frozenset()
    )
    additional_review_paths = changed_at_acceptance - ALLOWED_REVIEW_PATHS
    if (
        amendment_commit != certification_commit
        or not acceptance_parent
        or not _is_ancestor(implementation, acceptance_parent)
        or not {
            str(AMENDMENT_RELATIVE_PATH),
            str(CERTIFICATION_RELATIVE_PATH),
        }
        <= changed_at_acceptance
        or len(additional_review_paths) > 1
        or any(
            SCIENCE_PROTOCOL_PATH.fullmatch(path) is None
            for path in additional_review_paths
        )
        or _git_file_bytes(implementation, DESCRIPTOR_RELATIVE_PATH)
        != source_descriptor
    ):
        raise G0Error("execution-class lineage lacks one pure joint acceptance")

    scheduler = resolved.get("scheduler_g0")
    if not isinstance(scheduler, dict):
        raise G0Error("resolved config lacks scheduler G0 provenance")
    request_commit = scheduler.get("request_git_commit")
    for name in (
        "execution_safety_fingerprint_sha256",
        "execution_safety_descriptor_sha256",
        "execution_safety_certification_sha256",
    ):
        if not isinstance(scheduler.get(name), str) or SHA256.fullmatch(
            scheduler[name]
        ) is None:
            raise G0Error("resolved config contains invalid execution-safety hashes")
    if (
        not isinstance(request_commit, str)
        or GIT_COMMIT.fullmatch(request_commit) is None
        or scheduler.get("execution_safety_class_id") != EXECUTION_CLASS_ID
    ):
        raise G0Error("resolved config contains invalid execution-class identity")

    descriptor = _bootstrap_json(
        source_descriptor,
        context="execution-safety descriptor",
    )
    if set(descriptor) != {
        "execution_class_id",
        "fingerprint_schema",
        "fingerprint_sha256",
        "kind",
        "schema_version",
        "subject",
    }:
        raise G0Error("execution-safety descriptor fields differ")
    subject = descriptor.get("subject")
    if (
        descriptor.get("schema_version") != 1
        or descriptor.get("kind") != "qwen3_v2_execution_safety_descriptor"
        or descriptor.get("execution_class_id") != EXECUTION_CLASS_ID
        or descriptor.get("fingerprint_schema") != FINGERPRINT_SCHEMA
        or not isinstance(subject, dict)
        or descriptor.get("fingerprint_sha256") != _sha256_value(subject)
    ):
        raise G0Error("execution-safety descriptor identity is invalid")
    implementation_files = subject.get("implementation_files")
    if (
        not isinstance(implementation_files, dict)
        or set(implementation_files) != set(EXECUTION_SAFETY_IMPLEMENTATION_PATHS)
    ):
        raise G0Error("execution-safety descriptor critical file set differs")

    descriptor_sha256 = hashlib.sha256(source_descriptor).hexdigest()
    certification_sha256 = hashlib.sha256(source_certification).hexdigest()
    certification_core = dict(accepted_certification)
    certification_core.pop("review", None)
    certification_core_sha256 = _sha256_value(certification_core)
    certification_class = accepted_certification.get("execution_class")
    amendment_certification = accepted_amendment.get(
        "execution_safety_certification"
    )
    if (
        scheduler["execution_safety_descriptor_sha256"] != descriptor_sha256
        or scheduler["execution_safety_certification_sha256"]
        != certification_sha256
        or scheduler["execution_safety_fingerprint_sha256"]
        != descriptor["fingerprint_sha256"]
        or not isinstance(certification_class, dict)
        or certification_class.get("execution_class_id") != EXECUTION_CLASS_ID
        or certification_class.get("descriptor_sha256") != descriptor_sha256
        or certification_class.get("fingerprint_sha256")
        != descriptor["fingerprint_sha256"]
        or certification_class.get("supported_world_sizes") != [1, 2, 3, 4]
        or accepted_certification.get("certification_id") != CERTIFICATION_ID
        or not isinstance(amendment_certification, dict)
        or amendment_certification.get("execution_class_id") != EXECUTION_CLASS_ID
        or amendment_certification.get("certification_id") != CERTIFICATION_ID
        or amendment_certification.get("descriptor_path")
        != str(DESCRIPTOR_RELATIVE_PATH)
        or amendment_certification.get("descriptor_sha256") != descriptor_sha256
        or amendment_certification.get("certification_path")
        != str(CERTIFICATION_RELATIVE_PATH)
        or amendment_certification.get("certification_core_sha256")
        != certification_core_sha256
        or amendment_certification.get("fingerprint_sha256")
        != descriptor["fingerprint_sha256"]
        or amendment_certification.get("supported_world_sizes") != [1, 2, 3, 4]
    ):
        raise G0Error("execution-class artifacts do not bind one exact certification")

    for relative, expected_digest in implementation_files.items():
        if not isinstance(expected_digest, str) or SHA256.fullmatch(
            expected_digest
        ) is None:
            raise G0Error("execution-safety descriptor contains an invalid file hash")
        working_raw = _source_regular_bytes(
            SOURCE_ROOT / relative,
            context=f"execution-safety implementation {relative}",
            max_bytes=64 * 1024 * 1024,
        )
        if hashlib.sha256(working_raw).hexdigest() != expected_digest:
            raise G0Error(
                f"execution-safety critical file differs from certification: {relative}"
            )

    if (
        _require_clean_git() != code_commit
        or _git("rev-parse", "HEAD") != code_commit
        or _source_regular_bytes(
            SOURCE_ROOT / AMENDMENT_RELATIVE_PATH,
            context="source protocol amendment",
            max_bytes=MAX_AMENDMENT_BYTES,
        )
        != source_amendment
        or _source_regular_bytes(
            SOURCE_ROOT / DESCRIPTOR_RELATIVE_PATH,
            context="source execution-safety descriptor",
            max_bytes=MAX_DESCRIPTOR_BYTES,
        )
        != source_descriptor
        or _source_regular_bytes(
            SOURCE_ROOT / CERTIFICATION_RELATIVE_PATH,
            context="source execution-safety certification",
            max_bytes=MAX_CERTIFICATION_BYTES,
        )
        != source_certification
    ):
        raise G0Error("trusted execution-class lineage changed during bootstrap")
    return BootstrapLineage(
        amendment_sha256=hashlib.sha256(source_amendment).hexdigest(),
        amendment_git_commit=amendment_commit,
        reviewed_implementation_commit=implementation,
        request_git_commit=request_commit,
    )


def _bootstrap_science_protocol_lineage(
    *,
    code_commit: str,
    resolved: dict[str, Any],
    science_protocol_raw: bytes,
    unit_id: str,
) -> BootstrapScienceProtocol:
    """Authenticate one accepted per-experiment protocol before source import."""

    scheduler = resolved.get("scheduler_g0")
    if not isinstance(scheduler, dict):
        raise G0Error("resolved config lacks scheduler G0 provenance")
    configured_path = _science_protocol_path(
        scheduler.get("execution_science_protocol_path")
    )
    source_path = SOURCE_ROOT / configured_path
    source_raw = _source_regular_bytes(
        source_path,
        context="source execution science protocol",
        max_bytes=MAX_SCIENCE_PROTOCOL_BYTES,
    )
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    if (
        source_raw != science_protocol_raw
        or _git_file_bytes(code_commit, configured_path) != source_raw
        or scheduler.get("execution_science_protocol_sha256") != source_sha256
    ):
        raise G0Error(
            "execution science protocol CAS differs from the clean source commit"
        )

    accepted, implementation = _accepted_science_protocol_metadata(source_raw)
    proposed_raw = _git_file_bytes(implementation, configured_path)
    proposed = _bootstrap_yaml(
        proposed_raw,
        context="reviewed proposed execution science protocol",
    )
    if proposed.get("review") != PROPOSED_REVIEW:
        raise G0Error(
            "science protocol implementation did not contain a proposed artifact"
        )
    normalized = dict(accepted)
    normalized["review"] = dict(PROPOSED_REVIEW)
    if normalized != proposed:
        raise G0Error("accepted science protocol changed non-review terms")

    acceptance_commit = _git(
        "log",
        "-n1",
        "--format=%H",
        code_commit,
        "--",
        configured_path,
    )
    acceptance_parents = _commit_parents(acceptance_commit)
    acceptance_parent = acceptance_parents[0] if len(acceptance_parents) == 1 else ""
    changed_at_acceptance = (
        _changed_paths(acceptance_parent, acceptance_commit)
        if acceptance_parent
        else frozenset()
    )
    class_acceptance_commit = _git(
        "log",
        "-n1",
        "--format=%H",
        code_commit,
        "--",
        str(AMENDMENT_RELATIVE_PATH),
    )
    allowed_review_paths = frozenset({configured_path, HANDOFF_RELATIVE_PATH})
    if acceptance_commit == class_acceptance_commit:
        allowed_review_paths |= frozenset(
            {str(AMENDMENT_RELATIVE_PATH), str(CERTIFICATION_RELATIVE_PATH)}
        )
    if (
        not acceptance_parent
        or not _is_ancestor(implementation, acceptance_parent)
        or configured_path not in changed_at_acceptance
        or not changed_at_acceptance
        <= allowed_review_paths
        or _git_file_bytes(acceptance_commit, configured_path) != source_raw
    ):
        raise G0Error(
            "execution science protocol lacks one pure-review acceptance"
        )

    request_commit = scheduler.get("request_git_commit")
    if (
        not isinstance(request_commit, str)
        or GIT_COMMIT.fullmatch(request_commit) is None
    ):
        raise G0Error("execution science protocol request identity is invalid")

    expected_top = {
        "schema_version",
        "kind",
        "protocol_id",
        "scope",
        "execution_class",
        "science_config",
        "review_contract",
        "review",
    }
    scope = accepted.get("scope")
    execution_class = accepted.get("execution_class")
    science_config = accepted.get("science_config")
    protocol_id = accepted.get("protocol_id")
    if (
        set(accepted) != expected_top
        or accepted.get("schema_version") != 1
        or accepted.get("kind") != "opd_execution_science_protocol"
        or not isinstance(protocol_id, str)
        or IDENTIFIER.fullmatch(protocol_id) is None
        or not isinstance(scope, dict)
        or set(scope) != {"unit_id", "seed"}
        or scope.get("unit_id") != unit_id
        or type(scope.get("seed")) is not int
        or not 0 <= scope["seed"] < 2**63
        or resolved.get("seed") != scope["seed"]
        or not isinstance(execution_class, dict)
        or not isinstance(science_config, dict)
    ):
        raise G0Error("execution science protocol identity or scope is invalid")

    certification_raw = _source_regular_bytes(
        SOURCE_ROOT / CERTIFICATION_RELATIVE_PATH,
        context="source execution-safety certification",
        max_bytes=MAX_CERTIFICATION_BYTES,
    )
    certification = _bootstrap_yaml(
        certification_raw,
        context="accepted execution-safety certification",
    )
    certification_core = dict(certification)
    certification_core.pop("review", None)
    expected_execution_class = {
        "execution_class_id": EXECUTION_CLASS_ID,
        "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
        "descriptor_sha256": scheduler.get(
            "execution_safety_descriptor_sha256"
        ),
        "fingerprint_sha256": scheduler.get(
            "execution_safety_fingerprint_sha256"
        ),
        "certification_id": CERTIFICATION_ID,
        "certification_path": str(CERTIFICATION_RELATIVE_PATH),
        "certification_core_sha256": _sha256_value(certification_core),
        "supported_world_sizes": [1, 2, 3, 4],
    }
    if execution_class != expected_execution_class:
        raise G0Error(
            "execution science protocol does not bind the exact execution class"
        )

    override_vector = _bootstrap_hydra_override_vector(
        science_config.get("hydra_override_vector")
    )
    expected_science_sha256 = science_config.get(
        "storage_neutral_resolved_config_sha256"
    )
    if (
        not isinstance(expected_science_sha256, str)
        or SHA256.fullmatch(expected_science_sha256) is None
        or expected_science_sha256 != _bootstrap_science_config_sha256(resolved)
        or scheduler.get("execution_science_protocol_id") != protocol_id
        or scheduler.get("execution_science_protocol_git_commit")
        != acceptance_commit
        or scheduler.get("execution_science_protocol_reviewed_implementation_commit")
        != implementation
    ):
        raise G0Error("resolved science config differs from its accepted protocol")

    if (
        _require_clean_git() != code_commit
        or _git("rev-parse", "HEAD") != code_commit
        or _source_regular_bytes(
            source_path,
            context="source execution science protocol",
            max_bytes=MAX_SCIENCE_PROTOCOL_BYTES,
        )
        != source_raw
    ):
        raise G0Error("execution science protocol changed during bootstrap")
    return BootstrapScienceProtocol(
        science_protocol_path=configured_path,
        science_protocol_sha256=source_sha256,
        science_protocol_id=protocol_id,
        science_protocol_git_commit=acceptance_commit,
        science_protocol_reviewed_implementation_commit=implementation,
        unit_id=unit_id,
        hydra_override_vector=override_vector,
    )


def _validate_allocation_environment(
    gpu_count: int,
    environ: Mapping[str, str],
) -> tuple[str, ...]:
    if gpu_count not in ALLOWED_GPU_COUNTS:
        raise G0Error("adapter GPU count is outside the reviewed 1..4 capability")
    threads_per_rank = THREADS_PER_RANK[gpu_count]
    if any(environ.get(key) != str(threads_per_rank) for key in THREAD_KEYS):
        raise G0Error(
            f"per-rank CPU thread environment must be exactly {threads_per_rank}"
        )
    visible = environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = tuple(visible.split(",")) if visible else ()
    if len(devices) != gpu_count or len(set(devices)) != gpu_count or any(
        not value or value.strip() != value for value in devices
    ):
        raise G0Error(
            "scheduler-provided CUDA visibility differs from the adapter GPU count"
        )
    return devices


def _validate_environment(gpu_count: int) -> tuple[str, ...]:
    devices = _validate_allocation_environment(gpu_count, os.environ)
    for key, expected in FIXED_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise G0Error(f"fixed environment {key} differs from its deployment")
    forbidden = sorted(
        key
        for key in os.environ
        if key.startswith("SERVER_SCHEDULER_")
        or key in {"LD_AUDIT", "LD_LIBRARY_PATH", "LD_PRELOAD", "PATH", "PYTHONHOME", "PYTHONPATH"}
    )
    if forbidden:
        raise G0Error(f"ambient runtime injection reached G0 handler: {forbidden}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise G0Error("PYTHONNOUSERSITE=1 is required")
    if not (sys.flags.isolated and sys.flags.ignore_environment) or sys.flags.no_site:
        raise G0Error("G0 handler requires Python -I with fixed venv site packages")
    expected_prefix_start = (
        f"{FIXED_ENVIRONMENT['TMPDIR']}/.qwen3-v2-g0-pycache-disabled-{os.getpid()}-"
    )
    prefix_path = Path(_PROCESS_PYCACHE_PREFIX) if _PROCESS_PYCACHE_PREFIX else None
    prefix_metadata = prefix_path.lstat() if prefix_path is not None else None
    if (
        not sys.dont_write_bytecode
        or _PROCESS_PYCACHE_PREFIX is None
        or sys.pycache_prefix != _PROCESS_PYCACHE_PREFIX
        or not _PROCESS_PYCACHE_PREFIX.startswith(expected_prefix_start)
        or prefix_metadata is None
        or not stat.S_ISDIR(prefix_metadata.st_mode)
        or prefix_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(prefix_metadata.st_mode) != 0o700
        or bool(os.listdir(_PROCESS_PYCACHE_PREFIX))
    ):
        raise G0Error("G0 handler bytecode isolation is not active")
    expected_versions = {
        "accelerate": "1.10.1",
        "datasets": "4.0.0",
        "huggingface-hub": "0.36.2",
        "matplotlib": "3.10.5",
        "numpy": "1.26.4",
        "nvidia-nccl-cu12": "2.27.3",
        "omegaconf": "2.3.0",
        "pandas": "2.3.2",
        "pyarrow": "21.0.0",
        "pydantic": "2.11.7",
        "PyYAML": "6.0.2",
        "safetensors": "0.5.3",
        "scipy": "1.16.1",
        "statsmodels": "0.14.6",
        "tabulate": "0.9.0",
        "tokenizers": "0.22.0",
        "transformer-lens": "2.16.1",
        "transformers": "4.56.2",
    }
    observed = {name: importlib.metadata.version(name) for name in expected_versions}
    if observed != expected_versions:
        raise G0Error("fixed G0 runtime package versions differ from the deployment")
    if not importlib.metadata.version("torch").startswith("2.8.0+cu128"):
        raise G0Error("fixed G0 runtime Torch package version differs")
    if _sha256_file(FSDP_CONFIG) != FSDP_CONFIG_SHA256:
        raise G0Error("FSDP base configuration differs from its reviewed digest")
    if not MIB_REPOSITORY.is_dir() or MIB_REPOSITORY.is_symlink():
        raise G0Error("fixed MIB checkout is absent or unsafe")
    try:
        mib_commit = _mib_git_bytes("rev-parse", "HEAD").decode(
            "ascii", errors="strict"
        ).strip()
    except UnicodeDecodeError as error:
        raise G0Error("fixed MIB Git identity is not ASCII") from error
    if mib_commit != MIB_REVISION:
        raise G0Error("fixed MIB checkout differs from its reviewed revision")
    mib_status = _mib_git_bytes(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    try:
        submodules = _mib_git_bytes("submodule", "status", "--recursive").decode(
            "utf-8", errors="strict"
        ).splitlines()
    except UnicodeDecodeError as error:
        raise G0Error("fixed MIB submodule status is not UTF-8") from error
    if mib_status or not submodules or any(
        not row or row[0] != " " for row in submodules
    ):
        raise G0Error("fixed MIB checkout or submodule tree is dirty or incomplete")
    if not (MIB_REPOSITORY / "run_attribution.py").is_file():
        raise G0Error("fixed MIB checkout lacks its reviewed entrypoint")
    return devices


def _validate_config_and_certification(
    invocation: Invocation,
    payloads: dict[str, dict[str, Any]],
    prereg: bytes,
    amendment: bytes,
    raw_inputs: dict[str, bytes],
    *,
    code_commit: str,
    bootstrap_lineage: BootstrapLineage,
    bootstrap_science: BootstrapScienceProtocol,
) -> tuple[dict[str, Any], Any, Any, Any]:
    from posttrain_circuits.artifacts.config_bindings import ConfigBinding, validate_config_binding
    from posttrain_circuits.artifacts.protocol_amendments import (
        load_execution_class_amendment_bytes,
        resolve_accepted_execution_class_amendment,
        validate_execution_class_lineage_commit,
    )
    from posttrain_circuits.artifacts.execution_science_protocol import (
        canonical_science_config_sha256,
        resolve_accepted_execution_science_protocol,
    )
    from posttrain_circuits.artifacts.execution_safety_certification import (
        CERTIFICATION_CONTENT_NAME,
        CERTIFICATION_ID,
        CERTIFICATION_RELATIVE_PATH,
        DESCRIPTOR_CONTENT_NAME,
        DESCRIPTOR_RELATIVE_PATH,
        EXECUTION_CLASS_ID,
        current_execution_safety_fingerprint,
        execution_safety_config_projection,
        load_execution_safety_certification_bytes,
        load_execution_safety_descriptor_bytes,
    )
    from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
        PROTOCOL_AMENDMENT_ID,
    )

    binding_payload = payloads["config_binding_sha256"]
    resolved = payloads["resolved_config_sha256"]
    scientific = payloads["scientific_config_sha256"]
    execution = payloads["execution_config_sha256"]
    try:
        binding = ConfigBinding.from_dict(binding_payload)
        validate_config_binding(resolved, binding)
    except (TypeError, ValueError) as error:
        raise G0Error(f"G0 ConfigBinding is invalid: {error}") from error
    for name in ("execution_config_sha256", "resolved_config_sha256", "scientific_config_sha256"):
        if binding_payload.get(name) != invocation.input_hashes[name]:
            raise G0Error(f"ConfigBinding {name} differs from its CAS input")
    expected_inputs = {
        "execution_safety_certification": invocation.input_hashes[
            CERTIFICATION_CONTENT_NAME
        ],
        "execution_safety_descriptor": invocation.input_hashes[
            DESCRIPTOR_CONTENT_NAME
        ],
        "execution_science_protocol_path": invocation.input_hashes[
            "execution_science_protocol_sha256"
        ],
        "prereg_path": invocation.input_hashes["preregistration_sha256"],
        "protocol_amendment_path": invocation.input_hashes["protocol_amendment_sha256"],
    }
    if binding_payload.get("input_artifact_hashes") != expected_inputs:
        raise G0Error("ConfigBinding does not bind the exact G0 prerequisite bytes")
    expected_execution = {
        "allocation_contract": "manifest_driven_scheduler_gpu_v1",
        "scheduler_protocol": 2,
    }
    if binding_payload.get("execution_context") != expected_execution:
        raise G0Error("ConfigBinding execution context is not allocation-neutral")
    if execution.get("execution_context") != expected_execution:
        raise G0Error("G0 execution projection is not allocation-neutral")
    if (
        hashlib.sha256(prereg).hexdigest() != PREREGISTRATION_SHA256
        or invocation.input_hashes["preregistration_sha256"] != PREREGISTRATION_SHA256
    ):
        raise G0Error("preregistration content identity changed")
    descriptor_raw = raw_inputs[DESCRIPTOR_CONTENT_NAME]
    certification_raw = raw_inputs[CERTIFICATION_CONTENT_NAME]
    try:
        descriptor_payload = load_execution_safety_descriptor_bytes(
            descriptor_raw,
            code_root=SOURCE_ROOT,
        )
        certification_binding = load_execution_safety_certification_bytes(
            certification_raw,
            descriptor_raw,
            require_accepted=True,
            code_root=SOURCE_ROOT,
        )
    except ValueError as error:
        raise G0Error(f"G0 execution-safety certification is invalid: {error}") from error
    if (
        certification_binding.execution_class_id != EXECUTION_CLASS_ID
        or tuple(certification_binding.supported_world_sizes) != ALLOWED_GPU_COUNTS
        or certification_binding.review_status != "accepted"
        or certification_binding.descriptor_sha256
        != invocation.input_hashes[DESCRIPTOR_CONTENT_NAME]
        or certification_binding.certification_sha256
        != invocation.input_hashes[CERTIFICATION_CONTENT_NAME]
        or certification_binding.fingerprint_sha256
        != current_execution_safety_fingerprint(SOURCE_ROOT)
    ):
        raise G0Error(
            "G0 execution-safety certification differs from its exact CAS bytes"
        )
    amendment_payload = load_execution_class_amendment_bytes(amendment)
    source_amendment = SOURCE_ROOT / AMENDMENT_RELATIVE_PATH
    if _source_regular_bytes(
        source_amendment,
        context="source protocol amendment",
        max_bytes=MAX_AMENDMENT_BYTES,
    ) != amendment:
        raise G0Error("protocol amendment content differs from the clean source commit")
    amendment_binding = resolve_accepted_execution_class_amendment(
        code_root=SOURCE_ROOT,
        configured_path=str(AMENDMENT_RELATIVE_PATH),
        expected_head=code_commit,
    )
    expected_certification_terms = {
        "execution_class_id": certification_binding.execution_class_id,
        "certification_id": CERTIFICATION_ID,
        "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
        "descriptor_sha256": certification_binding.descriptor_sha256,
        "certification_path": str(CERTIFICATION_RELATIVE_PATH),
        "certification_core_sha256": (
            certification_binding.certification_core_sha256
        ),
        "fingerprint_sha256": certification_binding.fingerprint_sha256,
        "supported_world_sizes": list(
            certification_binding.supported_world_sizes
        ),
    }
    if (
        amendment_binding.sha256 != bootstrap_lineage.amendment_sha256
        or amendment_binding.git_commit != bootstrap_lineage.amendment_git_commit
        or amendment_binding.reviewed_implementation_commit
        != bootstrap_lineage.reviewed_implementation_commit
        or amendment_payload["review"]["status"] != "accepted"
        or amendment_payload.get("amendment_id") != PROTOCOL_AMENDMENT_ID
        or amendment_payload.get("execution_safety_certification")
        != expected_certification_terms
        or amendment_binding.certification_binding != certification_binding
        or certification_binding.reviewed_implementation_commit
        != amendment_binding.reviewed_implementation_commit
        or hashlib.sha256(amendment).hexdigest() != amendment_binding.sha256
        or invocation.input_hashes["protocol_amendment_sha256"] != amendment_binding.sha256
    ):
        raise G0Error("protocol amendment is not the accepted Git-reviewed content")

    for candidate_commit, role in (
        (bootstrap_lineage.request_git_commit, "G0 request commit"),
        (code_commit, "G0 execution commit"),
    ):
        try:
            validate_execution_class_lineage_commit(
                code_root=SOURCE_ROOT,
                candidate_commit=candidate_commit,
                current_binding=amendment_binding,
                expected_head=code_commit,
                role=role,
            )
        except ValueError as error:
            raise G0Error(f"{role} violates reviewed lineage: {error}") from error

    scheduler_config = resolved.get("scheduler_g0")
    if not isinstance(scheduler_config, dict):
        raise G0Error("resolved config lacks scheduler G0 provenance")
    expected_scheduler_fields = {
        "allocation_contract",
        "artifact_namespace",
        "batch_partition_protocol",
        "execution_safety_certification_sha256",
        "execution_safety_class_id",
        "execution_safety_descriptor_sha256",
        "execution_safety_fingerprint_sha256",
        "execution_safety_supported_world_sizes",
        "execution_science_protocol_git_commit",
        "execution_science_protocol_id",
        "execution_science_protocol_path",
        "execution_science_protocol_reviewed_implementation_commit",
        "execution_science_protocol_sha256",
        "model_revision",
        "protocol_amendment_id",
        "protocol_amendment_sha256",
        "request_git_commit",
        "reviewed_implementation_commit",
        "task",
        "teacher_revision",
        "tokenizer_fingerprint",
    }
    request_git_commit = scheduler_config.get("request_git_commit")
    if (
        set(scheduler_config) != expected_scheduler_fields
        or not isinstance(request_git_commit, str)
        or GIT_COMMIT.fullmatch(request_git_commit) is None
        or request_git_commit != bootstrap_lineage.request_git_commit
        or scheduler_config.get("allocation_contract")
        != "manifest_driven_scheduler_gpu_v1"
        or scheduler_config.get("artifact_namespace") != "qwen3-v2"
        or scheduler_config.get("batch_partition_protocol")
        != BATCH_PARTITION_PROTOCOL
        or scheduler_config.get("execution_safety_class_id")
        != certification_binding.execution_class_id
        or scheduler_config.get("execution_safety_fingerprint_sha256")
        != certification_binding.fingerprint_sha256
        or scheduler_config.get("execution_safety_descriptor_sha256")
        != certification_binding.descriptor_sha256
        or scheduler_config.get("execution_safety_certification_sha256")
        != certification_binding.certification_sha256
        or scheduler_config.get("execution_safety_supported_world_sizes")
        != list(certification_binding.supported_world_sizes)
        or scheduler_config.get("execution_science_protocol_path")
        != bootstrap_science.science_protocol_path
        or scheduler_config.get("execution_science_protocol_sha256")
        != bootstrap_science.science_protocol_sha256
        or scheduler_config.get("execution_science_protocol_id")
        != bootstrap_science.science_protocol_id
        or scheduler_config.get("execution_science_protocol_git_commit")
        != bootstrap_science.science_protocol_git_commit
        or scheduler_config.get(
            "execution_science_protocol_reviewed_implementation_commit"
        )
        != bootstrap_science.science_protocol_reviewed_implementation_commit
        or scheduler_config.get("model_revision") != MODEL_REVISION
        or scheduler_config.get("protocol_amendment_id") != PROTOCOL_AMENDMENT_ID
        or scheduler_config.get("protocol_amendment_sha256")
        != amendment_binding.sha256
        or scheduler_config.get("reviewed_implementation_commit")
        != amendment_binding.reviewed_implementation_commit
        or scheduler_config.get("task") != TASK
        or scheduler_config.get("teacher_revision") != TEACHER_REVISION
        or scheduler_config.get("tokenizer_fingerprint")
        != TOKENIZER_FINGERPRINT
    ):
        raise G0Error(
            "resolved config contains invalid request/execution-safety provenance"
        )
    if execution_safety_config_projection(resolved) != descriptor_payload["subject"][
        "resolved_config_safety_projection"
    ]:
        raise G0Error(
            "resolved config differs from the certified execution-safety projection"
        )
    science_raw = raw_inputs["execution_science_protocol_sha256"]
    try:
        resolved_science = resolve_accepted_execution_science_protocol(
            code_root=SOURCE_ROOT,
            configured_path=bootstrap_science.science_protocol_path,
            expected_head=code_commit,
        )
    except ValueError as error:
        raise G0Error(f"G0 execution science protocol is invalid: {error}") from error
    science_binding = resolved_science.binding
    from posttrain_circuits.core.config import compose_config

    try:
        composed_science = compose_config(
            list(science_binding.hydra_override_vector),
            config_root=SOURCE_ROOT / "configs",
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise G0Error(
            f"accepted execution science vector cannot be composed: {error}"
        ) from error
    science_execution_class = science_binding.payload.get("execution_class")
    expected_science_execution_class = {
        "execution_class_id": certification_binding.execution_class_id,
        "descriptor_path": str(DESCRIPTOR_RELATIVE_PATH),
        "descriptor_sha256": certification_binding.descriptor_sha256,
        "fingerprint_sha256": certification_binding.fingerprint_sha256,
        "certification_id": CERTIFICATION_ID,
        "certification_path": str(CERTIFICATION_RELATIVE_PATH),
        "certification_core_sha256": (
            certification_binding.certification_core_sha256
        ),
        "supported_world_sizes": list(certification_binding.supported_world_sizes),
    }
    if (
        science_binding.artifact_sha256
        != invocation.input_hashes["execution_science_protocol_sha256"]
        or resolved_science.raw != science_raw
        or resolved_science.acceptance_commit
        != bootstrap_science.science_protocol_git_commit
        or science_binding.artifact_sha256
        != bootstrap_science.science_protocol_sha256
        or science_binding.protocol_id != bootstrap_science.science_protocol_id
        or science_binding.unit_id != invocation.unit_id
        or resolved.get("seed") != science_binding.seed
        or science_binding.reviewed_implementation_commit
        != bootstrap_science.science_protocol_reviewed_implementation_commit
        or science_binding.execution_class_id
        != certification_binding.execution_class_id
        or science_binding.execution_safety_fingerprint_sha256
        != certification_binding.fingerprint_sha256
        or science_execution_class != expected_science_execution_class
        or science_binding.storage_neutral_resolved_config_sha256
        != canonical_science_config_sha256(resolved)
        or canonical_science_config_sha256(composed_science)
        != science_binding.storage_neutral_resolved_config_sha256
        or execution_safety_config_projection(composed_science)
        != descriptor_payload["subject"]["resolved_config_safety_projection"]
        or science_binding.hydra_override_vector
        != bootstrap_science.hydra_override_vector
    ):
        raise G0Error(
            "G0 resolved config and execution class differ from the accepted "
            "science protocol"
        )
    if _sha256_value(resolved) != invocation.input_hashes["resolved_config_sha256"]:
        raise G0Error("resolved G0 config hash is invalid")
    if scientific.get("input_artifact_hashes") != expected_inputs:
        raise G0Error("scientific config projection lost prerequisite identities")
    return resolved, certification_binding, amendment_binding, science_binding


def _common_overrides(
    root: Path,
    science_overrides: Sequence[str],
    *,
    experiment: str | None = None,
) -> list[str]:
    dataset = root / "dataset"
    initial = root / "initial_checkpoint.pt"
    demos = root / "teacher_demos"
    probes = root / "probes" / "manifest.json"
    readiness = root / "readiness" / "readiness.json"
    reviewed = list(_bootstrap_hydra_override_vector(list(science_overrides)))
    handler_owned = {
        "output_root",
        "protocol_amendment_path",
        "task.dataset_family_path",
        "state_source.store_path",
        "anti_shortcut.report_path",
        "production_safety.readiness_report",
        "production_safety.probe_cohort_manifest",
        "production_safety.initial_checkpoint_path",
        "production_safety.initial_checkpoint_hash",
    }
    if any(item.split("=", 1)[0] in handler_owned for item in reviewed):
        raise G0Error(
            "execution science protocol attempts to override a handler-owned path"
        )
    if experiment is not None:
        # Teacher scoring is the one reviewed auxiliary route.  Replace the
        # experiment selector in place so every other accepted override
        # (notably seed and shape) remains identical and no duplicate Hydra
        # key relies on last-write-wins behavior.
        matches = [
            index
            for index, item in enumerate(reviewed)
            if item.split("=", 1)[0] == "experiment"
        ]
        if len(matches) != 1:
            raise G0Error(
                "execution science protocol must select exactly one experiment"
            )
        reviewed[matches[0]] = f"experiment={experiment}"
    return [
        *reviewed,
        f"protocol_amendment_path={AMENDMENT_RELATIVE_PATH}",
        f"output_root={root}",
        f"task.dataset_family_path={dataset}",
        f"state_source.store_path={demos}",
        f"anti_shortcut.report_path={root / 'anti_shortcut.json'}",
        f"production_safety.readiness_report={readiness}",
        f"production_safety.probe_cohort_manifest={probes}",
        f"production_safety.initial_checkpoint_path={initial}",
    ]


def _stage_plan(
    root: Path,
    *,
    initial_checkpoint_sha256: str,
    gpu_count: int,
    science_overrides: Sequence[str],
) -> tuple[Stage, ...]:
    if gpu_count not in ALLOWED_GPU_COUNTS:
        raise G0Error("G0 stage plan GPU count is outside 1..4")
    common = _common_overrides(root, science_overrides)
    initial_binding = [
        *common,
        f"production_safety.initial_checkpoint_hash={initial_checkpoint_sha256}",
    ]
    dataset = root / "dataset"
    initial = root / "initial_checkpoint.pt"
    demos = root / "teacher_demos"
    calibration = root / "calibration"
    calibration_checkpoint = calibration / "FINAL_CHECKPOINT"
    probe_scores = root / "probe_scores.json"
    probes = root / "probes"
    bank = root / "rollout_bank"
    teacher_scores = root / "teacher_scores"
    circuits = root / "circuits"
    stages: list[Stage] = [
        Stage(
            "build_splits",
            "build_splits",
            (*common, "--output", str(dataset), "--confirm-production"),
        ),
        Stage(
            "export_initial_checkpoint",
            "export_initial_checkpoint",
            (*common, "--output", str(initial), "--confirm-production"),
        ),
        Stage(
            "build_teacher_demos",
            "build_teacher_demos",
            (*initial_binding, "--output", str(demos), "--confirm-production"),
        ),
        Stage(
            "audit_label_leakage",
            "audit_label_leakage",
            (
                *initial_binding,
                "--split-root",
                str(dataset),
                "--split",
                "validation",
                "--output",
                str(root / "label_leakage.json"),
            ),
        ),
        Stage(
            "evaluate_teacher_readiness",
            "evaluate_teacher_readiness",
            (
                *initial_binding,
                "--validation-split",
                str(dataset / "validation"),
                "--limit",
                "128",
                "--top-k",
                "128",
                "--output",
                str(root / "teacher_readiness.json"),
                "--confirm-production",
            ),
        ),
        Stage(
            "calibration_sft",
            "train",
            (
                *initial_binding,
                "trainer.max_steps=120",
                "trainer.checkpoint_every=20",
                "--output",
                str(calibration),
                "--confirm-production",
            ),
            distributed=True,
        ),
        Stage(
            "score_probe_candidates",
            "score_probe_candidates",
            (
                *initial_binding,
                "--initial-checkpoint",
                str(initial),
                "--calibration-checkpoint",
                str(calibration_checkpoint),
                "--calibration-run-manifest",
                str(calibration / "manifest.json"),
                "--world-size",
                str(gpu_count),
                "--dataset-family",
                str(dataset),
                "--probe-limit-per-split",
                "128",
                "--task-validation-limit",
                "128",
                "--output",
                str(probe_scores),
            ),
        ),
        Stage(
            "build_probe_cohorts",
            "build_probe_cohorts",
            (
                "--dataset-family",
                str(dataset),
                "--scores",
                str(probe_scores),
                "--limit-per-split",
                "128",
                "--output",
                str(probes),
            ),
        ),
        Stage(
            "build_rollout_bank",
            "build_rollout_bank",
            (*initial_binding, "--output", str(bank), "--confirm-production"),
        ),
        Stage(
            "score_teacher",
            "score_teacher",
            (
                *_common_overrides(
                    root,
                    science_overrides,
                    experiment="offline_soft",
                ),
                f"production_safety.initial_checkpoint_hash={initial_checkpoint_sha256}",
                "--bank",
                str(bank),
                "--output",
                str(teacher_scores),
                "--confirm-production",
            ),
        ),
        Stage(
            "evaluate_anti_shortcut",
            "evaluate_anti_shortcut",
            (*initial_binding, "--output", str(root / "anti_shortcut.json"), "--confirm-production"),
        ),
    ]
    for stage_name in ("final_answer", "first_rule_selection"):
        stage_root = circuits / stage_name
        stages.extend(
            (
                Stage(
                    f"discover_{stage_name}",
                    "discover_circuit",
                    (
                        *initial_binding,
                        "--stage",
                        stage_name,
                        "--checkpoint",
                        str(initial),
                        "--initial-checkpoint",
                        str(initial),
                        "--probe-cohort-manifest",
                        str(probes / "manifest.json"),
                        "--cohort",
                        "base_capable",
                        "--output",
                        str(stage_root / "circuit.json"),
                        "--confirm-production",
                    ),
                ),
                Stage(
                    f"evaluate_{stage_name}",
                    "evaluate_circuit",
                    (
                        *initial_binding,
                        "--circuit-artifact",
                        str(stage_root / "circuit.json"),
                        "--checkpoint",
                        str(initial),
                        "--initial-checkpoint",
                        str(initial),
                        "--probe-cohort-manifest",
                        str(probes / "manifest.json"),
                        "--cohort",
                        "base_capable",
                        "--output",
                        str(stage_root / "exact_patching.json"),
                        "--confirm-production",
                    ),
                ),
            )
        )
    resume_source = calibration / "checkpoints" / "step-00000020.pt"
    for replica in ("a", "b"):
        stages.append(
            Stage(
                f"resume_{replica}",
                "train",
                (
                    *initial_binding,
                    "trainer.max_steps=120",
                    "trainer.checkpoint_every=20",
                    "--resume",
                    str(resume_source),
                    "--output",
                    str(root / f"resume-{replica}"),
                    "--confirm-production",
                ),
                distributed=True,
            )
        )
    stages.extend(
        (
            Stage(
                "compare_distributed_resume",
                "compare_distributed_resume",
                (
                    *initial_binding,
                    "--workspace",
                    str(root),
                    "--resume-source",
                    str(resume_source),
                    "--world-size",
                    str(gpu_count),
                    "--output",
                    str(root / "distributed_resume.json"),
                ),
            ),
            Stage(
                "finalize_g0",
                "finalize_g0",
                (
                    *initial_binding,
                    "--base-scores",
                    str(probe_scores),
                    "--teacher-store-manifest",
                    str(teacher_scores / "manifest.json"),
                    "--teacher-readiness",
                    str(root / "teacher_readiness.json"),
                    "--label-leakage",
                    str(root / "label_leakage.json"),
                    "--anti-shortcut",
                    str(root / "anti_shortcut.json"),
                    "--probe-manifest",
                    str(probes / "manifest.json"),
                    "--final-circuit",
                    str(circuits / "final_answer" / "circuit.json"),
                    "--final-exact-patching",
                    str(circuits / "final_answer" / "exact_patching.json"),
                    "--process-circuit",
                    str(circuits / "first_rule_selection" / "circuit.json"),
                    "--process-exact-patching",
                    str(circuits / "first_rule_selection" / "exact_patching.json"),
                    "--final-compatibility",
                    str(circuits / "final_answer" / "mib_raw" / "compatibility.json"),
                    "--process-compatibility",
                    str(
                        circuits
                        / "first_rule_selection"
                        / "mib_raw"
                        / "compatibility.json"
                    ),
                    "--distributed-resume",
                    str(root / "distributed_resume.json"),
                    "--initial-checkpoint",
                    str(initial),
                    "--execution-safety-descriptor",
                    str(root / "execution_safety_descriptor.json"),
                    "--execution-safety-certification",
                    str(root / "execution_safety_certification.yaml"),
                    "--job-id",
                    "JOB_ID",
                    "--output",
                    str(root / "g0.json"),
                ),
            ),
        )
    )
    return tuple(stages)


def _script_parent_path() -> str:
    match = PROC_SELF_FD.fullmatch(os.path.abspath(__file__))
    if match is None:
        raise G0Error("deployment implementation must execute through a held descriptor")
    return f"/proc/{os.getpid()}/fd/{match.group(1)}"


def _child_environment() -> dict[str, str]:
    return dict(os.environ)


def _run_stage(
    stage: Stage,
    *,
    script_path: str,
    job_id: str,
    bootstrap_lineage: BootstrapLineage,
    bootstrap_science: BootstrapScienceProtocol,
    code_commit: str,
    gpu_count: int,
    allocation_sha256: str,
    execution_safety_class_id: str,
    execution_safety_fingerprint_sha256: str,
    execution_safety_descriptor_sha256: str,
    execution_safety_certification_sha256: str,
) -> None:
    if (
        gpu_count not in ALLOWED_GPU_COUNTS
        or SHA256.fullmatch(allocation_sha256) is None
        or IDENTIFIER.fullmatch(execution_safety_class_id) is None
        or SHA256.fullmatch(execution_safety_fingerprint_sha256) is None
        or SHA256.fullmatch(execution_safety_descriptor_sha256) is None
        or SHA256.fullmatch(execution_safety_certification_sha256) is None
        or IDENTIFIER.fullmatch(bootstrap_science.science_protocol_id) is None
        or SHA256.fullmatch(bootstrap_science.science_protocol_sha256) is None
        or GIT_COMMIT.fullmatch(bootstrap_science.science_protocol_git_commit) is None
        or GIT_COMMIT.fullmatch(
            bootstrap_science.science_protocol_reviewed_implementation_commit
        )
        is None
    ):
        raise G0Error("stage launch allocation context is invalid")
    argv = tuple(job_id if value == "JOB_ID" else value for value in stage.argv)
    scientific_argv = (
        script_path,
        "--scientific-cli",
        stage.cli,
        "--code-commit",
        code_commit,
        "--request-git-commit",
        bootstrap_lineage.request_git_commit,
        "--execution-safety-class-id",
        execution_safety_class_id,
        "--execution-safety-fingerprint-sha256",
        execution_safety_fingerprint_sha256,
        "--execution-safety-descriptor-sha256",
        execution_safety_descriptor_sha256,
        "--execution-safety-certification-sha256",
        execution_safety_certification_sha256,
        "--execution-science-protocol-id",
        bootstrap_science.science_protocol_id,
        "--execution-science-protocol-sha256",
        bootstrap_science.science_protocol_sha256,
        "--execution-science-protocol-git-commit",
        bootstrap_science.science_protocol_git_commit,
        "--execution-science-protocol-reviewed-implementation-commit",
        bootstrap_science.science_protocol_reviewed_implementation_commit,
        "--amendment-sha256",
        bootstrap_lineage.amendment_sha256,
        "--reviewed-implementation-commit",
        bootstrap_lineage.reviewed_implementation_commit,
        "--gpu-count",
        str(gpu_count),
        "--allocation-sha256",
        allocation_sha256,
        "--",
        *argv,
    )
    command = [sys.executable, "-I"]
    if stage.distributed:
        command.extend(
            (
                "-m",
                "accelerate.commands.launch",
                "--config_file",
                str(FSDP_CONFIG),
                "--num_processes",
                str(gpu_count),
                *scientific_argv,
            )
        )
    else:
        command.extend(scientific_argv)
    _log_phase("g0_stage_started", stage=stage.name, distributed=stage.distributed)
    subprocess.run(
        command,
        cwd=SOURCE_ROOT,
        env=_child_environment(),
        check=True,
    )
    _log_phase("g0_stage_completed", stage=stage.name)


def _resolve_final_checkpoint(run_root: Path) -> Path:
    manifest_path = run_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise G0Error(f"{run_root.name} run manifest is not a regular file")
    payload = _strict_json(manifest_path.read_bytes(), context=f"{run_root.name} run manifest")
    if payload.get("sha256") != _sha256_value(
        {key: value for key, value in payload.items() if key != "sha256"}
    ):
        raise G0Error(f"{run_root.name} run manifest SHA-256 is invalid")
    candidate = Path(str(payload.get("final_checkpoint_path", "")))
    expected = payload.get("final_checkpoint_sha256")
    checkpoint_root = (run_root / "checkpoints").resolve(strict=True)
    if (
        not candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.is_symlink()
        or candidate.resolve(strict=True) != candidate
        or not candidate.is_file()
        or not candidate.is_relative_to(checkpoint_root)
        or not isinstance(expected, str)
        or SHA256.fullmatch(expected) is None
        or _sha256_file(candidate) != expected
    ):
        raise G0Error(f"{run_root.name} final checkpoint binding is missing or invalid")
    return candidate


def _replace_checkpoint_placeholders(stages: tuple[Stage, ...], root: Path) -> tuple[Stage, ...]:
    replacements: dict[str, str] = {}
    for name in ("calibration", "resume-a", "resume-b"):
        run_root = root / name
        if (run_root / "manifest.json").is_file():
            replacements[str(run_root / "FINAL_CHECKPOINT")] = str(
                _resolve_final_checkpoint(run_root)
            )
    return tuple(
        Stage(
            stage.name,
            stage.cli,
            tuple(replacements.get(argument, argument) for argument in stage.argv),
            stage.distributed,
        )
        for stage in stages
    )


def _read_cgroup_value(path: Path) -> int | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if value == "max":
        return None
    parsed = int(value)
    return None if parsed >= 2**60 else parsed


def _cgroup_memory() -> dict[str, int | bool]:
    unified: Path | None = None
    legacy: Path | None = None
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        relative = Path(fields[2].lstrip("/"))
        controllers = fields[1].split(",") if fields[1] else []
        if fields[0] == "0" and not controllers:
            unified = relative
        if "memory" in controllers:
            legacy = relative
    if unified is not None:
        root = Path("/sys/fs/cgroup") / unified
        names = ("memory.max", "memory.current", "memory.peak")
    elif legacy is not None:
        root = Path("/sys/fs/cgroup/memory") / legacy
        names = ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
    else:
        raise G0Error("allocation has no readable memory cgroup")
    limit, current, peak = (_read_cgroup_value(root / name) for name in names)
    if (
        limit != NODE_MEMORY_BYTES
        or current is None
        or peak is None
        or current > limit
        or peak <= 0
        or peak > limit
    ):
        raise G0Error("memory cgroup is not an allocation-specific 192-GiB limit")
    headroom = limit - peak
    if headroom < MINIMUM_HEADROOM_BYTES:
        raise G0Error("G0 did not preserve the registered host-memory headroom")
    return {
        "current_bytes": current,
        "headroom_bytes": headroom,
        "limit_bytes": limit,
        "minimum_required_headroom_bytes": MINIMUM_HEADROOM_BYTES,
        "passed": True,
        "peak_bytes": peak,
        "requested_bytes": NODE_MEMORY_BYTES,
    }


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise G0Error(f"G0 artifact tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise G0Error(f"G0 artifact tree contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        rows.append({"path": relative, "sha256": _sha256_file(path), "size": path.stat().st_size})
    if not rows:
        raise G0Error("G0 artifact tree is empty")
    return rows


def _build_bundle(root: Path, target: Path, inventory: list[dict[str, Any]]) -> None:
    by_name = {str(row["path"]): row for row in inventory}
    with tarfile.open(target, mode="x:", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(by_name):
            path = root / Path(name)
            info = tarfile.TarInfo(name=name)
            info.size = int(by_name[name]["size"])
            info.mode = 0o440
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)


def _publish_json_once(descriptor: int, name: str, payload: object) -> None:
    _publish_bytes_once(descriptor, name, _published_json_bytes(payload))


def _publish_bytes_once(descriptor: int, name: str, raw: bytes) -> None:
    if name not in {REPORT_NAME, COMPLETION_NAME}:
        raise G0Error("JSON output name is not fixed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    output = os.open(name, flags, 0o640, dir_fd=descriptor)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(output, raw[offset:])
        os.fsync(output)
    finally:
        os.close(output)
    os.fsync(descriptor)


def _publish_file_once(descriptor: int, name: str, source: Path) -> str:
    if name != BUNDLE_NAME:
        raise G0Error("binary output name is not fixed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    output = os.open(name, flags, 0o640, dir_fd=descriptor)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = os.write(output, view)
                    if written < 1:
                        raise G0Error("short write while publishing G0 artifact bundle")
                    view = view[written:]
        os.fsync(output)
    finally:
        os.close(output)
    os.fsync(descriptor)
    return digest.hexdigest()


def _completion(invocation: Invocation, *, started_at: str, completed_at: str) -> dict[str, Any]:
    content = {
        "completed_at": completed_at,
        "completion_kind": COMPLETION_KIND,
        "execution": invocation.execution,
        "execution_config_sha256": invocation.input_hashes["execution_config_sha256"],
        "input_hashes": invocation.input_hashes,
        "plan_sha256": invocation.plan_sha256,
        "project": PROJECT,
        "resolved_config_sha256": invocation.input_hashes["resolved_config_sha256"],
        "run_id": invocation.run_id,
        "schema_version": 1,
        "scientific_config_sha256": invocation.input_hashes["scientific_config_sha256"],
        "scientific_validation": {name: True for name in GATE_NAMES},
        "started_at": started_at,
        "task": TASK,
        "unit_id": invocation.unit_id,
        "workflow_id": invocation.workflow_id,
    }
    return {**content, "sha256": _sha256_value(content)}


def _scientific_cli(argv: Sequence[str]) -> int:
    if (
        len(argv) < 30
        or argv[0] not in SCIENTIFIC_CLIS
        or argv[1] != "--code-commit"
        or argv[3] != "--request-git-commit"
        or argv[5] != "--execution-safety-class-id"
        or argv[7] != "--execution-safety-fingerprint-sha256"
        or argv[9] != "--execution-safety-descriptor-sha256"
        or argv[11] != "--execution-safety-certification-sha256"
        or argv[13] != "--execution-science-protocol-id"
        or argv[15] != "--execution-science-protocol-sha256"
        or argv[17] != "--execution-science-protocol-git-commit"
        or argv[19]
        != "--execution-science-protocol-reviewed-implementation-commit"
        or argv[21] != "--amendment-sha256"
        or argv[23] != "--reviewed-implementation-commit"
        or argv[25] != "--gpu-count"
        or argv[27] != "--allocation-sha256"
        or argv[29] != "--"
    ):
        raise G0Error("scientific child invocation differs from the fixed CLI ABI")
    _configure_bytecode_isolation()
    code_commit = argv[2]
    request_commit = argv[4]
    execution_safety_class_id = argv[6]
    execution_safety_fingerprint_sha256 = argv[8]
    execution_safety_descriptor_sha256 = argv[10]
    execution_safety_certification_sha256 = argv[12]
    execution_science_protocol_id = argv[14]
    execution_science_protocol_sha256 = argv[16]
    execution_science_protocol_git_commit = argv[18]
    execution_science_protocol_reviewed_implementation_commit = argv[20]
    amendment_sha256 = argv[22]
    reviewed_implementation_commit = argv[24]
    gpu_count_raw = argv[26]
    allocation_sha256 = argv[28]
    if (
        GIT_COMMIT.fullmatch(code_commit) is None
        or GIT_COMMIT.fullmatch(request_commit) is None
        or IDENTIFIER.fullmatch(execution_safety_class_id) is None
        or SHA256.fullmatch(execution_safety_fingerprint_sha256) is None
        or SHA256.fullmatch(execution_safety_descriptor_sha256) is None
        or SHA256.fullmatch(execution_safety_certification_sha256) is None
        or IDENTIFIER.fullmatch(execution_science_protocol_id) is None
        or SHA256.fullmatch(execution_science_protocol_sha256) is None
        or GIT_COMMIT.fullmatch(execution_science_protocol_git_commit) is None
        or GIT_COMMIT.fullmatch(
            execution_science_protocol_reviewed_implementation_commit
        )
        is None
        or SHA256.fullmatch(amendment_sha256) is None
        or GIT_COMMIT.fullmatch(reviewed_implementation_commit) is None
        or POSITIVE_INTEGER.fullmatch(gpu_count_raw) is None
        or int(gpu_count_raw) not in ALLOWED_GPU_COUNTS
        or SHA256.fullmatch(allocation_sha256) is None
        or _require_clean_git() != code_commit
    ):
        raise G0Error("scientific child lineage arguments are invalid")
    amendment = _source_regular_bytes(
        SOURCE_ROOT / AMENDMENT_RELATIVE_PATH,
        context="source protocol amendment",
        max_bytes=MAX_AMENDMENT_BYTES,
    )
    descriptor_raw = _source_regular_bytes(
        SOURCE_ROOT / DESCRIPTOR_RELATIVE_PATH,
        context="source execution-safety descriptor",
        max_bytes=MAX_DESCRIPTOR_BYTES,
    )
    certification_raw = _source_regular_bytes(
        SOURCE_ROOT / CERTIFICATION_RELATIVE_PATH,
        context="source execution-safety certification",
        max_bytes=MAX_CERTIFICATION_BYTES,
    )
    lineage = _bootstrap_accepted_lineage(
        code_commit=code_commit,
        resolved={
            "scheduler_g0": {
                "request_git_commit": request_commit,
                "execution_safety_class_id": execution_safety_class_id,
                "execution_safety_fingerprint_sha256": (
                    execution_safety_fingerprint_sha256
                ),
                "execution_safety_descriptor_sha256": (
                    execution_safety_descriptor_sha256
                ),
                "execution_safety_certification_sha256": (
                    execution_safety_certification_sha256
                ),
            }
        },
        amendment=amendment,
        descriptor_raw=descriptor_raw,
        certification_raw=certification_raw,
    )
    if (
        lineage.amendment_sha256 != amendment_sha256
        or lineage.reviewed_implementation_commit != reviewed_implementation_commit
    ):
        raise G0Error("scientific child amendment identity differs from its supervisor")
    _install_source_path()
    module = importlib.import_module(SCIENTIFIC_CLIS[argv[0]])
    main_function = getattr(module, "main", None)
    if not callable(main_function):
        raise G0Error("reviewed scientific CLI has no callable main")
    child_argv = list(argv[30:])
    if argv[0] == "finalize_g0":
        from posttrain_circuits.cli.finalize_g0 import ScientificInvocationContext

        main_function(
            child_argv,
            scientific_context=ScientificInvocationContext(
                allocation_sha256=allocation_sha256,
                code_commit=code_commit,
                execution_safety_class_id=execution_safety_class_id,
                execution_safety_fingerprint_sha256=(
                    execution_safety_fingerprint_sha256
                ),
                execution_safety_descriptor_sha256=(
                    execution_safety_descriptor_sha256
                ),
                execution_safety_certification_sha256=(
                    execution_safety_certification_sha256
                ),
                execution_science_protocol_id=execution_science_protocol_id,
                execution_science_protocol_sha256=(
                    execution_science_protocol_sha256
                ),
                execution_science_protocol_git_commit=(
                    execution_science_protocol_git_commit
                ),
                execution_science_protocol_reviewed_implementation_commit=(
                    execution_science_protocol_reviewed_implementation_commit
                ),
                protocol_amendment_sha256=amendment_sha256,
                request_git_commit=request_commit,
                reviewed_implementation_commit=reviewed_implementation_commit,
                world_size=int(gpu_count_raw),
            ),
        )
    else:
        main_function(child_argv)
    if _require_clean_git() != code_commit:
        raise G0Error("source checkout changed during scientific child execution")
    return 0


def _supervise(argv: Sequence[str] | None) -> int:
    started_at = _utc_now()
    _configure_bytecode_isolation()
    invocation = _parse_outer(argv)
    _validate_held_running_manifest(invocation)
    code_commit = _require_clean_git()
    payloads, prereg, amendment, raw_inputs = _read_inputs(invocation)
    bootstrap_lineage = _bootstrap_accepted_lineage(
        code_commit=code_commit,
        resolved=payloads["resolved_config_sha256"],
        amendment=amendment,
        descriptor_raw=raw_inputs["execution_safety_descriptor_sha256"],
        certification_raw=raw_inputs["execution_safety_certification_sha256"],
    )
    bootstrap_science = _bootstrap_science_protocol_lineage(
        code_commit=code_commit,
        resolved=payloads["resolved_config_sha256"],
        science_protocol_raw=raw_inputs["execution_science_protocol_sha256"],
        unit_id=invocation.unit_id,
    )
    _install_source_path()
    resolved, certification_binding, amendment_binding, science_binding = (
        _validate_config_and_certification(
            invocation,
            payloads,
            prereg,
            amendment,
            raw_inputs,
            code_commit=code_commit,
            bootstrap_lineage=bootstrap_lineage,
            bootstrap_science=bootstrap_science,
        )
    )
    _validate_environment(invocation.gpu_count)
    from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
        batch_token_contract,
        execution_context,
    )
    from posttrain_circuits.learning.training.execution_safety_kernel import (
        EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION,
        batch_token_contract as kernel_batch_token_contract,
        build_runtime_execution_safety_attestation,
        execution_safety_config_projection,
    )
    from posttrain_circuits.cli.compare_distributed_resume import (
        validate_distributed_resume_report,
    )
    from posttrain_circuits.core.config import compose_config
    from posttrain_circuits.cli.finalize_g0 import G0_CHECK_NAMES
    if os.listdir(invocation.output_descriptor):
        raise G0Error("output attempt is not empty")
    temp_root = Path(FIXED_ENVIRONMENT["TMPDIR"])
    temp_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="qwen3-v2-g0-", dir=temp_root))
    workspace = temporary / "qwen3-v2"
    workspace.mkdir(mode=0o750)
    bundle_temp = temporary / ".g0-artifacts.tar"
    try:
        (workspace / "execution_safety_descriptor.json").write_bytes(
            raw_inputs["execution_safety_descriptor_sha256"]
        )
        (workspace / "execution_safety_certification.yaml").write_bytes(
            raw_inputs["execution_safety_certification_sha256"]
        )
        (workspace / "execution_science_protocol.yaml").write_bytes(
            raw_inputs["execution_science_protocol_sha256"]
        )
        common = _common_overrides(
            workspace,
            science_binding.hydra_override_vector,
        )
        try:
            pre_stage_config = compose_config(common, config_root=SOURCE_ROOT / "configs")
            if (
                execution_safety_config_projection(pre_stage_config)
                != EXPECTED_EXECUTION_CONFIG_SAFETY_PROJECTION
            ):
                raise G0Error(
                    "runtime config differs from the certified safety projection"
                )
            kernel_batch_token_contract(
                invocation.gpu_count,
                pre_stage_config["task"]["num_examples"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise G0Error(
                f"runtime execution-safety pre-stage gate failed: {error}"
            ) from error
        for stage in (
            Stage(
                "build_splits",
                "build_splits",
                (
                    *common,
                    "--output",
                    str(workspace / "dataset"),
                    "--confirm-production",
                ),
            ),
            Stage(
                "export_initial_checkpoint",
                "export_initial_checkpoint",
                (
                    *common,
                    "--output",
                    str(workspace / "initial_checkpoint.pt"),
                    "--confirm-production",
                ),
            ),
        ):
            _run_stage(
                stage,
                script_path=_script_parent_path(),
                job_id=invocation.job_id,
                bootstrap_lineage=bootstrap_lineage,
                bootstrap_science=bootstrap_science,
                code_commit=code_commit,
                gpu_count=invocation.gpu_count,
                allocation_sha256=invocation.allocation_sha256,
                execution_safety_class_id=certification_binding.execution_class_id,
                execution_safety_fingerprint_sha256=(
                    certification_binding.fingerprint_sha256
                ),
                execution_safety_descriptor_sha256=(
                    certification_binding.descriptor_sha256
                ),
                execution_safety_certification_sha256=(
                    certification_binding.certification_sha256
                ),
            )
        initial_hash = _sha256_file(workspace / "initial_checkpoint.pt")
        stages = _stage_plan(
            workspace,
            initial_checkpoint_sha256=initial_hash,
            gpu_count=invocation.gpu_count,
            science_overrides=science_binding.hydra_override_vector,
        )[2:]
        script_path = _script_parent_path()
        for stage in stages:
            if stage.name == "score_probe_candidates":
                stages = _replace_checkpoint_placeholders(stages, workspace)
                stage = next(item for item in stages if item.name == "score_probe_candidates")
            _run_stage(
                stage,
                script_path=script_path,
                job_id=invocation.job_id,
                bootstrap_lineage=bootstrap_lineage,
                bootstrap_science=bootstrap_science,
                code_commit=code_commit,
                gpu_count=invocation.gpu_count,
                allocation_sha256=invocation.allocation_sha256,
                execution_safety_class_id=certification_binding.execution_class_id,
                execution_safety_fingerprint_sha256=(
                    certification_binding.fingerprint_sha256
                ),
                execution_safety_descriptor_sha256=(
                    certification_binding.descriptor_sha256
                ),
                execution_safety_certification_sha256=(
                    certification_binding.certification_sha256
                ),
            )
        inner = _strict_json((workspace / "g0.json").read_bytes(), context="inner G0 report")
        inner_digest = inner.get("sha256")
        prompt_population_size = int(resolved["task"]["num_examples"])
        expected_batch_token_contract = batch_token_contract(
            invocation.gpu_count,
            prompt_population_size,
        )
        runtime_resolved_config_sha256 = inner.get("resolved_config_sha256")
        descriptor_payload = _strict_json(
            raw_inputs["execution_safety_descriptor_sha256"],
            context="execution-safety descriptor",
        )
        runtime_config = compose_config(
            [
                *common,
                f"production_safety.initial_checkpoint_hash={initial_hash}",
            ],
            config_root=SOURCE_ROOT / "configs",
        )
        resume_payload = validate_distributed_resume_report(
            workspace / "distributed_resume.json",
            config=runtime_config,
            expected_world_size=invocation.gpu_count,
        )
        execution_safety_attestation = build_runtime_execution_safety_attestation(
            config=runtime_config,
            descriptor=descriptor_payload,
            fingerprint_sha256=certification_binding.fingerprint_sha256,
            resume=resume_payload,
            world_size=invocation.gpu_count,
        )
        if (
            inner.get("passed") is not True
            or not isinstance(inner.get("checks"), dict)
            or tuple(sorted(inner["checks"])) != G0_CHECK_NAMES
            or any(value is not True for value in inner["checks"].values())
            or inner_digest != _sha256_value({key: value for key, value in inner.items() if key != "sha256"})
            or inner.get("protocol_amendment_id") != amendment_binding.amendment_id
            or inner.get("protocol_amendment_sha256") != amendment_binding.sha256
            or inner.get("execution_safety_class_id")
            != certification_binding.execution_class_id
            or inner.get("execution_safety_fingerprint_sha256")
            != certification_binding.fingerprint_sha256
            or inner.get("execution_safety_descriptor_sha256")
            != certification_binding.descriptor_sha256
            or inner.get("execution_safety_certification_sha256")
            != certification_binding.certification_sha256
            or inner.get("execution_science_protocol_id")
            != science_binding.protocol_id
            or inner.get("execution_science_protocol_sha256")
            != science_binding.artifact_sha256
            or inner.get("execution_science_protocol_git_commit")
            != bootstrap_science.science_protocol_git_commit
            or inner.get(
                "execution_science_protocol_reviewed_implementation_commit"
            )
            != science_binding.reviewed_implementation_commit
            or inner.get("reviewed_implementation_commit")
            != amendment_binding.reviewed_implementation_commit
            or inner.get("request_git_commit")
            != resolved["scheduler_g0"]["request_git_commit"]
            or inner.get("allocation_sha256") != invocation.allocation_sha256
            or inner.get("world_size") != invocation.gpu_count
            or inner.get("batch_token_contract")
            != expected_batch_token_contract
            or not isinstance(runtime_resolved_config_sha256, str)
            or SHA256.fullmatch(runtime_resolved_config_sha256) is None
            or _sha256_value(runtime_config) != runtime_resolved_config_sha256
            or execution_safety_attestation["batch_token_contract"]
            != expected_batch_token_contract
        ):
            raise G0Error("inner G0 semantic decision did not pass")
        inventory = _artifact_inventory(workspace)
        _build_bundle(workspace, bundle_temp, inventory)
        bundle_digest = _sha256_file(bundle_temp)
        completed_at = _utc_now()
        report: dict[str, Any] = {
            "artifact_bundle_sha256": bundle_digest,
            "artifact_inventory": inventory,
            "artifact_namespace": "qwen3-v2",
            "batch_token_contract": expected_batch_token_contract,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "code_commit": code_commit,
            "completed_at": completed_at,
            "cgroup_memory": _cgroup_memory(),
            "enable_thinking": False,
            "execution": invocation.execution,
            "execution_context": execution_context(invocation.gpu_count),
            "execution_safety_certification_sha256": (
                certification_binding.certification_sha256
            ),
            "execution_safety_class_id": certification_binding.execution_class_id,
            "execution_safety_descriptor_sha256": (
                certification_binding.descriptor_sha256
            ),
            "execution_safety_fingerprint_sha256": (
                certification_binding.fingerprint_sha256
            ),
            "execution_safety_attestation": execution_safety_attestation,
            "execution_science_protocol_git_commit": (
                bootstrap_science.science_protocol_git_commit
            ),
            "execution_science_protocol_id": science_binding.protocol_id,
            "execution_science_protocol_reviewed_implementation_commit": (
                science_binding.reviewed_implementation_commit
            ),
            "execution_science_protocol_sha256": science_binding.artifact_sha256,
            "git_commit": code_commit,
            "inner_g0_report_sha256": _sha256_file(workspace / "g0.json"),
            "model_revision": MODEL_REVISION,
            "passed": True,
            "phase": TASK,
            "prereg_commit": _git("log", "-n", "1", "--format=%H", "--", "prereg/qwen3_v2.yaml"),
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_sha256": PREREGISTRATION_SHA256,
            "prereg_version": "qwen3_v2",
            "protocol_amendment_git_commit": amendment_binding.git_commit,
            "protocol_amendment_id": amendment_binding.amendment_id,
            "protocol_amendment_sha256": amendment_binding.sha256,
            "provenance_validation": {
                "certification_artifacts_unchanged": True,
                "current_execution_safety_kernel_matches_descriptor": True,
                "execution_safety_fingerprint_exact": True,
            },
            "prompt_protocol": "qwen3_non_thinking_v1",
            "protocol_track": "qwen3_v2",
            "resolved_config_sha256": invocation.input_hashes["resolved_config_sha256"],
            "runtime_resolved_config_sha256": runtime_resolved_config_sha256,
            "request_git_commit": resolved["scheduler_g0"]["request_git_commit"],
            "schema_version": 2,
            "started_at": started_at,
            "reviewed_implementation_commit": (
                amendment_binding.reviewed_implementation_commit
            ),
            "teacher_revision": TEACHER_REVISION,
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "tokenizer_revision": MODEL_REVISION,
            "world_size": invocation.gpu_count,
        }
        report["sha256"] = _sha256_value(report)
        if _require_clean_git() != code_commit:
            raise G0Error("source checkout changed before G0 publication")
        published_bundle = _publish_file_once(invocation.output_descriptor, BUNDLE_NAME, bundle_temp)
        if published_bundle != bundle_digest:
            raise G0Error("published G0 artifact bundle changed during copy")
        _publish_json_once(invocation.output_descriptor, REPORT_NAME, report)
        _publish_json_once(
            invocation.output_descriptor,
            COMPLETION_NAME,
            _completion(invocation, started_at=started_at, completed_at=completed_at),
        )
        return 0
    finally:
        shutil.rmtree(temporary)


def main() -> int:
    try:
        arguments = sys.argv[1:]
        if arguments and arguments[0] == "--scientific-cli":
            return _scientific_cli(arguments[1:])
        return _supervise(arguments)
    except (OSError, G0Error, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f"OPD Qwen3-v2 G0 failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
