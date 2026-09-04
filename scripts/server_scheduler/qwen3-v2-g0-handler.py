#!/usr/bin/python3
"""Foreground, fixed two-GPU Qwen3-v2 G0 handler for protocol v2.

ServerScheduler owns the physical allocation.  This supervisor accepts only
adapter-held content descriptors, preserves CUDA visibility, runs the reviewed
scientific stages serially, and publishes one report plus a complete immutable
artifact tar.  Failed scheduler attempts never reuse a staging directory.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Sequence


TASK = "qwen3_v2_g0"
PROFILE = "qwen3-v2-g0-2gpu"
PROJECT = "OPD"
REPORT_NAME = "g0.json"
BUNDLE_NAME = "g0_artifacts.tar"
COMPLETION_NAME = ".opd-scientific-completion.json"
COMPLETION_KIND = "scientific_attempt"
SOURCE_ROOT = Path("/home/del6500/projects/OPD")
SOURCE_PACKAGE_ROOT = SOURCE_ROOT / "src"
FSDP_CONFIG = SOURCE_ROOT / "configs" / "accelerate" / "fsdp_2gpu_server_scheduler.yaml"
FSDP_CONFIG_SHA256 = "9315ae24072cfc7bb54fb78a18c633b05e89a4b6659b5a946dee783cfe47a609"
MIB_REPOSITORY = Path("/scr/del6500/OPD/vendor/MIB-circuit-track-v1")
MIB_REVISION = "b759df34433c9e31043ba9e02908ce0bf20e894f"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEACHER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_FINGERPRINT = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
PREREGISTRATION_SHA256 = "8d6bdeab0b9302c8824c4709f556c6c41a896bd2cfce21e7794d131d176ba0a4"
GPU_COUNT = 2
CPU_CORE_COUNT = 16
THREADS_PER_RANK = CPU_CORE_COUNT // GPU_COUNT
NODE_MEMORY_BYTES = 192 * 1024**3
MINIMUM_HEADROOM_BYTES = max(32 * 1024**3, int(NODE_MEMORY_BYTES * 0.20))
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_PREREG_BYTES = 4 * 1024 * 1024
MAX_AMENDMENT_BYTES = 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
PROC_SELF_FD = re.compile(r"/proc/self/fd/(0|[1-9][0-9]*)\Z")
POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")
EXPECTED_INPUT_NAMES = (
    "config_binding_sha256",
    "execution_config_sha256",
    "gpu_preflight_completion_sha256",
    "gpu_preflight_report_sha256",
    "preregistration_sha256",
    "protocol_amendment_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
)
YAML_INPUT_NAMES = frozenset(
    {"preregistration_sha256", "protocol_amendment_sha256"}
)
JSON_INPUT_NAMES = frozenset(EXPECTED_INPUT_NAMES) - YAML_INPUT_NAMES
PREFLIGHT_COMPLETION_FIELDS = frozenset(
    {
        "completed_at",
        "completion_kind",
        "execution",
        "execution_config_sha256",
        "input_hashes",
        "output_files",
        "plan_sha256",
        "project",
        "resolved_config_sha256",
        "run_id",
        "schema_version",
        "scientific_config_sha256",
        "scientific_validation",
        "sha256",
        "started_at",
        "task",
        "unit_id",
        "workflow_id",
    }
)
GATE_NAMES = (
    "allocation_contract",
    "artifact_bundle",
    "batch_token_invariants",
    "config_binding",
    "distributed_resume",
    "g0_semantic_decision",
    "gpu_preflight_binding",
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
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "TMPDIR": "/scr/del6500/OPD/tmp",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
}
BASE_CONFIG_OVERRIDES = (
    "g0=qwen3_v2_eap_separation",
    "experiment=canonical_sft",
    "task.num_examples=256",
    "state_source.num_candidates=8",
)
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
    manifest_sha256: str
    allocation_sha256: str
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
        "manifest-sha256",
        "allocation-sha256",
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
    if args.attempt_completion_name != COMPLETION_NAME:
        raise G0Error("attempt completion name differs from the fixed ABI")
    if args.execution_profile != PROFILE:
        raise G0Error("execution profile differs from the fixed G0 profile")
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
    output_descriptor = _held_descriptor(args.output_attempt_handle, name="output attempt handle")
    if output_descriptor in descriptors or not stat.S_ISDIR(os.fstat(output_descriptor).st_mode):
        raise G0Error("output attempt handle is invalid or aliases an input")
    return Invocation(
        workflow_id=_identifier(args.workflow_id, name="workflow_id"),
        plan_sha256=_sha256(args.plan_sha256, name="plan_sha256"),
        unit_id=_identifier(args.unit_id, name="unit_id"),
        run_id=_sha256(args.run_id, name="run_id"),
        job_id=_identifier(args.job_id, name="job_id"),
        attempt=int(args.attempt),
        execution_profile=PROFILE,
        manifest_sha256=_sha256(args.manifest_sha256, name="manifest_sha256"),
        allocation_sha256=_sha256(args.allocation_sha256, name="allocation_sha256"),
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
        payload = _strict_json(raw, context=f"content {handle.name!r}")
        if handle.name in JSON_INPUT_NAMES and raw != _canonical_json(payload).encode("utf-8"):
            # Config CAS uses canonical bytes without a trailing newline.  The
            # published preflight artifacts intentionally use durable newlines.
            if handle.name not in {
                "gpu_preflight_completion_sha256",
                "gpu_preflight_report_sha256",
            }:
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


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("/usr/bin/git", *arguments),
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if GIT_COMMIT.fullmatch(value) is None:
        raise G0Error("Git did not return one immutable commit identity")
    return value


def _require_clean_git() -> str:
    result = subprocess.run(
        ("/usr/bin/git", "status", "--porcelain", "--untracked-files=no"),
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        raise G0Error("G0 requires a clean tracked source checkout")
    return _git("rev-parse", "HEAD")


def _validate_environment() -> tuple[str, ...]:
    for key, expected in FIXED_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise G0Error(f"fixed environment {key} differs from its deployment")
    if any(os.environ.get(key) != str(THREADS_PER_RANK) for key in THREAD_KEYS):
        raise G0Error(f"per-rank CPU thread environment must be exactly {THREADS_PER_RANK}")
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
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = tuple(visible.split(",")) if visible else ()
    if len(devices) != GPU_COUNT or len(set(devices)) != GPU_COUNT or any(
        not value or value.strip() != value for value in devices
    ):
        raise G0Error(f"scheduler-provided CUDA visibility must contain {GPU_COUNT} devices")
    if _sha256_file(FSDP_CONFIG) != FSDP_CONFIG_SHA256:
        raise G0Error("two-GPU FSDP configuration differs from its reviewed digest")
    if not MIB_REPOSITORY.is_dir() or MIB_REPOSITORY.is_symlink():
        raise G0Error("fixed MIB checkout is absent or unsafe")
    mib_commit = subprocess.run(
        ("/usr/bin/git", "-C", str(MIB_REPOSITORY), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if mib_commit != MIB_REVISION:
        raise G0Error("fixed MIB checkout differs from its reviewed revision")
    mib_status = subprocess.run(
        ("/usr/bin/git", "-C", str(MIB_REPOSITORY), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    submodules = subprocess.run(
        ("/usr/bin/git", "-C", str(MIB_REPOSITORY), "submodule", "status", "--recursive"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if mib_status or not submodules or any(
        not row or row[0] != " " for row in submodules
    ):
        raise G0Error("fixed MIB checkout or submodule tree is dirty or incomplete")
    if not (MIB_REPOSITORY / "run_attribution.py").is_file():
        raise G0Error("fixed MIB checkout lacks its reviewed entrypoint")
    return devices


def _validate_config_and_preflight(
    invocation: Invocation,
    payloads: dict[str, dict[str, Any]],
    prereg: bytes,
    amendment: bytes,
    raw_inputs: dict[str, bytes],
    *,
    code_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    _install_source_path()
    from posttrain_circuits.artifacts.config_bindings import ConfigBinding, validate_config_binding
    from posttrain_circuits.artifacts.protocol_amendments import (
        AMENDMENT_ID,
        AMENDMENT_RELATIVE_PATH,
        load_protocol_amendment_bytes,
        resolve_accepted_protocol_amendment,
        validate_accepted_lineage_commit,
        validate_two_gpu_g0_config,
    )
    from posttrain_circuits.core.config import compose_config
    from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
        GATE_NAMES as preflight_gates,
        PROFILE_NAME as preflight_profile,
        _validate_report as validate_preflight_report,
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
        "gpu_preflight_completion": invocation.input_hashes["gpu_preflight_completion_sha256"],
        "gpu_preflight_report": invocation.input_hashes["gpu_preflight_report_sha256"],
        "prereg_path": invocation.input_hashes["preregistration_sha256"],
        "protocol_amendment_path": invocation.input_hashes[
            "protocol_amendment_sha256"
        ],
    }
    if binding_payload.get("input_artifact_hashes") != expected_inputs:
        raise G0Error("ConfigBinding does not bind the exact G0 prerequisite bytes")
    expected_execution = {
        "distributed_process_count": GPU_COUNT,
        "execution_profile": PROFILE,
        "scheduler_protocol": 2,
    }
    if binding_payload.get("execution_context") != expected_execution:
        raise G0Error("ConfigBinding execution context differs from the fixed topology")
    if execution.get("execution_context") != expected_execution:
        raise G0Error("G0 execution projection differs from the fixed topology")
    if (
        hashlib.sha256(prereg).hexdigest() != PREREGISTRATION_SHA256
        or invocation.input_hashes["preregistration_sha256"] != PREREGISTRATION_SHA256
    ):
        raise G0Error("preregistration content identity changed")
    amendment_payload = load_protocol_amendment_bytes(amendment)
    source_amendment = SOURCE_ROOT / AMENDMENT_RELATIVE_PATH
    if source_amendment.read_bytes() != amendment:
        raise G0Error("protocol amendment content differs from the clean source commit")
    amendment_binding = resolve_accepted_protocol_amendment(
        code_root=SOURCE_ROOT,
        configured_path=str(AMENDMENT_RELATIVE_PATH),
        expected_head=code_commit,
    )
    if (
        amendment_payload["review"]["status"] != "accepted"
        or hashlib.sha256(amendment).hexdigest() != amendment_binding.sha256
        or invocation.input_hashes["protocol_amendment_sha256"]
        != amendment_binding.sha256
    ):
        raise G0Error("protocol amendment is not the accepted Git-reviewed content")

    scheduler_config = resolved.get("scheduler_g0")
    if not isinstance(scheduler_config, dict):
        raise G0Error("resolved config lacks scheduler G0 provenance")
    request_git_commit = scheduler_config.get("request_git_commit")
    preflight_git_commit = scheduler_config.get("gpu_preflight_git_commit")
    if (
        not isinstance(request_git_commit, str)
        or GIT_COMMIT.fullmatch(request_git_commit) is None
        or not isinstance(preflight_git_commit, str)
        or GIT_COMMIT.fullmatch(preflight_git_commit) is None
    ):
        raise G0Error("resolved config contains invalid request/preflight Git provenance")
    try:
        validate_accepted_lineage_commit(
            code_root=SOURCE_ROOT,
            candidate_commit=request_git_commit,
            current_binding=amendment_binding,
            expected_head=code_commit,
            role="G0 request commit",
        )
        validate_accepted_lineage_commit(
            code_root=SOURCE_ROOT,
            candidate_commit=preflight_git_commit,
            current_binding=amendment_binding,
            expected_head=code_commit,
            role="GPU preflight commit",
        )
    except ValueError as error:
        raise G0Error(f"G0 implementation lineage is invalid: {error}") from error

    expected = compose_config(list(BASE_CONFIG_OVERRIDES))
    expected["scheduler_g0"] = {
        "artifact_namespace": "qwen3-v2",
        "execution_profile": PROFILE,
        "gpu_preflight_git_commit": preflight_git_commit,
        "gpu_preflight_completion_sha256": invocation.input_hashes[
            "gpu_preflight_completion_sha256"
        ],
        "gpu_preflight_report_sha256": invocation.input_hashes["gpu_preflight_report_sha256"],
        "model_revision": MODEL_REVISION,
        "process_count": GPU_COUNT,
        "protocol_amendment_id": AMENDMENT_ID,
        "protocol_amendment_sha256": amendment_binding.sha256,
        "request_git_commit": request_git_commit,
        "reviewed_implementation_commit": (
            amendment_binding.reviewed_implementation_commit
        ),
        "task": TASK,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
    }
    if resolved != expected:
        raise G0Error("resolved config differs from the fixed Qwen3-v2 G0 protocol")
    validate_two_gpu_g0_config(resolved, amendment_payload)
    if _sha256_value(resolved) != invocation.input_hashes["resolved_config_sha256"]:
        raise G0Error("resolved G0 config hash is invalid")
    if scientific.get("input_artifact_hashes") != expected_inputs:
        raise G0Error("scientific config projection lost prerequisite identities")

    preflight = payloads["gpu_preflight_report_sha256"]
    completion = payloads["gpu_preflight_completion_sha256"]
    validate_preflight_report(
        preflight,
        resolved_config_sha256=str(preflight.get("resolved_config_sha256")),
        preregistration_sha256=PREREGISTRATION_SHA256,
    )
    preflight_execution = completion.get("execution")
    preflight_outputs = completion.get("output_files")
    preflight_validation = completion.get("scientific_validation")
    preflight_inputs = completion.get("input_hashes")
    completion_digest = completion.get("sha256")
    completion_unsigned = {
        key: value for key, value in completion.items() if key != "sha256"
    }
    if (
        set(completion) != PREFLIGHT_COMPLETION_FIELDS
        or completion_digest != _sha256_value(completion_unsigned)
        or completion.get("schema_version") != 1
        or completion.get("completion_kind") != "scientific"
        or completion.get("project") != PROJECT
        or preflight.get("git_commit") != preflight_git_commit
        or completion.get("task") != "qwen3_v2_gpu_preflight"
        or completion.get("workflow_id") != "qwen3-v2-gpu-preflight-v1"
        or completion.get("unit_id") != "gpu-preflight"
        or not isinstance(preflight_execution, dict)
        or preflight_execution.get("execution_profile") != preflight_profile
        or not isinstance(preflight_outputs, dict)
        or len(preflight_outputs) != 1
        or next(iter(preflight_outputs.values()), None)
        != hashlib.sha256(raw_inputs["gpu_preflight_report_sha256"]).hexdigest()
        or not isinstance(preflight_validation, dict)
        or tuple(sorted(preflight_validation)) != preflight_gates
        or any(value is not True for value in preflight_validation.values())
        or not isinstance(preflight_inputs, dict)
        or preflight_inputs.get("resolved_config_sha256")
        != preflight.get("resolved_config_sha256")
        or preflight_inputs.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or completion.get("resolved_config_sha256")
        != preflight.get("resolved_config_sha256")
        or completion.get("execution_config_sha256")
        != preflight_inputs.get("execution_config_sha256")
        or completion.get("scientific_config_sha256")
        != preflight_inputs.get("scientific_config_sha256")
    ):
        raise G0Error(
            "G0 prerequisite is not a successful accepted-lineage GPU preflight"
        )
    return resolved, preflight, amendment_binding


def _common_overrides(root: Path, *, experiment: str = "canonical_sft") -> list[str]:
    dataset = root / "dataset"
    initial = root / "initial_checkpoint.pt"
    demos = root / "teacher_demos"
    probes = root / "probes" / "manifest.json"
    readiness = root / "readiness" / "readiness.json"
    return [
        "g0=qwen3_v2_eap_separation",
        f"experiment={experiment}",
        "task.num_examples=256",
        "state_source.num_candidates=8",
        f"output_root={root}",
        f"task.dataset_family_path={dataset}",
        f"state_source.store_path={demos}",
        f"anti_shortcut.report_path={root / 'anti_shortcut.json'}",
        f"production_safety.readiness_report={readiness}",
        f"production_safety.probe_cohort_manifest={probes}",
        f"production_safety.initial_checkpoint_path={initial}",
    ]


def _stage_plan(root: Path, *, initial_checkpoint_sha256: str) -> tuple[Stage, ...]:
    common = _common_overrides(root)
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
                *_common_overrides(root, experiment="offline_soft"),
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
                    "--checkpoint-a",
                    str(root / "resume-a" / "FINAL_CHECKPOINT"),
                    "--checkpoint-b",
                    str(root / "resume-b" / "FINAL_CHECKPOINT"),
                    "--metrics-a",
                    str(root / "resume-a" / "metrics.jsonl"),
                    "--metrics-b",
                    str(root / "resume-b" / "metrics.jsonl"),
                    "--world-size",
                    str(GPU_COUNT),
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
                    "--compatibility",
                    str(circuits / "final_answer" / "mib_raw" / "compatibility.json"),
                    "--distributed-resume",
                    str(root / "distributed_resume.json"),
                    "--initial-checkpoint",
                    str(initial),
                    "--gpu-preflight",
                    str(root / "gpu_preflight.json"),
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
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SOURCE_PACKAGE_ROOT)
    return environment


def _run_stage(stage: Stage, *, script_path: str, job_id: str) -> None:
    argv = tuple(job_id if value == "JOB_ID" else value for value in stage.argv)
    command = [sys.executable, "-I"]
    if stage.distributed:
        command.extend(
            (
                "-m",
                "accelerate.commands.launch",
                "--config_file",
                str(FSDP_CONFIG),
                script_path,
                "--scientific-cli",
                stage.cli,
                "--",
                *argv,
            )
        )
    else:
        command.extend((script_path, "--scientific-cli", stage.cli, "--", *argv))
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
    payload = _strict_json(manifest_path.read_bytes(), context=f"{run_root.name} run manifest")
    candidate = Path(str(payload.get("final_checkpoint_path", "")))
    expected = payload.get("final_checkpoint_sha256")
    if not candidate.is_file() or not isinstance(expected, str) or _sha256_file(candidate) != expected:
        raise G0Error(f"{run_root.name} final checkpoint binding is missing or invalid")
    return candidate


def _replace_checkpoint_placeholders(stages: tuple[Stage, ...], root: Path) -> tuple[Stage, ...]:
    calibration = str(_resolve_final_checkpoint(root / "calibration"))
    resume_a = (
        str(_resolve_final_checkpoint(root / "resume-a"))
        if (root / "resume-a" / "manifest.json").is_file()
        else ""
    )
    resume_b = (
        str(_resolve_final_checkpoint(root / "resume-b"))
        if (root / "resume-b" / "manifest.json").is_file()
        else ""
    )
    replacements = {
        str(root / "calibration" / "FINAL_CHECKPOINT"): calibration,
        str(root / "resume-a" / "FINAL_CHECKPOINT"): resume_a,
        str(root / "resume-b" / "FINAL_CHECKPOINT"): resume_b,
    }
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
    if len(argv) < 2 or argv[1] != "--" or argv[0] not in SCIENTIFIC_CLIS:
        raise G0Error("scientific child invocation differs from the fixed CLI ABI")
    _install_source_path()
    module = importlib.import_module(SCIENTIFIC_CLIS[argv[0]])
    main_function = getattr(module, "main", None)
    if not callable(main_function):
        raise G0Error("reviewed scientific CLI has no callable main")
    main_function(list(argv[2:]))
    return 0


def _supervise(argv: Sequence[str] | None) -> int:
    started_at = _utc_now()
    invocation = _parse_outer(argv)
    _validate_environment()
    code_commit = _require_clean_git()
    payloads, prereg, amendment, raw_inputs = _read_inputs(invocation)
    resolved, preflight, amendment_binding = _validate_config_and_preflight(
        invocation,
        payloads,
        prereg,
        amendment,
        raw_inputs,
        code_commit=code_commit,
    )
    if os.listdir(invocation.output_descriptor):
        raise G0Error("output attempt is not empty")
    temp_root = Path(FIXED_ENVIRONMENT["TMPDIR"])
    temp_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="qwen3-v2-g0-", dir=temp_root))
    workspace = temporary / "qwen3-v2"
    workspace.mkdir(mode=0o750)
    bundle_temp = temporary / ".g0-artifacts.tar"
    try:
        (workspace / "gpu_preflight.json").write_bytes(raw_inputs["gpu_preflight_report_sha256"])
        common = _common_overrides(workspace)
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
            _run_stage(stage, script_path=_script_parent_path(), job_id=invocation.job_id)
        initial_hash = _sha256_file(workspace / "initial_checkpoint.pt")
        stages = _stage_plan(workspace, initial_checkpoint_sha256=initial_hash)[2:]
        script_path = _script_parent_path()
        for stage in stages:
            if stage.name == "score_probe_candidates":
                stages = _replace_checkpoint_placeholders(stages, workspace)
                stage = next(item for item in stages if item.name == "score_probe_candidates")
            elif stage.name == "compare_distributed_resume":
                stages = _replace_checkpoint_placeholders(stages, workspace)
                stage = next(item for item in stages if item.name == "compare_distributed_resume")
            _run_stage(stage, script_path=script_path, job_id=invocation.job_id)
        inner = _strict_json((workspace / "g0.json").read_bytes(), context="inner G0 report")
        inner_digest = inner.get("sha256")
        if (
            inner.get("passed") is not True
            or not isinstance(inner.get("checks"), dict)
            or any(value is not True for value in inner["checks"].values())
            or inner_digest != _sha256_value({key: value for key, value in inner.items() if key != "sha256"})
            or inner.get("protocol_amendment_id") != amendment_binding.amendment_id
            or inner.get("protocol_amendment_sha256") != amendment_binding.sha256
            or inner.get("reviewed_implementation_commit")
            != amendment_binding.reviewed_implementation_commit
            or inner.get("request_git_commit")
            != resolved["scheduler_g0"]["request_git_commit"]
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
            "batch_token_contract": {
                "world_size": GPU_COUNT,
                "per_device_batch_size": resolved["trainer"]["batch_size"],
                "gradient_accumulation_steps": resolved["trainer"][
                    "gradient_accumulation_steps"
                ],
                "effective_global_batch_size": (
                    GPU_COUNT
                    * resolved["trainer"]["batch_size"]
                    * resolved["trainer"]["gradient_accumulation_steps"]
                ),
                "token_budget": resolved["trainer"]["token_budget"],
                "token_budget_unit": resolved["trainer"]["token_budget_unit"],
                "max_optimizer_steps": resolved["trainer"]["max_steps"],
            },
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "code_commit": code_commit,
            "completed_at": completed_at,
            "cgroup_memory": _cgroup_memory(),
            "enable_thinking": False,
            "execution": invocation.execution,
            "execution_context": {
                "allocation_visibility": "preserved",
                "distributed_launcher": "accelerate_fsdp_foreground",
                "mode": "server_scheduler_foreground",
                "nccl_p2p_policy": "disabled",
                "visible_device_count": GPU_COUNT,
            },
            "git_commit": code_commit,
            "gpu_preflight_completion_sha256": invocation.input_hashes[
                "gpu_preflight_completion_sha256"
            ],
            "gpu_preflight_git_commit": preflight["git_commit"],
            "gpu_preflight_report_sha256": invocation.input_hashes[
                "gpu_preflight_report_sha256"
            ],
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
                "preflight_to_execution_metadata_only": True,
                "request_to_execution_metadata_only": True,
                "reviewed_implementation_to_execution_metadata_only": True,
            },
            "prompt_protocol": "qwen3_non_thinking_v1",
            "protocol_track": "qwen3_v2",
            "resolved_config_sha256": invocation.input_hashes["resolved_config_sha256"],
            "request_git_commit": resolved["scheduler_g0"]["request_git_commit"],
            "schema_version": 1,
            "started_at": started_at,
            "reviewed_implementation_commit": (
                amendment_binding.reviewed_implementation_commit
            ),
            "teacher_revision": TEACHER_REVISION,
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "tokenizer_revision": MODEL_REVISION,
            "world_size": GPU_COUNT,
        }
        report["sha256"] = _sha256_value(report)
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
