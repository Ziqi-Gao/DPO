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
AMENDMENT_RELATIVE_PATH = Path("prereg/amendments/qwen3_v2_g0_elastic_v1.yaml")
HANDOFF_RELATIVE_PATH = "docs/refactor/current_handoff.md"
PROPOSED_AMENDMENT_SHA256 = (
    "2d2444c9b2969b0b10a42d137184f8d11574d748ae488f6e40d5b5cc4fb6becc"
)
PROPOSED_REVIEW = {
    "status": "proposed",
    "reviewed_implementation_commit": None,
    "reviewer": None,
    "reviewed_at_utc": None,
    "rationale": None,
}
ALLOWED_REVIEW_PATHS = frozenset({str(AMENDMENT_RELATIVE_PATH), HANDOFF_RELATIVE_PATH})
ALLOWED_HANDOFF_PATHS = frozenset({HANDOFF_RELATIVE_PATH})
MAX_LINEAGE_COMMITS = 256
ALLOWED_GPU_COUNTS = (1, 2, 3, 4)
CPU_CORE_COUNT = 24
THREADS_PER_RANK = {count: CPU_CORE_COUNT // count for count in ALLOWED_GPU_COUNTS}
BATCH_PARTITION_PROTOCOL = "allocation_neutral_exact_global_batch_v1"
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


def _preflight_completion_name(world_size: int) -> str:
    return f"gpu_preflight_w{world_size}_completion_sha256"


def _preflight_report_name(world_size: int) -> str:
    return f"gpu_preflight_w{world_size}_report_sha256"


EXPECTED_INPUT_NAMES = (
    "config_binding_sha256",
    "execution_config_sha256",
    *tuple(
        name
        for world_size in ALLOWED_GPU_COUNTS
        for name in (
            _preflight_completion_name(world_size),
            _preflight_report_name(world_size),
        )
    ),
    "preregistration_sha256",
    "protocol_amendment_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
)
YAML_INPUT_NAMES = frozenset({"preregistration_sha256", "protocol_amendment_sha256"})
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
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "TMPDIR": "/scr/del6500/OPD/tmp",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
}
_PROCESS_PYCACHE_PREFIX: str | None = None
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
    gpu_count: int
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
class BootstrapLineage:
    amendment_sha256: str
    amendment_git_commit: str
    reviewed_implementation_commit: str
    request_git_commit: str
    preflight_git_commits: tuple[str, str, str, str]

    def preflight_git_commit(self, world_size: int) -> str:
        if world_size not in ALLOWED_GPU_COUNTS:
            raise G0Error("preflight lineage world size is outside 1..4")
        return self.preflight_git_commits[world_size - 1]


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
        gpu_count=int(args.gpu_count),
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
            if not (
                handle.name.startswith("gpu_preflight_w")
                and handle.name.endswith(
                    ("_completion_sha256", "_report_sha256")
                )
            ):
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
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        raise G0Error("G0 requires a clean source checkout")
    unsafe = _unsafe_untracked_paths()
    if unsafe:
        raise G0Error(f"G0 checkout contains unsafe ignored files: {unsafe!r}")
    return _git("rev-parse", "HEAD")


def _git_amendment_bytes(commit: str) -> bytes:
    if GIT_COMMIT.fullmatch(commit) is None:
        raise G0Error("amendment lookup commit is invalid")
    return _git_bytes("show", f"{commit}:{AMENDMENT_RELATIVE_PATH}")


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


def _linear_commit_path(ancestor: str, descendant: str, *, role: str) -> tuple[str, ...]:
    if not _is_ancestor(ancestor, descendant):
        raise G0Error(f"{role} does not descend from the reviewed implementation")
    commits: list[str] = []
    cursor = descendant
    while cursor != ancestor:
        if len(commits) >= MAX_LINEAGE_COMMITS:
            raise G0Error(f"{role} lineage exceeds the reviewed bound")
        parents = _commit_parents(cursor)
        if len(parents) != 1:
            raise G0Error(f"{role} lineage is not a single-parent chain")
        commits.append(cursor)
        cursor = parents[0]
    commits.reverse()
    return tuple(commits)


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


def _accepted_review(
    accepted_raw: bytes,
    proposed_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    accepted, implementation = _accepted_review_metadata(accepted_raw)
    normalized = dict(accepted)
    normalized["review"] = dict(PROPOSED_REVIEW)
    if normalized != proposed_payload:
        raise G0Error("accepted protocol amendment changed reviewed scientific terms")
    return accepted, implementation


def _validate_accepted_commit_chain(
    *,
    implementation: str,
    endpoint: str,
    accepted_raw: bytes,
    role: str,
) -> str:
    proposed_raw = _git_amendment_bytes(implementation)
    if hashlib.sha256(proposed_raw).hexdigest() != PROPOSED_AMENDMENT_SHA256:
        raise G0Error(f"{role} reviewed implementation amendment bytes differ")
    proposed = _bootstrap_yaml(proposed_raw, context="reviewed proposed amendment")
    if proposed.get("review") != PROPOSED_REVIEW:
        raise G0Error(f"{role} implementation did not contain the proposed amendment")
    _, reviewed_implementation = _accepted_review(accepted_raw, proposed)
    if reviewed_implementation != implementation:
        raise G0Error(f"{role} accepted amendment names another implementation")

    transition_commit: str | None = None
    parent = implementation
    for commit in _linear_commit_path(implementation, endpoint, role=role):
        changed = _changed_paths(parent, commit)
        if not changed:
            raise G0Error(f"{role} contains an empty commit")
        if str(AMENDMENT_RELATIVE_PATH) in changed:
            if transition_commit is not None or not changed <= ALLOWED_REVIEW_PATHS:
                raise G0Error(f"{role} contains an invalid amendment transition")
            if _git_amendment_bytes(commit) != accepted_raw:
                raise G0Error(f"{role} amendment transition differs from accepted bytes")
            transition_commit = commit
        elif not changed <= ALLOWED_HANDOFF_PATHS:
            raise G0Error(f"{role} contains a non-handoff implementation change")
        parent = commit
    if transition_commit is None or _git_amendment_bytes(endpoint) != accepted_raw:
        raise G0Error(f"{role} does not contain exactly one accepted review transition")
    return transition_commit


def _validate_handoff_only_path(ancestor: str, descendant: str, *, role: str) -> None:
    parent = ancestor
    for commit in _linear_commit_path(ancestor, descendant, role=role):
        changed = _changed_paths(parent, commit)
        if not changed or not changed <= ALLOWED_HANDOFF_PATHS:
            raise G0Error(f"{role} contains a post-acceptance implementation change")
        parent = commit


def _bootstrap_accepted_lineage(
    *,
    code_commit: str,
    resolved: dict[str, Any],
    amendment: bytes,
) -> BootstrapLineage:
    if _git("rev-parse", "HEAD") != code_commit:
        raise G0Error("trusted lineage HEAD changed before source import")
    source_path = SOURCE_ROOT / AMENDMENT_RELATIVE_PATH
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source_path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise G0Error("protocol amendment must be one non-linked regular file")
        source_raw = _read_held_file(
            descriptor,
            context="source protocol amendment",
            max_bytes=MAX_AMENDMENT_BYTES,
        )
        if source_raw != amendment or _git_amendment_bytes(code_commit) != source_raw:
            raise G0Error("held amendment differs from the clean source checkout")
        _, implementation = _accepted_review_metadata(source_raw)
        proposed_payload = _bootstrap_yaml(
            _git_amendment_bytes(implementation),
            context="reviewed proposed amendment",
        )
        _, implementation = _accepted_review(source_raw, proposed_payload)
        amendment_commit = _validate_accepted_commit_chain(
            implementation=implementation,
            endpoint=code_commit,
            accepted_raw=source_raw,
            role="execution HEAD",
        )

        scheduler = resolved.get("scheduler_g0")
        if not isinstance(scheduler, dict):
            raise G0Error("resolved config lacks scheduler G0 provenance")
        request_commit = scheduler.get("request_git_commit")
        preflight_matrix = scheduler.get("gpu_preflight_matrix")
        if (
            not isinstance(request_commit, str)
            or GIT_COMMIT.fullmatch(request_commit) is None
            or not isinstance(preflight_matrix, dict)
            or set(preflight_matrix) != {str(value) for value in ALLOWED_GPU_COUNTS}
        ):
            raise G0Error("resolved config contains invalid request/preflight Git provenance")
        preflight_commits: list[str] = []
        candidates = [(request_commit, "G0 request commit")]
        for world_size in ALLOWED_GPU_COUNTS:
            row = preflight_matrix[str(world_size)]
            if not isinstance(row, dict):
                raise G0Error("resolved config preflight matrix row is invalid")
            candidate = row.get("git_commit")
            if not isinstance(candidate, str) or GIT_COMMIT.fullmatch(candidate) is None:
                raise G0Error("resolved config contains invalid preflight Git provenance")
            preflight_commits.append(candidate)
            candidates.append((candidate, f"GPU preflight W={world_size} commit"))
        for candidate, role in candidates:
            candidate_amendment_commit = _validate_accepted_commit_chain(
                implementation=implementation,
                endpoint=candidate,
                accepted_raw=source_raw,
                role=role,
            )
            if candidate_amendment_commit != amendment_commit:
                raise G0Error(f"{role} names a different acceptance transition")
            _validate_handoff_only_path(candidate, code_commit, role=f"{role} to execution")

        after = os.fstat(descriptor)
        path_after = source_path.lstat()
        identity = lambda row: (  # noqa: E731
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if (
            _require_clean_git() != code_commit
            or _read_held_file(
                descriptor,
                context="source protocol amendment",
                max_bytes=MAX_AMENDMENT_BYTES,
            )
            != source_raw
            or identity(before) != identity(after)
            or identity(after) != identity(path_after)
            or _git_amendment_bytes(code_commit) != source_raw
        ):
            raise G0Error("trusted lineage changed during bootstrap validation")
        return BootstrapLineage(
            amendment_sha256=hashlib.sha256(source_raw).hexdigest(),
            amendment_git_commit=amendment_commit,
            reviewed_implementation_commit=implementation,
            request_git_commit=request_commit,
            preflight_git_commits=(
                preflight_commits[0],
                preflight_commits[1],
                preflight_commits[2],
                preflight_commits[3],
            ),
        )
    finally:
        os.close(descriptor)


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


def _validate_config_and_preflight(
    invocation: Invocation,
    payloads: dict[str, dict[str, Any]],
    prereg: bytes,
    amendment: bytes,
    raw_inputs: dict[str, bytes],
    *,
    code_commit: str,
    bootstrap_lineage: BootstrapLineage,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], Any]:
    from posttrain_circuits.artifacts.config_bindings import ConfigBinding, validate_config_binding
    from posttrain_circuits.artifacts.protocol_amendments import (
        AMENDMENT_ID,
        AMENDMENT_RELATIVE_PATH,
        load_protocol_amendment_bytes,
        resolve_accepted_protocol_amendment,
        validate_accepted_lineage_commit,
        validate_elastic_g0_config,
    )
    from posttrain_circuits.core.config import compose_config
    from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
        GATE_NAMES as preflight_gates,
        PROFILE_NAME as preflight_profile,
        _validate_report as validate_preflight_report,
        validate_preflight_workflow_id,
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
        **{
            f"gpu_preflight_w{world_size}_completion": invocation.input_hashes[
                _preflight_completion_name(world_size)
            ]
            for world_size in ALLOWED_GPU_COUNTS
        },
        **{
            f"gpu_preflight_w{world_size}_report": invocation.input_hashes[
                _preflight_report_name(world_size)
            ]
            for world_size in ALLOWED_GPU_COUNTS
        },
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
        amendment_binding.sha256 != bootstrap_lineage.amendment_sha256
        or amendment_binding.git_commit != bootstrap_lineage.amendment_git_commit
        or amendment_binding.reviewed_implementation_commit
        != bootstrap_lineage.reviewed_implementation_commit
        or amendment_payload["review"]["status"] != "accepted"
        or hashlib.sha256(amendment).hexdigest() != amendment_binding.sha256
        or invocation.input_hashes["protocol_amendment_sha256"] != amendment_binding.sha256
    ):
        raise G0Error("protocol amendment is not the accepted Git-reviewed content")

    preflights: dict[int, dict[str, Any]] = {}
    preflight_matrix: dict[str, dict[str, Any]] = {}
    seen_allocations: set[str] = set()
    seen_evidence: set[str] = set()
    seen_workflows: set[str] = set()
    for world_size in ALLOWED_GPU_COUNTS:
        report_name = _preflight_report_name(world_size)
        completion_name = _preflight_completion_name(world_size)
        preflight = payloads[report_name]
        completion = payloads[completion_name]
        preflight_execution = completion.get("execution")
        preflight_outputs = completion.get("output_files")
        preflight_validation = completion.get("scientific_validation")
        preflight_inputs = completion.get("input_hashes")
        completion_digest = completion.get("sha256")
        completion_unsigned = {
            key: value for key, value in completion.items() if key != "sha256"
        }
        if not isinstance(preflight_execution, dict):
            raise G0Error("GPU preflight completion execution is invalid")
        validate_preflight_report(
            preflight,
            resolved_config_sha256=str(preflight.get("resolved_config_sha256")),
            preregistration_sha256=PREREGISTRATION_SHA256,
            completion_execution=preflight_execution,
        )
        report_digest = invocation.input_hashes[report_name]
        completion_file_digest = invocation.input_hashes[completion_name]
        preflight_commit = preflight.get("git_commit")
        allocation_digest = preflight_execution.get("allocation_sha256")
        preflight_workflow_id = validate_preflight_workflow_id(
            completion.get("workflow_id")
        )
        if (
            set(completion) != PREFLIGHT_COMPLETION_FIELDS
            or completion_digest != _sha256_value(completion_unsigned)
            or completion.get("schema_version") != 1
            or completion.get("completion_kind") != "scientific"
            or completion.get("project") != PROJECT
            or preflight.get("world_size") != world_size
            or preflight_commit != bootstrap_lineage.preflight_git_commit(world_size)
            or completion.get("task") != "qwen3_v2_gpu_preflight"
            or completion.get("workflow_id") != preflight_workflow_id
            or completion.get("unit_id") != "gpu-preflight"
            or preflight_execution.get("execution_profile") != preflight_profile
            or not isinstance(allocation_digest, str)
            or SHA256.fullmatch(allocation_digest) is None
            or not isinstance(preflight_outputs, dict)
            or len(preflight_outputs) != 1
            or next(iter(preflight_outputs.values()), None)
            != hashlib.sha256(raw_inputs[report_name]).hexdigest()
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
            or allocation_digest in seen_allocations
            or preflight_workflow_id in seen_workflows
            or report_digest in seen_evidence
            or completion_file_digest in seen_evidence
        ):
            raise G0Error(
                "G0 prerequisite matrix is not four distinct successful preflights"
            )
        seen_allocations.add(allocation_digest)
        seen_workflows.add(preflight_workflow_id)
        seen_evidence.update((report_digest, completion_file_digest))
        preflights[world_size] = preflight
        preflight_matrix[str(world_size)] = {
            "allocation_sha256": allocation_digest,
            "completion_sha256": completion_file_digest,
            "git_commit": preflight_commit,
            "report_sha256": report_digest,
            "workflow_id": preflight_workflow_id,
            "world_size": world_size,
        }

    scheduler_config = resolved.get("scheduler_g0")
    if not isinstance(scheduler_config, dict):
        raise G0Error("resolved config lacks scheduler G0 provenance")
    request_git_commit = scheduler_config.get("request_git_commit")
    if (
        not isinstance(request_git_commit, str)
        or GIT_COMMIT.fullmatch(request_git_commit) is None
        or request_git_commit != bootstrap_lineage.request_git_commit
        or scheduler_config.get("gpu_preflight_matrix") != preflight_matrix
    ):
        raise G0Error("resolved config contains invalid request/preflight provenance")
    try:
        validate_accepted_lineage_commit(
            code_root=SOURCE_ROOT,
            candidate_commit=request_git_commit,
            current_binding=amendment_binding,
            expected_head=code_commit,
            role="G0 request commit",
        )
        for world_size, preflight in preflights.items():
            validate_accepted_lineage_commit(
                code_root=SOURCE_ROOT,
                candidate_commit=str(preflight["git_commit"]),
                current_binding=amendment_binding,
                expected_head=code_commit,
                role=f"GPU preflight W={world_size} commit",
            )
    except ValueError as error:
        raise G0Error(f"G0 implementation lineage is invalid: {error}") from error

    expected = compose_config(list(BASE_CONFIG_OVERRIDES))
    expected["scheduler_g0"] = {
        "allocation_contract": "manifest_driven_scheduler_gpu_v1",
        "artifact_namespace": "qwen3-v2",
        "batch_partition_protocol": BATCH_PARTITION_PROTOCOL,
        "gpu_preflight_matrix": preflight_matrix,
        "model_revision": MODEL_REVISION,
        "protocol_amendment_id": AMENDMENT_ID,
        "protocol_amendment_sha256": amendment_binding.sha256,
        "request_git_commit": request_git_commit,
        "reviewed_implementation_commit": amendment_binding.reviewed_implementation_commit,
        "task": TASK,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
    }
    if resolved != expected:
        raise G0Error("resolved config differs from the allocation-neutral G0 protocol")
    validate_elastic_g0_config(resolved, amendment_payload)
    if _sha256_value(resolved) != invocation.input_hashes["resolved_config_sha256"]:
        raise G0Error("resolved G0 config hash is invalid")
    if scientific.get("input_artifact_hashes") != expected_inputs:
        raise G0Error("scientific config projection lost prerequisite identities")
    return resolved, preflights, amendment_binding


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


def _stage_plan(
    root: Path,
    *,
    initial_checkpoint_sha256: str,
    gpu_count: int,
) -> tuple[Stage, ...]:
    if gpu_count not in ALLOWED_GPU_COUNTS:
        raise G0Error("G0 stage plan GPU count is outside 1..4")
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
    return dict(os.environ)


def _run_stage(
    stage: Stage,
    *,
    script_path: str,
    job_id: str,
    bootstrap_lineage: BootstrapLineage,
    code_commit: str,
    gpu_count: int,
    allocation_sha256: str,
) -> None:
    if gpu_count not in ALLOWED_GPU_COUNTS or SHA256.fullmatch(allocation_sha256) is None:
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
        "--preflight-git-commit",
        bootstrap_lineage.preflight_git_commit(gpu_count),
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
        len(argv) < 16
        or argv[0] not in SCIENTIFIC_CLIS
        or argv[1] != "--code-commit"
        or argv[3] != "--request-git-commit"
        or argv[5] != "--preflight-git-commit"
        or argv[7] != "--amendment-sha256"
        or argv[9] != "--reviewed-implementation-commit"
        or argv[11] != "--gpu-count"
        or argv[13] != "--allocation-sha256"
        or argv[15] != "--"
    ):
        raise G0Error("scientific child invocation differs from the fixed CLI ABI")
    _configure_bytecode_isolation()
    code_commit = argv[2]
    request_commit = argv[4]
    preflight_commit = argv[6]
    amendment_sha256 = argv[8]
    reviewed_implementation_commit = argv[10]
    gpu_count_raw = argv[12]
    allocation_sha256 = argv[14]
    if (
        GIT_COMMIT.fullmatch(code_commit) is None
        or GIT_COMMIT.fullmatch(request_commit) is None
        or GIT_COMMIT.fullmatch(preflight_commit) is None
        or SHA256.fullmatch(amendment_sha256) is None
        or GIT_COMMIT.fullmatch(reviewed_implementation_commit) is None
        or POSITIVE_INTEGER.fullmatch(gpu_count_raw) is None
        or int(gpu_count_raw) not in ALLOWED_GPU_COUNTS
        or SHA256.fullmatch(allocation_sha256) is None
        or _require_clean_git() != code_commit
    ):
        raise G0Error("scientific child lineage arguments are invalid")
    amendment = (SOURCE_ROOT / AMENDMENT_RELATIVE_PATH).read_bytes()
    lineage = _bootstrap_accepted_lineage(
        code_commit=code_commit,
        resolved={
            "scheduler_g0": {
                "request_git_commit": request_commit,
                "gpu_preflight_matrix": {
                    str(world_size): {"git_commit": preflight_commit}
                    for world_size in ALLOWED_GPU_COUNTS
                },
            }
        },
        amendment=amendment,
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
    child_argv = list(argv[16:])
    if argv[0] == "finalize_g0":
        from posttrain_circuits.cli.finalize_g0 import ScientificInvocationContext

        main_function(
            child_argv,
            scientific_context=ScientificInvocationContext(
                allocation_sha256=allocation_sha256,
                code_commit=code_commit,
                gpu_preflight_git_commit=preflight_commit,
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
    _validate_environment(invocation.gpu_count)
    code_commit = _require_clean_git()
    payloads, prereg, amendment, raw_inputs = _read_inputs(invocation)
    bootstrap_lineage = _bootstrap_accepted_lineage(
        code_commit=code_commit,
        resolved=payloads["resolved_config_sha256"],
        amendment=amendment,
    )
    _install_source_path()
    resolved, preflights, amendment_binding = _validate_config_and_preflight(
        invocation,
        payloads,
        prereg,
        amendment,
        raw_inputs,
        code_commit=code_commit,
        bootstrap_lineage=bootstrap_lineage,
    )
    preflight = preflights[invocation.gpu_count]
    from posttrain_circuits.scheduler_adapter.qwen3_v2_g0 import (
        batch_token_contract,
        execution_context,
    )
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
        selected_report_name = _preflight_report_name(invocation.gpu_count)
        (workspace / "gpu_preflight.json").write_bytes(raw_inputs[selected_report_name])
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
            _run_stage(
                stage,
                script_path=_script_parent_path(),
                job_id=invocation.job_id,
                bootstrap_lineage=bootstrap_lineage,
                code_commit=code_commit,
                gpu_count=invocation.gpu_count,
                allocation_sha256=invocation.allocation_sha256,
            )
        initial_hash = _sha256_file(workspace / "initial_checkpoint.pt")
        stages = _stage_plan(
            workspace,
            initial_checkpoint_sha256=initial_hash,
            gpu_count=invocation.gpu_count,
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
                code_commit=code_commit,
                gpu_count=invocation.gpu_count,
                allocation_sha256=invocation.allocation_sha256,
            )
        inner = _strict_json((workspace / "g0.json").read_bytes(), context="inner G0 report")
        inner_digest = inner.get("sha256")
        if (
            inner.get("passed") is not True
            or not isinstance(inner.get("checks"), dict)
            or tuple(sorted(inner["checks"])) != G0_CHECK_NAMES
            or any(value is not True for value in inner["checks"].values())
            or inner_digest != _sha256_value({key: value for key, value in inner.items() if key != "sha256"})
            or inner.get("protocol_amendment_id") != amendment_binding.amendment_id
            or inner.get("protocol_amendment_sha256") != amendment_binding.sha256
            or inner.get("reviewed_implementation_commit")
            != amendment_binding.reviewed_implementation_commit
            or inner.get("request_git_commit")
            != resolved["scheduler_g0"]["request_git_commit"]
            or inner.get("allocation_sha256") != invocation.allocation_sha256
            or inner.get("world_size") != invocation.gpu_count
            or inner.get("batch_token_contract")
            != batch_token_contract(invocation.gpu_count)
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
            "batch_token_contract": batch_token_contract(invocation.gpu_count),
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "code_commit": code_commit,
            "completed_at": completed_at,
            "cgroup_memory": _cgroup_memory(),
            "enable_thinking": False,
            "execution": invocation.execution,
            "execution_context": execution_context(invocation.gpu_count),
            "git_commit": code_commit,
            "gpu_preflight_completion_sha256": invocation.input_hashes[
                _preflight_completion_name(invocation.gpu_count)
            ],
            "gpu_preflight_git_commit": preflight["git_commit"],
            "gpu_preflight_matrix": resolved["scheduler_g0"]["gpu_preflight_matrix"],
            "gpu_preflight_report_sha256": invocation.input_hashes[
                selected_report_name
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
