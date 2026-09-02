#!/usr/bin/python3
"""Standalone, CPU-only repository preflight for an approved ServerScheduler attempt.

This file deliberately imports only the Python standard library.  The adapter
passes already-held content and output descriptors; this handler never resolves
caller-provided filesystem paths and never manages scheduler state or devices.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence


TASK = "repository_preflight"
PROJECT = "OPD"
REPORT_NAME = "preflight_report.json"
COMPLETION_NAME = ".opd-scientific-completion.json"
REPORT_KIND = "repository_preflight"
COMPLETION_KIND = "scientific_attempt"
SCHEMA_VERSION = 1
CONFIG_BINDING_SCHEMA_VERSION = 3
MAX_SMALL_INPUT_BYTES = 4 * 1024 * 1024
MAX_CONFIG_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PROC_FD = re.compile(r"/proc/self/fd/(0|[1-9][0-9]*)\Z")
POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")
THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
CONFIG_INPUT_NAMES = (
    "config_binding_sha256",
    "execution_config_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
)
EXPECTED_INPUT_NAMES = tuple(sorted(CONFIG_INPUT_NAMES))
GATE_NAMES = (
    "config_binding",
    "no_gpu_required",
    "runtime_isolation",
)
STORAGE_LOCATOR_PATHS = (
    ("output_root",),
    ("prereg_path",),
    ("state_source", "store_path"),
    ("task", "dataset_family_path"),
    ("anti_shortcut", "report_path"),
    ("production_safety", "readiness_report"),
    ("production_safety", "probe_cohort_manifest"),
    ("production_safety", "initial_checkpoint_path"),
    ("experiment", "random_reward_calibration_path"),
)
_MISSING = object()


class PreflightError(ValueError):
    """Fail-closed input, environment, or publication error."""


@dataclass(frozen=True)
class ContentHandle:
    name: str
    kind: str
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
        return {handle.name: handle.sha256 for handle in self.content_handles}

    @property
    def execution(self) -> dict[str, Any]:
        return {
            "allocation_sha256": self.allocation_sha256,
            "attempt": self.attempt,
            "execution_profile": self.execution_profile,
            "job_id": self.job_id,
            "manifest_sha256": self.manifest_sha256,
        }


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone OPD repository preflight",
        allow_abbrev=False,
    )
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
        default=None,
    )
    return parser


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise PreflightError(f"{name} must be a canonical identifier")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PreflightError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _descriptor(proc_path: object, *, name: str) -> int:
    if not isinstance(proc_path, str):
        raise PreflightError(f"{name} must be a held descriptor path")
    match = PROC_FD.fullmatch(proc_path)
    if match is None:
        raise PreflightError(f"{name} must use canonical /proc/self/fd/N form")
    descriptor = int(match.group(1))
    if descriptor < 3:
        raise PreflightError(f"{name} must not alias a standard stream")
    try:
        os.fstat(descriptor)
    except OSError as error:
        raise PreflightError(f"{name} descriptor is not open") from error
    return descriptor


def _parse_invocation(argv: Sequence[str] | None) -> Invocation:
    args = _parser().parse_args(argv)
    if args.attempt_completion_name != COMPLETION_NAME:
        raise PreflightError("attempt completion name differs from the fixed ABI")
    if not isinstance(args.attempt, str) or not POSITIVE_INTEGER.fullmatch(args.attempt):
        raise PreflightError("attempt must be a canonical positive integer")
    raw_handles = args.content_handle
    if not isinstance(raw_handles, list) or not raw_handles:
        raise PreflightError("at least one content handle is required")
    handles: list[ContentHandle] = []
    for raw in raw_handles:
        if not isinstance(raw, list) or len(raw) != 4:
            raise PreflightError("content handle differs from the fixed ABI")
        raw_name, kind, digest, proc_path = raw
        name = _identifier(raw_name, name="content handle name")
        if kind != "file":
            raise PreflightError(f"repository preflight input {name!r} must be a file")
        handles.append(
            ContentHandle(
                name=name,
                kind=kind,
                sha256=_sha256(digest, name=f"content handle {name!r}"),
                descriptor=_descriptor(proc_path, name=f"content handle {name!r}"),
            )
        )
    names = tuple(handle.name for handle in handles)
    if names != EXPECTED_INPUT_NAMES:
        raise PreflightError(
            "content handles must be the exact canonical repository-preflight inputs"
        )
    descriptors = tuple(handle.descriptor for handle in handles)
    if len(set(descriptors)) != len(descriptors):
        raise PreflightError("content handles must use distinct descriptors")
    output_descriptor = _descriptor(
        args.output_attempt_handle,
        name="output attempt handle",
    )
    if output_descriptor in descriptors:
        raise PreflightError("output attempt must not alias a content handle")
    return Invocation(
        workflow_id=_identifier(args.workflow_id, name="workflow_id"),
        plan_sha256=_sha256(args.plan_sha256, name="plan_sha256"),
        unit_id=_identifier(args.unit_id, name="unit_id"),
        run_id=_sha256(args.run_id, name="run_id"),
        job_id=_identifier(args.job_id, name="job_id"),
        attempt=int(args.attempt),
        execution_profile=_identifier(
            args.execution_profile,
            name="execution_profile",
        ),
        manifest_sha256=_sha256(args.manifest_sha256, name="manifest_sha256"),
        allocation_sha256=_sha256(args.allocation_sha256, name="allocation_sha256"),
        content_handles=tuple(handles),
        output_descriptor=output_descriptor,
    )


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
        raise PreflightError(f"value cannot be represented as canonical JSON: {error}") from error


def _sha256_value(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _published_json_bytes(payload: object) -> bytes:
    try:
        return (
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PreflightError(f"output cannot be represented as JSON: {error}") from error


def _reject_constant(value: str) -> None:
    raise PreflightError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PreflightError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except PreflightError:
        raise
    except json.JSONDecodeError as error:
        raise PreflightError(f"{context} is not strict JSON") from error


def _read_descriptor(
    descriptor: int,
    *,
    context: str,
    max_bytes: int,
) -> bytes:
    try:
        before = os.fstat(descriptor)
        flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    except OSError as error:
        raise PreflightError(f"cannot inspect {context} descriptor") from error
    if not stat.S_ISREG(before.st_mode):
        raise PreflightError(f"{context} must be a regular file")
    if flags & os.O_ACCMODE != os.O_RDONLY:
        raise PreflightError(f"{context} descriptor must be read-only")
    if before.st_size < 0 or before.st_size > max_bytes:
        raise PreflightError(f"{context} exceeds its byte limit")
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        try:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
        except OSError as error:
            raise PreflightError(f"cannot read {context} descriptor") from error
        if not chunk:
            raise PreflightError(f"{context} changed during its held-fd read")
        chunks.append(chunk)
        offset += len(chunk)
    try:
        after = os.fstat(descriptor)
    except OSError as error:
        raise PreflightError(f"cannot re-inspect {context} descriptor") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_after != identity_before:
        raise PreflightError(f"{context} changed during its held-fd read")
    return b"".join(chunks)


def _read_inputs(invocation: Invocation) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    inode_identities: set[tuple[int, int]] = set()
    for handle in invocation.content_handles:
        metadata = os.fstat(handle.descriptor)
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in inode_identities:
            raise PreflightError("content handles must not alias the same inode")
        inode_identities.add(inode)
        max_bytes = (
            MAX_SMALL_INPUT_BYTES
            if handle.name == "config_binding_sha256"
            else MAX_CONFIG_INPUT_BYTES
        )
        raw = _read_descriptor(
            handle.descriptor,
            context=f"content {handle.name!r}",
            max_bytes=max_bytes,
        )
        if hashlib.sha256(raw).hexdigest() != handle.sha256:
            raise PreflightError(f"content {handle.name!r} differs from its declared SHA-256")
        payload = _strict_json(raw, context=f"content {handle.name!r}")
        if not isinstance(payload, dict):
            raise PreflightError(f"content {handle.name!r} must be a JSON object")
        if raw != _canonical_json(payload).encode("utf-8"):
            raise PreflightError(f"content {handle.name!r} is not canonical JSON")
        payloads[handle.name] = payload
    return payloads


def _required_sha_mapping(value: object, *, context: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise PreflightError(f"{context} must be an object")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name.strip():
            raise PreflightError(f"{context} contains an invalid name")
        result[name] = _sha256(digest, name=f"{context}[{name!r}]")
    return result


def _required_string_mapping(value: object, *, context: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise PreflightError(f"{context} must be an object")
    result: dict[str, str] = {}
    for name, item in value.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(item, str)
            or not item.strip()
        ):
            raise PreflightError(f"{context} contains an invalid locator")
        result[name] = item
    return result


def _nested_get(payload: dict[str, Any], parts: tuple[str, ...]) -> object:
    value: object = payload
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _nested_set(payload: dict[str, Any], parts: tuple[str, ...], value: object) -> None:
    parent: object = payload
    for part in parts[:-1]:
        if not isinstance(parent, dict):
            raise PreflightError("resolved configuration has an invalid locator parent")
        parent = parent[part]
    if not isinstance(parent, dict):
        raise PreflightError("resolved configuration has an invalid locator parent")
    parent[parts[-1]] = value


def _nested_delete(payload: dict[str, Any], parts: tuple[str, ...]) -> None:
    parent: object = payload
    for part in parts[:-1]:
        if not isinstance(parent, dict):
            raise PreflightError("resolved configuration has an invalid locator parent")
        parent = parent[part]
    if not isinstance(parent, dict):
        raise PreflightError("resolved configuration has an invalid locator parent")
    del parent[parts[-1]]


def _validate_config_binding(
    payloads: dict[str, dict[str, Any]],
    *,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    binding = payloads["config_binding_sha256"]
    expected_binding_fields = {
        "execution_config_sha256",
        "execution_context",
        "input_artifact_hashes",
        "resolved_config_sha256",
        "schema_version",
        "scientific_config_sha256",
        "storage_locators",
    }
    if set(binding) != expected_binding_fields:
        raise PreflightError("ConfigBinding fields differ from schema v3")
    if type(binding.get("schema_version")) is not int or binding["schema_version"] != 3:
        raise PreflightError("ConfigBinding schema_version must be 3")
    execution_context = binding.get("execution_context")
    if not isinstance(execution_context, dict):
        raise PreflightError("ConfigBinding execution_context must be an object")
    artifact_hashes = _required_sha_mapping(
        binding.get("input_artifact_hashes"),
        context="ConfigBinding input_artifact_hashes",
    )
    storage_locators = _required_string_mapping(
        binding.get("storage_locators"),
        context="ConfigBinding storage_locators",
    )
    for field in (
        "execution_config_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
    ):
        _sha256(binding.get(field), name=f"ConfigBinding {field}")
        if binding[field] != input_hashes[field]:
            raise PreflightError(f"ConfigBinding {field} differs from its content handle")
    if len(
        {
            binding["execution_config_sha256"],
            binding["resolved_config_sha256"],
            binding["scientific_config_sha256"],
        }
    ) != 3:
        raise PreflightError("ConfigBinding projections must have distinct identities")
    resolved = payloads["resolved_config_sha256"]
    scientific = payloads["scientific_config_sha256"]
    execution = payloads["execution_config_sha256"]
    scientific_config = copy.deepcopy(resolved)
    observed_locators: dict[str, str] = {}
    for parts in STORAGE_LOCATOR_PATHS:
        observed = _nested_get(resolved, parts)
        if observed is _MISSING or observed is None or str(observed) == "":
            continue
        name = ".".join(parts)
        observed_locators[name] = str(observed)
        if name in artifact_hashes:
            _nested_set(
                scientific_config,
                parts,
                {"content_sha256": artifact_hashes[name]},
            )
        else:
            _nested_delete(scientific_config, parts)
    expected_scientific = {
        "config": scientific_config,
        "input_artifact_hashes": artifact_hashes,
        "schema_version": CONFIG_BINDING_SCHEMA_VERSION,
    }
    expected_execution = {
        "execution_context": execution_context,
        "schema_version": CONFIG_BINDING_SCHEMA_VERSION,
        "storage_locators": observed_locators,
    }
    if storage_locators != observed_locators:
        raise PreflightError("ConfigBinding storage locators differ from resolved config")
    if scientific != expected_scientific:
        raise PreflightError("scientific config projection differs from ConfigBinding")
    if execution != expected_execution:
        raise PreflightError("execution config projection differs from ConfigBinding")
    expected_hashes = {
        "resolved_config_sha256": _sha256_value(resolved),
        "scientific_config_sha256": _sha256_value(expected_scientific),
        "execution_config_sha256": _sha256_value(expected_execution),
    }
    for field, expected in expected_hashes.items():
        if binding[field] != expected:
            raise PreflightError(f"ConfigBinding {field} cannot be reproduced")
    return binding


def _validate_runtime_isolation() -> None:
    leaked = sorted(
        key
        for key in os.environ
        if key.startswith("SERVER_SCHEDULER_")
        or key
        in {
            "LD_AUDIT",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PATH",
            "PYTHONHOME",
            "PYTHONPATH",
        }
    )
    if leaked:
        raise PreflightError(f"ambient runtime injection variables reached the handler: {leaked}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PreflightError("PYTHONNOUSERSITE=1 is required")
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.ignore_environment):
        raise PreflightError("handler must run under isolated Python with site disabled")
    threads = tuple(os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS)
    if (
        any(value is None or not POSITIVE_INTEGER.fullmatch(value) for value in threads)
        or len(set(threads)) != 1
    ):
        raise PreflightError("validated CPU thread environment is incomplete or inconsistent")


def _validate_no_gpu() -> None:
    gpu_values = {
        key: os.environ.get(key, "")
        for key in (
            "CUDA_DEVICE_ORDER",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        )
    }
    exposed = sorted(key for key, value in gpu_values.items() if value)
    if exposed:
        raise PreflightError(f"CPU-only preflight received GPU visibility: {exposed}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_timestamp(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PreflightError(f"{name} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise PreflightError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PreflightError(f"{name} must use UTC")
    return value


def _report_payload(
    invocation: Invocation,
    *,
    binding: dict[str, Any],
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    gates = {name: True for name in GATE_NAMES}
    return {
        "allocation_sha256": invocation.allocation_sha256,
        "completed_at": completed_at,
        "execution": invocation.execution,
        "execution_config_sha256": binding["execution_config_sha256"],
        "input_hashes": invocation.input_hashes,
        "manifest_sha256": invocation.manifest_sha256,
        "plan_sha256": invocation.plan_sha256,
        "project": PROJECT,
        "report_kind": REPORT_KIND,
        "resolved_config_sha256": binding["resolved_config_sha256"],
        "run_id": invocation.run_id,
        "schema_version": SCHEMA_VERSION,
        "scientific_config_sha256": binding["scientific_config_sha256"],
        "scientific_validation": gates,
        "started_at": started_at,
        "task": TASK,
        "unit_id": invocation.unit_id,
        "workflow_id": invocation.workflow_id,
    }


def _completion_payload(
    invocation: Invocation,
    *,
    binding: dict[str, Any],
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    content: dict[str, Any] = {
        "completed_at": completed_at,
        "completion_kind": COMPLETION_KIND,
        "execution": invocation.execution,
        "execution_config_sha256": binding["execution_config_sha256"],
        "input_hashes": invocation.input_hashes,
        "plan_sha256": invocation.plan_sha256,
        "project": PROJECT,
        "resolved_config_sha256": binding["resolved_config_sha256"],
        "run_id": invocation.run_id,
        "schema_version": SCHEMA_VERSION,
        "scientific_config_sha256": binding["scientific_config_sha256"],
        "scientific_validation": {name: True for name in GATE_NAMES},
        "started_at": started_at,
        "task": TASK,
        "unit_id": invocation.unit_id,
        "workflow_id": invocation.workflow_id,
    }
    return {**content, "sha256": _sha256_value(content)}


def _output_entries(descriptor: int) -> set[str]:
    try:
        metadata = os.fstat(descriptor)
        entries = os.listdir(descriptor)
    except OSError as error:
        raise PreflightError("cannot inspect output attempt descriptor") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise PreflightError("output attempt handle must refer to a directory")
    if len(entries) != len(set(entries)):
        raise PreflightError("output attempt contains ambiguous entries")
    return set(entries)


def _read_output(descriptor: int, name: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        file_descriptor = os.open(name, flags, dir_fd=descriptor)
    except OSError as error:
        raise PreflightError(f"cannot open existing output {name!r}") from error
    try:
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PreflightError(f"existing output {name!r} must be one regular inode")
        return _read_descriptor(
            file_descriptor,
            context=f"existing output {name!r}",
            max_bytes=MAX_OUTPUT_BYTES,
        )
    finally:
        os.close(file_descriptor)


def _publish_once(descriptor: int, name: str, raw: bytes) -> None:
    if name not in {REPORT_NAME, COMPLETION_NAME}:
        raise PreflightError("output name is not part of the fixed handler contract")
    temporary_name = f".{name}.{secrets.token_hex(16)}.tmp"
    temporary_descriptor = -1
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        temporary_descriptor = os.open(
            temporary_name,
            flags,
            0o640,
            dir_fd=descriptor,
        )
        view = memoryview(raw)
        while view:
            written = os.write(temporary_descriptor, view)
            if written < 1:
                raise PreflightError(f"short write while publishing {name!r}")
            view = view[written:]
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_output(descriptor, name) != raw:
                raise PreflightError(f"conflicting immutable output already exists: {name}")
        os.unlink(temporary_name, dir_fd=descriptor)
        temporary_name = ""
        os.fsync(descriptor)
        if _read_output(descriptor, name) != raw:
            raise PreflightError(f"published output changed: {name}")
    except OSError as error:
        raise PreflightError(f"cannot publish immutable output {name!r}") from error
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=descriptor)
            except FileNotFoundError:
                pass


def _validate_existing_report(
    raw: bytes,
    *,
    invocation: Invocation,
    binding: dict[str, Any],
) -> dict[str, Any]:
    payload = _strict_json(raw, context="existing preflight report")
    if not isinstance(payload, dict) or raw != _published_json_bytes(payload):
        raise PreflightError("existing preflight report bytes are not canonical")
    expected_fields = set(
        _report_payload(
            invocation,
            binding=binding,
            started_at="placeholder",
            completed_at="placeholder",
        )
    )
    if set(payload) != expected_fields:
        raise PreflightError("existing preflight report fields differ from schema")
    started_at = _validate_timestamp(payload.get("started_at"), name="report started_at")
    completed_at = _validate_timestamp(payload.get("completed_at"), name="report completed_at")
    expected = _report_payload(
        invocation,
        binding=binding,
        started_at=started_at,
        completed_at=completed_at,
    )
    if payload != expected:
        raise PreflightError("existing preflight report differs from this invocation")
    if datetime.fromisoformat(completed_at[:-1] + "+00:00") < datetime.fromisoformat(
        started_at[:-1] + "+00:00"
    ):
        raise PreflightError("existing preflight report timestamps are inverted")
    return payload


def _validate_existing_completion(
    raw: bytes,
    *,
    invocation: Invocation,
    binding: dict[str, Any],
    report: dict[str, Any],
) -> None:
    payload = _strict_json(raw, context="existing completion draft")
    if not isinstance(payload, dict) or raw != _published_json_bytes(payload):
        raise PreflightError("existing completion draft bytes are not canonical")
    expected = _completion_payload(
        invocation,
        binding=binding,
        started_at=report["started_at"],
        completed_at=report["completed_at"],
    )
    if payload != expected:
        raise PreflightError("existing completion draft differs from this invocation")


def _publish_outputs(
    invocation: Invocation,
    *,
    binding: dict[str, Any],
    started_at: str,
) -> None:
    entries = _output_entries(invocation.output_descriptor)
    allowed = {REPORT_NAME, COMPLETION_NAME}
    if entries - allowed:
        raise PreflightError("output attempt contains entries outside the fixed contract")
    if COMPLETION_NAME in entries and REPORT_NAME not in entries:
        raise PreflightError("completion draft exists without its scientific report")
    if REPORT_NAME in entries:
        report = _validate_existing_report(
            _read_output(invocation.output_descriptor, REPORT_NAME),
            invocation=invocation,
            binding=binding,
        )
        if COMPLETION_NAME in entries:
            _validate_existing_completion(
                _read_output(invocation.output_descriptor, COMPLETION_NAME),
                invocation=invocation,
                binding=binding,
                report=report,
            )
            return
        completion = _completion_payload(
            invocation,
            binding=binding,
            started_at=report["started_at"],
            completed_at=report["completed_at"],
        )
        _publish_once(
            invocation.output_descriptor,
            COMPLETION_NAME,
            _published_json_bytes(completion),
        )
        return

    completed_at = _utc_now()
    report = _report_payload(
        invocation,
        binding=binding,
        started_at=started_at,
        completed_at=completed_at,
    )
    completion = _completion_payload(
        invocation,
        binding=binding,
        started_at=started_at,
        completed_at=completed_at,
    )
    _publish_once(
        invocation.output_descriptor,
        REPORT_NAME,
        _published_json_bytes(report),
    )
    _publish_once(
        invocation.output_descriptor,
        COMPLETION_NAME,
        _published_json_bytes(completion),
    )


def run(argv: Sequence[str] | None = None) -> int:
    started_at = _utc_now()
    invocation = _parse_invocation(argv)
    _validate_runtime_isolation()
    _validate_no_gpu()
    payloads = _read_inputs(invocation)
    binding = _validate_config_binding(payloads, input_hashes=invocation.input_hashes)
    _publish_outputs(
        invocation,
        binding=binding,
        started_at=started_at,
    )
    return 0


def main() -> int:
    try:
        return run()
    except PreflightError as error:
        print(f"repository preflight rejected: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"repository preflight failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
