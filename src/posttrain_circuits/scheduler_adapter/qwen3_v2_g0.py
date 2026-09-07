"""Scientific completion contract for the scheduler-managed Qwen3-v2 G0 gate."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.learning.training.fsdp_contract import (
    REQUESTED_FSDP_SHARDING_STRATEGY,
    effective_fsdp_sharding_strategy,
)
from posttrain_circuits.learning.training.execution_safety_kernel import (
    BATCH_PARTITION_PROTOCOL,
    GLOBAL_BATCH_SIZE,
    batch_token_contract as _kernel_batch_token_contract,
    build_runtime_execution_safety_attestation,
    validate_runtime_execution_safety_attestation,
)
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.artifacts.execution_safety_certification import (
    CERTIFICATION_CONTENT_NAME,
    DESCRIPTOR_CONTENT_NAME,
    EXECUTION_CLASS_ID,
    SUCCESSOR_AMENDMENT_RELATIVE_PATH,
)
from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json


TASK_NAME = "qwen3_v2_g0"
PROFILE_NAME = "qwen3-v2-g0-elastic"
WORKFLOW_ID_PREFIX = "qwen3-v2-g0-elastic-"
UNIT_ID = "g0"
REPORT_NAME = "g0.json"
BUNDLE_NAME = "g0_artifacts.tar"
OUTPUT_NAMES = tuple(sorted((REPORT_NAME, BUNDLE_NAME)))

ALLOWED_GPU_COUNTS = (1, 2, 3, 4)
CPU_CORE_COUNT = 24
MEMORY_MIB = 196608
GPU_MEMORY_MIB = 81920
GPU_UTILIZATION_PCT = 95
GPU_MODEL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
ESTIMATED_RUNTIME_SECONDS = 86400.0
NODE_MEMORY_BYTES = 192 * 1024**3
MINIMUM_HEADROOM_BYTES = max(32 * 1024**3, int(NODE_MEMORY_BYTES * 0.20))

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

PREREG_CONTENT_NAME = "preregistration_sha256"
PROTOCOL_AMENDMENT_CONTENT_NAME = "protocol_amendment_sha256"
EXECUTION_SCIENCE_PROTOCOL_CONTENT_NAME = "execution_science_protocol_sha256"
EXPECTED_INPUT_NAMES = (
    "config_binding_sha256",
    "execution_config_sha256",
    CERTIFICATION_CONTENT_NAME,
    DESCRIPTOR_CONTENT_NAME,
    EXECUTION_SCIENCE_PROTOCOL_CONTENT_NAME,
    PREREG_CONTENT_NAME,
    PROTOCOL_AMENDMENT_CONTENT_NAME,
    "resolved_config_sha256",
    "scientific_config_sha256",
)

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

PROVENANCE_VALIDATION = {
    "certification_artifacts_unchanged": True,
    "current_execution_safety_kernel_matches_descriptor": True,
    "execution_safety_fingerprint_exact": True,
}

REQUIRED_BUNDLE_PATHS = frozenset(
    {
        "anti_shortcut.json",
        "calibration/checkpoints/step-00000020.pt",
        "calibration/config_binding.json",
        "calibration/factorial_update_evidence.json",
        "calibration/manifest.json",
        "calibration/metrics.jsonl",
        "calibration/resolved_config.yaml",
        "circuits/final_answer/circuit.json",
        "circuits/final_answer/exact_patching.json",
        "circuits/final_answer/mib_raw/compatibility.json",
        "circuits/first_rule_selection/circuit.json",
        "circuits/first_rule_selection/exact_patching.json",
        "circuits/first_rule_selection/mib_raw/compatibility.json",
        "dataset/manifest.json",
        "distributed_resume.json",
        "execution_safety_certification.yaml",
        "execution_safety_descriptor.json",
        "execution_science_protocol.yaml",
        "g0.json",
        "initial_checkpoint.pt",
        "label_leakage.json",
        "probe_scores.json",
        "probes/manifest.json",
        "readiness/readiness.json",
        "resume-a/manifest.json",
        "resume-a/config_binding.json",
        "resume-a/factorial_update_evidence.json",
        "resume-a/metrics.jsonl",
        "resume-a/resolved_config.yaml",
        "resume-b/manifest.json",
        "resume-b/config_binding.json",
        "resume-b/factorial_update_evidence.json",
        "resume-b/metrics.jsonl",
        "resume-b/resolved_config.yaml",
        "rollout_bank/manifest.json",
        "teacher_demos/manifest.json",
        "teacher_readiness.json",
        "teacher_scores/manifest.json",
    }
)

REPORT_FIELDS = frozenset(
    {
        "artifact_bundle_sha256",
        "artifact_inventory",
        "artifact_namespace",
        "batch_token_contract",
        "chat_template_sha256",
        "code_commit",
        "completed_at",
        "cgroup_memory",
        "enable_thinking",
        "execution",
        "execution_context",
        "execution_safety_certification_sha256",
        "execution_safety_class_id",
        "execution_safety_descriptor_sha256",
        "execution_safety_fingerprint_sha256",
        "execution_safety_attestation",
        "execution_science_protocol_git_commit",
        "execution_science_protocol_id",
        "execution_science_protocol_reviewed_implementation_commit",
        "execution_science_protocol_sha256",
        "git_commit",
        "inner_g0_report_sha256",
        "model_revision",
        "passed",
        "phase",
        "prereg_commit",
        "prereg_path",
        "prereg_sha256",
        "prereg_version",
        "protocol_amendment_git_commit",
        "protocol_amendment_id",
        "protocol_amendment_sha256",
        "provenance_validation",
        "prompt_protocol",
        "protocol_track",
        "resolved_config_sha256",
        "runtime_resolved_config_sha256",
        "request_git_commit",
        "schema_version",
        "sha256",
        "started_at",
        "teacher_revision",
        "tokenizer_fingerprint",
        "tokenizer_revision",
        "world_size",
        "reviewed_implementation_commit",
    }
)
INVENTORY_FIELDS = frozenset({"path", "sha256", "size"})
CGROUP_MEMORY_FIELDS = frozenset(
    {
        "current_bytes",
        "headroom_bytes",
        "limit_bytes",
        "minimum_required_headroom_bytes",
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
PROTOCOL_AMENDMENT_ID = "qwen3_v2_g0_execution_class_v2"
PROTOCOL_AMENDMENT_PATH = str(SUCCESSOR_AMENDMENT_RELATIVE_PATH)
REPLAY_SCRATCH_ROOT = Path("/scr/del6500/OPD/tmp")
_REPLAY_PARENT_NAME = re.compile(r"\.qwen3-v2-g0-replay-[A-Za-z0-9_-]+\Z")


def batch_token_contract(
    world_size: int,
    prompt_population_size: int,
) -> dict[str, Any]:
    """Return the shared execution-kernel contract with adapter errors."""

    try:
        return _kernel_batch_token_contract(world_size, prompt_population_size)
    except ValueError as error:
        raise AdapterValidationError(str(error)) from error


def execution_context(world_size: int) -> dict[str, Any]:
    """Return report-only evidence for the already validated allocation."""

    if world_size not in ALLOWED_GPU_COUNTS:
        _fail("G0 world_size is outside the reviewed 1--4 GPU capability")
    return {
        "allocation_contract": "manifest_driven_scheduler_gpu_v1",
        "allocation_visibility": "preserved",
        "cpu_core_count": CPU_CORE_COUNT,
        "distributed_launcher": "accelerate_fsdp_foreground",
        "logical_device_indices": list(range(world_size)),
        "mode": "server_scheduler_foreground",
        "nccl_p2p_policy": "disabled",
        "threads_per_rank": CPU_CORE_COUNT // world_size,
        "visible_device_count": world_size,
    }


def _fail(message: str) -> None:
    raise AdapterValidationError(message)


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
        _fail(f"{name} must be an integer at least {minimum}")
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


def _safe_bundle_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail("G0 artifact inventory path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail("G0 artifact inventory path is not confined")
    normalized = str(path)
    if normalized != value:
        _fail("G0 artifact inventory path is not canonical")
    return normalized


def _validate_inventory(value: object) -> dict[str, tuple[str, int]]:
    if not isinstance(value, list) or not value:
        _fail("G0 artifact inventory must be a non-empty array")
    inventory: dict[str, tuple[str, int]] = {}
    ordered: list[str] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != INVENTORY_FIELDS:
            _fail("G0 artifact inventory fields differ from the reviewed contract")
        path = _safe_bundle_path(row["path"])
        digest = _sha256(row["sha256"], name=f"G0 artifact {path!r} sha256")
        size = _integer(row["size"], name=f"G0 artifact {path!r} size")
        if path in inventory:
            _fail("G0 artifact inventory contains a duplicate path")
        inventory[path] = (digest, size)
        ordered.append(path)
    if ordered != sorted(ordered):
        _fail("G0 artifact inventory is not canonically ordered")
    missing = REQUIRED_BUNDLE_PATHS - set(inventory)
    if missing:
        _fail(f"G0 artifact bundle lacks required evidence: {sorted(missing)}")
    return inventory


def _validate_cgroup_memory(value: object) -> None:
    if not isinstance(value, dict) or set(value) != CGROUP_MEMORY_FIELDS:
        _fail("G0 cgroup-memory fields differ from the reviewed contract")
    limit = _integer(value["limit_bytes"], name="G0 memory limit", minimum=1)
    current = _integer(value["current_bytes"], name="G0 current memory")
    peak = _integer(value["peak_bytes"], name="G0 peak memory", minimum=1)
    requested = _integer(value["requested_bytes"], name="G0 requested memory", minimum=1)
    headroom = _integer(value["headroom_bytes"], name="G0 memory headroom")
    minimum = _integer(
        value["minimum_required_headroom_bytes"],
        name="G0 minimum memory headroom",
        minimum=1,
    )
    if (
        value["passed"] is not True
        or limit != NODE_MEMORY_BYTES
        or requested != NODE_MEMORY_BYTES
        or current > limit
        or peak > limit
        or headroom != limit - peak
        or minimum != MINIMUM_HEADROOM_BYTES
        or headroom < minimum
    ):
        _fail("G0 cgroup-memory evidence did not pass the exact 192-GiB contract")


def _read_json_member(stream: BinaryIO, *, path: str, maximum: int) -> dict[str, Any]:
    raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        _fail(f"G0 bundle member {path!r} exceeds its size limit")
    try:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key {key!r}")
                result[key] = value
            return result

        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AdapterValidationError(f"G0 bundle member {path!r} is not strict JSON") from error
    if not isinstance(payload, dict):
        _fail(f"G0 bundle member {path!r} must be a JSON object")

    def require_finite(value: object) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            _fail(f"G0 bundle member {path!r} contains a non-finite number")
        if isinstance(value, dict):
            for item in value.values():
                require_finite(item)
        elif isinstance(value, list):
            for item in value:
                require_finite(item)

    require_finite(payload)
    return payload


def _validate_inner_g0(payload: dict[str, Any], *, report: Mapping[str, Any]) -> None:
    from posttrain_circuits.cli.finalize_g0 import G0_CHECK_NAMES

    digest = _sha256(payload.get("sha256"), name="inner G0 report sha256")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    if digest != sha256_value(unsigned):
        _fail("inner G0 report self-summary is invalid")
    checks = payload.get("checks")
    if (
        payload.get("phase") != "G0"
        or payload.get("passed") is not True
        or not isinstance(checks, dict)
        or tuple(sorted(checks)) != G0_CHECK_NAMES
        or any(value is not True for value in checks.values())
        or checks.get("distributed_checkpoint_resume") is not True
        or checks.get("execution_safety_descriptor") is not True
        or checks.get("execution_safety_certification") is not True
        or checks.get("execution_science_protocol") is not True
    ):
        _fail("inner G0 semantic decision did not pass every registered gate")
    if (
        payload.get("git_commit") != report["git_commit"]
        or payload.get("code_commit") != report["code_commit"]
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("teacher_revision") != TEACHER_REVISION
        or payload.get("tokenizer_revision") != TOKENIZER_REVISION
        or payload.get("prereg_sha256") != report["prereg_sha256"]
        or payload.get("protocol_amendment_id") != report["protocol_amendment_id"]
        or payload.get("protocol_amendment_sha256")
        != report["protocol_amendment_sha256"]
        or payload.get("execution_safety_class_id")
        != report["execution_safety_class_id"]
        or payload.get("execution_safety_fingerprint_sha256")
        != report["execution_safety_fingerprint_sha256"]
        or payload.get("execution_safety_descriptor_sha256")
        != report["execution_safety_descriptor_sha256"]
        or payload.get("execution_safety_certification_sha256")
        != report["execution_safety_certification_sha256"]
        or payload.get("execution_science_protocol_git_commit")
        != report["execution_science_protocol_git_commit"]
        or payload.get("execution_science_protocol_id")
        != report["execution_science_protocol_id"]
        or payload.get("execution_science_protocol_reviewed_implementation_commit")
        != report["execution_science_protocol_reviewed_implementation_commit"]
        or payload.get("execution_science_protocol_sha256")
        != report["execution_science_protocol_sha256"]
        or payload.get("reviewed_implementation_commit")
        != report["reviewed_implementation_commit"]
        or payload.get("request_git_commit") != report["request_git_commit"]
        or payload.get("allocation_sha256")
        != report["execution"]["allocation_sha256"]
        or payload.get("world_size") != report["world_size"]
        or payload.get("batch_token_contract") != report["batch_token_contract"]
        or payload.get("resolved_config_sha256")
        != report["runtime_resolved_config_sha256"]
    ):
        _fail("inner G0 report differs from the outer allocation binding")


def _new_replay_workspace() -> Path:
    """Allocate a fresh private location independent of the producing attempt."""

    if (
        REPLAY_SCRATCH_ROOT.is_symlink()
        or REPLAY_SCRATCH_ROOT.resolve(strict=True) != REPLAY_SCRATCH_ROOT
        or not REPLAY_SCRATCH_ROOT.is_dir()
    ):
        _fail("OPD replay scratch root is absent or unsafe")
    try:
        parent = Path(
            tempfile.mkdtemp(
                prefix=".qwen3-v2-g0-replay-",
                dir=REPLAY_SCRATCH_ROOT,
            )
        )
        metadata = os.lstat(parent)
        if (
            parent.parent != REPLAY_SCRATCH_ROOT
            or _REPLAY_PARENT_NAME.fullmatch(parent.name) is None
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("fresh G0 replay workspace is unsafe")
    except (OSError, ValueError) as error:
        raise AdapterValidationError(
            f"could not create a fresh G0 replay workspace: {error}"
        ) from error
    return parent / "qwen3-v2"


def _remove_replay_workspace(workspace: Path) -> None:
    parent = workspace.parent
    if (
        workspace.name != "qwen3-v2"
        or parent.parent != REPLAY_SCRATCH_ROOT
        or _REPLAY_PARENT_NAME.fullmatch(parent.name) is None
    ):
        _fail("refusing to clean an unrecognized replay workspace")
    try:
        metadata = os.lstat(parent)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        _fail("replay workspace parent changed type before cleanup")
    shutil.rmtree(parent)


def _extract_bundle_to_workspace(
    path: Path,
    inventory: Mapping[str, tuple[str, int]],
    workspace: Path,
) -> None:
    """Extract regular files beneath a fresh private replay directory."""

    if (
        REPLAY_SCRATCH_ROOT.is_symlink()
        or REPLAY_SCRATCH_ROOT.resolve(strict=True) != REPLAY_SCRATCH_ROOT
        or not REPLAY_SCRATCH_ROOT.is_dir()
    ):
        _fail("OPD replay scratch root is absent or unsafe")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW

    base_descriptor = os.open(REPLAY_SCRATCH_ROOT, directory_flags)
    created = False
    try:
        parent_metadata = os.stat(
            workspace.parent.name,
            dir_fd=base_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or stat.S_ISLNK(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.getuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o077
        ):
            _fail("fresh G0 replay workspace changed before extraction")
        parent_descriptor = os.open(
            workspace.parent.name, directory_flags, dir_fd=base_descriptor
        )
        try:
            try:
                os.stat(workspace.name, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                _fail("fresh G0 replay target already exists")
            os.mkdir(workspace.name, mode=0o700, dir_fd=parent_descriptor)
            created = True
            root_descriptor = os.open(
                workspace.name, directory_flags, dir_fd=parent_descriptor
            )
            try:
                extracted: dict[str, tuple[str, int]] = {}
                with tarfile.open(path, mode="r:") as archive:
                    for member in archive:
                        member_path = _safe_bundle_path(member.name)
                        expected = inventory.get(member_path)
                        if (
                            expected is None
                            or not member.isfile()
                            or member.issym()
                            or member.islnk()
                            or member.size != expected[1]
                            or member_path in extracted
                        ):
                            _fail("G0 replay archive differs from its regular-file inventory")
                        stream = archive.extractfile(member)
                        if stream is None:
                            _fail("G0 replay archive member is unreadable")
                        parts = PurePosixPath(member_path).parts
                        directory_descriptor = os.dup(root_descriptor)
                        try:
                            for component in parts[:-1]:
                                try:
                                    os.mkdir(component, mode=0o700, dir_fd=directory_descriptor)
                                except FileExistsError:
                                    pass
                                next_descriptor = os.open(
                                    component,
                                    directory_flags,
                                    dir_fd=directory_descriptor,
                                )
                                os.close(directory_descriptor)
                                directory_descriptor = next_descriptor
                            output = os.open(
                                parts[-1],
                                file_flags,
                                0o600,
                                dir_fd=directory_descriptor,
                            )
                            try:
                                digest = hashlib.sha256()
                                size = 0
                                while True:
                                    block = stream.read(1024 * 1024)
                                    if not block:
                                        break
                                    size += len(block)
                                    digest.update(block)
                                    view = memoryview(block)
                                    while view:
                                        written = os.write(output, view)
                                        if written < 1:
                                            _fail("short write while reconstructing G0 replay")
                                        view = view[written:]
                                metadata = os.fstat(output)
                                if not stat.S_ISREG(metadata.st_mode):
                                    _fail("G0 replay output is not a regular file")
                                os.fsync(output)
                            finally:
                                os.close(output)
                        finally:
                            os.close(directory_descriptor)
                            stream.close()
                        observed = (digest.hexdigest(), size)
                        if observed != expected:
                            _fail("G0 replay member bytes differ from the bound inventory")
                        extracted[member_path] = observed
                if extracted != dict(inventory):
                    _fail("G0 replay extraction omitted or added artifact bytes")
            finally:
                os.close(root_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if created:
            _remove_replay_workspace(workspace)
        raise
    finally:
        os.close(base_descriptor)


def _replay_extracted_workspace(
    workspace: Path,
    inner_payload: dict[str, Any],
    report: Mapping[str, Any],
) -> None:
    from posttrain_circuits.cli.finalize_g0 import (
        ScientificInvocationContext,
        replay_g0_decision,
    )
    from posttrain_circuits.cli.factorial_run_validation import _UniqueKeyLoader

    import yaml

    try:
        config = yaml.load(
            (workspace / "calibration" / "resolved_config.yaml").read_text(
                encoding="utf-8"
            ),
            Loader=_UniqueKeyLoader,
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("bundled G0 resolved config is invalid") from error
    if not isinstance(config, dict):
        raise ValueError("bundled G0 resolved config must be a mapping")
    source_workspace = Path(str(config.get("output_root", "")))
    if (
        not source_workspace.is_absolute()
        or ".." in source_workspace.parts
        or str(source_workspace) != config.get("output_root")
        or source_workspace == workspace
    ):
        raise ValueError("bundled G0 source workspace binding is invalid")
    if sha256_value(config) != report["runtime_resolved_config_sha256"]:
        raise ValueError("bundled G0 resolved config differs from the outer report")
    descriptor = json.loads(
        (workspace / "execution_safety_descriptor.json").read_text(encoding="utf-8")
    )
    resume = json.loads(
        (workspace / "distributed_resume.json").read_text(encoding="utf-8")
    )
    rebuilt_attestation = build_runtime_execution_safety_attestation(
        config=config,
        descriptor=descriptor,
        fingerprint_sha256=report["execution_safety_fingerprint_sha256"],
        resume=resume,
        world_size=report["world_size"],
    )
    if rebuilt_attestation != report["execution_safety_attestation"]:
        raise ValueError("bundled G0 execution-safety attestation differs")
    formal_binding = {
        "protocol_track": report["protocol_track"],
        "artifact_namespace": report["artifact_namespace"],
        "model_revision": report["model_revision"],
        "teacher_revision": report["teacher_revision"],
        "tokenizer_revision": report["tokenizer_revision"],
        "tokenizer_fingerprint": report["tokenizer_fingerprint"],
        "chat_template_sha256": report["chat_template_sha256"],
        "prompt_protocol": report["prompt_protocol"],
        "enable_thinking": report["enable_thinking"],
        "code_commit": report["code_commit"],
        "prereg_path": report["prereg_path"],
        "prereg_version": report["prereg_version"],
        "prereg_commit": report["prereg_commit"],
        "prereg_sha256": report["prereg_sha256"],
        "protocol_amendment_id": report["protocol_amendment_id"],
        "protocol_amendment_path": PROTOCOL_AMENDMENT_PATH,
        "protocol_amendment_git_commit": report["protocol_amendment_git_commit"],
        "protocol_amendment_sha256": report["protocol_amendment_sha256"],
        "reviewed_implementation_commit": report["reviewed_implementation_commit"],
        "execution_safety_class_id": report["execution_safety_class_id"],
        "execution_safety_fingerprint_sha256": report[
            "execution_safety_fingerprint_sha256"
        ],
        "execution_safety_descriptor_sha256": report[
            "execution_safety_descriptor_sha256"
        ],
        "execution_safety_certification_sha256": report[
            "execution_safety_certification_sha256"
        ],
    }
    replay_g0_decision(
        workspace,
        inner_payload,
        config=config,
        scientific_context=ScientificInvocationContext(
            allocation_sha256=report["execution"]["allocation_sha256"],
            code_commit=report["code_commit"],
            execution_safety_class_id=report["execution_safety_class_id"],
            execution_safety_fingerprint_sha256=report[
                "execution_safety_fingerprint_sha256"
            ],
            execution_safety_descriptor_sha256=report[
                "execution_safety_descriptor_sha256"
            ],
            execution_safety_certification_sha256=report[
                "execution_safety_certification_sha256"
            ],
            execution_science_protocol_id=report[
                "execution_science_protocol_id"
            ],
            execution_science_protocol_sha256=report[
                "execution_science_protocol_sha256"
            ],
            execution_science_protocol_git_commit=report[
                "execution_science_protocol_git_commit"
            ],
            execution_science_protocol_reviewed_implementation_commit=report[
                "execution_science_protocol_reviewed_implementation_commit"
            ],
            protocol_amendment_sha256=report["protocol_amendment_sha256"],
            request_git_commit=report["request_git_commit"],
            reviewed_implementation_commit=report["reviewed_implementation_commit"],
            world_size=report["world_size"],
        ),
        formal_binding=formal_binding,
        job_id=report["execution"]["job_id"],
    )


def _validate_bundle(
    path: Path,
    inventory: Mapping[str, tuple[str, int]],
    report: Mapping[str, Any],
) -> None:
    observed: dict[str, tuple[str, int]] = {}
    inner_payload: dict[str, Any] | None = None
    resume_payload: dict[str, Any] | None = None
    try:
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                member_path = _safe_bundle_path(member.name)
                if not member.isfile() or member.issym() or member.islnk():
                    _fail("G0 artifact bundle may contain only regular files")
                if member_path in observed:
                    _fail("G0 artifact bundle contains a duplicate member")
                stream = archive.extractfile(member)
                if stream is None:
                    _fail("G0 artifact bundle member is unreadable")
                digest = hashlib.sha256()
                captured = (
                    io.BytesIO()
                    if member_path in {"g0.json", "distributed_resume.json"}
                    else None
                )
                size = 0
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    digest.update(block)
                    if captured is not None:
                        captured.write(block)
                observed[member_path] = (digest.hexdigest(), size)
                if captured is not None:
                    captured.seek(0)
                    captured_payload = _read_json_member(
                        captured,
                        path=member_path,
                        maximum=32 * 1024 * 1024,
                    )
                    if member_path == "g0.json":
                        inner_payload = captured_payload
                    else:
                        resume_payload = captured_payload
    except (OSError, tarfile.TarError) as error:
        raise AdapterValidationError(f"G0 artifact bundle is invalid: {error}") from error
    if observed != dict(inventory):
        _fail("G0 artifact bundle bytes differ from its complete inventory")
    if inner_payload is None:
        _fail("G0 artifact bundle has no inner G0 report")
    if resume_payload is None:
        _fail("G0 artifact bundle has no distributed resume report")
    if observed["g0.json"][0] != report["inner_g0_report_sha256"]:
        _fail("outer G0 report binds a different inner G0 report")
    _validate_inner_g0(inner_payload, report=report)
    workspace = _new_replay_workspace()
    try:
        _extract_bundle_to_workspace(path, inventory, workspace)
        _replay_extracted_workspace(workspace, inner_payload, report)
    except AdapterValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise AdapterValidationError(f"G0 artifact replay failed: {error}") from error
    finally:
        _remove_replay_workspace(workspace)


def _validate_report(
    report: object,
    *,
    expected_inputs: Mapping[str, str],
    bundle_path: Path,
    expected_execution: object | None = None,
) -> None:
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
        _fail("G0 report fields differ from the reviewed contract")
    digest = _sha256(report["sha256"], name="G0 report sha256")
    unsigned = {key: value for key, value in report.items() if key != "sha256"}
    if digest != sha256_value(unsigned):
        _fail("G0 report self-summary is invalid")
    world_size = _integer(report["world_size"], name="G0 world_size", minimum=1)
    if world_size not in ALLOWED_GPU_COUNTS:
        _fail("G0 report world_size is outside the reviewed 1--4 GPU capability")
    for field in (
        "execution_safety_certification_sha256",
        "execution_safety_descriptor_sha256",
        "execution_safety_fingerprint_sha256",
        "execution_science_protocol_sha256",
        "runtime_resolved_config_sha256",
    ):
        _sha256(report[field], name=f"G0 report {field}")
    batch_contract = report["batch_token_contract"]
    if not isinstance(batch_contract, dict):
        _fail("G0 batch-token contract must be a mapping")
    prompt_population_size = _integer(
        batch_contract.get("prompt_population_size"),
        name="G0 prompt population size",
        minimum=1,
    )
    try:
        validate_runtime_execution_safety_attestation(
            report["execution_safety_attestation"],
            expected_fingerprint_sha256=report[
                "execution_safety_fingerprint_sha256"
            ],
            world_size=world_size,
        )
    except ValueError as error:
        raise AdapterValidationError(str(error)) from error
    science_protocol_id = report["execution_science_protocol_id"]
    if (
        not isinstance(science_protocol_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", science_protocol_id)
        is None
    ):
        _fail("G0 execution science protocol ID is invalid")
    if (
        report["schema_version"] != 2
        or isinstance(report["schema_version"], bool)
        or report["phase"] != TASK_NAME
        or report["passed"] is not True
        or report["protocol_track"] != PROTOCOL_TRACK
        or report["artifact_namespace"] != ARTIFACT_NAMESPACE
        or report["prompt_protocol"] != PROMPT_PROTOCOL
        or report["enable_thinking"] is not False
        or report["model_revision"] != MODEL_REVISION
        or report["teacher_revision"] != TEACHER_REVISION
        or report["tokenizer_revision"] != TOKENIZER_REVISION
        or report["tokenizer_fingerprint"] != TOKENIZER_FINGERPRINT
        or report["chat_template_sha256"] != CHAT_TEMPLATE_SHA256
        or report["prereg_path"] != PREREG_PATH
        or report["prereg_version"] != PREREG_VERSION
        or report["prereg_sha256"] != expected_inputs[PREREG_CONTENT_NAME]
        or report["protocol_amendment_id"] != PROTOCOL_AMENDMENT_ID
        or report["protocol_amendment_sha256"]
        != expected_inputs[PROTOCOL_AMENDMENT_CONTENT_NAME]
        or report["execution_safety_class_id"] != EXECUTION_CLASS_ID
        or report["execution_safety_descriptor_sha256"]
        != expected_inputs[DESCRIPTOR_CONTENT_NAME]
        or report["execution_safety_certification_sha256"]
        != expected_inputs[CERTIFICATION_CONTENT_NAME]
        or report["execution_science_protocol_sha256"]
        != expected_inputs[EXECUTION_SCIENCE_PROTOCOL_CONTENT_NAME]
        or batch_contract
        != batch_token_contract(world_size, prompt_population_size)
        or report["provenance_validation"] != PROVENANCE_VALIDATION
        or report["resolved_config_sha256"] != expected_inputs["resolved_config_sha256"]
        or report["execution_context"] != execution_context(world_size)
    ):
        _fail("G0 report differs from the scheduler-managed Qwen3-v2 protocol")
    git_commit = _git_commit(report["git_commit"], name="G0 git_commit")
    if report["code_commit"] != git_commit:
        _fail("G0 code_commit differs from git_commit")
    request_git_commit = _git_commit(
        report["request_git_commit"], name="G0 request_git_commit"
    )
    _git_commit(report["prereg_commit"], name="G0 prereg_commit")
    _git_commit(
        report["protocol_amendment_git_commit"],
        name="G0 protocol_amendment_git_commit",
    )
    reviewed_implementation_commit = _git_commit(
        report["reviewed_implementation_commit"],
        name="G0 reviewed_implementation_commit",
    )
    science_protocol_git_commit = _git_commit(
        report["execution_science_protocol_git_commit"],
        name="G0 execution_science_protocol_git_commit",
    )
    science_protocol_implementation_commit = _git_commit(
        report["execution_science_protocol_reviewed_implementation_commit"],
        name="G0 execution_science_protocol_reviewed_implementation_commit",
    )
    if reviewed_implementation_commit == git_commit:
        _fail("G0 amendment binding is self-referential")
    if reviewed_implementation_commit == request_git_commit:
        _fail("G0 request predates amendment acceptance")
    if science_protocol_git_commit == science_protocol_implementation_commit:
        _fail("G0 execution science protocol review is self-referential")
    started = _utc_timestamp(report["started_at"], name="G0 started_at")
    completed = _utc_timestamp(report["completed_at"], name="G0 completed_at")
    if completed < started:
        _fail("G0 completion timestamp precedes its start")
    execution = report["execution"]
    if not isinstance(execution, dict) or set(execution) != EXECUTION_FIELDS:
        _fail("G0 execution fields differ from the completion ABI")
    if expected_execution is not None:
        expected_payload = {
            name: getattr(expected_execution, name, None) for name in EXECUTION_FIELDS
        }
        if execution != expected_payload:
            _fail("G0 report execution differs from the validated running attempt")
    if execution["execution_profile"] != PROFILE_NAME:
        _fail("G0 report used an unreviewed execution profile")
    _integer(execution["attempt"], name="G0 attempt", minimum=1)
    for name in ("allocation_sha256", "manifest_sha256"):
        _sha256(execution[name], name=f"G0 execution {name}")
    if not isinstance(execution["job_id"], str) or not execution["job_id"].startswith("opd-"):
        _fail("G0 execution job_id is not an OPD job identity")
    _validate_cgroup_memory(report["cgroup_memory"])
    inventory = _validate_inventory(report["artifact_inventory"])
    bundle_digest = _sha256(report["artifact_bundle_sha256"], name="G0 bundle sha256")
    if bundle_digest != _sha256_file(bundle_path):
        _fail("G0 artifact bundle hash differs from the report")
    _sha256(report["inner_g0_report_sha256"], name="inner G0 report file sha256")
    _validate_bundle(bundle_path, inventory, report)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AdapterValidationError(f"cannot hash G0 artifact bundle: {error}") from error
    return digest.hexdigest()


def validate_qwen3_v2_g0_completion(completion: Mapping[str, Any], context: Any) -> None:
    """Validate the published G0 report, full artifact bundle, and fixed inputs."""

    if not isinstance(completion, Mapping):
        _fail("G0 completion must be a mapping")
    output_paths = getattr(context, "expected_output_paths", None)
    if not isinstance(output_paths, Mapping) or set(output_paths) != set(OUTPUT_NAMES):
        _fail("G0 output contract is invalid")
    expected_inputs = getattr(context, "expected_input_hashes", None)
    if not isinstance(expected_inputs, Mapping) or tuple(sorted(expected_inputs)) != EXPECTED_INPUT_NAMES:
        _fail("G0 content inputs differ from the reviewed contract")
    for name, digest in expected_inputs.items():
        _sha256(digest, name=f"G0 input {name}")
    if completion.get("input_hashes") != dict(expected_inputs):
        _fail("G0 completion input hashes differ")
    for field in (
        "execution_config_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
    ):
        if completion.get(field) != expected_inputs[field]:
            _fail(f"G0 completion {field} differs from its content input")
    gates = completion.get("scientific_validation")
    if not isinstance(gates, dict) or tuple(sorted(gates)) != GATE_NAMES:
        _fail("G0 completion gate names differ from the reviewed contract")
    if any(value is not True for value in gates.values()):
        _fail("G0 completion contains a failed scientific gate")

    try:
        output_paths[REPORT_NAME].lstat()
        output_paths[BUNDLE_NAME].lstat()
    except FileNotFoundError:
        return
    report, _file_sha256 = read_strict_json(
        output_paths[REPORT_NAME],
        context="Qwen3-v2 G0 report",
        max_bytes=64 * 1024 * 1024,
    )
    _validate_report(
        report,
        expected_inputs=expected_inputs,
        bundle_path=output_paths[BUNDLE_NAME],
        expected_execution=getattr(context, "expected_execution", None),
    )


__all__ = [
    "ARTIFACT_NAMESPACE",
    "ALLOWED_GPU_COUNTS",
    "BATCH_PARTITION_PROTOCOL",
    "BUNDLE_NAME",
    "CERTIFICATION_CONTENT_NAME",
    "CHAT_TEMPLATE_SHA256",
    "CPU_CORE_COUNT",
    "DESCRIPTOR_CONTENT_NAME",
    "ESTIMATED_RUNTIME_SECONDS",
    "EXPECTED_INPUT_NAMES",
    "EXECUTION_SCIENCE_PROTOCOL_CONTENT_NAME",
    "EXECUTION_CLASS_ID",
    "GATE_NAMES",
    "GLOBAL_BATCH_SIZE",
    "GPU_MEMORY_MIB",
    "GPU_MODEL",
    "GPU_UTILIZATION_PCT",
    "MEMORY_MIB",
    "MODEL_REVISION",
    "OUTPUT_NAMES",
    "PROFILE_NAME",
    "PROVENANCE_VALIDATION",
    "PROTOCOL_AMENDMENT_ID",
    "PROTOCOL_AMENDMENT_CONTENT_NAME",
    "REPORT_NAME",
    "TASK_NAME",
    "TEACHER_REVISION",
    "TOKENIZER_FINGERPRINT",
    "UNIT_ID",
    "WORKFLOW_ID_PREFIX",
    "batch_token_contract",
    "execution_context",
    "validate_qwen3_v2_g0_completion",
]
