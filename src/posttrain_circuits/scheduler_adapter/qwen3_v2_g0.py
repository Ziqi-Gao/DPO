"""Scientific completion contract for the fixed two-GPU Qwen3-v2 G0 gate."""

from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.strict_json import read_strict_json


TASK_NAME = "qwen3_v2_g0"
PROFILE_NAME = "qwen3-v2-g0-2gpu"
WORKFLOW_ID = "qwen3-v2-g0-v1"
UNIT_ID = "g0"
REPORT_NAME = "g0.json"
BUNDLE_NAME = "g0_artifacts.tar"
OUTPUT_NAMES = tuple(sorted((REPORT_NAME, BUNDLE_NAME)))

GPU_COUNT = 2
CPU_CORE_COUNT = 16
MEMORY_MIB = 196608
GPU_MEMORY_MIB = 81920
GPU_UTILIZATION_PCT = 95
GPU_MODEL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
ESTIMATED_RUNTIME_SECONDS = 43200.0
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
GPU_PREFLIGHT_COMPLETION_CONTENT_NAME = "gpu_preflight_completion_sha256"
GPU_PREFLIGHT_REPORT_CONTENT_NAME = "gpu_preflight_report_sha256"
EXPECTED_INPUT_NAMES = (
    "config_binding_sha256",
    "execution_config_sha256",
    GPU_PREFLIGHT_COMPLETION_CONTENT_NAME,
    GPU_PREFLIGHT_REPORT_CONTENT_NAME,
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
    "g0_semantic_decision",
    "gpu_preflight_binding",
    "memory_headroom",
    "offline_pinned_runtime",
    "protocol_amendment",
    "scientific_artifact_chain",
)

BATCH_TOKEN_CONTRACT = {
    "world_size": GPU_COUNT,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "effective_global_batch_size": 64,
    "token_budget": 2_000_000,
    "token_budget_unit": "global_nonpadding_model_input_tokens_processed",
    "max_optimizer_steps": 120,
}
PROVENANCE_VALIDATION = {
    "preflight_to_execution_metadata_only": True,
    "request_to_execution_metadata_only": True,
    "reviewed_implementation_to_execution_metadata_only": True,
}

REQUIRED_BUNDLE_PATHS = frozenset(
    {
        "anti_shortcut.json",
        "calibration/manifest.json",
        "circuits/final_answer/circuit.json",
        "circuits/final_answer/exact_patching.json",
        "circuits/final_answer/mib_raw/compatibility.json",
        "circuits/first_rule_selection/circuit.json",
        "circuits/first_rule_selection/exact_patching.json",
        "dataset/manifest.json",
        "distributed_resume.json",
        "g0.json",
        "initial_checkpoint.pt",
        "label_leakage.json",
        "probe_scores.json",
        "probes/manifest.json",
        "readiness/readiness.json",
        "resume-a/manifest.json",
        "resume-b/manifest.json",
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
        "git_commit",
        "gpu_preflight_completion_sha256",
        "gpu_preflight_git_commit",
        "gpu_preflight_report_sha256",
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
EXECUTION_CONTEXT = {
    "allocation_visibility": "preserved",
    "distributed_launcher": "accelerate_fsdp_foreground",
    "mode": "server_scheduler_foreground",
    "nccl_p2p_policy": "disabled",
    "visible_device_count": GPU_COUNT,
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
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterValidationError(f"G0 bundle member {path!r} is not strict JSON") from error
    if not isinstance(payload, dict):
        _fail(f"G0 bundle member {path!r} must be a JSON object")
    return payload


def _validate_inner_g0(payload: dict[str, Any], *, report: Mapping[str, Any]) -> None:
    digest = _sha256(payload.get("sha256"), name="inner G0 report sha256")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    if digest != sha256_value(unsigned):
        _fail("inner G0 report self-summary is invalid")
    checks = payload.get("checks")
    if (
        payload.get("phase") != "G0"
        or payload.get("passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
        or checks.get("distributed_checkpoint_resume") is not True
        or checks.get("gpu_preflight") is not True
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
        or payload.get("reviewed_implementation_commit")
        != report["reviewed_implementation_commit"]
        or payload.get("request_git_commit") != report["request_git_commit"]
        or payload.get("batch_token_contract") != BATCH_TOKEN_CONTRACT
    ):
        _fail("inner G0 report differs from the outer fixed protocol binding")


def _validate_bundle(path: Path, inventory: Mapping[str, tuple[str, int]], report: Mapping[str, Any]) -> None:
    observed: dict[str, tuple[str, int]] = {}
    inner_payload: dict[str, Any] | None = None
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
                captured = io.BytesIO() if member_path == "g0.json" else None
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
                    inner_payload = _read_json_member(
                        captured,
                        path="g0.json",
                        maximum=32 * 1024 * 1024,
                    )
    except (OSError, tarfile.TarError) as error:
        raise AdapterValidationError(f"G0 artifact bundle is invalid: {error}") from error
    if observed != dict(inventory):
        _fail("G0 artifact bundle bytes differ from its complete inventory")
    if inner_payload is None:
        _fail("G0 artifact bundle has no inner G0 report")
    if observed["g0.json"][0] != report["inner_g0_report_sha256"]:
        _fail("outer G0 report binds a different inner G0 report")
    _validate_inner_g0(inner_payload, report=report)


def _validate_report(
    report: object,
    *,
    expected_inputs: Mapping[str, str],
    bundle_path: Path,
) -> None:
    if not isinstance(report, dict) or set(report) != REPORT_FIELDS:
        _fail("G0 report fields differ from the reviewed contract")
    digest = _sha256(report["sha256"], name="G0 report sha256")
    unsigned = {key: value for key, value in report.items() if key != "sha256"}
    if digest != sha256_value(unsigned):
        _fail("G0 report self-summary is invalid")
    if (
        report["schema_version"] != 1
        or isinstance(report["schema_version"], bool)
        or report["phase"] != TASK_NAME
        or report["passed"] is not True
        or report["world_size"] != GPU_COUNT
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
        or report["protocol_amendment_id"] != "qwen3_v2_g0_2gpu_v1"
        or report["protocol_amendment_sha256"]
        != expected_inputs[PROTOCOL_AMENDMENT_CONTENT_NAME]
        or report["batch_token_contract"] != BATCH_TOKEN_CONTRACT
        or report["provenance_validation"] != PROVENANCE_VALIDATION
        or report["resolved_config_sha256"] != expected_inputs["resolved_config_sha256"]
        or report["gpu_preflight_report_sha256"]
        != expected_inputs[GPU_PREFLIGHT_REPORT_CONTENT_NAME]
        or report["gpu_preflight_completion_sha256"]
        != expected_inputs[GPU_PREFLIGHT_COMPLETION_CONTENT_NAME]
        or report["execution_context"] != EXECUTION_CONTEXT
    ):
        _fail("G0 report differs from the fixed Qwen3-v2 two-GPU protocol")
    git_commit = _git_commit(report["git_commit"], name="G0 git_commit")
    if report["code_commit"] != git_commit:
        _fail("G0 code_commit differs from git_commit")
    preflight_commit = _git_commit(
        report["gpu_preflight_git_commit"], name="GPU preflight git_commit"
    )
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
    if reviewed_implementation_commit == git_commit:
        _fail("G0 amendment binding is self-referential")
    if reviewed_implementation_commit in {preflight_commit, request_git_commit}:
        _fail("G0 preflight/request predates amendment acceptance")
    started = _utc_timestamp(report["started_at"], name="G0 started_at")
    completed = _utc_timestamp(report["completed_at"], name="G0 completed_at")
    if completed < started:
        _fail("G0 completion timestamp precedes its start")
    execution = report["execution"]
    if not isinstance(execution, dict) or set(execution) != EXECUTION_FIELDS:
        _fail("G0 execution fields differ from the completion ABI")
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
    )


__all__ = [
    "ARTIFACT_NAMESPACE",
    "BATCH_TOKEN_CONTRACT",
    "BUNDLE_NAME",
    "CHAT_TEMPLATE_SHA256",
    "CPU_CORE_COUNT",
    "ESTIMATED_RUNTIME_SECONDS",
    "EXPECTED_INPUT_NAMES",
    "GATE_NAMES",
    "GPU_COUNT",
    "GPU_MEMORY_MIB",
    "GPU_MODEL",
    "GPU_UTILIZATION_PCT",
    "MEMORY_MIB",
    "MODEL_REVISION",
    "OUTPUT_NAMES",
    "PROFILE_NAME",
    "PROVENANCE_VALIDATION",
    "PROTOCOL_AMENDMENT_CONTENT_NAME",
    "REPORT_NAME",
    "TASK_NAME",
    "TEACHER_REVISION",
    "TOKENIZER_FINGERPRINT",
    "UNIT_ID",
    "WORKFLOW_ID",
    "validate_qwen3_v2_g0_completion",
]
