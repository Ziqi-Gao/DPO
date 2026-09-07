"""Reviewed per-experiment bindings for a reusable execution-safety class.

An execution-safety certificate answers whether a fixed distributed execution
class is safe for its certified world sizes.  This artifact answers the
separate question of which scientific protocol/configuration one invocation is
meant to run.  It deliberately contains no resource request or GPU selection.

The resolver below implements the project's deliberately small two-commit
review contract.  The reviewed implementation commit contains the proposed
artifact; one later commit changes only reviewed metadata.  Candidate E may
share that commit with the separately validated execution-class acceptance;
future science-only reviews need change only this artifact (and optionally the
handoff).  Later documentation and review commits may exist, but changes to
project source, scripts, or configuration require a new science review.  The
reviewed artifact itself must remain byte-for-byte unchanged.  Distributed
execution safety is *not* re-proved here: that is the execution-class
fingerprint's job.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.git_provenance import (
    require_git_bytes,
    require_git_output,
)
from posttrain_circuits.artifacts.io import read_regular_bytes_nofollow


SCHEMA_VERSION = 1
KIND = "opd_execution_science_protocol"
REVIEW_MECHANISM = "implementation_commit_then_review_only_acceptance_commit_v1"
SUPPORTED_WORLD_SIZES = (1, 2, 3, 4)
SCIENCE_PROTOCOL_DIRECTORY = PurePosixPath("prereg/execution_science")
HANDOFF_RELATIVE_PATH = "docs/refactor/current_handoff.md"
JOINT_EXECUTION_CLASS_REVIEW_PATHS = frozenset(
    {
        "prereg/amendments/qwen3_v2_g0_execution_class_v2.yaml",
        (
            "prereg/execution_safety/"
            "qwen3_v2_elastic_training_v1.certification.yaml"
        ),
    }
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SCIENCE_PROTOCOL_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\.yaml\Z")
GPU_COUNT_IDENTITY_TOKEN = re.compile(
    r"(?:^|[-_.])(?:w(?:orld)?[-_]?[1-4]|gpu[-_]?[1-4]|[1-4][-_]?gpus?)(?:$|[-_.])",
    flags=re.IGNORECASE,
)
OVERRIDE_KEY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]*(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)*\Z"
)

# This allowlist mirrors the project's relocatable config binding.  Adding a
# locator is a reviewed code change: callers cannot supply ad-hoc exclusions.
APPROVED_STORAGE_LOCATOR_PATHS: tuple[tuple[str, ...], ...] = (
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

# These mappings identify scheduler/run provenance, not scientific behavior.
# Whole mappings are removed so job IDs, attempt numbers, allocation manifests,
# and scheduler-provided output locations cannot perturb scientific identity.
SCHEDULER_PROVENANCE_PATHS: tuple[tuple[str, ...], ...] = (
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

APPROVED_STORAGE_LOCATOR_NAMES = tuple(
    ".".join(parts) for parts in APPROVED_STORAGE_LOCATOR_PATHS
)
SCHEDULER_PROVENANCE_NAMES = tuple(
    ".".join(parts) for parts in SCHEDULER_PROVENANCE_PATHS
)

PROPOSED_REVIEW = {
    "status": "proposed",
    "reviewed_implementation_commit": None,
    "reviewer": None,
    "reviewed_at_utc": None,
    "rationale": None,
}

_FORBIDDEN_OVERRIDE_COMPONENTS = frozenset(
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
        "gpu_uuid",
        "gpus",
        "host_memory",
        "host_memory_mib",
        "local_rank",
        "memory",
        "memory_mib",
        "node_rank",
        "nproc_per_node",
        "num_cpus",
        "num_gpu",
        "num_gpus",
        "omp_num_threads",
        "mkl_num_threads",
        "pci_bus_id",
        "pci_bus_ids",
        "per_rank_threads",
        "rank",
        "resources",
        "supported_world_sizes",
        "thread_count",
        "threads",
        "utilization",
        "gpu_memory_utilization",
        "world_size",
        "world_sizes",
    }
)
_FORBIDDEN_OVERRIDE_PREFIXES = (
    "scheduler",
    "server_scheduler",
    "gpu_",
    "physical_gpu",
)
_FORBIDDEN_CONFIG_FIELD_NAMES = _FORBIDDEN_OVERRIDE_COMPONENTS | frozenset(
    {"physical_gpu", "physical_gpu_index", "physical_gpu_uuid"}
)
_FORBIDDEN_COLLAPSED_FIELD_NAMES = frozenset(
    re.sub(r"[^a-z0-9]", "", field)
    for field in _FORBIDDEN_CONFIG_FIELD_NAMES
)


class ExecutionScienceProtocolError(ValueError):
    """The per-experiment science protocol is malformed or unreviewed."""


@dataclass(frozen=True)
class ExecutionScienceProtocolBinding:
    payload: Mapping[str, Any]
    protocol_id: str
    unit_id: str
    seed: int
    execution_class_id: str
    execution_safety_fingerprint_sha256: str
    storage_neutral_resolved_config_sha256: str
    hydra_override_vector: tuple[str, ...]
    review_status: str
    reviewed_implementation_commit: str | None
    artifact_sha256: str


@dataclass(frozen=True)
class ResolvedExecutionScienceProtocol:
    """One accepted, Git-reviewed and byte-stable science protocol."""

    path: Path
    relative_path: str
    raw: bytes
    sha256: str
    acceptance_commit: str
    reviewed_implementation_commit: str
    binding: ExecutionScienceProtocolBinding


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ExecutionScienceProtocolError(
                "execution science protocol keys must be strings"
            )
        if key in result:
            raise ExecutionScienceProtocolError(
                f"execution science protocol contains duplicate key {key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _validate_json_value(value: object, *, context: str = "value") -> None:
    if value is None or isinstance(value, bool | str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExecutionScienceProtocolError(f"{context} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, context=f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExecutionScienceProtocolError(
                    f"{context} mapping keys must be strings"
                )
            _validate_json_value(item, context=f"{context}.{key}")
        return
    raise ExecutionScienceProtocolError(f"{context} must be strict JSON data")


def _strict_json(raw: bytes) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExecutionScienceProtocolError(
                    f"execution science protocol contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExecutionScienceProtocolError(
                    f"execution science protocol contains non-finite value {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutionScienceProtocolError(
            "execution science protocol is not strict UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutionScienceProtocolError(
            "execution science protocol must be a JSON mapping"
        )
    _validate_json_value(payload, context="execution science protocol")
    return payload


def _strict_yaml(raw: bytes) -> dict[str, Any]:
    try:
        payload = yaml.load(
            raw.decode("utf-8", errors="strict"), Loader=_UniqueKeyLoader
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ExecutionScienceProtocolError(
            "execution science protocol is not strict UTF-8 YAML"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutionScienceProtocolError(
            "execution science protocol must be a YAML mapping"
        )
    _validate_json_value(payload, context="execution science protocol")
    return payload


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ExecutionScienceProtocolError(f"{name} is not a canonical identifier")
    return value


def validate_count_neutral_identity(value: object, *, name: str) -> str:
    """Require an opaque identifier that does not steer a 1--4 GPU choice."""

    identity = _require_identifier(value, name=name)
    if GPU_COUNT_IDENTITY_TOKEN.search(identity):
        raise ExecutionScienceProtocolError(
            f"{name} must not encode a GPU count"
        )
    return identity


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ExecutionScienceProtocolError(f"{name} must be a lowercase SHA-256")
    return value


def _require_relative_artifact_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ExecutionScienceProtocolError(f"{name} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ExecutionScienceProtocolError(f"{name} must be a canonical relative path")
    return value


def _remove_path(payload: dict[str, Any], parts: tuple[str, ...]) -> None:
    parent: object = payload
    for part in parts[:-1]:
        if not isinstance(parent, dict) or part not in parent:
            return
        parent = parent[part]
    if isinstance(parent, dict):
        parent.pop(parts[-1], None)


def _reject_resource_steering_fields(value: object, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower().replace("-", "_")
            collapsed = re.sub(r"[^a-z0-9]", "", key.lower())
            resource_shaped = (
                lowered in _FORBIDDEN_CONFIG_FIELD_NAMES
                or collapsed in _FORBIDDEN_COLLAPSED_FIELD_NAMES
                or lowered.startswith("physical_gpu")
                or lowered.startswith("gpu_count")
                or lowered.startswith("gpu_uuid")
                or lowered.startswith("gpu_index")
                or lowered.startswith("cuda_device")
                or lowered.startswith("device_map")
                or lowered.startswith("nproc_per_node")
                or lowered.startswith("world_size")
                or (
                    lowered.startswith("cpu_")
                    and any(token in lowered for token in ("core", "thread"))
                )
                or (
                    lowered.startswith(("host_memory", "gpu_memory"))
                    and any(token in lowered for token in ("mib", "gib", "bytes"))
                )
            )
            if resource_shaped:
                dotted = ".".join((*path, key))
                raise ExecutionScienceProtocolError(
                    f"resolved config contains forbidden resource steering field {dotted!r}"
                )
            _reject_resource_steering_fields(item, path=(*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_resource_steering_fields(item, path=(*path, str(index)))


def canonical_science_config_projection(
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the storage- and scheduler-provenance-neutral science config.

    Only the checked-in allowlists above are excluded.  In particular, seed and
    other scientific parameters remain part of this digest even though they do
    not invalidate the separately checked execution-safety fingerprint.
    """

    if not isinstance(resolved_config, Mapping):
        raise ExecutionScienceProtocolError("resolved config must be a mapping")
    materialized = copy.deepcopy(dict(resolved_config))
    _validate_json_value(materialized, context="resolved config")
    for parts in APPROVED_STORAGE_LOCATOR_PATHS:
        _remove_path(materialized, parts)
    for parts in SCHEDULER_PROVENANCE_PATHS:
        _remove_path(materialized, parts)
    _reject_resource_steering_fields(materialized)
    return materialized


def canonical_science_config_sha256(resolved_config: Mapping[str, Any]) -> str:
    """Digest the canonical per-experiment science configuration."""

    return sha256_value(canonical_science_config_projection(resolved_config))


def validate_hydra_override_vector(value: object) -> tuple[str, ...]:
    """Validate a deterministic, allocation-neutral Hydra-style override vector."""

    if not isinstance(value, list | tuple):
        raise ExecutionScienceProtocolError("Hydra override vector must be a sequence")
    if len(value) > 128:
        raise ExecutionScienceProtocolError("Hydra override vector is too long")
    result: list[str] = []
    seen_keys: set[str] = set()
    for index, override in enumerate(value):
        if not isinstance(override, str) or not override or len(override) > 1024:
            raise ExecutionScienceProtocolError(
                f"Hydra override {index} must be a bounded non-empty string"
            )
        if override != override.strip() or any(ord(character) < 32 for character in override):
            raise ExecutionScienceProtocolError(
                f"Hydra override {index} contains non-canonical whitespace"
            )
        if "=" not in override:
            raise ExecutionScienceProtocolError(
                f"Hydra override {index} must use key=value syntax"
            )
        key, serialized_value = override.split("=", 1)
        if OVERRIDE_KEY.fullmatch(key) is None or not serialized_value:
            raise ExecutionScienceProtocolError(
                f"Hydra override {index} is not canonical key=value syntax"
            )
        lowered_parts = tuple(part.lower().replace("-", "_") for part in key.split("."))
        collapsed_parts = tuple(
            re.sub(r"[^a-z0-9]", "", part.lower()) for part in key.split(".")
        )
        if any(
            part in _FORBIDDEN_OVERRIDE_COMPONENTS
            or any(part.startswith(prefix) for prefix in _FORBIDDEN_OVERRIDE_PREFIXES)
            or collapsed in _FORBIDDEN_COLLAPSED_FIELD_NAMES
            for part, collapsed in zip(lowered_parts, collapsed_parts, strict=True)
        ):
            raise ExecutionScienceProtocolError(
                f"Hydra override {key!r} contains forbidden scheduler/resource steering"
            )
        if key in seen_keys:
            raise ExecutionScienceProtocolError(
                f"Hydra override key {key!r} is duplicated"
            )
        seen_keys.add(key)
        try:
            decoded = yaml.safe_load(serialized_value)
        except yaml.YAMLError as error:
            raise ExecutionScienceProtocolError(
                f"Hydra override {key!r} has invalid YAML value"
            ) from error
        _validate_json_value(decoded, context=f"Hydra override {key!r}")
        _reject_resource_steering_fields(decoded, path=(key,))
        result.append(override)
    return tuple(result)


def _validate_review(
    review: object, *, require_accepted: bool
) -> tuple[str, str | None]:
    if not isinstance(review, dict) or set(review) != set(PROPOSED_REVIEW):
        raise ExecutionScienceProtocolError(
            "execution science protocol review fields differ from schema"
        )
    status = review["status"]
    if status == "proposed":
        if review != PROPOSED_REVIEW:
            raise ExecutionScienceProtocolError(
                "proposed execution science protocol has review metadata"
            )
        if require_accepted:
            raise ExecutionScienceProtocolError(
                "execution science protocol remains proposed"
            )
        return "proposed", None
    if status != "accepted":
        raise ExecutionScienceProtocolError(
            "execution science protocol review status is invalid"
        )
    implementation_commit = review["reviewed_implementation_commit"]
    if (
        not isinstance(implementation_commit, str)
        or GIT_COMMIT.fullmatch(implementation_commit) is None
    ):
        raise ExecutionScienceProtocolError(
            "accepted science protocol lacks a reviewed implementation commit"
        )
    for field in ("reviewer", "rationale"):
        if not isinstance(review[field], str) or not review[field].strip():
            raise ExecutionScienceProtocolError(
                f"accepted science protocol lacks {field}"
            )
    reviewed_at = review["reviewed_at_utc"]
    if not isinstance(reviewed_at, str) or not reviewed_at.endswith("Z"):
        raise ExecutionScienceProtocolError(
            "science protocol review time must be explicit UTC"
        )
    try:
        parsed = datetime.fromisoformat(reviewed_at[:-1] + "+00:00")
    except ValueError as error:
        raise ExecutionScienceProtocolError(
            "science protocol review time is invalid"
        ) from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ExecutionScienceProtocolError(
            "science protocol review time must use UTC"
        )
    return "accepted", implementation_commit


def _validate_payload(
    payload: dict[str, Any], *, require_accepted: bool
) -> tuple[str, str | None]:
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
    if set(payload) != expected_top:
        raise ExecutionScienceProtocolError(
            "execution science protocol top-level fields differ from schema"
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ExecutionScienceProtocolError(
            "execution science protocol schema_version must be 1"
        )
    if payload["kind"] != KIND:
        raise ExecutionScienceProtocolError("execution science protocol kind differs")
    validate_count_neutral_identity(payload["protocol_id"], name="protocol_id")

    scope = payload["scope"]
    if not isinstance(scope, dict) or set(scope) != {"unit_id", "seed"}:
        raise ExecutionScienceProtocolError(
            "execution science protocol scope fields differ from schema"
        )
    validate_count_neutral_identity(scope["unit_id"], name="unit_id")
    seed = scope["seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ExecutionScienceProtocolError("science protocol seed is invalid")

    execution_class = payload["execution_class"]
    expected_execution_class = {
        "execution_class_id",
        "descriptor_path",
        "descriptor_sha256",
        "fingerprint_sha256",
        "certification_id",
        "certification_path",
        "certification_core_sha256",
        "supported_world_sizes",
    }
    if not isinstance(execution_class, dict) or set(execution_class) != expected_execution_class:
        raise ExecutionScienceProtocolError(
            "execution class reference fields differ from schema"
        )
    _require_identifier(execution_class["execution_class_id"], name="execution_class_id")
    _require_identifier(execution_class["certification_id"], name="certification_id")
    _require_relative_artifact_path(execution_class["descriptor_path"], name="descriptor_path")
    _require_relative_artifact_path(
        execution_class["certification_path"], name="certification_path"
    )
    for field in (
        "descriptor_sha256",
        "fingerprint_sha256",
        "certification_core_sha256",
    ):
        _require_sha256(execution_class[field], name=f"execution_class.{field}")
    world_sizes = execution_class["supported_world_sizes"]
    if (
        not isinstance(world_sizes, list)
        or any(type(value) is not int for value in world_sizes)
        or tuple(world_sizes) != SUPPORTED_WORLD_SIZES
    ):
        raise ExecutionScienceProtocolError(
            "execution class must reference certified world sizes 1, 2, 3, and 4"
        )

    science_config = payload["science_config"]
    expected_science_config = {
        "storage_neutral_resolved_config_sha256",
        "hydra_override_vector",
        "excluded_storage_locator_paths",
        "excluded_scheduler_provenance_paths",
    }
    if not isinstance(science_config, dict) or set(science_config) != expected_science_config:
        raise ExecutionScienceProtocolError(
            "science config binding fields differ from schema"
        )
    _require_sha256(
        science_config["storage_neutral_resolved_config_sha256"],
        name="science_config.storage_neutral_resolved_config_sha256",
    )
    validate_hydra_override_vector(science_config["hydra_override_vector"])
    if science_config["excluded_storage_locator_paths"] != list(
        APPROVED_STORAGE_LOCATOR_NAMES
    ):
        raise ExecutionScienceProtocolError(
            "science config storage-locator exclusions differ from the approved set"
        )
    if science_config["excluded_scheduler_provenance_paths"] != list(
        SCHEDULER_PROVENANCE_NAMES
    ):
        raise ExecutionScienceProtocolError(
            "science config scheduler-provenance exclusions differ from the approved set"
        )

    expected_review_contract = {
        "mechanism": REVIEW_MECHANISM,
        "acceptance_commit_self_reference": "forbidden",
        "non_review_change_at_acceptance": "forbidden",
    }
    if payload["review_contract"] != expected_review_contract:
        raise ExecutionScienceProtocolError(
            "science protocol review contract differs from schema"
        )
    return _validate_review(payload["review"], require_accepted=require_accepted)


def build_execution_science_protocol(
    *,
    protocol_id: str,
    unit_id: str,
    seed: int,
    execution_class_id: str,
    descriptor_path: str,
    descriptor_sha256: str,
    fingerprint_sha256: str,
    certification_id: str,
    certification_path: str,
    certification_core_sha256: str,
    resolved_config: Mapping[str, Any],
    hydra_override_vector: Sequence[str],
    review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one proposed or accepted per-experiment science binding."""

    overrides = validate_hydra_override_vector(hydra_override_vector)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "protocol_id": protocol_id,
        "scope": {
            "unit_id": unit_id,
            "seed": seed,
        },
        "execution_class": {
            "execution_class_id": execution_class_id,
            "descriptor_path": descriptor_path,
            "descriptor_sha256": descriptor_sha256,
            "fingerprint_sha256": fingerprint_sha256,
            "certification_id": certification_id,
            "certification_path": certification_path,
            "certification_core_sha256": certification_core_sha256,
            "supported_world_sizes": list(SUPPORTED_WORLD_SIZES),
        },
        "science_config": {
            "storage_neutral_resolved_config_sha256": (
                canonical_science_config_sha256(resolved_config)
            ),
            "hydra_override_vector": list(overrides),
            "excluded_storage_locator_paths": list(APPROVED_STORAGE_LOCATOR_NAMES),
            "excluded_scheduler_provenance_paths": list(
                SCHEDULER_PROVENANCE_NAMES
            ),
        },
        "review_contract": {
            "mechanism": REVIEW_MECHANISM,
            "acceptance_commit_self_reference": "forbidden",
            "non_review_change_at_acceptance": "forbidden",
        },
        "review": copy.deepcopy(dict(review) if review is not None else PROPOSED_REVIEW),
    }
    _validate_json_value(payload, context="execution science protocol")
    _validate_payload(payload, require_accepted=False)
    return payload


def load_execution_science_protocol_json_bytes(
    raw: bytes, *, require_accepted: bool = False
) -> dict[str, Any]:
    payload = _strict_json(raw)
    _validate_payload(payload, require_accepted=require_accepted)
    return payload


def load_execution_science_protocol_yaml_bytes(
    raw: bytes, *, require_accepted: bool = False
) -> dict[str, Any]:
    payload = _strict_yaml(raw)
    _validate_payload(payload, require_accepted=require_accepted)
    return payload


def validate_execution_science_protocol_bytes(
    raw: bytes,
    *,
    format: str = "yaml",
    require_accepted: bool = False,
) -> ExecutionScienceProtocolBinding:
    if format == "yaml":
        payload = load_execution_science_protocol_yaml_bytes(
            raw, require_accepted=require_accepted
        )
    elif format == "json":
        payload = load_execution_science_protocol_json_bytes(
            raw, require_accepted=require_accepted
        )
    else:
        raise ExecutionScienceProtocolError(
            "execution science protocol format must be 'json' or 'yaml'"
        )
    status, implementation_commit = _validate_review(
        payload["review"], require_accepted=require_accepted
    )
    return ExecutionScienceProtocolBinding(
        payload=copy.deepcopy(payload),
        protocol_id=payload["protocol_id"],
        unit_id=payload["scope"]["unit_id"],
        seed=payload["scope"]["seed"],
        execution_class_id=payload["execution_class"]["execution_class_id"],
        execution_safety_fingerprint_sha256=payload["execution_class"][
            "fingerprint_sha256"
        ],
        storage_neutral_resolved_config_sha256=payload["science_config"][
            "storage_neutral_resolved_config_sha256"
        ],
        hydra_override_vector=tuple(
            payload["science_config"]["hydra_override_vector"]
        ),
        review_status=status,
        reviewed_implementation_commit=implementation_commit,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
    )


def execution_science_protocol_core_sha256(payload: Mapping[str, Any]) -> str:
    """Return the review-neutral identity of a validated science protocol."""

    candidate = copy.deepcopy(dict(payload))
    _validate_payload(candidate, require_accepted=False)
    candidate.pop("review")
    return sha256_value(candidate)


def validate_execution_science_protocol_review_transition(
    *,
    proposed: Mapping[str, Any],
    accepted: Mapping[str, Any],
    implementation_commit: str,
    acceptance_commit: str,
) -> None:
    """Prove a pure review transition under a caller-proven two-commit lineage.

    ``acceptance_commit`` is supplied instead of stored in the artifact, which
    avoids making the acceptance commit self-referential.  This function does
    not query Git or assert ancestry.
    """

    if not isinstance(implementation_commit, str) or GIT_COMMIT.fullmatch(
        implementation_commit
    ) is None:
        raise ExecutionScienceProtocolError("implementation commit is invalid")
    if not isinstance(acceptance_commit, str) or GIT_COMMIT.fullmatch(
        acceptance_commit
    ) is None:
        raise ExecutionScienceProtocolError("acceptance commit is invalid")
    if acceptance_commit == implementation_commit:
        raise ExecutionScienceProtocolError(
            "acceptance commit must differ from the implementation commit"
        )
    proposed_payload = copy.deepcopy(dict(proposed))
    accepted_payload = copy.deepcopy(dict(accepted))
    _validate_payload(proposed_payload, require_accepted=False)
    _validate_payload(accepted_payload, require_accepted=True)
    if proposed_payload["review"] != PROPOSED_REVIEW:
        raise ExecutionScienceProtocolError(
            "implementation commit did not contain a proposed science protocol"
        )
    if (
        accepted_payload["review"]["reviewed_implementation_commit"]
        != implementation_commit
    ):
        raise ExecutionScienceProtocolError(
            "accepted science protocol names a different implementation commit"
        )
    normalized = copy.deepcopy(accepted_payload)
    normalized["review"] = copy.deepcopy(PROPOSED_REVIEW)
    if normalized != proposed_payload:
        raise ExecutionScienceProtocolError(
            "accepted science protocol changed non-review terms"
        )


def validate_execution_science_protocol_relative_path(configured_path: object) -> str:
    """Confine protocol artifacts to one count-neutral checked-in directory."""

    if not isinstance(configured_path, str) or not configured_path or "\\" in configured_path:
        raise ExecutionScienceProtocolError(
            "execution science protocol path must be a relative POSIX path"
        )
    path = PurePosixPath(configured_path)
    if (
        path.is_absolute()
        or str(path) != configured_path
        or len(path.parts) != 3
        or path.parent != SCIENCE_PROTOCOL_DIRECTORY
        or SCIENCE_PROTOCOL_FILENAME.fullmatch(path.name) is None
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ExecutionScienceProtocolError(
            "execution science protocol path must match "
            "prereg/execution_science/<canonical>.yaml"
        )
    validate_count_neutral_identity(path.stem, name="science protocol filename")
    return configured_path


def _git_output(code_root: Path, arguments: tuple[str, ...], *, context: str) -> str:
    try:
        return require_git_output(code_root, arguments)
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        raise ExecutionScienceProtocolError(
            f"Git could not {context}"
        ) from error


def _git_bytes(code_root: Path, arguments: tuple[str, ...], *, context: str) -> bytes:
    try:
        return require_git_bytes(code_root, arguments)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutionScienceProtocolError(
            f"Git could not {context}"
        ) from error


def _require_clean_git_checkout(code_root: Path) -> str:
    status = _git_output(
        code_root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--ignore-submodules=none",
        ),
        context="inspect the science-protocol checkout",
    )
    if status:
        raise ExecutionScienceProtocolError(
            "accepted execution science protocol requires a clean checkout"
        )
    head = _git_output(
        code_root,
        ("rev-parse", "HEAD"),
        context="resolve the science-protocol HEAD",
    )
    if GIT_COMMIT.fullmatch(head) is None:
        raise ExecutionScienceProtocolError("Git HEAD is not one immutable commit")
    return head


def _commit_changed_paths(code_root: Path, commit: str) -> tuple[str, ...]:
    raw = _git_bytes(
        code_root,
        (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            commit,
            "--",
        ),
        context="inspect the science-protocol acceptance diff",
    )
    if raw and not raw.endswith(b"\0"):
        raise ExecutionScienceProtocolError(
            "Git returned a malformed acceptance path stream"
        )
    try:
        paths = tuple(item.decode("utf-8", errors="strict") for item in raw.split(b"\0")[:-1])
    except UnicodeError as error:
        raise ExecutionScienceProtocolError(
            "science-protocol acceptance paths are not UTF-8"
        ) from error
    if any(not path or path.startswith("/") or ".." in PurePosixPath(path).parts for path in paths):
        raise ExecutionScienceProtocolError(
            "Git returned a non-canonical acceptance path"
        )
    return paths


def _changed_paths_between(
    code_root: Path, ancestor: str, descendant: str
) -> tuple[str, ...]:
    raw = _git_bytes(
        code_root,
        (
            "diff",
            "--name-only",
            "-z",
            f"{ancestor}..{descendant}",
            "--",
        ),
        context="inspect post-review science implementation changes",
    )
    if raw and not raw.endswith(b"\0"):
        raise ExecutionScienceProtocolError(
            "Git returned a malformed science implementation path stream"
        )
    try:
        paths = tuple(
            item.decode("utf-8", errors="strict")
            for item in raw.split(b"\0")
            if item
        )
    except UnicodeError as error:
        raise ExecutionScienceProtocolError(
            "science implementation paths are not UTF-8"
        ) from error
    if any(
        path.startswith("/") or ".." in PurePosixPath(path).parts
        for path in paths
    ):
        raise ExecutionScienceProtocolError(
            "Git returned a non-canonical science implementation path"
        )
    return paths


def _is_ancestor(code_root: Path, ancestor: str, descendant: str) -> bool:
    common = _git_output(
        code_root,
        ("merge-base", ancestor, descendant),
        context="verify science-protocol ancestry",
    )
    return common == ancestor


def resolve_accepted_execution_science_protocol(
    *,
    code_root: Path,
    configured_path: str,
    expected_head: str | None = None,
) -> ResolvedExecutionScienceProtocol:
    """Resolve one accepted protocol under the minimal two-commit Git contract."""

    relative_path = validate_execution_science_protocol_relative_path(configured_path)
    try:
        root = Path(code_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ExecutionScienceProtocolError(
            "execution science protocol code root is unavailable"
        ) from error
    path = root / relative_path
    try:
        raw = read_regular_bytes_nofollow(
            path,
            context="execution science protocol",
            max_bytes=4 * 1024 * 1024,
        )
    except ValueError as error:
        raise ExecutionScienceProtocolError(str(error)) from error
    binding = validate_execution_science_protocol_bytes(
        raw,
        format="yaml",
        require_accepted=True,
    )
    implementation = binding.reviewed_implementation_commit
    assert implementation is not None
    head = _require_clean_git_checkout(root)
    if expected_head is not None and head != expected_head:
        raise ExecutionScienceProtocolError(
            "science-protocol validation HEAD differs from execution provenance"
        )
    if not _is_ancestor(root, implementation, head) or implementation == head:
        raise ExecutionScienceProtocolError(
            "accepted science protocol does not descend from its reviewed implementation"
        )
    proposed_raw = _git_bytes(
        root,
        ("show", f"{implementation}:{relative_path}"),
        context="read the proposed science protocol",
    )
    proposed = load_execution_science_protocol_yaml_bytes(proposed_raw)
    if proposed["review"] != PROPOSED_REVIEW:
        raise ExecutionScienceProtocolError(
            "reviewed implementation commit lacks the proposed science protocol"
        )
    touching = _git_output(
        root,
        (
            "rev-list",
            "--reverse",
            "--ancestry-path",
            f"{implementation}..{head}",
            "--",
            relative_path,
        ),
        context="locate the science-protocol acceptance commit",
    ).splitlines()
    if len(touching) != 1 or GIT_COMMIT.fullmatch(touching[0]) is None:
        raise ExecutionScienceProtocolError(
            "science protocol must change exactly once after its implementation"
        )
    acceptance = touching[0]
    changed_paths = set(_commit_changed_paths(root, acceptance))
    allowed_acceptance_paths = {
        relative_path,
        HANDOFF_RELATIVE_PATH,
        *JOINT_EXECUTION_CLASS_REVIEW_PATHS,
    }
    if relative_path not in changed_paths or not changed_paths <= allowed_acceptance_paths:
        raise ExecutionScienceProtocolError(
            "science-protocol acceptance commit contains non-review changes"
        )
    accepted_raw = _git_bytes(
        root,
        ("show", f"{acceptance}:{relative_path}"),
        context="read the accepted science protocol",
    )
    accepted = load_execution_science_protocol_yaml_bytes(
        accepted_raw, require_accepted=True
    )
    validate_execution_science_protocol_review_transition(
        proposed=proposed,
        accepted=accepted,
        implementation_commit=implementation,
        acceptance_commit=acceptance,
    )
    changed_after_review = set(_changed_paths_between(root, implementation, head))
    scientific_implementation_changes = sorted(
        path
        for path in changed_after_review
        if path.startswith(("configs/", "scripts/", "src/"))
    )
    if scientific_implementation_changes:
        raise ExecutionScienceProtocolError(
            "scientific implementation changed after review: "
            f"{scientific_implementation_changes}"
        )
    if accepted_raw != raw:
        raise ExecutionScienceProtocolError(
            "accepted science protocol artifact changed after review"
        )
    if binding.artifact_sha256 != hashlib.sha256(raw).hexdigest():
        raise ExecutionScienceProtocolError(
            "execution science protocol content identity is inconsistent"
        )
    if _git_output(
        root,
        ("rev-parse", "HEAD"),
        context="recheck the science-protocol HEAD",
    ) != head:
        raise ExecutionScienceProtocolError(
            "science-protocol checkout changed during validation"
        )
    try:
        final_raw = read_regular_bytes_nofollow(
            path,
            context="execution science protocol",
            max_bytes=4 * 1024 * 1024,
        )
    except ValueError as error:
        raise ExecutionScienceProtocolError(str(error)) from error
    if final_raw != raw:
        raise ExecutionScienceProtocolError(
            "execution science protocol bytes changed during validation"
        )
    return ResolvedExecutionScienceProtocol(
        path=path,
        relative_path=relative_path,
        raw=raw,
        sha256=binding.artifact_sha256,
        acceptance_commit=acceptance,
        reviewed_implementation_commit=implementation,
        binding=binding,
    )


__all__ = [
    "APPROVED_STORAGE_LOCATOR_NAMES",
    "APPROVED_STORAGE_LOCATOR_PATHS",
    "ExecutionScienceProtocolBinding",
    "ExecutionScienceProtocolError",
    "ResolvedExecutionScienceProtocol",
    "KIND",
    "PROPOSED_REVIEW",
    "REVIEW_MECHANISM",
    "SCHEMA_VERSION",
    "SCHEDULER_PROVENANCE_NAMES",
    "SCHEDULER_PROVENANCE_PATHS",
    "SCIENCE_PROTOCOL_DIRECTORY",
    "SUPPORTED_WORLD_SIZES",
    "build_execution_science_protocol",
    "canonical_science_config_projection",
    "canonical_science_config_sha256",
    "execution_science_protocol_core_sha256",
    "resolve_accepted_execution_science_protocol",
    "load_execution_science_protocol_json_bytes",
    "load_execution_science_protocol_yaml_bytes",
    "validate_count_neutral_identity",
    "validate_execution_science_protocol_bytes",
    "validate_execution_science_protocol_relative_path",
    "validate_execution_science_protocol_review_transition",
    "validate_hydra_override_vector",
]
