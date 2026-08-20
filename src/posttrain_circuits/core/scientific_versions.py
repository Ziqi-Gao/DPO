"""Active scientific-schema versions and fail-closed compatibility checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from posttrain_circuits.core.hashing import sha256_value

CORE_PREREG_VERSION = "core_v2"
GENERATOR_VERSION = "proofgraph-v3-signed-paired"
LABEL_SEMANTICS = "signed_entailment"
CIRCUIT_PROBE_SCHEMA_VERSION = "circuit-probe-v2-stage-sequence"
DATASET_SCHEMA_VERSION = "proofgraph-dataset-v2-paired"
PROBE_COHORT_SCHEMA_VERSION = "probe-cohort-v2-paired"
TRAJECTORY_STORE_VERSION = 2
ROLLOUT_GENERATION_VERSION = "hf-rollout-v2-leftpad-eosclean-per-trajectory-rng"


def scientific_compatibility_fields(
    prereg_version: str = CORE_PREREG_VERSION,
) -> dict[str, str]:
    """Return the fields every core-v2 scientific artifact must bind."""

    return {
        "prereg_version": prereg_version,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "circuit_probe_schema_version": CIRCUIT_PROBE_SCHEMA_VERSION,
    }


def require_core_v2_artifact(
    artifact: Mapping[str, Any],
    *,
    require_circuit_schema: bool = False,
    require_hash: bool = False,
) -> None:
    """Reject pre-v2 or partially versioned artifacts in formal loaders."""

    require_scientific_artifact(
        artifact,
        expected_prereg_version=CORE_PREREG_VERSION,
        require_circuit_schema=require_circuit_schema,
        require_hash=require_hash,
    )


def require_scientific_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_prereg_version: str,
    require_circuit_schema: bool = False,
    require_hash: bool = False,
) -> None:
    """Reject artifacts that are stale relative to the configured preregistration."""

    expected = scientific_compatibility_fields(expected_prereg_version)
    keys: tuple[str, ...] = ("prereg_version", "generator_version", "label_semantics")
    if require_circuit_schema:
        keys = (*keys, "circuit_probe_schema_version")
    mismatches = {
        key: {"expected": expected[key], "observed": artifact.get(key)}
        for key in keys
        if artifact.get(key) != expected[key]
    }
    if mismatches:
        raise ValueError(f"artifact is incompatible with {expected_prereg_version}: {mismatches}")
    if require_hash:
        observed = artifact.get("sha256")
        content = {key: value for key, value in artifact.items() if key != "sha256"}
        expected_hash = sha256_value(content)
        if observed != expected_hash:
            raise ValueError(
                f"scientific artifact SHA-256 mismatch: expected={expected_hash}, observed={observed}"
            )
