"""Hash-pinned pre-training circuit-probe cohort construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.scientific_versions import (
    PROBE_COHORT_SCHEMA_VERSION,
    require_scientific_artifact,
    scientific_compatibility_fields,
)

COHORTS = ("base_capable", "challenge")
SUBSETS = ("discovery", "validation")


def _subset_manifest(
    cohort: str,
    subset: str,
    rows: list[dict[str, Any]],
    source_split_hash: str,
) -> dict[str, Any]:
    examples = [
        {
            "example_id": str(row["example_id"]),
            "example_sha256": sha256_value(row),
            "example": row,
        }
        for row in rows
    ]
    pair_group_ids = sorted(
        {str(row.get("pair_group_id", row.get("metadata", {}).get("pair_group_id", ""))) for row in rows}
    )
    if any(not value for value in pair_group_ids):
        raise ValueError("probe subset examples require pair_group_id")
    payload: dict[str, Any] = {
        "cohort": cohort,
        "subset": subset,
        "source_split_hash": source_split_hash,
        "num_examples": len(examples),
        "examples": examples,
        "pair_group_count": len(pair_group_ids),
        "pair_group_hash": sha256_value(pair_group_ids),
    }
    payload["sha256"] = sha256_value(payload)
    return payload


def build_probe_cohort_manifest(
    split_rows: dict[str, list[dict[str, Any]]],
    scores: dict[str, dict[str, bool]],
    *,
    source_split_hashes: dict[str, str],
    initial_student_checkpoint_hash: str,
    scoring_manifest_hash: str,
    learnability_evidence_hash: str,
    git_commit: str = "test-unfrozen",
    prereg_commit: str = "test-unfrozen",
    source_artifacts: dict[str, dict[str, str]] | None = None,
    candidate_selection_audit: dict[str, Any] | None = None,
    protocol_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Partition every frozen discovery/validation probe before training.

    Challenge examples must be initially unsolved and independently marked as
    learnable by a frozen pilot/calibration artifact. No probe can be silently
    dropped or shared between cohorts.
    """

    if set(split_rows) != set(SUBSETS) or set(source_split_hashes) != set(SUBSETS):
        raise ValueError("probe cohorts require exactly discovery and validation source splits")
    required_hashes = {
        "initial_student_checkpoint_hash": initial_student_checkpoint_hash,
        "scoring_manifest_hash": scoring_manifest_hash,
        "learnability_evidence_hash": learnability_evidence_hash,
        "git_commit": git_commit,
        "prereg_commit": prereg_commit,
        "created_at": utc_now(),
        "construction_phase": "before_confirmatory_training",
        "training_ancestry": [],
        "source_artifacts": source_artifacts or {},
        **{f"source_{key}_hash": value for key, value in source_split_hashes.items()},
    }
    if any(not str(value).strip() for value in required_hashes.values()):
        raise ValueError("probe cohort provenance hashes must all be non-empty")
    partitions: dict[str, dict[str, list[dict[str, Any]]]] = {
        cohort: {subset: [] for subset in SUBSETS} for cohort in COHORTS
    }
    seen_ids: set[str] = set()
    for subset in SUBSETS:
        for row in split_rows[subset]:
            example_id = str(row.get("example_id", ""))
            if not example_id or example_id in seen_ids:
                raise ValueError(f"probe IDs must be non-empty and globally unique: {example_id!r}")
            seen_ids.add(example_id)
            if example_id not in scores:
                raise ValueError(f"probe {example_id} has no initial-student score")
            score = scores[example_id]
            if set(score) < {"initial_correct", "learnable_after_post_training"}:
                raise ValueError(f"probe {example_id} lacks capability/learnability fields")
            if bool(score["initial_correct"]):
                cohort = "base_capable"
            elif bool(score["learnable_after_post_training"]):
                cohort = "challenge"
            else:
                raise ValueError(f"initially unsolved probe {example_id} lacks frozen learnability evidence")
            partitions[cohort][subset].append(row)
    manifests = {
        cohort: {
            subset: _subset_manifest(
                cohort,
                subset,
                partitions[cohort][subset],
                source_split_hashes[subset],
            )
            for subset in SUBSETS
        }
        for cohort in COHORTS
    }
    for cohort in COHORTS:
        for subset in SUBSETS:
            if manifests[cohort][subset]["num_examples"] < 1:
                raise ValueError(f"probe cohort {cohort}/{subset} cannot be empty")
    payload: dict[str, Any] = {
        "format_version": 2,
        "probe_cohort_schema_version": PROBE_COHORT_SCHEMA_VERSION,
        **scientific_compatibility_fields(str((protocol_bindings or {}).get("prereg_version", "core_v2"))),
        "frozen_before_training": True,
        **required_hashes,
        "selection_rules": {
            "base_capable": "initial_correct == true",
            "challenge": (
                "initial_correct == false and learnable_after_post_training == true "
                "from a frozen pilot/calibration artifact"
            ),
        },
        "candidate_selection_audit": candidate_selection_audit or {},
        "source_split_hashes": source_split_hashes,
        "cohorts": manifests,
        **(protocol_bindings or {}),
    }
    payload["sha256"] = sha256_value(payload)
    return payload


def write_probe_cohort_manifest(output: Path, manifest: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for cohort in COHORTS:
        for subset in SUBSETS:
            atomic_write_json(
                output / cohort / f"{subset}.json",
                manifest["cohorts"][cohort][subset],
            )
    atomic_write_json(output / "manifest.json", manifest)


def validate_probe_cohort_manifest(
    path: Path,
    *,
    expected_initial_checkpoint_hash: str | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("sha256", None)
    if expected != sha256_value(payload):
        raise ValueError("probe cohort manifest hash mismatch")
    payload["sha256"] = expected
    require_scientific_artifact(
        payload,
        expected_prereg_version=str(payload.get("prereg_version", "")),
    )
    if payload.get("probe_cohort_schema_version") != PROBE_COHORT_SCHEMA_VERSION:
        raise ValueError("pre-core-v2 probe cohort manifests are not accepted")
    if payload.get("frozen_before_training") is not True:
        raise ValueError("probe cohorts were not frozen before training")
    if str(payload.get("protocol_track", "")).startswith("qwen3_"):
        namespace = str(payload["protocol_track"]).replace("_", "-", 1)
        qwen3_expected = {
            "artifact_namespace": namespace,
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
        }
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in qwen3_expected.items()
            if payload.get(key) != value
        }
        for key in ("chat_template_sha256", "tokenizer_fingerprint"):
            if len(str(payload.get(key, ""))) != 64:
                mismatches[key] = {"expected": "64-hex binding", "observed": payload.get(key)}
        if mismatches:
            raise ValueError(f"Qwen3 probe cohort protocol binding mismatch: {mismatches}")
    if (
        not str(payload.get("git_commit", "")).strip()
        or not str(payload.get("prereg_commit", "")).strip()
        or payload.get("construction_phase") != "before_confirmatory_training"
        or payload.get("training_ancestry") != []
    ):
        raise ValueError("probe cohorts lack verifiable pre-training Git/prereg provenance")
    if (
        expected_initial_checkpoint_hash is not None
        and payload.get("initial_student_checkpoint_hash") != expected_initial_checkpoint_hash
    ):
        raise ValueError("probe cohort initial checkpoint hash mismatch")
    ids: set[str] = set()
    source_artifacts = payload.get("source_artifacts", {})
    if source_artifacts:
        if not isinstance(source_artifacts, dict) or set(source_artifacts) != set(SUBSETS):
            raise ValueError("probe source artifacts must cover discovery and validation")
        for subset, artifact in source_artifacts.items():
            if not isinstance(artifact, dict):
                raise ValueError(f"probe source artifact is malformed for {subset}")
            root = Path(str(artifact.get("path", "")))
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"probe source artifact is unavailable for {subset}: {root}")
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if source_manifest.get("sha256") != artifact.get("manifest_hash"):
                raise ValueError(f"probe source manifest hash mismatch for {subset}")
            if artifact.get("manifest_hash") != payload["source_split_hashes"][subset]:
                raise ValueError(f"probe source binding differs from subset hash for {subset}")
    elif payload.get("git_commit") != "test-unfrozen":
        raise ValueError("production probe cohorts require immutable source artifact paths")
    candidate_audit = payload.get("candidate_selection_audit", {})
    if candidate_audit:
        if not isinstance(candidate_audit, dict) or set(candidate_audit) != set(SUBSETS):
            raise ValueError("probe candidate audit must cover discovery and validation")
        for subset, audit in candidate_audit.items():
            if not isinstance(audit, dict):
                raise ValueError(f"probe candidate audit is malformed for {subset}")
            expected_audit_hash = audit.get("sha256")
            audit_payload = {key: value for key, value in audit.items() if key != "sha256"}
            if expected_audit_hash != sha256_value(audit_payload):
                raise ValueError(f"probe candidate audit hash mismatch for {subset}")
            selected = int(audit.get("selected_count", -1))
            excluded = int(audit.get("excluded_count", -1))
            if selected + excluded != int(audit.get("candidate_count", -2)):
                raise ValueError(f"probe candidate accounting mismatch for {subset}")
            frozen = sum(int(payload["cohorts"][cohort][subset]["num_examples"]) for cohort in COHORTS)
            if selected != frozen:
                raise ValueError(f"probe selected/frozen count mismatch for {subset}")
    for cohort in COHORTS:
        for subset in SUBSETS:
            manifest = payload["cohorts"][cohort][subset]
            subset_expected = manifest.get("sha256")
            subset_payload = {key: value for key, value in manifest.items() if key != "sha256"}
            if subset_expected != sha256_value(subset_payload):
                raise ValueError(f"probe subset hash mismatch for {cohort}/{subset}")
            if manifest["num_examples"] < 1:
                raise ValueError(f"probe subset is empty for {cohort}/{subset}")
            pair_ids = sorted(
                {
                    str(
                        row["example"].get(
                            "pair_group_id",
                            row["example"].get("metadata", {}).get("pair_group_id", ""),
                        )
                    )
                    for row in manifest["examples"]
                }
            )
            if manifest.get("pair_group_count") != len(pair_ids):
                raise ValueError(f"probe pair-group count mismatch for {cohort}/{subset}")
            if manifest.get("pair_group_hash") != sha256_value(pair_ids):
                raise ValueError(f"probe pair-group hash mismatch for {cohort}/{subset}")
            for row in manifest["examples"]:
                example_id = str(row["example_id"])
                exact = row.get("example")
                if not isinstance(exact, dict) or sha256_value(exact) != row.get("example_sha256"):
                    raise ValueError(f"probe example byte hash mismatch: {example_id}")
                if example_id in ids:
                    raise ValueError(f"probe appears in multiple cohorts/subsets: {example_id}")
                ids.add(example_id)
    return payload


def load_probe_examples(
    path: Path,
    *,
    cohort: str,
    subset: str,
    expected_initial_checkpoint_hash: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cohort not in COHORTS or subset not in SUBSETS:
        raise ValueError(f"invalid probe selection {cohort}/{subset}")
    payload = validate_probe_cohort_manifest(
        path,
        expected_initial_checkpoint_hash=expected_initial_checkpoint_hash,
    )
    subset_manifest = payload["cohorts"][cohort][subset]
    rows = [dict(item["example"]) for item in subset_manifest["examples"]]
    return rows, payload
