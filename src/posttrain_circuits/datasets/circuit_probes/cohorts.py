"""Pair-complete, hash-pinned circuit-probe cohort construction and loading."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import (
    PROBE_COHORT_SCHEMA_VERSION,
    require_scientific_artifact,
    scientific_compatibility_fields,
)
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json, utc_now
from posttrain_circuits.datasets.circuit_probes.contracts import (
    COHORTS,
    SOURCE_SPLITS,
    SUBSETS,
    PairDecision,
    ProbePair,
)
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.proofgraph.family import DatasetFamilyHandle
from posttrain_circuits.datasets.proofgraph.serialization import (
    deserialize_example,
    serialize_examples,
)

COHORT_FORMAT_VERSION = 3
_SCORE_FIELDS = {"initial_correct", "learnable_after_post_training"}
_ANCESTRY_FIELDS = {
    "calibration_checkpoint_sha256",
    "calibration_run_manifest_sha256",
    "calibration_run_id",
    "experiment_binding_sha256",
}
_PROTOCOL_BINDING_FIELDS = {
    "protocol_track",
    "artifact_namespace",
    "prompt_protocol",
    "enable_thinking",
    "chat_template_sha256",
    "tokenizer_fingerprint",
    "prereg_path",
    "prereg_version",
    "prereg_commit",
    "prereg_sha256",
    "code_commit",
    "model_revision",
    "teacher_revision",
    "tokenizer_revision",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def group_complete_pairs(
    examples: list[TaskExample],
    *,
    subset: str,
) -> list[ProbePair]:
    """Aggregate a frozen row order into complete two-sibling signed pairs."""

    if subset not in SUBSETS:
        raise ValueError(f"unknown circuit-probe subset {subset!r}")
    source_split = SOURCE_SPLITS[subset]
    order: list[str] = []
    grouped: dict[str, list[TaskExample]] = {}
    example_ids: set[str] = set()
    for example in examples:
        if not isinstance(example, TaskExample):
            raise TypeError("circuit-probe family rows must be TaskExample values")
        if not example.example_id or example.example_id in example_ids:
            raise ValueError(
                f"circuit-probe example IDs must be non-empty and unique: {example.example_id!r}"
            )
        example_ids.add(example.example_id)
        pair_group_id = str(example.pair_group_id)
        if not pair_group_id:
            raise ValueError(f"circuit-probe example lacks pair_group_id: {example.example_id}")
        if pair_group_id not in grouped:
            grouped[pair_group_id] = []
            order.append(pair_group_id)
        grouped[pair_group_id].append(example)

    pairs: list[ProbePair] = []
    for pair_group_id in order:
        siblings = grouped[pair_group_id]
        labels = {example.label for example in siblings}
        if (
            len(siblings) != 2
            or labels != {0, 1}
            or any(type(example.label) is not int for example in siblings)
        ):
            raise ValueError(
                f"circuit-probe pair must contain exactly one sibling per signed label: {pair_group_id}"
            )
        if siblings[0].query != siblings[1].query or siblings[0].rules != siblings[1].rules:
            raise ValueError(
                f"circuit-probe siblings do not share one signed graph: {pair_group_id}"
            )
        pairs.append(
            ProbePair(
                subset=subset,
                source_split=source_split,
                pair_group_id=pair_group_id,
                examples=(siblings[0], siblings[1]),
            )
        )
    if not pairs:
        raise ValueError(f"circuit-probe source split is empty: {source_split}")
    return pairs


def family_probe_pairs(
    family: DatasetFamilyHandle,
    *,
    subset: str,
    limit_pairs: int | None = None,
) -> list[ProbePair]:
    """Read a probe split only through a validated seven-split family handle."""

    if not isinstance(family, DatasetFamilyHandle):
        raise TypeError("circuit-probe sources require a DatasetFamilyHandle")
    if subset not in SUBSETS:
        raise ValueError(f"unknown circuit-probe subset {subset!r}")
    pairs = group_complete_pairs(list(family.examples(SOURCE_SPLITS[subset])), subset=subset)
    if limit_pairs is None:
        return pairs
    if type(limit_pairs) is not int or limit_pairs < 1:
        raise ValueError("circuit-probe limit must be a positive pair count")
    if len(pairs) < limit_pairs:
        raise ValueError(
            f"circuit-probe split {subset} has {len(pairs)} pairs; requested {limit_pairs}"
        )
    return pairs[:limit_pairs]


def flatten_pairs(pairs: list[ProbePair]) -> list[TaskExample]:
    return [example for pair in pairs for example in pair.examples]


def ordered_pair_population(pairs: list[ProbePair]) -> list[dict[str, Any]]:
    return [
        {
            "pair_group_id": pair.pair_group_id,
            "example_ids": list(pair.example_ids),
        }
        for pair in pairs
    ]


def _score_status(score: dict[str, Any], *, example_id: str) -> str:
    if not isinstance(score, dict) or set(score) != _SCORE_FIELDS:
        raise ValueError(f"probe score fields changed for {example_id}")
    initial = score["initial_correct"]
    learned = score["learnable_after_post_training"]
    if type(initial) is not bool or type(learned) is not bool:
        raise ValueError(f"probe score flags must be booleans for {example_id}")
    if initial:
        return "base_capable"
    if learned:
        return "challenge"
    return "excluded"


def _decide_pair(pair: ProbePair, scores: dict[str, dict[str, Any]]) -> PairDecision:
    statuses = {
        _score_status(scores[example.example_id], example_id=example.example_id)
        for example in pair.examples
    }
    if len(statuses) != 1:
        raise ValueError(
            "mixed sibling evidence has no registered pair-level decision rule: "
            f"{pair.pair_group_id} -> {sorted(statuses)}"
        )
    status = statuses.pop()
    if status == "excluded":
        return PairDecision(
            pair=pair,
            cohort=None,
            exclusion_reason="initially_unsolved_and_not_learned_by_frozen_calibration",
        )
    return PairDecision(pair=pair, cohort=status, exclusion_reason=None)


def _validate_eligibility_ancestry(rows: Any) -> list[dict[str, str]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("eligibility_evidence_ancestry must bind a calibration run")
    validated: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict) or set(raw) != _ANCESTRY_FIELDS:
            raise ValueError(f"eligibility ancestry row {index} fields changed")
        row = {key: str(raw[key]) for key in _ANCESTRY_FIELDS}
        for key in _ANCESTRY_FIELDS - {"calibration_run_id"}:
            if not _is_sha256(row[key]):
                raise ValueError(f"eligibility ancestry {key} is not a SHA-256")
        if not row["calibration_run_id"].strip():
            raise ValueError("eligibility ancestry calibration_run_id is empty")
        validated.append(row)
    return validated


def _pair_payload(pair: ProbePair) -> dict[str, Any]:
    return {
        "pair_group_id": pair.pair_group_id,
        "examples": serialize_examples(list(pair.examples)),
    }


def build_probe_cohort_manifest(
    family: DatasetFamilyHandle,
    scores: dict[str, dict[str, Any]],
    *,
    initial_student_checkpoint_hash: str,
    scoring_manifest_hash: str,
    eligibility_evidence_ancestry: list[dict[str, str]],
    limit_pairs_per_split: int | None = None,
    git_commit: str = "test-unfrozen",
    prereg_commit: str = "test-unfrozen",
    protocol_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze complete candidate pairs into disjoint pre-training cohorts."""

    if not isinstance(family, DatasetFamilyHandle):
        raise TypeError("probe cohort construction requires a DatasetFamilyHandle")
    if not _is_sha256(initial_student_checkpoint_hash):
        raise ValueError("initial student checkpoint binding must be a SHA-256")
    if not _is_sha256(scoring_manifest_hash):
        raise ValueError("probe scoring manifest binding must be a SHA-256")
    ancestry = _validate_eligibility_ancestry(eligibility_evidence_ancestry)
    pairs_by_subset = {
        subset: family_probe_pairs(
            family,
            subset=subset,
            limit_pairs=limit_pairs_per_split,
        )
        for subset in SUBSETS
    }
    candidate_ids = [
        example.example_id
        for subset in SUBSETS
        for pair in pairs_by_subset[subset]
        for example in pair.examples
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("circuit-probe candidate IDs are not globally unique")
    observed_score_ids = set(scores)
    expected_score_ids = set(candidate_ids)
    if observed_score_ids != expected_score_ids:
        raise ValueError(
            "probe score IDs differ from the exact candidate population: "
            f"missing={sorted(expected_score_ids - observed_score_ids)}, "
            f"extra={sorted(observed_score_ids - expected_score_ids)}"
        )

    decisions = {
        subset: [_decide_pair(pair, scores) for pair in pairs_by_subset[subset]]
        for subset in SUBSETS
    }
    partitions: dict[str, dict[str, list[ProbePair]]] = {
        cohort: {subset: [] for subset in SUBSETS} for cohort in COHORTS
    }
    selection: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        exclusions = []
        for decision in decisions[subset]:
            if decision.selected:
                assert decision.cohort is not None
                partitions[decision.cohort][subset].append(decision.pair)
            else:
                exclusions.append(
                    {
                        "pair_group_id": decision.pair.pair_group_id,
                        "example_ids": list(decision.pair.example_ids),
                        "reason": decision.exclusion_reason,
                    }
                )
        selected_count = sum(len(partitions[cohort][subset]) for cohort in COHORTS)
        selection[subset] = {
            "candidate_pair_count": len(pairs_by_subset[subset]),
            "selected_pair_count": selected_count,
            "excluded_pair_count": len(exclusions),
            "ordered_pair_population": ordered_pair_population(pairs_by_subset[subset]),
            "exclusions": exclusions,
        }

    cohorts = {
        cohort: {
            subset: {
                "pair_count": len(partitions[cohort][subset]),
                "num_examples": 2 * len(partitions[cohort][subset]),
                "pairs": [_pair_payload(pair) for pair in partitions[cohort][subset]],
            }
            for subset in SUBSETS
        }
        for cohort in COHORTS
    }
    for cohort in COHORTS:
        for subset in SUBSETS:
            if cohorts[cohort][subset]["pair_count"] < 1:
                raise ValueError(f"probe cohort {cohort}/{subset} cannot be empty")

    bindings = dict(protocol_bindings or {})
    unknown_bindings = set(bindings) - _PROTOCOL_BINDING_FIELDS
    if unknown_bindings:
        raise ValueError(f"unknown probe protocol bindings: {sorted(unknown_bindings)}")
    if "prereg_commit" in bindings and bindings["prereg_commit"] != prereg_commit:
        raise ValueError("probe score prereg commit differs from cohort construction")
    if "code_commit" in bindings and bindings["code_commit"] != git_commit:
        raise ValueError("probe score code commit differs from cohort construction")
    prereg_version = str(bindings.get("prereg_version", "core_v2"))
    payload: dict[str, Any] = {
        "format_version": COHORT_FORMAT_VERSION,
        "probe_cohort_schema_version": PROBE_COHORT_SCHEMA_VERSION,
        **scientific_compatibility_fields(prereg_version),
        "frozen_before_training": True,
        "construction_phase": "before_confirmatory_training",
        "confirmatory_training_ancestry": [],
        "eligibility_evidence_ancestry": ancestry,
        "initial_student_checkpoint_hash": initial_student_checkpoint_hash,
        "scoring_manifest_hash": scoring_manifest_hash,
        "source_dataset_family_hash": str(family.manifest.get("sha256", "")),
        "source_split_hashes": {
            subset: str(
                family.boundary(SOURCE_SPLITS[subset])["examples_file_sha256"]
            )
            for subset in SUBSETS
        },
        "git_commit": git_commit,
        "prereg_commit": prereg_commit,
        "created_at": utc_now(),
        "selection_rules": {
            "unit": "complete_pair_group",
            "limit_unit": "pair_group",
            "base_capable": "both siblings: initial_correct == true",
            "challenge": (
                "both siblings: initial_correct == false and "
                "learnable_after_post_training == true"
            ),
            "excluded": (
                "both siblings: initial_correct == false and "
                "learnable_after_post_training == false"
            ),
            "mixed_sibling_evidence": "fail_closed",
        },
        "candidate_selection": selection,
        "cohorts": cohorts,
        **bindings,
    }
    required_text = {
        "source_dataset_family_hash": payload["source_dataset_family_hash"],
        "git_commit": git_commit,
        "prereg_commit": prereg_commit,
    }
    if any(not str(value).strip() for value in required_text.values()):
        raise ValueError("probe cohort provenance bindings must be non-empty")
    for name, value in {
        "source_dataset_family_hash": payload["source_dataset_family_hash"],
        **payload["source_split_hashes"],
    }.items():
        if not _is_sha256(value):
            raise ValueError(f"probe cohort {name} must be a SHA-256")
    payload["sha256"] = sha256_value(payload)
    return payload


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError("probe cohort manifest must be a regular non-symlink file")

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"probe cohort manifest contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_from_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"probe cohort manifest contains non-finite value {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("probe cohort manifest is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("probe cohort manifest must be a JSON object")
    return payload


def _validated_pair_payload(
    raw: Any,
    *,
    subset: str,
) -> ProbePair:
    if not isinstance(raw, dict) or set(raw) != {"pair_group_id", "examples"}:
        raise ValueError(f"probe cohort pair fields changed in {subset}")
    raw_examples = raw["examples"]
    if not isinstance(raw_examples, list) or len(raw_examples) != 2:
        raise ValueError(f"probe cohort pair is incomplete in {subset}")
    try:
        examples = [deserialize_example(row) for row in raw_examples]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"probe cohort pair has invalid examples in {subset}") from error
    pair = group_complete_pairs(examples, subset=subset)[0]
    if pair.pair_group_id != raw["pair_group_id"]:
        raise ValueError(f"probe cohort pair_group_id mismatch in {subset}")
    return pair


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    digest = payload.get("sha256")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if digest != sha256_value(content):
        raise ValueError("probe cohort top-level manifest hash mismatch")
    if payload.get("format_version") != COHORT_FORMAT_VERSION:
        raise ValueError("legacy probe cohort formats are not accepted")
    require_scientific_artifact(
        payload,
        expected_prereg_version=str(payload.get("prereg_version", "")),
    )
    if payload.get("probe_cohort_schema_version") != PROBE_COHORT_SCHEMA_VERSION:
        raise ValueError("probe cohort scientific schema changed")
    if (
        payload.get("frozen_before_training") is not True
        or payload.get("construction_phase") != "before_confirmatory_training"
        or payload.get("confirmatory_training_ancestry") != []
    ):
        raise ValueError("probe cohorts are not frozen before confirmatory training")
    _validate_eligibility_ancestry(payload.get("eligibility_evidence_ancestry"))
    for name in (
        "initial_student_checkpoint_hash",
        "scoring_manifest_hash",
        "source_dataset_family_hash",
    ):
        if not _is_sha256(payload.get(name)):
            raise ValueError(f"probe cohort {name} is not a SHA-256")
    source_hashes = payload.get("source_split_hashes")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SUBSETS):
        raise ValueError("probe cohort source split bindings changed")
    if any(not _is_sha256(value) for value in source_hashes.values()):
        raise ValueError("probe cohort source split binding is not a SHA-256")
    if not str(payload.get("git_commit", "")).strip() or not str(
        payload.get("prereg_commit", "")
    ).strip():
        raise ValueError("probe cohorts lack Git/prereg provenance")

    candidate_selection = payload.get("candidate_selection")
    cohorts = payload.get("cohorts")
    if not isinstance(candidate_selection, dict) or set(candidate_selection) != set(SUBSETS):
        raise ValueError("probe cohort candidate selection must cover both subsets")
    if not isinstance(cohorts, dict) or set(cohorts) != set(COHORTS):
        raise ValueError("probe cohort partitions changed")
    global_example_ids: set[str] = set()
    selected_pair_ids: dict[str, set[str]] = {subset: set() for subset in SUBSETS}
    selected_pair_examples: dict[str, dict[str, list[str]]] = {
        subset: {} for subset in SUBSETS
    }
    for cohort in COHORTS:
        raw_subsets = cohorts[cohort]
        if not isinstance(raw_subsets, dict) or set(raw_subsets) != set(SUBSETS):
            raise ValueError(f"probe cohort {cohort} does not cover both subsets")
        for subset in SUBSETS:
            raw_subset = raw_subsets[subset]
            expected_fields = {"pair_count", "num_examples", "pairs"}
            if not isinstance(raw_subset, dict) or set(raw_subset) != expected_fields:
                raise ValueError(f"probe cohort subset fields changed for {cohort}/{subset}")
            raw_pairs = raw_subset["pairs"]
            if not isinstance(raw_pairs, list) or not raw_pairs:
                raise ValueError(f"probe cohort subset is empty for {cohort}/{subset}")
            if raw_subset["pair_count"] != len(raw_pairs) or raw_subset["num_examples"] != 2 * len(
                raw_pairs
            ):
                raise ValueError(f"probe cohort pair/row accounting changed for {cohort}/{subset}")
            for raw_pair in raw_pairs:
                pair = _validated_pair_payload(raw_pair, subset=subset)
                if pair.pair_group_id in selected_pair_ids[subset]:
                    raise ValueError(f"probe pair appears in multiple cohorts: {pair.pair_group_id}")
                selected_pair_ids[subset].add(pair.pair_group_id)
                selected_pair_examples[subset][pair.pair_group_id] = list(pair.example_ids)
                for example_id in pair.example_ids:
                    if example_id in global_example_ids:
                        raise ValueError(f"probe example appears more than once: {example_id}")
                    global_example_ids.add(example_id)

    global_population_pair_ids: set[str] = set()
    global_population_example_ids: set[str] = set()
    for subset in SUBSETS:
        audit = candidate_selection[subset]
        expected_fields = {
            "candidate_pair_count",
            "selected_pair_count",
            "excluded_pair_count",
            "ordered_pair_population",
            "exclusions",
        }
        if not isinstance(audit, dict) or set(audit) != expected_fields:
            raise ValueError(f"probe candidate selection fields changed for {subset}")
        population = audit["ordered_pair_population"]
        exclusions = audit["exclusions"]
        if not isinstance(population, list) or not isinstance(exclusions, list):
            raise ValueError(f"probe candidate selection is malformed for {subset}")
        population_ids: list[str] = []
        population_examples: dict[str, list[str]] = {}
        for row in population:
            if not isinstance(row, dict) or set(row) != {"pair_group_id", "example_ids"}:
                raise ValueError(f"probe ordered pair population fields changed for {subset}")
            pair_group_id = str(row["pair_group_id"])
            example_ids = row["example_ids"]
            if (
                not pair_group_id
                or pair_group_id in population_examples
                or pair_group_id in global_population_pair_ids
                or not isinstance(example_ids, list)
                or len(example_ids) != 2
                or any(not str(value) for value in example_ids)
                or len(set(str(value) for value in example_ids)) != 2
            ):
                raise ValueError(f"probe ordered pair population is invalid for {subset}")
            population_ids.append(pair_group_id)
            population_examples[pair_group_id] = [str(value) for value in example_ids]
            if global_population_example_ids.intersection(population_examples[pair_group_id]):
                raise ValueError(f"probe ordered population reuses an example ID in {subset}")
            global_population_pair_ids.add(pair_group_id)
            global_population_example_ids.update(population_examples[pair_group_id])
        excluded_ids: set[str] = set()
        for row in exclusions:
            if not isinstance(row, dict) or set(row) != {
                "pair_group_id",
                "example_ids",
                "reason",
            }:
                raise ValueError(f"probe exclusion fields changed for {subset}")
            pair_group_id = str(row["pair_group_id"])
            if (
                pair_group_id not in population_examples
                or pair_group_id in excluded_ids
                or [str(value) for value in row["example_ids"]]
                != population_examples[pair_group_id]
                or not str(row["reason"]).strip()
            ):
                raise ValueError(f"probe exclusion is invalid for {subset}: {pair_group_id}")
            excluded_ids.add(pair_group_id)
        for pair_group_id, example_ids in selected_pair_examples[subset].items():
            if population_examples.get(pair_group_id) != example_ids:
                raise ValueError(
                    f"probe selected pair differs from ordered population for {subset}: "
                    f"{pair_group_id}"
                )
        if (
            audit["candidate_pair_count"] != len(population_ids)
            or audit["selected_pair_count"] != len(selected_pair_ids[subset])
            or audit["excluded_pair_count"] != len(excluded_ids)
            or set(population_ids) != selected_pair_ids[subset] | excluded_ids
            or selected_pair_ids[subset] & excluded_ids
        ):
            raise ValueError(f"probe pair-level candidate accounting mismatch for {subset}")
    return payload


def write_probe_cohort_manifest(output: Path, manifest: dict[str, Any]) -> None:
    """Write the single top-level cohort manifest; no row or subset hashes exist."""

    _validate_payload(dict(manifest))
    output.mkdir(parents=True, exist_ok=True)
    extras = sorted(path.name for path in output.iterdir() if path.name != "manifest.json")
    if extras:
        raise ValueError(f"probe cohort output contains undeclared files: {extras}")
    atomic_write_json(output / "manifest.json", manifest)


def validate_probe_cohort_manifest(
    path: Path,
    *,
    expected_initial_checkpoint_hash: str | None = None,
    expected_manifest_hash: str | None = None,
) -> dict[str, Any]:
    payload = _validate_payload(_strict_json(path))
    if expected_manifest_hash is not None and payload["sha256"] != expected_manifest_hash:
        raise ValueError("probe cohort manifest differs from the expected frozen identity")
    if (
        expected_initial_checkpoint_hash is not None
        and payload["initial_student_checkpoint_hash"] != expected_initial_checkpoint_hash
    ):
        raise ValueError("probe cohort initial checkpoint hash mismatch")
    if str(payload.get("protocol_track", "")).startswith("qwen3_"):
        namespace = str(payload["protocol_track"]).replace("_", "-", 1)
        expected = {
            "artifact_namespace": namespace,
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
        }
        mismatches = {
            key: {"expected": value, "observed": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        for key in ("chat_template_sha256", "tokenizer_fingerprint"):
            if not _is_sha256(payload.get(key)):
                mismatches[key] = {
                    "expected": "64-hex binding",
                    "observed": payload.get(key),
                }
        if mismatches:
            raise ValueError(f"Qwen3 probe cohort protocol binding mismatch: {mismatches}")
    return payload


def load_probe_examples(
    path: Path,
    *,
    cohort: str,
    subset: str,
    expected_initial_checkpoint_hash: str | None = None,
    expected_manifest_hash: str | None = None,
) -> tuple[list[TaskExample], dict[str, Any]]:
    if cohort not in COHORTS or subset not in SUBSETS:
        raise ValueError(f"invalid probe selection {cohort}/{subset}")
    payload = validate_probe_cohort_manifest(
        path,
        expected_initial_checkpoint_hash=expected_initial_checkpoint_hash,
        expected_manifest_hash=expected_manifest_hash,
    )
    pairs = [
        _validated_pair_payload(raw, subset=subset)
        for raw in payload["cohorts"][cohort][subset]["pairs"]
    ]
    return flatten_pairs(pairs), payload
