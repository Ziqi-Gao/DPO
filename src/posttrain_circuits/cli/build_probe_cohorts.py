"""Freeze base-capable/challenge circuit probe manifests before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.probe_cohorts import (
    build_probe_cohort_manifest,
    write_probe_cohort_manifest,
)
from posttrain_circuits.cli._common import print_json
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.provenance import require_git_output


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _eligible_candidates(
    rows: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = []
    excluded = []
    for row in rows:
        example_id = str(row.get("example_id", ""))
        score = scores.get(example_id)
        if score is None:
            raise ValueError(f"probe candidate {example_id!r} has no score")
        initial = bool(score.get("initial_correct"))
        learned = bool(score.get("learnable_after_post_training"))
        if initial or learned:
            selected.append(row)
        else:
            excluded.append(
                {
                    "example_id": example_id,
                    "reason": "initially_unsolved_and_not_learned_by_frozen_calibration",
                }
            )
    audit: dict[str, Any] = {
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "excluded_count": len(excluded),
        "excluded": excluded,
    }
    audit["sha256"] = sha256_value(audit)
    return selected, audit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze hash-pinned circuit probe cohorts")
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--initial-checkpoint-hash", required=True)
    parser.add_argument("--learnability-evidence-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args(argv)
    source_names = {
        "discovery": "circuit_discovery",
        "validation": "circuit_validation",
    }
    split_rows = {
        subset: _jsonl(args.splits_root / source / "examples.jsonl")[: args.limit_per_split]
        if args.limit_per_split is not None
        else _jsonl(args.splits_root / source / "examples.jsonl")
        for subset, source in source_names.items()
    }
    source_hashes = {
        subset: str(
            json.loads((args.splits_root / source / "manifest.json").read_text(encoding="utf-8"))["sha256"]
        )
        for subset, source in source_names.items()
    }
    scores_payload = json.loads(args.scores.read_text(encoding="utf-8"))
    score_digest = scores_payload.get("sha256")
    score_content = {key: value for key, value in scores_payload.items() if key != "sha256"}
    if score_digest != sha256_value(score_content):
        raise ValueError("probe score artifact hash mismatch")
    score_rows = scores_payload.get("scores", scores_payload)
    if isinstance(score_rows, list):
        scores = {str(row["example_id"]): row for row in score_rows}
    elif isinstance(score_rows, dict):
        scores = {str(key): dict(value) for key, value in score_rows.items()}
    else:
        raise TypeError("probe scores must be a mapping or a list of score rows")
    candidate_audit = {}
    for subset, rows in split_rows.items():
        split_rows[subset], candidate_audit[subset] = _eligible_candidates(rows, scores)
    protocol_track = str(scores_payload.get("protocol_track", "core_v2"))
    prereg_path = "prereg/qwen3_v1.yaml" if protocol_track == "qwen3_v1" else "prereg/core_v2.yaml"
    manifest = build_probe_cohort_manifest(
        split_rows,
        scores,
        source_split_hashes=source_hashes,
        initial_student_checkpoint_hash=args.initial_checkpoint_hash,
        scoring_manifest_hash=sha256_file(args.scores),
        learnability_evidence_hash=args.learnability_evidence_hash,
        git_commit=require_git_output(["rev-parse", "HEAD"]),
        prereg_commit=require_git_output(["log", "-n", "1", "--format=%H", "--", prereg_path]),
        source_artifacts={
            subset: {
                "path": str((args.splits_root / source).resolve()),
                "manifest_hash": source_hashes[subset],
            }
            for subset, source in source_names.items()
        },
        candidate_selection_audit=candidate_audit,
        protocol_bindings={
            key: scores_payload[key]
            for key in (
                "protocol_track",
                "artifact_namespace",
                "prompt_protocol",
                "enable_thinking",
                "chat_template_sha256",
                "tokenizer_fingerprint",
            )
            if key in scores_payload
        },
    )
    write_probe_cohort_manifest(args.output, manifest)
    print_json({"output": str(args.output), "manifest": manifest})


if __name__ == "__main__":
    main()
