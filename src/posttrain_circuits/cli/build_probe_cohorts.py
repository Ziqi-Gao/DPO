"""Freeze base-capable/challenge circuit probe manifests before training."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.probe_cohorts import (
    build_probe_cohort_manifest,
    write_probe_cohort_manifest,
)
from posttrain_circuits.cli._common import print_json
from posttrain_circuits.core.hashing import sha256_file


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
    score_rows = scores_payload.get("scores", scores_payload)
    if isinstance(score_rows, list):
        scores = {str(row["example_id"]): row for row in score_rows}
    elif isinstance(score_rows, dict):
        scores = {str(key): dict(value) for key, value in score_rows.items()}
    else:
        raise TypeError("probe scores must be a mapping or a list of score rows")
    manifest = build_probe_cohort_manifest(
        split_rows,
        scores,
        source_split_hashes=source_hashes,
        initial_student_checkpoint_hash=args.initial_checkpoint_hash,
        scoring_manifest_hash=sha256_file(args.scores),
        learnability_evidence_hash=args.learnability_evidence_hash,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        prereg_commit=subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--", "prereg/core_v1.yaml"],
            text=True,
        ).strip(),
        source_artifacts={
            subset: {
                "path": str((args.splits_root / source).resolve()),
                "manifest_hash": source_hashes[subset],
            }
            for subset, source in source_names.items()
        },
    )
    write_probe_cohort_manifest(args.output, manifest)
    print_json({"output": str(args.output), "manifest": manifest})


if __name__ == "__main__":
    main()
