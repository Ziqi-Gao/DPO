"""Freeze base-capable/challenge circuit probe manifests before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.runs import require_git_output
from posttrain_circuits.datasets.circuit_probes.cohorts import (
    build_probe_cohort_manifest,
    family_probe_pairs,
    ordered_pair_population,
    write_probe_cohort_manifest,
)
from posttrain_circuits.cli._common import print_json
from posttrain_circuits.datasets.circuit_probes.contracts import SOURCE_SPLITS, SUBSETS
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze hash-pinned circuit probe cohorts")
    parser.add_argument("--dataset-family", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        required=True,
        help="Maximum ordered candidate pair groups per circuit split.",
    )
    args = parser.parse_args(argv)
    family = load_dataset_family(args.dataset_family)
    scores_payload = json.loads(args.scores.read_text(encoding="utf-8"))
    score_digest = scores_payload.get("sha256")
    score_content = {key: value for key, value in scores_payload.items() if key != "sha256"}
    if score_digest != sha256_value(score_content):
        raise ValueError("probe score artifact hash mismatch")
    score_rows = scores_payload.get("scores")
    if not isinstance(score_rows, dict):
        raise TypeError("probe score artifact requires an exact score mapping")
    scores = {str(key): dict(value) for key, value in score_rows.items()}
    expected_source_hashes = {
        subset: str(
            family.boundary(SOURCE_SPLITS[subset])["examples_file_sha256"]
        )
        for subset in SUBSETS
    }
    if scores_payload.get("source_dataset_family_hash") != family.manifest["sha256"]:
        raise ValueError("probe scores bind a different dataset family")
    if scores_payload.get("source_split_hashes") != expected_source_hashes:
        raise ValueError("probe scores bind different circuit source splits")
    expected_population = {
        subset: ordered_pair_population(
            family_probe_pairs(
                family,
                subset=subset,
                limit_pairs=args.limit_per_split,
            )
        )
        for subset in SUBSETS
    }
    if scores_payload.get("ordered_candidate_pairs") != expected_population:
        raise ValueError("probe score candidate pair order differs from the frozen family")
    prereg_path = str(scores_payload.get("prereg_path", ""))
    if not prereg_path:
        raise ValueError("probe scores do not bind a preregistration path")
    manifest = build_probe_cohort_manifest(
        family,
        scores,
        initial_student_checkpoint_hash=str(scores_payload.get("initial_checkpoint_sha256", "")),
        scoring_manifest_hash=str(score_digest),
        eligibility_evidence_ancestry=scores_payload.get(
            "eligibility_evidence_ancestry", []
        ),
        limit_pairs_per_split=args.limit_per_split,
        git_commit=require_git_output(["rev-parse", "HEAD"]),
        prereg_commit=require_git_output(["log", "-n", "1", "--format=%H", "--", prereg_path]),
        protocol_bindings={
            key: scores_payload[key]
            for key in (
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
            )
            if key in scores_payload
        },
    )
    write_probe_cohort_manifest(args.output, manifest)
    print_json({"output": str(args.output), "manifest": manifest})


if __name__ == "__main__":
    main()
