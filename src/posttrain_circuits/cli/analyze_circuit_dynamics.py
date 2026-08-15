"""Formal noise-corrected dynamics over checkpoint-bound circuit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.dynamics import circuit_stability_report
from posttrain_circuits.core.hashing import sha256_file
from posttrain_circuits.core.manifests import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"artifact is not a mapping: {path}")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze checkpoint-specific circuit dynamics")
    parser.add_argument("--circuits", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
    parser.add_argument("--transfers", nargs="+", type=Path, required=True)
    parser.add_argument("--activation-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.circuits) < 2:
        raise ValueError("formal dynamics requires at least two checkpoint circuit artifacts")
    if len(args.evaluations) != len(args.circuits) or len(args.transfers) != len(args.circuits) - 1:
        raise ValueError("dynamics artifacts must align: N circuits/evaluations and N-1 transfers")
    circuits = [_read(path) for path in args.circuits]
    evaluations = [_read(path) for path in args.evaluations]
    transfers = [_read(path) for path in args.transfers]
    cohort_keys = {(row.get("probe_cohort_manifest_hash"), row.get("probe_cohort")) for row in circuits}
    conventions = {(row.get("graph_convention"), row.get("node_or_edge_level")) for row in circuits}
    component_sets = {tuple(sorted(row.get("scores", {}))) for row in circuits}
    if len(cohort_keys) != 1 or len(conventions) != 1 or len(component_sets) != 1:
        raise ValueError("circuit artifacts differ in probe cohort, graph convention, or component naming")
    checkpoint_hashes = []
    for row in circuits:
        checkpoint_hash = str(row.get("checkpoint_sha256", ""))
        checkpoint_path = Path(str(row.get("checkpoint_path", "")))
        if len(checkpoint_hash) != 64 or not checkpoint_path.is_file():
            raise ValueError("circuit artifact lacks a valid checkpoint path/hash")
        if sha256_file(checkpoint_path) != checkpoint_hash:
            raise ValueError("checkpoint bytes no longer match circuit artifact")
        vectors = row.get("bootstrap_score_vectors", [])
        indices = row.get("bootstrap_resample_indices", [])
        raw_hashes = row.get("bootstrap_raw_graph_hashes", [])
        if not vectors or not (len(vectors) == len(indices) == len(raw_hashes)):
            raise ValueError("circuit artifact lacks complete bootstrap vectors/indices/raw graph hashes")
        checkpoint_hashes.append(checkpoint_hash)
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("dynamics requires distinct checkpoint byte hashes")
    transitions = []
    for index in range(len(circuits) - 1):
        heldout = evaluations[index + 1].get("heldout_exact_patching_effects", evaluations[index + 1])
        report = circuit_stability_report(
            source_scores={str(k): float(v) for k, v in circuits[index]["scores"].items()},
            target_scores={str(k): float(v) for k, v in circuits[index + 1]["scores"].items()},
            source_bootstrap_score_vectors=circuits[index]["bootstrap_score_vectors"],
            target_bootstrap_score_vectors=circuits[index + 1]["bootstrap_score_vectors"],
            activation_threshold=args.activation_threshold,
            cross_checkpoint_mask_transfer=transfers[index],
            heldout_exact_patching_effects=heldout,
        )
        transitions.append(
            {
                "source_checkpoint_sha256": checkpoint_hashes[index],
                "target_checkpoint_sha256": checkpoint_hashes[index + 1],
                **report,
            }
        )
    atomic_write_json(
        args.output,
        {
            "analysis_role": "primary_noise_corrected_circuit_dynamics",
            "thresholded_jaccard_role": "diagnostic_only",
            "probe_cohort": next(iter(cohort_keys)),
            "transitions": transitions,
        },
    )


if __name__ == "__main__":
    main()
