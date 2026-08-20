"""Formal noise-corrected dynamics over checkpoint-bound circuit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.dynamics import circuit_stability_report
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import formal_artifact_binding
from posttrain_circuits.core.readiness import require_formal_prerequisite_binding
from posttrain_circuits.core.scientific_versions import scientific_compatibility_fields


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"artifact is not a mapping: {path}")
    return payload


def _mask_transfer(payload: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    transfer = payload.get("cross_checkpoint_mask_transfer")
    if not isinstance(transfer, dict | list) or not transfer:
        raise ValueError("transfer artifact lacks cross_checkpoint_mask_transfer evidence")
    return transfer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze checkpoint-specific circuit dynamics")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--circuits", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
    parser.add_argument("--transfers", nargs="+", type=Path, required=True)
    parser.add_argument("--activation-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    qwen3 = str(config.get("protocol_track", "")).startswith("qwen3_")
    formal = formal_artifact_binding(config) if qwen3 else {}
    if len(args.circuits) < 2:
        raise ValueError("formal dynamics requires at least two checkpoint circuit artifacts")
    if len(args.evaluations) != len(args.circuits) or len(args.transfers) != len(args.circuits) - 1:
        raise ValueError("dynamics artifacts must align: N circuits/evaluations and N-1 transfers")
    circuits = [_read(path) for path in args.circuits]
    evaluations = [_read(path) for path in args.evaluations]
    transfer_artifacts = [_read(path) for path in args.transfers]
    transfers = [_mask_transfer(artifact) for artifact in transfer_artifacts]
    if qwen3:
        for name, artifact in (
            *(("circuit", row) for row in circuits),
            *(("exact-patching evaluation", row) for row in evaluations),
            *(("mask-transfer evaluation", row) for row in transfer_artifacts),
        ):
            require_formal_prerequisite_binding(artifact, formal, name=name)
    cohort_keys = {(row.get("probe_cohort_manifest_hash"), row.get("probe_cohort")) for row in circuits}
    stage_keys = {row.get("probe_stage") for row in circuits}
    conventions = {(row.get("graph_convention"), row.get("node_or_edge_level")) for row in circuits}
    component_sets = {tuple(sorted(row.get("scores", {}))) for row in circuits}
    if len(cohort_keys) != 1 or len(stage_keys) != 1 or len(conventions) != 1 or len(component_sets) != 1:
        raise ValueError(
            "circuit artifacts differ in probe cohort, stage, graph convention, or component naming"
        )
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
    cohort_manifest_hash, cohort = next(iter(cohort_keys))
    probe_stage = next(iter(stage_keys))
    for circuit, evaluation in zip(circuits, evaluations, strict=True):
        required = {
            "checkpoint_sha256": circuit.get("checkpoint_sha256"),
            "probe_cohort": cohort,
            "probe_cohort_manifest_hash": cohort_manifest_hash,
            "probe_stage": probe_stage,
        }
        mismatches = {
            key: {"expected": value, "observed": evaluation.get(key)}
            for key, value in required.items()
            if evaluation.get(key) != value
        }
        if mismatches:
            raise ValueError(f"exact-patching evaluation does not match its circuit: {mismatches}")
    for index, (artifact, transfer) in enumerate(zip(transfer_artifacts, transfers, strict=True)):
        source_hash = checkpoint_hashes[index]
        target_hash = checkpoint_hashes[index + 1]
        if artifact.get("checkpoint_sha256") != target_hash:
            raise ValueError("mask-transfer evaluation is not bound to the target checkpoint")
        rows = transfer if isinstance(transfer, list) else [transfer]
        if any(
            not isinstance(row, dict)
            or row.get("source_checkpoint") != source_hash
            or row.get("target_checkpoint") != target_hash
            for row in rows
        ):
            raise ValueError("mask-transfer evidence does not bind the requested checkpoint transition")
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
    payload: dict[str, Any] = {
        "format_version": 2,
        **scientific_compatibility_fields(str(config["prereg_version"])),
        **formal,
        "analysis_role": "primary_noise_corrected_circuit_dynamics",
        "thresholded_jaccard_role": "diagnostic_only",
        "probe_cohort": cohort,
        "probe_cohort_manifest_hash": cohort_manifest_hash,
        "probe_stage": probe_stage,
        "input_artifact_hashes": {
            str(path): sha256_file(path) for path in (*args.circuits, *args.evaluations, *args.transfers)
        },
        "transitions": transitions,
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
