"""Verify the complete CPU-only scientific-repair vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import require_core_v2_artifact
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.datasets.proofgraph.splits import assert_split_isolation
from posttrain_circuits.experiments.protocols.local_fork import LOCAL_FORK_SPEC
from posttrain_circuits.methods.registry import FACTORIAL_METHOD_IDS, get_method_spec
from posttrain_circuits.methods.rl import CANONICAL_GRPO_METHOD, TRL_GRPO_BACKEND
from posttrain_circuits.methods.sft import CANONICAL_SFT_METHOD
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.leakage import validate_label_leakage_artifact

OFFLINE_FACTORIAL_METHOD_IDS = tuple(
    method_id
    for method_id in FACTORIAL_METHOD_IDS
    if get_method_spec(method_id).requires_common_rollout_bank
)
PROBE_STAGES = ("first_rule_selection", "intermediate_conclusion", "final_answer")
PREREG_PATH = Path("prereg/core_v2.yaml")
PREREG_VERSION = "core_v2"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON mapping: {path}")
    return payload


def _require_hash(payload: dict[str, Any], *, name: str) -> None:
    observed = payload.get("sha256")
    expected = sha256_value({key: value for key, value in payload.items() if key != "sha256"})
    if observed != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected={expected}, observed={observed}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify repaired CPU G0 smoke artifacts")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    task = ProofGraphTask()
    family = load_dataset_family(args.root / "dataset")
    splits = family.splits
    proof_counts = {0: 0, 1: 0}
    for split in (
        "train",
        "validation",
        "iid_test",
        "ood_depth_test",
        "ood_structure_test",
        "circuit_discovery",
        "circuit_validation",
    ):
        examples = splits[split]
        for example in examples:
            result = task.verify(example, task.parse_response(task.canonical_target(example)))
            if result.reward != 1.0 or not example.canonical_proof:
                raise RuntimeError(f"invalid signed proof in CPU fixture: {example.example_id}")
            proof_counts[example.label] += 1
    assert_split_isolation(splits)
    if min(proof_counts.values()) < 1:
        raise RuntimeError("CPU fixture did not validate proofs for both labels")

    leakage = validate_label_leakage_artifact(_read(args.root / "label_leakage.json"))
    if leakage.get("passed") is not True:
        raise RuntimeError("CPU label-leakage audit failed")

    bank = TrajectoryStore(args.root / "common_bank").check_integrity()
    manifests = {
        method_id: _read(args.root / "factorial" / method_id / "manifest.json")
        for method_id in FACTORIAL_METHOD_IDS
    }
    for cell, manifest in manifests.items():
        if manifest.get("prereg_version") != PREREG_VERSION:
            raise RuntimeError(f"{cell} did not bind active preregistration")
    offline_hashes = {
        str(manifests[method_id].get("rollout_bank_hash"))
        for method_id in OFFLINE_FACTORIAL_METHOD_IDS
    }
    if offline_hashes != {str(bank["sha256"])}:
        raise RuntimeError(
            "offline factorial cells did not share the exact frozen rollout bank: "
            f"observed={sorted(offline_hashes)}, expected={bank['sha256']}"
        )

    sft_manifest = _read(args.root / "sft" / "run" / "manifest.json")
    if sft_manifest.get("experiment_cell") != CANONICAL_SFT_METHOD.method_id:
        raise RuntimeError("canonical SFT anchor is missing")

    grpo = _read(args.root / "grpo" / "grpo_update_evidence.json")
    require_core_v2_artifact(grpo, require_hash=True)
    if (
        grpo.get("backend") != TRL_GRPO_BACKEND
        or grpo.get("method_spec_sha256") != CANONICAL_GRPO_METHOD.scientific_sha256
        or grpo.get("parameters_changed") is not True
    ):
        raise RuntimeError("official tiny TRL GRPO did not produce a parameter update")

    fork = _read(args.root / "local_fork" / "results.json")
    require_core_v2_artifact(fork, require_circuit_schema=True, require_hash=True)
    branches = {str(row["branch"]) for row in fork.get("results", [])}
    expected_branches = set(LOCAL_FORK_SPEC.branch_ids)
    if branches != expected_branches or fork.get("valid_for_primary_analysis") is not True:
        raise RuntimeError("grouped local-fork branches are missing or output-KL unmatched")

    stage_evidence = {}
    for stage in PROBE_STAGES:
        root = args.root / "circuits" / stage
        circuit = _read(root / "circuit.json")
        exact = _read(root / "exact_patching.json")
        require_core_v2_artifact(circuit, require_circuit_schema=True, require_hash=True)
        require_core_v2_artifact(exact, require_circuit_schema=True, require_hash=True)
        if circuit.get("probe_stage") != stage or exact.get("probe_stage") != stage:
            raise RuntimeError(f"stage-specific circuit binding failed for {stage}")
        if circuit.get("tokenized_probe_manifest_hash") != exact.get("tokenized_probe_manifest_hash"):
            raise RuntimeError(f"discovery/exact manifest mismatch for {stage}")
        if not exact.get("curves"):
            raise RuntimeError(f"missing stage-specific faithfulness curve for {stage}")
        stage_evidence[stage] = {
            "circuit_sha256": circuit["sha256"],
            "exact_sha256": exact["sha256"],
            "target_manifest_hash": circuit["stage_target_manifest_hash"],
            "attribution_patching_spearman": exact["attribution_patching_spearman"],
        }

    if PREREG_VERSION != "core_v2" or Path("prereg/core_v2.yaml") != PREREG_PATH:
        raise RuntimeError("core_v2 is not the active preregistration")
    prereg_sha256 = sha256_file(PREREG_PATH)
    if any(manifest.get("prereg_sha256") != prereg_sha256 for manifest in manifests.values()):
        raise RuntimeError("factorial manifests do not record the frozen core_v2 SHA-256")

    report: dict[str, Any] = {
        "format_version": 2,
        "status": "passed_cpu_smoke_only",
        "real_g0_claimed": False,
        "proof_counts": {str(key): value for key, value in proof_counts.items()},
        "split_isolation": "passed",
        "label_leakage_metrics": leakage["metrics"],
        "common_rollout_bank_sha256": bank["sha256"],
        "factorial_cells": list(FACTORIAL_METHOD_IDS),
        "offline_shared_bank_verified": True,
        "canonical_sft": "passed",
        "official_trl_grpo": {
            "optimizer_steps": grpo["optimizer_steps"],
            "parameter_update_norm": grpo["parameter_update_norm"],
            "evidence_sha256": grpo["sha256"],
        },
        "local_fork_branches": sorted(branches),
        "circuit_stages": stage_evidence,
        "active_preregistration": str(PREREG_PATH),
        "active_preregistration_sha256": prereg_sha256,
    }
    report["sha256"] = sha256_value(report)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
