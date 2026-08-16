"""Finalize the single-seed Qwen pilot without direction-dependent hypothesis gates."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from posttrain_circuits.analysis.factorial import match_validation_accuracy
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.readiness import validate_readiness_report

CELLS = (
    "offline_hard",
    "online_hard",
    "offline_soft",
    "online_soft_opd",
    "offline_verified_replay",
    "online_verified_replay",
    "canonical_sft",
    "canonical_grpo",
)
MATCHED_PAIRS = (
    ("offline_soft", "online_soft_opd"),
    ("online_hard", "online_soft_opd"),
    ("online_verified_replay", "canonical_grpo"),
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"pilot artifact is not a mapping: {path}")
    return payload


def _hash_valid(payload: dict[str, Any]) -> bool:
    copy = dict(payload)
    expected = copy.pop("sha256", None)
    return expected == sha256_value(copy)


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite(child) for child in value)
    return True


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _curve(rows: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float]]:
    selected = [(float(row["step"]), float(row[metric])) for row in rows if row.get(metric) is not None]
    return [row[0] for row in selected], [row[1] for row in selected]


def _matched_summary(
    rows_by_cell: dict[str, list[dict[str, Any]]],
    validation_hash: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for left, right in MATCHED_PAIRS:
        left_steps, left_values = _curve(rows_by_cell[left], "validation_accuracy")
        right_steps, right_values = _curve(rows_by_cell[right], "validation_accuracy")
        if len(left_values) < 2 or len(right_values) < 2:
            results[f"{left}__{right}"] = {
                "valid": False,
                "reason": "both cells require at least two observed formal-validation checkpoints",
            }
            continue
        lower = max(min(left_values), min(right_values))
        upper = min(max(left_values), max(right_values))
        if lower > upper:
            results[f"{left}__{right}"] = {
                "valid": False,
                "reason": "observed validation-accuracy ranges do not overlap; extrapolation forbidden",
            }
            continue
        target = (lower + upper) / 2.0
        results[f"{left}__{right}"] = {
            "valid": True,
            "target_validation_accuracy": target,
            left: match_validation_accuracy(
                left_steps,
                left_values,
                target,
                validation_artifact_hash=validation_hash,
            ).__dict__,
            right: match_validation_accuracy(
                right_steps,
                right_values,
                target,
                validation_artifact_hash=validation_hash,
            ).__dict__,
        }
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Finalize the gated seed-42 Qwen pilot")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--job-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    g0 = _read(args.g0)
    bank = _read(args.bank_manifest)
    probes = _read(args.probe_manifest)
    validation = _read(args.validation_manifest)
    manifests: dict[str, dict[str, Any]] = {}
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell in CELLS:
        root = args.run_dir / "runs" / cell / "seed-42"
        manifests[cell] = _read(root / "manifest.json")
        rows_by_cell[cell] = _rows(root / "metrics.jsonl")
    local_fork = _read(args.run_dir / "local_fork" / "results.json")
    resume = _read(args.run_dir / "distributed_resume.json")
    dynamics_paths = sorted((args.run_dir / "dynamics").glob("*/*.json"))
    dynamics = [_read(path) for path in dynamics_paths]
    exact_paths = sorted((args.run_dir / "circuits" / "final").glob("*/*/exact_patching.json"))
    exact = [_read(path) for path in exact_paths]
    expected_initial_hash = str(probes.get("initial_student_checkpoint_hash", ""))
    readiness_path = args.g0.parent / "readiness" / "readiness.json"
    readiness = validate_readiness_report(
        readiness_path,
        expected_initial_checkpoint_hash=expected_initial_hash,
    )
    offline_hashes = {
        manifests[cell].get("rollout_bank_hash")
        for cell in ("offline_hard", "offline_soft", "offline_verified_replay")
    }
    base_accuracy = float(g0.get("metrics", {}).get("base_accuracy", 0.0))
    anchor_accuracies = {}
    for cell in ("canonical_sft", "canonical_grpo"):
        values = [
            float(row["validation_accuracy"])
            for row in rows_by_cell[cell]
            if row.get("validation_accuracy") is not None
        ]
        anchor_accuracies[cell] = values[-1] if values else float("nan")
    transition_rows = [transition for artifact in dynamics for transition in artifact.get("transitions", [])]
    random_margins = [
        float(row["selected_vs_matched_random_cpr_margin"])
        for row in exact
        if row.get("selected_vs_matched_random_cpr_margin") is not None
    ]
    validation_hash = str(validation.get("sha256", ""))
    matched = _matched_summary(rows_by_cell, validation_hash)
    checks = {
        "g0_passed_and_hash_valid": g0.get("passed") is True and _hash_valid(g0),
        "g0_git_binding": g0.get("git_commit")
        == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "full_readiness_report": readiness.get("ready") is True,
        "all_training_cells_present": len(manifests) == len(CELLS),
        "all_artifacts_finite": all(_finite(rows) for rows in rows_by_cell.values())
        and all(_finite(row) for row in dynamics)
        and all(_finite(row) for row in exact),
        "single_seed_42": all(int(row.get("seed", -1)) == 42 for row in manifests.values()),
        "offline_bank_shared": len(offline_hashes) == 1
        and next(iter(offline_hashes), None) == bank.get("sha256"),
        "bank_reward_mixture": 0
        < int(bank.get("reward_distribution", {}).get("positive", 0))
        < int(bank.get("total_trajectories", 0)),
        "teacher_mass": bool(g0.get("checks", {}).get("teacher_topk_mass")),
        "anchor_improves_accuracy": any(
            math.isfinite(value) and value > base_accuracy for value in anchor_accuracies.values()
        ),
        "local_fork_output_kl_matched": local_fork.get("valid_for_primary_analysis") is True,
        "distributed_resume": resume.get("passed") is True and int(resume.get("world_size", 0)) == 4,
        "circuit_artifact_count": len(dynamics) == 12 and len(exact) == 12,
        "circuit_exceeds_noise": bool(transition_rows)
        and any(float(row.get("excess_churn", float("-inf"))) > 0 for row in transition_rows),
        "circuit_beats_random": bool(random_margins) and any(value > 0 for value in random_margins),
        "mask_transfer_present": bool(transition_rows)
        and all(bool(row.get("cross_checkpoint_mask_transfer")) for row in transition_rows),
        "probe_cohorts_frozen": probes.get("frozen_before_training") is True,
        "formal_validation_bound": bool(validation_hash)
        and all(
            row.get("dataset_hashes", {}).get("validation_manifest") == validation_hash
            for row in manifests.values()
        ),
        "exact_initial_checkpoint_bound": len(expected_initial_hash) == 64
        and all(
            row.get("dataset_hashes", {}).get("initial_checkpoint") == expected_initial_hash
            for row in manifests.values()
        ),
        "matched_accuracy_summaries_complete": len(matched) == len(MATCHED_PAIRS)
        and all(bool(summary.get("valid")) or bool(summary.get("reason")) for summary in matched.values()),
    }
    output_kl = {
        cell: _curve(rows, "output_kl_from_initial")
        for cell, rows in rows_by_cell.items()
        if _curve(rows, "output_kl_from_initial")[1]
    }
    job_ids = [line.strip() for line in args.job_ids.read_text(encoding="utf-8").splitlines() if line]
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload: dict[str, Any] = {
        "phase": "single_seed_qwen_core_pilot",
        "passed": all(checks.values()),
        "readiness": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "seed": 42,
        "full_three_seed_factorial_launched": False,
        "gemma_replication_launched": False,
        "base_accuracy": base_accuracy,
        "anchor_accuracies": anchor_accuracies,
        "matched_validation_accuracy": matched,
        "output_kl_curves": output_kl,
        "job_ids": job_ids,
        "git_commit": git_commit,
        "prereg_commit": g0.get("prereg_commit"),
        "artifact_hashes": {
            str(path): sha256_file(path)
            for path in (
                args.g0,
                args.bank_manifest,
                args.probe_manifest,
                args.validation_manifest,
                readiness_path,
                args.run_dir / "local_fork" / "results.json",
                args.run_dir / "distributed_resume.json",
                *dynamics_paths,
                *exact_paths,
            )
        },
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)
    args.output.with_suffix(".md").write_text(
        "# Qwen seed-42 pilot report\n\n"
        f"Readiness: **{payload['readiness']}**\n\n"
        + "\n".join(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
        + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise SystemExit("pilot completed with readiness NO-GO")


if __name__ == "__main__":
    main()
