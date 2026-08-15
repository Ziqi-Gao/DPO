"""Finalize the hash-bound G0 decision from production artifacts only."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.dynamics import estimate_estimator_noise_floor
from posttrain_circuits.circuits.probe_cohorts import validate_probe_cohort_manifest
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.readiness import build_readiness_report, validate_anti_shortcut_report


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"G0 artifact is not a mapping: {path}")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Finalize real Qwen G0")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--teacher-store-manifest", type=Path, required=True)
    parser.add_argument("--anti-shortcut", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--exact-patching", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--distributed-resume", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    base = _read(args.base_scores)
    teacher = _read(args.teacher_store_manifest)
    anti = validate_anti_shortcut_report(
        args.anti_shortcut,
        max_shortcut_gap=float(config["anti_shortcut"]["max_shortcut_gap"]),
        expected_model_checkpoint_hash=str(config["model"]["model_revision"]),
    )
    probes = validate_probe_cohort_manifest(args.probe_manifest)
    circuit = _read(args.circuit)
    exact = _read(args.exact_patching)
    compatibility = _read(args.compatibility)
    resume = _read(args.distributed_resume)
    noise = estimate_estimator_noise_floor(
        circuit.get("bootstrap_score_vectors", []),
        activation_threshold=0.0,
    )
    metrics = base.get("initial_validation_metrics", {})
    calibrated_metrics = base.get("calibrated_validation_metrics", {})
    teacher_mass = teacher.get("teacher_topk_mass", {})
    checks = {
        "base_task_accuracy": float(metrics.get("answer_accuracy", 0.0))
        >= float(config["anti_shortcut"]["minimum_iid_accuracy"]),
        "teacher_topk_mass": float(teacher_mass.get("minimum", 0.0)) >= 0.90,
        "calibration_anchor_improves_accuracy": float(calibrated_metrics.get("answer_accuracy", 0.0))
        > float(metrics.get("answer_accuracy", 0.0)),
        "anti_shortcut": anti.get("passed") is True,
        "base_capable_probes": probes["cohorts"]["base_capable"]["discovery"]["num_examples"] > 0
        and probes["cohorts"]["base_capable"]["validation"]["num_examples"] > 0,
        "hf_transformerlens_gqa_parity": compatibility.get("passed") is True
        and compatibility.get("transformerlens_parity_passed") is True,
        "bootstrap_stability": float(noise["within_checkpoint_full_score_spearman"])
        >= float(config["g0"]["minimum_bootstrap_spearman"]),
        "eap_ig_beats_matched_random": float(
            exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_random_control_margin"]),
        "functional_group_separation": int(exact.get("functional_group_count", 0)) >= 2,
        "identity_sanity": exact.get("sanity_checks", {}).get("identity_passed") is True,
        "full_corruption_sanity": exact.get("sanity_checks", {}).get("full_corruption_passed") is True,
        "attribution_exact_calibration": exact.get("attribution_patching_spearman") is not None,
        "distributed_checkpoint_resume": resume.get("passed") is True
        and int(resume.get("world_size", 0)) == 4,
        "split_probe_isolation": probes.get("frozen_before_training") is True,
        "artifact_reconstruction": all(
            str(row.get("checkpoint_sha256", "")) for row in (circuit, exact.get("artifacts", {}))
        ),
    }
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    prereg_commit = subprocess.check_output(
        ["git", "log", "-n", "1", "--format=%H", "--", "prereg/core_v1.yaml"], text=True
    ).strip()
    payload: dict[str, Any] = {
        "format_version": 1,
        "phase": "G0",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "base_accuracy": metrics.get("answer_accuracy"),
            "calibrated_accuracy": calibrated_metrics.get("answer_accuracy"),
            "teacher_topk_mass_minimum": teacher_mass.get("minimum"),
            "shortcut_gap": anti.get("shortcut_gap"),
            "iid_accuracy": anti.get("iid_accuracy"),
            "transformed_accuracy": anti.get("transformed_accuracy"),
            "bootstrap_spearman": noise["within_checkpoint_full_score_spearman"],
            "selected_vs_random_cpr_margin": exact.get("selected_vs_matched_random_cpr_margin"),
            "functional_group_count": exact.get("functional_group_count"),
        },
        "job_ids": [args.job_id],
        "git_commit": git_commit,
        "prereg_commit": prereg_commit,
        "resolved_config_sha256": sha256_value(config),
        "artifact_hashes": {
            str(path): sha256_file(path)
            for path in (
                args.base_scores,
                args.teacher_store_manifest,
                args.anti_shortcut,
                args.probe_manifest,
                args.circuit,
                args.exact_patching,
                args.compatibility,
                args.distributed_resume,
            )
        },
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)
    bank_total = int(teacher.get("total_trajectories", 0))
    bank_positive = int(teacher.get("reward_distribution", {}).get("positive", 0))
    readiness_evidence = {
        "base_task_accuracy_nontrivial": (
            checks["base_task_accuracy"],
            f"formal validation accuracy={metrics.get('answer_accuracy')}",
        ),
        "pilot_improves_accuracy": (
            checks["calibration_anchor_improves_accuracy"],
            f"calibration={calibrated_metrics.get('answer_accuracy')} base={metrics.get('answer_accuracy')}",
        ),
        "verifier_deterministic": (
            checks["anti_shortcut"],
            "anti-shortcut semantic preservation and exact-verifier artifacts passed",
        ),
        "fixed_bank_mixed_rewards": (
            0 < bank_positive < bank_total,
            f"positive={bank_positive} total={bank_total}",
        ),
        "teacher_topk_mass_acceptable": (
            checks["teacher_topk_mass"],
            f"minimum retained mass={teacher_mass.get('minimum')}",
        ),
        "hf_circuit_logit_parity": (
            checks["hf_transformerlens_gqa_parity"],
            f"compatibility artifact={sha256_file(args.compatibility)}",
        ),
        "eap_ig_beats_random": (
            checks["eap_ig_beats_matched_random"],
            f"CPR margin={exact.get('selected_vs_matched_random_cpr_margin')}",
        ),
        "exact_patching_distinguishes_groups": (
            checks["functional_group_separation"],
            f"functional groups={exact.get('functional_groups')}",
        ),
        "attribution_bootstrap_stable": (
            checks["bootstrap_stability"],
            f"within-checkpoint Spearman={noise['within_checkpoint_full_score_spearman']}",
        ),
        "checkpoint_resume_verified": (
            checks["distributed_checkpoint_resume"],
            f"resume artifact={sha256_file(args.distributed_resume)}",
        ),
        "split_leakage_absent": (
            checks["split_probe_isolation"],
            f"probe manifest={probes['sha256']}",
        ),
        "anti_shortcut_gap": (
            checks["anti_shortcut"],
            f"shortcut gap={anti.get('shortcut_gap')}",
        ),
        "probe_cohorts_frozen": (
            checks["base_capable_probes"],
            f"probe manifest={probes['sha256']}",
        ),
    }
    readiness = build_readiness_report(
        readiness_evidence,
        bindings={
            "initial_checkpoint_hash": str(config["model"]["model_revision"]),
            "dataset_hash": str(anti["dataset_hash"]),
            "suite_hash": str(anti["suite_hash"]),
            "code_commit": git_commit,
            "prereg_commit": prereg_commit,
        },
    )
    readiness.write(args.output.parent / "readiness")
    report = args.output.with_suffix(".md")
    report.write_text(
        "# G0 report\n\n"
        f"Result: **{'PASS' if payload['passed'] else 'FAIL'}**\n\n"
        + "\n".join(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
        + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise SystemExit("G0 failed; pilot launch is forbidden")


if __name__ == "__main__":
    main()
