"""Finalize the hash-bound G0 decision from production artifacts only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.dynamics import estimate_estimator_noise_floor
from posttrain_circuits.circuits.probe_cohorts import validate_probe_cohort_manifest
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.provenance import require_git_output
from posttrain_circuits.core.readiness import build_readiness_report, validate_anti_shortcut_report
from posttrain_circuits.core.scientific_versions import (
    require_core_v2_artifact,
    scientific_compatibility_fields,
)
from posttrain_circuits.data.trajectory_store import TrajectoryStore
from posttrain_circuits.tasks.proofgraph.label_leakage import validate_label_leakage_artifact
from posttrain_circuits.teacher.evaluation import validate_teacher_readiness_artifact


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
    parser.add_argument("--teacher-readiness", type=Path, required=True)
    parser.add_argument("--label-leakage", type=Path, required=True)
    parser.add_argument("--anti-shortcut", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--final-circuit", "--circuit", dest="final_circuit", type=Path, required=True)
    parser.add_argument(
        "--final-exact-patching",
        "--exact-patching",
        dest="final_exact_patching",
        type=Path,
        required=True,
    )
    parser.add_argument("--process-circuit", type=Path, required=True)
    parser.add_argument("--process-exact-patching", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--distributed-resume", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    base = _read(args.base_scores)
    teacher = TrajectoryStore(args.teacher_store_manifest.parent).check_integrity()
    require_core_v2_artifact(teacher)
    initial_checkpoint_hash = sha256_file(args.initial_checkpoint)
    label_leakage = validate_label_leakage_artifact(_read(args.label_leakage))
    teacher_readiness = validate_teacher_readiness_artifact(
        _read(args.teacher_readiness),
        expected_bindings={"dataset_hash": str(label_leakage["dataset_hash"])},
    )
    anti = validate_anti_shortcut_report(
        args.anti_shortcut,
        max_shortcut_gap=float(config["anti_shortcut"]["max_shortcut_gap"]),
        expected_model_checkpoint_hash=initial_checkpoint_hash,
    )
    probes = validate_probe_cohort_manifest(
        args.probe_manifest,
        expected_initial_checkpoint_hash=initial_checkpoint_hash,
    )
    final_circuit = _read(args.final_circuit)
    final_exact = _read(args.final_exact_patching)
    process_circuit = _read(args.process_circuit)
    process_exact = _read(args.process_exact_patching)
    for artifact in (final_circuit, final_exact, process_circuit, process_exact):
        require_core_v2_artifact(
            artifact,
            require_circuit_schema=True,
            require_hash=True,
        )
    compatibility = _read(args.compatibility)
    resume = _read(args.distributed_resume)
    gpu_preflight = _read(args.gpu_preflight)
    gpu_digest = gpu_preflight.pop("sha256", None)
    if gpu_digest != sha256_value(gpu_preflight) or gpu_preflight.get("passed") is not True:
        raise ValueError("G0 requires a passed, hash-valid GPU preflight")
    gpu_preflight["sha256"] = gpu_digest
    final_noise = estimate_estimator_noise_floor(
        final_circuit.get("bootstrap_score_vectors", []), activation_threshold=0.0
    )
    process_noise = estimate_estimator_noise_floor(
        process_circuit.get("bootstrap_score_vectors", []), activation_threshold=0.0
    )
    metrics = base.get("initial_validation_metrics", {})
    calibrated_metrics = base.get("calibrated_validation_metrics", {})
    teacher_mass = teacher.get("teacher_topk_mass", {})
    bank_total = int(teacher.get("total_trajectories", 0))
    bank_positive = int(teacher.get("reward_distribution", {}).get("positive", 0))
    compatibility_payload = {
        key: value for key, value in compatibility.items() if key not in {"sha256", "hf_identity_max_error"}
    }
    checks = {
        "base_task_accuracy": float(metrics.get("answer_accuracy", 0.0))
        >= float(config["anti_shortcut"]["minimum_iid_accuracy"]),
        "teacher_topk_mass": float(teacher_mass.get("minimum", 0.0)) >= 0.90,
        "teacher_correctness": teacher_readiness.get("passed") is True,
        "label_leakage": label_leakage.get("passed") is True,
        "fixed_bank_mixed_rewards": 0 < bank_positive < bank_total,
        "calibration_anchor_improves_accuracy": float(calibrated_metrics.get("answer_accuracy", 0.0))
        > float(metrics.get("answer_accuracy", 0.0)),
        "anti_shortcut": anti.get("passed") is True,
        "base_capable_probes": probes["cohorts"]["base_capable"]["discovery"]["num_examples"] > 0
        and probes["cohorts"]["base_capable"]["validation"]["num_examples"] > 0,
        "probe_scoring_binding": base.get("initial_checkpoint_sha256") == initial_checkpoint_hash
        and probes.get("scoring_manifest_hash") == sha256_file(args.base_scores)
        and probes.get("learnability_evidence_hash") == base.get("sha256"),
        "hf_transformerlens_gqa_parity": compatibility.get("passed") is True
        and compatibility.get("transformerlens_parity_passed") is True
        and compatibility.get("sha256") == sha256_value(compatibility_payload)
        and final_circuit.get("model_compatibility_hash") == compatibility.get("sha256")
        and process_circuit.get("model_compatibility_hash") == compatibility.get("sha256"),
        "bootstrap_stability": min(
            float(final_noise["within_checkpoint_full_score_spearman"]),
            float(process_noise["within_checkpoint_full_score_spearman"]),
        )
        >= float(config["g0"]["minimum_bootstrap_spearman"]),
        "final_stage_eap_beats_matched_random": float(
            final_exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_selected_vs_random_cpr_margin"]),
        "process_stage_eap_beats_matched_random": float(
            process_exact.get("selected_vs_matched_random_cpr_margin", float("-inf"))
        )
        > float(config["g0"]["minimum_selected_vs_random_cpr_margin"]),
        "distinct_stage_manifests": final_circuit.get("probe_stage") == "final_answer"
        and process_circuit.get("probe_stage") in {"first_rule_selection", "intermediate_conclusion"}
        and final_circuit.get("stage_target_manifest_hash")
        != process_circuit.get("stage_target_manifest_hash"),
        "identity_sanity": all(
            artifact.get("sanity_checks", {}).get("identity_passed") is True
            for artifact in (final_exact, process_exact)
        ),
        "full_corruption_sanity": all(
            artifact.get("sanity_checks", {}).get("full_corruption_passed") is True
            for artifact in (final_exact, process_exact)
        ),
        "attribution_exact_calibration": all(
            float(artifact.get("attribution_patching_spearman", float("-inf")))
            >= float(config["g0"]["minimum_attribution_exact_spearman"])
            and float(artifact.get("attribution_patching_spearman_ci", {}).get("lower", float("-inf")))
            >= float(config["g0"]["minimum_spearman_bootstrap_lower_bound"])
            for artifact in (final_exact, process_exact)
        ),
        "distributed_checkpoint_resume": resume.get("passed") is True
        and int(resume.get("world_size", 0)) == 4,
        "split_probe_isolation": probes.get("frozen_before_training") is True,
        "artifact_reconstruction": all(
            circuit.get("checkpoint_sha256") == initial_checkpoint_hash
            and exact.get("artifacts", {}).get("checkpoint_sha256") == initial_checkpoint_hash
            for circuit, exact in (
                (final_circuit, final_exact),
                (process_circuit, process_exact),
            )
        ),
        "gpu_preflight": gpu_preflight.get("passed") is True and int(gpu_preflight.get("world_size", 0)) == 4,
    }
    git_commit = require_git_output(["rev-parse", "HEAD"])
    checks["gpu_preflight_binding"] = (
        gpu_preflight.get("git_commit") == git_commit
        and gpu_preflight.get("model_revision") == config["model"]["model_revision"]
        and gpu_preflight.get("teacher_revision") == config["teacher"]["model_revision"]
    )
    prereg_path = str(config.get("prereg_path", "prereg/core_v2.yaml"))
    prereg_commit = require_git_output(["log", "-n", "1", "--format=%H", "--", prereg_path])
    if config.get("protocol_track") == "qwen3_v1":
        qwen3_bindings = {
            "protocol_track": "qwen3_v1",
            "artifact_namespace": "qwen3-v1",
            "prompt_protocol": "qwen3_non_thinking_v1",
            "enable_thinking": False,
            "chat_template_sha256": config["model"]["prompt_protocol"]["chat_template_sha256"],
            "tokenizer_fingerprint": config["model"]["tokenizer_fingerprint"],
        }
        bound_artifacts = {
            "base_scores": base,
            "teacher_store": teacher,
            "probe_manifest": probes,
            "gpu_preflight": gpu_preflight,
        }
        checks["qwen3_protocol_bindings"] = all(
            all(artifact.get(key) == value for key, value in qwen3_bindings.items())
            for artifact in bound_artifacts.values()
        )
        query_hooks = compatibility.get("hook_positions", {}).get("query_projection", [])
        key_hooks = compatibility.get("hook_positions", {}).get("key_projection", [])
        checks["qwen3_qk_norm_hook_semantics"] = (
            bool(query_hooks and key_hooks)
            and all("pre_q_norm_pre_rope" in value for value in query_hooks)
            and all("pre_k_norm_pre_rope" in value for value in key_hooks)
        )
    payload: dict[str, Any] = {
        "format_version": 2,
        **scientific_compatibility_fields(),
        "phase": "G0",
        "protocol_track": str(config.get("protocol_track", "core_v2")),
        "protocol_prereg_version": str(config.get("protocol_track", "core_v2")),
        "artifact_namespace": str(config["model"].get("artifact_namespace", "legacy")),
        "prompt_protocol": str(config["model"].get("prompt_protocol", {}).get("name", "legacy_raw_v1")),
        "enable_thinking": False,
        "chat_template_sha256": str(
            config["model"].get("prompt_protocol", {}).get("chat_template_sha256", "legacy-unrecorded")
        ),
        "tokenizer_fingerprint": str(config["model"].get("tokenizer_fingerprint", "legacy-unrecorded")),
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "base_accuracy": metrics.get("answer_accuracy"),
            "calibrated_accuracy": calibrated_metrics.get("answer_accuracy"),
            "teacher_topk_mass_minimum": teacher_mass.get("minimum"),
            "shortcut_gap": anti.get("shortcut_gap"),
            "iid_accuracy": anti.get("iid_accuracy"),
            "transformed_accuracy": anti.get("transformed_accuracy"),
            "label_leakage": label_leakage.get("metrics"),
            "teacher_readiness": teacher_readiness.get("metrics"),
            "stages": {
                "final_answer": {
                    "bootstrap_spearman": final_noise["within_checkpoint_full_score_spearman"],
                    "selected_vs_random_cpr_margin": final_exact.get("selected_vs_matched_random_cpr_margin"),
                    "attribution_exact_spearman": final_exact.get("attribution_patching_spearman"),
                },
                str(process_circuit.get("probe_stage")): {
                    "bootstrap_spearman": process_noise["within_checkpoint_full_score_spearman"],
                    "selected_vs_random_cpr_margin": process_exact.get(
                        "selected_vs_matched_random_cpr_margin"
                    ),
                    "attribution_exact_spearman": process_exact.get("attribution_patching_spearman"),
                },
            },
        },
        "job_ids": [args.job_id],
        "git_commit": git_commit,
        "prereg_commit": prereg_commit,
        "prereg_path": prereg_path,
        "prereg_sha256": sha256_file(Path(prereg_path)),
        "resolved_config_sha256": sha256_value(config),
        "model_revision": str(config["model"]["model_revision"]),
        "teacher_revision": str(config["teacher"]["model_revision"]),
        "launch_environment": {
            name: os.environ.get(name)
            for name in (
                "MODEL_CONFIG",
                "TEACHER_CONFIG",
                "PRODUCTION_CONFIG",
                "G0_CONFIG",
                "PILOT_CONFIG",
                "PROJECT_ROOT",
                "PYTHON_BIN",
                "ACCELERATE_BIN",
                "OUTPUT_ROOT",
            )
        },
        "artifact_hashes": {
            str(path): sha256_file(path)
            for path in (
                args.initial_checkpoint,
                args.gpu_preflight,
                args.base_scores,
                args.teacher_store_manifest,
                args.teacher_readiness,
                args.label_leakage,
                args.anti_shortcut,
                args.probe_manifest,
                args.final_circuit,
                args.final_exact_patching,
                args.process_circuit,
                args.process_exact_patching,
                args.compatibility,
                args.distributed_resume,
            )
        },
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)
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
            checks["final_stage_eap_beats_matched_random"]
            and checks["process_stage_eap_beats_matched_random"],
            "both final-answer and process-stage selected circuits beat matched random controls",
        ),
        "exact_patching_distinguishes_groups": (
            checks["distinct_stage_manifests"],
            "distinct frozen target manifests supply stage-specific functional evidence",
        ),
        "attribution_bootstrap_stable": (
            checks["bootstrap_stability"],
            "both stage-specific within-checkpoint Spearman thresholds passed",
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
        "teacher_correctness": (
            checks["teacher_correctness"],
            f"teacher readiness artifact={sha256_file(args.teacher_readiness)}",
        ),
        "label_leakage": (
            checks["label_leakage"],
            f"label leakage artifact={sha256_file(args.label_leakage)}",
        ),
    }
    readiness = build_readiness_report(
        readiness_evidence,
        bindings={
            "initial_checkpoint_hash": initial_checkpoint_hash,
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
