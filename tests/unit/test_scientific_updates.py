from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from posttrain_circuits.circuits.dynamics import (
    circuit_stability_report,
    estimate_estimator_noise_floor,
)
from posttrain_circuits.circuits.probe_cohorts import (
    build_probe_cohort_manifest,
    validate_probe_cohort_manifest,
    write_probe_cohort_manifest,
)
from posttrain_circuits.cli.analyze_circuit_dynamics import _mask_transfer
from posttrain_circuits.cli.build_probe_cohorts import _eligible_candidates
from posttrain_circuits.cli.finalize_pilot import _curve
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.readiness import (
    require_factorial_prerequisites,
    validate_anti_shortcut_report,
)
from posttrain_circuits.tasks.proofgraph.anti_shortcut import (
    TRANSFORMATIONS,
    build_anti_shortcut_suite,
    evaluate_anti_shortcut_suite,
)
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.training.local_fork import (
    _probe_kl,
    calibrate_learning_rate_for_output_kl,
    output_kl_match_status,
)


@pytest.mark.unit
def test_anti_shortcut_transformations_preserve_exact_proof_semantics() -> None:
    task = ProofGraphTask()
    examples = [
        task.generate(10, {"positive": True, "distractors": 2}),
        task.generate(11, {"positive": False, "distractors": 2}),
    ]
    cases = build_anti_shortcut_suite(examples, seed=91, distractor_ood_count=8)
    assert len(cases) == len(examples) * len(TRANSFORMATIONS)
    assert {case.transformation for case in cases} == set(TRANSFORMATIONS)
    for case in cases:
        result = task.verify(
            case.example,
            task.parse_response(task.canonical_target(case.example)),
        )
        assert result.reward == 1.0
        assert case.example.metadata["source_example_id"] == case.source_example_id
    report = evaluate_anti_shortcut_suite(
        examples,
        cases,
        lambda example, _prompt: task.canonical_target(example),
        max_shortcut_gap=0.05,
        model_checkpoint_hash="base-commit",
    )
    assert report["iid_accuracy"] == 1.0
    assert report["transformed_accuracy"] == 1.0
    assert report["shortcut_gap"] == 0.0
    assert report["passed"] is True


@pytest.mark.unit
def test_anti_shortcut_gate_rejects_failed_or_wrong_checkpoint_report(tmp_path: Path) -> None:
    task = ProofGraphTask()
    examples = [task.generate(20, {"positive": True, "distractors": 1})]
    cases = build_anti_shortcut_suite(examples, seed=2, distractor_ood_count=4)

    def shortcut_predictor(example, _prompt):  # type: ignore[no-untyped-def]
        if "anti_shortcut_transformation" in example.metadata:
            return "<proof></proof><answer>0</answer>"
        return task.canonical_target(example)

    report = evaluate_anti_shortcut_suite(
        examples,
        cases,
        shortcut_predictor,
        max_shortcut_gap=0.05,
        model_checkpoint_hash="base-commit",
    )
    path = tmp_path / "anti-shortcut.json"
    atomic_write_json(path, report)
    with pytest.raises(RuntimeError, match="gate failed"):
        validate_anti_shortcut_report(
            path,
            max_shortcut_gap=0.05,
            expected_model_checkpoint_hash="base-commit",
        )
    passing = copy.deepcopy(report)
    passing.update({"shortcut_gap": 0.0, "transformed_accuracy": 1.0, "passed": True})
    passing.pop("sha256")
    from posttrain_circuits.core.hashing import sha256_value

    passing["sha256"] = sha256_value(passing)
    atomic_write_json(path, passing)
    with pytest.raises(RuntimeError, match="different initial student"):
        validate_anti_shortcut_report(
            path,
            max_shortcut_gap=0.05,
            expected_model_checkpoint_hash="other-commit",
        )


def _probe_rows() -> dict[str, list[dict[str, object]]]:
    return {
        "discovery": [
            {"example_id": "d-capable", "payload": 1},
            {"example_id": "d-challenge", "payload": 2},
        ],
        "validation": [
            {"example_id": "v-capable", "payload": 3},
            {"example_id": "v-challenge", "payload": 4},
        ],
    }


@pytest.mark.unit
def test_probe_cohorts_are_complete_disjoint_and_hash_pinned(tmp_path: Path) -> None:
    scores = {
        "d-capable": {"initial_correct": True, "learnable_after_post_training": True},
        "d-challenge": {"initial_correct": False, "learnable_after_post_training": True},
        "v-capable": {"initial_correct": True, "learnable_after_post_training": True},
        "v-challenge": {"initial_correct": False, "learnable_after_post_training": True},
    }
    manifest = build_probe_cohort_manifest(
        _probe_rows(),
        scores,
        source_split_hashes={"discovery": "discovery-hash", "validation": "validation-hash"},
        initial_student_checkpoint_hash="local-random-v1",
        scoring_manifest_hash="score-hash",
        learnability_evidence_hash="pilot-hash",
    )
    write_probe_cohort_manifest(tmp_path, manifest)
    validated = validate_probe_cohort_manifest(tmp_path / "manifest.json")
    assert validated["sha256"] == manifest["sha256"]
    assert validated["git_commit"] == "test-unfrozen"
    assert validated["prereg_commit"] == "test-unfrozen"
    assert validated["training_ancestry"] == []
    assert validated["cohorts"]["base_capable"]["discovery"]["num_examples"] == 1
    assert validated["cohorts"]["challenge"]["validation"]["num_examples"] == 1
    bad_scores = copy.deepcopy(scores)
    bad_scores["d-challenge"]["learnable_after_post_training"] = False
    with pytest.raises(ValueError, match="learnability"):
        build_probe_cohort_manifest(
            _probe_rows(),
            bad_scores,
            source_split_hashes={"discovery": "d", "validation": "v"},
            initial_student_checkpoint_hash="base",
            scoring_manifest_hash="scores",
            learnability_evidence_hash="pilot",
        )


@pytest.mark.unit
def test_unlearned_probe_candidates_are_excluded_with_a_hash_audit() -> None:
    rows = [
        {"example_id": "base"},
        {"example_id": "challenge"},
        {"example_id": "not-eligible"},
    ]
    scores = {
        "base": {"initial_correct": True, "learnable_after_post_training": True},
        "challenge": {"initial_correct": False, "learnable_after_post_training": True},
        "not-eligible": {"initial_correct": False, "learnable_after_post_training": False},
    }
    selected, audit = _eligible_candidates(rows, scores)
    assert [row["example_id"] for row in selected] == ["base", "challenge"]
    assert audit["candidate_count"] == 3
    assert audit["selected_count"] == 2
    assert audit["excluded"][0]["example_id"] == "not-eligible"
    from posttrain_circuits.core.hashing import sha256_value

    assert audit["sha256"] == sha256_value({key: value for key, value in audit.items() if key != "sha256"})


@pytest.mark.unit
def test_factorial_preflight_requires_both_hash_pinned_gates(tmp_path: Path) -> None:
    task = ProofGraphTask()
    examples = [task.generate(30, {"positive": True, "distractors": 1})]
    cases = build_anti_shortcut_suite(examples, seed=3, distractor_ood_count=4)
    report = evaluate_anti_shortcut_suite(
        examples,
        cases,
        lambda example, _prompt: task.canonical_target(example),
        max_shortcut_gap=0.05,
        model_checkpoint_hash="local-random-v1",
    )
    anti_path = tmp_path / "anti.json"
    atomic_write_json(anti_path, report)
    scores = {
        key: {
            "initial_correct": "capable" in key,
            "learnable_after_post_training": True,
        }
        for rows in _probe_rows().values()
        for key in [str(row["example_id"]) for row in rows]
    }
    probes = build_probe_cohort_manifest(
        _probe_rows(),
        scores,
        source_split_hashes={"discovery": "d", "validation": "v"},
        initial_student_checkpoint_hash="local-random-v1",
        scoring_manifest_hash="scores",
        learnability_evidence_hash="pilot",
    )
    probe_root = tmp_path / "probes"
    write_probe_cohort_manifest(probe_root, probes)
    config = compose_config(
        [
            "experiment=offline_soft",
            f"anti_shortcut.report_path={anti_path}",
            f"production_safety.probe_cohort_manifest={probe_root / 'manifest.json'}",
        ]
    )
    evidence = require_factorial_prerequisites(config)
    assert set(evidence) == {"anti_shortcut", "probe_cohorts"}


@pytest.mark.unit
def test_local_fork_uses_new_to_fork_kl_and_output_kl_calibration() -> None:
    fork = torch.tensor([[[2.0, 0.0]]])
    new = torch.tensor([[[0.5, 1.5]]])
    fork_log = fork.log_softmax(-1)
    new_log = new.log_softmax(-1)
    expected = float((new_log.exp() * (new_log - fork_log)).sum())
    reverse = float((fork_log.exp() * (fork_log - new_log)).sum())
    assert _probe_kl(fork, new) == pytest.approx(expected)
    assert expected != pytest.approx(reverse)
    calibrated = calibrate_learning_rate_for_output_kl(0.04, 0.01, 1e-4)
    assert calibrated == pytest.approx(2e-4)
    assert output_kl_match_status(0.04, 0.041, relative_tolerance=0.1)["within_tolerance"]


@pytest.mark.unit
def test_noise_floor_and_excess_churn_are_reported_with_causal_evidence() -> None:
    source = {"a": 1.0, "b": 0.3, "c": 0.0}
    target = {"a": 0.7, "b": 0.1, "c": 0.4}
    source_bootstrap = [
        {"a": 1.0, "b": 0.3, "c": 0.0},
        {"a": 0.95, "b": 0.32, "c": 0.01},
        {"a": 1.02, "b": 0.28, "c": -0.01},
    ]
    target_bootstrap = [
        {"a": 0.7, "b": 0.1, "c": 0.4},
        {"a": 0.68, "b": 0.12, "c": 0.39},
        {"a": 0.73, "b": 0.09, "c": 0.42},
    ]
    noise = estimate_estimator_noise_floor(source_bootstrap, activation_threshold=0.2)
    assert noise["estimated_noise_churn"] > 0
    report = circuit_stability_report(
        source_scores=source,
        target_scores=target,
        source_bootstrap_score_vectors=source_bootstrap,
        target_bootstrap_score_vectors=target_bootstrap,
        activation_threshold=0.2,
        cross_checkpoint_mask_transfer={"necessity": 0.4},
        heldout_exact_patching_effects={"necessity": 0.5, "sufficiency": 0.3},
    )
    assert report["excess_churn"] == pytest.approx(
        report["observed_cross_checkpoint_churn"] - report["estimated_noise_churn"]
    )
    assert report["thresholded_jaccard_role"] == "diagnostic_only_not_sole_churn_evidence"
    assert report["full_score_spearman_stability"] <= 1.0


@pytest.mark.unit
def test_dynamics_consumes_only_explicit_mask_transfer_evidence() -> None:
    transfer = [{"sparsity": 0.1, "necessity": 0.3}]
    assert _mask_transfer({"cross_checkpoint_mask_transfer": transfer}) == transfer
    with pytest.raises(ValueError, match="lacks cross_checkpoint_mask_transfer"):
        _mask_transfer({"cpr": 0.9})


@pytest.mark.unit
def test_pilot_curves_skip_intentionally_unevaluated_steps() -> None:
    steps, values = _curve(
        [
            {"step": 1, "validation_accuracy": None},
            {"step": 20, "validation_accuracy": 0.25},
            {"step": 40, "validation_accuracy": 0.5},
        ],
        "validation_accuracy",
    )
    assert steps == [20.0, 40.0]
    assert values == [0.25, 0.5]


@pytest.mark.unit
def test_gemma_core_replication_config_contains_only_minimum_cells() -> None:
    config = compose_config(["replication=gemma2_2b_core"])
    replication = config["replication"]
    assert replication["student"] == "gemma2_2b"
    assert replication["teacher"] == "gemma2_teacher_9b"
    assert replication["complete_factorial_required"] is False
    assert len(replication["confirmatory_contrasts"]) == 3
    prereg = Path("prereg/core_v1.yaml").read_text(encoding="utf-8")
    assert "local_fork_output_kl_matching_rule" in prereg
    assert "escalation_1p5b_to_3b" in prereg
    assert "observed_effect_direction: never_a_gate" in prereg
    assert "at_least_two_of_three_seeds_have_the_preregistered_direction" not in prereg
