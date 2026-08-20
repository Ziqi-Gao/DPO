from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from posttrain_circuits.analysis.shared_state import (
    SharedStateRecord,
    load_shared_state_partition,
    validate_shared_state_manifest,
    write_partitioned_shared_state,
)
from posttrain_circuits.analysis.stage7 import (
    SharedStateObservation,
    analyze_shared_state,
)
from posttrain_circuits.cli.analyze_shared_state import main as analyze_main
from posttrain_circuits.tasks.anchors import (
    BaseAccuracyBelowThreshold,
    build_fixed_anchor_pilots,
    require_base_accuracy,
    write_anchor_pilots,
)
from posttrain_circuits.tasks.math_bridge.canonical_prefixes import (
    CanonicalTeacherSolution,
    digit_counterfactual,
    equation_counterfactual,
    final_answer_counterfactual,
    load_canonical_teacher_solutions,
    make_prefix_target,
    operator_counterfactual,
    shared_prefix_next_token_metric,
    write_canonical_teacher_solutions,
)


def _shared_records(source_mode: str) -> list[SharedStateRecord]:
    prefix = "canonical" if source_mode == "canonical_prefix" else "natural"
    return [
        SharedStateRecord(
            record_id=f"{prefix}-{split}",
            task="small_addition",
            prompt="What is 12 + 7?",
            prefix="Compute 12 + ",
            continuation="7 = 19.",
            source_mode=source_mode,  # type: ignore[arg-type]
            split=split,  # type: ignore[arg-type]
            metadata={"teacher_id": "tiny-teacher"},
        )
        for split in ("discovery", "validation")
    ]


@pytest.mark.unit
def test_shared_state_partitions_are_hash_bound_and_never_pooled(
    tmp_path: Path,
) -> None:
    manifest = write_partitioned_shared_state(
        tmp_path,
        canonical_prefix=_shared_records("canonical_prefix"),
        natural_rollout=_shared_records("natural_rollout"),
    )
    validated = validate_shared_state_manifest(tmp_path / "manifest.json")
    assert validated == manifest
    canonical = load_shared_state_partition(
        tmp_path / "canonical_prefix.json",
        expected_mode="canonical_prefix",
    )
    natural = load_shared_state_partition(
        tmp_path / "natural_rollout.json",
        expected_mode="natural_rollout",
    )
    assert {record.source_mode for record in canonical} == {"canonical_prefix"}
    assert {record.source_mode for record in natural} == {"natural_rollout"}
    with pytest.raises(ValueError, match="expected natural_rollout"):
        load_shared_state_partition(
            tmp_path / "canonical_prefix.json",
            expected_mode="natural_rollout",
        )


@pytest.mark.unit
def test_anchor_pilots_are_fixed_disjoint_and_accuracy_gated(
    tmp_path: Path,
) -> None:
    first = build_fixed_anchor_pilots(
        seed=13,
        discovery_per_task=2,
        validation_per_task=2,
    )
    second = build_fixed_anchor_pilots(
        seed=13,
        discovery_per_task=2,
        validation_per_task=2,
    )
    assert first == second
    assert {example.task for example in first["discovery"]} == {
        "greater_than",
        "small_addition",
        "entity_tracking",
    }
    assert not (
        {example.prompt for example in first["discovery"]}
        & {example.prompt for example in first["validation"]}
    )
    manifest = write_anchor_pilots(tmp_path, first, seed=13)
    assert manifest["splits"]["discovery"]["example_count"] == 6
    predictions = {example.anchor_id: example.answer for example in first["validation"]}
    result = require_base_accuracy(
        first["validation"],
        predictions,
        threshold=0.8,
    )
    assert result.passed
    predictions[first["validation"][0].anchor_id] = "wrong"
    with pytest.raises(BaseAccuracyBelowThreshold):
        require_base_accuracy(
            first["validation"],
            predictions,
            threshold=0.8,
        )


@pytest.mark.unit
def test_math_bridge_counterfactuals_and_shared_prefix_metric(
    tmp_path: Path,
) -> None:
    canonical = CanonicalTeacherSolution(
        example_id="math-1",
        prompt="What is 12 + 7?",
        solution="Compute 12 + 7 = 19. Final answer: 19",
        final_answer="19",
        teacher_id="teacher",
        teacher_revision="revision",
    )
    canonical_path = tmp_path / "canonical_solutions.json"
    payload = write_canonical_teacher_solutions(canonical_path, [canonical])
    assert payload["teacher_revisions"] == ["revision"]
    assert load_canonical_teacher_solutions(canonical_path) == [canonical]
    target = make_prefix_target(
        canonical.prompt,
        canonical.solution,
        canonical.solution.index("7"),
        "second_operand",
    )
    assert target.canonical_prefix.endswith("+ ")
    counterfactuals = (
        digit_counterfactual(canonical.solution, replacement="3"),
        operator_counterfactual(canonical.solution, replacement="-"),
        equation_counterfactual(canonical.solution, replacement="20"),
        final_answer_counterfactual(canonical, replacement="20"),
    )
    assert {counterfactual.kind for counterfactual in counterfactuals} == {
        "digit",
        "operator",
        "equation",
        "final_answer",
    }
    assert all(
        counterfactual.intervened_text != counterfactual.original_text for counterfactual in counterfactuals
    )
    logits = torch.zeros((2, 3, 5))
    logits[:, -1, 2] = torch.tensor([3.0, 5.0])
    logits[:, -1, 1] = torch.tensor([1.0, 2.0])
    metric = shared_prefix_next_token_metric(
        logits,
        target_token_id=2,
        contrast_token_id=1,
    )
    assert metric.item() == pytest.approx(2.5)


def _stage7_observations() -> list[SharedStateObservation]:
    observations = []
    sources = {
        "canonical_prefix": ("canonical-discovery", "canonical-validation"),
        "natural_rollout": ("natural-discovery", "natural-validation"),
    }
    for source_mode, prompt_ids in sources.items():
        for training_seed in (11, 22, 33):
            for prompt_index, prompt_id in enumerate(prompt_ids):
                for circuit_replicate in (0, 1):
                    for state_source_index, state_source in enumerate(
                        ("fixed_bank", "current_policy"),
                    ):
                        for supervision_index, supervision in enumerate(
                            ("hard_teacher", "soft_teacher"),
                        ):
                            interaction = 0.7 * state_source_index * supervision_index
                            source_shift = 0.2 if source_mode == "natural_rollout" else 0.0
                            jitter = (
                                training_seed / 10_000 + prompt_index / 1_000 + circuit_replicate / 10_000
                            )
                            observations.append(
                                SharedStateObservation(
                                    record_id=prompt_id,
                                    source_mode=source_mode,  # type: ignore[arg-type]
                                    state_source=state_source,
                                    supervision=supervision,
                                    training_seed=training_seed,
                                    circuit_replicate=circuit_replicate,
                                    next_token_metric=(
                                        1.0
                                        + state_source_index
                                        + 2.0 * supervision_index
                                        + interaction
                                        + source_shift
                                        + jitter
                                    ),
                                    locking=0.3 + source_shift + jitter,
                                    cpr=0.6 + source_shift + jitter,
                                    cmd=0.2 + source_shift + jitter,
                                )
                            )
    return observations


@pytest.mark.unit
def test_stage7_three_level_factorial_fdr_and_cli(
    tmp_path: Path,
) -> None:
    records_dir = tmp_path / "shared"
    manifest = write_partitioned_shared_state(
        records_dir,
        canonical_prefix=_shared_records("canonical_prefix"),
        natural_rollout=_shared_records("natural_rollout"),
    )
    observations = _stage7_observations()
    result = analyze_shared_state(
        observations,
        bootstrap_samples=40,
        seed=9,
    )
    assert result["source_modes_pooled"] is False
    assert result["natural_rollout"]["sensitivity_analysis_only"] is True
    assert result["three_level_variability"] == [
        "training_seed",
        "prompt",
        "circuit_bootstrap",
    ]
    for source_mode in ("canonical_prefix", "natural_rollout"):
        intervals = result[source_mode]["metric_intervals"]
        assert set(intervals) == {"locking", "cpr", "cmd"}
        assert intervals["locking"]["lower"] <= intervals["locking"]["estimate"]
        coefficients = result[source_mode]["factorial_interaction"]["coefficients"]
        interaction_terms = [coefficient for coefficient in coefficients if ":" in coefficient["term"]]
        assert interaction_terms
        assert result[source_mode]["factorial_interaction"]["covariance_type"] == (
            "descriptive_seed_level_no_asymptotic_inference"
        )
        assert all(math.isnan(coefficient["fdr_p_value"]) for coefficient in coefficients)
        assert all(math.isnan(coefficient["standard_error"]) for coefficient in coefficients)

    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps({"observations": [asdict(observation) for observation in observations]}),
        encoding="utf-8",
    )
    output = tmp_path / "analysis.json"
    analyze_main(
        [
            "--shared-state-manifest",
            str(records_dir / "manifest.json"),
            "--observations",
            str(observations_path),
            "--output",
            str(output),
            "--bootstrap-samples",
            "20",
            "--seed",
            "9",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["shared_state_manifest_sha256"] == manifest["manifest_sha256"]
    assert payload["primary_estimand"] == ("canonical_prefix_next_token_metric")
