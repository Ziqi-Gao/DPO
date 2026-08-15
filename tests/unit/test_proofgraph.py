from __future__ import annotations

import copy
from dataclasses import asdict

import pytest

from posttrain_circuits.core.manifests import DatasetManifest
from posttrain_circuits.data.splits import assert_split_isolation, build_split
from posttrain_circuits.tasks.proofgraph.generator import CORRUPTIONS, ProofGraphTask


@pytest.mark.unit
def test_generation_is_deterministic_and_balanced(task: ProofGraphTask) -> None:
    first = task.generate(42, {"depth": 2})
    second = task.generate(42, {"depth": 2})
    assert asdict(first) == asdict(second)
    labels = [task.generate(seed, {}).label for seed in range(20)]
    assert sum(labels) == 10


@pytest.mark.unit
@pytest.mark.parametrize("depth", [1, 2, 4, 7])
def test_canonical_proofs_verify(task: ProofGraphTask, depth: int) -> None:
    example = task.generate(2, {"depth": depth, "positive": True})
    result = task.verify(example, task.parse_response(task.canonical_target(example)))
    assert result.parse_valid and result.proof_valid and result.answer_correct
    assert result.reward == 1.0
    assert len(result.step_results) == depth


@pytest.mark.unit
def test_negative_answer_with_empty_proof_verifies(task: ProofGraphTask) -> None:
    example = task.generate(3, {"positive": False})
    result = task.verify(example, task.parse_response(task.canonical_target(example)))
    assert result.reward == 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ("R01(F01)", "unknown_citation"),
        ("R99(F01)", "unknown_rule"),
    ],
)
def test_invalid_citations_and_rules_are_rejected(task: ProofGraphTask, replacement: str, error: str) -> None:
    example = task.generate(2, {"positive": True})
    target = task.canonical_target(example)
    if error == "unknown_citation":
        target = target.replace("R01(F01)", "R01(F99)")
    else:
        target = target.replace("R01(F01)", replacement)
    result = task.verify(example, task.parse_response(target))
    assert result.reward == 0.0
    assert result.error_code == error


@pytest.mark.unit
def test_conclusion_and_answer_mismatch_are_rejected(task: ProofGraphTask) -> None:
    example = task.generate(2, {"positive": True})
    wrong_conclusion = task.canonical_target(example).replace("-> I01", "-> WRONG", 1)
    assert task.verify(example, task.parse_response(wrong_conclusion)).error_code == "conclusion_mismatch"
    wrong_answer = task.canonical_target(example).replace("<answer>1", "<answer>0")
    assert task.verify(example, task.parse_response(wrong_answer)).reward == 0.0


@pytest.mark.unit
def test_parser_is_whitespace_tolerant_but_rejects_extra_text(task: ProofGraphTask) -> None:
    example = task.generate(2, {"depth": 1, "positive": True})
    target = task.canonical_target(example).replace(": ", ":    ").replace("</proof>", "\n</proof>")
    assert task.parse_response(target).parse_valid
    assert not task.parse_response(target + " trailing").parse_valid


@pytest.mark.unit
def test_all_counterfactuals_record_one_changed_field(task: ProofGraphTask) -> None:
    example = task.generate(2, {"positive": True, "distractors": 2})
    for index, corruption in enumerate(sorted(CORRUPTIONS)):
        pair = task.make_counterfactual(example, corruption, 100 + index)
        assert pair.corruption_type == corruption
        assert pair.changed_semantic_field
        assert pair.clean_prompt != pair.corrupt_prompt or corruption == "critical_rule_relocation"
        assert task.parse_response(pair.clean_target).parse_valid
        assert task.parse_response(pair.corrupt_target).parse_valid
    for corruption in {
        "fact_truth_flip",
        "necessary_fact_replacement",
        "critical_rule_consequent_replacement",
        "query_flip",
    }:
        pair = task.make_counterfactual(example, corruption, 5)
        assert pair.clean_target != pair.corrupt_target


@pytest.mark.unit
def test_negative_query_flip_builds_a_verified_positive_proof(task: ProofGraphTask) -> None:
    example = task.generate(3, {"depth": 3, "positive": False})
    pair = task.make_counterfactual(example, "query_flip", 99)
    result = task.verify(pair.corrupt_example, task.parse_response(pair.corrupt_target))
    assert pair.corrupt_example.label == 1
    assert len(pair.corrupt_example.canonical_proof) == 3
    assert result.reward == 1.0


@pytest.mark.unit
def test_split_isolation_and_manifest_hash_stability(task: ProofGraphTask) -> None:
    discovery = build_split(task, "circuit_discovery", 8, 11)
    validation = build_split(task, "circuit_validation", 8, 11)
    assert_split_isolation({"circuit_discovery": discovery, "circuit_validation": validation})
    payload = [asdict(example) for example in discovery]
    arguments = dict(
        dataset_id="test",
        generator_version=task.generator_version,
        git_commit="abc",
        task_config={"depth": 2},
        split_name="circuit_discovery",
        seed_range=(11, 18),
        num_examples=8,
        difficulty_distribution={"depth": 2},
    )
    first = DatasetManifest(**arguments).finalize(payload)
    second = DatasetManifest(**copy.deepcopy(arguments)).finalize(payload)
    assert first.created_at != ""
    assert first.sha256 == second.sha256
