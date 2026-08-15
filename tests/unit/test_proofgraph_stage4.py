from __future__ import annotations

import json

import pytest

from posttrain_circuits.cli.build_splits import main as build_splits_main
from posttrain_circuits.data.splits import (
    SPLITS,
    build_all_splits,
    build_split,
    difficulty_distribution,
)
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.tokenization import (
    semantic_token_indices,
    tokenization_audit,
)


@pytest.mark.unit
def test_chain_branch_and_converging_dag_are_real_templates() -> None:
    task = ProofGraphTask()
    chain = task.generate(
        2,
        {"structure": "chain", "depth": 3, "positive": True, "distractors": 0},
    )
    assert [str(rule.consequent) for rule in chain.rules.values()] == [
        "I01",
        "I02",
        "Q",
    ]
    assert all(len(rule.antecedents) == 1 for rule in chain.rules.values())

    branch = task.generate(
        2,
        {"structure": "branch", "depth": 2, "positive": True, "distractors": 0},
    )
    branch_rules = list(branch.rules.values())
    assert len(branch_rules) == 3
    assert len(branch_rules[-1].antecedents) == 2
    assert str(branch_rules[-1].consequent) == "Q"

    dag = task.generate(
        2,
        {
            "structure": "converging_dag",
            "depth": 3,
            "positive": True,
            "distractors": 0,
        },
    )
    dag_rules = list(dag.rules.values())
    assert len(dag_rules) == 4
    assert len(dag_rules[1].antecedents) == 2
    assert len(dag_rules[2].antecedents) == 2
    assert str(dag_rules[-1].consequent) == "Q"

    for example in (chain, branch, dag):
        result = task.verify(
            example,
            task.parse_response(task.canonical_target(example)),
        )
        assert result.reward == 1.0


@pytest.mark.unit
def test_configured_ranges_ood_depths_and_proof_multiplicity() -> None:
    task = ProofGraphTask()
    difficulty = {
        "depth_range": [2, 4],
        "ood_depth_range": [5, 7],
        "distractor_range": [2, 5],
        "structures": ["chain", "branch", "converging_dag"],
        "multiple_valid_proof_fraction": 0.5,
    }
    train = build_split(task, "train", 40, 17, difficulty)
    assert {example.label for example in train} == {0, 1}
    assert all(2 <= int(example.metadata["depth"]) <= 4 for example in train)
    assert all(2 <= int(example.metadata["distractors"]) <= 5 for example in train)
    assert {example.metadata["structure"] for example in train} == {
        "chain",
        "branch",
        "converging_dag",
    }
    assert {example.metadata["proof_multiplicity"] for example in train} == {1, 2}

    ood = build_split(task, "ood_depth_test", 20, 17, difficulty)
    assert all(5 <= int(example.metadata["depth"]) <= 7 for example in ood)
    discovery = build_split(task, "circuit_discovery", 20, 17, difficulty)
    assert all(example.metadata["unique_proof"] for example in discovery)
    assert all(example.metadata["proof_multiplicity"] == 1 for example in discovery)
    distribution = difficulty_distribution(train)
    assert sum(distribution["depth"].values()) == 40
    assert sum(distribution["label"].values()) == 40


@pytest.mark.unit
def test_all_seven_splits_are_globally_isolated() -> None:
    splits = build_all_splits(
        ProofGraphTask(),
        split_sizes={split: 4 for split in SPLITS},
        base_seed=11,
        difficulty={
            "depth_range": [2, 4],
            "ood_depth_range": [5, 7],
            "distractor_range": [1, 3],
            "structures": ["chain", "branch", "converging_dag"],
        },
    )
    assert tuple(splits) == SPLITS
    assert sum(len(examples) for examples in splits.values()) == 28


@pytest.mark.unit
def test_tokenization_audit_captures_every_repeated_occurrence(tokenizer) -> None:  # type: ignore[no-untyped-def]
    text = "Q R01 Q R01 Q"
    audit = tokenization_audit(
        tokenizer,
        text,
        {"query": ["Q"], "identifiers": ["R01"]},
        model_family="tiny",
    )
    assert len(audit["spans"]["query"]) == 3
    assert len(audit["spans"]["identifiers"]) == 2
    positions = semantic_token_indices(audit, ["query", "identifiers"])
    assert len(positions) == 5
    assert audit["sha256"]


@pytest.mark.integration
def test_unified_split_cli_writes_all_manifests_and_distributions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "splits"
    build_splits_main(
        [
            "task=a_implies_b",
            "task.num_examples=2",
            "--output",
            str(output),
        ]
    )
    global_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert global_manifest["leakage_check"] == "passed"
    assert set(global_manifest["split_hashes"]) == set(SPLITS)
    for split in SPLITS:
        manifest = json.loads((output / split / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["difficulty_distribution"]["depth"]
        assert manifest["difficulty_distribution"]["structure"]
