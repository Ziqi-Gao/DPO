from __future__ import annotations

import pytest

from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.manifests import write_dataset_family
from posttrain_circuits.datasets.proofgraph.splits import SPLITS, build_all_splits
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


@pytest.fixture
def task() -> ProofGraphTask:
    return ProofGraphTask()


@pytest.fixture
def tokenizer():  # type: ignore[no-untyped-def]
    return build_tiny_tokenizer()


@pytest.fixture
def tiny_model():  # type: ignore[no-untyped-def]
    return build_tiny_qwen(7)


@pytest.fixture
def dataset_family_path(tmp_path):  # type: ignore[no-untyped-def]
    root = tmp_path / "proofgraph-family"
    splits = build_all_splits(
        ProofGraphTask(),
        split_sizes={split: 4 for split in SPLITS},
        base_seed=314,
        difficulty={"depth": 2, "distractors": 2, "structure": "chain"},
    )
    write_dataset_family(root, splits)
    return root
