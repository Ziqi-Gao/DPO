from __future__ import annotations

import pytest

from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
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
