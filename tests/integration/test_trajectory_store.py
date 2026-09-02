from __future__ import annotations

import copy
import json

import pytest
import torch

pytest.importorskip("pyarrow")

from posttrain_circuits.artifacts.compatibility import ROLLOUT_GENERATION_VERSION
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.learning.contracts import TrajectoryBatch
from posttrain_circuits.learning.teacher.hf_scorer import HuggingFaceTeacherScorer
from posttrain_circuits.utils.smoke import build_fixed_bank, build_smoke_examples
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


@pytest.mark.integration
def test_trajectory_store_round_trips_policy_reward_teacher_and_masks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tokenizer = build_tiny_tokenizer()
    original = build_fixed_bank(build_smoke_examples(2), tokenizer, 11)
    scorer = HuggingFaceTeacherScorer(
        build_tiny_qwen(12),
        teacher_id="local/test-teacher",
        teacher_revision="test-revision",
        top_k=8,
        minimum_retained_mass=0.0,
    )
    original = scorer.score(TrajectoryBatch(original, policy_version=0)).records
    store = TrajectoryStore(tmp_path / "bank")
    manifest = store.write(
        original,
        behavior_policy={"id": "test-policy", "revision": "test-revision"},
        prompt_manifest_hash="prompt-manifest",
        sampling_configuration={"temperature": 1.0, "top_p": 1.0},
        verifier_version="proofgraph-exact-v1",
        teacher_version="test-revision",
        top_k=8,
    )
    restored = store.read()
    assert store.check_integrity()["sha256"] == manifest["sha256"]
    assert len(restored) == len(original)
    for expected, actual in zip(original, restored, strict=True):
        assert actual.trajectory_id == expected.trajectory_id
        assert actual.policy_version == expected.policy_version
        assert actual.verifier_reward == expected.verifier_reward
        assert actual.response_token_mask == expected.response_token_mask
        assert actual.teacher_topk_ids == expected.teacher_topk_ids
        assert actual.behavior_logprobs == pytest.approx(expected.behavior_logprobs)
        assert torch.allclose(
            torch.tensor(actual.teacher_topk_logprobs),
            torch.tensor(expected.teacher_topk_logprobs),
        )


@pytest.mark.integration
def test_rollout_bank_rejects_pre_eos_padding_rng_repair_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tokenizer = build_tiny_tokenizer()
    records = build_fixed_bank(build_smoke_examples(1), tokenizer, 13)
    store = TrajectoryStore(tmp_path / "rollout-bank")
    manifest = store.write(
        records,
        behavior_policy={"id": "test-policy", "revision": "test-revision"},
        prompt_manifest_hash="prompt-manifest",
        sampling_configuration={"temperature": 1.0, "top_p": 1.0},
        verifier_version="proofgraph-exact-v1",
        teacher_version=None,
        top_k=0,
        extra_metadata={
            "store_kind": "rollout_bank",
            "rollout_generation_version": ROLLOUT_GENERATION_VERSION,
        },
    )
    assert store.check_integrity()["sha256"] == manifest["sha256"]

    manifest_path = store.root / "manifest.json"
    stale_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale_manifest.pop("rollout_generation_version")
    manifest_path.write_text(json.dumps(stale_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible sampling/identity protocol"):
        store.check_integrity()


@pytest.mark.integration
def test_trajectory_store_rejects_duplicate_trajectory_ids_before_writing(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    tokenizer = build_tiny_tokenizer()
    records = build_fixed_bank(build_smoke_examples(2), tokenizer, 17)
    duplicate_records = copy.deepcopy(records)
    duplicate_records[1].trajectory_id = duplicate_records[0].trajectory_id
    root = tmp_path / "duplicate-bank"
    with pytest.raises(ValueError, match="duplicate trajectory_id"):
        TrajectoryStore(root).write(
            duplicate_records,
            behavior_policy={"id": "test-policy", "revision": "test-revision"},
            prompt_manifest_hash="prompt-manifest",
            sampling_configuration={"temperature": 1.0, "top_p": 1.0},
            verifier_version="proofgraph-exact-v1",
            teacher_version=None,
            top_k=0,
        )
    assert not root.exists()
