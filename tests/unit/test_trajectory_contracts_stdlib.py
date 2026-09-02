from __future__ import annotations

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.datasets.trajectories.manifests import (
    METADATA_FILENAME,
    TOKENS_FILENAME,
    build_store_manifest,
    validate_store_manifest,
)
from posttrain_circuits.datasets.trajectories.store import TrajectoryStore
from posttrain_circuits.learning.contracts import SamplingCursor, SamplingRequest


def _record(prompt_id: str, request: SamplingRequest, index: int, response: int) -> TrajectoryRecord:
    cursor = request.cursors[index]
    value = TrajectoryRecord(
        trajectory_id="",
        prompt_id=prompt_id,
        split="train",
        prompt_text=f"prompt-{prompt_id}",
        input_ids=[1, 2],
        response_ids=[response],
        response_text=str(response),
        response_token_mask=[True],
        behavior_policy_id="policy",
        behavior_policy_revision="revision",
        policy_version=3,
        sampling_request_seed=request.sampling_request_seed,
        actual_sampling_seed=request.seed_for(index, policy_version=3),
        sampling_cursor_id=cursor.cursor_id,
        sampling_protocol_id=request.sampling_protocol_id,
        sampling_temperature=1.0,
        top_p=1.0,
        behavior_logprobs=[-0.5],
    )
    value.trajectory_id = value.expected_trajectory_id
    value.validate()
    return value


class TrajectoryContractTests(unittest.TestCase):
    def test_batch_reordering_preserves_actual_seed_and_trajectory_id(self) -> None:
        cursors = {
            prompt_id: SamplingCursor(optimizer_step=4, retry_index=1, prompt_id=prompt_id, generation_index=0)
            for prompt_id in ("a", "b")
        }
        first = SamplingRequest(17, "sampling-protocol-v1", (cursors["a"], cursors["b"]))
        second = SamplingRequest(17, "sampling-protocol-v1", (cursors["b"], cursors["a"]))
        first_records = {
            "a": _record("a", first, 0, 7),
            "b": _record("b", first, 1, 8),
        }
        second_records = {
            "b": _record("b", second, 0, 8),
            "a": _record("a", second, 1, 7),
        }
        for prompt_id in ("a", "b"):
            self.assertEqual(
                first_records[prompt_id].actual_sampling_seed,
                second_records[prompt_id].actual_sampling_seed,
            )
            self.assertEqual(
                first_records[prompt_id].trajectory_id,
                second_records[prompt_id].trajectory_id,
            )

    def test_actual_seed_and_response_are_trajectory_identity_inputs(self) -> None:
        cursor = SamplingCursor(0, 0, "a", 0)
        first = _record("a", SamplingRequest(17, "sampling-protocol-v1", (cursor,)), 0, 7)
        changed_seed = _record("a", SamplingRequest(18, "sampling-protocol-v1", (cursor,)), 0, 7)
        changed_response = _record(
            "a", SamplingRequest(17, "sampling-protocol-v1", (cursor,)), 0, 8
        )
        self.assertNotEqual(first.trajectory_id, changed_seed.trajectory_id)
        self.assertNotEqual(first.trajectory_id, changed_response.trajectory_id)

    def test_manifest_file_and_row_inventory_is_exact(self) -> None:
        manifest = build_store_manifest(
            behavior_policy={"id": "policy", "revision": "revision"},
            prompt_manifest_hash="prompts",
            sampling_configuration={"temperature": 1.0},
            sampling_protocol_id="sampling-protocol-v1",
            verifier_version="verifier-v1",
            teacher_version=None,
            top_k=0,
            ordered_trajectory_ids=["trajectory-a"],
            teacher_scored=False,
            file_sha256={METADATA_FILENAME: "a" * 64, TOKENS_FILENAME: "b" * 64},
            row_counts={METADATA_FILENAME: 1, TOKENS_FILENAME: 1},
            reward_distribution={"mean": 0.0, "positive": 0},
            length_distribution={"minimum": 1, "maximum": 1, "mean": 1.0},
            effective_supervised_tokens=1,
        )
        validate_store_manifest(manifest)
        stale = copy.deepcopy(manifest)
        stale["files"]["teacher-00000.safetensors"] = "c" * 64
        stale["row_counts"]["teacher-00000.safetensors"] = 1
        with self.assertRaisesRegex(ValueError, "file inventory is not exact"):
            validate_store_manifest(stale)

    def test_store_rejects_mixed_teacher_scoring_before_dynamic_dependencies(self) -> None:
        request = SamplingRequest(17, "sampling-protocol-v1", (SamplingCursor(0, 0, "a", 0),))
        unscored = _record("a", request, 0, 7)
        scored = copy.deepcopy(unscored)
        scored.response_ids = [8]
        scored.response_text = "8"
        scored.teacher_id = "teacher"
        scored.teacher_revision = "revision"
        scored.teacher_topk_ids = [[1]]
        scored.teacher_topk_logprobs = [[-0.1]]
        scored.teacher_topk_mass = [0.9]
        scored.teacher_entropy = [0.2]
        scored.trajectory_id = scored.expected_trajectory_id
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "cannot mix teacher-scored and unscored"):
                TrajectoryStore(Path(directory) / "store").write(
                    [unscored, scored],
                    behavior_policy={"id": "policy", "revision": "revision"},
                    prompt_manifest_hash="prompts",
                    sampling_configuration={"temperature": 1.0},
                    verifier_version="verifier-v1",
                    teacher_version="teacher-revision",
                    top_k=1,
                )


if __name__ == "__main__":
    unittest.main()
