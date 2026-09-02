"""Deterministic local fixtures for end-to-end CPU smoke runs."""

from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from transformers import PreTrainedTokenizerBase

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.learning.contracts import (
    PromptBatch,
    SamplingCursor,
    SamplingRequest,
)
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample

SMOKE_SAMPLING_PROTOCOL_ID = "smoke-fixture-v1-stable-cursor"


def build_smoke_examples(count: int = 4, seed: int = 42) -> list[TaskExample]:
    task = ProofGraphTask()
    return [
        task.generate(seed + index, {"depth": 1 + index % 2, "positive": index % 2 == 0})
        for index in range(count)
    ]


def make_trajectory(
    example: TaskExample,
    tokenizer: PreTrainedTokenizerBase,
    *,
    successful: bool,
    policy_version: int,
    seed: int,
    behavior_policy_id: str,
    generation_group_id: str = "",
    generation_group_index: int = 0,
    prompt_group_size: int = 1,
    sampling_cursor: SamplingCursor | None = None,
    sampling_protocol_id: str = SMOKE_SAMPLING_PROTOCOL_ID,
) -> TrajectoryRecord:
    task = ProofGraphTask()
    prompt = task.render(example)
    response = task.canonical_target(example)
    if not successful:
        wrong_bit = 1 - example.label
        response = f"<proof>\n\n</proof>\n<answer>{wrong_bit}</answer>"
    verification = task.verify(example, task.parse_response(response))
    input_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    response_ids = [*list(tokenizer.encode(response, add_special_tokens=False)), tokenizer.eos_token_id]
    cursor = sampling_cursor or SamplingCursor(
        optimizer_step=0,
        retry_index=0,
        prompt_id=example.example_id,
        generation_index=generation_group_index,
    )
    request = SamplingRequest(
        sampling_request_seed=seed,
        sampling_protocol_id=sampling_protocol_id,
        cursors=(cursor,),
    )
    record = TrajectoryRecord(
        trajectory_id="",
        prompt_id=example.example_id,
        split="train",
        prompt_text=prompt,
        input_ids=input_ids,
        response_ids=response_ids,
        response_text=response,
        response_token_mask=[True] * len(response_ids),
        behavior_policy_id=behavior_policy_id,
        behavior_policy_revision="local-smoke-v1",
        policy_version=policy_version,
        sampling_request_seed=seed,
        actual_sampling_seed=request.seed_for(0, policy_version=policy_version),
        sampling_cursor_id=cursor.cursor_id,
        sampling_protocol_id=sampling_protocol_id,
        sampling_temperature=1.0,
        top_p=1.0,
        behavior_logprobs=[0.0] * len(response_ids),
        verifier_reward=verification.reward,
        verification_trace=asdict(verification),
        generation_group_id=generation_group_id,
        generation_group_index=generation_group_index,
        prompt_group_size=prompt_group_size,
        created_at=datetime.now(UTC).isoformat(),
    )
    record.trajectory_id = record.expected_trajectory_id
    record.validate()
    return record


def build_fixed_bank(
    examples: list[TaskExample], tokenizer: PreTrainedTokenizerBase, seed: int
) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for index, example in enumerate(examples):
        order = (True, False) if index % 2 == 0 else (False, True)
        for offset, successful in enumerate(order):
            records.append(
                make_trajectory(
                    example,
                    tokenizer,
                    successful=successful,
                    policy_version=0,
                    seed=seed + index * 2 + offset,
                    behavior_policy_id="common_mu_smoke",
                )
            )
    return records


def build_grouped_fork_bank(
    examples: list[TaskExample],
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
    *,
    group_size: int = 4,
) -> list[TrajectoryRecord]:
    """Build frozen same-prompt groups with both reward classes for fork tests."""

    if group_size < 4 or group_size % 2:
        raise ValueError("fork smoke groups require an even group_size of at least four")
    records: list[TrajectoryRecord] = []
    for prompt_index, example in enumerate(examples):
        group_id = "fork-group-" + sha256_value([example.example_id, seed])[:16]
        for group_index in range(group_size):
            records.append(
                make_trajectory(
                    example,
                    tokenizer,
                    successful=group_index < group_size // 2,
                    policy_version=0,
                    seed=seed + prompt_index * group_size + group_index,
                    behavior_policy_id="common_mu_grouped_smoke",
                    generation_group_id=group_id,
                    generation_group_index=group_index,
                    prompt_group_size=group_size,
                )
            )
    return records


def scripted_current_policy_generator(examples: list[TaskExample], tokenizer: PreTrainedTokenizerBase):
    by_id = {example.example_id: example for example in examples}

    def generate(
        model: Any,
        prompt_batch: PromptBatch,
        policy_version: int,
        sampling_request: SamplingRequest,
    ) -> list[TrajectoryRecord]:
        del model
        sampling_request.validate_for(prompt_batch)
        return [
            make_trajectory(
                by_id[prompt_id],
                tokenizer,
                successful=(int(sha256_value(prompt_id)[:2], 16) + policy_version) % 2 == 0,
                policy_version=policy_version,
                seed=sampling_request.sampling_request_seed,
                behavior_policy_id="current_policy_smoke_fixture",
                sampling_cursor=sampling_request.cursors[index],
                sampling_protocol_id=sampling_request.sampling_protocol_id,
            )
            for index, prompt_id in enumerate(prompt_batch.prompt_ids)
        ]

    return generate


def clone_successes(records: list[TrajectoryRecord]) -> list[TrajectoryRecord]:
    return [copy.deepcopy(record) for record in records if record.verifier_reward == 1.0]
