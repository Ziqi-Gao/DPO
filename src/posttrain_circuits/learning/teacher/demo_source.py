"""Accepted-view teacher demonstrations exposed to canonical SFT."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from posttrain_circuits.datasets.teacher_demos.contracts import (
    LOGPROB_AVAILABLE,
    TeacherDemoAttempt,
)
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.learning.contracts import PromptBatch, TrajectoryBatch

TEACHER_DEMO_CURSOR_PROTOCOL_ID = "teacher-demo-round-robin-v2-accepted-view"


def _as_training_record(attempt: TeacherDemoAttempt) -> TrajectoryRecord:
    attempt.validate()
    if not attempt.accepted:
        raise ValueError("canonical SFT cannot consume a rejected teacher attempt")
    behavior_logprobs = (
        list(attempt.behavior_logprobs)
        if attempt.logprob_status == LOGPROB_AVAILABLE
        else [float("nan")] * len(attempt.response_ids)
    )
    record = TrajectoryRecord(
        trajectory_id="",
        prompt_id=attempt.prompt_id,
        split="train",
        prompt_text=attempt.model_facing_prompt_text,
        input_ids=list(attempt.input_ids),
        response_ids=list(attempt.response_ids),
        response_text=attempt.response_text,
        response_token_mask=list(attempt.response_token_mask),
        behavior_policy_id=attempt.teacher_id,
        behavior_policy_revision=attempt.teacher_revision,
        policy_version=0,
        sampling_request_seed=attempt.sampling_request_seed,
        actual_sampling_seed=attempt.actual_sampling_seed,
        sampling_cursor_id=attempt.attempt_id,
        sampling_protocol_id=attempt.sampling_protocol_id,
        sampling_temperature=attempt.sampling_temperature,
        top_p=attempt.top_p,
        behavior_logprobs=behavior_logprobs,
        verifier_reward=attempt.verifier_reward,
        verification_trace=dict(attempt.verification_trace),
        teacher_id=attempt.teacher_id,
        teacher_revision=attempt.teacher_revision,
        raw_prompt_text=attempt.raw_prompt_text,
        prompt_protocol=attempt.prompt_protocol,
        enable_thinking=attempt.enable_thinking,
        chat_template_sha256=attempt.chat_template_sha256,
        raw_prompt_sha256=attempt.raw_prompt_sha256,
        model_facing_prompt_sha256=attempt.model_facing_prompt_sha256,
        tokenizer_fingerprint=attempt.tokenizer_fingerprint,
        top_k=attempt.top_k,
        min_p=attempt.min_p,
    )
    record.trajectory_id = record.expected_trajectory_id
    record.validate()
    return record


class TeacherDemoStateSource:
    def __init__(self, demonstrations: list[TeacherDemoAttempt]) -> None:
        if not demonstrations:
            raise ValueError("teacher demonstrations must be non-empty accepted attempts")
        for attempt in demonstrations:
            attempt.validate()
            if not attempt.accepted:
                raise ValueError("teacher-demo state source accepts only the accepted view")
        self.demonstrations = tuple(copy.deepcopy(demonstrations))
        self._by_prompt: dict[str, list[TeacherDemoAttempt]] = defaultdict(list)
        for attempt in self.demonstrations:
            self._by_prompt[attempt.prompt_id].append(attempt)
        self._cursor: dict[str, int] = defaultdict(int)

    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch:
        del model, step
        selected: list[TrajectoryRecord] = []
        for prompt_id in prompt_batch.prompt_ids:
            choices = self._by_prompt.get(prompt_id)
            if not choices:
                raise KeyError(
                    f"teacher-demo accepted view has no verified attempt for prompt {prompt_id}"
                )
            index = self._cursor[prompt_id] % len(choices)
            selected.append(_as_training_record(choices[index]))
            self._cursor[prompt_id] += 1
        return TrajectoryBatch(selected, policy_version=0)

    def refresh_if_needed(self, model: Any, step: int) -> None:
        del model, step

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "teacher_demo",
            "cursor_protocol_id": TEACHER_DEMO_CURSOR_PROTOCOL_ID,
            "cursor": dict(self._cursor),
            "attempt_ids": [attempt.attempt_id for attempt in self.demonstrations],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("cursor_protocol_id") != TEACHER_DEMO_CURSOR_PROTOCOL_ID:
            raise ValueError("teacher-demo checkpoint changed cursor protocol")
        expected = [attempt.attempt_id for attempt in self.demonstrations]
        if list(state["attempt_ids"]) != expected:
            raise ValueError("teacher-demo checkpoint does not match the accepted view")
        self._cursor = defaultdict(
            int,
            {str(key): int(value) for key, value in state["cursor"].items()},
        )
