"""Pure contracts for complete teacher-demo generation ledgers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

LOGPROB_AVAILABLE = "available"
LOGPROB_FIXTURE_UNAVAILABLE = "fixture_unavailable"
LOGPROB_STATUSES = {LOGPROB_AVAILABLE, LOGPROB_FIXTURE_UNAVAILABLE}
ATTEMPT_FIELDS = {
    "attempt_id",
    "prompt_id",
    "prompt_identity_sha256",
    "candidate_index",
    "raw_prompt_text",
    "model_facing_prompt_text",
    "input_ids",
    "response_ids",
    "response_text",
    "response_token_mask",
    "behavior_logprobs",
    "logprob_status",
    "finish_reason",
    "teacher_id",
    "teacher_revision",
    "sampling_request_seed",
    "actual_sampling_seed",
    "sampling_protocol_id",
    "sampling_temperature",
    "top_p",
    "top_k",
    "min_p",
    "verifier_reward",
    "verification_trace",
    "accepted",
    "prompt_protocol",
    "enable_thinking",
    "chat_template_sha256",
    "raw_prompt_sha256",
    "model_facing_prompt_sha256",
    "tokenizer_fingerprint",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class TeacherCandidateOutput:
    """One generator result before verifier evaluation."""

    response_text: str
    response_ids: list[int] | None
    token_logprobs: list[float] | None
    logprob_status: str
    finish_reason: str

    def validate(self) -> None:
        if self.logprob_status not in LOGPROB_STATUSES:
            raise ValueError("teacher candidate has an unknown logprob status")
        if not self.finish_reason.strip():
            raise ValueError("teacher candidate finish_reason is empty")
        if self.response_ids is not None and any(
            type(token_id) is not int or token_id < 0 for token_id in self.response_ids
        ):
            raise ValueError("teacher candidate response IDs are invalid")
        if self.logprob_status == LOGPROB_AVAILABLE:
            if self.response_ids is None or self.token_logprobs is None:
                raise ValueError("available teacher logprobs require actual response IDs and values")
            if len(self.response_ids) != len(self.token_logprobs):
                raise ValueError("teacher candidate IDs/logprobs are not token aligned")
            if any(type(value) not in {int, float} or not math.isfinite(value) for value in self.token_logprobs):
                raise ValueError("teacher candidate logprobs must be finite")
        elif self.token_logprobs is not None:
            raise ValueError("fixture-unavailable teacher candidates cannot carry pseudo-logprobs")


@dataclass(frozen=True)
class TeacherDemoAttempt:
    """Every teacher candidate, including verifier failures."""

    attempt_id: str
    prompt_id: str
    prompt_identity_sha256: str
    candidate_index: int
    raw_prompt_text: str
    model_facing_prompt_text: str
    input_ids: list[int]
    response_ids: list[int]
    response_text: str
    response_token_mask: list[bool]
    behavior_logprobs: list[float] | None
    logprob_status: str
    finish_reason: str
    teacher_id: str
    teacher_revision: str
    sampling_request_seed: int
    actual_sampling_seed: int
    sampling_protocol_id: str
    sampling_temperature: float
    top_p: float
    top_k: int
    min_p: float
    verifier_reward: float
    verification_trace: dict[str, Any]
    accepted: bool
    prompt_protocol: str
    enable_thinking: bool
    chat_template_sha256: str
    raw_prompt_sha256: str
    model_facing_prompt_sha256: str
    tokenizer_fingerprint: str

    def validate(self) -> None:
        required_text = {
            "attempt_id": self.attempt_id,
            "prompt_id": self.prompt_id,
            "teacher_id": self.teacher_id,
            "teacher_revision": self.teacher_revision,
            "sampling_protocol_id": self.sampling_protocol_id,
            "logprob_status": self.logprob_status,
            "finish_reason": self.finish_reason,
            "prompt_protocol": self.prompt_protocol,
        }
        if any(not isinstance(value, str) or not value.strip() for value in required_text.values()):
            raise ValueError("teacher-demo attempt has an empty identity field")
        if type(self.candidate_index) is not int or self.candidate_index < 0:
            raise ValueError("teacher-demo candidate_index must be non-negative")
        if self.attempt_id != f"{self.prompt_id}:candidate-{self.candidate_index:04d}":
            raise ValueError("teacher-demo attempt ID differs from prompt/candidate identity")
        if type(self.sampling_request_seed) is not int or type(self.actual_sampling_seed) is not int:
            raise ValueError("teacher-demo sampling seeds must be integers")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("teacher-demo top_k is invalid")
        if (
            type(self.sampling_temperature) not in {int, float}
            or type(self.top_p) not in {int, float}
            or type(self.min_p) not in {int, float}
            or not all(
                math.isfinite(float(value))
                for value in (self.sampling_temperature, self.top_p, self.min_p)
            )
            or self.sampling_temperature < 0
            or not 0 < self.top_p <= 1
            or not 0 <= self.min_p <= 1
        ):
            raise ValueError("teacher-demo sampling parameters are invalid")
        for name, value in (
            ("prompt_identity_sha256", self.prompt_identity_sha256),
            ("chat_template_sha256", self.chat_template_sha256),
            ("raw_prompt_sha256", self.raw_prompt_sha256),
            ("model_facing_prompt_sha256", self.model_facing_prompt_sha256),
            ("tokenizer_fingerprint", self.tokenizer_fingerprint),
        ):
            if not _is_sha256(value):
                raise ValueError(f"teacher-demo {name} is not a SHA-256")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.input_ids):
            raise ValueError("teacher-demo prompt token IDs are invalid")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.response_ids):
            raise ValueError("teacher-demo response token IDs are invalid")
        if (
            len(self.response_token_mask) != len(self.response_ids)
            or any(value is not True for value in self.response_token_mask)
        ):
            raise ValueError("teacher-demo response token mask is invalid")
        if self.logprob_status == LOGPROB_AVAILABLE:
            if self.behavior_logprobs is None or len(self.behavior_logprobs) != len(
                self.response_ids
            ):
                raise ValueError("teacher-demo logprobs are not aligned to response IDs")
            if any(
                type(value) not in {int, float} or not math.isfinite(float(value))
                for value in self.behavior_logprobs
            ):
                raise ValueError("teacher-demo behavior logprobs must be finite")
        elif self.logprob_status == LOGPROB_FIXTURE_UNAVAILABLE:
            if self.behavior_logprobs is not None:
                raise ValueError("fixture-unavailable attempts cannot contain pseudo-logprobs")
        else:
            raise ValueError("teacher-demo attempt has an unknown logprob status")
        if type(self.verifier_reward) not in {int, float} or not math.isfinite(
            float(self.verifier_reward)
        ):
            raise ValueError("teacher-demo verifier reward is invalid")
        if type(self.accepted) is not bool or self.accepted != (self.verifier_reward == 1.0):
            raise ValueError("teacher-demo acceptance differs from exact verifier success")
        if not isinstance(self.verification_trace, dict):
            raise ValueError("teacher-demo verification trace must be a mapping")
        if type(self.enable_thinking) is not bool:
            raise ValueError("teacher-demo enable_thinking must be boolean")


def attempt_to_payload(attempt: TeacherDemoAttempt) -> dict[str, Any]:
    attempt.validate()
    return asdict(attempt)


def attempt_from_payload(payload: dict[str, Any]) -> TeacherDemoAttempt:
    if not isinstance(payload, dict) or set(payload) != ATTEMPT_FIELDS:
        raise ValueError("teacher-demo attempt fields differ from the ledger contract")
    attempt = TeacherDemoAttempt(**payload)
    attempt.validate()
    return attempt
