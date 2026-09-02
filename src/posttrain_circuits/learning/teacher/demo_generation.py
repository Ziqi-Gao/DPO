"""Teacher candidate generation with complete, verifier-gated attempt provenance."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Protocol

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.teacher_demos.contracts import (
    LOGPROB_FIXTURE_UNAVAILABLE,
    TeacherCandidateOutput,
    TeacherDemoAttempt,
)
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.models.prompt_protocol import format_model_prompt
from posttrain_circuits.learning.teacher.seeding import (
    TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
    teacher_candidate_seed,
)

VERIFIER_VERSION = "proofgraph-exact-v1"


@contextmanager
def _generation_rng(seed: int, device: Any) -> Iterator[None]:
    import torch

    cuda_devices: list[int] = []
    if getattr(device, "type", None) == "cuda":
        device_index = device.index
        cuda_devices = [
            int(torch.cuda.current_device() if device_index is None else device_index)
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        yield


class HfTeacherCandidateGenerator:
    """Generate once and preserve the model's actual response IDs and token logprobs."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 256,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("teacher generation length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.task = ProofGraphTask()
        self.max_new_tokens = max_new_tokens
        self.model_config = model_config

    def __call__(
        self,
        *,
        example: TaskExample,
        candidate_index: int,
        actual_sampling_seed: int,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> TeacherCandidateOutput:
        del candidate_index
        import torch

        device = next(self.model.parameters()).device
        formatted = format_model_prompt(
            self.task.render(example),
            self.tokenizer,
            self.model_config,
        )
        encoded = self.tokenizer(
            formatted.model_facing_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        with _generation_rng(actual_sampling_seed, device), torch.no_grad():
            generated = self.model.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        prompt_length = int(encoded.input_ids.shape[1])
        response_ids = [int(value) for value in generated.sequences[0, prompt_length:].tolist()]
        scores = list(generated.scores)
        if len(scores) != len(response_ids):
            raise RuntimeError("teacher generation scores are not aligned to generated token IDs")
        token_logprobs = [
            float(score[0].float().log_softmax(dim=-1)[token_id].detach().cpu())
            for score, token_id in zip(scores, response_ids, strict=True)
        ]
        eos_token_id = self.tokenizer.eos_token_id
        finish_reason = (
            "eos"
            if response_ids and eos_token_id is not None and response_ids[-1] == int(eos_token_id)
            else "length"
            if len(response_ids) == self.max_new_tokens
            else "stopped"
        )
        output = TeacherCandidateOutput(
            response_text=self.tokenizer.decode(response_ids, skip_special_tokens=True),
            response_ids=response_ids,
            token_logprobs=token_logprobs,
            logprob_status="available",
            finish_reason=finish_reason,
        )
        output.validate()
        return output


class TeacherCandidateGenerator(Protocol):
    def __call__(
        self,
        *,
        example: TaskExample,
        candidate_index: int,
        actual_sampling_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ) -> TeacherCandidateOutput: ...


@dataclass(frozen=True)
class TeacherDemoGenerationConfig:
    teacher_id: str
    teacher_revision: str
    resolved_teacher_commit: str
    sampling_request_seed: int
    temperature: float
    top_p: float
    candidates_per_prompt: int
    top_k: int = 0
    min_p: float = 0.0
    verifier_version: str = VERIFIER_VERSION

    def __post_init__(self) -> None:
        if not self.teacher_id or not self.teacher_revision or not self.resolved_teacher_commit:
            raise ValueError("teacher ID, requested revision, and resolved commit must be non-empty")
        if type(self.sampling_request_seed) is not int:
            raise ValueError("sampling_request_seed must be an integer")
        if type(self.candidates_per_prompt) is not int or self.candidates_per_prompt < 1:
            raise ValueError("candidates_per_prompt must be positive")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("teacher sampling temperature/top_p are invalid")
        if self.top_k < 0 or not 0.0 <= self.min_p <= 1.0:
            raise ValueError("top_k/min_p are outside their valid ranges")


@dataclass(frozen=True)
class TeacherDemoGenerationResult:
    attempts: list[TeacherDemoAttempt]
    ordered_prompt_ids: list[str]
    zero_success_prompt_ids: list[str]
    total_candidates: int
    successful_candidates: int
    prompts_with_success: int
    total_prompts: int
    prompt_manifest_hash: str
    tokenizer_hash: str
    config: TeacherDemoGenerationConfig

    @property
    def accepted_attempts(self) -> list[TeacherDemoAttempt]:
        return [attempt for attempt in self.attempts if attempt.accepted]

    @property
    def retention_rate(self) -> float:
        return self.successful_candidates / self.total_candidates


def generate_teacher_demonstrations(
    examples: list[TaskExample],
    tokenizer: Any,
    candidate_generator: TeacherCandidateGenerator,
    config: TeacherDemoGenerationConfig,
    *,
    model_config: dict[str, Any] | None = None,
) -> TeacherDemoGenerationResult:
    """Record every candidate attempt; selection is represented only by the accepted view."""
    if not examples:
        raise ValueError("teacher demonstration generation requires prompts")
    ordered_prompt_ids = [example.example_id for example in examples]
    if len(ordered_prompt_ids) != len(set(ordered_prompt_ids)):
        raise ValueError("teacher demonstration prompts contain duplicate identities")
    task = ProofGraphTask()
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    attempts: list[TeacherDemoAttempt] = []
    success_prompt_ids: set[str] = set()
    for example in examples:
        raw_prompt = task.render(example)
        formatted = format_model_prompt(raw_prompt, tokenizer, model_config)
        input_ids = list(
            tokenizer.encode(formatted.model_facing_prompt, add_special_tokens=False)
        )
        prompt_identity_sha256 = sha256_value(asdict(example))
        for candidate_index in range(config.candidates_per_prompt):
            candidate_seed = teacher_candidate_seed(
                config.sampling_request_seed,
                prompt_identity_sha256,
                candidate_index,
            )
            output = candidate_generator(
                example=example,
                candidate_index=candidate_index,
                actual_sampling_seed=candidate_seed,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
            )
            output.validate()
            if output.response_ids is None:
                if output.logprob_status != LOGPROB_FIXTURE_UNAVAILABLE:
                    raise ValueError("only an explicit fixture may omit actual response IDs")
                response_ids = list(
                    tokenizer.encode(output.response_text, add_special_tokens=False)
                )
            else:
                response_ids = list(output.response_ids)
            verification = task.verify(example, task.parse_response(output.response_text))
            accepted = verification.reward == 1.0
            if accepted:
                success_prompt_ids.add(example.example_id)
            attempt = TeacherDemoAttempt(
                attempt_id=f"{example.example_id}:candidate-{candidate_index:04d}",
                prompt_id=example.example_id,
                prompt_identity_sha256=prompt_identity_sha256,
                candidate_index=candidate_index,
                raw_prompt_text=raw_prompt,
                model_facing_prompt_text=formatted.model_facing_prompt,
                input_ids=input_ids,
                response_ids=response_ids,
                response_text=output.response_text,
                response_token_mask=[True] * len(response_ids),
                behavior_logprobs=(
                    list(output.token_logprobs)
                    if output.token_logprobs is not None
                    else None
                ),
                logprob_status=output.logprob_status,
                finish_reason=output.finish_reason,
                teacher_id=config.teacher_id,
                teacher_revision=config.teacher_revision,
                sampling_request_seed=config.sampling_request_seed,
                actual_sampling_seed=candidate_seed,
                sampling_protocol_id=TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
                sampling_temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
                verifier_reward=float(verification.reward),
                verification_trace=asdict(verification),
                accepted=accepted,
                prompt_protocol=formatted.prompt_protocol,
                enable_thinking=formatted.enable_thinking,
                chat_template_sha256=formatted.chat_template_sha256,
                raw_prompt_sha256=formatted.raw_prompt_sha256,
                model_facing_prompt_sha256=formatted.model_facing_prompt_sha256,
                tokenizer_fingerprint=tokenizer_hash,
            )
            attempt.validate()
            attempts.append(attempt)
    zero_success_prompt_ids = [
        prompt_id for prompt_id in ordered_prompt_ids if prompt_id not in success_prompt_ids
    ]
    successful_candidates = sum(attempt.accepted for attempt in attempts)
    return TeacherDemoGenerationResult(
        attempts=attempts,
        ordered_prompt_ids=ordered_prompt_ids,
        zero_success_prompt_ids=zero_success_prompt_ids,
        total_candidates=len(attempts),
        successful_candidates=successful_candidates,
        prompts_with_success=len(success_prompt_ids),
        total_prompts=len(examples),
        prompt_manifest_hash=sha256_value([asdict(example) for example in examples]),
        tokenizer_hash=tokenizer_hash,
        config=config,
    )
