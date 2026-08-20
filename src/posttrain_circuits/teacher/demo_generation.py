"""Exact-verifier-gated teacher demonstration generation and storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from transformers import PreTrainedTokenizerBase

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.types import TrajectoryRecord
from posttrain_circuits.data.trajectory_store import TrajectoryStore
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.models.prompt_protocol import format_model_prompt
from posttrain_circuits.rollout.generation import generation_rng
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample

VERIFIER_VERSION = "proofgraph-exact-v1"


class HfTeacherCandidateGenerator:
    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
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
        generation_seed: int,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> str:
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
        with generation_rng(generation_seed, device), torch.no_grad():
            generated = self.model.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        response_ids = generated[0, encoded.input_ids.shape[1] :]
        return self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )


class TeacherCandidateGenerator(Protocol):
    def __call__(
        self,
        *,
        example: TaskExample,
        candidate_index: int,
        generation_seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ) -> str: ...


@dataclass(frozen=True)
class TeacherDemoGenerationConfig:
    teacher_id: str
    teacher_revision: str
    resolved_teacher_commit: str
    generation_seed: int
    temperature: float
    top_p: float
    candidates_per_prompt: int
    top_k: int = 0
    min_p: float = 0.0
    verifier_version: str = VERIFIER_VERSION

    def __post_init__(self) -> None:
        if not self.teacher_id or not self.teacher_revision or not self.resolved_teacher_commit:
            raise ValueError("teacher ID, requested revision, and resolved commit must be non-empty")
        if self.candidates_per_prompt < 1:
            raise ValueError("candidates_per_prompt must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0 or not 0.0 <= self.min_p <= 1.0:
            raise ValueError("top_k/min_p are outside their valid ranges")


@dataclass(frozen=True)
class TeacherDemoGenerationResult:
    records: list[TrajectoryRecord]
    total_candidates: int
    successful_candidates: int
    prompts_with_success: int
    total_prompts: int
    prompt_manifest_hash: str
    tokenizer_hash: str
    config: TeacherDemoGenerationConfig

    @property
    def retention_rate(self) -> float:
        return self.successful_candidates / self.total_candidates


def generate_teacher_demonstrations(
    examples: list[TaskExample],
    tokenizer: PreTrainedTokenizerBase,
    candidate_generator: TeacherCandidateGenerator,
    config: TeacherDemoGenerationConfig,
    *,
    model_config: dict[str, Any] | None = None,
) -> TeacherDemoGenerationResult:
    """Generate candidates from a teacher and retain only exact verifier successes."""
    if not examples:
        raise ValueError("teacher demonstration generation requires prompts")
    task = ProofGraphTask()
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    retained: list[TrajectoryRecord] = []
    prompts_with_success = 0
    for prompt_index, example in enumerate(examples):
        prompt_successes = 0
        raw_prompt = task.render(example)
        formatted = format_model_prompt(raw_prompt, tokenizer, model_config)
        prompt_text = formatted.model_facing_prompt
        input_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
        for candidate_index in range(config.candidates_per_prompt):
            candidate_seed = (
                config.generation_seed + prompt_index * config.candidates_per_prompt + candidate_index
            )
            response_text = candidate_generator(
                example=example,
                candidate_index=candidate_index,
                generation_seed=candidate_seed,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
            )
            verification = task.verify(example, task.parse_response(response_text))
            if verification.reward != 1.0:
                continue
            response_ids = list(tokenizer.encode(response_text, add_special_tokens=False))
            if tokenizer.eos_token_id is not None:
                response_ids.append(int(tokenizer.eos_token_id))
            record = TrajectoryRecord(
                trajectory_id=(
                    "teacher-demo-"
                    + sha256_value(
                        [
                            example.example_id,
                            candidate_index,
                            candidate_seed,
                            response_text,
                            config.resolved_teacher_commit,
                        ]
                    )[:20]
                ),
                prompt_id=example.example_id,
                split="train",
                prompt_text=prompt_text,
                input_ids=input_ids,
                response_ids=response_ids,
                response_text=response_text,
                response_token_mask=[True] * len(response_ids),
                behavior_policy_id=config.teacher_id,
                behavior_policy_revision=config.teacher_revision,
                policy_version=0,
                generation_seed=candidate_seed,
                sampling_temperature=config.temperature,
                top_p=config.top_p,
                behavior_logprobs=[0.0] * len(response_ids),
                verifier_reward=1.0,
                verification_trace=asdict(verification),
                teacher_id=config.teacher_id,
                teacher_revision=config.teacher_revision,
                created_at=datetime.now(UTC).isoformat(),
                raw_prompt_text=raw_prompt,
                prompt_protocol=formatted.prompt_protocol,
                enable_thinking=formatted.enable_thinking,
                chat_template_sha256=formatted.chat_template_sha256,
                raw_prompt_sha256=formatted.raw_prompt_sha256,
                model_facing_prompt_sha256=formatted.model_facing_prompt_sha256,
                tokenizer_fingerprint=tokenizer_hash,
                top_k=config.top_k,
                min_p=config.min_p,
            )
            record.validate()
            retained.append(record)
            prompt_successes += 1
        prompts_with_success += int(prompt_successes > 0)

    if not retained:
        raise ValueError("teacher produced no exact-verifier-success demonstrations")
    prompt_manifest_hash = sha256_value([asdict(example) for example in examples])
    total_candidates = len(examples) * config.candidates_per_prompt
    return TeacherDemoGenerationResult(
        records=retained,
        total_candidates=total_candidates,
        successful_candidates=len(retained),
        prompts_with_success=prompts_with_success,
        total_prompts=len(examples),
        prompt_manifest_hash=prompt_manifest_hash,
        tokenizer_hash=tokenizer_hash,
        config=config,
    )


def write_teacher_demo_store(
    root: Path,
    result: TeacherDemoGenerationResult,
) -> dict[str, object]:
    """Write an independent immutable teacher-demo store with complete generation provenance."""
    config = result.config
    return TrajectoryStore(root).write(
        result.records,
        behavior_policy={
            "id": config.teacher_id,
            "revision": config.teacher_revision,
            "resolved_commit": config.resolved_teacher_commit,
        },
        prompt_manifest_hash=result.prompt_manifest_hash,
        sampling_configuration={
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "min_p": config.min_p,
            "candidates_per_prompt": config.candidates_per_prompt,
            "generation_seed": config.generation_seed,
        },
        verifier_version=config.verifier_version,
        teacher_version=config.teacher_revision,
        top_k=0,
        extra_metadata={
            "store_kind": "teacher_demo",
            "tokenizer_hash": result.tokenizer_hash,
            "tokenizer_fingerprint": result.tokenizer_hash,
            "protocol_track": (
                "qwen3_v1" if result.records[0].prompt_protocol == "qwen3_non_thinking_v1" else "core_v2"
            ),
            "artifact_namespace": (
                "qwen3-v1" if result.records[0].prompt_protocol == "qwen3_non_thinking_v1" else "legacy"
            ),
            "prompt_protocol": result.records[0].prompt_protocol,
            "enable_thinking": result.records[0].enable_thinking,
            "chat_template_sha256": result.records[0].chat_template_sha256,
            "teacher_demo_generation": {
                "teacher_id": config.teacher_id,
                "teacher_revision": config.teacher_revision,
                "resolved_teacher_commit": config.resolved_teacher_commit,
                "generation_seed": config.generation_seed,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "min_p": config.min_p,
                "candidates_per_prompt": config.candidates_per_prompt,
                "verifier_version": config.verifier_version,
                "retention_rate": result.retention_rate,
                "candidates_generated": result.total_candidates,
                "successful_candidates": result.successful_candidates,
                "prompts_with_success": result.prompts_with_success,
                "total_prompts": result.total_prompts,
                "prompt_manifest_hash": result.prompt_manifest_hash,
            },
        },
    )


def read_teacher_demo_store(root: Path) -> tuple[list[TrajectoryRecord], dict[str, object]]:
    store = TrajectoryStore(root)
    manifest = store.check_integrity()
    if manifest.get("store_kind") != "teacher_demo":
        raise ValueError(f"{root} is not an independent teacher-demo store")
    records = store.read()
    if not records or any(record.verifier_reward != 1.0 for record in records):
        raise ValueError("teacher-demo store contains non-successful trajectories")
    if any(not record.teacher_id or not record.teacher_revision for record in records):
        raise ValueError("teacher-demo store is missing teacher provenance")
    return records, manifest
