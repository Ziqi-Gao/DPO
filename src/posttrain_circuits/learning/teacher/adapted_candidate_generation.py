"""Raw candidate measurements for an explicitly identified learned teacher.

This additive format does not pretend learned weights have an HF commit.
Sampling and verification use the existing teacher algorithm; independent
acceptance, population checks and storage publication happen separately.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.teacher_identity import TeacherIdentity
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.learning.teacher.demo_generation import HfTeacherCandidateGenerator
from posttrain_circuits.learning.teacher.seeding import (
    TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
    teacher_candidate_seed,
)
from posttrain_circuits.models.prompt_protocol import format_model_prompt


def _record(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _population(examples, rank, world_size):
    if (
        world_size not in (1, 4)
        or type(rank) is not int
        or not 0 <= rank < world_size
        or not examples
        or len({example.example_id for example in examples}) != len(examples)
    ):
        raise ValueError("invalid adapted teacher population or rank")


def _actual_output(output, tokenizer):
    output.validate()
    if (
        output.logprob_status != "available"
        or not output.response_ids
        or len(output.response_ids) > 256
        or output.token_logprobs is None
        or any(not math.isfinite(value) or value > 1e-5 for value in output.token_logprobs)
        or tokenizer.decode(output.response_ids, skip_special_tokens=True) != output.response_text
    ):
        raise ValueError("teacher measurement lacks genuine aligned response tokens/logprobs")


def _loaded_identity(loaded, identity):
    identity = TeacherIdentity.from_mapping(identity.to_mapping())
    if (
        identity.kind != "learned_dense_checkpoint"
        or loaded.teacher_checkpoint_sha256 != identity.teacher_checkpoint_sha256
        or loaded.tokenizer_hash != identity.tokenizer_fingerprint
        or loaded.prompt_protocol != identity.prompt_protocol
        or loaded.chat_template_sha256 != identity.chat_template_sha256
    ):
        raise ValueError("generator identity differs from verified loaded teacher")
    return identity


def measure_candidate_shard(
    examples: list[Any],
    loaded: Any,
    teacher: dict[str, Any],
    identity: TeacherIdentity,
    output: Path,
    *,
    rank: int,
    world_size: int,
    observe: Any,
) -> list[dict[str, Any]]:
    """Measure eight identity-seeded candidates per original ordered prompt.

    The caller must pin the population before execution. This function returns
    all genuine attempts, including verifier failures; it never retries them.
    """
    _population(examples, rank, world_size)
    identity = _loaded_identity(loaded, identity)
    output.mkdir(parents=True, exist_ok=False)
    generator = HfTeacherCandidateGenerator(
        loaded.model, loaded.tokenizer, max_new_tokens=256, model_config=teacher
    )
    task, attempts = ProofGraphTask(), []
    with (output / "attempts.jsonl").open("x") as stream:
        for index in range(rank, len(examples), world_size):
            example = examples[index]
            raw_prompt = task.render(example)
            formatted = format_model_prompt(raw_prompt, loaded.tokenizer, teacher)
            if (
                formatted.prompt_protocol != identity.prompt_protocol
                or formatted.chat_template_sha256 != identity.chat_template_sha256
                or formatted.enable_thinking is not False
            ):
                raise ValueError("candidate prompt protocol differs from learned teacher identity")
            input_ids = list(loaded.tokenizer.encode(formatted.model_facing_prompt, add_special_tokens=False))
            if not input_ids or len(input_ids) > 1246:
                raise ValueError("teacher demonstration exceeds original 1246-token prefix bound")
            prompt_identity = sha256_value(asdict(example))
            for candidate in range(8):
                seed = teacher_candidate_seed(31415, prompt_identity, candidate)
                observe.phase(
                    "adapted_teacher_candidate",
                    rank=rank,
                    index=index,
                    candidate_index=candidate,
                    example_id=example.example_id,
                )
                measured = generator(
                    example=example,
                    candidate_index=candidate,
                    actual_sampling_seed=seed,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                )
                _actual_output(measured, loaded.tokenizer)
                verified = task.verify(example, task.parse_response(measured.response_text))
                row = {
                    "attempt_id": f"{example.example_id}:candidate-{candidate:04d}",
                    "prompt_id": example.example_id,
                    "prompt_identity_sha256": prompt_identity,
                    "candidate_index": candidate,
                    "raw_prompt_text": raw_prompt,
                    "model_facing_prompt_text": formatted.model_facing_prompt,
                    "input_ids": input_ids,
                    "response_ids": list(measured.response_ids),
                    "response_text": measured.response_text,
                    "response_token_mask": [True] * len(measured.response_ids),
                    "behavior_logprobs": list(measured.token_logprobs),
                    "logprob_status": measured.logprob_status,
                    "finish_reason": measured.finish_reason,
                    "teacher_identity_sha256": identity.sha256,
                    "sampling_request_seed": 31415,
                    "actual_sampling_seed": seed,
                    "sampling_protocol_id": TEACHER_DEMO_SAMPLING_PROTOCOL_ID,
                    "sampling_temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "min_p": 0.0,
                    "verifier_reward": float(verified.reward),
                    "verification_trace": asdict(verified),
                    "accepted": verified.reward == 1.0,
                    "prompt_protocol": formatted.prompt_protocol,
                    "enable_thinking": formatted.enable_thinking,
                    "chat_template_sha256": formatted.chat_template_sha256,
                    "raw_prompt_sha256": formatted.raw_prompt_sha256,
                    "model_facing_prompt_sha256": formatted.model_facing_prompt_sha256,
                    "tokenizer_fingerprint": loaded.tokenizer_hash,
                }
                _record(stream, row)
                attempts.append(row)
    return attempts


def measure_readiness_shard(
    examples: list[Any],
    specifications: list[Any],
    loaded: Any,
    teacher: dict[str, Any],
    identity: TeacherIdentity,
    output: Path,
    *,
    source_index_start: int,
    rank: int,
    world_size: int,
    observe: Any,
) -> dict[str, Any]:
    """Original greedy/prefix measurements with an explicit source-row seed.

    Supplemental rows 128:256 keep seeds 42+128 through 42+255. They are never
    silently renumbered to the original readiness population's seeds.
    """
    from posttrain_circuits.cli.evaluate_teacher_readiness import _prefix_scores

    _population(examples, rank, world_size)
    _loaded_identity(loaded, identity)
    if type(source_index_start) is not int or source_index_start not in (0, 128):
        raise ValueError("unreviewed readiness source index")
    if len(examples) != 128:
        raise ValueError("formal and supplemental readiness each require exactly 128 rows")
    output.mkdir(parents=True, exist_ok=False)
    generator = HfTeacherCandidateGenerator(
        loaded.model, loaded.tokenizer, max_new_tokens=256, model_config=teacher
    )
    generations, prefix_scores = [], []
    with (output / "generations.jsonl").open("x") as stream:
        for index in range(rank, len(examples), world_size):
            example = examples[index]
            formatted = format_model_prompt(ProofGraphTask().render(example), loaded.tokenizer, teacher)
            if (
                formatted.prompt_protocol != loaded.prompt_protocol
                or formatted.chat_template_sha256 != loaded.chat_template_sha256
                or formatted.enable_thinking is not False
                or len(loaded.tokenizer.encode(formatted.model_facing_prompt, add_special_tokens=False))
                > 1246
            ):
                raise ValueError(
                    "readiness prompt differs from the verified teacher protocol or prefix bound"
                )
            seed = 42 + source_index_start + index
            observe.phase("adapted_teacher_readiness", rank=rank, index=index, example_id=example.example_id)
            result = generator(
                example=example,
                candidate_index=0,
                actual_sampling_seed=seed,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
            )
            _actual_output(result, loaded.tokenizer)
            row = {
                "example_id": example.example_id,
                "global_index": source_index_start + index,
                "actual_sampling_seed": seed,
                **asdict(result),
            }
            _record(stream, row)
            generations.append(row)
    selected = [s for s in specifications if s.stage in {"first_rule_selection", "intermediate_conclusion"}]
    with (output / "prefix-scores.jsonl").open("x") as stream:
        for index in range(rank, len(selected), world_size):
            observe.phase("adapted_teacher_readiness_prefix", rank=rank, index=index)
            pair = _prefix_scores(loaded.model, selected[index], top_k=128)
            if len(pair) != 2:
                raise ValueError("readiness scorer omitted a counterfactual side")
            for side, value in enumerate(pair):
                row = {"global_index": index, "side": side, "score": asdict(value)}
                _record(stream, row)
                prefix_scores.append(row)
    return {
        "rank": rank,
        "world_size": world_size,
        "generations": generations,
        "prefix_scores": prefix_scores,
    }
