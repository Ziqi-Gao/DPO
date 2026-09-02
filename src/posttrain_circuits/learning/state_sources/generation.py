"""Production Hugging Face current-policy generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import torch

from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord
from posttrain_circuits.datasets.trajectories.identities import generation_group_id
from posttrain_circuits.learning.contracts import PromptBatch, SamplingRequest
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.models.prompt_protocol import (
    LEGACY_PROMPT_PROTOCOL,
    format_model_prompts,
)

HF_SAMPLING_PROTOCOL_ID = "hf-generate-v3-stable-cursor-seed-eos"


@contextmanager
def generation_rng(seed: int, device: torch.device) -> Iterator[None]:
    """Isolate HF generation from, and restore, the caller's RNG streams."""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(seed)
        yield


@contextmanager
def _temporary_left_padding(tokenizer: Any) -> Iterator[None]:
    """Left-pad decoder-only generation without persistently mutating the tokenizer."""

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = original_padding_side


def _effective_prompt_ids(encoded: Any, row: int) -> list[int]:
    mask = encoded.attention_mask[row].to(dtype=torch.bool)
    return [int(value) for value in encoded.input_ids[row][mask].detach().cpu().tolist()]


def _trim_generated_row(
    row: int,
    generated_ids: torch.Tensor,
    scores: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    eos_token_ids: set[int],
    pad_token_id: int | None,
) -> tuple[list[int], list[float]]:
    response_ids: list[int] = []
    logprobs: list[float] = []
    for position, score in enumerate(scores):
        token = int(generated_ids[row, position])
        if pad_token_id is not None and token == pad_token_id and token not in eos_token_ids:
            break
        response_ids.append(token)
        logprobs.append(float(score[row].float().log_softmax(-1)[token].detach().cpu()))
        if token in eos_token_ids:
            break
    return response_ids, logprobs


def _hf_generate_trajectories_left_padded(
    model: Any,
    tokenizer: Any,
    prompt_batch: PromptBatch,
    policy_version: int,
    sampling_request: SamplingRequest,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    policy_id: str,
    policy_revision: str,
    top_k: int = 0,
    min_p: float = 0.0,
    model_config: dict[str, Any] | None = None,
) -> list[TrajectoryRecord]:
    sampling_request.validate_for(prompt_batch)
    device = next(model.parameters()).device
    formatted_prompts = format_model_prompts(
        prompt_batch.prompt_texts,
        tokenizer,
        model_config,
    )
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    model_facing_texts = [prompt.model_facing_prompt for prompt in formatted_prompts]
    add_special_tokens = formatted_prompts[0].prompt_protocol == LEGACY_PROMPT_PROTOCOL
    encoded = tokenizer(
        model_facing_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=add_special_tokens,
    ).to(device)
    individual_prompt_ids = [
        [int(value) for value in tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]]
        for text in model_facing_texts
    ]
    for row, expected in enumerate(individual_prompt_ids):
        batched = _effective_prompt_ids(encoded, row)
        if batched != expected:
            raise RuntimeError(
                f"batched tokenizer changed prompt token bytes: prompt_id={prompt_batch.prompt_ids[row]}"
            )
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": True,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
    records: list[TrajectoryRecord] = []
    prompt_counts = Counter(prompt_batch.prompt_ids)
    for prompt_id, count in prompt_counts.items():
        indices = sorted(
            cursor.generation_index
            for cursor in sampling_request.cursors
            if cursor.prompt_id == prompt_id
        )
        if indices != list(range(count)):
            raise ValueError(
                f"sampling cursors for prompt {prompt_id!r} must cover generation indices 0..{count - 1}"
            )
    raw_eos = getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id)
    if raw_eos is None:
        eos_token_ids: set[int] = set()
    elif isinstance(raw_eos, int):
        eos_token_ids = {raw_eos}
    else:
        eos_token_ids = {int(value) for value in raw_eos}
    generated_rows: list[tuple[list[int], list[float]]] = []
    if temperature > 0:
        for row, (prompt_id, prompt_text, cursor) in enumerate(
            zip(
                prompt_batch.prompt_ids,
                model_facing_texts,
                sampling_request.cursors,
                strict=True,
            )
        ):
            single_encoded = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=False,
                add_special_tokens=add_special_tokens,
            ).to(device)
            if _effective_prompt_ids(single_encoded, 0) != individual_prompt_ids[row]:
                raise RuntimeError(f"singleton tokenizer changed prompt token bytes: prompt_id={prompt_id}")
            row_seed = sampling_request.seed_for(row, policy_version=policy_version)
            with generation_rng(row_seed, device), torch.no_grad():
                row_sequences = model.generate(**single_encoded, **generation_kwargs)
            row_generated_ids = row_sequences.sequences[:, single_encoded.input_ids.shape[1] :]
            generated_rows.append(
                _trim_generated_row(
                    0,
                    row_generated_ids,
                    row_sequences.scores,
                    eos_token_ids=eos_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
            )
    else:
        with generation_rng(sampling_request.sampling_request_seed, device), torch.no_grad():
            sequences = model.generate(**encoded, **generation_kwargs)
        prompt_width = encoded.input_ids.shape[1]
        generated_ids = sequences.sequences[:, prompt_width:]
        generated_rows = [
            _trim_generated_row(
                row,
                generated_ids,
                sequences.scores,
                eos_token_ids=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
            for row in range(len(prompt_batch.prompt_ids))
        ]
    for row, prompt_id in enumerate(prompt_batch.prompt_ids):
        response_ids, logprobs = generated_rows[row]
        if not (len(response_ids) == len(logprobs)):
            raise RuntimeError("generated response IDs and behavior logprobs are misaligned")
        group_size = int(prompt_counts[prompt_id])
        cursor = sampling_request.cursors[row]
        group_index = cursor.generation_index
        group_id = generation_group_id(
            prompt_id=prompt_id,
            policy_version=policy_version,
            sampling_request_seed=sampling_request.sampling_request_seed,
            optimizer_step=cursor.optimizer_step,
            retry_index=cursor.retry_index,
            prompt_group_size=group_size,
            sampling_protocol_id=sampling_request.sampling_protocol_id,
        )
        formatted = formatted_prompts[row]
        record = TrajectoryRecord(
            trajectory_id="",
            prompt_id=prompt_id,
            split="train",
            prompt_text=formatted.model_facing_prompt,
            input_ids=individual_prompt_ids[row],
            response_ids=response_ids,
            response_text=tokenizer.decode(response_ids, skip_special_tokens=True),
            response_token_mask=[True] * len(response_ids),
            behavior_policy_id=policy_id,
            behavior_policy_revision=policy_revision,
            policy_version=policy_version,
            sampling_request_seed=sampling_request.sampling_request_seed,
            actual_sampling_seed=sampling_request.seed_for(row, policy_version=policy_version),
            sampling_cursor_id=cursor.cursor_id,
            sampling_protocol_id=sampling_request.sampling_protocol_id,
            sampling_temperature=temperature,
            top_p=top_p,
            behavior_logprobs=logprobs,
            generation_group_id=group_id,
            generation_group_index=group_index,
            prompt_group_size=group_size,
            created_at=datetime.now(UTC).isoformat(),
            raw_prompt_text=formatted.raw_prompt,
            prompt_protocol=formatted.prompt_protocol,
            enable_thinking=formatted.enable_thinking,
            chat_template_sha256=formatted.chat_template_sha256,
            raw_prompt_sha256=formatted.raw_prompt_sha256,
            model_facing_prompt_sha256=formatted.model_facing_prompt_sha256,
            tokenizer_fingerprint=tokenizer_hash,
            top_k=top_k,
            min_p=min_p,
        )
        record.trajectory_id = record.expected_trajectory_id
        record.validate()
        records.append(record)
    return records


def hf_generate_trajectories(
    model: Any,
    tokenizer: Any,
    prompt_batch: PromptBatch,
    policy_version: int,
    sampling_request: SamplingRequest,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    policy_id: str,
    policy_revision: str,
    top_k: int = 0,
    min_p: float = 0.0,
    model_config: dict[str, Any] | None = None,
) -> list[TrajectoryRecord]:
    with _temporary_left_padding(tokenizer):
        return _hf_generate_trajectories_left_padded(
            model,
            tokenizer,
            prompt_batch,
            policy_version,
            sampling_request,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            policy_id=policy_id,
            policy_revision=policy_revision,
            top_k=top_k,
            min_p=min_p,
            model_config=model_config,
        )
def build_proofgraph_hf_generator(
    *,
    tokenizer: Any,
    examples_by_id: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    policy_id: str,
    initial_policy_revision: str,
    top_k: int = 0,
    min_p: float = 0.0,
    model_config: dict[str, Any] | None = None,
):
    """Bind HF generation to exact ProofGraph verification for online training."""
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask

    task = ProofGraphTask()

    def generate(
        model: Any,
        prompt_batch: PromptBatch,
        policy_version: int,
        sampling_request: SamplingRequest,
    ) -> list[TrajectoryRecord]:
        records = hf_generate_trajectories(
            model,
            tokenizer,
            prompt_batch,
            policy_version,
            sampling_request,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            policy_id=policy_id,
            policy_revision=(f"{initial_policy_revision}@policy-{policy_version}"),
            top_k=top_k,
            min_p=min_p,
            model_config=model_config,
        )
        for record in records:
            example = examples_by_id[record.prompt_id]
            verification = task.verify(
                example,
                task.parse_response(record.response_text),
            )
            record.verifier_reward = verification.reward
            record.verification_trace = asdict(verification)
        return records

    return generate
