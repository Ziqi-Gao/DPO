"""Production Hugging Face current-policy generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import torch

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.types import PromptBatch, TrajectoryRecord


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


def hf_generate_trajectories(
    model: Any,
    tokenizer: Any,
    prompt_batch: PromptBatch,
    policy_version: int,
    seed: int,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    policy_id: str,
    policy_revision: str,
) -> list[TrajectoryRecord]:
    device = next(model.parameters()).device
    encoded = tokenizer(
        list(prompt_batch.prompt_texts), return_tensors="pt", padding=True, add_special_tokens=True
    ).to(device)
    individual_prompt_ids = [
        [int(value) for value in tokenizer(text, add_special_tokens=True)["input_ids"]]
        for text in prompt_batch.prompt_texts
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
        generation_kwargs.update(temperature=temperature, top_p=top_p)
    with generation_rng(seed, device), torch.no_grad():
        sequences = model.generate(**encoded, **generation_kwargs)
    records: list[TrajectoryRecord] = []
    prompt_counts = Counter(prompt_batch.prompt_ids)
    prompt_indices: dict[str, int] = defaultdict(int)
    prompt_width = encoded.input_ids.shape[1]
    generated_ids = sequences.sequences[:, prompt_width:]
    raw_eos = getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id)
    if raw_eos is None:
        eos_token_ids: set[int] = set()
    elif isinstance(raw_eos, int):
        eos_token_ids = {raw_eos}
    else:
        eos_token_ids = {int(value) for value in raw_eos}
    for row, prompt_id in enumerate(prompt_batch.prompt_ids):
        response_ids, logprobs = _trim_generated_row(
            row,
            generated_ids,
            sequences.scores,
            eos_token_ids=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        if not (len(response_ids) == len(logprobs)):
            raise RuntimeError("generated response IDs and behavior logprobs are misaligned")
        group_size = int(prompt_counts[prompt_id])
        group_index = prompt_indices[prompt_id]
        prompt_indices[prompt_id] += 1
        group_id = (
            "generation-group-" + sha256_value([prompt_id, policy_version, seed])[:16]
            if group_size > 1
            else ""
        )
        records.append(
            TrajectoryRecord(
                trajectory_id=f"traj-{sha256_value([prompt_id, policy_version, seed, row])[:16]}",
                prompt_id=prompt_id,
                split="train",
                prompt_text=prompt_batch.prompt_texts[row],
                input_ids=individual_prompt_ids[row],
                response_ids=response_ids,
                response_text=tokenizer.decode(response_ids, skip_special_tokens=False),
                response_token_mask=[True] * len(response_ids),
                behavior_policy_id=policy_id,
                behavior_policy_revision=policy_revision,
                policy_version=policy_version,
                generation_seed=seed,
                sampling_temperature=temperature,
                top_p=top_p,
                behavior_logprobs=logprobs,
                generation_group_id=group_id,
                generation_group_index=group_index,
                prompt_group_size=group_size,
                created_at=datetime.now(UTC).isoformat(),
            )
        )
    return records


def build_proofgraph_hf_generator(
    *,
    tokenizer: Any,
    examples_by_id: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    policy_id: str,
    initial_policy_revision: str,
):
    """Bind HF generation to exact ProofGraph verification for online training."""
    from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask

    task = ProofGraphTask()

    def generate(
        model: Any,
        prompt_batch: PromptBatch,
        policy_version: int,
        seed: int,
    ) -> list[TrajectoryRecord]:
        records = hf_generate_trajectories(
            model,
            tokenizer,
            prompt_batch,
            policy_version,
            seed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            policy_id=policy_id,
            policy_revision=(f"{initial_policy_revision}@policy-{policy_version}"),
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
