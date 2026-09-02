"""Stable sampling and trajectory identities with no runtime dependencies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

SAMPLING_CURSOR_PROTOCOL_ID = "sampling-cursor-v1"
TRAJECTORY_ID_PROTOCOL_ID = "trajectory-content-v1"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sampling_cursor_id(
    *,
    optimizer_step: int,
    retry_index: int,
    prompt_id: str,
    generation_index: int,
) -> str:
    """Identify one logical draw independently of batch position or prompt order."""

    if optimizer_step < 0 or retry_index < 0 or generation_index < 0:
        raise ValueError("sampling cursor indices must be non-negative")
    if not prompt_id:
        raise ValueError("sampling cursor prompt_id must be non-empty")
    digest = _digest(
        {
            "protocol_id": SAMPLING_CURSOR_PROTOCOL_ID,
            "optimizer_step": optimizer_step,
            "retry_index": retry_index,
            "prompt_id": prompt_id,
            "generation_index": generation_index,
        }
    )
    return f"sampling-cursor-{digest[:24]}"


def actual_sampling_seed(
    *,
    sampling_request_seed: int,
    policy_version: int,
    sampling_cursor_id_value: str,
    sampling_protocol_id: str,
) -> int:
    """Derive the RNG seed actually assigned to one logical draw."""

    if not sampling_cursor_id_value or not sampling_protocol_id:
        raise ValueError("sampling cursor and protocol IDs must be non-empty")
    digest = _digest(
        {
            "sampling_request_seed": sampling_request_seed,
            "policy_version": policy_version,
            "sampling_cursor_id": sampling_cursor_id_value,
            "sampling_protocol_id": sampling_protocol_id,
        }
    )
    return int(digest[:16], 16) & ((1 << 63) - 1)


def generation_group_id(
    *,
    prompt_id: str,
    policy_version: int,
    sampling_request_seed: int,
    optimizer_step: int,
    retry_index: int,
    prompt_group_size: int,
    sampling_protocol_id: str,
) -> str:
    """Identify a same-prompt sampling group without using its batch location."""

    if prompt_group_size < 2:
        return ""
    digest = _digest(
        {
            "protocol_id": "generation-group-v1",
            "prompt_id": prompt_id,
            "policy_version": policy_version,
            "sampling_request_seed": sampling_request_seed,
            "optimizer_step": optimizer_step,
            "retry_index": retry_index,
            "prompt_group_size": prompt_group_size,
            "sampling_protocol_id": sampling_protocol_id,
        }
    )
    return f"generation-group-{digest[:24]}"


def trajectory_id(
    *,
    prompt_id: str,
    split: str,
    prompt_text: str,
    raw_prompt_text: str,
    input_ids: Sequence[int],
    prompt_protocol: str,
    enable_thinking: bool,
    chat_template_sha256: str,
    raw_prompt_sha256: str,
    model_facing_prompt_sha256: str,
    tokenizer_fingerprint: str,
    behavior_policy_id: str,
    behavior_policy_revision: str,
    policy_version: int,
    actual_sampling_seed_value: int,
    sampling_protocol_id: str,
    sampling_temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    response_ids: Sequence[int],
    response_token_mask: Sequence[bool],
    generation_group_index: int,
    prompt_group_size: int,
) -> str:
    """Derive identity from scientific content, policy, actual seed, and response."""

    payload: dict[str, Any] = {
        "protocol_id": TRAJECTORY_ID_PROTOCOL_ID,
        "prompt": {
            "prompt_id": prompt_id,
            "split": split,
            "prompt_text": prompt_text,
            "raw_prompt_text": raw_prompt_text,
            "input_ids": list(input_ids),
            "prompt_protocol": prompt_protocol,
            "enable_thinking": enable_thinking,
            "chat_template_sha256": chat_template_sha256,
            "raw_prompt_sha256": raw_prompt_sha256,
            "model_facing_prompt_sha256": model_facing_prompt_sha256,
            "tokenizer_fingerprint": tokenizer_fingerprint,
        },
        "policy": {
            "id": behavior_policy_id,
            "revision": behavior_policy_revision,
            "version": policy_version,
        },
        "sampling": {
            "actual_sampling_seed": actual_sampling_seed_value,
            "sampling_protocol_id": sampling_protocol_id,
            "temperature": sampling_temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "generation_group_index": generation_group_index,
            "prompt_group_size": prompt_group_size,
        },
        "response": {
            "response_ids": list(response_ids),
            "response_token_mask": list(response_token_mask),
        },
    }
    return f"traj-{_digest(payload)[:24]}"
