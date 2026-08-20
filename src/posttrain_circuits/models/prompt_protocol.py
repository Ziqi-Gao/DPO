"""Central, hash-bound formatting for every model-facing prompt."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from posttrain_circuits.core.hashing import sha256_value

LEGACY_PROMPT_PROTOCOL = "legacy_raw_v1"
QWEN3_PROMPT_PROTOCOL = "qwen3_non_thinking_v1"


@dataclass(frozen=True)
class FormattedPrompt:
    raw_prompt: str
    model_facing_prompt: str
    prompt_protocol: str
    enable_thinking: bool
    chat_template_sha256: str
    raw_prompt_sha256: str
    model_facing_prompt_sha256: str

    def manifest_fields(self) -> dict[str, Any]:
        return asdict(self)


def chat_template_sha256(tokenizer: Any) -> str:
    template = str(getattr(tokenizer, "chat_template", None) or "")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def prompt_protocol_name(model_config: dict[str, Any] | None) -> str:
    if not model_config:
        return LEGACY_PROMPT_PROTOCOL
    protocol = model_config.get("prompt_protocol", LEGACY_PROMPT_PROTOCOL)
    if isinstance(protocol, dict):
        return str(protocol.get("name", LEGACY_PROMPT_PROTOCOL))
    return str(protocol)


def format_model_prompt(
    raw_prompt: str,
    tokenizer: Any,
    model_config: dict[str, Any] | None = None,
) -> FormattedPrompt:
    """Format one raw task prompt under the explicitly selected protocol.

    Qwen3 has exactly one allowed protocol: one user message, the pinned chat
    template, an assistant generation prompt, and thinking disabled. Legacy
    tracks remain byte-for-byte unchanged when no protocol is configured.
    """

    protocol = prompt_protocol_name(model_config)
    observed_template_hash = chat_template_sha256(tokenizer)
    if protocol == LEGACY_PROMPT_PROTOCOL:
        model_facing = raw_prompt
        enable_thinking = False
    elif protocol == QWEN3_PROMPT_PROTOCOL:
        if not model_config:
            raise ValueError("Qwen3 prompt formatting requires model configuration")
        configured = model_config.get("prompt_protocol")
        if not isinstance(configured, dict):
            raise ValueError("Qwen3 prompt_protocol must be a mapping")
        if configured.get("enable_thinking") is not False:
            raise ValueError("Qwen3 controlled protocol requires enable_thinking=false")
        expected_template_hash = str(configured.get("chat_template_sha256", ""))
        if len(expected_template_hash) != 64 or observed_template_hash != expected_template_hash:
            raise ValueError(
                "Qwen3 chat template differs from the preregistered bytes: "
                f"expected={expected_template_hash}, observed={observed_template_hash}"
            )
        model_facing = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enable_thinking = False
        if "<|im_start|>assistant\n<think>\n\n</think>\n\n" not in model_facing:
            raise ValueError("Qwen3 non-thinking assistant prefix is absent from formatted prompt")
    else:
        raise ValueError(f"unsupported prompt protocol {protocol!r}")
    return FormattedPrompt(
        raw_prompt=raw_prompt,
        model_facing_prompt=str(model_facing),
        prompt_protocol=protocol,
        enable_thinking=enable_thinking,
        chat_template_sha256=observed_template_hash,
        raw_prompt_sha256=sha256_value(raw_prompt),
        model_facing_prompt_sha256=sha256_value(str(model_facing)),
    )


def format_model_prompts(
    raw_prompts: list[str] | tuple[str, ...],
    tokenizer: Any,
    model_config: dict[str, Any] | None = None,
) -> list[FormattedPrompt]:
    return [format_model_prompt(prompt, tokenizer, model_config) for prompt in raw_prompts]


def prompt_manifest(
    prompts: list[FormattedPrompt],
    *,
    tokenizer_fingerprint: str,
) -> dict[str, Any]:
    if not prompts:
        raise ValueError("prompt manifest cannot be empty")
    protocols = {prompt.prompt_protocol for prompt in prompts}
    templates = {prompt.chat_template_sha256 for prompt in prompts}
    thinking = {prompt.enable_thinking for prompt in prompts}
    if len(protocols) != 1 or len(templates) != 1 or len(thinking) != 1:
        raise ValueError("prompt manifest mixes protocol bindings")
    content = {
        "format_version": 1,
        "prompt_protocol": next(iter(protocols)),
        "enable_thinking": next(iter(thinking)),
        "chat_template_sha256": next(iter(templates)),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "prompts": [prompt.manifest_fields() for prompt in prompts],
    }
    return {**content, "sha256": sha256_value(content)}
