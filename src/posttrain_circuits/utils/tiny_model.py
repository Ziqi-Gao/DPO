"""Local tiny tokenizer and random Qwen model fixtures; no Hub access."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM


def tiny_vocabulary() -> dict[str, int]:
    specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    tokens = [
        "FACTS",
        "RULES",
        "QUERY",
        "Is",
        "true",
        "OUTPUT",
        "FORMAT",
        "TRUE",
        "NOT",
        "AND",
        "->",
        "<proof>",
        "</proof>",
        "<answer>",
        "</answer>",
        "0",
        "1",
        "(",
        ")",
        ",",
        ":",
        "?",
    ]
    tokens.extend([f"F{index:02d}" for index in range(1, 33)])
    tokens.extend([f"R{index:02d}" for index in range(1, 33)])
    tokens.extend([f"S{index:02d}" for index in range(1, 9)])
    tokens.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    tokens.extend([f"I{index:02d}" for index in range(1, 8)])
    for index in range(96):
        base = f"SYM_{index:03d}"
        tokens.extend(
            [
                base,
                f"{base}_L",
                f"{base}_R",
                f"{base}_ALT",
                f"{base}_ALT_L",
                f"{base}_ALT_R",
            ]
        )
    ordered = specials + list(dict.fromkeys(tokens))
    return {token: index for index, token in enumerate(ordered)}


def build_tiny_tokenizer(path: Path | None = None) -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel(tiny_vocabulary(), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)
    return tokenizer


def build_tiny_qwen(seed: int = 0) -> Qwen2ForCausalLM:
    import torch

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=len(tiny_vocabulary()),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
        use_cache=False,
        attention_dropout=0.0,
    )
    return Qwen2ForCausalLM(config)
