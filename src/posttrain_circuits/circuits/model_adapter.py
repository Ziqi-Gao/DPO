"""HF/TransformerLens compatibility, hook, and GQA extraction gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json


def q_to_kv_head_mapping(
    num_query_heads: int,
    num_kv_heads: int,
) -> tuple[int, ...]:
    if num_query_heads < 1 or num_kv_heads < 1 or num_query_heads % num_kv_heads:
        raise ValueError("GQA requires query heads to be divisible by key/value heads")
    group_size = num_query_heads // num_kv_heads
    return tuple(index // group_size for index in range(num_query_heads))


@dataclass
class CompatibilityReport:
    architecture: str
    layer_count: int
    attention_head_count: int
    key_value_head_count: int
    q_to_kv_mapping: tuple[int, ...]
    hook_positions: dict[str, tuple[str, ...]]
    residual_identity_error: float
    mlp_identity_error: float
    hf_identity_tolerance: float
    hf_identity_passed: bool
    transformerlens_layer_count: int | None = None
    transformerlens_attention_head_count: int | None = None
    transformerlens_key_value_head_count: int | None = None
    transformerlens_logit_error: float | None = None
    transformerlens_tolerance: float | None = None
    transformerlens_parity_passed: bool | None = None
    passed: bool = False

    @property
    def hf_identity_max_error(self) -> float:
        return max(
            self.residual_identity_error,
            self.mlp_identity_error,
        )

    @property
    def sha256(self) -> str:
        return sha256_value(asdict(self))

    def write(self, path: Path) -> None:
        atomic_write_json(
            path,
            {
                **asdict(self),
                "hf_identity_max_error": self.hf_identity_max_error,
                "sha256": self.sha256,
            },
        )


def _identity_hook(
    module: torch.nn.Module,
    inputs: tuple[Any, ...],
    output: Any,
) -> Any:
    del module, inputs
    return output


def _validate_hf_structure(
    model: Any,
) -> tuple[tuple[Any, ...], int, int, int]:
    config = model.config
    layers = tuple(model.model.layers)
    configured_layers = int(config.num_hidden_layers)
    if len(layers) != configured_layers:
        raise RuntimeError(f"HF layer structure disagrees with config: {len(layers)} != {configured_layers}")
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    mapping = q_to_kv_head_mapping(query_heads, kv_heads)
    configured_head_dim = getattr(config, "head_dim", None)
    head_width = (
        int(configured_head_dim)
        if configured_head_dim is not None and int(configured_head_dim) > 0
        else int(config.hidden_size) // query_heads
    )
    if head_width < 1:
        raise RuntimeError("attention head dimension must be positive")
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        expected = {
            "q_proj": query_heads * head_width,
            "k_proj": kv_heads * head_width,
            "v_proj": kv_heads * head_width,
        }
        for projection_name, output_width in expected.items():
            projection = getattr(attention, projection_name)
            if int(projection.out_features) != output_width:
                raise RuntimeError(
                    f"layer {index} {projection_name} has "
                    f"{projection.out_features} outputs, expected "
                    f"{output_width}"
                )
    return layers, query_heads, kv_heads, len(mapping)


def _identity_error(
    model: Any,
    input_ids: torch.Tensor,
    modules: tuple[torch.nn.Module, ...],
    baseline: torch.Tensor,
) -> float:
    handles = [module.register_forward_hook(_identity_hook) for module in modules]
    try:
        hooked = model(input_ids=input_ids).logits
    finally:
        for handle in handles:
            handle.remove()
    return float((baseline - hooked).abs().max())


@torch.no_grad()
def check_hf_identity_compatibility(
    model: Any,
    input_ids: torch.Tensor,
    *,
    tolerance: float = 1e-6,
) -> CompatibilityReport:
    model.eval()
    layers, query_heads, kv_heads, mapping_length = _validate_hf_structure(model)
    if mapping_length != query_heads:
        raise RuntimeError("GQA query-to-KV mapping is incomplete")
    baseline = model(input_ids=input_ids).logits
    residual_error = _identity_error(
        model,
        input_ids,
        tuple(layers),
        baseline,
    )
    mlp_error = _identity_error(
        model,
        input_ids,
        tuple(layer.mlp for layer in layers),
        baseline,
    )
    identity_passed = max(residual_error, mlp_error) <= tolerance
    qk_norm = type(model).__name__ == "Qwen3ForCausalLM"
    report = CompatibilityReport(
        architecture=type(model).__name__,
        layer_count=len(layers),
        attention_head_count=query_heads,
        key_value_head_count=kv_heads,
        q_to_kv_mapping=q_to_kv_head_mapping(
            query_heads,
            kv_heads,
        ),
        hook_positions={
            "residual_stream": tuple(f"model.layers.{index}:output" for index in range(len(layers))),
            "mlp_output": tuple(f"model.layers.{index}.mlp:output" for index in range(len(layers))),
            **(
                {
                    "query_projection": tuple(
                        f"model.layers.{index}.self_attn.q_proj:output:pre_q_norm_pre_rope"
                        for index in range(len(layers))
                    ),
                    "key_projection": tuple(
                        f"model.layers.{index}.self_attn.k_proj:output:pre_k_norm_pre_rope"
                        for index in range(len(layers))
                    ),
                }
                if qk_norm
                else {}
            ),
        },
        residual_identity_error=residual_error,
        mlp_identity_error=mlp_error,
        hf_identity_tolerance=tolerance,
        hf_identity_passed=identity_passed,
        passed=identity_passed,
    )
    if not report.passed:
        raise RuntimeError(
            "HF identity-hook compatibility failed: "
            f"max error {report.hf_identity_max_error:.6g} exceeds "
            f"{tolerance:.6g}"
        )
    return report


def build_transformerlens_qwen_from_hf(
    hf_model: Any,
) -> Any:
    """Convert an already-loaded Qwen2 or Qwen3 HF model without Hub access."""

    architecture = type(hf_model).__name__
    if architecture not in {"Qwen2ForCausalLM", "Qwen3ForCausalLM"}:
        raise ValueError("offline TransformerLens conversion supports Qwen2ForCausalLM/Qwen3ForCausalLM")
    try:
        from transformer_lens import (
            HookedTransformer,
            HookedTransformerConfig,
        )

        if architecture == "Qwen3ForCausalLM":
            from transformer_lens.pretrained.weight_conversions.qwen3 import (
                convert_qwen3_weights as convert_weights,
            )
        else:
            from transformer_lens.pretrained.weight_conversions.qwen2 import (
                convert_qwen2_weights as convert_weights,
            )
    except ImportError as error:
        raise RuntimeError("TransformerLens compatibility requires the 'circuits' extra") from error
    config = hf_model.config
    configured_head_dim = getattr(config, "head_dim", None)
    head_width = (
        int(configured_head_dim)
        if configured_head_dim is not None and int(configured_head_dim) > 0
        else int(config.hidden_size) // int(config.num_attention_heads)
    )
    tl_config = HookedTransformerConfig(
        n_layers=int(config.num_hidden_layers),
        d_model=int(config.hidden_size),
        n_ctx=(
            min(2048, int(config.max_position_embeddings))
            if architecture == "Qwen3ForCausalLM"
            else int(config.max_position_embeddings)
        ),
        d_head=head_width,
        n_heads=int(config.num_attention_heads),
        n_key_value_heads=int(config.num_key_value_heads),
        d_mlp=int(config.intermediate_size),
        act_fn=str(config.hidden_act),
        d_vocab=int(config.vocab_size),
        eps=float(config.rms_norm_eps),
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_dim=head_width,
        rotary_adjacent_pairs=False,
        rotary_base=int(config.rope_theta),
        final_rms=True,
        gated_mlp=True,
        original_architecture=architecture,
        init_weights=False,
        device="cpu",
        dtype=torch.float32,
        default_prepend_bos=False,
        use_qk_norm=architecture == "Qwen3ForCausalLM",
    )
    tl_model = HookedTransformer(
        tl_config,
        tokenizer=None,
        move_to_device=True,
    ).eval()
    tl_model.load_and_process_state_dict(
        convert_weights(hf_model, tl_config),
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return tl_model


@torch.no_grad()
def require_transformerlens_parity(
    hf_model: Any,
    tl_model: Any,
    input_ids: torch.Tensor,
    *,
    tolerance: float,
    output_path: Path | None = None,
) -> CompatibilityReport:
    hf_model.eval()
    tl_model.eval()
    hf_logits = hf_model(input_ids=input_ids).logits
    tl_logits = tl_model(input_ids)
    if hf_logits.shape != tl_logits.shape:
        raise RuntimeError(
            f"HF/TransformerLens logit shapes differ: {tuple(hf_logits.shape)} != {tuple(tl_logits.shape)}"
        )
    logit_error = float((hf_logits - tl_logits).abs().max())
    report = check_hf_identity_compatibility(
        hf_model,
        input_ids,
        tolerance=tolerance,
    )
    tl_layers = int(tl_model.cfg.n_layers)
    tl_heads = int(tl_model.cfg.n_heads)
    tl_kv_heads = int(tl_model.cfg.n_key_value_heads)
    structure_matches = (
        tl_layers == report.layer_count
        and tl_heads == report.attention_head_count
        and tl_kv_heads == report.key_value_head_count
    )
    report.transformerlens_layer_count = tl_layers
    report.transformerlens_attention_head_count = tl_heads
    report.transformerlens_key_value_head_count = tl_kv_heads
    report.transformerlens_logit_error = logit_error
    report.transformerlens_tolerance = tolerance
    report.transformerlens_parity_passed = structure_matches and logit_error <= tolerance
    report.passed = report.hf_identity_passed and report.transformerlens_parity_passed
    if output_path is not None:
        report.write(output_path)
    if not report.passed:
        raise RuntimeError(
            "HF/TransformerLens extraction refused: "
            f"logit error {logit_error:.6g}, tolerance "
            f"{tolerance:.6g}, structure_matches={structure_matches}"
        )
    return report


def require_compatible_for_extraction(
    report: CompatibilityReport,
    *,
    require_transformerlens: bool,
) -> None:
    if not report.hf_identity_passed:
        raise RuntimeError("circuit extraction refused: HF identity hooks failed")
    if require_transformerlens and report.transformerlens_parity_passed is not True:
        raise RuntimeError("circuit extraction refused: no passing HF/TransformerLens parity evidence")
