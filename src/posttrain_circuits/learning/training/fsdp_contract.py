"""Reviewed requested/effective FSDP strategy contract for elastic G0."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


REQUESTED_FSDP_SHARDING_STRATEGY = "FULL_SHARD"
FSDP_TRANSFORMER_LAYER = "Qwen3DecoderLayer"
EFFECTIVE_FSDP_SHARDING_STRATEGY_BY_WORLD_SIZE = {
    1: "NO_SHARD",
    2: "FULL_SHARD",
    3: "FULL_SHARD",
    4: "FULL_SHARD",
}


def effective_fsdp_sharding_strategy(world_size: int) -> str:
    """Return the PyTorch 2.8 strategy expected after FSDP construction."""

    if type(world_size) is not int or world_size not in (
        EFFECTIVE_FSDP_SHARDING_STRATEGY_BY_WORLD_SIZE
    ):
        raise ValueError("FSDP world size must be one of 1, 2, 3, or 4")
    return EFFECTIVE_FSDP_SHARDING_STRATEGY_BY_WORLD_SIZE[world_size]


def full_state_dict_options(world_size: int) -> dict[str, bool]:
    """Return safe FSDP1 full-state options, including the W=1 workaround."""

    effective_fsdp_sharding_strategy(world_size)
    is_multi_process = world_size > 1
    return {
        "offload_to_cpu": is_multi_process,
        "rank0_only": is_multi_process,
    }


def _strategy_name(value: object) -> str:
    name = getattr(value, "name", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError("FSDP wrapper exposes no canonical sharding strategy")
    return name


def _modules(model: Any) -> Iterator[Any]:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise RuntimeError("prepared model does not expose a module tree")
    return iter(modules())


def validate_model_fsdp_sharding(
    model: Any,
    *,
    world_size: int,
    fsdp_type: type[Any] | None = None,
) -> dict[str, object]:
    """Enforce the reviewed Qwen3 FSDP tree and W=1..4 sharding result."""

    if fsdp_type is None:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        fsdp_type = FSDP
    modules = list(_modules(model))
    wrappers = [module for module in modules if isinstance(module, fsdp_type)]
    if not wrappers:
        raise RuntimeError("prepared exact-global model has no actual FSDP wrappers")
    if not isinstance(model, fsdp_type):
        raise RuntimeError("prepared exact-global model root is not an FSDP wrapper")

    transformer_blocks = [
        module for module in modules if type(module).__name__ == FSDP_TRANSFORMER_LAYER
    ]
    if not transformer_blocks:
        raise RuntimeError(
            f"prepared model exposes no {FSDP_TRANSFORMER_LAYER} transformer blocks"
        )
    directly_wrapped_blocks = [
        wrapped
        for wrapper in wrappers
        if type(wrapped := getattr(wrapper, "module", None)).__name__
        == FSDP_TRANSFORMER_LAYER
    ]
    if (
        len(directly_wrapped_blocks) != len(transformer_blocks)
        or {id(module) for module in directly_wrapped_blocks}
        != {id(module) for module in transformer_blocks}
    ):
        raise RuntimeError(
            "prepared FSDP tree does not directly wrap every "
            f"{FSDP_TRANSFORMER_LAYER} block"
        )
    expected_wrapper_count = len(transformer_blocks) + 1
    if len(wrappers) != expected_wrapper_count:
        raise RuntimeError(
            "prepared FSDP wrapper count differs from the reviewed transformer "
            f"auto-wrap tree: expected={expected_wrapper_count}, observed={len(wrappers)}"
        )
    expected = effective_fsdp_sharding_strategy(world_size)
    observed = [_strategy_name(wrapper.sharding_strategy) for wrapper in wrappers]
    if any(strategy != expected for strategy in observed):
        raise RuntimeError(
            "prepared FSDP wrapper effective strategies differ from the reviewed "
            f"W={world_size} strategy: expected={expected}, observed={observed}"
        )
    use_orig_params = [getattr(wrapper, "_use_orig_params", None) for wrapper in wrappers]
    if any(value is not False for value in use_orig_params):
        raise RuntimeError(
            "prepared FSDP wrappers must all expose _use_orig_params=False"
        )
    return {
        "requested_fsdp_sharding_strategy": REQUESTED_FSDP_SHARDING_STRATEGY,
        "effective_fsdp_sharding_strategy": expected,
        "fsdp_wrapper_count": len(wrappers),
    }


__all__ = [
    "EFFECTIVE_FSDP_SHARDING_STRATEGY_BY_WORLD_SIZE",
    "FSDP_TRANSFORMER_LAYER",
    "REQUESTED_FSDP_SHARDING_STRATEGY",
    "effective_fsdp_sharding_strategy",
    "full_state_dict_options",
    "validate_model_fsdp_sharding",
]
