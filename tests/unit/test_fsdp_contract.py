from __future__ import annotations

from types import SimpleNamespace
import unittest

from posttrain_circuits.learning.training.fsdp_contract import (
    FSDP_TRANSFORMER_LAYER,
    REQUESTED_FSDP_SHARDING_STRATEGY,
    effective_fsdp_sharding_strategy,
    full_state_dict_options,
    validate_model_fsdp_sharding,
)


class _FakeFSDP:
    def __init__(
        self,
        strategy: str,
        module: object,
        *,
        use_orig_params: bool | None = False,
    ) -> None:
        self.sharding_strategy = SimpleNamespace(name=strategy)
        self._use_orig_params = use_orig_params
        self.module = module
        self._module_tree: list[object] | None = None

    def modules(self):  # type: ignore[no-untyped-def]
        if self._module_tree is None:
            return iter((self, self.module))
        return iter(self._module_tree)


class Qwen3DecoderLayer:
    pass


class _FakeCausalLM:
    pass


def _fake_model(
    strategy: str,
    *,
    block_count: int = 2,
    wrapped_block_count: int | None = None,
    extra_wrappers: int = 0,
    use_orig_params: bool | None = False,
) -> _FakeFSDP:
    if wrapped_block_count is None:
        wrapped_block_count = block_count
    blocks = [Qwen3DecoderLayer() for _ in range(block_count)]
    root_module = _FakeCausalLM()
    root = _FakeFSDP(
        strategy,
        root_module,
        use_orig_params=use_orig_params,
    )
    child_wrappers = [
        _FakeFSDP(
            strategy,
            block,
            use_orig_params=use_orig_params,
        )
        for block in blocks[:wrapped_block_count]
    ]
    other_wrappers = [
        _FakeFSDP(
            strategy,
            _FakeCausalLM(),
            use_orig_params=use_orig_params,
        )
        for _ in range(extra_wrappers)
    ]
    root._module_tree = [
        root,
        root_module,
        *child_wrappers,
        *other_wrappers,
        *blocks,
    ]
    return root


class FSDPContractTests(unittest.TestCase):
    def test_all_reviewed_world_sizes_bind_requested_and_effective_strategy(self) -> None:
        expected = {
            1: "NO_SHARD",
            2: "FULL_SHARD",
            3: "FULL_SHARD",
            4: "FULL_SHARD",
        }
        for world_size, strategy in expected.items():
            with self.subTest(world_size=world_size):
                self.assertEqual(effective_fsdp_sharding_strategy(world_size), strategy)
                self.assertEqual(
                    validate_model_fsdp_sharding(
                        _fake_model(strategy),
                        world_size=world_size,
                        fsdp_type=_FakeFSDP,
                    ),
                    {
                        "requested_fsdp_sharding_strategy": (
                            REQUESTED_FSDP_SHARDING_STRATEGY
                        ),
                        "effective_fsdp_sharding_strategy": strategy,
                        "fsdp_wrapper_count": 3,
                    },
                )

    def test_missing_or_mixed_wrappers_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no actual FSDP wrappers"):
            validate_model_fsdp_sharding(
                SimpleNamespace(modules=lambda: iter(())),
                world_size=1,
                fsdp_type=_FakeFSDP,
            )
        mixed = _fake_model("FULL_SHARD")
        child_wrapper = next(
            module
            for module in mixed.modules()
            if isinstance(module, _FakeFSDP) and module is not mixed
        )
        child_wrapper.sharding_strategy = SimpleNamespace(name="NO_SHARD")
        with self.assertRaisesRegex(RuntimeError, "effective strategies differ"):
            validate_model_fsdp_sharding(
                mixed,
                world_size=2,
                fsdp_type=_FakeFSDP,
            )

    def test_root_only_and_partial_transformer_wrap_fail_for_every_world_size(self) -> None:
        expected = {1: "NO_SHARD", 2: "FULL_SHARD", 3: "FULL_SHARD", 4: "FULL_SHARD"}
        for world_size, strategy in expected.items():
            for wrapped_blocks, label in ((0, "root-only"), (1, "partial")):
                with self.subTest(
                    world_size=world_size,
                    topology=label,
                ):
                    with self.assertRaisesRegex(RuntimeError, "directly wrap every"):
                        validate_model_fsdp_sharding(
                            _fake_model(
                                strategy,
                                block_count=2,
                                wrapped_block_count=wrapped_blocks,
                            ),
                            world_size=world_size,
                            fsdp_type=_FakeFSDP,
                        )

    def test_extra_non_transformer_wrapper_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wrapper count differs"):
            validate_model_fsdp_sharding(
                _fake_model("FULL_SHARD", extra_wrappers=1),
                world_size=3,
                fsdp_type=_FakeFSDP,
            )
        with self.assertRaisesRegex(RuntimeError, "directly wrap every"):
            validate_model_fsdp_sharding(
                _fake_model(
                    "FULL_SHARD",
                    wrapped_block_count=1,
                    extra_wrappers=1,
                ),
                world_size=3,
                fsdp_type=_FakeFSDP,
            )

    def test_reviewed_transformer_layer_name_is_explicit(self) -> None:
        self.assertEqual(FSDP_TRANSFORMER_LAYER, "Qwen3DecoderLayer")

    def test_use_orig_params_must_be_explicitly_false_on_every_wrapper(self) -> None:
        for observed in (True, None):
            with self.subTest(observed=observed):
                with self.assertRaisesRegex(RuntimeError, "_use_orig_params=False"):
                    validate_model_fsdp_sharding(
                        _fake_model(
                            "FULL_SHARD",
                            use_orig_params=observed,
                        ),
                        world_size=2,
                        fsdp_type=_FakeFSDP,
                    )

    def test_w1_full_state_export_does_not_request_rank_zero_cpu_offload(self) -> None:
        self.assertEqual(
            full_state_dict_options(1),
            {"offload_to_cpu": False, "rank0_only": False},
        )
        self.assertEqual(
            full_state_dict_options(2),
            {"offload_to_cpu": True, "rank0_only": True},
        )


if __name__ == "__main__":
    unittest.main()
