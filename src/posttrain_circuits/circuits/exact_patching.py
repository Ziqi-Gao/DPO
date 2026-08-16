"""Exact head, MLP, residual, Q/K/V, and causal-path patching."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from posttrain_circuits.circuits.graph import (
    AblationSpec,
    CircuitEvaluation,
    CircuitMask,
    CircuitScores,
)
from posttrain_circuits.circuits.model_adapter import (
    q_to_kv_head_mapping,
)
from posttrain_circuits.circuits.probes import CircuitProbeSpec, TargetSequenceMetric

BehaviorMetric = Callable[[torch.Tensor], torch.Tensor] | TargetSequenceMetric


@dataclass(frozen=True)
class ExactTokenPair:
    pair_id: str
    clean_ids: torch.Tensor
    corrupt_ids: torch.Tensor
    clean_target_ids: tuple[int, ...] = ()
    corrupt_target_ids: tuple[int, ...] = ()
    clean_metric_positions: tuple[int, ...] = ()
    corrupt_metric_positions: tuple[int, ...] = ()
    clean_intervention_positions: tuple[int, ...] = ()
    corrupt_intervention_positions: tuple[int, ...] = ()
    stage: str = ""
    semantic_pair_hash: str = ""
    tokenized_pair_hash: str = ""

    def __post_init__(self) -> None:
        if self.clean_ids.shape != self.corrupt_ids.shape:
            raise ValueError(f"pair {self.pair_id} is not token-shape matched")
        if self.clean_target_ids or self.corrupt_target_ids:
            if not self.clean_target_ids or len(self.clean_target_ids) != len(self.corrupt_target_ids):
                raise ValueError(f"pair {self.pair_id} target sequences are not shape matched")
            if len(self.clean_metric_positions) != len(self.clean_target_ids):
                raise ValueError(f"pair {self.pair_id} clean metric positions are invalid")
            if len(self.corrupt_metric_positions) != len(self.corrupt_target_ids):
                raise ValueError(f"pair {self.pair_id} corrupt metric positions are invalid")

    @classmethod
    def from_probe(
        cls,
        probe: CircuitProbeSpec,
        *,
        device: torch.device | None = None,
    ) -> ExactTokenPair:
        clean = torch.tensor([probe.clean_input_ids], dtype=torch.long, device=device)
        corrupt = torch.tensor([probe.corrupt_input_ids], dtype=torch.long, device=device)
        return cls(
            pair_id=probe.probe_id,
            clean_ids=clean,
            corrupt_ids=corrupt,
            clean_target_ids=probe.clean_target_ids,
            corrupt_target_ids=probe.corrupt_target_ids,
            clean_metric_positions=probe.clean_metric_positions,
            corrupt_metric_positions=probe.corrupt_metric_positions,
            clean_intervention_positions=probe.clean_intervention_positions,
            corrupt_intervention_positions=probe.corrupt_intervention_positions,
            stage=probe.stage,
            semantic_pair_hash=probe.semantic_pair_hash,
            tokenized_pair_hash=probe.tokenized_pair_hash,
        )


def _metric_value(
    metric: BehaviorMetric,
    logits: torch.Tensor,
    pair: ExactTokenPair,
    *,
    side: str,
) -> torch.Tensor:
    if isinstance(metric, TargetSequenceMetric):
        return metric(logits, pair=pair, side=side)
    return metric(logits)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    module: torch.nn.Module
    hook_point: str
    head_index: int | None = None
    head_width: int | None = None
    semantic_position: str = ""

    @property
    def head_slice(self) -> slice | None:
        if self.head_index is None or self.head_width is None:
            return None
        start = self.head_index * self.head_width
        return slice(start, start + self.head_width)


def component_specs(model: Any) -> dict[str, ComponentSpec]:
    config = model.config
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    mapping = q_to_kv_head_mapping(query_heads, kv_heads)
    head_width = int(config.hidden_size) // query_heads
    specs: dict[str, ComponentSpec] = {}
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        for head_index in range(query_heads):
            name = f"layer.{layer_index}.attention_head.{head_index}.output"
            specs[name] = ComponentSpec(
                name,
                attention.o_proj,
                "input",
                head_index,
                head_width,
                "attention_head_output_before_o_projection",
            )
            q_name = f"layer.{layer_index}.q_head.{head_index}"
            specs[q_name] = ComponentSpec(
                q_name,
                attention.q_proj,
                "output",
                head_index,
                head_width,
                (f"q_projection_output_pre_rope:kv_head={mapping[head_index]}"),
            )
        for head_index in range(kv_heads):
            for kind in ("k", "v"):
                name = f"layer.{layer_index}.{kind}_head.{head_index}"
                specs[name] = ComponentSpec(
                    name,
                    getattr(attention, f"{kind}_proj"),
                    "output",
                    head_index,
                    head_width,
                    f"{kind}_projection_output_pre_rope",
                )
        mlp_name = f"layer.{layer_index}.mlp_out"
        specs[mlp_name] = ComponentSpec(
            mlp_name,
            layer.mlp,
            "output",
            semantic_position="mlp_output_before_residual_add",
        )
        residual_name = f"layer.{layer_index}.resid_out"
        specs[residual_name] = ComponentSpec(
            residual_name,
            layer,
            "output",
            semantic_position="residual_stream_after_block",
        )
    return specs


def component_modules(
    model: Any,
) -> dict[str, torch.nn.Module]:
    return {name: spec.module for name, spec in component_specs(model).items()}


def component_metadata(model: Any) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "hook_point": spec.hook_point,
            "head_index": spec.head_index,
            "head_width": spec.head_width,
            "semantic_position": spec.semantic_position,
        }
        for name, spec in component_specs(model).items()
    }


def normalize_component_name(
    model: Any,
    name: str,
) -> str | None:
    specs = component_specs(model)
    if name in specs:
        return name
    attention = re.fullmatch(
        r"a(\d+)\.h(\d+)(?:<([qkv])>)?",
        name,
    )
    if attention is not None:
        layer = int(attention.group(1))
        query_head = int(attention.group(2))
        kind = attention.group(3)
        if kind is None:
            normalized = f"layer.{layer}.attention_head.{query_head}.output"
        elif kind == "q":
            normalized = f"layer.{layer}.q_head.{query_head}"
        else:
            mapping = q_to_kv_head_mapping(
                int(model.config.num_attention_heads),
                int(model.config.num_key_value_heads),
            )
            normalized = f"layer.{layer}.{kind}_head.{mapping[query_head]}"
        return normalized if normalized in specs else None
    mlp = re.fullmatch(r"m(\d+)", name)
    if mlp is not None:
        normalized = f"layer.{int(mlp.group(1))}.mlp_out"
        return normalized if normalized in specs else None
    if name == "logits":
        normalized = f"layer.{int(model.config.num_hidden_layers) - 1}.resid_out"
        return normalized if normalized in specs else None
    return None


def normalize_circuit_scores(
    model: Any,
    scores: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    normalized = {}
    unsupported = []
    for name, score in scores.items():
        if "->" in name:
            sender, receiver = name.split("->", 1)
            sender_name = normalize_component_name(model, sender)
            receiver_name = normalize_component_name(model, receiver)
            if sender_name is None or receiver_name is None:
                unsupported.append(name)
                continue
            normalized[f"{sender_name}->{receiver_name}"] = score
        else:
            component = normalize_component_name(model, name)
            if component is None:
                unsupported.append(name)
                continue
            normalized[component] = score
    if not normalized:
        raise ValueError("no circuit scores map to exact interventions")
    return normalized, unsupported


def _tensor_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_output(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    return replacement


def _select_component(
    tensor: torch.Tensor,
    spec: ComponentSpec,
) -> torch.Tensor:
    head_slice = spec.head_slice
    return tensor if head_slice is None else tensor[..., head_slice]


def _replace_component(
    tensor: torch.Tensor,
    spec: ComponentSpec,
    replacement: torch.Tensor,
) -> torch.Tensor:
    head_slice = spec.head_slice
    if head_slice is None:
        return replacement.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
    result = tensor.clone()
    result[..., head_slice] = replacement.to(
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return result


class ExactPatchingBackend:
    version = "hf-exact-patching-v2"

    def __init__(
        self,
        clean_ids: torch.Tensor | ExactTokenPair,
        corrupt_ids: torch.Tensor | None = None,
    ) -> None:
        if isinstance(clean_ids, ExactTokenPair):
            if corrupt_ids is not None:
                raise ValueError("do not pass corrupt_ids with an ExactTokenPair")
            self.discovery_pair = clean_ids
        else:
            if corrupt_ids is None:
                raise ValueError("exact patching requires corrupt token IDs")
            self.discovery_pair = ExactTokenPair("discovery", clean_ids, corrupt_ids)

    def _capture(
        self,
        model: Any,
        ids: torch.Tensor,
        names: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        specs = component_specs(model)
        handles = []
        for name in names:
            if name not in specs:
                raise ValueError(f"unknown patching component {name!r}")
            spec = specs[name]
            if spec.hook_point == "output":

                def output_hook(
                    module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    component: ComponentSpec = spec,
                ) -> None:
                    del module, inputs
                    captured[component.name] = (
                        _select_component(
                            _tensor_output(output),
                            component,
                        )
                        .detach()
                        .clone()
                    )

                handles.append(spec.module.register_forward_hook(output_hook))
            else:

                def input_hook(
                    module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    *,
                    component: ComponentSpec = spec,
                ) -> None:
                    del module
                    captured[component.name] = (
                        _select_component(
                            inputs[0],
                            component,
                        )
                        .detach()
                        .clone()
                    )

                handles.append(spec.module.register_forward_pre_hook(input_hook))
        try:
            model(input_ids=ids)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    def _run_patched(
        self,
        model: Any,
        source_ids: torch.Tensor,
        replacements: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        specs = component_specs(model)
        handles = []
        for name, replacement in replacements.items():
            if name not in specs:
                raise ValueError(f"unknown patching component {name!r}")
            spec = specs[name]
            if spec.hook_point == "output":

                def output_hook(
                    module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    component: ComponentSpec = spec,
                    value: torch.Tensor = replacement,
                ) -> Any:
                    del module, inputs
                    changed = _replace_component(
                        _tensor_output(output),
                        component,
                        value,
                    )
                    return _replace_output(output, changed)

                handles.append(spec.module.register_forward_hook(output_hook))
            else:

                def input_hook(
                    module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    *,
                    component: ComponentSpec = spec,
                    value: torch.Tensor = replacement,
                ) -> tuple[Any, ...]:
                    del module
                    changed = _replace_component(
                        inputs[0],
                        component,
                        value,
                    )
                    return (changed, *inputs[1:])

                handles.append(spec.module.register_forward_pre_hook(input_hook))
        try:
            return model(input_ids=source_ids).logits
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def score_all_components(
        self,
        model: Any,
        metric: BehaviorMetric,
    ) -> CircuitScores:
        model.eval()
        names = tuple(component_specs(model))
        corrupt_cache = self._capture(
            model,
            self.discovery_pair.corrupt_ids,
            names,
        )
        clean_value = float(
            _metric_value(
                metric,
                model(input_ids=self.discovery_pair.clean_ids).logits,
                self.discovery_pair,
                side="clean",
            )
        )
        scores: dict[str, float] = {}
        for name in names:
            patched = self._run_patched(
                model,
                self.discovery_pair.clean_ids,
                {name: corrupt_cache[name]},
            )
            scores[name] = clean_value - float(
                _metric_value(metric, patched, self.discovery_pair, side="clean")
            )
        return CircuitScores(scores=scores, node_scores=scores)

    @torch.no_grad()
    def score_path(
        self,
        model: Any,
        metric: BehaviorMetric,
        *,
        sender: str,
        receiver: str,
    ) -> CircuitScores:
        specs = component_specs(model)
        if sender not in specs or receiver not in specs:
            raise ValueError("path sender and receiver must be components")
        if sender == receiver:
            raise ValueError("path sender and receiver must differ")
        corrupt_sender = self._capture(
            model,
            self.discovery_pair.corrupt_ids,
            (sender,),
        )[sender]
        receiver_cache: dict[str, torch.Tensor] = {}
        receiver_spec = specs[receiver]

        def capture_receiver(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            del module, inputs
            receiver_cache[receiver] = (
                _select_component(
                    _tensor_output(output),
                    receiver_spec,
                )
                .detach()
                .clone()
            )

        def capture_receiver_input(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
        ) -> None:
            del module
            receiver_cache[receiver] = (
                _select_component(
                    inputs[0],
                    receiver_spec,
                )
                .detach()
                .clone()
            )

        handle = (
            receiver_spec.module.register_forward_hook(capture_receiver)
            if receiver_spec.hook_point == "output"
            else receiver_spec.module.register_forward_pre_hook(capture_receiver_input)
        )
        try:
            self._run_patched(
                model,
                self.discovery_pair.clean_ids,
                {sender: corrupt_sender},
            )
        finally:
            handle.remove()
        clean = float(
            _metric_value(
                metric,
                model(input_ids=self.discovery_pair.clean_ids).logits,
                self.discovery_pair,
                side="clean",
            )
        )
        path_logits = self._run_patched(
            model,
            self.discovery_pair.clean_ids,
            {receiver: receiver_cache[receiver]},
        )
        path_name = f"{sender}->{receiver}"
        return CircuitScores(
            scores={
                path_name: clean
                - float(_metric_value(metric, path_logits, self.discovery_pair, side="clean"))
            },
            edge_scores={
                path_name: clean
                - float(_metric_value(metric, path_logits, self.discovery_pair, side="clean"))
            },
        )

    def _transmitted_receiver_cache(
        self,
        model: Any,
        pair: ExactTokenPair,
        paths: tuple[str, ...],
        *,
        sender_source_ids: torch.Tensor,
        run_source_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        specs = component_specs(model)
        split_paths = [path.split("->", 1) for path in paths]
        senders = tuple(dict.fromkeys(path[0] for path in split_paths))
        receivers = tuple(dict.fromkeys(path[1] for path in split_paths))
        if any(sender not in specs or receiver not in specs for sender, receiver in split_paths):
            raise ValueError("path mask contains an unknown endpoint")
        sender_cache = self._capture(
            model,
            sender_source_ids,
            senders,
        )
        receiver_cache: dict[str, torch.Tensor] = {}
        handles = []
        for receiver in receivers:
            spec = specs[receiver]
            if spec.hook_point == "output":

                def output_hook(
                    module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    output: Any,
                    *,
                    component: ComponentSpec = spec,
                ) -> None:
                    del module, inputs
                    receiver_cache[component.name] = (
                        _select_component(
                            _tensor_output(output),
                            component,
                        )
                        .detach()
                        .clone()
                    )

                handles.append(spec.module.register_forward_hook(output_hook))
            else:

                def input_hook(
                    module: torch.nn.Module,
                    inputs: tuple[Any, ...],
                    *,
                    component: ComponentSpec = spec,
                ) -> None:
                    del module
                    receiver_cache[component.name] = (
                        _select_component(
                            inputs[0],
                            component,
                        )
                        .detach()
                        .clone()
                    )

                handles.append(spec.module.register_forward_pre_hook(input_hook))
        try:
            self._run_patched(
                model,
                run_source_ids,
                sender_cache,
            )
        finally:
            for handle in handles:
                handle.remove()
        if set(receiver_cache) != set(receivers):
            raise RuntimeError("selected path receiver was not reached during forward")
        return receiver_cache

    def activation_statistics(
        self,
        model: Any,
        names: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, float]]:
        requested = names or tuple(component_specs(model))
        endpoints = tuple(
            dict.fromkeys(
                endpoint
                for name in requested
                for endpoint in (name.split("->", 1) if "->" in name else (name,))
            )
        )
        cache = self._capture(
            model,
            self.discovery_pair.clean_ids,
            endpoints,
        )
        base = {
            name: {
                "activation_norm": float(value.float().norm()),
                "activation_size": float(value.numel()),
            }
            for name, value in cache.items()
        }
        statistics = {}
        for name in requested:
            if "->" not in name:
                statistics[name] = base[name]
                continue
            sender, receiver = name.split("->", 1)
            statistics[name] = {
                "activation_norm": (base[sender]["activation_norm"] + base[receiver]["activation_norm"])
                / 2.0,
                "activation_size": (base[sender]["activation_size"] + base[receiver]["activation_size"]),
            }
        return statistics

    def _ablation_cache(
        self,
        model: Any,
        pair: ExactTokenPair,
        components: tuple[str, ...],
        kind: str,
    ) -> dict[str, torch.Tensor]:
        if kind == "counterfactual_replacement":
            return self._capture(
                model,
                pair.corrupt_ids,
                components,
            )
        if kind not in {"mean", "zero"}:
            raise ValueError("ablation must be counterfactual_replacement, mean, or zero")
        clean_cache = self._capture(
            model,
            pair.clean_ids,
            components,
        )
        cache = {}
        for name, activation in clean_cache.items():
            if kind == "zero":
                cache[name] = torch.zeros_like(activation)
            else:
                dimensions = tuple(range(activation.ndim - 1))
                mean = activation.mean(
                    dim=dimensions,
                    keepdim=True,
                )
                cache[name] = mean.expand_as(activation)
        return cache

    @torch.no_grad()
    def evaluate_mask_per_pair(
        self,
        model: Any,
        pairs: list[ExactTokenPair],
        mask: CircuitMask,
        ablation: AblationSpec,
        metric: BehaviorMetric,
    ) -> list[CircuitEvaluation]:
        if not pairs:
            raise ValueError("necessity/sufficiency require held-out validation pairs")
        path_flags = ["->" in name for name in mask.components]
        if any(path_flags) and not all(path_flags):
            raise ValueError("exact masks cannot mix nodes and sender-to-receiver paths")
        path_mode = all(path_flags)
        evaluations = []
        for pair in pairs:
            if path_mode:
                receivers = tuple(dict.fromkeys(name.split("->", 1)[1] for name in mask.components))
                if ablation.kind == "counterfactual_replacement":
                    cache = self._transmitted_receiver_cache(
                        model,
                        pair,
                        mask.components,
                        sender_source_ids=pair.corrupt_ids,
                        run_source_ids=pair.clean_ids,
                    )
                    clean_cache = self._transmitted_receiver_cache(
                        model,
                        pair,
                        mask.components,
                        sender_source_ids=pair.clean_ids,
                        run_source_ids=pair.corrupt_ids,
                    )
                else:
                    cache = self._ablation_cache(
                        model,
                        pair,
                        receivers,
                        ablation.kind,
                    )
                    clean_cache = self._capture(
                        model,
                        pair.clean_ids,
                        receivers,
                    )
            else:
                cache = self._ablation_cache(
                    model,
                    pair,
                    mask.components,
                    ablation.kind,
                )
                clean_cache = self._capture(
                    model,
                    pair.clean_ids,
                    mask.components,
                )
            clean = float(_metric_value(metric, model(input_ids=pair.clean_ids).logits, pair, side="clean"))
            corrupt = float(
                _metric_value(metric, model(input_ids=pair.corrupt_ids).logits, pair, side="corrupt")
            )
            patched = float(
                _metric_value(
                    metric,
                    self._run_patched(model, pair.clean_ids, cache),
                    pair,
                    side="clean",
                )
            )
            sufficient = float(
                _metric_value(
                    metric,
                    self._run_patched(model, pair.corrupt_ids, clean_cache),
                    pair,
                    side="corrupt",
                )
            )
            denominator = clean - corrupt
            faithfulness = (clean - patched) / denominator if abs(denominator) > 1e-12 else 0.0
            sufficiency = (sufficient - corrupt) / denominator if abs(denominator) > 1e-12 else 0.0
            evaluations.append(
                CircuitEvaluation(
                    clean,
                    corrupt,
                    patched,
                    faithfulness,
                    clean - patched,
                    sufficiency,
                )
            )
        return evaluations

    @torch.no_grad()
    def evaluate_mask(
        self,
        model: Any,
        pairs: list[ExactTokenPair],
        mask: CircuitMask,
        ablation: AblationSpec,
        metric: BehaviorMetric | None = None,
    ) -> CircuitEvaluation:
        if metric is None:
            raise ValueError("exact patching evaluation requires a behavior metric")
        evaluations = self.evaluate_mask_per_pair(
            model,
            pairs,
            mask,
            ablation,
            metric,
        )
        fields = tuple(asdict(evaluations[0]))
        means = {
            field: sum(float(getattr(evaluation, field)) for evaluation in evaluations) / len(evaluations)
            for field in fields
        }
        return CircuitEvaluation(**means)

    @torch.no_grad()
    def sanity_checks(
        self,
        model: Any,
        pair: ExactTokenPair,
        metric: BehaviorMetric,
        *,
        tolerance: float = 1e-5,
    ) -> dict[str, float | bool]:
        components = tuple(component_specs(model))
        clean_cache = self._capture(model, pair.clean_ids, components)
        corrupt_cache = self._capture(model, pair.corrupt_ids, components)
        clean_metric = float(
            _metric_value(metric, model(input_ids=pair.clean_ids).logits, pair, side="clean")
        )
        corrupt_metric = float(
            _metric_value(metric, model(input_ids=pair.corrupt_ids).logits, pair, side="corrupt")
        )
        identity_metric = float(
            _metric_value(
                metric,
                self._run_patched(model, pair.clean_ids, clean_cache),
                pair,
                side="clean",
            )
        )
        full_corruption_metric = float(
            _metric_value(
                metric,
                self._run_patched(model, pair.clean_ids, corrupt_cache),
                pair,
                side="clean",
            )
        )
        identity_error = abs(identity_metric - clean_metric)
        full_corruption_error = abs(full_corruption_metric - corrupt_metric)
        scale = max(1.0, abs(clean_metric), abs(corrupt_metric))
        return {
            "identity_error": identity_error,
            "identity_passed": identity_error <= tolerance * scale,
            "full_corruption_error": full_corruption_error,
            "full_corruption_passed": full_corruption_error <= tolerance * scale,
            "tolerance": tolerance,
        }
