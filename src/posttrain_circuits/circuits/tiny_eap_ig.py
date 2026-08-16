"""Genuine activation-space EAP-IG for deterministic tiny CPU smoke."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from posttrain_circuits.circuits.exact_patching import (
    ExactPatchingBackend,
    ExactTokenPair,
    _metric_value,
    component_specs,
)
from posttrain_circuits.circuits.graph import CircuitScores
from posttrain_circuits.circuits.probes import TargetSequenceMetric

BehaviorMetric = Callable[[torch.Tensor], torch.Tensor] | TargetSequenceMetric


@dataclass(frozen=True)
class TinyEapIgConfig:
    integrated_gradient_steps: int = 5

    def __post_init__(self) -> None:
        if self.integrated_gradient_steps < 2:
            raise ValueError("tiny EAP-IG requires at least two IG steps")


class TinyEapIgBackend:
    version = "tiny-hf-eap-ig-activations-v1"
    method = "EAP-IG-activations"

    def __init__(
        self,
        pairs: list[ExactTokenPair],
        *,
        integrated_gradient_steps: int,
    ) -> None:
        if len(pairs) < 2:
            raise ValueError("tiny EAP-IG uncertainty requires at least two pairs")
        self.pairs = pairs
        self.config = TinyEapIgConfig(integrated_gradient_steps)

    def _pair_scores(
        self,
        model: Any,
        pair: ExactTokenPair,
        metric: BehaviorMetric,
    ) -> dict[str, float]:
        exact = ExactPatchingBackend(pair)
        names = tuple(component_specs(model))
        with torch.no_grad():
            clean_cache = exact._capture(
                model,
                pair.clean_ids,
                names,
            )
            corrupt_cache = exact._capture(
                model,
                pair.corrupt_ids,
                names,
            )
        scores = {}
        for name in names:
            clean = clean_cache[name]
            corrupt = corrupt_cache[name]
            delta = clean - corrupt
            gradients = []
            for step in range(
                1,
                self.config.integrated_gradient_steps + 1,
            ):
                alpha = step / self.config.integrated_gradient_steps
                interpolated = (corrupt + alpha * delta).detach().requires_grad_(True)
                logits = exact._run_patched(
                    model,
                    pair.clean_ids,
                    {name: interpolated},
                )
                objective = _metric_value(metric, logits, pair, side="clean")
                gradient = torch.autograd.grad(
                    objective,
                    interpolated,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                gradients.append(gradient.detach())
            mean_gradient = torch.stack(gradients).mean(dim=0)
            scores[name] = float((delta.float() * mean_gradient.float()).sum())
        return scores

    def score_all_components(
        self,
        model: Any,
        metric: BehaviorMetric,
    ) -> CircuitScores:
        model.eval()
        per_pair = [self._pair_scores(model, pair, metric) for pair in self.pairs]
        names = tuple(per_pair[0])
        scores = {name: sum(pair[name] for pair in per_pair) / len(per_pair) for name in names}
        uncertainty = {}
        for name in names:
            values = [pair[name] for pair in per_pair]
            uncertainty[name] = statistics.stdev(values) / math.sqrt(len(values))
        return CircuitScores(
            scores=scores,
            uncertainty=uncertainty,
            node_scores=scores,
        )
