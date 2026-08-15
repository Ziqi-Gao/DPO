"""Circuit-backend protocol."""

from __future__ import annotations

from typing import Any, Protocol

from posttrain_circuits.core.types import CounterfactualPair


class CircuitBackend(Protocol):
    def score_components(self, model: Any, pairs: list[CounterfactualPair], metric: Any) -> Any: ...

    def evaluate_mask(
        self,
        model: Any,
        pairs: list[CounterfactualPair],
        mask: Any,
        ablation: Any,
    ) -> Any: ...
