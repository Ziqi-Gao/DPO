"""Task protocol."""

from __future__ import annotations

from typing import Any, Protocol

from posttrain_circuits.core.types import CounterfactualPair


class Task(Protocol):
    def generate(self, seed: int, difficulty: dict[str, Any]) -> Any: ...

    def render(self, example: Any) -> str: ...

    def parse_response(self, text: str) -> Any: ...

    def verify(self, example: Any, response: Any) -> Any: ...

    def make_counterfactual(self, example: Any, corruption: str, seed: int) -> CounterfactualPair: ...
