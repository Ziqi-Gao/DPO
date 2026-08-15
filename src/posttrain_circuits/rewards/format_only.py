"""Parseability-only reward with no access to examples or answers."""

from __future__ import annotations

from collections.abc import Sequence

from posttrain_circuits.tasks.proofgraph.parser import parse_response


class FormatOnlyReward:
    def __call__(self, prompts: Sequence[str], completions: Sequence[str], **kwargs: object) -> list[float]:
        del prompts, kwargs
        return [float(parse_response(completion).parse_valid) for completion in completions]
