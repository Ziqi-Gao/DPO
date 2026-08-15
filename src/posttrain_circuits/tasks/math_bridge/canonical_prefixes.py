"""Canonical math solutions, counterfactuals, and next-token interventions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.tasks.math_bridge.answer_normalization import (
    normalize_answer,
)

CounterfactualKind = Literal[
    "digit",
    "operator",
    "equation",
    "final_answer",
]


@dataclass(frozen=True)
class CanonicalTeacherSolution:
    example_id: str
    prompt: str
    solution: str
    final_answer: str
    teacher_id: str
    teacher_revision: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.example_id,
                self.prompt,
                self.solution,
                self.final_answer,
                self.teacher_id,
                self.teacher_revision,
            )
        ):
            raise ValueError("canonical teacher solution fields must be non-empty")
        match = re.search(
            r"Final answer:\s*(.+?)\s*$",
            self.solution,
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError("canonical solution needs an explicit final-answer marker")
        if normalize_answer(match.group(1)) != normalize_answer(self.final_answer):
            raise ValueError("canonical solution and final answer disagree")


@dataclass(frozen=True)
class CanonicalPrefixTarget:
    prompt: str
    canonical_prefix: str
    next_token_role: str
    target_text: str


@dataclass(frozen=True)
class MathCounterfactual:
    kind: CounterfactualKind
    original_text: str
    intervened_text: str
    span_start: int
    span_end: int
    original_value: str
    replacement_value: str

    def __post_init__(self) -> None:
        if self.original_text == self.intervened_text:
            raise ValueError("counterfactual intervention must change the text")
        if not 0 <= self.span_start < self.span_end <= len(self.original_text):
            raise ValueError("counterfactual span lies outside the original text")
        if self.original_text[self.span_start : self.span_end] != self.original_value:
            raise ValueError("counterfactual span does not match its original value")


def make_prefix_target(
    prompt: str,
    solution: str,
    split_at: int,
    role: str,
) -> CanonicalPrefixTarget:
    if not 0 < split_at < len(solution):
        raise ValueError("split_at must lie inside the canonical solution")
    if not role:
        raise ValueError("next-token role must be non-empty")
    return CanonicalPrefixTarget(
        prompt,
        solution[:split_at],
        role,
        solution[split_at:],
    )


def _replace_span(
    text: str,
    *,
    kind: CounterfactualKind,
    start: int,
    end: int,
    replacement: str,
) -> MathCounterfactual:
    original = text[start:end]
    if not replacement or replacement == original:
        raise ValueError("counterfactual replacement must be non-empty and different")
    return MathCounterfactual(
        kind=kind,
        original_text=text,
        intervened_text=text[:start] + replacement + text[end:],
        span_start=start,
        span_end=end,
        original_value=original,
        replacement_value=replacement,
    )


def digit_counterfactual(
    solution: str,
    *,
    occurrence: int = 0,
    replacement: str | None = None,
) -> MathCounterfactual:
    matches = list(re.finditer(r"\d", solution))
    if not 0 <= occurrence < len(matches):
        raise ValueError("digit occurrence is outside the canonical solution")
    match = matches[occurrence]
    original = match.group(0)
    value = replacement if replacement is not None else str((int(original) + 1) % 10)
    if not re.fullmatch(r"\d", value):
        raise ValueError("digit replacement must be exactly one digit")
    return _replace_span(
        solution,
        kind="digit",
        start=match.start(),
        end=match.end(),
        replacement=value,
    )


def operator_counterfactual(
    solution: str,
    *,
    occurrence: int = 0,
    replacement: str | None = None,
) -> MathCounterfactual:
    matches = list(re.finditer(r"(?<=\s)[+\-*/](?=\s)", solution))
    if not 0 <= occurrence < len(matches):
        raise ValueError("operator occurrence is outside the canonical solution")
    match = matches[occurrence]
    alternatives = {
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
    }
    value = replacement if replacement is not None else alternatives[match.group(0)]
    if value not in {"+", "-", "*", "/"}:
        raise ValueError("unsupported operator replacement")
    return _replace_span(
        solution,
        kind="operator",
        start=match.start(),
        end=match.end(),
        replacement=value,
    )


def equation_counterfactual(
    solution: str,
    *,
    occurrence: int = 0,
    replacement: str | None = None,
) -> MathCounterfactual:
    matches = list(
        re.finditer(
            r"-?\d+\s*[+\-*/]\s*-?\d+\s*=\s*(-?\d+)",
            solution,
        )
    )
    if not 0 <= occurrence < len(matches):
        raise ValueError("equation occurrence is outside the canonical solution")
    result_span = matches[occurrence].span(1)
    original = solution[result_span[0] : result_span[1]]
    value = replacement if replacement is not None else str(int(original) + 1)
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        raise ValueError("equation result replacement must be numeric")
    return _replace_span(
        solution,
        kind="equation",
        start=result_span[0],
        end=result_span[1],
        replacement=value,
    )


def final_answer_counterfactual(
    canonical: CanonicalTeacherSolution,
    *,
    replacement: str,
) -> MathCounterfactual:
    marker = re.search(
        r"Final answer:\s*(.+?)\s*$",
        canonical.solution,
        flags=re.IGNORECASE,
    )
    if marker is None:
        raise ValueError("canonical solution needs an explicit final answer")
    return _replace_span(
        canonical.solution,
        kind="final_answer",
        start=marker.start(1),
        end=marker.end(1),
        replacement=replacement,
    )


def shared_prefix_next_token_metric(
    logits: torch.Tensor,
    *,
    target_token_id: int,
    contrast_token_id: int,
) -> torch.Tensor:
    """Logit difference at the next token after a shared canonical prefix."""
    if logits.ndim != 3 or logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError("next-token metric expects [batch, sequence, vocabulary] logits")
    vocabulary = logits.shape[-1]
    if not 0 <= target_token_id < vocabulary:
        raise ValueError("target token ID is outside the vocabulary")
    if not 0 <= contrast_token_id < vocabulary:
        raise ValueError("contrast token ID is outside the vocabulary")
    if target_token_id == contrast_token_id:
        raise ValueError("target and contrast token IDs must differ")
    final_logits = logits[:, -1, :]
    return (final_logits[:, target_token_id] - final_logits[:, contrast_token_id]).mean()


def write_canonical_teacher_solutions(
    path: Path,
    solutions: list[CanonicalTeacherSolution],
) -> dict[str, object]:
    if not solutions:
        raise ValueError("canonical teacher-solution artifact must not be empty")
    ids = [solution.example_id for solution in solutions]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical teacher-solution IDs must be unique")
    rows = [asdict(solution) for solution in solutions]
    payload: dict[str, object] = {
        "schema_version": 1,
        "solution_count": len(rows),
        "solutions_sha256": sha256_value(rows),
        "teacher_ids": sorted(
            {solution.teacher_id for solution in solutions},
        ),
        "teacher_revisions": sorted(
            {solution.teacher_revision for solution in solutions},
        ),
        "solutions": rows,
    }
    atomic_write_json(path, payload)
    return payload


def load_canonical_teacher_solutions(
    path: Path,
) -> list[CanonicalTeacherSolution]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("solutions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("canonical teacher-solution artifact is empty")
    if payload.get("solution_count") != len(rows):
        raise ValueError("canonical teacher-solution count mismatch")
    if payload.get("solutions_sha256") != sha256_value(rows):
        raise ValueError("canonical teacher-solution hash mismatch")
    solutions = [CanonicalTeacherSolution(**row) for row in rows]
    if payload.get("teacher_ids") != sorted(
        {solution.teacher_id for solution in solutions},
    ):
        raise ValueError("canonical teacher ID provenance mismatch")
    if payload.get("teacher_revisions") != sorted(
        {solution.teacher_revision for solution in solutions},
    ):
        raise ValueError("canonical teacher revision provenance mismatch")
    return solutions
