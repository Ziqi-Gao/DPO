"""ProofGraph syntax and verification records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class Literal:
    atom: str
    negated: bool = False

    def __str__(self) -> str:
        return f"NOT {self.atom}" if self.negated else self.atom

    @classmethod
    def parse(cls, text: str) -> Literal:
        normalized = " ".join(text.strip().split())
        if normalized.startswith("NOT "):
            return cls(normalized[4:], True)
        if not normalized:
            raise ValueError("literal is empty")
        return cls(normalized)

    def flipped(self) -> Literal:
        return Literal(self.atom, not self.negated)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    antecedents: tuple[Literal, ...]
    consequent: Literal


@dataclass(frozen=True)
class ProofStep:
    step_id: str
    rule_id: str
    citations: tuple[str, ...]
    conclusion: Literal


@dataclass
class TaskExample:
    example_id: str
    facts: dict[str, Literal]
    rules: dict[str, Rule]
    query: Literal
    label: int
    canonical_proof: list[ProofStep]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedResponse:
    parse_valid: bool
    steps: list[ProofStep]
    answer: int | None
    raw_text: str
    error_code: str | None = None


@dataclass(frozen=True)
class StepVerification:
    step_id: str
    valid: bool
    error_code: str | None
    established: Literal | None


@dataclass
class VerificationResult:
    parse_valid: bool
    proof_valid: bool
    answer_correct: bool
    step_results: list[StepVerification]
    reward: float
    error_code: str | None
