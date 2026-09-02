"""Pure data contracts for pair-complete circuit-probe cohorts."""

from __future__ import annotations

from dataclasses import dataclass

from posttrain_circuits.datasets.proofgraph.contracts import TaskExample

COHORTS = ("base_capable", "challenge")
SUBSETS = ("discovery", "validation")
SOURCE_SPLITS = {
    "discovery": "circuit_discovery",
    "validation": "circuit_validation",
}


@dataclass(frozen=True)
class ProbePair:
    """One complete signed ProofGraph pair in its frozen source order."""

    subset: str
    source_split: str
    pair_group_id: str
    examples: tuple[TaskExample, TaskExample]

    @property
    def example_ids(self) -> tuple[str, str]:
        return (self.examples[0].example_id, self.examples[1].example_id)


@dataclass(frozen=True)
class PairDecision:
    """Pair-level eligibility result; siblings can never receive separate decisions."""

    pair: ProbePair
    cohort: str | None
    exclusion_reason: str | None

    @property
    def selected(self) -> bool:
        return self.cohort is not None
