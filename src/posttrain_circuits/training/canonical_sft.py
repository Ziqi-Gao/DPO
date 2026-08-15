"""Canonical response-token CE over verified teacher demonstrations."""

from __future__ import annotations

from typing import Any

from posttrain_circuits.supervision.verified_replay import VerifiedReplaySupervisor


class CanonicalSFTSupervisor(VerifiedReplaySupervisor):
    """SFT is replay where every sequence is a verified teacher demonstration."""

    def prepare_targets(self, trajectories, teacher: Any, verifier: Any):  # type: ignore[no-untyped-def]
        if any(record.verifier_reward != 1.0 for record in trajectories.records):
            raise ValueError("canonical SFT accepts only exact-verifier-success teacher demonstrations")
        return super().prepare_targets(trajectories, teacher, verifier)
