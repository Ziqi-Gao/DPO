"""Frozen pilot circuit matrix resolution."""

from __future__ import annotations

from itertools import product
from typing import Any

from posttrain_circuits.core.hashing import sha256_value

PILOT_CELLS = (
    "offline_hard",
    "online_hard",
    "offline_soft",
    "online_soft_opd",
    "offline_verified_replay",
    "online_verified_replay",
    "canonical_sft",
    "canonical_grpo",
)
PILOT_COHORTS = ("base_capable", "challenge")
PILOT_STAGE_MAP = {
    "process": "first_rule_selection",
    "final_answer": "final_answer",
}


def resolve_pilot_circuit_scope(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("pilot", {}).get("pilot_circuit_scope")
    if not isinstance(raw, dict):
        raise ValueError("pilot config has no pilot_circuit_scope")
    cells = tuple(map(str, raw.get("cells", ())))
    cohorts = tuple(map(str, raw.get("cohorts", ())))
    stages = tuple(
        (str(row.get("label")), str(row.get("probe_stage")))
        for row in raw.get("stages", ())
        if isinstance(row, dict)
    )
    if cells != PILOT_CELLS:
        raise ValueError(f"pilot circuit cells differ from the frozen eight-cell matrix: {cells}")
    if cohorts != PILOT_COHORTS:
        raise ValueError(f"pilot circuit cohorts differ from the frozen cohorts: {cohorts}")
    if dict(stages) != PILOT_STAGE_MAP or len(stages) != len(PILOT_STAGE_MAP):
        raise ValueError(f"pilot circuit stages differ from the frozen process/final map: {stages}")
    initial = [
        {"stage_label": label, "probe_stage": probe_stage, "cohort": cohort}
        for (label, probe_stage), cohort in product(stages, cohorts)
    ]
    final = [
        {
            "cell": cell,
            "stage_label": label,
            "probe_stage": probe_stage,
            "cohort": cohort,
        }
        for cell, (label, probe_stage), cohort in product(cells, stages, cohorts)
    ]
    payload: dict[str, Any] = {
        "cells": list(cells),
        "cohorts": list(cohorts),
        "stages": [{"label": label, "probe_stage": probe_stage} for label, probe_stage in stages],
        "initial_matrix": initial,
        "final_matrix": final,
        "initial_count": len(initial),
        "final_count": len(final),
    }
    payload["sha256"] = sha256_value(payload)
    return payload
