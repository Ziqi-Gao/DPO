"""Validate that all six factorial cells resolve to their actual components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json

FACTORIAL_CELLS = {
    "offline_hard": ("fixed_bank", "hard_teacher"),
    "online_hard": ("current_policy", "hard_teacher"),
    "offline_soft": ("fixed_bank", "soft_teacher"),
    "online_soft_opd": ("current_policy", "soft_teacher"),
    "offline_verified_replay": ("fixed_bank", "verified_replay"),
    "online_verified_replay": ("current_policy", "verified_replay"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate all factorial resolved configurations",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("configs"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows = []
    for experiment, (state_source, supervision) in FACTORIAL_CELLS.items():
        config = compose_config(
            [f"experiment={experiment}"],
            config_root=args.config_root,
        )
        actual = (
            config["state_source"]["name"],
            config["supervision"]["name"],
        )
        if actual != (state_source, supervision):
            raise RuntimeError(
                f"{experiment} resolved {actual}, expected {(state_source, supervision)}",
            )
        rows.append(
            {
                "experiment": experiment,
                "state_source": actual[0],
                "supervision": actual[1],
                "resolved_config_sha256": sha256_value(config),
            }
        )
    report = {
        "status": "passed",
        "cell_count": len(rows),
        "cells": rows,
    }
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
