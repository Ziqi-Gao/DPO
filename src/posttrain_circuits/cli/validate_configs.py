"""Validate that all six factorial cells resolve to their actual components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.methods.registry import FACTORIAL_METHOD_SPECS


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
    tracks = {
        "qwen2p5_core_v2": [],
        "qwen3_v1": [
            "production=qwen3_primary",
            "model=qwen3_1p7b",
            "teacher=qwen3_teacher_8b",
            "g0=qwen3_eap_separation",
            "pilot=qwen3_core",
        ],
        "qwen3_v2": [
            "production=qwen3_v2_primary",
            "model=qwen3_v2_1p7b",
            "teacher=qwen3_v2_teacher_8b",
            "g0=qwen3_v2_eap_separation",
            "pilot=qwen3_v2_core",
        ],
    }
    for track, track_overrides in tracks.items():
        for method in FACTORIAL_METHOD_SPECS:
            experiment = method.method_id
            config = compose_config(
                [*track_overrides, f"experiment={experiment}"],
                config_root=args.config_root,
            )
            actual = (
                config["state_source"]["name"],
                config["supervision"]["name"],
            )
            expected = (method.state_source, method.supervision)
            if actual != expected:
                raise RuntimeError(
                    f"{track}/{experiment} resolved {actual}, expected {expected}",
                )
            method.validate_resolved_config(config)
            rows.append(
                {
                    "track": track,
                    "experiment": experiment,
                    "state_source": actual[0],
                    "supervision": actual[1],
                    "model": config["model"].get("model_name_or_path", config["model"].get("model_id")),
                    "teacher": config["teacher"].get(
                        "model_name_or_path", config["teacher"].get("teacher_id")
                    ),
                    "method_spec_sha256": method.scientific_sha256,
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
