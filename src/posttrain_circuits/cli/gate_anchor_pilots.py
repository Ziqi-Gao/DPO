"""Require base-model accuracy before anchor circuit analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.tasks.anchors import (
    AnchorExample,
    require_base_accuracy,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gate anchor analysis on base accuracy",
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    example_payload: Any = json.loads(
        args.examples.read_text(encoding="utf-8"),
    )
    rows = example_payload.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("anchor example artifact contains no examples")
    if example_payload.get("examples_sha256") != sha256_value(rows):
        raise ValueError("anchor example artifact hash mismatch")
    examples = [AnchorExample(**row) for row in rows]

    prediction_payload: Any = json.loads(
        args.predictions.read_text(encoding="utf-8"),
    )
    predictions = prediction_payload.get("predictions") if isinstance(prediction_payload, dict) else None
    if not isinstance(predictions, dict):
        raise ValueError("prediction artifact needs an ID-to-text mapping")
    result = require_base_accuracy(
        examples,
        {str(key): str(value) for key, value in predictions.items()},
        threshold=args.threshold,
    )
    evidence = {
        **asdict(result),
        "examples_sha256": example_payload["examples_sha256"],
        "predictions_sha256": sha256_value(predictions),
    }
    atomic_write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
