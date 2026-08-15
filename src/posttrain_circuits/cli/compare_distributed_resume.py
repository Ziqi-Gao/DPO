"""Compare two independent four-rank resumes from the same checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.training.local_fork import state_hash


def _last_metric(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"resume metrics are empty: {path}")
    return rows[-1]


def _objective_loss(row: dict[str, object]) -> float:
    values = [
        float(value) for key, value in row.items() if key.endswith("_loss") and isinstance(value, int | float)
    ]
    if len(values) != 1:
        raise ValueError(f"expected exactly one objective loss metric, observed {len(values)}")
    return values[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare independent distributed resumes")
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--metrics-a", type=Path, required=True)
    parser.add_argument("--metrics-b", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    left = torch.load(args.checkpoint_a, map_location="cpu", weights_only=False)
    right = torch.load(args.checkpoint_b, map_location="cpu", weights_only=False)
    left_metric = _last_metric(args.metrics_a)
    right_metric = _last_metric(args.metrics_b)
    loss_error = abs(_objective_loss(left_metric) - _objective_loss(right_metric))
    checks = {
        "world_size_four": args.world_size == 4,
        "model_state_identical": state_hash(left["model"]) == state_hash(right["model"]),
        "prompt_scheduler_identical": left["prompt_scheduler"] == right["prompt_scheduler"],
        "state_source_identical": left["state_source"] == right["state_source"],
        "policy_version_identical": left["policy_version"] == right["policy_version"],
        "rollout_round_identical": left["online_rollout_round"] == right["online_rollout_round"],
        "next_loss_within_tolerance": loss_error <= args.tolerance,
    }
    atomic_write_json(
        args.output,
        {
            "passed": all(checks.values()),
            "world_size": args.world_size,
            "checks": checks,
            "next_loss_absolute_error": loss_error,
            "tolerance": args.tolerance,
        },
    )
    if not all(checks.values()):
        raise SystemExit("distributed checkpoint/resume comparison failed")


if __name__ == "__main__":
    main()
