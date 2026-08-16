"""Freeze the exact-reward marginal before random-reward GRPO begins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_circuits.core.hashing import sha256_file
from posttrain_circuits.rewards.random_matched import (
    build_random_reward_calibration,
    write_random_reward_calibration,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze random-reward positive marginal")
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.source_artifact.read_text(encoding="utf-8"))
    distribution = source.get("reward_distribution", {})
    positive = int(distribution.get("positive", -1))
    total = int(source.get("total_trajectories", -1))
    artifact = build_random_reward_calibration(
        positive_count=positive,
        total_count=total,
        source_artifact_hash=sha256_file(args.source_artifact),
        seed=args.seed,
    )
    write_random_reward_calibration(args.output, artifact)


if __name__ == "__main__":
    main()
