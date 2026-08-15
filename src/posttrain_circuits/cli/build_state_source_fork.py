"""Build the independently matched state-source fork artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from posttrain_circuits.cli._common import print_json
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.data.trajectory_store import TrajectoryStore
from posttrain_circuits.training.local_fork import (
    match_state_source_forks,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Match common-behavior, initial-student, current-checkpoint, and teacher-policy trajectories"
        )
    )
    parser.add_argument("--common-behavior", type=Path, required=True)
    parser.add_argument("--initial-student", type=Path, required=True)
    parser.add_argument(
        "--current-fork-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--teacher-policy", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/local_fork/state_source_fork.json"),
    )
    args = parser.parse_args(argv)
    source_paths = {
        "common_behavior": args.common_behavior,
        "initial_student": args.initial_student,
        "current_fork_checkpoint": args.current_fork_checkpoint,
        "teacher_policy": args.teacher_policy,
    }
    records = {}
    manifests = {}
    for name, path in source_paths.items():
        store = TrajectoryStore(path)
        records[name] = store.read()
        manifests[name] = {
            "path": str(path),
            "sha256": store.check_integrity()["sha256"],
        }
    artifact = match_state_source_forks(records)
    artifact["sources"] = manifests
    atomic_write_json(args.output, artifact)
    print_json(
        {
            "output": str(args.output),
            "matched_count_per_source": artifact["matched_count_per_source"],
            "sha256": artifact["sha256"],
        }
    )


if __name__ == "__main__":
    main()
