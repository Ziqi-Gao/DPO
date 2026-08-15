"""Build fixed greater-than, addition, and entity-tracking anchor sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_circuits.tasks.anchors import (
    build_fixed_anchor_pilots,
    write_anchor_pilots,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed anchor discovery/validation sets",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discovery-per-task", type=int, default=32)
    parser.add_argument("--validation-per-task", type=int, default=64)
    args = parser.parse_args(argv)
    splits = build_fixed_anchor_pilots(
        seed=args.seed,
        discovery_per_task=args.discovery_per_task,
        validation_per_task=args.validation_per_task,
    )
    manifest = write_anchor_pilots(
        args.output_dir,
        splits,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "manifest.json"),
                "tasks": manifest["tasks"],
                "discovery_examples": manifest["splits"]["discovery"]["example_count"],
                "validation_examples": manifest["splits"]["validation"]["example_count"],
                "base_accuracy_status": "not_evaluated",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
