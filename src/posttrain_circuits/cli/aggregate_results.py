"""Aggregate local JSONL metrics and planned contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_circuits.analysis.factorial import planned_contrasts
from posttrain_circuits.cli._common import print_json
from posttrain_circuits.core.manifests import atomic_write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate factorial run metrics")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--metric", default="parameter_update_norm")
    parser.add_argument("--output", type=Path, default=Path("outputs/aggregate/results.json"))
    args = parser.parse_args(argv)
    observations: list[dict[str, object]] = []
    values_by_cell: dict[str, list[float]] = {}
    for run_dir in args.run_dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        for row in rows:
            if args.metric not in row:
                continue
            value = float(row[args.metric])
            cell = str(manifest["experiment_cell"])
            values_by_cell.setdefault(cell, []).append(value)
            observations.append(
                {
                    "experiment_cell": cell,
                    "seed": int(manifest["seed"]),
                    "checkpoint": int(row.get("step", len(observations))),
                    "value": value,
                    "run_id": str(manifest["run_id"]),
                    "validation_manifest_hash": manifest.get("dataset_hashes", {}).get("validation_manifest"),
                }
            )
    cell_means = {cell: sum(values) / len(values) for cell, values in sorted(values_by_cell.items())}
    payload = {
        "metric": args.metric,
        "observations": observations,
        "observation_count": len(observations),
        "cell_means": cell_means,
        "contrasts": planned_contrasts(cell_means),
    }
    atomic_write_json(args.output, payload)
    print_json({"output": str(args.output), **payload})


if __name__ == "__main__":
    main()
