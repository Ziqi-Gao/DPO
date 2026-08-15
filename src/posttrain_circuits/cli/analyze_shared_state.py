"""Analyze shared canonical prefixes and separate natural rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posttrain_circuits.analysis.shared_state import (
    load_shared_state_partition,
    validate_shared_state_manifest,
)
from posttrain_circuits.analysis.stage7 import (
    SharedStateObservation,
    analyze_shared_state,
)
from posttrain_circuits.core.manifests import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 7 shared-state inference",
    )
    parser.add_argument(
        "--shared-state-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--observations",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_observations(path: Path) -> list[SharedStateObservation]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("observations") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("observation artifact must contain a non-empty row list")
    return [SharedStateObservation(**row) for row in rows]


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = validate_shared_state_manifest(args.shared_state_manifest)
    observations = _load_observations(args.observations)
    allowed_ids = {}
    for source_mode, entry in manifest["partitions"].items():
        records = load_shared_state_partition(
            args.shared_state_manifest.parent / entry["file"],
            expected_mode=source_mode,
        )
        allowed_ids[source_mode] = {record.record_id for record in records}
    unknown = sorted(
        {
            (observation.source_mode, observation.record_id)
            for observation in observations
            if observation.record_id not in allowed_ids[observation.source_mode]
        }
    )
    if unknown:
        raise ValueError(
            f"observations are not bound to the shared-state manifest: {unknown[:5]}",
        )
    result = analyze_shared_state(
        observations,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result["shared_state_manifest_sha256"] = manifest["manifest_sha256"]
    result["observation_artifact"] = str(args.observations)
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "canonical_observations": result["canonical_prefix"]["observation_count"],
                "natural_observations": result["natural_rollout"]["observation_count"],
                "source_modes_pooled": result["source_modes_pooled"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
