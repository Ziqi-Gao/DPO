"""Build a hash-bound shared-state dataset with separate source modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posttrain_circuits.analysis.shared_state import (
    SharedStateRecord,
    write_partitioned_shared_state,
)


def _load_rows(path: Path, expected_mode: str) -> list[SharedStateRecord]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} contains no shared-state records")
    records = [SharedStateRecord(**row) for row in rows]
    if any(record.source_mode != expected_mode for record in records):
        raise ValueError(f"{path} contains a mixed or incorrect source mode")
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a partitioned shared-state dataset",
    )
    parser.add_argument("--canonical-records", type=Path, required=True)
    parser.add_argument("--natural-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = write_partitioned_shared_state(
        args.output_dir,
        canonical_prefix=_load_rows(
            args.canonical_records,
            "canonical_prefix",
        ),
        natural_rollout=_load_rows(
            args.natural_records,
            "natural_rollout",
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "partitions": {
                    source_mode: entry["record_count"]
                    for source_mode, entry in manifest["partitions"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
