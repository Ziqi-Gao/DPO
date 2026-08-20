"""Resolve and verify the final checkpoint recorded by a run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from posttrain_circuits.core.hashing import sha256_file
from posttrain_circuits.core.provenance import validate_run_manifest_payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = validate_run_manifest_payload(json.loads(args.manifest.read_text(encoding="utf-8")))
    checkpoint = Path(str(payload.get("final_checkpoint_path", "")))
    expected = str(payload.get("final_checkpoint_sha256", ""))
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
        raise ValueError("run manifest final checkpoint binding is missing or invalid")
    print(checkpoint)


if __name__ == "__main__":
    main()
