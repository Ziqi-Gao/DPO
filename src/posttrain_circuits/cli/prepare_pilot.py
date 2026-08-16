"""Create a single-seed pilot manifest only from a passed G0 artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the gated Qwen core pilot")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    g0 = json.loads(args.g0.read_text(encoding="utf-8"))
    expected = g0.pop("sha256", None)
    if expected != sha256_value(g0) or g0.get("passed") is not True:
        raise RuntimeError("pilot launch requires a hash-valid G0 artifact with passed=true")
    g0["sha256"] = expected
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if g0.get("git_commit") != git_commit:
        raise RuntimeError("pilot launch requires the exact Git commit validated by G0")
    config = compose_config(args.overrides)
    payload: dict[str, Any] = {
        "phase": "single_seed_qwen_core_pilot",
        "status": "prepared",
        "seed": 42,
        "full_factorial": False,
        "cells": config["pilot"]["factorial_cells"],
        "anchors": config["pilot"]["anchors"],
        "g0_path": str(args.g0.resolve()),
        "g0_sha256": sha256_file(args.g0),
        "git_commit": git_commit,
        "prereg_commit": subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--", "prereg/core_v1.yaml"],
            text=True,
        ).strip(),
        "resolved_config_sha256": sha256_value(config),
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
