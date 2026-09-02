"""Create a single-seed pilot manifest only from a passed G0 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.compatibility import (
    require_scientific_artifact,
    scientific_compatibility_fields,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json, utc_now
from posttrain_circuits.artifacts.runs import formal_artifact_binding, require_git_output
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.experiments.protocols.specs import CONTROLLED_FACTORIAL
from posttrain_circuits.methods.registry import (
    ANCHOR_METHOD_IDS,
    FACTORIAL_METHOD_IDS,
    get_method_spec,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the gated Qwen core pilot")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    factorial_cells = tuple(map(str, config.get("pilot", {}).get("factorial_cells", ())))
    anchors = tuple(map(str, config.get("pilot", {}).get("anchors", ())))
    if factorial_cells != FACTORIAL_METHOD_IDS or anchors != ANCHOR_METHOD_IDS:
        raise ValueError(
            "pilot method matrix differs from the canonical registry: "
            f"factorial={factorial_cells}, anchors={anchors}"
        )
    formal = formal_artifact_binding(config)
    g0 = json.loads(args.g0.read_text(encoding="utf-8"))
    expected = g0.pop("sha256", None)
    if expected != sha256_value(g0) or g0.get("passed") is not True:
        raise RuntimeError("pilot launch requires a hash-valid G0 artifact with passed=true")
    g0["sha256"] = expected
    require_scientific_artifact(
        g0,
        expected_prereg_version=str(config["prereg_version"]),
        require_circuit_schema=True,
        require_hash=True,
    )
    git_commit = require_git_output(["rev-parse", "HEAD"])
    if g0.get("git_commit") != git_commit:
        raise RuntimeError("pilot launch requires the exact Git commit validated by G0")
    prereg_path = str(config["prereg_path"])
    binding_keys = (
        "protocol_track",
        "artifact_namespace",
        "model_revision",
        "teacher_revision",
        "tokenizer_revision",
        "prompt_protocol",
        "enable_thinking",
        "chat_template_sha256",
        "tokenizer_fingerprint",
        "prereg_path",
        "prereg_commit",
        "prereg_sha256",
        "code_commit",
    )
    mismatch = {
        key: {"expected": formal[key], "observed": g0.get(key)}
        for key in binding_keys
        if g0.get(key) != formal[key]
    }
    if mismatch:
        raise RuntimeError(f"pilot launch refuses a cross-protocol or stale G0 artifact: {mismatch}")
    payload: dict[str, Any] = {
        "format_version": 2,
        **scientific_compatibility_fields(str(config["prereg_version"])),
        "phase": "single_seed_qwen_core_pilot",
        "status": "prepared",
        "seed": 42,
        "full_factorial": False,
        "cells": list(factorial_cells),
        "anchors": list(anchors),
        "factorial_design_sha256": CONTROLLED_FACTORIAL.scientific_sha256,
        "method_spec_hashes": {
            method_id: get_method_spec(method_id).scientific_sha256
            for method_id in (*factorial_cells, *anchors)
        },
        "g0_path": str(args.g0.resolve()),
        "g0_sha256": sha256_file(args.g0),
        "git_commit": git_commit,
        "protocol_track": str(config.get("protocol_track", "core_v2")),
        "artifact_namespace": str(config["model"].get("artifact_namespace", "legacy")),
        "prereg_path": prereg_path,
        "prereg_sha256": sha256_file(Path(prereg_path)),
        "prereg_commit": require_git_output(["log", "-n", "1", "--format=%H", "--", prereg_path]),
        "resolved_config_sha256": sha256_value(config),
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
