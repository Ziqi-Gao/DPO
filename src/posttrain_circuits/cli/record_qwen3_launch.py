"""Write the immutable, environment-resolved launch contract for a Qwen3 job."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.provenance import require_git_output

REQUIRED_ENV = (
    "MODEL_CONFIG",
    "TEACHER_CONFIG",
    "PRODUCTION_CONFIG",
    "G0_CONFIG",
    "PILOT_CONFIG",
    "PROJECT_ROOT",
    "PYTHON_BIN",
    "ACCELERATE_BIN",
    "OUTPUT_ROOT",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record a Qwen3 launch contract")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Qwen3 launch requires explicit environment variables: {missing}")
    environment = {name: os.environ[name] for name in REQUIRED_ENV}
    expected_configs = {
        "MODEL_CONFIG": "qwen3_1p7b",
        "TEACHER_CONFIG": "qwen3_teacher_8b",
        "PRODUCTION_CONFIG": "qwen3_primary",
        "G0_CONFIG": "qwen3_eap_separation",
        "PILOT_CONFIG": "qwen3_core",
    }
    wrong = {
        name: {"expected": expected, "observed": environment[name]}
        for name, expected in expected_configs.items()
        if environment[name] != expected
    }
    if wrong:
        raise RuntimeError(f"Qwen3 launch config mismatch: {wrong}")
    project_root = Path(environment["PROJECT_ROOT"]).resolve()
    if project_root != Path.cwd().resolve():
        raise RuntimeError("Qwen3 PROJECT_ROOT does not match the launch checkout")
    if require_git_output(["status", "--porcelain"]):
        raise RuntimeError("Qwen3 formal launch refuses a dirty source checkout")
    for name in ("PYTHON_BIN", "ACCELERATE_BIN"):
        if not Path(environment[name]).is_file():
            raise RuntimeError(f"Qwen3 {name} is not a file: {environment[name]}")
    output_root = Path(environment["OUTPUT_ROOT"])
    if "qwen3-v1" not in output_root.parts:
        raise RuntimeError("Qwen3 OUTPUT_ROOT must use the qwen3-v1 namespace")
    overrides = [
        f"production={environment['PRODUCTION_CONFIG']}",
        f"model={environment['MODEL_CONFIG']}",
        f"teacher={environment['TEACHER_CONFIG']}",
        f"g0={environment['G0_CONFIG']}",
        f"pilot={environment['PILOT_CONFIG']}",
        f"output_root={environment['OUTPUT_ROOT']}",
    ]
    config = compose_config(overrides)
    prereg_path = Path(str(config["prereg_path"]))
    payload: dict[str, Any] = {
        "format_version": 1,
        "phase": args.phase,
        "protocol_track": "qwen3_v1",
        "artifact_namespace": "qwen3-v1",
        "launch_environment": environment,
        "git_commit": require_git_output(["rev-parse", "HEAD"]),
        "prereg_path": str(prereg_path),
        "prereg_sha256": sha256_file(prereg_path),
        "prereg_commit": require_git_output(["log", "-n", "1", "--format=%H", "--", str(prereg_path)]),
        "resolved_config_sha256": sha256_value(config),
        "model_revision": str(config["model"]["model_revision"]),
        "teacher_revision": str(config["teacher"]["model_revision"]),
        "prompt_protocol": str(config["model"]["prompt_protocol"]["name"]),
        "enable_thinking": bool(config["model"]["prompt_protocol"]["enable_thinking"]),
        "chat_template_sha256": str(config["model"]["prompt_protocol"]["chat_template_sha256"]),
        "tokenizer_fingerprint": str(config["model"]["tokenizer_fingerprint"]),
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
