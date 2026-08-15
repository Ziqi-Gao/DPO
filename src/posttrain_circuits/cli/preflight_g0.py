"""Read-only environment/provenance gate before submitting G0."""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from posttrain_circuits.circuits.mib_eap_ig import MIB_REVISION
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preflight real Qwen G0")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    checks: dict[str, bool] = {}
    blockers: list[str] = []
    try:
        git_commit = _git(["rev-parse", "HEAD"])
        checks["git_repository"] = True
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
        checks["git_repository"] = False
        blockers.append("Git repository/HEAD is unavailable")
    dirty = _git(["status", "--porcelain"]) if checks["git_repository"] else "unavailable"
    checks["clean_git"] = dirty == ""
    if dirty:
        blockers.append("working tree is not clean")
    prereg_commit = (
        _git(["log", "-n", "1", "--format=%H", "--", "prereg/core_v1.yaml"])
        if checks["git_repository"]
        else "unavailable"
    )
    checks["frozen_prereg"] = bool(prereg_commit)
    if not prereg_commit:
        blockers.append("prereg/core_v1.yaml has no frozen commit")
    mib_value = os.environ.get("MIB_REPOSITORY", "")
    mib_root = Path(mib_value) if mib_value else None
    mib_revision = "unavailable"
    if mib_root is not None and mib_root.is_dir():
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            mib_revision = subprocess.check_output(
                ["git", "-C", str(mib_root), "rev-parse", "HEAD"], text=True
            ).strip()
    checks["pinned_mib"] = mib_revision == MIB_REVISION
    if not checks["pinned_mib"]:
        blockers.append(f"MIB_REPOSITORY is absent or not pinned to {MIB_REVISION}; observed={mib_revision}")
    checks["slurm_client"] = shutil.which("sbatch") is not None
    checks["slurm_account"] = bool(os.environ.get("SLURM_ACCOUNT"))
    checks["slurm_gpu_partition"] = bool(
        os.environ.get("SLURM_GPU_PARTITION") or os.environ.get("SLURM_PARTITION")
    )
    if not checks["slurm_client"]:
        blockers.append("sbatch is unavailable")
    if not checks["slurm_account"]:
        blockers.append("SLURM_ACCOUNT is unset")
    if not checks["slurm_gpu_partition"]:
        blockers.append("SLURM_GPU_PARTITION/SLURM_PARTITION is unset")
    payload: dict[str, Any] = {
        "phase": "G0_preflight",
        "passed": all(checks.values()),
        "checks": checks,
        "external_blockers": blockers,
        "git_commit": git_commit,
        "prereg_commit": prereg_commit,
        "mib_revision": mib_revision,
        "resolved_config_sha256": sha256_value(config),
        "created_at": utc_now(),
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)
    args.output.with_suffix(".md").write_text(
        "# G0 preflight\n\n"
        f"Result: **{'PASS' if payload['passed'] else 'BLOCKED'}**\n\n"
        + "\n".join(f"- [{'x' if passed else ' '}] {name}" for name, passed in checks.items())
        + ("\n\n## Blockers\n\n" + "\n".join(f"- {item}" for item in blockers) if blockers else "")
        + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise SystemExit("G0 preflight blocked; no GPU job submitted")


if __name__ == "__main__":
    main()
