"""Read-only environment/provenance gate before submitting G0."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from posttrain_circuits.circuits.mib_eap_ig import MIB_REVISION
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.provenance import require_git_output, resolve_preregistration
from posttrain_circuits.core.scientific_versions import scientific_compatibility_fields


def _git(args: list[str]) -> str:
    return require_git_output(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preflight real Qwen G0")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    checks: dict[str, bool] = {}
    blockers: list[str] = []
    try:
        git_commit = _git(["rev-parse", "HEAD"])
        checks["git_repository"] = True
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        git_commit = "unavailable"
        checks["git_repository"] = False
        blockers.append("Git repository/HEAD is unavailable")
    dirty = _git(["status", "--porcelain"]) if checks["git_repository"] else "unavailable"
    checks["clean_git"] = dirty == ""
    if dirty:
        blockers.append("working tree is not clean")
    prereg = resolve_preregistration(config)
    prereg_path = str(prereg.path)
    prereg_version = prereg.version
    prereg_commit = prereg.git_commit if checks["git_repository"] else "unavailable"
    checks["frozen_prereg"] = bool(prereg_commit)
    if not prereg_commit:
        blockers.append(f"{prereg_path} has no frozen commit")
    mib_value = os.environ.get("MIB_REPOSITORY", "")
    mib_root = Path(mib_value) if mib_value else None
    mib_revision = "unavailable"
    if mib_root is not None and mib_root.is_dir():
        with contextlib.suppress(OSError, subprocess.CalledProcessError):
            external_git_environment = {
                key: value for key, value in os.environ.items() if key not in {"GIT_DIR", "GIT_WORK_TREE"}
            }
            mib_revision = subprocess.check_output(
                ["git", "-C", str(mib_root), "rev-parse", "HEAD"],
                text=True,
                env=external_git_environment,
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
    checks["cuda_torch_build"] = torch.version.cuda is not None
    if not checks["cuda_torch_build"]:
        blockers.append("the selected Python has a CPU-only torch build")
    checks["accelerate_available"] = importlib.util.find_spec("accelerate") is not None
    checks["transformer_lens_available"] = importlib.util.find_spec("transformer_lens") is not None
    if not checks["accelerate_available"]:
        blockers.append("accelerate is unavailable in the selected Python")
    if not checks["transformer_lens_available"]:
        blockers.append("transformer_lens is unavailable in the selected Python")
    requested_python = os.environ.get("PYTHON_BIN", "")
    checks["selected_python_matches_preflight"] = (
        bool(requested_python) and Path(requested_python).resolve() == Path(sys.executable).resolve()
    )
    if not checks["selected_python_matches_preflight"]:
        blockers.append(
            f"PYTHON_BIN does not match the preflight interpreter: {requested_python!r} != {sys.executable!r}"
        )
    checks["accelerate_executable"] = bool(
        os.environ.get("ACCELERATE_BIN") and Path(os.environ["ACCELERATE_BIN"]).is_file()
    )
    if not checks["accelerate_executable"]:
        blockers.append("ACCELERATE_BIN is unset or is not a file")
    gpu_error = ""
    try:
        gpu_preflight = json.loads(args.gpu_preflight.read_text(encoding="utf-8"))
        gpu_digest = gpu_preflight.pop("sha256", None)
        checks["gpu_preflight_hash"] = gpu_digest == sha256_value(gpu_preflight)
        checks["gpu_preflight_passed"] = gpu_preflight.get("passed") is True
        checks["gpu_preflight_world_size"] = int(gpu_preflight.get("world_size", 0)) == 4
        checks["gpu_preflight_git_commit"] = gpu_preflight.get("git_commit") == git_commit
        checks["gpu_preflight_model_revision"] = (
            gpu_preflight.get("model_revision") == config["model"]["model_revision"]
        )
        if str(config.get("protocol_track", "")).startswith("qwen3_"):
            namespace = str(config["model"]["artifact_namespace"])
            checks["gpu_preflight_teacher_revision"] = (
                gpu_preflight.get("teacher_revision") == config["teacher"]["model_revision"]
                and gpu_preflight.get("resolved_teacher_commit") == config["teacher"]["model_revision"]
            )
            checks["gpu_preflight_protocol"] = (
                gpu_preflight.get("protocol_track") == config["protocol_track"]
                and gpu_preflight.get("artifact_namespace") == namespace
                and gpu_preflight.get("prompt_protocol") == "qwen3_non_thinking_v1"
                and gpu_preflight.get("enable_thinking") is False
                and gpu_preflight.get("tokenizer_revision") == config["model"]["tokenizer_revision"]
                and gpu_preflight.get("tokenizer_fingerprint") == config["model"]["tokenizer_fingerprint"]
                and gpu_preflight.get("chat_template_sha256")
                == config["model"]["prompt_protocol"]["chat_template_sha256"]
                and gpu_preflight.get("prereg_path") == prereg_path
                and gpu_preflight.get("prereg_version") == prereg_version
                and gpu_preflight.get("prereg_sha256") == sha256_file(Path(prereg_path))
                and gpu_preflight.get("prereg_commit") == prereg_commit
            )
            rank_rows = gpu_preflight.get("rank_training_checks", [])
            checks["gpu_preflight_real_training_path"] = (
                isinstance(rank_rows, list)
                and len(rank_rows) == 4
                and gpu_preflight.get("rank_prompt_hashes_unique") is True
                and gpu_preflight.get("rank_zero_teacher_load_count") == 1
                and gpu_preflight.get("cgroup_memory", {}).get("passed") is True
                and all(
                    row.get("teacher_forward_finite") is True
                    and row.get("student_forward_finite") is True
                    and row.get("soft_teacher_loss_finite") is True
                    and row.get("gradients_finite") is True
                    and row.get("parameter_update_nonzero") is True
                    and row.get("fsdp_save_resume") is True
                    for row in rank_rows
                )
            )
        gpu_preflight["sha256"] = gpu_digest
        required_gpu_checks = [
            "gpu_preflight_hash",
            "gpu_preflight_passed",
            "gpu_preflight_world_size",
            "gpu_preflight_git_commit",
            "gpu_preflight_model_revision",
        ]
        if str(config.get("protocol_track", "")).startswith("qwen3_"):
            required_gpu_checks.extend(
                [
                    "gpu_preflight_teacher_revision",
                    "gpu_preflight_protocol",
                    "gpu_preflight_real_training_path",
                ]
            )
        if not all(checks[name] for name in required_gpu_checks):
            gpu_error = "GPU preflight is invalid, failed, or bound to different code/model inputs"
            blockers.append(gpu_error)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        gpu_error = str(exc)
        for name in (
            "gpu_preflight_hash",
            "gpu_preflight_passed",
            "gpu_preflight_world_size",
            "gpu_preflight_git_commit",
            "gpu_preflight_model_revision",
        ):
            checks[name] = False
        if str(config.get("protocol_track", "")).startswith("qwen3_"):
            for name in (
                "gpu_preflight_teacher_revision",
                "gpu_preflight_protocol",
                "gpu_preflight_real_training_path",
            ):
                checks[name] = False
        blockers.append(f"GPU preflight is unreadable: {gpu_error}")
    cache_error = ""
    try:
        for section in ("model", "teacher"):
            spec = config[section]
            AutoConfig.from_pretrained(
                str(spec["model_name_or_path"]),
                revision=str(spec["model_revision"]),
                local_files_only=True,
                trust_remote_code=bool(spec["trust_remote_code"]),
            )
            AutoTokenizer.from_pretrained(
                str(spec["tokenizer_name_or_path"]),
                revision=str(spec["tokenizer_revision"]),
                local_files_only=True,
                trust_remote_code=bool(spec["trust_remote_code"]),
            )
        checks["pinned_model_cache"] = True
    except (OSError, ValueError, KeyError) as exc:
        checks["pinned_model_cache"] = False
        cache_error = str(exc)
        blockers.append(f"pinned model/tokenizer cache is incomplete: {cache_error}")
    payload: dict[str, Any] = {
        "format_version": 2,
        **scientific_compatibility_fields(prereg_version),
        "phase": "G0_preflight",
        "passed": all(checks.values()),
        "checks": checks,
        "external_blockers": blockers,
        "git_commit": git_commit,
        "prereg_commit": prereg_commit,
        "prereg_path": prereg_path,
        "prereg_version": prereg_version,
        "prereg_sha256": sha256_file(Path(prereg_path)),
        "protocol_track": str(config.get("protocol_track", "core_v2")),
        "artifact_namespace": str(config["model"].get("artifact_namespace", "legacy")),
        "mib_revision": mib_revision,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "hf_home": os.environ.get("HF_HOME"),
        "model_cache_error": cache_error or None,
        "gpu_preflight_path": str(args.gpu_preflight.resolve()),
        "gpu_preflight_file_sha256": sha256_file(args.gpu_preflight),
        "gpu_preflight_error": gpu_error or None,
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
