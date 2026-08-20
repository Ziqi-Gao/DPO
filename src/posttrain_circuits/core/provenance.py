"""Run-directory provenance capture."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json

PREREG_PATH = Path("prereg/core_v2.yaml")
PREREG_VERSION = "core_v2"

_LAUNCH_ENVIRONMENT_FIELDS = (
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


def _launch_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _LAUNCH_ENVIRONMENT_FIELDS}


def git_output(args: list[str]) -> str | None:
    """Run Git against either a conventional checkout or this repo's .opd-git metadata."""

    try:
        environment = None
        fallback = Path(".opd-git")
        if fallback.is_dir() and not (Path(".git") / "HEAD").is_file():
            import os

            environment = {
                **os.environ,
                "GIT_DIR": str(fallback.resolve()),
                "GIT_WORK_TREE": str(Path.cwd()),
            }
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL, env=environment
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def require_git_output(args: list[str]) -> str:
    """Return Git output or fail closed when repository provenance is unavailable."""

    value = git_output(args)
    if value is None:
        raise RuntimeError(f"Git provenance command failed: git {' '.join(args)}")
    return value


_git = git_output


def dependency_versions() -> dict[str, str]:
    """Return every installed distribution version, normalized by package name."""
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            continue
        if name:
            versions[str(name).lower().replace("_", "-")] = distribution.version
    return dict(sorted(versions.items()))


def _prereg_commit(path: Path = PREREG_PATH) -> str:
    return _git(["log", "-n", "1", "--format=%H", "--", str(path)]) or "unavailable"


def _prereg_sha256(path: Path = PREREG_PATH) -> str:
    return sha256_file(path) if path.is_file() else "unavailable"


def _prereg_dirty(path: Path = PREREG_PATH) -> bool:
    return bool(_git(["status", "--porcelain", "--", str(path)]) or "")


@dataclass
class RunManifest:
    run_id: str
    experiment_cell: str
    seed: int
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    resolved_model_commit: str
    resolved_tokenizer_commit: str
    dataset_hashes: dict[str, str]
    rollout_bank_hash: str
    prompt_schedule_hash: str
    raw_prompt_schedule_hash: str = "legacy-unrecorded"
    model_facing_prompt_schedule_hash: str = "legacy-unrecorded"
    prompt_protocol: str = "legacy_raw_v1"
    enable_thinking: bool = False
    chat_template_sha256: str = "legacy-unrecorded"
    tokenizer_fingerprint: str = "legacy-unrecorded"
    protocol_track: str = "core_v2"
    artifact_namespace: str = "legacy"
    teacher_id: str | None = None
    teacher_revision: str | None = None
    resolved_teacher_commit: str | None = None
    teacher_demo_generation: dict[str, Any] | None = None
    resume_ancestry: list[str] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_time: str | None = None
    git_commit: str = field(default_factory=lambda: _git(["rev-parse", "HEAD"]) or "unavailable")
    dirty_working_tree: bool = field(default_factory=lambda: bool(_git(["status", "--porcelain"]) or ""))
    package_versions: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    launch_environment: dict[str, str | None] = field(default_factory=_launch_environment)
    prereg_git_commit: str = field(default_factory=_prereg_commit)
    prereg_sha256: str = field(default_factory=_prereg_sha256)
    prereg_dirty: bool = field(default_factory=_prereg_dirty)
    prereg_version: str = PREREG_VERSION
    prereg_path: str = str(PREREG_PATH)

    def bind_preregistration(self, path: Path, *, version: str) -> None:
        self.prereg_path = str(path)
        self.prereg_version = version
        self.prereg_git_commit = _prereg_commit(path)
        self.prereg_sha256 = _prereg_sha256(path)
        self.prereg_dirty = _prereg_dirty(path)
        self.dirty_working_tree = bool(_git(["status", "--porcelain"]) or "")

    def validate(self, *, require_git: bool) -> None:
        required_text = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "resolved_model_commit": self.resolved_model_commit,
            "resolved_tokenizer_commit": self.resolved_tokenizer_commit,
            "rollout_bank_hash": self.rollout_bank_hash,
            "prompt_schedule_hash": self.prompt_schedule_hash,
        }
        missing = [name for name, value in required_text.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"run manifest has empty required fields: {missing}")
        if not self.dataset_hashes or any(not value for value in self.dataset_hashes.values()):
            raise ValueError("run manifest requires non-empty dataset hashes")
        if self.teacher_id is not None and (not self.teacher_revision or not self.resolved_teacher_commit):
            raise ValueError("teacher runs require teacher revision and resolved commit")
        if require_git and self.git_commit == "unavailable":
            raise RuntimeError(
                "formal run refused because Git provenance is unavailable; "
                "initialize the repository and commit the experiment source first"
            )
        if require_git and (self.prereg_git_commit == "unavailable" or self.prereg_sha256 == "unavailable"):
            raise RuntimeError("formal run refused because prereg/core_v2.yaml has no frozen Git commit")
        if require_git and self.prereg_dirty:
            raise RuntimeError(
                "formal run refused because prereg/core_v2.yaml differs from its frozen Git commit"
            )
        if require_git and self.dirty_working_tree:
            raise RuntimeError("formal run refused because the source working tree is dirty")
        if self.prereg_version not in {PREREG_VERSION, "qwen3_v1"}:
            raise RuntimeError("formal run refused because its preregistration version is unknown")
        if self.protocol_track == "qwen3_v1":
            required_qwen3 = {
                "prereg_version": self.prereg_version == "qwen3_v1",
                "prereg_path": self.prereg_path == "prereg/qwen3_v1.yaml",
                "prompt_protocol": self.prompt_protocol == "qwen3_non_thinking_v1",
                "thinking_disabled": self.enable_thinking is False,
                "chat_template": len(self.chat_template_sha256) == 64,
                "tokenizer_fingerprint": len(self.tokenizer_fingerprint) == 64,
                "raw_prompt_hash": len(self.raw_prompt_schedule_hash) == 64,
                "model_facing_prompt_hash": len(self.model_facing_prompt_schedule_hash) == 64,
                "namespace": self.artifact_namespace == "qwen3-v1",
                "launch_environment": all(
                    self.launch_environment.get(name) for name in _LAUNCH_ENVIRONMENT_FIELDS
                ),
            }
            failures = [name for name, passed in required_qwen3.items() if not passed]
            if failures:
                raise RuntimeError(f"Qwen3 manifest protocol bindings are incomplete: {failures}")


def run_manifest_payload(manifest: RunManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["sha256"] = sha256_value(payload)
    return payload


def validate_run_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != sha256_value(content):
        raise ValueError("run manifest SHA-256 mismatch")
    return payload


def initialize_run_directory(
    run_dir: Path,
    resolved_config: dict[str, Any],
    manifest: RunManifest,
    *,
    require_git: bool = False,
) -> None:
    prereg_path = Path(str(resolved_config.get("prereg_path", PREREG_PATH)))
    prereg_version = "qwen3_v1" if resolved_config.get("protocol_track") == "qwen3_v1" else PREREG_VERSION
    manifest.bind_preregistration(prereg_path, version=prereg_version)
    manifest.package_versions = dependency_versions()
    manifest.environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    manifest.validate(require_git=require_git)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "evaluations").mkdir(exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )
    atomic_write_json(run_dir / "manifest.json", run_manifest_payload(manifest))
    atomic_write_json(run_dir / "environment.json", manifest.environment)
    (run_dir / "metrics.jsonl").touch()
    diff = _git(["diff", "--binary"]) or ""
    (run_dir / "git_diff.patch").write_text(diff + ("\n" if diff else ""), encoding="utf-8")


def finalize_run_directory(run_dir: Path, manifest: RunManifest) -> None:
    manifest.end_time = datetime.now(UTC).isoformat()
    atomic_write_json(run_dir / "manifest.json", run_manifest_payload(manifest))


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
