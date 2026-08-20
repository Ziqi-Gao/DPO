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


def _prereg_commit(path: Path) -> str:
    return _git(["log", "-n", "1", "--format=%H", "--", str(path)]) or "unavailable"


def _prereg_sha256(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "unavailable"


def _prereg_dirty(path: Path) -> bool:
    return bool(_git(["status", "--porcelain", "--", str(path)]) or "")


@dataclass(frozen=True)
class PreregistrationBinding:
    path: Path
    version: str
    git_commit: str
    sha256: str
    dirty: bool


def resolve_preregistration(config: dict[str, Any]) -> PreregistrationBinding:
    """Resolve the preregistration exclusively from the composed run config."""

    raw_path = str(config.get("prereg_path", "")).strip()
    version = str(config.get("prereg_version", "")).strip()
    if not raw_path or not version:
        raise ValueError("formal configuration requires prereg_path and prereg_version")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"configured preregistration does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or str(payload.get("version", "")) != version:
        raise ValueError(
            "configured preregistration version mismatch: "
            f"config={version!r}, file={payload.get('version')!r}, path={path}"
        )
    return PreregistrationBinding(
        path=path,
        version=version,
        git_commit=_prereg_commit(path),
        sha256=_prereg_sha256(path),
        dirty=_prereg_dirty(path),
    )


def formal_artifact_binding(config: dict[str, Any]) -> dict[str, Any]:
    """Return the complete model/protocol/prereg binding for a formal artifact."""

    prereg = resolve_preregistration(config)
    model = config.get("model", {})
    teacher = config.get("teacher", {})
    prompt = model.get("prompt_protocol", {})
    protocol_track = str(config.get("protocol_track", prereg.version))
    if protocol_track.startswith("qwen3_") and require_git_output(["status", "--porcelain"]):
        raise RuntimeError("Qwen3 formal artifact creation refuses a dirty source checkout")
    return {
        "protocol_track": protocol_track,
        "artifact_namespace": str(model.get("artifact_namespace", "legacy")),
        "model_revision": str(model.get("model_revision", "unavailable")),
        "teacher_revision": str(teacher.get("model_revision", "unavailable")),
        "tokenizer_revision": str(model.get("tokenizer_revision", "unavailable")),
        "tokenizer_fingerprint": str(model.get("tokenizer_fingerprint", "legacy-unrecorded")),
        "chat_template_sha256": str(prompt.get("chat_template_sha256", "legacy-unrecorded")),
        "prompt_protocol": str(prompt.get("name", "legacy_raw_v1")),
        "enable_thinking": bool(prompt.get("enable_thinking", False)),
        "code_commit": require_git_output(["rev-parse", "HEAD"]),
        "prereg_path": str(prereg.path),
        "prereg_version": prereg.version,
        "prereg_commit": prereg.git_commit,
        "prereg_sha256": prereg.sha256,
    }


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
    protocol_teacher_revision: str = "unbound"
    token_budget: int = 1024
    token_budget_unit: str = "global_nonpadding_model_input_tokens"
    token_budget_consumed: int = 0
    training_stop_reason: str | None = None
    metrics_sha256: str | None = None
    final_checkpoint_path: str | None = None
    final_checkpoint_sha256: str | None = None
    slurm_terminal_evidence_sha256: str | None = None
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
    prereg_git_commit: str = "unbound"
    prereg_sha256: str = "unbound"
    prereg_dirty: bool = True
    prereg_version: str = "unbound"
    prereg_path: str = "unbound"

    def bind_preregistration(self, binding: PreregistrationBinding) -> None:
        self.prereg_path = str(binding.path)
        self.prereg_version = binding.version
        self.prereg_git_commit = binding.git_commit
        self.prereg_sha256 = binding.sha256
        self.prereg_dirty = binding.dirty
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
        if self.token_budget < 1 or self.token_budget_unit != "global_nonpadding_model_input_tokens":
            raise ValueError("run manifest requires the registered global token budget and unit")
        if not 0 <= self.token_budget_consumed <= self.token_budget:
            raise ValueError("run manifest token consumption is outside its registered budget")
        if require_git and self.git_commit == "unavailable":
            raise RuntimeError(
                "formal run refused because Git provenance is unavailable; "
                "initialize the repository and commit the experiment source first"
            )
        if require_git and (self.prereg_git_commit == "unavailable" or self.prereg_sha256 == "unavailable"):
            raise RuntimeError(
                "formal run refused because its configured preregistration has no frozen Git commit"
            )
        if require_git and self.prereg_dirty:
            raise RuntimeError(
                "formal run refused because its configured preregistration differs from its frozen Git commit"
            )
        if require_git and self.dirty_working_tree:
            raise RuntimeError("formal run refused because the source working tree is dirty")
        if require_git or self.prereg_path != "unbound":
            prereg = Path(self.prereg_path)
            if not prereg.is_file():
                raise RuntimeError("formal run refused because its bound preregistration is missing")
            prereg_payload = yaml.safe_load(prereg.read_text(encoding="utf-8")) or {}
            if str(prereg_payload.get("version", "")) != self.prereg_version:
                raise RuntimeError("formal run refused because its preregistration version is inconsistent")
            if sha256_file(prereg) != self.prereg_sha256:
                raise RuntimeError("formal run refused because its preregistration SHA-256 changed")
            if require_git and _prereg_commit(prereg) != self.prereg_git_commit:
                raise RuntimeError("formal run refused because its preregistration Git commit changed")
        if self.protocol_track in {"qwen3_v1", "qwen3_v2"}:
            suffix = self.protocol_track.removeprefix("qwen3_")
            expected_namespace = f"qwen3-{suffix}"
            required_qwen3 = {
                "prereg_version": self.prereg_version == self.protocol_track,
                "prereg_path": self.prereg_path == f"prereg/{self.protocol_track}.yaml",
                "prompt_protocol": self.prompt_protocol == "qwen3_non_thinking_v1",
                "thinking_disabled": self.enable_thinking is False,
                "chat_template": len(self.chat_template_sha256) == 64,
                "tokenizer_fingerprint": len(self.tokenizer_fingerprint) == 64,
                "raw_prompt_hash": len(self.raw_prompt_schedule_hash) == 64,
                "model_facing_prompt_hash": len(self.model_facing_prompt_schedule_hash) == 64,
                "namespace": self.artifact_namespace == expected_namespace,
                "teacher_revision": len(self.protocol_teacher_revision) == 40,
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
    if require_git and manifest.git_commit == "unavailable":
        manifest.validate(require_git=True)
    if resolved_config.get("prereg_path") and resolved_config.get("prereg_version"):
        manifest.bind_preregistration(resolve_preregistration(resolved_config))
    elif require_git:
        resolve_preregistration(resolved_config)
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
