"""Run-directory provenance capture."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from posttrain_circuits.artifacts.config_bindings import (
    ConfigBinding,
    validate_config_binding,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.artifacts.protocol_amendments import (
    AMENDMENT_RELATIVE_PATH,
    ExecutionClassAmendmentBinding,
    ProtocolAmendmentBinding,
    SUCCESSOR_AMENDMENT_RELATIVE_PATH,
    resolve_accepted_execution_class_amendment,
    resolve_accepted_protocol_amendment,
)


PROTOCOL_AMENDMENT_BINDING_FIELDS = (
    "protocol_amendment_id",
    "protocol_amendment_path",
    "protocol_amendment_git_commit",
    "protocol_amendment_sha256",
    "reviewed_implementation_commit",
)


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
    return git_output(["log", "-n", "1", "--format=%H", "--", str(path)]) or "unavailable"


def _prereg_sha256(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "unavailable"


def _prereg_dirty(path: Path) -> bool:
    return bool(git_output(["status", "--porcelain", "--", str(path)]) or "")


def implementation_commit_is_preregistered(
    preregistration: dict[str, Any],
    implementation_commit: str,
) -> bool:
    """Allow a frozen implementation or an explicitly reviewed implementation-only amendment."""

    frozen = preregistration.get("frozen_implementation_commit")
    if frozen is None:
        return True
    if (
        not isinstance(frozen, str)
        or len(frozen) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in frozen)
    ):
        raise ValueError("preregistration frozen implementation commit is invalid")
    if implementation_commit == frozen:
        return True
    amendments = preregistration.get("reviewed_implementation_amendments", [])
    if not isinstance(amendments, list):
        raise ValueError("reviewed implementation amendments must be a list")
    for amendment in amendments:
        if not isinstance(amendment, dict):
            raise ValueError("reviewed implementation amendment must be a mapping")
        required = {
            "base_frozen_implementation_commit": frozen,
            "implementation_commit": implementation_commit,
            "review_status": "accepted",
            "scientific_scope": "implementation_only_no_preregistered_design_change",
        }
        if all(amendment.get(key) == value for key, value in required.items()) and all(
            isinstance(amendment.get(key), str) and str(amendment[key]).strip()
            for key in ("reviewer", "rationale")
        ):
            return True
    return False


@dataclass(frozen=True)
class PreregistrationBinding:
    path: Path
    version: str
    git_commit: str
    sha256: str
    dirty: bool


def resolve_protocol_amendment(
    config: dict[str, Any],
    *,
    expected_head: str | None = None,
) -> ProtocolAmendmentBinding | ExecutionClassAmendmentBinding | None:
    """Resolve the configured accepted amendment, if this run declares one."""

    raw_path = str(config.get("protocol_amendment_path", "")).strip()
    if not raw_path:
        return None
    resolver = {
        str(AMENDMENT_RELATIVE_PATH): resolve_accepted_protocol_amendment,
        str(SUCCESSOR_AMENDMENT_RELATIVE_PATH): (
            resolve_accepted_execution_class_amendment
        ),
    }.get(raw_path)
    if resolver is None:
        raise ValueError("configured protocol amendment path is not reviewed")
    return resolver(
        code_root=Path.cwd(), configured_path=raw_path, expected_head=expected_head
    )


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
    code_commit = require_git_output(["rev-parse", "HEAD"])
    amendment = resolve_protocol_amendment(config, expected_head=code_commit)
    binding = {
        "protocol_track": protocol_track,
        "artifact_namespace": str(model.get("artifact_namespace", "legacy")),
        "model_revision": str(model.get("model_revision", "unavailable")),
        "teacher_revision": str(teacher.get("model_revision", "unavailable")),
        "tokenizer_revision": str(model.get("tokenizer_revision", "unavailable")),
        "tokenizer_fingerprint": str(model.get("tokenizer_fingerprint", "legacy-unrecorded")),
        "chat_template_sha256": str(prompt.get("chat_template_sha256", "legacy-unrecorded")),
        "prompt_protocol": str(prompt.get("name", "legacy_raw_v1")),
        "enable_thinking": bool(prompt.get("enable_thinking", False)),
        "code_commit": code_commit,
        "prereg_path": str(prereg.path),
        "prereg_version": prereg.version,
        "prereg_commit": prereg.git_commit,
        "prereg_sha256": prereg.sha256,
    }
    if amendment is not None:
        binding.update(
            {
                "protocol_amendment_id": amendment.amendment_id,
                "protocol_amendment_path": str(amendment.path.relative_to(Path.cwd())),
                "protocol_amendment_git_commit": amendment.git_commit,
                "protocol_amendment_sha256": amendment.sha256,
                "reviewed_implementation_commit": amendment.reviewed_implementation_commit,
            }
        )
    return binding


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
    experiment_binding: dict[str, Any]
    experiment_binding_sha256: str
    factorial_design_sha256: str
    config_binding: dict[str, Any]
    resolved_config_sha256: str
    scientific_config_sha256: str
    execution_config_sha256: str
    execution_context: dict[str, Any]
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
    token_budget_unit: str = "global_nonpadding_model_input_tokens_processed"
    token_budget_consumed: int = 0
    training_stop_reason: str | None = None
    metrics_sha256: str | None = None
    final_checkpoint_path: str | None = None
    final_checkpoint_sha256: str | None = None
    resolved_config_yaml_sha256: str | None = None
    teacher_id: str | None = None
    teacher_revision: str | None = None
    resolved_teacher_commit: str | None = None
    teacher_demo_generation: dict[str, Any] | None = None
    resume_ancestry: list[str] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_time: str | None = None
    git_commit: str = field(
        default_factory=lambda: git_output(["rev-parse", "HEAD"]) or "unavailable"
    )
    dirty_working_tree: bool = field(
        default_factory=lambda: bool(git_output(["status", "--porcelain"]) or "")
    )
    package_versions: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    prereg_git_commit: str = "unbound"
    prereg_sha256: str = "unbound"
    prereg_dirty: bool = True
    prereg_version: str = "unbound"
    prereg_path: str = "unbound"
    protocol_amendment_id: str = "unbound"
    protocol_amendment_path: str = "unbound"
    protocol_amendment_git_commit: str = "unbound"
    protocol_amendment_sha256: str = "unbound"
    reviewed_implementation_commit: str = "unbound"

    def bind_preregistration(self, binding: PreregistrationBinding) -> None:
        self.prereg_path = str(binding.path)
        self.prereg_version = binding.version
        self.prereg_git_commit = binding.git_commit
        self.prereg_sha256 = binding.sha256
        self.prereg_dirty = binding.dirty
        self.dirty_working_tree = bool(git_output(["status", "--porcelain"]) or "")

    def bind_protocol_amendment(
        self,
        binding: ProtocolAmendmentBinding | ExecutionClassAmendmentBinding,
    ) -> None:
        self.protocol_amendment_id = binding.amendment_id
        self.protocol_amendment_path = str(binding.path.relative_to(Path.cwd()))
        self.protocol_amendment_git_commit = binding.git_commit
        self.protocol_amendment_sha256 = binding.sha256
        self.reviewed_implementation_commit = binding.reviewed_implementation_commit

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
        config_binding = ConfigBinding.from_dict(self.config_binding)
        expected_config_hashes = {
            "resolved_config_sha256": config_binding.resolved_config_sha256,
            "scientific_config_sha256": config_binding.scientific_config_sha256,
            "execution_config_sha256": config_binding.execution_config_sha256,
        }
        observed_config_hashes = {
            name: getattr(self, name) for name in expected_config_hashes
        }
        if observed_config_hashes != expected_config_hashes:
            raise ValueError("run manifest config hashes differ from ConfigBinding")
        if self.execution_context != self.config_binding["execution_context"]:
            raise ValueError("run manifest execution context differs from ConfigBinding")
        if (
            not isinstance(self.resolved_config_yaml_sha256, str)
            or len(self.resolved_config_yaml_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.resolved_config_yaml_sha256
            )
        ):
            raise ValueError("run manifest requires a resolved-config YAML byte digest")
        _validate_manifest_binding_hash(asdict(self), require=True)
        _validate_manifest_config_binding(asdict(self))
        if self.teacher_id is not None and (not self.teacher_revision or not self.resolved_teacher_commit):
            raise ValueError("teacher runs require teacher revision and resolved commit")
        if (
            self.token_budget < 1
            or self.token_budget_unit != "global_nonpadding_model_input_tokens_processed"
        ):
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
            amendment: ProtocolAmendmentBinding | ExecutionClassAmendmentBinding | None = None
            if self.protocol_amendment_path != "unbound":
                amendment = resolve_protocol_amendment(
                    {"protocol_amendment_path": self.protocol_amendment_path},
                    expected_head=self.git_commit if require_git else None,
                )
                observed_amendment = {
                    "protocol_amendment_id": self.protocol_amendment_id,
                    "protocol_amendment_path": self.protocol_amendment_path,
                    "protocol_amendment_git_commit": self.protocol_amendment_git_commit,
                    "protocol_amendment_sha256": self.protocol_amendment_sha256,
                    "reviewed_implementation_commit": self.reviewed_implementation_commit,
                }
                expected_amendment = {
                    "protocol_amendment_id": amendment.amendment_id,
                    "protocol_amendment_path": str(amendment.path.relative_to(Path.cwd())),
                    "protocol_amendment_git_commit": amendment.git_commit,
                    "protocol_amendment_sha256": amendment.sha256,
                    "reviewed_implementation_commit": amendment.reviewed_implementation_commit,
                }
                if observed_amendment != expected_amendment:
                    raise RuntimeError(
                        "formal run protocol-amendment binding changed"
                    )
            if (
                require_git
                and amendment is None
                and isinstance(prereg_payload, dict)
                and not implementation_commit_is_preregistered(
                    prereg_payload,
                    self.git_commit,
                )
            ):
                raise RuntimeError(
                    "formal run implementation differs from the preregistered commit; "
                    "an explicit reviewed amendment is required"
                )
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
                "execution_context": bool(self.execution_context),
            }
            failures = [name for name, passed in required_qwen3.items() if not passed]
            if failures:
                raise RuntimeError(f"Qwen3 manifest protocol bindings are incomplete: {failures}")


def run_manifest_payload(manifest: RunManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["sha256"] = sha256_value(payload)
    return payload


def _validate_manifest_binding_hash(
    payload: dict[str, Any],
    *,
    require: bool,
) -> bool:
    """Validate the mandatory ExperimentBinding linkage."""

    binding_fields = {
        "experiment_binding",
        "experiment_binding_sha256",
        "factorial_design_sha256",
    }
    present = binding_fields.intersection(payload)
    if not present:
        if require:
            raise ValueError("run manifest requires an ExperimentBinding payload")
        return False
    if present != binding_fields:
        raise ValueError(
            "run manifest ExperimentBinding linkage is partial: "
            f"missing={sorted(binding_fields - present)}"
        )
    raw_binding = payload.get("experiment_binding")
    if not isinstance(raw_binding, dict):
        raise ValueError("run manifest requires an ExperimentBinding payload")
    binding_hash = payload.get("experiment_binding_sha256")
    if (
        not isinstance(binding_hash, str)
        or len(binding_hash) != 64
        or any(character not in "0123456789abcdef" for character in binding_hash)
    ):
        raise ValueError("run manifest requires an ExperimentBinding SHA-256")
    if sha256_value(raw_binding) != binding_hash:
        raise ValueError("run manifest ExperimentBinding payload/hash mismatch")
    design_hash = payload.get("factorial_design_sha256")
    if (
        not isinstance(design_hash, str)
        or len(design_hash) != 64
        or any(character not in "0123456789abcdef" for character in design_hash)
    ):
        raise ValueError("run manifest requires a factorial design SHA-256")
    if raw_binding.get("factorial_design_sha256") != design_hash:
        raise ValueError("run manifest factorial design differs from its binding payload")
    return True


def _validate_manifest_config_binding(payload: dict[str, Any]) -> ConfigBinding:
    raw_binding = payload.get("config_binding")
    if not isinstance(raw_binding, dict):
        raise ValueError("run manifest requires a complete ConfigBinding payload")
    binding = ConfigBinding.from_dict(raw_binding)
    for name, expected in (
        ("resolved_config_sha256", binding.resolved_config_sha256),
        ("scientific_config_sha256", binding.scientific_config_sha256),
        ("execution_config_sha256", binding.execution_config_sha256),
    ):
        if payload.get(name) != expected:
            raise ValueError(f"run manifest {name} differs from ConfigBinding")
    if payload.get("execution_context") != raw_binding["execution_context"]:
        raise ValueError("run manifest execution context differs from ConfigBinding")
    experiment_binding = payload.get("experiment_binding")
    if (
        not isinstance(experiment_binding, dict)
        or experiment_binding.get("scientific_config_sha256")
        != binding.scientific_config_sha256
    ):
        raise ValueError("run manifest ExperimentBinding differs from scientific config")
    yaml_digest = payload.get("resolved_config_yaml_sha256")
    if (
        not isinstance(yaml_digest, str)
        or len(yaml_digest) != 64
        or any(character not in "0123456789abcdef" for character in yaml_digest)
    ):
        raise ValueError("run manifest requires a resolved-config YAML byte digest")
    return binding


def validate_run_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Hash-validate a current manifest; legacy envelopes are rejected."""

    if not isinstance(payload, dict):
        raise ValueError("run manifest payload must be a mapping")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != sha256_value(content):
        raise ValueError("run manifest SHA-256 mismatch")
    _validate_manifest_binding_hash(payload, require=True)
    _validate_manifest_config_binding(payload)
    return payload


def initialize_run_directory(
    run_dir: Path,
    resolved_config: dict[str, Any],
    config_binding: ConfigBinding,
    manifest: RunManifest,
    *,
    require_git: bool = False,
) -> None:
    import torch
    validate_config_binding(resolved_config, config_binding)
    if manifest.config_binding != config_binding.as_dict():
        raise ValueError("run manifest does not persist the supplied ConfigBinding")
    resolved_yaml = yaml.safe_dump(resolved_config, sort_keys=True)
    manifest.resolved_config_yaml_sha256 = hashlib.sha256(
        resolved_yaml.encode("utf-8")
    ).hexdigest()
    if require_git and manifest.git_commit == "unavailable":
        manifest.validate(require_git=True)
    if resolved_config.get("prereg_path") and resolved_config.get("prereg_version"):
        manifest.bind_preregistration(resolve_preregistration(resolved_config))
    elif require_git:
        resolve_preregistration(resolved_config)
    amendment = resolve_protocol_amendment(
        resolved_config,
        expected_head=manifest.git_commit if require_git else None,
    )
    if amendment is not None:
        manifest.bind_protocol_amendment(amendment)
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
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(resolved_yaml, encoding="utf-8")
    if sha256_file(resolved_path) != manifest.resolved_config_yaml_sha256:
        raise RuntimeError("resolved-config YAML bytes changed during publication")
    manifest.validate(require_git=require_git)
    atomic_write_json(run_dir / "config_binding.json", config_binding.as_dict())
    atomic_write_json(run_dir / "manifest.json", run_manifest_payload(manifest))
    atomic_write_json(run_dir / "environment.json", manifest.environment)
    (run_dir / "metrics.jsonl").touch()
    diff = git_output(["diff", "--binary"]) or ""
    (run_dir / "git_diff.patch").write_text(diff + ("\n" if diff else ""), encoding="utf-8")


def finalize_run_directory(run_dir: Path, manifest: RunManifest) -> None:
    resolved_path = run_dir / "resolved_config.yaml"
    binding_path = run_dir / "config_binding.json"
    resolved_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(resolved_config, dict):
        raise ValueError("persisted resolved config must be a mapping")
    raw_binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(raw_binding, dict):
        raise ValueError("persisted ConfigBinding must be a mapping")
    config_binding = ConfigBinding.from_dict(raw_binding)
    validate_config_binding(resolved_config, config_binding)
    if manifest.config_binding != config_binding.as_dict():
        raise ValueError("run manifest ConfigBinding changed before finalization")
    if sha256_file(resolved_path) != manifest.resolved_config_yaml_sha256:
        raise ValueError("resolved-config YAML bytes changed before finalization")
    manifest.end_time = datetime.now(UTC).isoformat()
    atomic_write_json(run_dir / "manifest.json", run_manifest_payload(manifest))


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
