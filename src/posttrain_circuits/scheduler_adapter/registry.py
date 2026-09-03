"""Code-owned handler/profile contracts; production remains intentionally empty."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.scheduler_adapter.environment import THREAD_ENVIRONMENT_KEYS
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError, DispatchError
from posttrain_circuits.scheduler_adapter.manifest import IDENTIFIER, SHA256, RunningManifest
from posttrain_circuits.scheduler_adapter.paths import (
    ATTEMPT_COMPLETION_NAME,
    PRODUCTION_CODE_ROOT,
    validate_path_chain,
)
from posttrain_circuits.scheduler_adapter.repository_preflight import (
    GATE_NAMES as REPOSITORY_PREFLIGHT_GATES,
    OUTPUT_NAME as REPOSITORY_PREFLIGHT_OUTPUT,
    PROFILE_NAME as REPOSITORY_PREFLIGHT_PROFILE_NAME,
    TASK_NAME as REPOSITORY_PREFLIGHT_TASK,
    validate_repository_preflight_completion,
)
from posttrain_circuits.scheduler_adapter.qwen3_v2_gpu_preflight import (
    GATE_NAMES as QWEN3_V2_GPU_PREFLIGHT_GATES,
    GPU_MODEL as QWEN3_V2_GPU_MODEL,
    OUTPUT_NAME as QWEN3_V2_GPU_PREFLIGHT_OUTPUT,
    PROFILE_NAME as QWEN3_V2_GPU_PREFLIGHT_PROFILE_NAME,
    TASK_NAME as QWEN3_V2_GPU_PREFLIGHT_TASK,
    validate_qwen3_v2_gpu_preflight_completion,
)
from posttrain_circuits.scheduler_adapter.secure_files import (
    HeldRegularFile,
    hold_regular_file,
    open_directory_nofollow,
    read_descriptor_bytes,
    sha256_descriptor,
)
from posttrain_circuits.scheduler_adapter.strict_json import parse_strict_json

if TYPE_CHECKING:
    from posttrain_circuits.scheduler_adapter.content_store import (
        OutputAttempt,
        ReadOnlyContentHandle,
    )
    from posttrain_circuits.workflows.contracts import WorkflowPlan, WorkflowUnit


CONFIG_HASH_FIELDS = frozenset(
    {
        "execution_config_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
    }
)
CONFIG_HASH_CONTENT_INPUTS: Mapping[str, str] = MappingProxyType(
    {
        "execution_config_sha256": "execution_config_sha256",
        "resolved_config_sha256": "resolved_config_sha256",
        "scientific_config_sha256": "scientific_config_sha256",
    }
)
CONFIG_BINDING_CONTENT_INPUT = "config_binding_sha256"
FORBIDDEN_FIXED_ENVIRONMENT = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        *THREAD_ENVIRONMENT_KEYS,
    }
)
ADAPTER_OWNED_OPTIONS = frozenset(
    {
        "--allocation-sha256",
        "--attempt",
        "--attempt-completion-name",
        "--content-handle",
        "--execution-profile",
        "--job-id",
        "--job-manifest",
        "--manifest-sha256",
        "--output-attempt-handle",
        "--plan-sha256",
        "--run-id",
        "--unit-id",
        "--workflow-id",
    }
)
PACKAGE_MANIFEST_KEYS = frozenset(
    {
        "dependency_lock_sha256",
        "deployment_id",
        "executable_sha256",
        "implementation_sha256",
        "runtime_flags",
        "runtime_version",
        "schema_version",
        "self_contained",
    }
)

SemanticValidator = Callable[[Mapping[str, Any], Any], None]
FIXED_RUNTIME_ROOT = Path("/usr/bin")
GPU_RUNTIME_ROOT = Path("/scr/del6500/OPD/envs/qwen3-v2-gpu-preflight-v1/bin")
ADDITIONAL_RUNTIME_ROOTS = (GPU_RUNTIME_ROOT,)


class ConfigBindingResolver(Protocol):
    """Code-owned dispatch boundary for a complete CAS ConfigBinding."""

    def resolve(
        self,
        *,
        manifest: RunningManifest,
        plan: "WorkflowPlan",
        unit: "WorkflowUnit",
        content_handles: tuple["ReadOnlyContentHandle", ...],
    ) -> Any: ...


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise AdapterValidationError(f"{name} is not a valid identifier")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise AdapterValidationError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AdapterValidationError(f"{name} must be a positive integer")
    return value


def _validate_string_mapping(value: Mapping[str, str], *, name: str) -> None:
    if not isinstance(value, Mapping):
        raise AdapterValidationError(f"{name} must be a mapping")
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or "\x00" in key
            or "=" in key
            or not isinstance(item, str)
            or "\x00" in item
        ):
            raise AdapterValidationError(f"{name} contains an invalid environment entry")


@dataclass(frozen=True)
class ExecutionProfileContract:
    """Exact code-owned admission envelope for one registered scheduler profile."""

    name: str
    kind: str
    process_count: int
    cpu_cores_min: int
    cpu_cores_max: int
    memory_mib_min: int
    memory_mib_max: int
    gpu_count: int
    gpu_memory_mib_min: int
    gpu_memory_mib_max: int
    gpu_utilization_pct_min: int
    gpu_utilization_pct_max: int
    exclusive_gpu: bool
    allowed_gpu_models: tuple[str, ...]

    def validate_contract(self) -> None:
        _identifier(self.name, name="execution profile name")
        if self.kind not in {"cpu", "gpu"}:
            raise AdapterValidationError("execution profile kind must be cpu or gpu")
        _positive_integer(self.process_count, name="execution profile process_count")
        for value, name in (
            (self.cpu_cores_min, "cpu_cores_min"),
            (self.cpu_cores_max, "cpu_cores_max"),
            (self.memory_mib_min, "memory_mib_min"),
            (self.memory_mib_max, "memory_mib_max"),
        ):
            _positive_integer(value, name=f"execution profile {name}")
        if self.cpu_cores_min > self.cpu_cores_max:
            raise AdapterValidationError("execution profile CPU range is inverted")
        if self.memory_mib_min > self.memory_mib_max:
            raise AdapterValidationError("execution profile memory range is inverted")
        for value, name, maximum in (
            (self.gpu_count, "gpu_count", None),
            (self.gpu_memory_mib_min, "gpu_memory_mib_min", None),
            (self.gpu_memory_mib_max, "gpu_memory_mib_max", None),
            (self.gpu_utilization_pct_min, "gpu_utilization_pct_min", 100),
            (self.gpu_utilization_pct_max, "gpu_utilization_pct_max", 100),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or (maximum is not None and value > maximum)
            ):
                raise AdapterValidationError(f"execution profile {name} is invalid")
        if not isinstance(self.exclusive_gpu, bool):
            raise AdapterValidationError("execution profile exclusive_gpu must be boolean")
        if not isinstance(self.allowed_gpu_models, tuple) or any(
            not isinstance(model, str) or not model.strip() or model != model.strip()
            for model in self.allowed_gpu_models
        ):
            raise AdapterValidationError("execution profile GPU model allowlist is invalid")
        if len(set(self.allowed_gpu_models)) != len(self.allowed_gpu_models):
            raise AdapterValidationError("execution profile GPU model allowlist has duplicates")
        if self.kind == "cpu":
            if any(
                value != 0
                for value in (
                    self.gpu_count,
                    self.gpu_memory_mib_min,
                    self.gpu_memory_mib_max,
                    self.gpu_utilization_pct_min,
                    self.gpu_utilization_pct_max,
                )
            ) or self.allowed_gpu_models:
                raise AdapterValidationError("CPU profile must have a zero/empty GPU contract")
        else:
            if (
                self.gpu_count < 1
                or self.gpu_memory_mib_min < 1
                or self.gpu_memory_mib_min > self.gpu_memory_mib_max
                or self.gpu_utilization_pct_min < 1
                or self.gpu_utilization_pct_min > self.gpu_utilization_pct_max
                or not self.allowed_gpu_models
            ):
                raise AdapterValidationError("GPU profile has an incomplete GPU contract")
            if self.process_count != self.gpu_count:
                raise AdapterValidationError(
                    "GPU profile process_count must equal its exact gpu_count"
                )

    def validate_allocation(
        self,
        manifest: RunningManifest,
        *,
        observed_gpu_models: tuple[str, ...] | None,
    ) -> None:
        self.validate_contract()
        if manifest.execution_profile != self.name:
            raise AdapterValidationError("running manifest execution profile is not this contract")
        allocation = manifest.allocation
        if not self.cpu_cores_min <= allocation.cpu_cores <= self.cpu_cores_max:
            raise AdapterValidationError("allocation CPU cores are outside the profile contract")
        if allocation.cpu_cores % self.process_count != 0:
            raise AdapterValidationError(
                "allocation CPU cores do not divide across the fixed process topology"
            )
        if not self.memory_mib_min <= allocation.memory_mib <= self.memory_mib_max:
            raise AdapterValidationError("allocation memory is outside the profile contract")
        if allocation.gpu_count != self.gpu_count:
            raise AdapterValidationError("allocation GPU count differs from the profile contract")
        if allocation.exclusive_gpu is not self.exclusive_gpu:
            raise AdapterValidationError("allocation GPU exclusivity differs from the profile contract")
        if self.kind == "cpu":
            if observed_gpu_models not in {None, ()}:
                raise AdapterValidationError("CPU profile cannot receive observed GPU models")
            return
        if not self.gpu_memory_mib_min <= allocation.gpu_memory_mib <= self.gpu_memory_mib_max:
            raise AdapterValidationError("allocation GPU memory is outside the profile contract")
        if not (
            self.gpu_utilization_pct_min
            <= allocation.gpu_utilization_pct
            <= self.gpu_utilization_pct_max
        ):
            raise AdapterValidationError(
                "allocation GPU utilization is outside the profile contract"
            )
        # Protocol v2 does not expose GPU model names in the running manifest.
        # ServerScheduler applies the registered model allowlist before launch;
        # the GPU worker independently records and validates CUDA device names.
        # Tests and future protocol revisions may still supply a trusted
        # observation here, in which case it is checked strictly.
        if observed_gpu_models is not None:
            if len(observed_gpu_models) != self.gpu_count or any(
                model not in self.allowed_gpu_models for model in observed_gpu_models
            ):
                raise AdapterValidationError(
                    "observed GPU models differ from the profile allowlist"
                )

    def identity_payload(self) -> dict[str, Any]:
        self.validate_contract()
        return {
            "allowed_gpu_models": list(self.allowed_gpu_models),
            "cpu_cores_max": self.cpu_cores_max,
            "cpu_cores_min": self.cpu_cores_min,
            "exclusive_gpu": self.exclusive_gpu,
            "gpu_count": self.gpu_count,
            "gpu_memory_mib_max": self.gpu_memory_mib_max,
            "gpu_memory_mib_min": self.gpu_memory_mib_min,
            "gpu_utilization_pct_max": self.gpu_utilization_pct_max,
            "gpu_utilization_pct_min": self.gpu_utilization_pct_min,
            "kind": self.kind,
            "memory_mib_max": self.memory_mib_max,
            "memory_mib_min": self.memory_mib_min,
            "name": self.name,
            "process_count": self.process_count,
        }


@dataclass(frozen=True)
class DeploymentContract:
    """Hash-bound interpreter, self-contained package, and dependency evidence."""

    deployment_id: str
    runtime_version: str
    runtime_flags: tuple[str, ...]
    executable: Path
    executable_sha256: str
    implementation: Path
    implementation_sha256: str
    dependency_lock: Path
    dependency_lock_sha256: str
    package_manifest: Path
    package_manifest_sha256: str
    deployment_identity_sha256: str

    def identity_payload(self) -> dict[str, Any]:
        _identifier(self.deployment_id, name="deployment_id")
        if not isinstance(self.runtime_version, str) or not self.runtime_version.strip():
            raise AdapterValidationError("deployment runtime_version must be non-empty")
        if self.runtime_flags not in {("-I", "-S"), ("-I",)}:
            raise AdapterValidationError(
                "deployment runtime_flags must enforce isolated Python"
            )
        for value, name in (
            (self.executable_sha256, "deployment executable_sha256"),
            (self.implementation_sha256, "deployment implementation_sha256"),
            (self.dependency_lock_sha256, "deployment dependency_lock_sha256"),
            (self.package_manifest_sha256, "deployment package_manifest_sha256"),
        ):
            _sha256(value, name=name)
        return {
            "dependency_lock": str(self.dependency_lock),
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "deployment_id": self.deployment_id,
            "executable": str(self.executable),
            "executable_sha256": self.executable_sha256,
            "implementation": str(self.implementation),
            "implementation_sha256": self.implementation_sha256,
            "package_manifest": str(self.package_manifest),
            "package_manifest_sha256": self.package_manifest_sha256,
            "runtime_flags": list(self.runtime_flags),
            "runtime_version": self.runtime_version,
            "schema_version": 1,
        }

    def validate_identity(self) -> None:
        _sha256(
            self.deployment_identity_sha256,
            name="deployment_identity_sha256",
        )
        if sha256_value(self.identity_payload()) != self.deployment_identity_sha256:
            raise AdapterValidationError("deployment identity differs from its reviewed digest")


@dataclass
class PreparedHandler:
    """Validated inodes and directory held from hash verification through wait."""

    spec: "HandlerSpec"
    profile: ExecutionProfileContract
    executable: HeldRegularFile
    implementation: HeldRegularFile
    dependency_lock: HeldRegularFile
    package_manifest: HeldRegularFile
    cwd_descriptor: int

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (
            self.executable.descriptor,
            self.implementation.descriptor,
            self.cwd_descriptor,
        )

    @property
    def cwd_proc_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.cwd_descriptor}")

    def executable_launch_path(self) -> str:
        """Return the reviewed absolute interpreter path after rechecking its held inode.

        CPython discovers ``pyvenv.cfg`` relative to the executable pathname.
        Executing the interpreter through ``/proc/self/fd/N`` defeats that
        discovery, so only the implementation and data handles use proc-fd
        paths. The executable descriptor remains open and is compared with the
        absolute path immediately before spawning.
        """

        path = self.spec.deployment.executable
        if not path.is_absolute() or path != self.executable.path:
            raise AdapterValidationError(
                "prepared executable path differs from the deployment contract"
            )

        def identity(metadata: os.stat_result) -> tuple[int, ...]:
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )

        try:
            held_before = os.fstat(self.executable.descriptor)
            path_before = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise AdapterValidationError(
                f"cannot revalidate fixed handler executable: {error}"
            ) from error
        if identity(path_before) != identity(held_before):
            raise AdapterValidationError(
                "fixed handler executable path no longer names the reviewed inode"
            )
        observed = sha256_descriptor(
            self.executable.descriptor,
            context="fixed handler executable before launch",
        )
        try:
            held_after = os.fstat(self.executable.descriptor)
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise AdapterValidationError(
                f"cannot revalidate fixed handler executable: {error}"
            ) from error
        if (
            observed != self.spec.deployment.executable_sha256
            or identity(held_after) != identity(held_before)
            or identity(path_after) != identity(held_before)
        ):
            raise AdapterValidationError(
                "fixed handler executable changed after deployment verification"
            )
        return str(path)

    def argv(
        self,
        manifest: RunningManifest,
        envelope: Any,
        *,
        content_handles: tuple["ReadOnlyContentHandle", ...],
        output_attempt: "OutputAttempt",
    ) -> tuple[str, ...]:
        """Use held inodes, read-only content fds, and one fixed staging fd."""

        names = tuple(handle.name for handle in content_handles)
        if tuple(sorted(names)) != names or len(set(names)) != len(names):
            raise AdapterValidationError("content handles must be canonical and unique")
        for handle in content_handles:
            _identifier(handle.name, name="content handle name")
            if handle.kind not in {"file", "tree"}:
                raise AdapterValidationError("content handle kind is invalid")
            _sha256(handle.sha256, name="content handle sha256")
            if handle.descriptor < 0:
                raise AdapterValidationError("content handle descriptor is closed")
        if (
            output_attempt.workflow_id != manifest.parameters.workflow_id
            or output_attempt.plan_sha256 != manifest.parameters.plan_sha256
            or output_attempt.unit_id != manifest.parameters.unit_id
            or output_attempt.job_id != envelope.job_id
            or output_attempt.attempt != envelope.attempt
            or output_attempt.descriptor < 0
        ):
            raise AdapterValidationError(
                "output attempt context differs from validated scheduler identities"
            )
        from posttrain_circuits.workflows.contracts import unit_identity_sha256

        argv = [
            self.executable_launch_path(),
            *self.spec.deployment.runtime_flags,
            self.implementation.proc_path,
            *self.spec.fixed_args,
            "--workflow-id",
            manifest.parameters.workflow_id,
            "--plan-sha256",
            manifest.parameters.plan_sha256,
            "--unit-id",
            manifest.parameters.unit_id,
            "--run-id",
            unit_identity_sha256(
                workflow_id=manifest.parameters.workflow_id,
                plan_sha256=manifest.parameters.plan_sha256,
                unit_id=manifest.parameters.unit_id,
            ),
            "--job-id",
            envelope.job_id,
            "--attempt",
            str(envelope.attempt),
            "--execution-profile",
            envelope.execution_profile,
            "--manifest-sha256",
            envelope.manifest_sha256,
            "--allocation-sha256",
            envelope.allocation_sha256,
        ]
        for handle in content_handles:
            argv.extend(
                (
                    "--content-handle",
                    handle.name,
                    handle.kind,
                    handle.sha256,
                    handle.proc_path,
                )
            )
        argv.extend(
            (
                "--output-attempt-handle",
                output_attempt.proc_path,
                "--attempt-completion-name",
                ATTEMPT_COMPLETION_NAME,
            )
        )
        return tuple(argv)

    def close(self) -> None:
        self.executable.close()
        self.implementation.close()
        self.dependency_lock.close()
        self.package_manifest.close()
        if self.cwd_descriptor >= 0:
            os.close(self.cwd_descriptor)
            self.cwd_descriptor = -1

    def __enter__(self) -> "PreparedHandler":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class HandlerSpec:
    """Complete code-owned scientific, deployment, and allocation contract."""

    task: str
    deployment: DeploymentContract
    profiles: Mapping[str, ExecutionProfileContract]
    fixed_args: tuple[str, ...]
    fixed_environment: Mapping[str, str]
    cwd: Path
    output_names: tuple[str, ...]
    required_gate_names: tuple[str, ...]
    config_hash_bindings: Mapping[str, str]
    semantic_validator_id: str
    semantic_validator: SemanticValidator

    def validate_contract(self) -> None:
        _identifier(self.task, name="handler task")
        self.deployment.validate_identity()
        if not isinstance(self.profiles, MappingProxyType) or not self.profiles:
            raise AdapterValidationError(
                "handler requires an immutable code-owned profile registry"
            )
        for name, profile in self.profiles.items():
            if not isinstance(profile, ExecutionProfileContract):
                raise AdapterValidationError("handler profile registry entry is invalid")
            profile.validate_contract()
            if name != profile.name:
                raise AdapterValidationError("handler profile key differs from profile.name")
        if not isinstance(self.fixed_args, tuple) or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in self.fixed_args
        ):
            raise AdapterValidationError("handler fixed_args must be non-empty strings")
        if any(
            argument == "--"
            or argument.split("=", 1)[0] in ADAPTER_OWNED_OPTIONS
            for argument in self.fixed_args
        ):
            raise AdapterValidationError("handler fixed_args contain an adapter-owned option")
        if not isinstance(self.fixed_environment, MappingProxyType):
            raise AdapterValidationError("handler fixed_environment must be immutable")
        _validate_string_mapping(self.fixed_environment, name="handler fixed_environment")
        forbidden = set(self.fixed_environment) & FORBIDDEN_FIXED_ENVIRONMENT
        if forbidden or any(
            key.startswith("SERVER_SCHEDULER_") for key in self.fixed_environment
        ):
            raise AdapterValidationError(
                f"handler fixed_environment contains forbidden keys: {sorted(forbidden)}"
            )
        for values, name in (
            (self.output_names, "handler output_names"),
            (self.required_gate_names, "handler required_gate_names"),
        ):
            if not isinstance(values, tuple) or not values:
                raise AdapterValidationError(f"{name} must be a non-empty tuple")
            for value in values:
                _identifier(value, name=name)
            if tuple(sorted(values)) != values or len(set(values)) != len(values):
                raise AdapterValidationError(f"{name} must be sorted and unique")
        if (
            not isinstance(self.config_hash_bindings, MappingProxyType)
            or set(self.config_hash_bindings) != CONFIG_HASH_FIELDS
            or dict(self.config_hash_bindings) != dict(CONFIG_HASH_CONTENT_INPUTS)
            or len(set(self.config_hash_bindings.values())) != 3
        ):
            raise AdapterValidationError(
                "handler must bind three independent named config contents exactly"
            )
        for source_name in self.config_hash_bindings.values():
            _identifier(source_name, name="handler config hash content identity")
        _identifier(self.semantic_validator_id, name="semantic_validator_id")
        if not callable(self.semantic_validator):
            raise AdapterValidationError("handler semantic_validator must be code-owned callable")

    def profile(self, name: str) -> ExecutionProfileContract:
        self.validate_contract()
        try:
            return self.profiles[name]
        except KeyError as error:
            raise AdapterValidationError(
                f"execution profile {name!r} is not code-owned for task {self.task!r}"
            ) from error

    def validate_unit_contract(self, unit: Any) -> None:
        self.validate_contract()
        if getattr(unit, "task", None) != self.task:
            raise AdapterValidationError("handler task differs from workflow unit task")
        if tuple(getattr(unit, "output_names", ())) != self.output_names:
            raise AdapterValidationError("workflow outputs differ from the handler contract")
        identities = tuple(getattr(unit, "content_inputs", ()))
        input_names = {identity.name for identity in identities}
        missing = set(self.config_hash_bindings.values()) - input_names
        if missing:
            raise AdapterValidationError(
                f"workflow unit lacks handler config-hash identities: {sorted(missing)}"
            )
        identity_by_name = {identity.name: identity for identity in identities}
        if CONFIG_BINDING_CONTENT_INPUT not in identity_by_name:
            raise AdapterValidationError(
                "workflow unit lacks config_binding_sha256 identity"
            )
        if any(
            getattr(identity_by_name[name], "kind", None) != "file"
            for name in (
                CONFIG_BINDING_CONTENT_INPUT,
                *self.config_hash_bindings.values(),
            )
        ):
            raise AdapterValidationError(
                "handler ConfigBinding identities must be immutable file contents"
            )
        if len(
            {
                identity_by_name[name].sha256
                for name in self.config_hash_bindings.values()
            }
        ) != 3:
            raise AdapterValidationError(
                "handler config-hash identities must not alias"
            )

    def prepare(
        self,
        manifest: RunningManifest,
        *,
        approved_code_root: Path,
        approved_runtime_root: Path,
        additional_runtime_roots: tuple[Path, ...] = (),
        observed_gpu_models: tuple[str, ...] | None = None,
    ) -> PreparedHandler:
        self.validate_contract()
        if self.task != manifest.task:
            raise AdapterValidationError("handler task does not match the running manifest")
        profile = self.profile(manifest.execution_profile)
        profile.validate_allocation(
            manifest, observed_gpu_models=observed_gpu_models
        )
        deployment = self.deployment
        validate_path_chain(
            deployment.executable,
            approved_roots=(
                approved_code_root,
                approved_runtime_root,
                *additional_runtime_roots,
            ),
            final_kind="file",
        )
        for path in (deployment.implementation, deployment.package_manifest):
            validate_path_chain(
                path, approved_roots=(approved_code_root,), final_kind="file"
            )
        validate_path_chain(
            deployment.dependency_lock,
            approved_roots=(
                approved_code_root,
                approved_runtime_root,
                *additional_runtime_roots,
            ),
            final_kind="file",
        )
        validate_path_chain(
            self.cwd, approved_roots=(approved_code_root,), final_kind="directory"
        )
        held: list[HeldRegularFile] = []
        cwd_descriptor = -1
        try:
            executable = hold_regular_file(
                deployment.executable,
                context="fixed handler executable",
                expected_sha256=deployment.executable_sha256,
            )
            held.append(executable)
            if os.fstat(executable.descriptor).st_mode & 0o111 == 0:
                raise AdapterValidationError("fixed handler executable is not executable")
            implementation = hold_regular_file(
                deployment.implementation,
                context="fixed handler implementation bundle",
                expected_sha256=deployment.implementation_sha256,
            )
            held.append(implementation)
            dependency_lock = hold_regular_file(
                deployment.dependency_lock,
                context="fixed handler dependency lock",
                expected_sha256=deployment.dependency_lock_sha256,
            )
            held.append(dependency_lock)
            package_manifest = hold_regular_file(
                deployment.package_manifest,
                context="fixed handler package manifest",
                expected_sha256=deployment.package_manifest_sha256,
            )
            held.append(package_manifest)
            self._validate_package_manifest(package_manifest)
            cwd_descriptor = open_directory_nofollow(
                self.cwd, context="fixed handler working directory"
            )
            return PreparedHandler(
                spec=self,
                profile=profile,
                executable=executable,
                implementation=implementation,
                dependency_lock=dependency_lock,
                package_manifest=package_manifest,
                cwd_descriptor=cwd_descriptor,
            )
        except BaseException:
            for item in reversed(held):
                item.close()
            if cwd_descriptor >= 0:
                os.close(cwd_descriptor)
            raise

    def _validate_package_manifest(self, held: HeldRegularFile) -> None:
        raw = read_descriptor_bytes(
            held.descriptor,
            context="fixed handler package manifest",
            max_bytes=1024 * 1024,
        )
        payload = parse_strict_json(raw, context="fixed handler package manifest")
        if not isinstance(payload, dict) or set(payload) != PACKAGE_MANIFEST_KEYS:
            raise AdapterValidationError("fixed handler package manifest fields are invalid")
        expected = {
            "dependency_lock_sha256": self.deployment.dependency_lock_sha256,
            "deployment_id": self.deployment.deployment_id,
            "executable_sha256": self.deployment.executable_sha256,
            "implementation_sha256": self.deployment.implementation_sha256,
            "runtime_flags": list(self.deployment.runtime_flags),
            "runtime_version": self.deployment.runtime_version,
            "schema_version": 1,
            "self_contained": True,
        }
        if payload != expected:
            raise AdapterValidationError(
                "fixed handler package manifest does not bind its complete deployment"
            )


_REPOSITORY_PREFLIGHT_PROFILE = ExecutionProfileContract(
    name=REPOSITORY_PREFLIGHT_PROFILE_NAME,
    kind="cpu",
    process_count=1,
    cpu_cores_min=1,
    cpu_cores_max=1,
    memory_mib_min=128,
    memory_mib_max=128,
    gpu_count=0,
    gpu_memory_mib_min=0,
    gpu_memory_mib_max=0,
    gpu_utilization_pct_min=0,
    gpu_utilization_pct_max=0,
    exclusive_gpu=False,
    allowed_gpu_models=(),
)
_REPOSITORY_PREFLIGHT_DEPLOYMENT = DeploymentContract(
    deployment_id="repository-preflight-python312-v1",
    runtime_version="Python 3.12.13",
    runtime_flags=("-I", "-S"),
    executable=Path("/usr/bin/python3.12"),
    executable_sha256="848c64ae0635d363f8bbfc768f94a3be497c0d51acd28cd5087e6e8a13c44801",
    implementation=(
        PRODUCTION_CODE_ROOT
        / "scripts"
        / "server_scheduler"
        / "repository-preflight-handler.py"
    ),
    implementation_sha256="2884dcd34609fa1f178be6856db76848e40a8fe0ce630152526c2bd24ee899a6",
    dependency_lock=(
        PRODUCTION_CODE_ROOT
        / "deployments"
        / "repository_preflight"
        / "dependency-lock.json"
    ),
    dependency_lock_sha256="106569bec10ba3892a86d2e2ad0cc38cf6da3ad62935524c0c2646654b40abe5",
    package_manifest=(
        PRODUCTION_CODE_ROOT
        / "deployments"
        / "repository_preflight"
        / "package-manifest.json"
    ),
    package_manifest_sha256="22b01e70e387c05886b24c75527a85ba6006e71025925e0679d4c0cd4765c3ef",
    deployment_identity_sha256="3e958f8ce86a220e7d443bf5605055d695e4d634603e639a02a32be23a5da6d5",
)
_REPOSITORY_PREFLIGHT_HANDLER = HandlerSpec(
    task=REPOSITORY_PREFLIGHT_TASK,
    deployment=_REPOSITORY_PREFLIGHT_DEPLOYMENT,
    profiles=MappingProxyType(
        {REPOSITORY_PREFLIGHT_PROFILE_NAME: _REPOSITORY_PREFLIGHT_PROFILE}
    ),
    fixed_args=(),
    fixed_environment=MappingProxyType({}),
    cwd=PRODUCTION_CODE_ROOT,
    output_names=(REPOSITORY_PREFLIGHT_OUTPUT,),
    required_gate_names=REPOSITORY_PREFLIGHT_GATES,
    config_hash_bindings=CONFIG_HASH_CONTENT_INPUTS,
    semantic_validator_id="repository-preflight-result-v1",
    semantic_validator=validate_repository_preflight_completion,
)

_QWEN3_V2_GPU_PREFLIGHT_PROFILE = ExecutionProfileContract(
    name=QWEN3_V2_GPU_PREFLIGHT_PROFILE_NAME,
    kind="gpu",
    process_count=4,
    cpu_cores_min=16,
    cpu_cores_max=16,
    memory_mib_min=196608,
    memory_mib_max=196608,
    gpu_count=4,
    gpu_memory_mib_min=81920,
    gpu_memory_mib_max=81920,
    gpu_utilization_pct_min=95,
    gpu_utilization_pct_max=95,
    exclusive_gpu=True,
    allowed_gpu_models=(QWEN3_V2_GPU_MODEL,),
)
_QWEN3_V2_GPU_PREFLIGHT_DEPLOYMENT = DeploymentContract(
    deployment_id="qwen3-v2-gpu-preflight-python312-cuda-v1",
    runtime_version=(
        "Python 3.12.13; PyTorch 2.8.0+cu128; CUDA 12.8; NCCL 2.27.3; "
        "Transformers 4.56.2"
    ),
    runtime_flags=("-I",),
    executable=GPU_RUNTIME_ROOT / "python",
    executable_sha256="848c64ae0635d363f8bbfc768f94a3be497c0d51acd28cd5087e6e8a13c44801",
    implementation=(
        PRODUCTION_CODE_ROOT
        / "scripts"
        / "server_scheduler"
        / "qwen3-v2-gpu-preflight-handler.py"
    ),
    implementation_sha256="f34a33df1b642ebde126b32642ec7f963620865ba813669244ee0801e50bfb4d",
    dependency_lock=(
        PRODUCTION_CODE_ROOT
        / "deployments"
        / "qwen3_v2_gpu_preflight"
        / "dependency-lock.json"
    ),
    dependency_lock_sha256="944f22e346fbda1bd2545cdac9f8c3e9e7046a576b9afc38911fc2f89a1f735d",
    package_manifest=(
        PRODUCTION_CODE_ROOT
        / "deployments"
        / "qwen3_v2_gpu_preflight"
        / "package-manifest.json"
    ),
    package_manifest_sha256="4bd708aa1551009ebab80c7aade9f32d9ae543b05cb5bd0ed5b7702b4a87d708",
    deployment_identity_sha256="b6a1b1f898ede8c4e663f820978fee2a62e84960a03e00dc6f04c9a3d9939a88",
)
_QWEN3_V2_GPU_PREFLIGHT_HANDLER = HandlerSpec(
    task=QWEN3_V2_GPU_PREFLIGHT_TASK,
    deployment=_QWEN3_V2_GPU_PREFLIGHT_DEPLOYMENT,
    profiles=MappingProxyType(
        {QWEN3_V2_GPU_PREFLIGHT_PROFILE_NAME: _QWEN3_V2_GPU_PREFLIGHT_PROFILE}
    ),
    fixed_args=(),
    fixed_environment=MappingProxyType(
        {
            "HF_HOME": "/scr/del6500/OPD/cache/huggingface",
            "HF_HUB_CACHE": "/scr/del6500/OPD/cache/huggingface/hub",
            "HF_HUB_OFFLINE": "1",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,ENV,GRAPH,NET,COLL",
            "NCCL_P2P_DISABLE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "TMPDIR": "/scr/del6500/OPD/tmp",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
        }
    ),
    cwd=PRODUCTION_CODE_ROOT,
    output_names=(QWEN3_V2_GPU_PREFLIGHT_OUTPUT,),
    required_gate_names=QWEN3_V2_GPU_PREFLIGHT_GATES,
    config_hash_bindings=CONFIG_HASH_CONTENT_INPUTS,
    semantic_validator_id="qwen3-v2-gpu-preflight-result-v1",
    semantic_validator=validate_qwen3_v2_gpu_preflight_completion,
)

# The CPU preflight and one bounded four-GPU preflight are migrated. Training,
# circuit analysis, G0, and factorial tasks remain fail-closed.
HANDLER_REGISTRY: Mapping[str, HandlerSpec] = MappingProxyType(
    {
        QWEN3_V2_GPU_PREFLIGHT_TASK: _QWEN3_V2_GPU_PREFLIGHT_HANDLER,
        REPOSITORY_PREFLIGHT_TASK: _REPOSITORY_PREFLIGHT_HANDLER,
    }
)


def require_handler(
    task: str, *, registry: Mapping[str, HandlerSpec] | None = None
) -> HandlerSpec:
    selected_registry = HANDLER_REGISTRY if registry is None else registry
    if not isinstance(selected_registry, MappingProxyType):
        raise DispatchError("OPD handler registry must be an immutable code-owned mapping")
    try:
        handler = selected_registry[task]
    except KeyError as error:
        raise DispatchError(f"no reviewed OPD handler is migrated for task {task!r}") from error
    if not isinstance(handler, HandlerSpec):
        raise DispatchError(f"invalid OPD handler registry entry for task {task!r}")
    handler.validate_contract()
    if handler.task != task:
        raise DispatchError(f"OPD handler registry key differs from task {task!r}")
    return handler
