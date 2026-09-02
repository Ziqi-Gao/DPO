"""Strict ConfigBinding resolution from already-verified OPD file-CAS handles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from posttrain_circuits.artifacts.config_bindings import (
    ConfigBinding,
    validate_config_binding,
)
from posttrain_circuits.artifacts.hashing import canonical_json, sha256_value
from posttrain_circuits.scheduler_adapter.content_store import (
    ContentStore,
    ReadOnlyContentHandle,
)
from posttrain_circuits.scheduler_adapter.errors import AdapterValidationError
from posttrain_circuits.scheduler_adapter.manifest import RunningManifest
from posttrain_circuits.scheduler_adapter.secure_files import read_descriptor_bytes
from posttrain_circuits.scheduler_adapter.strict_json import parse_strict_json
from posttrain_circuits.workflows.contracts import (
    ContentIdentity,
    WorkflowPlan,
    WorkflowUnit,
)


CONFIG_BINDING_INPUT_NAME = "config_binding_sha256"
CONFIG_JSON_INPUT_NAMES = (
    "execution_config_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
)
MAX_CONFIG_JSON_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BINDING_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ConfigContentIdentities:
    config_binding: ContentIdentity
    execution_config: ContentIdentity
    resolved_config: ContentIdentity
    scientific_config: ContentIdentity

    def as_tuple(self) -> tuple[ContentIdentity, ...]:
        identities = (
            self.config_binding,
            self.execution_config,
            self.resolved_config,
            self.scientific_config,
        )
        return tuple(sorted(identities))


@dataclass(frozen=True)
class ResolvedConfigBinding:
    binding: ConfigBinding
    resolved_config: dict[str, Any]
    scientific_config: dict[str, Any]
    execution_config: dict[str, Any]
    identities: ConfigContentIdentities


def _canonical_bytes(payload: object, *, context: str) -> bytes:
    try:
        raw = canonical_json(payload).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AdapterValidationError(f"{context} is not canonical JSON: {error}") from error
    restored = parse_strict_json(raw, context=context)
    if restored != payload:
        raise AdapterValidationError(f"{context} does not have a strict JSON round-trip")
    return raw


def _require_mapping(payload: object, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AdapterValidationError(f"{context} must be a JSON object")
    return payload


def _copy_mapping(payload: object, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AdapterValidationError(f"{context} must be a mapping")
    return dict(payload)


def _validate_projection_payloads(
    binding: ConfigBinding,
    *,
    resolved_config: dict[str, Any],
    scientific_config: dict[str, Any],
    execution_config: dict[str, Any],
) -> None:
    config_hashes = (
        binding.execution_config_sha256,
        binding.resolved_config_sha256,
        binding.scientific_config_sha256,
    )
    if len(set(config_hashes)) != 3:
        raise AdapterValidationError(
            "execution, resolved, and scientific config contents must not alias"
        )
    try:
        validate_config_binding(resolved_config, binding)
    except (TypeError, ValueError) as error:
        raise AdapterValidationError(
            f"ConfigBinding does not reproduce from resolved config: {error}"
        ) from error
    expected_scientific_keys = {
        "config",
        "input_artifact_hashes",
        "schema_version",
    }
    if (
        set(scientific_config) != expected_scientific_keys
        or scientific_config.get("schema_version") != 3
        or isinstance(scientific_config.get("schema_version"), bool)
        or not isinstance(scientific_config.get("config"), dict)
        or scientific_config.get("input_artifact_hashes")
        != dict(binding.input_artifact_hashes)
    ):
        raise AdapterValidationError("scientific config projection is invalid")
    expected_execution_keys = {
        "execution_context",
        "schema_version",
        "storage_locators",
    }
    if (
        set(execution_config) != expected_execution_keys
        or execution_config.get("schema_version") != 3
        or isinstance(execution_config.get("schema_version"), bool)
        or execution_config.get("storage_locators") != dict(binding.storage_locators)
        or execution_config.get("execution_context")
        != binding.as_dict()["execution_context"]
    ):
        raise AdapterValidationError("execution config projection is invalid")
    expected_hashes = {
        "execution_config_sha256": binding.execution_config_sha256,
        "resolved_config_sha256": binding.resolved_config_sha256,
        "scientific_config_sha256": binding.scientific_config_sha256,
    }
    payloads = {
        "execution_config_sha256": execution_config,
        "resolved_config_sha256": resolved_config,
        "scientific_config_sha256": scientific_config,
    }
    for name, payload in payloads.items():
        if sha256_value(payload) != expected_hashes[name]:
            raise AdapterValidationError(
                f"{name} canonical JSON hash differs from ConfigBinding"
            )


class ConfigBindingResolver:
    """Materialize and resolve the five-way plan/ConfigBinding CAS contract."""

    def __init__(self, content_store: ContentStore) -> None:
        if not isinstance(content_store, ContentStore):
            raise TypeError("ConfigBindingResolver requires an OPD ContentStore")
        self.content_store = content_store

    def materialize(
        self,
        binding: ConfigBinding,
        *,
        resolved_config: Mapping[str, Any],
        scientific_config: Mapping[str, Any],
        execution_config: Mapping[str, Any],
    ) -> ConfigContentIdentities:
        """Publish canonical JSON bytes only after every binding agrees."""

        if not isinstance(binding, ConfigBinding):
            raise AdapterValidationError("config binding must be ConfigBinding")
        binding_payload = binding.as_dict()
        try:
            strict_binding = ConfigBinding.from_dict(binding_payload)
        except (TypeError, ValueError) as error:
            raise AdapterValidationError(f"ConfigBinding is invalid: {error}") from error
        if strict_binding != binding:
            raise AdapterValidationError("ConfigBinding is not a strict round-trip")
        resolved = _copy_mapping(resolved_config, context="resolved config")
        scientific = _copy_mapping(scientific_config, context="scientific config")
        execution = _copy_mapping(execution_config, context="execution config")
        _validate_projection_payloads(
            binding,
            resolved_config=resolved,
            scientific_config=scientific,
            execution_config=execution,
        )
        payloads = {
            CONFIG_BINDING_INPUT_NAME: binding_payload,
            "execution_config_sha256": execution,
            "resolved_config_sha256": resolved,
            "scientific_config_sha256": scientific,
        }
        identities: dict[str, ContentIdentity] = {}
        for name, payload in payloads.items():
            raw = _canonical_bytes(payload, context=name)
            digest = sha256_value(payload)
            self.content_store.publish_file(sha256=digest, raw=raw)
            identities[name] = ContentIdentity(
                name=name,
                sha256=digest,
                kind="file",
            )
        return ConfigContentIdentities(
            config_binding=identities[CONFIG_BINDING_INPUT_NAME],
            execution_config=identities["execution_config_sha256"],
            resolved_config=identities["resolved_config_sha256"],
            scientific_config=identities["scientific_config_sha256"],
        )

    def resolve(
        self,
        *,
        manifest: RunningManifest,
        plan: WorkflowPlan,
        unit: WorkflowUnit,
        content_handles: tuple[ReadOnlyContentHandle, ...],
    ) -> ResolvedConfigBinding:
        """Resolve only handles exactly bound by manifest -> plan -> unit."""

        plan.validate()
        plan_sha256 = plan.sha256()
        parameters = manifest.parameters
        if (
            parameters.workflow_id != plan.workflow_id
            or parameters.plan_sha256 != plan_sha256
            or parameters.unit_id != unit.unit_id
            or manifest.task != unit.task
            or plan.unit(parameters.unit_id) != unit
        ):
            raise AdapterValidationError(
                "running manifest does not bind the exact config-bearing workflow unit"
            )
        identity_by_name = {identity.name: identity for identity in unit.content_inputs}
        handle_by_name = {handle.name: handle for handle in content_handles}
        if (
            len(handle_by_name) != len(content_handles)
            or set(handle_by_name) != set(identity_by_name)
        ):
            raise AdapterValidationError(
                "config resolver handles differ from workflow unit content inputs"
            )
        for name, identity in identity_by_name.items():
            handle = handle_by_name[name]
            if (
                handle.kind != identity.kind
                or handle.sha256 != identity.sha256
                or handle.descriptor < 0
            ):
                raise AdapterValidationError(
                    f"config resolver handle {name!r} differs from workflow identity"
                )
        required_names = {CONFIG_BINDING_INPUT_NAME, *CONFIG_JSON_INPUT_NAMES}
        if not required_names <= set(identity_by_name):
            raise AdapterValidationError(
                "workflow unit lacks the complete ConfigBinding content contract"
            )
        if any(identity_by_name[name].kind != "file" for name in required_names):
            raise AdapterValidationError("ConfigBinding contents must all be file identities")
        config_digests = tuple(
            identity_by_name[name].sha256 for name in CONFIG_JSON_INPUT_NAMES
        )
        if len(set(config_digests)) != 3:
            raise AdapterValidationError(
                "execution, resolved, and scientific config identities must not alias"
            )

        payloads = {
            name: self._read_canonical_mapping(
                handle_by_name[name],
                max_bytes=(
                    MAX_CONFIG_BINDING_BYTES
                    if name == CONFIG_BINDING_INPUT_NAME
                    else MAX_CONFIG_JSON_BYTES
                ),
            )
            for name in sorted(required_names)
        }
        try:
            binding = ConfigBinding.from_dict(payloads[CONFIG_BINDING_INPUT_NAME])
        except (TypeError, ValueError) as error:
            raise AdapterValidationError(
                f"CAS config_binding.json is invalid: {error}"
            ) from error
        if binding.as_dict() != payloads[CONFIG_BINDING_INPUT_NAME]:
            raise AdapterValidationError(
                "CAS config_binding.json is not a strict ConfigBinding round-trip"
            )
        expected_identity_hashes = {
            "execution_config_sha256": binding.execution_config_sha256,
            "resolved_config_sha256": binding.resolved_config_sha256,
            "scientific_config_sha256": binding.scientific_config_sha256,
        }
        for name, expected in expected_identity_hashes.items():
            if identity_by_name[name].sha256 != expected:
                raise AdapterValidationError(
                    f"workflow unit {name} differs from ConfigBinding"
                )
        resolved = payloads["resolved_config_sha256"]
        scientific = payloads["scientific_config_sha256"]
        execution = payloads["execution_config_sha256"]
        _validate_projection_payloads(
            binding,
            resolved_config=resolved,
            scientific_config=scientific,
            execution_config=execution,
        )
        identities = ConfigContentIdentities(
            config_binding=identity_by_name[CONFIG_BINDING_INPUT_NAME],
            execution_config=identity_by_name["execution_config_sha256"],
            resolved_config=identity_by_name["resolved_config_sha256"],
            scientific_config=identity_by_name["scientific_config_sha256"],
        )
        return ResolvedConfigBinding(
            binding=binding,
            resolved_config=resolved,
            scientific_config=scientific,
            execution_config=execution,
            identities=identities,
        )

    @staticmethod
    def _read_canonical_mapping(
        handle: ReadOnlyContentHandle, *, max_bytes: int
    ) -> dict[str, Any]:
        raw = read_descriptor_bytes(
            handle.descriptor,
            context=f"CAS {handle.name}",
            max_bytes=max_bytes,
        )
        payload = parse_strict_json(raw, context=f"CAS {handle.name}")
        mapping = _require_mapping(payload, context=f"CAS {handle.name}")
        if raw != _canonical_bytes(mapping, context=f"CAS {handle.name}"):
            raise AdapterValidationError(
                f"CAS {handle.name} bytes are not canonical JSON"
            )
        if sha256_value(mapping) != handle.sha256:
            raise AdapterValidationError(
                f"CAS {handle.name} canonical hash differs from its identity"
            )
        return mapping
