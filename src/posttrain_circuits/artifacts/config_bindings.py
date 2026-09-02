"""Separate scientific configuration identity from machine-local execution identity."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from posttrain_circuits.artifacts.hashing import canonical_json, sha256_value

_MISSING = object()
_STORAGE_LOCATOR_PATHS: tuple[tuple[str, ...], ...] = (
    ("output_root",),
    ("prereg_path",),
    ("state_source", "store_path"),
    ("task", "dataset_family_path"),
    ("anti_shortcut", "report_path"),
    ("production_safety", "readiness_report"),
    ("production_safety", "probe_cohort_manifest"),
    ("production_safety", "initial_checkpoint_path"),
    ("experiment", "random_reward_calibration_path"),
)


def _dotted(parts: tuple[str, ...]) -> str:
    return ".".join(parts)


def _get(payload: dict[str, Any], parts: tuple[str, ...]) -> Any:
    value: Any = payload
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _delete(payload: dict[str, Any], parts: tuple[str, ...]) -> None:
    parent: Any = payload
    for part in parts[:-1]:
        parent = parent[part]
    del parent[parts[-1]]


def _set(payload: dict[str, Any], parts: tuple[str, ...], value: Any) -> None:
    parent: Any = payload
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ConfigBinding:
    schema_version: int
    resolved_config_sha256: str
    scientific_config_sha256: str
    execution_config_sha256: str
    input_artifact_hashes: tuple[tuple[str, str], ...]
    storage_locators: tuple[tuple[str, str], ...]
    execution_context_json: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "resolved_config_sha256": self.resolved_config_sha256,
            "scientific_config_sha256": self.scientific_config_sha256,
            "execution_config_sha256": self.execution_config_sha256,
            "input_artifact_hashes": dict(self.input_artifact_hashes),
            "storage_locators": dict(self.storage_locators),
            "execution_context": json.loads(self.execution_context_json),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConfigBinding:
        expected = {
            "schema_version",
            "resolved_config_sha256",
            "scientific_config_sha256",
            "execution_config_sha256",
            "input_artifact_hashes",
            "storage_locators",
            "execution_context",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("config binding fields differ from schema")
        schema_version = payload["schema_version"]
        if type(schema_version) is not int or schema_version != 3:
            raise ValueError(f"unsupported config-binding schema: {schema_version}")
        inputs = payload["input_artifact_hashes"]
        locators = payload["storage_locators"]
        context = payload["execution_context"]
        if not isinstance(inputs, dict) or not isinstance(locators, dict):
            raise ValueError("config binding hashes/locators must be mappings")
        if not isinstance(context, dict):
            raise ValueError("config binding execution context must be a mapping")
        normalized_inputs = tuple(
            sorted(
                (
                    _require_name(name, name="input artifact name"),
                    _require_sha256(digest, name=f"input_artifact_hashes[{name!r}]"),
                )
                for name, digest in inputs.items()
            )
        )
        normalized_locators = tuple(
            sorted(
                (
                    _require_name(name, name="storage locator name"),
                    _require_name(locator, name=f"storage_locators[{name!r}]"),
                )
                for name, locator in locators.items()
            )
        )
        return cls(
            schema_version=schema_version,
            resolved_config_sha256=_require_sha256(
                payload["resolved_config_sha256"], name="resolved_config_sha256"
            ),
            scientific_config_sha256=_require_sha256(
                payload["scientific_config_sha256"], name="scientific_config_sha256"
            ),
            execution_config_sha256=_require_sha256(
                payload["execution_config_sha256"], name="execution_config_sha256"
            ),
            input_artifact_hashes=normalized_inputs,
            storage_locators=normalized_locators,
            execution_context_json=canonical_json(context),
        )


def bind_config(
    config: dict[str, Any],
    *,
    input_artifact_hashes: dict[str, str],
    execution_context: dict[str, Any] | None = None,
) -> ConfigBinding:
    """Bind relocatable scientific inputs by content and storage by concrete location.

    Storage locators are removed from the scientific projection. Scientific input
    content is represented only by the supplied artifact hashes; the concrete
    locators and explicit execution context form the execution projection.
    """

    if not isinstance(config, dict):
        raise TypeError("resolved config must be a mapping")
    if not isinstance(input_artifact_hashes, dict):
        raise TypeError("input artifact hashes must be a mapping")
    if execution_context is not None and not isinstance(execution_context, dict):
        raise TypeError("execution context must be a mapping or null")
    normalized_input_items = tuple(
        sorted(
            (
                _require_name(name, name="input artifact name"),
                _require_sha256(digest, name=f"input_artifact_hashes[{name!r}]"),
            )
            for name, digest in input_artifact_hashes.items()
        )
    )
    if len(dict(normalized_input_items)) != len(normalized_input_items):
        raise ValueError("input artifact names must be unique")
    normalized_inputs = dict(normalized_input_items)
    normalized_execution_json = canonical_json(execution_context or {})
    normalized_execution = json.loads(normalized_execution_json)
    scientific = copy.deepcopy(config)
    storage_locators: dict[str, str] = {}
    for parts in _STORAGE_LOCATOR_PATHS:
        observed = _get(config, parts)
        if observed is _MISSING or observed is None or str(observed) == "":
            continue
        name = _dotted(parts)
        storage_locators[name] = str(observed)
        if name in normalized_inputs:
            _set(scientific, parts, {"content_sha256": normalized_inputs[name]})
        else:
            _delete(scientific, parts)

    scientific_projection = {
        "schema_version": 3,
        "config": scientific,
        "input_artifact_hashes": normalized_inputs,
    }
    execution_projection = {
        "schema_version": 3,
        "storage_locators": storage_locators,
        "execution_context": normalized_execution,
    }
    return ConfigBinding(
        schema_version=3,
        resolved_config_sha256=sha256_value(config),
        scientific_config_sha256=sha256_value(scientific_projection),
        execution_config_sha256=sha256_value(execution_projection),
        input_artifact_hashes=normalized_input_items,
        storage_locators=tuple(sorted(storage_locators.items())),
        execution_context_json=normalized_execution_json,
    )


def recompute_config_binding(
    config: dict[str, Any],
    binding: ConfigBinding,
) -> ConfigBinding:
    """Recompute a binding from its immutable input and execution projections."""

    if (
        isinstance(binding.schema_version, bool)
        or not isinstance(binding.schema_version, int)
        or binding.schema_version != 3
    ):
        raise ValueError(f"unsupported config-binding schema: {binding.schema_version}")
    for name, digest in (
        ("resolved_config_sha256", binding.resolved_config_sha256),
        ("scientific_config_sha256", binding.scientific_config_sha256),
        ("execution_config_sha256", binding.execution_config_sha256),
    ):
        _require_sha256(digest, name=name)
    if not isinstance(binding.input_artifact_hashes, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2
        for item in binding.input_artifact_hashes
    ):
        raise ValueError("config binding input artifact hashes must be immutable pairs")
    if not isinstance(binding.storage_locators, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2
        for item in binding.storage_locators
    ):
        raise ValueError("config binding storage locators must be immutable pairs")
    normalized_inputs = tuple(
        sorted(
            (
                _require_name(name, name="input artifact name"),
                _require_sha256(digest, name=f"input_artifact_hashes[{name!r}]"),
            )
            for name, digest in binding.input_artifact_hashes
        )
    )
    if normalized_inputs != binding.input_artifact_hashes:
        raise ValueError("config binding input artifact hashes are not canonical")
    if len(dict(normalized_inputs)) != len(normalized_inputs):
        raise ValueError("config binding input artifact names must be unique")
    normalized_locators = tuple(
        sorted(
            (
                _require_name(name, name="storage locator name"),
                _require_name(locator, name=f"storage_locators[{name!r}]"),
            )
            for name, locator in binding.storage_locators
        )
    )
    if normalized_locators != binding.storage_locators:
        raise ValueError("config binding storage locators are not canonical")
    if len(dict(normalized_locators)) != len(normalized_locators):
        raise ValueError("config binding storage locator names must be unique")
    if not isinstance(binding.execution_context_json, str):
        raise ValueError("config binding execution context must be canonical JSON")
    try:
        execution_context = json.loads(binding.execution_context_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("config binding has invalid execution-context JSON") from error
    if not isinstance(execution_context, dict):
        raise ValueError("config binding execution context must decode to a mapping")
    if canonical_json(execution_context) != binding.execution_context_json:
        raise ValueError("config binding execution context is not canonical JSON")
    return bind_config(
        config,
        input_artifact_hashes=dict(normalized_inputs),
        execution_context=execution_context,
    )


def validate_config_binding(
    config: dict[str, Any],
    binding: ConfigBinding,
) -> ConfigBinding:
    """Fail closed when a persisted binding cannot be reproduced."""

    expected = recompute_config_binding(config, binding)
    if expected != binding:
        raise ValueError(
            "config binding does not match the resolved configuration: "
            f"expected={expected.as_dict()}, observed={binding.as_dict()}"
        )
    return binding
