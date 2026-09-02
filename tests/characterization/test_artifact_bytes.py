from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from posttrain_circuits.artifacts.config_bindings import (
    ConfigBinding,
    bind_config,
    recompute_config_binding,
    validate_config_binding,
)
from posttrain_circuits.artifacts.datasets import DatasetManifest
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json, publish_json_once
from posttrain_circuits.artifacts.runs import validate_run_manifest_payload
from posttrain_circuits.experiments.protocols.specs import (
    validate_run_manifest_experiment_binding,
)


@pytest.mark.unit
def test_atomic_json_retains_historical_bytes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "artifact.json"
    atomic_write_json(path, {"z": [2, 1], "a": {"ok": True}})
    assert path.read_bytes() == (
        b'{\n'
        b'  "a": {\n'
        b'    "ok": true\n'
        b'  },\n'
        b'  "z": [\n'
        b'    2,\n'
        b'    1\n'
        b'  ]\n'
        b'}\n'
    )


@pytest.mark.unit
def test_atomic_json_replace_failure_cleans_staging_file(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    def fail_replace(source, destination) -> None:  # type: ignore[no-untyped-def]
        raise OSError(f"replace failed: {source} -> {destination}")

    monkeypatch.setattr("posttrain_circuits.artifacts.io.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(tmp_path / "artifact.json", {"ok": True})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_publish_json_once_is_idempotent_and_never_clobbers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "published.json"
    first = {"workflow_id": "workflow-1", "plan": {"units": ["one"]}}
    assert publish_json_once(path, first) == first
    first_bytes = path.read_bytes()
    assert publish_json_once(path, copy.deepcopy(first)) == first
    with pytest.raises(FileExistsError, match="conflicting JSON artifact"):
        publish_json_once(path, {"workflow_id": "workflow-2"})
    assert path.read_bytes() == first_bytes
    assert [candidate for candidate in tmp_path.iterdir() if candidate.name.startswith(".")] == []


@pytest.mark.unit
def test_dataset_manifest_hash_excludes_time_but_binds_examples() -> None:
    arguments = {
        "dataset_id": "proofgraph-train-1",
        "generator_version": "proofgraph-v3-signed-paired",
        "git_commit": "abc",
        "task_config": {"depth": 2},
        "split_name": "train",
        "seed_range": (1, 2),
        "num_examples": 2,
        "difficulty_distribution": {"depth": {"2": 2}},
    }
    examples = [{"example_id": "one"}, {"example_id": "two"}]
    first = DatasetManifest(**arguments, created_at="2026-01-01T00:00:00+00:00").finalize(examples)
    second = DatasetManifest(**arguments, created_at="2027-01-01T00:00:00+00:00").finalize(examples)
    changed = DatasetManifest(**arguments, created_at=first.created_at).finalize(
        [{"example_id": "one"}, {"example_id": "changed"}]
    )
    assert first.sha256 == second.sha256
    assert first.sha256 != changed.sha256
    assert "created_at" not in first.content_payload()
    assert "sha256" not in first.content_payload()


@pytest.mark.unit
def test_run_manifest_reader_rejects_legacy_schema() -> None:
    payload = {
        "run_id": "historical-run-1",
        "experiment_cell": "offline_hard",
        "seed": 42,
        "model_id": "model",
        "dataset_hashes": {"train": "dataset"},
        "token_budget": 1024,
        "token_budget_unit": "global_nonpadding_model_input_tokens",
        "slurm_terminal_evidence_sha256": "a" * 64,
    }
    payload["sha256"] = sha256_value(payload)
    with pytest.raises(ValueError, match="ExperimentBinding"):
        validate_run_manifest_payload(payload)
    partial = dict(payload)
    partial.pop("sha256")
    partial["experiment_binding"] = {"factorial_design_sha256": "f" * 64}
    partial["sha256"] = sha256_value(partial)
    with pytest.raises(ValueError, match="linkage is partial"):
        validate_run_manifest_payload(partial)
    payload["seed"] = 7
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_run_manifest_payload(payload)


@pytest.mark.unit
def test_config_binding_separates_storage_from_scientific_identity() -> None:
    config = {
        "seed": 42,
        "output_root": "/data/del6500/OPD/outputs",
        "prereg_path": "prereg/qwen3_v2.yaml",
        "model": {"model_name_or_path": "Qwen/Qwen3-1.7B"},
        "supervision": {"name": "soft_teacher"},
        "trainer": {"token_budget": 1000},
        "state_source": {"store_path": "/data/del6500/OPD/banks/common"},
    }
    config["trainer"]["max_steps"] = 10
    inputs = {
        "prereg_path": "a" * 64,
        "state_source.store_path": "b" * 64,
    }
    first = bind_config(config, input_artifact_hashes=inputs)
    assert first.schema_version == 3
    assert isinstance(first.storage_locators, tuple)
    assert validate_config_binding(config, first) is first
    assert recompute_config_binding(config, first) == first
    relocated = copy.deepcopy(config)
    relocated["output_root"] = "/scr/del6500/OPD/outputs"
    relocated["state_source"]["store_path"] = "/scr/del6500/OPD/banks/common"
    second = bind_config(relocated, input_artifact_hashes=inputs)
    assert first.scientific_config_sha256 == second.scientific_config_sha256
    assert first.execution_config_sha256 != second.execution_config_sha256
    assert first.resolved_config_sha256 != second.resolved_config_sha256

    variants = []
    for path, value in (
        (("seed",), 43),
        (("model", "model_name_or_path"), "Qwen/Qwen3-4B"),
        (("supervision", "name"), "hard_teacher"),
        (("trainer", "token_budget"), 2000),
        (("trainer", "max_steps"), 11),
    ):
        variant = copy.deepcopy(config)
        if len(path) == 1:
            variant[path[0]] = value
        else:
            variant[path[0]][path[1]] = value
        variants.append(variant)
    assert all(
        bind_config(variant, input_artifact_hashes=inputs).scientific_config_sha256
        != first.scientific_config_sha256
        for variant in variants
    )
    rebound = bind_config(config, input_artifact_hashes={"prereg_path": "b" * 64})
    assert rebound.scientific_config_sha256 != first.scientific_config_sha256

    serialized = first.as_dict()
    serialized["input_artifact_hashes"]["prereg_path"] = "0" * 64
    assert dict(first.input_artifact_hashes) == inputs

    with pytest.raises(ValueError, match="does not match"):
        validate_config_binding(relocated, first)
    with pytest.raises(ValueError, match="does not match"):
        validate_config_binding(
            config,
            replace(first, resolved_config_sha256="0" * 64),
        )


@pytest.mark.unit
def test_config_binding_normalizes_overrides_and_rejects_invalid_bindings() -> None:
    config = {
        "seed": 42,
        "output_root": "/data/del6500/OPD/outputs",
        "prereg_path": "prereg/qwen3_v2.yaml",
    }
    inputs = {"prereg_path": "a" * 64}
    first = bind_config(
        config,
        input_artifact_hashes=inputs,
        execution_context={"workers": 4, "launcher": {"profile": "gpu", "precision": "bf16"}},
    )
    reordered = bind_config(
        config,
        input_artifact_hashes=inputs,
        execution_context={"launcher": {"precision": "bf16", "profile": "gpu"}, "workers": 4},
    )
    changed = bind_config(
        config,
        input_artifact_hashes=inputs,
        execution_context={"workers": 2, "launcher": {"profile": "gpu", "precision": "bf16"}},
    )
    assert first.execution_context_json == reordered.execution_context_json
    assert first.execution_config_sha256 == reordered.execution_config_sha256
    assert first.scientific_config_sha256 == changed.scientific_config_sha256
    assert first.resolved_config_sha256 == changed.resolved_config_sha256
    assert first.execution_config_sha256 != changed.execution_config_sha256
    assert validate_config_binding(config, first) is first

    with pytest.raises(ValueError, match="invalid execution-context JSON"):
        recompute_config_binding(
            config,
            replace(first, execution_context_json="{"),
        )
    with pytest.raises(ValueError, match="unsupported config-binding schema"):
        recompute_config_binding(config, replace(first, schema_version=2))
    with pytest.raises(ValueError, match="unsupported config-binding schema"):
        ConfigBinding.from_dict({**first.as_dict(), "schema_version": 2})
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        bind_config(config, input_artifact_hashes={"prereg_path": "invalid"})
