from __future__ import annotations

import json

import pytest

from posttrain_circuits.artifacts.config_bindings import bind_config
from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.runs import (
    RunManifest,
    finalize_run_directory,
    implementation_commit_is_preregistered,
    initialize_run_directory,
    resolve_preregistration,
)


def _config_binding():  # type: ignore[no-untyped-def]
    return bind_config(
        {"seed": 7},
        input_artifact_hashes={},
        execution_context={"entrypoint": "unit-test"},
    )


def _manifest(**overrides: object) -> RunManifest:
    config_binding = _config_binding()
    experiment_binding = {
        "fixture": "unit-run-binding-v3",
        "factorial_design_sha256": "f" * 64,
        "scientific_config_sha256": config_binding.scientific_config_sha256,
    }
    values: dict[str, object] = {
        "run_id": "run-1",
        "experiment_cell": "offline_hard",
        "seed": 7,
        "model_id": "local/tiny-qwen",
        "model_revision": "local-random-v1",
        "tokenizer_id": "local/tiny-tokenizer",
        "tokenizer_revision": "local-char-v1",
        "resolved_model_commit": "model-commit",
        "resolved_tokenizer_commit": "tokenizer-commit",
        "dataset_hashes": {"train": "dataset-hash"},
        "rollout_bank_hash": "bank-hash",
        "prompt_schedule_hash": "schedule-hash",
        "experiment_binding": experiment_binding,
        "experiment_binding_sha256": sha256_value(experiment_binding),
        "factorial_design_sha256": "f" * 64,
        "config_binding": config_binding.as_dict(),
        "resolved_config_sha256": config_binding.resolved_config_sha256,
        "scientific_config_sha256": config_binding.scientific_config_sha256,
        "execution_config_sha256": config_binding.execution_config_sha256,
        "execution_context": config_binding.as_dict()["execution_context"],
        "resolved_config_yaml_sha256": "0" * 64,
        "git_commit": "git-commit",
    }
    values.update(overrides)
    return RunManifest(**values)  # type: ignore[arg-type]


@pytest.mark.unit
def test_manifest_requires_nonempty_data_hashes() -> None:
    manifest = _manifest(dataset_hashes={})
    with pytest.raises(ValueError, match="dataset hashes"):
        manifest.validate(require_git=False)
    manifest = _manifest(rollout_bank_hash="")
    with pytest.raises(ValueError, match="empty required fields"):
        manifest.validate(require_git=False)


@pytest.mark.unit
def test_new_manifest_writer_rejects_historical_token_unit() -> None:
    manifest = _manifest(token_budget_unit="global_nonpadding_model_input_tokens")
    with pytest.raises(ValueError, match="registered global token budget and unit"):
        manifest.validate(require_git=False)


@pytest.mark.unit
def test_formal_run_refuses_unavailable_git(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = _manifest(git_commit="unavailable")
    with pytest.raises(RuntimeError, match="Git provenance is unavailable"):
        initialize_run_directory(
            tmp_path / "formal",
            {"seed": 7},
            _config_binding(),
            manifest,
            require_git=True,
        )
    assert not (tmp_path / "formal").exists()


@pytest.mark.unit
def test_formal_run_requires_clean_frozen_preregistration() -> None:
    missing = _manifest(
        prereg_git_commit="unavailable",
        prereg_sha256="prereg-sha",
        prereg_dirty=False,
    )
    with pytest.raises(RuntimeError, match="no frozen Git commit"):
        missing.validate(require_git=True)
    dirty = _manifest(
        prereg_git_commit="prereg-commit",
        prereg_sha256="prereg-sha",
        prereg_dirty=True,
    )
    with pytest.raises(RuntimeError, match="differs from its frozen Git commit"):
        dirty.validate(require_git=True)
    unbound = _manifest(
        prereg_git_commit="prereg-commit",
        prereg_sha256="prereg-sha",
        prereg_dirty=False,
        dirty_working_tree=False,
    )
    with pytest.raises(RuntimeError, match="bound preregistration is missing"):
        unbound.validate(require_git=True)
    binding = resolve_preregistration({"prereg_path": "prereg/core_v2.yaml", "prereg_version": "core_v2"})
    frozen = _manifest(
        prereg_git_commit=binding.git_commit,
        prereg_sha256=binding.sha256,
        prereg_dirty=False,
        dirty_working_tree=False,
        prereg_path=str(binding.path),
        prereg_version=binding.version,
    )
    frozen.validate(require_git=True)

    source_dirty = _manifest(
        prereg_git_commit=binding.git_commit,
        prereg_sha256=binding.sha256,
        prereg_dirty=False,
        dirty_working_tree=True,
        prereg_path=str(binding.path),
        prereg_version=binding.version,
    )
    with pytest.raises(RuntimeError, match="source working tree is dirty"):
        source_dirty.validate(require_git=True)


@pytest.mark.unit
def test_preregistered_implementation_requires_a_reviewed_amendment() -> None:
    frozen = "a" * 40
    current = "b" * 40
    preregistration = {"frozen_implementation_commit": frozen}
    assert implementation_commit_is_preregistered(preregistration, frozen)
    assert not implementation_commit_is_preregistered(preregistration, current)
    preregistration["reviewed_implementation_amendments"] = [
        {
            "base_frozen_implementation_commit": frozen,
            "implementation_commit": current,
            "review_status": "accepted",
            "scientific_scope": "implementation_only_no_preregistered_design_change",
            "reviewer": "independent-reviewer-id",
            "rationale": "Scheduler and package-structure refactor only.",
        }
    ]
    assert implementation_commit_is_preregistered(preregistration, current)


@pytest.mark.unit
def test_run_manifest_records_dependencies_and_end_time(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "run"
    manifest = _manifest(
        teacher_id="local/teacher",
        teacher_revision="teacher-revision",
        resolved_teacher_commit="teacher-commit",
    )
    initialize_run_directory(run_dir, {"seed": 7}, _config_binding(), manifest)
    initial = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert initial["tokenizer_revision"] == "local-char-v1"
    assert initial["resolved_model_commit"] == "model-commit"
    assert initial["resolved_tokenizer_commit"] == "tokenizer-commit"
    assert initial["resolved_teacher_commit"] == "teacher-commit"
    assert initial["prereg_sha256"] != "unavailable"
    assert "prereg_git_commit" in initial
    assert initial["package_versions"]["torch"]
    assert initial["end_time"] is None

    finalize_run_directory(run_dir, manifest)
    final = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert final["end_time"]
