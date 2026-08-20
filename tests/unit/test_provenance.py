from __future__ import annotations

import json

import pytest

from posttrain_circuits.core.provenance import (
    RunManifest,
    finalize_run_directory,
    initialize_run_directory,
)


def _manifest(**overrides: object) -> RunManifest:
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
def test_formal_run_refuses_unavailable_git(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = _manifest(git_commit="unavailable")
    with pytest.raises(RuntimeError, match="Git provenance is unavailable"):
        initialize_run_directory(tmp_path / "formal", {"seed": 7}, manifest, require_git=True)
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
    frozen = _manifest(
        prereg_git_commit="prereg-commit",
        prereg_sha256="prereg-sha",
        prereg_dirty=False,
        dirty_working_tree=False,
    )
    frozen.validate(require_git=True)

    source_dirty = _manifest(
        prereg_git_commit="prereg-commit",
        prereg_sha256="prereg-sha",
        prereg_dirty=False,
        dirty_working_tree=True,
    )
    with pytest.raises(RuntimeError, match="source working tree is dirty"):
        source_dirty.validate(require_git=True)


@pytest.mark.unit
def test_run_manifest_records_dependencies_and_end_time(tmp_path) -> None:  # type: ignore[no-untyped-def]
    run_dir = tmp_path / "run"
    manifest = _manifest(
        teacher_id="local/teacher",
        teacher_revision="teacher-revision",
        resolved_teacher_commit="teacher-commit",
    )
    initialize_run_directory(run_dir, {"seed": 7}, manifest)
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
