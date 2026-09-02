from __future__ import annotations

import json

import pytest

from posttrain_circuits.artifacts.config_bindings import bind_config
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.runs import (
    RunManifest,
    finalize_run_directory,
    initialize_run_directory,
    validate_run_manifest_payload,
)


def _manifest() -> tuple[RunManifest, object]:
    config_binding = bind_config(
        {"seed": 7},
        input_artifact_hashes={},
        execution_context={"entrypoint": "characterization-test"},
    )
    experiment_binding = {
        "fixture": "characterization-run-binding-v3",
        "factorial_design_sha256": "f" * 64,
        "scientific_config_sha256": config_binding.scientific_config_sha256,
    }
    manifest = RunManifest(
        run_id="characterization-run",
        experiment_cell="offline_hard",
        seed=7,
        model_id="local/tiny-qwen",
        model_revision="local-random-v1",
        tokenizer_id="local/tiny-tokenizer",
        tokenizer_revision="local-char-v1",
        resolved_model_commit="model-commit",
        resolved_tokenizer_commit="tokenizer-commit",
        dataset_hashes={"train": "dataset-hash"},
        rollout_bank_hash="bank-hash",
        prompt_schedule_hash="schedule-hash",
        experiment_binding=experiment_binding,
        experiment_binding_sha256=sha256_value(experiment_binding),
        factorial_design_sha256="f" * 64,
        config_binding=config_binding.as_dict(),
        resolved_config_sha256=config_binding.resolved_config_sha256,
        scientific_config_sha256=config_binding.scientific_config_sha256,
        execution_config_sha256=config_binding.execution_config_sha256,
        execution_context=config_binding.as_dict()["execution_context"],
        resolved_config_yaml_sha256="0" * 64,
        git_commit="git-commit",
        dirty_working_tree=False,
    )
    return manifest, config_binding


@pytest.mark.unit
def test_run_directory_file_layout_and_manifest_lifecycle(tmp_path) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "run"
    manifest, config_binding = _manifest()
    initialize_run_directory(root, {"seed": 7}, config_binding, manifest)
    assert {path.name for path in root.iterdir()} == {
        "checkpoints",
        "evaluations",
        "resolved_config.yaml",
        "config_binding.json",
        "manifest.json",
        "environment.json",
        "metrics.jsonl",
        "git_diff.patch",
    }
    initial = validate_run_manifest_payload(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    assert initial["end_time"] is None
    assert initial["scientific_config_sha256"] == config_binding.scientific_config_sha256
    assert initial["resolved_config_yaml_sha256"] == sha256_file(
        root / "resolved_config.yaml"
    )
    assert "launch_environment" not in initial
    assert "slurm_terminal_evidence_sha256" not in initial
    assert initial["token_budget_unit"] == "global_nonpadding_model_input_tokens_processed"
    finalize_run_directory(root, manifest)
    final = validate_run_manifest_payload(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    assert final["end_time"]
    assert final["sha256"] != initial["sha256"]
