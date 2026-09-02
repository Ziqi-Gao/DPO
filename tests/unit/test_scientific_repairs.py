from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.causal_circuits.model.runner import load_checkpoint_into_hf_model
from posttrain_circuits.datasets.circuit_probes.cohorts import (
    build_probe_cohort_manifest,
    family_probe_pairs,
    validate_probe_cohort_manifest,
    write_probe_cohort_manifest,
)
from posttrain_circuits.cli.aggregate_results import main as aggregate_results
from posttrain_circuits.cli.run_local_fork import main as run_local_fork
from posttrain_circuits.core.config import compose_config, validate_production_training_config
from posttrain_circuits.core.readiness import validate_anti_shortcut_report
from posttrain_circuits.datasets.proofgraph.anti_shortcut import (
    build_anti_shortcut_suite,
    evaluate_anti_shortcut_suite,
)
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.datasets.proofgraph.manifests import write_dataset_family
from posttrain_circuits.datasets.proofgraph.splits import SPLITS, build_all_splits
from posttrain_circuits.utils.tiny_model import build_tiny_qwen


@pytest.mark.unit
def test_anti_shortcut_zero_accuracy_is_never_ready(tmp_path: Path) -> None:
    task = ProofGraphTask()
    examples = [task.generate(44, {"positive": True, "distractors": 1})]
    cases = build_anti_shortcut_suite(examples, seed=9, distractor_ood_count=4)
    report = evaluate_anti_shortcut_suite(
        examples,
        cases,
        lambda _example, _prompt: "<proof></proof><answer>0</answer>",
        max_shortcut_gap=0.05,
        model_checkpoint_hash="checkpoint",
    )
    assert report["iid_accuracy"] == report["transformed_accuracy"] == 0.0
    assert report["shortcut_gap"] == 0.0
    assert report["passed"] is False
    path = tmp_path / "anti.json"
    atomic_write_json(path, report)
    with pytest.raises(RuntimeError, match="capability_failures"):
        validate_anti_shortcut_report(
            path,
            max_shortcut_gap=0.05,
            expected_model_checkpoint_hash="checkpoint",
        )


@pytest.mark.unit
def test_qwen_production_profile_resolves_without_smoke_defaults() -> None:
    config = compose_config(["production=qwen_primary", "experiment=offline_soft"])
    validate_production_training_config(config)
    assert config["model"]["model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config["teacher"]["model_name_or_path"] == "Qwen/Qwen2.5-7B-Instruct"
    assert config["task"]["name"] == "proofgraph"
    assert config["trainer"]["backend"] == "accelerate"
    assert config["trainer"]["max_steps"] > config["production_safety"]["max_smoke_steps"]
    assert config["trainer"]["token_budget"] > config["production_safety"]["max_smoke_tokens"]
    assert config["task"]["dataset_family_path"]


@pytest.mark.unit
def test_production_profile_rejects_wrong_teacher_family() -> None:
    with pytest.raises(ValueError, match="wrong production model/teacher pair"):
        compose_config(
            [
                "production=qwen_primary",
                "experiment=offline_soft",
                "teacher=gemma2_teacher_9b",
            ]
        )


def _probe_manifest(root: Path) -> dict[str, object]:
    split_examples = build_all_splits(
        ProofGraphTask(),
        split_sizes={split: 4 for split in SPLITS},
        base_seed=913,
        difficulty={},
    )
    write_dataset_family(root, split_examples)
    family = load_dataset_family(root)
    scores = {}
    for subset in ("discovery", "validation"):
        for index, pair in enumerate(
            family_probe_pairs(family, subset=subset, limit_pairs=2)
        ):
            for example in pair.examples:
                scores[example.example_id] = {
                    "initial_correct": index == 0,
                    "learnable_after_post_training": True,
                }
    return build_probe_cohort_manifest(
        family,
        scores,
        initial_student_checkpoint_hash="d" * 64,
        scoring_manifest_hash="e" * 64,
        eligibility_evidence_ancestry=[
            {
                "calibration_checkpoint_sha256": "a" * 64,
                "calibration_run_manifest_sha256": "b" * 64,
                "calibration_run_id": "calibration",
                "experiment_binding_sha256": "c" * 64,
            }
        ],
        limit_pairs_per_split=2,
    )


@pytest.mark.unit
def test_probe_manifest_rejects_exact_byte_tampering(tmp_path: Path) -> None:
    manifest = _probe_manifest(tmp_path / "family")
    output = tmp_path / "cohort"
    write_probe_cohort_manifest(output, manifest)
    path = output / "manifest.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    example = tampered["cohorts"]["base_capable"]["discovery"]["pairs"][0][
        "examples"
    ][0]
    example["example_id"] = str(example["example_id"]) + "-tampered"
    atomic_write_json(path, tampered)
    with pytest.raises(ValueError, match="top-level manifest hash mismatch"):
        validate_probe_cohort_manifest(path)


@pytest.mark.unit
def test_checkpoint_loader_changes_actual_qwen_logits(tmp_path: Path) -> None:
    first = build_tiny_qwen(1).eval()
    second = build_tiny_qwen(2).eval()
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    torch.save({"model": first.state_dict()}, first_path)
    torch.save({"model": second.state_dict()}, second_path)
    target = build_tiny_qwen(3).eval()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    load_checkpoint_into_hf_model(target, first_path, expected_sha256=sha256_file(first_path))
    first_logits = target(input_ids=ids).logits.detach().clone()
    load_checkpoint_into_hf_model(target, second_path, expected_sha256=sha256_file(second_path))
    second_logits = target(input_ids=ids).logits.detach().clone()
    assert not torch.equal(first_logits, second_logits)


@pytest.mark.unit
def test_local_fork_out_of_tolerance_exits_nonzero(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    bundle_id = "fork-" + sha256_value({"bundle_id": ""})[:16]
    payload = {
        "model_spec": {"model_name_or_path": "local/tiny-qwen", "seed": 1},
        "optimizer": {"param_groups": [{"lr": 1e-4}]},
        "manifest": {
            "bundle_id": bundle_id,
            "manifest_sha256": "a" * 64,
            "protocol_id": "shared_state_matched_output_kl_v1",
            "local_fork_spec_sha256": "b" * 64,
            "formal_binding": {},
        },
    }
    monkeypatch.setattr("posttrain_circuits.cli.run_local_fork.load_fork_bundle", lambda _path: payload)

    def fake_run_branch(*, branch, **_kwargs):  # type: ignore[no-untyped-def]
        kl = 0.1 if branch == "hard_teacher" else 0.5
        return {"probe_output_kl_new_to_fork": kl, "parameter_update_norm": 1.0}

    monkeypatch.setattr("posttrain_circuits.cli.run_local_fork.run_branch", fake_run_branch)
    monkeypatch.setattr(
        "posttrain_circuits.cli.run_local_fork.validate_local_fork_report",
        lambda report, **_kwargs: report,
    )
    output = tmp_path / "result.json"
    (tmp_path / "bundle.pt").write_bytes(b"bound-local-fork-test-bundle")
    with pytest.raises(RuntimeError, match="outside tolerance"):
        run_local_fork(
            [
                "--bundle",
                str(tmp_path / "bundle.pt"),
                "--output",
                str(output),
                "--horizons",
                "1",
            ]
        )
    assert json.loads(output.read_text(encoding="utf-8"))["valid_for_primary_analysis"] is False


@pytest.mark.unit
def test_aggregate_preserves_every_seed_and_checkpoint(tmp_path: Path) -> None:
    run_dirs = []
    for seed in (42, 43):
        run_dir = tmp_path / f"seed-{seed}"
        run_dir.mkdir()
        atomic_write_json(
            run_dir / "manifest.json",
            {
                "experiment_cell": "offline_hard",
                "seed": seed,
                "run_id": f"run-{seed}",
                "dataset_hashes": {"validation_manifest": "validation"},
            },
        )
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps({"step": step, "parameter_update_norm": seed + step}) for step in (1, 2))
            + "\n",
            encoding="utf-8",
        )
        run_dirs.append(run_dir)
    output = tmp_path / "aggregate.json"
    aggregate_results([*(str(path) for path in run_dirs), "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["observation_count"] == 4
    assert {(row["seed"], row["checkpoint"]) for row in payload["observations"]} == {
        (42, 1),
        (42, 2),
        (43, 1),
        (43, 2),
    }
