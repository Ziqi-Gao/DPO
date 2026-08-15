from __future__ import annotations

import json

import pytest

from posttrain_circuits.cli.build_teacher_demos import SmokeProofTeacher
from posttrain_circuits.cli.build_teacher_demos import main as build_teacher_demos_main
from posttrain_circuits.cli.train import main as train_main
from posttrain_circuits.teacher.demo_generation import (
    TeacherDemoGenerationConfig,
    generate_teacher_demonstrations,
    read_teacher_demo_store,
    write_teacher_demo_store,
)
from posttrain_circuits.utils.smoke import build_smoke_examples


@pytest.mark.integration
def test_teacher_demo_pipeline_retains_only_exact_successes_with_full_manifest(
    tmp_path,
    tokenizer,
) -> None:  # type: ignore[no-untyped-def]
    examples = build_smoke_examples(3, seed=9)
    config = TeacherDemoGenerationConfig(
        teacher_id="teacher/id",
        teacher_revision="requested-revision",
        resolved_teacher_commit="resolved-commit",
        generation_seed=101,
        temperature=0.7,
        top_p=0.9,
        candidates_per_prompt=4,
    )
    result = generate_teacher_demonstrations(
        examples,
        tokenizer,
        SmokeProofTeacher(),
        config,
    )
    assert result.total_candidates == 12
    assert result.successful_candidates == 3
    assert result.retention_rate == pytest.approx(0.25)
    assert all(record.verifier_reward == 1.0 for record in result.records)

    root = tmp_path / "teacher-demos"
    manifest = write_teacher_demo_store(root, result)
    records, loaded_manifest = read_teacher_demo_store(root)
    generation = loaded_manifest["teacher_demo_generation"]
    assert manifest["store_kind"] == "teacher_demo"
    assert generation["teacher_id"] == "teacher/id"
    assert generation["teacher_revision"] == "requested-revision"
    assert generation["resolved_teacher_commit"] == "resolved-commit"
    assert generation["generation_seed"] == 101
    assert generation["temperature"] == pytest.approx(0.7)
    assert generation["top_p"] == pytest.approx(0.9)
    assert generation["candidates_per_prompt"] == 4
    assert generation["verifier_version"] == "proofgraph-exact-v1"
    assert generation["retention_rate"] == pytest.approx(0.25)
    assert generation["prompt_manifest_hash"] == result.prompt_manifest_hash
    assert len(records) == 3
    assert all(record.teacher_id == "teacher/id" for record in records)


@pytest.mark.integration
def test_teacher_demo_manifest_tampering_is_detected(tmp_path, tokenizer) -> None:  # type: ignore[no-untyped-def]
    config = TeacherDemoGenerationConfig(
        teacher_id="teacher/id",
        teacher_revision="revision",
        resolved_teacher_commit="commit",
        generation_seed=2,
        temperature=1.0,
        top_p=1.0,
        candidates_per_prompt=1,
    )
    result = generate_teacher_demonstrations(
        build_smoke_examples(1),
        tokenizer,
        SmokeProofTeacher(),
        config,
    )
    root = tmp_path / "teacher-demos"
    write_teacher_demo_store(root, result)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["teacher_demo_generation"]["generation_seed"] = 999
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        read_teacher_demo_store(root)


@pytest.mark.integration
def test_canonical_sft_reads_independent_teacher_demo_store(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    store = tmp_path / "teacher-demos"
    run_dir = tmp_path / "canonical-sft"
    build_teacher_demos_main(
        [
            "experiment=canonical_sft",
            "task.num_examples=2",
            "state_source.num_candidates=2",
            "--output",
            str(store),
        ]
    )
    train_main(
        [
            "experiment=canonical_sft",
            f"state_source.store_path={store}",
            "trainer.batch_size=2",
            "trainer.max_steps=1",
            "trainer.max_completion_length=2",
            "--output",
            str(run_dir),
        ]
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8"))
    assert manifest["teacher_demo_generation"]["resolved_teacher_commit"]
    assert manifest["teacher_demo_generation"]["candidates_per_prompt"] == 2
    assert manifest["end_time"]
    assert metrics["generated_trajectories"] == 2
    assert metrics["successful_trajectories"] == 2
    assert metrics["retry_count"] == 0
    assert metrics["validation_accuracy"] is not None
    assert metrics["exact_proof_accuracy"] is not None
    assert metrics["format_validity"] is not None
