from __future__ import annotations

from dataclasses import asdict
import json

import pytest

from posttrain_circuits.cli.build_teacher_demos import SmokeProofTeacher
from posttrain_circuits.cli.build_teacher_demos import main as build_teacher_demos_main
from posttrain_circuits.cli.train import main as train_main
from posttrain_circuits.datasets.teacher_demos.store import (
    read_teacher_demo_store,
    write_teacher_demo_store,
)
from posttrain_circuits.learning.teacher.demo_generation import (
    TeacherDemoGenerationConfig,
    generate_teacher_demonstrations,
)
from posttrain_circuits.utils.smoke import build_smoke_examples


def _write(root, result):  # type: ignore[no-untyped-def]
    return write_teacher_demo_store(
        root,
        result.attempts,
        ordered_prompt_ids=result.ordered_prompt_ids,
        prompt_manifest_hash=result.prompt_manifest_hash,
        tokenizer_hash=result.tokenizer_hash,
        generation=asdict(result.config),
    )


@pytest.mark.integration
def test_teacher_demo_pipeline_ledgers_every_attempt_and_views_only_successes(
    tmp_path,
    tokenizer,
) -> None:  # type: ignore[no-untyped-def]
    examples = build_smoke_examples(3, seed=9)
    config = TeacherDemoGenerationConfig(
        teacher_id="teacher/id",
        teacher_revision="requested-revision",
        resolved_teacher_commit="resolved-commit",
        sampling_request_seed=101,
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
    assert len(result.attempts) == 12
    assert all(attempt.behavior_logprobs is None for attempt in result.attempts)

    root = tmp_path / "teacher-demos"
    manifest = _write(root, result)
    accepted, loaded_manifest = read_teacher_demo_store(root, require_formal=False)
    ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    view = json.loads((root / "accepted_view.json").read_text(encoding="utf-8"))
    generation = loaded_manifest["teacher_demo_generation"]
    assert manifest["store_kind"] == "teacher_demo_ledger"
    assert len(ledger["attempts"]) == 12
    assert len(view["accepted_attempts"]) == 3
    assert all("sha256" not in attempt for attempt in ledger["attempts"])
    assert generation["ledger_file_sha256"] == manifest["ledger_file_sha256"]
    assert generation["accepted_view_file_sha256"] == manifest["accepted_view_file_sha256"]
    assert len(accepted) == 3
    with pytest.raises(ValueError, match="fixture-unavailable"):
        read_teacher_demo_store(root, require_formal=True)


@pytest.mark.integration
def test_teacher_demo_ledger_tampering_is_detected(tmp_path, tokenizer) -> None:  # type: ignore[no-untyped-def]
    config = TeacherDemoGenerationConfig(
        teacher_id="teacher/id",
        teacher_revision="revision",
        resolved_teacher_commit="commit",
        sampling_request_seed=2,
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
    _write(root, result)
    ledger_path = root / "ledger.json"
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["attempts"][0]["sampling_request_seed"] = 999
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger file hash"):
        read_teacher_demo_store(root, require_formal=False)


@pytest.mark.integration
def test_canonical_sft_reads_only_the_teacher_demo_accepted_view(
    tmp_path,
    dataset_family_path,
) -> None:  # type: ignore[no-untyped-def]
    store = tmp_path / "teacher-demos"
    run_dir = tmp_path / "canonical-sft"
    build_teacher_demos_main(
        [
            "experiment=canonical_sft",
            f"task.dataset_family_path={dataset_family_path}",
            "task.num_examples=2",
            "state_source.num_candidates=2",
            "--output",
            str(store),
        ]
    )
    train_main(
        [
            "experiment=canonical_sft",
            f"task.dataset_family_path={dataset_family_path}",
            f"state_source.store_path={store}",
            "trainer.batch_size=2",
            "trainer.max_steps=1",
            "trainer.max_completion_length=2",
            "--output",
            str(run_dir),
        ]
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metric_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    metrics = metric_rows[-1]
    assert manifest["teacher_demo_generation"]["resolved_teacher_commit"]
    assert manifest["teacher_demo_generation"]["candidates_per_prompt"] == 2
    assert manifest["teacher_demo_generation"]["ledger_file_sha256"]
    assert manifest["teacher_demo_generation"]["accepted_view_file_sha256"]
    assert manifest["end_time"]
    assert metrics["generated_trajectories"] == 2
    assert metrics["successful_trajectories"] == 2
    assert metrics["retry_count"] == 0
