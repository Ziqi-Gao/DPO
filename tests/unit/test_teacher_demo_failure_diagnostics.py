from __future__ import annotations

import json
import stat
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from posttrain_circuits.cli import build_teacher_demos as cli
from posttrain_circuits.datasets.teacher_demos.store import read_teacher_demo_store
from posttrain_circuits.learning.teacher.demo_generation import (
    TeacherDemoGenerationConfig,
    generate_teacher_demonstrations,
)
from posttrain_circuits.utils.smoke import build_smoke_examples


def _result(tokenizer, *, reject: bool):  # type: ignore[no-untyped-def]
    teacher = cli.SmokeProofTeacher()

    def generate(**kwargs):  # type: ignore[no-untyped-def]
        if reject:
            kwargs["candidate_index"] = 1
        return teacher(**kwargs)

    return generate_teacher_demonstrations(
        build_smoke_examples(2, seed=9),
        tokenizer,
        generate,
        TeacherDemoGenerationConfig(
            teacher_id="teacher/id",
            teacher_revision="requested",
            resolved_teacher_commit="resolved",
            sampling_request_seed=17,
            temperature=0.7,
            top_p=0.9,
            candidates_per_prompt=1,
            max_prompt_tokens=4096,
            max_new_tokens=256,
        ),
    )


def _write(output, result):  # type: ignore[no-untyped-def]
    return cli._write_store_with_failure_diagnostics(
        output,
        result.attempts,
        ordered_prompt_ids=result.ordered_prompt_ids,
        prompt_manifest_hash=result.prompt_manifest_hash,
        tokenizer_hash=result.tokenizer_hash,
        generation=asdict(result.config),
    )


def test_failed_store_survives_source_cleanup_with_exact_private_bytes(
    tmp_path, tokenizer, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(cli, "_DIAGNOSTICS_ROOT", diagnostics)
    result = _result(tokenizer, reject=True)
    with tempfile.TemporaryDirectory(dir=tmp_path) as temporary:
        output = Path(temporary) / "teacher_demos"
        with pytest.raises(ValueError, match="zero-success"):
            _write(output, result)
        original = {path.name: path.read_bytes() for path in output.iterdir()}
    assert not output.exists()
    retained, = diagnostics.iterdir()
    assert stat.S_IMODE(retained.stat().st_mode) & 0o777 == 0o700
    assert set(original) == {"ledger.json", "accepted_view.json", "manifest.json"}
    assert {path.name: path.read_bytes() for path in retained.iterdir()} == original
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in retained.iterdir())
    _, manifest = read_teacher_demo_store(retained, require_formal=False)
    assert manifest["ready_for_formal_sft"] is False
    with pytest.raises(ValueError, match="zero-success"):
        read_teacher_demo_store(retained, require_formal=True)
    emitted = capsys.readouterr().err.removeprefix("Teacher-demo failure diagnostics: ")
    summary = json.loads(emitted)
    assert summary["path"] == str(retained)
    assert sum(summary["failure_code_counts"].values()) == 2
    assert summary["finish_reason_counts"] == {"oracle_fixture": 2}
    assert summary["attempt_count"] == 2
    assert summary["accepted_attempt_count"] == 0
    assert summary["total_prompts"] == summary["zero_success_prompt_count"] == 2
    assert summary["prompts_with_success"] == 0


def test_preservation_failure_never_replaces_scientific_error(
    tmp_path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    scientific_error = ValueError("original scientific rejection")

    def reject(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise scientific_error

    def fail_to_preserve(output, **kwargs):  # type: ignore[no-untyped-def]
        raise OSError("diagnostic disk failure")

    monkeypatch.setattr(cli, "write_teacher_demo_store", reject)
    monkeypatch.setattr(cli, "_preserve_failure_diagnostics", fail_to_preserve)
    with pytest.raises(ValueError) as captured:
        cli._write_store_with_failure_diagnostics(tmp_path / "store", [])
    assert captured.value is scientific_error
    assert "diagnostic disk failure" in capsys.readouterr().err


def test_unavailable_diagnostics_still_reports_current_candidate_coverage(
    tmp_path, tokenizer, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    result = _result(tokenizer, reject=True)
    successful = _result(tokenizer, reject=False)
    result = replace(result, attempts=[successful.attempts[0], result.attempts[1]])

    def fail_to_preserve(output, **kwargs):  # type: ignore[no-untyped-def]
        raise PermissionError("legacy scratch root is unavailable")

    monkeypatch.setattr(cli, "_preserve_failure_diagnostics", fail_to_preserve)
    output = tmp_path / "store"
    with pytest.raises(ValueError, match="zero-success"):
        _write(output, result)
    # The scientific writer still retained the original store, and its gate
    # still rejects the uncovered prompt despite one accepted demonstration.
    _, manifest = read_teacher_demo_store(output, require_formal=False)
    assert manifest["ready_for_formal_sft"] is False
    assert manifest["accepted_count"] == 1
    lines = capsys.readouterr().err.splitlines()
    assert "legacy scratch root is unavailable" in lines[0]
    summary = json.loads(lines[1].removeprefix("Teacher-demo failure diagnostics: "))
    assert summary["path"] is None
    assert summary["attempt_count"] == 2
    assert summary["accepted_attempt_count"] == 1
    assert summary["total_prompts"] == 2
    assert summary["prompts_with_success"] == summary["zero_success_prompt_count"] == 1
    assert sum(summary["failure_code_counts"].values()) == 1
    assert summary["finish_reason_counts"] == {"oracle_fixture": 2}


def test_portable_main_keeps_exact_failed_store_beside_output(
    tmp_path, dataset_family_path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    default_root = tmp_path / "unused-central-root"
    monkeypatch.setattr(cli, "_DIAGNOSTICS_ROOT", default_root)
    original_call = cli.SmokeProofTeacher.__call__

    def reject(self, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["candidate_index"] = 1
        return original_call(self, **kwargs)

    monkeypatch.setattr(cli.SmokeProofTeacher, "__call__", reject)
    output = tmp_path / "job-artifacts" / "teacher_demos"
    with pytest.raises(ValueError, match="zero-success"):
        cli.main(
            [
                "experiment=canonical_sft",
                f"task.dataset_family_path={dataset_family_path}",
                "task.num_examples=2",
                "state_source.num_candidates=2",
                "--output",
                str(output),
            ],
            diagnostics_with_output=True,
        )
    assert not default_root.exists()
    retained, = (output.parent / "teacher_demo_failure_diagnostics").iterdir()
    assert {path.name: path.read_bytes() for path in retained.iterdir()} == {
        path.name: path.read_bytes() for path in output.iterdir()
    }
    summary = json.loads(
        capsys.readouterr().err.removeprefix("Teacher-demo failure diagnostics: ")
    )
    assert summary["path"] == str(retained)
    assert summary["attempt_count"] == 4
    assert summary["zero_success_prompt_count"] == 2


def test_failure_summary_is_bounded_without_logging_responses(
    tmp_path, tokenizer, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    original = _result(tokenizer, reject=True).attempts[0]
    attempts = [
        replace(
            original,
            verification_trace={"error_code": f"error-{index:02d}-" + "x" * 1000},
            finish_reason=f"finish-{index:02d}-" + "x" * 1000,
            response_text="do not emit this raw response",
        )
        for index in range(30)
    ]
    scientific_error = ValueError("invalid attempt population")

    def reject(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise scientific_error

    monkeypatch.setattr(cli, "write_teacher_demo_store", reject)
    with pytest.raises(ValueError) as captured:
        cli._write_store_with_failure_diagnostics(tmp_path / "store", attempts)
    assert captured.value is scientific_error
    emitted = capsys.readouterr().err
    summary = json.loads(emitted.removeprefix("Teacher-demo failure diagnostics: "))
    assert "do not emit this raw response" not in emitted
    for field in ("failure_code_counts", "finish_reason_counts"):
        assert len(summary[field]) == 17
        assert sum(summary[field].values()) == 30
        assert all(len(value) <= 80 for value in summary[field])


def test_success_creates_no_diagnostics(tmp_path, tokenizer, monkeypatch, capsys) -> None:
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(cli, "_DIAGNOSTICS_ROOT", diagnostics)
    output = tmp_path / "store"
    manifest = _write(output, _result(tokenizer, reject=False))
    _, loaded = read_teacher_demo_store(output, require_formal=False)
    assert loaded == manifest
    assert manifest["zero_success_prompt_ids"] == []
    assert not diagnostics.exists()
    assert capsys.readouterr().err == ""


def test_existing_store_is_not_misreported_as_current_failure(
    tmp_path, tokenizer, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(cli, "_DIAGNOSTICS_ROOT", diagnostics)
    output = tmp_path / "store"
    _write(output, _result(tokenizer, reject=False))
    previous = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="not empty"):
        _write(output, _result(tokenizer, reject=True))
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous
    assert not diagnostics.exists()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("unsafe_name", ["ledger.json", "accepted_view.json", "manifest.json"])
def test_diagnostics_reject_symlinked_source_files(tmp_path, monkeypatch, unsafe_name) -> None:
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(cli, "_DIAGNOSTICS_ROOT", diagnostics)
    source = tmp_path / "store"
    source.mkdir()
    outside = tmp_path / "unrelated.json"
    outside.write_bytes(b"unrelated")
    for name in cli._DIAGNOSTIC_FILES:
        if name == unsafe_name:
            (source / name).symlink_to(outside)
        else:
            (source / name).write_bytes(b"{}")
    with pytest.raises(OSError):
        cli._preserve_failure_diagnostics(source)
    assert not diagnostics.exists()
    assert outside.read_bytes() == b"unrelated"


def test_diagnostics_reject_symlinked_destination_before_writing(tmp_path, monkeypatch) -> None:
    source = tmp_path / "store"
    source.mkdir()
    for name in cli._DIAGNOSTIC_FILES:
        (source / name).write_bytes(b"{}")
    outside = tmp_path / "unrelated"
    outside.mkdir()
    link = tmp_path / "diagnostics"
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(cli, "_DIAGNOSTICS_ROOT", link / "teacher_demos")
    with pytest.raises(ValueError, match="symlinks"):
        cli._preserve_failure_diagnostics(source)
    assert list(outside.iterdir()) == []
