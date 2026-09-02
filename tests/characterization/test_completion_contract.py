from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from posttrain_circuits.artifacts.completion import (
    ExecutionIdentity,
    ScientificCompletion,
    completion_payload,
    validate_completion_payload,
    write_completion_marker,
)
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value


def _execution(*, attempt: int = 1) -> ExecutionIdentity:
    return ExecutionIdentity(
        job_id=f"job-{attempt}",
        attempt=attempt,
        execution_profile=f"gpu-profile-{attempt}",
        manifest_sha256=("e" if attempt == 1 else "1") * 64,
        allocation_sha256=("f" if attempt == 1 else "2") * 64,
    )


def _marker(output, *, run_id: str = "run-1") -> ScientificCompletion:  # type: ignore[no-untyped-def]
    return ScientificCompletion(
        workflow_id="qwen3-v2-pilot",
        plan_sha256="9" * 64,
        unit_id="train-cell-soft-teacher-seed-42",
        task="train-cell",
        run_id=run_id,
        started_at="2026-09-01T00:00:00+00:00",
        completed_at="2026-09-01T00:01:00+00:00",
        scientific_config_sha256="a" * 64,
        execution_config_sha256="b" * 64,
        resolved_config_sha256="c" * 64,
        input_hashes={"dataset": "d" * 64},
        output_files={str(output): sha256_file(output)},
        scientific_validation={"artifact_hashes": True, "scientific_gate": True},
        execution=_execution(),
    )


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    payload["sha256"] = sha256_value(
        {key: value for key, value in payload.items() if key != "sha256"}
    )
    return payload


@pytest.mark.unit
def test_completion_marker_is_output_validated_and_idempotent(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "result.json"
    output.write_text('{"passed": true}\n', encoding="utf-8")
    path = tmp_path / "completion.json"
    marker = _marker(output)
    roots = (tmp_path,)
    first = write_completion_marker(path, marker, approved_roots=roots)
    second = write_completion_marker(path, marker, approved_roots=roots)
    assert second == first
    assert (
        validate_completion_payload(first, marker_path=path, approved_roots=roots)
        == first
    )
    assert not ({"status", "retry", "exit_code", "scheduler_state", "lease_id"} & set(first))
    for conflicting in (
        replace(marker, workflow_id="another-workflow"),
        replace(marker, plan_sha256="8" * 64),
        replace(marker, unit_id="another-unit"),
        replace(marker, run_id="run-2"),
    ):
        with pytest.raises(FileExistsError, match="conflicting"):
            write_completion_marker(path, conflicting, approved_roots=roots)


@pytest.mark.unit
def test_retried_execution_reuses_first_scientific_completion(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "result.bin"
    output.write_bytes(b"verified")
    path = tmp_path / "completion.json"
    roots = (tmp_path,)
    first = write_completion_marker(path, _marker(output), approved_roots=roots)
    first_bytes = path.read_bytes()
    retry = replace(
        _marker(output),
        started_at="2026-09-01T00:02:00Z",
        completed_at="2026-09-01T00:03:00Z",
        execution=_execution(attempt=2),
    )
    observed = write_completion_marker(path, retry, approved_roots=roots)
    assert observed == first
    assert observed["execution"] == first["execution"]
    assert observed["execution"]["attempt"] == 1
    assert path.read_bytes() == first_bytes


@pytest.mark.unit
def test_completion_refuses_failed_validation_tamper_and_missing_outputs(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "result.bin"
    output.write_bytes(b"valid")
    path = tmp_path / "completion.json"
    roots = (tmp_path,)
    marker = _marker(output)
    with pytest.raises(ValueError, match="validation failed"):
        write_completion_marker(
            path,
            replace(marker, scientific_validation={"scientific_gate": False}),
            approved_roots=roots,
        )
    payload = write_completion_marker(path, marker, approved_roots=roots)
    output.write_bytes(b"changed")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_completion_payload(payload, marker_path=path, approved_roots=roots)
    output.unlink()
    with pytest.raises(FileNotFoundError, match="output is missing"):
        validate_completion_payload(payload, marker_path=path, approved_roots=roots)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["sha256"] == payload["sha256"]


@pytest.mark.unit
def test_completion_schema_rejects_missing_extra_and_invalid_execution_fields(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "result.bin"
    output.write_bytes(b"valid")
    path = tmp_path / "completion.json"
    roots = (tmp_path,)
    base = completion_payload(_marker(output))

    missing = copy.deepcopy(base)
    missing.pop("unit_id")
    with pytest.raises(ValueError, match="fields differ from schema"):
        validate_completion_payload(
            _rehash(missing), marker_path=path, approved_roots=roots
        )

    extra = copy.deepcopy(base)
    extra["scheduler_state"] = "COMPLETED"
    with pytest.raises(ValueError, match="fields differ from schema"):
        validate_completion_payload(
            _rehash(extra), marker_path=path, approved_roots=roots
        )

    invalid_executions = []
    for mutation in ("missing", "extra", "zero", "string"):
        candidate = copy.deepcopy(base["execution"])
        assert isinstance(candidate, dict)
        if mutation == "missing":
            candidate.pop("job_id")
        elif mutation == "extra":
            candidate["lease_id"] = "legacy-lease"
        elif mutation == "zero":
            candidate["attempt"] = 0
        else:
            candidate["attempt"] = "2"
        invalid_executions.append(candidate)
    for invalid_execution in invalid_executions:
        invalid = copy.deepcopy(base)
        invalid["execution"] = invalid_execution
        with pytest.raises(ValueError):
            validate_completion_payload(
                _rehash(invalid), marker_path=path, approved_roots=roots
            )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("started_at", "2026-09-01T00:00:00", "timezone-aware UTC"),
        ("completed_at", "2026-09-01T01:00:00+01:00", "must use UTC"),
        ("completed_at", "not-a-time", "ISO-8601"),
        ("completed_at", "2026-08-31T23:59:59Z", "must not precede"),
        ("plan_sha256", "not-a-digest", "plan_sha256"),
    ),
)
def test_completion_rejects_invalid_time_and_plan_identity(
    tmp_path,
    field: str,
    value: str,
    message: str,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "result.bin"
    output.write_bytes(b"valid")
    marker = replace(_marker(output), **{field: value})
    with pytest.raises(ValueError, match=message):
        write_completion_marker(
            tmp_path / "completion.json",
            marker,
            approved_roots=(tmp_path,),
        )


@pytest.mark.unit
def test_completion_rejects_out_of_root_and_symlink_paths(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="outside approved roots"):
        write_completion_marker(
            approved / "completion.json",
            _marker(outside),
            approved_roots=(approved,),
        )

    output = approved / "result.bin"
    output.write_bytes(b"inside")
    with pytest.raises(ValueError, match="outside approved roots"):
        write_completion_marker(
            tmp_path / "outside-completion.json",
            _marker(output),
            approved_roots=(approved,),
        )

    output_link = approved / "result-link.bin"
    output_link.symlink_to(output)
    with pytest.raises(ValueError, match="output must not be a symlink"):
        write_completion_marker(
            approved / "symlink-output-completion.json",
            _marker(output_link),
            approved_roots=(approved,),
        )

    marker_target = approved / "marker-target.json"
    marker_target.write_text("{}\n", encoding="utf-8")
    marker_link = approved / "completion-link.json"
    marker_link.symlink_to(marker_target)
    with pytest.raises(ValueError, match="marker target must not be a symlink"):
        write_completion_marker(
            marker_link,
            _marker(output),
            approved_roots=(approved,),
        )

    actual_parent = approved / "actual-parent"
    actual_parent.mkdir()
    linked_parent = approved / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="must not traverse a symlink"):
        write_completion_marker(
            linked_parent / "completion.json",
            _marker(output),
            approved_roots=(approved,),
        )
