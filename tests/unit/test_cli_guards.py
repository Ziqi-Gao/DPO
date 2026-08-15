from __future__ import annotations

import pytest

from posttrain_circuits.cli.score_teacher import main as score_teacher_main


@pytest.mark.unit
@pytest.mark.parametrize("overrides", [[], ["model=qwen25_teacher_7b"]])
def test_teacher_dry_run_has_no_output_side_effects(tmp_path, overrides: list[str]) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "teacher-scores"
    score_teacher_main([*overrides, "--dry-run", "--output", str(output)])
    assert not output.exists()
