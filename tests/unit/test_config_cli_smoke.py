from __future__ import annotations

from collections.abc import Callable

import pytest

from posttrain_circuits.cli.build_rollout_bank import main as build_rollout_bank_main
from posttrain_circuits.cli.build_teacher_demos import main as build_teacher_demos_main
from posttrain_circuits.cli.calibrate_task import main as calibrate_task_main
from posttrain_circuits.cli.discover_circuit import main as discover_circuit_main
from posttrain_circuits.cli.generate_task import main as generate_task_main
from posttrain_circuits.cli.run_grpo import main as run_grpo_main
from posttrain_circuits.cli.train import main as train_main

ConfigCli = Callable[[list[str] | None], None]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry_point", "experiment", "state_source", "supervision"),
    [
        (build_rollout_bank_main, "offline_hard", "fixed_bank", "hard_teacher"),
        (
            build_teacher_demos_main,
            "canonical_sft",
            "teacher_demo",
            "canonical_sft",
        ),
        (calibrate_task_main, "online_hard", "current_policy", "hard_teacher"),
        (discover_circuit_main, "offline_soft", "fixed_bank", "soft_teacher"),
        (generate_task_main, "online_soft_opd", "current_policy", "soft_teacher"),
        (
            train_main,
            "offline_verified_replay",
            "fixed_bank",
            "verified_replay",
        ),
        (run_grpo_main, "canonical_grpo", "current_policy", "grpo"),
    ],
)
def test_every_config_driven_cli_smoke_resolves_dependencies_without_writes(
    entry_point: ConfigCli,
    experiment: str,
    state_source: str,
    supervision: str,
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / experiment
    entry_point([f"experiment={experiment}", "--dry-run", "--output", str(output)])
    report = capsys.readouterr().out
    assert f"name: {experiment}" in report
    assert f"name: {state_source}" in report
    assert f"name: {supervision}" in report
    assert not output.exists()
