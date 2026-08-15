from __future__ import annotations

from collections.abc import Callable

import pytest

from posttrain_circuits.cli.build_rollout_bank import main as build_rollout_bank
from posttrain_circuits.cli.build_teacher_demos import main as build_teacher_demos
from posttrain_circuits.cli.create_fork_bundle import main as create_fork_bundle
from posttrain_circuits.cli.discover_circuit import main as discover_circuit
from posttrain_circuits.cli.evaluate_circuit import main as evaluate_circuit
from posttrain_circuits.cli.run_grpo import main as run_grpo
from posttrain_circuits.cli.score_teacher import main as score_teacher
from posttrain_circuits.cli.train import main as train
from posttrain_circuits.models import loading

ProductionCli = Callable[[list[str] | None], None]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("entry_point", "arguments"),
    [
        (
            build_rollout_bank,
            ["model=qwen25_1p5b"],
        ),
        (
            build_teacher_demos,
            ["experiment=canonical_sft", "teacher=qwen25_teacher_7b"],
        ),
        (
            train,
            [
                "experiment=offline_hard",
                "model=qwen25_1p5b",
                "teacher=qwen25_teacher_7b",
            ],
        ),
        (
            run_grpo,
            ["experiment=canonical_grpo", "model=qwen25_1p5b"],
        ),
        (
            discover_circuit,
            ["model=qwen25_1p5b"],
        ),
        (
            score_teacher,
            ["model=qwen25_teacher_7b"],
        ),
        (
            evaluate_circuit,
            ["model=qwen25_1p5b"],
        ),
        (
            create_fork_bundle,
            ["model=qwen25_1p5b", "teacher=qwen25_teacher_7b"],
        ),
    ],
)
def test_every_production_cli_dry_run_never_loads_or_writes(
    entry_point: ProductionCli,
    arguments: list[str],
    monkeypatch,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("dry-run attempted to load or download a model")

    monkeypatch.setattr(
        loading.AutoModelForCausalLM,
        "from_pretrained",
        forbidden,
    )
    monkeypatch.setattr(
        loading.AutoTokenizer,
        "from_pretrained",
        forbidden,
    )
    output = tmp_path / "production-output"
    entry_point([*arguments, "--dry-run", "--output", str(output)])
    assert not output.exists()
