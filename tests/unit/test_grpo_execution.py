from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from posttrain_circuits.cli.run_grpo import main as run_grpo_main
from posttrain_circuits.training.grpo_backend import TrlGrpoBackend
from posttrain_circuits.training.grpo_data import build_grpo_rows_and_reward
from posttrain_circuits.utils.smoke import build_smoke_examples


@pytest.mark.unit
def test_grpo_reward_adapter_supports_exact_format_and_matched_random() -> None:
    examples = build_smoke_examples(2)
    rows, exact = build_grpo_rows_and_reward(
        examples,
        reward_name="exact",
        seed=3,
    )
    completions = [
        "<proof>\n"
        + ("S01: R01(F01,F02) -> Q\n" if examples[0].label else "")
        + f"</proof>\n<answer>{examples[0].label}</answer>",
        "invalid",
    ]
    assert len(rows) == 2
    assert exact([row["prompt"] for row in rows], completions)[1] == 0.0
    _, format_reward = build_grpo_rows_and_reward(
        examples,
        reward_name="format_only",
        seed=3,
    )
    assert (
        format_reward(
            [row["prompt"] for row in rows],
            completions,
        )[1]
        == 0.0
    )
    _, random_reward = build_grpo_rows_and_reward(
        examples,
        reward_name="matched_random",
        seed=3,
    )
    values = random_reward(
        [row["prompt"] for row in rows],
        completions,
    )
    assert sum(values) == sum(exact([row["prompt"] for row in rows], completions))


@pytest.mark.unit
def test_run_grpo_builds_official_backend_and_calls_train(
    monkeypatch,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    state = {"trained": False, "rows": 0}

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):  # type: ignore[no-untyped-def]
            state["rows"] = len(rows)
            return rows

    class FakeTrainer:
        def __init__(self, model):  # type: ignore[no-untyped-def]
            self.model = model
            self.state = SimpleNamespace(global_step=0)
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda _: 1.0,
            )

        def train(self):  # type: ignore[no-untyped-def]
            with torch.no_grad():
                next(self.model.parameters()).add_(0.01)
            self.state.global_step = 1
            state["trained"] = True
            return "ok"

    def fake_build(self, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["train_dataset"]
        assert callable(kwargs["reward_funcs"])
        return FakeTrainer(kwargs["model"])

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=FakeDataset),
    )
    monkeypatch.setattr(TrlGrpoBackend, "build", fake_build)
    run_grpo_main(
        [
            "experiment=canonical_grpo",
            "task.num_examples=2",
            "trainer.max_steps=1",
            "supervision.num_generations=2",
            "supervision.max_completion_length=4",
            "supervision.gradient_accumulation_steps=1",
            "--output",
            str(tmp_path / "grpo"),
        ]
    )
    assert state == {"trained": True, "rows": 2}
    evidence = json.loads(
        (tmp_path / "grpo" / "grpo_update_evidence.json").read_text(
            encoding="utf-8",
        )
    )
    assert evidence["optimizer_steps"] == 1
    assert evidence["parameter_update_norm"] > 0.0
    manifest = json.loads((tmp_path / "grpo" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_cell"] == "canonical_grpo"
    assert manifest["prereg_sha256"] != "unavailable"
    assert "prereg_git_commit" in manifest
    assert manifest["end_time"]
