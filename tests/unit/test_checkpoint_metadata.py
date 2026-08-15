from __future__ import annotations

import pytest
import torch

from posttrain_circuits.training.checkpointing import load_checkpoint, save_checkpoint


@pytest.mark.unit
def test_checkpoint_contains_full_runtime_and_provenance_state(tmp_path, tiny_model) -> None:  # type: ignore[no-untyped-def]
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "metadata.pt"
    save_checkpoint(
        path,
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler_state={"position": 3},
        state_source_state={"kind": "current_policy", "policy_version": 2},
        trainer_state={"cumulative_counts": {"prompts_consumed": 12.0}},
        global_step=4,
        policy_version=2,
        online_rollout_round=2,
        resolved_config={"seed": 19},
        manifest_hashes={"dataset": "data", "rollout_bank": "bank", "prompt_schedule": "schedule"},
        accelerator_state={"gradient_accumulation_steps": 4},
        scaler_state={"scale": 1024.0},
        git_commit="git-abc",
        dependency_versions={"torch": torch.__version__, "transformers": "4.56.2"},
        resume_ancestry=["parent.pt"],
    )

    payload = load_checkpoint(path, model=tiny_model, optimizer=optimizer, scheduler=scheduler)
    assert payload["state_source"]["policy_version"] == 2
    assert payload["trainer_state"]["cumulative_counts"]["prompts_consumed"] == 12.0
    assert payload["accelerator"]["gradient_accumulation_steps"] == 4
    assert payload["scaler"]["scale"] == 1024.0
    assert payload["git_commit"] == "git-abc"
    assert payload["dependency_versions"]["transformers"] == "4.56.2"
    assert payload["resume_ancestry"] == ["parent.pt"]
    assert set(payload["manifest_hashes"]) == {"dataset", "rollout_bank", "prompt_schedule"}
