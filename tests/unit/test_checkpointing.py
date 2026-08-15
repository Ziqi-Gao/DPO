from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from posttrain_circuits.core.seeding import seed_everything
from posttrain_circuits.training.checkpointing import load_checkpoint, save_checkpoint


@pytest.mark.unit
def test_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path, tiny_model) -> None:  # type: ignore[no-untyped-def]
    seed_everything(19)
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler_state={"position": 3},
        global_step=4,
        policy_version=2,
        online_rollout_round=2,
        resolved_config={"seed": 19},
        manifest_hashes={"train": "abc"},
        state_source_state={"kind": "current_policy", "policy_version": 2},
        accelerator_state={"gradient_accumulation_steps": 4},
        git_commit="git-abc",
        dependency_versions={"torch": torch.__version__, "transformers": "4.56.2"},
        resume_ancestry=["parent.pt"],
    )
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    with torch.no_grad():
        for parameter in tiny_model.parameters():
            parameter.add_(1.0)
    random.random()
    np.random.rand()
    torch.rand(())
    payload = load_checkpoint(path, model=tiny_model, optimizer=optimizer, scheduler=scheduler)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == pytest.approx(expected)
    assert payload["global_step"] == 4
    assert payload["policy_version"] == 2
    assert payload["prompt_scheduler"] == {"position": 3}


@pytest.mark.unit
def test_checkpoint_resume_reproduces_next_random_update(tmp_path, tiny_model) -> None:  # type: ignore[no-untyped-def]
    seed_everything(91)
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        model=tiny_model,
        optimizer=optimizer,
        scheduler=scheduler,
        prompt_scheduler_state={"position": 0},
        global_step=0,
        policy_version=0,
        online_rollout_round=0,
        resolved_config={"seed": 91},
        manifest_hashes={"train": "deterministic"},
    )

    def take_update() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 4))
        optimizer.zero_grad(set_to_none=True)
        loss = tiny_model(input_ids=input_ids).logits.float().square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return input_ids, {name: tensor.detach().clone() for name, tensor in tiny_model.state_dict().items()}

    expected_ids, expected_state = take_update()
    load_checkpoint(path, model=tiny_model, optimizer=optimizer, scheduler=scheduler)
    actual_ids, actual_state = take_update()
    assert torch.equal(actual_ids, expected_ids)
    assert actual_state.keys() == expected_state.keys()
    for name in expected_state:
        assert torch.allclose(actual_state[name], expected_state[name], atol=1e-7, rtol=1e-6)
