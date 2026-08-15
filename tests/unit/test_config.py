from __future__ import annotations

from pathlib import Path

import pytest

from posttrain_circuits.core.config import compose_config, validate_model_revision

EXPERIMENTS = [
    "phase0_sft",
    "offline_hard",
    "online_hard",
    "offline_soft",
    "online_soft_opd",
    "offline_verified_replay",
    "online_verified_replay",
    "canonical_sft",
    "canonical_grpo",
    "grpo_random_reward",
    "grpo_format_reward",
    "local_fork",
]


@pytest.mark.unit
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_configuration_resolves(experiment: str) -> None:
    config = compose_config([f"experiment={experiment}"], config_root=Path("configs"))
    assert config["experiment"]["name"] == experiment
    for dependency in ("state_source", "supervision"):
        if dependency in config["experiment"]:
            assert config[dependency]["name"] == config["experiment"][dependency]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("experiment", "state_source", "supervision"),
    [
        ("offline_hard", "fixed_bank", "hard_teacher"),
        ("online_hard", "current_policy", "hard_teacher"),
        ("offline_soft", "fixed_bank", "soft_teacher"),
        ("online_soft_opd", "current_policy", "soft_teacher"),
        ("offline_verified_replay", "fixed_bank", "verified_replay"),
        ("online_verified_replay", "current_policy", "verified_replay"),
        ("canonical_sft", "teacher_demo", "canonical_sft"),
        ("canonical_grpo", "current_policy", "grpo"),
    ],
)
def test_experiment_resolves_dependencies(experiment: str, state_source: str, supervision: str) -> None:
    config = compose_config([f"experiment={experiment}"], config_root=Path("configs"))
    assert config["state_source"]["name"] == state_source
    assert config["supervision"]["name"] == supervision


@pytest.mark.unit
def test_conflicting_explicit_dependency_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires state_source=fixed_bank"):
        compose_config(
            ["experiment=offline_hard", "state_source=current_policy"],
            config_root=Path("configs"),
        )


@pytest.mark.unit
def test_unpinned_official_revision_is_rejected() -> None:
    model = {
        "model_name_or_path": "organization/model",
        "model_revision": "main",
        "tokenizer_name_or_path": "organization/model",
        "tokenizer_revision": "main",
        "torch_dtype": "bfloat16",
        "attn_implementation": "eager",
        "gradient_checkpointing": True,
        "use_cache": False,
        "trust_remote_code": False,
    }
    with pytest.raises(ValueError, match="unpinned"):
        validate_model_revision(model)
    model["allow_unpinned_revision"] = True
    validate_model_revision(model)
