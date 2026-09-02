from __future__ import annotations

from pathlib import Path

import pytest

from posttrain_circuits.core.config import compose_config, validate_model_revision

EXPERIMENTS = [
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
def test_unregistered_training_experiment_is_rejected(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiment"
    experiment_root.mkdir(parents=True)
    (experiment_root / "unknown.yaml").write_text(
        "name: unknown\nstate_source: fixed_bank\nsupervision: hard_teacher\n",
        encoding="utf-8",
    )
    for source in Path("configs").iterdir():
        if source.name == "experiment":
            continue
        if source.is_dir():
            (tmp_path / source.name).symlink_to(source.resolve(), target_is_directory=True)
        elif source.name == "config.yaml":
            (tmp_path / source.name).symlink_to(source.resolve())
    with pytest.raises(ValueError, match="unregistered training experiment"):
        compose_config(["experiment=unknown"], config_root=tmp_path)


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
@pytest.mark.parametrize("override", ["unknown=1", "trainer.unknown=1"])
def test_unknown_override_keys_are_rejected(override: str) -> None:
    with pytest.raises(ValueError, match="unknown override key"):
        compose_config([override], config_root=Path("configs"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "override",
    ["experiment=../offline_hard", "experiment=offline/hard"],
)
def test_configuration_group_traversal_and_invalid_names_are_rejected(
    override: str,
) -> None:
    with pytest.raises(ValueError, match="valid configuration identifier"):
        compose_config([override], config_root=Path("configs"))


@pytest.mark.unit
def test_configuration_root_symlink_is_rejected(tmp_path: Path) -> None:
    config_link = tmp_path / "configs-link"
    config_link.symlink_to(Path("configs").resolve(), target_is_directory=True)
    with pytest.raises(ValueError, match="root must not be a symlink"):
        compose_config([], config_root=config_link)


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
