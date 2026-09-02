from __future__ import annotations

import base64
import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from posttrain_circuits.artifacts.checkpoints import torch_state_hash
from posttrain_circuits.artifacts.config_bindings import bind_config
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.experiments.protocols.local_fork import (
    LOCAL_FORK_SPEC,
    validate_local_fork_formal_binding,
    validate_local_fork_source_binding,
)
from posttrain_circuits.experiments.protocols.specs import (
    CONTROLLED_FACTORIAL,
    ExperimentBinding,
    build_experiment_binding,
    validate_checkpoint_experiment_binding_payload,
)
from posttrain_circuits.methods.controls import (
    FORMAT_REWARD_CONTROL,
    MATCHED_RANDOM_REWARD_CONTROL,
)
from posttrain_circuits.methods.opd import OPD_METHOD
from posttrain_circuits.methods.registry import (
    ANCHOR_METHOD_IDS,
    CONTROL_METHOD_IDS,
    FACTORIAL_METHOD_IDS,
    FACTORIAL_METHOD_SPECS,
    METHOD_REGISTRY,
    PILOT_METHOD_IDS,
    get_method_spec,
)
from posttrain_circuits.methods.rl import (
    CANONICAL_GRPO_METHOD,
    TRL_GRPO_BACKEND,
    TRL_GRPO_BATCH_CONTRACT,
    TRL_GRPO_VERSION,
)
from posttrain_circuits.methods.sft import CANONICAL_SFT_METHOD


@pytest.mark.unit
def test_registry_is_immutable_complete_and_hash_addressed() -> None:
    expected = (
        "offline_hard",
        "online_hard",
        "offline_soft",
        "online_soft_opd",
        "offline_verified_replay",
        "online_verified_replay",
        "canonical_sft",
        "canonical_grpo",
        "grpo_format_reward",
        "grpo_random_reward",
    )
    assert tuple(METHOD_REGISTRY) == expected
    assert len(METHOD_REGISTRY) == 10
    assert len({spec.scientific_sha256 for spec in METHOD_REGISTRY.values()}) == 10
    assert PILOT_METHOD_IDS == (*FACTORIAL_METHOD_IDS, *ANCHOR_METHOD_IDS)
    assert CONTROL_METHOD_IDS == ("grpo_format_reward", "grpo_random_reward")
    with pytest.raises(TypeError):
        METHOD_REGISTRY["new_method"] = OPD_METHOD  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        OPD_METHOD.objective = "changed"  # type: ignore[misc]


@pytest.mark.unit
def test_factorial_registry_is_exactly_two_by_three_with_one_named_opd() -> None:
    observed = {(spec.state_source, spec.supervision) for spec in FACTORIAL_METHOD_SPECS}
    expected = {
        (state_source, supervision)
        for state_source in ("fixed_bank", "current_policy")
        for supervision in ("hard_teacher", "soft_teacher", "verified_replay")
    }
    assert observed == expected
    assert CONTROLLED_FACTORIAL.cell_ids == FACTORIAL_METHOD_IDS
    assert [
        spec for spec in METHOD_REGISTRY.values() if spec.method_kind == "opd" and spec.is_online
    ] == [OPD_METHOD]
    assert METHOD_REGISTRY["online_soft_opd"] is OPD_METHOD
    assert OPD_METHOD.supervision == "soft_teacher"
    assert OPD_METHOD.reward_signal == "none"
    assert OPD_METHOD.reward_usage == "none"
    assert OPD_METHOD.uses_policy_gradient is False

    offline_soft = get_method_spec("offline_soft")
    assert offline_soft.soft_teacher_objective is OPD_METHOD.soft_teacher_objective
    assert offline_soft.objective == OPD_METHOD.objective
    assert offline_soft.soft_teacher_objective is not None
    assert offline_soft.soft_teacher_objective.topk_mode == "renormalized"
    assert offline_soft.soft_teacher_objective.teacher_top_k == 128
    assert offline_soft.soft_teacher_objective.include_eos is True
    assert offline_soft.soft_teacher_objective.fail_below_minimum_retained_mass is False


@pytest.mark.unit
def test_online_methods_freeze_refresh_lag_and_optimizer_boundary() -> None:
    online = [spec for spec in METHOD_REGISTRY.values() if spec.is_online]
    assert online
    for spec in online:
        assert spec.current_policy is not None
        assert spec.current_policy.refresh_interval == 1
        assert spec.current_policy.max_policy_lag == 0
        assert spec.current_policy.refresh_boundary == "optimizer_update"
        assert spec.current_policy.retry_forces_refresh is False


@pytest.mark.unit
def test_reward_gate_and_policy_gradient_are_distinct_method_semantics() -> None:
    replay = get_method_spec("online_verified_replay")
    assert replay.reward_signal == "exact"
    assert replay.reward_usage == "selection_gate"
    assert replay.uses_policy_gradient is False
    assert CANONICAL_SFT_METHOD.reward_usage == "selection_gate"
    assert CANONICAL_SFT_METHOD.requires_teacher_demo_store is True

    assert CANONICAL_GRPO_METHOD.role == "anchor"
    assert CANONICAL_GRPO_METHOD.reward_signal == "exact"
    assert CANONICAL_GRPO_METHOD.reward_usage == "policy_gradient"
    assert CANONICAL_GRPO_METHOD.training_backend == TRL_GRPO_BACKEND
    assert TRL_GRPO_VERSION == "0.22.2"
    assert TRL_GRPO_BATCH_CONTRACT == "0.22.2-default-steps_per_generation"
    assert FORMAT_REWARD_CONTROL.reward_signal == "format_only"
    assert MATCHED_RANDOM_REWARD_CONTROL.reward_signal == "matched_random"
    assert all(spec.uses_policy_gradient for spec in (FORMAT_REWARD_CONTROL, MATCHED_RANDOM_REWARD_CONTROL))
    canonical_payload = CANONICAL_GRPO_METHOD.to_payload()
    allowed_control_fields = {"method_id", "method_kind", "role", "reward_signal"}
    for control in (FORMAT_REWARD_CONTROL, MATCHED_RANDOM_REWARD_CONTROL):
        control_payload = control.to_payload()
        changed = {
            key
            for key in canonical_payload
            if canonical_payload[key] != control_payload[key]
        }
        assert changed == allowed_control_fields


@pytest.mark.unit
def test_every_registered_method_validates_its_composed_config() -> None:
    for method in METHOD_REGISTRY.values():
        config = compose_config([f"experiment={method.method_id}"])
        method.validate_resolved_config(config)

    control = compose_config(["experiment=grpo_format_reward"])
    control["experiment"]["use_teacher_loss"] = True
    with pytest.raises(ValueError, match="auxiliary-loss contract"):
        FORMAT_REWARD_CONTROL.validate_resolved_config(control)

    grpo = compose_config(["experiment=canonical_grpo"])
    grpo["supervision"]["temperature"] = 0.5
    with pytest.raises(ValueError, match="GRPO sampling differs"):
        CANONICAL_GRPO_METHOD.validate_resolved_config(grpo)


def _binding(method_id: str, *, seed: int = 42, bank: str = "a" * 64) -> ExperimentBinding:
    method = get_method_spec(method_id)
    soft = method.soft_teacher_objective
    requires_bank = method.requires_common_rollout_bank
    requires_demo = method.requires_teacher_demo_store
    requires_calibration = method.reward_signal == "matched_random"
    return ExperimentBinding(
        experiment_id=(
            f"qwen3_v2:{method_id}:seed-{seed}:{CONTROLLED_FACTORIAL.design_id}"
        ),
        method_id=method_id,
        method_spec_sha256=method.scientific_sha256,
        factorial_design_sha256=CONTROLLED_FACTORIAL.scientific_sha256,
        scientific_config_sha256="0" * 64,
        seed=seed,
        protocol_track="qwen3_v2",
        preregistration_sha256="0" * 64,
        implementation_commit="f" * 40,
        implementation_dirty=False,
        model_id="Qwen/Qwen3-1.7B",
        model_requested_revision="model-revision",
        model_resolved_revision="1" * 40,
        teacher_id="Qwen/Qwen3-8B",
        teacher_requested_revision="teacher-revision",
        teacher_resolved_revision="2" * 40,
        tokenizer_id="Qwen/Qwen3-1.7B",
        tokenizer_requested_revision="tokenizer-revision",
        tokenizer_resolved_revision="3" * 40,
        tokenizer_fingerprint="1" * 64,
        prompt_protocol="qwen3_non_thinking_v1",
        chat_template_sha256="2" * 64,
        enable_thinking=False,
        model_facing_prompt_schedule_sha256="3" * 64,
        probe_manifest_sha256="4" * 64,
        task_protocol_sha256="5" * 64,
        state_source_protocol_sha256="a" * 64,
        supervision_protocol_sha256="b" * 64,
        response_mask_protocol=(
            "trl_completion_mask_v1"
            if method.supervision == "grpo"
            else "response_tokens_only_including_eos_v1"
        ),
        max_completion_length=256,
        full_parameter_training=True,
        initial_checkpoint_sha256="1" * 64,
        train_dataset_sha256="2" * 64,
        validation_dataset_sha256="3" * 64,
        optimizer_spec_sha256="5" * 64,
        scheduler_spec_sha256="6" * 64,
        backend_id=method.training_backend,
        backend_version=method.training_backend_version,
        backend_batch_contract=method.training_batch_contract,
        resolved_batch_contract_sha256="7" * 64,
        token_budget=2_000_000,
        teacher_soft_objective=soft.objective if soft else None,
        teacher_topk_mode=soft.topk_mode if soft else None,
        teacher_top_k=soft.teacher_top_k if soft else None,
        teacher_topk_include_eos=soft.include_eos if soft else None,
        teacher_topk_minimum_retained_mass=(soft.minimum_retained_mass if soft else None),
        teacher_topk_fail_below_minimum_retained_mass=(
            soft.fail_below_minimum_retained_mass if soft else None
        ),
        offline_bank_manifest_sha256=bank if requires_bank else None,
        offline_bank_content_sha256="8" * 64 if requires_bank else None,
        offline_bank_initial_cursor_sha256="9" * 64 if requires_bank else None,
        teacher_demo_manifest_sha256="a" * 64 if requires_demo else None,
        teacher_demo_content_sha256="b" * 64 if requires_demo else None,
        teacher_demo_exact_verifier_gate=True if requires_demo else None,
        teacher_demo_generation_sha256="c" * 64 if requires_demo else None,
        reward_calibration_manifest_sha256="d" * 64 if requires_calibration else None,
        reward_calibration_content_sha256="e" * 64 if requires_calibration else None,
    )


def _complete_factorial_checkpoint_payload(binding: ExperimentBinding) -> dict[str, object]:
    model = {"weight": torch.tensor([1.0, 2.0])}
    token_budget = {
        "budget": binding.token_budget,
        "unit": "global_nonpadding_model_input_tokens_processed",
        "consumed": 4,
        "accepted_optimizer_updates": 1,
        "stop_reason": "max_steps_safety_limit",
    }
    prompt_state = {"position": 1}
    source_state = {"cursor": 1}
    return {
        "model": model,
        "optimizer": {
            "state": {0: {"exp_avg": torch.tensor([0.1, 0.2])}},
            "param_groups": [{"params": [0]}],
        },
        "scheduler": {"last_epoch": 1, "_step_count": 2},
        "rng": {
            "python": base64.b64encode(b"python").decode("ascii"),
            "numpy": base64.b64encode(b"numpy").decode("ascii"),
            "torch_cpu": [1, 2, 3],
            "torch_cuda": [],
        },
        "global_step": 1,
        "world_size": 1,
        "trainer_state": {
            "cumulative_counts": {"model_facing_input_tokens_processed": 4.0},
            "accumulation_micro_step": 0,
            "token_budget": token_budget,
        },
        "token_budget": token_budget,
        "prompt_scheduler": prompt_state,
        "prompt_scheduler_by_rank": [prompt_state],
        "state_source": source_state,
        "state_source_by_rank": [source_state],
        "rank_shard_hashes": ["9" * 64],
        "resume_ancestry": [],
        "parameter_update_norm": 0.5,
        "final_model_state_hash": torch_state_hash(model),
        "update_norm_baseline_checkpoint_path": "/data/del6500/OPD/initial.pt",
        "update_norm_baseline_checkpoint_sha256": binding.initial_checkpoint_sha256,
        "manifest_hashes": {
            "experiment_binding": binding.scientific_sha256,
            "factorial_design": binding.factorial_design_sha256,
            "method_spec": binding.method_spec_sha256,
            "dataset_train": binding.train_dataset_sha256,
            "dataset_validation": binding.validation_dataset_sha256,
            "initial_checkpoint": binding.initial_checkpoint_sha256,
            "model_facing_prompt_schedule": binding.model_facing_prompt_schedule_sha256,
            "probe_manifest": binding.probe_manifest_sha256,
        },
        "git_commit": binding.implementation_commit,
        "implementation_dirty": binding.implementation_dirty,
    }


@pytest.mark.unit
def test_factorial_checkpoint_binding_requires_every_runtime_state_class() -> None:
    binding = _binding("online_hard")
    complete = _complete_factorial_checkpoint_payload(binding)
    validate_checkpoint_experiment_binding_payload(complete, binding)
    for missing in (
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "global_step",
        "trainer_state",
        "token_budget",
        "prompt_scheduler_by_rank",
        "state_source_by_rank",
        "manifest_hashes",
    ):
        broken = copy.deepcopy(complete)
        broken.pop(missing)
        with pytest.raises(ValueError):
            validate_checkpoint_experiment_binding_payload(broken, binding)


@pytest.mark.unit
def test_factorial_checkpoint_binding_rejects_zero_update_and_model_tamper() -> None:
    binding = _binding("online_hard")
    no_update = _complete_factorial_checkpoint_payload(binding)
    no_update["parameter_update_norm"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        validate_checkpoint_experiment_binding_payload(no_update, binding)

    tampered = _complete_factorial_checkpoint_payload(binding)
    assert isinstance(tampered["model"], dict)
    tampered["model"]["weight"] = torch.tensor([7.0, 8.0])
    with pytest.raises(ValueError, match="model-state hash"):
        validate_checkpoint_experiment_binding_payload(tampered, binding)


@pytest.mark.unit
def test_factorial_bindings_require_full_comparable_grid_and_one_common_bank() -> None:
    bindings = tuple(_binding(method_id) for method_id in FACTORIAL_METHOD_IDS)
    CONTROLLED_FACTORIAL.validate_bindings(bindings)
    assert all(
        binding.method_spec_sha256 == get_method_spec(binding.method_id).scientific_sha256
        for binding in bindings
    )

    with pytest.raises(ValueError, match="full factorial grid"):
        CONTROLLED_FACTORIAL.validate_bindings(bindings[:-1])
    incompatible = (
        *bindings[:-1],
        replace(bindings[-1], model_facing_prompt_schedule_sha256="7" * 64),
    )
    with pytest.raises(ValueError, match="controlled comparison dimension"):
        CONTROLLED_FACTORIAL.validate_bindings(incompatible)
    bank_changed = tuple(
        replace(row, offline_bank_manifest_sha256="b" * 64)
        if row.method_id == "offline_soft"
        else row
        for row in bindings
    )
    with pytest.raises(ValueError, match="one frozen bank"):
        CONTROLLED_FACTORIAL.validate_bindings(bank_changed)
    state_protocol_changed = tuple(
        replace(row, state_source_protocol_sha256="c" * 64)
        if row.method_id == "offline_soft"
        else row
        for row in bindings
    )
    with pytest.raises(ValueError, match="state_source protocol changed within a level"):
        CONTROLLED_FACTORIAL.validate_bindings(state_protocol_changed)


@pytest.mark.unit
def test_method_specific_provenance_inputs_fail_closed() -> None:
    sft = _binding("canonical_sft")
    with pytest.raises(ValueError, match="teacher-demo binding"):
        replace(
            sft,
            teacher_demo_manifest_sha256=None,
            teacher_demo_content_sha256=None,
            teacher_demo_generation_sha256=None,
            teacher_demo_exact_verifier_gate=None,
        )
    assert sft.teacher_demo_manifest_sha256 == "a" * 64

    random_control = _binding("grpo_random_reward")
    with pytest.raises(ValueError, match="matched-random calibration"):
        replace(
            random_control,
            reward_calibration_manifest_sha256=None,
            reward_calibration_content_sha256=None,
        )
    assert random_control.reward_calibration_manifest_sha256 == "d" * 64

    malformed = _binding("online_hard").to_payload()
    malformed["seed"] = True
    with pytest.raises(ValueError, match="seed must be an integer"):
        ExperimentBinding.from_payload(malformed)
    missing = _binding("online_hard").to_payload()
    missing.pop("probe_manifest_sha256")
    with pytest.raises(ValueError, match="fields differ from schema"):
        ExperimentBinding.from_payload(missing)
    old_schema = _binding("online_hard").to_payload()
    old_schema["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported ExperimentBinding schema"):
        ExperimentBinding.from_payload(old_schema)


@pytest.mark.unit
def test_canonical_grpo_anchor_requires_the_same_scientific_nuisance_state() -> None:
    replay = _binding("online_verified_replay")
    grpo = _binding("canonical_grpo")
    CONTROLLED_FACTORIAL.validate_canonical_grpo_anchor_pair(replay, grpo)
    with pytest.raises(ValueError, match="shared nuisance variable"):
        CONTROLLED_FACTORIAL.validate_canonical_grpo_anchor_pair(
            replay,
            replace(grpo, probe_manifest_sha256="f" * 64),
        )
    with pytest.raises(ValueError, match="shared nuisance variable"):
        CONTROLLED_FACTORIAL.validate_canonical_grpo_anchor_pair(
            replay,
            replace(grpo, max_completion_length=grpo.max_completion_length + 1),
        )


@pytest.mark.unit
def test_experiment_binding_identity_is_deterministic_across_execution_times() -> None:
    config = compose_config(["experiment=online_hard"])
    config_binding = bind_config(
        config,
        input_artifact_hashes={
            "prereg_path": sha256_file(Path(str(config["prereg_path"])))
        },
        execution_context={"entrypoint": "unit-test"},
    )
    arguments = {
        "config": config,
        "config_binding": config_binding,
        "implementation_commit": "f" * 40,
        "implementation_dirty": False,
        "model_resolved_revision": "1" * 40,
        "tokenizer_resolved_revision": "2" * 40,
        "tokenizer_fingerprint": "3" * 64,
        "teacher_resolved_revision": "4" * 40,
        "initial_checkpoint_sha256": "5" * 64,
        "train_dataset_sha256": "6" * 64,
        "validation_dataset_sha256": "7" * 64,
        "model_facing_prompt_schedule_sha256": "8" * 64,
        "probe_manifest_sha256": "9" * 64,
        "task_protocol": {"generator": "proofgraph-v3", "depth": 2},
        "optimizer_spec": {"class": "AdamW", "learning_rate": 0.0005},
        "scheduler_spec": {"class": "LambdaLR", "schedule": "constant"},
        "resolved_batch_contract": {
            "world_size": 1,
            "per_rank_batch_size": 4,
            "gradient_accumulation_steps": 1,
        },
        "full_parameter_training": True,
    }
    first = build_experiment_binding(**arguments)
    second = build_experiment_binding(**arguments)
    assert first.experiment_id == (
        "core_v2:online_hard:seed-42:state_source_x_supervision_2x3_v1"
    )
    assert first == second
    assert first.scientific_sha256 == second.scientific_sha256
    assert replace(first, implementation_commit="e" * 40).scientific_sha256 != (
        first.scientific_sha256
    )
    assert replace(first, implementation_dirty=True).scientific_sha256 != (
        first.scientific_sha256
    )
    for invalid_template_hash in ("", "A" * 64, "a" * 63):
        invalid_config = copy.deepcopy(config)
        invalid_config["model"]["prompt_protocol"]["chat_template_sha256"] = (
            invalid_template_hash
        )
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            build_experiment_binding(
                **{
                    **arguments,
                    "config": invalid_config,
                    "config_binding": bind_config(
                        invalid_config,
                        input_artifact_hashes=dict(config_binding.input_artifact_hashes),
                        execution_context={"entrypoint": "unit-test"},
                    ),
                }
            )
    missing_template_config = copy.deepcopy(config)
    missing_template_config["model"]["prompt_protocol"].pop("chat_template_sha256")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_experiment_binding(
            **{
                **arguments,
                "config": missing_template_config,
                "config_binding": bind_config(
                    missing_template_config,
                    input_artifact_hashes=dict(config_binding.input_artifact_hashes),
                    execution_context={"entrypoint": "unit-test"},
                ),
            }
        )
    changed_steps_config = copy.deepcopy(config)
    changed_steps_config["trainer"]["max_steps"] += 1
    changed_steps_binding = bind_config(
        changed_steps_config,
        input_artifact_hashes=dict(config_binding.input_artifact_hashes),
        execution_context={"entrypoint": "unit-test"},
    )
    changed_steps_experiment = build_experiment_binding(
        **{
            **arguments,
            "config": changed_steps_config,
            "config_binding": changed_steps_binding,
        }
    )
    assert changed_steps_binding.scientific_config_sha256 != (
        config_binding.scientific_config_sha256
    )
    assert changed_steps_experiment.scientific_sha256 != first.scientific_sha256
    with pytest.raises(ValueError, match="storage locators"):
        build_experiment_binding(
            **{
                **arguments,
                "task_protocol": {
                    "generator": "proofgraph-v3",
                    "nested": {"output_path": "/data/should-not-be-scientific"},
                },
            }
        )


@pytest.mark.unit
def test_local_fork_protocol_freezes_shared_state_branch_semantics() -> None:
    assert LOCAL_FORK_SPEC.branch_ids == (
        "hard_teacher",
        "soft_teacher",
        "verified_replay",
        "centered_policy_gradient",
    )
    assert LOCAL_FORK_SPEC.horizons == (1, 5, 20)
    assert LOCAL_FORK_SPEC.primary_horizon == 20
    assert LOCAL_FORK_SPEC.minimum_group_size == 4
    assert LOCAL_FORK_SPEC.divergence == "KL(output_new || output_fork)"
    assert LOCAL_FORK_SPEC.primary_comparison_axis == "matched_output_kl_new_to_fork"
    branches = {branch.branch_id: branch for branch in LOCAL_FORK_SPEC.branches}
    assert branches["verified_replay"].reward_usage == "selection_gate"
    assert branches["verified_replay"].uses_policy_gradient is False
    assert branches["centered_policy_gradient"].reward_usage == "policy_gradient"
    assert branches["centered_policy_gradient"].uses_policy_gradient is True
    config = compose_config(["experiment=local_fork"])
    LOCAL_FORK_SPEC.validate_experiment_config(config["experiment"])


@pytest.mark.unit
def test_local_fork_source_binding_rejects_method_seed_and_dataset_relabeling() -> None:
    binding = _binding("canonical_sft")
    optimizer_parameter_names = [["model.weight", "model.bias"]]
    source = {
        "schema_version": 1,
        "manifest_path": "/data/del6500/OPD/runs/source/manifest.json",
        "manifest_file_sha256": "1" * 64,
        "manifest_payload_sha256": "2" * 64,
        "experiment_binding": binding.to_payload(),
        "experiment_binding_sha256": binding.scientific_sha256,
        "factorial_design_sha256": binding.factorial_design_sha256,
        "method_id": binding.method_id,
        "seed": binding.seed,
        "train_dataset_sha256": binding.train_dataset_sha256,
        "validation_dataset_sha256": binding.validation_dataset_sha256,
        "initial_checkpoint_sha256": binding.initial_checkpoint_sha256,
        "final_checkpoint_path": "/data/del6500/OPD/runs/source/checkpoints/final.pt",
        "final_checkpoint_sha256": "3" * 64,
        "checkpoint_state_hashes": {
            "model": "b" * 64,
            "model_parameter_names": "5" * 64,
            "trainable_parameter_names": "6" * 64,
            "optimizer": "7" * 64,
            "optimizer_moments": "8" * 64,
            "optimizer_parameter_references": "9" * 64,
            "optimizer_named": "e" * 64,
            "scheduler": "f" * 64,
            "rng": "0" * 64,
        },
        "effective_bundle_state_hashes": {
            "model": "b" * 64,
            "model_parameter_names": "5" * 64,
            "trainable_parameter_names": "6" * 64,
            "optimizer": "c" * 64,
            "optimizer_moments": "d" * 64,
            "optimizer_parameter_names": sha256_value(optimizer_parameter_names),
            "optimizer_named": "e" * 64,
            "scheduler": "f" * 64,
            "rng": "0" * 64,
        },
        "optimizer_parameter_names": optimizer_parameter_names,
        "trajectory_bank_sha256": "a" * 64,
    }
    assert validate_local_fork_source_binding(source) == source
    for field, replacement in (
        ("method_id", "offline_hard"),
        ("seed", 43),
        ("train_dataset_sha256", "f" * 64),
        ("validation_dataset_sha256", "e" * 64),
    ):
        relabeled = copy.deepcopy(source)
        relabeled[field] = replacement
        with pytest.raises(ValueError, match="ExperimentBinding was relabeled"):
            validate_local_fork_source_binding(relabeled)

    remapped = copy.deepcopy(source)
    remapped["optimizer_parameter_names"][0].reverse()
    with pytest.raises(ValueError, match="mapping hash changed"):
        validate_local_fork_source_binding(remapped)

    relabeled_trainable_mapping = copy.deepcopy(source)
    relabeled_trainable_mapping["checkpoint_state_hashes"][
        "trainable_parameter_names"
    ] = "a" * 64
    with pytest.raises(ValueError, match="trainable_parameter_names"):
        validate_local_fork_source_binding(relabeled_trainable_mapping)

    formal = {
        "protocol_track": "qwen3_v2",
        "artifact_namespace": "qwen3-v2",
        "model_revision": "revision",
        "teacher_revision": "teacher-revision",
        "tokenizer_revision": "tokenizer-revision",
        "tokenizer_fingerprint": "1" * 64,
        "chat_template_sha256": "2" * 64,
        "prompt_protocol": "qwen3_non_thinking_v1",
        "enable_thinking": False,
        "code_commit": "commit",
        "prereg_path": "/home/del6500/projects/OPD/prereg/qwen3_v2.yaml",
        "prereg_version": "qwen3_v2",
        "prereg_commit": "prereg-commit",
        "prereg_sha256": "3" * 64,
        "local_fork_source": source,
    }
    assert validate_local_fork_formal_binding(formal) == formal
    extended_formal = {**formal, "unexpected": "not-allowed"}
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        validate_local_fork_formal_binding(extended_formal)
