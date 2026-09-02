"""Reward controls for the canonical GRPO implementation."""

from posttrain_circuits.methods.rl import GRPO_OBJECTIVE
from posttrain_circuits.methods.specs import (
    REGISTERED_CURRENT_POLICY,
    TRL_GRPO_BACKEND,
    TRL_GRPO_BATCH_CONTRACT,
    TRL_GRPO_VERSION,
    MethodSpec,
)

FORMAT_REWARD_CONTROL = MethodSpec(
    method_id="grpo_format_reward",
    method_kind="reward_control",
    role="control",
    state_source="current_policy",
    supervision="grpo",
    objective=GRPO_OBJECTIVE,
    reward_signal="format_only",
    reward_usage="policy_gradient",
    training_backend=TRL_GRPO_BACKEND,
    training_backend_version=TRL_GRPO_VERSION,
    training_batch_contract=TRL_GRPO_BATCH_CONTRACT,
    current_policy=REGISTERED_CURRENT_POLICY,
)

MATCHED_RANDOM_REWARD_CONTROL = MethodSpec(
    method_id="grpo_random_reward",
    method_kind="reward_control",
    role="control",
    state_source="current_policy",
    supervision="grpo",
    objective=GRPO_OBJECTIVE,
    reward_signal="matched_random",
    reward_usage="policy_gradient",
    training_backend=TRL_GRPO_BACKEND,
    training_backend_version=TRL_GRPO_VERSION,
    training_batch_contract=TRL_GRPO_BATCH_CONTRACT,
    current_policy=REGISTERED_CURRENT_POLICY,
)

CONTROL_METHOD_SPECS: tuple[MethodSpec, ...] = (
    FORMAT_REWARD_CONTROL,
    MATCHED_RANDOM_REWARD_CONTROL,
)

__all__ = [
    "CONTROL_METHOD_SPECS",
    "FORMAT_REWARD_CONTROL",
    "MATCHED_RANDOM_REWARD_CONTROL",
]
