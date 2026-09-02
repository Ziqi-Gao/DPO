"""Canonical policy-gradient anchor definition."""

from posttrain_circuits.methods.specs import (
    REGISTERED_CURRENT_POLICY,
    TRL_GRPO_BACKEND,
    TRL_GRPO_BATCH_CONTRACT,
    TRL_GRPO_VERSION,
    MethodSpec,
)

GRPO_OBJECTIVE = "group_relative_policy_optimization"

CANONICAL_GRPO_METHOD = MethodSpec(
    method_id="canonical_grpo",
    method_kind="canonical_grpo",
    role="anchor",
    state_source="current_policy",
    supervision="grpo",
    objective=GRPO_OBJECTIVE,
    reward_signal="exact",
    reward_usage="policy_gradient",
    training_backend=TRL_GRPO_BACKEND,
    training_backend_version=TRL_GRPO_VERSION,
    training_batch_contract=TRL_GRPO_BATCH_CONTRACT,
    current_policy=REGISTERED_CURRENT_POLICY,
)

__all__ = [
    "CANONICAL_GRPO_METHOD",
    "GRPO_OBJECTIVE",
    "TRL_GRPO_BACKEND",
    "TRL_GRPO_BATCH_CONTRACT",
    "TRL_GRPO_VERSION",
]
