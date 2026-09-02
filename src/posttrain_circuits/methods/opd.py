"""The controlled 2x3 StateSource x Supervisor grid and unique OPD arm."""

from __future__ import annotations

from posttrain_circuits.methods.specs import (
    FACTORIAL_TRAINING_BACKEND,
    FACTORIAL_TRAINING_BACKEND_VERSION,
    FACTORIAL_TRAINING_BATCH_CONTRACT,
    REGISTERED_CURRENT_POLICY,
    REGISTERED_SOFT_TEACHER_OBJECTIVE,
    MethodKind,
    MethodSpec,
    RewardSignal,
    RewardUsage,
    StateSourceKind,
    SupervisionKind,
)


def _factorial_method(
    method_id: str,
    *,
    method_kind: MethodKind,
    state_source: StateSourceKind,
    supervision: SupervisionKind,
    objective: str,
    reward_signal: RewardSignal,
    reward_usage: RewardUsage,
) -> MethodSpec:
    return MethodSpec(
        method_id=method_id,
        method_kind=method_kind,
        role="factorial_cell",
        state_source=state_source,
        supervision=supervision,
        objective=objective,
        reward_signal=reward_signal,
        reward_usage=reward_usage,
        training_backend=FACTORIAL_TRAINING_BACKEND,
        training_backend_version=FACTORIAL_TRAINING_BACKEND_VERSION,
        training_batch_contract=FACTORIAL_TRAINING_BATCH_CONTRACT,
        current_policy=(
            REGISTERED_CURRENT_POLICY if state_source == "current_policy" else None
        ),
        soft_teacher_objective=(
            REGISTERED_SOFT_TEACHER_OBJECTIVE if supervision == "soft_teacher" else None
        ),
        requires_common_rollout_bank=state_source == "fixed_bank",
    )


OFFLINE_HARD_METHOD = _factorial_method(
    "offline_hard",
    method_kind="hard_distillation",
    state_source="fixed_bank",
    supervision="hard_teacher",
    objective="teacher_top1_response_token_cross_entropy",
    reward_signal="none",
    reward_usage="none",
)
ONLINE_HARD_METHOD = _factorial_method(
    "online_hard",
    method_kind="hard_distillation",
    state_source="current_policy",
    supervision="hard_teacher",
    objective="teacher_top1_response_token_cross_entropy",
    reward_signal="none",
    reward_usage="none",
)
OFFLINE_SOFT_METHOD = _factorial_method(
    "offline_soft",
    method_kind="soft_distillation",
    state_source="fixed_bank",
    supervision="soft_teacher",
    objective=REGISTERED_SOFT_TEACHER_OBJECTIVE.objective,
    reward_signal="none",
    reward_usage="none",
)

# This is the sole named OPD definition.  Registry and protocol modules import
# this exact object; they never reconstruct it from the method name.
OPD_METHOD = _factorial_method(
    "online_soft_opd",
    method_kind="opd",
    state_source="current_policy",
    supervision="soft_teacher",
    objective=REGISTERED_SOFT_TEACHER_OBJECTIVE.objective,
    reward_signal="none",
    reward_usage="none",
)
OFFLINE_VERIFIED_REPLAY_METHOD = _factorial_method(
    "offline_verified_replay",
    method_kind="verified_replay",
    state_source="fixed_bank",
    supervision="verified_replay",
    objective="sequence_normalized_nll_on_exact_verifier_successes",
    reward_signal="exact",
    reward_usage="selection_gate",
)
ONLINE_VERIFIED_REPLAY_METHOD = _factorial_method(
    "online_verified_replay",
    method_kind="verified_replay",
    state_source="current_policy",
    supervision="verified_replay",
    objective="sequence_normalized_nll_on_exact_verifier_successes",
    reward_signal="exact",
    reward_usage="selection_gate",
)

FACTORIAL_METHOD_SPECS: tuple[MethodSpec, ...] = (
    OFFLINE_HARD_METHOD,
    ONLINE_HARD_METHOD,
    OFFLINE_SOFT_METHOD,
    OPD_METHOD,
    OFFLINE_VERIFIED_REPLAY_METHOD,
    ONLINE_VERIFIED_REPLAY_METHOD,
)

__all__ = [
    "FACTORIAL_METHOD_SPECS",
    "OFFLINE_HARD_METHOD",
    "OFFLINE_SOFT_METHOD",
    "OFFLINE_VERIFIED_REPLAY_METHOD",
    "ONLINE_HARD_METHOD",
    "ONLINE_VERIFIED_REPLAY_METHOD",
    "OPD_METHOD",
]
