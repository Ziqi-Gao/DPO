"""Canonical SFT anchor definition."""

from posttrain_circuits.methods.specs import (
    FACTORIAL_TRAINING_BACKEND,
    FACTORIAL_TRAINING_BACKEND_VERSION,
    FACTORIAL_TRAINING_BATCH_CONTRACT,
    MethodSpec,
)

CANONICAL_SFT_METHOD = MethodSpec(
    method_id="canonical_sft",
    method_kind="canonical_sft",
    role="anchor",
    state_source="teacher_demo",
    supervision="canonical_sft",
    objective="sequence_normalized_teacher_demonstration_nll",
    reward_signal="exact",
    reward_usage="selection_gate",
    training_backend=FACTORIAL_TRAINING_BACKEND,
    training_backend_version=FACTORIAL_TRAINING_BACKEND_VERSION,
    training_batch_contract=FACTORIAL_TRAINING_BATCH_CONTRACT,
    requires_teacher_demo_store=True,
    requires_teacher_generation_provenance=True,
)

__all__ = ["CANONICAL_SFT_METHOD"]
