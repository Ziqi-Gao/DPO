from __future__ import annotations

from pathlib import Path

import pytest

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.cli.build_teacher_demos import SmokeProofTeacher
from posttrain_circuits.learning.teacher.demo_generation import (
    TeacherDemoGenerationConfig,
    generate_teacher_demonstrations,
)
from posttrain_circuits.learning.state_sources.current_policy import CurrentPolicyStateSource
from posttrain_circuits.learning.state_sources.fixed_bank import FixedBankStateSource
from posttrain_circuits.learning.teacher.demo_source import TeacherDemoStateSource
from posttrain_circuits.learning.supervision.hard_teacher import HardTeacherSupervisor
from posttrain_circuits.learning.supervision.soft_teacher import SoftTeacherSupervisor
from posttrain_circuits.learning.supervision.verified_replay import VerifiedReplaySupervisor
from posttrain_circuits.learning.training.canonical_sft import CanonicalSFTSupervisor
from posttrain_circuits.learning.training.factories import build_state_source, build_supervisor
from posttrain_circuits.utils.smoke import (
    build_fixed_bank,
    build_smoke_examples,
    scripted_current_policy_generator,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("experiment", "expected_type"),
    [
        ("offline_hard", HardTeacherSupervisor),
        ("online_hard", HardTeacherSupervisor),
        ("offline_soft", SoftTeacherSupervisor),
        ("online_soft_opd", SoftTeacherSupervisor),
        ("offline_verified_replay", VerifiedReplaySupervisor),
        ("online_verified_replay", VerifiedReplaySupervisor),
        ("canonical_sft", CanonicalSFTSupervisor),
    ],
)
def test_supervisor_factory_uses_resolved_group(experiment: str, expected_type: type[object]) -> None:
    config = compose_config([f"experiment={experiment}"], config_root=Path("configs"))
    assert isinstance(build_supervisor(config["supervision"], pad_token_id=0), expected_type)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("experiment", "expected_type"),
    [
        ("offline_hard", FixedBankStateSource),
        ("online_soft_opd", CurrentPolicyStateSource),
        ("canonical_sft", TeacherDemoStateSource),
    ],
)
def test_state_source_factory_uses_resolved_group(
    experiment: str,
    expected_type: type[object],
    tokenizer,
) -> None:  # type: ignore[no-untyped-def]
    examples = build_smoke_examples(4)
    bank = build_fixed_bank(examples, tokenizer, 5)
    teacher_demos = generate_teacher_demonstrations(
        examples,
        tokenizer,
        SmokeProofTeacher(),
        TeacherDemoGenerationConfig(
            teacher_id="teacher/id",
            teacher_revision="requested",
            resolved_teacher_commit="resolved",
            sampling_request_seed=5,
            temperature=0.0,
            top_p=1.0,
            candidates_per_prompt=1,
        ),
    ).accepted_attempts
    config = compose_config([f"experiment={experiment}"], config_root=Path("configs"))
    source = build_state_source(
        config["state_source"],
        fixed_bank=bank,
        current_generator=scripted_current_policy_generator(examples, tokenizer),
        teacher_demos=teacher_demos,
        seed=5,
    )
    assert isinstance(source, expected_type)
