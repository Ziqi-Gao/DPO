"""Controlled supervision signals."""

from posttrain_circuits.supervision.hard_teacher import HardTeacherSupervisor
from posttrain_circuits.supervision.soft_teacher import SoftTeacherSupervisor
from posttrain_circuits.supervision.verified_replay import VerifiedReplaySupervisor

__all__ = ["HardTeacherSupervisor", "SoftTeacherSupervisor", "VerifiedReplaySupervisor"]
