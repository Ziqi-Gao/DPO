"""Canonical scientific method definitions.

The objects exported here are definitions, not execution dispatch.  Runtime
code consumes the immutable registry instead of reconstructing method meaning
from experiment names in multiple packages.
"""

from posttrain_circuits.methods.registry import (
    ANCHOR_METHOD_IDS,
    CONTROL_METHOD_IDS,
    FACTORIAL_METHOD_IDS,
    FACTORIAL_METHOD_SPECS,
    METHOD_REGISTRY,
    PILOT_METHOD_IDS,
    get_method_spec,
    registered_method_ids,
)
from posttrain_circuits.methods.specs import CurrentPolicySpec, MethodSpec

__all__ = [
    "ANCHOR_METHOD_IDS",
    "CONTROL_METHOD_IDS",
    "CurrentPolicySpec",
    "FACTORIAL_METHOD_IDS",
    "FACTORIAL_METHOD_SPECS",
    "METHOD_REGISTRY",
    "MethodSpec",
    "PILOT_METHOD_IDS",
    "get_method_spec",
    "registered_method_ids",
]
