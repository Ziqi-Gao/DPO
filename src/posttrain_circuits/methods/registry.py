"""Single immutable registry for every active scientific method."""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping

from posttrain_circuits.methods.controls import CONTROL_METHOD_SPECS
from posttrain_circuits.methods.opd import FACTORIAL_METHOD_SPECS, OPD_METHOD
from posttrain_circuits.methods.rl import CANONICAL_GRPO_METHOD
from posttrain_circuits.methods.sft import CANONICAL_SFT_METHOD
from posttrain_circuits.methods.specs import MethodRole, MethodSpec

_METHOD_SPECS = (
    *FACTORIAL_METHOD_SPECS,
    CANONICAL_SFT_METHOD,
    CANONICAL_GRPO_METHOD,
    *CONTROL_METHOD_SPECS,
)

if len(_METHOD_SPECS) != 10:
    raise RuntimeError("the canonical method registry must contain exactly ten methods")
if len({spec.method_id for spec in _METHOD_SPECS}) != len(_METHOD_SPECS):
    raise RuntimeError("canonical method IDs must be unique")
if tuple(spec for spec in _METHOD_SPECS if spec.method_kind == "opd") != (OPD_METHOD,):
    raise RuntimeError("online_soft_opd must be the unique registered online OPD method")
_GRPO_SPECS = (CANONICAL_GRPO_METHOD, *CONTROL_METHOD_SPECS)
if len({spec.objective for spec in _GRPO_SPECS}) != 1:
    raise RuntimeError("canonical and control GRPO methods must share one objective")
if len(
    {
        (
            spec.training_backend,
            spec.training_backend_version,
            spec.training_batch_contract,
            spec.current_policy,
        )
        for spec in _GRPO_SPECS
    }
) != 1:
    raise RuntimeError("canonical and control GRPO methods must share backend/collection semantics")

METHOD_REGISTRY: Mapping[str, MethodSpec] = MappingProxyType(
    {spec.method_id: spec for spec in _METHOD_SPECS}
)
FACTORIAL_METHOD_IDS = tuple(spec.method_id for spec in FACTORIAL_METHOD_SPECS)
ANCHOR_METHOD_IDS = (CANONICAL_SFT_METHOD.method_id, CANONICAL_GRPO_METHOD.method_id)
CONTROL_METHOD_IDS = tuple(spec.method_id for spec in CONTROL_METHOD_SPECS)
PILOT_METHOD_IDS = (*FACTORIAL_METHOD_IDS, *ANCHOR_METHOD_IDS)


def get_method_spec(method_id: str) -> MethodSpec:
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as error:
        raise KeyError(f"unregistered scientific method: {method_id!r}") from error


def registered_method_ids(
    *, role: MethodRole | Literal["all"] = "all"
) -> tuple[str, ...]:
    if role == "all":
        return tuple(METHOD_REGISTRY)
    if role not in {"factorial_cell", "anchor", "control"}:
        raise ValueError(f"unsupported method role: {role!r}")
    return tuple(spec.method_id for spec in _METHOD_SPECS if spec.role == role)


__all__ = [
    "ANCHOR_METHOD_IDS",
    "CONTROL_METHOD_IDS",
    "FACTORIAL_METHOD_IDS",
    "FACTORIAL_METHOD_SPECS",
    "METHOD_REGISTRY",
    "PILOT_METHOD_IDS",
    "get_method_spec",
    "registered_method_ids",
]
