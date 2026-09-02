"""Immutable scientific contracts for OPD training methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from posttrain_circuits.artifacts.hashing import sha256_value

MethodRole = Literal["factorial_cell", "anchor", "control"]
MethodKind = Literal[
    "hard_distillation",
    "soft_distillation",
    "opd",
    "verified_replay",
    "canonical_sft",
    "canonical_grpo",
    "reward_control",
]
StateSourceKind = Literal["fixed_bank", "current_policy", "teacher_demo"]
SupervisionKind = Literal[
    "hard_teacher",
    "soft_teacher",
    "verified_replay",
    "canonical_sft",
    "grpo",
]
RewardSignal = Literal["none", "exact", "format_only", "matched_random"]
RewardUsage = Literal["none", "selection_gate", "policy_gradient"]
TrainingBackend = Literal["factorial_trainer", "trl.GRPOTrainer"]

FACTORIAL_TRAINING_BACKEND = "factorial_trainer"
FACTORIAL_TRAINING_BACKEND_VERSION = "in-repository-v1"
FACTORIAL_TRAINING_BATCH_CONTRACT = "optimizer-boundary-gradient-accumulation-v1"
TRL_GRPO_BACKEND = "trl.GRPOTrainer"
TRL_GRPO_VERSION = "0.22.2"
TRL_GRPO_BATCH_CONTRACT = "0.22.2-default-steps_per_generation"

_METHOD_ROLES = frozenset({"factorial_cell", "anchor", "control"})
_METHOD_KINDS = frozenset(
    {
        "hard_distillation",
        "soft_distillation",
        "opd",
        "verified_replay",
        "canonical_sft",
        "canonical_grpo",
        "reward_control",
    }
)
_STATE_SOURCES = frozenset({"fixed_bank", "current_policy", "teacher_demo"})
_SUPERVISION_KINDS = frozenset(
    {"hard_teacher", "soft_teacher", "verified_replay", "canonical_sft", "grpo"}
)
_REWARD_SIGNALS = frozenset({"none", "exact", "format_only", "matched_random"})
_REWARD_USAGES = frozenset({"none", "selection_gate", "policy_gradient"})
_TRAINING_BACKENDS = frozenset({"factorial_trainer", "trl.GRPOTrainer"})


def _require_nonempty(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CurrentPolicySpec:
    """Collection semantics shared by every registered online method.

    ``optimizer_update`` means retries and additional collection microbatches
    inside one update reuse the same policy version.  The version advances only
    after the prior optimizer update has completed.
    """

    refresh_interval: int = 1
    max_policy_lag: int = 0
    refresh_boundary: Literal["optimizer_update"] = "optimizer_update"
    retry_forces_refresh: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.refresh_interval, bool) or self.refresh_interval < 1:
            raise ValueError("current-policy refresh_interval must be positive")
        if isinstance(self.max_policy_lag, bool) or self.max_policy_lag < 0:
            raise ValueError("current-policy max_policy_lag must be non-negative")
        if self.refresh_boundary != "optimizer_update":
            raise ValueError("registered current-policy refresh boundary must be optimizer_update")
        if self.retry_forces_refresh is not False:
            raise ValueError("a sampling retry must not force a policy refresh")


REGISTERED_CURRENT_POLICY = CurrentPolicySpec()


@dataclass(frozen=True, slots=True)
class SoftTeacherObjectiveSpec:
    """The single top-k target semantics used by both soft factorial arms.

    Retained mass is recorded for the separate readiness/coverage gate; it is
    not a per-token rejection rule inside the training objective.
    """

    objective: str = "teacher_topk_forward_kl_on_response_prefixes"
    topk_mode: Literal["renormalized"] = "renormalized"
    teacher_top_k: int = 128
    include_eos: bool = True
    minimum_retained_mass: float = 0.90
    fail_below_minimum_retained_mass: bool = False

    def __post_init__(self) -> None:
        _require_nonempty(self.objective, name="soft teacher objective")
        if self.topk_mode != "renormalized":
            raise ValueError("registered soft targets must renormalize the retained top-k mass")
        if isinstance(self.teacher_top_k, bool) or self.teacher_top_k != 128:
            raise ValueError("registered soft targets require teacher_top_k=128")
        if self.include_eos is not True:
            raise ValueError("registered soft targets must include EOS")
        if self.minimum_retained_mass != 0.90:
            raise ValueError("registered soft targets require minimum_retained_mass=0.90")
        if self.fail_below_minimum_retained_mass is not False:
            raise ValueError("retained-mass readiness evidence must not filter training tokens")


REGISTERED_SOFT_TEACHER_OBJECTIVE = SoftTeacherObjectiveSpec()


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One immutable, hash-addressed scientific method definition."""

    method_id: str
    method_kind: MethodKind
    role: MethodRole
    state_source: StateSourceKind
    supervision: SupervisionKind
    objective: str
    reward_signal: RewardSignal
    reward_usage: RewardUsage
    training_backend: TrainingBackend
    training_backend_version: str
    training_batch_contract: str
    current_policy: CurrentPolicySpec | None = None
    soft_teacher_objective: SoftTeacherObjectiveSpec | None = None
    requires_common_rollout_bank: bool = False
    requires_teacher_demo_store: bool = False
    requires_teacher_generation_provenance: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_nonempty(self.method_id, name="method_id")
        _require_nonempty(self.objective, name="objective")
        _require_nonempty(self.training_backend_version, name="training_backend_version")
        _require_nonempty(self.training_batch_contract, name="training_batch_contract")
        if self.schema_version != 1:
            raise ValueError("unsupported MethodSpec schema version")
        for value, allowed, name in (
            (self.method_kind, _METHOD_KINDS, "method_kind"),
            (self.role, _METHOD_ROLES, "role"),
            (self.state_source, _STATE_SOURCES, "state_source"),
            (self.supervision, _SUPERVISION_KINDS, "supervision"),
            (self.reward_signal, _REWARD_SIGNALS, "reward_signal"),
            (self.reward_usage, _REWARD_USAGES, "reward_usage"),
            (self.training_backend, _TRAINING_BACKENDS, "training_backend"),
        ):
            if value not in allowed:
                raise ValueError(f"unsupported {name}: {value!r}")
        if (self.state_source == "current_policy") != (self.current_policy is not None):
            raise ValueError("only current-policy methods may define current-policy semantics")
        if self.requires_common_rollout_bank and (
            self.state_source != "fixed_bank" or self.role != "factorial_cell"
        ):
            raise ValueError("the common rollout bank is exclusive to offline factorial cells")
        if self.requires_teacher_demo_store != (self.state_source == "teacher_demo"):
            raise ValueError("teacher-demo methods must bind the independent teacher-demo store")
        if self.requires_teacher_generation_provenance != self.requires_teacher_demo_store:
            raise ValueError("teacher-demo methods must bind their teacher generation provenance")
        if self.reward_usage == "none" and self.reward_signal != "none":
            raise ValueError("an unused reward signal must be absent")
        if self.reward_usage != "none" and self.reward_signal == "none":
            raise ValueError("reward gate/policy-gradient methods require an explicit reward signal")
        if self.training_backend == TRL_GRPO_BACKEND:
            if self.supervision != "grpo" or self.reward_usage != "policy_gradient":
                raise ValueError("TRL GRPO is reserved for explicit policy-gradient methods")
            if (
                self.training_backend_version != TRL_GRPO_VERSION
                or self.training_batch_contract != TRL_GRPO_BATCH_CONTRACT
            ):
                raise ValueError("TRL GRPO methods must use the pinned version and batch contract")
        elif self.reward_usage == "policy_gradient":
            raise ValueError("registered policy-gradient methods must use the pinned TRL backend")
        elif (
            self.training_backend != FACTORIAL_TRAINING_BACKEND
            or self.training_backend_version != FACTORIAL_TRAINING_BACKEND_VERSION
            or self.training_batch_contract != FACTORIAL_TRAINING_BATCH_CONTRACT
        ):
            raise ValueError("factorial/SFT methods must use the registered in-repository backend")
        if (self.supervision == "soft_teacher") != (self.soft_teacher_objective is not None):
            raise ValueError("only soft-teacher methods may bind the registered top-k objective")
        if self.soft_teacher_objective is not None:
            if self.soft_teacher_objective is not REGISTERED_SOFT_TEACHER_OBJECTIVE:
                raise ValueError("soft-teacher methods must share the canonical objective object")
            if self.objective != self.soft_teacher_objective.objective:
                raise ValueError("soft-teacher objective text differs from its top-k protocol")
        if self.method_kind == "opd" and not (
            self.state_source == "current_policy"
            and self.supervision == "soft_teacher"
            and self.reward_signal == "none"
            and self.reward_usage == "none"
            and self.training_backend == FACTORIAL_TRAINING_BACKEND
        ):
            raise ValueError("OPD is current-policy soft distillation without a reward/PG term")

    @property
    def uses_policy_gradient(self) -> bool:
        return self.reward_usage == "policy_gradient"

    @property
    def is_online(self) -> bool:
        return self.state_source == "current_policy"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def scientific_sha256(self) -> str:
        return sha256_value(self.to_payload())

    def validate_resolved_config(self, config: Mapping[str, Any]) -> None:
        """Fail closed when a composed config changes registered method meaning."""

        experiment = config.get("experiment")
        state_source = config.get("state_source")
        supervision = config.get("supervision")
        if not isinstance(experiment, Mapping):
            raise ValueError("resolved method config requires an experiment mapping")
        if not isinstance(state_source, Mapping) or not isinstance(supervision, Mapping):
            raise ValueError("resolved method config requires state-source and supervision mappings")
        expected = {
            "experiment.name": self.method_id,
            "experiment.state_source": self.state_source,
            "experiment.supervision": self.supervision,
            "state_source.name": self.state_source,
            "supervision.name": self.supervision,
        }
        observed = {
            "experiment.name": experiment.get("name"),
            "experiment.state_source": experiment.get("state_source"),
            "experiment.supervision": experiment.get("supervision"),
            "state_source.name": state_source.get("name"),
            "supervision.name": supervision.get("name"),
        }
        mismatches = {
            name: {"expected": value, "observed": observed[name]}
            for name, value in expected.items()
            if observed[name] != value
        }
        if mismatches:
            raise ValueError(f"resolved method config differs from MethodSpec: {mismatches}")

        if self.current_policy is not None:
            online_expected = {
                "refresh_interval": self.current_policy.refresh_interval,
                "max_policy_lag": self.current_policy.max_policy_lag,
                "refresh_boundary": self.current_policy.refresh_boundary,
                "retry_forces_refresh": self.current_policy.retry_forces_refresh,
            }
            online_mismatches = {
                name: {"expected": value, "observed": state_source.get(name)}
                for name, value in online_expected.items()
                if state_source.get(name) != value
            }
            if online_mismatches:
                raise ValueError(
                    "current-policy config differs from its refresh/lag contract: "
                    f"{online_mismatches}"
                )
        backend_expected = {
            "training_backend": self.training_backend,
            "training_backend_version": self.training_backend_version,
            "training_batch_contract": self.training_batch_contract,
        }
        backend_mismatches = {
            name: {"expected": value, "observed": supervision.get(name)}
            for name, value in backend_expected.items()
            if supervision.get(name) != value
        }
        if backend_mismatches:
            raise ValueError(f"method backend contract changed: {backend_mismatches}")
        if self.soft_teacher_objective is not None:
            soft_expected = {
                "topk_mode": self.soft_teacher_objective.topk_mode,
                "teacher_top_k": self.soft_teacher_objective.teacher_top_k,
                "include_eos": self.soft_teacher_objective.include_eos,
                "minimum_retained_mass": self.soft_teacher_objective.minimum_retained_mass,
                "fail_below_minimum_retained_mass": (
                    self.soft_teacher_objective.fail_below_minimum_retained_mass
                ),
            }
            soft_mismatches = {
                name: {"expected": value, "observed": supervision.get(name)}
                for name, value in soft_expected.items()
                if supervision.get(name) != value
            }
            if soft_mismatches:
                raise ValueError(f"soft-teacher top-k protocol changed: {soft_mismatches}")
        if self.requires_common_rollout_bank:
            if state_source.get("behavior_policy_id") != "common_mu":
                raise ValueError("offline factorial methods require behavior_policy_id=common_mu")
            if state_source.get("include_successes_and_failures") is not True:
                raise ValueError("the common bank must retain successes and failures")
        if self.requires_teacher_demo_store:
            if state_source.get("require_exact_verifier_success") is not True:
                raise ValueError("canonical SFT demos must be exact-verifier gated")
            sft_expected = {
                "normalization": "sequence",
                "require_teacher_demo_store": True,
                "mask_prompt_tokens": True,
            }
            sft_mismatches = {
                name: {"expected": value, "observed": supervision.get(name)}
                for name, value in sft_expected.items()
                if supervision.get(name) != value
            }
            if sft_mismatches:
                raise ValueError(f"canonical SFT objective changed: {sft_mismatches}")

        experiment_reward = experiment.get("reward")
        if self.supervision == "grpo":
            if experiment_reward != self.reward_signal:
                raise ValueError(
                    f"{self.method_id} requires reward={self.reward_signal}, "
                    f"observed={experiment_reward!r}"
                )
            auxiliary_expected = {
                "use_policy_gradient": True,
                "use_teacher_loss": False,
                "use_sft_auxiliary_loss": False,
            }
            auxiliary_mismatches = {
                name: {"expected": value, "observed": experiment.get(name)}
                for name, value in auxiliary_expected.items()
                if experiment.get(name) is not value
            }
            if auxiliary_mismatches:
                raise ValueError(
                    "GRPO anchor/control auxiliary-loss contract changed: "
                    f"{auxiliary_mismatches}"
                )
            sampling_mismatches: dict[str, dict[str, Any]] = {}
            model = config.get("model")
            model_sampling = model.get("sampling_protocol") if isinstance(model, Mapping) else None
            for name in ("temperature", "top_p", "top_k", "min_p"):
                source_value = state_source.get(name)
                supervision_value = supervision.get(name)
                if source_value != supervision_value:
                    sampling_mismatches[name] = {
                        "expected_current_policy": source_value,
                        "observed_grpo": supervision_value,
                    }
                if (
                    isinstance(model_sampling, Mapping)
                    and name in model_sampling
                    and model_sampling.get(name) != supervision_value
                ):
                    sampling_mismatches[f"model.{name}"] = {
                        "expected_model_protocol": model_sampling.get(name),
                        "observed_grpo": supervision_value,
                    }
            if sampling_mismatches:
                raise ValueError(
                    "GRPO sampling differs from the registered current-policy/model protocol: "
                    f"{sampling_mismatches}"
                )
            grpo_expected = {"beta": 0.0, "loss_type": "dapo", "scale_rewards": False}
            grpo_mismatches = {
                name: {"expected": value, "observed": supervision.get(name)}
                for name, value in grpo_expected.items()
                if supervision.get(name) != value
            }
            if grpo_mismatches:
                raise ValueError(f"pinned canonical GRPO settings changed: {grpo_mismatches}")
        elif experiment_reward is not None:
            raise ValueError(f"non-GRPO method {self.method_id} cannot define a training reward")
        if self.supervision == "verified_replay":
            if experiment.get("use_verifier_reward") is not True:
                raise ValueError("verified replay requires the verifier as a selection gate")
        elif experiment.get("use_verifier_reward") is True:
            raise ValueError("verifier reward leaked into a non-replay factorial/SFT method")
        if not self.uses_policy_gradient and (
            experiment.get("use_policy_gradient") is True
            or supervision.get("use_policy_gradient") is True
        ):
            raise ValueError(f"policy-gradient term leaked into {self.method_id}")
        if self.method_kind == "opd" and supervision.get("use_task_rewards") is not False:
            raise ValueError("OPD soft targets must not consume task rewards")
