"""Current-policy source with explicit refresh and lag rejection."""

from __future__ import annotations

from typing import Any

from posttrain_circuits.core.types import PromptBatch, TrajectoryBatch
from posttrain_circuits.rollout.base import TrajectoryGenerator


class CurrentPolicyStateSource:
    def __init__(
        self,
        generator: TrajectoryGenerator,
        *,
        refresh_interval: int = 1,
        max_policy_lag: int = 0,
        seed: int = 0,
    ) -> None:
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be positive")
        self.generator = generator
        self.refresh_interval = refresh_interval
        self.max_policy_lag = max_policy_lag
        self.seed = seed
        self.policy_version = 0
        self._last_refresh_step = -1
        self._request_count = 0

    def refresh_if_needed(self, model: Any, step: int) -> None:
        del model
        if self._last_refresh_step < 0 or step - self._last_refresh_step >= self.refresh_interval:
            self.policy_version += 1
            self._last_refresh_step = step

    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch:
        self.refresh_if_needed(model, step)
        records = self.generator(
            model,
            prompt_batch,
            self.policy_version,
            self.seed + self._request_count,
        )
        self._request_count += 1
        for record in records:
            if record.policy_version != self.policy_version:
                raise ValueError(
                    f"stale or mismatched trajectory {record.trajectory_id}: requested policy "
                    f"version {self.policy_version}, generator returned {record.policy_version}"
                )
        batch = TrajectoryBatch(records, self.policy_version)
        batch.validate(max_policy_lag=self.max_policy_lag)
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "current_policy",
            "refresh_interval": self.refresh_interval,
            "max_policy_lag": self.max_policy_lag,
            "seed": self.seed,
            "policy_version": self.policy_version,
            "last_refresh_step": self._last_refresh_step,
            "request_count": self._request_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("refresh_interval", "max_policy_lag", "seed"):
            if int(state[name]) != int(getattr(self, name)):
                raise ValueError(f"current-policy checkpoint {name} does not match configuration")
        self.policy_version = int(state["policy_version"])
        self._last_refresh_step = int(state["last_refresh_step"])
        self._request_count = int(state["request_count"])
