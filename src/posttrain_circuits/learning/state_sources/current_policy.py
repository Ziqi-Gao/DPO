"""Current-policy source with explicit refresh and lag rejection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from posttrain_circuits.learning.contracts import (
    PromptBatch,
    SamplingCursor,
    SamplingRequest,
    TrajectoryBatch,
    TrajectoryGenerator,
)
from posttrain_circuits.methods.specs import REGISTERED_CURRENT_POLICY

CURRENT_POLICY_SAMPLING_PROTOCOL_ID = "current-policy-v1-stable-cursor"


class CurrentPolicyStateSource:
    def __init__(
        self,
        generator: TrajectoryGenerator,
        *,
        refresh_interval: int = REGISTERED_CURRENT_POLICY.refresh_interval,
        max_policy_lag: int = REGISTERED_CURRENT_POLICY.max_policy_lag,
        seed: int = 0,
        sampling_protocol_id: str = CURRENT_POLICY_SAMPLING_PROTOCOL_ID,
    ) -> None:
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be positive")
        self.generator = generator
        self.refresh_interval = refresh_interval
        self.max_policy_lag = max_policy_lag
        self.seed = seed
        if not sampling_protocol_id:
            raise ValueError("sampling_protocol_id must be non-empty")
        self.sampling_protocol_id = sampling_protocol_id
        self.policy_version = 0
        self._last_refresh_step = -1
        self._last_request_step = -1
        self._next_retry_index = 0

    def refresh_if_needed(self, model: Any, step: int) -> None:
        del model
        if self._last_refresh_step < 0 or step - self._last_refresh_step >= self.refresh_interval:
            self.policy_version += 1
            self._last_refresh_step = step

    def get_batch(self, model: Any, prompt_batch: PromptBatch, step: int) -> TrajectoryBatch:
        self.refresh_if_needed(model, step)
        if step != self._last_request_step:
            retry_index = 0
            self._last_request_step = step
            self._next_retry_index = 1
        else:
            retry_index = self._next_retry_index
            self._next_retry_index += 1
        generation_indices: dict[str, int] = defaultdict(int)
        cursors: list[SamplingCursor] = []
        for prompt_id in prompt_batch.prompt_ids:
            generation_index = generation_indices[prompt_id]
            generation_indices[prompt_id] += 1
            cursors.append(
                SamplingCursor(
                    optimizer_step=step,
                    retry_index=retry_index,
                    prompt_id=prompt_id,
                    generation_index=generation_index,
                )
            )
        sampling_request = SamplingRequest(
            sampling_request_seed=self.seed,
            sampling_protocol_id=self.sampling_protocol_id,
            cursors=tuple(cursors),
        )
        records = self.generator(
            model,
            prompt_batch,
            self.policy_version,
            sampling_request,
        )
        if len(records) != len(cursors):
            raise ValueError("current-policy generator returned the wrong trajectory count")
        for index, (record, cursor) in enumerate(zip(records, cursors, strict=True)):
            if record.policy_version != self.policy_version:
                raise ValueError(
                    f"stale or mismatched trajectory {record.trajectory_id}: requested policy "
                    f"version {self.policy_version}, generator returned {record.policy_version}"
                )
            expected_seed = sampling_request.seed_for(index, policy_version=self.policy_version)
            if (
                record.prompt_id != cursor.prompt_id
                or record.sampling_request_seed != self.seed
                or record.actual_sampling_seed != expected_seed
                or record.sampling_cursor_id != cursor.cursor_id
                or record.sampling_protocol_id != self.sampling_protocol_id
            ):
                raise ValueError("current-policy generator returned mismatched sampling provenance")
        batch = TrajectoryBatch(records, self.policy_version)
        batch.validate(max_policy_lag=self.max_policy_lag)
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "current_policy",
            "refresh_interval": self.refresh_interval,
            "max_policy_lag": self.max_policy_lag,
            "refresh_boundary": REGISTERED_CURRENT_POLICY.refresh_boundary,
            "retry_forces_refresh": REGISTERED_CURRENT_POLICY.retry_forces_refresh,
            "seed": self.seed,
            "sampling_protocol_id": self.sampling_protocol_id,
            "policy_version": self.policy_version,
            "last_refresh_step": self._last_refresh_step,
            "last_request_step": self._last_request_step,
            "next_retry_index": self._next_retry_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("refresh_interval", "max_policy_lag", "seed"):
            if int(state[name]) != int(getattr(self, name)):
                raise ValueError(f"current-policy checkpoint {name} does not match configuration")
        if state.get("refresh_boundary") != REGISTERED_CURRENT_POLICY.refresh_boundary:
            raise ValueError("current-policy checkpoint changed the optimizer-update boundary")
        if state.get("retry_forces_refresh") is not REGISTERED_CURRENT_POLICY.retry_forces_refresh:
            raise ValueError("current-policy checkpoint changed retry refresh semantics")
        if state.get("sampling_protocol_id") != self.sampling_protocol_id:
            raise ValueError("current-policy checkpoint changed sampling protocol")
        self.policy_version = int(state["policy_version"])
        self._last_refresh_step = int(state["last_refresh_step"])
        self._last_request_step = int(state["last_request_step"])
        self._next_retry_index = int(state["next_retry_index"])
