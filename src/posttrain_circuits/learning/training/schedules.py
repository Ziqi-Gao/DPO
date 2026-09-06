"""Deterministic prompt and learning-rate schedules.

The normal ``PromptScheduler`` deliberately keeps its historical rank-local
round-robin behavior.  Candidate E adds a separate, explicitly selected
partition protocol for a scheduler-chosen GPU world size.  That protocol
defines optimizer windows in global logical slots, rather than in rank-local
sample counters, so the same 64 prompts are consumed in the same global order
at every supported allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

from posttrain_circuits.learning.contracts import PromptBatch


ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1 = "allocation_neutral_exact_global_batch_v1"
LEGACY_BATCH_PARTITION_PROTOCOL = "rank_local_round_robin_v1"
_SUPPORTED_ALLOCATION_NEUTRAL_WORLD_SIZES = frozenset({1, 2, 3, 4})


def _require_exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ExactGlobalBatchPlan:
    """The rank-local view of one allocation-neutral 64-slot optimizer window.

    Logical slot ``s`` belongs to rank ``s % world_size``.  Every rank executes
    the same number of backward calls, which matters for FSDP/DDP collectives;
    smaller tails are placed only in the final call.  For three ranks that is
    exactly ``(4, 4, 4, 4, 4, 2)``, ``(4, 4, 4, 4, 4, 1)``, and
    ``(4, 4, 4, 4, 4, 1)``.
    """

    global_batch_size: int
    max_microbatch_size: int
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        _require_exact_int(self.global_batch_size, name="global_batch_size", minimum=1)
        _require_exact_int(
            self.max_microbatch_size,
            name="max_microbatch_size",
            minimum=1,
        )
        _require_exact_int(self.world_size, name="world_size", minimum=1)
        _require_exact_int(self.rank, name="rank", minimum=0)
        if self.world_size not in _SUPPORTED_ALLOCATION_NEUTRAL_WORLD_SIZES:
            raise ValueError(
                "allocation-neutral exact-global batching supports only world sizes 1, 2, 3, and 4"
            )
        if self.rank >= self.world_size:
            raise ValueError("rank is outside the allocation-neutral world")
        # These values are a reviewed scientific contract, not generic tuning
        # knobs.  A different pair needs a versioned protocol and review.
        if self.global_batch_size != 64:
            raise ValueError("allocation-neutral exact-global batching requires global_batch_size=64")
        if self.max_microbatch_size != 4:
            raise ValueError("allocation-neutral exact-global batching requires max_microbatch_size=4")

    @property
    def local_sequence_count(self) -> int:
        """Number of logical sequences assigned to this rank in one update."""

        return len(range(self.rank, self.global_batch_size, self.world_size))

    @property
    def microsteps_per_optimizer_update(self) -> int:
        """Collective-safe count of local backward calls for every rank."""

        denominator = self.world_size * self.max_microbatch_size
        return (self.global_batch_size + denominator - 1) // denominator

    @property
    def microbatch_sizes(self) -> tuple[int, ...]:
        """Rank-local physical microbatch sizes in optimizer-window order."""

        remaining = self.local_sequence_count
        sizes: list[int] = []
        for _ in range(self.microsteps_per_optimizer_update):
            size = min(self.max_microbatch_size, remaining)
            if size < 1:
                raise ValueError("allocation-neutral plan would issue an empty rank microbatch")
            sizes.append(size)
            remaining -= size
        if remaining != 0:
            raise ValueError("allocation-neutral plan did not consume its rank-local slot count")
        return tuple(sizes)

    def global_slots_for_microbatch(self, microbatch_index: int) -> tuple[int, ...]:
        """Return the allocation-independent logical slots for one local call."""

        _require_exact_int(microbatch_index, name="microbatch_index", minimum=0)
        sizes = self.microbatch_sizes
        if microbatch_index >= len(sizes):
            raise ValueError("microbatch index is outside the optimizer-window plan")
        local_offset = sum(sizes[:microbatch_index])
        return tuple(
            self.rank + (local_offset + offset) * self.world_size
            for offset in range(sizes[microbatch_index])
        )

    def accelerate_sequence_mean_loss_multiplier(self, local_batch_size: int) -> float:
        """Scale a local sequence mean before Accelerate's accumulation/DDP mean.

        Accelerate divides the submitted loss by the number of accumulation
        calls, and FSDP/DDP averages gradients over ``world_size`` ranks.  The
        returned multiplier therefore converts a local sequence mean into its
        exact contribution to the global 64-sequence mean.
        """

        _require_exact_int(local_batch_size, name="local_batch_size", minimum=1)
        if local_batch_size > self.max_microbatch_size:
            raise ValueError("local batch exceeds the allocation-neutral microbatch cap")
        return (
            self.microsteps_per_optimizer_update
            * self.world_size
            * local_batch_size
            / self.global_batch_size
        )

    def manual_distributed_sequence_mean_loss_multiplier(self, local_batch_size: int) -> float:
        """Scale a local sequence mean when the caller, not Accelerate, accumulates."""

        _require_exact_int(local_batch_size, name="local_batch_size", minimum=1)
        if local_batch_size > self.max_microbatch_size:
            raise ValueError("local batch exceeds the allocation-neutral microbatch cap")
        return self.world_size * local_batch_size / self.global_batch_size


@dataclass
class PromptScheduler:
    prompt_ids: list[str]
    prompt_texts: list[str]
    batch_size: int
    position: int = 0
    rank: int = 0
    world_size: int = 1
    batch_partition_protocol: str = LEGACY_BATCH_PARTITION_PROTOCOL
    global_batch_size: int | None = None
    max_microbatch_size: int | None = None
    global_slot_cursor: int = 0
    microbatch_index: int = 0

    def __post_init__(self) -> None:
        if not self.prompt_ids or len(self.prompt_ids) != len(self.prompt_texts):
            raise ValueError("prompt scheduler needs aligned, non-empty prompts")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid distributed prompt-scheduler rank")
        if self.batch_partition_protocol == LEGACY_BATCH_PARTITION_PROTOCOL:
            if self.global_batch_size is not None or self.max_microbatch_size is not None:
                raise ValueError("legacy prompt scheduler must not declare an exact-global batch contract")
            return
        if self.batch_partition_protocol != ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1:
            raise ValueError(f"unsupported prompt batch-partition protocol {self.batch_partition_protocol!r}")
        if self.global_batch_size is None or self.max_microbatch_size is None:
            raise ValueError("allocation-neutral prompt scheduler requires global and microbatch sizes")
        plan = ExactGlobalBatchPlan(
            global_batch_size=self.global_batch_size,
            max_microbatch_size=self.max_microbatch_size,
            rank=self.rank,
            world_size=self.world_size,
        )
        if self.batch_size != plan.max_microbatch_size:
            raise ValueError("allocation-neutral prompt scheduler batch_size must equal max_microbatch_size")
        _require_exact_int(self.global_slot_cursor, name="global_slot_cursor", minimum=0)
        _require_exact_int(self.microbatch_index, name="microbatch_index", minimum=0)
        if self.global_slot_cursor % plan.global_batch_size != 0:
            raise ValueError("global slot cursor must begin at an optimizer-window boundary")
        if self.microbatch_index >= plan.microsteps_per_optimizer_update:
            raise ValueError("microbatch index is outside the allocation-neutral optimizer window")

    @classmethod
    def for_distributed_rank(
        cls,
        prompt_ids: list[str],
        prompt_texts: list[str],
        batch_size: int,
        *,
        rank: int,
        world_size: int,
    ) -> PromptScheduler:
        """Build deterministic, disjoint strided prompt shards before training."""

        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed prompt shard")
        if len(prompt_ids) != len(prompt_texts):
            raise ValueError("distributed prompt shard inputs are not aligned")
        shard_ids = prompt_ids[rank::world_size]
        shard_texts = prompt_texts[rank::world_size]
        if not shard_ids:
            raise ValueError("distributed prompt shard is empty")
        return cls(
            shard_ids,
            shard_texts,
            min(batch_size, len(shard_ids)),
            rank=rank,
            world_size=world_size,
        )

    @classmethod
    def for_allocation_neutral_rank(
        cls,
        prompt_ids: list[str],
        prompt_texts: list[str],
        *,
        rank: int,
        world_size: int,
        global_batch_size: int,
        max_microbatch_size: int,
    ) -> PromptScheduler:
        """Build the rank view of the reviewed allocation-neutral batch protocol.

        Unlike :meth:`for_distributed_rank`, this keeps the full prompt
        population on every rank.  ``next_batch`` maps logical global slots to
        the rank's strided slice of one 64-sequence optimizer window.
        """

        if len(prompt_ids) != len(prompt_texts):
            raise ValueError("allocation-neutral prompt inputs are not aligned")
        return cls(
            list(prompt_ids),
            list(prompt_texts),
            max_microbatch_size,
            rank=rank,
            world_size=world_size,
            batch_partition_protocol=ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1,
            global_batch_size=global_batch_size,
            max_microbatch_size=max_microbatch_size,
        )

    @property
    def exact_global_batch_plan(self) -> ExactGlobalBatchPlan | None:
        if self.batch_partition_protocol != ALLOCATION_NEUTRAL_EXACT_GLOBAL_BATCH_V1:
            return None
        assert self.global_batch_size is not None
        assert self.max_microbatch_size is not None
        return ExactGlobalBatchPlan(
            global_batch_size=self.global_batch_size,
            max_microbatch_size=self.max_microbatch_size,
            rank=self.rank,
            world_size=self.world_size,
        )

    @property
    def microsteps_per_optimizer_update(self) -> int | None:
        plan = self.exact_global_batch_plan
        return plan.microsteps_per_optimizer_update if plan is not None else None

    def next_batch(self) -> PromptBatch:
        plan = self.exact_global_batch_plan
        if plan is not None:
            slots = plan.global_slots_for_microbatch(self.microbatch_index)
            indices = [
                (self.global_slot_cursor + slot) % len(self.prompt_ids)
                for slot in slots
            ]
            self.microbatch_index += 1
            if self.microbatch_index == plan.microsteps_per_optimizer_update:
                self.global_slot_cursor += plan.global_batch_size
                self.microbatch_index = 0
            return PromptBatch(
                tuple(self.prompt_ids[index] for index in indices),
                tuple(self.prompt_texts[index] for index in indices),
            )
        indices = [(self.position + offset) % len(self.prompt_ids) for offset in range(self.batch_size)]
        self.position = (self.position + self.batch_size) % len(self.prompt_ids)
        return PromptBatch(
            tuple(self.prompt_ids[index] for index in indices),
            tuple(self.prompt_texts[index] for index in indices),
        )

    def state_dict(self) -> dict[str, int | str]:
        plan = self.exact_global_batch_plan
        if plan is not None:
            return {
                "batch_partition_protocol": self.batch_partition_protocol,
                "global_batch_size": plan.global_batch_size,
                "max_microbatch_size": plan.max_microbatch_size,
                "global_slot_cursor": self.global_slot_cursor,
                "microbatch_index": self.microbatch_index,
                "rank": self.rank,
                "world_size": self.world_size,
            }
        return {"position": self.position, "rank": self.rank, "world_size": self.world_size}

    def load_state_dict(self, state: dict[str, int | str]) -> None:
        plan = self.exact_global_batch_plan
        if plan is not None:
            expected = {
                "batch_partition_protocol": self.batch_partition_protocol,
                "global_batch_size": plan.global_batch_size,
                "max_microbatch_size": plan.max_microbatch_size,
                "rank": self.rank,
                "world_size": self.world_size,
            }
            observed = {name: state.get(name) for name in expected}
            if observed != expected:
                raise ValueError(
                    "prompt-scheduler checkpoint belongs to a different allocation batch partition"
                )
            global_slot_cursor = state.get("global_slot_cursor")
            microbatch_index = state.get("microbatch_index")
            global_slot_cursor = _require_exact_int(
                global_slot_cursor,
                name="global_slot_cursor",
                minimum=0,
            )
            microbatch_index = _require_exact_int(
                microbatch_index,
                name="microbatch_index",
                minimum=0,
            )
            if global_slot_cursor % plan.global_batch_size != 0:
                raise ValueError("prompt-scheduler checkpoint has a partial global optimizer window")
            if microbatch_index >= plan.microsteps_per_optimizer_update:
                raise ValueError("prompt-scheduler checkpoint microbatch index is invalid")
            self.global_slot_cursor = global_slot_cursor
            self.microbatch_index = microbatch_index
            return
        if (
            int(state.get("rank", self.rank)) != self.rank
            or int(state.get("world_size", self.world_size)) != self.world_size
        ):
            raise ValueError("prompt-scheduler checkpoint belongs to a different rank shard")
        position = _require_exact_int(state.get("position"), name="position", minimum=0)
        if position >= len(self.prompt_ids):
            raise ValueError("prompt-scheduler checkpoint position is outside the prompt shard")
        self.position = position
