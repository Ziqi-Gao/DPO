"""In-memory trajectory dataset."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import overload

from posttrain_circuits.core.types import TrajectoryRecord


class TrajectoryDataset(Sequence[TrajectoryRecord]):
    def __init__(self, records: list[TrajectoryRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @overload
    def __getitem__(self, index: int) -> TrajectoryRecord: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TrajectoryRecord]: ...

    def __getitem__(self, index: int | slice) -> TrajectoryRecord | Sequence[TrajectoryRecord]:
        return self.records[index]

    def __iter__(self) -> Iterator[TrajectoryRecord]:
        return iter(self.records)
