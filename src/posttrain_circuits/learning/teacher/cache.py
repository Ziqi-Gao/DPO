"""Content-addressed teacher-score cache keys."""

from __future__ import annotations

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.datasets.trajectories.contracts import TrajectoryRecord


def teacher_cache_key(record: TrajectoryRecord, teacher_id: str, teacher_revision: str, top_k: int) -> str:
    return sha256_value(
        {
            "prompt": record.input_ids,
            "response": record.response_ids,
            "teacher_id": teacher_id,
            "teacher_revision": teacher_revision,
            "top_k": top_k,
        }
    )
