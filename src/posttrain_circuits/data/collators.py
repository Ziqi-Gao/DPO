"""Causal-LM trajectory collation with explicit response spans."""

from __future__ import annotations

import torch

from posttrain_circuits.core.types import SupervisionBatch, TrajectoryBatch


def collate_trajectories(
    trajectories: TrajectoryBatch,
    *,
    pad_token_id: int,
    include_teacher: bool = False,
) -> SupervisionBatch:
    if not trajectories.records:
        raise ValueError("cannot collate an empty trajectory batch")
    lengths = [len(record.input_ids) + len(record.response_ids) for record in trajectories.records]
    maximum = max(lengths)
    batch_size = len(lengths)
    input_ids = torch.full((batch_size, maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    response_mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    rewards = torch.empty(batch_size, dtype=torch.float32)
    top_k = max(
        (len(position) for record in trajectories.records for position in record.teacher_topk_ids),
        default=0,
    )
    topk_ids = torch.zeros((batch_size, maximum, top_k), dtype=torch.long) if include_teacher else None
    topk_logprobs = (
        torch.full((batch_size, maximum, top_k), float("-inf"), dtype=torch.float32)
        if include_teacher
        else None
    )
    topk_mass = torch.ones((batch_size, maximum), dtype=torch.float32) if include_teacher else None

    for row, record in enumerate(trajectories.records):
        record.validate()
        tokens = record.input_ids + record.response_ids
        input_ids[row, : len(tokens)] = torch.tensor(tokens)
        attention_mask[row, : len(tokens)] = True
        start = len(record.input_ids)
        for index, include in enumerate(record.response_token_mask):
            response_mask[row, start + index] = include
        rewards[row] = float(record.verifier_reward or 0.0)
        if include_teacher:
            if not record.teacher_topk_ids:
                raise ValueError(f"trajectory {record.trajectory_id} has no teacher scores")
            assert topk_ids is not None and topk_logprobs is not None and topk_mass is not None
            for index, ids in enumerate(record.teacher_topk_ids):
                width = len(ids)
                topk_ids[row, start + index, :width] = torch.tensor(ids)
                topk_logprobs[row, start + index, :width] = torch.tensor(record.teacher_topk_logprobs[index])
                topk_mass[row, start + index] = record.teacher_topk_mass[index]
    return SupervisionBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        rewards=rewards,
        teacher_topk_ids=topk_ids,
        teacher_topk_logprobs=topk_logprobs,
        teacher_topk_mass=topk_mass,
        sequence_ids=torch.arange(batch_size).unsqueeze(1).expand(batch_size, maximum),
    )
