"""Cross-checkpoint and cross-method exact mask transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from posttrain_circuits.circuits.graph import AblationSpec
from posttrain_circuits.circuits.masks import top_mask


@dataclass(frozen=True)
class MaskTransferResult:
    source_run: str
    source_checkpoint: str
    source_method: str
    target_run: str
    target_checkpoint: str
    target_method: str
    analysis_mode: str
    sparsity: float
    component_count: int
    faithfulness: float
    necessity: float
    sufficiency: float


def evaluate_mask_transfer(
    *,
    backend: Any,
    target_model: Any,
    validation_pairs: list[Any],
    metric: Any,
    source_scores: dict[str, float],
    sparsity: float,
    source_run: str,
    source_checkpoint: str,
    source_method: str,
    target_run: str,
    target_checkpoint: str,
    target_method: str,
) -> MaskTransferResult:
    mask = top_mask(source_scores, sparsity)
    evaluation = backend.evaluate_mask(
        target_model,
        validation_pairs,
        mask,
        AblationSpec("counterfactual_replacement"),
        metric,
    )
    if evaluation.sufficiency is None:
        raise RuntimeError("mask transfer did not compute sufficiency")
    return MaskTransferResult(
        source_run=source_run,
        source_checkpoint=source_checkpoint,
        source_method=source_method,
        target_run=target_run,
        target_checkpoint=target_checkpoint,
        target_method=target_method,
        analysis_mode="cross_checkpoint_and_method",
        sparsity=mask.sparsity,
        component_count=len(mask.components),
        faithfulness=evaluation.faithfulness,
        necessity=evaluation.necessity,
        sufficiency=evaluation.sufficiency,
    )


def transfer_matrix(
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for request in requests:
        result = evaluate_mask_transfer(**request)
        results.append(asdict(result))
    return results
