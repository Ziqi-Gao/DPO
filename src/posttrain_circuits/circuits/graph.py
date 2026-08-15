"""Serializable full-score and mask artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from posttrain_circuits.core.manifests import atomic_write_json, utc_now


@dataclass
class CircuitScores:
    scores: dict[str, float]
    uncertainty: dict[str, float] = field(default_factory=dict)
    node_scores: dict[str, float] = field(default_factory=dict)
    edge_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CircuitMask:
    components: tuple[str, ...]
    sparsity: float


@dataclass(frozen=True)
class AblationSpec:
    kind: str = "counterfactual_replacement"


@dataclass
class CircuitEvaluation:
    clean_metric: float
    corrupt_metric: float
    patched_metric: float
    faithfulness: float
    necessity: float
    sufficiency: float | None = None


@dataclass
class CircuitArtifact:
    run_id: str
    checkpoint_id: str
    task_manifest_hash: str
    pair_manifest_hash: str
    backend_version: str
    model_compatibility_hash: str
    node_or_edge_level: str
    integrated_gradient_steps: int
    ablation_baseline: str
    scores: dict[str, float]
    score_uncertainty: dict[str, float]
    node_scores: dict[str, float] = field(default_factory=dict)
    edge_scores: dict[str, float] = field(default_factory=dict)
    backend_name: str = ""
    backend_revision: str = ""
    attribution_method: str = ""
    discovery_pair_count: int = 0
    uncertainty_method: str = ""
    bootstrap_score_vectors: list[dict[str, float]] = field(default_factory=list)
    bootstrap_resample_indices: list[list[int]] = field(default_factory=list)
    bootstrap_raw_graph_hashes: list[str] = field(default_factory=list)
    primary_raw_graph_hash: str = ""
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    base_model_revision: str = ""
    resolved_model_commit: str = ""
    tokenizer_revision: str = ""
    tokenizer_hash: str = ""
    probe_cohort: str = ""
    probe_subset: str = ""
    probe_cohort_manifest_hash: str = ""
    graph_convention: str = ""
    created_at: str = field(default_factory=utc_now)

    def write(self, path: Path) -> None:
        atomic_write_json(path, asdict(self))
