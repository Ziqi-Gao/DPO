"""Causal-circuit protocols and serializable scientific artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from posttrain_circuits.artifacts.hashing import sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json, utc_now
from posttrain_circuits.datasets.proofgraph.contracts import CounterfactualPair


def q_to_kv_head_mapping(
    num_query_heads: int,
    num_kv_heads: int,
) -> tuple[int, ...]:
    """Return the total, range-checked GQA query-head mapping."""

    if num_query_heads < 1 or num_kv_heads < 1 or num_query_heads % num_kv_heads:
        raise ValueError("GQA requires query heads to be divisible by key/value heads")
    group_size = num_query_heads // num_kv_heads
    mapping = tuple(index // group_size for index in range(num_query_heads))
    if len(mapping) != num_query_heads or any(
        kv_head < 0 or kv_head >= num_kv_heads for kv_head in mapping
    ):
        raise RuntimeError("GQA query-to-KV mapping is incomplete or out of range")
    if any(mapping.count(kv_head) != group_size for kv_head in range(num_kv_heads)):
        raise RuntimeError("GQA query-to-KV groups do not have equal cardinality")
    return mapping


class CircuitBackend(Protocol):
    """Scientific backend boundary shared by discovery and validation."""

    def score_components(
        self,
        model: Any,
        pairs: list[CounterfactualPair],
        metric: Any,
    ) -> Any: ...

    def evaluate_mask(
        self,
        model: Any,
        pairs: list[CounterfactualPair],
        mask: Any,
        ablation: Any,
    ) -> Any: ...


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
    semantic_probe_manifest: dict[str, Any]
    tokenized_probe_manifest: dict[str, Any]
    discovery_pair_manifest: dict[str, Any]
    probe_cohort_manifest: dict[str, Any]
    model_compatibility: dict[str, Any]
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
    circuit_probe_schema_version: str = ""
    prereg_version: str = ""
    generator_version: str = ""
    label_semantics: str = ""
    probe_stage: str = ""
    semantic_probe_manifest_hash: str = ""
    tokenized_probe_manifest_hash: str = ""
    stage_target_manifest_hash: str = ""
    semantic_pair_hashes: list[str] = field(default_factory=list)
    tokenized_pair_hashes: list[str] = field(default_factory=list)
    protocol_track: str = ""
    artifact_namespace: str = ""
    model_revision: str = ""
    teacher_revision: str = ""
    tokenizer_fingerprint: str = ""
    chat_template_sha256: str = ""
    prompt_protocol: str = ""
    enable_thinking: bool = False
    code_commit: str = ""
    prereg_path: str = ""
    prereg_commit: str = ""
    prereg_sha256: str = ""
    protocol_amendment_id: str = ""
    protocol_amendment_path: str = ""
    protocol_amendment_git_commit: str = ""
    protocol_amendment_sha256: str = ""
    reviewed_implementation_commit: str = ""
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        structured_inputs = {
            "semantic_probe_manifest": self.semantic_probe_manifest,
            "tokenized_probe_manifest": self.tokenized_probe_manifest,
            "discovery_pair_manifest": self.discovery_pair_manifest,
            "probe_cohort_manifest": self.probe_cohort_manifest,
            "model_compatibility": self.model_compatibility,
        }
        empty = [name for name, payload in structured_inputs.items() if not payload]
        if empty:
            raise ValueError(
                "circuit discovery artifact requires embedded structured inputs: "
                + ", ".join(empty)
            )
        amendment = {
            "protocol_amendment_id": self.protocol_amendment_id,
            "protocol_amendment_path": self.protocol_amendment_path,
            "protocol_amendment_git_commit": self.protocol_amendment_git_commit,
            "protocol_amendment_sha256": self.protocol_amendment_sha256,
            "reviewed_implementation_commit": self.reviewed_implementation_commit,
        }
        if any(amendment.values()) and not all(amendment.values()):
            raise ValueError("circuit amendment binding must be complete or omitted")
        if all(amendment.values()):
            for name, length in (
                ("protocol_amendment_git_commit", 40),
                ("reviewed_implementation_commit", 40),
                ("protocol_amendment_sha256", 64),
            ):
                value = amendment[name]
                if not isinstance(value, str) or len(value) != length or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(f"circuit {name} is not a lowercase hexadecimal binding")

    def write(self, path: Path) -> None:
        payload = asdict(self)
        payload["sha256"] = sha256_value(payload)
        atomic_write_json(path, payload)
