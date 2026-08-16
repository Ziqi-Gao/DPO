"""Frozen semantic and tokenizer-specific stage circuit probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.types import CounterfactualPair
from posttrain_circuits.models.loading import tokenizer_fingerprint
from posttrain_circuits.tasks.proofgraph.generator import GENERATOR_VERSION, LABEL_SEMANTICS
from posttrain_circuits.tasks.proofgraph.renderer import render_proof_literal, render_step

CIRCUIT_PROBE_SCHEMA_VERSION = "circuit-probe-v2-stage-sequence"
PROBE_STAGES = (
    "first_rule_selection",
    "intermediate_conclusion",
    "final_answer",
)
PRIMARY_CORRUPTION = "active_support_path_swap"


@dataclass(frozen=True)
class SemanticCircuitProbeSpec:
    probe_id: str
    semantic_pair_group_id: str
    subset: str
    stage: str
    clean_context: str
    corrupt_context: str
    clean_target: str
    corrupt_target: str
    corruption_type: str
    corruption_class: str
    changed_semantic_field: str
    source_example_ids: tuple[str, str]
    semantic_pair_hash: str
    query_text: str
    fact_ids: tuple[str, ...]
    rule_ids: tuple[str, ...]
    schema_version: str = CIRCUIT_PROBE_SCHEMA_VERSION
    generator_version: str = GENERATOR_VERSION
    label_semantics: str = LABEL_SEMANTICS
    prereg_version: str = "core_v2"


@dataclass(frozen=True)
class CircuitProbeSpec:
    probe_id: str
    semantic_pair_group_id: str
    subset: str
    stage: str
    clean_context: str
    corrupt_context: str
    clean_model_input: str
    corrupt_model_input: str
    clean_target: str
    corrupt_target: str
    clean_input_ids: tuple[int, ...]
    corrupt_input_ids: tuple[int, ...]
    clean_target_ids: tuple[int, ...]
    corrupt_target_ids: tuple[int, ...]
    clean_metric_positions: tuple[int, ...]
    corrupt_metric_positions: tuple[int, ...]
    clean_intervention_positions: tuple[int, ...]
    corrupt_intervention_positions: tuple[int, ...]
    corruption_type: str
    corruption_class: str
    changed_semantic_field: str
    source_example_ids: tuple[str, str]
    semantic_pair_hash: str
    semantic_manifest_hash: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_hash: str
    alignment_checks: tuple[tuple[str, bool], ...]
    tokenized_pair_hash: str
    schema_version: str = CIRCUIT_PROBE_SCHEMA_VERSION
    generator_version: str = GENERATOR_VERSION
    label_semantics: str = LABEL_SEMANTICS
    prereg_version: str = "core_v2"

    def __post_init__(self) -> None:
        if self.stage not in PROBE_STAGES:
            raise ValueError(f"unknown circuit probe stage {self.stage!r}")
        if self.corruption_type == "query_flip" and self.corruption_class != "query_routing_only":
            raise ValueError("query_flip probes must be marked auxiliary query_routing_only")
        if self.corruption_class == "semantic_target_changing" and self.corruption_type == "query_flip":
            raise ValueError("query_flip cannot be a primary reasoning probe")
        if not self.clean_target_ids or not self.corrupt_target_ids:
            raise ValueError("circuit targets must contain at least one token")
        if len(self.clean_target_ids) != len(self.corrupt_target_ids):
            raise ValueError("clean/corrupt target sequences must have equal token length")
        if len(self.clean_input_ids) != len(self.corrupt_input_ids):
            raise ValueError("clean/corrupt model inputs must be token-shape matched")
        if len(self.clean_metric_positions) != len(self.clean_target_ids):
            raise ValueError("clean metric positions do not align with the target sequence")
        if len(self.corrupt_metric_positions) != len(self.corrupt_target_ids):
            raise ValueError("corrupt metric positions do not align with the target sequence")
        if any(not passed for _, passed in self.alignment_checks):
            failed = [name for name, passed in self.alignment_checks if not passed]
            raise ValueError(f"primary circuit probe failed tokenizer alignment: {failed}")
        if self.stage != "final_answer" and {self.clean_target, self.corrupt_target} <= {"0", "1"}:
            raise ValueError("process-stage probes cannot silently use final-answer targets")
        if self.stage == "final_answer" and set(self.clean_target + self.corrupt_target) - {"0", "1"}:
            raise ValueError("final-answer probes must target answer bits")


def _stage_context(pair: CounterfactualPair, stage: str, *, clean: bool) -> tuple[str, str]:
    example = pair.clean_example if clean else pair.corrupt_example
    prompt = pair.clean_prompt if clean else pair.corrupt_prompt
    proof = example.canonical_proof
    if not proof:
        raise ValueError("every signed circuit probe requires a nonempty canonical proof")
    if stage == "first_rule_selection":
        return f"{prompt}\n\n<proof>\nS01: ", proof[0].rule_id
    if stage == "intermediate_conclusion":
        first = proof[0]
        expected = example.query if example.label == 1 else example.query.flipped()
        if first.conclusion == expected:
            raise ValueError("intermediate-conclusion probes require proof depth above one")
        citations = ",".join(first.citations)
        context = f"{prompt}\n\n<proof>\n{first.step_id}: {first.rule_id}({citations}) -> "
        return context, render_proof_literal(first)
    if stage == "final_answer":
        body = "\n".join(render_step(step) for step in proof)
        return f"{prompt}\n\n<proof>\n{body}\n</proof>\n<answer>", str(example.label)
    raise ValueError(f"unknown circuit stage {stage!r}")


def build_semantic_probe_specs(
    pairs: list[CounterfactualPair],
    *,
    subset: str,
) -> list[SemanticCircuitProbeSpec]:
    if subset not in {"discovery", "validation"}:
        raise ValueError("circuit probe subset must be discovery or validation")
    specs = []
    for pair in pairs:
        clean = pair.clean_example
        corrupt = pair.corrupt_example
        if clean.query != corrupt.query:
            raise ValueError("primary circuit pairs must keep the query unchanged")
        if clean.rules != corrupt.rules:
            raise ValueError("primary circuit pairs must keep the rule set unchanged")
        corruption_class = str(corrupt.metadata.get("corruption_class", ""))
        if pair.corruption_type == PRIMARY_CORRUPTION and corruption_class != "semantic_target_changing":
            raise ValueError("primary support swaps must change the semantic target")
        semantic_pair_hash = sha256_value(
            {
                "pair_group_id": clean.pair_group_id,
                "query": str(clean.query),
                "rules": [asdict(rule) for rule in clean.rules.values()],
                "clean_facts": [asdict(value) for value in clean.facts.values()],
                "corrupt_facts": [asdict(value) for value in corrupt.facts.values()],
                "corruption": pair.corruption_type,
            }
        )
        for stage in PROBE_STAGES:
            clean_context, clean_target = _stage_context(pair, stage, clean=True)
            corrupt_context, corrupt_target = _stage_context(pair, stage, clean=False)
            if clean_target == corrupt_target:
                raise ValueError(f"stage {stage} did not change its behavioral target")
            specs.append(
                SemanticCircuitProbeSpec(
                    probe_id=f"{subset}-{stage}-{semantic_pair_hash[:16]}",
                    semantic_pair_group_id=clean.pair_group_id,
                    subset=subset,
                    stage=stage,
                    clean_context=clean_context,
                    corrupt_context=corrupt_context,
                    clean_target=clean_target,
                    corrupt_target=corrupt_target,
                    corruption_type=pair.corruption_type,
                    corruption_class=corruption_class,
                    changed_semantic_field=pair.changed_semantic_field,
                    source_example_ids=(clean.example_id, corrupt.example_id),
                    semantic_pair_hash=semantic_pair_hash,
                    query_text=str(clean.query),
                    fact_ids=tuple(clean.facts),
                    rule_ids=tuple(clean.rules),
                )
            )
    return specs


def semantic_probe_manifest(specs: list[SemanticCircuitProbeSpec]) -> dict[str, Any]:
    if not specs:
        raise ValueError("semantic circuit probe manifest cannot be empty")
    rows = [asdict(spec) for spec in specs]
    content = {
        "format_version": 1,
        "circuit_probe_schema_version": CIRCUIT_PROBE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "prereg_version": "core_v2",
        "probe_count": len(rows),
        "stages": list(PROBE_STAGES),
        "subsets": sorted({spec.subset for spec in specs}),
        "probes": rows,
    }
    return {**content, "sha256": sha256_value(content)}


def _token_spans(tokenizer: Any, text: str, values: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    spans = []
    for value in values:
        start = 0
        value_positions: list[int] = []
        while True:
            found = text.find(value, start)
            if found < 0:
                break
            end = found + len(value)
            value_positions.extend(
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > found and token_start < end
            )
            start = end
        if not value_positions:
            raise ValueError(f"semantic alignment value {value!r} is absent")
        spans.append(tuple(value_positions))
    return tuple(spans)


def _teacher_forced_encoding(
    tokenizer: Any,
    context: str,
    target: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], str]:
    full_text = context + target
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = tuple(int(value) for value in encoded["input_ids"])
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    boundary = len(context)
    target_positions = tuple(
        index for index, (start, end) in enumerate(offsets) if end > boundary and start >= boundary
    )
    if not target_positions or target_positions != tuple(range(target_positions[0], len(input_ids))):
        raise ValueError("target must tokenize to a contiguous suffix")
    if offsets[target_positions[0]][0] < boundary or target_positions[0] == 0:
        raise ValueError("tokenizer merged the context/target boundary")
    target_ids = tuple(input_ids[index] for index in target_positions)
    model_input_ids = input_ids[:-1]
    metric_positions = tuple(index - 1 for index in target_positions)
    model_input_text = tokenizer.decode(
        list(model_input_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    roundtrip = tuple(tokenizer.encode(model_input_text, add_special_tokens=False))
    if roundtrip != model_input_ids:
        raise ValueError("tokenized circuit model input is not text-roundtrip stable for MIB")
    return model_input_ids, target_ids, metric_positions, model_input_text


def tokenize_probe_specs(
    semantic_manifest: dict[str, Any],
    tokenizer: Any,
    *,
    tokenizer_id: str,
    tokenizer_revision: str,
) -> list[CircuitProbeSpec]:
    content = {key: value for key, value in semantic_manifest.items() if key != "sha256"}
    if semantic_manifest.get("sha256") != sha256_value(content):
        raise ValueError("semantic circuit probe manifest hash mismatch")
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    tokenized = []
    for row in semantic_manifest["probes"]:
        semantic = SemanticCircuitProbeSpec(**row)
        clean_ids, clean_targets, clean_positions, clean_text = _teacher_forced_encoding(
            tokenizer, semantic.clean_context, semantic.clean_target
        )
        corrupt_ids, corrupt_targets, corrupt_positions, corrupt_text = _teacher_forced_encoding(
            tokenizer, semantic.corrupt_context, semantic.corrupt_target
        )
        clean_query_spans = _token_spans(tokenizer, semantic.clean_context, (semantic.query_text,))
        corrupt_query_spans = _token_spans(tokenizer, semantic.corrupt_context, (semantic.query_text,))
        clean_identifier_spans = _token_spans(
            tokenizer, semantic.clean_context, semantic.fact_ids + semantic.rule_ids
        )
        corrupt_identifier_spans = _token_spans(
            tokenizer, semantic.corrupt_context, semantic.fact_ids + semantic.rule_ids
        )
        clean_identifier_positions = tuple(
            sorted({index for span in clean_identifier_spans for index in span})
        )
        corrupt_identifier_positions = tuple(
            sorted({index for span in corrupt_identifier_spans for index in span})
        )
        checks = (
            ("model_input_length_equal", len(clean_ids) == len(corrupt_ids)),
            ("target_sequence_length_equal", len(clean_targets) == len(corrupt_targets)),
            ("query_spans_aligned", clean_query_spans == corrupt_query_spans),
            ("identifier_spans_aligned", clean_identifier_positions == corrupt_identifier_positions),
            ("metric_positions_aligned", clean_positions == corrupt_positions),
            (
                "output_prefix_shape_aligned",
                len(tokenizer.encode(semantic.clean_context, add_special_tokens=False))
                == len(tokenizer.encode(semantic.corrupt_context, add_special_tokens=False)),
            ),
        )
        base = {
            "probe_id": semantic.probe_id,
            "semantic_pair_group_id": semantic.semantic_pair_group_id,
            "subset": semantic.subset,
            "stage": semantic.stage,
            "clean_context": semantic.clean_context,
            "corrupt_context": semantic.corrupt_context,
            "clean_model_input": clean_text,
            "corrupt_model_input": corrupt_text,
            "clean_target": semantic.clean_target,
            "corrupt_target": semantic.corrupt_target,
            "clean_input_ids": clean_ids,
            "corrupt_input_ids": corrupt_ids,
            "clean_target_ids": clean_targets,
            "corrupt_target_ids": corrupt_targets,
            "clean_metric_positions": clean_positions,
            "corrupt_metric_positions": corrupt_positions,
            "clean_intervention_positions": tuple(range(len(clean_ids))),
            "corrupt_intervention_positions": tuple(range(len(corrupt_ids))),
            "corruption_type": semantic.corruption_type,
            "corruption_class": semantic.corruption_class,
            "changed_semantic_field": semantic.changed_semantic_field,
            "source_example_ids": semantic.source_example_ids,
            "semantic_pair_hash": semantic.semantic_pair_hash,
            "semantic_manifest_hash": semantic_manifest["sha256"],
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "tokenizer_hash": tokenizer_hash,
            "alignment_checks": checks,
        }
        pair_hash = sha256_value(base)
        tokenized.append(CircuitProbeSpec(**base, tokenized_pair_hash=pair_hash))
    return tokenized


def tokenized_probe_manifest(
    specs: list[CircuitProbeSpec],
    *,
    semantic_manifest_hash: str,
) -> dict[str, Any]:
    if not specs:
        raise ValueError("tokenized circuit probe manifest cannot be empty")
    if any(spec.semantic_manifest_hash != semantic_manifest_hash for spec in specs):
        raise ValueError("tokenized probes disagree on semantic manifest")
    rows = [asdict(spec) for spec in specs]
    content = {
        "format_version": 1,
        "circuit_probe_schema_version": CIRCUIT_PROBE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "prereg_version": "core_v2",
        "semantic_manifest_hash": semantic_manifest_hash,
        "tokenizer_id": specs[0].tokenizer_id,
        "tokenizer_revision": specs[0].tokenizer_revision,
        "tokenizer_hash": specs[0].tokenizer_hash,
        "probe_count": len(rows),
        "stages": list(PROBE_STAGES),
        "subsets": sorted({spec.subset for spec in specs}),
        "probes": rows,
    }
    return {**content, "sha256": sha256_value(content)}


def sequence_log_probability(
    logits: torch.Tensor,
    target_ids: tuple[int, ...],
    metric_positions: tuple[int, ...],
) -> torch.Tensor:
    """Teacher-forced sequence log probability at explicit next-token positions."""

    if logits.ndim != 3:
        raise ValueError("sequence metric expects [batch, sequence, vocabulary] logits")
    if not target_ids or len(target_ids) != len(metric_positions):
        raise ValueError("target IDs and metric positions must be aligned and nonempty")
    positions = torch.tensor(metric_positions, dtype=torch.long, device=logits.device)
    targets = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
    if int(positions.min()) < 0 or int(positions.max()) >= logits.shape[1]:
        raise ValueError("sequence metric position is outside model logits")
    selected = logits.index_select(1, positions).log_softmax(dim=-1)
    gathered = selected.gather(
        -1,
        targets.view(1, -1, 1).expand(logits.shape[0], -1, 1),
    ).squeeze(-1)
    return gathered.sum(dim=-1)


def target_sequence_contrast(
    logits: torch.Tensor,
    probe: CircuitProbeSpec,
    *,
    side: str,
    mean: bool = True,
) -> torch.Tensor:
    if side == "clean":
        positions = probe.clean_metric_positions
    elif side == "corrupt":
        positions = probe.corrupt_metric_positions
    else:
        raise ValueError("sequence contrast side must be clean or corrupt")
    clean = sequence_log_probability(logits, probe.clean_target_ids, positions)
    corrupt = sequence_log_probability(logits, probe.corrupt_target_ids, positions)
    result = clean - corrupt
    return result.mean() if mean else result


class TargetSequenceMetric:
    """Marker metric evaluated against each pair's explicit frozen targets."""

    def __call__(
        self,
        logits: torch.Tensor,
        *,
        pair: Any,
        side: str,
    ) -> torch.Tensor:
        positions = (
            tuple(pair.clean_metric_positions) if side == "clean" else tuple(pair.corrupt_metric_positions)
        )
        clean = sequence_log_probability(logits, tuple(pair.clean_target_ids), positions)
        corrupt = sequence_log_probability(logits, tuple(pair.corrupt_target_ids), positions)
        return (clean - corrupt).mean()
