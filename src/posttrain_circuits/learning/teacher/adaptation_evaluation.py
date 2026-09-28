"""Measured development scores for a separately identified dense teacher.

The adapted teacher has an explicitly proposed 256-token evaluation budget.
This module does not alter the original student's or frozen teacher's budget,
and a development metric PASS is not formal scientific acceptance.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from posttrain_circuits.causal_circuits.metrics.probes import (
    build_semantic_probe_specs,
    semantic_probe_manifest,
    tokenize_probe_specs,
    tokenized_probe_manifest,
)
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.learning.teacher.evaluation import (
    TeacherPrefixScore,
    TeacherReadinessThresholds,
    summarize_teacher_scores,
)

ADAPTED_EVALUATION_MAX_NEW_TOKENS = 256


def build_adaptation_probes(examples: list[Any], loaded: Any, teacher: dict[str, Any]):
    """Preserve original pair selection/alignment; bound the actual population."""
    if not 0 < len(examples) <= 512 or len({e.example_id for e in examples}) != len(examples):
        raise ValueError("invalid or duplicate adapted-teacher evaluation population")
    task = ProofGraphTask()

    def tokenize(semantic):
        return tokenize_probe_specs(
            semantic,
            loaded.tokenizer,
            tokenizer_id=loaded.tokenizer_id,
            tokenizer_revision=loaded.requested_tokenizer_revision,
            model_config=teacher,
        )

    pairs, skipped, seen = [], [], set()
    for example in examples:
        if example.pair_group_id in seen or len(example.canonical_proof) < 2:
            continue
        pair = task.make_counterfactual(example, "active_support_path_swap", 42)
        try:
            tokenize(semantic_probe_manifest(build_semantic_probe_specs([pair], subset="validation")))
        except ValueError as error:
            skipped.append({"example_id": example.example_id, "reason": str(error)})
            continue
        pairs.append(pair)
        seen.add(example.pair_group_id)
    if not pairs:
        raise ValueError("no tokenizer-aligned depth>1 teacher prefix probes were available")
    semantic = semantic_probe_manifest(build_semantic_probe_specs(pairs, subset="validation"))
    tokenized = tokenize(semantic)
    if not 0 < len(tokenized) <= len(examples) * 3:
        raise ValueError("unexpected adapted-teacher prefix population")
    manifest = tokenized_probe_manifest(tokenized, semantic_manifest_hash=semantic["sha256"])
    return semantic, tokenized, manifest, skipped


def _record(stream, row):
    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def measure_development_shard(
    model: Any,
    loaded: Any,
    teacher: dict[str, Any],
    examples: list[Any],
    specifications: list[Any],
    output: Path,
    *,
    rank: int,
    world_size: int,
    observe: Any,
) -> dict[str, Any]:
    """Measure disjoint shards on the actual merged model; never train on dev."""
    from posttrain_circuits.cli.evaluate_teacher_readiness import _prefix_scores
    from posttrain_circuits.learning.teacher.demo_generation import HfTeacherCandidateGenerator

    if world_size not in (1, 4) or not 0 <= rank < world_size:
        raise ValueError("unreviewed development world size or rank")
    output.mkdir(parents=True, exist_ok=False)
    generator = HfTeacherCandidateGenerator(
        model, loaded.tokenizer, max_new_tokens=ADAPTED_EVALUATION_MAX_NEW_TOKENS, model_config=teacher
    )
    generated, scores = {}, []
    with (output / "generations.jsonl").open("x") as stream:
        for index in range(rank, len(examples), world_size):
            example = examples[index]
            observe.phase("merged_dev_generation", rank=rank, index=index, example_id=example.example_id)
            result = generator(
                example=example,
                candidate_index=0,
                actual_sampling_seed=42 + index,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
            )
            result.validate()
            if (
                result.logprob_status != "available"
                or result.response_ids is None
                or len(result.response_ids) > ADAPTED_EVALUATION_MAX_NEW_TOKENS
            ):
                raise ValueError("generation lacks bounded actual response tokens")
            generated[example.example_id] = result.response_text
            _record(
                stream,
                {
                    "example_id": example.example_id,
                    "global_index": index,
                    "actual_sampling_seed": 42 + index,
                    **asdict(result),
                },
            )
    selected = [
        spec for spec in specifications if spec.stage in {"first_rule_selection", "intermediate_conclusion"}
    ]
    with (output / "prefix-scores.jsonl").open("x") as stream:
        for index in range(rank, len(selected), world_size):
            observe.phase("merged_dev_prefix", rank=rank, index=index)
            pair = _prefix_scores(model, selected[index], top_k=128)
            if len(pair) != 2:
                raise ValueError("prefix scorer did not measure both sides")
            for side, value in enumerate(pair):
                row = {"global_index": index, "side": side, "score": asdict(value)}
                scores.append(row)
                _record(stream, row)
    return {"rank": rank, "world_size": world_size, "generated": generated, "scores": scores}


def combine_development_shards(examples: list[Any], specifications: list[Any], shards: list[dict[str, Any]]):
    """Require each generation and each side exactly once before reducing gates."""
    world_size = len(shards)
    if world_size not in (1, 4) or {s["rank"] for s in shards} != set(range(world_size)):
        raise ValueError("missing or duplicate development rank")
    if any(s["world_size"] != world_size for s in shards):
        raise ValueError("mixed development world sizes")
    generated, indexed = {}, {}
    selected = [
        spec for spec in specifications if spec.stage in {"first_rule_selection", "intermediate_conclusion"}
    ]
    for shard in shards:
        expected = {examples[i].example_id for i in range(shard["rank"], len(examples), world_size)}
        if set(shard["generated"]) != expected or set(generated) & expected:
            raise ValueError("development generation shard differs")
        generated.update(shard["generated"])
        for row in shard["scores"]:
            key = (row["global_index"], row["side"])
            if key in indexed or key[0] % world_size != shard["rank"] or key[1] not in (0, 1):
                raise ValueError("duplicate or misassigned development prefix side")
            indexed[key] = row["score"]
    if set(indexed) != {(index, side) for index in range(len(selected)) for side in (0, 1)}:
        raise ValueError("incomplete development prefix scores")
    scores = []
    for key in sorted(indexed):
        value = dict(indexed[key])
        value["target_ids"] = tuple(value["target_ids"])
        spec = selected[key[0]]
        expected_targets = spec.clean_target_ids if key[1] == 0 else spec.corrupt_target_ids
        expected_kind = "canonical" if key[1] == 0 else "corrupted_or_initial_student"
        if (
            value["probe_id"] != spec.probe_id
            or value["stage"] != spec.stage
            or value["prefix_kind"] != expected_kind
            or value["target_ids"] != tuple(expected_targets)
        ):
            raise ValueError("development prefix score does not match its actual probe side")
        if any(
            not math.isfinite(value[name])
            for name in (
                "minimum_topk_mass",
                "target_log_probability",
                "alternative_log_probability",
                "target_logprob_margin",
                "causal_shift_logprob",
            )
        ):
            raise ValueError("development prefix score is nonfinite")
        scores.append(TeacherPrefixScore(**value))
    return {
        **summarize_teacher_scores(examples, generated, scores, TeacherReadinessThresholds()),
        "evaluation_max_new_tokens": ADAPTED_EVALUATION_MAX_NEW_TOKENS,
        "historical_evaluation_max_new_tokens": 128,
        "original_128_token_readiness_pass_claim": False,
        "prefix_scores": [asdict(score) for score in scores],
        "formal_teacher_accepted": False,
        "scope": "independent teacher development, measured on merged dense weights",
    }
