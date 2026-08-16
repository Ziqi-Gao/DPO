"""Evaluate frozen-teacher correctness and step targets for the core-v2 gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from posttrain_circuits.circuits.probes import (
    CircuitProbeSpec,
    build_semantic_probe_specs,
    semantic_probe_manifest,
    tokenize_probe_specs,
    tokenized_probe_manifest,
)
from posttrain_circuits.cli._common import dry_run_report, print_json
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import PREREG_PATH, _git
from posttrain_circuits.data.splits import load_frozen_split
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.teacher.demo_generation import HfTeacherCandidateGenerator
from posttrain_circuits.teacher.evaluation import (
    TeacherPrefixScore,
    TeacherReadinessThresholds,
    evaluate_teacher_readiness,
)


def _prefix_score(model: object, spec: CircuitProbeSpec, *, side: str, top_k: int) -> TeacherPrefixScore:
    if side == "clean":
        input_ids = spec.clean_input_ids
        target_ids = spec.clean_target_ids
        positions = spec.clean_metric_positions
        context = spec.clean_context
        prefix_kind = "canonical"
    else:
        input_ids = spec.corrupt_input_ids
        target_ids = spec.corrupt_target_ids
        positions = spec.corrupt_metric_positions
        context = spec.corrupt_context
        prefix_kind = "corrupted_or_initial_student"
    device = next(model.parameters()).device  # type: ignore[attr-defined]
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=tensor).logits[0]  # type: ignore[operator]
    selected = logits[torch.tensor(positions, device=device)].float()
    targets = torch.tensor(target_ids, dtype=torch.long, device=device)
    top1 = bool(torch.equal(selected.argmax(-1), targets))
    width = min(top_k, selected.shape[-1])
    probabilities = selected.softmax(-1)
    top_values, top_ids = probabilities.topk(width, dim=-1)
    coverage = bool((top_ids == targets[:, None]).any(dim=-1).all())
    minimum_mass = float(top_values.sum(-1).min().cpu())
    context_length = (
        len(spec.clean_input_ids if side == "clean" else spec.corrupt_input_ids) - len(target_ids) + 1
    )
    expected_positions = tuple(range(context_length - 1, context_length - 1 + len(target_ids)))
    return TeacherPrefixScore(
        probe_id=spec.probe_id,
        stage=spec.stage,
        prefix_kind=prefix_kind,
        target_ids=tuple(target_ids),
        top1_correct=top1,
        target_in_topk=coverage,
        minimum_topk_mass=minimum_mass,
        causal_shift_valid=tuple(positions) == expected_positions and bool(context),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen teacher readiness")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    if args.dry_run:
        dry_run_report(config, args.output)
        return
    if not args.confirm_production:
        raise SystemExit("teacher readiness is a formal gate; pass --confirm-production")

    examples, dataset_manifest = load_frozen_split(
        args.validation_split,
        expected_split="validation",
    )
    examples = examples[: args.limit]
    loaded = load_model_and_tokenizer(config["teacher"], for_training=False)
    model = move_model_to_local_cuda(loaded.model)
    task = ProofGraphTask()
    generator = HfTeacherCandidateGenerator(
        model,
        loaded.tokenizer,
        max_new_tokens=int(config["trainer"]["max_completion_length"]),
    )
    generated = {
        example.example_id: generator(
            example=example,
            candidate_index=0,
            generation_seed=int(config["seed"]) + index,
            temperature=0.0,
            top_p=1.0,
        )
        for index, example in enumerate(examples)
    }

    pairs = []
    seen_groups = set()
    for example in examples:
        if example.pair_group_id in seen_groups or len(example.canonical_proof) < 2:
            continue
        pair = task.make_counterfactual(example, "active_support_path_swap", int(config["seed"]))
        try:
            trial_semantic = semantic_probe_manifest(build_semantic_probe_specs([pair], subset="validation"))
            tokenize_probe_specs(
                trial_semantic,
                loaded.tokenizer,
                tokenizer_id=loaded.tokenizer_id,
                tokenizer_revision=loaded.requested_tokenizer_revision,
            )
        except ValueError:
            continue
        pairs.append(pair)
        seen_groups.add(example.pair_group_id)
    if not pairs:
        raise ValueError("no tokenizer-aligned depth>1 teacher prefix probes were available")
    semantic = semantic_probe_manifest(build_semantic_probe_specs(pairs, subset="validation"))
    tokenized = tokenize_probe_specs(
        semantic,
        loaded.tokenizer,
        tokenizer_id=loaded.tokenizer_id,
        tokenizer_revision=loaded.requested_tokenizer_revision,
    )
    tokenized_manifest = tokenized_probe_manifest(
        tokenized,
        semantic_manifest_hash=semantic["sha256"],
    )
    prefix_scores = [
        _prefix_score(model, spec, side=side, top_k=args.top_k)
        for spec in tokenized
        if spec.stage in {"first_rule_selection", "intermediate_conclusion"}
        for side in ("clean", "corrupt")
    ]
    threshold_cfg = config["teacher_readiness"]
    thresholds = TeacherReadinessThresholds(
        **{field: float(threshold_cfg[field]) for field in asdict(TeacherReadinessThresholds())}
    )
    bindings = {
        "teacher_model_revision": loaded.resolved_model_commit,
        "tokenizer_revision": loaded.resolved_tokenizer_commit,
        "dataset_hash": str(dataset_manifest["sha256"]),
        "prefix_probe_hash": str(tokenized_manifest["sha256"]),
        "code_commit": _git(["rev-parse", "HEAD"]) or "unavailable",
        "prereg_commit": _git(["log", "-n", "1", "--format=%H", "--", str(PREREG_PATH)]) or "unavailable",
    }
    artifact = evaluate_teacher_readiness(
        examples,
        generated,
        prefix_scores,
        thresholds,
        bindings=bindings,
    )
    artifact["semantic_prefix_manifest"] = semantic
    artifact["tokenized_prefix_manifest"] = tokenized_manifest
    artifact["sha256"] = sha256_value({key: value for key, value in artifact.items() if key != "sha256"})
    atomic_write_json(args.output, artifact)
    print_json({"output": str(args.output), "passed": artifact["passed"]})
    if not artifact["passed"]:
        raise SystemExit("teacher readiness failed")


if __name__ == "__main__":
    main()
