"""Evaluate circuit faithfulness on held-out counterfactual pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from posttrain_circuits.circuits.exact_patching import (
    ExactPatchingBackend,
    ExactTokenPair,
    normalize_circuit_scores,
)
from posttrain_circuits.circuits.faithfulness import (
    faithfulness_sparsity_curve,
)
from posttrain_circuits.circuits.plots import (
    write_attribution_patching_calibration,
)
from posttrain_circuits.circuits.probe_cohorts import load_probe_examples
from posttrain_circuits.cli._common import (
    dry_run_report,
    print_json,
)
from posttrain_circuits.cli.discover_circuit import (
    _fixed_pairs,
    _padded_pair,
)
from posttrain_circuits.core.config import (
    compose_config,
    is_production_scale,
)
from posttrain_circuits.core.hashing import sha256_file
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.data.splits import deserialize_example
from posttrain_circuits.models.loading import load_model_and_tokenizer
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)


def _load_discovery_pair(
    artifact_path: Path,
    tokenizer: object,
    device: torch.device,
) -> ExactTokenPair:
    pair_path = artifact_path.with_name("fixed_discovery_pairs.json")
    if not pair_path.is_file():
        raise ValueError("circuit evaluation requires the immutable fixed_discovery_pairs.json artifact")
    payload = json.loads(pair_path.read_text(encoding="utf-8"))
    row = payload["pairs"][0]
    clean_ids, corrupt_ids = _padded_pair(
        tokenizer,
        row["clean_prompt"],
        row["corrupt_prompt"],
    )
    return ExactTokenPair(
        row["pair_id"],
        clean_ids.to(device),
        corrupt_ids.to(device),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact held-out circuit faithfulness")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument(
        "--circuit-artifact",
        type=Path,
        required=False,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe-cohort-manifest", type=Path)
    parser.add_argument("--cohort", choices=("base_capable", "challenge"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    output = args.output or (
        args.circuit_artifact.with_name("evaluation.json")
        if args.circuit_artifact is not None
        else Path(config["output_root"]) / "circuits" / "evaluation.json"
    )
    production = is_production_scale(config)
    if args.dry_run:
        dry_run_report(config, output)
        return
    if production and not args.confirm_production:
        raise SystemExit(
            "production circuit evaluation refused: inspect --dry-run, then pass --confirm-production"
        )
    if args.circuit_artifact is None:
        raise ValueError("--circuit-artifact is required outside --dry-run")
    artifact = json.loads(args.circuit_artifact.read_text(encoding="utf-8"))

    model_config = config["model"]
    if str(model_config["model_name_or_path"]).startswith("local/"):
        model = build_tiny_qwen(args.seed).eval()
        tokenizer = build_tiny_tokenizer()
    else:
        loaded = load_model_and_tokenizer(
            model_config,
            for_training=False,
        )
        model = loaded.model
        tokenizer = loaded.tokenizer
    if production and args.checkpoint is None:
        raise ValueError("production exact patching requires --checkpoint")
    if args.checkpoint is not None:
        checkpoint_sha256 = sha256_file(args.checkpoint)
        if checkpoint_sha256 != artifact.get("checkpoint_sha256"):
            raise ValueError("exact patching checkpoint hash differs from circuit discovery")
        payload = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("exact patching checkpoint is incomplete")
        model.load_state_dict(payload["model"], strict=True)
    model.eval()
    device = next(model.parameters()).device
    discovery_pair = _load_discovery_pair(
        args.circuit_artifact,
        tokenizer,
        device,
    )
    backend = ExactPatchingBackend(
        discovery_pair.clean_ids,
        discovery_pair.corrupt_ids,
    )
    raw_scores = {key: float(value) for key, value in artifact["scores"].items()}
    scores, unsupported = normalize_circuit_scores(
        model,
        raw_scores,
    )
    task = ProofGraphTask()
    validation_count = int(config["circuit"]["validation_pair_count"])
    if str(model_config["model_name_or_path"]).startswith("local/"):
        validation_count = max(2, min(validation_count, 2))
    if production:
        if args.probe_cohort_manifest is None or args.cohort is None:
            raise ValueError("production exact patching requires --probe-cohort-manifest and --cohort")
        if args.cohort != artifact.get("probe_cohort"):
            raise ValueError("exact patching cohort differs from discovery cohort")
        rows, probe_manifest = load_probe_examples(
            args.probe_cohort_manifest,
            cohort=args.cohort,
            subset="validation",
            expected_initial_checkpoint_hash=str(model_config["model_revision"]),
        )
        if probe_manifest["sha256"] != artifact.get("probe_cohort_manifest_hash"):
            raise ValueError("exact patching probe manifest differs from discovery")
        if len(rows) < validation_count:
            raise ValueError("frozen probe cohort has fewer validation examples than requested")
        examples = [deserialize_example(row) for row in rows[:validation_count]]
        heldout = [
            task.make_counterfactual(example, "query_flip", args.seed + 1_000_000 + index)
            for index, example in enumerate(examples)
        ]
    else:
        heldout = _fixed_pairs(
            task,
            count=validation_count,
            seed=args.seed + 1_000_000,
            task_config=config["task"],
        )
    validation_pairs = []
    for pair in heldout:
        clean_ids, corrupt_ids = _padded_pair(
            tokenizer,
            pair.clean_prompt,
            pair.corrupt_prompt,
        )
        validation_pairs.append(
            ExactTokenPair(
                pair.pair_id,
                clean_ids.to(device),
                corrupt_ids.to(device),
            )
        )
    answer_one = tokenizer.encode(
        "1",
        add_special_tokens=False,
    )[0]
    answer_zero = tokenizer.encode(
        "0",
        add_special_tokens=False,
    )[0]

    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, answer_one].mean() - logits[:, -1, answer_zero].mean()

    if all("->" in name for name in scores):
        patching_scores = {}
        for name in scores:
            sender, receiver = name.split("->", 1)
            patching_scores.update(
                backend.score_path(
                    model,
                    metric,
                    sender=sender,
                    receiver=receiver,
                ).scores
            )
    elif any("->" in name for name in scores):
        raise ValueError("circuit artifact mixes node and edge score levels")
    else:
        patching_scores = backend.score_all_components(
            model,
            metric,
        ).scores
    evaluation = faithfulness_sparsity_curve(
        backend,
        model,
        scores,
        metric,
        [float(value) for value in config["circuit"]["sparsity_grid"]],
        validation_pairs,
        patching_scores=patching_scores,
        random_seed=args.seed,
        random_repeats=int(config["circuit"]["random_mask_repeats"]),
        bootstrap_samples=int(config["circuit"]["prompt_bootstrap_samples"]),
    )
    evaluation["sanity_checks"] = backend.sanity_checks(
        model,
        validation_pairs[0],
        metric,
        tolerance=1e-5,
    )
    component_groups = {
        (
            "attention"
            if "attention_head" in name
            else "mlp"
            if "mlp" in name
            else "residual"
            if "resid" in name
            else "qkv"
            if any(token in name for token in ("q_head", "k_head", "v_head"))
            else "other"
        )
        for name in scores
    }
    evaluation["functional_groups"] = sorted(component_groups)
    evaluation["functional_group_count"] = len(component_groups)
    calibration_paths = write_attribution_patching_calibration(
        evaluation["calibration"],
        spearman=float(evaluation["attribution_patching_spearman"]),
        output_prefix=output.with_name("attribution_patching_calibration"),
    )
    evaluation["artifacts"] = {
        "calibration_plot": calibration_paths,
        "circuit_artifact": str(args.circuit_artifact),
        "checkpoint": (str(args.checkpoint) if args.checkpoint is not None else None),
        "checkpoint_sha256": artifact.get("checkpoint_sha256"),
        "probe_cohort": artifact.get("probe_cohort"),
        "probe_subset": "validation",
        "probe_cohort_manifest_hash": artifact.get("probe_cohort_manifest_hash"),
        "unsupported_source_components": unsupported,
        "mapped_component_count": len(scores),
    }
    atomic_write_json(output, evaluation)
    print_json(
        {
            "output": str(output),
            "cpr": evaluation["cpr"],
            "cpr_ci": evaluation["cpr_ci"],
            "cmd": evaluation["cmd"],
            "cmd_ci": evaluation["cmd_ci"],
            "attribution_patching_spearman": evaluation["attribution_patching_spearman"],
            "calibration_plot": calibration_paths,
            "backend_version": artifact["backend_version"],
        }
    )


if __name__ == "__main__":
    main()
