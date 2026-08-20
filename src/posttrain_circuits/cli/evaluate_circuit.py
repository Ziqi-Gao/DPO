"""Evaluate circuit faithfulness on held-out counterfactual pairs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from posttrain_circuits.circuits.cross_mask_transfer import evaluate_mask_transfer
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
from posttrain_circuits.circuits.probes import (
    CIRCUIT_PROBE_SCHEMA_VERSION,
    CircuitProbeSpec,
    TargetSequenceMetric,
)
from posttrain_circuits.cli._common import (
    dry_run_report,
    print_json,
)
from posttrain_circuits.core.config import (
    compose_config,
    is_production_scale,
)
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import formal_artifact_binding
from posttrain_circuits.core.scientific_versions import require_scientific_artifact
from posttrain_circuits.models.loading import (
    load_model_and_tokenizer,
    move_model_to_local_cuda,
    tokenizer_fingerprint,
)
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)


def _load_discovery_pair(
    device: torch.device,
    tokenized: list[CircuitProbeSpec],
    stage: str,
) -> ExactTokenPair:
    probes = [probe for probe in tokenized if probe.subset == "discovery" and probe.stage == stage]
    if not probes:
        raise ValueError("tokenized manifest contains no matching discovery-stage probe")
    return ExactTokenPair.from_probe(probes[0], device=device)


def _load_tokenized_probes(
    artifact: dict[str, object],
    tokenizer: object,
) -> tuple[list[CircuitProbeSpec], dict[str, object]]:
    path = Path(str(artifact.get("tokenized_probe_manifest_path", "")))
    if not path.is_file():
        raise ValueError("exact patching requires the frozen tokenized probe manifest from discovery")
    payload = json.loads(path.read_text(encoding="utf-8"))
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("sha256") != sha256_value(content):
        raise ValueError("tokenized circuit probe manifest hash mismatch")
    if payload.get("sha256") != artifact.get("tokenized_probe_manifest_hash"):
        raise ValueError("exact patching tokenized manifest differs from discovery")
    if payload.get("circuit_probe_schema_version") != CIRCUIT_PROBE_SCHEMA_VERSION:
        raise ValueError("exact patching rejects legacy circuit probe schemas")
    if payload.get("tokenizer_hash") != tokenizer_fingerprint(tokenizer):
        raise ValueError("exact patching tokenizer hash differs from frozen discovery probes")
    probes = [CircuitProbeSpec(**row) for row in payload["probes"]]
    return probes, payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact held-out circuit faithfulness")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument(
        "--circuit-artifact",
        type=Path,
        required=False,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--transfer-source-circuit", type=Path)
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
    require_scientific_artifact(
        artifact,
        expected_prereg_version=str(config["prereg_version"]),
        require_circuit_schema=True,
        require_hash=True,
    )

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
    if production and args.initial_checkpoint is None:
        raise ValueError("production exact patching requires --initial-checkpoint")
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
    if production and torch.cuda.is_available():
        model = move_model_to_local_cuda(model)
    model.eval()
    device = next(model.parameters()).device
    tokenized_probes, tokenized_manifest = _load_tokenized_probes(artifact, tokenizer)
    stage = str(artifact.get("probe_stage", ""))
    discovery_pair = _load_discovery_pair(device, tokenized_probes, stage)
    backend = ExactPatchingBackend(discovery_pair)
    raw_scores = {key: float(value) for key, value in artifact["scores"].items()}
    scores, unsupported = normalize_circuit_scores(
        model,
        raw_scores,
    )
    validation_count = int(config["circuit"]["validation_pair_count"])
    if str(model_config["model_name_or_path"]).startswith("local/"):
        validation_count = max(2, min(validation_count, 2))
    if production:
        if args.probe_cohort_manifest is None or args.cohort is None:
            raise ValueError("production exact patching requires --probe-cohort-manifest and --cohort")
        if args.cohort != artifact.get("probe_cohort"):
            raise ValueError("exact patching cohort differs from discovery cohort")
        _, probe_manifest = load_probe_examples(
            args.probe_cohort_manifest,
            cohort=args.cohort,
            subset="validation",
            expected_initial_checkpoint_hash=sha256_file(args.initial_checkpoint),
        )
        if probe_manifest["sha256"] != artifact.get("probe_cohort_manifest_hash"):
            raise ValueError("exact patching probe manifest differs from discovery")
    validation_probes = [
        probe for probe in tokenized_probes if probe.subset == "validation" and probe.stage == stage
    ]
    if len(validation_probes) < validation_count:
        raise ValueError("frozen tokenized manifest has too few held-out stage probes")
    validation_pairs = [
        ExactTokenPair.from_probe(probe, device=device) for probe in validation_probes[:validation_count]
    ]
    metric = TargetSequenceMetric()

    per_pair_patching_scores = []
    if all("->" in name for name in scores):
        for pair in validation_pairs:
            pair_backend = ExactPatchingBackend(pair)
            row = {}
            for name in scores:
                sender, receiver = name.split("->", 1)
                row.update(
                    pair_backend.score_path(
                        model,
                        metric,
                        sender=sender,
                        receiver=receiver,
                    ).scores
                )
            per_pair_patching_scores.append(row)
    elif any("->" in name for name in scores):
        raise ValueError("circuit artifact mixes node and edge score levels")
    else:
        per_pair_patching_scores = [
            ExactPatchingBackend(pair).score_all_components(model, metric).scores for pair in validation_pairs
        ]
    patching_scores = {
        name: sum(row[name] for row in per_pair_patching_scores) / len(per_pair_patching_scores)
        for name in scores
    }
    evaluation = faithfulness_sparsity_curve(
        backend,
        model,
        scores,
        metric,
        [float(value) for value in config["circuit"]["sparsity_grid"]],
        validation_pairs,
        patching_scores=patching_scores,
        patching_scores_per_pair=per_pair_patching_scores,
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
    evaluation["functional_evidence"] = {
        "probe_stage": stage,
        "semantic_probe_manifest_hash": artifact.get("semantic_probe_manifest_hash"),
        "tokenized_probe_manifest_hash": tokenized_manifest["sha256"],
        "stage_target_manifest_hash": artifact.get("stage_target_manifest_hash"),
        "selected_vs_matched_random_cpr_margin": evaluation.get("selected_vs_matched_random_cpr_margin"),
    }
    if args.transfer_source_circuit is not None:
        source_artifact = json.loads(args.transfer_source_circuit.read_text(encoding="utf-8"))
        require_scientific_artifact(
            source_artifact,
            expected_prereg_version=str(config["prereg_version"]),
            require_circuit_schema=True,
            require_hash=True,
        )
        if source_artifact.get("probe_cohort_manifest_hash") != artifact.get(
            "probe_cohort_manifest_hash"
        ) or source_artifact.get("probe_cohort") != artifact.get("probe_cohort"):
            raise ValueError("mask-transfer source uses a different frozen probe cohort")
        if source_artifact.get("graph_convention") != artifact.get("graph_convention"):
            raise ValueError("mask-transfer source uses a different graph convention")
        source_scores, source_unsupported = normalize_circuit_scores(
            model,
            {key: float(value) for key, value in source_artifact["scores"].items()},
        )
        if source_unsupported:
            raise ValueError(
                f"mask-transfer source contains unsupported target components: {source_unsupported}"
            )
        evaluation["cross_checkpoint_mask_transfer"] = [
            asdict(
                evaluate_mask_transfer(
                    backend=backend,
                    target_model=model,
                    validation_pairs=validation_pairs,
                    metric=metric,
                    source_scores=source_scores,
                    sparsity=float(sparsity),
                    source_run=str(source_artifact.get("run_id", "source")),
                    source_checkpoint=str(source_artifact["checkpoint_sha256"]),
                    source_method=str(source_artifact.get("attribution_method", "EAP-IG")),
                    target_run=str(artifact.get("run_id", "target")),
                    target_checkpoint=str(artifact["checkpoint_sha256"]),
                    target_method=str(artifact.get("attribution_method", "EAP-IG")),
                )
            )
            for sparsity in config["circuit"]["sparsity_grid"]
        ]
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
        "probe_stage": stage,
        "circuit_probe_schema_version": artifact.get("circuit_probe_schema_version"),
        "semantic_probe_manifest_hash": artifact.get("semantic_probe_manifest_hash"),
        "tokenized_probe_manifest_hash": tokenized_manifest["sha256"],
        "stage_target_manifest_hash": artifact.get("stage_target_manifest_hash"),
        "semantic_pair_hashes": artifact.get("semantic_pair_hashes"),
        "tokenized_pair_hashes": [probe.tokenized_pair_hash for probe in validation_probes],
        "target_strings": [[probe.clean_target, probe.corrupt_target] for probe in validation_probes],
        "target_token_ids": [
            [list(probe.clean_target_ids), list(probe.corrupt_target_ids)] for probe in validation_probes
        ],
        "target_metric_positions": [
            [list(probe.clean_metric_positions), list(probe.corrupt_metric_positions)]
            for probe in validation_probes
        ],
        "unsupported_source_components": unsupported,
        "mapped_component_count": len(scores),
    }
    evaluation.update(
        {
            "format_version": 2,
            "prereg_version": artifact.get("prereg_version"),
            "generator_version": artifact.get("generator_version"),
            "label_semantics": artifact.get("label_semantics"),
            "circuit_probe_schema_version": artifact.get("circuit_probe_schema_version"),
            "probe_stage": stage,
            "checkpoint_sha256": artifact.get("checkpoint_sha256"),
            "probe_cohort": artifact.get("probe_cohort"),
            "probe_cohort_manifest_hash": artifact.get("probe_cohort_manifest_hash"),
            "semantic_probe_manifest_hash": artifact.get("semantic_probe_manifest_hash"),
            "tokenized_probe_manifest_hash": tokenized_manifest["sha256"],
            "stage_target_manifest_hash": artifact.get("stage_target_manifest_hash"),
        }
    )
    evaluation.update(formal_artifact_binding(config))
    evaluation["sha256"] = sha256_value(evaluation)
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
