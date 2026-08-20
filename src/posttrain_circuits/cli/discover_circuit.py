"""Discover stage-specific circuits from frozen signed support-swap probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.circuits.exact_patching import ExactTokenPair
from posttrain_circuits.circuits.graph import CircuitArtifact
from posttrain_circuits.circuits.mib_eap_ig import MibEapIgAdapter, write_fixed_discovery_pairs
from posttrain_circuits.circuits.model_adapter import check_hf_identity_compatibility
from posttrain_circuits.circuits.probe_cohorts import load_probe_examples
from posttrain_circuits.circuits.probes import (
    CIRCUIT_PROBE_SCHEMA_VERSION,
    PRIMARY_CORRUPTION,
    PROBE_STAGES,
    CircuitProbeSpec,
    TargetSequenceMetric,
    build_semantic_probe_specs,
    semantic_probe_manifest,
    tokenize_probe_specs,
    tokenized_probe_manifest,
)
from posttrain_circuits.circuits.tiny_eap_ig import TinyEapIgBackend
from posttrain_circuits.cli._common import enforce_production_guard, print_json
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import formal_artifact_binding
from posttrain_circuits.core.types import CounterfactualPair
from posttrain_circuits.data.splits import deserialize_example
from posttrain_circuits.tasks.proofgraph.generator import (
    GENERATOR_VERSION,
    LABEL_SEMANTICS,
    ProofGraphTask,
)
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


def _padded_pair(tokenizer: Any, clean: str, corrupt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy helper retained for callers, now fail-closed instead of padding."""

    clean_ids = tokenizer.encode(clean, add_special_tokens=False)
    corrupt_ids = tokenizer.encode(corrupt, add_special_tokens=False)
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError("circuit pairs must be semantically token-aligned; padding is forbidden")
    return torch.tensor([clean_ids]), torch.tensor([corrupt_ids])


def _unique_pair_examples(examples: list[TaskExample], count: int) -> list[TaskExample]:
    selected = []
    seen = set()
    for example in examples:
        if not example.pair_group_id or example.pair_group_id in seen:
            continue
        selected.append(example)
        seen.add(example.pair_group_id)
        if len(selected) == count:
            return selected
    raise ValueError(f"frozen cohort has fewer than {count} distinct semantic pairs")


def _all_unique_pair_examples(examples: list[TaskExample]) -> list[TaskExample]:
    selected = []
    seen = set()
    for example in examples:
        if not example.pair_group_id or example.pair_group_id in seen:
            continue
        selected.append(example)
        seen.add(example.pair_group_id)
    return selected


def _support_swap_pairs(
    task: ProofGraphTask,
    examples: list[TaskExample],
    *,
    seed: int,
) -> list[CounterfactualPair]:
    return [
        task.make_counterfactual(example, PRIMARY_CORRUPTION, seed + index)
        for index, example in enumerate(examples)
    ]


def _fixed_pairs(
    task: ProofGraphTask,
    *,
    count: int,
    seed: int,
    task_config: dict[str, Any],
) -> list[CounterfactualPair]:
    if count < 2:
        raise ValueError("circuit discovery needs at least two pairs")
    difficulty = {
        "depth_range": task_config["depth_range"],
        "distractor_range": task_config["distractor_range"],
        "structures": task_config["structures"],
        "unique_proof": True,
        "multiple_valid_proofs": False,
        "label_semantics": LABEL_SEMANTICS,
        "paired_generation": True,
        "require_exactly_one_query_polarity": True,
    }
    examples = [task.generate_pair(seed + index, difficulty)[0] for index in range(count)]
    return _support_swap_pairs(task, examples, seed=seed + 100_000)


def _select_tokenizer_aligned_pairs(
    pairs: list[CounterfactualPair],
    *,
    count: int,
    subset: str,
    tokenizer: Any,
    tokenizer_id: str,
    tokenizer_revision: str,
    model_config: dict[str, Any] | None = None,
) -> tuple[list[CounterfactualPair], dict[str, Any]]:
    """Deterministically skip tokenizer-incompatible pairs within a frozen candidate order."""

    selected = []
    rejected = []
    for pair in pairs:
        try:
            candidate_manifest = semantic_probe_manifest(build_semantic_probe_specs([pair], subset=subset))
            tokenize_probe_specs(
                candidate_manifest,
                tokenizer,
                tokenizer_id=tokenizer_id,
                tokenizer_revision=tokenizer_revision,
                model_config=model_config,
            )
        except ValueError as error:
            rejected.append(
                {
                    "pair_group_id": pair.clean_example.pair_group_id,
                    "source_example_ids": [
                        pair.clean_example.example_id,
                        pair.corrupt_example.example_id,
                    ],
                    "reason": str(error),
                }
            )
            continue
        selected.append(pair)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"{subset} has {len(selected)} tokenizer-aligned pairs; requires {count}; rejected={rejected}"
        )
    audit = {
        "selection": "frozen_candidate_order_first_tokenizer_aligned",
        "required_pair_count": count,
        "candidate_pair_count": len(pairs),
        "selected_pair_group_ids": [pair.clean_example.pair_group_id for pair in selected],
        "rejected": rejected,
    }
    audit["sha256"] = sha256_value(audit)
    return selected, audit


def _pair_rows(probes: list[CircuitProbeSpec]) -> list[dict[str, Any]]:
    rows = []
    for probe in probes:
        rows.append(
            {
                "pair_id": probe.probe_id,
                "clean_prompt": probe.clean_model_input,
                "corrupt_prompt": probe.corrupt_model_input,
                "clean_target": probe.clean_target,
                "corrupt_target": probe.corrupt_target,
                "clean_input_ids": list(probe.clean_input_ids),
                "corrupt_input_ids": list(probe.corrupt_input_ids),
                "clean_target_ids": list(probe.clean_target_ids),
                "corrupt_target_ids": list(probe.corrupt_target_ids),
                "clean_metric_positions": list(probe.clean_metric_positions),
                "corrupt_metric_positions": list(probe.corrupt_metric_positions),
                "clean_intervention_positions": list(probe.clean_intervention_positions),
                "corrupt_intervention_positions": list(probe.corrupt_intervention_positions),
                "stage": probe.stage,
                "semantic_pair_hash": probe.semantic_pair_hash,
                "tokenized_pair_hash": probe.tokenized_pair_hash,
                "semantic_manifest_hash": probe.semantic_manifest_hash,
                "tokenizer_hash": probe.tokenizer_hash,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Discover a stage-specific circuit")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--probe-cohort-manifest", type=Path)
    parser.add_argument("--cohort", choices=("base_capable", "challenge"))
    parser.add_argument("--stage", choices=PROBE_STAGES, default="final_answer")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-production", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    output = args.output or Path(config["output_root"]) / "circuits" / "circuit.json"
    if not enforce_production_guard(
        config,
        dry_run=args.dry_run,
        confirm_production=args.confirm_production,
        output=output,
    ):
        return

    task = ProofGraphTask()
    seed = int(config["seed"])
    discovery_count = int(config["circuit"]["discovery_pair_count"])
    validation_count = int(config["circuit"]["validation_pair_count"])
    local_model = str(config["model"]["model_name_or_path"]).startswith("local/")
    probe_manifest: dict[str, Any] | None = None
    if local_model:
        discovery_count = max(2, min(discovery_count, 2))
        validation_count = max(2, min(validation_count, 2))
        tokenizer = build_tiny_tokenizer()
        discovery_candidates = _fixed_pairs(
            task,
            count=max(discovery_count * 8, discovery_count + 16),
            seed=seed,
            task_config=config["task"],
        )
        validation_candidates = _fixed_pairs(
            task,
            count=max(validation_count * 8, validation_count + 16),
            seed=seed + 1_000_000,
            task_config=config["task"],
        )
    else:
        if (
            args.checkpoint is None
            or args.initial_checkpoint is None
            or args.probe_cohort_manifest is None
            or args.cohort is None
        ):
            raise ValueError(
                "production discovery requires checkpoint, initial checkpoint, probe manifest, and cohort"
            )
        discovery_rows, probe_manifest = load_probe_examples(
            args.probe_cohort_manifest,
            cohort=args.cohort,
            subset="discovery",
            expected_initial_checkpoint_hash=sha256_file(args.initial_checkpoint),
        )
        validation_rows, validation_manifest = load_probe_examples(
            args.probe_cohort_manifest,
            cohort=args.cohort,
            subset="validation",
            expected_initial_checkpoint_hash=sha256_file(args.initial_checkpoint),
        )
        if validation_manifest["sha256"] != probe_manifest["sha256"]:
            raise ValueError("discovery and validation probes do not share one frozen cohort manifest")
        discovery_examples = _all_unique_pair_examples([deserialize_example(row) for row in discovery_rows])
        validation_examples = _all_unique_pair_examples([deserialize_example(row) for row in validation_rows])
        discovery_candidates = _support_swap_pairs(task, discovery_examples, seed=seed + 100_000)
        validation_candidates = _support_swap_pairs(task, validation_examples, seed=seed + 2_000_000)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(config["model"]["tokenizer_name_or_path"]),
            revision=str(config["model"]["tokenizer_revision"]),
            local_files_only=True,
            trust_remote_code=bool(config["model"]["trust_remote_code"]),
        )

    tokenizer_id = str(config["model"]["tokenizer_name_or_path"])
    tokenizer_revision = str(config["model"]["tokenizer_revision"])
    discovery_pairs, discovery_alignment = _select_tokenizer_aligned_pairs(
        discovery_candidates,
        count=discovery_count,
        subset="discovery",
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        model_config=config["model"],
    )
    validation_pairs, validation_alignment = _select_tokenizer_aligned_pairs(
        validation_candidates,
        count=validation_count,
        subset="validation",
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        model_config=config["model"],
    )

    semantic_specs = [
        *build_semantic_probe_specs(discovery_pairs, subset="discovery"),
        *build_semantic_probe_specs(validation_pairs, subset="validation"),
    ]
    semantic_manifest = semantic_probe_manifest(semantic_specs)
    semantic_content = {key: value for key, value in semantic_manifest.items() if key != "sha256"}
    semantic_content["tokenizer_alignment_selection"] = {
        "discovery": discovery_alignment,
        "validation": validation_alignment,
    }
    semantic_manifest = {**semantic_content, "sha256": sha256_value(semantic_content)}
    tokenized_specs = tokenize_probe_specs(
        semantic_manifest,
        tokenizer,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        model_config=config["model"],
    )
    tokenized_manifest = tokenized_probe_manifest(
        tokenized_specs, semantic_manifest_hash=str(semantic_manifest["sha256"])
    )
    semantic_path = output.with_name("semantic_probe_manifest.json")
    tokenized_path = output.with_name("tokenized_probe_manifest.json")
    atomic_write_json(semantic_path, semantic_manifest)
    atomic_write_json(tokenized_path, tokenized_manifest)
    selected = [
        probe for probe in tokenized_specs if probe.subset == "discovery" and probe.stage == args.stage
    ]
    if len(selected) != discovery_count:
        raise RuntimeError("stage-specific discovery probe count mismatch")
    pair_manifest_path = output.with_name("fixed_discovery_pairs.json")
    pair_manifest = write_fixed_discovery_pairs(
        pair_manifest_path,
        _pair_rows(selected),
        metadata={
            "probe_stage": args.stage,
            "semantic_probe_manifest_hash": semantic_manifest["sha256"],
            "tokenized_probe_manifest_hash": tokenized_manifest["sha256"],
            "probe_cohort": args.cohort,
            "probe_subset": "discovery",
            "probe_cohort_manifest_hash": probe_manifest["sha256"] if probe_manifest else "tiny",
        },
    )

    artifact_common = {
        "circuit_probe_schema_version": CIRCUIT_PROBE_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "probe_stage": args.stage,
        "semantic_probe_manifest_hash": semantic_manifest["sha256"],
        "tokenized_probe_manifest_hash": tokenized_manifest["sha256"],
        "stage_target_manifest_hash": sha256_value(
            [
                {
                    "probe_id": probe.probe_id,
                    "stage": probe.stage,
                    "clean_target_ids": probe.clean_target_ids,
                    "corrupt_target_ids": probe.corrupt_target_ids,
                    "clean_metric_positions": probe.clean_metric_positions,
                    "corrupt_metric_positions": probe.corrupt_metric_positions,
                }
                for probe in selected
            ]
        ),
        "semantic_probe_manifest_path": str(semantic_path.resolve()),
        "tokenized_probe_manifest_path": str(tokenized_path.resolve()),
        "semantic_pair_hashes": [probe.semantic_pair_hash for probe in selected],
        "tokenized_pair_hashes": [probe.tokenized_pair_hash for probe in selected],
        **formal_artifact_binding(config),
    }
    if not local_model:
        assert args.checkpoint is not None and probe_manifest is not None and args.cohort is not None
        checkpoint_sha256 = sha256_file(args.checkpoint)
        adapter = MibEapIgAdapter()
        scores = adapter.run(
            model=str(config["model"]["model_name_or_path"]),
            model_revision=str(config["model"]["model_revision"]),
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            level=str(config["circuit"]["node_or_edge_level"]),
            steps=int(config["circuit"]["integrated_gradient_steps"]),
            pair_manifest=pair_manifest_path,
            output_dir=output.parent / "mib_raw",
            bootstrap_replicates=int(config["circuit"]["mib_bootstrap_replicates"]),
            seed=seed,
            parity_tolerance=float(config["circuit"]["compatibility_tolerance"]),
        )
        compatibility_path = output.parent / "mib_raw" / "compatibility.json"
        compatibility_payload = json.loads(compatibility_path.read_text(encoding="utf-8"))
        compatibility_payload.update(formal_artifact_binding(config))
        compatibility_payload["sha256"] = sha256_value(
            {
                key: value
                for key, value in compatibility_payload.items()
                if key not in {"sha256", "hf_identity_max_error"}
            }
        )
        atomic_write_json(compatibility_path, compatibility_payload)
        scores.metadata["compatibility_hash"] = compatibility_payload["sha256"]
        raw_bootstrap_vectors = scores.metadata.get("bootstrap_score_vectors", [])
        if not isinstance(raw_bootstrap_vectors, list):
            raise RuntimeError("MIB bootstrap score vectors are malformed")
        bootstrap_vectors = [
            {str(name): float(value) for name, value in vector.items()}
            for vector in raw_bootstrap_vectors
            if isinstance(vector, dict)
        ]
        if len(bootstrap_vectors) != int(config["circuit"]["mib_bootstrap_replicates"]):
            raise RuntimeError("MIB did not retain every bootstrap score vector")
        if str(scores.metadata.get("checkpoint_sha256")) != checkpoint_sha256:
            raise RuntimeError("MIB circuit result is not bound to the requested checkpoint")
        raw_resample_indices = scores.metadata.get("bootstrap_resample_indices", [])
        raw_graph_hashes = scores.metadata.get("bootstrap_raw_graph_hashes", [])
        if not isinstance(raw_resample_indices, list) or not isinstance(raw_graph_hashes, list):
            raise RuntimeError("MIB bootstrap provenance is malformed")
        artifact = CircuitArtifact(
            run_id=str(config.get("run_id", "production")),
            checkpoint_id=checkpoint_sha256,
            task_manifest_hash=str(probe_manifest["cohorts"][args.cohort]["discovery"]["source_split_hash"]),
            pair_manifest_hash=pair_manifest["sha256"],
            backend_version=adapter.version,
            model_compatibility_hash=str(scores.metadata.get("compatibility_hash", "")),
            node_or_edge_level=str(config["circuit"]["node_or_edge_level"]),
            integrated_gradient_steps=int(config["circuit"]["integrated_gradient_steps"]),
            ablation_baseline=str(config["circuit"]["ablation_baseline"]),
            scores=scores.scores,
            score_uncertainty=scores.uncertainty,
            node_scores=scores.node_scores,
            edge_scores=scores.edge_scores,
            backend_name=adapter.backend_name,
            backend_revision=adapter.backend_revision,
            attribution_method=adapter.method,
            discovery_pair_count=pair_manifest["pair_count"],
            uncertainty_method=str(scores.metadata["uncertainty_method"]),
            bootstrap_score_vectors=bootstrap_vectors,
            bootstrap_resample_indices=[
                list(map(int, row)) for row in raw_resample_indices if isinstance(row, list)
            ],
            bootstrap_raw_graph_hashes=[str(value) for value in raw_graph_hashes],
            primary_raw_graph_hash=str(scores.metadata.get("primary_raw_graph_hash", "")),
            checkpoint_path=str(args.checkpoint.resolve()),
            checkpoint_sha256=checkpoint_sha256,
            base_model_revision=str(config["model"]["model_revision"]),
            resolved_model_commit=str(
                config["model"].get("resolved_model_commit", config["model"]["model_revision"])
            ),
            tokenizer_hash=str(tokenized_manifest["tokenizer_hash"]),
            probe_cohort=args.cohort,
            probe_subset="discovery",
            probe_cohort_manifest_hash=str(probe_manifest["sha256"]),
            graph_convention=str(config["circuit"]["node_or_edge_level"]),
            **artifact_common,
        )
    else:
        model = build_tiny_qwen(seed).eval()
        exact_pairs = [ExactTokenPair.from_probe(probe) for probe in selected]
        compatibility = check_hf_identity_compatibility(
            model,
            exact_pairs[0].clean_ids,
            tolerance=float(config["circuit"]["compatibility_tolerance"]),
        )
        backend = TinyEapIgBackend(
            exact_pairs,
            integrated_gradient_steps=int(config["circuit"]["smoke_steps"]),
        )
        scores = backend.score_all_components(model, TargetSequenceMetric())
        artifact = CircuitArtifact(
            run_id="tiny-base",
            checkpoint_id="local-random-v1",
            task_manifest_hash=sha256_value(config["task"]),
            pair_manifest_hash=pair_manifest["sha256"],
            backend_version=backend.version,
            model_compatibility_hash=compatibility.sha256,
            node_or_edge_level="node",
            integrated_gradient_steps=backend.config.integrated_gradient_steps,
            ablation_baseline="counterfactual_replacement",
            scores=scores.scores,
            score_uncertainty=scores.uncertainty,
            node_scores=scores.node_scores,
            edge_scores=scores.edge_scores,
            backend_name="tiny-hf-eap-ig",
            backend_revision="in-repository-v2",
            attribution_method=backend.method,
            discovery_pair_count=pair_manifest["pair_count"],
            uncertainty_method="prompt_standard_error",
            tokenizer_hash=str(tokenized_manifest["tokenizer_hash"]),
            **artifact_common,
        )
        compatibility.write(output.with_name("compatibility.json"))

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact.write(output)
    print_json(
        {
            "output": str(output),
            "backend": artifact.backend_version,
            "probe_stage": artifact.probe_stage,
            "components_scored": len(artifact.scores),
            "semantic_probe_manifest": str(semantic_path),
            "tokenized_probe_manifest": str(tokenized_path),
            "fixed_pairs": str(pair_manifest_path),
        }
    )


if __name__ == "__main__":
    main()
