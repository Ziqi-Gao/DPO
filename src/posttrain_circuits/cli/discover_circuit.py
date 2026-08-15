"""Discover production MIB EAP-IG or genuine tiny CPU EAP-IG circuits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.circuits.exact_patching import ExactTokenPair
from posttrain_circuits.circuits.graph import CircuitArtifact
from posttrain_circuits.circuits.mib_eap_ig import (
    MibEapIgAdapter,
    write_fixed_discovery_pairs,
)
from posttrain_circuits.circuits.model_adapter import (
    check_hf_identity_compatibility,
)
from posttrain_circuits.circuits.probe_cohorts import load_probe_examples
from posttrain_circuits.circuits.tiny_eap_ig import TinyEapIgBackend
from posttrain_circuits.cli._common import enforce_production_guard, print_json
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.types import CounterfactualPair
from posttrain_circuits.data.splits import deserialize_example
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.tokenization import (
    semantic_token_indices,
    tokenization_audit,
    write_tokenization_audit,
)
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)


def _padded_pair(
    tokenizer: Any,
    clean: str,
    corrupt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [clean, corrupt],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return encoded.input_ids[:1], encoded.input_ids[1:]


def _fixed_pairs(
    task: ProofGraphTask,
    *,
    count: int,
    seed: int,
    task_config: dict[str, Any],
) -> list[CounterfactualPair]:
    if count < 2:
        raise ValueError("circuit discovery needs at least two pairs")
    pairs = []
    difficulty = {
        "depth_range": task_config["depth_range"],
        "distractor_range": task_config["distractor_range"],
        "structures": task_config["structures"],
        "positive": True,
        "unique_proof": True,
        "multiple_valid_proofs": False,
    }
    for index in range(count):
        example = task.generate(seed + index, difficulty)
        pairs.append(
            task.make_counterfactual(
                example,
                "query_flip",
                seed + 100_000 + index,
            )
        )
    return pairs


def _pair_rows(
    pairs: list[CounterfactualPair],
) -> list[dict[str, str]]:
    return [
        {
            "pair_id": pair.pair_id,
            "clean_prompt": pair.clean_prompt,
            "corrupt_prompt": pair.corrupt_prompt,
            "clean_target": str(pair.clean_example.label),
            "corrupt_target": str(pair.corrupt_example.label),
        }
        for pair in pairs
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Discover a circuit")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe-cohort-manifest", type=Path)
    parser.add_argument("--cohort", choices=("base_capable", "challenge"))
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
    pair_count = int(config["circuit"]["discovery_pair_count"])
    if str(config["model"]["model_name_or_path"]).startswith("local/"):
        pair_count = max(2, min(pair_count, 2))
    local_model = str(config["model"]["model_name_or_path"]).startswith("local/")
    probe_manifest: dict[str, Any] | None = None
    if local_model:
        pairs = _fixed_pairs(
            task,
            count=pair_count,
            seed=seed,
            task_config=config["task"],
        )
    else:
        if args.checkpoint is None or args.probe_cohort_manifest is None or args.cohort is None:
            raise ValueError(
                "production circuit discovery requires --checkpoint, --probe-cohort-manifest, and --cohort"
            )
        rows, probe_manifest = load_probe_examples(
            args.probe_cohort_manifest,
            cohort=args.cohort,
            subset="discovery",
            expected_initial_checkpoint_hash=str(config["model"]["model_revision"]),
        )
        if len(rows) < pair_count:
            raise ValueError("frozen probe cohort has fewer discovery examples than requested")
        examples = [deserialize_example(row) for row in rows[:pair_count]]
        pairs = [
            task.make_counterfactual(example, "query_flip", seed + 100_000 + index)
            for index, example in enumerate(examples)
        ]
    pair_manifest_path = output.with_name("fixed_discovery_pairs.json")
    pair_manifest = write_fixed_discovery_pairs(
        pair_manifest_path,
        _pair_rows(pairs),
        metadata=(
            {
                "probe_cohort": args.cohort,
                "probe_subset": "discovery",
                "probe_cohort_manifest_hash": probe_manifest["sha256"],
                "initial_student_checkpoint_hash": probe_manifest["initial_student_checkpoint_hash"],
            }
            if probe_manifest is not None
            else {"mode": "tiny_smoke"}
        ),
    )

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
        compatibility_hash = str(scores.metadata.get("compatibility_hash", ""))
        if not compatibility_hash:
            raise RuntimeError("MIB runner returned no compatibility evidence")
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
        raw_indices = scores.metadata.get("bootstrap_resample_indices", [])
        raw_graph_hashes = scores.metadata.get("bootstrap_raw_graph_hashes", [])
        if not isinstance(raw_indices, list) or not isinstance(raw_graph_hashes, list):
            raise RuntimeError("MIB bootstrap provenance is malformed")
        if str(scores.metadata.get("checkpoint_sha256")) != checkpoint_sha256:
            raise RuntimeError("MIB circuit result is not bound to the requested checkpoint")
        artifact = CircuitArtifact(
            run_id=str(config.get("run_id", "production")),
            checkpoint_id=checkpoint_sha256,
            task_manifest_hash=str(probe_manifest["cohorts"][args.cohort]["discovery"]["source_split_hash"]),
            pair_manifest_hash=pair_manifest["sha256"],
            backend_version=adapter.version,
            model_compatibility_hash=compatibility_hash,
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
            bootstrap_resample_indices=[list(map(int, row)) for row in raw_indices],
            bootstrap_raw_graph_hashes=[str(value) for value in raw_graph_hashes],
            primary_raw_graph_hash=str(scores.metadata.get("primary_raw_graph_hash", "")),
            checkpoint_path=str(args.checkpoint.resolve()),
            checkpoint_sha256=checkpoint_sha256,
            base_model_revision=str(config["model"]["model_revision"]),
            resolved_model_commit=str(
                config["model"].get("resolved_model_commit", config["model"]["model_revision"])
            ),
            tokenizer_revision=str(config["model"]["tokenizer_revision"]),
            tokenizer_hash=str(scores.metadata.get("tokenizer_hash", "")),
            probe_cohort=args.cohort,
            probe_subset="discovery",
            probe_cohort_manifest_hash=str(probe_manifest["sha256"]),
            graph_convention=str(config["circuit"]["node_or_edge_level"]),
        )
        artifact.write(output)
        print_json(
            {
                "output": str(output),
                "backend": artifact.backend_version,
                "components_scored": len(scores.scores),
                "fixed_pairs": str(pair_manifest_path),
                "executed": True,
            }
        )
        return

    model = build_tiny_qwen(seed).eval()
    tokenizer = build_tiny_tokenizer()
    first_example = pairs[0].clean_example
    audit = tokenization_audit(
        tokenizer,
        pairs[0].clean_prompt,
        {
            "facts": [str(value) for value in first_example.facts.values()],
            "rules": [str(rule.consequent) for rule in first_example.rules.values()],
            "query": [str(first_example.query)],
            "identifiers": [
                *first_example.facts.keys(),
                *first_example.rules.keys(),
            ],
        },
        model_family=str(config["model"]["model_name_or_path"]),
    )
    circuit_metric_positions = semantic_token_indices(
        audit,
        ["query", "identifiers"],
    )
    write_tokenization_audit(
        output.with_name("tokenization_audit.json"),
        audit,
    )
    exact_pairs = []
    for pair in pairs:
        clean_ids, corrupt_ids = _padded_pair(
            tokenizer,
            pair.clean_prompt,
            pair.corrupt_prompt,
        )
        exact_pairs.append(
            ExactTokenPair(
                pair.pair_id,
                clean_ids,
                corrupt_ids,
            )
        )
    compatibility = check_hf_identity_compatibility(
        model,
        exact_pairs[0].clean_ids,
        tolerance=float(config["circuit"]["compatibility_tolerance"]),
    )
    backend = TinyEapIgBackend(
        exact_pairs,
        integrated_gradient_steps=int(config["circuit"]["smoke_steps"]),
    )
    clean_answer_id = tokenizer.encode(
        "1",
        add_special_tokens=False,
    )[0]
    corrupt_answer_id = tokenizer.encode(
        "0",
        add_special_tokens=False,
    )[0]

    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1, clean_answer_id].mean() - logits[:, -1, corrupt_answer_id].mean()

    scores = backend.score_all_components(model, metric)
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
        backend_revision="in-repository-v1",
        attribution_method=backend.method,
        discovery_pair_count=pair_manifest["pair_count"],
        uncertainty_method="prompt_standard_error",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact.write(output)
    compatibility.write(output.with_name("compatibility.json"))
    print_json(
        {
            "output": str(output),
            "backend": backend.version,
            "components_scored": len(scores.scores),
            "fixed_pairs": str(pair_manifest_path),
            "tokenization_audit": str(output.with_name("tokenization_audit.json")),
            "circuit_metric_positions": circuit_metric_positions,
            "note": "CPU smoke ran genuine activation-space EAP-IG.",
        }
    )


if __name__ == "__main__":
    main()
