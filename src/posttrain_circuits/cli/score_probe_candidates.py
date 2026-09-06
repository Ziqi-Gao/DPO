"""Score frozen ProofGraph probes under initial and calibration checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.artifacts.compatibility import scientific_compatibility_fields
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_write_json
from posttrain_circuits.artifacts.runs import formal_artifact_binding
from posttrain_circuits.causal_circuits.model.runner import load_checkpoint_into_hf_model
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.cli.factorial_run_validation import (
    validate_factorial_run_artifacts,
)
from posttrain_circuits.datasets.circuit_probes.cohorts import (
    family_probe_pairs,
    flatten_pairs,
    ordered_pair_population,
)
from posttrain_circuits.datasets.circuit_probes.contracts import SOURCE_SPLITS, SUBSETS
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.models.prompt_protocol import format_model_prompt
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.metrics import aggregate_verification
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample, VerificationResult


@torch.no_grad()
def _score_examples(
    model: Any,
    tokenizer: Any,
    examples: list[TaskExample],
    *,
    max_new_tokens: int,
    model_config: dict[str, Any] | None = None,
) -> tuple[dict[str, bool], list[VerificationResult]]:
    task = ProofGraphTask()
    device = next(model.parameters()).device
    results = []
    scores = {}
    model.eval()
    for example in examples:
        prompt = format_model_prompt(task.render(example), tokenizer, model_config)
        ids = tokenizer(
            prompt.model_facing_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(device)
        generated = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        response = tokenizer.decode(generated[0, ids.shape[1] :], skip_special_tokens=True)
        result = task.verify(example, task.parse_response(response))
        scores[example.example_id] = result.reward == 1.0
        results.append(result)
    return scores, results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score frozen probe candidates")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-run-manifest", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--dataset-family", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probe-limit-per-split",
        type=int,
        required=True,
        help="Ordered pair-group count per circuit split.",
    )
    parser.add_argument("--task-validation-limit", type=int, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    formal_binding = formal_artifact_binding(config)
    calibration_artifacts = validate_factorial_run_artifacts(
        args.calibration_run_manifest.parent.absolute(),
        expected_world_size=args.world_size,
        expected_resolved_config=config,
        expected_checkpoint_path=args.calibration_checkpoint,
        expected_code_commit=str(formal_binding["code_commit"]),
        expected_resume_ancestry=(),
    )
    if args.calibration_run_manifest.absolute() != calibration_artifacts.manifest_path:
        raise ValueError("calibration manifest path differs from its strict run root")
    if str(config["model"]["model_name_or_path"]).startswith("local/"):
        raise ValueError("production probe scoring cannot use a tiny model")
    family = load_dataset_family(args.dataset_family)
    discovery_pairs = family_probe_pairs(
        family,
        subset="discovery",
        limit_pairs=args.probe_limit_per_split,
    )
    validation_pairs = family_probe_pairs(
        family,
        subset="validation",
        limit_pairs=args.probe_limit_per_split,
    )
    discovery = flatten_pairs(discovery_pairs)
    validation = flatten_pairs(validation_pairs)
    task_validation = family.examples("validation")
    if args.probe_limit_per_split < 1 or args.task_validation_limit < 1:
        raise ValueError("probe/validation scoring limits are too small")
    task_validation = task_validation[: args.task_validation_limit]
    loaded = load_model_and_tokenizer(config["model"], for_training=False)
    model = move_model_to_local_cuda(loaded.model)
    initial_hash = sha256_file(args.initial_checkpoint)
    load_checkpoint_into_hf_model(model, args.initial_checkpoint, expected_sha256=initial_hash)
    max_new_tokens = int(config["anti_shortcut"]["max_completion_length"])
    initial_probe, _ = _score_examples(
        model,
        loaded.tokenizer,
        [*discovery, *validation],
        max_new_tokens=max_new_tokens,
        model_config=config["model"],
    )
    _, initial_validation_results = _score_examples(
        model,
        loaded.tokenizer,
        task_validation,
        max_new_tokens=max_new_tokens,
        model_config=config["model"],
    )
    calibration_hash = sha256_file(calibration_artifacts.checkpoint_path)
    calibration_run = calibration_artifacts.manifest
    calibration_binding = calibration_artifacts.content_binding()
    eligibility_evidence_ancestry = [
        {
            "calibration_checkpoint_sha256": calibration_hash,
            "calibration_run_manifest_sha256": str(calibration_run["sha256"]),
            "calibration_run_id": str(calibration_run.get("run_id", "")),
            "experiment_binding_sha256": str(
                calibration_run.get("experiment_binding_sha256", "")
            ),
            "factorial_update_evidence_sha256": str(
                calibration_artifacts.evidence["sha256"]
            ),
            "strict_run_artifact_binding_sha256": calibration_binding["sha256"],
        }
    ]
    load_checkpoint_into_hf_model(
        model,
        calibration_artifacts.checkpoint_path,
        expected_sha256=calibration_hash,
    )
    calibrated_probe, _ = _score_examples(
        model,
        loaded.tokenizer,
        [*discovery, *validation],
        max_new_tokens=max_new_tokens,
        model_config=config["model"],
    )
    _, calibrated_validation_results = _score_examples(
        model,
        loaded.tokenizer,
        task_validation,
        max_new_tokens=max_new_tokens,
        model_config=config["model"],
    )
    rows = {
        example_id: {
            "initial_correct": initial_probe[example_id],
            "learnable_after_post_training": calibrated_probe[example_id],
        }
        for example_id in initial_probe
    }
    metrics = aggregate_verification(initial_validation_results)
    payload = {
        "format_version": 2,
        **scientific_compatibility_fields(str(config["prereg_version"])),
        **formal_binding,
        "scores": rows,
        "initial_validation_metrics": metrics,
        "calibrated_validation_metrics": aggregate_verification(calibrated_validation_results),
        "initial_checkpoint_sha256": initial_hash,
        "initial_checkpoint_identity": str(config["model"]["model_revision"]),
        "protocol_track": str(config.get("protocol_track", "core_v2")),
        "artifact_namespace": str(config["model"].get("artifact_namespace", "legacy")),
        "prompt_protocol": loaded.prompt_protocol,
        "enable_thinking": False,
        "chat_template_sha256": loaded.chat_template_sha256,
        "tokenizer_fingerprint": loaded.tokenizer_hash,
        "calibration_checkpoint_sha256": calibration_hash,
        "calibration_run_artifact_binding": calibration_binding,
        "eligibility_evidence_ancestry": eligibility_evidence_ancestry,
        "source_dataset_family_hash": family.manifest["sha256"],
        "source_split_hashes": {
            subset: family.boundary(SOURCE_SPLITS[subset])["examples_file_sha256"]
            for subset in SUBSETS
        },
        "task_validation_split_hash": family.boundary("validation")[
            "examples_file_sha256"
        ],
        "ordered_candidate_pairs": {
            "discovery": ordered_pair_population(discovery_pairs),
            "validation": ordered_pair_population(validation_pairs),
        },
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
