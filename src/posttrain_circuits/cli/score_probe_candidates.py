"""Score frozen ProofGraph probes under initial and calibration checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.circuits.mib_runner import load_checkpoint_into_hf_model
from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import formal_artifact_binding
from posttrain_circuits.core.scientific_versions import scientific_compatibility_fields
from posttrain_circuits.data.splits import load_frozen_split
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.models.prompt_protocol import format_model_prompt
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.metrics import aggregate_verification
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample, VerificationResult


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
    parser.add_argument("--discovery-split", type=Path, required=True)
    parser.add_argument("--validation-split", type=Path, required=True)
    parser.add_argument("--task-validation-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-limit-per-split", type=int, required=True)
    parser.add_argument("--task-validation-limit", type=int, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    if str(config["model"]["model_name_or_path"]).startswith("local/"):
        raise ValueError("production probe scoring cannot use a tiny model")
    discovery, discovery_manifest = load_frozen_split(
        args.discovery_split, expected_split="circuit_discovery"
    )
    validation, validation_manifest = load_frozen_split(
        args.validation_split, expected_split="circuit_validation"
    )
    task_validation, task_validation_manifest = load_frozen_split(
        args.task_validation_split, expected_split="validation"
    )
    if args.probe_limit_per_split < 2 or args.task_validation_limit < 1:
        raise ValueError("probe/validation scoring limits are too small")
    discovery = discovery[: args.probe_limit_per_split]
    validation = validation[: args.probe_limit_per_split]
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
    calibration_hash = sha256_file(args.calibration_checkpoint)
    load_checkpoint_into_hf_model(model, args.calibration_checkpoint, expected_sha256=calibration_hash)
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
        **formal_artifact_binding(config),
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
        "source_split_hashes": {
            "discovery": discovery_manifest["sha256"],
            "validation": validation_manifest["sha256"],
            "task_validation": task_validation_manifest["sha256"],
        },
    }
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
