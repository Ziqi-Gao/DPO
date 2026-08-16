"""Evaluate the fixed semantics-preserving ProofGraph anti-shortcut suite."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from posttrain_circuits.circuits.mib_runner import load_checkpoint_into_hf_model
from posttrain_circuits.cli._common import enforce_production_guard, parse_cli, print_json
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.data.splits import build_split
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.tasks.proofgraph.anti_shortcut import (
    build_anti_shortcut_suite,
    evaluate_anti_shortcut_suite,
)
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.utils.tiny_model import build_tiny_qwen, build_tiny_tokenizer


def _predictor(model: torch.nn.Module, tokenizer: Any, max_new_tokens: int):
    @torch.no_grad()
    def predict(_example: Any, prompt: str) -> str:
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        encoded = encoded.to(next(model.parameters()).device)
        generate = getattr(model, "generate", None)
        if not callable(generate):
            raise TypeError("anti-shortcut evaluation model must provide generate()")
        generated = generate(
            input_ids=encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        return tokenizer.decode(generated[0, encoded.shape[1] :], skip_special_tokens=True)

    return predict


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Evaluate ProofGraph anti-shortcut robustness", argv)
    gate_config = config["anti_shortcut"]
    output = args.output or Path(str(gate_config["report_path"]))
    if not enforce_production_guard(
        config,
        dry_run=args.dry_run,
        confirm_production=args.confirm_production,
        output=output,
    ):
        return
    model_config = config["model"]
    if str(model_config["model_name_or_path"]).startswith("local/"):
        model = build_tiny_qwen(int(config["seed"])).eval()
        tokenizer = build_tiny_tokenizer()
        checkpoint_hash = str(model_config["model_revision"])
    else:
        loaded = load_model_and_tokenizer(model_config, for_training=False)
        model = move_model_to_local_cuda(loaded.model).eval()
        tokenizer = loaded.tokenizer
        checkpoint_value = config["production_safety"].get("initial_checkpoint_path")
        if not checkpoint_value:
            raise ValueError("production anti-shortcut evaluation requires initial_checkpoint_path")
        checkpoint_path = Path(str(checkpoint_value))
        checkpoint_hash = sha256_file(checkpoint_path)
        expected_hash = str(config["production_safety"].get("initial_checkpoint_hash", ""))
        if checkpoint_hash != expected_hash:
            raise ValueError("anti-shortcut initial checkpoint bytes differ from configured hash")
        load_checkpoint_into_hf_model(model, checkpoint_path, expected_sha256=checkpoint_hash)
    task_config = dict(config["task"])
    examples = build_split(
        ProofGraphTask(),
        "iid_test",
        int(gate_config["num_examples"]),
        int(task_config.get("seed", config["seed"])),
        task_config,
    )
    cases = build_anti_shortcut_suite(
        examples,
        seed=int(config["seed"]),
        distractor_ood_count=int(gate_config["distractor_ood_count"]),
    )
    report = evaluate_anti_shortcut_suite(
        examples,
        cases,
        _predictor(model, tokenizer, int(gate_config["max_completion_length"])),
        max_shortcut_gap=float(gate_config["max_shortcut_gap"]),
        model_checkpoint_hash=checkpoint_hash,
        minimum_iid_accuracy=float(gate_config["minimum_iid_accuracy"]),
        minimum_transformed_accuracy=float(gate_config["minimum_transformed_accuracy"]),
        minimum_per_transformation_accuracy=float(gate_config["minimum_per_transformation_accuracy"]),
        dataset_hash=sha256_value([asdict(example) for example in examples]),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        prereg_commit=subprocess.check_output(
            ["git", "log", "-n", "1", "--format=%H", "--", "prereg/core_v1.yaml"],
            text=True,
        ).strip(),
    )
    atomic_write_json(output, report)
    print_json({"output": str(output), **report})


if __name__ == "__main__":
    main()
