"""Run canonical or control GRPO through the official TRL trainer."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from posttrain_circuits.cli._common import (
    enforce_production_guard,
    parse_cli,
    print_json,
)
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import (
    RunManifest,
    finalize_run_directory,
    initialize_run_directory,
)
from posttrain_circuits.data.splits import build_split
from posttrain_circuits.models.loading import load_model_and_tokenizer
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.training.grpo_backend import (
    GrpoSettings,
    TrlGrpoBackend,
)
from posttrain_circuits.training.grpo_data import build_grpo_rows_and_reward
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)


def _parameter_update_norm(
    initial: list[torch.Tensor],
    model: torch.nn.Module,
) -> float:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(initial) != len(trainable):
        raise RuntimeError("trainable parameter set changed during GRPO")
    squared = sum(
        float(torch.sum((parameter.detach().cpu().float() - before.float()) ** 2).item())
        for before, parameter in zip(initial, trainable, strict=True)
    )
    return math.sqrt(squared)


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Run canonical TRL GRPO", argv)
    output = args.output or (Path(config["output_root"]) / "runs" / str(config["experiment"]["name"]))
    if not enforce_production_guard(
        config,
        dry_run=args.dry_run,
        confirm_production=args.confirm_production,
        output=output,
    ):
        return

    model_config = config["model"]
    is_tiny = str(model_config["model_name_or_path"]).startswith("local/")
    settings = GrpoSettings(
        max_completion_length=int(
            config["supervision"]["max_completion_length"],
        ),
        max_steps=int(config["trainer"]["max_steps"]),
        beta=float(config["supervision"]["beta"]),
        num_generations=int(config["supervision"]["num_generations"]),
        temperature=float(config["supervision"]["temperature"]),
        loss_type=str(config["supervision"]["loss_type"]),
        scale_rewards=config["supervision"]["scale_rewards"],
        gradient_accumulation_steps=int(
            config["supervision"]["gradient_accumulation_steps"],
        ),
        per_device_train_batch_size=int(config["trainer"]["batch_size"]),
        learning_rate=float(config["trainer"]["learning_rate"]),
        max_prompt_length=int(
            config["supervision"].get(
                "max_prompt_length",
                128 if is_tiny else 2048,
            )
        ),
        use_cpu=is_tiny,
        gradient_checkpointing=bool(
            model_config["gradient_checkpointing"],
        ),
        seed=int(config["seed"]),
    )
    if settings.per_device_train_batch_size % settings.num_generations != 0:
        raise ValueError(
            "GRPO train batch size must be divisible by num_generations",
        )

    if is_tiny:
        model = build_tiny_qwen(int(config["seed"]))
        tokenizer = build_tiny_tokenizer()
        resolved_model_commit = str(model_config["model_revision"])
        resolved_tokenizer_commit = str(model_config["tokenizer_revision"])
    else:
        loaded = load_model_and_tokenizer(
            model_config,
            for_training=True,
        )
        model = loaded.model
        tokenizer = loaded.tokenizer
        resolved_model_commit = loaded.resolved_model_commit
        resolved_tokenizer_commit = loaded.resolved_tokenizer_commit
    initial_parameters = (
        [parameter.detach().cpu().clone() for parameter in model.parameters() if parameter.requires_grad]
        if is_tiny
        else []
    )

    task_config = config["task"]
    examples = build_split(
        ProofGraphTask(),
        "train",
        int(task_config["num_examples"]),
        int(task_config.get("seed", config["seed"])),
        dict(task_config),
    )
    reward_name = str(config["experiment"]["reward"])
    matched_positive_rate = 0.5 if is_tiny and reward_name == "matched_random" else None
    rows, reward = build_grpo_rows_and_reward(
        examples,
        reward_name=reward_name,
        seed=int(config["seed"]),
        matched_positive_rate=matched_positive_rate,
    )
    prompt_hash = sha256_value(rows)
    source_hash = sha256_value(
        {
            "state_source": config["state_source"],
            "initial_policy_revision": resolved_model_commit,
            "seed": config["seed"],
        }
    )
    manifest = RunManifest(
        run_id=(
            f"{config['experiment']['name']}-seed{config['seed']}-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        ),
        experiment_cell=str(config["experiment"]["name"]),
        seed=int(config["seed"]),
        model_id=str(model_config["model_name_or_path"]),
        model_revision=str(model_config["model_revision"]),
        tokenizer_id=str(model_config["tokenizer_name_or_path"]),
        tokenizer_revision=str(model_config["tokenizer_revision"]),
        resolved_model_commit=resolved_model_commit,
        resolved_tokenizer_commit=resolved_tokenizer_commit,
        dataset_hashes={"train": sha256_value([asdict(example) for example in examples])},
        rollout_bank_hash=source_hash,
        prompt_schedule_hash=prompt_hash,
    )
    initialize_run_directory(output, config, manifest, require_git=not is_tiny)
    try:
        from datasets import Dataset
    except ImportError as error:
        raise RuntimeError(
            "canonical GRPO requires the 'rl' extra: pip install -e '.[rl]'",
        ) from error
    dataset = Dataset.from_list(rows)
    trainer = TrlGrpoBackend(settings).build(
        model=model,
        reward_funcs=reward,
        train_dataset=dataset,
        output_dir=str(output),
        processing_class=tokenizer,
    )
    result = trainer.train()
    global_step = int(
        getattr(getattr(trainer, "state", None), "global_step", 0),
    )
    if global_step < 1:
        raise RuntimeError("GRPO trainer completed without an optimizer step")
    update_norm = _parameter_update_norm(initial_parameters, model) if is_tiny else None
    if is_tiny and (update_norm is None or update_norm <= 0.0):
        raise RuntimeError("tiny GRPO optimizer step did not change parameters")
    evidence = {
        "backend": "trl.GRPOTrainer",
        "trl_train_called": True,
        "optimizer_steps": global_step,
        "parameter_update_norm": update_norm,
        "parameters_changed": (update_norm > 0.0 if update_norm is not None else None),
        "reward": reward_name,
        "tiny_smoke_matched_positive_rate": matched_positive_rate,
        "settings": settings.__dict__,
    }
    evidence_path = Path(output) / "grpo_update_evidence.json"
    atomic_write_json(evidence_path, evidence)
    finalize_run_directory(output, manifest)
    print_json(
        {
            "status": "completed",
            "settings": settings.__dict__,
            "output": str(output),
            "train_result": str(result),
            "optimizer_update_evidence": str(evidence_path),
            "optimizer_steps": global_step,
            "parameter_update_norm": update_norm,
        }
    )


if __name__ == "__main__":
    main()
