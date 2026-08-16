"""Run canonical or control GRPO through the official TRL trainer."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from posttrain_circuits.circuits.mib_runner import load_checkpoint_into_hf_model
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
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.data.splits import build_split, load_frozen_split
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample
from posttrain_circuits.training.evaluation import build_proofgraph_evaluator
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


def _cpu_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return value


@torch.no_grad()
def _probe_log_probs(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    return model(input_ids=input_ids.to(device)).logits.log_softmax(dim=-1).detach()


@torch.no_grad()
def _output_kl_new_to_initial(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    initial_log_probs: torch.Tensor,
) -> float:
    current = _probe_log_probs(model, input_ids)
    initial = initial_log_probs.to(current.device)
    return float((current.exp() * (current - initial)).sum(dim=-1).mean().detach().cpu())


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


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
    initial_checkpoint_hash: str | None = None
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
        initial_checkpoint_value = str(
            config.get("production_safety", {}).get("initial_checkpoint_path", "")
        ).strip()
        initial_checkpoint_hash = str(
            config.get("production_safety", {}).get("initial_checkpoint_hash", "")
        ).strip()
        if not initial_checkpoint_value or len(initial_checkpoint_hash) != 64:
            raise ValueError("production GRPO requires initial_checkpoint_path and its SHA-256")
        load_checkpoint_into_hf_model(
            model,
            Path(initial_checkpoint_value),
            expected_sha256=initial_checkpoint_hash,
        )
        model = move_model_to_local_cuda(model)
    initial_parameters = (
        [parameter.detach().cpu().clone() for parameter in model.parameters() if parameter.requires_grad]
        if is_tiny
        else []
    )

    task_config = config["task"]
    validation_examples: list[TaskExample] = []
    validation_manifest_hash = None
    if not is_tiny:
        validation_examples, validation_manifest = load_frozen_split(
            Path(str(task_config["validation_split_path"])),
            expected_split="validation",
        )
        validation_examples = validation_examples[: int(config["trainer"]["validation_examples"])]
        validation_manifest_hash = str(validation_manifest["sha256"])
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
    probe_input_ids: torch.Tensor | None = None
    initial_probe_log_probs: torch.Tensor | None = None
    initial_validation_metrics: dict[str, float] = {}
    if validation_examples:
        evaluator = build_proofgraph_evaluator(
            validation_examples,
            tokenizer,
            max_completion_length=int(config["supervision"]["max_completion_length"]),
        )
        initial_validation_metrics = evaluator(model)
        probe_input_ids = tokenizer(
            str(rows[0]["prompt"]),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        initial_probe_log_probs = _probe_log_probs(model, probe_input_ids)
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
        dataset_hashes={
            "train": sha256_value([asdict(example) for example in examples]),
            **(
                {
                    "validation": sha256_value([asdict(example) for example in validation_examples]),
                    "validation_manifest": validation_manifest_hash,
                }
                if validation_manifest_hash
                else {}
            ),
            **(
                {"initial_checkpoint": initial_checkpoint_hash} if initial_checkpoint_hash is not None else {}
            ),
        },
        rollout_bank_hash=source_hash,
        prompt_schedule_hash=prompt_hash,
    )
    initialize_run_directory(output, config, manifest, require_git=not is_tiny)
    if initial_validation_metrics:
        with (Path(output) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": 0,
                        **initial_validation_metrics,
                        "output_kl_from_initial": 0.0,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
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
    trained_model = (
        trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
    )
    validation_metrics: dict[str, float] = {}
    if validation_examples:
        validation_metrics = build_proofgraph_evaluator(
            validation_examples,
            tokenizer,
            max_completion_length=int(config["supervision"]["max_completion_length"]),
        )(trained_model)
    output_kl = None
    if probe_input_ids is not None and initial_probe_log_probs is not None:
        output_kl = _output_kl_new_to_initial(
            trained_model,
            probe_input_ids,
            initial_probe_log_probs,
        )
    checkpoint_path = Path(output) / "checkpoints" / f"step-{global_step:08d}.pt"
    _atomic_torch_save(
        checkpoint_path,
        {
            "model": _cpu_tree(trained_model.state_dict()),
            "optimizer": _cpu_tree(trainer.optimizer.state_dict()),
            "scheduler": _cpu_tree(trainer.lr_scheduler.state_dict()),
            "rng": RNGState.capture().as_dict(),
            "resolved_config": config,
            "global_step": global_step,
        },
    )
    evidence = {
        "backend": "trl.GRPOTrainer",
        "trl_train_called": True,
        "optimizer_steps": global_step,
        "parameter_update_norm": update_norm,
        "parameters_changed": (update_norm > 0.0 if update_norm is not None else None),
        "reward": reward_name,
        "tiny_smoke_matched_positive_rate": matched_positive_rate,
        "settings": settings.__dict__,
        "checkpoint": str(checkpoint_path),
        "validation_metrics": validation_metrics,
        "initial_validation_metrics": initial_validation_metrics,
        "output_kl_from_initial": output_kl,
    }
    evidence_path = Path(output) / "grpo_update_evidence.json"
    atomic_write_json(evidence_path, evidence)
    if validation_metrics:
        with (Path(output) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": global_step,
                        **validation_metrics,
                        "output_kl_from_initial": output_kl,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
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
