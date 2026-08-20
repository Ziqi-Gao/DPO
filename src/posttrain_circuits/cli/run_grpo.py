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
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json
from posttrain_circuits.core.provenance import (
    RunManifest,
    finalize_run_directory,
    initialize_run_directory,
)
from posttrain_circuits.core.readiness import require_factorial_prerequisites
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.data.splits import build_split, load_frozen_split
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.models.prompt_protocol import format_model_prompt
from posttrain_circuits.rewards.random_matched import validate_random_reward_calibration
from posttrain_circuits.tasks.proofgraph.generator import ProofGraphTask
from posttrain_circuits.tasks.proofgraph.schemas import TaskExample
from posttrain_circuits.training.evaluation import build_proofgraph_evaluator
from posttrain_circuits.training.grpo_backend import (
    GrpoSettings,
    TrlGrpoBackend,
    resolve_grpo_batch_contract,
)
from posttrain_circuits.training.grpo_data import build_grpo_rows_and_reward
from posttrain_circuits.training.local_fork import state_hash
from posttrain_circuits.training.token_budget import maximum_grpo_tokens_per_update
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
def _probe_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    device = next(model.parameters()).device
    mask = attention_mask.to(device)
    logits = model(
        input_ids=input_ids.to(device),
        attention_mask=mask,
    ).logits
    positions = mask.long().sum(dim=1) - 1
    selected = logits[torch.arange(logits.shape[0], device=device), positions]
    return selected.float().log_softmax(dim=-1).detach()


@torch.no_grad()
def _output_kl_new_to_initial(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    initial_log_probs: torch.Tensor,
) -> float:
    current = _probe_log_probs(model, input_ids, attention_mask)
    initial = initial_log_probs.to(current.device)
    prompt_kl = (current.exp() * (current - initial)).sum(dim=-1)
    return float(prompt_kl.mean().detach().cpu())


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


def _snapshot_parameter_state(
    state: dict[str, torch.Tensor],
    root: Path,
) -> dict[str, object]:
    """Persist one tensor per file so update-norm comparison stays bounded."""

    root.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for index, (name, tensor) in enumerate(state.items()):
        path = root / f"{index:06d}.pt"
        torch.save(tensor.detach().cpu(), path)
        tensors[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    payload: dict[str, object] = {"tensors": tensors}
    payload["sha256"] = sha256_value(payload)
    atomic_write_json(root / "manifest.json", payload)
    return payload


def _streaming_parameter_update_norm(
    snapshot: dict[str, object],
    final_state: dict[str, torch.Tensor],
) -> float:
    squared = 0.0
    rows = snapshot.get("tensors")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("GRPO initial parameter snapshot is empty")
    for name, raw in rows.items():
        if name not in final_state or not isinstance(raw, dict):
            raise ValueError(f"GRPO final state omitted snapshotted parameter {name}")
        path = Path(str(raw["path"]))
        if sha256_file(path) != raw["sha256"]:
            raise ValueError(f"GRPO initial parameter snapshot was modified: {name}")
        before = torch.load(path, map_location="cpu", weights_only=True).float()
        after = final_state[name].detach().cpu().float()
        squared += float(torch.sum((after - before) ** 2).item())
    return math.sqrt(squared)


def _world_info() -> tuple[int, int]:
    return int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))


@torch.no_grad()
def _distributed_parameter_checksum(model: torch.nn.Module, *, world_size: int) -> str:
    """Hash an all-reduced checksum over replicated or sharded live parameters."""

    device = next(model.parameters()).device
    summary = torch.zeros(3, dtype=torch.float64, device=device)
    for parameter in model.parameters():
        values = parameter.detach().double()
        summary[0] += values.numel()
        summary[1] += values.sum()
        summary[2] += torch.square(values).sum()
    if world_size > 1:
        if not torch.distributed.is_initialized():
            raise RuntimeError("distributed GRPO checksum requires an initialized process group")
        torch.distributed.all_reduce(summary)
    return sha256_value(
        {
            "world_size": world_size,
            "all_reduced_numel": float(summary[0].cpu()),
            "all_reduced_sum": float(summary[1].cpu()),
            "all_reduced_squared_sum": float(summary[2].cpu()),
        }
    )


def _directory_hashes(root: Path) -> dict[str, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("Accelerate checkpoint directory is empty")
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def _wait_for_everyone(trainer: object) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        accelerator.wait_for_everyone()


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
    prerequisite_evidence = {} if is_tiny else require_factorial_prerequisites(config)
    world_size, rank = _world_info()
    is_main_process = rank == 0
    resume_payload = None
    initial_budget_consumed = 0
    initial_optimizer_steps = 0
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        budget_state = resume_payload.get("token_budget")
        if not isinstance(budget_state, dict):
            raise ValueError("GRPO resume checkpoint has no token-budget state")
        if int(budget_state.get("budget", -1)) != int(config["trainer"]["token_budget"]):
            raise ValueError("GRPO resume checkpoint token budget differs from config")
        initial_budget_consumed = int(budget_state["consumed"])
        initial_optimizer_steps = int(resume_payload["global_step"])
    initial_checkpoint_hash: str | None = None
    max_prompt_length = int(
        config["supervision"].get(
            "max_prompt_length",
            128 if is_tiny else 2048,
        )
    )
    reserved_tokens_per_update = maximum_grpo_tokens_per_update(
        world_size=world_size,
        per_device_batch_size=int(config["trainer"]["batch_size"]),
        gradient_accumulation_steps=int(config["supervision"]["gradient_accumulation_steps"]),
        max_prompt_length=max_prompt_length,
        max_completion_length=int(config["supervision"]["max_completion_length"]),
    )
    settings = GrpoSettings(
        max_completion_length=int(
            config["supervision"]["max_completion_length"],
        ),
        max_steps=int(config["trainer"]["max_steps"]),
        beta=float(config["supervision"]["beta"]),
        num_generations=int(config["supervision"]["num_generations"]),
        temperature=float(config["supervision"]["temperature"]),
        top_p=float(config["supervision"].get("top_p", 1.0)),
        top_k=(
            int(config["supervision"]["top_k"]) if config["supervision"].get("top_k") is not None else None
        ),
        min_p=(
            float(config["supervision"]["min_p"]) if config["supervision"].get("min_p") is not None else None
        ),
        loss_type=str(config["supervision"]["loss_type"]),
        scale_rewards=config["supervision"]["scale_rewards"],
        gradient_accumulation_steps=int(
            config["supervision"]["gradient_accumulation_steps"],
        ),
        per_device_train_batch_size=int(config["trainer"]["batch_size"]),
        learning_rate=float(config["trainer"]["learning_rate"]),
        max_prompt_length=max_prompt_length,
        use_cpu=is_tiny,
        gradient_checkpointing=bool(
            model_config["gradient_checkpointing"],
        ),
        seed=int(config["seed"]),
        token_budget=int(config["trainer"]["token_budget"]),
        token_budget_unit=str(
            config["trainer"].get("token_budget_unit", "global_nonpadding_model_input_tokens")
        ),
        reserved_tokens_per_update=reserved_tokens_per_update,
        initial_token_budget_consumed=initial_budget_consumed,
        initial_optimizer_steps=initial_optimizer_steps,
    )
    batch_contract = resolve_grpo_batch_contract(settings, world_size=world_size)

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
    production_parameter_snapshot = None

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
    matched_positive_rate = None
    random_reward_calibration_hash = None
    if reward_name == "matched_random":
        if is_tiny:
            matched_positive_rate = 0.5
            random_reward_calibration_hash = sha256_value(
                {"fixture": "tiny-frozen-random-reward-calibration", "positive_rate": 0.5}
            )
        else:
            calibration_path = Path(str(config["experiment"].get("random_reward_calibration_path", "")))
            if not calibration_path.is_file():
                raise ValueError("random-reward GRPO requires a frozen calibration artifact")
            calibration = validate_random_reward_calibration(
                json.loads(calibration_path.read_text(encoding="utf-8"))
            )
            matched_positive_rate = float(calibration["positive_rate"])
            random_reward_calibration_hash = str(calibration["sha256"])
    rows, reward = build_grpo_rows_and_reward(
        examples,
        reward_name=reward_name,
        seed=int(config["seed"]),
        matched_positive_rate=matched_positive_rate,
        tokenizer=tokenizer,
        model_config=model_config,
    )
    prompt_hash = sha256_value(rows)
    probe_input_ids: torch.Tensor | None = None
    probe_attention_mask: torch.Tensor | None = None
    output_kl_probe_manifest_hash: str | None = None
    initial_probe_log_probs: torch.Tensor | None = None
    initial_validation_metrics: dict[str, float] = {}
    if validation_examples:
        evaluator = build_proofgraph_evaluator(
            validation_examples,
            tokenizer,
            max_completion_length=int(config["supervision"]["max_completion_length"]),
            model_config=model_config,
        )
        initial_validation_metrics = evaluator(model)
        probe_examples = validation_examples[: min(16, len(validation_examples))]
        encoded_probe = tokenizer(
            [
                format_model_prompt(
                    ProofGraphTask().render(example), tokenizer, model_config
                ).model_facing_prompt
                for example in probe_examples
            ],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        probe_input_ids = encoded_probe.input_ids
        probe_attention_mask = encoded_probe.attention_mask
        probe_manifest = {
            "kind": "multi_prompt_behavioral_output_kl",
            "example_ids": [example.example_id for example in probe_examples],
            "input_ids": probe_input_ids.tolist(),
            "attention_mask": probe_attention_mask.tolist(),
            "validation_manifest_hash": validation_manifest_hash,
            "tokenizer_revision": resolved_tokenizer_commit,
        }
        output_kl_probe_manifest_hash = sha256_value(probe_manifest)
        initial_probe_log_probs = _probe_log_probs(
            model,
            probe_input_ids,
            probe_attention_mask,
        )
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
            **{
                f"prerequisite_{name}": str(evidence.get("report_hash", evidence.get("manifest_hash")))
                for name, evidence in prerequisite_evidence.items()
            },
        },
        rollout_bank_hash=source_hash,
        prompt_schedule_hash=prompt_hash,
        raw_prompt_schedule_hash=sha256_value([ProofGraphTask().render(example) for example in examples]),
        model_facing_prompt_schedule_hash=prompt_hash,
        prompt_protocol=(loaded.prompt_protocol if not is_tiny else "legacy_raw_v1"),
        enable_thinking=False,
        chat_template_sha256=(loaded.chat_template_sha256 if not is_tiny else "legacy-unrecorded"),
        tokenizer_fingerprint=(loaded.tokenizer_hash if not is_tiny else "legacy-unrecorded"),
        protocol_track=str(config.get("protocol_track", "core_v2")),
        artifact_namespace=str(model_config.get("artifact_namespace", "legacy")),
        protocol_teacher_revision=str(config["teacher"].get("model_revision", "unbound")),
        token_budget=int(config["trainer"]["token_budget"]),
        token_budget_unit=str(
            config["trainer"].get("token_budget_unit", "global_nonpadding_model_input_tokens")
        ),
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
    if resume_payload is not None:
        accelerator_state_path = str(resume_payload.get("accelerate_state_path", ""))
        if not accelerator_state_path:
            raise ValueError("GRPO resume checkpoint has no Accelerate state path")
        trainer.accelerator.load_state(accelerator_state_path)
        trainer.state.global_step = initial_optimizer_steps
    _wait_for_everyone(trainer)
    accelerator = getattr(trainer, "accelerator", None)
    if not is_tiny:
        if accelerator is None:
            raise RuntimeError("production GRPO requires an Accelerator-managed trainer")
        initial_full_state = accelerator.get_state_dict(trainer.model)
        if is_main_process:
            production_parameter_snapshot = _snapshot_parameter_state(
                initial_full_state,
                Path(output) / "initial_parameter_snapshot",
            )
    if is_main_process:
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
    _wait_for_everyone(trainer)
    result = trainer.train()
    _wait_for_everyone(trainer)
    global_step = int(
        getattr(getattr(trainer, "state", None), "global_step", 0),
    )
    if global_step < 1:
        raise RuntimeError("GRPO trainer completed without an optimizer step")
    token_budget_callback = getattr(trainer, "token_budget_callback", None)
    if token_budget_callback is None:
        if not is_tiny:
            raise RuntimeError("production GRPO backend omitted token-budget enforcement")
        token_budget_state = {
            "budget": settings.token_budget,
            "unit": settings.token_budget_unit,
            "consumed": min(settings.reserved_tokens_per_update, settings.token_budget),
            "reserved_tokens_per_update": settings.reserved_tokens_per_update,
            "stop_reason": "max_steps_safety_limit",
        }
    else:
        token_budget_state = token_budget_callback.state_dict()
    raw_consumed = token_budget_state.get("consumed")
    raw_budget = token_budget_state.get("budget")
    if (
        isinstance(raw_consumed, bool)
        or not isinstance(raw_consumed, int)
        or isinstance(raw_budget, bool)
        or not isinstance(raw_budget, int)
    ):
        raise RuntimeError("GRPO token-budget state requires integer consumed and budget values")
    token_budget_consumed = raw_consumed
    token_budget_limit = raw_budget
    if token_budget_consumed > token_budget_limit:
        raise RuntimeError("GRPO completed beyond its registered global token budget")
    trained_model = (
        trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
    )
    if accelerator is not None:
        final_state = accelerator.get_state_dict(trainer.model)
    else:
        final_state = trained_model.state_dict()
    final_model_state_hash = state_hash(final_state) if final_state else None
    distributed_checksum = _distributed_parameter_checksum(trainer.model, world_size=world_size)
    if world_size > 1:
        from accelerate.utils import gather_object

        rank_model_hashes = gather_object(final_model_state_hash)
        rank_parameter_checksums = gather_object(distributed_checksum)
    else:
        rank_model_hashes = [final_model_state_hash]
        rank_parameter_checksums = [distributed_checksum]
    distributed_consistency_passed = len(set(rank_parameter_checksums)) == 1
    if is_main_process and not distributed_consistency_passed:
        raise RuntimeError("distributed GRPO ranks disagree on the final parameter checksum")
    optimizer_state = trainer.optimizer.state_dict()
    scheduler_state = trainer.lr_scheduler.state_dict()
    update_norm = None
    if is_main_process:
        if is_tiny:
            update_norm = _parameter_update_norm(initial_parameters, trained_model)
        else:
            assert production_parameter_snapshot is not None
            if not final_state:
                raise RuntimeError("main GRPO rank did not receive the full FSDP state dict")
            update_norm = _streaming_parameter_update_norm(
                production_parameter_snapshot,
                final_state,
            )
    if is_main_process and (update_norm is None or update_norm <= 0.0):
        raise RuntimeError("GRPO optimizer step did not change parameters")
    validation_metrics: dict[str, float] = {}
    if validation_examples:
        validation_metrics = build_proofgraph_evaluator(
            validation_examples,
            tokenizer,
            max_completion_length=int(config["supervision"]["max_completion_length"]),
            model_config=model_config,
        )(trainer.model)
    output_kl = None
    if (
        probe_input_ids is not None
        and probe_attention_mask is not None
        and initial_probe_log_probs is not None
    ):
        output_kl = _output_kl_new_to_initial(
            trainer.model,
            probe_input_ids,
            probe_attention_mask,
            initial_probe_log_probs,
        )
    checkpoint_path = Path(output) / "checkpoints" / f"step-{global_step:08d}.pt"
    accelerate_state_path = Path(output) / "checkpoints" / f"accelerate-step-{global_step:08d}"
    evidence_path = Path(output) / "grpo_update_evidence.json"
    _wait_for_everyone(trainer)
    accelerate_state_hashes = None
    if accelerator is not None:
        accelerator.save_state(str(accelerate_state_path))
        _wait_for_everyone(trainer)
        if is_main_process:
            accelerate_state_hashes = _directory_hashes(accelerate_state_path)
    if is_main_process:
        _atomic_torch_save(
            checkpoint_path,
            {
                "model": _cpu_tree(final_state),
                "optimizer": (
                    _cpu_tree(optimizer_state)
                    if accelerator is None
                    else {
                        "accelerate_state_path": str(accelerate_state_path),
                        "files": accelerate_state_hashes,
                    }
                ),
                "scheduler": _cpu_tree(scheduler_state),
                "rng": RNGState.capture().as_dict(),
                "resolved_config": config,
                "global_step": global_step,
                "world_size": world_size,
                "initial_checkpoint_sha256": initial_checkpoint_hash,
                "token_budget": token_budget_state,
                "accelerate_state_path": str(accelerate_state_path),
            },
        )
        checkpoint_sha256 = sha256_file(checkpoint_path)
        reward_artifact_hash = random_reward_calibration_hash or sha256_value(
            {"reward": reward_name, "seed": config["seed"]}
        )
        evidence = {
            "format_version": 2,
            "prereg_version": str(config.get("protocol_track", "core_v2")),
            "generator_version": ProofGraphTask.generator_version,
            "label_semantics": ProofGraphTask.label_semantics,
            "backend": "trl.GRPOTrainer",
            "trl_train_called": True,
            "main_process_only_writes": True,
            "optimizer_steps": global_step,
            "token_budget": token_budget_state,
            "parameter_update_norm": update_norm,
            "parameter_update_norm_method": "one-tensor-at-a-time-disk-snapshot",
            "parameters_changed": bool(update_norm and update_norm > 0.0),
            "reward": reward_name,
            "reward_artifact_hash": reward_artifact_hash,
            "random_reward_calibration_hash": random_reward_calibration_hash,
            "tiny_smoke_matched_positive_rate": matched_positive_rate,
            "settings": settings.__dict__,
            "batch_contract": batch_contract,
            "world_size": world_size,
            "checkpoint": str(checkpoint_path),
            "initial_checkpoint_hash": initial_checkpoint_hash
            or state_hash({str(index): tensor for index, tensor in enumerate(initial_parameters)}),
            "final_checkpoint_hash": checkpoint_sha256,
            "final_model_state_hash": final_model_state_hash,
            "rank_model_hashes": rank_model_hashes,
            "synchronized_final_model_hashes": (
                all(value is not None for value in rank_model_hashes) and len(set(rank_model_hashes)) == 1
            ),
            "distributed_parameter_checksum": distributed_checksum,
            "rank_distributed_parameter_checksums": rank_parameter_checksums,
            "distributed_consistency_passed": distributed_consistency_passed,
            "accelerate_state_path": str(accelerate_state_path) if accelerator is not None else None,
            "accelerate_state_hashes": accelerate_state_hashes,
            "validation_metrics": validation_metrics,
            "initial_validation_metrics": initial_validation_metrics,
            "output_kl_from_initial": output_kl,
            "output_kl_probe_manifest_hash": output_kl_probe_manifest_hash,
            "output_kl_probe_prompt_count": (
                int(probe_input_ids.shape[0]) if probe_input_ids is not None else 0
            ),
        }
        evidence["sha256"] = sha256_value(evidence)
        atomic_write_json(evidence_path, evidence)
        if validation_metrics:
            with (Path(output) / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": global_step,
                            **validation_metrics,
                            "output_kl_from_initial": output_kl,
                            "token_budget": token_budget_limit,
                            "token_budget_unit": token_budget_state["unit"],
                            "token_budget_consumed": token_budget_consumed,
                            "stop_reason": token_budget_state["stop_reason"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        manifest.token_budget_consumed = token_budget_consumed
        manifest.training_stop_reason = str(token_budget_state["stop_reason"])
        manifest.metrics_sha256 = sha256_file(Path(output) / "metrics.jsonl")
        manifest.final_checkpoint_path = str(checkpoint_path)
        manifest.final_checkpoint_sha256 = checkpoint_sha256
        finalize_run_directory(output, manifest)
        print_json(
            {
                "status": "completed",
                "settings": settings.__dict__,
                "batch_contract": batch_contract,
                "output": str(output),
                "train_result": str(result),
                "optimizer_update_evidence": str(evidence_path),
                "optimizer_steps": global_step,
                "parameter_update_norm": update_norm,
            }
        )
    _wait_for_everyone(trainer)


if __name__ == "__main__":
    main()
