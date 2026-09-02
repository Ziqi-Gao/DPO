"""Run canonical or control GRPO through the official TRL trainer."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from posttrain_circuits.artifacts.checkpoints import (
    accelerator_state_file_hashes,
    extend_checkpoint_ancestry,
    validate_accelerator_state_directory,
    validate_native_trainer_checkpoint_files,
)
from posttrain_circuits.artifacts.config_bindings import bind_config
from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import (
    atomic_torch_save,
    atomic_write_json,
    publish_path_no_clobber,
    publish_torch_once,
)
from posttrain_circuits.artifacts.runs import (
    RunManifest,
    finalize_run_directory,
    git_output,
    initialize_run_directory,
)
from posttrain_circuits.causal_circuits.model.runner import load_checkpoint_into_hf_model
from posttrain_circuits.cli._common import (
    enforce_production_guard,
    parse_cli,
    print_json,
)
from posttrain_circuits.core.readiness import require_factorial_prerequisites
from posttrain_circuits.core.seeding import RNGState
from posttrain_circuits.datasets.proofgraph.family import load_dataset_family
from posttrain_circuits.experiments.protocols.specs import (
    build_experiment_binding,
    validate_run_manifest_experiment_binding,
)
from posttrain_circuits.methods.registry import get_method_spec
from posttrain_circuits.methods.rl import (
    TRL_GRPO_BACKEND,
    TRL_GRPO_BATCH_CONTRACT,
    TRL_GRPO_VERSION,
)
from posttrain_circuits.methods.specs import REGISTERED_CURRENT_POLICY
from posttrain_circuits.models.loading import load_model_and_tokenizer, move_model_to_local_cuda
from posttrain_circuits.models.prompt_protocol import (
    format_model_prompt,
    format_model_prompts,
    prompt_schedule_hashes,
)
from posttrain_circuits.learning.rl.rewards.random_matched import validate_random_reward_calibration
from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
from posttrain_circuits.datasets.proofgraph.contracts import TaskExample
from posttrain_circuits.learning.training.evaluation import build_proofgraph_evaluator
from posttrain_circuits.learning.training.grpo_backend import (
    GrpoSettings,
    TrlGrpoBackend,
    require_pinned_trl_version,
    resolve_grpo_batch_contract,
)
from posttrain_circuits.learning.training.grpo_data import build_grpo_rows_and_reward
from posttrain_circuits.learning.training.local_fork import state_hash
from posttrain_circuits.learning.training.token_budget import maximum_grpo_tokens_per_update
from posttrain_circuits.utils.tiny_model import (
    build_tiny_qwen,
    build_tiny_tokenizer,
)

_PINNED_TRANSFORMERS_CHECKPOINT_VERSION = "4.56.2"


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


def _snapshot_parameter_state(
    state: dict[str, torch.Tensor],
    root: Path,
) -> dict[str, object]:
    """Persist one tensor per file so update-norm comparison stays bounded."""

    if root.is_symlink():
        raise ValueError("GRPO initial parameter snapshot root must not be a symlink")
    if root.exists() and any(root.iterdir()):
        raise ValueError("GRPO initial parameter snapshot root must start empty")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    tensors = {}
    for index, (name, tensor) in enumerate(state.items()):
        path = root / f"{index:06d}.pt"
        atomic_torch_save(path, tensor.detach().cpu())
        tensors[name] = {
            "path": str(path.resolve()),
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


class _InitialParameterSnapshotCallback(TrainerCallback):
    """Capture the actual post-resume model state before the first new update."""

    def __init__(self, *, accelerator: object, root: Path, is_main_process: bool) -> None:
        self.accelerator = accelerator
        self.root = root
        self.is_main_process = is_main_process
        self.snapshot: dict[str, object] | None = None

    def on_train_begin(
        self,
        args: object,
        state: object,
        control: object,
        **kwargs: object,
    ) -> object:
        del args, state
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("GRPO initial snapshot callback did not receive the model")
        full_state = self.accelerator.get_state_dict(model)  # type: ignore[attr-defined]
        if self.is_main_process:
            self.snapshot = _snapshot_parameter_state(full_state, self.root)
        return control


def _world_info() -> tuple[int, int]:
    return int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))


@torch.no_grad()
def _rank_local_parameter_summary(model: torch.nn.Module, *, rank: int) -> dict[str, object]:
    """Record, without pretending equality, each rank's live shard summary."""

    numel = 0
    value_sum = 0.0
    squared_sum = 0.0
    for parameter in model.parameters():
        values = parameter.detach().double()
        numel += values.numel()
        value_sum += float(values.sum().cpu())
        squared_sum += float(torch.square(values).sum().cpu())
    payload: dict[str, object] = {
        "rank": rank,
        "local_numel": numel,
        "local_sum": value_sum,
        "local_squared_sum": squared_sum,
    }
    payload["sha256"] = sha256_value(payload)
    return payload


def _gather_rank_objects(value: object, *, world_size: int) -> list[object]:
    if world_size == 1:
        return [value]
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("distributed GRPO evidence requires an initialized process group")
    gathered: list[object] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _validate_distributed_training_consensus(
    rows: list[object],
    *,
    world_size: int,
) -> list[dict[str, object]]:
    if len(rows) != world_size or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("GRPO distributed consensus omitted a rank")
    normalized = [dict(row) for row in rows if isinstance(row, dict)]
    if [row.get("rank") for row in normalized] != list(range(world_size)):
        raise RuntimeError("GRPO distributed consensus rank index changed")
    comparable = [
        {key: value for key, value in row.items() if key != "rank"}
        for row in normalized
    ]
    if any(row != comparable[0] for row in comparable[1:]):
        raise RuntimeError("GRPO ranks disagree on optimizer-step or token-budget state")
    return normalized


def _native_trainer_checkpoint_path(
    trainer: object,
    *,
    output: Path,
    global_step: int,
) -> Path:
    observed_transformers = importlib_metadata.version("transformers")
    if observed_transformers != _PINNED_TRANSFORMERS_CHECKPOINT_VERSION:
        raise RuntimeError(
            "native GRPO checkpoint contract requires "
            f"transformers=={_PINNED_TRANSFORMERS_CHECKPOINT_VERSION}, "
            f"observed={observed_transformers}"
        )
    if PREFIX_CHECKPOINT_DIR != "checkpoint":
        raise RuntimeError("pinned Transformers checkpoint directory contract changed")
    trainer_args = getattr(trainer, "args", None)
    configured_output = Path(str(getattr(trainer_args, "output_dir", ""))).absolute()
    expected_output = Path(output).absolute()
    if configured_output != expected_output:
        raise RuntimeError("GRPO Trainer output directory differs from the run directory")
    if type(global_step) is not int or global_step < 1:
        raise RuntimeError("native GRPO checkpoint requires a positive optimizer step")
    return expected_output / f"{PREFIX_CHECKPOINT_DIR}-{global_step}"


def _native_checkpoint_preflight_report(
    path: Path | None,
    *,
    expected_run_root: Path,
    rank: int,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    """Sample native-state and metadata targets without raising pre-collective."""

    run_root = Path(expected_run_root).absolute()
    target = Path(path).absolute() if path is not None else None
    targets: dict[str, str] = {}
    if target is not None:
        targets["native_state"] = str(target)
    if metadata_path is not None:
        targets["metadata"] = str(Path(metadata_path).absolute())
    report: dict[str, object] = {
        "rank": rank,
        "target": str(target) if target is not None else None,
        "targets": targets,
        "run_root": str(run_root),
        "passed": False,
        "error": None,
    }
    try:
        if not targets:
            raise ValueError("GRPO checkpoint preflight has no targets")
        if run_root.is_symlink() or not run_root.is_dir():
            raise ValueError("native Trainer checkpoint run root is not a regular directory")
        if run_root.resolve(strict=True) != run_root:
            raise ValueError("native Trainer checkpoint run root traverses a symlink")
        for name, raw_target in targets.items():
            candidate = Path(raw_target)
            try:
                relative_parent = candidate.parent.relative_to(run_root)
            except ValueError as error:
                raise ValueError(f"GRPO checkpoint {name} target is outside the run root") from error
            current = run_root
            for part in relative_parent.parts:
                current /= part
                if current.is_symlink() or not current.is_dir():
                    raise ValueError(f"GRPO checkpoint {name} has an unsafe parent")
            try:
                path_status = candidate.lstat()
            except FileNotFoundError:
                path_status = None
            if path_status is not None:
                raise FileExistsError(f"GRPO checkpoint {name} target already exists")
        report["passed"] = True
    except (OSError, ValueError) as error:
        report["error"] = str(error)
    return report


def _validate_native_checkpoint_preflight_reports(
    rows: list[object],
    *,
    world_size: int,
) -> Path | dict[str, Path]:
    if len(rows) != world_size or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("native Trainer checkpoint preflight omitted a rank")
    normalized = [dict(row) for row in rows if isinstance(row, dict)]
    if [row.get("rank") for row in normalized] != list(range(world_size)):
        raise RuntimeError("native Trainer checkpoint preflight rank order changed")
    target_maps = [row.get("targets") for row in normalized]
    roots = {row.get("run_root") for row in normalized}
    if (
        any(not isinstance(targets, dict) for targets in target_maps)
        or any(targets != target_maps[0] for targets in target_maps[1:])
        or len(roots) != 1
    ):
        raise RuntimeError("GRPO ranks disagree on the native Trainer checkpoint target")
    failures = [
        {"rank": row.get("rank"), "error": row.get("error")}
        for row in normalized
        if row.get("passed") is not True
    ]
    if failures:
        raise RuntimeError(f"native Trainer checkpoint preflight failed: {failures}")
    first = target_maps[0]
    assert isinstance(first, dict)
    validated = {str(name): Path(str(path)) for name, path in first.items()}
    if set(validated) == {"native_state"}:
        return validated["native_state"]
    return validated


def _collective_grpo_staging_root(
    output: Path,
    *,
    rank: int,
    world_size: int,
) -> Path:
    holder: list[object] = [None]
    if rank == 0:
        holder[0] = tempfile.mkdtemp(prefix=".grpo-checkpoint-stage-", dir=output)
    if world_size > 1:
        torch.distributed.broadcast_object_list(holder, src=0)
    if not isinstance(holder[0], str):
        raise RuntimeError("GRPO checkpoint staging root was not shared by rank zero")
    staging = Path(holder[0]).absolute()
    if staging.parent != output.absolute() or staging.is_symlink() or not staging.is_dir():
        raise RuntimeError("GRPO checkpoint staging root is outside the run directory")
    return staging


def _raise_collective_grpo_publication_error(
    error: str | None,
    *,
    rank: int,
    world_size: int,
) -> None:
    holder: list[object] = [error if rank == 0 else None]
    if world_size > 1:
        torch.distributed.broadcast_object_list(holder, src=0)
    if holder[0] is not None:
        raise RuntimeError(f"GRPO checkpoint no-clobber publication failed: {holder[0]}")


def _validate_grpo_resume_state_directory(
    payload: dict[str, object],
    *,
    checkpoint_path: Path,
) -> Path:
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("GRPO resume checkpoint lacks external optimizer state binding")
    state_path = optimizer.get("accelerate_state_path")
    if not isinstance(state_path, str) or not state_path:
        raise ValueError("GRPO resume checkpoint lacks its Accelerator state path")
    if payload.get("accelerate_state_path") != state_path:
        raise ValueError("GRPO resume checkpoint contains inconsistent Accelerator state paths")
    state_files = optimizer.get("files")
    resume_world_size = payload.get("world_size")
    state_files = validate_native_trainer_checkpoint_files(
        state_files,
        world_size=resume_world_size if type(resume_world_size) is int else 0,
    )
    state_sha256 = optimizer.get("content_sha256")
    if payload.get("accelerate_state_sha256") != state_sha256:
        raise ValueError("GRPO resume checkpoint contains inconsistent Accelerator state digests")
    state_root = checkpoint_path.resolve().parent.parent
    resolved_state_path = Path(state_path)
    validate_accelerator_state_directory(
        resolved_state_path,
        expected_run_root=state_root,
        expected_files=state_files,
        expected_sha256=str(state_sha256),
    )
    trainer_state_path = resolved_state_path / "trainer_state.json"
    if "trainer_state.json" not in state_files or not trainer_state_path.is_file():
        raise ValueError("GRPO resume directory is not a complete Trainer checkpoint")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if (
        not isinstance(trainer_state, dict)
        or type(trainer_state.get("global_step")) is not int
        or trainer_state.get("global_step") != payload.get("global_step")
    ):
        raise ValueError("GRPO Trainer state and bound checkpoint optimizer step differ")
    return resolved_state_path


def _wait_for_everyone(trainer: object) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        accelerator.wait_for_everyone()


def main(argv: list[str] | None = None) -> None:
    args, config = parse_cli("Run canonical TRL GRPO", argv)
    method = get_method_spec(str(config["experiment"]["name"]))
    if method.training_backend != TRL_GRPO_BACKEND or method.role not in {"anchor", "control"}:
        raise ValueError(f"run_grpo CLI does not execute registered method {method.method_id!r}")
    output = args.output or (Path(config["output_root"]) / "runs" / str(config["experiment"]["name"]))
    if not enforce_production_guard(
        config,
        dry_run=args.dry_run,
        confirm_production=args.confirm_production,
        output=output,
    ):
        return
    observed_trl_version = require_pinned_trl_version()

    model_config = config["model"]
    is_tiny = str(model_config["model_name_or_path"]).startswith("local/")
    prerequisite_evidence = {} if is_tiny else require_factorial_prerequisites(config)
    world_size, rank = _world_info()
    is_main_process = rank == 0
    resume_payload = None
    resume_state_path: Path | None = None
    resume_ancestry: list[str] = []
    initial_budget_consumed = 0
    initial_optimizer_steps = 0
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not isinstance(resume_payload, dict):
            raise ValueError("GRPO resume checkpoint payload must be a mapping")
        resume_state_path = _validate_grpo_resume_state_directory(
            resume_payload,
            checkpoint_path=args.resume,
        )
        resume_ancestry = extend_checkpoint_ancestry(
            resume_payload.get("resume_ancestry", []),
            args.resume,
        )
        budget_state = resume_payload.get("token_budget")
        if not isinstance(budget_state, dict):
            raise ValueError("GRPO resume checkpoint has no token-budget state")
        if int(budget_state.get("budget", -1)) != int(config["trainer"]["token_budget"]):
            raise ValueError("GRPO resume checkpoint token budget differs from config")
        initial_budget_consumed = int(budget_state["consumed"])
        initial_optimizer_steps = int(resume_payload["global_step"])
    initial_checkpoint_hash: str | None = None
    initial_checkpoint_path: Path | None = None
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
            config["trainer"].get(
                "token_budget_unit", "global_nonpadding_model_input_tokens_processed"
            )
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
        initial_checkpoint_path = Path(initial_checkpoint_value).absolute()
        load_checkpoint_into_hf_model(
            model,
            initial_checkpoint_path,
            expected_sha256=initial_checkpoint_hash,
        )
        model = move_model_to_local_cuda(model)
    if initial_checkpoint_hash is None:
        initial_checkpoint_hash = state_hash(model.state_dict())
    initial_parameters = (
        [parameter.detach().cpu().clone() for parameter in model.parameters() if parameter.requires_grad]
        if is_tiny
        else []
    )
    snapshot_callback: _InitialParameterSnapshotCallback | None = None

    task_config = config["task"]
    family = load_dataset_family(Path(str(task_config.get("dataset_family_path", ""))))
    family_train = family.examples("train")
    train_count = int(task_config.get("num_examples", len(family_train)))
    if train_count < 1 or train_count > len(family_train):
        raise ValueError("task.num_examples is outside the frozen train family split")
    examples = family_train[:train_count]
    family_validation = family.examples("validation")
    validation_limit = int(
        config["trainer"].get("validation_examples", len(family_validation))
    )
    if validation_limit < 1 or validation_limit > len(family_validation):
        raise ValueError("trainer.validation_examples is outside the frozen validation family split")
    validation_examples = family_validation[:validation_limit]
    validation_manifest_hash = str(family.manifest["sha256"])
    reward_name = str(config["experiment"]["reward"])
    matched_positive_rate = None
    random_reward_calibration_hash = None
    calibration: dict[str, object] | None = None
    if reward_name == "matched_random":
        if is_tiny:
            matched_positive_rate = 0.5
            calibration = {
                "artifact_kind": "tiny-frozen-random-reward-calibration",
                "positive_rate": 0.5,
                "individual_verifier_results_available_to_reward": False,
            }
            calibration["sha256"] = sha256_value(calibration)
            random_reward_calibration_hash = str(calibration["sha256"])
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
    formatted_training_prompts = format_model_prompts(
        [ProofGraphTask().render(example) for example in examples],
        tokenizer,
        model_config,
    )
    if [row["prompt"] for row in rows] != [
        prompt.model_facing_prompt for prompt in formatted_training_prompts
    ]:
        raise RuntimeError("GRPO rows differ from the registered model-facing prompt schedule")
    raw_prompt_schedule_hash, prompt_hash = prompt_schedule_hashes(
        formatted_training_prompts
    )
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
    run_id = (
        f"{config['experiment']['name']}-seed{config['seed']}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    train_dataset_hash = str(family.boundary("train")["examples_file_sha256"])
    validation_dataset_hash = str(
        family.boundary("validation")["examples_file_sha256"]
    )
    probe_manifest_binding_hash = str(
        prerequisite_evidence.get("probe_cohorts", {}).get(
            "manifest_hash",
            output_kl_probe_manifest_hash
            or sha256_value(
                {
                    "kind": "deterministic_smoke_probe",
                    "train_dataset": train_dataset_hash,
                    "seed": config["seed"],
                }
            ),
        )
    )
    dataset_hashes = {
        "train": train_dataset_hash,
        "validation": validation_dataset_hash,
        "family_manifest": str(family.manifest["sha256"]),
        "initial_checkpoint": initial_checkpoint_hash,
        "probe_manifest": probe_manifest_binding_hash,
        **({"validation_manifest": validation_manifest_hash} if validation_manifest_hash else {}),
        **{
            f"prerequisite_{name}": str(
                evidence.get("report_hash", evidence.get("manifest_hash"))
            )
            for name, evidence in prerequisite_evidence.items()
        },
    }
    teacher_config = config["teacher"]
    teacher_id = str(
        teacher_config.get("model_name_or_path", teacher_config.get("teacher_id", ""))
    )
    teacher_requested_revision = str(
        teacher_config.get("model_revision", teacher_config.get("teacher_revision", ""))
    )
    teacher_resolved_revision = str(
        teacher_config.get(
            "model_revision",
            teacher_config.get("resolved_teacher_commit", teacher_requested_revision),
        )
    )
    tokenizer_fingerprint = (
        loaded.tokenizer_hash if not is_tiny else sha256_value(tokenizer.get_vocab())
    )
    task_protocol = {
        key: value
        for key, value in task_config.items()
        if not any(marker in key.lower() for marker in ("path", "directory", "output_root"))
    }
    task_protocol.update(
        generator_version=ProofGraphTask.generator_version,
        label_semantics=ProofGraphTask.label_semantics,
    )
    full_parameter_training = all(parameter.requires_grad for parameter in model.parameters())
    config_input_hashes = {
        "prereg_path": sha256_file(Path(str(config["prereg_path"]))),
        "task.dataset_family_path": str(family.manifest["sha256"]),
    }
    initial_checkpoint_locator = str(
        config.get("production_safety", {}).get("initial_checkpoint_path", "")
    ).strip()
    if initial_checkpoint_locator:
        config_input_hashes["production_safety.initial_checkpoint_path"] = str(
            initial_checkpoint_hash
        )
    calibration_locator = str(
        config["experiment"].get("random_reward_calibration_path", "")
    ).strip()
    if not is_tiny and calibration is not None and calibration_locator:
        config_input_hashes["experiment.random_reward_calibration_path"] = str(
            calibration["sha256"]
        )
    prerequisite_locator_bindings = {
        "full_readiness": ("production_safety.readiness_report", "report_hash"),
        "anti_shortcut": ("anti_shortcut.report_path", "report_hash"),
        "probe_cohorts": (
            "production_safety.probe_cohort_manifest",
            "manifest_hash",
        ),
    }
    for evidence_name, (locator_name, hash_name) in prerequisite_locator_bindings.items():
        evidence = prerequisite_evidence.get(evidence_name)
        if isinstance(evidence, dict) and evidence.get(hash_name):
            config_input_hashes[locator_name] = str(evidence[hash_name])
    execution_context = {
        "entrypoint": "run_grpo",
        "output": str(Path(output).absolute()),
        "backend": method.training_backend,
        "world_size": world_size,
        "resume": str(args.resume.absolute()) if args.resume is not None else None,
    }
    config_binding = bind_config(
        config,
        input_artifact_hashes=config_input_hashes,
        execution_context=execution_context,
    )
    experiment_binding = build_experiment_binding(
        config=config,
        config_binding=config_binding,
        implementation_commit=git_output(["rev-parse", "HEAD"]) or "unavailable",
        implementation_dirty=bool(git_output(["status", "--porcelain"]) or ""),
        model_resolved_revision=resolved_model_commit,
        tokenizer_resolved_revision=resolved_tokenizer_commit,
        tokenizer_fingerprint=tokenizer_fingerprint,
        teacher_resolved_revision=teacher_resolved_revision,
        initial_checkpoint_sha256=initial_checkpoint_hash,
        train_dataset_sha256=train_dataset_hash,
        validation_dataset_sha256=validation_dataset_hash,
        model_facing_prompt_schedule_sha256=prompt_hash,
        probe_manifest_sha256=probe_manifest_binding_hash,
        task_protocol=task_protocol,
        optimizer_spec={
            "class": "torch.optim.AdamW",
            "trl_optim": "adamw_torch",
            "learning_rate": settings.learning_rate,
        },
        scheduler_spec={
            "class": "transformers.get_scheduler",
            "name": "linear",
            "warmup_ratio": 0.0,
            "warmup_steps": 0,
        },
        resolved_batch_contract={
            **batch_contract,
            "current_policy": asdict(REGISTERED_CURRENT_POLICY),
        },
        full_parameter_training=full_parameter_training,
        reward_calibration=calibration,
    )
    manifest = RunManifest(
        run_id=run_id,
        experiment_cell=str(config["experiment"]["name"]),
        seed=int(config["seed"]),
        model_id=str(model_config["model_name_or_path"]),
        model_revision=str(model_config["model_revision"]),
        tokenizer_id=str(model_config["tokenizer_name_or_path"]),
        tokenizer_revision=str(model_config["tokenizer_revision"]),
        resolved_model_commit=resolved_model_commit,
        resolved_tokenizer_commit=resolved_tokenizer_commit,
        dataset_hashes=dataset_hashes,
        rollout_bank_hash=source_hash,
        prompt_schedule_hash=prompt_hash,
        experiment_binding=experiment_binding.to_payload(),
        experiment_binding_sha256=experiment_binding.scientific_sha256,
        factorial_design_sha256=experiment_binding.factorial_design_sha256,
        config_binding=config_binding.as_dict(),
        resolved_config_sha256=config_binding.resolved_config_sha256,
        scientific_config_sha256=config_binding.scientific_config_sha256,
        execution_config_sha256=config_binding.execution_config_sha256,
        execution_context=execution_context,
        raw_prompt_schedule_hash=raw_prompt_schedule_hash,
        model_facing_prompt_schedule_hash=prompt_hash,
        prompt_protocol=experiment_binding.prompt_protocol,
        enable_thinking=experiment_binding.enable_thinking,
        chat_template_sha256=experiment_binding.chat_template_sha256,
        tokenizer_fingerprint=experiment_binding.tokenizer_fingerprint,
        protocol_track=str(config.get("protocol_track", "core_v2")),
        artifact_namespace=str(model_config.get("artifact_namespace", "legacy")),
        protocol_teacher_revision=str(config["teacher"].get("model_revision", "unbound")),
        teacher_id=teacher_id,
        teacher_revision=teacher_requested_revision,
        resolved_teacher_commit=teacher_resolved_revision,
        token_budget=int(config["trainer"]["token_budget"]),
        token_budget_unit=str(
            config["trainer"].get(
                "token_budget_unit", "global_nonpadding_model_input_tokens_processed"
            )
        ),
        git_commit=experiment_binding.implementation_commit,
        dirty_working_tree=experiment_binding.implementation_dirty,
        resume_ancestry=resume_ancestry,
    )
    validate_run_manifest_experiment_binding(asdict(manifest))
    if resume_payload is not None and (
        resume_payload.get("experiment_binding_sha256")
        != experiment_binding.scientific_sha256
        or resume_payload.get("factorial_design_sha256")
        != experiment_binding.factorial_design_sha256
    ):
        raise ValueError("GRPO resume checkpoint ExperimentBinding/design hashes changed")
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
    _wait_for_everyone(trainer)
    accelerator = getattr(trainer, "accelerator", None)
    if not is_tiny:
        if accelerator is None:
            raise RuntimeError("production GRPO requires an Accelerator-managed trainer")
    if is_main_process:
        initialize_run_directory(
            output,
            config,
            config_binding,
            manifest,
            require_git=not is_tiny,
        )
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
    if accelerator is not None:
        snapshot_callback = _InitialParameterSnapshotCallback(
            accelerator=accelerator,
            root=Path(output) / "initial_parameter_snapshot",
            is_main_process=is_main_process,
        )
        trainer.add_callback(snapshot_callback)
    if resume_state_path is None:
        result = trainer.train()
    else:
        result = trainer.train(resume_from_checkpoint=str(resume_state_path))
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
    final_model_state_hash = (
        state_hash(final_state) if is_main_process and final_state else None
    )
    rank_local_parameter_summaries = _gather_rank_objects(
        _rank_local_parameter_summary(trainer.model, rank=rank),
        world_size=world_size,
    )
    consensus_rows = _validate_distributed_training_consensus(
        _gather_rank_objects(
            {
                "rank": rank,
                "global_step": global_step,
                "token_budget": token_budget_state,
                "experiment_binding_sha256": experiment_binding.scientific_sha256,
                "current_policy_contract": asdict(REGISTERED_CURRENT_POLICY),
                "optimizer_boundary": "completed_optimizer_step",
            },
            world_size=world_size,
        ),
        world_size=world_size,
    )
    distributed_consistency_passed = True
    optimizer_state = trainer.optimizer.state_dict() if accelerator is None else None
    scheduler_state = trainer.lr_scheduler.state_dict()
    update_norm = None
    if is_main_process:
        if snapshot_callback is None:
            update_norm = _parameter_update_norm(initial_parameters, trained_model)
        else:
            if snapshot_callback.snapshot is None:
                raise RuntimeError("GRPO did not capture its post-resume update-norm baseline")
            if not final_state:
                raise RuntimeError("main GRPO rank did not receive the full FSDP state dict")
            update_norm = _streaming_parameter_update_norm(
                snapshot_callback.snapshot,
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
    accelerate_state_path = Path(output).absolute() / f"checkpoint-{global_step}"
    evidence_path = Path(output) / "grpo_update_evidence.json"
    _wait_for_everyone(trainer)
    accelerate_state_hashes = None
    accelerate_state_sha256 = None
    native_staging_checkpoint: Path | None = None
    if accelerator is not None:
        accelerate_state_path = _native_trainer_checkpoint_path(
            trainer,
            output=Path(output),
            global_step=global_step,
        )
    preflight_rows = _gather_rank_objects(
        _native_checkpoint_preflight_report(
            accelerate_state_path if accelerator is not None else None,
            expected_run_root=Path(output),
            rank=rank,
            metadata_path=checkpoint_path,
        ),
        world_size=world_size,
    )
    validated_targets = _validate_native_checkpoint_preflight_reports(
        preflight_rows,
        world_size=world_size,
    )
    expected_targets = {"metadata": checkpoint_path.absolute()}
    if accelerator is not None:
        expected_targets["native_state"] = accelerate_state_path.absolute()
    if validated_targets != expected_targets:
        raise RuntimeError("GRPO checkpoint preflight changed its targets")
    if accelerator is not None:
        _wait_for_everyone(trainer)
        save_trainer_checkpoint = getattr(trainer, "_save_checkpoint", None)
        if not callable(save_trainer_checkpoint):
            raise RuntimeError("production GRPO trainer cannot emit a resumable checkpoint")
        staging_root = _collective_grpo_staging_root(
            Path(output),
            rank=rank,
            world_size=world_size,
        )
        trainer_args = getattr(trainer, "args", None)
        original_output_dir = getattr(trainer_args, "output_dir", None)
        if not isinstance(original_output_dir, str):
            raise RuntimeError("production GRPO trainer has no mutable output directory")
        trainer_args.output_dir = str(staging_root)
        try:
            save_trainer_checkpoint(trainer.model, trial=None)
        finally:
            trainer_args.output_dir = original_output_dir
        _wait_for_everyone(trainer)
        native_staging_checkpoint = staging_root / f"{PREFIX_CHECKPOINT_DIR}-{global_step}"
        if not native_staging_checkpoint.is_dir():
            raise RuntimeError("native Trainer staging checkpoint directory was not created")
        if not (native_staging_checkpoint / "trainer_state.json").is_file():
            raise RuntimeError("native Trainer checkpoint omitted trainer_state.json")
        accelerate_state_hashes = accelerator_state_file_hashes(
            native_staging_checkpoint,
            expected_run_root=Path(output),
        )
        validate_native_trainer_checkpoint_files(
            accelerate_state_hashes,
            world_size=world_size,
        )
        accelerate_state_sha256 = sha256_value(accelerate_state_hashes)
    initial_snapshot = snapshot_callback.snapshot if snapshot_callback is not None else None
    snapshot_manifest_path = Path(output) / "initial_parameter_snapshot" / "manifest.json"
    snapshot_manifest_file_sha256 = (
        sha256_file(snapshot_manifest_path)
        if is_main_process and snapshot_manifest_path.is_file()
        else None
    )
    update_norm_baseline_checkpoint_sha256 = (
        sha256_file(args.resume) if args.resume is not None else initial_checkpoint_hash
    )
    update_norm_baseline_checkpoint_path = (
        Path(args.resume).absolute()
        if args.resume is not None
        else initial_checkpoint_path
    )
    checkpoint_payload = None
    if is_main_process:
        checkpoint_payload = {
                "model": _cpu_tree(final_state),
                "optimizer": (
                    _cpu_tree(optimizer_state)
                    if accelerator is None
                    else {
                        "accelerate_state_path": str(accelerate_state_path.resolve()),
                        "files": accelerate_state_hashes,
                        "content_sha256": accelerate_state_sha256,
                    }
                ),
                "scheduler": _cpu_tree(scheduler_state),
                "rng": RNGState.capture().as_dict(),
                "resolved_config": config,
                "global_step": global_step,
                "world_size": world_size,
                "initial_checkpoint_sha256": initial_checkpoint_hash,
                "git_commit": experiment_binding.implementation_commit,
                "implementation_dirty": experiment_binding.implementation_dirty,
                "experiment_binding_sha256": experiment_binding.scientific_sha256,
                "factorial_design_sha256": experiment_binding.factorial_design_sha256,
                "method_spec_sha256": method.scientific_sha256,
                "trl_version": observed_trl_version,
                "trl_batch_contract": TRL_GRPO_BATCH_CONTRACT,
                "current_policy_contract": asdict(REGISTERED_CURRENT_POLICY),
                "token_budget": token_budget_state,
                "accelerate_state_path": (
                    str(accelerate_state_path.resolve()) if accelerator is not None else None
                ),
                "accelerate_state_sha256": accelerate_state_sha256,
                "resume_ancestry": resume_ancestry,
                "update_norm_baseline_checkpoint_sha256": (
                    update_norm_baseline_checkpoint_sha256
                ),
                "update_norm_baseline_checkpoint_path": (
                    str(update_norm_baseline_checkpoint_path)
                    if update_norm_baseline_checkpoint_path is not None
                    else None
                ),
                "initial_parameter_snapshot_sha256": (
                    initial_snapshot.get("sha256")
                    if isinstance(initial_snapshot, dict)
                    else None
                ),
            }
    publication_error = None
    if is_main_process:
        assert checkpoint_payload is not None
        try:
            if native_staging_checkpoint is not None:
                publish_path_no_clobber(
                    native_staging_checkpoint,
                    accelerate_state_path,
                )
                with suppress(OSError):
                    native_staging_checkpoint.parent.rmdir()
            publish_torch_once(checkpoint_path, checkpoint_payload)
        except BaseException as error:  # every rank receives this before proceeding
            publication_error = f"{type(error).__name__}: {error}"
    _raise_collective_grpo_publication_error(
        publication_error,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process:
        checkpoint_sha256 = sha256_file(checkpoint_path)
        reward_artifact_hash = random_reward_calibration_hash or sha256_value(
            {"reward": reward_name, "seed": config["seed"]}
        )
        evidence = {
            "format_version": 2,
            "prereg_version": str(config.get("protocol_track", "core_v2")),
            "generator_version": ProofGraphTask.generator_version,
            "label_semantics": ProofGraphTask.label_semantics,
            "backend": TRL_GRPO_BACKEND,
            "backend_version": observed_trl_version,
            "backend_batch_contract": TRL_GRPO_BATCH_CONTRACT,
            "method_spec_sha256": method.scientific_sha256,
            "experiment_binding_sha256": experiment_binding.scientific_sha256,
            "factorial_design_sha256": experiment_binding.factorial_design_sha256,
            "resolved_batch_contract_sha256": (
                experiment_binding.resolved_batch_contract_sha256
            ),
            "current_policy_contract": asdict(REGISTERED_CURRENT_POLICY),
            "trl_train_called": True,
            "main_process_only_writes": True,
            "optimizer_steps": global_step,
            "token_budget": token_budget_state,
            "parameter_update_norm": update_norm,
            "parameter_update_norm_method": (
                "one-tensor-at-a-time-disk-snapshot"
                if snapshot_callback is not None
                else "in-memory-trainable-parameter-snapshot"
            ),
            "parameters_changed": bool(update_norm and update_norm > 0.0),
            "reward": reward_name,
            "reward_artifact_hash": reward_artifact_hash,
            "random_reward_calibration_hash": random_reward_calibration_hash,
            "tiny_smoke_matched_positive_rate": matched_positive_rate,
            "settings": settings.__dict__,
            "batch_contract": batch_contract,
            "world_size": world_size,
            "checkpoint": str(checkpoint_path.resolve()),
            "initial_checkpoint_hash": initial_checkpoint_hash
            or state_hash({str(index): tensor for index, tensor in enumerate(initial_parameters)}),
            "final_checkpoint_hash": checkpoint_sha256,
            "git_commit": experiment_binding.implementation_commit,
            "implementation_dirty": experiment_binding.implementation_dirty,
            "final_model_state_hash": final_model_state_hash,
            "rank_local_parameter_summaries": rank_local_parameter_summaries,
            "distributed_training_consensus": consensus_rows,
            "distributed_consistency_passed": distributed_consistency_passed,
            "accelerate_state_path": (
                str(accelerate_state_path.resolve()) if accelerator is not None else None
            ),
            "accelerate_state_hashes": accelerate_state_hashes,
            "accelerate_state_sha256": accelerate_state_sha256,
            "resume_ancestry": resume_ancestry,
            "update_norm_baseline_checkpoint_sha256": (
                update_norm_baseline_checkpoint_sha256
            ),
            "update_norm_baseline_checkpoint_path": (
                str(update_norm_baseline_checkpoint_path)
                if update_norm_baseline_checkpoint_path is not None
                else None
            ),
            "initial_parameter_snapshot": initial_snapshot,
            "initial_parameter_snapshot_manifest_path": (
                str(snapshot_manifest_path.resolve()) if snapshot_manifest_path.is_file() else None
            ),
            "initial_parameter_snapshot_manifest_file_sha256": (
                snapshot_manifest_file_sha256
            ),
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
        manifest.dataset_hashes["grpo_update_evidence"] = str(evidence["sha256"])
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
        manifest.final_checkpoint_path = str(checkpoint_path.resolve())
        manifest.final_checkpoint_sha256 = checkpoint_sha256
        manifest.resume_ancestry = list(resume_ancestry)
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
