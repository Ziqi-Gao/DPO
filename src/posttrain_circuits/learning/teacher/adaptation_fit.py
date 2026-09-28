"""Bounded teacher-only LoRA fitting with explicit replicated-DDP semantics.

Production plans admit exactly four ranks. Tiny CPU test plans are deliberately
marked and cannot run on CUDA. Checkpoints are locally fsynced and atomically
completed; the caller's hook must publish them to verified persistent storage.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.learning.teacher.adaptation import (
    LORA_MODULES,
    PREFLIGHT,
    _tensor_digest,
    collate,
    response_sequence_loss,
)


@dataclass(frozen=True)
class FitPlan:
    mode: str = "fit"
    world_size: int = 4
    train_examples: int = 8192
    epochs: int = 4
    global_batch_size: int = 64
    microbatch_size: int = 1
    seed: int = 271828
    lora_rank: int = 32
    lora_alpha: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 16
    checkpoint_steps: tuple[int, ...] = (128, 256, 384, 512)

    @classmethod
    def preflight(cls) -> FitPlan:
        return cls(mode="preflight", train_examples=512, epochs=1, warmup_steps=0, checkpoint_steps=(8,))

    @property
    def steps(self) -> int:
        return self.train_examples * self.epochs // self.global_batch_size

    def validate(self, *, device_type: str) -> None:
        if self.mode not in {"fit", "preflight", "cpu_test"}:
            raise ValueError("unknown teacher fit plan")
        if (
            self.global_batch_size != 64
            or self.microbatch_size != 1
            or self.world_size not in {2, 4}
            or self.train_examples < 64
            or self.train_examples % 64
            or self.epochs < 1
            or self.seed != 271828
            or self.learning_rate != 1e-4
            or self.weight_decay != 0.01
            or self.max_grad_norm != 1.0
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
            or not self.checkpoint_steps
            or self.checkpoint_steps[-1] != self.steps
            or self.checkpoint_steps[0] < 1
        ):
            raise ValueError("invalid teacher fit batch, optimizer, or checkpoint contract")
        if self.mode == "cpu_test":
            if device_type != "cpu" or self.lora_rank < 1 or self.lora_alpha < 1:
                raise ValueError("test fit plans require a CPU model")
            if not 0 <= self.warmup_steps < self.steps:
                raise ValueError("invalid test warmup")
        else:
            expected = FitPlan() if self.mode == "fit" else FitPlan.preflight()
            if self != expected:
                raise ValueError("production teacher fit plan differs from frozen settings")


def make_fit_examples(task_configuration: dict[str, Any], plan: FitPlan) -> dict[str, list[Any]]:
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.rendering import render_target
    from posttrain_circuits.datasets.proofgraph.splits import assert_split_isolation, build_split

    task = ProofGraphTask()
    splits = {
        "teacher_fit": build_split(task, "train", plan.train_examples, 70_000_042, task_configuration),
        "teacher_dev": build_split(
            task, "validation", 512 if plan.mode == "fit" else 32, 70_000_042, task_configuration
        ),
    }
    assert_split_isolation(splits)
    for examples in splits.values():
        for example in examples:
            if task.verify(example, task.parse_response(render_target(example))).reward != 1.0:
                raise ValueError("teacher fit canonical target does not verify")
    return splits


def encode_fit_example(example: Any, tokenizer: Any, teacher_config: dict[str, Any]) -> dict[str, Any]:
    """Keep all reviewed fit rows; the separate fit prefix limit is1280."""
    from posttrain_circuits.datasets.proofgraph.generation import ProofGraphTask
    from posttrain_circuits.datasets.proofgraph.rendering import render_target
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    prompt = format_model_prompt(ProofGraphTask().render(example), tokenizer, teacher_config)
    prefix = tokenizer.encode(prompt.model_facing_prompt, add_special_tokens=False)
    target_text = render_target(example)
    target = tokenizer.encode(target_text, add_special_tokens=False)
    if tokenizer.eos_token_id is None or not prefix or not target:
        raise ValueError("teacher fit requires nonempty prefix/target and EOS")
    target = [*target, tokenizer.eos_token_id]
    if len(prefix) > 1280 or len(target) > 256 or len(prefix) + len(target) > 1536:
        raise ValueError("teacher fit example exceeds the reviewed1280/256/1536 token envelope")
    return {
        "example_id": example.example_id,
        "input_ids": [*prefix, *target],
        "labels": [-100] * len(prefix) + target,
        "prefix_length": len(prefix),
        "response_length": len(target),
        "prompt_sha256": prompt.model_facing_prompt_sha256,
        "target_sha256": sha256_value(target_text),
    }


def learning_rate_trace(plan: FitPlan) -> tuple[float, ...]:
    if plan.mode == "preflight" or plan.warmup_steps == 0:
        return (plan.learning_rate,) * plan.steps
    return tuple(
        plan.learning_rate * step / plan.warmup_steps
        if step <= plan.warmup_steps
        else plan.learning_rate
        * 0.5
        * (1 + math.cos(math.pi * (step - plan.warmup_steps) / (plan.steps - plan.warmup_steps)))
        for step in range(1, plan.steps + 1)
    )


def epoch_order(plan: FitPlan, epoch: int) -> list[int]:
    import torch

    if not 0 <= epoch < plan.epochs:
        raise ValueError("epoch outside teacher fit plan")
    return torch.randperm(
        plan.train_examples, generator=torch.Generator().manual_seed(plan.seed + epoch)
    ).tolist()


def rank_window(global_indices: list[int], *, rank: int, world_size: int) -> list[int]:
    if len(global_indices) != 64 or len(set(global_indices)) != 64 or world_size not in {2, 4}:
        raise ValueError("invalid exact global teacher window")
    if not 0 <= rank < world_size:
        raise ValueError("invalid teacher rank")
    return global_indices[rank::world_size]


def _sync_error(error: BaseException | None, group: Any, phase: str) -> None:
    import torch.distributed as dist

    text = None if error is None else f"{type(error).__name__}: {error}"[:1000]
    errors: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, text, group=group)
    if any(value is not None for value in errors):
        raise RuntimeError(f"teacher fit {phase} failed across ranks: {errors}") from error


def _shared_call(function: Callable[[], Any], group: Any, phase: str) -> Any:
    error, result = None, None
    try:
        result = function()
    except BaseException as caught:
        error = caught
    _sync_error(error, group, phase)
    return result


def _agree(value: Any, group: Any, phase: str) -> None:
    import torch.distributed as dist

    values: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    if any(item != values[0] for item in values):
        raise RuntimeError(f"teacher fit ranks disagree on {phase}")


def _rng_state(device: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: Any) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _save_tensor_file(path: Path, value: Any) -> None:
    import torch

    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_fsynced(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _adapter_digest(model: Any) -> str:
    return _tensor_digest((name, value) for name, value in model.named_parameters() if value.requires_grad)


def _state_digest(value: Any) -> str:
    """Hash nested trainer/RNG state without materializing tensor-value lists."""
    import torch

    result = hashlib.sha256()

    def visit(row):
        if torch.is_tensor(row):
            result.update(_tensor_digest([("tensor", row)]).encode())
        elif isinstance(row, dict):
            result.update(b"dict")
            for key in sorted(row, key=repr):
                result.update(repr(key).encode())
                visit(row[key])
        elif isinstance(row, tuple | list):
            result.update(type(row).__name__.encode())
            for item in row:
                visit(item)
        else:
            result.update(json.dumps(row, sort_keys=True, allow_nan=False).encode())

    visit(value)
    return result.hexdigest()


def _load_fit_state(
    path: Path, manifest: dict[str, Any], model: Any, optimizer: Any, *, rank: int, device: Any
) -> None:
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    state = torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
    metadata = manifest["metadata"]
    if state["scheduler"] != {
        "next_update": metadata["completed_updates"],
        "lr_trace_sha256": metadata["lr_trace_sha256"],
    }:
        raise ValueError("teacher checkpoint scheduler differs")
    expected_adapter = get_peft_model_state_dict(model)
    if state["adapter"].keys() != expected_adapter.keys() or any(
        state["adapter"][key].shape != tensor.shape or state["adapter"][key].dtype != tensor.dtype
        for key, tensor in expected_adapter.items()
    ):
        raise ValueError("teacher checkpoint adapter tensors differ")
    rng = torch.load(path / f"rng-rank{rank}.pt", map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    _restore_rng(rng, device)


def _verify_resume_roundtrip(
    path: Path, model: Any, optimizer: Any, *, expected: dict[str, Any], group: Any, device: Any
) -> dict[str, str]:
    """Real destructive perturb/reload, restricted to the operational preflight."""
    import numpy as np
    import torch
    import torch.distributed as dist

    manifest = _shared_call(
        lambda: validate_fit_checkpoint(path, expected), group, "preflight checkpoint readback"
    )
    before = {
        "adapter": _adapter_digest(model),
        "optimizer": _state_digest(optimizer.state_dict()),
        "rng": _state_digest(_rng_state(device)),
    }

    def perturb():
        with torch.no_grad():
            for value in model.parameters():
                if value.requires_grad:
                    value.add_(1.0)
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()
        random.random()
        np.random.random()
        torch.rand(1)
        if device.type == "cuda":
            torch.rand(1, device=device)

    _shared_call(perturb, group, "preflight state perturbation")
    _shared_call(
        lambda: _load_fit_state(path, manifest, model, optimizer, rank=dist.get_rank(group), device=device),
        group,
        "preflight real checkpoint reload",
    )
    after = {
        "adapter": _adapter_digest(model),
        "optimizer": _state_digest(optimizer.state_dict()),
        "rng": _state_digest(_rng_state(device)),
    }
    _shared_call(
        lambda: _unchanged(before, after, "preflight restored state differs"),
        group,
        "preflight resume verification",
    )
    return after


def save_fit_checkpoint(
    root: Path,
    model: Any,
    optimizer: Any,
    *,
    metadata: dict[str, Any],
    group: Any,
    device: Any,
) -> tuple[Path, dict[str, Any]]:
    """All ranks publish one complete checkpoint at an optimizer boundary."""
    import torch.distributed as dist
    from peft import get_peft_model_state_dict

    rank = dist.get_rank(group)
    _agree(metadata, group, "checkpoint cursor")
    _agree(_adapter_digest(model), group, "checkpoint adapter")
    _agree(_state_digest(optimizer.state_dict()), group, "checkpoint optimizer")
    name = f"step-{metadata['completed_updates']:06d}"
    temporary = [None]

    def prepare():
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            if (root / name).exists():
                raise ValueError("teacher checkpoint already exists")
            temporary[0] = str(root / f".{name}.{uuid.uuid4().hex}.tmp")
            Path(temporary[0]).mkdir()

    _shared_call(prepare, group, "checkpoint prepare")
    dist.broadcast_object_list(temporary, src=0, group=group)
    staging = Path(temporary[0])

    def save_local():
        _save_tensor_file(staging / f"rng-rank{rank}.pt", _rng_state(device))
        _copy_fsynced(
            root.parent / f"fit-samples-rank{rank}.jsonl", staging / f"fit-samples-rank{rank}.jsonl"
        )
        if rank == 0:
            _copy_fsynced(root.parent / "train-metrics.jsonl", staging / "train-metrics.jsonl")
            model.save_pretrained(staging / "adapter", safe_serialization=True)
            adapter = {
                key: tensor.detach().cpu().clone() for key, tensor in get_peft_model_state_dict(model).items()
            }
            _save_tensor_file(
                staging / "trainer.pt",
                {
                    "adapter": adapter,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": {
                        "next_update": metadata["completed_updates"],
                        "lr_trace_sha256": metadata["lr_trace_sha256"],
                    },
                },
            )

    _shared_call(save_local, group, "checkpoint files")
    final = root / name
    manifest: list[Any] = [None]

    def publish():
        if rank == 0:
            files = []
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    files.append(
                        {
                            "path": path.relative_to(staging).as_posix(),
                            "size": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
            payload = {
                "artifact_kind": "teacher_adapter_fit_checkpoint",
                "format_version": 1,
                "accepted_science": False,
                "formal_teacher_accepted": False,
                "metadata": metadata,
                "files": files,
            }
            manifest[0] = {**payload, "sha256": sha256_value(payload)}
            _write_json(staging / "manifest.json", manifest[0])
            _fsync_directory(staging / "adapter")
            _fsync_directory(staging)
            if final.exists():
                raise ValueError("teacher checkpoint destination appeared during publication")
            os.rename(staging, final)
            _fsync_directory(root)

    _shared_call(publish, group, "checkpoint completion")
    dist.broadcast_object_list(manifest, src=0, group=group)
    return final, manifest[0]


def validate_fit_checkpoint(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Validate identity and all bytes before any state or RNG is restored."""
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("teacher checkpoint has a symbolic path")
    completion = path / "manifest.json"
    if completion.is_symlink() or not completion.is_file() or completion.stat().st_size > 1024 * 1024:
        raise ValueError("teacher checkpoint completion is missing or exceeds bound")
    value = json.loads(completion.read_bytes())
    if value.get("sha256") != sha256_value({key: row for key, row in value.items() if key != "sha256"}):
        raise ValueError("teacher checkpoint completion hash differs")
    if (
        value.get("artifact_kind") != "teacher_adapter_fit_checkpoint"
        or value.get("accepted_science") is not False
        or value.get("formal_teacher_accepted") is not False
    ):
        raise ValueError("teacher checkpoint scope differs")
    metadata = value["metadata"]
    for key, row in expected.items():
        if metadata.get(key) != row:
            raise ValueError(f"teacher checkpoint identity differs: {key}")
    count = metadata["completed_updates"]
    steps_per_epoch = metadata["train_examples"] // 64
    if (
        type(count) is not int
        or not 0 < count <= metadata["total_updates"]
        or metadata["completed_samples"] != count * 64
        or metadata["consumed_tokens"] != metadata["cumulative_token_counts"][count]
        or metadata["epoch"] != count // steps_per_epoch
        or metadata["next_epoch_offset"] != (count % steps_per_epoch) * 64
    ):
        raise ValueError("teacher checkpoint cursor is partial or inconsistent")
    required = {
        "trainer.pt",
        "train-metrics.jsonl",
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        *(f"rng-rank{rank}.pt" for rank in range(metadata["world_size"])),
        *(f"fit-samples-rank{rank}.jsonl" for rank in range(metadata["world_size"])),
    }
    names = [row["path"] for row in value["files"]]
    if (
        not required <= set(names)
        or set(names) - required - {"adapter/README.md"}
        or len(names) != len(set(names))
    ):
        raise ValueError("teacher checkpoint state files differ")
    allowed_root = {Path(name).parts[0] for name in names} | {
        "manifest.json",
        "merged",
        "dense-manifest.json",
        "dev-capability.json",
        "eval",
    }
    if any(item.name not in allowed_root for item in path.iterdir()) or any(
        item.is_symlink() for item in path.rglob("*")
    ):
        raise ValueError("teacher checkpoint has unknown entries or symlinks")
    extras = {"merged", "dense-manifest.json", "dev-capability.json", "eval"}
    actual_state_files = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "manifest.json" and item.relative_to(path).parts[0] not in extras
    }
    if actual_state_files != set(names):
        raise ValueError("teacher checkpoint has unbound state files")
    for row in value["files"]:
        file = path / row["path"]
        if any(item.is_symlink() for item in (file, *file.parents)) or not file.is_file():
            raise ValueError("teacher checkpoint state file is not a regular real path")
        if file.stat().st_size != row["size"] or sha256_file(file) != row["sha256"]:
            raise ValueError("teacher checkpoint file differs")
    if (
        metadata["train_metrics_path"] != "train-metrics.jsonl"
        or sha256_file(path / "train-metrics.jsonl") != metadata["train_metrics_sha256"]
    ):
        raise ValueError("teacher checkpoint immutable metrics prefix differs")
    return value


def run_adapter_fit(
    base_model: Any,
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    output: Path,
    *,
    identity: dict[str, Any],
    observe: Any,
    plan: FitPlan | None = None,
    control_group: Any = None,
    training_group: Any = None,
    resume_from: Path | None = None,
    checkpoint_hook: Callable[[Path, dict[str, Any]], Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run the same global sequence mean on every rank; never train on dev data.

    Groups are created by the orchestration layer. The control group is Gloo;
    training is NCCL on GPU or Gloo in CPU tests. The hook runs on every rank,
    receives only saved checkpoint identity, and must not mutate training state.
    """
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel

    plan = FitPlan() if plan is None else plan
    if not dist.is_initialized() or dist.get_backend(control_group) != "gloo":
        raise ValueError("teacher fitting requires initialized Gloo control group")
    rank, world = dist.get_rank(control_group), dist.get_world_size(control_group)
    device = next(base_model.parameters()).device

    def validate_inputs():
        plan.validate(device_type=device.type)
        if (
            world != plan.world_size
            or dist.get_world_size(training_group) != world
            or dist.get_rank(training_group) != rank
        ):
            raise ValueError("teacher allocation world size differs")
        if dist.get_backend(training_group) != ("nccl" if device.type == "cuda" else "gloo"):
            raise ValueError("teacher training group backend differs from its model device")
        if (
            not {"base_revision", "tokenizer_sha256", "fit_dataset_sha256", "protocol_sha256"}
            <= identity.keys()
        ):
            raise ValueError("teacher fit identity is incomplete")
        if identity["base_revision"] != PREFLIGHT["base_revision"]:
            raise ValueError("teacher fit base revision differs")
        if any(
            not isinstance(identity[key], str) or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None
            for key in ("tokenizer_sha256", "fit_dataset_sha256", "protocol_sha256")
        ):
            raise ValueError("teacher fit content identities must be SHA-256")
        if len(train_rows) != plan.train_examples or len({row["example_id"] for row in train_rows}) != len(
            train_rows
        ):
            raise ValueError("teacher fit requires complete unique fit rows")
        for row in train_rows:
            tokens, labels = row["input_ids"], row["labels"]
            prefix = next((index for index, value in enumerate(labels) if value != -100), len(labels))
            if (
                not 0 < prefix <= 1280
                or not 0 < len(tokens) - prefix <= 256
                or len(tokens) != len(labels)
                or len(tokens) > 1536
                or labels[prefix:] != tokens[prefix:]
                or any(value < 0 for value in tokens)
                or tokens[-1] != tokenizer.eos_token_id
            ):
                raise ValueError("teacher fit response or token envelope differs")
        if output.is_symlink() or not output.is_dir():
            raise ValueError("teacher fit output must be a real existing directory")

    _shared_call(validate_inputs, control_group, "input validation")
    trace = learning_rate_trace(plan)
    orders = [epoch_order(plan, epoch) for epoch in range(plan.epochs)]
    cumulative_tokens = [0]
    for order in orders:
        for offset in range(0, plan.train_examples, 64):
            cumulative_tokens.append(
                cumulative_tokens[-1]
                + sum(len(train_rows[index]["input_ids"]) for index in order[offset : offset + 64])
            )
    expected = {
        "identity": identity,
        "plan": asdict(plan),
        "plan_sha256": sha256_value(asdict(plan)),
        "rows_sha256": sha256_value(train_rows),
        "world_size": world,
        "train_examples": plan.train_examples,
        "total_updates": plan.steps,
        "lr_trace_sha256": sha256_value(trace),
        "epoch_order_sha256": [sha256_value(order) for order in orders],
        "cumulative_token_counts": cumulative_tokens,
    }
    # JSON roundtrip gives exactly the identity that checkpoint validation reads.
    expected = json.loads(json.dumps(expected))
    _agree(expected, control_group, "fit plan and data")
    resume = _shared_call(
        lambda: validate_fit_checkpoint(resume_from, expected) if resume_from else None,
        control_group,
        "resume identity",
    )
    random.seed(plan.seed)
    np.random.seed(plan.seed)
    torch.manual_seed(plan.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(plan.seed)
    _shared_call(lambda: observe.phase("teacher_adapter_setup"), control_group, "setup observation")

    def install_adapter():
        base_model.config.use_cache = False
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=plan.lora_rank,
                lora_alpha=plan.lora_alpha,
                lora_dropout=0.0,
                target_modules=list(LORA_MODULES),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
        modules = [name for name, value in model.named_modules() if hasattr(value, "lora_A")]
        if (
            not trainable
            or any("lora_" not in name for name, _ in trainable)
            or len(modules) != base_model.config.num_hidden_layers * len(LORA_MODULES)
            or {name.rsplit(".", 1)[-1] for name in modules} != set(LORA_MODULES)
        ):
            raise ValueError("teacher adapter parameter or module coverage differs")
        return model, trainable, modules

    model, trainable, modules = _shared_call(install_adapter, control_group, "adapter installation")
    _agree(_adapter_digest(model), control_group, "initial adapter")
    frozen = _shared_call(
        lambda: _tensor_digest(
            (name, value) for name, value in model.named_parameters() if not value.requires_grad
        ),
        control_group,
        "frozen base digest",
    )
    _agree(frozen, control_group, "frozen base")
    parameters = [value for _, value in trainable]
    optimizer = torch.optim.AdamW(parameters, lr=plan.learning_rate, weight_decay=plan.weight_decay)
    start, consumed = 0, 0

    def restore():
        nonlocal start, consumed
        if resume is None:
            return
        if resume["metadata"].get("frozen_base_sha256") != frozen:
            raise ValueError("teacher checkpoint frozen base differs")
        _load_fit_state(resume_from, resume, model, optimizer, rank=rank, device=device)
        start = resume["metadata"]["completed_updates"]
        consumed = resume["metadata"]["consumed_tokens"]

    _shared_call(restore, control_group, "checkpoint restore")
    _agree(_adapter_digest(model), control_group, "restored adapter")
    # Construction broadcasts from rank zero; all identities were already checked.
    ddp = DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        process_group=training_group,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    initial_adapter = _adapter_digest(model)
    model.train()
    local_count = 64 // world
    started = time.monotonic()
    checkpoints = []
    step_times = []
    resume_verification = None
    metrics_path = output / "train-metrics.jsonl"

    def open_logs():
        ledger_path = output / f"fit-samples-rank{rank}.jsonl"
        if resume_from is not None:
            _copy_fsynced(resume_from / ledger_path.name, ledger_path)
            if rank == 0:
                _copy_fsynced(resume_from / "train-metrics.jsonl", metrics_path)
        return (
            metrics_path.open("a" if resume_from else "x") if rank == 0 else None,
            ledger_path.open("a" if resume_from else "x"),
        )

    stream, ledger = _shared_call(open_logs, control_group, "metrics open")
    try:
        _shared_call(
            lambda: observe.phase("teacher_training_start", resumed_updates=start),
            control_group,
            "start observation",
        )
        for step in range(start, plan.steps):
            step_started = time.monotonic()
            epoch, window = divmod(step, plan.train_examples // 64)
            global_indices = orders[epoch][window * 64 : (window + 1) * 64]
            local_indices = rank_window(global_indices, rank=rank, world_size=world)
            for group in optimizer.param_groups:
                group["lr"] = trace[step]
            optimizer.zero_grad(set_to_none=True)
            local_loss = 0.0
            for microstep, index in enumerate(local_indices):
                context = ddp.no_sync() if microstep < local_count - 1 else contextlib.nullcontext()
                with context:

                    def forward(index=index):
                        batch = collate([train_rows[index]], tokenizer.pad_token_id, device)
                        logits = ddp(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False,
                        ).logits
                        loss = response_sequence_loss(logits, batch["labels"])
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("nonfinite teacher loss")
                        return loss

                    loss = _shared_call(forward, control_group, f"forward {step}:{microstep}")
                    local_loss += float(loss.detach().cpu())
                    # DDP averages W rank gradients; dividing by64/W produces
                    # exactly the global64 sequence mean, including variable lengths.
                    _shared_call(
                        lambda loss=loss: (loss / local_count).backward(),
                        control_group,
                        f"backward {step}:{microstep}",
                    )
                    del loss
            norm = _shared_call(
                lambda: torch.nn.utils.clip_grad_norm_(
                    parameters, plan.max_grad_norm, error_if_nonfinite=True
                ),
                control_group,
                "clip",
            )
            if not float(norm) > 0:
                _sync_error(ValueError("zero teacher gradient"), control_group, "gradient norm")
            else:
                _sync_error(None, control_group, "gradient norm")
            _shared_call(optimizer.step, control_group, "optimizer update")
            consumed += sum(len(train_rows[index]["input_ids"]) for index in global_indices)
            loss_sum = torch.tensor(local_loss, dtype=torch.float64)
            dist.all_reduce(loss_sum, group=control_group)
            step_times.append(time.monotonic() - step_started)
            row = {
                "step": step + 1,
                "epoch": epoch,
                "loss": float(loss_sum) / 64,
                "learning_rate": trace[step],
                "grad_norm": float(norm),
                "samples": (step + 1) * 64,
                "tokens": consumed,
                "global_example_ids_sha256": sha256_value(
                    [train_rows[index]["example_id"] for index in global_indices]
                ),
                "elapsed_seconds": time.monotonic() - started,
                "step_seconds": step_times[-1],
            }

            def log_metrics(row=row, local_indices=local_indices):
                ledger.write(
                    json.dumps(
                        {
                            "step": row["step"],
                            "example_ids": [train_rows[index]["example_id"] for index in local_indices],
                        }
                    )
                    + "\n"
                )
                ledger.flush()
                os.fsync(ledger.fileno())
                if stream is not None:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

            _shared_call(log_metrics, control_group, "metrics commit")
            _shared_call(
                lambda step=step: observe.phase(
                    "teacher_optimizer_window_committed", completed_optimizer_steps=step + 1
                ),
                control_group,
                "update observation",
            )
            if step + 1 in plan.checkpoint_steps:
                metadata = {
                    **expected,
                    "completed_updates": step + 1,
                    "completed_samples": (step + 1) * 64,
                    "consumed_tokens": consumed,
                    "epoch": (step + 1) // (plan.train_examples // 64),
                    "next_epoch_offset": ((step + 1) % (plan.train_examples // 64)) * 64,
                    "frozen_base_sha256": frozen,
                    "train_metrics_sha256": sha256_file(metrics_path),
                    "train_metrics_path": "train-metrics.jsonl",
                }
                path, manifest = save_fit_checkpoint(
                    output / "checkpoints",
                    model,
                    optimizer,
                    metadata=metadata,
                    group=control_group,
                    device=device,
                )
                checkpoints.append({"path": str(path), "sha256": manifest["sha256"], "step": step + 1})
                if plan.mode == "preflight":
                    resume_verification = _verify_resume_roundtrip(
                        path,
                        model,
                        optimizer,
                        expected={**expected, "frozen_base_sha256": frozen},
                        group=control_group,
                        device=device,
                    )
                if checkpoint_hook is not None:
                    rng, adapter_before = _rng_state(device), _adapter_digest(model)
                    try:
                        _shared_call(
                            lambda path=path, manifest=manifest: checkpoint_hook(path, manifest),
                            control_group,
                            "checkpoint hook",
                        )
                    finally:
                        _restore_rng(rng, device)
                    _agree(_adapter_digest(model), control_group, "post-hook adapter")
                    _shared_call(
                        lambda adapter_before=adapter_before: _unchanged(
                            _adapter_digest(model), adapter_before, "hook changed teacher adapter"
                        ),
                        control_group,
                        "hook model state",
                    )
        frozen_after = _shared_call(
            lambda: _tensor_digest(
                (name, value) for name, value in model.named_parameters() if not value.requires_grad
            ),
            control_group,
            "final frozen base",
        )
        _shared_call(
            lambda: _unchanged(frozen, frozen_after, "frozen teacher base changed"),
            control_group,
            "base validation",
        )
        final_adapter = _adapter_digest(model)
        _agree(final_adapter, control_group, "final adapter")
        if start < plan.steps:
            _shared_call(lambda: _different(initial_adapter, final_adapter), control_group, "adapter update")
    finally:
        if stream is not None:
            stream.close()
        ledger.close()
    return model, {
        "artifact_kind": "teacher_adapter_fit_execution",
        "accepted_science": False,
        "formal_teacher_accepted": False,
        "student_training_started": False,
        "completed_updates": plan.steps,
        "completed_samples": plan.steps * 64,
        "consumed_tokens": consumed,
        "world_size": world,
        "resumed_updates": start,
        "adapter_sha256": final_adapter,
        "frozen_base_sha256": frozen,
        "trainable_parameter_count": sum(value.numel() for value in parameters),
        "adapter_modules": modules,
        "optimizer_defaults": dict(optimizer.defaults),
        "lr_trace": list(trace),
        "step_seconds": step_times,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "same_world_resume_verified": resume_verification is not None,
        "resume_verification": resume_verification,
        "checkpoints": checkpoints,
        "checkpoint_publication": "local_atomic_fsync; caller verifies persistent storage",
    }


def _unchanged(before: Any, after: Any, message: str) -> None:
    if before != after:
        raise ValueError(message)


def _different(before: str, after: str) -> None:
    if before == after:
        raise ValueError("teacher adapter did not change")
