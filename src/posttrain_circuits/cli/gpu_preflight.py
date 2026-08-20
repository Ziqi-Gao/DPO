"""Short four-rank CUDA/NCCL and pinned-Qwen execution gate."""

from __future__ import annotations

import argparse
import os
import resource
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as functional

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_file, sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.provenance import require_git_output
from posttrain_circuits.models.loading import (
    assert_tokenizer_compatible,
    load_model_and_tokenizer,
)
from posttrain_circuits.models.prompt_protocol import format_model_prompt


def _read_cgroup_value(path: Path) -> int | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if value == "max":
        return None
    parsed = int(value)
    # cgroup-v1 represents an unlimited memory controller with a value close
    # to INT64_MAX. That is not evidence of a finite Slurm allocation limit.
    return None if parsed >= 2**60 else parsed


def cgroup_memory_snapshot(
    *,
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, int | None]:
    """Read the Slurm cgroup-v1 or cgroup-v2 memory controller without host fallbacks."""

    unified_relative: Path | None = None
    memory_relative: Path | None = None
    if proc_cgroup.is_file():
        for line in proc_cgroup.read_text(encoding="utf-8").splitlines():
            fields = line.split(":", 2)
            if len(fields) != 3:
                continue
            controllers = fields[1].split(",") if fields[1] else []
            relative = Path(fields[2].lstrip("/"))
            if fields[0] == "0" and not controllers:
                unified_relative = relative
            if "memory" in controllers:
                memory_relative = relative
    if unified_relative is not None:
        root = cgroup_root / unified_relative
        names = ("memory.max", "memory.current", "memory.peak")
    elif memory_relative is not None:
        root = cgroup_root / "memory" / memory_relative
        names = ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes")
    else:
        return {"limit_bytes": None, "current_bytes": None, "peak_bytes": None}
    return {
        "limit_bytes": _read_cgroup_value(root / names[0]),
        "current_bytes": _read_cgroup_value(root / names[1]),
        "peak_bytes": _read_cgroup_value(root / names[2]),
    }


def validate_memory_headroom(
    snapshot: dict[str, int | None],
    *,
    requested_gib: int,
    minimum_headroom_gib: int,
    minimum_headroom_fraction: float,
    observed_process_peak_bytes: int,
) -> dict[str, Any]:
    limit = snapshot.get("limit_bytes")
    cgroup_peak = snapshot.get("peak_bytes")
    if limit is None or limit <= 0:
        raise RuntimeError("GPU preflight cannot verify a finite Slurm cgroup memory limit")
    requested_bytes = requested_gib * 1024**3
    if limit < requested_bytes:
        raise RuntimeError(f"Slurm cgroup memory limit {limit} is below registered request {requested_bytes}")
    observed_peak = max(int(cgroup_peak or 0), observed_process_peak_bytes)
    headroom = limit - observed_peak
    minimum = max(int(minimum_headroom_gib * 1024**3), int(limit * minimum_headroom_fraction))
    return {
        **snapshot,
        "requested_bytes": requested_bytes,
        "observed_peak_bytes": observed_peak,
        "headroom_bytes": headroom,
        "minimum_required_headroom_bytes": minimum,
        "passed": observed_peak > 0 and headroom >= minimum,
    }


def _parameter_checksum(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    checksum = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in model.parameters():
        checksum += parameter.detach().double().sum()
    return checksum


def _qwen3_training_path(
    config: dict[str, Any],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    checkpoint_root: Path,
) -> dict[str, Any]:
    """Exercise the actual simultaneous teacher/student and FSDP runtime path."""

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )

    student_bundle = load_model_and_tokenizer(config["model"], for_training=True)
    vocab_size = int(student_bundle.model.config.vocab_size)
    student = FSDP(
        student_bundle.model,
        device_id=device,
        use_orig_params=True,
    )
    teacher = None
    teacher_metadata: list[Any] = [None]
    if rank == 0:
        teacher_bundle = load_model_and_tokenizer(config["teacher"], for_training=False)
        tokenizer_hash = assert_tokenizer_compatible(
            student_bundle.tokenizer,
            teacher_bundle.tokenizer,
        )
        teacher = teacher_bundle.model.to(device)
        teacher_metadata[0] = {
            "resolved_teacher_commit": teacher_bundle.resolved_model_commit,
            "tokenizer_fingerprint": tokenizer_hash,
        }
    dist.broadcast_object_list(teacher_metadata, src=0)
    if not isinstance(teacher_metadata[0], dict):
        raise RuntimeError("rank-zero teacher metadata broadcast failed")
    tokenizer_hash = str(teacher_metadata[0]["tokenizer_fingerprint"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    raw_prompt = (
        f"RANK-CANARY-{rank}\nFACTS F01: A\nRULES R01: A -> B\n"
        "QUERY: B\nOUTPUT FORMAT: <proof> ... </proof> <answer>0|1</answer>"
    )
    formatted = format_model_prompt(raw_prompt, student_bundle.tokenizer, config["model"])
    encoded = student_bundle.tokenizer(
        formatted.model_facing_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    prompt_hashes: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(prompt_hashes, formatted.model_facing_prompt_sha256)
    unique_rank_shards = len(set(str(value) for value in prompt_hashes)) == world_size

    optimizer.zero_grad(set_to_none=True)
    gathered_ids = [torch.empty_like(input_ids) for _ in range(world_size)]
    gathered_masks = [torch.empty_like(attention_mask) for _ in range(world_size)]
    dist.all_gather(gathered_ids, input_ids)
    dist.all_gather(gathered_masks, attention_mask)
    teacher_logits = torch.empty(
        (*input_ids.shape, vocab_size),
        dtype=torch.bfloat16,
        device=device,
    )
    if rank == 0:
        if teacher is None:
            raise RuntimeError("rank zero did not load the teacher")
        with torch.no_grad():
            batched_teacher_logits = teacher(
                input_ids=torch.cat(gathered_ids, dim=0),
                attention_mask=torch.cat(gathered_masks, dim=0),
            ).logits
        scatter_rows = list(batched_teacher_logits.split(input_ids.shape[0], dim=0))
    else:
        scatter_rows = None
    dist.scatter(teacher_logits, scatter_list=scatter_rows, src=0)
    student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
    soft_teacher_loss = functional.kl_div(
        student_logits.float().log_softmax(dim=-1),
        teacher_logits.float().softmax(dim=-1),
        reduction="batchmean",
    )
    soft_teacher_loss.backward()
    gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(bool(torch.isfinite(grad).all()) for grad in gradients)
    before_step = _parameter_checksum(student, device)
    optimizer.step()
    after_step = _parameter_checksum(student, device)
    update_norm_nonzero = bool((after_step - before_step).abs().item() > 0.0)

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    rank_path = checkpoint_root / f"rank-{rank:02d}.pt"
    model_state_config = ShardedStateDictConfig(offload_to_cpu=True)
    optim_state_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(
        student,
        StateDictType.SHARDED_STATE_DICT,
        model_state_config,
        optim_state_config,
    ):
        model_state = student.state_dict()
        optimizer_state = FSDP.optim_state_dict(student, optimizer)
    torch.save({"model": model_state, "optimizer": optimizer_state}, rank_path)
    saved_checksum = _parameter_checksum(student, device)
    with torch.no_grad():
        first_parameter = next(parameter for parameter in student.parameters() if parameter.numel())
        first_parameter.add_(1.0)
    checkpoint = torch.load(rank_path, map_location="cpu", weights_only=False)
    with FSDP.state_dict_type(
        student,
        StateDictType.SHARDED_STATE_DICT,
        model_state_config,
        optim_state_config,
    ):
        load_result = student.load_state_dict(checkpoint["model"])
        optimizer_state = FSDP.optim_state_dict_to_load(
            student,
            optimizer,
            checkpoint["optimizer"],
        )
        optimizer.load_state_dict(optimizer_state)
    restored_checksum = _parameter_checksum(student, device)
    resume_passed = (
        not load_result.missing_keys
        and not load_result.unexpected_keys
        and bool(torch.allclose(saved_checksum, restored_checksum, rtol=0.0, atol=1e-6))
    )
    row = {
        "rank": rank,
        "raw_prompt_sha256": formatted.raw_prompt_sha256,
        "model_facing_prompt_sha256": formatted.model_facing_prompt_sha256,
        "prompt_protocol": formatted.prompt_protocol,
        "enable_thinking": formatted.enable_thinking,
        "student_revision": student_bundle.resolved_model_commit,
        "teacher_revision": str(teacher_metadata[0]["resolved_teacher_commit"]),
        "tokenizer_revision": student_bundle.resolved_tokenizer_commit,
        "tokenizer_fingerprint": tokenizer_hash,
        "chat_template_sha256": student_bundle.chat_template_sha256,
        "teacher_forward_finite": bool(torch.isfinite(teacher_logits).all()),
        "student_forward_finite": bool(torch.isfinite(student_logits).all()),
        "soft_teacher_loss": float(soft_teacher_loss.detach()),
        "soft_teacher_loss_finite": bool(torch.isfinite(soft_teacher_loss)),
        "gradients_finite": gradients_finite,
        "parameter_update_nonzero": update_norm_nonzero,
        "unique_rank_prompt_shard": unique_rank_shards,
        "fsdp_save_resume": resume_passed,
        "checkpoint_sha256": sha256_file(rank_path),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "teacher_loaded_on_this_rank": rank == 0,
        "loading_strategy": "low_cpu_mem_student_rank_zero_teacher",
    }
    del teacher_logits, student_logits, teacher, student, optimizer
    torch.cuda.empty_cache()
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the short production GPU preflight")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = compose_config(args.overrides)
    if torch.version.cuda is None:
        raise RuntimeError("GPU preflight requires a CUDA-enabled torch build")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"GPU preflight requires exactly four ranks, observed {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    reduced = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(reduced)
    nccl_passed = float(reduced.item()) == 10.0
    device_rows: list[Any] = [None for _ in range(world_size)]
    props = torch.cuda.get_device_properties(device)
    dist.all_gather_object(
        device_rows,
        {
            "rank": rank,
            "name": props.name,
            "total_memory": props.total_memory,
            "capability": [props.major, props.minor],
        },
    )
    forward_finite = False
    resolved_model_commit = ""
    tokenizer_hash = ""
    qwen3_rows: list[Any] = [None for _ in range(world_size)]
    if str(config.get("protocol_track", "")).startswith("qwen3_"):
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise RuntimeError("Qwen3 production preflight requires HF_HUB_OFFLINE=1")
        if rank == 0 and require_git_output(["status", "--porcelain"]):
            raise RuntimeError("Qwen3 production preflight refuses a dirty source checkout")
        dist.barrier()
        rank_row = _qwen3_training_path(
            config,
            rank=rank,
            world_size=world_size,
            device=device,
            checkpoint_root=args.output.parent / "fsdp-resume",
        )
        dist.all_gather_object(qwen3_rows, rank_row)
        forward_finite = all(
            row["student_forward_finite"]
            and row["teacher_forward_finite"]
            and row["soft_teacher_loss_finite"]
            and row["gradients_finite"]
            and row["parameter_update_nonzero"]
            and row["unique_rank_prompt_shard"]
            and row["fsdp_save_resume"]
            for row in qwen3_rows
        )
        resolved_model_commit = str(qwen3_rows[0]["student_revision"])
        tokenizer_hash = str(qwen3_rows[0]["tokenizer_fingerprint"])
    elif rank == 0:
        loaded = load_model_and_tokenizer(config["model"], for_training=False)
        loaded.model.to(device)
        encoded = loaded.tokenizer("Prove: if A implies B and A, then B.", return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = loaded.model(**encoded).logits
        forward_finite = bool(torch.isfinite(logits).all().item())
        resolved_model_commit = loaded.resolved_model_commit
        tokenizer_hash = loaded.tokenizer_hash
        del loaded, logits, encoded
        torch.cuda.empty_cache()
    status = torch.tensor(int(forward_finite), device=device)
    dist.broadcast(status, src=0)
    if rank == 0:
        git_commit = require_git_output(["rev-parse", "HEAD"])
        payload: dict[str, Any] = {
            "phase": "gpu_preflight",
            "passed": nccl_passed and bool(status.item()),
            "world_size": world_size,
            "visible_cuda_devices": torch.cuda.device_count(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "nccl_all_reduce": nccl_passed,
            "qwen_forward_finite": bool(status.item()),
            "model_revision": str(config["model"]["model_revision"]),
            "resolved_model_commit": resolved_model_commit,
            "tokenizer_hash": tokenizer_hash,
            "protocol_track": str(config.get("protocol_track", "core_v2")),
            "artifact_namespace": str(config["model"].get("artifact_namespace", "legacy")),
            "resolved_config_sha256": sha256_value(config),
            "launch_environment": {
                name: os.environ.get(name)
                for name in (
                    "MODEL_CONFIG",
                    "TEACHER_CONFIG",
                    "PRODUCTION_CONFIG",
                    "G0_CONFIG",
                    "PILOT_CONFIG",
                    "PROJECT_ROOT",
                    "PYTHON_BIN",
                    "ACCELERATE_BIN",
                    "OUTPUT_ROOT",
                )
            },
            "devices": device_rows,
            "git_commit": git_commit,
            "code_commit": git_commit,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "created_at": utc_now(),
        }
        if str(config.get("protocol_track", "")).startswith("qwen3_"):
            prereg_path = Path(str(config["prereg_path"]))
            process_peak = sum(int(row["process_max_rss_bytes"]) for row in qwen3_rows)
            resource_budget = config.get("resource_budget", {})
            memory = validate_memory_headroom(
                cgroup_memory_snapshot(),
                requested_gib=int(resource_budget["node_memory_gib"]),
                minimum_headroom_gib=int(resource_budget["minimum_headroom_gib"]),
                minimum_headroom_fraction=float(resource_budget["minimum_headroom_fraction"]),
                observed_process_peak_bytes=process_peak,
            )
            payload["passed"] = bool(payload["passed"] and memory["passed"])
            payload.update(
                {
                    "teacher_revision": str(config["teacher"]["model_revision"]),
                    "resolved_teacher_commit": str(qwen3_rows[0]["teacher_revision"]),
                    "prompt_protocol": str(qwen3_rows[0]["prompt_protocol"]),
                    "enable_thinking": bool(qwen3_rows[0]["enable_thinking"]),
                    "chat_template_sha256": str(qwen3_rows[0]["chat_template_sha256"]),
                    "prereg_path": str(prereg_path),
                    "prereg_sha256": sha256_file(prereg_path),
                    "prereg_commit": require_git_output(
                        ["log", "-n", "1", "--format=%H", "--", str(prereg_path)]
                    ),
                    "prereg_version": str(config["prereg_version"]),
                    "tokenizer_revision": str(config["model"]["tokenizer_revision"]),
                    "tokenizer_fingerprint": str(qwen3_rows[0]["tokenizer_fingerprint"]),
                    "rank_training_checks": qwen3_rows,
                    "rank_prompt_hashes_unique": len(
                        {str(row["model_facing_prompt_sha256"]) for row in qwen3_rows}
                    )
                    == world_size,
                    "cgroup_memory": memory,
                    "rank_zero_teacher_load_count": sum(
                        int(row["teacher_loaded_on_this_rank"]) for row in qwen3_rows
                    ),
                }
            )
        payload["sha256"] = sha256_value(payload)
        atomic_write_json(args.output, payload)
        status.fill_(int(payload["passed"]))
    dist.broadcast(status, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if not nccl_passed or not bool(status.item()):
        raise SystemExit("GPU preflight failed")


if __name__ == "__main__":
    main()
