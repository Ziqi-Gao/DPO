#!/usr/bin/env python3
"""Independent two-H100 canary; never a G0 result or Blackwell certification.

Launch with torchrun --standalone --nnodes=1 --nproc_per_node=2. This module
imports only stdlib until path/cache/runtime/allocation checks have passed.
All outputs stay on the explicit node-local workspace; its caller must preserve
required artifacts to verified persistent storage before the allocation exits.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import signal
import socket
import stat
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 2
HOST_BYTES = 192 * 1024**3
MODELS = {
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}
DEPENDENCIES = {
    "torch": "2.8.0+cu128",
    "transformers": "4.56.2",
    "accelerate": "1.10.1",
    "tokenizers": "0.22.0",
    "safetensors": "0.5.3",
    "huggingface-hub": "0.36.2",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "pydantic": "2.11.7",
    "PyYAML": "6.0.2",
    "nvidia-nccl-cu12": "2.27.3",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def log_phase(rank, phase):
    print(json.dumps({"rank": rank, "phase": phase, "monotonic_seconds": time.monotonic()}), flush=True)


def runtime_identity(version=importlib.metadata.version, python_version=None):
    actual_python = python_version or platform.python_version()
    require(actual_python == "3.12.13", "preflight requires Python 3.12.13")
    actual = {name: version(name) for name in DEPENDENCIES}
    require(actual == DEPENDENCIES, f"preflight dependency mismatch: {actual}")
    return {"python": actual_python, "executable": sys.executable, "packages": actual}


def allocation_identity(environment):
    require(re.fullmatch(r"[1-9][0-9]*", environment.get("SLURM_JOB_ID", "")), "Slurm job ID missing")
    require(environment.get("SLURM_CPUS_PER_TASK") == "24", "preflight requires 24 allocated CPUs")
    require(environment.get("SLURM_MEM_PER_NODE") == "196608", "preflight requires 192 GiB Slurm memory")
    world = int(environment.get("WORLD_SIZE", "0"))
    rank, local_rank = int(environment.get("RANK", "-1")), int(environment.get("LOCAL_RANK", "-1"))
    require(
        world == WORLD_SIZE and rank in range(world) and local_rank == rank,
        "preflight requires single-node torchrun with exactly two ranks",
    )
    require(environment.get("LOCAL_WORLD_SIZE", "2") == "2", "local torchrun world differs")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    require(
        len(devices) == world
        and len(set(devices)) == world
        and all(value and value.strip() == value for value in devices),
        "CUDA visibility must expose two GPUs",
    )
    return {
        "job_id": environment["SLURM_JOB_ID"],
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "cuda_visible_devices": visible,
        "threads_per_rank": 12,
    }


def local_paths(work, output):
    require(work.is_absolute() and work.is_dir(), "work-dir must exist and be absolute")
    require(not any(path.is_symlink() for path in (work, *work.parents)), "work-dir traverses a symlink")
    work = work.resolve()
    require(work.stat().st_uid == os.getuid(), "work-dir is not owned by the current user")
    require(
        work != Path.home().resolve() and Path.home().resolve() not in work.parents, "work-dir is in HOME"
    )
    require(
        output.is_absolute() and output != work and work in output.parents and ".." not in output.parts,
        "output-dir must be below work-dir",
    )
    require(
        not any(path.is_symlink() for path in (output, *output.parents)), "output-dir traverses a symlink"
    )
    mount = json.loads(
        subprocess.run(
            ["findmnt", "--json", "--target", str(work), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    )["filesystems"][0]
    require(
        mount["fstype"] in {"xfs", "ext4", "ext3", "ext2", "btrfs"},
        "work-dir is not verified node-local disk",
    )
    return mount


def pinned_cache(hf_home):
    require(hf_home.is_absolute() and hf_home.is_dir(), "HF_HOME must be an existing absolute directory")
    base = hf_home.resolve()
    snapshots = {}
    for model, revision in MODELS.items():
        snapshot = base / "hub" / ("models--" + model.replace("/", "--")) / "snapshots" / revision
        require(snapshot.is_dir(), f"missing pinned offline snapshot: {model}@{revision}")
        paths = [snapshot / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")]
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            payload = json.loads(index.read_text())
            shards = set(payload["weight_map"].values())
            require(
                shards and all(isinstance(name, str) and Path(name).name == name for name in shards),
                "invalid pinned model shard names",
            )
            paths.extend(snapshot / name for name in sorted(shards))
            paths.append(index)
        else:
            paths.append(snapshot / "model.safetensors")
        require(
            all(
                path.is_file() and path.stat().st_size > 0 and base in path.resolve().parents
                for path in paths
            ),
            f"pinned snapshot is incomplete or references files outside HF_HOME: {model}",
        )
        snapshots[model] = {
            "revision": revision,
            "snapshot": str(snapshot),
            "files": [{"name": path.name, "size": path.stat().st_size} for path in paths],
        }
    return snapshots


def memory_envelope(proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    locations = []
    for line in proc.read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        require(".." not in Path(relative).parts, "invalid cgroup path")
        if hierarchy == "0" and not controllers:
            locations.append(
                (root, root / relative.lstrip("/"), ("memory.max", "memory.current", "memory.peak"))
            )
        elif "memory" in controllers.split(","):
            locations.insert(
                0,
                (
                    root / "memory",
                    root / "memory" / relative.lstrip("/"),
                    ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes"),
                )
            )
    require(locations, "no readable memory cgroup")
    boundary, directory, names = locations[0]
    levels = []
    while True:
        level = {"path": str(directory), "limit_bytes": None}
        if (directory / names[0]).is_file():
            raw_limit = (directory / names[0]).read_text().strip()
            limit = None if raw_limit == "max" else int(raw_limit)
            require(limit is None or limit > 0, "invalid cgroup memory limit")
            # cgroup v1 represents an unlimited limit as an enormous page-aligned integer.
            level["limit_bytes"] = limit if limit is not None and limit < 2**60 else None
            level["raw_limit"] = raw_limit
            for name, key in zip(names[1:], ("current_bytes", "peak_bytes"), strict=True):
                if (directory / name).is_file():
                    level[key] = int((directory / name).read_text().strip())
        levels.append(level)
        if directory == boundary:
            break
        require(boundary in directory.parents, "memory cgroup escaped controller root")
        directory = directory.parent
    finite = [level for level in levels if level["limit_bytes"] is not None]
    require(finite, "cgroup must expose a finite 192 GiB limit")
    limit = min(level["limit_bytes"] for level in finite)
    limiting = [level for level in finite if level["limit_bytes"] == limit]
    require(
        all("current_bytes" in level and "peak_bytes" in level for level in limiting),
        "limiting memory cgroup must expose current and peak usage",
    )
    # When task and job limits match, retain the higher aggregate job peak.
    selected = max(limiting, key=lambda level: level["peak_bytes"])
    current, peak = selected["current_bytes"], selected["peak_bytes"]
    require(
        limit == HOST_BYTES
        and all(0 < level["current_bytes"] <= level["peak_bytes"] <= limit for level in limiting),
        "cgroup must expose finite 192 GiB and real peak",
    )
    required = max(32 * 1024**3, math.ceil(limit * 0.20))
    require(limit - peak >= required, "192 GiB cgroup lacks 32 GiB and 20% headroom")
    return {
        "path": selected["path"],
        "limit_bytes": limit,
        "current_bytes": current,
        "peak_bytes": peak,
        "headroom_bytes": limit - peak,
        "minimum_headroom_bytes": required,
        "ancestors": levels,
        "passed": True,
    }


def publish_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def tensor_digest(torch, values):
    result = hashlib.sha256()
    for name, tensor in values:
        if torch.is_tensor(tensor):
            result.update(str(name).encode())
            result.update(str(tuple(tensor.shape)).encode())
            result.update(str(tensor.dtype).encode())
            result.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def optimizer_digest(torch, optimizer):
    return tensor_digest(
        torch,
        (
            (f"{group_index}:{parameter_index}:{name}", value)
            for group_index, group in enumerate(optimizer.param_groups)
            for parameter_index, parameter in enumerate(group["params"])
            for name, value in sorted(optimizer.state.get(parameter, {}).items())
        ),
    )


def canary_rows(tokenizer, slots, formatter, model_config):
    """Exact 1536 active tokens, including 256 masked response targets; synthetic load only."""
    response = tokenizer.encode(
        "<proof>\nS01: R01(F01) -> TRUE SYM_001\n</proof>\n<answer>1</answer>", add_special_tokens=False
    )
    filler = tokenizer.encode("\nDISTRACTOR: SYM_999", add_special_tokens=False)
    require(response and filler and len(response) <= 256, "canary tokenization cannot realize fixed shape")
    response = (response + filler * 256)[:256]
    rows = []
    for slot in slots:
        text = f"SLOT-CANARY-{slot:02d}\nFACTS F01: SYM_000\nRULES R01: SYM_000 -> SYM_001\nQUERY: SYM_001"
        prompt = formatter(text, tokenizer, model_config).model_facing_prompt
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        require(len(prefix) <= 1246, "canary prompt exceeded reviewed prefix bound")
        rows.append((prefix + filler * 1280)[:1280] + response)
    return rows


def checkpoint_names(world_size):
    return sorted(
        ["model-full.pt", "optimizer-full.pt", *[f"rank-{rank}-runtime.pt" for rank in range(world_size)]]
    )


def checkpoint_file(path):
    require(not any(item.is_symlink() for item in (path, *path.parents)), "checkpoint traverses a symlink")
    require(stat.S_ISREG(path.stat().st_mode), "checkpoint is not a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.name, "size": path.stat().st_size, "sha256": digest.hexdigest()}


def checkpoint_header(header, code_hash, world_size):
    require(
        set(header) == {"kind", "code_sha256", "world_size", "files"}
        and header["kind"] == "sdsc_h100_canary_checkpoint_v1"
        and header["code_sha256"] == code_hash
        and header["world_size"] == world_size,
        "checkpoint identity/world size mismatch before load",
    )
    require(isinstance(header["files"], list), "checkpoint files manifest missing")
    for record in header["files"]:
        require(
            isinstance(record, dict)
            and set(record) == {"path", "size", "sha256"}
            and isinstance(record["path"], str)
            and type(record["size"]) is int
            and record["size"] > 0
            and isinstance(record["sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", record["sha256"]),
            "invalid checkpoint file record",
        )
    require(
        [record["path"] for record in header["files"]] == checkpoint_names(world_size),
        "checkpoint must bind model, optimizer, and every rank runtime exactly once",
    )


def verify_checkpoint(directory, code_hash, world_size):
    manifest = directory / "manifest.json"
    require(not manifest.is_symlink(), "checkpoint manifest is a symlink")
    header = json.loads(manifest.read_text())
    checkpoint_header(header, code_hash, world_size)
    for record in header["files"]:
        require(
            checkpoint_file(directory / record["path"]) == record,
            "checkpoint file hash/size mismatch before load",
        )
    return header


def collective_phase(dist, operation, group=None):
    value, error = None, None
    try:
        value = operation()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    statuses = [None] * WORLD_SIZE
    dist.all_gather_object(statuses, error, group=group)
    require(not any(statuses), f"rank-local phase failed: {statuses}")
    return value


def run_canary(args, identity, evidence):
    import torch
    import torch.distributed as dist
    import yaml
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    sys.path.insert(0, str(ROOT / "src"))
    from posttrain_circuits.learning.supervision.losses import verified_replay_loss
    from posttrain_circuits.learning.training.execution_safety_kernel import batch_token_contract
    from posttrain_circuits.learning.training.fsdp_contract import (
        full_state_dict_options,
        validate_model_fsdp_sharding,
    )
    from posttrain_circuits.models.loading import assert_tokenizer_compatible, load_model_and_tokenizer
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    rank, world = identity["rank"], identity["world_size"]
    log_phase(rank, "gpu_and_distributed_checks")
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == world, "visible CUDA device count differs"
    )
    require(torch.version.cuda == "12.8", "Torch CUDA runtime differs from 12.8")
    require(tuple(torch.cuda.nccl.version()) == (2, 27, 3), "NCCL runtime differs from 2.27.3")
    properties = [torch.cuda.get_device_properties(index) for index in range(world)]
    require(
        all(
            "H100" in item.name and (item.major, item.minor) == (9, 0) and item.total_memory >= 75 * 1024**3
            for item in properties
        ),
        "two H100 80GB GPUs required",
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.set_num_threads(12)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=600))
    data_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=120))
    value = torch.tensor(float(rank + 1), device=device)
    work = dist.all_reduce(value, group=data_group, async_op=True)
    work.wait(timeout=timedelta(seconds=120))
    require(value.item() == 3.0, "first NCCL all-reduce returned incorrect value")
    evidence.update(
        nccl_all_reduce=True,
        gpu={
            "name": properties[rank].name,
            "total_memory_bytes": properties[rank].total_memory,
            "compute_capability": [9, 0],
            "logical_device": rank,
        },
    )
    model_config = yaml.safe_load((ROOT / "configs/model/qwen3_v2_1p7b.yaml").read_text())
    teacher_config = yaml.safe_load((ROOT / "configs/teacher/qwen3_v2_teacher_8b.yaml").read_text())
    for config, expected in ((model_config, "Qwen/Qwen3-1.7B"), (teacher_config, "Qwen/Qwen3-8B")):
        require(
            config["model_name_or_path"] == expected
            and config["model_revision"] == MODELS[expected]
            and config["tokenizer_name_or_path"] == expected
            and config["tokenizer_revision"] == MODELS[expected]
            and config["torch_dtype"] == "bfloat16"
            and config["trust_remote_code"] is False,
            "model configuration differs from the fixed offline pins",
        )
    log_phase(rank, "student_load")
    student_bundle = collective_phase(dist, lambda: load_model_and_tokenizer(model_config, for_training=True))
    model, tokenizer = student_bundle.model, student_bundle.tokenizer
    require(student_bundle.resolved_model_commit == MODELS["Qwen/Qwen3-1.7B"], "student revision mismatch")
    layer_types = {type(module) for module in model.modules() if type(module).__name__ == "Qwen3DecoderLayer"}
    require(len(layer_types) == 1, "student decoder layer type differs")
    student = FSDP(
        model,
        process_group=data_group,
        device_id=device,
        use_orig_params=False,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_types),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
    )
    evidence["fsdp"] = validate_model_fsdp_sharding(student, world_size=world, fsdp_type=FSDP)
    require(evidence["fsdp"]["fsdp_wrapper_count"] == 29, "student must contain 28 wrapped decoder layers")
    teacher = None

    def teacher_probe():
        nonlocal teacher
        if rank != 0:
            return None
        bundle = load_model_and_tokenizer(teacher_config, for_training=False)
        require(bundle.resolved_model_commit == MODELS["Qwen/Qwen3-8B"], "teacher revision mismatch")
        assert_tokenizer_compatible(tokenizer, bundle.tokenizer)
        teacher = bundle.model.to(device)
        prompt = format_model_prompt(
            "FACTS F01: A\nRULES R01: A -> B\nQUERY: B", bundle.tokenizer, teacher_config
        ).model_facing_prompt
        inputs = bundle.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.no_grad():
            output = teacher(**inputs).logits
        require(bool(torch.isfinite(output).all()), "teacher forward is not finite")
        return {"finite": True, "revision": bundle.resolved_model_commit}

    teacher_result = collective_phase(dist, teacher_probe)
    broadcast = [teacher_result]
    dist.broadcast_object_list(broadcast, src=0)
    evidence["rank_zero_teacher"] = broadcast[0]
    contract = batch_token_contract(world, 256)
    evidence["batch_contract"] = contract
    slots = list(range(rank, 64, world))
    require(
        contract["samples_by_rank"][rank] == len(slots)
        and contract["microbatch_schedule_by_rank"][rank] == [4] * 8,
        "global batch kernel changed",
    )
    rows = canary_rows(tokenizer, slots, format_model_prompt, model_config)
    local_tokens = torch.tensor(len(rows) * 1536, dtype=torch.int64)
    dist.all_reduce(local_tokens)
    require(local_tokens.item() == 64 * 1536 <= contract["token_budget"], "global token reservation differs")
    evidence["reserved_global_nonpadding_tokens"] = local_tokens.item()
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005, weight_decay=0.0, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    before = tensor_digest(torch, student.named_parameters())
    losses = []
    log_phase(rank, "global64_optimizer_window")
    for microstep in range(8):
        input_ids = torch.tensor(rows[microstep * 4 : (microstep + 1) * 4], device=device)
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        response_mask[:, -256:] = True
        with contextlib.nullcontext() if microstep == 7 else student.no_sync():
            logits = student(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
            require(bool(torch.isfinite(logits).all()), "student forward is not finite")
            loss = verified_replay_loss(
                logits.float(),
                input_ids,
                response_mask,
                torch.ones(4, device=device),
                normalization="sequence",
            )
            require(bool(torch.isfinite(loss)), "response-masked loss is not finite")
            losses.append(float(loss.detach()))
            (loss * (world * 4 / 64)).backward()
        del logits, loss, input_ids, response_mask
    gradients = [parameter.grad for parameter in student.parameters() if parameter.grad is not None]
    require(
        gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
        "nonfinite/empty gradients",
    )
    optimizer.step()
    scheduler.step()
    after = tensor_digest(torch, student.named_parameters())
    require(before != after, "full-parameter AdamW update was zero")
    evidence.update(
        losses=losses,
        finite_gradients=True,
        parameter_update_nonzero=True,
        local_global_slots=slots,
        local_parameter_sha256_before=before,
        local_parameter_sha256_after=after,
    )
    saved_optim = optimizer_digest(torch, optimizer)
    runtime = {
        "scheduler": scheduler.state_dict(),
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(device).clone(),
    }
    checkpoint = args.output_dir / "checkpoint"
    log_phase(rank, "full_state_save_and_restore")
    collective_phase(dist, lambda: checkpoint.mkdir(exist_ok=True))
    runtime_file = checkpoint / f"rank-{rank}-runtime.pt"
    collective_phase(dist, lambda: torch.save(runtime, runtime_file))
    options = full_state_dict_options(world)
    with FSDP.state_dict_type(
        student,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(**options),
        FullOptimStateDictConfig(**options),
    ):
        model_state = student.state_dict()
        optim_state = FSDP.optim_state_dict(student, optimizer)

    def save_full(model_payload, optimizer_payload):
        if rank == 0:
            torch.save(model_payload, checkpoint / "model-full.pt")
            torch.save(optimizer_payload, checkpoint / "optimizer-full.pt")
            publish_json(
                checkpoint / "manifest.json",
                {
                    "kind": "sdsc_h100_canary_checkpoint_v1",
                    "world_size": identity["world_size"],
                    "code_sha256": args.code_sha256,
                    "files": [checkpoint_file(checkpoint / name) for name in checkpoint_names(world)],
                },
            )

    collective_phase(dist, functools.partial(save_full, model_state, optim_state))
    del model_state, optim_state
    checkpoint_manifest = collective_phase(
        dist, lambda: verify_checkpoint(checkpoint, args.code_sha256, world)
    )
    with torch.no_grad():
        next(parameter for parameter in student.parameters() if parameter.numel()).add_(1)
        next(
            value
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value) and value.numel()
        ).add_(1)
    scheduler.step()
    torch.rand(1)
    torch.rand(1, device=device)
    state = collective_phase(
        dist, lambda: torch.load(checkpoint / "model-full.pt", map_location="cpu", weights_only=False)
    )
    with FSDP.state_dict_type(
        student, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    ):
        loaded = student.load_state_dict(state)
    del state
    full_optim = collective_phase(
        dist,
        lambda: torch.load(checkpoint / "optimizer-full.pt", map_location="cpu", weights_only=False)
        if rank == 0
        else None,
    )
    sharded = FSDP.scatter_full_optim_state_dict(full_optim, student, optim=optimizer, group=data_group)
    optimizer.load_state_dict(sharded)
    del full_optim, sharded
    restored = torch.load(runtime_file, map_location="cpu", weights_only=False)
    scheduler.load_state_dict(restored["scheduler"])
    torch.set_rng_state(restored["cpu_rng"])
    torch.cuda.set_rng_state(restored["cuda_rng"], device)
    require(
        not loaded.missing_keys
        and not loaded.unexpected_keys
        and tensor_digest(torch, student.named_parameters()) == after
        and optimizer_digest(torch, optimizer) == saved_optim
        and scheduler.state_dict() == runtime["scheduler"]
        and torch.equal(torch.get_rng_state(), runtime["cpu_rng"])
        and torch.equal(torch.cuda.get_rng_state(device), runtime["cuda_rng"]),
        "same-world checkpoint restore differs",
    )
    evidence["full_state_optimizer_scheduler_rng_restore"] = True
    evidence["checkpoint"] = {
        "directory": str(checkpoint),
        "world_size": world,
        "local_restored_parameter_sha256": after,
        "local_restored_optimizer_sha256": saved_optim,
        "large_files_require_caller_persistence": True,
        "files": checkpoint_manifest["files"],
        "all_files_verified_before_restore": True,
    }
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_reserved(device)
    require(0 < peak < properties[rank].total_memory, "GPU peak left no memory headroom")
    evidence.update(
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_gpu_reserved_bytes=peak,
        cgroup_memory=memory_envelope(),
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"], "CUDA visibility changed"
    )
    del teacher
    log_phase(rank, "rank_complete")
    return dist


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-sha256", required=True)
    args = parser.parse_args(argv)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id), "invalid run ID")
    require(re.fullmatch(r"[a-f0-9]{64}", args.code_sha256), "invalid code hash")
    identity = allocation_identity(os.environ)
    mount = local_paths(args.work_dir, args.output_dir)
    runtime = runtime_identity()
    snapshots = pinned_cache(args.hf_home)
    memory = memory_envelope()
    os.environ.update(
        HF_HOME=str(args.hf_home),
        HF_HUB_CACHE=str(args.hf_home / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="12",
        MKL_NUM_THREADS="12",
        TOKENIZERS_PARALLELISM="false",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_path = args.output_dir / f"rank-{identity['rank']}.json"
    require(
        not rank_path.exists() and not (args.output_dir / "checkpoint").exists(),
        "preflight output already used",
    )
    evidence = {
        "kind": "sdsc_h100_training_preflight_candidate_v1",
        "passed": False,
        "g0_passed": False,
        "execution_class_certified": False,
        "uses_blackwell_certificate": False,
        "scope": "synthetic_one_global64_optimizer_window_and_same_world_state_restore",
        "not_tested": [
            "teacher_demo_quality",
            "full_G0",
            "multistep_training_resume_equivalence",
            "other_world_sizes",
            "persistent_storage_publication",
            "full_dataset_host_memory",
        ],
        "run_id": args.run_id,
        "job_id": identity["job_id"],
        "code_sha256": args.code_sha256,
        "allocation": identity,
        "node": socket.gethostname(),
        "runtime": runtime,
        "pinned_cache": snapshots,
        "node_local_mount": mount,
        "initial_cgroup_memory": memory,
    }
    started, distributed = time.monotonic(), None

    def interrupted(signum, _frame):
        raise InterruptedError(f"preflight received signal {signum}")

    old_handlers = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        distributed = run_canary(args, identity, evidence)
        evidence["passed"] = True
        evidence["elapsed_seconds"] = time.monotonic() - started
        publish_json(rank_path, evidence)
        reports = [None] * WORLD_SIZE
        distributed.all_gather_object(reports, evidence)
        require(all(report["passed"] for report in reports), "one rank did not pass")
        if identity["rank"] == 0:
            publish_json(
                args.output_dir / "preflight.json",
                {
                    "kind": evidence["kind"],
                    "task": "qwen3-v2-preflight",
                    "passed": True,
                    "exit_code": 0,
                    "world_size": identity["world_size"],
                    "g0_passed": False,
                    "execution_class_certified": False,
                    "uses_blackwell_certificate": False,
                    "run_id": args.run_id,
                    "job_id": identity["job_id"],
                    "code_sha256": args.code_sha256,
                    "staged_hf_home": str(args.hf_home),
                    "ranks": reports,
                },
            )
        distributed.monitored_barrier(timeout=timedelta(seconds=120), wait_all_ranks=True)
    except BaseException as error:
        evidence.update(
            passed=False, error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started
        )
        failure = args.output_dir / f"rank-{identity['rank']}-failure.json"
        with contextlib.suppress(OSError):
            publish_json(failure, evidence)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        cleanup = distributed or sys.modules.get("torch.distributed")
        if cleanup is not None and cleanup.is_initialized():
            cleanup.destroy_process_group()
    print(json.dumps({"rank": identity["rank"], "passed": True, "g0_passed": False}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
