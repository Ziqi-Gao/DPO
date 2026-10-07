#!/usr/bin/env python3
"""Four-H100 pilot runtime canary, not a pilot result or execution certificate.

Use torchrun --standalone --nnodes=1 --nproc-per-node=4. Explicit node-local
work/cache/output and clean pinned science checkout are required. The caller
owns Slurm submission, signal supervision, and durable publication of outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import resource
import signal
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.dont_write_bytecode = True
WORLD_SIZE = 4
THREADS = 6
SCIENCE_HEAD = "0215c356355b29b5e2b407978a207db2156719e1"
DEPENDENCIES = {
    "accelerate": "1.10.1",
    "datasets": "4.0.0",
    "huggingface-hub": "0.36.2",
    "matplotlib": "3.10.5",
    "nvidia-nccl-cu12": "2.27.3",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "pandas": "2.3.2",
    "pyarrow": "21.0.0",
    "pydantic": "2.11.7",
    "PyYAML": "6.0.2",
    "safetensors": "0.5.3",
    "scipy": "1.16.1",
    "statsmodels": "0.14.6",
    "tabulate": "0.9.0",
    "tokenizers": "0.22.0",
    "torch": "2.8.0+cu128",
    "transformer-lens": "2.16.1",
    "transformers": "4.56.2",
    "trl": "0.22.2",
}


def helper():
    spec = importlib.util.spec_from_file_location(
        "_pilot_stateless", Path(__file__).with_name("sdsc_training_preflight.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = helper()
require = shared.require
MODELS = shared.MODELS
HOST_BYTES = shared.HOST_BYTES
# These helpers are world-size-independent; no mutable two-rank constants change.
local_paths = shared.local_paths
pinned_cache = shared.pinned_cache
memory_envelope = shared.memory_envelope
publish_json = shared.publish_json
canary_rows = shared.canary_rows
checkpoint_names = shared.checkpoint_names
checkpoint_file = shared.checkpoint_file
verify_checkpoint = shared.verify_checkpoint
tensor_digest = shared.tensor_digest
log_phase = shared.log_phase


def runtime_identity(version=importlib.metadata.version, python_version=None):
    actual_python = python_version or platform.python_version()
    require(actual_python == "3.12.13", "pilot preflight requires Python 3.12.13")
    actual = {name: version(name) for name in DEPENDENCIES}
    require(actual == DEPENDENCIES, f"pilot preflight dependencies differ: {actual}")
    return {"python": actual_python, "executable": sys.executable, "packages": actual}


def allocation_identity(environment):
    require(re.fullmatch(r"[1-9][0-9]*", environment.get("SLURM_JOB_ID", "")), "Slurm job ID missing")
    require(environment.get("SLURM_CPUS_PER_TASK") == "24", "pilot preflight requires 24 CPUs")
    require(environment.get("SLURM_MEM_PER_NODE") == "196608", "pilot preflight requires 192 GiB")
    world = int(environment.get("WORLD_SIZE", "0"))
    rank, local_rank = int(environment.get("RANK", "-1")), int(environment.get("LOCAL_RANK", "-1"))
    require(
        world == 4 and rank in range(4) and local_rank == rank and environment.get("LOCAL_WORLD_SIZE") == "4",
        "pilot preflight requires single-node torchrun with exactly four ranks",
    )
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    require(
        len(devices) == 4
        and len(set(devices)) == 4
        and all(value and value.strip() == value for value in devices),
        "CUDA visibility must expose exactly four unique GPUs",
    )
    return {
        "job_id": environment["SLURM_JOB_ID"],
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "cuda_visible_devices": visible,
        "threads_per_rank": THREADS,
    }


def science_identity(root):
    require(
        root.is_absolute() and root.is_dir() and ".." not in root.parts,
        "science-root must be an existing absolute directory",
    )
    require(not any(path.is_symlink() for path in (root, *root.parents)), "science-root traverses a symlink")

    def git(*argv):
        return subprocess.run(
            ["git", "-C", str(root), *argv], capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()

    require(git("rev-parse", "--show-toplevel") == str(root), "science-root is not a Git checkout root")
    head = git("rev-parse", "HEAD")
    require(head == SCIENCE_HEAD, "science HEAD differs from pinned producer")
    require(
        not git("status", "--porcelain=v1", "--untracked-files=all", "--ignored"),
        "scientific checkout contains modified, untracked, or ignored files",
    )
    lock = json.loads((root / "deployments/qwen3_v2_g0/dependency-lock.json").read_text())
    require(
        {row["name"]: row["version"] for row in lock["packages"]}
        == {name: value for name, value in DEPENDENCIES.items() if name != "trl"},
        "science dependency lock changed",
    )
    return {"head": head, "clean": True, "root": str(root)}


def staged_cache(work, hf_home):
    require(
        hf_home.is_absolute() and work in hf_home.parents and ".." not in hf_home.parts,
        "HF_HOME must be staged under node-local work-dir",
    )
    require(not any(path.is_symlink() for path in (hf_home, *hf_home.parents)), "HF_HOME traverses a symlink")
    local_paths(work, hf_home)
    local_paths(hf_home, hf_home / "mount-probe")
    return pinned_cache(hf_home)


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


def state_digest(torch, value):
    """Hash tensor bytes and all non-tensor optimizer/scheduler state recursively."""
    digest = hashlib.sha256()

    def walk(item):
        if torch.is_tensor(item):
            digest.update(b"tensor:")
            digest.update(tensor_digest(torch, [("value", item)]).encode())
        elif isinstance(item, dict):
            digest.update(b"mapping:" + str(len(item)).encode() + b":")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                walk(key)
                walk(item[key])
            digest.update(b";end-mapping;")
        elif isinstance(item, list | tuple):
            digest.update(type(item).__name__.encode() + b":" + str(len(item)).encode() + b":")
            for child in item:
                walk(child)
            digest.update(b";end-sequence;")
        elif item is None or type(item) in (str, bool, int, float):
            require(not isinstance(item, float) or math.isfinite(item), "nonfinite state metadata")
            digest.update(
                type(item).__name__.encode() + b":" + json.dumps(item, allow_nan=False).encode() + b";"
            )
        else:
            raise ValueError("unsupported state metadata type: " + type(item).__name__)

    walk(value)
    return digest.hexdigest()


def state_fingerprints(torch, student, optimizer, scheduler, device):
    return {
        "model": tensor_digest(torch, student.named_parameters()),
        "optimizer": state_digest(torch, optimizer.state_dict()),
        "scheduler": state_digest(torch, scheduler.state_dict()),
        "cpu_rng": state_digest(torch, torch.get_rng_state()),
        "cuda_rng": state_digest(torch, torch.cuda.get_rng_state(device)),
    }


def validate_rank_reports(reports, identity):
    require(len(reports) == 4, "missing rank reports")
    for rank, report in enumerate(reports):
        allocation = report.get("allocation", {})
        require(
            report.get("passed") is True
            and allocation.get("rank") == rank
            and allocation.get("local_rank") == rank
            and allocation.get("world_size") == 4
            and allocation.get("job_id") == identity["job_id"]
            and allocation.get("cuda_visible_devices") == identity["cuda_visible_devices"],
            "rank report identity/visibility mismatch",
        )
        require(report.get("local_global_slots") == list(range(rank, 64, 4)), "rank sample shards differ")
        require(report.get("reserved_global_nonpadding_tokens") == 64 * 1536, "global token count differs")
        require(
            report.get("full_state_optimizer_scheduler_rng_restore") is True,
            "complete state restore did not pass on every rank",
        )
    return reports


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
    from trl import GRPOConfig, GRPOTrainer

    sys.path.insert(0, str(args.science_root / "src"))
    from posttrain_circuits.learning.supervision.losses import verified_replay_loss
    from posttrain_circuits.learning.training.execution_safety_kernel import batch_token_contract
    from posttrain_circuits.learning.training.fsdp_contract import (
        full_state_dict_options,
        validate_model_fsdp_sharding,
    )
    from posttrain_circuits.models.loading import assert_tokenizer_compatible, load_model_and_tokenizer
    from posttrain_circuits.models.prompt_protocol import format_model_prompt

    evidence["trl_api_import"] = {
        "GRPOConfig": GRPOConfig.__name__,
        "GRPOTrainer": GRPOTrainer.__name__,
        "training_executed": False,
    }
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
        "four H100 80GB GPUs required",
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.set_num_threads(6)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats(device)
    dist.init_process_group("gloo", timeout=timedelta(seconds=600))
    data_group = dist.new_group(backend="nccl", timeout=timedelta(seconds=120))
    value = torch.tensor(float(rank + 1), device=device)
    work = dist.all_reduce(value, group=data_group, async_op=True)
    work.wait(timeout=timedelta(seconds=120))
    require(value.item() == 10.0, "first NCCL all-reduce returned incorrect value")
    evidence.update(
        nccl_all_reduce=True,
        gpu={
            "name": properties[rank].name,
            "total_memory_bytes": properties[rank].total_memory,
            "compute_capability": [9, 0],
            "logical_device": rank,
        },
    )
    model_config = yaml.safe_load((args.science_root / "configs/model/qwen3_v2_1p7b.yaml").read_text())
    teacher_config = yaml.safe_load(
        (args.science_root / "configs/teacher/qwen3_v2_teacher_8b.yaml").read_text()
    )
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
        and contract["microbatch_schedule_by_rank"][rank] == [4] * 4,
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
    for microstep in range(4):
        input_ids = torch.tensor(rows[microstep * 4 : (microstep + 1) * 4], device=device)
        response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        response_mask[:, -256:] = True
        with contextlib.nullcontext() if microstep == 3 else student.no_sync():
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
    saved_optim = state_digest(torch, optimizer.state_dict())
    runtime = {
        "scheduler": scheduler.state_dict(),
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(device).clone(),
    }
    saved_state = state_fingerprints(torch, student, optimizer, scheduler, device)
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
    perturbed_state = state_fingerprints(torch, student, optimizer, scheduler, device)
    require(
        all(perturbed_state[name] != value for name, value in saved_state.items()),
        "perturbation did not change every saved state category",
    )
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
        and state_digest(torch, optimizer.state_dict()) == saved_optim
        and scheduler.state_dict() == runtime["scheduler"]
        and torch.equal(torch.get_rng_state(), runtime["cpu_rng"])
        and torch.equal(torch.cuda.get_rng_state(device), runtime["cuda_rng"]),
        "same-world checkpoint restore differs",
    )
    restored_state = state_fingerprints(torch, student, optimizer, scheduler, device)
    require(restored_state == saved_state, "complete optimizer/scheduler/RNG hashes differ after restore")
    evidence["state_restore_sha256"] = {
        "saved": saved_state,
        "perturbed": perturbed_state,
        "restored": restored_state,
        "every_category_perturbed": True,
        "all_exact": True,
    }
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
    parser.add_argument("--science-root", type=Path, required=True)
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
    science = science_identity(args.science_root)
    snapshots = staged_cache(args.work_dir, args.hf_home)
    memory = memory_envelope()
    os.environ.update(
        HF_HOME=str(args.hf_home),
        HF_HUB_CACHE=str(args.hf_home / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="6",
        MKL_NUM_THREADS="6",
        TOKENIZERS_PARALLELISM="false",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Unique rank claims allow concurrent fresh torchrun ranks and forbid implicit retries.
    claim = args.output_dir / f"rank-{identity['rank']}.claim"
    with claim.open("x") as stream:
        stream.write(identity["job_id"] + "\n")
    require(
        not (args.output_dir / "preflight.json").exists() and not (args.output_dir / "checkpoint").exists(),
        "pilot preflight output already used",
    )
    evidence = {
        "kind": "sdsc_four_h100_pilot_preflight_v1",
        "passed": False,
        "g0_passed": False,
        "pilot_passed": False,
        "execution_class_certified": False,
        "uses_blackwell_certificate": False,
        "scope": "synthetic_one_global64_optimizer_window_and_same_world_state_restore",
        "not_tested": [
            "all_pilot_methods",
            "TRL_GRPO_training",
            "teacher_demo_quality",
            "full_G0",
            "full_pilot",
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
        "science": science,
        "pinned_cache": snapshots,
        "node_local_mount": mount,
        "initial_cgroup_memory": memory,
    }
    started, distributed = time.monotonic(), None

    def interrupted(signum, _frame):
        raise InterruptedError(f"pilot preflight received signal {signum}")

    handlers = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        distributed = run_canary(args, identity, evidence)
        evidence.update(passed=True, elapsed_seconds=time.monotonic() - started)
        publish_json(args.output_dir / f"rank-{identity['rank']}.json", evidence)
        reports = [None] * WORLD_SIZE
        distributed.all_gather_object(reports, evidence)
        validate_rank_reports(reports, identity)
        distributed.monitored_barrier(timeout=timedelta(seconds=120), wait_all_ranks=True)

        def publish_success():
            if identity["rank"] == 0:
                publish_json(
                    args.output_dir / "preflight.json",
                    {
                        "kind": evidence["kind"],
                        "task": "qwen3-v2-pilot-preflight",
                        "passed": True,
                        "exit_code": 0,
                        "world_size": 4,
                        "g0_passed": False,
                        "pilot_passed": False,
                        "execution_class_certified": False,
                        "uses_blackwell_certificate": False,
                        "run_id": args.run_id,
                        "job_id": identity["job_id"],
                        "code_sha256": args.code_sha256,
                        "staged_hf_home": str(args.hf_home),
                        "ranks": reports,
                    },
                )

        collective_phase(distributed, publish_success)
    except BaseException as error:
        evidence.update(
            passed=False, error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started
        )
        with contextlib.suppress(OSError):
            publish_json(args.output_dir / f"rank-{identity['rank']}-failure.json", evidence)
        raise
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
        cleanup = distributed or sys.modules.get("torch.distributed")
        if cleanup is not None and cleanup.is_initialized():
            cleanup.destroy_process_group()
    print(json.dumps({"rank": identity["rank"], "passed": True, "pilot_passed": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
