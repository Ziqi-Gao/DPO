"""Short four-rank CUDA/NCCL and pinned-Qwen execution gate."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from posttrain_circuits.core.config import compose_config
from posttrain_circuits.core.hashing import sha256_value
from posttrain_circuits.core.manifests import atomic_write_json, utc_now
from posttrain_circuits.core.provenance import require_git_output
from posttrain_circuits.models.loading import load_model_and_tokenizer


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
    if rank == 0:
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
            "devices": device_rows,
            "git_commit": git_commit,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "created_at": utc_now(),
        }
        payload["sha256"] = sha256_value(payload)
        atomic_write_json(args.output, payload)
    dist.barrier()
    dist.destroy_process_group()
    if not nccl_passed or not bool(status.item()):
        raise SystemExit("GPU preflight failed")


if __name__ == "__main__":
    main()
