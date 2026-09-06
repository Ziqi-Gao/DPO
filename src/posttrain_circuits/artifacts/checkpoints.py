"""Atomic checkpoints including optimizer, scheduler, RNG, and rollout state."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping

from posttrain_circuits.artifacts.hashing import sha256_file, sha256_value
from posttrain_circuits.artifacts.io import atomic_torch_save


def accelerator_state_file_hashes(
    root: Path,
    *,
    expected_run_root: Path,
) -> dict[str, str]:
    """Hash an external Accelerator state tree without following symlinks."""

    root = Path(root)
    expected_run_root = Path(expected_run_root)
    lexical_run_root = expected_run_root.absolute()
    lexical_root = root.absolute()
    try:
        relative = lexical_root.relative_to(lexical_run_root)
    except ValueError as error:
        raise ValueError("Accelerator state root is outside its expected run root") from error
    current = lexical_run_root
    if current.is_symlink():
        raise ValueError(f"expected run root must not be a symlink: {current}")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Accelerator state path must not traverse symlinks: {current}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_run_root = expected_run_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("Accelerator state root or expected run root is missing") from error
    if not resolved_root.is_dir():
        raise ValueError(f"Accelerator state root must be a directory: {root}")
    if resolved_root != resolved_run_root and not resolved_root.is_relative_to(resolved_run_root):
        raise ValueError(
            f"Accelerator state root is outside its expected run root: {resolved_root}"
        )

    hashes: dict[str, str] = {}

    def visit(directory: Path) -> None:
        children = sorted(directory.iterdir(), key=lambda path: path.name)
        if directory != resolved_root and not children:
            raise ValueError(
                f"Accelerator state tree contains an unbound empty directory: {directory}"
            )
        for child in children:
            if child.is_symlink():
                raise ValueError(f"Accelerator state tree must not contain symlinks: {child}")
            if child.is_dir():
                visit(child)
            elif child.is_file():
                relative = child.relative_to(resolved_root).as_posix()
                hashes[relative] = sha256_file(child)
            else:
                raise ValueError(
                    f"Accelerator state tree contains a non-regular entry: {child}"
                )

    visit(resolved_root)
    if not hashes:
        raise ValueError("Accelerator state directory is empty")
    return dict(sorted(hashes.items()))


def validate_accelerator_state_directory(
    root: Path,
    *,
    expected_run_root: Path,
    expected_files: dict[str, str],
    expected_sha256: str,
) -> dict[str, str]:
    """Reject missing, extra, replaced, relocated, or symlinked state files."""

    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("checkpoint lacks Accelerator state file hashes")
    for relative, digest in expected_files.items():
        if not isinstance(relative, str):
            raise ValueError("Accelerator state file name must be a string")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValueError(f"Accelerator state file name is not canonical: {relative!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Accelerator state file hash is invalid: {relative!r}")
    observed = accelerator_state_file_hashes(
        root,
        expected_run_root=expected_run_root,
    )
    if observed != expected_files:
        raise ValueError("Accelerator state directory content differs from its checkpoint binding")
    if expected_sha256 != sha256_value(observed):
        raise ValueError("Accelerator state directory digest differs from its checkpoint binding")
    return observed


def validate_native_trainer_checkpoint_files(
    files: object,
    *,
    world_size: int,
) -> dict[str, str]:
    """Require an exact, canonical inventory for a semantic Trainer resume.

    Required files are classified by their normalized *relative file name*.
    Directory components therefore cannot impersonate ``optimizer.pt`` or a
    rank RNG file.  The accepted alternatives are the exact names emitted by
    Transformers 4.56.2 and Accelerate 1.10.1 for one model and one optimizer;
    numeric aliases for additional model/optimizer objects are not accepted.
    """

    if not isinstance(files, dict) or not files:
        raise ValueError("native Trainer checkpoint has no bound files")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("native Trainer checkpoint world size is invalid")
    canonical: set[str] = set()
    for relative, digest in files.items():
        if not isinstance(relative, str):
            raise ValueError("native Trainer checkpoint contains a non-string file name")
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != relative
            or any(part in {"", "."} for part in path.parts)
        ):
            raise ValueError(f"native Trainer checkpoint file name is not canonical: {relative!r}")
        if relative in canonical:
            raise ValueError(f"native Trainer checkpoint duplicates a file: {relative}")
        canonical.add(relative)
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"native Trainer checkpoint file hash is invalid: {relative!r}")

    optional_top_level = {
        "config.json",
        "generation_config.json",
        "training_args.bin",
        "scaler.pt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "chat_template.jinja",
        "README.md",
    }
    model_top_level = {
        name
        for name in canonical
        if re.fullmatch(
            r"(?:model(?:-\d{5}-of-\d{5})?\.safetensors|"
            r"pytorch_model(?:-\d{5}-of-\d{5})?\.bin|"
            r"model\.safetensors\.index\.json|pytorch_model\.bin\.index\.json)",
            name,
        )
    }
    optimizer_top_level = canonical & {"optimizer.pt", "optimizer.bin"}
    scheduler_top_level = canonical & {"scheduler.pt"}
    rng_top_level = {
        name for name in canonical if re.fullmatch(r"rng_state(?:_\d+)?\.pth", name)
    }
    fsdp_rows: dict[tuple[str, int], set[str]] = {}
    fsdp_files: set[str] = set()
    for relative in canonical:
        match = re.fullmatch(
            r"(pytorch_model_fsdp|optimizer)_0/(\.metadata|__\d+_\d+\.distcp)",
            relative,
        )
        if match is None:
            continue
        key = (match.group(1), 0)
        fsdp_rows.setdefault(key, set()).add(match.group(2))
        fsdp_files.add(relative)
    incomplete_fsdp = [
        f"{kind}_{index}"
        for (kind, index), names in sorted(fsdp_rows.items())
        if ".metadata" not in names or not any(name.endswith(".distcp") for name in names)
    ]
    if incomplete_fsdp:
        raise ValueError(
            f"native Trainer checkpoint has incomplete FSDP directories: {incomplete_fsdp}"
        )

    trainer_files = canonical & {"trainer_state.json"}
    model_fsdp = {key for key in fsdp_rows if key[0] == "pytorch_model_fsdp"}
    optimizer_fsdp = {key for key in fsdp_rows if key[0] == "optimizer"}
    full_fsdp_model = canonical & {"pytorch_model_fsdp.bin"}
    unsharded_model = model_top_level & {"model.safetensors", "pytorch_model.bin"}
    shard_rows: list[tuple[str, int, int]] = []
    for name in model_top_level:
        match = re.fullmatch(
            r"(model|pytorch_model)-(\d{5})-of-(\d{5})\.(?:safetensors|bin)",
            name,
        )
        if match is not None:
            shard_rows.append((match.group(1), int(match.group(2)), int(match.group(3))))
    shard_formats = {row[0] for row in shard_rows}
    shard_totals = {row[2] for row in shard_rows}
    shard_inventory_valid = False
    if shard_rows and len(shard_formats) == 1 and len(shard_totals) == 1:
        total = next(iter(shard_totals))
        shard_inventory_valid = (
            total > 0
            and {row[1] for row in shard_rows} == set(range(1, total + 1))
            and (
                "model.safetensors.index.json"
                if next(iter(shard_formats)) == "model"
                else "pytorch_model.bin.index.json"
            )
            in model_top_level
        )
    top_level_model_valid = (
        len(unsharded_model) == 1
        and not shard_rows
        and not any(name.endswith(".index.json") for name in model_top_level)
    ) or (
        not unsharded_model
        and shard_inventory_valid
        and len(model_top_level) == len(shard_rows) + 1
    )
    model_variants = (
        int(top_level_model_valid)
        + int(full_fsdp_model == {"pytorch_model_fsdp.bin"})
        + int(bool(model_fsdp))
    )
    optimizer_variant_valid = (
        (top_level_model_valid and optimizer_top_level == {"optimizer.pt"})
        or (
            full_fsdp_model == {"pytorch_model_fsdp.bin"}
            and optimizer_top_level == {"optimizer.bin"}
        )
        or (bool(model_fsdp) and bool(optimizer_fsdp) and not optimizer_top_level)
    )
    missing: list[str] = []
    if len(trainer_files) != 1:
        missing.append("trainer state")
    if model_variants != 1 or len(model_fsdp) > 1:
        missing.append("exactly one model-state format")
    if not optimizer_variant_valid:
        missing.append("exactly one optimizer-state format")
    if len(optimizer_top_level) > 1 or len(optimizer_fsdp) > 1:
        missing.append("exactly one optimizer state")
    if len(scheduler_top_level) != 1:
        missing.append("scheduler state")

    if world_size == 1:
        if rng_top_level != {"rng_state.pth"}:
            missing.append("exactly one rank-0 RNG state")
    else:
        expected_rng = {f"rng_state_{rank}.pth" for rank in range(world_size)}
        if rng_top_level != expected_rng:
            missing.append(f"RNG state for exactly ranks {list(range(world_size))}")

    recognized = (
        optional_top_level
        | trainer_files
        | model_top_level
        | full_fsdp_model
        | optimizer_top_level
        | scheduler_top_level
        | rng_top_level
        | fsdp_files
    )
    extras = sorted(canonical - recognized)
    if extras:
        raise ValueError(f"native Trainer checkpoint contains unexpected files: {extras}")
    if missing:
        raise ValueError(f"native Trainer checkpoint is incomplete or ambiguous: {missing}")
    return files


def torch_state_hash(value: Any) -> str:
    """Hash nested PyTorch state with explicit type and length framing."""

    import torch

    digest = hashlib.sha256()

    def framed(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"T")
            tensor = item.detach().cpu().contiguous()
            framed(str(tensor.dtype).encode())
            framed(str(tuple(tensor.shape)).encode())
            framed(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"D")
            framed(str(len(item)).encode())
            for key in sorted(item, key=lambda value: repr(value)):
                update(key)
                update(item[key])
        elif isinstance(item, list):
            digest.update(b"L")
            framed(str(len(item)).encode())
            for child in item:
                update(child)
        elif isinstance(item, tuple):
            digest.update(b"Q")
            framed(str(len(item)).encode())
            for child in item:
                update(child)
        else:
            digest.update(b"V")
            framed(type(item).__qualname__.encode())
            framed(repr(item).encode())

    update(value)
    return digest.hexdigest()


def model_update_evidence(
    baseline: Mapping[str, Any],
    final: Mapping[str, Any],
) -> tuple[float, str]:
    """Recompute a full-state floating-parameter displacement and final hash."""

    import torch

    if not baseline or not final or set(baseline) != set(final):
        raise ValueError("baseline and final model state inventories differ")
    squared = 0.0
    floating_tensors = 0
    for name in sorted(final):
        before = baseline[name]
        after = final[name]
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            raise ValueError(f"model state is not tensor-valued: {name}")
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"baseline/final model tensor metadata differs: {name}")
        if torch.is_floating_point(after) or torch.is_complex(after):
            floating_tensors += 1
            if torch.is_complex(after):
                delta = after.detach().cpu().to(torch.complex128) - before.detach().cpu().to(
                    torch.complex128
                )
                squared += float(torch.sum(torch.abs(delta) ** 2).item())
            else:
                delta = after.detach().cpu().double() - before.detach().cpu().double()
                squared += float(torch.sum(delta * delta).item())
    norm = math.sqrt(squared)
    if floating_tensors < 1 or not math.isfinite(norm):
        raise ValueError("model update norm is not finite or has no floating tensors")
    return norm, torch_state_hash(dict(final))


def load_checkpoint_model_state(path: Path) -> dict[str, Any]:
    """Load the exact model mapping from a bound checkpoint, mmap when supported."""

    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError(f"baseline checkpoint lacks model state: {path}")
    model = dict(payload["model"])
    if not model:
        raise ValueError(f"baseline checkpoint model state is empty: {path}")
    return model


def checkpoint_runtime_state_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    """Bind each independently resumable Factorial checkpoint state class."""

    fields = [
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "trainer_state",
        "token_budget",
        "prompt_scheduler_by_rank",
        "state_source_by_rank",
        "manifest_hashes",
    ]
    # Allocation-neutral checkpoints carry rank-local scientific counters in
    # addition to the historical rank-0 alias.  Keep legacy checkpoints
    # readable, but bind the complete table whenever the producer emits it.
    if "trainer_state_by_rank" in payload:
        fields.append("trainer_state_by_rank")
    if payload.get("format") == "accelerate_fsdp_full_export_v1":
        # These are independently resumable state/identity classes for the
        # production Accelerate checkpoint.  ``None`` is a meaningful scaler
        # state under bf16, so presence rather than truthiness is required.
        fields.extend(
            (
                "scaler",
                "rank_shard_hashes",
                "accelerate_state_files",
                "accelerate_state_sha256",
            )
        )
    missing = [field for field in fields if field not in payload]
    if missing:
        raise ValueError(f"checkpoint runtime state is incomplete: {missing}")
    return {field: torch_state_hash(payload[field]) for field in fields}


def extend_checkpoint_ancestry(observed: object, checkpoint: Path) -> list[str]:
    """Validate unique content-addressed ancestry and append one checkpoint."""

    if not isinstance(observed, list):
        raise ValueError("checkpoint resume ancestry must be a list")
    normalized: list[str] = []
    for value in observed:
        if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("checkpoint resume ancestry is not content-addressed")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("checkpoint resume ancestry contains a cycle")
    current = f"sha256:{sha256_file(checkpoint)}"
    if current in normalized:
        raise ValueError("checkpoint resume ancestry contains a repeated checkpoint")
    return [*normalized, current]


def _capture_rng_state() -> dict[str, Any]:
    import base64
    import pickle
    import random

    import numpy as np
    import torch

    def encode(value: object) -> str:
        return base64.b64encode(pickle.dumps(value)).decode("ascii")

    return {
        "python": encode(random.getstate()),
        "numpy": encode(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": (
            [state.tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(payload: dict[str, Any]) -> None:
    import base64
    import pickle
    import random

    import numpy as np
    import torch

    def decode(value: str) -> object:
        return pickle.loads(base64.b64decode(value.encode("ascii")))

    random.setstate(decode(str(payload["python"])))
    np.random.set_state(decode(str(payload["numpy"])))  # type: ignore[arg-type]
    torch.set_rng_state(torch.tensor(payload["torch_cpu"], dtype=torch.uint8))
    if payload["torch_cuda"] and torch.cuda.is_available():
        states = [
            torch.tensor(state, dtype=torch.uint8)
            for state in payload["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(states)


def save_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    prompt_scheduler_state: dict[str, Any],
    global_step: int,
    policy_version: int,
    online_rollout_round: int,
    resolved_config: dict[str, Any],
    manifest_hashes: dict[str, str],
    state_source_state: dict[str, Any] | None = None,
    trainer_state: dict[str, Any] | None = None,
    accelerator_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    git_commit: str = "unavailable",
    implementation_dirty: bool = True,
    dependency_versions: dict[str, str] | None = None,
    resume_ancestry: list[str] | None = None,
    rank_shard_hashes: list[str] | None = None,
    world_size: int = 1,
    prompt_scheduler_states: list[dict[str, Any]] | None = None,
    state_source_states: list[dict[str, Any]] | None = None,
    parameter_update_norm: float | None = None,
    final_model_state_hash: str | None = None,
    update_norm_baseline_checkpoint_path: str | None = None,
    update_norm_baseline_checkpoint_sha256: str | None = None,
) -> None:
    import torch

    model_state = model.state_dict()
    prompt_states = prompt_scheduler_states or [prompt_scheduler_state]
    source_state = state_source_state or {}
    source_states = state_source_states or [source_state]
    runtime_trainer_state = trainer_state or {}
    payload = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler_state,
        "accelerator": accelerator_state,
        "rng": _capture_rng_state(),
        "prompt_scheduler": prompt_scheduler_state,
        "prompt_scheduler_by_rank": prompt_states,
        "state_source": source_state,
        "state_source_by_rank": source_states,
        "trainer_state": runtime_trainer_state,
        "token_budget": runtime_trainer_state.get("token_budget", {}),
        "global_step": global_step,
        "world_size": world_size,
        "policy_version": policy_version,
        "online_rollout_round": online_rollout_round,
        "resolved_config": resolved_config,
        "manifest_hashes": manifest_hashes,
        "git_commit": git_commit,
        "implementation_dirty": implementation_dirty,
        "dependency_versions": dependency_versions or {"torch": torch.__version__},
        "resume_ancestry": list(resume_ancestry or []),
        "rank_shard_hashes": list(rank_shard_hashes or []),
        "parameter_update_norm": parameter_update_norm,
        "final_model_state_hash": final_model_state_hash or torch_state_hash(model_state),
        "update_norm_baseline_checkpoint_path": update_norm_baseline_checkpoint_path,
        "update_norm_baseline_checkpoint_sha256": update_norm_baseline_checkpoint_sha256,
        "torch_version": torch.__version__,
    }
    atomic_torch_save(path, payload)


def load_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    expected_manifest_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        expected_manifest_hashes is not None
        and payload.get("manifest_hashes") != expected_manifest_hashes
    ):
        raise ValueError("checkpoint scientific manifest hashes differ from the resumed run")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng_state(payload["rng"])
    return payload
