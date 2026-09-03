#!/usr/bin/python3
"""Fixed four-rank Qwen3-v2 GPU preflight for ServerScheduler protocol v2.

The file is both the foreground supervisor and the torchrun worker.  It is
therefore the only project implementation byte stream in the deployment
contract.  Physical GPU selection remains entirely owned by ServerScheduler;
workers use only their process-local logical rank inside the inherited CUDA
visibility envelope.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TASK = "qwen3_v2_gpu_preflight"
PROFILE = "qwen3-v2-gpu-preflight-4gpu"
PROJECT = "OPD"
REPORT_NAME = "gpu_preflight.json"
COMPLETION_NAME = ".opd-scientific-completion.json"
COMPLETION_KIND = "scientific_attempt"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEACHER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_FINGERPRINT = "03ed1280ac090810a530b8ca225c5cb9398ca3d0f22465f67caf56146f75a13d"
CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
PREREGISTRATION_SHA256 = "8d6bdeab0b9302c8824c4709f556c6c41a896bd2cfce21e7794d131d176ba0a4"
GPU_MODEL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
GPU_COUNT = 4
NODE_MEMORY_BYTES = 192 * 1024**3
MINIMUM_HEADROOM_BYTES = max(32 * 1024**3, int(NODE_MEMORY_BYTES * 0.20))
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_PREREG_BYTES = 4 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PROC_SELF_FD = re.compile(r"/proc/self/fd/(0|[1-9][0-9]*)\Z")
POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")
EXPECTED_INPUT_NAMES = (
    "config_binding_sha256",
    "execution_config_sha256",
    "preregistration_sha256",
    "resolved_config_sha256",
    "scientific_config_sha256",
)
JSON_INPUT_NAMES = frozenset(EXPECTED_INPUT_NAMES) - {"preregistration_sha256"}
GATE_NAMES = (
    "allocation_contract",
    "config_binding",
    "cuda_runtime",
    "fsdp_resume",
    "memory_headroom",
    "nccl_all_reduce",
    "offline_pinned_models",
    "real_forward_backward",
)
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
FIXED_ENVIRONMENT = {
    "HF_HOME": "/scr/del6500/OPD/cache/huggingface",
    "HF_HUB_CACHE": "/scr/del6500/OPD/cache/huggingface/hub",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "TMPDIR": "/scr/del6500/OPD/tmp",
}


class PreflightError(RuntimeError):
    """Fail-closed invocation, runtime, or scientific validation error."""


@dataclass(frozen=True)
class ContentHandle:
    name: str
    sha256: str
    descriptor: int


@dataclass(frozen=True)
class Invocation:
    workflow_id: str
    plan_sha256: str
    unit_id: str
    run_id: str
    job_id: str
    attempt: int
    execution_profile: str
    manifest_sha256: str
    allocation_sha256: str
    content_handles: tuple[ContentHandle, ...]
    output_descriptor: int

    @property
    def input_hashes(self) -> dict[str, str]:
        return {item.name: item.sha256 for item in self.content_handles}

    @property
    def execution(self) -> dict[str, Any]:
        return {
            "allocation_sha256": self.allocation_sha256,
            "attempt": self.attempt,
            "execution_profile": self.execution_profile,
            "job_id": self.job_id,
            "manifest_sha256": self.manifest_sha256,
        }


class _SingleValue(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied exactly once")
        setattr(namespace, self.dest, values)


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise PreflightError(f"value is not canonical JSON: {error}") from error


def _sha256_value(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _published_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise PreflightError(f"{name} is not a canonical identifier")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PreflightError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _outer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for option in (
        "workflow-id",
        "plan-sha256",
        "unit-id",
        "run-id",
        "job-id",
        "attempt",
        "execution-profile",
        "manifest-sha256",
        "allocation-sha256",
        "output-attempt-handle",
        "attempt-completion-name",
    ):
        parser.add_argument(
            f"--{option}",
            dest=option.replace("-", "_"),
            action=_SingleValue,
            required=True,
            default=None,
        )
    parser.add_argument(
        "--content-handle",
        action="append",
        nargs=4,
        metavar=("NAME", "KIND", "SHA256", "PROC_PATH"),
        required=True,
    )
    return parser


def _held_descriptor(value: object, *, name: str) -> int:
    if not isinstance(value, str):
        raise PreflightError(f"{name} must be a descriptor path")
    match = PROC_SELF_FD.fullmatch(value)
    if match is None:
        raise PreflightError(f"{name} must use canonical /proc/self/fd/N form")
    descriptor = int(match.group(1))
    if descriptor < 3:
        raise PreflightError(f"{name} must not alias a standard stream")
    try:
        os.fstat(descriptor)
    except OSError as error:
        raise PreflightError(f"{name} descriptor is not open") from error
    return descriptor


def _parse_outer(argv: Sequence[str] | None) -> Invocation:
    args = _outer_parser().parse_args(argv)
    if args.attempt_completion_name != COMPLETION_NAME:
        raise PreflightError("attempt completion name differs from the fixed ABI")
    if args.execution_profile != PROFILE:
        raise PreflightError("execution profile differs from the fixed GPU profile")
    if not isinstance(args.attempt, str) or not POSITIVE_INTEGER.fullmatch(args.attempt):
        raise PreflightError("attempt must be a canonical positive integer")
    handles: list[ContentHandle] = []
    for raw_name, kind, digest, proc_path in args.content_handle:
        name = _identifier(raw_name, name="content handle name")
        if kind != "file":
            raise PreflightError(f"GPU preflight input {name!r} must be a file")
        handles.append(
            ContentHandle(
                name=name,
                sha256=_sha256(digest, name=f"content handle {name!r}"),
                descriptor=_held_descriptor(proc_path, name=f"content handle {name!r}"),
            )
        )
    if tuple(item.name for item in handles) != EXPECTED_INPUT_NAMES:
        raise PreflightError("content handles differ from the fixed GPU-preflight ABI")
    descriptors = tuple(item.descriptor for item in handles)
    if len(set(descriptors)) != len(descriptors):
        raise PreflightError("content handles alias the same descriptor")
    output_descriptor = _held_descriptor(
        args.output_attempt_handle, name="output attempt handle"
    )
    if output_descriptor in descriptors:
        raise PreflightError("output attempt aliases a content descriptor")
    output_stat = os.fstat(output_descriptor)
    if not stat.S_ISDIR(output_stat.st_mode):
        raise PreflightError("output attempt handle is not a directory")
    return Invocation(
        workflow_id=_identifier(args.workflow_id, name="workflow_id"),
        plan_sha256=_sha256(args.plan_sha256, name="plan_sha256"),
        unit_id=_identifier(args.unit_id, name="unit_id"),
        run_id=_sha256(args.run_id, name="run_id"),
        job_id=_identifier(args.job_id, name="job_id"),
        attempt=int(args.attempt),
        execution_profile=PROFILE,
        manifest_sha256=_sha256(args.manifest_sha256, name="manifest_sha256"),
        allocation_sha256=_sha256(args.allocation_sha256, name="allocation_sha256"),
        content_handles=tuple(handles),
        output_descriptor=output_descriptor,
    )


def _read_held_file(descriptor: int, *, context: str, max_bytes: int) -> bytes:
    before = os.fstat(descriptor)
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    if not stat.S_ISREG(before.st_mode) or flags & os.O_ACCMODE != os.O_RDONLY:
        raise PreflightError(f"{context} must be a held read-only regular file")
    if before.st_size < 1 or before.st_size > max_bytes:
        raise PreflightError(f"{context} has an invalid size")
    raw = os.pread(descriptor, before.st_size, 0)
    after = os.fstat(descriptor)
    identity = lambda row: (  # noqa: E731 - compact immutable inode comparison
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise PreflightError(f"{context} changed while being read")
    return raw


def _strict_json(raw: bytes, *, context: str) -> Any:
    def reject(value: str) -> None:
        raise PreflightError(f"{context} contains a non-finite number: {value}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PreflightError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"{context} is not strict UTF-8 JSON") from error


def _read_inputs(invocation: Invocation) -> tuple[dict[str, Any], bytes]:
    payloads: dict[str, Any] = {}
    prereg = b""
    inodes: set[tuple[int, int]] = set()
    for handle in invocation.content_handles:
        metadata = os.fstat(handle.descriptor)
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in inodes:
            raise PreflightError("content handles alias the same inode")
        inodes.add(inode)
        raw = _read_held_file(
            handle.descriptor,
            context=f"content {handle.name!r}",
            max_bytes=(
                MAX_PREREG_BYTES
                if handle.name == "preregistration_sha256"
                else MAX_INPUT_BYTES
            ),
        )
        if hashlib.sha256(raw).hexdigest() != handle.sha256:
            raise PreflightError(f"content {handle.name!r} differs from its digest")
        if handle.name == "preregistration_sha256":
            prereg = raw
            continue
        payload = _strict_json(raw, context=f"content {handle.name!r}")
        if not isinstance(payload, dict) or raw != _canonical_json(payload).encode("utf-8"):
            raise PreflightError(f"content {handle.name!r} is not canonical JSON object bytes")
        payloads[handle.name] = payload
    if not prereg:
        raise PreflightError("preregistration content is empty")
    return payloads, prereg


def _validate_config(
    invocation: Invocation, payloads: dict[str, Any], prereg: bytes
) -> dict[str, Any]:
    binding = payloads["config_binding_sha256"]
    resolved = payloads["resolved_config_sha256"]
    scientific = payloads["scientific_config_sha256"]
    execution = payloads["execution_config_sha256"]
    if set(binding) != {
        "execution_config_sha256",
        "execution_context",
        "input_artifact_hashes",
        "resolved_config_sha256",
        "schema_version",
        "scientific_config_sha256",
        "storage_locators",
    } or binding.get("schema_version") != 3:
        raise PreflightError("ConfigBinding fields differ from schema v3")
    for name in (
        "execution_config_sha256",
        "resolved_config_sha256",
        "scientific_config_sha256",
    ):
        if binding.get(name) != invocation.input_hashes[name]:
            raise PreflightError(f"ConfigBinding {name} differs from its CAS input")
    if binding.get("input_artifact_hashes") != {
        "prereg_path": invocation.input_hashes["preregistration_sha256"]
    }:
        raise PreflightError("ConfigBinding does not bind the preregistration bytes")
    if binding.get("storage_locators") != {"prereg_path": "prereg/qwen3_v2.yaml"}:
        raise PreflightError("ConfigBinding preregistration locator is not fixed")
    expected_execution = {
        "distributed_process_count": 4,
        "execution_profile": PROFILE,
        "scheduler_protocol": 2,
    }
    if binding.get("execution_context") != expected_execution:
        raise PreflightError("ConfigBinding execution context differs from the fixed topology")
    if execution != {
        "execution_context": expected_execution,
        "schema_version": 3,
        "storage_locators": {"prereg_path": "prereg/qwen3_v2.yaml"},
    }:
        raise PreflightError("execution config projection is invalid")
    if (
        hashlib.sha256(prereg).hexdigest()
        != invocation.input_hashes["preregistration_sha256"]
        or invocation.input_hashes["preregistration_sha256"]
        != PREREGISTRATION_SHA256
    ):
        raise PreflightError("preregistration content identity changed")
    model = resolved.get("model")
    teacher = resolved.get("teacher")
    budget = resolved.get("resource_budget")
    if not all(isinstance(value, dict) for value in (model, teacher, budget)):
        raise PreflightError("resolved GPU-preflight config is incomplete")
    if (
        resolved.get("config_kind") != TASK
        or resolved.get("protocol_track") != "qwen3_v2"
        or resolved.get("artifact_namespace") != "qwen3-v2"
        or resolved.get("prereg_path") != "prereg/qwen3_v2.yaml"
        or resolved.get("prereg_version") != "qwen3_v2"
        or model.get("model_name_or_path") != "Qwen/Qwen3-1.7B"
        or model.get("model_revision") != MODEL_REVISION
        or model.get("tokenizer_revision") != MODEL_REVISION
        or model.get("tokenizer_fingerprint") != TOKENIZER_FINGERPRINT
        or model.get("local_files_only") is not True
        or model.get("trust_remote_code") is not False
        or model.get("torch_dtype") != "bfloat16"
        or model.get("attn_implementation") != "sdpa"
        or teacher.get("model_name_or_path") != "Qwen/Qwen3-8B"
        or teacher.get("model_revision") != TEACHER_REVISION
        or teacher.get("tokenizer_revision") != TEACHER_REVISION
        or teacher.get("tokenizer_fingerprint") != TOKENIZER_FINGERPRINT
        or teacher.get("local_files_only") is not True
        or teacher.get("trust_remote_code") is not False
        or teacher.get("rank_zero_only_training_load") is not True
        or budget
        != {
            "loading_strategy": "low_cpu_mem_student_rank_zero_teacher",
            "minimum_headroom_fraction": 0.2,
            "minimum_headroom_gib": 32,
            "node_memory_gib": 192,
        }
    ):
        raise PreflightError("resolved config differs from the fixed Qwen3-v2 protocol")
    scientific_copy = copy.deepcopy(resolved)
    scientific_copy["prereg_path"] = {
        "content_sha256": invocation.input_hashes["preregistration_sha256"]
    }
    expected_scientific = {
        "config": scientific_copy,
        "input_artifact_hashes": {
            "prereg_path": invocation.input_hashes["preregistration_sha256"]
        },
        "schema_version": 3,
    }
    if scientific != expected_scientific:
        raise PreflightError("scientific config projection is invalid")
    if resolved != _fixed_config():
        raise PreflightError("resolved config has fields outside the fixed preflight contract")
    return resolved


def _fixed_config() -> dict[str, Any]:
    prompt = {
        "add_generation_prompt": True,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "enable_thinking": False,
        "messages": "single_user",
        "name": "qwen3_non_thinking_v1",
    }
    sampling = {
        "do_sample": True,
        "min_p": 0.0,
        "name": "qwen3_non_thinking_sampling_v1",
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    }
    common = {
        "allow_unpinned_revision": False,
        "attn_implementation": "sdpa",
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "prompt_protocol": prompt,
        "sampling_protocol": sampling,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "torch_dtype": "bfloat16",
        "trust_remote_code": False,
    }
    return {
        "artifact_namespace": "qwen3-v2",
        "config_kind": TASK,
        "model": {
            **copy.deepcopy(common),
            "gradient_checkpointing": True,
            "model_name_or_path": "Qwen/Qwen3-1.7B",
            "model_revision": MODEL_REVISION,
            "tokenizer_name_or_path": "Qwen/Qwen3-1.7B",
            "tokenizer_revision": MODEL_REVISION,
            "use_cache": False,
        },
        "prereg_path": "prereg/qwen3_v2.yaml",
        "prereg_version": "qwen3_v2",
        "protocol_track": "qwen3_v2",
        "resource_budget": {
            "loading_strategy": "low_cpu_mem_student_rank_zero_teacher",
            "minimum_headroom_fraction": 0.2,
            "minimum_headroom_gib": 32,
            "node_memory_gib": 192,
        },
        "teacher": {
            **copy.deepcopy(common),
            "gradient_checkpointing": False,
            "model_name_or_path": "Qwen/Qwen3-8B",
            "model_revision": TEACHER_REVISION,
            "rank_zero_only_training_load": True,
            "tokenizer_name_or_path": "Qwen/Qwen3-8B",
            "tokenizer_revision": TEACHER_REVISION,
            "use_cache": True,
        },
    }


def _validate_environment() -> tuple[str, ...]:
    for key, expected in FIXED_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            raise PreflightError(f"fixed environment {key} differs from its deployment")
    if any(os.environ.get(key) != "4" for key in THREAD_KEYS):
        raise PreflightError("per-rank CPU thread environment must be exactly four")
    forbidden = sorted(
        key
        for key in os.environ
        if key.startswith("SERVER_SCHEDULER_")
        or key in {"LD_AUDIT", "LD_LIBRARY_PATH", "LD_PRELOAD", "PATH", "PYTHONHOME", "PYTHONPATH"}
    )
    if forbidden:
        raise PreflightError(f"ambient runtime injection reached GPU handler: {forbidden}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise PreflightError("PYTHONNOUSERSITE=1 is required")
    if not (sys.flags.isolated and sys.flags.ignore_environment) or sys.flags.no_site:
        raise PreflightError("GPU handler requires Python -I with fixed venv site packages")
    expected_versions = {
        "accelerate": "1.10.1",
        "huggingface-hub": "0.36.2",
        "numpy": "1.26.4",
        "safetensors": "0.5.3",
        "tokenizers": "0.22.0",
        "transformers": "4.56.2",
    }
    observed_versions = {
        name: importlib.metadata.version(name) for name in expected_versions
    }
    if observed_versions != expected_versions:
        raise PreflightError("fixed GPU runtime package versions differ from the deployment")
    if not importlib.metadata.version("torch").startswith("2.8.0"):
        raise PreflightError("fixed GPU runtime Torch package version differs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = tuple(visible.split(",")) if visible else ()
    if len(devices) != GPU_COUNT or len(set(devices)) != GPU_COUNT or any(
        not item or item.strip() != item for item in devices
    ):
        raise PreflightError("scheduler-provided CUDA visibility must contain four devices")
    return devices


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    template = str(getattr(tokenizer, "chat_template", None) or "")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_text = backend.to_str() if backend is not None else ""
    content = {
        "backend_tokenizer_sha256": hashlib.sha256(backend_text.encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "special_token_ids": {
            name: getattr(tokenizer, name, None)
            for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
        },
        "tokenizer_class": type(tokenizer).__name__,
        "vocabulary": sorted(
            (str(token), int(index)) for token, index in tokenizer.get_vocab().items()
        ),
    }
    return _sha256_value(content)


def _resolved_commit(value: Any, fallback: str) -> str:
    init_kwargs = getattr(value, "init_kwargs", None)
    candidates = (
        getattr(getattr(value, "config", None), "_commit_hash", None),
        getattr(value, "_commit_hash", None),
        init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None,
    )
    return str(next((item for item in candidates if item), fallback))


def _load_model(config: dict[str, Any], *, training: bool) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_name_or_path"],
        revision=config["tokenizer_revision"],
        trust_remote_code=False,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise PreflightError("tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    fingerprint = _tokenizer_fingerprint(tokenizer)
    template_hash = hashlib.sha256(
        str(getattr(tokenizer, "chat_template", None) or "").encode()
    ).hexdigest()
    if fingerprint != TOKENIZER_FINGERPRINT or template_hash != CHAT_TEMPLATE_SHA256:
        raise PreflightError("offline tokenizer differs from the pinned Qwen3-v2 protocol")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        revision=config["model_revision"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if _resolved_commit(model, config["model_revision"]) != config["model_revision"]:
        raise PreflightError("offline model resolved to an unexpected commit")
    model.config.use_cache = bool(config["use_cache"])
    if training:
        if any(not parameter.requires_grad for parameter in model.parameters()):
            raise PreflightError("student is not a full-parameter training model")
        if config.get("gradient_checkpointing") is True:
            model.gradient_checkpointing_enable()
        model.train()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, tokenizer, fingerprint


def _format_prompt(raw_prompt: str, tokenizer: Any) -> tuple[str, str, str]:
    model_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if "<|im_start|>assistant\n<think>\n\n</think>\n\n" not in model_prompt:
        raise PreflightError("Qwen3 non-thinking assistant prefix is absent")
    return str(model_prompt), _sha256_value(raw_prompt), _sha256_value(str(model_prompt))


def _parameter_checksum(model: Any, device: Any) -> Any:
    import torch

    value = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in model.parameters():
        value += parameter.detach().double().sum()
    return value


def _rank_training(config: dict[str, Any], checkpoint_root: Path) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import (
        ShardedOptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != GPU_COUNT or not 0 <= local_rank < GPU_COUNT:
        raise PreflightError("torchrun topology differs from the fixed four-rank contract")
    if (
        torch.version.cuda != "12.8"
        or not str(torch.__version__).startswith("2.8.0+cu128")
    ):
        raise PreflightError("fixed runtime is not CUDA-enabled PyTorch 2.8.0")
    if torch.cuda.device_count() != GPU_COUNT:
        raise PreflightError("visible CUDA device count differs from the allocation")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    reduced = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(reduced)
    if float(reduced.item()) != 10.0:
        raise PreflightError("NCCL all-reduce returned an unexpected result")

    student_model, student_tokenizer, tokenizer_hash = _load_model(
        config["model"], training=True
    )
    vocab_size = int(student_model.config.vocab_size)
    student = FSDP(student_model, device_id=device, use_orig_params=True)
    teacher = None
    teacher_metadata: list[Any] = [None]
    if rank == 0:
        teacher, teacher_tokenizer, teacher_hash = _load_model(
            config["teacher"], training=False
        )
        probes = (
            "FACTS F01 A RULES R01 A -> Q",
            "<proof> S01: R01(F01) -> Q </proof> <answer>1</answer>",
        )
        if tokenizer_hash != teacher_hash or any(
            student_tokenizer.encode(text, add_special_tokens=False)
            != teacher_tokenizer.encode(text, add_special_tokens=False)
            for text in probes
        ):
            raise PreflightError("student and teacher tokenizers are not compatible")
        teacher = teacher.to(device)
        teacher_metadata[0] = {
            "resolved_teacher_commit": _resolved_commit(teacher, TEACHER_REVISION),
            "tokenizer_fingerprint": tokenizer_hash,
        }
    dist.broadcast_object_list(teacher_metadata, src=0)
    if not isinstance(teacher_metadata[0], dict):
        raise PreflightError("rank-zero teacher metadata broadcast failed")

    raw_prompt = (
        f"RANK-CANARY-{rank}\nFACTS F01: A\nRULES R01: A -> B\n"
        "QUERY: B\nOUTPUT FORMAT: <proof> ... </proof> <answer>0|1</answer>"
    )
    model_prompt, raw_hash, prompt_hash = _format_prompt(raw_prompt, student_tokenizer)
    encoded = student_tokenizer(
        model_prompt, add_special_tokens=False, return_tensors="pt"
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    prompt_hashes: list[Any] = [None] * world_size
    dist.all_gather_object(prompt_hashes, prompt_hash)
    unique_prompts = len(set(str(value) for value in prompt_hashes)) == world_size

    gathered_ids = [torch.empty_like(input_ids) for _ in range(world_size)]
    gathered_masks = [torch.empty_like(attention_mask) for _ in range(world_size)]
    dist.all_gather(gathered_ids, input_ids)
    dist.all_gather(gathered_masks, attention_mask)
    teacher_logits = torch.empty(
        (*input_ids.shape, vocab_size), dtype=torch.bfloat16, device=device
    )
    if rank == 0:
        if teacher is None:
            raise PreflightError("rank zero did not load the teacher")
        with torch.no_grad():
            batched = teacher(
                input_ids=torch.cat(gathered_ids, dim=0),
                attention_mask=torch.cat(gathered_masks, dim=0),
            ).logits
        scatter_rows = list(batched.split(input_ids.shape[0], dim=0))
    else:
        scatter_rows = None
    dist.scatter(teacher_logits, scatter_list=scatter_rows, src=0)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
    loss = functional.kl_div(
        student_logits.float().log_softmax(dim=-1),
        teacher_logits.float().softmax(dim=-1),
        reduction="batchmean",
    )
    loss.backward()
    gradients = [item.grad for item in student.parameters() if item.grad is not None]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(item).all()) for item in gradients
    )
    before = _parameter_checksum(student, device)
    optimizer.step()
    after = _parameter_checksum(student, device)
    update_nonzero = bool((after - before).abs().item() > 0.0)

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
    checkpoint_hash = _sha256_file(rank_path)
    saved = _parameter_checksum(student, device)
    with torch.no_grad():
        next(item for item in student.parameters() if item.numel()).add_(1.0)
    checkpoint = torch.load(rank_path, map_location="cpu", weights_only=False)
    with FSDP.state_dict_type(
        student,
        StateDictType.SHARDED_STATE_DICT,
        model_state_config,
        optim_state_config,
    ):
        load_result = student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(
            FSDP.optim_state_dict_to_load(student, optimizer, checkpoint["optimizer"])
        )
    restored = _parameter_checksum(student, device)
    resume_passed = (
        not load_result.missing_keys
        and not load_result.unexpected_keys
        and bool(torch.allclose(saved, restored, rtol=0.0, atol=1e-6))
    )
    properties = torch.cuda.get_device_properties(device)
    if properties.name != GPU_MODEL:
        raise PreflightError("CUDA reported an unreviewed GPU model")
    row = {
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "checkpoint_sha256": checkpoint_hash,
        "enable_thinking": False,
        "fsdp_save_resume": resume_passed,
        "gradients_finite": gradients_finite,
        "loading_strategy": "low_cpu_mem_student_rank_zero_teacher",
        "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        "model_facing_prompt_sha256": prompt_hash,
        "parameter_update_nonzero": update_nonzero,
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "prompt_protocol": "qwen3_non_thinking_v1",
        "rank": rank,
        "raw_prompt_sha256": raw_hash,
        "soft_teacher_loss": float(loss.detach()),
        "soft_teacher_loss_finite": bool(torch.isfinite(loss)),
        "student_forward_finite": bool(torch.isfinite(student_logits).all()),
        "student_revision": _resolved_commit(student_model, MODEL_REVISION),
        "teacher_forward_finite": bool(torch.isfinite(teacher_logits).all()),
        "teacher_loaded_on_this_rank": rank == 0,
        "teacher_revision": str(teacher_metadata[0]["resolved_teacher_commit"]),
        "tokenizer_fingerprint": str(teacher_metadata[0]["tokenizer_fingerprint"]),
        "tokenizer_revision": MODEL_REVISION,
        "unique_rank_prompt_shard": unique_prompts,
    }
    device_row = {
        "capability": [properties.major, properties.minor],
        "name": properties.name,
        "rank": rank,
        "total_memory": properties.total_memory,
    }
    del checkpoint, teacher_logits, student_logits, teacher, student, optimizer
    torch.cuda.empty_cache()
    return {"device": device_row, "rank": row}


def _read_cgroup_value(path: Path) -> int | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if value == "max":
        return None
    parsed = int(value)
    return None if parsed >= 2**60 else parsed


def _cgroup_memory() -> dict[str, int | bool]:
    unified: Path | None = None
    legacy: Path | None = None
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        relative = Path(fields[2].lstrip("/"))
        controllers = fields[1].split(",") if fields[1] else []
        if fields[0] == "0" and not controllers:
            unified = relative
        if "memory" in controllers:
            legacy = relative
    if unified is not None:
        root = Path("/sys/fs/cgroup") / unified
        names = ("memory.max", "memory.current", "memory.peak")
    elif legacy is not None:
        root = Path("/sys/fs/cgroup/memory") / legacy
        names = (
            "memory.limit_in_bytes",
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
        )
    else:
        raise PreflightError("allocation has no readable memory cgroup")
    limit, current, peak = (_read_cgroup_value(root / name) for name in names)
    if (
        limit != NODE_MEMORY_BYTES
        or current is None
        or current > limit
        or peak is None
        or peak <= 0
        or peak > limit
    ):
        raise PreflightError("memory cgroup is not an allocation-specific 192-GiB limit")
    headroom = limit - peak
    return {
        "current_bytes": current,
        "headroom_bytes": headroom,
        "limit_bytes": limit,
        "minimum_required_headroom_bytes": MINIMUM_HEADROOM_BYTES,
        "observed_peak_bytes": peak,
        "passed": headroom >= MINIMUM_HEADROOM_BYTES,
        "peak_bytes": peak,
        "requested_bytes": NODE_MEMORY_BYTES,
    }


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("/usr/bin/git", *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise PreflightError("Git did not return one immutable commit identity")
    return value


def _require_clean_git() -> None:
    result = subprocess.run(
        ("/usr/bin/git", "status", "--porcelain", "--untracked-files=no"),
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        raise PreflightError("GPU preflight requires a clean tracked source checkout")


def _publish_once(descriptor: int, name: str, payload: object) -> None:
    if not IDENTIFIER.fullmatch(name) and name != COMPLETION_NAME:
        raise PreflightError("output name is not fixed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor = os.open(name, flags, 0o640, dir_fd=descriptor)
    try:
        raw = _published_json_bytes(payload)
        offset = 0
        while offset < len(raw):
            offset += os.write(file_descriptor, raw[offset:])
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    os.fsync(descriptor)


def _completion(
    invocation: Invocation, *, started_at: str, completed_at: str
) -> dict[str, Any]:
    content = {
        "completed_at": completed_at,
        "completion_kind": COMPLETION_KIND,
        "execution": invocation.execution,
        "execution_config_sha256": invocation.input_hashes["execution_config_sha256"],
        "input_hashes": invocation.input_hashes,
        "plan_sha256": invocation.plan_sha256,
        "project": PROJECT,
        "resolved_config_sha256": invocation.input_hashes["resolved_config_sha256"],
        "run_id": invocation.run_id,
        "schema_version": 1,
        "scientific_config_sha256": invocation.input_hashes["scientific_config_sha256"],
        "scientific_validation": {name: True for name in GATE_NAMES},
        "started_at": started_at,
        "task": TASK,
        "unit_id": invocation.unit_id,
        "workflow_id": invocation.workflow_id,
    }
    return {**content, "sha256": _sha256_value(content)}


def _worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--worker-context", required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    args = parser.parse_args(argv)
    if args.local_rank is not None and args.local_rank != int(os.environ["LOCAL_RANK"]):
        raise PreflightError("torchrun local-rank argument differs from its environment")
    context = _strict_json(
        bytes.fromhex(args.worker_context), context="internal worker context"
    )
    if not isinstance(context, dict):
        raise PreflightError("internal worker context is invalid")
    config_path = Path(context["resolved_config_path"])
    output_path = Path(context["output_path"])
    checkpoint_root = Path(context["checkpoint_root"])
    config = _strict_json(config_path.read_bytes(), context="worker resolved config")
    if not isinstance(config, dict):
        raise PreflightError("worker resolved config is invalid")
    row = _rank_training(config, checkpoint_root)
    import torch
    import torch.distributed as dist

    gathered: list[Any] = [None] * GPU_COUNT
    dist.all_gather_object(gathered, row)
    if int(os.environ["RANK"]) == 0:
        rank_rows = [item["rank"] for item in gathered]
        device_rows = [item["device"] for item in gathered]
        cgroup = _cgroup_memory()
        process_peak = sum(int(item["process_max_rss_bytes"]) for item in rank_rows)
        cgroup["observed_peak_bytes"] = max(int(cgroup["peak_bytes"]), process_peak)
        cgroup["headroom_bytes"] = int(cgroup["limit_bytes"]) - int(
            cgroup["observed_peak_bytes"]
        )
        cgroup["passed"] = cgroup["headroom_bytes"] >= MINIMUM_HEADROOM_BYTES
        if not cgroup["passed"]:
            raise PreflightError("allocation does not preserve required host-memory headroom")
        _require_clean_git()
        git_commit = _git("rev-parse", "HEAD")
        prereg_commit = _git("log", "-n", "1", "--format=%H", "--", "prereg/qwen3_v2.yaml")
        report: dict[str, Any] = {
            "artifact_namespace": "qwen3-v2",
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "code_commit": git_commit,
            "cgroup_memory": cgroup,
            "created_at": _utc_now(),
            "devices": device_rows,
            "enable_thinking": False,
            "execution_context": {
                "allocation_visibility": "preserved",
                "distributed_launcher": "environment_rank_passthrough",
                "mode": "server_scheduler_foreground",
                "visible_device_count": GPU_COUNT,
            },
            "git_commit": git_commit,
            "model_revision": MODEL_REVISION,
            "nccl_all_reduce": True,
            "passed": True,
            "phase": "gpu_preflight",
            "prereg_commit": prereg_commit,
            "prereg_path": "prereg/qwen3_v2.yaml",
            "prereg_sha256": context["preregistration_sha256"],
            "prereg_version": "qwen3_v2",
            "prompt_protocol": "qwen3_non_thinking_v1",
            "protocol_track": "qwen3_v2",
            "qwen_forward_finite": all(
                item["student_forward_finite"]
                and item["teacher_forward_finite"]
                and item["soft_teacher_loss_finite"]
                and item["gradients_finite"]
                and item["parameter_update_nonzero"]
                and item["unique_rank_prompt_shard"]
                and item["fsdp_save_resume"]
                for item in rank_rows
            ),
            "rank_prompt_hashes_unique": len(
                {item["model_facing_prompt_sha256"] for item in rank_rows}
            )
            == GPU_COUNT,
            "rank_training_checks": rank_rows,
            "rank_zero_teacher_load_count": sum(
                int(item["teacher_loaded_on_this_rank"]) for item in rank_rows
            ),
            "resolved_config_sha256": context["resolved_config_sha256"],
            "resolved_model_commit": MODEL_REVISION,
            "resolved_teacher_commit": TEACHER_REVISION,
            "teacher_revision": TEACHER_REVISION,
            "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
            "tokenizer_hash": TOKENIZER_FINGERPRINT,
            "tokenizer_revision": MODEL_REVISION,
            "torch_cuda_version": str(torch.version.cuda),
            "torch_version": str(torch.__version__),
            "visible_cuda_devices": torch.cuda.device_count(),
            "world_size": GPU_COUNT,
        }
        if not report["qwen_forward_finite"]:
            raise PreflightError("real Qwen3-v2 forward/backward or resume gate failed")
        report["sha256"] = _sha256_value(report)
        completed_at = _utc_now()
        output_descriptor = os.open(output_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            if os.listdir(output_descriptor):
                raise PreflightError("output attempt is not empty")
            _publish_once(output_descriptor, REPORT_NAME, report)
            _publish_once(
                output_descriptor,
                COMPLETION_NAME,
                _completion(
                    Invocation(
                        workflow_id=context["workflow_id"],
                        plan_sha256=context["plan_sha256"],
                        unit_id=context["unit_id"],
                        run_id=context["run_id"],
                        job_id=context["job_id"],
                        attempt=context["attempt"],
                        execution_profile=PROFILE,
                        manifest_sha256=context["manifest_sha256"],
                        allocation_sha256=context["allocation_sha256"],
                        content_handles=tuple(
                            ContentHandle(name, digest, -1)
                            for name, digest in sorted(context["input_hashes"].items())
                        ),
                        output_descriptor=output_descriptor,
                    ),
                    started_at=context["started_at"],
                    completed_at=completed_at,
                ),
            )
        finally:
            os.close(output_descriptor)
    dist.barrier()
    dist.destroy_process_group()
    return 0


def _script_parent_path() -> str:
    match = PROC_SELF_FD.fullmatch(os.path.abspath(__file__))
    if match is None:
        raise PreflightError("deployment implementation must execute through a held descriptor")
    return f"/proc/{os.getpid()}/fd/{match.group(1)}"


def _supervise(argv: Sequence[str] | None) -> int:
    started_at = _utc_now()
    invocation = _parse_outer(argv)
    _validate_environment()
    payloads, prereg = _read_inputs(invocation)
    resolved = _validate_config(invocation, payloads, prereg)
    temp_root = Path(FIXED_ENVIRONMENT["TMPDIR"])
    temp_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="qwen3-v2-gpu-preflight-", dir=temp_root))
    try:
        config_path = temporary / "resolved-config.json"
        config_path.write_bytes(_canonical_json(resolved).encode("utf-8"))
        checkpoint_root = temporary / "fsdp-resume"
        checkpoint_root.mkdir(mode=0o750)
        context = {
            "allocation_sha256": invocation.allocation_sha256,
            "attempt": invocation.attempt,
            "checkpoint_root": str(checkpoint_root),
            "input_hashes": invocation.input_hashes,
            "job_id": invocation.job_id,
            "manifest_sha256": invocation.manifest_sha256,
            "output_path": f"/proc/{os.getpid()}/fd/{invocation.output_descriptor}",
            "plan_sha256": invocation.plan_sha256,
            "preregistration_sha256": invocation.input_hashes["preregistration_sha256"],
            "resolved_config_path": str(config_path),
            "resolved_config_sha256": invocation.input_hashes["resolved_config_sha256"],
            "run_id": invocation.run_id,
            "started_at": started_at,
            "unit_id": invocation.unit_id,
            "workflow_id": invocation.workflow_id,
        }
        from torch.distributed.run import main as torchrun_main

        torchrun_main(
            [
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=4",
                "--max-restarts=0",
                "--monitor-interval=1",
                _script_parent_path(),
                "--rank-worker",
                "--worker-context",
                _canonical_json(context).encode("utf-8").hex(),
            ]
        )
        return 0
    finally:
        shutil.rmtree(temporary)


def main() -> int:
    try:
        arguments = sys.argv[1:]
        if arguments and arguments[0] == "--rank-worker":
            return _worker_main(arguments[1:])
        return _supervise(arguments)
    except (OSError, PreflightError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(f"OPD Qwen3-v2 GPU preflight failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
