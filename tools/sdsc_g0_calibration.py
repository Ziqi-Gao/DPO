#!/usr/bin/env python3
"""Validate and plan the candidate two-H100 canonical-SFT calibration stage.

The default only validates inputs and prints the original scientific command
plan. --execute-calibration requires a real two-H100 allocation and bound,
completed upstream evidence. It produces training artifacts for independent
review, never G0 success or execution-class certification. It never submits a
job. The historical Blackwell certificate is not H100 authority.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

BASE_SCIENCE_SHA256 = "6c3942f4d6cf87329e1c70ba32b0b8d4cdbd526a3bbcfdcb1afd3e1674662657"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
LOCAL_FILESYSTEMS = {"ext2", "ext3", "ext4", "xfs", "btrfs"}
BLOCKERS = [
    "candidate H100 scope is not an accepted execution-class certification",
    "canonical-SFT calibration does not complete the remaining scientific G0 gates",
    "runtime-derived checkpoint hash is separate from the accepted frozen base config identity",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def helper(name):
    spec = importlib.util.spec_from_file_location(
        "_calibration_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def resolved_interpreter(path):
    actual = Path(sys.executable).resolve()
    require(
        path.is_absolute() and ".." not in path.parts and path.resolve() == actual,
        "planned interpreter differs from validated runtime",
    )
    return actual


def json_document(path, expected_sha256=None):
    raw = helper("sdsc_teacher_prepare").file_bytes(path, maximum=16 * 1024 * 1024)
    if expected_sha256 is not None:
        require(
            SHA256.fullmatch(expected_sha256) and sha(raw) == expected_sha256, "input report SHA mismatch"
        )

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(value, dict), "JSON document must be an object")
    canonical(value)  # Also reject nonfinite numbers before any scientific import.
    return value


def validate_snapshot(snapshot, report, provenance):
    snapshot = snapshot.get("manifest", snapshot)
    records = snapshot.get("files")
    require(snapshot.get("schema") == "quest-sdsc-snapshot-v1", "wrong preflight snapshot schema")
    require(isinstance(records, list) and records, "preflight snapshot file inventory is absent")
    require(
        sha(canonical(records)) == snapshot.get("code_sha256") == report["code_sha256"]
        and snapshot.get("run_id") == report["run_id"],
        "preflight report does not bind its source snapshot",
    )
    require(snapshot.get("git_head") == provenance["git_head"], "preflight science HEAD differs")
    inventory = {record["path"]: record for record in records}
    require(len(inventory) == len(records), "duplicate preflight snapshot paths")
    for record in provenance["scientific_files"]:
        require(inventory.get(record["path"]) == record, "preflight used different scientific file bytes")


def validate_memory(memory):
    limit = 192 * 1024**3
    peak = memory.get("peak_bytes")
    current = memory.get("current_bytes")
    required = max(32 * 1024**3, math.ceil(limit * 0.20))
    require(
        memory.get("passed") is True
        and memory.get("limit_bytes") == limit
        and type(peak) is int
        and type(current) is int
        and 0 < current <= peak <= limit
        and memory.get("headroom_bytes") == limit - peak >= required
        and memory.get("minimum_headroom_bytes") == required,
        "preflight cgroup memory/headroom evidence differs",
    )
    ancestors = memory.get("ancestors", [])
    finite = [entry["limit_bytes"] for entry in ancestors if entry.get("limit_bytes") is not None]
    require(finite and min(finite) == limit, "preflight cgroup ancestor limits are missing/inconsistent")
    selected = [entry for entry in ancestors if entry.get("path") == memory.get("path")]
    require(
        len(selected) == 1
        and all(
            selected[0].get(key) == memory[key] for key in ("limit_bytes", "current_bytes", "peak_bytes")
        ),
        "preflight selected cgroup is not bound to ancestor evidence",
    )


def validate_preflight(report, expected_contract):
    guards = helper("sdsc_training_preflight")
    require(
        report.get("kind") == "sdsc_h100_training_preflight_candidate_v1"
        and report.get("task") == "qwen3-v2-preflight"
        and report.get("passed") is True
        and report.get("exit_code") == 0
        and report.get("world_size") == 2,
        "a passed real two-H100 preflight report is required",
    )
    for key in ("g0_passed", "execution_class_certified", "uses_blackwell_certificate"):
        require(report.get(key) is False, "preflight must not claim G0/Blackwell certification")
    require(re.fullmatch(r"[1-9][0-9]*", str(report.get("job_id", ""))), "preflight job ID is missing")
    require(SHA256.fullmatch(report.get("code_sha256", "")), "preflight code identity is missing")
    ranks = report.get("ranks")
    require(isinstance(ranks, list) and len(ranks) == 2, "both rank reports are required")
    visibility = None
    checkpoint_files = None
    for rank, item in enumerate(ranks):
        require(item.get("passed") is True, "one preflight rank did not pass")
        require(
            all(item.get(key) == report[key] for key in ("kind", "run_id", "job_id", "code_sha256")),
            "preflight rank identity differs",
        )
        require(
            all(
                item.get(key) is False
                for key in ("g0_passed", "execution_class_certified", "uses_blackwell_certificate")
            ),
            "rank report claims unsupported certification",
        )
        allocation = item["allocation"]
        require(
            allocation.get("rank") == allocation.get("local_rank") == rank
            and allocation.get("world_size") == 2
            and allocation.get("threads_per_rank") == 12
            and allocation.get("job_id") == report["job_id"],
            "preflight rank allocation differs",
        )
        visible = allocation.get("cuda_visible_devices", "")
        devices = visible.split(",")
        require(
            len(devices) == len(set(devices)) == 2
            and all(device and device.strip() == device for device in devices)
            and visibility in (None, visible),
            "preflight CUDA visibility differs across ranks",
        )
        visibility = visible
        runtime = item["runtime"]
        require(
            runtime.get("python") == "3.12.13" and runtime.get("packages") == guards.DEPENDENCIES,
            "preflight runtime does not match the pinned training runtime",
        )
        for model, revision in guards.MODELS.items():
            require(item["pinned_cache"][model]["revision"] == revision, "preflight model revision differs")
        gpu = item["gpu"]
        require(
            "H100" in gpu.get("name", "")
            and gpu.get("compute_capability") == [9, 0]
            and gpu.get("total_memory_bytes", 0) >= 75 * 1024**3
            and gpu.get("logical_device") == rank,
            "preflight did not observe the required H100 devices",
        )
        require(
            item.get("nccl_all_reduce") is True
            and item.get("finite_gradients") is True
            and item.get("parameter_update_nonzero") is True
            and item.get("full_state_optimizer_scheduler_rng_restore") is True,
            "preflight collective/update/full-state restore evidence is missing",
        )
        for name in ("local_parameter_sha256_before", "local_parameter_sha256_after"):
            require(SHA256.fullmatch(item.get(name, "")), "preflight parameter digest is missing")
        require(
            item["local_parameter_sha256_before"] != item["local_parameter_sha256_after"],
            "preflight parameter update was zero",
        )
        losses = item.get("losses", [])
        require(
            len(losses) == 8 and all(math.isfinite(value) for value in losses), "invalid preflight losses"
        )
        require(
            item.get("batch_contract") == expected_contract
            and item.get("local_global_slots") == list(range(rank, 64, 2))
            and item.get("reserved_global_nonpadding_tokens") == 64 * 1536,
            "preflight batch/token/window semantics differ",
        )
        fsdp = item["fsdp"]
        require(
            fsdp.get("requested_fsdp_sharding_strategy") == "FULL_SHARD"
            and fsdp.get("effective_fsdp_sharding_strategy") == "FULL_SHARD"
            and fsdp.get("fsdp_wrapper_count") == 29,
            "preflight FSDP wrapper evidence differs",
        )
        require(
            item.get("rank_zero_teacher") == {"finite": True, "revision": guards.MODELS["Qwen/Qwen3-8B"]},
            "pinned teacher forward did not pass",
        )
        checkpoint = item["checkpoint"]
        files = checkpoint.get("files", [])
        require(
            checkpoint.get("world_size") == 2
            and checkpoint.get("all_files_verified_before_restore") is True
            and checkpoint.get("local_restored_parameter_sha256") == item["local_parameter_sha256_after"]
            and SHA256.fullmatch(checkpoint.get("local_restored_optimizer_sha256", ""))
            and [record["path"] for record in files] == guards.checkpoint_names(2)
            and (checkpoint_files is None or checkpoint_files == files),
            "preflight checkpoint inventory/restoration differs",
        )
        for record in files:
            require(
                type(record.get("size")) is int
                and record["size"] > 0
                and SHA256.fullmatch(record.get("sha256", "")),
                "preflight checkpoint file identity is missing",
            )
        checkpoint_files = files
        require(item["node_local_mount"]["fstype"] in LOCAL_FILESYSTEMS, "preflight scratch is not local")
        for key in ("initial_cgroup_memory", "cgroup_memory"):
            validate_memory(item[key])
        allocated, reserved = item["peak_gpu_allocated_bytes"], item["peak_gpu_reserved_bytes"]
        require(0 < allocated <= reserved < gpu["total_memory_bytes"], "preflight GPU memory has no headroom")
    require(ranks[0].get("node") and ranks[0]["node"] == ranks[1].get("node"), "preflight was not one node")
    return {"job_id": report["job_id"], "run_id": report["run_id"], "world_size": 2, "report_validated": True}


def storage_overrides(args):
    locations = {
        "output_root": args.output_dir,
        "task.dataset_family_path": args.output_dir / "dataset",
        "state_source.store_path": args.output_dir / "teacher_demos",
        "production_safety.initial_checkpoint_path": args.output_dir / "initial_checkpoint.pt",
        "production_safety.readiness_report": args.output_dir / "readiness.json",
        "production_safety.probe_cohort_manifest": args.output_dir / "probes/manifest.json",
        "anti_shortcut.report_path": args.output_dir / "anti_shortcut.json",
    }
    return [key + "=" + json.dumps(str(path)) for key, path in locations.items()]


def compose_base(args, api, provenance):
    teacher = helper("sdsc_teacher_prepare")
    resolved = api.resolve(
        code_root=args.science_root,
        configured_path=teacher.PROTOCOL,
        expected_head=provenance["science_git_head"],
    )
    require(tuple(resolved.binding.hydra_override_vector) == teacher.OVERRIDES, "accepted overrides differ")
    overrides = [*teacher.OVERRIDES, f"protocol_amendment_path={teacher.AMENDMENT}", *storage_overrides(args)]
    config = api.compose(overrides, config_root=args.science_root / "configs")
    require(
        api.config_sha(config)
        == resolved.binding.storage_neutral_resolved_config_sha256
        == BASE_SCIENCE_SHA256,
        "candidate base configuration differs from the accepted scientific hash",
    )
    require(
        config["production_safety"]["initial_checkpoint_hash"] == "", "base config already binds a checkpoint"
    )
    binding = api.binding(config)
    require(binding["code_commit"] == provenance["science_git_head"], "calibration formal HEAD differs")
    return config, overrides, binding, resolved


def validate_teacher(report, args, config, binding, resolved):
    teacher = helper("sdsc_teacher_prepare")
    require(
        report.get("kind") == "sdsc_teacher_prepare_v1"
        and report.get("task") == "qwen3-v2-teacher-prepare"
        and report.get("passed") is True
        and report.get("exit_code") == 0
        and report.get("science_git_head") == binding["code_commit"]
        and report.get("formal_binding") == binding,
        "teacher report is failed or bound to different science",
    )
    require(
        report.get("g0_passed") is False
        and report.get("execution_class_certified") is False
        and report.get("uses_blackwell_certificate_for_execution") is False,
        "teacher report claims unsupported execution authority",
    )
    require(re.fullmatch(r"[1-9][0-9]*", str(report.get("job_id", ""))), "teacher job ID missing")
    reviewed = report["accepted_science"]
    require(
        reviewed["sha256"] == resolved.sha256
        and reviewed["acceptance_commit"] == resolved.acceptance_commit
        and reviewed["reviewed_implementation_commit"] == resolved.reviewed_implementation_commit
        and reviewed["storage_neutral_resolved_config_sha256"] == BASE_SCIENCE_SHA256,
        "teacher accepted science protocol differs",
    )
    family_api = importlib.import_module("posttrain_circuits.datasets.proofgraph.family")
    store_api = importlib.import_module("posttrain_circuits.datasets.teacher_demos.store")
    train = importlib.import_module("posttrain_circuits.cli.train")
    family = family_api.load_dataset_family(args.teacher_root / "dataset")
    sizes = config["task"]["split_sizes"]
    require(sum(sizes.values()) == 144000, "full frozen dataset population is required")
    require(
        all(len(family.examples(split)) == count for split, count in sizes.items()),
        "dataset split sizes differ",
    )
    dataset = {
        "dataset_family_sha256": family.manifest["sha256"],
        "train_examples_file_sha256": family.boundary("train")["examples_file_sha256"],
        "ordered_prompt_ids": [example.example_id for example in family.examples("train")[:256]],
        "split_sizes": sizes,
    }
    require(report["dataset"] == dataset, "teacher report does not match the staged dataset family")
    prompt = report["prompt_preflight"]
    require(
        prompt["tokenizer_fingerprint"] == binding["tokenizer_fingerprint"]
        and prompt["chat_template_sha256"] == binding["chat_template_sha256"]
        and 0 < prompt["minimum_prompt_tokens"] <= prompt["maximum_prompt_tokens"] <= 1246,
        "teacher tokenizer/prompt envelope differs",
    )
    store_evidence = teacher.validate_store(
        SimpleNamespace(output_dir=args.teacher_root), binding, dataset, prompt
    )
    require(store_evidence == report["teacher_store"], "teacher store bytes differ from the passed report")
    accepted, manifest = store_api.read_teacher_demo_store(
        args.teacher_root / "teacher_demos", require_formal=True
    )
    train._require_qwen3_store_binding(manifest, config=config, expected_behavior_policy="Qwen/Qwen3-8B")
    prompt_ids, _ = train._teacher_demo_prompts(accepted)
    train._validate_allocation_neutral_prompt_population(
        prompt_ids,
        global_batch_size=64,
        expected_prompt_ids=dataset["ordered_prompt_ids"],
        expected_prompt_count=256,
    )
    lengths = train._validate_teacher_demo_model_input_lengths(
        accepted,
        manifest,
        expected_prompt_count=256,
        candidates_per_prompt=8,
        max_prompt_tokens=1246,
        max_new_tokens=256,
        max_model_input_length=1536,
    )
    return {"dataset": dataset, "teacher_store": store_evidence, "model_input_envelope": lengths}


def build_plan(args, base_overrides, *, initial_checkpoint_sha256=None):
    """Pure argv construction; a missing runtime hash yields no train command."""
    require(Path(args.python).is_absolute(), "planned Python executable must be absolute")
    export = [
        str(args.python),
        "-B",
        "-m",
        "posttrain_circuits.cli.export_initial_checkpoint",
        *base_overrides,
        "--output",
        str(args.output_dir / "initial_checkpoint.pt"),
        "--confirm-production",
    ]
    overrides = None
    train = None
    if initial_checkpoint_sha256 is not None:
        require(SHA256.fullmatch(initial_checkpoint_sha256), "invalid actual initial checkpoint SHA")
        overrides = [
            *base_overrides,
            "production_safety.initial_checkpoint_hash=" + initial_checkpoint_sha256,
        ]
        train = [
            str(args.python),
            "-B",
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(args.science_root / "configs/accelerate/fsdp_server_scheduler.yaml"),
            "--num_processes",
            "2",
            "--num_cpu_threads_per_process",
            "12",
            "--main_process_port",
            "0",
            "-m",
            "posttrain_circuits.cli.train",
            *overrides,
            "--output",
            str(args.output_dir / "canonical_sft"),
            "--confirm-production",
        ]
    return {
        "execution_enabled": False,
        "cwd": str(args.science_root),
        "environment_updates": {
            "PYTHONPATH": str(args.science_root / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HOME": str(args.hf_home),
            "HF_HUB_CACHE": str(args.hf_home / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "12",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        },
        "preserve_environment_exactly": ["CUDA_VISIBLE_DEVICES"],
        "remove_environment": ["PYTHONHOME"],
        "base_science_config_sha256": BASE_SCIENCE_SHA256,
        "runtime_initial_checkpoint_sha256": initial_checkpoint_sha256,
        "initial_checkpoint_hash_is_inherited_science_acceptance": False,
        "export_argv": export,
        "train_argv": train,
        "training_overrides": overrides,
        "post_training_validator": (
            "posttrain_circuits.cli.factorial_run_validation.validate_factorial_run_artifacts"
        ),
        "post_training_expected_world_size": 2,
        "post_training_expected_resume_ancestry": [],
        "scope": "canonical_sft_calibration_artifacts_only",
        "g0_passed": False,
        "factorial_ready": False,
        "pilot_passed": False,
        "execution_class_certified": False,
        "old_finalizer_compatibility_claimed": False,
        "blockers": [
            *BLOCKERS,
            *([] if initial_checkpoint_sha256 else ["initial checkpoint has not been exported and hashed"]),
        ],
    }


def validate_prerequisites(args, reports):
    require(
        args.prerequisites is not None and args.prerequisites_sha256 is not None,
        "execution requires control-plane prerequisites and their external SHA",
    )
    document = json_document(args.prerequisites, args.prerequisites_sha256)
    require(document.get("schema") == "quest-sdsc-calibration-prerequisites-v1", "wrong prerequisite schema")
    target = document["target"]
    require(
        target.get("run_id") == args.run_id
        and target.get("code_sha256") == args.code_sha256
        and re.fullmatch(r"[0-9a-f]{32}", target.get("intent_id", "")),
        "prerequisites do not bind this immutable invocation",
    )
    for name, report in reports.items():
        upstream = document["upstream"][name]
        expected_sha = getattr(args, name + "_report_sha256")
        require(
            upstream.get("report_sha256") == expected_sha and upstream.get("report") == report,
            "prerequisite report bytes/identity differ",
        )
        receipt, publication, status = (
            upstream["receipt"],
            upstream["publication_receipt"],
            upstream["status"],
        )
        for value in (receipt, publication):
            require(
                all(str(value.get(key)) == str(report[key]) for key in ("job_id", "run_id", "code_sha256")),
                "upstream receipt identity differs",
            )
        require(
            publication.get("passed") is True
            and publication.get("persisted") is True
            and publication.get("persistent_read_back_verified") is True
            and SHA256.fullmatch(upstream.get("publication_receipt_sha256", "")),
            "upstream durable publication is unverified",
        )
        filename = "teacher-prepare.json" if name == "teacher" else "preflight.json"
        records = [item for item in publication.get("files", []) if item.get("path") == filename]
        require(
            len(records) == 1 and records[0].get("sha256") == expected_sha,
            "publication does not bind the upstream report",
        )
        require(
            status.get("success") is True
            and status.get("state") == "COMPLETED"
            and status.get("job_id") == report["job_id"]
            and status.get("run_id") == report["run_id"]
            and status.get("result", {}).get("verified") is True
            and status.get("result", {}).get("result_sha256") == expected_sha,
            "upstream semantic completion status is unverified",
        )
        queue, accounting = status["queue"], status["accounting"]
        require(
            queue.get("returncode") == 0
            and not queue.get("stdout", "").strip()
            and accounting.get("returncode") == 0,
            "upstream queue/accounting query was inconclusive",
        )
        rows = [line.strip().split("|") for line in accounting.get("stdout", "").splitlines() if line.strip()]
        job = str(report["job_id"])
        exact = [row for row in rows if row[0] == job]
        related = [row for row in rows if row[0] == job or row[0].startswith(job + ".")]
        require(
            len(exact) == 1 and all(len(row) >= 3 and row[1:3] == ["COMPLETED", "0:0"] for row in related),
            "upstream Slurm job/step did not complete successfully",
        )
    return {"sha256": args.prerequisites_sha256, "intent_id": target["intent_id"], "upstream_verified": True}


def allocation_identity():
    require(
        re.fullmatch(r"[1-9][0-9]*", os.environ.get("SLURM_JOB_ID", "")), "real Slurm allocation required"
    )
    require(
        os.environ.get("SLURM_CPUS_PER_TASK") == "24" and os.environ.get("SLURM_MEM_PER_NODE") == "196608",
        "calibration requires 24 CPUs and 192 GiB host memory",
    )
    require(
        os.environ.get("WORLD_SIZE", "1") == "1" and os.environ.get("LOCAL_RANK", "0") == "0",
        "calibration wrapper must launch once, outside torchrun",
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    require(
        len(devices) == len(set(devices)) == 2 and all(value and value.strip() == value for value in devices),
        "exactly two scheduler-assigned visible GPUs are required",
    )
    return {
        "job_id": os.environ["SLURM_JOB_ID"],
        "cpus": 24,
        "memory_mib": 196608,
        "world_size": 2,
        "cuda_visible_devices": visible,
    }


def gpu_identity():
    import torch

    require(torch.cuda.is_available() and torch.cuda.device_count() == 2, "two visible CUDA GPUs required")
    require(
        torch.version.cuda == "12.8" and tuple(torch.cuda.nccl.version()) == (2, 27, 3),
        "CUDA/NCCL runtime differs from reviewed pins",
    )
    devices = []
    for rank in range(2):
        properties = torch.cuda.get_device_properties(rank)
        require(
            "H100" in properties.name
            and (properties.major, properties.minor) == (9, 0)
            and properties.total_memory >= 75 * 1024**3,
            "two H100 80GB devices are required",
        )
        devices.append(
            {
                "logical_device": rank,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [9, 0],
            }
        )
    return devices


def stage_inputs(source, destination):
    """Copy only validated teacher inputs; never link mutable outputs to inputs."""
    safe = helper("sdsc_teacher_prepare").real_path
    require(
        source != destination and source not in destination.parents and destination not in source.parents,
        "teacher input and output trees must be disjoint",
    )
    total = 0
    for name in ("dataset", "teacher_demos"):
        source_root = safe(source / name)
        require(source_root.is_dir(), "staged teacher input directory is missing")
        for directory, directories, filenames in os.walk(source_root, followlinks=False):
            for child in directories:
                require(not (Path(directory) / child).is_symlink(), "teacher input directory is a symlink")
            target_root = safe(destination / name / Path(directory).relative_to(source_root))
            target_root.mkdir(parents=True, exist_ok=True)
            for filename in filenames:
                original = safe(Path(directory) / filename)
                require(stat.S_ISREG(original.stat().st_mode), "teacher input is not a regular file")
                total += original.stat().st_size
                require(total <= 8 * 1024**3, "teacher input copy exceeds the eight-GiB budget")
                target = safe(target_root / filename)
                with original.open("rb") as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                target.chmod(0o444)
    return total


def child_environment(plan):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    for name in plan["remove_environment"]:
        environment.pop(name, None)
    environment.update(plan["environment_updates"])
    return environment


def run_command(argv, *, cwd, environment, log):
    """Await a foreground child; inherit the outer wrapper's bounded-kill group."""
    print(json.dumps({"phase": "scientific_command_started", "log": str(log), "argv": argv}), flush=True)
    with log.open("xb") as stream:
        process = subprocess.Popen(
            argv, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=False
        )
        previous = {}

        def interrupted(number, _frame):
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(number)
            raise InterruptedError("calibration child interrupted by signal " + str(number))

        try:
            for number in (signal.SIGTERM, signal.SIGINT):
                previous[number] = signal.signal(number, interrupted)
            require(process.wait() == 0, "scientific command failed; inspect " + str(log))
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    process.wait(timeout=10)
    print(json.dumps({"phase": "scientific_command_completed", "log": str(log)}), flush=True)


def initial_checkpoint_identity(path, config):
    import torch

    hashing = importlib.import_module("posttrain_circuits.artifacts.hashing")
    safe = helper("sdsc_teacher_prepare").real_path
    safe(path)
    require(stat.S_ISREG(path.stat().st_mode) and path.stat().st_size > 0, "initial checkpoint is absent")
    before = path.stat()
    digest = hashing.sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {
        "format": "initial_hf_full_state_v1",
        "model_id": config["model"]["model_name_or_path"],
        "model_revision": config["model"]["model_revision"],
        "resolved_model_commit": config["model"]["model_revision"],
        "tokenizer_id": config["model"]["tokenizer_name_or_path"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "resolved_tokenizer_commit": config["model"]["tokenizer_revision"],
        "tokenizer_hash": config["model"]["tokenizer_fingerprint"],
    }
    require(
        isinstance(payload, dict)
        and all(payload.get(key) == value for key, value in expected.items())
        and isinstance(payload.get("model"), dict)
        and payload["model"],
        "exported initial checkpoint model/tokenizer identity differs",
    )
    after = path.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "initial checkpoint changed while binding",
    )
    return {"path": str(path), "sha256": digest, "size": after.st_size, **expected}


def execute_calibration(args, config, overrides, binding, reviewed, api, report, teacher_report):
    teacher, guards = helper("sdsc_teacher_prepare"), helper("sdsc_training_preflight")
    require(not args.output_dir.exists(), "calibration output was already used; implicit resume is forbidden")
    args.output_dir.mkdir(parents=True)
    report.update(
        passed=False,
        exit_code=1,
        training_executed=False,
        resumable=False,
        persistent_storage_publication="caller_required",
        started_at_unix=time.time(),
    )
    try:
        report["allocation"] = allocation_identity()
        report["job_id"] = report["allocation"]["job_id"]
        report["initial_cgroup_memory"] = guards.memory_envelope()
        report["gpu"] = gpu_identity()
        teacher.publish(args.output_dir / "g0-calibration-start.json", report)
        report["input_bytes_copied"] = stage_inputs(args.teacher_root, args.output_dir)
        staged = SimpleNamespace(**{**vars(args), "teacher_root": args.output_dir})
        validate_teacher(teacher_report, staged, config, binding, reviewed)
        require(api.binding(config) == binding, "formal binding changed before export")
        plan = build_plan(args, overrides)
        environment = child_environment(plan)
        run_command(
            plan["export_argv"],
            cwd=args.science_root,
            environment=environment,
            log=args.output_dir / "export.log",
        )
        initial = initial_checkpoint_identity(args.output_dir / "initial_checkpoint.pt", config)
        report["initial_checkpoint"] = initial
        plan = build_plan(args, overrides, initial_checkpoint_sha256=initial["sha256"])
        runtime_config = api.compose(plan["training_overrides"], config_root=args.science_root / "configs")
        restored_base = copy.deepcopy(runtime_config)
        restored_base["production_safety"]["initial_checkpoint_hash"] = ""
        require(restored_base == config, "runtime checkpoint binding changed other scientific configuration")
        require(api.binding(runtime_config) == binding, "runtime formal model/amendment identity changed")
        report["runtime_science_config_sha256"] = api.config_sha(runtime_config)
        report["runtime_config_sha256"] = sha(canonical(runtime_config))
        report["plan"] = {**plan, "execution_enabled": True}
        teacher.publish(args.output_dir / "g0-calibration-plan.json", report["plan"])
        report["training_executed"] = True
        run_command(
            plan["train_argv"],
            cwd=args.science_root,
            environment=environment,
            log=args.output_dir / "train.log",
        )
        require(
            initial_checkpoint_identity(args.output_dir / "initial_checkpoint.pt", config) == initial,
            "initial checkpoint changed during training",
        )
        validator = importlib.import_module("posttrain_circuits.cli.factorial_run_validation")
        artifacts = validator.validate_factorial_run_artifacts(
            args.output_dir / "canonical_sft",
            expected_world_size=2,
            expected_resolved_config=runtime_config,
            expected_code_commit=binding["code_commit"],
            expected_resume_ancestry=[],
        )
        report["training_artifacts"] = artifacts.content_binding()
        report["training_artifact_root"] = str(args.output_dir / "canonical_sft")
        report["final_cgroup_memory"] = guards.memory_envelope()
        require(
            teacher.validate_checkout(args)["science_git_head"] == binding["code_commit"],
            "science provenance changed during execution",
        )
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == report["allocation"]["cuda_visible_devices"],
            "assigned CUDA visibility changed during execution",
        )
        report.update(passed=True, exit_code=0, elapsed_seconds=time.time() - report["started_at_unix"])
    except BaseException as error:
        report.update(
            error=f"{type(error).__name__}: {error}", elapsed_seconds=time.time() - report["started_at_unix"]
        )
        with contextlib.suppress(OSError):
            teacher.publish(args.output_dir / "g0-calibration.json", report)
        raise
    teacher.publish(args.output_dir / "g0-calibration.json", report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "science-root",
        "work-dir",
        "output-dir",
        "hf-home",
        "provenance-manifest",
        "teacher-root",
        "teacher-report",
        "preflight-report",
        "preflight-snapshot-manifest",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    for name in (
        "provenance-manifest-sha256",
        "teacher-report-sha256",
        "preflight-report-sha256",
        "code-sha256",
        "run-id",
    ):
        result.add_argument("--" + name, required=True)
    result.add_argument("--python", type=Path, default=Path(sys.executable).resolve())
    result.add_argument("--validate-only", action="store_true", help="default: validation and plan only")
    result.add_argument(
        "--execute-calibration", action="store_true", help="execute calibration evidence only"
    )
    result.add_argument("--prerequisites", type=Path)
    result.add_argument("--prerequisites-sha256")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(
        not (args.execute_calibration and args.validate_only), "choose validation or calibration execution"
    )
    if args.execute_calibration:
        allocation_identity()  # Refuse local/login execution before reading/importing any science.
    teacher = helper("sdsc_teacher_prepare")
    guards = helper("sdsc_training_preflight")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id), "invalid run ID")
    # Conda commonly supplies bin/python as a symlink to bin/python3.12. Bind
    # the real already-running executable, rather than rejecting that alias.
    args.python = resolved_interpreter(args.python)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            teacher.real_path(value)
        if name.endswith("sha256") and value is not None:
            require(SHA256.fullmatch(value), "invalid SHA-256")
    require(
        args.work_dir in args.teacher_root.parents
        and args.work_dir in args.hf_home.parents
        and args.science_root not in args.output_dir.parents
        and args.output_dir != args.science_root
        and not args.output_dir.exists()
        and "qwen3-v2" in args.output_dir.parts
        and "qwen3-v2" in args.teacher_root.parts,
        "fresh output and staged teacher/cache paths must be confined outside science in qwen3-v2 workspace",
    )
    for original in (args.teacher_root, args.hf_home):
        require(
            original != args.output_dir
            and original not in args.output_dir.parents
            and args.output_dir not in original.parents,
            "output must not overlap staged inputs/cache",
        )
    mount = guards.local_paths(args.work_dir, args.output_dir)
    runtime = guards.runtime_identity()
    require(
        args.python == Path(sys.executable).resolve(), "planned interpreter differs from validated runtime"
    )
    cache = guards.pinned_cache(args.hf_home)
    provenance = teacher.validate_checkout(args)
    raw_provenance = json_document(args.provenance_manifest, args.provenance_manifest_sha256)
    teacher_report = json_document(args.teacher_report, args.teacher_report_sha256)
    preflight = json_document(args.preflight_report, args.preflight_report_sha256)
    prerequisite_evidence = None
    if args.execute_calibration or args.prerequisites is not None:
        prerequisite_evidence = validate_prerequisites(
            args, {"teacher": teacher_report, "preflight": preflight}
        )
    validate_snapshot(json_document(args.preflight_snapshot_manifest), preflight, raw_provenance)
    previous = Path.cwd()
    try:
        os.chdir(args.science_root)
        api = teacher.scientific_api(args.science_root)
        config, overrides, binding, reviewed = compose_base(args, api, provenance)
        kernel = importlib.import_module("posttrain_circuits.learning.training.execution_safety_kernel")
        preflight_evidence = validate_preflight(preflight, kernel.batch_token_contract(2, 256))
        teacher_evidence = validate_teacher(teacher_report, args, config, binding, reviewed)
        require(teacher.validate_checkout(args) == provenance, "science checkout changed during validation")
        report = {
            "kind": "sdsc_g0_calibration_candidate_v1",
            "task": "qwen3-v2-g0-calibration",
            "validation_only": not args.execute_calibration,
            "inputs_validated": True,
            "training_executed": False,
            "run_id": args.run_id,
            "code_sha256": args.code_sha256,
            **provenance,
            "formal_binding": binding,
            "teacher_report_sha256": args.teacher_report_sha256,
            "preflight_report_sha256": args.preflight_report_sha256,
            "prerequisites": prerequisite_evidence,
            "prerequisites_sha256": args.prerequisites_sha256,
            "runtime": runtime,
            "node_local_mount": mount,
            "pinned_cache": cache,
            "teacher": teacher_evidence,
            "preflight": preflight_evidence,
            "plan": build_plan(args, overrides),
            "g0_passed": False,
            "factorial_ready": False,
            "pilot_passed": False,
            "execution_class_certified": False,
            "uses_blackwell_certificate_for_execution": False,
        }
        if args.execute_calibration:
            report = execute_calibration(
                args, config, overrides, binding, reviewed, api, report, teacher_report
            )
    finally:
        os.chdir(previous)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
