#!/usr/bin/env python3
"""Bounded canonical-SFT calibration consuming the independently accepted teacher.

Launch only after a real matching two-H100 preflight. All outputs remain in a
fresh node-local directory for caller-managed durable publication. The original
training CLI, exact token budget and strict artifact validator remain in use.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

TASK = "qwen3-v2-adapted-calibration"
KIND = "sdsc_adapted_canonical_sft_calibration_v1"
RESULT = "adapted-calibration.json"
PROTOCOL = "prereg/amendments/qwen3_adapted_student_calibration_v5.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ACCEPTANCE_INVENTORY_SHA256 = "8d53783b9fb1d386de5a0a291c2b28e225347168cc7cfed4c0c0aceaf01d9bed"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def helper(name):
    spec = importlib.util.spec_from_file_location(
        "_adapted_calibration_" + name, Path(__file__).with_name(name + ".py")
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
            str(args.science_root / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml"),
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
            "OPENBLAS_NUM_THREADS": "12",
            "NUMEXPR_NUM_THREADS": "12",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        },
        "preserve_environment_exactly": ["CUDA_VISIBLE_DEVICES"],
        "remove_environment": ["PYTHONHOME"],
        "student_protocol_sha256": args.student_protocol_sha256,
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
        "scope": "accepted_dense_teacher_canonical_sft_calibration_artifacts_only",
        "g0_passed": False,
        "factorial_ready": False,
        "pilot_passed": False,
        "execution_class_certified": False,
        "old_finalizer_compatibility_claimed": False,
        "blockers": [
            "calibration artifacts are not G0 or pilot acceptance",
            *([] if initial_checkpoint_sha256 else ["initial checkpoint has not been exported and hashed"]),
        ],
    }


def storage_overrides(args):
    locations = {
        "output_root": args.output_dir,
        "task.dataset_family_path": args.dataset_root,
        "state_source.store_path": args.output_dir / "teacher_demos",
        "production_safety.initial_checkpoint_path": args.output_dir / "initial_checkpoint.pt",
        "production_safety.readiness_report": args.output_dir / "readiness.json",
        "production_safety.probe_cohort_manifest": args.output_dir / "probes/manifest.json",
        "anti_shortcut.report_path": args.output_dir / "anti_shortcut.json",
    }
    return [key + "=" + json.dumps(str(path)) for key, path in locations.items()]


def scientific_api(root):
    helper("sdsc_adapted_training_preflight").activate_scientific_source(root)
    config = importlib.import_module("posttrain_circuits.core.config")
    protocols = importlib.import_module("posttrain_circuits.artifacts.adapted_student_protocol")
    runs = importlib.import_module("posttrain_circuits.artifacts.runs")
    inputs = importlib.import_module("posttrain_circuits.artifacts.adapted_teacher_sft")
    family = importlib.import_module("posttrain_circuits.datasets.proofgraph.family")
    return SimpleNamespace(
        compose=config.compose_config,
        resolve=protocols.resolve_adapted_student_protocol,
        validate=protocols.validate_student_protocol,
        binding=runs.formal_artifact_binding,
        load_teacher=inputs.read_accepted_teacher_sft,
        load_family=family.load_dataset_family,
    )


def compose_base(args, api):
    reviewed = api.resolve(args.science_root, expected_head=args.science_git_head)
    require(
        reviewed.head == args.science_git_head and reviewed.protocol_sha256 == args.student_protocol_sha256,
        "accepted student protocol does not match the actual scientific source",
    )
    overrides = [
        *helper("sdsc_teacher_prepare").OVERRIDES,
        "adapted_teacher=qwen3_accepted_student_v5",
        "protocol_amendment_path=" + PROTOCOL,
        *storage_overrides(args),
    ]
    config = api.compose(overrides, config_root=args.science_root / "configs")
    require(
        api.validate(config) == reviewed,
        "composed student configuration differs from its accepted review",
    )
    require(config["production_safety"]["initial_checkpoint_hash"] == "", "base config already binds weights")
    binding = api.binding(config)
    require(binding["code_commit"] == args.science_git_head, "formal scientific HEAD differs")
    return config, overrides, binding, reviewed


def teacher_inputs(args, config, api):
    """Use the science adapter's raw replay; never manufacture a legacy store."""
    accepted, manifest = api.load_teacher(args.teacher_input_root, config=config)
    family = api.load_family(args.dataset_root)
    sizes = config["task"]["split_sizes"]
    require(sum(sizes.values()) == 144000, "the complete original dataset family is required")
    require(
        all(len(family.examples(split)) == count for split, count in sizes.items()),
        "dataset family split sizes differ",
    )
    expected_ids = [example.example_id for example in family.examples("train")[:256]]
    require(len(expected_ids) == len(set(expected_ids)) == 256, "training prompt population differs")
    observed = list(dict.fromkeys(attempt.prompt_id for attempt in accepted))
    require(
        len(accepted) == 2048 and observed == expected_ids, "accepted teacher population or order differs"
    )
    return {
        "dataset_family_sha256": family.manifest["sha256"],
        "ordered_prompt_ids": expected_ids,
        "accepted_candidates": len(accepted),
        "covered_prompts": len(observed),
        "derived_manifest_sha256": manifest["sha256"],
        "teacher_identity": config["adapted_teacher"]["teacher_identity"],
        "source_originals_preserved": True,
    }


def validate_preflight_upstream(upstream, expected):
    """A queue disappearance or successful child exit alone never passes this gate."""
    report = upstream["report"]
    report_sha = upstream.get("report_sha256")
    require(SHA256.fullmatch(str(report_sha)), "preflight physical report pin is absent")
    helper("sdsc_adapted_training_preflight").validate_completed_report(report, expected)
    receipt, publication, status = (upstream["receipt"], upstream["publication_receipt"], upstream["status"])
    for value in (receipt, publication):
        require(
            all(str(value.get(key)) == str(report[key]) for key in ("job_id", "run_id", "code_sha256")),
            "preflight upstream invocation identity differs",
        )
    require(
        receipt.get("task") == report["task"]
        and publication.get("task") == report["task"]
        and publication.get("passed") is True
        and publication.get("persisted") is True
        and publication.get("persistent_read_back_verified") is True
        and SHA256.fullmatch(str(upstream.get("publication_receipt_sha256"))),
        "preflight durable publication is unverified",
    )
    records = [row for row in publication.get("files", []) if row.get("path") == "adapted-preflight.json"]
    require(
        len(records) == 1 and records[0].get("sha256") == report_sha, "preflight report publication differs"
    )
    require(
        all(str(status.get(key)) == str(report[key]) for key in ("job_id", "run_id", "task"))
        and status.get("success") is True
        and status.get("state") == "COMPLETED"
        and status.get("result", {}).get("verified") is True
        and status["result"].get("result_sha256") == report_sha,
        "preflight semantic completion was not verified",
    )
    queue, accounting = status["queue"], status["accounting"]
    require(
        queue.get("returncode") == 0
        and not queue.get("stdout", "").strip()
        and accounting.get("returncode") == 0,
        "preflight queue/accounting is inconclusive",
    )
    rows = [line.strip().split("|") for line in accounting.get("stdout", "").splitlines() if line.strip()]
    job = str(report["job_id"])
    related = [row for row in rows if row[0] == job or row[0].startswith(job + ".")]
    require(
        len([row for row in related if row[0] == job]) == 1
        and all(len(row) >= 3 and row[1:3] == ["COMPLETED", "0:0"] for row in related),
        "preflight Slurm job or step failed",
    )
    return {"job_id": job, "report_sha256": report_sha, "verified": True}


def validate_prerequisites(args, acceptance):
    document = json_document(args.prerequisites, args.prerequisites_sha256)
    require(
        document.get("schema") == "quest-sdsc-adapted-student-prerequisites-v1"
        and document.get("task") == TASK,
        "wrong adapted student prerequisite schema/task",
    )
    target = document["target"]
    require(
        target.get("run_id") == args.run_id
        and target.get("code_sha256") == args.code_sha256
        and re.fullmatch(r"[0-9a-f]{32}", target.get("intent_id", "")),
        "prerequisites do not bind this immutable invocation",
    )
    require(
        document.get("acceptance") == acceptance,
        "prerequisite accepted teacher differs from staged publication",
    )
    protocol = document["protocol"]
    require(
        protocol.get("head") == args.science_git_head
        and protocol.get("protocol_sha256") == args.student_protocol_sha256,
        "prerequisite student protocol differs",
    )
    expected = {
        "science_git_head": args.science_git_head,
        "student_protocol_sha256": args.student_protocol_sha256,
        "teacher_identity": acceptance["teacher_identity"],
        "accepted_teacher_sha256": acceptance["accepted_teacher_sha256"],
        "teacher_acceptance_inventory_sha256": acceptance["inventory_sha256"],
    }
    if "protocol_path" in protocol:
        expected["student_protocol_path"] = protocol["protocol_path"]
    upstream = document["preflight"]
    require(isinstance(upstream, dict), "real matched adapted preflight is required before calibration")
    require(upstream["report"]["run_id"] != args.run_id, "calibration requires a fresh release")
    return document, {
        "sha256": args.prerequisites_sha256,
        "intent_id": target["intent_id"],
        "preflight": validate_preflight_upstream(upstream, expected),
    }


def validate_protocol_proof(proof, reviewed):
    expected = {
        "head": reviewed.head,
        "protocol_sha256": reviewed.protocol_sha256,
        "artifact_sha256": reviewed.sha256,
        "implementation_commit": reviewed.reviewed_implementation_commit,
        "acceptance_commit": reviewed.git_commit,
        "science_file_sha256": reviewed.science_file_sha256,
    }
    require(
        all(proof.get(key) == value for key, value in expected.items()),
        "prerequisite protocol origin differs",
    )


def stage_teacher_inputs(source, destination):
    """Preserve the original nine files byte for byte within the published attempt."""
    safe = helper("sdsc_teacher_prepare").real_path
    safe(source)
    safe(destination)
    require(
        source != destination and source not in destination.parents and destination not in source.parents,
        "teacher source and result paths overlap",
    )
    require(source.is_dir() and not destination.exists(), "teacher result copy must be fresh")
    destination.mkdir(parents=True)
    count, total = 0, 0
    for folder, directories, filenames in os.walk(source, followlinks=False):
        for name in directories:
            require(not (Path(folder) / name).is_symlink(), "teacher source contains a directory symlink")
        for name in filenames:
            original = safe(Path(folder) / name)
            require(stat.S_ISREG(original.lstat().st_mode), "teacher source file is not regular")
            before = original.stat()
            total += before.st_size
            require(total <= 32 * 1024**2, "teacher source exceeds small-result staging budget")
            target = safe(destination / original.relative_to(source))
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            descriptor = os.open(original, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as incoming, target.open("xb") as outgoing:
                opened = os.fstat(incoming.fileno())
                require(
                    stat.S_ISREG(opened.st_mode)
                    and (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
                    == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                    "teacher source changed before copy",
                )
                for chunk in iter(lambda: incoming.read(1024**2), b""):
                    digest.update(chunk)
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = original.stat()
            require(
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                and target.stat().st_size == before.st_size
                and hashlib.sha256(target.read_bytes()).hexdigest() == digest.hexdigest(),
                "teacher input changed during copy or failed read-back",
            )
            target.chmod(0o444)
            count += 1
    require(count == 9, "teacher source bundle does not contain exactly nine files")
    return {"files": count, "bytes": total, "read_back_verified": True}


def execute_calibration(args, config, overrides, binding, reviewed, api, report):
    require(not args.output_dir.exists(), "calibration output already used; implicit resume is forbidden")
    args.output_dir.mkdir(parents=True)
    publish = helper("sdsc_training_preflight").publish_json
    guards = helper("sdsc_training_preflight")
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
        publish(args.output_dir / "adapted-calibration-start.json", report)
        report["teacher_input_staging"] = stage_teacher_inputs(
            args.teacher_input_root, args.output_dir / "teacher_demos"
        )
        staged = SimpleNamespace(**{**vars(args), "teacher_input_root": args.output_dir / "teacher_demos"})
        require(
            teacher_inputs(staged, config, api) == report["teacher_inputs"], "copied teacher source changed"
        )
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
        restored = copy.deepcopy(runtime_config)
        restored["production_safety"]["initial_checkpoint_hash"] = ""
        require(restored == config, "runtime initial checkpoint hash changed other scientific configuration")
        require(api.binding(runtime_config) == binding, "runtime formal model/teacher identity changed")
        require(api.validate(runtime_config) == reviewed, "runtime student protocol changed")
        report["runtime_config_sha256"] = sha(canonical(runtime_config))
        report["plan"] = {**plan, "execution_enabled": True}
        publish(args.output_dir / "adapted-calibration-plan.json", report["plan"])
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
            api.resolve(args.science_root, expected_head=args.science_git_head) == reviewed,
            "reviewed scientific source changed during execution",
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
            publish(args.output_dir / RESULT, report)
        raise
    publish(args.output_dir / RESULT, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "science-root",
        "work-dir",
        "output-dir",
        "hf-home",
        "teacher-input-root",
        "dataset-root",
        "prerequisites",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    for name in (
        "science-git-head",
        "run-id",
        "code-sha256",
        "prerequisites-sha256",
        "student-protocol-sha256",
    ):
        result.add_argument("--" + name, required=True)
    result.add_argument("--python", type=Path, default=Path(sys.executable).resolve())
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--execute-calibration", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(not (args.execute_calibration and args.validate_only), "choose validation or execution")
    if args.execute_calibration:
        allocation_identity()
    guard = helper("sdsc_training_preflight")
    teacher = helper("sdsc_teacher_prepare")
    args.python = resolved_interpreter(args.python)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            teacher.real_path(value)
        if name.endswith("sha256"):
            require(SHA256.fullmatch(value), "invalid SHA-256")
    require(re.fullmatch(r"[a-f0-9]{40}", args.science_git_head), "invalid real science Git HEAD")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.run_id), "invalid run ID")
    mount = guard.local_paths(args.work_dir, args.output_dir)
    require(not args.output_dir.exists(), "calibration requires a fresh output directory")
    for path in (args.teacher_input_root, args.dataset_root, args.hf_home, args.science_root):
        require(
            path.is_dir() and args.work_dir in path.parents, "inputs/science must be staged beneath node work"
        )
        require(
            path != args.output_dir
            and path not in args.output_dir.parents
            and args.output_dir not in path.parents,
            "input/science and output trees overlap",
        )
    require(
        args.work_dir in args.prerequisites.parents, "prerequisite proof must be staged beneath node work"
    )
    runtime = guard.runtime_identity()
    cache = guard.pinned_cache(args.hf_home)
    acceptance = helper("sdsc_student_contract").verify_teacher_acceptance(
        args.teacher_input_root / "acceptance", expected_inventory_sha256=ACCEPTANCE_INVENTORY_SHA256
    )
    prerequisite, proof = validate_prerequisites(args, acceptance)
    previous = Path.cwd()
    try:
        os.chdir(args.science_root)
        api = scientific_api(args.science_root)
        config, overrides, binding, reviewed = compose_base(args, api)
        validate_protocol_proof(prerequisite["protocol"], reviewed)
        inputs = teacher_inputs(args, config, api)
        report = {
            "kind": KIND,
            "task": TASK,
            "world_size": 2,
            "validation_only": not args.execute_calibration,
            "inputs_validated": True,
            "training_executed": False,
            "run_id": args.run_id,
            "code_sha256": args.code_sha256,
            "science_git_head": args.science_git_head,
            "student_protocol_sha256": args.student_protocol_sha256,
            "teacher_identity": acceptance["teacher_identity"],
            "accepted_teacher_sha256": acceptance["accepted_teacher_sha256"],
            "teacher_acceptance_inventory_sha256": acceptance["inventory_sha256"],
            "formal_binding": binding,
            "prerequisites": proof,
            "prerequisites_sha256": args.prerequisites_sha256,
            "runtime": runtime,
            "node_local_mount": mount,
            "pinned_cache": cache,
            "teacher_inputs": inputs,
            "plan": build_plan(args, overrides),
            "g0_passed": False,
            "factorial_ready": False,
            "pilot_passed": False,
            "execution_class_certified": False,
            "uses_blackwell_certificate_for_execution": False,
        }
        if args.execute_calibration:
            report = execute_calibration(args, config, overrides, binding, reviewed, api, report)
    finally:
        os.chdir(previous)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
