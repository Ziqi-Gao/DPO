#!/usr/bin/env python3
"""Node-local accepted preparation with bounded durable reports and full-state output."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GIB = 1024**3


def load_control(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("node plan changed")
    plan = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_branch.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != plan["control_sha256"]["tools/sdsc_student_branch.py"]
    ):
        raise ValueError("node controller changed")
    spec = importlib.util.spec_from_file_location("_preparation_node_control", source)
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)
    return control, control.validate_plan(plan)


def allocation(plan, control):
    # Frozen helper enforces W2, H100 visibility count, 24 CPU and 384 GiB.
    return control.helper("sdsc_student_lr_job").allocation(plan, control)


def mount(path, control):
    return control.helper("sdsc_student_lr_job").mount(path, control)


def memory_envelope(control, job):
    return control.helper("sdsc_student_lr_job").memory_envelope(control, job)


def copy_checkpoint(source, destination, row, control):
    """Exclusive stable stream copy plus full persistent read-back, never a fetch."""
    source, destination = control.safe(source), control.safe(destination)
    control.require(
        type(row.get("size")) is int
        and 0 < row["size"] <= 32 * GIB
        and re.fullmatch("[a-f0-9]{64}", row.get("sha256", "")),
        "checkpoint size/hash differs",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        control.require(
            stat.S_ISREG(before.st_mode) and before.st_size == row["size"], "checkpoint type/size differs"
        )
        digest, count = hashlib.sha256(), 0
        with os.fdopen(
            os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb"
        ) as output:
            while block := stream.read(8 * 1024**2):
                count += len(block)
                control.require(count <= row["size"], "checkpoint grew")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(stream.fileno())
    control.require(
        count == row["size"]
        and digest.hexdigest() == row["sha256"]
        and all(
            getattr(before, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "checkpoint changed during copy",
    )
    readback = hashlib.sha256()
    with os.fdopen(os.open(destination, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        control.require(stat.S_ISREG(info.st_mode) and info.st_size == count, "published checkpoint changed")
        while block := stream.read(8 * 1024**2):
            readback.update(block)
    control.require(readback.hexdigest() == row["sha256"], "persistent checkpoint readback differs")
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return dict(path=row["path"], size=count, sha256=readback.hexdigest())


def large_inventory(artifacts, control):
    rows = []
    for directory in ("checkpoints", "resume"):
        root = artifacts / directory
        if not root.exists():
            continue
        for folder, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                control.safe(Path(folder) / name)
            for name in sorted(files):
                path = Path(folder) / name
                info = path.lstat()
                control.require(
                    stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 32 * GIB, "unsafe full-state output"
                )
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while block := stream.read(8 * 1024**2):
                        digest.update(block)
                rows.append(
                    dict(
                        path=path.relative_to(artifacts).as_posix(),
                        size=info.st_size,
                        sha256=digest.hexdigest(),
                    )
                )
                control.require(
                    len(rows) <= 32 and sum(row["size"] for row in rows) <= 128 * GIB,
                    "full-state output exceeds bound",
                )
    return sorted(rows, key=lambda row: row["path"])


def persist(plan, control, artifacts, log, result, memory, measure):
    destination = control.safe(plan["result_dir"])
    control.require(not (destination / "receipt.json").exists(), "stage already published")
    large = []
    try:
        if artifacts is not None:
            for row in large_inventory(artifacts, control):
                large.append(
                    copy_checkpoint(artifacts / row["path"], destination / row["path"], row, control)
                )
            if result["stage_complete"]:
                report = control.document(artifacts / "prepare-report.json")
                published = {row["path"]: row for row in large}
                for row in report["checkpoints"]:
                    control.require(
                        published.get(row["path"]) == {key: row[key] for key in ("path", "size", "sha256")},
                        "dense checkpoint report differs from durable publication",
                    )
                control.require(
                    any(row["path"].startswith("resume/") for row in large),
                    "full-state resume artifacts absent",
                )
        else:
            control.require(not result["stage_complete"], "successful worker artifacts missing")
    except Exception as error:
        result.update(stage_complete=False, exit_code=1, checkpoint_publication_error=str(error)[:1500])
    result["large_artifacts"] = large
    try:
        measure("after_publication")
    except Exception as error:
        result.update(stage_complete=False, exit_code=1, memory_error=str(error)[:1500])
    records, total = [], 0
    for name in control.NAMES:
        if name in ("node-result.json", "memory.json"):
            raw = control.canonical(result if name == "node-result.json" else memory)
        elif artifacts is not None and (artifacts / name).exists():
            raw = control.read(artifacts / name)
        else:
            continue
        total += len(raw)
        control.require(total <= control.MAX_FETCH - control.CAP, "bounded report publication exceeded")
        control.write_once(destination / name, raw)
        control.require(control.read(destination / name) == raw, "persistent report readback differs")
        records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    if log is not None and log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 65536))
            control.write_once(destination / "worker.log", stream.read(65536))
    publication = dict(
        task=control.TASK,
        mode=plan["mode"],
        job_id=result["job_id"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=result["stage_complete"],
        stage_complete=result["stage_complete"],
        preparation_complete=result["stage_complete"] and plan["mode"] == "fit",
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        files=records,
        large_files=large,
        **dict.fromkeys(control.FLAGS, False),
    )
    raw = control.canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "publication readback differs")


def worker_argv(plan, source, science, inputs, artifacts):
    return [
        plan["python"],
        "-B",
        "-u",
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(science / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml"),
        "--num_processes",
        "2",
        "--num_cpu_threads_per_process",
        "12",
        "--main_process_port",
        "0",
        str(source / "tools/sdsc_student_branch_worker.py"),
        "--inputs-json",
        str(inputs),
        "--output-dir",
        str(artifacts),
    ]


def start_worker(plan, source, science, inputs, artifacts, environment, stream):
    return subprocess.Popen(
        worker_argv(plan, source, science, inputs, artifacts),
        cwd=science,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def build_worker_inputs(plan, work, science, identity, original_science, control):
    config = plan["resolved_config"]
    protocol = control.read(science / control.PROTOCOL_PATH)
    return dict(
        schema="quest-sdsc-student-branch-inputs-v3",
        mode=plan["mode"],
        job_id=identity["job_id"],
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
        science_root=str(science),
        original_science_files=original_science,
        dataset_root=str(work / "dataset"),
        hf_home=str(work / "huggingface"),
        protocol=dict(
            path=str(science / control.PROTOCOL_PATH),
            size=len(protocol),
            sha256=control.sha(protocol),
            protocol_sha256=plan["protocol"]["protocol_sha256"],
        ),
        original_config=dict(
            path=str(work / "config/resolved_config.yaml"), size=config["size"], sha256=config["sha256"]
        ),
        initial_checkpoint=dict(
            path=str(work / "checkpoints/initial.pt"), size=3441276375, sha256=control.INITIAL_SHA
        ),
    )


def main(argv=None):
    started = time.monotonic()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: node PLAN PLAN_SHA256")
    control, plan = load_control(*args)
    identity = allocation(plan, control)
    control.require(
        control.sha(control.read(Path(__file__)))
        == plan["control_sha256"]["tools/sdsc_student_branch_job.py"],
        "node wrapper changed",
    )
    control.verify_source(plan)
    worker_budget = control.worker_budget_seconds(plan)
    control.verify_parent(plan)
    control.verify_preflight(plan, accounting=False)
    source_root = Path(plan["parent"]["receipt"]["result_dir"])
    result_root = control.safe(plan["result_dir"])
    parent = result_root.parent
    while not parent.exists():
        parent = parent.parent
    mounts = {
        name: mount(path, control)
        for name, path in (("input", source_root), ("output", parent), ("cache", Path(plan["hf_home"])))
    }
    control.require(
        all(value["fstype"] in {"lustre", "nfs", "nfs4", "ceph"} for value in mounts.values())
        and control.PROJECT in result_root.parents,
        "actual GPU-node persistent mounts unavailable",
    )
    result_root.parent.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(mode=0o700)
    work = artifacts = log = process = None
    memory = {}
    result = dict(
        identity,
        task=control.TASK,
        mode=plan["mode"],
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
        stage_complete=False,
        mounts=mounts,
        **dict.fromkeys(control.FLAGS, False),
    )

    def measure(stage):
        try:
            memory[stage] = memory_envelope(control, identity["job_id"])
            raw = control.canonical(memory[stage])
            control.write_once(result_root / ("memory-" + stage + ".json"), raw)
            control.require(
                control.read(result_root / ("memory-" + stage + ".json")) == raw,
                "memory evidence readback differs",
            )
        except Exception as error:
            if hasattr(error, "evidence"):
                memory[stage] = error.evidence
            raise

    def interrupted(number, _frame):
        raise RuntimeError("preparation interrupted by signal " + str(number))

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        measure("initial")
        result["runtime"] = control.verify_runtime(plan)
        scratch = control.safe(
            os.environ.get("TMPDIR")
            or "/scratch/{}/job_{}".format(pwd.getpwuid(os.getuid()).pw_name, identity["job_id"])
        )
        control.require(scratch.is_dir() and scratch.stat().st_uid == os.getuid(), "owned scratch absent")
        local_mount = mount(scratch, control)
        control.require(
            local_mount["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"}
            and shutil.disk_usage(scratch).free >= 192 * GIB,
            "node-local free-space gate failed",
        )
        work = Path(tempfile.mkdtemp(prefix="opd-student-branch-" + identity["job_id"] + "-", dir=scratch))
        result.update(work_dir=str(work), local_mount=local_mount)
        source, science = work / "source", work / "science"
        staging = control.helper("sdsc_student_job")
        staging.code_hash, staging.run_id = plan["code_sha256"], plan["run_id"]
        staging.stage_source(Path(plan["release"]), source)
        result["science_restore"] = control.verify_science(plan, science)
        contract = control.helper("sdsc_student_contract", source)
        parent_receipt = plan["parent"]["receipt"]
        proof = control.document(parent_receipt["prerequisites_path"], parent_receipt["prerequisites_sha256"])
        old_science = proof["protocol"]["science_file_sha256"]
        control.require(
            len(old_science) == 49
            and control.sha(control.canonical(old_science)) == control._initial.NAMED_SCIENCE_MAP_SHA,
            "original 49-file science inventory differs",
        )
        for name, digest in old_science.items():
            control.require(
                staging.file_hash(control.safe(science / name)) == digest, "original scientific byte differs"
            )
        result["checkpoint_staging"] = contract.stage_inputs(
            [
                dict(row, source=str(source_root / row["path"]), path="initial.pt")
                for row in plan["checkpoints"]
            ],
            work / "checkpoints",
            maximum=4 * GIB,
        )
        config = plan["resolved_config"]
        result["config_staging"] = contract.stage_inputs(
            [dict(config, source=str(source_root / config["path"]), path="resolved_config.yaml")],
            work / "config",
            maximum=control.CAP,
        )
        result["dataset_staging"] = contract.stage_inputs(
            plan["dataset_inputs"], work / "dataset", maximum=2 * GIB
        )
        staging.models = (("models--Qwen--Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),)
        result["model_staging"] = staging.stage_models(Path(plan["hf_home"]), work / "huggingface")
        measure("after_staging")
        artifacts = work / "artifacts"
        artifacts.mkdir()
        inputs = build_worker_inputs(plan, work, science, identity, old_science, control)
        control.write_once(work / "inputs.json", control.canonical(inputs))
        environment = {
            key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH")
        }
        environment.update(
            HF_HOME=str(work / "huggingface"),
            HF_HUB_CACHE=str(work / "huggingface/hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="12",
            MKL_NUM_THREADS="12",
            OPENBLAS_NUM_THREADS="12",
            NUMEXPR_NUM_THREADS="12",
            TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
            TMPDIR=str(work),
            XDG_CACHE_HOME=str(work / "cache"),
        )
        log = work / "worker.log"
        with log.open("xb") as stream:
            process = start_worker(
                plan, source, science, work / "inputs.json", artifacts, environment, stream
            )
            deadline = started + worker_budget
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                control.require(remaining > 0, "preparation worker exceeded finite deadline")
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    measure("running-" + str(int(time.monotonic())))
        control.require(process.returncode == 0, "preparation worker failed; preserve outputs, never retry")
        measure("after_training")
        report = control.document(artifacts / "prepare-report.json")
        raw_records = {
            name: dict(path=name, size=len(raw), sha256=control.sha(raw))
            for name in control.NAMES
            if name != "prepare-report.json" and (artifacts / name).exists()
            for raw in (control.read(artifacts / name),)
        }
        control.validate_worker_report(report, plan, identity["job_id"], raw_records)
        control.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"],
            "CUDA assignment changed",
        )
        result.update(stage_complete=True, exit_code=0)
    except BaseException as error:
        result.update(error=type(error).__name__ + ": " + str(error)[:1500], exit_code=1)
    finally:
        if process is not None:
            try:
                control.helper("sdsc_student_lr_job").stop_worker_group(process)
            except Exception as error:
                result.update(stage_complete=False, exit_code=1, worker_shutdown_error=str(error)[:1500])
                artifacts = log = None
        try:
            measure("final")
        except Exception as error:
            result.update(stage_complete=False, exit_code=1, memory_error=str(error)[:1500])
        result["elapsed_seconds"] = time.monotonic() - started
        persist(plan, control, artifacts, log, result, memory, measure)
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
