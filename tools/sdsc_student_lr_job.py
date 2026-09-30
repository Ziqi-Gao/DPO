#!/usr/bin/env python3
"""Node-local staging and durable publication for a fixed two-rank learning-rate diagnostic."""

from __future__ import annotations

import contextlib
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


def memory_envelope(control, job, *, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    # Preserve the accepted aggregate guard and add explicit own-job OOM rejection.
    guard = control.helper("sdsc_student_memory")
    evidence = guard.memory_envelope(expected_bytes=384 * GIB, proc=proc, root=root)
    try:
        control.validate_job_memory_events(evidence, job)
    except ValueError as error:
        evidence.update(passed=False, error=type(error).__name__ + ": " + str(error))
        raise guard.MemoryEnvelopeError(str(error), evidence) from error
    return evidence


def load_control(plan, plan_sha):
    raw = Path(plan).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan_sha:
        raise ValueError("node plan changed")
    value = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_lr.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != value["control_sha256"]["tools/sdsc_student_lr.py"]:
        raise ValueError("node controller changed")
    spec = importlib.util.spec_from_file_location("_quality_node_control", source)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    return control, control.validate_plan(value)


def allocation(plan, control):
    env = os.environ
    job = env.get("SLURM_JOB_ID", "")
    expected = dict(
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="393216",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
    )
    control.require(
        re.fullmatch("[1-9][0-9]*", job) and all(env.get(k) == v for k, v in expected.items()),
        "actual diagnostic Slurm allocation differs",
    )
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    control.require(
        len(devices) == 2
        and len(set(devices)) == 2
        and all(item and item.strip() == item and item not in ("-1", "NoDevFiles") for item in devices),
        "exactly two distinct assigned GPUs are required",
    )
    return dict(
        job_id=job,
        cpus=24,
        memory_mib=393216,
        world_size=2,
        cuda_visible_devices=visible,
        account="nwu181",
        partition="nairr-gpu-shared",
    )


def mount(path, control):
    result = control.run(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"], 10
    )
    control.require(result["returncode"] == 0, "mount unavailable on allocated node")
    return json.loads(result["stdout"])["filesystems"][0]


LARGE_PATHS = ("checkpoints/lr-5e-4-step4.pt", "checkpoints/lr-5e-5-step4.pt")
MAX_CHECKPOINT = 8 * GIB


def copy_checkpoint(source, destination, record, control):
    """Exclusive, bounded stream copy with source and durable destination read-back."""
    source, destination = control.safe(source), control.safe(destination)
    control.require(
        type(record.get("size")) is int
        and 0 < record["size"] <= MAX_CHECKPOINT
        and re.fullmatch("[a-f0-9]{64}", record.get("sha256", "")),
        "invalid model-only checkpoint inventory",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        control.require(
            stat.S_ISREG(before.st_mode) and before.st_size == record["size"],
            "checkpoint source type/size differs",
        )
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(
            os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb"
        ) as output:
            while chunk := stream.read(8 * 1024**2):
                total += len(chunk)
                control.require(total <= record["size"], "checkpoint grew during publication")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(stream.fileno())
    control.require(
        total == before.st_size == after.st_size
        and (before.st_mtime_ns, before.st_ctime_ns) == (after.st_mtime_ns, after.st_ctime_ns)
        and digest.hexdigest() == record["sha256"],
        "checkpoint source hash/identity changed",
    )
    readback = hashlib.sha256()
    with os.fdopen(os.open(destination, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        observed = os.fstat(stream.fileno())
        control.require(
            stat.S_ISREG(observed.st_mode) and observed.st_size == total,
            "persistent checkpoint size/type differs",
        )
        while chunk := stream.read(8 * 1024**2):
            readback.update(chunk)
    control.require(readback.hexdigest() == record["sha256"], "persistent checkpoint read-back hash differs")
    return dict(path=record["path"], size=total, sha256=readback.hexdigest())


def persist_checkpoints(artifacts, destination, result, control, records):
    if artifacts is None or not (artifacts / "lr-probe.json").exists():
        control.require(not result["diagnostic_complete"], "successful diagnostic has no report")
        return records
    report = control.document(artifacts / "lr-probe.json")
    supplied = report.get("checkpoints", [])
    control.require(isinstance(supplied, list) and len(supplied) <= 2, "invalid checkpoint list")
    seen = set()
    for row in supplied:
        name = row.get("path")
        control.require(
            name in LARGE_PATHS
            and name not in seen
            and row.get("step") == 4
            and row.get("scope") == "diagnostic_model_only",
            "unreviewed or duplicate checkpoint publication",
        )
        seen.add(name)
        records.append(copy_checkpoint(artifacts / name, destination / name, row, control))
    if result["diagnostic_complete"]:
        control.require(seen == set(LARGE_PATHS), "successful diagnostic lacks both final model checkpoints")
    return records


def persist(plan, control, artifacts, log, result, memory, final_measure=None):
    """Publish stopped-worker outputs exactly; a failed diagnostic remains failed."""
    destination = control.safe(plan["result_dir"])
    control.require(not (destination / "receipt.json").exists(), "diagnostic already published")
    large_records = []
    try:
        persist_checkpoints(artifacts, destination, result, control, large_records)
    except Exception as error:
        result.update(
            diagnostic_complete=False,
            exit_code=1,
            checkpoint_publication_error=type(error).__name__ + ": " + str(error)[:1500],
        )
    result["large_artifacts"] = large_records
    if final_measure is not None:
        try:
            final_measure("after_publication")
        except Exception as error:
            result.update(
                diagnostic_complete=False,
                exit_code=1,
                memory_error=type(error).__name__ + ": " + str(error)[:1500],
            )
    if artifacts is not None:
        for name, value in [("node-result.json", result), ("memory.json", memory)]:
            control.write_once(artifacts / name, control.canonical(value))
    records = []
    total = 0
    for name in control.NAMES:
        path = artifacts / name if artifacts is not None else None
        if path is None or not path.exists():
            continue
        raw = control.read(path)
        total += len(raw)
        control.require(
            total <= control.MAX_FETCH - 256 * 1024, "diagnostic output exceeds bounded publication"
        )
        control.write_once(destination / name, raw)
        control.require(control.read(destination / name) == raw, "persistent diagnostic readback differs")
        records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    if artifacts is None:
        for name, value in [("node-result.json", result), ("memory.json", memory)]:
            raw = control.canonical(value)
            control.write_once(destination / name, raw)
            control.require(control.read(destination / name) == raw, "persistent diagnostic readback differs")
            records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    if log is not None and log.exists():
        # A bounded log tail is sufficient; raw scientific JSONL remains whole.
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 64 * 1024))
            control.write_once(destination / "worker.log", stream.read(64 * 1024))
    publication = dict(
        task=control.TASK,
        job_id=result["job_id"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=result["diagnostic_complete"],
        diagnostic_complete=result["diagnostic_complete"],
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
        persistent_read_back_verified=True,
        files=records,
        large_files=large_records,
        large_files_read_back_verified=True,
    )
    raw = control.canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "publication receipt readback differs")


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
        str(source / "tools/sdsc_student_lr_probe.py"),
        "--inputs-json",
        str(inputs),
        "--output-dir",
        str(artifacts),
    ]


def wait_worker(process, *, timeout, artifacts, control, measure, result_root):
    """Finite wait on the one launcher; publish bounded progress without retries."""
    deadline = time.monotonic() + timeout
    last_digest = None
    sequence = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            process.wait(timeout=min(30, remaining))
            return
        except subprocess.TimeoutExpired:
            measure("running-" + str(int(time.monotonic())))
            path = artifacts / "progress.json"
            if not path.exists():
                continue
            raw = control.read(path, 512 * 1024)
            digest = control.sha(raw)
            if digest == last_digest:
                continue
            sequence += 1
            control.require(sequence <= 64, "unexpected diagnostic progress frequency")
            progress = json.loads(raw)
            control.require(isinstance(progress, dict), "invalid diagnostic progress")
            # Full evidence stays in a bounded file; the login Slurm tail receives only a summary.
            print(
                json.dumps(
                    dict(
                        phase="lr_progress",
                        sequence=sequence,
                        arm=progress.get("arm"),
                        step=progress.get("step"),
                    )
                ),
                flush=True,
            )
            saved = result_root / ("progress-" + str(sequence) + ".json")
            control.write_once(saved, raw)
            control.require(control.read(saved) == raw, "progress read-back differs")
            last_digest = digest


def build_worker_inputs(plan, work, science, audit_inputs, identity, config, control):
    inputs = dict(
        schema="quest-sdsc-student-lr-inputs-v1",
        initial_checkpoint=dict(
            path=str(work / "checkpoints/initial.pt"),
            size=plan["checkpoints"][0]["size"],
            sha256=control.INITIAL_SHA,
        ),
        resolved_config=dict(
            path=str(work / "config/resolved_config.yaml"), size=config["size"], sha256=config["sha256"]
        ),
        dataset_root=str(work / "dataset"),
        dataset_manifest_sha256=next(
            row["sha256"] for row in plan["dataset_inputs"] if row["path"] == "manifest.json"
        ),
        science_root=str(science),
        teacher_bundle_root=str(work / "teacher-inputs"),
        data_audit=audit_inputs,
        hf_home=str(work / "huggingface"),
        job_id=identity["job_id"],
        run_id=plan["run_id"],
        parent_job_id=control.PARENT_JOB,
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
    )
    return inputs


def group_has_live_members(pgid, proc=Path("/proc")):
    """Only the launcher's new session/group; zombies cannot mutate artifacts."""
    for item in proc.iterdir():
        if not item.name.isdecimal():
            continue
        try:
            fields = (item / "stat").read_text().rsplit(") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # /proc/PID/stat fields 3, 5 and 6: state, group, session.
        if int(fields[2]) == pgid and int(fields[3]) == pgid and fields[0] not in {"Z", "X"}:
            return True
    return False


def stop_worker_group(process, *, alive=group_has_live_members):
    """Reap this launcher's group even after its leader has already exited."""
    pgid = process.pid
    for number, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        if not alive(pgid):
            process.wait(timeout=1)
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, number)
        deadline = time.monotonic() + timeout
        while alive(pgid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.2)
    if alive(pgid):
        raise RuntimeError("own worker group still live; cannot publish mutable artifacts")
    process.wait(timeout=1)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: node PLAN PLAN_SHA256")
    control, plan = load_control(*args)
    identity = allocation(plan, control)
    control.require(
        control.sha(control.read(Path(__file__))) == plan["control_sha256"]["tools/sdsc_student_lr_job.py"],
        "node wrapper changed",
    )
    control.verify_source(plan)
    control.verify_parent(plan)
    project = control.PROJECT
    source_root = Path(plan["parent"]["receipt"]["result_dir"])
    result_root = control.safe(plan["result_dir"])
    parent = result_root.parent
    while not parent.exists():
        parent = parent.parent
    input_mount = mount(source_root, control)
    output_mount = mount(parent, control)
    cache_mount = mount(Path(plan["hf_home"]), control)
    control.require(
        all(
            m["fstype"] in {"lustre", "nfs", "nfs4", "ceph"} for m in (input_mount, output_mount, cache_mount)
        )
        and project in result_root.parents,
        "GPU node has not confirmed persistent input/output/cache mounts",
    )
    result_root.parent.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(mode=0o700)
    work = artifacts = log = process = None
    memory = {}
    started = time.monotonic()
    result = dict(
        identity,
        task=control.TASK,
        run_id=plan["run_id"],
        parent_job_id=control.PARENT_JOB,
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
        diagnostic_complete=False,
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
        mounts=dict(input=input_mount, output=output_mount, cache=cache_mount),
    )

    def measure(stage):
        try:
            memory[stage] = memory_envelope(control, identity["job_id"])
            raw = control.canonical(memory[stage])
            control.write_once(result_root / ("memory-" + stage + ".json"), raw)
            control.require(
                control.read(result_root / ("memory-" + stage + ".json")) == raw,
                "memory evidence persistent readback differs",
            )
        except Exception as error:
            if hasattr(error, "evidence"):
                memory[stage] = error.evidence
            raise

    def interrupted(number, _frame):
        raise RuntimeError("diagnostic interrupted by signal " + str(number))

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        measure("initial")
        scratch = control.safe(
            os.environ.get("TMPDIR")
            or "/scratch/{}/job_{}".format(pwd.getpwuid(os.getuid()).pw_name, identity["job_id"])
        )
        control.require(
            scratch.is_dir() and scratch.stat().st_uid == os.getuid(), "owned node-local scratch absent"
        )
        local_mount = mount(scratch, control)
        control.require(
            local_mount["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"}
            and shutil.disk_usage(scratch).free >= 96 * 1024**3,
            "node-local filesystem/free-space requirement failed",
        )
        work = Path(tempfile.mkdtemp(prefix="opd-student-lr-" + identity["job_id"] + "-", dir=scratch))
        result.update(work_dir=str(work), local_mount=local_mount)
        source = work / "source"
        staging = control.helper("sdsc_student_job")
        staging.code_hash = plan["code_sha256"]
        staging.run_id = plan["run_id"]
        staging.stage_source(Path(plan["release"]), source)
        contract = control.helper("sdsc_student_contract", source)
        verifier = control.helper("sdsc_provenance", source)
        science = work / "science"
        parent_receipt = plan["parent"]["receipt"]
        proof = contract.document(
            parent_receipt["prerequisites_path"], parent_receipt["prerequisites_sha256"]
        )
        restored = verifier.verify(
            control.safe(parent_receipt["provenance_dir"]),
            parent_receipt["provenance_manifest_sha256"],
            parent_receipt["code_sha256"],
            science,
        )
        protocol = contract.resolved_protocol(science, parent_receipt["science_git_head"])
        control.require(protocol == proof["protocol"], "restored parent science protocol differs")
        for name, digest in protocol["science_file_sha256"].items():
            control.require(
                staging.file_hash(control.safe(science / name)) == digest,
                "restored scientific byte differs: " + name,
            )
        result["science_restore"] = dict(
            head=restored["git_head"],
            bundle_sha256=restored["bundle_sha256"],
            protocol_sha256=protocol["protocol_sha256"],
            science_file_sha256=protocol["science_file_sha256"],
            original_implementation_only=True,
        )
        checkpoint_rows = [
            dict(row, source=str(source_root / row["path"]), path=row["label"] + ".pt")
            for row in plan["checkpoints"]
        ]
        result["checkpoint_staging"] = contract.stage_inputs(
            checkpoint_rows, work / "checkpoints", maximum=4 * 1024**3
        )
        config = plan["resolved_config"]
        result["config_staging"] = contract.stage_inputs(
            [dict(config, source=str(source_root / config["path"]), path="resolved_config.yaml")],
            work / "config",
            maximum=control.CAP,
        )
        result["teacher_staging"] = contract.stage_inputs(
            plan["teacher_inputs"], work / "teacher-inputs", maximum=32 * 1024**2
        )
        result["teacher_acceptance"] = contract.verify_teacher_acceptance(
            work / "teacher-inputs/acceptance", expected_inventory_sha256=contract.INVENTORY_SHA256
        )
        audit_dir = work / "data-audit"
        audit_dir.mkdir()
        audit_inputs = {}
        audit_raw = control.decode_audit(plan["audit"])
        for key in ("report", "capture"):
            entry = plan["audit"][key]
            raw = audit_raw[key]
            control.require(
                len(raw) == entry["size"] and control.sha(raw) == entry["sha256"],
                "embedded data audit changed",
            )
            path = audit_dir / (key + ".json")
            control.write_once(path, raw)
            control.require(control.read(path) == raw, "local audit read-back failed")
            audit_inputs[key] = dict(path=str(path), size=len(raw), sha256=entry["sha256"])
        result["data_audit"] = audit_inputs
        result["dataset_staging"] = contract.stage_inputs(
            plan["dataset_inputs"], work / "dataset", maximum=2 * 1024**3
        )
        # Reuse the reviewed confined cache copier for only the pinned 1.7B snapshot.
        staging.models = (("models--Qwen--Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),)
        result["model_staging"] = staging.stage_models(Path(plan["hf_home"]), work / "huggingface")
        result["model_files"] = [
            dict(
                path=str(path.relative_to(work / "huggingface")),
                size=path.stat().st_size,
                sha256=staging.file_hash(path),
            )
            for path in sorted((work / "huggingface").rglob("*"))
            if path.is_file()
        ]
        measure("after_staging")
        artifacts = work / "artifacts"
        artifacts.mkdir()
        inputs = build_worker_inputs(plan, work, science, audit_inputs, identity, config, control)
        control.write_once(work / "inputs.json", control.canonical(inputs))
        environment = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH"):
            environment.pop(key, None)
        environment.update(
            PYTHONPATH=str(science / "src"),
            HF_HOME=str(work / "huggingface"),
            HF_HUB_CACHE=str(work / "huggingface/hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="12",
            MKL_NUM_THREADS="12",
            OPENBLAS_NUM_THREADS="12",
            NUMEXPR_NUM_THREADS="12",
            TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
            HF_DATASETS_OFFLINE="1",
            TMPDIR=str(work),
            XDG_CACHE_HOME=str(work / "cache"),
        )
        log = work / "worker.log"
        with log.open("xb") as stream:
            process = subprocess.Popen(
                worker_argv(plan, source, science, work / "inputs.json", artifacts),
                cwd=source,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(json.dumps(dict(phase="lr_worker_started", job_id=identity["job_id"])), flush=True)
            wait_worker(
                process,
                timeout=max(1, 3300 - int(time.monotonic() - started)),
                artifacts=artifacts,
                control=control,
                measure=measure,
                result_root=result_root,
            )
        control.require(
            process.returncode == 0, "diagnostic worker failed; preserve outputs, never retry automatically"
        )
        measure("after_training")
        report = control.document(artifacts / "lr-probe.json")
        control.require(
            report.get("diagnostic_complete") is True
            and report.get("passed") is True
            and all(report.get(k) is False for k in control.FLAGS),
            "worker incomplete or claims scientific acceptance",
        )
        raw_records = {}
        for name in ("lr-records.jsonl", "lr-prompts.jsonl", "lr-batches.jsonl"):
            raw = control.read(artifacts / name)
            raw_records[name] = dict(path=name, size=len(raw), sha256=control.sha(raw))
        control.validate_worker_report(report, plan, identity["job_id"], raw_records)
        control.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"],
            "assigned CUDA visibility changed",
        )
        result.update(diagnostic_complete=True, exit_code=0)
    except BaseException as error:
        result.update(error=type(error).__name__ + ": " + str(error)[:1500], exit_code=1)
    finally:
        if process is not None:
            try:
                stop_worker_group(process)
            except Exception as error:
                result.update(
                    diagnostic_complete=False,
                    exit_code=1,
                    worker_shutdown_error=type(error).__name__ + ": " + str(error)[:1500],
                )
                # Never hash changing worker artifacts as a final publication.
                artifacts = log = None
        try:
            measure("final")
        except Exception as error:
            result.update(diagnostic_complete=False, exit_code=1, memory_error=str(error))
        result["elapsed_seconds"] = time.monotonic() - started
        persist(plan, control, artifacts, log, result, memory, final_measure=measure)
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
