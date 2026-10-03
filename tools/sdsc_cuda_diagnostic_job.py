#!/usr/bin/env python3
"""One bounded CUDA infrastructure observation on the assigned allocation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GIB = 1024**3
WORKER_SECONDS = 210


class MemoryEnvelopeError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _counters(path):
    if not path.is_file():
        return None
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        if key in result:
            raise ValueError("duplicate memory counter")
        result[key] = int(value)
    return result


def memory_envelope(*, expected_bytes, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    evidence = {
        "expected_limit_bytes": expected_bytes,
        "observed_at_unix": time.time(),
        "ancestors": [],
        "passed": False,
    }
    try:

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        require(
            type(expected_bytes) is int and expected_bytes == 16 * GIB,
            "unreviewed CUDA diagnostic memory envelope",
        )
        locations = []
        for line in proc.read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            require(relative.startswith("/") and ".." not in Path(relative).parts, "invalid cgroup path")
            if hierarchy == "0" and not controllers:
                locations.append(
                    (root, root / relative.lstrip("/"), ("memory.max", "memory.current", "memory.peak"), 2)
                )
            elif "memory" in controllers.split(","):
                locations.insert(
                    0,
                    (
                        root / "memory",
                        root / "memory" / relative.lstrip("/"),
                        ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes"),
                        1,
                    ),
                )
        require(locations, "no readable memory cgroup")
        boundary, directory, names, version = locations[0]
        evidence["cgroup_version"] = version
        levels = evidence["ancestors"]
        while True:
            level = {"path": str(directory), "limit_bytes": None}
            levels.append(level)
            if (directory / names[0]).is_file():
                raw = (directory / names[0]).read_text().strip()
                limit = None if raw == "max" else int(raw)
                level["raw_limit"] = raw
                require(limit is None or limit > 0, "invalid cgroup memory limit")
                level["limit_bytes"] = limit if limit is not None and limit < 2**60 else None
                for name, key in zip(names[1:], ("current_bytes", "peak_bytes"), strict=True):
                    if (directory / name).is_file():
                        level[key] = int((directory / name).read_text().strip())
                level["memory_stat"] = _counters(directory / "memory.stat")
                level["memory_events"] = _counters(directory / "memory.events")
                level["memory_events_local"] = _counters(directory / "memory.events.local")
                level["memory_oom_control"] = _counters(directory / "memory.oom_control")
                failcnt = directory / "memory.failcnt"
                level["memory_failcnt"] = int(failcnt.read_text().strip()) if failcnt.is_file() else None
            if directory == boundary:
                break
            require(boundary in directory.parents, "memory cgroup escaped controller root")
            directory = directory.parent
        finite = [level for level in levels if level["limit_bytes"] is not None]
        require(finite, "cgroup must expose a finite CUDA diagnostic memory limit")
        limit = min(level["limit_bytes"] for level in finite)
        limiting = [level for level in finite if level["limit_bytes"] == limit]
        require(
            all("current_bytes" in level and "peak_bytes" in level for level in limiting),
            "limiting memory cgroup must expose current and peak usage",
        )
        selected = max(limiting, key=lambda level: level["peak_bytes"])
        current, peak = selected["current_bytes"], selected["peak_bytes"]
        required = GIB
        evidence.update(
            path=selected["path"],
            limit_bytes=limit,
            current_bytes=current,
            peak_bytes=peak,
            headroom_bytes=limit - peak,
            minimum_headroom_bytes=required,
            memory_stat=selected.get("memory_stat"),
            memory_events=selected.get("memory_events"),
            memory_events_local=selected.get("memory_events_local"),
            memory_oom_control=selected.get("memory_oom_control"),
            memory_failcnt=selected.get("memory_failcnt"),
        )
        require(
            limit == expected_bytes
            and all(0 < item["current_bytes"] <= item["peak_bytes"] <= limit for item in limiting),
            "cgroup limit or measured peak differs from reviewed CUDA diagnostic memory envelope",
        )
        require(
            limit - peak >= required,
            f"CUDA diagnostic lacks 1 GiB headroom: peak={peak}, current={current}",
        )
        evidence["passed"] = True
        return evidence
    except (ValueError, OSError) as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise MemoryEnvelopeError(str(error), evidence) from error


def load_control(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("node plan changed")
    plan = json.loads(raw)
    source = Path(__file__).with_name("sdsc_cuda_diagnostic.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != plan["control_sha256"]["tools/sdsc_cuda_diagnostic.py"]
    ):
        raise ValueError("node controller changed")
    spec = importlib.util.spec_from_file_location("_cuda_node_control", source)
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)
    control.require(
        control.sha(control.read(Path(__file__)))
        == plan["control_sha256"]["tools/sdsc_cuda_diagnostic_job.py"],
        "node wrapper changed",
    )
    return control, control.validate_plan(plan)


def allocation(plan, control):
    expected = dict(
        SLURM_CPUS_PER_TASK="4",
        SLURM_MEM_PER_NODE="16384",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
    )
    job = os.environ.get("SLURM_JOB_ID", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    control.require(
        re.fullmatch("[1-9][0-9]*", job) and all(os.environ.get(k) == v for k, v in expected.items()),
        "actual diagnostic allocation differs",
    )
    control.require(
        len(devices) == 2
        and len(set(devices)) == 2
        and all(x and x.strip() == x and x not in ("-1", "NoDevFiles") for x in devices),
        "exactly two assigned CUDA entries required",
    )
    return dict(job_id=job, cuda_visible_devices=visible, cpus=4, memory_mib=16384, world_size=1)


def observe_memory(control, job, **kwargs):
    value = memory_envelope(expected_bytes=16 * GIB, **kwargs)
    try:
        control._prior.validate_job_memory_events(value, job)
        control.require("job_" + job in Path(value["path"]).parts, "memory belongs to another job")
    except ValueError as error:
        value.update(passed=False, error=str(error))
        raise MemoryEnvelopeError(str(error), value) from error
    return value


def worker_argv(plan, work, identity):
    return [
        plan["python"],
        "-I",
        "-B",
        "-u",
        str(Path(plan["release"]) / "source/tools/sdsc_cuda_diagnostic_worker.py"),
        "--output-json",
        str(work / "cuda-diagnostic.json"),
        "--job-id",
        identity["job_id"],
        "--expected-cuda-visible-devices",
        identity["cuda_visible_devices"],
    ]


def persist(plan, control, work, result, memory, measure):
    destination = control.safe(plan["result_dir"])
    records = []
    if (
        work is not None
        and "worker_shutdown_error" not in result
        and (work / "cuda-diagnostic.json").exists()
    ):
        try:
            raw = control.read(work / "cuda-diagnostic.json")
            report = control.validate_worker_report(
                json.loads(raw), result["job_id"], result["cuda_visible_devices"]
            )
            result.update(diagnostic_complete=report["diagnostic_complete"], cuda_ready=report["cuda_ready"])
            control.write_once(destination / "cuda-diagnostic.json", raw)
            control.require(
                control.read(destination / "cuda-diagnostic.json") == raw, "report readback differs"
            )
            records.append(dict(path="cuda-diagnostic.json", size=len(raw), sha256=control.sha(raw)))
        except Exception as error:
            result.update(
                exit_code=1,
                diagnostic_complete=False,
                cuda_ready=False,
                report_publication_error=type(error).__name__ + ": " + str(error)[:1500],
            )
    if work is not None and (work / "worker.log").exists():
        with (work / "worker.log").open("rb") as stream:
            stream.seek(max(0, (work / "worker.log").stat().st_size - 65536))
            control.write_once(destination / "worker.log", stream.read(65536))
    try:
        measure("after_publication")
        if result["exit_code"] == 0:
            control.validate_memory(memory, result["job_id"])
    except Exception as error:
        result.update(exit_code=1, memory_error=str(error)[:1500])
    for name, obj in [("node-result.json", result), ("memory.json", memory)]:
        raw = control.canonical(obj)
        control.require(len(raw) <= control.MAX_FILE, "metadata exceeds bound")
        control.write_once(destination / name, raw)
        control.require(control.read(destination / name) == raw, "metadata readback differs")
        records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    publication = dict(
        task=control.TASK,
        job_id=result["job_id"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        code_sha256=plan["code_sha256"],
        diagnostic_complete=result["diagnostic_complete"],
        cuda_ready=result["cuda_ready"],
        persistent_read_back_verified=True,
        files=records,
        **dict.fromkeys(control.FLAGS, False),
    )
    raw = control.canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "receipt readback differs")


def main(argv=None):
    started = time.monotonic()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: node PLAN PLAN_SHA256")
    control, plan = load_control(*args)
    identity = allocation(plan, control)
    result = dict(
        identity,
        task=control.TASK,
        plan_sha256=control.sha(control.canonical(plan)),
        run_id=plan["run_id"],
        code_sha256=plan["code_sha256"],
        diagnostic_complete=False,
        cuda_ready=False,
        exit_code=1,
        **dict.fromkeys(control.FLAGS, False),
    )
    destination = control.safe(plan["result_dir"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)
    work = process = None
    deadline = started + WORKER_SECONDS
    memory = {}

    def measure(stage):
        try:
            memory[stage] = observe_memory(control, identity["job_id"])
        except MemoryEnvelopeError as error:
            memory[stage] = error.evidence
            raise

    def interrupted(number, _frame):
        raise RuntimeError("diagnostic interrupted by signal " + str(number))

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        control.verify_source(plan)
        control.require(time.monotonic() < deadline, "source verification exhausted diagnostic deadline")
        measure("initial")
        helper = control.helper("sdsc_student_lr_job")
        result["persistent_mount"] = helper.mount(destination, control)
        control.require(time.monotonic() < deadline, "persistent mount check exhausted diagnostic deadline")
        control.require(
            result["persistent_mount"]["fstype"] in {"lustre", "nfs", "nfs4", "ceph"},
            "persistent GPU-node mount unavailable",
        )
        scratch = control.safe(
            os.environ.get("TMPDIR")
            or "/scratch/{}/job_{}".format(pwd.getpwuid(os.getuid()).pw_name, identity["job_id"])
        )
        control.require(scratch.is_dir() and scratch.stat().st_uid == os.getuid(), "owned scratch absent")
        result["local_mount"] = helper.mount(scratch, control)
        control.require(time.monotonic() < deadline, "local mount check exhausted diagnostic deadline")
        control.require(
            result["local_mount"]["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"}
            and shutil.disk_usage(scratch).free >= 64 * control.CAP,
            "small node-local workspace unavailable",
        )
        work = Path(tempfile.mkdtemp(prefix="opd-cuda-" + identity["job_id"] + "-", dir=scratch))
        result["work_dir"] = str(work)
        environment = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
        environment.update(
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
        )
        control.require(time.monotonic() < deadline, "setup exhausted diagnostic deadline")
        with (work / "worker.log").open("xb") as stream:
            process = subprocess.Popen(
                worker_argv(plan, work, identity),
                cwd=work,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                control.require(remaining > 0, "CUDA worker exceeded finite deadline")
                try:
                    process.wait(timeout=min(15, remaining))
                except subprocess.TimeoutExpired:
                    measure("running-" + str(int(time.monotonic())))
        result["worker_exit_code"] = process.returncode
        report = control.validate_worker_report(
            control.document(work / "cuda-diagnostic.json"),
            identity["job_id"],
            identity["cuda_visible_devices"],
        )
        result.update(diagnostic_complete=report["diagnostic_complete"], cuda_ready=report["cuda_ready"])
        control.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"],
            "node CUDA visibility changed",
        )
        control.require(
            process.returncode == 0 and report["diagnostic_complete"] and report["cuda_ready"],
            "diagnostic captured CUDA failure; preserve evidence",
        )
        result["exit_code"] = 0
    except BaseException as error:
        result.update(exit_code=1, error=type(error).__name__ + ": " + str(error)[:1500])
    finally:
        if process is not None:
            try:
                control.helper("sdsc_student_lr_job").stop_worker_group(process)
            except Exception as error:
                result.update(
                    exit_code=1,
                    diagnostic_complete=False,
                    cuda_ready=False,
                    worker_shutdown_error=str(error)[:1500],
                )
        try:
            measure("final")
        except Exception as error:
            result.update(exit_code=1, memory_error=str(error)[:1500])
        result["elapsed_seconds"] = time.monotonic() - started
        persist(plan, control, work, result, memory, measure)
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
