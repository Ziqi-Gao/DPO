#!/usr/bin/env python3
"""Sequential, bounded import observations on one assigned allocation; no CUDA calls."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import pwd
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LABELS = ("A1", "B", "A2")
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
SAMPLE_SECONDS = 5.0
CHILD_SECONDS = 180.0
WORKER_SECONDS = 540.0
SHUTDOWN_SECONDS = 5.0
MAX_LOG = 1024**2
MAX_PROC = 512 * 1024
MAX_REPORT = 64 * 1024
MAX_TOTAL = 8 * 1024**2
FLAGS = dict.fromkeys(
    (
        "student_accepted",
        "g0_passed",
        "pilot_passed",
        "factorial_ready",
        "formal_initial_accepted",
        "execution_class_certified",
        "cuda_observed",
        "model_loaded",
        "training_started",
    ),
    False,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_control(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "node plan changed")
    plan = json.loads(raw)
    source = Path(__file__).with_name("sdsc_torch_import_probe.py")
    require(
        hashlib.sha256(source.read_bytes()).hexdigest()
        == plan["control_sha256"]["tools/sdsc_torch_import_probe.py"],
        "node controller changed",
    )
    spec = importlib.util.spec_from_file_location("_import_node_control", source)
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)
    require(
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        == plan["control_sha256"]["tools/sdsc_torch_import_probe_job.py"],
        "node wrapper changed",
    )
    return control, control.validate_plan(plan)


def allocation(plan):
    expected = dict(
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="16384",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
    )
    job, visible = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = visible.split(",")
    require(
        re.fullmatch("[1-9][0-9]*", job) and all(os.environ.get(k) == v for k, v in expected.items()),
        "actual import diagnostic allocation differs",
    )
    require(
        len(devices) == len(set(devices)) == 2
        and all(x and x.strip() == x and x not in {"-1", "NoDevFiles"} for x in devices),
        "exactly two assigned CUDA visibility entries required",
    )
    return dict(job_id=job, cuda_visible_devices=visible, cpus=24, memory_mib=16384, world_size=1)


def phase_environment(inherited, label):
    require(label in LABELS, "unknown import condition")
    result = {k: v for k, v in inherited.items() if k not in {"PYTHONHOME", "PYTHONPATH"}}
    result.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    if label == "B":
        result.update(dict.fromkeys(THREAD_KEYS, "12"))
    return result


def worker_argv(plan, work, identity, label):
    return [
        plan["python"],
        "-I",
        "-B",
        "-u",
        "-X",
        "importtime",
        str(Path(plan["release"]) / "source/tools/sdsc_torch_import_probe_worker.py"),
        "--output-json",
        str(work / (label + "-report.json")),
        "--label",
        label,
        "--job-id",
        identity["job_id"],
        "--expected-cuda-visible-devices",
        identity["cuda_visible_devices"],
        "--expected-thread-policy",
        "12" if label == "B" else "inherited",
    ]


def proc_snapshot(pid, label, elapsed, *, proc=Path("/proc")):
    """Read only this launched child's bounded process records; never dump environ."""
    require(type(pid) is int and pid > 0, "invalid owned child pid")
    directory = proc / str(pid)
    record = dict(pid=pid, label=label, elapsed_seconds=elapsed, observed_at_unix=time.time(), files={})
    for name, maximum in (("status", 16384), ("stat", 4096), ("io", 4096), ("wchan", 512)):
        try:
            with (directory / name).open("rb") as stream:
                raw = stream.read(maximum + 1)
            record["files"][name] = dict(
                text=raw[:maximum].decode("utf8", "replace"), truncated=len(raw) > maximum
            )
        except OSError as error:
            record["files"][name] = dict(error=type(error).__name__ + ": " + str(error)[:256])
    try:
        with (directory / "environ").open("rb") as stream:
            raw = stream.read(65537)
        keys = {*THREAD_KEYS, "CUDA_VISIBLE_DEVICES"}
        values = {}
        for item in raw[:65536].split(b"\0"):
            if b"=" in item:
                key, value = item.split(b"=", 1)
                key = key.decode("utf8", "replace")
                if key in keys:
                    values[key] = value.decode("utf8", "replace")[:1024]
        record["environment"] = {key: values.get(key) for key in sorted(keys)}
        record["environment_truncated"] = len(raw) > 65536
    except OSError as error:
        record["environment_error"] = type(error).__name__ + ": " + str(error)[:256]
    try:
        count = 0
        with os.scandir(directory / "task") as entries:
            for _ in entries:
                count += 1
                if count >= 4096:
                    break
        record.update(task_count=count, task_count_truncated=count >= 4096)
    except OSError as error:
        record["task_error"] = type(error).__name__ + ": " + str(error)[:256]
    return record


def stop_group(process, helper, *, deadline):
    """Terminate only the fresh session created for this child, within a finite budget."""
    for number in (signal.SIGTERM, signal.SIGKILL):
        if not helper.group_has_live_members(process.pid):
            process.wait(timeout=max(0.01, min(0.5, deadline - time.monotonic())))
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, number)
        phase_deadline = min(deadline, time.monotonic() + (2 if number == signal.SIGTERM else 3))
        while time.monotonic() < phase_deadline:
            process.poll()
            if not helper.group_has_live_members(process.pid):
                process.wait(timeout=0.5)
                return
            time.sleep(min(0.05, max(0, phase_deadline - time.monotonic())))
    require(not helper.group_has_live_members(process.pid), "owned import process group remains live")
    process.wait(timeout=0.5)


def artifact(path, maximum):
    require(path.is_file() and not path.is_symlink(), "missing or symlinked diagnostic artifact")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, "diagnostic artifact exceeds byte bound")
    return dict(path=path.name, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run_phase(plan, work, identity, label, inherited, control, *, deadline, measure):
    started = time.monotonic()
    budget = min(CHILD_SECONDS, deadline - started - SHUTDOWN_SECONDS)
    require(budget > 0, "allocation worker budget exhausted before next import")
    environment = phase_environment(inherited, label)
    environment["TMPDIR"] = str(work)
    args = worker_argv(plan, work, identity, label)
    phase = dict(
        label=label,
        thread_policy="12" if label == "B" else "inherited",
        thread_environment={key: environment.get(key) for key in THREAD_KEYS},
        cuda_visible_devices=identity["cuda_visible_devices"],
        argv=args,
        cwd=str(work),
        time_limit_seconds=budget,
        started_at_unix=time.time(),
        elapsed_seconds=None,
        pid=None,
        exit_code=None,
        timed_out=False,
        reaped=False,
        shutdown_error=None,
        error=None,
        report_valid=False,
        worker_diagnostic_complete=None,
        import_succeeded=None,
        import_ready=None,
        report=None,
        log=None,
        proc=None,
        observation_complete=False,
    )
    process = None
    selector = selectors.DefaultSelector()
    log, proc_rows = bytearray(), bytearray()
    seen = samples = 0
    proc_truncated = False
    helper = control.helper("sdsc_student_lr_job")
    try:
        process = subprocess.Popen(
            args,
            cwd=work,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        phase["pid"] = process.pid
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        next_sample = started
        while True:
            now = time.monotonic()
            if now >= next_sample:
                row = canonical(proc_snapshot(process.pid, label, now - started)) + b"\n"
                if len(proc_rows) + len(row) <= MAX_PROC:
                    proc_rows.extend(row)
                    samples += 1
                else:
                    proc_truncated = True
                measure("running-" + label + "-" + str(samples))
                next_sample = now + SAMPLE_SECONDS
            for key, _ in selector.select(timeout=min(0.2, max(0, started + budget - now))):
                with contextlib.suppress(BlockingIOError):
                    chunk = os.read(key.fd, 65536)
                    if chunk:
                        seen += len(chunk)
                        log.extend(chunk)
                        if len(log) > MAX_LOG:
                            del log[:-MAX_LOG]
                    else:
                        selector.unregister(key.fileobj)
            if process.poll() is not None:
                break
            if time.monotonic() >= started + budget:
                phase["timed_out"] = True
                break
    except BaseException as error:
        phase["error"] = type(error).__name__ + ": " + str(error)[:1500]
    finally:
        if process is not None:
            try:
                stop_group(process, helper, deadline=min(deadline, time.monotonic() + SHUTDOWN_SECONDS))
                phase["reaped"] = True
            except Exception as error:
                phase["shutdown_error"] = type(error).__name__ + ": " + str(error)[:1500]
            phase["exit_code"] = process.poll()
            # Bounded nonblocking final drain; a leaked writer cannot block publication.
            if process.stdout is not None:
                for _ in range(32):
                    try:
                        chunk = os.read(process.stdout.fileno(), 65536)
                    except (BlockingIOError, OSError):
                        break
                    if not chunk:
                        break
                    seen += len(chunk)
                    log.extend(chunk)
                    if len(log) > MAX_LOG:
                        del log[:-MAX_LOG]
                process.stdout.close()
        selector.close()
        phase["elapsed_seconds"] = time.monotonic() - started
        log_path, proc_path = work / (label + ".log"), work / (label + "-proc.jsonl")
        control.write_once(log_path, bytes(log))
        control.write_once(proc_path, bytes(proc_rows))
        phase["log"] = dict(artifact(log_path, MAX_LOG), bytes_seen=seen, truncated=seen > MAX_LOG)
        phase["proc"] = dict(artifact(proc_path, MAX_PROC), sample_count=samples, truncated=proc_truncated)
    report_path = work / (label + "-report.json")
    if phase["reaped"] and report_path.exists():
        try:
            phase["report"] = artifact(report_path, MAX_REPORT)
            report = json.loads(report_path.read_bytes())
            worker = control.helper("sdsc_torch_import_probe_worker")
            worker.validate_report(
                report,
                label=label,
                job_id=identity["job_id"],
                expected_cuda_visible_devices=identity["cuda_visible_devices"],
                expected_thread_policy=phase["thread_policy"],
            )
            require(
                report["pid"] == phase["pid"] and report["python_executable"] == plan["python"],
                "worker process or Python identity differs",
            )
            require(
                all(report["environment"].get(k) == environment.get(k) for k in THREAD_KEYS),
                "actual thread environment differs from launched condition",
            )
            require(all(worker.launch_checks(report).values()), "worker launch was rejected")
            require(
                report["runtime_actual"].get("python") == worker.EXPECTED_RUNTIME["python"]
                and (not report["import_succeeded"] or report["runtime_actual"] == worker.EXPECTED_RUNTIME)
                and report["runtime_error"] is None,
                "observed worker runtime differs from fixed runtime",
            )
            require(
                report["environment_after"] == report["environment"]
                if report["diagnostic_complete"]
                else report["environment_after"] is None,
                "worker mutated allocation or thread environment",
            )
            phase.update(
                report_valid=True,
                worker_diagnostic_complete=report["diagnostic_complete"],
                import_succeeded=report["import_succeeded"]
                if report["import_elapsed_seconds"] is not None
                else None,
                import_ready=report["import_ready"],
            )
            valid_end = (report["diagnostic_complete"] and phase["exit_code"] == 0) or (
                phase["timed_out"] and report["stack_dump_armed"] and report["import_started"]
            )
            phase["observation_complete"] = bool(
                valid_end and samples > 0 and phase["error"] is None and phase["shutdown_error"] is None
            )
        except Exception as error:
            phase["error"] = type(error).__name__ + ": " + str(error)[:1500]
    else:
        phase["error"] = phase["error"] or "worker report absent or child group not reaped"
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"],
        "node CUDA visibility changed",
    )
    return phase


def persist(plan, control, work, result, observations, memory, measure):
    destination = control.safe(plan["result_dir"])
    records = []
    if work is not None:
        raw = canonical(observations)
        require(len(raw) <= control.MAX_FILE, "aggregate exceeds bound")
        control.write_once(work / "import-observations.json", raw)
        for name in control.NAMES:
            if name in {"node-result.json", "memory.json"} or not (work / name).is_file():
                continue
            # An unreaped child may still mutate its report; never publish it as stable evidence.
            if name.endswith("-report.json") and any(
                row["label"] + "-report.json" == name and not row["reaped"] for row in observations["phases"]
            ):
                continue
            row = artifact(work / name, control.MAX_FILE)
            raw = control.read(work / name)
            control.write_once(destination / name, raw)
            require(control.read(destination / name) == raw, "diagnostic readback differs")
            records.append(row)
    try:
        measure("after_publication")
        control.validate_memory(memory, result["job_id"])
    except Exception as error:
        result.update(
            exit_code=2, diagnostic_complete=False, imports_ready=False, memory_error=str(error)[:1500]
        )
    for name, value in (("node-result.json", result), ("memory.json", memory)):
        raw = canonical(value)
        require(len(raw) <= control.MAX_FILE, "diagnostic metadata exceeds bound")
        control.write_once(destination / name, raw)
        require(control.read(destination / name) == raw, "metadata readback differs")
        records.append(dict(path=name, size=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    require(sum(row["size"] for row in records) <= MAX_TOTAL - control.CAP, "publication exceeds total bound")
    publication = dict(
        task=control.TASK,
        job_id=result["job_id"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        code_sha256=plan["code_sha256"],
        diagnostic_complete=result["diagnostic_complete"],
        imports_ready=result["imports_ready"],
        persistent_read_back_verified=True,
        files=records,
        **FLAGS,
    )
    raw = canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    require(control.read(destination / "receipt.json") == raw, "receipt readback differs")


def main(argv=None):
    started = time.monotonic()
    args = sys.argv[1:] if argv is None else argv
    require(len(args) == 2, "usage: node PLAN PLAN_SHA256")
    control, plan = load_control(*args)
    identity = allocation(plan)
    deadline = started + WORKER_SECONDS
    inherited = dict(os.environ)
    result = dict(
        identity,
        task=control.TASK,
        plan_sha256=control.sha(control.canonical(plan)),
        run_id=plan["run_id"],
        code_sha256=plan["code_sha256"],
        hostname=socket.gethostname(),
        diagnostic_complete=False,
        imports_ready=False,
        exit_code=2,
        **FLAGS,
    )
    observations = dict(
        schema="quest-sdsc-torch-import-observations-v1",
        job_id=identity["job_id"],
        plan_sha256=result["plan_sha256"],
        cuda_visible_devices=identity["cuda_visible_devices"],
        inherited_thread_environment={k: inherited.get(k) for k in THREAD_KEYS},
        cwd=None,
        python=plan["python"],
        diagnostic_complete=False,
        imports_ready=False,
        phases=[],
        sample_interval_seconds=SAMPLE_SECONDS,
        maximum_child_seconds=CHILD_SECONDS,
        worker_budget_seconds=WORKER_SECONDS,
        **FLAGS,
    )
    destination = control.safe(plan["result_dir"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o700)
    work = None
    memory = {}
    frozen_node = control.helper("sdsc_cuda_diagnostic_job")

    def measure(stage):
        try:
            memory[stage] = frozen_node.observe_memory(control, identity["job_id"])
        except frozen_node.MemoryEnvelopeError as error:
            memory[stage] = error.evidence
            raise

    def interrupted(number, _frame):
        raise RuntimeError("import diagnostic interrupted by signal " + str(number))

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        control.verify_source(plan)
        require(time.monotonic() < deadline, "source verification exhausted diagnostic budget")
        result["runtime"] = control.verify_runtime(plan)
        measure("initial")
        helper = control.helper("sdsc_student_lr_job")
        result["persistent_mount"] = helper.mount(destination, control)
        require(
            result["persistent_mount"]["fstype"] in {"lustre", "nfs", "nfs4", "ceph"},
            "persistent GPU-node mount unavailable",
        )
        scratch = control.safe(
            os.environ.get("TMPDIR")
            or "/scratch/{}/job_{}".format(pwd.getpwuid(os.getuid()).pw_name, identity["job_id"])
        )
        require(scratch.is_dir() and scratch.stat().st_uid == os.getuid(), "owned scratch absent")
        result["local_mount"] = helper.mount(scratch, control)
        require(
            result["local_mount"]["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"}
            and shutil.disk_usage(scratch).free >= 64 * control.CAP,
            "small node-local workspace unavailable",
        )
        work = Path(tempfile.mkdtemp(prefix="opd-torch-import-" + identity["job_id"] + "-", dir=scratch))
        observations["cwd"] = result["work_dir"] = str(work)
        for label in LABELS:
            phase = run_phase(
                plan, work, identity, label, inherited, control, deadline=deadline, measure=measure
            )
            observations["phases"].append(phase)
            if phase["shutdown_error"] is not None:
                raise RuntimeError("import child still live; no later condition may launch")
            require(
                phase["observation_complete"], "import instrumentation failed; no later condition may launch"
            )
        observations["diagnostic_complete"] = all(p["observation_complete"] for p in observations["phases"])
        observations["imports_ready"] = all(p["import_ready"] is True for p in observations["phases"])
        for phase in observations["phases"]:
            control.validate_proc_samples(phase, control.read(work / (phase["label"] + "-proc.jsonl")))
        reports = {label: control.document(work / (label + "-report.json")) for label in LABELS}
        control.validate_observations(
            observations, identity["job_id"], identity["cuda_visible_devices"], reports=reports
        )
        result["imports_ready"] = observations["imports_ready"]
        result["diagnostic_complete"] = observations["diagnostic_complete"]
        require(result["diagnostic_complete"], "import instrumentation evidence incomplete")
        result["exit_code"] = 0
    except BaseException as error:
        result.update(
            exit_code=2,
            diagnostic_complete=False,
            imports_ready=False,
            error=type(error).__name__ + ": " + str(error)[:1500],
        )
    finally:
        try:
            measure("final")
        except Exception as error:
            result.update(
                exit_code=2, diagnostic_complete=False, imports_ready=False, memory_error=str(error)[:1500]
            )
        result["elapsed_seconds"] = time.monotonic() - started
        try:
            persist(plan, control, work, result, observations, memory, measure)
        except Exception as error:
            result.update(
                exit_code=2,
                diagnostic_complete=False,
                imports_ready=False,
                publication_error=type(error).__name__ + ": " + str(error)[:1500],
            )
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
