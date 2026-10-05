#!/usr/bin/env python3
"""Observe diagnostic CUDA startup and invoke only the accepted probe worker."""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "quest-sdsc-student-name-cue-probe-startup-v1"
EXIT_SCHEMA = "quest-sdsc-student-name-cue-probe-exit-v1"
ORIGINAL = "sdsc_student_name_cue_probe_worker.py"
HELPER = "sdsc_cuda_diagnostic_worker.py"
HELPER_SHA = "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
FLAGS = dict.fromkeys(
    (
        "student_accepted",
        "g0_passed",
        "pilot_passed",
        "factorial_ready",
        "formal_initial_accepted",
        "execution_class_certified",
    ),
    False,
)
ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "PYTORCH_NVML_BASED_CUDA_CHECK",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "SLURM_JOB_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
)


def error_record(error):
    return dict(
        type=type(error).__name__, message=str(error)[:1024], traceback=traceback.format_exc()[-4096:]
    )


def publish(path, value):
    if not path.is_absolute():
        raise ValueError("execution evidence path must be absolute")
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > 1024**2:
        raise ValueError("execution evidence exceeds 1 MiB")
    temporary, created = path.with_name(path.name + f".tmp-{os.getpid()}"), False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def file_sha(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise ValueError("execution dependency exceeds bound")
    return hashlib.sha256(raw).hexdigest()


def load_pinned(name, expected):
    path = Path(__file__).with_name(name)
    if file_sha(path) != expected:
        raise ValueError("execution dependency changed: " + name)
    spec = importlib.util.spec_from_file_location("_focus_lr_execution_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_torch():
    return importlib.import_module("torch")


def startup(phase, job_id, expected_visible, worker_sha256):
    if not re.fullmatch(r"[a-f0-9]{64}", worker_sha256):
        raise ValueError("invalid accepted worker digest")
    before = {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}
    rank_text = before["RANK"]
    rank = int(rank_text) if rank_text in ("0", "1") else None
    report = dict(
        schema=SCHEMA,
        phase=phase,
        job_id=job_id,
        rank=rank if phase == "run" else None,
        scope="early_node" if phase == "probe" else "rank_entry",
        cuda_visible_devices=before["CUDA_VISIBLE_DEVICES"],
        expected_cuda_visible_devices=expected_visible,
        diagnostic_complete=False,
        cuda_ready=False,
        environment=before,
        devices=[],
        original_worker_sha256=worker_sha256,
        diagnostic_helper_sha256=HELPER_SHA,
        wrapper_sha256=file_sha(Path(__file__)),
        **FLAGS,
    )
    ids = expected_visible.split(",")
    checks = dict(
        job_matches=bool(re.fullmatch(r"[1-9][0-9]*", job_id)) and before["SLURM_JOB_ID"] == job_id,
        allocation_environment_matches=all(
            before[k] == v
            for k, v in {
                "SLURM_CPUS_PER_TASK": "24",
                "SLURM_MEM_PER_NODE": "393216",
                "SLURM_JOB_ACCOUNT": "nwu181",
                "SLURM_JOB_PARTITION": "nairr-gpu-shared",
                "OMP_NUM_THREADS": "12",
                "MKL_NUM_THREADS": "12",
                "OPENBLAS_NUM_THREADS": "12",
            }.items()
        ),
        visibility_matches=before["CUDA_VISIBLE_DEVICES"] == expected_visible
        and len(ids) == len(set(ids)) == 2
        and all(x and x.strip() == x and x not in ("-1", "NoDevFiles") for x in ids),
        rank_matches=phase == "probe"
        or (
            phase == "run"
            and rank is not None
            and before["LOCAL_RANK"] == rank_text
            and before["WORLD_SIZE"] == "2"
        ),
        original_worker_verified=False,
        diagnostic_helper_verified=False,
        runtime_matches=False,
        cuda_available=False,
        exactly_two_visible_devices=False,
        both_device_names_are_h100=False,
        cuda_init_succeeded=False,
        device_observations_complete=False,
    )
    try:
        if phase not in ("probe", "run") or not all(
            checks[k]
            for k in (
                "job_matches",
                "allocation_environment_matches",
                "visibility_matches",
                "rank_matches",
            )
        ):
            raise ValueError("execution allocation identity differs; CUDA observation skipped")
        report["original_worker_observed_sha256"] = file_sha(Path(__file__).with_name(ORIGINAL))
        checks["original_worker_verified"] = report["original_worker_observed_sha256"] == worker_sha256
        if not checks["original_worker_verified"]:
            raise ValueError("accepted diagnostic worker bytes differ")
        report["diagnostic_helper_observed_sha256"] = file_sha(Path(__file__).with_name(HELPER))
        helper = load_pinned(HELPER, HELPER_SHA)
        checks["diagnostic_helper_verified"] = True
        helper.progress("execution_environment", "observed", result=before)
        report["driver_version"] = helper.capture(helper.read_driver_version, api="read_driver_version")
        helper.progress("import_torch", "begin")
        torch = import_torch()
        helper.progress("import_torch", "end", result=dict(ok=True))
        report["runtime_actual"] = dict(
            python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda
        )
        checks["runtime_matches"] = report["runtime_actual"] == dict(
            python="3.12.13", torch="2.8.0+cu128", cuda="12.8"
        )
        helper.progress("runtime", "observed", result=report["runtime_actual"])
        report["cuda_is_available"] = helper.capture(torch.cuda.is_available, api="cuda.is_available")
        report["cuda_device_count"] = helper.capture(torch.cuda.device_count, api="cuda.device_count")
        report["cuda_init"] = helper.capture(torch.cuda.init, api="cuda.init")
        checks["cuda_available"] = helper.observed(report["cuda_is_available"]) is True
        count = helper.observed(report["cuda_device_count"])
        checks["exactly_two_visible_devices"] = type(count) is int and count == 2
        checks["cuda_init_succeeded"] = report["cuda_init"]["ok"] is True
        if checks["exactly_two_visible_devices"]:
            for index in range(2):
                report["devices"].append(
                    dict(
                        logical_device=index,
                        name=helper.capture(
                            lambda index=index: torch.cuda.get_device_name(index),
                            api="cuda.get_device_name",
                            logical_device=index,
                        ),
                        capability=helper.capture(
                            lambda index=index: list(torch.cuda.get_device_capability(index)),
                            api="cuda.get_device_capability",
                            logical_device=index,
                        ),
                        properties=helper.capture(
                            lambda index=index: helper.properties(torch.cuda, index),
                            api="cuda.get_device_properties",
                            logical_device=index,
                        ),
                    )
                )
        else:
            report["device_enumeration_skipped_reason"] = "visible count differs from two assigned GPUs"
        checks["both_device_names_are_h100"] = len(report["devices"]) == 2 and all(
            isinstance(helper.observed(row["name"]), str) and "H100" in helper.observed(row["name"])
            for row in report["devices"]
        )
        checks["device_observations_complete"] = len(report["devices"]) == 2 and all(
            row[key]["ok"] is True
            for row in report["devices"]
            for key in ("name", "capability", "properties")
        )
    except Exception as error:
        report["error"] = error_record(error)
    report["environment_after"] = {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}
    checks["environment_unchanged"] = report["environment_after"] == before
    checks["visibility_unchanged"] = report["environment_after"]["CUDA_VISIBLE_DEVICES"] == expected_visible
    report.update(checks=checks, diagnostic_complete=True, cuda_ready=all(checks.values()))
    return report


def invoke_original(argv, worker_sha256):
    return load_pinned(ORIGINAL, worker_sha256).main(argv)


def validate_report(
    report, *, worker_sha256, job_id, expected_cuda_visible_devices, scope, rank=None, original_argv=None
):
    """Recompute CUDA admission from recorded raw observations, never only check flags."""

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(scope in ("early_node", "rank_entry"), "unknown startup scope")
    require(
        report.get("schema") == SCHEMA
        and report.get("scope") == scope
        and report.get("phase") == ("probe" if scope == "early_node" else "run")
        and report.get("job_id") == job_id
        and report.get("rank") == rank
        and (report.get("rank") is None if scope == "early_node" else type(report.get("rank")) is int)
        and (rank is None if scope == "early_node" else type(rank) is int and rank in (0, 1))
        and report.get("expected_cuda_visible_devices") == expected_cuda_visible_devices
        and report.get("original_worker_sha256") == worker_sha256
        and report.get("diagnostic_helper_sha256") == HELPER_SHA
        and report.get("wrapper_sha256") == file_sha(Path(__file__))
        and report.get("diagnostic_complete") is True
        and type(report.get("cuda_ready")) is bool
        and all(report.get(key) is False for key in FLAGS),
        "startup identity or scope differs",
    )
    if original_argv is not None:
        require(
            scope == "rank_entry" and report.get("original_argv") == original_argv,
            "original worker argv differs",
        )
    checks = report.get("checks", {})
    require(
        isinstance(checks, dict)
        and set(checks)
        == {
            "job_matches",
            "allocation_environment_matches",
            "visibility_matches",
            "rank_matches",
            "original_worker_verified",
            "diagnostic_helper_verified",
            "runtime_matches",
            "cuda_available",
            "exactly_two_visible_devices",
            "both_device_names_are_h100",
            "cuda_init_succeeded",
            "device_observations_complete",
            "environment_unchanged",
            "visibility_unchanged",
        }
        and all(type(value) is bool for value in checks.values()),
        "startup check types differ",
    )
    before, after = report.get("environment", {}), report.get("environment_after", {})
    require(set(before) == set(after) == set(ENVIRONMENT_KEYS), "startup environment fields differ")
    require(
        all(value is None or isinstance(value, str) for row in (before, after) for value in row.values())
        and report.get("cuda_visible_devices") == before["CUDA_VISIBLE_DEVICES"],
        "startup environment observation types differ",
    )
    ids = expected_cuda_visible_devices.split(",")
    allocated = (
        before.get("SLURM_JOB_ID") == job_id
        and bool(re.fullmatch(r"[1-9][0-9]*", job_id))
        and len(ids) == len(set(ids)) == 2
        and all(x and x.strip() == x and x not in ("-1", "NoDevFiles") for x in ids)
        and report.get("cuda_visible_devices")
        == report.get("expected_cuda_visible_devices")
        == expected_cuda_visible_devices
        and before.get("CUDA_VISIBLE_DEVICES") == expected_cuda_visible_devices
        and before == after
        and all(
            before.get(k) == v
            for k, v in {
                "SLURM_CPUS_PER_TASK": "24",
                "SLURM_MEM_PER_NODE": "393216",
                "SLURM_JOB_ACCOUNT": "nwu181",
                "SLURM_JOB_PARTITION": "nairr-gpu-shared",
                "OMP_NUM_THREADS": "12",
                "MKL_NUM_THREADS": "12",
                "OPENBLAS_NUM_THREADS": "12",
            }.items()
        )
        and (
            scope == "early_node"
            or (
                before.get("RANK") == before.get("LOCAL_RANK") == str(rank)
                and before.get("WORLD_SIZE") == "2"
            )
        )
    )
    available, count, initialization = (
        report.get(key, {}) for key in ("cuda_is_available", "cuda_device_count", "cuda_init")
    )
    devices = report.get("devices", [])
    inventory = isinstance(devices, list) and len(devices) == 2
    if inventory:
        for index, device in enumerate(devices):
            name, capability, properties = (
                device.get(key, {}) for key in ("name", "capability", "properties")
            )
            inventory = inventory and (
                type(device.get("logical_device")) is int
                and device["logical_device"] == index
                and name.get("ok") is True
                and isinstance(name.get("value"), str)
                and "H100" in name["value"]
                and capability.get("ok") is True
                and properties.get("ok") is True
            )
    ready = bool(
        allocated
        and available.get("ok") is True
        and available.get("value") is True
        and count.get("ok") is True
        and type(count.get("value")) is int
        and count["value"] == 2
        and initialization.get("ok") is True
        and initialization.get("value") is None
        and inventory
        and report.get("runtime_actual") == dict(python="3.12.13", torch="2.8.0+cu128", cuda="12.8")
        and report.get("original_worker_observed_sha256") == worker_sha256
        and report.get("diagnostic_helper_observed_sha256") == HELPER_SHA
        and "error" not in report
    )
    require(
        report["cuda_ready"] is ready and all(checks.values()) is ready,
        "startup readiness differs from raw CUDA observations",
    )
    return report


def worker_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in ("probe", "run"):
        sub = commands.add_parser(phase)
        sub.add_argument("--job-id", required=True)
        sub.add_argument("--worker-sha256", required=True)
        sub.add_argument("--expected-cuda-visible-devices", required=True)
        if phase == "probe":
            sub.add_argument("--output-json", type=Path, required=True)
        else:
            sub.add_argument("--execution-output-dir", type=Path, required=True)
            sub.add_argument("--inputs-json", required=True)
            sub.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    worker_sha256 = args.worker_sha256
    report = startup(args.phase, args.job_id, args.expected_cuda_visible_devices, worker_sha256)
    original_argv = (
        ["--inputs-json", args.inputs_json, "--output-dir", args.output_dir] if args.phase == "run" else None
    )
    report["original_argv"] = original_argv
    rank = os.environ.get("RANK", "invalid")
    label = rank if rank in ("0", "1") else "invalid"
    path = (
        args.output_json
        if args.phase == "probe"
        else args.execution_output_dir / f"rank-{label}-startup.json"
    )
    publish(path, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["cuda_ready"]:
        return 1
    if args.phase == "probe":
        return 0
    result = dict(
        schema=EXIT_SCHEMA,
        job_id=args.job_id,
        rank=int(rank),
        original_worker_sha256=worker_sha256,
        original_argv=original_argv,
        worker_invoked=True,
        exit_code=1,
        **FLAGS,
    )
    try:
        result["exit_code"] = invoke_original(original_argv, worker_sha256)
    except BaseException as error:
        result["error"] = error_record(error)
    publish(args.execution_output_dir / f"rank-{rank}-exit.json", result)
    return result["exit_code"]


PROGRESS_SCHEMA = "quest-sdsc-student-name-cue-probe-progress-v1"
TRACE_SCHEMA = "quest-sdsc-student-name-cue-probe-trace-v1"
MAX_LOG_BYTES = 1024**2
MAX_META_BYTES = 128 * 1024
MAX_EVENTS = 64
MAX_EVENT_BYTES = 16384
STACK_SECONDS = 30
PROBE_SECONDS = 300
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
ENVIRONMENT_KEYS = tuple(dict.fromkeys((*ENVIRONMENT_KEYS, *THREAD_KEYS, "TMPDIR")))


def environment():
    return {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}


class TimedOutput:
    """Wrap only standard output; preserve each original progress line as raw text."""

    def __init__(self, stream, started):
        self.stream, self.started, self.pending = stream, started, ""

    def event(self, event, **fields):
        now = time.monotonic()
        row = dict(
            schema=PROGRESS_SCHEMA,
            event=event,
            monotonic_seconds=now,
            elapsed_seconds=now - self.started,
            utc=datetime.now(UTC).isoformat(),
            pid=os.getpid(),
            **fields,
        )
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(raw.encode()) > MAX_EVENT_BYTES:
            row = dict(
                schema=PROGRESS_SCHEMA,
                event="output_truncated",
                monotonic_seconds=now,
                elapsed_seconds=now - self.started,
                utc=datetime.now(UTC).isoformat(),
                pid=os.getpid(),
                original_event=event,
            )
            raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.stream.write(raw + "\n")
        self.stream.flush()

    def write(self, value):
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.event("frozen_progress", raw=line)
        if len(self.pending.encode()) > MAX_EVENT_BYTES:
            self.event("output_truncated", original_event="unfinished_frozen_progress")
            self.pending = ""
        return len(value)

    def flush(self):
        self.stream.flush()

    def finish(self):
        if self.pending:
            self.event("frozen_progress", raw=self.pending)
            self.pending = ""


def probe_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--expected-cuda-visible-devices", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    timed = TimedOutput(sys.stdout, started)
    before = environment()
    timed.event(
        "adapter_started",
        job_id=args.job_id,
        python=sys.executable,
        environment=before,
        worker_sha256=args.worker_sha256,
        adapter_sha256=file_sha(__file__),
    )
    code, armed = 1, False
    try:
        if not all(before[key] == "12" for key in THREAD_KEYS):
            raise ValueError("early probe requires the reviewed twelve-thread environment")
        if before["CUDA_VISIBLE_DEVICES"] != args.expected_cuda_visible_devices:
            raise ValueError("early probe CUDA visibility differs")
        faulthandler.enable(file=sys.stderr, all_threads=True)
        faulthandler.dump_traceback_later(STACK_SECONDS, repeat=True, file=sys.stderr, exit=False)
        armed = True
        timed.event("stack_dump_armed", interval_seconds=STACK_SECONDS)
        original_argv = [
            "probe",
            "--worker-sha256",
            args.worker_sha256,
            "--output-json",
            str(args.output_json),
            "--job-id",
            args.job_id,
            "--expected-cuda-visible-devices",
            args.expected_cuda_visible_devices,
        ]
        timed.event("frozen_worker_begin", original_argv=original_argv)
        with contextlib.redirect_stdout(timed):
            code = worker_main(original_argv)
        timed.finish()
        timed.event("frozen_worker_end", returncode=code)
        if type(code) is not int or code not in (0, 1):
            raise ValueError("unexpected frozen probe exit code")
    except BaseException as error:
        code = 1
        timed.event("adapter_error", error_type=type(error).__name__, error=str(error)[:1024])
    finally:
        if armed:
            faulthandler.cancel_dump_traceback_later()
        after = environment()
        if after != before:
            code = 1
            timed.event("adapter_error", error_type="ValueError", error="probe environment changed")
        timed.event("adapter_finished", returncode=code, environment=after)
    return code


def validate_trace_meta(
    meta,
    *,
    worker_sha256,
    job_id,
    expected_cuda_visible_devices,
    expected_python,
    expected_argv=None,
    log_bytes=None,
):
    """Accept bounded negative evidence; a successful probe additionally needs the raw v1 report."""

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(
        isinstance(meta, dict)
        and meta.get("schema") == TRACE_SCHEMA
        and meta.get("job_id") == job_id
        and re.fullmatch(r"[1-9][0-9]*", job_id)
        and meta.get("cuda_visible_devices") == expected_cuda_visible_devices
        and meta.get("python") == expected_python
        and meta.get("worker_sha256") == worker_sha256
        and meta.get("adapter_sha256") == file_sha(__file__)
        and all(meta.get(key) is False for key in FLAGS),
        "probe trace identity differs",
    )
    require(
        meta.get("maximum_probe_seconds") == PROBE_SECONDS
        and meta.get("stack_interval_seconds") == STACK_SECONDS,
        "probe trace fixed limits differ",
    )
    require(expected_argv is None or meta.get("argv") == expected_argv, "probe argv differs")
    require(
        isinstance(meta.get("argv"), list)
        and meta["argv"][:6] == [expected_python, "-I", "-B", "-u", "-X", "importtime"],
        "probe isolation/importtime differs",
    )
    for key in ("time_limit_seconds", "elapsed_seconds"):
        require(
            type(meta.get(key)) in (int, float) and math.isfinite(meta[key]) and meta[key] >= 0,
            "probe timing differs",
        )
    require(meta["time_limit_seconds"] <= PROBE_SECONDS, "probe exceeds reviewed limit")
    for key in ("timed_out", "reaped", "events_truncated", "probe_passed"):
        require(type(meta.get(key)) is bool, "probe boolean type differs")
    require(meta.get("pid") is None or (type(meta["pid"]) is int and meta["pid"] > 0), "invalid probe pid")
    require(meta.get("exit_code") is None or type(meta["exit_code"]) is int, "invalid probe exit code")
    for key in ("error", "shutdown_error"):
        require(meta.get(key) is None or isinstance(meta[key], str), "invalid probe error")
    env = meta.get("child_environment")
    require(
        isinstance(env, dict)
        and set(env) == set(ENVIRONMENT_KEYS)
        and all(value is None or isinstance(value, str) for value in env.values()),
        "probe environment fields differ",
    )
    require(
        env["CUDA_VISIBLE_DEVICES"] == expected_cuda_visible_devices
        and all(env[key] == "12" for key in THREAD_KEYS),
        "probe environment policy differs",
    )
    log = meta.get("log", {})
    require(
        isinstance(log, dict)
        and log.get("path") == "startup.log"
        and type(log.get("bytes_seen")) is int
        and log["bytes_seen"] >= 0
        and type(log.get("retained_bytes")) is int
        and log["retained_bytes"] == min(log["bytes_seen"], MAX_LOG_BYTES)
        and log.get("truncated") is (log["bytes_seen"] > MAX_LOG_BYTES)
        and isinstance(log.get("sha256"), str)
        and re.fullmatch(r"[a-f0-9]{64}", log["sha256"]),
        "probe log bounds differ",
    )
    if log_bytes is not None:
        require(
            len(log_bytes) == log["retained_bytes"]
            and hashlib.sha256(log_bytes).hexdigest() == log["sha256"],
            "probe log bytes differ",
        )
    events = meta.get("adapter_events")
    require(isinstance(events, list) and len(events) <= MAX_EVENTS, "probe events unbounded")
    if log_bytes is not None:
        lines = log_bytes.splitlines()
        # A retained tail may start in the middle of one original stdout line.
        if log["truncated"] and lines:
            lines = lines[1:]
        logged_events = []
        for line in lines:
            if len(line) > MAX_EVENT_BYTES:
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(row, dict) and row.get("schema") == PROGRESS_SCHEMA:
                logged_events.append(row)
        if not log["truncated"]:
            expected_logged = logged_events[: len(events)] if meta["events_truncated"] else logged_events
            require(events == expected_logged, "probe events differ from retained raw log")
        elif not meta["events_truncated"] and logged_events:
            require(
                events[-len(logged_events) :] == logged_events,
                "probe event suffix differs from retained raw log",
            )
    previous, previous_elapsed = -1.0, 0.0
    clock_origin = None
    for event in events:
        require(
            isinstance(event, dict)
            and event.get("schema") == PROGRESS_SCHEMA
            and event.get("pid") == meta["pid"]
            and isinstance(event.get("event"), str)
            and type(event.get("monotonic_seconds")) in (int, float)
            and math.isfinite(event["monotonic_seconds"])
            and event["monotonic_seconds"] >= previous
            and type(event.get("elapsed_seconds")) in (int, float)
            and math.isfinite(event["elapsed_seconds"])
            and previous_elapsed <= event["elapsed_seconds"] <= meta["elapsed_seconds"] + 1,
            "invalid timed probe event",
        )
        origin = event["monotonic_seconds"] - event["elapsed_seconds"]
        require(
            clock_origin is None or abs(origin - clock_origin) <= 1e-6, "probe clocks have different origins"
        )
        if clock_origin is None:
            clock_origin = origin
        previous = event["monotonic_seconds"]
        previous_elapsed = event["elapsed_seconds"]
        offset = datetime.fromisoformat(event["utc"]).utcoffset()
        require(offset is not None and offset.total_seconds() == 0, "probe timestamp is not UTC")
        require(len(json.dumps(event).encode()) <= MAX_EVENT_BYTES * 2, "probe event oversized")
    names = [event["event"] for event in events]
    entered = [event for event in events if event["event"] == "adapter_started"]
    finished = [event for event in events if event["event"] == "adapter_finished"]
    begun = [event for event in events if event["event"] == "frozen_worker_begin"]
    ended = [event for event in events if event["event"] == "frozen_worker_end"]
    armed = [event for event in events if event["event"] == "stack_dump_armed"]
    start_ok = (
        len(entered) == 1
        and entered[0].get("environment") == env
        and entered[0].get("python") == expected_python
        and entered[0].get("job_id") == job_id
        and entered[0].get("adapter_sha256") == meta["adapter_sha256"]
        and entered[0].get("worker_sha256") == worker_sha256
    )
    end_ok = (
        len(finished) == len(ended) == 1
        and finished[0].get("environment") == env
        and type(finished[0].get("returncode")) is int
        and finished[0]["returncode"] == 0
        and type(ended[0].get("returncode")) is int
        and ended[0]["returncode"] == 0
    )
    ordered = (
        bool(
            names
            and names[0] == "adapter_started"
            and names[-1] == "adapter_finished"
            and len(begun) == 1
            and begun[0].get("original_argv") == ["probe", *meta["argv"][-8:]]
            and len(armed) == 1
            and armed[0].get("interval_seconds") == STACK_SECONDS
            and names.index("stack_dump_armed")
            < names.index("frozen_worker_begin")
            < names.index("frozen_worker_end")
            < len(names) - 1
        )
        if end_ok
        else False
    )
    allocation_ok = all(
        env.get(key) == value
        for key, value in {
            "SLURM_JOB_ID": job_id,
            "SLURM_CPUS_PER_TASK": "24",
            "SLURM_MEM_PER_NODE": "393216",
            "SLURM_JOB_ACCOUNT": "nwu181",
            "SLURM_JOB_PARTITION": "nairr-gpu-shared",
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "12",
            "OPENBLAS_NUM_THREADS": "12",
        }.items()
    )
    passed = bool(
        allocation_ok
        and meta["time_limit_seconds"] > 5
        and meta["elapsed_seconds"] <= meta["time_limit_seconds"] + 1
        and meta["reaped"]
        and meta["exit_code"] == 0
        and not meta["timed_out"]
        and meta["error"] is None
        and meta["shutdown_error"] is None
        and not meta["events_truncated"]
        and start_ok
        and end_ok
        and ordered
        and "adapter_error" not in names
        and "output_truncated" not in names
    )
    require(meta["probe_passed"] is passed, "probe pass differs from timed raw evidence")
    require(len(json.dumps(meta).encode()) <= MAX_META_BYTES, "probe metadata oversized")
    return meta


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "probe":
        return probe_main(args[1:])
    return worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
