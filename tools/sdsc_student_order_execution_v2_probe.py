#!/usr/bin/env python3
"""Time and trace the frozen CUDA probe without changing its report or checks."""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

FROZEN_NAME = "sdsc_student_order_execution_worker.py"
FROZEN_SHA = "366b606532b6a476cda53dfd847eccc083adf6da226626bc3d196e20f48448e5"
PROGRESS_SCHEMA = "quest-sdsc-student-order-probe-progress-v2"
TRACE_SCHEMA = "quest-sdsc-student-order-probe-trace-v2"
MAX_LOG_BYTES = 1024**2
MAX_META_BYTES = 128 * 1024
MAX_EVENTS = 64
MAX_EVENT_BYTES = 16384
STACK_SECONDS = 30
PROBE_SECONDS = 300
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    *THREAD_KEYS,
    "TMPDIR",
    "SLURM_JOB_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION",
    "CUDA_DEVICE_ORDER",
    "PYTORCH_NVML_BASED_CUDA_CHECK",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
)
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


def environment():
    return {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}


def file_sha(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(2 * MAX_LOG_BYTES + 1)
    if len(raw) > 2 * MAX_LOG_BYTES:
        raise ValueError("probe dependency exceeds bound")
    return hashlib.sha256(raw).hexdigest()


def load_frozen():
    path = Path(__file__).with_name(FROZEN_NAME)
    if file_sha(path) != FROZEN_SHA:
        raise ValueError("frozen v1 CUDA wrapper changed")
    spec = importlib.util.spec_from_file_location("_order_v2_frozen_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
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
        frozen_wrapper_sha256=FROZEN_SHA,
        adapter_sha256=file_sha(__file__),
    )
    code, armed = 1, False
    try:
        if not all(before[key] == "12" for key in THREAD_KEYS):
            raise ValueError("early probe requires the reviewed twelve-thread environment")
        if before["CUDA_VISIBLE_DEVICES"] != args.expected_cuda_visible_devices:
            raise ValueError("early probe CUDA visibility differs")
        frozen = load_frozen()
        faulthandler.enable(file=sys.stderr, all_threads=True)
        faulthandler.dump_traceback_later(STACK_SECONDS, repeat=True, file=sys.stderr, exit=False)
        armed = True
        timed.event("stack_dump_armed", interval_seconds=STACK_SECONDS)
        original_argv = [
            "probe",
            "--output-json",
            str(args.output_json),
            "--job-id",
            args.job_id,
            "--expected-cuda-visible-devices",
            args.expected_cuda_visible_devices,
        ]
        timed.event("frozen_worker_begin", original_argv=original_argv)
        with contextlib.redirect_stdout(timed):
            code = frozen.main(original_argv)
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
    meta, *, job_id, expected_cuda_visible_devices, expected_python, expected_argv=None, log_bytes=None
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
        and meta.get("frozen_wrapper_sha256") == FROZEN_SHA
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
        and entered[0].get("frozen_wrapper_sha256") == FROZEN_SHA
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
            and begun[0].get("original_argv") == ["probe", *meta["argv"][-6:]]
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


if __name__ == "__main__":
    raise SystemExit(main())
