#!/usr/bin/env python3
"""Import-only Torch diagnostic; the node bounds and reaps this fresh process."""

from __future__ import annotations

import argparse
import faulthandler
import importlib
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

SCHEMA = "quest-sdsc-torch-import-probe-report-v1"
EXPECTED_RUNTIME = {"python": "3.12.13", "torch": "2.8.0+cu128", "cuda": "12.8"}
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "PYTORCH_NVML_BASED_CUDA_CHECK",
    *THREAD_KEYS,
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "KMP_AFFINITY",
    "GOMP_CPU_AFFINITY",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "TMPDIR",
    "SLURM_JOB_ID",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_GPUS",
    "SLURM_GPUS_ON_NODE",
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
        "cuda_observed",
        "model_loaded",
        "training_started",
    ),
    False,
)
STACK_INTERVAL_SECONDS = 30
MAX_REPORT_BYTES = 64 * 1024


def snapshot_environment():
    return {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}


def timestamp():
    return {"monotonic_seconds": time.monotonic(), "utc": datetime.now(UTC).isoformat()}


def error_record(error):
    return {
        "type": type(error).__name__,
        "message": str(error)[:1024],
        "traceback": traceback.format_exc()[-4096:],
    }


def atomic_report(path, report):
    raw = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("import report exceeds fixed byte limit")
    temporary = path.with_name(path.name + ".part")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def mark(report, name, output):
    event = {"phase": name, **timestamp()}
    report["events"].append(event)
    print(json.dumps({"schema": SCHEMA, "label": report["label"], **event}), flush=True)
    if output is not None:
        atomic_report(output, report)


def launch_checks(report):
    before = report["environment"]
    visible = report["expected_cuda_visible_devices"].split(",")
    return {
        "job_matches": bool(re.fullmatch(r"[1-9][0-9]*", report["job_id"]))
        and before["SLURM_JOB_ID"] == report["job_id"],
        "visibility_has_two_distinct_ids": len(visible) == 2
        and len(set(visible)) == 2
        and all(part and part.strip() == part for part in visible),
        "visibility_matches": before["CUDA_VISIBLE_DEVICES"] == report["expected_cuda_visible_devices"],
        "allocation_environment_matches": all(
            before[key] == value
            for key, value in {
                "SLURM_CPUS_PER_TASK": "24",
                "SLURM_MEM_PER_NODE": "16384",
                "SLURM_JOB_ACCOUNT": "nwu181",
                "SLURM_JOB_PARTITION": "nairr-gpu-shared",
            }.items()
        ),
        "thread_policy_matches": report["expected_thread_policy"] == "inherited"
        or all(before[key] == "12" for key in THREAD_KEYS),
    }


def computed_checks(report):
    return {
        **launch_checks(report),
        "environment_preserved": report["environment_after"] == report["environment"],
        "runtime_matches": report["runtime_actual"] == EXPECTED_RUNTIME,
        "import_succeeded": report["import_succeeded"],
        "diagnostic_complete": report["diagnostic_complete"],
    }


def diagnose(*, label, job_id, expected_cuda_visible_devices, expected_thread_policy, output=None):
    if label not in ("A1", "B", "A2") or expected_thread_policy != ("12" if label == "B" else "inherited"):
        raise ValueError("label and planned thread policy differ")
    report = {
        "schema": SCHEMA,
        "label": label,
        "job_id": job_id,
        "pid": os.getpid(),
        "expected_cuda_visible_devices": expected_cuda_visible_devices,
        "expected_thread_policy": expected_thread_policy,
        "runtime_expected": dict(EXPECTED_RUNTIME),
        "runtime_actual": {"python": platform.python_version()},
        "python_executable": sys.executable,
        "torch_file": None,
        "environment": snapshot_environment(),
        "environment_after": None,
        "events": [],
        "stack_dump_interval_seconds": STACK_INTERVAL_SECONDS,
        "stack_dump_armed": False,
        "import_started": False,
        "import_succeeded": False,
        "diagnostic_complete": False,
        "import_ready": False,
        "import_error": None,
        "instrumentation_error": None,
        "runtime_error": None,
        "import_elapsed_seconds": None,
        **FLAGS,
    }
    mark(report, "started", output)
    admission = launch_checks(report)
    if not all(admission.values()):
        report["environment_after"] = snapshot_environment()
        report["checks"] = computed_checks(report)
        mark(report, "launch_rejected", output)
        return report
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
        faulthandler.dump_traceback_later(STACK_INTERVAL_SECONDS, repeat=True, file=sys.stderr, exit=False)
        report["stack_dump_armed"] = True
        mark(report, "stack_dump_armed", output)
    except Exception as error:
        report["instrumentation_error"] = error_record(error)
        report["environment_after"] = snapshot_environment()
        report["checks"] = computed_checks(report)
        mark(report, "instrumentation_failed", output)
        return report
    report["import_started"] = True
    mark(report, "import_torch_begin", output)
    started = time.monotonic()
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        report["import_error"] = error_record(error)
    else:
        report["import_succeeded"] = True
    finally:
        report["import_elapsed_seconds"] = time.monotonic() - started
        mark(report, "import_torch_end", output)
    if report["import_succeeded"]:
        # Metadata only: do not invoke torch.cuda, allocate, or initialize a device.
        try:
            report["torch_file"] = str(torch.__file__)
            report["runtime_actual"].update({"torch": str(torch.__version__), "cuda": torch.version.cuda})
        except Exception as error:
            report["runtime_error"] = error_record(error)
    faulthandler.cancel_dump_traceback_later()
    report["environment_after"] = snapshot_environment()
    report["diagnostic_complete"] = True
    report["checks"] = computed_checks(report)
    report["import_ready"] = all(report["checks"].values())
    mark(report, "completed", output)
    return report


def validate_report(report, *, label, job_id, expected_cuda_visible_devices, expected_thread_policy):
    """Validate complete OR interrupted reports without trusting summary booleans."""
    expected = {
        "schema": SCHEMA,
        "label": label,
        "job_id": job_id,
        "expected_cuda_visible_devices": expected_cuda_visible_devices,
        "expected_thread_policy": expected_thread_policy,
        "runtime_expected": EXPECTED_RUNTIME,
        "stack_dump_interval_seconds": STACK_INTERVAL_SECONDS,
        **FLAGS,
    }
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("import report identity or scientific flags differ")
    if any(report.get(k) is not False for k in FLAGS):
        raise ValueError("scientific flags must be literal false")
    if label not in ("A1", "B", "A2") or expected_thread_policy != ("12" if label == "B" else "inherited"):
        raise ValueError("unplanned import condition")
    required = set(expected) | {
        "pid",
        "runtime_actual",
        "python_executable",
        "torch_file",
        "environment",
        "environment_after",
        "events",
        "stack_dump_armed",
        "import_started",
        "import_succeeded",
        "diagnostic_complete",
        "import_ready",
        "import_error",
        "instrumentation_error",
        "runtime_error",
        "import_elapsed_seconds",
    }
    if not required <= set(report) or set(report) - required - {"checks"}:
        raise ValueError("import report fields differ")
    if not isinstance(report["python_executable"], str) or not report["python_executable"]:
        raise ValueError("invalid Python executable")
    if report["torch_file"] is not None and not isinstance(report["torch_file"], str):
        raise ValueError("invalid Torch module location")
    if type(report["pid"]) is not int or report["pid"] <= 0:
        raise ValueError("invalid worker pid")
    for key in (
        "stack_dump_armed",
        "import_started",
        "import_succeeded",
        "diagnostic_complete",
        "import_ready",
    ):
        if type(report[key]) is not bool:
            raise ValueError("report status must be boolean")
    for key in ("environment", "environment_after"):
        env = report[key]
        if key == "environment_after" and env is None and not report["diagnostic_complete"]:
            continue
        if (
            not isinstance(env, dict)
            or set(env) != set(ENVIRONMENT_KEYS)
            or any(value is not None and not isinstance(value, str) for value in env.values())
        ):
            raise ValueError("environment observation differs")
    actual = report["runtime_actual"]
    if not isinstance(actual, dict) or "python" not in actual or set(actual) - set(EXPECTED_RUNTIME):
        raise ValueError("runtime observation differs")
    if any(value is not None and not isinstance(value, str) for value in actual.values()):
        raise ValueError("runtime metadata must be text")
    events = report["events"]
    allowed = {
        "started",
        "stack_dump_armed",
        "import_torch_begin",
        "import_torch_end",
        "completed",
        "launch_rejected",
        "instrumentation_failed",
    }
    if not isinstance(events, list) or not 1 <= len(events) <= 5:
        raise ValueError("import progress is missing or unbounded")
    previous = -1.0
    for event in events:
        if not isinstance(event, dict) or set(event) != {"phase", "monotonic_seconds", "utc"}:
            raise ValueError("invalid progress observation")
        value = event["monotonic_seconds"]
        if type(value) not in (int, float) or not math.isfinite(value) or value < previous:
            raise ValueError("invalid monotonic progress")
        previous = value
        if event["phase"] not in allowed or not isinstance(event["utc"], str):
            raise ValueError("invalid progress phase")
        offset = datetime.fromisoformat(event["utc"]).utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("progress timestamp must be UTC")
    phases = [event["phase"] for event in events]
    normal = ["started", "stack_dump_armed", "import_torch_begin", "import_torch_end", "completed"]
    if phases != normal[: len(phases)] and phases not in (
        ["started", "launch_rejected"],
        ["started", "instrumentation_failed"],
    ):
        raise ValueError("import phases are reordered or fabricated")
    if report["stack_dump_armed"] != ("stack_dump_armed" in phases):
        raise ValueError("stack instrumentation status differs")
    if report["import_started"] != ("import_torch_begin" in phases):
        raise ValueError("import start status differs")
    if report["diagnostic_complete"] != (phases[-1] == "completed"):
        raise ValueError("completion status differs")
    elapsed = report["import_elapsed_seconds"]
    ended = "import_torch_end" in phases
    if ended:
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid import duration")
        observed = (
            events[phases.index("import_torch_end")]["monotonic_seconds"]
            - events[phases.index("import_torch_begin")]["monotonic_seconds"]
        )
        if elapsed > observed + 0.001:
            raise ValueError("duration exceeds import interval")
    elif elapsed is not None or report["import_succeeded"]:
        raise ValueError("unfinished import has fabricated outcome")
    for key in ("import_error", "instrumentation_error", "runtime_error"):
        row = report[key]
        if row is not None and (
            not isinstance(row, dict)
            or set(row) != {"type", "message", "traceback"}
            or any(not isinstance(value, str) for value in row.values())
            or len(row["message"]) > 1024
            or len(row["traceback"]) > 4096
        ):
            raise ValueError("invalid exception observation")
    if ended and report["import_succeeded"] != (report["import_error"] is None):
        raise ValueError("import outcome and exception disagree")
    if ("instrumentation_failed" in phases) != (report["instrumentation_error"] is not None):
        raise ValueError("instrumentation exception and phase disagree")
    if not ended and any(report[key] is not None for key in ("import_error", "runtime_error", "torch_file")):
        raise ValueError("unfinished import contains later observations")
    if "launch_rejected" in phases and all(launch_checks(report).values()):
        raise ValueError("fabricated launch rejection")
    if report["import_started"] and not all(launch_checks(report).values()):
        raise ValueError("import started despite invalid launch")
    checks = computed_checks(report)
    if "checks" in report and (
        not isinstance(report["checks"], dict)
        or report["checks"] != checks
        or any(type(value) is not bool for value in report["checks"].values())
    ):
        raise ValueError("forged import checks")
    if report["import_ready"] != all(checks.values()):
        raise ValueError("forged import readiness")
    raw = json.dumps(report, allow_nan=False).encode()
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds byte limit")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=("A1", "B", "A2"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-cuda-visible-devices", required=True)
    parser.add_argument("--expected-thread-policy", required=True, choices=("inherited", "12"))
    args = parser.parse_args(argv)
    if args.output_json.exists() or args.output_json.with_name(args.output_json.name + ".part").exists():
        parser.error("output already exists")
    report = diagnose(
        label=args.label,
        job_id=args.job_id,
        expected_cuda_visible_devices=args.expected_cuda_visible_devices,
        expected_thread_policy=args.expected_thread_policy,
        output=args.output_json,
    )
    validate_report(
        report,
        label=args.label,
        job_id=args.job_id,
        expected_cuda_visible_devices=args.expected_cuda_visible_devices,
        expected_thread_policy=args.expected_thread_policy,
    )
    return 0 if report["diagnostic_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
