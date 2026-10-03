#!/usr/bin/env python3
"""Observe assigned CUDA devices without training or changing their visibility."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import re
import traceback
from pathlib import Path

SCHEMA = "quest-sdsc-cuda-diagnostic-report-v1"
EXPECTED_RUNTIME = {"python": "3.12.13", "torch": "2.8.0+cu128", "cuda": "12.8"}
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
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_NTASKS",
)
MAX_REPORT_BYTES = 1024**2
MAX_DRIVER_BYTES = 16 * 1024


def error_record(error):
    return {
        "type": type(error).__name__,
        "message": str(error)[:1024],
        "traceback": traceback.format_exc()[-4096:],
    }


def progress(api, event, *, result=None, logical_device=None):
    row = {
        "schema": "quest-sdsc-cuda-diagnostic-progress-v1",
        "api": api,
        "event": event,
        "logical_device": logical_device,
    }
    if result is not None:
        row["result"] = result
    raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw) > 8192:
        row["result"] = {"truncated": True, "preview": str(result)[:2048]}
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print(raw, flush=True)


def capture(function, *, api=None, logical_device=None):
    if api is not None:
        progress(api, "begin", logical_device=logical_device)
    try:
        result = {"ok": True, "value": function()}
    except Exception as error:
        result = {"ok": False, "error": error_record(error)}
    if api is not None:
        progress(api, "end", result=result, logical_device=logical_device)
    return result


def observed(row):
    return row.get("value") if row.get("ok") is True else None


def read_driver_version():
    with Path("/proc/driver/nvidia/version").open("rb") as stream:
        raw = stream.read(MAX_DRIVER_BYTES + 1)
    return {
        "text": raw[:MAX_DRIVER_BYTES].decode("utf-8", errors="replace"),
        "truncated": len(raw) > MAX_DRIVER_BYTES,
    }


def import_torch():
    return importlib.import_module("torch")


def properties(cuda, index):
    value = cuda.get_device_properties(index)
    uuid = getattr(value, "uuid", None)
    return {
        "total_memory_bytes": int(value.total_memory),
        "uuid": str(uuid) if uuid is not None else None,
        "uuid_exposed": uuid is not None,
    }


def tiny_sum(torch, index):
    """Four FP32 values on a logical device; no default-device or CVD mutation."""
    progress("torch.tensor", "begin", logical_device=index)
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=f"cuda:{index}")
    progress("torch.tensor", "end", logical_device=index)
    progress("tensor.sum.item", "begin", logical_device=index)
    value = float(tensor.sum().item())
    progress("tensor.sum.item", "end", logical_device=index)
    progress("cuda.synchronize", "begin", logical_device=index)
    torch.cuda.synchronize(index)
    progress("cuda.synchronize", "end", logical_device=index)
    return {
        "logical_device": index,
        "numel": 4,
        "dtype": "torch.float32",
        "sum": value if math.isfinite(value) else None,
        "finite": math.isfinite(value),
        "expected_sum": 10.0,
        "passed": math.isfinite(value) and value == 10.0,
    }


def diagnose(job_id, expected_cuda_visible_devices):
    before = {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}
    progress("allocation_environment", "observed", result=before)
    report = {
        "schema": SCHEMA,
        "job_id": job_id,
        "cuda_visible_devices": before["CUDA_VISIBLE_DEVICES"],
        "expected_cuda_visible_devices": expected_cuda_visible_devices,
        "diagnostic_complete": False,
        "cuda_ready": False,
        "environment": before,
        "runtime_expected": EXPECTED_RUNTIME,
        "runtime_actual": {"python": platform.python_version()},
        "driver_version": capture(read_driver_version, api="read_driver_version"),
        "devices": [],
        "allocation_attempted": False,
        "tiny_allocations": [],
        **FLAGS,
    }
    visible = expected_cuda_visible_devices.split(",")
    checks = {
        "job_matches": bool(re.fullmatch(r"[1-9][0-9]*", job_id)) and before["SLURM_JOB_ID"] == job_id,
        "expected_visibility_has_two_distinct_ids": len(visible) == 2
        and len(set(visible)) == 2
        and all(part and part.strip() == part for part in visible),
        "visibility_matches": before["CUDA_VISIBLE_DEVICES"] == expected_cuda_visible_devices,
        "allocation_environment_matches": all(
            before[key] == value
            for key, value in {
                "SLURM_CPUS_PER_TASK": "4",
                "SLURM_MEM_PER_NODE": "16384",
                "SLURM_JOB_ACCOUNT": "nwu181",
                "SLURM_JOB_PARTITION": "nairr-gpu-shared",
            }.items()
        ),
        "runtime_matches": False,
        "cuda_available": False,
        "exactly_two_visible_devices": False,
        "both_device_names_are_h100": False,
        "cuda_init_succeeded": False,
        "device_observations_complete": False,
        "tiny_allocations_passed": False,
    }
    admission = (
        "job_matches",
        "expected_visibility_has_two_distinct_ids",
        "visibility_matches",
        "allocation_environment_matches",
    )
    if not all(checks[key] for key in admission):
        report["cuda_observations_skipped_reason"] = "allocation identity or environment differs"
        return finish(report, checks, before, expected_cuda_visible_devices)
    try:
        progress("import_torch", "begin")
        torch = import_torch()
    except Exception as error:
        report["torch_import"] = {"ok": False, "error": error_record(error)}
        progress("import_torch", "end", result=report["torch_import"])
    else:
        report["torch_import"] = {"ok": True}
        progress("import_torch", "end", result=report["torch_import"])
        report["runtime_observations"] = {
            "torch": capture(lambda: str(torch.__version__)),
            "cuda": capture(lambda: torch.version.cuda),
        }
        report["runtime_actual"].update(
            {key: observed(row) for key, row in report["runtime_observations"].items()}
        )
        checks["runtime_matches"] = report["runtime_actual"] == EXPECTED_RUNTIME
        progress("runtime", "observed", result=report["runtime_actual"])
        # Do not short-circuit independent API observations after the first failure.
        report["cuda_is_available"] = capture(lambda: torch.cuda.is_available(), api="cuda.is_available")
        report["cuda_device_count"] = capture(lambda: torch.cuda.device_count(), api="cuda.device_count")
        report["cuda_init"] = capture(lambda: torch.cuda.init(), api="cuda.init")
        checks["cuda_available"] = observed(report["cuda_is_available"]) is True
        count = observed(report["cuda_device_count"])
        checks["exactly_two_visible_devices"] = type(count) is int and count == 2
        report["device_enumeration_permitted"] = checks["exactly_two_visible_devices"]
        if report["device_enumeration_permitted"]:
            for index in range(2):
                report["devices"].append(
                    {
                        "logical_device": index,
                        "name": capture(
                            lambda index=index: torch.cuda.get_device_name(index),
                            api="cuda.get_device_name",
                            logical_device=index,
                        ),
                        "capability": capture(
                            lambda index=index: list(torch.cuda.get_device_capability(index)),
                            api="cuda.get_device_capability",
                            logical_device=index,
                        ),
                        "properties": capture(
                            lambda index=index: properties(torch.cuda, index),
                            api="cuda.get_device_properties",
                            logical_device=index,
                        ),
                    }
                )
        else:
            report["device_enumeration_skipped_reason"] = (
                "visible CUDA count does not match two assigned GPUs"
            )
        checks["both_device_names_are_h100"] = len(report["devices"]) == 2 and all(
            isinstance(observed(row["name"]), str) and "H100" in observed(row["name"])
            for row in report["devices"]
        )
        checks["cuda_init_succeeded"] = report["cuda_init"]["ok"] is True
        checks["device_observations_complete"] = len(report["devices"]) == 2 and all(
            row[key]["ok"] is True
            for row in report["devices"]
            for key in ("name", "capability", "properties")
        )
        checks["environment_unchanged"] = {key: os.environ.get(key) for key in ENVIRONMENT_KEYS} == before
        checks["visibility_unchanged"] = (
            os.environ.get("CUDA_VISIBLE_DEVICES") == expected_cuda_visible_devices
        )
        if all(value for key, value in checks.items() if key != "tiny_allocations_passed"):
            report["allocation_attempted"] = True
            for index in range(2):
                report["tiny_allocations"].append(
                    {
                        "logical_device": index,
                        "result": capture(
                            lambda index=index: tiny_sum(torch, index),
                            api="tiny_allocation",
                            logical_device=index,
                        ),
                    }
                )
            checks["tiny_allocations_passed"] = all(
                row["result"]["ok"] is True and observed(row["result"])["passed"] is True
                for row in report["tiny_allocations"]
            )
    return finish(report, checks, before, expected_cuda_visible_devices)


def finish(report, checks, before, expected_cuda_visible_devices):
    after = {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}
    report["environment_after"] = after
    checks["environment_unchanged"] = after == before
    checks["visibility_unchanged"] = after["CUDA_VISIBLE_DEVICES"] == expected_cuda_visible_devices
    report["checks"] = checks
    report["diagnostic_complete"] = True
    report["cuda_ready"] = all(checks.values())
    return report


def publish(path, report):
    if not path.is_absolute():
        raise ValueError("diagnostic output must be an absolute path")
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("diagnostic report exceeds bounded output")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication without replacing a previous report or following its symlink.
        os.link(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-cuda-visible-devices", required=True)
    args = parser.parse_args(argv)
    report = diagnose(args.job_id, args.expected_cuda_visible_devices)
    raw = publish(args.output_json, report)
    print(raw.decode(), end="", flush=True)
    return 0 if report["diagnostic_complete"] and report["cuda_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
