#!/usr/bin/env python3
"""Observe CUDA startup, then call the byte-pinned original V4 worker unchanged."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import sys
import traceback
from pathlib import Path

SCHEMA = "quest-sdsc-student-order-execution-startup-v1"
EXIT_SCHEMA = "quest-sdsc-student-order-execution-exit-v1"
ORIGINAL = "sdsc_student_order_worker.py"
ORIGINAL_SHA = "e5429b4eff6885513f708207e17a49036a1dcb847f9044231d2ae1160d3c3382"
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
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2:
        raise ValueError("execution dependency exceeds bound")
    return hashlib.sha256(raw).hexdigest()


def load_pinned(name, expected):
    path = Path(__file__).with_name(name)
    if file_sha(path) != expected:
        raise ValueError("execution dependency changed: " + name)
    spec = importlib.util.spec_from_file_location("_order_execution_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_torch():
    return importlib.import_module("torch")


def startup(phase, job_id, expected_visible):
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
        original_worker_sha256=ORIGINAL_SHA,
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
        checks["original_worker_verified"] = report["original_worker_observed_sha256"] == ORIGINAL_SHA
        if not checks["original_worker_verified"]:
            raise ValueError("original V4 worker bytes differ")
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


def invoke_original(argv):
    return load_pinned(ORIGINAL, ORIGINAL_SHA).main(argv)


def validate_report(report, *, job_id, expected_cuda_visible_devices, scope, rank=None, original_argv=None):
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
        and report.get("original_worker_sha256") == ORIGINAL_SHA
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
        and report.get("original_worker_observed_sha256") == ORIGINAL_SHA
        and report.get("diagnostic_helper_observed_sha256") == HELPER_SHA
        and "error" not in report
    )
    require(
        report["cuda_ready"] is ready and all(checks.values()) is ready,
        "startup readiness differs from raw CUDA observations",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in ("probe", "run"):
        sub = commands.add_parser(phase)
        sub.add_argument("--job-id", required=True)
        sub.add_argument("--expected-cuda-visible-devices", required=True)
        if phase == "probe":
            sub.add_argument("--output-json", type=Path, required=True)
        else:
            sub.add_argument("--execution-output-dir", type=Path, required=True)
            sub.add_argument("--inputs-json", required=True)
            sub.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = startup(args.phase, args.job_id, args.expected_cuda_visible_devices)
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
        original_worker_sha256=ORIGINAL_SHA,
        original_argv=original_argv,
        worker_invoked=True,
        exit_code=1,
        **FLAGS,
    )
    try:
        result["exit_code"] = invoke_original(original_argv)
    except BaseException as error:
        result["error"] = error_record(error)
    publish(args.execution_output_dir / f"rank-{rank}-exit.json", result)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
