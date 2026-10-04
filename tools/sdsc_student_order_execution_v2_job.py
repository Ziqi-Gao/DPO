#!/usr/bin/env python3
"""Reviewed v2 startup tracing around unchanged V4 science and rank wrappers."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import pwd
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GIB = 1024**3


def load_control(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("execution node plan changed")
    plan = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_order_execution_v2.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != plan["execution"]["control_file_sha256"]["tools/sdsc_student_order_execution_v2.py"]
    ):
        raise ValueError("execution controller changed")
    spec = importlib.util.spec_from_file_location("_order_execution_node_control", source)
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)
    return control, control.validate_plan(plan)


def allocation(plan, control):
    return control.helper("sdsc_student_lr_job").allocation(plan, control)


def mount(path, control):
    return control.helper("sdsc_student_lr_job").mount(path, control)


def memory_envelope(control, job):
    return control.helper("sdsc_student_lr_job").memory_envelope(control, job)


def original_node(control):
    source = Path(__file__).with_name("sdsc_student_order_job.py")
    control.require(
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
        "frozen scientific node differs",
    )
    return control.helper("sdsc_student_order_job")


def persist(*args):
    return original_node(args[1]).persist(*args)


def build_worker_inputs(*args):
    return original_node(args[-1]).build_worker_inputs(*args)


def probe_argv(outer, execution_dir, identity):
    plan = outer["science_plan"]
    return [
        plan["python"],
        "-I",
        "-B",
        "-u",
        "-X",
        "importtime",
        str(Path(plan["release"]) / "source/tools/sdsc_student_order_execution_v2_probe.py"),
        "--output-json",
        str(execution_dir / "early-node-startup.json"),
        "--job-id",
        identity["job_id"],
        "--expected-cuda-visible-devices",
        identity["cuda_visible_devices"],
    ]


def stage_source(plan, destination, control):
    """Explicit manifest-bound staging; never mutate a frozen helper's globals."""
    release = control.safe(plan["release"])
    manifest = control.document(release / "manifest.json", plan["manifest_sha256"])
    control.require(
        manifest["code_sha256"] == plan["code_sha256"] and manifest["run_id"] == plan["run_id"],
        "release staging identity differs",
    )
    records = control.manifest_records(manifest)
    rows = [dict(row, source=str(release / "source" / name)) for name, row in records.items()]
    result = control.helper("sdsc_student_contract").stage_inputs(rows, destination, maximum=64 * 1024**2)
    for name, row in records.items():
        path = destination / name
        path.chmod(0o555 if row["mode"] & 0o111 else 0o444)
    for directory, _, _ in os.walk(destination, topdown=False):
        Path(directory).chmod(0o555)
    return result


def run_probe(outer, control, execution_dir, identity, environment, *, deadline, measure):
    """Keep a finite log tail and timed raw events while supervising only our own child."""
    adapter = control.helper("sdsc_student_order_execution_v2_probe")
    stopper = control.helper("sdsc_torch_import_probe_job")
    groups = control.helper("sdsc_student_lr_job")
    started = time.monotonic()
    limit = min(adapter.PROBE_SECONDS, max(0, deadline - started))
    env = dict(environment)
    env.update(dict.fromkeys(adapter.THREAD_KEYS, "12"))
    argv = probe_argv(outer, execution_dir, identity)
    meta = dict(
        schema=adapter.TRACE_SCHEMA,
        job_id=identity["job_id"],
        cuda_visible_devices=identity["cuda_visible_devices"],
        python=outer["science_plan"]["python"],
        frozen_wrapper_sha256=adapter.FROZEN_SHA,
        adapter_sha256=adapter.file_sha(adapter.__file__),
        argv=argv,
        maximum_probe_seconds=adapter.PROBE_SECONDS,
        stack_interval_seconds=adapter.STACK_SECONDS,
        time_limit_seconds=limit,
        shutdown_reserve_seconds=5,
        elapsed_seconds=0,
        child_environment={key: env.get(key) for key in adapter.ENVIRONMENT_KEYS},
        pid=None,
        exit_code=None,
        timed_out=False,
        reaped=False,
        events_truncated=False,
        probe_passed=False,
        error=None,
        shutdown_error=None,
        adapter_events=[],
        **adapter.FLAGS,
    )
    process = None
    selector = selectors.DefaultSelector()
    retained, pending = bytearray(), bytearray()
    seen, event_bytes = 0, 0

    def consume(chunk):
        nonlocal seen, event_bytes
        seen += len(chunk)
        retained.extend(chunk)
        if len(retained) > adapter.MAX_LOG_BYTES:
            del retained[: -adapter.MAX_LOG_BYTES]
        pending.extend(chunk)
        while b"\n" in pending:
            line, _, rest = pending.partition(b"\n")
            pending[:] = rest
            if len(line) > adapter.MAX_EVENT_BYTES:
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict) or event.get("schema") != adapter.PROGRESS_SCHEMA:
                continue
            size = len(line)
            if len(meta["adapter_events"]) < adapter.MAX_EVENTS and event_bytes + size < 96 * 1024:
                meta["adapter_events"].append(event)
                event_bytes += size
            else:
                meta["events_truncated"] = True
        if len(pending) > adapter.MAX_EVENT_BYTES:
            pending.clear()

    try:
        control.require(limit > 5, "setup exhausted original wall budget before CUDA probe")
        process = subprocess.Popen(
            argv,
            cwd=execution_dir.parent,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        meta["pid"] = process.pid
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        next_sample = started + 30
        while True:
            now = time.monotonic()
            if now >= next_sample:
                measure("cuda-probe-" + str(int(now - started)))
                next_sample = now + 30
            for key, _ in selector.select(timeout=min(0.2, max(0, started + limit - 5 - now))):
                with contextlib.suppress(BlockingIOError):
                    chunk = os.read(key.fd, 65536)
                    if chunk:
                        consume(chunk)
                    else:
                        selector.unregister(key.fileobj)
            if process.poll() is not None:
                break
            if time.monotonic() >= started + limit - 5:
                meta["timed_out"] = True
                break
    except BaseException as error:
        meta["error"] = type(error).__name__ + ": " + str(error)[:1500]
    finally:
        if process is not None:
            try:
                stopper.stop_group(process, groups, deadline=min(deadline, started + limit))
                meta["reaped"] = True
            except Exception as error:
                meta["shutdown_error"] = type(error).__name__ + ": " + str(error)[:1500]
            meta["exit_code"] = process.poll()
            if process.stdout is not None:
                for _ in range(32):
                    try:
                        chunk = os.read(process.stdout.fileno(), 65536)
                    except (BlockingIOError, OSError):
                        break
                    if not chunk:
                        break
                    consume(chunk)
                process.stdout.close()
        selector.close()
        meta["elapsed_seconds"] = time.monotonic() - started
        control.write_once(execution_dir / "startup.log", bytes(retained))
        meta["log"] = dict(
            path="startup.log",
            bytes_seen=seen,
            retained_bytes=len(retained),
            truncated=seen > adapter.MAX_LOG_BYTES,
            sha256=control.sha(bytes(retained)),
        )
    meta["probe_passed"] = bool(
        meta["reaped"]
        and meta["exit_code"] == 0
        and not meta["timed_out"]
        and meta["error"] is None
        and meta["shutdown_error"] is None
        and not meta["events_truncated"]
    )
    try:
        adapter.validate_trace_meta(
            meta,
            job_id=identity["job_id"],
            expected_cuda_visible_devices=identity["cuda_visible_devices"],
            expected_python=outer["science_plan"]["python"],
            expected_argv=argv,
            log_bytes=bytes(retained),
        )
    except Exception as error:
        meta.update(probe_passed=False, error="invalid probe trace: " + str(error)[:1500])
    control.write_once(execution_dir / "trace-meta.json", control.canonical(meta))
    return meta


def worker_argv(outer, source, science, inputs, artifacts, execution_dir, identity):
    plan = outer["science_plan"]
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
        str(source / "tools/sdsc_student_order_execution_worker.py"),
        "run",
        "--execution-output-dir",
        str(execution_dir),
        "--job-id",
        identity["job_id"],
        "--expected-cuda-visible-devices",
        identity["cuda_visible_devices"],
        "--inputs-json",
        str(inputs),
        "--output-dir",
        str(artifacts),
    ]


def start_worker(outer, source, science, inputs, artifacts, execution_dir, identity, environment, stream):
    return subprocess.Popen(
        worker_argv(outer, source, science, inputs, artifacts, execution_dir, identity),
        cwd=science,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def publish_execution(outer, control, source, result):
    destination = control.safe(Path(outer["science_plan"]["result_dir"]) / "execution")
    destination.mkdir(mode=0o700)
    rows, total = [], 0
    for name in control.EXECUTION_NAMES:
        if result.get("early_probe", {}).get("reaped") is False and name not in {
            "execution-node.json",
            "trace-meta.json",
            "startup.log",
        }:
            continue
        if name == "execution-node.json":
            raw = control.canonical(result)
        elif source is not None and (source / name).exists():
            raw = control.read(source / name, 1024**2 if name == "startup.log" else 128 * 1024)
        else:
            continue
        total += len(raw)
        control.require(total <= control.EXECUTION_MAX - 65536, "startup evidence bound exceeded")
        control.write_once(destination / name, raw)
        control.require(control.read(destination / name) == raw, "startup readback differs")
        rows.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    receipt = dict(
        schema="quest-sdsc-student-order-execution-publication-v2",
        task=control.TASK,
        job_id=result["job_id"],
        plan_sha256=control.sha(control.canonical(outer)),
        science_plan_sha256=control.sha(control.canonical(outer["science_plan"])),
        execution_core_sha256=outer["execution"]["core_sha256"],
        startup_passed=result["startup_passed"],
        persistent_read_back_verified=True,
        files=rows,
        **dict.fromkeys(control.FLAGS, False),
    )
    raw = control.canonical(receipt)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "startup receipt readback differs")


def main(argv=None):
    started = time.monotonic()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: node PLAN PLAN_SHA256")
    outer_control, outer = load_control(*args)
    control, plan = outer_control.science, outer["science_plan"]
    identity = allocation(plan, control)
    control.require(
        control.sha(control.read(Path(__file__)))
        == outer["execution"]["control_file_sha256"]["tools/sdsc_student_order_execution_v2_job.py"],
        "node wrapper changed",
    )
    outer_control.verify_source(outer)
    worker_budget = control.worker_budget_seconds(plan)
    control.verify_parent(plan)
    outer_control.verify_preflight(outer, accounting=False)
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
    execution_dir = None
    execution_result = dict(
        job_id=identity["job_id"],
        plan_sha256=control.sha(control.canonical(outer)),
        cuda_visible_devices=identity["cuda_visible_devices"],
        startup_passed=False,
        **dict.fromkeys(control.FLAGS, False),
    )
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
        work = Path(tempfile.mkdtemp(prefix="opd-student-order-" + identity["job_id"] + "-", dir=scratch))
        result.update(work_dir=str(work), local_mount=local_mount)
        execution_dir = work / "execution"
        execution_dir.mkdir()
        execution_result["work_dir"] = str(work)
        environment = {
            key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH")
        }
        environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work))
        trace = run_probe(
            outer,
            outer_control,
            execution_dir,
            identity,
            environment,
            deadline=started + worker_budget,
            measure=measure,
        )
        execution_result["early_probe"] = dict(
            probe_passed=trace["probe_passed"],
            reaped=trace["reaped"],
            timed_out=trace["timed_out"],
            elapsed_seconds=trace["elapsed_seconds"],
        )
        control.require(
            trace["reaped"] and trace["shutdown_error"] is None,
            "early CUDA probe process group not reaped; no large input staged",
        )
        control.require(trace["probe_passed"], "early allocation CUDA probe failed; no large input staged")
        outer_control.validate_startup(
            control.document(execution_dir / "early-node-startup.json"),
            identity["job_id"],
            identity["cuda_visible_devices"],
            "early_node",
        )
        process = None
        measure("after_cuda_probe")
        source, science = work / "source", work / "science"
        staging = control.helper("sdsc_student_job")
        result["source_staging"] = stage_source(plan, source, control)
        result["science_restore"] = outer_control.verify_execution(outer, science)
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
        result["model_staging"] = control.helper("sdsc_student_branch_qualify_job").stage_models(
            Path(plan["hf_home"]), work / "huggingface", control
        )
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
        control.require(started + worker_budget > time.monotonic(), "staging exhausted original wall budget")
        with log.open("xb") as stream:
            process = start_worker(
                outer,
                source,
                science,
                work / "inputs.json",
                artifacts,
                execution_dir,
                identity,
                environment,
                stream,
            )
            deadline = started + worker_budget
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                control.require(remaining > 0, "preparation worker exceeded finite deadline")
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    measure("running-" + str(int(time.monotonic())))
        for rank in (0, 1):
            outer_control.validate_startup(
                control.document(execution_dir / f"rank-{rank}-startup.json"),
                identity["job_id"],
                identity["cuda_visible_devices"],
                "rank_entry",
                rank,
                ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(artifacts)],
            )
        execution_result["startup_passed"] = True
        control.require(process.returncode == 0, "preparation worker failed; preserve outputs, never retry")
        for rank in (0, 1):
            outer_control.validate_rank_exit(
                control.document(execution_dir / f"rank-{rank}-exit.json"),
                identity["job_id"],
                rank,
                ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(artifacts)],
            )
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
                artifacts = None
                execution_result.update(startup_passed=False, worker_shutdown_error=str(error)[:1500])
        try:
            measure("final")
        except Exception as error:
            result.update(stage_complete=False, exit_code=1, memory_error=str(error)[:1500])
        result["elapsed_seconds"] = time.monotonic() - started
        persist(plan, control, artifacts, log, result, memory, measure)
        execution_result.update(
            scientific_stage_complete=result["stage_complete"],
            science_publication_sha256=control.sha(control.read(result_root / "receipt.json")),
            elapsed_seconds=time.monotonic() - started,
        )
        if "error" in result:
            execution_result["error"] = result["error"]
        publish_execution(outer, outer_control, execution_dir, execution_result)
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
