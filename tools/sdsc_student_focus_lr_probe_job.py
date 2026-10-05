#!/usr/bin/env python3
"""Bounded two-rank paired learning-rate diagnostic with integrated startup and publication."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import pwd
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GIB = 1024**3
LIVE_PROGRESS_MAX_BYTES = 16 * 1024
LIVE_UPDATE_TAIL_BYTES = 64 * 1024


def latest_update(artifacts, maximum_step, control):
    """Observe one complete producer row, without asserting completion or scoring."""
    if artifacts is None:
        return dict(available=False, reason="not_staged")
    path = control.safe(artifacts / "prepare-updates.jsonl")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return dict(available=False, reason="no_update_file")
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        control.require(stat.S_ISREG(info.st_mode), "nonregular update telemetry source")
        control.require(0 <= info.st_size <= control.MAX_FILE, "update telemetry source exceeds bound")
        start = max(0, info.st_size - LIVE_UPDATE_TAIL_BYTES)
        stream.seek(start)
        raw = stream.read(LIVE_UPDATE_TAIL_BYTES)
    partial_last = bool(raw) and not raw.endswith(b"\n")
    lines = raw.split(b"\n")
    if start:
        lines = lines[1:]  # The retained tail may begin in the middle of a row.
    lines = lines[:-1]  # Empty suffix or unfinished last row is never parsed.
    if not lines:
        return dict(available=False, reason="no_complete_update", partial_last_line=partial_last)
    last = lines[-1]
    if not 0 < len(last) <= LIVE_PROGRESS_MAX_BYTES:
        return dict(available=False, reason="complete_update_size", partial_last_line=partial_last)
    try:
        row = json.loads(last)
        step = row["step"]
        if type(step) is not int or not 1 <= step <= maximum_step:
            raise ValueError("invalid step")
        if not isinstance(row["ranks"], list) or len(row["ranks"]) != 2:
            raise ValueError("invalid ranks")
        ranks = []
        for rank, item in enumerate(row["ranks"]):
            metric, budget = item["metric"], item["token_budget"]
            loss, consumed = metric["verified_replay_loss"], budget["consumed"]
            if (
                type(item["rank"]) is not int
                or item["rank"] != rank
                or type(item["step"]) is not int
                or item["step"] != step
                or type(loss) not in (int, float)
                or not math.isfinite(loss)
                or loss < 0
                or type(consumed) is not int
                or not 0 < consumed <= 2000000
                or type(budget["accepted_optimizer_updates"]) is not int
                or budget["accepted_optimizer_updates"] != step
                or metric["model_facing_input_tokens_processed"] != consumed
            ):
                raise ValueError("invalid update observation")
            ranks.append(dict(rank=rank, step=step, loss=loss, cumulative_input_tokens=consumed))
        if ranks[0]["cumulative_input_tokens"] != ranks[1]["cumulative_input_tokens"]:
            raise ValueError("rank token contradiction")
        return dict(
            available=True,
            step=step,
            ranks=ranks,
            partial_last_line=partial_last,
            source_row_sha256=control.sha(last),
        )
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        return dict(available=False, reason="invalid_complete_update", partial_last_line=partial_last)


def publish_live_progress(plan, control, identity, artifacts, stage, elapsed_seconds):
    """Best-effort observational file, intentionally excluded from completion artifacts."""
    control.require(
        stage
        in {"initial", "after_cuda_probe", "after_staging", "after_training", "final", "after_publication"}
        or re.fullmatch(r"(?:running|cuda-probe)-[0-9]+", stage),
        "unreviewed progress phase",
    )
    control.require(
        type(elapsed_seconds) in (int, float) and math.isfinite(elapsed_seconds) and elapsed_seconds >= 0,
        "invalid progress clock",
    )
    phase = "training_or_evaluation" if stage.startswith("running-") else stage
    if stage.startswith("cuda-probe-"):
        phase = "early_cuda_probe"
    value = dict(
        schema="quest-sdsc-student-focus-lr-probe-live-progress-v1",
        scope="progress_only_not_completion",
        task=control.TASK,
        mode=plan["mode"],
        arm=plan["arm"],
        learning_rate=control.ARM_LEARNING_RATES[plan["arm"]],
        job_id=identity["job_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        source_code_sha256=plan["code_sha256"],
        observed_at=control.now(),
        elapsed_seconds=elapsed_seconds,
        phase=phase,
        latest_update=latest_update(artifacts, 32, control),
        completion_claim=False,
        **dict.fromkeys(control.FLAGS, False),
    )
    raw = control.canonical(value)
    control.require(len(raw) <= LIVE_PROGRESS_MAX_BYTES, "progress telemetry exceeds bound")
    destination = control.safe(plan["result_dir"]) / "live-progress.json"
    control.require(not destination.is_symlink(), "unsafe progress output")
    descriptor, temporary = tempfile.mkstemp(prefix=".live-progress-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return value


def load_control(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("execution node plan changed")
    plan = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_focus_lr_probe.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != plan["control_sha256"]["tools/sdsc_student_focus_lr_probe.py"]
    ):
        raise ValueError("execution controller changed")
    spec = importlib.util.spec_from_file_location("_focus_lr_execution_node_control", source)
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


def persist(plan, control, artifacts, log, result, memory, measure):
    destination = control.safe(plan["result_dir"])
    control.require(not (destination / "receipt.json").exists(), "stage already published")
    large = []
    try:
        if artifacts is not None:
            for row in original_node(control).large_inventory(artifacts, control):
                large.append(
                    original_node(control).copy_checkpoint(
                        artifacts / row["path"], destination / row["path"], row, control
                    )
                )
            if result["stage_complete"]:
                report = control.document(artifacts / "prepare-report.json")
                control.validate_large_outputs(report, large)
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
        schema="quest-sdsc-student-focus-lr-probe-publication-v1",
        task=control.TASK,
        arm=plan["arm"],
        diagnostic_complete=result["stage_complete"],
        selected_checkpoint=None,
        mode=plan["mode"],
        job_id=result["job_id"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=result["stage_complete"],
        stage_complete=result["stage_complete"],
        preparation_complete=False,
        persistent_read_back_verified=True,
        large_files_read_back_verified=True,
        files=records,
        large_files=large,
        **dict.fromkeys(control.FLAGS, False),
    )
    raw = control.canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "publication readback differs")


def build_worker_inputs(plan, work, science, identity, original_science, control):
    config = plan["resolved_config"]
    protocol = control.read(science / control.PROTOCOL_PATH)
    return dict(
        schema="quest-sdsc-student-focus-lr-probe-inputs-v1",
        mode=plan["mode"],
        arm=plan["arm"],
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


def probe_argv(plan, execution_dir, identity):
    return [
        plan["python"],
        "-I",
        "-B",
        "-u",
        "-X",
        "importtime",
        str(Path(plan["release"]) / "source/tools/sdsc_student_focus_lr_probe_startup.py"),
        "probe",
        "--worker-sha256",
        plan["control_sha256"]["tools/sdsc_student_focus_lr_probe_worker.py"],
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


def run_probe(plan, control, execution_dir, identity, environment, *, deadline, measure):
    """Keep a finite log tail and timed raw events while supervising only our own child."""
    adapter = control.helper("sdsc_student_focus_lr_probe_startup")
    stopper = control.helper("sdsc_torch_import_probe_job")
    groups = control.helper("sdsc_student_lr_job")
    started = time.monotonic()
    limit = min(adapter.PROBE_SECONDS, max(0, deadline - started))
    env = dict(environment)
    env.update(dict.fromkeys(adapter.THREAD_KEYS, "12"))
    argv = probe_argv(plan, execution_dir, identity)
    meta = dict(
        schema=adapter.TRACE_SCHEMA,
        job_id=identity["job_id"],
        cuda_visible_devices=identity["cuda_visible_devices"],
        python=plan["python"],
        worker_sha256=plan["control_sha256"]["tools/sdsc_student_focus_lr_probe_worker.py"],
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
            worker_sha256=plan["control_sha256"]["tools/sdsc_student_focus_lr_probe_worker.py"],
            job_id=identity["job_id"],
            expected_cuda_visible_devices=identity["cuda_visible_devices"],
            expected_python=plan["python"],
            expected_argv=argv,
            log_bytes=bytes(retained),
        )
    except Exception as error:
        meta.update(probe_passed=False, error="invalid probe trace: " + str(error)[:1500])
    control.write_once(execution_dir / "trace-meta.json", control.canonical(meta))
    return meta


def worker_argv(plan, source, science, inputs, artifacts, execution_dir, identity):
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
        str(source / "tools/sdsc_student_focus_lr_probe_startup.py"),
        "run",
        "--worker-sha256",
        plan["control_sha256"]["tools/sdsc_student_focus_lr_probe_worker.py"],
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


def start_worker(plan, source, science, inputs, artifacts, execution_dir, identity, environment, stream):
    return subprocess.Popen(
        worker_argv(plan, source, science, inputs, artifacts, execution_dir, identity),
        cwd=science,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def validate_startup(plan, control, report, job, visible, scope, rank=None, args=None):
    validated = control.helper("sdsc_student_focus_lr_probe_startup").validate_report(
        report,
        worker_sha256=plan["control_sha256"]["tools/sdsc_student_focus_lr_probe_worker.py"],
        job_id=job,
        expected_cuda_visible_devices=visible,
        scope=scope,
        rank=rank,
        original_argv=args,
    )

    control.require(validated["cuda_ready"] is True, "raw rank CUDA observations failed")
    return validated


def validate_rank_exit(plan, control, value, job, rank, args):
    control.require(
        value.get("schema") == control.helper("sdsc_student_focus_lr_probe_startup").EXIT_SCHEMA
        and value.get("job_id") == job
        and type(value.get("rank")) is int
        and value["rank"] == rank
        and value.get("original_worker_sha256")
        == plan["control_sha256"]["tools/sdsc_student_focus_lr_probe_worker.py"]
        and value.get("original_argv") == args
        and value.get("worker_invoked") is True
        and type(value.get("exit_code")) is int
        and value["exit_code"] == 0
        and "error" not in value
        and all(value.get(k) is False for k in control.FLAGS),
        "diagnostic rank exit failed",
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
        == plan["control_sha256"]["tools/sdsc_student_focus_lr_probe_job.py"],
        "node wrapper changed",
    )
    control.verify_source(plan)
    worker_budget = control.worker_budget_seconds(plan)
    control.verify_parent(plan)
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
    memory = {}
    result = dict(
        identity,
        task=control.TASK,
        mode=plan["mode"],
        arm=plan["arm"],
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
            try:
                publish_live_progress(plan, control, identity, artifacts, stage, time.monotonic() - started)
            except Exception as error:
                # Observability cannot change scientific results or hide memory failure.
                result["live_progress_error"] = type(error).__name__
                print(
                    json.dumps(dict(stage="live_progress_unavailable", error_type=type(error).__name__)),
                    file=sys.stderr,
                    flush=True,
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
        work = Path(tempfile.mkdtemp(prefix="opd-student-focus-lr-" + identity["job_id"] + "-", dir=scratch))
        result.update(work_dir=str(work), local_mount=local_mount)
        artifacts = work / "artifacts"
        artifacts.mkdir()
        execution_dir = artifacts
        environment = {
            key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH")
        }
        environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work))
        trace = run_probe(
            plan,
            control,
            execution_dir,
            identity,
            environment,
            deadline=started + worker_budget,
            measure=measure,
        )
        control.require(
            trace["reaped"] and trace["shutdown_error"] is None,
            "early CUDA probe process group not reaped; no large input staged",
        )
        control.require(trace["probe_passed"], "early allocation CUDA probe failed; no large input staged")
        validate_startup(
            plan,
            control,
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
        result["model_staging"] = control.helper("sdsc_student_branch_qualify_job").stage_models(
            Path(plan["hf_home"]), work / "huggingface", control
        )
        measure("after_staging")
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
                plan,
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
            validate_startup(
                plan,
                control,
                control.document(execution_dir / f"rank-{rank}-startup.json"),
                identity["job_id"],
                identity["cuda_visible_devices"],
                "rank_entry",
                rank,
                ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(artifacts)],
            )
        control.require(process.returncode == 0, "preparation worker failed; preserve outputs, never retry")
        for rank in (0, 1):
            validate_rank_exit(
                plan,
                control,
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
        control.validate_update_records(control.read(artifacts / "prepare-updates.jsonl"), report)
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
