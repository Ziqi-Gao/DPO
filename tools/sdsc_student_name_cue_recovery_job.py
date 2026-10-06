#!/usr/bin/env python3
"""Finite native-runtime recovery; frozen scientific worker and inner plan stay intact."""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import pwd
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

GIB = 1024**3


def load_control(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("outer execution plan changed")
    plan = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_name_cue_recovery.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != plan["execution"]["control_file_sha256"]["tools/sdsc_student_name_cue_recovery.py"]
    ):
        raise ValueError("recovery controller changed")
    spec = importlib.util.spec_from_file_location("_recovery_node_control", source)
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)
    control.validate_plan(plan)
    control.require(
        control.sha(Path(__file__).read_bytes())
        == plan["execution"]["control_file_sha256"]["tools/sdsc_student_name_cue_recovery_job.py"],
        "recovery node changed",
    )
    return control, plan


def audit_module(control):
    return control.helper("sdsc_student_name_cue_recovery_audit")


def context_value(plan, stage, work, identity, control):
    return dict(
        schema="quest-sdsc-student-name-cue-runtime-context-v1",
        plan_sha256=control.sha(control.canonical(plan)),
        science_plan_sha256=control.sha(control.canonical(plan["science_plan"])),
        source_code_sha256=plan["science_plan"]["code_sha256"],
        runtime_stage_sha256=control.sha(control.canonical(stage)),
        job_id=identity["job_id"],
        cuda_visible_devices=identity["cuda_visible_devices"],
        work_dir=str(work),
        source_dir=str(work / "source"),
        science_dir=str(work / "science"),
    )


def file_record(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("native origin is not a regular file")
        digest = hashlib.sha256()
        while chunk := stream.read(1024**2):
            digest.update(chunk)
        after = os.fstat(stream.fileno())

    def stamp(item):
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns, item.st_mode)

    if stamp(before) != stamp(after):
        raise ValueError("native origin changed while hashing")
    return dict(
        path=str(path),
        size=after.st_size,
        sha256=digest.hexdigest(),
        uid=after.st_uid,
        mode=stat.S_IMODE(after.st_mode),
    )


def torch_namespace_source(name, module, modules, stage, manifest):
    """Identify two fileless Torch singleton modules without trusting their display filenames."""
    alias, class_name, placeholder = {
        "torch.classes": ("classes", "_Classes", "_classes.py"),
        "torch.ops": ("ops", "_Ops", "_ops.py"),
    }[name]
    backing_name = "torch._" + alias
    backing, package = modules.get(backing_name), modules.get("torch")

    def require(condition):
        if not condition:
            raise ValueError("invalid Torch namespace identity or backing implementation: " + name)

    require(type(backing) is types.ModuleType and type(package) is types.ModuleType)
    own, defined, root = vars(module), vars(backing), vars(package)
    cls = type(module)
    require(
        cls is defined.get(class_name)
        and cls.__bases__ == (types.ModuleType,)
        and vars(cls).get("__module__") == backing_name
        and vars(cls).get("__file__") == placeholder
        and own.get("__name__") == name
        and "__file__" not in own
        and "__spec__" in own
        and own["__spec__"] is None
        and "__loader__" in own
        and own["__loader__"] is None
        and defined.get(alias) is module
        and root.get(alias) is module
        and defined.get("torch") is package
    )
    initializer = vars(cls).get("__init__")
    require(type(initializer) is types.FunctionType and initializer.__globals__ is defined)
    raw = defined.get("__file__")
    spec = defined.get("__spec__")
    require(
        defined.get("__name__") == backing_name
        and defined.get("__package__") == "torch"
        and isinstance(raw, str)
        and Path(raw).is_absolute()
        and type(spec) is importlib.machinery.ModuleSpec
        and spec.name == backing_name
        and spec.origin == raw
    )
    local = Path(stage["destination"]).resolve(strict=True)
    path = Path(raw).resolve(strict=True)
    relative = "lib/python3.12/site-packages/torch/" + placeholder
    require(path == local / relative)
    package_file = root.get("__file__")
    package_spec = root.get("__spec__")
    require(
        root.get("__name__") == "torch"
        and isinstance(package_file, str)
        and Path(package_file).is_absolute()
        and Path(package_file).resolve(strict=True) == path.parent / "__init__.py"
        and type(package_spec) is importlib.machinery.ModuleSpec
        and package_spec.name == "torch"
        and package_spec.origin == package_file
    )
    expected = [entry for entry in manifest["entries"] if entry["path"] == relative]
    require(len(expected) == 1 and expected[0]["type"] == "file")
    observed = file_record(path)
    require(all(observed[key] == expected[0][key] for key in ("size", "sha256", "mode")))
    return str(path)


def capture_runtime(plan, stage, context, manifest, *, phase, rank, control):
    """Retain raw observations even if validation fails; never claim source-path execution."""
    audit = audit_module(control)
    value = dict(
        schema=audit.RUNTIME_SCHEMA,
        phase=phase,
        rank=rank,
        job_id=context["job_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        context_sha256=control.sha(control.canonical(context)),
        runtime_root_sha256=stage["runtime_root_sha256"],
        executable=sys.executable,
        prefix=sys.prefix,
        base_prefix=sys.base_prefix,
        sys_path=list(sys.path),
        python_version=".".join(map(str, sys.version_info[:3])),
        packages={},
        environment={key: os.environ.get(key) for key in audit.ENV_KEYS},
        dont_write_bytecode=sys.dont_write_bytecode,
        modules=[],
        native_libraries=[],
        files=[],
        passed=False,
        error=None,
        **dict.fromkeys(control.FLAGS, False),
    )
    try:
        value["packages"] = {
            name: importlib.metadata.version(name) for name in control.science.RUNTIME_PACKAGES
        }
        files = {str(Path(sys.executable).resolve(strict=True))}

        def origin(raw):
            if (
                raw is None
                or raw in ("built-in", "frozen")
                or (isinstance(raw, str) and raw.startswith("<") and raw.endswith(">"))
            ):
                return raw
            if not isinstance(raw, str):
                raise ValueError("invalid Python module origin")
            if not Path(raw).is_absolute():
                raise ValueError("relative Python module origin: " + raw)
            path = str(Path(raw).resolve(strict=True))
            files.add(path)
            return path

        modules = dict(sys.modules)
        for name, module in sorted(modules.items()):
            if module is None:
                continue
            if name in ("torch.classes", "torch.ops"):
                files.add(torch_namespace_source(name, module, modules, stage, manifest))
                # These exact singleton instances have no own file or import spec. Their
                # inherited relative __file__ strings describe namespaces, not files.
                value["modules"].append(dict(name=name, file=None, spec_origin=None))
                continue
            value["modules"].append(
                dict(
                    name=name,
                    file=origin(getattr(module, "__file__", None)),
                    spec_origin=origin(getattr(getattr(module, "__spec__", None), "origin", None)),
                )
            )
        libraries = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            raw = fields[5]
            if "x" not in fields[1] and ".so" not in raw:
                continue
            if raw.endswith(" (deleted)"):
                raise ValueError("mapped native library was deleted")
            libraries.add(origin(raw))
        value["native_libraries"] = sorted(libraries)
        value["files"] = [file_record(path) for path in sorted(files)]
        value["passed"] = True
        audit.validate_runtime_origin(plan, stage, manifest, context, value, phase=phase, rank=rank)
    except BaseException as error:
        value.update(passed=False, error=type(error).__name__ + ": " + str(error)[:1500])
    return value


def entry_args(plan, context):
    work = Path(context["work_dir"])
    return [
        "--plan",
        str(work / "execution-plan.json"),
        "--plan-sha256",
        plan_digest(plan),
        "--context",
        str(work / "execution" / "runtime-context.json"),
    ]


def plan_digest(plan):
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def frozen_probe_args(plan, context):
    return [
        "--worker-sha256",
        plan["science_plan"]["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"],
        "--output-json",
        str(Path(context["work_dir"]) / "execution/early-node-startup.json"),
        "--job-id",
        context["job_id"],
        "--expected-cuda-visible-devices",
        context["cuda_visible_devices"],
    ]


def publish_origin(control, path, value):
    raw = control.canonical(value)
    control.require(len(raw) <= 16 * 1024**2, "runtime origins exceed execution bound")
    control.write_once(path, raw)


def probe_argv(plan, context, stage):
    return [
        stage["python"],
        "-I",
        "-B",
        "-u",
        "-X",
        "importtime",
        str(Path(context["source_dir"]) / "tools/sdsc_student_name_cue_recovery_job.py"),
        "runtime-entry",
        "--phase",
        "probe",
        *entry_args(plan, context),
        *frozen_probe_args(plan, context),
    ]


def worker_argv(plan, context, stage):
    return [
        stage["python"],
        "-B",
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(Path(context["source_dir"]) / "tools/sdsc_student_name_cue_recovery_job.py"),
        "runtime-entry",
        "--phase",
        "run",
        *entry_args(plan, context),
    ]


def runtime_entry(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("probe", "run"), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--context", type=Path, required=True)
    for flag in ("--worker-sha256", "--output-json", "--job-id", "--expected-cuda-visible-devices"):
        parser.add_argument(flag)
    args = parser.parse_args(argv)
    control, plan = load_control(args.plan, args.plan_sha256)
    audit, inner = audit_module(control), plan["science_plan"]
    context = control.document(args.context)
    work = Path(context["work_dir"])
    execution = work / "execution"
    control.require(
        args.context == execution / "runtime-context.json"
        and Path(args.plan) == work / "execution-plan.json",
        "runtime entry paths differ",
    )
    stage = control.document(execution / "runtime-stage.json")
    manifest = audit.validate_stage(
        plan, stage, control.read(execution / "runtime-manifest.json", 16 * 1024**2)
    )
    audit.validate_context(plan, context, stage)
    control.require(
        sys.executable == stage["python"]
        and Path(__file__).resolve()
        == Path(context["source_dir"]) / "tools/sdsc_student_name_cue_recovery_job.py",
        "runtime entry not executing relocated/staged bytes",
    )
    control.require(
        sys.dont_write_bytecode and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "runtime bytecode mutation enabled",
    )
    startup = control.science.helper("sdsc_student_name_cue_probe_startup")
    worker_sha = inner["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"]
    if args.phase == "probe":
        control.require(
            [args.worker_sha256, args.output_json, args.job_id, args.expected_cuda_visible_devices]
            == frozen_probe_args(plan, context)[1::2],
            "forwarded frozen probe arguments differ",
        )
    else:
        control.require(
            all(
                getattr(args, key) is None
                for key in ("worker_sha256", "output_json", "job_id", "expected_cuda_visible_devices")
            ),
            "rank entry has probe arguments",
        )
    rank = None if args.phase == "probe" else int(os.environ["RANK"])
    original = ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(work / "artifacts")]
    result = None
    try:
        if args.phase == "probe":
            code = startup.main(["probe", *frozen_probe_args(plan, context)])
            if code:
                return code
        else:
            report = startup.startup("run", context["job_id"], context["cuda_visible_devices"], worker_sha)
            report["original_argv"] = original
            startup.publish(execution / f"rank-{rank}-startup.json", report)
            if not report["cuda_ready"]:
                return 1
        # The original timed probe remains unchanged. These imports and closure hashing
        # are in the same supervised 300-second child, with its own stack diagnostics.
        faulthandler.enable(file=sys.stderr, all_threads=True)
        faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
        for name in ("torch", "numpy", "transformers", "datasets"):
            importlib.import_module(name)
        phase = "early" if args.phase == "probe" else "before"
        raw = capture_runtime(plan, stage, context, manifest, phase=phase, rank=rank, control=control)
        publish_origin(
            control,
            execution / ("runtime-early.json" if rank is None else f"runtime-rank-{rank}-before.json"),
            raw,
        )
        control.require(raw["passed"], "runtime origin proof failed: " + str(raw["error"]))
        if args.phase == "probe":
            return 0
        result = dict(
            schema=startup.EXIT_SCHEMA,
            job_id=context["job_id"],
            rank=rank,
            original_worker_sha256=worker_sha,
            original_argv=original,
            worker_invoked=True,
            exit_code=1,
            **startup.FLAGS,
        )
        faulthandler.cancel_dump_traceback_later()
        try:
            result["exit_code"] = startup.invoke_original(original, worker_sha)
        except BaseException as error:
            result["error"] = startup.error_record(error)
        finally:
            faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
            raw = capture_runtime(plan, stage, context, manifest, phase="after", rank=rank, control=control)
            publish_origin(control, execution / f"runtime-rank-{rank}-after.json", raw)
            if not raw["passed"]:
                result.update(
                    exit_code=1,
                    error=dict(
                        type="ValueError",
                        message="final native runtime origin proof failed: " + str(raw["error"]),
                    ),
                )
            startup.publish(execution / f"rank-{rank}-exit.json", result)
        return result["exit_code"]
    finally:
        faulthandler.cancel_dump_traceback_later()


def stage_runtime(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args(argv)
    control, plan = load_control(args.plan, args.plan_sha256)
    descriptor = plan["runtime_snapshot"]
    snapshot = control.helper("sdsc_runtime_snapshot")
    observed = snapshot.stage(
        Path(descriptor["manifest"]["path"]).parent,
        descriptor["manifest"]["sha256"],
        args.work / "runtime",
        artifact_root=control.RUNTIME_ARTIFACT_ROOT,
        destination_root=args.work,
        deadline=args.deadline,
    )
    control.require(observed["descriptor"] == descriptor, "staged runtime descriptor differs")
    control.write_once(args.work / "execution/runtime-stage.json", control.canonical(observed))
    return 0


def supervise(process, control, *, deadline, measure, label="running"):
    try:
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            control.require(remaining > 5, "execution child exhausted deadline")
            try:
                process.wait(timeout=min(30, remaining - 5))
            except subprocess.TimeoutExpired:
                measure(label + "-" + str(int(time.monotonic())))
        control.require(process.returncode == 0, "execution child failed; preserve evidence, never retry")
    finally:
        control.helper("sdsc_torch_import_probe_job").stop_group(
            process, control.helper("sdsc_student_lr_job"), deadline=deadline
        )


def persist_execution(plan, control, execution, result, memory):
    audit = audit_module(control)
    destination = Path(plan["science_plan"]["result_dir"]) / "execution"
    destination.mkdir(mode=0o700)
    if result["stage_complete"]:
        try:
            evidence = {
                name: control.read(execution / name, audit.MAX_FILE)
                for name in audit.NAMES
                if name not in ("memory.json", "recovery-node-result.json")
            }
            evidence.update(
                {
                    "memory.json": control.canonical(memory),
                    "recovery-node-result.json": control.canonical(result),
                }
            )
            audit.validate_execution_evidence(plan, evidence)
        except Exception as error:
            result.update(
                stage_complete=False,
                exit_code=1,
                evidence_error=type(error).__name__ + ": " + str(error)[:1500],
            )
    control.write_once(execution / "recovery-node-result.json", control.canonical(result))
    control.write_once(execution / "memory.json", control.canonical(memory))
    files = {
        name: control.read(execution / name, audit.MAX_FILE)
        for name in audit.NAMES
        if (execution / name).exists()
    }
    records = [dict(path=name, size=len(raw), sha256=control.sha(raw)) for name, raw in files.items()]
    publication = dict(
        schema=audit.PUBLICATION_SCHEMA,
        task=plan["task"],
        recovery_intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        science_plan_sha256=control.sha(control.canonical(plan["science_plan"])),
        job_id=result["job_id"],
        run_id=plan["science_plan"]["run_id"],
        code_sha256=plan["science_plan"]["code_sha256"],
        execution_complete=result["stage_complete"],
        stage_complete=result["stage_complete"],
        passed=result["stage_complete"],
        scientific_publication_sha256=control.sha(control.read(destination.parent / "receipt.json")),
        persistent_read_back_verified=True,
        files=records,
        large_files=[],
        **dict.fromkeys(control.FLAGS, False),
    )
    audit.validate_execution_publication(plan, publication, files)
    for name, raw in files.items():
        control.write_once(destination / name, raw)
        control.require(
            control.read(destination / name, audit.MAX_FILE) == raw, "execution publication readback differs"
        )
    control.write_once(destination / "receipt.json", control.canonical(publication))
    control.require(
        control.document(destination / "receipt.json") == publication, "execution receipt readback differs"
    )
    return publication


def run_probe(plan, context, stage, control, execution_dir, identity, environment, *, deadline, measure):
    """Keep a finite log tail and timed raw events while supervising only our own child."""
    adapter = control.helper("sdsc_student_name_cue_probe_startup")
    stopper = control.helper("sdsc_torch_import_probe_job")
    groups = control.helper("sdsc_student_lr_job")
    started = time.monotonic()
    limit = min(adapter.PROBE_SECONDS, max(0, deadline - started))
    env = dict(environment)
    env.update(dict.fromkeys(adapter.THREAD_KEYS, "12"))
    argv = probe_argv(plan, context, stage)
    meta = dict(
        schema=adapter.TRACE_SCHEMA,
        job_id=identity["job_id"],
        cuda_visible_devices=identity["cuda_visible_devices"],
        python=stage["python"],
        worker_sha256=plan["science_plan"]["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"],
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
            worker_sha256=plan["science_plan"]["control_sha256"][
                "tools/sdsc_student_name_cue_probe_worker.py"
            ],
            job_id=identity["job_id"],
            expected_cuda_visible_devices=identity["cuda_visible_devices"],
            expected_python=stage["python"],
            expected_argv=argv,
            log_bytes=bytes(retained),
        )
    except Exception as error:
        meta.update(probe_passed=False, error="invalid probe trace: " + str(error)[:1500])
    control.write_once(execution_dir / "trace-meta.json", control.canonical(meta))
    return meta


def node_main(argv):
    started = time.monotonic()
    if len(argv) != 2:
        raise ValueError("usage: recovery-node PLAN PLAN_SHA256")
    control, outer = load_control(*argv)
    plan, science_control = outer["science_plan"], control.science
    original_node = control.helper("sdsc_student_name_cue_probe_job")
    identity = original_node.allocation(plan, science_control)
    worker_budget = control.worker_budget_seconds(outer)
    result_root = science_control.safe(plan["result_dir"])
    parent = result_root.parent
    while not parent.exists():
        parent = parent.parent
    mounts = {
        name: original_node.mount(path, science_control)
        for name, path in (
            ("control_input", Path(plan["parents"]["control"]["receipt"]["result_dir"])),
            ("treatment_input", Path(plan["parents"]["treatment"]["receipt"]["result_dir"])),
            ("output", parent),
            ("cache", Path(plan["hf_home"])),
            ("runtime_archive", Path(outer["runtime_snapshot"]["archive"]["path"]).parent),
        )
    }
    control.require(
        all(v["fstype"] in {"lustre", "nfs", "nfs4", "ceph"} for v in mounts.values())
        and control.PROJECT in result_root.parents,
        "GPU-node persistent mounts unavailable",
    )
    result_root.parent.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(mode=0o700)
    # Keep early failures publishable even before node-local discovery/staging succeeds.
    execution = result_root / ".execution-incomplete"
    execution.mkdir(mode=0o700)
    artifacts = log = process = None
    memory = {}
    result = dict(
        identity,
        task=science_control.TASK,
        mode="probe",
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
        stage_complete=False,
        mounts=mounts,
        **dict.fromkeys(control.FLAGS, False),
    )
    recovery = dict(
        schema="quest-sdsc-student-name-cue-recovery-node-v1",
        job_id=identity["job_id"],
        plan_sha256=control.sha(control.canonical(outer)),
        science_plan_sha256=control.sha(control.canonical(plan)),
        stage_complete=False,
        exit_code=1,
        **dict.fromkeys(control.FLAGS, False),
    )

    def measure(phase):
        try:
            memory[phase] = original_node.memory_envelope(science_control, identity["job_id"])
            raw = control.canonical(memory[phase])
            control.write_once(result_root / ("memory-" + phase + ".json"), raw)
            control.require(
                control.read(result_root / ("memory-" + phase + ".json")) == raw, "memory readback differs"
            )
            original_node.publish_live_progress(
                plan, science_control, identity, artifacts, phase, time.monotonic() - started
            )
        except Exception as error:
            if hasattr(error, "evidence"):
                memory[phase] = error.evidence
            raise

    def interrupted(number, _frame):
        raise RuntimeError("recovery interrupted by signal " + str(number))

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        measure("initial")
        control.verify_source(outer)
        control.verify_failed(outer, accounting=False)
        science_control.verify_parent(plan)
        scratch = control.safe(
            os.environ.get("TMPDIR")
            or "/scratch/{}/job_{}".format(pwd.getpwuid(os.getuid()).pw_name, identity["job_id"])
        )
        control.require(scratch.is_dir() and scratch.stat().st_uid == os.getuid(), "owned scratch absent")
        local_mount = original_node.mount(scratch, science_control)
        descriptor = outer["runtime_snapshot"]
        control.require(
            local_mount["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"}
            and shutil.disk_usage(scratch).free
            >= 192 * GIB + descriptor["archive"]["size"] + descriptor["total_bytes"],
            "node-local free-space gate failed",
        )
        work = Path(tempfile.mkdtemp(prefix="opd-name-cue-recovery-" + identity["job_id"] + "-", dir=scratch))
        result.update(work_dir=str(work), local_mount=local_mount)
        recovery.update(work_dir=str(work), local_mount=local_mount)
        execution = work / "execution"
        execution.mkdir(mode=0o700)
        artifacts = work / "artifacts"
        artifacts.mkdir(mode=0o700)
        control.write_once(work / "execution-plan.json", control.canonical(outer))
        source, science = work / "source", work / "science"
        result["source_staging"] = original_node.stage_source(plan, source, science_control)
        result["science_restore"] = control.verify_execution(outer, science)
        manifest_raw = control.read(descriptor["manifest"]["path"], 16 * 1024**2)
        control.require(
            len(manifest_raw) == descriptor["manifest"]["size"]
            and control.sha(manifest_raw) == descriptor["manifest"]["sha256"],
            "runtime manifest bytes differ",
        )
        control.write_once(execution / "runtime-manifest.json", manifest_raw)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
        env.update(
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            TMPDIR=str(work),
            OMP_NUM_THREADS="12",
            MKL_NUM_THREADS="12",
            OPENBLAS_NUM_THREADS="12",
        )
        log = work / "worker.log"
        with log.open("xb") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-u",
                    str(source / "tools/sdsc_student_name_cue_recovery_job.py"),
                    "stage-runtime",
                    "--plan",
                    str(work / "execution-plan.json"),
                    "--plan-sha256",
                    control.sha(control.canonical(outer)),
                    "--work",
                    str(work),
                    "--deadline",
                    str(started + worker_budget - 5),
                ],
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            supervise(process, control, deadline=started + worker_budget, measure=measure)
        process = None
        stage = control.document(execution / "runtime-stage.json")
        audit = audit_module(control)
        audit.validate_stage(outer, stage, manifest_raw)
        context = context_value(outer, stage, work, identity, control)
        audit.validate_context(outer, context, stage)
        control.write_once(execution / "runtime-context.json", control.canonical(context))
        recovery["effective_python"] = stage["python"]
        result["runtime"] = dict(
            source_python=plan["python"],
            effective_python=stage["python"],
            runtime_stage_sha256=control.sha(control.canonical(stage)),
        )
        trace = run_probe(
            outer,
            context,
            stage,
            control,
            execution,
            identity,
            env,
            deadline=started + worker_budget,
            measure=measure,
        )
        control.require(
            trace["reaped"] and trace["shutdown_error"] is None and trace["probe_passed"],
            "relocated runtime early CUDA/origin probe failed; no large input staged",
        )
        original_node.validate_startup(
            plan,
            science_control,
            control.document(execution / "early-node-startup.json"),
            identity["job_id"],
            identity["cuda_visible_devices"],
            "early_node",
        )
        audit.validate_runtime_origin(
            outer,
            stage,
            json.loads(manifest_raw),
            context,
            control.document(execution / "runtime-early.json"),
            phase="early",
        )
        measure("after_cuda_probe")
        native_plan = control.document(
            plan["parents"]["control"]["plan_path"], plan["parents"]["control"]["plan_sha256"]
        )
        native_receipt = native_plan["parent"]["receipt"]
        proof = control.document(native_receipt["prerequisites_path"], native_receipt["prerequisites_sha256"])
        original = proof["protocol"]["science_file_sha256"]
        control.require(
            len(original) == 49
            and control.sha(control.canonical(original)) == science_control._initial.NAMED_SCIENCE_MAP_SHA,
            "original49 science differs",
        )
        for name, digest in original.items():
            control.require(
                control.helper("sdsc_student_job").file_hash(science / name) == digest,
                "frozen original scientific byte differs",
            )
        contract = control.helper("sdsc_student_contract", source)
        cp_rows = [
            dict(
                row["checkpoint"],
                source=str(control.safe(row["receipt"]["result_dir"]) / row["checkpoint"]["path"]),
                path=arm + ".pt",
            )
            for arm, row in plan["parents"].items()
        ]
        result["checkpoint_staging"] = contract.stage_inputs(cp_rows, work / "checkpoints", maximum=16 * GIB)
        result["config_staging"] = contract.stage_inputs(
            [
                dict(
                    plan["resolved_config"],
                    source=str(control.safe(native_receipt["result_dir"]) / plan["resolved_config"]["path"]),
                    path="resolved_config.yaml",
                )
            ],
            work / "config",
            maximum=control.CAP,
        )
        result["dataset_staging"] = contract.stage_inputs(
            plan["dataset_inputs"], work / "dataset", maximum=2 * GIB
        )
        result["model_staging"] = control.helper("sdsc_student_branch_qualify_job").stage_models(
            Path(plan["hf_home"]), work / "huggingface", science_control
        )
        measure("after_staging")
        inputs = original_node.build_worker_inputs(plan, work, science, identity, original, science_control)
        control.write_once(work / "inputs.json", control.canonical(inputs))
        env.update(
            HF_HOME=str(work / "huggingface"),
            HF_HUB_CACHE=str(work / "huggingface/hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            NUMEXPR_NUM_THREADS="12",
            XDG_CACHE_HOME=str(work / "cache"),
        )
        with log.open("ab") as stream:
            process = subprocess.Popen(
                worker_argv(outer, context, stage),
                cwd=science,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            supervise(process, control, deadline=started + worker_budget, measure=measure)
        process = None
        args = ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(artifacts)]
        for rank in (0, 1):
            original_node.validate_startup(
                plan,
                science_control,
                control.document(execution / f"rank-{rank}-startup.json"),
                identity["job_id"],
                identity["cuda_visible_devices"],
                "rank_entry",
                rank,
                args,
            )
            original_node.validate_rank_exit(
                plan,
                science_control,
                control.document(execution / f"rank-{rank}-exit.json"),
                identity["job_id"],
                rank,
                args,
            )
            for phase in ("before", "after"):
                audit.validate_runtime_origin(
                    outer,
                    stage,
                    json.loads(manifest_raw),
                    context,
                    control.document(execution / f"runtime-rank-{rank}-{phase}.json"),
                    phase=phase,
                    rank=rank,
                )
        measure("after_inference")
        report = control.document(artifacts / "name-cue-report.json")
        rows = {
            name: dict(path=name, size=len(raw), sha256=control.sha(raw))
            for name in science_control.NAMES
            if name != "name-cue-report.json" and (artifacts / name).exists()
            for raw in (science_control.read(artifacts / name),)
        }
        science_control.validate_worker_report(report, plan, identity["job_id"], rows)
        control.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"],
            "CUDA assignment changed",
        )
        control.require(time.monotonic() - started <= worker_budget, "recovery exceeded worker deadline")
        result.update(stage_complete=True, exit_code=0)
        recovery.update(stage_complete=True, exit_code=0)
    except BaseException as error:
        message = type(error).__name__ + ": " + str(error)[:1500]
        result.update(error=message, stage_complete=False, exit_code=1)
        recovery.update(error=message, stage_complete=False, exit_code=1)
    finally:
        if process is not None:
            try:
                control.helper("sdsc_torch_import_probe_job").stop_group(
                    process,
                    control.helper("sdsc_student_lr_job"),
                    deadline=min(started + 5400, time.monotonic() + 5),
                )
            except Exception as error:
                result.update(stage_complete=False, exit_code=1, worker_shutdown_error=str(error)[:1500])
                recovery.update(stage_complete=False, exit_code=1, worker_shutdown_error=str(error)[:1500])
        try:
            measure("final")
        except Exception as error:
            result.update(stage_complete=False, exit_code=1, memory_error=str(error)[:1500])
            recovery.update(stage_complete=False, exit_code=1, memory_error=str(error)[:1500])
        elapsed = time.monotonic() - started
        result["elapsed_seconds"] = recovery["elapsed_seconds"] = elapsed
        # The inner receipt remains exactly the frozen scientific schema. Startup
        # evidence has truthful relocated paths and is separately audited outside it.
        if artifacts is not None:
            for name in science_control.EXECUTION_NAMES:
                if (execution / name).exists():
                    control.write_once(artifacts / name, control.read(execution / name))
        original_node.persist(plan, science_control, artifacts, log, result, memory, measure)
        if not result["stage_complete"]:
            recovery.update(stage_complete=False, exit_code=1)
        persist_execution(outer, control, execution, recovery, memory)
    print(json.dumps(recovery, sort_keys=True))
    return recovery["exit_code"]


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "runtime-entry":
        return runtime_entry(args[1:])
    if args and args[0] == "stage-runtime":
        return stage_runtime(args[1:])
    return node_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
