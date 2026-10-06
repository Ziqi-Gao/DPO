#!/usr/bin/env python3
"""Independent native-runtime and execution evidence audit; no scientific acceptance."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import re
import stat
import sys
import types
from pathlib import Path

CAP = 1024**2
MAX_FILE = 16 * CAP
MAX_TOTAL = 64 * CAP
RUNTIME_SCHEMA = "quest-sdsc-student-name-invariant-native-origin-v1"
CONTEXT_SCHEMA = "quest-sdsc-student-name-invariant-runtime-context-v1"
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)
NAMES = (
    "runtime-stage.json",
    "runtime-manifest.json",
    "runtime-context.json",
    "runtime-early.json",
    *(f"runtime-rank-{rank}-{phase}.json" for rank in (0, 1) for phase in ("before", "after")),
)
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TMPDIR",
)
SYSTEM_ROOTS = ("/usr/lib", "/usr/lib64", "/lib", "/lib64")
SDSC_DRIVER_LIBRARIES = (
    "libcuda",
    "libnvidia-ml",
    "libnvidia-ptxjitcompiler",
    "libnvidia-nvvm",
    "libnvidia-allocator",
    "libnvidia-compiler",
    "libcudadebugger",
)
SDSC_DRIVER_PATH = re.compile(
    r"/cm/local/apps/cuda-driver/libs/(?P<version>[0-9]+(?:\.[0-9]+)+)/lib64/(?:"
    + "|".join(re.escape(name) for name in SDSC_DRIVER_LIBRARIES)
    + r")\.so\.(?P=version)"
)


def is_host_native_library(row, native_libraries):
    """Recognize root-managed host libraries, including SDSC's observed compute driver layout."""
    raw, uid, mode = row.get("path"), row.get("uid"), row.get("mode")
    if not (
        isinstance(raw, str)
        and Path(raw).is_absolute()
        and str(Path(raw)) == raw
        and ".." not in Path(raw).parts
        and type(uid) is int
        and uid == 0
        and type(mode) is int
        and not mode & 0o022
    ):
        return False
    return any(Path(raw).is_relative_to(Path(root)) for root in SYSTEM_ROOTS) or bool(
        SDSC_DRIVER_PATH.fullmatch(raw) and isinstance(native_libraries, list) and raw in native_libraries
    )


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def helper(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("_name_invariant_native_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("nonfinite JSON number: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def bound_helper(plan, name):
    relative = "tools/" + name + ".py"
    expected = plan["control_sha256"].get(relative)
    require(
        isinstance(expected, str)
        and plan["protocol"]["science_file_sha256"].get(relative) == expected
        and sha(Path(__file__).with_name(name + ".py").read_bytes()) == expected,
        "native dependency is not the accepted pinned source: " + relative,
    )
    return helper(name)


def source_hashes(plan):
    return plan["control_sha256"]


def validate_context(plan, context, stage):
    inner = plan
    expected = dict(
        schema=CONTEXT_SCHEMA,
        plan_sha256=sha(canonical(plan)),
        source_code_sha256=inner["code_sha256"],
        runtime_stage_sha256=sha(canonical(stage)),
    )
    require(
        set(context)
        == {*expected, "job_id", "cuda_visible_devices", "work_dir", "source_dir", "science_dir", "uid"},
        "runtime context fields differ",
    )
    require(all(context[k] == v for k, v in expected.items()), "runtime context identity differs")
    require(
        isinstance(context["job_id"], str) and re.fullmatch(r"[1-9][0-9]*", context["job_id"]),
        "runtime context job differs",
    )
    require(type(context["uid"]) is int and context["uid"] == 543540, "runtime allocation owner differs")
    work = Path(context["work_dir"])
    require(
        work.is_absolute() and ".." not in work.parts and str(work) == context["work_dir"],
        "unsafe execution work path",
    )
    require(
        context["source_dir"] == str(work / "source")
        and context["science_dir"] == str(work / "science")
        and stage["destination"] == str(work / "runtime"),
        "runtime context layout differs",
    )
    visible = context["cuda_visible_devices"]
    require(
        isinstance(visible, str)
        and len(visible.split(",")) == len(set(visible.split(","))) == 2
        and all(x and x.strip() == x for x in visible.split(",")),
        "runtime CUDA identity differs",
    )
    return context


def validate_stage(plan, stage, manifest_raw):
    descriptor = plan["runtime_snapshot"]
    require(
        isinstance(manifest_raw, bytes)
        and len(manifest_raw) == descriptor["manifest"]["size"] <= MAX_FILE
        and sha(manifest_raw) == descriptor["manifest"]["sha256"],
        "runtime manifest bytes differ",
    )
    manifest = strict_json(manifest_raw)
    snapshot = bound_helper(plan, "sdsc_runtime_snapshot")
    snapshot.validate_descriptor(descriptor)
    require(
        manifest["schema"] == snapshot.SCHEMA and snapshot.canonical(manifest) == manifest_raw,
        "runtime manifest schema or encoding differs",
    )
    files, total = snapshot.validate_entries(manifest["entries"])
    require(
        manifest["runtime_root_sha256"]
        == descriptor["runtime_root_sha256"]
        == snapshot.sha(
            snapshot.canonical(dict(root_mode=manifest["root_mode"], entries=manifest["entries"]))
        ),
        "runtime inventory digest differs",
    )
    for key in ("source_prefix", "python_relative_path", "python_sha256", "files_count", "total_bytes"):
        require(manifest[key] == descriptor[key], "runtime manifest descriptor differs: " + key)
    require(
        (files, total) == (descriptor["files_count"], descriptor["total_bytes"])
        and manifest["archive"] == dict(descriptor["archive"], path="runtime.bin"),
        "runtime archive/count differs",
    )
    require(
        set(stage)
        == {
            "schema",
            "descriptor",
            "destination",
            "python",
            "local_mount",
            "files_verified",
            "archive_verified",
            "runtime_root_sha256",
            "files_count",
            "total_bytes",
            "node_local_verified",
        },
        "runtime stage fields differ",
    )
    require(
        stage["schema"] == snapshot.STAGE_SCHEMA
        and stage["node_local_verified"] is True
        and stage["descriptor"] == descriptor
        and stage["files_verified"] is True
        and stage["archive_verified"] is True,
        "runtime was not verified on node-local storage",
    )
    require(
        stage["local_mount"]["fstype"] in snapshot.LOCAL_FILESYSTEMS, "runtime is not on a local filesystem"
    )
    local = Path(stage["destination"])
    require(
        local.is_absolute()
        and str(local) == stage["destination"]
        and ".." not in local.parts
        and local != Path(descriptor["source_prefix"]),
        "runtime relocation absent",
    )
    require(
        stage["python"] == str(local / descriptor["python_relative_path"]), "runtime actual Python differs"
    )
    require(
        all(stage[k] == descriptor[k] for k in ("runtime_root_sha256", "files_count", "total_bytes")),
        "runtime stage inventory differs",
    )
    return manifest


def validate_native_inventory(plan, stage, manifest, context, value):
    """File/origin identity only; this never proves locality, CUDA or job completion."""
    local = Path(stage["destination"])
    original = Path(plan["runtime_snapshot"]["source_prefix"])
    require(
        value["executable"] == stage["python"] and value["prefix"] == value["base_prefix"] == str(local),
        "effective runtime is not the relocated native prefix",
    )
    require(
        value["python_version"] == "3.12.13"
        and value["packages"] == bound_helper(plan, "sdsc_student_name_invariant").RUNTIME_PACKAGES,
        "native runtime versions differ",
    )
    source, science = Path(context["source_dir"]), Path(context["science_dir"])
    generated = bound_helper(plan, "sdsc_torch_generated_origins").validate_generated_origin(
        value["generated_origin"],
        work_dir=context["work_dir"],
        runtime_dir=str(local),
        expected_uid=context["uid"],
        runtime_manifest=manifest,
        context_sha256=sha(canonical(context)),
        modules=value["modules"],
        file_rows=value["files"],
        sys_path=value["sys_path"],
        native_libraries=value["native_libraries"],
        tmpdir=value["environment"]["TMPDIR"],
        dont_write_bytecode=value["dont_write_bytecode"],
    )
    paths = value["sys_path"]
    require(
        isinstance(paths, list)
        and paths
        and all(
            isinstance(p, str)
            and Path(p).is_absolute()
            and ".." not in Path(p).parts
            and not Path(p).is_relative_to(original)
            and (
                Path(p).is_relative_to(local)
                or Path(p) in (science, science / "src", source / "tools")
                or (generated is not None and Path(p) == Path(generated).parent)
            )
            for p in paths
        ),
        "foreign Python search path",
    )
    inventory = {r["path"]: r for r in manifest["entries"] if r["type"] == "file"}
    file_rows = value["files"]
    require(
        isinstance(file_rows, list)
        and len(file_rows) <= 50000
        and file_rows == sorted(file_rows, key=lambda r: r["path"]),
        "native file inventory/order differs",
    )
    files = {}
    for row in file_rows:
        require(
            set(row) == {"path", "size", "sha256", "uid", "mode"}
            and row["path"] not in files
            and type(row["size"]) is int
            and row["size"] >= 0
            and isinstance(row["sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
            and type(row["uid"]) is int
            and type(row["mode"]) is int,
            "native file record differs",
        )
        path = Path(row["path"])
        require(
            path.is_absolute()
            and str(path) == row["path"]
            and ".." not in path.parts
            and not path.is_relative_to(original),
            "native origin retained original runtime",
        )
        if path.is_relative_to(local):
            expected_file = inventory.get(str(path.relative_to(local)), {})
            require(
                all(row[k] == expected_file.get(k) for k in ("size", "sha256", "mode")),
                "imported or mapped runtime bytes differ",
            )
        elif path.is_relative_to(source):
            require(
                row["sha256"] == source_hashes(plan).get(str(path.relative_to(source))),
                "unbound deployed Python module",
            )
        elif path.is_relative_to(science):
            require(
                row["sha256"] == plan["protocol"]["science_file_sha256"].get(str(path.relative_to(science))),
                "unbound scientific Python module",
            )
        elif generated is not None and row["path"] == generated:
            pass  # Exact bytes, owner, mode, nlink and live binding checked above.
        else:
            require(
                is_host_native_library(row, value["native_libraries"]),
                "foreign or writable host native library",
            )
        files[row["path"]] = row
    require(
        stage["python"] in files
        and files[stage["python"]]["sha256"] == plan["runtime_snapshot"]["python_sha256"],
        "actual executable bytes absent/differ",
    )
    modules = value["modules"]
    require(
        isinstance(modules, list) and modules and len(modules) <= 50000, "module origins absent/excessive"
    )
    names = set()
    for row in modules:
        require(
            set(row) == {"name", "file", "spec_origin"}
            and isinstance(row["name"], str)
            and row["name"] not in names,
            "module origin record differs",
        )
        names.add(row["name"])
        for key in ("file", "spec_origin"):
            origin = row[key]
            if (
                origin is None
                or origin in ("built-in", "frozen")
                or (isinstance(origin, str) and origin.startswith("<") and origin.endswith(">"))
            ):
                continue
            require(
                isinstance(origin, str)
                and origin in files
                and (
                    any(Path(origin).is_relative_to(root) for root in (local, source, science))
                    or (
                        generated is not None
                        and origin == generated
                        and row["name"] == "_remote_module_non_scriptable"
                    )
                ),
                "Python module imported from a foreign origin",
            )
    require({"torch", "numpy", "transformers", "datasets"} <= names, "entry runtime imports incomplete")
    libraries = value["native_libraries"]
    require(
        isinstance(libraries, list)
        and libraries
        and libraries == sorted(set(libraries))
        and all(p in files for p in libraries),
        "native map evidence incomplete",
    )
    return dict(
        module_count=len(modules),
        native_library_count=len(libraries),
        files_verified=len(files),
    )


def validate_runtime_origin(plan, stage, manifest, context, value, *, phase, rank=None):
    """Validate observations rather than trusting the producer's success flag."""
    expected = dict(
        schema=RUNTIME_SCHEMA,
        phase=phase,
        rank=rank,
        job_id=context["job_id"],
        plan_sha256=sha(canonical(plan)),
        context_sha256=sha(canonical(context)),
        runtime_root_sha256=stage["runtime_root_sha256"],
    )
    keys = {
        *expected,
        "executable",
        "prefix",
        "base_prefix",
        "sys_path",
        "python_version",
        "packages",
        "environment",
        "dont_write_bytecode",
        "modules",
        "native_libraries",
        "files",
        "generated_origin",
        "passed",
        "error",
        *FLAGS,
    }
    require(
        isinstance(value, dict) and set(value) == keys and all(value[k] == v for k, v in expected.items()),
        "native runtime evidence identity differs",
    )
    require(
        (
            rank is None and value["rank"] is None
            if phase == "early"
            else type(rank) is int and type(value["rank"]) is int and rank in (0, 1)
        )
        and phase in ("early", "before", "after"),
        "native runtime phase/rank differs",
    )
    require(
        value["passed"] is True and value["error"] is None and all(value[k] is False for k in FLAGS),
        "native runtime attestation failed",
    )
    original = Path(plan["runtime_snapshot"]["source_prefix"])
    environment = value["environment"]
    require(
        set(environment) == set(ENV_KEYS)
        and all(v is None or isinstance(v, str) for v in environment.values()),
        "runtime environment fields differ",
    )
    require(
        environment["CUDA_VISIBLE_DEVICES"] == context["cuda_visible_devices"]
        and environment["PYTHONHOME"] is None
        and environment["PYTHONPATH"] is None
        and environment["PYTHONDONTWRITEBYTECODE"] == "1"
        and environment["PYTHONNOUSERSITE"] == "1"
        and value["dont_write_bytecode"] is True
        and environment["TMPDIR"] == context["work_dir"],
        "runtime isolation/visibility differs",
    )
    require(
        all(environment[k] == "12" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
        "runtime thread count differs",
    )
    require(
        not any(str(original) in (environment.get(k) or "") for k in ("LD_LIBRARY_PATH", "LD_PRELOAD")),
        "native search path retains source runtime",
    )
    return dict(phase=phase, rank=rank, **validate_native_inventory(plan, stage, manifest, context, value))


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
    value = dict(
        schema=RUNTIME_SCHEMA,
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
        environment={key: os.environ.get(key) for key in ENV_KEYS},
        dont_write_bytecode=sys.dont_write_bytecode,
        modules=[],
        native_libraries=[],
        files=[],
        generated_origin=None,
        passed=False,
        error=None,
        **dict.fromkeys(control.FLAGS, False),
    )
    try:
        value["packages"] = {name: importlib.metadata.version(name) for name in control.RUNTIME_PACKAGES}
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

        generated_helper = bound_helper(plan, "sdsc_torch_generated_origins")
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
        value["generated_origin"] = generated_helper.capture_generated_origin(
            work_dir=context["work_dir"],
            runtime_dir=stage["destination"],
            expected_uid=context["uid"],
            runtime_manifest=manifest,
            context_sha256=control.sha(control.canonical(context)),
            modules=modules,
        )
        value["passed"] = True
        validate_runtime_origin(plan, stage, manifest, context, value, phase=phase, rank=rank)
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
        str(work / "artifacts" / "runtime-context.json"),
    ]


def plan_digest(plan):
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def frozen_probe_args(plan, context):
    return [
        "--worker-sha256",
        plan["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"],
        "--output-json",
        str(Path(context["work_dir"]) / "artifacts/early-node-startup.json"),
        "--job-id",
        context["job_id"],
        "--expected-cuda-visible-devices",
        context["cuda_visible_devices"],
    ]


def publish_origin(control, path, value):
    raw = control.canonical(value)
    control.require(len(raw) <= 16 * 1024**2, "runtime origins exceed execution bound")
    control.write_once(path, raw)


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
    audit, inner = sys.modules[__name__], plan
    context = control.document(args.context)
    work = Path(context["work_dir"])
    execution = work / "artifacts"
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
        == Path(context["source_dir"]) / "tools/sdsc_student_name_invariant_native.py",
        "runtime entry not executing relocated/staged bytes",
    )
    control.require(context["uid"] == os.getuid(), "native entry owner differs")
    control.require(
        sys.dont_write_bytecode and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "runtime bytecode mutation enabled",
    )
    startup = bound_helper(plan, "sdsc_student_name_invariant_startup")
    worker_sha = inner["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"]
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
        for name in ("torch", "numpy", "transformers", "datasets", "accelerate", "torch.distributed.fsdp"):
            importlib.import_module(name)
        # Resolve the actual architecture and optimizer import closure before any weights.
        # Plain `import transformers` lazily defers the Torch generated-module side effect.
        from torch.optim import AdamW
        from transformers import Qwen3ForCausalLM

        control.require(
            Qwen3ForCausalLM.__name__ == "Qwen3ForCausalLM" and AdamW.__name__ == "AdamW",
            "training architecture/optimizer imports differ",
        )
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
    snapshot = bound_helper(plan, "sdsc_runtime_snapshot")
    observed = snapshot.stage(
        Path(descriptor["manifest"]["path"]).parent,
        descriptor["manifest"]["sha256"],
        args.work / "runtime",
        artifact_root=control.RUNTIME_ARTIFACT_ROOT,
        destination_root=args.work,
        deadline=args.deadline,
    )
    control.require(observed["descriptor"] == descriptor, "staged runtime descriptor differs")
    control.write_once(args.work / "artifacts/runtime-stage.json", control.canonical(observed))
    return 0


def load_control(path, expected):
    raw = Path(path).read_bytes()
    require(len(raw) <= 4 * CAP and sha(raw) == expected, "native entry plan bytes differ")
    plan = strict_json(raw)
    source = Path(__file__).with_name("sdsc_student_name_invariant.py")
    require(
        sha(source.read_bytes()) == plan["control_sha256"]["tools/sdsc_student_name_invariant.py"],
        "named native controller hash differs",
    )
    require(
        sha(Path(__file__).read_bytes())
        == plan["control_sha256"]["tools/sdsc_student_name_invariant_native.py"],
        "native adapter changed",
    )
    control = helper("sdsc_student_name_invariant")
    control.validate_plan(plan)
    return control, plan


def context_value(plan, stage, work, identity, control):
    return dict(
        schema=CONTEXT_SCHEMA,
        plan_sha256=sha(canonical(plan)),
        source_code_sha256=plan["code_sha256"],
        runtime_stage_sha256=sha(canonical(stage)),
        job_id=identity["job_id"],
        cuda_visible_devices=identity["cuda_visible_devices"],
        uid=os.getuid(),
        work_dir=str(work),
        source_dir=str(work / "source"),
        science_dir=str(work / "science"),
    )


def probe_argv(plan, context, stage):
    return [
        stage["python"],
        "-I",
        "-B",
        "-u",
        "-X",
        "importtime",
        str(Path(context["source_dir"]) / "tools/sdsc_student_name_invariant_native.py"),
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
        "accelerate.commands.launch",
        "--config_file",
        str(Path(context["science_dir"]) / "configs/accelerate/fsdp_2gpu_adapted_student_v2.yaml"),
        "--num_processes",
        "2",
        "--num_cpu_threads_per_process",
        "12",
        "--main_process_port",
        "0",
        str(Path(context["source_dir"]) / "tools/sdsc_student_name_invariant_native.py"),
        "runtime-entry",
        "--phase",
        "run",
        *entry_args(plan, context),
    ]


def validate_evidence(plan, files, job, *, expected_worker_exit=0):
    require(
        type(expected_worker_exit) is int
        and (expected_worker_exit == 0 or (expected_worker_exit == 2 and plan["mode"] == "fit")),
        "unreviewed worker exit interpretation",
    )
    control = bound_helper(plan, "sdsc_student_name_invariant")
    require(set(NAMES) <= files.keys(), "native evidence files incomplete")
    require(
        all(isinstance(files[n], bytes) and len(files[n]) <= MAX_FILE for n in NAMES)
        and sum(len(files[n]) for n in NAMES) <= MAX_TOTAL,
        "native evidence size bound exceeded",
    )
    stage = strict_json(files["runtime-stage.json"])
    manifest = validate_stage(plan, stage, files["runtime-manifest.json"])
    context = validate_context(plan, strict_json(files["runtime-context.json"]), stage)
    require(context["job_id"] == job, "native evidence belongs to another allocation")
    native = []
    for phase, rank, name in [
        ("early", None, "runtime-early.json"),
        *(
            (phase, rank, f"runtime-rank-{rank}-{phase}.json")
            for rank in (0, 1)
            for phase in ("before", "after")
        ),
    ]:
        native.append(
            validate_runtime_origin(
                plan, stage, manifest, context, strict_json(files[name]), phase=phase, rank=rank
            )
        )
    startup = bound_helper(plan, "sdsc_student_name_invariant_startup")
    worker_sha = plan["control_sha256"]["tools/sdsc_student_name_invariant_worker.py"]
    work = Path(context["work_dir"])
    args = ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(work / "artifacts")]
    for name, scope, rank in [
        ("early-node-startup.json", "early_node", None),
        ("rank-0-startup.json", "rank_entry", 0),
        ("rank-1-startup.json", "rank_entry", 1),
    ]:
        value = strict_json(files[name])
        startup.validate_report(
            value,
            worker_sha256=worker_sha,
            job_id=job,
            expected_cuda_visible_devices=context["cuda_visible_devices"],
            scope=scope,
            rank=rank,
            original_argv=args if rank is not None else None,
            expected_python=stage["python"],
        )
        require(value["cuda_ready"] is True, "native CUDA startup failed")
    for rank in (0, 1):
        value = strict_json(files[f"rank-{rank}-exit.json"])
        require(
            value.get("schema") == startup.EXIT_SCHEMA
            and value.get("job_id") == job
            and type(value.get("rank")) is int
            and value["rank"] == rank
            and value.get("worker_invoked") is True
            and value.get("original_worker_sha256") == worker_sha
            and value.get("original_argv") == args
            and type(value.get("exit_code")) is int
            and value["exit_code"] == expected_worker_exit
            and "error" not in value
            and all(value.get(k) is False for k in FLAGS),
            "native rank execution failed",
        )
    trace = startup.validate_trace_meta(
        strict_json(files["trace-meta.json"]),
        worker_sha256=worker_sha,
        job_id=job,
        expected_cuda_visible_devices=context["cuda_visible_devices"],
        expected_python=stage["python"],
        expected_argv=probe_argv(plan, context, stage),
        log_bytes=files["startup.log"],
    )
    require(trace["probe_passed"] is True, "native early trace failed")
    require(sum(len(files[n]) for n in control.EXECUTION_NAMES) <= 2 * CAP, "startup evidence exceeds2MiB")
    if "node-result.json" in files:
        node = strict_json(files["node-result.json"])
        require(
            node.get("job_id") == job
            and node.get("plan_sha256") == sha(canonical(plan))
            and node.get("work_dir") == context["work_dir"]
            and node.get("runtime")
            == dict(
                source_python=plan["python"],
                effective_python=stage["python"],
                runtime_stage_sha256=sha(canonical(stage)),
            ),
            "node/native context differs",
        )
    return dict(
        passed=True,
        runtime_root_sha256=stage["runtime_root_sha256"],
        native=native,
        probe_passed=True,
        worker_exit_verified=expected_worker_exit,
        execution_acceptance_claim=False,
    )


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    require(bool(args), "native entry action missing")
    if args[0] == "runtime-entry":
        return runtime_entry(args[1:])
    if args[0] == "stage-runtime":
        return stage_runtime(args[1:])
    raise ValueError("unknown native action")


if __name__ == "__main__":
    raise SystemExit(main())
