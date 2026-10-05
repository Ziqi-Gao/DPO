#!/usr/bin/env python3
"""Independent native-runtime and execution evidence audit; no scientific acceptance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

CAP = 1024**2
MAX_FILE = 16 * CAP
MAX_TOTAL = 64 * CAP
PUBLICATION_SCHEMA = "quest-sdsc-student-name-cue-recovery-publication-v1"
RUNTIME_SCHEMA = "quest-sdsc-student-name-cue-native-origin-v1"
CONTEXT_SCHEMA = "quest-sdsc-student-name-cue-runtime-context-v1"
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
    "recovery-node-result.json",
    "memory.json",
    "startup.log",
    "trace-meta.json",
    "early-node-startup.json",
    *(f"rank-{rank}-{phase}.json" for rank in (0, 1) for phase in ("startup", "exit")),
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
)
SYSTEM_ROOTS = ("/usr/lib", "/usr/lib64", "/lib", "/lib64")


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
    spec = importlib.util.spec_from_file_location("_recovery_audit_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_hashes(plan):
    return {**plan["science_plan"]["control_sha256"], **plan["execution"]["control_file_sha256"]}


def validate_context(plan, context, stage):
    inner = plan["science_plan"]
    expected = dict(
        schema=CONTEXT_SCHEMA,
        plan_sha256=sha(canonical(plan)),
        science_plan_sha256=sha(canonical(inner)),
        source_code_sha256=inner["code_sha256"],
        runtime_stage_sha256=sha(canonical(stage)),
    )
    require(
        set(context)
        == {*expected, "job_id", "cuda_visible_devices", "work_dir", "source_dir", "science_dir"},
        "runtime context fields differ",
    )
    require(all(context[k] == v for k, v in expected.items()), "runtime context identity differs")
    require(
        isinstance(context["job_id"], str) and re.fullmatch(r"[1-9][0-9]*", context["job_id"]),
        "runtime context job differs",
    )
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
    manifest = json.loads(manifest_raw)
    snapshot = helper("sdsc_runtime_snapshot")
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
        and value["packages"] == helper("sdsc_student_name_cue_probe").RUNTIME_PACKAGES,
        "native runtime versions differ",
    )
    source, science = Path(context["source_dir"]), Path(context["science_dir"])
    paths = value["sys_path"]
    require(
        isinstance(paths, list)
        and paths
        and all(
            isinstance(p, str)
            and Path(p).is_absolute()
            and ".." not in Path(p).parts
            and not Path(p).is_relative_to(original)
            and (Path(p).is_relative_to(local) or Path(p) in (science, science / "src", source / "tools"))
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
                row["sha256"]
                == plan["science_plan"]["protocol"]["science_file_sha256"].get(
                    str(path.relative_to(science))
                ),
                "unbound scientific Python module",
            )
        else:
            require(
                any(path.is_relative_to(Path(root)) for root in SYSTEM_ROOTS)
                and row["uid"] == 0
                and not row["mode"] & 0o022,
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
                and any(Path(origin).is_relative_to(root) for root in (local, source, science)),
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
        and value["dont_write_bytecode"] is True,
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


def validate_execution_publication(plan, publication, files=None):
    inner = plan["science_plan"]
    expected = dict(
        schema=PUBLICATION_SCHEMA,
        task=plan["task"],
        recovery_intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        science_plan_sha256=sha(canonical(inner)),
        run_id=inner["run_id"],
        code_sha256=inner["code_sha256"],
        persistent_read_back_verified=True,
        large_files=[],
    )
    require(
        set(publication)
        == {
            *expected,
            "job_id",
            "execution_complete",
            "stage_complete",
            "passed",
            "scientific_publication_sha256",
            "files",
            *FLAGS,
        },
        "execution publication fields differ",
    )
    require(
        all(publication[k] == v for k, v in expected.items())
        and publication["persistent_read_back_verified"] is True
        and all(publication[k] is False for k in FLAGS),
        "execution publication identity differs",
    )
    require(
        isinstance(publication["job_id"], str) and re.fullmatch(r"[1-9][0-9]*", publication["job_id"]),
        "execution publication job differs",
    )
    require(
        all(type(publication[k]) is bool for k in ("execution_complete", "stage_complete", "passed"))
        and publication["execution_complete"] == publication["stage_complete"] == publication["passed"],
        "execution publication completion differs",
    )
    require(
        isinstance(publication["scientific_publication_sha256"], str)
        and re.fullmatch(r"[a-f0-9]{64}", publication["scientific_publication_sha256"]),
        "scientific publication binding absent",
    )
    rows = publication["files"]
    require(isinstance(rows, list) and len(rows) <= 32, "execution publication inventory exceeds bound")
    records, total = {}, 0
    for row in rows:
        require(
            set(row) == {"path", "size", "sha256"}
            and row["path"] in NAMES
            and row["path"] not in records
            and type(row["size"]) is int
            and 0 <= row["size"] <= (CAP if row["path"] == "startup.log" else MAX_FILE)
            and isinstance(row["sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]),
            "execution publication record differs",
        )
        records[row["path"]] = row
        total += row["size"]
    require(
        total <= MAX_TOTAL and {"recovery-node-result.json", "memory.json"} <= records.keys(),
        "execution publication bound/required files differ",
    )
    if publication["passed"]:
        require(set(records) == set(NAMES), "successful execution evidence incomplete")
    if files is not None:
        require(set(files) == set(records), "execution fetched inventory differs")
        for name, raw in files.items():
            require(
                isinstance(raw, bytes)
                and len(raw) == records[name]["size"]
                and sha(raw) == records[name]["sha256"],
                "execution fetched bytes differ: " + name,
            )
        node = json.loads(files["recovery-node-result.json"])
        require(
            node.get("schema") == "quest-sdsc-student-name-cue-recovery-node-v1"
            and node.get("job_id") == publication["job_id"]
            and node.get("plan_sha256") == publication["plan_sha256"]
            and node.get("science_plan_sha256") == publication["science_plan_sha256"]
            and node.get("stage_complete") is publication["passed"]
            and all(node.get(k) is False for k in FLAGS),
            "execution publication/node identity differs",
        )
        if publication["passed"]:
            context = json.loads(files["runtime-context.json"])
            require(
                context.get("job_id") == publication["job_id"]
                and context.get("plan_sha256") == publication["plan_sha256"]
                and context.get("science_plan_sha256") == publication["science_plan_sha256"]
                and context.get("source_code_sha256") == publication["code_sha256"],
                "execution publication/context identity differs",
            )
    return records


def validate_execution_evidence(plan, files):
    """Successful execution requires origins on entry and after all scientific work."""
    require(
        set(files) == set(NAMES)
        and sum(map(len, files.values())) <= MAX_TOTAL
        and all(len(v) <= MAX_FILE for v in files.values()),
        "execution evidence inventory/bound differs",
    )
    objects = {k: json.loads(v) for k, v in files.items() if k.endswith(".json")}
    stage, context, node = (
        objects[n] for n in ("runtime-stage.json", "runtime-context.json", "recovery-node-result.json")
    )
    manifest = validate_stage(plan, stage, files["runtime-manifest.json"])
    validate_context(plan, context, stage)
    require(
        node["schema"] == "quest-sdsc-student-name-cue-recovery-node-v1"
        and node["job_id"] == context["job_id"]
        and node["plan_sha256"] == sha(canonical(plan))
        and node["science_plan_sha256"] == sha(canonical(plan["science_plan"]))
        and node["stage_complete"] is True
        and type(node["exit_code"]) is int
        and node["exit_code"] == 0
        and all(node[k] is False for k in FLAGS),
        "recovery node did not complete",
    )
    require(
        type(node["elapsed_seconds"]) in (float, int)
        and math.isfinite(node["elapsed_seconds"])
        and 0 < node["elapsed_seconds"] <= 4800,
        "recovery staging/inference exceeded original deadline",
    )
    summaries = [
        validate_runtime_origin(plan, stage, manifest, context, objects["runtime-early.json"], phase="early")
    ]
    for rank in (0, 1):
        for phase in ("before", "after"):
            summaries.append(
                validate_runtime_origin(
                    plan,
                    stage,
                    manifest,
                    context,
                    objects[f"runtime-rank-{rank}-{phase}.json"],
                    phase=phase,
                    rank=rank,
                )
            )
        require(
            objects[f"runtime-rank-{rank}-before.json"]["environment"]
            == objects[f"runtime-rank-{rank}-after.json"]["environment"],
            "worker changed native runtime environment",
        )
    inner = plan["science_plan"]
    control, startup = helper("sdsc_student_name_cue_probe"), helper("sdsc_student_name_cue_probe_startup")
    job = helper("sdsc_student_name_cue_probe_job")
    wrapper = helper("sdsc_student_name_cue_recovery_job")
    args = [
        "--inputs-json",
        str(Path(context["work_dir"]) / "inputs.json"),
        "--output-dir",
        str(Path(context["work_dir"]) / "artifacts"),
    ]
    identity = dict(job_id=context["job_id"], cuda_visible_devices=context["cuda_visible_devices"])
    startup.validate_trace_meta(
        objects["trace-meta.json"],
        worker_sha256=inner["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"],
        job_id=context["job_id"],
        expected_cuda_visible_devices=context["cuda_visible_devices"],
        expected_python=stage["python"],
        expected_argv=wrapper.probe_argv(plan, context, stage),
        log_bytes=files["startup.log"],
    )
    require(objects["trace-meta.json"]["probe_passed"] is True, "native early CUDA probe failed")
    job.validate_startup(
        inner,
        control,
        objects["early-node-startup.json"],
        identity["job_id"],
        identity["cuda_visible_devices"],
        "early_node",
    )
    for rank in (0, 1):
        job.validate_startup(
            inner,
            control,
            objects[f"rank-{rank}-startup.json"],
            identity["job_id"],
            identity["cuda_visible_devices"],
            "rank_entry",
            rank,
            args,
        )
        job.validate_rank_exit(
            inner, control, objects[f"rank-{rank}-exit.json"], identity["job_id"], rank, args
        )
    control.validate_memory(objects["memory.json"], context["job_id"])
    return dict(
        execution_complete=True,
        relocated_python=stage["python"],
        native_origins=summaries,
        original_startup_validated=True,
        scientific_acceptance=False,
        **dict.fromkeys(FLAGS, False),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fetched-dir", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    control = helper("sdsc_student_name_cue_recovery")
    plan = control.validate_plan(json.loads(args.plan.read_bytes()))
    root = args.fetched_dir / "execution"
    publication_raw = control.read(root / "receipt.json")
    require(sha(publication_raw) == args.receipt_sha256, "execution publication receipt SHA differs")
    publication = json.loads(publication_raw)
    records = validate_execution_publication(plan, publication)
    files = {name: control.read(root / name, MAX_FILE) for name in records}
    validate_execution_publication(plan, publication, files)
    require(
        sha(control.read(args.fetched_dir / "receipt.json")) == publication["scientific_publication_sha256"],
        "scientific publication changed",
    )
    inner_publication = json.loads(control.read(args.fetched_dir / "receipt.json"))
    control.science.publication_records(inner_publication, plan["science_plan"], publication["job_id"])
    require(
        inner_publication["passed"] is True and publication["passed"] is True,
        "execution audit requires complete scientific and execution publications",
    )
    result = validate_execution_evidence(plan, files)
    result["execution_publication_sha256"] = sha(control.read(root / "receipt.json"))
    control.write_once(args.output, canonical(result))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
