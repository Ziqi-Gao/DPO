#!/usr/bin/env python3
"""Resume a byte-identical calibration twice inside an explicit private mount namespace.

Default: validate the request/input inventory and print a plan. --execute requires
a real two-H100 allocation; it never submits jobs. The caller stages all inputs
and persists the complete workspace after this foreground process exits.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path

SCHEMA = "quest-sdsc-resume-request-v1"
REPORT_SCHEMA = "quest-sdsc-distributed-resume-v1"
SCIENCE_HEAD = "0215c356355b29b5e2b407978a207db2156719e1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
REFERENCE = "canonical_sft"
STEP20 = REFERENCE + "/checkpoints/step-00000020.pt"
FALSE_CLAIMS = dict(g0_passed=False, pilot_passed=False, execution_class_certified=False)
ROLES = {"science": "science", "tools": "tools", "hf_home": "huggingface", "mib": "mib"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def helper(name):
    spec = importlib.util.spec_from_file_location("_resume_" + name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def absolute(value):
    require(isinstance(value, str) and value, "path must be a string")
    path = Path(value)
    require(path.is_absolute() and str(path) == value and ".." not in path.parts, "noncanonical path")
    require(not any(ord(c) < 32 or c in ":,\\" for c in value), "unsafe bind path")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "path traverses a symlink")
    return path


def relative(value):
    require(isinstance(value, str) and value, "invalid inventory path")
    path = Path(value)
    require(not path.is_absolute() and str(path) == value and ".." not in path.parts, "unsafe inventory path")
    require(not any(ord(c) < 32 or c == "\\" for c in value), "invalid inventory characters")
    return path


def file_identity(path, *, maximum=256 * 1024**3, read=False):
    path = absolute(str(path))
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum, "invalid file or size")
        digest, size, chunks = hashlib.sha256(), 0, []
        while block := stream.read(1024 * 1024):
            size += len(block)
            require(size <= maximum, "file grew beyond limit")
            digest.update(block)
            if read:
                chunks.append(block)
        after = os.fstat(stream.fileno())
    current = path.lstat()

    def signature(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    require(
        signature(before) == signature(after) == signature(current) and size == before.st_size,
        "file changed during hashing",
    )
    result = {"size": size, "sha256": digest.hexdigest()}
    return (result, b"".join(chunks)) if read else result


def document(path, expected=None):
    identity, raw = file_identity(path, maximum=8 * 1024**2, read=True)
    require(expected is None or identity["sha256"] == expected, "document SHA mismatch")

    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(value, dict), "expected JSON object")
    canonical(value)
    return value


def publish(path, value):
    path = absolute(str(path))
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_request(request):
    fields = {
        "schema",
        "target",
        "work_dir",
        "workspace",
        "science_root",
        "hf_home",
        "python",
        "python_sha256",
        "reference_report",
        "reference_report_sha256",
        "publication_receipt",
        "publication_receipt_sha256",
        "expected_science_head",
        "expected_protocol_sha256",
        "container",
        "namespace_root",
        "readonly_binds",
    }
    require(set(request) == fields and request["schema"] == SCHEMA, "unsupported resume request schema")
    target = request["target"]
    require(set(target) == {"run_id", "code_sha256", "job_id"}, "invalid target fields")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", target["run_id"]), "invalid run ID")
    require(re.fullmatch(r"[1-9][0-9]*", target["job_id"]), "invalid target job ID")
    for name in (
        "python_sha256",
        "reference_report_sha256",
        "publication_receipt_sha256",
        "expected_protocol_sha256",
    ):
        require(SHA.fullmatch(request[name]), "invalid externally bound SHA")
    require(SHA.fullmatch(target["code_sha256"]), "invalid target code SHA")
    require(request["expected_science_head"] == SCIENCE_HEAD, "unsupported science HEAD")
    paths = {
        name: absolute(request[name])
        for name in (
            "work_dir",
            "workspace",
            "science_root",
            "hf_home",
            "python",
            "reference_report",
            "publication_receipt",
            "namespace_root",
        )
    }
    work, workspace, namespace = paths["work_dir"], paths["workspace"], paths["namespace_root"]
    require(work.is_dir() and work.stat().st_uid == os.getuid(), "work directory must be owned")
    require(work in workspace.parents and workspace.is_dir(), "workspace must be staged below work directory")
    require(work != Path.home() and Path.home() not in work.parents, "work directory is in HOME")
    require(
        namespace.parent == Path("/opd-pipeline")
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", namespace.name),
        "invalid namespace root",
    )
    for name in ("science_root", "hf_home", "reference_report", "publication_receipt"):
        require(
            work in paths[name].parents and workspace not in paths[name].parents,
            "inputs and proofs must be separate staged paths below work directory",
        )
    binds = request["readonly_binds"]
    require(set(binds) in ({"science", "tools", "hf_home"}, set(ROLES)), "invalid readonly bind roles")
    for role, pair in binds.items():
        require(set(pair) == {"host", "guest"}, "invalid readonly bind fields")
        host, guest = absolute(pair["host"]), absolute(pair["guest"])
        require(host.is_dir() and work in host.parents, "readonly bind is outside staged work")
        require(guest == namespace / ROLES[role], "readonly guest destination is not fixed")
        require(
            host != workspace and workspace not in host.parents and host not in workspace.parents,
            "readonly bind overlaps reference workspace",
        )
    require(
        binds["science"]["host"] == request["science_root"]
        and binds["hf_home"]["host"] == request["hf_home"],
        "readonly role source differs",
    )
    script = Path(binds["tools"]["host"]) / "sdsc_resume.py"
    require(
        script.is_file() and os.path.samestat(script.stat(), Path(__file__).stat()),
        "tools bind must contain the executing immutable adapter",
    )
    python = paths["python"]
    require(
        python.parent.name == "bin" and python.is_file() and os.access(python, os.X_OK),
        "expected real Conda Python executable",
    )
    require(file_identity(python)["sha256"] == request["python_sha256"], "Python executable changed")
    container = request["container"]
    require(set(container) == {"runtime", "image", "size", "mtime_ns"}, "invalid container identity")
    runtime, image = absolute(container["runtime"]), absolute(container["image"])
    require(runtime.is_file() and os.access(runtime, os.X_OK), "container runtime unavailable")
    info = image.stat()
    require(
        stat.S_ISREG(info.st_mode)
        and type(container["size"]) is int
        and container["size"] > 0
        and type(container["mtime_ns"]) is int
        and container["mtime_ns"] > 0
        and (info.st_size, info.st_mtime_ns) == (container["size"], container["mtime_ns"]),
        "container image changed or is not regular",
    )
    return paths


def input_inventory(request):
    report = document(Path(request["reference_report"]), request["reference_report_sha256"])
    receipt = document(Path(request["publication_receipt"]), request["publication_receipt_sha256"])
    require(
        report.get("task") == "qwen3-v2-g0-calibration"
        and report.get("passed") is True
        and type(report.get("exit_code")) is int
        and report.get("exit_code") == 0
        and report.get("training_executed") is True,
        "reference calibration did not complete real training",
    )
    require(all(report.get(key) is False for key in FALSE_CLAIMS), "reference claims unsupported acceptance")
    require(
        receipt.get("passed") is True
        and receipt.get("persisted") is True
        and receipt.get("persistent_read_back_verified") is True,
        "reference publication incomplete",
    )
    for name in (
        "task",
        "job_id",
        "run_id",
        "code_sha256",
        "science_git_head",
        "bundle_sha256",
        "provenance_manifest_sha256",
    ):
        require(
            report.get(name) and report[name] == receipt.get(name), "reference producer identity mismatch"
        )
    require(report["science_git_head"] == request["expected_science_head"], "reference science HEAD differs")
    source = absolute(report["training_artifact_root"]).parent
    require(
        Path(report["training_artifact_root"]).name == REFERENCE and source.name == "qwen3-v2",
        "reference run must retain its canonical_sft path",
    )
    require(
        str(source).startswith("/scratch/") and len(source.parts) >= 5,
        "source workspace must be a dedicated original node-local scratch path",
    )
    workspace = Path(request["workspace"])
    require(
        source != workspace and source not in workspace.parents and workspace not in source.parents,
        "namespace source must be distinct from the staged workspace",
    )
    records = receipt.get("files")
    require(isinstance(records, list) and records, "publication inventory missing")
    inventory = {}
    for record in records:
        require(set(record) == {"path", "size", "sha256"}, "invalid publication entry")
        relative(record["path"])
        require(
            record["path"] not in inventory
            and type(record["size"]) is int
            and record["size"] >= 0
            and SHA.fullmatch(record["sha256"]),
            "invalid or duplicate publication entry",
        )
        inventory[record["path"]] = record
    require(
        inventory.get("g0-calibration.json", {}).get("sha256") == request["reference_report_sha256"],
        "reference report is not bound by publication",
    )
    chosen = {}
    for name, record in inventory.items():
        if name == "artifacts/initial_checkpoint.pt" or any(
            name.startswith("artifacts/" + stem + "/") for stem in (REFERENCE, "dataset", "teacher_demos")
        ):
            local = name.removeprefix("artifacts/")
            observed = file_identity(workspace / local)
            require(observed == {key: record[key] for key in ("size", "sha256")}, "staged input SHA differs")
            chosen[local] = observed
    required = {
        "initial_checkpoint.pt",
        "dataset/manifest.json",
        "teacher_demos/manifest.json",
        REFERENCE + "/manifest.json",
        REFERENCE + "/resolved_config.yaml",
        STEP20,
    }
    require(required <= chosen.keys(), "required reference inputs are missing")
    require(
        any(name.startswith(STEP20.removesuffix(".pt") + ".accelerate/") for name in chosen),
        "step20 full Accelerate state is missing",
    )
    actual = set()
    for stem in (REFERENCE, "dataset", "teacher_demos"):
        for directory, directories, files in os.walk(workspace / stem, followlinks=False):
            for child in directories:
                absolute(str(Path(directory) / child))
            for child in files:
                path = absolute(str(Path(directory) / child))
                require(stat.S_ISREG(path.lstat().st_mode), "nonregular reference entry")
                actual.add(str(path.relative_to(workspace)))
    require(actual | {"initial_checkpoint.pt"} == set(chosen), "unlisted or missing reference files")
    return report, source, chosen


def build_namespace_command(request, source, plan_path, plan_hash):
    work, workspace = Path(request["work_dir"]), Path(request["workspace"])
    namespace = Path(request["namespace_root"])
    prefix = Path(request["python"]).parent.parent
    for other in (work, namespace, prefix):
        require(
            source != other and source not in other.parents and other not in source.parents,
            "source workspace overlaps a runtime/control bind",
        )
    binds = [(work, work, "rw"), (workspace, source, "rw"), (prefix, prefix, "ro")]
    binds.extend(
        (Path(pair["host"]), Path(pair["guest"]), "ro") for pair in request["readonly_binds"].values()
    )
    binds.extend(
        (workspace / stem, source / stem, "ro")
        for stem in (REFERENCE, "dataset", "teacher_demos", "initial_checkpoint.pt")
    )
    command = [request["container"]["runtime"], "exec", "--nv", "--cleanenv", "--no-home"]
    for host, guest, access in binds:
        command.extend(["--bind", f"{host}:{guest}:{access}"])
    command.extend(
        [
            "--pwd",
            str(namespace / "science"),
            request["container"]["image"],
            request["python"],
            "-I",
            "-B",
            "-u",
            str(namespace / "tools/sdsc_resume.py"),
            "--inside-plan",
            str(plan_path),
            "--inside-plan-sha256",
            plan_hash,
        ]
    )
    return command, [{"host": str(a), "guest": str(b), "access": c} for a, b, c in binds]


def container_environment(request, environment):
    require(environment.get("SLURM_JOB_ID") == request["target"]["job_id"], "target Slurm job differs")
    require(
        environment.get("SLURM_CPUS_PER_TASK") == "24" and environment.get("SLURM_MEM_PER_NODE") == "196608",
        "requires 24 CPUs and 192 GiB",
    )
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    require(
        len(visible.split(",")) == 2
        and len(set(visible.split(","))) == 2
        and all(re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in visible.split(",")),
        "exactly two assigned CUDA devices required",
    )
    clean = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
        "SINGULARITY_TMPDIR": str(Path(request["work_dir"]) / "resume-tmp"),
    }
    for name in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE"):
        # Unlike --env's comma-separated grammar this preserves a two-device value verbatim.
        clean["SINGULARITYENV_" + name] = environment[name]
    clean["SINGULARITYENV_TMPDIR"] = clean["SINGULARITY_TMPDIR"]
    clean["SINGULARITYENV_PYTHONDONTWRITEBYTECODE"] = "1"
    clean["SINGULARITYENV_PYTHONNOUSERSITE"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        clean["SINGULARITYENV_" + name] = "12"
    return clean


def training_environment(request, science, hf_home, visible, original):
    """Set every numerical runtime's thread budget before launching either rank."""
    environment = {
        "PATH": str(Path(request["python"]).parent) + ":/usr/bin:/bin",
        "HOME": str(Path(request["work_dir"]) / "resume-home"),
        "TMPDIR": str(Path(request["work_dir"]) / "resume-tmp"),
        "PYTHONPATH": str(science / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "12",
        "MKL_NUM_THREADS": "12",
        "OPENBLAS_NUM_THREADS": "12",
        "NUMEXPR_NUM_THREADS": "12",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "CUDA_VISIBLE_DEVICES": visible,
    }
    if original.get("LD_LIBRARY_PATH"):
        environment["LD_LIBRARY_PATH"] = original["LD_LIBRARY_PATH"]
    for name in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE"):
        environment[name] = original[name]
    return environment


def compare_runs(workspace, config, formal_binding):
    """Use the original validators and exact comparator with an explicit reference role."""
    api = importlib.import_module("posttrain_circuits.cli.compare_distributed_resume")
    source = workspace / STEP20
    source_hash = file_identity(source)["sha256"]
    common = dict(
        expected_world_size=2,
        expected_resolved_config=config,
        expected_code_commit=formal_binding["code_commit"],
    )
    reference = api.validate_factorial_run_artifacts(
        workspace / REFERENCE, expected_resume_ancestry=[], **common
    )
    source_binding = api._validate_resume_source_artifacts(
        source, source_sha256=source_hash, reference=reference, config=config, world_size=2
    )
    require(source_binding["token_budget"].get("stop_reason") is None, "step20 checkpoint is terminal")
    replicas = [
        api.validate_factorial_run_artifacts(
            workspace / name,
            expected_resume_ancestry=["sha256:" + source_hash],
            expected_update_norm_baseline_checkpoint_path=source,
            **common,
        )
        for name in ("resume-a", "resume-b")
    ]
    comparison = api._build_comparison_payload(
        config=config,
        world_size=2,
        tolerance=api.FIXED_DISTRIBUTED_RESUME_TOLERANCE,
        workspace=workspace,
        resume_source=source,
        resume_source_sha256=source_hash,
        resume_source_binding=source_binding,
        formal_binding=formal_binding,
        reference=reference,
        left=replicas[0],
        right=replicas[1],
    )
    comparison["sha256"] = sha(canonical(comparison))
    require(
        comparison.get("passed") is True
        and comparison.get("checks")
        and all(value is True for value in comparison["checks"].values()),
        "exact resume comparison failed",
    )
    return comparison


def validate_resume_report(path, *, config, expected_world_size, expected_formal_binding):
    """Replay every original comparison check, including after workspace relocation."""
    path = absolute(str(path))
    report = document(path)
    require(
        report.get("schema") == REPORT_SCHEMA
        and type(expected_world_size) is int
        and expected_world_size == 2
        and type(report.get("world_size")) is int
        and report.get("world_size") == 2,
        "wrong resume report/world size",
    )
    require(
        report.get("sha256") == sha(canonical({k: v for k, v in report.items() if k != "sha256"})),
        "resume report SHA mismatch",
    )
    require(
        report.get("passed") is True
        and type(report.get("exit_code")) is int
        and report.get("exit_code") == 0
        and all(report.get(key) is False for key in FALSE_CLAIMS),
        "failed or overstated resume report",
    )
    require(
        report.get("runtime_config_sha256") == sha(canonical(config))
        and report.get("source_workspace") == config.get("output_root"),
        "resume config binding changed",
    )
    require(
        file_identity(path.parent / STEP20)["sha256"]
        == report.get("comparison", {}).get("resume_source", {}).get("checkpoint", {}).get("sha256"),
        "step20 bytes changed after comparison",
    )
    expected = compare_runs(path.parent, config, expected_formal_binding)
    require(
        canonical(report.get("comparison")) == canonical(expected),
        "resume report differs from revalidated artifacts",
    )
    return expected


def run_foreground(argv, *, cwd, environment, log):
    """Inherit the outer batch wrapper's process group; forward signals and await cleanup."""
    with log.open("xb") as stream:
        process = subprocess.Popen(
            argv, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=False
        )
        previous = {}

        def interrupted(number, _frame):
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(number)
            raise InterruptedError(f"resume interrupted by signal {number}")

        try:
            for number in (signal.SIGTERM, signal.SIGINT):
                previous[number] = signal.signal(number, interrupted)
            require(process.wait() == 0, "resume command failed; inspect " + str(log))
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def training_commands(request, reference_report, source):
    overrides = reference_report["plan"]["training_overrides"]
    require(
        isinstance(overrides, list) and all(isinstance(x, str) and not x.startswith("-") for x in overrides),
        "invalid scientific override vector",
    )
    science = Path(request["readonly_binds"]["science"]["guest"])
    common = [
        request["python"],
        "-B",
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(science / "configs/accelerate/fsdp_server_scheduler.yaml"),
        "--num_processes",
        "2",
        "--num_cpu_threads_per_process",
        "12",
        "--main_process_port",
        "0",
        "-m",
        "posttrain_circuits.cli.train",
        *overrides,
        "--resume",
        str(source / STEP20),
    ]
    return [
        [*common, "--output", str(source / name), "--confirm-production"] for name in ("resume-a", "resume-b")
    ]


def inside(plan):
    request, source = plan["request"], absolute(plan["source_workspace"])
    workspace = Path(request["workspace"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan["cuda_visible_devices"], "CUDA visibility changed")
    require(os.environ.get("SLURM_JOB_ID") == request["target"]["job_id"], "Slurm job identity changed")
    probe = REFERENCE + "/manifest.json"
    require(
        os.path.samestat((workspace / probe).stat(), (source / probe).stat()),
        "private namespace does not expose the original path to the staged inode",
    )
    for name, identity in plan["inputs"].items():
        relative(name)
        require(file_identity(source / name) == identity, "namespace input bytes differ")
    require(
        Path(sys.executable).resolve() == Path(request["python"]) and platform.python_version() == "3.12.13",
        "namespace did not execute the pinned host Conda Python",
    )
    require(
        file_identity(Path(sys.executable))["sha256"] == request["python_sha256"], "namespace Python changed"
    )
    reference_report = document(Path(request["reference_report"]), request["reference_report_sha256"])
    science = Path(request["readonly_binds"]["science"]["guest"])
    hf_home = Path(request["readonly_binds"]["hf_home"]["guest"])
    bindings = helper("sdsc_science_binding")
    os.chdir(science)
    base, projection, accepted = bindings.reviewed_science(
        science, request["expected_science_head"], request["expected_protocol_sha256"]
    )
    compose = importlib.import_module("posttrain_circuits.core.config").compose_config
    config = compose(reference_report["plan"]["training_overrides"], config_root=science / "configs")
    require(
        config["output_root"] == str(source)
        and sha(canonical(config)) == reference_report["runtime_config_sha256"],
        "reference config differs",
    )
    bindings.verify_runtime_binding(
        base,
        config,
        expected_base_science_sha256=accepted.binding.storage_neutral_resolved_config_sha256,
        expected_checkpoint_sha256=reference_report["initial_checkpoint"]["sha256"],
        checkpoint_path=source / "initial_checkpoint.pt",
        science_projection=projection,
    )
    formal = importlib.import_module("posttrain_circuits.artifacts.runs").formal_artifact_binding(config)
    require(
        canonical(formal) == canonical(reference_report["formal_binding"]), "formal producer binding changed"
    )
    guards, calibration = helper("sdsc_training_preflight"), helper("sdsc_g0_calibration")
    runtime = guards.runtime_identity()
    lock = document(science / "deployments/qwen3_v2_g0/dependency-lock.json")
    packages = {row["name"]: importlib.metadata.version(row["name"]) for row in lock["packages"]}
    require(packages == {row["name"]: row["version"] for row in lock["packages"]}, "G0 dependencies differ")
    require(
        Path(sys.prefix).resolve() == Path(request["python"]).parent.parent, "using container Python prefix"
    )
    import torch

    require(
        Path(sys.prefix) in Path(torch.__file__).resolve().parents,
        "using container Torch instead of host runtime",
    )
    allocation = calibration.allocation_identity()
    gpu = calibration.gpu_identity()
    cache = guards.pinned_cache(hf_home)
    memory = guards.memory_envelope()
    api = importlib.import_module("posttrain_circuits.cli.compare_distributed_resume")
    reference = api.validate_factorial_run_artifacts(
        source / REFERENCE,
        expected_world_size=2,
        expected_resolved_config=config,
        expected_code_commit=formal["code_commit"],
        expected_resume_ancestry=[],
    )
    source_binding = api._validate_resume_source_artifacts(
        source / STEP20,
        source_sha256=plan["inputs"][STEP20]["sha256"],
        reference=reference,
        config=config,
        world_size=2,
    )
    require(source_binding["token_budget"].get("stop_reason") is None, "step20 checkpoint is terminal")
    del reference
    import gc

    gc.collect()
    environment = training_environment(request, science, hf_home, plan["cuda_visible_devices"], os.environ)
    report = dict(
        schema=REPORT_SCHEMA,
        task="qwen3-v2-g0-resume",
        target=request["target"],
        request_sha256=plan["request_sha256"],
        python_sha256=request["python_sha256"],
        science_git_head=request["expected_science_head"],
        world_size=2,
        source_workspace=str(source),
        runtime_config_sha256=sha(canonical(config)),
        reference_report_sha256=request["reference_report_sha256"],
        publication_receipt_sha256=request["publication_receipt_sha256"],
        namespace_binds=plan["binds"],
        container=request["container"],
        runtime=runtime,
        packages=packages,
        allocation=allocation,
        gpu=gpu,
        pinned_cache=cache,
        initial_cgroup_memory=memory,
        inputs=plan["inputs"],
        passed=False,
        exit_code=1,
        persistent_storage_publication="caller_required",
        **FALSE_CLAIMS,
    )
    try:
        for name in ("resume-a", "resume-b", "resume-a.log", "resume-b.log"):
            require(not (source / name).exists(), "resume output already exists; implicit retry forbidden")
        for name, command in zip(
            ("resume-a", "resume-b"), training_commands(request, reference_report, source), strict=True
        ):
            require(not (source / name).exists(), "resume output already exists; implicit retry forbidden")
            report["phase"] = name
            publish(source / "sdsc-resume-progress.json", report)
            run_foreground(command, cwd=science, environment=environment, log=source / (name + ".log"))
            api.validate_factorial_run_artifacts(
                source / name,
                expected_world_size=2,
                expected_resolved_config=config,
                expected_code_commit=formal["code_commit"],
                expected_resume_ancestry=["sha256:" + plan["inputs"][STEP20]["sha256"]],
                expected_update_norm_baseline_checkpoint_path=source / STEP20,
            )
        report["comparison"] = compare_runs(source, config, formal)
        report["final_cgroup_memory"] = guards.memory_envelope()
        require(
            helper("sdsc_teacher_prepare").git(science, "rev-parse", "HEAD").decode().strip()
            == request["expected_science_head"],
            "scientific HEAD changed",
        )
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == plan["cuda_visible_devices"], "CUDA visibility changed"
        )
        report.update(passed=True, exit_code=0, phase="compared")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["sha256"] = sha(canonical(report))
        publish(source / "distributed_resume.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inside-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inside-plan-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inside_plan is not None:
        require(
            args.request is None and not args.execute and SHA.fullmatch(args.inside_plan_sha256 or ""),
            "invalid namespace invocation",
        )
        inside(document(args.inside_plan, args.inside_plan_sha256))
        return 0
    require(
        args.request is not None and SHA.fullmatch(args.request_sha256 or ""),
        "externally bound request required",
    )
    request = document(args.request, args.request_sha256)
    paths = validate_request(request)
    reference, source, inputs = input_inventory(request)
    for name in (
        "resume-a",
        "resume-b",
        "resume-a.log",
        "resume-b.log",
        "distributed_resume.json",
        "sdsc-resume-progress.json",
    ):
        require(not (paths["workspace"] / name).exists(), "workspace already used for resume")
    plan_path = paths["work_dir"] / "sdsc-resume-control/plan.json"
    plan = {
        "request": request,
        "request_sha256": args.request_sha256,
        "source_workspace": str(source),
        "inputs": inputs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    _, plan["binds"] = build_namespace_command(request, source, plan_path, "0" * 64)
    command, _ = build_namespace_command(request, source, plan_path, sha(canonical(plan) + b"\n"))
    if not args.execute:
        print(json.dumps({"validation_only": True, "command": command, "plan": plan, **FALSE_CLAIMS}))
        return 0
    environment = container_environment(request, os.environ)
    mount = helper("sdsc_training_preflight").local_paths(paths["work_dir"], paths["workspace"])
    require(mount["fstype"] in {"xfs", "ext4", "ext3", "ext2", "btrfs"}, "scratch mount is not local")
    plan_path.parent.mkdir()  # Atomic single-use claim. Never reuse after interruption.
    for name in ("resume-tmp", "resume-home"):
        (paths["work_dir"] / name).mkdir()
    publish(plan_path, plan)
    try:
        run_foreground(
            command,
            cwd=paths["science_root"],
            environment=environment,
            log=plan_path.parent / "container.log",
        )
        # Bind inputs again after execution; publication remains the enclosing wrapper's responsibility.
        require(input_inventory(request)[2] == inputs, "original inputs changed during resume")
        report = document(paths["workspace"] / "distributed_resume.json")
        require(report.get("passed") is True and report.get("exit_code") == 0, "resume did not pass")
    except BaseException as error:
        publish(
            plan_path.parent / "failure.json",
            {
                "target": request["target"],
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
                **FALSE_CLAIMS,
            },
        )
        raise
    print(
        json.dumps(
            {
                "passed": True,
                "report": str(paths["workspace"] / "distributed_resume.json"),
                "persistent_storage_publication": "caller_required",
                **FALSE_CLAIMS,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
