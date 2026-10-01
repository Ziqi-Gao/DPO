#!/usr/bin/env python3
"""Node-local staging and durable publication for one inference diagnostic."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GIB = 1024**3


def memory_envelope(control, job, *, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    """Reuse the frozen measured 64-GiB envelope, additionally reject own-job OOM."""
    prior = control.helper("sdsc_student_instruction_job")
    try:
        evidence = prior.memory_envelope(control, proc=proc, root=root)
    except prior.MemoryEnvelopeError:
        raise
    try:
        limiting = [row for row in evidence["ancestors"] if row.get("limit_bytes") == 64 * GIB]
        control.require(
            limiting and all("job_" + job in Path(row["path"]).parts for row in limiting),
            "limiting memory cgroup must belong to this job",
        )
        control.validate_job_memory_events(evidence, job)
    except ValueError as error:
        evidence["passed"] = False
        evidence["error"] = str(error)
        raise prior.MemoryEnvelopeError(str(error), evidence) from error
    return evidence


def load_control(plan, plan_sha):
    raw = Path(plan).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan_sha:
        raise ValueError("node plan changed")
    value = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_initial.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != value["control_sha256"]["tools/sdsc_student_initial.py"]
    ):
        raise ValueError("node controller changed")
    spec = importlib.util.spec_from_file_location("_quality_node_control", source)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    return control, control.validate_plan(value)


def allocation(plan, control):
    env = os.environ
    job = env.get("SLURM_JOB_ID", "")
    expected = dict(
        SLURM_CPUS_PER_TASK="8",
        SLURM_MEM_PER_NODE="65536",
        SLURM_JOB_PARTITION="nairr-gpu-shared",
        SLURM_JOB_ACCOUNT="nwu181",
        SLURM_NTASKS="1",
        SLURM_JOB_NUM_NODES="1",
        SLURM_JOB_NAME=plan["job_name"],
    )
    control.require(
        re.fullmatch("[1-9][0-9]*", job) and all(env.get(k) == v for k, v in expected.items()),
        "actual diagnostic Slurm allocation differs",
    )
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    control.require(
        bool(visible)
        and "," not in visible
        and visible not in ("-1", "NoDevFiles")
        and visible.strip() == visible,
        "exactly one assigned GPU is required",
    )
    return dict(
        job_id=job,
        cpus=8,
        memory_mib=65536,
        world_size=1,
        cuda_visible_devices=visible,
        account="nwu181",
        partition="nairr-gpu-shared",
    )


def mount(path, control):
    result = control.run(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"], 10
    )
    control.require(result["returncode"] == 0, "mount unavailable on allocated node")
    return json.loads(result["stdout"])["filesystems"][0]


def persist(plan, control, artifacts, log, result, memory, *, final_measure=None):
    """Seal small immutable results only after durable readback and final measurement."""
    destination = control.safe(plan["result_dir"])
    control.require(not (destination / "receipt.json").exists(), "diagnostic already published")
    records, total = [], 0

    def publish(name, raw):
        nonlocal total
        total += len(raw)
        control.require(total <= control.MAX_FETCH - 256 * 1024, "publication exceeds bound")
        control.write_once(destination / name, raw)
        control.require(control.read(destination / name) == raw, "persistent readback differs")
        records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))

    if artifacts is not None:
        for name in control.NAMES:
            if name in {"node-result.json", "memory.json"}:
                continue
            path = artifacts / name
            if path.exists():
                publish(name, control.read(path))
    if log is not None and log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 64 * 1024))
            control.write_once(destination / "worker.log", stream.read(64 * 1024))
    if final_measure is not None:
        try:
            final_measure("after_publication")
        except Exception as error:
            result.update(diagnostic_complete=False, exit_code=1, memory_publication_error=str(error))
    for name, value in [("node-result.json", result), ("memory.json", memory)]:
        publish(name, control.canonical(value))
    publication = dict(
        task=control.TASK,
        job_id=result["job_id"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=result["diagnostic_complete"] and result["exit_code"] == 0,
        diagnostic_complete=result["diagnostic_complete"],
        persistent_read_back_verified=True,
        files=records,
        **dict.fromkeys(control.FLAGS, False),
    )
    raw = control.canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "receipt readback differs")


def wait_worker(process, *, timeout, artifacts, control, measure, result_root):
    """Observe bounded progress from this one child, without retry or scoring."""
    deadline = time.monotonic() + timeout
    completed = -1
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            process.wait(timeout=min(30, remaining))
            return
        except subprocess.TimeoutExpired:
            measure("running-" + str(int(time.monotonic())))
            path = artifacts / "progress.json"
            if not path.exists():
                continue
            raw = control.read(path, 4096)
            progress = json.loads(raw)
            count = progress.get("completed_examples")
            control.require(
                progress.get("schema") == "quest-sdsc-student-initial-progress-v1"
                and progress.get("phase") == "initial_native_scoring"
                and progress.get("total_examples") == 128
                and type(count) is int
                and completed <= count <= 128
                and all(progress.get(k) is False for k in control.FLAGS),
                "invalid native initial progress",
            )
            if count > completed:
                print(json.dumps(dict(phase="initial_native_scoring", completed_examples=count)), flush=True)
                control.write_once(result_root / ("progress-" + str(count) + ".json"), raw)
                completed = count


def group_has_live_members(pgid, proc=Path("/proc")):
    """Only the launcher's new session/group; zombies cannot mutate artifacts."""
    for item in proc.iterdir():
        if not item.name.isdecimal():
            continue
        try:
            fields = (item / "stat").read_text().rsplit(") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        # /proc/PID/stat fields 3, 5 and 6: state, group, session.
        if int(fields[2]) == pgid and int(fields[3]) == pgid and fields[0] not in {"Z", "X"}:
            return True
    return False


def stop_worker_group(process, *, alive=group_has_live_members):
    """Reap this launcher's group even after its leader has already exited."""
    pgid = process.pid
    for number, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        if not alive(pgid):
            process.wait(timeout=1)
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, number)
        deadline = time.monotonic() + timeout
        while alive(pgid) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.2)
    if alive(pgid):
        raise RuntimeError("own worker group still live; cannot publish mutable artifacts")
    process.wait(timeout=1)


def start_worker(plan, source, science, inputs, artifacts, environment, stream):
    return subprocess.Popen(
        [
            plan["python"],
            "-B",
            "-u",
            str(source / "tools/sdsc_student_initial_probe.py"),
            "--inputs-json",
            str(inputs),
            "--output-dir",
            str(artifacts),
        ],
        cwd=science,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def build_worker_inputs(plan, work, science, identity, config, control):
    return dict(
        schema="quest-sdsc-student-initial-inputs-v1",
        initial_checkpoint=dict(
            path=str(work / "checkpoints/initial.pt"),
            size=plan["checkpoints"][0]["size"],
            sha256=control.INITIAL_SHA,
        ),
        resolved_config=dict(
            path=str(work / "config/resolved_config.yaml"), size=config["size"], sha256=config["sha256"]
        ),
        dataset_root=str(work / "dataset"),
        dataset_manifest_sha256=next(
            row["sha256"] for row in plan["dataset_inputs"] if row["path"] == "manifest.json"
        ),
        science_root=str(science),
        hf_home=str(work / "huggingface"),
        job_id=identity["job_id"],
        run_id=plan["run_id"],
        parent_job_id=control.PARENT_JOB,
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
    )


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: node PLAN PLAN_SHA256")
    control, plan = load_control(*args)
    identity = allocation(plan, control)
    control.require(
        control.sha(control.read(Path(__file__)))
        == plan["control_sha256"]["tools/sdsc_student_initial_job.py"],
        "node wrapper changed",
    )
    control.verify_source(plan)
    control.verify_parent(plan)
    project = control.PROJECT
    source_root = Path(plan["parent"]["receipt"]["result_dir"])
    result_root = control.safe(plan["result_dir"])
    parent = result_root.parent
    while not parent.exists():
        parent = parent.parent
    input_mount = mount(source_root, control)
    output_mount = mount(parent, control)
    cache_mount = mount(Path(plan["hf_home"]), control)
    control.require(
        all(
            m["fstype"] in {"lustre", "nfs", "nfs4", "ceph"} for m in (input_mount, output_mount, cache_mount)
        )
        and project in result_root.parents,
        "GPU node has not confirmed persistent input/output/cache mounts",
    )
    result_root.parent.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(mode=0o700)
    work = artifacts = log = process = None
    memory = {}
    started = time.monotonic()
    result = dict(
        identity,
        task=control.TASK,
        run_id=plan["run_id"],
        parent_job_id=control.PARENT_JOB,
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
        diagnostic_complete=False,
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
        mounts=dict(input=input_mount, output=output_mount, cache=cache_mount),
    )

    def measure(stage):
        try:
            memory[stage] = memory_envelope(control, identity["job_id"])
            raw = control.canonical(memory[stage])
            control.write_once(result_root / ("memory-" + stage + ".json"), raw)
            control.require(
                control.read(result_root / ("memory-" + stage + ".json")) == raw,
                "memory evidence persistent readback differs",
            )
        except Exception as error:
            if hasattr(error, "evidence"):
                memory[stage] = error.evidence
            raise

    def interrupted(number, _frame):
        raise RuntimeError("diagnostic interrupted by signal " + str(number))

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    try:
        measure("initial")
        scratch = control.safe(
            os.environ.get("TMPDIR")
            or "/scratch/{}/job_{}".format(pwd.getpwuid(os.getuid()).pw_name, identity["job_id"])
        )
        control.require(
            scratch.is_dir() and scratch.stat().st_uid == os.getuid(), "owned node-local scratch absent"
        )
        local_mount = mount(scratch, control)
        control.require(
            local_mount["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"}
            and shutil.disk_usage(scratch).free >= 32 * 1024**3,
            "node-local filesystem/free-space requirement failed",
        )
        work = Path(tempfile.mkdtemp(prefix="opd-student-initial-" + identity["job_id"] + "-", dir=scratch))
        result.update(work_dir=str(work), local_mount=local_mount)
        source = work / "source"
        staging = control.helper("sdsc_student_job")
        staging.code_hash = plan["code_sha256"]
        staging.run_id = plan["run_id"]
        staging.stage_source(Path(plan["release"]), source)
        contract = control.helper("sdsc_student_contract", source)
        verifier = control.helper("sdsc_provenance", source)
        science = work / "science"
        parent_receipt = plan["parent"]["receipt"]
        proof = contract.document(
            parent_receipt["prerequisites_path"], parent_receipt["prerequisites_sha256"]
        )
        restored = verifier.verify(
            control.safe(parent_receipt["provenance_dir"]),
            parent_receipt["provenance_manifest_sha256"],
            parent_receipt["code_sha256"],
            science,
        )
        protocol = contract.resolved_protocol(science, parent_receipt["science_git_head"])
        control.require(protocol == proof["protocol"], "restored parent science protocol differs")
        for name, digest in protocol["science_file_sha256"].items():
            control.require(
                staging.file_hash(control.safe(science / name)) == digest,
                "restored scientific byte differs: " + name,
            )
        result["science_restore"] = dict(
            head=restored["git_head"],
            bundle_sha256=restored["bundle_sha256"],
            protocol_sha256=protocol["protocol_sha256"],
            science_file_sha256=protocol["science_file_sha256"],
            original_implementation_only=True,
        )
        checkpoint_rows = [
            dict(row, source=str(source_root / row["path"]), path=row["label"] + ".pt")
            for row in plan["checkpoints"]
        ]
        result["checkpoint_staging"] = contract.stage_inputs(
            checkpoint_rows, work / "checkpoints", maximum=4 * 1024**3
        )
        config = plan["resolved_config"]
        result["config_staging"] = contract.stage_inputs(
            [dict(config, source=str(source_root / config["path"]), path="resolved_config.yaml")],
            work / "config",
            maximum=control.CAP,
        )
        result["dataset_staging"] = contract.stage_inputs(
            plan["dataset_inputs"], work / "dataset", maximum=2 * 1024**3
        )
        # Reuse the reviewed confined cache copier for only the pinned 1.7B snapshot.
        staging.models = (("models--Qwen--Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),)
        result["model_staging"] = staging.stage_models(Path(plan["hf_home"]), work / "huggingface")
        result["model_files"] = [
            dict(
                path=str(path.relative_to(work / "huggingface")),
                size=path.stat().st_size,
                sha256=staging.file_hash(path),
            )
            for path in sorted((work / "huggingface").rglob("*"))
            if path.is_file()
        ]
        measure("after_staging")
        artifacts = work / "artifacts"
        artifacts.mkdir()
        inputs = build_worker_inputs(plan, work, science, identity, config, control)
        control.write_once(work / "inputs.json", control.canonical(inputs))
        environment = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH"):
            environment.pop(key, None)
        environment.update(
            PYTHONPATH=str(science / "src"),
            HF_HOME=str(work / "huggingface"),
            HF_HUB_CACHE=str(work / "huggingface/hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="8",
            MKL_NUM_THREADS="8",
            OPENBLAS_NUM_THREADS="8",
            NUMEXPR_NUM_THREADS="8",
            HF_DATASETS_OFFLINE="1",
            TMPDIR=str(work),
            XDG_CACHE_HOME=str(work / "cache"),
        )
        log = work / "worker.log"
        with log.open("xb") as stream:
            process = start_worker(
                plan, source, science, work / "inputs.json", artifacts, environment, stream
            )
            print(json.dumps(dict(phase="initial_worker_started", job_id=identity["job_id"])), flush=True)
            wait_worker(
                process,
                timeout=max(1, 1560 - int(time.monotonic() - started)),
                artifacts=artifacts,
                control=control,
                measure=measure,
                result_root=result_root,
            )
        control.require(
            process.returncode == 0, "inference worker failed; preserve outputs, never retry automatically"
        )
        measure("after_inference")
        report = control.document(artifacts / "initial-probe.json")
        control.require(
            report.get("diagnostic_complete") is True
            and report.get("passed") is True
            and all(report.get(k) is False for k in control.FLAGS),
            "worker incomplete or claims scientific acceptance",
        )
        raw_records = {}
        for name in ("initial-records.jsonl", "initial-prompts.jsonl"):
            raw = control.read(artifacts / name)
            raw_records[name] = dict(path=name, size=len(raw), sha256=control.sha(raw))
        control.validate_worker_report(report, plan, identity["job_id"], raw_records)
        control.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"],
            "assigned CUDA visibility changed",
        )
        result.update(diagnostic_complete=True, exit_code=0)
    except BaseException as error:
        result.update(error=type(error).__name__ + ": " + str(error)[:1500], exit_code=1)
    finally:
        if process is not None:
            try:
                stop_worker_group(process)
            except Exception as error:
                result.update(diagnostic_complete=False, exit_code=1, worker_shutdown_error=str(error))
                artifacts = None  # Never seal evidence still mutable by a live child.
        try:
            measure("final")
        except Exception as error:
            result.update(diagnostic_complete=False, exit_code=1, memory_error=str(error))
        result["elapsed_seconds"] = time.monotonic() - started
        persist(plan, control, artifacts, log, result, memory, final_measure=measure)
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
