#!/usr/bin/env python3
"""Node-local staging and durable publication for one inference diagnostic."""

from __future__ import annotations

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


def load_control(plan, plan_sha):
    raw = Path(plan).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan_sha:
        raise ValueError("node plan changed")
    value = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_quality.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != value["control_sha256"]["tools/sdsc_student_quality.py"]
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
        SLURM_CPUS_PER_TASK="24",
        SLURM_MEM_PER_NODE="196608",
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
        cpus=24,
        memory_mib=196608,
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


def persist(plan, control, artifacts, log, result, memory):
    """Publish stopped-worker outputs exactly; a failed diagnostic remains failed."""
    destination = control.safe(plan["result_dir"])
    control.require(not (destination / "receipt.json").exists(), "diagnostic already published")
    if artifacts is not None:
        for name, value in [("node-result.json", result), ("memory.json", memory)]:
            control.write_once(artifacts / name, control.canonical(value))
    records = []
    total = 0
    for name in control.NAMES:
        path = artifacts / name if artifacts is not None else None
        if path is None or not path.exists():
            continue
        raw = control.read(path)
        total += len(raw)
        control.require(
            total <= control.MAX_FETCH - 256 * 1024, "diagnostic output exceeds bounded publication"
        )
        control.write_once(destination / name, raw)
        control.require(control.read(destination / name) == raw, "persistent diagnostic readback differs")
        records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    if artifacts is None:
        for name, value in [("node-result.json", result), ("memory.json", memory)]:
            raw = control.canonical(value)
            control.write_once(destination / name, raw)
            records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    if log is not None and log.exists():
        # A bounded log tail is sufficient; raw scientific JSONL remains whole.
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 64 * 1024))
            control.write_once(destination / "worker.log", stream.read(64 * 1024))
    publication = dict(
        task=control.TASK,
        job_id=result["job_id"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=result["diagnostic_complete"],
        diagnostic_complete=result["diagnostic_complete"],
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        persistent_read_back_verified=True,
        files=records,
    )
    raw = control.canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "publication receipt readback differs")


def wait_worker(process, *, timeout, artifacts, control):
    """Wait once on the same child, exposing only completed-arm summaries."""
    deadline = time.monotonic() + timeout
    observed_arms = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            process.wait(timeout=min(30, remaining))
            return
        except subprocess.TimeoutExpired:
            progress_path = artifacts / "progress.json"
            if not progress_path.exists():
                continue
            progress = control.document(progress_path)
            arms = progress.get("arms", [])
            control.require(isinstance(arms, list) and len(arms) <= 12, "invalid diagnostic progress")
            if len(arms) > observed_arms:
                arm = arms[-1]
                print(
                    json.dumps(
                        dict(
                            phase="quality_progress",
                            completed_arms=len(arms),
                            checkpoint=arm["checkpoint"],
                            cohort=arm["cohort"],
                            cap=arm["cap"],
                            metrics=arm["metrics"],
                            teacher_forced=arm["teacher_forced"],
                        ),
                        sort_keys=True,
                    ),
                    flush=True,
                )
                observed_arms = len(arms)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: node PLAN PLAN_SHA256")
    control, plan = load_control(*args)
    identity = allocation(plan, control)
    control.require(
        control.sha(control.read(Path(__file__)))
        == plan["control_sha256"]["tools/sdsc_student_quality_job.py"],
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
        mounts=dict(input=input_mount, output=output_mount, cache=cache_mount),
    )
    memory_helper = control.helper("sdsc_student_memory")

    def measure(stage):
        try:
            memory[stage] = memory_helper.memory_envelope(expected_bytes=192 * 1024**3)
        except memory_helper.MemoryEnvelopeError as error:
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
            and shutil.disk_usage(scratch).free >= 96 * 1024**3,
            "node-local filesystem/free-space requirement failed",
        )
        work = Path(tempfile.mkdtemp(prefix="opd-student-quality-" + identity["job_id"] + "-", dir=scratch))
        result.update(work_dir=str(work), local_mount=local_mount)
        source = work / "source"
        staging = control.helper("sdsc_student_job")
        staging.code_hash = plan["code_sha256"]
        staging.run_id = plan["run_id"]
        staging.stage_source(Path(plan["release"]), source)
        contract = control.helper("sdsc_student_contract", source)
        checkpoint_rows = [
            dict(row, source=str(source_root / row["path"]), path=row["label"] + ".pt")
            for row in plan["checkpoints"]
        ]
        result["checkpoint_staging"] = contract.stage_inputs(
            checkpoint_rows, work / "checkpoints", maximum=64 * 1024**3
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
        inputs = dict(
            schema="quest-sdsc-student-quality-inputs-v1",
            checkpoints=[
                dict(
                    label=row["label"],
                    path=str(work / "checkpoints" / (row["label"] + ".pt")),
                    sha256=row["sha256"],
                    size=row["size"],
                )
                for row in plan["checkpoints"]
            ],
            resolved_config=str(work / "config/resolved_config.yaml"),
            dataset_root=str(work / "dataset"),
            hf_home=str(work / "huggingface"),
            job_id=identity["job_id"],
            run_id=plan["run_id"],
            parent_job_id=control.PARENT_JOB,
            source_code_sha256=plan["code_sha256"],
            parent_report_sha256=plan["parent"]["report_sha256"],
            parent_publication_sha256=plan["parent"]["publication_sha256"],
            persistent_result_dir=plan["result_dir"],
        )
        control.write_once(work / "inputs.json", control.canonical(inputs))
        environment = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH"):
            environment.pop(key, None)
        environment.update(
            PYTHONPATH=str(source / "src"),
            HF_HOME=str(work / "huggingface"),
            HF_HUB_CACHE=str(work / "huggingface/hub"),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="24",
            MKL_NUM_THREADS="24",
            OPENBLAS_NUM_THREADS="24",
            NUMEXPR_NUM_THREADS="24",
            TMPDIR=str(work),
            XDG_CACHE_HOME=str(work / "cache"),
        )
        log = work / "worker.log"
        with log.open("xb") as stream:
            process = subprocess.Popen(
                [
                    plan["python"],
                    "-B",
                    "-u",
                    str(source / "tools/sdsc_student_quality_probe.py"),
                    "--inputs-json",
                    str(work / "inputs.json"),
                    "--output-dir",
                    str(artifacts),
                ],
                cwd=source,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(json.dumps(dict(phase="quality_worker_started", job_id=identity["job_id"])), flush=True)
            wait_worker(
                process,
                timeout=max(1, 6600 - int(time.monotonic() - started)),
                artifacts=artifacts,
                control=control,
            )
        control.require(
            process.returncode == 0, "inference worker failed; preserve outputs, never retry automatically"
        )
        measure("after_inference")
        report = control.document(artifacts / "quality-probe.json")
        control.require(
            report.get("diagnostic_complete") is True
            and report.get("passed") is True
            and all(
                report.get(k) is False
                for k in ("student_accepted", "g0_passed", "pilot_passed", "factorial_ready")
            ),
            "worker incomplete or claims scientific acceptance",
        )
        raw_records = {}
        for name in ("quality-records.jsonl", "quality-prompts.jsonl"):
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
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        try:
            measure("final")
        except Exception as error:
            result.update(diagnostic_complete=False, exit_code=1, memory_error=str(error))
        result["elapsed_seconds"] = time.monotonic() - started
        persist(plan, control, artifacts, log, result, memory)
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
