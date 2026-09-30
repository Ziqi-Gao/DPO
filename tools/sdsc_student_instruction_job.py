#!/usr/bin/env python3
"""Node-local staging and durable publication for one inference diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
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


class MemoryEnvelopeError(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def memory_envelope(control, *, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    expected_bytes = 64 * 1024**3
    _counters = control.helper("sdsc_student_memory")._counters
    evidence = {
        "expected_limit_bytes": expected_bytes,
        "observed_at_unix": time.time(),
        "ancestors": [],
        "passed": False,
    }
    try:

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        require(
            type(expected_bytes) is int and expected_bytes == 64 * GIB, "unreviewed student memory envelope"
        )
        locations = []
        for line in proc.read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            require(relative.startswith("/") and ".." not in Path(relative).parts, "invalid cgroup path")
            if hierarchy == "0" and not controllers:
                locations.append(
                    (root, root / relative.lstrip("/"), ("memory.max", "memory.current", "memory.peak"), 2)
                )
            elif "memory" in controllers.split(","):
                locations.insert(
                    0,
                    (
                        root / "memory",
                        root / "memory" / relative.lstrip("/"),
                        ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.max_usage_in_bytes"),
                        1,
                    ),
                )
        require(locations, "no readable memory cgroup")
        boundary, directory, names, version = locations[0]
        evidence["cgroup_version"] = version
        levels = evidence["ancestors"]
        while True:
            level = {"path": str(directory), "limit_bytes": None}
            levels.append(level)
            if (directory / names[0]).is_file():
                raw = (directory / names[0]).read_text().strip()
                limit = None if raw == "max" else int(raw)
                level["raw_limit"] = raw
                require(limit is None or limit > 0, "invalid cgroup memory limit")
                level["limit_bytes"] = limit if limit is not None and limit < 2**60 else None
                for name, key in zip(names[1:], ("current_bytes", "peak_bytes"), strict=True):
                    if (directory / name).is_file():
                        level[key] = int((directory / name).read_text().strip())
                level["memory_stat"] = _counters(directory / "memory.stat")
                level["memory_events"] = _counters(directory / "memory.events")
                level["memory_events_local"] = _counters(directory / "memory.events.local")
                level["memory_oom_control"] = _counters(directory / "memory.oom_control")
                failcnt = directory / "memory.failcnt"
                level["memory_failcnt"] = int(failcnt.read_text().strip()) if failcnt.is_file() else None
            if directory == boundary:
                break
            require(boundary in directory.parents, "memory cgroup escaped controller root")
            directory = directory.parent
        finite = [level for level in levels if level["limit_bytes"] is not None]
        require(finite, "cgroup must expose a finite student memory limit")
        limit = min(level["limit_bytes"] for level in finite)
        limiting = [level for level in finite if level["limit_bytes"] == limit]
        require(
            all("current_bytes" in level and "peak_bytes" in level for level in limiting),
            "limiting memory cgroup must expose current and peak usage",
        )
        selected = max(limiting, key=lambda level: level["peak_bytes"])
        current, peak = selected["current_bytes"], selected["peak_bytes"]
        required = max(32 * GIB, math.ceil(limit * 0.20))
        evidence.update(
            path=selected["path"],
            limit_bytes=limit,
            current_bytes=current,
            peak_bytes=peak,
            headroom_bytes=limit - peak,
            minimum_headroom_bytes=required,
            memory_stat=selected.get("memory_stat"),
            memory_events=selected.get("memory_events"),
            memory_events_local=selected.get("memory_events_local"),
            memory_oom_control=selected.get("memory_oom_control"),
            memory_failcnt=selected.get("memory_failcnt"),
        )
        require(
            limit == expected_bytes
            and all(0 < item["current_bytes"] <= item["peak_bytes"] <= limit for item in limiting),
            "cgroup limit or measured peak differs from reviewed student memory envelope",
        )
        require(
            limit - peak >= required,
            f"{limit // GIB} GiB cgroup lacks 32 GiB and 20% headroom: "
            f"peak={peak}, current={current}, required={required}",
        )
        evidence["passed"] = True
        return evidence
    except (ValueError, OSError) as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise MemoryEnvelopeError(str(error), evidence) from error


def load_control(plan, plan_sha):
    raw = Path(plan).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan_sha:
        raise ValueError("node plan changed")
    value = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_instruction.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != value["control_sha256"]["tools/sdsc_student_instruction.py"]
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
            control.require(control.read(destination / name) == raw, "persistent diagnostic readback differs")
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
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
        persistent_read_back_verified=True,
        files=records,
    )
    raw = control.canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "publication receipt readback differs")


def wait_worker(process, *, timeout, artifacts, control, measure, result_root):
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
            measure("running-" + str(int(time.monotonic())))
            progress_path = artifacts / "progress.json"
            if not progress_path.exists():
                continue
            progress = control.document(progress_path)
            arms = progress.get("arms", [])
            control.require(isinstance(arms, list) and len(arms) <= 2, "invalid diagnostic progress")
            if len(arms) > observed_arms:
                arm = arms[-1]
                print(
                    json.dumps(
                        dict(
                            phase="instruction_progress",
                            completed_arms=len(arms),
                            arm=arm["arm"],
                            metrics=arm["metrics"],
                            teacher_forced=arm["teacher_forced"],
                        ),
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raw = control.read(progress_path)
                path = result_root / ("progress-arm-" + str(len(arms)) + ".json")
                control.write_once(path, raw)
                control.require(control.read(path) == raw, "progress persistent readback differs")
                observed_arms = len(arms)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: node PLAN PLAN_SHA256")
    control, plan = load_control(*args)
    identity = allocation(plan, control)
    control.require(
        control.sha(control.read(Path(__file__)))
        == plan["control_sha256"]["tools/sdsc_student_instruction_job.py"],
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
            memory[stage] = memory_envelope(control)
            raw = control.canonical(memory[stage])
            control.write_once(result_root / ("memory-" + stage + ".json"), raw)
            control.require(
                control.read(result_root / ("memory-" + stage + ".json")) == raw,
                "memory evidence persistent readback differs",
            )
        except MemoryEnvelopeError as error:
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
        work = Path(
            tempfile.mkdtemp(prefix="opd-student-instruction-" + identity["job_id"] + "-", dir=scratch)
        )
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
            no_teacher_acceptance_borrowed=True,
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
        inputs = dict(
            schema="quest-sdsc-student-instruction-inputs-v1",
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
            candidate_sha256=control.CANDIDATE_SHA,
            hf_home=str(work / "huggingface"),
            job_id=identity["job_id"],
            run_id=plan["run_id"],
            parent_job_id=control.PARENT_JOB,
            source_code_sha256=plan["code_sha256"],
        )
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
                    str(source / "tools/sdsc_student_instruction_probe.py"),
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
            print(json.dumps(dict(phase="instruction_worker_started", job_id=identity["job_id"])), flush=True)
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
        report = control.document(artifacts / "instruction-probe.json")
        control.require(
            report.get("diagnostic_complete") is True
            and report.get("passed") is True
            and all(report.get(k) is False for k in control.FLAGS),
            "worker incomplete or claims scientific acceptance",
        )
        raw_records = {}
        for name in ("instruction-records.jsonl", "instruction-prompts.jsonl"):
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
