#!/usr/bin/env python3
"""Node-local frozen-checkpoint qualification with bounded persistent raw evidence."""

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

GIB = 1024**3


def load_control(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("node plan changed")
    plan = json.loads(raw)
    source = Path(__file__).with_name("sdsc_student_order_qualify.py")
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != plan["control_sha256"]["tools/sdsc_student_order_qualify.py"]
    ):
        raise ValueError("node controller changed")
    spec = importlib.util.spec_from_file_location("_qualification_node_control", source)
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)
    return control, control.validate_plan(plan)


def start_worker(plan, source, science, inputs, artifacts, environment, stream):
    return subprocess.Popen(
        worker_argv(plan, source, science, inputs, artifacts),
        cwd=science,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def allocation(plan, control):
    return control.helper("sdsc_student_quality_job").allocation(plan, control)


def mount(path, control):
    return control.helper("sdsc_student_quality_job").mount(path, control)


def memory_envelope(control, job, *, proc=Path("/proc/self/cgroup"), root=Path("/sys/fs/cgroup")):
    guard = control.helper("sdsc_student_memory")
    evidence = guard.memory_envelope(expected_bytes=192 * GIB, proc=proc, root=root)
    try:
        control.validate_job_memory_events(evidence, job)
        control.require(
            "job_" + job in Path(evidence["path"]).parts, "limiting cgroup belongs to another job"
        )
    except ValueError as error:
        evidence.update(passed=False, error=str(error))
        raise guard.MemoryEnvelopeError(str(error), evidence) from error
    return evidence


def stage_models(cache, destination, control):
    """Copy only the pinned 1.7B cache through the immutable stream verifier."""
    model = "models--Qwen--Qwen3-1.7B"
    revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    snapshot = control.safe(cache / "hub" / model / "snapshots" / revision)
    control.require(snapshot.is_dir(), "pinned native model snapshot missing")
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "chat_template.jinja",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    rows = []
    for folder, dirs, files in os.walk(snapshot, followlinks=False):
        control.require(not any((Path(folder) / n).is_symlink() for n in dirs), "cache directory symlink")
        for name in files:
            if name not in names and not re.fullmatch(r"model-[0-9]+-of-[0-9]+\.safetensors", name):
                continue
            original = Path(folder) / name
            resolved = original.resolve(strict=True)
            control.require(
                cache in resolved.parents and resolved.is_file(), "cache link escaped trusted root"
            )
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                while block := stream.read(8 * 1024**2):
                    digest.update(block)
            rows.append(
                dict(
                    path=original.relative_to(cache).as_posix(),
                    source=str(resolved),
                    size=resolved.stat().st_size,
                    sha256=digest.hexdigest(),
                )
            )
    control.require(rows and sum(r["size"] for r in rows) <= 8 * GIB, "native cache exceeds bound")
    return control.helper("sdsc_student_contract").stage_inputs(rows, destination, maximum=8 * GIB)


def worker_argv(plan, source, science, inputs, artifacts):
    return [
        plan["python"],
        "-I",
        "-B",
        "-u",
        str(source / "tools/sdsc_student_order_qualify_worker.py"),
        "--inputs-json",
        str(inputs),
        "--output-dir",
        str(artifacts),
    ]


def build_worker_inputs(plan, work, science, identity, original_science, control):
    def row(path):
        raw = control.read(path)
        return dict(path=str(path), size=len(raw), sha256=control.sha(raw))

    parent = plan["fit"]["plan"]
    candidate = control.prepared_initial(plan["protocol"])
    checkpoint = plan["prepared_checkpoint"]
    return dict(
        schema="quest-sdsc-student-order-qualification-inputs-v1",
        job_id=identity["job_id"],
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
        science_root=str(science),
        hf_home=str(work / "huggingface"),
        dataset_root=str(work / "dataset"),
        original_science_files=original_science,
        original_config=row(work / "config/resolved_config.yaml"),
        preparation_protocol=dict(
            row(science / control._fit.PROTOCOL_PATH), protocol_sha256=parent["protocol"]["protocol_sha256"]
        ),
        qualification_protocol=dict(
            row(science / control.PROTOCOL_PATH), protocol_sha256=plan["protocol"]["protocol_sha256"]
        ),
        prepared_checkpoint=dict(
            path=str(work / "checkpoints/prepared-selected.pt"),
            size=checkpoint["size"],
            sha256=checkpoint["sha256"],
        ),
        preparation_evidence=dict(
            fit_job_id=candidate["fit_job_id"],
            publication_sha256=candidate["fit_publication_sha256"],
            report=row(work / "evidence/fit-report.json"),
            independent_audit=row(work / "evidence/fit-audit.json"),
            recovery={
                name: row(work / "evidence" / ("fit-" + name.replace("_", "-") + ".json"))
                for name in control.RECOVERY_NAMES
            },
        ),
    )


def persist(plan, control, artifacts, log, result, memory, measure):
    """Stopped-worker raw files are durable before the final success receipt."""
    destination = control.safe(plan["result_dir"])
    control.require(not (destination / "receipt.json").exists(), "qualification already published")
    records = []
    total = 0
    if artifacts is not None:
        for name in control.NAMES:
            if name in {"node-result.json", "memory.json"} or not (artifacts / name).exists():
                continue
            raw = control.read(artifacts / name)
            total += len(raw)
            control.require(
                total <= control.MAX_FETCH - control.CAP, "raw publication exceeds bounded envelope"
            )
            control.write_once(destination / name, raw)
            control.require(control.read(destination / name) == raw, "persistent raw readback differs")
            records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    if log is not None and log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 65536))
            control.write_once(destination / "worker.log", stream.read(65536))
    try:
        measure("after_publication")
        if result["stage_complete"]:
            control.validate_memory(memory, result["job_id"])
    except Exception as error:
        result.update(
            stage_complete=False, qualification_passed=False, exit_code=1, memory_error=str(error)[:1500]
        )
    for name, value in [("node-result.json", result), ("memory.json", memory)]:
        raw = control.canonical(value)
        total += len(raw)
        control.require(
            len(raw) <= control.MAX_FILE and total <= control.MAX_FETCH - control.CAP,
            "publication metadata exceeds bound",
        )
        control.write_once(destination / name, raw)
        control.require(control.read(destination / name) == raw, "persistent metadata readback differs")
        records.append(dict(path=name, size=len(raw), sha256=control.sha(raw)))
    publication = dict(
        task=control.TASK,
        job_id=result["job_id"],
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=control.sha(control.canonical(plan)),
        code_sha256=plan["code_sha256"],
        passed=result["stage_complete"],
        stage_complete=result["stage_complete"],
        qualification_passed=result["stage_complete"],
        persistent_read_back_verified=True,
        files=records,
        **dict.fromkeys(control.FLAGS, False),
    )
    raw = control.canonical(publication)
    control.write_once(destination / "receipt.json", raw)
    control.require(control.read(destination / "receipt.json") == raw, "publication readback differs")


def main(argv=None):
    started = time.monotonic()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: node PLAN PLAN_SHA256")
    control, plan = load_control(*args)
    identity = allocation(plan, control)
    control.require(
        control.sha(control.read(Path(__file__)))
        == plan["control_sha256"]["tools/sdsc_student_order_qualify_job.py"],
        "node wrapper changed",
    )
    manifest = control.verify_source(plan)
    parent_status = control.verify_parent(plan, accounting=False)
    fit = plan["fit"]["plan"]
    candidate = control.prepared_initial(plan["protocol"])
    source_root = Path(fit["result_dir"])
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
    memory = {}
    result = dict(
        identity,
        task=control.TASK,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
        stage_complete=False,
        qualification_passed=False,
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
        raise RuntimeError("qualification interrupted by signal " + str(number))

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
            and shutil.disk_usage(scratch).free >= 32 * GIB,
            "node-local free-space gate failed",
        )
        work = Path(
            tempfile.mkdtemp(prefix="opd-student-order-qualify-" + identity["job_id"] + "-", dir=scratch)
        )
        result.update(work_dir=str(work), local_mount=local_mount)
        source, science = work / "source", work / "science"
        contract = control.helper("sdsc_student_contract")
        source_rows = [
            dict(row, source=str(Path(plan["release"]) / "source" / row["path"])) for row in manifest["files"]
        ]
        result["source_staging"] = contract.stage_inputs(source_rows, source, maximum=48 * control.CAP)
        control.verify_source(plan, source)
        result["science_restore"] = control.verify_science(plan, science)
        original_receipt = fit["parent"]["receipt"]
        proof = control.document(
            original_receipt["prerequisites_path"], original_receipt["prerequisites_sha256"]
        )
        old_science = proof["protocol"]["science_file_sha256"]
        control.require(
            len(old_science) == 49
            and control.sha(control.canonical(old_science)) == control._initial.NAMED_SCIENCE_MAP_SHA,
            "original scientific inventory differs",
        )
        for name, digest in old_science.items():
            control.require(
                control.sha(control.read(science / name)) == digest, "original scientific byte differs"
            )
        checkpoint = plan["prepared_checkpoint"]
        result["checkpoint_staging"] = contract.stage_inputs(
            [dict(checkpoint, source=str(source_root / checkpoint["path"]), path="prepared-selected.pt")],
            work / "checkpoints",
            maximum=8 * GIB,
        )
        config = fit["resolved_config"]
        result["config_staging"] = contract.stage_inputs(
            [
                dict(
                    config,
                    source=str(Path(original_receipt["result_dir"]) / config["path"]),
                    path="resolved_config.yaml",
                )
            ],
            work / "config",
            maximum=control.CAP,
        )
        result["dataset_staging"] = contract.stage_inputs(
            fit["dataset_inputs"], work / "dataset", maximum=2 * GIB
        )
        result["model_staging"] = stage_models(Path(plan["hf_home"]), work / "huggingface", control)
        evidence = work / "evidence"
        evidence.mkdir()
        control.write_once(evidence / "fit-report.json", control.read(source_root / "prepare-report.json"))
        control.require(
            control.sha(control.read(evidence / "fit-report.json")) == candidate["fit_report_sha256"],
            "fit report staging differs",
        )
        control.write_once(
            evidence / "fit-audit.json", control.decode_audit(plan["fit"]["audit"], candidate, fit)[0]
        )
        frozen_recovery = control.decode_recovery_records(plan["fit"]["recovery"], candidate, fit)
        current_recovery = control.recovery_records(
            json.loads(frozen_recovery["execution_plan"]),
            parent_status,
            control.read(source_root / "execution/receipt.json"),
            control.read(source_root / "receipt.json"),
        )
        recovery_bytes = control.decode_recovery_records(current_recovery, candidate, fit)
        control.require(
            all(
                recovery_bytes[name] == frozen_recovery[name]
                for name in control.RECOVERY_NAMES
                if name != "verified_status"
            ),
            "immutable parent recovery evidence changed",
        )
        # Fresh verification above is mandatory; keep the frozen status bytes
        # for the same independently replayable bundle in worker and auditor.
        for name, raw in frozen_recovery.items():
            control.write_once(evidence / ("fit-" + name.replace("_", "-") + ".json"), raw)
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
            OMP_NUM_THREADS="24",
            MKL_NUM_THREADS="24",
            OPENBLAS_NUM_THREADS="24",
            NUMEXPR_NUM_THREADS="24",
            TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
            TMPDIR=str(work),
            XDG_CACHE_HOME=str(work / "cache"),
        )
        log = work / "worker.log"
        with log.open("xb") as stream:
            deadline = started + 7200 - 300
            control.require(time.monotonic() < deadline, "qualification staging exhausted finite deadline")
            process = start_worker(
                plan, source, science, work / "inputs.json", artifacts, environment, stream
            )
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                control.require(remaining > 0, "qualification worker exceeded finite deadline")
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    measure("running-" + str(int(time.monotonic())))
        control.require(process.returncode == 0, "qualification worker failed; preserve outputs, never retry")
        measure("after_inference")
        report = control.document(artifacts / "qualification-report.json")
        raw_records = {
            name: dict(path=name, size=len(raw), sha256=control.sha(raw))
            for name in control.NAMES
            if name != "qualification-report.json" and (artifacts / name).exists()
            for raw in (control.read(artifacts / name),)
        }
        control.validate_worker_report(report, plan, identity["job_id"], raw_records)
        control.require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == identity["cuda_visible_devices"],
            "CUDA assignment changed",
        )
        result.update(stage_complete=True, qualification_passed=True, exit_code=0)
    except BaseException as error:
        result.update(error=type(error).__name__ + ": " + str(error)[:1500], exit_code=1)
    finally:
        if process is not None:
            try:
                control.helper("sdsc_student_lr_job").stop_worker_group(process)
            except Exception as error:
                result.update(stage_complete=False, exit_code=1, worker_shutdown_error=str(error)[:1500])
                artifacts = log = None
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
