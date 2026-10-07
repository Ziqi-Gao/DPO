#!/usr/bin/env python3
"""Fit after a recovered preflight, preserving the frozen scientific node sequence.

Admission is versioned explicitly. Source staging, native runtime, worker inputs
and publication retain the inner plan and original named helpers. No old module
globals or functions are replaced.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_PLAN = 8 * 1024**2
MAX_SETUP_SECONDS = 60
CONTROL_NAME = "tools/sdsc_student_name_invariant_fit.py"
NODE_NAME = "tools/sdsc_student_name_invariant_fit_job.py"
ORIGINAL_NAME = "tools/sdsc_student_name_invariant_job.py"


def load_control(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_PLAN:
        raise ValueError("invalid fit plan file")
    raw = path.read_bytes()
    if len(raw) > MAX_PLAN or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("fit node plan changed")
    plan = json.loads(raw)
    pins = plan["execution"]["control_file_sha256"]
    root = Path(__file__).resolve().parents[1]
    for name in (CONTROL_NAME, NODE_NAME):
        source = root / name
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != pins[name]:
            raise ValueError("fit entry source changed: " + name)
    spec = importlib.util.spec_from_file_location("_name_invariant_fit_node_control", root / CONTROL_NAME)
    execution = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = execution
    spec.loader.exec_module(execution)
    execution.validate_plan(plan)
    execution.require(
        path.absolute() == Path(plan["science_plan"]["submission_dir"]) / "execution-plan.json",
        "fit node plan path differs",
    )
    return execution, plan


def verify_known_job(execution, inner, job):
    """Startup may precede sbatch's reply; existing evidence must name this job."""
    directory = execution.safe(inner["submission_dir"])
    receipt = directory / "receipt.json"
    if receipt.exists():
        value = execution.document(receipt)
        execution.science.validate_submission_receipt(value, inner)
        execution.require(value["job_id"] == job, "fit allocation differs from acknowledged job")
    live = directory / "live-binding.json"
    if live.exists():
        execution.require(
            execution.science.helper("sdsc_student_lr").valid_live_binding(
                inner, job, execution.document(live)
            ),
            "fit allocation differs from live binding",
        )


def run_scientific_node(base, control, plan, identity):
    """Original scientific sequence after bounded setup consumes reserve time.

    Preserve the 28,200-second worker clock. At most 60 seconds of outer setup
    consume part of the unchanged eight-hour allocation's 600-second reserve.
    """
    started = time.monotonic()
    worker_budget = control.worker_budget_seconds(plan)
    source_root = Path(plan["parent"]["receipt"]["result_dir"])
    result_root = control.safe(plan["result_dir"])
    parent = result_root.parent
    while not parent.exists():
        parent = parent.parent
    mounts = {
        name: base.mount(path, control)
        for name, path in (
            ("input", source_root),
            ("output", parent),
            ("cache", Path(plan["hf_home"])),
            ("runtime_archive", Path(plan["runtime_snapshot"]["archive"]["path"])),
        )
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
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=control.sha(control.canonical(plan)),
        stage_complete=False,
        mounts=mounts,
        **dict.fromkeys(control.FLAGS, False),
    )

    def measure(stage):
        try:
            memory[stage] = base.memory_envelope(control, identity["job_id"])
            raw = control.canonical(memory[stage])
            control.write_once(result_root / ("memory-" + stage + ".json"), raw)
            control.require(
                control.read(result_root / ("memory-" + stage + ".json")) == raw,
                "memory evidence readback differs",
            )
            try:
                base.publish_live_progress(
                    plan, control, identity, artifacts, stage, time.monotonic() - started
                )
            except Exception as error:
                # Observability cannot change the scientific result or hide memory failure.
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
        scratch = control.safe(
            os.environ.get("TMPDIR")
            or "/scratch/{}/job_{}".format(pwd.getpwuid(os.getuid()).pw_name, identity["job_id"])
        )
        control.require(scratch.is_dir() and scratch.stat().st_uid == os.getuid(), "owned scratch absent")
        local_mount = base.mount(scratch, control)
        control.require(
            local_mount["fstype"] in {"ext2", "ext3", "ext4", "xfs", "btrfs"}
            and shutil.disk_usage(scratch).free
            >= 192 * base.GIB
            + plan["runtime_snapshot"]["archive"]["size"]
            + plan["runtime_snapshot"]["total_bytes"],
            "node-local free-space gate failed",
        )
        work = Path(
            tempfile.mkdtemp(prefix="opd-student-name-invariant-" + identity["job_id"] + "-", dir=scratch)
        )
        work.chmod(0o700)
        control.require(
            work.lstat().st_uid == os.getuid() and stat.S_IMODE(work.lstat().st_mode) == 0o700,
            "new private work directory owner/mode differs",
        )
        result.update(work_dir=str(work), local_mount=local_mount)
        artifacts = work / "artifacts"
        artifacts.mkdir(mode=0o700)
        execution_dir = artifacts
        environment = {
            key: value for key, value in os.environ.items() if key not in ("PYTHONHOME", "PYTHONPATH")
        }
        environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(work))
        source, science = work / "source", work / "science"
        result["source_staging"] = base.stage_source(plan, source, control)
        result["science_restore"] = control.verify_science(plan, science)
        control.write_once(work / "execution-plan.json", control.canonical(plan))
        descriptor = plan["runtime_snapshot"]
        manifest_raw = control.read(descriptor["manifest"]["path"], 16 * control.CAP)
        control.require(
            len(manifest_raw) == descriptor["manifest"]["size"]
            and control.sha(manifest_raw) == descriptor["manifest"]["sha256"],
            "runtime manifest changed",
        )
        control.write_once(artifacts / "runtime-manifest.json", manifest_raw)
        log = work / "worker.log"
        with log.open("xb") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-u",
                    str(source / "tools/sdsc_student_name_invariant_native.py"),
                    "stage-runtime",
                    "--plan",
                    str(work / "execution-plan.json"),
                    "--plan-sha256",
                    control.sha(control.canonical(plan)),
                    "--work",
                    str(work),
                    "--deadline",
                    str(started + worker_budget - 5),
                ],
                cwd=work,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            base.supervise_stage(process, control, deadline=started + worker_budget, measure=measure)
        process = None
        runtime_stage = control.document(artifacts / "runtime-stage.json")
        native = control.helper("sdsc_student_name_invariant_native")
        native.validate_stage(plan, runtime_stage, manifest_raw)
        context = native.context_value(plan, runtime_stage, work, identity, control)
        native.validate_context(plan, context, runtime_stage)
        control.write_once(artifacts / "runtime-context.json", control.canonical(context))
        result["runtime"] = dict(
            source_python=plan["python"],
            effective_python=runtime_stage["python"],
            runtime_stage_sha256=control.sha(control.canonical(runtime_stage)),
        )
        trace = base.run_probe(
            plan,
            control,
            execution_dir,
            identity,
            environment,
            deadline=started + worker_budget,
            measure=measure,
            context=context,
            runtime_stage=runtime_stage,
        )
        control.require(
            trace["reaped"] and trace["shutdown_error"] is None,
            "early CUDA probe process group not reaped; no large input staged",
        )
        control.require(trace["probe_passed"], "early allocation CUDA probe failed; no large input staged")
        base.validate_startup(
            plan,
            control,
            control.document(execution_dir / "early-node-startup.json"),
            identity["job_id"],
            identity["cuda_visible_devices"],
            "early_node",
            expected_python=runtime_stage["python"],
        )
        process = None
        measure("after_cuda_probe")
        staging = control.helper("sdsc_student_job")
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
            maximum=4 * base.GIB,
        )
        config = plan["resolved_config"]
        result["config_staging"] = contract.stage_inputs(
            [dict(config, source=str(source_root / config["path"]), path="resolved_config.yaml")],
            work / "config",
            maximum=control.CAP,
        )
        result["dataset_staging"] = contract.stage_inputs(
            plan["dataset_inputs"], work / "dataset", maximum=2 * base.GIB
        )
        result["model_staging"] = control.helper("sdsc_student_branch_qualify_job").stage_models(
            Path(plan["hf_home"]), work / "huggingface", control
        )
        measure("after_staging")
        inputs = base.build_worker_inputs(plan, work, science, identity, old_science, control)
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
        with log.open("ab") as stream:
            process = base.start_worker(
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
            base.validate_startup(
                plan,
                control,
                control.document(execution_dir / f"rank-{rank}-startup.json"),
                identity["job_id"],
                identity["cuda_visible_devices"],
                "rank_entry",
                rank,
                ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(artifacts)],
                expected_python=runtime_stage["python"],
            )
        control.require(process.returncode == 0, "preparation worker failed; preserve outputs, never retry")
        for rank in (0, 1):
            base.validate_rank_exit(
                plan,
                control,
                control.document(execution_dir / f"rank-{rank}-exit.json"),
                identity["job_id"],
                rank,
                ["--inputs-json", str(work / "inputs.json"), "--output-dir", str(artifacts)],
            )
        native.validate_evidence(
            plan,
            {
                name: control.read(artifacts / name)
                for name in (*control.NATIVE_NAMES, *control.EXECUTION_NAMES)
            },
            identity["job_id"],
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
        base.persist(plan, control, artifacts, log, result, memory, measure)
    print(json.dumps(result, sort_keys=True))
    return result["exit_code"]


def main(argv=None):
    started = time.monotonic()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: fit-node PLAN PLAN_SHA256")
    execution, outer = load_control(*args)
    inner = execution.science_plan(outer)
    execution.require(inner["mode"] == "fit", "fit-only node rejects other modes")
    base = execution.helper("sdsc_student_name_invariant_job")
    execution.require(
        execution.sha(execution.read(Path(base.__file__))) == inner["control_sha256"][ORIGINAL_NAME],
        "original preparation node changed",
    )
    control = execution.science
    identity = base.allocation(inner, control)
    execution.verify_source(outer)
    execution.check_claims(outer)
    verify_known_job(execution, inner, identity["job_id"])
    execution.verify_execution(outer)
    control.verify_parent(inner)
    # Recheck recovered artifacts, original claims and retained fence against the
    # verified admission accounting snapshot instead of the old incompatible gate.
    execution.verify_recovered_preflight(outer, accounting=False)
    elapsed = time.monotonic() - started
    execution.require(elapsed <= MAX_SETUP_SECONDS, "fit setup exceeded60s; no model work")
    directory = execution.safe(inner["submission_dir"])
    entry = execution.entry_record(
        outer, identity["job_id"], at=execution.now(), setup_elapsed_seconds=elapsed
    )
    execution.write_once(directory / "fit-node-entry.json", execution.canonical(entry))
    execution.require(
        execution.read(directory / "fit-node-entry.json") == execution.canonical(entry),
        "fit node entry readback differs",
    )
    try:
        execution.require(
            time.monotonic() - started <= MAX_SETUP_SECONDS,
            "fit entry persistence exceeded setup budget; no model work",
        )
        result = run_scientific_node(base, control, inner, identity)
        execution.require(type(result) is int, "scientific node returned invalid status")
    except BaseException as error:
        exit_record = execution.exit_record(
            entry,
            returncode=1,
            node_returned=False,
            completed_at=execution.now(),
            error_type=type(error).__name__,
        )
        try:
            execution.write_once(directory / "fit-node-exit.json", execution.canonical(exit_record))
        except BaseException as publication_error:
            print(
                json.dumps({"fit_exit_publication_error": type(publication_error).__name__}),
                file=sys.stderr,
                flush=True,
            )
        raise
    exit_record = execution.exit_record(
        entry, returncode=result, node_returned=True, completed_at=execution.now()
    )
    execution.write_once(directory / "fit-node-exit.json", execution.canonical(exit_record))
    execution.require(
        execution.read(directory / "fit-node-exit.json") == execution.canonical(exit_record),
        "fit node exit readback differs",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
