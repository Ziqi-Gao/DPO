#!/usr/bin/env python3
"""Once-only infrastructure CUDA diagnostic; never a preparation/science gate."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import pwd
import re
import shlex
import stat
import sys
import tempfile
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
TASK = "sdsc-cuda-diagnostic-v1"
SCHEMA = "quest-sdsc-cuda-diagnostic-plan-v1"
CAP = 1024**2
MAX_FILE, MAX_FETCH, MAX_PLAN = CAP, 4 * CAP, 4 * CAP
PYTHON = str(PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
RESOURCES = dict(
    account="nwu181",
    partition="nairr-gpu-shared",
    qos="nairr-gpu-shared-normal",
    gpu_type="h100",
    gpus=2,
    cpus=4,
    mem_gib=16,
    time="00:05:00",
)
SCOPE = dict(training=False, model_loading=False, scientific_acceptance=False)
FAILED_JOB = "54623814"
FAILED_PLAN_SHA = "579e9e029568fca662a830c3e50496dac33b40ee8d71d0c91f7027ea1ff16630"
FAILED_PUBLICATION_SHA = "cf75c0bc2c533b148113103353a9c3e7982972c1235c10160bf800ce7272c396"
FAILED_REPORT_SHA = "de0f3152885fb97e15f46c4fb840bfa03dd805da4f90b617edf257ae01476ae6"
FROZEN = {
    "tools/sdsc_student_order.py": "999c0bc54991200bd7e1dc7838f432619d5862699433cc7b35b9f0b846123c40",
    "tools/sdsc_student_order_job.py": "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
    "tools/sdsc_student_order_worker.py": "e5429b4eff6885513f708207e17a49036a1dcb847f9044231d2ae1160d3c3382",
    "tools/sdsc_student_order_audit.py": "0b1ce443f468d16feb07490d12e6970cd232e3e73bb5f5791adf3ab82cedbb38",
}


def helper(name, root=ROOT):
    relative = "tools/" + name + ".py"
    path = root / relative
    if relative in FROZEN and hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN[relative]:
        raise ValueError("immutable shared helper changed: " + relative)
    spec = importlib.util.spec_from_file_location("_prepare_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_prior = helper("sdsc_student_order")
FROZEN.update(_prior.FROZEN)
require, canonical, sha, now = _prior.require, _prior.canonical, _prior.sha, _prior.now
safe, relative, write_once, run = _prior.safe, _prior.relative, _prior.write_once, _prior.run
job_queue, job_accounting = _prior.job_queue, _prior.job_accounting
live_binding, valid_live_binding = _prior._io.live_binding, _prior._io.valid_live_binding
FLAGS = _prior.FLAGS
TOOLS = tuple(
    dict.fromkeys(
        (
            "tools/sdsc_cuda_diagnostic.py",
            "tools/sdsc_cuda_diagnostic_job.py",
            "tools/sdsc_cuda_diagnostic_worker.py",
            *FROZEN,
            *_prior.TOOLS,
        )
    )
)
NAMES = ("cuda-diagnostic.json", "node-result.json", "memory.json")


def read(path, limit=MAX_FILE):
    return _prior._io.read(path, limit=limit)


def document(path, expected=None):
    raw = read(path)
    require(expected is None or sha(raw) == expected, "evidence SHA differs")
    return json.loads(raw)


def file_limit(name):
    return MAX_FILE


def manifest_records(manifest):
    rows = _prior.manifest_records(manifest)
    require(set(TOOLS) <= rows.keys(), "deployment misses CUDA diagnostic controls")
    return rows


def verify_source(plan, root=None):
    release = safe(plan["release"])
    manifest = document(release / "manifest.json", plan["manifest_sha256"])
    require(
        manifest["run_id"] == plan["run_id"] and manifest["code_sha256"] == plan["code_sha256"],
        "release differs",
    )
    records = manifest_records(manifest)
    for name, digest in plan["failed"]["plan"]["protocol"]["science_file_sha256"].items():
        require(
            name in records and records[name]["sha256"] == digest,
            "deployed source differs from accepted implementation: " + name,
        )
    root = safe(root or release / "source")
    for name, row in records.items():
        raw = read(root / name, 4 * CAP)
        require(len(raw) == row["size"] and sha(raw) == row["sha256"], "source bytes differ: " + name)
    require(
        all(records[name]["sha256"] == digest for name, digest in plan["control_sha256"].items()),
        "controls differ",
    )
    return manifest


def sbatch(plan):
    directory = safe(plan["submission_dir"])
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--account=nwu181",
        "--partition=nairr-gpu-shared",
        "--qos=nairr-gpu-shared-normal",
        "--gpus=h100:2",
        "--mem=16384M",
        "--time=" + plan["resources"]["time"],
        "--signal=B:TERM@30",
        "--job-name=" + plan["job_name"],
        "--comment=" + plan["intent_id"],
        "--chdir=" + str(directory),
        "--output=" + str(directory / "slurm-%j.out"),
        "--error=" + str(directory / "slurm-%j.err"),
        "--export=NONE",
        str(directory / "job.sh"),
        sha(canonical(plan)),
    ]


def job_script(plan):
    worker = str(Path(plan["release"]) / "source/tools/sdsc_cuda_diagnostic_job.py")
    launch = (
        "import hashlib,pathlib,runpy,sys;p=pathlib.Path(sys.argv[1]);"
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
        'sys.argv=[str(p)]+sys.argv[3:];runpy.run_path(str(p),run_name="__main__")'
    )
    return (
        "#!/bin/bash\nset -euo pipefail\numask 077\nexec "
        + shlex.join(
            [
                plan["python"],
                "-I",
                "-B",
                "-u",
                "-c",
                launch,
                worker,
                plan["control_sha256"]["tools/sdsc_cuda_diagnostic_job.py"],
                str(Path(plan["submission_dir"]) / "plan.json"),
            ]
        )
        + ' "$1"\n'
    ).encode()


def make_receipt(plan, job, reconciled=False):
    require(re.fullmatch("[1-9][0-9]*", job), "invalid job ID")
    return dict(
        task=TASK,
        job_id=job,
        intent_id=plan["intent_id"],
        run_id=plan["run_id"],
        plan_sha256=sha(canonical(plan)),
        code_sha256=plan["code_sha256"],
        result_dir=plan["result_dir"],
        resources=plan["resources"],
        reconciled=reconciled,
        received_at=now(),
    )


def validate_submission_receipt(receipt, plan):
    expected = make_receipt(plan, receipt.get("job_id", ""))
    require(
        all(
            receipt.get(key) == value
            for key, value in expected.items()
            if key not in {"received_at", "reconciled"}
        ),
        "submission receipt differs",
    )


def remote_action(request):
    plan = validate_plan(request["plan"])
    verify_source(plan)
    directory, claim = safe(plan["submission_dir"]), safe(plan["claim"])
    action = request["action"]
    if action == "dry-run":
        checks = admission(plan)
        return dict(
            dry_run=True,
            plan_sha256=sha(canonical(plan)),
            argv=sbatch(plan),
            resources=plan["resources"],
            source_verified=True,
            failed_prerequisite_reconciled=True,
            node_mounts_still_required=True,
            blockers=[],
            **checks,
        )
    if action == "submit":
        require(request.get("authorize") is True, "explicit authorization required")
        proof = request.get("dry_run")
        require(
            isinstance(proof, dict)
            and proof.get("dry_run") is True
            and proof.get("blockers") == []
            and proof.get("plan_sha256") == sha(canonical(plan))
            and proof.get("argv") == sbatch(plan),
            "matching dry-run required",
        )
        checks = admission(plan)
        directory.parent.mkdir(parents=True, exist_ok=True)
        claim.parent.mkdir(parents=True, exist_ok=True)
        write_once(
            claim, canonical(dict(intent_id=plan["intent_id"], plan_sha256=sha(canonical(plan)), at=now()))
        )
        directory.mkdir(mode=0o700)
        write_once(directory / "plan.json", canonical(plan))
        write_once(directory / "job.sh", job_script(plan))
        write_once(directory / "admission.json", canonical(checks))
        write_once(directory / "submission-started.json", canonical(dict(argv=sbatch(plan), at=now())))
        try:
            result = run(sbatch(plan))
            match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\s*", result["stdout"])
            require(result["returncode"] == 0 and match is not None, "missing trustworthy acknowledgement")
            receipt = make_receipt(plan, match.group(1))
            write_once(directory / "receipt.json", canonical(receipt))
            write_once(directory / "live-binding.json", canonical(live_binding(plan, receipt["job_id"])))
            return receipt
        except BaseException as error:
            write_once(
                directory / "unknown.json", canonical(dict(error=str(error)[:1000], no_retry=True, at=now()))
            )
            raise
    require(
        read(directory / "plan.json") == canonical(plan)
        and document(claim)["plan_sha256"] == sha(canonical(plan)),
        "permanent claims differ",
    )
    if action == "reconcile" and not (directory / "receipt.json").exists():
        user = pwd.getpwuid(os.getuid()).pw_name
        q = run(["squeue", "--noheader", "--user=" + user, "--name=" + plan["job_name"], "--format=%i|%j|%k"])
        a = run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--user=" + user,
                "--name=" + plan["job_name"],
                "--starttime=" + plan["created_at"][:10],
                "--format=JobIDRaw,JobName%100,Comment%100,User%64",
            ]
        )
        require(q["returncode"] == a["returncode"] == 0, "reconciliation unavailable; never retry")
        ids = set()
        for line in (q["stdout"] + "\n" + a["stdout"]).splitlines():
            fields = line.split("|")
            if (
                len(fields) >= 3
                and fields[1:3] == [plan["job_name"], plan["intent_id"]]
                and re.fullmatch("[1-9][0-9]*", fields[0])
                and (len(fields) == 3 or fields[3] == user)
            ):
                ids.add(fields[0])
        require(len(ids) == 1, "zero/ambiguous submission matches; never retry")
        write_once(directory / "receipt.json", canonical(make_receipt(plan, ids.pop(), True)))
    receipt = document(directory / "receipt.json")
    validate_submission_receipt(receipt, plan)
    if action == "reconcile":
        if not (directory / "live-binding.json").exists():
            write_once(directory / "live-binding.json", canonical(live_binding(plan, receipt["job_id"])))
        return receipt
    status = inspect_job(plan, receipt)
    if action == "status":
        return status
    require(action == "fetch", "unknown remote operation")
    files, total = {}, 0
    if status["publication_verified"]:
        records = publication_records(status["publication"], plan, receipt["job_id"])
        for name in (*records, "receipt.json"):
            data = read(safe(plan["result_dir"]) / name)
            expected = status["publication_sha256"] if name == "receipt.json" else records[name]["sha256"]
            require(sha(data) == expected, "artifact changed during fetch")
            files[name] = base64.b64encode(data).decode()
            total += len(data)
    for name, path in [
        ("worker.log", safe(plan["result_dir"]) / "worker.log"),
        ("slurm.out", directory / ("slurm-" + receipt["job_id"] + ".out")),
        ("slurm.err", directory / ("slurm-" + receipt["job_id"] + ".err")),
    ]:
        if path.exists():
            with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
                info = os.fstat(stream.fileno())
                require(stat.S_ISREG(info.st_mode), "nonregular log")
                stream.seek(max(0, info.st_size - 65536))
                data = stream.read(65536)
            files[name] = base64.b64encode(data).decode()
            total += len(data)
    require(total <= MAX_FETCH, "fetch exceeds bound; weights stay remote")
    return dict(status=status, receipt=receipt, files=files, bytes=total)


def validate_download(result, plan):
    validate_submission_receipt(result["receipt"], plan)
    require(
        isinstance(result["files"], dict)
        and set(result["files"]) <= set(NAMES) | {"receipt.json", "worker.log", "slurm.out", "slurm.err"},
        "unexpected fetched path",
    )
    files = {
        relative(name): base64.b64decode(value, validate=True) for name, value in result["files"].items()
    }
    require(
        sum(map(len, files.values())) == result["bytes"] <= MAX_FETCH
        and all(len(data) <= file_limit(name) for name, data in files.items()),
        "fetch size differs",
    )
    if result["status"]["publication_verified"]:
        publication = json.loads(files["receipt.json"])
        require(
            sha(files["receipt.json"]) == result["status"]["publication_sha256"]
            and publication == result["status"]["publication"],
            "publication differs",
        )
        rows = publication_records(publication, plan, result["receipt"]["job_id"])
        require(set(rows) <= files.keys(), "fetch missing published raw evidence")
        for name, row in rows.items():
            require(
                len(files[name]) == row["size"] and sha(files[name]) == row["sha256"],
                "fetched raw hash differs",
            )
    return files


def ssh_operation(cli, plan, action, authorize=False, dry_run=None):
    path = str(Path(plan["release"]) / "source/tools/sdsc_cuda_diagnostic.py")
    launch = (
        "import hashlib,pathlib,runpy,sys;p=pathlib.Path(sys.argv[1]);"
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
        'sys.argv=[str(p),"remote"];runpy.run_path(str(p),run_name="__main__")'
    )
    result = cli.ssh_call(
        [plan["python"], "-I", "-B", "-c", launch, path, plan["control_sha256"][TOOLS[0]]],
        data=canonical(dict(action=action, plan=plan, authorize=authorize, dry_run=dry_run)),
        timeout=240,
    )
    require(
        result.returncode == 0,
        "remote action failed/acknowledgement unknown; reconcile: "
        + result.stderr.decode("utf8", "replace")[-1500:],
    )
    require(len(result.stdout) <= 2 * MAX_FETCH, "remote response exceeds bound")
    return json.loads(result.stdout)


def claim_path():
    return (
        CONTROL
        / "cuda-diagnostic-claims"
        / (
            sha(canonical(dict(task=TASK, failed_job=FAILED_JOB, failed_plan_sha256=FAILED_PLAN_SHA)))
            + ".json"
        )
    )


def execution_intent(plan):
    return sha(
        canonical(
            {
                k: plan[k]
                for k in (
                    "task",
                    "run_id",
                    "code_sha256",
                    "manifest_sha256",
                    "control_sha256",
                    "resources",
                    "failed",
                )
            }
        )
    )[:32]


def bind_paths(plan):
    intent = execution_intent(plan)
    plan.update(
        intent_id=intent,
        release=str(CONTROL / "releases" / plan["run_id"]),
        submission_dir=str(CONTROL / "cuda-diagnostic-submissions" / intent),
        result_dir=str(PROJECT / "cuda-diagnostic" / intent),
        claim=str(claim_path()),
        job_name="opd-cuda-" + intent,
    )


def validate_plan(plan):
    keys = {
        "schema",
        "task",
        "run_id",
        "code_sha256",
        "manifest_sha256",
        "created_at",
        "control_sha256",
        "resources",
        "failed",
        "python",
        "scope",
        "intent_id",
        "release",
        "submission_dir",
        "result_dir",
        "claim",
        "job_name",
        *FLAGS,
    }
    require(
        isinstance(plan, dict) and set(plan) == keys and len(canonical(plan)) <= MAX_PLAN,
        "diagnostic plan shape differs",
    )
    require(
        plan["schema"] == SCHEMA
        and plan["task"] == TASK
        and canonical(plan["resources"]) == canonical(RESOURCES)
        and canonical(plan["scope"]) == canonical(SCOPE),
        "diagnostic task/resources/scope differs",
    )
    require(
        plan["python"] == PYTHON and all(plan[k] is False for k in FLAGS),
        "runtime or acceptance scope differs",
    )
    require(
        set(plan["control_sha256"]) == set(TOOLS)
        and all(re.fullmatch("[a-f0-9]{64}", h) for h in plan["control_sha256"].values())
        and all(plan["control_sha256"][k] == h for k, h in FROZEN.items()),
        "control identity differs",
    )
    failed = plan["failed"]
    require(
        set(failed) == {"job_id", "plan", "plan_sha256", "publication_sha256", "report_sha256"}
        and failed["job_id"] == FAILED_JOB
        and failed["plan_sha256"] == FAILED_PLAN_SHA == sha(canonical(failed["plan"]))
        and failed["publication_sha256"] == FAILED_PUBLICATION_SHA
        and failed["report_sha256"] == FAILED_REPORT_SHA,
        "unreviewed failure diagnostic target",
    )
    _prior.validate_plan(failed["plan"])
    require(failed["plan"]["mode"] == "preflight", "probe cannot replace fit")
    require(
        re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", plan["run_id"])
        and all(re.fullmatch("[a-f0-9]{64}", plan[k]) for k in ("code_sha256", "manifest_sha256")),
        "release identity differs",
    )
    expected = dict(plan)
    bind_paths(expected)
    require(
        all(
            plan[k] == expected[k]
            for k in ("intent_id", "release", "submission_dir", "result_dir", "claim", "job_name")
        ),
        "diagnostic intent/paths differ",
    )
    return plan


def validate_failed_evidence(report, publication):
    require(sha(canonical(report) + b"\n") == FAILED_REPORT_SHA, "failed report differs")
    require(
        publication["job_id"] == FAILED_JOB
        and publication["plan_sha256"] == FAILED_PLAN_SHA
        and publication["large_files"] == []
        and publication["passed"] is False,
        "failed publication differs",
    )
    require(
        report["error"] == "ValueError: two assigned H100s required"
        and report["execution_complete"] is False
        and report["raw_artifacts"] == []
        and all(report[k] is False for k in FLAGS),
        "failure is not the reviewed pre-update CUDA guard",
    )


def verify_failed(plan):
    failed = plan["failed"]["plan"]
    directory = safe(failed["submission_dir"])
    require(read(directory / "plan.json", MAX_PLAN) == canonical(failed), "failed stored plan differs")
    receipt = document(directory / "receipt.json")
    _prior.validate_submission_receipt(receipt, failed)
    require(receipt["job_id"] == FAILED_JOB, "failed receipt job differs")
    require(
        document(safe(failed["claim"]))["plan_sha256"] == FAILED_PLAN_SHA
        and document(safe(failed["scientific_claim"])) == _prior.scientific_claim_record(failed),
        "failed permanent claims differ",
    )
    status = _prior.inspect_job(failed, receipt)
    require(
        status["state"] == "FAILED"
        and not status["queue"]["stdout"].strip()
        and status["publication_verified"] is True
        and status["artifact_hashes_verified"] is True
        and status["publication_sha256"] == FAILED_PUBLICATION_SHA,
        "failed job not reconciled with verified publication",
    )
    rows = [line.split("|") for line in status["accounting"]["stdout"].splitlines() if line.strip()]
    require(
        {x[0] for x in rows} == {FAILED_JOB, FAILED_JOB + ".batch", FAILED_JOB + ".extern"}
        and all(
            x[1:3] == (["COMPLETED", "0:0"] if x[0].endswith(".extern") else ["FAILED", "1:0"]) for x in rows
        ),
        "failed terminal step accounting differs",
    )
    report = document(safe(failed["result_dir"]) / "prepare-report.json", FAILED_REPORT_SHA)
    validate_failed_evidence(report, status["publication"])
    return status


def admission(plan):
    failed = verify_failed(plan)
    require(
        not safe(plan["claim"]).exists() and not safe(plan["submission_dir"]).exists(),
        "diagnostic already claimed; reconcile, never retry",
    )
    require(
        safe(plan["python"]).is_file()
        and os.access(plan["python"], os.X_OK)
        and sha(read(plan["python"], 64 * CAP))
        == "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
        "fixed Python differs",
    )
    require(PROJECT.is_dir() and os.access(PROJECT, os.W_OK), "persistent storage unavailable")
    return dict(
        failed=failed, runtime=_prior.verify_runtime(plan), gpu_concurrency=_prior.check_gpu_ceiling()
    )


def validate_accounting(plan, job, queue, account, observed=None):
    require(queue["returncode"] == account["returncode"] == 0, "Slurm query unavailable")
    rows = [line.split("|") for line in account["stdout"].splitlines() if line.strip()]
    require(
        len({x[0] for x in rows}) == len(rows)
        and all(x[0] == job or x[0].startswith(job + ".") for x in rows),
        "ambiguous accounting",
    )
    own = [x for x in rows if x[0] == job]
    require(len(own) <= 1, "ambiguous job accounting")
    state = "UNKNOWN"
    complete = False
    terminal = False
    if own:
        row = own[0]
        require(
            len(row) >= 12
            and row[3:7] == [plan["job_name"], "nwu181", "nairr-gpu-shared", "nairr-gpu-shared-normal"]
            and (row[11] == plan["intent_id"] or (row[11] == "" and valid_live_binding(plan, job, observed))),
            "job identity differs",
        )
        state = row[1]
        terminal = not queue["stdout"].strip() and state in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
        }
        if terminal:
            require(
                {job + ".batch", job + ".extern"} <= {x[0] for x in rows},
                "terminal step accounting incomplete",
            )
            tres = dict(x.split("=", 1) for x in row[10].split(",") if "=" in x)
            require(
                row[7] == "4"
                and row[8] in {"16Gn", "16384Mn", "16384M", "16G"}
                and tres.get("cpu") == "4"
                and tres.get("gres/gpu") == "2"
                and tres.get("node") == "1",
                "actual diagnostic allocation differs",
            )
        if state == "COMPLETED" and terminal:
            require(all(x[1:3] == ["COMPLETED", "0:0"] for x in rows), "Slurm step failed")
            complete = True
    elif queue["stdout"].strip():
        q = [x.split("|") for x in queue["stdout"].splitlines() if x.strip()]
        require(len(q) == 1 and q[0][0] == job and len(q[0]) == 2, "queue identity differs")
        state = q[0][1]
    return dict(
        job_id=job,
        state=state,
        terminal=terminal,
        accounting_complete=complete,
        queue=queue,
        accounting=account,
    )


def validate_worker_report(report, job, visible=None):
    require(
        report.get("schema") == "quest-sdsc-cuda-diagnostic-report-v1"
        and report.get("job_id") == job
        and report.get("diagnostic_complete") is True
        and type(report.get("cuda_ready")) is bool
        and all(report.get(k) is False for k in FLAGS),
        "worker identity/scope differs",
    )
    require(
        visible is None or report.get("cuda_visible_devices") == visible, "worker CUDA visibility changed"
    )
    expected_checks = {
        "job_matches",
        "expected_visibility_has_two_distinct_ids",
        "visibility_matches",
        "allocation_environment_matches",
        "runtime_matches",
        "cuda_available",
        "exactly_two_visible_devices",
        "both_device_names_are_h100",
        "cuda_init_succeeded",
        "device_observations_complete",
        "tiny_allocations_passed",
        "environment_unchanged",
        "visibility_unchanged",
    }
    checks = report.get("checks", {})
    require(
        set(checks) == expected_checks
        and all(type(v) is bool for v in checks.values())
        and report["cuda_ready"] is all(checks.values()),
        "CUDA readiness/checks differ",
    )
    if report["cuda_ready"]:
        runtime = dict(python="3.12.13", torch="2.8.0+cu128", cuda="12.8")
        before, after = report.get("environment", {}), report.get("environment_after", {})
        assigned = report["cuda_visible_devices"]
        require(
            isinstance(assigned, str)
            and len(assigned.split(",")) == 2
            and len(set(assigned.split(","))) == 2
            and all(x and x.strip() == x for x in assigned.split(","))
            and report.get("expected_cuda_visible_devices") == assigned
            and before == after
            and before.get("CUDA_VISIBLE_DEVICES") == assigned
            and before.get("SLURM_JOB_ID") == job
            and all(
                before.get(k) == v
                for k, v in dict(
                    SLURM_CPUS_PER_TASK="4",
                    SLURM_MEM_PER_NODE="16384",
                    SLURM_JOB_ACCOUNT="nwu181",
                    SLURM_JOB_PARTITION="nairr-gpu-shared",
                ).items()
            ),
            "ready CUDA allocation evidence differs",
        )
        require(
            report.get("runtime_actual") == runtime
            and report.get("runtime_expected") == runtime
            and report.get("torch_import", {}).get("ok") is True
            and canonical(report.get("cuda_is_available")) == canonical(dict(ok=True, value=True))
            and report.get("cuda_device_count", {}).get("ok") is True
            and type(report["cuda_device_count"].get("value")) is int
            and report["cuda_device_count"]["value"] == 2
            and report.get("cuda_init", {}).get("ok") is True
            and report.get("allocation_attempted") is True
            and report.get("device_enumeration_permitted") is True,
            "ready CUDA raw runtime/count/init differs",
        )
        devices, allocations = report.get("devices", []), report.get("tiny_allocations", [])
        require(len(devices) == len(allocations) == 2, "ready CUDA logical device count differs")
        for index, (device, allocation) in enumerate(zip(devices, allocations, strict=True)):
            require(
                type(device.get("logical_device")) is int
                and device["logical_device"] == index
                and all(device.get(k, {}).get("ok") is True for k in ("name", "capability", "properties"))
                and isinstance(device["name"].get("value"), str)
                and "H100" in device["name"]["value"],
                "ready CUDA device observations differ",
            )
            expected = dict(
                logical_device=index,
                numel=4,
                dtype="torch.float32",
                sum=10.0,
                finite=True,
                expected_sum=10.0,
                passed=True,
            )
            require(
                type(allocation.get("logical_device")) is int
                and allocation["logical_device"] == index
                and allocation.get("result", {}).get("ok") is True
                and canonical(allocation["result"].get("value")) == canonical(expected),
                "ready CUDA tiny allocation differs",
            )
    return report


def publication_records(published, plan, job):
    expected = dict(
        task=TASK,
        job_id=job,
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        code_sha256=plan["code_sha256"],
        persistent_read_back_verified=True,
    )
    require(
        all(canonical(published.get(k)) == canonical(v) for k, v in expected.items())
        and all(published.get(k) is False for k in FLAGS)
        and type(published.get("diagnostic_complete")) is bool
        and type(published.get("cuda_ready")) is bool,
        "publication identity/scope differs",
    )
    rows = published.get("files")
    require(isinstance(rows, list) and len(rows) <= len(NAMES), "publication inventory differs")
    records = {}
    for row in rows:
        require(
            set(row) == {"path", "size", "sha256"}
            and row["path"] in NAMES
            and row["path"] not in records
            and type(row["size"]) is int
            and 0 <= row["size"] <= MAX_FILE
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "unsafe publication row",
        )
        records[row["path"]] = row
    require(sum(x["size"] for x in records.values()) <= MAX_FETCH - CAP, "publication exceeds bound")
    return records


def validate_memory(memory, job):
    phases = ("initial", "final", "after_publication")
    require(set(phases) <= memory.keys(), "memory publication phases missing")
    for row in memory.values():
        _prior.validate_job_memory_events(row, job)
        ancestors = row["ancestors"]
        finite = [x for x in ancestors if x.get("limit_bytes") is not None]
        require(
            finite and all(type(x["limit_bytes"]) is int and x["limit_bytes"] > 0 for x in finite),
            "invalid raw memory limits",
        )
        limit = min(x["limit_bytes"] for x in finite)
        limiting = [x for x in finite if x["limit_bytes"] == limit]
        require(
            limit == 16 * 1024**3
            and all(
                type(x.get("current_bytes")) is type(x.get("peak_bytes")) is int
                and 0 < x["current_bytes"] <= x["peak_bytes"] <= limit
                for x in limiting
            ),
            "raw diagnostic memory envelope failed",
        )
        selected = max(limiting, key=lambda x: x["peak_bytes"])
        require(
            "job_" + job in Path(selected["path"]).parts
            and row.get("passed") is True
            and row.get("expected_limit_bytes") == limit
            and row.get("limit_bytes") == limit
            and row.get("path") == selected["path"]
            and row.get("current_bytes") == selected["current_bytes"]
            and row.get("peak_bytes") == selected["peak_bytes"]
            and row.get("headroom_bytes") == limit - selected["peak_bytes"] >= 1024**3
            and row.get("minimum_headroom_bytes") == 1024**3
            and type(row.get("observed_at_unix")) in (int, float)
            and math.isfinite(row["observed_at_unix"]),
            "memory summary differs from raw hierarchy",
        )
    require(
        all(
            memory[left]["observed_at_unix"] <= memory[right]["observed_at_unix"]
            and memory[left]["peak_bytes"] <= memory[right]["peak_bytes"]
            for left, right in pairwise(phases)
        ),
        "memory phases moved backwards",
    )


def inspect_job(plan, receipt):
    job = receipt["job_id"]
    directory = safe(plan["submission_dir"])
    observed = (
        document(directory / "live-binding.json") if (directory / "live-binding.json").exists() else None
    )
    result = validate_accounting(plan, job, job_queue(job), job_accounting(job), observed)
    result.update(
        success=False,
        diagnostic_complete=False,
        cuda_ready=False,
        publication_verified=False,
        artifact_hashes_verified=False,
        checked_at=now(),
        **dict.fromkeys(FLAGS, False),
    )
    root = safe(plan["result_dir"])
    records = {}
    if (root / "receipt.json").exists():
        raw = read(root / "receipt.json")
        published = json.loads(raw)
        records = publication_records(published, plan, job)
        require(
            {name for name in NAMES if (root / name).exists()} == set(records), "persistent inventory differs"
        )
        for name, row in records.items():
            data = read(root / name)
            require(
                len(data) == row["size"] and sha(data) == row["sha256"], "persistent artifact hash differs"
            )
        result.update(
            publication_verified=True,
            artifact_hashes_verified=True,
            publication=published,
            publication_sha256=sha(raw),
        )
        if "cuda-diagnostic.json" in records:
            report = validate_worker_report(document(root / "cuda-diagnostic.json"), job)
            require(
                all(published[k] is report[k] for k in ("diagnostic_complete", "cuda_ready")),
                "published completion differs",
            )
            result.update(
                diagnostic_complete=report["diagnostic_complete"],
                cuda_ready=report["cuda_ready"],
                result_sha256=records["cuda-diagnostic.json"]["sha256"],
            )
    if result["accounting_complete"]:
        require(
            result["publication_verified"]
            and set(NAMES) == set(records)
            and result["diagnostic_complete"]
            and result["cuda_ready"],
            "completed allocation lacks successful CUDA evidence",
        )
        node = document(root / "node-result.json")
        require(
            node.get("job_id") == job
            and node.get("plan_sha256") == sha(canonical(plan))
            and node.get("exit_code") == 0
            and all(node.get(k) is False for k in FLAGS),
            "node completion differs",
        )
        validate_worker_report(report, job, node["cuda_visible_devices"])
        validate_memory(document(root / "memory.json"), job)
        result["success"] = True
    return result


def prepare(args, cli):
    record = cli.run_record(args.run_id)
    require(record["state"] == "deployed", "matching source sync required")
    manifest = record["manifest"]
    rows = manifest_records(manifest)
    require(
        record["deployment"]["code_sha256"] == manifest["code_sha256"]
        and all(sha(read(ROOT / k)) == rows[k]["sha256"] for k in TOOLS),
        "local/deployed control differs",
    )
    failed = document(safe(args.failed_plan.absolute()))
    fetch = safe(args.failed_fetch_dir.absolute())
    report = document(fetch / "prepare-report.json", FAILED_REPORT_SHA)
    publication = document(fetch / "receipt.json", FAILED_PUBLICATION_SHA)
    validate_failed_evidence(report, publication)
    plan = dict(
        schema=SCHEMA,
        task=TASK,
        run_id=manifest["run_id"],
        code_sha256=manifest["code_sha256"],
        manifest_sha256=sha(canonical(manifest) + b"\n"),
        created_at=now(),
        control_sha256={k: rows[k]["sha256"] for k in TOOLS},
        resources=RESOURCES,
        failed=dict(
            job_id=FAILED_JOB,
            plan=failed,
            plan_sha256=FAILED_PLAN_SHA,
            publication_sha256=FAILED_PUBLICATION_SHA,
            report_sha256=FAILED_REPORT_SHA,
        ),
        python=PYTHON,
        scope=SCOPE,
        **dict.fromkeys(FLAGS, False),
    )
    bind_paths(plan)
    validate_plan(plan)
    directory = ROOT / ".sdsc/cuda-diagnostic" / plan["intent_id"]
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(directory / "plan.json", canonical(plan))
    return dict(plan=str(directory / "plan.json"), plan_sha256=sha(canonical(plan)), resources=RESOURCES)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "fetch", "reconcile", "remote"))
    parser.add_argument("--run-id")
    for name in ("failed-plan", "failed-fetch-dir", "plan", "dry-run-file"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "remote":
        require(
            Path(__file__).resolve().is_relative_to(CONTROL / "releases"),
            "Slurm operations cannot run on Quest",
        )
        raw = sys.stdin.buffer.read(MAX_PLAN + CAP + 1)
        require(len(raw) <= MAX_PLAN + CAP, "remote request exceeds bound")
        result = remote_action(json.loads(raw))
    else:
        cli = helper("sdsc_cli")
        if args.action == "prepare":
            require(
                args.run_id and args.failed_plan and args.failed_fetch_dir,
                "prepare requires run-id/failed-plan/failed-fetch-dir",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "operation requires plan")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/cuda-diagnostic" and path.name == "plan.json",
                "local plan path differs",
            )
            plan = validate_plan(document(path))
            require(
                path.parent.name == plan["intent_id"]
                and all(sha(read(ROOT / k)) == v for k, v in plan["control_sha256"].items()),
                "local controls changed",
            )
            action = "dry-run" if args.action == "submit" and args.dry_run else args.action
            dry_run = None
            if action == "submit":
                require(
                    args.authorize and args.dry_run_file is not None,
                    "submit requires authorization and matching dryrun",
                )
                dry_run = document(safe(args.dry_run_file.absolute()))
                cli.require_master()
                write_once(path.parent / "submission-started.json", canonical(dict(at=now(), no_retry=True)))
            result = ssh_operation(cli, plan, action, args.authorize, dry_run)
            if action == "fetch":
                files = validate_download(result, plan)
                directory = safe(ROOT / ".sdsc/fetched" / result["receipt"]["job_id"])
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                destination = Path(tempfile.mkdtemp(prefix="fetch-", dir=directory))
                for name, raw in files.items():
                    write_once(destination / name, raw)
                    require(read(destination / name) == raw, "local readback differs")
                result.pop("files")
                result["destination"] = str(destination)
                write_once(destination / "fetch-manifest.json", canonical(result))
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
            write_once(path.parent / (action + "-" + stamp + ".json"), canonical(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
