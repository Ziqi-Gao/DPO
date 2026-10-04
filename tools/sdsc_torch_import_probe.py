#!/usr/bin/env python3
"""Once-only Torch import A/B/A diagnostic; never a preparation/science gate."""

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
TASK = "sdsc-torch-import-probe-v1"
SCHEMA = "quest-sdsc-torch-import-probe-plan-v1"
CAP = 1024**2
MAX_FILE, MAX_FETCH, MAX_PLAN = CAP, 8 * CAP, 8 * CAP
PYTHON = str(PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
RESOURCES = dict(
    account="nwu181",
    partition="nairr-gpu-shared",
    qos="nairr-gpu-shared-normal",
    gpu_type="h100",
    gpus=2,
    cpus=24,
    mem_gib=16,
    time="00:10:00",
)
SCOPE = dict(training=False, model_loading=False, scientific_acceptance=False)
FAILED_JOB = "54643463"
FAILED_PLAN_SHA = "34b45d2b888bf7bc6b7a96c72ce5a9b664a080356cfae2bc8f10dbd4eb9a730d"
FAILED_PUBLICATION_SHA = "d7ddc12da44da08189e057347b6be24dff23db1410a384513279f1e632c2002a"
FAILED_REPORT_SHA = "63099dc1c2d40c7cb4cffae46d53ad1511de72a7f316663f8312fa4837b79468"
FAILED_EXECUTION_PLAN_SHA = "73478b7e5a45324a4f957d3ba60fe2f64053c75cb8ef63cb3db8ff04daf94c07"
FAILED_EXECUTION_PUBLICATION_SHA = "d4b4ee9effd7bc7ca842c09d384d882b31e40c83e80cc94bd3b22f1f9b289bfd"

FROZEN = {
    "tools/sdsc_student_order.py": "999c0bc54991200bd7e1dc7838f432619d5862699433cc7b35b9f0b846123c40",
    "tools/sdsc_student_order_job.py": "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
    "tools/sdsc_student_order_worker.py": "e5429b4eff6885513f708207e17a49036a1dcb847f9044231d2ae1160d3c3382",
    "tools/sdsc_student_order_audit.py": "0b1ce443f468d16feb07490d12e6970cd232e3e73bb5f5791adf3ab82cedbb38",
    "tools/sdsc_cuda_diagnostic.py": "b7b082ddad9595d8365522834ec02b691b92287c72f0a506b5ef2ed0871f181c",
    "tools/sdsc_cuda_diagnostic_job.py": "9572e9bcaa572a6d7995d483da0d24e8ce81f7ff496139d7f36119d669149e4b",
    "tools/sdsc_cuda_diagnostic_worker.py": (
        "709b9732acdff7c027b91d8762d6a164e0649f6a1e7b4fdd1d87287a30f40e5b"
    ),
    "tools/sdsc_student_order_execution_contract.py": (
        "7163d33dee68c531dda03ea6e2a32136078e2557dfefd4c450a763929367b476"
    ),
    "tools/sdsc_student_order_execution.py": (
        "1692fae5b84070961de1d33d2774b75ac46e40ee5b0666adb798b8352ad2f1c7"
    ),
    "tools/sdsc_student_order_execution_job.py": (
        "ebf004742cca8f350ff090abfc8ce3fbb3d874a87623755014f9e8766783e51b"
    ),
    "tools/sdsc_student_order_execution_worker.py": (
        "366b606532b6a476cda53dfd847eccc083adf6da226626bc3d196e20f48448e5"
    ),
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
_recovery = helper("sdsc_student_order_execution")
_cuda = helper("sdsc_cuda_diagnostic")
FROZEN.update(_prior.FROZEN)
require, canonical, sha, now = _prior.require, _prior.canonical, _prior.sha, _prior.now
safe, relative, write_once, run = _prior.safe, _prior.relative, _prior.write_once, _prior.run
job_queue, job_accounting = _prior.job_queue, _prior.job_accounting
live_binding, valid_live_binding = _prior._io.live_binding, _prior._io.valid_live_binding
FLAGS = _prior.FLAGS
verify_runtime = _prior.verify_runtime
TOOLS = tuple(
    dict.fromkeys(
        (
            "tools/sdsc_torch_import_probe.py",
            "tools/sdsc_torch_import_probe_job.py",
            "tools/sdsc_torch_import_probe_worker.py",
            *FROZEN,
            *_prior.TOOLS,
        )
    )
)
NAMES = (
    "import-observations.json",
    "node-result.json",
    "memory.json",
    *(label + suffix for label in ("A1", "B", "A2") for suffix in ("-report.json", ".log", "-proc.jsonl")),
)


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
        "--cpus-per-task=24",
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
    worker = str(Path(plan["release"]) / "source/tools/sdsc_torch_import_probe_job.py")
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
                plan["control_sha256"]["tools/sdsc_torch_import_probe_job.py"],
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
    path = str(Path(plan["release"]) / "source/tools/sdsc_torch_import_probe.py")
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
        / "torch-import-probe-claims"
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
        submission_dir=str(CONTROL / "torch-import-probe-submissions" / intent),
        result_dir=str(PROJECT / "torch-import-probe" / intent),
        claim=str(claim_path()),
        job_name="opd-import-" + intent,
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
        set(failed)
        == {
            "job_id",
            "plan",
            "plan_sha256",
            "publication_sha256",
            "report_sha256",
            "execution_plan",
            "execution_plan_sha256",
            "execution_publication_sha256",
        }
        and failed["job_id"] == FAILED_JOB
        and failed["plan_sha256"] == FAILED_PLAN_SHA == sha(canonical(failed["plan"]))
        and failed["publication_sha256"] == FAILED_PUBLICATION_SHA
        and failed["report_sha256"] == FAILED_REPORT_SHA
        and failed["execution_plan_sha256"]
        == FAILED_EXECUTION_PLAN_SHA
        == sha(canonical(failed["execution_plan"]))
        and failed["execution_publication_sha256"] == FAILED_EXECUTION_PUBLICATION_SHA
        and failed["execution_plan"]["science_plan"] == failed["plan"],
        "unreviewed failure diagnostic target",
    )
    _prior.validate_plan(failed["plan"])
    _recovery.validate_plan(failed["execution_plan"])
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


def validate_failed_evidence(report, publication, execution_publication):
    require(sha(canonical(report)) == FAILED_REPORT_SHA, "failed node report differs")
    require(
        publication["job_id"] == FAILED_JOB
        and publication["plan_sha256"] == FAILED_PLAN_SHA
        and publication["large_files"] == []
        and publication["passed"] is False,
        "failed publication differs",
    )
    require(
        str(report.get("error", "")).startswith("TimeoutExpired:")
        and "timed out after 120 seconds" in report["error"]
        and report.get("exit_code") == 1
        and all(report[k] is False for k in FLAGS),
        "failure is not the reviewed early import timeout",
    )
    require(
        execution_publication["job_id"] == FAILED_JOB
        and execution_publication["plan_sha256"] == FAILED_EXECUTION_PLAN_SHA
        and execution_publication["startup_passed"] is False,
        "failed execution publication differs",
    )


def verify_failed(plan):
    failed = plan["failed"]["plan"]
    execution = plan["failed"]["execution_plan"]
    _recovery.check_claims(execution)
    directory = safe(failed["submission_dir"])
    receipt = document(directory / "receipt.json")
    _prior.validate_submission_receipt(receipt, failed)
    require(receipt["job_id"] == FAILED_JOB, "failed receipt job differs")
    require(
        document(directory / "execution-receipt.json")
        == dict(_recovery.claim_record(execution), job_id=FAILED_JOB, scientific_receipt=receipt),
        "failed execution acknowledgement differs",
    )
    status = _recovery.inspect_job(execution, receipt)
    require(
        status["state"] == "FAILED"
        and not status["queue"]["stdout"].strip()
        and status["queue"]["returncode"] == status["accounting"]["returncode"] == 0
        and status["publication_verified"] is True
        and status["artifact_hashes_verified"] is True
        and status["publication_sha256"] == FAILED_PUBLICATION_SHA
        and status["execution_publication_verified"] is True
        and status["execution_publication_sha256"] == FAILED_EXECUTION_PUBLICATION_SHA
        and status["success"] is False,
        "failed job not reconciled with paired verified publications",
    )
    rows = [line.split("|") for line in status["accounting"]["stdout"].splitlines() if line.strip()]
    require(
        len(rows) == 3
        and {x[0] for x in rows} == {FAILED_JOB, FAILED_JOB + ".batch", FAILED_JOB + ".extern"}
        and all(
            x[1:3] == (["COMPLETED", "0:0"] if x[0].endswith(".extern") else ["FAILED", "1:0"]) for x in rows
        ),
        "failed terminal step accounting differs",
    )
    report = document(safe(failed["result_dir"]) / "node-result.json", FAILED_REPORT_SHA)
    validate_failed_evidence(report, status["publication"], status["execution_publication"])
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
                row[7] == "24"
                and row[8] in {"16Gn", "16384Mn", "16384M", "16G"}
                and tres.get("cpu") == "24"
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


def validate_observations(report, job, visible=None, reports=None):
    worker = helper("sdsc_torch_import_probe_worker")
    require(
        isinstance(report, dict)
        and report.get("schema") == "quest-sdsc-torch-import-observations-v1"
        and report.get("job_id") == job
        and all(report.get(k) is False for k in worker.FLAGS),
        "observation identity/scope differs",
    )
    require(
        report.get("python") == PYTHON
        and report.get("sample_interval_seconds") == 5
        and report.get("maximum_child_seconds") == 180
        and report.get("worker_budget_seconds") == 540,
        "observation runtime/budget differs",
    )
    require(
        all(type(report.get(k)) is bool for k in ("diagnostic_complete", "imports_ready")),
        "observation completion type differs",
    )
    assigned = report.get("cuda_visible_devices")
    require(
        isinstance(assigned, str)
        and len(assigned.split(",")) == len(set(assigned.split(","))) == 2
        and all(x and x.strip() == x and x not in {"-1", "NoDevFiles"} for x in assigned.split(","))
        and (visible is None or visible == assigned),
        "observation visibility differs",
    )
    inherited = report.get("inherited_thread_environment")
    require(
        isinstance(inherited, dict)
        and set(inherited) == set(worker.THREAD_KEYS)
        and all(v is None or isinstance(v, str) for v in inherited.values()),
        "inherited threads differ",
    )
    phases = report.get("phases")
    require(
        isinstance(phases, list)
        and len(phases) <= 3
        and [p.get("label") for p in phases] == ["A1", "B", "A2"][: len(phases)],
        "observation conditions missing/reordered",
    )
    raw_reports = reports or {}
    complete, ready, last_end, pids = [], [], None, set()
    for phase in phases:
        label = phase["label"]
        policy = "12" if label == "B" else "inherited"
        threads = dict.fromkeys(worker.THREAD_KEYS, "12") if label == "B" else inherited
        require(
            phase.get("thread_policy") == policy
            and phase.get("thread_environment") == threads
            and phase.get("cuda_visible_devices") == assigned
            and phase.get("cwd") == report.get("cwd"),
            "condition differs from fixed intervention",
        )
        for k in ("timed_out", "reaped", "report_valid", "observation_complete"):
            require(type(phase.get(k)) is bool, "condition status type differs")
        for k in ("time_limit_seconds", "started_at_unix", "elapsed_seconds"):
            require(
                type(phase.get(k)) in (int, float) and math.isfinite(phase[k]) and phase[k] >= 0,
                "condition timing differs",
            )
        require(
            0 < phase["time_limit_seconds"] <= 180 and phase["elapsed_seconds"] <= 186,
            "condition exceeded bounded runtime",
        )
        if last_end is not None:
            require(phase["started_at_unix"] >= last_end - 0.1, "conditions overlap")
        last_end = phase["started_at_unix"] + phase["elapsed_seconds"]
        pid = phase.get("pid")
        require(pid is None or (type(pid) is int and pid > 0 and pid not in pids), "reused/invalid child pid")
        if pid is not None:
            pids.add(pid)
        require(phase.get("exit_code") is None or type(phase["exit_code"]) is int, "exit code differs")
        for key, suffix, cap in (
            ("report", "-report.json", 65536),
            ("log", ".log", CAP),
            ("proc", "-proc.jsonl", CAP // 2),
        ):
            row = phase.get(key)
            if row is None:
                require(key == "report", "condition lacks retained diagnostics")
                continue
            require(
                isinstance(row, dict)
                and row.get("path") == label + suffix
                and type(row.get("size")) is int
                and 0 <= row["size"] <= cap
                and isinstance(row.get("sha256"), str)
                and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
                "condition artifact binding differs",
            )
        log, proc = phase["log"], phase["proc"]
        require(
            type(log.get("bytes_seen")) is int
            and log["bytes_seen"] >= log["size"]
            and log["size"] == min(log["bytes_seen"], CAP)
            and log.get("truncated") is (log["bytes_seen"] > CAP)
            and type(proc.get("sample_count")) is int
            and 0 <= proc["sample_count"] <= 40
            and type(proc.get("truncated")) is bool,
            "bounded process/log summary differs",
        )
        valid_end = False
        expected_ready = None
        if phase["report_valid"]:
            require(label in raw_reports and phase["report"] is not None, "validated report missing")
            raw = raw_reports[label]
            worker.validate_report(
                raw,
                label=label,
                job_id=job,
                expected_cuda_visible_devices=assigned,
                expected_thread_policy=policy,
            )
            require(
                raw["pid"] == pid
                and raw["python_executable"] == PYTHON
                and {k: raw["environment"][k] for k in worker.THREAD_KEYS} == threads,
                "worker process/runtime/thread binding differs",
            )
            require(
                all(
                    raw[k] is phase[p]
                    for k, p in (
                        ("diagnostic_complete", "worker_diagnostic_complete"),
                        ("import_ready", "import_ready"),
                    )
                ),
                "worker observation summary differs",
            )
            require(
                phase["import_succeeded"]
                is (raw["import_succeeded"] if raw["import_elapsed_seconds"] is not None else None),
                "unknown import outcome differs",
            )
            runtime_ok = raw["runtime_actual"].get("python") == worker.EXPECTED_RUNTIME["python"]
            if raw["diagnostic_complete"]:
                runtime_ok = runtime_ok and raw["environment_after"] == raw["environment"]
                if raw["import_succeeded"]:
                    runtime_ok = runtime_ok and raw["runtime_actual"] == worker.EXPECTED_RUNTIME
            if phase["timed_out"]:
                require(
                    phase["elapsed_seconds"] >= phase["time_limit_seconds"] - 0.1
                    and type(phase["exit_code"]) is int,
                    "fabricated timeout boundary",
                )
            valid_end = (
                all(worker.launch_checks(raw).values())
                and runtime_ok
                and raw["instrumentation_error"] is None
                and raw["runtime_error"] is None
                and (
                    (raw["diagnostic_complete"] and phase["exit_code"] == 0)
                    or (phase["timed_out"] and raw["stack_dump_armed"] and raw["import_started"])
                )
            )
            expected_ready = raw["import_ready"]
        expected_complete = bool(
            valid_end
            and phase["reaped"]
            and proc["sample_count"] > 0
            and phase.get("error") is None
            and phase.get("shutdown_error") is None
        )
        require(phase["observation_complete"] is expected_complete, "condition completion fabricated")
        complete.append(expected_complete)
        ready.append(expected_ready is True and expected_complete)
    require(
        report["diagnostic_complete"] is (len(phases) == 3 and all(complete))
        and report["imports_ready"] is (len(phases) == 3 and all(ready)),
        "aggregate completion fabricated",
    )
    return report


def validate_proc_samples(phase, data):
    row = phase["proc"]
    require(len(data) == row["size"] and sha(data) == row["sha256"], "process sample bytes differ")
    samples = [json.loads(line) for line in data.splitlines()]
    require(len(samples) == row["sample_count"], "process sample count differs")
    previous = -1.0
    observed_pid = False
    for sample in samples:
        elapsed = sample.get("elapsed_seconds")
        require(
            sample.get("pid") == phase["pid"]
            and sample.get("label") == phase["label"]
            and type(elapsed) in (int, float)
            and math.isfinite(elapsed)
            and previous <= elapsed <= phase["elapsed_seconds"],
            "process sample identity/time differs",
        )
        previous = elapsed
        require(
            type(sample.get("observed_at_unix")) in (int, float)
            and math.isfinite(sample["observed_at_unix"]),
            "process sample timestamp differs",
        )
        files = sample.get("files")
        require(
            isinstance(files, dict) and set(files) == {"status", "stat", "io", "wchan"},
            "process sample fields differ",
        )
        for name, maximum in (("status", 16384), ("stat", 4096), ("io", 4096), ("wchan", 512)):
            record = files[name]
            require(
                isinstance(record, dict)
                and (
                    (
                        set(record) == {"text", "truncated"}
                        and isinstance(record["text"], str)
                        and len(record["text"].encode()) <= maximum * 3
                        and type(record["truncated"]) is bool
                    )
                    or (
                        set(record) == {"error"}
                        and isinstance(record["error"], str)
                        and len(record["error"]) <= 512
                    )
                ),
                "malformed process observation",
            )
        status = files["status"].get("text", "")
        match = re.search(r"^Pid:\s*(\d+)\s*$", status, re.MULTILINE)
        if match:
            require(int(match.group(1)) == phase["pid"], "raw status belongs to another process")
            observed_pid = True
        if "environment" in sample:
            require(
                sample.get("environment_truncated") is False
                and sample["environment"]
                == dict(phase["thread_environment"], CUDA_VISIBLE_DEVICES=phase["cuda_visible_devices"]),
                "sampled process environment differs",
            )
    if phase["observation_complete"]:
        require(observed_pid, "no live owned child observed")
    return samples


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
        and all(
            published.get(k) is False for k in (*FLAGS, "cuda_observed", "model_loaded", "training_started")
        )
        and type(published.get("diagnostic_complete")) is bool
        and type(published.get("imports_ready")) is bool,
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


validate_memory = _cuda.validate_memory


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
        imports_ready=False,
        publication_verified=False,
        artifact_hashes_verified=False,
        checked_at=now(),
        **dict.fromkeys(FLAGS, False),
    )
    root = safe(plan["result_dir"])
    records = {}
    report = None
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
        if "import-observations.json" in records:
            report = document(root / "import-observations.json")
            reports = {
                label: document(root / (label + "-report.json"))
                for label in (p["label"] for p in report["phases"] if p["report_valid"])
                if label + "-report.json" in records
            }
            validate_observations(report, job, reports=reports)
            require(report["plan_sha256"] == sha(canonical(plan)), "observation plan differs")
            for phase in report["phases"]:
                expected_argv = helper("sdsc_torch_import_probe_job").worker_argv(
                    plan,
                    Path(report["cwd"]),
                    dict(job_id=job, cuda_visible_devices=report["cuda_visible_devices"]),
                    phase["label"],
                )
                require(phase["argv"] == expected_argv, "observed worker argv differs")
                validate_proc_samples(phase, read(root / phase["proc"]["path"]))
                for key in ("report", "log", "proc"):
                    row = phase.get(key)
                    if row is not None:
                        require(
                            row["path"] in records
                            and all(row[k] == records[row["path"]][k] for k in ("path", "size", "sha256")),
                            "phase publication binding differs",
                        )
            require(
                all(not published[k] or report[k] for k in ("diagnostic_complete", "imports_ready")),
                "published observation completion differs",
            )
            result.update(
                diagnostic_complete=published["diagnostic_complete"],
                imports_ready=published["imports_ready"],
                result_sha256=records["import-observations.json"]["sha256"],
            )
    if result["accounting_complete"]:
        require(
            result["publication_verified"] and set(NAMES) == set(records) and result["diagnostic_complete"],
            "completed allocation lacks complete bounded observations",
        )
        node = document(root / "node-result.json")
        require(
            node.get("job_id") == job
            and node.get("plan_sha256") == sha(canonical(plan))
            and node.get("exit_code") == 0
            and all(
                node.get(k) is False for k in (*FLAGS, "cuda_observed", "model_loaded", "training_started")
            )
            and node.get("diagnostic_complete") is True
            and node.get("imports_ready") is report["imports_ready"],
            "node completion differs",
        )
        validate_observations(report, job, visible=node["cuda_visible_devices"], reports=reports)
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
    execution = document(safe(args.failed_plan.absolute()))
    failed = execution["science_plan"]
    fetch = safe(args.failed_fetch_dir.absolute())
    report = document(fetch / "node-result.json", FAILED_REPORT_SHA)
    publication = document(fetch / "receipt.json", FAILED_PUBLICATION_SHA)
    execution_publication = document(fetch / "execution/receipt.json", FAILED_EXECUTION_PUBLICATION_SHA)
    validate_failed_evidence(report, publication, execution_publication)
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
            execution_plan=execution,
            execution_plan_sha256=FAILED_EXECUTION_PLAN_SHA,
            execution_publication_sha256=FAILED_EXECUTION_PUBLICATION_SHA,
        ),
        python=PYTHON,
        scope=SCOPE,
        **dict.fromkeys(FLAGS, False),
    )
    bind_paths(plan)
    validate_plan(plan)
    directory = ROOT / ".sdsc/torch-import-probe" / plan["intent_id"]
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
                path.parent.parent == ROOT / ".sdsc/torch-import-probe" and path.name == "plan.json",
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
