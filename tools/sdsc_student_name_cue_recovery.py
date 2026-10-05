#!/usr/bin/env python3
"""One explicit infrastructure recovery of the immutable name-cue diagnostic."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pwd
import re
import shlex
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
RUNTIME_ARTIFACT_ROOT = PROJECT / "runtime-snapshots"
TASK = "qwen3-v2-student-name-cue-recovery-v1"
SCHEMA = "quest-sdsc-student-name-cue-recovery-plan-v1"
CONTRACT_PATH = "prereg/amendments/qwen3_student_name_cue_execution_recovery_v1.json"
CAP = 1024**2
MAX_PLAN, EXECUTION_MAX, MAX_FETCH, MAX_RESPONSE = 8 * CAP, 64 * CAP, 192 * CAP, 272 * CAP


def helper(name, root=ROOT):
    path = Path(root) / "tools" / (name + ".py")
    if "contract" in globals():
        expected = contract.FROZEN_DEPENDENCIES.get("tools/" + name + ".py")
        if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen recovery dependency changed: " + name)
    spec = importlib.util.spec_from_file_location("_name_cue_recovery_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = helper("sdsc_student_name_cue_recovery_contract")
science = helper("sdsc_student_name_cue_probe")
require, canonical, sha, now = science.require, science.canonical, science.sha, science.now
safe, relative, write_once, run = science.safe, science.relative, science.write_once, science.run
FLAGS = science.FLAGS
NEW_TOOLS = contract.CONTROL_PATHS
TOOLS = tuple(dict.fromkeys((*science.TOOLS, *NEW_TOOLS, CONTRACT_PATH)))
EXECUTION_NAMES = (
    "runtime-manifest.json",
    "runtime-context.json",
    "runtime-stage.json",
    "runtime-early.json",
    *(f"runtime-rank-{r}-{phase}.json" for r in (0, 1) for phase in ("before", "after")),
    "recovery-node-result.json",
    "memory.json",
    "startup.log",
    "trace-meta.json",
    "early-node-startup.json",
    *(f"rank-{r}-{phase}.json" for r in (0, 1) for phase in ("startup", "exit")),
)


def read(path, limit=16 * CAP):
    return science.read(path, limit)


def document(path, expected=None, limit=16 * CAP):
    raw = read(path, limit)
    require(expected is None or sha(raw) == expected, "artifact SHA differs")
    return json.loads(raw)


def resources():
    return science.resources()


def worker_budget_seconds(plan):
    return science.worker_budget_seconds(plan["science_plan"])


def binding(root=ROOT, expected_head=None):
    root = Path(root)
    return (
        helper("sdsc_student_name_cue_recovery_contract", root)
        .resolve_execution_contract(
            root,
            expected_head=expected_head,
            git_dir=root / (".opd-git" if (root / ".opd-git").is_dir() else ".git"),
        )
        .as_dict()
    )


def recovery_claim(plan):
    key = dict(
        failed_job=contract.FAILED["job_id"], science_identity=plan["science_plan"]["science_identity"]
    )
    return str(CONTROL / "student-name-cue-recovery-claims" / (sha(canonical(key)) + ".json"))


def claim_record(plan):
    return dict(
        task=TASK,
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        science_plan_sha256=sha(canonical(plan["science_plan"])),
        failed_job=contract.FAILED["job_id"],
        no_retry=True,
    )


def validate_runtime_snapshot(plan):
    value = plan["runtime_snapshot"]
    helper("sdsc_runtime_snapshot").validate_descriptor(value)
    require(value["manifest"]["size"] <= 16 * CAP, "recovery runtime manifest exceeds16MiB")
    require(
        value["source_prefix"] == contract.RUNTIME_PREFIX
        and value["python_relative_path"] == "bin/python3.12"
        and value["python_sha256"] == contract.PYTHON_SHA256,
        "runtime source identity differs",
    )
    manifest, archive = Path(value["manifest"]["path"]), Path(value["archive"]["path"])
    require(
        manifest.parent == archive.parent
        and manifest.name == "manifest.json"
        and archive.name == "runtime.bin"
        and manifest.parent.parent == RUNTIME_ARTIFACT_ROOT
        and re.fullmatch(r"snapshot-[a-f0-9]{32}", manifest.parent.name),
        "runtime artifact namespace differs",
    )
    return value


def validate_plan(plan):
    require(
        isinstance(plan, dict)
        and set(plan)
        == {
            "schema",
            "task",
            "science_plan",
            "execution",
            "failed",
            "runtime_snapshot",
            "intent_id",
            "recovery_claim",
            *FLAGS,
        },
        "recovery plan fields differ",
    )
    require(
        plan["schema"] == SCHEMA and plan["task"] == TASK and len(canonical(plan)) <= MAX_PLAN,
        "recovery scope/bound differs",
    )
    inner = science.validate_plan(plan["science_plan"])
    execution = plan["execution"]
    require(
        set(execution)
        == {
            "core_sha256",
            "artifact_sha256",
            "implementation_commit",
            "acceptance_commit",
            "head",
            "control_file_sha256",
            "science_binding",
        },
        "execution binding fields differ",
    )
    require(
        execution["science_binding"] == inner["protocol"]
        and execution["head"] == inner["provenance"]["head"],
        "execution and original science binding differ",
    )
    require(
        all(
            isinstance(execution[k], str) and re.fullmatch("[a-f0-9]{64}", execution[k])
            for k in ("core_sha256", "artifact_sha256")
        ),
        "invalid execution SHA",
    )
    require(
        all(
            isinstance(execution[k], str) and re.fullmatch("[a-f0-9]{40}", execution[k])
            for k in ("implementation_commit", "acceptance_commit", "head")
        )
        and execution["implementation_commit"] != execution["acceptance_commit"],
        "distinct execution acceptance required",
    )
    require(
        set(execution["control_file_sha256"]) == set(NEW_TOOLS)
        and all(re.fullmatch("[a-f0-9]{64}", v) for v in execution["control_file_sha256"].values()),
        "execution control inventory differs",
    )
    failed = plan["failed"]
    require(
        set(failed) == {"plan", "receipt", *(set(contract.FAILED) - {"intent_id", "plan_sha256", "job_id"})},
        "failed evidence fields differ",
    )
    old = science.validate_plan(failed["plan"])
    science.validate_submission_receipt(failed["receipt"], old)
    require(
        sha(canonical(old)) == contract.FAILED["plan_sha256"]
        and old["intent_id"] == contract.FAILED["intent_id"]
        and failed["receipt"]["job_id"] == contract.FAILED["job_id"],
        "only exact failed54681802 may recover",
    )
    require(
        all(
            failed[k] == v
            for k, v in contract.FAILED.items()
            if k not in ("intent_id", "plan_sha256", "job_id")
        ),
        "failed artifact pin differs",
    )
    require(
        inner["science_identity"] == old["science_identity"]
        and inner["intent_id"] != old["intent_id"]
        and inner["scientific_claim"] == old["scientific_claim"],
        "recovery must retain original science with fresh execution identity",
    )
    require(
        inner["parents"] == old["parents"]
        and inner["protocol"]["science_file_sha256"] == old["protocol"]["science_file_sha256"],
        "original inputs or named science changed",
    )
    require(
        plan["intent_id"] == inner["intent_id"] and plan["recovery_claim"] == recovery_claim(plan),
        "recovery claim differs",
    )
    require(all(plan[k] is False for k in FLAGS), "recovery cannot accept a model")
    validate_runtime_snapshot(plan)
    return plan


def verify_source(plan, root=None):
    manifest = science.verify_source(plan["science_plan"], root)
    rows = {r["path"]: r for r in manifest["files"]}
    expected = {
        **plan["execution"]["control_file_sha256"],
        CONTRACT_PATH: plan["execution"]["artifact_sha256"],
    }
    require(
        all(rows.get(p, {}).get("sha256") == h for p, h in expected.items()),
        "deployed recovery controls differ",
    )
    return manifest


def verify_execution(plan, destination=None):
    if destination is None:
        with tempfile.TemporaryDirectory(prefix="name-cue-recovery-verify-") as temp:
            return verify_execution(plan, Path(temp) / "science")
    result = science.verify_science(plan["science_plan"], destination)
    require(
        binding(destination, plan["execution"]["head"]) == plan["execution"],
        "restored accepted execution differs",
    )
    return result


def validate_failed_documents(plan, files):
    failed = plan["failed"]
    old = failed["plan"]
    job = contract.FAILED["job_id"]
    required = {
        "receipt.json": "publication_sha256",
        "node-result.json": "node_sha256",
        "trace-meta.json": "trace_sha256",
        "memory.json": "memory_sha256",
        "startup.log": "startup_log_sha256",
    }
    require(set(files) == set(required), "failed publication inventory differs")
    for name, key in required.items():
        require(sha(files[name]) == failed[key], "failed raw artifact differs: " + name)
    pub = json.loads(files["receipt.json"])
    rows = science.publication_records(pub, old, job)
    require(
        set(rows) == set(required) - {"receipt.json"}
        and pub["large_files"] == []
        and pub["diagnostic_complete"] is False
        and pub["stage_complete"] is False,
        "failed job produced scientific artifacts",
    )
    for name, row in rows.items():
        require(
            len(files[name]) == row["size"] and sha(files[name]) == row["sha256"],
            "failed published raw differs",
        )
    node = json.loads(files["node-result.json"])
    trace = json.loads(files["trace-meta.json"])
    work = Path(node["work_dir"])
    require(
        node["error"] == "ValueError: early CUDA probe failed; no large input staged"
        and node["stage_complete"] is False
        and type(node["exit_code"]) is int
        and node["exit_code"] == 1
        and node["job_id"] == job
        and node["plan_sha256"] == contract.FAILED["plan_sha256"],
        "not reviewed pre-inference failure",
    )
    require(
        all(
            k not in node
            for k in ("staged", "worker_exit_code", "worker_pid", "worker_result", "checkpoint_staging")
        ),
        "failed node has inference execution evidence",
    )
    startup = science.helper("sdsc_student_name_cue_probe_startup")
    startup.validate_trace_meta(
        trace,
        worker_sha256=old["control_sha256"]["tools/sdsc_student_name_cue_probe_worker.py"],
        job_id=job,
        expected_cuda_visible_devices=node["cuda_visible_devices"],
        expected_python=old["python"],
        expected_argv=science.helper("sdsc_student_name_cue_probe_job").probe_argv(
            old, work / "artifacts", dict(job_id=job, cuda_visible_devices=node["cuda_visible_devices"])
        ),
        log_bytes=files["startup.log"],
    )
    require(
        trace["timed_out"] is True
        and trace["reaped"] is True
        and trace["probe_passed"] is False
        and trace["maximum_probe_seconds"] == 300
        and trace["shutdown_error"] is None,
        "failed child not fully reconciled",
    )
    events = [json.loads(e["raw"]) for e in trace["adapter_events"] if e["event"] == "frozen_progress"]
    require(
        [e for e in events if e.get("api") == "import_torch"]
        == [
            dict(
                api="import_torch",
                event="begin",
                logical_device=None,
                schema="quest-sdsc-cuda-diagnostic-progress-v1",
            )
        ],
        "failed import completed or trace differs",
    )
    require(not any(str(e.get("api", "")).startswith("cuda.") for e in events), "CUDA already executed")
    require(all(node[k] is False for k in FLAGS), "failure acceptance flag differs")
    return dict(
        no_inference=True,
        early_import_timeout=True,
        failed_job=job,
        publication_sha256=failed["publication_sha256"],
    )


def validate_failed_terminal(status):
    job = contract.FAILED["job_id"]
    q = status["queue"]
    a = status["accounting"]
    require(
        status["job_id"] == job
        and status["state"] == "FAILED"
        and q["returncode"] == a["returncode"] == 0
        and not q["stdout"].strip(),
        "failed terminal status or queue differs",
    )
    rows = [line.split("|") for line in a["stdout"].splitlines() if line.strip()]
    require(
        len(rows) == 3
        and {r[0] for r in rows} == {job, job + ".batch", job + ".extern"}
        and all(
            r[1:3] == (["COMPLETED", "0:0"] if r[0].endswith(".extern") else ["FAILED", "1:0"]) for r in rows
        ),
        "failed job/batch/extern accounting differs",
    )
    require(
        status["publication_verified"] is True
        and status["publication_sha256"] == contract.FAILED["publication_sha256"]
        and status["success"] is False,
        "failed publication/status differs",
    )
    return status


def verify_failed(plan, *, accounting=True):
    failed = plan["failed"]
    old = failed["plan"]
    directory = safe(old["submission_dir"])
    require(
        read(directory / "plan.json") == canonical(old)
        and document(directory / "receipt.json") == failed["receipt"]
        and document(old["claim"])["plan_sha256"] == sha(canonical(old))
        and document(old["scientific_claim"]) == science.scientific_claim_record(old),
        "original failed permanent claims changed",
    )
    if accounting:
        status = science.inspect_job(old, failed["receipt"])
    else:
        status = document(Path(plan["science_plan"]["submission_dir"]) / "admission.json")["failed"]
    validate_failed_terminal(status)
    result = safe(old["result_dir"])
    files = {
        name: read(result / name)
        for name in ("receipt.json", "node-result.json", "trace-meta.json", "memory.json", "startup.log")
    }
    validate_failed_documents(plan, files)
    return status


def verify_runtime_snapshot(plan):
    value = validate_runtime_snapshot(plan)
    directory = Path(value["manifest"]["path"]).parent
    observed = helper("sdsc_runtime_snapshot").describe(
        directory, value["manifest"]["sha256"], artifact_root=RUNTIME_ARTIFACT_ROOT
    )
    require(observed == value, "immutable runtime manifest/descriptor changed")
    archive = safe(value["archive"]["path"])
    require(
        archive.is_file() and archive.stat().st_size == value["archive"]["size"],
        "runtime archive missing or size changed",
    )
    return dict(
        manifest_sha256=value["manifest"]["sha256"],
        archive_sha256=value["archive"]["sha256"],
        archive_bytes=value["archive"]["size"],
        node_full_archive_and_inventory_readback_required=True,
    )


def admission(plan):
    inner = plan["science_plan"]
    failed = verify_failed(plan, accounting=True)
    parents = science.verify_parent(inner, accounting=True)
    verify_execution(plan)
    require(
        not safe(plan["recovery_claim"]).exists()
        and not safe(inner["claim"]).exists()
        and not safe(inner["submission_dir"]).exists(),
        "recovery already claimed; reconcile, never retry",
    )
    require(
        safe(inner["python"]).is_file()
        and os.access(inner["python"], os.X_OK)
        and sha(read(inner["python"], 64 * CAP)) == contract.PYTHON_SHA256,
        "source runtime executable differs",
    )
    require(
        safe(inner["hf_home"]).is_dir() and PROJECT.is_dir() and os.access(PROJECT, os.W_OK),
        "storage unavailable",
    )
    return dict(
        failed=failed,
        parents=parents,
        runtime=science.verify_runtime(inner),
        runtime_snapshot=verify_runtime_snapshot(plan),
        gpu_concurrency=science.check_gpu_ceiling(),
    )


def check_claims(plan):
    inner = plan["science_plan"]
    directory = safe(inner["submission_dir"])
    require(
        read(directory / "execution-plan.json", MAX_PLAN) == canonical(plan)
        and read(directory / "plan.json") == canonical(inner)
        and document(plan["recovery_claim"]) == claim_record(plan)
        and document(inner["claim"])["plan_sha256"] == sha(canonical(inner)),
        "permanent recovery claims differ",
    )
    old = plan["failed"]["plan"]
    require(
        document(old["scientific_claim"]) == science.scientific_claim_record(old),
        "failed original scientific claim changed",
    )


def sbatch(plan):
    argv = science.sbatch(plan["science_plan"])
    argv[-1] = sha(canonical(plan))
    return argv


def job_script(plan):
    inner = plan["science_plan"]
    path = str(Path(inner["release"]) / "source/tools/sdsc_student_name_cue_recovery_job.py")
    launch = (
        "import hashlib,pathlib,runpy,sys;p=pathlib.Path(sys.argv[1]);"
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
        "sys.argv=[str(p)]+sys.argv[3:];runpy.run_path(str(p),run_name='__main__')"
    )
    return (
        "#!/bin/bash\nset -euo pipefail\numask 077\nexec "
        + shlex.join(
            [
                inner["python"],
                "-I",
                "-B",
                "-u",
                "-c",
                launch,
                path,
                plan["execution"]["control_file_sha256"]["tools/sdsc_student_name_cue_recovery_job.py"],
                str(Path(inner["submission_dir"]) / "execution-plan.json"),
            ]
        )
        + ' "$1"\n'
    ).encode()


def record_receipt(plan, job, reconciled=False):
    inner = plan["science_plan"]
    directory = safe(inner["submission_dir"])
    value = science.make_receipt(inner, job, reconciled)
    if not (directory / "receipt.json").exists():
        write_once(directory / "receipt.json", canonical(value))
    else:
        value = document(directory / "receipt.json")
        science.validate_submission_receipt(value, inner)
        require(value["job_id"] == job, "reconciled job differs")
    outer = dict(claim_record(plan), job_id=job, scientific_receipt=value)
    if not (directory / "execution-receipt.json").exists():
        write_once(directory / "execution-receipt.json", canonical(outer))
    require(document(directory / "execution-receipt.json") == outer, "execution receipt differs")
    return value


def reconcile(plan):
    inner = plan["science_plan"]
    user = pwd.getpwuid(os.getuid()).pw_name
    queue = run(
        ["squeue", "--noheader", "--user=" + user, "--name=" + inner["job_name"], "--format=%i|%j|%k"]
    )
    account = run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--user=" + user,
            "--name=" + inner["job_name"],
            "--starttime=" + inner["created_at"][:10],
            "--format=JobIDRaw,JobName%100,Comment%100,User%64",
        ]
    )
    require(queue["returncode"] == account["returncode"] == 0, "reconciliation unavailable; never retry")
    ids = set()
    for line in (queue["stdout"] + "\n" + account["stdout"]).splitlines():
        fields = line.split("|")
        if (
            len(fields) >= 3
            and fields[1:3] == [inner["job_name"], inner["intent_id"]]
            and re.fullmatch("[1-9][0-9]*", fields[0])
            and (len(fields) == 3 or fields[3] == user)
        ):
            ids.add(fields[0])
    require(len(ids) == 1, "zero/ambiguous acknowledgement; never resubmit")
    return record_receipt(plan, ids.pop(), True)


def remote_action(request):
    plan = validate_plan(request["plan"])
    verify_source(plan)
    inner = plan["science_plan"]
    directory = safe(inner["submission_dir"])
    action = request["action"]
    if action == "dry-run":
        return dict(
            dry_run=True,
            plan_sha256=sha(canonical(plan)),
            science_plan_sha256=sha(canonical(inner)),
            argv=sbatch(plan),
            resources=inner["resources"],
            source_verified=True,
            parent_verified=True,
            accepted_protocol_verified=True,
            accepted_execution_verified=True,
            node_mounts_still_required=True,
            blockers=[],
            **admission(plan),
        )
    if action == "submit":
        proof = request.get("dry_run")
        require(
            request.get("authorize") is True
            and isinstance(proof, dict)
            and proof.get("dry_run") is True
            and proof.get("blockers") == []
            and proof.get("plan_sha256") == sha(canonical(plan))
            and proof.get("argv") == sbatch(plan),
            "authorized matching dry-run required",
        )
        checks = admission(plan)
        for path, value in (
            (plan["recovery_claim"], claim_record(plan)),
            (inner["claim"], dict(intent_id=inner["intent_id"], plan_sha256=sha(canonical(inner)), at=now())),
        ):
            safe(path).parent.mkdir(parents=True, exist_ok=True)
            write_once(safe(path), canonical(value))
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=0o700)
        for name, raw in (
            ("plan.json", canonical(inner)),
            ("execution-plan.json", canonical(plan)),
            ("job.sh", job_script(plan)),
            ("admission.json", canonical(checks)),
            ("submission-started.json", canonical(dict(argv=sbatch(plan), at=now()))),
        ):
            write_once(directory / name, raw)
        try:
            result = run(sbatch(plan))
            match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\s*", result["stdout"])
            require(result["returncode"] == 0 and match is not None, "missing trustworthy acknowledgement")
            receipt = record_receipt(plan, match.group(1))
            write_once(
                directory / "live-binding.json", canonical(science.live_binding(inner, receipt["job_id"]))
            )
            return receipt
        except BaseException as error:
            write_once(
                directory / "unknown.json", canonical(dict(error=str(error)[:1000], no_retry=True, at=now()))
            )
            raise
    check_claims(plan)
    if action == "reconcile":
        receipt = reconcile(plan)
        if not (directory / "live-binding.json").exists():
            write_once(
                directory / "live-binding.json", canonical(science.live_binding(inner, receipt["job_id"]))
            )
        return receipt
    receipt = document(directory / "receipt.json")
    science.validate_submission_receipt(receipt, inner)
    require(
        document(directory / "execution-receipt.json")
        == dict(claim_record(plan), job_id=receipt["job_id"], scientific_receipt=receipt),
        "submission execution receipt differs",
    )
    status = inspect_job(plan, receipt)
    if action == "status":
        return status
    require(action == "fetch", "unknown operation")
    files = {}
    root = safe(inner["result_dir"])
    if status["publication_verified"]:
        records = science.publication_records(status["publication"], inner, receipt["job_id"])
        for name in (*records, "receipt.json"):
            raw = read(root / name)
            expected = status["publication_sha256"] if name == "receipt.json" else records[name]["sha256"]
            require(sha(raw) == expected, "scientific fetch changed")
            files[name] = raw
    if status["execution_publication_verified"]:
        for row in status["execution_publication"]["files"]:
            raw = read(root / "execution" / row["path"], 16 * CAP)
            require(sha(raw) == row["sha256"] and len(raw) == row["size"], "execution fetch changed")
            files["execution/" + row["path"]] = raw
        raw = read(root / "execution/receipt.json", EXECUTION_MAX)
        require(sha(raw) == status["execution_publication_sha256"], "execution receipt changed")
        files["execution/receipt.json"] = raw
    for name, path in (
        ("worker.log", root / "worker.log"),
        ("slurm.out", directory / ("slurm-" + receipt["job_id"] + ".out")),
        ("slurm.err", directory / ("slurm-" + receipt["job_id"] + ".err")),
    ):
        if path.exists():
            with safe(path).open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                files[name] = stream.read(65536)
    total = sum(map(len, files.values()))
    require(total <= MAX_FETCH, "bounded fetch exceeded")
    return dict(
        status=status,
        receipt=receipt,
        files={k: base64.b64encode(v).decode() for k, v in files.items()},
        bytes=total,
    )


def execution_publication(plan, job):
    root = safe(Path(plan["science_plan"]["result_dir"]) / "execution")
    if not (root / "receipt.json").exists():
        return dict(execution_publication_verified=False, execution_complete=False)
    raw = read(root / "receipt.json", CAP)
    pub = json.loads(raw)
    require(pub["job_id"] == job, "execution job differs")
    audit = helper("sdsc_student_name_cue_recovery_audit")
    rows = audit.validate_execution_publication(plan, pub)
    files = {name: read(root / name, 16 * CAP) for name in rows}
    audit.validate_execution_publication(plan, pub, files)
    if pub["execution_complete"]:
        audit.validate_execution_evidence(plan, files)
    return dict(
        execution_publication_verified=True,
        execution_complete=pub["execution_complete"],
        execution_publication=pub,
        execution_publication_sha256=sha(raw),
    )


def inspect_job(plan, receipt, *, accounting_snapshot=None):
    inner = plan["science_plan"]
    science.validate_submission_receipt(receipt, inner)
    job = receipt["job_id"]
    directory = safe(inner["submission_dir"])
    observed = (
        document(directory / "live-binding.json") if (directory / "live-binding.json").exists() else None
    )
    if accounting_snapshot is None:
        queue, account = science.job_queue(job), science.job_accounting(job)
    else:
        require(
            accounting_snapshot["job_id"] == job
            and accounting_snapshot["plan_sha256"] == sha(canonical(inner)),
            "accounting snapshot differs",
        )
        queue, account = accounting_snapshot["queue"], accounting_snapshot["accounting"]
    result = science.validate_accounting(inner, job, queue, account, observed)
    result.update(
        success=False,
        diagnostic_complete=False,
        stage_complete=False,
        preparation_complete=False,
        publication_verified=False,
        artifact_hashes_verified=False,
        checked_at=now(),
        **dict.fromkeys(FLAGS, False),
    )
    root = safe(inner["result_dir"])
    rows = {}
    if (root / "receipt.json").exists():
        raw = read(root / "receipt.json")
        pub = json.loads(raw)
        rows = science.publication_records(pub, inner, job)
        require(
            {n for n in science.NAMES if (root / n).exists()} == set(rows),
            "scientific publication inventory differs",
        )
        for name, row in rows.items():
            data = read(root / name)
            require(
                len(data) == row["size"] and sha(data) == row["sha256"], "scientific artifact hash differs"
            )
        result.update(
            publication_verified=True,
            artifact_hashes_verified=True,
            publication=pub,
            publication_sha256=sha(raw),
        )
    result.update(execution_publication(plan, job))
    if result["execution_publication_verified"]:
        require(
            result["publication_verified"]
            and result["execution_publication"]["scientific_publication_sha256"]
            == result["publication_sha256"],
            "outer publication does not bind original publication",
        )
    if result["accounting_complete"]:
        require(
            result["publication_verified"]
            and result["execution_publication_verified"]
            and result["execution_complete"]
            and set(rows) == set(science.NAMES)
            and pub["diagnostic_complete"] is True,
            "completed allocation lacks complete scientific and runtime evidence",
        )
        science.validate_worker_report(document(root / "name-cue-report.json"), inner, job, rows)
        science.validate_memory(document(root / "memory.json"), job)
        node = document(root / "node-result.json")
        require(
            node["stage_complete"] is True
            and type(node["exit_code"]) is int
            and node["exit_code"] == 0
            and node["job_id"] == job
            and node["plan_sha256"] == sha(canonical(inner))
            and all(node[k] is False for k in FLAGS),
            "original science node identity differs",
        )
        result.update(
            success=True,
            diagnostic_complete=True,
            stage_complete=True,
            result_sha256=sha(read(root / "name-cue-report.json")),
        )
    return result


def validate_download(result, plan):
    inner = plan["science_plan"]
    science.validate_submission_receipt(result["receipt"], inner)
    allowed = {
        *science.NAMES,
        "receipt.json",
        "worker.log",
        "slurm.out",
        "slurm.err",
        *("execution/" + n for n in (*EXECUTION_NAMES, "receipt.json")),
    }
    require(isinstance(result["files"], dict) and set(result["files"]) <= allowed, "unexpected fetch path")
    files = {n: base64.b64decode(v, validate=True) for n, v in result["files"].items()}
    require(sum(map(len, files.values())) == result["bytes"] <= MAX_FETCH, "fetch total bound differs")
    executable = {n[len("execution/") :]: v for n, v in files.items() if n.startswith("execution/")}
    require(
        sum(map(len, executable.values())) <= EXECUTION_MAX
        and all(
            len(v) <= (CAP if n in ("startup.log", "receipt.json") else 16 * CAP)
            for n, v in executable.items()
        ),
        "execution fetch bound differs",
    )
    scientific = {n: v for n, v in result["files"].items() if not n.startswith("execution/")}
    science.validate_download(
        dict(result, files=scientific, bytes=sum(len(files[n]) for n in scientific)), inner
    )
    status = result["status"]
    if status["execution_publication_verified"]:
        pub = json.loads(executable["receipt.json"])
        require(
            sha(executable["receipt.json"]) == status["execution_publication_sha256"]
            and pub == status["execution_publication"]
            and pub["job_id"] == result["receipt"]["job_id"]
            and status["publication_verified"] is True
            and pub["scientific_publication_sha256"] == status["publication_sha256"],
            "execution receipt binding differs",
        )
        audit = helper("sdsc_student_name_cue_recovery_audit")
        rows = audit.validate_execution_publication(
            plan, pub, {n: v for n, v in executable.items() if n != "receipt.json"}
        )
        require(set(executable) == {"receipt.json", *rows}, "execution download inventory differs")
        if pub["execution_complete"]:
            audit.validate_execution_evidence(
                plan, {n: v for n, v in executable.items() if n != "receipt.json"}
            )
    else:
        require(not executable, "unpublished execution files supplied")
    return files


def prepare(args, cli):
    inner = science.validate_plan(document(safe(args.science_plan.absolute())))
    execution = binding(ROOT, inner["provenance"]["head"])
    record = cli.run_record(inner["run_id"])
    require(record["state"] == "deployed", "first sync dry-run and sync")
    rows = {r["path"]: r for r in record["manifest"]["files"]}
    require(record["manifest"]["code_sha256"] == inner["code_sha256"], "release code differs")
    pins = {**execution["control_file_sha256"], CONTRACT_PATH: execution["artifact_sha256"]}
    require(
        all(rows.get(p, {}).get("sha256") == h == sha(read(ROOT / p)) for p, h in pins.items()),
        "deployed/local recovery controls differ",
    )
    old = document(
        ROOT / ".sdsc/student-name-cue-probe" / contract.FAILED["intent_id"] / "plan.json",
        contract.FAILED["plan_sha256"],
    )
    fetch = safe(args.failed_fetch_dir.absolute())
    fm = document(fetch / "fetch-manifest.json")
    failed = dict(
        plan=old,
        receipt=fm["receipt"],
        **{k: v for k, v in contract.FAILED.items() if k not in ("job_id", "intent_id", "plan_sha256")},
    )
    plan = dict(
        schema=SCHEMA,
        task=TASK,
        science_plan=inner,
        execution=execution,
        failed=failed,
        runtime_snapshot=document(safe(args.runtime_snapshot_json.absolute())),
        intent_id=inner["intent_id"],
        **dict.fromkeys(FLAGS, False),
    )
    plan["recovery_claim"] = recovery_claim(plan)
    validate_plan(plan)
    validate_failed_terminal(document(safe(args.failed_status_file.absolute())))
    validate_failed_documents(
        plan,
        {
            n: read(fetch / n)
            for n in ("receipt.json", "node-result.json", "trace-meta.json", "memory.json", "startup.log")
        },
    )
    directory = ROOT / ".sdsc/student-name-cue-recovery" / plan["intent_id"]
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(directory / "plan.json", canonical(plan))
    return dict(
        plan=str(directory / "plan.json"),
        plan_sha256=sha(canonical(plan)),
        science_plan_sha256=sha(canonical(inner)),
        resources=inner["resources"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "fetch", "reconcile", "remote"))
    for name in (
        "science-plan",
        "failed-fetch-dir",
        "failed-status-file",
        "runtime-snapshot-json",
        "plan",
        "dry-run-file",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "remote":
        require(Path(__file__).resolve().is_relative_to(CONTROL / "releases"), "no local Slurm operations")
        raw = sys.stdin.buffer.read(MAX_PLAN + CAP + 1)
        require(len(raw) <= MAX_PLAN + CAP, "remote request exceeds bound")
        result = remote_action(json.loads(raw))
    else:
        cli = science.helper("sdsc_cli")
        if args.action == "prepare":
            require(
                args.science_plan
                and args.failed_fetch_dir
                and args.failed_status_file
                and args.runtime_snapshot_json,
                "prepare requires fresh science plan, failed fetch/status and runtime descriptor",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "plan required")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-name-cue-recovery" and path.name == "plan.json",
                "local execution plan path differs",
            )
            plan = validate_plan(json.loads(read(path, MAX_PLAN)))
            require(
                path.parent.name == plan["intent_id"]
                and all(
                    sha(read(ROOT / p)) == h for p, h in plan["execution"]["control_file_sha256"].items()
                ),
                "local execution controls changed",
            )
            inner = plan["science_plan"]
            action = "dry-run" if args.action == "submit" and args.dry_run else args.action
            proof = None
            if action == "submit":
                require(
                    args.authorize and args.dry_run_file is not None, "authorize and dry-run-file required"
                )
                proof = document(safe(args.dry_run_file.absolute()))
                cli.require_master()
                write_once(path.parent / "submission-started.json", canonical(dict(at=now(), no_retry=True)))
            script = str(Path(inner["release"]) / "source/tools/sdsc_student_name_cue_recovery.py")
            launch = (
                "import hashlib,pathlib,runpy,sys;p=pathlib.Path(sys.argv[1]);"
                "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
                "sys.argv=[str(p),'remote'];runpy.run_path(str(p),run_name='__main__')"
            )
            response = cli.ssh_call(
                [
                    inner["python"],
                    "-I",
                    "-B",
                    "-c",
                    launch,
                    script,
                    plan["execution"]["control_file_sha256"]["tools/sdsc_student_name_cue_recovery.py"],
                ],
                data=canonical(dict(action=action, plan=plan, authorize=args.authorize, dry_run=proof)),
                timeout=300,
            )
            require(
                response.returncode == 0,
                "remote operation failed/acknowledgement unknown; reconcile: "
                + response.stderr.decode("utf8", "replace")[-1500:],
            )
            require(len(response.stdout) <= MAX_RESPONSE, "remote response bound exceeded")
            result = json.loads(response.stdout)
            if action == "fetch":
                files = validate_download(result, plan)
                directory = ROOT / ".sdsc/fetched" / result["receipt"]["job_id"]
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                destination = Path(tempfile.mkdtemp(prefix="fetch-", dir=directory))
                for name, raw in files.items():
                    (destination / name).parent.mkdir(parents=True, exist_ok=True)
                    write_once(destination / name, raw)
                    require(read(destination / name) == raw, "local fetch readback differs")
                result.pop("files")
                result["destination"] = str(destination)
                write_once(destination / "fetch-manifest.json", canonical(result))
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
            write_once(path.parent / (action + "-" + stamp + ".json"), canonical(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
