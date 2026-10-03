#!/usr/bin/env python3
"""One accepted execution recovery around the unchanged V4 scientific plan.

Generate the inner plan with sdsc_student_order.py prepare first. This adapter
owns recovery claims and startup evidence; the original auditor owns science.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
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
TASK = "qwen3-v2-student-order-execution-v1"
SCHEMA = "quest-sdsc-student-order-execution-plan-v1"
CONTRACT_PATH = "prereg/amendments/qwen3_student_order_execution_recovery_v1.json"
NEW_TOOLS = tuple(
    "tools/sdsc_student_order_execution" + suffix + ".py" for suffix in ("_contract", "", "_job", "_worker")
)
FROZEN = {
    "sdsc_student_order": "999c0bc54991200bd7e1dc7838f432619d5862699433cc7b35b9f0b846123c40",
    "sdsc_student_order_job": "d717f645c78e5c131d53b83ace99201bfb9dc28d33deac00c0e3a2cafcd195d5",
    "sdsc_cuda_diagnostic": "b7b082ddad9595d8365522834ec02b691b92287c72f0a506b5ef2ed0871f181c",
}


def helper(name, root=ROOT):
    import hashlib

    path = root / "tools" / (name + ".py")
    if name in FROZEN and hashlib.sha256(path.read_bytes()).hexdigest() != FROZEN[name]:
        raise ValueError("frozen dependency changed: " + name)
    spec = importlib.util.spec_from_file_location("_execution_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


science = helper("sdsc_student_order")
diag = helper("sdsc_cuda_diagnostic")
require, canonical, sha, now = science.require, science.canonical, science.sha, science.now
safe, read, document, write_once, run = (
    science.safe,
    science.read,
    science.document,
    science.write_once,
    science.run,
)
CONTROL, PROJECT, FLAGS = science.CONTROL, science.PROJECT, science.FLAGS
CAP = science.CAP
MAX_PLAN = 8 * CAP
EXECUTION_NAMES = (
    "early-node-startup.json",
    "rank-0-startup.json",
    "rank-1-startup.json",
    "rank-0-exit.json",
    "rank-1-exit.json",
    "execution-node.json",
    "startup.log",
)
EXECUTION_MAX = CAP
FROZEN_WORKER_SHA = "e5429b4eff6885513f708207e17a49036a1dcb847f9044231d2ae1160d3c3382"


def binding(root=ROOT, expected_head=None):
    return (
        helper("sdsc_student_order_execution_contract", root)
        .resolve_execution_contract(
            root,
            expected_head=expected_head,
            git_dir=root / (".opd-git" if (root / ".opd-git").is_dir() else ".git"),
        )
        .as_dict()
    )


def recovery_claim(plan):
    key = dict(
        execution_core=plan["execution"]["core_sha256"],
        science_identity=plan["science_plan"]["science_identity"],
    )
    return str(CONTROL / "student-order-execution-claims" / (sha(canonical(key)) + ".json"))


def claim_record(plan):
    return dict(
        task=TASK,
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        science_plan_sha256=sha(canonical(plan["science_plan"])),
        no_retry=True,
    )


def validate_plan(plan):
    require(
        isinstance(plan, dict)
        and set(plan)
        == {
            "schema",
            "task",
            "science_plan",
            "execution",
            "diagnostic_plan",
            "preflight_execution",
            "intent_id",
            "recovery_claim",
            *FLAGS,
        },
        "execution plan fields differ",
    )
    require(
        plan["schema"] == SCHEMA and plan["task"] == TASK and len(canonical(plan)) <= MAX_PLAN,
        "execution plan scope/bound differs",
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
        "execution/science provenance differs",
    )
    require(
        all(
            isinstance(execution[k], str) and re.fullmatch("[a-f0-9]{64}", execution[k])
            for k in ("core_sha256", "artifact_sha256")
        ),
        "execution hash differs",
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
        "execution controls differ",
    )
    contract = helper("sdsc_student_order_execution_contract")
    diagnostic = diag.validate_plan(plan["diagnostic_plan"])
    require(sha(canonical(diagnostic)) == contract.DIAGNOSTIC["plan_sha256"], "unreviewed diagnostic plan")
    require(
        plan["intent_id"] == inner["intent_id"] and plan["recovery_claim"] == recovery_claim(plan),
        "recovery identity differs",
    )
    require(all(plan[k] is False for k in FLAGS), "execution recovery cannot grant scientific acceptance")
    previous = plan["preflight_execution"]
    if inner["mode"] == "fit":
        require(
            isinstance(previous, dict) and previous["science_plan"]["mode"] == "preflight",
            "matching recovery preflight required",
        )
        validate_plan(previous)
        require(
            previous["science_plan"] == inner["preflight"]["plan"]
            and previous["execution"] == execution
            and previous["diagnostic_plan"] == diagnostic,
            "fit/recovery preflight binding differs",
        )
    else:
        require(previous is None, "preflight cannot resume an execution")
    return plan


def verify_source(plan, root=None):
    inner = plan["science_plan"]
    manifest = science.verify_source(inner, root)
    rows = science.manifest_records(manifest)
    expected = dict(plan["execution"]["control_file_sha256"])
    expected[CONTRACT_PATH] = plan["execution"]["artifact_sha256"]
    expected.update(helper("sdsc_student_order_execution_contract").FROZEN_DEPENDENCIES)
    require(
        all(rows.get(p, {}).get("sha256") == h for p, h in expected.items()),
        "deployed execution controls differ",
    )
    return manifest


def verify_execution(plan, destination=None):
    if destination is None:
        with tempfile.TemporaryDirectory(prefix="order-execution-verify-") as temporary:
            return verify_execution(plan, Path(temporary) / "science")
    result = science.verify_science(plan["science_plan"], destination)
    require(
        binding(destination, plan["execution"]["head"]) == plan["execution"],
        "restored execution acceptance differs",
    )
    return result


def validate_startup(report, job, cvd, scope, rank=None, original_argv=None):
    worker = helper("sdsc_student_order_execution_worker")
    checked = worker.validate_report(
        report,
        job_id=job,
        expected_cuda_visible_devices=cvd,
        scope=scope,
        rank=rank,
        original_argv=original_argv,
    )
    require(
        checked["cuda_ready"] is True and checked["diagnostic_complete"] is True,
        "actual allocation CUDA startup failed",
    )
    return checked


def validate_rank_exit(report, job, rank, original_argv):
    require(
        report.get("schema") == "quest-sdsc-student-order-execution-exit-v1"
        and report.get("job_id") == job
        and type(report.get("rank")) is int
        and report["rank"] == rank
        and report.get("original_worker_sha256") == FROZEN_WORKER_SHA
        and report.get("original_argv") == original_argv
        and report.get("worker_invoked") is True
        and type(report.get("exit_code")) is int
        and report["exit_code"] == 0
        and "error" not in report
        and all(report.get(k) is False for k in FLAGS),
        "original rank worker did not exit successfully",
    )
    return report


def execution_publication(plan, job):
    root = safe(Path(plan["science_plan"]["result_dir"]) / "execution")
    if not (root / "receipt.json").exists():
        return dict(execution_publication_verified=False, startup_passed=False)
    raw = read(root / "receipt.json", EXECUTION_MAX)
    receipt = json.loads(raw)
    expected = dict(
        schema="quest-sdsc-student-order-execution-publication-v1",
        task=TASK,
        job_id=job,
        plan_sha256=sha(canonical(plan)),
        science_plan_sha256=sha(canonical(plan["science_plan"])),
        execution_core_sha256=plan["execution"]["core_sha256"],
        persistent_read_back_verified=True,
    )
    require(
        all(receipt.get(k) == v for k, v in expected.items()) and all(receipt.get(k) is False for k in FLAGS),
        "execution publication identity differs",
    )
    rows = receipt.get("files")
    require(isinstance(rows, list) and len(rows) <= len(EXECUTION_NAMES), "execution inventory differs")
    records, total = {}, 0
    for row in rows:
        name = row.get("path")
        require(
            set(row) == {"path", "size", "sha256"}
            and name in EXECUTION_NAMES
            and name not in records
            and type(row["size"]) is int
            and 0 <= row["size"] <= 128 * 1024,
            "execution row differs",
        )
        data = read(root / name, 128 * 1024)
        require(len(data) == row["size"] and sha(data) == row["sha256"], "execution artifact hash differs")
        total += len(data)
        records[name] = row
    require(
        total <= EXECUTION_MAX
        and "execution-node.json" in records
        and type(receipt.get("startup_passed")) is bool,
        "execution publication incomplete",
    )
    node = document(root / "execution-node.json")
    require(
        node.get("job_id") == job
        and node.get("plan_sha256") == sha(canonical(plan))
        and all(node.get(k) is False for k in FLAGS),
        "execution node identity differs",
    )
    if receipt["startup_passed"]:
        require(
            node.get("startup_passed") is True and node.get("worker_shutdown_error") is None,
            "execution shutdown/startup failed",
        )
        for name, scope, rank in (
            ("early-node-startup.json", "early_node", None),
            ("rank-0-startup.json", "rank_entry", 0),
            ("rank-1-startup.json", "rank_entry", 1),
        ):
            require(name in records, "startup evidence missing")
            validate_startup(
                document(root / name),
                job,
                node["cuda_visible_devices"],
                scope,
                rank,
                [
                    "--inputs-json",
                    str(Path(node["work_dir"]) / "inputs.json"),
                    "--output-dir",
                    str(Path(node["work_dir"]) / "artifacts"),
                ]
                if rank is not None
                else None,
            )
    return dict(
        execution_publication_verified=True,
        startup_passed=receipt["startup_passed"],
        execution_publication=receipt,
        execution_publication_sha256=sha(raw),
    )


def inspect_job(plan, receipt, *, accounting_snapshot=None):
    inner = plan["science_plan"]
    result = science.inspect_job(inner, receipt, accounting_snapshot=accounting_snapshot)
    evidence = execution_publication(plan, receipt["job_id"])
    if evidence["execution_publication_verified"]:
        node = document(Path(inner["result_dir"]) / "execution/execution-node.json")
        require(
            result["publication_verified"]
            and node.get("science_publication_sha256") == result["publication_sha256"],
            "execution publication does not bind scientific publication",
        )
    if result["success"] and evidence["startup_passed"]:
        records = {r["path"] for r in evidence["execution_publication"]["files"]}
        require(node.get("scientific_stage_complete") is True, "execution node scientific completion differs")
        for rank in (0, 1):
            name = f"rank-{rank}-exit.json"
            require(name in records, "successful rank exit evidence missing")
            validate_rank_exit(
                document(Path(inner["result_dir"]) / "execution" / name),
                receipt["job_id"],
                rank,
                [
                    "--inputs-json",
                    str(Path(node["work_dir"]) / "inputs.json"),
                    "--output-dir",
                    str(Path(node["work_dir"]) / "artifacts"),
                ],
            )
    result.update(evidence)
    result["scientific_stage_complete"] = result["stage_complete"]
    result["success"] = (
        result["success"] and evidence["execution_publication_verified"] and evidence["startup_passed"]
    )
    result["stage_complete"] = result["stage_complete"] and result["success"]
    result["preparation_complete"] = result["preparation_complete"] and result["success"]
    return result


def verify_preflight(plan, *, accounting=True):
    if plan["science_plan"]["mode"] == "preflight":
        return None
    previous = plan["preflight_execution"]
    check_claims(previous)
    inner = previous["science_plan"]
    receipt = document(Path(inner["submission_dir"]) / "receipt.json")
    science.validate_submission_receipt(receipt, inner)
    require(
        document(Path(inner["submission_dir"]) / "execution-receipt.json")
        == dict(claim_record(previous), job_id=receipt["job_id"], scientific_receipt=receipt),
        "preflight execution acknowledgement missing or changed; reconcile",
    )
    saved = (
        None
        if accounting
        else document(Path(plan["science_plan"]["submission_dir"]) / "admission.json")["preflight"]
    )
    result = inspect_job(previous, receipt, accounting_snapshot=saved)
    require(
        result["success"] is True and result["stage_complete"] is True,
        "matching actual recovery preflight incomplete",
    )
    science.validate_preflight_audit(plan["science_plan"]["preflight"]["audit"], inner, result)
    return dict(result, plan_sha256=sha(canonical(inner)))


def prerequisite_status(plan):
    diagnostic = plan["diagnostic_plan"]
    # These existing entrypoints also check stored plans and permanent claims.
    prior = diag.verify_failed(diagnostic)
    observed = diag.remote_action(dict(action="status", plan=diagnostic))
    helper("sdsc_student_order_execution_contract").validate_prerequisites(prior, observed)
    return dict(failed=prior, diagnostic=observed)


def admission(plan):
    inner = plan["science_plan"]
    prerequisites = prerequisite_status(plan)
    science.verify_parent(inner, accounting=True)
    verify_execution(plan)
    preflight = verify_preflight(plan)
    require(
        not safe(plan["recovery_claim"]).exists()
        and not safe(inner["claim"]).exists()
        and not safe(inner["submission_dir"]).exists(),
        "execution already claimed; reconcile, never retry",
    )
    if inner["mode"] == "fit":
        require(not safe(inner["scientific_claim"]).exists(), "V4 fit already claimed")
    require(
        safe(inner["python"]).is_file()
        and os.access(inner["python"], os.X_OK)
        and sha(read(inner["python"], 64 * CAP))
        == "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
        "fixed runtime executable differs",
    )
    require(
        safe(inner["hf_home"]).is_dir() and PROJECT.is_dir() and os.access(PROJECT, os.W_OK),
        "storage unavailable",
    )
    return dict(
        prerequisites=prerequisites,
        preflight=preflight,
        runtime=science.verify_runtime(inner),
        gpu_concurrency=science.check_gpu_ceiling(),
    )


def sbatch(plan):
    argv = science.sbatch(plan["science_plan"])
    argv[-1] = sha(canonical(plan))
    return argv


def job_script(plan):
    inner = plan["science_plan"]
    path = str(Path(inner["release"]) / "source/tools/sdsc_student_order_execution_job.py")
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
                plan["execution"]["control_file_sha256"][NEW_TOOLS[2]],
                str(Path(inner["submission_dir"]) / "execution-plan.json"),
            ]
        )
        + ' "$1"\n'
    ).encode()


def check_claims(plan):
    inner = plan["science_plan"]
    directory = safe(inner["submission_dir"])
    require(
        read(directory / "execution-plan.json", MAX_PLAN) == canonical(plan)
        and read(directory / "plan.json") == canonical(inner)
        and document(plan["recovery_claim"]) == claim_record(plan)
        and document(inner["claim"])["plan_sha256"] == sha(canonical(inner)),
        "permanent execution claims differ",
    )
    if inner["mode"] == "fit":
        require(
            document(inner["scientific_claim"]) == science.scientific_claim_record(inner),
            "V4 fit scientific claim differs",
        )


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
        if inner["mode"] == "fit":
            safe(inner["scientific_claim"]).parent.mkdir(parents=True, exist_ok=True)
            write_once(safe(inner["scientific_claim"]), canonical(science.scientific_claim_record(inner)))
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
            raw = read(root / "execution" / row["path"], 128 * 1024)
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
    require(total <= science.MAX_FETCH, "bounded fetch exceeded")
    return dict(
        status=status,
        receipt=receipt,
        files={k: base64.b64encode(v).decode() for k, v in files.items()},
        bytes=total,
    )


def validate_download(result, plan):
    science.validate_submission_receipt(result["receipt"], plan["science_plan"])
    allowed = {
        *science.NAMES,
        "receipt.json",
        "worker.log",
        "slurm.out",
        "slurm.err",
        *("execution/" + name for name in (*EXECUTION_NAMES, "receipt.json")),
    }
    require(isinstance(result["files"], dict) and set(result["files"]) <= allowed, "unexpected fetched name")
    files = {name: base64.b64decode(value, validate=True) for name, value in result["files"].items()}
    total = sum(map(len, files.values()))
    execution_files = {k: v for k, v in files.items() if k.startswith("execution/")}
    require(
        sum(map(len, execution_files.values())) <= EXECUTION_MAX
        and all(len(v) <= 128 * 1024 for v in execution_files.values()),
        "download startup bound differs",
    )
    require(total == result["bytes"] <= science.MAX_FETCH, "download byte bound differs")
    scientific = {name: value for name, value in result["files"].items() if not name.startswith("execution/")}
    science.validate_download(
        dict(result, files=scientific, bytes=sum(len(files[n]) for n in scientific)), plan["science_plan"]
    )
    status = result["status"]
    if status["execution_publication_verified"]:
        raw = files["execution/receipt.json"]
        receipt = json.loads(raw)
        require(
            sha(raw) == status["execution_publication_sha256"]
            and receipt == status["execution_publication"]
            and receipt["plan_sha256"] == sha(canonical(plan)),
            "download execution receipt differs",
        )
        for row in receipt["files"]:
            raw = files["execution/" + row["path"]]
            require(len(raw) == row["size"] and sha(raw) == row["sha256"], "download execution bytes differ")
        require(
            set(execution_files)
            == {"execution/receipt.json", *("execution/" + r["path"] for r in receipt["files"])},
            "unexpected execution download inventory",
        )
    else:
        require(not execution_files, "unpublished execution files cannot be downloaded")
    return files


def prepare(args, cli):
    inner = science.validate_plan(document(safe(args.science_plan.absolute())))
    record = cli.run_record(inner["run_id"])
    require(
        record["state"] == "deployed" and record["manifest"]["code_sha256"] == inner["code_sha256"],
        "matching deployed release required",
    )
    execution = binding(ROOT, inner["protocol"]["head"])
    rows = science.manifest_records(record["manifest"])
    pins = dict(execution["control_file_sha256"], **{CONTRACT_PATH: execution["artifact_sha256"]})
    require(
        all(rows.get(p, {}).get("sha256") == h == sha(read(ROOT / p)) for p, h in pins.items()),
        "local/release execution pins differ",
    )
    plan = dict(
        schema=SCHEMA,
        task=TASK,
        science_plan=inner,
        execution=execution,
        diagnostic_plan=document(safe(args.diagnostic_plan.absolute())),
        preflight_execution=document(safe(args.preflight_execution_plan.absolute()))
        if args.preflight_execution_plan
        else None,
        intent_id=inner["intent_id"],
        **dict.fromkeys(FLAGS, False),
    )
    plan["recovery_claim"] = recovery_claim(plan)
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-order-execution" / plan["intent_id"]
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
    for name in ("science-plan", "diagnostic-plan", "preflight-execution-plan", "plan", "dry-run-file"):
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
                args.science_plan and args.diagnostic_plan,
                "prepare requires science-plan and diagnostic-plan",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "plan required")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-order-execution" and path.name == "plan.json",
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
            script = str(Path(inner["release"]) / "source/tools/sdsc_student_order_execution.py")
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
                    plan["execution"]["control_file_sha256"][NEW_TOOLS[1]],
                ],
                data=canonical(dict(action=action, plan=plan, authorize=args.authorize, dry_run=proof)),
                timeout=300,
            )
            require(
                response.returncode == 0,
                "remote operation failed/acknowledgement unknown; reconcile: "
                + response.stderr.decode("utf8", "replace")[-1500:],
            )
            require(len(response.stdout) <= science.MAX_RESPONSE, "remote response bound exceeded")
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
