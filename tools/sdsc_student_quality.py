#!/usr/bin/env python3
"""One finite, inference-only diagnostic on the preserved student checkpoints.

Use tools/sdsc check, sync --dry-run and sync first. Then prepare; inspect
submit --dry-run; submit --authorize once. Missing acknowledgement requires
reconcile. This separate diagnostic never advances scientific acceptance.
"""

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
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
TASK = "qwen3-v2-student-quality-diagnostic-v1"
PARENT_JOB = "54548846"
PARENT_INTENT = "594d09aee22c4bdabbe29b0501705344"
PARENT_REPORT_SHA = "91794e528ec83ff0c7f35bd3734d36022d60db2794c3e41c7686beb97fbff4f1"
RESOURCES = dict(
    account="nwu181",
    partition="nairr-gpu-shared",
    qos="nairr-gpu-shared-normal",
    gpus=1,
    gpu_type="h100",
    cpus=24,
    mem_gib=192,
    time="02:00:00",
)
CAP = 1024**2
MAX_FILE = 8 * CAP
MAX_FETCH = 16 * CAP
NAMES = (
    "quality-probe.json",
    "quality-records.jsonl",
    "quality-prompts.jsonl",
    "node-result.json",
    "memory.json",
)
TOOLS = (
    "tools/sdsc_student_quality.py",
    "tools/sdsc_student_quality_job.py",
    "tools/sdsc_student_quality_probe.py",
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    "tools/sdsc_student_contract.py",
    "tools/sdsc_student_job.py",
    "tools/sdsc_student_memory.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def safe(path):
    path = Path(path)
    require(
        path.is_absolute()
        and ".." not in path.parts
        and not any(ord(c) < 32 for c in str(path))
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "unsafe diagnostic path",
    )
    return path


def relative(name):
    path = PurePosixPath(name)
    require(
        isinstance(name, str)
        and name
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == name
        and "\\" not in name
        and not any(ord(c) < 32 for c in name),
        "unsafe diagnostic relative path",
    )
    return name


def read(path, limit=MAX_FILE):
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "unbounded/nonregular evidence")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == before.st_size <= limit and all(getattr(before, k) == getattr(after, k) for k in fields),
        "evidence changed during read",
    )
    return raw


def document(path, expected=None):
    raw = read(path)
    require(expected is None or sha(raw) == expected, "evidence SHA differs")
    return json.loads(raw)


def write_once(path, raw):
    path = safe(path)
    require(len(raw) <= 2 * MAX_FETCH, "control output too large")
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def helper(name, root=ROOT):
    spec = importlib.util.spec_from_file_location("_quality_" + name, root / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_records(manifest):
    rows = manifest["files"]
    require(
        isinstance(rows, list) and 0 < len(rows) < 10000 and manifest["code_sha256"] == sha(canonical(rows)),
        "snapshot inventory hash differs",
    )
    result = {}
    total = 0
    for row in rows:
        name = relative(row["path"])
        require(
            name not in result
            and set(row) == {"path", "size", "sha256", "mode"}
            and type(row["size"]) is int
            and 0 <= row["size"] <= 4 * CAP
            and row["mode"] in (0o644, 0o755)
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "invalid source inventory row",
        )
        result[name] = row
        total += row["size"]
    require(
        total == manifest["total_bytes"] <= 48 * CAP and set(TOOLS) <= set(result),
        "source snapshot lacks diagnostic controls or exceeds bound",
    )
    return result


def checkpoint_rows(publication, report):
    rows = {row["path"]: row for row in publication["files"]}
    require(len(rows) == len(publication["files"]), "duplicate parent publication entries")
    names = [
        ("initial", "artifacts/initial_checkpoint.pt"),
        ("step20", "artifacts/canonical_sft/checkpoints/step-00000020.pt"),
        ("step33", "artifacts/canonical_sft/checkpoints/step-00000033.pt"),
    ]
    result = [dict(rows[path], label=label) for label, path in names]
    require(
        result[0]["sha256"] == report["initial_checkpoint"]["sha256"]
        and result[-1]["sha256"] == report["training_artifacts"]["checkpoint"]["sha256"],
        "parent checkpoint identities differ",
    )
    for row in result:
        require(
            type(row["size"]) is int
            and 0 < row["size"] <= 32 * 1024**3
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]),
            "invalid checkpoint inventory",
        )
    return result, rows["artifacts/canonical_sft/resolved_config.yaml"]


def validate_plan(plan):
    require(
        plan["schema"] == "quest-sdsc-student-quality-plan-v1"
        and plan["task"] == TASK
        and canonical(plan["resources"]) == canonical(RESOURCES)
        and plan["scope"] == "inference_diagnostic_only",
        "unreviewed diagnostic task/resources/scope",
    )
    require(
        plan["parent"]["receipt"]["job_id"] == PARENT_JOB
        and plan["parent"]["receipt"]["intent_id"] == PARENT_INTENT
        and plan["parent"]["receipt"]["submission_dir"] == str(CONTROL / "submissions" / PARENT_INTENT)
        and plan["parent"]["report_sha256"] == PARENT_REPORT_SHA,
        "unreviewed parent job",
    )
    require(
        re.fullmatch("[a-f0-9]{32}", plan["intent_id"])
        and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", plan["run_id"])
        and re.fullmatch("[a-f0-9]{64}", plan["code_sha256"]),
        "invalid plan identities",
    )
    require(
        plan["intent_id"]
        == sha(
            canonical(
                [TASK, PARENT_REPORT_SHA, plan["control_sha256"]["tools/sdsc_student_quality_probe.py"]]
            )
        )[:32],
        "intent changed",
    )
    require(
        plan["release"] == str(CONTROL / "releases" / plan["run_id"])
        and plan["submission_dir"] == str(CONTROL / "student-quality-submissions" / plan["intent_id"])
        and plan["claim"] == str(CONTROL / "student-quality-claims" / (plan["intent_id"] + ".json"))
        and plan["result_dir"] == str(PROJECT / "student-quality" / plan["intent_id"])
        and plan["job_name"] == "opd-sq-" + plan["intent_id"],
        "diagnostic paths differ",
    )
    require(
        set(plan["control_sha256"]) == set(TOOLS)
        and all(re.fullmatch("[a-f0-9]{64}", x) for x in plan["control_sha256"].values()),
        "control pins differ",
    )
    require(
        plan["python"] == plan["parent"]["receipt"]["python"]
        and plan["hf_home"] == plan["parent"]["receipt"]["hf_home"]
        and plan["python"] == str(PROJECT / "envs/qwen3-v2-g0-py31213-cu128-v1/bin/python3.12")
        and plan["hf_home"] == str(PROJECT / "cache/huggingface"),
        "discovered runtime/cache changed",
    )
    for name in ("student_accepted", "g0_passed", "pilot_passed", "factorial_ready"):
        require(plan[name] is False, "diagnostic cannot claim scientific acceptance")
    return plan


def verify_source(plan, root=None):
    release = safe(plan["release"])
    manifest = document(release / "manifest.json", plan["manifest_sha256"])
    require(
        manifest["run_id"] == plan["run_id"] and manifest["code_sha256"] == plan["code_sha256"],
        "remote release identity differs",
    )
    records = manifest_records(manifest)
    root = safe(root or release / "source")
    for name, row in records.items():
        raw = read(root / name, 4 * CAP)
        require(len(raw) == row["size"] and sha(raw) == row["sha256"], "source bytes differ: " + name)
    require(
        all(records[name]["sha256"] == value for name, value in plan["control_sha256"].items()),
        "deployed controls differ",
    )
    return manifest


def verify_parent(plan, *, accounting=False):
    parent = plan["parent"]
    receipt = parent["receipt"]
    submitted = document(safe(receipt["submission_dir"]) / "receipt.json")
    require(submitted == {k: v for k, v in receipt.items() if k != "ok"}, "parent receipt changed")
    root = safe(receipt["result_dir"])
    require(PROJECT in root.parents, "parent results outside persistent project")
    report = document(root / "adapted-calibration.json", parent["report_sha256"])
    publication = document(root / "receipt.json", parent["publication_sha256"])
    require(
        report["passed"] is True
        and report["exit_code"] == 0
        and publication["passed"] is True
        and publication["persisted"] is True
        and publication["persistent_read_back_verified"] is True,
        "parent calibration is not verified/persistent",
    )
    checkpoints, config = checkpoint_rows(publication, report)
    require(checkpoints == plan["checkpoints"] and config == plan["resolved_config"], "planned input differs")
    proof = document(receipt["prerequisites_path"], receipt["prerequisites_sha256"])
    require(proof["dataset_inputs"] == plan["dataset_inputs"], "parent dataset inventory differs")
    if accounting:
        status = helper("sdsc_remote").status(
            dict(root=str(CONTROL), job_id=PARENT_JOB, intent_id=PARENT_INTENT)
        )
        require(
            status.get("success") is True and status["result"]["result_sha256"] == PARENT_REPORT_SHA,
            "parent no longer has successful accounting and verified results",
        )
    return report, publication


def sbatch(plan):
    directory = safe(plan["submission_dir"])
    resources = plan["resources"]
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=24",
        "--account=" + resources["account"],
        "--partition=" + resources["partition"],
        "--qos=" + resources["qos"],
        "--gpus=h100:1",
        "--mem=196608M",
        "--time=02:00:00",
        "--signal=B:TERM@180",
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
    worker = str(Path(plan["release"]) / "source/tools/sdsc_student_quality_job.py")
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
                plan["control_sha256"]["tools/sdsc_student_quality_job.py"],
                str(Path(plan["submission_dir"]) / "plan.json"),
            ]
        )
        + ' "$1"\n'
    ).encode()


def run(argv, timeout=45):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    result = subprocess.run(argv, capture_output=True, timeout=timeout, env=environment)
    require(len(result.stdout) <= CAP and len(result.stderr) <= CAP, "command output exceeds bound")
    return dict(returncode=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode())


def make_receipt(plan, job, reconciled=False):
    require(re.fullmatch("[1-9][0-9]*", job), "invalid Slurm job ID")
    return dict(
        task=TASK,
        job_id=job,
        intent_id=plan["intent_id"],
        run_id=plan["run_id"],
        plan_sha256=sha(canonical(plan)),
        code_sha256=plan["code_sha256"],
        result_dir=plan["result_dir"],
        resources=RESOURCES,
        reconciled=reconciled,
        received_at=now(),
    )


def live_binding(plan, job):
    result = run(["scontrol", "show", "job", "--oneliner", job])
    require(result["returncode"] == 0, "live Slurm binding unavailable; reconcile without resubmission")
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", result["stdout"]))
    expected = dict(
        JobId=job,
        JobName=plan["job_name"],
        Account="nwu181",
        Partition="nairr-gpu-shared",
        QOS="nairr-gpu-shared-normal",
        Comment=plan["intent_id"],
        Command=str(Path(plan["submission_dir"]) / "job.sh"),
        WorkDir=plan["submission_dir"],
    )
    require(all(fields.get(k) == v for k, v in expected.items()), "live Slurm identity differs")
    return dict(job_id=job, plan_sha256=sha(canonical(plan)), fields=fields, observed_at=now())


def valid_live_binding(plan, job, observed):
    if not isinstance(observed, dict):
        return False
    fields = observed.get("fields", {})
    expected = dict(
        JobId=job,
        JobName=plan["job_name"],
        Account="nwu181",
        Partition="nairr-gpu-shared",
        QOS="nairr-gpu-shared-normal",
        Comment=plan["intent_id"],
        Command=str(Path(plan["submission_dir"]) / "job.sh"),
        WorkDir=plan["submission_dir"],
    )
    return (
        observed.get("job_id") == job
        and observed.get("plan_sha256") == sha(canonical(plan))
        and all(fields.get(k) == v for k, v in expected.items())
    )


def validate_accounting(plan, job, queue, account, observed=None):
    require(queue["returncode"] == account["returncode"] == 0, "Slurm query unavailable")
    rows = [line.split("|") for line in account["stdout"].splitlines() if line.strip()]
    own = [row for row in rows if row[0] == job]
    require(
        len(own) <= 1 and all(row[0] == job or row[0].startswith(job + ".") for row in rows),
        "ambiguous Slurm accounting",
    )
    success = False
    state = "UNKNOWN"
    if own:
        row = own[0]
        require(len(row) >= 12, "incomplete Slurm accounting")
        require(
            row[3:7] == [plan["job_name"], "nwu181", "nairr-gpu-shared", "nairr-gpu-shared-normal"]
            and (row[11] == plan["intent_id"] or (row[11] == "" and valid_live_binding(plan, job, observed))),
            "Slurm job identity differs",
        )
        state = row[1]
        if state == "COMPLETED" and not queue["stdout"].strip():
            require(
                {job + ".batch", job + ".extern"} <= {x[0] for x in rows}
                and len({x[0] for x in rows}) == len(rows)
                and all(x[1:3] == ["COMPLETED", "0:0"] for x in rows),
                "Slurm step failed/incomplete",
            )
            tres = dict(x.split("=", 1) for x in row[10].split(",") if "=" in x)
            require(
                row[7] == "24"
                and row[8] in {"192Gn", "196608Mn", "196608M", "192G"}
                and tres.get("cpu") == "24"
                and tres.get("gres/gpu") == "1"
                and tres.get("node") == "1",
                "actual diagnostic allocation differs",
            )
            success = True
    elif queue["stdout"].strip():
        entries = [line.split("|") for line in queue["stdout"].splitlines() if line.strip()]
        require(len(entries) == 1 and entries[0][0] == job and len(entries[0]) >= 2, "queue identity differs")
        state = entries[0][1]
    return dict(job_id=job, state=state, accounting_complete=success, queue=queue, accounting=account)


def validate_worker_report(report, plan, job, records):
    require(
        report.get("diagnostic_complete") is True
        and report.get("passed") is True
        and all(
            report.get(k) is False
            for k in ("student_accepted", "g0_passed", "pilot_passed", "factorial_ready")
        ),
        "diagnostic failed or overclaims scientific acceptance",
    )
    require(
        report.get("job_id") == job
        and report.get("parent_job_id") == PARENT_JOB
        and report.get("run_id") == plan["run_id"]
        and report.get("source_code_sha256") == plan["code_sha256"],
        "diagnostic worker identity differs",
    )
    raw = report.get("raw_artifacts")
    require(
        isinstance(raw, list)
        and len(raw) == 2
        and {r.get("path") for r in raw} == {"quality-records.jsonl", "quality-prompts.jsonl"}
        and all(records.get(r["path"]) == r for r in raw),
        "worker raw artifact hash binding differs",
    )


def publication_records(published, plan, job):
    """Bind any sealed publication, including failed/partial worker evidence."""
    expected = dict(
        task=TASK,
        job_id=job,
        run_id=plan["run_id"],
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        code_sha256=plan["code_sha256"],
    )
    require(
        all(published.get(key) == value for key, value in expected.items())
        and published.get("persistent_read_back_verified") is True
        and type(published.get("passed")) is bool
        and type(published.get("diagnostic_complete")) is bool
        and published["passed"] is published["diagnostic_complete"]
        and all(
            published.get(key) is False
            for key in ("student_accepted", "g0_passed", "pilot_passed", "factorial_ready")
        ),
        "unbound persistent diagnostic publication",
    )
    rows = published.get("files")
    require(isinstance(rows, list), "publication inventory missing")
    records, total = {}, 0
    for row in rows:
        require(
            isinstance(row, dict)
            and set(row) == {"path", "size", "sha256"}
            and row["path"] in NAMES
            and row["path"] not in records
            and type(row["size"]) is int
            and 0 <= row["size"] <= MAX_FILE
            and re.fullmatch("[a-f0-9]{64}", str(row["sha256"])),
            "invalid/duplicate publication inventory",
        )
        total += row["size"]
        records[row["path"]] = row
    require(total <= MAX_FETCH, "publication exceeds bounded size")
    return records


def inspect_job(plan, receipt):
    job = receipt["job_id"]
    queue = run(["squeue", "--noheader", "--jobs=" + job, "--format=%i|%T"])
    account = run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs=" + job,
            "--format=JobIDRaw,State,ExitCode,JobName%100,Account,Partition,QOS,AllocCPUS,ReqMem,"
            "ElapsedRaw,AllocTRES%200,Comment%100",
        ]
    )
    observed_path = Path(plan["submission_dir"]) / "live-binding.json"
    observed = document(observed_path) if observed_path.exists() else None
    result = validate_accounting(plan, job, queue, account, observed)
    result.update(
        success=False,
        diagnostic_complete=False,
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        publication_verified=False,
        artifact_hashes_verified=False,
        evidence_state="unpublished_logs_only",
        checked_at=now(),
    )
    result_root = safe(plan["result_dir"])
    publication_path = result_root / "receipt.json"
    records = {}
    if publication_path.exists():
        publication_raw = read(publication_path)
        published = json.loads(publication_raw)
        records = publication_records(published, plan, job)
        require(
            {name for name in NAMES if (result_root / name).exists()} == set(records),
            "persistent scientific files differ from publication inventory",
        )
        for name, row in records.items():
            raw = read(result_root / name)
            require(
                len(raw) == row["size"] and sha(raw) == row["sha256"], "persistent diagnostic hash differs"
            )
        result.update(
            publication_verified=True,
            artifact_hashes_verified=True,
            evidence_state="verified_publication",
            publication=published,
            publication_sha256=sha(publication_raw),
        )
    if result["accounting_complete"]:
        require(result["publication_verified"], "completed diagnostic has no verified publication")
        require(set(NAMES) <= set(records), "incomplete successful diagnostic publication")
        require(
            result["publication"]["passed"] is True and result["publication"]["diagnostic_complete"] is True,
            "completed allocation has a failed diagnostic publication",
        )
        report = document(result_root / "quality-probe.json")
        validate_worker_report(report, plan, job, records)
        node = document(result_root / "node-result.json")
        require(
            node.get("diagnostic_complete") is True
            and node.get("exit_code") == 0
            and node.get("job_id") == job
            and node.get("plan_sha256") == sha(canonical(plan))
            and all(
                node.get(k) is False
                for k in ("student_accepted", "g0_passed", "pilot_passed", "factorial_ready")
            ),
            "node completion failed or unbound",
        )
        result.update(
            success=True,
            diagnostic_complete=True,
            result_sha256=sha(read(result_root / "quality-probe.json")),
        )
    return result


def remote_action(request):
    plan = validate_plan(request["plan"])
    verify_source(plan)
    directory = safe(plan["submission_dir"])
    claim = safe(plan["claim"])
    action = request["action"]
    if action == "dry-run":
        verify_parent(plan, accounting=True)
        require(not directory.exists() and not claim.exists(), "intent already exists; reconcile only")
        require(
            safe(plan["python"]).is_file()
            and os.access(plan["python"], os.X_OK)
            and sha(read(plan["python"], 64 * CAP))
            == "2777d5f6632ec0d7268ad754c28c96372e1e2097e15c22a6688db157cf750c19",
            "discovered Python executable differs",
        )
        require(
            safe(plan["hf_home"]).is_dir() and PROJECT.is_dir() and os.access(PROJECT, os.W_OK),
            "login-visible input/output paths unavailable; compute-node mounts still require verification",
        )
        return dict(
            dry_run=True,
            plan_sha256=sha(canonical(plan)),
            argv=sbatch(plan),
            resources=RESOURCES,
            scope=plan["scope"],
            blockers=[],
            source_verified=True,
            parent_verified=True,
            runtime_executable_verified=True,
            node_mounts_still_required=True,
        )
    if action == "submit":
        require(request.get("authorize") is True, "explicit authorization required")
        verify_parent(plan, accounting=True)
        require(not directory.exists() and not claim.exists(), "intent exists; reconcile, never resubmit")
        directory.parent.mkdir(parents=True, exist_ok=True)
        claim.parent.mkdir(parents=True, exist_ok=True)
        write_once(
            claim, canonical(dict(intent_id=plan["intent_id"], plan_sha256=sha(canonical(plan)), at=now()))
        )
        directory.mkdir(mode=0o700)
        write_once(directory / "plan.json", canonical(plan))
        write_once(directory / "job.sh", job_script(plan))
        write_once(directory / "submission-started.json", canonical(dict(argv=sbatch(plan), at=now())))
        try:
            result = run(sbatch(plan))
            match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\s*", result["stdout"])
            require(result["returncode"] == 0 and match is not None, "no trustworthy sbatch acknowledgement")
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
        "persistent plan/claim differs",
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
            row = line.split("|")
            if (
                len(row) >= 3
                and row[1:3] == [plan["job_name"], plan["intent_id"]]
                and re.fullmatch("[1-9][0-9]*", row[0])
                and (len(row) == 3 or row[3] == user)
            ):
                ids.add(row[0])
        require(len(ids) == 1, "zero/ambiguous matches: dispatch unknown; never retry")
        write_once(directory / "receipt.json", canonical(make_receipt(plan, ids.pop(), True)))
    receipt = document(directory / "receipt.json")
    require(
        receipt["plan_sha256"] == sha(canonical(plan))
        and receipt["intent_id"] == plan["intent_id"]
        and receipt["resources"] == RESOURCES
        and re.fullmatch("[1-9][0-9]*", receipt["job_id"]),
        "receipt differs",
    )
    if action == "reconcile":
        if not (directory / "live-binding.json").exists():
            write_once(directory / "live-binding.json", canonical(live_binding(plan, receipt["job_id"])))
        return receipt
    status = inspect_job(plan, receipt)
    if action == "status":
        return status
    require(action == "fetch", "unknown operation")
    files = {}
    total = 0
    # Full diagnostic data are never tailed. Only transport/worker logs are tails.
    if status.get("publication_verified"):
        records = publication_records(status["publication"], plan, receipt["job_id"])
        for name in (*(name for name in NAMES if name in records), "receipt.json"):
            path = safe(plan["result_dir"]) / name
            raw = read(path)
            total += len(raw)
            if name == "receipt.json":
                require(sha(raw) == status["publication_sha256"], "publication changed during fetch")
            else:
                row = records[name]
                require(len(raw) == row["size"] and sha(raw) == row["sha256"], "data changed during fetch")
            files[name] = base64.b64encode(raw).decode()
    for name, path in [
        ("worker.log", safe(plan["result_dir"]) / "worker.log"),
        ("slurm.out", directory / ("slurm-" + receipt["job_id"] + ".out")),
        ("slurm.err", directory / ("slurm-" + receipt["job_id"] + ".err")),
    ]:
        if path.exists():
            with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
                info = os.fstat(stream.fileno())
                require(stat.S_ISREG(info.st_mode), "nonregular log")
                stream.seek(max(0, info.st_size - 64 * 1024))
                raw = stream.read(64 * 1024)
            total += len(raw)
            files[name] = base64.b64encode(raw).decode()
    require(total <= MAX_FETCH, "diagnostic fetch exceeds bound; large results remain remote")
    return dict(status=status, receipt=receipt, files=files, bytes=total)


def validate_download(result, plan):
    files = {}
    total = 0
    require(
        result["receipt"]["intent_id"] == plan["intent_id"]
        and result["receipt"]["plan_sha256"] == sha(canonical(plan)),
        "fetch receipt differs",
    )
    for name, encoded in result["files"].items():
        require(
            name in (*NAMES, "receipt.json", "worker.log", "slurm.out", "slurm.err"),
            "unexpected fetched file",
        )
        raw = base64.b64decode(encoded, validate=True)
        total += len(raw)
        require(len(raw) <= MAX_FILE and total <= MAX_FETCH, "unbounded diagnostic fetch")
        if name in ("worker.log", "slurm.out", "slurm.err"):
            require(len(raw) <= 64 * 1024, "downloaded log exceeds tail bound")
        files[name] = raw
    require(total == result["bytes"], "fetched bytes differ")
    status = result["status"]
    scientific_names = set(files) & set(NAMES)
    records = {}
    if "receipt.json" in files:
        publication = json.loads(files["receipt.json"])
        records = publication_records(publication, plan, result["receipt"]["job_id"])
        require(
            status.get("publication_verified") is True
            and status.get("artifact_hashes_verified") is True
            and publication == status.get("publication")
            and sha(files["receipt.json"]) == status.get("publication_sha256"),
            "fetched publication differs from verified status",
        )
        require(
            scientific_names == set(records),
            "fetched scientific inventory differs or missing complete evidence",
        )
        for name, row in records.items():
            require(
                len(files[name]) == row["size"] and sha(files[name]) == row["sha256"],
                "downloaded scientific bytes differ: " + name,
            )
    else:
        require(
            not scientific_names
            and status.get("publication_verified") is False
            and status.get("artifact_hashes_verified") is False
            and status.get("evidence_state") == "unpublished_logs_only"
            and status.get("success") is False,
            "unverified fetch must contain only bounded logs without scientific files",
        )
    if status.get("success"):
        require(set(NAMES) | {"receipt.json"} <= set(files), "successful fetch missing complete evidence")
        require(publication["passed"] is True, "successful status has failed publication")
        validate_worker_report(
            json.loads(files["quality-probe.json"]), plan, result["receipt"]["job_id"], records
        )
    return files


def ssh_operation(cli, plan, action, authorize=False):
    remote_path = str(Path(plan["release"]) / "source/tools/sdsc_student_quality.py")
    launch = (
        "import hashlib,pathlib,runpy,sys;p=pathlib.Path(sys.argv[1]);"
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2];"
        'sys.argv=[str(p),"remote"];runpy.run_path(str(p),run_name="__main__")'
    )
    response = cli.ssh_call(
        [plan["python"], "-I", "-B", "-c", launch, remote_path, plan["control_sha256"][TOOLS[0]]],
        data=canonical(dict(action=action, plan=plan, authorize=authorize)),
        timeout=240,
    )
    require(
        response.returncode == 0,
        "remote action failed or acknowledgement unknown; use reconcile: "
        + response.stderr.decode("utf-8", "replace")[-1500:],
    )
    require(len(response.stdout) <= 2 * MAX_FETCH, "remote response exceeded bound")
    return json.loads(response.stdout)


def prepare(args, cli):
    record = cli.run_record(args.run_id)
    require(record["state"] == "deployed", "first run tools/sdsc sync --dry-run then sync")
    manifest = record["manifest"]
    records = manifest_records(manifest)
    require(record["deployment"]["code_sha256"] == manifest["code_sha256"], "deployment receipt differs")
    require(
        all(sha(read(ROOT / name)) == records[name]["sha256"] for name in TOOLS),
        "controls changed since synchronized snapshot",
    )
    fetch = safe(args.fetch_dir.absolute())
    report = document(fetch / "adapted-calibration.json", PARENT_REPORT_SHA)
    publication = document(fetch / "receipt.json")
    receipt = document(ROOT / ".sdsc/submissions" / (PARENT_INTENT + ".json"))["receipt"]
    status = document(safe(args.status_file.absolute()))
    helper("sdsc_student_supervise").status_outcome(status, receipt)
    require(
        status.get("success") is True and status["result"]["result_sha256"] == PARENT_REPORT_SHA,
        "local parent status is not semantically verified",
    )
    checkpoints, config = checkpoint_rows(publication, report)
    contract = helper("sdsc_student_contract")
    dataset = [dict(row, source=str(contract.DATASET_ROOT / row["path"])) for row in contract.DATASET_FILES]
    intent = sha(canonical([TASK, PARENT_REPORT_SHA, records[TOOLS[2]]["sha256"]]))[:32]
    plan = dict(
        schema="quest-sdsc-student-quality-plan-v1",
        task=TASK,
        scope="inference_diagnostic_only",
        intent_id=intent,
        run_id=manifest["run_id"],
        code_sha256=manifest["code_sha256"],
        manifest_sha256=sha(canonical(manifest) + b"\n"),
        created_at=now(),
        resources=RESOURCES,
        parent=dict(
            receipt=receipt,
            report_sha256=PARENT_REPORT_SHA,
            publication_sha256=sha(read(fetch / "receipt.json")),
        ),
        checkpoints=checkpoints,
        resolved_config=config,
        dataset_inputs=dataset,
        control_sha256={name: records[name]["sha256"] for name in TOOLS},
        release=str(CONTROL / "releases" / manifest["run_id"]),
        submission_dir=str(CONTROL / "student-quality-submissions" / intent),
        claim=str(CONTROL / "student-quality-claims" / (intent + ".json")),
        result_dir=str(PROJECT / "student-quality" / intent),
        job_name="opd-sq-" + intent,
        python=receipt["python"],
        hf_home=receipt["hf_home"],
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
    )
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-quality" / intent
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(directory / "plan.json", canonical(plan))
    return dict(
        plan=str(directory / "plan.json"),
        plan_sha256=sha(canonical(plan)),
        resources=RESOURCES,
        checkpoints=checkpoints,
        scope=plan["scope"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "fetch", "reconcile", "remote"))
    parser.add_argument("--run-id")
    parser.add_argument("--fetch-dir", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "remote":
        require(
            Path(__file__).resolve().is_relative_to(CONTROL / "releases"),
            "remote actions cannot run on Quest",
        )
        raw = sys.stdin.buffer.read(2 * MAX_FILE + 1)
        require(len(raw) <= 2 * MAX_FILE, "remote request exceeds bound")
        result = remote_action(json.loads(raw))
    else:
        cli = helper("sdsc_cli")
        if args.action == "prepare":
            require(
                args.run_id and args.fetch_dir and args.status_file,
                "prepare requires run-id/fetch-dir/status-file",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "operation requires --plan")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-quality" and path.name == "plan.json",
                "plan must be in project diagnostic metadata",
            )
            plan = validate_plan(document(path))
            require(path.parent.name == plan["intent_id"], "local intent directory differs")
            require(
                all(sha(read(ROOT / name)) == value for name, value in plan["control_sha256"].items()),
                "local pinned controls changed",
            )
            if args.action == "submit" and args.dry_run:
                result = ssh_operation(cli, plan, "dry-run")
                stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
                write_once(path.parent / ("dry-run-" + stamp + ".json"), canonical(result))
            else:
                require(args.action != "submit" or args.authorize, "submit requires --authorize")
                if args.action == "submit":
                    cli.require_master()
                    write_once(
                        path.parent / "submission-started.json", canonical(dict(at=now(), no_retry=True))
                    )
                result = ssh_operation(cli, plan, args.action, args.authorize)
                if args.action == "fetch":
                    files = validate_download(result, plan)
                    job_id = result["receipt"]["job_id"]
                    require(re.fullmatch(r"[1-9][0-9]*", job_id), "invalid fetched job ID")
                    fetched_parent = safe(ROOT / ".sdsc/fetched" / job_id)
                    fetched_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    fetched = Path(tempfile.mkdtemp(prefix="fetch-", dir=fetched_parent))
                    for name, raw in files.items():
                        write_once(fetched / name, raw)
                        require(read(fetched / name) == raw, "local fetched file failed read-back")
                    result.pop("files")
                    result["destination"] = str(fetched)
                    write_once(fetched / "fetch-manifest.json", canonical(result))
                stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
                write_once(path.parent / (args.action + "-" + stamp + ".json"), canonical(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
