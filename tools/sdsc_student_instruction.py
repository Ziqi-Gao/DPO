#!/usr/bin/env python3
"""One frozen train-only scientific prompt screen on the original student.

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
import math
import os
import pwd
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
TASK = "qwen3-v2-student-instruction-screen-v1"
PARENT_JOB = "54548846"
PARENT_INTENT = "594d09aee22c4bdabbe29b0501705344"
PARENT_REPORT_SHA = "91794e528ec83ff0c7f35bd3734d36022d60db2794c3e41c7686beb97fbff4f1"
RESOURCES = dict(
    account="nwu181",
    partition="nairr-gpu-shared",
    qos="nairr-gpu-shared-normal",
    gpus=1,
    gpu_type="h100",
    cpus=8,
    mem_gib=64,
    time="00:30:00",
)
CAP = 1024**2
MAX_FILE = 8 * CAP
MAX_FETCH = 16 * CAP
NAMES = (
    "instruction-probe.json",
    "instruction-records.jsonl",
    "instruction-prompts.jsonl",
    "node-result.json",
    "memory.json",
)
TOOLS = (
    "tools/sdsc_student_instruction.py",
    "tools/sdsc_student_instruction_job.py",
    "tools/sdsc_student_instruction_probe.py",
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    "tools/sdsc_student_contract.py",
    "tools/sdsc_student_job.py",
    "tools/sdsc_student_memory.py",
    "tools/sdsc_student_quality.py",
    "tools/sdsc_student_quality_probe.py",
    "tools/sdsc_provenance.py",
    "docs/refactor/sdsc_student_instruction_screen_20260930.md",
)


# Immutable, explicitly pinned pure I/O helpers. No old task/validator globals are changed.
SHARED_SHA = "5fcf65d7d999d886067ee5acb898364c696850b5bd778ac6e95976ae91e51e50"
_shared_path = Path(__file__).with_name("sdsc_student_quality.py")
if hashlib.sha256(_shared_path.read_bytes()).hexdigest() != SHARED_SHA:
    raise ValueError("frozen shared quality helper changed")
_shared_spec = importlib.util.spec_from_file_location("_instruction_shared_io", _shared_path)
_shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared)
require, canonical, sha, now = _shared.require, _shared.canonical, _shared.sha, _shared.now
safe, relative, read, document, write_once = (
    _shared.safe,
    _shared.relative,
    _shared.read,
    _shared.document,
    _shared.write_once,
)
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "teacher_accepted",
    "teacher_accepted_under_candidate",
    "accepted_science",
    "formal_prompt_accepted",
)
CANDIDATE_SHA = "8c44dc8a1bc86167e3787cdcd84e20537db69585e3d7ee30b920447072f03c1e"
BASELINE_SHA = "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
INITIAL_SHA = "85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4"
CONFIG_SHA = "05872b4521802813640004bf614dad65a6e1d2c3c3662a7502ca11bec5e15dbe"
SCREEN_DOC = "docs/refactor/sdsc_student_instruction_screen_20260930.md"
DATASET_NAMES = {"manifest.json", "train/examples.jsonl"}

LEGACY_INTENT = "6b36e12f9a946f983581e4302282744b"


def science_identity():
    return dict(
        task=TASK,
        parent_report_sha256=PARENT_REPORT_SHA,
        initial_sha256=INITIAL_SHA,
        candidate_sha256=CANDIDATE_SHA,
        baseline_sha256=BASELINE_SHA,
        population="original-family-first32-train-in-manifest-order",
        dataset_manifest_sha256="bee7baf767f04ee153ec7ad5f4da535d7fb3c31ba274cf3a0e5d66a01ac6c189",
        train_file_sha256="377538a779f31246eb9aee0ee3283755641149f8dd713c942693f3da2ab1bf4b",
        generation=dict(
            max_new_tokens=256,
            max_model_input_length=1536,
            do_sample=False,
            use_cache=False,
            truncation=False,
        ),
        precision="original-pinned-BF16-without-explicit-autocast",
    )


def execution_intent(pins):
    return sha(canonical(dict(science=science_identity(), controls=pins, resources=RESOURCES)))[:32]


def scientific_claim_path():
    return CONTROL / "student-instruction-scientific-claims" / (sha(canonical(science_identity())) + ".json")


def legacy_claim_path():
    return CONTROL / "student-instruction-claims" / (LEGACY_INTENT + ".json")


def scientific_claim_record(plan):
    return dict(
        science_identity=science_identity(),
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        submission_dir=plan["submission_dir"],
        no_retry=True,
    )


def legacy_reservation(plan):
    return dict(
        legacy_intent=LEGACY_INTENT,
        reservation="scientific-claim-cross-version-guard",
        **scientific_claim_record(plan),
    )


def require_unclaimed_science(plan):
    require(
        not safe(plan["scientific_claim"]).exists(),
        "scientific candidate already claimed; reconcile its winning intent, never resubmit",
    )
    require(
        not safe(legacy_claim_path()).exists()
        and not safe(CONTROL / "student-instruction-submissions" / LEGACY_INTENT).exists(),
        "legacy candidate submission/claim exists; reconcile, never resubmit",
    )


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
    initial = dict(rows["artifacts/initial_checkpoint.pt"], label="initial")
    config = rows["artifacts/canonical_sft/resolved_config.yaml"]
    require(
        initial["sha256"] == INITIAL_SHA == report["initial_checkpoint"]["sha256"]
        and initial["size"] == 3441276375
        and config["sha256"] == CONFIG_SHA
        and config["size"] == 7923,
        "fixed parent initial/config differs",
    )
    return [initial], config


def validate_plan(plan):
    require(
        plan["schema"] == "quest-sdsc-student-instruction-plan-v1"
        and plan["task"] == TASK
        and canonical(plan["resources"]) == canonical(RESOURCES)
        and plan["scope"] == "train-only-scientific-prompt-screen",
        "unreviewed diagnostic task/resources/scope",
    )
    require(
        plan.get("candidate_sha256") == CANDIDATE_SHA
        and plan.get("baseline_sha256") == BASELINE_SHA
        and plan.get("screen_document_sha256") == plan["control_sha256"][SCREEN_DOC]
        and plan["control_sha256"]["tools/sdsc_student_quality.py"] == SHARED_SHA,
        "frozen instruction/document/helper identity differs",
    )
    require(
        len(plan["checkpoints"]) == 1
        and plan["checkpoints"][0]
        == dict(label="initial", path="artifacts/initial_checkpoint.pt", size=3441276375, sha256=INITIAL_SHA)
        and plan["resolved_config"]
        == dict(path="artifacts/canonical_sft/resolved_config.yaml", size=7923, sha256=CONFIG_SHA),
        "only the fixed original initial/config may be staged",
    )
    expected_dataset = [
        dict(row, source=str(helper("sdsc_student_contract").DATASET_ROOT / row["path"]))
        for row in helper("sdsc_student_contract").DATASET_FILES
        if row["path"] in DATASET_NAMES
    ]
    require(plan["dataset_inputs"] == expected_dataset, "only original manifest and train split are allowed")
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
    require(plan["intent_id"] == execution_intent(plan["control_sha256"]), "intent changed")
    require(
        plan.get("scientific_claim") == str(scientific_claim_path()),
        "permanent scientific claim path differs",
    )
    require(
        plan["release"] == str(CONTROL / "releases" / plan["run_id"])
        and plan["submission_dir"] == str(CONTROL / "student-instruction-submissions" / plan["intent_id"])
        and plan["claim"] == str(CONTROL / "student-instruction-claims" / (plan["intent_id"] + ".json"))
        and plan["result_dir"] == str(PROJECT / "student-instruction" / plan["intent_id"])
        and plan["job_name"] == "opd-si-" + plan["intent_id"],
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
    for name in FLAGS:
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
    require(
        [r for r in proof["dataset_inputs"] if r["path"] in DATASET_NAMES] == plan["dataset_inputs"],
        "parent dataset inventory differs",
    )
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
        "--cpus-per-task=8",
        "--account=" + resources["account"],
        "--partition=" + resources["partition"],
        "--qos=" + resources["qos"],
        "--gpus=h100:1",
        "--mem=65536M",
        "--time=00:30:00",
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


def gpu_tres(value, *, complete):
    """Parse total/typed GPU TRES once; a complete CPU inventory can establish zero."""
    require(
        isinstance(value, str) and value not in ("", "N/A", "(null)", "None"),
        "missing authoritative GPU TRES inventory",
    )
    pairs = [item.split("=", 1) for item in value.split(",")]
    require(all(len(item) == 2 for item in pairs), "malformed TRES inventory")
    entries = dict(pairs)
    require(len(entries) == len(pairs), "duplicate TRES entries")
    if complete:
        require(
            all(re.fullmatch(r"[1-9][0-9]*", entries.get(key, "")) for key in ("cpu", "node")),
            "TRES lacks complete CPU/node request identity",
        )
    gpu = {}
    for key, count in entries.items():
        if "gpu" not in key.lower():
            continue
        match = re.fullmatch(r"gres/gpu(?::([A-Za-z0-9_-]+))?", key)
        require(match is not None and re.fullmatch(r"[0-9]+", count), "unrecognized GPU TRES entry")
        gpu[match.group(1) or "total"] = int(count)
    typed = sum(count for key, count in gpu.items() if key != "total")
    if "total" in gpu:
        require(len(gpu) == 1 or typed == gpu["total"], "generic/typed GPU totals conflict")
        return gpu["total"]
    return typed


def gpu_gres(value):
    """Parse Slurm TresPerJob/Node/Task or queue GRES; absence is unknown."""
    if value in (None, "", "N/A", "(null)", "None"):
        return None
    gpu = {}
    for token in value.split(","):
        if "gpu" not in token.lower():
            continue
        match = re.fullmatch(r"(?:gres[:/])?gpu(?::([A-Za-z0-9_-]+))?:([0-9]+)", token)
        require(match is not None, "unrecognized GPU GRES entry")
        kind = match.group(1) or "total"
        require(kind not in gpu, "duplicate GPU GRES entry")
        gpu[kind] = int(match.group(2))
    if not gpu:
        return None
    typed = sum(count for key, count in gpu.items() if key != "total")
    if "total" in gpu:
        require(len(gpu) == 1 or typed == gpu["total"], "generic/typed GPU GRES totals conflict")
        return gpu["total"]
    return typed


def live_gpu_request(job, user, queue_gres):
    observed = run(["scontrol", "show", "job", "--oneliner", job])
    require(observed["returncode"] == 0, "live GPU inventory unavailable")
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S+)", observed["stdout"])
    fields = dict(pairs)
    require(len(fields) == len(pairs), "ambiguous/duplicate live job fields")
    # Expanded arrays may be reported by a raw JobId with separate array identity.
    raw_id = fields.get("JobId", "")
    array_id = fields.get("ArrayJobId", "") + "_" + fields.get("ArrayTaskId", "")
    require(
        re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", raw_id)
        and (raw_id == job or ("_" in job and array_id == job))
        and re.fullmatch(re.escape(user) + r"\([0-9]+\)", fields.get("UserId", "")),
        "live GPU inventory has foreign/unknown job identity",
    )
    state = fields.get("JobState")
    require(
        state in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED", "STOPPED"},
        "live queue changed or state unknown; recheck without submission",
    )
    requested = gpu_tres(fields.get("ReqTRES"), complete=True)
    evidence = {"ReqTRES": fields["ReqTRES"]}
    allocated = fields.get("AllocTRES")
    if state in {"RUNNING", "COMPLETING", "SUSPENDED", "STOPPED"}:
        require(allocated not in (None, "", "N/A", "(null)", "None"), "active job lacks AllocTRES")
    if allocated not in (None, "", "N/A", "(null)", "None"):
        require(gpu_tres(allocated, complete=True) == requested, "requested/allocated GPU totals conflict")
        evidence["AllocTRES"] = allocated
    if "TRES" in fields:
        require(gpu_tres(fields["TRES"], complete=True) == requested, "TRES GPU totals conflict")
        evidence["TRES"] = fields["TRES"]
    for names, multiplier in (
        (("TresPerJob", "TRESPerJob"), 1),
        (("TresPerNode", "TRESPerNode"), fields.get("NumNodes")),
        (("TresPerTask", "TRESPerTask"), fields.get("NumTasks")),
    ):
        values = [fields[name] for name in names if name in fields]
        require(len(values) <= 1, "ambiguous per-unit GPU fields")
        if not values:
            continue
        amount = gpu_gres(values[0])
        if amount is not None:
            require(re.fullmatch(r"[1-9][0-9]*", str(multiplier)), "ambiguous GPU unit multiplicity")
            require(amount * int(multiplier) == requested, "per-unit/requested GPU totals conflict")
        evidence[names[0]] = values[0]
    per_node = gpu_gres(queue_gres)
    if per_node is not None:
        require(re.fullmatch(r"[1-9][0-9]*", fields.get("NumNodes", "")), "unknown queue GPU node count")
        require(per_node * int(fields["NumNodes"]) == requested, "queue/live GPU totals conflict")
    return dict(
        job_id=job,
        live_job_id=raw_id,
        state=state,
        user=user,
        queue_gres=queue_gres,
        requested_gpus=requested,
        authoritative_fields=evidence,
    )


def check_gpu_ceiling():
    """Read complete live GPU TRES for every own job; N/A queue GRES proves nothing."""
    user = pwd.getpwuid(os.getuid()).pw_name
    query = run(["squeue", "--noheader", "--array", "--user=" + user, "--format=%i|%T|%b"])
    require(query["returncode"] == 0, "GPU concurrency query unavailable")
    jobs, seen = [], set()
    for line in query["stdout"].splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        require(
            len(parts) == 3 and re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", parts[0]) and parts[0] not in seen,
            "ambiguous queue/array GPU inventory",
        )
        seen.add(parts[0])
        jobs.append(live_gpu_request(parts[0], user, parts[2].strip()))
    total = sum(job["requested_gpus"] for job in jobs)
    require(total + RESOURCES["gpus"] <= 4, "new screen would exceed four allocatable GPUs")
    return dict(
        existing_requested_gpus=total, new_gpus=RESOURCES["gpus"], jobs=jobs, limit=4, observation_only=True
    )


def job_script(plan):
    worker = str(Path(plan["release"]) / "source/tools/sdsc_student_instruction_job.py")
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
                plan["control_sha256"]["tools/sdsc_student_instruction_job.py"],
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
                row[7] == "8"
                and row[8] in {"64Gn", "65536Mn", "65536M", "64G"}
                and tres.get("cpu") == "8"
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
        and all(report.get(k) is False for k in FLAGS),
        "diagnostic failed or overclaims scientific acceptance",
    )
    require(
        report.get("job_id") == job
        and report.get("parent_job_id") == PARENT_JOB
        and report.get("run_id") == plan["run_id"]
        and report.get("source_code_sha256") == plan["code_sha256"],
        "diagnostic worker identity differs",
    )
    require(
        report.get("schema") == "quest-sdsc-student-instruction-probe-v1"
        and report.get("candidate_sha256") == CANDIDATE_SHA
        and report.get("baseline_sha256") == BASELINE_SHA
        and type(report.get("training_screen_viable")) is bool,
        "screen worker scope/candidate differs",
    )
    identity = report.get("initial_identity", {})
    require(
        identity.get("checkpoint_sha256") == INITIAL_SHA
        and identity.get("comparison_performed_before_load") is True
        and identity.get("pristine_comparison", {}).get("all_tensors_exact") is True
        and type(identity.get("pristine_comparison", {}).get("key_count")) is int
        and identity["pristine_comparison"]["key_count"] > 0,
        "original initial was not verified against pristine weights",
    )
    require(
        identity.get("size") == 3441276375
        and identity.get("embedding_tying_before") == identity.get("embedding_tying_after")
        and isinstance(identity.get("embedding_tying_before"), dict)
        and report.get("only_instruction_changed") is True
        and report.get("generation")
        == dict(
            max_new_tokens=256,
            max_model_input_length=1536,
            do_sample=False,
            use_cache=False,
            truncation=False,
        )
        and report.get("raw_record_count") == 64
        and len(report.get("ordered_train_ids", [])) == 32
        and len(set(report["ordered_train_ids"])) == 32
        and report.get("model_parameters_sha256_before") == report.get("model_parameters_sha256_after")
        and re.fullmatch("[a-f0-9]{64}", str(report.get("model_parameters_sha256_before"))),
        "incomplete or changed original-model screen",
    )
    expected_dataset = {row["path"]: row["sha256"] for row in plan["dataset_inputs"]}
    require(
        report.get("dataset_manifest_sha256") == expected_dataset["manifest.json"]
        and report.get("train_file_sha256") == expected_dataset["train/examples.jsonl"],
        "screen dataset identity differs",
    )
    arms = report.get("arms", [])
    require([row.get("arm") for row in arms] == ["baseline", "candidate"], "screen arms differ")
    for index, arm in enumerate(arms):
        metrics, forced = arm["metrics"], arm["teacher_forced"]
        require(
            arm.get("record_start") == index * 32
            and arm.get("record_count") == 32
            and metrics.get("count") == 32
            and forced.get("examples") == 32,
            "screen arm population differs",
        )
        for key in ("answer_accuracy", "exact_proof_accuracy", "format_validity"):
            value = metrics.get(key)
            require(
                type(value) in (int, float)
                and math.isfinite(value)
                and 0 <= value <= 1
                and float(value * 32).is_integer(),
                "invalid strict screen metric",
            )
        require(
            type(forced.get("response_tokens_including_eos")) is int
            and forced["response_tokens_including_eos"] > 0
            and type(forced.get("canonical_response_nll")) in (int, float)
            and math.isfinite(forced["canonical_response_nll"])
            and forced["canonical_response_nll"] >= 0
            and type(forced.get("canonical_response_token_accuracy")) in (int, float)
            and math.isfinite(forced["canonical_response_token_accuracy"])
            and 0 <= forced["canonical_response_token_accuracy"] <= 1,
            "missing/nonfinite teacher-forced diagnostic",
        )
    viable = all(
        arms[1]["metrics"][key] >= 4 / 32 and arms[1]["metrics"][key] > arms[0]["metrics"][key]
        for key in ("answer_accuracy", "exact_proof_accuracy")
    )
    require(report["training_screen_viable"] is viable, "training futility criterion differs")
    raw = report.get("raw_artifacts")
    require(
        isinstance(raw, list)
        and len(raw) == 2
        and {r.get("path") for r in raw} == {"instruction-records.jsonl", "instruction-prompts.jsonl"}
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
        and all(published.get(key) is False for key in FLAGS),
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
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
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
        report = document(result_root / "instruction-probe.json")
        validate_worker_report(report, plan, job, records)
        node = document(result_root / "node-result.json")
        require(
            node.get("diagnostic_complete") is True
            and node.get("exit_code") == 0
            and node.get("job_id") == job
            and node.get("plan_sha256") == sha(canonical(plan))
            and all(node.get(k) is False for k in FLAGS),
            "node completion failed or unbound",
        )
        result.update(
            success=True,
            diagnostic_complete=True,
            result_sha256=sha(read(result_root / "instruction-probe.json")),
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
        concurrent = check_gpu_ceiling()
        require_unclaimed_science(plan)
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
            gpu_concurrency=concurrent,
        )
    if action == "submit":
        require(request.get("authorize") is True, "explicit authorization required")
        verify_parent(plan, accounting=True)
        check_gpu_ceiling()
        require_unclaimed_science(plan)
        require(not directory.exists() and not claim.exists(), "intent exists; reconcile, never resubmit")
        directory.parent.mkdir(parents=True, exist_ok=True)
        claim.parent.mkdir(parents=True, exist_ok=True)
        science_claim = safe(plan["scientific_claim"])
        science_claim.parent.mkdir(parents=True, exist_ok=True)
        write_once(science_claim, canonical(scientific_claim_record(plan)))
        # Atomic reservation in the old scheme prevents the already deployed old CLI racing us.
        write_once(safe(legacy_claim_path()), canonical(legacy_reservation(plan)))
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
        and document(claim)["plan_sha256"] == sha(canonical(plan))
        and document(safe(plan["scientific_claim"])) == scientific_claim_record(plan)
        and document(safe(legacy_claim_path())) == legacy_reservation(plan),
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
            json.loads(files["instruction-probe.json"]), plan, result["receipt"]["job_id"], records
        )
    return files


def ssh_operation(cli, plan, action, authorize=False):
    remote_path = str(Path(plan["release"]) / "source/tools/sdsc_student_instruction.py")
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
    require(
        status.get("job_id") == PARENT_JOB
        and status.get("success") is True
        and status["result"]["result_sha256"] == PARENT_REPORT_SHA,
        "local parent status is not semantically verified",
    )
    checkpoints, config = checkpoint_rows(publication, report)
    contract = helper("sdsc_student_contract")
    dataset = [
        dict(row, source=str(contract.DATASET_ROOT / row["path"]))
        for row in contract.DATASET_FILES
        if row["path"] in DATASET_NAMES
    ]
    intent = execution_intent({name: records[name]["sha256"] for name in TOOLS})
    legacy_local = ROOT / ".sdsc/student-instruction" / LEGACY_INTENT
    require(
        not (legacy_local / "submission-started.json").exists(),
        "legacy local submission started; reconcile before any new preparation",
    )
    plan = dict(
        schema="quest-sdsc-student-instruction-plan-v1",
        task=TASK,
        scope="train-only-scientific-prompt-screen",
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
        candidate_sha256=CANDIDATE_SHA,
        baseline_sha256=BASELINE_SHA,
        screen_document_sha256=records[SCREEN_DOC]["sha256"],
        dataset_inputs=dataset,
        control_sha256={name: records[name]["sha256"] for name in TOOLS},
        release=str(CONTROL / "releases" / manifest["run_id"]),
        submission_dir=str(CONTROL / "student-instruction-submissions" / intent),
        claim=str(CONTROL / "student-instruction-claims" / (intent + ".json")),
        scientific_claim=str(scientific_claim_path()),
        result_dir=str(PROJECT / "student-instruction" / intent),
        job_name="opd-si-" + intent,
        python=receipt["python"],
        hf_home=receipt["hf_home"],
        student_accepted=False,
        g0_passed=False,
        pilot_passed=False,
        factorial_ready=False,
        teacher_accepted=False,
        teacher_accepted_under_candidate=False,
        accepted_science=False,
        formal_prompt_accepted=False,
    )
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-instruction" / intent
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
                path.parent.parent == ROOT / ".sdsc/student-instruction" and path.name == "plan.json",
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
