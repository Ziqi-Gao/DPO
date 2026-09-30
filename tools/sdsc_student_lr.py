#!/usr/bin/env python3
"""One fixed train-only two-arm learning-rate diagnostic on the original student.

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
TASK = "qwen3-v2-student-lr-diagnostic-v1"
PARENT_JOB = "54548846"
PARENT_INTENT = "594d09aee22c4bdabbe29b0501705344"
PARENT_REPORT_SHA = "91794e528ec83ff0c7f35bd3734d36022d60db2794c3e41c7686beb97fbff4f1"
RESOURCES = dict(
    account="nwu181",
    partition="nairr-gpu-shared",
    qos="nairr-gpu-shared-normal",
    gpus=2,
    gpu_type="h100",
    cpus=24,
    mem_gib=384,
    time="01:00:00",
)
CAP = 1024**2
MAX_FILE = 8 * CAP
MAX_FETCH = 16 * CAP
NAMES = (
    "lr-probe.json",
    "lr-batches.jsonl",
    "lr-records.jsonl",
    "lr-prompts.jsonl",
    "node-result.json",
    "memory.json",
)
TOOLS = (
    "tools/sdsc_student_lr.py",
    "tools/sdsc_student_lr_job.py",
    "tools/sdsc_student_lr_probe.py",
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    "tools/sdsc_student_contract.py",
    "tools/sdsc_student_job.py",
    "tools/sdsc_student_memory.py",
    "tools/sdsc_student_quality.py",
    "tools/sdsc_student_quality_probe.py",
    "tools/sdsc_provenance.py",
    "docs/refactor/sdsc_student_lr_diagnosis_20260930.md",
)


# Immutable, explicitly pinned pure I/O helpers. No old task/validator globals are changed.
SHARED_SHA = "5fcf65d7d999d886067ee5acb898364c696850b5bd778ac6e95976ae91e51e50"
_shared_path = Path(__file__).with_name("sdsc_student_quality.py")
if hashlib.sha256(_shared_path.read_bytes()).hexdigest() != SHARED_SHA:
    raise ValueError("frozen shared quality helper changed")
_shared_spec = importlib.util.spec_from_file_location("_lr_shared_io", _shared_path)
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
BASELINE_SHA = "8126867f5b5d70543fb71fad3e94aa5d71d77976909696a44a612f5388076ea6"
INITIAL_SHA = "85f33ccb8ab32a15098ba0a7035cc1186a369a1f999374daf3d5545e81954fe4"
CONFIG_SHA = "05872b4521802813640004bf614dad65a6e1d2c3c3662a7502ca11bec5e15dbe"
PROTOCOL_DOC = "docs/refactor/sdsc_student_lr_diagnosis_20260930.md"
DATASET_NAMES = {"manifest.json", "train/examples.jsonl"}

MAX_PLAN = 8 * CAP
MAX_AUDIT_CAPTURE = 2 * CAP
AUDIT_REPORT_SHA = "86e5cca158476439cc9f73881bd7e73869b4a6837d1089284369156c79f9c933"
AUDIT_CAPTURE_SHA = "aae6b2fd3fae601f265998416a9d2de8a7340c8672e1bcf2ef026508ebdff8b5"
ARMS = (("lr-5e-4", 0.0005), ("lr-5e-5", 0.00005))
LARGE_NAMES = tuple("checkpoints/" + label + "-step4.pt" for label, _ in ARMS)


def teacher_inputs():
    contract = helper("sdsc_student_contract")
    result = [
        dict(
            path=row["path"],
            source=str(contract.QUALIFICATION_ROOT / row["source"]),
            size=row["size"],
            sha256=row["sha256"],
        )
        for row in contract.STUDENT_INPUT_FILES
    ]
    result += [
        dict(row, path="acceptance/" + row["path"], source=str(contract.ACCEPTANCE_ROOT / row["path"]))
        for row in contract.ACCEPTANCE_FILES
    ]
    return result


def science_identity():
    contract = helper("sdsc_student_contract")
    return dict(
        task=TASK,
        parent_report_sha256=PARENT_REPORT_SHA,
        initial_sha256=INITIAL_SHA,
        original_instruction_sha256=BASELINE_SHA,
        candidate_instruction=None,
        teacher_accepted_sha256=contract.ACCEPTED_SHA256,
        teacher_inputs_sha256=sha(canonical(teacher_inputs())),
        dataset_manifest_sha256="bee7baf767f04ee153ec7ad5f4da535d7fb3c31ba274cf3a0e5d66a01ac6c189",
        train_file_sha256="377538a779f31246eb9aee0ee3283755641149f8dd713c942693f3da2ab1bf4b",
        arms=[dict(label=label, learning_rate=rate) for label, rate in ARMS],
        seed=42,
        optimizer_windows=4,
        global_batch_size=64,
        training_population="original-first256-train-manifest-order-demo-cursor-zero",
        ce_population="fixed-original-first64-selected-teacher-targets",
        generation_population="fixed-original-first32-train",
        generation_steps=[0, 4],
        generation=dict(
            max_new_tokens=256,
            max_model_input_length=1536,
            do_sample=False,
            use_cache=False,
            truncation=False,
        ),
        precision="original-v6-W2-FULL_SHARD-FP32-master-BF16-forward",
    )


def execution_intent(pins):
    return sha(
        canonical(
            dict(
                science=science_identity(),
                controls=pins,
                resources=RESOURCES,
                audit_report_sha256=AUDIT_REPORT_SHA,
                audit_capture_sha256=AUDIT_CAPTURE_SHA,
            )
        )
    )[:32]


def scientific_claim_path():
    return CONTROL / "student-lr-scientific-claims" / (sha(canonical(science_identity())) + ".json")


def scientific_claim_record(plan):
    return dict(
        science_identity=science_identity(),
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        submission_dir=plan["submission_dir"],
        no_retry=True,
    )


def require_unclaimed_science(plan):
    require(
        not safe(plan["scientific_claim"]).exists(),
        "scientific diagnostic already claimed; reconcile its winning intent, never resubmit",
    )


def audit_payload(raw):
    return dict(size=len(raw), sha256=sha(raw), base64=base64.b64encode(raw).decode())


def decode_audit(audit):
    require(isinstance(audit, dict) and set(audit) == {"report", "capture"}, "audit evidence missing")
    result = {}
    for name, expected, limit in (
        ("report", AUDIT_REPORT_SHA, CAP),
        ("capture", AUDIT_CAPTURE_SHA, MAX_AUDIT_CAPTURE),
    ):
        row = audit[name]
        require(
            isinstance(row, dict)
            and set(row) == {"size", "sha256", "base64"}
            and type(row["size"]) is int
            and 0 < row["size"] <= limit
            and row["sha256"] == expected
            and re.fullmatch("[a-f0-9]{64}", expected),
            "unreviewed CPU audit identity or size",
        )
        raw = base64.b64decode(row["base64"], validate=True)
        require(len(raw) == row["size"] and sha(raw) == expected, "CPU audit bytes differ")
        result[name] = raw
    report = json.loads(result["report"])
    require(
        report.get("schema") == "quest-sdsc-student-lr-data-audit-v1"
        and report.get("passed") is True
        and report.get("diagnostic_complete") is True
        and all(report.get(k) is False for k in FLAGS),
        "CPU data-path prerequisite did not pass or overclaims acceptance",
    )
    require(
        report.get("parent_job_id") == PARENT_JOB
        and report.get("initial_checkpoint_sha256") == INITIAL_SHA
        and report.get("baseline_instruction_sha256") == BASELINE_SHA
        and report.get("source_commit") == "6c04f804b302184b8ff95d00fab404e0531ed8d6"
        and report.get("capture_sha256") == AUDIT_CAPTURE_SHA
        and report.get("real_source_and_prepare_targets_verified") is True
        and report.get("sequence_shifted_ce_verified") is True
        and report.get("all_response_tokens_supervised") is True
        and report.get("all_responses_end_in_single_eos") is True
        and report.get("real_student_logits_verified") is False
        and report.get("training_started") is False,
        "CPU audit original inputs or evidence scope differs",
    )
    capture = json.loads(result["capture"])
    rows = capture.get("rows")
    require(
        capture.get("schema") == "quest-sdsc-student-lr-first64-capture-v1"
        and isinstance(rows, list)
        and len(rows) == 64
        and [row.get("global_slot") for row in rows] == list(range(64))
        and all(type(row.get("global_slot")) is int for row in rows)
        and [row.get("prompt_id") for row in rows] == report.get("ordered_prompt_ids")
        and len(set(report["ordered_prompt_ids"])) == 64
        and [row.get("attempt_id") for row in rows] == report.get("selected_attempt_ids")
        and sha(canonical(rows)) == report.get("window_sha256"),
        "CPU audit fixed first64 selection differs",
    )
    require(
        sum(len(row["input_ids"]) for row in rows) == report.get("first64_input_tokens") == 53440
        and sum(len(row["response_ids"]) for row in rows) == report.get("first64_response_tokens") == 5896,
        "CPU audit does not reproduce the historical first window",
    )
    return result


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
        plan["schema"] == "quest-sdsc-student-lr-plan-v1"
        and plan["task"] == TASK
        and canonical(plan["resources"]) == canonical(RESOURCES)
        and plan["scope"] == "train-only-learning-rate-diagnostic",
        "unreviewed diagnostic task/resources/scope",
    )
    require(len(canonical(plan)) <= MAX_PLAN, "diagnostic plan exceeds bound")
    require(
        plan.get("science_identity") == science_identity()
        and plan.get("protocol_document_sha256") == plan["control_sha256"][PROTOCOL_DOC]
        and plan["control_sha256"]["tools/sdsc_student_quality.py"] == SHARED_SHA,
        "frozen scientific/document/helper identity differs",
    )
    require(plan.get("teacher_inputs") == teacher_inputs(), "accepted teacher inputs differ")
    decode_audit(plan.get("audit"))
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
        and plan["submission_dir"] == str(CONTROL / "student-lr-submissions" / plan["intent_id"])
        and plan["claim"] == str(CONTROL / "student-lr-claims" / (plan["intent_id"] + ".json"))
        and plan["result_dir"] == str(PROJECT / "student-lr" / plan["intent_id"])
        and plan["job_name"] == "opd-slr-" + plan["intent_id"],
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
    require(proof["student_inputs"] == plan["teacher_inputs"], "parent teacher inputs differ")
    for row in plan["teacher_inputs"] + plan["dataset_inputs"]:
        info = safe(row["source"]).lstat()
        require(
            stat.S_ISREG(info.st_mode) and info.st_size == row["size"], "original input size/type changed"
        )
        if row["size"] <= 4 * CAP:
            require(sha(read(row["source"], 4 * CAP)) == row["sha256"], "small original input changed")
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
        "--gpus=h100:2",
        "--mem=393216M",
        "--time=01:00:00",
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
    require(total + RESOURCES["gpus"] <= 4, "new diagnostic would exceed four allocatable GPUs")
    return dict(
        existing_requested_gpus=total, new_gpus=RESOURCES["gpus"], jobs=jobs, limit=4, observation_only=True
    )


def job_script(plan):
    worker = str(Path(plan["release"]) / "source/tools/sdsc_student_lr_job.py")
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
                plan["control_sha256"]["tools/sdsc_student_lr_job.py"],
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


def validate_submission_receipt(receipt, plan):
    expected = dict(
        task=TASK,
        intent_id=plan["intent_id"],
        run_id=plan["run_id"],
        plan_sha256=sha(canonical(plan)),
        code_sha256=plan["code_sha256"],
        result_dir=plan["result_dir"],
        resources=RESOURCES,
    )
    require(
        all(canonical(receipt.get(k)) == canonical(v) for k, v in expected.items())
        and re.fullmatch("[1-9][0-9]*", str(receipt.get("job_id"))),
        "submission receipt differs",
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
                and row[8] in {"384Gn", "393216Mn", "393216M", "384G"}
                and tres.get("cpu") == "24"
                and tres.get("gres/gpu") == "2"
                and tres.get("node") == "1",
                "actual diagnostic allocation differs",
            )
            success = True
    elif queue["stdout"].strip():
        entries = [line.split("|") for line in queue["stdout"].splitlines() if line.strip()]
        require(len(entries) == 1 and entries[0][0] == job and len(entries[0]) >= 2, "queue identity differs")
        state = entries[0][1]
    return dict(job_id=job, state=state, accounting_complete=success, queue=queue, accounting=account)


def validate_nll(value, ids, expected_tokens):
    rows = value.get("per_example")
    require(
        isinstance(rows, list)
        and [r.get("prompt_id") for r in rows] == ids
        and value.get("examples") == len(ids),
        "CE diagnostic cohort differs",
    )
    for row in rows:
        require(
            type(row.get("response_tokens")) is int
            and row["response_tokens"] > 0
            and type(row.get("correct_tokens")) is int
            and 0 <= row["correct_tokens"] <= row["response_tokens"],
            "invalid CE token counts",
        )
        for name in ("hf_sequence_nll", "production_sequence_nll", "labels_ce_abs_error"):
            require(
                type(row.get(name)) in (int, float) and math.isfinite(row[name]) and row[name] >= 0,
                "invalid/nonfinite CE measurement",
            )
        error = abs(row["hf_sequence_nll"] - row["production_sequence_nll"])
        require(
            math.isclose(error, row["labels_ce_abs_error"], rel_tol=1e-12, abs_tol=1e-12)
            and error <= 2e-5 * max(1, abs(row["hf_sequence_nll"])),
            "independent labels CE differs",
        )
    tokens = sum(r["response_tokens"] for r in rows)
    require(tokens == expected_tokens == value.get("response_tokens_including_eos"), "CE denominator differs")
    expected = dict(
        sequence_mean_nll=sum(r["production_sequence_nll"] for r in rows) / len(rows),
        hf_sequence_mean_nll=sum(r["hf_sequence_nll"] for r in rows) / len(rows),
        token_mean_nll=sum(r["hf_sequence_nll"] * r["response_tokens"] for r in rows) / tokens,
        token_accuracy=sum(r["correct_tokens"] for r in rows) / tokens,
        max_labels_ce_abs_error=max(r["labels_ce_abs_error"] for r in rows),
    )
    require(
        all(
            type(value.get(k)) in (int, float) and math.isclose(value[k], v, rel_tol=1e-12, abs_tol=1e-12)
            for k, v in expected.items()
        ),
        "CE aggregate differs from per-example evidence",
    )


def validate_worker_report(report, plan, job, records):
    canonical(report)  # Reject nonfinite diagnostic numbers at any nesting depth.
    require(
        report.get("schema") == "quest-sdsc-student-lr-probe-v1"
        and report.get("diagnostic_complete") is True
        and report.get("passed") is True
        and all(report.get(k) is False for k in FLAGS),
        "diagnostic failed or overclaims scientific acceptance",
    )
    require(
        report.get("job_id") == job
        and report.get("parent_job_id") == PARENT_JOB
        and report.get("run_id") == plan["run_id"]
        and report.get("source_code_sha256") == plan["code_sha256"]
        and report.get("plan_sha256") == sha(canonical(plan)),
        "diagnostic worker identity differs",
    )
    require(
        report.get("initial_checkpoint_sha256") == INITIAL_SHA
        and report.get("original_instruction_sha256") == BASELINE_SHA
        and report.get("data_audit_sha256") == AUDIT_REPORT_SHA
        and report.get("data_capture_sha256") == AUDIT_CAPTURE_SHA,
        "diagnostic original checkpoint/instruction/CPU audit differs",
    )
    require(
        report.get("raw_record_count") == 128 and type(report.get("raw_record_count")) is int,
        "diagnostic must publish 128 complete generation responses",
    )
    arms = report.get("arms")
    require(isinstance(arms, list) and len(arms) == 2, "diagnostic must complete both fixed arms")
    audit = json.loads(decode_audit(plan["audit"])["report"])
    ids = audit["ordered_prompt_ids"]
    require(report.get("audited_window_sha256") == audit["window_sha256"], "worker audited window differs")
    previous_arm = None
    for arm_index, (arm, (label, lr)) in enumerate(zip(arms, ARMS, strict=True)):
        require(
            arm.get("arm") == label
            and type(arm.get("learning_rate")) is float
            and arm["learning_rate"] == lr,
            "diagnostic arm or learning rate differs",
        )
        initial = arm.get("initial_loading", {})
        require(
            initial.get("checkpoint_sha256") == INITIAL_SHA
            and initial.get("all_tensors_exact") is True
            and initial.get("all_parameters_trainable") is True
            and type(initial.get("key_count")) is int
            and initial["key_count"] > 0,
            "original initial exact load/full-parameter scope missing",
        )
        require(
            re.fullmatch("[a-f0-9]{64}", str(arm.get("initial_state_sha256")))
            and isinstance(arm.get("initial_rng_by_rank"), list)
            and len(arm["initial_rng_by_rank"]) == 2
            and all(re.fullmatch("[a-f0-9]{64}", str(x)) for x in arm["initial_rng_by_rank"]),
            "initial master/RNG evidence missing",
        )
        if previous_arm is not None:
            require(
                arm["initial_state_sha256"] == previous_arm["initial_state_sha256"]
                and arm["initial_rng_by_rank"] == previous_arm["initial_rng_by_rank"],
                "arms did not start from identical master weights/RNG",
            )
        expected_execution = dict(
            protocol="allocation_neutral_exact_global_batch_v1",
            global_batch_size=64,
            max_microbatch_size=4,
            max_model_input_length=1536,
            world_size=2,
            microsteps_per_optimizer_update=8,
            rank_local_sequence_counts=[32, 32],
            microbatch_sizes_by_rank=[[4] * 8, [4] * 8],
            requested_fsdp_sharding_strategy="FULL_SHARD",
            effective_fsdp_sharding_strategy="FULL_SHARD",
            fsdp_wrapper_count=29,
        )
        require(
            canonical(arm.get("prepared_execution")) == canonical(expected_execution),
            "prepared W2/FSDP/exact-global training differs",
        )
        validate_nll(arm["teacher_target_initial"], ids, 5896)
        validate_nll(arm["canonical_initial"], ids[:32], 2936)
        validate_nll(arm["canonical_final"], ids[:32], 2936)
        updates = arm.get("updates")
        require(
            isinstance(updates, list)
            and len(updates) == 4
            and [x.get("step") for x in updates] == [1, 2, 3, 4]
            and all(type(x.get("step")) is int for x in updates),
            "diagnostic must complete exactly four optimizer updates",
        )
        previous_ce = arm["teacher_target_initial"]
        for step, update in enumerate(updates, 1):
            validate_nll(update["teacher_target_before"], ids, 5896)
            validate_nll(update["teacher_target_after"], ids, 5896)
            require(
                all(
                    abs(left["production_sequence_nll"] - right["production_sequence_nll"])
                    <= 2e-5 * max(1.0, abs(right["production_sequence_nll"]))
                    for left, right in zip(
                        update["teacher_target_before"]["per_example"],
                        previous_ce["per_example"],
                        strict=True,
                    )
                ),
                "fixed CE changed between updates beyond numerical comparison tolerance",
            )
            previous_ce = update["teacher_target_after"]
            ranks = update.get("optimizer_by_rank")
            require(isinstance(ranks, list) and len(ranks) == 2, "optimizer rank evidence incomplete")
            for rank, row in enumerate(ranks):
                require(
                    type(row.get("rank")) is int
                    and row["rank"] == rank
                    and type(row.get("step")) is int
                    and row["step"] == step
                    and row.get("all_parameter_steps_equal") is True
                    and type(row.get("parameter_tensors")) is int
                    and row["parameter_tensors"] > 0
                    and type(row.get("scheduler_last_epoch")) is int
                    and row["scheduler_last_epoch"] == step
                    and canonical(row.get("microstep_sync")) == canonical([False] * 7 + [True])
                    and canonical(row.get("actual_optimizer_calls")) == canonical([step - 1] * 7 + [step]),
                    "actual optimizer/scheduler/microstep cadence differs",
                )
                if step == 1:
                    prediction = row.get("first_step_prediction", {})
                    require(
                        type(prediction.get("elements")) is int
                        and prediction["elements"] > 0
                        and prediction.get("gradient_squared", 0) > 0
                        and prediction.get("update_squared", 0) > 0
                        and type(prediction.get("roundoff_violations")) is int
                        and prediction["roundoff_violations"] == 0
                        and 0 <= prediction.get("max_roundoff_ratio", 2) <= 1
                        and prediction.get("gradient_dot_update", 1)
                        <= prediction.get("gradient_dot_roundoff_bound", 0),
                        "first-step AdamW prediction failed",
                    )
            cursors = update.get("cursors_by_rank")
            require(
                isinstance(cursors, list)
                and len(cursors) == 2
                and all(
                    x.get("rank") == rank
                    and len(x.get("cursor", {})) == step * 32
                    and all(type(v) is int and v == 1 for v in x["cursor"].values())
                    for rank, x in enumerate(cursors)
                )
                and set(cursors[0]["cursor"]).isdisjoint(cursors[1]["cursor"]),
                "training windows repeat or overlap prompts",
            )
            if previous_arm is not None:
                require(
                    cursors == previous_arm["updates"][step - 1]["cursors_by_rank"],
                    "LR arms used different source cursor windows",
                )
        generation = arm.get("generation")
        require(
            isinstance(generation, list)
            and len(generation) == 2
            and [x.get("step") for x in generation] == [0, 4],
            "diagnostic generation boundaries differ",
        )
        for index, row in enumerate(generation):
            require(
                type(row.get("step")) is int
                and row.get("record_count") == 32
                and row.get("record_start") == arm_index * 64 + index * 32
                and row.get("metrics", {}).get("count") == 32
                and row.get("strict_fp32_loading", {}).get("all_tensors_exact") is True
                and row["strict_fp32_loading"].get("loaded_before_bf16_copy") is True,
                "exported-model generation population/load differs",
            )
        parity = arm.get("root_vs_export_logits")
        require(
            isinstance(parity, list)
            and len(parity) == 4
            and [x.get("ordinal") for x in parity] == [0, 1, 2, 3]
            and all(
                type(x.get("elements")) is int
                and x["elements"] > 0
                and type(x.get("argmax_mismatches")) is int
                and 0 <= x["argmax_mismatches"] <= x.get("compared_positions", -1)
                and x.get("max_abs_error", -1) >= 0
                and x.get("rms_error", -1) >= 0
                and x.get("formal_parity_gate_applied") is False
                for x in parity
            ),
            "root/export finite diagnostic comparison missing",
        )
        previous_arm = arm
    require(
        report.get("generation") == science_identity()["generation"], "diagnostic generation contract differs"
    )
    raw = report.get("raw_artifacts")
    require(
        isinstance(raw, list)
        and len(raw) == 3
        and {r.get("path") for r in raw} == {"lr-records.jsonl", "lr-prompts.jsonl", "lr-batches.jsonl"}
        and all(records.get(r["path"]) == r for r in raw),
        "worker raw artifact hash binding differs",
    )
    worker_checkpoints(report.get("checkpoints"))


def validate_job_memory_events(evidence, job):
    """Reject observed allocation failures inside this job, not shared history."""
    require(isinstance(evidence, dict), "memory event evidence must be an object")
    ancestors = evidence.get("ancestors")
    require(isinstance(ancestors, list) and ancestors, "raw memory hierarchy missing")
    for row in ancestors:
        require(isinstance(row, dict) and isinstance(row.get("path"), str), "invalid memory ancestor")
        if "job_" + str(job) not in Path(row["path"]).parts:
            continue
        counters = []
        if row.get("memory_failcnt") is not None:
            counters.append(("memory_failcnt", row["memory_failcnt"]))
        for field, keys in (
            ("memory_events", ("oom", "oom_kill", "oom_group_kill")),
            ("memory_events_local", ("oom", "oom_kill", "oom_group_kill")),
            ("memory_oom_control", ("under_oom", "oom_kill")),
        ):
            values = row.get(field)
            if values is None:
                continue  # This cgroup/kernel did not expose that counter file.
            require(isinstance(values, dict), "invalid job memory counter mapping: " + field)
            counters.extend((field + "." + key, values[key]) for key in keys if key in values)
        for name, value in counters:
            require(type(value) is int and value >= 0, "invalid job memory counter: " + name)
            require(value == 0, "job memory allocation failure counter is nonzero: " + name)


def validate_memory(report, job):
    """Recompute raw limiting-ancestor peaks, including checkpoint publication."""
    phases = ("initial", "after_staging", "after_training", "final", "after_publication")
    require(
        isinstance(report, dict) and set(phases) <= set(report),
        "successful diagnostic omits memory/publication measurement",
    )
    for evidence in report.values():
        validate_job_memory_events(evidence, job)
    previous_peak, previous_time, boundaries = 0, 0, None
    limit = 384 * 1024**3
    for phase in phases:
        evidence = report[phase]
        ancestors = evidence.get("ancestors")
        require(isinstance(ancestors, list) and ancestors, "raw memory hierarchy missing")
        finite = [x for x in ancestors if x.get("limit_bytes") is not None]
        require(
            finite and all(type(x.get("limit_bytes")) is int and x["limit_bytes"] > 0 for x in finite),
            "invalid raw memory limit",
        )
        require(min(x["limit_bytes"] for x in finite) == limit, "actual memory allocation differs")
        limiting = [x for x in finite if x["limit_bytes"] == limit]
        current_boundaries = sorted((x["path"], x["limit_bytes"]) for x in finite)
        require(boundaries is None or current_boundaries == boundaries, "memory cgroup boundaries changed")
        boundaries = current_boundaries
        for row in limiting:
            require(
                "job_" + job in Path(row["path"]).parts
                and all(type(row.get(k)) is int for k in ("current_bytes", "peak_bytes"))
                and 0 < row["current_bytes"] <= row["peak_bytes"] <= limit,
                "memory peak/current evidence differs from this allocation",
            )
        selected = max(limiting, key=lambda x: x["peak_bytes"])
        peak = selected["peak_bytes"]
        observed = evidence.get("observed_at_unix")
        required = max(32 * 1024**3, math.ceil(limit * 0.20))
        require(
            evidence.get("passed") is True
            and evidence.get("expected_limit_bytes") == limit == evidence.get("limit_bytes")
            and evidence.get("path") == selected["path"]
            and evidence.get("peak_bytes") == peak
            and evidence.get("current_bytes") == selected["current_bytes"]
            and evidence.get("headroom_bytes") == limit - peak >= required
            and evidence.get("minimum_headroom_bytes") == required
            and type(observed) in (int, float)
            and math.isfinite(observed)
            and observed >= previous_time
            and peak >= previous_peak,
            "memory aggregate peak, chronology or required headroom differs",
        )
        previous_peak, previous_time = peak, observed


def worker_checkpoints(rows):
    require(isinstance(rows, list) and len(rows) == 2, "both model-only worker checkpoints required")
    physical = []
    for row, (arm, _) in zip(rows, ARMS, strict=True):
        require(
            isinstance(row, dict)
            and set(row) == {"path", "size", "sha256", "arm", "step", "scope"}
            and row["arm"] == arm
            and type(row["step"]) is int
            and row["step"] == 4
            and row["scope"] == "diagnostic_model_only"
            and row["path"] == "checkpoints/" + arm + "-step4.pt",
            "worker checkpoint scope/order differs",
        )
        physical.append({key: row[key] for key in ("path", "size", "sha256")})
    large_records(physical, complete=True)
    return physical


def validate_node_result(node, plan, job, large):
    expected = dict(
        task=TASK,
        job_id=job,
        parent_job_id=PARENT_JOB,
        run_id=plan["run_id"],
        source_code_sha256=plan["code_sha256"],
        plan_sha256=sha(canonical(plan)),
        world_size=2,
        cpus=24,
        memory_mib=393216,
        account="nwu181",
        partition="nairr-gpu-shared",
    )
    require(
        node.get("diagnostic_complete") is True
        and type(node.get("exit_code")) is int
        and node["exit_code"] == 0
        and all(node.get(k) is False for k in FLAGS)
        and all(canonical(node.get(k)) == canonical(v) for k, v in expected.items())
        and node.get("large_artifacts") == large,
        "node completion/allocation/checkpoint evidence failed or unbound",
    )


def large_records(rows, *, complete):
    require(isinstance(rows, list), "large model checkpoint inventory missing")
    result = {}
    for row in rows:
        require(
            isinstance(row, dict)
            and set(row) == {"path", "size", "sha256"}
            and row["path"] in LARGE_NAMES
            and row["path"] not in result
            and type(row["size"]) is int
            and 0 < row["size"] <= 8 * 1024**3
            and re.fullmatch("[a-f0-9]{64}", str(row["sha256"])),
            "invalid/duplicate/oversized model-only checkpoint",
        )
        result[row["path"]] = row
    require(not complete or set(result) == set(LARGE_NAMES), "both final model checkpoints required")
    return result


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
    large_records(published.get("large_files"), complete=published["passed"])
    require(published.get("large_files_read_back_verified") is True, "node checkpoint readback missing")
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
        large = large_records(published["large_files"], complete=published["passed"])
        for name, row in large.items():
            info = safe(result_root / name).lstat()
            require(
                stat.S_ISREG(info.st_mode) and info.st_size == row["size"],
                "persistent model checkpoint missing or size/type changed",
            )
        result.update(
            large_files_verified_by_node=True,
            large_files_rehashed_on_login=False,
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
        report = document(result_root / "lr-probe.json")
        validate_worker_report(report, plan, job, records)
        require(
            worker_checkpoints(report["checkpoints"]) == result["publication"]["large_files"],
            "worker/node checkpoint identity differs",
        )
        validate_memory(document(result_root / "memory.json"), job)
        node = document(result_root / "node-result.json")
        validate_node_result(node, plan, job, result["publication"]["large_files"])
        result.update(
            success=True,
            diagnostic_complete=True,
            result_sha256=sha(read(result_root / "lr-probe.json")),
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
            cpu_data_audit_verified=True,
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
        and document(safe(plan["scientific_claim"])) == scientific_claim_record(plan),
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
    validate_submission_receipt(receipt, plan)
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
    validate_submission_receipt(result["receipt"], plan)
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
        report = json.loads(files["lr-probe.json"])
        validate_worker_report(report, plan, result["receipt"]["job_id"], records)
        require(
            worker_checkpoints(report["checkpoints"]) == publication["large_files"],
            "fetched checkpoint identity differs",
        )
        validate_memory(json.loads(files["memory.json"]), result["receipt"]["job_id"])
        validate_node_result(
            json.loads(files["node-result.json"]),
            plan,
            result["receipt"]["job_id"],
            publication["large_files"],
        )
    return files


def ssh_operation(cli, plan, action, authorize=False):
    remote_path = str(Path(plan["release"]) / "source/tools/sdsc_student_lr.py")
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
    plan = dict(
        schema="quest-sdsc-student-lr-plan-v1",
        task=TASK,
        scope="train-only-learning-rate-diagnostic",
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
        science_identity=science_identity(),
        teacher_inputs=teacher_inputs(),
        audit=dict(
            report=audit_payload(read(args.audit_report.absolute(), CAP)),
            capture=audit_payload(read(args.audit_capture.absolute(), MAX_AUDIT_CAPTURE)),
        ),
        protocol_document_sha256=records[PROTOCOL_DOC]["sha256"],
        dataset_inputs=dataset,
        control_sha256={name: records[name]["sha256"] for name in TOOLS},
        release=str(CONTROL / "releases" / manifest["run_id"]),
        submission_dir=str(CONTROL / "student-lr-submissions" / intent),
        claim=str(CONTROL / "student-lr-claims" / (intent + ".json")),
        scientific_claim=str(scientific_claim_path()),
        result_dir=str(PROJECT / "student-lr" / intent),
        job_name="opd-slr-" + intent,
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
    directory = ROOT / ".sdsc/student-lr" / intent
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
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--audit-capture", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "remote":
        require(
            Path(__file__).resolve().is_relative_to(CONTROL / "releases"),
            "remote actions cannot run on Quest",
        )
        raw = sys.stdin.buffer.read(MAX_PLAN + 1024 + 1)
        require(len(raw) <= MAX_PLAN + 1024, "remote request exceeds bound")
        result = remote_action(json.loads(raw))
    else:
        cli = helper("sdsc_cli")
        if args.action == "prepare":
            require(
                args.run_id
                and args.fetch_dir
                and args.status_file
                and args.audit_report
                and args.audit_capture,
                "prepare requires run-id/fetch-dir/status-file/audit-report/audit-capture",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "operation requires --plan")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-lr" and path.name == "plan.json",
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
