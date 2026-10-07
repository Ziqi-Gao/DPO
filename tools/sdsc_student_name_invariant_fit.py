#!/usr/bin/env python3
"""One normal fit after a verified recovered preflight; both scientific sources remain immutable."""

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
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
TASK = "qwen3-v2-student-name-invariant-fit-v1"
SCHEMA = "quest-sdsc-student-name-invariant-fit-plan-v1"
CONTRACT_PATH = "prereg/amendments/qwen3_student_name_invariant_fit_execution_v1.json"
CAP = 1024**2
MAX_PLAN, EXECUTION_MAX, MAX_FETCH, MAX_RESPONSE = 8 * CAP, 8 * CAP, 232 * CAP, 336 * CAP
OBSERVED_TIMEOUT, SSH_TIMEOUT = 180, 900
CONTROLLER = "tools/sdsc_student_name_invariant_fit.py"
NODE = "tools/sdsc_student_name_invariant_fit_job.py"
FENCE = "tools/sdsc_student_execution_fence.py"
OBSERVED = "tools/sdsc_observed_command.py"
EXECUTION_NAMES = (
    "execution-plan.json",
    "execution-receipt.json",
    "admission.json",
    "submission-started.json",
    "submission-observation.json",
    "unknown.json",
    "fit-node-entry.json",
    "fit-node-exit.json",
)


def helper(name, root=ROOT):
    path = Path(root) / "tools" / (name + ".py")
    if "contract" in globals():
        expected = contract.FROZEN_DEPENDENCIES.get("tools/" + name + ".py")
        if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen recovery dependency changed: " + name)
    spec = importlib.util.spec_from_file_location("_name_invariant_recovery_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = helper("sdsc_student_name_invariant_fit_contract")
recovery = helper("sdsc_student_name_invariant_recovery")
science = helper("sdsc_student_name_invariant")
require, canonical, sha, now = science.require, science.canonical, science.sha, science.now
safe, relative, write_once = science.safe, science.relative, science.write_once
FLAGS, NEW_TOOLS = science.FLAGS, contract.CONTROL_PATHS


def read(path, limit=16 * CAP):
    return science.read(path, limit)


def decode(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out

    return json.loads(
        raw, object_pairs_hook=unique, parse_constant=lambda value: require(False, "nonfinite JSON value")
    )


def document(path, expected=None, limit=16 * CAP):
    raw = read(path, limit)
    require(expected is None or sha(raw) == expected, "artifact SHA differs")
    return decode(raw)


def binding(root=ROOT, expected_head=None):
    root = Path(root)
    return (
        helper("sdsc_student_name_invariant_fit_contract", root)
        .resolve_execution_contract(
            root,
            expected_head=expected_head,
            git_dir=root / (".opd-git" if (root / ".opd-git").is_dir() else ".git"),
        )
        .as_dict()
    )


def plan_sha(plan):
    return sha(canonical(plan))


def load_plan(path, expected_sha=None):
    return validate_plan(document(path, expected_sha, MAX_PLAN))


def science_plan(plan):
    return plan["science_plan"]


def claim_record(plan):
    return dict(
        task=TASK,
        intent_id=plan["intent_id"],
        plan_sha256=plan_sha(plan),
        science_plan_sha256=plan["science_plan_sha256"],
        preflight_job_id=contract.PREFLIGHT["job_id"],
        preflight_recovery_sha256=plan["preflight_recovery_sha256"],
        no_retry=True,
    )


def bound_helper(plan, name):
    relative_name = "tools/" + name + ".py"
    require(
        sha(read(ROOT / relative_name)) == plan["execution"]["control_file_sha256"][relative_name],
        "reviewed execution helper changed: " + name,
    )
    return helper(name)


def validate_source_descriptor(value):
    require(
        isinstance(value, dict)
        and set(value) == {"run_id", "code_sha256", "manifest_sha256", "release", "provenance"},
        "execution source fields differ",
    )
    require(
        isinstance(value["run_id"], str)
        and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value["run_id"])
        and value["release"] == str(CONTROL / "releases" / value["run_id"]),
        "execution release differs",
    )
    require(
        all(
            isinstance(value[k], str) and re.fullmatch("[a-f0-9]{64}", value[k])
            for k in ("code_sha256", "manifest_sha256")
        ),
        "execution source digest differs",
    )
    p = value["provenance"]
    require(
        isinstance(p, dict)
        and set(p) == {"directory", "manifest_sha256", "head"}
        and isinstance(p["manifest_sha256"], str)
        and re.fullmatch("[a-f0-9]{64}", p["manifest_sha256"])
        and isinstance(p["head"], str)
        and re.fullmatch("[a-f0-9]{40}", p["head"])
        and p["directory"] == str(CONTROL / "provenance-v2" / p["manifest_sha256"]),
        "execution provenance differs",
    )
    return value


def validate_plan(plan):
    require(
        isinstance(plan, dict)
        and set(plan)
        == {
            "schema",
            "task",
            "intent_id",
            "science_plan",
            "science_plan_sha256",
            "source",
            "execution",
            "preflight_recovery",
            "preflight_recovery_sha256",
            *FLAGS,
        },
        "fit outer fields differ",
    )
    require(
        plan["schema"] == SCHEMA and plan["task"] == TASK and len(canonical(plan)) <= MAX_PLAN,
        "fit outer scope/bound differs",
    )
    inner = science.validate_plan(plan["science_plan"])
    previous = recovery.validate_plan(plan["preflight_recovery"])
    require(
        plan_sha(previous) == plan["preflight_recovery_sha256"] == contract.PREFLIGHT["recovery_plan_sha256"],
        "not the exact accepted recovered preflight",
    )
    old = previous["science_plan"]
    require(
        plan_sha(old) == contract.PREFLIGHT["plan_sha256"]
        and inner["mode"] == "fit"
        and old["mode"] == "preflight",
        "fit needs the exact real preflight",
    )
    require(set(inner) == set(old), "scientific inner fields differ")
    allowed = {
        "mode",
        "resources",
        "science_identity",
        "created_at",
        "intent_id",
        "job_name",
        "submission_dir",
        "result_dir",
        "claim",
        "scientific_claim",
        "preflight",
    }
    require(
        all(canonical(inner[k]) == canonical(old[k]) for k in old if k not in allowed),
        "fit altered original science, inputs, release or runtime",
    )
    require(
        canonical(inner["preflight"]["plan"]) == canonical(old)
        and inner["preflight"]["audit"]["sha256"] == contract.PREFLIGHT["audit_sha256"]
        and inner["preflight"]["audit"]["document"]["publication_sha256"]
        == contract.PREFLIGHT["publication_sha256"]
        and inner["preflight"]["audit"]["document"]["result_sha256"] == contract.PREFLIGHT["result_sha256"]
        and inner["preflight"]["audit"]["document"]["job_id"] == contract.PREFLIGHT["job_id"],
        "original independent preflight audit differs",
    )
    require(
        inner["intent_id"] == plan["intent_id"] != old["intent_id"]
        and plan["science_plan_sha256"] == plan_sha(inner),
        "fit intent/inner digest differs",
    )
    source = validate_source_descriptor(plan["source"])
    execution = plan["execution"]
    require(
        isinstance(execution, dict)
        and set(execution)
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
        all(
            isinstance(execution[k], str) and re.fullmatch("[a-f0-9]{64}", execution[k])
            for k in ("core_sha256", "artifact_sha256")
        ),
        "execution digest differs",
    )
    require(
        all(
            isinstance(execution[k], str) and re.fullmatch("[a-f0-9]{40}", execution[k])
            for k in ("implementation_commit", "acceptance_commit", "head")
        )
        and execution["implementation_commit"] != execution["acceptance_commit"]
        and execution["head"] == source["provenance"]["head"],
        "distinct execution review/source required",
    )
    require(
        set(execution["control_file_sha256"]) == set(NEW_TOOLS)
        and all(
            isinstance(h, str) and re.fullmatch("[a-f0-9]{64}", h)
            for h in execution["control_file_sha256"].values()
        ),
        "execution controls differ",
    )
    current = execution["science_binding"]
    require(
        set(current) == set(inner["protocol"])
        and current["head"] == execution["head"]
        and all(canonical(current[k]) == canonical(inner["protocol"][k]) for k in current if k != "head"),
        "execution altered accepted scientific binding",
    )
    require(
        all(inner["control_sha256"][p] == h for p, h in recovery.contract.FROZEN_DEPENDENCIES.items()),
        "original scientific controls changed",
    )
    require(all(plan[k] is False for k in FLAGS), "fit transport cannot grant qualification")
    return plan


def verify_source(plan, root=None):
    science.verify_source(plan["science_plan"])
    source = plan["source"]
    manifest = document(safe(source["release"]) / "manifest.json", source["manifest_sha256"])
    require(
        manifest["run_id"] == source["run_id"] and manifest["code_sha256"] == source["code_sha256"],
        "execution release manifest differs",
    )
    rows = science.manifest_records(manifest)
    pins = {
        **contract.FROZEN_DEPENDENCIES,
        **plan["science_plan"]["protocol"]["science_file_sha256"],
        **plan["execution"]["control_file_sha256"],
        CONTRACT_PATH: plan["execution"]["artifact_sha256"],
    }
    require(all(rows.get(p, {}).get("sha256") == h for p, h in pins.items()), "execution source pins differ")
    source_root = safe(root or Path(source["release"]) / "source")
    for name, row in rows.items():
        raw = read(source_root / name, 4 * CAP)
        require(
            len(raw) == row["size"] and sha(raw) == row["sha256"], "execution source byte differs: " + name
        )
    return manifest


def verify_execution(plan, destination=None):
    if destination is None:
        with tempfile.TemporaryDirectory(prefix="name-invariant-fit-verify-") as temp:
            return verify_execution(plan, Path(temp) / "execution")
    source = plan["source"]
    p = source["provenance"]
    restored = helper("sdsc_provenance_v2").verify(
        safe(p["directory"]), p["manifest_sha256"], source["code_sha256"], destination
    )
    require(
        restored["git_head"] == p["head"] and binding(destination, p["head"]) == plan["execution"],
        "restored fit execution review differs",
    )
    # Genuine old science and old recovery acceptance remain independently restored.
    recovery.verify_execution(plan["preflight_recovery"])
    return restored


def verify_fence(plan):
    return recovery.verify_fence(plan["preflight_recovery"])


def verify_recovered_preflight(plan, *, accounting=True):
    previous = plan["preflight_recovery"]
    old = previous["science_plan"]
    recovery.check_claims(previous)
    fence = verify_fence(plan)
    require(fence["sha256"] == contract.PREFLIGHT["fence_sha256"], "prior durable fence differs")
    directory = safe(old["submission_dir"])
    receipt = document(directory / "receipt.json")
    science.validate_submission_receipt(receipt, old)
    require(receipt["job_id"] == contract.PREFLIGHT["job_id"], "foreign preflight job")
    require(
        document(directory / "execution-receipt.json")
        == dict(recovery.claim_record(previous), job_id=receipt["job_id"], scientific_receipt=receipt),
        "recovered preflight receipt differs",
    )
    observation = document(directory / "submission-observation.json", limit=2 * CAP)
    streams = recovery.completed_command(observation, recovery.sbatch(previous), timeout=180)
    match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\s*", streams["stdout"])
    require(
        match is not None and match.group(1) == receipt["job_id"], "preflight real acknowledgement differs"
    )
    require(
        document(directory / "live-binding.json")["job_id"] == receipt["job_id"],
        "preflight live binding missing",
    )
    for name, key in (
        ("recovery-node-entry.json", "entry_sha256"),
        ("recovery-node-exit.json", "exit_sha256"),
    ):
        document(directory / name, contract.PREFLIGHT[key], CAP)
    saved = (
        None
        if accounting
        else document(safe(plan["science_plan"]["submission_dir"]) / "admission.json")["preflight"]
    )
    result = recovery.inspect_job(previous, receipt, accounting_snapshot=saved)
    require(
        result["success"] is True
        and result["stage_complete"] is True
        and result["execution_entry_verified"] is True
        and result["execution_exit_verified"] is True
        and result["publication_sha256"] == contract.PREFLIGHT["publication_sha256"]
        and result["result_sha256"] == contract.PREFLIGHT["result_sha256"],
        "actual preflight is not fully verified",
    )
    science.validate_preflight_audit(plan["science_plan"]["preflight"]["audit"], old, result)
    return dict(result, plan_sha256=plan_sha(old), preflight_recovery_sha256=plan_sha(previous))


def admission(plan):
    inner = plan["science_plan"]
    fence = verify_fence(plan)
    parents = science.verify_parent(inner, accounting=True)
    verify_execution(plan)
    preflight = verify_recovered_preflight(plan)
    require(
        all(
            not safe(p).exists()
            for p in (inner["scientific_claim"], inner["claim"], inner["submission_dir"], inner["result_dir"])
        ),
        "fit already claimed; reconcile, never retry",
    )
    require(
        safe(inner["python"]).is_file()
        and os.access(inner["python"], os.X_OK)
        and sha(read(inner["python"], 64 * CAP)) == recovery.contract.PYTHON_SHA256,
        "fixed executable differs",
    )
    require(
        safe(inner["hf_home"]).is_dir() and PROJECT.is_dir() and os.access(PROJECT, os.W_OK),
        "storage unavailable",
    )
    runtime = science.verify_runtime(inner)
    snapshot = science.verify_runtime_snapshot(inner)
    gpu = science.check_gpu_ceiling()
    require(
        gpu["existing_allocatable_gpus"] == 0 and gpu["new_gpus"] == 2 and gpu["limit"] == 4,
        "reserve unknown old two GPUs; other allocatable GPUs prevent replacement",
    )
    gpu = dict(gpu, unresolved_old_gpu_reservation=2, total_reserved_gpus=4)
    return dict(
        fence=fence,
        parents=parents,
        preflight=preflight,
        runtime=runtime,
        runtime_snapshot=snapshot,
        gpu_concurrency=gpu,
    )


def check_claims(plan):
    inner = plan["science_plan"]
    directory = safe(inner["submission_dir"])
    require(
        read(directory / "execution-plan.json", MAX_PLAN) == canonical(plan)
        and read(directory / "plan.json", science.MAX_PLAN) == canonical(inner)
        and read(inner["scientific_claim"]) == canonical(science.scientific_claim_record(inner)),
        "fit permanent scientific claim/plan differs",
    )
    claim = document(inner["claim"])
    require(
        set(claim) == {"intent_id", "plan_sha256", "at"}
        and claim["intent_id"] == inner["intent_id"]
        and claim["plan_sha256"] == plan["science_plan_sha256"]
        and isinstance(claim["at"], str),
        "fit execution claim differs",
    )
    recovery.check_claims(plan["preflight_recovery"])
    # Scheduler startup may precede acknowledgement: receipts are not an entry prerequisite.
    # Receipt/live-binding may be written after the scheduler has started the node.


def entry_record(plan, job, *, at, setup_elapsed_seconds):
    require(isinstance(job, str) and re.fullmatch("[1-9][0-9]*", job), "invalid node job")
    require(
        type(setup_elapsed_seconds) in (int, float)
        and math.isfinite(setup_elapsed_seconds)
        and 0 <= setup_elapsed_seconds <= 60,
        "wrapper setup exceeded original allocation reserve",
    )
    require(isinstance(at, str) and bool(at), "missing entry time")
    return dict(
        schema="quest-sdsc-name-invariant-fit-node-entry-v1",
        job_id=job,
        outer_plan_sha256=sha(canonical(plan)),
        science_plan_sha256=plan["science_plan_sha256"],
        execution_head=plan["execution"]["head"],
        preflight_job_id=contract.PREFLIGHT["job_id"],
        preflight_publication_sha256=contract.PREFLIGHT["publication_sha256"],
        at=at,
        setup_elapsed_seconds=setup_elapsed_seconds,
        pre_model_guards_passed=True,
        **dict.fromkeys(FLAGS, False),
    )


def exit_record(entry, *, returncode, node_returned, completed_at, error_type=None):
    require(type(returncode) is int and type(node_returned) is bool, "invalid node return evidence")
    require(isinstance(completed_at, str) and bool(completed_at), "missing exit time")
    require(
        (node_returned and error_type is None)
        or (not node_returned and isinstance(error_type, str) and bool(error_type) and returncode != 0),
        "exception evidence differs",
    )
    value = {k: v for k, v in entry.items() if k not in {"schema", "pre_model_guards_passed"}}
    value.update(
        schema="quest-sdsc-name-invariant-fit-node-exit-v1",
        returncode=returncode,
        node_returned=node_returned,
        completed_at=completed_at,
    )
    if error_type is not None:
        value["error_type"] = error_type
    return value


def validate_node_evidence(plan, job, entry, exit=None, *, require_success=False):
    require(
        isinstance(entry, dict)
        and canonical(entry)
        == canonical(
            entry_record(
                plan, job, at=entry.get("at"), setup_elapsed_seconds=entry.get("setup_elapsed_seconds")
            )
        ),
        "node entry differs",
    )
    if exit is not None:
        require(
            isinstance(exit, dict)
            and canonical(exit)
            == canonical(
                exit_record(
                    entry,
                    returncode=exit.get("returncode"),
                    node_returned=exit.get("node_returned"),
                    completed_at=exit.get("completed_at"),
                    error_type=exit.get("error_type"),
                )
            ),
            "node exit differs",
        )
    if require_success:
        require(
            exit is not None and exit["node_returned"] is True and exit["returncode"] == 0,
            "successful original node return evidence missing",
        )
    return dict(
        entry_verified=True,
        exit_verified=exit is not None,
        node_returned_success=exit is not None and exit["node_returned"] is True and exit["returncode"] == 0,
    )


def sbatch(plan):
    argv = science.sbatch(plan["science_plan"])
    argv[-1] = sha(canonical(plan))
    return argv


def job_script(plan):
    inner = plan["science_plan"]
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
                str(Path(plan["source"]["release"]) / "source" / NODE),
                plan["execution"]["control_file_sha256"][NODE],
                str(Path(inner["submission_dir"]) / "execution-plan.json"),
            ]
        )
        + ' "$1"\n'
    ).encode()


def observation_streams(value, argv, *, timeout, utc_query=False):
    require(
        isinstance(value, dict)
        and value.get("schema") == "opd-observed-command-v1"
        and value.get("argv") == argv
        and value.get("timeout_seconds") == timeout
        and value.get("max_output_bytes") == CAP
        and value.get("stdin") == "DEVNULL"
        and value.get("removed_environment_prefixes") == ["SBATCH_", "SQUEUE_", "SACCT_"]
        and value.get("utc_query") is utc_query
        and value.get("retry_attempted") is False,
        "command observation identity differs",
    )
    require(
        type(value.get("pid")) is int
        and value["pid"] > 0
        and type(value.get("elapsed_seconds")) in (int, float)
        and math.isfinite(value["elapsed_seconds"])
        and value["elapsed_seconds"] >= 0,
        "command process/timing evidence differs",
    )
    for key in ("started_at", "ended_at"):
        stamp = value.get(key)
        require(isinstance(stamp, str), "command UTC timestamp missing")
        parsed = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        require(parsed.utcoffset() == dt.timedelta(0), "command timestamp is not UTC")
    streams = {}
    for name in ("stdout", "stderr"):
        row = value[name]
        require(
            isinstance(row, dict)
            and set(row)
            == {
                "base64",
                "retained_bytes",
                "retained_sha256",
                "observed_bytes",
                "observed_sha256",
                "eof",
                "truncated",
            },
            "stream fields differ",
        )
        raw = base64.b64decode(row["base64"], validate=True)
        require(
            type(row["retained_bytes"]) is int
            and len(raw) == row["retained_bytes"]
            and sha(raw) == row["retained_sha256"]
            and type(row["observed_bytes"]) is int
            and row["observed_bytes"] >= len(raw),
            "retained stream identity differs",
        )
        streams[name] = raw
    require(sum(map(len, streams.values())) <= CAP, "observed streams exceed bound")
    return streams


def completed_command(value, argv, *, timeout, utc_query=False):
    streams = observation_streams(value, argv, timeout=timeout, utc_query=utc_query)
    require(
        value.get("spawned") is True
        and value.get("child_reaped") is True
        and value.get("timed_out") is False
        and value.get("output_limited") is False
        and value.get("termination_requested") is False
        and value.get("error") is None
        and value.get("output_complete") is True
        and type(value.get("returncode")) is int
        and value["returncode"] == 0,
        "incomplete/failed command; acknowledgement unknown, never retry",
    )
    for name, raw in streams.items():
        row = value[name]
        require(
            row["eof"] is True
            and row["truncated"] is False
            and row["observed_bytes"] == len(raw)
            and row["observed_sha256"] == sha(raw),
            "complete observation stream differs",
        )
    return {k: v.decode("utf8", errors="strict") for k, v in streams.items()}


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
    directory = safe(inner["submission_dir"])
    if (directory / "receipt.json").exists():
        receipt = document(directory / "receipt.json")
        science.validate_submission_receipt(receipt, inner)
        return record_receipt(plan, receipt["job_id"], receipt["reconciled"])
    # A complete durable sbatch acknowledgement can precede a lost receipt write.
    # Partial/timeout output is never an acknowledgement, even if it contains a number.
    observation_path = directory / "submission-observation.json"
    if observation_path.exists():
        value = document(observation_path, limit=2 * CAP)
        try:
            streams = completed_command(value, sbatch(plan), timeout=OBSERVED_TIMEOUT)
            match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\s*", streams["stdout"])
        except (ValueError, KeyError, TypeError, UnicodeError):
            match = None
        if match is not None and match.group(1) != "54690409":
            job = match.group(1)
            live = science.live_binding(inner, job)
            receipt = record_receipt(plan, job, True)
            if not (directory / "live-binding.json").exists():
                write_once(directory / "live-binding.json", canonical(live))
            return receipt
    user = pwd.getpwuid(os.getuid()).pw_name
    commands = [
        ["squeue", "--noheader", "--user=" + user, "--name=" + inner["job_name"], "--format=%i|%j|%k"],
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--user=" + user,
            "--name=" + inner["job_name"],
            "--starttime=" + inner["created_at"][:10],
            "--format=JobIDRaw,JobName%100,Comment%100,User%64",
        ],
    ]
    outputs = []
    for argv in commands:
        value = bound_helper(plan, "sdsc_observed_command").run_observed(
            argv, timeout=45, max_output_bytes=CAP, utc_query=True
        )
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        write_once(directory / ("reconcile-" + argv[0] + "-" + stamp + ".json"), canonical(value))
        outputs.append(completed_command(value, argv, timeout=45, utc_query=True)["stdout"])
    ids = set()
    for line in "\n".join(outputs).splitlines():
        fields = line.split("|")
        if fields and fields[-1] == "":
            fields.pop()
        if (
            len(fields) in (3, 4)
            and fields[1:3] == [inner["job_name"], inner["intent_id"]]
            and re.fullmatch("[1-9][0-9]*", fields[0])
            and (len(fields) == 3 or fields[3] == user)
        ):
            ids.add(fields[0])
    require(len(ids) == 1 and "54690409" not in ids, "zero/ambiguous acknowledgement; never resubmit")
    job = ids.pop()
    live = science.live_binding(inner, job)
    receipt = record_receipt(plan, job, True)
    if not (directory / "live-binding.json").exists():
        write_once(directory / "live-binding.json", canonical(live))
    return receipt


def inspect_job(plan, receipt, *, accounting_snapshot=None):
    result = science.inspect_job(plan["science_plan"], receipt, accounting_snapshot=accounting_snapshot)
    directory = safe(plan["science_plan"]["submission_dir"])
    entry_path, exit_path = (directory / "fit-node-entry.json", directory / "fit-node-exit.json")
    result["execution_entry_verified"] = False
    result["execution_exit_verified"] = False
    if entry_path.exists():
        entry = document(entry_path, limit=CAP)
        exit = document(exit_path, limit=CAP) if exit_path.exists() else None
        evidence = validate_node_evidence(
            plan, receipt["job_id"], entry, exit, require_success=result["success"]
        )
        result.update(
            execution_entry_verified=evidence["entry_verified"],
            execution_exit_verified=evidence["exit_verified"],
            execution_entry_sha256=sha(read(entry_path, CAP)),
        )
        if exit is not None:
            result["execution_exit_sha256"] = sha(read(exit_path, CAP))
    require(
        not result["success"] or result["execution_exit_verified"],
        "successful science lacks fit node evidence",
    )
    result.update(
        execution_plan_sha256=sha(canonical(plan)),
        preflight_job_id=contract.PREFLIGHT["job_id"],
        preflight_publication_sha256=contract.PREFLIGHT["publication_sha256"],
        old_submission_outcome="unknown",
        **dict.fromkeys(FLAGS, False),
    )
    return result


def fetch(plan, receipt, status):
    inner = plan["science_plan"]
    root, directory = safe(inner["result_dir"]), safe(inner["submission_dir"])
    files = {}
    if status["publication_verified"]:
        rows = science.publication_records(status["publication"], inner, receipt["job_id"])
        for name in (*rows, "receipt.json"):
            raw = read(root / name, science.file_limit(name))
            expected = status["publication_sha256"] if name == "receipt.json" else rows[name]["sha256"]
            require(sha(raw) == expected, "scientific fetch changed")
            files[name] = raw
    for name, path in [
        ("worker.log", root / "worker.log"),
        ("slurm.out", directory / ("slurm-" + receipt["job_id"] + ".out")),
        ("slurm.err", directory / ("slurm-" + receipt["job_id"] + ".err")),
    ]:
        if path.exists():
            with safe(path).open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                files[name] = stream.read(65536)
    for name in EXECUTION_NAMES:
        if (directory / name).exists():
            files["execution/" + name] = read(
                directory / name, MAX_PLAN if name == "execution-plan.json" else 2 * CAP
            )
    require(
        sum(len(v) for k, v in files.items() if k.startswith("execution/")) <= EXECUTION_MAX,
        "execution fetch bound exceeded",
    )
    total = sum(map(len, files.values()))
    require(total <= MAX_FETCH, "fetch bound exceeded")
    return dict(
        status=status,
        receipt=receipt,
        files={k: base64.b64encode(v).decode() for k, v in files.items()},
        execution_files={
            k: dict(size=len(v), sha256=sha(v)) for k, v in files.items() if k.startswith("execution/")
        },
        bytes=total,
    )


def validate_download(result, plan):
    require(isinstance(result["files"], dict), "missing fetch files")
    recovery, original = {}, {}
    for name, value in result["files"].items():
        if name.startswith("execution/"):
            require(name[10:] in EXECUTION_NAMES, "unknown execution fetch path")
            recovery[name] = base64.b64decode(value, validate=True)
        else:
            original[name] = value
    require(
        sum(map(len, recovery.values())) <= EXECUTION_MAX
        and all(len(v) <= MAX_PLAN for v in recovery.values()),
        "execution fetch exceeded bounds",
    )
    require(
        result.get("execution_files") == {k: dict(size=len(v), sha256=sha(v)) for k, v in recovery.items()},
        "execution fetch inventory/hash differs",
    )
    count = sum(len(base64.b64decode(v, validate=True)) for v in original.values())
    require(
        result["bytes"] == count + sum(map(len, recovery.values())) <= MAX_FETCH,
        "aggregate fetch size differs",
    )
    files = science.validate_download(dict(result, files=original, bytes=count), plan["science_plan"])
    require(
        recovery.get("execution/execution-plan.json") == canonical(plan), "fetched execution plan differs"
    )
    require(
        decode(recovery["execution/execution-receipt.json"])
        == dict(claim_record(plan), job_id=result["receipt"]["job_id"], scientific_receipt=result["receipt"]),
        "fetched execution receipt differs",
    )
    if "execution/fit-node-entry.json" in recovery:
        entry = decode(recovery["execution/fit-node-entry.json"])
        exit = (
            decode(recovery["execution/fit-node-exit.json"])
            if "execution/fit-node-exit.json" in recovery
            else None
        )
        validate_node_evidence(
            plan, result["receipt"]["job_id"], entry, exit, require_success=result["status"]["success"]
        )
    else:
        require(not result["status"]["success"], "successful fetch lacks node entry")
    return {**files, **recovery}


def remote_action(request):
    require(
        isinstance(request, dict) and set(request) == {"action", "plan", "authorize", "dry_run"},
        "request fields differ",
    )
    plan = validate_plan(request["plan"])
    verify_source(plan)
    inner = plan["science_plan"]
    directory = safe(inner["submission_dir"])
    action = request["action"]
    if action == "dry-run":
        return dict(
            dry_run=True,
            plan_sha256=sha(canonical(plan)),
            science_plan_sha256=plan["science_plan_sha256"],
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
        proof = request["dry_run"]
        require(
            request["authorize"] is True
            and isinstance(proof, dict)
            and proof.get("dry_run") is True
            and proof.get("blockers") == []
            and proof.get("plan_sha256") == sha(canonical(plan))
            and proof.get("science_plan_sha256") == plan["science_plan_sha256"]
            and proof.get("argv") == sbatch(plan),
            "authorized matching dry-run required",
        )
        checks = admission(plan)
        for path, value in [
            (inner["scientific_claim"], science.scientific_claim_record(inner)),
            (
                inner["claim"],
                dict(intent_id=inner["intent_id"], plan_sha256=plan["science_plan_sha256"], at=now()),
            ),
        ]:
            safe(path).parent.mkdir(parents=True, exist_ok=True)
            write_once(safe(path), canonical(value))
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=0o700)
        for name, raw in [
            ("plan.json", canonical(inner)),
            ("execution-plan.json", canonical(plan)),
            ("job.sh", job_script(plan)),
            ("admission.json", canonical(checks)),
            ("submission-started.json", canonical(dict(argv=sbatch(plan), at=now(), no_retry=True))),
        ]:
            write_once(directory / name, raw)
        try:
            value = bound_helper(plan, "sdsc_observed_command").run_observed(
                sbatch(plan), timeout=OBSERVED_TIMEOUT, max_output_bytes=CAP
            )
            write_once(directory / "submission-observation.json", canonical(value))
            streams = completed_command(value, sbatch(plan), timeout=OBSERVED_TIMEOUT)
            match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\s*", streams["stdout"])
            require(match is not None, "missing trustworthy acknowledgement; never retry")
            receipt = record_receipt(plan, match.group(1))
            write_once(
                directory / "live-binding.json", canonical(science.live_binding(inner, receipt["job_id"]))
            )
            return receipt
        except BaseException as error:
            write_once(
                directory / "unknown.json",
                canonical(
                    dict(
                        error_type=type(error).__name__,
                        error=str(error)[:1000],
                        observation_sha256=sha(read(directory / "submission-observation.json", 2 * CAP))
                        if (directory / "submission-observation.json").exists()
                        else None,
                        no_retry=True,
                        at=now(),
                    )
                ),
            )
            raise
    require(action in {"reconcile", "status", "fetch", "logs"}, "unknown operation")
    check_claims(plan)
    verify_fence(plan)
    verify_execution(plan)
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
    return fetch(plan, receipt, status)


def build_inner(previous, audit):
    inner = decode(canonical(previous))
    inner.update(
        mode="fit",
        resources=science.resources("fit"),
        science_identity=science.science_identity("fit", previous["protocol"]),
        created_at=now(),
        preflight=dict(plan=previous, plan_sha256=plan_sha(previous), audit=audit),
    )
    intent = science.execution_intent(inner)
    inner.update(
        intent_id=intent,
        job_name="opd-sni-" + intent,
        submission_dir=str(CONTROL / "student-name-invariant-v1-submissions" / intent),
        result_dir=str(PROJECT / "student-name-invariant-v1" / intent),
        claim=str(CONTROL / "student-name-invariant-v1-claims" / (intent + ".json")),
    )
    inner["scientific_claim"] = str(science.scientific_claim_path(inner))
    return science.validate_plan(inner)


def prepare(args, cli):
    previous = recovery.validate_plan(
        document(
            safe(args.preflight_recovery_plan.absolute()),
            contract.PREFLIGHT["recovery_plan_sha256"],
            MAX_PLAN,
        )
    )
    raw_audit = read(safe(args.preflight_audit.absolute()), science.MAX_AUDIT)
    audited = decode(raw_audit)
    require(
        raw_audit == canonical(audited) + b"\n" and sha(raw_audit) == contract.PREFLIGHT["audit_sha256"],
        "exact original independent preflight audit required",
    )
    inner = build_inner(previous["science_plan"], dict(document=audited, sha256=sha(raw_audit)))
    execution = binding(ROOT, args.execution_git_head)
    record = cli.run_record(args.run_id)
    require(record["state"] == "deployed", "matching reviewed deployed execution required")
    manifest = record["manifest"]
    rows = science.manifest_records(manifest)
    pins = {
        **contract.FROZEN_DEPENDENCIES,
        **inner["protocol"]["science_file_sha256"],
        **execution["control_file_sha256"],
        CONTRACT_PATH: execution["artifact_sha256"],
    }
    require(
        record["deployment"]["code_sha256"] == manifest["code_sha256"]
        and all(rows.get(p, {}).get("sha256") == h == sha(read(ROOT / p)) for p, h in pins.items()),
        "deployed/local fit execution pins differ",
    )
    source = dict(
        run_id=manifest["run_id"],
        code_sha256=manifest["code_sha256"],
        manifest_sha256=sha(canonical(manifest) + b"\n"),
        release=str(CONTROL / "releases" / manifest["run_id"]),
        provenance=dict(
            directory=args.provenance_dir,
            manifest_sha256=args.provenance_manifest_sha256,
            head=args.execution_git_head,
        ),
    )
    plan = dict(
        schema=SCHEMA,
        task=TASK,
        intent_id=inner["intent_id"],
        science_plan=inner,
        science_plan_sha256=plan_sha(inner),
        source=source,
        execution=execution,
        preflight_recovery=previous,
        preflight_recovery_sha256=plan_sha(previous),
        **dict.fromkeys(FLAGS, False),
    )
    validate_plan(plan)
    # Preserve the historical local science plan format for the original raw auditor.
    inner_dir = ROOT / ".sdsc/student-name-invariant-v1" / inner["intent_id"]
    directory = ROOT / ".sdsc/student-name-invariant-fit" / inner["intent_id"]
    require(not inner_dir.exists() and not directory.exists(), "fit intent already prepared; no duplicate")
    inner_dir.mkdir(mode=0o700, parents=True)
    write_once(inner_dir / "plan.json", canonical(inner))
    directory.mkdir(mode=0o700, parents=True)
    write_once(directory / "plan.json", canonical(plan))
    return dict(
        plan=str(directory / "plan.json"),
        plan_sha256=plan_sha(plan),
        science_plan=str(inner_dir / "plan.json"),
        science_plan_sha256=plan_sha(inner),
        resources=inner["resources"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "submit", "status", "fetch", "logs", "reconcile", "remote")
    )
    for name in ("preflight-recovery-plan", "preflight-audit", "plan", "dry-run-file"):
        parser.add_argument("--" + name, type=Path)
    for name in ("run-id", "provenance-dir", "provenance-manifest-sha256", "execution-git-head"):
        parser.add_argument("--" + name)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "remote":
        require(Path(__file__).resolve().is_relative_to(CONTROL / "releases"), "no local Slurm operations")
        raw = sys.stdin.buffer.read(MAX_PLAN + CAP + 1)
        require(len(raw) <= MAX_PLAN + CAP, "remote request exceeds bound")
        result = remote_action(decode(raw))
    else:
        cli = science.helper("sdsc_cli")
        if args.action == "prepare":
            require(
                all(
                    (
                        args.preflight_recovery_plan,
                        args.preflight_audit,
                        args.run_id,
                        args.provenance_dir,
                        args.provenance_manifest_sha256,
                        args.execution_git_head,
                    )
                ),
                "exact preflight audit and independent execution release/provenance required",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "plan required")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-name-invariant-fit" and path.name == "plan.json",
                "local fit execution plan path differs",
            )
            plan = validate_plan(document(path, limit=MAX_PLAN))
            require(
                path.parent.name == plan["intent_id"]
                and all(
                    sha(read(ROOT / p)) == h for p, h in plan["execution"]["control_file_sha256"].items()
                ),
                "local fit controls changed",
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
                    str(Path(plan["source"]["release"]) / "source" / CONTROLLER),
                    plan["execution"]["control_file_sha256"][CONTROLLER],
                ],
                data=canonical(dict(action=action, plan=plan, authorize=args.authorize, dry_run=proof)),
                timeout=SSH_TIMEOUT,
            )
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
            # Preserve received raw bytes before decoding or interpreting acknowledgement.
            require(
                len(response.stdout) + len(response.stderr) <= MAX_RESPONSE, "remote response bound exceeded"
            )
            for name, raw in [("stdout", response.stdout), ("stderr", response.stderr)]:
                write_once(path.parent / (action + "-" + stamp + "." + name), raw)
            require(
                response.returncode == 0,
                "remote operation failed/acknowledgement unknown; reconcile: "
                + response.stderr.decode("utf8", "replace")[-1500:],
            )
            result = decode(response.stdout)
            if action in {"fetch", "logs"}:
                files = validate_download(result, plan)
                directory = ROOT / ".sdsc/fetched" / result["receipt"]["job_id"]
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                destination = Path(tempfile.mkdtemp(prefix="fetch-", dir=directory))
                for name, raw in files.items():
                    (destination / name).parent.mkdir(parents=True, exist_ok=True)
                    write_once(destination / name, raw)
                    require(
                        read(
                            destination / name,
                            MAX_PLAN if name.startswith("execution/") else science.file_limit(name),
                        )
                        == raw,
                        "local fetch readback differs",
                    )
                result.pop("files")
                result["destination"] = str(destination)
                write_once(destination / "fetch-manifest.json", canonical(result))
            write_once(path.parent / (action + "-" + stamp + ".json"), canonical(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
