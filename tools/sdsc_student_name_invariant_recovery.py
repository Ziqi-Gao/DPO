#!/usr/bin/env python3
"""One fenced replacement of an unknown preflight submission; science is unchanged."""

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
TASK = "qwen3-v2-student-name-invariant-recovery-v1"
SCHEMA = "quest-sdsc-student-name-invariant-recovery-plan-v1"
CONTRACT_PATH = "prereg/amendments/qwen3_student_name_invariant_execution_recovery_v1.json"
CAP = 1024**2
MAX_PLAN, RECOVERY_MAX, MAX_FETCH, MAX_RESPONSE = 8 * CAP, 8 * CAP, 232 * CAP, 336 * CAP
OBSERVED_TIMEOUT, SSH_TIMEOUT = 180, 900
CONTROLLER = "tools/sdsc_student_name_invariant_recovery.py"
NODE = "tools/sdsc_student_name_invariant_recovery_job.py"
FENCE = "tools/sdsc_student_execution_fence.py"
OBSERVED = "tools/sdsc_observed_command.py"
RECOVERY_NAMES = (
    "execution-plan.json",
    "execution-receipt.json",
    "admission.json",
    "submission-started.json",
    "submission-observation.json",
    "unknown.json",
    "recovery-node-entry.json",
    "recovery-node-exit.json",
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


contract = helper("sdsc_student_name_invariant_recovery_contract")
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
        helper("sdsc_student_name_invariant_recovery_contract", root)
        .resolve_execution_contract(
            root,
            expected_head=expected_head,
            git_dir=root / (".opd-git" if (root / ".opd-git").is_dir() else ".git"),
        )
        .as_dict()
    )


def recovery_claim(plan):
    key = dict(
        old_intent=contract.OLD["intent_id"],
        old_plan_sha256=contract.OLD["plan_sha256"],
        science_identity=plan["science_plan"]["science_identity"],
    )
    return str(CONTROL / "student-name-invariant-recovery-claims" / (sha(canonical(key)) + ".json"))


def claim_record(plan):
    return dict(
        task=TASK,
        intent_id=plan["intent_id"],
        plan_sha256=sha(canonical(plan)),
        science_plan_sha256=plan["science_plan_sha256"],
        old_intent=contract.OLD["intent_id"],
        old_plan_sha256=contract.OLD["plan_sha256"],
        fence_sha256=plan["fence"]["sha256"],
        no_retry=True,
    )


def bound_helper(plan, name):
    relative_name = "tools/" + name + ".py"
    require(
        sha(read(ROOT / relative_name)) == plan["execution"]["control_file_sha256"][relative_name],
        "reviewed execution helper changed: " + name,
    )
    return helper(name)


def validate_fence(plan):
    value = plan["fence"]
    require(
        isinstance(value, dict) and set(value) == {"path", "size", "sha256", "document"},
        "fence descriptor fields differ",
    )
    fence = bound_helper(plan, "sdsc_student_execution_fence")
    request = fence.build_request(plan["old_plan"])
    fence.validate_proof(value["document"], request=request)
    raw = fence.canonical(value["document"])
    require(
        value["path"] == request["proof_path"]
        and type(value["size"]) is int
        and 0 < value["size"] == len(raw) <= CAP
        and value["sha256"] == sha(raw),
        "fence descriptor content differs",
    )
    require(
        value["document"]["helper_sha256"] == plan["execution"]["control_file_sha256"][FENCE],
        "fence helper not bound to reviewed execution",
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
            "execution",
            "old_plan",
            "fence",
            "recovery_claim",
            *FLAGS,
        },
        "recovery plan fields differ",
    )
    require(
        plan["schema"] == SCHEMA and plan["task"] == TASK and len(canonical(plan)) <= MAX_PLAN,
        "recovery scope/bound differs",
    )
    inner, old = science.validate_plan(plan["science_plan"]), science.validate_plan(plan["old_plan"])
    require(
        sha(canonical(old)) == contract.OLD["plan_sha256"]
        and old["intent_id"] == contract.OLD["intent_id"]
        and old["control_sha256"]["tools/sdsc_student_name_invariant_job.py"] == contract.OLD["node_sha256"],
        "not the one pinned unknown original plan",
    )
    require(
        inner["mode"] == old["mode"] == "preflight" and set(inner) == set(old),
        "only unchanged preflight may recover",
    )
    allowed = {
        "created_at",
        "run_id",
        "code_sha256",
        "manifest_sha256",
        "release",
        "provenance",
        "intent_id",
        "job_name",
        "submission_dir",
        "result_dir",
        "claim",
        "protocol",
    }
    require(
        all(canonical(inner[k]) == canonical(old[k]) for k in old if k not in allowed),
        "original science, input, runtime or resources changed",
    )
    require(
        set(inner["protocol"]) == set(old["protocol"])
        and all(inner["protocol"][k] == old["protocol"][k] for k in old["protocol"] if k != "head"),
        "original accepted science changed",
    )
    require(
        inner["intent_id"] == plan["intent_id"] != old["intent_id"]
        and plan["science_plan_sha256"] == sha(canonical(inner))
        and inner["control_sha256"] == contract.FROZEN_DEPENDENCIES,
        "fresh execution or frozen control binding differs",
    )
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
        execution["science_binding"] == inner["protocol"]
        and execution["head"] == inner["provenance"]["head"],
        "execution/science lineage differs",
    )
    require(
        all(
            isinstance(execution[k], str) and re.fullmatch("[a-f0-9]{64}", execution[k])
            for k in ("core_sha256", "artifact_sha256")
        ),
        "invalid execution hash",
    )
    require(
        all(
            isinstance(execution[k], str) and re.fullmatch("[a-f0-9]{40}", execution[k])
            for k in ("implementation_commit", "acceptance_commit", "head")
        )
        and execution["implementation_commit"] != execution["acceptance_commit"],
        "distinct accepted execution review required",
    )
    require(
        isinstance(execution["control_file_sha256"], dict)
        and set(execution["control_file_sha256"]) == set(NEW_TOOLS)
        and all(
            isinstance(h, str) and re.fullmatch("[a-f0-9]{64}", h)
            for h in execution["control_file_sha256"].values()
        ),
        "execution control pins differ",
    )
    validate_fence(plan)
    require(
        plan["recovery_claim"] == recovery_claim(plan) and all(plan[k] is False for k in FLAGS),
        "recovery claim/scope differs",
    )
    return plan


def verify_source(plan, root=None):
    manifest = science.verify_source(plan["science_plan"], root)
    rows = {r["path"]: r for r in manifest["files"]}
    pins = {**plan["execution"]["control_file_sha256"], CONTRACT_PATH: plan["execution"]["artifact_sha256"]}
    require(
        all(rows.get(p, {}).get("sha256") == h for p, h in pins.items()), "deployed recovery controls differ"
    )
    return manifest


def verify_execution(plan, destination=None):
    if destination is None:
        with tempfile.TemporaryDirectory(prefix="name-invariant-recovery-verify-") as temp:
            return verify_execution(plan, Path(temp) / "science")
    result = science.verify_science(plan["science_plan"], destination)
    require(
        binding(destination, plan["execution"]["head"]) == plan["execution"],
        "restored accepted execution differs",
    )
    return result


def verify_fence(plan):
    value = validate_fence(plan)
    fence = bound_helper(plan, "sdsc_student_execution_fence")
    require(read(value["path"], CAP) == fence.canonical(value["document"]), "durable fence bytes differ")
    require(fence.verify(value["document"]) == value["document"], "durable fence verification differs")
    old = plan["old_plan"]
    require(
        document(old["scientific_claim"]) == science.scientific_claim_record(old),
        "original scientific claim changed",
    )
    return dict(path=value["path"], sha256=value["sha256"], verified=True, old_submission_outcome="unknown")


def admission(plan):
    inner = plan["science_plan"]
    fence = verify_fence(plan)
    parents = science.verify_parent(inner, accounting=True)
    verify_execution(plan)
    preflight = science.verify_preflight(inner)
    require(
        all(
            not safe(p).exists()
            for p in (plan["recovery_claim"], inner["claim"], inner["submission_dir"], inner["result_dir"])
        ),
        "recovery already claimed; reconcile, never retry",
    )
    require(
        safe(inner["python"]).is_file()
        and os.access(inner["python"], os.X_OK)
        and sha(read(inner["python"], 64 * CAP)) == contract.PYTHON_SHA256,
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
    inner, old = plan["science_plan"], plan["old_plan"]
    directory = safe(inner["submission_dir"])
    require(
        read(directory / "execution-plan.json", MAX_PLAN) == canonical(plan)
        and read(directory / "plan.json", science.MAX_PLAN) == canonical(inner)
        and read(plan["recovery_claim"]) == canonical(claim_record(plan)),
        "permanent recovery claims differ",
    )
    claim = document(inner["claim"])
    require(
        set(claim) == {"intent_id", "plan_sha256", "at"}
        and claim["intent_id"] == inner["intent_id"]
        and claim["plan_sha256"] == plan["science_plan_sha256"]
        and isinstance(claim["at"], str),
        "inner execution claim differs",
    )
    require(
        document(old["scientific_claim"]) == science.scientific_claim_record(old),
        "old scientific claim changed",
    )
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
        schema="quest-sdsc-name-invariant-recovery-node-entry-v1",
        job_id=job,
        outer_plan_sha256=sha(canonical(plan)),
        science_plan_sha256=plan["science_plan_sha256"],
        execution_head=plan["execution"]["head"],
        fence_sha256=plan["fence"]["sha256"],
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
        schema="quest-sdsc-name-invariant-recovery-node-exit-v1",
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
                str(Path(inner["release"]) / "source" / NODE),
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
    entry_path, exit_path = (directory / "recovery-node-entry.json", directory / "recovery-node-exit.json")
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
        "successful science lacks recovery node evidence",
    )
    result.update(
        recovery_plan_sha256=sha(canonical(plan)),
        fence_sha256=plan["fence"]["sha256"],
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
    for name in RECOVERY_NAMES:
        if (directory / name).exists():
            files["recovery/" + name] = read(
                directory / name, MAX_PLAN if name == "execution-plan.json" else 2 * CAP
            )
    require(
        sum(len(v) for k, v in files.items() if k.startswith("recovery/")) <= RECOVERY_MAX,
        "recovery fetch bound exceeded",
    )
    total = sum(map(len, files.values()))
    require(total <= MAX_FETCH, "fetch bound exceeded")
    return dict(
        status=status,
        receipt=receipt,
        files={k: base64.b64encode(v).decode() for k, v in files.items()},
        recovery_files={
            k: dict(size=len(v), sha256=sha(v)) for k, v in files.items() if k.startswith("recovery/")
        },
        bytes=total,
    )


def validate_download(result, plan):
    require(isinstance(result["files"], dict), "missing fetch files")
    recovery, original = {}, {}
    for name, value in result["files"].items():
        if name.startswith("recovery/"):
            require(name[9:] in RECOVERY_NAMES, "unknown recovery fetch path")
            recovery[name] = base64.b64decode(value, validate=True)
        else:
            original[name] = value
    require(
        sum(map(len, recovery.values())) <= RECOVERY_MAX
        and all(len(v) <= MAX_PLAN for v in recovery.values()),
        "recovery fetch exceeded bounds",
    )
    require(
        result.get("recovery_files") == {k: dict(size=len(v), sha256=sha(v)) for k, v in recovery.items()},
        "recovery fetch inventory/hash differs",
    )
    count = sum(len(base64.b64decode(v, validate=True)) for v in original.values())
    require(
        result["bytes"] == count + sum(map(len, recovery.values())) <= MAX_FETCH,
        "aggregate fetch size differs",
    )
    files = science.validate_download(dict(result, files=original, bytes=count), plan["science_plan"])
    require(recovery.get("recovery/execution-plan.json") == canonical(plan), "fetched execution plan differs")
    require(
        decode(recovery["recovery/execution-receipt.json"])
        == dict(claim_record(plan), job_id=result["receipt"]["job_id"], scientific_receipt=result["receipt"]),
        "fetched execution receipt differs",
    )
    if "recovery/recovery-node-entry.json" in recovery:
        entry = decode(recovery["recovery/recovery-node-entry.json"])
        exit = (
            decode(recovery["recovery/recovery-node-exit.json"])
            if "recovery/recovery-node-exit.json" in recovery
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
            (plan["recovery_claim"], claim_record(plan)),
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


def prepare(args, cli):
    inner = science.validate_plan(document(safe(args.science_plan.absolute()), limit=science.MAX_PLAN))
    execution = binding(ROOT, inner["provenance"]["head"])
    record = cli.run_record(inner["run_id"])
    require(
        record["state"] == "deployed" and record["manifest"]["code_sha256"] == inner["code_sha256"],
        "matching synced deployment required",
    )
    rows = {r["path"]: r for r in record["manifest"]["files"]}
    pins = {**execution["control_file_sha256"], CONTRACT_PATH: execution["artifact_sha256"]}
    require(
        all(rows.get(p, {}).get("sha256") == h == sha(read(ROOT / p)) for p, h in pins.items()),
        "deployed/local recovery controls differ",
    )
    old = document(
        ROOT / ".sdsc/student-name-invariant-v1" / contract.OLD["intent_id"] / "plan.json",
        contract.OLD["plan_sha256"],
    )
    raw = read(safe(args.fence_proof_file.absolute()), CAP)
    proof = decode(raw)
    fence = helper("sdsc_student_execution_fence")
    require(raw == fence.canonical(proof), "fence proof must contain exact canonical durable bytes")
    plan = dict(
        schema=SCHEMA,
        task=TASK,
        intent_id=inner["intent_id"],
        science_plan=inner,
        science_plan_sha256=sha(canonical(inner)),
        execution=execution,
        old_plan=old,
        fence=dict(
            path=fence.build_request(old)["proof_path"], size=len(raw), sha256=sha(raw), document=proof
        ),
        **dict.fromkeys(FLAGS, False),
    )
    plan["recovery_claim"] = recovery_claim(plan)
    validate_plan(plan)
    directory = ROOT / ".sdsc/student-name-invariant-recovery" / plan["intent_id"]
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_once(directory / "plan.json", canonical(plan))
    return dict(
        plan=str(directory / "plan.json"),
        plan_sha256=sha(canonical(plan)),
        science_plan_sha256=plan["science_plan_sha256"],
        resources=inner["resources"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "submit", "status", "fetch", "logs", "reconcile", "remote")
    )
    for name in ("science-plan", "fence-proof-file", "plan", "dry-run-file"):
        parser.add_argument("--" + name, type=Path)
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
                args.science_plan and args.fence_proof_file,
                "fresh science plan and actual durable fence proof required",
            )
            result = prepare(args, cli)
        else:
            require(args.plan is not None, "plan required")
            path = safe(args.plan.absolute())
            require(
                path.parent.parent == ROOT / ".sdsc/student-name-invariant-recovery"
                and path.name == "plan.json",
                "local recovery plan path differs",
            )
            plan = validate_plan(document(path, limit=MAX_PLAN))
            require(
                path.parent.name == plan["intent_id"]
                and all(
                    sha(read(ROOT / p)) == h for p, h in plan["execution"]["control_file_sha256"].items()
                ),
                "local recovery controls changed",
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
                    str(Path(inner["release"]) / "source" / CONTROLLER),
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
                            MAX_PLAN if name.startswith("recovery/") else science.file_limit(name),
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
