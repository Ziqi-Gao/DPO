"""Finite, read-only Quest watcher for one already submitted teacher diagnostic.

This process only queries status, reads bounded logs, and fetches small results.
It never submits, cancels, retries a job, accepts science, or starts training.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = "qwen3-v2-teacher-prompt-probe"
CAPABILITY_TASK = "qwen3-v2-teacher-capability-probe"
SCHEMA = "quest-sdsc-readonly-probe-watch-v1"
POLL_SECONDS = 300
MAX_SECONDS = 14 * 24 * 3600
CONTROL_FILES = (
    "tools/sdsc_probe_watch.py",
    "tools/sdsc_probe_watch",
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
)
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING", "REQUEUED"}
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}
READINESS = {
    "accepted_science": False,
    "full_teacher_ready": False,
    "g0_passed": False,
    "training_started": False,
    "readiness": False,
    "readiness_artifact_produced": False,
}
REPORT = "teacher-prompt-probe.json"
REPORTS = {TASK: REPORT, CAPABILITY_TASK: "teacher-capability-probe.json"}
CAPABILITY_CHECKS = {
    "answer_accuracy", "exact_proof_accuracy", "first_rule_top1_accuracy",
    "intermediate_top1_accuracy", "topk_mass", "topk_target_coverage",
    "corrupted_prefix_recovery", "causal_shift",
}
CAPABILITY_METRICS = {
    "answer_accuracy", "exact_proof_accuracy", "format_validity", "first_rule_top1_accuracy",
    "intermediate_top1_accuracy", "minimum_topk_mass", "topk_target_coverage",
    "corrupted_prefix_recovery_accuracy", "minimum_causal_shift_logprob",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_path(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "Parent traversal in control path")
    require(not any(part.is_symlink() for part in (path, *path.parents)), "Symlink control path")
    return path


def read_bytes(path, maximum=1024 * 1024):
    path = safe_path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "Invalid control file")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require(
        len(raw) == before.st_size <= maximum
        and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "Control file changed during read",
    )
    return raw


def read_json(path, maximum=1024 * 1024):
    return json.loads(read_bytes(path, maximum))


def atomic(path, value):
    path = safe_path(path)
    temporary = safe_path(path.with_name("." + path.name + ".tmp"))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def controls(root=ROOT):
    return {name: digest(read_bytes(root / name)) for name in CONTROL_FILES}


def job_binding(job_id, root=ROOT):
    require(isinstance(job_id, str) and re.fullmatch(r"[1-9][0-9]*", job_id), "Invalid job ID")
    matches = []
    for path in (root / ".sdsc/submissions").glob("*.json"):
        raw = read_bytes(path)
        value = json.loads(raw)
        if str(value.get("receipt", {}).get("job_id", "")) == job_id:
            matches.append((path, raw, value))
    require(len(matches) == 1, "Unknown or ambiguous submission; reconcile, never resubmit")
    path, raw, value = matches[0]
    receipt, request = value["receipt"], value.get("request", {})
    require(value.get("state") == "submitted" and receipt.get("ok") is True, "Submission is unresolved")
    require(re.fullmatch(r"[a-f0-9]{32}", path.stem), "Invalid intent filename")
    require(receipt.get("intent_id") == path.stem, "Intent filename/receipt mismatch")
    require(receipt.get("task") in REPORTS, "Watcher only supports an existing teacher diagnostic probe")
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", receipt.get("run_id", ""))
        and re.fullmatch(r"[a-f0-9]{64}", receipt.get("code_sha256", "")),
        "Invalid deployed identity",
    )
    for key in ("intent_id", "run_id", "code_sha256", "task", "resources", "python", "hf_home"):
        require(receipt.get(key) == request.get(key), "Submission request/receipt mismatch: " + key)
    run_path = root / ".sdsc/runs" / (receipt["run_id"] + ".json")
    run_raw = read_bytes(run_path, 4 * 1024 * 1024)
    run = json.loads(run_raw)
    require(run.get("state") == "deployed", "Source snapshot is not deployed")
    manifest = run.get("manifest", {})
    require(
        manifest.get("code_sha256") == receipt["code_sha256"]
        and digest(canonical(manifest.get("files"))) == receipt["code_sha256"],
        "Deployed manifest content hash differs",
    )
    deployment = run.get("deployment", {})
    require(
        deployment.get("ok") is True
        and deployment.get("run_id") == receipt["run_id"]
        and deployment.get("code_sha256") == receipt["code_sha256"],
        "Missing matching deployment receipt",
    )
    return {
        "receipt": receipt,
        "submission_record": str(path.relative_to(root)),
        "submission_record_sha256": digest(raw),
        "run_record": str(run_path.relative_to(root)),
        "run_record_sha256": digest(run_raw),
    }


def validate_plan(plan, root=ROOT):
    require(plan.get("schema") == SCHEMA, "Wrong read-only watcher schema")
    require(plan.get("quest_root") == str(root), "Wrong Quest checkout")
    require(plan.get("quest_host") == socket.gethostname().split(".")[0], "SSH master is on another host")
    require(plan.get("control_sha256") == controls(root), "Pinned control files changed; stop for review")
    require(plan.get("poll_seconds") == POLL_SECONDS, "Watcher cadence differs")
    require(
        type(plan.get("created_at_unix")) in (float, int)
        and type(plan.get("expires_at_unix")) in (float, int)
        and plan["expires_at_unix"] - plan["created_at_unix"] == MAX_SECONDS,
        "Watcher must expire within fourteen days",
    )
    require(plan.get("binding") == job_binding(plan.get("job_id"), root), "Pinned receipt/deployment changed")
    require(plan.get("allowed_operations") == ["status", "logs", "fetch"], "Watcher scope changed")
    return plan["binding"]["receipt"]


def prepare(job_id, flow_name, root=ROOT, now=None):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", flow_name), "Invalid flow directory name")
    require(flow_name not in {".", ".."}, "Invalid flow directory name")
    created = time.time() if now is None else now
    plan = {
        "schema": SCHEMA,
        "quest_root": str(root),
        "quest_host": socket.gethostname().split(".")[0],
        "job_id": job_id,
        "binding": job_binding(job_id, root),
        "control_sha256": controls(root),
        "allowed_operations": ["status", "logs", "fetch"],
        "poll_seconds": POLL_SECONDS,
        "created_at_unix": created,
        "expires_at_unix": created + MAX_SECONDS,
    }
    validate_plan(plan, root)
    parent = safe_path(root / ".sdsc/supervision")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = safe_path(parent / flow_name)
    directory.mkdir(mode=0o700, exist_ok=False)
    atomic(directory / "plan.json", plan)
    return directory / "plan.json", plan


def command(arguments, root=ROOT):
    require(
        (len(arguments) == 2 and arguments[0] in {"status", "fetch"})
        or (len(arguments) == 4 and arguments[0] == "logs" and arguments[2:] == ["--lines", "80"]),
        "Watcher refuses any mutating or unbounded command",
    )
    require(re.fullmatch(r"[1-9][0-9]*", arguments[1]), "Invalid observed job ID")
    # Each CLI call checks the runtime-derived ControlPath with ssh -O check
    # before a BatchMode remote call. It never falls back to authentication.
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(root / "tools/sdsc_cli.py"), *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=420,
    )
    require(len(result.stdout) <= 4 * 1024 * 1024, "Control response exceeds read budget")
    if result.returncode:
        raise RuntimeError("Read-only control operation stopped: " + result.stderr[-2000:])
    response = json.loads(result.stdout)
    require(isinstance(response, dict), "Control response is not an object")
    return response


def arm_summary(arm, expected_attempts):
    require(isinstance(arm, dict), "Missing completed diagnostic arm")
    for key in ("attempt_count", "accepted_count", "total_prompts", "prompts_with_success"):
        require(type(arm.get(key)) is int and arm[key] >= 0, "Invalid diagnostic counts")
    require(
        arm["attempt_count"] == expected_attempts
        and arm["total_prompts"] == 32
        and arm["prompts_with_success"] <= min(32, arm["accepted_count"])
        and arm["accepted_count"] <= expected_attempts,
        "Incomplete diagnostic population/counts",
    )
    return {
        key: arm[key] for key in ("attempt_count", "accepted_count", "total_prompts", "prompts_with_success")
    }


def accounting_complete(status, job_id):
    accounting, queue = status.get("accounting", {}), status.get("queue", {})
    rows = [line.split("|") for line in accounting.get("stdout", "").splitlines() if line.strip()]
    jobs = [row for row in rows if len(row) >= 3 and row[0] == job_id]
    steps = [row for row in rows if row[0].startswith(job_id + ".")]
    return (
        accounting.get("returncode") == queue.get("returncode") == 0
        and not queue.get("stdout", "").strip()
        and len(jobs) == 1
        and jobs[0][1:3] == ["COMPLETED", "0:0"]
        and all(len(row) >= 3 and row[1:3] == ["COMPLETED", "0:0"] for row in steps)
    )


def capability_summary(report):
    """Validate completed diagnostic measurements without accepting readiness."""
    ordered = report.get("ordered_prompt_ids")
    rows = report.get("full_generation_rows")
    require(
        isinstance(ordered, list) and len(ordered) == 128
        and all(isinstance(value, str) and value for value in ordered)
        and len(set(ordered)) == 128
        and isinstance(rows, list) and len(rows) == 128
        and all(isinstance(row, dict) for row in rows)
        and [row.get("example_id") for row in rows] == ordered,
        "Incomplete or reordered capability generation population",
    )
    for row in rows:
        require(
            all(type(row.get(key)) is bool
                for key in ("format_valid", "answer_correct", "exact_proof_correct")),
            "Invalid capability generation row",
        )
    require(
        type(report.get("prefix_score_count")) is int and report["prefix_score_count"] > 0,
        "Missing completed capability prefix scores",
    )
    metrics, checks = report.get("metrics"), report.get("checks")
    require(
        isinstance(metrics, dict) and set(metrics) == CAPABILITY_METRICS
        and all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values()),
        "Capability metrics must be complete finite numbers",
    )
    require(
        # Float32 exp(log_softmax).topk(128).sum can exceed one by an ULP.
        # Preserve that raw evidence; only this summed mass gets roundoff room.
        all(0 <= value <= (1 + 1e-6 if key == "minimum_topk_mass" else 1)
            for key, value in metrics.items() if key != "minimum_causal_shift_logprob"),
        "Capability probability metrics are out of range",
    )
    for metric, field in (
        ("answer_accuracy", "answer_correct"),
        ("exact_proof_accuracy", "exact_proof_correct"),
        ("format_validity", "format_valid"),
    ):
        require(metrics[metric] == sum(row[field] for row in rows) / 128, "Capability row/metric mismatch")
    require(
        isinstance(checks, dict) and set(checks) == CAPABILITY_CHECKS
        and all(type(value) is bool for value in checks.values())
        and type(report.get("metrics_passed")) is bool
        and report["metrics_passed"] == all(checks.values()),
        "Capability metrics_passed differs from the eight diagnostic checks",
    )
    return {
        "metrics": metrics, "checks": checks, "metrics_passed": report["metrics_passed"],
        "generation_count": len(rows), "prefix_score_count": report["prefix_score_count"],
        "ordered_prompt_ids": ordered,
    }


def conclusion(plan, status, fetched, root=ROOT):
    receipt = plan["binding"]["receipt"]
    require(receipt.get("task") in REPORTS, "Unsupported diagnostic receipt task")
    report_name = REPORTS[receipt["task"]]
    require(
        status.get("job_id") == plan["job_id"]
        and status.get("run_id") == receipt["run_id"]
        and status.get("task") == receipt["task"],
        "Remote status run/task/job identity differs",
    )
    require(
        fetched.get("job_id") == plan["job_id"] and fetched.get("intent_id") == receipt["intent_id"],
        "Fetch identity differs",
    )
    directory = safe_path(Path(fetched["destination"]))
    require(directory.parent == root / ".sdsc/fetched" / plan["job_id"], "Fetch is outside its job directory")
    require(directory.name.startswith("fetch-"), "Invalid bounded fetch destination")
    require(
        type(fetched.get("bytes")) is int and 0 <= fetched["bytes"] <= 8 * 1024 * 1024,
        "Fetch exceeds bounded result budget",
    )
    result = {
        "job_id": plan["job_id"],
        "run_id": receipt["run_id"],
        "code_sha256": receipt["code_sha256"],
        "task": receipt["task"],
        "observed_state": status["state"],
        "diagnostic_completed": False,
        "report_available": False,
        "fetch_directory": str(directory),
        **READINESS,
        "next_step": "Independent scientific review is required; this watcher starts no further work.",
        "comparison_note": (
            "Compare baseline with paired_candidate_zero; eight-sample coverage is separate."
            if receipt["task"] == TASK else
            "Frozen validation generation and prefix metrics are diagnostic only; "
            "no readiness artifact is produced."
        ),
    }
    if not (directory / report_name).exists():
        require(status.get("success") is not True, "Verified remote report is missing from the fetch")
        return result
    raw = read_bytes(directory / report_name)
    report = json.loads(raw)
    published = read_json(directory / "receipt.json")
    for value in (report, published):
        for key in ("job_id", "run_id", "code_sha256", "task", "hf_home"):
            require(value.get(key) == receipt.get(key), "Fetched report/publication identity differs: " + key)
    matches = [entry for entry in published.get("files", []) if entry.get("path") == report_name]
    require(
        len(matches) == 1 and matches[0].get("sha256") == digest(raw) and matches[0].get("size") == len(raw),
        "Fetched report differs from publication hash",
    )
    require(
        report.get("exploratory") is True
        and report.get("accepted_science") is False
        and report.get("full_teacher_ready") is False
        and report.get("g0_passed") is False
        and report.get("execution_class_certified") is False,
        "Diagnostic report claims scientific readiness",
    )
    if receipt["task"] == CAPABILITY_TASK:
        require(
            report.get("artifact_kind") == "teacher_capability_diagnostic"
            and all(report.get(key) is False for key in READINESS)
            and report.get("resumable") is False
            and type(report.get("metrics_passed")) is bool,
            "Capability diagnostic cannot publish or claim formal readiness",
        )
    result.update(report_available=True, report_sha256=digest(raw), worker_error=report.get("error"))
    if report.get("passed") is True:
        if receipt["task"] == CAPABILITY_TASK:
            result.update(capability_summary(report))
        else:
            result.update(
                baseline=arm_summary(report.get("arms", {}).get("baseline"), 32),
                candidate=arm_summary(report.get("arms", {}).get("candidate"), 256),
                paired_candidate_zero=arm_summary(report.get("paired_candidate_zero"), 32),
            )
    if status.get("success") is True:
        require(
            status.get("state") == "COMPLETED"
            and status.get("result", {}).get("verified") is True
            and accounting_complete(status, plan["job_id"])
            and status["result"].get("result_sha256") == digest(raw)
            and report.get("passed") is True
            and report.get("exit_code") == 0
            and published.get("passed") is True
            and published.get("persisted") is True
            and published.get("persistent_read_back_verified") is True,
            "Incomplete diagnostic execution/publication proof",
        )
        result["diagnostic_completed"] = True
    return result


def run_flow(plan, write, call=command, guard=None, conclude=conclusion, clock=time.time, sleep=time.sleep):
    guard = guard or (lambda: validate_plan(plan))
    state = {
        "schema": SCHEMA,
        "plan_sha256": digest(canonical(plan)),
        "job_id": plan["job_id"],
        "quest_host": plan["quest_host"],
        "pid": os.getpid(),
        "phase": "starting",
        **READINESS,
        "no_submission": True,
        "no_cancellation": True,
        "no_retry": True,
    }

    def record(**changes):
        state.update(changes, updated_at_unix=clock())
        write("state.json", dict(state))

    def boundary():
        guard()
        require(clock() < plan["expires_at_unix"], "Fourteen-day watcher deadline reached; job untouched")

    def checked_call(arguments):
        boundary()
        result = call(arguments)
        boundary()
        require(result.get("job_id") == plan["job_id"], "Remote observation job differs")
        return result

    record()
    try:
        while True:
            status = checked_call(["status", plan["job_id"]])
            receipt = plan["binding"]["receipt"]
            require(
                status.get("run_id") == receipt["run_id"] and status.get("task") == receipt["task"],
                "Remote status run/task identity differs",
            )
            observed = status.get("state", "UNKNOWN")
            normalized = observed.split("+", 1)[0].split(" ", 1)[0]
            record(phase="waiting", observed_status=status)
            require(normalized in ACTIVE | TERMINAL, "Unknown job state; reconcile the existing job manually")
            require(
                status.get("queue", {}).get("returncode") == 0
                and status.get("accounting", {}).get("returncode") == 0,
                "Scheduler observation failed; stop without retry",
            )
            # Accounting can report a final state while Slurm is still finishing
            # its allocation cleanup. Collect once both observations are final.
            if normalized in TERMINAL and not status["queue"].get("stdout", "").strip():
                record(phase="collecting_terminal_results")
                write("terminal-status.json", status)
                logs = checked_call(["logs", plan["job_id"], "--lines", "80"])
                write("logs.json", logs)
                fetched = checked_call(["fetch", plan["job_id"]])
                record(fetch=fetched)
                boundary()
                result = conclude(plan, status, fetched)
                require(all(result.get(key) is False for key in READINESS), "Watcher cannot accept science")
                write("conclusion.json", result)
                record(phase="finished", conclusion=result)
                return 0
            remaining = plan["expires_at_unix"] - clock()
            require(remaining > 0, "Fourteen-day watcher deadline reached; job untouched")
            sleep(min(POLL_SECONDS, remaining))
    except BaseException as error:
        record(phase="stopped", error=type(error).__name__ + ": " + str(error))
        raise


def run_plan(path, root=ROOT):
    path = safe_path(path)
    require(
        path.name == "plan.json" and path.parent.parent == root / ".sdsc/supervision", "Invalid plan path"
    )
    plan_raw = read_bytes(path)
    plan = json.loads(plan_raw)
    validate_plan(plan, root)
    state_path = path.parent / "state.json"
    require(not state_path.exists(), "Watcher already started; inspect evidence, never re-arm it")
    # A per-job lock also excludes competing watchers prepared in different flow
    # directories; a terminal or interrupted flow keeps its permanent state.
    lock_path = safe_path(root / ".sdsc/supervision" / (".probe-watch-" + plan["job_id"] + ".lock"))
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Watcher lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not state_path.exists(), "Another watcher already claimed this flow")

        def guard():
            require(read_bytes(path) == plan_raw, "Immutable watcher plan changed")
            validate_plan(plan, root)

        return run_flow(
            plan,
            lambda name, value: atomic(path.parent / name, value),
            call=lambda arguments: command(arguments, root),
            guard=guard,
            conclude=lambda *args: conclusion(*args, root=root),
        )
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser(
        "prepare", help="pin an existing probe's exact receipt and controls"
    )
    prepare_command.add_argument("--job-id", required=True)
    prepare_command.add_argument("--flow", required=True)
    run_command = commands.add_parser("run", help="observe one fresh prepared flow until terminal or stopped")
    run_command.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        path, plan = prepare(args.job_id, args.flow)
        print(json.dumps({"plan": str(path), "plan_sha256": digest(canonical(plan)), "job_id": args.job_id}))
        return 0
    return run_plan(args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
