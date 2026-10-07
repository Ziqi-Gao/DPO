"""Finite Quest-side supervision of one teacher -> one calibration run.

No service is installed. All remote operations go through the existing tools/sdsc
control plane and SSH master. This supervisor never retries a submission, never
cancels a job, and never starts a factorial/pilot or claims G0 acceptance.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL_FILES = (
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    "tools/sdsc_supervise.py",
    "tools/sdsc_supervise",
    "scripts/production/slurm_supervision.sh",
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_file(path, maximum=1024 * 1024):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink control path")
    require(path.is_file() and path.stat().st_size <= maximum, "Missing/oversized control file")
    return path.read_bytes()


def atomic(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    require(not path.is_symlink() and not temporary.exists(), "Unsafe state path")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def controls(root=ROOT):
    return {name: sha(safe_file(root / name)) for name in CONTROL_FILES}


def validate_plan(plan, root=ROOT):
    require(plan.get("schema") == "quest-sdsc-calibration-supervision-v1", "Wrong supervision schema")
    require(plan.get("quest_root") == str(root), "Supervisor must stay in the actual Quest checkout")
    require(
        plan.get("quest_host") == socket.gethostname().split(".")[0],
        "SSH master belongs to a different Quest host",
    )
    require(plan.get("control_sha256") == controls(root), "Control tools changed after plan review")
    require(
        plan.get("poll_seconds") == 300 and plan.get("deadline_seconds") == 8 * 3600,
        "Use bounded five-minute supervision",
    )
    arguments = plan.get("submit_arguments")
    require(isinstance(arguments, list) and len(arguments) >= 4, "Missing exact submission parameters")
    require(
        arguments[0] == "submit" and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", arguments[1]),
        "Invalid run",
    )
    require(
        len(arguments[2:]) % 2 == 1 and arguments[-1] == "--storage-confirmed", "Unexpected command flags"
    )
    options = {}
    for key, value in zip(arguments[2:-1:2], arguments[3:-1:2], strict=True):
        require(
            isinstance(key, str) and isinstance(value, str) and key not in options, "Duplicate/invalid option"
        )
        require(not value.startswith("--") and not any(ord(c) < 32 for c in value), "Invalid option value")
        options[key] = value
    fixed = {
        "--task": "qwen3-v2-g0-calibration",
        "--account": "nwu181",
        "--partition": "nairr-gpu-shared",
        "--qos": "nairr-gpu-shared-normal",
        "--gpu-type": "h100",
        "--gpus": "2",
        "--cpus": "24",
        "--mem-gib": "192",
    }
    variable = {
        "--time",
        "--python",
        "--hf-home",
        "--result-root",
        "--provenance-dir",
        "--provenance-manifest-sha256",
        "--teacher-job-id",
        "--preflight-job-id",
    }
    require(set(options) == set(fixed) | variable, "Unexpected or missing submission options")
    require(all(options[key] == value for key, value in fixed.items()), "Calibration resources/scope changed")
    require(options["--time"] in {"00:30:00", "01:00:00", "01:30:00", "02:00:00"}, "Unreviewed walltime")
    for name in ("--teacher-job-id", "--preflight-job-id"):
        require(re.fullmatch(r"[1-9][0-9]*", options[name]), "Invalid prerequisite job ID")
    require(options["--teacher-job-id"] != options["--preflight-job-id"], "Prerequisite jobs must differ")
    for name in ("--python", "--hf-home", "--result-root"):
        path = Path(options[name])
        require(
            path.is_absolute()
            and ".." not in path.parts
            and Path("/expanse/lustre/projects/nwu181/zgao12/OPD") in path.parents,
            "Runtime/storage outside verified project",
        )
    proof = options["--provenance-manifest-sha256"]
    require(re.fullmatch(r"[a-f0-9]{64}", proof), "Invalid provenance hash")
    require(
        options["--provenance-dir"] == "/home/zgao12/quest-runs/OPD/provenance/" + proof,
        "Provenance path mismatch",
    )
    record = json.loads(safe_file(root / ".sdsc/runs" / (arguments[1] + ".json"), maximum=4 * 1024 * 1024))
    require(
        record["state"] == "deployed" and record["manifest"]["code_sha256"] == plan.get("code_sha256"),
        "Frozen release is not deployed",
    )
    require(sha(canonical(record["manifest"]["files"])) == plan["code_sha256"], "Invalid deployed manifest")
    return options


def command(arguments):
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(ROOT / "tools/sdsc_cli.py"), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=420,
    )
    if result.returncode:
        raise RuntimeError("Control operation stopped: " + result.stderr[-2000:])
    # submit emits its durable intent identity before the JSON receipt.
    output = result.stdout
    if output.startswith("submission_intent="):
        output = output.split("\n", 1)[1]
    return json.loads(output)


def run_flow(plan, write, call=command, sleep=time.sleep, clock=time.monotonic, guard=None):
    options = validate_plan(plan)
    started = clock()
    state = {
        "schema": plan["schema"],
        "plan_sha256": sha(canonical(plan)),
        "pid": os.getpid(),
        "quest_host": plan["quest_host"],
        "g0_passed": False,
        "factorial_ready": False,
        "phase": "waiting_teacher",
        "teacher_job_id": options["--teacher-job-id"],
        "preflight_job_id": options["--preflight-job-id"],
        "calibration_run_id": plan["submit_arguments"][1],
    }
    guard = guard or (lambda: validate_plan(plan))

    def record(**changes):
        state.update(changes, updated_at_unix=time.time())
        write(state)

    def boundary():
        guard()
        require(
            clock() - started < plan["deadline_seconds"],
            "Finite supervision deadline reached; jobs remain untouched",
        )

    def checked_call(arguments):
        boundary()
        return call(arguments)

    def wait_success(job, phase):
        unknown = 0
        while True:
            status = checked_call(["status", job])
            require(status.get("job_id") == job, "Status job identity mismatch")
            observed = status.get("state", "UNKNOWN")
            record(phase=phase, observed_status=status)
            if status.get("success") is True:
                require(
                    observed == "COMPLETED" and status.get("result", {}).get("verified") is True,
                    "Incomplete success proof",
                )
                return status
            if observed.split("+", 1)[0].split(" ", 1)[0] in TERMINAL:
                try:
                    record(failed_job_id=job, failure_fetch=checked_call(["fetch", job]))
                except Exception as fetch_error:
                    record(failed_job_id=job, failure_fetch_error=str(fetch_error))
                raise RuntimeError("Prerequisite/calibration did not pass: " + job + " " + observed)
            unknown = 0 if observed in ACTIVE else unknown + 1
            require(unknown <= 2, "Three inconclusive scheduler observations; stop without resubmission")
            sleep(plan["poll_seconds"])

    record()
    try:
        wait_success(options["--preflight-job-id"], "checking_preflight")
        wait_success(options["--teacher-job-id"], "waiting_teacher")
        record(phase="teacher_passed", teacher_fetch=checked_call(["fetch", options["--teacher-job-id"]]))
        # Persist the submission boundary first. Every exception after this point
        # stops; only the CLI's durable intent/reconcile operation may resolve it.
        boundary()
        record(phase="submitting_calibration", submission_attempted=True)
        receipt = checked_call([*plan["submit_arguments"], "--authorize"])
        require(receipt.get("run_id") == plan["submit_arguments"][1], "Submission receipt run differs")
        require(receipt.get("code_sha256") == plan["code_sha256"], "Submission receipt hash differs")
        require(receipt.get("task") == "qwen3-v2-g0-calibration", "Submission task differs")
        job = str(receipt.get("job_id", ""))
        require(re.fullmatch(r"[1-9][0-9]*", job), "No trustworthy real job ID; reconcile, never resubmit")
        record(phase="calibration_submitted", calibration_job_id=job, receipt=receipt)
        wait_success(job, "waiting_calibration")
        record(
            phase="calibration_complete",
            calibration_fetch=checked_call(["fetch", job]),
            next_gate="full_G0_and_independent_H100_execution_review_still_required",
        )
        return 0
    except BaseException as error:
        record(phase="stopped", error=f"{type(error).__name__}: {error}", no_retry=True, jobs_cancelled=False)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--authorize", action="store_true")
    args = parser.parse_args(argv)
    require(args.authorize, "Explicit current-session authorization is required")
    path = args.plan.absolute()
    require(
        ROOT / ".sdsc/supervision" in path.parents, "Plan must be in this checkout's supervision directory"
    )
    plan = json.loads(safe_file(path))
    validate_plan(plan)
    directory = path.parent
    require(
        not (directory / "state.json").exists(),
        "This flow was already started; inspect state and receipts, never re-arm blindly",
    )
    lock = os.open(directory / "lock", os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (directory / "state.json").exists(), "Another supervisor has claimed this flow")
        # This starts with SSH master validation even if the first job is terminal.
        spec = importlib.util.spec_from_file_location("sdsc_cli", ROOT / "tools/sdsc_cli.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        cli.require_master()
        return run_flow(plan, lambda value: atomic(directory / "state.json", value))
    finally:
        os.close(lock)


if __name__ == "__main__":
    raise SystemExit(main())
