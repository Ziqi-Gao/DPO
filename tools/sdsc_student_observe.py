"""Read-only recovery for the already submitted adapted calibration 54506821."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "student_observer_contract", ROOT / "tools/sdsc_student_supervise.py"
)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)

FLOW = "student54506821-readonly-v1"
ORIGIN = "student54506703-to-calibration-v2"
ORIGIN_SHA = "a9a6cd30dd334803aaa5ffac6bf0521fb90c5567dc1a811883558415eabc48a0"
JOB = "54506821"
RUN = "20260928T204012Z-b542b2e7e7fa-41669358"
INTENT = "bffa12a0cfa5429baa44050fcdc2833f"
SCHEMA = "quest-sdsc-student-readonly-observer-v1"
SELF = "tools/sdsc_student_observe.py"


def origin(root):
    directory = root / ".sdsc/supervision" / ORIGIN
    _, original = watch.plan_file(directory / "plan.json", ORIGIN_SHA, root)
    state = watch.document(directory / "state.json")
    watch.require(
        state.get("phase") == "stopped"
        and state.get("plan_sha256") == ORIGIN_SHA
        and state.get("submission_attempted") is True
        and state.get("submission_outcome_unknown") is False
        and state.get("calibration_job_id") == JOB
        and state.get("calibration_run_id") == RUN,
        "recovery requires the stopped flow's known submitted calibration",
    )
    binding = watch.binding(JOB, root)
    watch.require(binding == state.get("calibration_binding"), "stopped flow differs from durable receipt")
    receipt = binding["receipt"]
    watch.validate_receipt(receipt, calibration=True, plan=original)
    watch.require(receipt["run_id"] == RUN and receipt["intent_id"] == INTENT, "wrong calibration identity")
    claim_path = root / ".sdsc/student-supervision-claims/preflight-54506703.json"
    claim = watch.document(claim_path)
    watch.require(
        claim.get("plan_sha256") == ORIGIN_SHA
        and claim.get("plan") == str(directory / "plan.json")
        and claim.get("run_id") == RUN
        and claim.get("preflight_job_id") == watch.PREFLIGHT,
        "original permanent submission claim differs",
    )
    evidence = [watch.pin(directory / "state.json", root), watch.pin(claim_path, root)]
    return original, binding, evidence


def prepare(root=ROOT, now=None):
    original, binding, evidence = origin(root)
    watch.no_competitor(root, root / ".sdsc/supervision" / FLOW, before_submit=False)
    created = time.time() if now is None else now
    expiry = min(created + watch.MAX_SECONDS, original["expires_at_unix"])
    watch.require(created < expiry, "original supervision deadline already expired")
    return {
        "schema": SCHEMA,
        "flow_id": FLOW,
        "job_id": JOB,
        "quest_root": str(root),
        "quest_host": socket.gethostname().split(".")[0],
        "created_at_unix": created,
        "expires_at_unix": expiry,
        "poll_seconds": watch.POLL,
        "allowed_operations": ["status", "fetch"],
        "observer_sha256": watch.sha(watch.read(root / SELF)),
        "original_plan_sha256": ORIGIN_SHA,
        "binding": binding,
        "evidence": evidence,
    }


def validate(plan, root=ROOT):
    expected = prepare(root, plan["created_at_unix"])
    watch.require(
        watch.canonical(plan) == watch.canonical(expected), "read-only observer plan/evidence changed"
    )
    return plan


def plan_file(path, expected, root=ROOT):
    path = watch.safe(path)
    watch.require(
        path == root / ".sdsc/supervision" / FLOW / "plan.json"
        and watch.SHA.fullmatch(expected or "")
        and watch.sha(watch.read(path)) == expected,
        "observer plan path or external hash differs",
    )
    return path, validate(watch.document(path), root)


def command(arguments, plan, root=ROOT):
    watch.require(
        arguments in (["status", JOB], ["fetch", JOB]), "observer permits only fixed-job status/fetch"
    )
    # This allowlist is checked before the broader original CLI helper is called.
    original = watch.document(root / ".sdsc/supervision" / ORIGIN / "plan.json")
    return watch.command(arguments, original, root)


def observe(plan, write, *, root=ROOT, call=None, guard=None, clock=time.time, sleep=time.sleep):
    call = call or (lambda argv: command(argv, plan, root))
    guard = guard or (lambda: validate(plan, root))
    state = {
        "schema": SCHEMA,
        "flow_id": FLOW,
        "job_id": JOB,
        "plan_sha256": watch.sha(watch.canonical(plan) + b"\n"),
        "pid": os.getpid(),
        "quest_host": plan["quest_host"],
        "phase": "observing",
        "submission_attempted": False,
        "jobs_cancelled": False,
        "g0_passed": False,
        "pilot_passed": False,
        "factorial_ready": False,
    }

    def record(**changes):
        state.update(changes, updated_at_unix=clock())
        write(dict(state))

    def boundary():
        guard()
        watch.require(clock() < plan["expires_at_unix"], "observer deadline reached; job remains untouched")

    def invoke(operation):
        boundary()
        return call([operation, JOB])

    transitions = 0
    record()
    try:
        receipt = plan["binding"]["receipt"]
        while True:
            status = invoke("status")
            record(observed_status=status)
            outcome = watch.status_outcome(status, receipt)
            if outcome == "transition":
                transitions += 1
            record(completion_transition_observations=transitions)
            watch.require(transitions <= 2, "completion queue did not clear within ten minutes")
            watch.require(
                not (transitions and outcome == "active"), "terminal accounting regressed to active"
            )
            if outcome in {"success", "failure"}:
                fetched = watch.validate_fetch(invoke("fetch"), receipt, status, root)
                record(last_fetch=fetched)
                watch.require(outcome == "success", "calibration failed scientific/accounting acceptance")
                record(
                    phase="calibration_complete", next_gate="complete_G0_requires_review_and_actual_evidence"
                )
                return 0
            boundary()
            sleep(min(watch.POLL, max(0, plan["expires_at_unix"] - clock())))
    except BaseException as error:
        record(phase="stopped", error=f"{type(error).__name__}: {error}", no_retry=True)
        raise


def run(path, expected, *, root=ROOT, call=None, clock=time.time, sleep=time.sleep):
    path, plan = plan_file(path, expected, root)
    fd = os.open(
        watch.safe(root / ".sdsc/student-supervision.lock"), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    previous = {}
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch.require(not (path.parent / "state.json").exists(), "observer already started; never re-arm")
        validate(plan, root)
        watch.require(clock() < plan["expires_at_unix"], "observer deadline reached")
        directory = watch.safe(root / ".sdsc/student-observation-claims")
        directory.mkdir(mode=0o700, exist_ok=True)
        claim_path = directory / ("job-" + JOB + ".json")
        claim_sha = watch.publish(claim_path, {"job_id": JOB, "plan_sha256": expected, "plan": str(path)})

        def interrupt(number, _frame):
            raise InterruptedError(f"observer interrupted by signal {number}; no retry")

        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, interrupt)

        def guard():
            watch.require(watch.sha(watch.read(path)) == expected, "immutable observer plan changed")
            watch.require(
                watch.sha(watch.read(claim_path)) == claim_sha, "permanent observation claim changed"
            )
            validate(plan, root)

        return observe(
            plan,
            lambda value: watch.publish(path.parent / "state.json", value, replace=True),
            root=root,
            call=call,
            guard=guard,
            clock=clock,
            sleep=sleep,
        )
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--output", type=Path, required=True)
    for name in ("run", "status"):
        command_parser = sub.add_parser(name)
        command_parser.add_argument("--plan", type=Path, required=True)
        command_parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan = prepare()
        path = watch.safe(args.output)
        watch.require(
            path == ROOT / ".sdsc/supervision" / FLOW / "plan.json", "wrong observer plan destination"
        )
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=False)
        digest = watch.publish(path, plan)
        print(json.dumps({"plan": str(path), "plan_sha256": digest, "started": False}, indent=2))
    elif args.command == "run":
        return run(args.plan, args.plan_sha256)
    else:
        path, _ = plan_file(args.plan, args.plan_sha256)
        state_path = path.parent / "state.json"
        print(
            json.dumps(
                {"valid": True, "state": watch.document(state_path) if state_path.exists() else None},
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Student observer stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from None
