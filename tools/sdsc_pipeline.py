#!/usr/bin/env python3
"""Finite Quest successor: verified calibration -> G0 -> seed-42 pilot.

No local Slurm commands; no remote service. A saved, authorized immutable plan
owns each stage once. SSH loss, changed control bytes, scientific failure or an
unresolved receipt stops progression without cancelling running work.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdsc_cli as cli
import sdsc_pipeline_remote as control
import sdsc_pipeline_storage as s

ROOT = Path(__file__).resolve().parents[1]


def validate_plan(plan):
    s.require(
        plan.get("schema") == "quest-sdsc-pipeline-plan-v1" and plan.get("authorized") is True,
        "authorized pipeline plan required",
    )
    s.require(
        plan.get("quest_root") == str(ROOT) and plan.get("quest_host") == socket.gethostname().split(".")[0],
        "plan belongs to a different Quest checkout/host",
    )
    s.require(
        plan["stages"] == list(control.STAGES) and plan["resources"] == control.resource_plan(),
        "stage/resource contract changed",
    )
    s.require(
        plan["poll_seconds"] == 300 and plan["deadline_seconds"] == 14 * 24 * 3600,
        "finite supervision bounds changed",
    )
    s.require(
        plan.get("full_factorial_authorized") is False, "three-seed/Gemma expansion is outside this flow"
    )
    s.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", plan["flow_id"]), "invalid pipeline flow ID")
    for name, expected in plan["control_files"].items():
        p = ROOT / s.relative(name)
        s.require(s.identity(p)["sha256"] == expected, "reviewed control bytes changed: " + name)
    parent = s.safe(plan["parent_state"])
    s.require(ROOT / ".sdsc/supervision" in parent.parents, "parent flow must be local task-owned state")
    manifest = cli.run_record(plan["run_id"])["manifest"]
    s.require(manifest["code_sha256"] == plan["code_sha256"], "local release record changed")
    return plan


def remote(plan, plan_hash, action, **fields):
    python = plan["g0_python"]
    script = s.CONTROL / "releases" / plan["run_id"] / "source/tools/sdsc_pipeline_remote.py"
    payload = {"action": action, "flow_id": plan["flow_id"], "plan_sha256": plan_hash, **fields}
    response = cli.ssh_call(
        [python, "-I", "-B", str(script)],
        data=s.canonical(payload),
        timeout=480 if action == "submit" else 180,
    )
    try:
        result = json.loads(response.stdout)
    except (ValueError, UnicodeDecodeError):
        raise RuntimeError(
            "remote receipt missing; reconcile the existing stage claim, never resubmit"
        ) from None
    s.require(
        response.returncode == 0 and result.get("ok") is True, result.get("error", "remote operation failed")
    )
    return result["value"]


def legacy(plan, plan_hash, job):
    intent_id, _ = cli.job_binding(job)
    return remote(plan, plan_hash, "legacy", binding={"job_id": job, "intent_id": intent_id})


def initial_inventory(calibration):
    publication = calibration["publication_receipt"]
    root = Path(calibration["submission"]["result_dir"])
    result = {}
    for entry in publication["files"]:
        p = s.relative(entry["path"])
        if p.parts[0] != "artifacts" or len(p.parts) < 2:
            continue
        if p.parts[1] not in {"canonical_sft", "dataset", "teacher_demos", "initial_checkpoint.pt"}:
            continue
        name = "g0/" + str(Path(*p.parts[1:]))
        s.require(name not in result, "duplicate inherited artifact")
        result[name] = {"size": entry["size"], "sha256": entry["sha256"], "storage": str(root / p)}
    s.require(
        {
            "g0/initial_checkpoint.pt",
            "g0/canonical_sft/manifest.json",
            "g0/dataset/manifest.json",
            "g0/teacher_demos/manifest.json",
        }
        <= result.keys(),
        "calibration lacks complete science inputs",
    )
    return result


def job_request(plan, stage, inventory, statuses, calibration, preflight):
    g0_root = str(Path(calibration["report"]["training_artifact_root"]).parent)
    s.require(
        g0_root.startswith("/scratch/") and g0_root.endswith("/qwen3-v2"),
        "invalid original calibration workspace",
    )
    return {
        "stage": stage,
        "run_id": plan["run_id"],
        "code_sha256": plan["code_sha256"],
        "flow_id": plan["flow_id"],
        "science_head": plan["science_head"],
        "protocol_sha256": plan["protocol_sha256"],
        "provenance_dir": plan["provenance_dir"],
        "provenance_sha256": plan["provenance_sha256"],
        "python": plan["g0_python"] if stage == "g0" else plan["environment"]["runtime"]["python_path"],
        "g0_python_sha256": plan["g0_python_sha256"],
        "environment": plan["environment"],
        "hf_home": plan["hf_home"],
        "adapter_review": plan["adapter_review"],
        "base_inventory": inventory,
        "statuses": statuses,
        "g0_root": g0_root,
        "calibration": calibration,
        "preflight2": preflight,
    }


def run_flow(plan, plan_hash, directory, *, call=None, sleep=time.sleep, clock=time.time):
    call = call or (lambda action, **fields: remote(plan, plan_hash, action, **fields))
    started = clock()
    state = {
        "schema": "quest-sdsc-pipeline-state-v1",
        "flow_id": plan["flow_id"],
        "plan_sha256": plan_hash,
        "pid": os.getpid(),
        "quest_host": plan["quest_host"],
        "phase": "waiting_calibration",
        "stages": {},
        "g0_passed": False,
        "pilot_passed": False,
        "full_factorial_started": False,
        "started_at_unix": started,
    }

    def record(**changes):
        state.update(changes, updated_at_unix=clock())
        s.atomic(directory / "state.json", state, replace=True)

    def boundary():
        validate_plan(plan)
        s.require(
            clock() - started < plan["deadline_seconds"],
            "finite supervisor deadline reached; jobs left untouched",
        )
        cli.require_master()

    unguarded_call = call

    def call(action, **fields):
        boundary()
        return unguarded_call(action, **fields)

    def pause():
        # Finite background watcher sleeps between on-demand scheduler queries;
        # it does not keep an agent turn or a remote service alive.
        sleep(plan["poll_seconds"])

    def fetch_stage(stage):
        result = call("fetch", stage=stage)
        destination = ROOT / ".sdsc/fetched" / ("pipeline-" + plan["flow_id"]) / stage
        for entry in result["files"]:
            raw = base64.b64decode(entry["base64"], validate=True)
            s.require(
                len(raw) == entry["size"] and s.sha(raw) == entry["sha256"], "fetched result SHA differs"
            )
            path = destination / s.relative(entry["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        return {
            "directory": str(destination),
            "files": len(result["files"]),
            "truncated": result["truncated"],
        }

    record()
    try:
        while True:
            boundary()
            parent = s.document(plan["parent_state"])
            if parent.get("phase") == "stopped":
                raise RuntimeError(
                    "teacher/calibration predecessor stopped: " + parent.get("error", "unknown reason")
                )
            job = parent.get("calibration_job_id")
            record(parent_phase=parent.get("phase"), calibration_job_id=job)
            if job and parent.get("phase") == "calibration_complete":
                break
            pause()
        boundary()
        calibration = legacy(plan, plan_hash, job)
        boundary()
        preflight = legacy(plan, plan_hash, plan["preflight_job_id"])
        s.atomic(directory / "calibration-evidence.json", calibration)
        s.atomic(directory / "preflight-evidence.json", preflight)
        inventory = initial_inventory(calibration)
        statuses = {}
        for stage in control.STAGES:
            boundary()
            record(phase="checking_" + stage, current_stage=stage)
            claimed = call("exists", stage=stage)
            if claimed["claimed"]:
                # Process restarts use the same durable remote stage identity.
                # No code path deletes claims or resubmits a claimed stage.
                receipt = call("reconcile", stage=stage)
            else:
                record(phase="submitting_" + stage, submission_boundary=True)
                try:
                    receipt = call(
                        "submit",
                        stage=stage,
                        job_request=job_request(plan, stage, inventory, statuses, calibration, preflight),
                    )
                except Exception:
                    # Check the master once before reconciliation; missing SSH
                    # never starts another connection/authentication attempt.
                    boundary()
                    receipt = call("reconcile", stage=stage)
            s.require(
                re.fullmatch(r"[1-9][0-9]*", receipt.get("job_id", ""))
                and receipt["stage"] == stage
                and receipt["run_id"] == plan["run_id"],
                "untrustworthy stage receipt",
            )
            s.atomic(directory / (stage + "-submission.json"), receipt)
            state["stages"][stage] = {"job_id": receipt["job_id"], "phase": "submitted"}
            record(phase="waiting_" + stage, submission_boundary=False)
            unknown = 0
            while True:
                boundary()
                status = call("status", stage=stage)
                s.atomic(directory / (stage + "-status.json"), status, replace=True)
                state["stages"][stage].update(state=status["state"], success=status["success"])
                record()
                if status["success"]:
                    s.require(
                        status["state"] == "COMPLETED" and status["result"]["verified"],
                        "incomplete stage success proof",
                    )
                    break
                if status["state"] == "FAILED":
                    record(failure_fetch=fetch_stage(stage))
                    raise RuntimeError("stage failed: " + stage + " job " + receipt["job_id"])
                unknown = unknown + 1 if status["state"] == "UNKNOWN" else 0
                s.require(unknown < 3, "three inconclusive accounting observations; stop without retry")
                pause()
            statuses[stage] = status
            state["stages"][stage]["fetch"] = fetch_stage(stage)
            inventory = s.merge_receipts(inventory, [item["value"] for item in status["publications"]])
            s.atomic(directory / (stage + "-inventory.json"), {"files": inventory})
            # Raw small reports/accounting are already fetched here as status
            # metadata. Large states stay on Expanse and are only staged there.
            if stage == "g0":
                record(g0_passed=True)
            if stage == "finalize":
                record(pilot_passed=True)
        record(phase="complete", final_inventory_files=len(inventory))
        return 0
    except BaseException as error:
        record(
            phase="stopped",
            error=type(error).__name__ + ": " + str(error),
            no_retry=True,
            jobs_cancelled=False,
        )
        return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--authorize", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--status", action="store_true", help="read saved flow state and current remote stage")
    p.add_argument(
        "--reconcile-stage", choices=control.STAGES, help="recover one existing receipt; never submit"
    )
    args = p.parse_args(argv)
    path = s.safe(args.plan.absolute())
    plan = s.document(path)
    validate_plan(plan)
    plan_hash = s.sha(s.read(path))
    if args.check:
        print(
            json.dumps(
                {
                    "validated": True,
                    "flow_id": plan["flow_id"],
                    "plan_sha256": plan_hash,
                    "stages": plan["stages"],
                }
            )
        )
        return 0
    if args.status:
        state = s.document(path.parent / "state.json")
        if state.get("current_stage"):
            state["current_remote_status"] = remote(plan, plan_hash, "status", stage=state["current_stage"])
        print(json.dumps(state, indent=2))
        return 0
    if args.reconcile_stage:
        print(json.dumps(remote(plan, plan_hash, "reconcile", stage=args.reconcile_stage), indent=2))
        return 0
    s.require(args.authorize, "saved user authorization and --authorize are required")
    s.require(ROOT / ".sdsc/supervision" in path.parents, "local supervisor state must stay in this project")
    directory = path.parent
    lock = os.open(directory / "lock", os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s.require(
            not (directory / "state.json").exists(),
            "flow already started; inspect/reconcile saved state, never re-arm blindly",
        )
        cli.require_master()
        return run_flow(plan, plan_hash, directory)
    finally:
        os.close(lock)


if __name__ == "__main__":
    sys.exit(main())
