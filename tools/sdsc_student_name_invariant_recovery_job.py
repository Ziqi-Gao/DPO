#!/usr/bin/env python3
"""Verify a retained old-intent fence before entering the unchanged preparation node."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

MAX_PLAN = 8 * 1024**2
MAX_SETUP_SECONDS = 60
CONTROL_NAME = "tools/sdsc_student_name_invariant_recovery.py"
NODE_NAME = "tools/sdsc_student_name_invariant_recovery_job.py"
ORIGINAL_NAME = "tools/sdsc_student_name_invariant_job.py"


def load_control(path, expected):
    path = Path(path)
    if path.is_symlink() or not 0 < path.stat().st_size <= MAX_PLAN:
        raise ValueError("invalid recovery plan file")
    raw = path.read_bytes()
    if len(raw) > MAX_PLAN or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("recovery node plan changed")
    plan = json.loads(raw)
    pins = plan["execution"]["control_file_sha256"]
    root = Path(__file__).resolve().parents[1]
    for name in (CONTROL_NAME, NODE_NAME):
        source = root / name
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != pins[name]:
            raise ValueError("recovery entry source changed: " + name)
    spec = importlib.util.spec_from_file_location(
        "_name_invariant_recovery_node_control", root / CONTROL_NAME
    )
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)
    control.validate_plan(plan)
    control.require(
        path.absolute() == Path(plan["science_plan"]["submission_dir"]) / "execution-plan.json",
        "recovery node plan path differs",
    )
    return control, plan


def main(argv=None):
    started = time.monotonic()
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        raise ValueError("usage: recovery-node PLAN PLAN_SHA256")
    control, outer = load_control(*args)
    inner = outer["science_plan"]
    original = control.helper("sdsc_student_name_invariant_job")
    control.require(
        control.sha(control.read(Path(original.__file__))) == inner["control_sha256"][ORIGINAL_NAME],
        "original preparation node changed",
    )
    identity = original.allocation(inner, control.science)
    control.verify_source(outer)
    control.check_claims(outer)
    control.verify_execution(outer)
    # Last pre-model check: the exact retained directory and proof must still exist.
    control.verify_fence(outer)
    elapsed = time.monotonic() - started
    control.require(elapsed <= MAX_SETUP_SECONDS, "recovery wrapper setup exceeded60s; no model work")
    directory = control.safe(inner["submission_dir"])
    entry = control.entry_record(outer, identity["job_id"], at=control.now(), setup_elapsed_seconds=elapsed)
    control.write_once(directory / "recovery-node-entry.json", control.canonical(entry))
    control.require(
        control.read(directory / "recovery-node-entry.json") == control.canonical(entry),
        "recovery node entry readback differs",
    )
    try:
        control.require(
            time.monotonic() - started <= MAX_SETUP_SECONDS,
            "recovery entry persistence exceeded setup budget; no model work",
        )
        # No patched globals, redirected output root, modified inputs or altered budget.
        result = original.main([str(directory / "plan.json"), control.sha(control.canonical(inner))])
        control.require(type(result) is int, "original preparation node returned invalid status")
    except BaseException as error:
        exit_record = control.exit_record(
            entry,
            returncode=1,
            node_returned=False,
            completed_at=control.now(),
            error_type=type(error).__name__,
        )
        try:
            control.write_once(directory / "recovery-node-exit.json", control.canonical(exit_record))
        except BaseException as publication_error:
            print(
                json.dumps({"recovery_exit_publication_error": type(publication_error).__name__}),
                file=sys.stderr,
                flush=True,
            )
        raise
    exit_record = control.exit_record(
        entry, returncode=result, node_returned=True, completed_at=control.now()
    )
    control.write_once(directory / "recovery-node-exit.json", control.canonical(exit_record))
    control.require(
        control.read(directory / "recovery-node-exit.json") == control.canonical(exit_record),
        "recovery node exit readback differs",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
