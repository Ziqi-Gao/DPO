"""Bind the reviewed finite supervisor to one accepted v5 preflight receipt.

Earlier implementations and historical plans remain unchanged. This adapter loads
an isolated copy of its helpers, fixes v5 identities from a hashed local profile,
and preserves its resource, provenance, accounting and one-submission guards.
After a real submission receipt, the first calibration query waits one ordinary
poll interval so newly submitted jobs can enter accounting. UNKNOWN still stops.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = "tools/sdsc_student_supervise_v5.py"
PROTOCOL = "prereg/amendments/qwen3_adapted_student_calibration_v5.json"
PROFILE_SCHEMA = "quest-sdsc-student-v5-supervision-profile-v1"
PROFILE_KEYS = {
    "schema",
    "preflight_job_id",
    "preflight_run_id",
    "science_git_head",
    "student_protocol_sha256",
    "student_protocol_artifact_sha256",
}


def load_base():
    spec = importlib.util.spec_from_file_location(
        "isolated_student_v5_supervision", ROOT / "tools/sdsc_student_supervise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile_values(base, path, expected_sha, root):
    path = base.safe(path)
    base.require(
        path.parent == root / ".sdsc/student-supervision-profiles"
        and path.suffix == ".json"
        and base.NAME.fullmatch(path.stem),
        "profile must be one bounded project-owned metadata file",
    )
    base.require(
        base.SHA.fullmatch(expected_sha or "") and base.sha(base.read(path)) == expected_sha,
        "profile hash differs",
    )
    profile = base.document(path)
    base.require(
        set(profile) == PROFILE_KEYS and profile["schema"] == PROFILE_SCHEMA, "profile fields differ"
    )
    base.require(
        re.fullmatch(r"[1-9][0-9]*", str(profile["preflight_job_id"]))
        and isinstance(profile["preflight_job_id"], str)
        and profile["preflight_job_id"] not in {
            "54506703", "54504895", "54496291", "54507345", "54507464", "54509682", "54509809"
        }
        and isinstance(profile["preflight_run_id"], str)
        and base.NAME.fullmatch(profile["preflight_run_id"])
        and re.fullmatch(r"[a-f0-9]{40}", str(profile["science_git_head"])),
        "v5 requires a fresh preflight and genuine scientific HEAD",
    )
    raw = base.read(root / PROTOCOL)
    protocol = json.loads(raw)
    core = copy.deepcopy(protocol)
    review = core.pop("review", {})
    base.require(
        protocol.get("protocol_id") == "qwen3-adapted-student-calibration-v5"
        and review.get("status") == "accepted"
        and re.fullmatch(r"[a-f0-9]{40}", str(review.get("reviewed_implementation_commit"))),
        "v5 protocol must already be independently accepted",
    )
    base.require(
        profile["student_protocol_artifact_sha256"] == base.sha(raw)
        and profile["student_protocol_sha256"] == base.sha(base.canonical(core)),
        "profile does not bind the accepted v5 artifact and scientific core",
    )
    return profile, path


def command_adapter(base, *, clock=time.time, sleep=time.sleep):
    original = base.command
    first_calibration_status = None

    def command(arguments, plan, root=ROOT):
        nonlocal first_calibration_status
        if arguments == ["status", first_calibration_status]:
            # run_flow has already persisted and bound the real receipt before
            # this call. Waiting cannot hide an unknown submission or retry it.
            base.require(clock() < plan["expires_at_unix"], "supervision deadline reached")
            sleep(min(base.POLL, max(0, plan["expires_at_unix"] - clock())))
            base.validate_plan(plan, root)
            base.require(clock() < plan["expires_at_unix"], "supervision deadline reached")
            first_calibration_status = None
        result = original(arguments, plan, root)
        if arguments == base.submit_arguments(plan):
            first_calibration_status = result.get("job_id")
        elif len(arguments) == 2 and arguments[0] == "status":
            # Preserve the exact response before the shared classifier can
            # reject it. This is bounded local evidence, not a success marker.
            base.publish(
                root / ".sdsc/supervision" / plan["flow_id"] / "last-status.json",
                {"observed_at_unix": clock(), "status": result},
                replace=True,
            )
        return result

    return command


def configure(path, expected_sha, root=ROOT):
    base = load_base()
    profile, path = profile_values(base, path, expected_sha, root)
    base.SCHEMA = "quest-sdsc-adapted-student-supervision-v5"
    base.PREFLIGHT = profile["preflight_job_id"]
    base.PREFLIGHT_RUN = profile["preflight_run_id"]
    base.HEAD = profile["science_git_head"]
    base.PROTOCOL_PATH = PROTOCOL
    base.FIXED_BINDINGS = {
        **base.FIXED_BINDINGS,
        **{
            key: profile[key]
            for key in ("science_git_head", "student_protocol_sha256", "student_protocol_artifact_sha256")
        },
    }
    # The profile and this adapter are part of the immutable control surface.
    # No credential, runtime directory or unrelated repository file is added.
    base.CONTROL_FILES = (*base.CONTROL_FILES, SELF, str(path.relative_to(root)), PROTOCOL)
    receipt = base.binding(base.PREFLIGHT, root)["receipt"]
    base.validate_receipt(receipt, calibration=False)
    base.command = command_adapter(base)
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    arguments, remaining = parser.parse_known_args(argv)
    return configure(arguments.profile, arguments.profile_sha256).main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
