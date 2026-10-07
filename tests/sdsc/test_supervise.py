"""Finite control-flow tests: no SSH, Slurm, model, or real sleeping."""

import copy
import importlib.util
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "supervise", Path(__file__).resolve().parents[2] / "tools/sdsc_supervise.py"
)
supervise = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervise)


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in supervise.CONTROL_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture immutable control\n")
        self.proof = "a" * 64
        self.code = supervise.sha(supervise.canonical([]))
        self.options = {
            "--task": "qwen3-v2-g0-calibration",
            "--account": "nwu181",
            "--partition": "nairr-gpu-shared",
            "--qos": "nairr-gpu-shared-normal",
            "--gpu-type": "h100",
            "--gpus": "2",
            "--cpus": "24",
            "--mem-gib": "192",
            "--time": "01:00:00",
            "--python": "/expanse/lustre/projects/nwu181/zgao12/OPD/envs/runtime/bin/python",
            "--hf-home": "/expanse/lustre/projects/nwu181/zgao12/OPD/cache/huggingface",
            "--result-root": "/expanse/lustre/projects/nwu181/zgao12/OPD/control-results",
            "--provenance-dir": "/home/zgao12/quest-runs/OPD/provenance/" + self.proof,
            "--provenance-manifest-sha256": self.proof,
            "--teacher-job-id": "200",
            "--preflight-job-id": "100",
        }
        self.plan = {
            "schema": "quest-sdsc-calibration-supervision-v1",
            "quest_root": str(self.root),
            "quest_host": socket.gethostname().split(".")[0],
            "control_sha256": supervise.controls(self.root),
            "poll_seconds": 300,
            "deadline_seconds": 28800,
            "code_sha256": self.code,
            "submit_arguments": [
                "submit",
                "fixture-run",
                *[item for pair in self.options.items() for item in pair],
                "--storage-confirmed",
            ],
        }
        path = self.root / ".sdsc/runs/fixture-run.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"state": "deployed", "manifest": {"code_sha256": self.code, "files": []}})
        )
        self.calls, self.states, self.sleeps = [], [], []
        self.elapsed = 0

    def status(self, job, state="COMPLETED", success=True):
        return {"job_id": job, "state": state, "success": success, "result": {"verified": success}}

    def call(self, arguments):
        self.calls.append(arguments)
        if arguments[0] == "status":
            return self.status(arguments[1])
        if arguments[0] == "fetch":
            return {"job_id": arguments[1], "files": ["small-report.json"]}
        self.assertEqual(arguments, [*self.plan["submit_arguments"], "--authorize"])
        return {
            "task": "qwen3-v2-g0-calibration",
            "run_id": "fixture-run",
            "code_sha256": self.code,
            "job_id": "300",
        }

    def flow(self, call=None, guard=lambda: None):
        real_validate = supervise.validate_plan
        with patch.object(
            supervise, "validate_plan", side_effect=lambda plan: real_validate(plan, self.root)
        ):
            return supervise.run_flow(
                self.plan,
                lambda value: self.states.append(copy.deepcopy(value)),
                call=call or self.call,
                sleep=lambda seconds: self.sleeps.append(seconds),
                clock=lambda: self.elapsed,
                guard=guard,
            )

    def test_exact_success_submits_once_after_both_gates(self):
        self.assertEqual(self.flow(), 0)
        self.assertEqual(
            [row[0] for row in self.calls], ["status", "status", "fetch", "submit", "status", "fetch"]
        )
        self.assertEqual(self.states[-1]["phase"], "calibration_complete")
        self.assertFalse(self.states[-1]["g0_passed"])

    def test_failed_teacher_never_submits(self):
        def call(arguments):
            if arguments == ["status", "200"]:
                self.calls.append(arguments)
                return self.status("200", "FAILED", False)
            return self.call(arguments)

        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            self.flow(call)
        self.assertNotIn("submit", [row[0] for row in self.calls])
        self.assertTrue(self.states[-1]["no_retry"])

    def test_completed_without_verified_artifact_stops(self):
        with self.assertRaisesRegex(RuntimeError, "did not pass"):
            self.flow(lambda arguments: self.status(arguments[1], "COMPLETED", False))

    def test_unknown_accounting_is_bounded_and_never_success(self):
        with self.assertRaisesRegex(ValueError, "Three inconclusive"):
            self.flow(lambda arguments: self.status(arguments[1], "UNKNOWN", False))
        self.assertEqual(self.sleeps, [300, 300])

    def test_active_job_waits_five_minutes(self):
        first = True

        def call(arguments):
            nonlocal first
            if arguments == ["status", "200"] and first:
                first = False
                return self.status("200", "RUNNING", False)
            return self.call(arguments)

        self.assertEqual(self.flow(call), 0)
        self.assertEqual(self.sleeps, [300])

    def test_lost_submission_receipt_never_retries(self):
        def call(arguments):
            if arguments[0] == "submit":
                self.calls.append(arguments)
                raise RuntimeError("UNKNOWN: reconcile intent")
            return self.call(arguments)

        with self.assertRaisesRegex(RuntimeError, "UNKNOWN"):
            self.flow(call)
        self.assertEqual(sum(row[0] == "submit" for row in self.calls), 1)
        self.assertTrue(self.states[-1]["submission_attempted"])

    def test_ssh_failure_stops_without_poll_retry(self):
        with self.assertRaisesRegex(RuntimeError, "manual authentication"):
            self.flow(lambda _args: (_ for _ in ()).throw(RuntimeError("manual authentication required")))
        self.assertFalse(self.sleeps)

    def test_source_change_stops_before_any_remote_operation(self):
        def guard():
            raise ValueError("Control tools changed")

        with self.assertRaisesRegex(ValueError, "Control tools changed"):
            self.flow(guard=guard)
        self.assertFalse(self.calls)

    def test_source_change_during_teacher_fetch_prevents_submission(self):
        changed = False

        def call(arguments):
            nonlocal changed
            result = self.call(arguments)
            if arguments == ["fetch", "200"]:
                changed = True
            return result

        def guard():
            if changed:
                raise ValueError("Control tools changed")

        with self.assertRaisesRegex(ValueError, "Control tools changed"):
            self.flow(call, guard)
        self.assertNotIn("submit", [row[0] for row in self.calls])
        self.assertNotIn("submission_attempted", self.states[-1])

    def test_teacher_fetch_exhausting_deadline_prevents_submission(self):
        def call(arguments):
            result = self.call(arguments)
            if arguments == ["fetch", "200"]:
                self.elapsed = self.plan["deadline_seconds"]
            return result

        with self.assertRaisesRegex(ValueError, "deadline reached"):
            self.flow(call)
        self.assertNotIn("submit", [row[0] for row in self.calls])
        self.assertNotIn("submission_attempted", self.states[-1])

    def test_resource_changes_and_extra_commands_are_rejected(self):
        for flag, value in (("--gpus", "4"), ("--task", "pilot"), ("--time", "12:00:00")):
            plan = copy.deepcopy(self.plan)
            plan["submit_arguments"][plan["submit_arguments"].index(flag) + 1] = value
            with self.assertRaises(ValueError):
                supervise.validate_plan(plan, self.root)
        plan = copy.deepcopy(self.plan)
        plan["submit_arguments"] += ["--authorize"]
        with self.assertRaises(ValueError):
            supervise.validate_plan(plan, self.root)

    def test_missing_authorization_and_wrong_host_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "authorization"):
            supervise.main(["--plan", str(self.root / "plan.json")])
        self.plan["quest_host"] = "another-host"
        with self.assertRaisesRegex(ValueError, "different Quest host"):
            supervise.validate_plan(self.plan, self.root)

    def test_changed_control_hash_and_symlink_are_rejected(self):
        (self.root / supervise.CONTROL_FILES[0]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed after plan"):
            supervise.validate_plan(self.plan, self.root)
        target = self.root / "link"
        target.symlink_to(self.root / supervise.CONTROL_FILES[0])
        with self.assertRaisesRegex(ValueError, "Symlink"):
            supervise.safe_file(target)


if __name__ == "__main__":
    unittest.main()
