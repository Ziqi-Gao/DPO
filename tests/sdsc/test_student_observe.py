"""Only CPU fixtures: recovery of an existing receipt never submits a job."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


observer = module("student_observe_fixture", ROOT / "tools/sdsc_student_observe.py")
fixtures = module("student_supervise_test_helpers", Path(__file__).with_name("test_student_supervise.py"))


class StudentObserverTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.StudentSupervisionTests()
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.root
        (self.root / observer.SELF).write_text("fixed observer fixture")
        manifest = self.base.release(observer.RUN)
        receipt = self.base.calibration
        receipt.update(job_id=observer.JOB, run_id=observer.RUN, intent_id=observer.INTENT)
        receipt["code_sha256"] = manifest["code_sha256"]
        receipt["prerequisites_path"] = (
            fixtures.watch.REMOTE + "/submissions/" + observer.INTENT + "/prerequisites.json"
        )
        receipt["result_dir"] = fixtures.watch.RESULT_ROOT + "/" + observer.RUN + "/" + observer.INTENT
        self.base.save_receipt(receipt)
        self.binding = observer.watch.binding(observer.JOB, self.root)
        self.original = copy.deepcopy(self.base.plan)
        self.original["flow_id"] = observer.ORIGIN
        self.original["calibration"] = {key: receipt[key] for key in self.original["calibration"]}
        self.origin_directory = self.root / ".sdsc/supervision" / observer.ORIGIN
        self.base.put(self.origin_directory / "plan.json", self.original)
        self.origin_state = dict(
            phase="stopped",
            plan_sha256=observer.ORIGIN_SHA,
            submission_attempted=True,
            submission_outcome_unknown=False,
            calibration_job_id=observer.JOB,
            calibration_run_id=observer.RUN,
            calibration_binding=self.binding,
        )
        self.base.put(self.origin_directory / "state.json", self.origin_state)
        self.origin_claim = self.root / ".sdsc/student-supervision-claims/preflight-54506703.json"
        self.base.put(
            self.origin_claim,
            dict(
                plan_sha256=observer.ORIGIN_SHA,
                plan=str(self.origin_directory / "plan.json"),
                run_id=observer.RUN,
                preflight_job_id=fixtures.watch.PREFLIGHT,
            ),
        )
        # Original full plan validation has its own 22-test suite. The receipt,
        # stopped-state, claim, new plan and observer state machine run for real.
        self.plan_mock = patch.object(observer.watch, "plan_file", return_value=(None, self.original))
        self.plan_mock.start()
        self.addCleanup(self.plan_mock.stop)
        self.plan = observer.prepare(self.root, now=1000)
        self.directory = self.root / ".sdsc/supervision" / observer.FLOW
        self.plan_path = self.directory / "plan.json"
        self.plan_sha = self.base.put(self.plan_path, self.plan)
        self.calls = []

    def call(self, argv):
        self.calls.append(argv)
        self.assertIn(argv, (["status", observer.JOB], ["fetch", observer.JOB]))
        return self.base.call(argv)

    def flow(self):
        return observer.observe(
            self.plan,
            self.base.states.append,
            root=self.root,
            call=self.call,
            clock=lambda: self.base.time,
            sleep=self.base.sleep,
        )

    def test_known_submitted_job_success_fetches_without_submit(self):
        self.assertEqual(self.flow(), 0)
        self.assertEqual(self.calls, [["status", observer.JOB], ["fetch", observer.JOB]])
        state = self.base.states[-1]
        self.assertEqual(state["phase"], "calibration_complete")
        self.assertFalse(state["submission_attempted"])
        self.assertFalse(state["g0_passed"])

    def test_running_waits_exactly_five_minutes(self):
        self.base.statuses[observer.JOB] = [self.base.status(self.base.calibration, "RUNNING")]
        self.assertEqual(self.flow(), 0)
        self.assertEqual(self.base.time, 1300)

    def test_failed_job_fetches_small_results_and_stops(self):
        self.base.statuses[observer.JOB] = [self.base.status(self.base.calibration, "FAILED", False)]
        with self.assertRaisesRegex(ValueError, "acceptance"):
            self.flow()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.base.states[-1]["phase"], "stopped")

    def test_unknown_keeps_raw_status_then_stops_without_retry(self):
        unknown = self.base.status(self.base.calibration, "UNKNOWN", False)
        self.base.statuses[observer.JOB] = [unknown]
        with self.assertRaisesRegex(ValueError, "unknown scheduler"):
            self.flow()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.base.states[-1]["observed_status"], unknown)

    def test_ssh_loss_stops_without_retry(self):
        def broken(_argv):
            raise RuntimeError("SSH master unavailable")

        self.base.failure = broken
        with self.assertRaisesRegex(RuntimeError, "SSH"):
            self.flow()
        self.assertEqual(len(self.calls), 1)

    def test_no_unknown_submission_wrong_job_or_live_original_recovery(self):
        for key, value in (
            ("phase", "waiting_calibration"),
            ("submission_attempted", False),
            ("submission_outcome_unknown", True),
            ("calibration_job_id", "54505782"),
            ("calibration_run_id", "other"),
        ):
            with self.subTest(key=key):
                wrong = {**self.origin_state, key: value}
                self.base.put(self.origin_directory / "state.json", wrong)
                with self.assertRaisesRegex(ValueError, "known submitted"):
                    observer.prepare(self.root, now=1000)

    def test_original_receipt_claim_or_state_mutation_stops(self):
        self.base.put(self.origin_claim, {"wrong": True})
        with self.assertRaisesRegex(ValueError, "claim"):
            self.flow()
        self.assertEqual(self.calls, [])

    def test_command_rejects_submit_cancel_and_other_jobs_before_cli(self):
        for argv in (["submit", observer.RUN], ["cancel", observer.JOB], ["status", "54506703"]):
            with patch.object(observer.watch, "command") as call, self.assertRaises(ValueError):
                observer.command(argv, self.plan, self.root)
            call.assert_not_called()
        with patch.object(observer.watch, "command", return_value={"fixture": True}) as call:
            observer.command(["status", observer.JOB], self.plan, self.root)
            call.assert_called_once_with(["status", observer.JOB], self.original, self.root)

    def test_actual_command_reaches_original_cli_with_only_subprocess_mocked(self):
        response = self.base.status(self.base.calibration, "RUNNING")
        completed = SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")
        with patch.object(observer.watch.subprocess, "run", return_value=completed) as launch:
            self.assertEqual(observer.command(["status", observer.JOB], self.plan, self.root), response)
        argv = launch.call_args.args[0]
        self.assertEqual(argv[-3:], [str(self.root / "tools/sdsc_cli.py"), "status", observer.JOB])
        self.assertEqual(launch.call_args.kwargs["timeout"], 420)
        self.assertNotIn("submit", argv)
        completed.returncode = 1
        completed.stderr = "SSH master unavailable"
        with (
            patch.object(observer.watch.subprocess, "run", return_value=completed),
            self.assertRaisesRegex(ValueError, "no retry"),
        ):
            observer.command(["status", observer.JOB], self.plan, self.root)

    def test_observer_deadline_cannot_extend_original_or_start_after_expiry(self):
        self.assertEqual(self.plan["expires_at_unix"], self.original["expires_at_unix"])
        self.base.time = self.plan["expires_at_unix"]
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.flow()
        self.assertEqual(self.calls, [])
        wrong = {**self.plan, "expires_at_unix": self.plan["expires_at_unix"] + 1}
        with self.assertRaisesRegex(ValueError, "changed"):
            observer.validate(wrong, self.root)

    def test_permanent_observation_claim_prevents_rearm_without_changing_original(self):
        old_state = (self.origin_directory / "state.json").read_bytes()
        old_claim = self.origin_claim.read_bytes()
        self.base.statuses[observer.JOB] = [self.base.status(self.base.calibration, "UNKNOWN", False)]
        with self.assertRaisesRegex(ValueError, "unknown scheduler"):
            observer.run(
                self.plan_path,
                self.plan_sha,
                root=self.root,
                call=self.call,
                clock=lambda: self.base.time,
                sleep=self.base.sleep,
            )
        self.assertEqual(old_state, (self.origin_directory / "state.json").read_bytes())
        self.assertEqual(old_claim, self.origin_claim.read_bytes())
        with self.assertRaisesRegex(ValueError, "already started"):
            observer.run(
                self.plan_path, self.plan_sha, root=self.root, call=self.call, clock=lambda: self.base.time
            )
        (self.directory / "state.json").unlink()
        with self.assertRaises(FileExistsError):
            observer.run(
                self.plan_path, self.plan_sha, root=self.root, call=self.call, clock=lambda: self.base.time
            )
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
