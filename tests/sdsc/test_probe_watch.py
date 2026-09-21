"""Finite read-only watch fixtures. No SSH, Slurm, model, or GPU operation."""

import fcntl
import importlib.util
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/sdsc_probe_watch.py"
SPEC = importlib.util.spec_from_file_location("probe_watch_test", SCRIPT)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


class ProbeWatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.job = "54351516"
        self.intent = "b" * 32
        self.run = "20260918-probe-fixture"
        for name in watch.CONTROL_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture control: " + name)
        files = [{"path": "source.py", "sha256": "a" * 64, "size": 2, "mode": 420}]
        self.code = watch.digest(watch.canonical(files))
        self.receipt = {
            "ok": True,
            "job_id": self.job,
            "intent_id": self.intent,
            "run_id": self.run,
            "code_sha256": self.code,
            "task": watch.TASK,
            "resources": {"gpus": 1},
            "python": "/runtime/python",
            "hf_home": "/model/cache",
        }
        self.submission = self.root / ".sdsc/submissions" / (self.intent + ".json")
        self.submission.parent.mkdir(parents=True)
        self.submission.write_text(
            json.dumps(
                {
                    "state": "submitted",
                    "receipt": self.receipt,
                    "request": {
                        key: value for key, value in self.receipt.items() if key not in {"ok", "job_id"}
                    },
                }
            )
        )
        self.run_record = self.root / ".sdsc/runs" / (self.run + ".json")
        self.run_record.parent.mkdir(parents=True)
        self.run_record.write_text(
            json.dumps(
                {
                    "state": "deployed",
                    "manifest": {"files": files, "code_sha256": self.code},
                    "deployment": {"ok": True, "run_id": self.run, "code_sha256": self.code},
                }
            )
        )
        self.plan_path, self.plan = watch.prepare(self.job, "probe-watch-fixture", root=self.root, now=100)

    def status(self, state="PENDING", success=False):
        return {
            "job_id": self.job,
            "run_id": self.run,
            "task": watch.TASK,
            "state": state,
            "success": success,
            "queue": {
                "returncode": 0,
                "stdout": "" if state != "PENDING" else self.job + "|PENDING|maintenance",
            },
            "accounting": {"returncode": 0, "stdout": self.job + "|" + state + "|0:0|00:00:01|\n"},
        }

    def drive(self, statuses, *, fault=None, guard=None, clock_start=100):
        now = [clock_start]
        calls, sleeps, writes = [], [], []
        iterator = iter(statuses)

        def call(arguments):
            calls.append(arguments)
            if fault and arguments[0] == fault[0]:
                raise RuntimeError(fault[1])
            if arguments[0] == "status":
                return next(iterator)
            return {"job_id": self.job}

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        def write(name, value):
            writes.append((name, json.loads(json.dumps(value))))

        def conclude(*_args):
            return {"diagnostic_completed": False, **watch.READINESS}

        self.calls, self.sleeps, self.writes = calls, sleeps, writes
        return watch.run_flow(
            self.plan,
            write,
            call=call,
            guard=guard or (lambda: None),
            conclude=conclude,
            sleep=sleep,
            clock=lambda: now[0],
        )

    def test_preparation_pins_exact_receipt_and_deployed_record(self):
        self.assertEqual(watch.validate_plan(self.plan, self.root), self.receipt)
        self.assertEqual(
            self.plan["binding"]["submission_record_sha256"], watch.digest(self.submission.read_bytes())
        )
        self.assertEqual(
            self.plan["binding"]["run_record_sha256"], watch.digest(self.run_record.read_bytes())
        )
        self.assertEqual(self.plan["allowed_operations"], ["status", "logs", "fetch"])
        with self.assertRaises(FileExistsError):
            watch.prepare(self.job, "probe-watch-fixture", root=self.root)

    def test_changed_controls_or_receipt_stop_before_remote_work(self):
        for name in ("control", "receipt", "deployment"):
            with self.subTest(name=name):
                path = {
                    "control": self.root / watch.CONTROL_FILES[0],
                    "receipt": self.submission,
                    "deployment": self.run_record,
                }[name]
                old = path.read_bytes()
                path.write_bytes(old + b" ")
                with self.assertRaisesRegex(ValueError, "changed"):
                    watch.validate_plan(self.plan, self.root)
                path.write_bytes(old)

    def test_unknown_or_ambiguous_submission_cannot_prepare(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            watch.prepare("999", "unknown", root=self.root)
        duplicate = self.submission.with_name("c" * 32 + ".json")
        duplicate.write_bytes(self.submission.read_bytes())
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            watch.prepare(self.job, "duplicate", root=self.root)

    def test_plan_is_bound_to_host_and_exact_cadence_deadline(self):
        for key, value in (
            ("quest_host", socket.gethostname() + "-other"),
            ("poll_seconds", 1),
            ("expires_at_unix", self.plan["expires_at_unix"] + 1),
            ("allowed_operations", ["status", "submit"]),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                watch.validate_plan({**self.plan, key: value}, self.root)

    def test_maintenance_wait_is_five_minutes_then_logs_and_fetch_once(self):
        self.assertEqual(self.drive([self.status(), self.status("RUNNING"), self.status("COMPLETED")]), 0)
        self.assertEqual(self.sleeps, [300, 300])
        self.assertEqual(
            self.calls,
            [
                ["status", self.job],
                ["status", self.job],
                ["status", self.job],
                ["logs", self.job, "--lines", "80"],
                ["fetch", self.job],
            ],
        )
        state = self.writes[-1][1]
        self.assertEqual(state["phase"], "finished")
        for key in watch.READINESS:
            self.assertIs(state[key], False)

    def test_failed_job_still_collects_once_without_retry_or_cancel(self):
        self.assertEqual(self.drive([self.status("FAILED")]), 0)
        self.assertEqual([call[0] for call in self.calls], ["status", "logs", "fetch"])
        self.assertEqual(self.sleeps, [])

    def test_accounting_terminal_waits_for_allocation_to_leave_queue(self):
        transitional = self.status("COMPLETED")
        transitional["queue"]["stdout"] = self.job + "|COMPLETING|None"
        self.assertEqual(self.drive([transitional, self.status("COMPLETED")]), 0)
        self.assertEqual(self.sleeps, [300])
        self.assertEqual([call[0] for call in self.calls], ["status", "status", "logs", "fetch"])

    def test_ssh_loss_stops_without_fallback_or_retry(self):
        with self.assertRaisesRegex(RuntimeError, "master unavailable"):
            self.drive([], fault=("status", "SSH master unavailable; authenticate manually"))
        self.assertEqual(self.calls, [["status", self.job]])
        self.assertEqual(self.writes[-1][1]["phase"], "stopped")
        self.assertEqual(self.sleeps, [])

    def test_fetch_failure_is_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            self.drive([self.status("FAILED")], fault=("fetch", "connection lost"))
        self.assertEqual([call[0] for call in self.calls], ["status", "logs", "fetch"])
        self.assertEqual(self.writes[-1][1]["phase"], "stopped")

    def test_unknown_state_does_not_assume_queue_disappearance_is_success(self):
        with self.assertRaisesRegex(ValueError, "Unknown job state"):
            self.drive([self.status("UNKNOWN")])
        self.assertEqual(self.calls, [["status", self.job]])
        self.assertFalse(any(name == "conclusion.json" for name, _ in self.writes))

    def test_identity_mismatch_and_changed_controls_stop(self):
        with self.assertRaisesRegex(ValueError, "job differs"):
            self.drive([{**self.status(), "job_id": "999"}])
        invoked = [0]

        def guard():
            invoked[0] += 1
            if invoked[0] == 2:
                raise ValueError("Pinned controls changed")

        with self.assertRaisesRegex(ValueError, "controls changed"):
            self.drive([self.status()], guard=guard)
        self.assertEqual(self.calls, [["status", self.job]])
        self.assertEqual(self.sleeps, [])

    def test_expiry_stops_with_no_final_remote_call(self):
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.drive([], clock_start=self.plan["expires_at_unix"])
        self.assertEqual(self.calls, [])
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.drive([self.status()], clock_start=self.plan["expires_at_unix"] - 5)
        self.assertEqual(self.sleeps, [5])
        self.assertEqual(self.calls, [["status", self.job]])

    def test_command_rejects_mutation_and_unbounded_logs(self):
        for arguments in (
            ["submit", self.job],
            ["cancel", self.job],
            ["reconcile", self.job],
            ["logs", self.job, "--lines", "100000"],
            ["status", "1; sbatch"],
        ):
            with self.subTest(arguments=arguments), patch.object(watch.subprocess, "run") as execute:
                with self.assertRaises(ValueError):
                    watch.command(arguments, self.root)
                execute.assert_not_called()

    def test_started_flows_cannot_be_rearmed(self):
        (self.plan_path.parent / "state.json").write_text('{"phase":"stopped"}')
        with patch.object(watch, "command") as execute, self.assertRaisesRegex(ValueError, "already started"):
            watch.run_plan(self.plan_path, self.root)
        execute.assert_not_called()

    def test_per_job_flock_rejects_a_competing_flow(self):
        lock = self.root / ".sdsc/supervision" / (".probe-watch-" + self.job + ".lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(watch, "command") as execute, self.assertRaises(BlockingIOError):
                watch.run_plan(self.plan_path, self.root)
            execute.assert_not_called()
        finally:
            os.close(descriptor)

    def fetched(self):
        directory = self.root / ".sdsc/fetched" / self.job / "fetch-fixture"
        directory.mkdir(parents=True)
        identity = {key: self.receipt[key] for key in ("job_id", "run_id", "code_sha256", "task", "hf_home")}

        def arm(count):
            return {
                "attempt_count": count,
                "accepted_count": 0,
                "prompts_with_success": 0,
                "total_prompts": 32,
            }

        report = {
            **identity,
            **watch.READINESS,
            "passed": True,
            "exit_code": 0,
            "exploratory": True,
            "execution_class_certified": False,
            "arms": {"baseline": arm(32), "candidate": arm(256)},
            "paired_candidate_zero": arm(32),
        }
        raw = watch.canonical(report)
        (directory / watch.REPORT).write_bytes(raw)
        publication = {
            **identity,
            "passed": True,
            "persisted": True,
            "persistent_read_back_verified": True,
            "files": [{"path": watch.REPORT, "size": len(raw), "sha256": watch.digest(raw)}],
        }
        (directory / "receipt.json").write_bytes(watch.canonical(publication))
        fetched = {
            "job_id": self.job,
            "intent_id": self.intent,
            "destination": str(directory),
            "files": [watch.REPORT, "receipt.json"],
            "bytes": len(raw) + len(watch.canonical(publication)),
        }
        status = self.status("COMPLETED", True)
        status["result"] = {"verified": True, "result_sha256": watch.digest(raw)}
        return fetched, status

    def test_verified_diagnostic_completion_never_means_scientific_readiness(self):
        fetched, status = self.fetched()
        conclusion = watch.conclusion(self.plan, status, fetched, self.root)
        self.assertTrue(conclusion["diagnostic_completed"])
        self.assertEqual(conclusion["candidate"]["prompts_with_success"], 0)
        self.assertEqual(conclusion["baseline"]["attempt_count"], 32)
        self.assertEqual(conclusion["paired_candidate_zero"]["attempt_count"], 32)
        for key in watch.READINESS:
            self.assertIs(conclusion[key], False)

    def test_status_success_without_zero_exit_accounting_is_rejected(self):
        fetched, status = self.fetched()
        status["accounting"]["stdout"] = self.job + "|COMPLETED|1:0|00:01:00|\n"
        with self.assertRaisesRegex(ValueError, "Incomplete diagnostic"):
            watch.conclusion(self.plan, status, fetched, self.root)

    def test_tampered_fetched_report_is_not_summarized(self):
        fetched, status = self.fetched()
        path = Path(fetched["destination"]) / watch.REPORT
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "publication hash"):
            watch.conclusion(self.plan, status, fetched, self.root)

    def test_terminal_failure_without_report_is_explicitly_incomplete(self):
        directory = self.root / ".sdsc/fetched" / self.job / "fetch-failure"
        directory.mkdir(parents=True)
        fetched = {"job_id": self.job, "intent_id": self.intent, "destination": str(directory), "bytes": 0}
        result = watch.conclusion(self.plan, self.status("TIMEOUT"), fetched, self.root)
        self.assertFalse(result["report_available"])
        self.assertFalse(result["diagnostic_completed"])


if __name__ == "__main__":
    unittest.main()
