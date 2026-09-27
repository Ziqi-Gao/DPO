"""Continuation boundary tests: no real SSH, Codex, Slurm, model, or GPU."""

import fcntl
import importlib.util
import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/sdsc_auto_continue.py"
SPEC = importlib.util.spec_from_file_location("auto_continue_test", SCRIPT)
auto = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto)
watch = auto.watch


class Child:
    def __init__(self, *, timeout=False, returncode=0, notice=None):
        self.pid = 12345
        self.returncode = None
        self.exit_code = returncode
        self.timeout = timeout
        self.notice = notice
        self.terminated = 0
        self.killed = 0

    def communicate(self, *, input, timeout):
        self.input = input
        self.timeout_argument = timeout
        if self.notice:
            self.notice()
        if self.timeout:
            raise subprocess.TimeoutExpired("fixture-child", timeout)
        self.returncode = self.exit_code

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def kill(self):
        self.killed += 1
        self.returncode = -9

    def wait(self, *, timeout):
        return self.returncode


class AutoContinueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.job = auto.SUPPORTED_JOB
        self.intent = "b" * 32
        for name in auto.CONTROL_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture " + name)
        prompt = self.root / auto.PROMPT_FILE
        prompt.parent.mkdir(parents=True)
        prompt.write_text("Follow the reviewed scientific gates. No duplicate submissions.\n")
        self.codex = self.root / "codex-fixture"
        self.codex.write_bytes(b"not an executable process: fixture only")
        self.codex.chmod(0o700)
        files = [{"path": "source.py", "sha256": "a" * 64, "size": 2, "mode": 420}]
        self.code = auto.digest(auto.canonical(files))
        self.addCleanup(patch.stopall)
        patch.object(auto, "SUPPORTED_CODE_SHA256", self.code).start()
        self.receipt = {
            "ok": True, "job_id": self.job, "intent_id": self.intent, "run_id": auto.SUPPORTED_RUN,
            "code_sha256": self.code, "task": watch.TASK, "resources": {"gpus": 1},
            "python": "/runtime/python", "hf_home": "/model/cache",
        }
        self.submission = self.root / ".sdsc/submissions" / (self.intent + ".json")
        self.submission.parent.mkdir(parents=True)
        auto.atomic(self.submission, {
            "state": "submitted", "receipt": self.receipt,
            "request": {key: value for key, value in self.receipt.items() if key not in {"ok", "job_id"}},
        })
        run_record = self.root / ".sdsc/runs" / (auto.SUPPORTED_RUN + ".json")
        run_record.parent.mkdir(parents=True)
        auto.atomic(run_record, {
            "state": "deployed", "manifest": {"files": files, "code_sha256": self.code},
            "deployment": {"ok": True, "run_id": auto.SUPPORTED_RUN, "code_sha256": self.code},
        })
        self.watch_path, self.watch_plan = watch.prepare(self.job, "watch-fixture", root=self.root, now=100)
        self.path, self.plan = self.prepare("auto-fixture")
        self.now = 100
        self.master = Mock()
        self.child = Child()
        self.spawn = Mock(return_value=self.child)
        self.watcher_state("waiting")

    def prepare(self, flow):
        return auto.prepare(self.job, self.watch_path, flow, self.codex,
                            authorize=True, root=self.root, now=100)

    def watcher_state(self, phase, **changes):
        state = {
            "schema": watch.SCHEMA, "job_id": self.job,
            "quest_host": socket.gethostname().split(".")[0],
            "plan_sha256": auto.digest(auto.canonical(self.watch_plan)),
            "updated_at_unix": self.now, "phase": phase,
            "observed_status": {"state": "RUNNING"},
            **changes,
        }
        auto.atomic(self.watch_path.parent / "state.json", state)

    def terminal(self, *, coverage=32, attempts=256, success=True, fields=None):
        directory = self.root / ".sdsc/fetched" / self.job / "fetch-fixture"
        directory.mkdir(parents=True, exist_ok=True)
        identity = {key: self.receipt[key] for key in ("job_id", "run_id", "code_sha256", "task", "hf_home")}
        # Fixed public training identities, independent of ignored local evidence.
        pairs = (
            ("1c5d657770adb6f4e654", "pos"), ("095b26e1aebe0f9eba88", "pos"),
            ("a1d03f5e2f991d6d61af", "neg"), ("ee10dd45ebe767be3360", "pos"),
            ("b80e0fb213a7e2c7ad87", "neg"), ("2b72a368118dc299cda2", "pos"),
            ("228a5c030a11cedc4c9d", "pos"), ("0364a0f58e50e5af49ec", "neg"),
            ("a35c14791e2573aebf20", "pos"), ("62d52ff5fb5a7935ebd6", "neg"),
            ("4a366c27ea4d2267cea8", "pos"), ("9c50e420deea4eb62a7d", "pos"),
            ("ce50ac78929b00a2282a", "neg"), ("f6eb65f4fc0c66d7be39", "neg"),
            ("35642e952a781cbd71e4", "neg"), ("4a7426eec1f2affdba65", "pos"),
        )
        ordered = [f"pgpair-{pair}-{label}" for pair, first in pairs
                   for label in (first, "neg" if first == "pos" else "pos")]

        def arm(count, covered):
            return {"attempt_count": count, "total_prompts": 32,
                    "accepted_count": covered, "prompts_with_success": covered}

        report = {
            **identity, **watch.READINESS, "passed": True, "exit_code": 0, "exploratory": True,
            "execution_class_certified": False, "resumable": False,
            "arms": {"baseline": arm(32, 0), "candidate": arm(attempts, coverage)},
            "paired_candidate_zero": arm(32, 0), "ordered_prompt_ids": ordered,
            "candidate_instructions_sha256": auto.V7_INSTRUCTIONS_SHA256,
            "baseline_instructions_sha256": auto.V5_INSTRUCTIONS_SHA256,
            "candidate_protocol": "qwen3-v2-g0-candidate-e-seed-42-prompt-v7",
            "baseline_protocol": "qwen3-v2-g0-candidate-e-seed-42-prompt-v5",
            "candidate_protocol_review": "proposed", "population_sha256": auto.POPULATION_SHA256,
            "sampling_request_seed": 31415, **(fields or {}),
        }
        raw = auto.canonical(report)
        (directory / watch.REPORT).write_bytes(raw)
        publication = {**identity, "passed": True, "persisted": True, "persistent_read_back_verified": True,
                       "files": [{"path": watch.REPORT, "size": len(raw), "sha256": auto.digest(raw)}]}
        auto.atomic(directory / "receipt.json", publication)
        fetched = {"job_id": self.job, "intent_id": self.intent, "destination": str(directory),
                   "bytes": len(raw) + len(auto.canonical(publication))}
        status = {
            "job_id": self.job, "run_id": auto.SUPPORTED_RUN, "task": watch.TASK,
            "state": "COMPLETED" if success else "FAILED", "success": success,
            "queue": {"returncode": 0, "stdout": ""},
            "accounting": {"returncode": 0,
                           "stdout": self.job + ("|COMPLETED|0:0|" if success else "|FAILED|1:0|")},
            "result": {"verified": True, "result_sha256": auto.digest(raw)},
        }
        auto.atomic(self.watch_path.parent / "terminal-status.json", status)
        self.watcher_state("finished", fetch=fetched, conclusion={"diagnostic_completed": True})
        return directory, status

    def drive(self, **kwargs):
        return auto.run_plan(self.path, self.root, master=self.master, spawn=self.spawn,
                             clock=lambda: self.now, sleep=kwargs.pop("sleep", Mock()), **kwargs)

    def state(self):
        return auto.read_json(self.path.parent / "state.json")

    def assert_no_launch(self):
        self.spawn.assert_not_called()
        self.assertFalse(auto.claim_path(self.root, self.job).exists())
        self.assertEqual(self.state()["phase"], "stopped")

    def test_prepare_requires_authorization_and_the_supported_job(self):
        with self.assertRaisesRegex(ValueError, "authorize"):
            auto.prepare(self.job, self.watch_path, "unauthorized", self.codex, root=self.root)
        with self.assertRaisesRegex(ValueError, "only job"):
            auto.prepare("123", self.watch_path, "other-job", self.codex, authorize=True, root=self.root)
        self.assertEqual(auto.validate_plan(self.plan, self.root), self.watch_plan)

    def test_failed_diagnostic_never_launches_even_when_report_passed(self):
        self.terminal(success=False)
        self.assertEqual(self.drive(), 2)
        self.assert_no_launch()
        self.master.assert_not_called()

    def test_partial_coverage_never_launches(self):
        self.terminal(coverage=31)
        self.assertEqual(self.drive(), 2)
        self.assert_no_launch()
        self.assertIn("31/32", (self.path.parent / "notice.md").read_text())

    def test_incomplete_candidate_population_never_launches(self):
        self.terminal(attempts=255)
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_complete_verified_population_launches_exactly_once(self):
        self.terminal()
        self.assertEqual(self.drive(), 0)
        self.master.assert_called_once_with()
        self.spawn.assert_called_once()
        argv, kwargs = self.spawn.call_args.args[0], self.spawn.call_args.kwargs
        self.assertEqual(argv, [str(self.codex), "exec", "--approve-for-me", "--skip-git-repo-check",
                                "-C", str(self.root), "--json", "-o",
                                str(self.path.parent / "continuation-last-message.txt"), "-"])
        self.assertFalse(
            {"--sandbox", "--dangerously-bypass-approvals-and-sandbox", "--full-auto"} & set(argv)
        )
        self.assertIs(kwargs["stdin"], subprocess.PIPE)
        self.assertEqual(kwargs["umask"], 0o077)
        self.assertNotIn("env", kwargs)
        self.assertIn(self.job.encode(), self.child.input)
        self.assertIn((self.root / auto.PROMPT_FILE).read_bytes(), self.child.input)
        self.assertEqual(self.child.timeout_argument, 8 * 3600)
        self.assertEqual(self.state()["phase"], "continuation_finished")
        self.assertIsNone(self.state()["training_started"])
        self.assertIsNone(self.state()["scientific_success"])
        self.assertTrue(auto.claim_path(self.root, self.job).is_file())
        for name in ("continuation-stdout.jsonl", "continuation-stderr.log", "continuation-last-message.txt"):
            self.assertEqual((self.path.parent / name).stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "already started"):
            self.drive()
        self.spawn.assert_called_once()

    def test_waiting_consumes_only_local_watcher_state(self):
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            self.now += seconds
            self.terminal()

        with patch.object(watch, "command", side_effect=AssertionError("unexpected remote query")):
            self.assertEqual(self.drive(sleep=sleep), 0)
        self.assertEqual(sleeps, [300])
        self.spawn.assert_called_once()

    def test_stopped_watcher_never_launches(self):
        self.watcher_state("stopped")
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_stale_finished_watcher_never_launches(self):
        self.terminal()
        self.now += auto.STALE_SECONDS + 1
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_unknown_slurm_state_never_launches(self):
        self.watcher_state("waiting", observed_status={"state": "UNKNOWN"})
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_ssh_failure_stops_without_claim_or_retry(self):
        self.terminal()
        self.master.side_effect = RuntimeError("SSH master unavailable; manual authentication required")
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()
        self.master.assert_called_once()

    def test_changed_controls_receipt_prompt_binary_and_watcher_plan_are_rejected(self):
        for path in (
            self.root / auto.OWN_CONTROLS[0], self.root / watch.CONTROL_FILES[0], self.submission,
            self.root / auto.PROMPT_FILE, self.codex, self.watch_path,
        ):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    auto.validate_plan(self.plan, self.root)
                path.write_bytes(original)
        self.spawn.assert_not_called()

    def test_changed_own_plan_fails_preparation_seal_before_ssh(self):
        self.path.write_bytes(self.path.read_bytes() + b" ")
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()
        self.master.assert_not_called()

    def test_host_root_expiry_authorization_and_resource_limit_injection_rejected(self):
        for key, value in (
            ("quest_host", "other-host"), ("quest_root", "/tmp/another-root"),
            ("expires_at_unix", self.plan["expires_at_unix"] + 1), ("authorized", False),
            ("poll_seconds", 1), ("stale_seconds", 999999), ("child_seconds", 999999),
            ("prompt_file", "../outside.md"), ("job_id", "1; sbatch"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                auto.validate_plan({**self.plan, key: value}, self.root)

    def test_wrong_v7_instruction_is_rejected_even_with_consistent_publication_hashes(self):
        self.terminal(fields={"candidate_instructions_sha256": "a" * 64})
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_wrong_protocol_is_rejected(self):
        self.terminal(fields={"candidate_protocol": "another-protocol"})
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_wrong_population_identity_is_rejected(self):
        self.terminal(fields={"population_sha256": "a" * 64})
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_different_ordered_prompt_ids_are_rejected(self):
        self.terminal(fields={"ordered_prompt_ids": [str(index) for index in range(32)]})
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_false_formal_readiness_claim_is_rejected(self):
        self.terminal(fields={"training_started": True})
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_report_tamper_is_rejected_without_trusting_cached_conclusion(self):
        directory, _ = self.terminal()
        path = directory / watch.REPORT
        path.write_bytes(path.read_bytes() + b" ")
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_accounting_nonzero_exit_rejects_false_success(self):
        _, status = self.terminal()
        status["accounting"]["stdout"] = self.job + "|COMPLETED|1:0|"
        auto.atomic(self.watch_path.parent / "terminal-status.json", status)
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_permanent_claim_in_another_flow_prevents_duplicate_launch(self):
        second, _ = self.prepare("second-flow")
        self.terminal()
        self.assertEqual(self.drive(), 0)
        with self.assertRaisesRegex(ValueError, "already claimed"):
            auto.run_plan(second, self.root, master=self.master, spawn=self.spawn)
        self.spawn.assert_called_once()

    def test_spawn_failure_retains_claim_and_cannot_rearm(self):
        self.terminal()
        self.spawn.side_effect = OSError("cannot spawn fixture")
        self.assertEqual(self.drive(), 1)
        self.assertTrue(auto.claim_path(self.root, self.job).exists())
        with self.assertRaisesRegex(ValueError, "already claimed"):
            self.prepare("attempted-retry")
        self.spawn.assert_called_once()

    def test_stopped_flow_cannot_be_rearmed_through_another_directory(self):
        second, _ = self.prepare("second-flow")
        self.terminal(coverage=0)
        self.assertEqual(self.drive(), 2)
        with self.assertRaisesRegex(ValueError, "already started"):
            auto.run_plan(second, self.root, master=self.master, spawn=self.spawn)
        self.spawn.assert_not_called()

    def test_job_lock_excludes_competing_process(self):
        path = auto.claim_path(self.root, self.job).with_suffix(".lock")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.drive()
        finally:
            os.close(descriptor)
        self.spawn.assert_not_called()

    def test_timeout_terminates_only_the_owned_child_and_retains_claim(self):
        self.terminal()
        self.child.timeout = True
        self.assertEqual(self.drive(), 124)
        self.assertEqual(self.child.terminated, 1)
        self.assertEqual(self.child.killed, 0)
        self.assertEqual(self.state()["reason"], "continuation_timeout")
        self.assertTrue(auto.claim_path(self.root, self.job).exists())
        self.spawn.assert_called_once()
        self.assertTrue((self.path.parent / "notice.md").is_file())
        self.assertTrue((self.path.parent / "launcher-notice.md").is_file())

    def test_timeout_preserves_existing_child_notice(self):
        self.terminal()
        self.child.timeout = True
        expected = "A submitted remote job still needs receipt reconciliation.\n"
        self.child.notice = lambda: (self.path.parent / "notice.md").write_text(expected)
        self.assertEqual(self.drive(), 124)
        self.assertEqual((self.path.parent / "notice.md").read_text(), expected)
        self.assertTrue((self.path.parent / "launcher-notice.md").is_file())

    def test_nonzero_codex_exit_is_process_completion_not_scientific_success(self):
        self.terminal()
        self.child.exit_code = 1
        self.assertEqual(self.drive(), 1)
        self.assertEqual(self.state()["phase"], "continuation_finished")
        self.assertEqual(self.state()["child_returncode"], 1)
        self.assertIsNone(self.state()["training_started"])

    def test_preserves_child_scientific_notice(self):
        self.terminal()
        expected = "Scientific prerequisite failed; no full teacher job submitted.\n"
        self.child.notice = lambda: (self.path.parent / "notice.md").write_text(expected)
        self.assertEqual(self.drive(), 0)
        self.assertEqual((self.path.parent / "notice.md").read_text(), expected)
        self.assertTrue(self.state()["child_notice_preserved"])
        self.assertTrue((self.path.parent / "launcher-notice.md").is_file())

    def test_terminal_evidence_change_during_ssh_check_prevents_claim(self):
        directory, _ = self.terminal()
        self.master.side_effect = lambda: (directory / watch.REPORT).write_text("tampered")
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()

    def test_actual_stdin_prompt_is_hash_checked_after_the_last_guard(self):
        self.terminal()
        inspect = auto.inspect_watcher
        inspections = [0]

        def change_after_inspection(*args):
            result = inspect(*args)
            inspections[0] += 1
            if inspections[0] == 2:
                (self.root / auto.PROMPT_FILE).write_text("Changed after final boundary check")
            return result

        with patch.object(auto, "inspect_watcher", side_effect=change_after_inspection):
            self.assertEqual(self.drive(), 1)
        self.assert_no_launch()
        self.assertIn("stdin prompt changed", self.state()["error"])

    def test_expired_plan_stops_before_any_remote_or_child_operation(self):
        self.now = self.plan["expires_at_unix"]
        self.assertEqual(self.drive(), 1)
        self.assert_no_launch()
        self.master.assert_not_called()


if __name__ == "__main__":
    unittest.main()
