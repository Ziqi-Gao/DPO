"""Capability observer boundaries; explicitly import sibling candidate fixtures.

No SSH, Slurm, model, GPU, or production-control operation is performed.
"""

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "capability_watch_fixture", Path(__file__).with_name("test_probe_watch.py")
)
fixture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_module)
watch = fixture_module.watch
REPORT = "teacher-capability-probe.json"


class CapabilityWatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.ProbeWatchTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.receipt["task"] = watch.CAPABILITY_TASK
        submission = json.loads(f.submission.read_text())
        submission["receipt"]["task"] = submission["request"]["task"] = watch.CAPABILITY_TASK
        f.submission.write_text(json.dumps(submission))
        f.plan_path, f.plan = watch.prepare(f.job, "capability-watch-fixture", root=f.root, now=100)

    def status(self, state="PENDING", success=False):
        return dict(self.fixture.status(state, success), task=watch.CAPABILITY_TASK)

    def fetched(self, metrics_passed=False):
        f = self.fixture
        fetched, status = f.fetched()
        status["task"] = watch.CAPABILITY_TASK
        directory = Path(fetched["destination"])
        original = json.loads((directory / watch.REPORT).read_text())
        (directory / watch.REPORT).unlink()
        value = 1.0 if metrics_passed else 0.0
        report = {
            **{key: original[key] for key in ("job_id", "run_id", "code_sha256", "task", "hf_home")},
            **watch.READINESS,
            "artifact_kind": "teacher_capability_diagnostic",
            "exploratory": True, "execution_class_certified": False, "resumable": False,
            "passed": True, "exit_code": 0, "metrics_passed": metrics_passed,
            "passed_means": "diagnostic completed and preserved; not scientific readiness",
            "ordered_prompt_ids": [f"fixture-{index:04d}" for index in range(128)],
            "full_generation_rows": [
                {"example_id": f"fixture-{index:04d}", "format_valid": metrics_passed,
                 "answer_correct": metrics_passed, "exact_proof_correct": metrics_passed,
                 "verification_reward": value,
                 "verification_error": None if metrics_passed else "response_syntax"}
                for index in range(128)
            ],
            "prefix_score_count": 16,
            "metrics": dict.fromkeys(watch.CAPABILITY_METRICS, value),
            "checks": dict.fromkeys(watch.CAPABILITY_CHECKS, metrics_passed),
        }
        self.seal(report, fetched, status)
        return report, fetched, status

    def seal(self, report, fetched, status):
        directory = Path(fetched["destination"])
        raw = watch.canonical(report)
        (directory / REPORT).write_bytes(raw)
        published = json.loads((directory / "receipt.json").read_text())
        published["files"] = [{"path": REPORT, "size": len(raw), "sha256": watch.digest(raw)}]
        published["passed"] = report["passed"]
        published_raw = watch.canonical(published)
        (directory / "receipt.json").write_bytes(published_raw)
        fetched.update(files=[REPORT, "receipt.json"], bytes=len(raw) + len(published_raw))
        status["result"] = {"verified": status["success"], "result_sha256": watch.digest(raw)}

    def conclude(self, fetched, status):
        f = self.fixture
        return watch.conclusion(f.plan, status, fetched, f.root)

    def test_receipt_selects_capability_task_and_unchanged_readonly_bounds(self):
        f = self.fixture
        self.assertEqual(watch.validate_plan(f.plan, f.root), f.receipt)
        self.assertEqual(f.plan["poll_seconds"], 300)
        self.assertEqual(f.plan["expires_at_unix"] - f.plan["created_at_unix"], 14 * 24 * 3600)
        self.assertEqual(f.plan["allowed_operations"], ["status", "logs", "fetch"])

    def test_diagnostic_completed_with_failed_metrics_remains_non_ready(self):
        report, fetched, status = self.fetched()
        result = self.conclude(fetched, status)
        self.assertTrue(result["diagnostic_completed"])
        self.assertFalse(result["metrics_passed"])
        self.assertEqual(result["metrics"], report["metrics"])
        self.assertEqual(result["checks"], report["checks"])
        self.assertEqual(result["generation_count"], 128)
        self.assertNotIn("candidate", result)
        self.assertTrue(all(result[key] is False for key in watch.READINESS))

    def test_successful_metrics_never_create_scientific_readiness(self):
        _, fetched, status = self.fetched(metrics_passed=True)
        result = self.conclude(fetched, status)
        self.assertTrue(result["diagnostic_completed"])
        self.assertTrue(result["metrics_passed"])
        self.assertTrue(all(result[key] is False for key in watch.READINESS))

    def test_flow_observes_once_and_does_not_submit_after_capability_pass(self):
        f = self.fixture
        report, fetched, status = self.fetched(metrics_passed=True)
        calls, writes = [], []

        def call(argv):
            calls.append(argv)
            return {"status": status, "logs": {"job_id": f.job}, "fetch": fetched}[argv[0]]

        self.assertEqual(watch.run_flow(
            f.plan, lambda name, value: writes.append((name, value)), call=call,
            guard=lambda: watch.validate_plan(f.plan, f.root),
            conclude=lambda *args: watch.conclusion(*args, root=f.root), clock=lambda: 100,
            sleep=lambda _: self.fail("terminal job must not sleep"),
        ), 0)
        self.assertEqual([argv[0] for argv in calls], ["status", "logs", "fetch"])
        self.assertEqual(writes[-1][1]["phase"], "finished")
        self.assertTrue(all(writes[-1][1][key] is False for key in watch.READINESS))
        self.assertEqual(writes[-1][1]["conclusion"]["metrics"], report["metrics"])

    def test_prompt_status_cannot_be_used_for_capability_receipt(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, "task identity"):
            f.drive([f.status("COMPLETED", True)])
        self.assertEqual(f.calls, [["status", f.job]])
        _, fetched, status = self.fetched()
        status["task"] = watch.TASK
        with self.assertRaisesRegex(ValueError, "task/job identity"):
            self.conclude(fetched, status)

    def test_unknown_task_cannot_prepare_even_when_receipt_request_agree(self):
        f = self.fixture
        value = json.loads(f.submission.read_text())
        value["receipt"]["task"] = value["request"]["task"] = "qwen3-v2-teacher-prepare"
        f.submission.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "diagnostic probe"):
            watch.prepare(f.job, "wrong-task", root=f.root)

    def test_all_formal_readiness_claims_and_missing_flags_are_rejected(self):
        report, fetched, status = self.fetched()
        for key in (*watch.READINESS, "execution_class_certified", "resumable"):
            for value in (True, None, 0):
                with self.subTest(key=key, value=value):
                    changed = dict(report, **{key: value})
                    self.seal(changed, fetched, status)
                    with self.assertRaisesRegex(ValueError, "readiness"):
                        self.conclude(fetched, status)
        self.seal(dict(report, artifact_kind="teacher_readiness"), fetched, status)
        with self.assertRaisesRegex(ValueError, "readiness"):
            self.conclude(fetched, status)

    def test_all_eight_checks_and_consistent_boolean_metrics_passed_required(self):
        report, fetched, status = self.fetched()
        for changes in (
            {"metrics_passed": True}, {"metrics_passed": 0}, {"metrics_passed": None},
            {"checks": {key: False for key in watch.CAPABILITY_CHECKS if key != "causal_shift"}},
            {"checks": dict(report["checks"], extra=False)},
            {"checks": dict(report["checks"], causal_shift=0)}, {"checks": None},
        ):
            self.seal(dict(report, **changes), fetched, status)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.conclude(fetched, status)

    def test_metrics_must_be_finite_complete_nonboolean_and_match_rows(self):
        report, fetched, status = self.fetched()
        for metrics in (
            dict(report["metrics"], minimum_causal_shift_logprob=float("nan")),
            dict(report["metrics"], minimum_causal_shift_logprob=float("inf")),
            dict(report["metrics"], minimum_causal_shift_logprob=True),
            dict(report["metrics"], minimum_topk_mass=-0.1),
            dict(report["metrics"], minimum_topk_mass=1.000002),
            dict(report["metrics"], topk_target_coverage=1.01),
            dict(report["metrics"], answer_accuracy=0.5),
            {key: value for key, value in report["metrics"].items() if key != "format_validity"},
            dict(report["metrics"], extra=0.0),
        ):
            self.seal(dict(report, metrics=metrics), fetched, status)
            with self.subTest(metrics=metrics), self.assertRaisesRegex(ValueError, "metric"):
                self.conclude(fetched, status)

    def test_float32_summed_topk_mass_roundoff_is_preserved_without_relaxing_accuracies(self):
        report, fetched, status = self.fetched(metrics_passed=True)
        # Reproduced on CPU PyTorch 2.8 float32, vocab 151936, top_k=128:
        # logits [-8.986421585083008, -0.7351513504981995, 1.1238117218017578,
        #         8.952812194824219, 2.521852493286133], remaining logits -1000.
        # logits.log_softmax(-1).exp().topk(128).values.sum(-1).min() gives this.
        observed_mass = 1.0000001192092896
        changed = dict(report, metrics=dict(report["metrics"], minimum_topk_mass=observed_mass))
        self.seal(changed, fetched, status)
        result = self.conclude(fetched, status)
        self.assertEqual(result["metrics"]["minimum_topk_mass"], observed_mass)
        self.assertEqual(result["checks"], report["checks"])
        self.assertTrue(result["metrics_passed"])
        self.assertTrue(all(result[key] is False for key in watch.READINESS))
        for metric in watch.CAPABILITY_METRICS - {"minimum_topk_mass", "minimum_causal_shift_logprob"}:
            altered = dict(report, metrics=dict(report["metrics"], **{metric: observed_mass}))
            self.seal(altered, fetched, status)
            with self.subTest(metric=metric), self.assertRaisesRegex(ValueError, "out of range"):
                self.conclude(fetched, status)

    def test_population_requires_exact_128_ordered_unique_ids(self):
        report, fetched, status = self.fetched()
        for changes in (
            {"full_generation_rows": report["full_generation_rows"][:-1]},
            {"full_generation_rows": list(reversed(report["full_generation_rows"]))},
            {"full_generation_rows": [None] * 128},
            {"ordered_prompt_ids": list(reversed(report["ordered_prompt_ids"]))},
            {"ordered_prompt_ids": [report["ordered_prompt_ids"][0]] * 128},
            {"ordered_prompt_ids": [None] * 128},
        ):
            self.seal(dict(report, **changes), fetched, status)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "population"):
                self.conclude(fetched, status)

    def test_positive_integer_prefix_score_count_required(self):
        report, fetched, status = self.fetched()
        for value in (0, -1, True, 1.0, None):
            self.seal(dict(report, prefix_score_count=value), fetched, status)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "prefix scores"):
                self.conclude(fetched, status)

    def test_partial_failed_report_preserves_error_without_completeness_claim(self):
        report, fetched, status = self.fetched()
        partial = {key: value for key, value in report.items() if key not in {
            "full_generation_rows", "ordered_prompt_ids", "metrics", "checks", "prefix_score_count"
        }}
        partial.update(passed=False, exit_code=1, error="fixture failure during prefix scoring")
        status.update(state="FAILED", success=False)
        self.seal(partial, fetched, status)
        result = self.conclude(fetched, status)
        self.assertFalse(result["diagnostic_completed"])
        self.assertEqual(result["worker_error"], partial["error"])
        self.assertNotIn("metrics_passed", result)
        self.assertTrue(all(result[key] is False for key in watch.READINESS))

    def test_hash_and_zero_exit_accounting_still_required(self):
        _, fetched, status = self.fetched()
        path = Path(fetched["destination"]) / REPORT
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "publication hash"):
            self.conclude(fetched, status)
        path.write_bytes(original)
        status["accounting"]["stdout"] = self.fixture.job + "|COMPLETED|1:0|\n"
        with self.assertRaisesRegex(ValueError, "Incomplete diagnostic"):
            self.conclude(fetched, status)

    def test_capability_ssh_loss_and_control_change_stop_without_retry(self):
        f = self.fixture
        with self.assertRaisesRegex(RuntimeError, "master unavailable"):
            f.drive([], fault=("status", "SSH master unavailable"))
        self.assertEqual(f.calls, [["status", f.job]])
        self.assertEqual(f.sleeps, [])
        path = f.root / watch.CONTROL_FILES[0]
        path.write_text("changed pinned candidate control")
        with patch.object(watch, "command") as call, self.assertRaisesRegex(ValueError, "changed"):
            watch.run_plan(f.plan_path, f.root)
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
