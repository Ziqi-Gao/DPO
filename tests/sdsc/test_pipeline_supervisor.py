"""Finite CPU simulations of actual Quest successor state transitions."""

import base64
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pipeline_supervisor_fixture", ROOT / "tools/sdsc_pipeline.py")
flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow)


class PipelineSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / ".sdsc/supervision/fixture"
        self.directory.mkdir(parents=True)
        self.parent_state = self.root / ".sdsc/supervision/predecessor/state.json"
        self.parent_state.parent.mkdir()
        flow.s.atomic(self.parent_state, {"phase": "calibration_complete", "calibration_job_id": "101"})
        self.plan = {
            "flow_id": "fixture",
            "quest_host": "cpu-fixture",
            "run_id": "release-fixture",
            "code_sha256": "a" * 64,
            "parent_state": str(self.parent_state),
            "preflight_job_id": "90",
            "poll_seconds": 300,
            "deadline_seconds": 14 * 24 * 3600,
            "science_head": "0215c356355b29b5e2b407978a207db2156719e1",
            "protocol_sha256": "b" * 64,
            "provenance_dir": "/home/zgao12/quest-runs/OPD/provenance/fixture",
            "provenance_sha256": "c" * 64,
            "g0_python": "/approved/runtime/bin/python3.12",
            "g0_python_sha256": "d" * 64,
            "hf_home": "/approved/cache",
            "environment": {"runtime": {"python_path": "/approved/pilot/bin/python3.12"}},
            "adapter_review": {"reviewer": "independent CPU fixture"},
        }
        self.calibration = {
            "submission": {"result_dir": "/approved/calibration-result"},
            "report": {"training_artifact_root": "/scratch/zgao12/job_101/outputs/qwen3-v2/canonical_sft"},
            "publication_receipt": {
                "files": [
                    {"path": "artifacts/" + name, "size": 1, "sha256": "e" * 64}
                    for name in (
                        "initial_checkpoint.pt",
                        "canonical_sft/manifest.json",
                        "dataset/manifest.json",
                        "teacher_demos/manifest.json",
                    )
                ]
            },
        }
        self.now = 0
        self.sleeps = []
        self.calls = []
        self.status_replies = {}
        self.claimed = set()
        self.submit_errors = set()
        self.reconcile_errors = set()
        self.hooks = {}
        self.master_available = True
        self.control_valid = True
        self.master_checks = 0
        self.legacy_calls = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def master(self):
        self.master_checks += 1
        if not self.master_available:
            raise OSError("fixture SSH master unavailable; manual authentication required")

    def validate(self, plan):
        if not self.control_valid:
            raise ValueError("reviewed control bytes changed")
        return plan

    def legacy(self, _plan, _hash, job):
        self.legacy_calls.append(job)
        return copy.deepcopy(self.calibration) if job == "101" else {"job_id": job}

    def receipt(self, stage):
        return {
            "job_id": str(200 + flow.control.STAGES.index(stage)),
            "stage": stage,
            "run_id": self.plan["run_id"],
            "code_sha256": self.plan["code_sha256"],
        }

    def successful_status(self, stage):
        return {
            "state": "COMPLETED",
            "success": True,
            "job_id": self.receipt(stage)["job_id"],
            "result": {"verified": True},
            "publications": [
                {
                    "value": {
                        "passed": True,
                        "persisted": True,
                        "persistent_read_back_verified": True,
                        "delta": {
                            "pilot/" + stage + ".json": {
                                "size": 1,
                                "sha256": "f" * 64,
                                "storage": "/approved/" + stage + ".json",
                            }
                        },
                    }
                }
            ],
        }

    def call(self, action, **fields):
        stage = fields.get("stage")
        self.calls.append((action, stage))
        hook = self.hooks.get((action, stage))
        if hook:
            hook()
        if action == "exists":
            return {"claimed": stage in self.claimed}
        if action == "submit":
            self.assertEqual(fields["job_request"]["stage"], stage)
            if stage == "preflight4":
                self.assertTrue(flow.s.document(self.directory / "state.json")["g0_passed"])
            if stage == "prepare":
                self.assertTrue(
                    flow.s.document(self.directory / "state.json")["stages"]["preflight4"]["success"]
                )
            self.claimed.add(stage)
            if stage in self.submit_errors:
                raise ConnectionError("fixture receipt lost after sbatch")
            return self.receipt(stage)
        if action == "reconcile":
            if stage in self.reconcile_errors:
                raise RuntimeError("fixture ambiguous accounting; keep claim")
            return self.receipt(stage)
        if action == "status":
            replies = self.status_replies.get(stage, [])
            return copy.deepcopy(replies.pop(0)) if replies else self.successful_status(stage)
        if action == "fetch":
            raw = json.dumps({"stage": stage, "fixture": True}).encode()
            return {
                "files": [
                    {
                        "path": "result.json",
                        "size": len(raw),
                        "sha256": flow.s.sha(raw),
                        "base64": base64.b64encode(raw).decode(),
                    }
                ],
                "truncated": False,
            }
        raise AssertionError("unexpected remote action " + action)

    def run_flow(self):
        with (
            patch.object(flow, "ROOT", self.root),
            patch.object(flow, "validate_plan", side_effect=self.validate),
            patch.object(flow.cli, "require_master", side_effect=self.master),
            patch.object(flow, "legacy", side_effect=self.legacy),
        ):
            result = flow.run_flow(
                self.plan, "0" * 64, self.directory, call=self.call, sleep=self.sleep, clock=self.clock
            )
        return result, flow.s.document(self.directory / "state.json")

    def submitted(self):
        return [stage for action, stage in self.calls if action == "submit"]

    def test_full_twelve_stage_order_and_scientific_success_do_not_expand_factorial(self):
        result, state = self.run_flow()
        self.assertEqual(result, 0)
        self.assertEqual(self.submitted(), list(flow.control.STAGES))
        self.assertEqual(len(self.submitted()), 12)
        self.assertEqual(state["phase"], "complete")
        self.assertTrue(state["g0_passed"] and state["pilot_passed"])
        self.assertFalse(state["full_factorial_started"])
        self.assertEqual(self.legacy_calls, ["101", "90"])
        self.assertEqual(
            [stage for action, stage in self.calls if action == "fetch"], list(flow.control.STAGES)
        )

    def test_g0_failure_fetches_evidence_and_forbids_preflight4_or_pilot(self):
        self.status_replies["g0"] = [{"state": "FAILED", "success": False}]
        result, state = self.run_flow()
        self.assertEqual(result, 1)
        self.assertEqual(self.submitted(), ["g0"])
        self.assertIn(("fetch", "g0"), self.calls)
        self.assertFalse(state["g0_passed"] or state["pilot_passed"])
        self.assertFalse(state["jobs_cancelled"])

    def test_three_unknown_observations_stop_without_resubmission(self):
        self.status_replies["g0"] = [{"state": "UNKNOWN", "success": False}] * 3
        result, state = self.run_flow()
        self.assertEqual(result, 1)
        self.assertEqual(self.submitted(), ["g0"])
        self.assertEqual(self.calls.count(("status", "g0")), 3)
        self.assertEqual(self.sleeps, [300, 300])
        self.assertTrue(state["no_retry"])
        self.assertNotIn(("reconcile", "g0"), self.calls)

    def test_lost_submit_receipt_reconciles_once_without_second_submit(self):
        self.submit_errors.add("g0")
        result, state = self.run_flow()
        self.assertEqual(result, 0)
        self.assertEqual(self.calls.count(("submit", "g0")), 1)
        self.assertEqual(self.calls.count(("reconcile", "g0")), 1)
        self.assertTrue(state["pilot_passed"])

    def test_unresolved_receipt_stops_and_preserves_claim(self):
        self.submit_errors.add("g0")
        self.reconcile_errors.add("g0")
        result, state = self.run_flow()
        self.assertEqual(result, 1)
        self.assertEqual(self.submitted(), ["g0"])
        self.assertEqual(self.calls.count(("reconcile", "g0")), 1)
        self.assertTrue(state["no_retry"])
        self.assertIn("g0", self.claimed)

    def test_existing_remote_claim_is_only_reconciled_not_resubmitted(self):
        self.claimed.add("g0")
        result, state = self.run_flow()
        self.assertEqual(result, 0)
        self.assertNotIn(("submit", "g0"), self.calls)
        self.assertEqual(self.calls.count(("reconcile", "g0")), 1)
        self.assertTrue(state["g0_passed"])

    def test_master_lost_after_submit_exception_stops_before_reconcile(self):
        self.submit_errors.add("g0")
        self.hooks[("submit", "g0")] = lambda: setattr(self, "master_available", False)
        result, state = self.run_flow()
        self.assertEqual(result, 1)
        self.assertNotIn(("reconcile", "g0"), self.calls)
        self.assertEqual(self.submitted(), ["g0"])
        self.assertIn("manual authentication", state["error"])

    def test_stopped_predecessor_prevents_all_remote_operations(self):
        flow.s.atomic(self.parent_state, {"phase": "stopped", "error": "teacher failed"}, replace=True)
        result, state = self.run_flow()
        self.assertEqual(result, 1)
        self.assertFalse(self.calls)
        self.assertFalse(self.legacy_calls)
        self.assertIn("teacher failed", state["error"])

    def test_deadline_expiring_during_exists_does_not_start_submission(self):
        self.hooks[("exists", "g0")] = lambda: setattr(self, "now", self.plan["deadline_seconds"])
        result, state = self.run_flow()
        self.assertEqual(result, 1)
        self.assertFalse(self.submitted())
        self.assertIn("deadline", state["error"])

    def test_control_change_between_exists_and_submit_stops_before_mutation(self):
        self.hooks[("exists", "g0")] = lambda: setattr(self, "control_valid", False)
        result, state = self.run_flow()
        self.assertEqual(result, 1)
        self.assertFalse(self.submitted())
        self.assertIn("control bytes changed", state["error"])

    def test_ssh_loss_during_exists_prevents_submission(self):
        self.hooks[("exists", "g0")] = lambda: setattr(self, "master_available", False)
        result, state = self.run_flow()
        self.assertEqual(result, 1)
        self.assertFalse(self.submitted())
        self.assertIn("SSH master unavailable", state["error"])

    def test_active_stage_queries_are_low_frequency_and_do_not_duplicate_submission(self):
        self.status_replies["g0"] = [{"state": "ACTIVE", "success": False}] * 2
        result, state = self.run_flow()
        self.assertEqual(result, 0)
        self.assertEqual(self.sleeps, [300, 300])
        self.assertEqual(self.calls.count(("submit", "g0")), 1)
        self.assertTrue(state["pilot_passed"])


if __name__ == "__main__":
    unittest.main()
