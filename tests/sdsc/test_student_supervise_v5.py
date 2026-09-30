"""CPU-only profile and observation timing checks; no scheduler calls."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "student_v5_fixture", ROOT / "tools/sdsc_student_supervise_v5.py"
)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class StudentV5SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = adapter.load_base()
        self.protocol = {
            "protocol_id": "qwen3-adapted-student-calibration-v5",
            "review": {"status": "accepted", "reviewed_implementation_commit": "1" * 40},
        }
        target = self.root / adapter.PROTOCOL
        target.parent.mkdir(parents=True)
        target.write_bytes(self.base.canonical(self.protocol) + b"\n")
        core = copy.deepcopy(self.protocol)
        core.pop("review")
        self.profile = {
            "schema": adapter.PROFILE_SCHEMA,
            "preflight_job_id": "54509990",
            "preflight_run_id": "new-v5-preflight",
            "science_git_head": "2" * 40,
            "student_protocol_sha256": self.base.sha(self.base.canonical(core)),
            "student_protocol_artifact_sha256": self.base.sha(target.read_bytes()),
        }
        self.path = self.root / ".sdsc/student-supervision-profiles/fixture.json"
        self.path.parent.mkdir(parents=True)
        self.save_profile()
        self.plan = {
            "flow_id": "fixture-v5",
            "expires_at_unix": 2000,
            "calibration": {
                "run_id": "fresh-calibration",
                "provenance_dir": self.base.REMOTE + "/provenance/" + "3" * 64,
                "provenance_manifest_sha256": "3" * 64,
            },
        }
        (self.root / ".sdsc/supervision/fixture-v5").mkdir(parents=True)

    def save_profile(self):
        self.path.write_bytes(self.base.canonical(self.profile) + b"\n")
        self.digest = self.base.sha(self.path.read_bytes())

    def test_explicit_profile_binds_accepted_protocol(self):
        profile, path = adapter.profile_values(self.base, self.path, self.digest, self.root)
        self.assertEqual(profile, self.profile)
        self.assertEqual(path, self.path)

    def test_profile_rejects_hash_extra_field_and_historical_preflight(self):
        with self.assertRaisesRegex(ValueError, "hash"):
            adapter.profile_values(self.base, self.path, "0" * 64, self.root)
        invalid_fields = [("command", "submit")]
        invalid_fields.extend(
            ("preflight_job_id", job)
            for job in ("54506703", "54504895", "54496291", "54507345", "54507464", "54509682", "54509809")
        )
        for key, value in invalid_fields:
            original = self.profile.copy()
            self.profile[key] = value
            self.save_profile()
            with self.assertRaises(ValueError):
                adapter.profile_values(self.base, self.path, self.digest, self.root)
            self.profile = original

    def test_proposed_or_changed_protocol_rejected(self):
        self.protocol["review"]["status"] = "proposed"
        (self.root / adapter.PROTOCOL).write_text(json.dumps(self.protocol))
        with self.assertRaisesRegex(ValueError, "accepted"):
            adapter.profile_values(self.base, self.path, self.digest, self.root)

    def test_changed_accepted_protocol_requires_new_profile(self):
        self.protocol["review"]["reviewed_implementation_commit"] = "3" * 40
        (self.root / adapter.PROTOCOL).write_bytes(self.base.canonical(self.protocol) + b"\n")
        with self.assertRaisesRegex(ValueError, "accepted v5 artifact"):
            adapter.profile_values(self.base, self.path, self.digest, self.root)

    def test_isolated_configuration_preserves_v2_and_pins_profile(self):
        old = adapter.load_base()
        with (
            patch.object(adapter, "load_base", return_value=self.base),
            patch.object(self.base, "binding", return_value={"receipt": {"fixture": True}}) as binding,
            patch.object(self.base, "validate_receipt") as validate,
        ):
            configured = adapter.configure(self.path, self.digest, self.root)
        binding.assert_called_once_with("54509990", self.root)
        validate.assert_called_once_with({"fixture": True}, calibration=False)
        self.assertEqual(configured.PREFLIGHT, "54509990")
        self.assertEqual(configured.PROTOCOL_PATH, adapter.PROTOCOL)
        self.assertEqual(
            configured.FIXED_BINDINGS["student_protocol_sha256"], self.profile["student_protocol_sha256"]
        )
        self.assertIn(str(self.path.relative_to(self.root)), configured.CONTROL_FILES)
        self.assertIn(adapter.SELF, configured.CONTROL_FILES)
        self.assertEqual(old.PREFLIGHT, "54506703")
        self.assertEqual(old.HEAD, "28c1026cece772a9e3d167d9cc64a64aaa0fd1b3")
        self.assertEqual(configured.SCHEMA, "quest-sdsc-adapted-student-supervision-v5")
        self.assertEqual(configured.POLL, 300)
        self.assertEqual(configured.MAX_SECONDS, 14 * 24 * 3600)
        self.assertEqual(
            configured.RESOURCES,
            {
                "account": "nwu181",
                "partition": "nairr-gpu-shared",
                "qos": "nairr-gpu-shared-normal",
                "gpu_type": "h100",
                "gpus": 2,
                "cpus": 24,
                "mem_gib": 192,
                "time": "02:00:00",
            },
        )

    def test_first_calibration_query_waits_once_and_preserves_raw_unknown(self):
        now = [1000]
        calls, sleeps = [], []
        submit = self.base.submit_arguments(self.plan)

        def original(arguments, _plan, _root):
            calls.append((arguments, now[0]))
            return {"job_id": "54509991"} if arguments == submit else {"state": "UNKNOWN", "success": False}

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        with (
            patch.object(self.base, "command", side_effect=original),
            patch.object(self.base, "validate_plan") as guard,
        ):
            command = adapter.command_adapter(self.base, clock=lambda: now[0], sleep=sleep)
            command(submit, self.plan, self.root)
            self.assertEqual(calls[-1][1], 1000)
            raw = command(["status", "54509991"], self.plan, self.root)
            self.assertEqual(calls[-1][1], 1300)
            command(["status", "54509991"], self.plan, self.root)
            guard.assert_called_once_with(self.plan, self.root)
        self.assertEqual(sleeps, [300])
        saved = json.loads((self.root / ".sdsc/supervision/fixture-v5/last-status.json").read_text())
        self.assertEqual(saved["status"], raw)
        self.assertEqual(sum(arguments == submit for arguments, _ in calls), 1)

    def test_expiry_or_changed_controls_during_wait_stops_before_query(self):
        for changed in (False, True):
            with self.subTest(changed=changed):
                now = [1950 if not changed else 1000]

                def sleep(seconds, now=now):
                    now[0] += seconds

                with (
                    patch.object(self.base, "command", return_value={"job_id": "54509991"}) as original,
                    patch.object(
                        self.base, "validate_plan", side_effect=ValueError("changed") if changed else None
                    ),
                ):
                    command = adapter.command_adapter(self.base, clock=lambda now=now: now[0], sleep=sleep)
                    command(self.base.submit_arguments(self.plan), self.plan, self.root)
                    with self.assertRaises(ValueError):
                        command(["status", "54509991"], self.plan, self.root)
                    self.assertEqual(original.call_count, 1)

    def test_failed_submission_is_not_repeated(self):
        with patch.object(self.base, "command", side_effect=ValueError("unknown receipt")) as original:
            command = adapter.command_adapter(self.base, sleep=lambda _seconds: self.fail("unexpected wait"))
            with self.assertRaisesRegex(ValueError, "unknown receipt"):
                command(self.base.submit_arguments(self.plan), self.plan, self.root)
            self.assertEqual(original.call_count, 1)


if __name__ == "__main__":
    unittest.main()
