"""CPU-only control and immutable transport checks; Slurm calls are mocked."""

import copy
import importlib.util
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pipeline_control_test", ROOT / "tools/sdsc_pipeline_remote.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)
storage = control.s


def response(stdout="", returncode=0):
    return {"returncode": returncode, "stdout": stdout, "stderr": ""}


class PipelineControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="opd-pipeline-control-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"flow_id": "flow-test", "run_id": "release-test", "code_sha256": "a" * 64}
        self.request = {
            "stage": "g0",
            "plan_sha256": "b" * 64,
            "job_request": {"stage": "g0", "python": "/pinned/bin/python3.12"},
        }

    def intent(self, stage="g0"):
        return {
            **self.plan,
            "stage": stage,
            "directory": str(self.root / stage),
            "job_name": "opd-pipe-opaque",
            "request_sha256": "b" * 64,
            "python": "/pinned/bin/python3.12",
            "created_at": "2026-09-18T03:14:00+00:00",
        }

    def test_scheduler_commands_have_explicit_utc_query_clock(self):
        with (
            patch.dict(os.environ, {"TZ": "America/Los_Angeles"}),
            patch.object(
                control.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")
            ) as run,
        ):
            self.assertEqual(control.command(["sacct", "--starttime", "2026-09-18T03:14:00"]), response())
        self.assertEqual(run.call_args.kwargs["env"]["TZ"], "UTC")
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def test_fixed_resources_cap_each_array_at_four_gpus_and_preserve_cleanup_margin(self):
        for stage, resources in control.resource_plan().items():
            args = control.sbatch(self.intent(stage))
            self.assertEqual(args[0], "sbatch")
            self.assertIn("--parsable", args)
            self.assertIn("--export=NONE", args)
            self.assertIn("--no-requeue", args)
            self.assertIn("--account=nwu181", args)
            self.assertIn("--cpus-per-task=24", args)
            self.assertIn("--mem=192G", args)
            self.assertIn("--signal=B:TERM@600", args)
            self.assertIn("--gpus=h100:" + str(resources["gpus"]), args)
            hours, minutes, seconds = map(int, resources["time"].split(":"))
            self.assertGreater(hours * 3600 + minutes * 60 + seconds, 600)
            concurrency = int(control.ARRAY[stage].rsplit("%", 1)[1]) if stage in control.ARRAY else 1
            self.assertLessEqual(concurrency * resources["gpus"], 4)
            partition = "nairr-gpu" if resources["gpus"] == 4 else "nairr-gpu-shared"
            self.assertIn("--partition=" + partition, args)
        self.assertEqual(control.ARRAY["train-cell"], "0-7%1")
        self.assertEqual(control.ARRAY["teacher-shard"], "0-15%4")

    def test_only_actual_complete_array_and_all_steps_can_pass(self):
        rows = "".join(
            f"123_{index}|COMPLETED|0:0|\n123_{index}.batch|COMPLETED|0:0|\n" for index in range(8)
        )
        self.assertEqual(
            control.inspect_accounting("123", "train-cell", response(), response(rows)), ("COMPLETED", True)
        )
        for extra in ("123_8|COMPLETED|0:0|\n", "123_8|FAILED|1:0|\n"):
            self.assertEqual(
                control.inspect_accounting("123", "train-cell", response(), response(rows + extra)),
                ("FAILED", False),
            )
        failed = rows.replace("123_4.batch|COMPLETED|0:0|", "123_4.batch|FAILED|1:0|")
        self.assertEqual(
            control.inspect_accounting("123", "train-cell", response(), response(failed)), ("FAILED", False)
        )
        self.assertEqual(
            control.inspect_accounting(
                "123", "train-cell", response(), response(rows.replace("123_7|COMPLETED|0:0|\n", ""))
            ),
            ("UNKNOWN", False),
        )

    def test_missing_accounting_query_failure_and_duplicate_rows_never_succeed(self):
        cases = [
            (response(), response()),
            (response(returncode=1), response()),
            (response(), response(returncode=1)),
            (response(), response("123|COMPLETED|0:0|\n123|COMPLETED|0:0|\n")),
        ]
        for queue, accounting in cases:
            with self.subTest(queue=queue, accounting=accounting):
                self.assertEqual(
                    control.inspect_accounting("123", "g0", queue, accounting), ("UNKNOWN", False)
                )
        self.assertEqual(
            control.inspect_accounting("123", "g0", response("123|RUNNING|"), response("123|COMPLETED|0:0|")),
            ("ACTIVE", False),
        )
        self.assertEqual(
            control.inspect_accounting("123", "g0", response(), response("123|RUNNING|0:0|")),
            ("UNKNOWN", False),
        )

    def test_inner_outer_stage_mismatch_fails_before_claim_or_query(self):
        request = copy.deepcopy(self.request)
        request["stage"] = "finalize"
        with (
            patch.object(control, "verify_input") as verify,
            patch.object(control, "command") as command,
            self.assertRaisesRegex(ValueError, "inner/outer"),
        ):
            control.submit(self.plan, self.root, request)
        verify.assert_not_called()
        command.assert_not_called()
        self.assertFalse((self.root / "finalize").exists())

    def test_timeout_leaves_single_durable_claim_and_recovers_real_id_without_resubmit(self):
        real_attempts = []

        def command(argv, timeout=60):
            self.assertEqual(argv[0], "sbatch")
            if "--test-only" in argv:
                return response("estimated start\n")
            real_attempts.append(argv)
            self.assertTrue((self.root / "g0/intent.json").is_file())
            self.assertTrue((self.root / "g0/submission-attempt.json").is_file())
            raise subprocess.TimeoutExpired(argv, timeout)

        with (
            patch.object(control, "verify_input", return_value=response()),
            patch.object(control, "queue_guard", return_value=response()),
            patch.object(control, "command", side_effect=command),
        ):
            with self.assertRaises(subprocess.TimeoutExpired):
                control.submit(self.plan, self.root, copy.deepcopy(self.request))
            with self.assertRaisesRegex(ValueError, "already claimed"):
                control.submit(self.plan, self.root, copy.deepcopy(self.request))
        self.assertEqual(len(real_attempts), 1)
        intent = storage.document(self.root / "g0/intent.json")
        calls = []

        def query(argv):
            calls.append(argv)
            if argv[0] == "squeue":
                return response()
            self.assertIn("2026-", argv[argv.index("--starttime") + 1])
            return response(
                f"765432|{intent['job_name']}|COMPLETED|0:0|\n765432.batch|batch|COMPLETED|0:0|\n"
            )

        with (
            patch.object(control, "checked_query", side_effect=query),
            patch.object(control, "command") as no_submit,
        ):
            receipt = control.reconcile(self.plan, self.root, "g0")
            self.assertEqual(receipt["job_id"], "765432")
            self.assertTrue(receipt["recovered"])
            self.assertEqual(control.reconcile(self.plan, self.root, "g0"), receipt)
            no_submit.assert_not_called()
        self.assertEqual(len(calls), 2)

    def test_ambiguous_response_is_saved_and_cannot_trigger_another_sbatch(self):
        with (
            patch.object(control, "verify_input", return_value=response()),
            patch.object(control, "queue_guard", return_value=response()),
            patch.object(control, "command", return_value=response("")) as command,
        ):
            with self.assertRaisesRegex(ValueError, "receipt ambiguous"):
                control.submit(self.plan, self.root, copy.deepcopy(self.request))
            with self.assertRaisesRegex(ValueError, "already claimed"):
                control.submit(self.plan, self.root, copy.deepcopy(self.request))
        self.assertEqual(sum("--test-only" not in call.args[0] for call in command.call_args_list), 1)
        self.assertEqual(storage.document(self.root / "g0/sbatch-response.json"), response())
        self.assertFalse((self.root / "g0/submission.json").exists())

    def test_unresolved_or_multiple_reconcile_candidates_keep_claim_and_never_submit(self):
        directory = self.root / "teacher-shard"
        directory.mkdir()
        intent = self.intent("teacher-shard")
        storage.atomic(directory / "intent.json", intent)
        for accounting in ("", "101|opd-pipe-opaque|RUNNING|0:0|\n102|opd-pipe-opaque|COMPLETED|0:0|\n"):
            with (
                patch.object(control, "checked_query", side_effect=[response(), response(accounting)]),
                patch.object(control, "command") as command,
                self.assertRaisesRegex(ValueError, "inconclusive"),
            ):
                control.reconcile(self.plan, self.root, "teacher-shard")
            command.assert_not_called()
            self.assertFalse((directory / "submission.json").exists())
        rows = "801_0|opd-pipe-opaque|COMPLETED|0:0|\n801_1|opd-pipe-opaque|COMPLETED|0:0|\n"
        with patch.object(control, "checked_query", side_effect=[response(), response(rows)]):
            self.assertEqual(control.reconcile(self.plan, self.root, "teacher-shard")["job_id"], "801")

    def test_rejected_resource_probe_has_no_real_submission(self):
        with (
            patch.object(control, "verify_input", return_value=response()),
            patch.object(control, "command", return_value=response(returncode=1)) as command,
            self.assertRaisesRegex(ValueError, "test-only rejected"),
        ):
            control.submit(self.plan, self.root, copy.deepcopy(self.request))
        self.assertTrue(all("--test-only" in call.args[0] for call in command.call_args_list))
        self.assertFalse((self.root / "g0/submission-attempt.json").exists())
        self.assertTrue((self.root / "g0/intent.json").is_file())

    def test_pending_opd_allocation_blocks_new_submission(self):
        with (
            patch.object(
                control,
                "checked_query",
                return_value=response("123|opd-pipe-current|PENDING|gres/gpu:h100:4|/scratch\n"),
            ),
            self.assertRaisesRegex(ValueError, "another pending/running"),
        ):
            control.queue_guard()


class PipelineStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="opd-pipeline-storage-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_immutable_publication_cannot_clobber_concurrent_writer(self):
        target = self.root / "receipt.json"
        real_link = os.link

        def race(source, destination):
            Path(destination).write_bytes(b"other-writer\n")
            real_link(source, destination)

        with patch.object(storage.os, "link", side_effect=race), self.assertRaises(FileExistsError):
            storage.atomic(target, {"passed": True})
        self.assertEqual(target.read_bytes(), b"other-writer\n")
        self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_atomic_explicit_replace_and_external_json_hash(self):
        path = self.root / "state.json"
        first = storage.atomic(path, {"state": "unknown"})
        self.assertEqual(storage.document(path, first), {"state": "unknown"})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            storage.atomic(path, {"state": "passed"})
        second = storage.atomic(path, {"state": "observed"}, replace=True)
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            storage.document(path, first)

    def test_read_hash_and_copy_reject_source_path_replacement_during_open_read(self):
        original_fstat = os.fstat
        for action in ("read", "identity", "copy"):
            path = self.root / (action + "-source")
            path.write_bytes(b"old")
            replacement = self.root / (action + "-replacement")
            replacement.write_bytes(b"new")
            info = path.stat()
            os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
            calls = 0

            def race(fd, replaced_path=path, incoming_path=replacement):
                nonlocal calls
                calls += 1
                if calls == 2:
                    os.replace(incoming_path, replaced_path)
                return original_fstat(fd)

            with (
                self.subTest(action=action),
                patch.object(storage.os, "fstat", side_effect=race),
                self.assertRaisesRegex(ValueError, "changed"),
            ):
                if action == "copy":
                    storage.copy(path, self.root / "destination")
                else:
                    getattr(storage, action)(path)

    def test_copy_checks_source_hash_and_readback_and_forbids_external_symlinks(self):
        source = self.root / "source"
        source.write_bytes(b"real-artifact")
        expected = storage.identity(source)
        destination = self.root / "copy"
        self.assertEqual(storage.copy(source, destination, expected), expected)
        source.write_bytes(b"bad-artifact!")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            storage.copy(source, self.root / "bad", expected)
        alias = self.root / "alias"
        alias.symlink_to(destination)
        with self.assertRaisesRegex(ValueError, "symlink"):
            storage.copy(alias, self.root / "outside")

    def test_publish_delta_preserves_unchanged_inputs_and_verifies_durable_changed_bytes(self):
        work = self.root / "work"
        (work / "g0").mkdir(parents=True)
        (work / "pilot").mkdir()
        old = work / "g0/input.json"
        old.write_bytes(b"unchanged")
        new = work / "pilot/new.json"
        new.write_bytes(b"result")
        before = {"g0/input.json": {**storage.identity(old), "storage": "/unused/prior"}}
        destination = self.root / "persistent"
        destination.mkdir()
        report = dict(
            flow_id="flow", stage="prepare", job_id="123", run_id="run", code_sha256="a" * 64, passed=True
        )
        with patch.object(storage, "persistent", return_value={"fstype": "lustre"}):
            receipt = storage.publish(
                {"g0": work / "g0", "pilot": work / "pilot"}, before, destination, report
            )
        self.assertEqual(set(receipt["delta"]), {"pilot/new.json"})
        self.assertTrue(receipt["persistent_read_back_verified"])
        copied = Path(receipt["delta"]["pilot/new.json"]["storage"])
        self.assertEqual(storage.identity(copied), storage.identity(new))
        self.assertEqual(storage.document(destination / "result.json", receipt["result_sha256"]), report)
        self.assertFalse((destination / "artifacts/g0/input.json").exists())

    def test_failed_copy_cannot_publish_a_success_receipt(self):
        output = self.root / "output"
        output.mkdir()
        (output / "artifact").write_text("result")
        destination = self.root / "persistent"
        destination.mkdir()
        report = dict(
            flow_id="flow", stage="prepare", job_id="123", run_id="run", code_sha256="a" * 64, passed=True
        )
        with (
            patch.object(storage, "persistent"),
            patch.object(storage, "copy", side_effect=OSError("storage lost")),
            self.assertRaisesRegex(OSError, "storage lost"),
        ):
            storage.publish({"pilot": output}, {}, destination, report)
        self.assertFalse((destination / "receipt.json").exists())

    def test_inventory_staging_rehashes_changed_inputs_before_use(self):
        project = self.root / "persistent"
        project.mkdir()
        source = project / "artifact"
        source.write_bytes(b"trusted")
        record = {**storage.identity(source), "storage": str(source)}
        source.write_bytes(b"corrupt")
        work = self.root / "work"
        work.mkdir()
        with (
            patch.object(storage, "PROJECT", project),
            patch.object(storage.shutil, "disk_usage", return_value=types.SimpleNamespace(free=1024**4)),
            self.assertRaisesRegex(ValueError, "SHA mismatch"),
        ):
            storage.stage_inventory({"g0/data": record}, {"g0": work / "g0"})

    def test_array_deltas_merge_identical_bytes_but_reject_conflicts_and_failed_publication(self):
        one = dict(
            passed=True,
            persisted=True,
            delta={"pilot/model": {"sha256": "a" * 64, "size": 1, "storage": "/one"}},
        )
        two = copy.deepcopy(one)
        two["delta"]["pilot/model"]["storage"] = "/two"
        merged = storage.merge_receipts({}, [one, two])
        self.assertEqual(merged["pilot/model"], one["delta"]["pilot/model"])
        self.assertEqual(merged["pilot/model"]["storage"], "/one")
        for difference in ({"sha256": "b" * 64}, {"size": 2}):
            conflicting = copy.deepcopy(two)
            conflicting["delta"]["pilot/model"].update(difference)
            with self.subTest(difference=difference), self.assertRaisesRegex(ValueError, "collision"):
                storage.merge_receipts({}, [one, conflicting])
        with self.assertRaisesRegex(ValueError, "failed stage"):
            storage.merge_receipts({}, [{**one, "passed": False}])


if __name__ == "__main__":
    unittest.main()
