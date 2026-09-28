"""No network or real scheduler calls: exercise the remote trust boundary."""

import ast
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "sdsc_remote", Path(__file__).resolve().parents[2] / "tools/sdsc_remote.py"
)
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


class RemoteBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "control"
        self.results = self.base / "persistent"
        self.results.mkdir()
        self.root_patch = patch.object(remote, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.request = {
            "root": str(self.root),
            "run_id": "test-run",
            "intent_id": "a" * 32,
            "task": "gpu-smoke",
            "authorized": True,
            "storage_confirmed": True,
            "python": sys.executable,
            "result_root": str(self.results),
            "resources": {
                "account": "nwu181",
                "partition": "nairr-gpu-shared",
                "qos": "nairr-gpu-shared-normal",
                "gpu_type": "h100",
                "gpus": 1,
                "cpus": 4,
                "mem_gib": 16,
                "time": "00:05:00",
            },
        }

    def archive(self, files=None, extra=None):
        files = files or {
            "tools/sdsc_job.sh": b"#!/bin/bash\nexit 0\n",
            "tools/sdsc_smoke.py": b"# Fixture only: no GPU execution\n",
            "tools/sdsc_preflight_job.sh": b"#!/bin/bash\nexit 0\n",
            "tools/sdsc_training_preflight.py": b"# Fixture only: no training execution\n",
            "src/changed.py": b'print("uncommitted")\n',
        }
        records = [
            {
                "path": name,
                "size": len(content),
                "sha256": remote.digest(content),
                "mode": 0o755 if name.endswith(".sh") else 0o644,
            }
            for name, content in sorted(files.items())
        ]
        manifest = {
            "schema": "quest-sdsc-snapshot-v1",
            "project": "OPD",
            "run_id": self.request["run_id"],
            "files": records,
            "code_sha256": remote.digest(remote.canonical(records)),
            "total_bytes": sum(item["size"] for item in records),
        }
        self.request["code_sha256"] = manifest["code_sha256"]
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:") as archive:
            for name, contents, mode in [("manifest.json", remote.canonical(manifest), 0o644)] + [
                ("source/" + record["path"], files[record["path"]], record["mode"]) for record in records
            ]:
                member = tarfile.TarInfo(name)
                member.mode, member.size = mode, len(contents)
                archive.addfile(member, io.BytesIO(contents))
            if extra:
                archive.addfile(extra)
        stream.seek(0)
        return stream

    def deploy(self):
        return remote.upload(self.request, self.archive())

    def submit(self, result=None):
        self.deploy()
        result = result or {"returncode": 0, "stdout": "12345;expanse\n", "stderr": ""}
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", return_value=result) as run,
        ):
            receipt = remote.submit(self.request)
        return receipt, run

    def bound(self):
        receipt, unused = self.submit()
        self.request["job_id"] = receipt["job_id"]
        return receipt

    def container(self):
        image = self.base / "image with spaces;$(not-executed).sif"
        image.write_bytes(b"Existing container fixture; never executed or transferred")
        return remote.current_container_identity(
            {"runtime": sys.executable, "image": str(image), "python": "/container/bin/python"}
        )

    def publish(self, receipt):
        directory = Path(receipt["result_dir"])
        identity = {key: receipt[key] for key in ("run_id", "job_id", "code_sha256")}
        result = dict(
            identity, passed=True, infrastructure_smoke=True, exit_code=0, container=receipt.get("container")
        )
        data = remote.canonical(result)
        (directory / "result.json").write_bytes(data)
        remote.atomic_json(
            directory / "receipt.json",
            dict(
                identity,
                persisted=True,
                passed=True,
                persistent_read_back_verified=True,
                files=[{"path": "result.json", "sha256": remote.digest(data), "size": len(data)}],
            ),
        )

    def test_upload_preserves_uncommitted_content_and_refuses_replacement(self):
        receipt = self.deploy()
        release = Path(receipt["release"])
        self.assertEqual((release / "source/src/changed.py").read_bytes(), b'print("uncommitted")\n')
        with self.assertRaisesRegex(ValueError, "immutable"):
            remote.upload(self.request, self.archive())

    def test_upload_rejects_traversal_before_writing(self):
        with self.assertRaisesRegex(ValueError, "Unsafe source path"):
            remote.upload(self.request, self.archive({"../escape": b"no"}))
        self.assertFalse(self.root.exists())

    def test_upload_rejects_secret_and_model_paths(self):
        for name in (
            "auth.json",
            ".ssh/key",
            ".env.production",
            "nested/.env.example",
            "model.safetensors",
            "checkpoints/model.txt",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Excluded"):
                remote.upload(self.request, self.archive({name: b"not transferred"}))

    def test_checked_in_environment_template_can_be_deployed(self):
        response = remote.upload(self.request, self.archive({".env.example": b"EXAMPLE=placeholder\n"}))
        self.assertTrue((Path(response["release"]) / "source/.env.example").is_file())

    def test_upload_rejects_symlink_archive_member(self):
        link = tarfile.TarInfo("source/link")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        with self.assertRaisesRegex(ValueError, "regular files only"):
            remote.upload(self.request, self.archive(extra=link))

    def test_upload_rejects_unlisted_tar_entries(self):
        extra = tarfile.TarInfo("source/unlisted")
        with self.assertRaisesRegex(ValueError, "inventory"):
            remote.upload(self.request, self.archive(extra=extra))

    def test_upload_rejects_wrong_manifest_hash(self):
        stream = self.archive()
        self.request["code_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            remote.upload(self.request, stream)

    def test_snapshot_mutation_blocks_submission(self):
        receipt = self.deploy()
        (Path(receipt["release"]) / "source/src/changed.py").write_text("changed after transfer")
        with patch.object(remote, "run") as run, self.assertRaisesRegex(ValueError, "source changed"):
            remote.submit(self.request)
        run.assert_not_called()

    def test_unbound_worker_files_block_before_submission(self):
        remote.upload(self.request, self.archive({"tools/sdsc_job.sh": b"#!/bin/bash\nexit 0\n"}))
        with patch.object(remote, "run") as run, self.assertRaisesRegex(ValueError, "both GPU smoke"):
            remote.submit(self.request)
        run.assert_not_called()

    def test_failed_cancel_is_reported_without_retry(self):
        self.bound()
        with (
            patch.object(
                remote, "run", return_value={"returncode": 1, "stdout": "", "stderr": "failure"}
            ) as run,
            self.assertRaisesRegex(ValueError, "Cancellation failed"),
        ):
            remote.cancel(self.request)
        self.assertEqual(run.call_count, 1)

    def test_submission_requires_authorization_and_explicit_resources(self):
        with patch.object(remote, "run") as run:
            for changed in (
                {"authorized": False},
                {"storage_confirmed": False},
                {"resources": dict(self.request["resources"], account="expanse_nairr_gpu")},
                {"resources": dict(self.request["resources"], qos="nairr-gpu-shared")},
                {"resources": dict(self.request["resources"], qos="unverified-qos")},
                {"resources": dict(self.request["resources"], time="00:05:01")},
                {"resources": dict(self.request["resources"], gpus=True)},
            ):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    remote.submit(dict(self.request, **changed))
        run.assert_not_called()

    def test_receipt_persists_identity_and_allocation_before_return(self):
        receipt, run = self.submit()
        self.assertEqual(receipt["job_id"], "12345")
        self.assertEqual(remote.read_json(Path(receipt["submission_dir"]) / "receipt.json"), receipt)
        argv = run.call_args[0][0]
        self.assertIn("--account=nwu181", argv)
        self.assertIn("--qos=nairr-gpu-shared-normal", argv)
        self.assertIn("--gpus=h100:1", argv)
        self.assertIn("--no-requeue", argv)
        self.assertNotIn("--constraint=lustre", argv)
        self.assertEqual(argv[-2:], [self.request["run_id"], self.request["code_sha256"]])
        intent = remote.read_json(Path(receipt["submission_dir"]) / "intent.json")
        self.assertEqual(argv, remote.build_sbatch_argv(intent, self.root / "releases" / receipt["run_id"]))
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run") as second,
            self.assertRaisesRegex(ValueError, "reconcile"),
        ):
            remote.submit(self.request)
        second.assert_not_called()

    def test_concurrent_intents_for_one_snapshot_only_submit_once(self):
        self.deploy()
        response = {"returncode": 0, "stdout": "12345\n", "stderr": ""}

        def attempt(intent_id):
            try:
                return remote.submit(dict(self.request, intent_id=intent_id))
            except ValueError as error:
                return {"error": str(error)}

        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", return_value=response) as run,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(attempt, ["a" * 32, "b" * 32]))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(sum(result.get("job_id") == "12345" for result in results), 1)
        self.assertEqual(sum("claimed" in result.get("error", "") for result in results), 1)

    def test_only_project_smoke_metadata_may_use_home(self):
        self.deploy()
        permitted = self.root / "smoke-results"
        forbidden = self.root / "other-results"
        permitted.mkdir(exist_ok=True)
        forbidden.mkdir()
        with (
            patch.object(remote.Path, "home", return_value=self.base),
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(
                remote, "run", return_value={"returncode": 0, "stdout": "12345\n", "stderr": ""}
            ) as run,
        ):
            with self.assertRaisesRegex(ValueError, "HOME permits only"):
                remote.submit(dict(self.request, result_root=str(forbidden)))
            run.assert_not_called()
            self.assertEqual(remote.submit(dict(self.request, result_root=str(permitted)))["job_id"], "12345")

    def test_timeout_leaves_durable_intent_without_blind_retry(self):
        self.deploy()
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", side_effect=subprocess.TimeoutExpired(["sbatch"], 45)) as run,
        ):
            response = remote.submit(self.request)
        self.assertEqual(response["state"], "unknown")
        self.assertEqual(run.call_count, 1)
        directory = self.root / "submissions" / self.request["intent_id"]
        self.assertTrue((directory / "intent.json").exists())
        self.assertFalse((directory / "receipt.json").exists())
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run") as second,
            self.assertRaisesRegex(ValueError, "reconcile"),
        ):
            remote.submit(self.request)
        second.assert_not_called()

    def test_reconcile_recovers_exact_named_job_without_submitting(self):
        self.submit({"returncode": 1, "stdout": "", "stderr": "transport lost"})
        name = "opd-" + self.request["intent_id"]
        responses = [
            {"returncode": 0, "stdout": "13579|" + name + "\n", "stderr": ""},
            {"returncode": 0, "stdout": "", "stderr": ""},
        ]
        with patch.object(remote, "run", side_effect=responses) as run:
            receipt = remote.reconcile(self.request)
        self.assertEqual(receipt["job_id"], "13579")
        self.assertTrue(receipt["recovered"])
        self.assertEqual([call.args[0][0] for call in run.call_args_list], ["squeue", "sacct"])

    def test_reconcile_zero_matches_remains_unknown(self):
        self.submit({"returncode": 1, "stdout": "", "stderr": "lost"})
        with patch.object(remote, "run", return_value={"returncode": 0, "stdout": "", "stderr": ""}):
            self.assertEqual(remote.reconcile(self.request)["state"], "unknown")

    def test_missing_intent_never_authorizes_resubmitting_consumed_run(self):
        self.deploy()
        remote.claim_run(self.root, self.request)
        with patch.object(remote, "run") as run:
            self.assertEqual(remote.reconcile(self.request)["state"], "unknown")
            with (
                patch.object(remote, "TEMPORARY_ROOTS", ()),
                self.assertRaisesRegex(ValueError, "original intent " + self.request["intent_id"]),
            ):
                remote.submit(dict(self.request, intent_id="b" * 32))
        run.assert_not_called()

    def test_queue_disappearance_and_missing_accounting_are_not_success(self):
        receipt = self.bound()
        self.publish(receipt)
        with patch.object(remote, "run", return_value={"returncode": 0, "stdout": "", "stderr": ""}):
            status = remote.status(self.request)
        self.assertFalse(status["success"])
        self.assertEqual(status["state"], "UNKNOWN")

    def test_status_requires_both_accounting_and_hash_valid_results(self):
        receipt = self.bound()
        self.publish(receipt)
        responses = [
            {"returncode": 0, "stdout": "", "stderr": ""},
            {"returncode": 0, "stdout": "12345|COMPLETED|0:0|00:01:00||\n", "stderr": ""},
        ]
        with patch.object(remote, "run", side_effect=responses):
            self.assertTrue(remote.status(self.request)["success"])
        (Path(receipt["result_dir"]) / "result.json").write_text("{}")
        with patch.object(remote, "run", side_effect=responses):
            self.assertFalse(remote.status(self.request)["success"])

    def test_failed_accounting_is_not_success_even_with_valid_artifact(self):
        receipt = self.bound()
        self.publish(receipt)
        with patch.object(
            remote,
            "run",
            return_value={"returncode": 0, "stdout": "12345|FAILED|1:0|00:00:04||\n", "stderr": ""},
        ):
            self.assertFalse(remote.status(self.request)["success"])

    def test_queued_job_or_failed_batch_step_cannot_claim_success(self):
        receipt = self.bound()
        self.publish(receipt)
        for queue, accounting in (
            ("12345|COMPLETING|node\n", "12345|COMPLETED|0:0||\n"),
            ("", "12345|COMPLETED|0:0||\n12345.batch|FAILED|1:0||\n"),
        ):
            with (
                self.subTest(queue=queue, accounting=accounting),
                patch.object(
                    remote,
                    "run",
                    side_effect=[
                        {"returncode": 0, "stdout": queue, "stderr": ""},
                        {"returncode": 0, "stdout": accounting, "stderr": ""},
                    ],
                ),
            ):
                self.assertFalse(remote.status(self.request)["success"])

    def test_unrelated_active_jobs_neither_block_success_nor_leak_rows(self):
        receipt = self.bound()
        self.publish(receipt)
        with patch.object(
            remote,
            "run",
            side_effect=[
                {"returncode": 0, "stdout": "99999|RUNNING|unrelated-node\n", "stderr": ""},
                {"returncode": 0, "stdout": "12345|COMPLETED|0:0||\n", "stderr": ""},
            ],
        ) as run:
            result = remote.status(self.request)
        self.assertTrue(result["success"])
        self.assertEqual(result["queue"]["stdout"], "")
        self.assertNotIn("--jobs=12345", run.call_args_list[0].args[0])

    def test_queue_query_failure_is_unknown_despite_completed_accounting(self):
        receipt = self.bound()
        self.publish(receipt)
        with patch.object(
            remote,
            "run",
            side_effect=[
                {"returncode": 1, "stdout": "", "stderr": "controller unavailable"},
                {"returncode": 0, "stdout": "12345|COMPLETED|0:0||\n", "stderr": ""},
            ],
        ):
            result = remote.status(self.request)
        self.assertFalse(result["success"])
        self.assertEqual(result["state"], "UNKNOWN")

    def test_failed_publication_or_nonzero_result_exit_is_not_verified(self):
        receipt = self.bound()
        for changed in ("receipt", "result"):
            self.publish(receipt)
            directory = Path(receipt["result_dir"])
            if changed == "receipt":
                value = remote.read_json(directory / "receipt.json")
                value["passed"] = False
                remote.atomic_json(directory / "receipt.json", value)
            else:
                result = remote.read_json(directory / "result.json")
                result["exit_code"] = 1
                data = remote.canonical(result)
                (directory / "result.json").write_bytes(data)
                publication = remote.read_json(directory / "receipt.json")
                publication["files"][0].update(size=len(data), sha256=remote.digest(data))
                remote.atomic_json(directory / "receipt.json", publication)
            self.assertFalse(remote.verify_result(receipt)["verified"])

    def test_cancel_is_bound_to_receipt_and_requires_authorization(self):
        self.bound()
        with patch.object(remote, "run") as run:
            for changed in ({"authorized": False}, {"job_id": "99999"}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    remote.cancel(dict(self.request, **changed))
            run.assert_not_called()
            run.return_value = {"returncode": 0, "stdout": "", "stderr": ""}
            remote.cancel(self.request)
        self.assertEqual(run.call_args[0][0], ["scancel", "--", "12345"])

    def test_fetch_returns_only_small_allowlisted_files_and_bounded_log_tails(self):
        receipt = self.bound()
        self.publish(receipt)
        directory = Path(receipt["result_dir"])
        (directory / "worker.log").write_text("line\n" * 100000)
        (directory / "checkpoint.pt").write_bytes(b"never fetch")
        fetched = remote.fetch(self.request)
        self.assertEqual(
            {item["path"] for item in fetched["files"]}, {"result.json", "receipt.json", "worker.log"}
        )
        self.assertLess(fetched["total_bytes"], 65536)
        self.assertEqual(remote.logs(dict(self.request, lines=5))["logs"]["worker.log"], "line\n" * 5)

    def test_fetch_refuses_symlink_result(self):
        receipt = self.bound()
        (Path(receipt["result_dir"]) / "result.json").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            remote.fetch(self.request)

    def test_remote_check_does_not_create_directories_or_invoke_scheduler(self):
        with patch.object(remote.shutil, "which", return_value=None), patch.object(remote, "run") as run:
            result = remote.check({"root": str(self.root)})
        self.assertFalse(result["storage_verified_on_gpu_node"])
        self.assertFalse(self.root.exists())
        run.assert_not_called()

    def test_check_uses_read_only_association_and_correct_quota_resource(self):
        def which(name):
            return "/commands/" + name if name in ("sacctmgr", "expanse-client") else None

        with (
            patch.object(remote.shutil, "which", side_effect=which),
            patch.object(remote, "run", return_value={"returncode": 0, "stdout": "", "stderr": ""}) as run,
        ):
            remote.check(self.request)
        self.assertEqual(run.call_args_list[0].args[0][:3], ["/commands/sacctmgr", "show", "assoc"])
        self.assertIn("format=Account,Partition,DefaultQOS,QOS%180", run.call_args_list[0].args[0])
        self.assertEqual(
            run.call_args_list[1].args[0], ["/commands/expanse-client", "user", "-r", "expanse_nairr_gpu"]
        )
        self.assertFalse(self.root.exists())

    def test_live_qos_evidence_intersects_account_and_partition_permissions(self):
        def which(name):
            return "/commands/" + name if name in ("sacctmgr", "scontrol") else None

        responses = [
            {
                "returncode": 0,
                "stdout": "nwu181||nairr-gpu-shared-normal|nairr-gpu-debug-normal,nairr-gpu-normal,"
                "nairr-gpu-shared-normal\n",
                "stderr": "",
            },
            {
                "returncode": 0,
                "stdout": "PartitionName=nairr-gpu-shared\n"
                " AllowQos=nairr-gpu-shared-normal,nairr-gpu-shared-eot\n",
                "stderr": "",
            },
        ]
        with (
            patch.object(remote.shutil, "which", side_effect=which),
            patch.object(remote, "run", side_effect=responses) as run,
        ):
            report = remote.check({"root": str(self.root)})
        evidence = report["qos_evidence"]
        self.assertTrue(evidence["verified"])
        self.assertFalse(evidence["initial_qos_jointly_allowed"])
        self.assertTrue(evidence["configured_qos_jointly_allowed"])
        self.assertEqual(evidence["jointly_allowed_qos"], ["nairr-gpu-shared-normal"])
        self.assertEqual(report["slurm_partition"], responses[1])
        self.assertEqual(
            run.call_args_list[1].args[0], ["/commands/scontrol", "show", "partition", "nairr-gpu-shared"]
        )
        self.assertEqual(run.call_args_list[1].kwargs["timeout"], 10)
        self.assertFalse(self.root.exists())

    def test_missing_qos_evidence_stays_unverified(self):
        for association, partition in (({}, {}), ({"returncode": 0, "stdout": ""}, {"returncode": 1})):
            evidence = remote.qos_evidence(association, partition)
            self.assertFalse(evidence["verified"])
            self.assertIsNone(evidence["configured_qos_jointly_allowed"])

    def test_accidental_local_dispatch_cannot_reach_scheduler(self):
        with (
            patch.object(remote, "REMOTE_USER", "not-the-current-user"),
            patch.object(remote, "run") as run,
            self.assertRaisesRegex(ValueError, "Unexpected remote user"),
        ):
            remote.dispatch(dict(self.request, action="submit"))
        run.assert_not_called()

    def test_expected_user_without_ssh_still_cannot_dispatch(self):
        current_user = remote.pwd.getpwuid(remote.os.getuid()).pw_name
        with (
            patch.object(remote, "REMOTE_USER", current_user),
            patch.dict(remote.os.environ, {}, clear=True),
            patch.object(remote, "run") as run,
            self.assertRaisesRegex(ValueError, "require an SSH connection"),
        ):
            remote.dispatch(dict(self.request, action="submit"))
        run.assert_not_called()

    def test_candidate_environment_check_reads_metadata_without_importing_torch(self):
        with (
            patch.object(remote.shutil, "which", return_value=None),
            patch.object(remote, "run", return_value={"returncode": 0, "stdout": "{}", "stderr": ""}) as run,
        ):
            report = remote.check(self.request)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], [sys.executable, "-I", "-c"])
        self.assertIn("importlib.metadata", argv[3])
        self.assertNotIn("import torch", argv[3])
        self.assertFalse(report["candidate_storage"]["gpu_node_verified"])
        self.assertFalse(report["candidate_storage"]["persistent_storage_confirmed"])
        self.assertFalse(self.root.exists())

    def test_optional_check_timeout_keeps_other_evidence(self):
        with (
            patch.object(remote.shutil, "which", return_value=None),
            patch.object(remote, "run", side_effect=subprocess.TimeoutExpired(["python"], 15)),
        ):
            report = remote.check(self.request)
        self.assertFalse(report["candidate_runtime"]["verified"])
        self.assertIn("candidate_storage", report)

    def test_subprocess_contract_has_no_shell_or_stdin_inheritance(self):
        completed = subprocess.CompletedProcess(["squeue"], 0, b"ok", b"")
        with patch.object(remote.subprocess, "run", return_value=completed) as run:
            remote.run(["squeue", "--jobs=12345"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_container_inspection_uses_only_metadata_without_nv_or_image_read(self):
        identity = self.container()
        request = {
            "root": str(self.root),
            "container": {key: identity[key] for key in remote.CONTAINER_PATH_KEYS},
        }
        with (
            patch.object(remote.shutil, "which", return_value=None),
            patch.object(remote, "run", return_value={"returncode": 0, "stdout": "{}", "stderr": ""}) as run,
            patch.object(Path, "read_bytes", side_effect=AssertionError("must not read the large image")),
        ):
            report = remote.check(request)
        self.assertEqual(report["candidate_container"], identity)
        argv = run.call_args.args[0]
        self.assertEqual(
            argv[:6],
            [identity["runtime"], "exec", "--cleanenv", "--no-home", identity["image"], identity["python"]],
        )
        self.assertNotIn("--nv", argv)
        self.assertNotIn("import torch", argv[-1])
        self.assertEqual(run.call_args.kwargs["timeout"], 25)
        self.assertFalse(self.root.exists())

    def test_container_identity_and_receipt_bind_exact_safe_seventh_argument(self):
        self.request["container"] = self.container()
        receipt, run = self.submit()
        self.assertEqual(receipt["container"], self.request["container"])
        self.assertEqual(json.loads(run.call_args.args[0][-1]), self.request["container"])
        self.assertEqual(run.call_args.args[0][-3:-1], [self.request["run_id"], self.request["code_sha256"]])
        intent = remote.read_json(Path(receipt["submission_dir"]) / "intent.json")
        self.assertEqual(intent["container"], self.request["container"])
        self.request["job_id"] = receipt["job_id"]
        self.publish(receipt)
        self.assertTrue(remote.verify_result(receipt)["verified"])
        mismatched = dict(receipt, container=dict(receipt["container"], size=1))
        self.assertFalse(remote.verify_result(mismatched)["verified"])

    def test_stale_container_image_blocks_before_claim_and_sbatch(self):
        self.deploy()
        self.request["container"] = self.container()
        Path(self.request["container"]["image"]).write_bytes(b"changed image")
        with (
            patch.object(remote, "run") as run,
            self.assertRaisesRegex(ValueError, "changed since inspection"),
        ):
            remote.submit(self.request)
        run.assert_not_called()
        self.assertFalse((self.root / "run-claims").exists())

    def test_container_image_symlink_and_extra_fields_are_rejected(self):
        identity = self.container()
        link = self.base / "latest.sif"
        link.symlink_to(identity["image"])
        with self.assertRaisesRegex(ValueError, "Symlink"):
            remote.validate_container(dict(identity, image=str(link)), verify_image=True)
        with self.assertRaisesRegex(ValueError, "exactly"):
            remote.validate_container(dict(identity, arbitrary_option="--bind=/"))

    def enable_preflight(self):
        self.request.update(task="qwen3-v2-preflight", hf_home=str(self.results / "huggingface"))
        self.request["resources"].update(gpus=2, cpus=24, mem_gib=192, time="02:00:00")
        for model, revision in remote.PREFLIGHT_MODELS:
            path = Path(self.request["hf_home"]) / "hub" / model / "snapshots" / revision
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
        patcher = patch.object(remote, "PREFLIGHT_STORAGE", self.results)
        patcher.start()
        self.addCleanup(patcher.stop)

    def publish_preflight(self, receipt, **changes):
        directory = Path(receipt["result_dir"])
        identity = {key: receipt[key] for key in ("run_id", "job_id", "code_sha256", "task", "hf_home")}
        result = dict(
            identity, passed=True, exit_code=0, world_size=2, g0_passed=False, execution_class_certified=False
        )
        result.update(changes)
        data = remote.canonical(result)
        (directory / "preflight.json").write_bytes(data)
        remote.atomic_json(
            directory / "receipt.json",
            dict(
                identity,
                passed=True,
                persisted=True,
                persistent_read_back_verified=True,
                files=[dict(path="preflight.json", sha256=remote.digest(data), size=len(data))],
            ),
        )

    def test_preflight_receipt_binds_fixed_profile_worker_and_model_cache(self):
        self.enable_preflight()
        receipt, run = self.submit()
        self.assertEqual(receipt["task"], "qwen3-v2-preflight")
        self.assertEqual(receipt["hf_home"], self.request["hf_home"])
        argv = run.call_args.args[0]
        for value in ("--gpus=h100:2", "--cpus-per-task=24", "--mem=192G", "--time=02:00:00"):
            self.assertIn(value, argv)
        self.assertEqual(argv[-1], receipt["hf_home"])
        self.assertTrue(any(value.endswith("/sdsc_preflight_job.sh") for value in argv))
        self.assertEqual(
            argv,
            remote.build_sbatch_argv(
                remote.read_json(Path(receipt["submission_dir"]) / "intent.json"),
                self.root / "releases" / self.request["run_id"],
            ),
        )

    def test_preflight_forbids_containers_wrong_resources_and_unconfirmed_cache(self):
        self.enable_preflight()
        self.deploy()
        cases = [
            dict(container=self.container()),
            dict(authorized=False),
            dict(hf_home="/home/zgao12/hf"),
            dict(result_root=str(self.root / "smoke-results")),
            dict(resources=dict(self.request["resources"], gpus=3)),
            dict(resources=dict(self.request["resources"], time="02:00:01")),
        ]
        for changes in cases:
            with (
                self.subTest(changes=changes),
                patch.object(remote, "TEMPORARY_ROOTS", ()),
                patch.object(remote, "run") as run,
                self.assertRaises(ValueError),
            ):
                remote.submit(dict(self.request, **changes))
            run.assert_not_called()
            self.assertFalse((self.root / "run-claims").exists())

    def test_preflight_missing_snapshot_refuses_submission_before_claim(self):
        self.enable_preflight()
        self.deploy()
        model, revision = remote.PREFLIGHT_MODELS[1]
        snapshot = Path(self.request["hf_home"]) / "hub" / model / "snapshots" / revision
        (snapshot / "config.json").unlink()
        snapshot.rmdir()
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run") as run,
            self.assertRaisesRegex(ValueError, "snapshot is missing"),
        ):
            remote.submit(self.request)
        run.assert_not_called()
        self.assertFalse((self.root / "run-claims").exists())

    def test_preflight_success_requires_accounting_identity_publication_and_no_g0_claim(self):
        self.enable_preflight()
        receipt = self.bound()
        responses = [
            {"returncode": 0, "stdout": "", "stderr": ""},
            {"returncode": 0, "stdout": "12345|COMPLETED|0:0||\n", "stderr": ""},
        ]
        self.publish_preflight(receipt)
        with patch.object(remote, "run", side_effect=responses):
            self.assertTrue(remote.status(self.request)["success"])
        for changes in (
            dict(task="gpu-smoke"),
            dict(hf_home="/wrong"),
            dict(job_id="987"),
            dict(passed=False),
            dict(exit_code=1),
            dict(g0_passed=True),
            dict(execution_class_certified=True),
            dict(world_size=1),
        ):
            self.publish_preflight(receipt, **changes)
            self.assertFalse(remote.verify_result(receipt)["verified"], changes)
        self.publish_preflight(receipt)
        (Path(receipt["result_dir"]) / "preflight.json").write_text("{}")
        self.assertFalse(remote.verify_result(receipt)["verified"])

    def test_preflight_result_fetch_only_selects_small_report_receipt_and_logs(self):
        self.enable_preflight()
        receipt = self.bound()
        self.publish_preflight(receipt)
        directory = Path(receipt["result_dir"])
        (directory / "worker.log").write_text("bounded\n" * 100000)
        (directory / "result.json").write_text("{}")
        (directory / "checkpoint.pt").write_bytes(b"never fetch")
        result = remote.fetch(self.request)
        self.assertEqual(
            {item["path"] for item in result["files"]}, {"preflight.json", "receipt.json", "worker.log"}
        )
        self.assertTrue(result["result"]["verified"])

    def test_preflight_receipt_cache_tampering_blocks_status_before_scheduler(self):
        self.enable_preflight()
        receipt = self.bound()
        path = Path(receipt["submission_dir"]) / "receipt.json"
        remote.atomic_json(path, dict(receipt, hf_home="/changed"))
        with patch.object(remote, "run") as run, self.assertRaisesRegex(ValueError, "binding mismatch"):
            remote.status(self.request)
        run.assert_not_called()

    def preflight_job_functions(self):
        script = (Path(__file__).resolve().parents[2] / "tools/sdsc_preflight_job.sh").read_text()
        embedded = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        module = ast.parse(embedded)
        definitions = ast.Module(
            body=[
                node
                for node in module.body
                if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)
            ],
            type_ignores=[],
        )
        namespace = {"models": remote.PREFLIGHT_MODELS}
        exec(compile(definitions, "sdsc_preflight_job.sh", "exec"), namespace)
        return namespace

    def test_preflight_job_stages_exact_source_and_detects_mutation(self):
        self.deploy()
        helpers = self.preflight_job_functions()
        helpers.update(code_hash=self.request["code_sha256"], run_id=self.request["run_id"])
        release = self.root / "releases" / self.request["run_id"]
        stage = self.base / "stage"
        stage.mkdir()
        helpers["stage_source"](release, stage)
        self.assertTrue((stage / "tools/sdsc_training_preflight.py").is_file())
        (release / "source/src/changed.py").write_text("mutated")
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            helpers["stage_source"](release, stage)

    def test_preflight_job_materializes_internal_model_links_and_rejects_external_links(self):
        self.enable_preflight()
        cache = Path(self.request["hf_home"])
        model, revision = remote.PREFLIGHT_MODELS[0]
        snapshot = cache / "hub" / model / "snapshots" / revision
        blob = cache / "hub" / model / "blobs" / "weight"
        blob.parent.mkdir()
        blob.write_bytes(b"fixture model")
        (snapshot / "model.safetensors").symlink_to(blob)
        helpers = self.preflight_job_functions()
        with patch.object(helpers["shutil"], "disk_usage", return_value=SimpleNamespace(free=128 * 1024**3)):
            staged = self.base / "staged-hf"
            helpers["stage_models"](cache, staged)
            target = staged / "hub" / model / "snapshots" / revision / "model.safetensors"
            self.assertEqual(target.read_bytes(), blob.read_bytes())
            self.assertFalse(target.is_symlink())
            (snapshot / "model.safetensors").unlink()
            (snapshot / "model.safetensors").symlink_to(self.base / "outside")
            (self.base / "outside").write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "escapes"):
                helpers["stage_models"](cache, self.base / "other-stage")

    def test_preflight_job_publishes_checkpoint_and_readback_receipt_without_fetching_it(self):
        self.enable_preflight()
        receipt = self.bound()
        helpers = self.preflight_job_functions()
        identity = {key: receipt[key] for key in ("task", "job_id", "run_id", "code_sha256", "hf_home")}
        helpers.update(identity=identity, result_root=Path(receipt["result_dir"]))
        output = self.base / "job-output"
        (output / "checkpoint").mkdir(parents=True)
        (output / "checkpoint/model.pt").write_bytes(b"fixture full checkpoint")
        report = dict(
            identity, passed=True, exit_code=0, world_size=2, g0_passed=False, execution_class_certified=False
        )
        helpers["publish"](output, None, report, True)
        self.assertTrue((Path(receipt["result_dir"]) / "artifacts/checkpoint/model.pt").is_file())
        self.assertTrue(remote.verify_result(receipt)["verified"])
        self.assertEqual(
            {entry["path"] for entry in remote.fetch(self.request)["files"]},
            {"preflight.json", "receipt.json"},
        )

    def test_preflight_supervisor_terminates_and_reaps_its_child_group(self):
        helpers = self.preflight_job_functions()
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        helpers["process"] = child
        try:
            helpers["stop_child"]()
            self.assertIsNotNone(child.returncode)
            self.assertNotEqual(child.returncode, 0)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_preflight_publication_failure_preserves_diagnosis_but_has_no_success_receipt(self):
        helpers = self.preflight_job_functions()
        helpers.update(identity={"task": "qwen3-v2-preflight"}, result_root=self.results)
        output = self.base / "output"
        output.mkdir()
        (output / "checkpoint.pt").symlink_to(self.base / "outside-checkpoint")
        report = {"passed": False, "error": "interrupted before completion"}
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            helpers["publish"](output, None, report, False)
        self.assertEqual(json.loads((self.results / "preflight.json").read_text()), report)
        self.assertFalse((self.results / "receipt.json").exists())

    def teacher_fixture(self, calibration=False):
        specification = importlib.util.spec_from_file_location(
            "teacher_remote_git_fixture", Path(__file__).with_name("test_provenance.py")
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        fixture = module.ProvenanceTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        verifier = Path(__file__).resolve().parents[2] / "tools/sdsc_provenance.py"
        fixture.write("tools/sdsc_provenance.py", verifier.read_bytes())
        fixture.write("tools/sdsc_teacher_prepare.py", b"# Never executed: control fixture\n")
        fixture.write("tools/sdsc_teacher_job.sh", b"#!/bin/bash\nexit 2\n")
        if calibration:
            fixture.write("tools/sdsc_g0_calibration.py", b"# Never executed: calibration fixture\n")
            fixture.write("tools/sdsc_calibration_job.sh", b"#!/bin/bash\nexit 2\n")
        fixture.refresh_wrapper()
        fixture.wrapper["project"] = "OPD"
        fixture.wrapper_path.write_bytes(module.provenance.canonical(fixture.wrapper))
        artifact = fixture.prepare()
        self.enable_preflight()
        self.request.update(
            task="qwen3-v2-teacher-prepare",
            run_id=fixture.wrapper["run_id"],
            code_sha256=fixture.wrapper["code_sha256"],
            provenance_manifest_sha256=artifact["manifest_sha256"],
        )
        self.request["resources"].update(gpus=1, time="04:00:00")
        release = self.root / "releases" / self.request["run_id"]
        (release / "source").mkdir(parents=True)
        for record in fixture.wrapper["files"]:
            target = release / "source" / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((fixture.root / record["path"]).read_bytes())
            target.chmod(record["mode"])
        remote.atomic_json(release / "manifest.json", fixture.wrapper)
        destination = self.root / "provenance" / artifact["manifest_sha256"]
        destination.mkdir(parents=True)
        for name in ("manifest.json", "wrapper-manifest.json", "history.bundle"):
            (destination / name).write_bytes((Path(artifact["artifact"]) / name).read_bytes())
        self.request["provenance_dir"] = str(destination)
        return fixture, release, destination

    def submit_teacher(self):
        fixture, release, destination = self.teacher_fixture()
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(
                remote, "run", return_value={"returncode": 0, "stdout": "12345\n", "stderr": ""}
            ) as call,
        ):
            receipt = remote.submit(self.request)
        self.request["job_id"] = receipt["job_id"]
        return receipt, call, fixture, release, destination

    def publish_teacher(self, receipt, **changes):
        directory = Path(receipt["result_dir"])
        identity = {
            key: receipt[key]
            for key in ("run_id", "job_id", "code_sha256", "task", "hf_home", *remote.PROVENANCE_BINDINGS)
        }
        result = dict(
            identity,
            passed=True,
            exit_code=0,
            g0_passed=False,
            execution_class_certified=False,
            partial_attempt_ledger_guaranteed=False,
            resumable=False,
        )
        result.update(changes)
        data = remote.canonical(result)
        (directory / "teacher-prepare.json").write_bytes(data)
        remote.atomic_json(
            directory / "receipt.json",
            dict(
                identity,
                passed=True,
                persisted=True,
                persistent_read_back_verified=True,
                files=[dict(path="teacher-prepare.json", sha256=remote.digest(data), size=len(data))],
            ),
        )

    def test_teacher_submit_verifies_real_git_and_binds_nine_worker_arguments(self):
        receipt, call, fixture, release, _artifact = self.submit_teacher()
        self.assertEqual(receipt["science_git_head"], fixture.head)
        self.assertEqual(receipt["provenance_manifest_sha256"], self.request["provenance_manifest_sha256"])
        argv = call.call_args.args[0]
        position = argv.index(str(release / "source/tools/sdsc_teacher_job.sh"))
        self.assertEqual(len(argv[position + 1 :]), 9)
        self.assertEqual(
            argv[-3:],
            [
                self.request["hf_home"],
                self.request["provenance_dir"],
                self.request["provenance_manifest_sha256"],
            ],
        )
        self.assertIn("--gpus=h100:1", argv)
        self.assertIn("--mem=192G", argv)
        self.assertIn("--time=04:00:00", argv)
        self.assertEqual(call.call_count, 1)

    def test_teacher_bad_provenance_fails_before_claim_or_scheduler(self):
        _fixture, _release, artifact = self.teacher_fixture()
        (artifact / "manifest.json").write_text("{}")
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run") as call,
            patch.object(remote, "claim_run") as claim,
            self.assertRaisesRegex(ValueError, "manifest hash"),
        ):
            remote.submit(self.request)
        claim.assert_not_called()
        call.assert_not_called()

    def test_teacher_provenance_cannot_rebind_identical_code_to_another_run(self):
        _fixture, release, _artifact = self.teacher_fixture()
        manifest = remote.read_json(release / "manifest.json")
        manifest["run_id"] = "another-run"
        new_release = release.with_name("another-run")
        release.rename(new_release)
        remote.atomic_json(new_release / "manifest.json", manifest)
        self.request["run_id"] = "another-run"
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run") as call,
            patch.object(remote, "claim_run") as claim,
            self.assertRaisesRegex(ValueError, "wrapper differs"),
        ):
            remote.submit(self.request)
        claim.assert_not_called()
        call.assert_not_called()

    def test_teacher_forbids_container_changed_resources_and_home_storage_before_provenance(self):
        self.teacher_fixture()
        for changes in (
            dict(container=self.container()),
            dict(authorized=False),
            dict(resources=dict(self.request["resources"], gpus=2)),
            dict(resources=dict(self.request["resources"], time="04:00:01")),
            dict(result_root=str(self.root)),
        ):
            with (
                self.subTest(changes=changes),
                patch.object(remote, "TEMPORARY_ROOTS", ()),
                patch.object(remote, "verify_provenance") as verify,
                patch.object(remote, "run") as call,
                self.assertRaises(ValueError),
            ):
                remote.submit(dict(self.request, **changes))
            verify.assert_not_called()
            call.assert_not_called()

    def test_teacher_status_requires_bound_provenance_publication_and_non_g0_result(self):
        receipt, *_ = self.submit_teacher()
        self.publish_teacher(receipt)
        with patch.object(
            remote,
            "run",
            side_effect=[
                {"returncode": 0, "stdout": "", "stderr": ""},
                {"returncode": 0, "stdout": "12345|COMPLETED|0:0||\n", "stderr": ""},
            ],
        ):
            status = remote.status(self.request)
        self.assertTrue(status["success"])
        self.assertEqual(status["provenance_manifest_sha256"], receipt["provenance_manifest_sha256"])
        for changed in (
            dict(provenance_manifest_sha256="e" * 64),
            dict(bundle_sha256="e" * 64),
            dict(science_git_head="e" * 40),
            dict(task="qwen3-v2-preflight"),
            dict(g0_passed=True),
            dict(execution_class_certified=True),
            dict(partial_attempt_ledger_guaranteed=True),
            dict(resumable=True),
        ):
            self.publish_teacher(receipt, **changed)
            self.assertFalse(remote.verify_result(receipt)["verified"], changed)

    def test_teacher_fetch_excludes_dataset_and_ledger_and_receipt_tampering_is_rejected(self):
        receipt, *_ = self.submit_teacher()
        self.publish_teacher(receipt)
        directory = Path(receipt["result_dir"])
        (directory / "dataset.jsonl").write_bytes(b"never fetch dataset")
        (directory / "ledger.json").write_bytes(b"never fetch ledger")
        (directory / "preflight.json").write_text("{}")
        fetched = remote.fetch(self.request)
        self.assertEqual(
            {item["path"] for item in fetched["files"]}, {"teacher-prepare.json", "receipt.json"}
        )
        self.assertTrue(fetched["result"]["verified"])
        altered = dict(receipt, provenance_dir="/changed")
        remote.atomic_json(Path(receipt["submission_dir"]) / "receipt.json", altered)
        with patch.object(remote, "run") as call, self.assertRaisesRegex(ValueError, "binding mismatch"):
            remote.status(self.request)
        call.assert_not_called()

    def calibration_fixture(self):
        fixture, release, artifact = self.teacher_fixture(calibration=True)
        provenance = remote.verify_provenance(
            self.root, release, remote.read_json(release / "manifest.json"), self.request
        )
        upstream = {}
        for role, job, token, task in (
            ("teacher", "101", "b", "qwen3-v2-teacher-prepare"),
            ("preflight", "102", "c", "qwen3-v2-preflight"),
        ):
            directory = self.root / "submissions" / (token * 32)
            directory.mkdir(parents=True)
            result = self.results / (role + "-run") / directory.name
            result.mkdir(parents=True)
            manifest = dict(remote.read_json(release / "manifest.json"), run_id=role + "-run")
            upstream_release = release.with_name(role + "-run")
            shutil.copytree(release / "source", upstream_release / "source")
            remote.atomic_json(upstream_release / "manifest.json", manifest)
            intent = dict(
                self.request,
                resources=dict(self.request["resources"]),
                task=task,
                intent_id=directory.name,
                run_id=role + "-run",
                submission_dir=str(directory),
                result_dir=str(result),
                created_at=remote.now(),
            )
            for key in remote.PROVENANCE_BINDINGS:
                intent.pop(key, None)
            if role == "teacher":
                intent.update(provenance)
            else:
                intent["resources"] = dict(intent["resources"], gpus=2, time="02:00:00")
            remote.atomic_json(directory / "intent.json", intent)
            receipt = remote.receipt_from(intent, job)
            remote.atomic_json(directory / "receipt.json", receipt)
            (self.publish_teacher if role == "teacher" else self.publish_preflight)(receipt)
            published = remote.read_json(result / "receipt.json")
            if role == "teacher":
                for name, data in (
                    ("dataset/manifest.json", b"{}\n"),
                    ("dataset/train.jsonl", b'{"id":1}\n'),
                    ("teacher_demos/manifest.json", b"{}\n"),
                    ("teacher_demos/demos.jsonl", b'{"id":2}\n'),
                ):
                    path = result / "artifacts" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    published["files"].append(
                        dict(path="artifacts/" + name, size=len(data), sha256=remote.digest(data))
                    )
            else:
                # Unrelated checkpoint bytes are deliberately absent: proof must not reread them.
                published["files"].append(
                    dict(path="artifacts/checkpoint/model.pt", size=11 * 1024**3, sha256="f" * 64)
                )
            remote.atomic_json(result / "receipt.json", published)
            upstream[role] = receipt
        self.request.update(task="qwen3-v2-g0-calibration", teacher_job_id="101", preflight_job_id="102")
        self.request["resources"].update(gpus=2, time="02:00:00")
        return fixture, release, artifact, upstream

    def calibration_scheduler(self, argv, timeout=30):
        if argv[0] == "squeue":
            return dict(returncode=0, stdout="", stderr="")
        if argv[0] == "sacct":
            job = next(value.split("=", 1)[1] for value in argv if value.startswith("--jobs="))
            return dict(
                returncode=0, stdout=f"{job}|COMPLETED|0:0||\n{job}.batch|COMPLETED|0:0||\n", stderr=""
            )
        self.assertEqual(argv[0], "sbatch")
        self.assertEqual(timeout, 45)
        return dict(returncode=0, stdout="103\n", stderr="")

    def test_calibration_actual_proof_and_eleven_arguments_bind_all_upstreams(self):
        _, release, _, upstream = self.calibration_fixture()
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", side_effect=self.calibration_scheduler) as call,
        ):
            receipt = remote.submit(self.request)
        proof_path = Path(receipt["prerequisites_path"])
        proof = remote.read_json(proof_path)
        self.assertEqual(receipt["prerequisites_sha256"], remote.digest(proof_path.read_bytes()))
        self.assertEqual(proof["target"]["run_id"], receipt["run_id"])
        self.assertEqual(proof["upstream"]["teacher"]["receipt"], upstream["teacher"])
        self.assertTrue(proof["upstream"]["preflight"]["status"]["success"])
        self.assertEqual(len(proof["upstream"]["teacher"]["file_verification"]["verified_files"]), 4)
        self.assertFalse(
            proof["upstream"]["preflight"]["file_verification"]["preflight_checkpoint_content_rehashed"]
        )
        for role in ("teacher", "preflight"):
            for prefix in ("report", "publication_receipt", "release_manifest"):
                value = proof["upstream"][role]
                raw = Path(value[prefix + "_path"]).read_bytes()
                self.assertEqual(remote.digest(raw), value[prefix + "_sha256"])
                self.assertEqual(json.loads(raw), value[prefix])
        calls = [item for item in call.call_args_list if item.args[0][0] == "sbatch"]
        self.assertEqual(len(calls), 1)
        argv = calls[0].args[0]
        position = argv.index(str(release / "source/tools/sdsc_calibration_job.sh"))
        self.assertEqual(len(argv[position + 1 :]), 11)
        self.assertEqual(argv[-2:], [str(proof_path), receipt["prerequisites_sha256"]])
        self.assertIn("--no-requeue", argv)
        self.assertIn("--signal=B:TERM@300", argv)
        # Consume the actual control-plane bytes with the production node wrapper.
        script = (Path(__file__).resolve().parents[2] / "tools/sdsc_calibration_job.sh").read_text()
        embedded = ast.parse(script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0])
        definitions = ast.Module(
            body=[
                node
                for node in embedded.body
                if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)
            ],
            type_ignores=[],
        )
        helpers = {}
        exec(compile(definitions, "sdsc_calibration_job.sh", "exec"), helpers)
        helpers.update(
            project=self.results,
            control_releases=self.root / "releases",
            submission=receipt["submission_dir"],
            run_id=receipt["run_id"],
            code_hash=receipt["code_sha256"],
            identity=dict(receipt),
            prerequisites_path=str(proof_path),
            prerequisites_hash=receipt["prerequisites_sha256"],
            mount=lambda path: {"fstype": "lustre"},
        )
        proofs = self.base / "wrapper-work/proofs"
        proofs.mkdir(parents=True)
        with patch.object(helpers["shutil"], "disk_usage", return_value=SimpleNamespace(free=2**40)):
            staged_proof, copied = helpers["stage_prerequisites"](
                proofs, self.base / "wrapper-work/inputs/qwen3-v2"
            )
        self.assertEqual(staged_proof, proof)
        self.assertEqual(copied["files"], 4)
        self.assertTrue(copied["read_back_verified"])

    def test_calibration_lost_receipt_reconciles_same_proof_without_another_sbatch(self):
        self.calibration_fixture()

        def uncertain_scheduler(argv, timeout=30):
            if argv[0] == "sbatch":
                raise subprocess.TimeoutExpired(argv, timeout)
            return self.calibration_scheduler(argv, timeout)

        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", side_effect=uncertain_scheduler) as call,
        ):
            result = remote.submit(self.request)
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(sum(item.args[0][0] == "sbatch" for item in call.call_args_list), 1)
        directory = self.root / "submissions" / self.request["intent_id"]
        original = remote.read_json(directory / "intent.json")
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", side_effect=self.calibration_scheduler) as call,
            self.assertRaisesRegex(ValueError, "Intent already exists"),
        ):
            remote.submit(self.request)
        self.assertFalse(any(item.args[0][0] == "sbatch" for item in call.call_args_list))
        name = "opd-" + self.request["intent_id"]
        user = remote.pwd.getpwuid(remote.os.getuid()).pw_name
        with patch.object(
            remote,
            "run",
            side_effect=[
                dict(returncode=0, stdout=f"103|{name}\n", stderr=""),
                dict(returncode=0, stdout=f"103|{name}|{user}\n", stderr=""),
            ],
        ) as call:
            recovered = remote.reconcile(self.request)
        self.assertEqual(recovered["job_id"], "103")
        for key in remote.PREREQUISITE_BINDINGS:
            self.assertEqual(recovered[key], original[key])
        self.assertFalse(any(item.args[0][0] == "sbatch" for item in call.call_args_list))

    def test_calibration_refuses_pending_or_failed_upstream_before_claim(self):
        self.calibration_fixture()
        for state, exit_code in (("RUNNING", "0:0"), ("FAILED", "1:0"), ("COMPLETED", "0:9")):

            def scheduler(argv, timeout=30, state=state, exit_code=exit_code):
                if argv[0] == "sacct" and "--jobs=101" in argv:
                    return dict(returncode=0, stdout=f"101|{state}|{exit_code}||\n", stderr="")
                return self.calibration_scheduler(argv, timeout)

            with (
                self.subTest(state=state),
                patch.object(remote, "TEMPORARY_ROOTS", ()),
                patch.object(remote, "run", side_effect=scheduler) as call,
                patch.object(remote, "claim_run") as claim,
                self.assertRaisesRegex(ValueError, "successful Slurm"),
            ):
                remote.submit(self.request)
            claim.assert_not_called()
            self.assertFalse(any(item.args[0][0] == "sbatch" for item in call.call_args_list))

    def test_calibration_teacher_data_hash_and_extra_file_block_before_claim(self):
        *_, upstream = self.calibration_fixture()
        data = Path(upstream["teacher"]["result_dir"]) / "artifacts/dataset/train.jsonl"
        original = data.read_bytes()
        for mutation in ("changed", "extra", "symlink"):
            extra = data.with_name("unpublished.jsonl")
            if mutation == "changed":
                data.write_bytes(b"X" * len(original))
            elif mutation == "extra":
                extra.write_text("unpublished")
            else:
                data.unlink()
                data.symlink_to(extra)
            with (
                self.subTest(mutation=mutation),
                patch.object(remote, "TEMPORARY_ROOTS", ()),
                patch.object(remote, "run", side_effect=self.calibration_scheduler) as call,
                patch.object(remote, "claim_run") as claim,
                self.assertRaises(ValueError),
            ):
                remote.submit(self.request)
            claim.assert_not_called()
            self.assertFalse(any(item.args[0][0] == "sbatch" for item in call.call_args_list))
            if data.is_symlink():
                data.unlink()
            data.write_bytes(original)
            extra.unlink(missing_ok=True)

    def test_calibration_other_active_opd_job_blocks_without_claim(self):
        self.calibration_fixture()

        def scheduler(argv, timeout=30):
            if argv[0] == "squeue" and "--format=%i|%T|%j" in argv:
                return dict(returncode=0, stdout="444|PENDING|opd-other\n", stderr="")
            return self.calibration_scheduler(argv, timeout)

        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", side_effect=scheduler) as call,
            patch.object(remote, "claim_run") as claim,
            self.assertRaisesRegex(ValueError, "remain active"),
        ):
            remote.submit(self.request)
        claim.assert_not_called()
        self.assertFalse(any(item.args[0][0] == "sbatch" for item in call.call_args_list))

    def test_calibration_cannot_supply_fabricated_proof_or_wrong_job(self):
        self.calibration_fixture()
        for changes in (
            dict(prerequisites_path="/fake"),
            dict(teacher_job_id="102"),
            dict(preflight_job_id="999"),
            dict(authorized=False),
            dict(resources=dict(self.request["resources"], time="02:00:01")),
        ):
            with (
                self.subTest(changes=changes),
                patch.object(remote, "TEMPORARY_ROOTS", ()),
                patch.object(remote, "run", side_effect=self.calibration_scheduler) as call,
                patch.object(remote, "claim_run") as claim,
                self.assertRaises(ValueError),
            ):
                remote.submit(dict(self.request, **changes))
            claim.assert_not_called()
            self.assertFalse(any(item.args[0][0] == "sbatch" for item in call.call_args_list))

    def test_calibration_bound_publication_status_fetch_and_tamper_rejection(self):
        self.calibration_fixture()
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", side_effect=self.calibration_scheduler),
        ):
            receipt = remote.submit(self.request)
        self.request["job_id"] = receipt["job_id"]
        root = Path(receipt["result_dir"])
        identity = {
            key: receipt[key]
            for key in (
                "task",
                "job_id",
                "run_id",
                "code_sha256",
                "hf_home",
                *remote.PROVENANCE_BINDINGS,
                *remote.PREREQUISITE_BINDINGS,
            )
        }
        report = dict(
            identity,
            passed=True,
            exit_code=0,
            g0_passed=False,
            execution_class_certified=False,
            pilot_passed=False,
            factorial_ready=False,
        )

        def publish():
            raw = remote.canonical(report)
            (root / "g0-calibration.json").write_bytes(raw)
            remote.atomic_json(
                root / "receipt.json",
                dict(
                    identity,
                    passed=True,
                    persisted=True,
                    persistent_read_back_verified=True,
                    files=[dict(path="g0-calibration.json", sha256=remote.digest(raw), size=len(raw))],
                ),
            )

        publish()
        with patch.object(remote, "run", side_effect=self.calibration_scheduler):
            self.assertTrue(remote.status(self.request)["success"])
        (root / "checkpoint.pt").write_bytes(b"never fetch")
        self.assertEqual(
            {item["path"] for item in remote.fetch(self.request)["files"]},
            {"g0-calibration.json", "receipt.json"},
        )
        for key in ("g0_passed", "execution_class_certified", "pilot_passed", "factorial_ready"):
            report[key] = True
            publish()
            self.assertFalse(remote.verify_result(receipt)["verified"])
            report[key] = False
        publish()
        Path(receipt["prerequisites_path"]).write_text("{}")
        self.assertFalse(remote.verify_result(receipt)["verified"])


if __name__ == "__main__":
    unittest.main()
