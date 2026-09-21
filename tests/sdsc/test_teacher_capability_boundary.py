"""Exploratory probe trust boundaries; no network, model, GPU or real Slurm."""

import ast
import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
TASK = "qwen3-v2-teacher-capability-probe"
REPORT = "teacher-capability-probe.json"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cli = module("capability_cli", REPO / "tools/sdsc_cli.py")
remote = module("capability_remote", REPO / "tools/sdsc_remote.py")


class ProbeClientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "OPD"
        self.root.mkdir()
        root_patch = patch.object(cli, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        cli.write_json(
            cli.state_root() / "runs/run-test.json",
            dict(state="deployed", manifest=dict(run_id="run-test", code_sha256="a" * 64)),
        )

    def arguments(self, dry_run=False):
        values = [
            "submit", "run-test", "--task", TASK,
            "--account", "nwu181", "--partition", "nairr-gpu-shared",
            "--qos", "nairr-gpu-shared-normal", "--gpu-type", "h100",
            "--gpus", "1", "--cpus", "24", "--mem-gib", "192", "--time", "00:30:00",
            "--hf-home", str(cli.PREFLIGHT_STORAGE / "cache/huggingface"),
            "--dry-run" if dry_run else "--authorize",
        ]
        if not dry_run:
            values += [
                "--python", "/confirmed/bin/python",
                "--result-root", str(cli.PREFLIGHT_STORAGE / "results"), "--storage-confirmed",
            ]
        return cli.parser().parse_args(values)

    def test_snapshot_only_dry_run_has_seven_arguments_and_no_remote_call(self):
        args = self.arguments(True)
        with patch.object(cli, "remote") as call, redirect_stdout(io.StringIO()):
            cli.submit_command(args)
        call.assert_not_called()
        plan = cli.read_json(cli.state_root() / "submit-plan.json")
        argv = plan["sbatch_argv"]
        script = next(i for i, arg in enumerate(argv) if arg.endswith("/sdsc_teacher_capability_job.sh"))
        self.assertEqual(len(argv[script + 1 :]), 7)
        self.assertEqual(argv[-1], args.hf_home)
        for value in ("--gpus=h100:1", "--cpus-per-task=24", "--mem=192G", "--time=00:30:00"):
            self.assertIn(value, argv)
        self.assertNotIn("provenance_manifest_sha256", plan)
        self.assertFalse(plan["authorized"])

    def test_changed_resource_envelope_is_rejected_before_network(self):
        for key, value in (("gpus", 2), ("cpus", 4), ("mem_gib", 16), ("time", "00:30:01")):
            args = self.arguments()
            setattr(args, key, value)
            with self.subTest(key=key), patch.object(cli, "remote") as call, self.assertRaises(cli.UserError):
                cli.submit_command(args)
            call.assert_not_called()

    def test_probe_cannot_borrow_provenance_or_upstream_authority(self):
        for key, value in (
            ("provenance_manifest_sha256", "a" * 64),
            ("teacher_job_id", "12345"),
            ("preflight_job_id", "12346"),
            ("authorize", False),
            ("result_root", cli.REMOTE_ROOT + "/smoke-results"),
        ):
            args = self.arguments()
            setattr(args, key, value)
            with self.subTest(key=key), patch.object(cli, "remote") as call, self.assertRaises(cli.UserError):
                cli.submit_command(args)
            call.assert_not_called()


class ProbeRemoteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "control"
        results = Path(temporary.name) / "persistent"
        results.mkdir()
        for name, value in (("ROOT", self.root), ("PREFLIGHT_STORAGE", results)):
            patcher = patch.object(remote, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        hf_home = results / "huggingface"
        for model, revision in remote.PREFLIGHT_MODELS:
            snapshot = hf_home / "hub" / model / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
        self.request = dict(
            root=str(self.root),
            run_id="test-run",
            intent_id="a" * 32,
            task=TASK,
            authorized=True,
            storage_confirmed=True,
            python=sys.executable,
            result_root=str(results),
            hf_home=str(hf_home),
            resources=dict(
                account="nwu181", partition="nairr-gpu-shared", qos="nairr-gpu-shared-normal",
                gpu_type="h100", gpus=1, cpus=24, mem_gib=192, time="00:30:00",
            ),
        )
        files = {
            "tools/sdsc_teacher_capability_job.sh": b"#!/bin/bash\nexit 0\n",
            "tools/sdsc_teacher_capability.py": b"# bounded diagnostic fixture\n",
            "tools/sdsc_teacher_prepare.py": b"# bounded runtime guard fixture\n",
            "tools/sdsc_training_preflight.py": b"# bounded allocation guard fixture\n",
            "src/candidate.py": b"UNCOMMITTED_CANDIDATE = True\n",
        }
        remote.upload(self.request, self.archive(files))

    def archive(self, files):
        records = [
            dict(path=name, size=len(data), sha256=remote.digest(data),
                 mode=0o755 if name.endswith(".sh") else 0o644)
            for name, data in sorted(files.items())
        ]
        manifest = dict(
            schema="quest-sdsc-snapshot-v1", project="OPD", run_id=self.request["run_id"],
            files=records, code_sha256=remote.digest(remote.canonical(records)),
            total_bytes=sum(record["size"] for record in records),
        )
        self.request["code_sha256"] = manifest["code_sha256"]
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:") as archive:
            entries = [("manifest.json", remote.canonical(manifest), 0o644)] + [
                ("source/" + record["path"], files[record["path"]], record["mode"])
                for record in records
            ]
            for name, data, mode in entries:
                member = tarfile.TarInfo(name)
                member.mode, member.size = mode, len(data)
                archive.addfile(member, io.BytesIO(data))
        stream.seek(0)
        return stream

    def submit(self, response=None):
        response = response or {"returncode": 0, "stdout": "12345;expanse\n", "stderr": ""}
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run", return_value=response) as run,
        ):
            receipt = remote.submit(self.request)
        return receipt, run

    def publish(self, receipt, **changes):
        identity = {key: receipt[key] for key in ("run_id", "job_id", "code_sha256", "task", "hf_home")}
        result = dict(
            identity,
            passed=True,
            exit_code=0,
            exploratory=True,
            accepted_science=False,
            full_teacher_ready=False,
            g0_passed=False,
            execution_class_certified=False,
            resumable=False,
            artifact_kind="teacher_capability_diagnostic",
            readiness=False,
            training_started=False,
            readiness_artifact_produced=False,
            metrics_passed=False,
        )
        result.update(changes)
        data = remote.canonical(result)
        root = Path(receipt["result_dir"])
        (root / REPORT).write_bytes(data)
        remote.atomic_json(
            root / "receipt.json",
            dict(
                identity,
                passed=True,
                persisted=True,
                persistent_read_back_verified=True,
                files=[dict(path=REPORT, size=len(data), sha256=remote.digest(data))],
            ),
        )

    def test_receipt_binds_snapshot_and_fixed_resources_without_formal_git_claim(self):
        receipt, run = self.submit()
        self.assertEqual(receipt["task"], TASK)
        self.assertNotIn("science_git_head", receipt)
        self.assertNotIn("provenance_manifest_sha256", receipt)
        argv = run.call_args.args[0]
        self.assertEqual(argv[-1], self.request["hf_home"])
        self.assertIn("--no-requeue", argv)
        self.assertIn("--time=00:30:00", argv)
        self.assertEqual(run.call_count, 1)
        with self.assertRaisesRegex(ValueError, "wrong task"):
            remote.registered_job(self.root, receipt["job_id"], "qwen3-v2-teacher-prepare")

    def test_bad_authority_storage_resources_fail_before_claim(self):
        for changes in (
            dict(authorized=False),
            dict(storage_confirmed=False),
            dict(provenance_manifest_sha256="a" * 64),
            dict(teacher_job_id="12345"),
            dict(hf_home="/home/zgao12/cache"),
            dict(container={}),
            dict(resources=dict(self.request["resources"], time="00:30:01")),
        ):
            with (
                self.subTest(changes=changes),
                patch.object(remote, "TEMPORARY_ROOTS", ()),
                patch.object(remote, "run") as run,
                self.assertRaises(ValueError),
            ):
                remote.submit(dict(self.request, **changes))
            run.assert_not_called()
            self.assertFalse((self.root / "run-claims").exists())

    def test_ambiguous_submission_is_not_retried_and_consumes_release(self):
        result, run = self.submit(dict(returncode=1, stdout="", stderr="connection lost"))
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(run.call_count, 1)
        self.request["intent_id"] = "b" * 32
        with (
            patch.object(remote, "TEMPORARY_ROOTS", ()),
            patch.object(remote, "run") as call,
            self.assertRaisesRegex(ValueError, "already claimed"),
        ):
            remote.submit(self.request)
        call.assert_not_called()

    def test_result_requires_explicit_exploratory_scope_and_real_accounting(self):
        receipt, _ = self.submit()
        self.request["job_id"] = receipt["job_id"]
        self.publish(receipt)
        replies = [
            dict(returncode=0, stdout="", stderr=""),
            dict(returncode=0, stdout="12345|COMPLETED|0:0||\n", stderr=""),
        ]
        with (
            patch.object(remote, "run", side_effect=replies),
            patch.object(remote.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="fixture")),
        ):
            self.assertTrue(remote.status(self.request)["success"])
        for key, value in (
            ("exploratory", False),
            ("accepted_science", True),
            ("accepted_science", {}),
            ("full_teacher_ready", True),
            ("resumable", True),
            ("g0_passed", True),
            ("execution_class_certified", True),
            ("passed", False),
            ("exit_code", 1),
            ("task", "qwen3-v2-teacher-prepare"),
            ("artifact_kind", "teacher_readiness"),
            ("readiness", True),
            ("training_started", True),
            ("readiness_artifact_produced", True),
            ("metrics_passed", None),
        ):
            with self.subTest(key=key, value=value):
                self.publish(receipt, **{key: value})
                self.assertFalse(remote.verify_result(receipt)["verified"])

    def test_fetch_excludes_raw_ledger_and_never_writes_scientific_inputs(self):
        receipt, _ = self.submit()
        self.request["job_id"] = receipt["job_id"]
        self.publish(receipt)
        root = Path(receipt["result_dir"])
        (root / "worker.log").write_text("diagnostic line\n" * 20000)
        (root / "attempt-ledger.jsonl").write_text("RAW MODEL OUTPUT\n" * 10000)
        response = remote.fetch(self.request)
        self.assertEqual({item["path"] for item in response["files"]}, {REPORT, "receipt.json", "worker.log"})
        self.assertLess(response["total_bytes"], 65536)


def wrapper_functions():
    script = (REPO / "tools/sdsc_teacher_capability_job.sh").read_text()
    tree = ast.parse(script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0])
    definitions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(definitions, "sdsc_teacher_capability_job.sh", "exec"), namespace)
    return namespace


class ProbeWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.cleanup)
        self.root = Path(self.temporary.name)
        self.helpers = wrapper_functions()
        self.identity = dict(
            task=TASK,
            run_id="probe-fixture",
            job_id="12345",
            code_sha256="a" * 64,
            hf_home="/confirmed/cache",
        )
        self.helpers.update(identity=self.identity, result_root=self.root)

    def cleanup(self):
        for directory, _, files in os.walk(self.temporary.name):
            Path(directory).chmod(0o700)
            for name in files:
                (Path(directory) / name).chmod(0o600)
        self.temporary.cleanup()

    def report(self):
        return dict(
            self.identity,
            passed=True,
            exit_code=0,
            exploratory=True,
            accepted_science=False,
            full_teacher_ready=False,
            g0_passed=False,
            execution_class_certified=False,
            resumable=False,
            artifact_kind="teacher_capability_diagnostic",
            readiness=False,
            training_started=False,
            readiness_artifact_produced=False,
            metrics_passed=False,
        )

    def test_worker_cannot_claim_formal_acceptance_or_borrow_another_identity(self):
        report = self.report()
        self.helpers["validate_report"](report, 0)
        for changes in (
            dict(accepted_science=True),
            dict(full_teacher_ready=True),
            dict(g0_passed=True),
            dict(exploratory=False),
            dict(resumable=True),
            dict(code_sha256="b" * 64),
            dict(artifact_kind="teacher_readiness"),
            dict(readiness=True),
            dict(training_started=True),
            dict(readiness_artifact_produced=True),
            dict(metrics_passed=None),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.helpers["validate_report"](dict(report, **changes), 0)
        with self.assertRaises(ValueError):
            self.helpers["validate_report"](report, 1)

    def test_failure_publication_preserves_raw_response_and_original_report(self):
        output = self.root / "node-output"
        output.mkdir()
        raw = b'{"response_text":"<proof>invalid raw response</proof>","accepted":false}\n'
        (output / "attempt-ledger.jsonl").write_bytes(raw)
        original = dict(self.report(), passed=False, phase="generation", error="fixture failed")
        (output / REPORT).write_text(json.dumps(original))
        report = dict(self.report(), passed=False, exit_code=1)
        with redirect_stdout(io.StringIO()):
            self.helpers["publish"](output, None, report, False)
        self.assertEqual((self.root / "artifacts/attempt-ledger.jsonl").read_bytes(), raw)
        self.assertEqual(json.loads((self.root / "artifacts" / REPORT).read_text()), original)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertFalse(receipt["passed"])
        self.assertTrue(receipt["persistent_read_back_verified"])
        for record in receipt["files"]:
            path = self.root / record["path"]
            self.assertEqual(self.helpers["file_hash"](path), record["sha256"])
            self.assertEqual(path.stat().st_size, record["size"])

    def test_staged_candidate_is_hash_bound_and_readonly_without_git(self):
        release = self.root / "release"
        release.mkdir()
        files = {
            "tools/sdsc_teacher_capability_job.sh": b"#!/bin/bash\n",
            "tools/sdsc_teacher_capability.py": b"# diagnostic\n",
            "tools/sdsc_teacher_prepare.py": b"# runtime guard\n",
            "tools/sdsc_training_preflight.py": b"# allocation guard\n",
            "src/candidate.py": b"UNCOMMITTED = True\n",
        }
        records = []
        for name, data in sorted(files.items()):
            path = release / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o644)
            records.append(
                dict(path=name, size=len(data), mode=0o644, sha256=self.helpers["file_hash"](path))
            )
        digest = self.helpers["hashlib"].sha256(self.helpers["canonical"](records)).hexdigest()
        (release / "manifest.json").write_text(
            json.dumps(dict(run_id="candidate", code_sha256=digest, files=records))
        )
        self.helpers.update(run_id="candidate", code_hash=digest)
        destination = self.root / "staged"
        destination.mkdir()
        self.helpers["stage_source"](release, destination)
        self.assertEqual((destination / "src/candidate.py").read_bytes(), files["src/candidate.py"])
        self.assertFalse((destination / "src/candidate.py").stat().st_mode & 0o222)
        self.assertFalse((destination / ".git").exists())
        (release / "source/src/candidate.py").write_bytes(b"TAMPERED = True\n")
        changed = self.root / "changed"
        changed.mkdir()
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.helpers["stage_source"](release, changed)


if __name__ == "__main__":
    unittest.main()
