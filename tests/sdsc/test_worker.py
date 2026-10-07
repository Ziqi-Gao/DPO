"""CPU-only worker contract checks; no Slurm command or real GPU is invoked."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sdsc_smoke", ROOT / "tools/sdsc_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
REMOTE_SPEC = importlib.util.spec_from_file_location("sdsc_remote", ROOT / "tools/sdsc_remote.py")
remote = importlib.util.module_from_spec(REMOTE_SPEC)
REMOTE_SPEC.loader.exec_module(remote)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opd-worker-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / "results"
        self.results.mkdir()

    def test_atomic_publication_readback_and_no_clobber(self):
        path = self.results / "result.json"
        self.assertEqual(smoke.publish(path, b"first"), hashlib.sha256(b"first").hexdigest())
        with self.assertRaises(FileExistsError):
            smoke.publish(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"first")
        self.assertEqual(list(self.results.iterdir()), [path])

    def test_result_storage_rejects_home_and_temporary(self):
        for home in (self.root, self.root / "unrelated-home"):
            with self.subTest(home=home), self.assertRaises(ValueError):
                smoke.validate_results(self.results, home, self.root / "stage")

    def test_gpu_shape_and_hardware_gates_need_no_real_torch(self):
        for count, name in ((0, "H100"), (2, "H100"), (1, "A100")):
            cuda = types.SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda count=count: count,
                get_device_name=lambda index, name=name: name,
            )
            with (
                self.subTest(count=count, name=name),
                patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}),
                self.assertRaises(RuntimeError),
            ):
                smoke.gpu_smoke()

    def run_worker(self, gpu=None, storage_error=None, publish_error=False, container=None, **resource_env):
        stage = self.root / "stage"
        stage.mkdir()
        (stage / ".sdsc-stage.json").write_text(
            json.dumps({"mount": {"fstype": "xfs"}, "code_sha256": "abc"})
        )
        control = self.root / "control"
        control.mkdir()
        args = ["/release", str(control), str(self.results), sys.executable, "run-1", "abc", str(stage)]
        if container is not None:
            args.append(json.dumps(container))
        original_publish = smoke.publish

        def failing_publish(path, data):
            if path.parent == self.results:
                raise OSError("durable storage unavailable")
            return original_publish(path, data)

        with (
            patch.object(smoke, "__file__", str(stage / "tools/sdsc_smoke.py")),
            patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "1234",
                    "CUDA_VISIBLE_DEVICES": "GPU-example",
                    "SLURM_CPUS_PER_TASK": "4",
                    "SLURM_MEM_PER_NODE": "16384",
                    **resource_env,
                },
            ),
            patch.object(smoke.signal, "signal"),
            patch.object(
                smoke, "validate_results", side_effect=storage_error, return_value={"fstype": "lustre"}
            ),
            patch.object(smoke, "gpu_smoke", side_effect=gpu, return_value={"finite": True}) as probe,
            patch.object(
                smoke, "publish", side_effect=failing_publish if publish_error else original_publish
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = smoke.main(args)
        return code, json.loads((control / "control-result.json").read_text()), probe

    def test_success_requires_bound_persistent_receipt(self):
        code, report, probe = self.run_worker()
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["exit_code"], 0)
        probe.assert_called_once()
        receipt = json.loads((self.results / "receipt.json").read_text())
        self.assertTrue(receipt["persisted"])
        self.assertEqual(receipt["job_id"], "1234")
        self.assertEqual(receipt["code_sha256"], "abc")
        for entry in receipt["files"]:
            data = (self.results / entry["path"]).read_bytes()
            self.assertEqual(entry["size"], len(data))
            self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
        bound_receipt = {
            "run_id": "run-1",
            "job_id": "1234",
            "code_sha256": "abc",
            "result_dir": str(self.results),
        }
        verification = remote.verify_result(bound_receipt)
        self.assertTrue(verification["verified"], verification)
        self.assertEqual(verification["result_sha256"], receipt["result_sha256"])
        self.assertFalse(remote.verify_result({**bound_receipt, "job_id": "5678"})["verified"])

    def test_unavailable_persistent_mount_blocks_gpu_and_success(self):
        code, report, probe = self.run_worker(storage_error=ValueError("mount missing"))
        self.assertNotEqual(code, 0)
        self.assertFalse(report["passed"])
        probe.assert_not_called()
        self.assertFalse((self.results / "receipt.json").exists())

    def test_signal_persists_failure_and_nonzero_exit(self):
        code, report, _ = self.run_worker(gpu=smoke.JobSignal(15))
        self.assertEqual(code, 143)
        self.assertFalse(report["passed"])
        self.assertFalse(json.loads((self.results / "receipt.json").read_text())["passed"])
        verification = remote.verify_result(
            {"run_id": "run-1", "job_id": "1234", "code_sha256": "abc", "result_dir": str(self.results)}
        )
        self.assertFalse(verification["verified"])
        self.assertIn("did not pass", verification["reason"])

    def test_failed_publication_is_never_success(self):
        code, report, _ = self.run_worker(publish_error=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertIn("publication_error", report)
        self.assertFalse((self.results / "receipt.json").exists())

    def test_wrong_cpu_contract_blocks_gpu(self):
        code, report, probe = self.run_worker(SLURM_CPUS_PER_TASK="8")
        self.assertNotEqual(code, 0)
        self.assertFalse(report["passed"])
        probe.assert_not_called()

    def test_wrong_reported_memory_contract_blocks_gpu(self):
        code, report, probe = self.run_worker(SLURM_MEM_PER_NODE="8192")
        self.assertNotEqual(code, 0)
        self.assertFalse(report["passed"])
        probe.assert_not_called()

    def test_container_report_keeps_host_publication_and_remote_binding(self):
        image = self.root / "torch.sif"
        image.write_bytes(b"image fixture")
        profile = {
            "runtime": sys.executable,
            "image": str(image),
            "python": "/usr/bin/python3",
            "size": image.stat().st_size,
            "mtime_ns": image.stat().st_mtime_ns,
        }
        with patch.object(smoke, "container_gpu_smoke", return_value=({"finite": True}, "diagnostic")):
            code, report, direct_probe = self.run_worker(container=profile)
        self.assertEqual(code, 0)
        direct_probe.assert_not_called()
        self.assertEqual(report["container"], profile)
        self.assertEqual(report["runtime_profile"]["kind"], "singularity")
        self.assertIn("diagnostic", (self.results / "worker.log").read_text())
        self.assertTrue(
            remote.verify_result(
                {
                    "run_id": "run-1",
                    "job_id": "1234",
                    "code_sha256": "abc",
                    "result_dir": str(self.results),
                    "container": profile,
                }
            )["verified"]
        )


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opd-container-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "torch.sif"
        self.image.write_bytes(b"mock container image")
        self.profile = {
            "runtime": sys.executable,
            "image": str(self.image),
            "python": "/usr/bin/python3",
            "size": self.image.stat().st_size,
            "mtime_ns": self.image.stat().st_mtime_ns,
        }
        self.stage = self.root / "stage"
        self.stage.mkdir()
        self.evidence = {
            "finite": True,
            "cpu_reference_matched": True,
            "visible_count": 1,
            "name": "NVIDIA H100",
            "cuda_visible_devices": "GPU-original",
        }

    def call_container(self, stdout=None, stderr="", returncode=0, stage=None):
        child = subprocess.CompletedProcess(
            [], returncode, json.dumps(self.evidence) if stdout is None else stdout, stderr
        )
        with (
            patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "GPU-original", "SINGULARITYENV_CUDA_VISIBLE_DEVICES": "injected"},
            ),
            patch.object(smoke.subprocess, "run", return_value=child) as execute,
        ):
            result = smoke.container_gpu_smoke(self.profile, stage or self.stage)
        return result, execute

    def test_container_command_is_explicit_and_preserves_slurm_visibility(self):
        (result, stderr), execute = self.call_container(stderr="warning")
        self.assertEqual(result, self.evidence)
        self.assertEqual(stderr, "warning")
        argv = execute.call_args.args[0]
        self.assertEqual(
            argv,
            [
                sys.executable,
                "exec",
                "--nv",
                "--cleanenv",
                "--no-home",
                "--bind",
                f"{self.stage}:{self.stage}:ro",
                "--pwd",
                str(self.stage),
                "--env",
                "CUDA_VISIBLE_DEVICES=GPU-original",
                "--env",
                "SLURM_CPUS_PER_TASK=4",
                str(self.image),
                "/usr/bin/python3",
                "-I",
                "-B",
                str(self.stage / "tools/sdsc_smoke.py"),
                "--gpu-only",
            ],
        )
        self.assertEqual(execute.call_args.kwargs["timeout"], 180)
        self.assertNotIn("shell", execute.call_args.kwargs)
        child_env = execute.call_args.kwargs["env"]
        self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "GPU-original")
        self.assertNotIn("SINGULARITYENV_CUDA_VISIBLE_DEVICES", child_env)

    def test_changed_image_fails_before_any_subprocess(self):
        self.image.write_bytes(b"changed")
        with patch.object(smoke.subprocess, "run") as execute, self.assertRaisesRegex(ValueError, "changed"):
            smoke.container_gpu_smoke(self.profile, self.stage)
        execute.assert_not_called()

    def test_symlink_image_is_rejected(self):
        alias = self.root / "alias.sif"
        alias.symlink_to(self.image)
        self.profile["image"] = str(alias)
        with patch.object(smoke.subprocess, "run") as execute, self.assertRaises(ValueError):
            smoke.container_gpu_smoke(self.profile, self.stage)
        execute.assert_not_called()

    def test_invalid_json_and_wrong_gpu_evidence_are_rejected(self):
        for output in (
            "not-json",
            "[]",
            json.dumps({**self.evidence, "finite": False}),
            json.dumps({**self.evidence, "cuda_visible_devices": "changed"}),
            json.dumps({**self.evidence, "visible_count": 2}),
            json.dumps({**self.evidence, "name": "A100"}),
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.call_container(stdout=output)

    def test_stderr_and_json_size_are_bounded(self):
        (_, stderr), _ = self.call_container(stderr="x" * 10000)
        self.assertEqual(len(stderr), 8192)
        with self.assertRaises(ValueError):
            self.call_container(stdout="x" * 65537)
        with self.assertRaisesRegex(RuntimeError, "exited 2"):
            self.call_container(returncode=2)

    def test_unsafe_bind_path_is_rejected(self):
        for component in ("bad:path", "bad,path", "bad\npath"):
            with self.subTest(component=component), self.assertRaises(ValueError):
                self.call_container(stage=self.root / component)

    def test_missing_cuda_visibility_rejected_before_execution(self):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
            patch.object(smoke.subprocess, "run") as execute,
            self.assertRaises(ValueError),
        ):
            smoke.container_gpu_smoke(self.profile, self.stage)
        execute.assert_not_called()

    def test_profile_requires_exact_keys_and_positive_integer_metadata(self):
        for profile in (
            {**self.profile, "extra": True},
            {**self.profile, "size": True},
            {**self.profile, "python": "python3"},
        ):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                smoke.validate_container(profile)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".sdsc-storage-test-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.results = self.home / "quest-runs/OPD/smoke-results/run-1/intent-1"
        self.results.mkdir(parents=True)
        self.stage = self.root / "node-scratch"

    def test_home_allows_only_small_smoke_run_intent_metadata(self):
        with patch.object(smoke, "mount_info", return_value={"fstype": "nfs4"}):
            result = smoke.validate_results(self.results, self.home, self.stage, "run-1")
        self.assertTrue(result["small_home_smoke_metadata_only"])
        self.assertTrue(result["write_read_probe_passed"])
        self.assertEqual(list(self.results.iterdir()), [])

    def test_home_rejects_wrong_identity_other_paths_and_deeper_paths(self):
        for path, run_id in (
            (self.results, "another-run"),
            (self.home, "run-1"),
            (self.results.parent, "run-1"),
            (self.results / "more", "run-1"),
        ):
            path.mkdir(exist_ok=True)
            with self.subTest(path=path, run_id=run_id), self.assertRaises(ValueError):
                smoke.validate_results(path, self.home, self.stage, run_id)

    def test_home_still_requires_real_network_mount(self):
        with patch.object(smoke, "mount_info", return_value={"fstype": "xfs"}), self.assertRaises(ValueError):
            smoke.validate_results(self.results, self.home, self.stage, "run-1")

    def test_symlink_ancestor_is_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.home, target_is_directory=True)
        path = alias / self.results.relative_to(self.home)
        with self.assertRaisesRegex(ValueError, "symlink"):
            smoke.validate_results(path, self.home, self.stage, "run-1")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opd-bootstrap-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.release = self.root / "release"
        self.source = self.release / "source"
        self.control = self.root / "control"
        self.scratch = self.root / "scratch"
        self.binary = self.root / "bin"
        for path in (self.source / "tools", self.control, self.scratch, self.binary):
            path.mkdir(parents=True)
        self.runner = self.source / "tools/sdsc_smoke.py"
        self.runner.write_text(
            "import os\nprint('STAGED', os.getcwd(), os.environ.get('CUDA_VISIBLE_DEVICES'))\n"
        )
        self.findmnt = self.binary / "findmnt"
        self.findmnt.write_text(
            '#!/bin/sh\nprintf \'%s\\n\' \'{"filesystems":[{"target":"/scratch",'
            '"source":"/dev/test","fstype":"xfs","options":"rw"}]}\'\n'
        )
        self.findmnt.chmod(0o755)
        data = self.runner.read_bytes()
        self.files = [
            {
                "path": "tools/sdsc_smoke.py",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "mode": 420,
            }
        ]
        self.write_manifest()

    def write_manifest(self):
        self.digest = hashlib.sha256(
            json.dumps(self.files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        (self.release / "manifest.json").write_text(
            json.dumps({"run_id": "test-run", "code_sha256": self.digest, "files": self.files})
        )

    def run_bootstrap(self, container_json=None, **extra_env):
        env = {
            **os.environ,
            "PATH": str(self.binary) + os.pathsep + os.environ["PATH"],
            "TMPDIR": str(self.scratch),
            "SLURM_JOB_ID": "987654321",
            "CUDA_VISIBLE_DEVICES": "GPU-preserved",
            **extra_env,
        }
        argv = [
            "bash",
            str(ROOT / "tools/sdsc_job.sh"),
            str(self.release),
            str(self.control),
            str(self.root / "results"),
            sys.executable,
            "test-run",
            self.digest,
        ]
        if container_json is not None:
            argv.append(container_json)
        return subprocess.run(
            argv,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )

    def test_verified_snapshot_executes_only_from_local_stage(self):
        process = self.run_bootstrap()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("STAGED " + str(self.scratch / "opd-987654321-"), process.stdout)
        self.assertIn("GPU-preserved", process.stdout)

    def test_container_argument_is_forwarded_after_stage_without_evaluation(self):
        self.runner.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        data = self.runner.read_bytes()
        self.files[0].update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.write_manifest()
        profile_json = '{"runtime":"/some/path with spaces", "literal":"$(never-executed)"}'
        process = self.run_bootstrap(container_json=profile_json)
        self.assertEqual(process.returncode, 0, process.stderr)
        args = json.loads(process.stdout)
        self.assertEqual(len(args), 8)
        self.assertTrue(args[6].startswith(str(self.scratch / "opd-987654321-")))
        self.assertEqual(args[7], profile_json)

    def test_modified_release_bytes_fail_before_execution(self):
        self.runner.write_text("raise RuntimeError('must not execute')\n")
        process = self.run_bootstrap()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("source file differs", process.stderr)
        self.assertFalse(json.loads((self.control / "control-result.json").read_text())["passed"])

    def test_external_symlink_is_never_followed(self):
        external = self.root / "external.py"
        external.write_bytes(self.runner.read_bytes())
        self.runner.unlink()
        self.runner.symlink_to(external)
        process = self.run_bootstrap()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("symlink", process.stderr)

    def test_remote_filesystem_cannot_be_node_local_scratch(self):
        self.findmnt.write_text(self.findmnt.read_text().replace('"xfs"', '"nfs4"'))
        process = self.run_bootstrap()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("not verified node-local", process.stderr)

    def test_missing_job_allocation_fails_before_any_compute(self):
        process = self.run_bootstrap(SLURM_JOB_ID="")
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("real Slurm", process.stderr)

    def test_unsafe_manifest_path_is_rejected(self):
        self.files[0]["path"] = "../outside.py"
        self.write_manifest()
        process = self.run_bootstrap()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("unsafe", process.stderr)


if __name__ == "__main__":
    unittest.main()
