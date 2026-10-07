"""CPU boundary fixtures for the foreground Slurm wrapper; no scheduler calls."""

import importlib.util
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pipeline_job_fixture", ROOT / "tools/sdsc_pipeline_job.py")
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


class PipelineJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "durable"
        self.project.mkdir()
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.work = self.root / "staged"
        self.work.mkdir()
        self.namespace = Path("/opd-pipeline/fixture")
        self.g0root = Path("/scratch/zgao12/job_100/outputs/qwen3-v2")
        self.request = {
            "flow_id": "fixture",
            "stage": "prepare",
            "run_id": "immutable-run",
            "code_sha256": "a" * 64,
            "base_inventory": {},
            "g0_root": str(self.g0root),
        }
        self.environment = {
            "SLURM_JOB_ID": "200",
            "SLURM_CPUS_PER_TASK": "24",
            "SLURM_MEM_PER_NODE": "196608",
            "CUDA_VISIBLE_DEVICES": "GPU-a",
            "TMPDIR": str(self.scratch),
        }
        job.CHILD = None
        self.addCleanup(setattr, job, "CHILD", None)

    def run_main(self, worker):
        request_path = self.root / "request.json"
        request_sha = job.s.atomic(request_path, self.request)
        with (
            patch.dict(os.environ, self.environment, clear=True),
            patch.object(job.s, "PROJECT", self.project),
            patch.object(job.s, "mount", return_value={"fstype": "xfs"}),
            patch.object(job.s, "persistent", return_value={"fstype": "lustre"}),
            patch.object(job.signal, "signal"),
            patch.object(job, "stop_child") as stop,
            patch.object(job, "worker", side_effect=worker),
        ):
            result = job.main(["--request", str(request_path), "--request-sha256", request_sha])
        return result, stop

    def inner_report(self, request, work, namespace, _job):
        directory = work / "g0" if request["stage"] == "g0" else work / "qwen3-v2/pilot"
        directory.mkdir(exist_ok=True, parents=True)
        filename = "g0.json" if request["stage"] == "g0" else "sdsc-stage-prepare-single.json"
        result = directory / filename
        job.s.atomic(result, {"passed": True, "job_id": "200", "run_id": request["run_id"]})
        return result

    def test_pilot_publication_uses_stable_absolute_scientific_path_and_real_byte_hash(self):
        result, stop = self.run_main(self.inner_report)
        self.assertEqual(result, 0)
        stop.assert_called_once()
        destination = self.project / "pipeline-results/fixture/prepare/single"
        receipt = job.s.document(destination / "receipt.json")
        outer = job.s.document(destination / "result.json", receipt["result_sha256"])
        inner = outer["scientific_report"]
        expected = self.namespace / "qwen3-v2/pilot/sdsc-stage-prepare-single.json"
        self.assertEqual(inner["path"], str(expected))
        self.assertEqual(inner["work_relative_path"], "qwen3-v2/pilot/sdsc-stage-prepare-single.json")
        logical = "pilot/" + Path(inner["path"]).relative_to(self.namespace / "qwen3-v2/pilot").as_posix()
        self.assertEqual(receipt["delta"][logical]["sha256"], inner["sha256"])
        self.assertEqual(
            job.s.document(receipt["delta"][logical]["storage"], inner["sha256"]), inner["value"]
        )
        self.assertTrue(receipt["passed"] and receipt["persistent_read_back_verified"])

    def test_g0_publication_retains_original_calibration_namespace(self):
        self.request["stage"] = "g0"
        self.environment["CUDA_VISIBLE_DEVICES"] = "GPU-a,GPU-b"
        result, _ = self.run_main(self.inner_report)
        self.assertEqual(result, 0)
        destination = self.project / "pipeline-results/fixture/g0/single"
        outer = job.s.document(destination / "result.json")
        receipt = job.s.document(destination / "receipt.json")
        self.assertEqual(outer["scientific_report"]["path"], str(self.g0root / "g0.json"))
        self.assertEqual(receipt["delta"]["g0/g0.json"]["sha256"], outer["scientific_report"]["sha256"])

    def test_failed_worker_is_published_as_failure_after_child_cleanup(self):
        def failure(_request, work, _namespace, _job):
            (work / "worker.log").write_text("CPU fixture: TERM interrupted worker")
            raise InterruptedError("CPU fixture signal")

        result, stop = self.run_main(failure)
        self.assertEqual(result, 1)
        stop.assert_called_once()
        destination = self.project / "pipeline-results/fixture/prepare/single"
        outer = job.s.document(destination / "result.json")
        receipt = job.s.document(destination / "receipt.json")
        self.assertFalse(outer["passed"])
        self.assertFalse(receipt["passed"])
        self.assertIn("InterruptedError", outer["error"])
        self.assertTrue((destination / "worker.log").exists())

    def test_copy_failure_never_has_a_success_receipt_or_zero_exit(self):
        with patch.object(job.s, "copy", side_effect=OSError("fixture persistent storage full")):
            result, _ = self.run_main(self.inner_report)
        self.assertEqual(result, 1)
        destination = self.project / "pipeline-results/fixture/prepare/single"
        self.assertFalse((destination / "receipt.json").exists())
        failure = job.s.document(destination / "publication-failure.json")
        self.assertFalse(failure["passed"])
        self.assertEqual(failure["exit_code"], 1)
        self.assertIn("storage full", failure["publication_error"])

    def test_array_wrapper_preserves_actual_allocation_and_logical_task_ids(self):
        self.request["stage"] = "train-cell"
        self.environment.update(
            CUDA_VISIBLE_DEVICES="GPU-a,GPU-b,GPU-c,GPU-d",
            SLURM_ARRAY_JOB_ID="190",
            SLURM_ARRAY_TASK_ID="3",
            SLURM_ARRAY_TASK_COUNT="8",
        )
        result, _ = self.run_main(self.inner_report)
        self.assertEqual(result, 0)
        outer = job.s.document(self.project / "pipeline-results/fixture/train-cell/3/result.json")
        self.assertEqual(outer["job_id"], "190_3")
        self.assertEqual(outer["allocation_job_id"], "200")

    def container_fixture(self):
        runtime = self.root / "singularity"
        runtime.write_bytes(b"fixture only")
        image = self.root / "image.sif"
        image.write_bytes(b"fixture image")
        python = self.root / "conda/bin/python3.12"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"fixture Python")
        request = {
            **self.request,
            "python": str(python),
            "environment": {
                "runtime": {"base_prefix": str(python.parent.parent)},
                "container": {
                    "runtime": str(runtime),
                    "image": str(image),
                    "size": image.stat().st_size,
                    "mtime_ns": image.stat().st_mtime_ns,
                },
            },
        }
        return request, image

    def test_container_bindings_keep_g0_inputs_readonly_and_gpu_visibility_exact(self):
        request, _ = self.container_fixture()
        request["stage"] = "g0"
        with patch.dict(
            os.environ,
            {
                **self.environment,
                "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
                "PYTHONPATH": "/malicious",
                "SINGULARITY_BIND": "/bad:/etc",
            },
            clear=True,
        ):
            argv, environment = job.container_command(
                request, self.work, self.namespace, self.g0root, [request["python"], "fixed.py"]
            )
        self.assertIn(f"{self.work / 'g0/canonical_sft'}:{self.g0root / 'canonical_sft'}:ro", argv)
        self.assertIn(
            f"{self.work / 'g0/initial_checkpoint.pt'}:{self.g0root / 'initial_checkpoint.pt'}:ro", argv
        )
        self.assertIn(f"{self.work / 'science'}:{self.namespace / 'science'}:ro", argv)
        self.assertEqual(environment["SINGULARITYENV_CUDA_VISIBLE_DEVICES"], "GPU-a,GPU-b")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("SINGULARITY_BIND", environment)
        self.assertIn("--cleanenv", argv)
        self.assertEqual(argv[-2:], [request["python"], "fixed.py"])

    def test_pilot_overlay_also_binds_its_verified_base_readonly(self):
        request, _ = self.container_fixture()
        base = self.root / "verified-base"
        base.mkdir()
        request["environment"]["runtime"]["base_prefix"] = str(base)
        argv, _ = job.container_command(request, self.work, self.namespace, self.g0root, ["fixture"])
        self.assertIn(f"{base}:{base}:ro", argv)
        self.assertIn(f"{self.root / 'conda'}:{self.root / 'conda'}:ro", argv)

    def test_numeric_import_threads_follow_allocated_cpus_per_rank_before_container_start(self):
        request, _ = self.container_fixture()
        names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        ambient = {name: "64" for name in names}
        ambient.update({"SINGULARITYENV_" + name: "64" for name in names})
        with patch.dict(os.environ, ambient, clear=True):
            for stage, threads in (("g0", "12"), ("preflight4", "6"), ("prepare", "24")):
                request["stage"] = stage
                with self.subTest(stage=stage):
                    _, environment = job.container_command(
                        request, self.work, self.namespace, self.g0root, ["fixture"]
                    )
                    for name in names:
                        self.assertEqual(environment["SINGULARITYENV_" + name], threads)
                        self.assertNotIn(name, environment)

    def test_later_stages_bind_all_g0_readonly_and_reject_changed_image(self):
        request, image = self.container_fixture()
        argv, _ = job.container_command(request, self.work, self.namespace, self.g0root, ["fixture"])
        self.assertIn(f"{self.work / 'g0'}:{self.g0root}:ro", argv)
        image.write_bytes(b"different image bytes")
        with self.assertRaisesRegex(ValueError, "image changed"):
            job.container_command(request, self.work, self.namespace, self.g0root, ["fixture"])

    def test_stop_waits_for_group_termination_and_kills_a_timed_out_group(self):
        process = Mock(pid=4321)
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 90), 0]
        job.CHILD = process
        with patch.object(job.os, "killpg") as kill:
            job.stop_child()
        self.assertEqual(kill.call_args_list[0].args, (4321, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (4321, signal.SIGKILL))
        self.assertEqual(process.wait.call_args_list[0].kwargs, {"timeout": 90})
        self.assertEqual(process.wait.call_args_list[1].kwargs, {"timeout": 15})

    def test_execute_creates_one_process_group_and_rejects_child_failure(self):
        process = Mock(pid=4321)
        process.wait.return_value = 7
        with (
            patch.object(job.subprocess, "Popen", return_value=process) as spawn,
            patch.object(job, "stop_child") as stop,
            self.assertRaisesRegex(ValueError, "stage worker failed"),
        ):
            job.execute(["fixture-worker"], self.work, {}, self.work / "worker.log")
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        stop.assert_called_once()

    def test_stage_proofs_copy_hash_verified_bytes_to_stable_namespace(self):
        raw = self.root / "original-report.json"
        raw_sha = job.s.atomic(raw, {"passed": True, "job_id": "123"})
        terminal = "123|COMPLETED|0:0|00:00:10|1K|123\n"
        status = {
            "job_id": "123",
            "accounting": {"stdout": terminal},
            "submission": {"job_id": "123"},
            "publications": [{"path": str(raw), "sha256": raw_sha}],
            "reports": [{"path": str(raw), "sha256": raw_sha}],
        }
        result = job.stage_proofs({"statuses": {"prepare": status}}, self.work, self.namespace)
        record = result["prepare"]
        self.assertEqual(record["terminal_path"], str(self.namespace / "proofs/prepare/terminal.txt"))
        self.assertEqual(record["terminal_sha256"], job.s.sha(terminal.encode()))
        self.assertEqual(record["reports"][0]["path"], str(self.namespace / "proofs/prepare/reports-0.json"))
        self.assertEqual((self.work / "proofs/prepare/reports-0.json").read_bytes(), raw.read_bytes())


if __name__ == "__main__":
    unittest.main()
