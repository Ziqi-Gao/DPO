"""Real local staging/publication and owned-child shutdown; no SSH, Slurm or GPU."""

import hashlib
import importlib.util
import io
import os
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]


def load_tool(name):
    spec = importlib.util.spec_from_file_location("fixture_" + name, REPO / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StudentJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.cleanup)
        self.job = load_tool("sdsc_student_job")
        self.contract = load_tool("sdsc_student_contract")
        self.results = self.root / "persistent"
        self.results.mkdir()
        self.job.result_root = self.results
        self.job.identity = dict(
            task="qwen3-v2-adapted-preflight",
            job_id="123456",
            run_id="student-fixture",
            code_sha256="a" * 64,
            science_git_head="b" * 40,
        )

    def cleanup(self):
        for directory, _, _ in os.walk(self.root):
            Path(directory).chmod(0o700)
        self.temporary.cleanup()

    def release(self):
        root = self.root / "release"
        files = []
        for name in (
            "sdsc_student_job.sh",
            "sdsc_student_job.py",
            "sdsc_student_contract.py",
            "sdsc_provenance.py",
            "sdsc_adapted_training_preflight.py",
            "sdsc_adapted_calibration.py",
        ):
            path = root / "source/tools" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = ("fixture content for " + name + "\n").encode()
            path.write_bytes(data)
            mode = 0o755 if name.endswith(".sh") else 0o644
            path.chmod(mode)
            files.append(
                dict(path="tools/" + name, size=len(data), mode=mode, sha256=hashlib.sha256(data).hexdigest())
            )
        self.job.run_id = "student-fixture"
        self.save_manifest(root, files)
        destination = self.root / "node-source"
        destination.mkdir()
        return root, destination, files

    def save_manifest(self, root, files):
        self.job.code_hash = hashlib.sha256(self.job.canonical(files)).hexdigest()
        (root / "manifest.json").write_bytes(
            self.job.canonical(dict(files=files, code_sha256=self.job.code_hash, run_id=self.job.run_id))
        )

    def report(self, passed):
        return dict(self.job.identity, passed=passed, exit_code=0 if passed else 1)

    def output(self):
        root = self.root / "node-output"
        (root / "nested").mkdir(parents=True)
        (root / "nested/metrics.json").write_bytes(b'{"loss":0.25}\n')
        log = self.root / "worker.log"
        log.write_bytes(b"worker evidence\n")
        return root, log

    def test_stage_release_verifies_bytes_and_keeps_executable_mode_readonly(self):
        root, destination, files = self.release()
        self.job.stage_source(root, destination)
        for row in files:
            target = destination / row["path"]
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), row["sha256"])
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o555 if row["mode"] == 0o755 else 0o444)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o555)

    def test_stage_release_rejects_same_size_changed_source(self):
        root, destination, files = self.release()
        source = root / "source" / files[0]["path"]
        source.write_bytes(b"X" * source.stat().st_size)
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.job.stage_source(root, destination)

    def test_stage_release_rejects_duplicate_path_even_with_new_manifest_hash(self):
        root, destination, files = self.release()
        files.append(dict(files[0]))
        self.save_manifest(root, files)
        with self.assertRaisesRegex(ValueError, "duplicate manifest path"):
            self.job.stage_source(root, destination)

    def test_stage_release_rejects_symlink_source(self):
        root, destination, files = self.release()
        source = root / "source" / files[0]["path"]
        outside = self.root / "outside.sh"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            self.job.stage_source(root, destination)

    def test_input_staging_rejects_changed_source_and_does_not_reuse_destination(self):
        source = self.root / "original.json"
        source.write_bytes(b"original")
        record = dict(
            self.contract.file_record(source), path="teacher-store/manifest.json", source=str(source)
        )
        destination = self.root / "teacher-inputs"
        evidence = self.contract.stage_inputs([record], destination, maximum=1024)
        self.assertTrue(evidence["read_back_verified"])
        self.assertEqual((destination / record["path"]).read_bytes(), b"original")
        with self.assertRaisesRegex(ValueError, "fresh nonempty"):
            self.contract.stage_inputs([record], destination, maximum=1024)
        source.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "staged input hash differs"):
            self.contract.stage_inputs([record], self.root / "tampered-inputs", maximum=1024)

    def test_success_receipt_follows_actual_persistent_readback_and_validation(self):
        output, log = self.output()
        report = self.report(True)
        original = self.contract.validate_publication
        checks = []

        def validate(root, actual_report, publication):
            self.assertFalse((root / "receipt.json").exists())
            checks.append(original(root, actual_report, publication))

        with (
            patch.object(self.contract, "validate_publication", side_effect=validate),
            redirect_stdout(io.StringIO()),
        ):
            self.job.publish(output, log, report, self.contract)
        self.assertEqual(len(checks), 1)
        receipt = self.contract.document(self.results / "receipt.json")
        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["persistent_read_back_verified"])
        self.assertEqual(
            {row["path"] for row in receipt["files"]},
            {"worker.log", "artifacts/nested/metrics.json", "adapted-preflight.json"},
        )
        for row in receipt["files"]:
            actual = self.contract.file_record(self.results / row["path"])
            self.assertEqual((actual["size"], actual["sha256"]), (row["size"], row["sha256"]))
        self.assertEqual((output / "nested/metrics.json").read_bytes(), b'{"loss":0.25}\n')

    def test_failed_attempt_persists_original_partial_evidence_without_success(self):
        output, log = self.output()
        with redirect_stdout(io.StringIO()):
            self.job.publish(output, log, self.report(False), self.contract)
        receipt = self.contract.document(self.results / "receipt.json")
        self.assertFalse(receipt["passed"])
        self.assertTrue(receipt["persisted"])
        self.assertEqual((self.results / "artifacts/nested/metrics.json").read_bytes(), b'{"loss":0.25}\n')

    def test_failed_persistent_readback_never_writes_receipt(self):
        output, log = self.output()
        original = self.contract.file_record

        def changed(path, **kwargs):
            record = original(path, **kwargs)
            return dict(record, sha256="0" * 64) if path.name == "metrics.json" else record

        with (
            patch.object(self.contract, "file_record", side_effect=changed),
            self.assertRaisesRegex(ValueError, "failed read-back"),
        ):
            self.job.publish(output, log, self.report(True), self.contract)
        self.assertFalse((self.results / "receipt.json").exists())

    def test_semantic_publication_failure_never_writes_receipt(self):
        output, log = self.output()
        with (
            patch.object(
                self.contract, "validate_publication", side_effect=ValueError("scientific artifact mismatch")
            ),
            self.assertRaisesRegex(ValueError, "scientific artifact mismatch"),
        ):
            self.job.publish(output, log, self.report(True), self.contract)
        self.assertFalse((self.results / "receipt.json").exists())

    def test_existing_publication_is_immutable(self):
        output, log = self.output()
        with redirect_stdout(io.StringIO()):
            self.job.publish(output, log, self.report(True), self.contract)
        original = {
            path.relative_to(self.results): path.read_bytes()
            for path in self.results.rglob("*")
            if path.is_file()
        }
        (output / "nested/metrics.json").write_bytes(b'{"loss":999}\n')
        with self.assertRaisesRegex(ValueError, "receipt|published|publication"):
            self.job.publish(output, log, self.report(False), self.contract)
        self.assertEqual(
            original,
            {
                path.relative_to(self.results): path.read_bytes()
                for path in self.results.rglob("*")
                if path.is_file()
            },
        )

    def test_publication_rejects_output_symlink(self):
        output, log = self.output()
        (output / "external").symlink_to(log)
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            self.job.publish(output, log, self.report(True), self.contract)
        self.assertFalse((self.results / "receipt.json").exists())

    def test_live_worker_cannot_publish(self):
        class LiveProcess:
            def poll(self):
                return None

        self.job.process = LiveProcess()
        with self.assertRaisesRegex(ValueError, "live worker"):
            self.job.publish(None, None, self.report(True), self.contract)
        self.assertEqual(list(self.results.iterdir()), [])

    def test_term_handler_larger_than_pipe_is_drained_with_bounded_retained_tail(self):
        code = """
import os, signal
def stop(signum, frame):
    for _ in range(64):
        os.write(1, b'x' * 65536)
    os.write(1, b'FINAL-SHUTDOWN-EVIDENCE\\n')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
os.write(1, b'READY\\n')
while True:
    signal.pause()
"""
        child = subprocess.Popen(
            [sys.executable, "-I", "-B", "-u", "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )

        def cleanup():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
            if child.stdout and not child.stdout.closed:
                child.stdout.close()

        self.addCleanup(cleanup)
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(5), "owned fixture failed to initialize")
        self.assertEqual(os.read(child.stdout.fileno(), 6), b"READY\n")
        self.job.process = child
        self.job.log = self.root / "shutdown.log"
        self.job.log.write_bytes(b"before signal\n")
        evidence = self.job.stop_child(grace_seconds=5, kill_seconds=2, drain_seconds=1, tail_limit=4096)
        self.assertEqual(evidence["exit_code"], 0)
        self.assertFalse(evidence["sigkill_sent"])
        self.assertTrue(evidence["pipe_drained"])
        self.assertEqual(evidence["received_bytes"], 64 * 65536 + len(b"FINAL-SHUTDOWN-EVIDENCE\n"))
        self.assertEqual(evidence["retained_bytes"], 4096)
        self.assertGreater(evidence["omitted_bytes"], 0)
        content = self.job.log.read_bytes()
        self.assertTrue(content.startswith(b"before signal\n"))
        self.assertIn(b"shutdown log truncated", content)
        self.assertTrue(content.endswith(b"FINAL-SHUTDOWN-EVIDENCE\n"))
        self.assertLess(len(content), 4500)


if __name__ == "__main__":
    unittest.main()
