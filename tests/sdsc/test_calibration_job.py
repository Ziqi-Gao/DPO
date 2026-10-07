"""CPU fixtures for upstream binding, immutable input staging and final publication."""

import ast
import contextlib
import hashlib
import io
import json
import signal
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/sdsc_calibration_job.sh"


def functions():
    source = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    definitions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(definitions, str(SCRIPT), "exec"), namespace)
    return namespace


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CalibrationJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.helpers = functions()
        self.project = self.root / "project"
        self.releases = self.root / "releases"
        self.submission = self.root / "submissions" / "intent-fixture"
        self.submission.mkdir(parents=True)
        self.work = self.root / "work"
        self.proofs = self.work / "proofs"
        self.proofs.mkdir(parents=True)
        self.destination = self.work / "inputs" / "qwen3-v2"
        self.identity = dict(
            task="qwen3-v2-g0-calibration",
            job_id="900",
            run_id="calibration-fixture",
            code_sha256="a" * 64,
            hf_home="/verified/huggingface",
            provenance_manifest_sha256="b" * 64,
            science_git_head="c" * 40,
            bundle_sha256="d" * 64,
        )
        self.helpers.update(
            project=self.project,
            control_releases=self.releases,
            submission=str(self.submission),
            run_id=self.identity["run_id"],
            code_hash=self.identity["code_sha256"],
            identity=self.identity,
            mount=lambda path: {"fstype": "lustre"},
            process=None,
        )
        self.addCleanup(patch.stopall)
        patch.object(
            self.helpers["shutil"], "disk_usage", return_value=types.SimpleNamespace(free=2**40)
        ).start()
        self.payload = {
            "schema": "quest-sdsc-calibration-prerequisites-v1",
            "task": self.identity["task"],
            "target": {
                "run_id": self.identity["run_id"],
                "code_sha256": "a" * 64,
                "intent_id": self.submission.name,
            },
            "upstream": {
                role: self.upstream(role, str(index))
                for index, role in enumerate(("teacher", "preflight"), 100)
            },
        }
        self.write_prerequisites()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, indent=2) + "\n").encode()
        path.write_bytes(raw)
        return sha(raw)

    def record(self, base, name, raw):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": name, "size": len(raw), "sha256": sha(raw)}

    def upstream(self, role, job):
        run = role + "-fixture"
        task = "qwen3-v2-teacher-prepare" if role == "teacher" else "qwen3-v2-preflight"
        root = self.project / role
        source_record = {"path": "src/science.py", "size": 1, "sha256": sha(b"x"), "mode": 420}
        code = sha(self.helpers["canonical"]([source_record]))
        receipt = {
            "task": task,
            "job_id": job,
            "run_id": run,
            "code_sha256": code,
            "result_dir": str(root),
            "hf_home": "/verified/huggingface",
        }
        report = {
            **receipt,
            "passed": True,
            "exit_code": 0,
            "g0_passed": False,
            "execution_class_certified": False,
        }
        report_name = "teacher-prepare.json" if role == "teacher" else "preflight.json"
        report_hash = self.write_json(root / report_name, report)
        entries = [{"path": report_name, "size": (root / report_name).stat().st_size, "sha256": report_hash}]
        if role == "teacher":
            for name in (
                "manifest.json",
                "train/examples.jsonl",
                "validation/examples.jsonl",
                "iid_test/examples.jsonl",
                "ood_depth_test/examples.jsonl",
                "ood_structure_test/examples.jsonl",
                "circuit_discovery/examples.jsonl",
                "circuit_validation/examples.jsonl",
            ):
                entries.append(self.record(root, "artifacts/dataset/" + name, b"CPU fixture dataset\n"))
            for name in ("manifest.json", "ledger.json", "accepted_view.json"):
                entries.append(self.record(root, "artifacts/teacher_demos/" + name, b"CPU fixture teacher\n"))
        publication = {
            **receipt,
            "passed": True,
            "persisted": True,
            "persistent_read_back_verified": True,
            "files": entries,
        }
        publication_hash = self.write_json(root / "receipt.json", publication)
        manifest = {
            "schema": "quest-sdsc-snapshot-v1",
            "run_id": run,
            "code_sha256": code,
            "files": [source_record],
        }
        manifest_path = self.releases / run / "manifest.json"
        manifest_hash = self.write_json(manifest_path, manifest)
        status = {
            "task": task,
            "run_id": run,
            "job_id": job,
            "state": "COMPLETED",
            "success": True,
            "queue": {"returncode": 0, "stdout": ""},
            "accounting": {
                "returncode": 0,
                "stdout": job + "|COMPLETED|0:0|100|\n" + job + ".batch|COMPLETED|0:0|100|\n",
            },
            "result": {"verified": True, "result_sha256": report_hash},
        }
        return {
            "receipt": receipt,
            "status": status,
            "result_dir": str(root),
            "report": report,
            "report_path": str(root / report_name),
            "report_sha256": report_hash,
            "publication_receipt": publication,
            "publication_receipt_path": str(root / "receipt.json"),
            "publication_receipt_sha256": publication_hash,
            "release_manifest": manifest,
            "release_manifest_path": str(manifest_path),
            "release_manifest_sha256": manifest_hash,
        }

    def write_prerequisites(self):
        path = self.submission / "prerequisites.json"
        digest = self.write_json(path, self.payload)
        self.helpers.update(prerequisites_path=str(path), prerequisites_hash=digest)
        self.identity.update(prerequisites_path=str(path), prerequisites_sha256=digest)

    def stage(self):
        return self.helpers["stage_prerequisites"](self.proofs, self.destination)

    def test_exact_inputs_and_original_proof_bytes_are_staged(self):
        preflight = Path(self.payload["upstream"]["preflight"]["result_dir"])
        (preflight / "unrelated-checkpoint.pt").write_bytes(b"not a training input")
        value, evidence = self.stage()
        self.assertEqual(value, self.payload)
        self.assertEqual(evidence["files"], 11)
        self.assertTrue(evidence["read_back_verified"])
        self.assertEqual(self.identity["teacher_job_id"], "100")
        self.assertEqual(self.identity["preflight_job_id"], "101")
        for role in ("teacher", "preflight"):
            proof = self.payload["upstream"][role]
            self.assertEqual(
                (self.proofs / (role + "-report.json")).read_bytes(), Path(proof["report_path"]).read_bytes()
            )
        self.assertTrue((self.destination / "dataset/train/examples.jsonl").is_file())
        self.assertFalse((self.destination / "teacher_demos/ledger.json").stat().st_mode & 0o222)
        self.assertFalse((self.destination / "unrelated-checkpoint.pt").exists())

    def test_trusted_manifest_hash_and_target_must_match(self):
        self.helpers["prerequisites_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "proof SHA"):
            self.stage()
        self.payload["target"]["intent_id"] = "another-intent"
        self.write_prerequisites()
        with self.assertRaisesRegex(ValueError, "target differs"):
            self.stage()

    def test_report_bytes_cannot_change_after_submission(self):
        path = Path(self.payload["upstream"]["teacher"]["report_path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "proof SHA"):
            self.stage()

    def test_failed_accounting_step_is_not_success(self):
        status = self.payload["upstream"]["teacher"]["status"]
        status["accounting"]["stdout"] += "100.extern|FAILED|1:0|1|\n"
        self.write_prerequisites()
        with self.assertRaisesRegex(ValueError, "accounting"):
            self.stage()

    def test_still_queued_job_is_rejected(self):
        self.payload["upstream"]["teacher"]["status"]["queue"]["stdout"] = "100|RUNNING|node\n"
        self.write_prerequisites()
        with self.assertRaisesRegex(ValueError, "still queued"):
            self.stage()

    def test_teacher_payload_corruption_is_rejected(self):
        root = Path(self.payload["upstream"]["teacher"]["result_dir"])
        path = root / "artifacts/teacher_demos/ledger.json"
        raw = path.read_bytes()
        path.write_bytes(b"!" + raw[1:])
        with self.assertRaisesRegex(ValueError, "artifact SHA"):
            self.stage()

    def test_unlisted_teacher_file_and_symlink_are_rejected(self):
        root = Path(self.payload["upstream"]["teacher"]["result_dir"])
        path = root / "artifacts/dataset/extra"
        path.write_bytes(b"unlisted")
        with self.assertRaisesRegex(ValueError, "input tree differs"):
            self.stage()
        path.unlink()
        for proof in self.proofs.iterdir():
            proof.unlink()
        path.symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.stage()

    def valid_report(self):
        return {
            **self.identity,
            "passed": True,
            "exit_code": 0,
            "g0_passed": False,
            "factorial_ready": False,
            "pilot_passed": False,
            "execution_class_certified": False,
        }

    def test_report_cannot_claim_g0_pilot_or_certification(self):
        self.helpers["validate_report"](self.valid_report(), 0)
        for key in ("g0_passed", "pilot_passed", "execution_class_certified", "factorial_ready"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.helpers["validate_report"]({**self.valid_report(), key: True}, 0)

    def test_failure_publication_preserves_stable_checkpoint_and_logs_without_resume_claim(self):
        output, result = self.work / "outputs/qwen3-v2", self.project / "calibration"
        result.mkdir(parents=True)
        self.helpers["result_root"] = result
        self.record(output, "canonical_sft/checkpoints/step-00000020.pt", b"CPU fixture metadata")
        self.record(
            output, "canonical_sft/checkpoints/step-00000020.accelerate/model.bin", b"CPU fixture state"
        )
        self.record(output, "canonical_sft/checkpoints/.partial.tmp", b"incomplete diagnostic")
        original = {"passed": False, "phase": "train"}
        self.write_json(output / "g0-calibration.json", original)
        log = self.work / "worker.log"
        log.write_text("CPU fixture failure\n")
        report = {**self.valid_report(), "passed": False, "exit_code": 1, "resumable": False}
        with contextlib.redirect_stdout(io.StringIO()):
            self.helpers["publish"](output, log, report, False)
        receipt = json.loads((result / "receipt.json").read_text())
        self.assertFalse(receipt["passed"])
        self.assertTrue(receipt["persistent_read_back_verified"])
        self.assertFalse(json.loads((result / "g0-calibration.json").read_text())["resumable"])
        for entry in receipt["files"]:
            path = result / entry["path"]
            self.assertEqual(sha(path.read_bytes()), entry["sha256"])
            self.assertEqual(path.stat().st_size, entry["size"])
        self.assertEqual(json.loads((result / "artifacts/g0-calibration.json").read_text()), original)

    def test_cannot_publish_while_child_is_writing(self):
        self.helpers["process"] = types.SimpleNamespace(poll=lambda: None)
        with self.assertRaisesRegex(ValueError, "worker is writing"):
            self.helpers["publish"](None, None, self.valid_report(), True)

    def test_stop_child_bounds_wait_and_signals_whole_owned_group(self):
        calls = []
        process = types.SimpleNamespace(pid=1234)
        waits = []

        def wait(timeout):
            waits.append(timeout)
            if timeout == 90:
                raise subprocess.TimeoutExpired("fixture", timeout)

        process.wait = wait
        self.helpers["process"] = process
        with patch.object(
            self.helpers["os"], "killpg", side_effect=lambda pid, sig: calls.append((pid, sig))
        ):
            self.helpers["stop_child"]()
        self.assertEqual(waits, [90, 10])
        self.assertEqual(calls[0], (1234, signal.SIGTERM))
        self.assertIn((1234, signal.SIGKILL), calls)


if __name__ == "__main__":
    unittest.main()
