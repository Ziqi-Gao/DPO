"""Teacher wrapper fixtures: real isolated Git restoration, no SSH/GPU/Slurm."""

import ast
import importlib.util
import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "teacher_git_fixtures", Path(__file__).with_name("test_provenance.py")
)
fixture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture_module)


def functions():
    script = (REPO / "tools/sdsc_teacher_job.sh").read_text()
    module = ast.parse(script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0])
    definitions = ast.Module(
        body=[
            node for node in module.body if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(definitions, "sdsc_teacher_job.sh", "exec"), namespace)
    return namespace


class TeacherJobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.helpers = functions()
        self.identity = dict(
            task="qwen3-v2-teacher-prepare",
            job_id="12345",
            run_id="teacher-fixture",
            code_sha256="a" * 64,
            hf_home="/verified/huggingface",
            provenance_manifest_sha256="b" * 64,
            science_git_head="c" * 40,
            bundle_sha256="d" * 64,
        )
        self.helpers.update(identity=self.identity, result_root=self.root)

    def git_fixture(self):
        fixture = fixture_module.ProvenanceTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        prepared = fixture.prepare()
        artifact = Path(prepared["artifact"])
        self.helpers.update(
            provenance_dir=str(artifact),
            control_provenance=artifact.parent,
            provenance_hash=prepared["manifest_sha256"],
            code_hash=prepared["wrapper_code_sha256"],
        )
        return fixture, prepared

    def test_restores_real_clean_science_separately_from_dirty_wrapper(self):
        fixture, prepared = self.git_fixture()
        # Keep restored readonly files under the fixture's permission-aware cleanup.
        work = Path(fixture.temp.name) / "work"
        work.mkdir()
        destination = work / "science"
        evidence, local_manifest = self.helpers["restore_science"](REPO, destination)
        self.assertEqual(evidence["git_head"], fixture.head)
        self.assertEqual(fixture_module.provenance.git(destination, "status", "--porcelain"), b"")
        self.assertEqual((destination / "docs/guide.md").read_bytes(), b"committed docs\n")
        self.assertEqual((fixture.root / "docs/guide.md").read_bytes(), b"dirty wrapper docs\n")
        self.assertFalse((destination / "tools/wrapper.py").exists())
        self.assertFalse((destination / "src/science.py").stat().st_mode & stat.S_IWUSR)
        self.assertEqual(self.helpers["file_hash"](local_manifest), prepared["manifest_sha256"])

    def test_wrong_trusted_provenance_hash_cannot_restore(self):
        fixture, _prepared = self.git_fixture()
        self.helpers["provenance_hash"] = "0" * 64
        destination = Path(fixture.temp.name) / "rejected-science"
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            self.helpers["restore_science"](REPO, destination)
        self.assertFalse(destination.exists())

    def test_artifact_outside_project_control_root_is_rejected(self):
        fixture, _prepared = self.git_fixture()
        self.helpers["control_provenance"] = self.root / "different-control-root"
        with self.assertRaisesRegex(ValueError, "separate artifact"):
            self.helpers["restore_science"](REPO, Path(fixture.temp.name) / "science")

    def valid_report(self):
        return dict(
            self.identity,
            passed=True,
            exit_code=0,
            g0_passed=False,
            execution_class_certified=False,
            partial_attempt_ledger_guaranteed=False,
        )

    def test_result_gate_binds_science_and_provenance_without_claiming_g0_or_resume(self):
        report = self.valid_report()
        self.helpers["validate_report"](report, 0)
        for changes in (
            dict(task="qwen3-v2-preflight"),
            dict(science_git_head="e" * 40),
            dict(provenance_manifest_sha256="e" * 64),
            dict(job_id="54321"),
            dict(passed=False),
            dict(g0_passed=True),
            dict(execution_class_certified=True),
            dict(partial_attempt_ledger_guaranteed=True),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.helpers["validate_report"](dict(report, **changes), 0)
        with self.assertRaises(ValueError):
            self.helpers["validate_report"](report, 1)

    def test_failed_run_preserves_dataset_existing_ledger_and_original_worker_report(self):
        output = self.root / "node-output"
        (output / "dataset").mkdir(parents=True)
        (output / "teacher_demos").mkdir()
        (output / "dataset/train.jsonl").write_bytes(b'{"prompt_id":"fixture"}\n')
        (output / "teacher_demos/attempt-ledger.jsonl").write_bytes(b'{"accepted":false}\n')
        original = {"phase": "teacher", "passed": False, "error": "fixture rejection"}
        (output / "teacher-prepare.json").write_text(json.dumps(original))
        log = self.root / "node.log"
        log.write_text("fixture diagnostic\n")
        report = dict(self.valid_report(), passed=False, exit_code=1, resumable=False)
        with redirect_stdout(io.StringIO()):
            self.helpers["publish"](output, log, report, False)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertFalse(receipt["passed"])
        self.assertTrue(receipt["persistent_read_back_verified"])
        for entry in receipt["files"]:
            path = self.root / entry["path"]
            self.assertEqual(entry["size"], path.stat().st_size)
            self.assertEqual(entry["sha256"], self.helpers["file_hash"](path))
        self.assertEqual(json.loads((self.root / "artifacts/teacher-prepare.json").read_text()), original)
        self.assertTrue((self.root / "artifacts/teacher_demos/attempt-ledger.jsonl").is_file())
        self.assertTrue((self.root / "artifacts/dataset/train.jsonl").is_file())

    def test_interrupted_generation_does_not_invent_missing_ledger(self):
        output = self.root / "node-output"
        output.mkdir()
        report = dict(self.valid_report(), passed=False, exit_code=1, resumable=False)
        with redirect_stdout(io.StringIO()):
            self.helpers["publish"](output, None, report, False)
        self.assertFalse((self.root / "artifacts/teacher_demos").exists())
        published = json.loads((self.root / "teacher-prepare.json").read_text())
        self.assertFalse(published["partial_attempt_ledger_guaranteed"])
        self.assertFalse(published["resumable"])


if __name__ == "__main__":
    unittest.main()
