"""Execute the real bounded deployment script on CPU-only filesystem fixtures."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "pipeline_prepare_fixture", ROOT / "tools/sdsc_pipeline_prepare.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)
s = prepare.s


class PipelinePrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pipeline-prepare-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.control = self.root / "control"
        self.release = self.control / "releases/run-fixture"
        self.source = self.release / "source"
        tool = self.source / "tools/control.py"
        tool.parent.mkdir(parents=True)
        tool.write_bytes(b"# reviewed CPU fixture\n")
        record = {"path": "tools/control.py", **s.identity(tool), "mode": 0o644}
        self.manifest = {
            "run_id": "run-fixture",
            "code_sha256": s.sha(s.canonical([record])),
            "files": [record],
        }
        s.atomic(self.release / "manifest.json", self.manifest)
        self.g0 = self.root / "base/bin/python3.12"
        self.g0.parent.mkdir(parents=True)
        self.g0.write_bytes(b"# explicit non-running G0 interpreter fixture\n")
        self.g0.chmod(0o755)
        pilot = self.root / "pilot/bin/python"
        pilot.parent.mkdir(parents=True)
        observed = {
            "python_path": str(pilot),
            "base_prefix": str(self.g0.parent.parent),
            "python": "3.12.13",
            "packages": {"trl": "0.22.2"},
        }
        # A bounded executable probe fixture, not a Python/GPU implementation.
        pilot.write_text("#!" + sys.executable + "\nprint(" + repr(json.dumps(observed)) + ")\n")
        pilot.chmod(0o755)
        image = self.root / "fixed.sif"
        image.write_bytes(b"CPU fixture container identity")
        engine = self.root / "singularity"
        engine.write_bytes(b"# never invoked\n")
        engine.chmod(0o755)
        vendor = self.root / "vendor"
        vendor.mkdir()
        vendor_manifest = self.root / "vendor.json"
        vendor_sha = s.atomic(vendor_manifest, {"head": "b" * 40, "files": []})
        wrapper = s.canonical(self.manifest) + b"\n"
        bundle = b"CPU fixture immutable bundle"
        proof = {
            "git_head": "0215c356355b29b5e2b407978a207db2156719e1",
            "wrapper": {
                "run_id": self.manifest["run_id"],
                "code_sha256": self.manifest["code_sha256"],
                "manifest_sha256": s.sha(wrapper),
            },
            "bundle": {"path": "history.bundle", "size": len(bundle), "sha256": s.sha(bundle)},
        }
        proof_raw = s.canonical(proof) + b"\n"
        proof_sha = s.sha(proof_raw)
        self.provenance = self.control / "provenance" / proof_sha
        self.provenance.mkdir(parents=True)
        (self.provenance / "manifest.json").write_bytes(proof_raw)
        (self.provenance / "wrapper-manifest.json").write_bytes(wrapper)
        (self.provenance / "history.bundle").write_bytes(bundle)
        self.plan = {
            "flow_id": "flow-fixture",
            "authorized": True,
            "run_id": self.manifest["run_id"],
            "code_sha256": self.manifest["code_sha256"],
            "science_head": proof["git_head"],
            "provenance_sha256": proof_sha,
            "provenance_dir": str(self.provenance),
            "control_files": {record["path"]: record["sha256"]},
            "g0_python": str(self.g0),
            "g0_python_sha256": s.identity(self.g0)["sha256"],
            "stages": list(prepare.remote.STAGES),
            "resources": prepare.remote.resource_plan(),
            "environment": {
                "runtime": {**observed, "python_sha256": s.identity(pilot)["sha256"]},
                "container": {
                    "runtime": str(engine),
                    "image": str(image),
                    "size": image.stat().st_size,
                    "mtime_ns": image.stat().st_mtime_ns,
                },
                "mib": {
                    "path": str(vendor),
                    "manifest": str(vendor_manifest),
                    "manifest_sha256": vendor_sha,
                    "head": "b" * 40,
                },
            },
        }

    def deploy(self, plan):
        # Only fixed fixture roots/login identity are substituted. Execute the
        # production script itself with its original validation and writes.
        script = prepare.DEPLOY_SCRIPT.replace(
            "'/home/zgao12/quest-runs/OPD'", repr(str(self.control))
        ).replace("'/home/zgao12'", repr(str(Path.home())))
        environment = {**os.environ, "SSH_CONNECTION": "CPU fixture only"}
        environment.pop("SLURM_JOB_ID", None)
        return subprocess.run(
            [sys.executable, "-I", "-B", "-c", script],
            input=s.canonical(plan),
            capture_output=True,
            timeout=10,
            env=environment,
        )

    def test_review_requires_exact_frozen_release_size_mode_and_hash(self):
        review = {"files": self.plan["control_files"]}
        with patch.object(prepare, "CONTROL_FILES", ("tools/control.py",)):
            self.assertEqual(prepare.validated_controls(self.source, self.manifest, review), review["files"])
            (self.source / "tools/control.py").write_bytes(b"# newer un-frozen bytes\n")
            review["files"]["tools/control.py"] = s.identity(self.source / "tools/control.py")["sha256"]
            with self.assertRaisesRegex(ValueError, "frozen release"):
                prepare.validated_controls(self.source, self.manifest, review)

    def test_real_deploy_roundtrip_remote_plan_reader_and_duplicate_refusal(self):
        response = self.deploy(self.plan)
        self.assertEqual(response.returncode, 0, response.stderr.decode())
        receipt = json.loads(response.stdout)
        self.assertFalse(receipt["started"])
        expected = s.sha(s.canonical(self.plan) + b"\n")
        self.assertEqual(receipt["plan_sha256"], expected)
        with patch.object(s, "CONTROL", self.control):
            plan, directory = prepare.remote.plan_for(
                {"flow_id": self.plan["flow_id"], "plan_sha256": expected}
            )
        self.assertEqual(plan, self.plan)
        before = (directory / "plan.json").read_bytes()
        repeated = self.deploy(self.plan)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn(b"flow already exists", repeated.stderr)
        self.assertEqual((directory / "plan.json").read_bytes(), before)
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ["plan.json"])

    def test_changed_inputs_refuse_before_remote_claim(self):
        bad_plans = []
        for field in ("python_sha256", "python", "base_prefix"):
            plan = copy.deepcopy(self.plan)
            plan["environment"]["runtime"][field] = "0" * 64 if field == "python_sha256" else "/changed"
            bad_plans.append(plan)
        image = copy.deepcopy(self.plan)
        image["environment"]["container"]["size"] += 1
        bad_plans.append(image)
        provenance = copy.deepcopy(self.plan)
        provenance["provenance_sha256"] = "0" * 64
        bad_plans.append(provenance)
        controls = copy.deepcopy(self.plan)
        controls["control_files"]["tools/control.py"] = "0" * 64
        bad_plans.append(controls)
        for plan in bad_plans:
            with self.subTest(plan=plan):
                response = self.deploy(plan)
                self.assertNotEqual(response.returncode, 0)
                self.assertFalse((self.control / "pipelines" / plan["flow_id"]).exists())
        bundle = self.provenance / "history.bundle"
        bundle.unlink()
        bundle.symlink_to(self.g0)
        response = self.deploy(self.plan)
        self.assertNotEqual(response.returncode, 0)
        self.assertIn(b"symlink", response.stderr)
        self.assertFalse((self.control / "pipelines" / self.plan["flow_id"]).exists())


if __name__ == "__main__":
    unittest.main()
