"""Exercise the real bounded transport and Git verifier without SSH."""

import contextlib
import importlib.util
import io
import json
import sys
import tarfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_provenance as fixtures

provenance = fixtures.provenance

SPEC = importlib.util.spec_from_file_location(
    "upload", Path(__file__).resolve().parents[2] / "tools/sdsc_provenance_upload.py"
)
upload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(upload)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProvenanceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        fixture = self.fixture
        fixture.write("tools/sdsc_provenance.py", Path(provenance.__file__).read_bytes())
        fixture.refresh_wrapper()
        prepared = provenance.prepare(fixture.root, fixture.wrapper_path)
        self.artifact = Path(prepared["artifact"])
        self.wrapper = provenance.load_wrapper(fixture.wrapper_path)
        self.remote = Path(fixture.temp.name) / "remote"
        release = self.remote / "releases" / self.wrapper["run_id"]
        (release / "source/tools").mkdir(parents=True)
        (release / "manifest.json").write_text(json.dumps(self.wrapper))
        (release / "source/tools/sdsc_provenance.py").write_bytes(Path(provenance.__file__).read_bytes())
        self.archive = upload.archive_artifact(self.artifact)
        self.request = {
            "run_id": self.wrapper["run_id"],
            "manifest_sha256": prepared["manifest_sha256"],
            "code_sha256": self.wrapper["code_sha256"],
            "archive_sha256": provenance.sha(self.archive),
        }

    def receive(self, archive=None):
        raw = self.archive if archive is None else archive
        source = upload.REMOTE.replace(
            'pathlib.Path("/home/zgao12/quest-runs/OPD")', f"pathlib.Path({str(self.remote)!r})"
        )
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["remote", json.dumps(self.request)]),
            patch.object(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(raw))),
            patch("pwd.getpwuid", return_value=types.SimpleNamespace(pw_name="zgao12")),
            contextlib.redirect_stdout(output),
        ):
            exec(compile(source, "remote-upload", "exec"), {"__name__": "__main__"})
        return json.loads(output.getvalue())

    def test_real_upload_and_verified_existing_artifact(self):
        first = self.receive()
        self.assertTrue(first["verified"])
        self.assertEqual(first, self.receive())
        destination = Path(first["remote_artifact"])
        self.assertEqual({p.name for p in destination.iterdir()}, set(upload.FILES))
        self.assertEqual(
            (destination / "history.bundle").read_bytes(), (self.artifact / "history.bundle").read_bytes()
        )

    def test_network_corruption_rejected(self):
        with self.assertRaisesRegex(AssertionError, "Upload hash differs"):
            self.receive(self.archive[:-1] + b"x")

    def test_interrupted_permissions_completed_after_content_verification(self):
        destination = Path(self.receive()["remote_artifact"])
        destination.chmod(0o700)
        (destination / "history.bundle").chmod(0o600)
        self.assertTrue(self.receive()["verified"])
        self.assertFalse(destination.stat().st_mode & 0o222)
        self.assertFalse((destination / "history.bundle").stat().st_mode & 0o222)

    def test_existing_extra_file_is_rejected(self):
        destination = Path(self.receive()["remote_artifact"])
        destination.chmod(0o700)
        (destination / "unexpected").write_text("do not recursively chmod this")
        with self.assertRaisesRegex(AssertionError, "Unexpected provenance files"):
            self.receive()

    def test_external_manifest_hash_is_required(self):
        self.request["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            self.receive()

    def test_extra_archive_path_rejected(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            entry = tarfile.TarInfo("../escape")
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b"x"))
        raw = buffer.getvalue()
        self.request["archive_sha256"] = provenance.sha(raw)
        with self.assertRaises(AssertionError):
            self.receive(raw)

    def test_release_verifier_corruption_rejected(self):
        path = self.remote / "releases" / self.wrapper["run_id"] / "source/tools/sdsc_provenance.py"
        path.write_text("raise RuntimeError('untrusted')")
        with self.assertRaises(AssertionError):
            self.receive()


if __name__ == "__main__":
    unittest.main()
