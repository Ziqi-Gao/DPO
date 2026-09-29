"""Real local Git fixtures; no network, GPU, or job submission."""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[2] / "tools/sdsc_provenance.py"
SPEC = importlib.util.spec_from_file_location("sdsc_provenance", TOOL)
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        self.command("init", "--quiet", "--template=")
        self.command("config", "user.name", "Fixture")
        self.command("config", "user.email", "fixture@example.invalid")
        self.write("src/science.py", b"VALUE = 1\n")
        self.write("configs/science.yaml", b"batch: 64\n")
        self.write("prereg/accepted.yaml", b"review: accepted\n")
        self.write("scripts/server_scheduler/handler.py", b"SCIENCE = True\n")
        self.write("deployments/lock.json", b"{}\n")
        self.write("pyproject.toml", b"[project]\nname = 'fixture'\n")
        self.write("docs/guide.md", b"committed docs\n")
        self.write("AGENTS.md", b"committed instructions\n")
        self.write(".gitignore", b".sdsc/\n__pycache__/\n*.ignored.py\n")
        self.command("add", ".")
        self.command("commit", "--quiet", "-m", "Reviewed implementation")
        self.ancestor = self.command("rev-parse", "HEAD").decode().strip()
        self.write(
            "prereg/accepted.yaml", b"review: accepted\nimplementation: " + self.ancestor.encode() + b"\n"
        )
        self.command("add", "prereg")
        self.command("commit", "--quiet", "-m", "Acceptance")
        self.head = self.command("rev-parse", "HEAD").decode().strip()
        self.command("update-ref", "refs/remotes/public/master", self.head)
        self.write("tools/wrapper.py", b"WRAPPER = True\n")
        self.write("docs/guide.md", b"dirty wrapper docs\n")
        self.write("AGENTS.md", b"dirty wrapper instructions\n")
        self.wrapper_path = self.root / ".sdsc/wrapper.json"
        self.refresh_wrapper()

    def cleanup(self):
        for folder, dirs, files in os.walk(self.temp.name):
            Path(folder).chmod(0o700)
            for name in dirs:
                path = Path(folder) / name
                if not path.is_symlink():
                    path.chmod(0o700)
            for name in files:
                path = Path(folder) / name
                if not path.is_symlink():
                    path.chmod(0o600)
        self.temp.cleanup()

    def command(self, *args):
        result = subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            env=provenance.git_env(),
            check=True,
        )
        return result.stdout

    def write(self, path, data):
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
        return full

    def refresh_wrapper(self):
        records = []
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root).as_posix()
            if path.is_dir() or any(p in {".git", ".opd-git", ".sdsc"} for p in path.parts):
                continue
            if path.is_symlink() or "__pycache__" in path.parts:
                continue
            data = path.read_bytes()
            records.append(
                {
                    "path": relative,
                    "size": len(data),
                    "sha256": provenance.sha(data),
                    "mode": 0o755 if path.stat().st_mode & 0o111 else 0o644,
                }
            )
        self.wrapper = {
            "schema": "quest-sdsc-snapshot-v1",
            "run_id": "fixture-wrapper",
            "git_head": self.head,
            "files": records,
            "code_sha256": provenance.sha(provenance.canonical(records)),
            "total_bytes": sum(item["size"] for item in records),
        }
        self.wrapper_path.parent.mkdir(exist_ok=True)
        self.wrapper_path.write_bytes(provenance.canonical(self.wrapper) + b"\n")

    def prepare(self):
        return provenance.prepare(self.root, self.wrapper_path, accepted_ancestors=[self.ancestor])

    def verify(self, prepared, destination=None):
        return provenance.verify(
            prepared["artifact"], prepared["manifest_sha256"], prepared["wrapper_code_sha256"], destination
        )

    def tamper(self, artifact, name, data):
        path = Path(artifact) / name
        path.chmod(0o600)
        path.write_bytes(data)

    def test_roundtrip_real_history_clean_readonly_science_dirty_docs_separate(self):
        before_index = (self.root / ".git/index").read_bytes()
        before_status = self.command("status", "--porcelain=v1")
        prepared = self.prepare()
        restored = Path(self.temp.name) / "restored"
        verified = self.verify(prepared, restored)
        self.assertTrue(verified["verified"])
        self.assertEqual(provenance.git(restored, "rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual(provenance.git(restored, "show", self.ancestor + ":src/science.py"), b"VALUE = 1\n")
        self.assertEqual(provenance.git(restored, "status", "--porcelain"), b"")
        self.assertEqual((restored / "docs/guide.md").read_bytes(), b"committed docs\n")
        self.assertFalse((restored / "tools/wrapper.py").exists())
        self.assertFalse((restored / "src/science.py").stat().st_mode & stat.S_IWUSR)
        self.assertFalse((restored / ".git").stat().st_mode & stat.S_IWUSR)
        self.assertEqual(before_index, (self.root / ".git/index").read_bytes())
        self.assertEqual(before_status, self.command("status", "--porcelain=v1"))
        self.assertEqual((self.root / "docs/guide.md").read_bytes(), b"dirty wrapper docs\n")
        manifest = json.loads((Path(prepared["artifact"]) / "manifest.json").read_bytes())
        self.assertEqual(manifest["accepted_ancestors"], [self.ancestor])
        self.assertIn("pyproject.toml", [item["path"] for item in manifest["scientific_files"]])

    def test_dirty_science_staged_and_unstaged_rejected(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                self.write("src/science.py", b"VALUE = 2\n")
                if staged:
                    self.command("add", "src/science.py")
                self.refresh_wrapper()
                with self.assertRaises(provenance.ProvenanceError):
                    self.prepare()

    def test_assume_unchanged_cannot_hide_modified_bytes(self):
        self.command("update-index", "--assume-unchanged", "src/science.py")
        self.write("src/science.py", b"VALUE = 999\n")
        self.refresh_wrapper()
        with self.assertRaisesRegex(provenance.ProvenanceError, "Uncommitted scientific"):
            self.prepare()

    def test_ignored_scientific_shadow_rejected_bytecode_not_deployed(self):
        self.write("src/__pycache__/science.cpython-311.pyc", b"cache")
        prepared = self.prepare()
        restored = Path(self.temp.name) / "no-cache"
        self.verify(prepared, restored)
        self.assertFalse((restored / "src/__pycache__").exists())
        self.write("src/shadow.ignored.py", b"shadow")
        with self.assertRaisesRegex(provenance.ProvenanceError, "Untracked scientific"):
            self.prepare()

    def test_new_or_deleted_scientific_file_rejected(self):
        self.write("configs/new.yaml", b"changed: true")
        with self.assertRaisesRegex(provenance.ProvenanceError, "Untracked scientific"):
            self.prepare()
        (self.root / "configs/new.yaml").unlink()
        (self.root / "src/science.py").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_stale_wrapper_docs_rejected(self):
        self.write("docs/guide.md", b"newer docs")
        with self.assertRaisesRegex(provenance.ProvenanceError, "Wrapper snapshot is stale"):
            self.prepare()

    def test_metadata_hooks_private_refs_not_copied_or_executed(self):
        marker = Path(self.temp.name) / "hook-ran"
        hook = self.root / ".git/hooks/post-checkout"
        hook.parent.mkdir()
        hook.write_text("#!/bin/sh\ntouch " + str(marker) + "\n")
        hook.chmod(0o755)
        self.command("config", "remote.private.url", "https://example.invalid/private-not-exported")
        self.write("private-branch.txt", b"PRIVATE BRANCH CONTENT")
        self.command("add", "private-branch.txt")
        private_tree = self.command("write-tree").decode().strip()
        private_commit = (
            subprocess.run(
                ["git", "-C", str(self.root), "commit-tree", private_tree, "-p", self.head],
                input=b"Private branch\n",
                capture_output=True,
                env=provenance.git_env(),
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        self.command("update-ref", "refs/heads/private", private_commit)
        # Fixture-only undo of the staged private file; production never edits the source index.
        self.command("rm", "--cached", "private-branch.txt")
        (self.root / "private-branch.txt").unlink()
        prepared = self.prepare()
        restored = Path(self.temp.name) / "isolated"
        self.verify(prepared, restored)
        self.assertFalse(marker.exists())
        self.assertFalse((restored / ".git/hooks").exists())
        self.assertNotIn(b"example.invalid", (restored / ".git/config").read_bytes())
        with self.assertRaises(provenance.ProvenanceError):
            provenance.git(restored, "cat-file", "-e", private_commit)
        self.assertEqual(provenance.git(restored, "for-each-ref"), b"")

    def test_bundle_tampering_detected_before_git_import(self):
        prepared = self.prepare()
        path = Path(prepared["artifact"]) / "history.bundle"
        self.tamper(prepared["artifact"], "history.bundle", path.read_bytes() + b"tampering")
        with patch.object(provenance, "init_repo") as init:
            with self.assertRaisesRegex(provenance.ProvenanceError, "Bundle hash"):
                self.verify(prepared)
            init.assert_not_called()

    def test_manifest_or_external_wrapper_anchor_mismatch(self):
        prepared = self.prepare()
        with self.assertRaisesRegex(provenance.ProvenanceError, "Wrapper identity"):
            provenance.verify(prepared["artifact"], prepared["manifest_sha256"], "0" * 64)
        with self.assertRaisesRegex(provenance.ProvenanceError, "manifest hash"):
            provenance.verify(prepared["artifact"], "0" * 64, prepared["wrapper_code_sha256"])

    def test_forged_science_inventory_rejected_against_bundle(self):
        prepared = self.prepare()
        manifest = json.loads((Path(prepared["artifact"]) / "manifest.json").read_bytes())
        manifest["scientific_files"][0]["sha256"] = "0" * 64
        raw = provenance.canonical(manifest) + b"\n"
        self.tamper(prepared["artifact"], "manifest.json", raw)
        prepared["manifest_sha256"] = provenance.sha(raw)
        with self.assertRaisesRegex(provenance.ProvenanceError, "Scientific bytes differ"):
            self.verify(prepared)

    def test_extra_bundle_ref_rejected_even_with_updated_manifest_hash(self):
        prepared = self.prepare()
        artifact = Path(prepared["artifact"])
        bundle = (artifact / "history.bundle").read_bytes()
        extra = self.head.encode() + b" refs/heads/extra\n"
        bundle = bundle.replace(b"\n\nPACK", b"\n" + extra + b"\nPACK", 1)
        self.tamper(artifact, "history.bundle", bundle)
        manifest = json.loads((artifact / "manifest.json").read_bytes())
        manifest["bundle"].update(size=len(bundle), sha256=provenance.sha(bundle))
        raw = provenance.canonical(manifest) + b"\n"
        self.tamper(artifact, "manifest.json", raw)
        prepared["manifest_sha256"] = provenance.sha(raw)
        with self.assertRaisesRegex(provenance.ProvenanceError, "exactly HEAD"):
            self.verify(prepared)

    def test_symlink_science_artifact_and_destination_rejected(self):
        outside = Path(self.temp.name) / "outside.py"
        outside.write_bytes(b"VALUE = 1\n")
        (self.root / "src/science.py").unlink()
        (self.root / "src/science.py").symlink_to(outside)
        with self.assertRaisesRegex(provenance.ProvenanceError, "Symlink"):
            self.prepare()
        (self.root / "src/science.py").unlink()
        self.write("src/science.py", b"VALUE = 1\n")
        prepared = self.prepare()
        artifact = Path(prepared["artifact"])
        linked = Path(self.temp.name) / "linked-artifact"
        linked.symlink_to(artifact, target_is_directory=True)
        with self.assertRaisesRegex(provenance.ProvenanceError, "Symlink"):
            provenance.verify(linked, prepared["manifest_sha256"], prepared["wrapper_code_sha256"])
        with self.assertRaisesRegex(provenance.ProvenanceError, "new"):
            self.verify(prepared, self.root)

    def test_unpublished_head_or_unrelated_ancestor_rejected(self):
        self.command("update-ref", "refs/remotes/public/master", self.ancestor)
        with self.assertRaises(provenance.ProvenanceError):
            self.prepare()
        self.command("update-ref", "refs/remotes/public/master", self.head)
        with self.assertRaises(provenance.ProvenanceError):
            provenance.prepare(self.root, self.wrapper_path, accepted_ancestors=["0" * 40])

    def local_prelude(self, count=1):
        commits = []
        for number in range(count):
            self.write("docs/local-history.md", f"Retained local state {number}\n".encode())
            self.command("add", "docs/local-history.md")
            self.command("commit", "--quiet", "-m", "Local history before reviewed pair")
            commits.append(self.command("rev-parse", "HEAD").decode().strip())
        return commits

    def local_pair(self, hidden_path=None):
        public_base = self.command("rev-parse", "refs/remotes/public/master").decode().strip()
        self.write("src/science.py", b"VALUE = 2\n")
        self.write("prereg/accepted.yaml", b"review: proposed\n")
        self.command("add", "src/science.py", "prereg/accepted.yaml")
        if hidden_path:
            self.write(hidden_path, b"synthetic export rejection fixture\n")
            self.command("add", hidden_path)
        self.command("commit", "--quiet", "-m", "Local successor implementation")
        implementation = self.command("rev-parse", "HEAD").decode().strip()
        self.write(
            "prereg/accepted.yaml", b"review: accepted\nimplementation: " + implementation.encode() + b"\n"
        )
        self.command("add", "prereg/accepted.yaml")
        if hidden_path:
            self.command("rm", "--quiet", hidden_path)
        self.command("commit", "--quiet", "-m", "Local successor acceptance")
        self.head = self.command("rev-parse", "HEAD").decode().strip()
        self.refresh_wrapper()
        return {
            "public_base": public_base,
            "implementation": implementation,
            "acceptance": self.head,
            "commits": self.command("rev-list", "--reverse", public_base + ".." + self.head)
            .decode()
            .splitlines(),
        }

    def prepare_local(self, successor):
        return provenance.prepare(
            self.root,
            self.wrapper_path,
            accepted_ancestors=[self.ancestor],
            reviewed_local_implementation=successor["implementation"],
            reviewed_local_acceptance=successor["acceptance"],
        )

    def test_explicit_local_pair_roundtrip_keeps_genuine_head_and_public_ref(self):
        successor = self.local_pair()
        index = (self.root / ".git/index").read_bytes()
        status = self.command("status", "--porcelain=v1")
        with self.assertRaises(provenance.ProvenanceError):
            self.prepare()  # Default behavior still rejects unpublished HEAD.
        prepared = self.prepare_local(successor)
        self.assertEqual(prepared["local_successor"], successor)
        destination = Path(self.temp.name) / "local-successor"
        verified = self.verify(prepared, destination)
        self.assertEqual(verified["local_successor"], successor)
        self.assertEqual(provenance.git(destination, "rev-parse", "HEAD").decode().strip(), self.head)
        self.assertEqual((destination / "src/science.py").read_bytes(), b"VALUE = 2\n")
        self.assertEqual(provenance.git(destination, "status", "--porcelain"), b"")
        self.assertEqual(
            self.command("rev-parse", "refs/remotes/public/master").decode().strip(), successor["public_base"]
        )
        self.assertEqual((self.root / ".git/index").read_bytes(), index)
        self.assertEqual(self.command("status", "--porcelain=v1"), status)

    def test_local_mode_requires_both_full_shas_and_exact_order(self):
        successor = self.local_pair()
        for implementation, acceptance in (
            (None, successor["acceptance"]),
            (successor["implementation"], None),
            (successor["implementation"][:12], successor["acceptance"]),
            (successor["acceptance"], successor["implementation"]),
            (successor["public_base"], successor["acceptance"]),
            (successor["implementation"], successor["implementation"]),
        ):
            with (
                self.subTest(implementation=implementation, acceptance=acceptance),
                self.assertRaises(provenance.ProvenanceError),
            ):
                provenance.prepare(
                    self.root,
                    self.wrapper_path,
                    reviewed_local_implementation=implementation,
                    reviewed_local_acceptance=acceptance,
                )

    def test_local_history_roundtrip_preserves_every_commit_before_final_pair(self):
        preceding = self.local_prelude(2)
        successor = self.local_pair()
        self.assertEqual(successor["commits"][:-2], preceding)
        before_status = self.command("status", "--porcelain=v1")
        with self.assertRaises(provenance.ProvenanceError):
            self.prepare()
        prepared = self.prepare_local(successor)
        destination = Path(self.temp.name) / "local-history"
        self.assertEqual(self.verify(prepared, destination)["local_successor"], successor)
        self.assertEqual(
            provenance.git(destination, "rev-list", "--reverse", successor["public_base"] + "..HEAD")
            .decode()
            .splitlines(),
            successor["commits"],
        )
        for number, commit in enumerate(preceding):
            self.assertEqual(
                provenance.git(destination, "show", commit + ":docs/local-history.md"),
                f"Retained local state {number}\n".encode(),
            )
        self.assertEqual(before_status, self.command("status", "--porcelain=v1"))
        self.assertEqual(
            self.command("rev-parse", "refs/remotes/public/master").decode().strip(), successor["public_base"]
        )

    def test_64_commit_local_history_roundtrip_preserves_exact_lineage_and_source(self):
        self.local_prelude(62)
        successor = self.local_pair()
        self.assertEqual(len(successor["commits"]), 64)
        index = (self.root / ".git/index").read_bytes()
        status = self.command("status", "--porcelain=v1")
        prepared = self.prepare_local(successor)
        destination = Path(self.temp.name) / "64-commit-history"
        verified = self.verify(prepared, destination)
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["local_successor"], successor)
        self.assertEqual(
            provenance.git(destination, "rev-list", "--reverse", successor["public_base"] + "..HEAD")
            .decode()
            .splitlines(),
            successor["commits"],
        )
        self.assertEqual(provenance.git(destination, "status", "--porcelain"), b"")
        self.assertEqual(self.command("rev-parse", "HEAD").decode().strip(), successor["acceptance"])
        self.assertEqual(
            self.command("rev-parse", "refs/remotes/public/master").decode().strip(), successor["public_base"]
        )
        self.assertEqual((self.root / ".git/index").read_bytes(), index)
        self.assertEqual(self.command("status", "--porcelain=v1"), status)

    def test_65_commit_local_history_rejected_before_export_or_import(self):
        self.local_prelude(63)
        successor = self.local_pair()
        self.assertEqual(len(successor["commits"]), 65)
        with self.assertRaisesRegex(provenance.ProvenanceError, "commit limit"):
            self.prepare_local(successor)
        self.assertFalse((self.root / ".sdsc/provenance").exists())
        # The same actual 65-commit claim is also rejected by the manifest
        # boundary used before any isolated Git import on the consumer side.
        with self.assertRaisesRegex(provenance.ProvenanceError, "bounded unique commit inventory"):
            provenance.validate_local_successor(successor, self.head, successor["public_base"])

    def test_local_mode_rejects_intervening_or_post_acceptance_commits(self):
        successor = self.local_pair()
        self.write("src/extra.py", b"UNREVIEWED = True\n")
        self.command("add", "src/extra.py")
        self.command("commit", "--quiet", "-m", "Extra commit after acceptance")
        self.head = self.command("rev-parse", "HEAD").decode().strip()
        self.refresh_wrapper()
        with self.assertRaisesRegex(provenance.ProvenanceError, "identities or commit order"):
            self.prepare_local(successor)
        # Relabeling the new HEAD as acceptance must not allow a commit between
        # the explicit implementation and acceptance anchors.
        successor["acceptance"] = self.head
        with self.assertRaisesRegex(provenance.ProvenanceError, "identities or commit order"):
            self.prepare_local(successor)

    def test_local_mode_rejects_unsafe_file_even_if_acceptance_deletes_it(self):
        successor = self.local_pair(hidden_path="secrets/fixture.txt")
        self.assertFalse((self.root / "secrets/fixture.txt").exists())
        with self.assertRaisesRegex(provenance.ProvenanceError, "credential path"):
            self.prepare_local(successor)

    def test_local_history_rejects_unsafe_earlier_tree_already_deleted_before_pair(self):
        self.local_prelude(32)
        self.write("secrets/fixture.txt", b"synthetic export rejection fixture\n")
        self.command("add", "secrets/fixture.txt")
        self.command("commit", "--quiet", "-m", "Unsafe historical path fixture")
        self.command("rm", "--quiet", "secrets/fixture.txt")
        self.command("commit", "--quiet", "-m", "Remove path before final pair")
        successor = self.local_pair()
        with self.assertRaisesRegex(provenance.ProvenanceError, "credential path"):
            self.prepare_local(successor)
        # Even a supplied artifact whose producer skipped the audit must fail
        # on restore after its authenticated bundle has been imported.
        with patch.object(provenance, "audit_local_successor"):
            prepared = self.prepare_local(successor)
        with self.assertRaisesRegex(provenance.ProvenanceError, "credential path"):
            self.verify(prepared)

    def test_local_mode_rejects_public_base_inside_final_pair(self):
        successor = self.local_pair()
        for tip in (successor["implementation"], successor["acceptance"]):
            self.command("update-ref", "refs/remotes/public/master", tip)  # Fixture-only mutation.
            with self.subTest(tip=tip), self.assertRaises(provenance.ProvenanceError):
                self.prepare_local(successor)

    def test_local_mode_rejects_nonancestor_public_base(self):
        successor = self.local_pair()
        tree = self.command("rev-parse", "HEAD^{tree}").decode().strip()
        unrelated = self.command("commit-tree", tree, "-m", "Unrelated public fixture").decode().strip()
        self.command("update-ref", "refs/remotes/public/master", unrelated)
        with self.assertRaisesRegex(provenance.ProvenanceError, "merge-base"):
            self.prepare_local(successor)

    def test_local_history_rejects_merge_before_final_pair(self):
        preceding = self.local_prelude(1)[0]
        tree = self.command("rev-parse", "HEAD^{tree}").decode().strip()
        merged = self.command(
            "commit-tree", tree, "-p", preceding, "-p", self.head, "-m", "Historical merge fixture"
        ).decode().strip()
        self.command("update-ref", "HEAD", merged)
        successor = self.local_pair()
        with self.assertRaisesRegex(provenance.ProvenanceError, "must be linear"):
            self.prepare_local(successor)

    def test_local_history_is_rechecked_after_bundle_import(self):
        successor = self.local_pair()
        prepared = self.prepare_local(successor)
        artifact = Path(prepared["artifact"])
        manifest = json.loads((artifact / "manifest.json").read_bytes())
        # A self-consistent claim with the older public ancestor still fails
        # against the actual imported commit graph, despite a recomputed hash.
        manifest["local_successor"]["public_base"] = self.ancestor
        manifest["public_tracking_tip"] = self.ancestor
        raw = provenance.canonical(manifest) + b"\n"
        self.tamper(artifact, "manifest.json", raw)
        prepared["manifest_sha256"] = provenance.sha(raw)
        with self.assertRaisesRegex(provenance.ProvenanceError, "exactly the declared"):
            self.verify(prepared)

    def test_local_history_rejects_missing_forged_or_reordered_earlier_commits_on_restore(self):
        self.local_prelude(2)
        successor = self.local_pair()
        prepared = self.prepare_local(successor)
        artifact = Path(prepared["artifact"])
        original = json.loads((artifact / "manifest.json").read_bytes())
        commits = successor["commits"]
        for changed in (commits[1:], ["0" * 40, *commits[1:]], [commits[1], commits[0], *commits[2:]]):
            manifest = dict(original, local_successor=dict(successor, commits=changed))
            raw = provenance.canonical(manifest) + b"\n"
            self.tamper(artifact, "manifest.json", raw)
            with self.subTest(commits=changed), self.assertRaisesRegex(
                provenance.ProvenanceError, "exactly the declared"
            ):
                provenance.verify(artifact, provenance.sha(raw), prepared["wrapper_code_sha256"])

    def test_local_manifest_rejects_null_extra_fields_and_forged_commit_order(self):
        successor = self.local_pair()
        prepared = self.prepare_local(successor)
        artifact = Path(prepared["artifact"])
        original = json.loads((artifact / "manifest.json").read_bytes())
        for changed in (
            None,
            dict(successor, skip_review=True),
            dict(successor, commits=list(reversed(successor["commits"]))),
            dict(successor, commits=None),
            dict(successor, commits=[None, *successor["commits"]]),
            dict(successor, commits=[successor["public_base"], *successor["commits"]]),
            dict(successor, commits=[successor["implementation"], *successor["commits"]]),
            dict(successor, commits=["0" * 40] * (provenance.MAX_LOCAL_COMMITS + 1)),
        ):
            manifest = dict(original, local_successor=changed)
            raw = provenance.canonical(manifest) + b"\n"
            self.tamper(artifact, "manifest.json", raw)
            with (
                self.subTest(changed=changed),
                patch.object(provenance, "init_repo") as init,
                self.assertRaises(provenance.ProvenanceError),
            ):
                provenance.verify(artifact, provenance.sha(raw), prepared["wrapper_code_sha256"])
            init.assert_not_called()

    def test_local_mode_rejects_dirty_science_after_accepted_pair(self):
        successor = self.local_pair()
        self.write("src/science.py", b"VALUE = 3\n")
        self.refresh_wrapper()
        with self.assertRaisesRegex(provenance.ProvenanceError, "Uncommitted scientific"):
            self.prepare_local(successor)

    def test_opd_git_metadata_supported_without_copy(self):
        (self.root / ".git").rename(self.root / ".opd-git")
        prepared = self.prepare()
        self.assertEqual(prepared["git_head"], self.head)
        self.assertTrue(self.verify(prepared)["verified"])

    def test_committed_symlink_is_never_materialized(self):
        (self.root / "src/link.py").symlink_to("../../outside.py")
        self.command("add", "src/link.py")
        self.command("commit", "--quiet", "-m", "Unsupported link fixture")
        self.head = self.command("rev-parse", "HEAD").decode().strip()
        self.command("update-ref", "refs/remotes/public/master", self.head)
        self.refresh_wrapper()
        with self.assertRaisesRegex(provenance.ProvenanceError, "Committed symlinks"):
            self.prepare()

    def test_wrapper_traversal_or_private_paths_rejected_before_read(self):
        for path in ("../outside.py", "src/../../outside.py", ".ssh/id_rsa", "tools/auth.json"):
            with self.subTest(path=path):
                record = {"path": path, "size": 0, "sha256": "0" * 64, "mode": 0o644}
                with self.assertRaises(provenance.ProvenanceError):
                    provenance.validate_records([record])

    def test_bundle_file_symlink_rejected_before_open(self):
        prepared = self.prepare()
        artifact = Path(prepared["artifact"])
        artifact.chmod(0o700)
        bundle = artifact / "history.bundle"
        outside = Path(self.temp.name) / "outside.bundle"
        outside.write_bytes(bundle.read_bytes())
        bundle.unlink()
        bundle.symlink_to(outside)
        with self.assertRaisesRegex(provenance.ProvenanceError, "Symlink"):
            self.verify(prepared)

    def test_fifo_is_rejected_without_blocking_open(self):
        fifo = Path(self.temp.name) / "unsafe.bundle"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(provenance.ProvenanceError, "regular file"):
            provenance.read_regular(fifo)

    def test_cli_help_has_no_execution_side_effects(self):
        result = subprocess.run([sys.executable, str(TOOL), "--help"], capture_output=True, check=True)
        self.assertIn(b"create,verify,restore", result.stdout)
        self.assertFalse((self.root / ".sdsc/provenance").exists())


if __name__ == "__main__":
    unittest.main()
