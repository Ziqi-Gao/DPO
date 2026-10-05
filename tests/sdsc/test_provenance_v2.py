"""Real full-bundle Git boundaries for the additive 256-commit format."""

from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


provenance = load("_provenance_v2_test", ROOT / "tools/sdsc_provenance_v2.py")
legacy = load("_provenance_v1_test", ROOT / "tools/sdsc_provenance.py")
legacy_fixtures = load("_provenance_v1_fixtures", Path(__file__).with_name("test_provenance.py"))


@pytest.fixture
def repository():
    fixture = legacy_fixtures.ProvenanceTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def prepare(fixture, successor=None):
    options = dict(accepted_ancestors=[fixture.ancestor])
    if successor is not None:
        options.update(
            reviewed_local_implementation=successor["implementation"],
            reviewed_local_acceptance=successor["acceptance"],
        )
    return provenance.prepare(fixture.root, fixture.wrapper_path, **options)


def verify(prepared, destination=None):
    return provenance.verify(
        prepared["artifact"], prepared["manifest_sha256"], prepared["wrapper_code_sha256"], destination
    )


def rewrite_manifest(fixture, prepared, change):
    value = json.loads((Path(prepared["artifact"]) / "manifest.json").read_bytes())
    change(value)
    raw = provenance.canonical(value) + b"\n"
    fixture.tamper(prepared["artifact"], "manifest.json", raw)
    prepared["manifest_sha256"] = provenance.sha(raw)


@pytest.mark.parametrize("count", [129, 256])
def test_actual_extended_history_roundtrip_preserves_every_commit_and_rejects_v1(repository, count):
    fixture = repository
    preceding = fixture.local_prelude(count - 2)
    successor = fixture.local_pair()
    assert len(successor["commits"]) == count
    before_index = (fixture.root / ".git/index").read_bytes()
    before_status = fixture.command("status", "--porcelain=v1")
    with pytest.raises(legacy.ProvenanceError, match="commit limit"):
        legacy.prepare(
            fixture.root,
            fixture.wrapper_path,
            reviewed_local_implementation=successor["implementation"],
            reviewed_local_acceptance=successor["acceptance"],
        )
    with pytest.raises(legacy.ProvenanceError, match="bounded unique"):
        legacy.validate_local_successor(successor, fixture.head, successor["public_base"])
    prepared = prepare(fixture, successor)
    artifact = Path(prepared["artifact"])
    assert artifact.parent == fixture.root / ".sdsc/provenance-v2"
    assert not (fixture.root / ".sdsc/provenance").exists()
    assert set(p.name for p in artifact.iterdir()) == {
        "manifest.json",
        "wrapper-manifest.json",
        "history.bundle",
    }
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    assert manifest["schema"] == "quest-sdsc-git-provenance-v2"
    assert manifest["accepted_ancestors"] == [fixture.ancestor]
    assert (
        (artifact / "history.bundle")
        .read_bytes()
        .startswith(b"# v2 git bundle\n" + fixture.head.encode() + b" HEAD\n\nPACK")
    )
    destination = Path(fixture.temp.name) / "restored-v2"
    observed = verify(prepared, destination)
    assert observed["local_successor"] == successor
    assert provenance.git(destination, "rev-parse", "HEAD").decode().strip() == fixture.head
    assert (
        provenance.git(destination, "rev-list", "--reverse", successor["public_base"] + "..HEAD")
        .decode()
        .splitlines()
        == successor["commits"]
    )
    assert provenance.git(destination, "status", "--porcelain") == b""
    for index in (0, len(preceding) // 2, len(preceding) - 1):
        assert (
            provenance.git(destination, "show", preceding[index] + ":docs/local-history.md")
            == f"Retained local state {index}\n".encode()
        )
    assert provenance.git(destination, "merge-base", "--is-ancestor", fixture.ancestor, "HEAD") == b""
    assert (destination / "src/science.py").read_bytes() == b"VALUE = 2\n"
    assert (destination / "docs/guide.md").read_bytes() == b"committed docs\n"
    assert not (destination / "tools/wrapper.py").exists()
    assert not destination.stat().st_mode & stat.S_IWUSR
    assert not (destination / ".git").stat().st_mode & stat.S_IWUSR
    assert before_index == (fixture.root / ".git/index").read_bytes()
    assert before_status == fixture.command("status", "--porcelain=v1")
    assert fixture.command("rev-parse", "HEAD").decode().strip() == fixture.head
    assert (
        fixture.command("rev-parse", "refs/remotes/public/master").decode().strip()
        == successor["public_base"]
    )
    with pytest.raises(legacy.ProvenanceError, match="schema"):
        legacy.verify(artifact, prepared["manifest_sha256"], prepared["wrapper_code_sha256"])


def test_real_257_commits_rejected_before_export_and_import(repository, monkeypatch):
    fixture = repository
    # A small valid artifact lets the consumer reject a real oversized history
    # declaration before any import, while the source actually contains 257 commits.
    prepared = prepare(fixture)
    fixture.local_prelude(255)
    successor = fixture.local_pair()
    assert len(successor["commits"]) == 257
    before = set((fixture.root / ".sdsc/provenance-v2").iterdir())
    with pytest.raises(provenance.ProvenanceError, match="commit limit"):
        prepare(fixture, successor)
    assert set((fixture.root / ".sdsc/provenance-v2").iterdir()) == before
    rewrite_manifest(
        fixture, prepared, lambda value: value.update(git_head=fixture.head, local_successor=successor)
    )
    monkeypatch.setattr(provenance, "init_repo", lambda *_: pytest.fail("oversized history imported"))
    with pytest.raises(provenance.ProvenanceError, match="bounded unique"):
        verify(prepared)


def test_v2_rejects_v1_and_keeps_all_byte_limits_and_legacy_globals(repository):
    old = repository.prepare()
    with pytest.raises(provenance.ProvenanceError, match="schema"):
        verify(old)
    assert legacy.MAX_LOCAL_COMMITS == 128 and legacy.SCHEMA == "quest-sdsc-git-provenance-v1"
    assert provenance.MAX_LOCAL_COMMITS == 256
    assert provenance.MAX_TOTAL == legacy.MAX_TOTAL == 64 * 1024**2
    assert provenance.MAX_FILE == legacy.MAX_FILE == 4 * 1024**2
    assert provenance.SCIENCE_ROOTS == legacy.SCIENCE_ROOTS
    assert provenance.SCIENCE_FILES == legacy.SCIENCE_FILES


@pytest.mark.parametrize(
    "corruption", ["schema", "manifest_hash", "wrapper_hash", "bundle_hash", "extra_ref", "prerequisite"]
)
def test_rejection_precedes_git_import(repository, monkeypatch, corruption):
    fixture = repository
    prepared = prepare(fixture)
    artifact = Path(prepared["artifact"])
    if corruption == "schema":
        rewrite_manifest(fixture, prepared, lambda v: v.update(schema=legacy.SCHEMA))
    elif corruption == "manifest_hash":
        prepared["manifest_sha256"] = "0" * 64
    elif corruption == "wrapper_hash":
        prepared["wrapper_code_sha256"] = "0" * 64
    else:
        bundle = (artifact / "history.bundle").read_bytes()
        if corruption == "bundle_hash":
            fixture.tamper(artifact, "history.bundle", bundle + b"changed")
        else:
            extra = (
                fixture.head.encode() + b" refs/heads/extra\n"
                if corruption == "extra_ref"
                else b"-" + fixture.ancestor.encode() + b" prerequisite\n"
            )
            bundle = bundle.replace(b"\n\nPACK", b"\n" + extra + b"\nPACK", 1)
            fixture.tamper(artifact, "history.bundle", bundle)
            rewrite_manifest(
                fixture,
                prepared,
                lambda v: v["bundle"].update(size=len(bundle), sha256=provenance.sha(bundle)),
            )
    monkeypatch.setattr(provenance, "init_repo", lambda *_: pytest.fail("invalid artifact imported"))
    with pytest.raises(provenance.ProvenanceError):
        verify(prepared)


@pytest.mark.parametrize(
    "corruption",
    ["missing_commit", "reordered_commit", "forged_commit", "public_base", "ancestor", "tree", "science"],
)
def test_rehashed_false_lineage_and_tree_are_checked_against_actual_bundle(repository, corruption):
    fixture = repository
    fixture.local_prelude(2)
    successor = fixture.local_pair()
    prepared = prepare(fixture, successor)

    def mutate(value):
        if corruption == "missing_commit":
            value["local_successor"]["commits"].pop(0)
        elif corruption == "reordered_commit":
            value["local_successor"]["commits"][:2] = reversed(value["local_successor"]["commits"][:2])
        elif corruption == "forged_commit":
            value["local_successor"]["commits"][0] = "0" * 40
        elif corruption == "public_base":
            value["public_tracking_tip"] = value["local_successor"]["public_base"] = fixture.ancestor
        elif corruption == "ancestor":
            value["accepted_ancestors"] = ["0" * 40]
        elif corruption == "tree":
            value["committed_tree_sha256"] = "0" * 64
        else:
            value["scientific_files"][0]["sha256"] = "0" * 64

    rewrite_manifest(fixture, prepared, mutate)
    with pytest.raises(provenance.ProvenanceError):
        verify(prepared)


@pytest.mark.parametrize(
    "corruption",
    [
        "post_acceptance",
        "merge",
        "unrelated_public",
        "public_inside_pair",
        "missing_anchor",
        "dirty_science",
        "hidden_science",
    ],
)
def test_producer_keeps_review_ancestry_linearity_and_exact_science_contract(repository, corruption):
    fixture = repository
    if corruption == "merge":
        before = fixture.local_prelude(1)[0]
        tree = fixture.command("rev-parse", "HEAD^{tree}").decode().strip()
        merged = (
            fixture.command("commit-tree", tree, "-p", before, "-p", fixture.head, "-m", "merge fixture")
            .decode()
            .strip()
        )
        fixture.command("update-ref", "HEAD", merged)
    successor = fixture.local_pair()
    if corruption == "post_acceptance":
        fixture.local_prelude(1)
        fixture.head = fixture.command("rev-parse", "HEAD").decode().strip()
        fixture.refresh_wrapper()
    elif corruption == "unrelated_public":
        tree = fixture.command("rev-parse", "HEAD^{tree}").decode().strip()
        other = fixture.command("commit-tree", tree, "-m", "unrelated").decode().strip()
        fixture.command("update-ref", "refs/remotes/public/master", other)
    elif corruption == "public_inside_pair":
        fixture.command("update-ref", "refs/remotes/public/master", successor["implementation"])
    elif corruption == "missing_anchor":
        successor["implementation"] = None
    elif corruption == "dirty_science":
        fixture.write("src/science.py", b"CHANGED = True\n")
        fixture.refresh_wrapper()
    elif corruption == "hidden_science":
        fixture.write("src/hidden.ignored.py", b"CHANGED = True\n")
    with pytest.raises(provenance.ProvenanceError):
        prepare(fixture, successor)


def test_unsafe_deleted_tree_beyond_old_capacity_is_rejected_at_both_boundaries(repository, monkeypatch):
    fixture = repository
    fixture.local_prelude(128)
    fixture.write("secrets/fixture.txt", b"synthetic rejection fixture\n")
    fixture.command("add", "secrets/fixture.txt")
    fixture.command("commit", "--quiet", "-m", "unsafe historical fixture")
    fixture.command("rm", "--quiet", "secrets/fixture.txt")
    fixture.command("commit", "--quiet", "-m", "delete historical fixture")
    successor = fixture.local_pair()
    assert len(successor["commits"]) == 132
    with pytest.raises(provenance.ProvenanceError, match="credential path"):
        prepare(fixture, successor)
    with monkeypatch.context() as patched:
        patched.setattr(provenance, "audit_local_successor", lambda *_: None)
        prepared = prepare(fixture, successor)
    with pytest.raises(provenance.ProvenanceError, match="credential path"):
        verify(prepared)


@pytest.mark.parametrize("where", ["artifact", "destination", "existing_destination"])
def test_restore_cannot_follow_links_or_replace_existing_checkout(repository, where):
    fixture = repository
    prepared = prepare(fixture)
    outside = Path(fixture.temp.name) / "outside"
    outside.mkdir()
    if where == "artifact":
        linked = Path(fixture.temp.name) / "linked-artifact"
        linked.symlink_to(prepared["artifact"], target_is_directory=True)
        prepared = dict(prepared, artifact=str(linked))
        destination = None
    elif where == "destination":
        linked = Path(fixture.temp.name) / "linked-parent"
        linked.symlink_to(outside, target_is_directory=True)
        destination = linked / "new"
    else:
        destination = outside
    with pytest.raises(provenance.ProvenanceError):
        verify(prepared, destination)
    assert not list(outside.iterdir())
