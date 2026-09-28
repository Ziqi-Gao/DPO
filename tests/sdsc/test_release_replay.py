"""Real Git and local transport receivers exercise historical replay without SSH."""

import contextlib
import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_provenance as fixtures

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_replay", ROOT / "tools/sdsc_release_replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)
REMOTE_SPEC = importlib.util.spec_from_file_location("replay_receiver", ROOT / "tools/sdsc_remote.py")
remote = importlib.util.module_from_spec(REMOTE_SPEC)
REMOTE_SPEC.loader.exec_module(remote)


@pytest.fixture
def history(monkeypatch):
    fixture = fixtures.ProvenanceTests()
    fixture.setUp()
    root = fixture.root
    for name in replay.CONTROLS[1:]:
        fixture.write(name, (ROOT / name).read_bytes())
    fixture.refresh_wrapper()
    fixture.wrapper.update(project="OPD", source_root=str(root))
    fixture.wrapper_path.write_bytes(replay.json_bytes(fixture.wrapper))
    prepared = fixture.prepare()
    source = fixture.wrapper
    contents = {row["path"]: (root / row["path"]).read_bytes() for row in source["files"]}
    fixture.write(
        ".sdsc/runs/fixture-wrapper.json", replay.json_bytes({"state": "deployed", "manifest": source})
    )
    fixture.write(
        ".sdsc/check.json", replay.json_bytes({"connected": True, "control_python": sys.executable})
    )
    fixture.write("tools/sdsc_release_replay.py", (ROOT / "tools/sdsc_release_replay.py").read_bytes())
    fixture.write("docs/handoff.md", b"later handoff commit\n")
    fixture.command("add", "docs/handoff.md")
    fixture.command("commit", "--quiet", "-m", "Later handoff")
    current_head = fixture.command("rev-parse", "HEAD").decode().strip()
    target = Path(fixture.temp.name) / "remote"
    release = target / "releases/fixture-wrapper"
    release.mkdir(parents=True)
    (release / "manifest.json").write_bytes(replay.json_bytes(source))
    for row in source["files"]:
        path = release / "source" / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[row["path"]])
        path.chmod(row["mode"])
    monkeypatch.setattr(replay.cli, "REMOTE_ROOT", str(target))
    monkeypatch.setattr(remote, "ROOT", target)
    calls = []

    def upload(request, data, **kwargs):
        calls.append(request["action"])
        assert request["action"] == "upload"
        assert kwargs["control_python"] == sys.executable
        return {"ok": True, **remote.upload(dict(request, root=str(target)), io.BytesIO(data))}

    def ssh(argv, data=b"", timeout=60):
        assert argv[:4] == [sys.executable, "-I", "-B", "-c"]
        assert argv[4] in {replay.REMOTE_SOURCE, replay.uploader.REMOTE}
        is_source = argv[4] == replay.REMOTE_SOURCE
        calls.append("read_source" if is_source else "upload_provenance")
        source_code = argv[4].replace(
            'pathlib.Path("/home/zgao12/quest-runs/OPD")', f"pathlib.Path({str(target)!r})"
        )
        output = io.BytesIO() if is_source else io.StringIO()
        stdout = SimpleNamespace(buffer=output) if is_source else output
        with (
            patch.object(sys, "argv", ["receiver", argv[5]]),
            patch.object(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data))),
            patch("pwd.getpwuid", return_value=SimpleNamespace(pw_name="zgao12")),
            contextlib.redirect_stdout(stdout),
        ):
            exec(compile(source_code, "replay-test-receiver", "exec"), {"__name__": "__main__"})
        value = output.getvalue()
        return SimpleNamespace(returncode=0, stdout=value if is_source else value.encode())

    monkeypatch.setattr(replay.cli, "remote", upload)
    monkeypatch.setattr(replay.cli, "ssh_call", ssh)
    value = SimpleNamespace(
        fixture=fixture,
        root=root,
        source=source,
        contents=contents,
        prepared=prepared,
        current_head=current_head,
        target=target,
        calls=calls,
    )
    try:
        yield value
    finally:
        fixture.doCleanups()


def prepare(history, **kwargs):
    return replay.prepare(
        history.source["run_id"],
        history.prepared["artifact"],
        history.prepared["manifest_sha256"],
        repo=history.root,
        **kwargs,
    )


def state(plan):
    return replay.document(Path(plan["state_path"]))


def test_real_historical_replay_roundtrip_preserves_git_and_all_source_bytes(history):
    history.fixture.write("src/science.py", b"uncommitted new science must not enter replay\n")
    old_index = (history.root / ".git/index").read_bytes()
    old_refs = history.fixture.command("show-ref")
    original_manifest = replay.json_bytes(history.source)
    original_bundle = (Path(history.prepared["artifact"]) / "history.bundle").read_bytes()
    plan = prepare(history)
    assert history.calls == []
    assert state(plan)["phase"] == "prepared"
    assert plan["run_id"] != history.source["run_id"]
    assert plan["manifest"]["files"] == history.source["files"]
    assert plan["manifest"]["git_head"] == history.source["git_head"] != history.current_head
    assert plan["replay_origin"]["preparation_git_head"] == history.current_head
    assert plan["source_origins"]["src/science.py"] == "genuine_committed_blob"
    created = replay.document(Path(plan["provenance_created"]))
    assert set(created) == {"artifact", "manifest_sha256", "bundle_sha256", "git_head", "wrapper_code_sha256"}
    assert created["manifest_sha256"] != history.prepared["manifest_sha256"]
    assert (Path(created["artifact"]) / "history.bundle").read_bytes() == original_bundle
    restored = history.root / ".sdsc/restored-replay"
    replay.provenance.verify(
        created["artifact"], created["manifest_sha256"], created["wrapper_code_sha256"], restored
    )
    assert replay.provenance.git(restored, "rev-parse", "HEAD").decode().strip() == history.source["git_head"]
    for row in history.source["files"]:
        assert (Path(plan["stage"]) / "source" / row["path"]).read_bytes() == history.contents[row["path"]]
    report = replay.deploy(plan["plan_path"], repo=history.root)
    assert history.calls == ["upload", "upload_provenance"]
    assert state(plan)["phase"] == "deployed"
    uploaded = replay.document(Path(report["provenance_uploaded"]))
    assert uploaded["verified"] is True and uploaded["git_head"] == history.source["git_head"]
    assert uploaded["run_id"] == plan["run_id"]
    assert replay.document(Path(report["provenance_created"])) == created
    assert (history.root / ".git/index").read_bytes() == old_index
    assert history.fixture.command("show-ref") == old_refs
    assert history.fixture.command("rev-parse", "HEAD").decode().strip() == history.current_head
    assert (
        history.root / "src/science.py"
    ).read_bytes() == b"uncommitted new science must not enter replay\n"
    assert (history.target / "releases/fixture-wrapper/manifest.json").read_bytes() == original_manifest
    assert (Path(history.prepared["artifact"]) / "history.bundle").read_bytes() == original_bundle
    with pytest.raises(replay.ReplayError, match="reconcile"):
        replay.deploy(plan["plan_path"], repo=history.root)
    assert history.calls == ["upload", "upload_provenance"]


def test_missing_dirty_file_needs_explicit_read_and_verified_original_archive(history):
    history.fixture.write("docs/guide.md", b"later dirty docs cannot stand in for original\n")
    with pytest.raises(replay.ReplayError, match="explicit remote"):
        prepare(history)
    assert history.calls == []
    plan = prepare(history, allow_remote_source=True)
    assert history.calls == ["read_source"]
    assert plan["source_origins"]["docs/guide.md"] == "verified_remote_historical_file"
    assert (Path(plan["stage"]) / "source/docs/guide.md").read_bytes() == history.contents["docs/guide.md"]
    assert (history.root / "docs/guide.md").read_bytes() != history.contents["docs/guide.md"]


def test_current_symlink_is_never_followed_and_safe_committed_blob_is_used(history):
    path = history.root / "src/science.py"
    path.unlink()
    outside = Path(history.fixture.temp.name) / "outside.py"
    outside.write_bytes(b"must not be read or copied\n")
    path.symlink_to(outside)
    plan = prepare(history)
    assert plan["source_origins"]["src/science.py"] == "genuine_committed_blob"
    assert (Path(plan["stage"]) / "source/src/science.py").read_bytes() == history.contents["src/science.py"]
    assert path.is_symlink()


@pytest.mark.parametrize("mutation", ["file", "archive", "extra", "symlink", "controls", "plan"])
def test_changed_dry_run_or_controls_blocks_every_network_operation(history, mutation):
    plan = prepare(history)
    stage = Path(plan["stage"])
    if mutation == "controls":
        history.fixture.write("tools/sdsc_release_replay.py", b"changed control\n")
    elif mutation == "plan":
        path = Path(plan["plan_path"])
        path.chmod(0o600)
        value = replay.document(path)
        value["manifest"]["git_head"] = history.current_head
        path.write_bytes(replay.json_bytes(value))
    elif mutation == "extra":
        (stage / "source").chmod(0o700)
        (stage / "source/extra.py").write_bytes(b"extra\n")
    elif mutation == "symlink":
        path = stage / "source/src/science.py"
        path.parent.chmod(0o700)
        path.unlink()
        path.symlink_to(history.root / "src/science.py")
    else:
        path = stage / ("source.tar" if mutation == "archive" else "source/src/science.py")
        path.chmod(0o600)
        path.write_bytes(b"tampered\n")
    with pytest.raises((ValueError, OSError)):
        replay.deploy(plan["plan_path"], repo=history.root)
    assert history.calls == []


@pytest.mark.parametrize("phase", ["release", "provenance"])
def test_missing_upload_receipt_persists_unknown_and_cannot_retry(history, monkeypatch, phase):
    plan = prepare(history)
    calls = []

    def lost(*args, **kwargs):
        calls.append("lost")
        raise subprocess.TimeoutExpired("fixture transport", 1)

    monkeypatch.setattr(replay.cli, "remote" if phase == "release" else "ssh_call", lost)
    with pytest.raises(subprocess.TimeoutExpired):
        replay.deploy(plan["plan_path"], repo=history.root)
    assert state(plan)["phase"] == "unknown_" + phase
    assert state(plan)["reconciliation_required"] is True
    with pytest.raises(replay.ReplayError, match="reconcile"):
        replay.deploy(plan["plan_path"], repo=history.root)
    assert calls == ["lost"]


def test_existing_release_is_not_overwritten_and_original_run_remains_intact(history):
    plan = prepare(history)
    existing = history.target / "releases" / plan["run_id"]
    existing.mkdir()
    marker = existing / "owner.txt"
    marker.write_bytes(b"preserve existing release\n")
    with pytest.raises(ValueError, match="Release exists"):
        replay.deploy(plan["plan_path"], repo=history.root)
    assert marker.read_bytes() == b"preserve existing release\n"
    assert state(plan)["phase"] == "unknown_release"


def test_persistent_deploy_claim_blocks_competing_deployer(history):
    plan = prepare(history)
    (Path(plan["plan_path"]).parent / "deploy.claim").mkdir()
    with pytest.raises(FileExistsError):
        replay.deploy(plan["plan_path"], repo=history.root)
    assert history.calls == []


@pytest.mark.parametrize("ok", [None, False])
def test_release_receipt_must_explicitly_confirm_success(history, monkeypatch, ok):
    plan = prepare(history)
    original = replay.cli.remote

    def rejected(*args, **kwargs):
        response = original(*args, **kwargs)
        response["ok"] = ok
        return response

    monkeypatch.setattr(replay.cli, "remote", rejected)
    with pytest.raises(replay.ReplayError, match="receipt differs"):
        replay.deploy(plan["plan_path"], repo=history.root)
    assert state(plan)["phase"] == "unknown_release"
    assert history.calls == ["upload"]


def test_untrusted_source_provenance_or_wrong_run_record_is_rejected(history):
    with pytest.raises(ValueError, match="manifest hash"):
        replay.prepare(history.source["run_id"], history.prepared["artifact"], "0" * 64, repo=history.root)
    record = history.root / ".sdsc/runs/fixture-wrapper.json"
    value = replay.document(record)
    value["manifest"]["git_head"] = history.current_head
    record.write_bytes(replay.json_bytes(value))
    with pytest.raises(ValueError, match="trusted provenance wrapper"):
        prepare(history)
    assert history.calls == []


@pytest.mark.parametrize("mutation", ["hash", "manifest", "symlink"])
def test_remote_original_source_corruption_rejected(history, mutation):
    history.fixture.write("docs/guide.md", b"new local docs\n")
    release = history.target / "releases/fixture-wrapper"
    path = release / "source/docs/guide.md"
    if mutation == "hash":
        path.write_bytes(b"wrong remote historical bytes\n")
    elif mutation == "manifest":
        (release / "manifest.json").write_bytes(b"{}\n")
    else:
        path.unlink()
        path.symlink_to(history.root / "docs/guide.md")
    with pytest.raises(AssertionError):
        prepare(history, allow_remote_source=True)


@pytest.mark.parametrize("kind", ["extra", "duplicate", "symlink"])
def test_source_archive_rejects_extra_duplicate_and_nonregular_members(history, kind):
    manifest = history.source
    record = next(row for row in manifest["files"] if row["path"] == "docs/guide.md")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in [
            ("manifest.json", replay.json_bytes(manifest)),
            ("source/docs/guide.md", history.contents[record["path"]]),
        ]:
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            if kind == "symlink" and name.startswith("source/"):
                item.type, item.linkname, item.size = tarfile.SYMTYPE, "/tmp/outside", 0
            archive.addfile(item, io.BytesIO(data))
        if kind != "symlink":
            item = tarfile.TarInfo("../outside" if kind == "extra" else "manifest.json")
            item.size = 0
            archive.addfile(item, io.BytesIO())
    with pytest.raises(replay.ReplayError, match="inventory"):
        replay.decode_source_archive(buffer.getvalue(), manifest, [record])
