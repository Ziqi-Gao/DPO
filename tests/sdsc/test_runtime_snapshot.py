"""Real streaming producer/consumer and fail-closed runtime archive boundaries."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_runtime_snapshot_tests", ROOT / "tools/sdsc_runtime_snapshot.py"
)
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source, artifacts, local = [tmp_path / name for name in ("source", "artifacts", "local")]
    for path in (source, artifacts, local):
        path.mkdir()
    prefix = source / "runtime"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib/python3.12/site-packages/demo").mkdir(parents=True)
    (prefix / "bin/python3.12").write_bytes(b"native executable fixture\x00\xff")
    (prefix / "bin/python3.12").chmod(0o755)
    (prefix / "bin/python").symlink_to("python3.12")
    (prefix / "lib/python3.12/site-packages/demo/__init__.py").write_bytes(b"VALUE = 42\n")
    (prefix / "lib/python3.12/site-packages/demo/empty").write_bytes(b"")
    (prefix / "lib/python3.12/site-packages/demo/data").write_bytes(bytes(range(256)) * 4097)
    (prefix / "lib64").symlink_to("lib")
    (prefix / "bin/data").symlink_to("../lib/python3.12/site-packages/demo/data")
    monkeypatch.setattr(
        snapshot,
        "mount_info",
        lambda p: dict(target=str(local), fstype="ext4", source="/dev/test", options="rw,noatime"),
    )
    return dict(
        source=source,
        artifacts=artifacts,
        local=local,
        prefix=prefix,
        artifact=artifacts / "snapshot",
        destination=local / "runtime",
    )


def prepare(f):
    return snapshot.prepare(f["prefix"], f["artifact"], source_root=f["source"], artifact_root=f["artifacts"])


def stage(f, descriptor, *, rehearsal=False):
    method = snapshot.rehearse if rehearsal else snapshot.stage
    return method(
        f["artifact"],
        descriptor["manifest"]["sha256"],
        f["destination"],
        artifact_root=f["artifacts"],
        destination_root=f["local"],
    )


def manifest(f):
    return json.loads((f["artifact"] / "manifest.json").read_bytes())


def mutate_manifest(f, descriptor, change):
    value = manifest(f)
    change(value)
    raw = snapshot.canonical(value)
    (f["artifact"] / "manifest.json").write_bytes(raw)
    descriptor["manifest"]["sha256"] = snapshot.sha(raw)
    descriptor["manifest"]["size"] = len(raw)


def test_real_full_snapshot_roundtrip_is_deterministic_and_preserves_source(fixture):
    f = fixture
    initial, source_stamps, source_mode = snapshot.scan(f["prefix"])
    first = prepare(f)
    second = snapshot.prepare(
        f["prefix"], f["artifacts"] / "again", source_root=f["source"], artifact_root=f["artifacts"]
    )
    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert first["manifest"]["sha256"] == second["manifest"]["sha256"]
    assert first["archive"]["size"] == len(snapshot.MAGIC) + first["total_bytes"]
    assert snapshot.scan(f["prefix"]) == (initial, source_stamps, source_mode)
    value = snapshot.load_manifest(f["artifact"], first["manifest"]["sha256"], artifact_root=f["artifacts"])
    assert snapshot.verify_archive(f["artifact"], value, artifact_root=f["artifacts"]) == first["archive"]
    result = stage(f, first)
    assert result["schema"] == snapshot.STAGE_SCHEMA
    assert result["node_local_verified"] is True
    assert result["files_verified"] is result["archive_verified"] is True
    assert result["descriptor"] == first
    assert Path(result["python"]).read_bytes() == (f["prefix"] / "bin/python3.12").read_bytes()
    assert stat.S_IMODE(Path(result["python"]).stat().st_mode) == 0o755
    assert os.readlink(f["destination"] / "bin/data") == "../lib/python3.12/site-packages/demo/data"
    assert not (f["destination"] / ".snapshot-incomplete").exists()
    assert snapshot.scan(f["prefix"]) == (initial, source_stamps, source_mode)
    assert snapshot.scan(f["destination"])[0] == initial
    assert not set(result) & {"runtime_relocatable", "cuda_ready", "student_accepted"}


def test_actual_python_venv_transport_and_native_executable_relocation(tmp_path):
    # This proves real executable/venv transport, not closure of the production
    # Conda environment: this fixture intentionally retains its external base.
    source, artifacts, local = [tmp_path / name for name in ("source", "artifacts", "local")]
    for path in (source, artifacts, local):
        path.mkdir()
    prefix = source / "venv"
    subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--copies", "--without-pip", str(prefix)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    python_name = "bin/python" + str(sys.version_info.major) + "." + str(sys.version_info.minor)
    # CPython's venv uses a regular copied versioned executable with --copies.
    assert not (prefix / python_name).is_symlink()
    descriptor = snapshot.prepare(
        prefix,
        artifacts / "one",
        source_root=source,
        artifact_root=artifacts,
        python_relative_path=python_name,
    )
    result = snapshot.rehearse(
        artifacts / "one",
        descriptor["manifest"]["sha256"],
        local / "moved",
        artifact_root=artifacts,
        destination_root=local,
    )
    assert result["schema"] == snapshot.REHEARSAL_SCHEMA and result["node_local_verified"] is False
    command = [
        result["python"],
        "-I",
        "-B",
        "-c",
        "import json,sys;print(json.dumps([sys.executable,sys.prefix,sys.base_prefix]))",
    ]
    observed = json.loads(
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=20).stdout
    )
    assert observed[:2] == [result["python"], result["destination"]]
    assert observed[2] != result["destination"]  # Closure proof belongs to the caller.
    assert (local / "moved/pyvenv.cfg").read_bytes() == (prefix / "pyvenv.cfg").read_bytes()


@pytest.mark.parametrize(
    "kind", ["absolute_internal", "external", "escape", "dangling", "cycle", "file_parent"]
)
def test_unsafe_symlink_source_fails_without_completion(fixture, kind):
    f = fixture
    targets = dict(
        absolute_internal=str(f["prefix"] / "bin/python3.12"),
        external="/etc/passwd",
        escape="../outside",
        dangling="missing",
        cycle="bad",
        file_parent="bin/python3.12/child",
    )
    (f["prefix"] / "bad").symlink_to(targets[kind])
    with pytest.raises(ValueError):
        prepare(f)
    assert not (f["artifact"] / "manifest.json").exists()


@pytest.mark.parametrize(
    "kind",
    ["fifo", "special_mode", "unowned", "source_ancestor_link", "artifact_ancestor_link", "escape_root"],
)
def test_special_files_and_owned_root_boundaries(fixture, monkeypatch, kind):
    f = fixture
    if kind == "fifo":
        os.mkfifo(f["prefix"] / "fifo")
    elif kind == "special_mode":
        (f["prefix"] / "bin/python3.12").chmod(0o4755)
    elif kind == "unowned":
        monkeypatch.setattr(snapshot.os, "getuid", lambda: -1)
    elif kind.endswith("ancestor_link"):
        link = f["source"].parent / "alias"
        target = f["source"] if kind == "source_ancestor_link" else f["artifacts"]
        link.symlink_to(target, target_is_directory=True)
        if kind == "source_ancestor_link":
            f["source"], f["prefix"] = link, link / "runtime"
        else:
            f["artifacts"], f["artifact"] = link, link / "snapshot"
    else:
        f["source"] = f["artifacts"]
    with pytest.raises(ValueError):
        prepare(f)


@pytest.mark.parametrize("limit", ["MAX_ENTRIES", "MAX_FILE_BYTES", "MAX_TOTAL_BYTES", "MAX_MANIFEST_BYTES"])
def test_fixed_size_and_count_bounds(fixture, monkeypatch, limit):
    monkeypatch.setattr(snapshot, limit, 1)
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not (fixture["artifact"] / "manifest.json").exists()


def test_source_mutation_before_open_rejected(fixture, monkeypatch):
    f = fixture
    original = snapshot.open_relative

    def replace(fd, name, flags):
        path = f["prefix"] / name
        path.write_bytes(path.read_bytes() + b"changed")
        return original(fd, name, flags)

    monkeypatch.setattr(snapshot, "open_relative", replace)
    with pytest.raises(ValueError, match="source file changed"):
        prepare(f)
    assert not (f["artifact"] / "manifest.json").exists()


@pytest.mark.parametrize("change", ["new_file", "removed_file", "changed_symlink", "replaced_root"])
def test_post_pack_inventory_detects_changes(fixture, monkeypatch, change):
    f = fixture
    original, count = snapshot.scan, 0

    def changing(prefix, **kw):
        nonlocal count
        count += 1
        if count == 2:
            if change == "new_file":
                (prefix / "new").write_bytes(b"added")
            elif change == "removed_file":
                (prefix / "lib/python3.12/site-packages/demo/empty").unlink()
            elif change == "changed_symlink":
                (prefix / "bin/python").unlink()
                (prefix / "bin/python").symlink_to("./python3.12")
            else:
                replacement = prefix.with_name("replacement")
                shutil.copytree(prefix, replacement, symlinks=True)
                prefix.rename(prefix.with_name("old"))
                replacement.rename(prefix)
        return original(prefix, **kw)

    monkeypatch.setattr(snapshot, "scan", changing)
    with pytest.raises(ValueError, match="inventory changed"):
        prepare(f)
    assert not (f["artifact"] / "manifest.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "entry_count",
        "files_count",
        "total_bytes",
        "runtime_root_sha256",
        "python_sha256",
        "python_relative_path",
        "schema",
        "extra",
    ],
)
def test_rehashed_manifest_semantic_corruption_rejected(fixture, field):
    f = fixture
    descriptor = prepare(f)

    def change(value):
        value[field] = True if field in ("entry_count", "files_count", "total_bytes") else "invalid"

    mutate_manifest(f, descriptor, change)
    with pytest.raises(ValueError):
        stage(f, descriptor)
    assert not f["destination"].exists()


@pytest.mark.parametrize(
    "change", ["traversal", "duplicate", "link_parent", "absolute_symlink", "invalid_hash"]
)
def test_rehashed_manifest_inventory_rejection(fixture, change):
    f = fixture
    descriptor = prepare(f)

    def mutate(value):
        rows = value["entries"]
        if change == "traversal":
            rows[0]["path"] = "../outside"
        elif change == "duplicate":
            rows[1] = dict(rows[0])
        elif change == "link_parent":
            rows[0] = dict(path="bin", type="symlink", mode=0o777, target="lib")
        elif change == "absolute_symlink":
            next(r for r in rows if r["type"] == "symlink")["target"] = "/etc/passwd"
        else:
            next(r for r in rows if r["type"] == "file")["sha256"] = "z" * 64
        value["runtime_root_sha256"] = snapshot.sha(
            snapshot.canonical(dict(root_mode=value["root_mode"], entries=rows))
        )

    mutate_manifest(f, descriptor, mutate)
    with pytest.raises(ValueError):
        stage(f, descriptor)
    assert not f["destination"].exists()


@pytest.mark.parametrize("change", ["hash", "short", "long", "magic", "file_bytes"])
def test_archive_corruption_never_returns_verified_stage(fixture, change):
    f = fixture
    descriptor = prepare(f)
    archive = f["artifact"] / "runtime.bin"
    original = archive.read_bytes()
    if change == "short":
        changed = original[:-1]
    elif change == "long":
        changed = original + b"x"
    elif change == "magic":
        changed = b"X" + original[1:]
    else:
        index = len(snapshot.MAGIC)
        changed = original[:index] + bytes([original[index] ^ 1]) + original[index + 1 :]
    archive.write_bytes(changed)
    if change in ("magic", "file_bytes"):
        mutate_manifest(f, descriptor, lambda v: v["archive"].update(sha256=snapshot.sha(changed)))
    with pytest.raises(ValueError):
        stage(f, descriptor)
    assert not (f["destination"] / "bin/python").exists()


def test_source_archive_symlink_and_nonlocal_stage_fail_closed(fixture, monkeypatch):
    f = fixture
    descriptor = prepare(f)
    monkeypatch.setattr(
        snapshot, "mount_info", lambda _: dict(target="/", fstype="lustre", source="shared", options="rw")
    )
    with pytest.raises(ValueError, match="not verified node-local"):
        stage(f, descriptor)
    assert not f["destination"].exists()
    result = stage(f, descriptor, rehearsal=True)
    assert result["node_local_verified"] is False and result["schema"] == snapshot.REHEARSAL_SCHEMA
    f["destination"] = f["local"] / "again"
    archive = f["artifact"] / "runtime.bin"
    moved = archive.with_name("moved")
    archive.rename(moved)
    archive.symlink_to(moved.name)
    with pytest.raises(ValueError, match="symlink"):
        stage(f, descriptor, rehearsal=True)


def test_existing_partial_paths_and_expired_deadline_are_never_reused(fixture):
    f = fixture
    descriptor = prepare(f)
    with pytest.raises(ValueError, match="fresh"):
        prepare(f)
    with pytest.raises(ValueError, match="deadline"):
        snapshot.stage(
            f["artifact"],
            descriptor["manifest"]["sha256"],
            f["destination"],
            artifact_root=f["artifacts"],
            destination_root=f["local"],
            deadline=time.monotonic() - 1,
        )
    assert f["destination"].with_name(f["destination"].name + ".snapshot-incomplete").is_dir()
    with pytest.raises(ValueError, match="fresh"):
        stage(f, descriptor)


def test_descriptor_rejects_unbounded_or_foreign_shapes(fixture):
    descriptor = prepare(fixture)
    for mutate in (
        lambda d: d.update(files_count=True),
        lambda d: d.update(total_bytes=snapshot.MAX_TOTAL_BYTES + 1),
        lambda d: d["archive"].update(path="/other/runtime.bin"),
        lambda d: d.update(extra=True),
        lambda d: d["manifest"].update(size=snapshot.MAX_MANIFEST_BYTES + 1),
    ):
        changed = json.loads(json.dumps(descriptor))
        mutate(changed)
        with pytest.raises(ValueError):
            snapshot.validate_descriptor(changed)


def test_cli_real_prepare_verify_rehearse_and_no_execution_claims(fixture, capsys):
    f = fixture
    common = ["--artifact-dir", str(f["artifact"]), "--artifact-root", str(f["artifacts"])]
    snapshot.main(["prepare", *common, "--prefix", str(f["prefix"]), "--source-root", str(f["source"])])
    descriptor = json.loads(capsys.readouterr().out)
    snapshot.main(["verify", *common, "--manifest-sha256", descriptor["manifest"]["sha256"]])
    assert json.loads(capsys.readouterr().out) == descriptor
    snapshot.main(
        [
            "rehearse",
            *common,
            "--manifest-sha256",
            descriptor["manifest"]["sha256"],
            "--destination",
            str(f["destination"]),
            "--destination-root",
            str(f["local"]),
        ]
    )
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["schema"] == snapshot.REHEARSAL_SCHEMA and not evidence["node_local_verified"]


def test_chunked_io_handles_boundary_files_without_large_reads(fixture, monkeypatch):
    monkeypatch.setattr(snapshot, "CHUNK", 113)
    descriptor = prepare(fixture)
    assert stage(fixture, descriptor)["total_bytes"] == descriptor["total_bytes"]


@pytest.mark.parametrize("phase", ["prepare", "stage"])
def test_free_space_checked_before_creating_partial_artifact(fixture, monkeypatch, phase):
    f = fixture
    descriptor = prepare(f) if phase == "stage" else None
    monkeypatch.setattr(snapshot.os, "statvfs", lambda _: SimpleNamespace(f_bavail=1, f_frsize=4096))
    with pytest.raises(ValueError, match="free space"):
        prepare(f) if phase == "prepare" else stage(f, descriptor)
    assert not (f["artifact"] if phase == "prepare" else f["destination"]).exists()


def test_source_file_changed_during_streaming_is_rejected(fixture, monkeypatch):
    f = fixture
    called = False

    def change(_):
        nonlocal called
        if f["artifact"].exists() and not called:
            called = True
            path = f["prefix"] / "bin/python3.12"
            raw = path.read_bytes()
            path.write_bytes(b"X" + raw[1:])

    monkeypatch.setattr(snapshot, "check_deadline", change)
    with pytest.raises(ValueError, match="source file changed while"):
        prepare(f)
    assert called and not (f["artifact"] / "manifest.json").exists()


def test_source_archive_changed_during_local_copy_is_rejected(fixture, monkeypatch):
    f = fixture
    descriptor = prepare(f)
    called = False

    def change(_):
        nonlocal called
        local_copy = (
            f["destination"].with_name(f["destination"].name + ".snapshot-incomplete") / "runtime.bin"
        )
        if local_copy.exists() and not called:
            called = True
            path = f["artifact"] / "runtime.bin"
            with path.open("r+b") as stream:
                stream.write(b"X")

    monkeypatch.setattr(snapshot, "check_deadline", change)
    with pytest.raises(ValueError, match="archive changed during copy"):
        stage(f, descriptor)
    assert called
    assert not (f["destination"] / "bin").exists()


def test_actual_mount_parser_prefers_most_specific_and_decodes_paths(monkeypatch, tmp_path):
    raw = (
        "1 0 1:1 / / rw - overlay overlay rw\n"
        "2 1 8:1 / /scratch rw - ext4 /dev/test rw\n"
        "3 2 8:2 / /scratch/space\\040here rw - xfs /dev/other rw\n"
    )
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **k: raw if str(p) == "/proc/self/mountinfo" else original(p, *a, **k),
    )
    assert snapshot.mount_info(Path("/scratch/space here/runtime")) == dict(
        target="/scratch/space here", fstype="xfs", source="/dev/other", options="rw"
    )
    assert snapshot.mount_info(tmp_path)["fstype"] == "overlay"


def test_manifest_duplicate_keys_noncanonical_and_wrong_trust_hash_rejected(fixture):
    f = fixture
    descriptor = prepare(f)
    with pytest.raises(ValueError, match="SHA mismatch"):
        snapshot.load_manifest(f["artifact"], "0" * 64, artifact_root=f["artifacts"])
    path = f["artifact"] / "manifest.json"
    original = path.read_bytes()
    variants = [
        original.replace(b'{"archive":', b'{"extra":0,"extra":1,"archive":', 1),
        json.dumps(manifest(f), indent=2).encode(),
    ]
    for raw in variants:
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            snapshot.load_manifest(f["artifact"], snapshot.sha(raw), artifact_root=f["artifacts"])
    path.write_bytes(original)
    assert (
        snapshot.describe(f["artifact"], descriptor["manifest"]["sha256"], artifact_root=f["artifacts"])
        == descriptor
    )


def test_real_inherited_sgid_directories_preserved_without_file_privilege_bits(fixture):
    f = fixture
    for path in [f["prefix"], f["prefix"] / "bin", f["prefix"] / "lib", f["prefix"] / "lib/python3.12"]:
        path.chmod(0o2755)
    f["local"].chmod(0o2700)
    before = snapshot.scan(f["prefix"])
    descriptor = prepare(f)
    value = manifest(f)
    assert value["root_mode"] == 0o2755
    assert next(r for r in value["entries"] if r["path"] == "bin")["mode"] == 0o2755
    result = stage(f, descriptor)
    assert snapshot.scan(f["prefix"]) == before
    assert snapshot.scan(f["destination"])[0] == before[0]
    assert stat.S_IMODE(f["destination"].stat().st_mode) == 0o2755
    assert stat.S_IMODE(Path(result["python"]).stat().st_mode) == 0o755
    assert not f["destination"].with_name(f["destination"].name + ".snapshot-incomplete").exists()


@pytest.mark.parametrize(
    "target,bits",
    [
        ("root", 0o1000),
        ("root", 0o4000),
        ("directory", 0o1000),
        ("directory", 0o4000),
        ("file", 0o2000),
        ("file", 0o4000),
    ],
)
def test_other_special_bits_still_rejected_at_source(fixture, target, bits):
    f = fixture
    path = (
        f["prefix"]
        if target == "root"
        else f["prefix"] / ("bin" if target == "directory" else "bin/python3.12")
    )
    path.chmod(0o755 | bits)
    with pytest.raises(ValueError, match="special permission"):
        prepare(f)
    assert not (f["artifact"] / "manifest.json").exists()


@pytest.mark.parametrize("target,bits", [("root", 0o1000), ("directory", 0o4000), ("file", 0o2000)])
def test_rehashed_manifest_cannot_introduce_unsafe_special_mode(fixture, target, bits):
    f = fixture
    descriptor = prepare(f)

    def change(value):
        if target == "root":
            value["root_mode"] |= bits
        else:
            next(r for r in value["entries"] if r["type"] == target)["mode"] |= bits
        value["runtime_root_sha256"] = snapshot.sha(
            snapshot.canonical(dict(root_mode=value["root_mode"], entries=value["entries"]))
        )

    mutate_manifest(f, descriptor, change)
    with pytest.raises(ValueError, match="mode"):
        stage(f, descriptor)
    assert not f["destination"].exists()
