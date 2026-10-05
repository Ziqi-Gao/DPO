"""Bounded metadata overlap, exact archive bytes, and directory-FD lifetime."""

import hashlib
import importlib.util
import json
import os
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original_cases = load("runtime_snapshot_io_fixtures", ROOT / "tests/sdsc/test_runtime_snapshot.py")
fixture = original_cases.fixture
snapshot = original_cases.snapshot


def fd_count():
    return len(os.listdir("/proc/self/fd"))


def test_exact_sorted_archive_bytes_source_identity_and_phase_counters(fixture):
    f = fixture
    expected_files = {
        "bin/python3.12": b"native executable fixture\x00\xff",
        "lib/python3.12/site-packages/demo/__init__.py": b"VALUE = 42\n",
        "lib/python3.12/site-packages/demo/data": bytes(range(256)) * 4097,
        "lib/python3.12/site-packages/demo/empty": b"",
    }
    for i in range(42):
        directory = f["prefix"] / "extras" / f"d{i:02}" / "inner"
        directory.mkdir(parents=True)
        for j in range(3):
            content = bytes([i, j]) * (100 + i)
            name = f"extras/d{i:02}/inner/file{j}"
            (f["prefix"] / name).write_bytes(content)
            expected_files[name] = content
    expected_files["extras/λ.txt"] = "雪".encode()
    (f["prefix"] / "extras/λ.txt").write_bytes(expected_files["extras/λ.txt"])
    (f["prefix"] / "extras/relative").symlink_to("d00/inner/file0")
    f["prefix"].chmod(0o2755)
    source_before = snapshot.scan(f["prefix"])
    diagnostics = {}
    got = snapshot.prepare(
        f["prefix"],
        f["artifact"],
        source_root=f["source"],
        artifact_root=f["artifacts"],
        diagnostics=diagnostics,
    )
    # Explicit bytes independently specify ordering, empty files, Unicode paths,
    # and symlink exclusion; this cannot become a compare-to-self reference.
    expected_archive = b"OPD_NATIVE_RUNTIME_V1\n" + b"".join(
        expected_files[name] for name in sorted(expected_files)
    )
    assert (f["artifact"] / "runtime.bin").read_bytes() == expected_archive
    assert got["archive"]["sha256"] == hashlib.sha256(expected_archive).hexdigest()
    value = json.loads((f["artifact"] / "manifest.json").read_bytes())
    actual_files = [row for row in value["entries"] if row["type"] == "file"]
    assert [row["path"] for row in actual_files] == sorted(expected_files)
    for row in actual_files:
        content = expected_files[row["path"]]
        assert row["size"] == len(content)
        assert row["sha256"] == hashlib.sha256(content).hexdigest()
    assert value["root_mode"] == 0o2755
    assert snapshot.scan(f["prefix"]) == source_before
    assert [p["phase"] for p in diagnostics["phases"]] == [
        "initial_inventory",
        "serial_pack",
        "final_inventory",
        "archive_readback",
    ]
    assert all(p["completed"] and p["elapsed_seconds"] >= 0 for p in diagnostics["phases"])
    initial, packing, final, readback = diagnostics["phases"]
    assert initial["entries"] == final["entries"] == len(source_before[0])
    assert initial["metadata_requests"] == initial["entries"]
    assert packing["files_packed"] == got["files_count"] == len(expected_files)
    assert packing["bytes_packed"] == got["total_bytes"] == sum(map(len, expected_files.values()))
    assert packing["peak_cached_fds"] == 32
    assert packing["parent_cache_hits"] >= 84
    assert readback["bytes_read"] == got["archive"]["size"] == len(expected_archive)


def test_metadata_actual_overlap_never_exceeds_four_and_leaves_no_fds(fixture, monkeypatch):
    f = fixture
    for i in range(24):
        (f["prefix"] / f"a{i:02}").write_bytes(b"x")
    original = snapshot.os.stat
    gate = threading.Barrier(4, timeout=3)
    lock = threading.Lock()
    active = peak = calls = 0

    def observed(*args, **kwargs):
        nonlocal active, peak, calls
        if not threading.current_thread().name.startswith("runtime-metadata"):
            return original(*args, **kwargs)
        with lock:
            active += 1
            peak = max(peak, active)
            calls += 1
            sequence = calls
        try:
            if sequence <= 4:
                gate.wait()
            time.sleep(0.001)
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(snapshot.os, "stat", observed)
    before = fd_count()
    counters = {}
    entries, _, _ = snapshot.scan(f["prefix"], counters=counters)
    assert peak == counters["peak_pending"] == 4
    assert active == 0 and calls == len(entries)
    assert fd_count() == before
    assert not any(t.name.startswith("runtime-metadata") for t in threading.enumerate())


def test_metadata_worker_failure_drains_before_directory_close(fixture, monkeypatch):
    f = fixture
    for name in ("aaa", "aab", "aac", "aad"):
        (f["prefix"] / name).write_bytes(b"x")
    original = snapshot.os.stat
    barrier = threading.Barrier(4, timeout=3)
    completed, bad_fd = [], []

    def fail_one(name, *args, **kwargs):
        if not threading.current_thread().name.startswith("runtime-metadata"):
            return original(name, *args, **kwargs)
        barrier.wait()
        if name == "aaa":
            raise OSError("injected metadata failure")
        time.sleep(0.02)
        try:
            os.fstat(kwargs["dir_fd"])
            return original(name, *args, **kwargs)
        except OSError as error:
            bad_fd.append(error)
            raise
        finally:
            completed.append(name)

    monkeypatch.setattr(snapshot.os, "stat", fail_one)
    before = fd_count()
    with pytest.raises(OSError, match="injected metadata failure"):
        snapshot.scan(f["prefix"])
    assert sorted(completed) == ["aab", "aac", "aad"] and not bad_fd
    assert fd_count() == before
    assert not any(t.name.startswith("runtime-metadata") for t in threading.enumerate())


def test_scanned_directory_replaced_with_external_link_never_followed(fixture, monkeypatch):
    f = fixture
    outside = f["source"] / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"must not be read")
    original = snapshot.os.stat
    changed = False

    def replace(name, *args, **kwargs):
        nonlocal changed
        value = original(name, *args, **kwargs)
        if name == "bin" and kwargs.get("dir_fd") is not None and not changed:
            changed = True
            (f["prefix"] / "bin").rename(f["prefix"] / "bin-before")
            (f["prefix"] / "bin").symlink_to(outside, target_is_directory=True)
        return value

    monkeypatch.setattr(snapshot.os, "stat", replace)
    before = fd_count()
    with pytest.raises((OSError, ValueError)):
        original_cases.prepare(f)
    assert changed and not f["artifact"].exists()
    assert fd_count() == before


def test_parent_cache_cap_eviction_reuse_and_all_descriptors_closed(fixture, monkeypatch):
    f = fixture
    for i in range(40):
        directory = f["prefix"] / f"cache{i:02}" / "deep"
        directory.mkdir(parents=True)
        (directory / "data").write_bytes(bytes([i]))
    _, initial, _ = snapshot.scan(f["prefix"])
    original_open, original_close = os.open, os.close
    root_fd = original_open(f["prefix"], os.O_RDONLY | os.O_DIRECTORY)
    owned, peak = set(), 0

    def tracked_open(path, flags, *args, **kwargs):
        nonlocal peak
        fd = original_open(path, flags, *args, **kwargs)
        if flags & os.O_DIRECTORY:
            owned.add(fd)
            peak = max(peak, len(owned))
        return fd

    def tracked_close(fd):
        owned.discard(fd)
        return original_close(fd)

    monkeypatch.setattr(snapshot.os, "open", tracked_open)
    monkeypatch.setattr(snapshot.os, "close", tracked_close)
    cache = snapshot.ParentDirectoryCache(root_fd, initial)
    try:
        for i in range(40):
            for _ in range(3):
                fd = snapshot.open_relative(cache, f"cache{i:02}/deep/data", os.O_RDONLY)
                with os.fdopen(fd, "rb") as stream:
                    assert stream.read() == bytes([i])
        assert peak == cache.counters["peak_cached_fds"] == 32
        assert cache.counters["directory_opens"] == 80
        assert cache.counters["parent_cache_hits"] == 80
    finally:
        cache.close()
        assert not owned
        os.fstat(root_fd)  # Borrowed root remains open.
        original_close(root_fd)


@pytest.mark.parametrize("mutation", ["chmod", "replace_by_symlink", "deadline"])
def test_cached_parent_revalidation_and_cleanup_on_error(fixture, mutation):
    f = fixture
    _, initial, _ = snapshot.scan(f["prefix"])
    before = fd_count()
    root_fd = os.open(f["prefix"], os.O_RDONLY | os.O_DIRECTORY)
    cache = snapshot.ParentDirectoryCache(root_fd, initial)
    try:
        fd = cache.open("bin/python3.12", os.O_RDONLY)
        os.close(fd)
        if mutation == "chmod":
            (f["prefix"] / "bin").chmod(0o700)
        elif mutation == "replace_by_symlink":
            (f["prefix"] / "bin").rename(f["prefix"] / "old-bin")
            (f["prefix"] / "bin").symlink_to("old-bin")
        else:
            cache.deadline = time.monotonic() - 1
        with pytest.raises(ValueError, match="directory changed|deadline"):
            cache.open("bin/python3.12", os.O_RDONLY)
    finally:
        cache.close()
        os.close(root_fd)
    assert fd_count() == before


def test_deadline_expiry_in_metadata_batch_drains_and_records_incomplete_phase(fixture, monkeypatch):
    f = fixture
    original = snapshot.os.stat

    def slow(*args, **kwargs):
        if threading.current_thread().name.startswith("runtime-metadata"):
            time.sleep(0.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(snapshot.os, "stat", slow)
    before = fd_count()
    diagnostics = {}
    with pytest.raises(ValueError, match="deadline"):
        snapshot.prepare(
            f["prefix"],
            f["artifact"],
            source_root=f["source"],
            artifact_root=f["artifacts"],
            deadline=time.monotonic() + 0.005,
            diagnostics=diagnostics,
        )
    assert len(diagnostics["phases"]) == 1 and diagnostics["phases"][0]["completed"] is False
    assert not f["artifact"].exists() and fd_count() == before


def test_serial_pack_failure_closes_cache_without_success_manifest(fixture, monkeypatch):
    f = fixture
    original = snapshot.open_relative
    calls = 0

    def broken(fd, name, flags):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected serial pack error")
        return original(fd, name, flags)

    monkeypatch.setattr(snapshot, "open_relative", broken)
    before = fd_count()
    diagnostics = {}
    with pytest.raises(OSError, match="injected serial pack error"):
        snapshot.prepare(
            f["prefix"],
            f["artifact"],
            source_root=f["source"],
            artifact_root=f["artifacts"],
            diagnostics=diagnostics,
        )
    assert fd_count() == before
    assert diagnostics["phases"][-1]["phase"] == "serial_pack"
    assert diagnostics["phases"][-1]["completed"] is False
    assert not (f["artifact"] / "manifest.json").exists()


def test_partial_worker_submission_failure_drains_before_fd_close(fixture, monkeypatch):
    f = fixture
    for name in ("aaa", "aab"):
        (f["prefix"] / name).write_bytes(b"x")
    original_stat, original_pool = snapshot.os.stat, snapshot.ThreadPoolExecutor
    started = threading.Event()
    completed = []

    def blocked_stat(name, *args, **kwargs):
        if threading.current_thread().name.startswith("runtime-metadata"):
            started.set()
            time.sleep(0.02)
            os.fstat(kwargs["dir_fd"])
            completed.append(name)
        return original_stat(name, *args, **kwargs)

    class FailingSubmitPool(original_pool):
        submissions = 0

        def submit(self, *args, **kwargs):
            self.submissions += 1
            if self.submissions == 2:
                assert started.wait(2)
                raise RuntimeError("injected thread submission failure")
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(snapshot.os, "stat", blocked_stat)
    monkeypatch.setattr(snapshot, "ThreadPoolExecutor", FailingSubmitPool)
    before = fd_count()
    with pytest.raises(RuntimeError, match="injected thread submission failure"):
        snapshot.scan(f["prefix"])
    assert completed == ["aaa"] and fd_count() == before
    assert not any(t.name.startswith("runtime-metadata") for t in threading.enumerate())


def test_submit_enqueues_then_raises_before_returned_future_keeps_nested_fd_alive(fixture, monkeypatch):
    original_stat, original_pool = snapshot.os.stat, snapshot.ThreadPoolExecutor
    started = threading.Event()
    completed, bad_fd = [], []

    def delayed_stat(name, *args, **kwargs):
        if threading.current_thread().name.startswith("runtime-metadata") and name == "python3.12":
            started.set()
            time.sleep(0.04)
            try:
                os.fstat(kwargs["dir_fd"])
                completed.append(name)
            except OSError as error:
                bad_fd.append(error.errno)
                raise
        return original_stat(name, *args, **kwargs)

    class EnqueueThenFailPool(original_pool):
        def submit(self, fn, fd, name):
            result = super().submit(fn, fd, name)
            if name == "python3.12":
                assert started.wait(2)
                raise RuntimeError("enqueued task lost its future")
            return result

    monkeypatch.setattr(snapshot.os, "stat", delayed_stat)
    monkeypatch.setattr(snapshot, "ThreadPoolExecutor", EnqueueThenFailPool)
    before = fd_count()
    with pytest.raises(RuntimeError, match="enqueued task lost its future"):
        snapshot.scan(fixture["prefix"])
    assert completed == ["python3.12"] and not bad_fd
    assert fd_count() == before
