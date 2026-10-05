"""Byte-preserving native runtime snapshots; no install, prefix rewrite, or execution.

The caller supplies owned roots and a trusted manifest SHA. A snapshot includes
the complete prefix, including package metadata and bytecode. It is not a secret
scrubber: the caller must select the already reviewed runtime, not a data/cache
or credentials directory. Only relative, internal, resolvable symlinks survive.

runtime.bin is deliberately not tar: a fixed magic followed by regular files in
manifest order. All names, sizes and link targets come from the bounded, hashed
manifest. No archive-controlled extraction paths or generic tar parser exist.
Successful staging proves identical bytes, not that Conda/native libraries are
relocatable. The consumer must independently prove interpreter, import, and
native-library origins from the relocated prefix before using it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path, PurePosixPath

SCHEMA = "quest-sdsc-native-runtime-snapshot-v1"
STAGE_SCHEMA = "quest-sdsc-native-runtime-stage-v1"
REHEARSAL_SCHEMA = "quest-sdsc-native-runtime-rehearsal-v1"
MAGIC = b"OPD_NATIVE_RUNTIME_V1\n"
MAX_ENTRIES = 250_000
MAX_FILE_BYTES = 8 * 1024**3
MAX_TOTAL_BYTES = 32 * 1024**3
MAX_MANIFEST_BYTES = 64 * 1024**2
FREE_RESERVE_BYTES = 1024**3
MAX_DEPTH = 64
CHUNK = 1024**2
LOCAL_FILESYSTEMS = frozenset(("ext2", "ext3", "ext4", "xfs", "btrfs"))
HEX = re.compile(r"[a-f0-9]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_deadline(deadline):
    if deadline is not None:
        require(time.monotonic() < deadline, "runtime snapshot deadline exceeded")


def check_space(path, payload_bytes):
    observed = os.statvfs(path)
    available = observed.f_bavail * observed.f_frsize
    require(available >= payload_bytes + FREE_RESERVE_BYTES, "insufficient runtime snapshot free space")


def relative(value):
    require(isinstance(value, str) and value and len(value.encode()) <= 4096, "invalid relative path")
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and str(path) == value
        and not any(part in (".", "..") for part in path.parts)
        and len(path.parts) <= MAX_DEPTH
        and "\\" not in value
        and all(ord(c) >= 32 and ord(c) != 127 for c in value),
        "unsafe relative path",
    )
    return value


def absolute(value):
    require(isinstance(value, str | Path), "invalid absolute path type")
    path = Path(value)
    require(path.is_absolute() and str(path) == str(value) and ".." not in path.parts, "unsafe absolute path")
    require(all(ord(c) >= 32 and ord(c) != 127 for c in str(path)), "unsafe absolute path")
    return path


def owned_path(value, root, *, exists=True):
    root, path = absolute(root), absolute(value)
    require(path == root or root in path.parents, "path escapes supplied root")
    for part in [*reversed(path.parents), path]:
        try:
            item = part.lstat()
        except FileNotFoundError:
            require(not exists and part == path, "missing path ancestor")
            continue
        require(not stat.S_ISLNK(item.st_mode), "symlink in owned path")
        if part == root or root in part.parents:
            require(item.st_uid == os.getuid(), "path is not owned by current user")
        if part != path:
            require(stat.S_ISDIR(item.st_mode), "non-directory path ancestor")
    require(root.is_dir(), "owned root must be a directory")
    return path


def stamp(item):
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def mode(item):
    # Shared project directories legitimately inherit SGID (group inheritance),
    # including the original Conda prefix. Preserve it only for directories;
    # executable privilege bits, sticky bits and SGID files remain forbidden.
    require(
        not item.st_mode & 0o5000 and (stat.S_ISDIR(item.st_mode) or not item.st_mode & 0o2000),
        "special permission bits forbidden",
    )
    return stat.S_IMODE(item.st_mode)


def open_relative(root_fd, name, flags):
    parts = PurePosixPath(relative(name)).parts
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = following
        return os.open(parts[-1], flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
    finally:
        os.close(current)


def scan(prefix, *, deadline=None):
    """lstat/openat traversal; never follow a source symlink while inventorying."""
    entries, stamps, total = [], {}, 0
    root_fd = os.open(prefix, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        root_stat = os.fstat(root_fd)
        require(root_stat.st_uid == os.getuid(), "runtime root is not owned")
        stamps[""] = stamp(root_stat)

        def walk(fd, parent):
            nonlocal total
            with os.scandir(fd) as listing:
                names = sorted(entry.name for entry in listing)
            require(len(names) + len(entries) <= MAX_ENTRIES, "runtime entry count exceeds bound")
            for name in names:
                check_deadline(deadline)
                path = relative(parent + "/" + name if parent else name)
                item = os.stat(name, dir_fd=fd, follow_symlinks=False)
                require(item.st_uid == os.getuid(), "runtime entry is not owned")
                stamps[path] = stamp(item)
                row = dict(path=path, mode=mode(item))
                if stat.S_ISDIR(item.st_mode):
                    row["type"] = "directory"
                    entries.append(row)
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        require(stamp(os.fstat(child)) == stamps[path], "source directory changed")
                        walk(child, path)
                        require(stamp(os.fstat(child)) == stamps[path], "source directory changed")
                    finally:
                        os.close(child)
                elif stat.S_ISREG(item.st_mode):
                    require(0 <= item.st_size <= MAX_FILE_BYTES, "runtime file exceeds bound")
                    total += item.st_size
                    require(total <= MAX_TOTAL_BYTES, "runtime total exceeds bound")
                    row.update(type="file", size=item.st_size)
                    entries.append(row)
                elif stat.S_ISLNK(item.st_mode):
                    row.update(type="symlink", target=os.readlink(name, dir_fd=fd))
                    entries.append(row)
                else:
                    raise ValueError("non-regular runtime entry forbidden")
                require(len(entries) <= MAX_ENTRIES, "runtime entry count exceeds bound")

        walk(root_fd, "")
        require(stamp(os.fstat(root_fd)) == stamps[""], "source runtime changed")
        entries.sort(key=lambda row: row["path"])
        validate_entries(entries, hashed=False)
        return entries, stamps, mode(root_stat)
    finally:
        os.close(root_fd)


def validate_entries(entries, *, hashed=True):
    require(isinstance(entries, list) and 0 < len(entries) <= MAX_ENTRIES, "invalid runtime inventory")
    seen, total, files = {}, 0, 0
    for row in entries:
        require(isinstance(row, dict), "invalid runtime record")
        kind = row.get("type")
        fields = {"path", "type", "mode"}
        if kind == "file":
            fields |= {"size", "sha256"} if hashed else {"size"}
        elif kind == "symlink":
            fields |= {"target"}
        require(kind in ("file", "directory", "symlink") and set(row) == fields, "invalid runtime record")
        name = relative(row["path"])
        allowed_mode = 0o2777 if kind == "directory" else 0o777
        require(
            type(row["mode"]) is int and row["mode"] >= 0 and not row["mode"] & ~allowed_mode,
            "invalid runtime mode",
        )
        require(name not in seen, "duplicate runtime path")
        for ancestor in PurePosixPath(name).parents:
            if str(ancestor) != ".":
                require(
                    seen.get(str(ancestor), {}).get("type") == "directory",
                    "missing or non-directory ancestor",
                )
        if kind == "file":
            require(type(row["size"]) is int and 0 <= row["size"] <= MAX_FILE_BYTES, "invalid runtime size")
            if hashed:
                require(
                    isinstance(row["sha256"], str) and HEX.fullmatch(row["sha256"]), "invalid runtime SHA"
                )
            files += 1
            total += row["size"]
        elif kind == "symlink":
            target = row["target"]
            require(
                isinstance(target, str)
                and 0 < len(target.encode()) <= 4096
                and not target.startswith("/")
                and "\\" not in target
                and all(ord(c) >= 32 and ord(c) != 127 for c in target)
                and row["mode"] == 0o777,
                "absolute or invalid symlink target",
            )
        seen[name] = row
    require(
        list(seen) == sorted(seen) and files > 0 and total <= MAX_TOTAL_BYTES, "invalid inventory order/total"
    )

    def resolve(parts, visited):
        current = []
        for index, component in enumerate(parts):
            if component in ("", "."):
                continue
            if component == "..":
                require(current, "symlink escapes runtime")
                current.pop()
                continue
            current.append(component)
            name = "/".join(current)
            row = seen.get(name)
            require(row is not None, "dangling symlink target")
            if row["type"] == "symlink":
                require(name not in visited and len(visited) < MAX_DEPTH, "cyclic symlink target")
                return resolve(current[:-1] + row["target"].split("/") + parts[index + 1 :], visited | {name})
            if index < len(parts) - 1:
                require(row["type"] == "directory", "non-directory symlink traversal")
        return "/".join(current)

    for name, row in seen.items():
        if row["type"] == "symlink":
            resolve(name.split("/")[:-1] + row["target"].split("/"), {name})
    return files, total


def digest_file(path, limit, *, deadline=None):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(fd)
        require(
            stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_size <= limit,
            "invalid bounded owned file",
        )
        digest, size = hashlib.sha256(), 0
        while chunk := stream.read(CHUNK):
            check_deadline(deadline)
            size += len(chunk)
            require(size <= limit, "file exceeds bound")
            digest.update(chunk)
        require(size == before.st_size and stamp(before) == stamp(os.fstat(fd)), "file changed while reading")
    return dict(size=size, sha256=digest.hexdigest())


def write_once(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(
    prefix, artifact_dir, *, source_root, artifact_root, python_relative_path="bin/python3.12", deadline=None
):
    prefix = owned_path(prefix, source_root)
    artifact = owned_path(artifact_dir, artifact_root, exists=False)
    require(prefix.is_dir() and not artifact.exists(), "runtime or fresh artifact directory required")
    require(
        prefix != artifact and prefix not in artifact.parents and artifact not in prefix.parents,
        "source and artifact overlap",
    )
    relative(python_relative_path)
    entries, initial, root_mode = scan(prefix, deadline=deadline)
    check_space(artifact.parent, len(MAGIC) + sum(row.get("size", 0) for row in entries))
    artifact.mkdir(mode=0o700)
    archive_path = artifact / "runtime.bin"
    root_fd = os.open(prefix, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    digest, size = hashlib.sha256(), 0
    try:
        with archive_path.open("xb") as output:
            output.write(MAGIC)
            digest.update(MAGIC)
            size += len(MAGIC)
            for row in entries:
                if row["type"] != "file":
                    continue
                fd = open_relative(root_fd, row["path"], os.O_RDONLY)
                with os.fdopen(fd, "rb") as stream:
                    require(stamp(os.fstat(fd)) == initial[row["path"]], "source file changed before packing")
                    file_hash, count = hashlib.sha256(), 0
                    while chunk := stream.read(CHUNK):
                        check_deadline(deadline)
                        count += len(chunk)
                        require(count <= row["size"], "source file grew while packing")
                        output.write(chunk)
                        digest.update(chunk)
                        file_hash.update(chunk)
                    require(
                        count == row["size"] and stamp(os.fstat(fd)) == initial[row["path"]],
                        "source file changed while packing",
                    )
                row["sha256"] = file_hash.hexdigest()
                size += count
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(root_fd)
    _, final, final_mode = scan(prefix, deadline=deadline)
    require(initial == final and root_mode == final_mode, "runtime inventory changed while packing")
    files, total = validate_entries(entries)
    interpreter = next((r for r in entries if r["path"] == python_relative_path), {})
    require(
        interpreter.get("type") == "file" and interpreter["mode"] & 0o111,
        "regular executable Python required",
    )
    manifest = dict(
        schema=SCHEMA,
        source_prefix=str(prefix),
        python_relative_path=python_relative_path,
        python_sha256=interpreter["sha256"],
        root_mode=root_mode,
        entries=entries,
        entry_count=len(entries),
        files_count=files,
        total_bytes=total,
        runtime_root_sha256=sha(canonical(dict(root_mode=root_mode, entries=entries))),
        archive=dict(path="runtime.bin", size=size, sha256=digest.hexdigest()),
    )
    raw = canonical(manifest)
    require(len(raw) <= MAX_MANIFEST_BYTES, "runtime manifest exceeds bound")
    require(
        digest_file(archive_path, MAX_TOTAL_BYTES + len(MAGIC), deadline=deadline)
        == {k: manifest["archive"][k] for k in ("size", "sha256")},
        "archive readback differs",
    )
    write_once(artifact / "manifest.json", raw)
    return describe(artifact, sha(raw), artifact_root=artifact_root)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_manifest(artifact_dir, expected_manifest_sha256, *, artifact_root):
    artifact = owned_path(artifact_dir, artifact_root)
    path = owned_path(artifact / "manifest.json", artifact_root)
    require(
        isinstance(expected_manifest_sha256, str) and HEX.fullmatch(expected_manifest_sha256),
        "invalid trust SHA",
    )
    observed = digest_file(path, MAX_MANIFEST_BYTES)
    require(observed["sha256"] == expected_manifest_sha256, "manifest SHA mismatch")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.getuid()
            and before.st_size <= MAX_MANIFEST_BYTES,
            "invalid owned manifest",
        )
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
        require(stamp(before) == stamp(os.fstat(stream.fileno())), "manifest changed while reading")
    require(len(raw) == observed["size"] and sha(raw) == expected_manifest_sha256, "manifest changed")
    value = json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )
    fields = {
        "schema",
        "source_prefix",
        "python_relative_path",
        "python_sha256",
        "root_mode",
        "entries",
        "entry_count",
        "files_count",
        "total_bytes",
        "runtime_root_sha256",
        "archive",
    }
    require(
        isinstance(value, dict) and set(value) == fields and value["schema"] == SCHEMA,
        "runtime manifest schema differs",
    )
    require(canonical(value) == raw, "noncanonical runtime manifest")
    absolute(value["source_prefix"])
    require(
        type(value["root_mode"]) is int and value["root_mode"] >= 0 and not value["root_mode"] & ~0o2777,
        "invalid runtime root mode",
    )
    files, total = validate_entries(value["entries"])
    require(
        all(type(value[k]) is int for k in ("entry_count", "files_count", "total_bytes"))
        and (value["entry_count"], value["files_count"], value["total_bytes"])
        == (len(value["entries"]), files, total),
        "runtime inventory counts differ",
    )
    require(
        value["runtime_root_sha256"]
        == sha(canonical(dict(root_mode=value["root_mode"], entries=value["entries"]))),
        "runtime inventory SHA differs",
    )
    relative(value["python_relative_path"])
    interpreter = next((r for r in value["entries"] if r["path"] == value["python_relative_path"]), {})
    require(
        interpreter.get("type") == "file"
        and interpreter["mode"] & 0o111
        and value["python_sha256"] == interpreter["sha256"],
        "runtime Python identity differs",
    )
    a = value["archive"]
    require(
        isinstance(a, dict)
        and set(a) == {"path", "size", "sha256"}
        and a["path"] == "runtime.bin"
        and type(a["size"]) is int
        and a["size"] == len(MAGIC) + total
        and isinstance(a["sha256"], str)
        and HEX.fullmatch(a["sha256"]),
        "runtime archive identity differs",
    )
    return value


def validate_descriptor(value):
    fields = {
        "manifest",
        "archive",
        "runtime_root_sha256",
        "source_prefix",
        "python_relative_path",
        "python_sha256",
        "files_count",
        "total_bytes",
    }
    require(isinstance(value, dict) and set(value) == fields, "snapshot descriptor fields differ")
    absolute(value["source_prefix"])
    relative(value["python_relative_path"])
    for key in ("runtime_root_sha256", "python_sha256"):
        require(isinstance(value[key], str) and HEX.fullmatch(value[key]), "snapshot descriptor SHA differs")
    require(
        type(value["files_count"]) is int
        and 0 < value["files_count"] <= MAX_ENTRIES
        and type(value["total_bytes"]) is int
        and 0 <= value["total_bytes"] <= MAX_TOTAL_BYTES,
        "snapshot descriptor counts differ",
    )
    for key, filename, bound in (
        ("manifest", "manifest.json", MAX_MANIFEST_BYTES),
        ("archive", "runtime.bin", MAX_TOTAL_BYTES + len(MAGIC)),
    ):
        row = value[key]
        require(
            isinstance(row, dict) and set(row) == {"path", "size", "sha256"},
            "snapshot file descriptor differs",
        )
        path = absolute(row["path"])
        require(
            path.name == filename
            and type(row["size"]) is int
            and 0 < row["size"] <= bound
            and isinstance(row["sha256"], str)
            and HEX.fullmatch(row["sha256"]),
            "snapshot file identity differs",
        )
    require(
        Path(value["manifest"]["path"]).parent == Path(value["archive"]["path"]).parent
        and value["archive"]["size"] == len(MAGIC) + value["total_bytes"],
        "snapshot archive framing differs",
    )
    return value


def describe(artifact_dir, expected_manifest_sha256, *, artifact_root):
    value = load_manifest(artifact_dir, expected_manifest_sha256, artifact_root=artifact_root)
    result = {
        k: value[k]
        for k in (
            "runtime_root_sha256",
            "source_prefix",
            "python_relative_path",
            "python_sha256",
            "files_count",
            "total_bytes",
        )
    }
    result.update(
        manifest=dict(
            path=str(Path(artifact_dir) / "manifest.json"),
            size=len(canonical(value)),
            sha256=expected_manifest_sha256,
        ),
        archive=dict(value["archive"], path=str(Path(artifact_dir) / "runtime.bin")),
    )
    return validate_descriptor(result)


def verify_archive(artifact_dir, manifest, *, artifact_root, deadline=None):
    path = owned_path(Path(artifact_dir) / "runtime.bin", artifact_root)
    actual = digest_file(path, MAX_TOTAL_BYTES + len(MAGIC), deadline=deadline)
    require(
        actual == {k: manifest["archive"][k] for k in ("size", "sha256")}, "runtime archive SHA/size differs"
    )
    return dict(path=str(path), **actual)


def mount_info(path):
    def decode(text):
        return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), text)

    path, matches = absolute(path), []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        before, after = left.split(), right.split()
        target = Path(decode(before[4]))
        if path == target or target in path.parents:
            matches.append(
                dict(target=str(target), fstype=after[0], source=decode(after[1]), options=before[5])
            )
    require(matches, "runtime mount evidence absent")
    return max(matches, key=lambda row: len(Path(row["target"]).parts))


def stage(
    artifact_dir, expected_manifest_sha256, destination, *, artifact_root, destination_root, deadline=None
):
    return _extract(
        artifact_dir,
        expected_manifest_sha256,
        destination,
        artifact_root=artifact_root,
        destination_root=destination_root,
        deadline=deadline,
        rehearsal=False,
    )


def rehearse(
    artifact_dir, expected_manifest_sha256, destination, *, artifact_root, destination_root, deadline=None
):
    """Explicit CPU rehearsal: byte verification only, never node-local admission."""
    return _extract(
        artifact_dir,
        expected_manifest_sha256,
        destination,
        artifact_root=artifact_root,
        destination_root=destination_root,
        deadline=deadline,
        rehearsal=True,
    )


def _extract(
    artifact_dir,
    expected_manifest_sha256,
    destination,
    *,
    artifact_root,
    destination_root,
    deadline,
    rehearsal,
):
    descriptor = describe(artifact_dir, expected_manifest_sha256, artifact_root=artifact_root)
    manifest = load_manifest(artifact_dir, expected_manifest_sha256, artifact_root=artifact_root)
    destination = owned_path(destination, destination_root, exists=False)
    staging = owned_path(
        destination.with_name(destination.name + ".snapshot-incomplete"), destination_root, exists=False
    )
    require(not destination.exists() and not staging.exists(), "runtime destination must be fresh")
    local_mount = mount_info(destination.parent)
    require(
        rehearsal or local_mount["fstype"] in LOCAL_FILESYSTEMS,
        "runtime destination is not verified node-local storage",
    )
    require("rw" in local_mount["options"].split(","), "runtime destination mount is not writable")
    check_space(destination.parent, descriptor["archive"]["size"] + descriptor["total_bytes"])
    # A fresh staging directory remains visibly incomplete after any failure.
    # It is never reused or silently removed by a subsequent call.
    destination.mkdir(mode=0o700)
    staging.mkdir(mode=0o700)
    local_archive = staging / "runtime.bin"
    source = owned_path(descriptor["archive"]["path"], artifact_root)
    with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as incoming:
        before = os.fstat(incoming.fileno())
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == os.getuid()
            and before.st_size == descriptor["archive"]["size"],
            "invalid source archive",
        )
        digest, size = hashlib.sha256(), 0
        with local_archive.open("xb") as output:
            while chunk := incoming.read(CHUNK):
                check_deadline(deadline)
                size += len(chunk)
                require(size <= descriptor["archive"]["size"], "archive grew during copy")
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        require(stamp(before) == stamp(os.fstat(incoming.fileno())), "archive changed during copy")
    require(
        size == descriptor["archive"]["size"] and digest.hexdigest() == descriptor["archive"]["sha256"],
        "copied archive SHA/size differs",
    )
    # Read back the local copy before extracting any runtime entry.
    require(
        digest_file(local_archive, MAX_TOTAL_BYTES + len(MAGIC), deadline=deadline)
        == {k: descriptor["archive"][k] for k in ("size", "sha256")},
        "local archive readback differs",
    )
    with os.fdopen(os.open(local_archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        require(stream.read(len(MAGIC)) == MAGIC, "runtime archive magic differs")
        for row in manifest["entries"]:
            check_deadline(deadline)
            path = destination / row["path"]
            if row["type"] == "directory":
                path.mkdir(mode=0o700)
            elif row["type"] == "file":
                digest, remaining = hashlib.sha256(), row["size"]
                with path.open("xb") as output:
                    while remaining:
                        check_deadline(deadline)
                        chunk = stream.read(min(CHUNK, remaining))
                        require(chunk, "truncated runtime archive")
                        output.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                require(digest.hexdigest() == row["sha256"], "extracted runtime file SHA differs")
                path.chmod(row["mode"])
        require(stream.read(1) == b"", "trailing runtime archive bytes")
    for row in manifest["entries"]:
        path = destination / row["path"]
        if row["type"] == "symlink":
            path.symlink_to(row["target"])
    for row in reversed(manifest["entries"]):
        if row["type"] == "directory":
            (destination / row["path"]).chmod(row["mode"])
    destination.chmod(manifest["root_mode"])
    actual, _, root_mode = scan(destination, deadline=deadline)
    for row in actual:
        if row["type"] == "file":
            row["sha256"] = digest_file(destination / row["path"], MAX_FILE_BYTES, deadline=deadline)[
                "sha256"
            ]
    require(
        actual == manifest["entries"] and root_mode == manifest["root_mode"], "extracted inventory differs"
    )
    # Remove only our verified temporary archive, and only after full readback.
    local_archive.unlink()
    staging.rmdir()
    return dict(
        schema=REHEARSAL_SCHEMA if rehearsal else STAGE_SCHEMA,
        node_local_verified=not rehearsal,
        descriptor=descriptor,
        destination=str(destination),
        python=str(destination / manifest["python_relative_path"]),
        local_mount=local_mount,
        files_verified=True,
        archive_verified=True,
        runtime_root_sha256=manifest["runtime_root_sha256"],
        files_count=manifest["files_count"],
        total_bytes=manifest["total_bytes"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("prepare")
    create.add_argument("--prefix", required=True)
    create.add_argument("--source-root", required=True)
    create.add_argument("--python-relative-path", default="bin/python3.12")
    for name in ("describe", "verify", "stage", "rehearse"):
        item = actions.add_parser(name)
        item.add_argument("--manifest-sha256", required=True)
        if name in ("stage", "rehearse"):
            item.add_argument("--destination", required=True)
            item.add_argument("--destination-root", required=True)
    for item in actions.choices.values():
        item.add_argument("--artifact-dir", required=True)
        item.add_argument("--artifact-root", required=True)
    args = parser.parse_args(argv)
    options = dict(artifact_root=args.artifact_root)
    if args.action == "prepare":
        result = prepare(
            args.prefix,
            args.artifact_dir,
            source_root=args.source_root,
            python_relative_path=args.python_relative_path,
            **options,
        )
    elif args.action in ("stage", "rehearse"):
        result = (stage if args.action == "stage" else rehearse)(
            args.artifact_dir,
            args.manifest_sha256,
            args.destination,
            destination_root=args.destination_root,
            **options,
        )
    else:
        result = describe(args.artifact_dir, args.manifest_sha256, **options)
        if args.action == "verify":
            value = load_manifest(args.artifact_dir, args.manifest_sha256, **options)
            verify_archive(args.artifact_dir, value, **options)
    print(canonical(result).decode(), end="")


if __name__ == "__main__":
    main()
