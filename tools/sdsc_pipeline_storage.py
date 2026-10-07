"""Bounded immutable metadata and verified node-local/durable artifact transport."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath

PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PERSISTENT = {"lustre", "nfs", "nfs4", "ceph"}
LOCAL = {"ext2", "ext3", "ext4", "xfs", "btrfs"}
MODELS = (
    ("models--Qwen--Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("models--Qwen--Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def helper(name):
    spec = importlib.util.spec_from_file_location("_pipeline_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def safe(value):
    p = Path(value)
    require(
        p.is_absolute() and ".." not in p.parts and not any(ord(c) < 32 for c in str(p)),
        "unsafe absolute path",
    )
    require(not any(x.is_symlink() for x in (p, *p.parents)), "symlink path is forbidden")
    return p


def relative(value):
    require(
        isinstance(value, str) and value and "\\" not in value and not any(ord(c) < 32 for c in value),
        "bad relative path",
    )
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == value, "unsafe relative path")
    return p


def read(path, maximum=16 * 1024**2):
    p = safe(path)
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as f:
        before = os.fstat(f.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "metadata exceeds regular-file bound"
        )
        data = f.read(maximum + 1)
        after = os.fstat(f.fileno())
    require(
        len(data) == before.st_size and stamp(before) == stamp(after) == stamp(safe(p).lstat()),
        "metadata changed while reading",
    )
    return data


def document(path, expected=None):
    raw = read(path)
    require(expected is None or sha(raw) == expected, "metadata SHA mismatch")

    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, "duplicate JSON key")
            result[k] = v
        return result

    result = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(result, dict), "metadata must be a JSON object")
    canonical(result)
    return result


def atomic(path, value, *, replace=False):
    p = safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not p.exists(), "refusing to overwrite immutable metadata")
    tmp = p.with_name("." + p.name + "." + str(os.getpid()) + ".tmp")
    with tmp.open("xb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    if replace:
        os.replace(tmp, p)
    else:
        try:
            os.link(tmp, p)
        finally:
            tmp.unlink()
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha(read(p))


def identity(path):
    p = safe(path)
    h = hashlib.sha256()
    total = 0
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as f:
        before = os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode), "nonregular artifact")
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            total += len(chunk)
            h.update(chunk)
            require(total <= before.st_size, "artifact grew during hashing")
        after = os.fstat(f.fileno())
    require(
        total == before.st_size and stamp(before) == stamp(after) == stamp(safe(p).lstat()),
        "artifact changed during hashing",
    )
    return {"size": total, "sha256": h.hexdigest()}


def copy(source, destination, record=None):
    source, destination = safe(source), safe(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), "copy target already exists")
    h = hashlib.sha256()
    total = 0
    with (
        os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as incoming,
        destination.open("xb") as outgoing,
    ):
        before = os.fstat(incoming.fileno())
        require(stat.S_ISREG(before.st_mode), "nonregular artifact")
        for chunk in iter(lambda: incoming.read(8 * 1024**2), b""):
            total += len(chunk)
            require(total <= before.st_size, "artifact grew during copy")
            h.update(chunk)
            outgoing.write(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
        after = os.fstat(incoming.fileno())
    observed = {"size": total, "sha256": h.hexdigest()}
    require(
        total == before.st_size and stamp(before) == stamp(after) == stamp(safe(source).lstat()),
        "artifact changed during copy",
    )
    require(record is None or observed == {k: record[k] for k in observed}, "copied artifact SHA mismatch")
    require(identity(destination) == observed, "copy read-back mismatch")
    destination.chmod(record.get("mode", 0o644) if record else 0o644)
    return observed


def mount(path):
    result = subprocess.run(
        ["findmnt", "--json", "--target", str(safe(path)), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return json.loads(result.stdout)["filesystems"][0]


def persistent(path, *, writable=False):
    p = safe(path)
    require(PROJECT in p.parents and p.is_dir(), "persistent path outside project or missing")
    info = mount(p)
    require(info["fstype"] in PERSISTENT, "mount is not verified persistent storage")
    if writable:
        probe = p / (".node-probe-" + str(os.getpid()))
        with probe.open("xb") as f:
            f.write(b"quest-sdsc-pipeline")
            f.flush()
            os.fsync(f.fileno())
        require(probe.read_bytes() == b"quest-sdsc-pipeline", "persistent write/read failed")
        probe.unlink()
    return info


def records(root):
    root = safe(root)
    result = {}
    if not root.exists():
        return result
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in {".cache", "__pycache__"}]
        require(not any((Path(folder) / n).is_symlink() for n in dirs), "symlink output directory")
        for name in sorted(files):
            path = safe(Path(folder) / name)
            result[str(path.relative_to(root))] = identity(path)
    return result


def stage_inventory(inventory, roots):
    total = sum(r["size"] for r in inventory.values())
    first = next(iter(roots.values()))
    require(
        shutil.disk_usage(first.parent).free >= total + 256 * 1024**3,
        "node scratch lacks 256 GiB checkpoint margin",
    )
    for name, record in inventory.items():
        p = relative(name)
        require(len(p.parts) > 1 and p.parts[0] in roots, "unknown artifact root")
        source = safe(record["storage"])
        require(PROJECT in source.parents, "artifact storage outside project")
        copy(source, roots[p.parts[0]].joinpath(*p.parts[1:]), record)


def stage_release(run_id, code_hash, destination):
    release = CONTROL / "releases" / run_id
    manifest = document(release / "manifest.json")
    require(
        manifest["run_id"] == run_id
        and manifest["code_sha256"] == code_hash == sha(canonical(manifest["files"])),
        "release identity differs",
    )
    seen = set()
    for entry in manifest["files"]:
        name = str(relative(entry["path"]))
        require(name not in seen and entry["size"] <= 4 * 1024**2, "invalid source inventory")
        seen.add(name)
        copy(release / "source" / name, destination / name, entry)
    return manifest


def stage_models(cache, destination):
    persistent(cache)
    files = []
    for model, revision in MODELS:
        snapshot = safe(cache / "hub" / model / "snapshots" / revision)
        for folder, dirs, names in os.walk(snapshot, followlinks=False):
            require(not any((Path(folder) / n).is_symlink() for n in dirs), "model snapshot directory link")
            for name in names:
                p = Path(folder) / name
                if name.startswith(".") or ".cache" in p.parts:
                    continue
                resolved = p.resolve(strict=True)
                require(cache in resolved.parents and resolved.is_file(), "model file outside pinned cache")
                if name.endswith((".json", ".safetensors", ".txt", ".model", ".jinja")):
                    files.append((p, resolved))
    require(files and sum(p.stat().st_size for _, p in files) < 24 * 1024**3, "unexpected model inventory")
    for logical, real in files:
        copy(real, destination / logical.relative_to(cache))


def merge_receipts(base, receipts):
    """Merge independent array deltas; two writers of one changed path are rejected."""
    result = dict(base)
    written = {}
    for receipt in receipts:
        require(
            receipt.get("passed") is True and receipt.get("persisted") is True, "unpublished or failed stage"
        )
        for name, entry in receipt["delta"].items():
            relative(name)
            require(
                name not in written or all(written[name][k] == entry[k] for k in ("size", "sha256")),
                "array publication collision",
            )
            if name not in written:
                result[name] = entry
                written[name] = entry
    return result


def publish(roots, before, destination, report):
    persistent(destination, writable=True)
    report_sha = atomic(destination / "result.json", report)
    delta = {}
    for role, root in roots.items():
        for name, record in records(root).items():
            key = role + "/" + name
            if key in before and all(before[key][k] == record[k] for k in ("size", "sha256")):
                continue
            target = destination / "artifacts" / role / name
            copy(root / name, target, record)
            delta[key] = {**record, "storage": str(target)}
    receipt = {k: report[k] for k in ("flow_id", "stage", "job_id", "run_id", "code_sha256", "passed")}
    receipt.update(persisted=True, persistent_read_back_verified=True, result_sha256=report_sha, delta=delta)
    atomic(destination / "receipt.json", receipt)
    return receipt
