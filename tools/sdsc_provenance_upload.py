"""Upload a separately hashed, bounded Git provenance artifact through the SSH master.

This never submits a job and never copies .git/config, credentials, or local refs.
Only artifacts prepared by sdsc_provenance.py are accepted. An existing remote
artifact is verified, never overwritten; a failed upload is not retried here.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import tarfile
from pathlib import Path


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cli = module("sdsc_cli")
provenance = module("sdsc_provenance")
FILES = ("manifest.json", "wrapper-manifest.json", "history.bundle")


def archive_artifact(artifact):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in FILES:
            data, _ = provenance.read_regular(
                artifact / name, provenance.MAX_TOTAL if name == "history.bundle" else provenance.MAX_FILE
            )
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(data), 0o444
            archive.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


REMOTE = r"""
import hashlib, io, json, os, pathlib, pwd, re, stat, sys, tarfile, tempfile

request = json.loads(sys.argv[1])
assert pwd.getpwuid(os.getuid()).pw_name == "zgao12", "Unexpected remote identity"
root = pathlib.Path("/home/zgao12/quest-runs/OPD")
run_id, expected, code_sha = (request[key] for key in ("run_id", "manifest_sha256", "code_sha256"))
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", run_id)
assert all(re.fullmatch(r"[a-f0-9]{64}", value) for value in (expected, code_sha))
def safe(path):
    assert path.is_absolute() and ".." not in path.parts
    assert not any(p.is_symlink() for p in (path, *path.parents)), "Symlink path forbidden"
    return path
def bounded(path):
    path = safe(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and before.st_size <= 4 * 1024 * 1024
        data = stream.read(4 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
        assert len(data) == before.st_size and before.st_mtime_ns == after.st_mtime_ns
    return data
release = safe(root / "releases" / run_id)
manifest = json.loads(bounded(release / "manifest.json"))
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
assert manifest["run_id"] == run_id and manifest["code_sha256"] == code_sha
assert hashlib.sha256(canonical(manifest["files"])).hexdigest() == code_sha
entry = next(item for item in manifest["files"] if item["path"] == "tools/sdsc_provenance.py")
verifier_path = safe(release / "source/tools/sdsc_provenance.py")
verifier_bytes = bounded(verifier_path)
assert len(verifier_bytes) == entry["size"] and hashlib.sha256(verifier_bytes).hexdigest() == entry["sha256"]
namespace = {"__name__": "sdsc_provenance", "__file__": str(verifier_path)}
exec(compile(verifier_bytes, str(verifier_path), "exec"), namespace)
parent = safe(root / "provenance")
parent.mkdir(exist_ok=True, mode=0o700)
assert parent.stat().st_uid == os.getuid()
destination = safe(parent / expected)
raw = sys.stdin.buffer.read(72 * 1024 * 1024 + 1)
assert len(raw) <= 72 * 1024 * 1024, "Provenance archive exceeds bound"
assert hashlib.sha256(raw).hexdigest() == request["archive_sha256"], "Upload hash differs"
expected_names = {"manifest.json", "wrapper-manifest.json", "history.bundle"}
if destination.exists():
    assert {p.name for p in destination.iterdir()} == expected_names, "Unexpected provenance files"
    report = namespace["verify"](destination, expected, code_sha)
else:
    with tempfile.TemporaryDirectory(prefix=".upload-", dir=parent) as temporary:
        staging = pathlib.Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            members = archive.getmembers()
            assert len(members) == 3 and {item.name for item in members} == expected_names
            for item in members:
                maximum = 64 * 1024 * 1024 if item.name == "history.bundle" else 4 * 1024 * 1024
                assert item.isfile() and 0 < item.size <= maximum and not item.pax_headers
                data = archive.extractfile(item).read(maximum + 1)
                assert len(data) == item.size
                with (staging / item.name).open("xb") as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
        report = namespace["verify"](staging, expected, code_sha)
        wrapper = namespace["load_wrapper"](staging / "wrapper-manifest.json")
        assert wrapper == manifest, "Provenance wrapper differs from deployed release"
        destination.mkdir(mode=0o700)  # Exclusive: never replace a competing destination.
        for name in expected_names:
            os.rename(staging / name, destination / name)
wrapper = namespace["load_wrapper"](destination / "wrapper-manifest.json")
assert wrapper == manifest
assert {p.name for p in destination.iterdir()} == expected_names
# Only the three already validated regular files are present. Complete permissions
# if an earlier upload was interrupted between publication and chmod.
namespace["readonly_tree"](destination)
assert not any(p.stat().st_mode & 0o222 for p in (destination, *destination.iterdir()))
for directory in (destination, parent):
    fd = os.open(str(directory), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
print(json.dumps(dict(report, remote_artifact=str(destination), run_id=run_id)))
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    record = cli.run_record(args.run_id)
    if record["state"] != "deployed":
        raise ValueError("Upload the matching wrapper release first")
    code_sha = record["manifest"]["code_sha256"]
    artifact = provenance.safe_path(args.artifact)
    verified = provenance.verify(artifact, args.manifest_sha256, code_sha)
    wrapper = provenance.load_wrapper(artifact / "wrapper-manifest.json")
    if wrapper != record["manifest"]:
        raise ValueError("Provenance wrapper differs from deployed run record")
    archive = archive_artifact(artifact)
    plan = dict(
        verified,
        run_id=args.run_id,
        code_sha256=code_sha,
        archive_sha256=provenance.sha(archive),
        archive_bytes=len(archive),
        destination=cli.REMOTE_ROOT + "/provenance/" + args.manifest_sha256,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    check = cli.read_json(cli.state_root() / "check.json")
    if not check.get("connected"):
        raise ValueError("A successful tools/sdsc check is required")
    result = cli.ssh_call(
        [check["control_python"], "-I", "-B", "-c", REMOTE, json.dumps(plan)], archive, timeout=180
    )
    if result.returncode:
        raise ValueError("Provenance upload failed; inspect the remote artifact before any continuation")
    receipt = json.loads(result.stdout)
    for key in ("manifest_sha256", "wrapper_code_sha256", "git_head", "bundle_sha256"):
        if receipt.get(key) != verified[key]:
            raise ValueError("Provenance upload receipt identity mismatch")
    if receipt.get("run_id") != args.run_id or receipt.get("remote_artifact") != plan["destination"]:
        raise ValueError("Provenance upload receipt destination mismatch")
    cli.write_json(cli.state_root() / ("provenance-upload-" + args.manifest_sha256 + ".json"), receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
