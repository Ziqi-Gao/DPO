"""Explicit historical release replay, never a snapshot of current Quest edits.

prepare() verifies an existing deployed release and its genuine Git provenance,
then freezes a byte-identical source copy with a fresh run ID. The old real Git
HEAD and bundle remain unchanged; the current Quest HEAD is recorded separately
as preparation metadata. Current edits enter the replay only when their bytes
already exactly match the old inventory. Missing historical dirty files may be
read through the existing SSH master with explicit allow_remote_source=True.

deploy() requires the saved dry-run and unchanged controls/staged bytes. It uses
the existing immutable release upload and provenance upload interfaces, never
submits a job, and never retries an interrupted upload. Inspect state.json and
reconcile the exact destinations before taking any further action after failure.
No source refs, index, worktree, old release, or old provenance is modified.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.util
import io
import json
import os
import tarfile
import uuid
from pathlib import Path


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cli = module("sdsc_cli")
provenance = module("sdsc_provenance")
uploader = module("sdsc_provenance_upload")
SCHEMA = "quest-sdsc-release-replay-v1"
CONTROLS = (
    "tools/sdsc_release_replay.py",
    "tools/sdsc_cli.py",
    "tools/sdsc_remote.py",
    "tools/sdsc_provenance.py",
    "tools/sdsc_provenance_upload.py",
)
MAX_ARCHIVE = 64 * 1024 * 1024


class ReplayError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def json_bytes(value):
    return provenance.canonical(value) + b"\n"


def document(path):
    return json.loads(provenance.read_regular(path)[0])


def controls(root):
    return {name: provenance.sha(provenance.read_regular(root / name)[0]) for name in CONTROLS}


def matches(data, mode, record):
    return (len(data), provenance.sha(data), mode) == (record["size"], record["sha256"], record["mode"])


def write_exclusive(path, data, mode=0o644):
    path = provenance.safe_path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(mode)


def set_state(directory, phase, **fields):
    path = directory / "state.json"
    value = document(path) if path.exists() else {"schema": SCHEMA, "events": []}
    value.update(phase=phase, updated_at=cli.now(), **fields)
    value["events"].append({"phase": phase, "at": value["updated_at"]})
    cli.write_json(path, value)


REMOTE_SOURCE = r"""
import hashlib, io, json, os, pathlib, pwd, stat, sys, tarfile
request = json.loads(sys.argv[1])
assert pwd.getpwuid(os.getuid()).pw_name == "zgao12"
root = pathlib.Path("/home/zgao12/quest-runs/OPD")
def safe(path):
    assert path.is_absolute() and ".." not in path.parts
    assert not any(p.is_symlink() for p in (path, *path.parents))
    return path
def read(path, limit):
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        assert stat.S_ISREG(before.st_mode) and before.st_size <= limit
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        assert len(data) == before.st_size and before.st_mtime_ns == after.st_mtime_ns
    return data, 0o755 if before.st_mode & 0o111 else 0o644
run_id = request["run_id"]
assert isinstance(run_id, str) and run_id and all(c.isalnum() or c in "._-" for c in run_id)
assert len(run_id) <= 96
release = safe(root / "releases" / run_id)
raw, _ = read(release / "manifest.json", 4 * 1024 * 1024)
assert hashlib.sha256(raw).hexdigest() == request["manifest_sha256"]
manifest = json.loads(raw)
assert manifest["run_id"] == run_id
records = {row["path"]: row for row in manifest["files"]}
paths = request["paths"]
assert isinstance(paths, list) and paths == sorted(set(paths)) and set(paths) <= set(records)
assert sum(records[path]["size"] for path in paths) <= 48 * 1024 * 1024
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
    item = tarfile.TarInfo("manifest.json"); item.size = len(raw); item.mode = 0o644
    archive.addfile(item, io.BytesIO(raw))
    for name in paths:
        relative = pathlib.PurePosixPath(name)
        assert not relative.is_absolute() and ".." not in relative.parts and str(relative) == name
        record = records[name]
        data, mode = read(release / "source" / name, 2 * 1024 * 1024)
        assert len(data) == record["size"] and mode == record["mode"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        item = tarfile.TarInfo("source/" + name); item.size = len(data); item.mode = mode
        archive.addfile(item, io.BytesIO(data))
assert buffer.tell() <= 64 * 1024 * 1024
sys.stdout.buffer.write(buffer.getvalue())
"""


def decode_source_archive(raw, source_manifest, missing):
    require(len(raw) <= MAX_ARCHIVE, "Historical source archive exceeds bound")
    records = {row["path"]: row for row in missing}
    expected = {"manifest.json", *("source/" + path for path in records)}
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members = archive.getmembers()
        require(
            len(members) == len(expected)
            and {item.name for item in members} == expected
            and all(item.isfile() and not item.pax_headers for item in members),
            "Historical source archive inventory or regular-file contract differs",
        )
        manifest_member = archive.getmember("manifest.json")
        require(manifest_member.size <= provenance.MAX_FILE, "Historical manifest exceeds bound")
        require(
            archive.extractfile(manifest_member).read() == json_bytes(source_manifest),
            "Historical source manifest differs from trusted provenance",
        )
        for path, record in records.items():
            member = archive.getmember("source/" + path)
            require(member.size == record["size"], "Historical source size differs")
            data = archive.extractfile(member).read(record["size"] + 1)
            require(matches(data, member.mode, record), "Historical source bytes or mode differ")
            contents[path] = data
    return contents


def fetch_missing(root, source_manifest, missing):
    check = document(root / ".sdsc/check.json")
    require(check.get("connected") is True, "A successful existing-master connection check is required")
    request = {
        "run_id": source_manifest["run_id"],
        "manifest_sha256": provenance.sha(json_bytes(source_manifest)),
        "paths": sorted(row["path"] for row in missing),
    }
    result = cli.ssh_call(
        [check["control_python"], "-I", "-B", "-c", REMOTE_SOURCE, json.dumps(request)], timeout=150
    )
    require(result.returncode == 0, "Historical source read failed; no authentication retry was attempted")
    return decode_source_archive(result.stdout, source_manifest, missing)


def historical_contents(root, source_manifest, *, allow_remote_source):
    records, blobs = provenance.tree_records(root, source_manifest["git_head"])
    committed = {row["path"]: row for row in records}
    contents, origins, missing = {}, {}, []
    for record in source_manifest["files"]:
        name = record["path"]
        require(
            cli.exclude_reason(name) is None and record["size"] <= cli.MAX_FILE,
            "Ineligible historical source",
        )
        try:
            data, mode = provenance.read_regular(root / name, cli.MAX_FILE)
        except (OSError, ValueError):
            data, mode = b"", None
        if matches(data, mode, record):
            contents[name], origins[name] = data, "matching_current_file"
        elif committed.get(name) == record:
            contents[name], origins[name] = blobs[name], "genuine_committed_blob"
        else:
            missing.append(record)
    if missing:
        require(
            allow_remote_source,
            "Historical dirty source unavailable locally; explicit remote source read required",
        )
        contents.update(fetch_missing(root, source_manifest, missing))
        origins.update({record["path"]: "verified_remote_historical_file" for record in missing})
    require(
        set(contents) == {row["path"] for row in source_manifest["files"]}, "Incomplete historical source"
    )
    return contents, origins


def prepare(
    source_run_id, provenance_artifact, provenance_manifest_sha256, *, repo=None, allow_remote_source=False
):
    """Save a full dry-run without deploying; return the immutable plan and receipt paths."""
    root = provenance.safe_path(cli.ROOT if repo is None else repo)
    require(cli.IDENTIFIER.fullmatch(source_run_id), "Invalid source run ID")
    control_hashes = controls(root)
    current_head = provenance.git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    source_record = document(root / ".sdsc/runs" / (source_run_id + ".json"))
    require(source_record.get("state") == "deployed", "Replay requires a deployed source release")
    source_manifest = provenance.load_wrapper(root / ".sdsc/runs" / (source_run_id + ".json"))
    require(source_manifest["run_id"] == source_run_id, "Source run record identity differs")
    artifact = provenance.safe_path(provenance_artifact)
    provenance.verify(artifact, provenance_manifest_sha256, source_manifest["code_sha256"])
    source_proof, wrapper, bundle = provenance.load_artifact(
        artifact, provenance_manifest_sha256, source_manifest["code_sha256"]
    )
    require(wrapper == source_manifest, "Source release differs from trusted provenance wrapper")
    provenance.git(root, "merge-base", "--is-ancestor", source_manifest["git_head"], current_head)
    parent = provenance.safe_path(root / ".sdsc/replays")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = parent / ("replay-" + uuid.uuid4().hex)
    directory.mkdir(mode=0o700)
    set_state(directory, "preparing", source_run_id=source_run_id)
    try:
        contents, origins = historical_contents(
            root, source_manifest, allow_remote_source=allow_remote_source
        )
        require(controls(root) == control_hashes, "Replay controls changed during preparation")
        require(
            provenance.git(root, "rev-parse", "HEAD").decode().strip() == current_head,
            "Quest HEAD changed during preparation",
        )
        run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id += "-" + source_manifest["code_sha256"][:12] + "-" + uuid.uuid4().hex[:8]
        require(run_id != source_run_id, "Replay requires a fresh run identity")
        origin = {
            "schema": SCHEMA,
            "source_run_id": source_run_id,
            "source_wrapper_manifest_sha256": provenance.sha(json_bytes(source_manifest)),
            "source_provenance_manifest_sha256": provenance_manifest_sha256,
            "source_bundle_sha256": source_proof["bundle"]["sha256"],
            "source_git_head": source_manifest["git_head"],
            "preparation_git_head": current_head,
            "source_files_identical": True,
        }
        manifest = dict(
            copy.deepcopy(source_manifest), run_id=run_id, created_at=cli.now(), replay_origin=origin
        )
        stage = directory / "stage"
        stage.mkdir(mode=0o700)
        for record in manifest["files"]:
            write_exclusive(stage / "source" / record["path"], contents[record["path"]], record["mode"])
        write_exclusive(stage / "manifest.json", json_bytes(manifest))
        archive = cli.build_archive(manifest, contents)
        require(len(archive) <= MAX_ARCHIVE, "Replay source archive exceeds bound")
        write_exclusive(stage / "source.tar", archive)
        rebound = stage / "provenance"
        rebound.mkdir(mode=0o700)
        rebound_manifest = copy.deepcopy(source_proof)
        rebound_manifest.update(created_at=cli.now(), replay_origin=origin)
        rebound_manifest["wrapper"] = {
            "run_id": run_id,
            "code_sha256": manifest["code_sha256"],
            "manifest_sha256": provenance.sha(json_bytes(manifest)),
        }
        rebound_raw = json_bytes(rebound_manifest)
        rebound_sha = provenance.sha(rebound_raw)
        write_exclusive(rebound / "history.bundle", bundle)
        write_exclusive(rebound / "wrapper-manifest.json", json_bytes(manifest))
        write_exclusive(rebound / "manifest.json", rebound_raw)
        verified = provenance.verify(rebound, rebound_sha, manifest["code_sha256"])
        created = {
            "artifact": str(rebound),
            "manifest_sha256": rebound_sha,
            "bundle_sha256": verified["bundle_sha256"],
            "git_head": verified["git_head"],
            "wrapper_code_sha256": manifest["code_sha256"],
        }
        write_exclusive(directory / "provenance-created.json", json_bytes(created))
        run_record = {
            "state": "preview",
            "manifest": manifest,
            "excluded": [],
            "archive_bytes": len(archive),
            "remote_release": cli.REMOTE_ROOT + "/releases/" + run_id,
            "replay_plan": str(directory / "plan.json"),
        }
        run_path = root / ".sdsc/runs" / (run_id + ".json")
        write_exclusive(run_path, json_bytes(run_record))
        plan = {
            "schema": SCHEMA,
            "source_root": str(root),
            "run_id": run_id,
            "source_manifest": source_manifest,
            "source_provenance_artifact": str(artifact),
            "source_provenance_manifest_sha256": provenance_manifest_sha256,
            "manifest": manifest,
            "control_files": control_hashes,
            "replay_origin": origin,
            "source_origins": origins,
            "archive_bytes": len(archive),
            "archive_sha256": provenance.sha(archive),
            "stage": str(stage),
            "run_record": str(run_path),
            "remote_release": run_record["remote_release"],
            "provenance_created": str(directory / "provenance-created.json"),
            "provenance_uploaded": str(directory / "provenance-uploaded.json"),
            "provenance_dir": cli.REMOTE_ROOT + "/provenance/" + rebound_sha,
            "provenance_manifest_sha256": rebound_sha,
        }
        write_exclusive(directory / "plan.json", json_bytes(plan), 0o444)
        provenance.readonly_tree(stage)
        set_state(directory, "prepared", run_id=run_id, plan_sha256=provenance.sha(json_bytes(plan)))
        return dict(plan, plan_path=str(directory / "plan.json"), state_path=str(directory / "state.json"))
    except Exception as error:
        set_state(directory, "prepare_failed", error=type(error).__name__ + ": " + str(error)[:1000])
        raise


def validate_prepared(plan_path, root):
    path = provenance.safe_path(plan_path)
    require(
        path.parent.parent == root / ".sdsc/replays" and path.name == "plan.json",
        "Unexpected replay plan path",
    )
    plan = document(path)
    state = document(path.parent / "state.json")
    require(
        state.get("phase") == "prepared",
        "Replay is not prepared; reconcile prior state, never repeat deployment",
    )
    require(provenance.sha(json_bytes(plan)) == state["plan_sha256"], "Saved dry-run plan changed")
    require(plan["schema"] == SCHEMA and plan["source_root"] == str(root), "Replay project identity differs")
    require(controls(root) == plan["control_files"], "Replay controls changed after dry-run")
    stage = provenance.safe_path(plan["stage"])
    require(stage == path.parent / "stage", "Replay stage escaped its plan")
    manifest = provenance.load_wrapper(stage / "manifest.json")
    require(manifest == plan["manifest"], "Replay wrapper changed after dry-run")
    source = plan["source_manifest"]
    provenance.verify(
        plan["source_provenance_artifact"], plan["source_provenance_manifest_sha256"], source["code_sha256"]
    )
    source_proof, source_wrapper, source_bundle = provenance.load_artifact(
        plan["source_provenance_artifact"], plan["source_provenance_manifest_sha256"], source["code_sha256"]
    )
    require(source_wrapper == source, "Historical source wrapper changed")
    require(
        manifest["run_id"] == plan["run_id"] != source["run_id"]
        and manifest["files"] == source["files"]
        and manifest["git_head"] == source["git_head"]
        and manifest["code_sha256"] == source["code_sha256"],
        "Replay must have a fresh ID and identical source bytes and genuine HEAD",
    )
    contents = {}
    expected_paths = {record["path"] for record in manifest["files"]}
    observed = set()
    for folder, dirs, files in os.walk(stage / "source", followlinks=False):
        for name in [*dirs, *files]:
            item = provenance.safe_path(Path(folder) / name)
            if item.is_file():
                observed.add(item.relative_to(stage / "source").as_posix())
    require(observed == expected_paths, "Staged source inventory changed")
    for record in manifest["files"]:
        data, mode = provenance.read_regular(stage / "source" / record["path"], cli.MAX_FILE)
        require(matches(data, mode, record), "Staged source bytes changed")
        contents[record["path"]] = data
    archive, _ = provenance.read_regular(stage / "source.tar", MAX_ARCHIVE)
    require(
        len(archive) == plan["archive_bytes"]
        and provenance.sha(archive) == plan["archive_sha256"]
        and archive == cli.build_archive(manifest, contents),
        "Staged source archive changed",
    )
    created_path = path.parent / "provenance-created.json"
    require(plan["provenance_created"] == str(created_path), "Provenance receipt path escaped replay")
    created = document(created_path)
    require(created["artifact"] == str(stage / "provenance"), "Provenance artifact escaped replay")
    require(
        created["manifest_sha256"] == plan["provenance_manifest_sha256"], "Provenance receipt anchor changed"
    )
    verified = provenance.verify(created["artifact"], created["manifest_sha256"], manifest["code_sha256"])
    require(
        all(created[key] == verified[key] for key in ("git_head", "bundle_sha256", "wrapper_code_sha256")),
        "Provenance creation receipt changed",
    )
    rebound, new_wrapper, new_bundle = provenance.load_artifact(
        created["artifact"], created["manifest_sha256"], manifest["code_sha256"]
    )
    require(
        new_wrapper == manifest and new_bundle == source_bundle, "Replay changed original history or wrapper"
    )
    require(rebound["git_head"] == source_proof["git_head"], "Replay changed genuine scientific HEAD")
    run_path = root / ".sdsc/runs" / (plan["run_id"] + ".json")
    require(plan["run_record"] == str(run_path), "Run record escaped replay")
    run_record = document(run_path)
    require(
        run_record["state"] == "preview" and run_record["manifest"] == manifest,
        "Run already deployed or changed",
    )
    return plan, run_record, archive, created


def deploy(plan_path, *, repo=None):
    """Deploy one saved dry-run once; an interrupted phase requires read-only reconciliation."""
    root = provenance.safe_path(cli.ROOT if repo is None else repo)
    plan, run_record, archive, created = validate_prepared(plan_path, root)
    directory = Path(plan_path).absolute().parent
    # Persistent exclusive claim also blocks simultaneous deploy() calls that
    # both observed the prepared state. Never remove this claim, even on failure.
    (directory / "deploy.claim").mkdir(mode=0o700)
    stage = "release"
    try:
        check = document(root / ".sdsc/check.json")
        require(check.get("connected") is True, "A successful existing-master connection check is required")
        require(controls(root) == plan["control_files"], "Replay controls changed before upload")
        set_state(directory, "uploading_release")
        result = cli.remote(
            {"action": "upload", "run_id": plan["run_id"], "code_sha256": plan["manifest"]["code_sha256"]},
            archive,
            control_python=check["control_python"],
        )
        expected = {
            "run_id": plan["run_id"],
            "code_sha256": plan["manifest"]["code_sha256"],
            "release": cli.REMOTE_ROOT + "/releases/" + plan["run_id"],
            "total_bytes": plan["manifest"]["total_bytes"],
            "file_count": len(plan["manifest"]["files"]),
        }
        require(
            result.get("ok") is True and all(result.get(key) == value for key, value in expected.items()),
            "Release upload receipt differs",
        )
        cli.write_json(directory / "release-uploaded.json", result)
        cli.write_json(
            Path(plan["run_record"]),
            dict(run_record, state="deployed", deployment=result, deployed_at=cli.now()),
        )
        set_state(directory, "release_uploaded")
        stage = "provenance"
        require(controls(root) == plan["control_files"], "Replay controls changed before provenance upload")
        artifact = Path(created["artifact"])
        verified = provenance.verify(artifact, created["manifest_sha256"], created["wrapper_code_sha256"])
        upload_archive = uploader.archive_artifact(artifact)
        upload_plan = dict(
            verified,
            run_id=plan["run_id"],
            code_sha256=created["wrapper_code_sha256"],
            archive_sha256=provenance.sha(upload_archive),
            archive_bytes=len(upload_archive),
            destination=plan["provenance_dir"],
        )
        set_state(directory, "uploading_provenance")
        response = cli.ssh_call(
            [check["control_python"], "-I", "-B", "-c", uploader.REMOTE, json.dumps(upload_plan)],
            upload_archive,
            timeout=180,
        )
        require(response.returncode == 0, "Provenance upload returned no accepted receipt")
        uploaded = json.loads(response.stdout)
        require(
            uploaded.get("verified") is True
            and all(
                uploaded.get(key) == verified[key]
                for key in ("manifest_sha256", "wrapper_code_sha256", "git_head", "bundle_sha256")
            )
            and uploaded.get("run_id") == plan["run_id"]
            and uploaded.get("remote_artifact") == plan["provenance_dir"],
            "Provenance upload receipt differs",
        )
        uploaded_path = directory / "provenance-uploaded.json"
        write_exclusive(uploaded_path, json_bytes(uploaded))
        set_state(directory, "deployed")
        return {
            "run_id": plan["run_id"],
            "code_sha256": plan["manifest"]["code_sha256"],
            "science_git_head": created["git_head"],
            "remote_release": result["release"],
            "provenance_dir": uploaded["remote_artifact"],
            "provenance_manifest_sha256": uploaded["manifest_sha256"],
            "provenance_created": str(directory / "provenance-created.json"),
            "provenance_uploaded": str(uploaded_path),
            "state_path": str(directory / "state.json"),
        }
    except Exception as error:
        set_state(
            directory,
            "unknown_" + stage,
            error=type(error).__name__ + ": " + str(error)[:1000],
            reconciliation_required=True,
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("prepare")
    preview.add_argument("--source-run-id", required=True)
    preview.add_argument("--provenance-artifact", type=Path, required=True)
    preview.add_argument("--provenance-manifest-sha256", required=True)
    preview.add_argument("--allow-remote-source", action="store_true")
    preview.add_argument("--dry-run", action="store_true", required=True)
    upload = commands.add_parser("deploy")
    upload.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.source_run_id,
            args.provenance_artifact,
            args.provenance_manifest_sha256,
            allow_remote_source=args.allow_remote_source,
        )
        for record in result["manifest"]["files"]:
            print(f"{record['size']:>9}  {record['path']}")
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "run_id",
                        "plan_path",
                        "state_path",
                        "archive_bytes",
                        "provenance_created",
                        "provenance_uploaded",
                    )
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(deploy(args.plan), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
