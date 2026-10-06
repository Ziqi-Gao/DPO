"""One permanent filesystem fence for the specified unknown student intent.

No scheduler call, cancellation, retry, receipt fabrication, or scientific
acceptance occurs here.  The unchanged old node exclusively creates result_dir
before runtime/CUDA/model work.  Winning that same mkdir and retaining its inode
fences that execution path; it does not preclude a late scheduler allocation.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import stat
from pathlib import Path

OLD_INTENT = "19665dfbd26d7b0024a46d9b7b811ce4"
OLD_PLAN_SHA = "b88f012fb8c30a9527d6300680dbabe29a944fdff64b29339a28f0c34764af5f"
OLD_NODE_SHA = "4d9b45d6a5ea6cba717982d32aebc434121c16735a768d8f196e5c050b1ad31f"
CONTROL = Path("/home/zgao12/quest-runs/OPD")
PROJECT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
EXPECTED_UID = 543540
MAX_FILE = 4 * 1024**2
MAX_DOCUMENT = 1024**2
MARKER_NAME = "execution-fence.json"
FILESYSTEMS = {
    "control": {
        "fstype": "nfs",
        "source": "10.22.100.113:/pool3/home/zgao12",
        "filesystem_path": "/quest-runs/OPD",
    },
    "project": {
        "fstype": "lustre",
        "source": "10.22.101.123@o2ib:10.22.101.124@o2ib:/expanse/projects",
        "filesystem_path": "/nwu181/zgao12/OPD",
    },
}
FLAGS = (
    "student_accepted",
    "g0_passed",
    "pilot_passed",
    "factorial_ready",
    "formal_initial_accepted",
    "execution_class_certified",
)
# Actual retained original remote bytes, including the unknown record.  These
# pins are evidence for this one intent, not caller-configurable policy.
SIDEFILES = {
    "admission.json": (980, "fe7f02450e4305e7a90d6510e665664612fd14074d26cf16cc65079b9c3f547b"),
    "unknown.json": (1007, "dc336ea5f2f87662191770ff21abcb0e2b1b12fac3de637cd2796d4b202892ab"),
    "submission-started.json": (931, "d922f9bee7be2a4dac5e9d9e04aea3537d0f1871df62912b1fc0cd7a1e4687e6"),
    "job.sh": (652, "cdf0247f03d630ecd66e6428abb6717d7ab15563572d47b9fa1a58061f9922ae"),
}
OLD_CLAIMS = {
    "claim": (169, "1302768110285faf16f3304ca4564c2a0d2ecfb71df9e24fc3f29d1a9ed3a0c9"),
    "scientific_claim": (677, "b067c4d79d7bc51ffa64b11a4ccb54bc447ff341011792b1205740c96d32c920"),
}


class FenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FenceError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _flags():
    return dict.fromkeys(FLAGS, False)


def _helper_sha():
    return sha(Path(__file__).read_bytes())


def _time():
    return dt.datetime.now(dt.UTC).isoformat()


def _timestamp(value):
    require(
        isinstance(value, str) and dt.datetime.fromisoformat(value).utcoffset() == dt.timedelta(0),
        "invalid UTC timestamp",
    )


def build_request(old_plan):
    """Pure deterministic request for the one pinned original plan only."""
    raw = canonical(old_plan)
    require(sha(raw) == OLD_PLAN_SHA and len(raw) <= MAX_DOCUMENT, "not the pinned old plan")
    submission = CONTROL / "student-name-invariant-v1-submissions" / OLD_INTENT
    result = PROJECT / "student-name-invariant-v1" / OLD_INTENT
    require(old_plan["intent_id"] == OLD_INTENT and old_plan["mode"] == "preflight", "old scope differs")
    require(
        old_plan["submission_dir"] == str(submission) and old_plan["result_dir"] == str(result),
        "old path differs",
    )
    require(
        old_plan["control_sha256"]["tools/sdsc_student_name_invariant_job.py"] == OLD_NODE_SHA,
        "old node differs",
    )
    release = Path(old_plan["release"])
    require(release.parent == CONTROL / "releases", "old release namespace differs")
    files = {str(submission / "plan.json"): dict(size=len(raw), sha256=sha(raw))}
    for name, (size, digest) in SIDEFILES.items():
        files[str(submission / name)] = dict(size=size, sha256=digest)
    for key, (size, digest) in OLD_CLAIMS.items():
        files[old_plan[key]] = dict(size=size, sha256=digest)
    for name, digest in old_plan["control_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "invalid control name")
        files[str(release / "source" / name)] = dict(size=None, sha256=digest)
    return dict(
        schema="opd-student-execution-fence-request-v1",
        old_plan=old_plan,
        old_intent=OLD_INTENT,
        old_plan_sha256=OLD_PLAN_SHA,
        old_node_sha256=OLD_NODE_SHA,
        expected_uid=EXPECTED_UID,
        expected_filesystems=FILESYSTEMS,
        old_files=files,
        result_dir=str(result),
        claim_path=str(CONTROL / "student-execution-fence-claims" / (OLD_INTENT + ".json")),
        proof_path=str(CONTROL / "student-execution-fences" / OLD_INTENT / "proof.json"),
        marker_path=str(result / MARKER_NAME),
        old_submission_outcome="unknown",
        **_flags(),
    )


def _request(value):
    require(
        isinstance(value, dict) and canonical(value) == canonical(build_request(value.get("old_plan"))),
        "fence request differs",
    )
    require(os.getuid() == EXPECTED_UID, "wrong execution UID")
    return value


def _identity(info):
    return dict(device=info.st_dev, inode=info.st_ino, uid=info.st_uid, mode=stat.S_IMODE(info.st_mode))


def _stamp(info):
    return dict(
        _identity(info),
        size=info.st_size,
        nlink=info.st_nlink,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
    )


def _owned(info, directory):
    require((stat.S_ISDIR if directory else stat.S_ISREG)(info.st_mode), "wrong filesystem object type")
    mode = stat.S_IMODE(info.st_mode)
    require(info.st_uid == EXPECTED_UID and not mode & 0o022, "wrong owner or group/other writable object")
    require(not mode & (0o5000 if directory else 0o7000), "unsafe special mode")
    if not directory:
        require(info.st_nlink == 1, "hardlinked evidence is forbidden")


def _root(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "noncanonical path")
    for root in (CONTROL, PROJECT):
        if path == root or root in path.parents:
            return path, root
    raise FenceError("path outside fixed roots")


def _expected_filesystem(path):
    path, root = _root(path)
    base = FILESYSTEMS["control" if root == CONTROL else "project"]
    return dict(base, filesystem_path=str(Path(base["filesystem_path"]) / path.relative_to(root)))


def _unescape(value):
    require(re.search(r"\\(?!040|011|012|134)", value) is None, "unknown mountinfo escape")
    return re.sub(r"\\(040|011|012|134)", lambda match: chr(int(match[1], 8)), value)


def _observe_filesystem(fd, path):
    """Read mountinfo locally; retain only the one mount backing this actual FD."""
    path = Path(path)
    info = os.fstat(fd)
    require(os.readlink(f"/proc/self/fd/{fd}") == str(path), "opened path changed or was deleted")
    with open("/proc/self/mountinfo", "rb") as stream:
        raw = stream.read(MAX_FILE + 1)
    require(len(raw) <= MAX_FILE, "mountinfo exceeds bound")
    device = f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}"
    candidates = []
    for line in raw.decode().splitlines():
        left, separator, right = line.partition(" - ")
        require(bool(separator), "malformed mountinfo")
        fields, suffix = left.split(), right.split()
        require(len(fields) >= 6 and len(suffix) >= 3, "malformed mountinfo row")
        mountpoint = Path(_unescape(fields[4]))
        if fields[2] != device or (path != mountpoint and mountpoint not in path.parents):
            continue
        mount = dict(
            mount_id=int(fields[0]),
            parent_id=int(fields[1]),
            device_major_minor=device,
            root=_unescape(fields[3]),
            mountpoint=str(mountpoint),
            options=fields[5],
            fstype=suffix[0],
            source=_unescape(suffix[1]),
            super_options=suffix[2],
        )
        portable = dict(
            fstype=mount["fstype"],
            source=mount["source"],
            filesystem_path=str(Path(mount["root"]) / path.relative_to(mountpoint)),
        )
        candidates.append((len(mountpoint.parts), dict(portable=portable, mount=mount)))
    require(bool(candidates), "actual FD has no matching mountinfo entry")
    depth = max(row[0] for row in candidates)
    matches = [row[1] for row in candidates if row[0] == depth]
    require(len(matches) == 1, "ambiguous backing mount")
    return matches[0]


def _validate_filesystem(value, path, device):
    require(isinstance(value, dict) and set(value) == {"portable", "mount"}, "filesystem observation differs")
    require(
        canonical(value["portable"]) == canonical(_expected_filesystem(path)), "unreviewed backing filesystem"
    )
    mount = value["mount"]
    require(
        isinstance(mount, dict)
        and set(mount)
        == {
            "mount_id",
            "parent_id",
            "device_major_minor",
            "root",
            "mountpoint",
            "options",
            "fstype",
            "source",
            "super_options",
        },
        "mount observation differs",
    )
    require(
        type(mount["mount_id"]) is int
        and type(mount["parent_id"]) is int
        and mount["device_major_minor"] == f"{os.major(device)}:{os.minor(device)}"
        and "rw" in mount["options"].split(",")
        and "ro" not in mount["options"].split(",")
        and "ro" not in mount["super_options"].split(","),
        "mount/device/writeability differs",
    )
    path, mountpoint, fsroot = Path(path), Path(mount["mountpoint"]), Path(mount["root"])
    require(
        mountpoint.is_absolute()
        and fsroot.is_absolute()
        and ".." not in mountpoint.parts
        and ".." not in fsroot.parts
        and (path == mountpoint or mountpoint in path.parents),
        "invalid mount mapping",
    )
    observed = dict(
        fstype=mount["fstype"],
        source=mount["source"],
        filesystem_path=str(fsroot / path.relative_to(mountpoint)),
    )
    require(
        canonical(observed) == canonical(value["portable"]), "mount mapping contradicts filesystem identity"
    )


def _filesystem(fd, path):
    value = _observe_filesystem(fd, path)
    _validate_filesystem(value, path, os.fstat(fd).st_dev)
    return value


def _portable_record(row):
    # Never erase installation observations. Only a verified backing filesystem
    # permits ignoring the client-local device number in cross-host comparisons.
    _validate_filesystem(row["filesystem"], row["path"], row["stamp"]["device"])
    return dict(
        row,
        stamp={k: v for k, v in row["stamp"].items() if k != "device"},
        filesystem=row["filesystem"]["portable"],
    )


def _portable_directory(path, identity, filesystem):
    _validate_filesystem(filesystem, path, identity["device"])
    return dict(
        identity={k: v for k, v in identity.items() if k != "device"}, filesystem=filesystem["portable"]
    )


def _parent(path, *, create=False):
    """Open every directory without following symlinks; create only below roots."""
    path, root = _root(path)
    require(path != root, "root cannot be an evidence file")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    current = Path("/")
    try:
        for part in path.parent.parts[1:]:
            current /= part
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            except FileNotFoundError:
                require(create and root in current.parents, "required parent absent")
                os.mkdir(part, mode=0o700, dir_fd=fd)
                os.fsync(fd)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            os.close(fd)
            fd = child
            if current == root or root in current.parents:
                _owned(os.fstat(fd), True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _exists(path):
    try:
        fd = _parent(path)
    except (FileNotFoundError, FenceError) as error:
        if isinstance(error, FileNotFoundError) or str(error) == "required parent absent":
            return False
        raise
    try:
        try:
            os.stat(Path(path).name, dir_fd=fd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False
    finally:
        os.close(fd)


def _read(path, limit=MAX_FILE):
    parent = _parent(path)
    fd = None
    try:
        fd = os.open(
            Path(path).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent
        )
        before = os.fstat(fd)
        _owned(before, False)
        filesystem = _filesystem(fd, path)
        require(0 < before.st_size <= limit, "evidence size outside bound")
        chunks, total = [], 0
        while chunk := os.read(fd, min(65536, limit + 1 - total)):
            total += len(chunk)
            require(total <= limit, "evidence grew beyond bound")
            chunks.append(chunk)
        after = os.fstat(fd)
        current = os.stat(Path(path).name, dir_fd=parent, follow_symlinks=False)
        require(_stamp(before) == _stamp(after) == _stamp(current), "evidence changed during read")
        raw = b"".join(chunks)
        require(len(raw) == before.st_size, "short evidence read")
        require(_filesystem(fd, path) == filesystem, "backing mount changed during read")
        return raw, dict(
            path=str(path), size=len(raw), sha256=sha(raw), stamp=_stamp(after), filesystem=filesystem
        )
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def _document(path):
    raw, record = _read(path, MAX_DOCUMENT)
    value = json.loads(raw)
    require(canonical(value) == raw, "document must be canonical JSON without LF")
    return value, record


def _write_once(path, value):
    raw = canonical(value)
    require(len(raw) <= MAX_DOCUMENT, "new evidence too large")
    parent = _parent(path, create=True)
    fd = None
    try:
        fd = os.open(
            Path(path).name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            require(written > 0, "short evidence write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.fsync(parent)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)
    observed, record = _read(path, MAX_DOCUMENT)
    require(observed == raw, "new evidence readback differs")
    return dict(record, document=value)


def _snapshot(request):
    rows = {}
    for path, expected in request["old_files"].items():
        raw, record = _read(path)
        require(
            record["sha256"] == expected["sha256"]
            and (expected["size"] is None or len(raw) == expected["size"]),
            "old evidence differs: " + path,
        )
        rows[path] = record
    submission = Path(request["old_plan"]["submission_dir"])
    for name in ("receipt.json", "live-binding.json"):
        require(not _exists(submission / name), "old acknowledgement now exists; stop fencing")
    return rows


def _roots():
    values = {}
    for root in (CONTROL, PROJECT):
        fd = _parent(root / "unused-readonly-name")
        try:
            values[str(root)] = dict(identity=_identity(os.fstat(fd)), filesystem=_filesystem(fd, root))
        finally:
            os.close(fd)
    return values


def dry_run(request):
    """Read only. Absence is a prerequisite, never proof of installation."""
    request = _request(request)
    roots = _roots()
    rows = _snapshot(request)
    for key in ("claim_path", "proof_path", "result_dir"):
        require(not _exists(request[key]), "fence/old result already exists; never adopt or retry")
    return dict(
        schema="opd-student-execution-fence-dry-run-v1",
        request_sha256=sha(canonical(request)),
        helper_sha256=_helper_sha(),
        old_files=rows,
        roots=roots,
        checked_at=_time(),
        ready=True,
        **_flags(),
    )


def _directory(path, *, filesystem=False):
    parent = _parent(path)
    fd = None
    try:
        fd = os.open(
            Path(path).name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent
        )
        info = os.fstat(fd)
        _owned(info, True)
        observed = _filesystem(fd, path)
        result = (_identity(info), sorted(os.listdir(fd)))
        return (*result, observed) if filesystem else result
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def install(request, *, authorize=False, dry_run=None):
    """One permanent claim then one exclusive mkdir. Never repairs partial state."""
    request = _request(request)
    require(authorize is True, "explicit fence authorization required")
    require(isinstance(dry_run, dict), "matching dry-run required")
    _timestamp(dry_run.get("checked_at"))
    fresh = globals()["dry_run"](request)
    require(
        set(fresh) == set(dry_run)
        and all(canonical(fresh[k]) == canonical(dry_run[k]) for k in fresh if k != "checked_at"),
        "dry-run changed",
    )
    claim_doc = dict(
        schema="opd-student-execution-fence-claim-v1",
        request_sha256=sha(canonical(request)),
        helper_sha256=_helper_sha(),
        at=_time(),
        old_intent=OLD_INTENT,
        no_retry=True,
        old_submission_outcome="unknown",
        **_flags(),
    )
    claim = _write_once(request["claim_path"], claim_doc)
    # This is the irrevocable gate. A lost race leaves the consumed claim intact.
    parent = _parent(request["result_dir"], create=True)
    created = None
    try:
        os.mkdir(Path(request["result_dir"]).name, mode=0o700, dir_fd=parent)
        created = os.open(
            Path(request["result_dir"]).name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent,
        )
        info = os.fstat(created)
        _owned(info, True)
        identity = _identity(info)
        directory_filesystem = _filesystem(created, request["result_dir"])
        require(not os.listdir(created), "newly created fence directory is not empty")
        os.fsync(created)
        os.fsync(parent)
    finally:
        if created is not None:
            os.close(created)
        os.close(parent)
    require(
        _directory(request["result_dir"], filesystem=True) == (identity, [], directory_filesystem),
        "new fence path no longer names the mkdir winner",
    )
    marker_doc = dict(
        schema="opd-student-execution-fence-marker-v1",
        request_sha256=sha(canonical(request)),
        helper_sha256=_helper_sha(),
        claim_sha256=claim["sha256"],
        old_intent=OLD_INTENT,
        old_plan_sha256=OLD_PLAN_SHA,
        directory_identity=identity,
        directory_filesystem=directory_filesystem,
        old_submission_outcome="unknown",
        prevents_old_model_work=True,
        scheduler_cancellation=False,
        allocation_absence_proven=False,
        **_flags(),
    )
    marker = _write_once(request["marker_path"], marker_doc)
    require(_directory(request["result_dir"]) == (identity, [MARKER_NAME]), "fence directory changed")
    require(_snapshot(request) == fresh["old_files"], "old evidence changed during fence")
    require(_roots() == fresh["roots"], "namespace roots changed during fence")
    proof = dict(
        schema="opd-student-execution-fence-proof-v1",
        request=request,
        request_sha256=sha(canonical(request)),
        helper_sha256=_helper_sha(),
        completed_at=_time(),
        claim=claim,
        marker=marker,
        directory_identity=identity,
        directory_filesystem=directory_filesystem,
        installation_host=os.uname().nodename,
        old_files=fresh["old_files"],
        roots=fresh["roots"],
        installed=True,
        old_submission_outcome="unknown",
        scheduler_cancellation=False,
        allocation_absence_proven=False,
        **_flags(),
    )
    _write_once(request["proof_path"], proof)
    return verify(proof)


def validate_proof(proof, *, request=None):
    """Pure binding checks; filesystem validity additionally requires verify()."""
    require(isinstance(proof, dict), "missing fence proof")
    bound = proof.get("request")
    require(
        isinstance(bound, dict) and canonical(bound) == canonical(build_request(bound.get("old_plan"))),
        "proof request differs",
    )
    require(request is None or canonical(request) == canonical(bound), "unexpected proof request")
    expected_keys = {
        "schema",
        "request",
        "request_sha256",
        "helper_sha256",
        "completed_at",
        "claim",
        "marker",
        "directory_identity",
        "directory_filesystem",
        "installation_host",
        "old_files",
        "roots",
        "installed",
        "old_submission_outcome",
        "scheduler_cancellation",
        "allocation_absence_proven",
        *FLAGS,
    }
    require(
        set(proof) == expected_keys and proof["schema"] == "opd-student-execution-fence-proof-v1",
        "proof schema differs",
    )
    require(
        proof["request_sha256"] == sha(canonical(bound)) and proof["helper_sha256"] == _helper_sha(),
        "proof binding differs",
    )
    _timestamp(proof["completed_at"])
    require(
        isinstance(proof["installation_host"], str) and 0 < len(proof["installation_host"]) < 256,
        "invalid installation host",
    )
    require(
        proof["installed"] is True
        and proof["old_submission_outcome"] == "unknown"
        and proof["scheduler_cancellation"] is False
        and proof["allocation_absence_proven"] is False
        and all(proof[k] is False for k in FLAGS),
        "proof scope differs",
    )
    identity = proof["directory_identity"]
    require(
        set(identity) == {"device", "inode", "uid", "mode"}
        and all(type(v) is int for v in identity.values())
        and identity["device"] >= 0
        and identity["inode"] > 0
        and identity["uid"] == EXPECTED_UID
        and identity["mode"] in (0o700, 0o2700),
        "fence directory identity differs",
    )
    require(
        isinstance(proof["roots"], dict) and set(proof["roots"]) == {str(CONTROL), str(PROJECT)},
        "namespace root inventory differs",
    )
    _validate_filesystem(proof["directory_filesystem"], bound["result_dir"], identity["device"])
    for root_path, root_row in proof["roots"].items():
        require(
            isinstance(root_row, dict) and set(root_row) == {"identity", "filesystem"},
            "namespace root row differs",
        )
        root_identity = root_row["identity"]
        require(
            isinstance(root_identity, dict)
            and set(root_identity) == {"device", "inode", "uid", "mode"}
            and all(type(v) is int for v in root_identity.values())
            and root_identity["device"] >= 0
            and root_identity["inode"] > 0
            and root_identity["uid"] == EXPECTED_UID
            and 0 <= root_identity["mode"] <= 0o7777
            and not root_identity["mode"] & 0o5022,
            "namespace root identity malformed",
        )
        _validate_filesystem(root_row["filesystem"], root_path, root_identity["device"])
    for key, path_key in (("claim", "claim_path"), ("marker", "marker_path")):
        row = proof[key]
        require(
            set(row) == {"path", "size", "sha256", "stamp", "filesystem", "document"},
            "proof file row differs",
        )
        _record(row, document=True)
        raw = canonical(row["document"])
        require(
            row["path"] == bound[path_key]
            and type(row["size"]) is int
            and row["size"] == len(raw)
            and row["sha256"] == sha(raw),
            "proof file content binding differs",
        )
        doc = row["document"]
        require(
            doc["request_sha256"] == proof["request_sha256"]
            and doc["helper_sha256"] == proof["helper_sha256"]
            and doc["old_intent"] == OLD_INTENT
            and doc["old_submission_outcome"] == "unknown"
            and all(doc[k] is False for k in FLAGS),
            "proof file scope differs",
        )
    claim, marker = proof["claim"]["document"], proof["marker"]["document"]
    _timestamp(claim["at"])
    require(
        set(claim)
        == {
            "schema",
            "request_sha256",
            "helper_sha256",
            "at",
            "old_intent",
            "no_retry",
            "old_submission_outcome",
            *FLAGS,
        },
        "claim keys differ",
    )
    require(
        set(marker)
        == {
            "schema",
            "request_sha256",
            "helper_sha256",
            "claim_sha256",
            "old_intent",
            "old_plan_sha256",
            "directory_identity",
            "directory_filesystem",
            "old_submission_outcome",
            "prevents_old_model_work",
            "scheduler_cancellation",
            "allocation_absence_proven",
            *FLAGS,
        },
        "marker keys differ",
    )
    require(
        claim.get("schema") == "opd-student-execution-fence-claim-v1" and claim.get("no_retry") is True,
        "claim differs",
    )
    require(
        marker.get("schema") == "opd-student-execution-fence-marker-v1"
        and marker.get("old_plan_sha256") == OLD_PLAN_SHA
        and marker.get("claim_sha256") == proof["claim"]["sha256"]
        and marker.get("directory_identity") == identity
        and marker.get("directory_filesystem") == proof["directory_filesystem"]
        and marker.get("prevents_old_model_work") is True
        and marker.get("scheduler_cancellation") is False
        and marker.get("allocation_absence_proven") is False,
        "marker differs",
    )
    require(set(proof["old_files"]) == set(bound["old_files"]), "old evidence inventory differs")
    for path, row in proof["old_files"].items():
        _record(row)
        expected = bound["old_files"][path]
        require(
            row["path"] == path
            and row["sha256"] == expected["sha256"]
            and (expected["size"] is None or row["size"] == expected["size"]),
            "old evidence binding differs",
        )
    return proof


def _record(row, *, document=False):
    keys = {"path", "size", "sha256", "stamp", "filesystem"} | ({"document"} if document else set())
    require(
        isinstance(row, dict)
        and set(row) == keys
        and type(row["size"]) is int
        and 0 < row["size"] <= MAX_FILE,
        "invalid file record",
    )
    stamp = row["stamp"]
    require(
        isinstance(stamp, dict)
        and set(stamp) == {"device", "inode", "uid", "mode", "size", "nlink", "mtime_ns", "ctime_ns"}
        and all(type(v) is int for v in stamp.values())
        and stamp["device"] >= 0
        and stamp["inode"] > 0
        and stamp["uid"] == EXPECTED_UID
        and stamp["size"] == row["size"]
        and stamp["nlink"] == 1
        and 0 <= stamp["mode"] <= 0o7777
        and not stamp["mode"] & 0o7022
        and stamp["mtime_ns"] >= 0
        and stamp["ctime_ns"] >= 0,
        "invalid file stamp",
    )
    _validate_filesystem(row["filesystem"], row["path"], stamp["device"])


def verify(request_or_proof, *, return_observation=False):
    """Read only; require exact durable proof, original bytes and fenced inode."""
    is_proof = request_or_proof.get("schema") == "opd-student-execution-fence-proof-v1"
    require(type(return_observation) is bool, "invalid observation request")
    request = _request(request_or_proof["request"] if is_proof else request_or_proof)
    proof, _ = _document(request["proof_path"])
    if is_proof:
        require(
            canonical(proof) == canonical(request_or_proof), "durable proof differs from expected document"
        )
    validate_proof(proof, request=request)
    current_files, roots = _snapshot(request), _roots()
    require(
        canonical({p: _portable_record(r) for p, r in current_files.items()})
        == canonical({p: _portable_record(r) for p, r in proof["old_files"].items()}),
        "old files no longer match installation",
    )
    require(
        canonical({p: _portable_directory(p, r["identity"], r["filesystem"]) for p, r in roots.items()})
        == canonical(
            {p: _portable_directory(p, r["identity"], r["filesystem"]) for p, r in proof["roots"].items()}
        ),
        "namespace root identity differs",
    )
    current = {}
    for key, path_key in (("claim", "claim_path"), ("marker", "marker_path")):
        doc, record = _document(request[path_key])
        current[key] = dict(record, document=doc)
        require(
            canonical(_portable_record(current[key])) == canonical(_portable_record(proof[key])),
            "durable claim/marker identity differs",
        )
    identity, names, filesystem = _directory(request["result_dir"], filesystem=True)
    require(
        names == [MARKER_NAME]
        and canonical(_portable_directory(request["result_dir"], identity, filesystem))
        == canonical(
            _portable_directory(
                request["result_dir"], proof["directory_identity"], proof["directory_filesystem"]
            )
        ),
        "fenced inode/inventory differs",
    )
    again, _ = _document(request["proof_path"])
    require(canonical(again) == canonical(proof), "proof changed during verification")
    if return_observation:
        return dict(
            proof=proof,
            verification=dict(
                schema="opd-student-execution-fence-verification-v1",
                checked_at=_time(),
                current_host=os.uname().nodename,
                old_files=current_files,
                roots=roots,
                directory_identity=identity,
                directory_filesystem=filesystem,
                **current,
            ),
            **_flags(),
        )
    return proof


def reconcile(request):
    """Read-only lost-ack recovery. Partial/losing attempts are never adopted."""
    request = _request(request)
    if _exists(request["proof_path"]):
        try:
            proof = verify(request)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(
                state="unknown", installed=False, error=type(error).__name__ + ": " + str(error), **_flags()
            )
        return dict(state="fenced", installed=True, proof=proof, **_flags())
    started = _exists(request["claim_path"]) or _exists(request["result_dir"])
    return dict(
        state="unknown" if started else "not_started",
        installed=False,
        old_submission_outcome="unknown",
        retry_authorized=False,
        **_flags(),
    )
