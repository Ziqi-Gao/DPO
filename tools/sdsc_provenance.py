"""Prepare/verify genuine Git provenance; never upload, submit, or edit the source repo.

The bundle contains only HEAD's reachable, already committed history. It is not a
credential scrubber for committed history. A locally recorded public ref bounds
the default export; this is not an online assertion about GitHub. An explicit
local-successor mode permits bounded linear unpublished history ending in the
explicit implementation/acceptance pair. It audits every newly exported tree,
records their actual lineage and never moves the public ref. Scientific
acceptance is still checked by the scientific validators, not inferred from
these Git labels.

Restore creates a separate, read-only committed checkout. Dirty tools/docs remain
in the separately bound wrapper snapshot and must never overlay that checkout.
The returned manifest SHA must be carried by a trusted submission record; a hash
read from the same untrusted artifact directory is not an authentication anchor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import uuid
from pathlib import Path, PurePosixPath

SCIENCE_ROOTS = ("src", "configs", "prereg", "scripts/server_scheduler", "deployments", "third_party")
SCIENCE_FILES = ("pyproject.toml", "environment.yml", "requirements-cpu.lock")
SCHEMA = "quest-sdsc-git-provenance-v1"
MAX_FILE = 4 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
# Bound genuine local history without moving public refs or dropping earlier
# review commits. Every exported tree and the bundle retain their byte limits.
MAX_LOCAL_COMMITS = 128
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PRIVATE_DIRS = {".git", ".opd-git", ".codex", ".ssh", ".sdsc", ".aws", ".azure", "credentials", "secrets"}


class ProvenanceError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in [path, *path.parents]):
        raise ProvenanceError("Symlink or parent traversal in path")
    return path


def relative_path(value):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ProvenanceError("Invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or ".." in path.parts or "\\" in value:
        raise ProvenanceError("Unsafe relative path")
    parts = [p.lower() for p in path.parts]
    name = parts[-1]
    if any(p in PRIVATE_DIRS for p in parts) or (
        name != ".env.example"
        and (
            name.startswith(".env")
            or name == "auth.json"
            or name.startswith(("id_rsa", "id_ed25519"))
            or name.endswith((".pem", ".key", ".p12", ".pfx"))
            or re.search(r"(^|[._-])(credentials?|secrets?|passwords?)([._-]|$)", name)
        )
    ):
        raise ProvenanceError("Private metadata or credential path is forbidden")
    return value


def read_regular(path, limit=MAX_FILE):
    path = safe_path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ProvenanceError("Expected a bounded regular file")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ProvenanceError("Expected a bounded regular file")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) > limit or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ProvenanceError("File changed while reading or exceeds size limit")
    return data, 0o755 if before.st_mode & 0o111 else 0o644


def git_env():
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_OPTIONAL_LOCKS="0", GIT_NO_LAZY_FETCH="1"
    )
    return env


def git(root, *args, data=None):
    root = safe_path(root)
    metadata = root / (".opd-git" if (root / ".opd-git").is_dir() else ".git")
    safe_path(metadata)
    argv = [
        "git",
        "--no-replace-objects",
        f"--git-dir={metadata}",
        f"--work-tree={root}",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.autocrlf=false",
        "-c",
        "core.attributesFile=/dev/null",
        "-c",
        "protocol.allow=never",
        *args,
    ]
    result = subprocess.run(argv, input=data, capture_output=True, env=git_env(), timeout=120)
    if result.returncode:
        # Git diagnostics can contain config values or committed content. Do not echo them.
        raise ProvenanceError("Git provenance command failed: " + args[0])
    return result.stdout


def init_repo(root):
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "init", "--quiet", "--template=", str(root)],
        capture_output=True,
        env=git_env(),
        timeout=30,
    )
    if result.returncode:
        raise ProvenanceError("Cannot initialize isolated Git metadata")


def is_science(path):
    return path in SCIENCE_FILES or any(path == p or path.startswith(p + "/") for p in SCIENCE_ROOTS)


def tree_records(root, head):
    """Read raw committed blobs, never checkout filters, textconv, hooks, or smudge."""
    entries = []
    for row in git(root, "ls-tree", "-r", "-l", "-z", head).split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, kind, oid, size = metadata.split()
        path = relative_path(raw_path.decode("utf-8"))
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ProvenanceError("Committed symlinks and submodules are unsupported")
        if int(size) > MAX_FILE:
            raise ProvenanceError("Committed source file exceeds size limit")
        entries.append((path, oid, int(size), 0o755 if mode == b"100755" else 0o644))
    if sum(item[2] for item in entries) > MAX_TOTAL or not entries:
        raise ProvenanceError("Committed tree is empty or exceeds source budget")
    batch = git(root, "cat-file", "--batch", data=b"".join(item[1] + b"\n" for item in entries))
    offset, records, contents = 0, [], {}
    for path, oid, size, mode in entries:
        end = batch.index(b"\n", offset)
        if batch[offset:end] != oid + b" blob " + str(size).encode():
            raise ProvenanceError("Unexpected Git blob response")
        content = batch[end + 1 : end + 1 + size]
        if len(content) != size or batch[end + 1 + size : end + 2 + size] != b"\n":
            raise ProvenanceError("Truncated Git blob")
        offset = end + 2 + size
        records.append({"path": path, "size": size, "sha256": sha(content), "mode": mode})
        contents[path] = content
    if offset != len(batch):
        raise ProvenanceError("Trailing Git blob data")
    return sorted(records, key=lambda item: item["path"]), contents


def verify_science_workspace(root, head, records):
    expected = {item["path"]: item for item in records if is_science(item["path"])}
    if not expected:
        raise ProvenanceError("No tracked scientific files")
    git(root, "diff", "--cached", "--quiet", "--no-ext-diff", head, "--", *SCIENCE_ROOTS, *SCIENCE_FILES)
    for path, record in expected.items():
        data, mode = read_regular(root / path)
        if (len(data), sha(data), mode) != (record["size"], record["sha256"], record["mode"]):
            raise ProvenanceError("Uncommitted scientific file: " + path)
    # Inspect ignored paths too; .gitignore must not hide a Python/source shadow.
    # Bytecode caches are never reconstructed or added to the execution sys.path.
    for scope in SCIENCE_ROOTS:
        base = safe_path(root / scope)
        if not base.exists():
            continue
        for folder, dirs, files in os.walk(base, followlinks=False):
            for name in [*dirs, *files]:
                path = safe_path(Path(folder) / name)
                relative = path.relative_to(root).as_posix()
                if path.is_dir():
                    continue
                if "__pycache__" in path.parts and path.suffix == ".pyc":
                    continue
                if relative not in expected:
                    raise ProvenanceError("Untracked scientific file: " + relative)
    for name in SCIENCE_FILES:
        if safe_path(root / name).exists() and name not in expected:
            raise ProvenanceError("Untracked scientific configuration: " + name)


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ProvenanceError("Missing file inventory")
    seen, total = [], 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256", "mode"}:
            raise ProvenanceError("Invalid file record")
        path = relative_path(record["path"])
        if type(record["size"]) is not int or not 0 <= record["size"] <= MAX_FILE:
            raise ProvenanceError("Invalid file size")
        if record["mode"] not in (0o644, 0o755) or not HEX64.fullmatch(str(record["sha256"])):
            raise ProvenanceError("Invalid file hash/mode")
        seen.append(path)
        total += record["size"]
    if seen != sorted(set(seen)) or total > MAX_TOTAL:
        raise ProvenanceError("Unsorted, duplicate, or oversized inventory")


def load_wrapper(path):
    raw, _ = read_regular(path)
    value = json.loads(raw)
    value = value.get("manifest", value)  # Accept tools/sdsc's local run record.
    if value.get("schema") != "quest-sdsc-snapshot-v1":
        raise ProvenanceError("Expected an SDSC snapshot manifest")
    validate_records(value.get("files"))
    if sha(canonical(value["files"])) != value.get("code_sha256"):
        raise ProvenanceError("Wrapper file inventory hash mismatch")
    if value.get("total_bytes") != sum(item["size"] for item in value["files"]):
        raise ProvenanceError("Wrapper byte count mismatch")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", str(value.get("run_id", ""))):
        raise ProvenanceError("Invalid wrapper run identity")
    return value


def verify_wrapper_workspace(root, wrapper, records):
    science = [item for item in records if is_science(item["path"])]
    if science != [item for item in wrapper["files"] if is_science(item["path"])]:
        raise ProvenanceError("Wrapper scientific files differ from committed tree")
    for record in wrapper["files"]:
        data, mode = read_regular(root / record["path"])
        if (len(data), sha(data), mode) != (record["size"], record["sha256"], record["mode"]):
            raise ProvenanceError("Wrapper snapshot is stale: " + record["path"])


def readonly_tree(root):
    for folder, dirs, files in os.walk(root, topdown=False):
        for name in files:
            path = Path(folder) / name
            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        for name in dirs:
            (Path(folder) / name).chmod(0o555)
    root.chmod(0o555)


def validate_local_successor(value, head, public_tip):
    """Validate the explicit, bounded local export claim before importing Git."""
    if not isinstance(value, dict) or set(value) != {
        "public_base",
        "implementation",
        "acceptance",
        "commits",
    }:
        raise ProvenanceError("Invalid local successor record")
    identities = [value[name] for name in ("public_base", "implementation", "acceptance")]
    if any(not isinstance(item, str) or not HEX40.fullmatch(item) for item in identities):
        raise ProvenanceError("Local successor requires full real commit SHAs")
    commits = value["commits"]
    if (
        not isinstance(commits, list)
        or not 2 <= len(commits) <= MAX_LOCAL_COMMITS
        or any(not isinstance(item, str) or not HEX40.fullmatch(item) for item in commits)
        or len(set(commits)) != len(commits)
        or public_tip in commits
    ):
        raise ProvenanceError("Local successor requires a bounded unique commit inventory")
    if (
        len(set(identities)) != 3
        or value["public_base"] != public_tip
        or value["acceptance"] != head
        or commits[-2:] != [value["implementation"], value["acceptance"]]
    ):
        raise ProvenanceError("Local successor identities or commit order differ")
    return value


def unpublished_commits(root, public_tip, head):
    """Read at most one over the limit so oversized history fails before export."""
    commits = git(
        root, "rev-list", "--reverse", "--max-count=" + str(MAX_LOCAL_COMMITS + 1), public_tip + ".." + head
    ).decode().splitlines()
    if len(commits) > MAX_LOCAL_COMMITS:
        raise ProvenanceError("Local successor history exceeds the unpublished commit limit")
    return commits


def audit_local_successor(root, value, head, public_tip):
    """Require bounded linear history and audit every newly exported tree.

    This proves history and safe export only. The scientific resolver must still
    establish independent review and the review-only acceptance transition.
    """
    value = validate_local_successor(value, head, public_tip)
    previous = value["public_base"]
    git(root, "cat-file", "-e", previous + "^{commit}")
    git(root, "merge-base", "--is-ancestor", previous, value["implementation"])
    git(root, "merge-base", "--is-ancestor", value["implementation"], head)
    commits = unpublished_commits(root, previous, head)
    if commits != value["commits"]:
        raise ProvenanceError("Local successor must contain exactly the declared commit history")
    for commit in commits:
        parents = git(root, "rev-list", "--parents", "-n", "1", commit).decode().strip().split()
        if parents != [commit, previous]:
            raise ProvenanceError("Local successor must be linear through the implementation/acceptance pair")
        # Inspect every unpublished tree: deleting an unsafe path in a later
        # commit must not sneak its historical blob into the exported bundle.
        tree_records(root, commit)
        previous = commit


def prepare(
    repo,
    wrapper_manifest,
    public_ref="refs/remotes/public/master",
    accepted_ancestors=(),
    *,
    reviewed_local_implementation=None,
    reviewed_local_acceptance=None,
):
    """Create .sdsc/provenance/<unique>/ from HEAD without changing refs/index/worktree."""
    repo = safe_path(repo)
    if not re.fullmatch(r"refs/remotes/[A-Za-z0-9._/-]+", public_ref) or ".." in public_ref:
        raise ProvenanceError("A concrete local public tracking ref is required")
    head = git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tip = git(repo, "rev-parse", "--verify", public_ref + "^{commit}").decode().strip()
    if not HEX40.fullmatch(head) or not HEX40.fullmatch(tip):
        raise ProvenanceError("Only SHA-1 Git repositories are supported by this bundle format")
    if git(repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise ProvenanceError("Shallow history cannot carry complete acceptance ancestry")
    metadata = repo / (".opd-git" if (repo / ".opd-git").is_dir() else ".git")
    if (metadata / "info/grafts").exists():
        raise ProvenanceError("Grafted history is unsupported")
    local_successor = None
    if reviewed_local_implementation is not None or reviewed_local_acceptance is not None:
        local_successor = {
            "public_base": tip,
            "implementation": reviewed_local_implementation,
            "acceptance": reviewed_local_acceptance,
            "commits": unpublished_commits(repo, tip, head),
        }
        audit_local_successor(repo, local_successor, head, tip)
    else:
        # Default behavior remains fail-closed for unpublished scientific HEAD.
        git(repo, "merge-base", "--is-ancestor", head, tip)
    ancestors = sorted(set(accepted_ancestors))
    for ancestor in ancestors:
        if not HEX40.fullmatch(ancestor):
            raise ProvenanceError("Accepted ancestor must be a full real commit SHA")
        git(repo, "merge-base", "--is-ancestor", ancestor, head)
    records, _ = tree_records(repo, head)
    verify_science_workspace(repo, head, records)
    wrapper = load_wrapper(wrapper_manifest)
    if wrapper.get("git_head") != head:
        raise ProvenanceError("Wrapper HEAD differs from provenance HEAD")
    verify_wrapper_workspace(repo, wrapper, records)
    parent = safe_path(repo / ".sdsc/provenance")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=parent) as temporary:
        staging = Path(temporary)
        bundle_path = staging / "history.bundle"
        # HEAD only: no --all, tags, other branches, config, hooks, reflogs, or credentials.
        git(repo, "bundle", "create", "--version=2", str(bundle_path), "HEAD")
        bundle, _ = read_regular(bundle_path, MAX_TOTAL)
        if (
            git(repo, "rev-parse", "HEAD").decode().strip() != head
            or git(repo, "rev-parse", "--verify", public_ref + "^{commit}").decode().strip() != tip
        ):
            raise ProvenanceError("HEAD or public tracking ref changed during preparation")
        verify_science_workspace(repo, head, records)
        verify_wrapper_workspace(repo, wrapper, records)
        wrapper_bytes = canonical(wrapper) + b"\n"
        (staging / "wrapper-manifest.json").write_bytes(wrapper_bytes)
        manifest = {
            "schema": SCHEMA,
            "git_head": head,
            "public_tracking_ref": public_ref,
            "public_tracking_tip": tip,
            "accepted_ancestors": ancestors,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),  # noqa: UP017 (Python 3.8)
            "source_root": str(repo),
            "scientific_roots": list(SCIENCE_ROOTS),
            "scientific_root_files": list(SCIENCE_FILES),
            "scientific_files": [item for item in records if is_science(item["path"])],
            "committed_tree_sha256": sha(canonical(records)),
            "bundle": {"path": "history.bundle", "sha256": sha(bundle), "size": len(bundle)},
            "wrapper": {
                "run_id": wrapper["run_id"],
                "code_sha256": wrapper["code_sha256"],
                "manifest_sha256": sha(wrapper_bytes),
            },
        }
        if local_successor is not None:
            manifest["local_successor"] = local_successor
        manifest_bytes = canonical(manifest) + b"\n"
        (staging / "manifest.json").write_bytes(manifest_bytes)
        expected = sha(manifest_bytes)
        verify(staging, expected, wrapper["code_sha256"])
        destination = parent / ("provenance-" + uuid.uuid4().hex)
        destination.mkdir(mode=0o700)
        for name in ("history.bundle", "wrapper-manifest.json", "manifest.json"):
            os.rename(staging / name, destination / name)
        readonly_tree(destination)
    result = {
        "artifact": str(destination),
        "manifest_sha256": expected,
        "bundle_sha256": manifest["bundle"]["sha256"],
        "git_head": head,
        "wrapper_code_sha256": wrapper["code_sha256"],
    }
    if local_successor is not None:
        result["local_successor"] = local_successor
    return result


def load_artifact(artifact, expected_manifest_sha256, expected_wrapper_sha256):
    artifact = safe_path(artifact)
    if not HEX64.fullmatch(expected_manifest_sha256) or not HEX64.fullmatch(expected_wrapper_sha256):
        raise ProvenanceError("Trusted manifest and wrapper SHA-256 values are required")
    raw, _ = read_regular(artifact / "manifest.json")
    if sha(raw) != expected_manifest_sha256:
        raise ProvenanceError("Provenance manifest hash mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or not HEX40.fullmatch(str(manifest.get("git_head"))):
        raise ProvenanceError("Invalid provenance schema or HEAD")
    if "local_successor" in manifest:
        validate_local_successor(
            manifest["local_successor"], manifest["git_head"], manifest.get("public_tracking_tip")
        )
    if manifest.get("scientific_roots") != list(SCIENCE_ROOTS) or manifest.get(
        "scientific_root_files"
    ) != list(SCIENCE_FILES):
        raise ProvenanceError("Scientific scope cannot be overridden")
    validate_records(manifest.get("scientific_files"))
    wrapper_raw, _ = read_regular(artifact / "wrapper-manifest.json")
    wrapper = load_wrapper(artifact / "wrapper-manifest.json")
    if (
        manifest.get("wrapper")
        != {
            "run_id": wrapper["run_id"],
            "code_sha256": expected_wrapper_sha256,
            "manifest_sha256": sha(wrapper_raw),
        }
        or wrapper["code_sha256"] != expected_wrapper_sha256
    ):
        raise ProvenanceError("Wrapper identity mismatch")
    if wrapper.get("git_head") != manifest["git_head"]:
        raise ProvenanceError("Wrapper HEAD mismatch")
    bundle, _ = read_regular(artifact / "history.bundle", MAX_TOTAL)
    if manifest.get("bundle") != {"path": "history.bundle", "sha256": sha(bundle), "size": len(bundle)}:
        raise ProvenanceError("Bundle hash/size mismatch")
    header = b"# v2 git bundle\n" + manifest["git_head"].encode() + b" HEAD\n\n"
    if not bundle.startswith(header + b"PACK"):
        raise ProvenanceError("Bundle must contain exactly HEAD and no prerequisites or extra refs")
    return manifest, wrapper, bundle


def reconstruct(root, manifest, wrapper, bundle_path):
    init_repo(root)
    git(root, "bundle", "verify", str(bundle_path))
    git(root, "bundle", "unbundle", str(bundle_path))
    git(root, "fsck", "--full", "--strict", "--no-reflogs")
    if "local_successor" in manifest:
        audit_local_successor(
            root, manifest["local_successor"], manifest["git_head"], manifest.get("public_tracking_tip")
        )
    git(root, "update-ref", "--no-deref", "HEAD", manifest["git_head"])
    for ancestor in manifest["accepted_ancestors"]:
        if not HEX40.fullmatch(ancestor):
            raise ProvenanceError("Invalid accepted ancestor")
        git(root, "merge-base", "--is-ancestor", ancestor, manifest["git_head"])
    records, contents = tree_records(root, manifest["git_head"])
    if sha(canonical(records)) != manifest["committed_tree_sha256"]:
        raise ProvenanceError("Committed tree hash mismatch")
    science = [item for item in records if is_science(item["path"])]
    if science != manifest["scientific_files"] or science != [
        item for item in wrapper["files"] if is_science(item["path"])
    ]:
        raise ProvenanceError("Scientific bytes differ between bundle, audit, and wrapper")
    # Raw blobs + read-tree preserve Git identity without running checkout filters/hooks.
    for record in records:
        path = safe_path(root / record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(contents[record["path"]])
        path.chmod(record["mode"])
    git(root, "read-tree", "HEAD")
    git(root, "diff", "--quiet", "--no-ext-diff", "HEAD", "--")
    return records


def verify(artifact, expected_manifest_sha256, expected_wrapper_sha256, destination=None):
    """Validate in isolation; optionally publish a new read-only scientific checkout."""
    manifest, wrapper, bundle = load_artifact(artifact, expected_manifest_sha256, expected_wrapper_sha256)
    destination = safe_path(destination) if destination is not None else None
    if destination is not None and (destination.exists() or not destination.parent.is_dir()):
        raise ProvenanceError("Restore destination must be new and have an existing parent")
    with tempfile.TemporaryDirectory(
        prefix="sdsc-provenance-", dir=destination.parent if destination else None
    ) as temporary:
        staging = Path(temporary)
        # Re-read/import these already hash-verified bytes, avoiding artifact-path replacement races.
        bundle_path = staging / "history.bundle"
        bundle_path.write_bytes(bundle)
        checkout = staging / "checkout"
        reconstruct(checkout, manifest, wrapper, bundle_path)
        if destination is not None:
            # Exclusive creation prevents overwriting an existing user's checkout.
            destination.mkdir(mode=0o700)
            for path in checkout.iterdir():
                os.rename(path, destination / path.name)
            readonly_tree(destination)
    result = {
        "verified": True,
        "git_head": manifest["git_head"],
        "manifest_sha256": expected_manifest_sha256,
        "bundle_sha256": manifest["bundle"]["sha256"],
        "wrapper_code_sha256": expected_wrapper_sha256,
        "checkout": str(destination) if destination else None,
    }
    if "local_successor" in manifest:
        result["local_successor"] = manifest["local_successor"]
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Prepare local immutable provenance only")
    create.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    create.add_argument("--wrapper-manifest", type=Path, required=True)
    create.add_argument("--public-ref", default="refs/remotes/public/master")
    create.add_argument("--accepted-ancestor", action="append", default=[])
    create.add_argument(
        "--reviewed-local-implementation",
        help="explicit genuine implementation SHA in bounded linear history after the recorded public tip",
    )
    create.add_argument(
        "--reviewed-local-acceptance",
        help="explicit genuine acceptance SHA, directly after implementation and equal to HEAD",
    )
    for name in ("verify", "restore"):
        child = commands.add_parser(name)
        child.add_argument("--artifact", type=Path, required=True)
        child.add_argument("--expected-manifest-sha256", required=True)
        child.add_argument("--expected-wrapper-sha256", required=True)
        if name == "restore":
            child.add_argument("--destination", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "create":
            result = prepare(
                args.repo,
                args.wrapper_manifest,
                args.public_ref,
                args.accepted_ancestor,
                reviewed_local_implementation=args.reviewed_local_implementation,
                reviewed_local_acceptance=args.reviewed_local_acceptance,
            )
        else:
            result = verify(
                args.artifact,
                args.expected_manifest_sha256,
                args.expected_wrapper_sha256,
                getattr(args, "destination", None),
            )
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ProvenanceError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        message = str(exc) if isinstance(exc, ProvenanceError) else "Invalid or inaccessible provenance input"
        print(json.dumps({"error": message}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
