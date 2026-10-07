#!/bin/bash
# Invoked only by remote Expanse sbatch; no resource defaults or submission here.
set -euo pipefail
if [[ ( $# != 6 && $# != 7 ) || $4 != /* || ! -x $4 ]]; then
    printf '%s\n' 'usage: sdsc_job.sh RELEASE SUBMISSION RESULTS ABSOLUTE_PYTHON RUN_ID CODE_SHA256 [CONTAINER_JSON]' >&2
    exit 2
fi
exec "$4" -I -B -u - "$@" <<'PY'
import hashlib, json, os, pathlib, pwd, re, signal, stat, subprocess, sys, tempfile

release, submission, results, python, run_id, digest = sys.argv[1:7]
job_id = os.environ.get("SLURM_JOB_ID", "")
def interrupted(signum, _frame):
    raise RuntimeError("bootstrap interrupted by signal %s" % signum)
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
try:
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("worker requires a real Slurm job allocation")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("invalid run identity")
    home = pathlib.Path.home().resolve()
    tmp = os.environ.get("TMPDIR")
    if not tmp:
        tmp = "/scratch/%s/job_%s" % (pwd.getpwuid(os.getuid()).pw_name, job_id)
    scratch = pathlib.Path(tmp)
    if not scratch.is_absolute() or not scratch.is_dir() or scratch.is_symlink():
        raise ValueError("TMPDIR (or existing documented job scratch) is unavailable")
    scratch = scratch.resolve(strict=True)
    if scratch == home or home in scratch.parents or scratch.stat().st_uid != os.getuid():
        raise ValueError("job scratch must be owned by this user and outside HOME")
    mount = json.loads(subprocess.run(
        ["findmnt", "--json", "--target", str(scratch), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        check=True, capture_output=True, text=True, timeout=10).stdout)["filesystems"][0]
    if mount["fstype"] not in {"ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs"}:
        raise ValueError("job scratch filesystem is not verified node-local: %s" % mount)
    root = pathlib.Path(release)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("invalid release directory")
    root = root.resolve(strict=True)
    source = root / "source"
    manifest_path = root / "manifest.json"
    if source.is_symlink() or manifest_path.is_symlink():
        raise ValueError("release symlinks are forbidden")
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    computed = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode()).hexdigest()
    if computed != digest or manifest["code_sha256"] != digest or manifest["run_id"] != run_id:
        raise ValueError("release identity or content hash mismatch")
    stage = pathlib.Path(tempfile.mkdtemp(prefix="opd-%s-" % job_id, dir=scratch))
    seen = set()
    for entry in files:
        name = pathlib.PurePosixPath(entry["path"])
        if (name.is_absolute() or not name.parts or ".." in name.parts
                or str(name) != entry["path"] or str(name) in seen or str(name) == ".sdsc-stage.json"):
            raise ValueError("unsafe or duplicate manifest path")
        seen.add(str(name))
        path = source.joinpath(*name.parts)
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError("symlink in source snapshot")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or entry["mode"] not in (420, 493):
            raise ValueError("manifest entries must be regular code files")
        data = path.read_bytes()
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("source file differs from manifest: %s" % name)
        target = stage.joinpath(*name.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(entry["mode"])
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            raise OSError("staged source failed read-back: %s" % name)
    if "tools/sdsc_smoke.py" not in seen:
        raise ValueError("smoke worker is missing from release")
    (stage / ".sdsc-stage.json").write_text(json.dumps({"mount": mount, "code_sha256": digest}))
    os.chdir(stage)
    os.execv(python, [python, "-I", "-B", "-u", str(stage / "tools/sdsc_smoke.py"),
                     *sys.argv[1:7], str(stage), *sys.argv[7:]])
except BaseException as error:
    report = {"infrastructure_smoke": True, "passed": False, "phase": "bootstrap",
              "run_id": run_id, "job_id": job_id, "code_sha256": digest,
              "error": "%s: %s" % (type(error).__name__, error)}
    print(json.dumps(report), file=sys.stderr, flush=True)
    try:
        with open(pathlib.Path(submission) / "control-result.json", "x") as stream:
            json.dump(report, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        pass
    sys.exit(1)
PY
