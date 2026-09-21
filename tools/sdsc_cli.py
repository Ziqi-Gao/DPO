"""Quest-side, standard-library-only Expanse control commands. No local Slurm."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parent
TARGET = "zgao12@login.expanse.sdsc.edu"
REMOTE_ROOT = "/home/zgao12/quest-runs/" + ROOT.name
MAX_FILE = 2 * 1024 * 1024
MAX_TOTAL = 48 * 1024 * 1024
SOURCE_DIRS = {
    "src",
    "tools",
    "scripts",
    "tests",
    "configs",
    "docs",
    "prereg",
    "deployments",
    "third_party",
    ".github",
}
ROOT_FILES = {
    "AGENTS.md",
    "README.md",
    "LICENSE",
    "Makefile",
    "pyproject.toml",
    "environment.yml",
    "requirements-cpu.lock",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".env.example",
}
EXTENSIONS = {
    ".py",
    ".sh",
    ".slurm",
    ".sbatch",
    ".md",
    ".rst",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".toml",
    ".ini",
    ".cfg",
    ".lock",
    ".csv",
    ".tsv",
    ".html",
    ".css",
    ".js",
    ".svg",
}
DENIED_DIRS = {
    ".git",
    ".opd-git",
    ".codex",
    ".agents",
    ".ssh",
    ".aws",
    ".azure",
    ".sdsc",
    ".venv",
    "venv",
    "env",
    "envs",
    "node_modules",
    "site-packages",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "cache",
    "caches",
    ".cache",
    "checkpoints",
    "checkpoint",
    "outputs",
    "runs",
    "logs",
    "wandb",
    "weights",
    "secrets",
    "credentials",
    "dist",
    "build",
}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
PREFLIGHT_STORAGE = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
TASK_SCRIPTS = {
    "gpu-smoke": "sdsc_job.sh",
    "qwen3-v2-preflight": "sdsc_preflight_job.sh",
    "qwen3-v2-teacher-prepare": "sdsc_teacher_job.sh",
    "qwen3-v2-teacher-prompt-probe": "sdsc_teacher_probe_job.sh",
    "qwen3-v2-teacher-capability-probe": "sdsc_teacher_capability_job.sh",
    "qwen3-v2-g0-calibration": "sdsc_calibration_job.sh",
}
TASK_RESULTS = {
    "gpu-smoke": "result.json",
    "qwen3-v2-preflight": "preflight.json",
    "qwen3-v2-teacher-prepare": "teacher-prepare.json",
    "qwen3-v2-teacher-prompt-probe": "teacher-prompt-probe.json",
    "qwen3-v2-teacher-capability-probe": "teacher-capability-probe.json",
    "qwen3-v2-g0-calibration": "g0-calibration.json",
}

DIAGNOSTIC_TASKS = {"qwen3-v2-teacher-prompt-probe", "qwen3-v2-teacher-capability-probe"}


class UserError(Exception):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def safe_local_path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in [path, *path.parents]):
        raise UserError("Refusing symlink in local control or fetch path")
    return path


def state_root(root=None):
    root = ROOT if root is None else root
    path = safe_local_path(root / ".sdsc")
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def write_json(path, value):
    path = safe_local_path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path):
    path = safe_local_path(path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise UserError(f"Missing control record: {path}") from None


def git_command(root):
    # This Quest checkout uses .opd-git. Never reset/checkout/commit for deployment.
    if (root / ".opd-git").is_dir():
        return ["git", f"--git-dir={root / '.opd-git'}", f"--work-tree={root}"]
    return ["git", "-C", str(root)]


def git_read(root, *args):
    result = subprocess.run(git_command(root) + list(args), capture_output=True, timeout=30)
    if result.returncode:
        raise UserError("Git inventory failed; no snapshot was produced")
    return result.stdout


def exclude_reason(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or any(ord(c) < 32 for c in relative):
        return "unsafe path"
    parts = [p.lower() for p in path.parts]
    if any(p in DENIED_DIRS or p.startswith("checkpoint-") for p in parts):
        return "environment/cache/output/private directory"
    name = parts[-1]
    if name != ".env.example" and (
        name.startswith(".env")
        or name.endswith(".env")
        or name == "auth.json"
        or re.search(r"(^|[._-])(credentials?|secrets?|passwords?)([._-]|$)", name)
        or name
        in {
            "token",
            "tokens",
            "token.json",
            "tokens.json",
            "token.txt",
            "tokens.txt",
            "access_token",
            "refresh_token",
            "authorized_keys",
            "known_hosts",
        }
        or name.startswith("id_rsa")
        or name.startswith("id_ed25519")
        or name.endswith((".pem", ".key", ".p12", ".pfx", ".kdbx"))
    ):
        return "credential/private configuration filename"
    if len(parts) == 1:
        return None if path.name in ROOT_FILES else "not in small source allowlist"
    if path.parts[0] not in SOURCE_DIRS:
        return "not in source directories"
    if path.suffix.lower() not in EXTENSIONS and not (
        path.parts[0] in {"scripts", "tools"} and not path.suffix
    ):
        return "not a small source/configuration file type"
    return None


def safe_source_bytes(root, relative):
    path = root
    for component in PurePosixPath(relative).parts:
        path = path / component
        if path.is_symlink():
            raise UserError("symlink (not followed)")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise UserError("not a regular file")
        if before.st_size > MAX_FILE:
            raise UserError("file exceeds 2 MiB source limit")
        data = stream.read(MAX_FILE + 1)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise UserError("file changed while snapshotting; rerun dry-run")
    if len(data) > MAX_FILE or b"\x00" in data:
        raise UserError("oversized or binary file")
    # Known credential filenames are excluded before opening. Never display matches.
    if re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", data):
        raise UserError("private-key marker: snapshot refused")
    return data, 0o755 if before.st_mode & 0o111 else 0o644


def inventory(root=None):
    root = ROOT if root is None else root
    candidates = git_read(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    files, contents, excluded = [], {}, []
    for raw in sorted(set(candidates.split(b"\0")) - {b""}):
        relative = raw.decode("utf-8")
        reason = exclude_reason(relative)
        if reason:
            excluded.append({"path": relative, "reason": reason})
            continue
        try:
            data, mode = safe_source_bytes(root, relative)
        except FileNotFoundError:
            excluded.append({"path": relative, "reason": "deleted from working tree"})
            continue
        except UserError as exc:
            if "snapshot refused" in str(exc) or "changed while" in str(exc):
                raise UserError(f"{relative}: {exc}") from None
            excluded.append({"path": relative, "reason": str(exc)})
            continue
        files.append({"path": relative, "size": len(data), "sha256": sha(data), "mode": mode})
        contents[relative] = data
    files.sort(key=lambda item: item["path"])
    if sum(f["size"] for f in files) > MAX_TOTAL:
        raise UserError("Snapshot exceeds 48 MiB source budget; inspect eligible files")
    return files, contents, excluded


def make_manifest(files, root=None):
    root = ROOT if root is None else root
    digest = sha(canonical(files))
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id += "-" + digest[:12] + "-" + uuid.uuid4().hex[:8]
    return {
        "schema": "quest-sdsc-snapshot-v1",
        "run_id": run_id,
        "project": root.name,
        "source_root": str(root.resolve()),
        "git_head": git_read(root, "rev-parse", "HEAD").decode().strip(),
        "created_at": now(),
        "code_sha256": digest,
        "files": files,
        "total_bytes": sum(f["size"] for f in files),
    }


def build_archive(manifest, contents):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        records = [("manifest.json", canonical(manifest) + b"\n", 0o644)]
        records.extend(("source/" + f["path"], contents[f["path"]], f["mode"]) for f in manifest["files"])
        for path, data, mode in records:
            entry = tarfile.TarInfo(path)
            entry.size, entry.mode, entry.mtime = len(data), mode, 0
            archive.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


def ssh_base():
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True, check=True).stdout.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", host):
        raise UserError("Unexpected Quest hostname")
    control = str(Path.home() / ".ssh" / "cm" / ("sdsc-" + host))
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-S",
        control,
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ProxyCommand=false",
    ]


def require_master():
    args = ssh_base()
    result = subprocess.run([*args, "-O", "check", TARGET], capture_output=True, timeout=15)
    if result.returncode:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise UserError(
            f"Existing SSH master unavailable at {args[4]}: {message}\n"
            "Stopped; authenticate manually on this same Quest host. No login retry was attempted."
        )
    return args


def ssh_call(argv, data=b"", timeout=60):
    args = require_master()
    # OpenSSH takes a shell command; quote every argv element exactly once.
    # ProxyCommand=false prevents a new connection if the master dies after check.
    return subprocess.run([*args, TARGET, shlex.join(argv)], input=data, capture_output=True, timeout=timeout)


def remote(payload, data=b"", control_python=None):
    if control_python is None:
        check = read_json(state_root() / "check.json")
        if not check.get("connected"):
            raise UserError("Run tools/sdsc check successfully before remote operations")
        control_python = check["control_python"]
    payload = dict(payload, root=REMOTE_ROOT)
    source = (ROOT / "tools" / "sdsc_remote.py").read_text()
    timeout = (
        300 if payload.get("action") == "submit" and payload.get("task") == "qwen3-v2-g0-calibration" else 150
    )
    result = ssh_call([control_python, "-c", source, canonical(payload).decode()], data, timeout=timeout)
    try:
        response = json.loads(result.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise UserError("Remote reply missing/invalid; do not repeat a submission. Use reconcile.") from None
    if result.returncode or "error" in response:
        raise UserError(response.get("error", "Remote command failed"))
    return response


def container_options(args):
    keys = ("runtime", "image", "python")
    values = {key: getattr(args, "container_" + key, None) for key in keys}
    if all(value is None for value in values.values()):
        return None
    if not all(isinstance(value, str) and value for value in values.values()):
        raise UserError("Provide --container-runtime, --container-image, and --container-python together")
    if any(
        not Path(value).is_absolute() or ".." in Path(value).parts or any(ord(char) < 32 for char in value)
        for value in values.values()
    ):
        raise UserError("Container runtime, image, and Python paths must be explicit absolute paths")
    return values


def checked_container(specification):
    """Use only a matching successful login metadata inspection, without network."""
    placeholder = dict(specification, size=0, mtime_ns=0)
    blocker = "container identity/PyTorch metadata unconfirmed; run check with the same three container flags"
    try:
        check = read_json(state_root() / "check.json")
        inventory = check["inventory"]
        identity = inventory["candidate_container"]
        inspection = inventory["container_inspection"]
        metadata = json.loads(inspection["stdout"])
        valid = (
            check.get("connected") is True
            and inspection.get("returncode") == 0
            and isinstance(identity, dict)
            and set(identity) == {"runtime", "image", "python", "size", "mtime_ns"}
            and all(identity[key] == value for key, value in specification.items())
            and all(type(identity[key]) is int and identity[key] > 0 for key in ("size", "mtime_ns"))
            and metadata.get("supported_python") is True
            and isinstance(metadata.get("packages", {}).get("torch"), str)
            and bool(metadata["packages"]["torch"])
        )
        if valid:
            return identity, []
    except (OSError, UserError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return placeholder, [blocker]


def check_command(args):
    container = container_options(args)
    state = state_root()
    report = {
        "checked_at": now(),
        "quest_root": str(ROOT),
        "target": TARGET,
        "control_path": ssh_base()[4],
        "connected": False,
    }
    try:
        # Discovery only; never activate an environment or import CUDA on login.
        probe = b"""set -eu
for name in python3 python3.13 python3.12 python3.11 python3.10 python3.9 python3.8; do
  candidate=$(command -v "$name" 2>/dev/null) || continue
  if "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; then
    printf '%s\\n' "$candidate"
    exit 0
  fi
done
exit 2
"""
        result = ssh_call(["bash", "-s"], probe)
        interpreter = result.stdout.decode().strip()
        if result.returncode or not interpreter.startswith("/") or "\n" in interpreter:
            raise UserError("No remote control Python >=3.8 discovered; no environment was installed")
        report["control_python"] = interpreter
        request = {"action": "check"}
        if container is not None:
            request["container"] = container
        for name in ("python", "result_root"):
            if getattr(args, name, None):
                request[name] = getattr(args, name)
        report["inventory"] = remote(request, control_python=interpreter)
        report["connected"] = True
    except (UserError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
        write_json(state / "check.json", report)
        raise UserError(str(exc)) from None
    write_json(state / "check.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def sync_command(args):
    state = state_root()
    files, contents, excluded = inventory()
    if args.dry_run:
        manifest = make_manifest(files)
        record = {
            "state": "preview",
            "manifest": manifest,
            "excluded": excluded,
            "archive_bytes": len(build_archive(manifest, contents)),
            "remote_release": REMOTE_ROOT + "/releases/" + manifest["run_id"],
        }
        write_json(state / "last-preview.json", record)
        write_json(state / "runs" / (manifest["run_id"] + ".json"), record)
        for item in files:
            print(f"{item['size']:>9}  {item['path']}")
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "run_id": manifest["run_id"],
                    "file_count": len(files),
                    "total_bytes": manifest["total_bytes"],
                    "archive_bytes": record["archive_bytes"],
                    "code_sha256": manifest["code_sha256"],
                    "excluded_count": len(excluded),
                    "plan": str(state / "last-preview.json"),
                    "destination": record["remote_release"],
                },
                indent=2,
            )
        )
        return
    preview = read_json(state / "last-preview.json")
    manifest = preview["manifest"]
    if sha(canonical(files)) != manifest["code_sha256"]:
        raise UserError("Files changed since preview; run tools/sdsc sync --dry-run again")
    run_path = state / "runs" / (manifest["run_id"] + ".json")
    if read_json(run_path)["state"] == "deployed":
        raise UserError("Snapshot already deployed. Make a new dry-run for a fresh run_id")
    result = remote(
        {"action": "upload", "run_id": manifest["run_id"], "code_sha256": manifest["code_sha256"]},
        build_archive(manifest, contents),
    )
    if result.get("code_sha256") != manifest["code_sha256"] or result.get("run_id") != manifest["run_id"]:
        raise UserError("Upload response identity mismatch")
    write_json(run_path, dict(preview, state="deployed", deployment=result, deployed_at=now()))
    print(json.dumps(result, indent=2))


def run_record(run_id):
    if not IDENTIFIER.fullmatch(run_id):
        raise UserError("Invalid run_id")
    return read_json(state_root() / "runs" / (run_id + ".json"))


def resources(args):
    if args.task not in TASK_SCRIPTS:
        raise UserError("Unsupported task")
    value = {
        name: getattr(args, name)
        for name in ("account", "partition", "qos", "gpu_type", "gpus", "cpus", "mem_gib", "time")
    }
    preflight = args.task == "qwen3-v2-preflight"
    teacher = args.task == "qwen3-v2-teacher-prepare"
    probe = args.task in DIAGNOSTIC_TASKS
    calibration = args.task == "qwen3-v2-g0-calibration"
    expected = {
        "account": "nwu181",
        "partition": "nairr-gpu-shared",
        "qos": "nairr-gpu-shared-normal",
        "gpu_type": "h100",
        "gpus": 2 if preflight or calibration else 1,
        "cpus": 24 if preflight or teacher or calibration or probe else 4,
        "mem_gib": 192 if preflight or teacher or calibration or probe else 16,
    }
    if any(value[k] != v for k, v in expected.items()):
        raise UserError(
            f"{args.task} requires account nwu181, partition nairr-gpu-shared, "
            f"QoS nairr-gpu-shared-normal, {expected['gpus']} H100, "
            f"{expected['cpus']} CPUs, {expected['mem_gib']} GiB"
        )
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", value["time"])
    limit = 1800 if probe else 14400 if teacher else 7200 if preflight or calibration else 300
    if (
        not match
        or int(match[2]) >= 60
        or int(match[3]) >= 60
        or not 0 < int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) <= limit
    ):
        raise UserError(f"{args.task} walltime must be positive HH:MM:SS and at most {limit} seconds")
    return value


def submit_command(args):
    container = container_options(args)
    preflight = args.task == "qwen3-v2-preflight"
    teacher = args.task == "qwen3-v2-teacher-prepare"
    calibration = args.task == "qwen3-v2-g0-calibration"
    provenance_task = teacher or calibration
    model_task = preflight or provenance_task or args.task in DIAGNOSTIC_TASKS
    if model_task and container is not None:
        raise UserError(f"{args.task} requires a host runtime; container flags are forbidden")
    if not model_task and args.hf_home is not None:
        raise UserError("--hf-home is only supported for Qwen3 tasks")
    if provenance_task:
        if not re.fullmatch(r"[a-f0-9]{64}", args.provenance_manifest_sha256 or ""):
            raise UserError("This task requires --provenance-manifest-sha256")
        expected = REMOTE_ROOT + "/provenance/" + args.provenance_manifest_sha256
        if args.provenance_dir != expected:
            raise UserError("This task requires --provenance-dir " + expected)
    elif args.provenance_dir is not None or args.provenance_manifest_sha256 is not None:
        raise UserError("Provenance arguments are only supported for teacher preparation and calibration")
    if calibration:
        for name in ("teacher_job_id", "preflight_job_id"):
            if not re.fullmatch(r"[1-9][0-9]*", getattr(args, name) or ""):
                raise UserError(
                    "Calibration requires --" + name.replace("_", "-") + " with a real numeric job ID"
                )
        if args.teacher_job_id == args.preflight_job_id:
            raise UserError("Calibration requires two distinct upstream jobs")
    elif args.teacher_job_id is not None or args.preflight_job_id is not None:
        raise UserError("Upstream job arguments are only supported for calibration")
    record = run_record(args.run_id)
    manifest = record["manifest"]
    requested = resources(args)
    intent_id = uuid.uuid4().hex
    payload = {
        "action": "submit",
        "task": args.task,
        "run_id": args.run_id,
        "code_sha256": manifest["code_sha256"],
        "intent_id": intent_id,
        "resources": requested,
        "python": args.python,
        "result_root": args.result_root,
        "storage_confirmed": args.storage_confirmed,
        "authorized": bool(args.authorize),
    }
    if model_task:
        payload["hf_home"] = args.hf_home
    if provenance_task:
        payload.update(
            provenance_dir=args.provenance_dir, provenance_manifest_sha256=args.provenance_manifest_sha256
        )
    if calibration:
        payload.update(teacher_job_id=args.teacher_job_id, preflight_job_id=args.preflight_job_id)
    blockers = []
    if model_task:
        for name in ("hf_home", "result_root"):
            path = Path(getattr(args, name) or "/")
            if (
                not path.is_absolute()
                or ".." in path.parts
                or any(ord(char) < 32 for char in str(path))
                or PREFLIGHT_STORAGE not in (path, *path.parents)
            ):
                blockers.append(f"{name} must be an explicit path within {PREFLIGHT_STORAGE}")
    if container is not None:
        payload["container"], container_blockers = checked_container(container)
        blockers.extend(container_blockers)
    if record["state"] != "deployed":
        blockers.append("snapshot not uploaded (preview only)")
    if not args.python or not args.python.startswith("/"):
        blockers.append("explicit discovered remote Python executable required")
    if not args.result_root or not args.result_root.startswith("/"):
        blockers.append("explicit persistent result root required")
    if not args.storage_confirmed:
        blockers.append("persistent storage must be confirmed; GPU-node mount check still runs in job")
    plan = dict(
        payload,
        dry_run=True,
        blockers=blockers,
        remote_root=REMOTE_ROOT,
        command_transport="SSH only",
        proposed_job_name="opd-" + intent_id,
    )
    if args.dry_run:
        # Share the exact argv builder with SDSC; importing it executes no operation.
        spec = importlib.util.spec_from_file_location("sdsc_preview", TOOL_DIR / "sdsc_remote.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        intent = dict(
            payload,
            submission_dir=REMOTE_ROOT + "/submissions/" + intent_id,
            result_dir=(args.result_root or "/<UNCONFIRMED_PERSISTENT_ROOT>")
            + "/"
            + args.run_id
            + "/"
            + intent_id,
            python=args.python or "/<UNCONFIRMED_REMOTE_PYTHON>",
        )
        if model_task:
            intent["hf_home"] = args.hf_home or "/<UNCONFIRMED_MODEL_CACHE>"
        if calibration:
            intent["prerequisites_path"] = intent["submission_dir"] + "/prerequisites.json"
            # No SSH in dry-run: show the argument slot without claiming completed validation.
            intent["prerequisites_sha256"] = "0" * 64
            plan["prerequisites_verification"] = (
                "Required remotely before any claim or sbatch; not checked by dry-run"
            )
            plan["prerequisites_sha256_is_placeholder"] = True
        plan["sbatch_argv"] = helper.build_sbatch_argv(intent, Path(REMOTE_ROOT) / "releases" / args.run_id)
        plan["worker_script"] = str(TOOL_DIR / TASK_SCRIPTS[args.task])
        write_json(state_root() / "submit-plan.json", plan)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if not args.authorize:
        raise UserError("Submission requires explicit user authorization and --authorize")
    if blockers:
        raise UserError("; ".join(blockers))
    state = state_root()
    for path in (state / "submissions").glob("*.json"):
        previous = read_json(path)
        if previous.get("state") in {"unknown", "sending"}:
            raise UserError(f"Unresolved submission intent {path.stem}; reconcile before any new submission")
        if previous["request"]["run_id"] == args.run_id:
            raise UserError(f"run_id already has intent {path.stem}; reconcile, never blindly resubmit")
    require_master()
    intent_path = state / "submissions" / (intent_id + ".json")
    intent = {"state": "sending", "created_at": now(), "request": payload}
    write_json(intent_path, intent)
    print(f"submission_intent={intent_id}", flush=True)
    try:
        reply = remote(payload)
        if not re.fullmatch(r"[0-9]+", str(reply.get("job_id", ""))):
            raise UserError("No valid Slurm job ID in receipt")
        for name in ("intent_id", "run_id", "code_sha256"):
            if reply.get(name) != payload[name]:
                raise UserError("Submission receipt identity mismatch")
        if reply.get("container") != payload.get("container"):
            raise UserError("Submission receipt container identity mismatch")
        if reply.get("task", "gpu-smoke") != args.task or reply.get("hf_home") != payload.get("hf_home"):
            raise UserError("Submission receipt task or model-cache identity mismatch")
        if any(
            reply.get(key) != payload.get(key) for key in ("provenance_dir", "provenance_manifest_sha256")
        ):
            raise UserError("Submission receipt provenance identity mismatch")
        if calibration:
            validate_calibration_receipt(reply, payload)
        intent.update(state="submitted", receipt=reply)
    except (UserError, subprocess.TimeoutExpired, OSError, KeyboardInterrupt) as exc:
        intent.update(state="unknown", error=str(exc))
        write_json(intent_path, intent)
        raise UserError(
            f"Submission outcome UNKNOWN. Run tools/sdsc reconcile {intent_id}; do not resubmit."
        ) from None
    write_json(intent_path, intent)
    print(json.dumps(reply, indent=2))


def reconcile_command(args):
    if not re.fullmatch(r"[a-f0-9]{32}", args.intent_id):
        raise UserError("Invalid intent ID")
    path = state_root() / "submissions" / (args.intent_id + ".json")
    intent = read_json(path)
    result = remote({"action": "reconcile", "intent_id": args.intent_id})
    if result.get("job_id"):
        for key in ("intent_id", "run_id", "code_sha256"):
            if result.get(key) != intent["request"][key]:
                raise UserError("Recovered receipt identity mismatch")
        if result.get("container") != intent["request"].get("container"):
            raise UserError("Recovered receipt container identity mismatch")
        if result.get("task", "gpu-smoke") != intent["request"].get("task", "gpu-smoke") or result.get(
            "hf_home"
        ) != intent["request"].get("hf_home"):
            raise UserError("Recovered receipt task or model-cache identity mismatch")
        if any(
            result.get(key) != intent["request"].get(key)
            for key in ("provenance_dir", "provenance_manifest_sha256")
        ):
            raise UserError("Recovered receipt provenance identity mismatch")
        if intent["request"].get("task") == "qwen3-v2-g0-calibration":
            validate_calibration_receipt(result, intent["request"])
        intent.update(state="submitted", receipt=result)
    else:
        intent.update(state="unknown", reconciliation=result)
    write_json(path, intent)
    print(json.dumps(result, indent=2))


def validate_calibration_receipt(receipt, request):
    if any(receipt.get(key) != request.get(key) for key in ("teacher_job_id", "preflight_job_id")):
        raise UserError("Calibration receipt upstream job identity mismatch")
    expected = REMOTE_ROOT + "/submissions/" + request["intent_id"] + "/prerequisites.json"
    if receipt.get("prerequisites_path") != expected or not re.fullmatch(
        r"[a-f0-9]{64}", receipt.get("prerequisites_sha256", "")
    ):
        raise UserError("Calibration receipt lacks bound prerequisites")


def job_binding(job_id):
    if not re.fullmatch(r"[0-9]+", job_id):
        raise UserError("Job ID must be a single numeric Slurm job ID")
    matches = []
    for path in (state_root() / "submissions").glob("*.json"):
        intent = read_json(path)
        if str(intent.get("receipt", {}).get("job_id")) == job_id:
            matches.append((path.stem, intent))
    if len(matches) != 1:
        raise UserError("Job not uniquely bound to this project; reconcile its submission intent first")
    return matches[0]


def job_command(args):
    if args.command == "cancel" and not args.authorize:
        raise UserError("Cancellation requires explicit user authorization and --authorize")
    intent_id, _intent = job_binding(args.job_id)
    payload = {"action": args.command, "intent_id": intent_id, "job_id": args.job_id}
    if args.command == "logs":
        if not 1 <= args.lines <= 200:
            raise UserError("--lines must be between 1 and 200")
        payload["lines"] = args.lines
    if args.command == "cancel":
        payload["authorized"] = True
    result = remote(payload)
    if args.command == "fetch":
        destination = safe_local_path(state_root() / "fetched" / args.job_id)
        destination.mkdir(parents=True, mode=0o700, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="fetch-", dir=destination))
        total, saved = 0, []
        allowed = {
            TASK_RESULTS[_intent["request"].get("task", "gpu-smoke")],
            "receipt.json",
            "worker.log",
            "control-result.json",
            f"slurm-{args.job_id}.out",
            f"slurm-{args.job_id}.err",
        }
        for item in result.get("files", []):
            name = item["path"]
            if name not in allowed:
                raise UserError("Unexpected remote fetch file; refused")
            data = base64.b64decode(item["data_b64"], validate=True)
            total += len(data)
            if (
                len(data) > 1024 * 1024
                or total > 8 * 1024 * 1024
                or len(data) != item["size"]
                or sha(data) != item["sha256"]
            ):
                raise UserError("Fetch size limit or hash validation failed")
            with (destination / name).open("xb") as stream:
                stream.write(data)
            saved.append(name)
        result = {
            "job_id": args.job_id,
            "intent_id": intent_id,
            "destination": str(destination),
            "files": saved,
            "bytes": total,
            "note": "Small artifacts only; consult status for validated completion.",
        }
        write_json(destination / "fetch-manifest.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check existing SSH master and discover remote environment")
    check.add_argument("--python", help="optional discovered runtime to inspect without importing torch")
    check.add_argument("--result-root", help="optional persistent path to inspect on login node only")
    sync = sub.add_parser("sync", help="preview/upload immutable source snapshot")
    sync.add_argument("--dry-run", action="store_true", help="local inventory only; no SSH or upload")
    submit = sub.add_parser("submit", help="explicitly authorized, bounded GPU smoke or Qwen3 preflight")
    submit.add_argument("run_id")
    submit.add_argument("--task", choices=list(TASK_SCRIPTS), required=True)
    for name in ("account", "partition", "qos", "gpu-type", "time"):
        submit.add_argument("--" + name, required=True)
    for name in ("gpus", "cpus", "mem-gib"):
        submit.add_argument("--" + name, type=int, required=True)
    submit.add_argument(
        "--python", help="absolute host Python for staging; runs GPU smoke when no container is selected"
    )
    submit.add_argument(
        "--result-root", help="confirmed persistent directory; HOME only allows smoke-results"
    )
    submit.add_argument("--storage-confirmed", action="store_true")
    submit.add_argument("--teacher-job-id", help="calibration only: successful registered teacher task")
    submit.add_argument(
        "--preflight-job-id", help="calibration only: successful registered two-H100 preflight"
    )
    submit.add_argument("--hf-home", help="explicit persistent HF cache for Qwen3 tasks")
    submit.add_argument(
        "--provenance-dir", help="teacher preparation: separately verified remote Git artifact"
    )
    submit.add_argument(
        "--provenance-manifest-sha256", help="teacher preparation: trusted provenance manifest hash"
    )
    for command in (check, submit):
        command.add_argument(
            "--container-runtime", help="absolute discovered Singularity/Apptainer executable"
        )
        command.add_argument(
            "--container-image", help="absolute existing shared image; never pulled or copied"
        )
        command.add_argument("--container-python", help="absolute discovered Python path inside that image")
    choice = submit.add_mutually_exclusive_group(required=True)
    choice.add_argument("--dry-run", action="store_true", help="local plan only; never calls sbatch")
    choice.add_argument("--authorize", action="store_true", help="only after explicit user approval")
    reconcile = sub.add_parser("reconcile", help="recover ambiguous submission; never resubmits")
    reconcile.add_argument("intent_id")
    for name in ("status", "logs", "fetch", "cancel"):
        command = sub.add_parser(name)
        command.add_argument("job_id")
        if name == "logs":
            command.add_argument("--lines", type=int, default=80)
        if name == "cancel":
            command.add_argument("--authorize", action="store_true")
    return result


def main():
    args = parser().parse_args()
    handlers = {
        "check": check_command,
        "sync": sync_command,
        "submit": submit_command,
        "reconcile": reconcile_command,
    }
    try:
        handlers.get(args.command, job_command)(args)
    except (UserError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"sdsc: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
