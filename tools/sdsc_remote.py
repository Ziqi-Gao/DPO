#!/usr/bin/env python3
"""Bounded SDSC operations; sent over the authenticated SSH control connection.

This module uses only Python 3.8+ stdlib. It is not a daemon or local scheduler.
Every Slurm subprocess in this file executes on the SSH destination only.
"""

import base64
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import pwd
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path("/home/zgao12/quest-runs/OPD")
REMOTE_USER = "zgao12"
PREFLIGHT_STORAGE = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
TASK_SCRIPTS = {
    "gpu-smoke": "sdsc_job.sh",
    "qwen3-v2-preflight": "sdsc_preflight_job.sh",
    "qwen3-v2-teacher-prepare": "sdsc_teacher_job.sh",
    "qwen3-v2-teacher-prompt-probe": "sdsc_teacher_probe_job.sh",
    "qwen3-v2-teacher-capability-probe": "sdsc_teacher_capability_job.sh",
    "qwen3-v2-teacher-adapt": "sdsc_teacher_adapt_job.sh",
    "qwen3-v2-teacher-fit-preflight": "sdsc_teacher_fit_job.sh",
    "qwen3-v2-teacher-fit": "sdsc_teacher_fit_job.sh",
    "qwen3-v2-teacher-qualify": "sdsc_teacher_qualify_job.sh",
    "qwen3-v2-adapted-preflight": "sdsc_student_launch.sh",
    "qwen3-v2-adapted-calibration": "sdsc_student_launch.sh",
    "qwen3-v2-g0-calibration": "sdsc_calibration_job.sh",
}
TASK_WORKERS = {
    "gpu-smoke": "sdsc_smoke.py",
    "qwen3-v2-preflight": "sdsc_training_preflight.py",
    "qwen3-v2-teacher-prepare": "sdsc_teacher_prepare.py",
    "qwen3-v2-teacher-prompt-probe": "sdsc_teacher_probe.py",
    "qwen3-v2-teacher-capability-probe": "sdsc_teacher_capability.py",
    "qwen3-v2-teacher-adapt": "sdsc_teacher_adapt.py",
    "qwen3-v2-teacher-fit-preflight": "sdsc_teacher_fit.py",
    "qwen3-v2-teacher-fit": "sdsc_teacher_fit.py",
    "qwen3-v2-teacher-qualify": "sdsc_teacher_qualify.py",
    "qwen3-v2-adapted-preflight": "sdsc_student_job.py",
    "qwen3-v2-adapted-calibration": "sdsc_student_job.py",
    "qwen3-v2-g0-calibration": "sdsc_g0_calibration.py",
}
TASK_RESULTS = {
    "gpu-smoke": "result.json",
    "qwen3-v2-preflight": "preflight.json",
    "qwen3-v2-teacher-prepare": "teacher-prepare.json",
    "qwen3-v2-teacher-prompt-probe": "teacher-prompt-probe.json",
    "qwen3-v2-teacher-capability-probe": "teacher-capability-probe.json",
    "qwen3-v2-teacher-adapt": "teacher-adapt.json",
    "qwen3-v2-teacher-fit-preflight": "teacher-fit.json",
    "qwen3-v2-teacher-fit": "teacher-fit.json",
    "qwen3-v2-teacher-qualify": "teacher-qualify.json",
    "qwen3-v2-adapted-preflight": "adapted-preflight.json",
    "qwen3-v2-adapted-calibration": "adapted-calibration.json",
    "qwen3-v2-g0-calibration": "g0-calibration.json",
}
DIAGNOSTIC_TASKS = {
    "qwen3-v2-teacher-prompt-probe",
    "qwen3-v2-teacher-capability-probe",
    "qwen3-v2-teacher-adapt",
    "qwen3-v2-teacher-fit-preflight",
    "qwen3-v2-teacher-fit",
}
TEACHER_FIT_TASKS = {"qwen3-v2-teacher-fit-preflight": "preflight", "qwen3-v2-teacher-fit": "full-fit"}
TEACHER_FIT_BINDINGS = ("execution_plan_sha256", "actual_plan_sha256")
TEACHER_QUALIFY_TASK = "qwen3-v2-teacher-qualify"
STUDENT_TASKS = {"qwen3-v2-adapted-preflight": "preflight", "qwen3-v2-adapted-calibration": "calibration"}
STUDENT_BINDINGS = (
    "student_protocol_sha256",
    "student_protocol_artifact_sha256",
    "adapted_teacher_sha256",
    "teacher_acceptance_sha256",
    "teacher_acceptance_inventory_sha256",
)
STUDENT_FIXED_BINDINGS = {
    "adapted_teacher_sha256": "6928f2537dcca5f2d65c1498659e1ebf011845eb72ef364b9544036c2238e9c7",
    "teacher_acceptance_sha256": "5d6952823441bde567cdf7f5fad8b4625c58ee7e82425aad76c10433d0ec5337",
    "teacher_acceptance_inventory_sha256": "8d53783b9fb1d386de5a0a291c2b28e225347168cc7cfed4c0c0aceaf01d9bed",
}
TEACHER_QUALIFY_BINDINGS = (
    "adapted_teacher_sha256",
    "protocol_sha256",
    "protocol_artifact_sha256",
    "science_implementation_sha256",
    "claims_path",
    "claims_sha256",
)
TEACHER_QUALIFY_SMALL_RESULTS = {
    "qualification-manifest.json",
    "progress.json",
    "stage-summary.json",
    "acceptance-evidence.json",
}
TEACHER_FIT_SMALL_RESULTS = {
    "checkpoint-manifest.json",
    "data-isolation.json",
    "train-metrics.jsonl",
    "progress.json",
    "checkpoint-selection.json",
    "memory-environment-validate.json",
    "memory-environment-0.json",
    "memory-environment-1.json",
    "memory-environment-2.json",
    "memory-environment-3.json",
}
TEACHER_ADAPT_SMALL_RESULTS = {
    "checkpoint-manifest.json",
    "data-isolation.json",
    "train-metrics.jsonl",
    "dev-attempts.jsonl",
    "dev-capability.json",
    "progress.json",
}
PROVENANCE_BINDINGS = ("provenance_dir", "provenance_manifest_sha256", "science_git_head", "bundle_sha256")
PROVENANCE_TASKS = {
    "qwen3-v2-teacher-prepare",
    "qwen3-v2-g0-calibration",
    TEACHER_QUALIFY_TASK,
    *STUDENT_TASKS,
}
PREREQUISITE_BINDINGS = ("teacher_job_id", "preflight_job_id", "prerequisites_path", "prerequisites_sha256")
PREFLIGHT_MODELS = (
    ("models--Qwen--Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("models--Qwen--Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)
CONTAINER_PATH_KEYS = {"runtime", "image", "python"}
CONTAINER_KEYS = CONTAINER_PATH_KEYS | {"size", "mtime_ns"}
METADATA_SCRIPT = (
    "import sys,json,importlib.metadata as m\n"
    "packages={}\n"
    'for name in ("torch","numpy","transformers","accelerate"):\n'
    " try: packages[name]=m.version(name)\n"
    " except m.PackageNotFoundError: packages[name]=None\n"
    'print(json.dumps({"executable":sys.executable,"version":sys.version,'
    '"supported_python":sys.version_info>=(3,9),"packages":packages,'
    '"gpu_runtime_verified":False}))\n'
)
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_FILE = 4 * 1024 * 1024
MAX_FETCH_FILE = 1024 * 1024
MAX_FETCH_TOTAL = 8 * 1024 * 1024
TEMPORARY_ROOTS = tuple(map(Path, ("/tmp", "/var/tmp", "/dev/shm")))
DENIED_PARTS = {
    ".git",
    ".opd-git",
    ".codex",
    ".ssh",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".cache",
    "checkpoints",
    "checkpoint",
    "outputs",
    "results",
    "wandb",
    "runs",
    "credentials",
    "secrets",
    ".sdsc",
    "sdsc-results",
}
DENIED_NAMES = {
    "auth.json",
    "credentials.json",
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "authorized_keys",
    "known_hosts",
}
DENIED_SUFFIXES = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".pem",
    ".key",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".parquet",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()  # noqa: UP017 - control Python supports 3.8.


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def identifier(value, kind):
    patterns = {
        "run_id": r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}",
        "intent_id": r"[0-9a-f]{32}",
        "job_id": r"[1-9][0-9]*",
        "code_sha256": r"[0-9a-f]{64}",
    }
    require(isinstance(value, str) and re.fullmatch(patterns[kind], value), "Invalid " + kind)
    return value


def safe_path(path):
    """Refuse symlink components, including existing ancestors of new paths."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Absolute path required")
    require(not any(ord(c) < 32 for c in str(path)), "Control characters in path")
    for part in [path, *path.parents]:
        require(not part.is_symlink(), "Symlink path is not allowed: " + str(part))
    return path


def root_path(request):
    require(request.get("root") == str(ROOT), "Unexpected remote project root")
    return safe_path(ROOT)


def run(argv, timeout=30):
    """Arguments are always passed directly, never evaluated by a shell."""
    result = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-65536:].decode("utf-8", "replace"),
        "stderr": result.stderr[-16384:].decode("utf-8", "replace"),
    }


def atomic_json(path, value):
    path = safe_path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_bytes(path, limit=MAX_FETCH_FILE):
    path = safe_path(path)
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "Expected regular file")
    require(path.stat().st_size <= limit, "File exceeds size limit: " + path.name)
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "File changed beyond size limit")
    return data


def read_json(path):
    return json.loads(read_bytes(path))


def check_probe(argv, timeout):
    try:
        return run(argv, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error), "verified": False}


def container_paths(value):
    require(isinstance(value, dict), "Container configuration must be an object")
    paths = {}
    for key in CONTAINER_PATH_KEYS:
        require(isinstance(value.get(key), str), "Container paths must be explicit strings")
        path = Path(value[key])
        require(
            path.is_absolute() and ".." not in path.parts and not any(ord(char) < 32 for char in str(path)),
            "Container paths must be absolute and clean",
        )
        paths[key] = path
    return paths


def current_container_identity(value):
    """Stat an existing shared image; never hash, copy, pull, or install it."""
    require(
        isinstance(value, dict) and set(value) == CONTAINER_PATH_KEYS,
        "Container inspection needs exactly runtime, image, and python",
    )
    paths = container_paths(value)
    require(
        paths["runtime"].is_file() and os.access(str(paths["runtime"]), os.X_OK),
        "Selected container runtime is not executable",
    )
    image_path = safe_path(paths["image"])
    info = image_path.stat()
    require(
        stat.S_ISREG(info.st_mode) and info.st_size > 0 and os.access(str(image_path), os.R_OK),
        "Container image must be an existing readable regular file",
    )
    return dict(value, size=info.st_size, mtime_ns=info.st_mtime_ns)


def validate_container(value, verify_image=False):
    require(
        isinstance(value, dict) and set(value) == CONTAINER_KEYS,
        "Container identity must contain exactly runtime, image, python, size, mtime_ns",
    )
    container_paths(value)
    require(
        all(type(value[key]) is int and value[key] >= 0 for key in ("size", "mtime_ns")),
        "Container image identity requires integer size and mtime_ns",
    )
    if verify_image:
        actual = current_container_identity({key: value[key] for key in CONTAINER_PATH_KEYS})
        require(actual == value, "Container image changed since inspection; check it again before submission")
    return value


def qos_evidence(association, partition):
    """Describe the account/partition intersection without choosing a QoS silently."""
    evidence = {
        "account": "nwu181",
        "partition": "nairr-gpu-shared",
        "initial_qos": "nairr-gpu-shared",
        "configured_qos": "nairr-gpu-shared-normal",
        "verified": False,
        "initial_qos_jointly_allowed": None,
        "configured_qos_jointly_allowed": None,
    }
    if association.get("returncode") != 0 or partition.get("returncode") != 0:
        return evidence
    rows = [[field.strip() for field in line.split("|")] for line in association["stdout"].splitlines()]
    rows = [
        row for row in rows if len(row) >= 4 and row[0] == "nwu181" and row[1] in ("", "nairr-gpu-shared")
    ]
    allowed = re.search(r"\bAllowQos=(\S+)", partition["stdout"])
    if not rows or not allowed or not re.search(r"\bPartitionName=nairr-gpu-shared\s", partition["stdout"]):
        return evidence
    account_qos = {name for row in rows for name in row[3].split(",") if name}
    partition_qos = set(allowed.group(1).split(","))
    joint = account_qos if partition_qos == {"ALL"} else account_qos & partition_qos
    evidence.update(
        verified=True,
        account_default_qos=sorted({row[2] for row in rows if row[2]}),
        jointly_allowed_qos=sorted(joint),
        initial_qos_jointly_allowed=evidence["initial_qos"] in joint,
        configured_qos_jointly_allowed=evidence["configured_qos"] in joint,
    )
    return evidence


def check(request):
    root = root_path(request)
    commands = {
        name: shutil.which(name)
        for name in (
            "python3",
            "bash",
            "sbatch",
            "squeue",
            "sacct",
            "scancel",
            "findmnt",
            "sha256sum",
            "conda",
            "apptainer",
            "singularity",
            "expanse-client",
            "sacctmgr",
            "scontrol",
        )
    }
    existing = root
    while not existing.exists():
        existing = existing.parent
    report = {
        "identity": {
            "user": pwd.getpwuid(os.getuid()).pw_name,
            "uid": os.getuid(),
            "hostname": socket.gethostname(),
            "home": str(Path.home()),
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "commands": commands,
        "paths": {
            "root": str(root),
            "root_exists": root.is_dir(),
            "existing_parent": str(existing),
            "parent_writable": os.access(str(existing), os.W_OK),
        },
        "storage_verified_on_gpu_node": False,
        "environment_selection": "unselected",
    }
    if commands["findmnt"]:
        report["home_mount"] = check_probe(
            [commands["findmnt"], "-T", str(Path.home()), "-n", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            timeout=10,
        )
    if commands["conda"]:
        report["conda_environments"] = check_probe([commands["conda"], "env", "list", "--json"], timeout=10)
    if commands["sacctmgr"]:
        report["slurm_associations"] = check_probe(
            [
                commands["sacctmgr"],
                "show",
                "assoc",
                "user=" + report["identity"]["user"],
                "account=nwu181",
                "format=Account,Partition,DefaultQOS,QOS%180",
                "-nP",
            ],
            timeout=15,
        )
    if commands["scontrol"]:
        report["slurm_partition"] = check_probe(
            [commands["scontrol"], "show", "partition", "nairr-gpu-shared"], timeout=10
        )
    report["qos_evidence"] = qos_evidence(
        report.get("slurm_associations", {}), report.get("slurm_partition", {})
    )
    if commands["expanse-client"]:
        report["quota"] = check_probe(
            [commands["expanse-client"], "user", "-r", "expanse_nairr_gpu"], timeout=15
        )
    if commands["bash"]:
        # This constant script contains no caller-supplied shell fragments.
        report["modules"] = check_probe(
            [
                commands["bash"],
                "-lc",
                "if type module >/dev/null 2>&1; then module -t list 2>&1; "
                'module -t avail 2>&1 | head -n 100; else printf "module unavailable\\n"; fi',
            ],
            timeout=15,
        )
    if request.get("python") is not None:
        candidate = Path(request["python"])
        require(
            candidate.is_absolute()
            and candidate.is_file()
            and os.access(str(candidate), os.X_OK)
            and not any(ord(c) < 32 for c in str(candidate)),
            "Candidate Python must be an explicit executable path",
        )
        report["candidate_runtime"] = check_probe([str(candidate), "-I", "-c", METADATA_SCRIPT], timeout=15)
    if request.get("container") is not None:
        identity = current_container_identity(request["container"])
        report["candidate_container"] = identity
        report["container_inspection"] = check_probe(
            [
                identity["runtime"],
                "exec",
                "--cleanenv",
                "--no-home",
                identity["image"],
                identity["python"],
                "-I",
                "-c",
                METADATA_SCRIPT,
            ],
            timeout=25,
        )
    if request.get("result_root") is not None:
        candidate = safe_path(request["result_root"])
        report["candidate_storage"] = {
            "path": str(candidate),
            "exists": candidate.is_dir(),
            "readable": os.access(str(candidate), os.R_OK),
            "writable": os.access(str(candidate), os.W_OK),
            "gpu_node_verified": False,
            "persistent_storage_confirmed": False,
        }
        if candidate.exists() and commands["findmnt"]:
            report["candidate_storage"]["login_mount"] = check_probe(
                [commands["findmnt"], "-T", str(candidate), "-n", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
                timeout=10,
            )
    report["ready_for_transfer"] = report["paths"]["parent_writable"]
    report["missing_slurm_commands"] = [
        name for name in ("sbatch", "squeue", "sacct", "scancel") if not commands[name]
    ]
    return report


def validate_file_record(record):
    require(set(record) == {"path", "size", "sha256", "mode"}, "Unexpected file record fields")
    name = record["path"]
    require(
        isinstance(name, str) and name and "\\" not in name and not any(ord(c) < 32 for c in name),
        "Invalid source path",
    )
    path = PurePosixPath(name)
    require(
        not path.is_absolute() and str(path) == name and not any(part in ("..", ".") for part in path.parts),
        "Unsafe source path",
    )
    require(not any(part.lower() in DENIED_PARTS for part in path.parts), "Excluded source path")
    require(
        path.name.lower() not in DENIED_NAMES
        and (not path.name.lower().startswith(".env.") or name == ".env.example")
        and path.suffix.lower() not in DENIED_SUFFIXES,
        "Excluded source file",
    )
    require(type(record["size"]) is int and 0 <= record["size"] <= MAX_FILE, "Invalid file size")
    require(type(record["mode"]) is int and record["mode"] in (0o644, 0o755), "Invalid file mode")
    identifier(record["sha256"], "code_sha256")


def validate_manifest(manifest, request):
    require(isinstance(manifest, dict), "Manifest must be an object")
    require(manifest.get("run_id") == identifier(request.get("run_id"), "run_id"), "Run identity mismatch")
    require(manifest.get("project") == "OPD", "Unexpected project")
    require(isinstance(manifest.get("files"), list) and manifest["files"], "Empty file manifest")
    files = manifest["files"]
    for record in files:
        validate_file_record(record)
    require(len({record["path"] for record in files}) == len(files), "Duplicate source path")
    require(files == sorted(files, key=lambda record: record["path"]), "Files must be sorted")
    code_hash = digest(canonical(files))
    require(
        code_hash == identifier(request.get("code_sha256"), "code_sha256") == manifest.get("code_sha256"),
        "Code manifest hash mismatch",
    )
    require(
        manifest.get("total_bytes") == sum(record["size"] for record in files) <= MAX_ARCHIVE,
        "Manifest byte total mismatch or limit exceeded",
    )
    return files


def upload(request, stream):
    root = root_path(request)
    identifier(request.get("run_id"), "run_id")
    archive = stream.read(MAX_ARCHIVE + 1)
    require(len(archive) <= MAX_ARCHIVE, "Archive exceeds 64 MiB limit")
    # Prevalidate the complete tar before creating anything on SDSC.
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        members = tar.getmembers()
        require(
            len(members) <= 20000 and all(member.isfile() for member in members),
            "Tar must contain regular files only",
        )
        names = [member.name for member in members]
        require(
            len(set(names)) == len(names) and "manifest.json" in names, "Duplicate or missing tar entries"
        )
        manifest_member = tar.getmember("manifest.json")
        require(manifest_member.size <= MAX_FETCH_FILE, "Manifest exceeds size limit")
        manifest = json.load(tar.extractfile(manifest_member))
        records = validate_manifest(manifest, request)
        expected = {"manifest.json"} | {"source/" + record["path"] for record in records}
        require(set(names) == expected, "Tar inventory differs from manifest")
        for record in records:
            member = tar.getmember("source/" + record["path"])
            require(member.size == record["size"] and member.mode == record["mode"], "Tar metadata mismatch")
            require(
                digest(tar.extractfile(member).read()) == record["sha256"], "Source content hash mismatch"
            )
        releases = safe_path(root / "releases")
        releases.mkdir(parents=True, exist_ok=True, mode=0o700)
        release = safe_path(releases / request["run_id"])
        claim = safe_path(releases / ("." + request["run_id"] + ".upload"))
        require(not release.exists(), "Release exists; snapshots are immutable")
        # Claim persists on failure: a repeated invocation can never overwrite a partial upload.
        claim.mkdir(mode=0o700)
        temporary = claim / "release"
        temporary.mkdir(mode=0o700)
        for record in records:
            destination = temporary / "source" / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                output.write(tar.extractfile("source/" + record["path"]).read())
            destination.chmod(record["mode"])
        atomic_json(temporary / "manifest.json", manifest)
        require(not release.exists(), "Release appeared during upload")
        temporary.rename(release)
        claim.rmdir()
        safe_path(root / "smoke-results").mkdir(exist_ok=True, mode=0o700)
    return {
        "run_id": request["run_id"],
        "code_sha256": request["code_sha256"],
        "release": str(release),
        "total_bytes": manifest["total_bytes"],
        "file_count": len(records),
    }


def release_manifest(root, request):
    release = safe_path(root / "releases" / identifier(request.get("run_id"), "run_id"))
    manifest = read_json(release / "manifest.json")
    for record in validate_manifest(manifest, request):
        path = safe_path(release / "source" / record["path"])
        require(digest(read_bytes(path, MAX_FILE)) == record["sha256"], "Deployed source changed")
        require(stat.S_IMODE(path.stat().st_mode) == record["mode"], "Deployed source mode changed")
    return release, manifest


def validate_resources(resources, task="gpu-smoke"):
    require(task in TASK_SCRIPTS, "Unsupported task")
    preflight = task == "qwen3-v2-preflight"
    teacher = task == "qwen3-v2-teacher-prepare"
    probe = task in DIAGNOSTIC_TASKS
    calibration = task == "qwen3-v2-g0-calibration"
    student = task in STUDENT_TASKS
    expected = {
        "account": "nwu181",
        "partition": "nairr-gpu-shared",
        "qos": "nairr-gpu-shared-normal",
        "gpu_type": "h100",
        "gpus": 2 if preflight or calibration or student else 1,
        "cpus": 24 if preflight or teacher or calibration or probe or student else 4,
        "mem_gib": 192 if preflight or teacher or calibration or probe or student else 16,
    }
    if task in TEACHER_FIT_TASKS or task == TEACHER_QUALIFY_TASK:
        expected.update(partition="nairr-gpu", qos="nairr-gpu-normal", gpus=4, cpus=24, mem_gib=192)
    require(
        isinstance(resources, dict) and set(resources) == set(expected) | {"time"},
        "All resource parameters must be explicit",
    )
    require(
        all(
            type(resources[key]) is type(value) and resources[key] == value for key, value in expected.items()
        ),
        "Only the exact reviewed resource profile is supported for " + task,
    )
    match = re.fullmatch(r"([0-9]{2}):([0-9]{2}):([0-9]{2})", str(resources["time"]))
    require(match is not None, "Walltime must be HH:MM:SS")
    hours, minutes, seconds = map(int, match.groups())
    extended_diagnostic = task in {"qwen3-v2-teacher-prompt-probe", "qwen3-v2-teacher-adapt"}
    limit = 1800 if probe else 14400 if teacher else 7200 if preflight or calibration else 300
    if extended_diagnostic:
        limit = 3600
    if task in TEACHER_FIT_TASKS:
        limit = 3600 if TEACHER_FIT_TASKS[task] == "preflight" else 86400
    if task == TEACHER_QUALIFY_TASK:
        limit = 7200
    if task in STUDENT_TASKS:
        limit = 3600 if STUDENT_TASKS[task] == "preflight" else 7200
    require(
        minutes < 60 and seconds < 60 and 0 < hours * 3600 + minutes * 60 + seconds <= limit,
        "Walltime exceeds the bounded task profile",
    )


def receipt_from(intent, job_id, recovered=False):
    receipt = {
        key: intent[key]
        for key in (
            "intent_id",
            "run_id",
            "code_sha256",
            "resources",
            "result_dir",
            "submission_dir",
            "python",
            "created_at",
        )
    }
    receipt.update(job_id=identifier(job_id, "job_id"), submitted_at=now(), recovered=recovered)
    receipt["container"] = intent.get("container")
    receipt["task"] = intent.get("task", "gpu-smoke")
    receipt["hf_home"] = intent.get("hf_home")
    if intent.get("task") in PROVENANCE_TASKS:
        receipt.update({key: intent[key] for key in PROVENANCE_BINDINGS})
    if (
        intent.get("task") in {"qwen3-v2-g0-calibration", TEACHER_QUALIFY_TASK}
        or intent.get("task") in TEACHER_FIT_TASKS
        or intent.get("task") in STUDENT_TASKS
    ):
        receipt.update({key: intent[key] for key in PREREQUISITE_BINDINGS})
    if intent.get("task") in TEACHER_FIT_TASKS:
        receipt.update({key: intent[key] for key in TEACHER_FIT_BINDINGS})
    if intent.get("task") == TEACHER_QUALIFY_TASK:
        receipt.update({key: intent[key] for key in TEACHER_QUALIFY_BINDINGS})
    if intent.get("task") in STUDENT_TASKS:
        receipt.update({key: intent[key] for key in STUDENT_BINDINGS})
    return receipt


def build_sbatch_argv(intent, release):
    """Pure shared renderer for an exact dry-run and the eventual remote call."""
    task = intent.get("task", "gpu-smoke")
    validate_resources(intent.get("resources"), task)
    model_task = task != "gpu-smoke"
    require(not model_task or intent.get("container") is None, "Qwen3 tasks forbid container arguments")
    intent_id = identifier(intent.get("intent_id"), "intent_id")
    run_id = identifier(intent.get("run_id"), "run_id")
    code_hash = identifier(intent.get("code_sha256"), "code_sha256")
    resources = intent["resources"]
    release = Path(release)
    directory, result_dir, python = (Path(intent[key]) for key in ("submission_dir", "result_dir", "python"))
    for path in (release, directory, result_dir, python):
        require(
            path.is_absolute() and ".." not in path.parts and not any(ord(c) < 32 for c in str(path)),
            "Absolute clean invocation paths required",
        )
    script = release / "source/tools" / TASK_SCRIPTS[task]
    argv = [
        "sbatch",
        "--parsable",
        "--account=" + resources["account"],
        "--partition=" + resources["partition"],
        "--qos=" + resources["qos"],
        "--gpus=h100:" + str(resources["gpus"]),
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=" + str(resources["cpus"]),
        "--mem=" + str(resources["mem_gib"]) + "G",
        "--time=" + resources["time"],
        "--export=NONE",
        "--no-requeue",
        "--signal=B:TERM@900"
        if task == "qwen3-v2-teacher-fit"
        else "--signal=B:TERM@300"
        if task
        in {
            "qwen3-v2-g0-calibration",
            "qwen3-v2-teacher-adapt",
            "qwen3-v2-teacher-fit-preflight",
            TEACHER_QUALIFY_TASK,
            *STUDENT_TASKS,
        }
        else "--signal=B:TERM@60",
        "--job-name=opd-" + intent_id,
        "--comment=quest-" + intent_id,
        "--chdir=" + str(release),
        "--output=" + str(directory / "slurm-%j.out"),
        "--error=" + str(directory / "slurm-%j.err"),
        str(script),
        str(release),
        str(directory),
        str(result_dir),
        str(python),
        run_id,
        code_hash,
    ]
    if intent.get("container") is not None:
        argv.append(canonical(validate_container(intent["container"])).decode("utf-8"))
    if model_task:
        hf_home = intent.get("hf_home")
        require(
            isinstance(hf_home, str)
            and Path(hf_home).is_absolute()
            and ".." not in Path(hf_home).parts
            and not any(ord(c) < 32 for c in hf_home),
            "Explicit absolute HF cache required",
        )
        argv.append(hf_home)
    if task in PROVENANCE_TASKS:
        provenance_hash = identifier(intent.get("provenance_manifest_sha256"), "code_sha256")
        provenance_dir = intent.get("provenance_dir")
        require(
            provenance_dir == str(ROOT / "provenance" / provenance_hash),
            "Teacher provenance directory must match its trusted manifest hash",
        )
        argv.extend((provenance_dir, provenance_hash))
    if task == "qwen3-v2-g0-calibration":
        prerequisites_hash = identifier(intent.get("prerequisites_sha256"), "code_sha256")
        prerequisites_path = intent.get("prerequisites_path")
        require(
            prerequisites_path == str(directory / "prerequisites.json"), "Fixed prerequisites path required"
        )
        argv.extend((prerequisites_path, prerequisites_hash))
    if task in TEACHER_FIT_TASKS:
        argv.append(TEACHER_FIT_TASKS[task])
    return argv


def teacher_fit_contract(release):
    # Called only after release_manifest verifies every source byte. This
    # reviewed module is stdlib-only and never imports model/runtime code.
    path = safe_path(release / "source/tools/sdsc_teacher_fit_contract.py")
    spec = importlib.util.spec_from_file_location("sdsc_verified_teacher_fit_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teacher_qualify_contract(release):
    path = safe_path(release / "source/tools/sdsc_teacher_qualify_contract.py")
    spec = importlib.util.spec_from_file_location("sdsc_verified_teacher_qualify_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teacher_qualification_prerequisites(root, release, request, provenance):
    contract = teacher_qualify_contract(release)
    resolved = run(
        [
            request["python"],
            "-I",
            "-B",
            str(release / "source/tools/sdsc_teacher_qualify_contract.py"),
            "--resolve-protocol",
            provenance["provenance_dir"],
            provenance["provenance_manifest_sha256"],
            request["code_sha256"],
            provenance["science_git_head"],
        ],
        timeout=120,
    )
    require(
        resolved["returncode"] == 0,
        "Accepted qualification protocol resolution failed: " + resolved["stderr"],
    )
    protocol = json.loads(resolved["stdout"])
    require(
        protocol.get("review_status") == "accepted"
        and protocol.get("head") == provenance["science_git_head"],
        "Qualification needs genuine accepted protocol provenance",
    )
    release_files = {row["path"]: row for row in read_json(release / "manifest.json")["files"]}
    require(
        isinstance(protocol.get("science_file_sha256"), dict)
        and protocol["science_file_sha256"]
        and all(
            release_files.get(name, {}).get("sha256") == expected
            for name, expected in protocol["science_file_sha256"].items()
        ),
        "Qualification release differs from reviewed named scientific/transport source",
    )
    upstream = upstream_evidence(
        root, request["teacher_job_id"], "qwen3-v2-teacher-fit", "selected_full_teacher_fit"
    )
    contract.verify_fit_origin(upstream, protocol)
    require(upstream["receipt"]["run_id"] != request["run_id"], "Qualification needs a fresh release")
    selected = contract.selected_checkpoint(upstream)
    proof = dict(
        schema="quest-sdsc-teacher-qualification-prerequisites-v1",
        task=TEACHER_QUALIFY_TASK,
        target={key: request[key] for key in ("run_id", "code_sha256", "intent_id")},
        teacher_job_id=request["teacher_job_id"],
        created_at=now(),
        protocol=protocol,
        upstream=upstream,
        selected_checkpoint=selected,
        fit_evidence=contract.fit_evidence(upstream),
        admission_queue=teacher_fit_queue_check(),
    )
    require(len(canonical(proof)) + 1 <= MAX_FILE, "Qualification proof exceeds metadata bound")
    return proof


def student_contract(release):
    path = safe_path(release / "source/tools/sdsc_student_contract.py")
    spec = importlib.util.spec_from_file_location("sdsc_verified_student_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.BINDINGS) == STUDENT_BINDINGS, "Student transport binding contract differs")
    return module


def student_bindings(proof):
    values = {
        "student_protocol_sha256": proof["protocol"]["protocol_sha256"],
        "student_protocol_artifact_sha256": proof["protocol"]["artifact_sha256"],
        "adapted_teacher_sha256": proof["selected_checkpoint"]["adapted_teacher_sha256"],
        "teacher_acceptance_sha256": proof["acceptance"]["accepted_teacher_sha256"],
        "teacher_acceptance_inventory_sha256": proof["acceptance"]["inventory_sha256"],
    }
    for value in values.values():
        identifier(value, "code_sha256")
    require(
        all(values[key] == value for key, value in STUDENT_FIXED_BINDINGS.items()),
        "Student prerequisite changed the accepted teacher",
    )
    return values


def student_prerequisites(root, release, request, provenance):
    """Verify actual completed producers before consuming a fresh run claim."""
    contract = student_contract(release)
    resolved = run(
        [
            request["python"],
            "-I",
            "-B",
            str(release / "source/tools/sdsc_student_contract.py"),
            "--resolve-protocol",
            provenance["provenance_dir"],
            provenance["provenance_manifest_sha256"],
            request["code_sha256"],
            provenance["science_git_head"],
        ],
        timeout=180,
    )
    require(resolved["returncode"] == 0, "Accepted adapted-student protocol resolution failed")
    protocol = json.loads(resolved["stdout"])
    require(
        protocol.get("review_status") == "accepted"
        and protocol.get("head") == provenance["science_git_head"],
        "Student needs genuine accepted protocol provenance",
    )
    for name in ("implementation_commit", "acceptance_commit"):
        require(
            isinstance(protocol.get(name), str) and re.fullmatch(r"[0-9a-f]{40}", protocol[name]),
            "Student protocol needs real review commit identities",
        )
    require(
        protocol["implementation_commit"] != protocol["acceptance_commit"],
        "Student protocol implementation cannot self-accept",
    )
    for name in ("protocol_sha256", "artifact_sha256"):
        identifier(protocol.get(name), "code_sha256")
    release_files = {row["path"]: row for row in read_json(release / "manifest.json")["files"]}
    require(
        isinstance(protocol.get("science_file_sha256"), dict)
        and protocol["science_file_sha256"]
        and all(
            release_files.get(name, {}).get("sha256") == expected
            for name, expected in protocol["science_file_sha256"].items()
        ),
        "Student release differs from reviewed named science",
    )
    require(
        release_files.get("prereg/amendments/qwen3_adapted_student_calibration_v2.json", {}).get("sha256")
        == protocol["artifact_sha256"],
        "Student protocol artifact differs from release",
    )
    qualification = upstream_evidence(
        root, contract.QUALIFICATION_JOB_ID, TEACHER_QUALIFY_TASK, "accepted_teacher_qualification"
    )
    fit = upstream_evidence(root, contract.FIT_JOB_ID, "qwen3-v2-teacher-fit", "selected_full_teacher_fit")
    require(
        request["teacher_job_id"] == contract.QUALIFICATION_JOB_ID,
        "Student qualification job differs from fixed accepted origin",
    )
    for upstream in (qualification, fit):
        require(upstream["receipt"]["run_id"] != request["run_id"], "Student needs a fresh release")
    require(
        qualification["receipt"].get("teacher_job_id") == contract.FIT_JOB_ID,
        "Accepted teacher qualification does not bind the selected fit",
    )
    selected = teacher_qualify_contract(release).selected_checkpoint(fit)
    require(
        selected.get("adapted_teacher_sha256") == contract.DENSE_SHA256,
        "Student checkpoint differs from independent acceptance",
    )
    acceptance = contract.verify_teacher_acceptance(
        contract.ACCEPTANCE_ROOT, expected_inventory_sha256=contract.INVENTORY_SHA256
    )
    student_inputs, dataset_inputs = contract.input_plans(qualification)
    preflight = None
    if STUDENT_TASKS[request["task"]] == "calibration":
        preflight = upstream_evidence(
            root,
            request["preflight_job_id"],
            "qwen3-v2-adapted-preflight",
            "matching_adapted_student_preflight",
        )
        expected = {
            "science_git_head": protocol["head"],
            "student_protocol_sha256": protocol["protocol_sha256"],
            "student_protocol_artifact_sha256": protocol["artifact_sha256"],
            "teacher_job_id": contract.QUALIFICATION_JOB_ID,
            "python": request["python"],
            "hf_home": request["hf_home"],
            **STUDENT_FIXED_BINDINGS,
        }
        require(
            all(preflight["receipt"].get(key) == value for key, value in expected.items()),
            "Preflight protocol, teacher or runtime differs from calibration",
        )
        require(preflight["receipt"]["run_id"] != request["run_id"], "Calibration needs a fresh release")
        prior_files = {row["path"]: row for row in preflight["release_manifest"]["files"]}
        require(
            all(
                prior_files.get(name, {}).get("sha256") == expected
                for name, expected in protocol["science_file_sha256"].items()
            ),
            "Preflight named science differs from calibration",
        )
    proof = {
        "schema": "quest-sdsc-adapted-student-prerequisites-v1",
        "task": request["task"],
        "target": {key: request[key] for key in ("run_id", "code_sha256", "intent_id")},
        "created_at": now(),
        "protocol": protocol,
        "qualification": qualification,
        "fit": fit,
        "acceptance": acceptance,
        "selected_checkpoint": selected,
        "student_inputs": student_inputs,
        "dataset_inputs": dataset_inputs,
        "preflight": preflight,
        "admission_queue": calibration_queue_check(),
    }
    student_bindings(proof)
    require(len(canonical(proof)) + 1 <= MAX_FILE, "Adapted-student prerequisite metadata exceeds limit")
    return proof


def verify_student_result(receipt, result, published):
    release, unused = release_manifest(ROOT, receipt)
    contract = student_contract(release)
    require(
        all(
            value.get(key) == receipt.get(key)
            for value in (result, published)
            for key in (*PREREQUISITE_BINDINGS, *STUDENT_BINDINGS)
        ),
        "Adapted-student result/prerequisite binding differs",
    )
    proof_raw = read_bytes(safe_path(receipt["prerequisites_path"]), MAX_FILE)
    require(digest(proof_raw) == receipt["prerequisites_sha256"], "Student prerequisite proof changed")
    proof = json.loads(proof_raw)
    require(
        proof.get("schema") == "quest-sdsc-adapted-student-prerequisites-v1"
        and proof.get("task") == receipt["task"]
        and proof.get("target") == {key: receipt[key] for key in ("run_id", "code_sha256", "intent_id")}
        and proof["protocol"]["head"] == receipt["science_git_head"]
        and proof["qualification"]["receipt"]["job_id"] == receipt["teacher_job_id"] == "54496291"
        and proof["fit"]["receipt"]["job_id"] == "54494742"
        and all(receipt.get(key) == value for key, value in student_bindings(proof).items()),
        "Student prerequisite identity differs",
    )
    if STUDENT_TASKS[receipt["task"]] == "preflight":
        require(
            proof.get("preflight") is None and receipt.get("preflight_job_id") is None,
            "Adapted preflight borrowed unrelated execution evidence",
        )
    else:
        require(
            proof["preflight"]["receipt"]["job_id"] == receipt["preflight_job_id"],
            "Adapted calibration preflight binding differs",
        )
    contract.validate_report(result, proof)
    contract.validate_publication(safe_path(receipt["result_dir"]), result, published)


def teacher_fit_queue_check():
    """One conservative pre-submit check for a four-GPU attempt, no retry loop."""
    user = pwd.getpwuid(os.getuid()).pw_name
    queue = run(
        [
            "squeue",
            "--noheader",
            "--user=" + user,
            "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED",
            "--format=%i|%T|%b",
        ]
    )
    require(queue["returncode"] == 0, "Cannot verify the four-GPU concurrency bound")
    for line in queue["stdout"].splitlines():
        fields = [field.strip() for field in line.split("|")]
        require(len(fields) == 3 and fields[0] and fields[1], "Malformed GPU queue response")
        require(
            fields[2] in {"", "N/A", "(null)", "None"},
            "Other GPU/GRES jobs remain queued or active; four-GPU teacher work is blocked",
        )
    return queue


def teacher_fit_prerequisites(root, release, manifest, request):
    contract = teacher_fit_contract(release)
    mode = TEACHER_FIT_TASKS[request["task"]]
    execution = contract.execution_plan(manifest)
    execution_sha = contract.sha256_value(execution)
    actual_sha = contract.sha256_value(contract.actual_plan(mode))
    require(
        request.get("execution_plan_sha256") == execution_sha
        and request.get("actual_plan_sha256") == actual_sha,
        "Requested teacher fit plans differ from intended release",
    )
    upstream = {
        "teacher": upstream_evidence(
            root, request["teacher_job_id"], "qwen3-v2-teacher-adapt", "one_gpu_adaptation_preflight"
        )
    }
    if mode == "full-fit":
        upstream["preflight"] = upstream_evidence(
            root,
            request["preflight_job_id"],
            "qwen3-v2-teacher-fit-preflight",
            "four_gpu_teacher_fit_preflight",
        )
        report = upstream["preflight"]["report"]
        require(
            report.get("execution_plan_sha256") == execution_sha,
            "Four-GPU preflight does not match the intended model/runtime/trainer kernel",
        )
    require(
        all(value["receipt"]["run_id"] != request["run_id"] for value in upstream.values()),
        "Teacher fit requires a fresh source release",
    )
    proof = {
        "schema": "quest-sdsc-teacher-fit-prerequisites-v1",
        "task": request["task"],
        "mode": mode,
        "created_at": now(),
        "target": {key: request[key] for key in ("run_id", "code_sha256", "intent_id")},
        "execution_plan_sha256": execution_sha,
        "actual_plan_sha256": actual_sha,
        "teacher_job_id": request["teacher_job_id"],
        "preflight_job_id": request.get("preflight_job_id"),
        "upstream": upstream,
        "admission_queue": teacher_fit_queue_check(),
    }
    require(len(canonical(proof)) + 1 <= MAX_FILE, "Teacher fit prerequisite proof exceeds size bound")
    return proof


def verify_provenance(root, release, manifest, request):
    """Verify genuine Git before claiming a run, using hash-bound release code only."""
    expected = identifier(request.get("provenance_manifest_sha256"), "code_sha256")
    artifact = safe_path(request.get("provenance_dir", ""))
    require(
        artifact == root / "provenance" / expected and artifact.is_dir(),
        "Teacher provenance directory must exist and match its trusted manifest hash",
    )
    verifier = safe_path(release / "source/tools/sdsc_provenance.py")
    record = next((item for item in manifest["files"] if item["path"] == "tools/sdsc_provenance.py"), None)
    require(record is not None, "Snapshot must bind the provenance verifier")
    source = read_bytes(verifier, MAX_FILE)
    require(
        len(source) == record["size"] and digest(source) == record["sha256"],
        "Provenance verifier differs from the source snapshot",
    )
    namespace = {"__name__": "sdsc_bound_provenance", "__file__": str(verifier)}
    exec(compile(source, str(verifier), "exec"), namespace)
    result = namespace["verify"](artifact, expected, request["code_sha256"])
    require(
        namespace["load_wrapper"](artifact / "wrapper-manifest.json") == manifest,
        "Provenance wrapper differs from the submitted release",
    )
    require(
        result.get("verified") is True
        and result.get("manifest_sha256") == expected
        and result.get("wrapper_code_sha256") == request["code_sha256"],
        "Provenance verification identity mismatch",
    )
    return {
        "provenance_dir": str(artifact),
        "provenance_manifest_sha256": expected,
        "science_git_head": result["git_head"],
        "bundle_sha256": result["bundle_sha256"],
    }


def registered_job(root, job_id, task):
    """Find exactly one locally owned submission, never an arbitrary Slurm ID."""
    identifier(job_id, "job_id")
    submissions = safe_path(root / "submissions")
    matches = []
    if submissions.is_dir():
        for directory in sorted(submissions.iterdir()):
            if not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            path = safe_path(directory / "receipt.json")
            if path.is_file() and read_json(path).get("job_id") == job_id:
                request = {"root": str(root), "intent_id": directory.name, "job_id": job_id}
                receipt, unused = bound_receipt(request)
                require(receipt.get("task") == task, "Upstream job has the wrong task")
                matches.append((request, receipt))
    require(len(matches) == 1, "Upstream job must have exactly one registered submission receipt")
    return matches[0]


def verify_published_file(path, record):
    """Stream a required training input; never load a model or checkpoint here."""
    path = safe_path(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == record["size"],
            "Published training input size mismatch: " + str(path),
        )
        hasher = hashlib.sha256()
        size = 0
        while True:
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            require(size <= record["size"], "Published training input grew during verification")
            hasher.update(block)
        after = os.fstat(stream.fileno())
    require(
        (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        and size == record["size"]
        and hasher.hexdigest() == record["sha256"],
        "Published training input hash mismatch: " + str(path),
    )


def upstream_evidence(root, job_id, task, role):
    request, receipt = registered_job(root, job_id, task)
    evidence = status(request)
    require(
        evidence.get("success") is True and evidence.get("state") == "COMPLETED",
        role + " must have successful Slurm accounting and verified persistent results",
    )
    result_dir = safe_path(receipt["result_dir"])
    require(
        PREFLIGHT_STORAGE in (result_dir, *result_dir.parents),
        "Upstream results must be in the persistent project storage",
    )
    report_path = safe_path(result_dir / TASK_RESULTS[task])
    report_bytes = read_bytes(report_path)
    report = json.loads(report_bytes)
    require(
        digest(report_bytes) == evidence["result"]["result_sha256"],
        "Upstream report changed after status verification",
    )
    publication_path = safe_path(result_dir / "receipt.json")
    publication_bytes = read_bytes(publication_path)
    publication = json.loads(publication_bytes)
    # Revalidate against the exact report/receipt pair after the scheduler query.
    require(verify_result(receipt).get("verified") is True, "Upstream publication changed")
    records = publication.get("files")
    require(isinstance(records, list), "Publication inventory must be a list")
    by_path = {}
    for record in records:
        require(isinstance(record, dict), "Invalid published file record")
        relative = record.get("path")
        require(
            isinstance(relative, str)
            and relative
            and str(PurePosixPath(relative)) == relative
            and not PurePosixPath(relative).is_absolute()
            and ".." not in PurePosixPath(relative).parts
            and not any(ord(c) < 32 for c in relative),
            "Unsafe published file path",
        )
        require(
            relative not in by_path and type(record.get("size")) is int and record["size"] >= 0,
            "Duplicate or invalid published file record",
        )
        identifier(record.get("sha256"), "code_sha256")
        by_path[relative] = record
    report_record = by_path.get(report_path.name, {})
    require(
        report_record.get("sha256") == digest(report_bytes)
        and report_record.get("size") == len(report_bytes),
        "Publication report hash mismatch",
    )
    verified_files = []
    if role == "teacher":
        for prefix in ("artifacts/dataset", "artifacts/teacher_demos"):
            expected = {name for name in by_path if name.startswith(prefix + "/")}
            require(expected, "Teacher publication is missing " + prefix)
            tree = safe_path(result_dir / prefix)
            require(tree.is_dir(), "Teacher input directory is missing")
            actual = set()
            for path in tree.rglob("*"):
                safe_path(path)
                if path.is_dir():
                    continue
                require(
                    path.is_file() and stat.S_ISREG(path.stat().st_mode),
                    "Teacher inputs must be regular files",
                )
                actual.add(path.relative_to(result_dir).as_posix())
            require(actual == expected, "Teacher input inventory differs from publication")
            for name in sorted(expected):
                verify_published_file(result_dir / name, by_path[name])
                verified_files.append(by_path[name])
    release, manifest = release_manifest(root, receipt)
    manifest_path = safe_path(release / "manifest.json")
    manifest_bytes = read_bytes(manifest_path, MAX_FILE)
    require(json.loads(manifest_bytes) == manifest, "Upstream source manifest changed")
    require(
        read_bytes(report_path) == report_bytes and read_bytes(publication_path) == publication_bytes,
        "Upstream proof changed during input verification",
    )
    return {
        "receipt": receipt,
        "result_dir": str(result_dir),
        "status": evidence,
        "report_path": str(report_path),
        "report_sha256": digest(report_bytes),
        "report": report,
        "publication_receipt_path": str(publication_path),
        "publication_receipt_sha256": digest(publication_bytes),
        "publication_receipt": publication,
        "release_manifest_path": str(manifest_path),
        "release_manifest_sha256": digest(manifest_bytes),
        "release_manifest": manifest,
        "file_verification": {
            "scope": "all_teacher_dataset_and_store_files" if role == "teacher" else "report_and_source_only",
            "verified_files": verified_files,
            "preflight_checkpoint_content_rehashed": False,
        },
    }


def calibration_queue_check():
    """One conservative admission check, not a queue, scheduler, or retry loop."""
    user = pwd.getpwuid(os.getuid()).pw_name
    queue = run(
        [
            "squeue",
            "--noheader",
            "--user=" + user,
            "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED",
            "--format=%i|%T|%j",
        ]
    )
    require(queue["returncode"] == 0, "Cannot verify current OPD queue; calibration is blocked")
    active = []
    for line in queue["stdout"].splitlines():
        fields = [field.strip() for field in line.split("|")]
        require(len(fields) == 3, "Malformed queue response; calibration is blocked")
        if fields[2].startswith("opd-"):
            active.append(fields[0])
    require(not active, "Other OPD jobs remain active; calibration is blocked: " + ",".join(active))
    return queue


def calibration_prerequisites(root, request, provenance):
    upstream = {
        "teacher": upstream_evidence(root, request["teacher_job_id"], "qwen3-v2-teacher-prepare", "teacher"),
        "preflight": upstream_evidence(root, request["preflight_job_id"], "qwen3-v2-preflight", "preflight"),
    }
    manifest_bytes = read_bytes(safe_path(Path(provenance["provenance_dir"]) / "manifest.json"), MAX_FILE)
    require(digest(manifest_bytes) == provenance["provenance_manifest_sha256"], "Provenance manifest changed")
    science = json.loads(manifest_bytes)
    for value in upstream.values():
        require(value["receipt"]["run_id"] != request["run_id"], "Calibration needs a fresh source release")
        records = [
            item
            for item in value["release_manifest"]["files"]
            if any(
                item["path"] == name or item["path"].startswith(name + "/")
                for name in science["scientific_roots"]
            )
            or item["path"] in science["scientific_root_files"]
        ]
        require(records == science["scientific_files"], "Upstream scientific source differs from calibration")
    require(
        upstream["teacher"]["report"].get("science_git_head") == provenance["science_git_head"],
        "Teacher scientific Git identity differs from calibration",
    )
    proof = {
        "schema": "quest-sdsc-calibration-prerequisites-v1",
        "task": "qwen3-v2-g0-calibration",
        "created_at": now(),
        "target": {key: request[key] for key in ("run_id", "code_sha256", "intent_id")},
        "upstream": upstream,
        "admission_queue": calibration_queue_check(),
    }
    require(len(canonical(proof)) + 1 <= MAX_FILE, "Calibration proof exceeds its bounded size")
    return proof


def claim_run(root, request):
    """A consumed snapshot cannot acquire a second submission identity."""
    claims = safe_path(root / "run-claims")
    claims.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = safe_path(claims / (request["run_id"] + ".json"))
    try:
        with path.open("xb") as stream:
            stream.write(
                canonical({key: request[key] for key in ("run_id", "intent_id", "code_sha256")}) + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(str(claims), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        original = ""
        with contextlib.suppress(OSError, ValueError, TypeError, AttributeError):
            original = " " + identifier(read_json(path).get("intent_id"), "intent_id")
        raise ValueError(
            "Run is already claimed; reconcile its original intent" + original + ", never submit another"
        ) from None


def submit(request):
    root = root_path(request)
    require(request.get("authorized") is True, "Explicit submission authorization required")
    require(request.get("storage_confirmed") is True, "Persistent result storage must be confirmed")
    task = request.get("task")
    validate_resources(request.get("resources"), task)
    preflight = task == "qwen3-v2-preflight"
    calibration = task == "qwen3-v2-g0-calibration"
    teacher_fit = task in TEACHER_FIT_TASKS
    teacher_qualify = task == TEACHER_QUALIFY_TASK
    student = task in STUDENT_TASKS
    provenance_task = task in PROVENANCE_TASKS
    model_task = preflight or provenance_task or task in DIAGNOSTIC_TASKS
    require(not model_task or request.get("container") is None, "Qwen3 tasks forbid container arguments")
    require(model_task or request.get("hf_home") is None, "Smoke does not use a model cache")
    require(
        provenance_task or all(request.get(key) is None for key in PROVENANCE_BINDINGS),
        "This task does not accept provenance arguments",
    )
    require(
        calibration
        or teacher_fit
        or teacher_qualify
        or student
        or all(request.get(key) is None for key in PREREQUISITE_BINDINGS),
        "This task does not accept upstream prerequisites",
    )
    require(
        teacher_fit or all(request.get(key) is None for key in TEACHER_FIT_BINDINGS),
        "This task does not accept teacher fit plan identities",
    )
    require(
        all(request.get(key) is None for key in TEACHER_QUALIFY_BINDINGS),
        "Qualification checkpoint/protocol/claims must be resolved remotely, not supplied",
    )
    require(
        all(request.get(key) is None for key in STUDENT_BINDINGS),
        "Student identities must be resolved remotely, not supplied",
    )
    if student:
        teacher_job = identifier(request.get("teacher_job_id"), "job_id")
        require(teacher_job == "54496291", "Adapted student requires accepted teacher job 54496291")
        require(
            request.get("prerequisites_path") is None and request.get("prerequisites_sha256") is None,
            "Adapted student rejects caller-generated prerequisites",
        )
        preflight_job = None
        if STUDENT_TASKS[task] == "calibration":
            preflight_job = identifier(request.get("preflight_job_id"), "job_id")
            require(
                preflight_job not in {teacher_job, "54494742", "54345604"},
                "Adapted calibration requires a fresh matching preflight",
            )
        else:
            require(request.get("preflight_job_id") is None, "Adapted preflight rejects prior preflight ID")
    if teacher_qualify:
        teacher_job = identifier(request.get("teacher_job_id"), "job_id")
        preflight_job = None
        require(
            request.get("preflight_job_id") is None
            and request.get("prerequisites_path") is None
            and request.get("prerequisites_sha256") is None,
            "Qualification rejects caller-generated prerequisites",
        )
    if calibration:
        require(
            request.get("prerequisites_path") is None and request.get("prerequisites_sha256") is None,
            "Calibration prerequisites must be generated from live remote verification",
        )
        teacher_job = identifier(request.get("teacher_job_id"), "job_id")
        preflight_job = identifier(request.get("preflight_job_id"), "job_id")
        require(teacher_job != preflight_job, "Two distinct upstream jobs are required")
    if teacher_fit:
        require(
            request.get("prerequisites_path") is None and request.get("prerequisites_sha256") is None,
            "Teacher fit prerequisites must be generated from live remote verification",
        )
        teacher_job = identifier(request.get("teacher_job_id"), "job_id")
        require(teacher_job == "54493015", "Teacher fit requires the fixed one-GPU adaptation preflight")
        preflight_job = None
        if TEACHER_FIT_TASKS[task] == "full-fit":
            preflight_job = identifier(request.get("preflight_job_id"), "job_id")
            require(preflight_job != teacher_job, "Full fit needs a distinct four-GPU preflight")
        else:
            require(
                request.get("preflight_job_id") is None, "Four-GPU preflight rejects another preflight ID"
            )
    intent_id = identifier(request.get("intent_id"), "intent_id")
    release, manifest = release_manifest(root, request)
    required_files = {"tools/" + TASK_SCRIPTS[task], "tools/" + TASK_WORKERS[task]}
    if task in DIAGNOSTIC_TASKS:
        required_files.update({"tools/sdsc_teacher_prepare.py", "tools/sdsc_training_preflight.py"})
    if task == "qwen3-v2-teacher-adapt":
        required_files.update(
            {
                "tools/sdsc_teacher_probe.py",
                "tools/sdsc_teacher_capability.py",
                "src/posttrain_circuits/learning/teacher/adaptation.py",
            }
        )
    if teacher_fit:
        contract = teacher_fit_contract(release)
        required_files.update(contract.KERNEL_PATHS)
    if provenance_task:
        required_files.add("tools/sdsc_provenance.py")
    if student:
        required_files.update(
            {
                "tools/sdsc_student_contract.py",
                "tools/sdsc_adapted_training_preflight.py",
                "tools/sdsc_adapted_calibration.py",
                "tools/sdsc_teacher_qualify_contract.py",
                "src/posttrain_circuits/artifacts/adapted_student_protocol.py",
                "src/posttrain_circuits/artifacts/adapted_teacher_sft.py",
            }
        )
    if teacher_qualify:
        required_files.update(
            {
                "tools/sdsc_teacher_qualify_contract.py",
                "src/posttrain_circuits/artifacts/teacher_adaptation_protocol.py",
                "src/posttrain_circuits/artifacts/teacher_acceptance.py",
            }
        )
    require(
        required_files <= {record["path"] for record in manifest["files"]},
        "Snapshot manifest must bind both preflight worker files"
        if preflight
        else "Snapshot manifest must bind teacher worker and provenance verifier files"
        if provenance_task
        else "Snapshot manifest must bind both GPU smoke worker files",
    )
    python = Path(request.get("python", ""))
    require(
        python.is_absolute()
        and ".." not in python.parts
        and not any(ord(c) < 32 for c in str(python))
        and python.is_file()
        and os.access(str(python), os.X_OK),
        "Selected Python is not executable",
    )
    if request.get("container") is not None:
        validate_container(request["container"], verify_image=True)
    result_root = safe_path(request.get("result_root", ""))
    require(
        result_root.is_dir() and os.access(str(result_root), os.W_OK),
        "Result root must exist and be writable",
    )
    small_smoke_metadata = not model_task and result_root == root / "smoke-results"
    require(
        (small_smoke_metadata or (Path.home() != result_root and Path.home() not in result_root.parents))
        and not any(parent in result_root.parents or parent == result_root for parent in TEMPORARY_ROOTS),
        "Results require persistent storage; HOME permits only this project smoke-results metadata",
    )
    hf_home = None
    if model_task:
        require(
            PREFLIGHT_STORAGE in (result_root, *result_root.parents),
            "Qwen3 results must use the verified persistent project storage",
        )
        hf_home = safe_path(request.get("hf_home", ""))
        require(
            PREFLIGHT_STORAGE in (hf_home, *hf_home.parents) and hf_home.is_dir(),
            "Qwen3 HF cache must exist within verified persistent project storage",
        )
        for model, revision in PREFLIGHT_MODELS:
            snapshot = safe_path(hf_home / "hub" / model / "snapshots" / revision)
            require(snapshot.is_dir(), "Pinned model snapshot is missing: " + model)
    provenance = verify_provenance(root, release, manifest, request) if provenance_task else {}
    prerequisites = calibration_prerequisites(root, request, provenance) if calibration else None
    if teacher_fit:
        prerequisites = teacher_fit_prerequisites(root, release, manifest, request)
    if teacher_qualify:
        prerequisites = teacher_qualification_prerequisites(root, release, request, provenance)
    if student:
        prerequisites = student_prerequisites(root, release, request, provenance)
    script = safe_path(release / "source/tools" / TASK_SCRIPTS[task])
    require(script.is_file(), "Snapshot lacks the reviewed worker script")
    submissions = safe_path(root / "submissions")
    submissions.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = safe_path(submissions / intent_id)
    require(not directory.exists(), "Intent already exists; reconcile it, never resubmit blindly")
    claim_run(root, request)
    directory.mkdir(mode=0o700)
    result_dir = safe_path(result_root / request["run_id"] / intent_id)
    result_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    intent = {
        "intent_id": intent_id,
        "run_id": request["run_id"],
        "code_sha256": manifest["code_sha256"],
        "resources": request["resources"],
        "python": str(python),
        "created_at": now(),
        "submission_dir": str(directory),
        "result_dir": str(result_dir),
        "task": task,
        "hf_home": str(hf_home) if hf_home is not None else None,
        "container": request.get("container"),
        **provenance,
    }
    if calibration or teacher_fit or teacher_qualify or student:
        prerequisites_path = safe_path(directory / "prerequisites.json")
        atomic_json(prerequisites_path, prerequisites)
        prerequisites_bytes = read_bytes(prerequisites_path, MAX_FILE)
        require(prerequisites_bytes == canonical(prerequisites) + b"\n", "Prerequisites read-back mismatch")
        intent.update(
            teacher_job_id=teacher_job,
            preflight_job_id=preflight_job,
            prerequisites_path=str(prerequisites_path),
            prerequisites_sha256=digest(prerequisites_bytes),
        )
    if teacher_fit:
        intent.update({key: prerequisites[key] for key in TEACHER_FIT_BINDINGS})
    if teacher_qualify:
        contract = teacher_qualify_contract(release)
        protocol = prerequisites["protocol"]
        binding = dict(
            prerequisites["target"],
            teacher_job_id=teacher_job,
            adapted_teacher_sha256=prerequisites["selected_checkpoint"]["adapted_teacher_sha256"],
            protocol_sha256=protocol["protocol_sha256"],
            prerequisites_sha256=intent["prerequisites_sha256"],
            supplemental_namespace=contract.CLAIM_NAMESPACE,
            validation_file_sha256=contract.VALIDATION_SHA256,
            supplemental_order_sha256=contract.SUPPLEMENTAL_ORDER_SHA256,
        )
        claims = dict(
            schema="quest-sdsc-teacher-qualification-reservations-v1",
            binding=binding,
            reservations=contract.reserve_claims(root, binding),
            stage_root=str(root / "teacher-qualification-claims/stages" / intent_id),
            stage_claim_ids={role: intent_id + ":" + role for role in contract.ROLES},
        )
        claims_path = directory / "claims.json"
        atomic_json(claims_path, claims)
        intent.update(
            adapted_teacher_sha256=binding["adapted_teacher_sha256"],
            protocol_sha256=protocol["protocol_sha256"],
            protocol_artifact_sha256=protocol["artifact_sha256"],
            science_implementation_sha256=protocol["science_implementation_sha256"],
            claims_path=str(claims_path),
            claims_sha256=digest(read_bytes(claims_path, MAX_FILE)),
        )
    if student:
        intent.update(student_bindings(prerequisites))
    atomic_json(directory / "intent.json", intent)
    argv = build_sbatch_argv(intent, release)
    atomic_json(directory / "submission.json", {"state": "submitting", "argv": argv, "at": now()})
    try:
        result = run(argv, timeout=45)
        match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\s*", result["stdout"])
        if result["returncode"] == 0 and match:
            receipt = receipt_from(intent, match.group(1))
            atomic_json(directory / "receipt.json", receipt)
            return receipt
        atomic_json(directory / "submission.json", {"state": "unknown", "at": now(), "response": result})
    except (OSError, subprocess.TimeoutExpired):
        atomic_json(directory / "submission.json", {"state": "unknown", "at": now()})
    return {
        "state": "unknown",
        "intent_id": intent_id,
        "message": "No trustworthy submission receipt. Reconcile this intent; do not retry submission.",
    }


def intent_and_directory(request):
    directory = safe_path(
        root_path(request) / "submissions" / identifier(request.get("intent_id"), "intent_id")
    )
    intent = read_json(directory / "intent.json")
    require(
        intent.get("intent_id") == request["intent_id"] and intent.get("submission_dir") == str(directory),
        "Submission identity mismatch",
    )
    return intent, directory


def reconcile(request):
    directory = safe_path(
        root_path(request) / "submissions" / identifier(request.get("intent_id"), "intent_id")
    )
    if not (directory / "intent.json").exists():
        return {
            "state": "unknown",
            "intent_id": request["intent_id"],
            "message": (
                "Remote intent is absent or incomplete; retain any run claim and never resubmit blindly."
            ),
        }
    intent, directory = intent_and_directory(request)
    if (directory / "receipt.json").exists():
        return read_json(directory / "receipt.json")
    name = "opd-" + intent["intent_id"]
    user = pwd.getpwuid(os.getuid()).pw_name
    queue = run(["squeue", "--noheader", "--user=" + user, "--name=" + name, "--format=%i|%j"])
    search_start = (dt.datetime.fromisoformat(intent["created_at"]) - dt.timedelta(days=1)).date().isoformat()
    accounting = run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--user=" + user,
            "--name=" + name,
            "--starttime=" + search_start,
            "--format=JobIDRaw,JobName%100,User%64",
        ]
    )
    require(
        queue["returncode"] == 0 and accounting["returncode"] == 0,
        "Scheduler lookup failed; submission remains unknown",
    )
    ids = set()
    for line in queue["stdout"].splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) == 2 and fields[1] == name and re.fullmatch(r"[1-9][0-9]*", fields[0]):
            ids.add(fields[0])
    for line in accounting["stdout"].splitlines():
        fields = [field.strip() for field in line.split("|")]
        if (
            len(fields) >= 3
            and fields[1] == name
            and fields[2] == user
            and re.fullmatch(r"[1-9][0-9]*", fields[0])
        ):
            ids.add(fields[0])
    if len(ids) == 1:
        receipt = receipt_from(intent, ids.pop(), recovered=True)
        atomic_json(directory / "receipt.json", receipt)
        return receipt
    return {
        "state": "unknown",
        "intent_id": intent["intent_id"],
        "candidate_job_ids": sorted(ids),
        "message": "Zero or ambiguous matches do not authorize another submission.",
    }


def bound_receipt(request):
    intent, directory = intent_and_directory(request)
    receipt = read_json(directory / "receipt.json")
    job_id = identifier(request.get("job_id"), "job_id")
    require(receipt.get("job_id") == job_id, "Job ID is not bound to this submission")
    for key in (
        "intent_id",
        "run_id",
        "code_sha256",
        "result_dir",
        "submission_dir",
        "resources",
        "python",
        "created_at",
        "container",
        "hf_home",
        *PROVENANCE_BINDINGS,
        *PREREQUISITE_BINDINGS,
        *TEACHER_FIT_BINDINGS,
        *TEACHER_QUALIFY_BINDINGS,
    ):
        require(receipt.get(key) == intent.get(key), "Receipt binding mismatch")
    require(receipt.get("task", "gpu-smoke") == intent.get("task", "gpu-smoke"), "Receipt task mismatch")
    return receipt, directory


def teacher_adapt_preflight_plan():
    return {
        "version": "qwen3-v2-teacher-adapt-preflight-v1",
        "base_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "seed": 271828,
        "train_examples": 256,
        "dev_examples": 32,
        "optimizer_steps": 8,
        "global_batch_size": 32,
        "microbatch_size": 1,
        "max_sequence_length": 1536,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "lora_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "loss": "mean_of_response_token_means_per_sequence",
        "teacher_readiness_claim": False,
    }


def validate_adaptation_preflight_report(result):
    checks = {
        "isolated_data",
        "finite_loss_and_gradients",
        "eight_real_optimizer_steps",
        "nonzero_adapter_update",
        "frozen_base_before_merge",
        "complete_module_coverage",
        "dense_export_reload",
        "merged_dev_evaluated",
        "cgroup_headroom",
    }
    plan = result.get("adaptation_plan")
    require(
        isinstance(plan, dict)
        and canonical(plan) == canonical(teacher_adapt_preflight_plan())
        and result.get("adaptation_plan_sha256") == digest(canonical(plan))
        and type(result.get("optimizer_steps")) is int
        and result["optimizer_steps"] == 8
        and isinstance(result.get("checks"), dict)
        and set(result["checks"]) == checks
        and all(value is True for value in result["checks"].values()),
        "Adaptation preflight plan, optimizer steps, or required checks differ",
    )


def verify_adaptation_manifest(root, result, published):
    """Bind the small checkpoint inventory; consumers must rehash model bytes.

    Status polling must not read ~16 GB of weights every five minutes. The
    wrapper records its durable read-back; every later checkpoint consumer must
    verify the actual bytes against this bound inventory before model loading.
    """
    expected_hash = identifier(result.get("checkpoint_manifest_sha256"), "code_sha256")
    path = safe_path(root / "artifacts/checkpoint-manifest.json")
    raw = read_bytes(path)
    require(digest(raw) == expected_hash, "Adaptation checkpoint manifest hash differs")
    manifest = json.loads(raw)
    require(isinstance(manifest, dict), "Invalid adaptation checkpoint manifest")
    content = {key: value for key, value in manifest.items() if key != "sha256"}
    require(
        manifest.get("sha256") == digest(canonical(content))
        and manifest["sha256"] == result.get("adapted_teacher_sha256")
        and manifest.get("artifact_kind") == "adapted_dense_teacher"
        and manifest.get("base_revision") == "b968826d9c46dd6066d109eabc6255188de91218"
        and manifest.get("formal_teacher_accepted") is False,
        "Adaptation checkpoint identity or nonaccepting scope differs",
    )
    require(
        manifest.get("adaptation_plan_sha256") == result.get("adaptation_plan_sha256"),
        "Adaptation checkpoint and report refer to different plans",
    )
    records = published.get("files")
    require(isinstance(records, list), "Missing adaptation publication inventory")
    by_path = {}
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get("path"), str), "Bad publication record")
        require(record["path"] not in by_path, "Duplicate publication path")
        by_path[record["path"]] = record
    require(
        by_path.get("artifacts/checkpoint-manifest.json")
        == {"path": "artifacts/checkpoint-manifest.json", "size": len(raw), "sha256": expected_hash},
        "Checkpoint manifest is not bound by the publication receipt",
    )
    metrics_path = safe_path(root / "artifacts/train-metrics.jsonl")
    require(
        metrics_path.is_file()
        and stat.S_ISREG(metrics_path.stat().st_mode)
        and metrics_path.stat().st_size <= MAX_FETCH_FILE,
        "Bounded adaptation training metrics are missing",
    )
    metrics_raw = read_bytes(metrics_path)
    metrics_sha = digest(metrics_raw)
    require(
        isinstance(result.get("adaptation_dataset"), dict)
        and bool(result["adaptation_dataset"])
        and type(result.get("consumed_tokens")) is int
        and result["consumed_tokens"] > 0
        and type(result.get("optimizer_steps")) is int
        and result["optimizer_steps"] == 8,
        "Adaptation training data or actual update counts are missing",
    )
    provenance = dict(
        run_id=result.get("run_id"),
        job_id=result.get("job_id"),
        code_sha256=result.get("code_sha256"),
        dataset_manifest_sha256=digest(canonical(result["adaptation_dataset"])),
        train_metrics_sha256=metrics_sha,
        optimizer_steps=result["optimizer_steps"],
        consumed_tokens=result["consumed_tokens"],
    )
    require(
        canonical(manifest.get("training_provenance")) == canonical(provenance)
        and by_path.get("artifacts/train-metrics.jsonl")
        == {"path": "artifacts/train-metrics.jsonl", "size": len(metrics_raw), "sha256": metrics_sha},
        "Adaptation training provenance differs from report, actual metrics or publication receipt",
    )
    files = manifest.get("files")
    require(isinstance(files, list) and bool(files), "Missing adapted checkpoint files")
    seen = set()
    for record in files:
        require(isinstance(record, dict) and isinstance(record.get("path"), str), "Bad checkpoint file")
        name = PurePosixPath(record["path"])
        require(
            not name.is_absolute()
            and len(name.parts) >= 2
            and name.parts[0] in {"merged", "adapter"}
            and ".." not in name.parts
            and str(name) == record["path"]
            and record["path"] not in seen
            and type(record.get("size")) is int
            and record["size"] >= 0,
            "Unsafe checkpoint file inventory",
        )
        identifier(record.get("sha256"), "code_sha256")
        seen.add(record["path"])
        relative = "artifacts/" + record["path"]
        require(
            by_path.get(relative) == dict(record, path=relative),
            "Checkpoint file publication binding differs",
        )
        file = safe_path(root / relative)
        require(
            file.is_file() and stat.S_ISREG(file.stat().st_mode) and file.stat().st_size == record["size"],
            "Persistent checkpoint file metadata differs",
        )
    required = {
        "merged/config.json",
        "merged/tokenizer_config.json",
        "merged/tokenizer.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
    }
    require(
        required <= seen
        and any(name.startswith("merged/") and name.endswith(".safetensors") for name in seen),
        "Adapted dense checkpoint/tokenizer/adapter inventory is incomplete",
    )
    require(
        {name for name in by_path if name.startswith(("artifacts/merged/", "artifacts/adapter/"))}
        == {"artifacts/" + name for name in seen},
        "Publication contains unlisted checkpoint files",
    )


def verify_teacher_fit_artifacts(receipt, result, published):
    release, manifest = release_manifest(ROOT, receipt)
    contract = teacher_fit_contract(release)
    expected_execution = contract.execution_plan(manifest)
    contract.validate_report(result, expected_execution, receipt["task"])
    require(
        all(result.get(key) == receipt.get(key) for key in (*TEACHER_FIT_BINDINGS, *PREREQUISITE_BINDINGS)),
        "Teacher fit report plan/prerequisite binding differs from submission",
    )
    proof_raw = read_bytes(safe_path(receipt["prerequisites_path"]), MAX_FILE)
    require(digest(proof_raw) == receipt["prerequisites_sha256"], "Teacher fit prerequisite proof changed")
    proof = json.loads(proof_raw)
    require(
        proof.get("schema") == "quest-sdsc-teacher-fit-prerequisites-v1"
        and proof.get("task") == receipt["task"]
        and proof.get("mode") == TEACHER_FIT_TASKS[receipt["task"]]
        and proof.get("target") == {key: receipt[key] for key in ("run_id", "code_sha256", "intent_id")}
        and all(
            proof.get(key) == receipt.get(key)
            for key in (*TEACHER_FIT_BINDINGS, "teacher_job_id", "preflight_job_id")
        )
        and receipt.get("teacher_job_id") == "54493015",
        "Teacher fit prerequisite identity differs",
    )
    if TEACHER_FIT_TASKS[receipt["task"]] == "full-fit":
        require(
            result.get("same_world_resume_prerequisite_job_id") == receipt.get("preflight_job_id"),
            "Full teacher fit borrowed an unrelated resume prerequisite",
        )
    root = safe_path(receipt["result_dir"])
    expected = contract.checkpoint_records(root / "artifacts", result)
    records = published.get("files")
    require(isinstance(records, list), "Missing teacher fit publication inventory")
    by_path = {}
    for row in records:
        require(
            isinstance(row, dict) and isinstance(row.get("path"), str) and row["path"] not in by_path,
            "Invalid/duplicate teacher fit publication record",
        )
        by_path[row["path"]] = row
    for name, row in expected.items():
        relative = "artifacts/" + name
        require(by_path.get(relative) == dict(row, path=relative), "Teacher fit publication binding differs")
    scope = ("artifacts/checkpoints/",)
    require(
        {name for name in by_path if name.startswith(scope)}
        == {"artifacts/" + name for name in expected if name.startswith("checkpoints/")},
        "Teacher fit publication contains unlisted checkpoints",
    )


def verify_result(receipt):
    try:
        root = safe_path(receipt["result_dir"])
        published = read_json(root / "receipt.json")
        task = receipt.get("task", "gpu-smoke")
        require(task in TASK_SCRIPTS, "Unsupported result task")
        preflight = task == "qwen3-v2-preflight"
        teacher = task == "qwen3-v2-teacher-prepare"
        probe = task in DIAGNOSTIC_TASKS
        calibration = task == "qwen3-v2-g0-calibration"
        qualification = task == TEACHER_QUALIFY_TASK
        student = task in STUDENT_TASKS
        model_task = preflight or teacher or calibration or probe or qualification or student
        result_name = TASK_RESULTS[task]
        result_bytes = read_bytes(root / result_name)
        result = json.loads(result_bytes)
        for value in (published, result):
            require(
                all(str(value.get(key)) == str(receipt[key]) for key in ("run_id", "job_id", "code_sha256")),
                "Result identity mismatch",
            )
        require(result.get("container") == receipt.get("container"), "Result container identity mismatch")
        if model_task:
            require(
                all(
                    value.get("task") == task and value.get("hf_home") == receipt.get("hf_home")
                    for value in (published, result)
                ),
                "Qwen3 task or model-cache identity mismatch",
            )
            require(
                (not preflight or result.get("world_size") == 2)
                and result.get("g0_passed") is False
                and result.get("execution_class_certified") is False,
                "Preparation/preflight is not evidence for G0 or execution-class certification",
            )
        if task in PROVENANCE_TASKS:
            require(
                all(
                    value.get(key) == receipt.get(key)
                    for value in (published, result)
                    for key in PROVENANCE_BINDINGS
                ),
                "Teacher result provenance identity mismatch",
            )
        if teacher:
            require(
                result.get("partial_attempt_ledger_guaranteed") is False and result.get("resumable") is False,
                "Teacher preparation cannot claim guaranteed partial ledger or resume",
            )
        if probe and task not in TEACHER_FIT_TASKS:
            require(
                result.get("exploratory") is True
                and result.get("accepted_science") is False
                and result.get("full_teacher_ready") is False
                and result.get("resumable") is False,
                "Prompt diagnostics cannot claim accepted science, full teacher readiness or resume",
            )
        if task == "qwen3-v2-teacher-capability-probe":
            require(
                result.get("artifact_kind") == "teacher_capability_diagnostic"
                and result.get("readiness") is False
                and result.get("training_started") is False
                and result.get("readiness_artifact_produced") is False
                and type(result.get("metrics_passed")) is bool,
                "Capability metrics must remain explicitly diagnostic",
            )
        if task == "qwen3-v2-teacher-adapt":
            require(
                result.get("artifact_kind") == "teacher_adaptation_preflight"
                and result.get("adaptation_preflight") is True
                and result.get("readiness_artifact_produced") is False
                and result.get("student_training_started") is False
                and result.get("training_started") is True,
                "Teacher adaptation is a bounded preflight, not formal teacher readiness",
            )
            validate_adaptation_preflight_report(result)
            verify_adaptation_manifest(root, result, published)
        if task in TEACHER_FIT_TASKS:
            verify_teacher_fit_artifacts(receipt, result, published)
        if qualification:
            release, unused = release_manifest(ROOT, receipt)
            contract = teacher_qualify_contract(release)
            proof_raw = read_bytes(safe_path(receipt["prerequisites_path"]), MAX_FILE)
            claims_raw = read_bytes(safe_path(receipt["claims_path"]), MAX_FILE)
            require(
                digest(proof_raw) == receipt["prerequisites_sha256"]
                and digest(claims_raw) == receipt["claims_sha256"],
                "Qualification proof/claims changed",
            )
            proof, claims = json.loads(proof_raw), json.loads(claims_raw)
            contract.validate_report(result, proof)
            contract.validate_acceptance_file(root / "artifacts", result, claims)
            scientific = {
                "artifacts/" + name: dict(row, path="artifacts/" + name)
                for name, row in contract.validate_inventory(root / "artifacts", result).items()
            }
            require(
                all(
                    result.get(key) == receipt.get(key)
                    for key in (*PREREQUISITE_BINDINGS, *TEACHER_QUALIFY_BINDINGS)
                ),
                "Qualification result bindings differ",
            )
            inventory = contract.record_map(published.get("files"))
            require(
                sum(row["size"] for row in inventory.values()) <= contract.MAX_OUTPUT_BYTES,
                "Qualification publication exceeds its bounded output",
            )
            require(
                all(inventory.get(name) == row for name, row in scientific.items())
                and sum(row["size"] for name, row in inventory.items() if name not in scientific)
                <= contract.MAX_LOG_BYTES,
                "Qualification scientific publication differs or diagnostics exceed their bound",
            )
            for name, row in inventory.items():
                require(
                    Path(name).suffix not in {".safetensors", ".pt", ".pth", ".ckpt", ".bin"},
                    "Qualification publication duplicates weights or trainer state",
                )
                path = safe_path(root / name)
                require(
                    path.is_file()
                    and stat.S_ISREG(path.stat().st_mode)
                    and path.stat().st_size == row["size"],
                    "Qualification published evidence size differs",
                )
                if name != "worker.log":
                    verify_published_file(path, row)
        if calibration:
            require(
                all(
                    value.get(key) == receipt.get(key)
                    for value in (published, result)
                    for key in PREREQUISITE_BINDINGS
                ),
                "Calibration prerequisite identity mismatch",
            )
            require(
                result.get("pilot_passed") is False and result.get("factorial_ready") is False,
                "Calibration cannot claim pilot or factorial acceptance",
            )
            proof = read_bytes(safe_path(receipt["prerequisites_path"]), MAX_FILE)
            require(digest(proof) == receipt["prerequisites_sha256"], "Submission prerequisites changed")
            proof = json.loads(proof)
            require(
                proof.get("schema") == "quest-sdsc-calibration-prerequisites-v1"
                and proof.get("task") == task
                and proof.get("target")
                == {key: receipt[key] for key in ("run_id", "code_sha256", "intent_id")}
                and all(
                    proof["upstream"][role]["receipt"]["job_id"] == receipt[role + "_job_id"]
                    for role in ("teacher", "preflight")
                ),
                "Calibration proof identity mismatch",
            )
        if student:
            verify_student_result(receipt, result, published)
        files = published.get("files", [])
        matches = [entry for entry in files if entry.get("path") == result_name]
        require(
            len(matches) == 1
            and matches[0].get("sha256") == digest(result_bytes)
            and matches[0].get("size") == len(result_bytes),
            "Published result hash mismatch",
        )
        require(
            (model_task or result.get("infrastructure_smoke") is True)
            and result.get("passed") is True
            and type(result.get("exit_code")) is int
            and result["exit_code"] == 0
            and published.get("passed") is True
            and published.get("persisted") is True
            and published.get("persistent_read_back_verified") is True,
            "GPU task or publication did not pass",
        )
        return {"verified": True, "result_sha256": digest(result_bytes)}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"verified": False, "reason": str(error)}


def status(request):
    receipt, unused = bound_receipt(request)
    job_id = receipt["job_id"]
    user = pwd.getpwuid(os.getuid()).pw_name
    queue = run(["squeue", "--noheader", "--user=" + user, "--format=%i|%T|%R"])
    # A completed ID may be purged from squeue. Query this user's queue once,
    # then keep only the receipt-bound ID; other jobs are neither shown nor used.
    queue["stdout"] = "\n".join(
        line for line in queue["stdout"].splitlines() if line.split("|", 1)[0].strip() == job_id
    )
    accounting = run(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs=" + job_id,
            "--format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS",
        ]
    )
    records = [[field.strip() for field in line.split("|")] for line in accounting["stdout"].splitlines()]
    jobs = [fields for fields in records if len(fields) >= 3 and fields[0] == job_id]
    steps = [fields for fields in records if fields and fields[0].startswith(job_id + ".")]
    result = verify_result(receipt)
    completed = (
        queue["returncode"] == 0
        and not queue["stdout"].strip()
        and accounting["returncode"] == 0
        and len(jobs) == 1
        and jobs[0][1] == "COMPLETED"
        and jobs[0][2] == "0:0"
        and all(len(step) >= 3 and step[1] == "COMPLETED" and step[2] == "0:0" for step in steps)
    )
    return {
        "job_id": job_id,
        "run_id": receipt["run_id"],
        "task": receipt.get("task", "gpu-smoke"),
        "hf_home": receipt.get("hf_home"),
        **{key: receipt[key] for key in PROVENANCE_BINDINGS if key in receipt},
        **{key: receipt[key] for key in PREREQUISITE_BINDINGS if key in receipt},
        **{key: receipt[key] for key in TEACHER_FIT_BINDINGS if key in receipt},
        **{key: receipt[key] for key in TEACHER_QUALIFY_BINDINGS if key in receipt},
        **{key: receipt[key] for key in STUDENT_BINDINGS if key in receipt},
        "queue": queue,
        "accounting": accounting,
        "result": result,
        "success": completed and result["verified"],
        "state": jobs[0][1]
        if len(jobs) == 1 and queue["returncode"] == 0 and accounting["returncode"] == 0
        else "UNKNOWN",
    }


def bounded_tail(path, lines=100):
    safe_path(path)
    if not path.exists():
        return None
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "Log is not a regular file")
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        return b"\n".join(stream.read(65536).splitlines()[-lines:]) + b"\n"


def selected_files(receipt, directory, lines):
    result_dir = safe_path(receipt["result_dir"])
    report_name = TASK_RESULTS[receipt.get("task", "gpu-smoke")]
    files = [
        ("slurm-" + receipt["job_id"] + ".out", directory / ("slurm-" + receipt["job_id"] + ".out"), True),
        ("slurm-" + receipt["job_id"] + ".err", directory / ("slurm-" + receipt["job_id"] + ".err"), True),
        ("worker.log", result_dir / "worker.log", True),
        ("control-result.json", directory / "control-result.json", False),
        (report_name, result_dir / report_name, False),
        ("receipt.json", result_dir / "receipt.json", False),
    ]
    if receipt.get("task") == "qwen3-v2-teacher-adapt":
        files.extend(
            (name, result_dir / "artifacts" / name, False) for name in sorted(TEACHER_ADAPT_SMALL_RESULTS)
        )
    if receipt.get("task") in TEACHER_FIT_TASKS:
        files.extend(
            (name, result_dir / "artifacts" / name, False) for name in sorted(TEACHER_FIT_SMALL_RESULTS)
        )
    if receipt.get("task") == TEACHER_QUALIFY_TASK:
        files.extend(
            (name, result_dir / "artifacts" / name, False) for name in sorted(TEACHER_QUALIFY_SMALL_RESULTS)
        )
    if receipt.get("task") == "qwen3-v2-adapted-calibration":
        files.extend(
            (name, result_dir / "artifacts" / name, True) for name in ("train.log", "export.log")
        )
    return files


def logs(request):
    receipt, directory = bound_receipt(request)
    lines = request.get("lines", 100)
    require(type(lines) is int and 1 <= lines <= 200, "Log lines must be between 1 and 200")
    output = {}
    for name, path, tail in selected_files(receipt, directory, lines):
        if tail:
            value = bounded_tail(path, lines)
            if value is not None:
                output[name] = value.decode("utf-8", "replace")
    return {"job_id": receipt["job_id"], "logs": output}


def fetch(request):
    receipt, directory = bound_receipt(request)
    files = []
    skipped = []
    total = 0
    task = receipt.get("task", "gpu-smoke")
    optional = (
        TEACHER_ADAPT_SMALL_RESULTS
        if task == "qwen3-v2-teacher-adapt"
        else TEACHER_FIT_SMALL_RESULTS
        if task in TEACHER_FIT_TASKS
        else TEACHER_QUALIFY_SMALL_RESULTS
        if task == TEACHER_QUALIFY_TASK
        else set()
    )
    optional = optional - {"checkpoint-manifest.json", "train-metrics.jsonl"}
    for name, path, tail in selected_files(receipt, directory, 200):
        safe_path(path)
        if not path.exists():
            continue
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "Fetch input is not a regular file")
        size = path.stat().st_size
        if name in optional and (size > MAX_FETCH_FILE or total + size > MAX_FETCH_TOTAL):
            skipped.append(
                {
                    "path": name,
                    "size": size,
                    "reason": "optional_file_exceeds_1MiB_limit"
                    if size > MAX_FETCH_FILE
                    else "optional_file_exceeds_total_fetch_limit",
                }
            )
            continue
        value = bounded_tail(path, 200) if tail else read_bytes(path)
        total += len(value)
        require(total <= MAX_FETCH_TOTAL, "Fetch exceeds total limit")
        files.append(
            {
                "path": name,
                "sha256": digest(value),
                "size": len(value),
                "data_b64": base64.b64encode(value).decode("ascii"),
                "tail_only": tail,
            }
        )
    return {
        "job_id": receipt["job_id"],
        "run_id": receipt["run_id"],
        "code_sha256": receipt["code_sha256"],
        "task": receipt.get("task", "gpu-smoke"),
        "hf_home": receipt.get("hf_home"),
        **{key: receipt[key] for key in PROVENANCE_BINDINGS if key in receipt},
        **{key: receipt[key] for key in PREREQUISITE_BINDINGS if key in receipt},
        **{key: receipt[key] for key in TEACHER_FIT_BINDINGS if key in receipt},
        "files": files,
        "skipped": skipped,
        "total_bytes": total,
        "result": verify_result(receipt),
    }


def cancel(request):
    require(request.get("authorized") is True, "Explicit cancellation authorization required")
    receipt, unused = bound_receipt(request)
    result = run(["scancel", "--", receipt["job_id"]])
    require(result["returncode"] == 0, "Cancellation failed; inspect this job before deciding another action")
    return {"job_id": receipt["job_id"], "cancel": result}


def dispatch(request, stream=None):
    require(pwd.getpwuid(os.getuid()).pw_name == REMOTE_USER, "Unexpected remote user")
    require(bool(os.environ.get("SSH_CONNECTION")), "Remote operations require an SSH connection")
    require(isinstance(request, dict), "Request must be an object")
    root_path(request)
    if request.get("action") == "upload":
        return upload(request, stream if stream is not None else sys.stdin.buffer)
    actions = {
        "check": check,
        "submit": submit,
        "reconcile": reconcile,
        "status": status,
        "logs": logs,
        "fetch": fetch,
        "cancel": cancel,
    }
    require(request.get("action") in actions, "Unknown action")
    return actions[request["action"]](request)


def main():
    try:
        require(len(sys.argv) == 2, "One JSON request is required")
        result = dispatch(json.loads(sys.argv[1]))
        print(json.dumps({"ok": True, **result}, sort_keys=True, ensure_ascii=False))
    except Exception as error:
        # A malformed request or failed bounded command gets one JSON error, never
        # a traceback containing caller data. KeyboardInterrupt/SystemExit propagate.
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
