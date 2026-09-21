#!/usr/bin/env python3
"""Bounded SDSC operations; sent over the authenticated SSH control connection.

This module uses only Python 3.8+ stdlib. It is not a daemon or local scheduler.
Every Slurm subprocess in this file executes on the SSH destination only.
"""

import base64
import contextlib
import datetime as dt
import hashlib
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
    "qwen3-v2-g0-calibration": "sdsc_calibration_job.sh",
}
TASK_WORKERS = {
    "gpu-smoke": "sdsc_smoke.py",
    "qwen3-v2-preflight": "sdsc_training_preflight.py",
    "qwen3-v2-teacher-prepare": "sdsc_teacher_prepare.py",
    "qwen3-v2-teacher-prompt-probe": "sdsc_teacher_probe.py",
    "qwen3-v2-teacher-capability-probe": "sdsc_teacher_capability.py",
    "qwen3-v2-g0-calibration": "sdsc_g0_calibration.py",
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
PROVENANCE_BINDINGS = ("provenance_dir", "provenance_manifest_sha256", "science_git_head", "bundle_sha256")
PROVENANCE_TASKS = {"qwen3-v2-teacher-prepare", "qwen3-v2-g0-calibration"}
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
    expected = {
        "account": "nwu181",
        "partition": "nairr-gpu-shared",
        "qos": "nairr-gpu-shared-normal",
        "gpu_type": "h100",
        "gpus": 2 if preflight or calibration else 1,
        "cpus": 24 if preflight or teacher or calibration or probe else 4,
        "mem_gib": 192 if preflight or teacher or calibration or probe else 16,
    }
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
    limit = 1800 if probe else 14400 if teacher else 7200 if preflight or calibration else 300
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
    if intent.get("task") == "qwen3-v2-g0-calibration":
        receipt.update({key: intent[key] for key in PREREQUISITE_BINDINGS})
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
        "--signal=B:TERM@300" if task == "qwen3-v2-g0-calibration" else "--signal=B:TERM@60",
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
    return argv


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
    provenance_task = task in PROVENANCE_TASKS
    model_task = preflight or provenance_task or task in DIAGNOSTIC_TASKS
    require(not model_task or request.get("container") is None, "Qwen3 tasks forbid container arguments")
    require(model_task or request.get("hf_home") is None, "Smoke does not use a model cache")
    require(
        provenance_task or all(request.get(key) is None for key in PROVENANCE_BINDINGS),
        "This task does not accept provenance arguments",
    )
    require(
        calibration or all(request.get(key) is None for key in PREREQUISITE_BINDINGS),
        "Only calibration accepts upstream prerequisites",
    )
    if calibration:
        require(
            request.get("prerequisites_path") is None and request.get("prerequisites_sha256") is None,
            "Calibration prerequisites must be generated from live remote verification",
        )
        teacher_job = identifier(request.get("teacher_job_id"), "job_id")
        preflight_job = identifier(request.get("preflight_job_id"), "job_id")
        require(teacher_job != preflight_job, "Two distinct upstream jobs are required")
    intent_id = identifier(request.get("intent_id"), "intent_id")
    release, manifest = release_manifest(root, request)
    required_files = {"tools/" + TASK_SCRIPTS[task], "tools/" + TASK_WORKERS[task]}
    if task in DIAGNOSTIC_TASKS:
        required_files.update({"tools/sdsc_teacher_prepare.py", "tools/sdsc_training_preflight.py"})
    if provenance_task:
        required_files.add("tools/sdsc_provenance.py")
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
    if calibration:
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
    ):
        require(receipt.get(key) == intent.get(key), "Receipt binding mismatch")
    require(receipt.get("task", "gpu-smoke") == intent.get("task", "gpu-smoke"), "Receipt task mismatch")
    return receipt, directory


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
        model_task = preflight or teacher or calibration or probe
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
        if probe:
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
    return [
        ("slurm-" + receipt["job_id"] + ".out", directory / ("slurm-" + receipt["job_id"] + ".out"), True),
        ("slurm-" + receipt["job_id"] + ".err", directory / ("slurm-" + receipt["job_id"] + ".err"), True),
        ("worker.log", result_dir / "worker.log", True),
        ("control-result.json", directory / "control-result.json", False),
        (report_name, result_dir / report_name, False),
        ("receipt.json", result_dir / "receipt.json", False),
    ]


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
    total = 0
    for name, path, tail in selected_files(receipt, directory, 200):
        safe_path(path)
        if not path.exists():
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
        "files": files,
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
