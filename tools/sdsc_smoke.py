#!/usr/bin/env python3
"""Staged Expanse infrastructure smoke; never a scientific readiness result."""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

NETWORK_FILESYSTEMS = {"lustre", "nfs", "nfs4", "gpfs", "beegfs", "ceph", "fuse.ceph", "panfs"}


def mount_info(path: Path) -> dict:
    value = subprocess.run(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(value.stdout)["filesystems"][0]


def validate_results(path: Path, home: Path, stage: Path, run_id: str | None = None) -> dict:
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("persistent results must be a pre-existing absolute directory")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("persistent result path cannot traverse any symlink")
    path = path.resolve(strict=True)
    home = home.resolve()
    small_home_metadata = False
    if path == home or home in path.parents:
        allowed = home / "quest-runs/OPD/smoke-results"
        try:
            relative = path.relative_to(allowed)
        except ValueError:
            raise ValueError("HOME results are limited to the small smoke metadata directory") from None
        if (
            len(relative.parts) != 2
            or relative.parts[0] != run_id
            or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", p) for p in relative.parts)
        ):
            raise ValueError("HOME smoke results require the exact run_id/intent directory")
        small_home_metadata = True
    for forbidden in (stage.resolve(), Path("/tmp"), Path("/var/tmp")):
        if path == forbidden or forbidden in path.parents:
            raise ValueError("results cannot be in temporary storage")
    mount = mount_info(path)
    if mount["fstype"] not in NETWORK_FILESYSTEMS:
        raise ValueError("result storage is not a verified shared filesystem")
    fd, probe = tempfile.mkstemp(prefix=".opd-storage-probe-", dir=path)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"OPD Expanse persistence probe\n")
            stream.flush()
            os.fsync(stream.fileno())
        if Path(probe).read_bytes() != b"OPD Expanse persistence probe\n":
            raise OSError("persistent storage read-back failed")
    finally:
        Path(probe).unlink(missing_ok=True)
    return {
        **mount,
        "path": str(path),
        "device": path.stat().st_dev,
        "small_home_smoke_metadata_only": small_home_metadata,
        "write_read_probe_passed": True,
    }


def publish(path: Path, data: bytes) -> str:
    """Atomic no-clobber publication with same-filesystem fsync and read-back."""
    fd, name = tempfile.mkstemp(prefix=".publishing-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        digest = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise OSError(f"published file failed read-back: {path}")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def gpu_smoke() -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one Slurm-visible CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name.upper():
        raise RuntimeError(f"allocated GPU is not H100: {name}")
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    inputs = (torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128) % 97) / 128
    expected = inputs @ inputs.T
    device_inputs = inputs.to("cuda:0")
    torch.cuda.synchronize()
    started = time.monotonic()
    actual = device_inputs @ device_inputs.T
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    if not bool(torch.isfinite(actual).all()) or not torch.allclose(
        actual.cpu(), expected, atol=1e-4, rtol=1e-4
    ):
        raise RuntimeError("CUDA matrix multiply is nonfinite or differs from CPU reference")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != visible:
        raise RuntimeError("CUDA visibility changed during smoke")
    props = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "logical_device": 0,
        "visible_count": 1,
        "cuda_visible_devices": visible,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "memory_bytes": props.total_memory,
        "matrix_shape": [128, 128],
        "finite": True,
        "cpu_reference_matched": True,
        "compute_seconds": seconds,
    }


def validate_container(profile: dict) -> None:
    """Check the observed image metadata; this is not a cryptographic image identity."""
    if not isinstance(profile, dict) or set(profile) != {"runtime", "image", "python", "size", "mtime_ns"}:
        raise ValueError("container profile requires exactly runtime/image/python/size/mtime_ns")
    for key in ("runtime", "image", "python"):
        value = profile[key]
        if not isinstance(value, str) or not Path(value).is_absolute() or any(ord(c) < 32 for c in value):
            raise ValueError(f"container {key} must be an absolute path without control characters")
    for key in ("size", "mtime_ns"):
        if type(profile[key]) is not int or profile[key] <= 0:
            raise ValueError(f"container {key} must be a positive integer")
    if not Path(profile["runtime"]).is_file() or not os.access(profile["runtime"], os.X_OK):
        raise ValueError("container runtime is unavailable on this compute node")
    image = Path(profile["image"]).lstat()
    if (
        not stat.S_ISREG(image.st_mode)
        or image.st_size != profile["size"]
        or image.st_mtime_ns != profile["mtime_ns"]
    ):
        raise ValueError("container image is not the observed regular file (size/mtime changed)")


def container_gpu_smoke(profile: dict, stage: Path) -> tuple[dict, str]:
    validate_container(profile)
    if not stage.is_absolute() or any(character in str(stage) for character in ":,\n\r"):
        raise ValueError("container stage bind path contains a delimiter")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or any(ord(c) < 32 for c in visible):
        raise ValueError("container smoke requires explicit Slurm CUDA_VISIBLE_DEVICES")
    argv = [
        profile["runtime"],
        "exec",
        "--nv",
        "--cleanenv",
        "--no-home",
        "--bind",
        f"{stage}:{stage}:ro",
        "--pwd",
        str(stage),
        "--env",
        f"CUDA_VISIBLE_DEVICES={visible}",
        "--env",
        "SLURM_CPUS_PER_TASK=4",
        profile["image"],
        profile["python"],
        "-I",
        "-B",
        str(stage / "tools/sdsc_smoke.py"),
        "--gpu-only",
    ]
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(("SINGULARITY", "APPTAINER"))
    }
    child = subprocess.run(
        argv, capture_output=True, text=True, errors="replace", timeout=180, check=False, env=environment
    )
    stderr = child.stderr[-8192:]
    if child.returncode:
        raise RuntimeError(f"container GPU child exited {child.returncode}: {stderr}")
    if len(child.stdout) > 65536:
        raise ValueError("container GPU child exceeded its JSON output limit")
    try:
        result = json.loads(child.stdout)
    except json.JSONDecodeError:
        raise ValueError(f"container GPU child returned invalid JSON; stderr: {stderr}") from None
    if (
        not isinstance(result, dict)
        or result.get("finite") is not True
        or result.get("cpu_reference_matched") is not True
        or type(result.get("visible_count")) is not int
        or result["visible_count"] != 1
        or "H100" not in str(result.get("name", "")).upper()
        or result.get("cuda_visible_devices") != visible
    ):
        raise ValueError("container GPU evidence failed the finite/single-H100/visibility/reference contract")
    validate_container(profile)
    return result, stderr


class JobSignal(RuntimeError):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


def main(argv: list[str]) -> int:
    if len(argv) not in (7, 8):
        raise ValueError("worker must be launched by sdsc_job.sh")
    release, submission_name, result_name, python, run_id, code_hash, stage_name = argv[:7]
    submission, result_dir, stage = map(Path, (submission_name, result_name, stage_name))
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id) or Path(__file__).resolve().parent.parent != stage.resolve():
        raise ValueError("worker must execute from its staged Slurm allocation")
    started = time.monotonic()
    report = {
        "infrastructure_smoke": True,
        "passed": False,
        "run_id": run_id,
        "code_sha256": code_hash,
        "job_id": job_id,
        "node": socket.gethostname(),
        "release": release,
        "staged_source": str(stage),
        "result_path": str(result_dir),
        "python": python,
        "python_version": sys.version,
        "container": None,
        "runtime_profile": {"kind": "host-python", "python": python},
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "host_memory_env_verified": os.environ.get("SLURM_MEM_PER_NODE") == "16384",
    }

    def interrupted(signum, _frame):
        raise JobSignal(signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    persistent_ready, exit_code = False, 1
    log = ["Starting infrastructure smoke; no models, datasets, or scientific training.\n"]
    try:
        report["scratch"] = json.loads((stage / ".sdsc-stage.json").read_text())
        report["persistent_storage"] = validate_results(result_dir, Path.home(), stage, run_id)
        persistent_ready = True
        if os.environ.get("SLURM_CPUS_PER_TASK") != "4":
            raise RuntimeError("first smoke requires SLURM_CPUS_PER_TASK=4")
        if os.environ.get("SLURM_MEM_PER_NODE") not in (None, "16384"):
            raise RuntimeError("first smoke requires 16384 MiB host memory when Slurm reports it")
        if len(argv) == 8:
            profile = json.loads(argv[7])
            validate_container(profile)
            report["container"] = profile
            report["runtime_profile"] = {
                "kind": "singularity",
                "container": profile,
                "image_identity": "path-size-mtime_ns; not a cryptographic hash",
            }
            report["gpu"], child_stderr = container_gpu_smoke(profile, stage)
            if child_stderr:
                log.append("Container stderr (bounded tail):\n" + child_stderr + "\n")
        else:
            report["gpu"] = gpu_smoke()
        report["passed"], exit_code = True, 0
        log.append("Finite GPU calculation matched CPU reference.\n")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        exit_code = 128 + error.signum if isinstance(error, JobSignal) else 1
        log.append(traceback.format_exc())
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_host_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["exit_code"] = exit_code
        try:
            if not persistent_ready:
                raise RuntimeError("persistent storage was not validated; result is incomplete")
            log_bytes = "".join(log).encode()
            result_bytes = (json.dumps(report, indent=2) + "\n").encode()
            log_hash = publish(result_dir / "worker.log", log_bytes)
            result_hash = publish(result_dir / "result.json", result_bytes)
            receipt = {
                "run_id": run_id,
                "code_sha256": code_hash,
                "job_id": job_id,
                "result_sha256": result_hash,
                "worker_log_sha256": log_hash,
                "passed": report["passed"],
                "persisted": True,
                "persistent_read_back_verified": True,
                "files": [
                    {"path": "result.json", "sha256": result_hash, "size": len(result_bytes)},
                    {"path": "worker.log", "sha256": log_hash, "size": len(log_bytes)},
                ],
            }
            publish(result_dir / "receipt.json", (json.dumps(receipt, indent=2) + "\n").encode())
        except BaseException as error:
            report["passed"], exit_code = False, 1
            report["publication_error"] = f"{type(error).__name__}: {error}"
            report["exit_code"] = exit_code
        try:
            publish(submission / "control-result.json", (json.dumps(report, indent=2) + "\n").encode())
        except OSError as error:
            print(f"control metadata publication failed: {error}", file=sys.stderr)
            exit_code = 1
        print(json.dumps(report), flush=True)
    return exit_code


if __name__ == "__main__":
    if sys.argv[1:] == ["--gpu-only"]:
        print(json.dumps(gpu_smoke()), flush=True)
    else:
        sys.exit(main(sys.argv[1:]))
