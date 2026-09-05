#!/usr/bin/python3
"""Prepare the immutable login-node runtime and MIB checkout for Qwen3-v2 G0.

This helper is never a scheduler entrypoint and never touches a GPU.  It builds
the complete Python environment, verifies the existing offline model cache,
pins the external MIB/EAP-IG sources, and atomically publishes both read-only
trees under OPD-owned scratch storage.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


SYSTEM_PYTHON = Path("/usr/bin/python3.12")
SYSTEM_PYTHON_SHA256 = "848c64ae0635d363f8bbfc768f94a3be497c0d51acd28cd5087e6e8a13c44801"
BASE_RUNTIME = Path("/scr/del6500/OPD/envs/qwen3-v2-gpu-preflight-v1")
RUNTIME = Path("/scr/del6500/OPD/envs/qwen3-v2-g0-v1")
MIB_REPOSITORY = Path("/scr/del6500/OPD/vendor/MIB-circuit-track-v1")
MIB_URL = "https://github.com/hannamw/MIB-circuit-track.git"
MIB_REVISION = "b759df34433c9e31043ba9e02908ce0bf20e894f"
EAP_URL = "https://github.com/hannamw/EAP-IG.git"
HF_HOME = Path("/scr/del6500/OPD/cache/huggingface")
MODELS = (
    ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)
PYPI_PACKAGES = (
    "accelerate==1.10.1",
    "datasets==4.0.0",
    "huggingface-hub==0.36.2",
    "matplotlib==3.10.5",
    "nvidia-nccl-cu12==2.27.3",
    "numpy==1.26.4",
    "omegaconf==2.3.0",
    "pandas==2.3.2",
    "pyarrow==21.0.0",
    "pydantic==2.11.7",
    "pyyaml==6.0.2",
    "safetensors==0.5.3",
    "scipy==1.16.1",
    "statsmodels==0.14.6",
    "tabulate==0.9.0",
    "tokenizers==0.22.0",
    "transformer-lens==2.16.1",
    "transformers==4.56.2",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(*command: str, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=environment)


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ("/usr/bin/git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _offline_check_script(repository: Path) -> str:
    return (
        "import importlib.metadata as metadata\n"
        "import pathlib\n"
        "import sys\n"
        "import torch\n"
        "from transformers import AutoConfig, AutoTokenizer\n"
        f"repository = pathlib.Path({str(repository)!r})\n"
        "sys.path.insert(0, str(repository / 'EAP-IG'))\n"
        "sys.path.insert(0, str(repository))\n"
        "from eap.attribute import attribute\n"
        "from eap.attribute_node import attribute_node\n"
        "from eap.graph import Graph\n"
        "from transformer_lens import HookedTransformer\n"
        "assert attribute and attribute_node and Graph and HookedTransformer\n"
        "assert torch.__version__.startswith('2.8.0+cu128'), torch.__version__\n"
        "assert torch.version.cuda == '12.8', torch.version.cuda\n"
        "assert metadata.version('nvidia-nccl-cu12') == '2.27.3'\n"
        f"for repo_id, revision in {MODELS!r}:\n"
        "    AutoConfig.from_pretrained(repo_id, revision=revision, "
        "local_files_only=True, trust_remote_code=False)\n"
        "    AutoTokenizer.from_pretrained(repo_id, revision=revision, "
        "local_files_only=True, trust_remote_code=False)\n"
    )


def _make_read_only(root: Path) -> None:
    for current, directories, files in os.walk(root):
        for name in directories:
            (Path(current) / name).chmod(0o550)
        for name in files:
            path = Path(current) / name
            path.chmod(0o550 if path.stat().st_mode & 0o111 else 0o440)
    root.chmod(0o550)


def _make_writable(root: Path) -> None:
    for current, directories, files in os.walk(root):
        for name in directories:
            (Path(current) / name).chmod(0o750)
        for name in files:
            path = Path(current) / name
            path.chmod(0o750 if path.stat().st_mode & 0o111 else 0o640)
    root.chmod(0o750)


def prepare() -> None:
    if not SYSTEM_PYTHON.is_file() or SYSTEM_PYTHON.is_symlink():
        raise RuntimeError("fixed system Python must be a regular non-symlink file")
    if _sha256_file(SYSTEM_PYTHON) != SYSTEM_PYTHON_SHA256:
        raise RuntimeError("fixed system Python digest differs from the deployment contract")
    base_python = BASE_RUNTIME / "bin" / "python"
    if (
        not base_python.is_file()
        or base_python.is_symlink()
        or _sha256_file(base_python) != SYSTEM_PYTHON_SHA256
    ):
        raise RuntimeError("validated GPU-preflight base runtime is unavailable")
    for target in (RUNTIME, MIB_REPOSITORY):
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"refusing to replace existing immutable path: {target}")
        target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    HF_HOME.mkdir(mode=0o750, parents=True, exist_ok=True)
    runtime_stage = Path(tempfile.mkdtemp(prefix=".qwen3-v2-g0-runtime-", dir=RUNTIME.parent))
    mib_stage = Path(tempfile.mkdtemp(prefix=".mib-circuit-track-", dir=MIB_REPOSITORY.parent))
    published_mib = False
    try:
        shutil.rmtree(runtime_stage)
        shutil.rmtree(mib_stage)
        shutil.copytree(BASE_RUNTIME, runtime_stage, symlinks=True)
        _make_writable(runtime_stage)
        python = runtime_stage / "bin" / "python"
        if python.is_symlink() or not stat.S_ISREG(python.stat().st_mode):
            raise RuntimeError("venv --copies did not create a regular Python executable")
        if _sha256_file(python) != SYSTEM_PYTHON_SHA256:
            raise RuntimeError("copied venv Python differs from the reviewed interpreter")
        _run(
            str(python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--index-url",
            "https://pypi.org/simple",
            *PYPI_PACKAGES,
        )
        _run("/usr/bin/git", "clone", "--no-checkout", MIB_URL, str(mib_stage))
        _run("/usr/bin/git", "-C", str(mib_stage), "checkout", "--detach", MIB_REVISION)
        _run("/usr/bin/git", "-C", str(mib_stage), "submodule", "init")
        _run(
            "/usr/bin/git",
            "-C",
            str(mib_stage),
            "config",
            "submodule.EAP-IG.url",
            EAP_URL,
        )
        _run(
            "/usr/bin/git",
            "-C",
            str(mib_stage),
            "submodule",
            "update",
            "--recursive",
            "--checkout",
        )
        if _git_revision(mib_stage) != MIB_REVISION:
            raise RuntimeError("staged MIB checkout differs from its reviewed revision")
        submodules = subprocess.run(
            ("/usr/bin/git", "-C", str(mib_stage), "submodule", "status", "--recursive"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if not submodules or any(not row or row[0] != " " for row in submodules):
            raise RuntimeError("staged MIB submodule tree is incomplete")
        online_environment = {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_HOME / "hub"),
            "PATH": "/usr/bin:/bin",
        }
        offline_environment = {
            **online_environment,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        _run(
            str(python),
            "-I",
            "-c",
            _offline_check_script(mib_stage),
            environment=offline_environment,
        )
        _make_read_only(runtime_stage)
        _make_read_only(mib_stage)
        os.rename(mib_stage, MIB_REPOSITORY)
        published_mib = True
        os.rename(runtime_stage, RUNTIME)
    except BaseException:
        if runtime_stage.exists() and runtime_stage.parent == RUNTIME.parent:
            shutil.rmtree(runtime_stage)
        if mib_stage.exists() and mib_stage.parent == MIB_REPOSITORY.parent:
            shutil.rmtree(mib_stage)
        if published_mib and MIB_REPOSITORY.parent == Path("/scr/del6500/OPD/vendor"):
            shutil.rmtree(MIB_REPOSITORY)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform one networked login-node preparation; never allocate a GPU",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute is required")
    prepare()
    print(RUNTIME)
    print(MIB_REPOSITORY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
