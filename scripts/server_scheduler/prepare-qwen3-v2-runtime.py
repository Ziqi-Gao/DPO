#!/usr/bin/python3
"""One-shot login-node preparation for the fixed Qwen3-v2 GPU-preflight runtime.

This is not a scheduler entrypoint and never submits work.  It creates a copied
Python venv, installs the exact direct runtime dependencies, downloads the two
pinned Hugging Face snapshots before any GPU allocation, performs offline
metadata/tokenizer checks, and atomically publishes the read-only venv.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


SYSTEM_PYTHON = Path("/usr/bin/python3.12")
SYSTEM_PYTHON_SHA256 = "848c64ae0635d363f8bbfc768f94a3be497c0d51acd28cd5087e6e8a13c44801"
RUNTIME = Path("/scr/del6500/OPD/envs/qwen3-v2-gpu-preflight-v1")
HF_HOME = Path("/scr/del6500/OPD/cache/huggingface")
TMP_ROOT = Path("/scr/del6500/OPD/tmp")
MODELS = (
    ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
)
PYPI_PACKAGES = (
    "accelerate==1.10.1",
    "huggingface-hub==0.36.2",
    "numpy==1.26.4",
    "safetensors==0.5.3",
    "tokenizers==0.22.0",
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


def _download_script() -> str:
    rows = repr(MODELS)
    return (
        "from huggingface_hub import snapshot_download\n"
        f"for repo_id, revision in {rows}:\n"
        f"    snapshot_download(repo_id=repo_id, revision=revision, cache_dir={str(HF_HOME / 'hub')!r})\n"
    )


def _offline_check_script() -> str:
    rows = repr(MODELS)
    return (
        "import torch\n"
        "from transformers import AutoConfig, AutoTokenizer\n"
        "assert torch.__version__.startswith('2.8.0+cu128'), torch.__version__\n"
        "assert torch.version.cuda == '12.8', torch.version.cuda\n"
        f"for repo_id, revision in {rows}:\n"
        "    AutoConfig.from_pretrained(repo_id, revision=revision, "
        "local_files_only=True, trust_remote_code=False)\n"
        "    AutoTokenizer.from_pretrained(repo_id, revision=revision, "
        "local_files_only=True, trust_remote_code=False)\n"
    )


def prepare() -> None:
    if not SYSTEM_PYTHON.is_file() or SYSTEM_PYTHON.is_symlink():
        raise RuntimeError("fixed system Python must be a regular non-symlink file")
    if _sha256_file(SYSTEM_PYTHON) != SYSTEM_PYTHON_SHA256:
        raise RuntimeError("fixed system Python digest differs from the deployment contract")
    if RUNTIME.exists() or RUNTIME.is_symlink():
        raise RuntimeError(f"refusing to replace existing immutable runtime: {RUNTIME}")
    RUNTIME.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    HF_HOME.mkdir(mode=0o750, parents=True, exist_ok=True)
    TMP_ROOT.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".qwen3-v2-gpu-runtime-", dir=RUNTIME.parent))
    try:
        shutil.rmtree(temporary)
        _run(str(SYSTEM_PYTHON), "-m", "venv", "--copies", str(temporary))
        python = temporary / "bin" / "python"
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
            "https://download.pytorch.org/whl/cu128",
            "torch==2.8.0",
        )
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
        online_environment = {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_HOME / "hub"),
            "PATH": "/usr/bin:/bin",
        }
        _run(str(python), "-c", _download_script(), environment=online_environment)
        offline_environment = {
            **online_environment,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        _run(str(python), "-I", "-c", _offline_check_script(), environment=offline_environment)
        for root, directories, files in os.walk(temporary):
            for name in directories:
                (Path(root) / name).chmod(0o550)
            for name in files:
                path = Path(root) / name
                path.chmod(0o550 if path.stat().st_mode & 0o111 else 0o440)
        temporary.chmod(0o550)
        os.rename(temporary, RUNTIME)
    except BaseException:
        if temporary.exists() and temporary.parent == RUNTIME.parent:
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the one-shot networked login-node preparation",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("explicit --execute is required; this script never runs inside a GPU job")
    prepare()
    print(RUNTIME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
