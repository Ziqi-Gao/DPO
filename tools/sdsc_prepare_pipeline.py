#!/usr/bin/env python3
"""One-shot, login-only preparation of the separate pilot runtime and pinned MIB.

Run on Expanse through the existing SSH master. No GPU or Slurm calls. Existing
targets are verified, never repaired or replaced. A partial install fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/expanse/lustre/projects/nwu181/zgao12/OPD")
BASE = ROOT / "envs/qwen3-v2-g0-py31213-cu128-v1"
PILOT = ROOT / "envs/qwen3-v2-pilot-trl0222-overlay-v1"
MIB = ROOT / "vendor/MIB-circuit-track-b759df34433c9e31043ba9e02908ce0bf20e894f"
REVISION = "b759df34433c9e31043ba9e02908ce0bf20e894f"
PINS = {
    "accelerate": "1.10.1",
    "datasets": "4.0.0",
    "huggingface-hub": "0.36.2",
    "matplotlib": "3.10.5",
    "nvidia-nccl-cu12": "2.27.3",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "pandas": "2.3.2",
    "pyarrow": "21.0.0",
    "pydantic": "2.11.7",
    "PyYAML": "6.0.2",
    "safetensors": "0.5.3",
    "scipy": "1.16.1",
    "statsmodels": "0.14.6",
    "tabulate": "0.9.0",
    "tokenizers": "0.22.0",
    "torch": "2.8.0+cu128",
    "transformer-lens": "2.16.1",
    "transformers": "4.56.2",
    "trl": "0.22.2",
}


def run(argv, *, env=None, timeout=3600):
    print(json.dumps({"command": argv}), flush=True)
    return subprocess.run(
        argv,
        check=True,
        env=env,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-prepared",
        action="store_true",
        help="verify inspected installed targets only; never reinstall or fetch",
    )
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") or not str(Path.home()).startswith("/home/zgao12"):
        raise ValueError("Preparation requires the authenticated Expanse login user, outside allocations")
    bootstrap = ROOT / "bootstrap/pipeline-v2"
    bootstrap.mkdir(parents=True, exist_ok=True)
    evidence = bootstrap / "environment.json"
    if evidence.exists():
        print(evidence.read_text())
        return
    claim = bootstrap / "prepare.claim"
    if args.verify_prepared:
        if not claim.is_file() or not PILOT.is_dir() or not MIB.is_dir():
            raise ValueError("Inspected completed installation targets and original claim required")
        old_pid = int(claim.read_text())
        if Path("/proc").joinpath(str(old_pid)).exists():
            raise ValueError("Original preparer is still alive; do not run competing verification")
    else:
        with claim.open("x") as f:
            f.write(str(os.getpid()))
        if PILOT.exists() or MIB.exists():
            raise ValueError(
                "Fresh targets required; partial preparation needs inspection, never blind retry"
            )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(ROOT / "tmp"),
        PIP_NO_INPUT="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        HF_HOME=str(ROOT / "cache/huggingface"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
    )
    python = str((PILOT / "bin/python").resolve(strict=True)) if args.verify_prepared else None
    git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]
    if not args.verify_prepared:
        print(
            run(
                [
                    str((BASE / "bin/python").resolve(strict=True)),
                    "-I",
                    "-m",
                    "venv",
                    "--copies",
                    "--system-site-packages",
                    str(PILOT),
                ],
                env=env,
            ),
            flush=True,
        )
        constraints = bootstrap / "constraints.txt"
        constraints.write_text("".join(f"{k}=={v}\n" for k, v in PINS.items()))
        python = str((PILOT / "bin/python").resolve(strict=True))
        env["PIP_REQUIRE_VIRTUALENV"] = "true"
        print(
            run(
                [
                    python,
                    "-I",
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--index-url",
                    "https://pypi.org/simple",
                    "--constraint",
                    str(constraints),
                    "trl==0.22.2",
                ],
                env=env,
            ),
            flush=True,
        )
        print(run([python, "-I", "-m", "pip", "check"], env=env), flush=True)
        MIB.parent.mkdir(parents=True, exist_ok=True)
        git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]
        print(
            run(
                [
                    *git,
                    "clone",
                    "--no-checkout",
                    "https://github.com/hannamw/MIB-circuit-track.git",
                    str(MIB),
                ],
                env=env,
            ),
            flush=True,
        )
        for args in (
            ["checkout", "--detach", REVISION],
            ["submodule", "init"],
            ["config", "submodule.EAP-IG.url", "https://github.com/hannamw/EAP-IG.git"],
            ["submodule", "update", "--init", "--recursive", "--checkout"],
        ):
            print(run([*git, "-C", str(MIB), *args], env=env), flush=True)
    else:
        print(run([python, "-I", "-m", "pip", "check"], env=env, timeout=120), flush=True)
    head = run([*git, "-C", str(MIB), "rev-parse", "HEAD"], env=env).strip()
    modules = run([*git, "-C", str(MIB), "submodule", "status", "--recursive"], env=env)
    if head != REVISION or not modules or any(not line.startswith(" ") for line in modules.splitlines()):
        raise ValueError("MIB/submodule revision differs")
    check = """import sys,json,platform,hashlib,importlib.metadata as m
from pathlib import Path
import torch
from transformers import AutoConfig,AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
sys.path[:0]=[sys.argv[1],sys.argv[1]+'/EAP-IG/src']
from eap.attribute import attribute
from eap.attribute_node import attribute_node
from eap.graph import Graph
from transformer_lens import HookedTransformer
pins=json.loads(sys.argv[2]); actual={k:m.version(k) for k in pins}
assert actual==pins,(actual,pins)
assert platform.python_version()=='3.12.13'
assert torch.version.cuda=='12.8'
models=[('Qwen/Qwen3-1.7B','70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
        ('Qwen/Qwen3-8B','b968826d9c46dd6066d109eabc6255188de91218')]
for model,revision in models:
    AutoConfig.from_pretrained(model,revision=revision,local_files_only=True,trust_remote_code=False)
    AutoTokenizer.from_pretrained(model,revision=revision,local_files_only=True,trust_remote_code=False)
print(json.dumps({'python':platform.python_version(),'python_path':sys.executable,
 'python_sha256':hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
 'base_prefix':sys.base_prefix,'packages':actual,'cuda_runtime':torch.version.cuda,'gpu_verified':False}))
"""
    identity = json.loads(
        run([python, "-I", "-B", "-c", check, str(MIB), json.dumps(PINS)], env=env, timeout=600).splitlines()[
            -1
        ]
    )
    if run(
        [
            *git,
            "-C",
            str(MIB),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        env=env,
    ):
        raise ValueError("Pinned MIB source became dirty")
    image = Path(
        "/expanse/projects/qstore/installs/containers/singularity/Expanse-Air/pytorch/pytorch-nvcr-25.03.sif"
    )
    records = []
    # Public fresh repository metadata is retained for genuine rev-parse and submodules.
    # Never copy user Git configuration, hooks, SSH files or credentials.
    for folder, dirs, files in os.walk(MIB, followlinks=False):
        dirs[:] = [name for name in dirs if name not in {"hooks", "__pycache__"}]
        for name in sorted(files):
            p = Path(folder) / name
            if p.is_symlink() or not p.is_file():
                raise ValueError("Unexpected MIB symlink/non-regular source")
            records.append(
                {
                    "path": str(p.relative_to(MIB)),
                    "size": p.stat().st_size,
                    "sha256": digest(p),
                    "mode": 0o755 if p.stat().st_mode & 0o111 else 0o644,
                }
            )
    manifest = {
        "schema": "quest-sdsc-public-vendor-v1",
        "head": head,
        "submodules": modules,
        "files": sorted(records, key=lambda r: r["path"]),
    }
    vendor = bootstrap / "mib-manifest.json"
    vendor.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    result = {
        "schema": "quest-sdsc-pipeline-runtime-v1",
        "preparation_threads": {
            name: env[name]
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "passed": True,
        "runtime": identity,
        "mib": {"path": str(MIB), "head": head, "manifest": str(vendor), "manifest_sha256": digest(vendor)},
        "container": {
            "runtime": str(Path("/cm/local/apps/singularitypro/4.1/bin/singularity").resolve(strict=True)),
            "image": str(image.resolve(strict=True)),
            "size": image.stat().st_size,
            "mtime_ns": image.stat().st_mtime_ns,
        },
    }
    with evidence.open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
