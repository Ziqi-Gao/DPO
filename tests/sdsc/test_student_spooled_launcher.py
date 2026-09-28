"""Slurm copies batch scripts; a launcher's own directory is not the release."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = load("sdsc_cli")
remote = load("sdsc_remote")


@pytest.fixture
def launch_fixture(tmp_path):
    # Argument metacharacters must survive as literal path text, never shell code.
    release = tmp_path / "release space;$(touch EXPANDED);`touch EXPANDED2`"
    tools = release / "source/tools"
    tools.mkdir(parents=True)
    worker = tools / "sdsc_student_job.py"
    shutil.copyfile(ROOT / "tools/sdsc_student_job.py", worker)
    python = tmp_path / "fake python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['OPD_LAUNCH_ARGUMENTS']).write_text(json.dumps(sys.argv[1:]))\n"
        "if not pathlib.Path(sys.argv[4]).is_file():\n"
        "    print('worker not found: ' + sys.argv[4], file=sys.stderr)\n"
        "    sys.exit(2)\n"
    )
    python.chmod(0o755)
    spool = tmp_path / "slurm-spool/job54504816/slurm_script"
    spool.parent.mkdir(parents=True)
    recorded = tmp_path / "arguments.json"
    arguments = [
        str(release),
        str(tmp_path / "submission;literal"),
        str(tmp_path / "results with spaces"),
        str(python),
        "fixture-run",
        "a" * 64,
        str(tmp_path / "HF$(touch CACHE_EXPANDED)"),
        str(tmp_path / "provenance space"),
        "b" * 64,
    ]
    environment = dict(os.environ, OPD_LAUNCH_ARGUMENTS=str(recorded))
    return tmp_path, worker, spool, recorded, arguments, environment


@pytest.mark.parametrize("task", ["qwen3-v2-adapted-preflight", "qwen3-v2-adapted-calibration"])
def test_actual_spooled_task_launcher_finds_accepted_worker_and_preserves_arguments(launch_fixture, task):
    tmp_path, worker, spool, recorded, arguments, environment = launch_fixture
    assert cli.TASK_SCRIPTS[task] == remote.TASK_SCRIPTS[task]
    shutil.copyfile(ROOT / "tools" / cli.TASK_SCRIPTS[task], spool)
    result = subprocess.run(
        ["/bin/bash", str(spool), *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(recorded.read_text()) == ["-I", "-B", "-u", str(worker), *arguments]
    assert worker.read_bytes() == (ROOT / "tools/sdsc_student_job.py").read_bytes()
    assert not (tmp_path / "EXPANDED").exists()
    assert not (tmp_path / "EXPANDED2").exists()
    assert not (tmp_path / "CACHE_EXPANDED").exists()


@pytest.mark.parametrize(
    "mutation", ["missing_argument", "relative_release", "relative_python", "nonexecutable_python"]
)
def test_locator_rejects_invalid_invocation_without_starting_python(launch_fixture, mutation):
    tmp_path, _, spool, recorded, arguments, environment = launch_fixture
    shutil.copyfile(ROOT / "tools" / cli.TASK_SCRIPTS["qwen3-v2-adapted-preflight"], spool)
    if mutation == "missing_argument":
        arguments.pop()
    elif mutation == "relative_release":
        arguments[0] = "relative/release"
    elif mutation == "relative_python":
        arguments[3] = "relative/python"
    else:
        Path(arguments[3]).chmod(0o644)
    result = subprocess.run(
        ["/bin/bash", str(spool), *arguments], cwd=tmp_path, env=environment, capture_output=True, timeout=10
    )
    assert result.returncode == 2
    assert not recorded.exists()
