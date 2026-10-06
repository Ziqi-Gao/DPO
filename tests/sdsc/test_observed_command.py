"""Real local child processes only; no scheduler or network invocation."""

import base64
import hashlib
import importlib.util
import json
import signal
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "_observed_command_test", ROOT / "tools/sdsc_observed_command.py"
)
observed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observed)


def python(code):
    return [sys.executable, "-I", "-B", "-c", code]


def raw(result, name):
    row = result[name]
    data = base64.b64decode(row["base64"], validate=True)
    assert len(data) == row["retained_bytes"]
    assert hashlib.sha256(data).hexdigest() == row["retained_sha256"]
    return data


def test_timeout_preserves_ack_and_error_and_reaps_direct_child():
    argv = python("import os,time;os.write(1,b'987654321\\n');os.write(2,b'partial error\\n');time.sleep(10)")
    result = observed.run_observed(argv, timeout=0.3)
    assert raw(result, "stdout") == b"987654321\n"
    assert raw(result, "stderr") == b"partial error\n"
    assert result["timed_out"] and result["termination_requested"] and result["child_reaped"]
    assert result["returncode"] == -signal.SIGKILL
    assert not result["output_complete"] and not result["retry_attempted"]
    assert result["argv"] == argv
    assert result["started_at"] <= result["ended_at"]
    assert result["elapsed_seconds"] < 4
    assert "job_id" not in result and "passed" not in result


def test_output_cap_is_combined_and_preserves_exact_retained_bytes():
    result = observed.run_observed(
        python("import os,time;os.write(1,b'x'*1048576);time.sleep(10)"), timeout=3, max_output_bytes=128
    )
    assert result["output_limited"] and result["child_reaped"]
    assert not result["output_complete"]
    assert raw(result, "stdout") == b"x" * 128
    assert result["stdout"]["truncated"]
    assert result["stdout"]["observed_bytes"] > 128
    assert (
        result["stdout"]["observed_sha256"]
        == hashlib.sha256(b"x" * result["stdout"]["observed_bytes"]).hexdigest()
    )
    assert sum(result[name]["retained_bytes"] for name in ("stdout", "stderr")) <= 128


def test_normal_nonzero_and_invalid_utf8_are_preserved():
    result = observed.run_observed(
        python("import os,sys;os.write(1,b'\\xff\\x00');os.write(2,b'no\\n');sys.exit(7)"), timeout=3
    )
    assert result["returncode"] == 7 and result["child_reaped"]
    assert result["output_complete"] and not result["timed_out"]
    assert raw(result, "stdout") == b"\xff\x00"
    assert raw(result, "stderr") == b"no\n"
    json.dumps(result)


def test_stdin_eof_and_environment_filter(monkeypatch):
    values = {
        "SBATCH_WAIT": "1",
        "SQUEUE_FORMAT": "fixture",
        "SACCT_FORMAT": "fixture",
        "SLURM_CONF": "fixture-only",
        "OPD_OBSERVED_FIXTURE": "retained",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    code = (
        "import json,os,sys;print(json.dumps({'stdin':sys.stdin.buffer.read().decode(),"
        "'env':{k:os.environ.get(k) for k in " + repr(list(values)) + "}}))"
    )
    result = observed.run_observed(python(code), timeout=3)
    data = json.loads(raw(result, "stdout"))
    assert result["returncode"] == 0 and result["output_complete"]
    assert data == {
        "stdin": "",
        "env": {
            "SBATCH_WAIT": None,
            "SQUEUE_FORMAT": None,
            "SACCT_FORMAT": None,
            "SLURM_CONF": "fixture-only",
            "OPD_OBSERVED_FIXTURE": "retained",
        },
    }


def test_utc_override_is_local_to_read_query(monkeypatch, tmp_path):
    monkeypatch.setenv("TZ", "Pacific/Honolulu")
    script = tmp_path / "squeue"
    script.write_text("#!" + sys.executable + "\nimport os\nprint(os.environ['TZ'])\n")
    script.chmod(0o700)
    result = observed.run_observed([str(script)], timeout=3, utc_query=True)
    assert raw(result, "stdout") == b"UTC\n"
    result = observed.run_observed(python("import os;print(os.environ['TZ'])"), timeout=3)
    assert raw(result, "stdout") == b"Pacific/Honolulu\n"


@pytest.mark.parametrize(
    "argv", [["sbatch", "--test-only"], ["scontrol", "update"], ["/bin/sh", "-c", "true"]]
)
def test_utc_override_rejects_nonqueries_before_spawn(monkeypatch, argv):
    def forbidden(*args, **kwargs):
        pytest.fail("must reject before launching any process")

    monkeypatch.setattr(observed.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="UTC"):
        observed.run_observed(argv, timeout=1, utc_query=True)


def test_launch_error_is_an_observation_not_a_retry(tmp_path):
    result = observed.run_observed([str(tmp_path / "does-not-exist")], timeout=1)
    assert not result["spawned"] and not result["child_reaped"]
    assert result["returncode"] is None and not result["output_complete"]
    assert "FileNotFoundError" in result["error"]
    assert raw(result, "stdout") == raw(result, "stderr") == b""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": True},
        {"timeout": float("nan")},
        {"timeout": 0},
        {"timeout": 1, "max_output_bytes": True},
        {"timeout": 1, "max_output_bytes": 0},
    ],
)
def test_invalid_bounds_rejected_before_spawn(monkeypatch, kwargs):
    monkeypatch.setattr(observed.subprocess, "Popen", lambda *a, **k: pytest.fail("no spawn"))
    with pytest.raises(ValueError):
        observed.run_observed(["fixture"], **kwargs)
