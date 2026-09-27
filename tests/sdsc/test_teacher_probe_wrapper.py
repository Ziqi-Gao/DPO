"""Real local process/pipe regressions; no SSH, Slurm, model, or GPU."""

import ast
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools/sdsc_teacher_probe_job.sh"


def wrapper_functions():
    source = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(source)
    definitions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom | ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(definitions, str(SCRIPT), "exec"), namespace)
    return namespace


def test_interrupted_wrapper_preserves_shutdown_pipe_after_original_log_closes(tmp_path):
    helpers = wrapper_functions()
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-u", "-c", """
import os, signal
def stopped(number, frame):
    print('SHUTDOWN_EVIDENCE_AFTER_TERM', flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
print('READY', flush=True)
while True:
    signal.pause()
"""],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True, bufsize=0,
    )
    log = tmp_path / "worker.log"
    helpers.update(process=child, log=log)
    try:
        # Match the real wrapper's unwind: its with-log scope has already
        # closed when the outer exception handler invokes stop_child().
        with log.open("xb") as stream:
            assert child.stdout.readline() == b"READY\n"
            stream.write(b"READY\n")
        helpers["stop_child"]()
        assert child.returncode == 0
        assert log.read_bytes() == b"READY\nSHUTDOWN_EVIDENCE_AFTER_TERM\n"
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        child.stdout.close()


def test_real_wrapper_signal_unwinds_reader_then_saves_child_shutdown(tmp_path):
    harness = r'''
import ast, pathlib, signal, subprocess, sys
source = pathlib.Path(sys.argv[1]).read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
tree = ast.parse(source)
definitions = ast.Module(
    body=[node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))],
    type_ignores=[])
exec(compile(definitions, sys.argv[1], "exec"))
log = pathlib.Path(sys.argv[2])
process = subprocess.Popen([sys.executable, '-I', '-B', '-u', '-c', """
import signal
def stopped(number, frame):
    print('TERM_TAIL_FROM_REAL_WRAPPER_INTERRUPT', flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
print('READY', flush=True)
while True:
    signal.pause()
"""], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
signal.signal(signal.SIGTERM, interrupted)
try:
    with log.open('xb') as stream:
        stream.write(process.stdout.readline())
        stream.flush()
        print('WRAPPER_READY', flush=True)
        forward_output(process.stdout, stream)
except RuntimeError:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    result = stop_child(grace_seconds=1)
    assert result['pipe_drained'] and result['exit_code'] == 0
'''
    log = tmp_path / "worker.log"
    with subprocess.Popen(
        [sys.executable, "-I", "-B", "-u", "-c", harness, str(SCRIPT), str(log)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as wrapper:
        assert wrapper.stdout.readline() == b"WRAPPER_READY\n"
        wrapper.send_signal(signal.SIGTERM)
        output, error = wrapper.communicate(timeout=5)
        assert wrapper.returncode == 0, error.decode()
    assert log.read_bytes() == b"READY\nTERM_TAIL_FROM_REAL_WRAPPER_INTERRUPT\n"


@contextmanager
def ready_child(script):
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-u", "-c", script],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True, bufsize=0,
    )
    try:
        assert child.stdout.readline() == b"READY\n"
        yield child
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        child.stdout.close()


def test_shutdown_drains_more_than_pipe_capacity_before_waiting(tmp_path):
    helpers = wrapper_functions()
    log = tmp_path / "worker.log"
    log.write_bytes(b"READY\n")
    with ready_child(r"""
import os, signal
def stopped(number, frame):
    data = b'X' * (512 * 1024) + b'\nLAST_SHUTDOWN_EVIDENCE\n'
    while data:
        data = data[os.write(1, data):]
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
print('READY', flush=True)
while True:
    signal.pause()
""") as child:
        helpers.update(process=child, log=log)
        result = helpers["stop_child"](grace_seconds=2)
        assert child.returncode == 0
        assert not result["sigkill_sent"]
        assert result["pipe_drained"]
        assert result["omitted_bytes"] == 0
        assert log.read_bytes() == b"READY\n" + b"X" * (512 * 1024) + b"\nLAST_SHUTDOWN_EVIDENCE\n"


def test_shutdown_tail_is_bounded_and_preserves_last_evidence(tmp_path):
    helpers = wrapper_functions()
    log = tmp_path / "worker.log"
    log.write_bytes(b"READY\n")
    with ready_child(r"""
import os, signal
def stopped(number, frame):
    data = b'X' * (256 * 1024) + b'\nLAST_SHUTDOWN_EVIDENCE\n'
    while data:
        data = data[os.write(1, data):]
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
print('READY', flush=True)
while True:
    signal.pause()
""") as child:
        helpers.update(process=child, log=log)
        result = helpers["stop_child"](grace_seconds=2, tail_limit=1024)
        assert child.returncode == 0
        assert result["retained_bytes"] == 1024
        assert result["omitted_bytes"] > 250 * 1024
        assert b"shutdown log truncated:" in log.read_bytes()
        assert log.read_bytes().endswith(b"\nLAST_SHUTDOWN_EVIDENCE\n")
        assert log.stat().st_size < 1200


def test_native_block_forces_kill_and_preserves_faulthandler_stderr(tmp_path):
    helpers = wrapper_functions()
    log = tmp_path / "worker.log"
    log.write_bytes(b"READY\n")
    # A mutex self-deadlock in PyDLL holds the GIL: the Python TERM handler
    # cannot run. faulthandler's C handler still emits a stack to the real pipe.
    with ready_child(r"""
import ctypes, faulthandler, signal, sys
def deferred(number, frame):
    print('PYTHON_HANDLER_RAN', flush=True)
signal.signal(signal.SIGTERM, deferred)
faulthandler.register(signal.SIGTERM, file=sys.stderr, all_threads=True, chain=True)
library = ctypes.PyDLL(None)
mutex = ctypes.create_string_buffer(128)
assert library.pthread_mutex_init(ctypes.byref(mutex), None) == 0
assert library.pthread_mutex_lock(ctypes.byref(mutex)) == 0
print('READY', flush=True)
library.pthread_mutex_lock(ctypes.byref(mutex))
""") as child:
        helpers.update(process=child, log=log)
        time.sleep(0.05)
        result = helpers["stop_child"](grace_seconds=0.1, kill_seconds=1, drain_seconds=0.2)
        assert child.returncode == -signal.SIGKILL
        assert result["sigkill_sent"]
        assert result["pipe_drained"]
        assert result["elapsed_seconds"] < 2
        raw = log.read_bytes()
        assert b"Current thread" in raw
        assert b'File "<string>"' in raw
        assert b"PYTHON_HANDLER_RAN" not in raw


def test_normal_forwarding_preserves_exact_binary_bytes(tmp_path, capfdbinary):
    helpers = wrapper_functions()
    log = tmp_path / "worker.log"
    with subprocess.Popen(
        [sys.executable, "-I", "-B", "-u", "-c", "import os; os.write(1, b'first\\nlast\\xff')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
    ) as child:
        with log.open("xb") as stream:
            helpers["forward_output"](child.stdout, stream)
        assert child.wait(timeout=5) == 0
    assert log.read_bytes() == b"first\nlast\xff"
    assert capfdbinary.readouterr().out == log.read_bytes()


def progress_identity():
    return dict(task="qwen3-v2-teacher-prompt-probe", job_id="12345",
                run_id="wrapper-fixture", code_sha256="a" * 64)


def progress_snapshot():
    return dict(progress_identity(), artifact_kind="teacher_probe_progress", not_a_completion_report=True,
                passed=False, accepted_science=False, full_teacher_ready=False, g0_passed=False,
                training_started=False, stage="import_torch",
                completed_attempt_counts={"baseline": 0, "candidate": 0})


def test_failure_progress_is_bounded_and_never_invents_missing_coverage(tmp_path):
    helpers = wrapper_functions()
    helpers.update(identity=progress_identity())
    missing = helpers["failure_progress"](tmp_path)
    assert not missing["progress_available"]
    assert "completed_attempt_counts" not in missing
    path = tmp_path / "progress.json"
    snapshot = progress_snapshot()
    path.write_text(json.dumps(snapshot))
    found = helpers["failure_progress"](tmp_path)
    assert found["progress_available"]
    assert found["not_a_quality_estimate"]
    assert found["progress"] == snapshot
    for invalid in (dict(snapshot, job_id="12346"), dict(snapshot, passed=True)):
        path.write_text(json.dumps(invalid))
        assert not helpers["failure_progress"](tmp_path)["progress_available"]
    path.write_bytes(b" " * (64 * 1024 + 1))
    assert not helpers["failure_progress"](tmp_path)["progress_available"]


def test_failure_publication_hashes_native_crash_evidence(tmp_path):
    helpers = wrapper_functions()
    output, persistent = tmp_path / "node-output", tmp_path / "persistent"
    output.mkdir()
    persistent.mkdir()
    snapshot = progress_snapshot()
    evidence = {
        "progress.json": json.dumps(snapshot).encode(),
        "progress.jsonl": json.dumps(snapshot).encode() + b"\n",
        "worker-stacks.log": b'Current thread 0x1234:\n  File "worker.py", line 7 in blocked_import\n',
        "attempt-ledger.jsonl": b'{"accepted":false,"response_text":"unchanged raw response"}\n',
    }
    for name, raw in evidence.items():
        (output / name).write_bytes(raw)
    log = tmp_path / "worker.log"
    log.write_bytes(b"READY\nTRACE_AFTER_TERM\n")
    helpers.update(identity=progress_identity(), result_root=persistent)
    report = dict(progress_identity(), passed=False,
                  execution_diagnostics=helpers["failure_progress"](output))
    with redirect_stdout(io.StringIO()):
        helpers["publish"](output, log, report, False)
    receipt = json.loads((persistent / "receipt.json").read_text())
    assert not receipt["passed"]
    assert receipt["persistent_read_back_verified"]
    for name, raw in evidence.items():
        assert (persistent / "artifacts" / name).read_bytes() == raw
    for record in receipt["files"]:
        raw = (persistent / record["path"]).read_bytes()
        assert record["size"] == len(raw)
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
